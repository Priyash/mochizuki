"""Runs inside one rented RunPod machine: fetch one batch, run the worker, serve the results,
then delete this machine. Standard library only; the pipeline starts it with:

    curl -fsSL "$MOCHI_AGENT_URL" -o /work/pod_agent.py && python3 /work/pod_agent.py

Environment (set by pipeline/runpod_backend.py when it creates the machine):
    MOCHI_BATCH       batch id; the worker refuses inputs of any other batch
    MOCHI_BUNDLE_URL  public URL of the batch's inputs (.tar.gz: manifest.json, worker.py, inputs)
    MOCHI_TOKEN       per-batch secret; every HTTP request must send it as X-Token
    MOCHI_DEADLINE    unix time of the hard stop: the worker is killed after it
    MOCHI_KEEP_S      how long finished results wait to be collected before the machine deletes itself
    MOCHI_ENGINE_SET  gpu | stub
    MOCHI_PACKAGES    JSON list of pip packages for the GPU engines
    MOCHI_PORT        HTTP port (default 8000), reached through RunPod's proxy

Money safety: the machine deletes itself (pod-scoped key RunPod injects) when results are
collected, when they wait longer than MOCHI_KEEP_S, or past the deadline. The pipeline also
deletes it from outside, and removes any mochizuki-* machine it no longer tracks.
"""
from __future__ import annotations

import hmac
import http.server
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("MOCHI_ROOT", "/work"))
INP, OUT = ROOT / "in", ROOT / "out"
BATCH = os.environ.get("MOCHI_BATCH", "")
TOKEN = os.environ.get("MOCHI_TOKEN", "")
DEADLINE = float(os.environ.get("MOCHI_DEADLINE", time.time() + 4 * 3600))
KEEP = float(os.environ.get("MOCHI_KEEP_S", 9000))
ENGINE_SET = os.environ.get("MOCHI_ENGINE_SET", "gpu")
PORT = int(os.environ.get("MOCHI_PORT", 8000))

state = {"state": "setup", "batch": BATCH, "started": time.time(), "finished": None,
         "exit_code": None, "log": []}
lock = threading.Lock()
proc: subprocess.Popen | None = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with lock:
        state["log"] = (state["log"] + [line[:300]])[-40:]


def finish(st: str, code=None) -> None:
    with lock:
        if state["state"] in ("done", "error"):
            return
        state.update(state=st, exit_code=code, finished=time.time())
    log(f"batch {st} (exit code {code})")


def terminate() -> None:
    """Delete this machine so it stops costing money. Tries runpodctl, then the REST API."""
    pid = os.environ.get("RUNPOD_POD_ID", "")
    log(f"deleting machine {pid}")
    if not pid:
        os._exit(0)
    for cmd in (["runpodctl", "remove", "pod", pid], ["runpodctl", "pod", "delete", pid]):
        try:
            if subprocess.run(cmd, timeout=90, capture_output=True).returncode == 0:
                return
        except Exception:  # noqa: BLE001
            pass
    key = os.environ.get("RUNPOD_API_KEY", "")
    for url in (f"https://api.runpod.io/v2/pods/{pid}", f"https://rest.runpod.io/v1/pods/{pid}"):
        try:
            req = urllib.request.Request(url, method="DELETE", headers={"Authorization": f"Bearer {key}"})
            urllib.request.urlopen(req, timeout=60)
            return
        except Exception as e:  # noqa: BLE001
            log(f"delete via {url.split('/')[2]} failed: {e}")
    os._exit(0)          # last resort: stop the container; the pipeline deletes the machine from outside


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _ok(self) -> bool:
        if TOKEN and hmac.compare_digest(self.headers.get("X-Token", ""), TOKEN):
            return True
        self.send_error(403)
        return False

    def _json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if not self._ok():
            return
        if self.path == "/status":
            with lock:
                return self._json(dict(state, now=time.time()))
        if self.path.startswith("/file/"):
            name = self.path[len("/file/"):]
            f = (OUT / name).resolve()
            if OUT.resolve() not in f.parents or not f.is_file():
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(f.stat().st_size))
            self.end_headers()
            with open(f, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    self.wfile.write(chunk)
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        if not self._ok():
            return
        if self.path == "/done":                   # the pipeline has the results: stop paying now
            self._json({"ok": True})
            threading.Thread(target=lambda: (time.sleep(2), terminate()), daemon=True).start()
            return
        self.send_error(404)


def watchdog() -> None:
    while True:
        time.sleep(20)
        now = time.time()
        if now > DEADLINE and state["state"] in ("setup", "running"):
            log("deadline reached: stopping the worker")
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
            finish("error", "deadline")
        fin = state["finished"]
        if (fin and now > fin + KEEP) or now > DEADLINE + KEEP:
            log("results were not collected in time")
            terminate()
            return


def fetch_bundle() -> None:
    url = os.environ["MOCHI_BUNDLE_URL"]
    tgz = ROOT / "bundle.tar.gz"
    for i in range(6):
        try:
            urllib.request.urlretrieve(url, tgz)
            break
        except Exception as e:  # noqa: BLE001
            log(f"bundle download failed ({e}); retrying")
            time.sleep(10 * (i + 1))
    else:
        raise RuntimeError("could not download the batch inputs")
    INP.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz) as t:
        try:
            t.extractall(INP, filter="data")
        except TypeError:                          # Python older than 3.12
            t.extractall(INP)


def main() -> int:
    global proc
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if time.time() > DEADLINE:                     # a restarted container after the deadline: just go
        terminate()
        return 0
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()
    log(f"agent for {BATCH} listening on {PORT}")
    try:
        fetch_bundle()
        pkgs = ["imageio-ffmpeg"] + (json.loads(os.environ.get("MOCHI_PACKAGES", "[]")) if ENGINE_SET == "gpu" else [])
        if not os.environ.get("MOCHI_SKIP_PIP"):
            log(f"installing {len(pkgs)} packages")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", *pkgs], check=False)
        os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with lock:
            state["state"] = "running"
        logf = open(ROOT / "worker.log", "w")  # noqa: SIM115
        proc = subprocess.Popen([sys.executable, str(INP / "worker.py"), "--manifest", str(INP / "manifest.json"),
                                 "--inputs", str(INP), "--out", str(OUT), "--engine-set", ENGINE_SET,
                                 "--expect-batch", BATCH], stdout=logf, stderr=subprocess.STDOUT)
        code = proc.wait()
        tail = (ROOT / "worker.log").read_text(errors="replace")[-1500:]
        log("worker log tail: " + tail.replace("\n", " | ")[-280:])
        finish("done" if code in (0, 3) else "error", code)
    except Exception as e:  # noqa: BLE001
        log(f"agent failed: {e!r}")
        finish("error", repr(e)[:200])
    while True:                                     # keep serving until collected, or the watchdog deletes us
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
