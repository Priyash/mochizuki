"""Google Colab (pay-as-you-go compute units) as the bf16 GPU backend for stills and keyframes.

Two halves, talking through the `batches` GitHub release (as the Mac backend does):

  producer side (GitHub Actions)   ColabBackend: packs a batch as `colab-<id>.tar.gz`, checks the
                                   agreed unit cap, and starts the `colab-run` workflow.
  runner (GitHub Actions)          `python -m pipeline colab-runner`: checks the real compute-unit
                                   balance, claims the batch, opens a Colab VM with the Colab CLI
                                   (A100 first, then L4), runs the same worker as on Kaggle there,
                                   downloads the results, stops the VM, uploads
                                   `colab-<id>.results.tar.gz`. A heartbeat goes up every few minutes.

Money guards (Colab bills compute units while a VM is assigned):
  - colab.total_cap_units: the producer sizes every batch to what is left of the agreed cap,
    counting each batch's wall-clock time at the dearest allowed GPU's rate;
  - colab.min_balance_units: the runner does not start a VM below this real balance;
  - the VM is stopped in a `finally`, every run first stops any mochizuki VM left behind, and the
    job on the VM releases its own VM at its deadline even if nobody is listening any more.

Sign-in: the Colab CLI's Application Default Credentials (`--auth adc`), from the GitHub secret
COLAB_ADC_JSON (docs/colab.md). Nothing here buys units; the owner does.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from .mac_backend import Release, ReleaseBackend, _log, _names

NAME = "colab"
DEFAULTS = {
    "enabled": False,
    "stages": ["refs", "keyframes"],
    "gpus": ["A100", "L4"],
    "units_per_hour": {"T4": 1.19, "L4": 1.71, "A100": 5.40},
    "total_cap_units": 100,
    "min_balance_units": 3,
    "batch_hours": 4,
    "setup_minutes": 15,
    "max_wait_hours": 12,
    "stale_minutes": 30,
    "heartbeat_minutes": 5,
    "release_after_minutes": 20,
}

JOB = r'''
# mochizuki job on a Colab VM. Written by pipeline/colab_backend.py; runs in the VM's kernel.
import json, os, subprocess, sys, tarfile, threading, time
batch, deadline = os.environ["MOCHI_BATCH"], float(os.environ["MOCHI_DEADLINE"])
release_after = float(os.environ.get("MOCHI_RELEASE_AFTER", "1200"))

def release_vm(why):
    print(f"[job] releasing this VM: {why}", flush=True)
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as e:
        print(f"[job] could not release the VM ({e})", flush=True)

# A last guard if the runner is gone: release the VM after the deadline, whatever happens.
threading.Timer(max(60.0, deadline + release_after - time.time()), release_vm, ["deadline"]).start()
os.makedirs("/content/in", exist_ok=True)
with tarfile.open("/content/in.tar.gz") as t:
    t.extractall("/content/in")
os.environ.setdefault("HF_HOME", "/content/hf")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
pkgs = json.loads(os.environ.get("MOCHI_PACKAGES", "[]"))
print(f"[job] installing {len(pkgs)} packages", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
cmd = [sys.executable, "/content/in/worker.py", "--manifest", "/content/in/manifest.json", "--inputs", "/content/in",
       "--out", "/content/out", "--engine-set", os.environ.get("MOCHI_ENGINE_SET", "gpu"), "--expect-batch", batch]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:                       # streaming keeps the CLI's exec alive and shows progress
    print(line.rstrip(), flush=True)
code = p.wait()
print(f"[job] worker exit code {code}", flush=True)
with tarfile.open("/content/out.tar.gz", "w:gz") as t:
    t.add("/content/out", arcname="out")
print("[job] MOCHI-DONE", flush=True)
# If the runner never collects and stops us, give the VM back anyway.
threading.Timer(release_after, release_vm, ["results not collected"]).start()
'''


def settings(p) -> dict:
    cfg = {**DEFAULTS, **(p.config.get("colab", {}) if p is not None else {})}
    cfg["units_per_hour"] = {**DEFAULTS["units_per_hour"], **cfg.get("units_per_hour", {})}
    return cfg


# ==================================================================== producer side
class ColabBackend(ReleaseBackend):
    name = NAME

    def __init__(self, p, repo: str, gh=None, fetch=None, now=time.time):
        super().__init__(p, repo, gh, fetch, now, cfg=settings(p))

    def rate(self) -> float:
        """Units per hour of the dearest GPU we may get: the conservative rate for the cap."""
        r = self.cfg["units_per_hour"]
        return max(float(r.get(g, 6.0)) for g in self.cfg["gpus"])

    def spent_units(self, led) -> float:
        rows = led.db.execute("SELECT submitted_at, finished_at FROM batches WHERE backend=? "
                              "AND submitted_at IS NOT NULL", (self.name,)).fetchall()
        secs = sum(max(0.0, (r["finished_at"] or self.now()) - r["submitted_at"]) for r in rows)
        return secs / 3600 * self.rate()

    def budget(self, p, led) -> tuple[float, float]:
        from .kaggle import BudgetUsedUp  # noqa: PLC0415
        cap, spent, rate = float(self.cfg["total_cap_units"]), self.spent_units(led), self.rate()
        setup = float(self.cfg["setup_minutes"]) * 60
        secs = min(float(self.cfg["batch_hours"]) * 3600, (cap - spent) / rate * 3600 - setup - 600)
        if secs <= 900:
            raise BudgetUsedUp(f"the agreed Colab cap is used up ({spent:.0f} of {cap:.0f} compute units by the "
                               "producer's count); raise colab.total_cap_units in config.json to go on")
        self.log.append(f"Colab batch budget {secs / 3600:.1f} h; about {spent:.0f} of {cap:.0f} units used")
        return secs, setup

    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str:
        code, out = self.rel._gh(["api", "-X", "POST", f"repos/{self.rel.repo}/actions/workflows/colab-run.yml/dispatches",
                                  "-f", "ref=main", "-f", "inputs[mode]=run"])
        self.log.append("colab-run started" if code == 0 else f"colab-run not started ({out.strip()[-200:]}); "
                        "its schedule will pick the batch up")
        return Path(folder).parent.name


def from_env(p):
    """The Colab backend when config.json switches it on."""
    if not settings(p).get("enabled"):
        return None
    return ColabBackend(p, os.environ.get("GITHUB_REPOSITORY", ""))


# ==================================================================== runner side
class ColabCLI:
    """The `colab` command (google-colab-cli), signed in with Application Default Credentials."""

    def __init__(self, exe: str = "colab", auth: str | None = None):
        self.exe = exe
        self.auth = auth if auth is not None else os.environ.get("COLAB_AUTH", "adc")

    def _cmd(self, args: list[str]) -> list[str]:
        return [self.exe, *(["--auth", self.auth] if self.auth else []), *args]

    def run(self, args: list[str], timeout: float | None = 600) -> tuple[int, str]:
        try:
            r = subprocess.run(self._cmd(args), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return 124, f"timed out after {timeout} s: {(e.stdout or '')[-500:]}"
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def stream(self, args: list[str], deadline: float, on_line=None) -> tuple[int, str]:
        """Run a long command, echoing its output; kill it at `deadline` (unix time)."""
        p = subprocess.Popen(self._cmd(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail: list[str] = []
        killer = threading.Timer(max(1.0, deadline - time.time()), p.kill)
        killer.start()
        try:
            for line in p.stdout:
                line = line.rstrip()
                print(line, flush=True)
                tail = (tail + [line])[-60:]
                if on_line:
                    on_line(line)
            code = p.wait()
        finally:
            killer.cancel()
        return code, "\n".join(tail)

    def balance(self) -> float | None:
        code, out = self.run(["usage"], timeout=120)
        m = re.search(r"Current balance:\s*([\d.]+)", out)
        return float(m.group(1)) if code == 0 and m else None

    def sessions(self) -> list[str]:
        code, out = self.run(["sessions"], timeout=120)
        return re.findall(r"\b(mochi-[\w-]+)", out) if code == 0 else []


def _wait_cap(cfg: dict) -> float:
    return float(cfg["batch_hours"]) * 3600 + float(cfg["setup_minutes"]) * 60 + 1800


def run_remote(cli: ColabCLI, cfg: dict, session: str, inputs_tgz: Path, batch_id: str, out_tgz: Path,
               packages: list[str], engine_set: str = "gpu", beat=None) -> dict:
    """One VM: open it (first GPU that is granted), run the job, fetch the results, always stop it."""
    gpu = None
    tried = []
    for g in cfg["gpus"]:
        code, out = cli.run(["new", "-s", session, "--gpu", g], timeout=900)
        if code == 0:
            gpu = g
            break
        tried.append(f"{g}: {out.strip().splitlines()[-1] if out.strip() else code}"[:200])
    if gpu is None:
        return {"ok": False, "gpu": None, "error": "no GPU granted: " + "; ".join(tried)}
    t0 = time.time()
    _log(f"session {session} on {gpu}")
    try:
        code, out = cli.run(["upload", "-s", session, str(inputs_tgz), "/content/in.tar.gz"], timeout=1800)
        if code != 0:
            return {"ok": False, "gpu": gpu, "error": f"upload failed: {out[-300:]}"}
        job = inputs_tgz.parent / "colab_job.py"
        job.write_text(JOB)
        deadline = time.time() + _wait_cap(cfg)
        env = [f"MOCHI_BATCH={batch_id}", f"MOCHI_DEADLINE={deadline:.0f}", f"MOCHI_ENGINE_SET={engine_set}",
               f"MOCHI_PACKAGES={json.dumps(packages)}",
               f"MOCHI_RELEASE_AFTER={float(cfg['release_after_minutes']) * 60:.0f}"]
        args = ["exec", "-s", session, "-f", str(job), "--timeout", "0"]
        for e in env:
            args += ["--env", e]
        done = {"n": 0}

        def on_line(line):
            if " ok in " in line or " failed: " in line:
                done["n"] += 1
                if beat:
                    beat(done["n"])

        code, tail = cli.stream(args, deadline, on_line)
        if "MOCHI-DONE" not in tail:
            return {"ok": False, "gpu": gpu, "error": f"job did not finish (exit {code}): {tail[-600:]}",
                    "seconds": time.time() - t0}
        code, out = cli.run(["download", "-s", session, "/content/out.tar.gz", str(out_tgz)], timeout=3600)
        if code != 0 or not out_tgz.exists():
            return {"ok": False, "gpu": gpu, "error": f"download failed: {out[-300:]}", "seconds": time.time() - t0}
        return {"ok": True, "gpu": gpu, "seconds": time.time() - t0}
    finally:
        code, out = cli.run(["stop", "-s", session], timeout=300)
        _log(f"session {session} stopped" if code == 0 else f"session {session} stop said: {out.strip()[-200:]}")


def take_one(rel: Release, work: Path, cli: ColabCLI, cfg: dict, packages: list[str]) -> str | None:
    """Claim the oldest waiting Colab batch and run it. The batch id, or None if nothing ran."""
    a = rel.assets()
    waiting = sorted((x for x in a.values() if x["name"].startswith(f"{NAME}-") and x["name"].endswith(".tar.gz")
                      and not x["name"].endswith(".results.tar.gz")), key=lambda x: x.get("created_at", ""))
    waiting = [x for x in waiting if _names(x["name"][len(NAME) + 1:-7], NAME)["claim"] not in a]
    if not waiting:
        _log("no Colab batch is waiting")
        return None
    for s in cli.sessions():                      # a VM left behind by a run that died still bills: stop it
        cli.run(["stop", "-s", s], timeout=300)
        _log(f"stopped leftover session {s}")
    bal = cli.balance()
    if bal is not None and bal < float(cfg["min_balance_units"]):
        _log(f"only {bal:.1f} compute units left (minimum {cfg['min_balance_units']}); not starting a VM")
        return None
    asset = waiting[0]
    batch_id = asset["name"][len(NAME) + 1:-7]
    n = _names(batch_id, NAME)
    d = work / batch_id
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    claim = d / n["claim"]
    claim.write_text(json.dumps({"host": socket.gethostname(), "at": time.time(), "balance": bal}))
    rel.upload(claim)
    _log(f"claimed {batch_id} (balance {bal if bal is not None else '?'} units)")
    rel.download(asset, d / n["inputs"])
    with tarfile.open(d / n["inputs"]) as t:          # the worker travels with the inputs
        try:
            t.extractall(d / "in", filter="data")
        except TypeError:
            t.extractall(d / "in")
    manifest = json.loads((d / "in" / "manifest.json").read_text(encoding="utf-8"))
    total = len(manifest["jobs"])
    from .kaggle import WORKER  # noqa: PLC0415
    shutil.copyfile(WORKER, d / "in" / "worker.py")
    full = d / "full.tar.gz"
    with tarfile.open(full, "w:gz") as t:
        for f in sorted((d / "in").iterdir()):
            t.add(f, arcname=f.name)

    def send(done):
        f = d / n["progress"]
        f.write_text(json.dumps({"done": done, "total": total, "at": time.time()}))
        try:
            rel.upload(f)
        except Exception as e:  # noqa: BLE001 - a missed heartbeat is not fatal
            _log(f"heartbeat failed: {e}")

    stop = threading.Event()
    done = {"n": 0}

    def beat_loop():
        while not stop.wait(float(cfg["heartbeat_minutes"]) * 60):
            send(done["n"])

    def beat(k):
        done["n"] = k
        send(k)

    send(0)
    threading.Thread(target=beat_loop, daemon=True).start()
    out_tgz = d / "out.tar.gz"
    try:
        r = run_remote(cli, cfg, f"mochi-{batch_id[-11:]}", full, batch_id, out_tgz, packages, beat=beat)
    finally:
        stop.set()
    res_tgz = d / n["results"]
    if r["ok"]:
        shutil.copyfile(out_tgz, res_tgz)
    else:                                          # an error result frees the jobs; they do not count against shots
        _log(f"{batch_id}: {r['error']}")
        (d / "out").mkdir(exist_ok=True)
        (d / "out" / "results.json").write_text(json.dumps(
            {"batch_id": batch_id, "jobs": {}, "error": f"colab: {r['error']}"[:1500]}))
        with tarfile.open(res_tgz, "w:gz") as t:
            t.add(d / "out", arcname="out")
    rel.upload(res_tgz)
    _log(f"{batch_id}: results uploaded ({'ok' if r['ok'] else 'failed'}, GPU {r.get('gpu')}, "
         f"{r.get('seconds', 0) / 60:.0f} min on the VM)")
    return batch_id


def test_still(p, cli: ColabCLI, cfg: dict, packages: list[str], out_dir: Path, ref: str = "style") -> dict:
    """One reference still on Colab, outside the ledger and the release: proves sign-in, GPU, engine."""
    from .kaggle import WORKER  # noqa: PLC0415
    batch_id = time.strftime("colabtest-%Y%m%d-%H%M%S")
    d = out_dir / batch_id
    (d / "in").mkdir(parents=True, exist_ok=True)
    style = p.root / "style_bible" / "style.txt"
    r = next(x for x in p.ref_plan() if x["name"] == ref)
    prompt = "\n".join(x for x in ((style.read_text(encoding="utf-8").strip() if style.exists() else ""),
                                   r["prompt"]) if x)
    job = {"key": f"test-{ref}", "shot_id": f"test-{ref}", "kind": "ref", "idx": 0, "prompt": prompt,
           "width": r.get("width", 1664), "height": r.get("height", 928), "est_seconds": 600}
    manifest = {"batch_id": batch_id, "stage": "refs", "chapter": 0, "backend": NAME, "deadline_seconds": 2400,
                "engines": p.config["engines"], "jobs": [job]}
    (d / "in" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(WORKER, d / "in" / "worker.py")
    full = d / "full.tar.gz"
    with tarfile.open(full, "w:gz") as t:
        for f in sorted((d / "in").iterdir()):
            t.add(f, arcname=f.name)
    before = cli.balance()
    small = {**cfg, "batch_hours": 0.75, "setup_minutes": 20}
    res = run_remote(cli, small, f"mochi-test-{batch_id[-6:]}", full, batch_id, d / "out.tar.gz", packages)
    after = cli.balance()
    report = {"batch": batch_id, **res, "balance_before": before, "balance_after": after,
              "units_used": round(before - after, 2) if before is not None and after is not None else None}
    if res.get("ok"):
        with tarfile.open(d / "out.tar.gz") as t:
            try:
                t.extractall(d, filter="data")
            except TypeError:
                t.extractall(d)
        rf = d / "out" / "results.json"
        if not rf.exists():
            report.update(ok=False, error="the worker wrote no results; see the log above")
            return report
        rj = json.loads(rf.read_text())
        report["job"] = rj.get("jobs", {}).get(job["key"])
        report["still"] = str(d / "out" / rj["jobs"][job["key"]]["file"]) if report["job"] and \
            report["job"].get("ok") else None
    return report
