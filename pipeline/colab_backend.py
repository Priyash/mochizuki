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
import sys
import tarfile
import threading
import time
from pathlib import Path

from .log import log as _structured_log
from .release import Release, ReleaseBackend
from .release import names as _names

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

def pack():
    # What the worker has made so far, ready to download at any moment: if the runner loses this
    # job, it still fetches every finished output. Written aside, then swapped in whole.
    with tarfile.open("/content/out.tar.part", "w") as t:   # outputs are PNG/MP4: compressing gains nothing
        if os.path.isdir("/content/out"):
            t.add("/content/out", arcname="out")
    os.replace("/content/out.tar.part", "/content/out.tar.gz")

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:                       # streaming keeps the CLI's exec alive and shows progress
    print(line.rstrip(), flush=True)
    if " ok in " in line or " failed: " in line:
        pack()
code = p.wait()
print(f"[job] worker exit code {code}", flush=True)
pack()
print("[job] MOCHI-DONE", flush=True)
# If the runner never collects and stops us, give the VM back anyway.
threading.Timer(release_after, release_vm, ["results not collected"]).start()
'''


def settings(p) -> dict:
    cfg = {**DEFAULTS, **(p.config.get("colab", {}) if p is not None else {})}
    cfg["units_per_hour"] = {**DEFAULTS["units_per_hour"], **cfg.get("units_per_hour", {})}
    return cfg


def _log(msg: str, event: str = "note", level: str = "info", **fields) -> None:
    _structured_log("colab", event, msg, level, **fields)


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
        """Units the agreed cap has used, counted at the dearest allowed GPU's rate (conservative).

        Only VM time counts, never time a batch spent waiting for a runner or for units:
          - finished batches count the VM time the runner reported (gpu_seconds);
          - a finished batch with no report counts nothing if it simply waited out max_wait_hours
            (no VM ever opened), and a full batch if its runner died mid-run (the VM may have billed);
          - the open batch counts as a full batch: room is kept for it until it reports."""
        full = float(self.cfg["batch_hours"]) * 3600 + float(self.cfg["setup_minutes"]) * 60
        never_taken = float(self.cfg["max_wait_hours"]) * 3600 * 0.95
        secs = 0.0
        for r in led.db.execute("SELECT submitted_at, finished_at, gpu_seconds FROM batches WHERE backend=? "
                                "AND submitted_at IS NOT NULL", (self.name,)).fetchall():
            if r["finished_at"] is None:
                secs += full
            elif r["gpu_seconds"] is not None:
                secs += float(r["gpu_seconds"])
            elif r["finished_at"] - r["submitted_at"] < never_taken:
                secs += min(full, r["finished_at"] - r["submitted_at"])
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

    _PY = ("import json,sys\n"
           "from colab_cli.auth import AuthProvider, get_credentials\n"
           "from colab_cli.client import Client, Prod\n"
           "c = Client(Prod(), get_credentials(None, provider=AuthProvider.ADC))\n"
           "if sys.argv[1] == 'list':\n"
           "    print(json.dumps([{'endpoint': a.endpoint, 'accelerator': str(a.accelerator.value)}"
           " for a in c.list_assignments()]))\n"
           "else:\n"
           "    c.unassign(sys.argv[2]); print('unassigned')\n")

    def assignments(self) -> list[dict] | None:
        """Every VM assigned to the account on Colab's side (local CLI state does not survive a runner)."""
        try:
            r = subprocess.run([sys.executable, "-c", self._PY, "list"], capture_output=True, text=True, timeout=120)
            return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    def unassign(self, endpoint: str) -> bool:
        try:
            r = subprocess.run([sys.executable, "-c", self._PY, "unassign", endpoint],
                               capture_output=True, text=True, timeout=120)
            return r.returncode == 0
        except Exception:  # noqa: BLE001
            return False


def stop_gpu_vms(cli, keep: set | None = None, why: str = "left behind") -> int | None:
    """Stop every GPU VM on the account except `keep`: a GPU VM nobody is using still bills units.
    The account's Colab GPUs belong to this pipeline (docs/colab.md). How many it stopped, or None
    if the account's VMs could not be listed (so nobody can say whether one is running)."""
    found = cli.assignments()
    if found is None:
        _log("could not list the account's Colab VMs", event="vms.list_failed", level="warn")
        return None
    n = 0
    for a in found:
        acc = str(a.get("accelerator", "")).upper()
        if a["endpoint"] in (keep or set()) or acc in ("", "NONE", "CPU"):
            continue
        ok = cli.unassign(a["endpoint"])
        _log(f"{'stopped' if ok else 'could not stop'} a {acc} VM {why}", event="vm.stopped" if ok else "vm.stop_failed", level="info" if ok else "error", gpu=acc)
        n += ok
    return n


def _wait_cap(cfg: dict) -> float:
    return float(cfg["batch_hours"]) * 3600 + float(cfg["setup_minutes"]) * 60 + 1800


def run_remote(cli: ColabCLI, cfg: dict, session: str, inputs_tgz: Path, batch_id: str, out_tgz: Path,
               packages: list[str], engine_set: str = "gpu", beat=None) -> dict:
    """One VM: open it (first GPU that is granted), run the job, fetch the results, always stop it.

    The result says what really happened, including GPU VMs that came up after their request had
    given up ("late_vms": they are stopped, but they ran and may have billed) and whether the
    account's VMs could be checked at all ("vms_unchecked")."""
    gpu = None
    tried = []
    late, unchecked = 0, False
    before = {a["endpoint"] for a in (cli.assignments() or [])}
    for g in cfg["gpus"]:
        hm = ["--high-mem"] if g in cfg.get("high_mem_gpus", []) else []
        code, out = cli.run(["new", "-s", session, "--gpu", g, *hm], timeout=2400)
        if code == 0:
            gpu = g
            break
        tried.append(_redact(f"{g}: {out.strip().splitlines()[-1] if out.strip() else code}")[:200])
        n = stop_gpu_vms(cli, keep=before, why=f"that came up after the {g} request gave up")
        late, unchecked = late + (n or 0), unchecked or n is None
    if gpu is None:
        error = "no GPU granted in time: " + "; ".join(tried)
        if late:
            error += f" ({late} GPU VM(s) came up late and were stopped)"
        return {"ok": False, "gpu": None, "late_vms": late, "vms_unchecked": unchecked, "error": error}
    t0 = time.time()
    _log(f"session {session} on {gpu}", event="vm.granted", session=session, gpu=gpu)
    try:
        code, out = cli.run(["upload", "-s", session, str(inputs_tgz), "/content/in.tar.gz"], timeout=1800)
        if code != 0:
            return {"ok": False, "gpu": gpu, "late_vms": late, "error": f"upload failed: {out[-300:]}"}
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
            # The job re-packs its outputs after every finished job: fetch what it made so far.
            partial = salvage(cli, session, out_tgz)
            return {"ok": False, "partial": partial, "gpu": gpu, "late_vms": late,
                    "error": f"job did not finish (exit {code}): {tail[-600:]}", "seconds": time.time() - t0}
        if not _download(cli, session, out_tgz):
            return {"ok": False, "gpu": gpu, "late_vms": late, "error": "download failed twice",
                    "seconds": time.time() - t0}
        return {"ok": True, "gpu": gpu, "late_vms": late, "seconds": time.time() - t0}
    finally:
        code, out = cli.run(["stop", "-s", session], timeout=300)
        _log(f"session {session} stopped" if code == 0 else f"session {session} stop said: {out.strip()[-200:]}", event="vm.stopped" if code == 0 else "vm.stop_failed", level="info" if code == 0 else "error", session=session)
        # Anything still up now bills for nothing: a GPU granted late while this one ran, or this
        # one if `stop` failed.
        stop_gpu_vms(cli, keep=before, why="still running after the batch")


def _download(cli, session: str, out_tgz: Path, tries: int = 2, timeout: int = 3600) -> bool:
    for _ in range(tries):
        code, _out = cli.run(["download", "-s", session, "/content/out.tar.gz", str(out_tgz)], timeout=timeout)
        if code == 0 and out_tgz.exists():
            return True
    return False


def salvage(cli, session: str, out_tgz: Path) -> bool:
    """After a run that did not finish, fetch the outputs the job packed so far, before the VM is
    stopped. False if there was nothing to fetch (the job made nothing, or the VM is gone)."""
    ok = _download(cli, session, out_tgz, tries=1, timeout=900)
    _log(f"session {session}: " + ("fetched what the worker had made" if ok else "nothing could be fetched"),
         event="salvage.done" if ok else "salvage.none", level="info" if ok else "warn")
    return ok


def take_one(rel: Release, work: Path, cli: ColabCLI, cfg: dict, packages: list[str]) -> str | None:
    """Claim the oldest waiting Colab batch and run it. The batch id, or None if nothing ran."""
    stop_gpu_vms(cli)            # first, whatever happens: a VM left behind by a run that died still bills
    a = rel.assets()
    waiting = sorted((x for x in a.values() if x["name"].startswith(f"{NAME}-") and x["name"].endswith(".tar.gz")
                      and not x["name"].endswith(".results.tar.gz")), key=lambda x: x.get("created_at", ""))
    waiting = [x for x in waiting if _names(x["name"][len(NAME) + 1:-7], NAME)["claim"] not in a]
    if not waiting:
        _log("no Colab batch is waiting", event="batch.none")
        return None
    bal = cli.balance()
    if bal is not None and bal < float(cfg["min_balance_units"]):
        _log(f"only {bal:.1f} compute units left (minimum {cfg['min_balance_units']}); not starting a VM", event="balance.low", level="warn", balance=bal, minimum=cfg["min_balance_units"])
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
    _log(f"claimed {batch_id} (balance {bal if bal is not None else '?'} units)", event="batch.claimed", batch=batch_id, balance=bal)
    rel.download(asset, d / n["inputs"])
    with tarfile.open(d / n["inputs"]) as t:          # the worker travels with the inputs
        try:
            t.extractall(d / "in", filter="data")
        except TypeError:
            t.extractall(d / "in")
    manifest = json.loads((d / "in" / "manifest.json").read_text(encoding="utf-8"))
    total = len(manifest["jobs"])
    if bal is not None:                            # never let the balance run dry mid-batch: Colab would kill
        rate = max(float(cfg["units_per_hour"].get(g, 6.0)) for g in cfg["gpus"])   # the VM and lose its stills
        setup = float(cfg["setup_minutes"]) * 60
        afford = (bal - 1.0) / rate * 3600 - setup - 600
        if afford < 900:
            _log(f"{bal:.1f} units buy under 15 min of work at the {rate} units/h rate; not starting a VM", event="balance.low", level="warn", balance=bal, rate=rate)
            rel.delete(n["claim"])
            return None
        if afford < float(manifest.get("deadline_seconds", 1e9)):
            _log(f"balance {bal:.1f} units: this batch gets {afford / 60:.0f} min; the rest waits for the next one", event="batch.shrunk", level="warn", batch=batch_id, minutes=round(afford / 60), balance=bal)
            manifest["deadline_seconds"] = afford
            (d / "in" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            cfg = {**cfg, "batch_hours": afford / 3600}
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
            _log(f"heartbeat failed: {e}", event="heartbeat.failed", level="warn")

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
    if r.get("gpu") is None:                       # no GPU in time: the batch waits for the next run
        if r.get("late_vms"):                      # ... but a VM did come up late: it ran, and may have billed
            after = cli.balance()
            used = round(bal - after, 2) if bal is not None and after is not None else None
            _log(f"{batch_id}: {r['error']}; left waiting. The late VM(s) ran until stopped: "
                 f"{used if used is not None else 'an unknown number of'} units used",
                 event="gpu.late", level="warn", batch=batch_id, vms=r["late_vms"], units=used, balance=after)
        elif r.get("vms_unchecked"):
            _log(f"{batch_id}: {r['error']}; left waiting. The account's VMs could not be listed, so a late VM "
                 "can't be ruled out: check the Colab console", event="gpu.unchecked", level="warn", batch=batch_id)
        else:
            _log(f"{batch_id}: {r['error']}; left waiting, no VM came up, nothing billed",
                 event="gpu.none", level="warn", batch=batch_id, billed=0)
        for k in ("claim", "progress"):
            rel.delete(n[k])
        return None
    after = cli.balance()
    usage = {"vm_seconds": round(r.get("seconds", 0) + 120), "gpu": r.get("gpu"),   # + opening the VM
             "units_used": round(bal - after, 2) if bal is not None and after is not None else None}
    _log(f"{batch_id}: {usage['units_used']} units used, balance now {after}", event="units.used", batch=batch_id, units=usage["units_used"], balance=after, gpu=usage["gpu"])
    res_tgz = d / n["results"]
    (d / "out").mkdir(exist_ok=True)
    rj, base = _read_results(out_tgz, d / "res") if (r["ok"] or r.get("partial")) else (None, None)
    if not r["ok"]:                                # jobs it did not finish are freed; they do not count against shots
        _log(f"{batch_id}: {r['error']}" + (f"; kept the {len(rj['jobs'])} job(s) it finished" if rj else ""),
             event="batch.failed", level="error", batch=batch_id, gpu=r.get("gpu"))
        if rj:                                     # the run's log tail stays out of "error", where a word
            rj["runner_error"] = r["error"][:1500]  # like "stale" would make the producer drop every output
            rj["error"] = rj.get("error") or "colab: the run did not finish; the outputs it made were kept"
    if rj is None:
        why = r.get("error") or "the worker wrote no results"
        rj, base = {"batch_id": batch_id, "jobs": {}, "error": f"colab: {why}"[:1500]}, d / "out"
    rj.update(usage)
    (base / "results.json").write_text(json.dumps(rj))
    with tarfile.open(res_tgz, "w:gz") as t:
        t.add(base, arcname="out")
    _retry(lambda: rel.upload(res_tgz), what=f"upload of {res_tgz.name}")
    _log(f"{batch_id}: results uploaded ({'ok' if r['ok'] else 'failed'}, GPU {r.get('gpu')}, "
         f"{r.get('seconds', 0) / 60:.0f} min on the VM)", event="batch.done", batch=batch_id, ok=r["ok"], gpu=r.get("gpu"), vm_min=round(r.get("seconds", 0) / 60))
    return batch_id


def _read_results(out_tgz: Path, dest: Path) -> tuple[dict | None, Path | None]:
    """The worker's results.json and its folder from a downloaded archive; (None, None) if the
    archive is missing, broken, or holds no readable results."""
    try:
        with tarfile.open(out_tgz) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:
                t.extractall(dest)
        found = list(dest.rglob("results.json"))
        if not found:
            return None, None
        return json.loads(found[0].read_text(encoding="utf-8")), found[0].parent
    except (OSError, tarfile.TarError, ValueError) as e:
        _log(f"could not read the results in {out_tgz.name}: {e}", event="results.unreadable", level="warn")
        return None, None


def _retry(fn, what: str, tries: int = 3, wait: float = 10.0):
    """GitHub's API fails now and then; results that cost units are worth a second and third try."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            _log(f"{what} failed ({e}); trying again", event="retry", level="warn")
            time.sleep(wait * (i + 1))


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
    stop_gpu_vms(cli)
    before = cli.balance()
    if before is not None and before < float(cfg["min_balance_units"]):
        return {"batch": batch_id, "ok": False, "gpu": None, "balance_before": before, "units_used": 0.0,
                "error": f"the signed-in Google account has {before} compute units; Colab gives GPUs only to "
                         "accounts with paid units. Buy units on that same account (docs/colab.md)"}
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


# ==================================================================== smoke test (no GPU, no units)
SMOKE_JOB = r'''
import os, sys, time, json, threading, urllib.request, shutil, platform
out = {"python": sys.version.split()[0], "platform": platform.platform(),
       "env_ok": os.environ.get("MOCHI_SMOKE") == "42"}
try:
    out["input"] = open("/content/smoke_in.txt").read().strip()
except Exception as e:
    out["input"] = f"missing: {e}"
try:
    from google.colab import runtime
    out["can_release_vm"] = hasattr(runtime, "unassign")
except Exception as e:
    out["can_release_vm"] = f"no: {e}"
for name, url in (("huggingface", "https://huggingface.co/api/models/Qwen/Qwen-Image-2512"),
                  ("pypi", "https://pypi.org/simple/diffusers/")):
    t = time.time()
    try:
        urllib.request.urlopen(url, timeout=30).read(2000)
        out[f"net_{name}"] = f"ok in {time.time() - t:.1f}s"
    except Exception as e:
        out[f"net_{name}"] = f"failed: {e}"
du = shutil.disk_usage("/content")
out["disk_free_gb"] = round(du.free / 1e9, 1)
try:
    out["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
except Exception:
    pass
ticks = []
threading.Timer(2, lambda: ticks.append(1)).start()
for i in range(4):
    print(f"[smoke] still streaming {i}", flush=True)
    time.sleep(1)
out["timers_work"] = bool(ticks)
open("/content/smoke_out.json", "w").write(json.dumps(out))
print("[smoke] " + json.dumps(out), flush=True)
print("[job] MOCHI-DONE", flush=True)
'''


def _redact(text: str) -> str:
    """Action logs are public: never print the Google account's address."""
    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<account>", text or "")


def smoke(cli: ColabCLI, work: Path) -> dict:
    """The whole Colab round trip on a CPU VM (free, no GPU): sign-in, balance, open, upload,
    exec with --env and streaming, download, stop, and that no VM is left behind."""
    work.mkdir(parents=True, exist_ok=True)
    rep: dict = {"steps": {}}
    st = rep["steps"]

    def step(name, ok, detail=""):
        st[name] = {"ok": bool(ok), "detail": _redact(str(detail))[-400:]}
        _log(f"{'OK  ' if ok else 'FAIL'} {name} {st[name]['detail']}", event="smoke.step", level="info" if ok else "error", step=name, ok=bool(ok))
        return ok

    code, out = cli.run(["--help"], timeout=60)
    step("cli installed", code == 0, out.strip().splitlines()[0] if out.strip() else code)
    code, out = cli.run(["usage"], timeout=120)
    bal = cli.balance()
    rep["balance_units"] = bal
    if not step("sign-in and balance", bal is not None, f"balance {bal} units" if bal is not None else out):
        return rep
    rep["gpu_needs_units"] = bal < 3
    vms = cli.assignments()
    step("list the account's VMs", vms is not None, f"{len(vms)} running: " + ", ".join(
        str(v.get("accelerator")) for v in vms) if vms is not None else "failed")
    rep["stopped_left_behind"] = stop_gpu_vms(cli)
    session = time.strftime("mochi-smoke-%H%M%S")
    t0 = time.time()
    code, out = cli.run(["new", "-s", session], timeout=900)
    if not step("open a CPU VM", code == 0, f"in {time.time() - t0:.0f}s" if code == 0 else out):
        return rep
    try:
        (work / "smoke_in.txt").write_text("hello from GitHub\n")
        (work / "smoke_job.py").write_text(SMOKE_JOB)
        code, out = cli.run(["upload", "-s", session, str(work / "smoke_in.txt"), "/content/smoke_in.txt"], 600)
        step("upload", code == 0, out)
        lines = []
        code, tail = cli.stream(["exec", "-s", session, "-f", str(work / "smoke_job.py"), "--timeout", "0",
                                 "--env", "MOCHI_SMOKE=42"], time.time() + 900, lines.append)
        step("run code (exec, streaming)", "MOCHI-DONE" in tail, f"exit {code}")
        info = next((json.loads(x.split("[smoke] ", 1)[1]) for x in lines if x.startswith("[smoke] {")), {})
        rep["vm"] = info
        step("env passed in", info.get("env_ok"), "")
        step("uploaded file seen on the VM", info.get("input") == "hello from GitHub", info.get("input"))
        step("VM can release itself", info.get("can_release_vm") is True, info.get("can_release_vm"))
        step("VM reaches Hugging Face", str(info.get("net_huggingface", "")).startswith("ok"),
             info.get("net_huggingface"))
        step("VM reaches PyPI", str(info.get("net_pypi", "")).startswith("ok"), info.get("net_pypi"))
        step("timers keep running", info.get("timers_work"), "")
        dl = work / "smoke_out.json"
        code, out = cli.run(["download", "-s", session, "/content/smoke_out.json", str(dl)], 600)
        step("download", code == 0 and dl.exists() and json.loads(dl.read_text()).get("env_ok"), out)
    finally:
        code, out = cli.run(["stop", "-s", session], timeout=300)
        step("stop the VM", code == 0, out)
    left = cli.sessions()
    step("no VM left behind", session not in left, ", ".join(left))
    rep["ok"] = all(v["ok"] for v in st.values())
    return rep


def gpu_probe(cli: ColabCLI, gpus=None, wait_s: int = 900) -> dict:
    """Which GPUs Colab gives this account right now, and the real units per hour of each.
    Each VM is stopped as soon as it is granted."""
    # MOCHI_PROBE="A100:hm,G4:hm,L4" asks for each GPU in turn; ":hm" adds --high-mem
    spec = gpus or os.environ.get("MOCHI_PROBE") or "T4,L4,G4,A100,H100"
    gpus = spec.split(",") if isinstance(spec, str) else list(spec)
    rep = {"balance_before": cli.balance(), "gpus": {}}
    stop_gpu_vms(cli)
    for item in gpus:
        g, _, opt = item.strip().partition(":")
        s = f"mochi-probe-{g.lower()}"
        t0 = time.time()
        g_label = g + (":hm" if opt == "hm" else "")
        code, out = cli.run(["new", "-s", s, "--gpu", g, *(["--high-mem"] if opt == "hm" else [])],
                            timeout=wait_s + 60)
        took = round(time.time() - t0)
        if code == 0:
            _, use = cli.run(["usage"], timeout=120)
            m = re.search(r"([\d.]+)\s*(?:compute units\s*)?(?:/|per)\s*h", use)
            cli.run(["stop", "-s", s], timeout=300)
            rep["gpus"][g_label] = {"granted": True, "seconds": took, "units_per_hour": float(m.group(1)) if m else None,
                              "usage": _redact(use.strip())[-300:]}
        else:
            last = re.sub(r"\x1b\[[0-9;]*m", "", out.strip().splitlines()[-1] if out.strip() else str(code))
            rep["gpus"][g_label] = {"granted": False, "seconds": took, "error": _redact(last)[-250:]}
            stop_gpu_vms(cli, why=f"that came up after the {g} request gave up")
        r_ = rep["gpus"][g_label]
        _log(f"{g_label}: {r_}", event="probe.gpu", gpu=g_label, granted=r_["granted"],
             units_per_hour=r_.get("units_per_hour"), wait_s=r_["seconds"])
    stop_gpu_vms(cli)
    rep["balance_after"] = cli.balance()
    return rep
