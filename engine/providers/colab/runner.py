"""Runner side (GitHub Actions): `python -m engine colab-runner` takes the waiting Colab batch, opens a
VM (the first GPU granted, in config order), runs the worker there, fetches what it made, always
stops the VM, and uploads the results to the release."""
from __future__ import annotations

import json
import shutil
import socket
import tarfile
import threading
import time
from pathlib import Path

from ..base import WORKER
from ..release import Release
from ..release import names as _names
from .cli import ColabCLI, _redact
from .config import NAME, SESSION_PREFIX
from .config import log as _log

JOB = r'''
# mochizuki job on a Colab VM. Written by engine/providers/colab/runner.py; runs in the VM's kernel.
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
        _log(
            f"{'stopped' if ok else 'could not stop'} a {acc} VM {why}",
            event="vm.stopped" if ok else "vm.stop_failed",
            level="info" if ok else "error",
            gpu=acc,
        )
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
        _log(
            f"session {session} stopped" if code == 0 else f"session {session} stop said: {out.strip()[-200:]}",
            event="vm.stopped" if code == 0 else "vm.stop_failed",
            level="info" if code == 0 else "error",
            session=session,
        )
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
        _log(
            f"only {bal:.1f} compute units left (minimum {cfg['min_balance_units']}); not starting a VM",
            event="balance.low",
            level="warn",
            balance=bal,
            minimum=cfg["min_balance_units"],
        )
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
    _log(
        f"claimed {batch_id} (balance {bal if bal is not None else '?'} units)",
        event="batch.claimed",
        batch=batch_id,
        balance=bal,
    )
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
            _log(
                f"{bal:.1f} units buy under 15 min of work at the {rate} units/h rate; not starting a VM",
                event="balance.low",
                level="warn",
                balance=bal,
                rate=rate,
            )
            rel.delete(n["claim"])
            return None
        if afford < float(manifest.get("deadline_seconds", 1e9)):
            _log(
                f"balance {bal:.1f} units: this batch gets {afford / 60:.0f} min; the rest waits for the next one",
                event="batch.shrunk",
                level="warn",
                batch=batch_id,
                minutes=round(afford / 60),
                balance=bal,
            )
            manifest["deadline_seconds"] = afford
            (d / "in" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            cfg = {**cfg, "batch_hours": afford / 3600}
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
        r = run_remote(cli, cfg, f"{SESSION_PREFIX}{batch_id[-11:]}", full, batch_id, out_tgz, packages, beat=beat)
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
    _log(
        f"{batch_id}: {usage['units_used']} units used, balance now {after}",
        event="units.used",
        batch=batch_id,
        units=usage["units_used"],
        balance=after,
        gpu=usage["gpu"],
    )
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
    assert base is not None                       # set with the results, or just above
    rj.update(usage)
    (base / "results.json").write_text(json.dumps(rj))
    with tarfile.open(res_tgz, "w:gz") as t:
        t.add(base, arcname="out")
    _retry(lambda: rel.upload(res_tgz), what=f"upload of {res_tgz.name}")
    _log(
        f"{batch_id}: results uploaded ({'ok' if r['ok'] else 'failed'}, GPU {r.get('gpu')}, "
        f"{r.get('seconds', 0) / 60:.0f} min on the VM)",
        event="batch.done",
        batch=batch_id,
        ok=r["ok"],
        gpu=r.get("gpu"),
        vm_min=round(r.get("seconds", 0) / 60),
    )
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
