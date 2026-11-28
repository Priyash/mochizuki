"""Checks run by hand before real work: one reference still outside the ledger (test_still), the
whole round trip on a free CPU VM (smoke), and which GPUs the account gets now (gpu_probe)."""
from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import time
from typing import Any
from pathlib import Path

from ..base import WORKER
from .cli import ColabCLI, _redact
from .config import NAME, SESSION_PREFIX
from .config import log as _log
from .runner import run_remote, stop_gpu_vms

def test_still(p, cli: ColabCLI, cfg: dict, packages: list[str], out_dir: Path, ref: str = "style") -> dict:
    """One reference still on Colab, outside the ledger and the release: proves sign-in, GPU, engine."""
    batch_id = time.strftime("colabtest-%Y%m%d-%H%M%S")
    d = out_dir / batch_id
    (d / "in").mkdir(parents=True, exist_ok=True)
    style = p.style_file
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
    res = run_remote(cli, small, f"{SESSION_PREFIX}test-{batch_id[-6:]}", full, batch_id, d / "out.tar.gz", packages)
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
    assert bal is not None                       # step() returned early otherwise
    rep["gpu_needs_units"] = bal < 3
    vms = cli.assignments()
    step("list the account's VMs", vms is not None, f"{len(vms)} running: " + ", ".join(
        str(v.get("accelerator")) for v in vms) if vms is not None else "failed")
    rep["stopped_left_behind"] = stop_gpu_vms(cli)
    session = SESSION_PREFIX + time.strftime("smoke-%H%M%S")
    t0 = time.time()
    code, out = cli.run(["new", "-s", session], timeout=900)
    if not step("open a CPU VM", code == 0, f"in {time.time() - t0:.0f}s" if code == 0 else out):
        return rep
    try:
        (work / "smoke_in.txt").write_text("hello from GitHub\n")
        (work / "smoke_job.py").write_text(SMOKE_JOB)
        code, out = cli.run(["upload", "-s", session, str(work / "smoke_in.txt"), "/content/smoke_in.txt"], 600)
        step("upload", code == 0, out)
        lines: list[str] = []
        code, tail = cli.stream(["exec", "-s", session, "-f", str(work / "smoke_job.py"), "--timeout", "0",
                                 "--env", "MOCHI_SMOKE=42"], time.time() + 900, lines.append)
        step("run code (exec, streaming)", "MOCHI-DONE" in tail, f"exit {code}")
        info: dict = next((json.loads(x.split("[smoke] ", 1)[1]) for x in lines if x.startswith("[smoke] {")), {})
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
    rep: dict[str, Any] = {"balance_before": cli.balance(), "gpus": {}}
    stop_gpu_vms(cli)
    for item in gpus:
        g, _, opt = item.strip().partition(":")
        s = f"{SESSION_PREFIX}probe-{g.lower()}"
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
