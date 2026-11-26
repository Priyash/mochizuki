"""Colab backend: the producer queues a batch, the runner runs it on a (fake) Colab VM, the
producer collects it; money guards and failures free the jobs."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from pipeline import colab_backend as cb
from pipeline import kaggle as kg
from pipeline import mac_backend as mb
from pipeline import steps

from test_mac import FakeRelease
from test_pipeline import copy_project


class FakeColab:
    """Plays the `colab` CLI: one VM is a folder; exec runs the job's worker with stub engines."""

    def __init__(self, root: Path, grant=("A100",), balance=50.0, leftover=()):
        self.root, self.grant, self.bal = root, set(grant), balance
        self.vms: dict[str, Path] = {}
        self.left = list(leftover)
        self.calls: list[list[str]] = []

    def run(self, args, timeout=None):
        self.calls.append(args)
        cmd = args[0]
        if cmd == "usage":
            return 0, f"Current balance: {self.bal:.2f} compute units\nUsage rate: 0.00/hr\n"
        if cmd == "sessions":
            return 0, "\n".join(self.left)
        s = args[args.index("-s") + 1]
        if cmd == "new":
            gpu = args[args.index("--gpu") + 1]
            if gpu not in self.grant:
                return 1, "[colab] Backend rejected accelerator"
            self.vms[s] = self.root / s
            self.vms[s].mkdir(parents=True)
            return 0, "connected"
        if cmd == "upload":
            shutil.copyfile(args[3], self.vms[s] / Path(args[4]).name)
            return 0, ""
        if cmd == "download":
            shutil.copyfile(self.vms[s] / Path(args[3]).name, args[4])
            return 0, ""
        if cmd == "stop":
            if s in self.left:
                self.left.remove(s)
            self.vms.pop(s, None)
            return 0, ""
        return 1, f"unknown {args}"

    def stream(self, args, deadline, on_line=None):
        self.calls.append(args)
        s = args[args.index("-s") + 1]
        vm = self.vms[s]
        env = dict(a.split("=", 1) for a in args[args.index("--env"):][1::2])
        with tarfile.open(vm / "in.tar.gz") as t:
            t.extractall(vm / "in", filter="data")
        r = subprocess.run([sys.executable, str(vm / "in" / "worker.py"), "--manifest", str(vm / "in" / "manifest.json"),
                            "--inputs", str(vm / "in"), "--out", str(vm / "out"), "--engine-set", "stub",
                            "--expect-batch", env["MOCHI_BATCH"]], capture_output=True, text=True)
        lines = (r.stdout + r.stderr).splitlines() + ["[job] MOCHI-DONE"]
        for ln in lines:
            if on_line:
                on_line(ln)
        with tarfile.open(vm / "out.tar.gz", "w:gz") as t:
            t.add(vm / "out", arcname="out")
        return 0, "\n".join(lines[-60:])

    def balance(self):
        return self.bal

    def sessions(self):
        return list(self.left)


def _setup(tmp_path, **cfg):
    d = tmp_path / "p"
    d.mkdir()
    proj = copy_project(d, cfg_edit=lambda c: c.update(colab={"enabled": True, **cfg},
                                                       kaggle={**c["kaggle"], "stages": ["video"]}))
    for f in (proj.root / "style_bible" / "refs").glob("*.png"):
        f.unlink()                                       # chapter 1 needs its reference stills
    fr = FakeRelease(tmp_path / "release")
    return proj, steps.open_ledger(proj), fr, cb.ColabBackend(proj, "o/r", gh=fr.gh)


def test_batch_runs_on_colab_and_comes_back(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    r = kg.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "built_and_submitted" and r["stage"] == "refs", r
    bid = led.active_batch("colab")["id"]
    assert fr.names() == [f"colab-{bid}.tar.gz"]
    fake = FakeColab(tmp_path / "vms", grant=("L4",), leftover=["mochi-old"])
    assert cb.take_one(mb.Release("o/r", gh=fr.gh), tmp_path / "w", fake, cb.settings(proj), []) == bid
    assert fake.vms == {} and fake.left == []           # its VM and a leftover one are both stopped
    assert ["new", "-s", f"mochi-{bid[-11:]}", "--gpu", "A100"] in fake.calls   # A100 asked first, L4 granted
    r = kg.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "collected" and r["made"] > 0 and r["failed"] == 0, r
    assert fr.names() == []


def test_low_balance_or_no_gpu_never_bills_a_shot(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    kg.tick(proj, led, colab, 1, engine_set="stub")
    bid = led.active_batch("colab")["id"]
    rel = mb.Release("o/r", gh=fr.gh)
    poor = FakeColab(tmp_path / "vms", balance=1.0)
    assert cb.take_one(rel, tmp_path / "w", poor, cb.settings(proj), []) is None
    assert not any(c[0] == "new" for c in poor.calls)    # no VM below the minimum balance
    nogpu = FakeColab(tmp_path / "vms2", grant=())
    assert cb.take_one(rel, tmp_path / "w2", nogpu, cb.settings(proj), []) == bid
    r = kg.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "collected" and r["made"] == 0, r
    assert led.failures("ref-style", "ref") == 0          # the jobs are free, not failed


def test_the_agreed_cap_stops_new_batches(tmp_path):
    proj, led, fr, colab = _setup(tmp_path, total_cap_units=10)
    led.create_batch("b-old", "refs", 1, 3600, backend="colab")
    led.set_batch("b-old", status="done", submitted_at=time.time() - 2 * 3600, finished_at=time.time())
    r = kg.tick(proj, led, colab, 1, engine_set="stub")   # 2 h at the A100 rate is past 10 units
    assert r["action"] == "resting" and "Colab cap is used up" in r["reason"], r
    assert fr.names() == []


def test_colab_test_still_reports_units(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    fake = FakeColab(tmp_path / "vms")
    r = cb.test_still(proj, fake, cb.settings(proj), [], tmp_path / "t")
    assert r["ok"] and r["gpu"] == "A100" and Path(r["still"]).exists() and r["units_used"] == 0


def test_a_stuck_kaggle_batch_moves_to_colab_only_when_colab_can_take_it(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    proj = copy_project(d, cfg_edit=lambda c: c.update(colab={"enabled": True, "total_cap_units": 10}))
    for f in (proj.root / "style_bible" / "refs").glob("*.png"):
        f.unlink()
    led = steps.open_ledger(proj)
    k = kg.LocalKaggle(proj.media)
    k.status = lambda ref, handle=None: "queued"            # Kaggle never starts it
    fr = FakeRelease(tmp_path / "release")
    colab = cb.ColabBackend(proj, "o/r", gh=fr.gh)
    assert kg.tick(proj, led, k, 1, engine_set="stub")["action"] == "built_and_submitted"
    b = led.active_batch("kaggle")
    led.set_batch(b["id"], submitted_at=time.time() - 5 * 3600)
    led.create_batch("b-spent", "refs", 1, 3600, backend="colab")   # the Colab cap is used up
    led.set_batch("b-spent", status="done", submitted_at=time.time() - 2 * 3600, finished_at=time.time())
    out = kg.tick_all(proj, led, [k, colab], 1, engine_set="stub")
    assert out[0]["action"] == "waiting" and led.active_batch("kaggle")["id"] == b["id"], out
    led.set_batch("b-spent", submitted_at=time.time() - 60)      # now there is room again
    out = kg.tick_all(proj, led, [k, colab], 1, engine_set="stub")
    assert out[0]["action"] == "handed_over", out
