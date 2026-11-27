"""Colab backend: the producer queues a batch, the runner runs it on a (fake) Colab VM, the
producer collects it; money guards and failures free the jobs."""
from __future__ import annotations

import json
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
    assert cb.take_one(rel, tmp_path / "w2", nogpu, cb.settings(proj), []) is None
    assert not any(n.endswith((".claim.json", ".results.tar.gz")) for n in fr.names())   # given back as it was
    assert colab.status(None, bid) == "queued"            # the next run tries again; nothing was billed
    ok = FakeColab(tmp_path / "vms3")
    assert cb.take_one(rel, tmp_path / "w3", ok, cb.settings(proj), []) == bid
    r = kg.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "collected" and r["made"] > 0, r
    assert led.failures("ref-style", "ref") == 0


def test_waiting_in_the_queue_costs_nothing_against_the_cap(tmp_path):
    proj, led, fr, colab = _setup(tmp_path, total_cap_units=20)
    led.create_batch("b-waited", "refs", 1, 3600, backend="colab")   # waited out 12 h, no VM ever opened
    led.set_batch("b-waited", status="failed", submitted_at=time.time() - 12.5 * 3600, finished_at=time.time())
    led.create_batch("b-ran", "refs", 1, 3600, backend="colab")      # waited 10 h, then 20 min on a VM
    led.set_batch("b-ran", status="done", submitted_at=time.time() - 10 * 3600, finished_at=time.time(),
                  gpu_seconds=1200)
    assert abs(colab.spent_units(led) - 1200 / 3600 * 5.4) < 0.01


def test_vm_time_and_units_come_back_with_the_results(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    kg.tick(proj, led, colab, 1, engine_set="stub")
    bid = led.active_batch("colab")["id"]
    fc = FakeColab(tmp_path / "vms")
    assert cb.take_one(mb.Release("o/r", gh=fr.gh), tmp_path / "w", fc, cb.settings(proj), []) == bid
    kg.tick(proj, led, colab, 1, engine_set="stub")
    row = led.db.execute("SELECT gpu_seconds FROM batches WHERE id=?", (bid,)).fetchone()
    assert row["gpu_seconds"] is not None and row["gpu_seconds"] < 3600


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


def test_no_units_means_the_test_still_never_opens_a_vm(tmp_path):
    proj = _setup(tmp_path)[0]
    cli = FakeColab(tmp_path / "vms", balance=0.0)
    r = cb.test_still(proj, cli, cb.settings(proj), [], tmp_path / "t")
    assert not r["ok"] and "Buy units" in r["error"]
    assert not any(c[0] == "new" for c in cli.calls)


def test_logs_never_show_the_account_address():
    assert cb._redact("signed in as some.one+x@gmail.com ok") == "signed in as <account> ok"


def test_blank_stills_stop_the_batch_after_two(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mochi_worker", kg.WORKER)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    calls = []

    def blank(self, job, inputs, out):
        calls.append(job["key"])
        raise RuntimeError("reference still came out blank or NaN at step 1 (in the transformer)")
    monkeypatch.setattr(w.StubImage, "run", blank)
    proj = _setup(tmp_path)[0]
    jobs = [{"key": f"r{i}", "shot_id": f"r{i}", "kind": "ref", "idx": 0, "prompt": "x", "est_seconds": 1}
            for i in range(6)]
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"batch_id": "b", "stage": "refs", "engines": proj.config["engines"], "jobs": jobs}))
    r = w.run(man, tmp_path, tmp_path / "out", "stub")
    assert calls == ["r0", "r1"]                         # no GPU time for the other 4
    assert r["groups"]["refs"]["stopped_broken"] == 4
    assert all(not v["ok"] for v in r["jobs"].values())
    from pipeline.ledger import not_a_failure
    assert not_a_failure(r["jobs"]["r0"]["error"])       # and they do not count against the shots


def test_a_small_balance_shrinks_the_batch_instead_of_running_dry(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    kg.tick(proj, led, colab, 1, engine_set="stub")
    fc = FakeColab(tmp_path / "vms", balance=8.0)       # 7 spare units at 5.4/h is about 1.3 h
    cb.take_one(mb.Release("o/r", gh=fr.gh), tmp_path / "w", fc, cb.settings(proj), [])
    man = json.loads(next((tmp_path / "w").rglob("in/manifest.json")).read_text())
    assert man["deadline_seconds"] < 1.3 * 3600 - 900
