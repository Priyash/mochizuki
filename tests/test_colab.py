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

from test_mac import FakeRelease
from test_pipeline import copy_project

from engine import scheduler as sched
from engine import steps
from engine.providers import colab as cb
from engine.providers.base import WORKER
from engine.providers.kaggle import LocalKaggle
from engine.providers.release import Release


class FakeColab:
    """Plays the `colab` CLI: one VM is a folder; exec runs the job's worker with stub engines."""

    def __init__(self, root: Path, grant=("A100",), balance=50.0, leftover=()):
        self.root, self.grant, self.bal = root, set(grant), balance
        self.vms: dict[str, Path] = {}
        self.left = list(leftover)
        self.calls: list[list[str]] = []
        self.gpu: dict[str, str] = {}
        self.orphans: list[dict] = []

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
            self.gpu[s] = gpu
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

    def assignments(self):
        return [{"endpoint": k, "accelerator": v} for k, v in self.gpu.items() if k in self.vms] + self.orphans

    def unassign(self, endpoint):
        self.calls.append(["unassign", endpoint])
        self.orphans = [o for o in self.orphans if o["endpoint"] != endpoint]
        self.vms.pop(endpoint, None)
        return True

    def stream(self, args, deadline, on_line=None):
        self.calls.append(args)
        s = args[args.index("-s") + 1]
        vm = self.vms[s]
        env = dict(a.split("=", 1) for a in args[args.index("--env"):][1::2])
        with tarfile.open(vm / "in.tar.gz") as t:
            t.extractall(vm / "in", filter="data")
        r = subprocess.run(
            [
                sys.executable,
                str(vm / "in" / "worker.py"),
                "--manifest",
                str(vm / "in" / "manifest.json"),
                "--inputs",
                str(vm / "in"),
                "--out",
                str(vm / "out"),
                "--engine-set",
                "stub",
                "--expect-batch",
                env["MOCHI_BATCH"],
            ],
            capture_output=True,
            text=True,
        )
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
    return proj, steps.open_ledger(proj), fr, cb.ColabProvider(proj, "o/r", gh=fr.gh)


def test_batch_runs_on_colab_and_comes_back(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    r = sched.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "built_and_submitted" and r["stage"] == "refs", r
    bid = led.active_batch("colab")["id"]
    assert fr.names() == [f"colab-{bid}.tar.gz"]
    fake = FakeColab(tmp_path / "vms", grant=("L4",))
    fake.orphans = [{"endpoint": "old-vm", "accelerator": "A100"}]
    assert cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", fake, cb.settings(proj), []) == bid
    assert fake.vms == {} and fake.orphans == []        # its VM and a leftover one are both stopped
    assert ["new", "-s", f"mochi-{bid[-11:]}", "--gpu", "A100"] in fake.calls   # A100 asked first, L4 granted
    r = sched.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "collected" and r["made"] > 0 and r["failed"] == 0, r
    assert fr.names() == []


def test_low_balance_or_no_gpu_never_bills_a_shot(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")
    bid = led.active_batch("colab")["id"]
    rel = Release("o/r", gh=fr.gh)
    poor = FakeColab(tmp_path / "vms", balance=1.0)
    assert cb.take_one(rel, tmp_path / "w", poor, cb.settings(proj), []) is None
    assert not any(c[0] == "new" for c in poor.calls)    # no VM below the minimum balance
    nogpu = FakeColab(tmp_path / "vms2", grant=())
    assert cb.take_one(rel, tmp_path / "w2", nogpu, cb.settings(proj), []) is None
    assert not any(n.endswith((".claim.json", ".results.tar.gz")) for n in fr.names())   # given back as it was
    assert colab.status(None, bid) == "queued"            # the next run tries again; nothing was billed
    ok = FakeColab(tmp_path / "vms3")
    assert cb.take_one(rel, tmp_path / "w3", ok, cb.settings(proj), []) == bid
    r = sched.tick(proj, led, colab, 1, engine_set="stub")
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
    sched.tick(proj, led, colab, 1, engine_set="stub")
    bid = led.active_batch("colab")["id"]
    fc = FakeColab(tmp_path / "vms")
    assert cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", fc, cb.settings(proj), []) == bid
    sched.tick(proj, led, colab, 1, engine_set="stub")
    row = led.db.execute("SELECT gpu_seconds FROM batches WHERE id=?", (bid,)).fetchone()
    assert row["gpu_seconds"] is not None and row["gpu_seconds"] < 3600


def test_the_agreed_cap_stops_new_batches(tmp_path):
    proj, led, fr, colab = _setup(tmp_path, total_cap_units=10)
    led.create_batch("b-old", "refs", 1, 3600, backend="colab")
    led.set_batch("b-old", status="done", submitted_at=time.time() - 2 * 3600, finished_at=time.time())
    r = sched.tick(proj, led, colab, 1, engine_set="stub")   # 2 h at the A100 rate is past 10 units
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
    k = LocalKaggle(proj.media)
    k.status = lambda ref, handle=None: "queued"            # Kaggle never starts it
    fr = FakeRelease(tmp_path / "release")
    colab = cb.ColabProvider(proj, "o/r", gh=fr.gh)
    assert sched.tick(proj, led, k, 1, engine_set="stub")["action"] == "built_and_submitted"
    b = led.active_batch("kaggle")
    led.set_batch(b["id"], submitted_at=time.time() - 5 * 3600)
    led.create_batch("b-spent", "refs", 1, 3600, backend="colab")   # the Colab cap is used up
    led.set_batch("b-spent", status="done", submitted_at=time.time() - 2 * 3600, finished_at=time.time())
    out = sched.tick_all(proj, led, [k, colab], 1, engine_set="stub")
    assert out[0]["action"] == "waiting" and led.active_batch("kaggle")["id"] == b["id"], out
    led.set_batch("b-spent", submitted_at=time.time() - 60)      # now there is room again
    out = sched.tick_all(proj, led, [k, colab], 1, engine_set="stub")
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
    spec = importlib.util.spec_from_file_location("mochi_worker", WORKER)
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
    from engine.ledger import not_a_failure
    assert not_a_failure(r["jobs"]["r0"]["error"])       # and they do not count against the shots


def test_a_small_balance_shrinks_the_batch_instead_of_running_dry(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")
    fc = FakeColab(tmp_path / "vms", balance=8.0)       # 7 spare units at 5.4/h is about 1.3 h
    cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", fc, cb.settings(proj), [])
    man = json.loads(next((tmp_path / "w").rglob("in/manifest.json")).read_text())
    assert man["deadline_seconds"] < 1.3 * 3600 - 900


def test_a_gpu_vm_left_on_colab_is_stopped_before_anything_else(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")
    fc = FakeColab(tmp_path / "vms")
    fc.orphans = [{"endpoint": "gpu-a100-xyz", "accelerator": "A100"}, {"endpoint": "cpu-1", "accelerator": "NONE"}]
    cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", fc, cb.settings(proj), [])
    assert ["unassign", "gpu-a100-xyz"] in fc.calls      # billing GPU VM stopped
    assert ["unassign", "cpu-1"] not in fc.calls          # free CPU VM left alone


def test_a_gpu_that_comes_up_after_the_request_gave_up_is_stopped(tmp_path):
    proj = _setup(tmp_path)[0]

    class Late(FakeColab):
        def run(self, args, timeout=None):
            if args[0] == "new":                         # the request times out, the VM appears anyway
                self.calls.append(args)
                self.orphans.append({"endpoint": "late-" + args[args.index("--gpu") + 1],
                                     "accelerator": args[args.index("--gpu") + 1]})
                return 1, "Read timed out."
            return super().run(args, timeout)
    fc = Late(tmp_path / "vms")
    r = cb.run_remote(fc, cb.settings(proj), "mochi-x", tmp_path / "none.tgz", "b", tmp_path / "o.tgz", [])
    assert r["gpu"] is None
    assert ["unassign", "late-A100"] in fc.calls and ["unassign", "late-L4"] in fc.calls
    assert fc.orphans == []
    assert r["late_vms"] == 2 and "came up late" in r["error"]    # it never says "no GPU" alone after one ran


class _Late(FakeColab):
    """Every GPU request times out, and the VM appears anyway (it bills until stopped)."""
    def run(self, args, timeout=None):
        if args[0] == "new":
            self.calls.append(args)
            gpu = args[args.index("--gpu") + 1]
            self.orphans.append({"endpoint": "late-" + gpu, "accelerator": gpu})
            self.bal -= 0.5
            return 1, "Read timed out."
        return super().run(args, timeout)


def test_a_late_gpu_is_reported_as_run_not_as_nothing_billed(tmp_path, capsys):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")
    assert cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", _Late(tmp_path / "vms"), cb.settings(proj), []) is None
    said = capsys.readouterr().out
    assert "gpu.late" in said and "1.0 units used" in said and "nothing billed" not in said
    assert not any(n.endswith(".claim.json") for n in fr.names())    # the batch waits for the next run


def test_no_gpu_and_no_late_vm_is_reported_as_nothing_billed(tmp_path, capsys):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")
    assert cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", FakeColab(tmp_path / "vms", grant=()),
                       cb.settings(proj), []) is None
    said = capsys.readouterr().out
    assert "gpu.none" in said and "no VM came up, nothing billed" in said


def test_unlisted_vms_are_never_reported_as_nothing_billed(tmp_path, capsys):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")
    blind = FakeColab(tmp_path / "vms", grant=())
    blind.assignments = lambda: None                    # the CLI could not list the account's VMs
    assert cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", blind, cb.settings(proj), []) is None
    said = capsys.readouterr().out
    assert "gpu.unchecked" in said and "nothing billed" not in said


def test_a_leftover_vm_is_stopped_even_when_no_batch_waits(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    fake = FakeColab(tmp_path / "vms")
    fake.orphans = [{"endpoint": "dead-run-a100", "accelerator": "A100"}]
    assert cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", fake, cb.settings(proj), []) is None
    assert ["unassign", "dead-run-a100"] in fake.calls


def test_a_gpu_granted_late_while_another_runs_is_stopped_after_the_batch(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")

    class LateA100(FakeColab):                           # A100 times out but comes up; L4 is granted
        def run(self, args, timeout=None):
            if args[0] == "new" and "A100" in args:
                self.calls.append(args)
                return 1, "Read timed out."
            if args[0] == "upload":                      # ... the A100 appears while the L4 is busy
                self.orphans.append({"endpoint": "late-A100", "accelerator": "A100"})
            return super().run(args, timeout)
    fake = LateA100(tmp_path / "vms", grant=("L4",))
    assert cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", fake, cb.settings(proj), [])
    assert ["unassign", "late-A100"] in fake.calls and fake.orphans == []


def test_a_run_that_does_not_finish_keeps_what_it_made(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")
    bid = led.active_batch("colab")["id"]

    class Dropped(FakeColab):                            # the stream drops before MOCHI-DONE; the job had
        def stream(self, args, deadline, on_line=None):  # packed its outputs as it went
            code, tail = super().stream(args, deadline, on_line)
            return 255, tail.replace("[job] MOCHI-DONE", "connection went stale")

    assert (
        cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", Dropped(tmp_path / "vms"), cb.settings(proj), []) == bid
    )
    r = sched.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "collected" and r["made"] > 0 and not r["stale"], r   # "stale" in the log drops nothing


def test_a_failed_results_download_keeps_them_in_the_release(tmp_path):
    proj, led, fr, colab = _setup(tmp_path)
    sched.tick(proj, led, colab, 1, engine_set="stub")
    bid = led.active_batch("colab")["id"]
    assert (
        cb.take_one(Release("o/r", gh=fr.gh), tmp_path / "w", FakeColab(tmp_path / "vms"), cb.settings(proj), []) == bid
    )
    fails = {"n": 1}
    def flaky(url, dest):
        if fails["n"]:
            fails["n"] -= 1
            raise OSError("connection reset")
        Release._urllib(url, dest)
    colab = cb.ColabProvider(proj, "o/r", gh=fr.gh, fetch=flaky)
    r = sched.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "waiting" and r["status"] == "collecting", r
    assert f"colab-{bid}.results.tar.gz" in fr.names()            # nothing deleted after a failed download
    r = sched.tick(proj, led, colab, 1, engine_set="stub")
    assert r["action"] == "collected" and r["made"] > 0, r
    assert fr.names() == []


class _Clock:
    """time.time for the runner, moved on by its sleeps."""

    def __init__(self):
        self.now, self.slept = 1_800_000_000.0, []

    def time(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


class _GrantsLate(FakeColab):
    """Colab says Bad Gateway to every request until round `at`, then grants the A100."""

    def __init__(self, root, at, clock):
        super().__init__(root, grant=())
        self.at, self.clock, self.asked = at, clock, 0

    def run(self, args, timeout=None):
        if args[0] == "new":
            self.asked += 1
            self.clock.now += 60                          # each request takes a while to fail
            if (self.asked + 1) // 2 >= self.at:
                self.grant = {"A100", "L4"}
            elif args[args.index("--gpu") + 1] not in self.grant:
                self.calls.append(args)
                return 1, "accelerator=A100: Bad Gateway"
        if args[0] == "upload":                           # the test ends once a GPU is granted
            return 1, "stopped here"
        return super().run(args, timeout)


def test_the_runner_keeps_asking_for_a_gpu_until_one_is_granted(tmp_path, monkeypatch):
    proj = _setup(tmp_path)[0]
    clock = _Clock()
    monkeypatch.setattr(cb.runner.time, "time", clock.time)
    fc = _GrantsLate(tmp_path / "vms", at=3, clock=clock)
    r = cb.run_remote(fc, cb.settings(proj), "mochi-x", tmp_path / "none.tgz", "b", tmp_path / "o.tgz", [],
                      gpu_wait=3600, sleep=clock.sleep)
    assert r["gpu"] == "A100" and clock.slept == [cb.runner.GPU_RETRY_SECONDS] * 2   # rounds 1 and 2 failed


def test_the_runner_stops_asking_when_its_wait_is_over_and_nothing_billed(tmp_path, monkeypatch):
    proj = _setup(tmp_path)[0]
    clock = _Clock()
    monkeypatch.setattr(cb.runner.time, "time", clock.time)
    fc = _GrantsLate(tmp_path / "vms", at=99, clock=clock)
    r = cb.run_remote(fc, cb.settings(proj), "mochi-x", tmp_path / "none.tgz", "b", tmp_path / "o.tgz", [],
                      gpu_wait=900, sleep=clock.sleep)
    assert r["gpu"] is None and "round(s) of requests" in r["error"] and "Bad Gateway" in r["error"]
    assert clock.now - 1_800_000_000.0 < 900 + 2 * 60 + cb.runner.GPU_RETRY_SECONDS    # it never overstays
    assert r["late_vms"] == 0


def test_the_gpu_wait_fits_the_github_job_or_is_one_round(tmp_path, monkeypatch):
    proj = _setup(tmp_path)[0]
    cfg = {**cb.settings(proj), "gpu_wait_minutes": 150, "batch_hours": 1.0, "setup_minutes": 15}
    monkeypatch.delenv("COLAB_JOB_DEADLINE", raising=False)
    assert cb.runner.gpu_wait_seconds(cfg, now=0) == 0                      # no known end: one round only
    monkeypatch.setenv("COLAB_JOB_DEADLINE", str(330 * 60))
    run = cb.runner._wait_cap(cfg)                                          # what the batch may need
    assert cb.runner.gpu_wait_seconds(cfg, now=0) == min(150 * 60, 330 * 60 - run - 300)
    four = {**cfg, "batch_hours": 4.0}                                       # a production batch: less time to ask
    assert cb.runner.gpu_wait_seconds(four, now=0) == 330 * 60 - cb.runner._wait_cap(four) - 300
    assert cb.runner.gpu_wait_seconds({**cfg, "batch_hours": 5.0}, now=0) == 0   # no time left to wait at all
