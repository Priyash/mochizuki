"""RunPod backend: pod lifecycle, money guards and stage routing, against a fake RunPod."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from pipeline import kaggle as kg
from pipeline import runpod_backend as rpb
from pipeline import steps

from test_pipeline import REPO, copy_project


def _mk(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


class FakeRunPod:
    """Plays RunPod's API, its HTTPS proxy and the pod agent; runs the worker here with stubs."""

    def __init__(self, root: Path, refuse=(), balance_ok=True):
        self.root = root
        self.pods: dict[str, dict] = {}
        self.released: dict[str, Path] = {}
        self.refuse = set(refuse)
        self.balance_ok = balance_ok
        self.calls: list[str] = []
        self.n = 0

    def gh(self, args):
        self.calls.append("gh " + " ".join(args[:2]))
        if args[:2] == ["release", "upload"]:
            self.released[Path(args[3]).name] = Path(args[3])
        if args[:2] == ["release", "delete-asset"]:
            self.released.pop(args[3], None)
        return 0, ""

    def http(self, method, url, body=None, headers=None, timeout=60):
        self.calls.append(f"{method} {url}")
        if url.startswith(rpb.API):
            return self._api(method, url[len(rpb.API):], body)
        pod_id = url.split("//")[1].split("-")[0]
        pod = self.pods.get(pod_id)
        if not pod or pod["status"] == "TERMINATED":
            return 404, b"no pod"
        if headers.get("X-Token") != pod["env"]["MOCHI_TOKEN"]:
            return 403, b"forbidden"
        path = url.split(".net", 1)[1]
        if path == "/status":
            return 200, json.dumps({"state": "done", "log": ["fake"]}).encode()
        if path.startswith("/file/"):
            f = pod["out"] / path[6:]
            return (200, f.read_bytes()) if f.exists() else (404, b"")
        if path == "/done":
            return 200, b"{}"
        return 404, b""

    def _api(self, method, path, body):
        if method == "GET" and path == "/pods":
            return 200, json.dumps({"pods": [{"id": k, "name": v["name"]} for k, v in self.pods.items()
                                             if v["status"] != "TERMINATED"]}).encode()
        if method == "POST" and path == "/pods":
            if not body:
                return 422, b'{"title": "invalid"}'
            if not self.balance_ok:
                return 402, b'{"title": "insufficient balance"}'
            if body["gpu"]["id"] in self.refuse:
                return 400, b'{"detail": "no placement"}'
            self.n += 1
            pid = f"pod{self.n}"
            env = body["env"]
            work = self.root / pid
            tgz = self.released[env["MOCHI_BUNDLE_URL"].rsplit("/", 1)[1]]
            with tarfile.open(tgz) as t:
                t.extractall(work / "in", filter="data")
            r = subprocess.run([sys.executable, str(work / "in" / "worker.py"), "--manifest",
                                str(work / "in" / "manifest.json"), "--inputs", str(work / "in"), "--out",
                                str(work / "out"), "--engine-set", "stub", "--expect-batch", env["MOCHI_BATCH"]],
                               capture_output=True, text=True)
            assert r.returncode in (0, 3), r.stdout + r.stderr
            self.pods[pid] = {"name": body["name"], "env": env, "gpu": body["gpu"]["id"], "status": "RUNNING",
                              "out": work / "out", "cloud": body["cloud"]}
            return 201, json.dumps({"id": pid, "status": "PROVISIONING", "cost": 0.34}).encode()
        if path.startswith("/pods/"):
            pid = path.split("/")[2]
            if pid not in self.pods or self.pods[pid]["status"] == "TERMINATED":
                return 404, b'{"title": "not found"}'
            if method == "DELETE":
                self.pods[pid]["status"] = "TERMINATED"
                return 204, b""
            return 200, json.dumps({"id": pid, "status": self.pods[pid]["status"]}).encode()
        if path.startswith("/catalog/gpus"):
            return 200, json.dumps({"gpus": [{"id": "NVIDIA GeForce RTX 4090", "price": {"community": 0.34},
                                              "availability": "HIGH"}]}).encode()
        return 404, b'{"title": "not found"}'

    def live(self):
        return [k for k, v in self.pods.items() if v["status"] != "TERMINATED"]


def backend(proj, fake, **cfg):
    raw = json.loads((proj.root / "config.json").read_text())
    raw["runpod"] = {**raw.get("runpod", {}), "enabled": True, "total_cap_usd": 30, **cfg}
    (proj.root / "config.json").write_text(json.dumps(raw))
    proj.config = raw
    return rpb.RunPodBackend(proj, "secret-key", "o/r", "abc123", http=fake.http, gh=fake.gh)


def test_smoke_runs_one_clip_and_deletes_the_pod(tmp_path):
    proj = copy_project(_mk(tmp_path / "p"))
    fake = FakeRunPod(tmp_path / "pods")
    rp = backend(proj, fake)
    r = rpb.smoke(proj, rp, "stub", sleep=lambda s: None)
    assert r["ok"] and Path(r["clip"]).exists(), r
    assert fake.live() == []                                   # deleted after collecting
    assert fake.released == {}                                 # inputs file removed too
    pod = fake.pods["pod1"]
    assert pod["cloud"] == "COMMUNITY" and pod["gpu"] == "NVIDIA GeForce RTX 4090"
    assert pod["env"]["MOCHI_AGENT_URL"].endswith("/o/r/abc123/worker/pod_agent.py")
    assert "secret-key" not in json.dumps(pod["env"])           # the account key never goes to the pod


def test_push_tries_the_next_gpu_but_stops_on_an_empty_balance(tmp_path):
    proj = copy_project(_mk(tmp_path / "p"))
    fake = FakeRunPod(tmp_path / "pods", refuse={"NVIDIA GeForce RTX 4090"})
    r = rpb.smoke(proj, backend(proj, fake), "stub", sleep=lambda s: None)
    assert r["ok"] and fake.pods["pod1"]["gpu"] == "NVIDIA GeForce RTX 3090"
    broke = FakeRunPod(tmp_path / "pods2", balance_ok=False)
    rp = backend(proj, broke)
    try:
        rpb.smoke(proj, rp, "stub", sleep=lambda s: None)
        raise AssertionError("should have failed")
    except rpb.RunPodError as e:
        assert "402" in str(e) and str(e).count("NVIDIA") == 1  # did not try the second GPU


def test_runpod_only_takes_its_stages_unless_kaggle_is_stuck(tmp_path):
    proj = copy_project(_mk(tmp_path / "p"), cfg_edit=lambda c: [c.update(runpod={"enabled": True, "total_cap_usd": 30})])
    for f in (proj.root / "style_bible" / "refs").glob("*.png"):
        f.unlink()                                             # chapter 1 needs reference stills first
    led = steps.open_ledger(proj)
    fake = FakeRunPod(tmp_path / "pods")
    rp = backend(proj, fake)
    r = kg.tick(proj, led, rp, 1, engine_set="stub")
    assert r["action"] == "idle" and "only takes video" in r["reason"]
    led.create_batch("bk-stuck", "refs", 1, 3600, backend="kaggle")
    led.set_batch("bk-stuck", status="failed", note="stuck in Kaggle's queue; handed to the other backend")
    r = kg.tick(proj, led, rp, 1, engine_set="stub")
    assert r["action"] == "built_and_submitted" and r["stage"] == "refs", r
    b = led.active_batch("runpod")
    assert b["handle"].startswith("pod1 ") and b["handle"].endswith(" 0.340")
    r = kg.tick(proj, led, rp, 1, engine_set="stub")
    assert r["action"] == "collected" and r["made"] > 0, r
    assert fake.live() == []


def test_cap_stops_new_batches_and_orphans_are_deleted(tmp_path):
    proj = copy_project(_mk(tmp_path / "p"))
    led = steps.open_ledger(proj)
    fake = FakeRunPod(tmp_path / "pods")
    rp = backend(proj, fake, total_cap_usd=2, usd_per_hour=0.4)
    led.create_batch("b-old", "video", 1, 3600, backend="runpod")
    led.set_batch("b-old", status="done", submitted_at=time.time() - 5 * 3600, finished_at=time.time())
    try:
        rp.budget(proj, led)
        raise AssertionError("cap not enforced")
    except kg.BudgetUsedUp as e:
        assert "$2.00 of $2.00" in str(e)
    fake.pods["stray"] = {"name": "mochizuki-b-gone", "status": "RUNNING", "env": {}, "out": tmp_path}
    fake.pods["mine"] = {"name": "someone-else", "status": "RUNNING", "env": {}, "out": tmp_path}
    out = kg.tick_all(proj, led, [rp], 1, engine_set="stub")
    assert {"action": "cleaned_up", "backend": "runpod", "pods_deleted": 1} in out
    assert fake.live() == ["mine"]                              # only our own stray pod is touched


def test_check_never_creates_a_pod(tmp_path):
    proj = copy_project(_mk(tmp_path / "p"))
    fake = FakeRunPod(tmp_path / "pods")
    r = rpb.check(backend(proj, fake))
    assert r["ok"] and r["pods_write"]["code"] == 422 and fake.pods == {}
    assert r["catalog_read"]["gpus"]["NVIDIA GeForce RTX 4090"]["price"] == 0.34


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_pod_agent_serves_results_behind_its_token(tmp_path):
    proj = copy_project(_mk(tmp_path / "p"))
    data = tmp_path / "bundle"
    data.mkdir()
    shutil.copyfile(REPO / "worker" / "worker.py", data / "worker.py")
    from test_pipeline import write_png
    write_png(data / "a.png", 64, 36)
    job = {"key": "t1", "shot_id": "s", "kind": "take", "idx": 0, "model": "Wan 2.2 5B", "seconds": 1,
           "prompt": "x", "image": "a.png", "est_seconds": 10}
    (data / "manifest.json").write_text(json.dumps({"batch_id": "B1", "engines": proj.config["engines"],
                                                    "deadline_seconds": 3000, "jobs": [job]}))
    tgz = tmp_path / "B1.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        for f in data.iterdir():
            t.add(f, arcname=f.name)
    port = _free_port()
    env = dict(os.environ, MOCHI_ROOT=str(tmp_path / "work"), MOCHI_BATCH="B1", MOCHI_TOKEN="tok",
               MOCHI_BUNDLE_URL=tgz.as_uri(), MOCHI_DEADLINE=str(time.time() + 600), MOCHI_ENGINE_SET="stub",
               MOCHI_PORT=str(port), MOCHI_SKIP_PIP="1")
    env.pop("RUNPOD_POD_ID", None)
    agent = subprocess.Popen([sys.executable, str(REPO / "worker" / "pod_agent.py")], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"

    def get(path, token="tok", method="GET"):
        req = urllib.request.Request(base + path, method=method, headers={"X-Token": token})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, b""
        except OSError:
            return 0, b""

    try:
        st = {}
        for _ in range(120):
            code, raw = get("/status")
            st = json.loads(raw) if code == 200 else {}
            if st.get("state") in ("done", "error"):
                break
            time.sleep(0.5)
        assert st.get("state") == "done", st
        assert get("/status", token="wrong")[0] == 403
        assert get("/file/../../etc/passwd")[0] in (403, 404)
        code, raw = get("/file/results.json")
        res = json.loads(raw)
        assert res["jobs"]["t1"]["ok"]
        assert get("/file/" + res["jobs"]["t1"]["file"])[1][:8]          # the clip downloads
        assert get("/done", method="POST")[0] == 200
        assert agent.wait(20) == 0                                       # no pod id here: it just exits
    finally:
        if agent.poll() is None:
            agent.kill()
