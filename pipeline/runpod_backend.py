"""RunPod as a paid, capped GPU backend beside Kaggle.

Same batch contract as Kaggle (publish inputs, push, status, output). One batch = one rented
machine (a "pod"), created for the batch and deleted when it is collected:

  publish  the packed inputs are uploaded as a file of the `batches` GitHub release (the repo is
           public, so the pod downloads them without any credentials)
  push     a pod is created on the cheapest allowed GPU (Community Cloud). It downloads
           worker/pod_agent.py at this commit, which fetches the inputs, runs the worker and
           serves the results over RunPod's HTTPS proxy, behind a per-batch token
  status   the pod's state from the RunPod API, then the agent's own status
  output   every result file is downloaded, then the pod is deleted (and the inputs file too)

Money: RunPod bills only from the prepaid balance the owner loads; the pipeline never touches
payment. On top, config.json -> runpod.total_cap_usd is counted against wall-clock time of
every pod at runpod.usd_per_hour, each batch is sized to what is left, the pod is killed at its
deadline, deletes itself when its results are collected or not collected in time, and any
mochizuki-* pod the ledger no longer tracks is deleted on every run.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/v2"
PREFIX = "mochizuki-"
DEFAULTS = {
    "enabled": False,
    "gpus": ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA RTX 5000 Ada Generation"],
    "cloud": "COMMUNITY",
    "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "min_cuda": "12.8",
    "disk_gb": 120,
    "usd_per_hour": 0.40,
    "max_usd_per_hour": 0.60,
    "total_cap_usd": 0,
    "batch_hours": 3,
    "setup_minutes": 20,
    "keep_results_hours": 2.5,
    "stages": ["video"],
    "take_over_when_kaggle_stuck": True,
    "port": 8000,
    "release": "batches",
}


class RunPodError(RuntimeError):
    pass


def settings(p) -> dict:
    return {**DEFAULTS, **(p.config.get("runpod", {}) if p is not None else {})}


class RunPodBackend:
    name = "runpod"
    accelerator = ""

    def __init__(self, p, api_key: str, repo: str, sha: str = "main", http=None, gh=None, now=time.time):
        self.cfg = settings(p)
        self.key = api_key
        self.repo = repo
        self.sha = sha or "main"
        self._http = http or self._urllib
        self._gh = gh or self._gh_cli
        self.now = now
        self.stages = tuple(self.cfg["stages"]) if self.cfg.get("stages") else None
        self.take_over = bool(self.cfg.get("take_over_when_kaggle_stuck"))
        self.log: list[str] = []
        self._bundle: dict[str, str] = {}
        self._engine_set = "gpu"

    # ---------------------------------------------------------------- plumbing
    def _urllib(self, method: str, url: str, body=None, headers=None, timeout=60) -> tuple[int, bytes]:
        data = json.dumps(body).encode() if body is not None else None
        h = dict(headers or {})
        if data is not None:
            h["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:  # noqa: BLE001 - network trouble is a status, not a crash
            return 0, repr(e).encode()

    @staticmethod
    def _gh_cli(args: list[str]) -> tuple[int, str]:
        r = subprocess.run(["gh", *args], capture_output=True, text=True)
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def api(self, method: str, path: str, body=None) -> tuple[int, dict]:
        code, raw = self._http(method, API + path, body, {"Authorization": f"Bearer {self.key}"})
        try:
            obj = json.loads(raw or b"{}")
        except ValueError:
            obj = {"detail": raw[:300].decode(errors="replace") if isinstance(raw, bytes) else str(raw)[:300]}
        if code >= 400 or code == 0:
            self.log.append(f"runpod {method} {path} -> {code}: {str(obj)[:300]}")
        return code, obj if isinstance(obj, dict) else {"items": obj}

    def token(self, batch_id: str) -> str:
        """Per-batch secret, derived so it never has to be stored (the ledger is public)."""
        return hmac.new(self.key.encode(), batch_id.encode(), hashlib.sha256).hexdigest()[:40]

    def proxy(self, pod_id: str) -> str:
        return f"https://{pod_id}-{self.cfg['port']}.proxy.runpod.net"

    def agent(self, pod_id: str, batch_id: str, method: str, path: str, timeout=60) -> tuple[int, bytes]:
        return self._http(method, self.proxy(pod_id) + path, None, {"X-Token": self.token(batch_id)}, timeout)

    @staticmethod
    def _split(handle: str | None) -> tuple[str, str]:
        parts = (handle or "").split()
        return (parts + ["", ""])[0], (parts + ["", ""])[1]

    # ---------------------------------------------------------------- money
    def spent_usd(self, led) -> float:
        """Wall-clock time of every pod at the hourly price RunPod quoted for it (handle's 3rd field)."""
        rows = led.db.execute("SELECT submitted_at, finished_at, handle FROM batches WHERE backend=?"
                              " AND submitted_at IS NOT NULL", (self.name,)).fetchall()
        usd = 0.0
        for r in rows:
            parts = (r["handle"] or "").split()
            try:
                rate = float(parts[2])
            except (IndexError, ValueError):
                rate = float(self.cfg["usd_per_hour"])
            usd += max(0.0, (r["finished_at"] or self.now()) - r["submitted_at"]) / 3600 * rate
        return usd

    def budget(self, p, led) -> tuple[float, float]:
        from .kaggle import BudgetUsedUp  # noqa: PLC0415
        rate, cap = float(self.cfg["usd_per_hour"]), float(self.cfg["total_cap_usd"])
        setup = float(self.cfg["setup_minutes"]) * 60
        spent = self.spent_usd(led)
        left_s = (cap - spent) / rate * 3600 if rate > 0 else 0
        budget = min(float(self.cfg["batch_hours"]) * 3600, left_s - setup - 1200)
        if budget <= 900:
            raise BudgetUsedUp(f"the RunPod cap is used up (${spent:.2f} of ${cap:.2f}); "
                               "raise runpod.total_cap_usd in config.json to go on")
        return budget, setup

    def gpu_hours_left(self) -> float | None:
        return None

    # ---------------------------------------------------------------- the batch contract
    def publish_dataset(self, folder: Path, ref: str, message: str) -> None:
        """Pack the batch's inputs and put them in the `batches` release. `message` is the batch id."""
        batch_id = message
        tgz = Path(folder).parent / f"{batch_id}.tar.gz"
        with tarfile.open(tgz, "w:gz") as t:
            for f in sorted(Path(folder).iterdir()):
                if f.name != "dataset-metadata.json":
                    t.add(f, arcname=f.name)
        rel = self.cfg["release"]
        code, out = self._gh(["release", "view", rel, "-R", self.repo, "--json", "tagName"])
        if code != 0:
            code, out = self._gh(["release", "create", rel, "-R", self.repo, "--prerelease", "--title", rel,
                                  "--notes", "Inputs of RunPod batches, deleted after each batch. Do not edit."])
            if code != 0 and "already exists" not in out:
                raise RunPodError(f"could not create release {rel}: {out.strip()[-300:]}")
        code, out = self._gh(["release", "upload", rel, str(tgz), "-R", self.repo, "--clobber"])
        if code != 0:
            raise RunPodError(f"upload of {tgz.name} failed: {out.strip()[-300:]}")
        self._bundle[batch_id] = f"https://github.com/{self.repo}/releases/download/{rel}/{tgz.name}"
        self.log.append(f"inputs of {batch_id}: {tgz.stat().st_size // 1024} KB uploaded")

    def wait_dataset_ready(self, ref: str, **_) -> None:
        return None

    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str:
        """Create the pod. Returns the handle '<pod id> <batch id>'."""
        batch_id = Path(folder).parent.name
        boot = (Path(folder) / "bootstrap.py").read_text() if (Path(folder) / "bootstrap.py").exists() else ""
        engine_set = "stub" if "ENGINE_SET = 'stub'" in boot else self._engine_set
        from .kaggle import PACKAGES  # noqa: PLC0415
        c = self.cfg
        env = {
            "MOCHI_BATCH": batch_id,
            "MOCHI_BUNDLE_URL": self._bundle.get(batch_id) or
                                f"https://github.com/{self.repo}/releases/download/{c['release']}/{batch_id}.tar.gz",
            "MOCHI_TOKEN": self.token(batch_id),
            "MOCHI_DEADLINE": str(int(self.now() + timeout_s)),
            "MOCHI_KEEP_S": str(int(float(c["keep_results_hours"]) * 3600)),
            "MOCHI_ENGINE_SET": engine_set,
            "MOCHI_PACKAGES": json.dumps(PACKAGES),
            "MOCHI_PORT": str(c["port"]),
            "MOCHI_AGENT_URL": f"https://raw.githubusercontent.com/{self.repo}/{self.sha}/worker/pod_agent.py",
        }
        start = ('mkdir -p /work && curl -fsSL "$MOCHI_AGENT_URL" -o /work/pod_agent.py '
                 '&& exec python3 /work/pod_agent.py')
        tried = []
        for gpu in c["gpus"]:
            body = {"name": PREFIX + batch_id, "image": c["image"], "cloud": c["cloud"],
                    "gpu": {"id": gpu, "count": 1, "minCudaVersion": c["min_cuda"]},
                    "disk": int(c["disk_gb"]), "ports": [f"{c['port']}/http"], "env": env,
                    "entrypoint": ["bash", "-c"], "cmd": [start]}
            code, obj = self.api("POST", "/pods", body)
            if code in (200, 201) and obj.get("id"):
                pod = obj["id"]
                cost = obj.get("cost")
                self.log.append(f"pod {pod} for {batch_id} on {gpu} ({c['cloud']}), ${cost}/h, "
                                f"hard stop in {timeout_s / 3600:.1f} h")
                if cost is not None and float(cost) > float(c["max_usd_per_hour"]):
                    self.terminate(pod)
                    raise RunPodError(f"{gpu} came at ${cost}/h, over runpod.max_usd_per_hour; pod deleted")
                return f"{pod} {batch_id} {float(cost if cost is not None else c['usd_per_hour']):.3f}"
            tried.append(f"{gpu}: {code} {str(obj.get('detail') or obj.get('title') or obj)[:120]}")
            if code in (401, 402, 403):
                break                               # bad key, empty balance, no permission: another GPU won't help
        raise RunPodError("no pod could be created. " + "; ".join(tried))

    def status(self, ref: str, handle: str | None = None) -> str:
        pod, batch = self._split(handle)
        if not pod:
            return "error"
        code, obj = self.api("GET", f"/pods/{pod}")
        if code == 404:
            return "error"                          # gone: deleted itself, or by hand
        if code != 200:
            return "unknown"
        st = str(obj.get("status", "")).upper()
        if st in ("PROVISIONING", "STARTING"):
            return "queued"
        if st in ("EXITED", "ERROR", "TERMINATED"):
            self.log.append(f"pod {pod} is {st}")
            return "error"
        code, raw = self.agent(pod, batch, "GET", "/status", timeout=30)
        if code != 200:
            return "running"                        # proxy not up yet while the container boots
        s = json.loads(raw or b"{}")
        self.log.append(f"pod {pod}: {s.get('state')}; " + " | ".join(s.get("log", [])[-2:])[-300:])
        return {"done": "complete", "error": "error"}.get(s.get("state"), "running")

    def output(self, ref: str, dest: Path, handle: str | None = None) -> None:
        """Download every result, then delete the pod and the inputs file, whatever happens."""
        pod, batch = self._split(handle)
        out = Path(dest) / "out"
        out.mkdir(parents=True, exist_ok=True)
        try:
            code, raw = self.agent(pod, batch, "GET", "/file/results.json")
            if code != 200:
                self.log.append(f"no results from pod {pod}: {code}")
                return
            (out / "results.json").write_bytes(raw)
            for r in json.loads(raw).get("jobs", {}).values():
                if r.get("ok") and r.get("file"):
                    code, data = self.agent(pod, batch, "GET", f"/file/{r['file']}", timeout=600)
                    if code == 200:
                        (out / r["file"]).write_bytes(data)
                    else:
                        self.log.append(f"could not download {r['file']}: {code}")
        finally:
            self.agent(pod, batch, "POST", "/done", timeout=20)
            self.terminate(pod)
            if batch:
                self._gh(["release", "delete-asset", self.cfg["release"], f"{batch}.tar.gz", "-R", self.repo, "-y"])

    def terminate(self, pod: str) -> bool:
        if not pod:
            return False
        code, _ = self.api("DELETE", f"/pods/{pod}")
        ok = code in (200, 202, 204, 404)
        self.log.append(f"pod {pod} deleted" if ok else f"pod {pod} could NOT be deleted ({code})")
        return ok

    def cleanup(self, led) -> int:
        """Delete every mochizuki-* pod that no open batch owns. Run before anything else."""
        code, obj = self.api("GET", "/pods")
        if code != 200:
            return 0
        pods = obj.get("pods", obj.get("items", []))
        owned = {self._split(b["handle"])[0] for b in led.active_batches()
                 if b["backend"] == self.name and b["status"] == "submitted" and b["handle"]}
        n = 0
        for pod in pods:
            if str(pod.get("name", "")).startswith(PREFIX) and pod.get("id") not in owned:
                n += self.terminate(pod["id"])
        return n


def from_env(p):
    """The backend if RUNPOD_API_KEY is set and config.json switches RunPod on, else None."""
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key or not settings(p).get("enabled"):
        return None
    return RunPodBackend(p, key, os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_SHA", "main"))


def smoke(p, rp: RunPodBackend, engine_set: str = "stub", model: str = "Wan 2.2 5B",
          seconds: float = 2, max_minutes: float = 45, poll: float = 30, sleep=time.sleep) -> dict:
    """One tiny batch end to end on a real pod: create, run, collect, delete. Not in the ledger.

    stub: no GPU code runs (checks the plumbing, ~$0.03). gpu: one short real clip on `model`.
    """
    import shutil  # noqa: PLC0415
    import struct  # noqa: PLC0415
    import zlib  # noqa: PLC0415
    from .kaggle import BOOTSTRAP, PACKAGES, WORKER  # noqa: PLC0415
    import uuid  # noqa: PLC0415
    batch_id = time.strftime("smoke-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    work = p.media / "batches" / batch_id
    data, kern = work / "dataset", work / "kernel"
    data.mkdir(parents=True)
    kern.mkdir(parents=True)
    w, h = 1280, 704
    raw = b"".join(b"\x00" + bytes([40 + y // 8, 70, 120]) * w for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))  # noqa: E731
    (data / "smoke.png").write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    job = {"key": "smoke-take", "shot_id": "smoke", "kind": "take", "idx": 0, "model": model, "seconds": seconds,
           "prompt": "3D CG anime, a slow sunrise over a calm sea, gentle camera push in", "image": "smoke.png",
           "est_seconds": 1500}
    manifest = {"batch_id": batch_id, "stage": "video", "chapter": 0, "backend": rp.name,
                "deadline_seconds": max_minutes * 60 - 300, "engines": p.config["engines"], "jobs": [job]}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(WORKER, data / "worker.py")
    (kern / "bootstrap.py").write_text(BOOTSTRAP.format(batch_id=batch_id, engine_set=engine_set,
                                                        packages=", ".join(repr(x) for x in PACKAGES)))
    t0 = time.time()
    rp.publish_dataset(data, "", batch_id)
    handle = rp.push(kern, "", int(max_minutes * 60))
    report = {"batch": batch_id, "pod": handle.split()[0], "engine_set": engine_set, "model": model}
    st = "queued"
    try:
        while time.time() - t0 < max_minutes * 60:
            st = rp.status("", handle)
            if st in ("complete", "error"):
                break
            sleep(poll)
    finally:
        rp.output("", work / "download", handle)        # downloads what exists, always deletes the pod
    res_f = work / "download" / "out" / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    r = res.get("jobs", {}).get("smoke-take", {})
    clip = work / "download" / "out" / str(r.get("file", "-"))
    report.update(status=st, minutes=round((time.time() - t0) / 60, 1), ok=bool(r.get("ok")) and clip.exists(),
                  job=r, clip=str(clip) if clip.exists() else None, error=res.get("error"),
                  est_cost_usd=round((time.time() - t0) / 3600 * float(rp.cfg["usd_per_hour"]), 3))
    return report


def check(rp: RunPodBackend) -> dict:
    """Read-only check of the key: can it list pods, read the catalog and balance, and create pods?

    The write check sends an empty create request: RunPod rejects it as invalid (422/400) when
    the key may create pods, or refuses it (401/403) when it may not. No pod is ever created.
    """
    r: dict = {}
    code, obj = rp.api("GET", "/pods")
    r["pods_read"] = {"code": code, "ok": code == 200,
                      "mochizuki_pods": [p.get("id") for p in obj.get("pods", []) if str(p.get("name", "")).startswith(PREFIX)]}
    code, obj = rp.api("GET", f"/catalog/gpus?include=AVAILABILITY&product=POD&cloud={rp.cfg['cloud']}"
                              f"&minCudaVersion={rp.cfg['min_cuda']}")
    gpus = {g.get("id"): g for g in obj.get("gpus", [])}
    r["catalog_read"] = {"code": code, "ok": code == 200, "gpus": {
        gid: ({"price": (gpus[gid].get("price") or {}).get(rp.cfg["cloud"].lower()),
               "availability": gpus[gid].get("availability")} if gid in gpus else "not in catalog")
        for gid in rp.cfg["gpus"]}}
    for cloud in ("COMMUNITY", "SECURE"):           # what is in stock right now, 24 GB or more, any price
        code, obj = rp.api("GET", f"/catalog/gpus?include=AVAILABILITY&product=POD&cloud={cloud}")
        r[f"in_stock_{cloud.lower()}"] = sorted(
            (f"${(g.get('price') or {}).get(cloud.lower())} {g.get('availability')} {g.get('memory')}GB {g.get('id')}"
             for g in obj.get("gpus", []) if (g.get("memory") or 0) >= 24 and g.get("availability") not in (None, "NONE")),
            key=lambda x: float(x.split()[0][1:]) if x.split()[0][1:] not in ("None", "") else 99)
    for path in ("/billing/balance", "/billing", "/account/balance"):
        code, obj = rp.api("GET", path)
        if code != 404:
            r["balance"] = {"path": path, "code": code, "body": {k: v for k, v in obj.items() if k != "detail"} or obj}
            break
    code, obj = rp.api("POST", "/pods", {})
    r["pods_write"] = {"code": code, "ok": code in (400, 422),
                       "detail": str(obj.get("detail") or obj.get("title") or "")[:200]}
    r["ok"] = r["pods_read"]["ok"] and r["catalog_read"]["ok"] and r["pods_write"]["ok"]
    return r


def deep_check(rp: RunPodBackend, sha: str) -> dict:
    """Everything that can be tested without credit: the exact create request (an empty balance
    must refuse it with 402; anything created anyway is deleted at once), the image tag, the
    agent download, the inputs release round trip, and the API's 404 paths."""
    import urllib.parse  # noqa: PLC0415
    c, r = rp.cfg, {}
    # 1. the image exists on Docker Hub
    repo_name, tag = c["image"].split(":", 1)
    ns, name = repo_name.split("/", 1)
    code, raw = rp._http("GET", f"https://hub.docker.com/v2/namespaces/{ns}/repositories/{name}/tags/{tag}")
    r["image"] = {"tag": c["image"], "code": code, "ok": code == 200}
    if code != 200:
        code, raw = rp._http("GET", f"https://hub.docker.com/v2/namespaces/{ns}/repositories/{name}/tags"
                                    "?page_size=100&ordering=last_updated")
        try:
            names = [t["name"] for t in json.loads(raw).get("results", [])]
        except ValueError:
            names = []
        r["image"]["recent_tags"] = [n for n in names if "torch2" in n][:15]
    # 2. the agent the pod downloads, at this commit
    url = f"https://raw.githubusercontent.com/{rp.repo}/{sha}/worker/pod_agent.py"
    code, raw = rp._http("GET", url)
    r["agent_download"] = {"url": url, "code": code, "ok": code == 200 and b"def terminate" in (raw or b"")}
    # 3. inputs release: upload, public download, delete
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / "dataset"
        folder.mkdir()
        (folder / "manifest.json").write_text('{"probe": true}')
        bid = "probe-" + time.strftime("%H%M%S")
        try:
            rp.publish_dataset(folder, "", bid)
            code, raw = rp._http("GET", rp._bundle[bid])
            r["inputs_release"] = {"url": rp._bundle[bid], "public_download": code, "ok": code == 200}
        except Exception as e:  # noqa: BLE001
            r["inputs_release"] = {"ok": False, "error": str(e)[:300]}
        finally:
            rp._gh(["release", "delete-asset", c["release"], f"{bid}.tar.gz", "-R", rp.repo, "-y"])
    # 4. the exact create request; with no credit RunPod must refuse it
    tried = {}
    for gpu in c["gpus"]:
        body = {"name": PREFIX + "dryrun", "image": c["image"], "cloud": c["cloud"],
                "gpu": {"id": gpu, "count": 1, "minCudaVersion": c["min_cuda"]},
                "disk": int(c["disk_gb"]), "ports": [f"{c['port']}/http"], "env": {"MOCHI_DRYRUN": "1"},
                "entrypoint": ["bash", "-c"], "cmd": ["sleep 5"]}
        code, obj = rp.api("POST", "/pods", body)
        tried[gpu] = {"code": code, "detail": str(obj.get("detail") or obj.get("title") or "")[:200]}
        if code in (200, 201) and obj.get("id"):
            tried[gpu]["created_and_deleted"] = rp.terminate(obj["id"])
    r["create_request"] = {"tried": tried,
                           "ok": any(v["code"] == 402 for v in tried.values()) and
                           not any(v["code"] in (400, 422) and "valid" in v["detail"].lower() for v in tried.values())}
    # 5. what the API answers for a pod that is gone (the backend treats 404 as gone)
    code, _ = rp.api("GET", "/pods/doesnotexist0")
    code2, _ = rp.api("DELETE", "/pods/doesnotexist0")
    r["gone_pod"] = {"get": code, "delete": code2, "ok": code in (404, 400) and code2 in (404, 400, 204)}
    code, obj = rp.api("GET", "/pods")
    r["list_shape"] = {"keys": sorted(obj.keys())[:5], "ok": "pods" in obj}
    r["ok"] = all(v.get("ok") for v in r.values() if isinstance(v, dict))
    return r
