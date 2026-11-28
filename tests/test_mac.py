"""Mac backend: a batch waits in the release, the Mac worker takes it, the producer collects it."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from engine import kaggle as kg
from engine import mac_backend as mb
from engine import steps

from test_pipeline import copy_project


class FakeRelease:
    """The `batches` release in a folder: gh upload / delete-asset / api, downloads as file:// URLs."""

    def __init__(self, root: Path):
        self.dir = root
        self.dir.mkdir(parents=True, exist_ok=True)
        self.created: dict[str, str] = {}

    def gh(self, args):
        if args[:2] == ["release", "upload"]:
            src = Path(args[3])
            shutil.copyfile(src, self.dir / src.name)
            self.created.setdefault(src.name, datetime.now(timezone.utc).isoformat())
            return 0, ""
        if args[:2] == ["release", "delete-asset"]:
            (self.dir / args[3]).unlink(missing_ok=True)
            self.created.pop(args[3], None)
            return 0, ""
        if args[0] == "api":
            assets = [{"name": f.name, "browser_download_url": f.as_uri(), "created_at": self.created[f.name],
                       "updated_at": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat()}
                      for f in sorted(self.dir.iterdir())]
            return 0, json.dumps({"assets": assets})
        return 0, ""

    def names(self):
        return sorted(f.name for f in self.dir.iterdir())


def _setup(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    proj = copy_project(d, cfg_edit=lambda c: c.update(mac={"enabled": True}, kaggle={**c["kaggle"], "stages": ["video"]}))
    for f in (proj.root / "style_bible" / "refs").glob("*.png"):
        f.unlink()                                       # chapter 1 needs its reference stills
    fr = FakeRelease(tmp_path / "release")
    mac = mb.MacBackend(proj, "o/r", gh=fr.gh)
    return proj, steps.open_ledger(proj), fr, mac


def test_batch_goes_to_the_mac_and_comes_back(tmp_path):
    proj, led, fr, mac = _setup(tmp_path)
    r = kg.tick(proj, led, mac, 1, engine_set="stub")
    assert r["action"] == "built_and_submitted" and r["stage"] == "refs", r
    b = led.active_batch("mac")
    assert fr.names() == [f"mac-{b['id']}.tar.gz"]
    assert kg.tick(proj, led, mac, 1, engine_set="stub")["status"] == "queued"

    worker_rel = mb.Release("o/r", gh=fr.gh)
    cfg = mb.settings(proj)
    assert mb.worker_loop(worker_rel, tmp_path / "mac", mb.StubEngine(), cfg, once=True) == 1
    assert f"mac-{b['id']}.results.tar.gz" in fr.names() and f"mac-{b['id']}.claim.json" in fr.names()

    r = kg.tick(proj, led, mac, 1, engine_set="stub")
    assert r["action"] == "collected" and r["made"] > 0 and r["failed"] == 0, r
    assert fr.names() == []                               # every file of the batch is cleaned up


def test_a_silent_mac_or_no_mac_frees_the_jobs(tmp_path):
    proj, led, fr, mac = _setup(tmp_path)
    kg.tick(proj, led, mac, 1, engine_set="stub")
    b = led.active_batch("mac")
    later = mb.MacBackend(proj, "o/r", gh=fr.gh, now=lambda: time.time() + 80 * 3600)
    assert later.status("", b["id"]) == "error" and "no mac worker took" in later.log[-1]
    claim = tmp_path / f"mac-{b['id']}.claim.json"
    claim.write_text("{}")
    fr.gh(["release", "upload", "batches", str(claim)])
    soon = mb.MacBackend(proj, "o/r", gh=fr.gh, now=lambda: time.time() + 3600)
    assert soon.status("", b["id"]) == "error" and "went quiet" in soon.log[-1]
    assert mac.status("", b["id"]) == "running"


def test_kaggle_leaves_refs_to_the_mac(tmp_path):
    proj, led, fr, mac = _setup(tmp_path)
    k = kg.LocalKaggle(proj.media)
    k.stages = ("video",)
    out = kg.tick_all(proj, led, [k, mac], 1, engine_set="stub")
    assert out[0]["action"] == "idle" and "only takes video" in out[0]["reason"]
    assert out[1]["action"] == "built_and_submitted" and out[1]["backend"] == "mac"


def test_self_test_writes_one_still(tmp_path):
    proj, led, fr, mac = _setup(tmp_path)
    r = mb.self_test(proj, mb.StubEngine(), tmp_path / "s.png")
    assert Path(r["still"]["file"]).exists() and Path(r["edit"]["file"]).exists()


def _fake_repo(tmp_path, folder="q4", parts=("transformer", "text_encoder", "vae")):
    files = [(f"{folder}/{p}/w.safetensors", 10) for p in parts] + [(f"{folder}/tokenizer/vocab.json", 1),
                                                                    ("other/x.safetensors", 5)]
    calls = []

    def download(repo, local_dir, allow_patterns):
        calls.append((repo, sorted(allow_patterns)))
        for n in allow_patterns:
            f = Path(local_dir) / n
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"w")
    return files, calls, download


def test_model_folder_is_downloaded_once_and_put_in_the_command(tmp_path):
    files, calls, download = _fake_repo(tmp_path)
    e = {"cmd": ["mflux-generate-qwen-edit", "--model", "{model_dir}"], "download": {"repo": "a/b", "folder": "q4"}}
    cmd = mb.resolve_cmd(e, tmp_path, download, lambda repo: files)
    assert cmd[-1] == str(tmp_path / "a--b" / "q4")
    assert calls == [("a/b", ["q4/text_encoder/w.safetensors", "q4/tokenizer/vocab.json",
                              "q4/transformer/w.safetensors", "q4/vae/w.safetensors"])]
    mb.resolve_cmd(e, tmp_path, download, lambda repo: files)
    assert len(calls) == 1                                # not downloaded twice


def test_an_empty_model_download_is_an_error_not_complete(tmp_path):
    files, calls, _ = _fake_repo(tmp_path)
    e = {"cmd": ["x", "--model", "{model_dir}"], "download": {"repo": "a/b", "folder": "q4"}}
    try:
        mb.resolve_cmd(e, tmp_path, lambda repo, local_dir, allow_patterns: None, lambda repo: files)
        raise AssertionError("should have failed")
    except RuntimeError as err:
        assert "has no transformer" in str(err)
    assert not (tmp_path / "a--b" / "q4" / ".complete").exists()


def test_a_missing_folder_names_what_the_repo_has(tmp_path):
    files, _, download = _fake_repo(tmp_path)
    e = {"cmd": ["x", "--model", "{model_dir}"], "download": {"repo": "a/b", "folder": "gone"}}
    try:
        mb.resolve_cmd(e, tmp_path, download, lambda repo: files)
        raise AssertionError("should have failed")
    except RuntimeError as err:
        assert "no folder 'gone'" in str(err) and "q4" in str(err)


def test_a_folder_marked_complete_but_empty_is_fetched_again(tmp_path):
    files, calls, download = _fake_repo(tmp_path)
    t = tmp_path / "a--b" / "q4"
    t.mkdir(parents=True)
    (t / ".complete").write_text("ok")                    # what the old version left on the Mac
    e = {"cmd": ["x", "--model", "{model_dir}"], "download": {"repo": "a/b", "folder": "q4"}}
    mb.resolve_cmd(e, tmp_path, download, lambda repo: files)
    assert len(calls) == 1 and (t / "vae" / "w.safetensors").exists()


def test_a_missing_tokenizer_is_taken_from_the_base_model(tmp_path):
    files = [(f"q4/{p}/w.safetensors", 10) for p in ("transformer", "text_encoder", "vae")]
    calls = []

    def download(repo, local_dir, allow_patterns):
        calls.append(repo)
        for n in allow_patterns:
            if repo == "base/model" and n.startswith("tokenizer/"):
                n = "tokenizer/vocab.json"
            elif repo == "base/model":
                continue
            f = Path(local_dir) / n
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"w")
    e = {"cmd": ["x", "--model", "{model_dir}"],
         "download": {"repo": "a/b", "folder": "q4", "tokenizer_from": "base/model"}}
    mb.resolve_cmd(e, tmp_path, download, lambda repo: files)
    assert calls == ["a/b", "base/model"]
    assert (tmp_path / "a--b" / "q4" / "tokenizer" / "vocab.json").exists()
