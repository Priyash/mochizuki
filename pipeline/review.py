"""The hand-off between the producer (GitHub Actions) and the reviewer (a Claude session).

The producer exports small review images and a queue.json to the `state` branch. The
reviewer looks at them, writes its verdicts to reviews/<anything>.json on that branch,
and the producer applies them on its next run. A review file may only judge and select:

    {"verdicts": [{"key": "...", "verdict": "pass" | "fail", "note": "one line"}],
     "select":   ["<keyframe key>", ...]}

Approvals are not accepted here. Gates are approved by the repository owner, in the issue.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from . import steps
from .ledger import Ledger
from .project import Project


def _ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def _still(src: Path, out: Path, width: int = 960) -> None:
    _ff(["-i", str(src), "-vf", f"scale={width}:-2", "-frames:v", "1", "-q:v", "4", str(out)])


def _strip(src: Path, seconds: float, out: Path) -> None:
    """Six frames of a take, evenly spaced, tiled 3 by 2: the reviewer sees the motion as stills."""
    fps = 6 / max(seconds, 1)
    _ff(["-i", str(src), "-vf", f"fps={fps:.4f},scale=480:-2,tile=3x2", "-frames:v", "1", "-q:v", "4", str(out)])


def export(p: Project, led: Ledger, chapter: int, outdir: Path) -> dict:
    """Write review images and queue.json for everything a reviewer has to decide."""
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    items = []
    for s in p.chapter(chapter):
        st = steps.state(p, led, s)
        base = {"shot_id": s.id, "type": s.type, "seconds": s.seconds, "shot": s.shot, "sound": s.sound,
                "refs": [p.rel(r) for r in p.refs_for(s, 3)]}
        if st == "keyframes_review":
            cands = []
            for j in led.jobs(s.id, "keyframe"):
                if j["status"] != "done" or j["verdict"] == "fail":
                    continue
                img = f"{j['key']}.jpg"
                _still(p.fetch(j["path"]), outdir / img)
                cands.append({"key": j["key"], "image": img, "verdict": j["verdict"]})
            items.append({**base, "what": "keyframe", "candidates": cands,
                          "task": "select the best candidate, or fail every candidate that is not good enough"})
        elif st == "take_review":
            for j in led.jobs(s.id, "take"):
                if j["status"] == "done" and j["verdict"] is None:
                    img = f"{j['key']}.jpg"
                    _strip(p.fetch(j["path"]), s.seconds, outdir / img)
                    items.append({**base, "what": "take", "key": j["key"], "image": img,
                                  "attempt": j["idx"] + 1, "cap": s.takes,
                                  "task": "pass or fail this take (six frames, left to right, top to bottom)"})
    (outdir / "queue.json").write_text(json.dumps({"chapter": chapter, "items": items}, indent=1,
                                                  ensure_ascii=False), encoding="utf-8")
    return {"items": len(items)}


def board_sheet(p: Project, led: Ledger, chapter: int, out: Path) -> list[str]:
    """The chosen keyframe of every shot, in order, five to a row. Returns the shot ids in order."""
    work = out.parent / "_board"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    ids = []
    for n, s in enumerate(p.chapter(chapter)):
        sel = led.selected(s.id)
        if sel:
            _still(p.fetch(sel["path"]), work / f"{n:03d}.jpg", width=384)
            ids.append(s.id)
    rows = (len(ids) + 4) // 5
    _ff(["-framerate", "1", "-i", str(work / "%03d.jpg"), "-vf",
         f"scale=384:216:force_original_aspect_ratio=decrease,pad=384:216:(ow-iw)/2:(oh-ih)/2,tile=5x{rows}:padding=4",
         "-frames:v", "1", "-q:v", "4", str(out)])
    shutil.rmtree(work, ignore_errors=True)
    return ids


def apply(led: Ledger, folder: Path) -> dict:
    """Apply reviewer files, then delete them. Anything but verdicts and selections is refused."""
    done, errors, refused = 0, [], []
    for f in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except ValueError as e:
            errors.append(f"{f.name}: not JSON ({e})")
            f.unlink()
            continue
        for k in data:
            if k not in ("verdicts", "select", "note"):
                refused.append(f"{f.name}: '{k}' is not something a reviewer can do")
        for v in data.get("verdicts", []):
            try:
                led.verdict(v["key"], v["verdict"], "agent", str(v.get("note", ""))[:300])
                done += 1
            except (KeyError, ValueError) as e:
                errors.append(f"{f.name}: {v.get('key')}: {e}")
        for key in data.get("select", []):
            try:
                led.select(key)
                done += 1
            except ValueError as e:
                errors.append(f"{f.name}: select {key}: {e}")
        f.unlink()
    return {"applied": done, "errors": errors, "refused": refused}
