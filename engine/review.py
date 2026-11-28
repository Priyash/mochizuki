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
from .project import Project, ref_shot


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
    for r in p.refs_missing(chapter):
        if steps.ref_state(p, led, r) != "review":
            continue
        cands = []
        for j in led.jobs(ref_shot(r["name"]).id, "ref"):
            if j["status"] == "done" and j["verdict"] != "fail":
                img = f"{j['key']}.jpg"
                _still(p.fetch(j["path"]), outdir / img, width=1280)
                cands.append({"key": j["key"], "image": img, "verdict": j["verdict"]})
        items.append({"what": "ref", "name": r["name"], "prompt": r["prompt"], "candidates": cands,
                      "task": "reference still for gate 1: select the most striking, on-style, emotionally strong "
                              "candidate, or fail every candidate that is not good enough"})
    shots = p.chapter(chapter)
    for n, s in enumerate(shots):
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
            src = p.continuity_source(s)
            drawn_from = None
            if src and led.selected(src):
                drawn_from = {"shot_id": src, "image": f"source_{src}.jpg"}
                if not (outdir / drawn_from["image"]).exists():
                    _still(p.fetch(led.selected(src)["path"]), outdir / drawn_from["image"], width=640)
            neighbours = []                       # chosen keyframes of the shots either side, for continuity
            for m in (n - 1, n + 1):
                if 0 <= m < len(shots) and led.selected(shots[m].id):
                    img = f"neighbour_{shots[m].id}.jpg"
                    if not (outdir / img).exists():
                        _still(p.fetch(led.selected(shots[m].id)["path"]), outdir / img, width=640)
                    neighbours.append({"shot_id": shots[m].id, "shot": shots[m].shot, "image": img})
            items.append({**base, "what": "keyframe", "candidates": cands, "neighbours": neighbours,
                          "drawn_from": drawn_from, "scene": p.scene_of(s)[0],
                          "task": "select the best candidate, or fail every candidate that is not good enough. "
                                  "It must match the neighbouring shots' chosen keyframes in costume, set, light and "
                                  "time of day unless the shot description says that changes, and must stay true to "
                                  "the keyframe it was drawn from (its scene anchor)"})
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


def _sheet(paths: list[Path], out: Path, cols: int, w: int, h: int) -> None:
    """Tile `paths` into one picture, `cols` to a row. Every tile is made exactly w x h first: when the
    size changes partway through a sequence (a square still among wide ones), ffmpeg restarts its filters
    and the tile drops everything before it, which left the 28 Nov reference sheet with two pictures."""
    work = out.parent / "_sheet"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for n, f in enumerate(paths):
        _ff(["-i", str(f), "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
             f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2", "-frames:v", "1", "-q:v", "3", str(work / f"{n:03d}.jpg")])
    rows = (len(paths) + cols - 1) // cols
    _ff(["-framerate", "1", "-i", str(work / "%03d.jpg"), "-vf", f"tile={cols}x{rows}:padding=4",
         "-frames:v", "1", "-q:v", "3", str(out)])
    shutil.rmtree(work, ignore_errors=True)


def board_sheet(p: Project, led: Ledger, chapter: int, out: Path) -> list[str]:
    """The chosen keyframe of every shot, in order, five to a row. Returns the shot ids in order."""
    picked = [(s.id, led.selected(s.id)) for s in p.chapter(chapter)]
    picked = [(i, sel) for i, sel in picked if sel]
    _sheet([p.fetch(sel["path"]) for _, sel in picked], out, 5, 384, 216)
    return [i for i, _ in picked]


def refs_sheet(p: Project, led: Ledger, out: Path, chapter: int | None = None) -> list[str]:
    """The chosen reference stills, two to a row, large enough to judge. Returns their names in order."""
    picked = [(r["name"], led.selected(ref_shot(r["name"]).id)) for r in p.refs_missing(chapter)]
    picked = [(n, sel) for n, sel in picked if sel]
    _sheet([p.fetch(sel["path"]) for _, sel in picked], out, 2, 960, 540)
    return [n for n, _ in picked]


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
