"""Cut a chapter with ffmpeg: a quick animatic, or the 4K picture from upscaled sources.

For each shot, in order: the passing take if there is one, else the selected keyframe
held for the shot's length, else a plain card. So the cut exists from day one and fills
in as shots are approved. With --4k, the 3840x2160 versions are used where they exist.
Stills are held frames here; their slow pushes and particles are added in the editor.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .ledger import Ledger
from .project import Project, job_key

FPS = "24"
CARD = {"Still": "0x4a5568", "Simple": "0x2b6cb0", "Hero": "0x6b46c1"}


def _ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def _norm(w: int, h: int, crf: int, hold: float = 0) -> list[str]:
    # 16 fps hero clips land on the 24 fps timeline by frame repeats, the way anime is shot on 2s and 3s.
    # `hold` freezes the last frame if a clip is a fraction shorter than its shot.
    pad = f"tpad=stop_mode=clone:stop_duration={hold}," if hold else ""
    return ["-vf", f"{pad}scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-r", FPS, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]


def _four_k(p: Project, led: Ledger, shot_id: str, source_key: str) -> str | None:
    for j in led.jobs(shot_id, "upscale"):
        if j["status"] == "done" and j["path"] and p.fetch(j["path"]).exists():
            for attempt in range(10):
                if j["key"] == job_key(shot_id, "upscale", j["idx"], source_key) and j["idx"] % 10 == attempt:
                    return str(p.fetch(j["path"]))
    return None


def assemble(p: Project, led: Ledger, chapter: int, four_k: bool = False) -> dict:
    w, h, crf = (3840, 2160, 14) if four_k else (640, 360, 23)
    tag = "4k" if four_k else "animatic"
    work = p.media / "animatic" / f"chapter_{chapter}_{tag}_parts"
    work.mkdir(parents=True, exist_ok=True)
    used = {"take": 0, "keyframe": 0, "card": 0, "4k": 0}
    parts = []
    for n, s in enumerate(p.chapter(chapter)):
        seg = work / f"{n:03d}_{s.id}.mp4"
        passed = [j for j in led.jobs(s.id, "take") if j["status"] == "done" and j["verdict"] == "pass"]
        sel = led.selected(s.id)
        src, kind = None, "card"
        if passed and p.fetch(passed[-1]["path"]).exists():
            src, kind = str(p.fetch(passed[-1]["path"])), "take"
            up = _four_k(p, led, s.id, passed[-1]["key"]) if four_k else None
        elif sel and p.fetch(sel["path"]).exists():
            src, kind = str(p.fetch(sel["path"])), "keyframe"
            up = _four_k(p, led, s.id, sel["key"]) if four_k else None
        else:
            up = None
        if up:
            src = up
            used["4k"] += 1
        if kind == "take":
            _ff(["-i", src, "-t", str(s.seconds), *_norm(w, h, crf, hold=s.seconds), str(seg)])
        elif kind == "keyframe":
            _ff(["-loop", "1", "-i", src, "-t", str(s.seconds), *_norm(w, h, crf), str(seg)])
        else:
            _ff(["-f", "lavfi", "-i", f"color=c={CARD.get(s.type, 'gray')}:s={w}x{h}:r={FPS}",
                 "-t", str(s.seconds), *_norm(w, h, crf), str(seg)])
        used[kind] += 1
        parts.append(seg)
    listing = work / "list.txt"
    listing.write_text("".join(f"file '{x.name}'\n" for x in parts), encoding="utf-8")
    out = p.media / "animatic" / f"chapter_{chapter}{'_4k' if four_k else ''}.mp4"
    _ff(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])
    return {"file": str(out), "size": f"{w}x{h}", "shots": len(parts), "from": used,
            "seconds": sum(s.seconds for s in p.chapter(chapter))}
