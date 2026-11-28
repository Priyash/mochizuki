"""Cut a chapter with ffmpeg: a quick animatic, or the 4K picture from upscaled sources.

For each shot, in order: its 3D render when config cgi.use_in_cut is on (pipeline/cgi), else
the passing take if there is one, else the selected keyframe held for the shot's length, else a
plain card. So the cut exists from day one and fills
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


def cgi_clip(p: Project, shot_id: str, four_k: bool) -> str | None:
    """The shot's 3D render (pipeline/cgi), when config cgi.use_in_cut is on: the final render for the
    4K cut; the preview, else the final, for the animatic."""
    cfg = p.config.get("cgi", {})
    if not cfg.get("use_in_cut"):
        return None
    work = Path(cfg.get("work", "media/cgi"))
    folder = (work if work.is_absolute() else p.root / work) / shot_id
    names = [f"{shot_id}.mp4", f"{shot_id}.mov"] if four_k else [f"{shot_id}-preview.mp4", f"{shot_id}.mp4"]
    for name in names:
        if (folder / name).exists():
            return str(folder / name)
    return None


def assemble(p: Project, led: Ledger, chapter: int, four_k: bool = False) -> dict:
    w, h, crf = (3840, 2160, 14) if four_k else (640, 360, 23)
    tag = "4k" if four_k else "animatic"
    work = p.media / "animatic" / f"chapter_{chapter}_{tag}_parts"
    work.mkdir(parents=True, exist_ok=True)
    used = {"cgi": 0, "take": 0, "keyframe": 0, "card": 0, "4k": 0}
    parts = []
    for n, s in enumerate(p.chapter(chapter)):
        seg = work / f"{n:03d}_{s.id}.mp4"
        passed = [j for j in led.jobs(s.id, "take") if j["status"] == "done" and j["verdict"] == "pass"]
        sel = led.selected(s.id)
        src, kind = None, "card"
        cg = cgi_clip(p, s.id, four_k)
        if cg:
            src, kind, up = cg, "cgi", None
        elif passed and p.fetch(passed[-1]["path"]).exists():
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
        if kind in ("take", "cgi"):
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
    extra = dialogue_tracks(p, chapter, out) if used["cgi"] else {}
    return {"file": str(out), "size": f"{w}x{h}", "shots": len(parts), "from": used,
            "seconds": sum(s.seconds for s in p.chapter(chapter)), **extra}


def dialogue_tracks(p: Project, chapter: int, video: Path) -> dict:
    """The automated dialogue under the cut: the primary language as the main audio, the others as extra
    tracks (YouTube can carry several), plus an .srt per language next to the file."""
    from .cgi import dialogue  # noqa: PLC0415 - only for the 3D pipeline
    script = dialogue.load(p)
    langs = script.get("languages", ["en"])
    primary = script.get("primary", langs[0])
    order = [primary] + [x for x in langs if x != primary]
    tracks = [(x, dialogue.chapter_audio(p, chapter, x)) for x in order]
    tracks = [(x, t) for x, t in tracks if t]
    if not tracks:
        return {}
    args = ["-i", str(video)]
    for _, t in tracks:
        args += ["-i", str(t)]
    maps = ["-map", "0:v"] + [m for i in range(len(tracks)) for m in ("-map", f"{i + 1}:a")]
    meta = [m for i, (x, _) in enumerate(tracks) for m in (f"-metadata:s:a:{i}", f"language={'jpn' if x == 'ja' else 'eng'}")]
    tmp = video.with_suffix(".tmp.mp4")
    _ff([*args, *maps, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", *meta, "-shortest", str(tmp)])
    tmp.replace(video)
    subs = {x: str(dialogue.subtitles(p, chapter, x)) for x in order}
    for x, f in subs.items():
        Path(f).replace(video.with_suffix(f".{x}.srt"))
        subs[x] = str(video.with_suffix(f".{x}.srt"))
    return {"audio": [x for x, _ in tracks], "subtitles": subs}
