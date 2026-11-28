"""Cut a chapter with ffmpeg: a quick animatic, or the 4K picture from upscaled sources.

For each shot, in order: its 3D render when config cgi.use_in_cut is on (engine/cgi), else
the passing take if there is one, else the selected keyframe held for the shot's length, else a
plain card. So the cut exists from day one and fills
in as shots are approved. With --4k, the 3840x2160 versions are used where they exist.
Stills are held frames here; their slow pushes and particles are added in the editor.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .cgi import config as cgi_config
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


CUTS_DIR = "animatic"       # media/animatic/: every chapter cut, the animatic and the 4K
ISO_639_2 = {"en": "eng", "ja": "jpn", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por",
             "zh": "zho", "ko": "kor", "hi": "hin"}     # an audio track's language tag (players show it)


def cut_path(p: Project, chapter: int, four_k: bool = False, language: str | None = None) -> Path:
    """Where a chapter's cut is written: chapter_N.mp4 (the animatic), chapter_N_4k.mp4 (the 4K picture);
    a dub adds its language (chapter_N.ja.mp4), and its subtitles are chapter_N.ja.srt beside it."""
    return p.media / CUTS_DIR / f"chapter_{chapter}{'_4k' if four_k else ''}{'.' + language if language else ''}.mp4"


def finished_clip(p: Project, led: Ledger, shot_id: str) -> str | None:
    """The shot's AI finish, when the film's config puts finishes in the cut (engines.ai_finish.use_in_cut):
    its newest finished job that no one has failed."""
    if not p.engine("ai_finish").get("use_in_cut"):
        return None
    done = [j for j in led.jobs(shot_id, "finish") if j["status"] == "done" and j["verdict"] != "fail"]
    return str(p.fetch(done[-1]["path"])) if done and p.fetch(done[-1]["path"]).exists() else None


def cgi_clip(p: Project, shot_id: str, four_k: bool, language: str | None = None) -> str | None:
    """The shot's 3D render (engine/cgi), when config cgi.use_in_cut is on: the final render for the
    4K cut; the preview, else the final, for the animatic. With `language`, only the shot's render
    in that language (a dub, 1-11.ja.mp4: made where someone speaks on screen)."""
    cfg = cgi_config.settings(p)
    if not cfg.get("use_in_cut"):
        return None
    folder = cgi_config.work_dir(p, cfg, shot_id)
    names = [f"{shot_id}.mp4", f"{shot_id}.mov"] if four_k else [f"{shot_id}-preview.mp4", f"{shot_id}.mp4"]
    if language:
        names = [n.replace(".", f".{language}.", 1) for n in names]
    for name in names:
        if (folder / name).exists():
            return str(folder / name)
    return None


def assemble(p: Project, led: Ledger, chapter: int, four_k: bool = False, language: str | None = None) -> dict:
    """Cut a chapter. With `language`, a dub: each speaking shot in that language's render, under
    that language's dialogue alone (chapter_N.<language>.mp4). A speaking shot with no render in it
    keeps the primary picture, and is listed in "undubbed" for the reviewer: its lips do not match."""
    w, h, crf = (3840, 2160, 14) if four_k else (640, 360, 23)
    tag = "4k" if four_k else "animatic"
    work = p.media / CUTS_DIR / f"chapter_{chapter}_{tag}{'_' + language if language else ''}_parts"
    work.mkdir(parents=True, exist_ok=True)
    used = {"finish": 0, "cgi": 0, "take": 0, "keyframe": 0, "card": 0, "4k": 0}
    parts = []
    undubbed = []
    for n, s in enumerate(p.chapter(chapter)):
        seg = work / f"{n:03d}_{s.id}.mp4"
        passed = [j for j in led.jobs(s.id, "take") if j["status"] == "done" and j["verdict"] == "pass"]
        sel = led.selected(s.id)
        src, kind = None, "card"
        dub = cgi_clip(p, s.id, four_k, language) if language else None
        fin = None if dub else finished_clip(p, led, s.id)      # a finish has the primary language's lips
        cg = dub or cgi_clip(p, s.id, four_k)
        if language and not dub and _speaks_in(p, s.id, language):
            undubbed.append(s.id)
        if fin:
            src, kind, up = fin, "finish", None
        elif cg:
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
        if kind in ("take", "cgi", "finish") and src is not None:
            _ff(["-i", src, "-t", str(s.seconds), *_norm(w, h, crf, hold=s.seconds), str(seg)])
        elif kind == "keyframe" and src is not None:
            _ff(["-loop", "1", "-i", src, "-t", str(s.seconds), *_norm(w, h, crf), str(seg)])
        else:
            _ff(["-f", "lavfi", "-i", f"color=c={CARD.get(s.type, 'gray')}:s={w}x{h}:r={FPS}",
                 "-t", str(s.seconds), *_norm(w, h, crf), str(seg)])
        used[kind] += 1
        parts.append(seg)
    listing = work / "list.txt"
    listing.write_text("".join(f"file '{x.name}'\n" for x in parts), encoding="utf-8")
    out = cut_path(p, chapter, four_k, language)
    _ff(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])
    extra = dialogue_tracks(p, chapter, out, language) if used["cgi"] else {}
    return {"file": str(out), "size": f"{w}x{h}", "shots": len(parts), "from": used,
            "seconds": sum(s.seconds for s in p.chapter(chapter)), **({"undubbed": undubbed} if language else {}),
            **extra}


def _speaks_in(p: Project, shot_id: str, language: str) -> bool:
    """Does someone speak `language` on screen in this shot (so its dub needs its own lips)?"""
    from .audio import dialogue  # noqa: PLC0415 - only when the film has dialogue
    return language in dialogue.dubbed_languages(dialogue.shot_manifest(p, shot_id))


def dialogue_tracks(p: Project, chapter: int, video: Path, language: str | None = None) -> dict:
    """The automated dialogue under the cut: the primary language as the main audio, the others as extra
    tracks (YouTube can carry several), plus an .srt per language next to the file. Under a dub, only
    its own language (its lips match no other); its .srt is the one the main cut writes."""
    from .audio import dialogue, mix, subtitles  # noqa: PLC0415 - only when the film has dialogue
    script = dialogue.load(p)
    langs = script.get("languages", ["en"])
    primary = script.get("primary", langs[0])
    order = [language] if language else [primary] + [x for x in langs if x != primary]
    tracks = [(x, mix.chapter_dialogue(p, chapter, x)) for x in order]
    tracks = [(x, t) for x, t in tracks if t]
    if not tracks:
        return {}
    args = ["-i", str(video)]
    for _, t in tracks:
        args += ["-i", str(t)]
    maps = ["-map", "0:v"] + [m for i in range(len(tracks)) for m in ("-map", f"{i + 1}:a")]
    meta = [
        m
        for i, (x, _) in enumerate(tracks)
        for m in (f"-metadata:s:a:{i}", f"language={ISO_639_2.get(x, x)}")
    ]
    tmp = video.with_suffix(".tmp.mp4")
    _ff([*args, *maps, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", *meta, "-shortest", str(tmp)])
    tmp.replace(video)
    subs: dict[str, str] = {}
    for x in [] if language else order:
        srt = video.with_suffix(f".{x}.srt")
        subtitles.write_srt(p, chapter, x).replace(srt)
        subs[x] = str(srt)
    return {"audio": [x for x, _ in tracks], "subtitles": subs}
