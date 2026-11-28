"""`python -m engine cgi ...`: the 3D pipeline from the command line (docs/cgi-pipeline.md).

    cgi check                           what is installed, which character models exist
    cgi sheets --chapter 1              every shot's acting sheet in one line each, with problems
    cgi sheet 1-11                      one sheet in full (copy it into style_bible/acting.json to edit)
    cgi plan 1-02 [--preview]           write the shot's plan.json without rendering
    cgi render 1-02 [--preview]         build, save the .blend and render the shot
    cgi render --chapter 1 --preview    every shot of a chapter (the animatic)
    cgi capture 1-05 --who tsumugi take.csv     a face performance for a shot (Live Link Face CSV,
                                                a saved track, or a video read by MediaPipe)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..log import set_file
from . import plan as planner
from .acting import capture as cap
from .acting import direction as dirn
from .config import by_chapter, settings, work_dir
from .render import runner as rnd


def _size(text: str | None) -> tuple[int, int] | None:
    """A render size, "WxH" -> (W, H)."""
    if not text:
        return None
    w, h = (int(x) for x in text.lower().split("x"))
    return w, h


def add_arguments(a: argparse.ArgumentParser) -> None:
    sub = a.add_subparsers(dest="cgi_cmd", required=True)
    sub.add_parser("check", help="Blender, the VRM add-on, MediaPipe, ffmpeg and the character models")
    s = sub.add_parser("sheets", help="the acting sheets of a chapter, one line each")
    s.add_argument("--chapter", type=int, required=True)
    s = sub.add_parser("sheet", help="one shot's acting sheet in full")
    s.add_argument("shot")
    s = sub.add_parser("plan", help="write a shot's plan.json (no Blender needed)")
    s.add_argument("shot")
    s.add_argument("--preview", action="store_true")
    s = sub.add_parser("render", help="build and render a shot, or a whole chapter")
    s.add_argument("shot", nargs="?")
    s.add_argument("--chapter", type=int)
    s.add_argument("--preview", action="store_true", help="the preview size (config cgi.preview_resolution)")
    s.add_argument("--frames", default="", help="only these frames, e.g. 1-24")
    s.add_argument("--no-render", action="store_true", help="build and save the .blend only")
    s.add_argument("--prores", action="store_true", help="encode ProRes 422 HQ (.mov) for the edit")
    s.add_argument("--size", default="", help="render at this size instead, e.g. 640x360")
    s.add_argument("--guides", nargs="?", const="depth,lines", default="",
                   help="also render the AI finish's guide passes (depth, lines; default both)")
    s.add_argument("--board", action="store_true", help="one frame from the middle of each shot, tiled into "
                                                        "a storyboard sheet (with --chapter)")
    s = sub.add_parser("voices", help="voice the dialogue of a shot or a chapter (text to speech, lip sync, tracks)")
    s.add_argument("shot", nargs="?")
    s.add_argument("--chapter", type=int)
    s.add_argument("--engine", default="", help="override every voice's engine (stub: no model, for tests)")
    s.add_argument("--languages", default="", help="e.g. en or en,ja (default: dialogue.json languages)")
    s.add_argument("--python", default="", help="the Python that has kokoro installed (default: this one)")
    s = sub.add_parser("subtitles", help="write the chapter's .srt files from the voiced lines")
    s.add_argument("--chapter", type=int, required=True)
    s = sub.add_parser("auto", help="the whole chapter, unattended: voices, renders, board, cut with sound")
    s.add_argument("--chapter", type=int, required=True)
    s.add_argument("--final", action="store_true", help="final 4K frames instead of previews")
    s.add_argument("--shard", default="0/1", help="this machine's share of the shots, e.g. 2/8")
    s.add_argument("--stage", default="all", choices=["voices", "render", "finish", "all"])
    s.add_argument("--engine", default="", help="voice engine override (stub for a dry run)")
    s.add_argument("--size", default="", help="render size override, e.g. 640x360")
    s.add_argument("--force", action="store_true", help="re-render shots even if nothing changed")
    s.add_argument("--shots", default="", help="only these shots, e.g. 1-02,1-11")
    s.add_argument("--prune", action="store_true",
                   help="keep only the first, middle and last frame of each render (for runners and caches)")
    s = sub.add_parser("capture", help="a face performance for a shot")
    s.add_argument("shot")
    s.add_argument("source", help="Live Link Face .csv, a saved .json track, or a video")
    s.add_argument("--who", required=True, help="the character it is for (tsumugi, sayo...)")
    s.add_argument("--start", type=float, default=0.0, help="seconds into the recording where the shot begins")


def _frames(text: str) -> tuple[int, int] | None:
    if not text:
        return None
    a, _, b = text.partition("-")
    return int(a), int(b or a)


def check(p) -> int:
    cfg = settings(p)
    ok = True
    exe = rnd.find_blender(cfg)
    if exe:
        try:
            out = subprocess.run([exe, "--version"] if "python" not in Path(exe).name.lower()
                                 else [exe, "-c", "import bpy; print('Blender', bpy.app.version_string)"],
                                 capture_output=True, text=True, timeout=120).stdout.strip().splitlines()
            print(f"Blender     {out[0] if out else exe}  ({exe})")
        except Exception as e:  # noqa: BLE001
            print(f"Blender     found at {exe} but it did not start: {e}")
            ok = False
    else:
        print("Blender     not found. Install Blender 5.x from blender.org (free); then this finds it.")
        ok = False
    print("VRM add-on  install it inside Blender: Preferences > Get Extensions > search 'VRM' (only needed "
          "once you have VRoid models)")
    try:
        import mediapipe  # noqa: F401, PLC0415
        print("MediaPipe   installed (face capture from video)")
    except ImportError:
        print("MediaPipe   not installed: pip install mediapipe   (only for face capture from a video; "
              "Live Link Face CSVs need nothing)")
    print(f"ffmpeg      {'ok' if shutil.which('ffmpeg') else 'missing: brew install ffmpeg'}")
    print("\nCharacter models (a missing one renders as a stand-in maquette):")
    for name, spec in cfg["characters"].items():
        if spec.get("procedural"):
            print(f"  {name:8} procedural ({spec['procedural']})")
            continue
        for ch, path in sorted(spec.get("model", {}).items()):
            there = (p.root / path).exists()
            print(f"  {name:8} chapter {ch}: {path}  {'ok' if there else '(not made yet)'}")
    return 0 if ok else 3


def sheets(p, chapter: int) -> int:
    bad = 0
    acted = dirn.load_acting(p)
    for s in p.chapter(chapter):
        d = dirn.direction_for(p, s.id)
        c = d.camera
        probs = dirn.problems(d)
        bad += bool(probs)
        who = ",".join(d.characters) or "-"
        src = "sheet" if s.id in acted else "draft"
        print(f"{s.id:5} {s.type:6} {c.framing:6} {c.angle:13} {int(c.lens):3}mm {c.move:6} {who:16} "
              f"beats={len(d.beats)} {d.timing:6} {d.performance:10} [{src}/{d.status}]"
              + (f"  PROBLEM: {'; '.join(probs)}" if probs else ""))
    return 3 if bad else 0


def capture_cmd(p, args) -> int:
    cfg = settings(p)
    src = Path(args.source).expanduser()
    if not src.exists():
        print(f"no such file: {src}")
        return 2
    if src.suffix.lower() in (".csv", ".json"):
        tr = cap.read_any(src)
    else:
        from ..mac.common import MODELS  # noqa: PLC0415
        tr = cap.face_from_video(src, cap.ensure_face_model(MODELS))
    d = dirn.direction_for(p, args.shot)
    if args.who not in d.characters:
        print(f"{args.who} is not in shot {args.shot} ({d.characters}); add them to its sheet first")
        return 2
    tr = cap.trim(tr, args.start, d.seconds)
    if tr.seconds + 0.05 < d.seconds:
        print(f"warning: the take is {tr.seconds:.1f} s, the shot {d.seconds:.0f} s; the last pose will be held")
    out = cap.Track.save(tr, work_dir(p, cfg, args.shot) / f"face-{args.who}.json")
    print(json.dumps({"saved": str(out), "fps": tr.fps, "frames": len(tr), "seconds": round(tr.seconds, 2),
                      "channels": len(tr.names), "from": tr.meta.get("from")}, indent=1))
    return 0


def run(p, args) -> int:
    set_file(work_dir(p, settings(p), "logs") / "cgi.jsonl")
    cmd = args.cgi_cmd
    if cmd == "check":
        return check(p)
    if cmd == "sheets":
        return sheets(p, args.chapter)
    if cmd == "sheet":
        d = dirn.direction_for(p, args.shot)
        print(json.dumps(d.to_dict(), indent=1))
        probs = dirn.problems(d)
        if probs:
            print("PROBLEMS:\n  " + "\n  ".join(probs), file=sys.stderr)
        return 3 if probs else 0
    if cmd == "plan":
        f = planner.write(planner.build(p, args.shot, preview=args.preview))
        print(f)
        return 0
    if cmd == "capture":
        return capture_cmd(p, args)
    if cmd == "voices":
        from ..audio import dialogue  # noqa: PLC0415
        shots = [args.shot] if args.shot else [s.id for s in p.chapter(args.chapter)] if args.chapter else []
        if not shots:
            print("give a shot id, or --chapter N")
            return 2
        langs = [x for x in args.languages.split(",") if x] or None
        done = [dialogue.voice_shot(p, sid, langs, args.engine or None, args.python or None) for sid in shots]
        print(json.dumps([d for d in done if d.get("lines")], indent=1))
        return 0
    if cmd == "subtitles":
        from ..audio import dialogue, subtitles  # noqa: PLC0415
        print(json.dumps({x: str(subtitles.write_srt(p, args.chapter, x))
                          for x in dialogue.load(p).get("languages", ["en"])}, indent=1))
        return 0
    if cmd == "auto":
        from . import auto  # noqa: PLC0415
        size = _size(args.size)
        a, _, b = args.shard.partition("/")
        r = auto.run(p, args.chapter, stage=args.stage, final=args.final, shard=(int(a), int(b or 1)),
                     engine=args.engine or None, size=size, force=args.force,
                     only=[x.strip() for x in args.shots.split(",") if x.strip()] or None, prune=args.prune)
        print(json.dumps(r, indent=1, default=str))
        return 3 if r.get("failed") else 0
    if cmd == "render":
        shots = [args.shot] if args.shot else [s.id for s in p.chapter(args.chapter)] if args.chapter else []
        if not shots:
            print("give a shot id, or --chapter N")
            return 2
        size = _size(args.size)
        results, failed = [], []
        for sid in shots:
            frames = _frames(args.frames)
            if args.board:
                d = dirn.direction_for(p, sid)
                mid = max(1, int(round(d.seconds * settings(p)["fps"] / 2)))
                frames = (mid, mid)
            try:
                results.append(rnd.render_shot(p, sid, preview=args.preview or args.board, frames=frames,
                                               no_render=args.no_render, codec="prores" if args.prores else None,
                                               size=size, guides=tuple(g for g in args.guides.split(",") if g)))
            except Exception as e:  # noqa: BLE001 - one bad shot must not stop a chapter
                rnd.log("shot.failed", f"{sid}: {e}", level="error", shot=sid)
                failed.append(sid)
        out: dict[str, Any] = {"rendered": results, "failed": failed}
        if args.board and results:
            out["board"] = str(rnd.board(p, [r["shot"] for r in results], args.chapter or 0))
        print(json.dumps(out, indent=1, default=str))
        return 3 if failed else 0
    return 2


__all__ = ["add_arguments", "run", "by_chapter"]
