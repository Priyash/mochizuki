"""Run Blender on a shot's plan, check what it made, and encode the shot.

    render_shot(p, "1-02", preview=True)  ->  {"clip": ".../1-02-preview.mp4", "frames": 96, ...}

Blender is found in config cgi.blender, else the BLENDER environment variable, else the Mac's
/Applications/Blender.app, else `blender` on PATH. cgi.blender may also name a Python that has
the bpy module (pip install bpy), which runs build.py directly.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ...log import log as _log
from .. import plan as planner
from ..config import settings

BUILD = Path(__file__).parent / "blender" / "build.py"
MAC_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def log(event: str, msg: str = "", level: str = "info", **kw) -> None:
    _log("cgi", event, msg, level, **kw)


def find_blender(cfg: dict) -> str | None:
    for cand in (cfg.get("blender"), os.environ.get("BLENDER"), MAC_BLENDER, shutil.which("blender")):
        if cand and Path(cand).exists():
            return str(cand)
    return None


def command(exe: str, plan_file: Path, no_render: bool = False) -> list[str]:
    """How to run build.py: through Blender's own Python, or a Python with bpy."""
    extra = ["--no-render"] if no_render else []
    if "python" in Path(exe).name.lower():
        return [exe, str(BUILD), "--plan", str(plan_file), *extra]
    return [exe, "-b", "--python-exit-code", "1", "--python", str(BUILD), "--", "--plan", str(plan_file), *extra]


def run_blender(cmd: list[str], shot: str, timeout: float) -> list[dict]:
    """Stream Blender's output: its CGI lines become log events; the rest is kept for errors."""
    events: list[dict] = []
    tail: list[str] = []
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last_note = 0.0
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith("CGI "):
                try:
                    ev = json.loads(line[4:])
                except json.JSONDecodeError:
                    continue
                events.append(ev)
                if ev["event"] == "render.frame":
                    if time.time() - last_note > 10 or ev["frame"] == ev["of"]:
                        log("render.frame", f"{shot}: frame {ev['frame']}/{ev['of']}", shot=shot,
                            frame=ev["frame"], of=ev["of"], seconds=ev.get("seconds"))
                        last_note = time.time()
                else:
                    log(ev["event"], ev.get("msg", ""), shot=shot,
                        **{k: v for k, v in ev.items() if k not in ("event", "msg")})
            elif line.startswith("cgi.blender:"):
                log("blender.note", line[len("cgi.blender:"):].strip(), level="warn", shot=shot)
            else:
                tail = (tail + [line])[-40:]
            if time.time() - t0 > timeout:
                proc.kill()
                raise RuntimeError(f"{shot}: Blender took more than {timeout:.0f} s; stopped")
    finally:
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{shot}: Blender exited with {code}:\n" + "\n".join(tail[-15:]))
    return events


def check_frames(folder: Path, first: int, last: int) -> list[Path]:
    """Every frame was written, and none is a blank, one-colour picture (unless the shot is black)."""
    files = [folder / f"{f:04d}.png" for f in range(first, last + 1)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} frames missing in {folder} (first: {missing[0]})")
    return files


_warned_no_pillow = False


def blank(png: Path) -> bool:
    """One flat colour: the lights or the camera are wrong. Needs Pillow; without it the check is off,
    and says so once."""
    global _warned_no_pillow
    try:
        from PIL import Image, ImageStat  # noqa: PLC0415
    except ImportError:
        if not _warned_no_pillow:
            _warned_no_pillow = True
            log("check.off", "the blank-frame check is off: Pillow is not installed (pip install pillow)", level="warn")
        return False
    st = ImageStat.Stat(Image.open(png).convert("L"))
    return st.extrema[0][1] - st.extrema[0][0] < 3


def encode(folder: Path, first: int, fps: int, out: Path, crf: int, codec: str = "libx264") -> Path:
    args = ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-start_number", str(first),
            "-i", str(folder / "%04d.png")]
    if codec == "prores":
        args += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
        out = out.with_suffix(".mov")
    else:
        args += ["-c:v", codec, "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    subprocess.run([*args, str(out)], check=True)
    return out


def mux_audio(clip: Path, wav: Path, offset_s: float = 0.0) -> Path:
    """Put the shot's dialogue track under its picture (AAC), starting where the frames start."""
    tmp = clip.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-ss", f"{offset_s:.3f}", "-i", str(wav),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    str(tmp)], check=True)
    tmp.replace(clip)
    return clip


def render_shot(p, shot_id: str, preview: bool = False, frames: tuple[int, int] | None = None,
                no_render: bool = False, codec: str | None = None, timeout: float = 6 * 3600,
                size: tuple[int, int] | None = None) -> dict:
    cfg = settings(p)
    exe = find_blender(cfg)
    if not exe:
        raise RuntimeError("Blender was not found: install Blender 5.x from blender.org (or set config "
                           "cgi.blender / the BLENDER environment variable)")
    t0 = time.time()
    plan = planner.build(p, shot_id, preview=preview, frames_only=frames, size=size)
    plan_file = planner.write(plan)
    who = {c["name"]: ("model" if c["model"] else ("lantern" if c.get("procedural") else "stand-in"))
           for c in plan["characters"]}
    log("plan.written", f"{shot_id}: {plan['frames']} frames at {plan['resolution'][0]}x{plan['resolution'][1]}",
        shot=shot_id, plan=str(plan_file), characters=json.dumps(who) if who else None,
        faces=json.dumps({c["name"]: c["face"].get("_source") for c in plan["characters"]}) or None)
    events = run_blender(command(exe, plan_file, no_render), shot_id, timeout)
    result = {"shot": shot_id, "plan": str(plan_file), "blend": plan["output"]["blend"],
              "characters": who, "events": len(events)}
    if no_render:
        return result
    first, last = plan["frame_range"]
    folder = Path(plan["output"]["frames"])
    files = check_frames(folder, first, last)
    if not plan.get("blackout") and blank(files[len(files) // 2]):
        raise RuntimeError(f"{shot_id}: the frames came out blank (one flat colour): check the lights in the .blend")
    enc = cfg["encode"]
    name = f"{shot_id}-preview.mp4" if preview else f"{shot_id}.mp4"
    if (first, last) != (1, plan["frames"]):               # part of a shot never replaces the whole shot's clip
        name = name.replace(".mp4", f"-f{first}-{last}.mp4")
    clip = encode(folder, first, plan["fps"], Path(plan["output"]["dir"]) / name,
                  enc["preview_crf"] if preview else enc["crf"], codec or enc.get("codec", "libx264"))
    track = (plan.get("audio") or {}).get("en") or next(iter((plan.get("audio") or {}).values()), None)
    if track and Path(track).exists() and clip.suffix == ".mp4":
        clip = mux_audio(clip, Path(track), (first - 1) / plan["fps"])
    secs = time.time() - t0
    per = [e.get("seconds", 0) for e in events if e.get("event") == "render.frame"]
    result.update({"clip": str(clip), "frames": last - first + 1, "seconds": round(secs, 1),
                   "seconds_per_frame": round(sum(per) / len(per), 2) if per else None})
    log("shot.done", f"{shot_id}: {clip.name} in {secs:.0f} s", shot=shot_id, clip=str(clip),
        frames=result["frames"], seconds_per_frame=result["seconds_per_frame"])
    return result


def board(p, shots: list[str], chapter: int, cols: int = 4, width: int = 480) -> Path:
    """A storyboard sheet: each shot's middle frame, in order, labelled with its id when ffmpeg can draw text."""
    cfg = settings(p)
    work = Path(cfg["work"]) if Path(cfg["work"]).is_absolute() else p.root / cfg["work"]
    tiles = []
    for sid in shots:
        folders = sorted((d for d in (work / sid).glob("frames-*") if any(d.glob("*.png"))),
                         key=lambda d: max(f.stat().st_mtime for f in d.glob("*.png")))
        if folders:                                        # the newest render; its middle frame
            pngs = sorted(folders[-1].glob("*.png"))
            tiles.append((sid, pngs[len(pngs) // 2]))
    if not tiles:
        raise RuntimeError("no frames to put on the board")
    out = work / (f"board-ch{chapter}.png" if chapter else "board.png")
    n = len(tiles)
    layout = "|".join(f"{'+'.join(['w0'] * (i % cols)) or 0}_{'+'.join(['h0'] * (i // cols)) or 0}" for i in range(n))

    def run(labels: bool) -> None:
        args, filters = ["ffmpeg", "-loglevel", "error", "-y"], []
        for i, (sid, f) in enumerate(tiles):
            args += ["-i", str(f)]
            text = (f",drawtext=text='{sid}':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5"
                    if labels else "")
            filters.append(f"[{i}]scale={width}:-2{text}[t{i}]")
        stack = (f"{''.join(f'[t{i}]' for i in range(n))}xstack=inputs={n}:layout={layout}:fill=black[out]"
                 if n > 1 else "[t0]copy[out]")
        subprocess.run([*args, "-filter_complex", ";".join([*filters, stack]), "-map", "[out]", "-frames:v", "1",
                        str(out)], check=True, capture_output=True)

    try:
        run(True)
    except subprocess.CalledProcessError:
        run(False)                                         # no font for drawtext here: an unlabelled board
    return out
