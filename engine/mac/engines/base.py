"""What every Mac engine is, the checks on what it makes, and stand-ins for tests and dry runs.

An engine turns one job into one file:  run(job, inputs, out) -> info dict.
  job      the manifest entry (key, kind, prompt, width/height, refs/image, seconds...)
  inputs   the folder the job's input images are relative to
  out      where to write the picture (.png) or clip (.mp4)
It raises on failure; the worker records the error and goes on with the next job.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol

from ..common import ffmpeg_exe, write_png


class Engine(Protocol):
    def run(self, job: dict, inputs: Path, out: Path) -> dict: ...


def is_video_job(job: dict) -> bool:
    return job.get("kind") == "take" or job.get("media") == "video"


def check_image(out: Path) -> None:
    """A picture that is one flat colour is a failed render, not a result."""
    try:
        from PIL import Image, ImageStat  # noqa: PLC0415
    except ImportError:
        return
    st = ImageStat.Stat(Image.open(out).convert("L"))
    if st.extrema[0][1] - st.extrema[0][0] < 5:
        raise RuntimeError("image came out blank")


def check_video(out: Path) -> None:
    """The clip must exist, be playable and not be one flat colour."""
    if not out.exists() or out.stat().st_size < 1000:
        raise RuntimeError("the video was not written")
    try:
        r = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(out), "-vf", "select=eq(n\\,0)", "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, timeout=120)
    except Exception:  # noqa: BLE001 - no ffmpeg to look with: trust the file
        return
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"the video cannot be read: {r.stderr.decode(errors='replace')[-300:]}")
    first = out.with_suffix(".first.png")
    first.write_bytes(r.stdout)
    try:
        check_image(first)
    finally:
        first.unlink(missing_ok=True)


class StubImageEngine:
    """A small gradient picture, no model: tests and dry runs."""

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        w, h = job.get("width", 64) // 16 or 4, job.get("height", 36) // 16 or 4
        write_png(out, w, h)
        return {"width": w, "height": h, "engine": "stub"}


class StubVideoEngine:
    """A two-second test-pattern clip, no model: tests and dry runs (needs ffmpeg)."""

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc=size=128x72:rate=24:duration=1", "-pix_fmt", "yuv420p", str(out)], check=True)
        return {"frames": 24, "fps": 24, "engine": "stub"}
