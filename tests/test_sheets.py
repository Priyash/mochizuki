"""Contact sheets for the owner and the reviewer: every picture gets its tile, whatever its size."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from engine import review

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@needs_ffmpeg
def test_a_square_still_among_wide_ones_keeps_every_tile(tmp_path):
    image = pytest.importorskip("PIL.Image")
    colours = [((320, 180), "red", (255, 0, 0)), ((200, 200), "green", (0, 128, 0)),
               ((320, 180), "blue", (0, 0, 255)), ((320, 180), "yellow", (255, 255, 0))]
    paths = []
    for i, ((w, h), name, _) in enumerate(colours):
        f = tmp_path / f"{i}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c={name}:s={w}x{h}",
                        "-frames:v", "1", str(f)], check=True)
        paths.append(f)
    out = tmp_path / "sheet.jpg"
    review._sheet(paths, out, 2, 160, 90)           # 28 Nov: the square bowl made ffmpeg drop the tiles before it
    im = image.open(out).convert("RGB")
    for i, (_, _, rgb) in enumerate(colours):
        x, y = (i % 2) * 164 + 80, (i // 2) * 94 + 45
        got = im.getpixel((x, y))
        assert sum(abs(a - b) for a, b in zip(got, rgb, strict=True)) < 90, (i, got, rgb)
