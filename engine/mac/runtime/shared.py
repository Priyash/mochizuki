"""Small helpers every Mac module uses: the structured log, seeds, a stand-in PNG, ffmpeg."""
from __future__ import annotations

import shutil
import struct
import zlib
from pathlib import Path

from ...generators import seed_for  # noqa: F401 - the engines import it from here
from ...log import log as _structured_log

WORK = Path.home() / "mochizuki-mac"            # pictures, logs, models, training runs, LoRAs
MODELS = WORK / "models"
LORAS = WORK / "loras"


def log(msg: str, event: str = "note", level: str = "info", comp: str = "mac", **fields) -> None:
    """One structured line (engine/log.py): time, level, component, event, message, key=value."""
    _structured_log(comp, event, msg, level, **fields)


def write_png(path: Path, w: int, h: int) -> None:
    """A small gradient PNG, without Pillow: the stub engines' output."""
    raw = b"".join(b"\x00" + bytes([60, 80 + y * 100 // h, 140]) * w for y in range(h))
    ck = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))  # noqa: E731
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + ck(b"IDAT", zlib.compress(raw)) + ck(b"IEND", b""))


def ffmpeg_exe() -> str:
    """ffmpeg on PATH, else the one imageio-ffmpeg ships (mlx-video pulls it in)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # noqa: PLC0415
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("ffmpeg is not installed: brew install ffmpeg") from e
