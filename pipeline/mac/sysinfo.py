"""What the Mac has: chip, memory, disk, and how it is coping (free memory, swap, heat)."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .common import log


def _sh(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001 - not on a Mac, or the tool is missing
        return ""


def health() -> dict:
    """Free memory, swap and thermal state on macOS (empty elsewhere): what to check when a render crawls."""
    out: dict = {}
    m = re.search(r"free percentage:\s*(\d+)%", _sh("memory_pressure", "-Q"))
    if m:
        out["mem_free_pct"] = int(m.group(1))
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", _sh("sysctl", "-n", "vm.swapusage"))
    if m:
        out["swap_gb"] = round(float(m.group(1)) / (1024 if m.group(2) == "M" else 1), 1)
    t = _sh("pmset", "-g", "therm")
    if t:
        lv = re.findall(r"(thermal|performance) warning level.*?(\d+)", t, re.I)
        out["thermal"] = "warning" if any(int(v) > 0 for _, v in lv) else "nominal"
    return out


def health_warning(h: dict) -> str:
    """'' when all is well, else a short note for the log line."""
    notes = []
    if h.get("swap_gb", 0) > 4 or h.get("mem_free_pct", 100) < 10:
        notes.append("memory is tight, close other apps")
    if h.get("thermal") == "warning":
        notes.append("the Mac is throttling for heat")
    return "; ".join(notes)


def dir_gb(path: Path) -> float:
    try:
        return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    except OSError:
        return 0.0


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version  # noqa: PLC0415
        return version(pkg)
    except Exception:  # noqa: BLE001
        return "not installed"


def describe(folder: Path, need_gb: float = 40) -> dict:
    """Log one line about this Mac (chip, memory, macOS, free disk, tool versions) and return it."""
    chip = _sh("sysctl", "-n", "machdep.cpu.brand_string").strip() or "?"
    mem = _sh("sysctl", "-n", "hw.memsize").strip()
    ram = f"{int(mem) / 2**30:.0f} GB" if mem.isdigit() else "?"
    free = shutil.disk_usage(Path.home()).free / 1e9
    info = {"chip": chip, "ram": ram, "macos": _sh("sw_vers", "-productVersion").strip() or "?",
            "free_gb": round(free), "mflux": _version("mflux"), "mlx_video": _version("mlx-video")}
    log(f"this Mac: {chip}, {ram} memory, macOS {info['macos']}, {free:.0f} GB free disk, mflux {info['mflux']}, "
        f"mlx-video {info['mlx_video']}; output goes to {folder}", event="mac.info", **info)
    if free < need_gb:
        log(f"{free:.0f} GB free; the models need about {need_gb:.0f} GB", event="disk.low", level="warn",
            free_gb=round(free))
    return info
