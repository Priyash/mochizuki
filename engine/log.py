"""Structured logs for the workers and runners.

One line per event on the terminal, easy to read and to grep:

    14:44:01 INFO  mac    render.done     ref-style ok in 205 s  key=ref-style seconds=205 step=1/24

and, when a log file is set (set_file, or the MOCHI_LOG_FILE environment variable), the same
event as one JSON object per line, for searching and for sending to me when something fails:

    {"ts": "2026-11-27T09:14:01Z", "level": "info", "comp": "mac", "event": "render.done",
     "msg": "ref-style ok in 205 s", "key": "ref-style", "seconds": 205, "step": "1/24"}

Levels: debug, info, warn, error. MOCHI_LOG_LEVEL=debug shows debug lines too.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_lock = threading.Lock()
_file: Path | None = Path(os.environ["MOCHI_LOG_FILE"]) if os.environ.get("MOCHI_LOG_FILE") else None


def set_file(path: Path | str | None) -> Path | None:
    """Also write every event as JSON lines to `path` (its folder is created)."""
    global _file
    _file = Path(path).expanduser() if path else None
    if _file:
        _file.parent.mkdir(parents=True, exist_ok=True)
    return _file


def _fmt(v) -> str:
    if isinstance(v, float):
        v = round(v, 2)
    s = str(v)
    return f'"{s}"' if (" " in s or not s) else s


def log(comp: str, event: str, msg: str = "", level: str = "info", **fields) -> None:
    """Log one event. `event` is a short dotted name (batch.claimed, render.done, vm.stopped);
    `fields` are the facts worth searching for later (key, seconds, units, gpu...)."""
    level = level if level in LEVELS else "info"
    if LEVELS[level] < LEVELS.get(os.environ.get("MOCHI_LOG_LEVEL", "info"), 20):
        return
    now = time.time()
    kv = "  ".join(f"{k}={_fmt(v)}" for k, v in fields.items() if v is not None)
    line = f"{time.strftime('%H:%M:%S', time.localtime(now))} {level.upper():5} {comp:6} {event:16} {msg}"
    if kv:
        line += ("  " if msg else "") + kv
    with _lock:
        print(line, file=sys.stderr if level == "error" else sys.stdout, flush=True)
        if _file:
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "level": level, "comp": comp,
                   "event": event, "msg": msg, **{k: v for k, v in fields.items() if v is not None}}
            try:
                with _file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
            except OSError:
                pass


class Logger:
    """log() bound to one component: L = Logger("mac"); L.info("batch.claimed", "...", batch=id)."""

    def __init__(self, comp: str):
        self.comp = comp

    def debug(self, event, msg="", **kw):
        log(self.comp, event, msg, "debug", **kw)

    def info(self, event, msg="", **kw):
        log(self.comp, event, msg, "info", **kw)

    def warn(self, event, msg="", **kw):
        log(self.comp, event, msg, "warn", **kw)

    def error(self, event, msg="", **kw):
        log(self.comp, event, msg, "error", **kw)
