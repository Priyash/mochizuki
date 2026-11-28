"""Running the model tools (mflux, mlx-video, mflux-train) and turning their output into structured logs."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any
import threading
import time
from pathlib import Path

from .common import log
from .sysinfo import dir_gb, health, health_warning

_STEP = re.compile(r"(\d+)/(\d+) \[")         # tqdm: "12/30 [01:02<..."
_BAR = re.compile(r"\d+%\|")


def run_streaming(cmd: list[str], timeout: float, label: str = "", every: float = 60.0, raw: bool | None = None,
                  cache: Path | None = None, kind: str = "render", tool: str = "mflux",
                  cwd: Path | None = None) -> tuple[int, str]:
    """Run a command and log its progress as structured lines:

      <kind>.loading     while the model loads (and model.download.progress while `cache` grows)
      <kind>.loaded      when the first step starts
      <kind>.step        each step: seconds per step, time left; memory/heat every 5 steps
      <kind>.status      when a step takes longer than `every` seconds (warns if nothing moves for 15 min)
      <tool>.out         any other output line (warn/error when it looks like one)

    MOCHI_RAW=1 shows the tool's own output instead. Returns (exit code, last few KB of output).
    The command is killed after `timeout` seconds."""
    raw = os.environ.get("MOCHI_RAW") == "1" if raw is None else raw
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    killer = threading.Timer(timeout, p.kill)
    killer.start()
    st: dict[str, Any] = {"step": 0, "total": 0, "first_step_at": None, "last_change": t0, "last_log": t0}
    stop = threading.Event()
    tag = f"{label} " if label else ""

    def status():
        while not stop.wait(5):
            now = time.time()
            if now - st["last_log"] < every:
                continue
            st["last_log"] = now
            h = health()
            if st["first_step_at"] is None:
                msg, ev = f"{tag}loading the model ({(now - t0) / 60:.1f} min so far)", f"{kind}.loading"
                if cache is not None:                   # the tool downloads a model the first time it is used
                    gb = dir_gb(cache)
                    grew = gb - st.get("cache_gb", gb)
                    st["cache_gb"] = gb
                    h["model_cache_gb"] = round(gb, 1)
                    if grew > 0.05:
                        msg = (f"{tag}downloading the model (first use): {gb:.1f} GB so far, "
                               f"{grew * 1000 / max(1.0, every):.0f} MB/s")
                        ev = "model.download.progress"
                        st["last_change"] = now
            else:
                msg, ev = (f"{tag}still on step {st['step']}/{st['total']} "
                           f"({(now - st['last_change']) / 60:.1f} min on this step)"), f"{kind}.status"
            level = "info"
            if now - st["last_change"] > max(900, 6 * every):
                msg += f"; nothing has moved for {(now - st['last_change']) / 60:.0f} min"
                level = "warn"
            warn = health_warning(h)
            if warn:
                msg += f"; {warn}"
                level = "warn"
            log(msg, event=ev, level=level, elapsed_min=round((now - t0) / 60, 1), **h)

    def line(text: str):
        m = _STEP.search(text)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            if step == st["step"] or step == 0:
                return
            now = time.time()
            if st["first_step_at"] is None:                         # the model is loaded: stepping starts
                st["first_step_at"] = st["bar_start"] = now
                log(f"{tag}model loaded in {(now - t0) / 60:.1f} min; {total} steps to go", event=f"{kind}.loaded",
                    load_min=round((now - t0) / 60, 1), steps=total)
            elif step < st["step"]:                                 # a new progress bar: the next stage
                st["bar_start"] = now
                log(f"{tag}next stage: {total} steps", event=f"{kind}.stage", steps=total)
            per = (now - st["bar_start"]) / (step - 1) if step > 1 else now - st["bar_start"]
            st.update(step=step, total=total, last_change=now, last_log=now)
            h = health() if step % 5 == 0 or step == total else {}
            log(f"{tag}step {step}/{total}, {per:.0f} s a step, about {per * (total - step) / 60:.0f} min left",
                event=f"{kind}.step", step=f"{step}/{total}", sec_per_step=round(per),
                min_left=round(per * (total - step) / 60), **h)
            return
        text = text.strip()
        if not text or _BAR.search(text):
            return
        bad = any(k in text for k in ("Error", "Traceback", "error:", "Killed"))
        warn = any(k in text for k in ("Warning", "WARNING", "⚠"))
        log(text[:500], event=f"{tool}.out", level="error" if bad else "warn" if warn else "info")

    threading.Thread(target=status, daemon=True).start()
    tail, buf = b"", ""
    assert p.stdout is not None
    try:
        while True:
            chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
            if not chunk:
                break
            tail = (tail + chunk)[-6000:]
            if raw:
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
            buf += chunk.decode("utf-8", "replace")
            parts = re.split(r"[\r\n]", buf)
            buf = parts.pop()
            for part in parts:
                if raw:
                    m = _STEP.search(part)
                    if m:
                        st["first_step_at"] = st["first_step_at"] or time.time()
                        st.update(step=int(m.group(1)), total=int(m.group(2)), last_change=time.time())
                else:
                    line(part)
        if buf and not raw:
            line(buf)
        code = p.wait()
    finally:
        stop.set()
        killer.cancel()
    if code < 0:
        log(f"{tag}stopped after {timeout / 60:.0f} min (time limit)", event="cmd.timeout", level="error",
            limit_min=round(timeout / 60))
    return code, tail.decode("utf-8", "replace")
