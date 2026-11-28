"""Checks you run on the Mac before it takes real work; nothing is sent to GitHub.

    self_test   one reference still from the key-art prompt, then one keyframe edited from it
    video_test  one short take animated from the self-test still (Wan 2.2)
Each prints what it is doing as structured lines and says plainly which part failed and why.
"""
from __future__ import annotations

import time
from typing import Any
from pathlib import Path

from .common import log
from .sysinfo import describe

EDIT_PROMPT = ("Low-angle shot from floor level, camera looking up at the two of them; same place, same people, "
               "same clothes, now at dusk.")
VIDEO_PROMPT = ("The grandmother brings the wooden mallet down into the mortar in one smooth swing while the girl "
                "pulls her hands back, laughing; steam drifts up; lantern light flickers softly; the camera holds "
                "still.")


STYLE_2D = ("Hand-drawn 2D anime feature film, traditional cel animation look with clean ink lines and flat cel "
            "shading, painted watercolour backgrounds, anime proportions and large expressive eyes, subtle emotional "
            "acting in faces and hands, cinematic 16:9 framing, Japanese feature-film quality. No text, no logos, no "
            "watermark.")
_3D_TO_2D = (("3D CGI animated feature film still: fully 3D-rendered cel-shaded character models with toon shading and "
               "clean ink outlines, 3D lighting with soft shadows and depth.",
               "2D hand-drawn anime feature film still: traditional cel animation with clean ink lines, flat cel "
               "shading and painted backgrounds."),
              ("3D CGI render, cel-shaded 3D models, volumetric light,", "Hand-drawn 2D cel animation, painted light,"))


def key_art_prompt(p, look: str = "3d") -> tuple[str, dict]:
    """The style text plus the key-art prompt. look="2d" swaps every 3D wording for its 2D twin, so
    the two looks can be compared with everything else the same (model, prompt, seed)."""
    style = p.style_file
    r = next(x for x in p.ref_plan() if x["name"] == "style")
    style_text = style.read_text(encoding="utf-8").strip() if style.exists() else ""
    body = r["prompt"]
    if look == "2d":
        style_text = STYLE_2D
        for a, b in _3D_TO_2D:
            body = body.replace(a, b)
    return "\n".join(x for x in (style_text, body) if x), r


def _step(n: str, what: str, fn) -> dict[str, Any]:
    log(f"self-test {n}: {what}", event="selftest.start", step=n)
    t0 = time.time()
    try:
        info = fn()
    except Exception as e:  # noqa: BLE001 - say clearly which part failed, and why
        log(f"self-test {n} failed after {(time.time() - t0) / 60:.1f} min. Send me this line and the lines above it. "
            f"Error: {e}", event="selftest.failed", level="error", step=n, error=str(e)[-800:])
        return {"ok": False, "error": str(e)[-1500:]}
    took = round(time.time() - t0, 1)
    log(f"self-test {n} done in {took / 60:.1f} min", event="selftest.step_done", step=n, seconds=took)
    return {"ok": True, "seconds": took, **(info or {})}


def self_test(p, engines, out: Path, edit_only: bool = False, look: str = "3d", variants: int = 1) -> dict:
    """1/2 a reference still from the key-art prompt (with variants > 1: that many, each with its own
    seed, named <name>-v1.png ...); 2/2 a keyframe edited from the first one."""
    prompt, r = key_art_prompt(p, look)
    out.parent.mkdir(parents=True, exist_ok=True)
    describe(out.parent)
    stale = out.with_name(out.stem + "-edit.png")
    if stale.exists() and not edit_only:
        stale.unlink()                     # never leave an old edit that could be mistaken for this run's
    if variants > 1 and not edit_only:
        made = {}
        for v in range(1, variants + 1):
            f = out.with_name(f"{out.stem}-v{v}.png")
            made[f.name] = _step(f"still {v}/{variants}", f"variant {v} of the reference still (its own seed)",
                                 lambda f=f, v=v: {**engines.run({"key": f"selftest-style-v{v}", "prompt": prompt,
                                                                  "width": r.get("width", 1664),
                                                                  "height": r.get("height", 928)}, p.root, f),
                                                   "file": str(f)})
        ok = [n for n, x in made.items() if x["ok"]]
        if not ok:
            return {"stills": made}
        import shutil  # noqa: PLC0415
        shutil.copyfile(out.with_name(ok[0]), out)     # the keyframe test edits the first good variant
        log(f"{len(ok)} of {variants} variants made: {', '.join(ok)} in {out.parent}", event="selftest.variants",
            made=len(ok))
        edit_only = True
    if edit_only and not out.exists():
        log(f"--edit-only: no still at {out} (moved or renamed?), so making one first", event="selftest.no_still",
            level="warn", file=str(out))
    if edit_only and out.exists():
        log(f"self-test: reusing {out}; testing only the keyframe engine", event="selftest.edit_only", file=str(out))
        still: dict[str, Any] = {"ok": True, "file": str(out), "seconds": 0, "reused": True}
    else:
        still = _step("1/2", "one reference still (text to image); the first run also downloads the model",
                       lambda: {**engines.run({"key": "selftest-style", "prompt": prompt, "width": r.get("width", 1664),
                                               "height": r.get("height", 928)}, p.root, out), "file": str(out)})
        if not still["ok"]:
            return {"still": still}
    edit_out = out.with_name(out.stem + "-edit.png")
    edit = _step("2/2", "one keyframe edited from that still (image edit)",
                 lambda: {**engines.run({"key": "selftest-edit", "prompt": EDIT_PROMPT, "refs": [out.name],
                                         "width": r.get("width", 1664), "height": r.get("height", 928)},
                                        out.parent, edit_out), "file": str(edit_out)})
    if edit["ok"]:
        per = float(still["seconds"] or edit["seconds"])
        log(f"self-test passed: {out.name} and {edit_out.name} in {out.parent}. At this speed 24 stills take about "
            f"{24 * per / 3600:.1f} h.", event="selftest.passed", est_hours_24=round(24 * per / 3600, 1))
    return {"still": still, "edit": edit}


def video_test(p, engines, still: Path, out: Path, seconds: float = 2.0) -> dict:
    """One short take animated from `still` with the video engine."""
    out.parent.mkdir(parents=True, exist_ok=True)
    describe(out.parent, need_gb=25)
    if not still.exists():
        msg = f"no still at {still}: run `mac-worker --self-test` first, or pass --image"
        log(msg, event="selftest.failed", level="error", step="video")
        return {"video": {"ok": False, "error": msg}}
    video = _step("video", f"a {seconds:g}-second take from {still.name} (Wan 2.2); the first run downloads about 19 GB",
                  lambda: {**engines.run({"key": "selftest-take", "kind": "take", "prompt": VIDEO_PROMPT,
                                          "image": still.name, "seconds": seconds}, still.parent, out),
                           "file": str(out)})
    if video["ok"]:
        log(f"video test passed: {out}. At this speed a 10-minute film is about "
            f"{600 / seconds * video['seconds'] / 3600:.0f} h of rendering (one take per shot).",
            event="selftest.passed", est_hours_film=round(600 / seconds * video["seconds"] / 3600))
    return {"video": video}
