"""Checks a film's own files when it loads (config.json and shots.csv), so a mistake is one clear
list of problems up front, not a KeyError deep inside a batch on a paid GPU.

Every problem is reported at once, each saying where it is and what is wrong. Only what the engine
relies on is checked; anything else in the config is the film's own business.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import EngineError
from .generators import CATALOG, FINISH_KEYS, GPU, GUIDES, engine_settings

SHOT_COLUMNS = ("id", "chapter", "shot", "seconds", "type", "model", "takes")   # "sound" is optional
VIDEO_ENGINE_KEYS = ("model", "width", "height", "fps", "max_frames")
PROVIDER_SECTIONS = ("kaggle", "colab", "mac")
CHAPTER_TABLES = ("model", "height_m", "look")      # cgi.characters.<name>: tables keyed by chapter
GPU_STAGES = frozenset(g.stage for g in CATALOG.values() if g.built and GPU in g.runs_on)


class FilmError(EngineError):
    """The film's files have problems; `problems` lists every one."""

    def __init__(self, film: str, problems: list[str]):
        self.problems = problems
        super().__init__(f"{film}: {len(problems)} problem(s) in the film's files:\n"
                         + "\n".join(f"  - {x}" for x in problems))


def read_checked(path: Path, problems_of: Callable[[dict], list[str]] | None = None) -> dict:
    """One of the style bible's JSON files ({} when there is none), with every problem in it raised at
    once (FilmError) instead of a KeyError in the middle of a render."""
    if not path.exists():
        return {}
    name = f"{path.parent.name}/{path.name}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise FilmError(name, [f"not valid JSON: {e}"]) from e
    if not isinstance(data, dict):
        raise FilmError(name, ["the file must hold one JSON object"])
    bad = problems_of(data) if problems_of else []
    if bad:
        raise FilmError(name, bad)
    return data


def _number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _section(cfg: dict, name: str) -> dict:
    """A config section, or {} when it is missing or not an object (that is reported separately)."""
    v = cfg.get(name)
    return v if isinstance(v, dict) else {}


def _int(v: Any) -> int | None:
    try:
        return int(str(v).strip())
    except ValueError:
        return None


def _box(v: Any) -> bool:
    """A crop box: x, y, width, height as fractions of the picture, inside it."""
    if not isinstance(v, list) or len(v) != 4 or not all(_number(n) for n in v):
        return False
    x, y, w, h = v
    return 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 - x + 1e-9 and 0 < h <= 1 - y + 1e-9


def _finish_problems(engines: dict) -> list[str]:
    """engines.ai_finish and its variants (ai_finish_720: only what it changes), each as the worker
    will see it (engine_settings)."""
    out: list[str] = []
    if "ai_finish" not in engines:
        if any(n.startswith("ai_finish_") for n in engines):
            out.append("config.json engines: a variant of ai_finish needs engines.ai_finish, its base")
        return out
    for name in sorted(n for n in engines if n == "ai_finish" or n.startswith("ai_finish_")):
        if not isinstance(engines[name], dict):
            out.append(f"config.json engines.{name}: must be an object")
            continue
        fin, where = engine_settings(engines, name), f"config.json engines.{name}"
        lacking = [k for k in FINISH_KEYS if k not in fin]
        if lacking:
            out.append(f"{where}: needs {', '.join(lacking)}")
        if fin.get("control", GUIDES[0]) not in GUIDES:
            out.append(f"{where}.control: must be one of {', '.join(GUIDES)}")
        if "style" in fin and not isinstance(fin["style"], str):
            out.append(f"{where}.style: must be text")
        for key in ("crops", "who"):
            if key in fin and not isinstance(fin[key], dict):
                out.append(f"{where}.{key}: must map reference still names to values")
        for ref, box in (fin.get("crops") or {}).items() if isinstance(fin.get("crops"), dict) else ():
            if not _box(box):
                out.append(f"{where}.crops.{ref}: must be [x, y, width, height], fractions inside the picture")
        g = fin.get("transformer_gguf")
        if g is not None and not (isinstance(g, dict) and isinstance(g.get("repo"), str)):
            out.append(f"{where}.transformer_gguf: needs repo (a Hugging Face repo with the GGUF files)")
        if fin.get("offload", "model") not in ("model", "group"):
            out.append(f"{where}.offload: must be model (the default) or group")
    return out


def config_problems(cfg: Any) -> list[str]:
    if not isinstance(cfg, dict):
        return ["config.json: must be a JSON object"]
    out = []
    for section in ("models", "engines", "budget", "keyframes"):
        if not isinstance(cfg.get(section), dict):
            out.append(f"config.json: the \"{section}\" section is missing (or is not an object)")
    models = _section(cfg, "models")
    for name, m in models.items():
        if not isinstance(m, dict) or not _number(m.get("rate_usd_per_s")):
            out.append(f"config.json models.{name}: needs rate_usd_per_s (a number; 0 when it is free)")
    budget = _section(cfg, "budget")
    if budget and not _number(budget.get("total_usd")):
        out.append("config.json budget.total_usd: must be a number (0 on the free plan)")
    per = budget.get("per_chapter_usd", {}) if budget else {}
    if not isinstance(per, dict) or any(_int(k) is None or not _number(v) for k, v in per.items()):
        out.append("config.json budget.per_chapter_usd: must map chapter numbers to amounts, e.g. {\"1\": 5}")
    kf = _section(cfg, "keyframes")
    if kf and not _number(kf.get("price_usd")):
        out.append("config.json keyframes.price_usd: must be a number (0 on the free plan)")
    if kf and (_int(kf.get("candidates")) or 0) < 1:
        out.append("config.json keyframes.candidates: must be a whole number, at least 1")
    for section in PROVIDER_SECTIONS:
        stages = _section(cfg, section).get("stages")
        unknown = sorted(set(stages or []) - GPU_STAGES)
        if unknown:
            out.append(f"config.json {section}.stages: {', '.join(unknown)} is not a GPU stage "
                       f"({', '.join(sorted(GPU_STAGES))})")
    out += _finish_problems(_section(cfg, "engines"))
    chars = _section(cfg, "cgi").get("characters", {})
    for name, c in (chars.items() if isinstance(chars, dict) else []):
        if not isinstance(c, dict):
            out.append(f"config.json cgi.characters.{name}: must be an object")
            continue
        for table in CHAPTER_TABLES:
            t = c.get(table, {})
            if not isinstance(t, dict) or any(_int(k) is None for k in t):
                out.append(f"config.json cgi.characters.{name}.{table}: "
                           f"must be keyed by chapter number, e.g. {{\"1\": ...}}")
    return out


def shot_problems(cfg: dict, rows: list[dict], columns: list[str]) -> list[str]:
    missing = [c for c in SHOT_COLUMNS if c not in columns]
    if missing:
        return [f"shots.csv: missing column(s) {', '.join(missing)}; it needs {', '.join(SHOT_COLUMNS)}"]
    models, engines = _section(cfg, "models"), _section(cfg, "engines")
    out, seen = [], set()
    gpu_models: dict[str, list[str]] = {}
    for n, r in enumerate(rows, start=2):                      # line 1 is the header
        where = f"shots.csv line {n} ({r.get('id') or 'no id'})"
        sid = (r.get("id") or "").strip()
        if not sid:
            out.append(f"{where}: no id")
        elif sid in seen:
            out.append(f"{where}: the id {sid} is used twice")
        seen.add(sid)
        if (_int(r.get("chapter")) or 0) < 1:
            out.append(f"{where}: chapter must be a whole number, at least 1")
        if (_int(r.get("seconds")) or 0) < 1:
            out.append(f"{where}: seconds must be a whole number, at least 1")
        if _int(r.get("takes")) is None or (_int(r.get("takes")) or 0) < 0:
            out.append(f"{where}: takes must be a whole number, 0 or more")
        model = r.get("model", "")
        if model not in models:
            out.append(f"{where}: the model {model!r} is not in config.json models")
            continue
        m = models[model] if isinstance(models[model], dict) else {}
        if m.get("generates_video") and m.get("provider", "editor") == "kaggle":
            gpu_models.setdefault(model, []).append(sid)
    for model, used_by in gpu_models.items():                  # once per model, not once per shot
        e = engines.get(model)
        lacking = [k for k in VIDEO_ENGINE_KEYS if not isinstance(e, dict) or k not in e]
        if lacking:
            shots = ", ".join(used_by[:3]) + (f" and {len(used_by) - 3} more" if len(used_by) > 3 else "")
            out.append(f"config.json engines.{model}: needs {', '.join(lacking)} (the GPU makes its takes: {shots})")
    return out


def problems(cfg: Any, rows: list[dict], columns: list[str]) -> list[str]:
    """Every problem in a film's config and shot sheet; empty when the film is fine."""
    out = config_problems(cfg)
    if isinstance(cfg, dict):
        out += shot_problems(cfg, rows, columns)
    unique: list[str] = []
    for x in out:                                               # one line per problem, in order
        if x not in unique:
            unique.append(x)
    return unique
