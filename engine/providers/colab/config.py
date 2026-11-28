"""Colab's settings: the film's config.json -> "colab", over the defaults here."""
from __future__ import annotations

from ...log import log as _structured_log

NAME = "colab"
DEFAULTS = {
    "enabled": False,
    "stages": ["refs", "keyframes"],
    "gpus": ["A100", "L4"],
    "units_per_hour": {"T4": 1.19, "L4": 1.71, "A100": 5.40},
    "total_cap_units": 100,
    "min_balance_units": 3,
    "batch_hours": 4,
    "setup_minutes": 15,
    "max_wait_hours": 12,
    "stale_minutes": 30,
    "heartbeat_minutes": 5,
    "release_after_minutes": 20,
}
SESSION_PREFIX = "mochi-"               # every Colab VM session this engine opens is named so
RUNNER_WORKFLOW = "colab-run.yml"       # the GitHub workflow that runs a waiting batch on a VM


def settings(p) -> dict:
    cfg = {**DEFAULTS, **(p.config.get("colab", {}) if p is not None else {})}
    cfg["units_per_hour"] = {**DEFAULTS["units_per_hour"], **cfg.get("units_per_hour", {})}
    return cfg


def log(msg: str, event: str = "note", level: str = "info", **fields) -> None:
    _structured_log("colab", event, msg, level, **fields)
