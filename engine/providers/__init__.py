"""Where GPU work runs.

Batch providers, behind one interface (base.Provider), asked in this order:
  cloud     Kaggle (free T4s, about 30 h a week)           kaggle/
            Colab (paid A100/L4, inside the agreed cap)    colab/
  offline   your Mac (free, when mac.enabled is on)        mac.py
Each takes the stages its config lists ("stages"). Any provider runs any stage the worker knows, so a
model can move to another provider by editing the film's config, not the code.

Per-call model APIs (paid.py: fal, and a free mock for tests) are reached with get().
"""
from __future__ import annotations

import os
from pathlib import Path

from . import colab, mac
from .base import CLOUD, IN_FLIGHT, OFFLINE, WORKER, WORKER_PACKAGES, Provider, Status
from .kaggle import KaggleProvider, LocalKaggle
from .paid import get

LOCAL = "local"                       # --kaggle local: the stand-in that runs the worker on this machine
REAL = "real"
RELEASE_PROVIDERS = (colab, mac)      # their batches wait in the `batches` release; asked after Kaggle, in this order
KAGGLE_KEY_FILE = Path("~/.kaggle/kaggle.json")


def kaggle_provider(mode: str, p) -> KaggleProvider:
    return LocalKaggle(p.media) if mode == LOCAL else KaggleProvider(p)


def has_kaggle_credentials(env=None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("KAGGLE_USERNAME") and (env.get("KAGGLE_API_TOKEN") or env.get("KAGGLE_KEY")
                                                 or KAGGLE_KEY_FILE.expanduser().exists()))


def available(p, kaggle_mode: str = REAL) -> tuple[list[Provider], list[str]]:
    """Every provider the film may use now, in the order they are asked, and the names of the release
    providers its config switches off (their waiting batches should go back to the pool)."""
    out: list[Provider] = []
    off: list[str] = []
    if kaggle_mode == LOCAL or has_kaggle_credentials():
        out.append(kaggle_provider(kaggle_mode, p))
    for mod in RELEASE_PROVIDERS:
        prov = mod.from_env(p)
        if prov is None:
            off.append(mod.NAME)
        else:
            out.append(prov)
    return out, off


__all__ = ["CLOUD", "IN_FLIGHT", "LOCAL", "OFFLINE", "REAL", "WORKER", "WORKER_PACKAGES", "KaggleProvider",
           "LocalKaggle", "Provider", "Status", "available", "colab", "get", "has_kaggle_credentials",
           "kaggle_provider", "mac"]
