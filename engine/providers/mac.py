"""The Mac as a provider: your own Apple Silicon machine, offline and free.

A batch for the Mac waits in the `batches` release until `python -m engine mac-worker`, running on the
Mac, takes it (engine/mac/worker.py). It is used only when the film's config switches mac.enabled on.
"""
from __future__ import annotations

import os

from ..mac.config import NAME, settings
from .base import OFFLINE
from .release import ReleaseProvider


class MacProvider(ReleaseProvider):
    name = NAME
    kind = OFFLINE

    def __init__(self, p, repo: str, gh=None, fetch=None, **kw):
        super().__init__(p, repo, gh, fetch, cfg=settings(p), **kw)


def from_env(p):
    """The Mac provider when the film's config switches it on (it needs no secret)."""
    if not settings(p).get("enabled"):
        return None
    return MacProvider(p, os.environ.get("GITHUB_REPOSITORY", ""))
