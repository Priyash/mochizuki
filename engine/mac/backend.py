"""Producer side: the Mac as a backend the pipeline can send batches to (when mac.enabled is on)."""
from __future__ import annotations

import os

from ..release import ReleaseBackend
from .config import NAME, settings


class MacBackend(ReleaseBackend):
    """A batch for the Mac waits in the `batches` release until `mac-worker` on the Mac takes it."""
    name = NAME

    def __init__(self, p, repo: str, gh=None, fetch=None, **kw):
        super().__init__(p, repo, gh, fetch, cfg=settings(p), **kw)


def from_env(p):
    """The Mac backend when config.json switches it on (it needs no secret)."""
    if not settings(p).get("enabled"):
        return None
    return MacBackend(p, os.environ.get("GITHUB_REPOSITORY", ""))
