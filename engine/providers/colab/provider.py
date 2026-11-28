"""Producer side: Colab as a provider. A batch waits in the `batches` release as colab-<id>.tar.gz; the
colab-run workflow then runs it on a Colab VM (runner.py) and uploads the results."""
from __future__ import annotations

import os
import time
from pathlib import Path

from ...errors import BudgetUsedUp
from ..base import CLOUD
from ..release import ReleaseProvider
from .config import NAME, RUNNER_WORKFLOW, settings


class ColabProvider(ReleaseProvider):
    """Colab's paid GPUs (A100, L4), inside the agreed unit cap."""

    name = NAME
    kind = CLOUD

    def __init__(self, p, repo: str, gh=None, fetch=None, now=time.time):
        super().__init__(p, repo, gh, fetch, now, cfg=settings(p))

    def rate(self) -> float:
        """Units per hour of the dearest GPU we may get: the conservative rate for the cap."""
        r = self.cfg["units_per_hour"]
        return max(float(r.get(g, 6.0)) for g in self.cfg["gpus"])

    def spent_units(self, led) -> float:
        """Units the agreed cap has used, counted at the dearest allowed GPU's rate (conservative).

        Only VM time counts, never time a batch spent waiting for a runner or for units:
          - finished batches count the VM time the runner reported (gpu_seconds);
          - a finished batch with no report counts nothing if it simply waited out max_wait_hours
            (no VM ever opened), and a full batch if its runner died mid-run (the VM may have billed);
          - the open batch counts as a full batch: room is kept for it until it reports."""
        full = float(self.cfg["batch_hours"]) * 3600 + float(self.cfg["setup_minutes"]) * 60
        never_taken = float(self.cfg["max_wait_hours"]) * 3600 * 0.95
        secs = 0.0
        for r in led.db.execute("SELECT submitted_at, finished_at, gpu_seconds FROM batches WHERE backend=? "
                                "AND submitted_at IS NOT NULL", (self.name,)).fetchall():
            if r["finished_at"] is None:
                secs += full
            elif r["gpu_seconds"] is not None:
                secs += float(r["gpu_seconds"])
            elif r["finished_at"] - r["submitted_at"] < never_taken:
                secs += min(full, r["finished_at"] - r["submitted_at"])
        return secs / 3600 * self.rate()

    def budget(self, p, led) -> tuple[float, float]:
        cap, spent, rate = float(self.cfg["total_cap_units"]), self.spent_units(led), self.rate()
        setup = float(self.cfg["setup_minutes"]) * 60
        secs = min(float(self.cfg["batch_hours"]) * 3600, (cap - spent) / rate * 3600 - setup - 600)
        if secs <= 900:
            raise BudgetUsedUp(f"the agreed Colab cap is used up ({spent:.0f} of {cap:.0f} compute units by the "
                               "producer's count); raise colab.total_cap_units in config.json to go on")
        self.log.append(f"Colab batch budget {secs / 3600:.1f} h; about {spent:.0f} of {cap:.0f} units used")
        return secs, setup

    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str:
        code, out = self.rel._gh(["api", "-X", "POST", f"repos/{self.rel.repo}/actions/workflows/{RUNNER_WORKFLOW}/dispatches",
                                  "-f", "ref=main", "-f", "inputs[mode]=run"])
        self.log.append(f"{RUNNER_WORKFLOW} started" if code == 0 else f"{RUNNER_WORKFLOW} not started ({out.strip()[-200:]}); "
                        "its schedule will pick the batch up")
        return Path(folder).parent.name


def from_env(p):
    """The Colab provider when the film's config switches it on."""
    if not settings(p).get("enabled"):
        return None
    return ColabProvider(p, os.environ.get("GITHUB_REPOSITORY", ""))
