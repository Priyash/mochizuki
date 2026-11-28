"""What a GPU provider is to the scheduler.

A provider runs batches of jobs on GPUs: in the cloud (Kaggle, Colab) or offline on your own machine
(the Mac). The scheduler (engine/scheduler) drives every provider through this one interface, so it
holds no provider's rules, and a new provider is one class.

One batch, as the scheduler drives it:
  budget()                    how much GPU time the next batch may use (raises BudgetUsedUp to rest)
  refs()                      the names its inputs and its run travel under
  package()                   provider-specific files beside the packed inputs
  publish_dataset()           upload the inputs; wait_dataset_ready() until they can be used
  push()                      start the run; returns a handle to find it by
  status()                    where the run stands: a Status
  output()                    download what it made; raises if it could not (the scheduler retries)
  cancel()                    stop a run that is being written off
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from ..errors import ProviderError
from ..project import REPO

WORKER = REPO / "worker" / "worker.py"      # runs every batch on the GPU: the same file on every provider
WORKER_PACKAGES = ["diffusers>=0.36", "transformers>=4.51", "accelerate>=1.2", "gguf>=0.10", "bitsandbytes>=0.45",
                   "spandrel>=0.4", "ftfy", "imageio-ffmpeg", "huggingface_hub>=0.30", "sentencepiece"]

CLOUD = "cloud"          # someone else's GPUs (Kaggle, Colab)
OFFLINE = "offline"      # your own machine (the Mac)
RUN_GRACE_S = 1200       # a run's hard stop is this long past its own budget, so it cannot overspend


class Status(StrEnum):
    """Where a run stands. A StrEnum: it equals its value: Status.QUEUED == "queued"."""
    QUEUED = "queued"
    RUNNING = "running"
    UNKNOWN = "unknown"          # the provider could not be asked: ask again next time
    COLLECTING = "collecting"    # finished, but its download failed: try again next time
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"



IN_FLIGHT = (Status.QUEUED, Status.RUNNING, Status.UNKNOWN, Status.COLLECTING)   # not over yet: wait


class Provider(ABC):
    """One place batches of GPU jobs can run. Subclasses set `name` and implement the abstract methods."""

    name: str = ""
    kind: str = CLOUD
    stages: tuple[str, ...] | None = None    # the stages it takes; None: every stage
    take_over: bool = False                  # takes any stage while Kaggle's queue is stuck
    hands_over_when_queued: bool = False     # a batch stuck in its queue may go to another provider
    deadline_margin_s: float = 300           # the worker stops this long before the run's hard end
    accelerator: str = ""

    def __init__(self) -> None:
        self.log: list[str] = []             # what the provider said, for the run's status.json

    @property
    def capabilities(self) -> frozenset[str]:
        """What its GPUs have (engine.generators: BF16). Subclasses set it, or compute it."""
        return frozenset()

    # ---- sizing a batch
    @abstractmethod
    def budget(self, p, led) -> tuple[float, float]:
        """(GPU seconds the next batch may use, setup seconds). Raises BudgetUsedUp to rest."""

    def gpu_hours_left(self) -> float | None:
        """What the provider itself says is left this week, if it says (None: it doesn't)."""
        return None

    def refs(self, p) -> tuple[str, str]:
        """(inputs, run): the names a batch travels under. Only labels, unless the provider names things."""
        return f"{self.name}/{p.name}", f"{self.name}/{p.name}-worker"

    def accelerator_for(self, p) -> str:
        return self.accelerator

    def run_timeout(self, p, est_seconds: float) -> int:
        """The run's hard stop: just past its own budget, so it cannot overspend."""
        return int(est_seconds + RUN_GRACE_S)

    def queue_patience_s(self, p) -> float | None:
        """How long a batch may sit in this provider's queue before it is handed to another (None: forever)."""
        return None

    def package(self, p, work: Path, batch_id: str, engine_set: str) -> None:  # noqa: B027 - an optional hook
        """Write files this provider needs beside the packed inputs (work/dataset, work/kernel)."""

    def account_check(self) -> dict | None:
        """Can we sign in, and what does the account say? None when there is nothing to check."""
        return None

    def cleanup(self, led) -> int:
        """Stop paid machines nobody tracks any more. How many it stopped."""
        return 0

    # ---- one batch
    @abstractmethod
    def publish_dataset(self, folder: Path, ref: str, message: str) -> None:
        """Upload the packed inputs in `folder` under `ref`; `message` is the batch id."""

    def wait_dataset_ready(self, ref: str, timeout: float = 900, poll: float = 15) -> None:  # noqa: B027 - an optional hook
        """Block until uploaded inputs can be used (at most `timeout` s). Most providers need no wait."""

    @abstractmethod
    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str | None:
        """Start the run. A handle to find it by, or None if `refs` is enough."""

    @abstractmethod
    def status(self, ref: str, handle: str | None = None) -> str:
        """Where the run stands: one of Status."""

    @abstractmethod
    def output(self, ref: str, dest: Path, handle: str | None = None) -> None:
        """Download what the run made into `dest`. Raise if it could not: nothing may be lost."""

    def cancel(self, ref: str, handle: str | None = None) -> None:
        """Stop a run that is being written off."""
        raise ProviderError(f"{self.name} cannot cancel a run")
