"""A stand-in for Kaggle that runs the worker on this machine with stub engines: free and offline.

Tests and `--kaggle local` use it. Named anything but "kaggle", it plays a second provider instead
(budget, names and timeouts of a generic provider), so two providers side by side can be tested.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ...generators import CAPABILITIES
from ..base import Provider, Status
from .provider import NAME, KaggleError, KaggleProvider, session_budget

STUB_ENGINES = "stub"
WORKER_OK_CODES = (0, 3)     # the worker's exit codes for "ran" (3: some jobs failed, the run did not)
SECOND_PROVIDER_SETUP_S = 300.0


def _no_cli(args: list[str]) -> tuple[int, str]:
    raise KaggleError("the local stand-in never runs the kaggle command")


class LocalKaggle(KaggleProvider):
    capabilities = CAPABILITIES      # stand-in engines run anything

    def __init__(self, root: Path, name: str = NAME):
        super().__init__(runner=_no_cli)
        self.name = name
        self.root = root
        self.datasets: dict[str, Path] = {}
        self.runs: dict[str, str] = {}
        self.outputs: dict[str, Path] = {}
        if name != NAME:                               # a second provider: Kaggle's own rules do not apply
            self.hands_over_when_queued = False
            self.deadline_margin_s = Provider.deadline_margin_s

    @property
    def _is_kaggle(self) -> bool:
        return self.name == NAME

    # ---- sizing: Kaggle's when it plays Kaggle, a generic provider's otherwise
    def budget(self, p, led) -> tuple[float, float]:
        return super().budget(p, led) if self._is_kaggle else (session_budget(p), SECOND_PROVIDER_SETUP_S)

    def refs(self, p) -> tuple[str, str]:
        return super().refs(p) if self._is_kaggle else Provider.refs(self, p)

    def accelerator_for(self, p) -> str:
        return super().accelerator_for(p) if self._is_kaggle else Provider.accelerator_for(self, p)

    def run_timeout(self, p, est_seconds: float) -> int:
        return super().run_timeout(p, est_seconds) if self._is_kaggle else Provider.run_timeout(self, p, est_seconds)

    def account_check(self) -> dict | None:
        return None

    def gpu_hours_left(self) -> float | None:
        return None

    # ---- one batch, run here
    def publish_dataset(self, folder: Path, ref: str, message: str) -> None:
        self.datasets[ref] = folder

    def wait_dataset_ready(self, ref: str, **_) -> None:
        if ref not in self.datasets:
            raise KaggleError(f"dataset {ref} was never published")

    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str:
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        ref = meta["id"] + ("" if self._is_kaggle else f"-{self.name}")
        data = self.datasets[meta["dataset_sources"][0]] if self._is_kaggle else folder.parent / "dataset"
        out = self.root / "local_kaggle" / ref.replace("/", "_") / "out"
        shutil.rmtree(out, ignore_errors=True)
        batch = re.search(r"BATCH_ID = '([^']+)'", (folder / "bootstrap.py").read_text()).group(1)
        r = subprocess.run([sys.executable, str(data / "worker.py"), "--manifest", str(data / "manifest.json"),
                            "--inputs", str(data), "--out", str(out), "--engine-set", STUB_ENGINES,
                            "--expect-batch", batch], capture_output=True, text=True)
        self.runs[ref] = Status.COMPLETE if r.returncode in WORKER_OK_CODES else Status.ERROR
        self.outputs[ref] = out
        if r.returncode not in WORKER_OK_CODES:
            print(r.stdout[-2000:], r.stderr[-2000:])
        return ref

    def status(self, ref: str, handle: str | None = None) -> str:
        return self.runs.get(handle or ref, Status.UNKNOWN)

    def output(self, ref: str, dest: Path, handle: str | None = None) -> None:
        src = self.outputs.get(handle or ref)
        dest.mkdir(parents=True, exist_ok=True)
        if src and src.exists():
            shutil.copytree(src, dest / "out", dirs_exist_ok=True)

    def cancel(self, ref: str, handle: str | None = None) -> None:
        self.runs[handle or ref] = Status.CANCELLED
