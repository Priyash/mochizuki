"""Modal as a second free GPU backend, beside Kaggle.

Same batch contract as Kaggle: the packed batch folder (manifest, worker.py, inputs) is
uploaded to a Modal Volume, the deployed function `mochizuki-worker.run_batch` is spawned
on it, and the results are read back from the Volume when the call has finished.

Staying free: Modal's Starter plan gives $30 of compute a month. With no payment method on
the account Modal cannot bill anything; work just stops when the credit is gone. On top of
that the pipeline keeps its own cap (config.json -> modal.monthly_cap_usd, below $30), counts
GPU + CPU + memory time at modal.usd_per_hour, sizes each batch to fit what is left, and
gives every call a hard timeout just past the batch's budget.
"""
from __future__ import annotations

from pathlib import Path

APP = "mochizuki-worker"
FUNCTION = "run_batch"
VOLUME = "mochizuki"


class ModalBackend:
    name = "modal"

    def __init__(self, p=None, modal_module=None):
        if modal_module is None:
            import modal as modal_module  # noqa: PLC0415
        self.modal = modal_module
        cfg = (p.config.get("modal", {}) if p is not None else {})
        self.app = cfg.get("app", APP)
        self.vol = self.modal.Volume.from_name(cfg.get("volume", VOLUME), create_if_missing=True)
        self.log: list[str] = []

    # ---- the same methods pipeline/kaggle.py calls on Kaggle
    def publish_dataset(self, folder: Path, ref: str, message: str) -> None:
        """Upload the batch's input folder. `message` is the batch id."""
        with self.vol.batch_upload(force=True) as up:
            up.put_directory(str(folder), f"/batches/{message}/dataset")
        self.log.append(f"uploaded {folder.name} for {message}")

    def wait_dataset_ready(self, ref: str, **_) -> None:
        return None                                   # a committed upload is ready at once

    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str:
        """Start the batch. Returns a handle: '<call id> <batch id>'."""
        batch_id = Path(folder).parent.name
        fn = self.modal.Function.from_name(self.app, FUNCTION).with_options(gpu=accelerator, timeout=int(timeout_s))
        call = fn.spawn(batch_id)
        self.log.append(f"spawned {batch_id} on {accelerator}, timeout {timeout_s} s: {call.object_id}")
        return f"{call.object_id} {batch_id}"

    def status(self, ref: str, handle: str | None = None) -> str:
        if not handle:
            return "unknown"
        call_id = handle.split()[0]
        exc = self.modal.exception
        try:
            self.modal.FunctionCall.from_id(call_id).get(timeout=0)
            st = "complete"
        except exc.FunctionTimeoutError:
            st = "error"                              # hit its hard stop; partial results are still read
        except exc.OutputExpiredError:
            st = "complete"                           # finished long ago; results are on the volume
        except (exc.TimeoutError, TimeoutError):
            st = "running"
        except Exception as e:  # noqa: BLE001
            self.log.append(f"call {call_id} failed: {e!r}"[:400])
            st = "error"
        self.log.append(f"status {call_id}: {st}")
        return st

    def output(self, ref: str, dest: Path, handle: str | None = None) -> None:
        batch_id = (handle or "").split()[-1]
        root = f"/batches/{batch_id}/out"
        dest = Path(dest) / "out"
        dest.mkdir(parents=True, exist_ok=True)
        try:
            entries = self.vol.listdir(root, recursive=True)
        except Exception as e:  # noqa: BLE001
            self.log.append(f"no output for {batch_id}: {e!r}"[:300])
            return
        for e in entries:
            if getattr(e, "type", None) is not None and "DIRECTORY" in str(e.type).upper():
                continue
            rel = e.path.split("/out/", 1)[-1] if "/out/" in e.path else Path(e.path).name
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as fh:
                self.vol.read_file_into_fileobj(e.path, fh)

    def gpu_hours_left(self) -> float | None:
        return None                                   # the pipeline's own monthly cap governs Modal
