"""The scheduler: decides what each chapter needs next from the GPU and drives every provider through
the same batch steps. It holds no provider's rules (those live in engine/providers).

  batches   one batch's life: build (reserve + pack), submit, collect, discard, cancel_run
  loop      the producer loop: next_stage, tick (one provider), tick_all (every provider)
"""
from ..ledger import MISSING
from .batches import COLLECT_TRIES, STAGES, batch_dir, build, cancel_run, collect, discard, pack, submit
from .loop import Action, code_time, engine_broken, next_stage, queue_stuck, someone_else_can_take, tick, tick_all

__all__ = ["COLLECT_TRIES", "MISSING", "STAGES", "Action", "batch_dir", "build", "cancel_run", "code_time", "collect",
           "discard", "engine_broken", "next_stage", "pack", "queue_stuck", "someone_else_can_take", "submit", "tick",
           "tick_all"]
