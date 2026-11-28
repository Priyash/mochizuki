"""The engine's errors, grouped by what went wrong. Each message says why, and what to change."""
from __future__ import annotations


class EngineError(RuntimeError):
    """Something the engine cannot do as asked."""


class BatchError(EngineError):
    """A batch is not in the state an action needs: one open batch per provider, a missing input,
    nothing running to collect."""


class ProviderError(EngineError):
    """A GPU provider refused or failed: signing in, uploading, starting, reading or cancelling a run."""


class BudgetUsedUp(ProviderError):
    """A provider's allowance is spent for now (Kaggle's week, the agreed Colab cap). Not a failure:
    the provider rests until it refills."""
