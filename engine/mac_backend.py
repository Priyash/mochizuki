"""Compatibility names for code written before the Mac side became the engine.mac package.

New code imports from engine.mac (and engine.release); see engine/mac/__init__.py.
"""
from .mac.backend import MacBackend, from_env
from .mac.common import log as _log
from .mac.config import DEFAULTS, NAME, settings
from .mac.engines import MfluxImageEngine as MfluxEngine
from .mac.engines import StubImageEngine as StubEngine
from .mac.models import hf_cache_dir, prefetch_model, resolve_cmd, watch_download
from .mac.process import run_streaming
from .mac.selftest import self_test
from .mac.sysinfo import health as mac_memory
from .mac.worker import run_batch, take_one, worker_loop
from .release import RELEASE, Release, ReleaseBackend
from .release import names as _names

__all__ = ["DEFAULTS", "NAME", "RELEASE", "MacBackend", "MfluxEngine", "Release", "ReleaseBackend", "StubEngine",
           "from_env", "hf_cache_dir", "mac_memory", "prefetch_model", "resolve_cmd", "run_batch", "run_streaming",
           "self_test", "settings", "take_one", "watch_download", "worker_loop", "_log", "_names"]
