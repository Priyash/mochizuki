"""Google Colab (pay-as-you-go compute units) as the bf16 GPU backend for stills and keyframes.

Two halves, talking through the `batches` GitHub release (as the Mac backend does):

  producer side (GitHub Actions)   ColabProvider (provider.py): packs a batch as `colab-<id>.tar.gz`, checks the
                                   agreed unit cap, and starts the `colab-run` workflow.
  runner (GitHub Actions)          `python -m engine colab-runner` (runner.py): checks the real compute-unit
                                   balance, claims the batch, opens a Colab VM with the Colab CLI
                                   (A100 first, then L4), runs the same worker as on Kaggle there,
                                   downloads the results, stops the VM, uploads
                                   `colab-<id>.results.tar.gz`. A heartbeat goes up every few minutes.

Money guards (Colab bills compute units while a VM is assigned):
  - colab.total_cap_units: the producer sizes every batch to what is left of the agreed cap,
    counting each batch's wall-clock time at the dearest allowed GPU's rate;
  - colab.min_balance_units: the runner does not start a VM below this real balance;
  - the VM is stopped in a `finally`, every run first stops any mochizuki VM left behind, and the
    job on the VM releases its own VM at its deadline even if nobody is listening any more.

Sign-in: the Colab CLI's Application Default Credentials (`--auth adc`), from the GitHub secret
COLAB_ADC_JSON (docs/colab.md). Nothing here buys units; the owner does.

Modules: config (settings), provider (producer side), cli (the colab command), runner (one batch on a
VM), checks (a test still, a free smoke test, which GPUs the account gets).
"""
from ..release import Release
from .checks import gpu_probe, smoke, test_still
from .cli import ColabCLI, _redact
from .config import DEFAULTS, NAME, RUNNER_WORKFLOW, SESSION_PREFIX, settings
from .provider import ColabProvider, from_env
from .runner import JOB, run_remote, salvage, stop_gpu_vms, take_one

__all__ = ["DEFAULTS", "JOB", "NAME", "RUNNER_WORKFLOW", "SESSION_PREFIX", "ColabCLI", "ColabProvider", "Release",
           "_redact", "from_env", "gpu_probe", "run_remote", "salvage", "settings", "smoke", "stop_gpu_vms",
           "take_one", "test_still"]
