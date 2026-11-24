# Rendering stills and keyframes on your Mac

Your own Apple Silicon Mac makes the reference stills (gate 1) and the keyframes for free, with
[mflux](https://pypi.org/project/mflux/) (Apple's MLX, native on M-series chips). Kaggle's T4s
can't: Qwen-Image overflows in their fp16. Video and the 4K upscale stay on Kaggle or RunPod.

The producer on GitHub keeps working as before. When it needs stills or keyframes, it puts a batch
in the repo's `batches` release and waits. The worker on your Mac takes it, renders it, and
uploads the results; the next producer run collects them and the review and gates carry on.
If no Mac takes a batch within 72 hours, its jobs are freed again.

## What you need

- An Apple Silicon Mac (M1 or later), macOS 14 or later.
- **Memory:** 48 GB or more for the 8-bit models (the default). With 32 GB, use the 4-bit models
  below. Less than 32 GB will not work for these models.
- About 70 GB free disk for the two models (downloaded once, on the first run).
- Python 3.10 or later, git, and the GitHub CLI.

## One-time setup (Terminal)

```bash
# 1. Tools
brew install git gh python@3.12
gh auth login                      # GitHub.com, HTTPS, log in with the browser

# 2. The project and mflux
git clone https://github.com/Priyash/mochizuki.git
cd mochizuki
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U mflux

# 3. Check both engines on this Mac (nothing is sent to GitHub)
python -m pipeline mac-worker --self-test
```

The self-test downloads the models (once; this takes a while), makes one style still and one
edited frame from it, and prints how long each took. Look at the two pictures it names in
`~/mochizuki-mac/`. If either command fails, see "If something fails" below.

## Running it

```bash
cd mochizuki && source .venv/bin/activate && git pull
caffeinate -i python -m pipeline mac-worker
```

It checks for work every minute and keeps going until you press Ctrl-C. `caffeinate -i` stops the
Mac from sleeping while it runs; keep it on power. To take only what is waiting and stop, add
`--once`.

What you will see:

```
mac worker for Priyash/mochizuki: waiting for batches (Ctrl-C to stop)
[mac 10:02:11] claimed b20261124-...
[mac 10:05:40] ref-style.ref52... ok in 205 s [1/24]
...
[mac 11:31:02] b20261124-...: 24/24 made, results uploaded
```

To start work at once instead of waiting for the next scheduled producer run, start it from
GitHub: Actions → produce → Run workflow.

## 32 GB Macs: the 4-bit models

In `config.json`, under `mac.engines`, change the two `--model` values:

- stills: `mlx-community/Qwen-Image-2512-4bit`
- keyframes: a 4-bit Qwen-Image-Edit-2511 for mflux (for example from `fcreait/Qwen-Image-Edit-mflux`,
  downloaded to a folder, with `"--base-model", "qwen"` added to the command)

Then run the self-test again.

## Settings (`config.json` → `mac`)

| Setting | Meaning |
|---|---|
| `enabled` | the producer sends stills and keyframes to the Mac |
| `stages` | which stages go to the Mac (`refs`, `keyframes`) |
| `engines.refs.cmd`, `engines.keyframe.cmd` | the mflux command and its options (model, steps) |
| `batch_hours` | how much work goes into one batch |
| `max_wait_hours` | after this long untaken, a batch is dropped and its jobs freed |
| `stale_minutes` | a Mac that stops sending its heartbeat this long is given up on |

## If something fails

- **`mflux is not installed`:** activate the venv (`source .venv/bin/activate`) and `pip install -U mflux`.
- **An mflux error about an unknown option:** mflux changes its command line between versions.
  Run `mflux-generate-qwen --help` (or `mflux-generate-qwen-edit --help`) and adjust the command
  in `config.json`, then run the self-test again.
- **Out of memory, or the Mac grinds to a halt:** use the 4-bit models.
- **`could not tell the repository`:** run it inside the `mochizuki` folder after `gh auth login`,
  or pass `--repo Priyash/mochizuki`.

Nothing here costs money, and the worker never touches Kaggle, RunPod or the ledger.
