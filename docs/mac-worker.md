# Rendering on your Mac

> **Where files are:** the engine is `engine/`; each film is a folder in `films/` (this repository's is `films/mochizuki/`). Film paths below (`shots.csv`, `config.json`, `style_bible/`, `prompts/`, `assets/`, `media/`, `ledger.sqlite`) are inside the film's folder. `python -m engine where` prints it.

Your Apple Silicon Mac renders the film offline and for free: reference stills, keyframes,
video takes, and it trains the LoRAs that lock the film's look and its characters. Everything runs
with MLX, Apple's own machine-learning framework, natively on the M-series GPU.

| Job | Engine (default) | Model size | Licence |
|---|---|---|---|
| Reference stills | Z-Image Turbo (mflux) | ~20 GB | Apache 2.0 |
| Stills, 3D-leaning | Z-Image Base (`--engines zimage_base`) | ~20 GB | Apache 2.0 |
| Keyframes (image edit) | FLUX.2 [klein] 4B (mflux) | ~16 GB | Apache 2.0 |
| Video takes | Wan 2.2 TI2V-5B, 8-bit (mlx-video) | ~19 GB | Apache 2.0 |
| LoRA training | mflux-train on Z-Image Base | — | — |

Models download on first use into the Hugging Face cache or `~/mochizuki-mac/models`, with a
progress line every 30-60 seconds. Final gate renders can still go to Colab with Qwen-Image when
it has GPUs; the Mac is the always-available path.

## What you need

- Apple Silicon (M1 or later), macOS 14 or later, **32 GB memory or more**.
- About 80 GB free disk for all models.
- Python 3.12, git, the GitHub CLI and ffmpeg.
- Keep the Mac plugged in, on a hard surface, with heavy apps closed while it renders.

## One-time setup (Terminal)

```bash
brew install git gh python@3.12 ffmpeg
gh auth login                                  # GitHub.com, HTTPS, log in with the browser

cd ~/Documents/Projects/mochizuki              # your clone (or git clone https://github.com/Priyash/mochizuki.git)
git pull
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U mflux                            # stills, keyframes, training
pip install git+https://github.com/Blaizzy/mlx-video.git   # video (Wan 2.2)
```

## The checks

```bash
source .venv/bin/activate && git pull

caffeinate -i python -m engine mac-worker --self-test                     # a still and a keyframe
caffeinate -i python -m engine mac-worker --self-test --engines zimage_base   # the same, 3D-leaning engine
caffeinate -i python -m engine mac-worker --self-test --edit-only         # only the keyframe, from the last still
caffeinate -i python -m engine mac-worker --video-test                    # a 2-second take from the last still
caffeinate -i python -m engine mac-worker --video-test --seconds 4 --image ~/some.png
```

Pictures and clips go to `~/mochizuki-mac/` (`selftest-style.png`, `selftest-style-edit.png`,
`selftest-take.mp4`). Nothing is sent to GitHub. A failed step says which one and why.

## Training a LoRA (the film's look, a character)

A LoRA is a small add-on (~100-300 MB) that teaches the model one look or one face. Train one for
the 3D cel-shaded style, then one per main character.

1. Put 15-30 images of the look in a folder: approved stills, renders, frames you like. Varied
   subjects and framings, the same look. Optionally a caption beside each: `img07.png` + `img07.txt`.
2. Check the setup without training:
   ```bash
   python -m engine mac-worker --train-lora style3d --images ~/mochizuki-look --trigger mchz3d \
     --caption "3D CG anime film still, cel-shaded 3D models" --dry-run
   ```
3. Train (several hours on a 32 GB Mac; run it overnight):
   ```bash
   caffeinate -i python -m engine mac-worker --train-lora style3d --images ~/mochizuki-look --trigger mchz3d \
     --caption "3D CG anime film still, cel-shaded 3D models"
   ```

Progress shows as `train.step` lines; preview pictures appear in
`~/mochizuki-mac/train/style3d/training/` every few hundred steps, so you can stop early when the
look is right (Ctrl-C keeps the checkpoints). When it finishes, the LoRA is saved as
`~/mochizuki-mac/loras/style3d.safetensors` and listed in `~/mochizuki-mac/loras.json`: from then on
every still on this Mac uses it, with its trigger word in front of the prompt. Change `scale` or
`roles` in that file, or delete its entry to stop using it. Settings: `config.json` → `mac.train`.

## Working through batches

When `mac.enabled` is on in `config.json`, the producer on GitHub sends batches to the Mac:

```bash
caffeinate -i python -m engine mac-worker            # until Ctrl-C;  --once: what waits, then stop
```

It checks for work every minute, claims a batch, renders it (stills, keyframes or takes, each with
the right engine), sends a heartbeat every few minutes, and uploads the results. If no Mac takes a
batch within 72 hours, its jobs are freed again.

## Logs

Every line has the same shape: time, level, component, event, message, and the facts as `key=value`:

```
21:53:14 INFO  mac    render.start     rendering ref-style: text to image, 1664x928, 9 steps, z-image-turbo  role=refs loras=1
21:55:02 INFO  mac    render.step      ref-style: step 5/9, 14 s a step, about 1 min left  mem_free_pct=31 swap_gb=0.4 thermal=nominal
21:57:40 INFO  mac    video.done       selftest-take: saved ~/mochizuki-mac/selftest-take.mp4 (18.2 min)
```

Status lines warn when memory runs short, the Mac throttles for heat, or nothing has moved for 15
minutes. The same events are written as JSON lines to `~/mochizuki-mac/logs/worker-<date>.jsonl`:
send that file when something fails. `MOCHI_RAW=1` shows the tools' own output instead.

## Settings (`config.json` → `mac`)

| Setting | Meaning |
|---|---|
| `enabled`, `stages` | whether the producer sends batches to the Mac, and which stages |
| `engines.refs`, `.keyframe`, `.video` | the command for each job type: swap models here, not in code |
| `engines_<name>` | other engine sets, picked with `--engines <name>` |
| `loras` | LoRAs for every Mac (this Mac's own are in `~/mochizuki-mac/loras.json`) |
| `train` | LoRA training: base model, total steps, rank, learning rate, memory options |
| `batch_hours`, `max_wait_hours`, `stale_minutes` | batch size and patience |

## If something fails

- **`... is not installed`:** activate the venv and run the matching `pip install` above.
- **A Hugging Face "gated" or 401 error:** open the model's page on huggingface.co, accept its
  terms, then run `hf auth login` once.
- **Out of memory, or `memory is tight` warnings:** close other apps; for video try
  `--seconds 2`; for training lower `mac.train.max_resolution` to 768.
- **An unknown-option error:** the tools change their options between versions; send the log line.

## How the code is laid out (`engine/mac/`)

| Module | Its one job |
|---|---|
| `config.py` | settings and engine sets |
| `common.py` | structured log, seeds, paths, ffmpeg |
| `sysinfo.py` | what the Mac has, and how it is coping |
| `models.py` | model downloads with progress, and checks before use |
| `process.py` | running a model tool and turning its output into log lines |
| `engines/` | `image.py` (mflux), `video.py` (mlx-video), `base.py` (checks, stand-ins), router in `__init__.py` |
| `worker.py` | claim a batch, render it, upload results |
| `selftest.py` | the checks above |
| `train.py` | LoRA training |
| `backend.py` | the producer's side of the Mac backend |
| `cli.py` | the `mac-worker` command |

The producer and the Mac talk only through the `batches` release (`engine/providers/release.py`).
