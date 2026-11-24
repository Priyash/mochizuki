# Mochizuki

A ten-minute anime short, produced by a pipeline that runs on free GPUs and finishes in 4K.

The story runs in five chapters, from a rice-pounding night on the Noto coast to a lunar colony. The look is **full 3D CG anime**: cel-shaded characters and sets, clean ink outlines and painterly light. The pipeline turns the shot sheet (`shots.csv`, 138 shots, 600 s) into generation jobs. It runs them in batches on Kaggle's free GPUs, then cuts the chapter at 3840×2160.

It costs no money. It does cost time: the free GPU quota is about 30 hours a week.

## How a shot is made

| Shot type | Count | Model | Native size | Takes |
| --- | --- | --- | --- | --- |
| Hero (faces, action, dialogue) | 56 | Wan 2.2 I2V A14B, GGUF on a T4 | 1280×720, 16 fps | up to 4 |
| Simple (inserts, landscapes in motion) | 27 | Wan 2.2 TI2V 5B | 1280×704, 24 fps | up to 3 |
| Still (held frames; pushes and particles added in the edit) | 55 | none | keyframe | 0 |

1. **Keyframes:** Qwen-Image-Edit-2511 draws 3 candidates per shot from your reference stills. A shot gets a second set of 3 only if all three fail.
   - **Scenes keep it coherent.** `style_bible/scenes.json` groups the 138 shots into 20 scenes. Each scene's anchor keyframe is drawn and chosen first; every other shot in the scene is then drawn from it, so set, light, costume and time of day carry through. A scene that returns to an earlier place (the hearth, the porch, the ruined farmhouse) draws its anchor from that earlier scene, and a match cut ("lights become sparks") is drawn from the shot before it.
   - **Characters age, not redesign.** Tsumugi at 27 and 87, Sayo very old, the bowl's later seams and Akari's forms are painted as edits of the approved earlier still, and each shot uses the version for its chapter.
2. **Board (human gate):** you pick one keyframe per shot and approve the chapter.
3. **Takes:** the video model animates the chosen keyframe, one take at a time. A retake runs only after the previous take has failed review. Shots longer than one model clip (about 5 s on the 14B model) are made as chained clips, each starting on the last frame of the one before.
4. **Animatic (human gate):** a 640×360 cut of the chapter, which you approve.
5. **4K:** takes are upscaled with Real-ESRGAN `realesr-animevideov3`, which keeps frames steady so the picture doesn't flicker. Keyframes are upscaled with Real-CUGAN 3x. Both are cropped to exactly 3840×2160.
6. **Edit:** DaVinci Resolve (free) for the cut, grade, sound and music. Dialogue is recorded after the picture (afureko), and actors match the mouths.

Every model used is licensed for publishing your film: Wan 2.2 and Qwen-Image-Edit under Apache 2.0, Real-ESRGAN under BSD-3, Real-CUGAN under MIT.

### About the 4K

The 4K is upscaled from 720p, not rendered at 4K. No free model makes native 4K video today. A clean 720p source upscaled by an anime-trained model looks sharp on a 4K screen. Fine texture will be softer than in a native 4K render. If that matters for a festival print, route your best Hero shots to a paid model instead (see "Paid fallback").

## What has and has not been tested

| Part | State |
| --- | --- |
| Ledger, gates, take and failure caps, no double-making, resume | Tested |
| Kaggle batch packing, weekly GPU cap, session budget, stale-dataset guard | Tested |
| Kaggle CLI output parsing | Tested against recorded output, not a live account |
| Whole loop to a 4K cut, using a local stand-in for Kaggle and stub engines | Tested. Gives 3840×2160 at full chapter length |
| GitHub producer loop: Claude's reviews, owner-only gates, media in releases | Tested with stand-ins for Kaggle and `gh`; the workflows themselves run on first push |
| **GPU engines (Qwen, Wan 14B and 5B, the upscalers) on a real T4** | **Not run yet.** The first Kaggle run is the test |
| GPU-time estimates in `config.json` | My estimates, until real runs replace them |
| The `fal` paid fallback | Written but never run; endpoint ids are placeholders |

`python -m pytest -q` runs 40 tests in about 3 min. They need ffmpeg and nothing else.

## Setup

You need Python 3.10+, ffmpeg and a free Kaggle account.

1. On kaggle.com, verify your phone number in your account settings. GPUs and internet in notebooks need it.
2. In account settings, under API, create a new token. Then either put the token in `KAGGLE_API_TOKEN` in `.env`, or save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`.
3. Put your Kaggle username in `config.json` → `kaggle.username`, or set `KAGGLE_USERNAME` in `.env`.
4. Install the CLI: `pip install -r requirements.txt`.
5. Check it: `kaggle quota -v` should show your GPU hours.
6. Put reference stills in `style_bible/refs/` and approve the look (see `style_bible/README.md`).

```bash
cp .env.example .env      # fill it in; .env is never committed
python -m pipeline plan   # dry run: GPU hours per shot, money (always $0 on the free plan)
```

## First real run: three test shots

Before you commit a week of quota, test one keyframe and one take of each kind on a real T4:

```bash
python -m pipeline batch --chapter 1 --stage keyframes --shot 1-03
python -m pipeline submit
python -m pipeline collect          # rerun until it reports done (a batch takes 20-60 min)
python -m pipeline queue --chapter 1
```

Repeat for `1-04` (Hero, 6 s, two chained clips) and `1-02` (Simple, 5B). Look at the files in `media/`. If a GPU engine fails, the reason is in `media/batches/<id>/download/out/results.json`. After a real run, compare the `gpu_seconds` it records with the estimates in `config.json`, and correct them so that `plan` and the batch sizing are accurate.

## Running it automatically

`auto` takes one step each time it is called. It collects a finished batch, or packs and submits the next one. It stops when a person is needed. It never approves a gate and never raises a cap.

```bash
python -m pipeline auto --chapter 1 --loop        # runs until the chapter needs you
```

Or from cron, every 30 minutes:

```cron
*/30 * * * * cd /path/to/mochizuki && python -m pipeline auto --chapter 1 >> auto.log 2>&1
```

Where a person (or a review agent following `AGENTS.md`) steps in:

```bash
python -m pipeline queue --chapter 1                 # what needs a verdict
python -m pipeline verdict <key> pass --note "on model"
python -m pipeline select <key>                      # one keyframe per shot
python -m pipeline approve --chapter 1 board --by human
python -m pipeline assemble --chapter 1              # media/animatic/chapter_1.mp4
python -m pipeline approve --chapter 1 animatic --by human
python -m pipeline assemble --chapter 1 --4k         # media/animatic/chapter_1_4k.mp4
python -m pipeline status --chapter 1                # states, 4K progress, GPU hours this week
```

Kaggle rules the pipeline works inside:

- Sessions are 12 h at most. Batches are sized to 10 h, including about 25 min for setup.
- The free GPU quota is about 30 h a week and resets on Saturday at 00:00 UTC. `weekly_gpu_hours` (default 28) keeps a stuck run from using it all.
- One batch runs at a time.

Try the loop for free on your own machine, using stub engines in place of the GPU:

```bash
KAGGLE_USERNAME=local python -m pipeline auto --chapter 1 --loop --kaggle local --engine-set stub
```

## Letting Claude drive it (GitHub Actions)

Here, nothing runs on your machine. The work is split three ways:

- **The producer** is the `produce` workflow. It runs every two hours, or straight after a review or a command. Each run:
  - applies Claude's verdicts;
  - collects a finished Kaggle batch or starts the next one;
  - uploads new media to this repo's releases (`media-chN`, `4k-chN`, `cuts`);
  - exports the next review to the `state` branch;
  - posts what is needed to the **Mochizuki production** issue.
- **The reviewer** is a Claude scheduled task. It follows the "Reviewer run" in `AGENTS.md`: it looks at the review images, picks keyframes, passes or fails takes, pushes its verdicts and starts the producer.
- **You** approve the two gates by commenting on the issue. Only comments by the repository owner are obeyed:

| Comment | What it does |
| --- | --- |
| `/approve refs` | Approves the reference stills Claude picked; they are committed to `style_bible/refs/` and keyframes start |
| `/reject <name>` | Rejects one picked reference still (for example `/reject sayo`); another is picked or painted |
| `/approve board N` | Approves chapter N's keyframes; takes start |
| `/approve animatic N` | Approves the cut; the 4K pass starts |
| `/chapter N` | Moves the producer to chapter N |
| `/status` | Replies with where the chapter stands |

Setup, once:

1. Settings → Secrets and variables → Actions → New repository secret. Add `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN`. If you have an old-style `kaggle.json` key, add it as `KAGGLE_KEY` instead of `KAGGLE_API_TOKEN`.
2. Reference stills: put your own in `style_bible/refs/` on `main`, or let Kaggle paint them. For every still listed in `style_bible/ref_prompts.json` that is missing, the producer paints 4 candidates with Qwen-Image-2512 (Apache 2.0), Claude picks the best, and you approve them on the issue.
3. Actions → produce → Run workflow, to start at once rather than wait up to two hours.

What it uses:

- GitHub Actions minutes: each run takes 1 to 3 minutes, and runs about every two hours. That is a few hundred minutes a month, inside GitHub Free's allowance for private repositories.
- Release storage: the media of the whole film is several GB, and GitHub doesn't charge for release files.

Its limits:

- Claude reviews still frames and can't watch motion, so watch every Hero take yourself before you approve the animatic.
- The first real Kaggle run is still the test of the GPU models.

## Where each stage runs

| Stage | Where | Why |
|---|---|---|
| Reference stills, keyframes | **your Mac** (`python -m pipeline mac-worker`, see [docs/mac-worker.md](docs/mac-worker.md)) | Qwen-Image needs bf16; Kaggle's T4s only have fp16 and give blank images |
| Video, 4K upscale | **Kaggle** T4s (free, 30 GPU h/week) | |
| Video, when Kaggle is stuck or slow | **RunPod** (prepaid, capped by `runpod.total_cap_usd`) | optional, off until you load credit |

Each backend takes only the stages in its `config.json` section (`kaggle.stages`, `mac.stages`, `runpod.stages`).

## How long it takes

Estimated GPU time on the free plan, before real runs correct it:

| Chapter | GPU hours (first takes pass → every take used) |
| --- | --- |
| 1 | 23 – 74 |
| 2 | 13 – 39 |
| 3 | 20 – 64 |
| 4 | 31 – 102 |
| 5 | 19 – 62 |
| Film | **106 – 340 h**, about 4 to 12 weeks of free quota |

My guess for the likely case, about two takes per moving shot, is about 180 h: 6 to 7 weeks. To finish faster, run two Kaggle accounts from different people in parallel (each must be a real person; Kaggle forbids multiple accounts for one person). Or rent a GPU for the Hero shots only.

## Paid fallback

If the first test shows a Hero shot that Wan 2.2 can't do well enough, change that shot's model to `Kling 4` in `shots.csv` and set a money cap in `config.json → budget`. Kaggle batches skip shots routed to fal. `python -m pipeline video --chapter N --provider fal` runs them. Before any paid run, check the endpoint ids in `config.json → providers.fal`: they are placeholders.

## Layout

```
config.json          models, engines, Kaggle limits, caps
shots.csv            the shot sheet: 138 shots, 600 s
style_bible/         style.txt and reference stills (gate 1)
prompts/             optional per-shot prompt files
pipeline/            the producer: ledger, steps, Kaggle batches, assemble
worker/worker.py     runs on Kaggle: reference stills, keyframes, Wan takes, 4K upscale
.github/workflows/   tests, the producer (produce.yml) and the owner's commands (gates.yml)
tests/               40 tests, no GPU or network needed
AGENTS.md            rules for an AI agent running the loop
```

Generated media, the ledger and `.env` are not committed.
