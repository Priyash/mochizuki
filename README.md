# Mochizuki

*A ten-minute anime short, made by a pipeline that runs on free GPUs and finishes in 4K.*

Five chapters, from a rice-pounding night on the Noto coast to a lunar colony. The look is **full 3D CG anime**: cel-shaded characters and sets, clean ink outlines, painterly light.

| | |
| --- | --- |
| **Film** | 600 s, 138 shots in 5 chapters (`shots.csv`) |
| **Picture** | 3840×2160 |
| **Money** | Free GPUs (Kaggle, GitHub, your Mac). Colab units only for reference stills, inside the cap you set |
| **Time** | Kaggle's free quota is about 30 GPU hours a week |
| **You** | Own every creative decision. The pipeline stops at each gate and waits for you |

---

## How a shot is made

There are two paths. The 3D path is the main one; the generative path is kept for design work and as an option.

### 1. The 3D CG path (main)

Characters, props and sets are built **once**. Each shot gets an **acting sheet** (camera, beats, gaze, timing), and Blender's real-time renderer draws every frame natively in 4K. Shots stay coherent because they share one set of models. Full guide: **[docs/cgi-pipeline.md](docs/cgi-pipeline.md)**.

```bash
python -m engine cgi check
python -m engine cgi render 1-02 --preview          # the mallet falls into the mortar
python -m engine cgi render --chapter 1 --board     # chapter 1 as a storyboard from the real sets
python -m engine cgi auto --chapter 1               # unattended: voices, renders, board, cut
python -m engine cgi continuity --chapter 1         # the 180 and 30 degree rules, eyelines, light, props at every cut
python -m engine cgi arc --chapter 1                # each shot's feeling against the chapter's intended emotional arc
python -m engine cgi culture --chapter 1            # what the culture pack's display rules change in the acting
```

- **Characters from code:** anime bodies, faces and hair for each age, so no modelling is needed.
- **Dialogue, automated:** every line voiced in English and Japanese (Kokoro-82M), fitted into its shot, lip-synced and subtitled. One audio track per language.
- **Continuity supervisor:** every cut checked before rendering, from the shots' plans: the 180° and 30° rules, eyelines, light, props and positions. A deliberate break is accepted in the acting sheet, with its reason.
- **Feeling, by design:** an intended emotional arc per chapter (Russell's circumplex, Reagan's six story shapes) that drafted shots follow and every shot is measured against, and a culture pack of display rules (who shows what, before whom) played into the acting.
- **Dubbing:** each language has its own lips. Speaking shots are rendered once per language, each chapter gets a dubbed cut (`chapter_1.ja.mp4`), and every line's sync is checked against broadcast limits (ITU-R BT.1359).
- **Unattended:** GitHub's runners (`.github/workflows/cgi.yml`) remake a chapter whenever its sheets or lines change.
- The cut uses these renders when `cgi.use_in_cut` is on in `config.json`.

### 2. The generative path

| Shot type | Count | Model | Native size | Takes |
| --- | --- | --- | --- | --- |
| Hero (faces, action, dialogue) | 56 | Wan 2.2 I2V A14B, GGUF on a T4 | 1280×720, 16 fps | up to 4 |
| Simple (inserts, landscapes in motion) | 27 | Wan 2.2 TI2V 5B | 1280×704, 24 fps | up to 3 |
| Still (held frames, moved in the edit) | 55 | none | keyframe | 0 |

1. **Keyframes:** Qwen-Image-Edit-2511 draws 3 candidates per shot from the reference stills. Scenes (`style_bible/scenes.json`) keep it coherent: each scene's anchor is chosen first, and the other shots are drawn from it. Characters age as edits of the approved earlier still, never as a redesign.
2. **Board** *(your gate)*: one keyframe per shot; you approve the chapter.
3. **Takes:** Wan animates the chosen keyframe. A retake runs only after a take fails review. Long shots are chained clips, each starting on the last frame of the one before.
4. **Animatic** *(your gate)*: a 640×360 cut you approve.
5. **4K:** Real-ESRGAN `realesr-animevideov3` for takes (steady, no flicker), Real-CUGAN 3x for keyframes, both cropped to 3840×2160. This is upscaled from 720p, so fine texture is softer than a native render.
6. **Edit:** DaVinci Resolve (free) for the grade, sound and music.

Every model is licensed for publishing the film: Wan 2.2 and Qwen-Image-Edit (Apache 2.0), Real-ESRGAN (BSD-3), Real-CUGAN (MIT).

---

## The chain

Every stage of making the film, in one place (`engine/stages/chain.py`): what it consumes, what it produces, the approval it waits for, who runs it, and how it skips work that has not changed. See where the film stands:

```bash
python -m engine chain               # every chapter, stage by stage
python -m engine chain --chapter 1   # one chapter
```

Planned stages (music) are listed and marked not built.

---

## Where each stage runs

| Stage | Where | Notes |
| --- | --- | --- |
| 3D shots: renders, dialogue and dubs, the chapter cuts | GitHub runners or your Mac | Native 4K, no GPU queue |
| Reference stills | Google Colab, A100 or L4, pay-as-you-go ([docs/colab.md](docs/colab.md)) | Qwen-Image needs bf16; Kaggle's T4s give blank images |
| Keyframes | Off since 28 Nov | The 3D pipeline draws the storyboard from the real sets. Add `keyframes` back to `colab.stages` to use them |
| Stills and keyframes, free but slower | Your Mac ([docs/mac-worker.md](docs/mac-worker.md)) | Off until you set `mac.enabled` |
| Video takes, 4K upscale | Kaggle T4s, free | About 30 GPU hours a week |

Each backend takes only the stages listed in its section of `config.json` (`kaggle.stages`, `colab.stages`, `mac.stages`).

---

## Setup

You need Python 3.11+, ffmpeg and a free Kaggle account. Paths in this section are inside the film's folder, `films/mochizuki/`; commands pick it up by themselves (choose another with `--film` or `FILM`).

1. On kaggle.com, verify your phone number (GPUs and notebook internet need it).
2. Under **Settings → API**, create a token. Put it in `KAGGLE_API_TOKEN` in `.env`, or save `kaggle.json` to `~/.kaggle/`.
3. Set your Kaggle username in `config.json → kaggle.username`, or `KAGGLE_USERNAME` in `.env`.
4. `pip install -e ".[kaggle]"`, then check `kaggle quota -v` shows your GPU hours.
5. Put reference stills in `style_bible/refs/` and approve the look (see [style_bible/README.md](style_bible/README.md)).

```bash
cp .env.example .env      # fill it in; .env is never committed
python -m engine check    # the film's config.json and shots.csv: every problem at once
python -m engine plan   # dry run: GPU hours per shot, and money
```

---

## Running it

### On your machine

`auto` takes one step each time: it collects a finished batch, or packs and submits the next. It stops when a person is needed, and it never approves a gate or raises a cap.

```bash
python -m engine auto --chapter 1 --loop        # runs until the chapter needs you
```

Or from cron: `*/30 * * * * cd /path/to/mochizuki && python -m engine auto --chapter 1 >> auto.log 2>&1`

Where a person (or a review agent following `AGENTS.md`) steps in:

```bash
python -m engine queue --chapter 1                 # what needs a verdict
python -m engine verdict <key> pass --note "on model"
python -m engine select <key>                      # one keyframe per shot
python -m engine approve --chapter 1 board --by human
python -m engine assemble --chapter 1              # media/animatic/chapter_1.mp4
python -m engine approve --chapter 1 animatic --by human
python -m engine assemble --chapter 1 --4k         # media/animatic/chapter_1_4k.mp4
python -m engine status --chapter 1                # states, 4K progress, GPU hours this week
```

Kaggle's limits: sessions last 12 h at most, so batches are sized to 10 h including setup. The quota resets on Saturday at 00:00 UTC, and `weekly_gpu_hours` (default 28) stops a stuck run from using all of it. One batch runs at a time.

Try the whole loop for free, with stub engines in place of the GPU:

```bash
KAGGLE_USERNAME=local python -m engine auto --chapter 1 --loop --kaggle local --engine-set stub
```

### On GitHub, with Claude reviewing

Nothing runs on your machine.

- **The producer** (`produce` workflow, every two hours or when kicked) applies Claude's verdicts, collects or starts a batch, uploads media to releases, and posts what is needed to the **Mochizuki production** issue.
- **The reviewer** is a Claude scheduled task following the "Reviewer run" in `AGENTS.md`.
- **The 3D renderer** (`cgi` workflow) remakes a chapter whenever its sheets, lines, arc or culture pack change. It uploads the board, a cut per language, the subtitles and a review sheet to the release `cgi-chN`. The review sheet holds each line's lip sync check, the continuity findings at every cut and each shot's feeling against the arc. Claude reviews it by the "CG review" in `AGENTS.md`.
- **You** approve the gates by commenting on the issue. Only the repository owner's comments count:

| Comment | What it does |
| --- | --- |
| `/approve refs` | Makes the picked reference stills the film's look |
| `/reject <name>` | Rejects one picked still (e.g. `/reject sayo`); a new one is painted |
| `/approve board N` | Approves chapter N's keyframes; takes start |
| `/approve animatic N` | Approves the cut; the 4K pass starts |
| `/chapter N` | Moves the producer to chapter N |
| `/status` | Replies with where the chapter stands |

Setup, once: add the repository secrets `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` (or `KAGGLE_KEY` for an old-style key), and `COLAB_ADC_JSON` for Colab ([docs/colab.md](docs/colab.md)). Then **Actions → produce → Run workflow** to start at once. Missing reference stills listed in `style_bible/ref_prompts.json` are painted for you (4 candidates each), Claude picks one, and you approve.

It fits inside GitHub Free: a few hundred Actions minutes a month, and release storage isn't charged. Claude judges still frames and can't see motion, so **watch every Hero take yourself** before you approve an animatic.

---

## First real run: three test shots

Before spending a week of quota, run one take of each kind on a real T4, once each shot has a chosen keyframe: `1-03` (Hero, 14B), `1-04` (Hero, 6 s, two chained clips) and `1-02` (Simple, 5B).

```bash
python -m engine batch --chapter 1 --stage video --shot 1-03
python -m engine submit
python -m engine collect          # rerun until it reports done
python -m engine queue --chapter 1
```

If an engine fails, the reason is in `media/batches/<id>/download/out/results.json`. Then compare the `gpu_seconds` it recorded with the estimates in `config.json` and correct them, so `plan` and batch sizing are accurate.

---

## The AI finish

The finish repaints a shot's 3D render in the look of the approved reference stills, guided by the render's
depth (or its lines) so the camera, staging, timing and lip sync stay exactly as rendered. It uses Wan VACE,
configured in the film's `config.json` under `engines.ai_finish`. It never runs on its own until you set
`auto` there, and finished shots go into the cut only when you set `use_in_cut`.

The finish paints its own look (`style` in `engines.ai_finish`, in place of `style_bible/style.txt`, which
describes the toon render), from the shot's subjects among the approved stills: never the style scene, and
each character sheet cut to its front view (`crops`), with a line saying who it shows (`who`). The guides are
sharp, on black: depth as disparity (the nearest surface white), lines white on black.

Two settings: `ai_finish` is the quick check (Wan2.1-VACE-1.3B at 832×480, 30 steps); `ai_finish_720` is the
film's (the 14B with the lightx2v 4-step distillation, as a GGUF, at 1280×720). On a 16 GB T4 the 14B waits in
the machine's memory and goes to the GPU a block at a time (`offload: group`).

The first test, one shot (1-05, the laugh), outside the ledger: Actions > **finish-test** > Run workflow, or

```bash
# renders the shot and its guides, sends one finish per guide to a Kaggle T4, and puts each finish beside
# its guide and the 3D render on the release finish-tests (with a middle frame beside the stills)
python -m engine cgi render 1-05 --size 1280x720 --guides
python -m engine finish-test 1-05 --engine ai_finish            # then --engine ai_finish_720
```

Several engines (`--engine ai_finish,ai_finish_720`) run in one GPU session. `--backend colab` runs the same
test on Colab with your units, only when you have said yes: the inputs wait in the `batches` release for the
scheduled Colab runner, and are taken down if no runner takes them within 75 minutes.

Its summary gives the GPU seconds per second of film, which prices the whole film. A finish for the film
itself goes through the ledger: `python -m engine batch --chapter 1 --stage ai_finish --shot 1-05`, then
`submit` and `collect`; it lands in `films/mochizuki/media/finished/1-05/`.

---

## What is tested

| Part | State |
| --- | --- |
| Ledger, gates, take and failure caps, resume | Tested |
| Kaggle batch packing, weekly cap, session budget, stale-dataset guard | Tested |
| Whole loop to a 4K cut, with a stand-in for Kaggle and stub engines | Tested |
| GitHub producer loop and the owner's commands | Tested with stand-ins, and running live |
| Colab round trip | Run live: 24 reference stills on an A100 (27 Nov) |
| 3D pipeline: sheets, plans, dialogue, renders | Tested, and rendered in Blender 5.3 |
| Lip sync and dubbing | Tested on synthetic speech with known timing, and on real English and Japanese speech (macOS voices): every line inside the broadcast limits. **Not yet heard with Kokoro**, which runs on the GitHub runners |
| Continuity supervisor, emotional arc, culture pack | Tested, and run on all five chapters |
| **Wan 14B, Wan 5B and the upscalers on a real T4** | **Not run yet.** The first Kaggle run is the test |
| GPU-time estimates in `config.json` | Estimates, until real runs replace them |
| AI finish on hero shots | Built and tested with stand-ins; **not run on a GPU yet** (the 1-05 test, `finish-test`, is next) |
| The `fal` paid fallback | Written, never run; endpoint ids are placeholders |

`pip install -e ".[dev]"` installs the checks CI runs on every pull request (Python 3.11 and 3.12):

```bash
ruff check .          # lint
mypy                  # types (the Blender scripts, which run inside Blender, are left out)
python -m pytest -q   # 270 tests in about 3 minutes, no GPU or network; they need ffmpeg
```

---

## How long it takes

Estimated Kaggle GPU time, before real runs correct it:

| Chapter | GPU hours (first takes pass → every take used) |
| --- | --- |
| 1 | 23 – 74 |
| 2 | 13 – 39 |
| 3 | 20 – 64 |
| 4 | 31 – 102 |
| 5 | 19 – 62 |
| **Film** | **106 – 340 h**, about 4 to 12 weeks of free quota |

The likely case (about two takes per moving shot) is around 180 h, or 6 to 7 weeks. To go faster, add a second Kaggle account owned by a different real person (Kaggle forbids several accounts per person), or rent a GPU for the Hero shots.

---

## Paid fallback

If the first test shows a Hero shot Wan 2.2 can't do well enough, set that shot's model to `Kling 4` in `shots.csv` and a money cap in `config.json → budget`. Kaggle batches skip shots routed to fal; `python -m engine video --chapter N --provider fal` runs them. Check the endpoint ids in `config.json → providers.fal` first: they are placeholders.

---

## Layout

The engine holds no film. Each film is a folder in `films/`, so the same engine makes the next film too
(see [films/README.md](films/README.md)).

```
engine/                  the engine (python -m engine): film, ledger, review, cut
engine/stages/           the chain: every stage, what it consumes and produces, its gate
engine/generators.py     every AI model and tool: what it makes, where it is configured, what it needs
engine/scheduler/        the producer loop, and one batch's life: build, submit, collect
engine/providers/        where GPU work runs, behind one interface: kaggle/, colab/ (cloud), mac.py (offline)
engine/cgi/              the 3D CG pipeline: acting/ (sheets, arc, culture), staging/, plan, continuity, render/
engine/audio/            sound: voice, dialogue, lip sync per language, subtitles, mix
engine/mac/              the Mac (offline): generators/, runtime/, worker/, training/
worker/worker.py         runs on Kaggle and Colab: stills, keyframes, Wan takes, 4K upscale
films/mochizuki/         this film
  shots.csv              the shot sheet: 138 shots, 600 s
  config.json            models, engines, backends, caps, the cast
  style_bible/           the look: style, reference stills, acting sheets, sets, lights, dialogue, arc, culture
  prompts/               optional per-shot prompt files
  assets/characters/     your own character models (never committed)
  media/                 everything the engine makes, and the ledger (never committed)
docs/                    guides: the 3D pipeline, Colab, the Mac worker
.github/workflows/       tests, the producer, the owner's commands, Colab runs, 3D renders
tests/                   no GPU or network needed
AGENTS.md                rules for an AI agent running the loop
```

Generated media, the ledger and `.env` are never committed.
