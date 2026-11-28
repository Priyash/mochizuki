# The 3D CG pipeline: build it once, perform it, render it

> **Where files are:** the engine is `engine/`; each film is a folder in `films/` (this repository's is `films/mochizuki/`). Film paths below (`shots.csv`, `config.json`, `style_bible/`, `prompts/`, `assets/`, `media/`, `ledger.sqlite`) are inside the film's folder. `python -m engine where` prints it.

This is the new direction for making Mochizuki: a real 3D CG anime pipeline that runs on your Mac.
It replaces "generate every shot with a video model" as the way the film's moving picture is made.
The generative models stay, but in the jobs they are good at.

It is written in `engine/cgi/` (plain Python, tested) and `engine/cgi/render/blender/` (runs inside Blender).

## Why change direction

The current pipeline makes each shot from scratch: Qwen paints keyframes from reference stills, and Wan 2.2 animates
each keyframe from a text prompt. A review of the code and of every test so far shows three limits that more GPU time
will not fix:

| What we need | What the generative pipeline gives | Why |
| --- | --- | --- |
| **Emotion** | Generic or wrong acting. The key-art tests came out with raised arms, crying when it should have been held in, and a third person in a two-person scene. | A prompt cannot direct a performance: when a look happens, how long it is held, how much is held back. |
| **Coherency** | Each shot re-invents the face, the costume, the set and the light, held together only by reference images. Long shots are chained 5 s clips that drift at the joins. | Nothing persists between shots. A video model sees one clip at a time. |
| **Speed** | 106 to 340 T4 GPU-hours for the film (README). About 30 minutes per 3.4 s clip on the Mac (budget in `config.json`, not yet measured). Up to 4 takes per hero shot. Native 720p, upscaled to 4K. Colab has granted no GPU in every try since 26 Nov. | Diffusion video is slow. Retakes are a lottery: a retake is a new random performance, not a fix. |

The research says the same. Keeping characters and places consistent across shots is still an active research problem
(HoloCine, CVPR 2026; VideoMemory, 2026), and the best results need 14B-parameter models on data-centre GPUs.
Anime-specific work (AniMatrix, 2026) finds that video models trained on real footage flatten anime's conventions.
And Alibaba's newer Wan versions (2.5 and later) are API-only, so Wan 2.2 is still the newest open Wan.

## The direction

Make the film the way 3D CG anime studios do. Build each character, prop and set **once**. Give each shot an **acting sheet**.
**Perform** the emotional shots. Let a real-time renderer draw every frame the same way every time. Use AI to make the
assets, to read human performances, and later to add a painterly finish to hero shots.

```
 shots.csv ──► acting sheet ──────────────┐   (camera, beats, restraint, gaze, timing; you approve it)
 style_bible/acting.json                  │
                                          ▼
 iPhone take (Live Link Face) ──► capture ──► retarget ──► face curves ─┐
 or any video (MediaPipe)        52 ARKit shapes   + anime layer       │
                                                                        ▼
 sets.json + lights.json ───────────────────────────────────────►  PLAN (one JSON per shot)
 VRoid models (assets/characters) ─┐                                    │
 props from code (bowl seams = data)                                    ▼
                                   └──────────────────────────► Blender 5 EEVEE: toon shading,
                                                                ink lines, native 4K, .blend per shot
                                                                        │
                                              (phase 3, GPU) optional AI finish on hero shots
                                                                        ▼
                                                          assemble ──► Resolve
```

### How each requirement is met

**Deeply emotional.** Emotion is directed, layer by layer, instead of being hoped for:

| Lever | Where it lives | What it does |
| --- | --- | --- |
| Performance | `cgi capture` | A human acts the shot. The iPhone records 52 face shapes at 60 fps. They are smoothed (One Euro filter), mapped onto the character and exaggerated 15%, because a real face under-reads on an anime face. |
| Beats | acting sheet | When each feeling arrives, who has it, how strong it is, and how long it is held. |
| Restraint | acting sheet | Held-in feeling is played the anime way: the mouth and brows move less, the eyes do more (wet eyes, bigger highlights, more blinks). Sayo's lines in 1-11 and 1-24 are written like this. |
| Eyes and blinks | `timing.blinks` | A blink lands on each change of thought. Slow blinks come with tenderness and awe, more blinks with fear or held-in tears. |
| Gaze and eyelines | `plan.gaze_curves` | The eyes lead and the head follows. Someone off screen can still be looked at from their mark, so eyelines match across cuts. |
| Timing | `timing` | Pose to pose, with key poses found by Ramer–Douglas–Peucker. Characters move on twos, fast moves on ones, and the pose is held for three frames where a hit lands. |
| Camera | `layout` | Framing is fixed in metres of subject, so a CU is a CU at any lens. Emotional close-ups push in. Two-shots favour the first character named. A prop in front of a face is cheated out of the shot, as a camera crew would. |
| Light | `style_bible/lights.json` | Every scene has its own rig. The toon shader keeps the colour of the light, so the lantern reads warm and the moon reads cool on the same face. |

Chapter 1's acting sheets are drafted in `style_bible/acting.json`. They cover every hero shot and the key inserts. Each
one says why the beat is played that way. All are `draft` until you change them to `approved`.

**Coherent by construction.** Every shot of a scene loads the same set (`style_bible/sets.json`), props, marks and light.
Continuity is data: the bowl has one gold seam until 3-06 mends the second crack (`cgi.props_state`). Each character is
one model per age, and the same model appears in every shot. Each shot is saved as a `.blend` file you can open and adjust
by hand.

**Super fast.** EEVEE is Blender's real-time renderer. On the Mac's GPU it should draw a toon frame in around a second
(an estimate), and the animatic plays back live in the viewport. The film is 14,400 frames at 24 fps. At an estimated 0.5–3 s per native 4K frame on the M2 Max,
that is roughly 2–12 hours for the whole film, unattended, with no takes. A re-render is the same shot with your change,
not a new random performance. `cgi render --chapter 1 --board` makes a storyboard of the whole chapter from the real sets
in minutes.

> These Mac numbers are estimates until measured. The first run of `cgi render 1-02 --preview` logs `seconds_per_frame`.
> In this cloud sandbox, which has no GPU, Blender's software renderer took about 8 s per 640×360 frame. That was enough
> to test everything, but it says nothing about the Mac's speed.

## What is built now (phase 1)

| Part | Module | State |
| --- | --- | --- |
| Acting sheets: rules that draft all 138 shots, hand-written chapter 1, validation | `direction.py` | Tested. Every shot's sheet validates |
| Camera layout: framing, lens, angle, push/pull/pan/tilt/track/drift, cheating props | `layout.py`, `plan.py` | Tested |
| Anime timing: key poses, twos/threes/mixed, hit stops, emotion curves, blinks | `timing.py` | Tested |
| Face capture: Live Link Face CSV, MediaPipe video, One Euro, resampling | `capture.py` | CSV path tested. MediaPipe path written, to be run on the Mac |
| Retarget: ARKit → VRoid shapes, restraint, anime layer, learned per-character mapping, Akari's light | `retarget.py` | Tested |
| Shot plans for the whole film | `plan.py` | Tested on all 138 shots |
| Blender builder: toon shading that keeps light colour, ink lines at constant screen width, sets and props from code, stand-in actors whose faces act, VRM models posed (arms down, kneeling) and driven by the plan | `blender/` | Run in Blender 5.3 here, including a real VRM model through the VRM add-on |
| Runner: Blender or a bpy Python, structured logs, frame checks, H.264 or ProRes, storyboard | `render.py`, `cli.py` | Tested, plus one real-Blender test |
| The cut uses CG renders when `cgi.use_in_cut` is on | `assemble.py` | Tested |
| Characters made from code: anime bodies, faces and hair for each age (Tsumugi at 7, 27 and 87, Sayo, the child), playing the same face channels as a VRoid model | `figure.py`, `blender/anime.py` | Tested, and rendered in Blender 5.3 here |
| Dialogue: the script in English and Japanese, voices (Kokoro-82M, or any engine by command), lines fitted into their shot, lip sync, one track per language, subtitles | `dialogue.py`, `voice.py`, `lipsync.py` | Tested with the stub voice. Kokoro runs on the GitHub runners |
| The chapter unattended: voices, sharded renders that skip unchanged shots, board, cut, review sheet | `auto.py`, `.github/workflows/cgi.yml` | Tested |

Sets exist for scenes 1A (porch, Tsukimi night), 1B (snow) and 1C (the hearth). Any other scene renders on a plain
floor until its set is written.

## Characters made from code

No modelling is needed either. Without a model file, each character is built from code in every shot, so it is the same
character everywhere:

* **The body** (`figure.py`, pure Python) has anime proportions for the character's age. A child of 7 is about 4.7 heads
  tall, with her eyes low in a round face. An adult woman is about 6.6 heads. At 80 and over the body is shorter and
  rounder, with a bowed back. The joints are worked out for the posture: standing, or kneeling in seiza on the heels
  with the hands on the lap. For a shot of the hands, the hands are held in front. The shot plan reads the eyes and hands
  from the same numbers, so the camera frames the face the builder makes.
* **The clothes** are smooth tubes along those joints (Blender's Skin modifier): a work jacket with sleeves, loose
  trousers, a sash and a white collar. The grandmother's clothes are rounder, with an apron.
* **The head** is a sphere with a narrowing jaw and a small chin. Its lighting normals are those of a ball eased
  towards the front, so light falls on the face in one clean shape, as in anime CG.
* **The face is drawn on the head:** the parts are flat and never outlined, and they are lit with the skin.
  * One shader draws both eyes: the white with the lid's shadow on it, an iris dark under the lid and warm below, a
    pupil, and two catchlights. All of it is clipped to the eye's shape, so the gaze moves inside the eye.
  * The upper lid is a thick line with a flick at the outer corner. Eyes close over the lower lid, and joy turns
    them into arches.
  * Brows, a mouth with shapes for A, I, U, E, O, smile and frown, blush and tears complete the face.
  * Old faces get lines at the eyes and beside the mouth.
* **Hair** is a cap whose edge follows the hairline, plus clumps: bangs, side locks, and then a bob, long hair, or
  combed-back strands into a bun with a hairpin.
* `cgi.characters.<name>.look` in config.json sets each look by chapter (age, hair style, colours). A VRM file in
  `assets/characters/` still replaces the character, and `"stand_in": "maquette"` brings back the simple stand-in.

Characters take light and shadow from the set but cast no shadow on themselves: a toon face shadowing itself turns to
grain. Rendering is also cleaner and faster now. Keys hold until the next one, as the plan intends (twos, hit stops,
blinks); before this fix, Blender eased between them when run as a module. Screen-space GI is off, and light shadows
are hard.

## Dialogue, automated

No actors are needed. `style_bible/dialogue.json` holds every line of the film, in English and Japanese, with the
Japanese reading in kana. Each line also says who speaks, when (its earliest start in the shot), the feeling, how it
is said, and whether the speaker is on screen. The lines the shot sheet only describes (the meal Sayo lays out in
1-22, Tsumugi teaching the child in 5-04) and the laughs are written out too.

1. **Voices.** Each character has a voice per language and per age: Tsumugi at 7, 27 and 87 sound different.
   * The default engine is **Kokoro-82M** (Apache 2.0). It has English and Japanese voices and is fast on a CPU, so
     it runs on GitHub's free runners as well as the Mac.
   * Any other engine plugs in as a command (`"engine": "command"`), for example Qwen3-TTS through mlx-audio on the
     Mac, which takes a direction such as "restrained, holding back tears". The acting sheet's emotion, intensity and
     restraint are passed to it.
   * Each voice then gets its own treatment: pitch, and for Akari a hollow, shimmering timbre, like a voice from
     inside paper. Every line is set to the same loudness (-18 LUFS).
   * Lines are cached, so a change re-voices only the lines that changed.
2. **Placing lines.** Lines never overlap, and each starts at its cue or after the line before it, plus a breath.
   If the lines would run past the cut, they first start earlier, then the breaths shorten, then the speech is sped
   up by at most 15%. Anything still too long is listed in the review sheet, so the line can be rewritten.
3. **Lip sync** (kuchipaku). The loudness of the voice opens and closes the mouth, and the vowels give its shape
   (A, I, U, E, O). For Japanese the vowels come from the kana; for English, from the spelling. The mouth moves in
   anime's few steps and is held on twos, so it reads as drawn mouth flaps. While a character speaks, their smile
   or grief softens around the words. Akari has no mouth, so its light pulses as it speaks.
4. **In the cut.** English is the main audio track and Japanese is a second track; YouTube can carry both. Each
   language gets an `.srt` file next to the video. Laughs are not subtitled.

```bash
python -m engine cgi voices --chapter 1           # voice, place and lip-sync every line of chapter 1
python -m engine cgi voices 1-11 --engine stub    # a dry run with a test tone instead of a model
python -m engine cgi subtitles --chapter 1        # media/cgi/subs/chapter_1.en.srt and .ja.srt
```

Kokoro needs `pip install "kokoro>=0.9.4" "misaki[ja]" soundfile` and espeak-ng, on Python 3.10 to 3.12. The model
downloads once from Hugging Face.

## Automated end to end

`python -m engine cgi auto --chapter N` makes a chapter with no one at the keyboard:

1. **Voices:** every line voiced, placed and lip-synced.
2. **Render:** every shot built and rendered (previews, or `--final` for 4K). A shot whose plan has not changed since
   its last render is skipped. `--shard 2/8` takes one share of the shots, so several machines can work on one chapter.
3. **Finish:** the storyboard, the chapter cut with its dialogue tracks and subtitles, and `media/cgi/review-chN.json`.
   For each shot, the review sheet holds the intent from the acting sheet, three frames to judge, the lines placed in
   it, and any line that runs over.

`.github/workflows/cgi.yml` runs the three stages on GitHub's free runners, splitting the renders across a matrix of
machines. It runs when acting sheets, sets, lights, dialogue or the CG code change on `main`, or when started by hand.
The board, the cut, the subtitles and the review sheet are uploaded to the release `cgi-chN`. Claude reviews them,
edits the sheets, sets and lines, and pushes, which starts the next round.

Nothing waits for a person. A sheet set to `"status": "hold"` is skipped: that is the owner's veto. Anything that
costs money (Colab units, a paid model, a cloud GPU) still needs the owner.

## On the Mac: first run

1. Install **Blender 5.3** from blender.org (free). In Blender, go to Edit → Preferences → Get Extensions, search for **VRM**,
   and install it.
2. Pull the repository, then check:
   ```bash
   cd ~/Documents/Projects/mochizuki && git pull && source .venv/bin/activate
   python -m engine cgi check
   ```
3. Render the opening shot, the mallet falling into the mortar, as a preview, then the whole of chapter 1 as a storyboard:
   ```bash
   python -m engine cgi render 1-02 --preview
   python -m engine cgi render --chapter 1 --board
   ```
   Clips, frames, plans and `.blend` files go in `media/cgi/<shot>/`, and the board in `media/cgi/board-ch1.png`.
   Logs go in `media/cgi/logs/cgi.jsonl`.
4. Make a character in **VRoid Studio** (free, Mac) and export it as VRM to `assets/characters/tsumugi-07.vrm`
   (see `assets/characters/README.md`). Shots with Tsumugi now use her instead of the stand-in.
5. Act a hero shot. Install **Live Link Face** on the iPhone (free), choose the ARKit mode, record yourself playing 1-05
   (the laugh), and AirDrop the CSV:
   ```bash
   python -m engine cgi capture 1-05 --who tsumugi ~/Downloads/take.csv --start 0.4
   python -m engine cgi render 1-05 --preview
   ```
   A plain video works too (MediaPipe; `pip install mediapipe` first): `cgi capture 1-05 --who tsumugi take.mov`.
6. Final 4K: drop `--preview`. Add `--prores` for the edit in Resolve.

Useful: `cgi sheets --chapter 1` lists every shot's sheet in one line each. `cgi sheet 1-11` prints one in full, so you
can copy it into `style_bible/acting.json` and edit it.

## What to train (and what not)

* **A character LoRA per character, for the stills** (already wired: `mac-worker --train-lora`). Render the VRoid model
  on a turntable in Blender: 30 views, a few expressions, a few lights. Train Z-Image on those renders. Every still,
  poster and thumbnail is then on model, because the training images are the model.
* **A face mapping per character** (`retarget.fit_mapping`). You hold 6–10 poses (neutral, smile, laugh, sad, surprised,
  angry, and each mouth vowel) in front of the iPhone. It fits a ridge regression from ARKit's shapes to the character's
  shapes. Training takes milliseconds. This only matters for models whose shapes are not VRoid's.
* **Later, a style LoRA for an AI finish** (phase 3, needs a GPU). A Wan 2.2 5B LoRA trained on your approved key art and
  toon renders, so a finish pass on hero shots stays in the film's look.
* **Not** training or fine-tuning a video model. There is no dataset, no compute, and it would not solve coherency.

## Research borrowed

| Work | What we take from it |
| --- | --- |
| AniMatrix (Tencent HY, arXiv 2605.03652, 2026) | Anime is conventions, not physics, so make them explicit categorical directives. That is the acting sheet's fixed vocabulary of framing, timing, emotion and restraint. |
| One Euro filter (Casiez, Roussel, Vogel, CHI 2012) | Smoothing a captured face without lag on fast moves. |
| Ramer–Douglas–Peucker | Turning continuous capture into key poses, so motion reads pose to pose. |
| Wan-Animate (arXiv 2509.14055) and Wan-Animate-2 (Apache 2.0, Aug 2026) | Transfers a performer's body and face onto a character image. This is the generative alternative for a hero shot, and a candidate finish pass. It needs a CUDA GPU. |
| Diffutoon-2 (arXiv 2611.30945, Nov 2026) | Toon shading with a video diffusion model, guided by the outlines and blurred colours of the input. This is the shape of the phase-3 finish on our own renders, which already have clean outlines and colour. |
| TurboDiffusion (arXiv 2512.16093), Wan2.2-Lightning, FastWan 5B MLX (3 steps) | Any video model we still run uses a 3–4 step distilled version, not 40–50 steps. |
| SAM 3D Body (Meta; SAM License, commercial use allowed) | Body capture from an ordinary video, on the MHR rig rather than SMPL, whose licence forbids commercial use. Phase 2, on a Kaggle or Colab GPU. |
| Audio2Face-3D (NVIDIA Open Model License) | Faces from the recorded dialogue, with an emotion input. A phase-2 option for lines that are not captured. |
| AnimaMimic (CVPR 2026) | Animating a rigged mesh from a video model's motion. This is the "AI reference actor" route if you do not want to act a shot. |
| Blender 5.3 (Nov 2026) EEVEE Material Lighting nodes | Per-light control in shaders. This is the upgrade path for the toon shader: different shadow shapes per light, and normals bent on the face. |

## Next phases

**Phase 2: bodies and words.**
* Body capture from video: SAM 3D Body on a GPU writes BVH files, which are retargeted to the VRM humanoid. MediaPipe pose
  on the Mac is the fallback.
* Lip sync is done (vowels and loudness). Audio2Face-3D is the upgrade for close-ups that need the full face to speak.
* Sets for chapters 2–5, the lunar colony and Akari's swarm.
* Spring-bone hair and sleeves.
* Physics for the chapter 4 action, where Blender's rigid bodies and low gravity do what video models cannot keep consistent.

**Phase 3: finish.**
* Compositing: bloom on the lanterns, film grain, glows.
* An optional AI pass on hero shots, conditioned on our own outlines and colours.
* ProRes into Resolve.

## Risks, plainly

* **The look depends on craft.** A VRoid model with default shading looks like a game avatar, not a feature film. What
  lifts it is the shading (two shadow tones, face normals bent towards a clean shadow shape), the light rigs, the camera,
  and later the finish pass. Expect several iterations on the shader and lights for the first scene.
* **The acting is only as good as the actor.** Capture gives exactly the performance it is given. For the hero shots, you
  or someone you cast acts them. The procedural beats are for the rest.
* **Kneeling is approximate** until body capture: arms down, thighs forward, shins folded.
* **Synthetic voices are not actors.** Kokoro is clear and warm but has a narrow emotional range; it will not cry or
  break on a word. The quiet lines suit it. For the lines that carry the film (Sayo's "So we made it better", Tsumugi's
  "No. It's finished."), an engine that takes directions (Qwen3-TTS) or a cast actor will be better, and both plug in
  without changing anything else.
* **Characters made from code are simple.** They have mitten hands without fingers, clothes without folds, and hair
  in clumps. They act, and read as anime characters, but a modelled character (VRoid) or the phase-3 finish pass will
  look richer. They do not yet hold props: in 1-04 the mallet swings in front of Sayo while her arms hang.
* **Licences checked:**
  * Blender's output is yours.
  * The VRM add-on is MIT/GPL.
  * Models you make in VRoid Studio may be used commercially (pixiv's VRoid Studio guidelines; check the licence of any
    preset or third-party item).
  * MediaPipe is Apache 2.0. Live Link Face is a free app; the CSV it records stays on your phone until you share it.
  * SAM 3D Body's licence allows commercial use.
  * Audio2Face-3D is under the NVIDIA Open Model License.
  * Wan is Apache 2.0.
  * Kokoro-82M and its voices are Apache 2.0; its Japanese reader (misaki, pyopenjtalk, UniDic) is BSD-licensed.
  * The sample VRM used to test the importer (Seed-san, VRM Public License) is not in the repository.

## How it fits the rest

* Reference stills and keyframes, from Colab, Kaggle or the Mac, become design work: character sheets for building the
  VRoid models, colour keys for the light rigs, and storyboards.
* GPU backends remain for the GPU-only extras of phases 2 and 3.
* `assemble` cuts the chapter from CG renders when `cgi.use_in_cut` is on, falling back shot by shot to takes, keyframes
  and cards.
* Gates are unchanged. You approve acting sheets (`"status": "approved"`), the board and the animatic.

## Sources

* AniMatrix: https://arxiv.org/abs/2605.03652
* Wan-Animate: https://arxiv.org/abs/2509.14055
* Wan-Animate-2: https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B
* Diffutoon-2: https://arxiv.org/abs/2611.30945
* TurboDiffusion: https://arxiv.org/abs/2512.16093
* HoloCine: https://arxiv.org/abs/2510.20822
* AnimaMimic: https://arxiv.org/abs/2512.14133
* SAM 3D Body: https://github.com/facebookresearch/sam-3d
* Audio2Face-3D: https://arxiv.org/abs/2508.16401
* VRoid Studio guidelines: https://vroid.com/en/studio/guidelines
* VRM Add-on for Blender: https://github.com/saturday06/VRM-Addon-for-Blender
* MediaPipe Face Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
* Live Link Face: https://apps.apple.com/us/app/-/id1495370836
* Blender 5.3: https://digitalproduction.com/2026/11/11/blender-5-3-splat-happens-and-everything-else-gets-better/
* Local video models and licences, Nov 2026: https://vdf.ai/blog/best-local-video-generation-models/
