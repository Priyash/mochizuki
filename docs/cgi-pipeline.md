# The 3D CG pipeline: build it once, perform it, render it

This is the new direction for making Mochizuki: a real 3D CG anime pipeline that runs on your Mac.
It replaces "generate every shot with a video model" as the way the film's moving picture is made.
The generative models stay, but in the jobs they are good at.

It is written in `pipeline/cgi/` (plain Python, tested) and `pipeline/cgi/blender/` (runs inside Blender).

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

Sets exist for scenes 1A (porch, Tsukimi night), 1B (snow) and 1C (the hearth). Any other scene renders on a plain
floor until its set is written.

## On the Mac: first run

1. Install **Blender 5.3** from blender.org (free). In Blender, go to Edit → Preferences → Get Extensions, search for **VRM**,
   and install it.
2. Pull the repository, then check:
   ```bash
   cd ~/Documents/Projects/mochizuki && git pull && source .venv/bin/activate
   python -m pipeline cgi check
   ```
3. Render the opening shot, the mallet falling into the mortar, as a preview, then the whole of chapter 1 as a storyboard:
   ```bash
   python -m pipeline cgi render 1-02 --preview
   python -m pipeline cgi render --chapter 1 --board
   ```
   Clips, frames, plans and `.blend` files go in `media/cgi/<shot>/`, and the board in `media/cgi/board-ch1.png`.
   Logs go in `media/cgi/logs/cgi.jsonl`.
4. Make a character in **VRoid Studio** (free, Mac) and export it as VRM to `assets/characters/tsumugi-07.vrm`
   (see `assets/characters/README.md`). Shots with Tsumugi now use her instead of the stand-in.
5. Act a hero shot. Install **Live Link Face** on the iPhone (free), choose the ARKit mode, record yourself playing 1-05
   (the laugh), and AirDrop the CSV:
   ```bash
   python -m pipeline cgi capture 1-05 --who tsumugi ~/Downloads/take.csv --start 0.4
   python -m pipeline cgi render 1-05 --preview
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
* Lip sync from the recorded dialogue: Audio2Face-3D, or visemes from phonemes.
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
* **Stand-ins are for timing.** Their faces show when a feeling lands, not what it looks like.
* **Licences checked:**
  * Blender's output is yours.
  * The VRM add-on is MIT/GPL.
  * Models you make in VRoid Studio may be used commercially (pixiv's VRoid Studio guidelines; check the licence of any
    preset or third-party item).
  * MediaPipe is Apache 2.0. Live Link Face is a free app; the CSV it records stays on your phone until you share it.
  * SAM 3D Body's licence allows commercial use.
  * Audio2Face-3D is under the NVIDIA Open Model License.
  * Wan is Apache 2.0.
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
