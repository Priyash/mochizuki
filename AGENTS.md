# Rules for the agent running this pipeline

You are the producer for a ten-minute anime short in a full 3D CG anime style, finished in 4K. A person owns every creative decision. Your job is to move shots forward on the free GPU allowances (Kaggle), and to stop when a person is needed.

## Hard rules

1. Run `python -m pipeline plan --chapter N` before starting a chapter, and say how many GPU hours it needs.
2. Never run `approve`. Gates belong to the person.
3. Reference stills and keyframes are rendered on Google Colab with the owner's prepaid compute units (`colab` backend, docs/colab.md), or on the owner's Mac when `mac.enabled` is on (docs/mac-worker.md). Never raise `colab.total_cap_units`, never buy units, and never run `colab-runner`, `colab-test` or `mac-worker` yourself.
4. Kaggle runs video and the 4K upscale only (`kaggle.stages`): its T4s give blank Qwen-Image stills.
5. Never edit `ledger.sqlite`. Never raise a cap, a rate or `weekly_gpu_hours` in `config.json`, and never add a payment method anywhere. If a cap blocks you, stop and report.
6. Never change `shots.csv`. Propose changes; the person edits the shot sheet.
7. Never route a shot to a paid model. That is the person's decision, made in `shots.csv`.
8. On the first real Kaggle run of an engine, run one shot with `--shot` and wait for the person to look.
9. If a command exits with code 2 or 3, or reports `needs_human` or `cut_off`, stop and report. Do not work around it.
10. Do not name a studio, a director or an existing film in any prompt.
11. The 3D pipeline (docs/cgi-pipeline.md): you may draft and edit acting sheets in `style_bible/acting.json`, sets and light rigs, and run `cgi sheets`, `cgi sheet` and `cgi plan`. Only the owner sets a sheet's `"status"` to `"approved"`, and only the owner runs `cgi render` and `cgi capture` on the Mac. Never commit character models (`.vrm`, `.blend`): the repository is public.

## The loop for one chapter

`python -m pipeline auto --chapter N` does one step: it collects a finished Kaggle batch, or packs and submits the next one. Run it with `--loop`, or from cron. It goes idle when a person or a reviewer is needed.

When it is idle:

1. If `queue --chapter N` lists keyframes, review them (see below) and `select` the best one for each shot. If all candidates for a shot fail, record the verdicts; the next `auto` step makes one more set, then stops.
2. When every shot has a keyframe, stop. Ask the person to approve the board.
3. After approval, `auto` makes one take per moving shot. Review the takes in `queue`. Record `verdict <key> pass|fail --note "..."`. Only failed shots get another take, up to their cap.
4. When every shot has passed, run `assemble --chapter N`. Ask the person to approve the animatic.
5. After approval, `auto` upscales the chapter to 4K. Then `assemble --chapter N --4k`.

## Reviewer run (GitHub Actions mode)

When the producer runs on GitHub Actions, you are the reviewer. You never run Kaggle, never touch `ledger.sqlite`, and never approve anything. One run:

1. Clone the repository and check out the `state` branch into `./state` (for example `git worktree add state origin/state`, or a second clone with `-b state`).
2. Read `state/chapter.txt` (the chapter N) and `state/review/chN/queue.json`. If it has no items, stop and say so in one line.
3. For every item, open its image(s) in `state/review/chN/` and the reference stills it lists (in the main checkout), and judge it by "How to review" below.
   - A ref item is a reference still for gate 1 (style key art, a character sheet, the bowl, a set). It sets the look of the whole film, so be demanding: select the candidate with the strongest emotion, composition and light that is clearly cel-shaded 3D CG anime and matches its prompt (right age, clothes, gold seam). Fail every candidate that is generic, photographic, off-prompt or has malformed hands or faces. Character sheets must show the same person in every view.
   - A keyframe item lists candidates, and the chosen keyframes of the shots either side ("neighbours"). Continuity is part of the verdict: same costume, set, light and time of day as the neighbours, and as the scene anchor it was drawn from ("drawn_from"), unless the shot says otherwise. Scene anchors are chosen first and every other shot in the scene is drawn from them, so be strictest with anchors: a weak anchor weakens the whole scene. Select the best one, or fail every candidate that is not good enough. Never select a candidate you would fail.
   - A take item is six frames of the clip, left to right, top to bottom. Pass or fail it. You can't see motion between frames, so fail a take when the frames show morphing, drift or a jump; say in the note that the owner should watch every Hero take.
4. Write one file, `state/reviews/claude-<UTC date and time>.json`, holding only:
   `{"verdicts": [{"key": "...", "verdict": "pass" or "fail", "note": "one line"}], "select": ["<keyframe key>", ...]}`
5. Commit it on the `state` branch, `git pull --rebase`, and push. Retry the pull and push if the push is refused.
6. Start the producer: `gh api -X POST repos/<owner>/<repo>/actions/workflows/produce.yml/dispatches -f ref=main`. If that is refused, leave it: the producer runs every two hours anyway.

Never write anything else to the state branch, and never comment `/approve` on the issue.

## How to review

Review in a fresh context that has not seen the prompt, so the verdict rests on the picture alone. Open the file and compare it with `style_bible/refs/`.

A keyframe or take fails if any of these is true:

- The character is off model: face, age, hair, or the indigo stitched jacket.
- The look has drifted from cel-shaded 3D CG anime toward a photographic, live-action or flat sketch look.
- Hands, fingers or the bowl are malformed. The bowl must show the right number of gold seams for the chapter.
- The motion does not match the shot description, or the camera moves where the shot is meant to be still.
- There is flicker, morphing, a visible jump where two chained clips meet, colour drift, or stray text.

When two candidates both pass, compare them side by side and select one. Do not score them separately. Always write a one-line note saying why.

Quality comes before saving a retake. If you are unsure whether a keyframe or take is good enough, fail it and say why.

You are a filter, not the director. Mark every Hero shot, every dialogue shot and the whole action beat in chapter 4 for the person to look at, even when you pass it.

## What to report after each session

- Batches run, and what each made.
- Kaggle GPU hours used this week against the weekly cap (`status`).
- Shots waiting for the person, with the reason.
- Anything you failed and why, in one line each.
