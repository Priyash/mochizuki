# Style bible (gate 1)

Everything the keyframe engine knows about the look comes from this folder. You approve it before any GPU time is spent.

- `style.txt` is put at the start of every prompt: the full 3D CG anime look, the same for the whole film. Rewrite it in your own words. Never name a studio, a director or an existing film.
- `chapter_styles.json` adds one line per chapter after it: where the chapter takes place and its palette (amber and indigo; green to ash grey; rust dusk; lunar grey with green terraces; black and gold).
- `ref_prompts.json` lists the reference stills Kaggle paints when `refs/` lacks them. Claude picks the best candidate of each and you approve them with `/approve refs` on the production issue; they are then committed here.
- `refs/` holds reference stills. Up to three go with each keyframe request:
  - `style.png` (or `style_*.png`) is always sent.
  - Character and object refs are picked when their name appears in the shot description: `tsumugi_ch1.png`, `tsumugi_ch3.png`, `tsumugi_ch4.png`, `sayo.png`, `akari.png`, `bowl_ch1.png`, `bowl_ch3.png`, `bowl_ch4.png`.
  - A chapter-specific file (`_ch3`) beats a general one for that chapter.

Suggested set before chapter 1: style, Tsumugi at 7, Sayo, the bowl with one gold seam, and the farmhouse porch at dusk.

Where the refs come from is up to you: your own drawings, a commissioned artist, or stills you generate and approve. They must be yours to use.

## For the 3D pipeline

These files direct the 3D CG pipeline ([docs/cgi-pipeline.md](../../../docs/cgi-pipeline.md)). None of them needs a GPU.

- `scenes.json`: which shots make each scene. A scene shares one set, light and line of action, and the continuity check runs within it.
- `acting.json`: the hand-written acting sheets (camera, beats, gaze, restraint, timing). A shot without one renders from a draft of its description. `"continuity": {"<rule>": "why"}` accepts a deliberate break at the cut into the shot.
- `sets.json` and `lights.json`: each scene's set, marks and props, and its light rig.
- `dialogue.json`: every line in English and Japanese, with the kana reading that Japanese lip sync needs.
- `arc.json`: the intended emotional arc of each chapter, as points in the chapter's time or one of six story shapes. With `"fill_drafts": true`, drafted shots take their feeling from it.
- `culture.json`: the culture pack, display rules for how feeling is shown, by whom and before whom (here Japanese: elders hold grief in, a child's joy shows). A rule never changes a field a hand-written sheet gives.

`acting.json`, `arc.json` and `culture.json` are drafts until you set `"status": "approved"`.
