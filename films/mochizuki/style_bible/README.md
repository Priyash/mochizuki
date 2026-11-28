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
