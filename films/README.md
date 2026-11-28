# Films

The engine (`engine/`) holds no film. Each film is one folder here, and the same engine makes any of them,
chapter by chapter.

```
films/<name>/
  config.json          models, engines, GPU backends and caps, and the cast for the 3D pipeline (cgi.characters,
                       cgi.props_state)
  shots.csv            the shot sheet: id, chapter, shot, seconds, type, model, takes, sound
  style_bible/         style.txt, ref_prompts.json, scenes.json, chapter_styles.json, acting.json, sets.json,
                       lights.json, dialogue.json, and refs/ (the approved reference stills)
  prompts/             optional: <shot id>.txt replaces a shot's prompt
  assets/characters/   optional: your character models (.vrm); never committed
  media/               made by the engine: stills, keyframes, takes, renders, cuts, and ledger.sqlite
```

## A new film

1. Copy `films/mochizuki/` to `films/<name>/` and empty `media/` (keep `media/.gitkeep`).
2. Write the new `shots.csv`, the style bible and the cast (`config.json` → `cgi.characters`).
3. Run it with `python -m engine --film films/<name> plan`. With more than one film here, every command needs
   `--film` or the `FILM` environment variable. On GitHub, set the repository variable `FILM`.

`python -m engine where [film|ledger|refs|media|cgi]` prints the chosen film's paths, so scripts never spell them.
