"""After the cgi workflow's finish job: list the files to upload to the release cgi-chN (and zip the
review frames), or with --summary write the run's summary in Markdown.

    python .github/scripts/cgi_publish.py 1            # file paths, one per line
    python .github/scripts/cgi_publish.py 1 --summary  # Markdown for $GITHUB_STEP_SUMMARY
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.assemble import CUTS_DIR, cut_path  # noqa: E402
from engine.audio import dialogue  # noqa: E402
from engine.project import Project, find_film  # noqa: E402

FILM = find_film()                 # $FILM, else the only film in films/
MEDIA = FILM / "media"


def review(chapter: int) -> dict:
    f = MEDIA / "cgi" / f"review-ch{chapter}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"shots": []}


def dub_languages(p: Project) -> list[str]:
    """The film's languages other than its primary one: each may have a dubbed cut."""
    script = dialogue.load(p)
    langs = script.get("languages", [])
    return [x for x in langs if x != script.get("primary", langs[0] if langs else None)]


def files(chapter: int) -> list[Path]:
    p = Project(FILM)
    cgi, cut = MEDIA / "cgi", MEDIA / CUTS_DIR
    out = [cgi / f"board-ch{chapter}.png", cut_path(p, chapter), cgi / f"review-ch{chapter}.json"]
    out += [cut_path(p, chapter, language=x) for x in dub_languages(p)]       # the dubbed cuts
    out += sorted(cut.glob(f"chapter_{chapter}.*.srt"))
    frames = [FILM / f for s in review(chapter)["shots"] for f in s.get("frames", [])]
    if frames:
        z = ROOT / f"cgi-review-frames-ch{chapter}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in frames:
                if f.exists():
                    zf.write(f, f.relative_to(FILM))
        out.append(z)
    logs = sorted((cgi / "logs").glob("shard-*.jsonl"))
    if logs:
        merged = ROOT / f"cgi-log-ch{chapter}.jsonl"
        merged.write_text("".join(x.read_text(encoding="utf-8") for x in logs), encoding="utf-8")
        out.append(merged)
    return [f for f in out if f.exists()]


def summary(chapter: int) -> str:
    p = Project(FILM)
    rows, missing = [], []
    for s in p.chapter(chapter):
        folder = MEDIA / "cgi" / s.id
        stamp = next((f for f in sorted(folder.glob("render-*.json")) if f.name.count(".") == 1), None)  # not a dub's
        info = json.loads(stamp.read_text()) if stamp else {}
        man = folder / "dialogue.json"
        lines = len(json.loads(man.read_text(encoding="utf-8"))["lines"]) if man.exists() else 0
        clip = info.get("clip") and Path(info["clip"]).name
        if not clip:
            missing.append(s.id)
        rows.append(
            f"| {s.id} | {s.seconds} s | {clip or '**not rendered**'} "
            f"| {info.get('seconds_per_frame') or ''} | {lines or ''} |"
        )
    over = [o for s in review(chapter)["shots"] for o in s.get("overruns", [])]
    off_sync = [(s["shot"], x) for s in review(chapter)["shots"] for x in s.get("lipsync", []) if not x.get("ok")]
    cuts = [(s["shot"], x) for s in review(chapter)["shots"] for x in s.get("continuity", []) if not x.get("accepted")]
    off_arc = [(s["shot"], s["arc"]["off"]) for s in review(chapter)["shots"] if s.get("arc", {}).get("off")]
    text = [f"## CG chapter {chapter}", "",
            f"Release: `cgi-ch{chapter}` (board, cut with dialogue, dubbed cuts, subtitles, review sheet and frames).",
            ""]
    if missing:
        text += [f"**{len(missing)} shots have no render:** {', '.join(missing)}. See the render jobs' logs.", ""]
    if over:
        text += (
            ["**Lines that run past their shot:**", ""]
            + [f"- {o['who']} ({o['language']}): {o['text']}… by {o['by_seconds']} s" for o in over]
            + [""]
        )
    if cuts:
        text += (
            ["**Continuity at the cuts** (fix the sheet, or accept with a reason in its `continuity`):", ""]
            + [f"- {sid} after {x['after'] or '-'}, {x['rule']}: {x['message']}" for sid, x in cuts]
            + [""]
        )
    if off_arc:
        text += (
            ["**Shots off the emotional arc** (style_bible/arc.json):", ""]
            + [f"- {sid}: {why}" for sid, why in off_arc]
            + [""]
        )
    if off_sync:
        text += (
            ["**Lines whose lips are out of sync:**", ""]
            + [f"- {sid} {x['who']} ({x['language']}): picture first {x['picture_first_ms']} ms, "
               f"closures {x['closures'][0]}/{x['closures'][1]}, open in silence {x['open_in_silence']}"
               for sid, x in off_sync]
            + [""]
        )
    text += ["| Shot | Length | Clip | s/frame | Lines |", "| --- | --- | --- | --- | --- |", *rows]
    return "\n".join(text) + "\n"


if __name__ == "__main__":
    n = int(sys.argv[1])
    if "--summary" in sys.argv:
        print(summary(n))
    else:
        print("\n".join(str(f) for f in files(n)))
