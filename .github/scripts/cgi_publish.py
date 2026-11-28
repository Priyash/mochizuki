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

from engine.project import Project, find_film  # noqa: E402

FILM = find_film()                 # $FILM, else the only film in films/
MEDIA = FILM / "media"


def review(chapter: int) -> dict:
    f = MEDIA / "cgi" / f"review-ch{chapter}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"shots": []}


def files(chapter: int) -> list[Path]:
    cgi, cut = MEDIA / "cgi", MEDIA / "animatic"
    out = [cgi / f"board-ch{chapter}.png", cut / f"chapter_{chapter}.mp4", cgi / f"review-ch{chapter}.json"]
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
        stamp = next(iter(sorted(folder.glob("render-*.json"))), None)
        info = json.loads(stamp.read_text()) if stamp else {}
        man = folder / "dialogue.json"
        lines = len(json.loads(man.read_text(encoding="utf-8"))["lines"]) if man.exists() else 0
        clip = info.get("clip") and Path(info["clip"]).name
        if not clip:
            missing.append(s.id)
        rows.append(f"| {s.id} | {s.seconds} s | {clip or '**not rendered**'} | {info.get('seconds_per_frame') or ''} | {lines or ''} |")
    over = [o for s in review(chapter)["shots"] for o in s.get("overruns", [])]
    text = [f"## CG chapter {chapter}", "",
            f"Release: `cgi-ch{chapter}` (board, cut with dialogue, subtitles, review sheet and frames).", ""]
    if missing:
        text += [f"**{len(missing)} shots have no render:** {', '.join(missing)}. See the render jobs' logs.", ""]
    if over:
        text += ["**Lines that run past their shot:**", ""] + [f"- {o['who']} ({o['language']}): {o['text']}… by {o['by_seconds']} s"
                                                             for o in over] + [""]
    text += ["| Shot | Length | Clip | s/frame | Lines |", "| --- | --- | --- | --- | --- |", *rows]
    return "\n".join(text) + "\n"


if __name__ == "__main__":
    n = int(sys.argv[1])
    if "--summary" in sys.argv:
        print(summary(n))
    else:
        print("\n".join(str(f) for f in files(n)))
