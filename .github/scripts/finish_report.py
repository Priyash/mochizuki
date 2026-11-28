"""The AI finish test's report: how long one shot took on the GPU, and what that means for the film.

    python .github/scripts/finish_report.py 1-05        # Markdown, for $GITHUB_STEP_SUMMARY

It reads the newest results.json that came back with the shot's finish, and scales the measured time
to every shot the finish would take (engines.ai_finish.shots) across the film.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import steps  # noqa: E402
from engine.project import Project, find_film  # noqa: E402

KAGGLE_WEEK_HOURS = 30.0       # Kaggle's free GPU quota a week


def latest(p: Project, shot: str) -> tuple[dict, dict] | None:
    """The newest results.json holding this shot's finish, and its job's entry."""
    found = sorted(p.media.glob("batches/*/download/**/results.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in found:
        r = json.loads(f.read_text(encoding="utf-8"))
        for key, job in r.get("jobs", {}).items():
            if key.startswith(f"{shot}.finish"):
                return r, job
    return None


def report(p: Project, shot: str) -> str:
    head = f"## AI finish: {shot}\n\n"
    got = latest(p, shot)
    if got is None:
        return head + "No results came back from Kaggle: see this run's log.\n"
    results, job = got
    if not job.get("ok"):
        return head + f"**The finish failed** on the GPU:\n\n```\n{job.get('error', '?')[:1500]}\n```\n"
    s = p.shot(shot)
    gpu = float(job["gpu_seconds"])
    per = gpu / s.seconds
    load = (results.get("groups", {}).get(steps.FINISH) or {}).get("load_seconds")
    chosen = [x for c in sorted({x.chapter for x in p.shots}) for x in steps.finish_shots(p, c)]
    film_s = sum(x.seconds for x in chosen)
    film_h = per * film_s / 3600
    cfg = p.engine(steps.FINISH)
    lines = [
        "| | |\n| --- | --- |",
        f"| Shot | {shot}: {s.shot} ({s.seconds} s) |",
        f"| Model | {cfg.get('model')} at {job.get('width')}x{job.get('height')}, {job.get('fps')} fps, "
        f"{cfg.get('steps')} steps |",
        f"| Repainting | **{gpu:.0f} s of GPU**: {per:.0f} s for each second of film |",
        f"| Loading the model | {load:.0f} s, once per Kaggle session |" if load is not None else "",
        f"| The whole Kaggle session | {results.get('elapsed', 0) / 60:.0f} min |",
        f"| Every shot it would take | {len(chosen)} shots ({cfg.get('shots', 'Hero')}), {film_s} s of film |",
        f"| **The film, at this speed** | **about {film_h:.1f} GPU hours**: {film_h / KAGGLE_WEEK_HOURS:.1f} weeks of "
        f"Kaggle's free quota |",
    ]
    note = (f"\nThe config's guess was {cfg.get('gpu_seconds_per_output_second', '?')} s per second of film; the run "
            f"measured {per:.0f}. Set `engines.ai_finish.gpu_seconds_per_output_second` to it, so `plan` and the "
            "batch sizes are right.\n")
    return head + "\n".join(x for x in lines if x) + "\n" + note


if __name__ == "__main__":
    print(report(Project(find_film()), sys.argv[1]))
