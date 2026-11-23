"""The dry run: what each shot would cost in money and in Kaggle GPU time. Calls nothing."""
from __future__ import annotations

from .project import Project, Shot


def shot_plan(p: Project, s: Shot) -> dict:
    k = p.config["keyframes"]
    n, cap = k["candidates"], k.get("max_candidates", k["candidates"])
    variants = len(p.variants(s))
    take = p.take_cost(s)
    kf_gpu = p.gpu_estimate("keyframe", s)
    take_gpu = p.gpu_estimate("take", s) if p.model(s.model).get("generates_video") else 0.0
    up_gpu = p.gpu_estimate("upscale", s) * (1 + variants)
    return {
        "id": s.id, "type": s.type, "model": s.model, "seconds": s.seconds,
        "billed": p.billed_seconds(s), "cap": s.takes,
        "kf_best": round((n + variants) * p.keyframe_cost(), 2),
        "kf_worst": round((cap + variants) * p.keyframe_cost(), 2),
        "video_best": round(take if s.takes else 0, 2),      # first take passes
        "video_worst": round(take * s.takes, 2),            # every take up to the cap
        "gpu_best": (n + variants) * kf_gpu + (take_gpu if s.takes else 0) + up_gpu,
        "gpu_worst": (cap + variants) * kf_gpu + take_gpu * s.takes + up_gpu,
    }


def plan(p: Project, chapter: int | None = None) -> dict:
    rows = [shot_plan(p, s) for s in p.chapter(chapter)]
    tot = lambda k: round(sum(r[k] for r in rows), 2)  # noqa: E731
    weekly = p.config.get("kaggle", {}).get("weekly_gpu_hours", 28)
    return {
        "rows": rows,
        "shots": len(rows),
        "seconds": sum(r["seconds"] for r in rows),
        "kf_best": tot("kf_best"), "kf_worst": tot("kf_worst"),
        "video_best": tot("video_best"), "video_worst": tot("video_worst"),
        "best": round(tot("kf_best") + tot("video_best"), 2),
        "worst": round(tot("kf_worst") + tot("video_worst"), 2),
        "gpu_hours_best": round(tot("gpu_best") / 3600, 1),
        "gpu_hours_worst": round(tot("gpu_worst") / 3600, 1),
        "weeks_best": round(tot("gpu_best") / 3600 / weekly, 1),
        "weeks_worst": round(tot("gpu_worst") / 3600 / weekly, 1),
    }


def render(pl: dict, title: str) -> str:
    out = [title, "", f"{'id':<6}{'type':<8}{'model':<17}{'sec':>4}{'cap':>4}{'GPU h, 1 take':>15}"
                      f"{'GPU h, cap':>12}{'$ cap':>8}"]
    for r in pl["rows"]:
        out.append(f"{r['id']:<6}{r['type']:<8}{r['model']:<17}{r['seconds']:>4}{r['cap']:>4}"
                   f"{r['gpu_best'] / 3600:>15.2f}{r['gpu_worst'] / 3600:>12.2f}"
                   f"{r['kf_worst'] + r['video_worst']:>8.2f}")
    out += ["",
            f"shots {pl['shots']}, running time {pl['seconds']} s",
            f"Kaggle GPU time  {pl['gpu_hours_best']} h to {pl['gpu_hours_worst']} h "
            f"(~{pl['weeks_best']} to {pl['weeks_worst']} weeks of free quota)",
            f"money            ${pl['best']:.2f} to ${pl['worst']:.2f}",
            "GPU times are estimates from config.json until real runs replace them (see `status`).",
            "Dry run: no model was called and nothing was spent."]
    return "\n".join(out)
