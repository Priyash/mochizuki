"""Command line: python -m pipeline <command>. Run from the repo root."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import assemble as asm
from . import kaggle as kg
from . import plan as planner
from . import providers, steps
from .ledger import BudgetExceeded
from .project import Project


def load_env(root: Path) -> None:
    env = root / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


def kaggle_backend(name: str, p: Project):
    return kg.LocalKaggle(p.media) if name == "local" else kg.KaggleCLI()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline")
    ap.add_argument("--root", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def chap(p, required=True):
        p.add_argument("--chapter", type=int, required=required)

    def sync(p):
        p.add_argument("--provider", default="mock", choices=["mock", "fal"])
        p.add_argument("--shot", help="only this shot id")

    def backend(p):
        p.add_argument("--kaggle", default="real", choices=["real", "local"],
                       help="local runs the worker here with stub engines, for testing")

    a = sub.add_parser("plan", help="dry run: GPU time and money, nothing is called"); chap(a, False)

    # run-now steps (mock for testing, fal for the paid fallback)
    for name in ("keyframes", "video", "upscale"):
        a = sub.add_parser(name, help=f"{name} right now with a sync provider"); chap(a); sync(a)

    # Kaggle
    a = sub.add_parser("batch", help="pack the next Kaggle batch for one stage"); chap(a)
    a.add_argument("--stage", required=True, choices=kg.STAGES); a.add_argument("--shot")
    a.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])
    a = sub.add_parser("submit", help="push the open batch to Kaggle"); backend(a)
    a = sub.add_parser("collect", help="download and record the open batch if it has finished"); backend(a)
    a = sub.add_parser("auto", help="one automation step: collect or start the next batch"); chap(a); backend(a)
    a.add_argument("--loop", action="store_true", help="keep going until a person is needed")
    a.add_argument("--every", type=int, default=0, help="minutes between steps (default: poll_seconds)")
    a.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])

    # review and gates
    a = sub.add_parser("queue", help="what is waiting for a verdict"); chap(a)
    a = sub.add_parser("verdict", help="record a critic's verdict")
    a.add_argument("key"); a.add_argument("verdict", choices=["pass", "fail"])
    a.add_argument("--by", default="agent", choices=["agent", "human"]); a.add_argument("--note", default="")
    a = sub.add_parser("select", help="choose the keyframe a shot will use"); a.add_argument("key")
    a = sub.add_parser("approve", help="a person approves a gate"); chap(a)
    a.add_argument("gate", choices=["board", "animatic"])
    a.add_argument("--by", required=True, choices=["human"])
    a = sub.add_parser("status", help="shot states, 4K progress, GPU time used"); chap(a, False)
    a = sub.add_parser("resolve", help="close a job whose call was cut off")
    a.add_argument("key"); a.add_argument("--charged", required=True, choices=["yes", "no"])
    a = sub.add_parser("assemble", help="cut the chapter: animatic, or --4k picture"); chap(a)
    a.add_argument("--4k", dest="four_k", action="store_true")

    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    load_env(root)
    p = Project(root)

    if args.cmd == "plan":
        title = f"PLAN, chapter {args.chapter}" if args.chapter else "PLAN, whole film"
        print(planner.render(planner.plan(p, args.chapter), title))
        return 0

    led = steps.open_ledger(p)
    try:
        if args.cmd in ("keyframes", "video", "upscale"):
            ex = steps.SyncExecutor(providers.get(args.provider, p),
                                    only_provider="fal" if args.provider == "fal" else None)
            fn = {"keyframes": steps.keyframes, "video": steps.video, "upscale": steps.upscale}[args.cmd]
            out = fn(p, led, ex, args.chapter, args.shot)
        elif args.cmd == "batch":
            out = kg.build(p, led, args.stage, args.chapter, args.shot, engine_set=args.engine_set)
        elif args.cmd == "submit":
            b = led.active_batch()
            if not b:
                raise ValueError("no open batch; run `batch` first")
            out = kg.submit(p, led, kaggle_backend(args.kaggle, p), b["id"])
        elif args.cmd == "collect":
            b = led.active_batch()
            if not b:
                raise ValueError("no open batch")
            out = kg.collect(p, led, kaggle_backend(args.kaggle, p), b["id"])
        elif args.cmd == "auto":
            backend_ = kaggle_backend(args.kaggle, p)
            wait = (args.every * 60) or p.config.get("kaggle", {}).get("poll_seconds", 120)
            while True:
                out = kg.tick(p, led, backend_, args.chapter, engine_set=args.engine_set)
                print(json.dumps(out, ensure_ascii=False), flush=True)
                if not args.loop or out["action"] == "idle":
                    break
                if out["action"] == "waiting":
                    time.sleep(wait)
            return 0
        elif args.cmd == "queue":
            out = steps.review_queue(p, led, args.chapter)
        elif args.cmd == "verdict":
            led.verdict(args.key, args.verdict, args.by, args.note); out = {"ok": True}
        elif args.cmd == "select":
            led.select(args.key); out = {"ok": True}
        elif args.cmd == "approve":
            if args.gate == "board":
                missing = [s.id for s in p.chapter(args.chapter) if not led.selected(s.id)]
                if missing:
                    print(f"cannot approve: no keyframe selected for {missing}")
                    return 2
            led.approve(args.chapter, args.gate, args.by); out = {"ok": True}
        elif args.cmd == "status":
            out = steps.status(p, led, args.chapter)
            week = kg.week_start()
            out["kaggle_gpu_hours_this_week"] = round(led.gpu_seconds_since(week) / 3600, 2)
            out["kaggle_weekly_cap_hours"] = p.config.get("kaggle", {}).get("weekly_gpu_hours")
            b = led.active_batch()
            out["open_batch"] = dict(b) if b else None
        elif args.cmd == "resolve":
            led.resolve(args.key, args.charged == "yes"); out = {"ok": True}
        elif args.cmd == "assemble":
            out = asm.assemble(p, led, args.chapter, four_k=args.four_k)
        else:  # pragma: no cover
            return 2
    except BudgetExceeded as e:
        print(f"STOPPED, budget cap reached: {e}")
        return 3
    except (kg.KaggleError, PermissionError, ValueError, KeyError, RuntimeError) as e:
        print(f"refused: {e}")
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
