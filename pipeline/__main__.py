"""Command line: python -m pipeline <command>. Run from the repo root."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import assemble as asm
from . import ci
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
    k = kg.LocalKaggle(p.media) if name == "local" else kg.KaggleCLI()
    stages = p.config.get("kaggle", {}).get("stages")
    if stages and name != "local":
        k.stages = tuple(stages)                    # config kaggle.stages: only these run on Kaggle
    return k


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

    a = sub.add_parser("mac-worker", help="on your Apple Silicon Mac: render queued stills and keyframes with mflux")
    a.add_argument("--once", action="store_true", help="take what is waiting, then stop")
    a.add_argument("--self-test", action="store_true", help="make one still locally; nothing is sent to GitHub")
    a.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])
    a.add_argument("--work", default=str(Path.home() / "mochizuki-mac"))
    a.add_argument("--repo", default=os.environ.get("MOCHI_REPO", ""), help="owner/repo (default: this clone's)")
    a = sub.add_parser("colab-runner", help="take the waiting Colab batch and run it on a Colab VM (GitHub Actions)")
    a.add_argument("--work", default="colab-work")
    a = sub.add_parser("colab-test", help="one reference still on a Colab VM, outside the ledger")
    a.add_argument("--out", default="colab-test")
    a = sub.add_parser("kaggle-probe", help="test the stills engine on a Kaggle T4 outside the ledger")
    a.add_argument("--variants", default="fp16,fp32", help="precisions to try, in order (fp16, fp32, bf16, config)")
    a.add_argument("--minutes", type=int, default=60)
    a.add_argument("--out", default="probe-out")
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

    # GitHub Actions
    a = sub.add_parser("ci-step", help="one producer run on GitHub Actions"); chap(a, False)
    a.add_argument("--state", default="state", help="checkout of the state branch")
    a.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"]); backend(a)
    a.add_argument("--no-store", action="store_true", help="keep media on this machine (testing)")
    a = sub.add_parser("gate", help="run the owner's command from an issue comment (COMMENT_BODY)")
    a.add_argument("--state", default="state")

    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    load_env(root)
    p = Project(root)

    if args.cmd == "plan":
        title = f"PLAN, chapter {args.chapter}" if args.chapter else "PLAN, whole film"
        print(planner.render(planner.plan(p, args.chapter), title))
        return 0

    if args.cmd == "mac-worker":
        from . import mac_backend  # noqa: PLC0415
        cfg = mac_backend.settings(p)
        engine = mac_backend.StubEngine() if args.engine_set == "stub" else mac_backend.MfluxEngine(cfg)
        if args.self_test:
            r = mac_backend.self_test(p, engine, Path(args.work) / "selftest-style.png")
            print(json.dumps(r, indent=1))
            return 0
        repo = args.repo
        if not repo:
            r = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                               capture_output=True, text=True, cwd=p.root)
            repo = r.stdout.strip()
        if not repo:
            print("could not tell the repository: run inside the clone with gh logged in, or pass --repo owner/repo")
            return 2
        print(f"mac worker for {repo}: waiting for batches (Ctrl-C to stop)", flush=True)
        n = mac_backend.worker_loop(mac_backend.Release(repo), Path(args.work), engine, cfg, once=args.once)
        print(f"{n} batch(es) done")
        return 0
    if args.cmd in ("colab-runner", "colab-test"):
        from . import colab_backend as cb  # noqa: PLC0415
        cfg, cli = cb.settings(p), cb.ColabCLI()
        if args.cmd == "colab-test":
            r = cb.test_still(p, cli, cfg, list(kg.PACKAGES), Path(args.out))
            print(json.dumps(r, indent=1, default=str))
            return 0 if r.get("ok") and r.get("still") else 3
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        b = cb.take_one(cb.Release(repo), Path(args.work), cli, cfg, list(kg.PACKAGES))
        print(f"ran {b}" if b else "nothing ran")
        return 0
    if args.cmd == "kaggle-probe":
        from . import kaggle_probe  # noqa: PLC0415
        kgc = kg.KaggleCLI()
        r = kaggle_probe.run(p, kgc, Path(args.out), args.variants.split(","), args.minutes)
        print("\n".join(kgc.log[-6:]))
        print(json.dumps(r, indent=1, default=str))
        return 0 if any(j.get("ok") for j in r["jobs"].values()) else 3
    if args.cmd == "ci-step":
        from .store import ReleaseStore  # noqa: PLC0415
        state = Path(args.state)
        state.mkdir(parents=True, exist_ok=True)
        if not args.no_store:
            p.store = ReleaseStore()
        has_creds = os.environ.get("KAGGLE_USERNAME") and (
            os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY") or Path("~/.kaggle/kaggle.json").expanduser().exists())
        backends = []
        if has_creds or args.kaggle == "local":
            backends.append(kaggle_backend(args.kaggle, p))
        from . import colab_backend, mac_backend  # noqa: PLC0415
        for mod in (colab_backend, mac_backend):        # Kaggle first, then Colab, then the Mac
            b_ = mod.from_env(p)
            if b_ is not None:
                backends.append(b_)
                continue
            for b in led.active_batches():              # switched off: its waiting batches go back to the pool
                if b["backend"] == mod.NAME:
                    kg.discard(led, b["id"], f"the {mod.NAME} backend was switched off in config.json")
                    if not args.no_store:
                        rel = mac_backend.Release(os.environ.get("GITHUB_REPOSITORY", ""))
                        for name in mac_backend._names(b["id"], mod.NAME).values():
                            rel.delete(name)
                    print(f"{mod.NAME} batch {b['id']} dropped; its jobs are free again", flush=True)
        out = ci.step(p, led, args.chapter or ci.chapter_of(state), state, backends, engine_set=args.engine_set)
        print(json.dumps(out["notice"], ensure_ascii=False))
        return 0
    if args.cmd == "gate":
        state = Path(args.state)
        state.mkdir(parents=True, exist_ok=True)
        print(ci.command(p, led, state, os.environ.get("COMMENT_BODY", ""),
                         os.environ.get("COMMENT_AUTHOR", ""), os.environ.get("GITHUB_REPOSITORY_OWNER", "")))
        return 0
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
            out["open_batches"] = [dict(b) for b in led.active_batches()]
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
