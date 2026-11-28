"""Command line: python -m engine [--film films/<name>] <command>."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import assemble as asm
from . import ci, providers, scheduler, stages, steps
from . import plan as planner
from .cgi import config as cgi_config
from .errors import EngineError
from .film_check import FilmError
from .ledger import BudgetExceeded
from .project import CONFIG_FILE, REPO, Project, find_film
from .providers.base import WORKER_PACKAGES
from .providers.kaggle import settings as kaggle_settings
from .providers.kaggle import week_start
from .providers.kaggle.probe import notebook_ref as kaggle_probe_ref
from .providers.release import Release
from .providers.release import names as release_names
from .scheduler import Action

# `python -m engine where <what>`: the film's paths, so workflows and scripts never spell them
WHERE = {
    "film": lambda p: p.root,
    "ledger": lambda p: p.ledger_file,
    "refs": lambda p: p.refs_dir,
    "media": lambda p: p.media,
    "cgi": lambda p: cgi_config.work_dir(p, cgi_config.settings(p), ""),
    "kaggle-worker": lambda p: providers.KaggleProvider().refs(p)[1],     # <user>/<the film's worker notebook>
    "kaggle-probe": lambda p: kaggle_probe_ref(p),                         # <user>/<film>-probe
}


def load_env(root: Path) -> None:
    env = root / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


def finish_test(p: Project, args) -> int:
    """`finish-test`: the shot's finish on Kaggle or Colab, the results on the release finish-tests."""
    from .providers.kaggle import finish_test as ft  # noqa: PLC0415
    from .store import ReleaseStore  # noqa: PLC0415
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("error: set GITHUB_REPOSITORY (owner/name): the test's files go to its release", file=sys.stderr)
        return 2
    if args.chosen_refs and p.refs_missing():
        p.store = ReleaseStore(repo)
        done, _, lost = ci.copy_chosen_refs(p, steps.open_ledger(p))
        print(f"stills chosen in the ledger, for this test only: {', '.join(done) or 'none'}"
              + (f"; could not fetch {', '.join(lost)}" if lost else ""), flush=True)
    kgc = providers.KaggleProvider()
    r = ft.run(p, kgc, Release(repo, tag=ft.RELEASE), args.shot,
               [e.strip() for e in args.engine.split(",") if e.strip()],
               [c.strip() for c in args.controls.split(",") if c.strip()], Path(args.work), args.minutes,
               backend=args.backend, batches=Release(repo))
    print("\n".join(kgc.log[-6:]))
    print(json.dumps(r, indent=1, default=str))
    return 0 if r["jobs"] and all(j["ok"] for j in r["jobs"].values()) else 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="engine")
    ap.add_argument("--film", help="the film's folder (default: $FILM, else the only film in films/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def chap(p, required=True):
        p.add_argument("--chapter", type=int, required=required)

    def sync(p):
        p.add_argument("--provider", default="mock", choices=["mock", "fal"])
        p.add_argument("--shot", help="only this shot id")

    def backend(p):
        p.add_argument("--kaggle", default="real", choices=["real", "local"],
                       help="local runs the worker here with stub engines, for testing")

    a = sub.add_parser("mac-worker", help="on your Apple Silicon Mac: render stills, keyframes and video, "
                                         "run self-tests, train LoRAs (docs/mac-worker.md)")
    from .mac import cli as mac_cli  # noqa: PLC0415
    mac_cli.add_arguments(a)
    a = sub.add_parser("cgi", help="the 3D CG pipeline on your Mac: acting sheets, captures, Blender renders "
                                  "(docs/cgi-pipeline.md)")
    from .cgi import cli as cgi_cli  # noqa: PLC0415
    cgi_cli.add_arguments(a)
    a = sub.add_parser("colab-runner", help="take the waiting Colab batch and run it on a Colab VM (GitHub Actions)")
    a.add_argument("--work", default="colab-work")
    a = sub.add_parser("colab-test", help="one reference still on a Colab VM, outside the ledger")
    a.add_argument("--out", default="colab-test")
    sub.add_parser("colab-gpus", help="which Colab GPUs this account gets now, and their units per hour")
    a = sub.add_parser("colab-smoke", help="the Colab round trip on a free CPU VM: no GPU, no units")
    a.add_argument("--work", default="colab-smoke")
    sub.add_parser("colab-sweep", help="stop any Colab GPU VM left running (the colab-run workflow's last step)")
    a = sub.add_parser("kaggle-probe", help="test the stills engine on a Kaggle T4 outside the ledger")
    a.add_argument("--variants", default="fp16,fp32", help="precisions to try, in order (fp16, fp32, bf16, config)")
    a.add_argument("--minutes", type=int, default=60)
    a.add_argument("--out", default="probe-out")
    a = sub.add_parser("finish-test", help="one shot's AI finish on a Kaggle T4 (or Colab), outside the ledger "
                                          "(render it with --guides first)")
    a.add_argument("shot")
    a.add_argument("--engine", default=steps.FINISH,
                   help="finish engines, comma-separated: ai_finish, a variant such as ai_finish_720, or both")
    a.add_argument("--controls", default="depth,lines", help="the guides to finish from, one job each")
    a.add_argument("--backend", default="kaggle", choices=["kaggle", "colab"],
                   help="colab spends the owner's units: only with the owner's yes")
    a.add_argument("--minutes", type=int, default=120, help="the notebook's time limit")
    a.add_argument("--work", default="finish-test")
    a.add_argument("--chosen-refs", action="store_true",
                   help="if style_bible/refs lacks the stills, take the candidates chosen in the ledger "
                        "(a test checkout only: nothing is committed or approved)")
    a = sub.add_parser("plan", help="dry run: GPU time and money, nothing is called")
    chap(a, False)

    # run-now steps (mock for testing, fal for the paid fallback)
    for name in ("keyframes", "video", "upscale"):
        a = sub.add_parser(name, help=f"{name} right now with a sync provider")
        chap(a)
        sync(a)

    # Kaggle
    a = sub.add_parser("batch", help="pack the next Kaggle batch for one stage")
    chap(a)
    backend(a)
    a.add_argument("--stage", required=True, choices=scheduler.STAGES)
    a.add_argument("--shot")
    a.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])
    a = sub.add_parser("submit", help="push the open batch to Kaggle")
    backend(a)
    a = sub.add_parser("collect", help="download and record the open batch if it has finished")
    backend(a)
    a = sub.add_parser("auto", help="one automation step: collect or start the next batch")
    chap(a)
    backend(a)
    a.add_argument("--loop", action="store_true", help="keep going until a person is needed")
    a.add_argument("--every", type=int, default=0, help="minutes between steps (default: poll_seconds)")
    a.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])

    # review and gates
    a = sub.add_parser("queue", help="what is waiting for a verdict")
    chap(a)
    a = sub.add_parser("verdict", help="record a critic's verdict")
    a.add_argument("key")
    a.add_argument("verdict", choices=["pass", "fail"])
    a.add_argument("--by", default="agent", choices=["agent", "human"])
    a.add_argument("--note", default="")
    a = sub.add_parser("select", help="choose the keyframe a shot will use")
    a.add_argument("key")
    a = sub.add_parser("approve", help="a person approves a gate")
    chap(a)
    a.add_argument("gate", choices=["board", "animatic"])
    a.add_argument("--by", required=True, choices=["human"])
    a = sub.add_parser("status", help="shot states, 4K progress, GPU time used")
    chap(a, False)
    a = sub.add_parser("resolve", help="close a job whose call was cut off")
    a.add_argument("key")
    a.add_argument("--charged", required=True, choices=["yes", "no"])
    a = sub.add_parser("assemble", help="cut the chapter: animatic, or --4k picture")
    chap(a)
    a.add_argument("--4k", dest="four_k", action="store_true")

    # GitHub Actions
    a = sub.add_parser("ci-step", help="one producer run on GitHub Actions")
    chap(a, False)
    a.add_argument("--state", default="state", help="checkout of the state branch")
    a.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])
    backend(a)
    a.add_argument("--no-store", action="store_true", help="keep media on this machine (testing)")
    a = sub.add_parser("gate", help="run the owner's command from an issue comment (COMMENT_BODY)")
    a.add_argument("--state", default="state")
    sub.add_parser("check", help="check the film's config.json and shots.csv; every problem at once")
    a = sub.add_parser("chain", help="where the film stands, stage by stage (one chapter, or every chapter)")
    chap(a, False)
    a = sub.add_parser("where", help="print one of the film's paths or Kaggle notebooks, for scripts and workflows")
    a.add_argument("what", nargs="?", default="film", choices=sorted(WHERE))

    args = ap.parse_args(argv)
    load_env(REPO)                       # the engine's .env: credentials, and FILM if it is set there
    try:
        root = find_film(args.film)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    load_env(root)                       # a film may add its own (its own Kaggle account, say)
    try:
        p = Project(root)
    except FilmError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.cmd == "check":
        chapters = sorted({s.chapter for s in p.shots})
        print(f"{p.name}: no problems. {len(p.shots)} shots in {len(chapters)} chapters, "
              f"{sum(s.seconds for s in p.shots)} s; {len(p.config['models'])} models, "
              f"{len(p.config['engines'])} engines")
        return 0

    if args.cmd == "where":
        found = WHERE[args.what](p)
        print(found.relative_to(REPO) if isinstance(found, Path) and found.is_relative_to(REPO) else found)
        return 0

    if args.cmd == "plan":
        title = f"PLAN, chapter {args.chapter}" if args.chapter else "PLAN, whole film"
        print(planner.render(planner.plan(p, args.chapter), title))
        return 0

    if args.cmd == "mac-worker":
        from .mac import cli as mac_cli  # noqa: PLC0415
        return mac_cli.run(p, args)
    if args.cmd == "cgi":
        from .cgi import cli as cgi_cli  # noqa: PLC0415
        return cgi_cli.run(p, args)
    if args.cmd in ("colab-runner", "colab-test", "colab-smoke", "colab-gpus", "colab-sweep"):
        from .log import set_file  # noqa: PLC0415
        from .providers import colab as cb  # noqa: PLC0415
        set_file(os.environ.get("MOCHI_LOG_FILE") or "colab-logs/colab.jsonl")
        cfg, cli = cb.settings(p), cb.ColabCLI()
        if args.cmd == "colab-sweep":
            n = cb.stop_gpu_vms(cli, why="found by the end-of-run sweep")
            print("could not list the account's Colab VMs: check the Colab console" if n is None
                  else f"stopped {n} GPU VM(s)")
            return 0 if n is not None else 3
        if args.cmd == "colab-gpus":
            r = cb.gpu_probe(cli)
            print(json.dumps(r, indent=1, default=str))
            return 0
        if args.cmd == "colab-smoke":
            r = cb.smoke(cli, Path(args.work))
            print(json.dumps(r, indent=1, default=str))
            return 0 if r.get("ok") else 3
        if args.cmd == "colab-test":
            r = cb.test_still(p, cli, cfg, list(WORKER_PACKAGES), Path(args.out))
            print(json.dumps(r, indent=1, default=str))
            return 0 if r.get("ok") and r.get("still") else 3
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        b = cb.take_one(Release(repo), Path(args.work), cli, cfg, list(WORKER_PACKAGES))
        print(f"ran {b}" if b else "nothing ran")
        return 0
    if args.cmd == "kaggle-probe":
        from .providers.kaggle import probe as kaggle_probe  # noqa: PLC0415
        kgc = providers.KaggleProvider()
        r = kaggle_probe.run(p, kgc, Path(args.out), args.variants.split(","), args.minutes)
        print("\n".join(kgc.log[-6:]))
        print(json.dumps(r, indent=1, default=str))
        return 0 if any(j.get("ok") for j in r["jobs"].values()) else 3
    if args.cmd == "finish-test":
        return finish_test(p, args)
    led = steps.open_ledger(p)
    if args.cmd == "chain":
        print(stages.report(p, led, [args.chapter] if args.chapter else sorted({s.chapter for s in p.shots})))
        return 0
    if args.cmd == "ci-step":
        from .store import ReleaseStore  # noqa: PLC0415
        state = Path(args.state)
        state.mkdir(parents=True, exist_ok=True)
        if not args.no_store:
            p.store = ReleaseStore()
        backends, off = providers.available(p, args.kaggle)     # Kaggle first, then Colab, then the Mac
        for name in off:                                 # switched off: its waiting batches go back to the pool
            for b in led.active_batches():
                if b["backend"] == name:
                    scheduler.discard(led, b["id"], f"the {name} backend was switched off in {CONFIG_FILE}")
                    if not args.no_store:
                        rel = Release(os.environ.get("GITHUB_REPOSITORY", ""))
                        for asset in release_names(b["id"], name).values():
                            rel.delete(asset)
                    print(f"{name} batch {b['id']} dropped; its jobs are free again", flush=True)
        out: Any = ci.step(p, led, args.chapter or ci.chapter_of(state), state, backends, engine_set=args.engine_set)
        print(json.dumps(out["notice"], ensure_ascii=False))
        return 0
    if args.cmd == "gate":
        state = Path(args.state)
        state.mkdir(parents=True, exist_ok=True)
        if os.environ.get("GITHUB_REPOSITORY"):          # on GitHub: approved stills come from the releases
            from .store import ReleaseStore  # noqa: PLC0415
            p.store = ReleaseStore()
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
            out = scheduler.build(p, led, args.stage, args.chapter, args.shot,
                                  provider=providers.kaggle_provider(args.kaggle, p), engine_set=args.engine_set)
        elif args.cmd == "submit":
            prov = providers.kaggle_provider(args.kaggle, p)
            b = led.active_batch(prov.name)
            if not b:
                raise ValueError("no open batch; run `batch` first")
            out = scheduler.submit(p, led, prov, b["id"])
        elif args.cmd == "collect":
            prov = providers.kaggle_provider(args.kaggle, p)
            b = led.active_batch(prov.name)
            if not b:
                raise ValueError("no open batch")
            out = scheduler.collect(p, led, prov, b["id"])
        elif args.cmd == "auto":
            prov = providers.kaggle_provider(args.kaggle, p)
            wait = (args.every * 60) or kaggle_settings(p).get("poll_seconds", 120)
            while True:
                out = scheduler.tick(p, led, prov, args.chapter, engine_set=args.engine_set)
                print(json.dumps(out, ensure_ascii=False), flush=True)
                if not args.loop or out["action"] == Action.IDLE:
                    break
                if out["action"] == Action.WAITING:
                    time.sleep(wait)
            return 0
        elif args.cmd == "queue":
            out = steps.review_queue(p, led, args.chapter)
        elif args.cmd == "verdict":
            led.verdict(args.key, args.verdict, args.by, args.note)
            out = {"ok": True}
        elif args.cmd == "select":
            led.select(args.key)
            out = {"ok": True}
        elif args.cmd == "approve":
            if args.gate == "board":
                missing = [s.id for s in p.chapter(args.chapter) if not led.selected(s.id)]
                if missing:
                    print(f"cannot approve: no keyframe selected for {missing}")
                    return 2
            led.approve(args.chapter, args.gate, args.by)
            out = {"ok": True}
        elif args.cmd == "status":
            out = steps.status(p, led, args.chapter)
            out["kaggle_gpu_hours_this_week"] = round(led.gpu_seconds_since(week_start()) / 3600, 2)
            out["kaggle_weekly_cap_hours"] = kaggle_settings(p).get("weekly_gpu_hours")
            out["open_batches"] = [dict(b) for b in led.active_batches()]
        elif args.cmd == "resolve":
            led.resolve(args.key, args.charged == "yes")
            out = {"ok": True}
        elif args.cmd == "assemble":
            out = asm.assemble(p, led, args.chapter, four_k=args.four_k)
        else:  # pragma: no cover
            return 2
    except BudgetExceeded as e:
        print(f"STOPPED, budget cap reached: {e}")
        return 3
    except (EngineError, PermissionError, ValueError, KeyError, RuntimeError) as e:
        print(f"refused: {e}")
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
