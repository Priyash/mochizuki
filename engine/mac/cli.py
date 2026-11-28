"""`python -m engine mac-worker ...`: everything you run on the Mac.

    mac-worker                       take batches meant for the Mac until Ctrl-C (--once: what waits, then stop)
    mac-worker --self-test           one still and one keyframe from the key-art prompt
    mac-worker --self-test --edit-only     only the keyframe, from the last self-test still
    mac-worker --video-test          a short take animated from the self-test still (Wan 2.2)
    mac-worker --train-lora NAME --images DIR    train a style or character LoRA
    --engines NAME                   use config mac.engines_<NAME> (e.g. zimage_base)
Every line printed is a structured log line; the same events go to ~/mochizuki-mac/logs/worker-<date>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from ..log import set_file
from ..providers.release import Release
from .common import WORK, log
from .config import settings, with_engine_set
from .engines import build
from .selftest import key_art_prompt, self_test, video_test
from .train import LOCAL_LORAS, train_lora
from .worker import worker_loop


def add_arguments(a: argparse.ArgumentParser) -> None:
    a.add_argument("--once", action="store_true", help="take what is waiting, then stop")
    a.add_argument("--self-test", action="store_true", help="one still and one keyframe here; nothing is sent to GitHub")
    a.add_argument("--edit-only", action="store_true", help="self-test: reuse the last self-test still, test only the keyframe engine")
    a.add_argument("--look", default="3d", choices=["3d", "2d"],
                   help="self-test: the 3D (default) or 2D wording of the key art; files get -2d in their name")
    a.add_argument("--variants", type=int, default=1, help="self-test: this many stills, each with its own seed")
    a.add_argument("--video-test", action="store_true", help="animate the self-test still (or --image) into a short take")
    a.add_argument("--image", default="", help="video test: the still to animate")
    a.add_argument("--seconds", type=float, default=2.0, help="video test: length of the take")
    a.add_argument("--train-lora", default="", metavar="NAME", help="train a LoRA called NAME from --images")
    a.add_argument("--images", default="", help="training: folder of 15-30 images (an optional NAME.txt caption beside each)")
    a.add_argument("--trigger", default="", help="training: the trigger word (default mchz_<NAME>)")
    a.add_argument("--caption", default="", help="training: caption for images without their own .txt")
    a.add_argument("--train-steps", type=int, default=0, help="training: total steps (default config mac.train.steps_total)")
    a.add_argument("--scale", type=float, default=0.8, help="training: how strongly the engines apply the LoRA")
    a.add_argument("--dry-run", action="store_true", help="training: build and check the config, do not train")
    a.add_argument("--engines", default="", help="another engine set from config mac.engines_<name>, e.g. zimage_base")
    a.add_argument("--engine-set", default="gpu", choices=["gpu", "stub"])
    a.add_argument("--work", default=str(WORK))
    a.add_argument("--repo", default=os.environ.get("MOCHI_REPO", ""), help="owner/repo (default: this clone's)")


def _repo(p, given: str) -> str:
    if given:
        return given
    r = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                       capture_output=True, text=True, cwd=p.root)
    return r.stdout.strip()


def _mode(args) -> str:
    return ("train" if args.train_lora else "video-test" if args.video_test else
            "self-test" if args.self_test else "worker")


def run(p, args) -> int:
    """The mac-worker command. Exit codes: 0 done, 2 bad arguments, 3 the test or training failed."""
    if os.environ.get("MOCHI_RAW") != "1":           # structured lines only; MOCHI_RAW=1 shows the tools' own bars
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    lf = set_file(Path(args.work) / "logs" / time.strftime("worker-%Y%m%d.jsonl"))
    cfg = settings(p, local_loras=LOCAL_LORAS)
    if args.engines:
        try:
            cfg = with_engine_set(cfg, args.engines)
        except KeyError as e:
            log(str(e).strip("'\""), event="worker.bad_args", level="error")
            return 2
    mode = _mode(args)
    log(f"mac worker: {mode}, engines {args.engines or 'default'} ({args.engine_set}); JSON log {lf}",
        event="worker.start", mode=mode, engines=args.engines or "default", loras=len(cfg.get("loras", [])),
        log_file=str(lf))
    engines = build(cfg, args.engine_set)
    work = Path(args.work)

    if mode == "train":
        if not args.images:
            log("--train-lora needs --images DIR", event="worker.bad_args", level="error")
            return 2
        try:
            r = train_lora(cfg, args.train_lora, Path(args.images), trigger=args.trigger or None,
                           caption=args.caption, preview=key_art_prompt(p)[1]["prompt"],
                           steps_total=args.train_steps or None, scale=args.scale, dry_run=args.dry_run)
        except Exception as e:  # noqa: BLE001
            log(f"LoRA {args.train_lora} failed: {e}", event="train.failed", level="error", error=str(e)[-800:])
            return 3
        print(json.dumps(r, indent=1))
        return 0
    if mode == "video-test":
        still = Path(args.image).expanduser() if args.image else work / "selftest-style.png"
        r = video_test(p, engines, still, work / "selftest-take.mp4", seconds=args.seconds)
        print(json.dumps(r, indent=1))
        return 0 if r["video"]["ok"] else 3
    if mode == "self-test":
        name = "selftest-style.png" if args.look == "3d" else "selftest-style-2d.png"
        r = self_test(p, engines, work / name, edit_only=args.edit_only, look=args.look, variants=args.variants)
        print(json.dumps(r, indent=1))
        return 0 if all(v.get("ok", True) for v in r.values()) else 3

    repo = _repo(p, args.repo)
    if not repo:
        log("could not tell the repository: run inside the clone with gh logged in, or pass --repo owner/repo",
            event="worker.bad_args", level="error")
        return 2
    log(f"waiting for batches for {repo} (Ctrl-C to stop)", event="worker.waiting", repo=repo)
    n = worker_loop(Release(repo), work, engines, cfg, once=args.once)
    log(f"{n} batch(es) done", event="worker.stop", batches=n)
    return 0
