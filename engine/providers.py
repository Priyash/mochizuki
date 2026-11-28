"""Providers that run a job immediately: run(spec, out_path).

mock: free placeholder files made with ffmpeg, for testing the whole loop on any laptop.
fal:  paid fallback for hero shots. NOT run against the live service; see README.
Kaggle is not here: it runs jobs in batches, see engine/kaggle.py.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .project import Project

COLORS = {"ref": "0x744210", "keyframe": "0x4a5568", "variant": "0x2c7a7b", "take": "0x6b46c1", "upscale": "0x2f855a"}


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


class MockProvider:
    name = "mock"

    def __init__(self, project: Project):
        self.p = project
        self.has_ffmpeg = shutil.which("ffmpeg") is not None

    def run(self, spec: dict, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if not self.has_ffmpeg:
            out.write_bytes(f"mock {spec['kind']}".encode())
            return
        kind = spec["kind"]
        if kind in ("keyframe", "variant", "ref"):
            _ffmpeg(["-f", "lavfi", "-i", f"color=c={COLORS[kind]}:s=640x360", "-frames:v", "1", str(out)])
        elif kind == "take":
            _ffmpeg(["-loop", "1", "-i", str(self.p.fetch(spec["image"])), "-t", str(spec["seconds"]), "-r", "12",
                     "-vf", "scale=640:360,format=yuv420p", str(out)])
        elif kind == "upscale":
            shutil.copyfile(self.p.fetch(spec["source"]), out)
        else:
            raise ValueError(f"unknown job kind {kind}")


class FalProvider:
    """Paid fallback. Calls fal with the endpoints named in config.json.

    Endpoint ids and argument names differ by model and are not verified here.
    Check each one against its model page, then test on a single shot with --shot.
    """
    name = "fal"

    def __init__(self, project: Project):
        self.p = project
        self.cfg = project.config["providers"]["fal"]
        if not os.environ.get("FAL_KEY"):
            raise RuntimeError("FAL_KEY is not set. Copy .env.example to .env and fill it in.")
        try:
            import fal_client  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError("pip install fal-client") from e
        self.fal = fal_client

    @staticmethod
    def _download(url: str, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f)

    def run(self, spec: dict, out: Path) -> None:
        if spec["kind"] != "take":
            raise RuntimeError("the paid fallback only makes takes; keyframes and 4K run on Kaggle")
        v = self.cfg["video"][spec["model"]]
        if v["endpoint"].startswith("VERIFY"):
            raise RuntimeError(f"Set providers.fal.video['{spec['model']}'].endpoint in config.json first.")
        args = {"prompt": spec["prompt"], v["image_arg"]: self.fal.upload_file(str(self.p.fetch(spec["image"]))),
                v["duration_arg"]: max(3, int(spec["seconds"])), **v.get("args", {})}
        result = self.fal.subscribe(v["endpoint"], arguments=args)
        self._download(result["video"]["url"], out)


def get(name: str, project: Project):
    if name == "mock":
        return MockProvider(project)
    if name == "fal":
        return FalProvider(project)
    raise ValueError(f"unknown provider {name!r}")
