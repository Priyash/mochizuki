"""Takes on the Mac: image to video with Wan 2.2 TI2V-5B through mlx-video (MLX, native on Apple Silicon).

The model folder (about 19 GB, 8-bit) is fetched once into ~/mochizuki-mac/models. Shots longer
than one clip are chained like on Kaggle: each next clip starts from the previous clip's last
frame, and the clips are joined without repeating that frame.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ...generators import wan_frames as frames_for  # noqa: F401 - engine.mac.generators exports it
from ...generators import wan_segments as segments
from ..runtime.downloads import resolve_cmd
from ..runtime.shared import MODELS, ffmpeg_exe, log, seed_for
from ..runtime.tools import run_streaming
from .base import check_video


class MlxVideoEngine:
    """One mlx-video Wan 2.2 run per clip; chained clips for longer shots."""

    def __init__(self, cfg: dict, models: Path | None = None):
        self.cfg = cfg
        self.e = cfg["engines"]["video"]
        self.models = models or MODELS
        self._cmd: list[str] | None = None
        try:
            import mlx_video  # noqa: F401, PLC0415
        except ImportError as err:
            raise RuntimeError("mlx-video is not installed: pip install git+https://github.com/Blaizzy/mlx-video.git "
                               "(see docs/mac-worker.md)") from err

    def cmd(self) -> list[str]:
        if self._cmd is None:
            self._cmd = resolve_cmd(self.e, self.models)
        return self._cmd

    def _fit(self, image: Path, dest: Path) -> Path:
        """The start image scaled and centre-cropped to the model's size."""
        w, h = int(self.e["width"]), int(self.e["height"])
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(image), "-vf",
                        f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}", str(dest)], check=True)
        return dest

    def _last_frame(self, clip: Path, dest: Path) -> Path:
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-sseof", "-0.1", "-i", str(clip), "-frames:v", "1",
                        "-update", "1", str(dest)], check=True)
        return dest

    def _join(self, clips: list[Path], out: Path, fps: int) -> None:
        """Join clips, dropping the first frame of every clip after the first (it repeats the last one)."""
        if len(clips) == 1:
            shutil.copyfile(clips[0], out)
            return
        inputs, chains = [], []
        for i, c in enumerate(clips):
            inputs += ["-i", str(c)]
            trim = "" if i == 0 else "trim=start_frame=1,setpts=PTS-STARTPTS,"
            chains.append(f"[{i}:v]{trim}fps={fps}[v{i}]")
        graph = ";".join(chains) + ";" + "".join(f"[v{i}]" for i in range(len(clips))) + \
            f"concat=n={len(clips)}:v=1:a=0[out]"
        subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", str(out)],
                       check=True)

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        fps, max_frames = int(self.e.get("fps", 24)), int(self.e.get("max_frames", 81))
        segs = segments(float(job.get("seconds", 3)), fps, max_frames)
        image = inputs / job["image"]
        if not image.exists():
            raise FileNotFoundError(f"input {job['image']} missing from the batch")
        log(f"animating {job['key']}: {job.get('seconds', 3)} s as {len(segs)} clip(s) of {segs} frames, "
            f"{self.e['width']}x{self.e['height']} at {fps} fps", event="video.start", key=job["key"],
            clips=len(segs), frames=sum(segs) - len(segs) + 1)
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="mochi-video-") as tmp_name:
            tmp = Path(tmp_name)
            start = self._fit(image, tmp / "start-0.png")
            clips = []
            for n, frames in enumerate(segs):
                clip = tmp / f"clip-{n}.mp4"
                cmd = [*self.cmd(), "--image", str(start), "--prompt", job["prompt"],
                       "--width", str(self.e["width"]), "--height", str(self.e["height"]),
                       "--num-frames", str(frames), "--seed", str(seed_for(job["key"]) + n), "--output-path", str(clip),
                       *self.e.get("extra_args", [])]
                code, tail = run_streaming(cmd, float(self.e.get("seconds_per_clip", 1800)) * 2,
                                           label=f"{job['key']} clip {n + 1}/{len(segs)}:", kind="video",
                                           tool="mlx-video")
                if code != 0 or not clip.exists():
                    raise RuntimeError(f"mlx-video exit {code}: {tail.strip()[-600:]}")
                clips.append(clip)
                if n + 1 < len(segs):
                    start = self._last_frame(clip, tmp / f"start-{n + 1}.png")
            self._join(clips, out, fps)
        check_video(out)
        took = time.time() - t0
        log(f"{job['key']}: saved {out} ({took / 60:.1f} min)", event="video.done", key=job["key"],
            seconds=round(took), file=str(out))
        return {"seconds": round(took, 1), "segments": segs, "fps": fps, "width": self.e["width"],
                "height": self.e["height"], "seed": seed_for(job["key"]), "engine": "mlx-video wan2.2-ti2v-5b"}
