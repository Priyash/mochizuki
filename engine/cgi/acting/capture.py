"""A recorded human performance, as curves the rig can play.

The most emotional shots are acted, not generated: a person performs the shot (on an
iPhone with the free Live Link Face app, which records Apple's 52 ARKit face shapes and the
head's rotation at 60 fps as a CSV; or on any camera, read by Google's MediaPipe face
landmarker, Apache 2.0, which estimates the same 52 shapes on the Mac). Both become a
Track: names, fps and one row of values per frame, smoothed with the One Euro filter
(Casiez et al., CHI 2012) and resampled to the film's 24 fps.

Body motion is the next step (docs/cgi-pipeline.md): SAM 3D Body on a GPU, or MediaPipe pose.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ARKIT = ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
         "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
         "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
         "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
         "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
         "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
         "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
         "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
         "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
         "noseSneerLeft", "noseSneerRight", "tongueOut"]
HEAD = ["headYaw", "headPitch", "headRoll"]
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
                  "float16/1/face_landmarker.task")


def camel(name: str) -> str:
    """EyeBlinkLeft (Live Link Face) -> eyeBlinkLeft (ARKit, MediaPipe)."""
    return name[:1].lower() + name[1:] if name else name


@dataclass
class Track:
    fps: float
    names: list
    values: list                   # one list of floats per frame
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.values)

    @property
    def seconds(self) -> float:
        return len(self.values) / self.fps if self.fps else 0.0

    def channel(self, name: str, default: float = 0.0) -> list[float]:
        if name not in self.names:
            return [default] * len(self.values)
        i = self.names.index(name)
        return [row[i] for row in self.values]

    def to_dict(self) -> dict:
        return {"fps": self.fps, "names": self.names, "values": [[round(x, 4) for x in r] for r in self.values],
                "source": self.source, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> Track:
        return cls(float(d["fps"]), list(d["names"]), [list(map(float, r)) for r in d["values"]],
                   d.get("source", ""), d.get("meta", {}))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Track:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- readers
def _timecode_seconds(tc: str, fps: float) -> float:
    """Live Link Face timecode HH:MM:SS:FF.fff (FF counts frames at the recording rate)."""
    parts = tc.strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"not a Live Link Face timecode: {tc!r}")
    h, m, s, ff = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
    return h * 3600 + m * 60 + s + ff / fps


def read_live_link_csv(path: Path, fps: float = 60.0) -> Track:
    """A Live Link Face recording (the app's CSV). Missing shapes read as 0."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    cols = {camel(k.strip()): k for k in rows[0] if k}
    names = [n for n in ARKIT + HEAD if n in cols]
    if not any(n in cols for n in ARKIT):
        raise ValueError(f"{path} has none of the 52 ARKit face shapes; is it a Live Link Face CSV?")
    times = []
    if "timecode" in cols:
        try:
            times = [_timecode_seconds(r[cols["timecode"]], fps) for r in rows]
        except ValueError:
            times = []
    values = [[float(r[cols[n]] or 0) for n in names] for r in rows]
    tr = Track(fps, names, values, source=str(path), meta={"kind": "face", "from": "live-link-face"})
    if times and times[-1] > times[0]:
        span = times[-1] - times[0]
        tr.fps = round((len(rows) - 1) / span, 3) if span > 0 else fps
    return tr


def _probe(video: Path) -> dict:
    exe = shutil.which("ffprobe")
    if not exe:
        raise RuntimeError("ffprobe is not installed (it comes with ffmpeg: brew install ffmpeg)")
    r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate:stream_side_data=rotation", "-of", "json", str(video)],
                       capture_output=True, text=True, check=True)
    st = json.loads(r.stdout)["streams"][0]
    num, den = (st.get("r_frame_rate") or "30/1").split("/")
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        rot = int(sd.get("rotation", 0) or 0)
    w, h = int(st["width"]), int(st["height"])
    if abs(rot) in (90, 270):
        w, h = h, w
    return {"width": w, "height": h, "fps": float(num) / float(den or 1)}


def _frames_rgb(video: Path, w: int, h: int):
    """Decoded frames as raw RGB bytes, through ffmpeg (no OpenCV needed)."""
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={w}:{h}",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    assert proc.stdout is not None
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield buf
    finally:
        proc.stdout.close()
        proc.wait()


def head_angles(m) -> tuple[float, float, float]:
    """Yaw, pitch, roll (radians) of MediaPipe's 4x4 facial transformation matrix."""
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    pitch = math.asin(max(-1.0, min(1.0, -r[1][2])))
    yaw = math.atan2(r[0][2], r[2][2])
    roll = math.atan2(r[1][0], r[1][1])
    return yaw, pitch, roll


def face_from_video(video: Path, model: Path, long_side: int = 720) -> Track:
    """The 52 face shapes and head rotation of the person in `video`, frame by frame (MediaPipe).
    Frames with no face keep the previous values."""
    try:
        import mediapipe as mp  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError("face capture from video needs MediaPipe: pip install mediapipe") from e
    info = _probe(video)
    k = long_side / max(info["width"], info["height"])
    w, h = int(info["width"] * k) // 2 * 2, int(info["height"] * k) // 2 * 2
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    names = ARKIT + HEAD
    values, last, found = [], [0.0] * len(names), 0
    with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, buf in enumerate(_frames_rgb(video, w, h)):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.frombuffer(buf, np.uint8).reshape(h, w, 3))
            res = lm.detect_for_video(img, int(i * 1000 / info["fps"]))
            if res.face_blendshapes:
                found += 1
                score = {c.category_name: c.score for c in res.face_blendshapes[0]}
                row = [float(score.get(n, 0.0)) for n in ARKIT]
                row += list(head_angles(res.facial_transformation_matrixes[0])) if res.facial_transformation_matrixes \
                    else last[len(ARKIT):]
                last = row
            values.append(list(last))
    if not found:
        raise RuntimeError(f"no face found in {video}: frame the face larger, in even light")
    return Track(info["fps"], names, values, source=str(video),
                 meta={"kind": "face", "from": "mediapipe", "frames_with_face": found, "frames": len(values)})


def ensure_face_model(models_dir: Path) -> Path:
    """MediaPipe's face landmarker model (about 4 MB), downloaded once."""
    target = models_dir / "face_landmarker.task"
    if not target.exists():
        import urllib.request  # noqa: PLC0415
        models_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_MODEL_URL, target)       # noqa: S310 - a fixed https URL
    return target


# ---------------------------------------------------------------- cleaning
class OneEuro:
    """The One Euro filter: steady when still, quick when moving (Casiez, Roussel, Vogel 2012)."""

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x: float | None = None
        self.dx: float | None = None

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def __call__(self, x: float) -> float:
        if self.x is None:
            self.x, self.dx = x, 0.0
            return x
        dx = (x - self.x) * self.freq
        a_d = self._alpha(self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * (self.dx or 0.0)
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx))
        self.x = a * x + (1 - a) * self.x
        return self.x


def smooth(tr: Track, min_cutoff: float = 1.5, beta: float = 0.05) -> Track:
    filters = [OneEuro(tr.fps, min_cutoff, beta) for _ in tr.names]
    vals = [[filters[i](x) for i, x in enumerate(row)] for row in tr.values]
    return Track(tr.fps, list(tr.names), vals, tr.source, {**tr.meta, "smoothed": [min_cutoff, beta]})


def resample(tr: Track, fps: float, frames: int | None = None) -> Track:
    """Linear resampling to `fps`; `frames` pads (holding the last row) or trims to a length."""
    if not tr.values:
        return Track(fps, list(tr.names), [], tr.source, dict(tr.meta))
    n = frames if frames is not None else max(1, int(round(tr.seconds * fps)))
    out = []
    for f in range(n):
        x = f * tr.fps / fps
        i = int(math.floor(x))
        if i >= len(tr.values) - 1:
            out.append(list(tr.values[-1]))
            continue
        t = x - i
        a, b = tr.values[i], tr.values[i + 1]
        out.append([a[k] + (b[k] - a[k]) * t for k in range(len(a))])
    return Track(fps, list(tr.names), out, tr.source, {**tr.meta, "resampled_from": tr.fps})


def trim(tr: Track, start_s: float = 0.0, seconds: float | None = None) -> Track:
    a = int(round(start_s * tr.fps))
    b = len(tr.values) if seconds is None else a + int(round(seconds * tr.fps))
    return Track(tr.fps, list(tr.names), [list(r) for r in tr.values[a:b]], tr.source, dict(tr.meta))


def read_any(path: Path) -> Track:
    """A face performance from a Live Link Face CSV, a saved Track (.json) or a video."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".csv":
        return read_live_link_csv(path)
    if suf == ".json":
        return Track.load(path)
    raise ValueError(f"read a video with face_from_video(); {path.name} is not a .csv or .json track")
