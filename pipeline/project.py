"""Load the project: config.json, shots.csv, prompts and the media folders."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

CHARACTER_AGES = {1: 7, 2: 8, 3: 27, 4: 87, 5: 87}   # Tsumugi's age by chapter


@dataclass(frozen=True)
class Shot:
    id: str
    chapter: int
    shot: str
    seconds: int
    type: str
    model: str
    takes: int
    sound: str


def wan_frames(seconds: float, fps: int, max_frames: int) -> int:
    """Wan needs 4k+1 frames. Smallest such count covering the shot, capped by the model."""
    need = max(1, math.ceil(seconds * fps))
    frames = ((need - 1 + 3) // 4) * 4 + 1
    cap = ((max_frames - 1) // 4) * 4 + 1
    return max(5, min(frames, cap))


def wan_segments(seconds: float, fps: int, max_frames: int) -> list[int]:
    """Frame counts of the clips that cover a shot. Each is 4k+1 frames and at most max_frames.

    Long shots are chained: each next clip starts from the previous clip's last frame,
    so it shares one frame with it. Total new frames >= the shot's length.
    """
    need = max(1, math.ceil(seconds * fps))
    cap = ((max_frames - 1) // 4) * 4 + 1
    first = wan_frames(seconds, fps, max_frames)
    segs, have = [first], first
    while have < need:
        nxt = wan_frames((need - have + 1) / fps, fps, max_frames)
        segs.append(nxt)
        have += nxt - 1
    return segs


class Project:
    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        with open(self.root / "shots.csv", newline="", encoding="utf-8") as f:
            self.shots = [
                Shot(r["id"], int(r["chapter"]), r["shot"], int(r["seconds"]),
                     r["type"], r["model"], int(r["takes"]), r.get("sound", ""))
                for r in csv.DictReader(f)
            ]
        unknown = {s.model for s in self.shots} - set(self.config["models"])
        if unknown:
            raise ValueError(f"shots.csv names models missing from config.json: {sorted(unknown)}")
        self.media = self.root / "media"
        self.store = None          # where media lives when it is not on this machine (pipeline/store.py)

    # ---- files. The ledger keeps paths relative to the project, so it can move between machines.
    def rel(self, path) -> str:
        path = Path(path)
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def fetch(self, path) -> Path:
        """Absolute path of a media file, downloaded from the store first if it is not here."""
        f = Path(path)
        f = f if f.is_absolute() else self.root / f
        if not f.exists() and self.store is not None:
            self.store.fetch(self, f)
        return f

    # ---- lookups
    def chapter(self, n: int | None) -> list[Shot]:
        return [s for s in self.shots if n is None or s.chapter == n]

    def shot(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise KeyError(shot_id)

    def model(self, name: str) -> dict:
        return self.config["models"][name]

    def provider_of(self, shot: Shot) -> str:
        return self.model(shot.model).get("provider", "editor")

    def engine(self, name: str) -> dict:
        return self.config.get("engines", {}).get(name, {})

    # ---- money
    def billed_seconds(self, shot: Shot) -> int:
        """Providers bill whole clips, and most have a minimum length."""
        m = self.model(shot.model)
        if not m.get("generates_video"):
            return 0
        return max(math.ceil(shot.seconds), int(m.get("min_seconds", 0)))

    def take_cost(self, shot: Shot) -> float:
        return round(self.billed_seconds(shot) * self.model(shot.model)["rate_usd_per_s"], 4)

    def keyframe_cost(self) -> float:
        return float(self.config["keyframes"]["price_usd"])

    def variants(self, shot: Shot) -> list[str]:
        return list(self.model(shot.model).get("keyframe_variants", []))

    # ---- GPU time (estimates; the ledger records what each job really took)
    def generated_seconds(self, shot: Shot) -> float:
        """Length of clip the model actually makes, after the 4k+1 frame rule and its cap."""
        eng = self.engine(shot.model)
        if not eng:
            return float(self.billed_seconds(shot))
        return sum(wan_segments(shot.seconds, eng["fps"], eng["max_frames"])) / eng["fps"]

    def gpu_estimate(self, kind: str, shot: Shot) -> float:
        if kind in ("keyframe", "variant", "ref"):
            return float(self.config["keyframes"].get("gpu_seconds_each", 0))
        if kind == "take":
            per = self.model(shot.model).get("gpu_seconds_per_output_second", 0)
            return per * self.generated_seconds(shot)
        if kind == "upscale":
            up = self.engine("upscale")
            if self.model(shot.model).get("generates_video"):
                return up.get("gpu_seconds_per_output_second", 0) * self.generated_seconds(shot)
            return 10.0
        return 0.0

    # ---- reference stills (gate 1)
    def ref_plan(self) -> list[dict]:
        f = self.root / "style_bible" / "ref_prompts.json"
        return json.loads(f.read_text(encoding="utf-8")).get("refs", []) if f.exists() else []

    def ref_path(self, name: str) -> Path:
        return self.root / "style_bible" / "refs" / f"{name}.png"

    def refs_missing(self) -> list[dict]:
        """Planned reference stills that are not in style_bible/refs yet."""
        return [r for r in self.ref_plan() if not self.ref_path(r["name"]).exists()]

    # ---- prompts
    def prompt(self, shot: Shot, extra: str = "") -> str:
        style = self.root / "style_bible" / "style.txt"
        own = self.root / "prompts" / f"{shot.id}.txt"
        chapters = self.root / "style_bible" / "chapter_styles.json"
        chapter_line = json.loads(chapters.read_text(encoding="utf-8")).get(str(shot.chapter), "") \
            if chapters.exists() else ""
        parts = [style.read_text(encoding="utf-8").strip() if style.exists() else "",
                 chapter_line,
                 own.read_text(encoding="utf-8").strip() if own.exists() else shot.shot,
                 extra]
        return "\n".join(p for p in parts if p)

    def references(self) -> list[Path]:
        refs = self.root / "style_bible" / "refs"
        return sorted(p for p in refs.glob("*") if p.is_file()
                      and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) if refs.exists() else []

    def refs_for(self, shot: Shot, limit: int = 3) -> list[Path]:
        """Pick reference stills for a shot by name.

        Files are named after what they show: tsumugi_ch1.png, sayo.png, akari.png,
        bowl_ch3.png, style.png. A name that appears in the shot text is used, and a
        chapter-specific file beats a general one. style.* is always included.
        """
        text = f"{shot.shot} {shot.sound}".lower()
        by_name: dict[str, list[Path]] = {}
        style = []
        for ref in self.references():
            stem = ref.stem.lower()
            if stem.startswith("style"):
                style.append(ref)
                continue
            name = re.split(r"[_\-]", stem)[0]
            by_name.setdefault(name, []).append(ref)
        chosen: list[Path] = []
        for name, files in sorted(by_name.items()):
            if name not in text and not (name == "tsumugi" and re.search(r"\b(she|her|child|girl)\b", text)):
                continue
            exact = [f for f in files if f"ch{shot.chapter}" in f.stem.lower()]
            chosen.append((exact or sorted(files))[0])
        chosen = chosen[: max(0, limit - (1 if style else 0))]
        return (style[:1] + chosen)[:limit]


REF_CHAPTER = 0          # reference stills belong to no chapter


def ref_shot(name: str) -> Shot:
    return Shot(f"ref-{name}", REF_CHAPTER, name, 0, "Ref", "Editor", 0, "")


def job_key(shot_id: str, kind: str, idx: int, prompt: str) -> str:
    """Same shot, kind, index and prompt is the same job: it is never made twice."""
    h = hashlib.sha1(f"{shot_id}|{kind}|{idx}|{prompt}".encode("utf-8")).hexdigest()[:10]
    return f"{shot_id}.{kind}{idx}.{h}"
