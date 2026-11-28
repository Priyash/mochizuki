"""Load a film: its config.json, shots.csv, style bible, prompts and media folders.

The engine (this package) holds no film. Each film is a folder, films/<name>/:

  config.json     models, engines, backends, caps, and the film's cast for the 3D pipeline
  shots.csv       the shot sheet
  style_bible/    the look: style, reference stills, scenes, acting sheets, sets, lights, dialogue
  prompts/        optional per-shot prompt files
  assets/         the film's own character models (never committed)
  media/          everything the engine makes for it, and its ledger (not committed)

`find_film` picks the folder: --film, else $FILM, else the only film in films/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generators import wan_frames, wan_segments  # noqa: F401 - re-exported: engine.project.wan_frames

REPO = Path(__file__).resolve().parents[1]
FILMS = REPO / "films"
FILM_ENV = "FILM"                      # environment variable naming the film's folder

# The layout of a film folder. The names are spelled here only: code uses these constants or the
# Project attributes built from them.
CONFIG_FILE = "config.json"
SHOTS_FILE = "shots.csv"
STYLE_BIBLE_DIR = "style_bible"
REFS_DIR = "refs"                      # inside the style bible
PROMPTS_DIR = "prompts"
ASSETS_DIR = "assets"
MEDIA_DIR = "media"
LEDGER_FILE = "ledger.sqlite"
# ... and the style bible's files
STYLE_FILE = "style.txt"
SCENES_FILE = "scenes.json"
REF_PROMPTS_FILE = "ref_prompts.json"
CHAPTER_STYLES_FILE = "chapter_styles.json"
ACTING_FILE = "acting.json"
SETS_FILE = "sets.json"
LIGHTS_FILE = "lights.json"
DIALOGUE_FILE = "dialogue.json"


def find_film(given: str | Path | None = None) -> Path:
    """The film folder to work on: `given` (--film), else $FILM, else the only film in films/."""
    choice = given or os.environ.get(FILM_ENV)
    if choice:
        d = Path(choice).expanduser()
        if not d.is_absolute() and not d.exists():
            d = REPO / d                          # a path relative to the repository, from anywhere
        if not (d / CONFIG_FILE).exists():
            raise FileNotFoundError(f"{d} is not a film folder: it has no {CONFIG_FILE}")
        return d.resolve()
    films = sorted(f.parent for f in FILMS.glob(f"*/{CONFIG_FILE}"))
    if len(films) == 1:
        return films[0].resolve()
    if not films:
        raise FileNotFoundError(f"no film in {FILMS}: make films/<name>/ with {CONFIG_FILE} and {SHOTS_FILE}")
    raise ValueError(f"{len(films)} films: choose one with --film or {FILM_ENV} "
                     f"({', '.join(f.name for f in films)})")


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


class Project:
    """One film. `root` is its folder (see find_film); every path the ledger keeps is relative to it."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()
        self.name = self.root.name
        self.config_file = self.root / CONFIG_FILE
        self.shots_file = self.root / SHOTS_FILE
        self.style_bible = self.root / STYLE_BIBLE_DIR
        self.style_file = self.style_bible / STYLE_FILE
        self.refs_dir = self.style_bible / REFS_DIR
        self.prompts_dir = self.root / PROMPTS_DIR
        self.assets_dir = self.root / ASSETS_DIR
        self.media = self.root / MEDIA_DIR
        self.ledger_file = self.root / LEDGER_FILE
        self.config = json.loads(self.config_file.read_text(encoding="utf-8"))
        with open(self.shots_file, newline="", encoding="utf-8") as f:
            self.shots = [
                Shot(r["id"], int(r["chapter"]), r["shot"], int(r["seconds"]),
                     r["type"], r["model"], int(r["takes"]), r.get("sound", ""))
                for r in csv.DictReader(f)
            ]
        unknown = {s.model for s in self.shots} - set(self.config["models"])
        if unknown:
            raise ValueError(f"{SHOTS_FILE} names models missing from {CONFIG_FILE}: {sorted(unknown)}")
        self.store: Any = None     # where media lives when it is not on this machine (store.py)

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

    # ---- scenes: coherence inside a scene and across matched cuts
    def scenes(self) -> dict:
        f = self.style_bible / SCENES_FILE
        return json.loads(f.read_text(encoding="utf-8")).get("scenes", {}) if f.exists() else {}

    def scene_of(self, shot: Shot) -> tuple[str | None, dict]:
        for sid, sc in self.scenes().items():
            if shot.id in sc.get("shots", []):
                return sid, sc
        return None, {}

    def continuity_source(self, shot: Shot) -> str | None:
        """The shot whose chosen keyframe this shot's keyframe is drawn from, if any.

        A match cut ("... become(s) ...") follows the shot before it; any other shot follows
        its scene's anchor; an anchor follows the anchor of the scene it comes 'after'.
        """
        order = [s.id for s in self.shots]
        i = order.index(shot.id)
        if re.search(r"\bbecomes?\b", shot.shot.lower()) and i > 0:
            return order[i - 1]
        sid, sc = self.scene_of(shot)
        if not sid:
            return None
        if sc.get("anchor") and sc["anchor"] != shot.id:
            return sc["anchor"]
        after = self.scenes().get(sc.get("after", ""), {})
        return after.get("anchor")

    # ---- reference stills (gate 1)
    def ref_plan(self) -> list[dict]:
        f = self.style_bible / REF_PROMPTS_FILE
        return json.loads(f.read_text(encoding="utf-8")).get("refs", []) if f.exists() else []

    def ref_path(self, name: str) -> Path:
        return self.refs_dir / f"{name}.png"

    @staticmethod
    def ref_chapter(r: dict) -> int:
        """The first chapter a planned still is needed for: its _chN suffix, else chapter 1."""
        m = re.search(r"_ch(\d+)$", r["name"])
        return int(r.get("chapter") or (m.group(1) if m else 1))

    def refs_missing(self, chapter: int | None = None) -> list[dict]:
        """Planned reference stills not in style_bible/refs yet (only those needed up to `chapter`)."""
        return [r for r in self.ref_plan() if not self.ref_path(r["name"]).exists()
                and (chapter is None or self.ref_chapter(r) <= chapter)]

    # ---- prompts
    def prompt(self, shot: Shot, extra: str = "") -> str:
        style = self.style_file
        own = self.prompts_dir / f"{shot.id}.txt"
        chapters = self.style_bible / CHAPTER_STYLES_FILE
        chapter_line = json.loads(chapters.read_text(encoding="utf-8")).get(str(shot.chapter), "") \
            if chapters.exists() else ""
        scene_line = self.scene_of(shot)[1].get("setting", "")
        parts = [style.read_text(encoding="utf-8").strip() if style.exists() else "",
                 chapter_line, scene_line,
                 own.read_text(encoding="utf-8").strip() if own.exists() else shot.shot,
                 extra]
        return "\n".join(p for p in parts if p)

    def references(self) -> list[Path]:
        refs = self.refs_dir
        return sorted(p for p in refs.glob("*") if p.is_file()
                      and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) if refs.exists() else []

    def refs_for(self, shot: Shot, limit: int = 3) -> list[Path]:
        """Pick the reference stills a shot's keyframe is drawn from.

        style.* always goes first. Every other still is a subject (tsumugi, sayo, bowl,
        hearth ...) in one or more chapter versions (tsumugi_ch1, tsumugi_ch3 ...). A subject
        is used when its name, or a word in its ref_prompts.json "match" list, appears in
        the shot; the version is the latest one at or before the shot's chapter, so Tsumugi
        ages with the story. Lower "priority" wins the remaining slots (characters first).
        """
        text = f"{shot.shot} {shot.sound}".lower()
        plan = {r["name"]: r for r in self.ref_plan()}
        style: list[Path] = []
        subjects: dict[str, list[tuple[int, Path, dict]]] = {}
        for ref in self.references():
            stem = ref.stem.lower()
            if stem.startswith("style"):
                style.append(ref)
                continue
            m = re.match(r"^(.*?)(?:_ch(\d+))?$", stem)
            assert m is not None                      # the pattern matches any name
            subjects.setdefault(m.group(1), []).append((int(m.group(2) or 0), ref, plan.get(stem, {})))
        chosen = []
        for subject, files in subjects.items():
            words = {subject.split("_")[0]} | {w for _, _, e in files for w in e.get("match", [])}
            hit = any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)
            if not hit:
                continue
            eligible = [f for f in files if f[0] <= shot.chapter] or files
            ch, path, _ = max(eligible, key=lambda f: (f[0], f[1].name))
            prio = min((e.get("priority", 2) for _, _, e in files), default=2)
            chosen.append((prio, subject, path))
        chosen.sort(key=lambda c: (c[0], c[1]))
        return (style[:1] + [path for _, _, path in chosen])[:limit]


REF_CHAPTER = 0          # reference stills belong to no chapter


def ref_shot(name: str) -> Shot:
    return Shot(f"ref-{name}", REF_CHAPTER, name, 0, "Ref", "Editor", 0, "")


def job_key(shot_id: str, kind: str, idx: int, prompt: str) -> str:
    """Same shot, kind, index and prompt is the same job: it is never made twice."""
    h = hashlib.sha1(f"{shot_id}|{kind}|{idx}|{prompt}".encode("utf-8")).hexdigest()[:10]
    return f"{shot_id}.{kind}{idx}.{h}"
