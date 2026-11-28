"""A stage of making a film, and who runs it.

A stage consumes artifacts (the film's own files, or what earlier stages made) and produces new ones.
The chain (chain.py) is every stage, ordered by those dependencies; the owner's approvals (gates) sit
between them. Every stage works on one chapter at a time, and on the film as all its chapters.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Who runs a stage
SCHEDULER = "scheduler"   # GPU work: the producer sends it to a provider (engine/scheduler, engine/providers)
CGI = "cgi"               # CPU work: `python -m engine cgi auto` (GitHub runners or your Mac)
ASSEMBLE = "assemble"     # the cut: `python -m engine assemble`, or the 3D pipeline's finish step

# The owner's approvals. A stage with a gate runs only after it.
REFS_GATE = "refs"            # the reference stills become the film's look (/approve refs)
BOARD_GATE = "board"          # the chapter's keyframes (/approve board N)
ANIMATIC_GATE = "animatic"    # the chapter's cut (/approve animatic N)


@dataclass(frozen=True)
class Stage:
    name: str
    does: str                                 # one line: what it does
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    runner: str
    generators: tuple[str, ...] = ()          # engine.generators names
    gate: str | None = None                   # the approval its inputs need first
    skips_unchanged: str = ""                 # how it avoids redoing work that has not changed
    built: bool = True
    # For scheduler stages: is there work now, does the chain stop here for a person, reserve its jobs
    wants: Callable | None = None             # (p, led, chapter) -> bool
    halts: Callable | None = None             # (p, led, chapter) -> bool
    plan: Callable | None = None              # (p, led, executor, chapter, only) -> summary (engine/steps)
    # For `python -m engine chain`: where the chapter stands
    progress: Callable | None = None          # (p, led, chapter) -> short text
