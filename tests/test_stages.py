"""The chain of stages: no loops, nothing consumed that nobody makes, and the scheduler follows it."""
from __future__ import annotations

import subprocess
import sys
from graphlib import CycleError

import pytest

from engine import generators as gen
from engine import scheduler as sched
from engine import stages, steps
from engine.project import REPO
from engine.stages import chain

from test_pipeline import copy_project


def test_the_chain_never_loops_and_every_stage_follows_what_it_consumes():
    order = [s.name for s in stages.order()]
    assert sorted(order) == sorted(s.name for s in stages.CHAIN)
    deps = stages.dependencies()
    for name in order:
        assert all(order.index(d) < order.index(name) for d in deps[name]), name


def test_everything_consumed_is_made_by_a_stage_or_given_by_the_film():
    made = [a for s in stages.CHAIN for a in s.produces]
    assert len(made) == len(set(made)), "an artifact with two makers"
    for s in stages.CHAIN:
        for a in s.consumes:
            assert a in made or a in stages.FILM_INPUTS, (s.name, a)


def test_a_loop_is_refused(monkeypatch):
    a = stages.Stage("a", "x", consumes=("b out",), produces=("a out",), runner=stages.CGI)
    b = stages.Stage("b", "x", consumes=("a out",), produces=("b out",), runner=stages.CGI)
    monkeypatch.setattr(chain, "CHAIN", (a, b))
    monkeypatch.setattr(chain, "BY_NAME", {"a": a, "b": b})
    with pytest.raises(CycleError):
        chain.order()


def test_stages_and_generators_name_each_other():
    for s in stages.CHAIN:
        for g in s.generators:
            assert g in gen.CATALOG and gen.CATALOG[g].stage == s.name, (s.name, g)
            assert gen.CATALOG[g].built == s.built, (s.name, g)
    for g in gen.CATALOG.values():
        assert g.stage in stages.BY_NAME, g.name


def test_the_scheduler_runs_the_chains_gpu_stages_in_its_order():
    assert sched.STAGES == ("refs", "keyframes", "video", "upscale")
    for s in stages.CHAIN:
        if s.built:
            assert s.progress, s.name
        if s.runner == stages.SCHEDULER and s.built:
            assert s.wants and s.plan, s.name


def test_the_report_covers_every_stage_for_the_chapters_asked(tmp_path):
    proj = copy_project(tmp_path)
    led = steps.open_ledger(proj)
    text = stages.report(proj, led, [1, 2])
    for s in stages.CHAIN:
        assert s.name in text
    assert "not built" in text and "chapter 2" in text
    assert "next GPU stage: chapter 1: keyframes" in text            # stand-in stills are in refs/ already


def test_the_chain_command(tmp_path):
    proj = copy_project(tmp_path)
    r = subprocess.run([sys.executable, "-m", "engine", "--film", str(proj.root), "chain", "--chapter", "1"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    assert "next GPU stage: chapter 1:" in r.stdout and "ai_finish" in r.stdout
