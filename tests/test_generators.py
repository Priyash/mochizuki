"""The generator catalog, the worker's registry, and which provider may run which stage."""
from __future__ import annotations

import importlib.util

import pytest

from engine import generators as gen
from engine import scheduler as sched
from engine import steps
from engine.providers import colab as cb
from engine.providers.base import WORKER
from engine.providers.kaggle import KaggleProvider, LocalKaggle
from engine.providers.mac import MacProvider

from test_pipeline import copy_project
from test_mac import FakeRelease


def _worker():
    spec = importlib.util.spec_from_file_location("worker_under_test", WORKER)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    return w


def test_every_gpu_stage_has_a_built_generator():
    for stage in sched.STAGES:
        assert any(g.built for g in gen.for_stage(stage)), stage


def test_the_worker_and_the_catalog_name_the_same_generators():
    w = _worker()
    built_gpu = {g.name for g in gen.CATALOG.values() if g.built and gen.GPU in g.runs_on}
    assert set(w.GENERATORS) == built_gpu


def test_the_worker_picks_a_generator_by_name_or_by_takes_and_never_guesses():
    w = _worker()
    assert w.generator_for("refs", [{"kind": "ref"}]) == "refs"
    assert w.generator_for("refs_fp16", [{"kind": "ref"}]) == "refs"            # the probe's variants
    assert w.generator_for("keyframe", [{"kind": "variant"}]) == "keyframe"
    assert w.generator_for("Wan 2.2 14B", [{"kind": "take"}, {"kind": "take"}]) == "video"
    with pytest.raises(ValueError, match="no generator"):
        w.generator_for("Something New", [{"kind": "keyframe"}])               # once, this ran Wan


def test_shared_rules_match_the_workers_copies():
    w = _worker()
    for key in ("1-05.take0.abc", "x", "chapter 4"):
        assert w.seed_for(key) == gen.seed_for(key)


def test_kaggle_never_takes_a_stage_its_t4s_cannot_run(tmp_path):
    proj = copy_project(tmp_path, cfg_edit=lambda c: c["kaggle"].update(stages=["keyframes", "video"]))
    led = steps.open_ledger(proj)
    kaggle = KaggleProvider(proj, runner=lambda args: (1, "the kaggle command must not be called"))
    r = sched.tick(proj, led, kaggle, 1, engine_set="stub")
    assert r["action"] == "idle" and "keyframes needs bf16" in r["reason"], r
    assert not led.active_batch("kaggle")


def test_which_providers_have_bf16(tmp_path):
    proj = copy_project(tmp_path)
    fr = FakeRelease(tmp_path / "release")
    assert gen.BF16 not in KaggleProvider(proj).capabilities
    assert gen.BF16 in LocalKaggle(proj.media).capabilities                      # stand-ins run anything
    assert gen.BF16 in MacProvider(proj, "o/r", gh=fr.gh).capabilities
    colab = cb.ColabProvider(proj, "o/r", gh=fr.gh)
    assert gen.BF16 in colab.capabilities                                        # A100 and L4
    colab.cfg = {**colab.cfg, "gpus": ["A100", "T4"]}
    assert gen.BF16 not in colab.capabilities                                    # a T4 could come up instead


def test_stage_needs_come_from_the_catalog():
    assert gen.needs_for("refs") == {gen.BF16} and gen.needs_for("keyframes") == {gen.BF16}
    assert gen.needs_for("video") == frozenset() and gen.needs_for("upscale") == frozenset()
    assert gen.needs_for("finish") == frozenset()                                # not built: needs nothing yet
