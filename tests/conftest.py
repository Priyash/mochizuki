"""Shared test helpers: the film the tests read, and a copy of it each test may change."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from engine.project import CONFIG_FILE, FILMS, SHOTS_FILE, STYLE_BIBLE_DIR, Project

FILM = FILMS / "mochizuki"     # the tests check this film's real shot sheet, style bible and config


def copy_film(dest: Path) -> Path:
    """The film's config, shot sheet and style bible, copied into `dest`."""
    shutil.copy(FILM / CONFIG_FILE, dest / CONFIG_FILE)
    shutil.copy(FILM / SHOTS_FILE, dest / SHOTS_FILE)
    shutil.copytree(FILM / STYLE_BIBLE_DIR, dest / STYLE_BIBLE_DIR)
    return dest


@pytest.fixture()
def proj(tmp_path) -> Project:
    """A copy of the film to change freely. (test_pipeline.py has its own, with stand-in stills.)"""
    return Project(copy_film(tmp_path))
