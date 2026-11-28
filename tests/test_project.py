"""Which film the engine works on: --film, else $FILM, else the only film in films/."""
from __future__ import annotations

import pytest

from engine import project
from engine.project import CONFIG_FILE, FILM_ENV, find_film


def _film(root, name):
    d = root / name
    d.mkdir(parents=True)
    (d / CONFIG_FILE).write_text("{}")
    return d


def test_the_only_film_is_chosen_by_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "FILMS", tmp_path)
    monkeypatch.delenv(FILM_ENV, raising=False)
    only = _film(tmp_path, "one")
    assert find_film() == only.resolve()


def test_several_films_need_a_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "FILMS", tmp_path)
    monkeypatch.delenv(FILM_ENV, raising=False)
    _film(tmp_path, "one")
    two = _film(tmp_path, "two")
    with pytest.raises(ValueError, match="choose one"):
        find_film()
    assert find_film(str(two)) == two.resolve()                      # --film
    monkeypatch.setenv(FILM_ENV, str(two))
    assert find_film() == two.resolve()                               # $FILM
    assert find_film(str(tmp_path / "one")) == (tmp_path / "one").resolve()   # --film wins over $FILM


def test_a_folder_without_a_config_is_not_a_film(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "FILMS", tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="not a film folder"):
        find_film(str(tmp_path / "empty"))
    monkeypatch.delenv(FILM_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="no film"):
        find_film()


def test_this_repository_has_one_film_and_the_engine_finds_it(monkeypatch):
    monkeypatch.delenv(FILM_ENV, raising=False)
    p = project.Project(find_film())
    assert p.name == "mochizuki" and p.shots and p.config_file.exists() and p.style_file.exists()
    assert p.root.parent == project.FILMS and p.media == p.root / project.MEDIA_DIR
