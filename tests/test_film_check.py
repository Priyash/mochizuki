"""A film's own files are checked when it loads: every problem at once, each saying where it is."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import FILM, copy_film

from engine import film_check
from engine.film_check import FilmError
from engine.project import CONFIG_FILE, REPO, SHOTS_FILE, Project


def _cfg():
    return json.loads((FILM / CONFIG_FILE).read_text(encoding="utf-8"))


def _rows(text: str):
    import csv
    import io
    r = csv.DictReader(io.StringIO(text))
    return list(r), list(r.fieldnames or [])


def test_the_film_in_this_repository_has_no_problems():
    rows, cols = _rows((FILM / SHOTS_FILE).read_text(encoding="utf-8"))
    assert film_check.problems(_cfg(), rows, cols) == []


def test_missing_sections_and_bad_values_are_all_reported():
    cfg = _cfg()
    del cfg["engines"]
    cfg["budget"]["total_usd"] = "lots"
    cfg["keyframes"]["candidates"] = 0
    cfg["kaggle"]["stages"] = ["video", "paint"]
    found = film_check.config_problems(cfg)
    assert any('"engines" section is missing' in x for x in found)
    assert any("budget.total_usd" in x for x in found)
    assert any("keyframes.candidates" in x for x in found)
    assert any("kaggle.stages: paint is not a GPU stage" in x for x in found)


def test_bad_shot_rows_are_reported_with_their_line():
    cfg = _cfg()
    text = ("id,chapter,shot,seconds,type,model,takes,sound\n"
            "1-01,1,a,3,Hero,Wan 2.2 14B,4,\n"
            "1-01,1,b,3,Hero,Wan 2.2 14B,4,\n"             # id twice
            "1-03,0,c,3,Hero,Wan 2.2 14B,4,\n"             # chapter 0
            "1-04,1,d,0,Hero,No Such Model,x,\n")          # seconds 0, unknown model, takes not a number
    rows, cols = _rows(text)
    found = film_check.shot_problems(cfg, rows, cols)
    assert any("line 3 (1-01): the id 1-01 is used twice" in x for x in found)
    assert any("line 4 (1-03): chapter" in x for x in found)
    assert any("line 5 (1-04): seconds" in x for x in found)
    assert any("line 5 (1-04): takes" in x for x in found)
    assert any("'No Such Model' is not in config.json models" in x for x in found)


def test_a_gpu_model_without_its_engine_is_reported_once():
    cfg = _cfg()
    del cfg["engines"]["Wan 2.2 14B"]["fps"]
    rows, cols = _rows((FILM / SHOTS_FILE).read_text(encoding="utf-8"))
    found = [x for x in film_check.shot_problems(cfg, rows, cols) if "engines.Wan 2.2 14B" in x]
    assert len(found) == 1 and "needs fps" in found[0] and "more" in found[0]


def test_a_missing_column_stops_the_shot_check_with_one_clear_line():
    rows, cols = _rows("id,chapter,shot\n1-01,1,a\n")
    found = film_check.shot_problems(_cfg(), rows, cols)
    assert found == ["shots.csv: missing column(s) seconds, type, model, takes; it needs "
                     "id, chapter, shot, seconds, type, model, takes"]



def test_the_ai_finish_settings_are_checked_when_present():
    cfg = _cfg()
    del cfg["engines"]["ai_finish"]["fps"]
    cfg["engines"]["ai_finish"]["control"] = "normals"
    found = film_check.config_problems(cfg)
    assert any("engines.ai_finish: needs fps" in x for x in found)
    assert any("engines.ai_finish.control" in x for x in found)
    assert any("engines.ai_finish_720: needs fps" in x for x in found)          # a variant inherits its base's
    del cfg["engines"]["ai_finish"]
    assert any("needs engines.ai_finish, its base" in x for x in film_check.config_problems(cfg))
    del cfg["engines"]["ai_finish_720"]
    assert not any("ai_finish" in x for x in film_check.config_problems(cfg))   # a film may go without


def test_a_finish_variant_is_checked_as_the_worker_will_see_it():
    cfg = _cfg()
    assert not film_check.config_problems(cfg)
    v = cfg["engines"]["ai_finish_720"]
    v["crops"] = {"sayo": [0.5, 0.0, 0.6, 1.0]}                                  # runs off the picture
    v["transformer_gguf"] = {"prefer": ["Q4_K_M"]}                              # no repo
    v["offload"] = "disk"
    found = film_check.config_problems(cfg)
    assert any("ai_finish_720.crops.sayo" in x for x in found)
    assert any("ai_finish_720.transformer_gguf: needs repo" in x for x in found)
    assert any("ai_finish_720.offload" in x for x in found)
    assert not any("engines.ai_finish." in x or "engines.ai_finish:" in x for x in found)   # the base is fine

def test_a_character_table_must_be_keyed_by_chapter():
    cfg = _cfg()
    cfg["cgi"]["characters"]["sayo"]["height_m"] = {"one": 1.45}
    assert any("cgi.characters.sayo.height_m" in x for x in film_check.config_problems(cfg))


def test_loading_a_broken_film_lists_every_problem(tmp_path):
    copy_film(tmp_path)
    cfg = json.loads((tmp_path / CONFIG_FILE).read_text())
    del cfg["budget"]
    cfg["colab"]["stages"] = ["sing"]
    (tmp_path / CONFIG_FILE).write_text(json.dumps(cfg))
    with pytest.raises(FilmError) as e:
        Project(tmp_path)
    assert len(e.value.problems) == 2 and "2 problem(s)" in str(e.value)


def test_the_check_command(tmp_path):
    copy_film(tmp_path)
    ok = subprocess.run([sys.executable, "-m", "engine", "--film", str(tmp_path), "check"],
                        capture_output=True, text=True, cwd=REPO)
    assert ok.returncode == 0 and "no problems" in ok.stdout, ok.stderr
    (tmp_path / CONFIG_FILE).write_text("{ not json")
    bad = subprocess.run([sys.executable, "-m", "engine", "--film", str(tmp_path), "check"],
                         capture_output=True, text=True, cwd=REPO)
    assert bad.returncode == 2 and "not valid JSON" in bad.stderr
