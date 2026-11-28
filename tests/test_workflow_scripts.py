"""The workflows' helper scripts (.github/scripts), run against a stand-in for the gh command."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from engine.project import REPO

SCRIPTS = REPO / ".github" / "scripts"


def _stub_gh(tmp_path: Path, body: str) -> dict:
    """A `gh` on PATH that runs `body` (bash); returns the environment to run a script with."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text("#!/bin/bash\n" + body)
    gh.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "STUB": str(tmp_path),
            "GITHUB_REPOSITORY": "o/r", "GITHUB_RUN_ID": "300", "WAIT_POLL_SECONDS": "0"}


WAIT_GH = '''
n=$(cat "$STUB/polls" 2>/dev/null || echo 0)
if [ "$1 $2" = "run view" ]; then echo '{"createdAt":"2026-11-28T10:05:00Z"}'; exit 0; fi
echo $((n+1)) > "$STUB/polls"
s=$([ "$n" -lt 2 ] && echo in_progress || echo completed)
echo "[{\\"databaseId\\":200,\\"status\\":\\"$s\\",\\"createdAt\\":\\"2026-11-28T10:00:00Z\\"},
       {\\"databaseId\\":400,\\"status\\":\\"in_progress\\",\\"createdAt\\":\\"2026-11-28T10:10:00Z\\"},
       {\\"databaseId\\":300,\\"status\\":\\"in_progress\\",\\"createdAt\\":\\"2026-11-28T10:05:00Z\\"}]"
'''


def test_a_run_waits_only_for_the_other_workflows_earlier_runs(tmp_path):
    env = _stub_gh(tmp_path, WAIT_GH)
    r = subprocess.run([sys.executable, str(SCRIPTS / "wait_for_runs.py"), "produce.yml", "--minutes", "1"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("waiting for produce.yml run(s) 200\n") == 2      # 400 came later, 300 is this run
    assert "no earlier produce.yml run is active" in r.stdout


def test_a_run_goes_ahead_with_a_warning_after_its_limit(tmp_path):
    env = _stub_gh(tmp_path, WAIT_GH.replace('-lt 2', '-lt 999'))
    env["WAIT_POLL_SECONDS"] = "0.01"
    r = subprocess.run([sys.executable, str(SCRIPTS / "wait_for_runs.py"), "produce.yml", "--minutes", "0.002"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "::warning::1 earlier produce.yml run(s) still active" in r.stdout


def test_the_production_issue_is_named_after_the_film(tmp_path):
    env = _stub_gh(tmp_path, 'echo "$*" >> "$STUB/log"\n'
                             '[ "$1 $2" = "issue create" ] && echo "https://github.com/o/r/issues/7"\nexit 0\n')
    state = tmp_path / "state"
    state.mkdir()
    (state / "status.json").write_text(json.dumps({"notice": {"kind": "refs", "text": "refs ready"}}))
    env.update(FILM="films/moon-rabbit", OUTCOME="success")
    r = subprocess.run([sys.executable, str(SCRIPTS / "notify.py"), str(state)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "log").read_text()
    assert "--title Moon Rabbit production" in log and "/approve refs" in log
    assert (state / "issue.txt").read_text() == "7" and "comment 7 --body refs ready" in log

