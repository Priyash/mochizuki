"""Wait while runs of another workflow, started before this one, are still going.

    python .github/scripts/wait_for_runs.py produce.yml --minutes 65

The producer and the owner's commands both change the state branch and the ledger, so they must not
overlap. They are in different concurrency groups, because a shared group keeps only one waiting run
and a newer producer run would cancel a waiting owner command. So each waits here for the other.
It waits only for runs created before itself, so two runs never wait for each other. After
`--minutes` it goes ahead with a warning (by then the other run has hit its own timeout).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ACTIVE = {"queued", "in_progress", "waiting", "requested", "pending"}
DEFAULT_MINUTES = 30.0


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args, "-R", os.environ["GITHUB_REPOSITORY"]], check=True,
                          capture_output=True, text=True).stdout


def earlier_active(workflow: str, me: str, my_id: str) -> list[dict]:
    """Runs of `workflow` that are not finished and were created before this run."""
    runs = json.loads(gh("run", "list", "--workflow", workflow, "--limit", "30",
                         "--json", "databaseId,status,createdAt"))
    return [r for r in runs if r["status"] in ACTIVE and r["createdAt"] < me and str(r["databaseId"]) != my_id]


def main(argv: list[str]) -> int:
    workflow = argv[0]
    minutes = float(argv[argv.index("--minutes") + 1]) if "--minutes" in argv else DEFAULT_MINUTES
    poll = float(os.environ.get("WAIT_POLL_SECONDS", "20"))
    my_id = os.environ["GITHUB_RUN_ID"]
    me = json.loads(gh("run", "view", my_id, "--json", "createdAt"))["createdAt"]
    deadline = time.time() + minutes * 60
    while True:
        busy = earlier_active(workflow, me, my_id)
        if not busy:
            print(f"no earlier {workflow} run is active")
            return 0
        if time.time() > deadline:
            print(f"::warning::{len(busy)} earlier {workflow} run(s) still active after {minutes:g} min; going ahead")
            return 0
        print(f"waiting for {workflow} run(s) {', '.join(str(r['databaseId']) for r in busy)}", flush=True)
        time.sleep(poll)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
