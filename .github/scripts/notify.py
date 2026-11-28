"""Post the producer's notice to the production issue, only when it has changed."""
import json
import os
import subprocess
import sys
from pathlib import Path

state = Path(sys.argv[1])
repo = os.environ["GITHUB_REPOSITORY"]
owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "")
status = state / "status.json"
if os.environ.get("OUTCOME") not in (None, "", "success") or not status.exists():
    kind, text = "failed", f"@{owner} The producer run failed: {os.environ.get('RUN_URL', '')}"
else:
    n = json.loads(status.read_text(encoding="utf-8"))["notice"]
    kind, text = n["kind"], n["text"]
last_f = state / "last_notice.json"
last = json.loads(last_f.read_text(encoding="utf-8")) if last_f.exists() else {}
# Progress notices (running, review, idle) are posted when the kind changes; anything that
# needs the owner is posted whenever its text changes.
if text == last.get("text") or (kind in ("running", "review", "idle") and kind == last.get("kind")):
    sys.exit(0)


def gh(*args):
    return subprocess.run(["gh", *args, "-R", repo], check=True, capture_output=True, text=True).stdout.strip()


issue = state / "issue.txt"
if not issue.exists():
    film = Path(os.environ.get("FILM") or "film").name.replace("-", " ").title()   # set by the workflow
    url = gh("issue", "create", "--title", f"{film} production",
             "--body", "The producer posts here when something is needed. Commands (owner only): "
                       "`/approve refs`, `/reject <name>`, `/approve board N`, `/approve animatic N`, "
                       "`/chapter N`, `/status`.")
    issue.write_text(url.rstrip("/").rsplit("/", 1)[-1])
gh("issue", "comment", issue.read_text().strip(), "--body", text)
last_f.write_text(json.dumps({"kind": kind, "text": text}), encoding="utf-8")
