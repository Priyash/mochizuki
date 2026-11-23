#!/usr/bin/env bash
# Commit the ledger and review files to the state branch, retrying if Claude pushed meanwhile.
set -euo pipefail
[ -f ledger.sqlite ] && cp ledger.sqlite state/ledger.sqlite
cd state
git config user.name "mochizuki-producer"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
git diff --cached --quiet && exit 0
git commit -qm "$1"
for i in 1 2 3 4 5; do
  if git pull -q --rebase origin state && git push -q origin HEAD:state; then exit 0; fi
  sleep $((i * 5))
done
echo "could not push the state branch" >&2
exit 1
