"""Media storage in GitHub releases, so the pipeline can run on a fresh GitHub Actions machine.

Git holds the code and the small state (ledger, review images). Large media (keyframes,
takes, 4K files, chapter cuts) are release files, one release per chapter and kind:

    media-ch1   keyframes and takes of chapter 1
    4k-ch1      the 4K versions
    cuts        animatics and finished 4K chapters

A file is fetched only when a step needs it, and every new file is uploaded after a step.
Everything goes through the `gh` command, which GitHub Actions has signed in already.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


class StoreError(RuntimeError):
    pass


def tag_for(p, f: Path) -> str | None:
    """Which release a media file belongs in, from its place under media/."""
    try:
        parts = Path(f).resolve().relative_to(p.media).parts
    except ValueError:
        return None
    if len(parts) >= 2 and parts[0] == "animatic":
        return "cuts" if Path(f).suffix == ".mp4" and len(parts) == 2 else None
    if len(parts) == 3 and parts[0] in ("keyframes", "takes", "4k"):
        chapter = parts[1].split("-")[0]
        return f"{'4k' if parts[0] == '4k' else 'media'}-ch{chapter}"
    return None


class ReleaseStore:
    def __init__(self, repo: str | None = None, runner=None):
        self.repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
        if not self.repo:
            raise StoreError("set GITHUB_REPOSITORY (owner/name) to use the release store")
        self._run = runner or self._subprocess
        self._assets: dict[str, set[str]] = {}

    @staticmethod
    def _subprocess(args: list[str]) -> tuple[int, str]:
        r = subprocess.run(["gh", *args], capture_output=True, text=True)
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def assets(self, tag: str) -> set[str] | None:
        if tag not in self._assets:
            code, out = self._run(["release", "view", tag, "-R", self.repo, "--json", "assets",
                                   "--jq", ".assets[].name"])
            if code != 0:
                return None                     # no such release yet
            self._assets[tag] = {x.strip() for x in out.splitlines() if x.strip()}
        return self._assets[tag]

    def fetch(self, p, f: Path) -> bool:
        tag = tag_for(p, f)
        if not tag or f.name not in (self.assets(tag) or set()):
            return False
        f.parent.mkdir(parents=True, exist_ok=True)
        code, out = self._run(["release", "download", tag, "-R", self.repo, "-p", f.name,
                               "-D", str(f.parent), "--skip-existing"])
        if code != 0:
            raise StoreError(f"download of {f.name} from {tag} failed: {out.strip()[-300:]}")
        return f.exists()

    def _ensure_release(self, tag: str) -> None:
        if self.assets(tag) is not None:
            return
        code, out = self._run(["release", "create", tag, "-R", self.repo, "--prerelease",
                               "--title", tag, "--notes", "Media store for the Mochizuki pipeline. Do not edit."])
        if code != 0 and "already exists" not in out:
            raise StoreError(f"could not create release {tag}: {out.strip()[-300:]}")
        self._assets[tag] = set()

    def upload(self, p, f: Path, clobber: bool = False) -> bool:
        tag = tag_for(p, f)
        if not tag or not f.exists():
            return False
        self._ensure_release(tag)
        if f.name in self._assets[tag] and not clobber:
            return False
        args = ["release", "upload", tag, str(f), "-R", self.repo] + (["--clobber"] if clobber else [])
        code, out = self._run(args)
        if code != 0:
            raise StoreError(f"upload of {f.name} to {tag} failed: {out.strip()[-300:]}")
        self._assets[tag].add(f.name)
        return True

    def push(self, p) -> int:
        """Upload every media file made on this machine that the store does not have yet."""
        n = 0
        for folder in ("keyframes", "takes", "4k"):
            for f in sorted((p.media / folder).glob("*/*")):
                if f.is_file():
                    n += self.upload(p, f)
        return n

    def url(self, p, f: Path) -> str | None:
        tag = tag_for(p, f)
        return f"https://github.com/{self.repo}/releases/download/{tag}/{f.name}" if tag else None
