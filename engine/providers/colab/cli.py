"""The `colab` command (google-colab-cli), signed in with Application Default Credentials, and
redacting what it prints before it reaches a log."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time

from .config import SESSION_PREFIX


class ColabCLI:
    """The `colab` command (google-colab-cli), signed in with Application Default Credentials."""

    def __init__(self, exe: str = "colab", auth: str | None = None):
        self.exe = exe
        self.auth = auth if auth is not None else os.environ.get("COLAB_AUTH", "adc")

    def _cmd(self, args: list[str]) -> list[str]:
        return [self.exe, *(["--auth", self.auth] if self.auth else []), *args]

    def run(self, args: list[str], timeout: float | None = 600) -> tuple[int, str]:
        try:
            r = subprocess.run(self._cmd(args), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return 124, f"timed out after {timeout} s: {(e.stdout or '')[-500:]}"
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def stream(self, args: list[str], deadline: float, on_line=None) -> tuple[int, str]:
        """Run a long command, echoing its output; kill it at `deadline` (unix time)."""
        p = subprocess.Popen(self._cmd(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail: list[str] = []
        killer = threading.Timer(max(1.0, deadline - time.time()), p.kill)
        killer.start()
        try:
            for line in p.stdout:
                line = line.rstrip()
                print(line, flush=True)
                tail = (tail + [line])[-60:]
                if on_line:
                    on_line(line)
            code = p.wait()
        finally:
            killer.cancel()
        return code, "\n".join(tail)

    def balance(self) -> float | None:
        code, out = self.run(["usage"], timeout=120)
        m = re.search(r"Current balance:\s*([\d.]+)", out)
        return float(m.group(1)) if code == 0 and m else None

    def sessions(self) -> list[str]:
        code, out = self.run(["sessions"], timeout=120)
        return re.findall(rf"\b({re.escape(SESSION_PREFIX)}[\w-]+)", out) if code == 0 else []

    _PY = ("import json,sys\n"
           "from colab_cli.auth import AuthProvider, get_credentials\n"
           "from colab_cli.client import Client, Prod\n"
           "c = Client(Prod(), get_credentials(None, provider=AuthProvider.ADC))\n"
           "if sys.argv[1] == 'list':\n"
           "    print(json.dumps([{'endpoint': a.endpoint, 'accelerator': str(a.accelerator.value)}"
           " for a in c.list_assignments()]))\n"
           "else:\n"
           "    c.unassign(sys.argv[2]); print('unassigned')\n")

    def assignments(self) -> list[dict] | None:
        """Every VM assigned to the account on Colab's side (local CLI state does not survive a runner)."""
        try:
            r = subprocess.run([sys.executable, "-c", self._PY, "list"], capture_output=True, text=True, timeout=120)
            return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    def unassign(self, endpoint: str) -> bool:
        try:
            r = subprocess.run([sys.executable, "-c", self._PY, "unassign", endpoint],
                               capture_output=True, text=True, timeout=120)
            return r.returncode == 0
        except Exception:  # noqa: BLE001
            return False


def _redact(text: str) -> str:
    """Action logs are public: never print the Google account's address."""
    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<account>", text or "")
