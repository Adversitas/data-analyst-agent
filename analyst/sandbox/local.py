from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from analyst.sandbox.base import _ProcessSandbox

WORKER = Path(__file__).with_name("worker.py")
# Only what Python itself needs to start. Notably absent: ANTHROPIC_API_KEY & friends.
_PASSTHROUGH = ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG")


class LocalSandbox(_ProcessSandbox):
    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        # The agent works on a copy, so it cannot modify the original files.
        self._workdir = Path(tempfile.mkdtemp(prefix="analyst-"))
        shutil.copytree(data_dir, self._workdir / "data")

    def _command(self) -> list[str]:
        return [sys.executable, "-u", str(WORKER)]

    def _env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def _cwd(self) -> str:
        return str(self._workdir)

    def close(self) -> None:
        super().close()
        shutil.rmtree(self._workdir, ignore_errors=True)
