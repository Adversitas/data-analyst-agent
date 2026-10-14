from __future__ import annotations

import uuid
from pathlib import Path

from analyst.sandbox.base import _ProcessSandbox

WORKER = Path(__file__).with_name("worker.py")
IMAGE = "analyst-sandbox:latest"  # built from docker/Dockerfile.sandbox


class DockerSandbox(_ProcessSandbox):
    def __init__(self, data_dir: Path) -> None:
        super().__init__()
        self._data_dir = Path(data_dir).resolve()
        self._name = f"analyst-{uuid.uuid4().hex[:8]}"

    def _command(self) -> list[str]:
        return [
            "docker", "run", "-i", "--rm", "--name", self._name,
            "--network", "none",
            "--read-only", "--tmpfs", "/tmp:size=64m",
            "--memory", "1g", "--cpus", "1", "--pids-limit", "128",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-v", f"{self._data_dir}:/work/data:ro",
            "-v", f"{WORKER.resolve()}:/app/worker.py:ro",
            "-w", "/work",
            IMAGE, "python", "-u", "/app/worker.py",
        ]

    def _kill(self) -> None:
        # Killing the `docker run` client does not stop the container, so do it by name.
        import subprocess

        subprocess.run(["docker", "kill", self._name], capture_output=True)
        super()._kill()
