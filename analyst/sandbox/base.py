"""Sandbox: a persistent Python session the agent drives one snippet at a time.

Two implementations share the same worker script:
- LocalSandbox:  a subprocess with a scrubbed environment (API keys never reach it),
                 a temp working directory and a hard timeout. NOT a security boundary.
- DockerSandbox: the same worker inside a container with no network, a read-only
                 filesystem, and memory/CPU limits. Use this for untrusted models/data.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MAX_OUTPUT_CHARS = 4000


@dataclass
class ExecResult:
    stdout: str
    error: str | None
    timed_out: bool = False
    restarted: bool = False  # True when state was lost and the session was reset


class Sandbox(Protocol):
    def execute(self, code: str, timeout: float = 30.0) -> ExecResult: ...
    def close(self) -> None: ...


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


class _ProcessSandbox:
    """Shared plumbing: a worker process spoken to over line-delimited JSON."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()

    def _command(self) -> list[str]:
        raise NotImplementedError

    def _env(self) -> dict[str, str] | None:
        return None

    def _cwd(self) -> str | None:
        return None

    def _start(self) -> None:
        self._lines = queue.Queue()
        self._proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=self._env(),
            cwd=self._cwd(),
        )
        threading.Thread(target=self._pump, args=(self._proc, self._lines), daemon=True).start()

    @staticmethod
    def _pump(proc: subprocess.Popen[str], lines: queue.Queue[str | None]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)  # EOF: the worker exited

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
        if self._proc:
            self._proc.wait()
        self._proc = None

    def execute(self, code: str, timeout: float = 30.0) -> ExecResult:
        restarted = False
        if self._proc is None or self._proc.poll() is not None:
            self._start()
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps({"code": code}) + "\n")
            self._proc.stdin.flush()
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self._kill()
            return ExecResult(
                "", f"Timed out after {timeout:.0f}s. The Python session was reset: "
                "all variables are gone, reload your data.", timed_out=True, restarted=True)
        except (BrokenPipeError, OSError):
            line = None
        if line is None:
            self._kill()
            return ExecResult(
                "", "The Python process died (out of memory or exit call). "
                "The session was reset: all variables are gone.", restarted=True)
        payload = json.loads(line)
        return ExecResult(truncate(payload["stdout"]), payload["error"] and truncate(payload["error"]),
                          restarted=restarted)

    def close(self) -> None:
        self._kill()


def make_sandbox(kind: str, data_dir: Path) -> Sandbox:
    from analyst.sandbox.docker import DockerSandbox
    from analyst.sandbox.local import LocalSandbox

    if kind == "local":
        return LocalSandbox(data_dir)
    if kind == "docker":
        return DockerSandbox(data_dir)
    raise ValueError(f"Unknown sandbox {kind!r} (expected 'local' or 'docker')")
