"""Long-lived Python worker: executes code snippets in a persistent namespace.

Protocol (one JSON object per line): the parent writes {"code": "..."} to stdin and
reads one {"stdout": "...", "error": "..." | null} line from stdout. User code gets its
own stdout/stderr, so it can never corrupt the protocol channel.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback


def main() -> None:
    channel = sys.stdout
    namespace: dict = {"__name__": "__sandbox__"}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        code = json.loads(line)["code"]
        buffer = io.StringIO()
        error = None
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                exec(compile(code, "<agent>", "exec"), namespace)
            except BaseException:  # noqa: BLE001 - the agent must see every failure
                error = traceback.format_exc(limit=-6)
        channel.write(json.dumps({"stdout": buffer.getvalue(), "error": error}) + "\n")
        channel.flush()


if __name__ == "__main__":
    main()
