"""The agent loop: the model plans, writes code, reads the result, and iterates until it
submits a final answer or runs out of steps.

Everything the agent does is recorded as Steps, so a run can be replayed, scored and shown.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from analyst.llm.base import LLMBackend, ToolCall
from analyst.prompts import SYSTEM_PROMPT, TOOLS
from analyst.sandbox import Sandbox, make_sandbox
from analyst.sandbox.base import truncate


@dataclass
class Step:
    kind: str  # "code" | "note" | "tool_error" | "nudge"
    text: str = ""
    code: str = ""
    stdout: str = ""
    error: str | None = None
    seconds: float = 0.0


@dataclass
class RunResult:
    status: str  # answered | max_steps | no_answer | refused
    value: str | None = None
    explanation: str = ""
    steps: list[Step] = field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def code_errors(self) -> int:
        return sum(1 for s in self.steps if s.kind == "code" and s.error)


class DataAnalystAgent:
    def __init__(
        self,
        backend: LLMBackend,
        sandbox: str | Callable[[Path], Sandbox] = "local",
        max_turns: int = 12,
        exec_timeout: float = 30.0,
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        self.backend = backend
        self._sandbox_factory = (
            (lambda data_dir: make_sandbox(sandbox, data_dir)) if isinstance(sandbox, str) else sandbox
        )
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.on_step = on_step or (lambda step: None)

    def _record(self, result: RunResult, step: Step) -> None:
        result.steps.append(step)
        self.on_step(step)

    def run(self, question: str, data_dir: Path) -> RunResult:
        result = RunResult(status="max_steps")
        started = time.monotonic()
        sandbox = self._sandbox_factory(Path(data_dir))
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Question: {question}\n\nThe data files are in ./data."}
        ]
        nudged = False
        try:
            for _ in range(self.max_turns):
                response = self.backend.complete(SYSTEM_PROMPT, messages, TOOLS)
                result.turns += 1
                result.input_tokens += response.input_tokens
                result.output_tokens += response.output_tokens
                messages.append(
                    {"role": "assistant", "text": response.text, "tool_calls": response.tool_calls}
                )
                if response.text.strip():
                    self._record(result, Step("note", text=response.text.strip()))

                if response.stop_reason == "refusal":
                    result.status = "refused"
                    break

                if not response.tool_calls:
                    if nudged:
                        result.status = "no_answer"
                        break
                    nudged = True
                    nudge = "Use the run_python tool to analyse the data, then submit with final_answer."
                    self._record(result, Step("nudge", text=nudge))
                    messages.append({"role": "user", "content": nudge})
                    continue

                tool_results = []
                for call in response.tool_calls:
                    if call.name == "final_answer":
                        value = str(call.input.get("value", "")).strip()
                        if not value:
                            tool_results.append(self._tool_error(result, call, "`value` must be non-empty."))
                            continue
                        result.status = "answered"
                        result.value = value
                        result.explanation = str(call.input.get("explanation", "")).strip()
                        return self._finish(result, started)
                    tool_results.append(self._run_tool(result, sandbox, call))
                messages.append({"role": "tool", "results": tool_results})
            return self._finish(result, started)
        finally:
            sandbox.close()

    def _run_tool(self, result: RunResult, sandbox: Sandbox, call: ToolCall) -> dict[str, Any]:
        if call.name != "run_python":
            return self._tool_error(result, call, f"Unknown tool {call.name!r}.")
        code = call.input.get("code")
        if not isinstance(code, str) or not code.strip():
            return self._tool_error(result, call, "`code` must be a non-empty string.")
        t0 = time.monotonic()
        out = sandbox.execute(code, timeout=self.exec_timeout)
        self._record(
            result,
            Step("code", code=code, stdout=out.stdout, error=out.error, seconds=time.monotonic() - t0),
        )
        content = out.stdout
        if out.error:
            content += f"\n[error]\n{out.error}"
        return {"id": call.id, "content": truncate(content) or "(no output)", "is_error": bool(out.error)}

    def _tool_error(self, result: RunResult, call: ToolCall, message: str) -> dict[str, Any]:
        self._record(result, Step("tool_error", text=message))
        return {"id": call.id, "content": message, "is_error": True}

    @staticmethod
    def _finish(result: RunResult, started: float) -> RunResult:
        result.seconds = time.monotonic() - started
        return result
