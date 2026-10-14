"""An 'oracle' backend that plays each task's reference solution through the real agent loop
and sandbox. If it does not score 100%, the benchmark (not the model) is broken."""
from __future__ import annotations

import re
from typing import Any

from analyst.llm.base import LLMResponse, ToolCall
from analyst.llm.scripted import ScriptedBackend
from evals.tasks import Task


def oracle_backend(task: Task) -> ScriptedBackend:
    def submit(messages: list[dict[str, Any]]) -> LLMResponse:
        output = messages[-1]["results"][0]["content"]
        match = re.search(r"RESULT=(.*)", output)
        value = match.group(1).strip() if match else f"NO RESULT IN: {output[:200]}"
        return LLMResponse(
            tool_calls=[ToolCall("2", "final_answer", {"value": value, "explanation": "reference solution"})],
            stop_reason="tool_use",
        )

    run_reference = LLMResponse(
        tool_calls=[ToolCall("1", "run_python", {"code": task.reference_code})], stop_reason="tool_use"
    )
    return ScriptedBackend([run_reference, submit], model="oracle")
