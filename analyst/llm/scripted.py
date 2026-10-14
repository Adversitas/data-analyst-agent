"""A fake backend that replays a fixed script. Used to test the agent loop and to
validate the benchmark (an "oracle" agent that runs the reference solution)."""
from __future__ import annotations

from typing import Any, Callable

from analyst.llm.base import LLMResponse

Step = LLMResponse | Callable[[list[dict[str, Any]]], LLMResponse]


class ScriptedBackend:
    def __init__(self, steps: list[Step], model: str = "scripted") -> None:
        self.model = model
        self._steps = list(steps)
        self.calls: list[list[dict[str, Any]]] = []  # messages seen on each call

    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._steps:
            raise AssertionError("ScriptedBackend ran out of steps")
        step = self._steps.pop(0)
        return step(messages) if callable(step) else step
