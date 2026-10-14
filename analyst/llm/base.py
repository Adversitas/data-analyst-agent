"""Provider-neutral message types.

The agent keeps its conversation in this neutral form; each backend translates it to its
provider's wire format. That keeps the agent loop identical across Claude, local models
and the scripted fake used in tests.

Neutral messages:
  {"role": "user", "content": str}
  {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...]}
  {"role": "tool", "results": [{"id": str, "content": str, "is_error": bool}, ...]}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens | refusal
    input_tokens: int = 0
    output_tokens: int = 0


class LLMBackend(Protocol):
    model: str

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...
