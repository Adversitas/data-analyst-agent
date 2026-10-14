from __future__ import annotations

from typing import Any

import anthropic

from analyst import settings
from analyst.llm.base import LLMResponse, ToolCall


def to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks: list[dict[str, Any]] = []
            if m["text"]:
                blocks.append({"type": "text", "text": m["text"]})
            for call in m["tool_calls"]:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                )
            out.append({"role": "assistant", "content": blocks})
        else:  # tool results travel in a single user message
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r["id"],
                            "content": r["content"],
                            "is_error": r["is_error"],
                        }
                        for r in m["results"]
                    ],
                }
            )
    return out


class AnthropicBackend:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or settings.MODEL
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=settings.MAX_TOKENS,
            system=system,
            tools=[
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ],
            messages=to_anthropic_messages(messages),
        )
        return LLMResponse(
            text="".join(b.text for b in response.content if b.type == "text"),
            tool_calls=[
                ToolCall(b.id, b.name, dict(b.input)) for b in response.content if b.type == "tool_use"
            ],
            stop_reason=response.stop_reason or "end_turn",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
