"""Backend for any OpenAI-compatible chat-completions server (LM Studio, llama.cpp, vLLM)."""
from __future__ import annotations

import json
from typing import Any

import httpx

from analyst import settings
from analyst.llm.base import LLMResponse, ToolCall


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m["text"] or None}
            if m["tool_calls"]:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.input)},
                    }
                    for c in m["tool_calls"]
                ]
            out.append(msg)
        else:
            for r in m["results"]:
                out.append({"role": "tool", "tool_call_id": r["id"], "content": r["content"]})
    return out


def parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    calls = []
    for i, raw in enumerate(raw_calls):
        fn = raw["function"]
        try:
            args = json.loads(fn["arguments"] or "{}")
        except json.JSONDecodeError:
            # Small local models emit malformed JSON; surface it so the agent can report it back.
            args = {"_parse_error": fn["arguments"]}
        calls.append(ToolCall(raw.get("id") or f"call_{i}", fn["name"], args))
    return calls


class OpenAICompatBackend:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, client: httpx.Client | None = None) -> None:
        self.model = model or settings.MODEL
        self.base_url = (base_url or settings.OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.client = client or httpx.Client(timeout=300)

    def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": to_openai_messages(system, messages),
                "tools": [{"type": "function", "function": t} for t in tools],
                "temperature": 0,
                "max_tokens": settings.MAX_TOKENS,
            },
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        calls = parse_tool_calls(message.get("tool_calls") or [])
        finish = choice.get("finish_reason")
        stop = "tool_use" if calls else {"length": "max_tokens"}.get(finish, "end_turn")
        usage = body.get("usage", {})
        return LLMResponse(
            text=message.get("content") or "",
            tool_calls=calls,
            stop_reason=stop,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
