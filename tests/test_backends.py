import httpx

from analyst.llm.anthropic_backend import to_anthropic_messages
from analyst.llm.base import ToolCall
from analyst.llm.openai_backend import OpenAICompatBackend, parse_tool_calls, to_openai_messages
from analyst.prompts import TOOLS

CONVERSATION = [
    {"role": "user", "content": "Question?"},
    {"role": "assistant", "text": "Let me look.", "tool_calls": [ToolCall("t1", "run_python", {"code": "print(1)"})]},
    {"role": "tool", "results": [{"id": "t1", "content": "1", "is_error": False}]},
]


def test_anthropic_conversion():
    out = to_anthropic_messages(CONVERSATION)
    assert out[1]["content"] == [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "t1", "name": "run_python", "input": {"code": "print(1)"}},
    ]
    assert out[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "1", "is_error": False}]}


def test_anthropic_conversion_omits_empty_text_block():
    msg = to_anthropic_messages([{"role": "assistant", "text": "", "tool_calls": [ToolCall("t", "n", {})]}])
    assert [b["type"] for b in msg[0]["content"]] == ["tool_use"]


def test_openai_conversion():
    out = to_openai_messages("SYS", CONVERSATION)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2]["tool_calls"][0]["function"] == {"name": "run_python", "arguments": '{"code": "print(1)"}'}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "1"}


def test_malformed_tool_arguments_are_surfaced_not_raised():
    calls = parse_tool_calls([{"id": "x", "function": {"name": "run_python", "arguments": "{not json"}}])
    assert calls[0].input == {"_parse_error": "{not json"}


def test_openai_backend_round_trip_with_a_mock_server():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": "abc", "type": "function",
                                "function": {"name": "final_answer",
                                             "arguments": '{"value": "7", "explanation": "x"}'}}]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    backend = OpenAICompatBackend("local-model", base_url="http://test/v1",
                                  client=httpx.Client(transport=httpx.MockTransport(handler)))
    response = backend.complete("SYS", CONVERSATION[:1], TOOLS)
    assert response.stop_reason == "tool_use"
    assert response.tool_calls[0].input == {"value": "7", "explanation": "x"}
    assert (response.input_tokens, response.output_tokens) == (11, 3)
    assert b'"local-model"' in seen["body"]
