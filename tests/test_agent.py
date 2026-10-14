import pytest

from analyst.agent import DataAnalystAgent
from analyst.llm.base import LLMResponse, ToolCall
from analyst.llm.scripted import ScriptedBackend


def code(src: str, id_: str = "c") -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id_, "run_python", {"code": src})], stop_reason="tool_use",
                       input_tokens=10, output_tokens=5)


def answer(value: str = "42", id_: str = "a") -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id_, "final_answer", {"value": value, "explanation": "because"})],
                       stop_reason="tool_use", input_tokens=10, output_tokens=5)


@pytest.fixture()
def data_dir(tmp_path):
    (tmp_path / "x.csv").write_text("a\n1\n2\n")
    return tmp_path


def run(backend, data_dir, **kwargs):
    return DataAnalystAgent(backend, **kwargs).run("question?", data_dir)


def test_happy_path(data_dir):
    result = run(ScriptedBackend([code("print(1 + 1)"), answer("2")]), data_dir)
    assert result.status == "answered" and result.value == "2" and result.explanation == "because"
    assert result.turns == 2 and result.input_tokens == 20 and result.output_tokens == 10


def test_stdout_is_fed_back_to_the_model(data_dir):
    backend = ScriptedBackend([code("print('hello')"), answer()])
    run(backend, data_dir)
    tool_message = backend.calls[1][-1]
    assert tool_message["role"] == "tool" and "hello" in tool_message["results"][0]["content"]


def test_agent_recovers_from_a_code_error(data_dir):
    backend = ScriptedBackend([code("1/0"), code("print('fixed')"), answer()])
    result = run(backend, data_dir)
    assert result.status == "answered" and result.code_errors == 1
    first_result = backend.calls[1][-1]["results"][0]
    assert first_result["is_error"] and "ZeroDivisionError" in first_result["content"]


def test_variables_persist_across_tool_calls(data_dir):
    backend = ScriptedBackend([code("n = 7"), code("print(n * 6)"), answer()])
    run(backend, data_dir)
    assert "42" in backend.calls[2][-1]["results"][0]["content"]


def test_stops_at_max_turns(data_dir):
    backend = ScriptedBackend([code("pass", str(i)) for i in range(5)])
    result = run(backend, data_dir, max_turns=3)
    assert result.status == "max_steps" and result.turns == 3 and result.value is None


def test_text_only_reply_is_nudged_once_then_given_up(data_dir):
    chatter = LLMResponse(text="I think it is 42.", stop_reason="end_turn")
    backend = ScriptedBackend([chatter, answer()])
    assert run(backend, data_dir).status == "answered"

    backend = ScriptedBackend([chatter, chatter])
    assert run(backend, data_dir).status == "no_answer"


def test_empty_final_answer_is_rejected_and_the_agent_continues(data_dir):
    backend = ScriptedBackend([answer("  "), answer("5")])
    result = run(backend, data_dir)
    assert result.status == "answered" and result.value == "5"
    assert any(s.kind == "tool_error" for s in result.steps)


def test_unknown_tool_and_malformed_input_are_reported_not_fatal(data_dir):
    bad_tool = LLMResponse(tool_calls=[ToolCall("1", "rm_rf", {})], stop_reason="tool_use")
    bad_input = LLMResponse(tool_calls=[ToolCall("2", "run_python", {"_parse_error": "{oops"})],
                            stop_reason="tool_use")
    backend = ScriptedBackend([bad_tool, bad_input, answer()])
    result = run(backend, data_dir)
    assert result.status == "answered"
    assert [s.kind for s in result.steps].count("tool_error") == 2


def test_refusal_ends_the_run(data_dir):
    result = run(ScriptedBackend([LLMResponse(stop_reason="refusal")]), data_dir)
    assert result.status == "refused"


def test_multiple_tool_calls_in_one_turn_all_get_results(data_dir):
    both = LLMResponse(
        tool_calls=[ToolCall("1", "run_python", {"code": "print('a')"}),
                    ToolCall("2", "run_python", {"code": "print('b')"})],
        stop_reason="tool_use")
    backend = ScriptedBackend([both, answer()])
    run(backend, data_dir)
    results = backend.calls[1][-1]["results"]
    assert [r["id"] for r in results] == ["1", "2"]


def test_sandbox_is_closed_even_when_the_backend_raises(data_dir):
    closed = []

    class Spy:
        def execute(self, code, timeout=30.0): raise AssertionError("unused")
        def close(self): closed.append(True)

    class Boom:
        model = "boom"
        def complete(self, *args): raise RuntimeError("api down")

    with pytest.raises(RuntimeError):
        DataAnalystAgent(Boom(), sandbox=lambda d: Spy()).run("q", data_dir)
    assert closed == [True]


def test_on_step_callback_streams_steps(data_dir):
    seen = []
    DataAnalystAgent(ScriptedBackend([code("print(1)"), answer()]), on_step=seen.append).run("q", data_dir)
    assert [s.kind for s in seen] == ["code"]
