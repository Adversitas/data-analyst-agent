import pytest

from evals.run_evals import classify, summarise
from evals.tasks import Task


def task(kind="number", expected=10.0):
    return Task("t", "d", "q", expected, kind, "pass")


@pytest.mark.parametrize("kind, expected, status, value, outcome", [
    ("number", 10.0, "answered", "10", "pass"),
    ("number", 10.0, "answered", "11", "wrong_answer"),
    ("number", 10.0, "answered", "UNANSWERABLE", "false_refusal"),
    ("unanswerable", "UNANSWERABLE", "answered", "4.2", "hallucinated"),
    ("unanswerable", "UNANSWERABLE", "answered", "UNANSWERABLE", "pass"),
    ("number", 10.0, "max_steps", None, "max_steps"),
    ("number", 10.0, "no_answer", None, "no_answer"),
])
def test_classify(kind, expected, status, value, outcome):
    assert classify(task(kind, expected), status, value) == outcome


def test_summarise_excludes_infra_errors_from_the_pass_rate():
    runs = [
        {"outcome": "pass", "turns": 2, "code_errors": 0, "input_tokens": 100, "output_tokens": 10, "seconds": 1},
        {"outcome": "wrong_answer", "turns": 4, "code_errors": 2, "input_tokens": 300, "output_tokens": 30, "seconds": 3},
        {"outcome": "infra_error", "error": "429", "seconds": 0},
    ]
    summary = summarise(runs)
    assert summary["pass_rate"] == 0.5 and summary["infra_errors"] == 1
    assert summary["mean_turns"] == 3 and summary["mean_input_tokens"] == 200
