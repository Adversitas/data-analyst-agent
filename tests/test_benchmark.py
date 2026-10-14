"""Validates the benchmark itself: every task is solvable, answers are unambiguous, and
the scoring rejects wrong answers."""
import pytest

from analyst.agent import DataAnalystAgent
from evals.oracle import oracle_backend
from evals.tasks import generate, parse_number


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("bench"))


def test_generation_is_deterministic(tmp_path):
    a = [(t.id, t.expected) for t in generate(tmp_path / "a", seed=5)]
    b = [(t.id, t.expected) for t in generate(tmp_path / "b", seed=5)]
    assert a == b


def test_task_ids_are_unique(tasks):
    assert len({t.id for t in tasks}) == len(tasks)


def test_oracle_solves_every_task_through_the_real_sandbox(tasks):
    failures = []
    for task in tasks:
        result = DataAnalystAgent(oracle_backend(task)).run(task.question, task.data_dir)
        if result.status != "answered" or not task.check(result.value):
            failures.append((task.id, task.expected, result.value))
    assert not failures, failures


def test_naive_answers_fail_the_traps(tasks):
    """Computing revenue without deduplicating or cleaning currency must not pass."""
    revenue = next(t for t in tasks if t.id == "orders_revenue")
    import pandas as pd

    raw = pd.read_csv(revenue.data_dir / "orders.csv")
    with_duplicates = raw["amount"].str.replace("€", "").str.replace(",", "").astype(float).sum()
    assert not revenue.check(f"{with_duplicates:.2f}")


@pytest.mark.parametrize("text, expected", [
    ("1,234.50", 1234.5), ("€1,234.50", 1234.5), ("12.5%", 12.5), ("-3", -3.0), ("abc", None), ("nan", None),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_checkers(tasks):
    by_id = {t.id: t for t in tasks}
    assert by_id["orders_unanswerable"].check("UNANSWERABLE")
    assert not by_id["orders_unanswerable"].check("4.2")
    assert not by_id["orders_revenue"].check(None)
    top3 = by_id["orders_top3_products"]
    assert top3.check(", ".join(top3.expected))
    assert not top3.check(", ".join(reversed(top3.expected)))  # order matters
    label = by_id["orders_top_country"]
    assert label.check(f" '{label.expected.lower()}' ")  # case/quote/space insensitive
