"""Run the benchmark against a model and report pass rate, cost-in-tokens, and failure modes.

  python -m evals.run_evals --backend anthropic --model claude-sonnet-5-5 --trials 3
  python -m evals.run_evals --backend openai --model <lm-studio-model> --label mistral-nemo
  python -m evals.run_evals --oracle          # validates the benchmark, needs no model
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from analyst import settings
from analyst.agent import DataAnalystAgent
from analyst.llm.factory import make_backend
from evals.oracle import oracle_backend
from evals.tasks import Task, generate

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def classify(task: Task, status: str, value: str | None) -> str:
    """Why a run failed: the distinction matters more than the pass rate alone."""
    if task.check(value):
        return "pass"
    if status != "answered":
        return status  # max_steps | no_answer | refused
    if task.kind == "unanswerable":
        return "hallucinated"  # answered a question the data cannot answer
    if value and "UNANSWERABLE" in value.upper():
        return "false_refusal"  # gave up on an answerable question
    return "wrong_answer"


def run_one(agent_factory, task: Task) -> dict:
    started = time.monotonic()
    try:
        result = agent_factory(task).run(task.question, task.data_dir)
    except Exception as exc:  # noqa: BLE001 - an API failure must not abort the whole eval
        return {"task": task.id, "outcome": "infra_error", "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.monotonic() - started, 2)}
    return {
        "task": task.id,
        "outcome": classify(task, result.status, result.value),
        "value": result.value,
        "expected": task.expected,
        "turns": result.turns,
        "code_errors": result.code_errors,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "seconds": round(result.seconds, 2),
        "explanation": result.explanation,
        "code": [s.code for s in result.steps if s.kind == "code"],
    }


def summarise(runs: list[dict]) -> dict:
    scored = [r for r in runs if r["outcome"] != "infra_error"]
    passed = [r for r in scored if r["outcome"] == "pass"]

    def mean(key: str) -> float:
        return round(statistics.mean(r[key] for r in scored), 2) if scored else 0.0

    return {
        "runs": len(runs),
        "infra_errors": len(runs) - len(scored),
        "pass_rate": round(len(passed) / len(scored), 3) if scored else None,
        "mean_turns": mean("turns"),
        "mean_code_errors": mean("code_errors"),
        "mean_input_tokens": mean("input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_seconds": mean("seconds"),
        "outcomes": dict(Counter(r["outcome"] for r in runs)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=settings.BACKEND, choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--sandbox", default="local", choices=["local", "docker"])
    parser.add_argument("--trials", type=int, default=1, help="runs per task (models are non-deterministic)")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--tasks", nargs="*", help="only run these task ids")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--label", default=None, help="name for the results file")
    parser.add_argument("--oracle", action="store_true", help="play reference solutions (no model needed)")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="analyst-bench-"))
    tasks = generate(workdir, seed=args.seed)
    if args.tasks:
        tasks = [t for t in tasks if t.id in args.tasks]

    if args.oracle:
        model = "oracle"
        factory = lambda task: DataAnalystAgent(oracle_backend(task), sandbox=args.sandbox)  # noqa: E731
    else:
        backend = make_backend(args.backend, args.model)
        model = backend.model
        factory = lambda task: DataAnalystAgent(  # noqa: E731
            backend, sandbox=args.sandbox, max_turns=args.max_turns)

    print(f"model={model} sandbox={args.sandbox} tasks={len(tasks)} trials={args.trials}\n")
    runs = []
    for task in tasks:
        for trial in range(args.trials):
            run = run_one(factory, task)
            run["trial"] = trial
            runs.append(run)
            print(f"  {task.id:<32} {run['outcome']:<14} turns={run.get('turns', '-'):<3} "
                  f"{run['seconds']}s" + (f"  {run['error']}" if "error" in run else ""))

    summary = summarise(runs)
    print(f"\npass rate: {summary['pass_rate']}   outcomes: {summary['outcomes']}")
    print(f"mean turns {summary['mean_turns']}, code errors {summary['mean_code_errors']}, "
          f"tokens in/out {summary['mean_input_tokens']}/{summary['mean_output_tokens']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    label = args.label or model.replace("/", "_")
    out = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{label}.json"
    out.write_text(json.dumps({
        "model": model, "backend": None if args.oracle else args.backend, "sandbox": args.sandbox,
        "seed": args.seed, "trials": args.trials, "max_turns": args.max_turns,
        "summary": summary, "runs": runs,
    }, indent=2, default=str), encoding="utf-8")
    print(f"saved {out}")
    return 0 if summary["infra_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
