"""Ask the agent a question about a folder of data files, watching it work.

  python -m analyst.cli ./my_data "Which region grew fastest year over year?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyst import settings
from analyst.agent import DataAnalystAgent, Step
from analyst.llm.factory import make_backend

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def show(step: Step) -> None:
    if step.kind == "note":
        print(f"\n[thinking] {step.text}")
    elif step.kind == "code":
        print("\n[run_python]")
        print("  " + step.code.strip().replace("\n", "\n  "))
        if step.stdout.strip():
            print("[output]\n  " + step.stdout.strip().replace("\n", "\n  "))
        if step.error:
            print("[error]\n  " + step.error.strip().replace("\n", "\n  "))
    else:
        print(f"\n[{step.kind}] {step.text}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, help="folder containing the data files")
    parser.add_argument("question")
    parser.add_argument("--backend", default=settings.BACKEND, choices=["anthropic", "openai"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--sandbox", default="local", choices=["local", "docker"])
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"error: {args.data_dir} is not a directory", file=sys.stderr)
        return 2

    agent = DataAnalystAgent(make_backend(args.backend, args.model), sandbox=args.sandbox,
                             max_turns=args.max_turns, on_step=show)
    result = agent.run(args.question, args.data_dir)

    print("\n" + "=" * 60)
    if result.status == "answered":
        print(f"ANSWER: {result.value}\n{result.explanation}")
    else:
        print(f"No answer ({result.status}) after {result.turns} turns.")
    print(f"[{result.turns} turns, {result.code_errors} code errors, "
          f"{result.input_tokens}/{result.output_tokens} tokens in/out, {result.seconds:.1f}s]")
    return 0 if result.status == "answered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
