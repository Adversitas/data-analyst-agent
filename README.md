# Data Analyst Agent

An autonomous agent that answers questions about messy data files. You point it at a folder
of CSVs and ask a question in plain English; it **plans, writes and runs Python in a sandbox,
reads the results, fixes its own errors, and submits a final answer**, with no human in the
loop between question and answer.

```bash
python -m analyst.cli ./my_data "Which region had the highest revenue last quarter?"
```

The CLI streams every step (code written, output, errors) as it happens.

The point of the project is not the agent loop, which is ~120 lines. It is the two things
around it that make an agent trustworthy: **a sandbox that contains what the model runs**, and
**a benchmark with ground-truth answers that measures whether it is right**.

---

## How it works

```
 question + data folder
        │
        ▼
 ┌─ agent loop (analyst/agent.py) ────────────────────────────┐
 │  model ──tool call──► run_python(code) ──► Sandbox          │
 │    ▲                        │                               │
 │    └──── stdout / traceback ◄┘      (persistent session)     │
 │  ... repeat up to N turns ...                               │
 │  model ──► final_answer(value, explanation)                 │
 └─────────────────────────────────────────────────────────────┘
        │
        ▼
 answer + full step trace (code, outputs, errors, tokens, time)
```

- **Two tools only.** `run_python` and `final_answer`. All the analysis is code the model
  writes, so it can handle questions nobody anticipated.
- **A persistent Python session.** Variables survive between calls, so the agent can load
  data once and build on it, like a notebook.
- **Self-correction.** Tracebacks go back to the model as tool errors. Timeouts and crashes
  reset the session and tell the model its variables are gone.
- **Structured answers.** `final_answer(value, explanation)` forces a machine-checkable value,
  and `UNANSWERABLE` is a first-class answer so the agent can decline instead of inventing.
- **Provider-neutral.** The loop talks to a small `LLMBackend` interface: Claude
  (`anthropic_backend.py`), any OpenAI-compatible server such as LM Studio
  (`openai_backend.py`), or a scripted fake for tests.

## The sandbox

Model-written code is untrusted code. Two implementations of the same interface:

| | `local` | `docker` |
|---|---|---|
| Isolation | subprocess, temp dir, scrubbed environment, hard timeout | container: **no network**, read-only root, **non-root user**, 1 GB memory, 1 CPU, 128 pids, all capabilities dropped |
| API keys visible to model code | no (environment is allow-listed) | no |
| Original data modifiable | no (agent works on a copy) | no (mounted read-only) |
| Security boundary | **No.** Convenience for development | Yes, for the properties above |

Both are covered by tests (`tests/test_sandbox.py`, `tests/test_docker_sandbox.py`), including
the failure modes: infinite loops are killed, a dying process is detected and recovered, and
code that writes to stdout cannot corrupt the control protocol.

## The benchmark

`evals/tasks.py` generates three seeded datasets with deliberate real-world dirt and 14 questions
whose answers are known. **Ground truth is computed from the clean data before the corruption
is applied**, so a task is only passed by correctly undoing the dirt.

| Dataset | Dirt injected | Tasks |
|---|---|---|
| `orders` (400 orders) | 25 duplicate rows, amounts as `"€1,234.50"` / `"-€45.00"` strings, 15 spellings of 5 countries (`FR`, `France`, `france `, `UK`, ...), two date formats | revenue, top country, top-3 products, peak month, refund rate, average order value, **an unanswerable question** |
| `sensors` (1 440 readings) | `-999` sentinel for missing temperature, `"N/A"` in a numeric column | mean temperature, count above threshold, correlation, station with most gaps |
| `hr` (2 files, 150 staff) | needs a join, orphan department ids, missing salaries, two date formats | top location, median salary, hires in 2022 |

Each task has a **reference solution** (plain pandas on the dirty files). `tests/test_benchmark.py`
runs all 14 through the real agent loop and sandbox and requires 100%. That proves each
expected answer is reachable, so any failure in a real run is the model's, not the benchmark's.
The generator also rejects seeds that create ties ("top 3" with a tie at third place) so every
answer is unambiguous.

### Reading the results

`evals/run_evals.py` reports more than a pass rate. Failures are classified, because they are
not equally bad:

| Outcome | Meaning |
|---|---|
| `wrong_answer` | Answered confidently with the wrong value |
| `hallucinated` | **Answered a question the data cannot answer** |
| `false_refusal` | Said `UNANSWERABLE` when the answer was computable |
| `max_steps` / `no_answer` | Never reached a final answer |
| `infra_error` | API/network failure; excluded from the pass rate |

It also records turns, code errors, tokens in/out and wall time per run, and saves every
run's code and answer as JSON in `evals/results/`.

## Results

**No model results yet.** The benchmark and pipeline are validated (the oracle scores 14/14 in
both sandboxes), but running a real model needs an `ANTHROPIC_API_KEY` or a local server, and
none was configured when this was built. Numbers will be added to this section from the JSON
files in `evals/results/`, not written by hand.

Caveats to keep in mind when they arrive: 14 tasks is small, so one task is ~7 points; use
`--trials 3` or more because models are non-deterministic; and the data is synthetic.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate                      # Linux / macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                      # then set ANTHROPIC_API_KEY

pytest                                      # 52 tests, no API key needed
python -m evals.run_evals --oracle          # validates the benchmark pipeline, no model needed

python -m evals.run_evals --trials 3                                  # Claude
python -m evals.run_evals --backend openai --model <name> --label x   # LM Studio / any server

# hardened sandbox
docker build -t analyst-sandbox:latest -f docker/Dockerfile.sandbox docker
python -m evals.run_evals --sandbox docker --trials 3
```

## Layout

```
analyst/
  agent.py          the loop: plan, run code, read output, retry, answer
  prompts.py        system prompt and the two tool definitions
  llm/              backends: Claude, OpenAI-compatible, scripted (tests)
  sandbox/          worker.py + local and Docker executors
  cli.py            ask a question, watch the steps stream by
evals/
  tasks.py          seeded dirty datasets, questions, ground truth, scoring
  oracle.py         plays the reference solutions to validate the benchmark
  run_evals.py      runs a model over the benchmark, classifies failures
docker/             sandbox image
tests/              unit tests: loop, backends, sandbox, benchmark, scoring
```

## Limitations

- The local sandbox is for development. Use Docker for anything involving untrusted prompts or data.
- The agent only analyses; it has no tools that change anything outside its sandbox.
- No cost-per-answer in dollars (prices change); token counts are recorded instead.
- Single-agent, single-question: no memory across questions and no multi-agent planning.
