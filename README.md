# agents_profiler — B-POP engine

A Python implementation of **B-POP (Bi-level Parametric Optimization)**,
following the technical proposal in `B-POP_Python_Solution_Proposal_EN.docx`
and its validation experiment in
`BPOP_Technical_Solution_Proposal_with_Validation_Experiment.docx`, plus a
data-ingestion layer that builds B-POP problems from AI-agent-workflow
trace rows (the normalized trace schema from
`AI_Agent_Profiler_Product_Technical_Design_Review.docx`).

## Scope — what this is, and isn't

This repo is the **B-POP optimizer, a trace-ingestion path, and a small
Observe → Profile → Mine → Score → Recommend test environment** built on top
of them — not a full implementation of the "AI Agent Profiler" product
described in `AI_Agent_Profiler_Product_Technical_Design_Review.docx`. That
design covers a much larger system (OTLP trace collection, flow
reconstruction, replay/shadow execution, privacy controls) that does not
exist here. The only thing borrowed from that doc is its §7 trace schema,
used as the input shape for `src/ingest/` and `src/profiler/`. Concretely,
this repo currently has:

- a working leader/follower B-POP solver (`src/bpop/`), validated against
  one hand-specified problem (the doc3 capacity-allocation experiment);
- a small ingestion function that maps JSONL trace rows into *that same*
  capacity-allocation problem shape, using one assumption (`value_per_used_call`,
  defaulting to 2x observed cost per call) that is not derived from traces;
- an Observe/Mine/Score/Recommend pipeline (`src/profiler/`) that takes a
  trace file for one workflow and produces per-node determinism
  classification, cost-weighted scoring, and rule-based recommendations;
- a synthetic trace-data generator (`src/rawdata_gen/`, vendored from
  [`agents_profile_rawdata_gen`](https://github.com/ipintotech-lab/agents_profile_rawdata_gen))
  so the pipeline can be exercised without a real trace file;
- a dashboard UI (`src/api/static/index.html`) at `/` that runs an uploaded
  trace file, or data generated in-browser, through that pipeline and
  renders the report;
- a thin FastAPI wrapper (`src/api/`) exposing the raw solver (`/solve`),
  the pipeline (`/analyze`), and the generator (`/generate`) over HTTP.

It has no trace collector, no replay/shadow execution, and no privacy
controls — those remain out of scope per the design doc above.

## Architecture

- `src/bpop/problem.py` — typed bi-level problem model (`BilevelProblem`,
  `UpperLevelProblem`, `LowerLevelProblem`, `Variable`, `LinearConstraint`).
- `src/bpop/parametric.py` — converts the lower level into a PPOPT-compatible
  mpLP, with the leader's variables treated as the parameter.
- `src/bpop/ppopt_adapter.py` — runs PPOPT and normalizes its output into
  `CriticalRegion` objects (`y*(x) = A@x + b`, valid while `E@x <= f`).
- `src/bpop/upper_solver.py` — substitutes `y*(x)` into the upper level for
  one region and solves the resulting LP with PuLP.
- `src/bpop/solver.py` — `BPOP.solve(problem)`: orchestrates the above across
  all regions and picks the best feasible global candidate (boundary-tolerant).
- `src/bpop/result.py` — explainable `BPOPResult` (winning region,
  substitution, objective, and every region's evidence).
- `src/ingest/` — loads normalized agent-workflow trace rows (JSON Lines)
  and maps them into a capacity-allocation `BilevelProblem` (leader =
  capacity allocated per run, follower = capacity actually used, estimated
  from observed LLM-call counts and cost).
- `src/profiler/` — Observe (group trace rows into node outcomes) → Mine
  (classify each node deterministic/non-deterministic by whether the same
  `input_state` always yields the same `(decision.selected, result)`) →
  Score (cost-weighted scoring per node) → Recommend (rule-based
  suggestions), orchestrated by `run_pipeline()`. `fingerprint.py` adds
  action fingerprints per node -- each action grouped by its starting
  point (`trigger`) and logical identity at three levels (business: node +
  trigger + input; context: + the triggering step's result; execution: +
  model and agent version), with output distributions, dominant-output
  rates and a variance explanation. It's evidence alongside the
  classifier, not a change to it. `patterns.py` mines repeated step
  sequences and repeated identical calls, and charges each step for the
  LLM call that chose it (`decision_costs`). `opportunities.py` builds the
  report's `summary`: totals plus ranked savings opportunities
  (`EXTRACT_DETERMINISTIC_SUBSTEP`, `CACHE_RESULT`,
  `CONDITIONAL_LLM_BYPASS`, `REVIEW_HIDDEN_STATE`), each with its evidence
  and the observed cost it would remove. The dashboard leads with that
  summary, then repetition, then nodes (costliest first); capacity planning
  and the per-node score list are folded under "advanced".
- `src/rawdata_gen/` — synthetic trace-data generator vendored from
  [`agents_profile_rawdata_gen`](https://github.com/ipintotech-lab/agents_profile_rawdata_gen)
  (copied source, not a dependency, since that repo has no packaging
  metadata); `generate_configured(agent_count, actions_per_agent,
  total_lines)` builds mock agent runs for the dashboard's "Generate sample
  data" tab.
- `src/api/` — a thin FastAPI wrapper exposing the raw solver (`/solve`,
  `/examples/capacity-allocation`), the Observe→Recommend pipeline
  (`/analyze`), the sample-data generator (`/generate`), and `/health`, plus
  the dashboard UI at `/` (`src/api/static/index.html`).

## Usage

```python
from bpop import BPOP, BilevelProblem, UpperLevelProblem, LowerLevelProblem, Variable, LinearConstraint

upper = UpperLevelProblem(
    variables=[Variable(name="x", lower=0, upper=10)],
    objective_x_coeffs={"x": 2}, objective_y_coeffs={"y": -4}, sense="minimize",
)
lower = LowerLevelProblem(
    variables=[Variable(name="y", lower=0, upper=6)],
    objective_coeffs={"y": 1}, sense="maximize",
    constraints=[LinearConstraint(y_coeffs={"y": 1}, sense="<=", rhs_constant=0, x_coeffs={"x": 1})],
)
problem = BilevelProblem(upper=upper, lower=lower)
result = BPOP.solve(problem)
print(result.explanation)
```

Or, from agent-workflow trace rows:

```python
from ingest import build_capacity_problem_from_traces
from ingest.loader import load_traces
from bpop import BPOP

events = load_traces("data/sample_traces.jsonl")
problem = build_capacity_problem_from_traces(events, workflow="cve-remediation-agent")
result = BPOP.solve(problem)
print(result.explanation)
```

Or run the Observe → Profile → Mine → Score → Recommend pipeline directly:

```python
from ingest.loader import load_traces
from profiler import run_pipeline

events = load_traces("data/sample_traces.jsonl")
report = run_pipeline(events, workflow="cve-remediation-agent")
```

The same pipeline is reachable without writing any Python via the dashboard
at `/` (upload a trace file, or use the "Generate sample data" tab to build
one in-browser via `/generate`), or directly over HTTP:

```bash
curl -F file=@data/sample_traces.jsonl localhost:8000/analyze
curl -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"agent_count": 2, "actions_per_agent": [4, 6], "total_lines": 25}'
```

`/analyze` also accepts OpenTelemetry OTLP JSON and Claude Code session
transcripts (`~/.claude/projects/<project>/*.jsonl`), and `-F file=@...`
can be repeated to analyze several files together. To avoid uploading raw
session data (prompts, tool outputs, commands), convert locally first --
`--hash-inputs` replaces every `input_state` value with a hash, which
leaves determinism results unchanged:

```bash
python -m ingest.convert ~/.claude/projects/<project>/*.jsonl --hash-inputs -o traces.jsonl
curl -F file=@traces.jsonl localhost:8000/analyze
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Note: `pulp>=3` ships a Rust backend with a variable-multiplication bug
(as of pulp 4.0.0, numeric coefficient * LpVariable raises `TypeError`);
`pyproject.toml` pins `pulp<3`.

## Run

```bash
PYTHONPATH=src python3 examples/capacity_allocation_demo.py
PYTHONPATH=src python3 examples/agent_workflow_demo.py
pytest
```

Or as an HTTP service:

```bash
PYTHONPATH=src uvicorn api.main:app --reload
curl localhost:8000/examples/capacity-allocation
```

Then open `localhost:8000/` for the dashboard.

Deployed on Railway at the project's service URL (`railpack.json` sets the
start command; `requirements.txt` lists runtime deps directly, since
Railpack's installer only recognizes `requirements.txt`/lockfiles, not a
bare `pyproject.toml`).

## Status (B-POP solver: Phase 1 / B-LP, per the B-POP proposal's own roadmap)

Implements basic B-LP using PPOPT + PuLP, no AI in the core, validated against
the proposal's acceptance-gate experiment (`tests/test_capacity_allocation.py`):
x\*=6, y\*=6, F\*=-12, two critical regions `[0,6]` and `[6,10]`, explicit
solutions `y*(x)=x` / `y*(x)=6`, boundary-tie handling at x=6, and a fully
explainable result. B-QP / B-MILP / B-MIQP (later B-POP phases) are not yet
implemented — and, per the Scope section above, neither is the broader AI
Agent Profiler product.
