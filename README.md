# agents_profiler — B-POP engine

A Python implementation of **B-POP (Bi-level Parametric Optimization)**,
following the technical proposal in `B-POP_Python_Solution_Proposal_EN.docx`
and its validation experiment in
`BPOP_Technical_Solution_Proposal_with_Validation_Experiment.docx`, plus a
data-ingestion layer that builds B-POP problems from AI-agent-workflow
trace rows (the normalized trace schema from
`AI_Agent_Profiler_Product_Technical_Design_Review.docx`).

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

## Status (Phase 1 MVP per the proposal's roadmap)

Implements basic B-LP using PPOPT + PuLP, no AI in the core, validated against
the proposal's acceptance-gate experiment (`tests/test_capacity_allocation.py`):
x\*=6, y\*=6, F\*=-12, two critical regions `[0,6]` and `[6,10]`, explicit
solutions `y*(x)=x` / `y*(x)=6`, boundary-tie handling at x=6, and a fully
explainable result. B-QP / B-MILP / B-MIQP (later phases) are not yet
implemented.
