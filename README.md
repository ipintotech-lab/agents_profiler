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
- a synthetic trace-data generator (`src/rawdata_gen/`, merged in from
  [`agents_profile_rawdata_gen`](https://github.com/ipintotech-lab/agents_profile_rawdata_gen),
  which it replaces) that writes mock agent events or Claude Code session
  transcripts, so the pipeline can be exercised without a real trace file;
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
  and the observed cost it would remove. `CACHE_RESULT` is never suggested
  for a call whose name says it changes something (create, update, send,
  edit, ...; `has_side_effect`), since skipping it would skip the change.
  The dashboard leads with that
  summary, then repetition, then nodes (costliest first); capacity planning
  and the per-node score list are folded under "advanced". `behavior.py`
  builds the report's `behavior` section (see "Behavior"), and `history.py`
  keeps events across uploads for it.
- `src/rawdata_gen/` — synthetic trace-data generator, merged in from
  [`agents_profile_rawdata_gen`](https://github.com/ipintotech-lab/agents_profile_rawdata_gen)
  and developed here now; `generate_configured(agent_count,
  actions_per_agent, total_lines)` builds mock agent runs,
  `claude_code_sessions.generate_sessions(...)` builds Claude Code
  transcripts, and `python -m rawdata_gen` is the CLI (see "Generating
  sample data").
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

### Generating sample data

`python -m rawdata_gen` (run from `src/`, or with `PYTHONPATH=src`) asks what
to generate, like the old repo's `run.sh`; flags skip the questions:

```bash
# the mock agents' trace events (1-6 agents, 4-20 actions each, lines in multiples of 20)
python -m rawdata_gen --agents 2 --actions 4 6 --lines 40 -o data/agent_traces.jsonl
# Claude Code session transcripts, the format under ~/.claude/projects/
python -m rawdata_gen --format claude-code --sessions 5 --prompts 3 -o data/sessions.jsonl
# ...with attributes left out, to see how the report flags incomplete data
python -m rawdata_gen --format claude-code --omit cost tokens
```

`--seed` reproduces the same data. The dashboard's "Generate sample data" tab
does the same through `/generate` (`"format": "claude_code"`, `"sessions"`,
`"prompts_per_session"`, `"omit"`).

The six mock agents (in `--agents` order) each plant findings the report
should surface, and a trap it should not:

| agent | workflow | planted | trap |
|---|---|---|---|
| security-agent | cve-remediation | rule-like `classify_severity`; varying `analyze_remediation` | caching `create_or_update_ticket` (it writes) |
| support-agent | support-ticket-triage | varying `classify_ticket`, `draft_response` | caching `send_acknowledgement` |
| release-agent | deployment-incident | varying `assess_incident` | a rule for `assess_incident` |
| code-review-agent | pull-request-review | an LLM planner walking the fixed chain `fetch_diff → run_linter → run_tests`; rule-like `decide_verdict`; varying `write_review_comments` | — |
| data-pipeline-agent | etl-incident-response | cacheable `get_table_schema`; rule-like `decide_retry`; varying `diagnose_failure` | caching `check_job_status` (its status changes) |
| sales-agent | lead-qualification | cacheable `enrich_company`, `assign_rep`; rule-like `score_lead`; varying `choose_sequence`, `write_outreach` | caching `check_crm_duplicate` |

The full list, with the reason for each, is `EXPECTED_FINDINGS` in
`src/rawdata_gen/ground_truth.py`.

Each generated run also records the request that started it (`user_input`,
e.g. "Remediate CVE-2026-4867"), as real traces do.

The CVE agent also plants a **region**: `analyze_remediation` follows
`fix_available` every time for high/critical CVEs and flips 40% of the time
for low/medium (averaging the 80% in `DETERMINISM_PROFILE`). The extra flips
use a per-run RNG, so every other node's data is unchanged.

When the analyzed workflow is one of these agents, the report adds
**06 Expected vs. found** (`report["expected_vs_found"]`): each planted
finding and trap next to what sections 01 and 04 listed, findings nobody
planted, and, when ground truth is sent, per-node determinism. **07 Validate**
then grades each "same answer every time" call against the known answer.
"Not judged: too few repeats" there is not an error: a step is only called
after the same input repeats, so small data (a few hundred lines) leaves most
steps unjudged. The dashboard defaults to 2,000 lines and won't generate fewer
than 1,000 (the CLI and `/generate` still accept any multiple of 20).

### Behavior

Report section **04 Behavior** (`report["behavior"]`, `src/profiler/behavior.py`)
describes each action of the analyzed workflow: calls, cost avg / p50 / p95,
latency p50 / p95, consistency (same input → same output, over repeated
inputs), a split by model and by agent version, and how much of the answer
each low-cardinality input field explains (normalized information gain;
fields with 2–12 values and ≥5 calls per value).

A **region** is an input value under which a simple lookup rule (a constant,
or one other field) is right ≥95% of the time, while the field's other
values, each given its own best rule, are still wrong ≥10% of the time, and
the gap passes a one-sided binomial test (p < 0.01). Letting the rest pick its
own rule keeps plain two-field functions ("reject if lint fails, else follow
the tests") from reading as regions. Regions need no repeated inputs, only
≥15 calls per value. On the mock agents the planted region shows from about
5,000 lines (10 of 10 seeds); at p < 0.01 an unplanned region shows up in
roughly 1 of 10 datasets, and it is listed as "found, not planted".

"Compare outputs" (`output_match` on `/analyze`) sets what counts as the same
answer for this section only: `exact` (decision and result), `decision`, or
`normalized` (lowercased, numbers and long ids masked). The determinism
classifier always compares exactly.

**History** is opt-in: tick "Keep these events in history" (`keep_history=true`
on `/analyze`) to store the upload's events in SQLite at `BEHAVIOR_DB`
(default `data/behavior.sqlite`). Events are deduplicated by content.
`GET /behavior?workflow=&node=&where=field=value&output_match=` builds the
same section from everything kept; `GET /behavior/history` summarizes it;
`DELETE /behavior/history?workflow=&older_than_days=` prunes it. It stores
inputs and outputs on the server. On Railway, point `BEHAVIOR_DB` at a
mounted volume, or history is lost on every deploy.

### Compiled workflows

Under **02 Repetition**, **Compiled workflows** (`report["workflow_compile"]`,
`src/profiler/workflow.py`) groups runs by task first: runs whose step sets
overlap by Jaccard ≥ 0.6 form one group, and groups under 3 runs are left
out. Mixing tasks creates false ordering conflicts; grouping removes them.
Next-step LLM calls are not steps. For each group it reports:

- one step order (mean position of each step) and the remaining
  **conflicts** (pairs seen in both orders) and **cycles**;
- **branches**: two steps that never run together;
- **parameters**: input fields that never change (hard-code) vs. ones that
  do (template slots);
- a **held-out** check: the order is learned from 4 of every 5 runs (groups
  of 5+) and replayed on the 5th, as whole-run match and pair agreement.

**Step order: ASP solver** (`order_resolver=asp` on `/analyze`) is optional.
It needs `pip install -e ".[asp]"` (clingo). It finds the order that
contradicts the fewest observed pairs (minimum feedback arc set, ≤15 steps,
5 s limit) and shows it next to the simple order, with both evidence and
held-out scores. It is used only when it contradicts less. Without clingo
it falls back to simple, and `GET /health` reports `asp.available`.
`data/order_conflicts_sample.jsonl` has a real cycle: the solver contradicts
28 observations vs. 36, but it is no better on held-out runs (pairs 80% vs.
83%). Fitting the training runs better is not the same as predicting better.

### POC: expected vs. found

The dashboard's POC tab (`POST /poc/run`) checks how much data the profiler
needs before its calls stop changing. Every phase grades all six agents'
workflows (it used to grade only the largest one). Phase 6 puts the planted
findings above next to what the report found on the same data: planted
findings found, traps flagged anyway (false alarms), findings nobody planted,
and per-node determinism expected vs. classified, at each sample size.
Synthetic agents are simpler than real ones, so a full match doesn't prove
the profiler is right on real traces, but each miss or false alarm is a
concrete case to look at.

The comparison lives in `src/profiler/expected.py`, shared by the POC and the
report. `GET /poc/expected` lists what each agent was built to contain; the
POC tab shows it in its guide before you run anything. Results follow the
report's numbered layout: 01 Summary, 02 Expected vs. found, 03 Calibration,
04 Sample size, 05 Stability, 06 Capacity, 07 Savings.

### AI Agent FinOps

The FinOps tab (`POST /finops`, in `src/profiler/finops.py`) shows spend
across all the agents in the data at once. It takes the same inputs as
`/analyze`: JSON or JSON Lines files, or a Hugging Face link
(`hf_url`, `hf_limit`, `hf_token`). Its optional fields are `output_match` and
`runs_per_month`. **Load demo data** generates the six mock agents
(5,000 lines, seed 0) and analyzes them.

Aggregate usage exports are detected automatically (JSON, JSON Lines or
CSV, `src/ingest/usage.py` and `src/profiler/usage.py`). These are token
counters per workload (`record_id`, `workload_id`, input / output /
cache-read / cache-creation tokens), or agent/trigger activations. They have
no steps, so the tab shows a usage report instead: cache-read share,
workload concentration (top 1 / top 3, Herfindahl index), records ≥10× the
median, and repeated agent/trigger pairs, with no $ figures. `/analyze`
rejects them with a 422 that points to this tab.

- **Avoidable spend is a minimum.** The findings are the same ones the report
  makes (scripted sequences, caches, rule bypasses, hidden state, regions).
  The headline sums the observed cost of the calls each finding would remove.
  It takes the largest first and skips any finding that touches an action
  already counted, so a cached call inside a scripted sequence counts once.
- **Risk is a rule of thumb.** It comes from evidence size and the kind of
  change. Low risk with strong evidence goes to "Ready to try". A cache or
  rule on an action that gives different results for the same input goes to
  "Investigate first". On 10 seeds at 2,000 and 5,000 lines, no planted trap
  was ever marked ready.
- **Not built yet: replay.** Nothing is verified against past calls, and the
  "Replay-verified" column stays empty.
- **Traces carry no dates.** Monthly figures scale the observed cost per run
  by `runs_per_month`.

The demo data gives about 25% avoidable spend (20–29% across seeds). Real
traces that don't record cost show $0, and the coverage card says so.

### Run tests

The "Run tests" tab (`POST /run_sweep`) is a health check of the pipeline:
24 settings, one change at a time from a baseline (agent count, variance,
demand model, 1,000–10,000 lines, capacity overrides). Each case reports
pass/fail, time, and how many of the analyzed agent's planted findings the
report found (`planted_found` / `planted_total` / `false_alarms`), plus the
behavior regions found (`regions`) and whether the planted one was among them
(`region_found`; only from ~5,000 lines).

### When the data is incomplete

Trace sources record different things: Claude Code transcripts have tokens,
cost, timing and real tool inputs; `cx-cmu/agent_trajectories` has no
tokens, cost or timing; FreeInference has anonymized inputs and no cost.
Missing attributes would otherwise just read as 0, so the report opens with a
notice listing what is missing and which numbers that empties (the
`data_quality` key in `/analyze`'s response).

When tokens are recorded but cost isn't, the notice offers an opt-in cost
estimate: enter $ per 1M input / output tokens and re-run. Over HTTP, pass
both `price_input_per_mtok` and `price_output_per_mtok` form fields to
`/analyze`. Recorded costs are never replaced, and the report labels the cost
as estimated -- an upper bound, since cached input is priced at the full rate.

### Hugging Face datasets

Public agent-trace datasets can be read straight from a Hugging Face link
(`src/ingest/huggingface.py`, plain HTTP -- no `huggingface_hub` needed).
It streams only the first `--limit` records (sessions / trajectories) of
one JSON / JSON Lines file, `.zst` / `.gz` included. Two dataset formats
are converted to trace events besides the ones above:

- [`cx-cmu/agent_trajectories`](https://huggingface.co/datasets/cx-cmu/agent_trajectories)
  -- OpenAI-style chat trajectories (`id` + `messages`; `chat_trajectory.py`).
  Gated: accept its conditions on the dataset page, then set `HF_TOKEN` or
  run `hf auth login`. No per-message tokens or cost in the format.
- [`harvardMadsys/freeinference_agentic_trace`](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace)
  -- gateway sessions (`session_id` + `requests`; `freeinference.py`), with
  real tokens and latency but no cost. Argument values are anonymized, so
  a call with a placeholder (`<path>`) never counts as a repeated input.

```bash
python -m ingest.huggingface https://huggingface.co/datasets/cx-cmu/agent_trajectories --list
python -m ingest.huggingface https://huggingface.co/datasets/cx-cmu/agent_trajectories \
    --file swebench.jsonl --limit 10 --out-dir cmu_sessions    # raw records, one JSON file each
python -m ingest.convert https://huggingface.co/datasets/cx-cmu/agent_trajectories \
    --hf-file swebench.jsonl --limit 10 -o traces.jsonl         # converted trace events
curl -F hf_url=https://huggingface.co/datasets/cx-cmu/agent_trajectories \
    -F hf_file=swebench.jsonl -F hf_limit=10 -F hf_token=$HF_TOKEN localhost:8000/analyze
```

The dashboard's "Hugging Face" tab does the same: paste a link, "List
files", pick one, run. `/analyze` caps `hf_limit` at 500 records; a gated
dataset needs a token in the form or `HF_TOKEN` set on the server.

#### First time? Step by step

1. **Open-access dataset** (e.g. FreeInference): nothing to set up. In the
   dashboard open the "Hugging Face" tab, paste the dataset link, click
   "List files", pick a file, set "Records" (start with 10) and click
   "Run pipeline".
2. **Gated dataset** (e.g. `cx-cmu/agent_trajectories` -- the dataset page
   shows a form instead of the files):
   1. Create a free account at https://huggingface.co/join and sign in.
   2. On the dataset page, fill in the access form and submit. Some
      datasets approve at once; others show "pending" until the authors
      approve by hand.
   3. Create a token at https://huggingface.co/settings/tokens: type
      **Read** is simplest. A **fine-grained** token also works, but only
      with "Read access to contents of all public gated repos you can
      access" ticked.
   4. Give the token to the profiler, one of:
      - dashboard: paste it in the tab's "Token" field (it's sent only to
        this server, for that request);
      - command line: `export HF_TOKEN=hf_...`, or
        `python -m pip install datasets huggingface_hub` then `hf auth login` (saves it to
        `~/.cache/huggingface/token`, which is picked up automatically);
      - deployed server: set `HF_TOKEN` in the Railway service's variables.
3. **Which file?** One dataset often has several files. "List files" (or
   `python -m ingest.huggingface <link> --list`) shows the ones the
   profiler can read -- JSON / JSON Lines, `.zst` / `.gz` included;
   Parquet isn't supported. Metadata files (manifests, reports, tool
   inventories) are listed too but aren't traces -- pick a data file such
   as `swebench.jsonl` or `release/<week>/traces/<day>.jsonl.zst`.
4. **Errors:**
   - *HTTP 401, "no Hugging Face token found"* -- step 2.4.
   - *HTTP 403, "the token was rejected or hasn't been granted access"* --
     access is still pending, the request was for another account, or a
     fine-grained token lacks the gated-repos permission (step 2.3).
     `hf auth whoami` shows which account a saved token belongs to.
   - *"has N readable files; pick one"* -- choose a file (step 3).
   - *"could not parse"* -- the file isn't in a trace format the profiler
     knows; see the formats listed above.

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

### Multi-tenant (SaaS) mode

`AUTH_MODE` picks the mode: `demo` (default, the public demo above; not for
customer data), `dev` (local test sign-in) or `oidc` (production). Outside
demo mode, customer routes need a session, data is scoped per tenant,
uploads go through `/api/files` (validated, quota-limited, private
storage) and analyses run in a sandboxed worker (`python -m saas.worker`).
What's done, what's left and what needs infrastructure:
[`docs/security/saas-readiness.md`](docs/security/saas-readiness.md).

## Status (B-POP solver: Phase 1 / B-LP, per the B-POP proposal's own roadmap)

Implements basic B-LP using PPOPT + PuLP, no AI in the core, validated against
the proposal's acceptance-gate experiment (`tests/test_capacity_allocation.py`):
x\*=6, y\*=6, F\*=-12, two critical regions `[0,6]` and `[6,10]`, explicit
solutions `y*(x)=x` / `y*(x)=6`, boundary-tie handling at x=6, and a fully
explainable result. B-QP / B-MILP / B-MIQP (later B-POP phases) are not yet
implemented — and, per the Scope section above, neither is the broader AI
Agent Profiler product.
