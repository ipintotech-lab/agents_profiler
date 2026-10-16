"""HTTP service exposing the B-POP solver, plus a small test-environment
UI that runs uploaded trace files through the full
Observe -> Profile -> Mine -> Score -> Recommend pipeline.

The raw ``/solve`` endpoint lets a caller POST a BilevelProblem
definition directly and get back the leader/follower optimum with the
full critical-region audit trail; ``/analyze`` is the higher-level,
trace-file-in / report-out path backing the UI at ``/``.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
import random
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from bpop import (
    BPOP,
    BilevelProblem,
    LinearConstraint,
    LowerLevelProblem,
    UpperLevelProblem,
    Variable,
)
from bpop.result import BPOPResult
from ingest import DEFAULT_DEMAND_MODEL, normalize_raw_json, parse_raw_text
from ingest.huggingface import fetch_records, list_data_files, parse_hf_url
from ingest.trace_schema import TraceEvent
from profiler import history, run_pipeline, run_poc, run_sweep
from profiler.prices import PRICE_TABLE, lookup_price
from profiler.sweep import _build_cases
from profiler.behavior import OUTPUT_MATCHES, build_behavior
from profiler.finops import build_finops
from rawdata_gen import (
    generate_configured,
    verify_seed_range_reproducibility,
    verify_seed_reproducibility,
)
from rawdata_gen.agents.common import to_jsonl
from rawdata_gen.claude_code_sessions import generate_sessions
from rawdata_gen.generate_traces import AGENT_GENERATORS, build_dataset_manifest
from rawdata_gen.ground_truth import EXPECTED_FINDINGS


class GenerateRequest(BaseModel):
    # "events": the mock agents' trace events; "claude_code": Claude Code
    # session transcripts (sessions / prompts_per_session / omit apply)
    format: Literal["events", "claude_code"] = "events"
    agent_count: int = 3
    actions_per_agent: list[int] = [5, 5, 5]
    total_lines: int | None = None
    variance: str = "mid"
    seed: int | None = None
    sessions: int = 3
    prompts_per_session: int = 3
    omit: list[str] = []


class ValidateSeedRangeRequest(BaseModel):
    agent_count: int
    actions_per_agent: list[int]
    total_lines: int | None = None
    variance: str = "mid"
    seed_start: int = 1
    seed_end: int = 30


app = FastAPI(
    title="B-POP Solver API",
    description="Bi-level parametric optimization (leader/follower) solver prototype.",
    version="0.1.0",
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _capacity_allocation_demo_problem() -> BilevelProblem:
    """The doc3 §8 validation-experiment problem (x*=6, y*=6, F*=-12)."""
    upper = UpperLevelProblem(
        variables=[Variable(name="x", lower=0, upper=10)],
        objective_x_coeffs={"x": 2},
        objective_y_coeffs={"y": -4},
        sense="minimize",
    )
    lower = LowerLevelProblem(
        variables=[Variable(name="y", lower=0, upper=6)],
        objective_coeffs={"y": 1},
        sense="maximize",
        constraints=[
            LinearConstraint(y_coeffs={"y": 1}, sense="<=", rhs_constant=0, x_coeffs={"x": 1}, label="y<=x"),
        ],
    )
    return BilevelProblem(name="capacity-allocation", upper=upper, lower=lower)


def _serialize_result(result: BPOPResult) -> dict:
    return {
        "problem_name": result.problem_name,
        "x": result.x,
        "y": result.y,
        "objective": result.objective,
        "tied_regions": result.tied_regions,
        "candidates": [
            {
                "region_index": c.region.index,
                "feasible": c.feasible,
                "x": c.x,
                "y": c.y,
                "objective": c.objective,
                "infeasibility_reason": c.infeasibility_reason,
                "y_star": c.region.describe_y_star(),
                "region_bounds": c.region.describe_region(),
            }
            for c in result.candidates
        ],
        "explanation": result.explanation,
    }


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _parse_trace_events(raw: bytes, filename: str = "uploaded file") -> list[TraceEvent]:
    text = raw.decode("utf-8")
    stripped = text.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail=f"{filename} is empty")
    try:
        rows = normalize_raw_json(parse_raw_text(stripped))
        return [TraceEvent.model_validate(row) for row in rows]
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"could not parse {filename}: {exc}") from exc


# Records read from a Hugging Face dataset per request -- each record is a
# whole session/trajectory, so this bounds download size and pipeline time.
HF_MAX_LIMIT = 500


def _hf_error(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=502, detail=f"could not read from Hugging Face: {exc}")


@app.post("/huggingface/files")
async def huggingface_files(url: str = Form(...), token: str | None = Form(None)) -> dict:
    """The JSON / JSON Lines files in a Hugging Face dataset that /analyze
    can read, for picking one in the dashboard. A link that already names a
    file returns just that file."""
    try:
        ref = parse_hf_url(url)
        files = await run_in_threadpool(list_data_files, ref, token or None)
    except (ValueError, RuntimeError, OSError) as exc:
        raise _hf_error(exc) from exc
    if ref.path and ref.path in files:
        files = [ref.path]
    return {"repo_id": ref.repo_id, "revision": ref.revision, "files": files}


def _run_signature(events: list[TraceEvent]) -> str:
    """Content hash of one run: id, workflow, user input and the whole
    (node, input, result) sequence."""
    payload = [
        [e.run_id, e.workflow, e.user_input, e.node, e.input_state, e.result] for e in events
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _drop_runs_already_seen(
    file_events: list[TraceEvent], seen_runs: set[str]
) -> tuple[list[TraceEvent], int]:
    """Remove runs that an earlier file already contributed (resumed-session
    history), counting each once. Runs are only deduplicated *across* files,
    never within one -- repetition inside a file is the signal, not an
    artifact. Returns the kept events and how many runs were dropped."""
    by_run: dict[str, list[TraceEvent]] = defaultdict(list)
    for e in file_events:
        by_run[e.run_id].append(e)
    kept: list[TraceEvent] = []
    dropped = 0
    for run_id, run_events in by_run.items():
        signature = _run_signature(run_events)
        if signature in seen_runs:
            dropped += 1
            continue
        seen_runs.add(signature)
        kept.extend(run_events)
    return kept, dropped


async def _load_events(
    file: list[UploadFile] | None, hf_url: str | None, hf_file: str | None, hf_limit: int, hf_token: str | None
) -> tuple[list[TraceEvent], list[dict]]:
    """Events from uploaded files and / or a Hugging Face dataset file, and
    a summary per source."""
    events: list[TraceEvent] = []
    file_summaries: list[dict] = []
    seen_digests: dict[str, str] = {}
    # Resumed sessions copy their history into a new file, so the same run
    # can arrive inside two uploads. A run is the same run when its id,
    # workflow, user input and whole event sequence match -- a genuine
    # repeat habit has different content or different ids, so it is never
    # collapsed.
    seen_runs: set[str] = set()
    if not file and not hf_url:
        raise HTTPException(status_code=400, detail="upload a trace file or give a Hugging Face dataset link")
    for upload in file or []:
        raw = await upload.read()
        name = upload.filename or f"file {len(file_summaries) + 1}"
        digest = hashlib.sha256(raw).hexdigest()
        if digest in seen_digests:
            raise HTTPException(status_code=400, detail=f"{name} has the same content as {seen_digests[digest]}")
        seen_digests[digest] = name
        file_events = _parse_trace_events(raw, name)
        kept, duplicates = _drop_runs_already_seen(file_events, seen_runs)
        events.extend(kept)
        file_summaries.append(
            {"name": name, "num_events": len(kept), **({"duplicate_runs": duplicates} if duplicates else {})}
        )
    if hf_url:
        if not 1 <= hf_limit <= HF_MAX_LIMIT:
            raise HTTPException(status_code=400, detail=f"hf_limit must be between 1 and {HF_MAX_LIMIT}")
        try:
            records, path = await run_in_threadpool(fetch_records, hf_url, hf_file or None, hf_limit, hf_token or None)
        except (ValueError, RuntimeError, OSError) as exc:
            raise _hf_error(exc) from exc
        name = f"{parse_hf_url(hf_url).repo_id}/{path}"
        try:
            hf_events = [TraceEvent.model_validate(row) for row in normalize_raw_json(records)]
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"could not parse {name}: {exc}") from exc
        events.extend(hf_events)
        file_summaries.append({"name": name, "num_events": len(hf_events), "num_records": len(records)})
    return events, file_summaries


@app.post("/analyze")
async def analyze(
    file: list[UploadFile] | None = File(None),  # noqa: B008
    hf_url: str | None = Form(None),
    hf_file: str | None = Form(None),
    hf_limit: int = Form(10),
    hf_token: str | None = Form(None),
    workflow: str | None = Form(None),
    unit_cost: float | None = Form(None),
    unit_value: float | None = Form(None),
    capacity_headroom: float = Form(2.0),
    ground_truth: str | None = Form(None),
    demand_model: str = Form(DEFAULT_DEMAND_MODEL),
    price_input_per_mtok: float | None = Form(None),
    price_output_per_mtok: float | None = Form(None),
    model: str | None = Form(None),
    output_match: str = Form("exact"),
    keep_history: bool = Form(False),
) -> dict:
    """Run the uploaded trace file(s) through Observe -> Profile -> Mine ->
    Score -> Recommend for one workflow (auto-picked if not given).

    ``file`` may be repeated to upload several files (e.g. several Claude
    Code sessions); each is parsed on its own -- so they can be in different
    formats -- and their events are analyzed together. Uploading the same
    content twice is rejected, since duplicated events would read as
    repeated identical inputs and inflate determinism evidence.

    ``hf_url`` adds the first ``hf_limit`` records of a Hugging Face
    dataset file (``hf_file``, or the one the link names), fetched
    server-side; ``hf_token`` (or the server's ``HF_TOKEN``) is needed for
    gated datasets.

    ``ground_truth`` is an optional JSON-encoded ``{"agent/node": {"determinism_rate": float}}``
    manifest (as returned by ``/generate``) -- when present, the report
    includes a "validation" section scoring the profiler's classification
    against it.

    ``price_input_per_mtok`` / ``price_output_per_mtok`` ($ per 1M tokens)
    opt in to estimating cost for LLM calls that have tokens but no
    recorded cost -- offered by the report's data notice when cost is
    missing. ``model`` (e.g. ``claude-sonnet-4``) fills both prices from a
    small built-in list (see ``GET /prices``); unknown names are rejected
    rather than guessed. When no prices are given at all and the traces
    record no cost, the summary still prices its deduplicated savings at
    a few example rates (``summary.savings_scenarios``).

    ``output_match`` sets how the behavior section compares outputs
    (exact / decision / normalized). ``keep_history`` also adds the events
    to the behavior history (``GET /behavior``) -- opt-in, since it stores
    them on the server.
    """
    if output_match not in OUTPUT_MATCHES:
        raise HTTPException(status_code=400, detail=f"output_match must be one of {', '.join(OUTPUT_MATCHES)}")
    if (price_input_per_mtok is None) != (price_output_per_mtok is None):
        raise HTTPException(status_code=400, detail="give both token prices (input and output) or neither")
    token_prices = None
    if price_input_per_mtok is not None:
        if price_input_per_mtok < 0 or price_output_per_mtok < 0:
            raise HTTPException(status_code=400, detail="token prices can't be negative")
        token_prices = (price_input_per_mtok, price_output_per_mtok)
    elif model is not None:
        try:
            token_prices = lookup_price(model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    events, file_summaries = await _load_events(file, hf_url, hf_file, hf_limit, hf_token)
    parsed_ground_truth = None
    if ground_truth:
        try:
            parsed_ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"could not parse ground_truth: {exc}") from exc
    try:
        report = run_pipeline(
            events,
            workflow=workflow or None,
            unit_cost=unit_cost,
            unit_value=unit_value,
            capacity_headroom=capacity_headroom,
            ground_truth=parsed_ground_truth,
            demand_model=demand_model,
            token_prices=token_prices,
            output_match=output_match,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report["observe"]["files"] = file_summaries
    if keep_history:
        try:
            kept = await run_in_threadpool(
                history.add, events, history.dataset_label([f["name"] for f in file_summaries])
            )
        except ValueError as exc:
            kept = {"error": str(exc)}
        report["history"] = kept
    return report


@app.get("/prices")
async def prices() -> dict:
    """The built-in rough price table ($ per 1M input / output tokens) used
    when ``model=`` is passed to ``/analyze`` -- estimates for scenario math
    only; exact prices should be given via ``price_input_per_mtok`` /
    ``price_output_per_mtok``."""
    return {
        "prices": [
            {"model": name, "price_input_per_mtok": pin, "price_output_per_mtok": pout}
            for name, (pin, pout) in sorted(PRICE_TABLE.items())
        ],
        "note": "rough list-price estimates; providers change prices -- override with exact prices when it matters",
    }


@app.get("/behavior")
async def behavior(
    workflow: str | None = None,
    node: str | None = None,
    where: str | None = None,
    output_match: str = "exact",
) -> dict:
    """Behavior statistics over every event kept in history (see
    ``/analyze``'s ``keep_history``) -- the same section the report shows
    for one upload, growing with each kept upload. ``where`` (``field=value``)
    keeps only events whose input has that value; ``node`` keeps one action."""
    if output_match not in OUTPUT_MATCHES:
        raise HTTPException(status_code=400, detail=f"output_match must be one of {', '.join(OUTPUT_MATCHES)}")
    events = await run_in_threadpool(history.load, workflow)
    if node:
        events = [e for e in events if e.node == node]
    try:
        result = build_behavior(events, workflow, output_match, where)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {**result, "history": history.summary()}


@app.get("/behavior/history")
async def behavior_history() -> dict:
    return history.summary()


@app.delete("/behavior/history")
async def behavior_history_clear(workflow: str | None = None, older_than_days: float | None = None) -> dict:
    deleted = await run_in_threadpool(history.clear, workflow, older_than_days)
    return {"deleted": deleted, **history.summary()}


@app.post("/finops")
async def finops(
    file: list[UploadFile] | None = File(None),  # noqa: B008
    hf_url: str | None = Form(None),
    hf_file: str | None = Form(None),
    hf_limit: int = Form(10),
    hf_token: str | None = Form(None),
    output_match: str = Form("exact"),
    runs_per_month: float | None = Form(None),
    price_input_per_mtok: float | None = Form(None),
    price_output_per_mtok: float | None = Form(None),
) -> dict:
    """AI Agent FinOps view of the uploaded traces (same inputs as
    ``/analyze``): spend across every workflow, avoidable spend as a lower
    bound, and opportunities with a heuristic risk. ``runs_per_month``
    projects the $ figures from the observed cost per run. When the traces
    record no cost, per-token prices turn token savings into estimated
    dollars."""
    if output_match not in OUTPUT_MATCHES:
        raise HTTPException(status_code=400, detail=f"output_match must be one of {', '.join(OUTPUT_MATCHES)}")
    if runs_per_month is not None and runs_per_month <= 0:
        raise HTTPException(status_code=400, detail="runs_per_month must be positive")
    if (price_input_per_mtok is None) != (price_output_per_mtok is None):
        raise HTTPException(status_code=400, detail="give both token prices, or neither")
    events, file_summaries = await _load_events(file, hf_url, hf_file, hf_limit, hf_token)
    try:
        result = await run_in_threadpool(
            build_finops, events, output_match, runs_per_month, price_input_per_mtok, price_output_per_mtok
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["files"] = file_summaries
    return result


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    """Generate synthetic trace data (``rawdata_gen``) for the dashboard's
    "generate sample data" path: the mock agents' events, or Claude Code
    session transcripts. Returns JSONL /analyze accepts as an upload, plus a
    summary."""
    seed = req.seed if req.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)
    if req.format == "claude_code":
        return _generate_claude_code(req, rng, seed)
    try:
        runs = generate_configured(
            req.agent_count, req.actions_per_agent, req.total_lines, rng=rng, variance=req.variance
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "jsonl": to_jsonl(runs),
        "num_events": sum(len(run.events) for run in runs),
        "num_runs": len(runs),
        "runs": [
            {"run_id": run.run_id, "agent": run.agent, "workflow": run.workflow, "num_actions": len(run.events)}
            for run in runs
        ],
        **build_dataset_manifest(runs, seed=seed),
    }


def _generate_claude_code(req: GenerateRequest, rng: random.Random, seed: int) -> dict:
    try:
        sessions = generate_sessions(req.sessions, req.prompts_per_session, rng, tuple(req.omit))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rows = [row for session in sessions for row in session]
    runs = []
    for session in sessions:
        prompts = [r for r in session if r["type"] == "user" and isinstance(r["message"]["content"], str)]
        for prompt in prompts:
            runs.append(
                {
                    "run_id": prompt["promptId"],
                    "agent": "claude-code",
                    "workflow": f"claude-code/{prompt['cwd'].rsplit('/', 1)[-1]}",
                    # tool calls: the request's records other than the prompt are tool results
                    "num_actions": sum(1 for r in session if r.get("promptId") == prompt["promptId"]) - 1,
                    "prompt": prompt["message"]["content"],
                }
            )
    return {
        "format": "claude_code",
        "jsonl": "\n".join(json.dumps(row, separators=(",", ":")) for row in rows),
        "num_events": len(rows),
        "num_runs": len(runs),
        "num_sessions": len(sessions),
        "omitted": sorted(set(req.omit)),
        "runs": runs,
        "ground_truth": None,
        "dataset_id": uuid.uuid4().hex,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
    }


@app.post("/validate_seed")
def validate_seed(req: GenerateRequest) -> dict:
    """Regenerate the requested dataset twice from the same seed (a fresh
    ``seed`` is resolved and reported if none was given) and confirm the
    output is byte-identical -- direct evidence that the seed actually
    reproduces its dataset, instead of assuming reproducibility because
    the code merely accepts a seed parameter."""
    seed = req.seed if req.seed is not None else random.SystemRandom().randrange(2**32)
    try:
        return verify_seed_reproducibility(
            req.agent_count, req.actions_per_agent, req.total_lines, req.variance, seed
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/validate_seed_range")
def validate_seed_range(req: ValidateSeedRangeRequest) -> dict:
    """``verify_seed_reproducibility`` across every seed in
    [seed_start, seed_end] (default 1-30), so reproducibility is checked
    as a batch rather than trusting a single sampled seed."""
    try:
        return verify_seed_range_reproducibility(
            req.agent_count, req.actions_per_agent, req.total_lines, req.variance, req.seed_start, req.seed_end
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/run_sweep")
def run_sweep_endpoint(seed: int = 1, max_cases: int | None = None) -> dict:
    """Regression/stress-test the pipeline across a grid of agent counts,
    variances, demand models, total_lines volumes (1000-10000), and
    formula overrides -- a QA tool for the dashboard's "Run tests" tab,
    not something an end user tunes by hand. ``max_cases`` truncates the
    grid for a quick smoke run."""
    cases = _build_cases()[:max_cases] if max_cases else None
    return run_sweep(seed=seed, cases=cases)


POC_MAX_SEED_COUNT = 10
POC_MAX_CALIBRATION_SAMPLE_SIZE = 20000


@app.get("/poc/expected")
def poc_expected() -> list[dict]:
    """What each mock agent was built to contain, in generator order -- the
    "expected" side of the POC's expected-vs-found comparison."""
    runs = generate_configured(len(AGENT_GENERATORS), [8] * len(AGENT_GENERATORS), rng=random.Random(0))
    return [
        {"agent": run.agent, "workflow": run.workflow, "findings": EXPECTED_FINDINGS.get(run.workflow, [])}
        for run in runs
    ]


@app.post("/poc/run")
def run_poc_endpoint(
    calibration_seed_count: int = 5,
    validation_seed_count: int = 5,
    calibration_sample_size: int = 2000,
) -> dict:
    """Run the sample-size stabilization POC (see profiler/experiment.py)
    over HTTP. Defaults to a small number of seeds (5 + 5, vs. the
    Python-callable ``run_poc()``'s rigorous 30 + 30 default) purely so
    an HTTP call stays responsive -- the full 30-seed run is meant to be
    invoked directly from Python/tests, not through this endpoint.

    Inputs are capped server-side -- measured 10 calibration + 10
    validation seeds at calibration_sample_size=20000 at ~70s end to end,
    while a 10x larger sample size (500000) blows well past Railway's
    proxy request timeout and returns a non-JSON "upstream error" with
    the container left computing a response nobody will receive."""
    if not (1 <= calibration_seed_count <= POC_MAX_SEED_COUNT):
        raise HTTPException(
            status_code=400,
            detail=f"calibration_seed_count must be between 1 and {POC_MAX_SEED_COUNT} over HTTP "
            "-- run the rigorous 30-seed version directly from Python/tests instead.",
        )
    if not (1 <= validation_seed_count <= POC_MAX_SEED_COUNT):
        raise HTTPException(
            status_code=400,
            detail=f"validation_seed_count must be between 1 and {POC_MAX_SEED_COUNT} over HTTP "
            "-- run the rigorous 30-seed version directly from Python/tests instead.",
        )
    if not (1 <= calibration_sample_size <= POC_MAX_CALIBRATION_SAMPLE_SIZE):
        raise HTTPException(
            status_code=400,
            detail=f"calibration_sample_size must be between 1 and {POC_MAX_CALIBRATION_SAMPLE_SIZE} over HTTP "
            "-- larger sample sizes risk the request timing out at Railway's proxy; run directly from Python/tests instead.",
        )
    calibration_seeds = tuple(range(1, 1 + calibration_seed_count))
    validation_seeds = tuple(range(101, 101 + validation_seed_count))
    try:
        return run_poc(
            calibration_seeds=calibration_seeds,
            validation_seeds=validation_seeds,
            calibration_sample_size=calibration_sample_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/examples/capacity-allocation")
def capacity_allocation_example() -> dict:
    """Solve the hardcoded validation-experiment problem (doc3 §8)."""
    result = BPOP.solve(_capacity_allocation_demo_problem())
    return _serialize_result(result)


@app.post("/solve")
def solve(problem: BilevelProblem) -> dict:
    try:
        result = BPOP.solve(problem)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_result(result)
