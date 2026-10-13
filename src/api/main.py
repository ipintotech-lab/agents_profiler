"""HTTP service exposing the B-POP solver, plus a small test-environment
UI that runs uploaded trace files through the full
Observe -> Profile -> Mine -> Score -> Recommend pipeline.

The raw ``/solve`` endpoint lets a caller POST a BilevelProblem
definition directly and get back the leader/follower optimum with the
full critical-region audit trail; ``/analyze`` is the higher-level,
trace-file-in / report-out path backing the UI at ``/``.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
from ingest import DEFAULT_DEMAND_MODEL, normalize_raw_json
from ingest.trace_schema import TraceEvent
from profiler import run_pipeline, run_poc, run_sweep
from rawdata_gen import (
    generate_configured,
    verify_seed_range_reproducibility,
    verify_seed_reproducibility,
)
from rawdata_gen.agents.common import to_jsonl
from rawdata_gen.generate_traces import build_dataset_manifest


class GenerateRequest(BaseModel):
    agent_count: int
    actions_per_agent: list[int]
    total_lines: int | None = None
    variance: str = "mid"
    seed: int | None = None


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


def _parse_trace_events(raw: bytes) -> list[TraceEvent]:
    text = raw.decode("utf-8")
    stripped = text.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    try:
        # A single JSON document (array or object, pretty-printed or not)
        # takes priority -- only fall back to JSONL (one JSON value per
        # line) when the whole blob doesn't parse as one document, since a
        # pretty-printed array has multiple non-empty lines too.
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            lines = [line for line in stripped.splitlines() if line.strip()]
            parsed = [json.loads(line) for line in lines]
        rows = normalize_raw_json(parsed)
        return [TraceEvent.model_validate(row) for row in rows]
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"could not parse trace file: {exc}") from exc


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),  # noqa: B008
    workflow: str | None = Form(None),
    unit_cost: float | None = Form(None),
    unit_value: float | None = Form(None),
    capacity_headroom: float = Form(2.0),
    ground_truth: str | None = Form(None),
    demand_model: str = Form(DEFAULT_DEMAND_MODEL),
) -> dict:
    """Run the uploaded trace file through Observe -> Profile -> Mine ->
    Score -> Recommend for one workflow (auto-picked if not given).

    ``ground_truth`` is an optional JSON-encoded ``{"agent/node": {"determinism_rate": float}}``
    manifest (as returned by ``/generate``) -- when present, the report
    includes a "validation" section scoring the profiler's classification
    against it.
    """
    raw = await file.read()
    events = _parse_trace_events(raw)
    parsed_ground_truth = None
    if ground_truth:
        try:
            parsed_ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"could not parse ground_truth: {exc}") from exc
    try:
        return run_pipeline(
            events,
            workflow=workflow or None,
            unit_cost=unit_cost,
            unit_value=unit_value,
            capacity_headroom=capacity_headroom,
            ground_truth=parsed_ground_truth,
            demand_model=demand_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    """Generate synthetic trace data (vendored from
    ipintotech-lab/agents_profile_rawdata_gen) for the dashboard's
    "generate sample data" path. Returns the same JSONL format /analyze
    accepts as an upload, plus a summary."""
    seed = req.seed if req.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(seed)
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
def run_sweep_endpoint(seed: int = 1) -> dict:
    """Regression/stress-test the pipeline across a grid of agent counts,
    variances, demand models, total_lines volumes (1000-10000), and
    formula overrides -- a QA tool for the dashboard's "Run tests" tab,
    not something an end user tunes by hand."""
    return run_sweep(seed=seed)


POC_MAX_SEED_COUNT = 10
POC_MAX_CALIBRATION_SAMPLE_SIZE = 2000


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

    Inputs are capped server-side: a request asking for more seeds or a
    larger calibration sample than the defaults support can burn enough
    CPU/memory on a single call to take the whole service down (this has
    happened in production -- a calibration_sample_size=20000 request
    tripped the Railway plan's CPU limit and restarted the container)."""
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
            "-- larger sample sizes are too CPU/memory-heavy for an HTTP request; run directly from Python/tests instead.",
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
