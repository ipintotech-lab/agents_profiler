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
from ingest.trace_schema import TraceEvent
from profiler import run_pipeline
from rawdata_gen import generate_configured
from rawdata_gen.agents.common import to_jsonl


class GenerateRequest(BaseModel):
    agent_count: int
    actions_per_agent: list[int]
    total_lines: int | None = None


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
        if stripped.startswith("["):
            rows = json.loads(stripped)
        else:
            rows = [json.loads(line) for line in stripped.splitlines() if line.strip()]
        return [TraceEvent.model_validate(row) for row in rows]
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"could not parse trace file: {exc}") from exc


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),  # noqa: B008
    workflow: str | None = Form(None),
    unit_cost: float | None = Form(None),
    unit_value: float | None = Form(None),
    capacity_headroom: float = Form(2.0),
) -> dict:
    """Run the uploaded trace file through Observe -> Profile -> Mine ->
    Score -> Recommend for one workflow (auto-picked if not given)."""
    raw = await file.read()
    events = _parse_trace_events(raw)
    try:
        return run_pipeline(
            events,
            workflow=workflow or None,
            unit_cost=unit_cost,
            unit_value=unit_value,
            capacity_headroom=capacity_headroom,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    """Generate synthetic trace data (vendored from
    ipintotech-lab/agents_profile_rawdata_gen) for the dashboard's
    "generate sample data" path. Returns the same JSONL format /analyze
    accepts as an upload, plus a summary."""
    try:
        runs = generate_configured(req.agent_count, req.actions_per_agent, req.total_lines)
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
    }


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
