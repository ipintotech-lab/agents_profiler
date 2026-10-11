"""Minimal HTTP service exposing the B-POP solver.

Lets a caller POST a BilevelProblem definition and get back the
leader/follower optimum with the full critical-region audit trail —
the same thing ``examples/capacity_allocation_demo.py`` prints, over
HTTP instead of stdout.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

from bpop import (
    BPOP,
    BilevelProblem,
    LinearConstraint,
    LowerLevelProblem,
    UpperLevelProblem,
    Variable,
)
from bpop.result import BPOPResult

app = FastAPI(
    title="B-POP Solver API",
    description="Bi-level parametric optimization (leader/follower) solver prototype.",
    version="0.1.0",
)


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
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
