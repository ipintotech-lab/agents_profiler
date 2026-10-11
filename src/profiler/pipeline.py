"""Orchestrates the Observe -> Profile -> Mine -> Score -> Recommend
pipeline over an uploaded batch of trace events, for a single workflow.

Profile delegates to the existing B-POP solver (``src/bpop``) via the
trace-to-problem mapping in ``src/ingest``; Observe/Mine/Score/Recommend
are implemented in ``mine.py``.
"""
from __future__ import annotations

from bpop import BPOP
from bpop.result import BPOPResult
from ingest import build_capacity_problem_from_traces, compute_default_formula
from ingest.trace_schema import TraceEvent

from .determinism import Classification
from .mine import mine, observe, recommend, score
from .validation import validate


def _serialize_bpop_result(result: BPOPResult) -> dict:
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


def run_pipeline(
    events: list[TraceEvent],
    workflow: str | None = None,
    unit_cost: float | None = None,
    unit_value: float | None = None,
    capacity_headroom: float = 2.0,
    ground_truth: dict[str, dict] | None = None,
) -> dict:
    observed = observe(events)
    if not observed["workflows"]:
        raise ValueError("no workflow found in the uploaded trace file")

    selected_workflow = workflow or observed["workflows"][0]["name"]
    if selected_workflow not in {w["name"] for w in observed["workflows"]}:
        raise ValueError(f"workflow '{selected_workflow}' not present in the uploaded trace file")

    formula_defaults = compute_default_formula(events, selected_workflow, capacity_headroom=capacity_headroom)
    formula_used = {
        "unit_cost": unit_cost if unit_cost is not None else formula_defaults["unit_cost"],
        "unit_value": unit_value if unit_value is not None else formula_defaults["unit_value"],
        "capacity_headroom": capacity_headroom,
    }

    problem = build_capacity_problem_from_traces(
        events,
        workflow=selected_workflow,
        unit_cost=unit_cost,
        value_per_used_call=unit_value,
        capacity_headroom=capacity_headroom,
    )
    bpop_result = BPOP.solve(problem)

    mined = mine(events, selected_workflow)
    scored = score(mined)
    recommendations = recommend(scored)

    agents_summary = {}
    predictions: dict[str, Classification] = {}
    for s in scored:
        bucket = agents_summary.setdefault(
            s.agent,
            {
                "num_actions": 0,
                "deterministic": 0,
                "non_deterministic": 0,
                "insufficient_evidence": 0,
                "actions": [],
            },
        )
        bucket["num_actions"] += 1
        if s.classification == Classification.DETERMINISTIC:
            bucket["deterministic"] += 1
        elif s.classification == Classification.NON_DETERMINISTIC:
            bucket["non_deterministic"] += 1
        else:
            bucket["insufficient_evidence"] += 1
        bucket["actions"].append(
            {
                "node": s.node,
                "occurrences": s.occurrences,
                "classification": s.classification.value,
                "deterministic": s.classification == Classification.DETERMINISTIC,
                "evidence": s.evidence.to_dict(),
                "distinct_outcomes": s.distinct_outcomes,
                "avg_cost": s.avg_cost,
                "avg_latency_ms": s.avg_latency_ms,
                "cost_component": round(s.cost_component, 4),
                "score": round(s.score, 4),
            }
        )
        predictions[f"{s.agent}/{s.node}"] = s.classification

    report = {
        "observe": {**observed, "selected_workflow": selected_workflow},
        "formula": {"used": formula_used, "defaults": formula_defaults},
        "profile": _serialize_bpop_result(bpop_result),
        "mine": {"by_agent": agents_summary},
        "score": {
            "weights": {"cost_weight": 0.5, "nondeterminism_weight": 0.5},
            "formula": (
                "score = cost_weight * cost_component + nondeterminism_weight * nondeterminism_component, "
                "where nondeterminism_component is 1.0 for NON_DETERMINISTIC, 0.0 for DETERMINISTIC, "
                "and 0.5 for INSUFFICIENT_EVIDENCE"
            ),
        },
        "recommend": recommendations,
    }

    if ground_truth:
        report["validation"] = validate(predictions, ground_truth).to_dict()

    return report
