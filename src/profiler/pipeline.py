"""Orchestrates the Observe -> Profile -> Mine -> Score -> Recommend
pipeline over an uploaded batch of trace events, for a single workflow.

Profile delegates to the existing B-POP solver (``src/bpop``) via the
trace-to-problem mapping in ``src/ingest``; Observe/Mine/Score/Recommend
are implemented in ``mine.py``.
"""
from __future__ import annotations

from collections import defaultdict

from bpop import BPOP
from bpop.result import BPOPResult
from ingest import (
    DEFAULT_DEMAND_MODEL,
    build_capacity_problem_from_traces,
    compute_default_formula,
    compute_formula_for_all_workflows,
    sensitivity_analysis,
)
from ingest.trace_schema import TraceEvent

from .behavior import build_behavior
from .calibration import calibrate_determinism_thresholds
from .data_quality import check_data_quality, estimate_missing_costs
from .determinism import Classification
from .expected import compare_workflow, has_expected, summarize
from .mine import mine, observe, recommend, score
from .opportunities import build_summary
from .patterns import decision_costs, find_repetition
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
    demand_model: str = DEFAULT_DEMAND_MODEL,
    sensitivity_ratios: list[float] | None = None,
    token_prices: tuple[float, float] | None = None,
    output_match: str = "exact",
) -> dict:
    """``token_prices`` -- $ per 1M input / output tokens -- opts in to
    estimating cost for LLM calls that have tokens but no recorded cost.
    ``output_match`` -- how the behavior section compares outputs (see
    ``behavior.OUTPUT_MATCHES``); the determinism classification always
    compares them exactly."""
    recorded = events
    estimated = None
    if token_prices is not None:
        events, estimated = estimate_missing_costs(events, *token_prices)
    observed = observe(events)
    if not observed["workflows"]:
        raise ValueError("no workflow found in the uploaded trace file")

    selected_workflow = workflow or observed["workflows"][0]["name"]
    if selected_workflow not in {w["name"] for w in observed["workflows"]}:
        raise ValueError(f"workflow '{selected_workflow}' not present in the uploaded trace file")
    if estimated is not None:
        # how many of the analyzed workflow's calls got an estimate
        filled = sum(1 for a, b in zip(recorded, events, strict=True) if a.workflow == selected_workflow and a.cost != b.cost)
        estimated = {**estimated, "events": filled} if filled else None

    formula_defaults = compute_default_formula(
        events, selected_workflow, capacity_headroom=capacity_headroom, demand_model=demand_model
    )
    formula_by_workflow = compute_formula_for_all_workflows(
        events, capacity_headroom=capacity_headroom, demand_model=demand_model
    )
    formula_used = {
        "unit_cost": unit_cost if unit_cost is not None else formula_defaults["unit_cost"],
        "unit_value": unit_value if unit_value is not None else formula_defaults["unit_value"],
        "capacity_headroom": capacity_headroom,
        "demand_model": demand_model,
    }

    problem = build_capacity_problem_from_traces(
        events,
        workflow=selected_workflow,
        unit_cost=unit_cost,
        value_per_used_call=unit_value,
        capacity_headroom=capacity_headroom,
        demand_model=demand_model,
    )
    bpop_result = BPOP.solve(problem)
    sensitivity = sensitivity_analysis(
        events,
        selected_workflow,
        ratios=sensitivity_ratios,
        unit_cost=unit_cost,
        capacity_headroom=capacity_headroom,
        demand_model=demand_model,
    )

    mined = mine(events, selected_workflow)
    scored = score(mined)
    recommendations = recommend(scored)

    # Charge each step for the LLM call that chose it, so a tool's cost
    # isn't shown as $0 when an LLM call was spent deciding to run it.
    relevant = [e for e in events if e.workflow == selected_workflow]
    shares = decision_costs(relevant)
    decision_cost: dict[tuple[str, str], float] = defaultdict(float)
    node_tokens: dict[tuple[str, str], int] = defaultdict(int)
    for e in relevant:
        decision_cost[(e.agent, e.node)] += shares.get(id(e), 0.0)
        node_tokens[(e.agent, e.node)] += e.tokens.input + e.tokens.output
    repetition = find_repetition(events, selected_workflow)

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
                "decision_cost": decision_cost[(s.agent, s.node)],
                "total_cost": s.avg_cost * s.occurrences + decision_cost[(s.agent, s.node)],
                "tokens": node_tokens[(s.agent, s.node)],
                "avg_latency_ms": s.avg_latency_ms,
                "cost_component": round(s.cost_component, 4),
                "score": round(s.score, 4),
                "fingerprints": s.fingerprints,
            }
        )
        predictions[f"{s.agent}/{s.node}"] = s.classification

    # B-POP <-> Mine/Score/Recommend coupling (architectural decision, P1):
    # B-POP solves one workflow-level capacity allocation (a single x*/y*
    # scalar pair), while Score/Recommend rank many individual nodes. Mixing
    # the B-POP objective into each node's score would conflate two
    # different decision scopes -- capacity sizing vs. per-action cost/risk
    # -- behind one opaque number, and would make Score depend on Profile
    # succeeding. Instead the coupling stays *contextual*: B-POP's result is
    # surfaced alongside the recommendations as read-only context a reader
    # can use to judge whether a costly node's spend lines up with the
    # workflow's sized capacity, but it never changes `score`.
    bpop_capacity_context = (
        f"B-POP sizes '{selected_workflow}' capacity at x*={bpop_result.x.get('x', 0):.3f} "
        f"(allocated) against y*={bpop_result.y.get('y', 0):.3f} (used), from a "
        f"{demand_model} demand model over observed per-run usage; objective={bpop_result.objective:.4f}. "
        "This is workflow-level context for the recommendations below, not an input to any node's score."
    )

    for bucket in agents_summary.values():
        bucket["actions"].sort(key=lambda a: -a["total_cost"])

    report = {
        "summary": build_summary(events, selected_workflow, repetition, scored),
        "data_quality": check_data_quality([e for e in recorded if e.workflow == selected_workflow], estimated),
        "observe": {**observed, "selected_workflow": selected_workflow},
        "formula": {"used": formula_used, "defaults": formula_defaults, "per_workflow": formula_by_workflow},
        "profile": {**_serialize_bpop_result(bpop_result), "sensitivity": sensitivity},
        "mine": {"by_agent": agents_summary},
        "score": {
            "weights": {"cost_weight": 0.5, "nondeterminism_weight": 0.5},
            "formula": (
                "score = cost_weight * cost_component + nondeterminism_weight * nondeterminism_component, "
                "where nondeterminism_component is 1.0 for NON_DETERMINISTIC, 0.0 for DETERMINISTIC, "
                "and 0.5 for INSUFFICIENT_EVIDENCE. B-POP's capacity solve is not part of this formula -- "
                "see 'capacity_context'."
            ),
        },
        "recommend": recommendations,
        "repetition": repetition,
        "capacity_context": bpop_capacity_context,
        "behavior": build_behavior(events, selected_workflow, output_match),
    }

    if has_expected(selected_workflow):
        # data from one of the mock agents: show what it was built to contain
        compared = compare_workflow(
            events, selected_workflow, ground_truth, stats=mined, repetition=repetition, behavior=report["behavior"]
        )
        report["expected_vs_found"] = {
            **compared,
            "workflow": selected_workflow,
            "summary": summarize(compared["rows"], compared["unplanned"], compared["determinism"]),
        }

    if ground_truth:
        report["validation"] = validate(predictions, ground_truth).to_dict()
        report["calibration"] = calibrate_determinism_thresholds(events, selected_workflow, ground_truth)

    return report
