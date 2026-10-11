"""Map raw agent-workflow trace rows into a B-POP capacity-allocation
problem, structurally identical to the proposal's validation
experiment (§8 of the technical solution proposal) but with the
leader/follower parameters estimated from observed trace data instead
of hand-picked constants.

Mapping:
  - follower's decision variable y = LLM capacity *used* by a run
    (count of LLM-call events in that run).
  - leader's decision variable x = LLM capacity *allocated* per run
    (a budget the leader could set, e.g. a per-run call quota).
  - max observed usage across runs plays the role of the follower's
    fixed maximum demand (the "6" in the validation experiment).
  - average cost per LLM call is the leader's unit cost of allocating
    capacity; ``value_per_used_call`` is the assumed benefit per call
    actually used (not observable in traces, so it is a caller-supplied
    assumption, defaulting to 2x the unit cost).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from bpop import BPOP
from bpop.problem import (
    BilevelProblem,
    LinearConstraint,
    LowerLevelProblem,
    UpperLevelProblem,
    Variable,
)

from .trace_schema import TraceEvent

#: Supported demand models for sizing the follower's fixed maximum demand
#: (and, via ``capacity_headroom``, the leader's capacity bound). "max" uses
#: the single largest observed per-run usage, which overfits to one outlier
#: run; "pXX" uses the XXth percentile of observed per-run usage instead, so
#: a single abnormally large run doesn't singlehandedly size the capacity
#: plan. "p95" is the default per the demand-model-selection requirement.
DEMAND_MODELS = ("max", "p50", "p75", "p90", "p95", "p99")
DEFAULT_DEMAND_MODEL = "p95"


def _is_llm_call(event: TraceEvent) -> bool:
    return event.model is not None


def _usage_and_cost_per_run(events: list[TraceEvent], workflow: str) -> tuple[dict[str, int], dict[str, float]]:
    relevant = [e for e in events if e.workflow == workflow]
    if not relevant:
        raise ValueError(f"no trace events found for workflow '{workflow}'")

    usage_per_run: dict[str, int] = defaultdict(int)
    cost_per_run: dict[str, float] = defaultdict(float)
    for e in relevant:
        if _is_llm_call(e):
            usage_per_run[e.run_id] += 1
            cost_per_run[e.run_id] += e.cost

    if not usage_per_run:
        raise ValueError(f"no LLM-call events found for workflow '{workflow}'")
    return usage_per_run, cost_per_run


def _resolve_demand(usage_values: list[int], demand_model: str) -> float:
    if demand_model not in DEMAND_MODELS:
        raise ValueError(f"demand_model must be one of {DEMAND_MODELS}, got '{demand_model}'")
    if demand_model == "max":
        return float(max(usage_values))
    percentile = int(demand_model[1:])
    return float(np.percentile(usage_values, percentile))


def usage_statistics(usage_values: list[int]) -> dict[str, float]:
    """Summary statistics over observed per-run LLM-call usage, so a caller
    can see the distribution a demand model was computed from rather than
    just the single resolved number."""
    values = np.asarray(usage_values, dtype=float)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def compute_default_formula(
    events: list[TraceEvent],
    workflow: str,
    capacity_headroom: float = 2.0,
    demand_model: str = DEFAULT_DEMAND_MODEL,
) -> dict:
    """The formula values ``build_capacity_problem_from_traces`` would pick
    on its own for this workflow, exposed so a caller (e.g. a UI) can show
    them before letting the user override any of them.

    Also reports ``usage_stats`` (the full observed per-run usage
    distribution) and ``provenance`` -- for each formula field, whether it
    is OBSERVED directly from trace data, DERIVED by a computation over
    observed data, or ASSUMED (a default not backed by any trace signal).
    """
    usage_per_run, cost_per_run = _usage_and_cost_per_run(events, workflow)
    demand = _resolve_demand(list(usage_per_run.values()), demand_model)
    total_calls = sum(usage_per_run.values())
    total_cost = sum(cost_per_run.values())
    unit_cost = total_cost / total_calls if total_calls else 0.0
    return {
        "unit_cost": unit_cost,
        "unit_value": 2.0 * unit_cost,
        "capacity_headroom": capacity_headroom,
        "max_demand": demand,
        "demand_model": demand_model,
        "usage_stats": usage_statistics(list(usage_per_run.values())),
        "provenance": {
            "unit_cost": {
                "label": "DERIVED",
                "note": "total observed LLM-call cost / total observed LLM-call count for this workflow",
            },
            "unit_value": {
                "label": "ASSUMED",
                "note": "not observable in traces (no 'value of a used call' signal exists) -- "
                "defaults to 2x the derived unit_cost",
            },
            "capacity_headroom": {
                "label": "ASSUMED",
                "note": "a caller-chosen multiplier with no trace signal behind it; defaults to 2.0",
            },
            "max_demand": {
                "label": "DERIVED",
                "note": f"the {demand_model} of observed per-run LLM-call usage ({demand:.3f}), "
                "not simply the single largest observed run",
            },
        },
    }


def compute_formula_for_all_workflows(
    events: list[TraceEvent],
    capacity_headroom: float = 2.0,
    demand_model: str = DEFAULT_DEMAND_MODEL,
) -> dict[str, dict]:
    """``compute_default_formula`` for every workflow present in ``events``,
    not just the one selected for the B-POP solve -- so provenance,
    usage_stats, and demand-model-derived values can be compared across
    workflows instead of only ever being visible for whichever workflow
    happens to be selected.

    A workflow with no LLM-call events (so ``compute_default_formula``
    can't derive ``unit_cost``/``max_demand`` for it) is reported with an
    ``"error"`` key instead of raising, so one unprofilable workflow
    doesn't hide the others' results.
    """
    workflows = sorted({e.workflow for e in events if e.workflow})
    per_workflow: dict[str, dict] = {}
    for wf in workflows:
        try:
            per_workflow[wf] = compute_default_formula(
                events, wf, capacity_headroom=capacity_headroom, demand_model=demand_model
            )
        except ValueError as exc:
            per_workflow[wf] = {"error": str(exc)}
    return per_workflow


def sensitivity_analysis(
    events: list[TraceEvent],
    workflow: str,
    ratios: list[float] | None = None,
    unit_cost: float | None = None,
    capacity_headroom: float = 2.0,
    demand_model: str = DEFAULT_DEMAND_MODEL,
) -> list[dict]:
    """Re-solve the capacity-allocation problem across a grid of
    unit_value/unit_cost ratios, holding unit_cost (and everything else)
    fixed. ``unit_value`` is ASSUMED, not observed, so this shows how
    sensitive the B-POP leader/follower optimum is to that assumption
    instead of presenting a single ratio's result as if it were certain.
    """

    ratios = ratios if ratios is not None else [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    usage_per_run, cost_per_run = _usage_and_cost_per_run(events, workflow)
    total_calls = sum(usage_per_run.values())
    total_cost = sum(cost_per_run.values())
    observed_unit_cost = total_cost / total_calls if total_calls else 0.0
    resolved_unit_cost = unit_cost if unit_cost is not None else observed_unit_cost

    results = []
    for ratio in ratios:
        problem = build_capacity_problem_from_traces(
            events,
            workflow=workflow,
            unit_cost=resolved_unit_cost,
            value_per_used_call=ratio * resolved_unit_cost,
            capacity_headroom=capacity_headroom,
            demand_model=demand_model,
        )
        result = BPOP.solve(problem)
        results.append(
            {
                "ratio": ratio,
                "unit_value": ratio * resolved_unit_cost,
                "x": result.x,
                "y": result.y,
                "objective": result.objective,
            }
        )
    return results


def build_capacity_problem_from_traces(
    events: list[TraceEvent],
    workflow: str,
    unit_cost: float | None = None,
    value_per_used_call: float | None = None,
    capacity_headroom: float = 2.0,
    demand_model: str = DEFAULT_DEMAND_MODEL,
) -> BilevelProblem:
    """Build a capacity-allocation BilevelProblem from observed traces
    for a single workflow.

    :param events: normalized trace events (any workflow; filtered internally).
    :param workflow: the workflow name to analyze.
    :param unit_cost: the leader's unit cost of allocating capacity.
        Defaults to the observed average cost per LLM call.
    :param value_per_used_call: assumed benefit per unit of capacity
        actually used by the follower. Defaults to 2x the observed
        average cost per LLM call.
    :param capacity_headroom: the leader's maximum allocatable capacity
        is this multiple of the demand-model-resolved per-run usage.
    :param demand_model: how the follower's fixed maximum demand is sized
        from observed per-run usage -- "max" or a "pXX" percentile
        (default "p95"). See ``DEMAND_MODELS``.
    """
    usage_per_run, cost_per_run = _usage_and_cost_per_run(events, workflow)
    max_demand = _resolve_demand(list(usage_per_run.values()), demand_model)
    total_calls = sum(usage_per_run.values())
    total_cost = sum(cost_per_run.values())
    observed_unit_cost = total_cost / total_calls if total_calls else 0.0
    resolved_unit_cost = unit_cost if unit_cost is not None else observed_unit_cost
    unit_value = value_per_used_call if value_per_used_call is not None else 2.0 * observed_unit_cost
    leader_capacity_bound = capacity_headroom * max_demand

    upper = UpperLevelProblem(
        variables=[Variable(name="x", lower=0.0, upper=leader_capacity_bound)],
        objective_x_coeffs={"x": resolved_unit_cost},
        objective_y_coeffs={"y": -unit_value},
        sense="minimize",
    )
    lower = LowerLevelProblem(
        variables=[Variable(name="y", lower=0.0, upper=max_demand)],
        objective_coeffs={"y": 1.0},
        sense="maximize",
        constraints=[
            LinearConstraint(y_coeffs={"y": 1.0}, sense="<=", rhs_constant=0.0, x_coeffs={"x": 1.0}, label="y<=x"),
        ],
    )
    return BilevelProblem(name=f"capacity-allocation[{workflow}]", upper=upper, lower=lower)
