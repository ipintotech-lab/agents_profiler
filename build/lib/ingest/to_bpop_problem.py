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

from bpop.problem import (
    BilevelProblem,
    LinearConstraint,
    LowerLevelProblem,
    UpperLevelProblem,
    Variable,
)

from .trace_schema import TraceEvent


def _is_llm_call(event: TraceEvent) -> bool:
    return event.model is not None


def build_capacity_problem_from_traces(
    events: list[TraceEvent],
    workflow: str,
    value_per_used_call: float | None = None,
    capacity_headroom: float = 2.0,
) -> BilevelProblem:
    """Build a capacity-allocation BilevelProblem from observed traces
    for a single workflow.

    :param events: normalized trace events (any workflow; filtered internally).
    :param workflow: the workflow name to analyze.
    :param value_per_used_call: assumed benefit per unit of capacity
        actually used by the follower. Defaults to 2x the observed
        average cost per LLM call.
    :param capacity_headroom: the leader's maximum allocatable capacity
        is this multiple of the observed maximum per-run usage.
    """
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

    max_demand = float(max(usage_per_run.values()))
    total_calls = sum(usage_per_run.values())
    total_cost = sum(cost_per_run.values())
    unit_cost = total_cost / total_calls if total_calls else 0.0
    unit_value = value_per_used_call if value_per_used_call is not None else 2.0 * unit_cost
    leader_capacity_bound = capacity_headroom * max_demand

    upper = UpperLevelProblem(
        variables=[Variable(name="x", lower=0.0, upper=leader_capacity_bound)],
        objective_x_coeffs={"x": unit_cost},
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
