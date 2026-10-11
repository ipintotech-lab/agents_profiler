from pathlib import Path

import pytest

from bpop import BPOP
from ingest import (
    build_capacity_problem_from_traces,
    compute_default_formula,
    compute_formula_for_all_workflows,
    sensitivity_analysis,
)
from ingest.loader import load_traces
from ingest.trace_schema import TraceEvent

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_traces.jsonl"


def test_load_traces():
    events = load_traces(DATA_PATH)
    assert len(events) == 13
    assert all(e.workflow == "cve-remediation-agent" for e in events)


def test_build_and_solve_capacity_problem():
    events = load_traces(DATA_PATH)
    # explicit "max" demand model: this test is about the historical
    # single-largest-run semantics, not the new p95 default.
    problem = build_capacity_problem_from_traces(events, workflow="cve-remediation-agent", demand_model="max")

    result = BPOP.solve(problem)

    # max observed LLM calls in a single run is 5 (run3); the follower
    # should never be allocated to use more than that.
    assert result.y["y"] <= 5.0 + 1e-6
    # feasibility: the leader never allocates below what the follower uses.
    assert result.x["x"] >= result.y["y"] - 1e-6
    assert result.objective is not None
    assert "capacity-allocation" in result.explanation


def test_default_demand_model_is_p95_and_does_not_exceed_max():
    events = load_traces(DATA_PATH)
    problem_default = build_capacity_problem_from_traces(events, workflow="cve-remediation-agent")
    problem_max = build_capacity_problem_from_traces(events, workflow="cve-remediation-agent", demand_model="max")
    # p95 of {2, 3, 5} is 4.8 -- strictly less than the observed max of 5.
    assert problem_default.lower.variables[0].upper < problem_max.lower.variables[0].upper
    assert problem_default.lower.variables[0].upper == pytest.approx(4.8)


def test_demand_model_rejects_unknown_value():
    events = load_traces(DATA_PATH)
    with pytest.raises(ValueError):
        build_capacity_problem_from_traces(events, workflow="cve-remediation-agent", demand_model="p999")


def test_compute_default_formula_reports_usage_stats_and_provenance():
    events = load_traces(DATA_PATH)
    formula = compute_default_formula(events, "cve-remediation-agent")

    assert formula["demand_model"] == "p95"
    stats = formula["usage_stats"]
    assert stats["count"] == 3
    assert stats["max"] == 5.0
    assert stats["min"] == 2.0

    provenance = formula["provenance"]
    assert provenance["unit_cost"]["label"] == "DERIVED"
    assert provenance["unit_value"]["label"] == "ASSUMED"
    assert provenance["capacity_headroom"]["label"] == "ASSUMED"
    assert provenance["max_demand"]["label"] == "DERIVED"


def test_sensitivity_analysis_sweeps_value_cost_ratios():
    events = load_traces(DATA_PATH)
    results = sensitivity_analysis(events, "cve-remediation-agent", ratios=[1.0, 2.0, 5.0])

    assert [r["ratio"] for r in results] == [1.0, 2.0, 5.0]
    # unit_value scales linearly with the ratio for a fixed unit_cost.
    assert results[1]["unit_value"] == pytest.approx(2 * results[0]["unit_value"])
    assert results[2]["unit_value"] == pytest.approx(5 * results[0]["unit_value"])
    # higher value/cost ratio should never make the objective worse (more
    # negative is better here, since the upper problem minimizes cost - value).
    objectives = [r["objective"] for r in results]
    assert objectives[-1] <= objectives[0] + 1e-9


def _llm_event(**overrides) -> TraceEvent:
    base = {
        "trace_id": "t1",
        "agent": "agent-a",
        "workflow": "wf-a",
        "run_id": "r1",
        "event_type": "llm_call",
        "node": "step",
        "input_state": {"k": "v"},
        "result": "success",
        "cost": 0.01,
        "latency_ms": 100,
        "model": "demo-llm",
    }
    base.update(overrides)
    return TraceEvent.model_validate(base)


def test_compute_formula_for_all_workflows_covers_every_workflow_present():
    events = [
        _llm_event(workflow="wf-a", run_id="r1", cost=0.01),
        _llm_event(workflow="wf-a", run_id="r2", cost=0.02),
        _llm_event(workflow="wf-b", run_id="r3", cost=0.05),
        _llm_event(workflow="wf-b", run_id="r4", cost=0.07),
    ]
    per_workflow = compute_formula_for_all_workflows(events)

    assert set(per_workflow) == {"wf-a", "wf-b"}
    assert per_workflow["wf-a"]["unit_cost"] == pytest.approx(0.015)
    assert per_workflow["wf-b"]["unit_cost"] == pytest.approx(0.06)
    assert "provenance" in per_workflow["wf-a"]
    assert "usage_stats" in per_workflow["wf-b"]


def test_compute_formula_for_all_workflows_reports_error_for_workflow_without_llm_calls():
    events = [
        _llm_event(workflow="wf-a", event_type="llm_call"),
        _llm_event(workflow="wf-b", event_type="tool_call", model=None),
    ]
    per_workflow = compute_formula_for_all_workflows(events)

    assert "provenance" in per_workflow["wf-a"]
    assert "error" in per_workflow["wf-b"]
