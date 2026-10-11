from pathlib import Path

from ingest.loader import load_traces
from ingest.trace_schema import TraceEvent
from profiler import run_pipeline
from profiler.determinism import Classification
from profiler.mine import mine, observe, recommend, score

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_traces.jsonl"


def _event(**overrides) -> TraceEvent:
    base = {
        "trace_id": "t1",
        "agent": "agent-a",
        "workflow": "wf",
        "run_id": "r1",
        "event_type": "action",
        "node": "step",
        "input_state": {"k": "v"},
        "result": "success",
        "cost": 0.01,
        "latency_ms": 100,
    }
    base.update(overrides)
    return TraceEvent.model_validate(base)


def test_observe_counts_sample_traces():
    events = load_traces(DATA_PATH)
    observed = observe(events)
    assert observed["num_events"] == 13
    assert observed["num_agents"] == 1
    assert observed["agents"] == ["security-agent"]
    assert observed["num_workflows"] == 1
    assert observed["workflows"][0] == {
        "name": "cve-remediation-agent",
        "num_events": 13,
        "num_runs": 3,
    }


def test_mine_classifies_repeated_consistent_input_as_deterministic():
    # 20 identical-input, identical-outcome observations easily clears the
    # default thresholds (min_repeated_inputs=10 is about *distinct*
    # repeated inputs, so we use a lowered threshold here to keep the
    # fixture small while still exercising the real DETERMINISTIC path).
    events = [_event(run_id=f"r{i}", result="success") for i in range(20)]
    stats = mine(events, "wf", min_repeated_inputs=1, min_repeated_observations=20)
    assert len(stats) == 1
    assert stats[0].classification == Classification.DETERMINISTIC
    assert stats[0].occurrences == 20


def test_mine_classifies_same_input_different_outcome_as_non_deterministic():
    # A single repeated input with conflicting outcomes is sufficient
    # evidence of non-determinism regardless of thresholds/sample size.
    events = [
        _event(run_id="r1", result="success"),
        _event(run_id="r2", result="failure"),
    ]
    stats = mine(events, "wf")
    assert len(stats) == 1
    assert stats[0].classification == Classification.NON_DETERMINISTIC
    assert len(stats[0].distinct_outcomes) == 2


def test_mine_no_repeated_inputs_is_insufficient_evidence():
    # Every input occurs exactly once -- no input was ever tested twice,
    # so there is no basis to call this deterministic, however much data
    # there is. This is the exact bug the three-state classifier fixes.
    events = [
        _event(input_state={"k": "a"}, result="success"),
        _event(input_state={"k": "b"}, result="failure"),
        _event(input_state={"k": "c"}, result="success"),
    ]
    stats = mine(events, "wf")
    assert len(stats) == 1
    assert stats[0].classification == Classification.INSUFFICIENT_EVIDENCE


def test_mine_many_unique_inputs_no_repeats_stays_insufficient_evidence():
    events = [_event(input_state={"k": i}, result="success") for i in range(100)]
    stats = mine(events, "wf")
    assert len(stats) == 1
    assert stats[0].classification == Classification.INSUFFICIENT_EVIDENCE
    assert stats[0].evidence.repeated_inputs == 0


def test_score_normalizes_cost_against_most_expensive_node_and_penalizes_nondeterminism():
    events = [
        _event(node="cheap", cost=0.01, run_id="r1", result="success"),
        _event(node="cheap", cost=0.01, run_id="r2", result="success"),
        _event(node="expensive", cost=0.05, run_id="r1", result="success"),
        _event(node="expensive", cost=0.05, run_id="r2", result="failure"),
    ]
    stats = score(mine(events, "wf"))
    by_node = {s.node: s for s in stats}

    assert by_node["expensive"].cost_component == 1.0
    assert by_node["expensive"].classification == Classification.NON_DETERMINISTIC
    assert by_node["expensive"].score == 1.0

    # "cheap" has one repeated input with a consistent outcome but only 2
    # observations -- far below the default thresholds, so it is
    # INSUFFICIENT_EVIDENCE (nondeterminism_component=0.5), not DETERMINISTIC.
    assert round(by_node["cheap"].cost_component, 4) == 0.2
    assert by_node["cheap"].classification == Classification.INSUFFICIENT_EVIDENCE
    assert round(by_node["cheap"].score, 4) == 0.35


def test_recommend_orders_worst_first_and_messages_match_bucket():
    events = [
        _event(node="cheap_stable", cost=0.0, run_id="r1", result="success"),
        _event(node="expensive_flaky", cost=0.05, run_id="r1", result="success"),
        _event(node="expensive_flaky", cost=0.05, run_id="r2", result="failure"),
    ]
    recs = recommend(score(mine(events, "wf")))
    assert recs[0]["node"] == "expensive_flaky"
    assert recs[0]["classification"] == "NON_DETERMINISTIC"
    assert recs[0]["deterministic"] is False
    assert "non-deterministic and high-cost" in recs[0]["message"]
    assert recs[-1]["node"] == "cheap_stable"
    assert recs[-1]["classification"] == "INSUFFICIENT_EVIDENCE"
    assert "insufficient evidence to classify" in recs[-1]["message"]

    assert "cost_component = avg_cost / max_cost" in recs[0]["derivation"]
    assert "non-deterministic (2 distinct outcomes)" in recs[0]["derivation"]


def test_recommend_never_calls_insufficient_evidence_node_safe_to_cache():
    events = [_event(input_state={"k": i}, result="success") for i in range(50)]
    recs = recommend(score(mine(events, "wf")))
    assert recs[0]["classification"] == "INSUFFICIENT_EVIDENCE"
    assert "safe to cache" not in recs[0]["message"]
    assert "insufficient evidence" in recs[0]["message"]


def test_run_pipeline_end_to_end_on_sample_traces():
    events = load_traces(DATA_PATH)
    report = run_pipeline(events)

    assert report["observe"]["selected_workflow"] == "cve-remediation-agent"
    assert report["profile"]["x"]["x"] >= report["profile"]["y"]["y"] - 1e-6

    by_agent = report["mine"]["by_agent"]
    assert "security-agent" in by_agent
    assert by_agent["security-agent"]["num_actions"] == 6
    # Every node in this fixture has only 1 distinct canonical input, so
    # none of them clear the repeated-input evidence thresholds -- the
    # correct, non-overclaiming outcome is INSUFFICIENT_EVIDENCE for all.
    assert by_agent["security-agent"]["insufficient_evidence"] == 6
    assert by_agent["security-agent"]["deterministic"] == 0
    assert by_agent["security-agent"]["non_deterministic"] == 0

    assert len(report["recommend"]) == 6
    scores = [r["score"] for r in report["recommend"]]
    assert scores == sorted(scores, reverse=True)
    assert "validation" not in report


def test_run_pipeline_formula_override_changes_profile_result():
    events = load_traces(DATA_PATH)
    default_report = run_pipeline(events)
    overridden_report = run_pipeline(events, unit_cost=10.0, unit_value=0.01)

    assert overridden_report["formula"]["used"]["unit_cost"] == 10.0
    assert overridden_report["profile"]["objective"] != default_report["profile"]["objective"]


def test_run_pipeline_unknown_workflow_raises():
    events = load_traces(DATA_PATH)
    try:
        run_pipeline(events, workflow="does-not-exist")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_run_pipeline_with_ground_truth_adds_validation_section():
    events = load_traces(DATA_PATH)
    ground_truth = {"security-agent/classify_severity": {"determinism_rate": 1.0}}
    report = run_pipeline(events, ground_truth=ground_truth)

    assert "validation" in report
    validation = report["validation"]
    assert validation["ground_truth_nodes"] == 1
    # The sample fixture's single observation per node is INSUFFICIENT_EVIDENCE,
    # which must never be silently scored as correct or incorrect.
    assert validation["insufficient_evidence_nodes"] == 1
    assert validation["classified_nodes"] == 0
    assert validation["precision"] is None


def test_run_pipeline_reports_capacity_context_separately_from_score():
    events = load_traces(DATA_PATH)
    report = run_pipeline(events)

    assert "capacity_context" in report
    assert report["observe"]["selected_workflow"] in report["capacity_context"]
    # The architectural decision (B-POP stays decoupled from per-node
    # scoring) must hold: no recommendation dict carries any B-POP field.
    for rec in report["recommend"]:
        assert "x" not in rec and "y" not in rec and "objective" not in rec


def test_run_pipeline_includes_sensitivity_sweep_in_profile():
    events = load_traces(DATA_PATH)
    report = run_pipeline(events, sensitivity_ratios=[1.0, 3.0])

    sensitivity = report["profile"]["sensitivity"]
    assert [s["ratio"] for s in sensitivity] == [1.0, 3.0]


def test_run_pipeline_demand_model_changes_formula_defaults():
    events = load_traces(DATA_PATH)
    report_p95 = run_pipeline(events)
    report_max = run_pipeline(events, demand_model="max")

    assert report_p95["formula"]["defaults"]["demand_model"] == "p95"
    assert report_max["formula"]["defaults"]["demand_model"] == "max"
    assert report_p95["formula"]["defaults"]["max_demand"] < report_max["formula"]["defaults"]["max_demand"]
