from pathlib import Path

from ingest.loader import load_traces
from ingest.trace_schema import TraceEvent
from profiler import run_pipeline
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


def test_mine_classifies_same_input_same_outcome_as_deterministic():
    events = [
        _event(run_id="r1", result="success"),
        _event(run_id="r2", result="success"),
    ]
    stats = mine(events, "wf")
    assert len(stats) == 1
    assert stats[0].deterministic is True
    assert stats[0].occurrences == 2


def test_mine_classifies_same_input_different_outcome_as_non_deterministic():
    events = [
        _event(run_id="r1", result="success"),
        _event(run_id="r2", result="failure"),
    ]
    stats = mine(events, "wf")
    assert len(stats) == 1
    assert stats[0].deterministic is False
    assert len(stats[0].distinct_outcomes) == 2


def test_mine_different_inputs_do_not_affect_determinism():
    events = [
        _event(input_state={"k": "a"}, result="success"),
        _event(input_state={"k": "b"}, result="failure"),
    ]
    stats = mine(events, "wf")
    assert len(stats) == 1
    assert stats[0].deterministic is True


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
    assert by_node["expensive"].deterministic is False
    assert by_node["expensive"].score == 1.0

    assert round(by_node["cheap"].cost_component, 4) == 0.2
    assert by_node["cheap"].deterministic is True
    assert round(by_node["cheap"].score, 4) == 0.1


def test_recommend_orders_worst_first_and_messages_match_bucket():
    events = [
        _event(node="cheap_stable", cost=0.0, run_id="r1", result="success"),
        _event(node="expensive_flaky", cost=0.05, run_id="r1", result="success"),
        _event(node="expensive_flaky", cost=0.05, run_id="r2", result="failure"),
    ]
    recs = recommend(score(mine(events, "wf")))
    assert recs[0]["node"] == "expensive_flaky"
    assert recs[0]["deterministic"] is False
    assert "non-deterministic and high-cost" in recs[0]["message"]
    assert recs[-1]["node"] == "cheap_stable"
    assert "healthy, no action needed" in recs[-1]["message"]


def test_run_pipeline_end_to_end_on_sample_traces():
    events = load_traces(DATA_PATH)
    report = run_pipeline(events)

    assert report["observe"]["selected_workflow"] == "cve-remediation-agent"
    assert report["profile"]["x"]["x"] >= report["profile"]["y"]["y"] - 1e-6

    by_agent = report["mine"]["by_agent"]
    assert "security-agent" in by_agent
    assert by_agent["security-agent"]["num_actions"] == 6
    assert by_agent["security-agent"]["deterministic"] == 6
    assert by_agent["security-agent"]["non_deterministic"] == 0

    assert len(report["recommend"]) == 6
    scores = [r["score"] for r in report["recommend"]]
    assert scores == sorted(scores, reverse=True)


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
