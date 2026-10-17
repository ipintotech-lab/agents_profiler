import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest import TraceEvent
from ingest.loader import load_traces
from profiler import asp
from profiler.workflow import compile_workflow, cycles, evidence_lost, precedence

CONFLICT_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "order_conflicts_sample.jsonl"
needs_clingo = pytest.mark.skipif(not asp.available(), reason="clingo not installed")


def _run(run_id, steps, **inputs):
    return [
        TraceEvent(trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="tool_call", node=n,
                   input_state=inputs.get(n, {}), result="ok")
        for n in steps
    ]


def _router(run_id):
    return TraceEvent(trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="decision", node="llm_call",
                      decision={"type": "next_action", "selected": "x"})


def test_mixed_tasks_are_grouped_so_their_orders_do_not_conflict():
    # "status" opens one task and closes the other: mixed, that's a conflict
    events = []
    for i in range(6):
        events += _run(f"fix{i}", ["status", "test", "read", "edit"])
        events += _run(f"doc{i}", ["read", "glob", "edit_md", "status"])
    result = compile_workflow(events, "wf")
    assert result["ungrouped"]["num_conflicts"] > 0
    assert result["groups_found"] == 2 and result["num_conflicts"] == 0
    orders = sorted(g["order"] for g in result["groups"])
    assert orders == [["read", "glob", "edit_md", "status"], ["status", "test", "read", "edit"]]
    for g in result["groups"]:
        assert g["holdout"]["match_rate"] == 1.0 and g["holdout"]["pair_agreement"] == 1.0


def test_next_step_llm_calls_are_not_workflow_steps():
    events = []
    for i in range(5):
        for e in _run(f"r{i}", ["a", "b"]):
            events += [_router(f"r{i}"), e]
    (group,) = compile_workflow(events, "wf")["groups"]
    assert group["order"] == ["a", "b"]


def test_parameters_split_constant_from_variable_and_check_held_out_runs():
    events = []
    for i in range(10):
        events += _run(f"r{i}", ["fetch", "push"], fetch={"queue": "support", "ticket": f"T{i}"}, push={"remote": "origin"})
    (group,) = compile_workflow(events, "wf")["groups"]
    assert group["parameters"]["fetch"] == {"constant": {"queue": "support"}, "variable": {"ticket": 8}}
    assert group["parameters"]["push"] == {"constant": {"remote": "origin"}, "variable": {}}
    assert group["scriptable"] is False
    h = group["holdout"]
    assert (h["train_runs"], h["holdout_runs"]) == (8, 2)
    assert h["constant_params_checked"] == 4 and h["constant_param_accuracy"] == 1.0


def test_either_or_steps_are_branches_and_optional_for_matching():
    events = []
    for i in range(10):
        events += _run(f"r{i}", ["check", "diagnose", "page_owner" if i % 2 else "rerun", "note"])
    (group,) = compile_workflow(events, "wf")["groups"]
    assert group["branches"] == [["page_owner", "rerun"]]
    assert set(group["required_steps"]) == {"check", "diagnose", "note"}
    assert group["holdout"]["match_rate"] == 1.0


def test_small_groups_are_left_out():
    events = _run("r1", ["a", "b"]) + _run("r2", ["a", "b"])
    result = compile_workflow(events, "wf")
    assert result["groups"] == [] and result["runs_in_small_groups"] == 2


def test_unknown_resolver_is_rejected():
    with pytest.raises(ValueError):
        compile_workflow([], "wf", "magic")


def test_sample_has_a_cycle_the_simple_order_cannot_settle():
    events = load_traces(CONFLICT_SAMPLE)
    (group,) = compile_workflow(events, "ticket-intake-agent")["groups"]
    assert group["cycles"] and group["num_conflicts"] == 3
    assert group["chosen"] == "simple" and "asp" not in group


@needs_clingo
def test_asp_never_contradicts_more_evidence_than_the_simple_order():
    events = load_traces(CONFLICT_SAMPLE)
    (group,) = compile_workflow(events, "ticket-intake-agent", "asp")["groups"]
    d = group["diff"]
    assert group["asp"]["status"] == "optimal"
    assert d["evidence_lost_asp"] < d["evidence_lost_simple"]
    assert group["chosen"] == "asp" and group["order"] == group["asp"]["order"]
    assert {"holdout_match_simple", "holdout_match_asp", "holdout_pairs_simple", "holdout_pairs_asp"} <= set(d)


@needs_clingo
def test_asp_finds_the_minimum_on_a_cyclic_triangle():
    prec = {("a", "b"): 60, ("b", "a"): 40, ("b", "c"): 70, ("c", "b"): 30, ("c", "a"): 55, ("a", "c"): 45}
    assert cycles(prec, ["a", "b", "c"]) == [["a", "b", "c"]]
    solved = asp.resolve_order(["a", "b", "c"], prec)
    best = min(evidence_lost(list(p), prec) for p in __import__("itertools").permutations("abc"))
    assert solved["status"] == "optimal" and solved["evidence_lost"] == best == evidence_lost(solved["order"], prec)


@needs_clingo
def test_asp_agrees_with_the_simple_order_when_there_is_no_conflict():
    events = [e for i in range(6) for e in _run(f"r{i}", ["a", "b", "c"])]
    (group,) = compile_workflow(events, "wf", "asp")["groups"]
    assert group["diff"]["same_order"] and group["chosen"] == "simple"


def test_asp_reports_unavailable_without_clingo(monkeypatch):
    monkeypatch.setattr(asp, "clingo", None)
    events = load_traces(CONFLICT_SAMPLE)
    result = compile_workflow(events, "ticket-intake-agent", "asp")
    (group,) = result["groups"]
    assert result["asp"]["available"] is False
    assert group["asp"]["status"] == "unavailable" and group["chosen"] == "simple" and "diff" not in group


def test_asp_skips_workflows_over_the_step_limit(monkeypatch):
    monkeypatch.setattr(asp, "MAX_STEPS", 2)
    assert asp.resolve_order(["a", "b", "c"], precedence([]))["status"] == ("skipped" if asp.available() else "unavailable")


def test_api_analyze_reports_compiled_workflows_and_validates_resolver():
    client = TestClient(app)
    files = {"file": ("s.jsonl", CONFLICT_SAMPLE.read_bytes(), "application/x-ndjson")}
    data = client.post("/analyze", files=files, data={"order_resolver": "asp"}).json()
    wc = data["workflow_compile"]
    assert wc["order_resolver"] == "asp" and wc["groups"][0]["num_conflicts"] == 3
    assert client.post("/analyze", files=files, data={"order_resolver": "magic"}).status_code == 400
    assert "available" in client.get("/health").json()["asp"]
    assert json.loads(json.dumps(wc)) == wc
