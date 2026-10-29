import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest import TraceEvent
from profiler.context import AnalysisContext, WorkflowContext, parse_context
from profiler.finops import assess, build_finops
from profiler.mine import mine, score
from profiler.opportunities import find_opportunities
from profiler.patterns import find_repetition


def _tool(run_id, node="get_owner", **declared):
    return TraceEvent(
        trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="tool_call", node=node,
        input_state={"region": "emea"}, result="rep:1", tool={"name": node}, workflow_version="v1",
        cost=0.01, **declared,
    )


def _cache(events, context=None):
    found = find_opportunities(events, "wf", find_repetition(events, "wf"), score(mine(events, "wf")), context)
    return [o for o in found if o["type"] == "CACHE_RESULT"]


def test_parse_rejects_bad_input_with_a_reason():
    assert parse_context("") is None
    with pytest.raises(ValueError, match="could not parse"):
        parse_context("{")
    with pytest.raises(ValueError, match="failure_impact"):
        parse_context(json.dumps({"workflows": {"*": {"failure_impact": "huge"}}}))
    with pytest.raises(ValueError, match="both"):
        parse_context(json.dumps({"workflows": {"*": {"side_effect_operations": ["x"], "read_only_operations": ["x"]}}}))
    context = parse_context(json.dumps({"workflows": {"*": {"failure_impact": "low"}, "wf": {"failure_impact": "high", "model": "GLM-4.6"}}}))
    assert (context.source, context.status) == ("user_declared", "unverified")
    assert (context.for_workflow("wf").failure_impact, context.for_workflow("wf").model) == ("high", "GLM-4.6")
    assert context.for_workflow("other").failure_impact == "low"


def test_read_only_answer_fills_an_undeclared_trace_but_never_overrides_it():
    events = [_tool(f"r{i}") for i in range(12)]
    read_only = WorkflowContext(read_only_operations=["get_owner"])
    (cache,) = _cache(events, read_only)
    assert cache["side_effects"]["signal"] == "user_declared"
    risk, reasons = assess(cache)
    assert risk == "low" and "you said it only reads (not verified)" in reasons
    # the trace says it writes: the user's read-only answer doesn't count
    assert _cache([_tool(f"r{i}", has_side_effects=True) for i in range(12)], read_only) == []


def test_changes_something_answer_wins_over_the_trace():
    events = [_tool(f"r{i}", has_side_effects=False) for i in range(12)]
    assert len(_cache(events)) == 1
    assert _cache(events, WorkflowContext(side_effect_operations=["get_owner"])) == []


def test_no_context_and_empty_context_change_nothing():
    events = [_tool(f"r{i}", has_side_effects=False) for i in range(12)]
    plain = build_finops(events)
    empty = build_finops(events, context=AnalysisContext())
    assert empty["context"]["hidden_by_context"] == {} and plain["context"] is None
    assert [(o["type"], o["risk"]) for o in empty["opportunities"]] == [(o["type"], o["risk"]) for o in plain["opportunities"]]


def test_high_impact_raises_risk_and_asks_for_sign_off():
    events = [_tool(f"r{i}", has_side_effects=False) for i in range(12)]
    context = AnalysisContext(workflows={"wf": WorkflowContext(failure_impact="high")})
    (cache,) = [o for o in build_finops(events, context=context)["opportunities"] if o["type"] == "CACHE_RESULT"]
    assert cache["risk"] == "medium" and "a wrong decision is high-impact (your answer)" in cache["risk_reasons"]
    assert cache["requires_approval"] is True


def test_disallowed_changes_are_hidden_and_counted():
    events = [_tool(f"r{i}", has_side_effects=False) for i in range(12)]
    context = AnalysisContext(workflows={"*": WorkflowContext(allowed_optimizations=["DETERMINISTIC_REGION"])})
    result = build_finops(events, context=context)
    assert not any(o["type"] == "CACHE_RESULT" for o in result["opportunities"])
    assert result["context"]["hidden_by_context"] == {"wf": {"CACHE_RESULT": 1}}
    assert result["overview"]["avoidable"] == 0


def test_endpoints_accept_context_and_reject_bad_json():
    client = TestClient(app)
    lines = "\n".join(_tool(f"r{i}", has_side_effects=False).model_dump_json() for i in range(12))
    files = {"file": ("t.jsonl", lines.encode(), "application/json")}
    bad = client.post("/finops", files=files, data={"context": "{"})
    assert bad.status_code == 400 and "could not parse" in bad.json()["detail"]
    context = json.dumps({"workflows": {"*": {"failure_impact": "high"}}})
    ok = client.post("/finops", files=files, data={"context": context})
    assert ok.status_code == 200 and ok.json()["context"]["status"] == "unverified"
    llm = TraceEvent(trace_id="t", run_id="r0", workflow="wf", agent="a", event_type="decision", node="classify",
                     decision={"type": "route", "selected": "a"}, model="m1", cost=0.02)
    files = {"file": ("t.jsonl", f"{lines}\n{llm.model_dump_json()}".encode(), "application/json")}
    report = client.post("/analyze", files=files, data={"context": context}).json()
    assert report["context"]["applied"]["failure_impact"] == "high"
