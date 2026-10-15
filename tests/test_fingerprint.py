import pytest

from ingest import TraceEvent
from profiler import fingerprint as fpm
from profiler.determinism import Classification, classify
from profiler.fingerprint import (
    ActionContext,
    FingerprintLevel,
    build_business_fingerprint,
    build_context_fingerprint,
    build_execution_fingerprint,
    calculate_group_statistics,
    fingerprint_node,
    is_compatible,
    reconstruct_contexts,
)

CTX = ActionContext(trigger="cve_analysis_completed", trigger_result="fixed_version=3.0.2")


def _event(trace="t1", run="r1", selected="upgrade", model="m1", version="v1", **overrides):
    fields = {
        "trace_id": trace, "run_id": run, "workflow": "wf", "agent": "a", "event_type": "decision", "node": "analyze_remediation",
        "decision": {"type": "choose_remediation", "selected": selected}, "model": model, "workflow_version": version,
        "input_state": {"package": "openssl", "version": "3.0.1", "severity": "HIGH"}, "result": "success",
    }
    fields.update(overrides)
    return TraceEvent(**fields)


def _ids(event, ctx=CTX):
    return (
        build_business_fingerprint(event, ctx).fingerprint_id,
        build_context_fingerprint(event, ctx).fingerprint_id,
        build_execution_fingerprint(event, ctx).fingerprint_id,
    )


def test_same_relevant_input_from_different_sessions_runs_and_times_has_one_fingerprint():
    a = _event(trace="t1", run="r1", latency_ms=10, cost=0.1, user_input="fix openssl")
    b = _event(trace="t2", run="r2", latency_ms=999, cost=0.9, user_input="another request", parent="p")
    assert _ids(a) == _ids(b)


def test_different_business_input_changes_the_business_fingerprint():
    a = _event()
    b = _event(input_state={"package": "openssl", "version": "3.0.1", "severity": "LOW"})
    assert _ids(a)[0] != _ids(b)[0]


def test_different_trigger_changes_the_business_fingerprint():
    other = ActionContext(trigger="manual_review", trigger_result=CTX.trigger_result)
    assert _ids(_event())[0] != _ids(_event(), other)[0]


def test_different_tool_result_keeps_business_but_changes_context_fingerprint():
    other = ActionContext(trigger=CTX.trigger, trigger_result="fixed_version=missing")
    (b1, c1, _), (b2, c2, _) = _ids(_event(), CTX), _ids(_event(), other)
    assert b1 == b2
    assert c1 != c2


def test_different_model_or_version_keeps_context_but_changes_execution_fingerprint():
    base = _ids(_event())
    for changed in (_event(model="m2"), _event(version="v2")):
        ids = _ids(changed)
        assert ids[:2] == base[:2]
        assert ids[2] != base[2]


def test_canonicalization_ignores_field_order_and_hash_is_stable():
    a = _event(input_state={"package": "openssl", "version": "3.0.1", "severity": "HIGH"})
    b = _event(input_state={"severity": "HIGH", "version": "3.0.1", "package": "openssl"})
    fa, fb = build_business_fingerprint(a, CTX), build_business_fingerprint(b, CTX)
    assert fa.canonical_value == fb.canonical_value
    assert fa.fingerprint_id == fb.fingerprint_id == build_business_fingerprint(a, CTX).fingerprint_id


def test_schema_version_and_level_govern_compatibility():
    fp = build_business_fingerprint(_event(), CTX)
    older = build_business_fingerprint(_event(), CTX)
    older.schema_version = fpm.FINGERPRINT_SCHEMA_VERSION - 1
    assert is_compatible(fp, build_business_fingerprint(_event(), CTX))
    assert not is_compatible(fp, older)
    assert older.fingerprint_id != fp.fingerprint_id
    assert not is_compatible(fp, build_context_fingerprint(_event(), CTX))


def test_group_statistics_dominant_output_rate():
    events = [_event(selected="upgrade")] * 997 + [_event(selected="patch")] * 3
    stats = calculate_group_statistics(events)
    assert stats["observations"] == 1000
    assert stats["unique_output_count"] == 2
    assert stats["dominant_output"] == "upgrade/success"
    assert stats["dominant_output_count"] == 997
    assert stats["dominant_output_rate"] == pytest.approx(0.997)
    assert stats["verdict"] == "differs"
    # the strict classifier still calls it non-deterministic
    assert classify(events).classification == Classification.NON_DETERMINISTIC


def test_starting_point_is_explicit_trigger_or_previous_step_or_run_start():
    first = _event(node="classify", decision={"type": "severity", "selected": "high"}, result="ok-1")
    second = _event(node="route")
    router = _event(node="llm_call", decision={"type": "next_action", "selected": "x"})
    third = _event(node="notify")
    explicit = _event(node="notify", trigger="classify")
    ctx = reconstruct_contexts([first, second, router, third, explicit])
    assert ctx[id(first)] == ActionContext("run_start", None)
    assert ctx[id(second)] == ActionContext("classify", "ok-1")
    assert ctx[id(third)].trigger == "route"  # routing LLM calls aren't starting points
    assert ctx[id(explicit)] == ActionContext("classify", "ok-1")


def test_context_explains_variance_when_tool_result_decides_the_output():
    events, contexts = [], {}
    for i in range(10):
        has_fix = i % 2 == 0
        e = _event(trace=f"t{i}", selected="upgrade" if has_fix else "patch")
        events.append(e)
        contexts[id(e)] = ActionContext(CTX.trigger, "fixed" if has_fix else "missing")
    report = fingerprint_node(events, contexts)
    assert report["levels"]["business"]["agreement"] == pytest.approx(0.5)
    assert report["levels"]["context"]["agreement"] == pytest.approx(1.0)
    assert report["variance"]["factor"] == "context"
    assert report["triggers"][0]["trigger"] == CTX.trigger
    assert report["groups"][0]["splits"]["context"]["groups"] == 2


def test_pipeline_reports_fingerprints_per_node_without_changing_classification():
    from ingest.loader import load_traces
    from profiler.pipeline import run_pipeline

    report = run_pipeline(load_traces("data/sample_traces.jsonl"))
    actions = [a for agent in report["mine"]["by_agent"].values() for a in agent["actions"]]
    assert all(a["fingerprints"]["schema_version"] == fpm.FINGERPRINT_SCHEMA_VERSION for a in actions)
    assert {a["node"]: a["fingerprints"]["triggers"][0]["trigger"] for a in actions}["classify_severity"] == "run_start"
    assert FingerprintLevel.BUSINESS.value in actions[0]["fingerprints"]["levels"]
