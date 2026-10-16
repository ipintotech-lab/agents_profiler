import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest.trace_schema import TraceEvent
from profiler import history
from profiler.behavior import build_behavior, filter_where, output_key_fn
from profiler.experiment import _dataset


def _event(i: int, node: str = "step", state: dict | None = None, selected: str = "a", result: str = "ok") -> TraceEvent:
    return TraceEvent.model_validate(
        {
            "trace_id": f"t{i}",
            "agent": "agent-a",
            "workflow": "wf",
            "run_id": f"r{i}",
            "event_type": "action",
            "node": node,
            "input_state": state or {"k": "v"},
            "decision": {"type": "choice", "selected": selected},
            "result": result,
            "cost": 0.01,
            "latency_ms": 100 + i,
            "model": "m1",
        }
    )


@pytest.fixture(scope="module")
def cve_behavior():
    events, _ = _dataset(5000, 0)
    return build_behavior(events, "cve-remediation-agent")


def _node(behavior: dict, name: str) -> dict:
    return next(n for n in behavior["nodes"] if n["node"] == name)


def test_planted_region_is_found_with_its_rule(cve_behavior):
    regions = _node(cve_behavior, "analyze_remediation")["regions"]
    assert regions, "the high/critical region should show at 5000 lines"
    for r in regions:
        assert r["field"] == "severity" and r["value"] in ("high", "critical")
        assert r["rule_field"] == "fix_available"
        assert r["rule"] == {"false": "mitigate · success", "true": "upgrade · success"}
        assert r["rule_rate"] >= 0.95 > r["rest_rate"]
        assert r["p_value"] < 0.01
        assert r["saving"] > 0


def test_pure_function_has_no_region_but_a_deciding_field(cve_behavior):
    node = _node(cve_behavior, "classify_severity")
    assert node["regions"] == []
    assert node["deciding_field"] == "severity"
    assert node["fields"][0]["importance"] == 1.0


def test_node_stats_have_percentiles_and_models(cve_behavior):
    node = _node(cve_behavior, "analyze_remediation")
    assert node["is_llm"] and node["models"][0]["name"] == "demo-llm-v1"
    assert node["cost"]["p50"] <= node["cost"]["p95"]
    assert node["latency_ms"]["p50"] <= node["latency_ms"]["p95"]
    assert 0 < node["consistency"]["rate"] <= 1
    # component has ~50 values: too many to slice
    assert "component" not in {f["field"] for f in node["fields"]}


def test_two_field_function_is_not_a_region():
    # "reject when lint fails, else follow the tests": deterministic everywhere
    events = [
        _event(i, state={"lint": lint, "tests": tests}, selected="reject" if not lint else ("approve" if tests else "reject"))
        for i, (lint, tests) in enumerate([(a, b) for a in (True, False) for b in (True, False)] * 20)
    ]
    assert build_behavior(events)["nodes"][0]["regions"] == []


def test_small_lucky_slice_is_not_a_region():
    # value "x" agrees 6/6 by chance; the rest is a coin flip -- too few calls to trust
    events = [_event(i, state={"f": "x"}, selected="a") for i in range(6)]
    events += [_event(10 + i, state={"f": "y"}, selected="ab"[i % 2]) for i in range(30)]
    assert build_behavior(events)["nodes"][0]["regions"] == []


def test_output_match_modes():
    a = _event(1, selected="Created", result="ticket 4411 created")
    b = _event(2, selected="created", result="Ticket 98  created")
    exact, decision, normalized = (output_key_fn(m) for m in ("exact", "decision", "normalized"))
    assert exact(a) != exact(b)
    assert decision(a) != decision(b)  # decisions compared as recorded
    assert normalized(a) == normalized(b)
    with pytest.raises(ValueError):
        output_key_fn("fuzzy")


def test_filter_where_matches_report_labels():
    events = [_event(1, state={"ok": True}), _event(2, state={"ok": False}), _event(3, state={"n": 1})]
    assert [e.trace_id for e in filter_where(events, "ok=true")] == ["t1"]
    assert filter_where(events, None) == events
    with pytest.raises(ValueError):
        filter_where(events, "ok")


def test_history_dedupes_and_clears(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_DB", str(tmp_path / "h.sqlite"))
    events = [_event(i) for i in range(5)]
    assert history.add(events, "a") == {"added": 5, "duplicates": 0, "total": 5}
    assert history.add(events[:3] + [_event(9)], "b") == {"added": 1, "duplicates": 3, "total": 6}
    assert [e.trace_id for e in history.load("wf")] == ["t0", "t1", "t2", "t3", "t4", "t9"]
    assert history.load("other") == []
    s = history.summary()
    assert s["total"] == 6 and s["workflows"][0]["uploads"] == 2
    assert history.clear(older_than_days=1) == 0
    assert history.clear("wf") == 6
    assert history.summary()["total"] == 0


def test_api_keeps_history_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_DB", str(tmp_path / "h.sqlite"))
    client = TestClient(app)
    body = "\n".join(json.dumps(e.model_dump(mode="json")) for e in [_event(i) for i in range(4)])
    files = {"file": ("t.jsonl", body.encode(), "application/x-ndjson")}
    report = client.post("/analyze", files=files).json()
    assert report["behavior"]["nodes"][0]["observations"] == 4
    assert "history" not in report
    assert client.get("/behavior/history").json()["total"] == 0

    report = client.post("/analyze", files=files, data={"keep_history": "true", "output_match": "decision"}).json()
    assert report["history"]["added"] == 4
    assert report["behavior"]["output_match"] == "decision"
    kept = client.get("/behavior", params={"workflow": "wf", "where": "k=v"}).json()
    assert kept["num_events"] == 4 and kept["history"]["total"] == 4
    assert client.get("/behavior", params={"where": "k"}).status_code == 400
    assert client.post("/analyze", files=files, data={"output_match": "fuzzy"}).status_code == 400
    assert client.delete("/behavior/history").json()["deleted"] == 4
