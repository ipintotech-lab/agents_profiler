import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest import normalize_raw_json


def _base_event(**overrides) -> dict:
    base = {
        "trace_id": "t1",
        "run_id": "r1",
        "workflow": "wf",
        "event_type": "action",
        "node": "step",
        "result": "success",
        "cost": 0.01,
        "latency_ms": 100,
    }
    base.update(overrides)
    return base


def test_normalize_passes_through_already_correct_events():
    rows = normalize_raw_json([_base_event()])
    assert rows == [_base_event()]


def test_normalize_maps_aliased_field_names():
    aliased = {
        "traceId": "t1",
        "spanId": "r1",
        "workflow_name": "wf",
        "eventType": "action",
        "name": "step",
        "duration_ms": 250,
        "output": "ok",
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "tool_name": "search",
    }
    rows = normalize_raw_json([aliased])
    assert rows[0]["trace_id"] == "t1"
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["workflow"] == "wf"
    assert rows[0]["event_type"] == "action"
    assert rows[0]["node"] == "step"
    assert rows[0]["latency_ms"] == 250
    assert rows[0]["result"] == "ok"
    assert rows[0]["tokens"] == {"input": 10, "output": 4}
    assert rows[0]["tool"] == {"name": "search"}


def test_normalize_unwraps_events_key():
    rows = normalize_raw_json({"events": [_base_event()]})
    assert rows == [_base_event()]


def test_normalize_unwraps_nested_container_keys():
    rows = normalize_raw_json({"data": {"traces": [_base_event()]}})
    assert rows == [_base_event()]


def test_normalize_otel_span_export():
    otel = {
        "resourceSpans": [
            {
                "resource": {"attributes": [{"key": "workflow", "value": {"stringValue": "wf"}}]},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "t1",
                                "spanId": "r1",
                                "name": "step",
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "1250000000",
                                "attributes": [
                                    {"key": "event_type", "value": {"stringValue": "action"}},
                                    {"key": "result", "value": {"stringValue": "ok"}},
                                ],
                            }
                        ]
                    }
                ],
            }
        ]
    }
    rows = normalize_raw_json(otel)
    assert len(rows) == 1
    row = rows[0]
    assert row["trace_id"] == "t1"
    assert row["run_id"] == "r1"
    assert row["workflow"] == "wf"
    assert row["event_type"] == "action"
    assert row["node"] == "step"
    assert row["result"] == "ok"
    assert row["latency_ms"] == pytest.approx(250.0)


def test_normalize_rejects_record_missing_required_field():
    with pytest.raises(ValueError, match="workflow"):
        normalize_raw_json([{"trace_id": "t1", "run_id": "r1", "event_type": "action", "node": "step"}])


def test_analyze_endpoint_accepts_aliased_json_upload():
    client = TestClient(app)
    aliased = [
        {
            "traceId": "t1",
            "spanId": f"r{i}",
            "workflow_name": "wf",
            "eventType": "llm_call",
            "name": "step",
            "duration_ms": 100,
            "model": "demo-llm",
            "cost_usd": 0.02,
        }
        for i in range(5)
    ]
    resp = client.post(
        "/analyze",
        files={"file": ("aliased.json", json.dumps(aliased), "application/json")},
    )
    assert resp.status_code == 200
    assert resp.json()["observe"]["num_events"] == 5


def test_analyze_endpoint_accepts_wrapped_json_upload():
    client = TestClient(app)
    wrapped = {"events": [_base_event(run_id=f"r{i}", model="demo-llm") for i in range(5)]}
    resp = client.post(
        "/analyze",
        files={"file": ("wrapped.json", json.dumps(wrapped), "application/json")},
    )
    assert resp.status_code == 200
    assert resp.json()["observe"]["num_events"] == 5


def test_analyze_endpoint_rejects_file_with_unmappable_record():
    client = TestClient(app)
    bad = [{"trace_id": "t1", "run_id": "r1", "event_type": "action", "node": "step"}]
    resp = client.post(
        "/analyze",
        files={"file": ("bad.json", json.dumps(bad), "application/json")},
    )
    assert resp.status_code == 422
