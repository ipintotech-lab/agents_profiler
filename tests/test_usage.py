import json

from fastapi.testclient import TestClient

from api.main import app
from ingest.usage import (
    detect_kind,
    normalize_activations,
    normalize_usage,
    parse_records,
)
from profiler.usage import activations_report, usage_report


def _usage(rid, workload, inp=10, out=10, cache_read=0, cache_create=0):
    return {"record_id": rid, "workload_id": workload, "input_tokens": inp, "output_tokens": out,
            "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_create}


USAGE = [_usage(f"r{i}", "w1" if i < 3 else f"w{i}", cache_read=80) for i in range(6)] + [_usage("big", "w1", cache_read=100_000)]
ACTIVATIONS = [{"event_id": f"e{i}", "agent_id": "a1" if i < 4 else f"a{i}", "trigger_id": "t1"} for i in range(6)]


def test_detects_usage_and_activation_exports_but_not_trace_events():
    assert detect_kind(USAGE) == "usage"
    assert detect_kind(ACTIVATIONS) == "activations"
    assert detect_kind([{**USAGE[0], "node": "x"}]) is None
    assert detect_kind([{"run_id": "r", "workflow": "w", "node": "n"}]) is None


def test_parses_csv_as_well_as_json_lines():
    csv_text = "record_id,workload_id,input_tokens,output_tokens\nr1,w1,5,7\n"
    rows = parse_records(csv_text)
    assert detect_kind(rows) == "usage"
    assert normalize_usage(rows)[0] == _usage("r1", "w1", 5, 7)
    assert parse_records("\n".join(json.dumps(r) for r in USAGE)) == USAGE


def test_usage_report_concentration_cache_share_and_outliers():
    r = usage_report(normalize_usage(USAGE))
    assert r["records"] == 7 and r["workloads"] == 4
    assert r["by_workload"][0]["workload_id"] == "w1"
    assert r["concentration"]["top1_share"] > 0.99
    assert r["cache_read_share"] > 0.99
    assert [o["record_id"] for o in r["outliers"]] == ["big"]
    assert r["totals"]["all"] == r["totals"]["non_cache_read"] + r["totals"]["cache_read_input_tokens"]


def test_activations_report_counts_repeated_pairs_and_fan_out():
    r = activations_report(normalize_activations(ACTIVATIONS))
    assert r["repeated_pairs"] == 1 and r["activations_in_repeated_pairs"] == 4
    assert r["top_pairs"][0] == {"agent_id": "a1", "trigger_id": "t1", "count": 4}
    assert r["fan_out"][0] == {"trigger_id": "t1", "agents": 3}


def test_api_finops_returns_aggregate_report_and_analyze_points_to_it():
    client = TestClient(app)
    files = [
        ("file", ("usage.jsonl", "\n".join(json.dumps(r) for r in USAGE).encode(), "application/x-ndjson")),
        ("file", ("act.csv", b"event_id,agent_id,trigger_id\ne1,a1,t1\ne2,a1,t1\n", "text/csv")),
    ]
    data = client.post("/finops", files=files).json()
    assert data["kind"] == "aggregate"
    assert data["usage"]["records"] == 7 and data["activations"]["records"] == 2
    assert [f["kind"] for f in data["files"]] == ["usage", "activations"]
    res = client.post("/analyze", files=[files[0]])
    assert res.status_code == 422 and "FinOps tab" in res.json()["detail"]


def test_negative_token_counter_is_rejected():
    client = TestClient(app)
    body = json.dumps([_usage("r1", "w1", inp=-1)]).encode()
    res = client.post("/finops", files={"file": ("u.json", body, "application/json")})
    assert res.status_code == 422
