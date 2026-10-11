import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest.trace_schema import TraceEvent
from rawdata_gen import generate_configured
from rawdata_gen.agents.common import to_jsonl


def test_generate_configured_one_run_per_agent_without_total_lines():
    runs = generate_configured(3, [4, 5, 7])
    assert len(runs) == 3
    assert [len(r.events) for r in runs] == [4, 5, 7]
    assert {r.agent for r in runs} == {"security-agent", "support-agent", "release-agent"}


def test_generate_configured_respects_total_lines_round_robin():
    runs = generate_configured(2, [4, 6], total_lines=25)
    total = sum(len(r.events) for r in runs)
    assert total == 25


def test_generate_configured_rejects_out_of_range_action_count():
    with pytest.raises(ValueError):
        generate_configured(1, [3])


def test_generated_events_validate_against_trace_schema():
    runs = generate_configured(3, [4, 4, 4], total_lines=12)
    jsonl = to_jsonl(runs)
    events = [TraceEvent.model_validate(json.loads(line)) for line in jsonl.splitlines()]
    assert len(events) == 12


def test_generate_endpoint_returns_valid_jsonl():
    client = TestClient(app)
    resp = client.post("/generate", json={"agent_count": 2, "actions_per_agent": [4, 6], "total_lines": 25})
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_events"] == 25
    lines = data["jsonl"].splitlines()
    assert len(lines) == 25
    json.loads(lines[0])


def test_generate_endpoint_rejects_bad_action_count():
    client = TestClient(app)
    resp = client.post("/generate", json={"agent_count": 1, "actions_per_agent": [2]})
    assert resp.status_code == 400
