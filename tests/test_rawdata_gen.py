import json
import random

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest.trace_schema import TraceEvent
from rawdata_gen import (
    generate_configured,
    verify_seed_range_reproducibility,
    verify_seed_reproducibility,
)
from rawdata_gen.agents.common import to_jsonl


def test_generate_configured_one_run_per_agent_without_total_lines():
    runs = generate_configured(3, [4, 5, 7])
    assert len(runs) == 3
    assert [len(r.events) for r in runs] == [4, 5, 7]
    assert {r.agent for r in runs} == {"security-agent", "support-agent", "release-agent"}


def test_generate_configured_respects_total_lines_round_robin():
    runs = generate_configured(2, [4, 6], total_lines=40)
    total = sum(len(r.events) for r in runs)
    assert total == 40


def test_generate_configured_rejects_out_of_range_action_count():
    with pytest.raises(ValueError):
        generate_configured(1, [3])
    with pytest.raises(ValueError):
        generate_configured(1, [21])


def test_generate_configured_allows_up_to_twenty_actions():
    runs = generate_configured(1, [20])
    assert len(runs[0].events) == 20


def test_generate_configured_rejects_total_lines_not_multiple_of_twenty():
    with pytest.raises(ValueError):
        generate_configured(1, [4], total_lines=25)
    with pytest.raises(ValueError):
        generate_configured(1, [4], total_lines=10)


def test_generate_configured_is_random_across_calls():
    jsonl_a = to_jsonl(generate_configured(3, [4, 5, 7], total_lines=20))
    jsonl_b = to_jsonl(generate_configured(3, [4, 5, 7], total_lines=20))
    assert jsonl_a != jsonl_b


def test_generated_events_validate_against_trace_schema():
    runs = generate_configured(3, [4, 4, 4], total_lines=20)
    jsonl = to_jsonl(runs)
    events = [TraceEvent.model_validate(json.loads(line)) for line in jsonl.splitlines()]
    assert len(events) == 20


def test_generate_endpoint_returns_valid_jsonl():
    client = TestClient(app)
    resp = client.post("/generate", json={"agent_count": 2, "actions_per_agent": [4, 6], "total_lines": 40})
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_events"] == 40
    lines = data["jsonl"].splitlines()
    assert len(lines) == 40
    json.loads(lines[0])


def test_generate_endpoint_rejects_bad_action_count():
    client = TestClient(app)
    resp = client.post("/generate", json={"agent_count": 1, "actions_per_agent": [2]})
    assert resp.status_code == 400


def test_generate_configured_rejects_invalid_variance():
    with pytest.raises(ValueError):
        generate_configured(1, [4], variance="extreme")


def test_generate_configured_default_variance_is_mid():
    import random

    rng_a = random.Random(42)
    rng_b = random.Random(42)
    runs_mid = generate_configured(1, [4], rng=rng_a)
    runs_explicit_mid = generate_configured(1, [4], rng=rng_b, variance="mid")
    assert to_jsonl(runs_mid) == to_jsonl(runs_explicit_mid)


def test_variance_high_produces_wider_spread_than_low():
    import random
    import statistics

    def sample_costs(variance: str, n: int = 200) -> list[float]:
        rng = random.Random(0)
        costs = []
        for _ in range(n):
            run = generate_configured(1, [4], rng=rng, variance=variance)[0]
            costs.extend(
                event["cost"] for event in run.events if event.get("cost") is not None
            )
        return costs

    low_costs = sample_costs("low")
    high_costs = sample_costs("high")
    assert statistics.pstdev(high_costs) > statistics.pstdev(low_costs)


def test_generate_endpoint_rejects_bad_variance():
    client = TestClient(app)
    resp = client.post(
        "/generate", json={"agent_count": 1, "actions_per_agent": [4], "variance": "extreme"}
    )
    assert resp.status_code == 400


def test_generate_endpoint_same_seed_is_reproducible():
    client = TestClient(app)
    body = {"agent_count": 2, "actions_per_agent": [4, 6], "total_lines": 40, "seed": 12345}
    resp_a = client.post("/generate", json=body)
    resp_b = client.post("/generate", json=body)
    assert resp_a.status_code == resp_b.status_code == 200
    data_a, data_b = resp_a.json(), resp_b.json()
    assert data_a["jsonl"] == data_b["jsonl"]
    assert data_a["seed"] == data_b["seed"] == 12345


def test_generate_endpoint_different_seeds_differ():
    client = TestClient(app)
    base = {"agent_count": 2, "actions_per_agent": [4, 6], "total_lines": 40}
    resp_a = client.post("/generate", json={**base, "seed": 1})
    resp_b = client.post("/generate", json={**base, "seed": 2})
    assert resp_a.json()["jsonl"] != resp_b.json()["jsonl"]


def test_generate_endpoint_without_seed_still_reports_a_resolved_seed():
    client = TestClient(app)
    resp = client.post("/generate", json={"agent_count": 1, "actions_per_agent": [4]})
    assert resp.status_code == 200
    assert isinstance(resp.json()["seed"], int)


def test_generate_endpoint_response_has_dataset_identity_fields():
    client = TestClient(app)
    resp = client.post("/generate", json={"agent_count": 1, "actions_per_agent": [4], "seed": 7})
    data = resp.json()
    assert isinstance(data["dataset_id"], str) and data["dataset_id"]
    assert data.get("generated_at")
    assert data["seed"] == 7


def test_generate_endpoint_dataset_id_differs_across_calls_even_with_same_seed():
    client = TestClient(app)
    body = {"agent_count": 1, "actions_per_agent": [4], "seed": 7}
    resp_a = client.post("/generate", json=body)
    resp_b = client.post("/generate", json=body)
    assert resp_a.json()["dataset_id"] != resp_b.json()["dataset_id"]


def test_analyze_endpoint_accepts_demand_model_override():
    client = TestClient(app)
    gen = client.post("/generate", json={"agent_count": 1, "actions_per_agent": [10], "seed": 1}).json()
    resp = client.post(
        "/analyze",
        files={"file": ("generated.jsonl", gen["jsonl"], "application/x-ndjson")},
        data={"demand_model": "max"},
    )
    assert resp.status_code == 200
    assert resp.json()["formula"]["defaults"]["demand_model"] == "max"


def test_analyze_endpoint_rejects_invalid_demand_model():
    client = TestClient(app)
    gen = client.post("/generate", json={"agent_count": 1, "actions_per_agent": [10], "seed": 1}).json()
    resp = client.post(
        "/analyze",
        files={"file": ("generated.jsonl", gen["jsonl"], "application/x-ndjson")},
        data={"demand_model": "p999"},
    )
    assert resp.status_code == 400


def test_verify_seed_reproducibility_true_for_matching_seed():
    result = verify_seed_reproducibility(2, [4, 6], 40, "mid", seed=12345)
    assert result["seed"] == 12345
    assert result["reproducible"] is True
    assert result["num_events"] == 40
    assert isinstance(result["jsonl_sha256"], str) and len(result["jsonl_sha256"]) == 64


def test_verify_seed_reproducibility_fingerprint_differs_across_seeds():
    result_a = verify_seed_reproducibility(1, [4], None, "mid", seed=1)
    result_b = verify_seed_reproducibility(1, [4], None, "mid", seed=2)
    assert result_a["jsonl_sha256"] != result_b["jsonl_sha256"]


def test_validate_seed_endpoint_confirms_reproducibility():
    client = TestClient(app)
    resp = client.post(
        "/validate_seed",
        json={"agent_count": 2, "actions_per_agent": [4, 6], "total_lines": 40, "seed": 99},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["seed"] == 99
    assert data["reproducible"] is True


def test_validate_seed_endpoint_resolves_a_seed_when_none_given():
    client = TestClient(app)
    resp = client.post("/validate_seed", json={"agent_count": 1, "actions_per_agent": [4]})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["seed"], int)
    assert data["reproducible"] is True


def test_validate_seed_endpoint_rejects_bad_action_count():
    client = TestClient(app)
    resp = client.post("/validate_seed", json={"agent_count": 1, "actions_per_agent": [2], "seed": 1})
    assert resp.status_code == 400


def test_verify_seed_range_reproducibility_covers_every_seed_in_range():
    result = verify_seed_range_reproducibility(1, [4], None, "mid", seed_start=1, seed_end=5)
    assert result["seed_start"] == 1
    assert result["seed_end"] == 5
    assert result["count"] == 5
    assert result["all_reproducible"] is True
    assert result["failures"] == []
    assert [r["seed"] for r in result["results"]] == [1, 2, 3, 4, 5]


def test_verify_seed_range_reproducibility_rejects_inverted_range():
    with pytest.raises(ValueError):
        verify_seed_range_reproducibility(1, [4], None, "mid", seed_start=10, seed_end=5)


def test_validate_seed_range_endpoint_default_range_is_1_to_30():
    client = TestClient(app)
    resp = client.post("/validate_seed_range", json={"agent_count": 1, "actions_per_agent": [4]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["seed_start"] == 1
    assert data["seed_end"] == 30
    assert data["count"] == 30
    assert data["all_reproducible"] is True


def test_validate_seed_range_endpoint_accepts_custom_range():
    client = TestClient(app)
    resp = client.post(
        "/validate_seed_range",
        json={"agent_count": 1, "actions_per_agent": [4], "seed_start": 100, "seed_end": 102},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert [r["seed"] for r in data["results"]] == [100, 101, 102]


def test_validate_seed_range_endpoint_rejects_bad_action_count():
    client = TestClient(app)
    resp = client.post("/validate_seed_range", json={"agent_count": 1, "actions_per_agent": [2]})
    assert resp.status_code == 400


def test_all_six_agents_generate_their_planned_nodes():
    from rawdata_gen.generate_traces import AGENT_GENERATORS
    from rawdata_gen.ground_truth import EXPECTED_FINDINGS

    runs = generate_configured(len(AGENT_GENERATORS), [8] * len(AGENT_GENERATORS), 2000, rng=random.Random(3))
    nodes: dict[str, set[str]] = {}
    for run in runs:
        for e in run.events:
            nodes.setdefault(e["workflow"], set()).add(e["node"])
    assert set(nodes) == set(EXPECTED_FINDINGS)
    for workflow, findings in EXPECTED_FINDINGS.items():
        for f in findings:
            for node in f.get("steps", [f.get("node")]):
                assert node in nodes[workflow], (workflow, node)


def test_every_generated_event_carries_the_request_that_started_its_run():
    from rawdata_gen.generate_traces import AGENT_GENERATORS

    for gen in AGENT_GENERATORS:
        for run in gen(random.Random(7)):
            requests = {e.get("user_input") for e in run.events}
            assert len(requests) == 1 and None not in requests, (run.workflow, requests)
