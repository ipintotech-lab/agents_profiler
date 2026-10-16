import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from profiler.experiment import _dataset
from profiler.finops import _overlaps, build_finops
from rawdata_gen.ground_truth import EXPECTED_FINDINGS


@pytest.fixture(scope="module")
def demo():
    events, _ = _dataset(5000, 0)
    return events, build_finops(events, runs_per_month=100_000)


def test_headline_counts_each_action_once(demo):
    _, f = demo
    counted = [o for o in f["opportunities"] if o["counted"]]
    assert len(counted) == f["overview"]["counted_opportunities"]
    for i, a in enumerate(counted):
        for b in counted[i + 1 :]:
            assert not (a["workflow"] == b["workflow"] and _overlaps(a, b)), (a["title"], b["title"])
    avoidable = f["overview"]["avoidable"]
    assert avoidable == pytest.approx(sum(o["saving"] for o in counted), abs=1e-5)
    assert 0 < avoidable < f["overview"]["spend"]
    assert avoidable <= sum(o["saving"] or 0 for o in f["opportunities"])
    assert sum(w["avoidable"] for w in f["workflows"]) == pytest.approx(avoidable, abs=1e-5)


def test_regions_are_findings_and_not_overlapping_each_other(demo):
    _, f = demo
    regions = [o for o in f["opportunities"] if o["type"] == "DETERMINISTIC_REGION" and o["title"] == "analyze_remediation"]
    assert regions and all(o["where"].startswith("severity=") for o in regions)
    if len(regions) > 1:
        assert not _overlaps(regions[0], regions[1])  # different slices remove different calls


def test_traps_are_never_ready(demo):
    _, f = demo
    traps = {(wf, t["type"], t["node"]) for wf, rows in EXPECTED_FINDINGS.items() for t in rows if not t["expect"]}
    for o in f["opportunities"]:
        if o["status"] == "ready":
            assert all((o["workflow"], o["type"], n) not in traps for n in o["nodes"]), o


def test_hidden_state_actions_are_not_safe_to_cache(demo):
    _, f = demo
    for wf in {o["workflow"] for o in f["opportunities"]}:
        found = [o for o in f["opportunities"] if o["workflow"] == wf]
        hidden = {n for o in found if o["type"] == "REVIEW_HIDDEN_STATE" for n in o["nodes"]}
        for o in found:
            if o["type"] in ("CACHE_RESULT", "CONDITIONAL_LLM_BYPASS") and hidden & set(o["nodes"]):
                assert o["risk"] == "high" and o["status"] == "investigate"


def test_projection_scales_by_runs(demo):
    _, f = demo
    p = f["projection"]
    assert p["factor"] == pytest.approx(100_000 / f["overview"]["runs"], rel=1e-4)
    assert p["avoidable"] == pytest.approx(f["overview"]["avoidable"] * p["factor"], abs=0.01)
    events, _ = demo
    assert build_finops(events[:400])["projection"] is None


def test_api_finops(demo):
    events, _ = demo
    client = TestClient(app)
    body = "\n".join(json.dumps(e.model_dump(mode="json")) for e in events[:1000])
    files = {"file": ("t.jsonl", body.encode(), "application/x-ndjson")}
    data = client.post("/finops", files=files, data={"runs_per_month": "5000"}).json()
    assert data["overview"]["calls"] == 1000 and data["projection"]["runs_per_month"] == 5000
    assert data["files"][0]["name"] == "t.jsonl"
    assert client.post("/finops", files=files, data={"runs_per_month": "0"}).status_code == 400
    assert client.post("/finops", files=files, data={"output_match": "fuzzy"}).status_code == 400
    assert client.post("/finops").status_code == 400


def test_token_mode_and_price_estimation_when_cost_is_unrecorded(demo):
    events, _ = demo
    # a trace that records tokens but no cost (e.g. a pi session)
    costless = [e.model_copy(update={"cost": 0.0}) for e in events]
    f = build_finops(costless)
    assert f["overview"]["spend"] == 0
    assert f["overview"]["tokens"] > 0
    # savings fall back to tokens, and findings are still selected/counted
    assert f["overview"]["avoidable"] == 0
    assert f["overview"]["avoidable_tokens"] > 0
    assert f["overview"]["counted_opportunities"] > 0
    assert f["top_waste"], "token savings still rank the biggest waste"
    assert all(w["avoidable_tokens"] >= 0 for w in f["workflows"])

    # per-token prices turn token savings into estimated dollars
    priced = build_finops(costless, price_input_per_mtok=3.0, price_output_per_mtok=15.0)
    o = priced["overview"]
    assert o["spend_est"] > 0
    assert 0 < o["avoidable_est"] <= o["spend_est"]
    assert any(o_.get("saving_est") for o_ in priced["opportunities"])
    # recorded cost always wins over estimation
    assert build_finops(events, price_input_per_mtok=3.0, price_output_per_mtok=15.0)["overview"]["spend"] > 0
