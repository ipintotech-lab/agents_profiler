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


def test_traps_are_never_low_risk(demo):
    _, f = demo
    traps = {(wf, t["type"], t["node"]) for wf, rows in EXPECTED_FINDINGS.items() for t in rows if not t["expect"]}
    for o in f["opportunities"]:
        if o["risk"] == "low":
            assert all((o["workflow"], o["type"], n) not in traps for n in o["nodes"]), o


def test_hidden_state_actions_are_not_safe_to_cache(demo):
    _, f = demo
    for wf in {o["workflow"] for o in f["opportunities"]}:
        found = [o for o in f["opportunities"] if o["workflow"] == wf]
        hidden = {n for o in found if o["type"] == "REVIEW_HIDDEN_STATE" for n in o["nodes"]}
        for o in found:
            if o["type"] in ("CACHE_RESULT", "CONDITIONAL_LLM_BYPASS") and hidden & set(o["nodes"]):
                assert o["risk"] == "high"


def test_nothing_is_called_ready_or_safe_without_replay(demo):
    _, f = demo
    assert f["opportunities"]
    for o in f["opportunities"]:
        assert o["status"] == "opportunity_detected" and o["validation"] == "not replayed"
    text = json.dumps({k: v for k, v in f.items() if k not in ("status_totals", "deprecated")}).lower()
    for phrase in ("ready to try", "safe to cache", '"ready"'):
        assert phrase not in text



def test_findings_group_by_a_shared_step_of_one_kind(demo):
    _, f = demo
    by_group = {}
    for o in f["opportunities"]:
        by_group.setdefault(o["group"], []).append(o)
    assert f["overview"]["groups"] == len(by_group) <= f["overview"]["opportunities"]
    for members in by_group.values():
        lead = members[0]  # the largest
        assert all(o["saving"] is None or (o["saving"] or 0) <= (lead["saving"] or 0) for o in members)
        for o in members[1:]:
            assert (o["workflow"], o["type"]) == (lead["workflow"], lead["type"])
            assert set(o["nodes"]) & set(lead["nodes"])


def test_next_steps_lead_groups_and_never_say_apply(demo):
    _, f = demo
    steps = f["next_steps"]
    assert 0 < len(steps) <= 3
    assert len({n["group"] for n in steps}) == len(steps)
    assert all(n["step"] in ("investigate", "declare", "review", "replay") for n in steps)
    if any(o["risk"] == "high" for o in f["opportunities"]):
        assert any(n["step"] == "investigate" for n in steps)
    for n in steps:
        assert ("side effects not declared" in n["risk_reasons"]) == (n["step"] == "declare") or n["step"] == "investigate"


def test_deprecated_ready_keys_alias_low_risk(demo):
    _, f = demo
    assert f["overview"]["ready_saving"] == f["overview"]["low_risk_saving"]
    assert f["projection"]["ready_saving"] == f["projection"]["low_risk_saving"]
    assert f["status_totals"]["ready"] == f["risk_totals"]["low"]
    assert f["status_totals"]["investigate"] == f["risk_totals"]["high"]


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
