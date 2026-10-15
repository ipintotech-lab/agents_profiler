import random

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest import TraceEvent, normalize_raw_json
from profiler.data_quality import check_data_quality, estimate_missing_costs
from profiler.pipeline import run_pipeline
from rawdata_gen.__main__ import main as cli_main
from rawdata_gen.claude_code_sessions import generate_sessions


def _events(omit=(), seed=7):
    sessions = generate_sessions(2, 3, random.Random(seed), omit)
    return [TraceEvent(**e) for e in normalize_raw_json([r for s in sessions for r in s])]


def test_generated_claude_code_sessions_are_complete():
    events = _events()
    assert any(e.tool and e.tool.name == "Read" for e in events)
    dq = check_data_quality(events)
    assert dq["complete"], dq["gaps"]
    assert not dq["can_estimate_cost"]


def test_generator_is_reproducible():
    assert generate_sessions(2, 2, random.Random(1)) == generate_sessions(2, 2, random.Random(1))


def test_generator_rejects_unknown_omit():
    with pytest.raises(ValueError):
        generate_sessions(1, 1, random.Random(1), ("prompts",))


@pytest.mark.parametrize("omit, gap", [(("cost",), "cost"), (("tokens", "cost"), "tokens"), (("timing",), "llm_latency")])
def test_omitted_attributes_show_up_as_gaps(omit, gap):
    dq = check_data_quality(_events(omit))
    assert not dq["complete"]
    assert gap in {g["key"] for g in dq["gaps"]}
    assert all(g["effect"] for g in dq["gaps"])


def test_cost_can_be_estimated_only_when_tokens_exist():
    assert check_data_quality(_events(("cost",)))["can_estimate_cost"]
    assert not check_data_quality(_events(("cost", "tokens")))["can_estimate_cost"]


def test_estimate_fills_missing_cost_and_keeps_recorded_cost():
    events = _events(("cost",))
    filled, info = estimate_missing_costs(events, 3.0, 15.0)
    assert info["events"] > 0
    llm = [e for e in filled if e.model]
    assert all(e.cost > 0 for e in llm)
    e = llm[0]
    assert e.cost == pytest.approx((e.tokens.input * 3 + e.tokens.output * 15) / 1e6)
    recorded = _events()
    assert estimate_missing_costs(recorded, 100.0, 100.0)[1]["events"] == 0
    with pytest.raises(ValueError):
        estimate_missing_costs(events, -1, 1)


def test_pipeline_reports_data_quality_and_estimate():
    events = _events(("cost",))
    report = run_pipeline(events)
    assert report["data_quality"]["can_estimate_cost"]
    report = run_pipeline(events, token_prices=(3.0, 15.0))
    statuses = {g["key"]: g["status"] for g in report["data_quality"]["gaps"]}
    assert statuses["cost"] == "estimated"


def test_generate_and_analyze_endpoints_with_missing_cost():
    client = TestClient(app)
    gen = client.post("/generate", json={"format": "claude_code", "sessions": 2, "prompts_per_session": 2, "omit": ["cost"], "seed": 4})
    assert gen.status_code == 200
    data = gen.json()
    assert data["num_sessions"] == 2 and data["num_runs"] == 4 and data["omitted"] == ["cost"]
    assert all(r["num_actions"] > 0 for r in data["runs"])
    files = {"file": ("sessions.jsonl", data["jsonl"].encode())}

    report = client.post("/analyze", files=files).json()
    assert {g["key"] for g in report["data_quality"]["gaps"]} == {"cost"}

    priced = client.post("/analyze", files=files, data={"price_input_per_mtok": "3", "price_output_per_mtok": "15"})
    assert priced.status_code == 200
    assert priced.json()["data_quality"]["gaps"][0]["status"] == "estimated"

    assert client.post("/analyze", files=files, data={"price_input_per_mtok": "3"}).status_code == 400
    assert client.post("/generate", json={"format": "claude_code", "omit": ["bogus"]}).status_code in (400, 422)


def test_cli_writes_both_formats(tmp_path, capsys):
    assert cli_main(["--format", "claude-code", "--sessions", "1", "--prompts", "2", "--seed", "1", "-o", str(tmp_path / "s.jsonl")]) == 0
    assert (tmp_path / "s.jsonl").read_text().count("\n") > 10
    assert cli_main(["--agents", "2", "--actions", "4", "6", "--lines", "40", "--seed", "1", "-o", str(tmp_path / "e.jsonl")]) == 0
    assert (tmp_path / "e.jsonl").read_text().count("\n") == 40
    with pytest.raises(SystemExit):
        cli_main(["--agents", "2", "--actions", "4"])
