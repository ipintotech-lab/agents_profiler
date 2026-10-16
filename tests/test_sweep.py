from fastapi.testclient import TestClient

from api.main import app
from profiler.sweep import TOTAL_LINES_STEPS, _build_cases, run_sweep
from rawdata_gen.generate_traces import AGENT_GENERATORS


def test_build_cases_covers_every_option_of_every_dimension():
    cases = _build_cases()
    groups = {c["group"] for c in cases}
    assert groups == {"baseline", "agent_count", "variance", "demand_model", "total_lines", "formula_override"}

    agent_counts = {c["config"]["agent_count"] for c in cases if c["group"] == "agent_count"}
    assert agent_counts == set(range(1, len(AGENT_GENERATORS) + 1))

    variances = {c["config"]["variance"] for c in cases if c["group"] == "variance"}
    assert variances == {"low", "mid", "high"}

    demand_models = {c["config"]["demand_model"] for c in cases if c["group"] == "demand_model"}
    assert demand_models == {"p50", "p75", "p90", "p95", "p99", "max"}

    total_lines = {c["config"]["total_lines"] for c in cases if c["group"] == "total_lines"}
    assert total_lines == set(TOTAL_LINES_STEPS)


def test_run_sweep_all_cases_pass_and_report_timing():
    result = run_sweep(seed=1)
    assert result["num_cases"] == len(result["results"])
    assert result["all_passed"], [r for r in result["results"] if not r["passed"]]
    for row in result["results"]:
        assert row["elapsed_ms"] >= 0
        assert row["num_events"] > 0
        # every case analyzes one of the mock agents, so it is graded against what it was built to contain
        assert 0 < row["planted_found"] <= row["planted_total"]
        assert row["false_alarms"] == 0
        assert row["regions"] >= 0 and row["region_found"] in (True, False)
    # the planted region needs ~15 calls per severity: present in the largest trace
    largest = max(result["results"], key=lambda r: r["num_events"])
    assert largest["region_found"], largest


def test_run_sweep_endpoint_returns_sweep_report():
    client = TestClient(app)
    resp = client.post("/run_sweep", params={"seed": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["all_passed"]
    assert body["num_cases"] == len(body["results"])
