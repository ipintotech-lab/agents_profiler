from pathlib import Path

from bpop import BPOP
from ingest import build_capacity_problem_from_traces
from ingest.loader import load_traces

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_traces.jsonl"


def test_load_traces():
    events = load_traces(DATA_PATH)
    assert len(events) == 13
    assert all(e.workflow == "cve-remediation-agent" for e in events)


def test_build_and_solve_capacity_problem():
    events = load_traces(DATA_PATH)
    problem = build_capacity_problem_from_traces(events, workflow="cve-remediation-agent")

    result = BPOP.solve(problem)

    # max observed LLM calls in a single run is 5 (run3); the follower
    # should never be allocated to use more than that.
    assert result.y["y"] <= 5.0 + 1e-6
    # feasibility: the leader never allocates below what the follower uses.
    assert result.x["x"] >= result.y["y"] - 1e-6
    assert result.objective is not None
    assert "capacity-allocation" in result.explanation
