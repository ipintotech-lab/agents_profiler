"""Load sample agent-workflow trace rows, build a B-POP capacity-
allocation problem from them, solve it, and print the explainable
result -- the "get row data from the agents workflow and run B-POP
on it" end-to-end path."""
from pathlib import Path

from bpop import BPOP
from ingest import build_capacity_problem_from_traces
from ingest.loader import load_traces

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_traces.jsonl"

if __name__ == "__main__":
    events = load_traces(DATA_PATH)
    problem = build_capacity_problem_from_traces(events, workflow="cve-remediation-agent")
    result = BPOP.solve(problem)
    print(result.explanation)
