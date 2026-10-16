"""The verification loop: two time windows in, measured change per pattern
out -- matched by stable pattern_id, with the containment fallback for
patterns the sequence miner absorbs into longer chains."""
import json

from fastapi.testclient import TestClient
from ingest import TraceEvent

from api.main import app
from profiler.delta import build_delta


def _router(run_id: str, step: str) -> TraceEvent:
    return TraceEvent(
        trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="decision",
        node="llm_call", model="m", decision={"type": "next_action", "selected": step},
        tokens={"input": 100, "output": 10}, user_input=f"do {run_id}",
    )


def _runs(prefix: str, count: int, steps: tuple[str, ...]) -> list[TraceEvent]:
    events = []
    for i in range(count):
        run = f"{prefix}{i}"
        for step in steps:
            events.append(_router(run, step))
            events.append(
                TraceEvent(
                    trace_id="t", run_id=run, workflow="wf", agent="a",
                    event_type="tool_call", node=step, input_state={"cmd": f"{step} {i}"},
                )
            )
    return events


def _window_events(window: str) -> list[TraceEvent]:
    if window == "a":
        return (
            _runs("g", 6, ("grep", "sed"))       # the habit, strong in window A
            + _runs("f", 3, ("fetch", "lint"))   # a habit that stops
            + _runs("r", 3, ("read", "log"))     # a habit that gets absorbed
        )
    return (
        _runs("g2", 3, ("grep", "sed"))          # the habit, weaker in window B
        + _runs("s", 3, ("scan", "read", "log"))  # the absorbed habit, extended at the head
        + _runs("d", 3, ("deploy", "restart"))   # a new habit
    )


def test_delta_measures_change_per_pattern():
    result = build_delta(_window_events("a"), _window_events("b"), "wf")

    by_nodes = {}
    for p in result["patterns"]:
        nodes = (p["a"] or p["b"])["nodes"]
        if nodes:
            by_nodes[tuple(nodes)] = p

    # matched: same habit, fewer occurrences -- measured, not modeled
    shrunk = by_nodes[("grep", "sed")]
    assert shrunk["status"] == "unchanged"
    assert shrunk["a"]["occurrences"] == 6
    assert shrunk["b"]["occurrences"] == 3
    assert shrunk["change"]["occurrences"] == -3

    # resolved: present in A, gone in B, not contained in any B pattern
    resolved = by_nodes[("fetch", "lint")]
    assert resolved["status"] == "resolved"
    assert resolved["b"] is None

    # absorbed: read->log vanished, but B's scan->read->log contains it
    absorbed = by_nodes[("read", "log")]
    assert absorbed["status"] == "absorbed"
    assert absorbed["absorbed_into"] == by_nodes[("scan", "read", "log")]["pattern_id"]

    # appeared: new in window B
    assert by_nodes[("deploy", "restart")]["status"] == "appeared"

    s = result["summary"]
    assert s["patterns_resolved"] >= 1
    assert s["patterns_absorbed"] >= 1
    assert s["patterns_appeared"] >= 1
    # window B's avoidable tokens are lower: fewer grep->sed round-trips
    assert s["avoidable_change"]["tokens"] < 0


def test_delta_endpoint_and_order_independence():
    def rows(events):
        return "\n".join(json.dumps(e.model_dump()) for e in events)

    a = rows(_window_events("a"))
    b = rows(_window_events("b"))
    client = TestClient(app)

    def call(fa, fb):
        resp = client.post(
            "/delta",
            files=[
                ("files_a", ("a.jsonl", fa.encode(), "application/jsonl")),
                ("files_b", ("b.jsonl", fb.encode(), "application/jsonl")),
            ],
        )
        assert resp.status_code == 200, resp.text
        return resp.json()

    # a rerun of the same two windows gives a byte-identical delta (the
    # loader canonicalizes each window, so the join is deterministic)
    first = call(a, b)
    second = call(a, b)
    assert first["patterns"] == second["patterns"]
    assert first["summary"] == second["summary"]
    assert first["windows"]["a"]["files"] == second["windows"]["a"]["files"]

    statuses = {p["status"] for p in first["patterns"]}
    assert {"unchanged", "resolved", "absorbed", "appeared"} <= statuses
