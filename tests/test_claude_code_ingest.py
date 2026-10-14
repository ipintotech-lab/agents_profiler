import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest import TraceEvent, normalize_raw_json


def _assistant(msg_id, ts, content, usage=None, stop="tool_use", cwd="/home/u/proj"):
    return {
        "type": "assistant",
        "sessionId": "s1",
        "uuid": f"u-{msg_id}-{ts}",
        "timestamp": ts,
        "cwd": cwd,
        "isSidechain": False,
        "version": "2.1.0",
        "message": {
            "id": msg_id,
            "model": "claude-test",
            "content": content,
            "stop_reason": stop,
            "usage": usage or {"input_tokens": 10, "cache_read_input_tokens": 0, "output_tokens": 10},
        },
    }


def _transcript():
    return [
        {"type": "mode", "sessionId": "s1", "mode": "normal"},
        {"type": "user", "sessionId": "s1", "promptId": "p1", "timestamp": "2026-01-01T00:00:00.000Z",
         "message": {"role": "user", "content": "list files"}},
        # one API response split across two records (text block + tool_use block)
        _assistant("m1", "2026-01-01T00:00:01.000Z", [{"type": "text", "text": "ok"}]),
        _assistant("m1", "2026-01-01T00:00:02.000Z",
                   [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}}], cwd="/home/u/proj/src"),
        {"type": "user", "sessionId": "s1", "promptId": "p1", "timestamp": "2026-01-01T00:00:02.500Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "a.py"}]}},
        _assistant("m2", "2026-01-01T00:00:04.000Z", [{"type": "text", "text": "done"}],
                   usage={"input_tokens": 10, "cache_read_input_tokens": 0, "output_tokens": 30}, stop="end_turn"),
        {"type": "cost-state", "sessionId": "s1", "modelUsage": {"claude-test": {"costUSD": 1.0}}},
    ]


def test_claude_code_transcript_converts_to_llm_and_tool_events():
    rows = normalize_raw_json(_transcript())
    events = [TraceEvent.model_validate(r) for r in rows]
    llm = [e for e in events if e.event_type == "decision"]
    tools = [e for e in events if e.event_type == "tool_call"]

    assert len(llm) == 2  # m1's two records collapse into one call
    assert len(tools) == 1
    assert {e.workflow for e in events} == {"claude-code/proj"}  # mid-session cd doesn't split it
    assert {e.run_id for e in events} == {"p1"}
    assert {e.trace_id for e in events} == {"s1"}

    first, second = llm
    assert first.input_state == {"trigger": "user_prompt"}
    assert first.decision.selected == "Bash"
    assert first.latency_ms == pytest.approx(2000)
    assert second.input_state == {"trigger": "tool_result:Bash"}
    assert second.decision.selected == "respond"
    assert second.result == "end_turn"

    # cost-state total split by price-weighted tokens (output weighs 5x input)
    assert first.cost + second.cost == pytest.approx(1.0)
    assert second.cost / first.cost == pytest.approx((10 + 5 * 30) / (10 + 5 * 10))

    bash = tools[0]
    assert bash.node == "Bash"
    assert bash.tool.name == "Bash"
    assert bash.input_state == {"command": "ls"}
    assert bash.parent == "m1"
    assert bash.result.startswith("success:")
    assert bash.latency_ms == pytest.approx(500)


def test_claude_code_transcript_without_cost_state_has_zero_cost():
    rows = normalize_raw_json([r for r in _transcript() if r["type"] != "cost-state"])
    assert all(r.get("cost", 0.0) == 0.0 for r in rows)


def test_analyze_endpoint_accepts_claude_code_jsonl_upload():
    body = "\n".join(json.dumps(r) for r in _transcript())
    response = TestClient(app).post("/analyze", files={"file": ("session.jsonl", body.encode())})
    assert response.status_code == 200, response.text
    assert response.json()["observe"]["workflows"][0]["name"] == "claude-code/proj"
