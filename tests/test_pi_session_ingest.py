import pytest

from ingest import TraceEvent, normalize_raw_json
from ingest.pi_session import convert_pi_session, is_pi_session


def _user(rid, ts, text, parent=None):
    return {
        "type": "message",
        "id": rid,
        "parentId": parent,
        "timestamp": ts,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _assistant(rid, parent, ts, content, usage=None, stop="toolUse", model="test-model"):
    return {
        "type": "message",
        "id": rid,
        "parentId": parent,
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": content,
            "model": model,
            "stopReason": stop,
            "usage": usage or {},
        },
    }


def _tool_result(rid, parent, ts, call_id, tool, text, error=False):
    return {
        "type": "message",
        "id": rid,
        "parentId": parent,
        "timestamp": ts,
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": tool,
            "isError": error,
            "content": [{"type": "text", "text": text}],
        },
    }


def _transcript():
    return [
        {"type": "session", "version": 3, "id": "s1", "timestamp": "2026-01-01T00:00:00.000Z", "cwd": "/home/u/proj"},
        {"type": "model_change", "id": "mc1", "parentId": None, "timestamp": "2026-01-01T00:00:00.100Z",
         "provider": "litellm", "modelId": "test-model"},
        _user("u1", "2026-01-01T00:00:00.200Z", "list files"),
        _assistant(
            "a1", "u1", "2026-01-01T00:00:01.000Z",
            [{"type": "toolCall", "id": "t1", "name": "bash",
              "arguments": {"command": "ls", "description": "list files"}}],
            usage={"input": 100, "output": 10, "cacheRead": 50, "cost": {"total": 0.5}},
        ),
        _tool_result("r1", "a1", "2026-01-01T00:00:02.000Z", "t1", "bash", "a.py\nb.py"),
        _assistant("a2", "r1", "2026-01-01T00:00:03.000Z", [{"type": "text", "text": "done"}],
                   usage={"input": 200, "output": 5, "cost": {"total": 0.2}}, stop="stop"),
    ]


def test_pi_session_is_detected():
    assert is_pi_session(_transcript())
    # headerless file of message records still detected
    assert is_pi_session(_transcript()[1:])
    assert not is_pi_session([{"type": "assistant", "sessionId": "s", "message": {}}])  # claude code
    assert not is_pi_session([])


def test_pi_session_converts_to_llm_and_tool_events():
    rows = normalize_raw_json(_transcript())
    events = [TraceEvent.model_validate(r) for r in rows]
    llm = [e for e in events if e.event_type == "decision"]
    tools = [e for e in events if e.event_type == "tool_call"]

    assert len(llm) == 2
    assert len(tools) == 1
    assert {e.workflow for e in events} == {"pi/proj"}
    assert {e.trace_id for e in events} == {"s1"}
    assert {e.run_id for e in events} == {"u1"}  # the user prompt starts the run
    assert {e.agent for e in events} == {"pi"}
    assert {e.user_input for e in events} == {"list files"}

    first, second = llm
    assert first.node == "llm_call ← user_prompt"
    assert first.model == "test-model"
    assert first.tokens.input == 150  # input + cacheRead
    assert first.tokens.output == 10
    assert first.cost == 0.5  # per-message cost from usage.cost.total
    assert first.latency_ms == 800.0  # from the record just before the reply
    assert first.decision.selected == "Bash: ls"

    assert second.node == "llm_call ← bash"  # reacted to the bash result
    assert second.decision.selected == "respond"
    assert second.cost == 0.2

    (tool,) = tools
    assert tool.node == "Bash: ls"
    assert tool.tool.name == "bash"
    assert tool.input_state == {"command": "ls"}  # cosmetic "description" dropped
    assert tool.result.startswith("success:")
    assert tool.latency_ms == 1000.0  # toolCall -> toolResult
    assert tool.parent == "a1"
    assert tool.trigger == "user_prompt"

    # LLM-call inputs carry a digest of what they reacted to, so calls
    # answering different prompts aren't lumped together as one input
    assert first.input_state["trigger"] == "user_prompt"
    assert first.input_state["reaction"]
    assert second.input_state["trigger"] == "bash"
    assert second.input_state["reaction"] != first.input_state["reaction"]


def test_pi_session_error_tool_result_marks_error():
    transcript = _transcript()
    transcript[4] = _tool_result("r1", "a1", "2026-01-01T00:00:02.000Z", "t1", "bash", "boom", error=True)
    transcript[5] = _assistant("a2", "r1", "2026-01-01T00:00:03.000Z", [{"type": "text", "text": "oops"}], stop="stop")
    events = [TraceEvent.model_validate(r) for r in normalize_raw_json(transcript)]
    tools = [e for e in events if e.event_type == "tool_call"]
    (tool,) = tools
    assert tool.result.startswith("error:")
    second_llm = [e for e in events if e.event_type == "decision"][1]
    assert second_llm.node == "llm_call ← bash"  # (error) suffix stripped from the node name


def test_pi_session_headerless_uses_first_message_id():
    rows = convert_pi_session(_transcript()[1:])  # drop the session header
    events = [TraceEvent.model_validate(r) for r in rows]
    assert {e.trace_id for e in events} == {"u1"}  # first message record's id
    assert {e.workflow for e in events} == {"pi/unknown"}


def test_pi_session_ignores_non_agent_records():
    transcript = _transcript() + [
        {"type": "custom_message", "id": "c1", "parentId": "a2", "timestamp": "2026-01-01T00:00:04.000Z",
         "customType": "notify", "content": "done", "display": True},
        {"type": "message", "id": "b1", "parentId": "a2", "timestamp": "2026-01-01T00:00:05.000Z",
         "message": {"role": "bashExecution", "command": "pi update", "output": "ok", "exitCode": 0}},
    ]
    events = [TraceEvent.model_validate(r) for r in normalize_raw_json(transcript)]
    assert len(events) == 3  # unchanged: 2 llm calls + 1 tool call
