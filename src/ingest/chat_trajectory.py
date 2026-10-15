"""Convert an OpenAI-style chat trajectory -- one agent attempt at a task,
as in ``cx-cmu/agent_trajectories`` on Hugging Face -- into flat
TraceEvent-shaped dicts.

A record is ``{"id", "messages": [{"role", "content", "tool_calls",
"tool_call_id"}, ...], ...}`` with optional ``benchmark`` / ``domain`` /
``task_id`` / ``source_model`` / ``pass``. As with Claude Code transcripts
(see ``claude_code.py``):

* **LLM call** -- one per ``assistant`` message: ``event_type="decision"``,
  ``node="llm_call ← <what it reacted to>"`` and ``decision.selected`` = the
  tools it called next, or ``"respond"``.
* **Tool call** -- one per ``tool_calls`` entry, paired with its ``tool``
  message by ``tool_call_id`` (or, when that's missing, in call order).
  ``input_state`` is the parsed arguments and ``result`` is
  ``"output:<digest>"`` of the tool message, so the same task run several
  times (the dataset's ``pass`` 1-4) shows which calls repeat with the same
  input and output.

Identity: ``trace_id`` = ``run_id`` = the record ``id``, ``workflow`` = the
benchmark, ``agent`` = ``source_model`` (also the LLM calls' ``model``). The format has no
per-message tokens, cost or timing, so those stay 0.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .claude_code import bash_operation, tool_operation

# Long task prompts (e.g. needle-in-haystack contexts of ~500K chars) are
# kept only as context in the report; cut them down.
_MAX_USER_INPUT_CHARS = 2000


def is_chat_trajectory(record: Any) -> bool:
    messages = record.get("messages") if isinstance(record, dict) else None
    return (
        isinstance(messages, list)
        and bool(messages)
        and all(isinstance(m, dict) and "role" in m for m in messages)
    )


def _digest(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def _arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"arguments": raw}
        return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    return {} if raw is None else {"arguments": raw}


def chat_tool_operation(name: str, arguments: dict) -> str:
    """Like ``tool_operation``, but any tool that takes a shell ``command``
    (``execute_bash``, ``terminal``, ...) is named by the command it ran."""
    command = arguments.get("command") or arguments.get("cmd")
    if isinstance(command, str) and command.strip():
        return f"{name}: {bash_operation(command)}"
    return tool_operation(name, arguments)


def convert_chat_trajectory(record: dict, index: int = 0) -> list[dict]:
    record_id = str(record.get("id") or f"trajectory-{index}")
    model = record.get("source_model") or record.get("model")
    benchmark = record.get("benchmark") or "chat-trajectory"
    prompt = next((_text(m.get("content")) for m in record["messages"] if m.get("role") == "user"), "") or None
    if prompt and len(prompt) > _MAX_USER_INPUT_CHARS:
        prompt = prompt[:_MAX_USER_INPUT_CHARS] + "…"
    common = {
        "trace_id": record_id,
        "run_id": record_id,
        "workflow": str(benchmark),
        "agent": model or str(benchmark),
        "user_input": prompt,
    }

    events: list[dict] = []
    pending: dict[str, dict] = {}  # tool_call_id -> tool event awaiting its result
    unanswered: list[dict] = []  # same events, in call order, for results without an id
    trigger = "user_prompt"
    results_since_call: list[str] = []

    for position, message in enumerate(record["messages"]):
        role = message.get("role")
        if role == "user":
            trigger, results_since_call = "user_prompt", []
            continue
        if role == "tool":
            call = pending.pop(message.get("tool_call_id"), None) if message.get("tool_call_id") else None
            if call is None and unanswered:
                call = unanswered[0]
            if call is not None:
                # by identity: two calls can have equal contents
                unanswered = [c for c in unanswered if c is not call]
                pending = {k: c for k, c in pending.items() if c is not call}
                call["result"] = f"output:{_digest(message.get('content'))}"
                results_since_call.append(call["node"])
            continue
        if role != "assistant":
            continue

        if results_since_call:
            trigger = " + ".join(sorted(set(results_since_call)))
        results_since_call = []
        call_id = f"{record_id}:m{position}"
        tool_calls = [tc for tc in message.get("tool_calls") or [] if isinstance(tc, dict)]
        llm_call = {
            **common,
            "event_type": "decision",
            "node": f"llm_call ← {trigger.split(' + ')[0].split(':')[0]}",
            "model": model,
            "input_state": {"trigger": trigger},
            "trigger": trigger,
            "result": "tool_calls" if tool_calls else "stop",
        }
        events.append(llm_call)

        nodes = []
        for tc in tool_calls:
            function = tc.get("function") or {}
            name = function.get("name") or tc.get("name") or "unknown"
            arguments = _arguments(function.get("arguments", tc.get("arguments")))
            node = chat_tool_operation(name, arguments)
            nodes.append(node)
            tool_event = {
                **common,
                "event_type": "tool_call",
                "node": node,
                "parent": call_id,
                "trigger": trigger,
                "tool": {"name": name},
                "input_state": arguments,
                "result": "no_result",
            }
            events.append(tool_event)
            unanswered.append(tool_event)
            if tc.get("id"):
                pending[tc["id"]] = tool_event
        llm_call["decision"] = {"type": "next_action", "selected": "+".join(sorted(set(nodes))) or "respond"}
    return events


def convert_chat_trajectories(records: list) -> list[dict]:
    return [event for i, record in enumerate(records) for event in convert_chat_trajectory(record, i)]
