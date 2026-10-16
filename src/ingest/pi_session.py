"""Convert a pi session transcript (the JSONL files under
``~/.pi/agent/sessions/<project>/<timestamp>_<sessionId>.jsonl``, written
by the `pi <https://github.com/nicepkg/pi>`_ coding-agent harness) into
flat TraceEvent-shaped dicts.

A transcript is a ``sessionId`` header record followed by a linked list of
records (``id`` / ``parentId`` / ``timestamp``): ``message`` records
carrying the actual turns, plus metadata records (``model_change``,
``thinking_level_change``, ``custom_message``, ...) which are ignored.
A ``message`` record's payload (the ``message`` key) is one of:

* ``role: "user"`` -- a user prompt. Starts a run: its record id becomes
  ``run_id`` and its text becomes ``user_input``.
* ``role: "assistant"`` -- one API response. One record per response (pi
  does not split responses across records), with ``model``, ``usage`` and
  ``stopReason``. Emitted as ``event_type="decision"`` with
  ``node="llm_call ← <trigger>"`` (what the model was reacting to: the user
  prompt or a tool's result) and ``decision.selected`` = the tools it chose
  to call next, or ``"respond"``. Tokens come from ``usage`` and cost from
  ``usage.cost.total`` (real per-message USD -- unlike Claude Code
  transcripts, no session-level splitting is needed).
* ``role: "toolResult"`` -- the result of one tool call, paired with the
  ``toolCall`` block by ``toolCallId``.
* ``role: "bashExecution"`` -- user-initiated shell commands in the pi UI,
  not agent actions; ignored.

Tool calls are the ``toolCall`` content blocks of assistant messages:
``node`` is the tool plus the operation it performed (reusing
``claude_code.tool_operation``: ``bash: git status``, ``read: .py``),
``input_state`` is the call's arguments, and ``result`` is
``"success:<digest>"`` / ``"error:<digest>"`` of the tool result -- so
determinism means "same input -> same output", which is what a cache
would need.

Identity: ``trace_id`` = the session id from the header record,
``workflow`` = ``pi/<project dir>`` (the session's ``cwd``),
``agent`` = ``"pi"``, ``run_id`` = the id of the user-prompt record that
started the turn (the session id before the first prompt).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from .claude_code import tool_operation

# Tool-name spellings to normalize before deriving the node name, so pi's
# lowercase ``bash`` reuses the Claude Code ``Bash: <operation>`` naming.
_NODE_TOOL_NAMES = {"bash": "Bash"}

# Tool arguments that are only a human-readable label and don't change what
# the tool does -- dropped so the same command counts as the same input.
_COSMETIC_TOOL_ARGS = {"bash": ("description",), "Bash": ("description", "timeout")}


def is_pi_session(rows: Any) -> bool:
    """True for a list of parsed pi session records: a ``session`` header
    record, or (for headerless files) ``message`` records in pi's shape --
    a ``message`` payload with a ``role`` inside a record with a
    ``parentId`` chain."""
    if not isinstance(rows, list) or not rows:
        return False
    return any(
        isinstance(r, dict)
        and (
            (r.get("type") == "session" and isinstance(r.get("id"), str) and "cwd" in r)
            or (
                r.get("type") == "message"
                and "parentId" in r
                and isinstance(r.get("message"), dict)
                and r["message"].get("role") in ("user", "assistant")
            )
        )
        for r in rows
    )


def _ts_ms(record: dict) -> float | None:
    ts = record.get("timestamp")
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000


def _digest(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _text(content: Any) -> str | None:
    """The text of a user prompt's content blocks (``None`` if there is none)."""
    if isinstance(content, str):
        return content or None
    text = "\n".join(
        c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text"
    ).strip()
    return text or None


def _node(name: str, arguments: dict) -> str:
    return tool_operation(_NODE_TOOL_NAMES.get(name, name), arguments)


def convert_pi_session(rows: list[dict]) -> list[dict]:
    rows = [r for r in rows if isinstance(r, dict)]
    header = next((r for r in rows if r.get("type") == "session"), {})
    first_message = next((r for r in rows if r.get("type") == "message"), {})
    trace_id = header.get("id") or first_message.get("id") or "unknown"
    workflow = f"pi/{PurePosixPath(header.get('cwd') or '').name or 'unknown'}"

    llm_calls: dict[str, dict] = {}
    tool_calls: dict[str, dict] = {}
    # Chronological order of all events, so a run reads as the sequence of
    # steps the agent actually took (pattern mining depends on it).
    order: list[tuple[str, str]] = []

    run_id = trace_id
    user_input: str | None = None
    trigger = "user_prompt"
    # Digest of what the next LLM call reacts to (the prompt, or the tool
    # result it answers) -- the input a determinism judgment needs, since
    # the trigger label alone would make every turn-start call look like
    # the same input.
    reaction: str | None = None
    # Timestamp of the latest record of any type. An LLM call is timed from
    # the record just before its reply starts -- a turn can be resumed
    # hours after the last tool result, and that idle time isn't model
    # latency.
    last_seen_ms: float | None = None

    for r in rows:
        record_ms = _ts_ms(r)
        previous_ms = last_seen_ms
        if record_ms is not None:
            last_seen_ms = record_ms
        if r.get("type") != "message" or not isinstance(r.get("message"), dict):
            continue
        message = r["message"]
        role = message.get("role")
        message_id = r.get("id")

        if role == "user":
            if message_id:
                run_id = message_id
            text = _text(message.get("content"))
            if text is not None:
                user_input = text
                reaction = _digest(text)
            trigger = "user_prompt"
            continue

        if role == "toolResult":
            status = "error" if message.get("isError") else "success"
            tagged = f"{status}:{_digest(message.get('content'))}"
            reaction = tagged
            call = tool_calls.get(message.get("toolCallId"))
            if call is not None:
                call["result"] = tagged
                start_ms = call.pop("_start_ms", None)
                if record_ms is not None and start_ms is not None:
                    call["latency_ms"] = max(record_ms - start_ms, 0.0)
            name = message.get("toolName") or "unknown"
            trigger = name + (" (error)" if message.get("isError") else "")
            continue

        if role != "assistant" or not message_id:
            continue

        common = {
            "trace_id": trace_id,
            "run_id": run_id,
            "workflow": workflow,
            "agent": "pi",
            "workflow_version": str(header["version"]) if header.get("version") is not None else None,
            "user_input": user_input,
        }
        ts = record_ms
        usage = message.get("usage") or {}
        call = {
            **common,
            "event_type": "decision",
            "node": f"llm_call ← {trigger.split(':')[0].removesuffix(' (error)')}",
            "model": message.get("model"),
            "input_state": {"trigger": trigger, **({"reaction": reaction} if reaction is not None else {})},
            "trigger": trigger,
            "tokens": {
                "input": usage.get("input", 0) + usage.get("cacheRead", 0) + usage.get("cacheWrite", 0),
                "output": usage.get("output", 0),
            },
            "cost": ((usage.get("cost") or {}).get("total")) or 0.0,
            "result": message.get("stopReason"),
            "decision": None,  # filled below, once the tool calls are known
            "_start_ms": previous_ms,
        }
        if ts is not None and previous_ms is not None:
            call["latency_ms"] = max(ts - previous_ms, 0.0)
        llm_calls[message_id] = call
        order.append(("llm", message_id))

        tools: list[str] = []
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "toolCall":
                continue
            name = block.get("name") or "unknown"
            arguments = block.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {"arguments": arguments}
            node = _node(name, arguments)
            tool_id = block.get("id")
            tools.append(node)
            tool_calls[tool_id] = {
                **common,
                "event_type": "tool_call",
                "node": node,
                "parent": message_id,
                # a tool call starts from whatever the LLM call that chose it reacted to
                "trigger": trigger,
                "tool": {"name": name},
                "input_state": {
                    k: v for k, v in arguments.items() if k not in _COSMETIC_TOOL_ARGS.get(name, ())
                },
                "result": "no_result",
                "_start_ms": ts,
            }
            order.append(("tool", tool_id))
        call["decision"] = {"type": "next_action", "selected": "+".join(sorted(set(tools))) or "respond"}

    for call in llm_calls.values():
        call.pop("_start_ms")
    for call in tool_calls.values():
        call.pop("_start_ms", None)
    return [llm_calls[key] if kind == "llm" else tool_calls[key] for kind, key in order]
