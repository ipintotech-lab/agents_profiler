"""Convert a Claude Code session transcript (the JSONL files under
``~/.claude/projects/<project>/<sessionId>.jsonl``) into flat
TraceEvent-shaped dicts.

A transcript is a log of many record types (``user``, ``assistant``,
``attachment``, ``system``, ...). Only three things become trace events:

* **LLM call** -- one per API response. Claude Code writes one
  ``assistant`` record per content block, all sharing ``message.id``, so
  records are grouped by that id. Emitted as ``event_type="decision"``,
  ``node="llm_call"``, with ``input_state={"trigger": ...}`` (what the
  model was reacting to: the user prompt, or which tool's result) and
  ``decision.selected`` = the tools it chose to call, or ``"respond"``.
* **Tool call** -- one per ``tool_use`` block, paired with its
  ``tool_result`` by ``tool_use_id``. ``node`` is the tool name,
  ``input_state`` is the tool input, and ``result`` is
  ``"success:<digest>"`` / ``"error:<digest>"`` where the digest hashes the
  tool output -- so determinism means "same input -> same output", which is
  what a cache would need.

Identity: ``trace_id`` = session, ``run_id`` = the user prompt that started
the turn (``promptId``), ``workflow`` = ``claude-code/<project dir>`` (the session's first cwd).

Cost: transcripts carry no per-message cost, only a session-level
``cost-state`` record with real USD per model. That total is split across
the model's LLM calls in proportion to their price-weighted tokens
(relative weights below). Sessions with no ``cost-state`` record (e.g.
still running) get ``cost=0``.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

# Relative per-token price weights (input = 1). Used only to split the
# session's recorded cost across calls, never to invent an absolute cost.
_WEIGHT_INPUT = 1.0
_WEIGHT_CACHE_READ = 0.1
_WEIGHT_CACHE_WRITE_5M = 1.25
_WEIGHT_CACHE_WRITE_1H = 2.0
_WEIGHT_OUTPUT = 5.0

_SYNTHETIC_MODEL = "<synthetic>"


def is_claude_code_transcript(rows: list) -> bool:
    return any(
        isinstance(r, dict)
        and r.get("type") in ("assistant", "user")
        and "sessionId" in r
        and isinstance(r.get("message"), dict)
        for r in rows
    )


def _ts_ms(record: dict) -> float | None:
    ts = record.get("timestamp")
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000


def _digest(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _weighted_tokens(usage: dict) -> float:
    cache_creation = usage.get("cache_creation") or {}
    write_1h = cache_creation.get("ephemeral_1h_input_tokens")
    write_5m = cache_creation.get("ephemeral_5m_input_tokens")
    if write_1h is None and write_5m is None:
        write_5m, write_1h = usage.get("cache_creation_input_tokens", 0), 0
    return (
        _WEIGHT_INPUT * usage.get("input_tokens", 0)
        + _WEIGHT_CACHE_READ * usage.get("cache_read_input_tokens", 0)
        + _WEIGHT_CACHE_WRITE_5M * (write_5m or 0)
        + _WEIGHT_CACHE_WRITE_1H * (write_1h or 0)
        + _WEIGHT_OUTPUT * usage.get("output_tokens", 0)
    )


def _user_trigger(record: dict, tool_names_by_id: dict[str, str]) -> str | None:
    """What an LLM call following this user record is reacting to."""
    content = record["message"].get("content")
    if isinstance(content, str):
        return "user_prompt"
    tool_results = [c for c in content or [] if isinstance(c, dict) and c.get("type") == "tool_result"]
    if tool_results:
        names = sorted({tool_names_by_id.get(c.get("tool_use_id"), "unknown") for c in tool_results})
        return "tool_result:" + "+".join(names)
    return "user_prompt"


def convert_claude_code_transcript(rows: list[dict]) -> list[dict]:
    rows = [r for r in rows if isinstance(r, dict)]
    session_cost_by_model: dict[str, float] = {}
    for r in rows:
        if r.get("type") == "cost-state":
            for model, usage in (r.get("modelUsage") or {}).items():
                session_cost_by_model[model] = float(usage.get("costUSD") or 0.0)

    llm_calls: dict[str, dict] = {}
    llm_call_order: list[str] = []
    tool_calls: dict[str, dict] = {}
    tool_call_order: list[str] = []
    tool_names_by_id: dict[str, str] = {}

    current_prompt: dict[str, str] = {}
    session_root: dict[str, str] = {}
    last_trigger: dict[str, tuple[str, float | None]] = {}

    for r in rows:
        kind = r.get("type")
        session = r.get("sessionId")
        if kind not in ("user", "assistant") or not session or not isinstance(r.get("message"), dict):
            continue
        sidechain = bool(r.get("isSidechain"))
        lane = f"{session}:{sidechain}"

        if kind == "user":
            if r.get("promptId"):
                current_prompt[session] = r["promptId"]
            if r.get("isMeta"):
                continue
            last_trigger[lane] = (_user_trigger(r, tool_names_by_id), _ts_ms(r))
            for c in r["message"].get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("tool_use_id") in tool_calls:
                    call = tool_calls[c["tool_use_id"]]
                    status = "error" if c.get("is_error") else "success"
                    call["result"] = f"{status}:{_digest(c.get('content'))}"
                    end_ms, start_ms = _ts_ms(r), call.pop("_start_ms", None)
                    if end_ms is not None and start_ms is not None:
                        call["latency_ms"] = max(end_ms - start_ms, 0.0)
            continue

        message = r["message"]
        model = message.get("model")
        message_id = message.get("id") or r.get("uuid")
        if not message_id or model == _SYNTHETIC_MODEL:
            continue

        # Pin the workflow to the session's first cwd so a mid-session
        # ``cd`` doesn't split one session into several workflows.
        cwd = session_root.setdefault(session, r.get("cwd") or "")
        common = {
            "trace_id": session,
            "run_id": current_prompt.get(session, session),
            "workflow": f"claude-code/{PurePosixPath(cwd).name or 'unknown'}",
            "agent": "claude-code-subagent" if sidechain else "claude-code",
            "workflow_version": r.get("version"),
        }
        ts = _ts_ms(r)

        call = llm_calls.get(message_id)
        if call is None:
            trigger, trigger_ms = last_trigger.get(lane, ("user_prompt", None))
            call = {
                **common,
                "event_type": "decision",
                "node": "llm_call",
                "model": model,
                "input_state": {"trigger": trigger},
                "_tools": [],
                "_start_ms": trigger_ms,
            }
            llm_calls[message_id] = call
            llm_call_order.append(message_id)
        usage = message.get("usage") or {}
        call["_usage"] = usage
        call["result"] = message.get("stop_reason") or call.get("result")
        if ts is not None and call["_start_ms"] is not None:
            call["latency_ms"] = max(ts - call["_start_ms"], 0.0)

        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_id, name = block.get("id"), block.get("name") or "unknown"
            tool_names_by_id[tool_id] = name
            call["_tools"].append(name)
            tool_calls[tool_id] = {
                **common,
                "event_type": "tool_call",
                "node": name,
                "parent": message_id,
                "tool": {"name": name},
                "input_state": block.get("input") or {},
                "result": "no_result",
                "_start_ms": ts,
            }
            tool_call_order.append(tool_id)

    weights_by_model: dict[str, float] = defaultdict(float)
    for call in llm_calls.values():
        call["_weight"] = _weighted_tokens(call["_usage"])
        weights_by_model[call["model"]] += call["_weight"]

    events: list[dict] = []
    for message_id in llm_call_order:
        call = llm_calls[message_id]
        usage = call.pop("_usage")
        weight = call.pop("_weight")
        tools = call.pop("_tools")
        call.pop("_start_ms")
        call["tokens"] = {
            "input": usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0),
            "output": usage.get("output_tokens", 0),
        }
        call["decision"] = {"type": "next_action", "selected": "+".join(sorted(set(tools))) or "respond"}
        total_weight = weights_by_model[call["model"]]
        model_cost = session_cost_by_model.get(call["model"], 0.0)
        call["cost"] = model_cost * weight / total_weight if total_weight else 0.0
        events.append(call)
    for tool_id in tool_call_order:
        call = tool_calls[tool_id]
        call.pop("_start_ms", None)
        events.append(call)
    return events
