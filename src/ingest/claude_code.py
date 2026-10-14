"""Convert a Claude Code session transcript (the JSONL files under
``~/.claude/projects/<project>/<sessionId>.jsonl``) into flat
TraceEvent-shaped dicts.

A transcript is a log of many record types (``user``, ``assistant``,
``attachment``, ``system``, ...). Only three things become trace events:

* **LLM call** -- one per API response. Claude Code writes one
  ``assistant`` record per content block, all sharing ``message.id``, so
  records are grouped by that id. Emitted as ``event_type="decision"``
  with ``node="llm_call ← <tool>"`` (what the model was reacting to: the
  user prompt or a tool's result), ``input_state={"trigger": ...}`` (the
  exact operation, e.g. ``Bash: git status (error)``) and
  ``decision.selected`` = the tools it chose to call next, or ``"respond"``.
* **Tool call** -- one per ``tool_use`` block, paired with its
  ``tool_result`` by ``tool_use_id``. ``node`` is the tool plus the
  operation it performed (see ``tool_operation``: ``Bash: git status``,
  ``Read: .py``, ``computer: screenshot``), ``input_state`` is the tool
  input, and ``result`` is ``"success:<digest>"`` / ``"error:<digest>"``
  where the digest hashes the tool output -- so determinism means "same
  input -> same output", which is what a cache would need.

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
import shlex
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

# Tool arguments that are only a human-readable label and don't change what
# the tool does -- dropped so the same command counts as the same input.
_COSMETIC_TOOL_ARGS = {"Bash": ("description",)}


# Programs whose first sub-command is what identifies the operation
# (``git status`` vs ``git push``), not the program alone.
_SUBCOMMAND_PROGRAMS = {"git", "gh", "railway", "npm", "pnpm", "yarn", "docker", "kubectl", "uv", "pip", "cargo", "go", "make"}


def bash_operation(command: str) -> str:
    """Name the operation a Bash command performs, e.g.
    ``cd x && .venv/bin/python -m pytest -q`` -> ``python -m pytest``,
    ``git push origin main`` -> ``git push``. Falls back to ``other``."""
    segments = [seg.strip() for seg in command.replace("\n", "&&").replace(";", "&&").split("&&")]
    for seg in segments:
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        while tokens and ("=" in tokens[0] and not tokens[0].startswith("=")):
            tokens = tokens[1:]  # VAR=value prefixes
        if tokens and tokens[0] in ("timeout",):
            tokens = [t for t in tokens[1:] if not t[:1].isdigit() and not t.startswith("-")] or []
        if not tokens or tokens[0] in ("cd", "export", "set", "source", "."):
            continue
        program = PurePosixPath(tokens[0]).name
        rest = [t for t in tokens[1:] if not t.startswith("-")]
        if program.startswith("python") and len(tokens) > 2 and tokens[1] == "-m":
            return f"python -m {tokens[2]}"
        if program.startswith("python"):
            return "python"
        if program in _SUBCOMMAND_PROGRAMS and rest:
            return f"{program} {rest[0]}"
        return program
    return "other"


# Arguments that name *which operation* a tool performs (e.g. the browser
# tool's ``action``: screenshot / left_click). Checked in order.
_OPERATION_ARGS = ("action", "operation", "method", "subcommand", "mode", "type")
# Arguments holding a file path: the operation is the file type, and the
# path itself is the input.
_PATH_ARGS = ("file_path", "notebook_path", "path")


def tool_operation(name: str, tool_input: dict) -> str:
    """Node name for a tool call: the tool plus the operation it performed,
    so the profiler compares like with like (``Bash: git status``,
    ``Read: .py``, ``computer: screenshot``) instead of lumping every use
    of a tool together."""
    command = tool_input.get("command")
    if name == "Bash" and isinstance(command, str):
        return f"Bash: {bash_operation(command)}"
    for arg in _OPERATION_ARGS:
        value = tool_input.get(arg)
        if isinstance(value, str) and 0 < len(value) <= 40 and "\n" not in value:
            return f"{name}: {value}"
    for arg in _PATH_ARGS:
        value = tool_input.get(arg)
        if isinstance(value, str) and value:
            suffix = PurePosixPath(value).suffix
            return f"{name}: {suffix or PurePosixPath(value).name}"
    return name


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
        names = sorted(
            {
                tool_names_by_id.get(c.get("tool_use_id"), "unknown") + (" (error)" if c.get("is_error") else "")
                for c in tool_results
            }
        )
        return " + ".join(names)
    return "user_prompt"


def convert_claude_code_transcript(rows: list[dict]) -> list[dict]:
    rows = [r for r in rows if isinstance(r, dict)]
    session_cost_by_model: dict[str, float] = {}
    for r in rows:
        if r.get("type") == "cost-state":
            for model, usage in (r.get("modelUsage") or {}).items():
                session_cost_by_model[model] = float(usage.get("costUSD") or 0.0)

    llm_calls: dict[str, dict] = {}
    # Chronological order of all events, so a run reads as the sequence of
    # steps the agent actually took (pattern mining depends on it).
    order: list[tuple[str, str]] = []
    tool_calls: dict[str, dict] = {}
    tool_names_by_id: dict[str, str] = {}

    current_prompt: dict[str, str] = {}
    session_root: dict[str, str] = {}
    last_trigger: dict[str, str] = {}
    # Timestamp of the latest record of *any* type per lane. An LLM call is
    # timed from the record just before its reply starts, not from its
    # trigger: a turn can be started by a queued message, a task
    # notification or a resumed session hours after the last tool result,
    # and that idle time isn't model latency.
    last_seen_ms: dict[str, float] = {}

    for r in rows:
        kind = r.get("type")
        session = r.get("sessionId")
        record_ms = _ts_ms(r)
        previous_ms = None
        if session and record_ms is not None:
            seen_lane = f"{session}:{bool(r.get('isSidechain'))}"
            previous_ms = last_seen_ms.get(seen_lane)
            last_seen_ms[seen_lane] = record_ms
        if kind not in ("user", "assistant") or not session or not isinstance(r.get("message"), dict):
            continue
        sidechain = bool(r.get("isSidechain"))
        lane = f"{session}:{sidechain}"

        if kind == "user":
            if r.get("promptId"):
                current_prompt[session] = r["promptId"]
            if r.get("isMeta"):
                continue
            last_trigger[lane] = _user_trigger(r, tool_names_by_id)
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
            trigger = last_trigger.get(lane, "user_prompt")
            call = {
                **common,
                "event_type": "decision",
                "node": f"llm_call ← {trigger.split(' + ')[0].split(':')[0].removesuffix(' (error)')}",
                "model": model,
                "input_state": {"trigger": trigger},
                "_tools": [],
                "_start_ms": previous_ms,
            }
            llm_calls[message_id] = call
            order.append(("llm", message_id))
        usage = message.get("usage") or {}
        call["_usage"] = usage
        call["result"] = message.get("stop_reason") or call.get("result")
        if ts is not None and call["_start_ms"] is not None:
            call["latency_ms"] = max(ts - call["_start_ms"], 0.0)

        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_id, name = block.get("id"), block.get("name") or "unknown"
            tool_input = block.get("input") or {}
            node = tool_operation(name, tool_input)
            tool_names_by_id[tool_id] = node
            call["_tools"].append(node)
            tool_calls[tool_id] = {
                **common,
                "event_type": "tool_call",
                "node": node,
                "parent": message_id,
                "tool": {"name": name},
                "input_state": {
                    k: v for k, v in tool_input.items() if k not in _COSMETIC_TOOL_ARGS.get(name, ())
                },
                "result": "no_result",
                "_start_ms": ts,
            }
            order.append(("tool", tool_id))

    weights_by_model: dict[str, float] = defaultdict(float)
    for call in llm_calls.values():
        call["_weight"] = _weighted_tokens(call["_usage"])
        weights_by_model[call["model"]] += call["_weight"]

    for call in llm_calls.values():
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
    for call in tool_calls.values():
        call.pop("_start_ms", None)
    return [llm_calls[key] if kind == "llm" else tool_calls[key] for kind, key in order]
