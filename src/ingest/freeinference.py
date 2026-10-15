"""Convert a FreeInference agentic-trace session -- one line of
``release/<week>/traces/<day>.jsonl.zst`` in
``harvardMadsys/freeinference_agentic_trace`` on Hugging Face -- into flat
TraceEvent-shaped dicts.

A session is ``{"session_id", "harness", "requests": [...]}``; each request
is one LLM call with its ``tool_calls``, and a tool call that launched a
subagent carries that subagent's whole session under ``subagent``.

* **LLM call** -- one per request: ``event_type="decision"``,
  ``node="llm_call ← <what triggered it>"`` (``user_prompt``, or the tools
  whose results it reacted to), real ``provider_tokens`` and gateway
  latency, ``decision.selected`` = the tools it called, or ``"respond"``.
* **Tool call** -- one per tool call. The dataset releases no message text:
  argument values are typed placeholders (``<path>``, ``<str>``) and tool
  output is only an outcome plus a token count. So ``input_state`` is the
  argument *shape*, and ``result`` is ``"<outcome>:<result_tokens>tok"`` --
  a coarse stand-in for the output that keeps differently-sized outputs
  from reading as identical. Two calls whose arguments hold placeholders
  can't be known to share an input, so such a call's ``input_state`` also
  gets a unique ``anonymized`` id: it never counts as a repeat, and only
  calls with fully concrete arguments feed determinism and caching.

Identity: ``trace_id`` = session, ``run_id`` = session + user turn,
``workflow`` = ``freeinference/<harness>``. No cost is recorded, so it stays 0.
"""
from __future__ import annotations

import re
from typing import Any

from .claude_code import bash_operation, tool_operation

_PLACEHOLDER = re.compile(r"^<\w+>$")
_HAS_PLACEHOLDER = re.compile(r"<\w+>")


def is_freeinference_session(record: Any) -> bool:
    return isinstance(record, dict) and "session_id" in record and isinstance(record.get("requests"), list)


def _tool_node(name: str, arguments: dict) -> str:
    command = arguments.get("command")
    if isinstance(command, str) and command.strip() and not _PLACEHOLDER.match(command):
        return f"{name}: {bash_operation(command)}"
    # placeholder values say nothing about which operation ran
    named = {k: v for k, v in arguments.items() if not (isinstance(v, str) and _PLACEHOLDER.match(v))}
    return tool_operation(name, named)


def _trigger(request: dict, previous_tools: list[str]) -> str:
    kind = request.get("step_trigger")
    if kind == "user" or kind is None:
        return "user_prompt"
    if kind == "tool" and previous_tools:
        return " + ".join(sorted(set(previous_tools)))
    return str(kind)


def convert_freeinference_session(
    session: dict, trace_id: str | None = None, run_prefix: str | None = None, subagent: bool = False
) -> list[dict]:
    session_id = str(session["session_id"])
    trace_id = trace_id or session_id
    harness = session.get("harness") or "unknown"
    events: list[dict] = []
    previous_tools: list[str] = []

    for request in session["requests"]:
        turn = request.get("turn_index", 0)
        common = {
            "trace_id": trace_id,
            "run_id": run_prefix or f"{session_id}:turn{turn}",
            "workflow": f"freeinference/{harness}",
            "agent": f"{harness}-subagent" if subagent else harness,
        }
        trigger = _trigger(request, previous_tools)
        call_id = f"{session_id}:{request.get('step', len(events))}"
        tokens = request.get("provider_tokens") or {}
        start, end = request.get("start_ms"), request.get("end_ms")
        llm_call = {
            **common,
            "event_type": "decision",
            "node": f"llm_call ← {trigger.split(' + ')[0].split(':')[0].removesuffix(' (error)')}",
            "model": request.get("model"),
            "input_state": {"trigger": trigger},
            "trigger": trigger,
            "tokens": {"input": tokens.get("prompt") or 0, "output": tokens.get("completion") or 0},
            "latency_ms": max(end - start, 0) if start is not None and end is not None else 0.0,
            "result": request.get("finish_reason"),
        }
        events.append(llm_call)

        nodes, previous_tools = [], []
        for tc in request.get("tool_calls") or []:
            name = tc.get("name") or "unknown"
            arguments = {str(pair[0]): pair[1] for pair in tc.get("arguments") or [] if len(pair) == 2}
            node = _tool_node(name, arguments)
            input_state = dict(arguments)
            if any(_HAS_PLACEHOLDER.search(str(v)) for v in arguments.values()):
                input_state["anonymized"] = f"{call_id}:{tc.get('call_index', len(nodes))}"
            nodes.append(node)
            outcome = tc.get("outcome") or "unknown"
            previous_tools.append(node + (" (error)" if outcome == "error" else ""))
            events.append(
                {
                    **common,
                    "event_type": "tool_call",
                    "node": node,
                    "parent": call_id,
                    "trigger": trigger,
                    "tool": {"name": name},
                    "input_state": input_state,
                    "latency_ms": tc.get("latency_ms") or 0.0,
                    "result": f"{outcome}:{tc.get('result_tokens') or 0}tok",
                }
            )
            if isinstance(tc.get("subagent"), dict) and is_freeinference_session(tc["subagent"]):
                events.extend(
                    convert_freeinference_session(tc["subagent"], trace_id, common["run_id"], subagent=True)
                )
        llm_call["decision"] = {"type": "next_action", "selected": "+".join(sorted(set(nodes))) or "respond"}
    return events


def convert_freeinference_sessions(sessions: list) -> list[dict]:
    return [event for session in sessions for event in convert_freeinference_session(session)]
