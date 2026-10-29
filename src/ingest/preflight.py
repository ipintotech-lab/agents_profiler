"""What an upload holds when it converts to no trace events, so the
report can say "this transcript has no agent activity" -- with the counts
behind it -- instead of a generic "no workflow found".

Only counts raw records; nothing is kept. Counts describe the file, not
the agent: a ``cost-state`` record saying $0 with no model usage is not
evidence the work cost nothing.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .claude_code import is_claude_code_transcript

NO_AGENT_ACTIVITY = "recognized_no_agent_activity"
NO_EVENTS = "no_events"


def _records(parsed: Any) -> list:
    if isinstance(parsed, list):
        return parsed
    return [parsed] if parsed is not None else []


def describe_empty(parsed: Any) -> dict:
    """Coverage and diagnostics for a parsed upload that yielded no events."""
    records = _records(parsed)
    if not is_claude_code_transcript(records):
        return {
            "status": NO_EVENTS,
            "recognized_format": None,
            "coverage": {"raw_records": len(records), "converted_events": 0},
            "diagnostics": [{"code": "NO_EVENTS", "severity": "blocking",
                             "message": "The file parsed but holds no trace events."}],
            "next_actions": ["Upload a file with one event per LLM call or tool call."],
        }
    types = Counter(r.get("type") for r in records if isinstance(r, dict))
    sessions = {r["sessionId"] for r in records if isinstance(r, dict) and r.get("sessionId")}
    tool_uses = sum(
        1
        for r in records
        if isinstance(r, dict) and r.get("type") == "assistant" and isinstance(r.get("message"), dict)
        for block in r["message"].get("content") or []
        if isinstance(block, dict) and block.get("type") == "tool_use"
    )
    diagnostics = []
    if not types["assistant"]:
        diagnostics.append({"code": "NO_ASSISTANT_RESPONSES", "severity": "blocking",
                            "message": "No assistant responses, so there are no LLM calls to analyze."})
    if not tool_uses:
        diagnostics.append({"code": "NO_TOOL_EXECUTIONS", "severity": "blocking",
                            "message": "No tool calls, so there are no repeated steps to find."})
    if types["cost-state"]:
        diagnostics.append({"code": "COST_UNVERIFIABLE", "severity": "warning",
                            "message": "The session's cost record can't be tied to any LLM call, "
                            "so it doesn't show what the work cost."})
    return {
        "status": NO_AGENT_ACTIVITY,
        "recognized_format": "claude_code_transcript",
        "coverage": {
            "raw_records": len(records),
            "sessions": len(sessions),
            "user_records": types["user"],
            "assistant_records": types["assistant"],
            "tool_use_records": tool_uses,
            "converted_events": 0,
        },
        "diagnostics": diagnostics,
        "next_actions": [
            "Upload a session in which the agent answered and ran tools (a .jsonl under ~/.claude/projects/).",
            "Or generate sample data to try the profiler.",
        ],
    }


def summary_line(name: str, preflight: dict) -> str:
    """One sentence for an error message."""
    c = preflight["coverage"]
    if preflight["status"] == NO_AGENT_ACTIVITY:
        return (
            f"{name} is a Claude Code transcript with {c['raw_records']} records but "
            f"{c['assistant_records']} assistant responses and {c['tool_use_records']} tool calls"
        )
    return f"{name} has {c['raw_records']} records but no trace events"
