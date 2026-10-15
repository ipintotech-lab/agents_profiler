"""The report's summary: totals for the analyzed workflow and a ranked list
of savings opportunities built from the evidence the other stages already
collected -- repeated sequences, repeated calls, and fingerprint groups.

Each opportunity names one optimization type, what to do in plain words,
the evidence behind it, and the observed cost it would remove. Savings are
the cost of the observed calls that would not have run, not a forecast.
"""
from __future__ import annotations

import re

from ingest.trace_schema import TraceEvent

from .mine import ActionStats

MAX_PER_TYPE = 5
# An LLM step that picks the same answer 80% of the time still looks
# consistent 3 times in a row about half the time (0.8^3), but 5 in a row
# only a third of the time (0.8^5). Measured on the POC's mock agents (10
# seeds): 3 -> 5 cut "replace with a rule" false alarms at 500 lines from 5
# to 1 and found every planted one from 2000 lines on, same as before.
MIN_BYPASS_OBSERVATIONS = 5

# A call that changes something (writes a ticket, sends a message, edits a
# file) can return the same output for the same input every time and still
# not be safe to cache: skipping the repeat skips the change. Traces don't
# say which calls have side effects, so this goes by the operation's name.
# These words mark a write wherever they appear in the name...
WRITE_WORDS = frozenset({
    "create", "update", "upsert", "delete", "remove", "insert", "write", "edit", "replace", "send", "post",
    "publish", "submit", "patch", "put", "merge", "commit", "push", "deploy", "rollback", "rerun",
})
# ...and these only as the first word ("log_activity" writes, "fetch_error_log" doesn't).
WRITE_VERBS = frozenset({"log", "page", "notify", "trigger", "schedule", "apply", "pause", "cancel", "close", "book"})

# Plain-language meaning of each optimization type, shown next to the name.
OPTIMIZATION_TYPES = {
    "EXTRACT_DETERMINISTIC_SUBSTEP": "The same steps run in the same order again and again. Turning them into one script removes the LLM calls that pick each next step.",
    "CACHE_RESULT": "The same operation ran with the identical input and gave the same output every time. Reusing the first result skips the repeats. Operations whose name says they change something (create, update, send, edit...) are never suggested.",
    "CONDITIONAL_LLM_BYPASS": "An LLM call made the same choice every time it saw the same starting point and input. A rule could make that choice without calling the model.",
    "REVIEW_HIDDEN_STATE": "The same request gave different results even with identical context and configuration, so something the traces don't record is changing the outcome.",
}


def _totals(events: list[TraceEvent]) -> dict:
    return {
        "cost": sum(e.cost for e in events),
        "llm_calls": sum(1 for e in events if e.model),
        "tool_calls": sum(1 for e in events if e.tool is not None or e.event_type == "tool_call"),
        "tokens": sum(e.tokens.input + e.tokens.output for e in events),
        "runs": len({e.run_id for e in events}),
        "sessions": len({e.trace_id for e in events}),
    }


def _sequence_opportunities(sequences: list[dict]) -> list[dict]:
    return [
        {
            "type": "EXTRACT_DETERMINISTIC_SUBSTEP",
            "title": " → ".join(p["steps"]),
            "action": "Combine these steps into one scripted step",
            "evidence": f"{p['occurrences']}× in {p['runs']} runs · {p['llm_calls_between_steps']} LLM calls between steps",
            "saving": p["cost_between_steps"],
            "observed_cost": p["sequence_cost"],
        }
        for p in sequences
        if p["llm_calls_between_steps"] > 0
    ]


def _words(name: str) -> list[list[str]]:
    """Words of each part of a name: ``"mcp__jira__createIssue"`` ->
    ``[["mcp"], ["jira"], ["create", "issue"]]``."""
    parts = re.split(r"[.:/]|__", name)
    return [
        [w.lower() for w in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", part)]
        for part in parts
        if part
    ]


def has_side_effect(name: str) -> bool:
    """Whether an operation's name says it changes something."""
    return any(
        any(w in WRITE_WORDS for w in words) or (words and words[0] in WRITE_VERBS)
        for words in _words(name)
    )


def _cache_opportunities(calls: list[dict], names: dict[str, set[str]]) -> list[dict]:
    return [
        {
            "type": "CACHE_RESULT",
            "title": c["node"],
            "input": c["input"],
            "action": "Reuse the first result, or skip the repeat",
            "evidence": f"{c['count']}× with identical input, same output every time · {c['redundant_calls']} repeats",
            "saving": c["cost_with_decision"] * c["redundant_calls"] / c["count"],
            "observed_cost": c["cost_with_decision"],
        }
        for c in calls
        if c["redundant_calls"] > 0
        and not any(has_side_effect(n) for n in names.get(c["node"], set()) | {c["node"]})
    ]


def _fingerprint_opportunities(stats: list[ActionStats], llm_nodes: set[tuple[str, str]]) -> list[dict]:
    found = []
    for s in stats:
        fp = s.fingerprints
        # Only an LLM call can be replaced by a rule, and only an LLM call's
        # unexplained variance points at hidden state -- a tool whose output
        # changes for the same input is just reading a world that changed.
        if not fp or (s.agent, s.node) not in llm_nodes:
            continue
        if fp["variance"].get("factor") == "unexplained":
            found.append(
                {
                    "type": "REVIEW_HIDDEN_STATE",
                    "title": s.node,
                    "action": "Find what changes the result: state or inputs the traces don't capture",
                    "evidence": fp["variance"]["explanation"],
                    "saving": None,
                    "observed_cost": s.avg_cost * s.occurrences,
                }
            )
        for g in fp["groups"]:
            if g["verdict"] != "consistent" or g["observations"] < MIN_BYPASS_OBSERVATIONS or g["avg_cost"] <= 0:
                continue
            found.append(
                {
                    "type": "CONDITIONAL_LLM_BYPASS",
                    "title": s.node,
                    "input": g["input"],
                    "trigger": g["trigger"],
                    "action": f"Replace the LLM call with a rule that returns {g['dominant_output']}",
                    "evidence": f"same choice {g['observations']} of {g['observations']} times after {g['trigger']}",
                    "saving": g["avg_cost"] * (g["observations"] - 1),
                    "observed_cost": g["total_cost"],
                }
            )
    return found


def find_opportunities(
    events: list[TraceEvent], workflow: str, repetition: dict, stats: list[ActionStats]
) -> list[dict]:
    """Every opportunity found, largest saving first (the summary shows only
    a few of each type)."""
    relevant = [e for e in events if e.workflow == workflow]
    llm_nodes = {(e.agent, e.node) for e in relevant if e.model}
    tool_names: dict[str, set[str]] = {}
    for e in relevant:
        if e.tool is not None:
            tool_names.setdefault(e.node, set()).add(e.tool.name)
    opportunities = (
        _sequence_opportunities(repetition["sequences"])
        + _cache_opportunities(repetition["calls"], tool_names)
        + _fingerprint_opportunities(stats, llm_nodes)
    )
    opportunities.sort(key=lambda o: (o["saving"] is None, -(o["saving"] or 0), -o["observed_cost"]))
    return opportunities


def build_summary(events: list[TraceEvent], workflow: str, repetition: dict, stats: list[ActionStats]) -> dict:
    relevant = [e for e in events if e.workflow == workflow]
    opportunities = find_opportunities(events, workflow, repetition, stats)
    # A few of each type, so one kind of finding can't fill the list.
    per_type: dict[str, int] = {}
    shown = []
    for o in opportunities:
        per_type[o["type"]] = per_type.get(o["type"], 0) + 1
        if per_type[o["type"]] <= MAX_PER_TYPE:
            shown.append(o)
    return {
        "totals": _totals(relevant),
        "opportunities": shown,
        "num_opportunities": len(opportunities),
        "optimization_types": OPTIMIZATION_TYPES,
        "notes": (
            "Savings are the observed cost of calls that would not have run -- measured from these traces, "
            "not a forecast. Opportunities can overlap (a cached call may sit inside a scripted sequence), "
            "so don't add them up as a guaranteed total."
        ),
    }
