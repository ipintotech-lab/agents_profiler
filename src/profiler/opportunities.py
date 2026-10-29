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
# not be safe to cache: skipping the repeat skips the change. What the
# trace declares (``has_side_effects``, ``cache_eligible``) decides; when
# nothing is declared, the operation's name is a weak hint -- the word lists
# below catch common writes but are not complete.
# These words mark a write wherever they appear in the name...
WRITE_WORDS = frozenset({
    "create", "update", "upsert", "delete", "remove", "insert", "write", "edit", "replace", "send", "post",
    "publish", "submit", "patch", "put", "merge", "commit", "push", "deploy", "rollback", "rerun",
    "payment", "charge", "refund", "transfer", "webhook",
})
# ...and these only as the first word ("log_activity" writes, "fetch_error_log" doesn't).
WRITE_VERBS = frozenset({
    "log", "page", "notify", "trigger", "schedule", "apply", "pause", "cancel", "close", "book",
    "assign", "sync", "process", "set", "mark", "toggle", "record", "track", "increment", "decrement",
    "handle", "refresh", "reset", "approve", "reject", "archive", "enqueue", "register",
})

# Plain-language meaning of each optimization type, shown next to the name.
OPTIMIZATION_TYPES = {
    "EXTRACT_DETERMINISTIC_SUBSTEP": "The same steps run in the same order again and again. Turning them into one script removes the LLM calls that pick each next step.",
    "CACHE_RESULT": "The same operation ran with the identical input, model and version and gave the same output every time. Reusing the first result skips the repeats. Operations declared to change something are never suggested; when nothing is declared, neither are operations whose name says they change something (create, update, send, edit...).",
    "CONDITIONAL_LLM_BYPASS": "An LLM call made the same choice every time it saw the same starting point and input, on the same model and version. A rule could make that choice without calling the model.",
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


def _sequence_opportunities(sequences: list[dict], signals: dict[str, dict]) -> list[dict]:
    return [
        {
            "type": "EXTRACT_DETERMINISTIC_SUBSTEP",
            "title": " → ".join(p["steps"]),
            "action": "Combine these steps into one scripted step",
            "evidence": f"{p['occurrences']}× in {p['runs']} runs · {p['llm_calls_between_steps']} LLM calls between steps",
            "nodes": list(p["steps"]),
            "changing_steps": [n for n in p["steps"] if signals[n]["changes_something"]],
            "observations": p["occurrences"],
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


def declared_side_effects(events: list[TraceEvent]) -> dict[str, dict]:
    """Per node, what its events declare. A node is side-effect-free (or
    idempotent, or cache-eligible) only if every event says so; one event
    declaring a side effect (or not cache-eligible) is enough to mark it."""
    by_node: dict[str, list[TraceEvent]] = {}
    for e in events:
        by_node.setdefault(e.node, []).append(e)

    def combine(values: list[bool | None], unsafe: bool) -> bool | None:
        if unsafe in values:
            return unsafe
        return (not unsafe) if values and all(v == (not unsafe) for v in values) else None

    declared = {}
    for node, group in by_node.items():
        ttls = [e.cache_ttl_seconds for e in group if e.cache_ttl_seconds is not None]
        keys = sorted({k for e in group for k in e.invalidation_keys or []})
        declared[node] = {
            "has_side_effects": combine([e.has_side_effects for e in group], True),
            "idempotent": combine([e.idempotent for e in group], False),
            "cache_eligible": combine([e.cache_eligible for e in group], False),
            "cache_ttl_seconds": min(ttls) if ttls else None,
            "invalidation_keys": keys or None,
        }
    return declared


def side_effect_signal(node: str, names: set[str], declared: dict) -> dict:
    """Whether ``node`` changes something (``changes_something``: True /
    False / None) and where that comes from (``signal``): ``declared`` by
    the trace, a ``heuristic_name_match`` on its operation names (not
    authoritative), or ``unknown``. ``blocks_cache`` rules caching out: a
    declared side effect or ``cache_eligible=false`` always does, a name
    match does unless the trace declares the node cache-eligible."""
    if declared["has_side_effects"] is not None:
        signal, changes = "declared", declared["has_side_effects"]
    elif any(has_side_effect(n) for n in names | {node}):
        signal, changes = "heuristic_name_match", True
    else:
        signal, changes = "unknown", None
    blocks_cache = (
        declared["cache_eligible"] is False
        or declared["has_side_effects"] is True
        or (signal == "heuristic_name_match" and declared["cache_eligible"] is not True)
    )
    return {**declared, "signal": signal, "changes_something": changes, "blocks_cache": blocks_cache}


def _cache_opportunities(calls: list[dict], signals: dict[str, dict]) -> list[dict]:
    return [
        {
            "type": "CACHE_RESULT",
            "title": c["node"],
            "input": c["input"],
            "identity": c["identity"],
            "side_effects": signals[c["node"]],
            "action": "Reuse the first result, or skip the repeat",
            "evidence": f"{c['count']}× with identical input, same output every time · {c['redundant_calls']} repeats",
            "nodes": [c["node"]],
            "observations": c["count"],
            "saving": c["cost_with_decision"] * c["redundant_calls"] / c["count"],
            "observed_cost": c["cost_with_decision"],
        }
        for c in calls
        if c["redundant_calls"] > 0
        and not signals[c["node"]]["blocks_cache"]
    ]


def _fingerprint_opportunities(
    stats: list[ActionStats], llm_nodes: set[tuple[str, str]], signals: dict[str, dict]
) -> list[dict]:
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
                    "nodes": [s.node],
                    "observations": s.occurrences,
                    "saving": None,
                    "observed_cost": s.avg_cost * s.occurrences,
                }
            )
        # a rule standing in for the call would skip what it declares it changes
        if signals[s.node]["has_side_effects"] is True:
            continue
        # One configuration at a time: the same choice across two models is
        # two observations of different actions, not one rule.
        for g in fp["groups"]:
            for c in g["configs"]:
                if c["verdict"] != "consistent" or c["observations"] < MIN_BYPASS_OBSERVATIONS or c["avg_cost"] <= 0:
                    continue
                identity = c["identity"]
                found.append(
                    {
                        "type": "CONDITIONAL_LLM_BYPASS",
                        "title": s.node,
                        "input": g["input"],
                        "trigger": g["trigger"],
                        "identity": identity,
                        "side_effects": signals[s.node],
                        "action": f"Replace the LLM call with a rule that returns {c['dominant_output']}",
                        "evidence": (
                            f"same choice {c['observations']} of {c['observations']} times after {g['trigger']} "
                            f"on model {identity['model']}, version {identity['workflow_version']}"
                        ),
                        "nodes": [s.node],
                        "observations": c["observations"],
                        "saving": c["avg_cost"] * (c["observations"] - 1),
                        "observed_cost": c["total_cost"],
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
    signals = {
        node: side_effect_signal(node, tool_names.get(node, set()), declared)
        for node, declared in declared_side_effects(relevant).items()
    }
    opportunities = (
        _sequence_opportunities(repetition["sequences"], signals)
        + _cache_opportunities(repetition["calls"], signals)
        + _fingerprint_opportunities(stats, llm_nodes, signals)
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
