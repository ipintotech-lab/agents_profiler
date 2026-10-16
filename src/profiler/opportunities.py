"""The report's summary: totals for the analyzed workflow and a ranked list
of savings opportunities built from the evidence the other stages already
collected -- repeated sequences, repeated calls, and fingerprint groups.

Each opportunity names one optimization type, what to do in plain words,
the evidence behind it, and the observed cost it would remove. Savings are
the cost of the observed calls that would not have run, not a forecast.
"""
from __future__ import annotations

import hashlib
import json
import re

from ingest.trace_schema import TraceEvent

from .patterns import step_signature
from .prices import SCENARIOS

from .mine import ActionStats

MAX_PER_TYPE = 5
# A sequence whose steps follow each other at least this often is a fixed
# procedure: the LLM calls between the steps only rubber-stamp the next
# one, so the sequence can become one scripted step. Below it the model
# is making a real choice between steps -- the finding is then a repeated
# *shape*, not a script waiting to be extracted.
SCRIPTED_TRANSITION_RATE = 0.9
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

# A cache candidate is worth a summary line only when its repeats carried
# real work to skip. The measure is what the operation returned (and any
# cost of its own), never the operation's name: a result of a few bytes
# never decision-mattered, however often it repeated -- measured on the
# POC's mock agents and real sessions alike, 80 chars cleanly separates
# "the agent read this" from "the agent printed a marker".
MIN_CACHEABLE_RESULT_CHARS = 80


def _pattern_id(o_type: str, *identity) -> str:
    """A stable ID for one finding, so two analyses of two time windows
    can match "pattern A in January" to "pattern A in February" -- the
    spine of the verify-savings loop. Keys on what makes the pattern
    itself (type + nodes / input / trigger), never on when it ran or how
    often: the same habit next month gets the same ID."""
    return hashlib.sha1("|".join((o_type, *(str(i) for i in identity))).encode()).hexdigest()[:12]

# Plain-language meaning of each optimization type, shown next to the name.
OPTIMIZATION_TYPES = {
    "EXTRACT_DETERMINISTIC_SUBSTEP": "The same steps run in the same order again and again -- the next step follows virtually every time, so the LLM calls between them only rubber-stamp it. Turning them into one script removes those calls.",
    "REPEATED_AGENT_PATTERN": "The agent pays a full model call just to choose the next basic step -- the cost is the steering, not the tools. The fix is tuning, not tooling: batch related commands into one call, or route these choices to a cheaper model.",
    "CACHE_RESULT": "The same operation ran with the identical input and gave the same output every time. Reusing the first result skips the repeats. Operations whose name says they change something (create, update, send, edit...) are never suggested.",
    "CONDITIONAL_LLM_BYPASS": "An LLM call made the same choice every time it saw the same starting point and input. A rule could make that choice without calling the model.",
    "REVIEW_HIDDEN_STATE": "A tool returned different results for the identical input -- a flaky tool or a world that changed between calls. Find what changes before automating around it. (LLM calls are exempt: a model choosing differently is expected, not a finding.)",
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
    """Repeated step sequences, in two honest flavors: a fixed procedure
    (the next step follows virtually every time -- extract it into a
    script) vs. a recurring shape the model steers (the steps vary; a
    skill or single command covering the shape removes the round-trips)."""
    out = []
    for p in sequences:
        if p["llm_calls_between_steps"] <= 0:
            continue
        base = {
            # the title shows one real occurrence's actual arguments, so the
            # customer sees what was grepped / sedded / edited -- the tool
            # names alone say nothing about the work
            "title": (
                " → ".join(p["signature"]) if p.get("signature") else " → ".join(p["steps"])
            ),
            "nodes": list(p["steps"]),
            # stable across analyses: the same habit in next month's traffic
            # gets the same ID (keys on the step nodes, not the example's
            # arguments, which vary by task)
            "pattern_id": _pattern_id("SEQUENCE", *p["steps"]),
            "observations": p["occurrences"],
            "saving": p["cost_between_steps"],
            "saving_tokens": p["tokens_between_steps"],
            "saving_tokens_input": p.get("tokens_between_steps_input", 0),
            "saving_tokens_output": p.get("tokens_between_steps_output", 0),
            "saving_llm_calls": p["llm_calls_between_steps"],
            "observed_cost": p["sequence_cost"],
            # one real occurrence, so the customer sees what the pattern
            # actually does -- not just the tool names
            "example": p.get("example"),
        }
        rate = p.get("transition_rate")
        where = f"{p['occurrences']}× in {p['runs']} runs" + (
            f" across {p['sessions']} sessions" if p.get("sessions", 1) > 1 else ""
        )
        if rate is not None and rate >= SCRIPTED_TRANSITION_RATE:
            out.append(
                {
                    **base,
                    "type": "EXTRACT_DETERMINISTIC_SUBSTEP",
                    "action": "Combine these steps into one scripted step",
                    "evidence": (
                        f"{where} · the next step follows "
                        f"{rate:.0%} of the time · {p['llm_calls_between_steps']} LLM calls"
                        + (f" · {p['tokens_between_steps']:,} tokens" if p["tokens_between_steps"] else "")
                        + " between steps"
                    ),
                }
            )
        else:
            follow = f"{rate:.0%} of the time" if rate is not None else "each transition"
            out.append(
                {
                    **base,
                    "type": "REPEATED_AGENT_PATTERN",
                    "action": (
                        "Tune how the agent steps: batch these into one call where possible "
                        "(e.g. one shell command instead of search, then modify, then write), "
                        "or route these low-judgment 'what's next' calls to a cheaper model"
                    ),
                    "evidence": (
                        f"{where} · the model chooses the next step "
                        f"({follow}) · {p['llm_calls_between_steps']} LLM calls"
                        + (f" · {p['tokens_between_steps']:,} tokens" if p["tokens_between_steps"] else "")
                        + " between steps"
                    ),
                }
            )
    return out


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
            # stable across analyses: the same cached call next month keeps
            # its ID (node + what it was called with)
            "pattern_id": _pattern_id("CACHE", c["node"], json.dumps(c["input"], sort_keys=True, default=str)),
            "action": "Reuse the first result, or skip the repeat",
            "evidence": f"{c['count']}× with identical input, same output every time · {c['redundant_calls']} repeats",
            "nodes": [c["node"]],
            "observations": c["count"],
            "saving": c["cost_with_decision"] * c["redundant_calls"] / c["count"],
            "saving_tokens": c["tokens_with_decision"] * c["redundant_calls"] / c["count"],
            "saving_tokens_input": c.get("tokens_with_decision_input", 0) * c["redundant_calls"] / c["count"],
            "saving_tokens_output": c.get("tokens_with_decision_output", 0) * c["redundant_calls"] / c["count"],
            "observed_cost": c["cost_with_decision"],
        }
        for c in calls
        if c["redundant_calls"] > 0
        and (c.get("result_size", 0) >= MIN_CACHEABLE_RESULT_CHARS or c["cost"] > 0)
        and not any(has_side_effect(n) for n in names.get(c["node"], set()) | {c["node"]})
    ]


def _fingerprint_opportunities(stats: list[ActionStats], llm_nodes: set[tuple[str, str]]) -> list[dict]:
    found = []
    for s in stats:
        fp = s.fingerprints
        if not fp:
            continue
        is_llm = (s.agent, s.node) in llm_nodes
        # An LLM call's unexplained variance is the model choosing --
        # expected, not a finding. A *tool* returning different results
        # for the identical input is flakiness or a changing world, and
        # that genuinely matters: fix it before automating around it.
        if not is_llm and fp["variance"].get("factor") == "unexplained":
            found.append(
                {
                    "type": "REVIEW_HIDDEN_STATE",
                    "title": s.node,
                    "pattern_id": _pattern_id("HIDDEN_STATE", s.agent, s.node),
                    "action": "Find what changes the result: a flaky tool, or state the traces don't capture",
                    "evidence": fp["variance"]["explanation"],
                    "nodes": [s.node],
                    "observations": s.occurrences,
                    "saving": None,
                    "observed_cost": s.avg_cost * s.occurrences,
                }
            )
        # Only an LLM call can be replaced by a rule.
        if not is_llm:
            continue
        for g in fp["groups"]:
            # Worth replacing only when the repeated choice cost something --
            # money when the traces record it, tokens otherwise.
            if g["verdict"] != "consistent" or g["observations"] < MIN_BYPASS_OBSERVATIONS:
                continue
            if g["avg_cost"] <= 0 and g.get("avg_tokens", 0) <= 0:
                continue
            found.append(
                {
                    "type": "CONDITIONAL_LLM_BYPASS",
                    "title": s.node,
                    "pattern_id": _pattern_id(
                        "BYPASS", s.node, g["trigger"], json.dumps(g["input"], sort_keys=True, default=str)
                    ),
                    "input": g["input"],
                    "trigger": g["trigger"],
                    "action": f"Replace the LLM call with a rule that returns {g['dominant_output']}",
                    "evidence": f"same choice {g['observations']} of {g['observations']} times after {g['trigger']}",
                    "nodes": [s.node],
                    "observations": g["observations"],
                    "saving": g["avg_cost"] * (g["observations"] - 1),
                    "saving_tokens": g.get("avg_tokens", 0.0) * (g["observations"] - 1),
                    "saving_tokens_input": g.get("avg_tokens_input", 0.0) * (g["observations"] - 1),
                    "saving_tokens_output": g.get("avg_tokens_output", 0.0) * (g["observations"] - 1),
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
    opportunities.sort(
        key=lambda o: (
            o["saving"] is None,
            -(o["saving"] or 0),
            -(o.get("saving_tokens") or 0),
            -o["observed_cost"],
        )
    )
    return opportunities


def deduplicate(opportunities: list[dict]) -> list[dict]:
    """Largest saving first, skipping any opportunity that shares a step
    with one already taken -- the same underlying loop shows up as several
    overlapping findings, and adding them up would double-count the same
    calls. The result is a lower bound: every action counted at most once."""
    has_dollars = any(o["saving"] for o in opportunities)

    def rank(o: dict) -> float:
        return o["saving"] if has_dollars and o["saving"] else (o.get("saving_tokens") or 0)

    per_input = {"CACHE_RESULT", "CONDITIONAL_LLM_BYPASS"}

    def scope(o: dict) -> tuple:
        return (o["type"], json.dumps(o.get("input"), sort_keys=True, default=str), o.get("trigger"))

    def overlaps(a: dict, b: dict) -> bool:
        # two findings touch the same action when they share a node, unless
        # both are per-input findings of one kind for different inputs
        if not set(a["nodes"]) & set(b["nodes"]):
            return False
        return not (a["type"] == b["type"] in per_input and scope(a) != scope(b))

    taken: list[dict] = []
    for o in sorted((o for o in opportunities if rank(o) > 0), key=lambda o: -rank(o)):
        if not any(overlaps(o, t) for t in taken):
            taken.append(o)
    return taken


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

    # One honest total: opportunities overlap (the same loop appears as
    # several sequences), so the deduplicated figure counts each action at
    # most once and is labeled a lower bound.
    dedup = deduplicate(opportunities)
    has_dollars = any(o["saving"] for o in opportunities)
    savings = {
        "cost": sum(o["saving"] or 0 for o in dedup),
        "tokens": sum(o.get("saving_tokens") or 0 for o in dedup),
        "tokens_input": sum(o.get("saving_tokens_input") or 0 for o in dedup),
        "tokens_output": sum(o.get("saving_tokens_output") or 0 for o in dedup),
        "llm_calls": sum(o.get("saving_llm_calls") or 0 for o in dedup),
        "opportunities_counted": len(dedup),
        "opportunities_total": len(opportunities),
        "raw_sum_tokens": sum(o.get("saving_tokens") or 0 for o in opportunities),
    }
    # When the traces record no cost, token savings still mean nothing to a
    # wallet -- so the summary prices them at a few example rates, clearly
    # labeled as estimates.
    scenarios = None
    if not has_dollars and savings["tokens"] > 0:
        scenarios = [
            {
                "label": label,
                "price_input_per_mtok": price_in,
                "price_output_per_mtok": price_out,
                "saving_est": savings["tokens_input"] / 1e6 * price_in
                + savings["tokens_output"] / 1e6 * price_out,
            }
            for label, price_in, price_out in SCENARIOS
        ]
    return {
        "totals": _totals(relevant),
        "opportunities": shown,
        "num_opportunities": len(opportunities),
        "savings": savings,
        "savings_scenarios": scenarios,
        "optimization_types": OPTIMIZATION_TYPES,
        "notes": (
            "Savings are the observed cost of calls that would not have run -- measured from these traces, "
            "not a forecast. When the traces record no cost, savings are in tokens instead "
            "(`saving_tokens`, `saving_llm_calls`). `savings` is deduplicated: opportunities overlap "
            "(the same loop can appear as several findings), so it counts each action at most once and "
            "is a lower bound. Token savings count input and output tokens and include re-sent context: "
            "most of an agent's input tokens are prompt context sent again on every call, which providers "
            "typically serve from cache at a steep discount -- treat token savings as an upper bound on "
            "the bill, and the dollar scenarios as rough estimates at example rates."
        ),
    }
