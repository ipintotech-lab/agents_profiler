"""The report's summary: totals for the analyzed workflow and a ranked list
of savings opportunities built from the evidence the other stages already
collected -- repeated sequences, repeated calls, and fingerprint groups.

Each opportunity names one optimization type, what to do in plain words,
the evidence behind it, and the observed cost it would remove. Savings are
the cost of the observed calls that would not have run, not a forecast.
"""
from __future__ import annotations

from ingest.trace_schema import TraceEvent

from .mine import ActionStats

MAX_PER_TYPE = 5
MIN_BYPASS_OBSERVATIONS = 3

# Plain-language meaning of each optimization type, shown next to the name.
OPTIMIZATION_TYPES = {
    "EXTRACT_DETERMINISTIC_SUBSTEP": "The same steps run in the same order again and again. Turning them into one script removes the LLM calls that pick each next step.",
    "CACHE_RESULT": "The same operation ran with the identical input and gave the same output every time. Reusing the first result skips the repeats.",
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


def _cache_opportunities(calls: list[dict]) -> list[dict]:
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


def build_summary(events: list[TraceEvent], workflow: str, repetition: dict, stats: list[ActionStats]) -> dict:
    relevant = [e for e in events if e.workflow == workflow]
    llm_nodes = {(e.agent, e.node) for e in relevant if e.model}
    opportunities = (
        _sequence_opportunities(repetition["sequences"])
        + _cache_opportunities(repetition["calls"])
        + _fingerprint_opportunities(stats, llm_nodes)
    )
    opportunities.sort(key=lambda o: (o["saving"] is None, -(o["saving"] or 0), -o["observed_cost"]))
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
