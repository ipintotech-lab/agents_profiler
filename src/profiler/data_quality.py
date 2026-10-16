"""Which of the attributes the report relies on the trace data actually has.

Trace formats differ in what they record: a Claude Code transcript has
tokens, cost, timing, real tool inputs and output digests; a public
benchmark dump may have no tokens or cost at all, and an anonymized one
only placeholder inputs. Missing attributes don't break the pipeline --
they silently turn into zeros -- so the report says up front which ones
are missing and which of its numbers that empties, instead of letting a
"$0 saving" read as "nothing to save".

When tokens are recorded but cost isn't, cost can be estimated from a
per-token price the user gives (``estimate_missing_costs``); the report
then labels the cost as estimated.
"""
from __future__ import annotations

from ingest.trace_schema import TraceEvent

# Below this share of events having an attribute, it counts as partial.
COMPLETE_SHARE = 0.9

# (key, what it is, which events need it, how to tell it's there, what is empty without it)
_ATTRIBUTES = [
    (
        "tokens",
        "LLM token counts",
        "llm",
        lambda e: e.tokens.input + e.tokens.output > 0,
        "Token totals show 0 and cost can't be estimated from tokens.",
    ),
    (
        "cost",
        "cost ($)",
        "llm",
        lambda e: e.cost > 0,
        "Savings show $0, so opportunities are ranked by repeated tokens and frequency instead of money -- set a price to see dollars.",
    ),
    (
        "llm_latency",
        "LLM call latency",
        "llm",
        lambda e: e.latency_ms > 0,
        "Time spent waiting on the model can't be measured.",
    ),
    (
        "tool_latency",
        "tool call latency",
        "tool",
        lambda e: e.latency_ms > 0,
        "Time spent in tools can't be measured.",
    ),
    (
        "tool_input",
        "real tool inputs",
        "tool",
        lambda e: bool(e.input_state) and "anonymized" not in e.input_state,
        "Repeated calls with the same input can't be recognized, so caching and rule opportunities are under-counted.",
    ),
    (
        "tool_output",
        "tool outputs",
        "tool",
        lambda e: bool(e.result) and e.result != "no_result",
        "Whether the same input gives the same output can't be checked, so determinism and caching can't be judged.",
    ),
    (
        "user_input",
        "the user's request",
        "all",
        lambda e: bool(e.user_input),
        "The report can't show which request started each run.",
    ),
]


def _is_llm(e: TraceEvent) -> bool:
    return bool(e.model) or e.node.startswith("llm_call")


def _is_tool(e: TraceEvent) -> bool:
    return e.tool is not None or e.event_type == "tool_call"


def check_data_quality(events: list[TraceEvent], estimated: dict | None = None) -> dict:
    """Coverage of each attribute the report relies on, and the gaps.

    ``estimated`` (from ``estimate_missing_costs``) marks cost as filled in
    by an estimate rather than recorded."""
    groups = {"llm": [e for e in events if _is_llm(e)], "tool": [e for e in events if _is_tool(e)], "all": events}
    attributes, gaps = [], []
    for key, label, group, present, effect in _ATTRIBUTES:
        pool = groups[group]
        if not pool:
            continue
        share = sum(1 for e in pool if present(e)) / len(pool)
        status = "ok" if share >= COMPLETE_SHARE else "partial" if share > 0 else "missing"
        entry = {"key": key, "label": label, "coverage": round(share, 3), "status": status}
        if key == "cost" and estimated:
            entry["status"] = "estimated"
            entry["effect"] = (
                f"Cost is estimated for {estimated['events']} LLM calls from their tokens "
                f"(${estimated['input_per_mtok']:g} / ${estimated['output_per_mtok']:g} per 1M input / output tokens). "
                "Every input token is priced at the full rate (cached input usually costs less), so treat it as an upper bound."
            )
            gaps.append(entry)
        elif status != "ok":
            entry["effect"] = effect
            gaps.append(entry)
        attributes.append(entry)
    keys = {g["key"]: g["status"] for g in gaps}
    llm_tokens_ok = "tokens" not in keys and groups["llm"]
    return {
        "complete": not gaps,
        "attributes": attributes,
        "gaps": gaps,
        # cost is missing but tokens are there: the user can opt in to an estimate
        "can_estimate_cost": bool(keys.get("cost") in ("missing", "partial") and llm_tokens_ok),
    }


def estimate_missing_costs(
    events: list[TraceEvent], input_per_mtok: float, output_per_mtok: float
) -> tuple[list[TraceEvent], dict]:
    """Fill in cost for LLM calls that have tokens but no cost, at the given
    $ per 1M input / output tokens. Recorded costs are kept as they are."""
    if input_per_mtok < 0 or output_per_mtok < 0:
        raise ValueError("token prices can't be negative")
    filled, count = [], 0
    for e in events:
        if _is_llm(e) and e.cost <= 0 and e.tokens.input + e.tokens.output > 0:
            cost = (e.tokens.input * input_per_mtok + e.tokens.output * output_per_mtok) / 1_000_000
            e = e.model_copy(update={"cost": cost})
            count += 1
        filled.append(e)
    return filled, {"events": count, "input_per_mtok": input_per_mtok, "output_per_mtok": output_per_mtok}
