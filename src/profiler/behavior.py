"""Behavior: per-action statistics that answer "under what conditions is
this action predictable?" -- cost percentiles, consistency, which models ran
it, which input fields decide its answer, and *regions*: an input-field
value under which the answer never changes while it does elsewhere.

Built from one upload (``build_behavior(events, ...)``) or from every upload
kept in history (``profiler.history``); the same code serves both.

Two notions of "the same answer", kept apart on purpose:

* **consistency** -- the same *input* repeated gives the same output (what
  ``determinism.classify`` judges; needs repeated inputs).
* **regions / deciding fields** -- with ``field=value``, a simple lookup
  rule (a constant, or one other low-cardinality field) gives the answer,
  while no such rule works for the field's other values. Needs no exact
  repeats, only enough observations per value; a region means a rule can
  answer for that slice ("when severity=critical, fix_available=true →
  upgrade, false → mitigate").

Small slices fit a rule by chance, so a region must beat the rest of the
node by a one-sided binomial test (``REGION_MAX_P``), not just look consistent.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable

from ingest.trace_schema import TraceEvent

from .determinism import _input_key, _outcome_key

OUTPUT_MATCHES = ("exact", "decision", "normalized")
"""How two outputs count as "the same": ``exact`` -- the decision and the
result as recorded; ``decision`` -- the decision only (ignores tool output
text); ``normalized`` -- exact after lowercasing, collapsing whitespace and
masking numbers and long ids, so "Ticket 4411 created" equals "ticket 98
created"."""

MAX_FIELD_VALUES = 12
"""Fields with more distinct values (ids, free text) can't be sliced
meaningfully: each value is seen too rarely."""

MIN_OBS_PER_FIELD_VALUE = 5
"""A field is ranked only when it has this many observations per distinct
value on average -- below that, information gain is inflated by chance."""

MIN_REGION_OBS = 15
"""Observations a value needs before it can be a region, and the rest of the
node needs to compare against."""

REGION_MIN_RATE = 0.95
REGION_MIN_REST_VARIANCE = 0.10
"""A region is a value where a lookup rule is right >= 95% of the time while
the field's other values, each with its own best rule, are wrong >= 10%."""

REGION_MAX_P = 0.01
"""One-sided binomial p-value: the chance the slice looks this consistent
if it varied like the rest. Strict, because every field and value of every
node is tested."""

DECIDING_FIELD_MIN_IMPORTANCE = 0.95

_ID_RE = re.compile(r"\b[0-9a-f]{8,}\b|\b[0-9a-f]{8}-[0-9a-f-]{27,}\b")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WS_RE = re.compile(r"\s+")


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = _ID_RE.sub("<id>", value.lower())
    return _WS_RE.sub(" ", _NUM_RE.sub("#", text)).strip()


def output_key_fn(output_match: str) -> Callable[[TraceEvent], tuple]:
    if output_match not in OUTPUT_MATCHES:
        raise ValueError(f"output_match must be one of {', '.join(OUTPUT_MATCHES)}")
    if output_match == "decision":
        return lambda e: (e.decision.selected,) if e.decision else (e.result,)
    if output_match == "normalized":
        return lambda e: (
            _normalize_text(e.decision.selected) if e.decision else None,
            _normalize_text(e.result),
        )
    return _outcome_key


def _value_label(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)


def _output_label(key: tuple) -> str:
    parts = [str(p) for p in key if p is not None]
    return " · ".join(parts) if parts else "(none)"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def _cost(events: list[TraceEvent]) -> dict:
    costs = [e.cost for e in events]
    return {
        "total": round(sum(costs), 6),
        "avg": round(sum(costs) / len(costs), 6),
        "p50": round(_percentile(costs, 0.5), 6),
        "p95": round(_percentile(costs, 0.95), 6),
    }


def _consistency(events: list[TraceEvent], out: Callable) -> dict:
    """Over inputs seen more than once: the share of their observations that
    gave that input's most common output. ``rate`` is None without repeats."""
    by_input: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        by_input[_input_key(e)][out(e)] += 1
    repeated = [c for c in by_input.values() if sum(c.values()) > 1]
    observations = sum(sum(c.values()) for c in repeated)
    agreeing = sum(max(c.values()) for c in repeated)
    return {
        "rate": round(agreeing / observations, 4) if observations else None,
        "repeated_inputs": len(repeated),
        "repeated_observations": observations,
    }


def _entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c)


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _sliceable(values: dict[str, Counter]) -> bool:
    n = sum(sum(c.values()) for c in values.values())
    return 2 <= len(values) <= MAX_FIELD_VALUES and n >= len(values) * MIN_OBS_PER_FIELD_VALUE


def _rule_misses(events: list[TraceEvent], rule: str | None, out: Callable) -> int:
    """Observations a lookup rule gets wrong: the rule answers each value of
    field ``rule`` (or everything, when None) with its most common output."""
    cells: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        cells[_value_label(e.input_state.get(rule)) if rule else ""][out(e)] += 1
    return sum(sum(c.values()) - max(c.values()) for c in cells.values())


def _best_rule(
    events: list[TraceEvent], out: Callable, candidates: list[str], skip: str, guard: bool = True
) -> tuple[int, str | None]:
    """The lookup rule with the fewest misses on ``events``: a constant, or
    one candidate field -- with ``guard``, only fields with enough
    observations per value to not overfit."""
    rules = [None] + [
        f for f in candidates
        if f != skip
        and (not guard or len({_value_label(e.input_state.get(f)) for e in events}) * MIN_OBS_PER_FIELD_VALUE <= len(events))
    ]
    misses, _, rule = min((_rule_misses(events, r, out), i, r) for i, r in enumerate(rules))
    return misses, rule


def _region(field: str, value: str, events: list[TraceEvent], out: Callable, candidates: list[str]) -> dict | None:
    """Is ``field=value`` a region? Inside it, the best lookup rule -- a
    constant, or one other low-cardinality field -- must be right >= 95% of
    the time, while the field's other values, each given *its own* best
    rule, are still wrong >= 10% of the time; and the gap must pass the
    binomial test. Letting the rest pick its own rule keeps plain
    two-field functions ("reject if lint fails, else follow the tests")
    from reading as regions."""
    inside = [e for e in events if _value_label(e.input_state.get(field)) == value]
    rest: dict[str, list[TraceEvent]] = defaultdict(list)
    for e in events:
        if field in e.input_state and _value_label(e.input_state[field]) != value:
            rest[_value_label(e.input_state[field])].append(e)
    rest_n = sum(len(g) for g in rest.values())
    if len(inside) < MIN_REGION_OBS or rest_n < MIN_REGION_OBS:
        return None
    misses, rule = _best_rule(inside, out, candidates, field)
    if 1 - misses / len(inside) < REGION_MIN_RATE:
        return None
    # unguarded: the rest gets every chance to look rule-like too
    rest_variance = sum(_best_rule(g, out, candidates, field, guard=False)[0] for g in rest.values()) / rest_n
    if rest_variance < REGION_MIN_REST_VARIANCE:
        return None
    p_value = _binom_cdf(misses, len(inside), rest_variance)
    if p_value >= REGION_MAX_P:
        return None
    answers: dict[str, Counter] = defaultdict(Counter)
    for e in inside:
        answers[_value_label(e.input_state.get(rule)) if rule else ""][out(e)] += 1
    return {
        "field": field,
        "value": value,
        "where": f"{field}={value}",
        "observations": len(inside),
        "rule_field": rule,
        "rule": {k: _output_label(c.most_common(1)[0][0]) for k, c in sorted(answers.items())},
        "rule_rate": round(1 - misses / len(inside), 4),
        "rest_rate": round(1 - rest_variance, 4),
        "p_value": p_value,
    }


def _fields(events: list[TraceEvent], out: Callable) -> tuple[list[dict], list[dict]]:
    """Rank the input fields by how much of the answer they explain, and find
    regions. Returns ``(fields, regions)``."""
    total_entropy = _entropy(Counter(out(e) for e in events))
    by_field: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for e in events:
        for field, value in e.input_state.items():
            by_field[field][_value_label(value)][out(e)] += 1
    candidates = [f for f, values in by_field.items() if _sliceable(values)]
    fields, regions = [], []
    for field in candidates:
        values = by_field[field]
        n = sum(sum(c.values()) for c in values.values())
        slices = []
        for value, counts in values.items():
            (dominant, hits), = counts.most_common(1)
            slices.append(
                {
                    "value": value,
                    "observations": sum(counts.values()),
                    "dominant_output": _output_label(dominant),
                    "dominant_rate": round(hits / sum(counts.values()), 4),
                }
            )
            region = _region(field, value, events, out, candidates)
            if region:
                regions.append(region)
        slices.sort(key=lambda s: -s["observations"])
        conditional = sum(sum(c.values()) / n * _entropy(c) for c in values.values())
        importance = (total_entropy - conditional) / total_entropy if total_entropy > 0 else 0.0
        fields.append({"field": field, "values": len(values), "importance": round(importance, 4), "slices": slices})
    fields.sort(key=lambda f: -f["importance"])
    return fields, regions


def _by(events: list[TraceEvent], key: Callable, out: Callable) -> list[dict]:
    groups: dict[str, list[TraceEvent]] = defaultdict(list)
    for e in events:
        groups[key(e)].append(e)
    return [
        {"name": name, "observations": len(group), "cost": _cost(group), "consistency": _consistency(group, out)}
        for name, group in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]


def filter_where(events: list[TraceEvent], where: str | None) -> list[TraceEvent]:
    """Keep the events whose input has ``field=value`` (``where`` as the
    report prints it); None keeps everything."""
    if not where:
        return events
    if "=" not in where:
        raise ValueError("where must look like field=value")
    field, value = where.split("=", 1)
    return [e for e in events if field in e.input_state and _value_label(e.input_state[field]) == value]


def build_behavior(
    events: list[TraceEvent],
    workflow: str | None = None,
    output_match: str = "exact",
    where: str | None = None,
) -> dict:
    out = output_key_fn(output_match)
    scoped = filter_where([e for e in events if workflow is None or e.workflow == workflow], where)
    groups: dict[tuple[str, str], list[TraceEvent]] = defaultdict(list)
    for e in scoped:
        groups[(e.agent or "", e.node)].append(e)
    nodes = []
    for (agent, node), group in groups.items():
        outputs = Counter(out(e) for e in group)
        (dominant, hits), = outputs.most_common(1)
        fields, regions = _fields(group, out)
        deciding = next(
            (f["field"] for f in fields if f["importance"] >= DECIDING_FIELD_MIN_IMPORTANCE and len(outputs) > 1),
            None,
        )
        is_llm = any(e.model for e in group)
        for r in regions:
            in_region = filter_where(group, r["where"])
            # observed cost of the LLM calls a rule would answer instead
            r["saving"] = round(sum(e.cost for e in in_region), 6) if is_llm else None
        latencies = [e.latency_ms for e in group]
        nodes.append(
            {
                "agent": agent,
                "node": node,
                "is_llm": is_llm,
                "observations": len(group),
                "runs": len({e.run_id for e in group}),
                "cost": _cost(group),
                "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
                "distinct_outputs": len(outputs),
                "dominant_output": _output_label(dominant),
                "dominant_rate": round(hits / len(group), 4),
                "consistency": _consistency(group, out),
                "models": _by(group, lambda e: e.model, out) if is_llm else [],
                "versions": _by(group, lambda e: e.workflow_version or "(none)", out),
                "fields": fields,
                "deciding_field": deciding,
                "regions": regions,
            }
        )
    nodes.sort(key=lambda n: -n["cost"]["total"])
    regions = [r for n in nodes for r in n["regions"]]
    return {
        "workflow": workflow,
        "where": where,
        "output_match": output_match,
        "num_events": len(scoped),
        "summary": {
            "nodes": len(nodes),
            "regions": len(regions),
            "deciding_fields": sum(1 for n in nodes if n["deciding_field"]),
            "region_saving": round(sum(r["saving"] or 0 for r in regions), 6),
            "versions": len({e.workflow_version for e in scoped}),
            "models": len({e.model for e in scoped if e.model}),
        },
        "nodes": nodes,
    }


def region_rule_text(region: dict) -> str:
    if region["rule_field"] is None:
        return f"always '{next(iter(region['rule'].values()))}'"
    pairs = ", ".join(f"{k} → '{v}'" for k, v in region["rule"].items())
    return f"the answer follows {region['rule_field']} ({pairs})"


def behavior_findings(behavior: dict) -> list[dict]:
    """Regions as findings, in the shape ``expected.compare_workflow`` grades."""
    return [
        {
            "type": "DETERMINISTIC_REGION",
            "title": n["node"],
            "where": r["where"],
            "evidence": (
                f"when {r['where']}: {region_rule_text(r)} -- right {r['rule_rate']:.0%} of "
                f"{r['observations']} times; the same rule is right only {r['rest_rate']:.0%} elsewhere"
            ),
            "saving": r["saving"],
        }
        for n in behavior["nodes"]
        for r in n["regions"]
    ]
