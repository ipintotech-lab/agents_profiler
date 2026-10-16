"""The verification loop: analyze two time windows of traffic, match
findings by their stable ``pattern_id``, and report the *measured* change
per pattern -- the difference between "we predict savings" and "we
measured what changed after you acted".

Both windows are canonicalized before joining (the loader sorts files by
content digest and emits runs in canonical order), so a delta is itself
order-independent: the same two corpora give the same delta in any
upload order.

Join semantics:

- equality on ``pattern_id`` is the fast path: the same habit in both
  windows, with ``variant_id`` explaining whether the chain grew, shrank
  or stayed the same extension.
- a pattern present in only one window is *resolved* (A only: the habit
  stopped) or *appeared* (B only: a new habit) -- unless the
  containment fallback applies: sequence mining can absorb a 2-step
  pattern into a longer chain with a different head (declared residual,
  2/25 on field data), so a vanished pattern whose steps appear
  contiguously inside a new window-B pattern is reported as *absorbed*,
  not resolved.
"""
from __future__ import annotations

from ingest.trace_schema import TraceEvent

from .data_quality import estimate_missing_costs
from .mine import mine, score
from .opportunities import build_summary, find_opportunities
from .patterns import find_repetition


def _window(events: list[TraceEvent], workflow: str, token_prices: tuple[float, float] | None) -> dict:
    """Everything one time window contributes to a delta: the full
    (uncapped) opportunity set, joined by pattern_id, plus the summary's
    deduplicated savings -- the honest per-window headline."""
    if token_prices is not None:
        events, _ = estimate_missing_costs(events, *token_prices)
    repetition = find_repetition(events, workflow)
    stats = score(mine(events, workflow))
    # uncapped: the verification loop must see every pattern, not the
    # report's display cap
    opportunities = find_opportunities(events, workflow, repetition, stats)
    summary = build_summary(events, workflow, repetition, stats)
    return {
        "opportunities": {o["pattern_id"]: o for o in opportunities},
        "savings": summary["savings"],
        "totals": summary["totals"],
    }


def _contains(outer: list, inner: list) -> bool:
    """Whether ``inner`` appears contiguously inside ``outer``."""
    return any(outer[i : i + len(inner)] == inner for i in range(len(outer) - len(inner) + 1))


def _metrics(o: dict, runs: int) -> dict:
    return {
        "occurrences": o["observations"],
        # occurrences per run: a habit's rate, so a drop in absolute count
        # caused by less traffic doesn't read as a fix (window sizes differ)
        "rate": round(o["observations"] / runs, 4) if runs else None,
        "saving": o.get("saving"),
        "saving_tokens": o.get("saving_tokens"),
        "saving_llm_calls": o.get("saving_llm_calls"),
        "variant_id": o.get("variant_id"),
        "nodes": o.get("nodes"),
        "title": o.get("title"),
        "type": o["type"],
    }


def build_delta(
    events_a: list[TraceEvent],
    events_b: list[TraceEvent],
    workflow: str,
    token_prices: tuple[float, float] | None = None,
) -> dict:
    """Measured change per pattern between two time windows of the same
    workflow. Order-independent by construction (both windows arrive
    canonicalized); output sorted by pattern_id."""
    a = _window(events_a, workflow, token_prices)
    b = _window(events_b, workflow, token_prices)

    patterns = []
    resolved, appeared, absorbed = [], [], []
    runs_a, runs_b = a["totals"]["runs"], b["totals"]["runs"]
    for pid in sorted(set(a["opportunities"]) | set(b["opportunities"])):
        oa = a["opportunities"].get(pid)
        ob = b["opportunities"].get(pid)
        if oa is not None and ob is not None:
            grew = None
            if oa.get("variant_id") != ob.get("variant_id"):
                na, nb = len(oa.get("nodes") or []), len(ob.get("nodes") or [])
                grew = "grew" if nb > na else "shrank" if nb < na else "changed shape"
            patterns.append(
                {
                    "pattern_id": pid,
                    "status": grew or "unchanged",
                    "a": _metrics(oa, runs_a),
                    "b": _metrics(ob, runs_b),
                    "change": {
                        "occurrences": ob["observations"] - oa["observations"],
                        "saving": (ob.get("saving") or 0) - (oa.get("saving") or 0),
                        "saving_tokens": (ob.get("saving_tokens") or 0) - (oa.get("saving_tokens") or 0),
                        "saving_llm_calls": (ob.get("saving_llm_calls") or 0) - (oa.get("saving_llm_calls") or 0),
                    },
                }
            )
        elif oa is not None:
            # vanished from window B: absorbed into a longer chain, or
            # genuinely resolved
            host = next(
                (
                    other_pid
                    for other_pid, o in b["opportunities"].items()
                    if other_pid not in a["opportunities"] and _contains(o.get("nodes") or [], oa.get("nodes") or [])
                ),
                None,
            )
            if host is not None:
                absorbed.append(pid)
                patterns.append(
                    {
                        "pattern_id": pid,
                        "status": "absorbed",
                        "absorbed_into": host,
                        "a": _metrics(oa, runs_a),
                        "b": None,
                    }
                )
            else:
                resolved.append(pid)
                patterns.append({"pattern_id": pid, "status": "resolved", "a": _metrics(oa, runs_a), "b": None})
        else:
            appeared.append(pid)
            patterns.append({"pattern_id": pid, "status": "appeared", "a": None, "b": _metrics(ob, runs_b)})

    sav_a, sav_b = a["savings"], b["savings"]
    return {
        "workflow": workflow,
        "windows": {
            "a": {"totals": a["totals"], "savings": sav_a},
            "b": {"totals": b["totals"], "savings": sav_b},
        },
        "summary": {
            "patterns_matched": sum(1 for p in patterns if p["status"] in (None, "unchanged", "grew", "shrank", "changed shape")),
            "patterns_resolved": len(resolved),
            "patterns_absorbed": len(absorbed),
            "patterns_appeared": len(appeared),
            # measured, not modeled: the change in what the two windows'
            # traffic actually spent on avoidable calls. A drop means the
            # fixes landed; a rise means new habits or more traffic -- the
            # per-pattern rows say which.
            "avoidable_change": {
                "cost": (sav_b["cost"] or 0) - (sav_a["cost"] or 0),
                "tokens": (sav_b["tokens"] or 0) - (sav_a["tokens"] or 0),
                "llm_calls": (sav_b["llm_calls"] or 0) - (sav_a["llm_calls"] or 0),
            },
            "note": (
                "Measured from the two uploaded windows, not forecast. 'Avoidable' per window is "
                "the deduplicated lower bound; its change is the measured delta. Resolved = the "
                "pattern is gone from window B; absorbed = it was subsumed into a longer chain "
                "(sequence mining can do this); appeared = new in window B."
            ),
        },
        "patterns": patterns,
    }
