"""AI Agent FinOps view: spend across every workflow in the data, how much
of it these traces show to be avoidable, and which savings carry low
observational risk. Nothing is replay-verified yet, so nothing is called
safe or ready.

Built from the same evidence as the report (opportunities, behavior
regions), across all workflows instead of one. Three rules keep the
numbers honest:

- Avoidable spend is a lower bound: the cost of observed calls that an
  opportunity would have removed. Opportunities overlap (a cached call can
  sit inside a scripted sequence), so the headline counts each action at
  most once -- the largest opportunities first, any later one touching an
  already-counted action is left out of the total.
- Risk is a heuristic from evidence size and the kind of change, not a
  replay result; every opportunity says it hasn't been replayed.
- Traces carry no timestamps, so there's no "this month". A monthly figure
  only appears when the caller gives runs per month, and is labelled as a
  projection from the observed cost per run.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from ingest.trace_schema import TraceEvent

from .behavior import build_behavior, region_rule_text
from .mine import mine, observe, score
from .opportunities import OPTIMIZATION_TYPES, find_opportunities, has_side_effect
from .patterns import find_repetition, is_router_call

REGION_TYPE = "DETERMINISTIC_REGION"
TYPES = {
    **OPTIMIZATION_TYPES,
    REGION_TYPE: "Under one input value the action always gives the answer a simple rule gives, while the "
    "rest of its calls still need the LLM. Route that slice to the rule.",
}

STRONG_EVIDENCE = 30
MODERATE_EVIDENCE = 10
TOP_ACTIONS = 15
TOP_WASTE = 5

# The columns of the recommendations board, by observational risk. Every
# opportunity's status stays "opportunity_detected" until replay exists.
RISK_LABELS = {
    "low": "Low observational risk — not replay-verified",
    "medium": "Needs review",
    "high": "Investigate first",
}
STATUS = "opportunity_detected"
# Deprecated names of the risk columns, kept one release for old callers.
LEGACY_STATUS_BY_RISK = {"low": "ready", "medium": "review", "high": "investigate"}


def _region_opportunities(behavior: dict) -> list[dict]:
    return [
        {
            "type": REGION_TYPE,
            "title": n["node"],
            "where": r["where"],
            "action": f"When {r['where']}, skip the LLM: {region_rule_text(r)}",
            "evidence": (
                f"rule right {r['rule_rate']:.0%} of {r['observations']} calls when {r['where']}; "
                f"best simple rule right only {r['rest_rate']:.0%} elsewhere (p={r['p_value']:.2g})"
            ),
            "saving": r["saving"],
            "observed_cost": r["saving"],
            "nodes": [n["node"]],
            "observations": r["observations"],
            "p_value": r["p_value"],
        }
        for n in behavior["nodes"]
        for r in n["regions"]
    ]


def evidence_strength(observations: int) -> str:
    if observations >= STRONG_EVIDENCE:
        return "strong"
    return "moderate" if observations >= MODERATE_EVIDENCE else "thin"


def assess(o: dict) -> tuple[str, list[str]]:
    """Heuristic risk of acting on ``o`` (low / medium / high), with the
    reasons. Not a replay result."""
    n, kind = o["observations"], o["type"]
    if kind == "REVIEW_HIDDEN_STATE":
        return "high", ["results change for the same input: fix the cause before automating anything"]
    if kind == "CACHE_RESULT":
        # side-effecting calls are never suggested for caching
        if n >= MODERATE_EVIDENCE:
            return "low", [f"identical input and output {n} times", "read-only by name"]
        return "medium", [f"only {n} identical calls seen"]
    if kind == "CONDITIONAL_LLM_BYPASS":
        if n >= 2 * MODERATE_EVIDENCE:
            return "low", [f"same choice in all {n} calls"]
        return "medium", [f"same choice in all {n} calls, but few calls", "replaces an LLM decision"]
    if kind == REGION_TYPE:
        if n >= 50 and o["p_value"] < 0.001:
            return "low", [f"{n} calls in the slice", f"p={o['p_value']:.2g}"]
        return "medium", [f"{n} calls in the slice", "replaces an LLM decision on part of its inputs"]
    # EXTRACT_DETERMINISTIC_SUBSTEP
    reasons = []
    if any(has_side_effect(s) for s in o["nodes"]):
        reasons.append("the script must repeat steps that change something")
    if n < 2 * MODERATE_EVIDENCE:
        reasons.append(f"sequence seen {n} times")
    return ("medium", reasons) if reasons else ("low", [f"sequence seen {n} times", "no step changes anything by name"])


def _scope(o: dict) -> tuple:
    return (o["type"], json.dumps(o.get("input"), sort_keys=True, default=str), o.get("trigger"), o.get("where"))


def _overlaps(a: dict, b: dict) -> bool:
    """Whether two opportunities could remove the same calls: they touch a
    common action, unless both are per-input findings (cache / rule
    bypass / region) of one kind for different inputs."""
    if not set(a["touches"]) & set(b["touches"]):
        return False
    per_input = {"CACHE_RESULT", "CONDITIONAL_LLM_BYPASS", REGION_TYPE}
    return not (a["type"] == b["type"] in per_input and _scope(a) != _scope(b))


def select_non_overlapping(opportunities: list[dict]) -> list[dict]:
    """Largest saving first, skipping any that overlaps one already taken."""
    taken: list[dict] = []
    for o in sorted((o for o in opportunities if o["saving"]), key=lambda o: -o["saving"]):
        if not any(o["workflow"] == t["workflow"] and _overlaps(o, t) for t in taken):
            taken.append(o)
    return taken


def _coverage(events: list[TraceEvent]) -> dict:
    llm = [e for e in events if e.model]
    keys = Counter((e.workflow, e.node, json.dumps(e.input_state, sort_keys=True, default=str)) for e in events)
    repeated = sum(c for c in keys.values() if c > 1)
    return {
        "llm_calls": len(llm),
        "llm_calls_with_cost": sum(1 for e in llm if e.cost > 0),
        "llm_calls_with_tokens": sum(1 for e in llm if e.tokens.input + e.tokens.output > 0),
        "calls_with_repeated_input": repeated,
        "calls": len(events),
    }


def _model_mix(events: list[TraceEvent]) -> list[dict]:
    by: dict[str, list[TraceEvent]] = defaultdict(list)
    for e in events:
        if e.model:
            by[e.model].append(e)
    rows = [
        {
            "model": m,
            "calls": len(group),
            "spend": round(sum(e.cost for e in group), 6),
            "avg_cost": round(sum(e.cost for e in group) / len(group), 6),
            "workflows": sorted({e.workflow for e in group}),
        }
        for m, group in by.items()
    ]
    return sorted(rows, key=lambda r: -r["spend"])


def build_finops(events: list[TraceEvent], output_match: str = "exact", runs_per_month: float | None = None) -> dict:
    observed = observe(events)
    if not observed["workflows"]:
        raise ValueError("no workflow found in the uploaded trace file")
    total_spend = sum(e.cost for e in events)
    total_runs = len({(e.workflow, e.run_id) for e in events})
    # a projection scales every $ figure by the same factor: it assumes the
    # observed mix of workflows holds for the month
    factor = runs_per_month / total_runs if runs_per_month and total_runs else None

    opportunities: list[dict] = []
    workflows: list[dict] = []
    actions: list[dict] = []
    for wf in (w["name"] for w in observed["workflows"]):
        relevant = [e for e in events if e.workflow == wf]
        stats = score(mine(events, wf))
        found = find_opportunities(events, wf, find_repetition(events, wf), stats)
        behavior = build_behavior(events, wf, output_match)
        found += _region_opportunities(behavior)
        # A sequence saves the routing calls between its steps, and a cached
        # call's saving includes its share of the routing call that chose it,
        # so both touch the routing actions too.
        routers = sorted({e.node for e in relevant if is_router_call(e)})
        for o in found:
            o["touches"] = o["nodes"] + (routers if o["type"] in ("EXTRACT_DETERMINISTIC_SUBSTEP", "CACHE_RESULT") else [])
            risk, reasons = assess(o)
            o.update(
                workflow=wf,
                risk=risk,
                risk_reasons=reasons,
                evidence_strength=evidence_strength(o["observations"]),
                status=STATUS,
                validation="not replayed",
            )
        # results that change for the same input make reusing a result or
        # fixing a choice unsafe until the cause is found (a region is
        # exempt: the action varying outside the slice is what it describes)
        hidden = {n for o in found if o["type"] == "REVIEW_HIDDEN_STATE" for n in o["nodes"]}
        for o in found:
            if o["type"] in ("CACHE_RESULT", "CONDITIONAL_LLM_BYPASS") and hidden & set(o["nodes"]):
                o["risk"] = "high"
                o["risk_reasons"] = ["the same action gives different results for the same input elsewhere", *o["risk_reasons"]]
        opportunities += found
        spend = sum(e.cost for e in relevant)
        runs = len({e.run_id for e in relevant})
        workflows.append(
            {
                "workflow": wf,
                "agents": sorted({e.agent for e in relevant if e.agent}),
                "runs": runs,
                "calls": len(relevant),
                "llm_calls": sum(1 for e in relevant if e.model),
                "spend": round(spend, 6),
                "llm_spend": round(sum(e.cost for e in relevant if e.model), 6),
                "cost_per_run": round(spend / runs, 6) if runs else 0.0,
                "opportunities": len(found),
                "regions": behavior["summary"]["regions"],
            }
        )
        for n in behavior["nodes"]:
            mine_for_node = [o for o in found if n["node"] in o["nodes"] and o["saving"]]
            best = max(mine_for_node, key=lambda o: o["saving"], default=None)
            actions.append(
                {
                    **{k: n[k] for k in ("agent", "node", "is_llm", "observations", "cost", "latency_ms", "consistency", "models", "deciding_field")},
                    "workflow": wf,
                    "regions": [
                        {k: r[k] for k in ("where", "rule_rate", "rest_rate", "observations", "saving", "p_value")}
                        | {"rule_text": region_rule_text(r)}
                        for r in n["regions"]
                    ],
                    "hidden_state": any(o["type"] == "REVIEW_HIDDEN_STATE" and n["node"] in o["nodes"] for o in found),
                    "best": (
                        {k: best[k] for k in ("type", "action", "saving", "risk", "evidence_strength")} if best else None
                    ),
                }
            )

    selected = select_non_overlapping(opportunities)
    counted = {id(o) for o in selected}
    for o in opportunities:
        o["counted"] = id(o) in counted
    avoidable_by_wf: dict[str, float] = defaultdict(float)
    for o in selected:
        avoidable_by_wf[o["workflow"]] += o["saving"]
    for w in workflows:
        w["avoidable"] = round(avoidable_by_wf[w["workflow"]], 6)
        w["avoidable_share"] = round(w["avoidable"] / w["spend"], 4) if w["spend"] else None
    workflows.sort(key=lambda w: -w["avoidable"])
    avoidable = sum(o["saving"] for o in selected)
    opportunities.sort(key=lambda o: (o["saving"] is None, -(o["saving"] or 0)))
    actions.sort(key=lambda a: -a["cost"]["total"])

    risk_totals = {
        r: {
            "count": sum(1 for o in opportunities if o["risk"] == r),
            "saving": round(sum(o["saving"] or 0 for o in opportunities if o["risk"] == r and o["counted"]), 6),
        }
        for r in RISK_LABELS
    }
    low_risk_saving = risk_totals["low"]["saving"]
    return {
        "overview": {
            "spend": round(total_spend, 6),
            "llm_spend": round(sum(e.cost for e in events if e.model), 6),
            "avoidable": round(avoidable, 6),
            "avoidable_share": round(avoidable / total_spend, 4) if total_spend else None,
            "low_risk_saving": low_risk_saving,
            "ready_saving": low_risk_saving,  # deprecated alias of low_risk_saving
            "workflows": len(workflows),
            "agents": len({e.agent for e in events if e.agent}),
            "runs": total_runs,
            "calls": len(events),
            "opportunities": len(opportunities),
            "counted_opportunities": len(selected),
        },
        "projection": (
            {
                "runs_per_month": runs_per_month,
                "factor": round(factor, 6),
                "spend": round(total_spend * factor, 2),
                "avoidable": round(avoidable * factor, 2),
                "low_risk_saving": round(low_risk_saving * factor, 2),
                "ready_saving": round(low_risk_saving * factor, 2),  # deprecated alias
            }
            if factor
            else None
        ),
        "coverage": _coverage(events),
        "workflows": workflows,
        "top_waste": selected[:TOP_WASTE],
        "actions": actions[:TOP_ACTIONS],
        "num_actions": len(actions),
        "opportunities": opportunities,
        "risk_totals": risk_totals,
        "risk_labels": RISK_LABELS,
        # deprecated: risk_totals under the old ready / review / investigate names
        "status_totals": {LEGACY_STATUS_BY_RISK[r]: t for r, t in risk_totals.items()},
        "deprecated": ["overview.ready_saving", "projection.ready_saving", "status_totals"],
        "models": _model_mix(events),
        "types": TYPES,
        "output_match": output_match,
        "notes": (
            "Avoidable spend is the observed cost of calls an opportunity would have removed, each action "
            "counted once: a lower bound, not a forecast. Risk is a heuristic from evidence size and the kind "
            "of change; nothing here has been replayed against past calls yet, so no finding is verified safe "
            "to deploy."
        ),
    }
