"""Expected vs. found: what a mock agent was built to contain
(``rawdata_gen.ground_truth.EXPECTED_FINDINGS``) next to the opportunities
the report lists for the same data.

Each mock agent plants specific findings (a cacheable lookup, a rule-like
LLM choice, a scripted chain, hidden state) and some traps the report
should NOT flag. The generator is synthetic, so a match doesn't prove the
profiler works on real agents -- but every miss or false alarm is a
concrete case to look at. Used by the POC (Phase 6) and by the report for
generated data.
"""
from __future__ import annotations

from ingest.trace_schema import TraceEvent
from rawdata_gen.ground_truth import EXPECTED_FINDINGS

from .behavior import behavior_findings, build_behavior
from .mine import ActionStats, mine
from .opportunities import find_opportunities
from .patterns import find_repetition
from .validation import validate

NOTE = (
    "Expected = what each mock agent's generator was written to contain (see "
    "rawdata_gen/ground_truth.py). Found = the opportunities the report lists for the same data. "
    "'Not expected' rows are traps: the report should not flag them. 'Found, not planted' are "
    "findings nobody planted -- often real (two runs that happened to share an input), sometimes "
    "a sign the profiler flags something it shouldn't."
)


def has_expected(workflow: str) -> bool:
    return workflow in EXPECTED_FINDINGS


def _matches(expected: dict, found: dict) -> bool:
    if found["type"] != expected["type"]:
        return False
    if "steps" in expected:
        steps, want = found["title"].split(" → "), expected["steps"]
        return any(steps[i : i + len(want)] == want for i in range(len(steps) - len(want) + 1))
    if "field" in expected and found.get("where", "").split("=", 1)[0] != expected["field"]:
        return False
    return found["title"] == expected["node"]


def compare_workflow(
    events: list[TraceEvent],
    workflow: str,
    ground_truth: dict | None = None,
    stats: list[ActionStats] | None = None,
    repetition: dict | None = None,
    behavior: dict | None = None,
) -> dict:
    """Planted findings for ``workflow`` next to the opportunities the report
    lists for it. ``stats`` / ``repetition`` reuse what the pipeline already
    mined, ``behavior`` the regions; ``ground_truth`` (per-node planted
    determinism) adds the determinism table."""
    stats = stats if stats is not None else mine(events, workflow)
    repetition = repetition if repetition is not None else find_repetition(events, workflow)
    behavior = behavior if behavior is not None else build_behavior(events, workflow)
    found = find_opportunities(events, workflow, repetition, stats) + behavior_findings(behavior)
    rows, matched = [], set()
    for exp in EXPECTED_FINDINGS.get(workflow, []):
        hits = [i for i, o in enumerate(found) if _matches(exp, o)]
        matched.update(hits)
        first = found[hits[0]] if hits else None
        rows.append(
            {
                "workflow": workflow,
                "type": exp["type"],
                "target": " → ".join(exp["steps"]) if "steps" in exp else exp["node"],
                "expected": exp["expect"],
                "found": bool(hits),
                "agrees": bool(hits) == exp["expect"],
                "why": exp["why"],
                "evidence": "; ".join(found[i]["evidence"] for i in hits) if hits else None,
                "saving": first["saving"] if first else None,
            }
        )
    unplanned: dict[tuple[str, str], dict] = {}
    for i, o in enumerate(found):
        # padding copies (``<node>_followup_N``) aren't part of any agent's design
        if i in matched or "_followup_" in o["title"]:
            continue
        target = f"{o['title']} when {o['where']}" if "where" in o else o["title"]
        entry = unplanned.setdefault(
            (o["type"], target),
            {"workflow": workflow, "type": o["type"], "target": target, "count": 0, "saving": 0.0, "evidence": o["evidence"]},
        )
        entry["count"] += 1
        entry["saving"] += o["saving"] or 0.0
    predictions = {f"{s.agent}/{s.node}": s.classification for s in stats}
    determinism = [
        {**row, "workflow": workflow}
        for row in (validate(predictions, ground_truth).per_node if ground_truth else [])
        if "_followup_" not in row["node"]
    ]
    return {"rows": rows, "unplanned": list(unplanned.values()), "determinism": determinism}


def planted_found(rows: list[dict]) -> tuple[int, int]:
    planted = [r for r in rows if r["expected"]]
    return sum(1 for r in planted if r["found"]), len(planted)


def summarize(rows: list[dict], unplanned: list[dict], determinism: list[dict]) -> dict:
    hit, total = planted_found(rows)
    region_hit, region_total = planted_found([r for r in rows if r["type"] == "DETERMINISTIC_REGION"])
    traps = [r for r in rows if not r["expected"]]
    return {
        "planted_found": hit,
        "planted_total": total,
        "false_alarms": sum(1 for r in traps if r["found"]),
        "traps": len(traps),
        "unplanned": len(unplanned),
        "regions_found": region_hit,
        "regions_total": region_total,
        "unplanned_regions": sum(1 for u in unplanned if u["type"] == "DETERMINISTIC_REGION"),
        "determinism_agree": sum(1 for d in determinism if d["truth"] == d["predicted"]),
        "determinism_insufficient": sum(1 for d in determinism if d["outcome"] == "insufficient_evidence"),
        "determinism_nodes": len(determinism),
    }
