"""Observe, Mine, Score, and Recommend over normalized agent-workflow
traces -- the stages that sit around the B-POP "Profile" step.

Determinism rule (Mine): delegated to ``determinism.classify`` -- a
three-state DETERMINISTIC / NON_DETERMINISTIC / INSUFFICIENT_EVIDENCE
classification built from repeated-input evidence. See that module's
docstring for the full rationale; the short version is that no observed
contradiction is not the same as proof of determinism.

Score rule: a node's risk score is a weighted combination of its
normalized cost (relative to the most expensive node in the same
workflow) and a non-determinism penalty:

    score = cost_weight * cost_component + nondeterminism_weight * nondeterminism_component

``nondeterminism_component`` is 1.0 for NON_DETERMINISTIC, 0.0 for
DETERMINISTIC, and 0.5 for INSUFFICIENT_EVIDENCE -- an explicit "we
don't know yet" midpoint, not a claim of safety.

Recommend rule: a decision table over (classification, cost_component)
per node, described in ``_recommend_for_node``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ingest.trace_schema import TraceEvent

from .determinism import (
    MIN_REPEAT_COVERAGE,
    MIN_REPEATED_INPUTS,
    MIN_REPEATED_OBSERVATIONS,
    Classification,
    DeterminismEvidence,
    classify,
)

COST_WEIGHT = 0.5
NONDETERMINISM_WEIGHT = 0.5
HIGH_COST_THRESHOLD = 0.5

_NONDETERMINISM_COMPONENT = {
    Classification.DETERMINISTIC: 0.0,
    Classification.NON_DETERMINISTIC: 1.0,
    Classification.INSUFFICIENT_EVIDENCE: 0.5,
}


@dataclass
class ActionStats:
    agent: str
    node: str
    occurrences: int
    evidence: DeterminismEvidence
    avg_cost: float
    avg_latency_ms: float
    cost_component: float = 0.0
    score: float = 0.0

    @property
    def classification(self) -> Classification:
        return self.evidence.classification

    @property
    def distinct_outcomes(self) -> list[str]:
        return self.evidence.distinct_outcomes


def observe(events: list[TraceEvent]) -> dict:
    """Stage 1: what's in the upload, with no interpretation yet."""
    agents = sorted({e.agent for e in events if e.agent})
    workflows: dict[str, int] = defaultdict(int)
    runs_per_workflow: dict[str, set[str]] = defaultdict(set)
    for e in events:
        workflows[e.workflow] += 1
        runs_per_workflow[e.workflow].add(e.run_id)

    return {
        "num_events": len(events),
        "num_agents": len(agents),
        "agents": agents,
        "num_workflows": len(workflows),
        "workflows": [
            {"name": name, "num_events": count, "num_runs": len(runs_per_workflow[name])}
            for name, count in sorted(workflows.items(), key=lambda kv: -kv[1])
        ],
    }


def mine(
    events: list[TraceEvent],
    workflow: str,
    *,
    min_repeated_inputs: int = MIN_REPEATED_INPUTS,
    min_repeat_coverage: float = MIN_REPEAT_COVERAGE,
    min_repeated_observations: int = MIN_REPEATED_OBSERVATIONS,
) -> list[ActionStats]:
    """Stage 3: group a workflow's events by (agent, node) and classify
    each node's determinism from repeated-input evidence."""
    relevant = [e for e in events if e.workflow == workflow and e.agent]
    groups: dict[tuple[str, str], list[TraceEvent]] = defaultdict(list)
    for e in relevant:
        groups[(e.agent, e.node)].append(e)

    stats: list[ActionStats] = []
    for (agent, node), group_events in groups.items():
        evidence = classify(
            group_events,
            min_repeated_inputs=min_repeated_inputs,
            min_repeat_coverage=min_repeat_coverage,
            min_repeated_observations=min_repeated_observations,
        )
        avg_cost = sum(e.cost for e in group_events) / len(group_events)
        avg_latency_ms = sum(e.latency_ms for e in group_events) / len(group_events)
        stats.append(
            ActionStats(
                agent=agent,
                node=node,
                occurrences=len(group_events),
                evidence=evidence,
                avg_cost=avg_cost,
                avg_latency_ms=avg_latency_ms,
            )
        )
    return stats


def score(stats: list[ActionStats]) -> list[ActionStats]:
    """Stage 4: attach cost_component and score to each ActionStats,
    normalized against the most expensive node in this set."""
    max_cost = max((s.avg_cost for s in stats), default=0.0)
    for s in stats:
        s.cost_component = (s.avg_cost / max_cost) if max_cost > 0 else 0.0
        nondeterminism_component = _NONDETERMINISM_COMPONENT[s.classification]
        s.score = COST_WEIGHT * s.cost_component + NONDETERMINISM_WEIGHT * nondeterminism_component
    return stats


def _recommend_for_node(s: ActionStats) -> str:
    high_cost = s.cost_component >= HIGH_COST_THRESHOLD
    cost_note = "high-cost" if high_cost else "low-cost"

    if s.classification == Classification.INSUFFICIENT_EVIDENCE:
        return (
            f"'{s.node}' ({s.agent}) is {cost_note} but has insufficient evidence to classify "
            f"determinism ({s.evidence.repeated_inputs} repeated inputs, "
            f"{s.evidence.repeat_coverage:.0%} repeat coverage) — collect more repeated identical "
            "inputs before trusting any cache/pin recommendation for this node."
        )
    if s.classification == Classification.NON_DETERMINISTIC and high_cost:
        return (
            f"'{s.node}' ({s.agent}) is non-deterministic and high-cost "
            f"({len(s.distinct_outcomes)} distinct outcomes) — pin the model/prompt "
            "or cache its decision to cut repeated spend on a step that doesn't reproduce."
        )
    if s.classification == Classification.NON_DETERMINISTIC:
        return (
            f"'{s.node}' ({s.agent}) is non-deterministic but low-cost — monitor for drift, "
            "no immediate action needed."
        )
    if s.classification == Classification.DETERMINISTIC and high_cost:
        return (
            f"'{s.node}' ({s.agent}) is deterministic and high-cost — safe to cache: "
            "the same input always produces the same output, so repeated calls are pure waste."
        )
    return f"'{s.node}' ({s.agent}) is deterministic and low-cost — healthy, no action needed."


def _derivation_for_node(s: ActionStats, max_cost: float) -> str:
    """Spell out, in words and numbers, how this node's score followed
    from the formula — so a repeated score (e.g. the costliest node in a
    workflow always landing on cost_component=1.0) reads as the formula's
    normalization doing its job, not as evidence the input data is static."""
    nondeterminism_component = _NONDETERMINISM_COMPONENT[s.classification]
    if s.classification == Classification.DETERMINISTIC:
        determinism_note = "deterministic, so nondeterminism_component=0.0"
    elif s.classification == Classification.NON_DETERMINISTIC:
        determinism_note = (
            f"non-deterministic ({len(s.distinct_outcomes)} distinct outcomes), "
            "so nondeterminism_component=1.0"
        )
    else:
        determinism_note = (
            "insufficient evidence to classify determinism "
            f"({s.evidence.repeated_inputs} repeated inputs, {s.evidence.repeat_coverage:.0%} repeat "
            "coverage), so nondeterminism_component=0.5 (uncertain, not a safety claim)"
        )
    return (
        f"cost_component = avg_cost / max_cost = {s.avg_cost:.5f} / {max_cost:.5f} = {s.cost_component:.3f} "
        f"(max_cost is the highest avg_cost among this workflow's nodes — the costliest node always "
        f"normalizes to 1.0, regardless of how randomized its raw cost was); "
        f"{determinism_note}; "
        f"score = {COST_WEIGHT} * {s.cost_component:.3f} + {NONDETERMINISM_WEIGHT} * {nondeterminism_component:.1f} "
        f"= {s.score:.3f}."
    )


def recommend(stats: list[ActionStats]) -> list[dict]:
    """Stage 5: one rule-based recommendation per node, worst-scoring first."""
    ordered = sorted(stats, key=lambda s: -s.score)
    max_cost = max((s.avg_cost for s in stats), default=0.0)
    return [
        {
            "agent": s.agent,
            "node": s.node,
            "score": round(s.score, 4),
            "classification": s.classification.value,
            "deterministic": s.classification == Classification.DETERMINISTIC,
            "evidence": s.evidence.to_dict(),
            "message": _recommend_for_node(s),
            "derivation": _derivation_for_node(s, max_cost),
        }
        for s in ordered
    ]
