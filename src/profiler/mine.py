"""Observe, Mine, Score, and Recommend over normalized agent-workflow
traces -- the stages that sit around the B-POP "Profile" step.

Determinism rule (Mine): for a given (agent, node), group occurrences by
their ``input_state``. An action is deterministic if every occurrence of
the same input_state always produced the same (decision.selected, result)
pair; it is non-deterministic if any input_state was seen with more than
one distinct outcome.

Score rule: a node's risk score is a weighted combination of its
normalized cost (relative to the most expensive node in the same
workflow) and a non-determinism penalty (1.0 if non-deterministic, 0.0
otherwise):

    score = cost_weight * cost_component + nondeterminism_weight * nondeterminism_component

Recommend rule: a fixed decision table over (deterministic?, cost_component)
per node, described in ``_recommend_for_node``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from ingest.trace_schema import TraceEvent

COST_WEIGHT = 0.5
NONDETERMINISM_WEIGHT = 0.5
HIGH_COST_THRESHOLD = 0.5


def _outcome_key(event: TraceEvent) -> tuple[str | None, str | None]:
    return (event.decision.selected if event.decision else None, event.result)


def _input_key(event: TraceEvent) -> str:
    return json.dumps(event.input_state, sort_keys=True)


@dataclass
class ActionStats:
    agent: str
    node: str
    occurrences: int
    deterministic: bool
    distinct_outcomes: list[str]
    avg_cost: float
    avg_latency_ms: float
    cost_component: float = 0.0
    score: float = 0.0


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


def mine(events: list[TraceEvent], workflow: str) -> list[ActionStats]:
    """Stage 3: group a workflow's events by (agent, node) and classify
    each as deterministic or not."""
    relevant = [e for e in events if e.workflow == workflow and e.agent]
    groups: dict[tuple[str, str], list[TraceEvent]] = defaultdict(list)
    for e in relevant:
        groups[(e.agent, e.node)].append(e)

    stats: list[ActionStats] = []
    for (agent, node), group_events in groups.items():
        outcomes_by_input: dict[str, set[tuple[str | None, str | None]]] = defaultdict(set)
        for e in group_events:
            outcomes_by_input[_input_key(e)].add(_outcome_key(e))
        deterministic = all(len(outcomes) <= 1 for outcomes in outcomes_by_input.values())
        distinct_outcomes = sorted(
            {outcome for outcomes in outcomes_by_input.values() for outcome in outcomes},
            key=lambda o: (o[0] or "", o[1] or ""),
        )
        avg_cost = sum(e.cost for e in group_events) / len(group_events)
        avg_latency_ms = sum(e.latency_ms for e in group_events) / len(group_events)
        stats.append(
            ActionStats(
                agent=agent,
                node=node,
                occurrences=len(group_events),
                deterministic=deterministic,
                distinct_outcomes=[f"{sel or '-'}/{res or '-'}" for sel, res in distinct_outcomes],
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
        nondeterminism_component = 0.0 if s.deterministic else 1.0
        s.score = COST_WEIGHT * s.cost_component + NONDETERMINISM_WEIGHT * nondeterminism_component
    return stats


def _recommend_for_node(s: ActionStats) -> str:
    high_cost = s.cost_component >= HIGH_COST_THRESHOLD
    if not s.deterministic and high_cost:
        return (
            f"'{s.node}' ({s.agent}) is non-deterministic and high-cost "
            f"({len(s.distinct_outcomes)} distinct outcomes) — pin the model/prompt "
            "or cache its decision to cut repeated spend on a step that doesn't reproduce."
        )
    if not s.deterministic:
        return (
            f"'{s.node}' ({s.agent}) is non-deterministic but low-cost — monitor for drift, "
            "no immediate action needed."
        )
    if s.deterministic and high_cost:
        return (
            f"'{s.node}' ({s.agent}) is deterministic and high-cost — safe to cache: "
            "the same input always produces the same output, so repeated calls are pure waste."
        )
    return f"'{s.node}' ({s.agent}) is deterministic and low-cost — healthy, no action needed."


def recommend(stats: list[ActionStats]) -> list[dict]:
    """Stage 5: one rule-based recommendation per node, worst-scoring first."""
    ordered = sorted(stats, key=lambda s: -s.score)
    return [
        {
            "agent": s.agent,
            "node": s.node,
            "score": round(s.score, 4),
            "deterministic": s.deterministic,
            "message": _recommend_for_node(s),
        }
        for s in ordered
    ]
