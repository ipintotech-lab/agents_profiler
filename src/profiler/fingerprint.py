"""Action fingerprints: the logical identity of an action, independent of
which session, trace, or moment it was observed in.

Two observations with the same fingerprint are the same situation from the
agent's point of view, so a different output between them is evidence about
the agent, not about the data. Three nested levels let the report say
*what explains* variance instead of only flagging it:

* **business**  -- node, action type, starting point (``trigger``) and the
  action's own input (``input_state``). "Same situation, same decision?"
* **context**   -- business + the result of the step that triggered the
  action (what the agent had just seen). "Same situation and same supporting
  context?"
* **execution** -- context + model and agent version. "Same situation,
  context and configuration?"

If agreement jumps from business to context, the context explains most of
the variance; from context to execution, the configuration does; if it stays
low at execution, the variance comes from the agent itself or from state the
traces don't carry.

Identity fields only. Instance metadata -- trace/run ids, timestamps,
latency, tokens, cost, and the free-text ``user_input`` -- stays out, or
every observation would be unique. Cost is aggregated over each group
instead.

This layer adds evidence; it doesn't change ``determinism.classify``, which
still owns DETERMINISTIC / NON_DETERMINISTIC / INSUFFICIENT_EVIDENCE.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

from ingest.trace_schema import TraceEvent

from .determinism import _display_value, _outcome_key

FINGERPRINT_SCHEMA_VERSION = 1
"""Bump whenever the fields that take part in a fingerprint change: it's part
of the hashed value, so fingerprints from different schema versions never
compare equal."""

RUN_START = "run_start"
"""Starting point of the first step in a run that carries no explicit trigger."""

ROUTER_DECISION_TYPE = "next_action"
"""LLM calls that only choose the next step; not a starting point themselves."""

MAX_REPORTED_GROUPS = 15
MAX_REPRESENTATIVES = 3
MAX_REPORTED_OUTPUTS = 5
MAX_VALUE_CHARS = 160
"""Display only: values in the report are cut to this length (identity uses
the full values)."""

EXCLUDED_FROM_IDENTITY = (
    "trace_id",
    "run_id",
    "parent",
    "timestamp",
    "latency_ms",
    "tokens",
    "cost",
    "user_input",
)


class FingerprintLevel(str, Enum):
    BUSINESS = "business"
    CONTEXT = "context"
    EXECUTION = "execution"


LEVELS = (FingerprintLevel.BUSINESS, FingerprintLevel.CONTEXT, FingerprintLevel.EXECUTION)


@dataclass
class ActionFingerprint:
    level: FingerprintLevel
    node: str
    action_type: str
    trigger: str
    business_input: dict
    context: dict | None = None
    execution: dict | None = None
    schema_version: int = FINGERPRINT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "level": self.level.value,
            "node": self.node,
            "action_type": self.action_type,
            "trigger": self.trigger,
            "input": self.business_input,
        }
        if self.context is not None:
            d["context"] = self.context
        if self.execution is not None:
            d["execution"] = self.execution
        return d

    @property
    def canonical_value(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    @property
    def fingerprint_id(self) -> str:
        """SHA-256 of ``canonical_value`` (first 16 hex chars)."""
        return hashlib.sha256(self.canonical_value.encode()).hexdigest()[:16]


def is_compatible(a: ActionFingerprint, b: ActionFingerprint) -> bool:
    """Whether two fingerprints may be compared at all: same schema version
    and same level."""
    return a.schema_version == b.schema_version and a.level == b.level


@dataclass
class ActionContext:
    """What an action started from, reconstructed from its run."""

    trigger: str
    trigger_result: str | None


def _is_router(event: TraceEvent) -> bool:
    return event.decision is not None and event.decision.type == ROUTER_DECISION_TYPE


def action_type(event: TraceEvent) -> str:
    if event.decision is not None:
        return event.decision.type
    if event.tool is not None:
        return event.tool.name
    return event.event_type


def reconstruct_contexts(events: list[TraceEvent]) -> dict[int, ActionContext]:
    """For each event (keyed by ``id(event)``): its starting point and the
    result of the step it started from. Events must be chronological within a
    run. An explicit ``event.trigger`` wins; otherwise the starting point is
    the previous step in the run (routing LLM calls skipped), or
    ``run_start``."""
    contexts: dict[int, ActionContext] = {}
    last_step: dict[str, TraceEvent] = {}
    last_result_by_node: dict[str, dict[str, str | None]] = defaultdict(dict)
    for e in events:
        previous = last_step.get(e.run_id)
        if e.trigger is not None:
            # The explicit trigger names a step; its latest result in this run is the context.
            trigger_node = e.trigger.split(" + ")[0].removesuffix(" (error)")
            trigger_result = last_result_by_node[e.run_id].get(trigger_node)
            contexts[id(e)] = ActionContext(trigger=e.trigger, trigger_result=trigger_result)
        elif previous is not None:
            contexts[id(e)] = ActionContext(trigger=previous.node, trigger_result=previous.result)
        else:
            contexts[id(e)] = ActionContext(trigger=RUN_START, trigger_result=None)
        if not _is_router(e):
            last_step[e.run_id] = e
            last_result_by_node[e.run_id][e.node] = e.result
    return contexts


def build_business_fingerprint(event: TraceEvent, ctx: ActionContext) -> ActionFingerprint:
    return ActionFingerprint(
        level=FingerprintLevel.BUSINESS,
        node=event.node,
        action_type=action_type(event),
        trigger=ctx.trigger,
        business_input=event.input_state,
    )


def build_context_fingerprint(event: TraceEvent, ctx: ActionContext) -> ActionFingerprint:
    fp = build_business_fingerprint(event, ctx)
    fp.level = FingerprintLevel.CONTEXT
    fp.context = {"trigger_result": ctx.trigger_result}
    return fp


def build_execution_fingerprint(event: TraceEvent, ctx: ActionContext) -> ActionFingerprint:
    fp = build_context_fingerprint(event, ctx)
    fp.level = FingerprintLevel.EXECUTION
    fp.execution = {"model": event.model, "agent_version": event.workflow_version}
    return fp


_BUILDERS = {
    FingerprintLevel.BUSINESS: build_business_fingerprint,
    FingerprintLevel.CONTEXT: build_context_fingerprint,
    FingerprintLevel.EXECUTION: build_execution_fingerprint,
}

FIELDS_BY_LEVEL = {
    FingerprintLevel.BUSINESS: ["node", "action_type", "trigger", "input"],
    FingerprintLevel.CONTEXT: ["node", "action_type", "trigger", "input", "context.trigger_result"],
    FingerprintLevel.EXECUTION: [
        "node", "action_type", "trigger", "input", "context.trigger_result", "execution.model", "execution.agent_version",
    ],
}


def output_label(event: TraceEvent) -> str:
    selected, result = _outcome_key(event)
    return f"{selected or '-'}/{result or '-'}"


@dataclass
class FingerprintGroup:
    fingerprint: ActionFingerprint
    events: list[TraceEvent] = field(default_factory=list)


def group_by_fingerprint(
    events: list[TraceEvent],
    contexts: dict[int, ActionContext],
    level: FingerprintLevel,
    _cache: dict[tuple[int, FingerprintLevel], ActionFingerprint] | None = None,
) -> dict[str, FingerprintGroup]:
    groups: dict[str, FingerprintGroup] = {}
    for e in events:
        key = (id(e), level)
        fp = _cache.get(key) if _cache is not None else None
        if fp is None:
            fp = _BUILDERS[level](e, contexts[id(e)])
            if _cache is not None:
                _cache[key] = fp
        groups.setdefault(fp.fingerprint_id, FingerprintGroup(fingerprint=fp)).events.append(e)
    return groups


def calculate_group_statistics(events: list[TraceEvent]) -> dict:
    """Output distribution of one fingerprint group. ``dominant_output_rate``
    is ``max(output_count) / observations``; ``agreement_rate`` is the chance
    two observations drawn from the group agree (1.0 when seen once)."""
    outputs = Counter(output_label(e) for e in events)
    n = len(events)
    dominant, dominant_count = outputs.most_common(1)[0]
    agreement = sum(c * (c - 1) for c in outputs.values()) / (n * (n - 1)) if n > 1 else 1.0
    total_cost = sum(e.cost for e in events)
    total_tokens = sum(e.tokens.input + e.tokens.output for e in events)
    total_tokens_input = sum(e.tokens.input for e in events)
    total_tokens_output = sum(e.tokens.output for e in events)
    return {
        "observations": n,
        "sessions": len({e.trace_id for e in events}),
        "runs": len({e.run_id for e in events}),
        "outputs": [{"output": o, "count": c} for o, c in outputs.most_common(MAX_REPORTED_OUTPUTS)],
        "unique_output_count": len(outputs),
        "dominant_output": dominant,
        "dominant_output_count": dominant_count,
        "dominant_output_rate": dominant_count / n,
        "agreement_rate": agreement,
        "verdict": "differs" if len(outputs) > 1 else ("consistent" if n > 1 else "seen_once"),
        "total_cost": total_cost,
        "avg_cost": total_cost / n,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / n,
        "total_tokens_input": total_tokens_input,
        "total_tokens_output": total_tokens_output,
        "avg_tokens_input": total_tokens_input / n,
        "avg_tokens_output": total_tokens_output / n,
    }


def _level_summary(groups: dict[str, FingerprintGroup]) -> dict:
    """Agreement across the groups that were seen more than once:
    observation-weighted dominant-output rate. Single observations can't
    agree or disagree with anything, so they're left out (and counted)."""
    repeated = [g for g in groups.values() if len(g.events) > 1]
    observations = sum(len(g.events) for g in repeated)
    dominant = sum(Counter(output_label(e) for e in g.events).most_common(1)[0][1] for g in repeated)
    return {
        "groups": len(groups),
        "repeated_groups": len(repeated),
        "repeated_observations": observations,
        "differing_groups": sum(1 for g in repeated if len({output_label(e) for e in g.events}) > 1),
        "agreement": dominant / observations if observations else None,
    }


def compare_fingerprint_levels(levels: dict[str, dict]) -> dict:
    """Variance attribution between levels -- explanatory correlation, not
    causal inference."""
    business, context, execution = (levels[lvl.value]["agreement"] for lvl in LEVELS)
    evidence = " (from {} / {} / {} repeated observations)".format(
        *(levels[lvl.value]["repeated_observations"] for lvl in LEVELS)
    )
    if business is None:
        return {"factor": None, "explanation": "No business fingerprint was observed more than once -- nothing to compare yet."}
    if business == 1.0:
        return {"factor": None, "explanation": "Every repeated business fingerprint produced the same output."}
    if context is None or execution is None:
        return {
            "factor": "undetermined",
            "explanation": (
                f"Business agreement is {business:.0%}, but once the triggering step's result (or the "
                "configuration) is added no fingerprint repeats -- the context differs every time, so the "
                "traces can't yet tell whether it explains the variance."
            ),
        }
    gain_context = context - business
    gain_execution = execution - context
    if max(gain_context, gain_execution) < 0.05:
        return {
            "factor": "unexplained",
            "explanation": (
                f"Agreement stays at {business:.0%} even with identical context and configuration -- "
                f"the variance comes from the agent itself or from state the traces don't carry{evidence}."
            ),
        }
    factor = "context" if gain_context >= gain_execution else "execution"
    what = "the triggering step's result" if factor == "context" else "model / agent version"
    return {
        "factor": factor,
        "explanation": (
            f"Business agreement {business:.0%} -> context {context:.0%} -> execution {execution:.0%}: "
            f"{what} explains most of the observed variance{evidence} -- correlation, not proven cause."
        ),
    }


def _short(value: object) -> object:
    value = _display_value(value)
    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
        return value[:MAX_VALUE_CHARS] + "..."
    return value


def _representatives(events: list[TraceEvent]) -> list[dict]:
    reps, seen = [], set()
    for e in events:
        if e.trace_id in seen:
            continue
        seen.add(e.trace_id)
        reps.append(
            {
                "trace_id": e.trace_id,
                "run_id": e.run_id,
                "output": output_label(e),
                "user_input": _short(e.user_input) if e.user_input is not None else None,
            }
        )
        if len(reps) == MAX_REPRESENTATIVES:
            break
    return reps


def fingerprint_node(events: list[TraceEvent], contexts: dict[int, ActionContext]) -> dict:
    """Fingerprint evidence for one (agent, node): per-level summaries, the
    variance explanation, and the business-level groups organized by
    starting point (trigger), each with how it splits at the deeper levels."""
    cache: dict[tuple[int, FingerprintLevel], ActionFingerprint] = {}
    by_level = {lvl: group_by_fingerprint(events, contexts, lvl, cache) for lvl in LEVELS}
    levels = {lvl.value: _level_summary(groups) for lvl, groups in by_level.items()}

    groups = []
    for fp_id, group in by_level[FingerprintLevel.BUSINESS].items():
        fp = group.fingerprint
        splits = {}
        for lvl in (FingerprintLevel.CONTEXT, FingerprintLevel.EXECUTION):
            sub = group_by_fingerprint(group.events, contexts, lvl, cache)
            splits[lvl.value] = _level_summary(sub)
        groups.append(
            {
                "fingerprint_id": fp_id,
                "fingerprint_level": fp.level.value,
                "trigger": fp.trigger,
                "action_type": fp.action_type,
                "input": {k: _short(v) for k, v in fp.business_input.items()},
                **calculate_group_statistics(group.events),
                "splits": splits,
                "representatives": _representatives(group.events),
            }
        )
    # Starting points with differing outputs first, then the most observed.
    groups.sort(key=lambda g: (g["verdict"] != "differs", -g["observations"], g["trigger"]))

    by_trigger: dict[str, list[TraceEvent]] = defaultdict(list)
    for e in events:
        by_trigger[contexts[id(e)].trigger].append(e)
    triggers = [
        {
            "trigger": trigger,
            "distinct_inputs": len({json.dumps(e.input_state, sort_keys=True, default=str) for e in trigger_events}),
            **calculate_group_statistics(trigger_events),
        }
        for trigger, trigger_events in by_trigger.items()
    ]
    triggers.sort(key=lambda t: -t["observations"])

    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "fields": {lvl.value: FIELDS_BY_LEVEL[lvl] for lvl in LEVELS},
        "excluded_from_identity": list(EXCLUDED_FROM_IDENTITY),
        "levels": levels,
        "variance": compare_fingerprint_levels(levels),
        "triggers": triggers,
        "groups": groups[:MAX_REPORTED_GROUPS],
        "num_groups": len(groups),
    }
