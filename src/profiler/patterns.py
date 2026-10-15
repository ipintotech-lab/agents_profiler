"""Repetition mining: find what an agent does over and over, so the report
can say where tokens could be saved and which multi-step sequences could be
combined into one flow.

Two findings:

* **Repeated sequences** -- the same ordered run of operations (e.g.
  ``Bash: git add -> Bash: git commit -> Bash: git push``) recurring across
  runs. When an agent picks each next step with an LLM call (events whose
  ``decision.type`` is ``"next_action"``, as the Claude Code converter emits
  them), those in-between calls are what a combined, scripted flow would
  remove -- their observed tokens and cost are the saving estimate.
* **Repeated calls** -- the same operation with the identical input seen
  more than once. When every repeat gave the same output it's a cache
  candidate (each repeat after the first was redundant); when outputs
  differ the step depends on changing state and can't simply be cached.

Both read events in the order given, which must be chronological within a
run.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from ingest.trace_schema import TraceEvent

from .determinism import _display_value

MIN_SEQUENCE_LENGTH = 2
MAX_SEQUENCE_LENGTH = 6
MIN_SEQUENCE_OCCURRENCES = 3
MAX_REPORTED_PATTERNS = 20
MAX_OVERLAP_FRACTION = 0.5

_ROUTER_DECISION_TYPE = "next_action"


@dataclass
class _Step:
    event: TraceEvent
    # LLM "choose the next step" calls between the previous step and this one.
    glue_cost: float = 0.0
    glue_tokens: int = 0
    glue_calls: int = 0


def _is_router_call(event: TraceEvent) -> bool:
    return event.decision is not None and event.decision.type == _ROUTER_DECISION_TYPE


def _steps_by_run(events: list[TraceEvent]) -> dict[str, list[_Step]]:
    runs: dict[str, list[_Step]] = defaultdict(list)
    pending: dict[str, _Step] = {}
    for e in events:
        if _is_router_call(e):
            glue = pending.setdefault(e.run_id, _Step(event=e))
            glue.glue_cost += e.cost
            glue.glue_tokens += e.tokens.input + e.tokens.output
            glue.glue_calls += 1
            continue
        glue = pending.pop(e.run_id, None)
        step = _Step(event=e)
        if glue is not None:
            step.glue_cost, step.glue_tokens, step.glue_calls = glue.glue_cost, glue.glue_tokens, glue.glue_calls
        runs[e.run_id].append(step)
    return runs


def _step_summary(step: _Step) -> dict:
    return {
        "node": step.event.node,
        "input": {k: _display_value(v) for k, v in step.event.input_state.items()},
        "user_input": _display_value(step.event.user_input) if step.event.user_input is not None else None,
    }


def repeated_sequences(events: list[TraceEvent]) -> list[dict]:
    runs = _steps_by_run(events)
    occurrences: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    for run_id, steps in runs.items():
        nodes = [s.event.node for s in steps]
        for n in range(MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH + 1):
            last_start: dict[tuple[str, ...], int] = {}
            for i in range(len(nodes) - n + 1):
                key = tuple(nodes[i : i + n])
                # Count non-overlapping occurrences only, so "Edit, Edit, Edit"
                # isn't counted as two "Edit, Edit" pairs.
                if key in last_start and i < last_start[key] + n:
                    continue
                last_start[key] = i
                occurrences[key].append((run_id, i))

    frequent = {k: v for k, v in occurrences.items() if len(v) >= MIN_SEQUENCE_OCCURRENCES}

    # Drop a pattern when a longer one containing it occurs just as often --
    # the longer one is the real flow.
    def dominated(key: tuple[str, ...]) -> bool:
        count = len(frequent[key])
        for other, other_occ in frequent.items():
            if len(other) > len(key) and len(other_occ) >= count:
                for i in range(len(other) - len(key) + 1):
                    if other[i : i + len(key)] == key:
                        return True
        return False

    patterns = []
    for key, occ in frequent.items():
        if len(set(key)) == 1 and len(key) > MIN_SEQUENCE_LENGTH:
            continue  # "X, X, X" adds nothing over "X, X"
        if dominated(key):
            continue
        n = len(key)
        # Glue *inside* the pattern: the next-step LLM calls before steps 2..n.
        inner = [step for run_id, i in occ for step in runs[run_id][i + 1 : i + n]]
        saving_cost = sum(s.glue_cost for s in inner)
        saving_tokens = sum(s.glue_tokens for s in inner)
        saving_calls = sum(s.glue_calls for s in inner)
        # Total observed cost of every occurrence (steps + routing between
        # them) -- meaningful for any trace format, even ones that don't mark
        # routing calls.
        sequence_cost = saving_cost + sum(s.event.cost for run_id, i in occ for s in runs[run_id][i : i + n])
        first_run, first_i = occ[0]
        patterns.append(
            {
                "steps": list(key),
                "length": n,
                "occurrences": len(occ),
                "runs": len({run_id for run_id, _ in occ}),
                "llm_calls_between_steps": saving_calls,
                "tokens_between_steps": saving_tokens,
                "cost_between_steps": saving_cost,
                "sequence_cost": sequence_cost,
                "example": [_step_summary(s) for s in runs[first_run][first_i : first_i + n]],
                "_occ": occ,
            }
        )
    patterns.sort(key=lambda p: (-p["cost_between_steps"], -p["sequence_cost"], -p["occurrences"], -p["length"]))

    # Many patterns are shifted/shortened views of the same stretch of work
    # (Read->Edit, Edit->Read->Edit, ...). Keep a pattern only if most of its
    # occurrences cover steps no already-kept pattern covers.
    covered: set[tuple[str, int]] = set()
    kept = []
    for p in patterns:
        spans = [{(run_id, j) for j in range(i, i + p["length"])} for run_id, i in p.pop("_occ")]
        overlapping = sum(1 for span in spans if span & covered)
        if overlapping > len(spans) * MAX_OVERLAP_FRACTION:
            continue
        for span in spans:
            covered |= span
        kept.append(p)
        if len(kept) == MAX_REPORTED_PATTERNS:
            break
    return kept


def repeated_calls(events: list[TraceEvent]) -> list[dict]:
    groups: dict[tuple[str, str], list[TraceEvent]] = defaultdict(list)
    for e in events:
        if _is_router_call(e):
            continue
        groups[(e.node, json.dumps(e.input_state, sort_keys=True, default=str))].append(e)

    calls = []
    for (node, _), group in groups.items():
        if len(group) < 2:
            continue
        outcomes = {(e.decision.selected if e.decision else None, e.result) for e in group}
        same_output = len(outcomes) == 1
        calls.append(
            {
                "node": node,
                "input": {k: _display_value(v) for k, v in group[0].input_state.items()},
                "count": len(group),
                "runs": len({e.run_id for e in group}),
                "distinct_outputs": len(outcomes),
                "same_output": same_output,
                "redundant_calls": len(group) - 1 if same_output else 0,
                "cost": sum(e.cost for e in group),
                "user_inputs": [
                    _display_value(t)
                    for t in dict.fromkeys(e.user_input for e in group if e.user_input is not None)
                ][:3],
            }
        )
    calls.sort(key=lambda c: (-c["redundant_calls"], -c["count"]))
    return calls[:MAX_REPORTED_PATTERNS]


def find_repetition(events: list[TraceEvent], workflow: str) -> dict:
    relevant = [e for e in events if e.workflow == workflow]
    sequences = repeated_sequences(relevant)
    calls = repeated_calls(relevant)
    return {
        "sequences": sequences,
        "calls": calls,
        "notes": (
            "Sequences: the same ordered steps recurring (non-overlapping) across runs; "
            "'between steps' counts the LLM calls that only chose the next step inside the "
            "sequence -- what combining it into one scripted flow would remove. "
            "Calls: the same operation with the identical input more than once; when every "
            "repeat gave the same output, each repeat after the first was redundant (cache candidate)."
        ),
    }
