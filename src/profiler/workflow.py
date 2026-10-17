"""Workflow compilation: from repeated runs to one step order per task,
checked against runs it wasn't built from.

``patterns.py`` finds contiguous step chains that repeat; this module asks
the next question -- *what program do these runs follow?* -- in four steps:

* **Group by task.** One workflow often mixes several kinds of request (a
  Claude Code project sees "fix the tests" and "update the README"). Their
  step orders differ, and mixed together they look like conflicts: ``git
  status`` first in one task and last in another. Runs are grouped by the
  set of steps they use (Jaccard similarity to a group's first run) before
  any order is mined, so a group is one kind of task.
* **Precedence.** Within a group, for each pair of steps seen in the same
  run, the number of runs where one's first occurrence came before the
  other's. A *conflict* is a pair seen in both orders.
* **Order.** The group's *core* steps (in at least half its runs) are put in
  one order. The simple resolver sorts them by mean position in the run.
  When precedence is cyclic (A<B, B<C, C<A by majority) no order satisfies
  every run, and ``order_resolver="asp"`` hands the choice to an optimizer
  (``asp.py``, optional ``clingo``) that minimizes the precedence
  observations the order contradicts. Both are reported side by side.
* **Branches.** Two steps that each appear in at least
  ``BRANCH_MIN_SHARE`` of a group's runs but never in the same run are
  either/or alternatives (e.g. page the owner *or* trigger a rerun).
* **Held-out check.** Every fifth run of a group is held out; the order and
  the constant parameters are learned from the rest and replayed against
  the held-out runs. A run *matches* when it contains every required core
  step (branch steps are optional) and its steps follow the compiled order;
  *pair agreement* is the share of step pairs that ran in the compiled
  order, which still says something when no single order fits.

Parameters: for each core step, an ``input_state`` key is *constant* when
every training run gave it the same value (it can be hard-coded in a
script) and *variable* otherwise (it needs a template slot).

Steps are the events' ``node``; next-step LLM calls (``decision.type ==
"next_action"``) are left out, as ``patterns.py`` does -- they are what a
compiled flow removes, not steps of it. Events must be chronological
within a run.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict

from ingest.trace_schema import TraceEvent

from . import asp
from .patterns import _ROUTER_DECISION_TYPE

GROUP_SIMILARITY = 0.6
MIN_GROUP_RUNS = 3
CORE_STEP_SHARE = 0.5
BRANCH_MIN_SHARE = 0.2
HOLDOUT_EVERY = 5
MIN_HOLDOUT_GROUP_RUNS = 5
MAX_REPORTED_GROUPS = 12
MAX_REPORTED_CONFLICTS = 10
ORDER_RESOLVERS = ("simple", "asp")


def _runs(events: list[TraceEvent], workflow: str) -> list[tuple[str, list[TraceEvent]]]:
    runs: dict[str, list[TraceEvent]] = {}
    for e in events:
        if e.workflow != workflow:
            continue
        if e.decision is not None and e.decision.type == _ROUTER_DECISION_TYPE:
            continue
        runs.setdefault(e.run_id, []).append(e)
    return [(run_id, steps) for run_id, steps in runs.items() if steps]


def _first_steps(steps: list[TraceEvent]) -> dict[str, TraceEvent]:
    """Each node's first event in the run, in run order."""
    first: dict[str, TraceEvent] = {}
    for e in steps:
        first.setdefault(e.node, e)
    return first


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def group_runs(runs: list[tuple[str, list[TraceEvent]]]) -> list[list[tuple[str, list[TraceEvent]]]]:
    """Greedy grouping by step set: each run joins the first group whose
    first run's step set is at least ``GROUP_SIMILARITY`` similar, else
    starts a new group. Deterministic for a given run order."""
    groups: list[tuple[set, list]] = []
    for run in runs:
        nodes = {e.node for e in run[1]}
        for seed, members in groups:
            if _jaccard(nodes, seed) >= GROUP_SIMILARITY:
                members.append(run)
                break
        else:
            groups.append((nodes, [run]))
    return [members for _, members in groups]


def precedence(runs: list[tuple[str, list[TraceEvent]]]) -> dict[tuple[str, str], int]:
    """``(a, b) -> runs where a's first occurrence preceded b's``."""
    prec: dict[tuple[str, str], int] = defaultdict(int)
    for _, steps in runs:
        order = list(_first_steps(steps))
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                prec[a, b] += 1
    return prec


def conflicts(prec: dict[tuple[str, str], int], steps: list[str]) -> list[dict]:
    out = []
    for a, b in itertools.combinations(sorted(steps), 2):
        ab, ba = prec.get((a, b), 0), prec.get((b, a), 0)
        if ab and ba:
            first, second, more, fewer = (a, b, ab, ba) if ab >= ba else (b, a, ba, ab)
            out.append({"first": first, "then": second, "runs": more, "reverse_runs": fewer,
                        "minority_share": round(fewer / (ab + ba), 4)})
    out.sort(key=lambda c: (-c["minority_share"], -c["reverse_runs"], c["first"], c["then"]))
    return out


def cycles(prec: dict[tuple[str, str], int], steps: list[str]) -> list[list[str]]:
    """3-cycles in the majority-precedence graph: no order can follow the
    majority on all three pairs."""
    majority = {(a, b) for a, b in itertools.permutations(steps, 2) if prec.get((a, b), 0) > prec.get((b, a), 0)}
    return [
        [a, b, c] for a, b, c in itertools.permutations(sorted(steps), 3)
        if a < b and a < c and (a, b) in majority and (b, c) in majority and (c, a) in majority
    ]


def evidence_lost(order: list[str], prec: dict[tuple[str, str], int]) -> int:
    """Precedence observations among ``order``'s steps that it contradicts."""
    return sum(prec.get((b, a), 0) for a, b in itertools.combinations(order, 2))


def evidence_total(order: list[str], prec: dict[tuple[str, str], int]) -> int:
    return sum(prec.get((a, b), 0) + prec.get((b, a), 0) for a, b in itertools.combinations(order, 2))


def simple_order(runs: list[tuple[str, list[TraceEvent]]], steps: list[str]) -> list[str]:
    """Steps sorted by mean relative position of their first occurrence."""
    positions: dict[str, list[float]] = defaultdict(list)
    for _, steps_in_run in runs:
        order = list(_first_steps(steps_in_run))
        for i, node in enumerate(order):
            positions[node].append(i / max(len(order) - 1, 1))
    return sorted(steps, key=lambda n: (sum(positions[n]) / len(positions[n]), n))


def core_steps(runs: list[tuple[str, list[TraceEvent]]]) -> list[str]:
    present: dict[str, int] = defaultdict(int)
    for _, steps in runs:
        for node in _first_steps(steps):
            present[node] += 1
    return sorted(n for n, c in present.items() if c / len(runs) >= CORE_STEP_SHARE)


def _value_key(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def parameters(runs: list[tuple[str, list[TraceEvent]]], steps: list[str]) -> dict[str, dict]:
    """Per core step: constant keys (same value in every run that ran the
    step, with that value) and variable keys (with their distinct count)."""
    values: dict[str, dict[str, list]] = {n: defaultdict(list) for n in steps}
    ran: dict[str, int] = defaultdict(int)
    for _, steps_in_run in runs:
        first = _first_steps(steps_in_run)
        for node in steps:
            if node in first:
                ran[node] += 1
                for key, value in first[node].input_state.items():
                    values[node][key].append(value)
    out = {}
    for node in steps:
        constant, variable = {}, {}
        for key, vals in sorted(values[node].items()):
            distinct = {_value_key(v) for v in vals}
            if len(vals) == ran[node] and len(distinct) == 1:
                constant[key] = vals[0]
            else:
                variable[key] = len(distinct)
        out[node] = {"constant": constant, "variable": variable}
    return out


def branches(runs: list[tuple[str, list[TraceEvent]]], steps: list[str]) -> list[list[str]]:
    """Pairs of steps, each in at least ``BRANCH_MIN_SHARE`` of the runs,
    that never ran in the same run."""
    sets = [set(_first_steps(s)) for _, s in runs]
    common = [n for n in sorted(steps) if sum(n in s for s in sets) / len(sets) >= BRANCH_MIN_SHARE]
    return [[a, b] for a, b in itertools.combinations(common, 2) if not any(a in s and b in s for s in sets)]


def _matches(run_steps: list[TraceEvent], order: list[str], required: set[str]) -> bool:
    first = _first_steps(run_steps)
    wanted = set(order)
    seen = [n for n in first if n in wanted]
    return required <= set(first) and seen == [n for n in order if n in first]


def _pair_agreement(held, order: list[str]) -> float | None:
    """Share of step pairs, over held-out runs, that ran in the compiled
    order -- softer than a whole-run match when no single order fits."""
    agree = total = 0
    for _, steps in held:
        pos = {n: i for i, n in enumerate(_first_steps(steps))}
        for a, b in itertools.combinations([n for n in order if n in pos], 2):
            total += 1
            agree += pos[a] < pos[b]
    return round(agree / total, 4) if total else None


def _holdout(train, held, order: list[str], required: set[str], params: dict[str, dict]) -> dict | None:
    if not held:
        return None
    matched = sum(1 for _, steps in held if _matches(steps, order, required))
    checked = right = 0
    for _, steps in held:
        first = _first_steps(steps)
        for node, p in params.items():
            if node not in first:
                continue
            for key, value in p["constant"].items():
                checked += 1
                right += _value_key(first[node].input_state.get(key)) == _value_key(value)
    return {
        "train_runs": len(train),
        "holdout_runs": len(held),
        "matched_runs": matched,
        "match_rate": round(matched / len(held), 4),
        "pair_agreement": _pair_agreement(held, order),
        "constant_params_checked": checked,
        "constant_param_accuracy": round(right / checked, 4) if checked else None,
    }


def _resolve(runs, steps: list[str], prec, resolver: str) -> dict:
    simple = simple_order(runs, steps)
    result = {
        "simple": {"order": simple, "evidence_lost": evidence_lost(simple, prec)},
        "chosen": "simple",
        "order": simple,
    }
    if resolver == "asp":
        solved = asp.resolve_order(steps, prec)
        result["asp"] = solved
        if solved["status"] in ("optimal", "timeout") and solved["evidence_lost"] < result["simple"]["evidence_lost"]:
            result["chosen"] = "asp"
            result["order"] = solved["order"]
        if "order" in solved:
            moved = [n for n, m in zip(simple, solved["order"], strict=True) if n != m]
            result["diff"] = {
                "same_order": not moved,
                "positions_changed": len(moved),
                "evidence_lost_simple": result["simple"]["evidence_lost"],
                "evidence_lost_asp": solved["evidence_lost"],
            }
    return result


def _compile_group(index: int, runs, resolver: str) -> dict:
    held = [r for i, r in enumerate(runs) if len(runs) >= MIN_HOLDOUT_GROUP_RUNS and i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1]
    held_ids = {run_id for run_id, _ in held}
    train = [r for r in runs if r[0] not in held_ids]
    steps = core_steps(train)
    prec = precedence(train)
    all_steps = sorted({n for _, s in train for n in _first_steps(s)})
    found = conflicts(prec, steps)
    alternatives = branches(train, all_steps)
    required = set(steps) - {n for pair in alternatives for n in pair}
    resolved = _resolve(train, steps, prec, resolver)
    params = parameters(train, resolved["order"])
    if "diff" in resolved and held:
        # the same held-out runs replayed against each order
        for name in ("simple", "asp"):
            order = resolved[name]["order"]
            resolved["diff"][f"holdout_match_{name}"] = round(
                sum(_matches(s, order, required) for _, s in held) / len(held), 4
            )
            resolved["diff"][f"holdout_pairs_{name}"] = _pair_agreement(held, order)
    return {
        "group": index,
        "runs": len(runs),
        "run_ids": [run_id for run_id, _ in runs[:5]],
        "core_steps": len(steps),
        "optional_steps": sorted(set(all_steps) - set(steps)),
        "conflicts": found[:MAX_REPORTED_CONFLICTS],
        "num_conflicts": len(found),
        "cycles": cycles(prec, steps)[:5],
        "branches": alternatives,
        "evidence_total": evidence_total(steps, prec),
        **resolved,
        "parameters": params,
        "scriptable": not found and all(not p["variable"] for p in params.values()),
        "required_steps": sorted(required),
        "holdout": _holdout(train, held, resolved["order"], required, params),
    }


def compile_workflow(events: list[TraceEvent], workflow: str, order_resolver: str = "simple") -> dict:
    """Group the workflow's runs by task and compile one step order per
    group. ``order_resolver`` is ``"simple"`` or ``"asp"`` (needs the
    optional ``clingo`` package; reported, not raised, when missing)."""
    if order_resolver not in ORDER_RESOLVERS:
        raise ValueError(f"order_resolver must be one of {', '.join(ORDER_RESOLVERS)}")
    runs = _runs(events, workflow)
    groups = group_runs(runs)
    kept = [g for g in groups if len(g) >= MIN_GROUP_RUNS]
    kept.sort(key=lambda g: -len(g))

    # The same check before grouping, to show what grouping by task removed.
    every_step = sorted({n for _, s in runs for n in _first_steps(s)})
    ungrouped_prec = precedence(runs)
    ungrouped_conflicts = conflicts(ungrouped_prec, every_step)

    compiled = [_compile_group(i + 1, g, order_resolver) for i, g in enumerate(kept[:MAX_REPORTED_GROUPS])]
    return {
        "workflow": workflow,
        "order_resolver": order_resolver,
        "asp": asp.status(),
        "runs": len(runs),
        "groups_found": len(groups),
        "groups_reported": len(compiled),
        "runs_in_small_groups": sum(len(g) for g in groups if len(g) < MIN_GROUP_RUNS),
        "ungrouped": {
            "num_conflicts": len(ungrouped_conflicts),
            "cycles": len(cycles(ungrouped_prec, every_step)),
        },
        "num_conflicts": sum(g["num_conflicts"] for g in compiled),
        "groups": compiled,
        "settings": {
            "group_similarity": GROUP_SIMILARITY,
            "min_group_runs": MIN_GROUP_RUNS,
            "core_step_share": CORE_STEP_SHARE,
            "branch_min_share": BRANCH_MIN_SHARE,
            "holdout_every": HOLDOUT_EVERY,
        },
    }
