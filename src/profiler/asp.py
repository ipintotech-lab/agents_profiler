"""Optional Answer Set Programming resolver for conflicting step orders.

Install with ``pip install agents-profiler-bpop[asp]`` (adds ``clingo``).
Without it everything else works; ``resolve_order`` reports
``status="unavailable"`` and the simple order is used.

The problem: given ``prec[a, b]`` -- runs where step a came before b --
pick one total order of the steps that contradicts as few observations as
possible. When precedence is cyclic no order satisfies every run, so this
is an optimization (minimum feedback arc set), which counting can't settle.

Encoding (generate / test / optimize):

* choose one direction for every pair of steps,
* the chosen relation must be transitive and acyclic,
* minimize the observations of the opposite direction.

Bounded by ``MAX_STEPS`` (grounding grows with steps cubed) and
``TIME_LIMIT_S``; on timeout the best order found so far is returned with
``status="timeout"``.
"""
from __future__ import annotations

import itertools

try:
    import clingo
except ImportError:  # optional dependency
    clingo = None

MAX_STEPS = 15
TIME_LIMIT_S = 5.0

PROGRAM = """
{ before(X,Y) ; before(Y,X) } = 1 :- step(X), step(Y), X < Y.
before(X,Z) :- before(X,Y), before(Y,Z).
:- before(X,X).
#minimize { W,X,Y : before(X,Y), prec(Y,X,W) }.
#show before/2.
"""


def available() -> bool:
    return clingo is not None


def status() -> dict:
    return {
        "available": available(),
        "version": clingo.__version__ if clingo else None,
        "max_steps": MAX_STEPS,
        "time_limit_s": TIME_LIMIT_S,
    }


def _facts(steps: list[str], prec: dict[tuple[str, str], int]) -> tuple[str, dict[str, str]]:
    # short ids, so any node name (spaces, arrows, colons) stays out of the program text
    names = {f's{i}': s for i, s in enumerate(steps)}
    ids = {s: i for i, s in names.items()}
    lines = [f"step({i})." for i in names]
    for a, b in itertools.permutations(steps, 2):
        w = prec.get((a, b), 0)
        if w:
            lines.append(f"prec({ids[a]},{ids[b]},{w}).")
    return "\n".join(lines), names


def resolve_order(steps: list[str], prec: dict[tuple[str, str], int]) -> dict:
    if not available():
        return {"status": "unavailable", "reason": "clingo is not installed (pip install agents-profiler-bpop[asp])"}
    if len(steps) > MAX_STEPS:
        return {"status": "skipped", "reason": f"{len(steps)} steps is over the solver limit of {MAX_STEPS}"}
    if len(steps) < 2:
        return {"status": "optimal", "order": list(steps), "evidence_lost": 0, "models": 1}

    facts, names = _facts(steps, prec)
    ctl = clingo.Control(["--opt-mode=opt", "--warn=none"])
    ctl.add("base", [], facts + PROGRAM)
    ctl.ground([("base", [])])

    best: dict = {}

    def on_model(model) -> None:
        before = [(str(s.arguments[0]), str(s.arguments[1])) for s in model.symbols(shown=True)]
        wins = {i: 0 for i in names}
        for a, _ in before:
            wins[a] += 1
        order = [names[i] for i in sorted(wins, key=lambda i: -wins[i])]
        best.update(order=order, evidence_lost=int(model.cost[0]) if model.cost else 0)
        best["models"] = best.get("models", 0) + 1

    with ctl.solve(on_model=on_model, async_=True) as handle:
        finished = handle.wait(TIME_LIMIT_S)
        if not finished:
            handle.cancel()
        result = handle.get()
    if not best:
        return {"status": "unsatisfiable" if result.unsatisfiable else "timeout", "reason": "no order found"}
    return {"status": "optimal" if finished else "timeout", **best}
