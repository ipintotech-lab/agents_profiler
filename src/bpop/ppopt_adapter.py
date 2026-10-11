"""Layer 1 adapter: run PPOPT's multi-parametric solver and normalize
its output into our own CriticalRegion model."""
from __future__ import annotations

from ppopt.mp_solvers.solve_mpqp import solve_mpqp

from .parametric import to_mplp
from .problem import BilevelProblem
from .regions import CriticalRegion


def solve_lower_level(problem: BilevelProblem) -> list[CriticalRegion]:
    """Solve the follower's problem parametrically and return the
    explicit solution y*(x) as a list of critical regions."""
    mplp, y_order, x_order = to_mplp(problem)
    solution = solve_mpqp(mplp)

    regions: list[CriticalRegion] = []
    for idx, cr in enumerate(solution.critical_regions):
        regions.append(
            CriticalRegion(
                index=idx,
                x_order=x_order,
                y_order=y_order,
                A=cr.A,
                b=cr.b,
                E=cr.E,
                f=cr.f,
            )
        )
    return regions
