"""Top-level B-POP API: BPOP.solve(problem) -> BPOPResult.

Orchestrates the full pipeline: parametric lower-level solve ->
per-region upper-level solve -> global comparison across regions,
with boundary-tolerant tie handling.
"""
from __future__ import annotations

from .ppopt_adapter import solve_lower_level
from .problem import BilevelProblem
from .result import BPOPResult, RegionCandidate
from .upper_solver import solve_region

_DEFAULT_TOLERANCE = 1e-6


class BPOP:
    """Namespace matching the proposal's suggested internal API:
    ``result = BPOP.solve(problem)``."""

    @staticmethod
    def solve(problem: BilevelProblem, tolerance: float = _DEFAULT_TOLERANCE) -> BPOPResult:
        critical_regions = solve_lower_level(problem)
        if not critical_regions:
            raise ValueError("no critical regions found for the lower-level problem")

        candidates: list[RegionCandidate] = [solve_region(problem, r) for r in critical_regions]
        feasible = [c for c in candidates if c.feasible]
        if not feasible:
            raise ValueError("no critical region yielded a feasible, bounded leader solution")

        sign = 1.0 if problem.upper.sense == "minimize" else -1.0
        best_value = min(sign * c.objective for c in feasible)
        tied = [c for c in feasible if abs(sign * c.objective - best_value) <= tolerance]
        winner = tied[0]

        return BPOPResult(
            problem_name=problem.name,
            x=winner.x,
            y=winner.y,
            objective=winner.objective,
            winning_region=winner.region,
            critical_regions=critical_regions,
            candidates=candidates,
            tied_regions=[c.region.index for c in tied],
            tolerance=tolerance,
        )
