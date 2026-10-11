"""Explainable result objects for B-POP."""
from __future__ import annotations

from dataclasses import dataclass, field

from .regions import CriticalRegion


@dataclass
class RegionCandidate:
    """The leader's best solution restricted to a single critical region."""

    region: CriticalRegion
    feasible: bool
    x: dict[str, float] | None = None
    y: dict[str, float] | None = None
    objective: float | None = None
    infeasibility_reason: str | None = None

    def describe(self, precision: int = 4) -> str:
        if not self.feasible:
            return f"CR{self.region.index}: infeasible ({self.infeasibility_reason})"
        x_s = ", ".join(f"{k}={round(v, precision)}" for k, v in self.x.items())
        y_s = ", ".join(f"{k}={round(v, precision)}" for k, v in self.y.items())
        return (
            f"CR{self.region.index} [{self.region.describe_region(precision)}]: "
            f"y*(x) -> {self.region.describe_y_star(precision)}; "
            f"solved {x_s}, {y_s}, objective={round(self.objective, precision)}"
        )


@dataclass
class BPOPResult:
    """The global B-POP solution, with full audit trail."""

    problem_name: str
    x: dict[str, float]
    y: dict[str, float]
    objective: float
    winning_region: CriticalRegion
    critical_regions: list[CriticalRegion]
    candidates: list[RegionCandidate] = field(default_factory=list)
    tied_regions: list[int] = field(default_factory=list)
    tolerance: float = 1e-6

    @property
    def explanation(self) -> str:
        lines = [f"B-POP result for '{self.problem_name}':"]
        x_s = ", ".join(f"{k}={round(v, 6)}" for k, v in self.x.items())
        y_s = ", ".join(f"{k}={round(v, 6)}" for k, v in self.y.items())
        lines.append(f"  optimum: {x_s}, {y_s}, objective={round(self.objective, 6)}")
        lines.append(
            f"  selected region: CR{self.winning_region.index} "
            f"[{self.winning_region.describe_region()}], "
            f"explicit lower-level solution {self.winning_region.describe_y_star()}"
        )
        if self.tied_regions:
            others = ", ".join(f"CR{i}" for i in self.tied_regions if i != self.winning_region.index)
            if others:
                lines.append(
                    f"  note: boundary tie with {others} within tolerance {self.tolerance}; "
                    "both regions agree on the global optimum."
                )
        lines.append("  evidence (all region candidates):")
        for c in self.candidates:
            lines.append(f"    {c.describe()}")
        return "\n".join(lines)
