"""Common internal representation of a critical region, independent of
the underlying mp-solver (PPOPT today)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CriticalRegion:
    """A region of parameter (leader-variable) space on which the
    follower's explicit solution y*(x) = A @ x + b is a single affine
    function, valid while E @ x <= f.
    """

    index: int
    x_order: list[str]
    y_order: list[str]
    A: np.ndarray  # y*(x) = A @ x + b
    b: np.ndarray
    E: np.ndarray  # region membership: E @ x <= f
    f: np.ndarray

    def contains(self, x: np.ndarray, tol: float = 1e-7) -> bool:
        return bool(np.all(self.E @ x.reshape(-1, 1) - self.f <= tol))

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Explicit lower-level solution y*(x) for x inside this region."""
        return (self.A @ x.reshape(-1, 1) + self.b).flatten()

    def describe_y_star(self, precision: int = 4) -> str:
        parts = []
        for j, yname in enumerate(self.y_order):
            row = [f"{round(float(self.A[j][i]), precision)}*{xname}" for i, xname in enumerate(self.x_order) if abs(self.A[j][i]) > 1e-9]
            const = round(float(self.b[j, 0]), precision)
            expr = " + ".join(row) if row else ""
            if const or not expr:
                expr = f"{expr} + {const}" if expr else f"{const}"
            parts.append(f"{yname} = {expr}")
        return "; ".join(parts)

    def describe_region(self, precision: int = 4) -> str:
        lines = []
        for i in range(self.E.shape[0]):
            terms = " + ".join(
                f"{round(float(self.E[i, k]), precision)}*{name}"
                for k, name in enumerate(self.x_order)
                if abs(self.E[i, k]) > 1e-9
            )
            lines.append(f"{terms} <= {round(float(self.f[i, 0]), precision)}")
        return "; ".join(lines)
