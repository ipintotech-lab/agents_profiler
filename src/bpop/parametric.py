"""Layer 2 of the B-POP pipeline: convert the lower level into a
PPOPT-compatible multi-parametric LP, with the leader's variables x
treated as the parameter theta.
"""
from __future__ import annotations

import numpy as np
from ppopt.mplp_program import MPLP_Program

from .problem import BilevelProblem, LinearConstraint


def _row(coeffs: dict[str, float], order: list[str]) -> np.ndarray:
    return np.array([coeffs.get(name, 0.0) for name in order], dtype=float)


def _normalize_to_le(c: LinearConstraint, y_order: list[str], x_order: list[str]) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (A_row, b_scalar, F_row) such that A_row @ y <= b_scalar + F_row @ theta
    always holds for this constraint, regardless of its original sense."""
    a = _row(c.y_coeffs, y_order)
    f = _row(c.x_coeffs, x_order)
    b = c.rhs_constant
    if c.sense == ">=":
        a, f, b = -a, -f, -b
    return a, b, f


def to_mplp(problem: BilevelProblem) -> tuple[MPLP_Program, list[str], list[str]]:
    """Build the mpLP program for the lower level, parametric in the
    leader's variables.

    Returns (mplp_program, y_order, x_order) where y_order/x_order give
    the column ordering used for the follower's and leader's variables
    respectively, matching the matrices inside ``mplp_program``.
    """
    y_order = problem.lower.variable_names()
    x_order = problem.upper.variable_names()
    n_y, n_x = len(y_order), len(x_order)

    rows_a: list[np.ndarray] = []
    rows_b: list[float] = []
    rows_f: list[np.ndarray] = []
    equality_indices: list[int] = []

    for c in problem.lower.all_constraints():
        a, b, f = _normalize_to_le(c, y_order, x_order)
        rows_a.append(a)
        rows_b.append(b)
        rows_f.append(f)
        if c.sense == "==":
            equality_indices.append(len(rows_a) - 1)

    A = np.vstack(rows_a) if rows_a else np.zeros((0, n_y))
    b = np.array(rows_b, dtype=float).reshape(-1, 1)
    F = np.vstack(rows_f) if rows_f else np.zeros((0, n_x))

    # Lower level: follower maximizes -> PPOPT minimizes, so negate.
    sign = -1.0 if problem.lower.sense == "maximize" else 1.0
    c_vec = (sign * _row(problem.lower.objective_coeffs, y_order)).reshape(-1, 1)

    H = np.zeros((n_y, n_x))  # no cross (x^T H y) term in these problems

    # Parameter (theta = x) box, taken from the leader's own variable bounds.
    rows_at: list[np.ndarray] = []
    rows_bt: list[float] = []
    for v in problem.upper.variables:
        row_ub = np.zeros(n_x)
        row_ub[x_order.index(v.name)] = 1.0
        rows_at.append(row_ub)
        rows_bt.append(v.upper)
        row_lb = np.zeros(n_x)
        row_lb[x_order.index(v.name)] = -1.0
        rows_at.append(row_lb)
        rows_bt.append(-v.lower)
    A_t = np.vstack(rows_at) if rows_at else np.zeros((0, n_x))
    b_t = np.array(rows_bt, dtype=float).reshape(-1, 1)

    program = MPLP_Program(
        A=A,
        b=b,
        c=c_vec,
        H=H,
        A_t=A_t,
        b_t=b_t,
        F=F,
        equality_indices=equality_indices or None,
    )
    return program, y_order, x_order
