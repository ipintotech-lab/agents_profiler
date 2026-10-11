"""Layer 3 of the B-POP pipeline: substitute y*(x) into the upper-level
problem for a single critical region and solve the resulting
single-level LP with PuLP."""
from __future__ import annotations

import numpy as np
import pulp

from .problem import BilevelProblem, LinearConstraint
from .regions import CriticalRegion
from .result import RegionCandidate


def _substitute_y(y_coeffs: dict[str, float], region: CriticalRegion) -> tuple[dict[str, float], float]:
    """Substitute y = y*(x) = region.A @ x + region.b into a linear
    combination of follower variables, returning the resulting linear
    combination of leader variables and a constant term."""
    x_contribution: dict[str, float] = {}
    const = 0.0
    for j, yname in enumerate(region.y_order):
        coeff = y_coeffs.get(yname, 0.0)
        if coeff == 0.0:
            continue
        for i, xname in enumerate(region.x_order):
            a_ji = float(region.A[j, i])
            if a_ji != 0.0:
                x_contribution[xname] = x_contribution.get(xname, 0.0) + coeff * a_ji
        const += coeff * float(region.b[j, 0])
    return x_contribution, const


def _add_constraint(
    model: pulp.LpProblem,
    x_vars: dict[str, pulp.LpVariable],
    c: LinearConstraint,
    region: CriticalRegion,
    label: str,
) -> None:
    """Add ``y_coeffs.y <sense> rhs_constant + x_coeffs.x`` to the model,
    substituting y = y*(x) wherever the constraint references a
    follower (lower-level) variable."""
    x_contribution, const = _substitute_y(c.y_coeffs, region)

    combined: dict[str, float] = dict(x_contribution)
    for xname, coeff in c.x_coeffs.items():
        combined[xname] = combined.get(xname, 0.0) - coeff
    rhs = c.rhs_constant - const

    expr = pulp.lpSum(coeff * x_vars[name] for name, coeff in combined.items() if coeff != 0.0)
    if c.sense == "<=":
        model += expr <= rhs, label
    elif c.sense == ">=":
        model += expr >= rhs, label
    else:
        model += expr == rhs, label


def solve_region(problem: BilevelProblem, region: CriticalRegion) -> RegionCandidate:
    """Solve the leader's problem restricted to a single critical region,
    with y substituted by its explicit function of x on that region."""
    x_names = problem.upper.variable_names()
    model = pulp.LpProblem("bpop_region", pulp.LpMinimize)
    x_vars = {name: pulp.LpVariable(name) for name in x_names}

    # Region membership: E @ x <= f.
    for i in range(region.E.shape[0]):
        expr = pulp.lpSum(float(region.E[i, k]) * x_vars[name] for k, name in enumerate(x_names))
        model += expr <= float(region.f[i, 0]), f"region_bound_{i}"

    # The leader's own constraints (besides variable bounds already
    # implied by the region, but kept for problems with extra coupling).
    # Constraints that reference follower variables have y*(x) substituted in.
    for i, c in enumerate(problem.upper.constraints):
        _add_constraint(model, x_vars, c, region, f"upper_constraint_{i}")

    # Objective: upper.objective_x_coeffs . x + upper.objective_y_coeffs . y*(x).
    obj_terms = [coeff * x_vars[name] for name, coeff in problem.upper.objective_x_coeffs.items()]
    y_contribution, const_term = _substitute_y(problem.upper.objective_y_coeffs, region)
    obj_terms += [coeff * x_vars[name] for name, coeff in y_contribution.items()]

    objective = pulp.lpSum(obj_terms) + const_term
    if problem.upper.sense == "minimize":
        model += objective
    else:
        model += -objective

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return RegionCandidate(region=region, feasible=False, infeasibility_reason=status)

    x_star = {name: var.value() for name, var in x_vars.items()}
    x_vec = [x_star[name] for name in region.x_order]

    y_star_vec = region.evaluate(np.array(x_vec))
    y_star = dict(zip(region.y_order, (float(v) for v in y_star_vec)))

    obj_value = sum(problem.upper.objective_x_coeffs.get(n, 0.0) * x_star[n] for n in x_names)
    obj_value += sum(problem.upper.objective_y_coeffs.get(n, 0.0) * y_star[n] for n in region.y_order)

    return RegionCandidate(region=region, feasible=True, x=x_star, y=y_star, objective=obj_value)
