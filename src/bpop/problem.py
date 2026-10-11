"""Typed problem definition for bi-level parametric optimization (B-POP).

Mirrors the "Suggested Internal API" from the B-POP technical proposal:
a leader (upper level, variables x) picks a decision; a follower (lower
level, variables y) reacts optimally to it; y*(x) is explicit and the
leader's problem is solved per critical region.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Sense = Literal["<=", ">=", "=="]
Objective = Literal["minimize", "maximize"]


class Variable(BaseModel):
    """A decision variable with box bounds."""

    name: str
    lower: float = 0.0
    upper: float


class LinearConstraint(BaseModel):
    """A linear constraint in the canonical B-POP form::

        sum(y_coeffs[v] * v for v in y_vars) <sense> rhs_constant + sum(x_coeffs[v] * v for v in x_vars)

    ``x_coeffs`` lets a lower-level constraint depend on the upper-level
    (parameter) variables, e.g. ``y <= x`` is ``y_coeffs={"y": 1}``,
    ``sense="<="``, ``rhs_constant=0``, ``x_coeffs={"x": 1}``.
    Upper-level-only constraints simply omit ``y_coeffs``.
    """

    y_coeffs: dict[str, float] = Field(default_factory=dict)
    x_coeffs: dict[str, float] = Field(default_factory=dict)
    sense: Sense
    rhs_constant: float = 0.0
    label: str | None = None


class LowerLevelProblem(BaseModel):
    """The follower's problem, parametric in the leader's variables x."""

    variables: list[Variable]
    objective_coeffs: dict[str, float]
    sense: Objective = "maximize"
    constraints: list[LinearConstraint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_objective_vars(self) -> LowerLevelProblem:
        names = {v.name for v in self.variables}
        unknown = set(self.objective_coeffs) - names
        if unknown:
            raise ValueError(f"lower-level objective references unknown variables: {unknown}")
        return self

    def variable_names(self) -> list[str]:
        return [v.name for v in self.variables]

    def all_constraints(self) -> list[LinearConstraint]:
        """Constraints list including each variable's box bounds."""
        out = list(self.constraints)
        for v in self.variables:
            out.append(LinearConstraint(y_coeffs={v.name: 1.0}, sense="<=", rhs_constant=v.upper, label=f"{v.name}<=ub"))
            out.append(LinearConstraint(y_coeffs={v.name: 1.0}, sense=">=", rhs_constant=v.lower, label=f"{v.name}>=lb"))
        return out


class UpperLevelProblem(BaseModel):
    """The leader's problem. Objective may reference both x and y
    (the follower's reaction); y is substituted per critical region
    before this problem is solved as a plain LP."""

    variables: list[Variable]
    objective_x_coeffs: dict[str, float] = Field(default_factory=dict)
    objective_y_coeffs: dict[str, float] = Field(default_factory=dict)
    sense: Objective = "minimize"
    constraints: list[LinearConstraint] = Field(default_factory=list)

    def variable_names(self) -> list[str]:
        return [v.name for v in self.variables]

    def all_constraints(self) -> list[LinearConstraint]:
        out = list(self.constraints)
        for v in self.variables:
            out.append(LinearConstraint(x_coeffs={v.name: 1.0}, sense="<=", rhs_constant=v.upper, label=f"{v.name}<=ub"))
            out.append(LinearConstraint(x_coeffs={v.name: 1.0}, sense=">=", rhs_constant=v.lower, label=f"{v.name}>=lb"))
        return out


class BilevelProblem(BaseModel):
    """A complete bi-level problem: upper level (leader) + lower level (follower)."""

    name: str = "bilevel-problem"
    upper: UpperLevelProblem
    lower: LowerLevelProblem

    @model_validator(mode="after")
    def _check_shared_names(self) -> BilevelProblem:
        x_names = set(self.upper.variable_names())
        y_names = set(self.lower.variable_names())
        if x_names & y_names:
            raise ValueError(f"upper- and lower-level variable names must be disjoint, got overlap: {x_names & y_names}")
        unknown_x = set(self.upper.objective_y_coeffs) - y_names
        if unknown_x:
            raise ValueError(f"upper objective references unknown lower-level variables: {unknown_x}")
        for c in self.lower.constraints:
            unknown = set(c.x_coeffs) - x_names
            if unknown:
                raise ValueError(f"lower-level constraint references unknown upper-level variables: {unknown}")
        for c in self.upper.constraints:
            unknown_y = set(c.y_coeffs) - y_names
            if unknown_y:
                raise ValueError(f"upper-level constraint references unknown lower-level variables: {unknown_y}")
            unknown_x = set(c.x_coeffs) - x_names
            if unknown_x:
                raise ValueError(f"upper-level constraint references unknown upper-level variables: {unknown_x}")
        return self
