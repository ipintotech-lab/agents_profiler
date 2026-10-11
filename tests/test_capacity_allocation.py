"""Acceptance-gate regression test: Experiment 1 from the B-POP
validation proposal (capacity allocation, leader/follower).

Pass conditions mirror the proposal's §8.6 success criteria:
solution correctness, critical-region correctness, explicit-solution
correctness, feasibility, boundary handling, and explainability.
"""
import numpy as np
import pytest

from bpop import (
    BPOP,
    BilevelProblem,
    LinearConstraint,
    LowerLevelProblem,
    UpperLevelProblem,
    Variable,
)

TOL = 1e-6


@pytest.fixture
def capacity_allocation_problem() -> BilevelProblem:
    upper = UpperLevelProblem(
        variables=[Variable(name="x", lower=0, upper=10)],
        objective_x_coeffs={"x": 2},
        objective_y_coeffs={"y": -4},
        sense="minimize",
    )
    lower = LowerLevelProblem(
        variables=[Variable(name="y", lower=0, upper=6)],
        objective_coeffs={"y": 1},
        sense="maximize",
        constraints=[
            LinearConstraint(y_coeffs={"y": 1}, sense="<=", rhs_constant=0, x_coeffs={"x": 1}, label="y<=x"),
        ],
    )
    return BilevelProblem(name="capacity-allocation", upper=upper, lower=lower)


def test_solution_correctness(capacity_allocation_problem):
    result = BPOP.solve(capacity_allocation_problem)
    assert result.x["x"] == pytest.approx(6.0, abs=TOL)
    assert result.y["y"] == pytest.approx(6.0, abs=TOL)
    assert result.objective == pytest.approx(-12.0, abs=TOL)


def test_critical_region_bounds(capacity_allocation_problem):
    result = BPOP.solve(capacity_allocation_problem)
    assert len(result.critical_regions) == 2
    bounds = []
    for region in result.critical_regions:
        # sample the region's feasible x-interval by probing the boundary inequalities
        lo = max(
            (float(region.f[i, 0]) / float(region.E[i, 0]))
            for i in range(region.E.shape[0])
            if region.E[i, 0] < 0
        )
        hi = min(
            (float(region.f[i, 0]) / float(region.E[i, 0]))
            for i in range(region.E.shape[0])
            if region.E[i, 0] > 0
        )
        bounds.append((round(lo, 6), round(hi, 6)))
    assert sorted(bounds) == [(0.0, 6.0), (6.0, 10.0)]


def test_explicit_solution_correctness(capacity_allocation_problem):
    result = BPOP.solve(capacity_allocation_problem)
    for region in result.critical_regions:
        lo = max(
            (float(region.f[i, 0]) / float(region.E[i, 0]))
            for i in range(region.E.shape[0])
            if region.E[i, 0] < 0
        )
        hi = min(
            (float(region.f[i, 0]) / float(region.E[i, 0]))
            for i in range(region.E.shape[0])
            if region.E[i, 0] > 0
        )
        midpoint = (lo + hi) / 2

        y_mid = region.evaluate(np.array([midpoint]))[0]
        if hi <= 6.0 + 1e-9:
            assert y_mid == pytest.approx(midpoint, abs=TOL)  # CR1: y*(x) = x
        else:
            assert y_mid == pytest.approx(6.0, abs=TOL)  # CR2: y*(x) = 6


def test_feasibility(capacity_allocation_problem):
    result = BPOP.solve(capacity_allocation_problem)
    for candidate in result.candidates:
        if not candidate.feasible:
            continue
        x = candidate.x["x"]
        y = candidate.y["y"]
        assert 0 - TOL <= x <= 10 + TOL
        assert 0 - TOL <= y <= 6 + TOL
        assert y <= x + TOL


def test_boundary_handling(capacity_allocation_problem):
    result = BPOP.solve(capacity_allocation_problem)
    assert set(result.tied_regions) == {0, 1}
    for candidate in result.candidates:
        assert candidate.objective == pytest.approx(-12.0, abs=TOL)


def test_explainability(capacity_allocation_problem):
    result = BPOP.solve(capacity_allocation_problem)
    explanation = result.explanation
    assert "CR0" in explanation
    assert "CR1" in explanation
    assert "boundary tie" in explanation
    assert "-12" in explanation


def test_upper_level_constraint_on_leader_variable():
    """A pure upper-level constraint (no y_coeffs) must bind directly:
    'x <= 4' is x_coeffs={"x": -1}, rhs_constant=4 under the canonical
    convention (0 <= rhs_constant + x_coeffs.x)."""
    upper = UpperLevelProblem(
        variables=[Variable(name="x", lower=0, upper=10)],
        objective_x_coeffs={"x": 2},
        objective_y_coeffs={"y": -4},
        sense="minimize",
        constraints=[
            LinearConstraint(x_coeffs={"x": -1.0}, sense="<=", rhs_constant=4.0, label="leader_cap_at_4"),
        ],
    )
    lower = LowerLevelProblem(
        variables=[Variable(name="y", lower=0, upper=6)],
        objective_coeffs={"y": 1},
        sense="maximize",
        constraints=[
            LinearConstraint(y_coeffs={"y": 1}, sense="<=", rhs_constant=0, x_coeffs={"x": 1}, label="y<=x"),
        ],
    )
    problem = BilevelProblem(name="capped-leader", upper=upper, lower=lower)
    result = BPOP.solve(problem)
    assert result.x["x"] <= 4 + TOL
    assert result.x["x"] == pytest.approx(4.0, abs=TOL)


def test_upper_level_constraint_on_follower_variable():
    """An upper-level constraint referencing a follower variable (y_coeffs)
    must have y*(x) substituted in per region, not be silently dropped."""
    upper = UpperLevelProblem(
        variables=[Variable(name="x", lower=0, upper=10)],
        objective_x_coeffs={"x": 2},
        objective_y_coeffs={"y": -4},
        sense="minimize",
        constraints=[
            LinearConstraint(y_coeffs={"y": 1.0}, sense="<=", rhs_constant=4.0, label="usage_cap_at_4"),
        ],
    )
    lower = LowerLevelProblem(
        variables=[Variable(name="y", lower=0, upper=6)],
        objective_coeffs={"y": 1},
        sense="maximize",
        constraints=[
            LinearConstraint(y_coeffs={"y": 1}, sense="<=", rhs_constant=0, x_coeffs={"x": 1}, label="y<=x"),
        ],
    )
    problem = BilevelProblem(name="usage-capped", upper=upper, lower=lower)
    result = BPOP.solve(problem)
    assert result.y["y"] <= 4 + TOL
    assert result.y["y"] == pytest.approx(4.0, abs=TOL)


def test_upper_level_constraint_unknown_variable_rejected():
    upper = UpperLevelProblem(
        variables=[Variable(name="x", lower=0, upper=10)],
        objective_x_coeffs={"x": 2},
        objective_y_coeffs={"y": -4},
        sense="minimize",
        constraints=[
            LinearConstraint(y_coeffs={"nonexistent": 1.0}, sense="<=", rhs_constant=4.0),
        ],
    )
    lower = LowerLevelProblem(
        variables=[Variable(name="y", lower=0, upper=6)],
        objective_coeffs={"y": 1},
        sense="maximize",
        constraints=[
            LinearConstraint(y_coeffs={"y": 1}, sense="<=", rhs_constant=0, x_coeffs={"x": 1}, label="y<=x"),
        ],
    )
    with pytest.raises(ValueError, match="unknown lower-level variables"):
        BilevelProblem(name="bad-ref", upper=upper, lower=lower)
