"""Grid-search calibration of the determinism classifier's evidence
thresholds against known synthetic ground truth.

``classify()`` (see ``determinism.py``) only has three tunable
thresholds -- ``min_repeated_inputs``, ``min_repeat_coverage``,
``min_repeated_observations`` -- and they only affect the
DETERMINISTIC/INSUFFICIENT_EVIDENCE boundary: NON_DETERMINISTIC is
assigned the instant any repeated input shows two outcomes, regardless
of threshold choice. A false negative (predicted DETERMINISTIC, truth
NON_DETERMINISTIC) happens when the sample never happened to repeat the
one input that would reveal the contradiction; raising the thresholds
means more repeated evidence is required before committing to
DETERMINISTIC, trading false negatives against a higher
insufficient-evidence rate. This module searches that tradeoff against
labeled synthetic data -- the generator's known ``determinism_rate`` per
node -- instead of guessing at it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ingest.trace_schema import TraceEvent

from .determinism import (
    MIN_REPEAT_COVERAGE,
    MIN_REPEATED_INPUTS,
    MIN_REPEATED_OBSERVATIONS,
)
from .mine import mine
from .validation import ValidationReport, validate

DEFAULT_MIN_REPEATED_INPUTS_GRID: tuple[int, ...] = (3, 5, 10, 15, 20)
DEFAULT_MIN_REPEAT_COVERAGE_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
DEFAULT_MIN_REPEATED_OBSERVATIONS_GRID: tuple[int, ...] = (10, 20, 30, 40)


@dataclass
class ThresholdCandidate:
    min_repeated_inputs: int
    min_repeat_coverage: float
    min_repeated_observations: int
    report: ValidationReport

    def to_dict(self) -> dict:
        return {
            "min_repeated_inputs": self.min_repeated_inputs,
            "min_repeat_coverage": self.min_repeat_coverage,
            "min_repeated_observations": self.min_repeated_observations,
            "false_negatives": self.report.false_negatives,
            "false_positives": self.report.false_positives,
            "accuracy": self.report.accuracy,
            "coverage": round(self.report.coverage, 4),
            "insufficient_evidence_rate": round(self.report.insufficient_evidence_rate, 4),
        }


def _rank_key(candidate: ThresholdCandidate) -> tuple:
    """Fewest false negatives first (missing real non-determinism is the
    costliest mistake this classifier can make), then highest accuracy,
    then lowest insufficient-evidence rate (prefer being decisive)."""
    r = candidate.report
    return (
        r.false_negatives,
        -(r.accuracy if r.accuracy is not None else 0.0),
        r.insufficient_evidence_rate,
    )


def _evaluate(
    events: list[TraceEvent],
    workflow: str,
    ground_truth: dict[str, dict],
    min_repeated_inputs: int,
    min_repeat_coverage: float,
    min_repeated_observations: int,
) -> ThresholdCandidate:
    stats = mine(
        events,
        workflow,
        min_repeated_inputs=min_repeated_inputs,
        min_repeat_coverage=min_repeat_coverage,
        min_repeated_observations=min_repeated_observations,
    )
    predictions = {f"{s.agent}/{s.node}": s.classification for s in stats}
    report = validate(predictions, ground_truth)
    return ThresholdCandidate(
        min_repeated_inputs=min_repeated_inputs,
        min_repeat_coverage=min_repeat_coverage,
        min_repeated_observations=min_repeated_observations,
        report=report,
    )


def calibrate_determinism_thresholds(
    events: list[TraceEvent],
    workflow: str,
    ground_truth: dict[str, dict],
    *,
    min_repeated_inputs_grid: tuple[int, ...] = DEFAULT_MIN_REPEATED_INPUTS_GRID,
    min_repeat_coverage_grid: tuple[float, ...] = DEFAULT_MIN_REPEAT_COVERAGE_GRID,
    min_repeated_observations_grid: tuple[int, ...] = DEFAULT_MIN_REPEATED_OBSERVATIONS_GRID,
    top_n: int = 5,
) -> dict:
    """Grid-search the three determinism-classification thresholds
    against ``ground_truth`` and report the best-scoring combination plus
    a leaderboard, alongside how the module's current hardcoded defaults
    score on the same data -- so a reader can see whether the shipped
    defaults are already optimal for this dataset or should change.
    """
    candidates = [
        _evaluate(events, workflow, ground_truth, min_inputs, min_coverage, min_observations)
        for min_inputs in min_repeated_inputs_grid
        for min_coverage in min_repeat_coverage_grid
        for min_observations in min_repeated_observations_grid
    ]
    candidates.sort(key=_rank_key)

    current = _evaluate(
        events, workflow, ground_truth, MIN_REPEATED_INPUTS, MIN_REPEAT_COVERAGE, MIN_REPEATED_OBSERVATIONS
    )
    best = candidates[0]

    return {
        "current_defaults": current.to_dict(),
        "best_found": best.to_dict(),
        "improves_on_current": _rank_key(best) < _rank_key(current),
        "leaderboard": [c.to_dict() for c in candidates[:top_n]],
    }
