"""Three-state determinism classification, built from repeated-input
evidence rather than absence of contradiction.

Core design principle:

    Absence of observed non-determinism is NOT evidence of determinism.

A node is classified DETERMINISTIC only once enough *repeated* identical
canonical inputs have been observed and every one of them produced the
same outcome. A node with a thousand observations but no repeated input
-- however much data that is -- stays INSUFFICIENT_EVIDENCE, because no
input was ever tested twice.

A single repeated input that produced two different outcomes is direct,
sufficient evidence of non-determinism regardless of how small the
sample is.

Thresholds are explicit module constants (not hidden magic numbers) and
can be overridden per call for experimentation/testing.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from ingest.trace_schema import TraceEvent

# --- Evidence thresholds -----------------------------------------------
# A node earns DETERMINISTIC only once ALL three hold. These are
# intentionally visible/documented/testable rather than buried in logic.
#
# P5 calibration: grid-searched against several synthetic ground-truth
# datasets via profiler/calibration.py. (3, 0.1, 10) tied-or-beat the
# previous defaults (10, 0.3, 30) on every dataset tried -- same 0 false
# negatives/positives, but roughly double the classified-node coverage --
# so it was adopted as the MVP default. This is a grid-search optimum on
# the datasets tried, not a proof it's optimal everywhere; if a workflow's
# traces start producing false negatives, re-run the calibrator against
# that workflow's own ground truth before raising these back up.
MIN_REPEATED_INPUTS = 3
"""Minimum number of *distinct* canonical inputs that must each have
been observed more than once."""

MIN_REPEAT_COVERAGE = 0.1
"""Minimum fraction of all observations that belong to a repeated input:
``repeated_observations / occurrences``."""

MIN_REPEATED_OBSERVATIONS = 10
"""Minimum total number of observations belonging to repeated inputs."""


class Classification(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    NON_DETERMINISTIC = "NON_DETERMINISTIC"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


def _outcome_key(event: TraceEvent) -> tuple[str | None, str | None]:
    return (event.decision.selected if event.decision else None, event.result)


def _input_key(event: TraceEvent) -> str:
    """Canonical key for an event's input_state -- order-independent."""
    return json.dumps(event.input_state, sort_keys=True)


@dataclass
class DeterminismEvidence:
    occurrences: int
    unique_inputs: int
    repeated_inputs: int
    repeated_observations: int
    repeat_coverage: float
    conflicting_inputs: int
    consistent_repeated_inputs: int
    determinism_confidence: float
    classification: Classification
    reason: str
    distinct_outcomes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "occurrences": self.occurrences,
            "unique_inputs": self.unique_inputs,
            "repeated_inputs": self.repeated_inputs,
            "repeated_observations": self.repeated_observations,
            "repeat_coverage": round(self.repeat_coverage, 4),
            "conflicting_inputs": self.conflicting_inputs,
            "consistent_repeated_inputs": self.consistent_repeated_inputs,
            "determinism_confidence": round(self.determinism_confidence, 4),
            "classification": self.classification.value,
            "reason": self.reason,
            "distinct_outcomes": self.distinct_outcomes,
        }


def classify(
    events: list[TraceEvent],
    *,
    min_repeated_inputs: int = MIN_REPEATED_INPUTS,
    min_repeat_coverage: float = MIN_REPEAT_COVERAGE,
    min_repeated_observations: int = MIN_REPEATED_OBSERVATIONS,
) -> DeterminismEvidence:
    """Classify one (agent, node) group's events into DETERMINISTIC /
    NON_DETERMINISTIC / INSUFFICIENT_EVIDENCE, carrying the repeated-input
    evidence the call was based on.

    ``repeat_coverage`` is defined as ``repeated_observations /
    occurrences`` -- the fraction of all observations that come from an
    input seen more than once.
    """
    outcomes_by_input: dict[str, set[tuple[str | None, str | None]]] = defaultdict(set)
    counts_by_input: dict[str, int] = defaultdict(int)
    for e in events:
        key = _input_key(e)
        outcomes_by_input[key].add(_outcome_key(e))
        counts_by_input[key] += 1

    occurrences = len(events)
    unique_inputs = len(counts_by_input)
    repeated_input_keys = [k for k, c in counts_by_input.items() if c >= 2]
    repeated_inputs = len(repeated_input_keys)
    repeated_observations = sum(counts_by_input[k] for k in repeated_input_keys)
    repeat_coverage = (repeated_observations / occurrences) if occurrences else 0.0
    conflicting_keys = [k for k in repeated_input_keys if len(outcomes_by_input[k]) > 1]
    conflicting_inputs = len(conflicting_keys)
    consistent_repeated_inputs = repeated_inputs - conflicting_inputs

    determinism_confidence = (
        (consistent_repeated_inputs / repeated_inputs) * min(1.0, repeated_observations / min_repeated_observations)
        if repeated_inputs
        else 0.0
    )

    distinct_outcome_tuples = sorted(
        {outcome for outcomes in outcomes_by_input.values() for outcome in outcomes},
        key=lambda o: (o[0] or "", o[1] or ""),
    )
    distinct_outcomes = [f"{sel or '-'}/{res or '-'}" for sel, res in distinct_outcome_tuples]

    if conflicting_inputs > 0:
        classification = Classification.NON_DETERMINISTIC
        reason = (
            f"{conflicting_inputs} repeated identical input(s) produced more than one "
            "distinct outcome -- direct evidence of non-determinism."
        )
    elif (
        repeated_inputs >= min_repeated_inputs
        and repeat_coverage >= min_repeat_coverage
        and repeated_observations >= min_repeated_observations
    ):
        classification = Classification.DETERMINISTIC
        reason = (
            f"{repeated_inputs} distinct inputs repeated (>= {min_repeated_inputs}), "
            f"{repeat_coverage:.1%} repeat coverage (>= {min_repeat_coverage:.0%}), "
            f"{repeated_observations} repeated observations (>= {min_repeated_observations}) "
            "all consistent -- enough repeated evidence to call this deterministic."
        )
    else:
        classification = Classification.INSUFFICIENT_EVIDENCE
        missing = []
        if repeated_inputs < min_repeated_inputs:
            missing.append(f"only {repeated_inputs}/{min_repeated_inputs} distinct inputs repeated")
        if repeat_coverage < min_repeat_coverage:
            missing.append(f"repeat coverage {repeat_coverage:.1%} is below the configured {min_repeat_coverage:.0%} threshold")
        if repeated_observations < min_repeated_observations:
            missing.append(f"only {repeated_observations}/{min_repeated_observations} repeated observations")
        reason = "; ".join(missing) + " -- not enough repeated identical inputs to conclude determinism."

    return DeterminismEvidence(
        occurrences=occurrences,
        unique_inputs=unique_inputs,
        repeated_inputs=repeated_inputs,
        repeated_observations=repeated_observations,
        repeat_coverage=repeat_coverage,
        conflicting_inputs=conflicting_inputs,
        consistent_repeated_inputs=consistent_repeated_inputs,
        determinism_confidence=determinism_confidence,
        classification=classification,
        reason=reason,
        distinct_outcomes=distinct_outcomes,
    )
