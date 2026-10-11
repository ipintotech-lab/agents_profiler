"""Validation metrics comparing the profiler's determinism classification
against known ground truth from the synthetic generator.

This is the only way to tell whether the three-state classifier in
``determinism.py`` actually works, rather than just producing output
that looks plausible. ``INSUFFICIENT_EVIDENCE`` is reported separately
and is never counted as correct or incorrect -- it is a "no decision"
outcome, not a wrong one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .determinism import Classification

DEFAULT_DETERMINISTIC_RATE_THRESHOLD = 1.0
"""Ground-truth label cutoff: a node's determinism_rate must be exactly
(>=) this to be labeled DETERMINISTIC. Below it, the generator can -- by
construction -- produce a different output for the same canonical input,
so the ground-truth label is NON_DETERMINISTIC, however rarely that
actually fires in a given sample."""


@dataclass
class ValidationReport:
    ground_truth_nodes: int
    classified_nodes: int
    insufficient_evidence_nodes: int
    insufficient_evidence_rate: float
    coverage: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    accuracy: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    confusion_matrix: dict[str, int]
    per_node: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ground_truth_nodes": self.ground_truth_nodes,
            "classified_nodes": self.classified_nodes,
            "insufficient_evidence_nodes": self.insufficient_evidence_nodes,
            "insufficient_evidence_rate": round(self.insufficient_evidence_rate, 4),
            "coverage": round(self.coverage, 4),
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": _round_or_none(self.precision),
            "recall": _round_or_none(self.recall),
            "accuracy": _round_or_none(self.accuracy),
            "false_positive_rate": _round_or_none(self.false_positive_rate),
            "false_negative_rate": _round_or_none(self.false_negative_rate),
            "confusion_matrix": self.confusion_matrix,
            "per_node": self.per_node,
        }


def _round_or_none(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def validate(
    predictions: dict[str, Classification],
    ground_truth: dict[str, dict],
    *,
    deterministic_rate_threshold: float = DEFAULT_DETERMINISTIC_RATE_THRESHOLD,
) -> ValidationReport:
    """Compare the profiler's per-node classification against known
    ground truth.

    ``predictions`` maps ``"agent/node"`` -> ``Classification`` (from
    mine()'s output). ``ground_truth`` maps ``"agent/node"`` ->
    ``{"determinism_rate": float}`` (from the generator's dataset
    manifest). NON_DETERMINISTIC is treated as the positive class,
    matching the profiler's risk-flagging framing (a false negative is a
    node that silently varies but gets called safe).
    """
    evaluable_nodes = [key for key in ground_truth if key in predictions]

    tp = tn = fp = fn = 0
    insufficient = 0
    per_node: list[dict] = []

    for key in evaluable_nodes:
        rate = ground_truth[key]["determinism_rate"]
        truth_label = (
            Classification.DETERMINISTIC
            if rate >= deterministic_rate_threshold
            else Classification.NON_DETERMINISTIC
        )
        predicted = predictions[key]
        outcome: str
        if predicted == Classification.INSUFFICIENT_EVIDENCE:
            insufficient += 1
            outcome = "insufficient_evidence"
        elif predicted == truth_label == Classification.NON_DETERMINISTIC:
            tp += 1
            outcome = "true_positive"
        elif predicted == truth_label == Classification.DETERMINISTIC:
            tn += 1
            outcome = "true_negative"
        elif predicted == Classification.NON_DETERMINISTIC and truth_label == Classification.DETERMINISTIC:
            fp += 1
            outcome = "false_positive"
        else:  # predicted DETERMINISTIC, truth NON_DETERMINISTIC
            fn += 1
            outcome = "false_negative"

        per_node.append(
            {
                "node": key,
                "determinism_rate": rate,
                "truth": truth_label.value,
                "predicted": predicted.value,
                "outcome": outcome,
            }
        )

    ground_truth_nodes = len(evaluable_nodes)
    classified_nodes = tp + tn + fp + fn
    coverage = (classified_nodes / ground_truth_nodes) if ground_truth_nodes else 0.0
    insufficient_evidence_rate = (insufficient / ground_truth_nodes) if ground_truth_nodes else 0.0

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    accuracy = (tp + tn) / classified_nodes if classified_nodes else None
    false_positive_rate = fp / (fp + tn) if (fp + tn) else None
    false_negative_rate = fn / (fn + tp) if (fn + tp) else None

    return ValidationReport(
        ground_truth_nodes=ground_truth_nodes,
        classified_nodes=classified_nodes,
        insufficient_evidence_nodes=insufficient,
        insufficient_evidence_rate=insufficient_evidence_rate,
        coverage=coverage,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        confusion_matrix={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        per_node=per_node,
    )
