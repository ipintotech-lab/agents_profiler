from profiler.determinism import Classification
from profiler.validation import validate


def test_true_positive_and_true_negative_counted_correctly():
    predictions = {
        "a/flaky": Classification.NON_DETERMINISTIC,
        "a/stable": Classification.DETERMINISTIC,
    }
    ground_truth = {
        "a/flaky": {"determinism_rate": 0.5},
        "a/stable": {"determinism_rate": 1.0},
    }
    report = validate(predictions, ground_truth)
    assert report.true_positives == 1
    assert report.true_negatives == 1
    assert report.false_positives == 0
    assert report.false_negatives == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.accuracy == 1.0
    assert report.insufficient_evidence_nodes == 0
    assert report.coverage == 1.0


def test_insufficient_evidence_is_never_scored_as_correct_or_incorrect():
    predictions = {"a/unknown": Classification.INSUFFICIENT_EVIDENCE}
    ground_truth = {"a/unknown": {"determinism_rate": 1.0}}
    report = validate(predictions, ground_truth)
    assert report.insufficient_evidence_nodes == 1
    assert report.classified_nodes == 0
    assert report.coverage == 0.0
    assert report.insufficient_evidence_rate == 1.0
    assert report.precision is None
    assert report.recall is None
    assert report.accuracy is None


def test_false_positive_and_false_negative():
    predictions = {
        "a/wrongly_flagged": Classification.NON_DETERMINISTIC,
        "a/missed_flaky": Classification.DETERMINISTIC,
    }
    ground_truth = {
        "a/wrongly_flagged": {"determinism_rate": 1.0},
        "a/missed_flaky": {"determinism_rate": 0.2},
    }
    report = validate(predictions, ground_truth)
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.precision == 0.0
    assert report.recall == 0.0


def test_nodes_without_ground_truth_are_ignored():
    predictions = {"a/no_truth": Classification.DETERMINISTIC}
    ground_truth: dict = {}
    report = validate(predictions, ground_truth)
    assert report.ground_truth_nodes == 0
    assert report.coverage == 0.0
    assert report.precision is None
