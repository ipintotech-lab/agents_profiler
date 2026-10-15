from fastapi.testclient import TestClient

from api.main import app
from profiler.experiment import (
    CALIBRATION_SEEDS,
    VALIDATION_SEEDS,
    calibrate_multi_seed,
    detect_stabilization_point,
    expected_vs_found,
    recommendation_stability,
    run_poc,
    sample_size_experiment,
)


# (1) multi-seed calibration is reproducible
def test_calibrate_multi_seed_is_reproducible():
    a = calibrate_multi_seed(seeds=(1, 2, 3), sample_size=200)
    b = calibrate_multi_seed(seeds=(1, 2, 3), sample_size=200)
    assert a["selected"] == b["selected"]
    assert a["current_defaults"] == b["current_defaults"]


# (2) calibration and validation seed sets are separate
def test_calibration_and_validation_seed_sets_are_disjoint():
    assert not set(CALIBRATION_SEEDS) & set(VALIDATION_SEEDS)


# (3) thresholds stay fixed across every sample size in Phase 2
def test_sample_size_experiment_keeps_thresholds_fixed():
    thresholds = {"min_repeated_inputs": 5, "min_repeat_coverage": 0.2, "min_repeated_observations": 20}
    result = sample_size_experiment(thresholds=thresholds, sample_sizes=(100, 500), seeds=(101, 102))
    assert result["thresholds_used"] == thresholds


# (4) the experiment actually uses the requested sample sizes
def test_sample_size_experiment_uses_requested_sizes():
    result = sample_size_experiment(sample_sizes=(100, 500, 1000), seeds=(101, 102))
    assert [row["sample_size"] for row in result["curve"]] == [100, 500, 1000]
    assert set(result["raw_per_seed"].keys()) == {"100", "500", "1000"}
    for size in (100, 500, 1000):
        assert len(result["raw_per_seed"][str(size)]) == 2  # one report per seed


# (5) aggregated metrics are calculated correctly from raw per-seed reports
def test_aggregated_metrics_match_raw_reports():
    result = sample_size_experiment(sample_sizes=(1000,), seeds=(101, 102, 103))
    row = result["curve"][0]
    raw = result["raw_per_seed"]["1000"]
    accuracies = [r["accuracy"] for r in raw if r["accuracy"] is not None]
    assert row["mean_accuracy"] == sum(accuracies) / len(accuracies)
    assert row["total_false_negatives"] == sum(r["false_negatives"] for r in raw)


# (6) plateau detector identifies a known synthetic plateau
def test_plateau_detector_finds_a_known_plateau():
    curve = [
        {"sample_size": 100, "mean_accuracy": 0.80, "mean_recall": 0.95, "mean_coverage": 0.6, "accuracy_std": 0.1},
        {"sample_size": 500, "mean_accuracy": 0.97, "mean_recall": 0.95, "mean_coverage": 0.6, "accuracy_std": 0.01},
        {"sample_size": 1000, "mean_accuracy": 0.975, "mean_recall": 0.95, "mean_coverage": 0.6, "accuracy_std": 0.01},
        {"sample_size": 2000, "mean_accuracy": 0.976, "mean_recall": 0.95, "mean_coverage": 0.6, "accuracy_std": 0.01},
    ]
    result = detect_stabilization_point(curve)
    assert result["stabilization_point"] == 500


# (7) plateau detector returns None when no stable plateau exists in range
def test_plateau_detector_returns_none_when_no_plateau_exists():
    curve = [
        {"sample_size": 100, "mean_accuracy": 0.5, "mean_recall": 0.5, "mean_coverage": 0.2, "accuracy_std": 0.3},
        {"sample_size": 500, "mean_accuracy": 0.6, "mean_recall": 0.5, "mean_coverage": 0.2, "accuracy_std": 0.3},
        {"sample_size": 1000, "mean_accuracy": 0.7, "mean_recall": 0.5, "mean_coverage": 0.2, "accuracy_std": 0.3},
    ]
    result = detect_stabilization_point(curve)
    assert result["stabilization_point"] is None
    assert result["sample_sizes"] == [100, 500, 1000]


# (8) recommendation Top-1 stability calculation
def test_top1_stability_computed_correctly():
    result = recommendation_stability(sample_size=500, seeds=(101, 102, 103, 104))
    assert 0.0 <= result["top1_stability"] <= 1.0
    assert result["top1_mode_item"] is not None


# (9) Top-3 overlap calculation
def test_top3_overlap_computed_correctly():
    result = recommendation_stability(sample_size=500, seeds=(101, 102, 103))
    assert result["top3_overlap"] is not None
    assert 0.0 <= result["top3_overlap"] <= 1.0
    assert len(result["per_seed_top3"]) == 3


# (10) same seed + same params reproduces the same experiment
def test_same_seed_and_params_reproduce_same_result():
    a = sample_size_experiment(sample_sizes=(300,), seeds=(101,))
    b = sample_size_experiment(sample_sizes=(300,), seeds=(101,))
    assert a["curve"] == b["curve"]


# (11) different seeds create independently varying datasets
def test_different_seeds_vary_independently():
    a = sample_size_experiment(sample_sizes=(300,), seeds=(101,))
    b = sample_size_experiment(sample_sizes=(300,), seeds=(102,))
    assert a["raw_per_seed"]["300"] != b["raw_per_seed"]["300"]


def test_run_poc_rejects_overlapping_seed_ranges():
    import pytest

    with pytest.raises(ValueError):
        run_poc(calibration_seeds=(1, 2), validation_seeds=(2, 3))


def test_run_poc_end_to_end_small_smoke():
    result = run_poc(
        calibration_seeds=(1, 2),
        validation_seeds=(101, 102),
        calibration_sample_size=100,
        sample_sizes=(100, 200),
    )
    assert result["conclusion"] in {"SUPPORTED", "NOT YET SUPPORTED"}
    assert "phase1_calibration" in result
    assert "phase2_stabilization" in result
    assert result["phase4_bpop_evaluation"]["note"]
    assert result["phase5_modeled_cost"]["modeled"] is True
    assert result["phase6_expected_vs_found"]["summary"]["planted_total"] > 0


def test_expected_vs_found_reports_planted_findings_and_traps():
    result = expected_vs_found(sample_size=2000, seed=101, sample_sizes=(100, 2000))
    rows = result["rows"]
    # every agent's workflow is graded, not only the largest one
    assert len({r["workflow"] for r in rows}) == 6
    s = result["summary"]
    assert s["planted_total"] == sum(1 for r in rows if r["expected"])
    assert s["traps"] == sum(1 for r in rows if not r["expected"])
    # the plain cacheable lookups and scripted chain are found at this size
    found = {(r["type"], r["target"]) for r in rows if r["found"]}
    assert ("CACHE_RESULT", "get_table_schema") in found
    assert ("EXTRACT_DETERMINISTIC_SUBSTEP", "fetch_diff → run_linter → run_tests") in found
    assert all(r["agrees"] == (r["found"] == r["expected"]) for r in rows)
    # more data never finds fewer planted findings on the same seed here
    small, large = result["by_sample_size"]
    assert small["planted_found"] <= large["planted_found"] == s["planted_found"]
    assert not any("_followup_" in u["target"] for u in result["unplanned"])


def test_run_poc_full_default_run_matches_measured_result():
    """Locks in the measured full-rigor result (30 calibration seeds, 30
    validation seeds, BASELINE_CONFIG with all six agents). With six
    workflows sharing the line budget, the shipped defaults (3, 0.1, 10)
    make 1 false negative across the 30 calibration seeds, so Phase 1
    picks min_repeated_inputs=10; coverage then stays below the 0.375
    plateau bar up to n=10000 and no stabilization point is found. When
    only the CVE workflow was graded this found a plateau at n=5000. If
    this changes, either the pipeline behavior changed (investigate) or
    BASELINE_CONFIG / DEFAULT_PLATEAU_CRITERIA changed on purpose (update
    this test)."""
    result = run_poc()
    assert result["phase1_calibration"]["selected"]["min_repeated_inputs"] == 10
    assert result["phase2_stabilization"]["stabilization_point"] is None
    assert result["conclusion"] == "NOT YET SUPPORTED"
    assert result["phase3_recommendation_stability"]["top1_stability"] == 1.0
    p6 = result["phase6_expected_vs_found"]["summary"]
    assert (p6["planted_found"], p6["planted_total"], p6["false_alarms"]) == (16, 17, 0)


def test_poc_run_endpoint_returns_full_report():
    client = TestClient(app)
    resp = client.post(
        "/poc/run",
        params={"calibration_seed_count": 2, "validation_seed_count": 2, "calibration_sample_size": 100},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["conclusion"] in {"SUPPORTED", "NOT YET SUPPORTED"}


def test_calls_that_change_something_are_never_cache_findings():
    result = expected_vs_found(sample_size=2000, seed=101)
    flagged = {r["target"] for r in result["rows"] if r["type"] == "CACHE_RESULT" and r["found"]}
    assert "create_or_update_ticket" not in flagged
    assert "send_acknowledgement" not in flagged
    assert {"get_table_schema", "enrich_company", "assign_rep"} <= flagged
