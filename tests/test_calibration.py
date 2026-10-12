from ingest.trace_schema import TraceEvent
from profiler.calibration import calibrate_determinism_thresholds
from profiler.pipeline import run_pipeline


def _event(**overrides) -> TraceEvent:
    base = {
        "trace_id": "t1",
        "agent": "agent-a",
        "workflow": "wf",
        "run_id": "r1",
        "event_type": "llm_call",
        "model": "demo-llm",
        "node": "step",
        "input_state": {"k": "v"},
        "result": "success",
        "cost": 0.01,
        "latency_ms": 100,
    }
    base.update(overrides)
    return TraceEvent.model_validate(base)


def _repeated_node_events(node: str, num_distinct_inputs: int, repeats: int) -> list[TraceEvent]:
    """``num_distinct_inputs`` distinct canonical inputs, each repeated
    exactly ``repeats`` times with the SAME outcome every time -- by
    construction, no sample ever shows a contradiction, so this node's
    classification is governed entirely by the threshold cutoffs, not by
    any conflicting observation."""
    return [
        _event(node=node, run_id=f"{node}-{i}-{j}", input_state={"case": i}, result="ok")
        for i in range(num_distinct_inputs)
        for j in range(repeats)
    ]


def _build_borderline_dataset() -> list[TraceEvent]:
    # 10 distinct inputs x 3 repeats = 30 observations, no conflicts.
    # Ground truth says this node IS non-deterministic (rate 0.8) even
    # though this particular sample never happened to show it -- the
    # exact scenario that produces a false negative at low thresholds.
    borderline = _repeated_node_events("borderline", num_distinct_inputs=10, repeats=3)
    # A second node that's genuinely deterministic at every threshold,
    # so the dataset isn't a single-node edge case.
    stable = _repeated_node_events("stable", num_distinct_inputs=10, repeats=3)
    return borderline + stable


_GROUND_TRUTH = {
    "agent-a/borderline": {"determinism_rate": 0.8},
    "agent-a/stable": {"determinism_rate": 1.0},
}


def test_low_thresholds_produce_a_false_negative_on_the_borderline_node():
    events = _build_borderline_dataset()
    result = calibrate_determinism_thresholds(
        events,
        "wf",
        _GROUND_TRUTH,
        min_repeated_inputs_grid=(10,),
        min_repeat_coverage_grid=(0.3,),
        min_repeated_observations_grid=(30,),
    )
    assert result["current_defaults"]["false_negatives"] == 1


def test_raising_thresholds_converts_the_false_negative_into_insufficient_evidence():
    events = _build_borderline_dataset()
    result = calibrate_determinism_thresholds(
        events,
        "wf",
        _GROUND_TRUTH,
        min_repeated_inputs_grid=(10,),
        min_repeat_coverage_grid=(0.3,),
        min_repeated_observations_grid=(40,),
    )
    only_candidate = result["leaderboard"][0]
    assert only_candidate["false_negatives"] == 0
    assert only_candidate["insufficient_evidence_rate"] > 0


def test_calibration_finds_a_better_or_equal_configuration_than_current_defaults():
    events = _build_borderline_dataset()
    result = calibrate_determinism_thresholds(events, "wf", _GROUND_TRUTH)
    assert result["best_found"]["false_negatives"] <= result["current_defaults"]["false_negatives"]
    assert result["improves_on_current"] is True
    assert result["best_found"]["false_negatives"] == 0


def test_leaderboard_is_sorted_by_fewest_false_negatives_first():
    events = _build_borderline_dataset()
    result = calibrate_determinism_thresholds(events, "wf", _GROUND_TRUTH, top_n=5)
    fn_values = [c["false_negatives"] for c in result["leaderboard"]]
    assert fn_values == sorted(fn_values)


def test_run_pipeline_includes_calibration_alongside_validation_when_ground_truth_given():
    events = _build_borderline_dataset()
    report = run_pipeline(events, workflow="wf", ground_truth=_GROUND_TRUTH)
    assert "validation" in report
    assert "calibration" in report
    assert set(report["calibration"]) == {"current_defaults", "best_found", "improves_on_current", "leaderboard"}


def test_run_pipeline_omits_calibration_without_ground_truth():
    events = _build_borderline_dataset()
    report = run_pipeline(events, workflow="wf")
    assert "validation" not in report
    assert "calibration" not in report
