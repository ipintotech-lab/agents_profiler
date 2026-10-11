from ingest.trace_schema import TraceEvent
from profiler.determinism import Classification, classify


def _event(**overrides) -> TraceEvent:
    base = {
        "trace_id": "t1",
        "agent": "agent-a",
        "workflow": "wf",
        "run_id": "r1",
        "event_type": "action",
        "node": "step",
        "input_state": {"k": "v"},
        "result": "success",
        "cost": 0.01,
        "latency_ms": 100,
    }
    base.update(overrides)
    return TraceEvent.model_validate(base)


def test_100_unique_inputs_no_repeats_is_insufficient_evidence():
    events = [_event(input_state={"k": i}) for i in range(100)]
    evidence = classify(events)
    assert evidence.classification == Classification.INSUFFICIENT_EVIDENCE
    assert evidence.repeated_inputs == 0
    assert evidence.occurrences == 100


def test_one_conflicting_repeated_input_is_non_deterministic_regardless_of_thresholds():
    events = [
        _event(run_id="r1", result="success"),
        _event(run_id="r2", result="failure"),
    ]
    # Even with thresholds lowered to the minimum, a single conflicting
    # repeated input must still win -- it is direct evidence, not a
    # statistical judgment call.
    evidence = classify(events, min_repeated_inputs=1, min_repeat_coverage=0.0, min_repeated_observations=1)
    assert evidence.classification == Classification.NON_DETERMINISTIC
    assert evidence.conflicting_inputs == 1


def test_enough_consistent_repeats_clears_thresholds_to_deterministic():
    events = [_event(run_id=f"r{i}", result="success") for i in range(40)]
    evidence = classify(
        events,
        min_repeated_inputs=1,
        min_repeat_coverage=0.30,
        min_repeated_observations=30,
    )
    assert evidence.classification == Classification.DETERMINISTIC
    assert evidence.repeated_inputs == 1
    assert evidence.repeated_observations == 40


def test_default_thresholds_require_many_distinct_repeated_inputs():
    # Only 2 distinct inputs, each repeated many times -- still short of
    # the default MIN_REPEATED_INPUTS=10, so this must not be called
    # deterministic even though repeat_coverage and observation count
    # are both high.
    events = [_event(input_state={"k": "a"}, run_id=f"a{i}") for i in range(50)]
    events += [_event(input_state={"k": "b"}, run_id=f"b{i}") for i in range(50)]
    evidence = classify(events)
    assert evidence.classification == Classification.INSUFFICIENT_EVIDENCE
    assert evidence.repeated_inputs == 2


def test_repeat_coverage_is_repeated_observations_over_occurrences():
    events = [_event(run_id="r1"), _event(run_id="r2")]
    events += [_event(input_state={"k": "unique"}, run_id="r3")]
    evidence = classify(events)
    assert evidence.occurrences == 3
    assert evidence.repeated_observations == 2
    assert evidence.repeat_coverage == 2 / 3
