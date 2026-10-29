from ingest import TraceEvent
from ingest.normalize import normalize_event
from profiler.finops import assess, build_finops
from profiler.mine import mine, score
from profiler.opportunities import declared_side_effects, find_opportunities
from profiler.patterns import find_repetition


def _tool(run_id, node="assign_rep", **declared):
    return TraceEvent(
        trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="tool_call", node=node,
        input_state={"region": "emea"}, result="rep:1", tool={"name": node}, workflow_version="v1", **declared,
    )


def _opportunities(events):
    return find_opportunities(events, "wf", find_repetition(events, "wf"), score(mine(events, "wf")))


def _cache(events):
    return [o for o in _opportunities(events) if o["type"] == "CACHE_RESULT"]


def test_declared_fields_are_read_under_common_aliases():
    raw = {"trace_id": "t", "run_id": "r", "workflow": "wf", "event_type": "tool_call", "node": "n",
           "hasSideEffects": False, "cacheable": True, "cacheTtlSeconds": 60, "invalidationKeys": ["k"]}
    event = TraceEvent.model_validate(normalize_event(raw))
    assert (event.has_side_effects, event.cache_eligible, event.cache_ttl_seconds, event.invalidation_keys) == (
        False, True, 60, ["k"],
    )
    assert TraceEvent.model_validate(normalize_event({**raw, "hasSideEffects": None})).has_side_effects is None


def test_a_node_is_side_effect_free_only_if_every_event_says_so():
    assert declared_side_effects([_tool("r1", has_side_effects=False)] * 2)["assign_rep"]["has_side_effects"] is False
    mixed = [_tool("r1", has_side_effects=False), _tool("r2")]
    assert declared_side_effects(mixed)["assign_rep"]["has_side_effects"] is None
    one_write = [_tool("r1", has_side_effects=False), _tool("r2", has_side_effects=True)]
    assert declared_side_effects(one_write)["assign_rep"]["has_side_effects"] is True


def test_declaration_overrides_the_name_heuristic():
    # "assign_*" reads as a write by name; nothing declared -> not suggested
    assert _cache([_tool(f"r{i}") for i in range(4)]) == []
    (cache,) = _cache([_tool(f"r{i}", has_side_effects=False) for i in range(4)])
    assert cache["side_effects"]["signal"] == "declared"
    # a read-sounding name declared as a write is not suggested either
    assert _cache([_tool(f"r{i}", node="get_owner", has_side_effects=True) for i in range(4)]) == []


def test_cache_eligible_decides_when_declared():
    assert _cache([_tool(f"r{i}", node="get_owner", cache_eligible=False) for i in range(4)]) == []
    (cache,) = _cache([_tool(f"r{i}", cache_eligible=True) for i in range(4)])
    assert cache["side_effects"]["signal"] == "heuristic_name_match"


def test_undeclared_side_effects_are_never_low_risk():
    (unknown,) = _cache([_tool(f"r{i}", node="get_owner") for i in range(12)])
    assert unknown["side_effects"]["signal"] == "unknown"
    risk, reasons = assess(unknown)
    assert risk == "medium" and "side effects not declared" in reasons
    (declared,) = _cache([_tool(f"r{i}", node="get_owner", has_side_effects=False) for i in range(12)])
    assert assess(declared)[0] == "low"


def test_rule_bypass_is_not_suggested_for_a_call_declared_to_change_something():
    def llm(run_id, **declared):
        return TraceEvent(
            trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="decision", node="classify",
            decision={"type": "choose_severity", "selected": "HIGH"}, input_state={"cve": "CVE-1"},
            model="m1", workflow_version="v1", cost=0.02, result="success", **declared,
        )

    def bypass(events):
        return [o for o in _opportunities(events) if o["type"] == "CONDITIONAL_LLM_BYPASS"]

    assert len(bypass([llm(f"r{i}") for i in range(6)])) == 1
    assert bypass([llm(f"r{i}", has_side_effects=True) for i in range(6)]) == []


def _llm(run_id, selected, **fields):
    return TraceEvent(
        trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="decision", node="llm_call",
        decision={"type": "next_action", "selected": selected}, input_state={"trigger": "user_prompt"},
        model="m1", workflow_version="v1", cost=0.02, **fields,
    )


def test_a_sequence_with_undeclared_steps_is_never_low_risk():
    def chain(**declared):
        events = []
        for i in range(24):
            events += [_llm(f"r{i}", "fetch"), _tool(f"r{i}", node="fetch", **declared),
                       _llm(f"r{i}", "parse"), _tool(f"r{i}", node="parse", **declared)]
        (seq,) = [o for o in _opportunities(events) if o["type"] == "EXTRACT_DETERMINISTIC_SUBSTEP"]
        return assess(seq)

    risk, reasons = chain()
    assert risk == "medium" and "side effects not declared" in reasons
    assert chain(has_side_effects=False)[0] == "low"


def test_varying_outputs_on_a_partly_recorded_input_are_not_hidden_state():
    def found(**fields):
        events = [_llm(f"r{i}", ["a", "b"][i % 2], **fields) for i in range(20)]
        stats = score(mine(events, "wf"))
        hidden = [o for o in _opportunities(events) if o["type"] == "REVIEW_HIDDEN_STATE"]
        return stats[0].classification.value, hidden

    classification, hidden = found()
    assert classification == "NON_DETERMINISTIC" and hidden
    classification, hidden = found(input_complete=False)
    assert classification == "INSUFFICIENT_EVIDENCE" and hidden == []
    (action,) = build_finops([_llm(f"r{i}", "a", input_complete=False) for i in range(5)])["actions"]
    assert action["input_recorded"] is False
