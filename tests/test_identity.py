from dataclasses import replace

from ingest import TraceEvent
from profiler.finops import assess
from profiler.identity import UNKNOWN, action_identity, missing_fields
from profiler.mine import mine, score
from profiler.opportunities import find_opportunities
from profiler.patterns import find_repetition, repeated_calls


def _tool(run_id, agent="a", model=None, version="v1", node="lookup", **inp):
    return TraceEvent(
        trace_id="t", run_id=run_id, workflow="wf", agent=agent, event_type="tool_call", node=node,
        input_state=inp or {"q": "x"}, result="success:same", model=model, workflow_version=version,
    )


def _llm(run_id, model="m1", version="v1"):
    return TraceEvent(
        trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="decision", node="classify",
        decision={"type": "choose_severity", "selected": "HIGH"}, input_state={"cve": "CVE-1"},
        model=model, workflow_version=version, cost=0.02, result="success",
    )


def test_different_models_or_versions_are_different_identities():
    assert action_identity(_tool("r1", model="m1")) != action_identity(_tool("r2", model="m2"))
    assert action_identity(_tool("r1", version="v1")) != action_identity(_tool("r2", version="v2"))
    assert action_identity(_tool("r1", model="m1")) == action_identity(_tool("r2", model="m1"))


def test_missing_model_or_version_is_unknown_never_equal_to_a_known_one():
    missing = action_identity(_tool("r1", model=None, version=None))
    assert missing.model == UNKNOWN and missing.workflow_version == UNKNOWN
    assert missing != action_identity(_tool("r2", model="m1", version=None))
    assert missing_fields(missing, is_llm=True) == ["workflow_version", "model"]
    # a tool call has no model by nature
    assert missing_fields(missing, is_llm=False) == ["workflow_version"]


def test_same_node_name_on_different_agents_is_not_merged():
    assert action_identity(_tool("r1", agent="a")) != action_identity(_tool("r2", agent="b"))


def test_schema_version_is_part_of_the_identity():
    identity = action_identity(_tool("r1"))
    assert replace(identity, schema_version=identity.schema_version + 1) != identity


def test_repeated_calls_split_by_model_and_agent():
    events = [_tool(f"r{i}", model="m1") for i in range(3)] + [_tool(f"s{i}", model="m2") for i in range(3)]
    events += [_tool(f"u{i}", agent="b", model="m1") for i in range(2)]
    calls = repeated_calls(events)
    assert sorted((c["identity"]["agent"], c["identity"]["model"], c["count"]) for c in calls) == [
        ("a", "m1", 3), ("a", "m2", 3), ("b", "m1", 2),
    ]
    assert all(c["identity"]["missing"] == [] for c in calls)


def test_rule_bypass_is_one_configuration_at_a_time():
    events = [_llm(f"r{i}", model="m1") for i in range(6)] + [_llm(f"s{i}", model="m2") for i in range(3)]
    found = find_opportunities(events, "wf", find_repetition(events, "wf"), score(mine(events, "wf")))
    bypass = [o for o in found if o["type"] == "CONDITIONAL_LLM_BYPASS"]
    # m2 alone is below MIN_BYPASS_OBSERVATIONS; pooled together it would have passed
    assert [(o["identity"]["model"], o["observations"]) for o in bypass] == [("m1", 6)]


def test_unrecorded_configuration_is_never_low_risk():
    known = {"type": "CACHE_RESULT", "observations": 50, "nodes": ["lookup"], "identity": {"missing": []},
             "side_effects": {"has_side_effects": False, "cache_eligible": None}}
    assert assess(known)[0] == "low"
    risk, reasons = assess({**known, "identity": {"missing": ["workflow_version"]}})
    assert risk == "medium" and "workflow_version not recorded" in reasons
