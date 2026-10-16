from ingest import TraceEvent
from profiler.patterns import find_repetition, repeated_calls, repeated_sequences


def _router(run_id, cost=0.1):
    return TraceEvent(
        trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="decision", node="llm_call",
        model="m", decision={"type": "next_action", "selected": "x"}, cost=cost, tokens={"input": 100, "output": 10},
    )


def _tool(run_id, node, result="success:1", **inp):
    return TraceEvent(
        trace_id="t", run_id=run_id, workflow="wf", agent="a", event_type="tool_call", node=node,
        input_state=inp, result=result,
    )


def _git_flow(run_id):
    return [
        _router(run_id), _tool(run_id, "git add", files="a.py"),
        _router(run_id), _tool(run_id, "git commit", msg=run_id),
        _router(run_id), _tool(run_id, "git push", remote="origin"),
    ]


def test_repeated_sequence_is_found_and_priced_by_llm_calls_between_steps():
    events = [e for run in ("r1", "r2", "r3") for e in _git_flow(run)]
    sequences = repeated_sequences(events)
    top = sequences[0]
    assert top["steps"] == ["git add", "git commit", "git push"]
    assert top["occurrences"] == 3
    assert top["runs"] == 3
    # two next-step LLM calls inside each occurrence (before commit and push)
    assert top["llm_calls_between_steps"] == 6
    assert abs(top["cost_between_steps"] - 0.6) < 1e-9
    assert top["tokens_between_steps"] == 6 * 110
    assert [s["node"] for s in top["example"]] == top["steps"]
    # sub-sequences that occur exactly as often are folded into the full flow
    assert all(p["steps"] != ["git add", "git commit"] for p in sequences)


def test_sequences_seen_fewer_than_three_times_are_not_reported():
    events = [e for run in ("r1", "r2") for e in _git_flow(run)]
    assert repeated_sequences(events) == []


def test_repeated_call_with_same_output_counts_redundant_repeats():
    events = [_tool(f"r{i}", "Read", result="success:same", file_path="a.py") for i in range(4)]
    events += [_tool("r1", "Read", result=f"success:{i}", file_path="b.py") for i in range(2)]
    calls = {c["input"]["file_path"]: c for c in repeated_calls(events)}
    assert calls["a.py"]["same_output"] is True
    assert calls["a.py"]["redundant_calls"] == 3
    assert calls["b.py"]["same_output"] is False
    assert calls["b.py"]["redundant_calls"] == 0


def test_find_repetition_only_uses_the_selected_workflow():
    other = [e.model_copy(update={"workflow": "other"}) for run in ("r1", "r2", "r3") for e in _git_flow(run)]
    assert find_repetition(other, "wf")["sequences"] == []


def test_sequences_are_found_in_traces_without_routing_markers():
    # Generic trace formats: every event is a step, nothing marks "choose the next step".
    events = [
        _tool(run, node).model_copy(update={"cost": 0.01})
        for run in ("r1", "r2", "r3")
        for node in ("classify", "lookup", "reply")
    ]
    top = repeated_sequences(events)[0]
    assert top["steps"] == ["classify", "lookup", "reply"]
    assert top["llm_calls_between_steps"] == 0
    assert abs(top["sequence_cost"] - 0.09) < 1e-9


def test_decision_cost_is_split_across_the_steps_one_call_chose():
    from profiler.patterns import decision_costs

    a, b = _tool("r1", "Read", file_path="a.py"), _tool("r1", "Read", file_path="b.py")
    events = [_router("r1", cost=0.4), a, b]
    shares = decision_costs(events)
    assert shares[id(a)] == shares[id(b)] == 0.2


def test_a_call_that_answered_the_user_is_not_charged_to_the_next_step():
    from profiler.patterns import decision_costs

    answer = TraceEvent.model_validate(
        {**_router("r1", cost=0.5).model_dump(), "decision": {"type": "next_action", "selected": "respond"}}
    )
    step = _tool("r1", "Read", file_path="a.py")
    assert decision_costs([answer, step]).get(id(step), 0.0) == 0.0
    assert repeated_sequences([e for _ in range(3) for e in (answer, step, _router("r1"), step)])[0]["cost_between_steps"] > 0


def test_summary_ranks_opportunities_and_prices_cache_with_deciding_cost():
    from profiler.mine import mine
    from profiler.opportunities import build_summary

    events = [e for run in ("r1", "r2", "r3") for e in _git_flow(run)]
    events += [e for i in range(3) for e in (_router(f"c{i}", cost=0.3), _tool(f"c{i}", "Read", result="success:same", file_path="a.py"))]
    summary = build_summary(events, "wf", find_repetition(events, "wf"), mine(events, "wf"))
    types = {o["type"] for o in summary["opportunities"]}
    assert {"EXTRACT_DETERMINISTIC_SUBSTEP", "CACHE_RESULT"} <= types
    cache = next(o for o in summary["opportunities"] if o["type"] == "CACHE_RESULT")
    assert abs(cache["saving"] - 0.6) < 1e-9  # two repeats × the 0.3 routing call that chose each
    savings = [o["saving"] for o in summary["opportunities"] if o["saving"] is not None]
    assert savings == sorted(savings, reverse=True)
    assert summary["totals"]["tool_calls"] == 12


def test_has_side_effect_reads_the_operation_name():
    from profiler.opportunities import has_side_effect

    for name in ("create_or_update_ticket", "messaging.send_reply", "Edit", "Write",
                 "str_replace_editor", "mcp__jira__createIssue", "log_activity"):
        assert has_side_effect(name), name
    for name in ("Read", "Grep", "fetch_error_log", "scheduler.get_logs", "get_table_schema",
                 "search__web_search", "assign_rep", "check_job_status"):
        assert not has_side_effect(name), name



def test_fixed_checklist_is_scriptable_but_model_steered_shape_is_not():
    """A sequence whose steps always follow each other is a fixed procedure
    (EXTRACT_DETERMINISTIC_SUBSTEP); one where the model chooses between
    options is only a recurring shape (REPEATED_AGENT_PATTERN)."""
    from profiler.opportunities import _sequence_opportunities

    # checklist: fetch -> lint -> test, every run, no branching
    def checklist(run_id):
        return [
            _router(run_id), _tool(run_id, "fetch"),
            _router(run_id), _tool(run_id, "lint"),
            _router(run_id), _tool(run_id, "test"),
        ]

    # shape: search is followed by edit in half the runs and by read in the
    # other half -- the model chooses, so search -> edit is not a script
    def steered(run_id, second):
        return [
            _router(run_id), _tool(run_id, "search", q=run_id),
            _router(run_id), _tool(run_id, second),
        ]

    events = [e for run in ("r1", "r2", "r3", "r4") for e in checklist(run)]
    events += [e for run, second in (("s1", "edit"), ("s2", "read"), ("s3", "edit"), ("s4", "read"),
                                      ("s5", "edit"), ("s6", "read")) for e in steered(run, second)]

    ops = {tuple(o["nodes"]): o for o in _sequence_opportunities(repeated_sequences(events))}
    script = ops[("fetch", "lint", "test")]
    assert script["type"] == "EXTRACT_DETERMINISTIC_SUBSTEP"
    assert "100% of the time" in script["evidence"]

    shape = ops[("search", "edit")]
    assert shape["type"] == "REPEATED_AGENT_PATTERN"
    assert "50% of the time" in shape["evidence"]
    assert "batch" in shape["action"] and "cheaper model" in shape["action"]
    assert shape["example"][0]["node"] == "search"  # a real occurrence travels with the finding


def test_sequence_title_shows_the_actual_arguments_not_just_tool_names():
    """``Bash: grep → Bash: sed`` says nothing about the work; the title must
    show a real occurrence's commands (``grep -n x file`` …), so a customer
    can tell what was actually done."""
    from profiler.opportunities import _sequence_opportunities

    def run(run_id):
        return [
            _router(run_id), _tool(run_id, "Bash: grep", command=f'grep -n "{run_id}" src/app.tsx'),
            _router(run_id), _tool(run_id, "Bash: sed", command=f"sed -i 's/{run_id}/x/' src/app.tsx"),
        ]

    events = [e for r in ("r1", "r2", "r3") for e in run(r)]
    (op,) = [o for o in _sequence_opportunities(repeated_sequences(events)) if o["nodes"] == ["Bash: grep", "Bash: sed"]]
    assert op["title"] == 'grep -n "r1" src/app.tsx → sed -i \'s/r1/x/\' src/app.tsx'
