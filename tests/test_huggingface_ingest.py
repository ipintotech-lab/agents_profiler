import gzip
import io
import json
import urllib.error

import pytest
from fastapi.testclient import TestClient

import ingest.huggingface as hf
from api.main import app
from ingest import TraceEvent, normalize_raw_json
from ingest.huggingface import HFRef, fetch_records, parse_hf_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://huggingface.co/datasets/cx-cmu/agent_trajectories?utm_source=chatgpt.com",
         HFRef("cx-cmu/agent_trajectories")),
        ("https://huggingface.co/datasets/cx-cmu/agent_trajectories/blob/main/swebench.jsonl",
         HFRef("cx-cmu/agent_trajectories", "main", "swebench.jsonl")),
        ("https://huggingface.co/datasets/org/name/resolve/v1/a/b.jsonl.zst", HFRef("org/name", "v1", "a/b.jsonl.zst")),
        ("https://huggingface.co/datasets/org/name/tree/main/release", HFRef("org/name", "main", "release")),
        ("https://huggingface.co/datasets/org/name/viewer/default/train", HFRef("org/name")),
        ("hf://datasets/cx-cmu/agent_trajectories/swebench.jsonl",
         HFRef("cx-cmu/agent_trajectories", "main", "swebench.jsonl")),
        ("hf://datasets/org/name@dev/x.jsonl", HFRef("org/name", "dev", "x.jsonl")),
        ("org/name", HFRef("org/name")),
    ],
)
def test_parse_hf_url(url, expected):
    assert parse_hf_url(url) == expected


def test_parse_hf_url_rejects_non_dataset_links():
    with pytest.raises(ValueError):
        parse_hf_url("https://huggingface.co/some-model")


def _chat_record(record_id, pass_no, ls_output="a.py"):
    return {
        "id": record_id,
        "benchmark": "swebench",
        "domain": "django",
        "task_id": "1",
        "source_model": "Qwen3",
        "pass": pass_no,
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "execute_bash", "arguments": "{\"command\": \"ls -la\"}"}},
                {"id": "c2", "type": "function", "function": {"name": "str_replace_editor", "arguments": "{\"command\": \"view\", \"path\": \"/repo/x.py\"}"}},
            ]},
            {"role": "tool", "content": ls_output, "tool_call_id": "c1"},
            {"role": "tool", "content": "print(1)", "tool_call_id": "c2"},
            {"role": "assistant", "content": "Done."},
        ],
    }


def test_chat_trajectory_converts_to_llm_and_tool_events():
    events = [TraceEvent.model_validate(r) for r in normalize_raw_json([_chat_record("t1", 1)])]
    llm = [e for e in events if e.event_type == "decision"]
    tools = [e for e in events if e.event_type == "tool_call"]

    assert [e.node for e in llm] == ["llm_call ← user_prompt", "llm_call ← execute_bash"]
    assert llm[0].decision.selected == "execute_bash: ls+str_replace_editor: view"
    assert llm[1].decision.selected == "respond"
    assert llm[1].trigger == "execute_bash: ls + str_replace_editor: view"
    assert [e.node for e in tools] == ["execute_bash: ls", "str_replace_editor: view"]
    assert tools[0].input_state == {"command": "ls -la"}
    assert tools[0].result.startswith("output:") and tools[0].result != tools[1].result
    assert {(e.trace_id, e.run_id, e.workflow, e.agent) for e in events} == {("t1", "t1", "swebench", "Qwen3")}
    # only LLM calls carry a model -- the report counts LLM calls by it
    assert {e.model for e in llm} == {"Qwen3"} and {e.model for e in tools} == {None}
    assert events[0].user_input == "Fix the bug."


def test_chat_trajectory_passes_of_one_task_share_tool_results():
    events = normalize_raw_json([_chat_record("t1", 1), _chat_record("t2", 2)])
    ls_results = {e["result"] for e in events if e["node"] == "execute_bash: ls"}
    assert len(ls_results) == 1  # same input, same output across passes


def test_chat_trajectory_pairs_results_without_ids_in_call_order():
    record = _chat_record("t1", 1)
    for m in record["messages"]:
        m.pop("tool_call_id", None)
        for tc in m.get("tool_calls") or []:
            tc.pop("id")
    tools = [e for e in normalize_raw_json(record) if e["event_type"] == "tool_call"]
    assert all(e["result"] != "no_result" for e in tools)


def _fi_session(session_id="s1", subagent=None):
    return {
        "session_id": session_id,
        "harness": "claude-code",
        "requests": [
            {"step": 0, "turn_index": 0, "step_trigger": "user", "start_ms": 1000, "end_ms": 3000, "model": "glm",
             "provider_tokens": {"prompt": 100, "completion": 20, "cached": 0}, "finish_reason": "tool_calls",
             "tool_calls": [
                 {"call_index": 0, "name": "terminal", "arguments": [["command", "git status"]],
                  "outcome": "ok", "result_tokens": 12, "latency_ms": 50, "subagent": None},
                 {"call_index": 1, "name": "read_file", "arguments": [["path", "<path>"]],
                  "outcome": "error", "result_tokens": 3, "latency_ms": 5, "subagent": subagent},
             ]},
            {"step": 1, "turn_index": 0, "step_trigger": "tool", "start_ms": 3100, "end_ms": 3500, "model": "glm",
             "provider_tokens": {"prompt": 150, "completion": 5}, "finish_reason": "stop", "tool_calls": []},
        ],
    }


def test_freeinference_session_converts_to_llm_and_tool_events():
    events = [TraceEvent.model_validate(r) for r in normalize_raw_json(_fi_session())]
    llm = [e for e in events if e.event_type == "decision"]
    tools = [e for e in events if e.event_type == "tool_call"]

    assert [e.node for e in llm] == ["llm_call ← user_prompt", "llm_call ← read_file"]
    assert llm[1].trigger == "read_file (error) + terminal: git status"
    assert llm[0].tokens.input == 100 and llm[0].tokens.output == 20 and llm[0].latency_ms == 2000
    assert llm[0].decision.selected == "read_file+terminal: git status"
    assert [e.node for e in tools] == ["terminal: git status", "read_file"]
    assert tools[0].input_state == {"command": "git status"}
    assert tools[0].result == "ok:12tok"
    assert {e.workflow for e in events} == {"freeinference/claude-code"}
    assert {e.run_id for e in events} == {"s1:turn0"}


def test_freeinference_anonymized_inputs_never_count_as_repeats():
    events = normalize_raw_json([_fi_session("s1"), _fi_session("s2")])
    reads = [e["input_state"] for e in events if e["node"] == "read_file"]
    commands = [e["input_state"] for e in events if e["node"] == "terminal: git status"]
    assert reads[0] != reads[1]
    assert commands[0] == commands[1]


def test_freeinference_subagent_sessions_are_nested_into_the_run():
    events = normalize_raw_json(_fi_session(subagent=_fi_session("sub")))
    sub = [e for e in events if e["agent"] == "claude-code-subagent"]
    assert len(sub) == 4
    assert {(e["trace_id"], e["run_id"]) for e in sub} == {("s1", "s1:turn0")}


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _serve(monkeypatch, files: dict[str, bytes], gated: bool = False):
    requested = []

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        requested.append((url, request.unredirected_hdrs.get("Authorization")))
        if gated and not request.unredirected_hdrs.get("Authorization"):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        if "/api/datasets/" in url:
            return _FakeResponse(json.dumps({"siblings": [{"rfilename": f} for f in [*files, "data.parquet"]]}).encode())
        for name, body in files.items():
            if url.endswith(f"/resolve/main/{name}"):
                return _FakeResponse(body)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(hf.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", "/nonexistent")
    return requested


def _jsonl(records):
    return "".join(json.dumps(r) + "\n" for r in records).encode()


def test_fetch_records_streams_only_the_limit(monkeypatch):
    _serve(monkeypatch, {"swebench.jsonl": _jsonl([_chat_record(f"t{i}", 1) for i in range(5)])})
    records, path = fetch_records("https://huggingface.co/datasets/org/name/blob/main/swebench.jsonl", limit=2)
    assert path == "swebench.jsonl"
    assert [r["id"] for r in records] == ["t0", "t1"]


def test_fetch_records_reads_gzip_and_picks_the_only_readable_file(monkeypatch):
    _serve(monkeypatch, {"traces/day.jsonl.gz": gzip.compress(_jsonl([_fi_session("a"), _fi_session("b")]))})
    records, path = fetch_records("https://huggingface.co/datasets/org/name?utm_source=x", limit=None)
    assert path == "traces/day.jsonl.gz"
    assert [r["session_id"] for r in records] == ["a", "b"]


def test_fetch_records_asks_for_a_file_when_there_are_several(monkeypatch):
    _serve(monkeypatch, {"a.jsonl": b"", "b.jsonl": b""})
    with pytest.raises(ValueError, match="a.jsonl, b.jsonl"):
        fetch_records("https://huggingface.co/datasets/org/name")


def test_gated_dataset_without_token_explains_how_to_get_access(monkeypatch):
    _serve(monkeypatch, {"x.jsonl": b"{}\n"}, gated=True)
    with pytest.raises(PermissionError, match="accept its conditions"):
        fetch_records("https://huggingface.co/datasets/org/name", file="x.jsonl")


def test_token_is_sent_as_an_unredirected_header(monkeypatch):
    requested = _serve(monkeypatch, {"x.jsonl": _jsonl([_chat_record("t1", 1)])}, gated=True)
    fetch_records("https://huggingface.co/datasets/org/name", file="x.jsonl", token="hf_secret")
    assert requested[-1][1] == "Bearer hf_secret"


def test_analyze_accepts_a_hugging_face_link(monkeypatch):
    _serve(monkeypatch, {"swebench.jsonl": _jsonl([_chat_record(f"t{i}", i) for i in range(1, 4)])})
    client = TestClient(app)
    files = client.post("/huggingface/files", data={"url": "https://huggingface.co/datasets/org/name"})
    assert files.json()["files"] == ["swebench.jsonl"]

    resp = client.post("/analyze", data={"hf_url": "https://huggingface.co/datasets/org/name", "hf_limit": "3"})
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert report["observe"]["files"] == [{"name": "org/name/swebench.jsonl", "num_events": 12, "num_records": 3}]
    assert report["observe"]["selected_workflow"] == "swebench"


def test_analyze_maps_hugging_face_errors(monkeypatch):
    _serve(monkeypatch, {"x.jsonl": b"{}\n"}, gated=True)
    client = TestClient(app)
    resp = client.post("/analyze", data={"hf_url": "https://huggingface.co/datasets/org/name", "hf_file": "x.jsonl"})
    assert resp.status_code == 403
    assert client.post("/analyze", data={}).status_code == 400
    assert client.post("/analyze", data={"hf_url": "org/name", "hf_limit": "0"}).status_code == 400


def test_redundant_tool_namespace_is_dropped():
    from ingest.chat_trajectory import convert_chat_trajectory, tool_name

    assert tool_name("swebench__swebench_execute_bash") == "swebench_execute_bash"
    assert tool_name("swebench_execute_bash") == "swebench_execute_bash"
    assert tool_name("FruityVice__get_fruit_nutrition") == "FruityVice__get_fruit_nutrition"
    assert tool_name("search__web_search") == "search__web_search"
    record = {"id": "r", "benchmark": "swebench", "messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "swebench__swebench_str_replace_editor", "arguments": '{"command": "view", "path": "x.py"}'}},
            {"id": "b", "function": {"name": "swebench_str_replace_editor", "arguments": '{"command": "view", "path": "y.py"}'}},
        ]},
    ]}
    names = {e["tool"]["name"] for e in convert_chat_trajectory(record) if e.get("tool")}
    assert names == {"swebench_str_replace_editor"}
