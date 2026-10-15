import json
import random

from test_claude_code_ingest import _transcript

from ingest import TraceEvent, hash_input_values, normalize_raw_json, parse_raw_text
from ingest.convert import main as convert_main
from profiler.pipeline import run_pipeline
from rawdata_gen import generate_configured
from rawdata_gen.agents.common import to_jsonl


def test_hash_input_values_keeps_keys_and_hides_values():
    rows = hash_input_values([{"node": "Bash", "input_state": {"command": "export TOKEN=secret", "n": 3}}])
    state = rows[0]["input_state"]
    assert set(state) == {"command", "n"}
    assert all(v.startswith("sha256:") for v in state.values())
    assert "secret" not in json.dumps(rows)


def test_hash_input_values_maps_equal_values_to_equal_hashes():
    a, b, c = hash_input_values(
        [{"input_state": {"x": {"k": 1, "j": 2}}}, {"input_state": {"x": {"j": 2, "k": 1}}}, {"input_state": {"x": 2}}]
    )
    assert a["input_state"] == b["input_state"]
    assert a["input_state"] != c["input_state"]


def test_hashing_inputs_does_not_change_pipeline_results():
    runs = generate_configured(3, [6, 6, 6], total_lines=1500, rng=random.Random(7))
    rows = normalize_raw_json(parse_raw_text(to_jsonl(runs)))
    plain = run_pipeline([TraceEvent.model_validate(r) for r in rows])
    hashed = run_pipeline([TraceEvent.model_validate(r) for r in hash_input_values(rows)])
    # Verdicts must match; only the displayed input values differ (hashed).
    def verdicts(report):
        return [
            (r["node"], r["classification"], r["score"], {k: v for k, v in r["evidence"].items() if k != "inputs"})
            for r in report["recommend"]
        ]

    assert verdicts(plain) == verdicts(hashed)


def test_convert_cli_writes_hashed_trace_events(tmp_path):
    source = tmp_path / "session.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in _transcript()))
    out = tmp_path / "traces.jsonl"
    assert convert_main([str(source), "--hash-inputs", "-o", str(out)]) == 0

    text = out.read_text()
    assert '"ls"' not in text  # the Bash command
    assert "list files" not in text  # the user prompt
    events = [TraceEvent.model_validate_json(line) for line in text.splitlines()]
    bash = next(e for e in events if e.node == "Bash: ls")
    assert bash.input_state["command"].startswith("sha256:")


def test_convert_cli_reports_unparseable_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('[{"trace_id": "t"}]')
    assert convert_main([str(bad)]) == 1
    assert "bad.json" in capsys.readouterr().err


def test_hash_inputs_also_hashes_the_user_input():
    from ingest import hash_input_values

    (row,) = hash_input_values([{"user_input": "my secret prompt", "input_state": {"a": 1}}])
    assert row["user_input"].startswith("sha256:")
