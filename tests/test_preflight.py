import json

import pytest
from fastapi.testclient import TestClient
from test_claude_code_ingest import _transcript

from api.main import app


def _metadata_only():
    """The shape of a session that recorded prompts and metadata but no
    agent activity (sanitized: types and counts only)."""
    user = {"type": "user", "sessionId": "s9", "promptId": "p1", "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {"role": "user", "content": "hello"}}
    return (
        [user] * 3
        + [{"type": "attachment", "sessionId": "s9"}] * 15
        + [{"type": "last-prompt", "sessionId": "s9"}] * 5
        + [{"type": "cost-state", "sessionId": "s9", "totalCostUSD": 0, "modelUsage": {}}] * 2
    )


def _jsonl(rows):
    return "\n".join(json.dumps(r) for r in rows).encode()


@pytest.mark.parametrize("endpoint", ["/analyze", "/finops"])
def test_metadata_only_transcript_says_what_it_holds(endpoint):
    response = TestClient(app).post(endpoint, files=[("file", ("s9.jsonl", _jsonl(_metadata_only())))])
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "recognized_no_agent_activity"
    assert "s9.jsonl is a Claude Code transcript with 25 records but 0 assistant responses" in body["detail"]
    (preflight,) = body["preflight"]
    assert preflight["coverage"] == {"raw_records": 25, "sessions": 1, "user_records": 3, "assistant_records": 0,
                                     "tool_use_records": 0, "converted_events": 0}
    assert [d["code"] for d in preflight["diagnostics"]] == [
        "NO_ASSISTANT_RESPONSES", "NO_TOOL_EXECUTIONS", "COST_UNVERIFIABLE"]


def test_a_metadata_only_file_beside_a_real_session_is_reported_not_rejected():
    response = TestClient(app).post("/finops", files=[
        ("file", ("real.jsonl", _jsonl(_transcript()))), ("file", ("s9.jsonl", _jsonl(_metadata_only())))])
    assert response.status_code == 200
    empty = [f for f in response.json()["files"] if f["num_events"] == 0]
    assert [(f["name"], f["preflight"]["status"]) for f in empty] == [("s9.jsonl", "recognized_no_agent_activity")]


@pytest.mark.parametrize(
    ("content", "status", "code"),
    [
        (b"  ", 400, "empty_file"),
        (b"{not json", 422, "invalid_format"),
        (b'[{"trace_id": "t"}]', 422, "unsupported_schema"),
        (b"[]", 422, "no_events"),
    ],
)
def test_upload_errors_carry_a_code(content, status, code):
    response = TestClient(app).post("/analyze", files=[("file", ("x.json", content))])
    assert (response.status_code, response.json()["code"]) == (status, code)
    assert isinstance(response.json()["detail"], str)
