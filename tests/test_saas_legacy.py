"""The original routes once customers sign in: login required, history
per tenant, QA tools for platform admins, no server Hugging Face token."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from ingest import huggingface
from tests.saas_helpers import configure

ORIGIN = {"origin": "http://testserver"}


def _client(email):
    client = TestClient(app)
    client.headers.update(ORIGIN)
    client.post("/auth/dev-login", data={"email": email})
    return client


@pytest.fixture
def trace():
    # generated in demo mode, where the generator is public
    return TestClient(app).post("/generate", json={"agent_count": 1, "actions_per_agent": [10], "seed": 1}).json()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/analyze"),
        ("post", "/finops"),
        ("post", "/huggingface/files"),
        ("get", "/behavior"),
        ("get", "/behavior/history"),
        ("delete", "/behavior/history"),
    ],
)
def test_customer_routes_need_a_session_in_saas_mode(monkeypatch, tmp_path, method, path):
    configure(monkeypatch, tmp_path, "dev")
    assert getattr(TestClient(app), method)(path).status_code == 401


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/generate", {"agent_count": 1, "actions_per_agent": [10]}),
        ("/validate_seed", {"agent_count": 1, "actions_per_agent": [3]}),
        ("/validate_seed_range", {"agent_count": 1, "actions_per_agent": [3], "seed_end": 2}),
        ("/run_sweep", None),
        ("/poc/run", None),
        ("/solve", {}),
    ],
)
def test_qa_tools_are_for_platform_admins(monkeypatch, tmp_path, path, body):
    configure(monkeypatch, tmp_path, "dev", PLATFORM_ADMINS="root@ops.example")
    assert TestClient(app).post(path, json=body).status_code == 401
    assert _client("alice@a.example").post(path, json=body).status_code == 403
    if path == "/generate":
        assert _client("root@ops.example").post(path, json=body).status_code == 200


def test_history_is_per_tenant(monkeypatch, tmp_path, trace):
    configure(monkeypatch, tmp_path, "dev")
    alice, bob = _client("alice@a.example"), _client("bob@b.example")
    res = alice.post("/analyze", files={"file": ("run.jsonl", trace["jsonl"])}, data={"keep_history": "true"})
    assert res.status_code == 200 and res.json()["history"]["added"] > 0
    assert alice.get("/behavior/history").json()["total"] > 0
    assert "path" not in alice.get("/behavior/history").json()
    assert bob.get("/behavior/history").json()["total"] == 0
    # same content kept by bob is bob's own copy
    bob.post("/analyze", files={"file": ("run.jsonl", trace["jsonl"])}, data={"keep_history": "true"})
    assert bob.delete("/behavior/history").json()["deleted"] > 0
    assert alice.get("/behavior/history").json()["total"] > 0


def test_demo_history_is_unchanged_and_separate(monkeypatch, tmp_path, trace):
    configure(monkeypatch, tmp_path, "demo")
    demo = TestClient(app)
    demo.post("/analyze", files={"file": ("run.jsonl", trace["jsonl"])}, data={"keep_history": "true"})
    total = demo.get("/behavior/history").json()["total"]
    assert total > 0
    configure(monkeypatch, tmp_path, "dev")
    assert _client("alice@a.example").get("/behavior/history").json()["total"] == 0


def test_signed_in_customers_never_borrow_the_server_hf_token(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev")
    monkeypatch.setenv("HF_TOKEN", "server-secret")
    used = []

    def fake_open(url, token, ref):
        used.append(token)
        raise PermissionError("denied")

    monkeypatch.setattr(huggingface, "_open", fake_open)
    alice = _client("alice@a.example")
    alice.post("/huggingface/files", data={"url": "https://huggingface.co/datasets/org/gated"})
    alice.post("/huggingface/files", data={"url": "https://huggingface.co/datasets/org/gated", "token": "own"})
    assert used == [None, "own"]
    configure(monkeypatch, tmp_path, "demo")
    TestClient(app).post("/huggingface/files", data={"url": "https://huggingface.co/datasets/org/gated"})
    assert used[-1] == "server-secret"  # the operator's own demo keeps its behavior
