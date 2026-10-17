"""Jobs: queued by the API, run by the worker in a sandboxed child, and
tenant-scoped like files."""
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from api.main import app
from saas import db, files, worker
from saas.config import get_settings
from tests.saas_helpers import configure

ORIGIN = {"origin": "http://testserver"}


def _client(email):
    client = TestClient(app)
    client.headers.update(ORIGIN)
    client.post("/auth/dev-login", data={"email": email})
    return client


@pytest.fixture
def setup(monkeypatch, tmp_path):
    # demo mode: the generator is a public QA tool there
    trace = TestClient(app).post("/generate", json={"agent_count": 1, "actions_per_agent": [10], "seed": 1}).json()
    configure(monkeypatch, tmp_path, "dev", JOB_TIMEOUT_S=60, JOB_MEMORY_BYTES=1 << 31)
    files.upload_limiter.reset()
    alice, bob = _client("alice@a.example"), _client("bob@b.example")
    out = alice.post("/api/files", files={"file": ("run.jsonl", trace["jsonl"].encode())}).json()["files"][0]
    assert out["status"] == "READY", out
    return alice, bob, out["id"]


def test_job_runs_in_the_worker_and_the_result_is_tenant_scoped(setup):
    alice, bob, fid = setup
    job = alice.post(f"/api/files/{fid}/jobs", json={"demand_model": "max"})
    assert job.status_code == 202 and job.json()["status"] == "QUEUED"
    job_id = job.json()["id"]
    assert worker.run_once() and not worker.run_once()
    done = alice.get(f"/api/jobs/{job_id}").json()
    assert done["status"] == "COMPLETED", done
    report = alice.get(f"/api/results/{done['result_id']}").json()
    assert report["formula"]["defaults"]["demand_model"] == "max"
    assert report["observe"]["files"][0]["name"] == "run.jsonl"
    assert alice.get(f"/api/files/{fid}").json()["status"] == "COMPLETED"
    # bob sees none of it
    assert bob.get(f"/api/jobs/{job_id}").status_code == 404
    assert bob.get(f"/api/results/{done['result_id']}").status_code == 404
    assert bob.get("/api/jobs").json()["jobs"] == []
    assert bob.post(f"/api/files/{fid}/jobs", json={}).status_code == 404  # no cross-tenant reanalyze


def test_reanalysis_and_finops_jobs(setup):
    alice, _, fid = setup
    alice.post(f"/api/files/{fid}/jobs", json={})
    worker.run_once()
    job_id = alice.post(f"/api/files/{fid}/jobs", json={"kind": "finops", "runs_per_month": 100}).json()["id"]
    worker.run_once()
    done = alice.get(f"/api/jobs/{job_id}").json()
    assert done["status"] == "COMPLETED"
    assert "files" in alice.get(f"/api/results/{done['result_id']}").json()


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "shell"},
        {"command": "rm -rf /"},
        {"output_match": "regex"},
        {"capacity_headroom": -1},
        {"token_prices": [-1, 2]},
        {"tenant_id": "0" * 32},
    ],
)
def test_job_parameters_are_whitelisted(setup, body):
    alice, _, fid = setup
    assert alice.post(f"/api/files/{fid}/jobs", json=body).status_code == 422


def test_active_job_limit(setup, monkeypatch, tmp_path):
    alice, _, fid = setup
    configure(monkeypatch, tmp_path, "dev", MAX_ACTIVE_JOBS=2)
    codes = [alice.post(f"/api/files/{fid}/jobs", json={}).status_code for _ in range(3)]
    assert codes == [202, 202, 429]


def test_a_failing_file_fails_the_job_without_echoing_content(setup, monkeypatch):
    alice, _, _ = setup
    secret = '{"api_key": "pretend-secret-value"}'
    fid = alice.post("/api/files", files={"file": ("bad.json", secret.encode())}).json()["files"][0]["id"]
    job_id = alice.post(f"/api/files/{fid}/jobs", json={}).json()["id"]
    worker.run_once()
    done = alice.get(f"/api/jobs/{job_id}").json()
    assert done["status"] == "FAILED" and done["error"] and "pretend-secret-value" not in done["error"]
    assert alice.get(f"/api/files/{fid}").json()["status"] == "FAILED"


def test_sandbox_child_gets_no_secrets_and_is_resource_limited(setup, monkeypatch):
    # the worker's own environment has secrets; the child must not see them
    monkeypatch.setenv("HF_TOKEN", "server-secret")
    probe = (
        "import os, resource, sys; sys.stdout.write(repr([os.environ.get('HF_TOKEN'), os.environ.get('SAAS_DB'), "
        "resource.getrlimit(resource.RLIMIT_AS)[0], resource.getrlimit(resource.RLIMIT_FSIZE)[0]]))"
    )
    seen = {}
    real_run = subprocess.run

    def run(argv, **kwargs):
        seen.update(kwargs)
        return real_run([argv[0], "-I", "-c", probe], **{k: v for k, v in kwargs.items() if k != "input"})

    monkeypatch.setattr(worker.subprocess, "run", run)
    outcome = worker.run_sandboxed("analyze", {}, "x.jsonl", b"{}")
    assert outcome == {"error": "the analysis failed"}  # the probe's output isn't JSON
    assert set(seen["env"]) == {"PATH", "LANG", "PYTHONHASHSEED", "TMPDIR"}
    child = real_run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=True,
                     env={"PATH": "/usr/bin:/bin"}, preexec_fn=seen["preexec_fn"])
    assert child.stdout == repr([None, None, get_settings().job_memory_bytes, worker.MAX_CHILD_FILE_BYTES])


def test_memory_hungry_job_is_stopped(setup, monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev", JOB_MEMORY_BYTES=512 << 20, JOB_TIMEOUT_S=30)
    real_run = subprocess.run
    monkeypatch.setattr(worker.subprocess, "run",
                        lambda argv, **kw: real_run([argv[0], "-I", "-c", "x = bytearray(1 << 30)"], **kw))
    outcome = worker.run_sandboxed("analyze", {}, "x.jsonl", b"{}")
    assert outcome == {"error": "the analysis exceeded its CPU or memory limit"}


@pytest.mark.parametrize(
    ("code", "error"),
    [
        # busy: RLIMIT_CPU kills it (wall clock widened so it doesn't win the race)
        ("while True: pass", "the analysis exceeded its CPU or memory limit"),
        # idle: only the wall-clock timeout catches it
        ("import time; time.sleep(30)", "the analysis took longer than 1s"),
    ],
)
def test_slow_jobs_are_stopped(setup, monkeypatch, tmp_path, code, error):
    configure(monkeypatch, tmp_path, "dev", JOB_TIMEOUT_S=1)
    real_run = subprocess.run
    wall = 10 if "while" in code else 1
    monkeypatch.setattr(worker.subprocess, "run",
                        lambda argv, **kw: real_run([argv[0], "-I", "-c", code], **{**kw, "timeout": wall}))
    assert worker.run_sandboxed("analyze", {}, "x.jsonl", b"{}") == {"error": error}


def test_job_for_a_file_deleted_while_queued_is_cancelled(setup):
    alice, _, fid = setup
    job_id = alice.post(f"/api/files/{fid}/jobs", json={}).json()["id"]
    alice.delete(f"/api/files/{fid}")
    assert worker.run_once() is False  # delete already cancelled it; nothing queued
    assert alice.get(f"/api/jobs/{job_id}").json()["status"] == "CANCELLED"
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0
