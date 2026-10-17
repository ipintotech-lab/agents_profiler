"""Files API: tenant isolation and upload security, end to end through
the HTTP app (no browser -- every check is the backend's)."""
import gzip
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from saas import db, files
from saas.config import MB
from tests.saas_helpers import configure

ORIGIN = {"origin": "http://testserver"}
TRACE = b'{"agent": "a", "node": "n", "input": 1}\n{"agent": "a", "node": "n", "input": 2}\n'


def _client(email: str) -> TestClient:
    client = TestClient(app)
    client.headers.update(ORIGIN)
    assert client.post("/auth/dev-login", data={"email": email}, follow_redirects=False).status_code == 303
    return client


@pytest.fixture
def two_tenants(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev")
    files.upload_limiter.reset()
    alice, bob = _client("alice@a.example"), _client("bob@b.example")
    res = alice.post("/api/files", files={"file": ("trace.jsonl", TRACE, "application/json")})
    assert res.status_code == 201, res.text
    return alice, bob, res.json()["files"][0]["id"]


def _upload(client, *parts):
    return client.post("/api/files", files=[("file", part) for part in parts])


# --- tenant isolation -------------------------------------------------------


def test_owner_can_use_their_file(two_tenants):
    alice, _, fid = two_tenants
    assert alice.get(f"/api/files/{fid}").json()["status"] == "READY"
    assert [f["id"] for f in alice.get("/api/files").json()["files"]] == [fid]
    res = alice.get(f"/api/files/{fid}/download")
    assert res.content == TRACE and res.headers["content-type"] == "application/octet-stream"
    assert res.headers["content-disposition"].startswith("attachment;")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/files/{fid}"),
        ("get", "/api/files/{fid}/download"),
        ("post", "/api/files/{fid}/download-link"),
        ("delete", "/api/files/{fid}"),
    ],
)
def test_other_tenant_gets_404_for_every_operation(two_tenants, method, path):
    alice, bob, fid = two_tenants
    res = getattr(bob, method)(path.format(fid=fid))
    assert res.status_code == 404
    assert bob.get("/api/files").json()["files"] == []
    assert alice.get(f"/api/files/{fid}").status_code == 200  # untouched


def test_unknown_and_other_tenant_ids_look_the_same(two_tenants):
    _, bob, fid = two_tenants
    assert bob.get(f"/api/files/{fid}").json() == bob.get(f"/api/files/{'0' * 32}").json()


def test_browser_supplied_tenant_is_ignored(two_tenants):
    _, bob, fid = two_tenants
    with db.connect() as conn:
        alice_tenant = conn.execute("SELECT tenant_id FROM files WHERE id = ?", (fid,)).fetchone()[0]
    for extra in ({"tenant_id": alice_tenant}, {"X-Tenant-ID": alice_tenant}):
        assert bob.get(f"/api/files/{fid}", params=extra, headers=extra).status_code == 404


def test_member_cannot_delete_another_members_file(two_tenants):
    _, _, fid = two_tenants
    carol = _client("carol@a.example")
    with db.connect() as conn:
        tenant, = conn.execute("SELECT tenant_id FROM files WHERE id = ?", (fid,)).fetchone()
        carol_id, = conn.execute("SELECT id FROM users WHERE subject = 'carol@a.example'").fetchone()
        conn.execute("DELETE FROM memberships WHERE user_id = ?", (carol_id,))
        db.add_member(conn, carol_id, tenant, "member")
        conn.execute("UPDATE sessions SET tenant_id = ? WHERE user_id = ?", (tenant, carol_id))
    assert carol.get(f"/api/files/{fid}").status_code == 200
    assert carol.delete(f"/api/files/{fid}").status_code == 403


def test_delete_removes_the_object_and_derived_results(two_tenants, tmp_path):
    alice, _, fid = two_tenants
    with db.connect() as conn:
        row = conn.execute("SELECT tenant_id, storage_key, uploaded_by FROM files WHERE id = ?", (fid,)).fetchone()
        job_id, result_id = db.new_id(), db.new_id()
        conn.execute("INSERT INTO jobs (id, tenant_id, file_id, created_by, kind, params, status, created_at) "
                     "VALUES (?, ?, ?, ?, 'analyze', '{}', 'COMPLETED', ?)",
                     (job_id, row["tenant_id"], fid, row["uploaded_by"], db.now()))
        result_key = f"tenants/{row['tenant_id']}/results/{result_id}"
        conn.execute("INSERT INTO results (id, tenant_id, job_id, file_id, storage_key, size, created_at) "
                     "VALUES (?, ?, ?, ?, ?, 2, ?)", (result_id, row["tenant_id"], job_id, fid, result_key, db.now()))
    (tmp_path / "storage" / result_key).parent.mkdir(parents=True)
    (tmp_path / "storage" / result_key).write_text("{}")
    assert alice.delete(f"/api/files/{fid}").json() == {"deleted": fid, "results_deleted": 1}
    assert not (tmp_path / "storage" / row["storage_key"]).exists()
    assert not (tmp_path / "storage" / result_key).exists()
    assert alice.get(f"/api/files/{fid}").status_code == 404
    assert alice.get(f"/api/files/{fid}/download").status_code == 404


def test_actions_are_audited_without_content(two_tenants):
    alice, _, fid = two_tenants
    alice.get(f"/api/files/{fid}/download")
    with db.connect() as conn:
        rows = conn.execute("SELECT action, result, detail FROM audit_events").fetchall()
    assert ("file.upload", "ok") in [(r["action"], r["result"]) for r in rows]
    assert ("file.download", "ok") in [(r["action"], r["result"]) for r in rows]
    assert all(TRACE.decode()[:20] not in (r["detail"] or "") for r in rows)


# --- authentication / CSRF --------------------------------------------------


def test_unauthenticated_requests_are_refused(two_tenants):
    _, _, fid = two_tenants
    anon = TestClient(app)
    assert anon.post("/api/files", files={"file": ("a.json", b"{}")}).status_code == 401
    assert anon.get("/api/files").status_code == 401
    assert anon.get(f"/api/files/{fid}/download").status_code == 401


def test_cross_site_writes_are_refused(two_tenants):
    alice, _, fid = two_tenants
    evil = {"origin": "https://evil.example"}
    assert alice.post("/api/files", files={"file": ("a.json", b"{}")}, headers=evil).status_code == 403
    assert alice.delete(f"/api/files/{fid}", headers=evil).status_code == 403
    assert alice.get(f"/api/files/{fid}").status_code == 200


def test_files_api_is_off_in_demo_mode(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "demo")
    assert TestClient(app).get("/api/files").status_code == 404


# --- upload security --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "stored"),
    [
        ("../../etc/passwd.json", "passwd.json"),
        ("..\\..\\file.json", "file.json"),
        ("x" * 1000 + ".json", None),
        ("résumé 日本.json", "résumé 日本.json"),
    ],
)
def test_hostile_filenames_are_metadata_only(two_tenants, tmp_path, name, stored):
    alice, _, _ = two_tenants
    res = _upload(alice, (name, b'{"a": 1}', "application/json"))
    out = res.json()["files"][0]
    assert out["status"] == "READY"
    assert out["filename"] == stored if stored else len(out["filename"]) <= 255
    with db.connect() as conn:
        key = conn.execute("SELECT storage_key FROM files WHERE id = ?", (out["id"],)).fetchone()[0]
    assert key.endswith(out["id"]) and ".." not in key
    assert not any(p.name.startswith("passwd") for p in tmp_path.rglob("*"))
    disposition = alice.get(f"/api/files/{out['id']}/download").headers["content-disposition"]
    assert "/" not in disposition.split("filename=")[1].split(";")[0]


@pytest.mark.parametrize(
    ("part", "reason"),
    [
        (("empty.json", b"", "application/json"), "empty"),
        (("fake.json", b"\x7fELF\x02\x01\x01", "application/json"), "executable"),
        (("spoofed.json", b"MZ\x90\x00", "application/json"), "executable"),
        (("data.jsonl", gzip.compress(TRACE), "application/x-ndjson"), "gzip"),
        (("run.exe", b'{"a": 1}', "application/json"), "extension"),
        (("page.json", b"<script>alert(1)</script>", "application/json"), "not valid"),
    ],
)
def test_bad_content_is_rejected_whatever_the_name_or_content_type(two_tenants, tmp_path, part, reason):
    alice, _, _ = two_tenants
    out = _upload(alice, part).json()["files"][0]
    assert out["status"] == "REJECTED" and reason in out["reason"]
    assert alice.get(f"/api/files/{out['id']}/download").status_code == 409
    assert not list((tmp_path / "storage").rglob(out["id"]))


def test_duplicates_are_flagged(two_tenants):
    alice, _, fid = two_tenants
    assert _upload(alice, ("again.jsonl", TRACE)).json()["files"][0]["duplicate_of"] == fid


def test_oversized_files_and_bodies_are_refused(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev", MAX_UPLOAD_BYTES=100, MAX_FILES_PER_UPLOAD=2)
    alice = _client("alice@a.example")
    assert _upload(alice, ("big.json", json.dumps({"a": "x" * 200}).encode())).status_code == 413
    huge = b"x" * (2 * 100 + 2 * MB)
    assert alice.post("/api/files", content=huge, headers={"content-type": "application/json"}).status_code == 413
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0


def test_too_many_files_per_upload(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev", MAX_FILES_PER_UPLOAD=2)
    alice = _client("alice@a.example")
    parts = [(f"{i}.json", json.dumps({"i": i}).encode()) for i in range(3)]
    assert _upload(alice, *parts).status_code == 413


def test_tenant_storage_and_daily_quotas(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev", TENANT_STORAGE_BYTES=30, DAILY_UPLOAD_BYTES=1000)
    alice = _client("alice@a.example")
    assert _upload(alice, ("a.json", b'{"a": "0123456789"}')).json()["files"][0]["status"] == "READY"
    res = _upload(alice, ("b.json", b'{"b": "0123456789"}'))
    assert res.status_code == 413 and "storage" in res.json()["detail"]
    configure(monkeypatch, tmp_path / "2", "dev", DAILY_UPLOAD_BYTES=30)
    bob = _client("bob@b.example")
    _upload(bob, ("a.json", b'{"a": "0123456789"}'))
    bob.delete(f"/api/files/{bob.get('/api/files').json()['files'][0]['id']}")
    res = _upload(bob, ("b.json", b'{"b": "0123456789"}'))
    assert res.status_code == 413 and "daily" in res.json()["detail"]


def test_upload_rate_limit(two_tenants, monkeypatch):
    alice, _, _ = two_tenants
    monkeypatch.setattr(files.upload_limiter, "limit", 2)
    files.upload_limiter.reset()
    codes = [_upload(alice, (f"{i}.json", json.dumps({"i": i}).encode())).status_code for i in range(3)]
    assert codes == [201, 201, 429]


def test_download_links_are_one_use_and_expire(two_tenants):
    alice, _, fid = two_tenants
    url = alice.post(f"/api/files/{fid}/download-link").json()["url"]
    anon = TestClient(app)
    assert anon.get(url).content == TRACE
    assert anon.get(url).status_code == 404
    url = alice.post(f"/api/files/{fid}/download-link").json()["url"]
    with db.connect() as conn:
        conn.execute("UPDATE download_tokens SET expires_at = '2000-01-01T00:00:00+00:00'")
    assert anon.get(url).status_code == 404
    assert anon.get("/api/downloads/forged").status_code == 404


def test_scanner_verdict_rejects_infected_files(two_tenants, monkeypatch):
    from saas.scanner import ScanResult

    class Infected:
        def scan(self, data):
            return ScanResult(clean=False, scanned=True, signature="Eicar")

    monkeypatch.setattr(files, "get_scanner", Infected)
    alice, _, _ = two_tenants
    out = _upload(alice, ("eicar.json", b'{"x": 1}')).json()["files"][0]
    assert out["status"] == "REJECTED" and "malware" in out["reason"]
