from fastapi.testclient import TestClient

from api.main import _STATIC_DIR, app
from saas.headers import script_hashes


def test_page_scripts_are_pinned_by_hash_and_framing_is_denied():
    res = TestClient(app).get("/")
    csp = res.headers["content-security-policy"]
    hashes = script_hashes((_STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    assert len(hashes) == 1 and f"script-src {hashes[0]}" in csp
    assert "'unsafe-inline'" not in csp.split("script-src", 1)[1].split(";", 1)[0]
    assert "frame-ancestors 'none'" in csp and res.headers["x-frame-options"] == "DENY"
    assert res.headers["x-content-type-options"] == "nosniff"
    assert "cache-control" not in res.headers


def test_api_responses_are_not_cached():
    res = TestClient(app).get("/health")
    assert res.headers["cache-control"] == "no-store"
    assert res.headers["referrer-policy"] == "same-origin"
