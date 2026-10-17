import base64
import json
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from api.main import app
from saas import auth, db
from saas.config import get_settings
from tests.saas_helpers import configure

ISSUER = "https://idp.example.com"
ORIGIN = "https://profiler.example.com"


def _id_token(**claims):
    def part(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{part({'alg': 'RS256'})}.{part(claims)}.sig"


@pytest.fixture
def oidc(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "oidc", APP_ORIGIN=ORIGIN, OIDC_ISSUER=ISSUER,
              OIDC_CLIENT_ID="client", OIDC_CLIENT_SECRET="secret")
    auth.discovery.cache_clear()
    monkeypatch.setattr(auth, "_get_json", lambda url: {
        "issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize", "token_endpoint": f"{ISSUER}/token"})
    claims = {"iss": ISSUER, "aud": "client", "sub": "user-1", "email": "a@example.com", "exp": time.time() + 300}
    exchanged = []

    def exchange(code, verifier):
        exchanged.append((code, verifier))
        return {"id_token": _id_token(**{**claims, "nonce": state["nonce"]})}

    state = {}
    monkeypatch.setattr(auth, "exchange_code", exchange)
    yield claims, state, exchanged
    auth.discovery.cache_clear()
    get_settings.cache_clear()


def _start_login(client, state):
    res = client.get("/auth/login", follow_redirects=False)
    assert res.status_code == 303
    q = parse_qs(urlsplit(res.headers["location"]).query)
    assert q["code_challenge_method"] == ["S256"] and q["redirect_uri"] == [f"{ORIGIN}/auth/callback"]
    state["nonce"] = q["nonce"][0]
    return q["state"][0]


def test_oidc_login_sets_a_secure_httponly_session_and_me_reports_the_tenant(oidc):
    _, state, exchanged = oidc
    client = TestClient(app, base_url=ORIGIN)
    s = _start_login(client, state)
    res = client.get(f"/auth/callback?state={s}&code=abc", follow_redirects=False)
    assert res.status_code == 303 and res.headers["location"] == "/"
    cookie = res.headers["set-cookie"]
    assert cookie.startswith("__Host-session=") and "HttpOnly" in cookie and "Secure" in cookie
    assert "samesite=lax" in cookie.lower()
    assert exchanged[0][0] == "abc"
    me = client.get("/auth/me").json()
    assert me["signed_in"] and me["email"] == "a@example.com" and me["role"] == "owner"


def test_state_is_single_use(oidc):
    _, state, _ = oidc
    client = TestClient(app, base_url=ORIGIN)
    s = _start_login(client, state)
    assert client.get(f"/auth/callback?state={s}&code=abc", follow_redirects=False).status_code == 303
    assert client.get(f"/auth/callback?state={s}&code=abc", follow_redirects=False).status_code == 401


def test_unknown_state_is_refused_and_audited(oidc):
    client = TestClient(app, base_url=ORIGIN)
    assert client.get("/auth/callback?state=forged&code=abc").status_code == 401
    with db.connect() as conn:
        assert conn.execute("SELECT result FROM audit_events WHERE action = 'login'").fetchone()[0] == "denied"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"iss": "https://evil.example.com"}, "issuer"),
        ({"aud": "other-client"}, "audience"),
        ({"exp": time.time() - 10}, "expired"),
        ({"nonce": "replayed"}, "nonce"),
        ({"aud": ["client", "other"], "azp": "other"}, "authorized party"),
    ],
)
def test_bad_id_tokens_are_rejected(oidc, monkeypatch, change, reason):
    claims, state, _ = oidc
    client = TestClient(app, base_url=ORIGIN)
    s = _start_login(client, state)
    bad = {**claims, "nonce": state["nonce"], **change}
    monkeypatch.setattr(auth, "exchange_code", lambda code, verifier: {"id_token": _id_token(**bad)})
    res = client.get(f"/auth/callback?state={s}&code=abc")
    assert res.status_code == 401 and reason in res.json()["detail"]


def test_dev_login_is_off_outside_dev_mode(oidc):
    client = TestClient(app, base_url=ORIGIN)
    assert client.post("/auth/dev-login", data={"email": "x@example.com"}).status_code == 404


def test_session_cookie_from_the_browser_is_the_only_identity(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev")
    client = TestClient(app)
    assert client.get("/auth/me").json()["signed_in"] is False
    client.post("/auth/dev-login", data={"email": "A@Example.com"})
    assert client.get("/auth/me").json()["email"] == "a@example.com"
    client.cookies.set("session", "guessed-token")
    assert client.get("/auth/me").json()["signed_in"] is False
    get_settings.cache_clear()


def test_expired_session_and_logout(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev")
    client = TestClient(app)
    client.post("/auth/dev-login", data={"email": "a@example.com"})
    token = client.cookies["session"]
    assert client.post("/auth/logout", headers={"origin": "http://testserver"}, follow_redirects=False).status_code == 303
    client.cookies.set("session", token)
    assert client.get("/auth/me").json()["signed_in"] is False
    client.post("/auth/dev-login", data={"email": "a@example.com"})
    with db.connect() as conn:
        conn.execute("UPDATE sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
    assert client.get("/auth/me").json()["signed_in"] is False
    get_settings.cache_clear()


def test_new_login_issues_a_new_session_token(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "dev")
    client = TestClient(app)
    client.post("/auth/dev-login", data={"email": "a@example.com"})
    first = client.cookies["session"]
    client.post("/auth/dev-login", data={"email": "a@example.com"})
    assert client.cookies["session"] != first
    get_settings.cache_clear()
