"""Authentication: OIDC login (authorization code + PKCE), server-side
sessions, and the ``Principal`` every protected route depends on.

- The browser only ever holds a random session token in an HttpOnly,
  SameSite=Lax cookie (``__Host-`` prefixed and Secure over https). The
  database stores its SHA-256, so a leaked database can't be replayed.
- Tenant and role are looked up from the session and the membership
  table on every request; nothing the browser sends can change them.
- The ID token comes straight from the provider's token endpoint over
  verified TLS with client authentication, which OIDC Core 3.1.3.7 allows
  in place of checking its signature; issuer, audience, expiry and nonce
  are still checked.
- State-changing requests carrying the session cookie must come from
  ``APP_ORIGIN`` (``Origin`` / ``Referer`` check) -- the CSRF control on
  top of SameSite.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cache

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse

from saas import db
from saas.config import get_settings

AUTH_REQUEST_TTL_S = 600
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    role: str
    email: str | None
    platform_admin: bool


def cookie_name() -> str:
    # __Host- cookies must be Secure, Path=/ and have no Domain: the browser
    # then refuses one set by a sibling subdomain (session fixation).
    return "__Host-session" if get_settings().secure_cookies else "session"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(response, user_id: str, tenant_id: str) -> None:
    """A fresh session token for every login (no fixation), set as the
    cookie on ``response``."""
    s = get_settings()
    token = secrets.token_urlsafe(32)
    expires = datetime.now(UTC) + timedelta(seconds=s.session_ttl_s)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, tenant_id, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (_hash(token), user_id, tenant_id, db.now(), expires.isoformat(timespec="seconds")),
        )
    response.set_cookie(
        cookie_name(), token, max_age=s.session_ttl_s, path="/",
        httponly=True, secure=s.secure_cookies, samesite="lax",
    )


def _check_origin(request: Request) -> None:
    if request.method in SAFE_METHODS:
        return
    origin = get_settings().app_origin
    sent = request.headers.get("origin")
    if sent is None:
        referer = request.headers.get("referer", "")
        sent = origin if referer == origin or referer.startswith(origin + "/") else referer
    if sent != origin:
        raise HTTPException(status_code=403, detail="cross-site request refused")


def principal(request: Request) -> Principal:
    """The signed-in user and their active tenant, from the session cookie.
    401 without a valid session; 403 for a cross-site write."""
    s = get_settings()
    if not s.saas:
        raise HTTPException(status_code=404, detail="not available in demo mode")
    token = request.cookies.get(cookie_name())
    if not token:
        raise HTTPException(status_code=401, detail="sign in first")
    with db.connect() as conn:
        row = conn.execute(
            "SELECT s.user_id, s.tenant_id, s.expires_at, m.role, u.email, u.issuer, u.subject "
            "FROM sessions s JOIN users u ON u.id = s.user_id "
            "JOIN memberships m ON m.user_id = s.user_id AND m.tenant_id = s.tenant_id "
            "WHERE s.token_hash = ?",
            (_hash(token),),
        ).fetchone()
    if row is None or row["expires_at"] <= db.now():
        raise HTTPException(status_code=401, detail="session expired; sign in again")
    _check_origin(request)
    admins = s.platform_admins
    return Principal(
        user_id=row["user_id"],
        tenant_id=row["tenant_id"],
        role=row["role"],
        email=row["email"],
        platform_admin=bool(admins) and (f"{row['issuer']}|{row['subject']}" in admins or row["email"] in admins),
    )


def platform_admin(p: Principal = Depends(principal)) -> Principal:  # noqa: B008
    if not p.platform_admin:
        raise HTTPException(status_code=403, detail="platform admins only")
    return p


def optional_principal(request: Request) -> Principal | None:
    """In demo mode ``None`` (the public demo); in SaaS mode the principal,
    so a legacy route can require login only when customers use it."""
    return principal(request) if get_settings().saas else None


# --- OIDC provider ---------------------------------------------------------


def _get_json(url: str) -> dict:
    if not url.startswith("https://"):
        raise RuntimeError(f"refusing non-https OIDC URL {url}")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=10) as r:
        return json.load(r)


@cache
def discovery() -> dict:
    issuer = get_settings().oidc_issuer
    doc = _get_json(f"{issuer}/.well-known/openid-configuration")
    if doc.get("issuer", "").rstrip("/") != issuer:
        raise RuntimeError("OIDC discovery issuer does not match OIDC_ISSUER")
    return doc


def exchange_code(code: str, code_verifier: str) -> dict:
    """POST the code to the token endpoint (client_secret_basic) and return
    the token response."""
    s = get_settings()
    endpoint = discovery()["token_endpoint"]
    if not endpoint.startswith("https://"):
        raise RuntimeError("OIDC token endpoint is not https")
    basic = base64.b64encode(
        f"{urllib.parse.quote(s.oidc_client_id, safe='')}:{urllib.parse.quote(s.oidc_client_secret, safe='')}".encode()
    ).decode()
    body = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": f"{s.app_origin}/auth/callback",
        "code_verifier": code_verifier,
    }).encode()
    request = urllib.request.Request(
        endpoint, data=body,
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as r:
        return json.load(r)


def _b64url_json(part: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def id_token_claims(id_token: str, nonce: str) -> dict:
    """Claims of an ID token received directly from the token endpoint,
    after the checks OIDC Core 3.1.3.7 requires besides the signature."""
    s = get_settings()
    try:
        _, payload, _ = id_token.split(".")
        claims = _b64url_json(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="malformed ID token") from exc
    aud = claims.get("aud")
    audiences = aud if isinstance(aud, list) else [aud]
    now = time.time()
    problems = [
        str(claims.get("iss", "")).rstrip("/") != s.oidc_issuer and "issuer",
        s.oidc_client_id not in audiences and "audience",
        len(audiences) > 1 and claims.get("azp") != s.oidc_client_id and "authorized party",
        (not isinstance(claims.get("exp"), (int, float)) or claims["exp"] < now) and "expired",
        not secrets.compare_digest(str(claims.get("nonce", "")), nonce) and "nonce",
        not claims.get("sub") and "subject",
    ]
    failed = [p for p in problems if p]
    if failed:
        raise HTTPException(status_code=401, detail=f"ID token rejected ({', '.join(failed)})")
    return claims


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def _sign_in(issuer: str, subject: str, email: str | None) -> RedirectResponse:
    with db.connect() as conn:
        user_id, created = db.upsert_user(conn, issuer, subject, email)
        tenant_id = db.default_tenant(conn, user_id, email or "My organization")
        db.audit(conn, "login", tenant_id=tenant_id, user_id=user_id, detail="first login" if created else None)
    response = RedirectResponse("/", status_code=303)
    start_session(response, user_id, tenant_id)
    return response


router = APIRouter(prefix="/auth", include_in_schema=False)


@router.get("/login")
def login() -> RedirectResponse:
    s = get_settings()
    if s.auth_mode != "oidc":
        raise HTTPException(status_code=404, detail="OIDC login is not configured")
    state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    verifier, challenge = _pkce()
    cutoff = (datetime.now(UTC) - timedelta(seconds=AUTH_REQUEST_TTL_S)).isoformat(timespec="seconds")
    with db.connect() as conn:
        conn.execute("DELETE FROM auth_requests WHERE created_at < ?", (cutoff,))
        conn.execute(
            "INSERT INTO auth_requests (state, nonce, code_verifier, created_at) VALUES (?, ?, ?, ?)",
            (state, nonce, verifier, db.now()),
        )
    query = urllib.parse.urlencode({
        "response_type": "code",
        "client_id": s.oidc_client_id,
        "redirect_uri": f"{s.app_origin}/auth/callback",
        "scope": s.oidc_scopes,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })
    return RedirectResponse(f"{discovery()['authorization_endpoint']}?{query}", status_code=303)


@router.get("/callback")
def callback(state: str = "", code: str = "", error: str = "") -> RedirectResponse:
    s = get_settings()
    if s.auth_mode != "oidc":
        raise HTTPException(status_code=404, detail="OIDC login is not configured")
    cutoff = (datetime.now(UTC) - timedelta(seconds=AUTH_REQUEST_TTL_S)).isoformat(timespec="seconds")
    with db.connect() as conn:
        # one use: the row is deleted whether or not the login succeeds
        row = conn.execute(
            "DELETE FROM auth_requests WHERE state = ? AND created_at >= ? RETURNING nonce, code_verifier",
            (state, cutoff),
        ).fetchone()
        if row is None or error or not code:
            db.audit(conn, "login", "denied", detail=error or "unknown or expired state")
    if row is None or error or not code:
        raise HTTPException(status_code=401, detail="sign-in failed or expired; start again")
    tokens = exchange_code(code, row["code_verifier"])
    claims = id_token_claims(tokens.get("id_token", ""), row["nonce"])
    email = claims.get("email") if claims.get("email_verified", True) else None
    return _sign_in(s.oidc_issuer, str(claims["sub"]), email)


@router.post("/dev-login")
def dev_login(email: str = Form(...)) -> RedirectResponse:
    """Local testing only (``AUTH_MODE=dev`` refuses a non-localhost
    origin): sign in as ``email`` with no provider."""
    if get_settings().auth_mode != "dev":
        raise HTTPException(status_code=404, detail="not found")
    return _sign_in("dev", email.strip().lower(), email.strip().lower())


@router.post("/logout")
def logout(request: Request) -> RedirectResponse:
    token = request.cookies.get(cookie_name())
    response = RedirectResponse("/", status_code=303)
    if token:
        _check_origin(request)
        with db.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash(token),))
    response.delete_cookie(cookie_name(), path="/", secure=get_settings().secure_cookies, httponly=True, samesite="lax")
    return response


@router.get("/me")
def me(request: Request) -> dict:
    s = get_settings()
    if not s.saas:
        return {"auth_mode": s.auth_mode, "signed_in": False}
    try:
        p = principal(request)
    except HTTPException:
        return {"auth_mode": s.auth_mode, "signed_in": False}
    with db.connect() as conn:
        (name,) = conn.execute("SELECT name FROM tenants WHERE id = ?", (p.tenant_id,)).fetchone()
    return {"auth_mode": s.auth_mode, "signed_in": True, "email": p.email, "tenant": name, "role": p.role,
            "platform_admin": p.platform_admin}
