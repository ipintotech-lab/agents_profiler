import pytest

from saas import db
from saas.config import get_settings
from tests.saas_helpers import configure


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path)
    yield
    get_settings.cache_clear()


def test_dev_mode_refuses_a_public_origin(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ORIGIN", "https://profiler.example.com")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="localhost"):
        get_settings()


def test_oidc_mode_needs_provider_settings_and_https(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "oidc")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="OIDC_ISSUER"):
        get_settings()
    for k, v in {"OIDC_ISSUER": "https://idp.example.com", "OIDC_CLIENT_ID": "c", "OIDC_CLIENT_SECRET": "s"}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="https"):
        get_settings()
    monkeypatch.setenv("APP_ORIGIN", "https://profiler.example.com")
    get_settings.cache_clear()
    assert get_settings().secure_cookies


def test_first_login_creates_a_tenant_the_user_owns_and_later_logins_reuse_it():
    with db.connect() as conn:
        uid, created = db.upsert_user(conn, "iss", "sub-1", "a@example.com")
        tid = db.default_tenant(conn, uid, "a@example.com")
        assert created and db.role_in(conn, uid, tid) == "owner"
        again, created_again = db.upsert_user(conn, "iss", "sub-1", "a@example.com")
        assert again == uid and not created_again and db.default_tenant(conn, uid, "x") == tid
        other, _ = db.upsert_user(conn, "iss", "sub-2", None)
        assert db.default_tenant(conn, other, "b") != tid and db.role_in(conn, other, tid) is None


def test_same_subject_from_another_issuer_is_another_user():
    with db.connect() as conn:
        a, _ = db.upsert_user(conn, "https://idp-a", "123", None)
        b, _ = db.upsert_user(conn, "https://idp-b", "123", None)
    assert a != b


def test_roles_are_restricted_and_audit_detail_is_truncated():
    with db.connect() as conn:
        uid, _ = db.upsert_user(conn, "iss", "s", None)
        tid = db.default_tenant(conn, uid, "t")
        with pytest.raises(ValueError):
            db.add_member(conn, uid, tid, "superuser")
        db.audit(conn, "login", tenant_id=tid, user_id=uid, detail="x" * 1000)
        (detail,) = conn.execute("SELECT detail FROM audit_events").fetchone()
    assert len(detail) == 200
