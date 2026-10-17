"""SaaS settings, read from the environment.

``AUTH_MODE`` picks the mode:

- ``demo`` (default): the public demo as before; the ``/api`` routes are
  off and nothing customer-owned is stored. Not for customer data.
- ``oidc``: production. Customer-data routes need a session from the
  OIDC provider (``OIDC_ISSUER``, ``OIDC_CLIENT_ID``, ``OIDC_CLIENT_SECRET``).
- ``dev``: local testing. ``/auth/dev-login`` signs in without a provider;
  refused unless ``APP_ORIGIN`` is a localhost address.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from urllib.parse import urlsplit

AUTH_MODES = ("demo", "dev", "oidc")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]", "testserver")

MB = 1024 * 1024


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    auth_mode: str
    app_origin: str
    db_path: str
    storage_dir: str
    session_ttl_s: int
    # upload limits (backend-enforced; the page's checks are only UX)
    max_file_bytes: int
    max_files_per_upload: int
    tenant_storage_bytes: int
    daily_upload_bytes: int
    max_active_jobs: int
    # worker sandbox
    job_timeout_s: int
    job_memory_bytes: int
    # OIDC
    oidc_issuer: str
    oidc_client_id: str
    oidc_client_secret: str
    oidc_scopes: str
    # malware scanning: "off" or "clamd"
    malware_scanner: str
    clamd_host: str
    clamd_port: int
    # OIDC subjects ("issuer|sub") or emails allowed to run the QA tools
    platform_admins: frozenset[str]

    @property
    def saas(self) -> bool:
        return self.auth_mode != "demo"

    @property
    def secure_cookies(self) -> bool:
        return self.app_origin.startswith("https://")


def validate(s: Settings) -> None:
    if s.auth_mode not in AUTH_MODES:
        raise RuntimeError(f"AUTH_MODE must be one of {', '.join(AUTH_MODES)}, not {s.auth_mode!r}")
    if s.auth_mode == "dev" and urlsplit(s.app_origin).hostname not in {h.strip("[]") for h in LOCAL_HOSTS}:
        raise RuntimeError("AUTH_MODE=dev signs in without a password; it only runs with a localhost APP_ORIGIN")
    if s.auth_mode == "oidc":
        missing = [n for n, v in (("OIDC_ISSUER", s.oidc_issuer), ("OIDC_CLIENT_ID", s.oidc_client_id),
                                  ("OIDC_CLIENT_SECRET", s.oidc_client_secret), ("APP_ORIGIN", s.app_origin)) if not v]
        if missing:
            raise RuntimeError(f"AUTH_MODE=oidc needs {', '.join(missing)}")
        if not s.secure_cookies:
            raise RuntimeError("AUTH_MODE=oidc needs an https:// APP_ORIGIN")
    if s.malware_scanner not in ("off", "clamd"):
        raise RuntimeError("MALWARE_SCANNER must be off or clamd")


@cache
def get_settings() -> Settings:
    s = Settings(
        auth_mode=os.environ.get("AUTH_MODE", "demo"),
        app_origin=os.environ.get("APP_ORIGIN", "http://localhost:8000").rstrip("/"),
        db_path=os.environ.get("SAAS_DB", "data/saas.sqlite"),
        storage_dir=os.environ.get("STORAGE_DIR", "data/storage"),
        session_ttl_s=_int("SESSION_TTL_S", 8 * 3600),
        max_file_bytes=_int("MAX_UPLOAD_BYTES", 50 * MB),
        max_files_per_upload=_int("MAX_FILES_PER_UPLOAD", 10),
        tenant_storage_bytes=_int("TENANT_STORAGE_BYTES", 2048 * MB),
        daily_upload_bytes=_int("DAILY_UPLOAD_BYTES", 500 * MB),
        max_active_jobs=_int("MAX_ACTIVE_JOBS", 3),
        job_timeout_s=_int("JOB_TIMEOUT_S", 300),
        job_memory_bytes=_int("JOB_MEMORY_BYTES", 2048 * MB),
        oidc_issuer=os.environ.get("OIDC_ISSUER", "").rstrip("/"),
        oidc_client_id=os.environ.get("OIDC_CLIENT_ID", ""),
        oidc_client_secret=os.environ.get("OIDC_CLIENT_SECRET", ""),
        oidc_scopes=os.environ.get("OIDC_SCOPES", "openid email profile"),
        malware_scanner=os.environ.get("MALWARE_SCANNER", "off"),
        clamd_host=os.environ.get("CLAMD_HOST", "127.0.0.1"),
        clamd_port=_int("CLAMD_PORT", 3310),
        platform_admins=frozenset(a.strip() for a in os.environ.get("PLATFORM_ADMINS", "").split(",") if a.strip()),
    )
    validate(s)
    return s
