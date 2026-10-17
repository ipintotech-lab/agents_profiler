"""Shared setup for the SaaS tests: a fresh database and storage per
test, in the given AUTH_MODE."""
from saas.config import get_settings


def configure(monkeypatch, tmp_path, mode="dev", **env):
    values = {
        "AUTH_MODE": mode,
        "APP_ORIGIN": "http://testserver",
        "SAAS_DB": str(tmp_path / "saas.sqlite"),
        "STORAGE_DIR": str(tmp_path / "storage"),
        "BEHAVIOR_DB": str(tmp_path / "behavior.sqlite"),
        **env,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()
    return get_settings()
