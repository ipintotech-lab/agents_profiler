import pytest

from saas.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings():
    # settings are cached per process; tests change AUTH_MODE and paths
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
