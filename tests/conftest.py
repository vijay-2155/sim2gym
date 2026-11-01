import pytest


@pytest.fixture(autouse=True)
def _hermetic_demo_env(monkeypatch):
    """Local .env demo settings must not change test expectations."""
    for k in ("SIM2GYM_DEFAULT_MODEL", "SIM2GYM_ALLOW_PAID"):
        monkeypatch.delenv(k, raising=False)
