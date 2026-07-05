"""Tests for environment config helpers."""

from __future__ import annotations

import pytest

from app.core import config


@pytest.fixture(autouse=True)
def _clear_supabase_env(monkeypatch):
    for name in (
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_SERVICE_KEY",
        "SUPABASE_SECRET_KEY_DEV",
        "SUPABASE_SECRET_KEY_PROD",
        "SUPABASE_URL",
        "SUPABASE_URL_DEV",
        "SUPABASE_URL_PROD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_get_supabase_service_key_prefers_secret_key(monkeypatch):
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_new")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "legacy")
    assert config.get_supabase_service_key() == "sb_secret_new"


def test_get_supabase_service_key_falls_back_to_service_role(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "eyJlegacy")
    assert config.get_supabase_service_key() == "eyJlegacy"


def test_get_supabase_service_key_uses_env_specific_dev(monkeypatch):
    monkeypatch.setenv("SUPABASE_SECRET_KEY_DEV", "sb_secret_dev")
    monkeypatch.setenv("SUPABASE_SECRET_KEY_PROD", "sb_secret_prod")
    assert config.get_supabase_service_key(env="dev") == "sb_secret_dev"
    assert config.get_supabase_service_key(env="prod") == "sb_secret_prod"


def test_get_supabase_url_uses_env_specific_dev(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL_DEV", "https://dev.supabase.co")
    monkeypatch.setenv("SUPABASE_URL_PROD", "https://prod.supabase.co")
    assert config.get_supabase_url(env="dev") == "https://dev.supabase.co"
    assert config.get_supabase_url(env="prod") == "https://prod.supabase.co"


def test_supabase_env_configured(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL_DEV", "https://dev.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY_DEV", "sb_secret_dev")
    assert config.supabase_env_configured("dev") is True
    assert config.supabase_env_configured("prod") is False


def test_get_supabase_service_key_required_raises(monkeypatch):
    with pytest.raises(ValueError, match="SUPABASE_SECRET_KEY"):
        config.get_supabase_service_key(required=True)
