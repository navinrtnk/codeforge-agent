"""Tests for application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.config import Settings


def test_settings_have_development_defaults() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.environment == "development"
    assert settings.database_url == "sqlite:///./codeforge.db"
    assert settings.workspace_root == Path(".")
    assert settings.model_provider == "openai"


def test_settings_load_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEFORGE_DATABASE_URL", "sqlite:///custom.db")
    monkeypatch.setenv("CODEFORGE_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("CODEFORGE_OPENAI_API_KEY", "secret-value")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.database_url == "sqlite:///custom.db"
    assert settings.model_provider == "anthropic"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_settings_reject_unknown_environment() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[call-arg]
            environment="staging",  # type: ignore[arg-type]
            _env_file=None,
        )
