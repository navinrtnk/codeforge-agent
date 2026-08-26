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
    assert ".git" in settings.repository_ignore_patterns
    assert settings.max_file_size_bytes == 1_000_000
    assert settings.model_provider == "openai"


def test_settings_load_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEFORGE_DATABASE_URL", "sqlite:///custom.db")
    monkeypatch.setenv("CODEFORGE_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("CODEFORGE_OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("CODEFORGE_REPOSITORY_IGNORE_PATTERNS", '["vendor", "*.log"]')
    monkeypatch.setenv("CODEFORGE_MAX_FILE_SIZE_BYTES", "2048")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.database_url == "sqlite:///custom.db"
    assert settings.model_provider == "anthropic"
    assert settings.repository_ignore_patterns == ("vendor", "*.log")
    assert settings.max_file_size_bytes == 2048
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_settings_reject_unknown_environment() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[call-arg]
            environment="staging",  # type: ignore[arg-type]
            _env_file=None,
        )
