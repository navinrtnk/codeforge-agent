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
    assert settings.index_chunk_size_lines == 200
    assert settings.max_tool_output_bytes == 100_000
    assert settings.agent_max_iterations == 8
    assert settings.agent_max_output_tokens == 4096
    assert settings.agent_timeout_seconds == 120.0
    assert settings.conversation_history_token_budget == 8192
    assert settings.conversation_summary_max_tokens == 512
    assert settings.conversation_summary_input_token_budget == 8192
    assert settings.git_allow_mutations is False
    assert settings.git_timeout_seconds == 15.0
    assert settings.git_max_output_bytes == 100_000
    assert settings.git_context_max_bytes == 16_000
    assert settings.git_context_max_files == 50
    assert settings.git_context_timeout_seconds == 5.0
    assert settings.code_writes_enabled is False
    assert settings.patch_max_bytes == 100_000
    assert settings.patch_max_files == 10
    assert settings.model_provider == "openai"


def test_settings_load_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEFORGE_DATABASE_URL", "sqlite:///custom.db")
    monkeypatch.setenv("CODEFORGE_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("CODEFORGE_OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("CODEFORGE_REPOSITORY_IGNORE_PATTERNS", '["vendor", "*.log"]')
    monkeypatch.setenv("CODEFORGE_MAX_FILE_SIZE_BYTES", "2048")
    monkeypatch.setenv("CODEFORGE_INDEX_CHUNK_SIZE_LINES", "50")
    monkeypatch.setenv("CODEFORGE_MAX_TOOL_OUTPUT_BYTES", "4096")
    monkeypatch.setenv("CODEFORGE_AGENT_MAX_ITERATIONS", "3")
    monkeypatch.setenv("CODEFORGE_AGENT_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("CODEFORGE_AGENT_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CODEFORGE_CONVERSATION_HISTORY_TOKEN_BUDGET", "1024")
    monkeypatch.setenv("CODEFORGE_CONVERSATION_SUMMARY_MAX_TOKENS", "128")
    monkeypatch.setenv("CODEFORGE_CONVERSATION_SUMMARY_INPUT_TOKEN_BUDGET", "2048")
    monkeypatch.setenv("CODEFORGE_GIT_ALLOW_MUTATIONS", "true")
    monkeypatch.setenv("CODEFORGE_GIT_TIMEOUT_SECONDS", "10")
    monkeypatch.setenv("CODEFORGE_GIT_MAX_OUTPUT_BYTES", "4096")
    monkeypatch.setenv("CODEFORGE_GIT_CONTEXT_MAX_BYTES", "2048")
    monkeypatch.setenv("CODEFORGE_GIT_CONTEXT_MAX_FILES", "10")
    monkeypatch.setenv("CODEFORGE_GIT_CONTEXT_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("CODEFORGE_CODE_WRITES_ENABLED", "true")
    monkeypatch.setenv("CODEFORGE_PATCH_MAX_BYTES", "4096")
    monkeypatch.setenv("CODEFORGE_PATCH_MAX_FILES", "3")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.database_url == "sqlite:///custom.db"
    assert settings.model_provider == "anthropic"
    assert settings.repository_ignore_patterns == ("vendor", "*.log")
    assert settings.max_file_size_bytes == 2048
    assert settings.index_chunk_size_lines == 50
    assert settings.max_tool_output_bytes == 4096
    assert settings.agent_max_iterations == 3
    assert settings.agent_max_output_tokens == 1024
    assert settings.agent_timeout_seconds == 30.0
    assert settings.conversation_history_token_budget == 1024
    assert settings.conversation_summary_max_tokens == 128
    assert settings.conversation_summary_input_token_budget == 2048
    assert settings.git_allow_mutations is True
    assert settings.git_timeout_seconds == 10.0
    assert settings.git_max_output_bytes == 4096
    assert settings.git_context_max_bytes == 2048
    assert settings.git_context_max_files == 10
    assert settings.git_context_timeout_seconds == 2.0
    assert settings.code_writes_enabled is True
    assert settings.patch_max_bytes == 4096
    assert settings.patch_max_files == 3
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_settings_reject_unknown_environment() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[call-arg]
            environment="staging",  # type: ignore[arg-type]
            _env_file=None,
        )
