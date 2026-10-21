"""Application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CODEFORGE_",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite:///./codeforge.db"
    workspace_root: Path = Path(".")
    repository_ignore_patterns: tuple[str, ...] = (
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "*.pyc",
    )
    max_file_size_bytes: int = Field(default=1_000_000, gt=0)
    index_chunk_size_lines: int = Field(default=200, gt=0)
    max_tool_output_bytes: int = Field(default=100_000, ge=256)
    agent_max_iterations: int = Field(default=8, ge=1, le=100)
    agent_max_output_tokens: int = Field(default=4096, ge=1)
    agent_timeout_seconds: float = Field(default=120.0, gt=0)
    conversation_history_token_budget: int = Field(default=8192, ge=256)
    conversation_summary_max_tokens: int = Field(default=512, ge=1)
    conversation_summary_input_token_budget: int = Field(default=8192, ge=256)
    git_allow_mutations: bool = False
    git_timeout_seconds: float = Field(default=15.0, gt=0)
    git_max_output_bytes: int = Field(default=100_000, ge=256)
    git_context_max_bytes: int = Field(default=16_000, ge=512)
    git_context_max_files: int = Field(default=50, ge=1, le=1000)
    git_context_timeout_seconds: float = Field(default=5.0, gt=0)
    code_writes_enabled: bool = False
    patch_max_bytes: int = Field(default=100_000, ge=1)
    patch_max_files: int = Field(default=10, ge=1, le=100)
    test_execution_enabled: bool = False
    test_python_executable: Path | None = None
    test_timeout_seconds: float = Field(default=60.0, gt=0)
    test_max_output_bytes: int = Field(default=20_000, ge=256)
    model_provider: Literal["openai", "anthropic"] = "openai"
    model_name: str | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide application settings."""
    return Settings()
