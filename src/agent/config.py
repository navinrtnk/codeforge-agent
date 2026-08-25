"""Application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
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
    model_provider: Literal["openai", "anthropic"] = "openai"
    model_name: str | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide application settings."""
    return Settings()
