"""Tests for model provider selection."""

import pytest

from agent.anthropic_client import AnthropicModelClient
from agent.config import Settings
from agent.model_factory import create_model_client
from agent.models_api import ModelConfigurationError
from agent.openai_client import OpenAIModelClient


def settings(**overrides: object) -> Settings:
    return Settings.model_validate(overrides)


def test_factory_creates_openai_client() -> None:
    client = create_model_client(
        settings(
            model_provider="openai",
            model_name="gpt-test",
            openai_api_key="openai-key",
        )
    )

    assert isinstance(client, OpenAIModelClient)
    assert client.model == "gpt-test"


def test_factory_creates_anthropic_client() -> None:
    client = create_model_client(
        settings(
            model_provider="anthropic",
            model_name="claude-test",
            anthropic_api_key="anthropic-key",
        )
    )

    assert isinstance(client, AnthropicModelClient)
    assert client.model == "claude-test"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_name": None}, "CODEFORGE_MODEL_NAME"),
        (
            {"model_provider": "openai", "model_name": "gpt-test"},
            "CODEFORGE_OPENAI_API_KEY",
        ),
        (
            {"model_provider": "anthropic", "model_name": "claude-test"},
            "CODEFORGE_ANTHROPIC_API_KEY",
        ),
    ],
)
def test_factory_rejects_missing_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ModelConfigurationError, match=message):
        create_model_client(settings(**overrides))
