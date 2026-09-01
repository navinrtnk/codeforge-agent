"""Model client construction from application settings."""

from agent.anthropic_client import AnthropicModelClient
from agent.config import Settings
from agent.models_api import ModelClient, ModelConfigurationError
from agent.openai_client import OpenAIModelClient


def create_model_client(settings: Settings) -> ModelClient:
    """Create the configured provider client after validating credentials."""
    if not settings.model_name:
        raise ModelConfigurationError("CODEFORGE_MODEL_NAME is required")

    if settings.model_provider == "openai":
        if settings.openai_api_key is None:
            raise ModelConfigurationError("CODEFORGE_OPENAI_API_KEY is required")
        return OpenAIModelClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.model_name,
        )

    if settings.anthropic_api_key is None:
        raise ModelConfigurationError("CODEFORGE_ANTHROPIC_API_KEY is required")
    return AnthropicModelClient(
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=settings.model_name,
    )
