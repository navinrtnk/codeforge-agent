"""Provider-neutral embeddings, deterministic fakes, and vector validation."""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import openai

from agent.config import Settings


class EmbeddingError(Exception):
    """A configuration, provider, or malformed-vector failure safe to report."""


class EmbeddingIndexError(EmbeddingError):
    """The repository needs reindexing with the active embedding model."""


class EmbeddingClient(Protocol):
    """Embed ordered batches in a versioned vector space."""

    model_key: str
    dimensions: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_vector(vector: list[float], dimensions: int) -> list[float]:
    """Validate finite, nonzero vectors and normalize without overflowing their norm."""
    if (
        not isinstance(vector, list)
        or len(vector) != dimensions
        or dimensions < 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        )
    ):
        raise EmbeddingError("Embedding vector has invalid dimensions or values")
    scale = max(abs(value) for value in vector)
    if not scale:
        raise EmbeddingError("Embedding vector must not be zero")
    scaled = [value / scale for value in vector]
    length = math.sqrt(math.fsum(value * value for value in scaled))
    return [value / length for value in scaled]


async def embed_texts(client: EmbeddingClient, texts: list[str]) -> list[list[float]]:
    """Validate a complete ordered batch before any caller stores its vectors."""
    vectors = await client.embed(texts)
    if len(vectors) != len(texts):
        raise EmbeddingError("Embedding response has the wrong number of vectors")
    return [normalize_vector(vector, client.dimensions) for vector in vectors]


@dataclass
class FakeEmbeddingClient:
    """Queued vectors allow retrieval tests to express semantic relationships explicitly."""

    responses: list[list[list[float]]]
    dimensions: int = 2
    model_key: str = "fake-v1"
    requests: list[list[str]] = field(default_factory=list)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(list(texts))
        if not self.responses:
            raise EmbeddingError("Fake embedding client has no queued response")
        return self.responses.pop(0)


class OpenAIEmbeddingClient:
    """Use the embeddings API with bounded input and explicit float output."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        *,
        max_input_bytes: int = 8000,
        timeout_seconds: float = 30,
        client: Any | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.model = model
        self.max_input_bytes = max_input_bytes
        self.model_key = f"openai:{model}:{dimensions}:utf8-{max_input_bytes}:v1"
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        inputs = [
            text.encode("utf-8")[: self.max_input_bytes].decode("utf-8", errors="ignore").strip()
            or "(whitespace)"
            for text in texts
        ]
        try:
            response = await self._client.embeddings.create(
                input=inputs,
                model=self.model,
                dimensions=self.dimensions,
                encoding_format="float",
            )
        except Exception as error:
            raise EmbeddingError("Embedding provider request failed") from error
        try:
            indices = [item.index for item in response.data]
            if any(type(index) is not int for index in indices) or sorted(indices) != list(
                range(len(texts))
            ):
                raise EmbeddingError("Embedding response contains invalid input indices")
            return [
                list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)
            ]
        except (AttributeError, TypeError, ValueError) as error:
            raise EmbeddingError("Embedding provider returned a malformed response") from error


def create_embedding_client(settings: Settings) -> EmbeddingClient:
    """Embeddings use OpenAI independently of the conversational model provider."""
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        raise EmbeddingError("CODEFORGE_OPENAI_API_KEY is required for embeddings")
    return OpenAIEmbeddingClient(
        settings.openai_api_key.get_secret_value(),
        settings.embedding_model,
        settings.embedding_dimensions,
        max_input_bytes=settings.embedding_max_input_bytes,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
