"""Embedding contracts and the OpenAI adapter without live provider requests."""

import asyncio
import math
from types import SimpleNamespace
from typing import Any

import pytest

from agent.config import Settings
from agent.embeddings import (
    EmbeddingError,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
    create_embedding_client,
    embed_texts,
    normalize_vector,
)


@pytest.mark.parametrize(
    "vector", [[], [1.0], [0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0], [True, 1.0], ["1", 1.0]]
)
def test_invalid_vectors_are_rejected(vector: Any) -> None:
    with pytest.raises(EmbeddingError):
        normalize_vector(vector, 2)


def test_normalization_handles_large_and_tiny_magnitudes() -> None:
    assert normalize_vector([3.0, 4.0], 2) == pytest.approx([0.6, 0.8])
    assert normalize_vector([1e308, 1e308], 2) == pytest.approx([2**-0.5, 2**-0.5])
    assert normalize_vector([1e-300, 0.0], 2) == [1.0, 0.0]


def test_fake_records_ordered_batches_and_count_is_validated() -> None:
    client = FakeEmbeddingClient([[[1.0, 0.0]]])
    with pytest.raises(EmbeddingError, match="number"):
        asyncio.run(embed_texts(client, ["one", "two"]))
    assert client.requests == [["one", "two"]]


class EmbeddingsSDK:
    def __init__(self, indices: list[int] | None = None, *, fail: bool = False) -> None:
        self.indices = indices if indices is not None else [1, 0]
        self.fail = fail
        self.requests: list[dict[str, Any]] = []

    async def create(self, **arguments: Any) -> SimpleNamespace:
        self.requests.append(arguments)
        if self.fail:
            raise RuntimeError("password=private-value")
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index == 0), float(index != 0)])
                for index in self.indices
            ]
        )


def test_adapter_bounds_utf8_inputs_and_restores_response_order() -> None:
    sdk = EmbeddingsSDK()
    client = OpenAIEmbeddingClient(
        "test", "test-model", 2, max_input_bytes=4, client=SimpleNamespace(embeddings=sdk)
    )
    vectors = asyncio.run(embed_texts(client, ["日本語", "  "]))
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert sdk.requests == [
        {
            "input": ["日", "(whitespace)"],
            "model": "test-model",
            "dimensions": 2,
            "encoding_format": "float",
        }
    ]
    assert asyncio.run(client.embed([])) == [] and len(sdk.requests) == 1


@pytest.mark.parametrize("indices", [[0, 0], [0], [0, 2], [-1, 0]])
def test_adapter_rejects_incomplete_or_misindexed_responses(indices: list[int]) -> None:
    client = OpenAIEmbeddingClient(
        "test", "test", 2, client=SimpleNamespace(embeddings=EmbeddingsSDK(indices))
    )
    with pytest.raises(EmbeddingError, match="indices"):
        asyncio.run(client.embed(["a", "b"]))


def test_adapter_provider_errors_do_not_leak_credentials() -> None:
    client = OpenAIEmbeddingClient(
        "test", "test", 2, client=SimpleNamespace(embeddings=EmbeddingsSDK(fail=True))
    )
    with pytest.raises(EmbeddingError) as error:
        asyncio.run(client.embed(["a"]))
    assert "private-value" not in str(error.value)


def test_malformed_provider_vectors_are_reported_as_embedding_errors() -> None:
    class MalformedSDK:
        async def create(self, **arguments: Any) -> SimpleNamespace:
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=None)])

    client = OpenAIEmbeddingClient(
        "test", "test", 2, client=SimpleNamespace(embeddings=MalformedSDK())
    )
    with pytest.raises(EmbeddingError, match="malformed"):
        asyncio.run(client.embed(["text"]))


def test_vector_space_identity_includes_model_dimensions_and_preparation() -> None:
    clients = [
        OpenAIEmbeddingClient("test", "model-a", 2),
        OpenAIEmbeddingClient("test", "model-b", 2),
        OpenAIEmbeddingClient("test", "model-a", 3),
        OpenAIEmbeddingClient("test", "model-a", 2, max_input_bytes=100),
    ]
    assert len({client.model_key for client in clients}) == 4


def test_embedding_factory_requires_independent_openai_credentials() -> None:
    settings = Settings(model_provider="anthropic", _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        create_embedding_client(settings)


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_dimensions": 0},
        {"embedding_batch_size": 0},
        {"embedding_batch_size": 33},
        {"embedding_max_input_bytes": 8001},
        {"embedding_timeout_seconds": 0},
        {"search_lexical_weight": 0, "search_semantic_weight": 0},
        {"search_semantic_weight": -1},
        {"search_semantic_weight": math.nan},
    ],
)
def test_embedding_configuration_limits(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)  # type: ignore[call-arg]
