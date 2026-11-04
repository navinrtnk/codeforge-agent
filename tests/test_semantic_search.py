"""Semantic retrieval, incremental embeddings, and hybrid ranking integration."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, text

from agent.config import Settings
from agent.database import Database
from agent.embedding_index import index_with_embeddings
from agent.embeddings import EmbeddingError, FakeEmbeddingClient
from agent.indexing import RepositoryIndexer
from agent.main import create_app
from agent.models import ChunkEmbedding, CodeChunk, Repository
from agent.models_api import FakeModelClient, ModelResponse, TextContent, TokenUsage, ToolCall
from agent.repository_access import RepositoryAccess


def application(tmp_path: Path, embeddings: FakeEmbeddingClient, **overrides: Any) -> FastAPI:
    settings = Settings(  # type: ignore[call-arg]
        database_url="sqlite://",
        workspace_root=tmp_path,
        _env_file=None,
        **{"embeddings_enabled": True, "embedding_batch_size": 2, **overrides},
    )
    return create_app(settings, embedding_client=embeddings)


def register(client: TestClient, path: Path) -> str:
    created = client.post("/repositories", json={"name": "Semantic", "path": str(path)})
    assert created.status_code == 201
    return f"/repositories/{created.json()['id']}"


def repository(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    (path / "storage.py").write_text("persist_record()\n", encoding="utf-8")
    return path


def test_semantic_retrieval_finds_code_without_query_keywords_and_is_scoped(tmp_path: Path) -> None:
    path = repository(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "private.py").write_text("private_record()\n", encoding="utf-8")
    embeddings = FakeEmbeddingClient([[[1, 0]], [[1, 0]], [[1, 0]]])
    with TestClient(application(tmp_path, embeddings)) as client:
        url = register(client, path)
        other_url = register(client, other)
        assert client.post(url + "/index").status_code == 200
        assert client.post(other_url + "/index").status_code == 200
        assert client.get(url + "/search", params={"q": "save data"}).json() == []
        result = client.get(url + "/search", params={"q": "save data", "mode": "semantic"})
    assert result.status_code == 200
    assert [item["path"] for item in result.json()] == ["storage.py"]
    assert result.json()[0]["score"] == pytest.approx(1)
    assert result.json()[0]["snippet"] == "persist_record()\n"
    assert embeddings.requests[-1] == ["save data"]


def test_unchanged_and_shifted_chunks_reuse_vectors_in_batches(tmp_path: Path) -> None:
    path = repository(tmp_path)
    source = path / "storage.py"
    source.write_text("first()\nsecond()\nthird()\n", encoding="utf-8")
    embeddings = FakeEmbeddingClient([[[1, 0], [0, 1]], [[1, 1]], [[-1, 0]]])
    with TestClient(application(tmp_path, embeddings, index_chunk_size_lines=1)) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        assert embeddings.requests == [["first()\n", "second()\n"], ["third()\n"]]
        assert client.post(url + "/index").json()["skipped"] == 1
        assert len(embeddings.requests) == 2
        source.write_text("inserted()\nfirst()\nsecond()\nthird()\n", encoding="utf-8")
        assert client.post(url + "/index").json()["updated"] == 1
        assert embeddings.requests[-1] == ["inserted()\n"]
        assert len(embeddings.requests) == 3


def test_file_rename_reuses_embeddings_and_updates_result_locations(tmp_path: Path) -> None:
    path = repository(tmp_path)
    embeddings = FakeEmbeddingClient([[[1, 0]], [[1, 0]]])
    with TestClient(application(tmp_path, embeddings)) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        (path / "storage.py").rename(path / "renamed.py")
        result = client.post(url + "/index")
        assert result.json()["indexed"] == 1 and result.json()["deleted"] == 1
        assert len(embeddings.requests) == 1
        found = client.get(url + "/search", params={"q": "save", "mode": "semantic"})
        assert [item["path"] for item in found.json()] == ["renamed.py"]


def test_identical_chunks_are_embedded_once_and_deleted_vectors_cascade(tmp_path: Path) -> None:
    path = repository(tmp_path)
    (path / "copy.py").write_text("persist_record()\n", encoding="utf-8")
    embeddings = FakeEmbeddingClient([[[1, 0]]])
    app = application(tmp_path, embeddings)
    with TestClient(app) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        database: Database = app.state.database
        with database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(ChunkEmbedding)) == 2
        for file in path.iterdir():
            file.unlink()
        assert client.post(url + "/index").json()["deleted"] == 2
        with database.session_factory() as session:
            assert session.scalar(select(func.count()).select_from(ChunkEmbedding)) == 0
        assert client.get(url + "/search", params={"q": "save", "mode": "semantic"}).json() == []
    assert embeddings.requests == [["persist_record()\n"]]


def test_enabling_embeddings_backfills_existing_index_and_supports_schema_upgrade(
    tmp_path: Path,
) -> None:
    path = repository(tmp_path)
    embeddings = FakeEmbeddingClient([[[1, 0]]])
    app = application(tmp_path, embeddings, embeddings_enabled=False)
    with TestClient(app) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        assert embeddings.requests == []
        assert (
            client.get(url + "/search", params={"q": "save", "mode": "semantic"}).status_code == 503
        )
        database: Database = app.state.database
        with database.engine.begin() as connection:
            connection.execute(text("DROP TABLE chunk_embeddings"))
        database.create_schema()
        database.create_schema()
        app.state.settings.embeddings_enabled = True
        assert client.post(url + "/index").json()["skipped"] == 1
        assert embeddings.requests == [["persist_record()\n"]]


def test_model_switch_requires_reindexing_and_cannot_mix_vector_spaces(tmp_path: Path) -> None:
    path = repository(tmp_path)
    first = FakeEmbeddingClient([[[1, 0]]])
    app = application(tmp_path, first)
    with TestClient(app) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        second = FakeEmbeddingClient([[[0, 1, 0]], [[0, 1, 0]]], dimensions=3, model_key="fake-v2")
        app.state.embedding_client = second
        assert (
            client.get(url + "/search", params={"q": "save", "mode": "hybrid"}).status_code == 409
        )
        assert second.requests == []
        assert client.post(url + "/index").status_code == 200
        result = client.get(url + "/search", params={"q": "save", "mode": "semantic"})
        assert result.status_code == 200 and result.json()[0]["score"] == pytest.approx(1)
        database: Database = app.state.database
        with database.session_factory() as session:
            embedding = session.scalar(select(ChunkEmbedding))
            assert embedding is not None and embedding.model_key == "fake-v2"


@pytest.mark.parametrize("invalid", [[], [[0, 0]], [[float("nan"), 0]], [[1]]])
def test_failed_embedding_batch_rolls_back_chunks_and_lexical_index(
    tmp_path: Path, invalid: list[list[float]]
) -> None:
    path = repository(tmp_path)
    embeddings = FakeEmbeddingClient([[[1, 0]], [[0, 1]], invalid])
    app = application(tmp_path, embeddings, embedding_batch_size=1)
    with TestClient(app) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        (path / "storage.py").write_text("changed_record()\n", encoding="utf-8")
        (path / "other.py").write_text("new_record()\n", encoding="utf-8")
        assert client.post(url + "/index").status_code == 502
        assert client.get(url + "/search", params={"q": "persist_record"}).json()
        assert client.get(url + "/search", params={"q": "changed_record"}).json() == []
        assert client.get(url + "/index/status").json()["file_count"] == 1
        database: Database = app.state.database
        with database.session_factory() as session:
            embedding = session.scalar(select(ChunkEmbedding))
            assert embedding is not None and embedding.vector == [1.0, 0.0]


@pytest.mark.parametrize(
    ("lexical", "semantic", "expected"),
    [
        (0.99, 0.01, "lexical.py"),
        (0.01, 0.99, "storage.py"),
        (0, 1, "storage.py"),
    ],
)
def test_hybrid_weights_change_ranking_without_duplicate_chunks(
    tmp_path: Path, lexical: float, semantic: float, expected: str
) -> None:
    path = repository(tmp_path)
    (path / "lexical.py").write_text("# transactions\n", encoding="utf-8")
    embeddings = FakeEmbeddingClient([[[0, 1], [1, 0]], [[1, 0]]])
    with TestClient(
        application(
            tmp_path, embeddings, search_lexical_weight=lexical, search_semantic_weight=semantic
        )
    ) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        result = client.get(url + "/search", params={"q": "transactions", "mode": "hybrid"})
    assert result.status_code == 200
    assert result.json()[0]["path"] == expected
    assert len(result.json()) == 2
    assert len({item["path"] for item in result.json()}) == 2


def test_lexical_only_hybrid_weight_needs_no_embeddings_or_credentials(tmp_path: Path) -> None:
    path = repository(tmp_path)
    embeddings = FakeEmbeddingClient([])
    with TestClient(
        application(
            tmp_path,
            embeddings,
            embeddings_enabled=False,
            search_lexical_weight=1,
            search_semantic_weight=0,
        )
    ) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        result = client.get(url + "/search", params={"q": "persist_record", "mode": "hybrid"})
    assert result.status_code == 200 and result.json()[0]["path"] == "storage.py"
    assert embeddings.requests == []


def test_ties_have_stable_path_order_and_results_respect_limit(tmp_path: Path) -> None:
    path = repository(tmp_path)
    (path / "aaa.py").write_text("another_record()\n", encoding="utf-8")
    embeddings = FakeEmbeddingClient([[[1, 0], [1, 0]], [[2, 0]]])
    with TestClient(application(tmp_path, embeddings)) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        result = client.get(url + "/search", params={"q": "save", "mode": "semantic", "limit": 1})
    assert [item["path"] for item in result.json()] == ["aaa.py"]


def test_missing_and_corrupt_vectors_require_reindex_and_are_repaired(tmp_path: Path) -> None:
    path = repository(tmp_path)
    embeddings = FakeEmbeddingClient([[[1, 0]], [[1, 0]], [[1, 0]]])
    app = application(tmp_path, embeddings)
    with TestClient(app) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        database: Database = app.state.database
        with database.session_factory() as session:
            embedding = session.scalar(select(ChunkEmbedding))
            assert embedding is not None
            embedding.vector = [0, 0]
            session.commit()
        assert (
            client.get(url + "/search", params={"q": "save", "mode": "semantic"}).status_code == 409
        )
        assert client.post(url + "/index").status_code == 200
        with database.session_factory() as session:
            embedding = session.scalar(select(ChunkEmbedding))
            assert embedding is not None and embedding.vector == [1, 0]
            session.delete(embedding)
            session.commit()
        assert (
            client.get(url + "/search", params={"q": "save", "mode": "semantic"}).status_code == 409
        )


def test_agent_search_tool_can_use_semantic_retrieval(tmp_path: Path) -> None:
    path = repository(tmp_path)
    embeddings = FakeEmbeddingClient([[[1, 0]], [[1, 0]]])
    app = application(tmp_path, embeddings)
    app.state.model_client = FakeModelClient(
        [
            ModelResponse(
                "search",
                "fake",
                "test",
                (
                    ToolCall(
                        "search",
                        "search_code",
                        {
                            "query": "save data",
                            "mode": "semantic",
                        },
                    ),
                ),
                "tool_use",
                TokenUsage(1, 1),
            ),
            ModelResponse(
                "answer",
                "fake",
                "test",
                (TextContent("Found storage.py."),),
                "end_turn",
                TokenUsage(1, 1),
            ),
        ]
    )
    with TestClient(app) as client:
        url = register(client, path)
        assert client.post(url + "/index").status_code == 200
        result = client.post(url + "/runs", json={"task": "Find where data is saved"})
    assert result.status_code == 201 and result.json()["status"] == "completed"
    event = result.json()["tool_events"][0]
    assert event["is_error"] is False
    assert event["result"]["results"][0]["path"] == "storage.py"


def test_cancelled_embedding_refresh_rolls_back_index_changes(tmp_path: Path) -> None:
    path = repository(tmp_path)
    database = Database("sqlite://")
    database.create_schema()
    indexer = RepositoryIndexer(RepositoryAccess(tmp_path))

    class BlockingClient(FakeEmbeddingClient):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            entered.set()
            await asyncio.Event().wait()
            raise EmbeddingError("Unreachable")

    async def exercise() -> None:
        with database.session_factory() as session:
            repo = Repository(name="Test", path=str(path))
            session.add(repo)
            session.commit()
            indexer.index(session, repo)
            (path / "storage.py").write_text("changed_record()\n", encoding="utf-8")
            task = asyncio.create_task(
                index_with_embeddings(session, repo, indexer, BlockingClient([]))
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            chunk = session.scalar(select(CodeChunk))
            assert chunk is not None and chunk.content == "persist_record()\n"

    entered = asyncio.Event()
    asyncio.run(exercise())
    database.dispose()


def test_overlapping_index_requests_reuse_the_first_completed_batch(tmp_path: Path) -> None:
    path = repository(tmp_path)

    async def exercise() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        class GatedClient(FakeEmbeddingClient):
            async def embed(self, texts: list[str]) -> list[list[float]]:
                entered.set()
                await release.wait()
                return await super().embed(texts)

        embeddings = GatedClient([[[1, 0]]])
        app = application(tmp_path, embeddings)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
        ):
            created = await client.post("/repositories", json={"name": "Test", "path": str(path)})
            url = f"/repositories/{created.json()['id']}/index"
            first = asyncio.create_task(client.post(url))
            await asyncio.wait_for(entered.wait(), timeout=2)
            second = asyncio.create_task(client.post(url))
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
            assert [result.status_code for result in results] == [200, 200]
            assert results[1].json()["skipped"] == 1
            assert embeddings.requests == [["persist_record()\n"]]

    asyncio.run(exercise())
