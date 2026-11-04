"""Exact cosine retrieval and weighted reciprocal-rank fusion with lexical search."""

import heapq
import math
import uuid
from collections.abc import Iterator
from dataclasses import replace
from typing import Literal

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from agent.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    EmbeddingIndexError,
    content_hash,
    embed_texts,
    normalize_vector,
)
from agent.models import ChunkEmbedding, CodeChunk, IndexedFile
from agent.search import InvalidSearchQueryError, SearchResult, lexical_search


def semantic_search(
    session: Session,
    repository_id: uuid.UUID,
    vector: list[float],
    client: EmbeddingClient,
    *,
    limit: int = 20,
) -> list[SearchResult]:
    """Stream repository vectors and retain only the best cosine matches."""
    query = normalize_vector(vector, client.dimensions)
    rows = session.execute(
        select(CodeChunk, IndexedFile, ChunkEmbedding)
        .join(IndexedFile, CodeChunk.indexed_file_id == IndexedFile.id)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == CodeChunk.id)
        .where(
            IndexedFile.repository_id == repository_id, ChunkEmbedding.model_key == client.model_key
        )
        .execution_options(yield_per=200)
    )

    def candidates() -> Iterator[SearchResult]:
        for chunk, file, embedding in rows:
            if embedding.content_hash != content_hash(chunk.content):
                raise EmbeddingIndexError("Embedding index is stale; reindex the repository")
            try:
                stored = normalize_vector(embedding.vector, client.dimensions)
            except EmbeddingError as error:
                raise EmbeddingIndexError(
                    "Invalid stored embedding; reindex the repository"
                ) from error
            score = max(
                -1.0, min(1.0, math.fsum(a * b for a, b in zip(query, stored, strict=True)))
            )
            yield SearchResult(
                path=file.path,
                language=file.language,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                snippet=chunk.content,
                score=score,
            )

    return heapq.nsmallest(
        limit, candidates(), key=lambda item: (-item.score, item.path, item.start_line)
    )


async def search_with_embeddings(
    session: Session,
    repository_id: uuid.UUID,
    query: str,
    client: EmbeddingClient,
    *,
    mode: Literal["semantic", "hybrid"] = "hybrid",
    limit: int = 20,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> list[SearchResult]:
    """Fuse ranks so BM25 and cosine scores need not share the same numeric scale."""
    query = query.strip()
    if not query:
        raise InvalidSearchQueryError("Search query must not be blank")
    if (
        not math.isfinite(lexical_weight)
        or not math.isfinite(semantic_weight)
        or min(lexical_weight, semantic_weight) < 0
        or lexical_weight + semantic_weight <= 0
    ):
        raise ValueError("Search weights must be finite, nonnegative, and not both zero")
    if mode == "hybrid" and semantic_weight == 0:
        return lexical_search(session, repository_id, query, limit=limit)
    chunks, embeddings = session.execute(
        select(func.count(CodeChunk.id), func.count(ChunkEmbedding.chunk_id))
        .select_from(CodeChunk)
        .join(IndexedFile)
        .outerjoin(
            ChunkEmbedding,
            and_(
                ChunkEmbedding.chunk_id == CodeChunk.id,
                ChunkEmbedding.model_key == client.model_key,
            ),
        )
        .where(IndexedFile.repository_id == repository_id)
    ).one()
    if chunks != embeddings:
        raise EmbeddingIndexError("Embedding model is missing or changed; reindex the repository")
    if not chunks:
        return []
    vector = (await embed_texts(client, [query]))[0]
    candidate_limit = max(100, limit * 4) if mode == "hybrid" else limit
    semantic = semantic_search(session, repository_id, vector, client, limit=candidate_limit)
    if mode == "semantic":
        return semantic
    lexical = (
        lexical_search(session, repository_id, query, limit=candidate_limit)
        if lexical_weight
        else []
    )
    combined: dict[tuple[str, int, int], SearchResult] = {}
    total_weight = lexical_weight + semantic_weight
    for results, weight in ((lexical, lexical_weight), (semantic, semantic_weight)):
        for rank, result in enumerate(results, 1):
            key = (result.path, result.start_line, result.end_line)
            score = weight / total_weight / (60 + rank)
            existing = combined.get(key)
            combined[key] = replace(
                existing or result, score=score + (existing.score if existing else 0)
            )
    return sorted(combined.values(), key=lambda item: (-item.score, item.path, item.start_line))[
        :limit
    ]
