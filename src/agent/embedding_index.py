"""Atomic index refresh with batched embedding reuse by chunk content."""

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from agent.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    content_hash,
    embed_texts,
    normalize_vector,
)
from agent.indexing import IndexingResult, RepositoryIndexer
from agent.models import ChunkEmbedding, CodeChunk, IndexedFile, Repository


async def index_with_embeddings(
    session: Session,
    repository: Repository,
    indexer: RepositoryIndexer,
    client: EmbeddingClient,
    *,
    batch_size: int = 32,
) -> IndexingResult:
    """Preserve unchanged vectors even when a changed file is rechunked or moved."""
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    try:
        stored = session.execute(
            select(ChunkEmbedding, CodeChunk.content)
            .join(CodeChunk)
            .join(IndexedFile)
            .where(
                IndexedFile.repository_id == repository.id,
                ChunkEmbedding.model_key == client.model_key,
            )
        )
        cache: dict[str, list[float]] = {}
        for item, text in stored:
            if item.content_hash != content_hash(text):
                continue
            try:
                cache[item.content_hash] = normalize_vector(item.vector, client.dimensions)
            except EmbeddingError:
                # A successful reindex must also repair malformed persisted evidence.
                continue
        result = indexer.index(session, repository, commit=False)
        chunks = list(
            session.scalars(
                select(CodeChunk)
                .join(IndexedFile)
                .where(IndexedFile.repository_id == repository.id)
                .options(selectinload(CodeChunk.embedding))
                .order_by(IndexedFile.path, CodeChunk.chunk_index)
            )
        )
        chunk_keys = [(chunk, content_hash(chunk.content)) for chunk in chunks]
        missing = {key: chunk.content for chunk, key in chunk_keys if key not in cache}
        items = list(missing.items())
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            vectors = await embed_texts(client, [text for _, text in batch])
            cache.update((key, vector) for (key, _), vector in zip(batch, vectors, strict=True))
        for chunk, key in chunk_keys:
            if chunk.embedding is None:
                chunk.embedding = ChunkEmbedding(
                    model_key=client.model_key,
                    content_hash=key,
                    vector=cache[key],
                )
            else:
                chunk.embedding.model_key = client.model_key
                chunk.embedding.content_hash = key
                chunk.embedding.vector = cache[key]
        session.commit()
        return result
    except BaseException:
        session.rollback()
        raise
