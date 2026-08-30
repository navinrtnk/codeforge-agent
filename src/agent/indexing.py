"""Deterministic, incremental repository indexing."""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from agent.models import CodeChunk, CodeSymbol, IndexedFile, Repository, utc_now
from agent.repository_access import RepositoryAccess, RepositoryAccessError
from agent.search import sync_repository_fts
from agent.symbols import ExtractedSymbol, extract_python_symbols

LANGUAGES_BY_EXTENSION = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".css": "css",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".md": "markdown",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "shell",
    ".sql": "sql",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}
LANGUAGES_BY_FILENAME = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}


@dataclass(frozen=True, slots=True)
class Chunk:
    """One line-based source chunk before persistence."""

    index: int
    start_line: int
    end_line: int
    content: str


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Outcome counts for one repository indexing run."""

    discovered: int
    indexed: int
    updated: int
    skipped: int
    deleted: int
    failed: int


@dataclass(frozen=True, slots=True)
class IndexStatus:
    """Current persisted index statistics for a repository."""

    file_count: int
    chunk_count: int
    last_indexed_at: datetime | None


def detect_language(path: str | Path) -> str:
    """Detect a source language from a filename or extension."""
    file_path = Path(path)
    filename_language = LANGUAGES_BY_FILENAME.get(file_path.name.lower())
    if filename_language is not None:
        return filename_language
    return LANGUAGES_BY_EXTENSION.get(file_path.suffix.lower(), "unknown")


def hash_content(content: str) -> str:
    """Return a stable SHA-256 hash for UTF-8 text."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def chunk_text(content: str, chunk_size_lines: int) -> list[Chunk]:
    """Split text into deterministic, non-overlapping line chunks."""
    if chunk_size_lines < 1:
        raise ValueError("Chunk size must be positive")

    lines = content.splitlines(keepends=True)
    chunks: list[Chunk] = []
    for offset in range(0, len(lines), chunk_size_lines):
        selected_lines = lines[offset : offset + chunk_size_lines]
        chunks.append(
            Chunk(
                index=len(chunks),
                start_line=offset + 1,
                end_line=offset + len(selected_lines),
                content="".join(selected_lines),
            )
        )
    return chunks


class RepositoryIndexer:
    """Create and incrementally update a persisted repository index."""

    def __init__(self, access: RepositoryAccess, chunk_size_lines: int = 200) -> None:
        if chunk_size_lines < 1:
            raise ValueError("Chunk size must be positive")
        self.access = access
        self.chunk_size_lines = chunk_size_lines

    def index(self, session: Session, repository: Repository) -> IndexingResult:
        """Synchronize persisted files and chunks with repository contents."""
        discovered_files = self.access.list_files(Path(repository.path))
        statement = (
            select(IndexedFile)
            .where(IndexedFile.repository_id == repository.id)
            .options(selectinload(IndexedFile.chunks), selectinload(IndexedFile.symbols))
        )
        existing_files = {file.path: file for file in session.scalars(statement)}
        discovered_paths = {file.path for file in discovered_files}
        indexed = updated = skipped = failed = 0

        for file in discovered_files:
            try:
                content = self.access.read_text(Path(repository.path), Path(file.path)).content
            except RepositoryAccessError:
                failed += 1
                continue

            content_hash = hash_content(content)
            language = detect_language(file.path)
            symbols = extract_python_symbols(content) if language == "python" else []
            existing_file = existing_files.get(file.path)
            if (
                existing_file is not None
                and existing_file.content_hash == content_hash
                and existing_file.chunk_size_lines == self.chunk_size_lines
                and _symbols_match(existing_file.symbols, symbols)
            ):
                skipped += 1
                continue

            chunks = chunk_text(content, self.chunk_size_lines)
            if existing_file is None:
                existing_file = IndexedFile(
                    repository=repository,
                    path=file.path,
                    content_hash=content_hash,
                    language=language,
                    size_bytes=file.size_bytes,
                    chunk_size_lines=self.chunk_size_lines,
                )
                session.add(existing_file)
                indexed += 1
            else:
                existing_file.content_hash = content_hash
                existing_file.language = language
                existing_file.size_bytes = file.size_bytes
                existing_file.chunk_size_lines = self.chunk_size_lines
                existing_file.indexed_at = utc_now()
                existing_file.chunks.clear()
                existing_file.symbols.clear()
                session.flush()
                updated += 1

            existing_file.chunks.extend(
                CodeChunk(
                    chunk_index=chunk.index,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                )
                for chunk in chunks
            )
            existing_file.symbols.extend(
                CodeSymbol(
                    name=symbol.name,
                    qualified_name=symbol.qualified_name,
                    kind=symbol.kind,
                    signature=symbol.signature,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                )
                for symbol in symbols
            )

        deleted_paths = set(existing_files) - discovered_paths
        for deleted_path in deleted_paths:
            session.delete(existing_files[deleted_path])
        session.flush()
        sync_repository_fts(session, repository.id)
        session.commit()

        return IndexingResult(
            discovered=len(discovered_files),
            indexed=indexed,
            updated=updated,
            skipped=skipped,
            deleted=len(deleted_paths),
            failed=failed,
        )


def _symbols_match(
    stored_symbols: list[CodeSymbol],
    extracted_symbols: list[ExtractedSymbol],
) -> bool:
    stored = [
        (
            symbol.name,
            symbol.qualified_name,
            symbol.kind,
            symbol.signature,
            symbol.start_line,
            symbol.end_line,
        )
        for symbol in stored_symbols
    ]
    extracted = [
        (
            symbol.name,
            symbol.qualified_name,
            symbol.kind,
            symbol.signature,
            symbol.start_line,
            symbol.end_line,
        )
        for symbol in extracted_symbols
    ]
    return stored == extracted


def get_index_status(session: Session, repository_id: uuid.UUID) -> IndexStatus:
    """Return persisted file and chunk counts for a repository."""
    file_count, last_indexed_at = session.execute(
        select(func.count(IndexedFile.id), func.max(IndexedFile.indexed_at)).where(
            IndexedFile.repository_id == repository_id
        )
    ).one()
    chunk_count = session.scalar(
        select(func.count(CodeChunk.id))
        .join(IndexedFile)
        .where(IndexedFile.repository_id == repository_id)
    )
    return IndexStatus(
        file_count=file_count,
        chunk_count=chunk_count or 0,
        last_indexed_at=last_indexed_at,
    )
