"""Repository-scoped exact, lexical, and symbol search."""

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import case, func, select, text
from sqlalchemy.orm import Session

from agent.models import CodeChunk, CodeSymbol, IndexedFile


class InvalidSearchQueryError(ValueError):
    """Raised when a query contains no searchable text."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One matching source chunk."""

    path: str
    language: str
    start_line: int
    end_line: int
    snippet: str
    score: float


@dataclass(frozen=True, slots=True)
class SymbolSearchResult:
    """One matching code symbol."""

    path: str
    language: str
    name: str
    qualified_name: str
    kind: str
    signature: str
    start_line: int
    end_line: int


def exact_search(
    session: Session,
    repository_id: uuid.UUID,
    query: str,
    *,
    case_sensitive: bool = False,
    limit: int = 20,
) -> list[SearchResult]:
    """Find a literal substring in persisted source chunks."""
    normalized_query = _validate_query(query)
    condition = (
        func.instr(CodeChunk.content, normalized_query) > 0
        if case_sensitive
        else func.instr(func.lower(CodeChunk.content), normalized_query.lower()) > 0
    )
    statement = (
        select(CodeChunk, IndexedFile)
        .join(IndexedFile)
        .where(IndexedFile.repository_id == repository_id, condition)
        .order_by(IndexedFile.path, CodeChunk.chunk_index)
        .limit(limit)
    )
    results: list[SearchResult] = []
    for chunk, indexed_file in session.execute(statement):
        snippet, start_line, end_line = _exact_snippet(
            chunk.content,
            chunk.start_line,
            normalized_query,
            case_sensitive=case_sensitive,
        )
        results.append(
            SearchResult(
                path=indexed_file.path,
                language=indexed_file.language,
                start_line=start_line,
                end_line=end_line,
                snippet=snippet,
                score=1.0,
            )
        )
    return results


def lexical_search(
    session: Session,
    repository_id: uuid.UUID,
    query: str,
    *,
    limit: int = 20,
) -> list[SearchResult]:
    """Run a ranked SQLite FTS5 search over source chunks."""
    fts_query = _fts_query(_validate_query(query))
    statement = text(
        """
        SELECT path, language, start_line, end_line,
               snippet(code_chunks_fts, 6, '<mark>', '</mark>', ' … ', 24) AS snippet,
               bm25(code_chunks_fts) AS rank
        FROM code_chunks_fts
        WHERE code_chunks_fts MATCH :query
          AND repository_id = :repository_id
        ORDER BY rank, path, CAST(start_line AS INTEGER)
        LIMIT :limit
        """
    )
    rows = session.execute(
        statement,
        {"query": fts_query, "repository_id": str(repository_id), "limit": limit},
    ).mappings()
    return [
        SearchResult(
            path=row["path"],
            language=row["language"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            snippet=row["snippet"],
            score=-float(row["rank"]),
        )
        for row in rows
    ]


def symbol_search(
    session: Session,
    repository_id: uuid.UUID,
    query: str,
    *,
    limit: int = 20,
) -> list[SymbolSearchResult]:
    """Search extracted symbols by name and qualified name."""
    normalized_query = _validate_query(query).lower()
    contains = f"%{_escape_like(normalized_query)}%"
    prefix = f"{_escape_like(normalized_query)}%"
    lowered_name = func.lower(CodeSymbol.name)
    lowered_qualified_name = func.lower(CodeSymbol.qualified_name)
    ranking = case(
        (lowered_name == normalized_query, 0),
        (lowered_qualified_name == normalized_query, 1),
        (lowered_name.like(prefix, escape="\\"), 2),
        else_=3,
    )
    statement = (
        select(CodeSymbol, IndexedFile)
        .join(IndexedFile)
        .where(
            IndexedFile.repository_id == repository_id,
            lowered_qualified_name.like(contains, escape="\\"),
        )
        .order_by(ranking, CodeSymbol.qualified_name, IndexedFile.path)
        .limit(limit)
    )
    return [
        SymbolSearchResult(
            path=indexed_file.path,
            language=indexed_file.language,
            name=symbol.name,
            qualified_name=symbol.qualified_name,
            kind=symbol.kind,
            signature=symbol.signature,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
        )
        for symbol, indexed_file in session.execute(statement)
    ]


def sync_repository_fts(session: Session, repository_id: uuid.UUID) -> None:
    """Replace one repository's FTS rows from relational chunks."""
    repository_key = str(repository_id)
    session.execute(
        text("DELETE FROM code_chunks_fts WHERE repository_id = :repository_id"),
        {"repository_id": repository_key},
    )
    rows = session.execute(
        select(CodeChunk, IndexedFile)
        .join(IndexedFile)
        .where(IndexedFile.repository_id == repository_id)
    )
    entries = [
        {
            "chunk_id": str(chunk.id),
            "repository_id": repository_key,
            "path": indexed_file.path,
            "language": indexed_file.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "content": chunk.content,
        }
        for chunk, indexed_file in rows
    ]
    if entries:
        session.execute(
            text(
                """
                INSERT INTO code_chunks_fts(
                    chunk_id, repository_id, path, language, start_line, end_line, content
                ) VALUES (
                    :chunk_id, :repository_id, :path, :language, :start_line, :end_line, :content
                )
                """
            ),
            entries,
        )


def _validate_query(query: str) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        raise InvalidSearchQueryError("Search query must not be blank")
    return normalized_query


def _fts_query(query: str) -> str:
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens:
        raise InvalidSearchQueryError("Search query has no searchable terms")
    return " ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _exact_snippet(
    content: str,
    chunk_start_line: int,
    query: str,
    *,
    case_sensitive: bool,
) -> tuple[str, int, int]:
    searched_content = content if case_sensitive else content.lower()
    searched_query = query if case_sensitive else query.lower()
    match_offset = searched_content.find(searched_query)
    match_line_offset = content.count("\n", 0, match_offset)
    lines = content.splitlines(keepends=True)
    context_start = max(0, match_line_offset - 2)
    context_end = min(len(lines), match_line_offset + 3)
    return (
        "".join(lines[context_start:context_end]),
        chunk_start_line + context_start,
        chunk_start_line + context_end - 1,
    )
