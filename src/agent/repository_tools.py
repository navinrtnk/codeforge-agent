"""Read-only tools operating within one trusted repository context."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.embeddings import EmbeddingClient, EmbeddingError
from agent.models_api import JsonObject
from agent.repository_access import RepositoryAccessError
from agent.search import InvalidSearchQueryError, exact_search, lexical_search
from agent.semantic_search import search_with_embeddings
from agent.tools import ToolContext, ToolExecutionError, ToolRegistry


class ToolArguments(BaseModel):
    """Strict base model for all tool argument schemas."""

    model_config = ConfigDict(extra="forbid")


class ListFilesArguments(ToolArguments):
    """Arguments for repository file discovery."""

    limit: int = Field(default=500, ge=1, le=5_000)


class ListFilesTool:
    """List safe, non-ignored files in the active repository."""

    name = "list_files"
    description = "List files in the active repository with relative paths and byte sizes."
    arguments_model = ListFilesArguments

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        validated = ListFilesArguments.model_validate(arguments)
        try:
            files = context.access.list_files(Path(context.repository.path))
        except RepositoryAccessError as error:
            raise ToolExecutionError(str(error)) from error
        selected = files[: validated.limit]
        return {
            "files": [{"path": file.path, "size_bytes": file.size_bytes} for file in selected],
            "count": len(selected),
            "total": len(files),
            "truncated": len(selected) < len(files),
        }


class ReadFileArguments(ToolArguments):
    """Arguments for a bounded text-file read."""

    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class ReadFileTool:
    """Read a safe UTF-8 file or inclusive line range."""

    name = "read_file"
    description = "Read a UTF-8 repository file using a relative path and optional line range."
    arguments_model = ReadFileArguments

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        validated = ReadFileArguments.model_validate(arguments)
        try:
            content = context.access.read_text(
                Path(context.repository.path),
                Path(validated.path),
                start_line=validated.start_line,
                end_line=validated.end_line,
            )
        except RepositoryAccessError as error:
            raise ToolExecutionError(str(error)) from error
        return {
            "path": content.path,
            "content": content.content,
            "start_line": content.start_line,
            "end_line": content.end_line,
            "total_lines": content.total_lines,
        }


class SearchCodeArguments(ToolArguments):
    """Arguments for indexed repository search."""

    query: str = Field(min_length=1, max_length=500)
    mode: Literal["lexical", "exact", "semantic", "hybrid"] = "lexical"
    case_sensitive: bool = False
    limit: int = Field(default=20, ge=1, le=100)


class SearchCodeTool:
    """Search persisted source chunks in the active repository."""

    name = "search_code"
    description = "Search indexed code using lexical, exact, semantic, or hybrid ranking."
    arguments_model = SearchCodeArguments

    def __init__(
        self,
        embedding_client: EmbeddingClient | None = None,
        *,
        lexical_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ) -> None:
        self.embedding_client = embedding_client
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        validated = SearchCodeArguments.model_validate(arguments)
        try:
            if validated.mode == "exact":
                results = exact_search(
                    context.session,
                    context.repository.id,
                    validated.query,
                    case_sensitive=validated.case_sensitive,
                    limit=validated.limit,
                )
            elif (validated.mode == "semantic" or validated.mode == "hybrid") and not (
                validated.mode == "hybrid" and self.semantic_weight == 0
            ):
                if self.embedding_client is None:
                    raise ToolExecutionError("Semantic search requires enabled embeddings")
                results = await search_with_embeddings(
                    context.session,
                    context.repository.id,
                    validated.query,
                    self.embedding_client,
                    mode=validated.mode,
                    limit=validated.limit,
                    lexical_weight=self.lexical_weight,
                    semantic_weight=self.semantic_weight,
                )
            else:
                results = lexical_search(
                    context.session,
                    context.repository.id,
                    validated.query,
                    limit=validated.limit,
                )
        except (InvalidSearchQueryError, EmbeddingError) as error:
            raise ToolExecutionError(str(error)) from error
        return {
            "results": [
                {
                    "path": result.path,
                    "language": result.language,
                    "start_line": result.start_line,
                    "end_line": result.end_line,
                    "snippet": result.snippet,
                    "score": result.score,
                }
                for result in results
            ],
            "count": len(results),
        }


def create_repository_tool_registry(
    embedding_client: EmbeddingClient | None = None,
    *,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> ToolRegistry:
    """Create the explicit set of repository tools available to agents."""
    registry = ToolRegistry()
    registry.register(ListFilesTool())
    registry.register(ReadFileTool())
    registry.register(
        SearchCodeTool(
            embedding_client,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
        )
    )
    return registry
