"""Validated Git tools using the same persisted execution boundary as file tools."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agent.git_service import GitError, GitService
from agent.models_api import JsonObject
from agent.repository_tools import ToolArguments, create_repository_tool_registry
from agent.tools import ToolContext, ToolExecutionError, ToolRegistry

type GitAction = Literal[
    "status", "diff", "log", "branches", "create_branch", "checkout", "stage", "unstage", "commit"
]


class GitEmptyArguments(ToolArguments):
    """Arguments for status and branch inspection."""


class GitDiffArguments(ToolArguments):
    """Select the working tree or staged patch."""

    staged: bool = False


class GitLogArguments(ToolArguments):
    """Bound recent commit history."""

    limit: int = Field(default=20, ge=1, le=100)


class GitBranchArguments(ToolArguments):
    """An explicit local branch name."""

    name: str = Field(min_length=1, max_length=255)


class GitPathsArguments(ToolArguments):
    """Explicit repository-relative file paths, never directories or patterns."""

    paths: list[str] = Field(min_length=1, max_length=100)


class GitCommitArguments(ToolArguments):
    """The message for committing the staged index."""

    message: str = Field(min_length=1, max_length=500)


class GitTool:
    """One fixed Git operation; the model cannot supply command-line options."""

    def __init__(
        self,
        service: GitService,
        action: GitAction,
        description: str,
        arguments_model: type[BaseModel],
    ) -> None:
        self.service = service
        self.action = action
        self.name = "git_" + action
        self.description = description
        self.arguments_model = arguments_model

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        if self.action not in {"status", "diff", "log", "branches"} and not context.allow_writes:
            raise ToolExecutionError("This run is read-only")
        path = Path(context.repository.path)
        try:
            if self.action == "status":
                return await self.service.status(path)
            if self.action == "branches":
                return await self.service.branches(path)
            if self.action == "diff":
                return await self.service.diff(
                    path, staged=GitDiffArguments.model_validate(arguments).staged
                )
            if self.action == "log":
                return await self.service.log(
                    path, limit=GitLogArguments.model_validate(arguments).limit
                )
            if self.action == "create_branch":
                return await self.service.create_branch(
                    path, GitBranchArguments.model_validate(arguments).name
                )
            if self.action == "checkout":
                return await self.service.checkout(
                    path, GitBranchArguments.model_validate(arguments).name
                )
            if self.action == "stage":
                return await self.service.stage(
                    path, GitPathsArguments.model_validate(arguments).paths
                )
            if self.action == "unstage":
                return await self.service.unstage(
                    path, GitPathsArguments.model_validate(arguments).paths
                )
            return await self.service.commit(
                path, GitCommitArguments.model_validate(arguments).message
            )
        except GitError as error:
            raise ToolExecutionError(str(error)) from error


def create_agent_tool_registry(
    service: GitService,
    *,
    allow_mutations: bool | None = None,
) -> ToolRegistry:
    """Combine file tools with Git inspection and optionally Git mutation tools."""
    registry = create_repository_tool_registry()
    definitions: list[tuple[GitAction, str, type[BaseModel]]] = [
        (
            "status",
            "Inspect the active repository's Git status and current branch.",
            GitEmptyArguments,
        ),
        ("diff", "Read the working tree or staged Git patch.", GitDiffArguments),
        ("log", "List recent Git commit IDs and subjects.", GitLogArguments),
        ("branches", "List local Git branches and the current checkout.", GitEmptyArguments),
    ]
    if service.allow_mutations and allow_mutations is not False:
        definitions.extend(
            [
                (
                    "create_branch",
                    "Create a local branch at HEAD without switching.",
                    GitBranchArguments,
                ),
                (
                    "checkout",
                    "Switch to an existing local branch when the working tree is clean.",
                    GitBranchArguments,
                ),
                (
                    "stage",
                    "Stage specific repository-relative files or tracked deletions.",
                    GitPathsArguments,
                ),
                (
                    "unstage",
                    "Unstage specific files while preserving working files.",
                    GitPathsArguments,
                ),
                (
                    "commit",
                    "Commit all currently staged changes with the supplied message.",
                    GitCommitArguments,
                ),
            ]
        )
    for action, description, arguments_model in definitions:
        registry.register(GitTool(service, action, description, arguments_model))
    return registry
