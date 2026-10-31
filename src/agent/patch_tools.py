"""Schema-validated patch tools with per-run write authorization."""

from pathlib import Path

from pydantic import BaseModel, Field

from agent.models_api import JsonObject
from agent.patch_service import PatchService
from agent.patches import PatchError
from agent.repository_tools import ToolArguments
from agent.tools import ToolContext, ToolExecutionError


class PatchArguments(ToolArguments):
    """A complete unified diff using repository-relative text-file paths."""

    patch: str = Field(min_length=1)


class PatchTool:
    """Validate a patch or apply it only when both service and run allow writes."""

    arguments_model = PatchArguments

    def __init__(self, service: PatchService, *, dry_run: bool) -> None:
        self.service = service
        self.dry_run = dry_run
        self.name = "validate_patch" if dry_run else "apply_patch"
        self.description = (
            "Dry-run a unified text diff against current files without writing."
            if dry_run
            else "Apply a validated unified text diff and return the updated Git diff."
        )

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        if not self.dry_run and not context.allow_writes:
            raise ToolExecutionError("This run is read-only")
        try:
            return await self.service.execute(
                Path(context.repository.path),
                PatchArguments.model_validate(arguments).patch,
                dry_run=self.dry_run,
            )
        except PatchError as error:
            raise ToolExecutionError(str(error)) from error
