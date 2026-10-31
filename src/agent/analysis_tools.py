"""Opt-in agent tools with independently persisted static analysis evidence."""

import asyncio
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, Field

from agent.models import AgentRun, AgentRunMode, AgentRunStatus, AnalysisExecution
from agent.models_api import JsonObject
from agent.project_tools import AnalysisError, AnalysisTool
from agent.repository_tools import ToolArguments
from agent.static_analysis import StaticAnalysisRunner
from agent.tools import ToolContext, ToolExecutionError, redact_json_object


class AnalysisArguments(ToolArguments):
    """Empty paths select the project; otherwise check relative Python files/directories."""

    paths: list[str] = Field(default_factory=list, max_length=50)


class StaticAnalysisTool:
    """Configurations may load plugins; require the same authority as test execution."""

    arguments_model = AnalysisArguments

    def __init__(self, runner: StaticAnalysisRunner, tool: AnalysisTool) -> None:
        self.runner = runner
        self.tool = tool
        self.name = f"run_{tool}"
        self.description = (
            f"Run {tool} on the project or relative Python paths without automatic fixes. "
            "Return static diagnostics, not test verification."
        )

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        run = context.session.get(AgentRun, context.agent_run_id) if context.agent_run_id else None
        if (
            not context.allow_writes
            or run is None
            or run.repository_id != context.repository.id
            or context.tool_call_id is None
            or run.mode is not AgentRunMode.WRITE
            or run.status is not AgentRunStatus.RUNNING
        ):
            raise ToolExecutionError("Static analysis requires a matching active write-mode run")
        paths = AnalysisArguments.model_validate(arguments).paths
        try:
            result = await self.runner.run(Path(context.repository.path), self.tool, paths)
        except AnalysisError as error:
            raise ToolExecutionError(str(error)) from error
        except asyncio.CancelledError:
            self._persist(
                context,
                {
                    "tool": self.tool,
                    "status": "cancelled",
                    "clean": False,
                    "diagnostics": [],
                    "diagnostics_complete": False,
                    "diagnostics_truncated": False,
                },
            )
            raise
        data: JsonObject = asdict(result)
        # JSON-compatible collections before database serialization.
        data["targets"] = list(result.targets)
        data["diagnostics"] = [asdict(diagnostic) for diagnostic in result.diagnostics]
        process = asdict(result.process)
        process["command"] = list(result.process.command)
        process["error_message"] = process.pop("error")
        data["process"] = process
        sanitized = redact_json_object(data)
        execution = self._persist(context, sanitized)
        return {"analysis_execution_id": str(execution.id), **sanitized}

    def _persist(self, context: ToolContext, data: JsonObject) -> AnalysisExecution:
        execution = AnalysisExecution(
            agent_run_id=context.agent_run_id,
            tool_call_id=context.tool_call_id,
            tool_name=self.name,
            result=redact_json_object(data),
        )
        context.session.add(execution)
        context.session.commit()
        return execution
