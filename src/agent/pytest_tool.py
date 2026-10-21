"""Run pytest through validated arguments and persist independent test evidence."""

import asyncio
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from agent.models import AgentRun, AgentRunMode, AgentRunStatus, PytestExecution
from agent.models_api import JsonObject
from agent.pytest_runner import COUNTS, PytestRunner, PytestRunnerError
from agent.repository_tools import ToolArguments
from agent.tools import ToolContext, ToolExecutionError, redact_json_object


class PytestArguments(ToolArguments):
    """An empty list runs the configured suite; otherwise run these relative test paths."""

    paths: list[str] = Field(default_factory=list, max_length=50)


class PytestTool:
    """Allow local repository code execution only in an explicitly enabled write run."""

    name = "run_pytest"
    description = "Run pytest on the full suite or relative test paths and return test evidence."
    arguments_model = PytestArguments

    def __init__(self, runner: PytestRunner) -> None:
        self.runner = runner

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        if not context.allow_writes:
            raise ToolExecutionError("Test execution requires a write-mode run")
        run = context.session.get(AgentRun, context.agent_run_id) if context.agent_run_id else None
        if (
            run is None
            or run.repository_id != context.repository.id
            or context.tool_call_id is None
            or run.mode is not AgentRunMode.WRITE
            or run.status is not AgentRunStatus.RUNNING
        ):
            raise ToolExecutionError("Test execution requires a matching active run")
        paths = PytestArguments.model_validate(arguments).paths
        started = perf_counter()
        try:
            result = await self.runner.run(Path(context.repository.path), paths)
        except PytestRunnerError as error:
            raise ToolExecutionError(str(error)) from error
        except asyncio.CancelledError:
            self._persist(
                context,
                {
                    "command": [],
                    "targets": paths,
                    "config_file": None,
                    "status": "cancelled",
                    "counts": dict.fromkeys(COUNTS, 0),
                    "summary_complete": False,
                    "verified": False,
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                    "error_message": "Test execution cancelled",
                    "duration_ms": (perf_counter() - started) * 1000,
                },
            )
            raise
        data: JsonObject = asdict(result.process)
        data["error_message"] = data.pop("error")
        data.update(
            targets=list(result.targets),
            config_file=result.config_file,
            status=result.status,
            counts=result.counts,
            summary_complete=result.summary_complete,
            verified=result.verified,
        )
        data["command"] = list(result.process.command)
        sanitized = redact_json_object(data)
        execution = self._persist(context, sanitized)
        return {"test_execution_id": str(execution.id), **sanitized}

    @staticmethod
    def _persist(context: ToolContext, data: JsonObject) -> PytestExecution:
        execution = PytestExecution(
            agent_run_id=context.agent_run_id,
            tool_call_id=context.tool_call_id,
            **redact_json_object(data),
        )
        context.session.add(execution)
        context.session.commit()
        return execution
