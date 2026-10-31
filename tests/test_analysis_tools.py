"""Static tools are opt-in and preserve sanitized evidence outside conversation limits."""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent.analysis_tools import StaticAnalysisTool
from agent.config import Settings
from agent.database import Database
from agent.main import create_app
from agent.models import AgentRun, AgentRunMode, AgentRunStatus, AnalysisExecution, Repository
from agent.models_api import FakeModelClient, ModelResponse, TextContent, TokenUsage, ToolCall
from agent.process_runner import ProcessResult, ProcessRunner
from agent.repository_access import RepositoryAccess
from agent.static_analysis import StaticAnalysisRunner
from agent.tools import ToolContext, ToolExecutor, ToolRegistry


class BlockingRunner(ProcessRunner):
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
    ) -> ProcessResult:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("Unreachable")


def client_for(tmp_path: Path, model: FakeModelClient, *, enabled: bool = True) -> TestClient:
    settings = Settings(  # type: ignore[call-arg]
        database_url=f"sqlite:///{tmp_path / 'analysis.db'}",
        workspace_root=tmp_path,
        code_writes_enabled=True,
        analysis_execution_enabled=enabled,
        max_tool_output_bytes=512,
        _env_file=None,
    )
    return TestClient(create_app(settings, model_client=model))


def responses(
    tool: str = "run_ruff", arguments: dict[str, object] | None = None
) -> FakeModelClient:
    return FakeModelClient(
        [
            ModelResponse(
                "analysis",
                "fake",
                "model",
                (ToolCall("analysis-call", tool, arguments or {"paths": []}),),
                "tool_use",
                TokenUsage(10, 5),
            ),
            ModelResponse(
                "final",
                "fake",
                "model",
                (TextContent("Reported diagnostics."),),
                "end_turn",
                TokenUsage(10, 5),
            ),
        ]
    )


def register(client: TestClient, tmp_path: Path, source: str) -> str:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "sample.py").write_text(source)
    response = client.post("/repositories", json={"name": "Analysis", "path": str(repository)})
    return str(response.json()["id"])


@pytest.mark.parametrize(
    ("tool", "source", "code"),
    [
        ("run_ruff", "import os\n", "F401"),
        ("run_mypy", 'x: int = "wrong"\n', "assignment"),
    ],
)
def test_real_tools_persist_findings_and_api_retrieves_them(
    tmp_path: Path,
    tool: str,
    source: str,
    code: str,
) -> None:
    model = responses(tool)
    with client_for(tmp_path, model) as client:
        repository_id = register(client, tmp_path, source)
        response = client.post(
            f"/repositories/{repository_id}/runs",
            json={
                "task": "Check code",
                "mode": "write",
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["status"] == "completed"
        evidence = body["analysis_results"][0]
        assert evidence["tool_call_id"] == "analysis-call" and evidence["tool_name"] == tool
        result = evidence["result"]
        assert result["status"] == "findings" and result["clean"] is False
        assert result["diagnostics_complete"] is True
        assert result["diagnostics"][0]["code"] == code
        assert result["diagnostics"][0]["file"] == "sample.py"
        assert body["tool_events"][0]["is_error"] is False
        # Tool conversational output is bounded; independent evidence retains full diagnostics.
        assert result["process"]["stdout"]
        assert body["test_results"] == []
        saved = client.get(f"/repositories/{repository_id}/runs/{body['id']}").json()
        assert saved["analysis_results"] == body["analysis_results"]
    assert {"run_ruff", "run_mypy"} <= {definition.name for definition in model.requests[0].tools}


@pytest.mark.parametrize(("enabled", "mode"), [(False, "write"), (True, "read_only")])
def test_disabled_or_read_only_runs_do_not_expose_analysis(
    tmp_path: Path,
    enabled: bool,
    mode: str,
) -> None:
    model = responses()
    with client_for(tmp_path, model, enabled=enabled) as client:
        repository_id = register(client, tmp_path, "import os\n")
        body = client.post(
            f"/repositories/{repository_id}/runs",
            json={
                "task": "Check",
                "mode": mode,
            },
        ).json()
        assert body["analysis_results"] == []
        assert body["tool_events"][0]["is_error"] is True
    assert not {"run_ruff", "run_mypy"} & {item.name for item in model.requests[0].tools}


def test_diagnostics_and_process_output_are_redacted_before_persisting(tmp_path: Path) -> None:
    model = responses("run_mypy")
    with client_for(tmp_path, model) as client:
        repository_id = register(client, tmp_path, 'reveal_type("password=hunter2")\n')
        body = client.post(
            f"/repositories/{repository_id}/runs",
            json={
                "task": "Check",
                "mode": "write",
            },
        ).json()
        assert "hunter2" not in str(body)
        assert "REDACTED" in str(body["analysis_results"])


@pytest.mark.parametrize("arguments", [{"paths": [], "fix": True}, {"paths": ["../outside.py"]}])
def test_extra_arguments_and_traversal_are_rejected(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    model = responses(arguments=arguments)
    with client_for(tmp_path, model) as client:
        repository_id = register(client, tmp_path, "import os\n")
        body = client.post(
            f"/repositories/{repository_id}/runs",
            json={
                "task": "Check",
                "mode": "write",
            },
        ).json()
        assert body["analysis_results"] == []
        assert body["tool_events"][0]["is_error"] is True


def test_cancelled_execution_is_persisted_and_wrong_run_mode_rejected(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    access = RepositoryAccess(tmp_path)

    async def exercise() -> None:
        blocking = BlockingRunner()
        registry = ToolRegistry()
        registry.register(
            StaticAnalysisTool(
                StaticAnalysisRunner(access, enabled=True, process_runner=blocking), "ruff"
            )
        )
        with database.session_factory() as session:
            repository = Repository(name="Analysis", path=str(tmp_path))
            run = AgentRun(
                repository=repository,
                task="Check",
                mode=AgentRunMode.READ_ONLY,
                status=AgentRunStatus.RUNNING,
            )
            session.add(repository)
            session.commit()
            executor = ToolExecutor(registry)
            context = ToolContext(repository, session, access, allow_writes=True)
            await executor.execute(
                ToolCall("bad-mode", "run_ruff", {"paths": []}),
                context,
                agent_run_id=run.id,
                sequence_number=1,
            )
            assert not blocking.entered.is_set()
            assert session.scalar(select(AnalysisExecution)) is None
            run.mode = AgentRunMode.WRITE
            session.commit()
            task = asyncio.create_task(
                executor.execute(
                    ToolCall("cancelled", "run_ruff", {"paths": []}),
                    context,
                    agent_run_id=run.id,
                    sequence_number=2,
                )
            )
            await asyncio.wait_for(blocking.entered.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            saved = session.scalar(select(AnalysisExecution))
            assert saved is not None
            assert saved.result["status"] == "cancelled" and saved.result["clean"] is False
            assert saved.tool_call_id == "cancelled"

    asyncio.run(exercise())
    database.dispose()
