"""Agent integration: test evidence is persisted, redacted, and execution is opt-in."""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent.config import Settings
from agent.database import Database
from agent.main import create_app
from agent.models import AgentRun, AgentRunMode, AgentRunStatus, PytestExecution, Repository
from agent.models_api import FakeModelClient, ModelResponse, TextContent, TokenUsage, ToolCall
from agent.process_runner import ProcessResult, ProcessRunner
from agent.pytest_runner import PytestRunner
from agent.pytest_tool import PytestTool
from agent.repository_access import RepositoryAccess
from agent.tools import ToolContext, ToolExecutor, ToolRegistry


def client_for(tmp_path: Path, model: FakeModelClient, *, enabled: bool = True) -> TestClient:
    settings = Settings(  # type: ignore[call-arg]
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        workspace_root=tmp_path,
        code_writes_enabled=True,
        test_execution_enabled=enabled,
        _env_file=None,
    )
    return TestClient(create_app(settings, model_client=model))


def responses(paths: list[str] | None = None) -> FakeModelClient:
    return FakeModelClient(
        [
            ModelResponse(
                "tests",
                "fake",
                "test-model",
                (ToolCall("test-call", "run_pytest", {"paths": paths or []}),),
                "tool_use",
                TokenUsage(10, 5),
            ),
            ModelResponse(
                "final",
                "fake",
                "test-model",
                (TextContent("Reported test evidence."),),
                "end_turn",
                TokenUsage(10, 5),
            ),
        ]
    )


def register(client: TestClient, tmp_path: Path, source: str) -> str:
    path = tmp_path / "repo"
    path.mkdir()
    (path / "test_case.py").write_text(source, encoding="utf-8")
    result = client.post("/repositories", json={"name": "Tests", "path": str(path)})
    return str(result.json()["id"])


def test_passing_targeted_run_persists_test_evidence_and_tool_results(tmp_path: Path) -> None:
    model = responses(["test_case.py"])
    with client_for(tmp_path, model) as client:
        repository_id = register(client, tmp_path, "def test_pass():\n    assert True\n")
        created = client.post(
            f"/repositories/{repository_id}/runs", json={"task": "Run tests", "mode": "write"}
        )
        assert created.status_code == 201 and created.json()["status"] == "completed"
        body = created.json()
        evidence = body["test_results"][0]
        assert evidence["status"] == "passed" and evidence["verified"] is True
        assert evidence["counts"]["passed"] == 1
        assert evidence["targets"] == ["test_case.py"]
        assert evidence["tool_call_id"] == "test-call"
        assert body["tool_events"][0]["result"]["test_execution_id"] == evidence["id"]
        saved = client.get(f"/repositories/{repository_id}/runs/{body['id']}").json()
        assert saved["test_results"] == body["test_results"]
    assert "run_pytest" in {definition.name for definition in model.requests[0].tools}
    assert '"verified":true' in str(model.requests[1].messages[-1].content)


def test_failing_tests_are_structured_data_with_redacted_output(tmp_path: Path) -> None:
    model = responses()
    source = "def test_failure():\n    print('password=hunter2')\n    assert False\n"
    with client_for(tmp_path, model) as client:
        repository_id = register(client, tmp_path, source)
        body = client.post(
            f"/repositories/{repository_id}/runs", json={"task": "Test", "mode": "write"}
        ).json()
        evidence = body["test_results"][0]
        assert evidence["status"] == "failed" and evidence["verified"] is False
        assert evidence["counts"]["failed"] == 1
        assert "hunter2" not in str(body)
        assert "[REDACTED]" in evidence["stdout"]
        # The tool completed normally; pytest assertion failures are in structured data.
        assert body["tool_events"][0]["is_error"] is False


@pytest.mark.parametrize(("enabled", "mode"), [(False, "write"), (True, "read_only")])
def test_disabled_or_read_only_runs_cannot_execute_tests(
    tmp_path: Path, enabled: bool, mode: str
) -> None:
    model = responses()
    with client_for(tmp_path, model, enabled=enabled) as client:
        repository_id = register(client, tmp_path, "raise AssertionError('must not execute')\n")
        body = client.post(
            f"/repositories/{repository_id}/runs", json={"task": "Test", "mode": mode}
        ).json()
        assert body["test_results"] == []
        assert body["tool_events"][0]["is_error"] is True
    assert "run_pytest" not in {definition.name for definition in model.requests[0].tools}


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


def test_cancelled_test_execution_is_persisted(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    access = RepositoryAccess(tmp_path)

    async def exercise() -> None:
        blocking = BlockingRunner()
        registry = ToolRegistry()
        registry.register(PytestTool(PytestRunner(access, enabled=True, process_runner=blocking)))
        with database.session_factory() as session:
            repository = Repository(name="Tests", path=str(tmp_path))
            run = AgentRun(
                repository=repository,
                task="Tests",
                mode=AgentRunMode.WRITE,
                status=AgentRunStatus.RUNNING,
            )
            session.add(repository)
            session.commit()
            task = asyncio.create_task(
                ToolExecutor(registry).execute(
                    ToolCall("cancelled-call", "run_pytest", {"paths": []}),
                    ToolContext(repository, session, access, allow_writes=True),
                    agent_run_id=run.id,
                    sequence_number=1,
                )
            )
            await asyncio.wait_for(blocking.entered.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            saved = session.scalar(select(PytestExecution))
            assert saved is not None
            assert saved.status == "cancelled" and saved.verified is False
            assert saved.tool_call_id == "cancelled-call"

    asyncio.run(exercise())
    database.dispose()
