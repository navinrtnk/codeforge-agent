"""Enforced repair integration tests using real patching and pytest execution."""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent.agent_runner import AgentRunner
from agent.config import Settings
from agent.database import Database
from agent.git_service import GitService
from agent.main import create_app
from agent.models import AgentRun, AgentRunMode, AgentRunStatus, AgentRunType, Repository
from agent.models_api import FakeModelClient, ModelResponse, TextContent, TokenUsage, ToolCall
from agent.patch_service import PatchService
from agent.patch_tools import PatchTool
from agent.process_runner import ProcessResult, ProcessRunner
from agent.pytest_runner import PytestRunner
from agent.pytest_tool import PytestTool
from agent.repository_access import RepositoryAccess
from agent.tools import ToolRegistry


def response(*content: TextContent | ToolCall) -> ModelResponse:
    return ModelResponse("response", "fake", "test", content, "end_turn", TokenUsage(10, 5))


def patch(before: int = 0, after: int = 1, *, call_id: str = "patch") -> ModelResponse:
    return response(
        TextContent("The value is incorrect. I will update it and verify the existing test."),
        ToolCall(
            call_id,
            "apply_patch",
            {
                "patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n"
                f"-value = {before}\n+value = {after}\n"
            },
        ),
    )


def client_for(tmp_path: Path, model: FakeModelClient, **settings: object) -> TestClient:
    options: dict[str, object] = {
        "database_url": "sqlite://",
        "workspace_root": tmp_path,
        "code_writes_enabled": True,
        "test_execution_enabled": True,
        "_env_file": None,
        **settings,
    }
    return TestClient(create_app(Settings(**options), model_client=model))  # type: ignore[arg-type]


def register(client: TestClient, tmp_path: Path, *, test: str | None = None) -> tuple[str, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "app.py").write_text("value = 0\n", encoding="utf-8", newline="\n")
    (repository / "test_app.py").write_text(
        test
        if test is not None
        else "from app import value\ndef test_value():\n    assert value == 1\n",
        encoding="utf-8",
    )
    result = client.post("/repositories", json={"name": "Repair", "path": str(repository)})
    assert result.status_code == 201
    return f"/repositories/{result.json()['id']}/runs", repository


@pytest.mark.parametrize("output_limit", [256, 100_000])
def test_repair_automatically_verifies_and_stops_without_another_model_call(
    tmp_path: Path, output_limit: int
) -> None:
    model = FakeModelClient(
        [
            response(
                TextContent("I will investigate."),
                ToolCall("read", "read_file", {"path": "app.py"}),
            ),
            patch(),
        ]
    )
    with client_for(tmp_path, model, max_tool_output_bytes=output_limit) as client:
        url, repository = register(client, tmp_path)
        created = client.post(url + "/bug-fix", json={"task": "Fix the wrong value"})
        assert created.status_code == 201
        body = created.json()
        assert body["run_type"] == "bug_fix" and body["mode"] == "write"
        assert body["status"] == "completed" and body["verified"] is True
        assert body["repair_attempts"] == 1 and body["error_message"] is None
        assert (repository / "app.py").read_text() == "value = 1\n"
        assert body["test_results"][0]["verified"] is True
        assert body["test_results"][0]["targets"] == []
        assert body["tool_events"][-1]["tool_name"] == "run_pytest"
        assert client.get(url + f"/{body['id']}").json() == body
    assert len(model.requests) == 2
    assert "Investigate the bug" in str(model.requests[0].messages[0])


def test_model_selected_test_success_cannot_replace_full_suite_verification(tmp_path: Path) -> None:
    model = FakeModelClient(
        [
            response(ToolCall("subset", "run_pytest", {"paths": ["test_smoke.py"]})),
            patch(after=2),
        ]
    )
    with client_for(tmp_path, model, bug_fix_max_attempts=1) as client:
        url, repository = register(client, tmp_path)
        (repository / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "failed" and body["verified"] is False
        assert [item["verified"] for item in body["test_results"]] == [True, False]
        assert body["test_results"][-1]["targets"] == []


def test_multiple_patches_in_one_batch_are_verified_together(tmp_path: Path) -> None:
    model = FakeModelClient(
        [response(*patch(after=2).content, *patch(before=2, call_id="patch-2").content)]
    )
    with client_for(tmp_path, model, bug_fix_max_attempts=1) as client:
        url, _ = register(client, tmp_path)
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "completed" and body["verified"] is True
        assert body["repair_attempts"] == 1 and len(body["test_results"]) == 1


def test_failed_verification_is_fed_back_before_a_successful_retry(tmp_path: Path) -> None:
    model = FakeModelClient([patch(after=2), patch(before=2)])
    with client_for(tmp_path, model) as client:
        url, _ = register(client, tmp_path)
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "completed" and body["verified"] is True
        assert body["repair_attempts"] == 2
        assert [item["status"] for item in body["test_results"]] == ["failed", "passed"]
    feedback = str(model.requests[1].messages)
    assert "assert 2 == 1" in feedback
    assert "repair is unverified" in feedback


def test_retry_budget_stops_further_patches_and_retains_unverified_edits(tmp_path: Path) -> None:
    model = FakeModelClient([patch(after=2), patch(before=2, after=3), patch(before=3)])
    with client_for(tmp_path, model, bug_fix_max_attempts=2) as client:
        url, repository = register(client, tmp_path)
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "failed" and body["verified"] is False
        assert body["repair_attempts"] == 2 and len(body["test_results"]) == 2
        assert "unverified" in body["error_message"] and "2 attempts" in body["error_message"]
        assert (repository / "app.py").read_text() == "value = 3\n"
    assert len(model.requests) == 2


@pytest.mark.parametrize("test", ["", "import pytest\ndef test_skip():\n    pytest.skip('skip')\n"])
def test_empty_or_skipped_suites_do_not_verify_a_repair(tmp_path: Path, test: str) -> None:
    with client_for(tmp_path, FakeModelClient([patch()]), bug_fix_max_attempts=1) as client:
        url, _ = register(client, tmp_path, test=test)
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "failed" and body["verified"] is False
        assert body["test_results"][0]["verified"] is False


def test_claimed_success_without_a_patch_is_unverified(tmp_path: Path) -> None:
    model = FakeModelClient([response(TextContent("Fixed and all tests passed!"))])
    with client_for(tmp_path, model) as client:
        url, _ = register(client, tmp_path)
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "failed" and body["verified"] is False
        assert body["repair_attempts"] == 0 and body["test_results"] == []
        assert "unverified" in body["error_message"]


def test_iteration_limit_does_not_skip_verification_of_the_last_patch(tmp_path: Path) -> None:
    with client_for(tmp_path, FakeModelClient([patch()]), agent_max_iterations=1) as client:
        url, _ = register(client, tmp_path)
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "completed" and body["verified"] is True


def test_iteration_limit_bounds_investigation_without_patches(tmp_path: Path) -> None:
    model = FakeModelClient([response(ToolCall("read", "read_file", {"path": "app.py"}))])
    with client_for(tmp_path, model, agent_max_iterations=1) as client:
        url, _ = register(client, tmp_path)
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "failed" and body["verified"] is False
        assert "maximum of 1 model iterations" in body["error_message"]


def test_test_timeout_is_unverified(tmp_path: Path) -> None:
    with client_for(
        tmp_path, FakeModelClient([patch()]), bug_fix_max_attempts=1, test_timeout_seconds=0.05
    ) as client:
        url, _ = register(
            client, tmp_path, test="import time\ndef test_slow():\n    time.sleep(5)\n"
        )
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "failed" and body["verified"] is False
        assert body["test_results"][0]["status"] == "timeout"


def test_rejected_patches_consume_attempts_without_running_tests(tmp_path: Path) -> None:
    with client_for(
        tmp_path, FakeModelClient([patch(before=99)]), bug_fix_max_attempts=1
    ) as client:
        url, repository = register(client, tmp_path)
        body = client.post(url + "/bug-fix", json={"task": "Fix value"}).json()
        assert body["status"] == "failed" and body["repair_attempts"] == 1
        assert body["tool_events"][0]["is_error"] is True
        assert body["test_results"] == []
        assert (repository / "app.py").read_text() == "value = 0\n"


@pytest.mark.parametrize("setting", ["code_writes_enabled", "test_execution_enabled"])
def test_bug_fix_requires_both_operator_permissions(tmp_path: Path, setting: str) -> None:
    model = FakeModelClient([])
    with client_for(tmp_path, model, **{setting: False}) as client:
        url, _ = register(client, tmp_path)
        result = client.post(url + "/bug-fix", json={"task": "Fix value"})
        assert result.status_code == 403
        assert client.get(url).json() == []
    assert model.requests == []


def test_bug_fix_rejects_blank_tasks_and_missing_repositories(tmp_path: Path) -> None:
    with client_for(tmp_path, FakeModelClient([])) as client:
        url, _ = register(client, tmp_path)
        assert client.post(url + "/bug-fix", json={"task": "  "}).status_code == 422
        assert (
            client.post(
                "/repositories/00000000-0000-0000-0000-000000000000/runs/bug-fix",
                json={"task": "Fix value"},
            ).status_code
            == 404
        )


class BlockingProcess(ProcessRunner):
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


@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_during_verification_are_persisted(
    tmp_path: Path, cancel: bool
) -> None:
    database = Database("sqlite://")
    database.create_schema()
    access = RepositoryAccess(tmp_path)
    (tmp_path / "app.py").write_text("value = 0\n", encoding="utf-8", newline="\n")

    async def exercise() -> None:
        blocking = BlockingProcess()
        registry = ToolRegistry()
        registry.register(
            PatchTool(PatchService(access, GitService(access), allow_writes=True), dry_run=False)
        )
        registry.register(PytestTool(PytestRunner(access, enabled=True, process_runner=blocking)))
        runner = AgentRunner(
            FakeModelClient([patch()]),
            registry,
            access,
            mode=AgentRunMode.WRITE,
            run_type=AgentRunType.BUG_FIX,
            timeout_seconds=10 if cancel else 0.5,
        )
        with database.session_factory() as session:
            repository = Repository(name="Repair", path=str(tmp_path))
            session.add(repository)
            session.commit()
            task = asyncio.create_task(runner.run(session, repository, "Fix value"))
            await asyncio.wait_for(blocking.entered.wait(), timeout=2)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
            run = session.scalar(select(AgentRun))
            assert run is not None and run.status is AgentRunStatus.FAILED
            assert run.verified is False and run.repair_attempts == 1
            assert run.error_message is not None and "unverified" in run.error_message
            assert ("cancelled" if cancel else "timed out") in run.error_message
            assert run.test_results[0].status == "cancelled"
            assert run.test_results[0].verified is False

    asyncio.run(exercise())
    database.dispose()
