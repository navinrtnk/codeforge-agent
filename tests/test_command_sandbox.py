"""Command policy, Docker lifecycle, failure handling, and cancellation without a daemon."""

import asyncio
import csv
import sys
from pathlib import Path
from typing import Any

import pytest

from agent.command_sandbox import DockerCommandSandbox, LocalCommandSandbox, SandboxError
from agent.config import Settings
from agent.process_runner import ProcessResult, ProcessRunner
from agent.pytest_runner import PytestRunner
from agent.repository_access import RepositoryAccess


class RecordingProcess(ProcessRunner):
    def __init__(
        self, *, create_code: int = 0, cleanup_code: int = 0, timeout: bool = False
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.environments: list[dict[str, str]] = []
        self.timeouts: list[float] = []
        self.create_code, self.cleanup_code, self.timeout = create_code, cleanup_code, timeout

    async def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
    ) -> ProcessResult:
        self.commands.append(command)
        self.environments.append(environment)
        self.timeouts.append(timeout_seconds)
        code = (
            self.create_code
            if command[1] == "create"
            else self.cleanup_code
            if command[1] == "rm"
            else 0
        )
        return ProcessResult(
            command,
            code,
            "1 passed in 0.01s\n" if command[1] == "start" else "",
            "",
            1,
            timed_out=self.timeout and command[1] == "start",
        )


@pytest.mark.parametrize(
    "command",
    [
        (),
        (sys.executable, "-c", "print('unsafe')"),
        (sys.executable, "-m", "pip"),
        ("/bin/sh", "-m", "pytest"),
        (sys.executable, "script.py"),
    ],
)
@pytest.mark.parametrize("docker", [False, True])
def test_non_allowlisted_commands_never_reach_a_process(
    tmp_path: Path, command: tuple[str, ...], docker: bool
) -> None:
    process = RecordingProcess()
    sandbox = (
        DockerCommandSandbox(tmp_path, process_runner=process)
        if docker
        else LocalCommandSandbox(process_runner=process)
    )
    with pytest.raises(SandboxError, match="allowlist"):
        asyncio.run(sandbox.run(command, cwd=tmp_path, environment={}))
    assert process.commands == []


def test_operator_allowlist_can_disable_a_tool(tmp_path: Path) -> None:
    sandbox = LocalCommandSandbox(allowed_commands=("ruff",))
    with pytest.raises(SandboxError):
        asyncio.run(sandbox.run((sys.executable, "-m", "pytest"), cwd=tmp_path, environment={}))


def test_docker_limits_mounts_environment_and_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / 'repo, with "quotes"'
    repository.mkdir()
    monkeypatch.setenv("CODEFORGE_OPENAI_API_KEY", "private-value")
    process = RecordingProcess()
    sandbox = DockerCommandSandbox(
        tmp_path,
        cpus=0.5,
        memory_mb=256,
        pids_limit=32,
        tmpfs_mb=32,
        max_timeout_seconds=4,
        process_runner=process,
    )
    result = asyncio.run(
        sandbox.run(
            (sys.executable, "-m", "pytest", "--rootdir", "/repo"),
            cwd=repository,
            environment={
                "CODEFORGE_OPENAI_API_KEY": "private-value",
                "PYTHONPATH": "/host",
                "PYTHONIOENCODING": "utf-8",
                "PATH": "/host/bin",
            },
            timeout_seconds=10,
        )
    )
    create, start, remove = process.commands
    assert [item[1] for item in process.commands] == ["create", "start", "rm"]
    assert create[create.index("--network") + 1] == "none"
    assert create[create.index("--user") + 1] == "65534:65534"
    assert create[create.index("--cpus") + 1] == "0.5"
    assert create[create.index("--memory") + 1] == "256m"
    assert create[create.index("--memory-swap") + 1] == "256m"
    assert create[create.index("--pids-limit") + 1] == "32"
    assert {
        "--pull=never",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
    } <= set(create)
    mount = next(csv.reader([create[create.index("--mount") + 1]]))
    assert mount == [
        "type=bind",
        f"source={repository}",
        "target=/repo",
        "readonly",
        "bind-recursive=disabled",
    ]
    assert "private-value" not in str(process.commands) + str(process.environments)
    assert "PYTHONPATH=/host" not in create and "PATH=/host/bin" not in create
    assert "PYTHONIOENCODING=utf-8" in create
    name = create[create.index("--name") + 1]
    assert start[-1] == remove[-1] == name
    assert "--force" in remove and "--volumes" in remove
    assert process.timeouts[0] <= 4 and process.timeouts[1] <= 4 and process.timeouts[2] == 5
    assert result.command == (
        "/opt/codeforge/.venv/bin/python",
        "-m",
        "pytest",
        "--rootdir",
        "/repo",
    )
    assert result.exit_code == 0 and result.error is None


def test_explicit_network_and_daemon_host_path_mapping(tmp_path: Path) -> None:
    repo = tmp_path / "nested"
    repo.mkdir()
    process = RecordingProcess()
    sandbox = DockerCommandSandbox(
        tmp_path,
        host_workspace_root=Path("/daemon/workspace"),
        network_enabled=True,
        process_runner=process,
    )
    asyncio.run(sandbox.run((sys.executable, "-m", "pytest"), cwd=repo, environment={}))
    create = process.commands[0]
    assert create[create.index("--network") + 1] == "bridge"
    assert "source=/daemon/workspace/nested" in create[create.index("--mount") + 1]


def test_docker_rejects_repository_escape_before_starting(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "link").symlink_to(tmp_path, target_is_directory=True)
    process = RecordingProcess()
    sandbox = DockerCommandSandbox(workspace, process_runner=process)
    with pytest.raises(SandboxError, match="outside"):
        asyncio.run(
            sandbox.run((sys.executable, "-m", "pytest"), cwd=workspace / "link", environment={})
        )
    assert process.commands == []


def test_unexpected_creation_failure_still_attempts_cleanup(tmp_path: Path) -> None:
    class FailingProcess(RecordingProcess):
        async def run(
            self,
            command: tuple[str, ...],
            *,
            cwd: Path,
            environment: dict[str, str],
            timeout_seconds: float = 60,
            max_output_bytes: int = 20_000,
        ) -> ProcessResult:
            if command[1] == "create":
                self.commands.append(command)
                raise RuntimeError("CLI connection failed unexpectedly")
            return await super().run(
                command,
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )

    process = FailingProcess()
    sandbox = DockerCommandSandbox(tmp_path, process_runner=process)
    with pytest.raises(RuntimeError, match="connection"):
        asyncio.run(sandbox.run((sys.executable, "-m", "pytest"), cwd=tmp_path, environment={}))
    assert [command[1] for command in process.commands] == ["create", "rm"]


@pytest.mark.parametrize(
    ("create_code", "cleanup_code", "timeout"), [(125, 1, False), (0, 1, False), (0, 0, True)]
)
def test_docker_failures_never_verify_or_fall_back_to_local(
    tmp_path: Path, create_code: int, cleanup_code: int, timeout: bool
) -> None:
    process = RecordingProcess(create_code=create_code, cleanup_code=cleanup_code, timeout=timeout)
    sandbox = DockerCommandSandbox(tmp_path, process_runner=process)
    runner = PytestRunner(RepositoryAccess(tmp_path), enabled=True, sandbox=sandbox)
    result = asyncio.run(runner.run(tmp_path, []))
    assert not result.verified
    assert result.status == ("timeout" if timeout else "execution_error")
    assert all(command[0] == "docker" for command in process.commands)
    assert process.commands[-1][1] == "rm"
    assert ("start" in [command[1] for command in process.commands]) is (create_code == 0)


@pytest.mark.parametrize("phase", ["create", "start", "rm"])
def test_cancellation_waits_for_creation_and_cleanup(tmp_path: Path, phase: str) -> None:
    async def exercise() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        class BlockingProcess(RecordingProcess):
            async def run(
                self,
                command: tuple[str, ...],
                *,
                cwd: Path,
                environment: dict[str, str],
                timeout_seconds: float = 60,
                max_output_bytes: int = 20_000,
            ) -> ProcessResult:
                if command[1] == phase:
                    entered.set()
                    await release.wait()
                return await super().run(
                    command,
                    cwd=cwd,
                    environment=environment,
                    timeout_seconds=timeout_seconds,
                    max_output_bytes=max_output_bytes,
                )

        process = BlockingProcess()
        sandbox = DockerCommandSandbox(tmp_path, process_runner=process)
        task = asyncio.create_task(
            sandbox.run((sys.executable, "-m", "pytest"), cwd=tmp_path, environment={})
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert process.commands[-1][1] == "rm"
        if phase == "create":
            assert [command[1] for command in process.commands] == ["create", "rm"]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "overrides",
    [
        {"command_execution_backend": "shell"},
        {"sandbox_cpus": 0},
        {"sandbox_cpus": float("nan")},
        {"sandbox_memory_mb": 0},
        {"sandbox_pids_limit": -1},
        {"sandbox_timeout_seconds": float("inf")},
        {"sandbox_allowed_commands": ["sh"]},
        {"sandbox_image": "--privileged"},
        {"sandbox_python_executable": "python"},
        {"docker_host_workspace_root": "relative/path"},
    ],
)
def test_invalid_sandbox_settings_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, **overrides)  # type: ignore[call-arg]
