"""Allowlisted local commands and disposable Docker execution boundaries."""

import asyncio
import csv
import io
import logging
import math
import os
import sys
import uuid
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Protocol

from agent.config import Settings
from agent.process_runner import ProcessResult, ProcessRunner

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """An invalid command or repository boundary; never fall back to local execution."""


class CommandSandbox(Protocol):
    """A command backend with its own repository and null-device paths."""

    null_device: str

    def repository_path(self, repository: Path) -> Path: ...

    async def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
    ) -> ProcessResult: ...


class LocalCommandSandbox:
    """Preserve trusted local execution with a fixed interpreter/module allowlist."""

    null_device = os.devnull

    def __init__(
        self,
        python_executable: Path | None = None,
        *,
        allowed_commands: tuple[str, ...] = ("pytest", "ruff", "mypy"),
        max_timeout_seconds: float = 60,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self.python = str((python_executable or Path(sys.executable)).expanduser().absolute())
        if (
            not set(allowed_commands) <= {"pytest", "ruff", "mypy"}
            or not math.isfinite(max_timeout_seconds)
            or max_timeout_seconds <= 0
        ):
            raise ValueError("Invalid command sandbox policy")
        self.allowed_commands = allowed_commands
        self.max_timeout_seconds = max_timeout_seconds
        self.process_runner = process_runner or ProcessRunner()

    def repository_path(self, repository: Path) -> Path:
        return repository

    def _validate(self, command: tuple[str, ...], timeout: float, output_limit: int) -> None:
        if (
            len(command) < 3
            or command[0] != self.python
            or command[1] != "-m"
            or command[2] not in self.allowed_commands
        ):
            raise SandboxError("Command is not in the configured sandbox allowlist")
        if timeout <= 0 or output_limit < 256:
            raise SandboxError("Invalid sandbox execution limits")

    async def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
    ) -> ProcessResult:
        self._validate(command, timeout_seconds, max_output_bytes)
        return await self.process_runner.run(
            command,
            cwd=cwd,
            environment=environment,
            timeout_seconds=min(timeout_seconds, self.max_timeout_seconds),
            max_output_bytes=max_output_bytes,
        )


async def _settle(task: asyncio.Task[ProcessResult]) -> tuple[ProcessResult, bool]:
    """Finish bounded lifecycle operations even if cancellation arrives during cleanup."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


class DockerCommandSandbox(LocalCommandSandbox):
    """Run the selected tool in a non-root, read-only, resource-limited container."""

    null_device = "/dev/null"

    def __init__(
        self,
        workspace_root: Path,
        python_executable: Path | None = None,
        *,
        image: str = "codeforge-agent:local",
        container_python: str = "/opt/codeforge/.venv/bin/python",
        host_workspace_root: Path | None = None,
        cpus: float = 1,
        memory_mb: int = 512,
        pids_limit: int = 128,
        tmpfs_mb: int = 128,
        network_enabled: bool = False,
        allowed_commands: tuple[str, ...] = ("pytest", "ruff", "mypy"),
        max_timeout_seconds: float = 60,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        super().__init__(
            python_executable,
            allowed_commands=allowed_commands,
            max_timeout_seconds=max_timeout_seconds,
            process_runner=process_runner,
        )
        self.workspace_root = workspace_root.expanduser().resolve()
        if (
            not math.isfinite(cpus)
            or cpus <= 0
            or memory_mb < 16
            or pids_limit < 16
            or tmpfs_mb < 16
            or (host_workspace_root is not None and not host_workspace_root.is_absolute())
        ):
            raise ValueError("Invalid Docker sandbox resources or host workspace path")
        self.host_workspace_root = host_workspace_root
        self.image = image
        self.container_python = container_python
        self.cpus, self.memory_mb = cpus, memory_mb
        self.pids_limit, self.tmpfs_mb = pids_limit, tmpfs_mb
        self.network_enabled = network_enabled

    def repository_path(self, repository: Path) -> Path:
        return Path("/repo")

    async def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
    ) -> ProcessResult:
        self._validate(command, timeout_seconds, max_output_bytes)
        repository = cwd.resolve()
        if not repository.is_relative_to(self.workspace_root) or not repository.is_dir():
            raise SandboxError("Sandbox repository is outside the workspace or missing")
        source = (
            self.host_workspace_root / repository.relative_to(self.workspace_root)
            if self.host_workspace_root is not None
            else repository
        )
        mount = io.StringIO()
        csv.writer(mount, lineterminator="").writerow(
            [
                "type=bind",
                f"source={source}",
                "target=/repo",
                "readonly",
                "bind-recursive=disabled",
            ]
        )
        name = "codeforge-sandbox-" + uuid.uuid4().hex
        arguments = [
            "docker",
            "create",
            "--name",
            name,
            "--pull=never",
            "--init",
            "--read-only",
            "--network",
            "bridge" if self.network_enabled else "none",
            "--user",
            "65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--no-healthcheck",
            "--log-driver=none",
            "--cpus",
            str(self.cpus),
            "--memory",
            f"{self.memory_mb}m",
            "--memory-swap",
            f"{self.memory_mb}m",
            "--pids-limit",
            str(self.pids_limit),
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={self.tmpfs_mb}m,mode=1777",
            "--mount",
            mount.getvalue(),
            "--workdir",
            "/repo",
            "--entrypoint",
            self.container_python,
        ]
        # Only tool controls enter the container; host paths and credentials stay outside.
        safe_environment = {
            key: environment[key]
            for key in (
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                "PYTHONIOENCODING",
                "PYTHONUNBUFFERED",
                "NO_COLOR",
            )
            if key in environment
        }
        safe_environment.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPYCACHEPREFIX="/tmp/pycache",
            HOME="/tmp",
            TMPDIR="/tmp",
        )
        for key, value in sorted(safe_environment.items()):
            arguments.extend(["--env", f"{key}={value}"])
        logical_command = (self.container_python, *command[1:])
        arguments.extend([self.image, *command[1:]])
        docker_environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "PATH",
                "SYSTEMROOT",
                "WINDIR",
                "TEMP",
                "TMP",
                "TMPDIR",
                "HOME",
                "USERPROFILE",
                "DOCKER_HOST",
                "DOCKER_CONTEXT",
                "DOCKER_CONFIG",
                "DOCKER_TLS_VERIFY",
                "DOCKER_CERT_PATH",
            }
        }
        started = perf_counter()
        timeout = min(timeout_seconds, self.max_timeout_seconds)
        creation = asyncio.create_task(
            self.process_runner.run(
                tuple(arguments),
                cwd=repository,
                environment=docker_environment,
                timeout_seconds=min(timeout, 10),
                max_output_bytes=max_output_bytes,
            )
        )
        result: ProcessResult
        try:
            created = await asyncio.shield(creation)
            if created.error or created.timed_out or created.exit_code != 0:
                result = replace(created, error="Docker sandbox could not create the container")
            else:
                remaining = timeout - (perf_counter() - started)
                if remaining <= 0:
                    result = ProcessResult(logical_command, None, "", "", 0, timed_out=True)
                else:
                    result = await self.process_runner.run(
                        ("docker", "start", "--attach", name),
                        cwd=repository,
                        environment=docker_environment,
                        timeout_seconds=remaining,
                        max_output_bytes=max_output_bytes,
                    )
        finally:
            # Settle create before remove so cancellation cannot race a late creation.
            cancelled = False
            # Unexpected CLI failure must not skip a best-effort container removal.
            with suppress(Exception):
                _, cancelled = await _settle(creation)
            cleanup, cleanup_cancelled = await _settle(
                asyncio.create_task(
                    self.process_runner.run(
                        ("docker", "rm", "--force", "--volumes", name),
                        cwd=repository,
                        environment=docker_environment,
                        timeout_seconds=5,
                        max_output_bytes=1024,
                    )
                )
            )
            cleanup_failed = bool(cleanup.error or cleanup.timed_out or cleanup.exit_code != 0)
            creation_succeeded = (
                not creation.cancelled()
                and creation.exception() is None
                and creation.result().exit_code == 0
            )
            if cleanup_failed and creation_succeeded:
                logger.error("Docker sandbox cleanup failed; container may remain: %s", name)
            if cancelled or cleanup_cancelled:
                raise asyncio.CancelledError
        if cleanup_failed and created.exit_code == 0:
            result = replace(result, error=f"Docker sandbox cleanup failed: {name}")
        return replace(
            result,
            command=logical_command,
            duration_ms=(perf_counter() - started) * 1000,
        )


def create_command_sandbox(settings: Settings, python_executable: Path | None) -> CommandSandbox:
    """Construct an explicit backend; Docker failures never select the local one."""
    if settings.command_execution_backend == "local":
        return LocalCommandSandbox(
            python_executable,
            allowed_commands=settings.sandbox_allowed_commands,
            max_timeout_seconds=settings.sandbox_timeout_seconds,
        )
    return DockerCommandSandbox(
        settings.workspace_root,
        python_executable,
        image=settings.sandbox_image,
        container_python=settings.sandbox_python_executable,
        host_workspace_root=settings.docker_host_workspace_root,
        cpus=settings.sandbox_cpus,
        memory_mb=settings.sandbox_memory_mb,
        pids_limit=settings.sandbox_pids_limit,
        tmpfs_mb=settings.sandbox_tmpfs_mb,
        network_enabled=settings.sandbox_network_enabled,
        allowed_commands=settings.sandbox_allowed_commands,
        max_timeout_seconds=settings.sandbox_timeout_seconds,
    )
