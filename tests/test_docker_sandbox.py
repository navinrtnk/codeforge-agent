"""Opt-in tests against a real Linux Docker daemon and the built worker image."""

import asyncio
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.command_sandbox import DockerCommandSandbox
from agent.config import Settings
from agent.main import create_app
from agent.models_api import FakeModelClient, ModelResponse, TokenUsage, ToolCall
from agent.process_runner import ProcessResult, ProcessRunner
from agent.pytest_runner import PytestRunner
from agent.repository_access import RepositoryAccess
from agent.static_analysis import StaticAnalysisRunner

pytestmark = pytest.mark.skipif(
    os.environ.get("CODEFORGE_DOCKER_TESTS") != "1",
    reason="Set CODEFORGE_DOCKER_TESTS=1 after building codeforge-agent:local",
)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    path = tmp_path / "repo, with space"
    path.mkdir(mode=0o755)
    return path


class ObservingProcess(ProcessRunner):
    def __init__(self) -> None:
        self.name = ""
        self.starting = asyncio.Event()

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
            self.name = command[command.index("--name") + 1]
        if command[1] == "start":
            self.starting.set()
        return await super().run(
            command,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )


def assert_removed(name: str) -> None:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=" + name],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not result.stdout.strip()


def test_real_filesystem_network_identity_resources_and_environment_isolation(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEFORGE_OPENAI_API_KEY", "private-value")
    outside = tmp_path / "host-secret.txt"
    outside.write_text("private-host-content")
    (repository / "escape").symlink_to(outside)
    source = """import os
import socket
from pathlib import Path
import pytest

def test_isolation():
    assert os.geteuid() == 65534
    assert "CODEFORGE_OPENAI_API_KEY" not in os.environ
    assert not Path("/var/run/docker.sock").exists()
    assert not Path("/repo/escape").exists()
    assert set(os.listdir("/sys/class/net")) == {"lo"}
    with socket.socket() as connection:
        connection.settimeout(0.2)
        with pytest.raises(OSError):
            connection.connect(("1.1.1.1", 53))
    for path in ("/repo/forbidden", "/etc/forbidden"):
        with pytest.raises(OSError):
            Path(path).write_text("must not write")
    Path("/tmp/allowed").write_text("temporary")
    status = Path("/proc/self/status").read_text()
    assert "CapEff:\\t0000000000000000" in status
    assert "NoNewPrivs:\\t1" in status
    assert int(Path("/sys/fs/cgroup/memory.max").read_text()) == 256 * 1024**2
    assert int(Path("/sys/fs/cgroup/pids.max").read_text()) == 32
    quota, period = map(int, Path("/sys/fs/cgroup/cpu.max").read_text().split())
    assert quota / period == 0.5
"""
    (repository / "test_isolation.py").write_text(source)
    observer = ObservingProcess()
    sandbox = DockerCommandSandbox(
        tmp_path, cpus=0.5, memory_mb=256, pids_limit=32, tmpfs_mb=32, process_runner=observer
    )
    result = asyncio.run(
        PytestRunner(RepositoryAccess(tmp_path), enabled=True, sandbox=sandbox).run(repository, [])
    )
    assert result.verified, result.process
    assert (repository / "test_isolation.py").read_text() == source
    assert not (repository / "forbidden").exists()
    assert_removed(observer.name)


@pytest.mark.parametrize("tool", ["ruff", "mypy"])
def test_real_analysis_preserves_sources_and_normalizes_paths(
    tmp_path: Path, repository: Path, tool: str
) -> None:
    source = 'import os\nx: int = "bad"\n'
    (repository / "app.py").write_text(source)
    (repository / "mypy.ini").write_text("[mypy]\nfiles = app.py\n")
    (repository / "ruff.toml").write_text('fix = true\n[lint]\nselect = ["F"]\n')
    sandbox = DockerCommandSandbox(tmp_path)
    runner = StaticAnalysisRunner(RepositoryAccess(tmp_path), enabled=True, sandbox=sandbox)
    result = asyncio.run(runner.run(repository, tool, []))  # type: ignore[arg-type]
    assert result.status == "findings" and result.diagnostics_complete, result.process
    assert result.diagnostics[0].file == "app.py"
    assert (repository / "app.py").read_text() == source


def test_real_timeout_removes_container_and_children(tmp_path: Path, repository: Path) -> None:
    (repository / "test_slow.py").write_text(
        "import subprocess, sys, time\ndef test_slow():\n"
        '    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n'
        '    print("started", flush=True)\n    time.sleep(30)\n'
    )
    observer = ObservingProcess()
    runner = PytestRunner(
        RepositoryAccess(tmp_path),
        enabled=True,
        timeout_seconds=3,
        sandbox=DockerCommandSandbox(tmp_path, process_runner=observer),
    )
    result = asyncio.run(runner.run(repository, []))
    assert result.status == "timeout" and not result.verified
    assert_removed(observer.name)


def test_real_cancellation_removes_running_container(tmp_path: Path, repository: Path) -> None:
    (repository / "test_slow.py").write_text("import time\ndef test_slow():\n    time.sleep(30)\n")

    async def exercise() -> None:
        observer = ObservingProcess()
        runner = PytestRunner(
            RepositoryAccess(tmp_path),
            enabled=True,
            sandbox=DockerCommandSandbox(tmp_path, process_runner=observer),
        )
        task = asyncio.create_task(runner.run(repository, []))
        await asyncio.wait_for(observer.starting.wait(), timeout=5)
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert_removed(observer.name)

    asyncio.run(exercise())


def test_real_missing_image_fails_without_executing_locally(
    tmp_path: Path, repository: Path
) -> None:
    (repository / "test_marker.py").write_text(
        'from pathlib import Path\ndef test_marker():\n    Path("executed-locally").touch()\n'
    )
    sandbox = DockerCommandSandbox(tmp_path, image="codeforge-agent:nonexistent-test-image")
    result = asyncio.run(
        PytestRunner(RepositoryAccess(tmp_path), enabled=True, sandbox=sandbox).run(repository, [])
    )
    assert result.status == "execution_error" and not result.verified
    assert not (repository / "executed-locally").exists()


def test_real_bug_fix_run_automatically_verifies_in_docker(
    tmp_path: Path, repository: Path
) -> None:
    (repository / "app.py").write_text("value = 0\n")
    (repository / "test_app.py").write_text(
        "from app import value\ndef test_value():\n    assert value == 1\n"
    )
    model = FakeModelClient(
        [
            ModelResponse(
                "repair",
                "fake",
                "test",
                (
                    ToolCall(
                        "patch",
                        "apply_patch",
                        {
                            "patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n"
                            "-value = 0\n+value = 1\n",
                        },
                    ),
                ),
                "tool_use",
                TokenUsage(1, 1),
            )
        ]
    )
    settings = Settings(  # type: ignore[call-arg]
        database_url="sqlite://",
        workspace_root=tmp_path,
        code_writes_enabled=True,
        test_execution_enabled=True,
        command_execution_backend="docker",
        _env_file=None,
    )
    with TestClient(create_app(settings, model_client=model)) as client:
        created = client.post("/repositories", json={"name": "Docker", "path": str(repository)})
        url = f"/repositories/{created.json()['id']}/runs/bug-fix"
        result = client.post(url, json={"task": "Fix value"})
    assert result.status_code == 201 and result.json()["verified"] is True, result.text
    assert result.json()["test_results"][0]["command"][0] == "/opt/codeforge/.venv/bin/python"
