"""Subprocess contracts: literal arguments, capture limits, and termination."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from agent.process_runner import ProcessRunner


def environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR"}
    }


def test_argument_arrays_unicode_stderr_and_exit_code(tmp_path: Path) -> None:
    literal = "$(whoami); echo unexpected"
    result = asyncio.run(
        ProcessRunner().run(
            (
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); print('你好'); "
                "print('error', file=sys.stderr); sys.exit(7)",
                literal,
            ),
            cwd=tmp_path,
            environment=environment(),
        )
    )
    assert result.exit_code == 7
    assert result.stdout == literal + "\n你好\n"
    assert result.stderr == "error\n"
    assert result.timed_out is False and result.error is None
    assert result.duration_ms >= 0


def test_output_limit_keeps_head_and_tail_without_killing_process(tmp_path: Path) -> None:
    result = asyncio.run(
        ProcessRunner().run(
            (
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'START' + b'\\xff' * 100000 + b'END'); "
                "sys.stderr.write('e' * 100000 + 'TAIL')",
            ),
            cwd=tmp_path,
            environment=environment(),
            max_output_bytes=512,
        )
    )
    assert result.exit_code == 0
    assert result.stdout.startswith("START") and result.stdout.endswith("END")
    assert result.stderr.endswith("TAIL")
    assert result.stdout_truncated is True and result.stderr_truncated is True
    assert len(result.stdout.encode("utf-8")) <= 512
    assert len(result.stderr.encode("utf-8")) <= 512


def test_timeout_retains_partial_output_and_reaps_process(tmp_path: Path) -> None:
    result = asyncio.run(
        ProcessRunner().run(
            (sys.executable, "-u", "-c", "import time; print('started'); time.sleep(30)"),
            cwd=tmp_path,
            environment=environment(),
            timeout_seconds=0.3,
        )
    )
    assert result.timed_out is True
    assert result.exit_code is not None and result.exit_code != 0
    assert "started" in result.stdout
    assert result.duration_ms < 3000


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group termination")
def test_timeout_terminates_descendants_holding_output_pipes(tmp_path: Path) -> None:
    script = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print(child.pid, flush=True); time.sleep(30)"
    )
    result = asyncio.run(
        ProcessRunner().run(
            (sys.executable, "-u", "-c", script),
            cwd=tmp_path,
            environment=environment(),
            timeout_seconds=0.3,
        )
    )
    assert result.timed_out is True
    child = int(result.stdout.strip())
    status = Path(f"/proc/{child}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_cancelled_subprocess_is_reaped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processes: list[asyncio.subprocess.Process] = []
    original = asyncio.create_subprocess_exec

    async def exercise() -> None:
        entered = asyncio.Event()

        async def launch(*arguments: str, **kwargs: Any) -> asyncio.subprocess.Process:
            process = await original(*arguments, **kwargs)
            processes.append(process)
            entered.set()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
        task = asyncio.create_task(
            ProcessRunner().run(
                (sys.executable, "-c", "import time; time.sleep(30)"),
                cwd=tmp_path,
                environment=environment(),
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert processes[0].returncode is not None


def test_missing_executable_is_a_structured_failure(tmp_path: Path) -> None:
    result = asyncio.run(
        ProcessRunner().run(
            (str(tmp_path / "missing"),),
            cwd=tmp_path,
            environment=environment(),
        )
    )
    assert result.exit_code is None
    assert result.error == "Unable to start configured executable"
