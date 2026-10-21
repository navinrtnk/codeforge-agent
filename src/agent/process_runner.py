"""Bounded asynchronous subprocesses for local development command runners."""

import asyncio
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

TRUNCATION_MARKER = b"\n... [output truncated] ...\n"


@dataclass(frozen=True)
class ProcessResult:
    """Command completion information, with bounded head/tail output excerpts."""

    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    error: str | None = None


class _Capture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.capacity = limit - len(TRUNCATION_MARKER)
        self.head_limit = self.capacity // 2
        self.tail_limit = self.capacity - self.head_limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def append(self, data: bytes) -> None:
        self.total += len(data)
        remaining = max(0, self.head_limit - len(self.head))
        self.head.extend(data[:remaining])
        self.tail.extend(data[remaining:])
        if len(self.tail) > self.tail_limit:
            del self.tail[: -self.tail_limit]

    @property
    def truncated(self) -> bool:
        return (
            self.total > self.capacity
            or len((self.head + self.tail).decode("utf-8", errors="replace").encode("utf-8"))
            > self.limit
        )

    def text(self) -> str:
        def decode(data: bytearray, limit: int, *, keep_tail: bool = False) -> str:
            encoded = data.decode("utf-8", errors="replace").encode("utf-8")
            excerpt = encoded[-limit:] if keep_tail else encoded[:limit]
            return excerpt.decode("utf-8", errors="ignore")

        if not self.truncated:
            return decode(self.head + self.tail, self.limit)
        head = decode(self.head, self.head_limit)
        tail = decode(self.tail, self.tail_limit, keep_tail=True)
        marker = TRUNCATION_MARKER.decode() if self.truncated else ""
        return head + marker + tail


class ProcessRunner:
    """Run argument arrays without a shell, draining output beyond the capture limit."""

    async def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
    ) -> ProcessResult:
        if not command or timeout_seconds <= 0 or max_output_bytes < 256:
            raise ValueError("Invalid subprocess command or limits")
        started = perf_counter()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except OSError:
            return ProcessResult(
                command,
                None,
                "",
                "",
                (perf_counter() - started) * 1000,
                error="Unable to start configured executable",
            )
        stdout, stderr = _Capture(max_output_bytes), _Capture(max_output_bytes)

        async def drain(stream: asyncio.StreamReader | None, capture: _Capture) -> None:
            assert stream is not None
            while chunk := await stream.read(8192):
                capture.append(chunk)

        completion = asyncio.gather(
            drain(process.stdout, stdout),
            drain(process.stderr, stderr),
            process.wait(),
        )
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(completion), timeout=timeout_seconds)
        except TimeoutError:
            timed_out = True
            self._kill(process)
            await completion
        except BaseException:
            self._kill(process)
            await completion
            raise
        finally:
            # POSIX process groups include children that have closed their pipes.
            self._kill(process)
        return ProcessResult(
            command,
            process.returncode,
            stdout.text(),
            stderr.text(),
            (perf_counter() - started) * 1000,
            timed_out,
            stdout.truncated,
            stderr.truncated,
        )

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:
                process.kill()
        except ProcessLookupError:
            pass
