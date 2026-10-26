"""Bounded Ruff/mypy execution with tool-independent, repository-relative diagnostics."""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from agent.process_runner import ProcessResult, ProcessRunner
from agent.project_tools import AnalysisError, AnalysisTool, detect_configuration
from agent.repository_access import RepositoryAccess, RepositoryAccessError


@dataclass(frozen=True)
class Diagnostic:
    """Positions are one-based; optional end positions are exclusive."""

    file: str
    line: int
    column: int
    end_line: int | None
    end_column: int | None
    code: str | None
    message: str
    severity: str


@dataclass(frozen=True)
class AnalysisResult:
    """Static analysis evidence is not proof of correct runtime behavior."""

    process: ProcessResult
    tool: AnalysisTool
    targets: tuple[str, ...]
    config_file: str | None
    status: str
    diagnostics: tuple[Diagnostic, ...]
    diagnostics_complete: bool
    diagnostics_truncated: bool
    clean: bool


def _position(value: Any, offset: int = 0) -> int:
    if type(value) is not int or value + offset < 1:
        raise ValueError("Invalid position")
    return int(value) + offset


def _optional_position(value: Any, offset: int = 0) -> int | None:
    # mypy uses -1 when an end location is unavailable.
    if value is None or (type(value) is int and value == -1):
        return None
    return _position(value, offset)


def normalize_diagnostics(
    output: str, tool: AnalysisTool, repository: Path, *, limit: int = 200
) -> tuple[tuple[Diagnostic, ...], bool, bool]:
    """Parse JSON lines, retaining valid records without trusting paths or malformed output."""
    diagnostics: list[Diagnostic] = []
    complete, truncated = True, False
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            filename = record["filename" if tool == "ruff" else "file"]
            if not isinstance(filename, str) or not filename or "\0" in filename:
                raise ValueError("Invalid diagnostic path")
            path = (repository / filename).resolve()
            if not path.is_relative_to(repository) or ".git" in path.relative_to(repository).parts:
                raise ValueError("Diagnostic outside repository")
            start = record["location"] if tool == "ruff" else record
            end = record.get("end_location") if tool == "ruff" else record
            row_key = "row" if tool == "ruff" else "line"
            offset = 0 if tool == "ruff" else 1
            end_line = end.get("row" if tool == "ruff" else "end_line") if end else None
            end_column = end.get("column" if tool == "ruff" else "end_column") if end else None
            message = record["message"]
            code = record.get("code")
            severity = "error" if tool == "ruff" else record["severity"]
            if (
                not isinstance(message, str)
                or (code is not None and not isinstance(code, str))
                or severity not in {"error", "warning", "note"}
            ):
                raise ValueError("Invalid diagnostic")
            diagnostic = Diagnostic(
                path.relative_to(repository).as_posix(),
                _position(start[row_key]),
                _position(start["column"], offset),
                _optional_position(end_line),
                _optional_position(end_column, offset),
                code,
                message,
                severity,
            )
            if len(diagnostics) < limit:
                diagnostics.append(diagnostic)
            else:
                truncated = True
        except ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError:
            complete = False
    return tuple(diagnostics), complete and not truncated, truncated


class StaticAnalysisRunner:
    """Run trusted local configurations with fixed argv, bounded output, and no shell."""

    def __init__(
        self,
        access: RepositoryAccess,
        *,
        enabled: bool = False,
        python_executable: Path | None = None,
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
        max_diagnostics: int = 200,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self.access = access
        self.enabled = enabled
        self.python = (python_executable or Path(sys.executable)).expanduser().absolute()
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        if max_diagnostics < 1:
            raise ValueError("Diagnostic limit must be positive")
        self.max_diagnostics = max_diagnostics
        self.process_runner = process_runner or ProcessRunner()

    def _targets(self, repository: Path, paths: list[str]) -> tuple[str, ...]:
        if len(paths) > 50:
            raise AnalysisError("At most 50 analysis paths may be requested")
        targets = []
        for value in paths:
            relative = PurePosixPath(value)
            if (
                not value
                or "\\" in value
                or "\0" in value
                or relative.is_absolute()
                or PureWindowsPath(value).drive
                or ".." in relative.parts
                or any(part.lower() == ".git" for part in relative.parts)
            ):
                raise AnalysisError("Analysis paths must stay inside the repository")
            candidate = (repository / value).resolve()
            if not candidate.is_relative_to(repository):
                raise AnalysisError("Analysis paths must stay inside the repository")
            if not candidate.is_dir() and (
                not candidate.is_file() or candidate.suffix not in {".py", ".pyi"}
            ):
                raise AnalysisError("Analysis targets must be existing Python files or directories")
            # Prefix with ./ so even a filename starting with '-' is unambiguous.
            targets.append("./" + candidate.relative_to(repository).as_posix())
        return tuple(targets)

    async def run(self, path: Path, tool: AnalysisTool, paths: list[str]) -> AnalysisResult:
        """Respect root project settings but never enable fixes, installs, or inherited flags."""
        if not self.enabled:
            raise AnalysisError("Local static analysis is disabled")
        if tool not in {"ruff", "mypy"}:
            raise AnalysisError("Unsupported analysis tool")
        try:
            repository = self.access.resolve_repository(path)
            targets = self._targets(repository, paths)
            config = detect_configuration(self.access, repository, tool)
        except RepositoryAccessError as error:
            raise AnalysisError(str(error)) from error
        command = [str(self.python), "-m", tool]
        if tool == "ruff":
            command += ["check", "--output-format", "json-lines", "--no-fix", "--no-fix-only"]
            command += ["--no-cache"]
            command += (
                ["--config", str(repository / config.file)] if config.file else ["--isolated"]
            )
            command += ["--", *(targets or (".",))]
        else:
            command += [
                "--config-file",
                str(repository / config.file) if config.file else os.devnull,
                "--output",
                "json",
                "--no-incremental",
                "--cache-dir",
                os.devnull,
                "--no-install-types",
                "--no-pretty",
                "--no-color-output",
                "--no-error-summary",
                "--show-column-numbers",
                "--show-error-codes",
                "--junit-xml",
                os.devnull,
            ]
            command += ["--", *targets] if targets else ([] if config.has_files else ["--", "."])
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "TZ"}
        environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", NO_COLOR="1")
        process = await self.process_runner.run(
            tuple(command),
            cwd=repository,
            environment=environment,
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        diagnostics, complete, truncated = normalize_diagnostics(
            process.stdout, tool, repository, limit=self.max_diagnostics
        )
        complete = complete and not (process.stdout_truncated or process.stderr_truncated)
        status = {0: "passed", 1: "findings"}.get(
            process.exit_code if process.exit_code is not None else -1, "execution_error"
        )
        if process.error:
            status = "execution_error"
        if process.timed_out:
            status = "timeout"
        # Nonzero exit without parseable findings cannot be presented as a normal check.
        if status == "findings" and not diagnostics:
            status = "execution_error" if complete else "incomplete"
        has_errors = any(diagnostic.severity != "note" for diagnostic in diagnostics)
        if status == "passed" and (not complete or has_errors):
            status = "incomplete"
        complete = complete and status in {"passed", "findings"}
        return AnalysisResult(
            process,
            tool,
            targets,
            config.file,
            status,
            diagnostics,
            complete,
            truncated or process.stdout_truncated,
            status == "passed" and complete,
        )
