"""Detect pytest projects and execute full or path-targeted test runs."""

import configparser
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from agent.process_runner import ProcessResult, ProcessRunner
from agent.repository_access import RepositoryAccess, RepositoryAccessError

COUNTS = ("passed", "failed", "skipped", "deselected", "xfailed", "xpassed", "errors", "warnings")
COUNT = r"\d+ (?:passed|failed|skipped|deselected|xfailed|xpassed|errors?|warnings?)"
SUMMARY = re.compile(rf"^(?:{COUNT})(?:, {COUNT})* in \d+(?:\.\d+)?s(?: \([^\n]+\))?$")


class PytestRunnerError(Exception):
    """An invalid repository, configuration, or requested test path."""


@dataclass(frozen=True)
class PytestResult:
    """Structured execution outcome; passing an empty suite is not verification."""

    process: ProcessResult
    targets: tuple[str, ...]
    config_file: str | None
    status: str
    counts: dict[str, int]
    summary_complete: bool
    verified: bool


def parse_summary(output: str) -> tuple[dict[str, int], bool]:
    """Read the final complete pytest totals line, including a captured output tail."""
    counts = dict.fromkeys(COUNTS, 0)
    for line in reversed(output.splitlines()):
        line = line.strip("= ")
        if SUMMARY.fullmatch(line):
            for number, category in re.findall(r"(\d+) (\w+)", line.split(" in ", 1)[0]):
                key = {"error": "errors", "warning": "warnings"}.get(category, category)
                counts[key] = int(number)
            return counts, True
    return counts, False


class PytestRunner:
    """Execute trusted local project tests using an operator-selected interpreter."""

    def __init__(
        self,
        access: RepositoryAccess,
        *,
        enabled: bool = False,
        python_executable: Path | None = None,
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
        process_runner: ProcessRunner | None = None,
    ) -> None:
        self.access = access
        self.enabled = enabled
        # Keep the venv path: resolving its symlink would select the base interpreter.
        self.python = (python_executable or Path(sys.executable)).expanduser().absolute()
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.process_runner = process_runner or ProcessRunner()

    def detect_configuration(self, repository: Path) -> str | None:
        """Inspect only local configuration files; never import project code."""
        try:
            for name in ("pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"):
                candidate = repository / name
                if not candidate.exists():
                    continue
                content = self.access.read_text(repository, Path(name)).content
                if name in {"pytest.ini", ".pytest.ini"}:
                    parser = configparser.ConfigParser(interpolation=None)
                    parser.read_string(content)
                    return name
                if name == "pyproject.toml":
                    config = tomllib.loads(content)
                    tool = config.get("tool", {})
                    pytest_config = tool.get("pytest", {}) if isinstance(tool, dict) else {}
                    if isinstance(pytest_config, dict) and isinstance(
                        pytest_config.get("ini_options"), dict
                    ):
                        return name
                else:
                    parser = configparser.ConfigParser(interpolation=None)
                    parser.read_string(content)
                    section = "pytest" if name == "tox.ini" else "tool:pytest"
                    if parser.has_section(section):
                        return name
        except (RepositoryAccessError, configparser.Error, tomllib.TOMLDecodeError) as error:
            raise PytestRunnerError("Unable to read pytest configuration") from error
        return None

    def _targets(self, repository: Path, paths: list[str]) -> tuple[str, ...]:
        if len(paths) > 50:
            raise PytestRunnerError("At most 50 test paths may be requested")
        validated = []
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
                raise PytestRunnerError("Test paths must stay inside the repository")
            candidate = (repository / value).resolve()
            if not candidate.is_relative_to(repository):
                raise PytestRunnerError("Test paths must stay inside the repository")
            if not candidate.is_dir() and (not candidate.is_file() or candidate.suffix != ".py"):
                raise PytestRunnerError("Test targets must be existing Python files or directories")
            validated.append(candidate.relative_to(repository).as_posix())
        return tuple(validated)

    async def run(self, path: Path, paths: list[str]) -> PytestResult:
        """Run pytest with fixed options and no inherited model credentials or pytest flags."""
        if not self.enabled:
            raise PytestRunnerError("Local test execution is disabled")
        try:
            repository = self.access.resolve_repository(path)
            targets = self._targets(repository, paths)
            config = self.detect_configuration(repository)
        except RepositoryAccessError as error:
            raise PytestRunnerError(str(error)) from error
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "TZ"}
        environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        environment.update(
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"
        )
        arguments = [str(self.python), "-m", "pytest"]
        # An explicit root/config prevents pytest discovering parent-project configuration.
        arguments.extend(
            ["--rootdir", str(repository), "-c", str(repository / config) if config else os.devnull]
        )
        arguments.extend(
            [
                "--confcutdir",
                str(repository),
                "-q",
                "--color=no",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                "--",
                *targets,
            ]
        )
        process = await self.process_runner.run(
            tuple(arguments),
            cwd=repository,
            environment=environment,
            timeout_seconds=self.timeout_seconds,
            max_output_bytes=self.max_output_bytes,
        )
        counts, complete = parse_summary(process.stdout)
        status = {
            0: "passed",
            1: "failed",
            2: "collection_error" if counts["errors"] else "interrupted",
            3: "internal_error",
            4: "usage_error",
            5: "no_tests",
        }.get(process.exit_code if process.exit_code is not None else -1, "execution_error")
        if process.error or "No module named pytest" in process.stderr:
            status = "execution_error"
        if process.timed_out:
            status = "timeout"
        verified = (
            status == "passed"
            and complete
            and counts["passed"] > 0
            and not (counts["failed"] or counts["errors"])
        )
        return PytestResult(process, targets, config, status, counts, complete, verified)
