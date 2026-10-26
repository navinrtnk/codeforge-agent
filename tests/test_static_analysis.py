"""Project detection, real tool execution, and conservative diagnostic normalization."""

import asyncio
import json
from pathlib import Path

import pytest

from agent.process_runner import ProcessResult, ProcessRunner
from agent.project_tools import AnalysisError, AnalysisTool, detect_configuration
from agent.repository_access import RepositoryAccess
from agent.static_analysis import StaticAnalysisRunner, normalize_diagnostics


def runner(
    tmp_path: Path,
    *,
    max_diagnostics: int = 200,
    max_output_bytes: int = 20_000,
    process_runner: ProcessRunner | None = None,
) -> StaticAnalysisRunner:
    return StaticAnalysisRunner(
        RepositoryAccess(tmp_path),
        enabled=True,
        max_diagnostics=max_diagnostics,
        max_output_bytes=max_output_bytes,
        process_runner=process_runner,
    )


@pytest.mark.parametrize(
    ("tool", "name", "content"),
    [
        ("ruff", ".ruff.toml", "line-length = 88\n"),
        ("ruff", "ruff.toml", "line-length = 88\n"),
        ("ruff", "pyproject.toml", "[tool.ruff]\nline-length = 88\n"),
        ("mypy", "mypy.ini", "[mypy]\nfiles = src\n"),
        ("mypy", ".mypy.ini", "[mypy]\nfiles = src\n"),
        ("mypy", "pyproject.toml", '[tool.mypy]\nfiles = ["src"]\n'),
        ("mypy", "setup.cfg", "[mypy]\nfiles = src\n"),
    ],
)
def test_configuration_formats(tmp_path: Path, tool: AnalysisTool, name: str, content: str) -> None:
    (tmp_path / name).write_text(content)
    config = detect_configuration(RepositoryAccess(tmp_path), tmp_path, tool)
    assert config.file == name
    assert config.has_files is (tool == "mypy")


def test_configuration_precedence_and_irrelevant_sections(tmp_path: Path) -> None:
    access = RepositoryAccess(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "sample"\n')
    assert detect_configuration(access, tmp_path, "ruff").file is None
    (tmp_path / "ruff.toml").write_text("line-length = 88\n")
    (tmp_path / ".ruff.toml").write_text("line-length = 100\n")
    assert detect_configuration(access, tmp_path, "ruff").file == ".ruff.toml"
    (tmp_path / "setup.cfg").write_text("[mypy]\n")
    (tmp_path / "mypy.ini").write_text("[mypy]\n")
    assert detect_configuration(access, tmp_path, "mypy").file == "mypy.ini"


@pytest.mark.parametrize("content", ['extend = "../ruff.toml"', "[malformed"])
def test_invalid_or_extended_config_is_rejected(tmp_path: Path, content: str) -> None:
    (tmp_path / "ruff.toml").write_text(content)
    with pytest.raises(AnalysisError):
        detect_configuration(RepositoryAccess(tmp_path), tmp_path, "ruff")


def test_real_ruff_never_fixes_and_uses_project_rules(tmp_path: Path) -> None:
    source = "import os\n"
    (tmp_path / "sample.py").write_text(source)
    (tmp_path / "ruff.toml").write_text('fix = true\nfix-only = true\n[lint]\nselect = ["F"]\n')
    result = asyncio.run(runner(tmp_path).run(tmp_path, "ruff", ["sample.py"]))
    assert result.status == "findings", result.process
    assert result.diagnostics_complete and not result.clean
    diagnostic = result.diagnostics[0]
    assert (diagnostic.file, diagnostic.line, diagnostic.column, diagnostic.code) == (
        "sample.py",
        1,
        8,
        "F401",
    )
    assert (tmp_path / "sample.py").read_text() == source
    assert not (tmp_path / ".ruff_cache").exists()
    (tmp_path / "ruff.toml").write_text('[lint]\nignore = ["F401"]\n')
    assert asyncio.run(runner(tmp_path).run(tmp_path, "ruff", [])).clean


def test_real_mypy_configured_files_targeting_and_positions(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("x: int = 1\n")
    (tmp_path / "bad.py").write_text('x: int = "bad"\n')
    (tmp_path / "mypy.ini").write_text("[mypy]\nfiles = good.py\n")
    service = runner(tmp_path)
    configured = asyncio.run(service.run(tmp_path, "mypy", []))
    assert configured.clean, configured.process
    targeted = asyncio.run(service.run(tmp_path, "mypy", ["bad.py"]))
    assert targeted.status == "findings", targeted.process
    diagnostic = targeted.diagnostics[0]
    assert (diagnostic.file, diagnostic.line, diagnostic.column, diagnostic.code) == (
        "bad.py",
        1,
        10,
        "assignment",
    )
    assert diagnostic.end_column == 15
    assert targeted.diagnostics_complete and not targeted.clean
    assert not (tmp_path / ".mypy_cache").exists()


@pytest.mark.parametrize("tool", ["ruff", "mypy"])
def test_no_parent_configuration_and_clean_project(tmp_path: Path, tool: AnalysisTool) -> None:
    (tmp_path / "ruff.toml").write_text("invalid = true\n")
    (tmp_path / "mypy.ini").write_text("[mypy]\nplugins = nonexistent_plugin\n")
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "sample.py").write_text("x: int = 1\n")
    result = asyncio.run(runner(tmp_path).run(repository, tool, []))
    assert result.clean and result.config_file is None, result.process


@pytest.mark.parametrize(
    "target", ["../out.py", "/tmp/out.py", "C:/out.py", ".git/config", "a\\b.py", "missing.py"]
)
def test_target_validation(tmp_path: Path, target: str) -> None:
    with pytest.raises(AnalysisError):
        asyncio.run(runner(tmp_path).run(tmp_path, "ruff", [target]))


def test_symlinks_and_config_reads_cannot_escape(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (tmp_path / "out.py").write_text("x = 1\n")
    (repository / "link.py").symlink_to(tmp_path / "out.py")
    with pytest.raises(AnalysisError):
        asyncio.run(runner(tmp_path).run(repository, "mypy", ["link.py"]))
    (tmp_path / "ruff.toml").write_text("line-length = 100\n")
    (repository / "ruff.toml").symlink_to(tmp_path / "ruff.toml")
    with pytest.raises(AnalysisError):
        asyncio.run(runner(tmp_path).run(repository, "ruff", []))
    with pytest.raises(AnalysisError, match="disabled"):
        asyncio.run(StaticAnalysisRunner(RepositoryAccess(tmp_path)).run(repository, "ruff", []))


def mypy_record(file: str = "sample.py") -> dict[str, object]:
    return {
        "file": file,
        "line": 2,
        "column": 0,
        "end_line": 2,
        "end_column": 4,
        "code": "assignment",
        "message": "wrong type",
        "severity": "error",
    }


def test_normalization_handles_paths_notes_bad_records_and_caps(tmp_path: Path) -> None:
    record = mypy_record(str(tmp_path / "space name.py"))
    note = {**mypy_record(), "severity": "note", "code": None}
    diagnostics, complete, truncated = normalize_diagnostics(
        "\n".join(json.dumps(item) for item in [record, note]), "mypy", tmp_path
    )
    assert complete and not truncated
    assert diagnostics[0].file == "space name.py" and diagnostics[0].column == 1
    assert diagnostics[0].end_column == 5 and diagnostics[1].severity == "note"
    output = "\n".join(
        [
            json.dumps(record),
            "not-json",
            "[]",
            "null",
            "{}",
            json.dumps(mypy_record("../outside.py")),
            json.dumps({**record, "line": True}),
            json.dumps(note),
        ]
    )
    diagnostics, complete, truncated = normalize_diagnostics(output, "mypy", tmp_path, limit=1)
    assert len(diagnostics) == 1 and not complete and truncated


class CapturingProcess(ProcessRunner):
    def __init__(self, result: ProcessResult) -> None:
        self.result = result
        self.environment: dict[str, str] = {}
        self.command: tuple[str, ...] = ()

    async def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float = 60,
        max_output_bytes: int = 20_000,
    ) -> ProcessResult:
        self.environment = environment
        self.command = command
        return self.result


@pytest.mark.parametrize(
    ("exit_code", "output", "truncated", "timed_out", "status"),
    [
        (0, "", False, False, "passed"),
        (0, "bad-json", False, False, "incomplete"),
        (0, "", True, False, "incomplete"),
        (1, "", False, False, "execution_error"),
        (2, "", False, False, "execution_error"),
        (None, "", False, True, "timeout"),
    ],
)
def test_conservative_outcomes_and_scrubbed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int | None,
    output: str,
    truncated: bool,
    timed_out: bool,
    status: str,
) -> None:
    monkeypatch.setenv("CODEFORGE_OPENAI_API_KEY", "private")
    monkeypatch.setenv("MYPYPATH", "/outside")
    monkeypatch.setenv("RUFF_OUTPUT_FORMAT", "text")
    process = CapturingProcess(
        ProcessResult((), exit_code, output, "", 1, timed_out=timed_out, stdout_truncated=truncated)
    )
    result = asyncio.run(runner(tmp_path, process_runner=process).run(tmp_path, "mypy", []))
    assert result.status == status
    assert result.clean is (status == "passed")
    assert not (
        set(process.environment) & {"CODEFORGE_OPENAI_API_KEY", "MYPYPATH", "RUFF_OUTPUT_FORMAT"}
    )
    assert "--no-install-types" in process.command


def test_real_output_and_diagnostic_limits(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("\n".join(f"import missing_{n}" for n in range(40)) + "\n")
    limited = asyncio.run(runner(tmp_path, max_diagnostics=1).run(tmp_path, "ruff", []))
    assert len(limited.diagnostics) == 1 and limited.diagnostics_truncated
    assert not limited.diagnostics_complete and not limited.clean
    bounded = asyncio.run(runner(tmp_path, max_output_bytes=1024).run(tmp_path, "ruff", []))
    assert bounded.process.stdout_truncated
    assert not bounded.diagnostics_complete and not bounded.clean
    assert len(bounded.process.stdout.encode()) <= 1024


def test_notes_with_unknown_end_positions_are_valid(tmp_path: Path) -> None:
    record = {**mypy_record(), "severity": "note", "end_line": -1, "end_column": -1}
    process = CapturingProcess(ProcessResult((), 0, json.dumps(record), "", 1))
    result = asyncio.run(runner(tmp_path, process_runner=process).run(tmp_path, "mypy", []))
    assert result.clean and result.diagnostics_complete
    assert result.diagnostics[0].end_line is None


@pytest.mark.parametrize("tool", ["ruff", "mypy"])
def test_missing_executable_and_config_errors(tmp_path: Path, tool: AnalysisTool) -> None:
    missing = StaticAnalysisRunner(
        RepositoryAccess(tmp_path),
        enabled=True,
        python_executable=tmp_path / "missing-python",
    )
    assert asyncio.run(missing.run(tmp_path, tool, [])).status == "execution_error"
    (tmp_path / "sample.py").write_text("x = 1\n")
    config, text = (
        ("ruff.toml", 'target-version = "invalid"\n')
        if tool == "ruff"
        else ("mypy.ini", "[mypy]\nplugins = nonexistent_plugin\n")
    )
    (tmp_path / config).write_text(text)
    result = asyncio.run(runner(tmp_path).run(tmp_path, tool, []))
    assert result.status == "execution_error" and not result.clean
    assert not result.diagnostics_complete
