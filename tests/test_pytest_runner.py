"""Run real pytest projects and distinguish verification from incomplete execution."""

import asyncio
from pathlib import Path

import pytest

from agent.pytest_runner import PytestRunner, PytestRunnerError, parse_summary
from agent.repository_access import RepositoryAccess


def runner(tmp_path: Path, *, timeout: float = 10, output_limit: int = 20_000) -> PytestRunner:
    return PytestRunner(
        RepositoryAccess(tmp_path),
        enabled=True,
        timeout_seconds=timeout,
        max_output_bytes=output_limit,
    )


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("pytest.ini", "[pytest]\ntestpaths = tests\n"),
        (".pytest.ini", "[pytest]\n"),
        ("pyproject.toml", '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'),
        ("tox.ini", "[pytest]\ntestpaths = tests\n"),
        ("setup.cfg", "[tool:pytest]\ntestpaths = tests\n"),
    ],
)
def test_configuration_detection(tmp_path: Path, name: str, content: str) -> None:
    (tmp_path / name).write_text(content, encoding="utf-8")
    assert runner(tmp_path).detect_configuration(tmp_path) == name


def test_configured_suite_and_targeted_paths(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n', encoding="utf-8"
    )
    (tmp_path / "tests/test_pass.py").write_text(
        "def test_pass():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "other/test_fail.py").write_text(
        "def test_fail():\n    assert False\n", encoding="utf-8"
    )
    client = runner(tmp_path)
    full = asyncio.run(client.run(tmp_path, []))
    assert full.status == "passed" and full.verified is True
    assert full.counts["passed"] == 1 and full.config_file == "pyproject.toml"
    targeted = asyncio.run(client.run(tmp_path, ["other/test_fail.py"]))
    assert targeted.status == "failed" and targeted.verified is False
    assert targeted.counts["failed"] == 1
    assert targeted.targets == ("other/test_fail.py",)


@pytest.mark.parametrize(
    ("source", "status", "field"),
    [
        ("def test_fail():\n    assert False\n", "failed", "failed"),
        ("def broken(:\n", "collection_error", "errors"),
        ("import pytest\n@pytest.mark.skip\ndef test_skip():\n    pass\n", "passed", "skipped"),
    ],
)
def test_failure_collection_and_skipped_only_are_not_verification(
    tmp_path: Path,
    source: str,
    status: str,
    field: str,
) -> None:
    (tmp_path / "test_case.py").write_text(source, encoding="utf-8")
    result = asyncio.run(runner(tmp_path).run(tmp_path, []))
    assert result.status == status
    assert result.counts[field] == 1
    assert result.summary_complete is True and result.verified is False


def test_empty_suite_and_usage_error(tmp_path: Path) -> None:
    assert asyncio.run(runner(tmp_path).run(tmp_path, [])).status == "no_tests"
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = --invalid-option-for-test\n", encoding="utf-8"
    )
    result = asyncio.run(runner(tmp_path).run(tmp_path, []))
    assert result.status == "usage_error" and result.process.exit_code == 4
    assert result.verified is False


def test_timeout_is_not_verification(tmp_path: Path) -> None:
    (tmp_path / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(30)\n", encoding="utf-8"
    )
    result = asyncio.run(runner(tmp_path, timeout=0.5).run(tmp_path, []))
    assert result.status == "timeout" and result.process.timed_out is True
    assert result.verified is False


def test_truncated_output_preserves_final_pytest_summary(tmp_path: Path) -> None:
    (tmp_path / "test_output.py").write_text(
        "def test_output():\n    print('x' * 10000)\n    assert False\n", encoding="utf-8"
    )
    result = asyncio.run(runner(tmp_path, output_limit=1024).run(tmp_path, []))
    assert result.process.stdout_truncated is True
    assert result.status == "failed" and result.counts["failed"] == 1
    assert len(result.process.stdout.encode("utf-8")) <= 1024


def test_environment_and_parent_configuration_are_not_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = --invalid-parent-option\n", encoding="utf-8"
    )
    (tmp_path / "conftest.py").write_text(
        "raise AssertionError('parent conftest loaded')\n", encoding="utf-8"
    )
    (repository / "test_env.py").write_text(
        "import os\ndef test_env():\n"
        "    assert 'CODEFORGE_OPENAI_API_KEY' not in os.environ\n"
        "    assert 'PYTEST_ADDOPTS' not in os.environ\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEFORGE_OPENAI_API_KEY", "private-value")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--invalid-inherited-option")
    result = asyncio.run(runner(tmp_path).run(repository, []))
    assert result.status == "passed" and result.verified is True
    assert result.config_file is None


@pytest.mark.parametrize(
    "path",
    ["../outside.py", "/tmp/outside.py", "C:/outside.py", ".git/config", "missing.py", "--help"],
)
def test_invalid_targets_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(PytestRunnerError):
        asyncio.run(runner(tmp_path).run(tmp_path, [path]))


def test_symlink_escape_and_disabled_execution(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "test_outside.py"
    outside.write_text("def test_pass():\n    pass\n", encoding="utf-8")
    (repository / "test_link.py").symlink_to(outside)
    with pytest.raises(PytestRunnerError, match="inside"):
        asyncio.run(runner(tmp_path).run(repository, ["test_link.py"]))
    with pytest.raises(PytestRunnerError, match="disabled"):
        asyncio.run(PytestRunner(RepositoryAccess(tmp_path)).run(repository, []))


def test_summary_parser_uses_only_complete_final_totals() -> None:
    counts, complete = parse_summary(
        "noise: 100 passed\n== 1 failed, 2 passed, 1 warning in 0.12s =="
    )
    assert complete is True and counts["passed"] == 2 and counts["failed"] == 1
    assert counts["warnings"] == 1
    assert parse_summary("truncated passed in 0.12s")[1] is False
