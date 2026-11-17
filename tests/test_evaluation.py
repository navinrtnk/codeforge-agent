"""Exercise grading through real agent runs, Git copies, and independent pytest suites."""

import asyncio
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent.config import Settings
from agent.evaluation import (
    EvaluationCase,
    EvaluationSuite,
    evaluate_case,
    evaluate_suite,
    fixture_path,
    scripted_client,
    snapshot,
    write_reports,
)
from agent.models_api import FakeModelClient, ModelRequest, ModelResponse, TokenUsage

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "evaluations" / "baseline.json"


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]


def cases() -> list[EvaluationCase]:
    return EvaluationSuite.model_validate_json(MANIFEST.read_text()).cases


def test_baselines_are_repeatable_and_leave_original_fixtures_unchanged(tmp_path: Path) -> None:
    before = snapshot(MANIFEST.parent)
    first = asyncio.run(evaluate_suite(MANIFEST, settings()))
    second = asyncio.run(evaluate_suite(MANIFEST, settings()))
    assert snapshot(MANIFEST.parent) == before
    assert all(result.success for result in first + second)
    for left, right in zip(first, second, strict=True):
        assert left.model_dump(exclude={"agent_seconds", "total_seconds"}) == right.model_dump(
            exclude={"agent_seconds", "total_seconds"}
        )
    read, repair = first
    assert read.changed_files == 0 and not read.agent_verified
    assert read.expected_tests_verified is None and read.iterations == 2
    assert repair.initial_tests_status == "failed" and repair.final_tests_status == "passed"
    assert repair.agent_verified and repair.expected_tests_verified
    assert repair.changed_files == 1 and repair.added_lines == repair.removed_lines == 1
    assert repair.patch_bytes > 0 and repair.repair_attempts == 1 and repair.tool_calls == 3
    write_reports(
        first,
        tmp_path,
        live=False,
        manifest=MANIFEST,
        settings=settings(openai_api_key="private-key"),
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed"] == 2 and report["mode"] == "scripted"
    assert len(report["manifest_sha256"]) == 64
    assert "private-key" not in (tmp_path / "report.json").read_text()
    assert "explain-addition" in (tmp_path / "report.md").read_text()


def test_independent_tests_reject_a_patch_that_only_passes_visible_tests() -> None:
    case = cases()[1].model_copy(deep=True)
    case.script[1].arguments["patch"] = (
        case.script[1].arguments["patch"].replace("+    return left + right", "+    return 5")
    )
    result = asyncio.run(evaluate_case(case, MANIFEST.parent, settings(), scripted_client))
    assert result.agent_verified and not result.expected_tests_verified and not result.success
    assert result.final_tests_status == "failed"


def test_answer_expectation_failure_and_token_accounting() -> None:
    case = cases()[0].model_copy(deep=True)
    case.answer_contains = ["missing explanation"]

    def factory(case: EvaluationCase) -> FakeModelClient:
        client = scripted_client(case)
        assert isinstance(client, FakeModelClient)
        client.responses = [replace(r, usage=TokenUsage(11, 7)) for r in client.responses]
        return client

    result = asyncio.run(evaluate_case(case, MANIFEST.parent, settings(), factory))
    assert not result.success and result.status == "completed"
    assert result.input_tokens == 22 and result.output_tokens == 14


def test_cases_without_scripts_can_use_an_injected_model() -> None:
    case = cases()[0].model_copy(update={"script": []})
    result = asyncio.run(
        evaluate_case(case, MANIFEST.parent, settings(), lambda _: scripted_client(cases()[0]))
    )
    assert result.success
    with pytest.raises(ValueError, match="response script"):
        scripted_client(case)


@pytest.mark.parametrize("oracle", ["", "def test_skip():\n    import pytest\n    pytest.skip()\n"])
def test_empty_or_skipped_expected_tests_do_not_verify_repair(tmp_path: Path, oracle: str) -> None:
    shutil.copytree(MANIFEST.parent, tmp_path / "suite")
    (tmp_path / "suite/expected/addition/test_expected.py").write_text(oracle)
    result = asyncio.run(evaluate_case(cases()[1], tmp_path / "suite", settings(), scripted_client))
    assert not result.success and not result.expected_tests_verified


def test_case_errors_do_not_stop_later_cases(tmp_path: Path) -> None:
    shutil.copytree(MANIFEST.parent, tmp_path / "suite")
    suite = EvaluationSuite(cases=cases())
    suite.cases[0].fixture = "missing"
    manifest = tmp_path / "suite/cases.json"
    manifest.write_text(suite.model_dump_json())
    results = asyncio.run(evaluate_suite(manifest, settings()))
    assert len(results) == 2
    assert results[0].status == "error" and results[0].error == "ValueError"
    assert results[1].success


@pytest.mark.parametrize("path", ["../outside", "/tmp", "missing"])
def test_escaping_or_missing_fixture_is_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError):
        fixture_path(tmp_path, path)


def test_symlink_fixture_is_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "link").symlink_to(MANIFEST)
    with pytest.raises(ValueError, match="symlink"):
        fixture_path(tmp_path, "fixture")


def test_invalid_manifest_is_rejected_before_running() -> None:
    with pytest.raises(ValueError, match="unique"):
        EvaluationSuite(cases=[cases()[0], cases()[0]])
    with pytest.raises(ValueError, match="expected tests"):
        EvaluationCase.model_validate(
            {
                "id": "bad",
                "kind": "repair",
                "fixture": "fixtures/addition",
                "task": "fix",
                "script": [{"text": "done"}],
            }
        )


def test_timeout_is_reported_and_cancellation_propagates(tmp_path: Path) -> None:
    class SlowClient:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            await asyncio.sleep(10)
            raise AssertionError("Should be cancelled")

    result = asyncio.run(
        evaluate_case(
            cases()[0],
            MANIFEST.parent,
            settings(agent_timeout_seconds=0.02),
            lambda _: SlowClient(),
        )
    )
    assert result.status == "failed" and not result.success

    async def cancel() -> None:
        task = asyncio.create_task(
            evaluate_case(cases()[0], MANIFEST.parent, settings(), lambda _: SlowClient())
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())


def test_cli_reports_and_exit_status(tmp_path: Path) -> None:
    process = subprocess.run(
        [sys.executable, "-m", "agent.evaluation", str(MANIFEST), "--output", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert "2/2" in process.stdout and (tmp_path / "report.md").exists()
    manifest = tmp_path / "bad.json"
    suite = EvaluationSuite(cases=[cases()[0]])
    suite.cases[0].fixture = "missing"
    manifest.write_text(suite.model_dump_json())
    process = subprocess.run(
        [sys.executable, "-m", "agent.evaluation", str(manifest), "--output", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 1
    assert json.loads((tmp_path / "report.json").read_text())["passed"] == 0
