"""Repeatable evaluations of the production runner against trusted repository fixtures."""

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from agent.agent_runs import build_agent_runner
from agent.command_sandbox import create_command_sandbox
from agent.config import Settings
from agent.database import Database
from agent.indexing import RepositoryIndexer
from agent.main import create_app
from agent.model_factory import create_model_client
from agent.models import AgentMessageRole, AgentRunMode, AgentRunStatus, AgentRunType, Repository
from agent.models_api import (
    FakeModelClient,
    ModelClient,
    ModelResponse,
    TextContent,
    TokenUsage,
    ToolCall,
)
from agent.pytest_runner import PytestRunner
from agent.repository_access import RepositoryAccess


class ScriptStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str = ""


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
    kind: Literal["read_only", "repair"]
    fixture: str
    task: str = Field(min_length=1)
    expected_tests: str | None = None
    answer_contains: list[str] = Field(default_factory=list)
    script: list[ScriptStep] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def expectations(self) -> EvaluationCase:
        if self.kind == "repair" and not self.expected_tests:
            raise ValueError("Repair cases require expected tests")
        if self.kind == "read_only" and not self.answer_contains:
            raise ValueError("Read-only cases require answer expectations")
        return self


class EvaluationSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cases: list[EvaluationCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_ids(self) -> EvaluationSuite:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique")
        return self


class EvaluationResult(BaseModel):
    case_id: str
    kind: str
    success: bool = False
    status: str = "error"
    error: str | None = None
    agent_verified: bool = False
    expected_tests_verified: bool | None = None
    initial_tests_status: str | None = None
    final_tests_status: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    iterations: int = 0
    tool_calls: int = 0
    repair_attempts: int = 0
    patch_bytes: int = 0
    changed_files: int = 0
    added_lines: int = 0
    removed_lines: int = 0
    agent_seconds: float = 0
    total_seconds: float = 0


IGNORED = (".git", "__pycache__", ".pytest_cache", ".venv")


def fixture_path(root: Path, value: str) -> Path:
    path = root / value
    if Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("Fixture paths must stay under the suite directory")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_dir():
        raise ValueError("Fixture must be a directory inside the suite")
    if path.is_symlink() or any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError("Evaluation fixtures cannot contain symlinks")
    return path


def git(path: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, timeout=15
    ).stdout


def snapshot(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"))
        if file.is_file() and not any(part in IGNORED for part in file.relative_to(path).parts)
    }


def scripted_client(case: EvaluationCase) -> ModelClient:
    if not case.script:
        raise ValueError("Scripted evaluations require a response script")
    return FakeModelClient(
        [
            ModelResponse(
                str(index),
                "fake",
                "scripted-baseline",
                (ToolCall(str(index), step.tool, step.arguments),)
                if step.tool
                else (TextContent(step.text),),
                "tool_use" if step.tool else "end_turn",
                TokenUsage(0, 0),
            )
            for index, step in enumerate(case.script)
        ]
    )


async def expected_tests(
    repository: Path, tests: Path, destination: Path, settings: Settings
) -> tuple[str, bool]:
    """Grade a separate copy; expected tests are never exposed during the agent run."""
    shutil.copytree(repository, destination, ignore=shutil.ignore_patterns(*IGNORED))
    target = destination / "evaluation_tests"
    if target.exists():
        raise ValueError("Reserved evaluation_tests directory already exists")
    shutil.copytree(tests, target)
    access = RepositoryAccess(destination.parent)
    result = await PytestRunner(
        access,
        enabled=True,
        python_executable=settings.test_python_executable,
        timeout_seconds=settings.test_timeout_seconds,
        max_output_bytes=settings.test_max_output_bytes,
        sandbox=create_command_sandbox(settings, settings.test_python_executable),
    ).run(destination, ["evaluation_tests"])
    return result.status, result.verified


async def evaluate_case(
    case: EvaluationCase,
    root: Path,
    settings: Settings,
    client_factory: Callable[[EvaluationCase], ModelClient],
) -> EvaluationResult:
    result = EvaluationResult(case_id=case.id, kind=case.kind)
    started = perf_counter()
    try:
        fixture = fixture_path(root, case.fixture)
        oracle = fixture_path(root, case.expected_tests) if case.expected_tests else None
        with tempfile.TemporaryDirectory(prefix="codeforge-eval-") as directory:
            workspace = Path(directory)
            repository_path = workspace / "repository"
            shutil.copytree(fixture, repository_path, ignore=shutil.ignore_patterns(*IGNORED))
            git(repository_path, "init", "-q")
            git(repository_path, "add", ".")
            git(
                repository_path,
                "-c",
                "user.name=CodeForge Evaluation",
                "-c",
                "user.email=evaluation@localhost",
                "commit",
                "-qm",
                "fixture",
            )
            before = snapshot(repository_path)
            # Evaluation never imports external MCP tools or mutates the service database.
            configuration = settings.model_copy(
                update={
                    "workspace_root": workspace,
                    "database_url": "sqlite://",
                    "code_writes_enabled": case.kind == "repair",
                    "test_execution_enabled": case.kind == "repair",
                    "analysis_execution_enabled": False,
                    "git_allow_mutations": False,
                    "embeddings_enabled": False,
                    "mcp_servers": {},
                }
            )
            initial_verified = False
            if oracle is not None:
                result.initial_tests_status, initial_verified = await expected_tests(
                    repository_path, oracle, workspace / "initial", configuration
                )
            database = Database("sqlite://")
            database.create_schema()
            try:
                access = RepositoryAccess(workspace, ignore_patterns=IGNORED)
                with database.session_factory() as session:
                    repository = Repository(name=case.id, path=str(repository_path))
                    session.add(repository)
                    session.commit()
                    RepositoryIndexer(access).index(session, repository)
                    app = create_app(configuration)
                    request = Request({"type": "http", "app": app})
                    runner = build_agent_runner(
                        request,
                        access,
                        client_factory(case),
                        mode=AgentRunMode.WRITE
                        if case.kind == "repair"
                        else AgentRunMode.READ_ONLY,
                        run_type=AgentRunType.BUG_FIX
                        if case.kind == "repair"
                        else AgentRunType.TASK,
                    )
                    agent_started = perf_counter()
                    run = await runner.run(session, repository, case.task)
                    result.agent_seconds = round(perf_counter() - agent_started, 6)
                    result.status = run.status.value
                    result.agent_verified = run.verified
                    result.repair_attempts = run.repair_attempts
                    result.tool_calls = len(run.tool_events)
                    responses = [m for m in run.messages if m.response_id is not None]
                    result.iterations = len(responses)
                    result.input_tokens = sum(m.input_tokens or 0 for m in responses)
                    result.output_tokens = sum(m.output_tokens or 0 for m in responses)
                    answer = "\n".join(
                        str(block.get("text", ""))
                        for message in run.messages
                        if message.role is AgentMessageRole.ASSISTANT
                        for block in message.content
                    )
                    after = snapshot(repository_path)
                    result.changed_files = sum(
                        before.get(name) != after.get(name) for name in before.keys() | after.keys()
                    )
                    git(repository_path, "add", "-N", ".")
                    result.patch_bytes = len(git(repository_path, "diff", "--binary", "HEAD"))
                    for line in (
                        git(repository_path, "diff", "--numstat", "HEAD").decode().splitlines()
                    ):
                        added, removed, _ = line.split("\t", 2)
                        if added.isdigit():
                            result.added_lines += int(added)
                            result.removed_lines += int(removed)
                    if oracle is not None:
                        (
                            result.final_tests_status,
                            result.expected_tests_verified,
                        ) = await expected_tests(
                            repository_path, oracle, workspace / "final", configuration
                        )
                    completed = run.status is AgentRunStatus.COMPLETED
                    if case.kind == "repair":
                        result.success = bool(
                            completed
                            and run.verified
                            and result.expected_tests_verified
                            and result.initial_tests_status == "failed"
                            and not initial_verified
                            and result.changed_files
                        )
                    else:
                        result.success = (
                            completed
                            and not result.changed_files
                            and all(
                                text.casefold() in answer.casefold()
                                for text in case.answer_contains
                            )
                            and result.expected_tests_verified is not False
                        )
            finally:
                database.dispose()
    except Exception as error:
        # Reports omit raw provider errors, paths, prompts, and credentials.
        result.status = "error"
        result.error = type(error).__name__
    result.total_seconds = round(perf_counter() - started, 6)
    return result


async def evaluate_suite(
    manifest: Path,
    settings: Settings,
    *,
    client_factory: Callable[[EvaluationCase], ModelClient] = scripted_client,
) -> list[EvaluationResult]:
    suite = EvaluationSuite.model_validate_json(manifest.read_text(encoding="utf-8"))
    return [
        await evaluate_case(case, manifest.parent, settings, client_factory)
        for case in sorted(suite.cases, key=lambda case: case.id)
    ]


def write_reports(
    results: list[EvaluationResult],
    output: Path,
    *,
    live: bool,
    manifest: Path,
    settings: Settings,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    passed = sum(result.success for result in results)
    report = {
        "schema_version": 1,
        "mode": "live" if live else "scripted",
        "total": len(results),
        "passed": passed,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "configuration": settings.model_dump(
            mode="json",
            include={
                "model_provider",
                "model_name",
                "agent_max_iterations",
                "agent_max_output_tokens",
                "agent_timeout_seconds",
                "bug_fix_max_attempts",
                "command_execution_backend",
                "sandbox_image",
                "sandbox_python_executable",
                "sandbox_cpus",
                "sandbox_memory_mb",
                "sandbox_pids_limit",
                "sandbox_network_enabled",
                "sandbox_timeout_seconds",
                "test_timeout_seconds",
                "patch_max_bytes",
                "patch_max_files",
            },
        ),
        "cases": [result.model_dump() for result in results],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Evaluation report",
        "",
        f"Mode: {report['mode']}. Passed: {passed}/{len(results)}.",
        "",
        "| Case | Success | Agent verified | Expected tests | Tokens in/out | Turns | "
        "Tools | Patch bytes | Seconds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {r.case_id} | {r.success} | {r.agent_verified} | {r.expected_tests_verified} | "
        f"{r.input_tokens}/{r.output_tokens} | {r.iterations} | {r.tool_calls} | "
        f"{r.patch_bytes} | {r.total_seconds:.3f} |"
        for r in results
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--live", action="store_true", help="Use configured model (incurs API costs)"
    )
    args = parser.parse_args()
    settings = Settings()
    results = asyncio.run(
        evaluate_suite(
            args.manifest,
            settings,
            client_factory=(lambda _: create_model_client(settings))
            if args.live
            else scripted_client,
        )
    )
    write_reports(results, args.output, live=args.live, manifest=args.manifest, settings=settings)
    print(f"{sum(result.success for result in results)}/{len(results)} evaluation cases passed")
    raise SystemExit(0 if all(result.success for result in results) else 1)


if __name__ == "__main__":
    main()
