"""Patch transactions against real temporary files, including injected write failures."""

import asyncio
import difflib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from agent.git_service import GitService
from agent.patch_service import PatchService
from agent.patches import PatchError
from agent.repository_access import RepositoryAccess


def patch_for(path: str, before: str | None, after: str | None) -> str:
    return "".join(
        difflib.unified_diff(
            (before or "").splitlines(keepends=True),
            (after or "").splitlines(keepends=True),
            fromfile="a/" + path if before is not None else "/dev/null",
            tofile="b/" + path if after is not None else "/dev/null",
        )
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    path = tmp_path / "repository"
    path.mkdir()
    (path / "first.py").write_text("old\n", encoding="utf-8")
    (path / "delete.py").write_text("remove\n", encoding="utf-8")
    return path


def service(
    tmp_path: Path,
    *,
    allow_writes: bool = True,
    max_files: int = 10,
    max_patch_bytes: int = 100_000,
    max_file_bytes: int = 1_000_000,
) -> PatchService:
    access = RepositoryAccess(
        tmp_path, ignore_patterns=(".venv",), max_file_size_bytes=max_file_bytes
    )
    return PatchService(
        access,
        GitService(access),
        allow_writes=allow_writes,
        max_files=max_files,
        max_patch_bytes=max_patch_bytes,
    )


def mixed_patch() -> str:
    return (
        patch_for("first.py", "old\n", "new\n")
        + patch_for("delete.py", "remove\n", None)
        + patch_for("nested/new.py", None, "created\n")
    )


def test_dry_run_changes_nothing_then_apply_creates_modifies_and_deletes(
    tmp_path: Path, repository: Path
) -> None:
    client = service(tmp_path)
    result = asyncio.run(client.execute(repository, mixed_patch()))
    assert result["dry_run"] is True and result["applied"] is False
    assert (repository / "first.py").read_text() == "old\n"
    assert (repository / "delete.py").exists()
    assert not (repository / "nested").exists()
    applied = asyncio.run(client.execute(repository, mixed_patch(), dry_run=False))
    assert applied["applied"] is True
    assert [file["action"] for file in applied["files"]] == ["modify", "delete", "create"]
    assert applied["git_diff"]["available"] is False
    assert (repository / "first.py").read_text() == "new\n"
    assert not (repository / "delete.py").exists()
    assert (repository / "nested/new.py").read_text() == "created\n"


def test_stale_second_file_rejects_entire_patch_before_writing(
    tmp_path: Path, repository: Path
) -> None:
    (repository / "delete.py").write_text("user work\n", encoding="utf-8")
    with pytest.raises(PatchError, match="stale"):
        asyncio.run(service(tmp_path).execute(repository, mixed_patch(), dry_run=False))
    assert (repository / "first.py").read_text() == "old\n"
    assert (repository / "delete.py").read_text() == "user work\n"
    assert not (repository / "nested").exists()


def test_partial_application_failure_rolls_back_contents_permissions_and_directories(
    tmp_path: Path,
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = service(tmp_path)
    (repository / "first.py").chmod(0o755)
    original_replace = client._replace

    def fail_creation(path: Path, content: bytes, mode: int) -> None:
        if path.name == "new.py":
            raise OSError("Injected write failure")
        original_replace(path, content, mode)

    monkeypatch.setattr(client, "_replace", fail_creation)
    with pytest.raises(PatchError, match="original files restored"):
        asyncio.run(client.execute(repository, mixed_patch(), dry_run=False))
    assert (repository / "first.py").read_text() == "old\n"
    assert (repository / "delete.py").read_text() == "remove\n"
    assert stat.S_IMODE((repository / "first.py").stat().st_mode) == 0o755
    assert not (repository / "nested").exists()
    assert not list(repository.rglob(".codeforge-patch-*"))


def test_concurrent_external_edit_is_preserved_on_failure(
    tmp_path: Path,
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = service(tmp_path)
    original_replace = client._replace

    def external_edit(path: Path, content: bytes, mode: int) -> None:
        if path.name == "first.py" and content == b"new\n":
            (repository / "delete.py").write_text("external user change\n", encoding="utf-8")
        original_replace(path, content, mode)

    monkeypatch.setattr(client, "_replace", external_edit)
    with pytest.raises(PatchError, match="changed during"):
        asyncio.run(client.execute(repository, mixed_patch(), dry_run=False))
    assert (repository / "first.py").read_text() == "old\n"
    assert (repository / "delete.py").read_text() == "external user change\n"


def test_incomplete_rollback_retains_recovery_copies(
    tmp_path: Path,
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = service(tmp_path)
    original_replace = client._replace
    counter = 0

    def failing_disk(path: Path, content: bytes, mode: int) -> None:
        nonlocal counter
        counter += 1
        if counter > 1:
            raise OSError("Disk failure")
        original_replace(path, content, mode)

    monkeypatch.setattr(client, "_replace", failing_disk)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(PatchError, match="Rollback incomplete") as captured:
        asyncio.run(client.execute(repository, mixed_patch(), dry_run=False))
    recovery = Path(str(captured.value).split("saved at ", 1)[1])
    manifest = json.loads((recovery / "manifest.json").read_text())
    originals = {
        entry["path"]: (recovery / entry["backup"]).read_bytes()
        for entry in manifest
        if entry["backup"] is not None
    }
    assert originals[str(repository / "first.py")] == b"old\n"
    assert originals[str(repository / "delete.py")] == b"remove\n"


@pytest.mark.parametrize(
    "path",
    [
        "../outside.py",
        "/tmp/outside.py",
        "C:/outside.py",
        "dir/../outside.py",
        ".git/config",
        ".GIT/config",
        "dir\\file.py",
        "./first.py",
    ],
)
def test_unsafe_paths_rejected(tmp_path: Path, repository: Path, path: str) -> None:
    with pytest.raises(PatchError):
        asyncio.run(
            service(tmp_path).execute(repository, patch_for(path, None, "new\n"), dry_run=False)
        )
    assert (repository / "first.py").read_text() == "old\n"


def test_symlinks_hardlinks_and_ignored_paths_rejected(tmp_path: Path, repository: Path) -> None:
    (repository / "link.py").symlink_to(repository / "first.py")
    (repository / "directory-link").symlink_to(tmp_path, target_is_directory=True)
    os.link(repository / "first.py", repository / "hard.py")
    for path, before in (
        ("link.py", "old\n"),
        ("directory-link/new.py", None),
        ("hard.py", "old\n"),
        (".venv/new.py", None),
    ):
        with pytest.raises(PatchError):
            asyncio.run(
                service(tmp_path).execute(
                    repository, patch_for(path, before, "new\n"), dry_run=False
                )
            )


def test_custom_git_metadata_protected(tmp_path: Path, repository: Path) -> None:
    (repository / "metadata").mkdir()
    (repository / "metadata/config").write_text("old\n", encoding="utf-8")
    (repository / ".git").write_text("gitdir: metadata\n", encoding="utf-8")
    with pytest.raises(PatchError, match="Git metadata"):
        asyncio.run(
            service(tmp_path).execute(
                repository, patch_for("metadata/config", "old\n", "new\n"), dry_run=False
            )
        )


def test_limits_and_disabled_writes(tmp_path: Path, repository: Path) -> None:
    for client in (
        service(tmp_path, max_patch_bytes=10),
        service(tmp_path, max_files=1),
        service(tmp_path, max_file_bytes=3),
    ):
        with pytest.raises(PatchError, match="limit"):
            asyncio.run(client.execute(repository, mixed_patch(), dry_run=False))
    disabled = service(tmp_path, allow_writes=False)
    assert asyncio.run(disabled.execute(repository, mixed_patch()))["dry_run"] is True
    with pytest.raises(PatchError, match="disabled"):
        asyncio.run(disabled.execute(repository, mixed_patch(), dry_run=False))


def test_existing_creation_target_and_binary_source_rejected(
    tmp_path: Path, repository: Path
) -> None:
    with pytest.raises(PatchError, match="overwrite"):
        asyncio.run(service(tmp_path).execute(repository, patch_for("first.py", None, "new\n")))
    (repository / "first.py").write_bytes(b"old\0\n")
    with pytest.raises(PatchError, match="Binary"):
        asyncio.run(service(tmp_path).execute(repository, patch_for("first.py", "old\n", "new\n")))


def test_updated_git_diff_and_index_preservation(tmp_path: Path, repository: Path) -> None:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-b", "main")
    git("add", "first.py")
    before_index = git("write-tree")
    result = asyncio.run(
        service(tmp_path).execute(
            repository, patch_for("first.py", "old\n", "new\n"), dry_run=False
        )
    )
    assert result["git_diff"]["available"] is True
    assert "+new" in result["git_diff"]["patch"]
    assert git("write-tree") == before_index
    assert (repository / "delete.py").read_text() == "remove\n"
