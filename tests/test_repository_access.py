"""Tests for safe repository filesystem access."""

from pathlib import Path

import pytest

from agent.repository_access import (
    BinaryFileError,
    FileTooLargeError,
    InvalidLineRangeError,
    PathOutsideRepositoryError,
    PathOutsideWorkspaceError,
    RepositoryAccess,
    RepositoryPathError,
)

DEFAULT_IGNORES = (".git", ".venv", "node_modules", "__pycache__", "build", "*.pyc")


def test_resolve_repository_accepts_relative_workspace_path(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    access = RepositoryAccess(tmp_path)

    assert access.resolve_repository(Path("repository")) == repository.resolve()


def test_resolve_repository_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    access = RepositoryAccess(workspace)

    with pytest.raises(PathOutsideWorkspaceError):
        access.resolve_repository(outside)


def test_resolve_repository_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "linked-repository").symlink_to(outside, target_is_directory=True)
    access = RepositoryAccess(workspace)

    with pytest.raises(PathOutsideWorkspaceError):
        access.resolve_repository(workspace / "linked-repository")


def test_resolve_file_rejects_traversal_and_escaping_symlink(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    (repository / "linked.txt").symlink_to(outside_file)
    access = RepositoryAccess(tmp_path)

    with pytest.raises(PathOutsideRepositoryError):
        access.resolve_file(repository, Path("../secret.txt"))
    with pytest.raises(PathOutsideRepositoryError):
        access.resolve_file(repository, Path("linked.txt"))


def test_list_files_excludes_generated_paths_and_symlinks(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "src").mkdir()
    (repository / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    (repository / "src" / "main.pyc").write_bytes(b"compiled")
    (repository / ".git").mkdir()
    (repository / ".git" / "config").write_text("ignored", encoding="utf-8")
    (repository / "build").mkdir()
    (repository / "build" / "output.txt").write_text("ignored", encoding="utf-8")
    (repository / "README.md").write_text("read me", encoding="utf-8")
    (repository / "linked.txt").symlink_to(repository / "README.md")
    access = RepositoryAccess(tmp_path, ignore_patterns=DEFAULT_IGNORES)

    files = access.list_files(repository)

    assert [file.path for file in files] == ["README.md", "src/main.py"]
    assert files[0].size_bytes == len("read me")


def test_list_files_applies_custom_glob_patterns(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "included.py").touch()
    (repository / "generated.min.js").touch()
    access = RepositoryAccess(tmp_path, ignore_patterns=("*.min.js",))

    assert [file.path for file in access.list_files(repository)] == ["included.py"]


def test_read_text_returns_requested_line_range(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "example.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    access = RepositoryAccess(tmp_path)

    result = access.read_text(repository, Path("example.py"), start_line=2, end_line=3)

    assert result.path == "example.py"
    assert result.content == "two\nthree\n"
    assert result.start_line == 2
    assert result.end_line == 3
    assert result.total_lines == 3


def test_read_text_clamps_end_line_to_file_length(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "example.py").write_text("one\ntwo", encoding="utf-8")
    access = RepositoryAccess(tmp_path)

    result = access.read_text(repository, Path("example.py"), end_line=20)

    assert result.content == "one\ntwo"
    assert result.end_line == 2


def test_read_text_rejects_oversized_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "large.txt").write_text("12345", encoding="utf-8")
    access = RepositoryAccess(tmp_path, max_file_size_bytes=4)

    with pytest.raises(FileTooLargeError):
        access.read_text(repository, Path("large.txt"))


@pytest.mark.parametrize("content", [b"text\x00data", b"\xff\xfe"])
def test_read_text_rejects_binary_content(tmp_path: Path, content: bytes) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "binary.dat").write_bytes(content)
    access = RepositoryAccess(tmp_path)

    with pytest.raises(BinaryFileError):
        access.read_text(repository, Path("binary.dat"))


@pytest.mark.parametrize(
    ("start_line", "end_line"),
    [(0, None), (-1, 2), (3, 2)],
)
def test_read_text_rejects_invalid_line_ranges(
    tmp_path: Path,
    start_line: int,
    end_line: int | None,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "example.py").write_text("pass\n", encoding="utf-8")
    access = RepositoryAccess(tmp_path)

    with pytest.raises(InvalidLineRangeError):
        access.read_text(
            repository,
            Path("example.py"),
            start_line=start_line,
            end_line=end_line,
        )


def test_read_text_rejects_start_line_past_file_length(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "example.py").write_text("pass\n", encoding="utf-8")
    access = RepositoryAccess(tmp_path)

    with pytest.raises(InvalidLineRangeError):
        access.read_text(repository, Path("example.py"), start_line=2)


def test_read_text_rejects_missing_file(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    access = RepositoryAccess(tmp_path)

    with pytest.raises(RepositoryPathError):
        access.read_text(repository, Path("missing.py"))
