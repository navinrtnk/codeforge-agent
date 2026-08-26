"""Safe, bounded access to files inside registered repositories."""

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path


class RepositoryAccessError(Exception):
    """Base error for repository access failures."""


class PathOutsideWorkspaceError(RepositoryAccessError):
    """Raised when a repository resolves outside the configured workspace."""


class PathOutsideRepositoryError(RepositoryAccessError):
    """Raised when a file resolves outside its repository."""


class RepositoryPathError(RepositoryAccessError):
    """Raised when a repository or file path is missing or has the wrong type."""


class RepositoryNotFoundError(RepositoryPathError):
    """Raised when a repository path does not exist."""


class FileTooLargeError(RepositoryAccessError):
    """Raised when a file exceeds the configured read limit."""


class BinaryFileError(RepositoryAccessError):
    """Raised when a requested file is not UTF-8 text."""


class InvalidLineRangeError(RepositoryAccessError):
    """Raised when a requested line range is invalid."""


@dataclass(frozen=True, slots=True)
class RepositoryFile:
    """Metadata for one repository file."""

    path: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class FileContent:
    """A bounded section of a repository text file."""

    path: str
    content: str
    start_line: int
    end_line: int
    total_lines: int


class RepositoryAccess:
    """Resolve and read repository paths without crossing trusted boundaries."""

    def __init__(
        self,
        workspace_root: Path,
        ignore_patterns: tuple[str, ...] = (),
        max_file_size_bytes: int = 1_000_000,
    ) -> None:
        resolved_workspace = workspace_root.expanduser().resolve()
        if not resolved_workspace.is_dir():
            raise RepositoryPathError("Workspace root is not a directory")
        if max_file_size_bytes < 1:
            raise ValueError("Maximum file size must be positive")

        self.workspace_root = resolved_workspace
        self.ignore_patterns = ignore_patterns
        self.max_file_size_bytes = max_file_size_bytes

    def resolve_repository(self, repository_path: Path) -> Path:
        """Resolve a repository directory within the configured workspace."""
        candidate = repository_path.expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()

        if not resolved.is_relative_to(self.workspace_root):
            raise PathOutsideWorkspaceError("Repository path is outside the workspace root")
        if not resolved.exists():
            raise RepositoryNotFoundError("Repository path does not exist")
        if not resolved.is_dir():
            raise RepositoryPathError("Repository path is not a directory")
        return resolved

    def resolve_file(self, repository_path: Path, file_path: Path) -> Path:
        """Resolve an existing regular file within a repository."""
        repository = self.resolve_repository(repository_path)
        if file_path.is_absolute():
            raise PathOutsideRepositoryError("File path must be relative to the repository")

        resolved = (repository / file_path).resolve()
        if not resolved.is_relative_to(repository):
            raise PathOutsideRepositoryError("File path is outside the repository")
        if not resolved.is_relative_to(self.workspace_root):
            raise PathOutsideWorkspaceError("File path is outside the workspace root")
        if not resolved.exists():
            raise RepositoryPathError("File does not exist")
        if not resolved.is_file():
            raise RepositoryPathError("File path is not a regular file")
        return resolved

    def list_files(self, repository_path: Path) -> list[RepositoryFile]:
        """List non-ignored regular files without following directory symlinks."""
        repository = self.resolve_repository(repository_path)
        files: list[RepositoryFile] = []

        for current_root, directory_names, file_names in os.walk(repository, followlinks=False):
            root = Path(current_root)
            relative_root = root.relative_to(repository)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if not (root / name).is_symlink() and not self._is_ignored(relative_root / name)
            )

            for name in sorted(file_names):
                relative_path = relative_root / name
                candidate = root / name
                if self._is_ignored(relative_path) or candidate.is_symlink():
                    continue
                if candidate.is_file():
                    files.append(
                        RepositoryFile(
                            path=relative_path.as_posix(),
                            size_bytes=candidate.stat().st_size,
                        )
                    )

        return sorted(files, key=lambda file: file.path)

    def read_text(
        self,
        repository_path: Path,
        file_path: Path,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> FileContent:
        """Read a UTF-8 file or an inclusive, one-indexed line range."""
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise InvalidLineRangeError("Line range must be positive and ordered")

        repository = self.resolve_repository(repository_path)
        resolved_file = self.resolve_file(repository, file_path)
        with resolved_file.open("rb") as file:
            raw_content = file.read(self.max_file_size_bytes + 1)
        if len(raw_content) > self.max_file_size_bytes:
            raise FileTooLargeError(f"File exceeds the {self.max_file_size_bytes}-byte read limit")

        if b"\x00" in raw_content:
            raise BinaryFileError("File contains null bytes")
        try:
            text = raw_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BinaryFileError("File is not valid UTF-8 text") from error

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        if total_lines > 0 and start_line > total_lines:
            raise InvalidLineRangeError("Start line exceeds file length")
        selected_end = total_lines if end_line is None else min(end_line, total_lines)
        content = "".join(lines[start_line - 1 : selected_end])
        return FileContent(
            path=resolved_file.relative_to(repository).as_posix(),
            content=content,
            start_line=start_line,
            end_line=selected_end,
            total_lines=total_lines,
        )

    def _is_ignored(self, relative_path: Path) -> bool:
        path = relative_path.as_posix()
        return any(
            fnmatch.fnmatch(path, pattern)
            or fnmatch.fnmatch(relative_path.name, pattern)
            or pattern in relative_path.parts
            for pattern in self.ignore_patterns
        )
