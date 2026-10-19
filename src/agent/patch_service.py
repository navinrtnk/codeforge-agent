"""Validate all patch targets before applying files, with compensating rollback."""

import fnmatch
import json
import os
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from agent.git_service import GitError, GitService
from agent.models_api import JsonObject
from agent.patches import PatchError, apply_hunks, parse_unified_diff
from agent.repository_access import RepositoryAccess, RepositoryAccessError


@dataclass(frozen=True)
class PreparedChange:
    """The exact original state and proposed replacement for one validated file."""

    path: Path
    before: bytes | None
    after: bytes | None
    mode: int


class PatchService:
    """Apply bounded text patches while preserving unrelated working-tree changes."""

    def __init__(
        self,
        access: RepositoryAccess,
        git_service: GitService,
        *,
        allow_writes: bool = False,
        max_patch_bytes: int = 100_000,
        max_files: int = 10,
    ) -> None:
        if max_patch_bytes < 1 or max_files < 1:
            raise ValueError("Patch byte and file limits must be positive")
        self.access = access
        self.git = git_service
        self.allow_writes = allow_writes
        self.max_patch_bytes = max_patch_bytes
        self.max_files = max_files

    async def execute(self, path: Path, patch: str, *, dry_run: bool = True) -> JsonObject:
        """Validate a complete patch, optionally write it, and report the updated diff."""
        if not dry_run and not self.allow_writes:
            raise PatchError("Code writes are disabled")
        if len(patch.encode("utf-8")) > self.max_patch_bytes:
            raise PatchError("Patch exceeds the configured byte limit")
        parsed = parse_unified_diff(patch)
        if len(parsed) > self.max_files:
            raise PatchError("Patch exceeds the configured changed-file limit")
        try:
            repository = self.access.resolve_repository(path)
            async with self.git.repository_lock(repository):
                protected = self._metadata_paths(repository)
                changes = []
                for file_patch in parsed:
                    target = self._target(repository, file_patch.path, protected)
                    before, mode = self._read(target)
                    if file_patch.old_path is None and before is not None:
                        raise PatchError("A creation patch cannot overwrite an existing file")
                    if file_patch.old_path is not None and before is None:
                        raise PatchError("Patch source file does not exist")
                    try:
                        original = before.decode("utf-8") if before is not None else ""
                    except UnicodeDecodeError as error:
                        raise PatchError("Patch source must be UTF-8 text") from error
                    if "\0" in original:
                        raise PatchError("Binary files cannot be patched")
                    after = apply_hunks(file_patch, original).encode("utf-8")
                    if len(after) > self.access.max_file_size_bytes:
                        raise PatchError("Patched file exceeds the configured file-size limit")
                    changes.append(
                        PreparedChange(target, before, after if file_patch.new_path else None, mode)
                    )
                # Reject aliases, including case-insensitive aliases on Windows.
                if len({os.path.normcase(str(change.path)) for change in changes}) != len(changes):
                    raise PatchError("Patch targets must name distinct files")
                if any(
                    left.path in right.path.parents
                    for left in changes
                    for right in changes
                    if left is not right
                ):
                    raise PatchError("Patch targets cannot be parent and child paths")
                if not dry_run:
                    self._commit(repository, protected, changes)
                result: JsonObject = {
                    "applied": not dry_run,
                    "dry_run": dry_run,
                    "files": [
                        {
                            "path": change.path.relative_to(repository).as_posix(),
                            "action": "create"
                            if change.before is None
                            else "delete"
                            if change.after is None
                            else "modify",
                        }
                        for change in changes
                    ],
                }
                if not dry_run:
                    try:
                        result["git_diff"] = {"available": True, **await self.git.diff(repository)}
                    except GitError, RepositoryAccessError:
                        result["git_diff"] = {"available": False, "reason": "Git diff unavailable"}
                return result
        except RepositoryAccessError as error:
            raise PatchError(str(error)) from error
        except OSError as error:
            raise PatchError("Patch filesystem operation failed") from error

    def _metadata_paths(self, repository: Path) -> tuple[Path, ...]:
        """Protect Git metadata even for linked worktrees or custom gitdir locations."""
        git_dir = repository / ".git"
        if git_dir.is_file():
            pointer = self._small_text(git_dir)
            if not pointer.startswith("gitdir: "):
                raise PatchError("Invalid Git metadata pointer")
            git_dir = (repository / pointer[8:].strip()).resolve()
        common = git_dir / "commondir"
        if common.is_file():
            return git_dir.resolve(), (git_dir / self._small_text(common).strip()).resolve()
        return (git_dir.resolve(),)

    @staticmethod
    def _small_text(path: Path) -> str:
        with path.open("rb") as source:
            data = source.read(4097)
        if len(data) > 4096:
            raise PatchError("Git metadata pointer is too large")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PatchError("Git metadata pointer must be UTF-8") from error

    def _target(self, repository: Path, value: str, protected: tuple[Path, ...]) -> Path:
        relative = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or PureWindowsPath(value).drive
            or relative.is_absolute()
            or any(part in {".", "..", ""} for part in value.split("/"))
            or any(part.lower() in {".git", ".hg", ".svn"} for part in relative.parts)
        ):
            raise PatchError("Patch paths must be relative files inside the repository")
        target = repository / value
        for component in (target, *target.parents):
            if component == repository:
                break
            if component.is_symlink():
                raise PatchError("Patch paths cannot contain symlinks")
            if component != target and component.exists() and not component.is_dir():
                raise PatchError("Patch parent paths must be directories")
        resolved = target.resolve()
        if not resolved.is_relative_to(repository) or any(
            resolved.is_relative_to(root) for root in protected
        ):
            raise PatchError("Patch path is outside the repository or targets Git metadata")
        if any(
            fnmatch.fnmatch(value, pattern)
            or any(fnmatch.fnmatch(part, pattern) for part in relative.parts)
            for pattern in self.access.ignore_patterns
        ):
            raise PatchError("Patch targets an ignored path")
        return target

    def _read(self, path: Path) -> tuple[bytes | None, int]:
        if not path.exists():
            return None, 0o644
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise PatchError("Patch targets must be regular files without hard links")
        with path.open("rb") as source:
            content = source.read(self.access.max_file_size_bytes + 1)
        if len(content) > self.access.max_file_size_bytes:
            raise PatchError("Patch source exceeds the configured file-size limit")
        return content, stat.S_IMODE(metadata.st_mode)

    @staticmethod
    def _replace(path: Path, content: bytes, mode: int) -> None:
        """Replace one file atomically using a temporary file in its directory."""
        descriptor, name = tempfile.mkstemp(prefix=".codeforge-patch-", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            temporary.chmod(mode)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _commit(
        self, repository: Path, protected: tuple[Path, ...], changes: list[PreparedChange]
    ) -> None:
        attempted: list[PreparedChange] = []
        created_directories: list[Path] = []
        try:
            for change in changes:
                self._target(repository, change.path.relative_to(repository).as_posix(), protected)
                if self._read(change.path) != (change.before, change.mode):
                    raise PatchError("File changed after patch validation")
            for change in changes:
                self._target(repository, change.path.relative_to(repository).as_posix(), protected)
                if self._read(change.path) != (change.before, change.mode):
                    raise PatchError("File changed during patch application")
                missing = []
                parent = change.path.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for directory in reversed(missing):
                    directory.mkdir()
                    created_directories.append(directory)
                attempted.append(change)
                if change.after is None:
                    change.path.unlink()
                else:
                    self._replace(change.path, change.after, change.mode)
        except BaseException as error:
            failures = []
            for change in reversed(attempted):
                try:
                    self._target(
                        repository, change.path.relative_to(repository).as_posix(), protected
                    )
                    current, _ = self._read(change.path)
                    if current not in (change.before, change.after):
                        raise PatchError("File changed during rollback")
                    if change.before is None:
                        change.path.unlink(missing_ok=True)
                    else:
                        self._replace(change.path, change.before, change.mode)
                except (OSError, PatchError) as rollback_error:
                    failures.append(rollback_error)
            for directory in reversed(created_directories):
                with suppress(OSError):
                    directory.rmdir()
            if failures:
                recovery = Path(tempfile.mkdtemp(prefix="codeforge-patch-recovery-"))
                manifest = []
                for index, change in enumerate(attempted):
                    backup = str(index) if change.before is not None else None
                    if backup is not None:
                        (recovery / backup).write_bytes(change.before or b"")
                    manifest.append(
                        {"path": str(change.path), "backup": backup, "mode": change.mode}
                    )
                (recovery / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                raise PatchError(
                    f"Rollback incomplete; original files saved at {recovery}"
                ) from error
            if isinstance(error, PatchError):
                raise
            if isinstance(error, Exception):
                raise PatchError("Patch application failed; original files restored") from error
            raise
