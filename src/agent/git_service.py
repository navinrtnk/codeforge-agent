"""Bounded local Git operations within a registered workspace."""

import asyncio
import os
from pathlib import Path, PurePosixPath

from agent.models_api import JsonObject
from agent.repository_access import RepositoryAccess


class GitError(Exception):
    """A Git failure safe to report through the tool boundary."""


class GitService:
    """Expose fixed Git operations, without shell commands or remote access."""

    def __init__(
        self,
        access: RepositoryAccess,
        *,
        allow_mutations: bool = False,
        timeout_seconds: float = 15,
        max_output_bytes: int = 100_000,
    ) -> None:
        if timeout_seconds <= 0 or max_output_bytes < 256:
            raise ValueError("Git timeout must be positive and output limit at least 256")
        self.access = access
        self.allow_mutations = allow_mutations
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self._locks: dict[Path, asyncio.Lock] = {}

    async def _command(
        self, repository: Path, *arguments: str, allowed_codes: tuple[int, ...] = (0,)
    ) -> tuple[int, str]:
        environment = {
            key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
        }
        environment.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_TERMINAL_PROMPT="0",
            GIT_OPTIONAL_LOCKS="0",
            GIT_LITERAL_PATHSPECS="1",
            LC_ALL="C",
        )
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "--no-pager",
                "-c",
                "core.hooksPath=" + os.devnull,
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "submodule.recurse=false",
                "-c",
                "color.ui=false",
                "-c",
                "protocol.allow=never",
                "-c",
                "maintenance.auto=false",
                "-c",
                "gc.auto=0",
                "-C",
                str(repository),
                *arguments,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as error:
            raise GitError("Git executable is not installed") from error

        async def read(stream: asyncio.StreamReader | None) -> bytes:
            assert stream is not None
            chunks = bytearray()
            while chunk := await stream.read(8192):
                chunks.extend(chunk)
                if len(chunks) > self.max_output_bytes:
                    raise GitError("Git output exceeded the configured byte limit")
            return bytes(chunks)

        readers = [
            asyncio.create_task(read(process.stdout)),
            asyncio.create_task(read(process.stderr)),
        ]
        try:
            async with asyncio.timeout(self.timeout_seconds):
                stdout, stderr = await asyncio.gather(*readers)
                code = await process.wait()
        except BaseException as error:
            if process.returncode is None:
                process.kill()
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            await process.communicate()
            if isinstance(error, TimeoutError):
                raise GitError("Git command timed out") from error
            raise
        if code not in allowed_codes:
            # Git errors may contain repository data or configured credentials.
            raise GitError(
                "Git operation failed: " + stderr.decode("utf-8", errors="replace").strip()
            )
        return code, stdout.decode("utf-8", errors="replace")

    async def _repository(self, path: Path) -> Path:
        repository = self.access.resolve_repository(path)
        _, root = await self._command(repository, "rev-parse", "--show-toplevel")
        if Path(root.strip()).resolve() != repository:
            raise GitError("Registered path must be the Git working tree root")
        for option in ("--git-dir", "--git-common-dir"):
            _, directory = await self._command(
                repository, "rev-parse", "--path-format=absolute", option
            )
            if not Path(directory.strip()).resolve().is_relative_to(self.access.workspace_root):
                raise GitError("Git metadata is outside the configured workspace")
        code, _ = await self._command(
            repository, "config", "--get-regexp", r"^filter\.", allowed_codes=(0, 1)
        )
        if code == 0:
            raise GitError("Git operations do not support repositories with configured filters")
        return repository

    async def _head(self, repository: Path) -> str | None:
        code, result = await self._command(
            repository, "rev-parse", "--verify", "HEAD", allowed_codes=(0, 128)
        )
        return result.strip() if code == 0 else None

    async def _status(self, repository: Path) -> JsonObject:
        _, output = await self._command(
            repository,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=all",
        )
        records = iter(output.split("\0"))
        files = []
        for record in records:
            if not record:
                continue
            item = {"status": record[:2], "path": record[3:]}
            if "R" in record[:2] or "C" in record[:2]:
                item["original_path"] = next(records)
            files.append(item)
        code, branch = await self._command(
            repository, "symbolic-ref", "--quiet", "--short", "HEAD", allowed_codes=(0, 1)
        )
        return {
            "branch": branch.strip() if code == 0 else None,
            "head": await self._head(repository),
            "detached": code == 1,
            "clean": not files,
            "files": files,
        }

    async def status(self, path: Path) -> JsonObject:
        """Report staged, unstaged, untracked, and conflicted files."""
        return await self._status(await self._repository(path))

    async def diff(self, path: Path, *, staged: bool = False) -> JsonObject:
        """Return a patch without external diff or text conversion programs."""
        repository = await self._repository(path)
        arguments = [
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--ignore-submodules=all",
        ]
        if staged:
            arguments.append("--cached")
        _, patch = await self._command(repository, *arguments, "--")
        return {"patch": patch, "staged": staged}

    async def branches(self, path: Path) -> JsonObject:
        """List local branches and identify the current checkout."""
        repository = await self._repository(path)
        _, output = await self._command(
            repository, "for-each-ref", "--format=%(refname:short)", "refs/heads/"
        )
        state = await self._status(repository)
        return {
            "branches": output.splitlines(),
            "current": state["branch"],
            "detached": state["detached"],
        }

    async def log(self, path: Path, *, limit: int = 20) -> JsonObject:
        """Return recent commit IDs and subjects; unborn repositories have none."""
        if not 1 <= limit <= 100:
            raise GitError("History limit must be between 1 and 100")
        repository = await self._repository(path)
        if await self._head(repository) is None:
            return {"commits": []}
        _, output = await self._command(
            repository, "log", f"-{limit}", "-z", "--format=%H%x00%s", "--no-show-signature"
        )
        fields = output.removesuffix("\0").split("\0")
        return {
            "commits": [
                {"id": fields[index], "subject": fields[index + 1]}
                for index in range(0, len(fields), 2)
            ]
        }

    async def _prepare_mutation(self, path: Path) -> Path:
        if not self.allow_mutations:
            raise GitError("Git mutations are disabled")
        repository = await self._repository(path)
        _, unmerged = await self._command(repository, "ls-files", "--unmerged")
        if unmerged:
            raise GitError("Resolve merge conflicts before modifying Git state")
        for marker in (
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "rebase-merge",
            "rebase-apply",
            "sequencer",
        ):
            _, location = await self._command(repository, "rev-parse", "--git-path", marker)
            if (repository / location.strip()).exists():
                raise GitError("Finish the active merge, rebase, or sequencer operation first")
        return repository

    async def _branch_name(self, repository: Path, name: str) -> None:
        if not name or name.startswith("-") or name == "HEAD" or "@{" in name:
            raise GitError("Invalid local branch name")
        code, _ = await self._command(
            repository, "check-ref-format", "refs/heads/" + name, allowed_codes=(0, 1)
        )
        if code:
            raise GitError("Invalid local branch name")

    async def _paths(self, repository: Path, paths: list[str]) -> list[str]:
        if not paths or len(paths) > 100:
            raise GitError("Provide between 1 and 100 explicit file paths")
        metadata = []
        for option in ("--git-dir", "--git-common-dir"):
            _, directory = await self._command(
                repository, "rev-parse", "--path-format=absolute", option
            )
            metadata.append(Path(directory.strip()).resolve())
        for value in paths:
            path = PurePosixPath(value)
            if (
                not value
                or "\\" in value
                or path.is_absolute()
                or ".." in path.parts
                or any(part.lower() == ".git" for part in path.parts)
            ):
                raise GitError("Git paths must be relative files inside the repository")
            candidate = repository / value
            if (
                not candidate.resolve().is_relative_to(repository)
                or any(candidate.resolve().is_relative_to(directory) for directory in metadata)
                or candidate.is_dir()
                or value == "."
            ):
                raise GitError("Git paths must be relative files inside the repository")
            if not candidate.exists():
                _, tracked = await self._command(repository, "ls-files", "-z", "--", value)
                if tracked.split("\0") != [path.as_posix(), ""]:
                    raise GitError("Deleted paths must identify one tracked file")
        return paths

    async def create_branch(self, path: Path, name: str) -> JsonObject:
        """Create a branch at HEAD without switching or overwriting an existing branch."""
        async with self._locks.setdefault(path.resolve(), asyncio.Lock()):
            repository = await self._prepare_mutation(path)
            await self._branch_name(repository, name)
            if await self._head(repository) is None:
                raise GitError("Create an initial commit before creating another branch")
            await self._command(repository, "branch", "--", name)
            return {"created": name}

    async def checkout(self, path: Path, name: str) -> JsonObject:
        """Switch to an existing local branch only when all files are clean."""
        async with self._locks.setdefault(path.resolve(), asyncio.Lock()):
            repository = await self._prepare_mutation(path)
            await self._branch_name(repository, name)
            if not (await self._status(repository))["clean"]:
                raise GitError("Working tree must be clean before switching branches")
            await self._command(repository, "show-ref", "--verify", "refs/heads/" + name)
            await self._command(repository, "switch", "--no-guess", name)
            return await self._status(repository)

    async def stage(self, path: Path, paths: list[str]) -> JsonObject:
        """Stage explicit files or tracked deletions with literal path matching."""
        async with self._locks.setdefault(path.resolve(), asyncio.Lock()):
            repository = await self._prepare_mutation(path)
            await self._command(repository, "add", "--", *await self._paths(repository, paths))
            return await self._status(repository)

    async def unstage(self, path: Path, paths: list[str]) -> JsonObject:
        """Remove explicit staged changes while preserving working files."""
        async with self._locks.setdefault(path.resolve(), asyncio.Lock()):
            repository = await self._prepare_mutation(path)
            selected = await self._paths(repository, paths)
            if await self._head(repository) is None:
                await self._command(repository, "rm", "--cached", "--force", "--", *selected)
            else:
                await self._command(repository, "restore", "--staged", "--", *selected)
            return await self._status(repository)

    async def commit(self, path: Path, message: str) -> JsonObject:
        """Commit the reviewed index without hooks, signing, or amend."""
        if not message.strip() or len(message) > 500 or "\0" in message:
            raise GitError("Commit message must contain between 1 and 500 characters")
        async with self._locks.setdefault(path.resolve(), asyncio.Lock()):
            repository = await self._prepare_mutation(path)
            state = await self._status(repository)
            if state["detached"]:
                raise GitError("Switch to a local branch before committing")
            _, staged = await self._command(
                repository,
                "diff",
                "--cached",
                "--name-only",
                "--no-ext-diff",
                "--no-textconv",
                "--",
            )
            if not staged:
                raise GitError("There are no staged changes to commit")
            await self._command(
                repository, "-c", "commit.gpgSign=false", "commit", "--no-verify", "-m", message
            )
            return {"commit": await self._head(repository)}
