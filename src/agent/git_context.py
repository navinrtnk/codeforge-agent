"""Fresh, redacted, bounded Git snapshots for the beginning of an agent run."""

import asyncio
from pathlib import Path

from agent.git_service import GitError, GitService
from agent.models_api import JsonObject
from agent.repository_access import RepositoryAccessError
from agent.tools import bound_json_object, redact_json_object, serialize_json


class GitContextBuilder:
    """Collect status and patches without making Git a prerequisite for questions."""

    def __init__(
        self,
        service: GitService,
        *,
        max_bytes: int = 16_000,
        max_files: int = 50,
        timeout_seconds: float = 5,
    ) -> None:
        if max_bytes < 512 or max_files < 1 or timeout_seconds <= 0:
            raise ValueError("Invalid Git context limits")
        self.service = service
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.timeout_seconds = timeout_seconds

    async def build(self, path: Path) -> JsonObject | None:
        """Return a best-effort snapshot; plain directories have no Git context."""
        snapshot: JsonObject | None = None
        try:
            async with asyncio.timeout(self.timeout_seconds):
                state = await self.service.status(path)
                files = state["files"]
                snapshot = {
                    "branch": state["branch"],
                    "head": state["head"],
                    "detached": state["detached"],
                    "clean": state["clean"],
                    "files": files[: self.max_files],
                    "total_changed_files": len(files),
                    "files_truncated": len(files) > self.max_files,
                    "staged_patch": {"patch": "", "truncated": False, "available": False},
                    "unstaged_patch": {"patch": "", "truncated": False, "available": False},
                }
                for key, staged in (("staged_patch", True), ("unstaged_patch", False)):
                    if state["clean"]:
                        snapshot[key]["available"] = True
                        continue
                    try:
                        diff = await self.service.diff(path, staged=staged)
                        snapshot[key].update(patch=diff["patch"], available=True)
                    except GitError:
                        # Do not insert raw Git stderr into an automatically sent prompt.
                        snapshot[key]["reason"] = "Git patch inspection unavailable"
        except GitError, RepositoryAccessError, TimeoutError:
            if snapshot is None:
                return None
            for key in ("staged_patch", "unstaged_patch"):
                if not snapshot[key]["available"]:
                    snapshot[key]["reason"] = "Git context collection interrupted"
        return self._bound(redact_json_object(snapshot))

    def _bound(self, snapshot: JsonObject) -> JsonObject:
        """Retain file metadata first, then share remaining bytes between patch previews."""
        patches = {key: snapshot[key]["patch"] for key in ("staged_patch", "unstaged_patch")}
        for key, patch in patches.items():
            snapshot[key].update(patch="", truncated=bool(patch))
        while self._size(snapshot) > self.max_bytes and snapshot["files"]:
            snapshot["files"].pop()
            snapshot["files_truncated"] = True
        for index, (key, patch) in enumerate(patches.items()):
            target = self._size(snapshot) + max(0, self.max_bytes - self._size(snapshot)) // (
                2 - index
            )
            low, high = 0, len(patch)
            while low < high:
                middle = (low + high + 1) // 2
                snapshot[key].update(patch=patch[:middle], truncated=middle < len(patch))
                if self._size(snapshot) <= target:
                    low = middle
                else:
                    high = middle - 1
            snapshot[key].update(patch=patch[:low], truncated=low < len(patch))
        return bound_json_object(snapshot, self.max_bytes)

    @staticmethod
    def _size(snapshot: JsonObject) -> int:
        return len(serialize_json(snapshot).encode("utf-8"))
