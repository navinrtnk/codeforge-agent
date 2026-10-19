"""Strict unified diffs with exact hunk matching and no fuzzy application."""

import re
from dataclasses import dataclass


class PatchError(Exception):
    """A rejected patch or recoverable application failure."""


@dataclass(frozen=True)
class PatchLine:
    """One context, removed, or added line, including its original line ending."""

    kind: str
    text: str


@dataclass(frozen=True)
class Hunk:
    """An exact old/new line-range replacement."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[PatchLine, ...]


@dataclass(frozen=True)
class FilePatch:
    """A modification, creation, or deletion of one UTF-8 file."""

    old_path: str | None
    new_path: str | None
    hunks: tuple[Hunk, ...]

    @property
    def path(self) -> str:
        """Return the effective repository-relative target."""
        assert self.new_path is not None or self.old_path is not None
        return self.new_path if self.new_path is not None else str(self.old_path)


HEADER = re.compile(r"^@@ -(\d{1,9})(?:,(\d{1,9}))? \+(\d{1,9})(?:,(\d{1,9}))? @@(?: .*)?$")
NO_NEWLINE = "\\ No newline at end of file"


def split_lines(text: str) -> list[str]:
    """Split only on LF, preserving CRLF and a final line without a newline."""
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _path(header: str, prefix: str) -> str | None:
    value = header.rstrip("\r\n").split("\t", 1)[0]
    if value == "/dev/null":
        return None
    if not value or value.startswith('"'):
        raise PatchError("Quoted or empty diff paths are not supported")
    return value.removeprefix(prefix)


def parse_unified_diff(patch: str) -> tuple[FilePatch, ...]:
    """Parse ordinary text diffs; reject binary, rename, and permission changes."""
    if "\0" in patch:
        raise PatchError("Patches must be UTF-8 text without null bytes")
    lines = split_lines(patch)
    files: list[FilePatch] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip("\r\n")
        if line.startswith(("diff --git ", "index ")) or line in {
            "new file mode 100644",
            "deleted file mode 100644",
        }:
            index += 1
            continue
        if (
            not line.startswith("--- ")
            or index + 1 >= len(lines)
            or not lines[index + 1].startswith("+++ ")
        ):
            raise PatchError("Expected unified diff file headers; unsupported patch format")
        old_path = _path(lines[index][4:], "a/")
        new_path = _path(lines[index + 1][4:], "b/")
        if old_path is None and new_path is None:
            raise PatchError("A patch must name a source or destination file")
        if old_path is not None and new_path is not None and old_path != new_path:
            raise PatchError(
                "Renames are not supported; use separate deletion and creation patches"
            )
        index += 2
        hunks: list[Hunk] = []
        while index < len(lines) and lines[index].startswith("@@"):
            match = HEADER.fullmatch(lines[index].rstrip("\r\n"))
            if match is None:
                raise PatchError("Malformed hunk header")
            old_start, old_count, new_start, new_count = (
                int(value) if value is not None else 1 for value in match.groups()
            )
            if (old_count and not old_start) or (new_count and not new_start):
                raise PatchError("Nonempty hunk ranges start at line one or later")
            index += 1
            body: list[PatchLine] = []
            old_seen = new_seen = 0
            while old_seen < old_count or new_seen < new_count:
                if index >= len(lines) or lines[index][:1] not in {" ", "-", "+"}:
                    raise PatchError("Hunk contents do not match their declared line counts")
                current = lines[index]
                body.append(PatchLine(current[0], current[1:]))
                old_seen += current[0] in {" ", "-"}
                new_seen += current[0] in {" ", "+"}
                index += 1
                if index < len(lines) and lines[index].rstrip("\r\n") == NO_NEWLINE:
                    body[-1] = PatchLine(body[-1].kind, body[-1].text.removesuffix("\n"))
                    index += 1
                if old_seen > old_count or new_seen > new_count:
                    raise PatchError("Hunk contents exceed their declared line counts")
            hunks.append(Hunk(old_start, old_count, new_start, new_count, tuple(body)))
        if not hunks or not any(line.kind in {"-", "+"} for hunk in hunks for line in hunk.lines):
            raise PatchError("Each file patch must contain a changed text hunk")
        files.append(FilePatch(old_path, new_path, tuple(hunks)))
    if not files:
        raise PatchError("Patch contains no file changes")
    if len({file.path for file in files}) != len(files):
        raise PatchError("A file may appear only once in a patch")
    return tuple(files)


def apply_hunks(patch: FilePatch, original: str) -> str:
    """Return the full resulting text or reject stale or overlapping hunks."""
    source = split_lines(original)
    output: list[str] = []
    cursor = 0
    for hunk in patch.hunks:
        position = hunk.old_start if hunk.old_count == 0 else hunk.old_start - 1
        if not cursor <= position <= len(source):
            raise PatchError("Hunk ranges overlap or exceed the current file")
        output.extend(source[cursor:position])
        cursor = position
        new_position = hunk.new_start if hunk.new_count == 0 else hunk.new_start - 1
        if new_position != len(output):
            raise PatchError("New hunk position does not match the resulting file")
        for line in hunk.lines:
            if line.kind in {" ", "-"}:
                if cursor >= len(source) or source[cursor] != line.text:
                    raise PatchError("Patch is stale: context or removed lines do not match")
                cursor += 1
            if line.kind in {" ", "+"}:
                output.append(line.text)
    output.extend(source[cursor:])
    if any(not line.endswith("\n") for line in output[:-1]):
        raise PatchError("A no-newline marker is only valid at the end of a file")
    result = "".join(output)
    if patch.new_path is None and result:
        raise PatchError("A deletion patch must remove the complete file")
    return result
