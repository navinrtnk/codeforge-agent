"""Unified diff contracts: exact content, ranges, line endings, and rejected formats."""

import difflib

import pytest

from agent.patches import PatchError, apply_hunks, parse_unified_diff


def diff(old: str, new: str, old_path: str = "a/file.py", new_path: str = "b/file.py") -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=old_path,
            tofile=new_path,
            n=1,
        )
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("one\ntwo\nthree\n", "one\nnew\nthree\n"),
        ("one\ntwo\n", "insert\none\ntwo\nlast\n"),
        ("first\nsecond\nthird\n", "first\nthird\n"),
        ("value = 1\r\n", "value = 2\r\n"),
        ("你好\n", "こんにちは\n"),
        (
            "".join(f"line {number}\n" for number in range(10)),
            "".join(
                f"changed {number}\n" if number in {1, 8} else f"line {number}\n"
                for number in range(10)
            ),
        ),
    ],
)
def test_exact_hunk_application(old: str, new: str) -> None:
    parsed = parse_unified_diff(diff(old, new))
    assert len(parsed) == 1
    assert apply_hunks(parsed[0], old) == new


def test_create_delete_and_git_headers() -> None:
    created = parse_unified_diff(
        "diff --git a/new.py b/new.py\nnew file mode 100644\n"
        + diff("", "new\n", "/dev/null", "b/new.py")
    )[0]
    assert created.old_path is None and created.new_path == "new.py"
    assert apply_hunks(created, "") == "new\n"
    deleted = parse_unified_diff(diff("old\n", "", "a/old.py", "/dev/null"))[0]
    assert deleted.new_path is None
    assert apply_hunks(deleted, "old\n") == ""


def test_no_final_newline_markers() -> None:
    patch = (
        "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n"
        "-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"
    )
    assert apply_hunks(parse_unified_diff(patch)[0], "old") == "new"


@pytest.mark.parametrize(
    "patch",
    [
        "",
        "not a diff",
        "GIT binary patch\nliteral 0\n",
        "diff --git a/file.py b/file.py\nold mode 100644\nnew mode 100755\n",
        "--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-old\n+new\n",
        "--- a/file.py\n+++ b/file.py\n@@ bad @@\n-old\n+new\n",
        "--- a/file.py\n+++ b/file.py\n@@ -1,2 +1 @@\n-old\n+new\n",
        "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n context\n",
        "--- a/file.py\n+++ b/file.py\n@@ -0 +0 @@\n-old\n+new\n",
        "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new\n+extra\n",
    ],
)
def test_malformed_or_unsupported_patches_are_rejected(patch: str) -> None:
    with pytest.raises(PatchError):
        parse_unified_diff(patch)


def test_duplicate_files_rejected() -> None:
    patch = diff("old\n", "new\n")
    with pytest.raises(PatchError, match="only once"):
        parse_unified_diff(patch + patch)


def test_stale_and_mispositioned_hunks_rejected() -> None:
    patch = diff("old\n", "new\n")
    with pytest.raises(PatchError, match="stale"):
        apply_hunks(parse_unified_diff(patch)[0], "user changed this\n")
    with pytest.raises(PatchError, match="position"):
        apply_hunks(parse_unified_diff(patch.replace("+1 @@", "+2 @@"))[0], "old\n")
