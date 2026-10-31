"""Inspect project tool configuration without importing repository code."""

import configparser
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent.repository_access import RepositoryAccess, RepositoryAccessError

AnalysisTool = Literal["ruff", "mypy"]


class AnalysisError(Exception):
    """Invalid analysis configuration or repository targets."""


@dataclass(frozen=True)
class ToolConfiguration:
    """Selected root configuration and whether it supplies mypy target files."""

    file: str | None = None
    has_files: bool = False


def detect_configuration(
    access: RepositoryAccess, repository: Path, tool: AnalysisTool
) -> ToolConfiguration:
    """Follow tool precedence, restricting discovery to the repository root."""
    names = (
        (".ruff.toml", "ruff.toml", "pyproject.toml")
        if tool == "ruff"
        else ("mypy.ini", ".mypy.ini", "pyproject.toml", "setup.cfg")
    )
    try:
        for name in names:
            if not (repository / name).exists():
                continue
            text = access.read_text(repository, Path(name)).content
            if name.endswith(".toml"):
                document = tomllib.loads(text)
                if name == "pyproject.toml":
                    tools = document.get("tool", {})
                    if not isinstance(tools, dict):
                        raise AnalysisError("Invalid project tool configuration")
                    section = tools.get(tool)
                    if not isinstance(section, dict):
                        continue
                else:
                    section = document
                # Ruff's inherited config can escape the root; disallow extension for now.
                if tool == "ruff" and "extend" in section:
                    raise AnalysisError("Ruff configuration extension is not supported")
                return ToolConfiguration(name, bool(section.get("files")))
            parser = configparser.ConfigParser(interpolation=None)
            parser.read_string(text)
            if parser.has_section("mypy"):
                return ToolConfiguration(name, bool(parser.get("mypy", "files", fallback="")))
    except (RepositoryAccessError, configparser.Error, tomllib.TOMLDecodeError) as error:
        raise AnalysisError(f"Unable to read {tool} configuration") from error
    return ToolConfiguration()
