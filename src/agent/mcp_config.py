"""Operator-controlled MCP connections; repository content cannot change this policy."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    model_validator,
)

ServerName = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,15}$")]
ToolName = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_-]{1,40}$")]


class MCPServerConfig(BaseModel):
    """An explicit allowlist is required before starting or contacting a server."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["stdio", "streamable_http"] = "stdio"
    command: str | None = Field(default=None, min_length=1)
    args: list[str] = Field(default_factory=list, max_length=100)
    cwd: Path | None = None
    env: dict[str, SecretStr] = Field(default_factory=dict)
    url: AnyHttpUrl | None = None
    headers: dict[str, SecretStr] = Field(default_factory=dict)
    allowed_tools: list[ToolName] = Field(default_factory=list, max_length=32)
    read_only: bool = False
    timeout_seconds: float = Field(default=15, gt=0, le=300, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("MCP tool allowlist must not contain duplicates")
        if self.transport == "stdio":
            if not self.command or not self.command.strip() or self.url is not None or self.headers:
                raise ValueError(
                    "Stdio MCP servers require a command and cannot specify HTTP settings"
                )
        elif self.url is None or self.command is not None or self.args or self.env or self.cwd:
            raise ValueError("HTTP MCP servers require a URL and cannot specify process settings")
        return self
