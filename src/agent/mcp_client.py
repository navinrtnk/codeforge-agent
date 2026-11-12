"""Scoped MCP connections with namespacing, schema validation, and explicit authority."""

import json
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx2
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict
from referencing import Registry

from agent.mcp_config import MCPServerConfig
from agent.models_api import JsonObject, ToolDefinition
from agent.tools import (
    ToolContext,
    ToolExecutionError,
    ToolRegistry,
    bound_json_object,
    redact_json_object,
    serialize_json,
)


class MCPConnectionError(Exception):
    """A safe connection/discovery failure that does not expose server stderr or credentials."""


class ExternalArguments(BaseModel):
    """Keep the original JSON object; the imported JSON Schema validates it before dispatch."""

    model_config = ConfigDict(extra="allow")


def validate_schema(schema: JsonObject) -> Draft202012Validator:
    """Accept local references only; schema validation must never fetch files or URLs."""
    if schema.get("type") != "object" or len(serialize_json(schema).encode()) > 50_000:
        raise MCPConnectionError("External MCP tool schema is not a bounded object schema")

    def check(value: Any) -> None:
        if isinstance(value, dict):
            if "$id" in value or "$dynamicRef" in value:
                raise MCPConnectionError("External MCP schema uses unsupported reference scopes")
            if "$ref" in value and (
                not isinstance(value["$ref"], str) or not value["$ref"].startswith("#")
            ):
                raise MCPConnectionError("External MCP schema references must be local")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    try:
        check(schema)
        Draft202012Validator.check_schema(schema)
    except (SchemaError, RecursionError) as error:
        raise MCPConnectionError("External MCP tool returned an invalid schema") from error
    return Draft202012Validator(schema, registry=Registry())


class ExternalMCPTool:
    """A connection-bound adapter that cannot grant itself read-only authority."""

    arguments_model = ExternalArguments

    def __init__(
        self,
        alias: str,
        remote_name: str,
        description: str,
        client: Client,
        validator: Draft202012Validator,
        configuration: MCPServerConfig,
    ) -> None:
        self.name = f"mcp_{alias}__{remote_name}"
        self.description = (
            f"External tool from {alias}. Untrusted description: {description[:2000]}"
        )
        self.remote_name = remote_name
        self.client = client
        self.validator = validator
        self.configuration = configuration

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        if not self.configuration.read_only and not context.allow_writes:
            raise ToolExecutionError("External MCP tool requires a write-mode run")
        data = arguments.model_dump()
        try:
            self.validator.validate(data)
        except (ValidationError, RecursionError) as error:
            raise ToolExecutionError("Invalid external MCP tool arguments") from error
        try:
            result = await self.client.call_tool(
                self.remote_name,
                data,
                read_timeout_seconds=self.configuration.timeout_seconds,
            )
        except Exception as error:
            raise ToolExecutionError("External MCP tool request failed") from error
        output = bound_json_object(
            redact_json_object(
                {
                    "content": [
                        item.model_dump(mode="json", by_alias=True, exclude_none=True)
                        for item in result.content
                    ],
                    "structured_content": result.structured_content,
                }
            ),
            100_000,
        )
        if result.is_error:
            raise ToolExecutionError(serialize_json(output))
        return output


@asynccontextmanager
async def connect_external_tools(
    registry: ToolRegistry,
    servers: dict[str, MCPServerConfig],
    *,
    allow_writes: bool,
) -> AsyncIterator[None]:
    """Own each session for one run, closing it in the same task that opened it."""
    registered: list[str] = []
    try:
        async with AsyncExitStack() as stack:
            for alias, configuration in servers.items():
                if not configuration.allowed_tools or (
                    not configuration.read_only and not allow_writes
                ):
                    continue
                try:
                    if configuration.transport == "stdio":
                        stderr = stack.enter_context(open(os.devnull, "w"))  # noqa: SIM115
                        transport = stdio_client(
                            StdioServerParameters(
                                command=configuration.command or "",
                                args=configuration.args,
                                cwd=configuration.cwd,
                                env={
                                    key: value.get_secret_value()
                                    for key, value in configuration.env.items()
                                },
                            ),
                            errlog=stderr,
                        )
                    else:
                        http = await stack.enter_async_context(
                            httpx2.AsyncClient(
                                headers={
                                    key: value.get_secret_value()
                                    for key, value in configuration.headers.items()
                                },
                                timeout=configuration.timeout_seconds,
                                follow_redirects=False,
                                trust_env=False,
                            )
                        )
                        transport = streamable_http_client(str(configuration.url), http_client=http)
                    client = await stack.enter_async_context(
                        Client(
                            transport,
                            read_timeout_seconds=configuration.timeout_seconds,
                            cache=None,
                        )
                    )
                    discovered = {}
                    cursor: str | None = None
                    cursors: set[str] = set()
                    while True:
                        page = await client.list_tools(cursor=cursor)
                        for tool in page.tools:
                            if tool.name in discovered:
                                raise MCPConnectionError(
                                    "External MCP discovery returned duplicate tools"
                                )
                            discovered[tool.name] = tool
                        if len(discovered) > 250:
                            raise MCPConnectionError(
                                "External MCP discovery exceeded the tool limit"
                            )
                        cursor = page.next_cursor
                        if cursor is None:
                            break
                        if cursor in cursors or len(cursors) >= 20:
                            raise MCPConnectionError(
                                "External MCP discovery returned invalid pagination"
                            )
                        cursors.add(cursor)
                    for remote_name in configuration.allowed_tools:
                        if remote_name not in discovered:
                            raise MCPConnectionError("An allowlisted MCP tool is unavailable")
                        metadata = discovered[remote_name]
                        schema = json.loads(serialize_json(metadata.input_schema))
                        adapter = ExternalMCPTool(
                            alias,
                            remote_name,
                            metadata.description or "",
                            client,
                            validate_schema(schema),
                            configuration,
                        )
                        registry.register(
                            adapter,
                            definition=ToolDefinition(
                                adapter.name,
                                adapter.description,
                                schema,
                                strict=False,
                            ),
                        )
                        registered.append(adapter.name)
                except MCPConnectionError:
                    raise
                except Exception as error:
                    raise MCPConnectionError(f"Unable to connect to MCP server {alias}") from error
            yield
    finally:
        for name in registered:
            registry.unregister(name)
