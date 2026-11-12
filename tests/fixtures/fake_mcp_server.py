"""A real MCP subprocess/HTTP server used to exercise CodeForge's client boundary."""

import argparse
import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
    ToolAnnotations,
)
from starlette.responses import Response


async def serve() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--http-token")
    options = parser.parse_args()
    if options.state:
        options.state.write_text(json.dumps({"pid": os.getpid()}))
    calls: list[str] = []

    async def list_tools(
        context: ServerRequestContext[Any], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        }
        if options.mode == "external_ref":
            schema["properties"]["text"] = {"$ref": "file:///etc/passwd"}
        tools = [
            Tool(
                name=name,
                description="Echo a string.",
                input_schema=schema,
                annotations=ToolAnnotations(read_only_hint=True),
            )
            for name in ("echo", "blocked")
        ]
        if options.mode == "duplicate":
            tools.append(tools[0])
        if options.mode == "missing":
            tools = []
        if options.mode == "paginated":
            return ListToolsResult(
                tools=tools[:1] if not params or not params.cursor else tools[1:],
                next_cursor="second" if not params or not params.cursor else None,
            )
        if options.mode == "loop":
            return ListToolsResult(
                tools=[] if params and params.cursor else tools, next_cursor="same"
            )
        return ListToolsResult(tools=tools)

    async def call_tool(
        context: ServerRequestContext[Any], params: CallToolRequestParams
    ) -> CallToolResult:
        calls.append(params.name)
        if options.state:
            state = json.loads(options.state.read_text())
            state["calls"] = calls
            options.state.write_text(json.dumps(state))
        value = (params.arguments or {}).get("text", "")
        if value == "slow":
            await asyncio.sleep(5)
        if value == "disconnect":
            os._exit(0)
        data = {
            "echo": value,
            "model_key_in_env": "CODEFORGE_OPENAI_API_KEY" in os.environ,
            "explicit_env": os.environ.get("MCP_TEST_VALUE"),
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(data))],
            structured_content=data,
            is_error=value == "error",
        )

    server: Server[Any] = Server("fake", on_list_tools=list_tools, on_call_tool=call_tool)
    if options.http:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        if options.state:
            options.state.write_text(
                json.dumps({"pid": os.getpid(), "port": listener.getsockname()[1]})
            )
        app = server.streamable_http_app()

        async def authenticated_app(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http" and options.http_token:
                headers = dict(scope["headers"])
                if headers.get(b"authorization") != f"Bearer {options.http_token}".encode():
                    await Response(status_code=401)(scope, receive, send)
                    return
            await app(scope, receive, send)

        await uvicorn.Server(uvicorn.Config(authenticated_app, log_level="error")).serve(
            sockets=[listener]
        )
    else:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(serve())
