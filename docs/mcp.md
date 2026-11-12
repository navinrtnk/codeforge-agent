# MCP support

CodeForge can serve its repository tools to an MCP client and use approved tools from
external MCP servers. Both paths use the official Python MCP SDK.

## Connect a client to CodeForge

Run `uv run codeforge-mcp` (or `uv run python -m agent.mcp_server`) over stdio. Configure
your MCP host to launch that command from this project directory. Protocol messages use
stdout; diagnostics use stderr. No model key is needed for ordinary repository tools.

Point `CODEFORGE_DATABASE_URL` at the same database as the HTTP API and set
`CODEFORGE_WORKSPACE_ROOT`. Register and index repositories through the HTTP API first.
Each MCP call takes the registered `repository_id`, plus the normal tool arguments:

| Tool | Arguments beyond `repository_id` |
|---|---|
| `read_file` | `path`, optional `start_line` and `end_line` |
| `search_code` | `query`, optional `mode`, `case_sensitive`, and `limit` |
| `git_diff` | Optional `staged` |
| `run_pytest` | Optional relative test `paths` |

`run_pytest` is advertised only when both `CODEFORGE_CODE_WRITES_ENABLED` and
`CODEFORGE_TEST_EXECUTION_ENABLED` are true. It uses the same interpreter, sandbox,
limits, and persisted test evidence as HTTP agent runs. Other tools remain read-only.
Tool calls are recorded as agent runs and tool events. Calls must stay inside the
configured workspace, including searches of previously indexed content.

The bundled entry point exposes stdio only. Access to that process grants the caller
the configured repository permissions; do not publish it as an unauthenticated network
service. Repository text and tool results are data, not instructions.

## Add external tools to agent runs

Set `CODEFORGE_MCP_SERVERS` to a JSON object in the service environment or `.env`:

```dotenv
CODEFORGE_MCP_SERVERS='{"docs":{"command":"/path/to/python","args":["/path/to/docs_server.py"],"allowed_tools":["search_docs"],"read_only":true}}'
```

For Streamable HTTP, use an explicit URL:

```json
{
  "docs": {
    "transport": "streamable_http",
    "url": "https://your-server.example/mcp",
    "headers": {"Authorization": "Bearer YOUR_SERVER_TOKEN"},
    "allowed_tools": ["search_docs"],
    "read_only": true,
    "timeout_seconds": 15
  }
}
```

The model sees the namespaced name `mcp_docs__search_docs`. Only listed tools are imported;
an empty allowlist starts no connection. By default, external tools are available only
in write-mode runs. Set `read_only: true` only after reviewing the selected tools yourself;
server annotations cannot grant that authority. Connections belong to one agent run and
close on completion, failure, or cancellation. There are no background reconnect loops.

Stdio commands are launched without a shell. Their `env` object supplies explicit values;
model credentials are not inherited automatically. HTTP headers are also explicit, ambient
proxy settings are ignored, and redirects are disabled. Server aliases use letters, digits,
and underscores (up to 16 characters, starting with a letter). Allowlisted tool names use
letters, digits, underscores, or hyphens (up to 40 characters) to fit model function names.

Imported arguments are validated against each tool's JSON Schema before dispatch.
Local schema references are supported; file/URL references and dynamic reference scopes
are rejected. Discovery is bounded, duplicate names and unavailable allowlisted tools fail
closed, and results go through the normal redaction, size limits, and audit trail.
The per-server timeout bounds MCP requests; the agent run still has its overall time budget.

External servers are trusted integrations, not repository sandboxes. A stdio server has
the service's host permissions, and external tools can have effects beyond the active
repository. Approve the server command/URL and each tool deliberately, keep credentials
out of repository files, and use only servers you trust. Server-provided instructions,
descriptions, and results cannot change CodeForge's permissions.

See the [official SDK client guide](https://py.sdk.modelcontextprotocol.io/client/)
and [server guide](https://py.sdk.modelcontextprotocol.io/advanced/low-level-server/).
