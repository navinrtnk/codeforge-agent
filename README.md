# CodeForge Agent

[![tests](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml)

A lightweight AI software engineering agent built with Python and FastAPI.

## Requirements

- Python 3.14.7
- [uv](https://docs.astral.sh/uv/)
- Make (optional)

## Setup

Install the project and its development dependencies:

```bash
uv sync
```

Copy `.env.example` to `.env` to customize the database, workspace, or model
configuration. Environment variables use the `CODEFORGE_` prefix.

Start the development server:

```bash
uv run uvicorn agent.main:app --reload
```

Then visit `http://127.0.0.1:8000/health`. The endpoint returns:

```json
{"status": "ok"}
```

Register an existing repository directory:

```bash
curl -X POST http://127.0.0.1:8000/repositories \
  -H "Content-Type: application/json" \
  -d '{"name":"CodeForge Agent","path":"/absolute/path/to/codeforge-agent"}'
```

List registered repositories:

```bash
curl http://127.0.0.1:8000/repositories
```

## Repository access

Registered repositories must resolve inside `CODEFORGE_WORKSPACE_ROOT`. Repository
file access rejects absolute paths, path traversal, and symlinks that escape the
repository. Generated directories such as `.git`, `.venv`, `node_modules`, and
build caches are excluded from file discovery by default.

Configure exclusions with `CODEFORGE_REPOSITORY_IGNORE_PATTERNS` as a JSON array.
Text reads are limited to `CODEFORGE_MAX_FILE_SIZE_BYTES` bytes and reject binary
or non-UTF-8 content. These controls form the boundary used by future agent tools.

## Repository indexing

Start or refresh a repository index using its registration ID:

```bash
curl -X POST http://127.0.0.1:8000/repositories/REPOSITORY_ID/index
```

Inspect the current persisted index:

```bash
curl http://127.0.0.1:8000/repositories/REPOSITORY_ID/index/status
```

The index stores each text file's language, SHA-256 content hash, and deterministic
line-based chunks. Reindexing skips unchanged files, replaces chunks for changed
files, removes deleted files, and reports unreadable or binary files without
aborting the run. Set `CODEFORGE_INDEX_CHUNK_SIZE_LINES` to change the default
200-line chunk size.

## Code search

Search indexed code using ranked SQLite FTS5 retrieval:

```bash
curl "http://127.0.0.1:8000/repositories/REPOSITORY_ID/search?q=repository+index"
```

Use literal substring matching when exact spelling matters:

```bash
curl "http://127.0.0.1:8000/repositories/REPOSITORY_ID/search?q=RepositoryIndexer&mode=exact&case_sensitive=true"
```

Search Python classes, functions, async functions, and methods by name or
qualified name:

```bash
curl "http://127.0.0.1:8000/repositories/REPOSITORY_ID/symbols?q=RepositoryIndexer.index"
```

Search results are restricted to the requested repository and include file path,
language, line range, and a matching snippet. Python symbols are extracted with
the standard-library AST and refreshed during incremental indexing.

## Model providers

CodeForge Agent has a provider-neutral async model interface with adapters for the
OpenAI Responses API and Anthropic Messages API. Configure one provider in `.env`:

```dotenv
CODEFORGE_MODEL_PROVIDER=openai
CODEFORGE_MODEL_NAME=YOUR_MODEL_ID
CODEFORGE_OPENAI_API_KEY=YOUR_API_KEY
```

For Anthropic, set `CODEFORGE_MODEL_PROVIDER=anthropic` and provide
`CODEFORGE_ANTHROPIC_API_KEY`. Model IDs are intentionally configuration values so
the application does not silently change models. Credentials are validated when a
real provider client is created, not when the FastAPI service starts.

The normalized interface preserves text, parallel tool calls, tool results, stop
reasons, and token usage. A deterministic fake client supports agent-loop tests
without API credentials or network requests.

## Development

Run all quality checks:

```bash
make check
```

The individual commands are:

```bash
make test
make lint
make typecheck
make format
```

If Make is unavailable, run the corresponding `uv run` commands defined in the
`Makefile`.
