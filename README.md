# CodeForge Agent

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
