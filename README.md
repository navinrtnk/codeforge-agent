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
