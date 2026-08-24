# CodeForge Agent

A lightweight AI software engineering agent built with Python and FastAPI.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Make (optional)

## Setup

Install the project and its development dependencies:

```bash
uv sync
```

Start the development server:

```bash
uv run uvicorn agent.main:app --reload
```

Then visit `http://127.0.0.1:8000/health`. The endpoint returns:

```json
{"status": "ok"}
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
