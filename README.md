# CodeForge Agent

[![tests](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml)

A local coding agent that helps you find your way around a repository, review changes,
and fix bugs with test feedback. It works with OpenAI or Anthropic models and keeps
conversation history in SQLite so you can pick up where you left off.

## Get started

You'll need Python 3.14.7, Git, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Set your workspace and model credentials in `.env`:

```dotenv
CODEFORGE_WORKSPACE_ROOT=/absolute/path/to/your/repos
CODEFORGE_MODEL_PROVIDER=openai
CODEFORGE_MODEL_NAME=YOUR_MODEL_ID
CODEFORGE_OPENAI_API_KEY=YOUR_API_KEY
```

For Anthropic, use `anthropic` and `CODEFORGE_ANTHROPIC_API_KEY` instead.
Then start the API:

```bash
uv run uvicorn agent.main:app --reload
```

Open the [API docs](http://127.0.0.1:8000/docs) to register a repository inside your
workspace and index it. Use its ID to ask a question:

```bash
curl -X POST http://127.0.0.1:8000/repositories/REPOSITORY_ID/runs \
  -H "Content-Type: application/json" \
  -d '{"task":"Explain how this project works"}'
```

## What you can do

- Search code by keyword, exact text, symbol, or meaning.
- Ask follow-up questions in a saved conversation.
- Review staged changes or compare a branch with its base.
- Apply patches and retry bug fixes until tests pass or the attempt limit is reached.
- Connect MCP clients to CodeForge, or bring approved external tools into agent runs.

Runs start read-only. Editing, tests, static analysis, and embeddings each need an
explicit setting. Start with the [usage guide](docs/usage.md) when you're ready to enable them.
For isolated execution, see [Docker setup and limits](docs/sandbox.md).
For MCP clients and external servers, see the [MCP guide](docs/mcp.md).

## Development

Run lint, type checks, and tests with `make check`.
