# CodeForge Agent

[![tests](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml)

A local AI software engineering agent that helps you understand a codebase. It
indexes source code, answers repository questions, inspects Git changes, and
remembers conversations across runs.

Built with Python, FastAPI, SQLite, and OpenAI or Anthropic models. Code editing
and automated, test-verified fixes are planned; Git mutations are disabled by default.

## Run locally

Requires Python 3.14.7, Git, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Edit `.env` with your repository workspace and model credentials:

```dotenv
CODEFORGE_WORKSPACE_ROOT=/absolute/path/to/your/repos
CODEFORGE_MODEL_PROVIDER=openai
CODEFORGE_MODEL_NAME=YOUR_MODEL_ID
CODEFORGE_OPENAI_API_KEY=YOUR_API_KEY
```

For Anthropic, set the provider to `anthropic` and use
`CODEFORGE_ANTHROPIC_API_KEY` instead.

```bash
uv run uvicorn agent.main:app --reload
```

Open [interactive API docs](http://127.0.0.1:8000/docs) to explore all endpoints.

## Examples

**1. Register and index a repository**

The repository must be inside your configured workspace. Replace
`REPOSITORY_ID` below with the `id` returned by registration.

```bash
curl -X POST http://127.0.0.1:8000/repositories \
  -H "Content-Type: application/json" \
  -d '{"name":"my-project","path":"/absolute/path/to/your/repos/my-project"}'

curl -X POST http://127.0.0.1:8000/repositories/REPOSITORY_ID/index
```

**2. Search indexed code**

```bash
curl "http://127.0.0.1:8000/repositories/REPOSITORY_ID/search?q=validation"
```

**3. Ask the agent about the repository**

```bash
curl -X POST http://127.0.0.1:8000/repositories/REPOSITORY_ID/runs \
  -H "Content-Type: application/json" \
  -d '{"task":"Explain how validation works and summarize any uncommitted changes"}'
```

The response includes the answer in the assistant messages, run status, and tool
history. Runs are saved in SQLite; conversation endpoints support follow-up questions.

Run lint, type checks, and tests with `make check`.
