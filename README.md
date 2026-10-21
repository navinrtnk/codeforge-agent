# CodeForge Agent

[![tests](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml)

A local AI software engineering agent that helps you understand a codebase. It
indexes source code, answers repository questions, inspects Git changes, and
remembers conversations across runs.

Built with Python, FastAPI, SQLite, and OpenAI or Anthropic models. It can apply
validated text patches and run pytest; automatic repair loops are planned.

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

Runs default to `"mode":"read_only"`. To edit files, enable
`CODEFORGE_CODE_WRITES_ENABLED=true` in `.env` and send `"mode":"write"` with the
task (or conversation message). The agent can dry-run unified diffs with
`validate_patch`, then use `apply_patch` for bounded edits with rollback on write
failure. Limits: `CODEFORGE_PATCH_MAX_BYTES` and `CODEFORGE_PATCH_MAX_FILES`.
Reindex after editing to refresh search.

Enable `CODEFORGE_TEST_EXECUTION_ENABLED=true` to expose `run_pytest` in write-mode
runs. Ask for all tests or specific paths, such as `tests/test_api.py`; results
are saved under `test_results`. Tests execute trusted repository code locally.
`CODEFORGE_TEST_TIMEOUT_SECONDS` and `CODEFORGE_TEST_MAX_OUTPUT_BYTES` bound execution;
set `CODEFORGE_TEST_PYTHON_EXECUTABLE` to a project interpreter with its dependencies
installed, or leave it unset to use the service's Python. Docker isolation comes later.

Run lint, type checks, and tests with `make check`.
