# CodeForge Agent

[![tests](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/navinrtnk/codeforge-agent/actions/workflows/tests.yml)

A lightweight AI software engineering agent built with Python and FastAPI.

## Requirements

- Python 3.14.7
- Git (available on `PATH` for Git tools)
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

## Repository tools

The tool registry is an explicit allowlist. It currently exposes read-only tools
to list repository files, read UTF-8 source files, and search the persisted code
index. Pydantic schemas validate every model-generated argument before execution,
and the trusted application context fixes the active repository so a tool call
cannot select an arbitrary workspace.

Every invocation is recorded against its agent run with sanitized arguments,
result, duration, and error state. Common credentials are redacted before storage
or model consumption, and `CODEFORGE_MAX_TOOL_OUTPUT_BYTES` bounds each result.
Unknown tools, malformed arguments, path traversal, and unexpected execution
failures return safe errors. Git inspection tools are available by default;
Git mutation tools are registered only when explicitly enabled in configuration.

## Agent runs

Create a bounded agent run for a registered repository:

```bash
curl -X POST http://127.0.0.1:8000/repositories/REPOSITORY_ID/runs \
  -H "Content-Type: application/json" \
  -d '{"task":"Explain how repository indexing works"}'
```

The runner sends the task and registered tool schemas to the configured model,
executes requested tools, returns their sanitized results, and repeats until the
model produces a final answer. Every conversation message, provider token count,
and tool event is persisted. List runs at
`/repositories/REPOSITORY_ID/runs` or retrieve one at
`/repositories/REPOSITORY_ID/runs/RUN_ID`.

`CODEFORGE_AGENT_MAX_ITERATIONS`, `CODEFORGE_AGENT_MAX_OUTPUT_TOKENS`, and
`CODEFORGE_AGENT_TIMEOUT_SECONDS` bound model execution. Runs that exceed a limit
or encounter a provider failure are retained with a `failed` status and a safe
error message.

## Conversation memory

Create a discussion attached to a registered repository:

```bash
curl -X POST http://127.0.0.1:8000/repositories/REPOSITORY_ID/conversations \
  -H "Content-Type: application/json" \
  -d '{"title":"Understand validation"}'
```

Append a question using the returned conversation ID:

```bash
curl -X POST http://127.0.0.1:8000/repositories/REPOSITORY_ID/conversations/CONVERSATION_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"content":"How is repository access validated?"}'
```

Send follow-up questions to the same endpoint to continue with earlier context.
Each append executes synchronously and returns a normal agent run with messages,
tool events, and completion status. Context persists across application restarts.
Standalone `/runs` requests remain independent of conversations.

| Endpoint | Behavior |
| --- | --- |
| `GET /repositories/{id}/conversations` | List discussions, with `offset` and `limit` |
| `GET /repositories/{id}/conversations/{conversation_id}` | Inspect metadata and summary |
| `GET /repositories/{id}/conversations/{conversation_id}/messages` | Read original messages, with `offset` and `limit` |

Messages include their run ID and turn number. Historical messages remain stored
once trimmed from model context, including failed turns. Only completed turns
are replayed, and system instructions are rebuilt for each run. Ordered
conversation turns link to the existing run/message records; startup adds two
new tables without changing or deleting existing runs.

`CODEFORGE_CONVERSATION_HISTORY_TOKEN_BUDGET` limits prior context (default 8192).
The provider-neutral estimator conservatively counts one token per serialized
UTF-8 byte plus message overhead; it is not an exact provider tokenizer. This
budget covers historical messages and their summary, excluding the current task,
system prompt, tool schemas, and the current run's new tool exchanges. Trimming
keeps the newest complete turns so tool calls and results are never split.

Older dialogue is summarized using the configured model. Summaries focus on user
goals, preferences, decisions, and open questions; raw tool output is excluded.
`CODEFORGE_CONVERSATION_SUMMARY_MAX_TOKENS` limits each summary response (default
512), and `CODEFORGE_CONVERSATION_SUMMARY_INPUT_TOKEN_BUDGET` caps source excerpt
bytes per summary call (default 8192, excluding fixed instructions). Large sources
are processed in batches with the preceding summary. The checkpoint is saved
only when every batch succeeds. Summaries are redacted and shortened to fit their
reserved share of history context. They are lossy, untrusted memory, and repository
facts must be checked against current files or the index.

A summary failure returns `502`, a summary timeout returns `504`, and an overlapping
append returns `409`. Summary generation and run execution each use
`CODEFORGE_AGENT_TIMEOUT_SECONDS`. Normal failures and request cancellation release
the database claim. Abrupt process termination can leave `is_running` set;
automatic crash recovery is not included. Summary calls add provider usage beyond
the token counts recorded on the subsequent agent run.

## Local Git integration

Agent runs can use `git_status`, `git_diff`, `git_log`, and `git_branches` to
inspect the active repository. `git_diff` accepts `staged=true` to review the
index, and `git_log` accepts a `limit` from 1 to 100. Registered directories must
be Git working tree roots, with Git metadata inside `CODEFORGE_WORKSPACE_ROOT`.
File inspection tools still work for registered directories without Git.

Every standalone run and conversation follow-up automatically collects a fresh
Git snapshot before the first model request. It includes the branch, HEAD,
clean/detached state, changed-file statuses and paths, and previews of staged and
unstaged patches. Untracked files are listed by name without reading their
contents. Deleted and renamed paths are included in Git status. A snapshot is
historical evidence from the start of the run; the agent should use inspection
tools to check details and recheck state before mutations.

Snapshots are redacted before they reach the model or database. They appear as
untrusted data alongside the current task, and are saved as a
`repository_context` block in that run's user message. Conversation history and
summary generation exclude these blocks so follow-ups receive current Git state
rather than replaying older snapshots.

Configure automatic context with:

| Setting | Default | Purpose |
| --- | --- | --- |
| `CODEFORGE_GIT_CONTEXT_MAX_BYTES` | `16000` | Maximum serialized snapshot JSON bytes, excluding fixed prompt framing |
| `CODEFORGE_GIT_CONTEXT_MAX_FILES` | `50` | Maximum changed-file entries in the snapshot |
| `CODEFORGE_GIT_CONTEXT_TIMEOUT_SECONDS` | `5` | Total snapshot collection deadline, within the agent run timeout |

File metadata is prioritized over patch previews. Truncated file lists retain the
total count, and patch previews indicate whether they were truncated or unavailable.
Exceptionally large metadata may be returned as a bounded preview envelope. These
context limits are separate from the Git tools' command-output limits. A failed
patch read preserves collected status, while a failed status read simply omits
automatic context. Non-Git directories, missing Git, configured filters, and
inspection timeouts do not prevent normal agent questions.

Enable local mutations with `CODEFORGE_GIT_ALLOW_MUTATIONS=true`. This adds:

| Tool | Behavior |
| --- | --- |
| `git_create_branch` | Create a named local branch at HEAD without switching |
| `git_checkout` | Switch to an existing local branch when the working tree is clean |
| `git_stage` | Stage explicit relative file paths, including tracked deletions |
| `git_unstage` | Unstage explicit files while preserving working files |
| `git_commit` | Commit the entire staged index with the supplied message |

Branch tools accept `name`, staging tools accept a `paths` array, and the commit
tool accepts `message`. Configure `user.name` and `user.email` locally in each
repository before committing; global Git configuration is not loaded. Review
status and the staged diff before requesting a commit. For example, ask the agent
to stage `src/example.py`, review the staged patch, and commit it with the message
`fix example validation`.

Commands run asynchronously without a shell. `CODEFORGE_GIT_TIMEOUT_SECONDS`
limits each command, and `CODEFORGE_GIT_MAX_OUTPUT_BYTES` limits each captured
output stream. Commands that exceed either limit are stopped and reported as
tool errors. Git events use the same redacted audit history as repository tools.

Branch switches reject staged, unstaged, or untracked changes. Mutations reject
merge conflicts and active merge/rebase operations, and commits reject detached
HEAD and empty indexes. Unstaging is the supported way to reverse staging; it
also works before the initial commit. Hooks, external diff, text conversion,
submodule recursion, signing, and automatic maintenance are disabled. Repositories
with configured Git filters are rejected to prevent filters running during file
comparison or staging. Remote operations and destructive history/worktree reset
operations are not exposed.

Use mutation tools with trusted local repositories in one application worker.
Mutations are serialized per repository inside that worker; external Git clients
and multiple workers are not coordinated. Submodule worktree changes are excluded
from inspection. If a command times out after changing Git state, inspect status
before retrying; the service does not automatically reset or delete working files.

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
