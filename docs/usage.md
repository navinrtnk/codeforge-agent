# Using CodeForge

The [interactive API docs](http://127.0.0.1:8000/docs) list the request and response
schemas. Register a repository under `CODEFORGE_WORKSPACE_ROOT`, then call
`POST /repositories/REPOSITORY_ID/index` to make its code searchable. Reindex after edits.

## Questions and search

`POST /repositories/REPOSITORY_ID/runs` accepts `{"task":"Your question"}`. Responses
include assistant messages, tool history, status, and persisted verification evidence.
Conversation endpoints let you continue a discussion with recent messages and a summary
of older context; write authority does not carry over to the next message.

`GET /repositories/REPOSITORY_ID/search?q=validation` uses lexical search by default.
Use `mode=exact` for literal text and `case_sensitive=true` when needed. The `/symbols`
endpoint searches Python declarations by name.

For semantic or hybrid search, enable `CODEFORGE_EMBEDDINGS_ENABLED=true`, provide
`CODEFORGE_OPENAI_API_KEY`, and reindex. Indexing sends bounded chunk text to OpenAI and
stores vectors locally, reusing unchanged content. Search and the agent's `search_code`
tool accept `mode=semantic` and `mode=hybrid`. Changing the embedding model requires
reindexing. Set `CODEFORGE_EMBEDDING_MODEL`, `CODEFORGE_EMBEDDING_DIMENSIONS`,
`CODEFORGE_SEARCH_LEXICAL_WEIGHT`, and `CODEFORGE_SEARCH_SEMANTIC_WEIGHT` as needed.

## Editing and verification

To edit, set `CODEFORGE_CODE_WRITES_ENABLED=true` and send `"mode":"write"` with the
task or conversation message. The agent validates unified diffs before bounded patch
application. `CODEFORGE_PATCH_MAX_BYTES` and `CODEFORGE_PATCH_MAX_FILES` limit edits.

Enable `CODEFORGE_TEST_EXECUTION_ENABLED=true` for pytest in write-mode runs.
An empty target list runs the configured suite; relative Python files or directories
select a subset. Results are saved under `test_results`. Empty, skipped-only, cancelled,
and timed-out suites do not verify a fix.

`POST /repositories/REPOSITORY_ID/runs/bug-fix` with `{"task":"Fix the bug"}` enforces
full-suite tests after each patch batch. Failures feed the next repair attempt, up to
`CODEFORGE_BUG_FIX_MAX_ATTEMPTS` (default 3), the model-turn limit, and the run timeout.
Check `verified`, `repair_attempts`, and `error_message`; unverified edits remain available
for inspection.

Enable `CODEFORGE_ANALYSIS_EXECUTION_ENABLED=true` for Ruff and mypy diagnostics in
write-mode runs. Results appear under `analysis_results`. Static analysis does not replace
test verification. Root tool configuration is detected; Ruff config extension is unsupported.

Local execution runs trusted repository code with the service's host permissions.
`CODEFORGE_TEST_PYTHON_EXECUTABLE` and `CODEFORGE_ANALYSIS_PYTHON_EXECUTABLE` can select
project interpreters with their dependencies installed. Tool timeouts and output limits
are configurable. For Docker execution, image dependencies, and isolation limits, use the
[sandbox guide](sandbox.md).

## Code reviews

`POST /repositories/REPOSITORY_ID/reviews` accepts `{}` for unstaged tracked changes,
`{"staged":true}` for the index, or `{"base_ref":"main"}` for committed changes since
the merge base with `main`. It returns validated findings, a summary, risk assessment,
and skipped files. Binary and metadata-only changes require manual review. Oversized
reviews are rejected; adjust `CODEFORGE_REVIEW_MAX_FILES` and
`CODEFORGE_REVIEW_MAX_CONTEXT_BYTES` as needed.
