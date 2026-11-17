# Evaluating changes

Run the small, offline regression baseline from the project directory:

```bash
uv run python -m agent.evaluation evaluations/baseline.json --output /tmp/codeforge-baseline
```

The command writes `report.json` and `report.md` and exits with status 1 if any case
fails. The bundled read-only and repair cases use scripted model responses, so they
need no credentials and report zero model tokens. These are harness baselines, not
claims about a model's repair quality. Save reports before and after a change to compare.

The starting scripted baseline is 2/2 successful cases: two model turns and one tool
call for explanation; two turns, three tool calls (including automatic verification),
one repair attempt, and one added/removed line for repair.

Add `--live` to evaluate your configured model instead. This sends fixture content to
the provider and incurs normal API costs. Model sampling and provider behavior can
vary; only scripted runs are deterministic apart from timing. The harness runs cases
in ID order, with a fresh Git repository, index, and in-memory database for each case.
It reuses production agent limits, repair retries, interpreter, and sandbox settings.
External MCP tools, embeddings, Git mutations, and static analysis are disabled for
these evaluations so they cannot introduce unrelated services or modify Git history.

Each manifest case declares an ID, `kind` (`read_only` or `repair`), relative `fixture`
directory, task, and (for offline runs) scripted responses. Live cases can omit `script`.
Read-only cases require `answer_contains`
substrings and must leave repository files unchanged. This is a basic answer check,
not a semantic judge. Repair cases require an `expected_tests` directory containing
pytest tests. Paths must stay under the manifest directory and cannot contain symlinks.
The checked-in manifest shows the format; duplicate case IDs and unknown fields fail
validation before running cases.

Expected tests run in separate repository copies before and after the agent. The agent
never sees them during its run. Repair success requires an initially failing expected
suite, a completed and test-verified repair, changed files, and a passing expected suite.
Timeouts, empty suites, skipped-only suites, and collection errors cannot establish a
successful repair. Expected tests should load fixture modules explicitly by file path,
as the bundled example does. `evaluation_tests` is reserved for grading.

Reports distinguish the agent's verification from independent expected-test verification.
They record status, input/output tokens, model iterations, tool calls (including automatic
verification), repair attempts, changed files, Git patch bytes and added/removed lines,
plus agent and total elapsed time. Binary changes contribute bytes, not line counts.
JSON reports include the manifest fingerprint and selected model, budget, and sandbox
settings so comparisons can identify configuration changes. Credentials are excluded.
Per-case infrastructure failures are reported by exception type without stopping later
cases. Cancellation stops the run and releases temporary resources.

Only evaluate trusted fixtures and tests. Local pytest runs repository code with host
permissions; temporary copies prevent accidental fixture edits but are not a security
boundary. Expected tests are a correctness check, not protection against adversarial
code tampering with the test process. Use the [Docker sandbox](sandbox.md) when needed.
