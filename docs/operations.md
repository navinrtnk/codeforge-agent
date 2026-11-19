# Running and maintaining CodeForge

Start locally using the README, or use `docker compose up --build`. The API binds to
localhost in Compose. `GET /health` checks that the service is responding; it does not
test model credentials, repository permissions, or Docker access. Use `/docs` for the
interactive API and `/openapi.json` for the schema. See [usage](usage.md) for common tasks.

## Logs and errors

CodeForge writes JSON events to stderr for requests, agent runs, and returned model
responses. Events include request/run IDs when available. A valid UUID in `X-Request-ID`
is reused; otherwise the service generates one and returns it in the response header.
Prompts, repository content, request bodies, credentials, and exception text are omitted
from these operational logs. Uvicorn and third-party libraries have their own log settings.

API errors use `detail`, `code` (such as `http_404`), `request_id`, and `errors`.
Validation errors include locations and error types, without echoed input values.
Unexpected failures return a generic 500. Existing status codes and HTTP error headers
are preserved. This adds fields to the previous `detail` response, and validation
`detail` is now a string; clients should use `errors` for field locations.

## Usage and cost estimates

`GET /usage` groups returned model responses by provider and model, with call counts,
input/output tokens, known estimated USD costs, and `unpriced_calls`. Usage is persisted
for agent runs, conversations (including summaries), and reviews, even if a later tool
call or response validation fails. Agent usage rows retain the run and HTTP request IDs.
Direct library calls outside these workflows are not automatically metered.

Configure rates per million tokens using the exact provider/model name returned by the
API. Example rates below are illustrative, not current provider prices:

```dotenv
CODEFORGE_MODEL_PRICES='{"openai:YOUR_MODEL_ID":{"input_per_million":"1.00","output_per_million":"4.00"}}'
```

Missing rates produce null costs, not zero. Known subtotals can exclude unpriced calls;
check `unpriced_calls` before using totals. Rates apply when each response is recorded,
so later configuration changes do not rewrite historical estimates. This is not billing:
cached tokens, tiered pricing, retries without returned usage, embedding requests, and
provider-side charges are not reconciled. Pre-existing runs are not backfilled.

## Checks and deployment

`make check` runs lint, types, and all tests. `make test-unit` and `make test-integration`
run the two test groups separately. `make coverage` adds a terminal summary and writes
XML to `/tmp/codeforge-coverage.xml`. Real Docker tests require a built
`codeforge-agent:local` image, Docker access, and `CODEFORGE_DOCKER_TESTS=1`.
CI runs both groups with combined branch coverage, builds the image, exercises the
Docker integration tests, uploads coverage XML, and checks the container's health endpoint.
Coverage measures the host Python process; it does not instrument code inside worker containers.

Back up the SQLite database and workspace before upgrades. Schema creation adds missing
tables and the existing additive run columns; it is not a general migration framework.
Stop writers before copying SQLite files, or use SQLite's online backup facilities.
Keep `/data` on persistent storage and ensure the container user can access repositories.
Test restoration and model/tool settings on a disposable repository before regular use.

## Deployment limits

The API has no authentication or tenant isolation. Run it for trusted local callers,
or put authentication, TLS, access controls, and request limits in front of it before
network exposure. Repository text and tool output are untrusted data. Local tests run
repository code with host permissions; Docker narrows execution but is not a VM boundary.
Access to the Docker socket grants substantial host privileges. Review [sandbox limits](sandbox.md)
and [external MCP trust](mcp.md) before enabling either integration.

There is no distributed job queue, model billing reconciliation, or built-in backup service.
Runs are tied to the service process; a restart can interrupt them. Multiple processes
do not share in-memory repository locks. Use a single service process and perform
backups and monitoring externally for this local-first deployment.
