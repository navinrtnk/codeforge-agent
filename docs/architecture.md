# How CodeForge fits together

The FastAPI service registers repositories under one configured workspace. SQLite stores
the index, conversations, agent runs, tool events, verification evidence, and model usage.
Lexical search uses FTS5; optional embeddings add semantic and hybrid retrieval.

HTTP tasks and conversations share the same agent runner. Each model turn sees a bounded
Git snapshot and explicitly registered tools. Read-only is the default; editing and command
execution require service settings and a write-mode run. Bug fixes use the repair loop,
which runs pytest after patch batches and keeps failed or incomplete fixes unverified.
Reviews use a separate bounded prompt and validate findings against changed Git lines.

The MCP entry point exposes native repository tools. Approved external MCP tools are
namespaced, allowlisted, and connected for the lifetime of one agent run. Commands use
the selected local or Docker backend. See [usage](usage.md), [MCP](mcp.md), and
[sandbox boundaries](sandbox.md) for configuration and limits.

The original 20-epic roadmap is implemented. The [evaluation harness](evaluation.md)
provides repeatable regression fixtures; live-model quality still needs representative
evaluation cases for your repositories.
