# Command execution and Docker isolation

The command backend runs `pytest`, `ruff`, and `mypy` through a fixed interpreter/module
allowlist. Git inspection and patch application remain application operations. Existing
write-run authorization and test/analysis enable flags still apply to both backends.

`CODEFORGE_COMMAND_EXECUTION_BACKEND=local` is the compatibility default and executes
trusted repository code with the service's host permissions. It is not an isolation boundary.
The allowlist limits entry points, not what Python tests, configuration, or plugins can execute.

## Run with Docker

Requires a Linux Docker Engine 25+ daemon and permission to use it. Build the bundled
application/worker image once:

```bash
docker build -t codeforge-agent:local .
```

Set `CODEFORGE_COMMAND_EXECUTION_BACKEND=docker` when running the API locally. No image
is pulled or built during command execution. Set `CODEFORGE_SANDBOX_IMAGE` for a trusted
custom image with project dependencies preinstalled; its Python must be at
`CODEFORGE_SANDBOX_PYTHON_EXECUTABLE` (default `/opt/codeforge/.venv/bin/python`).
Host test/analysis interpreter settings select local interpreters only; Docker uses this
container interpreter. Install dependencies while building the trusted image, not while
running repository commands.

Each command gets a non-root container with a read-only repository at `/repo`, a read-only
root filesystem, all capabilities dropped, no new privileges, and bounded temporary storage.
Nested bind mounts are excluded. Networking is disabled by default. Test fixtures may write
to `/tmp`; tests or plugins that require writing into the checkout will fail. Mounted files
must be readable by UID/GID 65534. Model credentials and the Docker socket are not passed to
workers. The entire repository is readable, including any secrets stored in its files.

| Setting | Default |
|---|---|
| `CODEFORGE_SANDBOX_ALLOWED_COMMANDS` | `["pytest","ruff","mypy"]` |
| `CODEFORGE_SANDBOX_CPUS` | 1 |
| `CODEFORGE_SANDBOX_MEMORY_MB` | 512 (swap disabled) |
| `CODEFORGE_SANDBOX_PIDS_LIMIT` | 128 |
| `CODEFORGE_SANDBOX_TMPFS_MB` | 128 |
| `CODEFORGE_SANDBOX_TIMEOUT_SECONDS` | 60 |
| `CODEFORGE_SANDBOX_NETWORK_ENABLED` | false |

The smaller of the sandbox and tool timeout applies. Local execution also enforces the
allowlist and timeout; CPU, memory, PID, and filesystem isolation require Docker. Cleanup
has an additional bounded five-second budget. Cancellation waits for a pending container
creation before removal; timeout, failure, and success all remove the worker and its anonymous
volumes. Cleanup failure prevents verified success and logs the container name for inspection.
Daemon unavailability or abrupt service/host termination can leave containers behind:
inspect `docker ps -a --filter name=codeforge-sandbox-` and remove the identified containers.
There is no automatic fallback to local execution.

## Run the application with Compose

`docker compose up --build` starts the API on localhost with persistent database storage.
Workspace contents come from `CODEFORGE_WORKSPACE_ROOT` and appear at `/workspace` inside
the API. Register repositories using that container path. The base setup does not mount a
Docker socket. Execution remains disabled until the existing execution flags are enabled.

To use the host Docker daemon from the containerized API, opt in explicitly:

```bash
export CODEFORGE_WORKSPACE_ROOT=/absolute/path/to/workspace
export CODEFORGE_DOCKER_GID=$(stat -c %g /var/run/docker.sock)
docker compose -f compose.yaml -f compose.sandbox.yaml up --build
```

The host workspace setting maps `/workspace` back to the daemon's bind-mount source.
`CODEFORGE_DOCKER_HOST_WORKSPACE_ROOT` can provide the same mapping for other deployments;
without it, the service and daemon must see the repository at the same absolute path.
The API user (10001) needs read access to the workspace and write access for patching.
Keep model credentials in runtime configuration; `.env` and host environments are excluded
from the image build context.

Docker containers share a kernel and are not virtual machines. The API and daemon are trusted;
mounting the Docker socket gives the API control over that daemon and can grant host-level
privileges. Run the service only for trusted callers, keep Docker and the kernel updated,
and use a dedicated host/VM for hostile repositories. Allowing networking adds outbound data
access. CPU/memory controls depend on the daemon's cgroup support.

See the [Docker run reference](https://docs.docker.com/reference/cli/docker/container/run/)
and [resource constraints](https://docs.docker.com/engine/containers/resource_constraints/).

To run the real isolation tests after building the image:

```bash
CODEFORGE_DOCKER_TESTS=1 uv run pytest tests/test_docker_sandbox.py
```
