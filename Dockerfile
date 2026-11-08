FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM docker:29-cli AS docker-cli
FROM python:3.14.7-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/codeforge
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.14.7-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 codeforge \
    && useradd --uid 10001 --gid codeforge --no-create-home codeforge \
    && mkdir -p /data /workspace \
    && chown codeforge:codeforge /data
COPY --from=build /opt/codeforge/.venv /opt/codeforge/.venv
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
ENV PATH="/opt/codeforge/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/data
WORKDIR /data
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
