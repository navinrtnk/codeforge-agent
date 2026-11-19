.PHONY: install run test test-unit test-integration coverage lint format typecheck check

install:
	uv sync

run:
	uv run uvicorn agent.main:app --reload

test:
	uv run pytest

test-unit:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

coverage:
	uv run pytest --cov=agent --cov-report=term-missing --cov-report=xml:/tmp/codeforge-coverage.xml

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

check: lint typecheck test

