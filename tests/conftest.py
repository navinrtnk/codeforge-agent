"""Keep subprocess, database, and HTTP integration checks independently runnable."""

from pathlib import Path

import pytest

UNIT_MODULES = {
    "test_anthropic_client",
    "test_config",
    "test_embeddings",
    "test_model_factory",
    "test_models_api",
    "test_openai_client",
    "test_patches",
    "test_symbols",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if Path(str(item.path)).stem not in UNIT_MODULES:
            item.add_marker(pytest.mark.integration)
