"""Tests for Python symbol extraction."""

from agent.symbols import extract_python_symbols


def test_extract_python_symbols_includes_classes_functions_and_methods() -> None:
    content = """
class RepositoryIndexer(BaseIndexer):
    def index(self, repository: str) -> int:
        def normalize(value: str) -> str:
            return value.strip()
        return 1

    async def refresh(self) -> None:
        pass

async def build_agent(name: str = "codeforge") -> bool:
    return True
""".lstrip()

    symbols = extract_python_symbols(content)

    assert [(symbol.qualified_name, symbol.kind) for symbol in symbols] == [
        ("RepositoryIndexer", "class"),
        ("RepositoryIndexer.index", "method"),
        ("RepositoryIndexer.index.normalize", "function"),
        ("RepositoryIndexer.refresh", "async_method"),
        ("build_agent", "async_function"),
    ]
    assert symbols[0].signature == "class RepositoryIndexer(BaseIndexer)"
    assert symbols[1].signature == "def index(self, repository: str) -> int"
    assert symbols[-1].signature == "async def build_agent(name: str='codeforge') -> bool"
    assert symbols[0].start_line == 1
    assert symbols[0].end_line == 8


def test_extract_python_symbols_returns_empty_for_invalid_python() -> None:
    assert extract_python_symbols("def broken(") == []
