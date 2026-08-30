"""Python source symbol extraction."""

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractedSymbol:
    """A declaration extracted from Python source."""

    name: str
    qualified_name: str
    kind: str
    signature: str
    start_line: int
    end_line: int


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[ExtractedSymbol] = []
        self.scopes: list[tuple[str, bool]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [ast.unparse(base) for base in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"
        self._add_symbol(node, "class", signature)
        self.scopes.append((node.name, True))
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        is_async: bool,
    ) -> None:
        is_method = bool(self.scopes and self.scopes[-1][1])
        if is_method:
            kind = "async_method" if is_async else "method"
        else:
            kind = "async_function" if is_async else "function"
        prefix = "async def" if is_async else "def"
        signature = f"{prefix} {node.name}({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        self._add_symbol(node, kind, signature)
        self.scopes.append((node.name, False))
        self.generic_visit(node)
        self.scopes.pop()

    def _add_symbol(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
        signature: str,
    ) -> None:
        qualified_name = ".".join([*(name for name, _ in self.scopes), node.name])
        self.symbols.append(
            ExtractedSymbol(
                name=node.name,
                qualified_name=qualified_name,
                kind=kind,
                signature=signature,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            )
        )


def extract_python_symbols(content: str) -> list[ExtractedSymbol]:
    """Extract classes and functions, returning no symbols for invalid syntax."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    visitor = _SymbolVisitor()
    visitor.visit(tree)
    return visitor.symbols
