import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "evaluated_calculator", Path(__file__).parents[1] / "calculator.py"
)
calculator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calculator)


def test_addition_edge_cases():
    assert calculator.add(0, 7) == 7
    assert calculator.add(-4, 3) == -1
    assert calculator.add(2.5, 0.5) == 3
