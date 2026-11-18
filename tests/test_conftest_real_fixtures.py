"""Hermetic checks of the real-data fixtures in ``tests/conftest.py``.

CI deselects the slow real-data tests, so nothing else exercises
``real_path_module`` (the module-scoped twin of ``real_path``).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

_UNSET_VARIABLE = "MERXEN_TEST_VARIABLE_THAT_IS_NEVER_SET"
_FUNCTION_SCOPED_FIXTURES = {"real_path", "real_aliases", "monkeypatch"}


@pytest.fixture(scope="module")
def module_resolver(
    real_path_module: Callable[[str], Path],
) -> Callable[[str], Path]:
    """A module-scoped fixture requesting the module-scoped resolver."""
    return real_path_module


def test_real_path_module_resolves_in_module_fixture_and_skips_when_unset(
    module_resolver: Callable[[str], Path],
) -> None:
    with pytest.raises(pytest.skip.Exception):
        module_resolver(_UNSET_VARIABLE)


def _fixture_scope(decorator: ast.expr) -> str | None:
    """Return the scope string of a ``pytest.fixture(...)`` decorator, if any."""
    if not isinstance(decorator, ast.Call):
        return None
    for keyword in decorator.keywords:
        if keyword.arg == "scope" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return None


def test_no_wider_scoped_fixture_requests_a_function_scoped_fixture() -> None:
    tests_root = Path(__file__).parent
    offenders: list[str] = []
    for path in sorted(tests_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            scopes = {_fixture_scope(d) for d in node.decorator_list}
            if not scopes & {"module", "session", "package", "class"}:
                continue
            requested = {arg.arg for arg in node.args.args}
            bad = requested & _FUNCTION_SCOPED_FIXTURES
            if bad:
                relative = path.relative_to(tests_root)
                offenders.append(f"{relative}::{node.name} requests {sorted(bad)}")
    assert not offenders, offenders
