"""Fixtures for workflow configuration smoke tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.fixture
def dwight_config_text() -> str:
    """Return the workstation execution profile as source text."""
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "workflows" / "conf" / "dwight.config").read_text()


@pytest.fixture
def combined_config_text(dwight_config_text: str) -> str:
    """Return base scientific defaults plus the default workstation profile."""
    repo_root = Path(__file__).resolve().parents[2]
    base_text = (repo_root / "workflows" / "nextflow.config").read_text()
    return f"{base_text}\n{dwight_config_text}"


# Workflow config files that define params, relative to ``workflows/``.
PARAM_CONFIG_FILES: tuple[str, ...] = (
    "nextflow.config",
    "conf/dwight.config",
    "conf/annotation.config",
    "conf/dwight.annotation.config",
)


def _params_blocks(text: str) -> list[str]:
    """Return the bodies of every ``params { ... }`` block of a config."""
    bodies = []
    for match in re.finditer(r"(?m)^\s*params\s*\{", text):
        depth = 1
        index = match.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            index += 1
        bodies.append(text[match.end() : index - 1])
    return bodies


def parse_config_params(text: str) -> dict[str, str]:
    """Return ``name -> raw value text`` of the params a config defines.

    Covers ``params { name = value }`` blocks and ``params.name = value``
    lines. Values are the raw Groovy text after ``=``, trailing comments
    included.

    Args:
        text: Nextflow config source.

    Returns:
        Param name to raw value text.

    Raises:
        AssertionError: If a param is defined twice in the same file.
    """
    assignments: list[tuple[str, str]] = []
    for body in _params_blocks(text):
        assignments.extend(re.findall(r"(?m)^[ \t]*(\w+)\s*=\s*(.*)$", body))
    assignments.extend(re.findall(r"(?m)^\s*params\.(\w+)\s*=\s*(.*)$", text))
    params: dict[str, str] = {}
    for name, value in assignments:
        assert name not in params, f"param {name} defined twice"
        params[name] = value.strip()
    return params


@pytest.fixture(scope="session")
def workflow_config_params() -> dict[str, dict[str, str]]:
    """Return the params of each workflow config file (``PARAM_CONFIG_FILES``)."""
    workflows = Path(__file__).resolve().parents[2] / "workflows"
    return {
        relative: parse_config_params((workflows / relative).read_text())
        for relative in PARAM_CONFIG_FILES
    }
