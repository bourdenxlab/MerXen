"""Fixtures for workflow configuration smoke tests.

Tests are hermetic (``tests/conftest.py``): no private site config is
visible, so ``nextflow config`` here resolves the null site params. A test that
needs site values uses :func:`example_site_env`, which points both site-config
variables at the tracked templates (``workflows/conf/*.config.example``,
placeholder values that mirror the real layout).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / "workflows"
SITE_CONFIG_EXAMPLE = WORKFLOWS_DIR / "conf" / "site.config.example"
DWIGHT_SITE_CONFIG_EXAMPLE = WORKFLOWS_DIR / "conf" / "dwight.site.config.example"


@pytest.fixture
def example_site_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Point both site-config variables at the tracked templates.

    The variables are set in ``os.environ`` for the test, so the subprocesses it
    starts (``nextflow config``, ``nextflow run``) inherit them.

    Returns:
        Variable name to template path.
    """
    env = {
        "MERXEN_SITE_CONFIG": str(SITE_CONFIG_EXAMPLE),
        "MERXEN_DWIGHT_SITE_CONFIG": str(DWIGHT_SITE_CONFIG_EXAMPLE),
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


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
    "conf/site.config.example",
    "conf/dwight.site.config.example",
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


@pytest.fixture
def parse_params() -> Callable[[str], dict[str, str]]:
    """Return :func:`parse_config_params` (a conftest cannot be imported)."""
    return parse_config_params


@pytest.fixture(scope="session")
def workflow_config_params() -> dict[str, dict[str, str]]:
    """Return the params of each workflow config file (``PARAM_CONFIG_FILES``)."""
    return {
        relative: parse_config_params((WORKFLOWS_DIR / relative).read_text())
        for relative in PARAM_CONFIG_FILES
    }


# Process section labels, as written at four spaces inside a process body.
_SECTION_LABEL = re.compile(
    r"^    (input|output|when|script|stub|shell|exec):\s*$", re.MULTILINE
)
_PROCESS_START = re.compile(r"^process (\w+) \{\s*$")
# Sections whose source text Nextflow runs (and hashes, for ``script:``).
SOURCE_SECTIONS: tuple[str, ...] = ("script", "stub", "shell", "exec")


def _toggle_triple_quote(line: str, open_quote: str | None) -> str | None:
    """Return the triple quote still open after ``line``, or ``None``."""
    for quote in ('"""', "'''"):
        if open_quote in (None, quote) and line.count(quote) % 2 == 1:
            open_quote = None if open_quote == quote else quote
    return open_quote


def parse_process_sections(text: str) -> dict[str, dict[str, str]]:
    """Return ``process -> section -> exact source text`` of a Nextflow file.

    A process starts at ``process NAME {`` (column 0) and ends at the first
    line that is exactly ``}`` outside a triple-quoted string (heredocs inside
    a script may hold a ``}`` at column 0). A section runs from the line after
    its label up to the next label or the end of the process; labels are only
    recognised outside triple-quoted strings. The Groovy statements before the
    opening triple quote belong to the section, as they do to the
    ``task.source`` Nextflow hashes.

    Args:
        text: Source of one ``.nf`` file.

    Returns:
        Process name to section label to section text, in file order.

    Raises:
        AssertionError: If a process is not closed or is defined twice.
    """
    lines = text.splitlines(keepends=True)
    processes: dict[str, dict[str, str]] = {}
    index = 0
    while index < len(lines):
        match = _PROCESS_START.match(lines[index].rstrip("\n"))
        if match is None:
            index += 1
            continue
        name = match.group(1)
        assert name not in processes, f"process {name} defined twice"
        sections: dict[str, str] = {}
        current: str | None = None
        open_quote: str | None = None
        end = index + 1
        while True:
            assert end < len(lines), f"process {name} is not closed"
            line = lines[end]
            if open_quote is None:
                if line.rstrip("\n") == "}":
                    break
                label = _SECTION_LABEL.match(line)
                if label is not None:
                    current = label.group(1)
                    assert current not in sections, f"{name}: two {current}: blocks"
                    sections[current] = ""
                    end += 1
                    continue
            if current is not None:
                sections[current] += line
            open_quote = _toggle_triple_quote(line, open_quote)
            end += 1
        processes[name] = sections
        index = end + 1
    return processes


@pytest.fixture(scope="session")
def process_sections() -> Callable[[str], dict[str, dict[str, str]]]:
    """Return :func:`parse_process_sections` (a conftest cannot be imported)."""
    return parse_process_sections
