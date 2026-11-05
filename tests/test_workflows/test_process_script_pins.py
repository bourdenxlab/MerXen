"""Pin the source text of every process ``script:`` (and ``stub:``) section.

Nextflow hashes the source text of a process ``script:`` section
(``task.source``) into every task hash, so any edit to it, even a comment or a
blank line, invalidates the cache of every completed task of that process. The
pins make such an edit a reviewed decision: a refactor (the platform registry of
AT-1 among them) must leave every pin as it is.

Regenerate the pins after an intended, announced change with::

    MERXEN_UPDATE_GOLDEN=1 pytest tests/test_workflows/test_process_script_pins.py

and say in the pull request which caches it invalidates and why. The only
change planned is the QC registration-reference edit of ``modules/qc.nf`` in
AT-6 (docs/plans/atera-support-plan.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO_ROOT / "workflows"
_PINS_PATH = Path(__file__).parent / "golden" / "process_script_pins.json"
_UPDATE = os.environ.get("MERXEN_UPDATE_GOLDEN") == "1"

_SourceParser = Callable[[str], dict[str, dict[str, str]]]


def _nf_files() -> list[Path]:
    """Every Nextflow script of the workflow (modules, main.nf, acceptance)."""
    return sorted(_WORKFLOWS.rglob("*.nf"))


def _relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _current_pins(parse: _SourceParser) -> dict[str, str]:
    """Return ``"<file>::<process>::<section>" -> sha256`` for the tree."""
    pins: dict[str, str] = {}
    for path in _nf_files():
        for process, sections in parse(path.read_text()).items():
            for label in ("script", "stub", "shell", "exec"):
                if label in sections:
                    key = f"{_relative(path)}::{process}::{label}"
                    digest = hashlib.sha256(sections[label].encode()).hexdigest()
                    pins[key] = digest
    return pins


def _write_pins(pins: dict[str, str]) -> None:
    _PINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PINS_PATH.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def current_pins(process_sections: _SourceParser) -> dict[str, str]:
    """Pins of the tree as it is; rewrites the pin file under MERXEN_UPDATE_GOLDEN."""
    pins = _current_pins(process_sections)
    if _UPDATE:
        _write_pins(pins)
    return pins


@pytest.fixture(scope="module")
def pinned(current_pins: dict[str, str]) -> dict[str, str]:
    """The committed pins."""
    assert _PINS_PATH.is_file(), (
        f"{_PINS_PATH.name} is missing: generate it with MERXEN_UPDATE_GOLDEN=1"
    )
    return dict(json.loads(_PINS_PATH.read_text()))


def test_every_process_section_is_pinned(
    current_pins: dict[str, str], pinned: dict[str, str]
) -> None:
    """A new, renamed or removed process section must be pinned explicitly."""
    added = sorted(set(current_pins) - set(pinned))
    removed = sorted(set(pinned) - set(current_pins))
    assert not added and not removed, (
        f"process sections not pinned: {added}; pinned but gone: {removed}. "
        "Regenerate with MERXEN_UPDATE_GOLDEN=1 and review the diff."
    )


def test_process_script_text_is_unchanged(
    current_pins: dict[str, str], pinned: dict[str, str]
) -> None:
    """Every ``script:``/``stub:`` text is byte-identical to its pin."""
    changed = sorted(
        key
        for key in set(current_pins) & set(pinned)
        if current_pins[key] != pinned[key]
    )
    assert not changed, (
        f"process source text changed: {changed}. Nextflow hashes this text, so "
        "the change invalidates the cache of every completed task of these "
        "processes. Only change it for an announced reason (the one planned is the "
        "modules/qc.nf QC edit of AT-6), then regenerate the pins with "
        "MERXEN_UPDATE_GOLDEN=1."
    )


@pytest.mark.parametrize("path", _nf_files(), ids=_relative)
def test_parser_finds_every_process_and_label(
    path: Path, process_sections: _SourceParser
) -> None:
    """The parser sees every process and every source-section label of a file."""
    text = path.read_text()
    processes = process_sections(text)
    assert len(processes) == len(re.findall(r"(?m)^process ", text))
    labels = re.findall(r"(?m)^\s*(?:script|stub|shell|exec):\s*$", text)
    parsed = sum(
        label in sections
        for sections in processes.values()
        for label in ("script", "stub", "shell", "exec")
    )
    assert parsed == len(labels)
    for name, sections in processes.items():
        assert "script" in sections, f"{name} has no script: section"
        assert sections["script"].strip(), f"{name} has an empty script: section"


def test_main_nf_defines_no_process() -> None:
    """Process pins cover modules/ and acceptance/; main.nf only wires them."""
    assert not re.search(r"(?m)^process ", (_WORKFLOWS / "main.nf").read_text())


def test_pinned_inventory(pinned: dict[str, str]) -> None:
    """Today's inventory: 40 processes, 40 ``script:`` and 4 ``stub:`` sections."""
    labels = [key.rsplit("::", 1)[1] for key in pinned]
    assert labels.count("script") == 40
    assert labels.count("stub") == 4
    assert {key for key in pinned if key.endswith("::stub")} == {
        f"workflows/modules/pciseq.nf::{name}::stub"
        for name in (
            "PCISEQ_REFERENCE",
            "PCISEQ_PREPARE",
            "PCISEQ_COMPUTE",
            "PCISEQ_FINALIZE",
        )
    }


def test_parser_keeps_heredoc_braces_and_groovy_prelude(
    process_sections: _SourceParser,
) -> None:
    """A heredoc ``}`` at column 0 does not end the process (modules/qc.nf)."""
    qc = process_sections((_WORKFLOWS / "modules" / "qc.nf").read_text())["QC"]
    script = qc["script"]
    # The Groovy statements before the shell string belong to the section.
    assert "    def " in script[: script.index('"""')]
    assert "\n}\nJSON\n" in script
    assert script.rstrip().endswith('"""')
