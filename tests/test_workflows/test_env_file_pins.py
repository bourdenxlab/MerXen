"""Pin the bytes of every conda env file under ``envs/``.

Nextflow names a conda env after the full text of its env file, comments
included, and that name enters the hash of every task that uses the env. Any
edit to an env file, even a reworded comment, therefore builds a new, untested
env and recomputes every completed task of its processes once. The pins make
such an edit a reviewed decision, in the same way as the process-script pins
(``test_process_script_pins.py``).

Regenerate the pins after an intended, announced change with::

    MERXEN_UPDATE_GOLDEN=1 pytest tests/test_workflows/test_env_file_pins.py

and say in the pull request which task caches it invalidates and why.

Changes made so far:

- AT-6: ``environment.mender.yml`` pins ``pynndescent==0.5.13``,
  ``numba==0.55.2`` and ``llvmlite==0.38.1``. Unpinned, a freshly built env
  resolved pynndescent 0.7, which fails to compile under numba 0.55 on inputs
  above 4,096 cells. The pin changes the ``MENDER_COMPUTE`` task hash of every
  platform once (docs/plans/atera-support-plan.md, AT-6).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENVS = _REPO_ROOT / "envs"
_PINS_PATH = Path(__file__).parent / "golden" / "env_file_pins.json"
_UPDATE = os.environ.get("MERXEN_UPDATE_GOLDEN") == "1"


def _current_pins() -> dict[str, str]:
    """Return ``"envs/<file>" -> sha256`` of the file bytes for the tree."""
    return {
        path.relative_to(_REPO_ROOT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(_ENVS.glob("*.yml"))
    }


@pytest.fixture(scope="module")
def current_pins() -> dict[str, str]:
    """Pins of the tree as it is; rewrites the pin file under MERXEN_UPDATE_GOLDEN."""
    pins = _current_pins()
    if _UPDATE:
        _PINS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PINS_PATH.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
    return pins


@pytest.fixture(scope="module")
def pinned(current_pins: dict[str, str]) -> dict[str, str]:
    """The committed pins."""
    assert _PINS_PATH.is_file(), (
        f"{_PINS_PATH.name} is missing: generate it with MERXEN_UPDATE_GOLDEN=1"
    )
    return dict(json.loads(_PINS_PATH.read_text()))


def test_every_env_file_is_pinned(
    current_pins: dict[str, str], pinned: dict[str, str]
) -> None:
    """A new, renamed or removed env file must be pinned explicitly."""
    added = sorted(set(current_pins) - set(pinned))
    removed = sorted(set(pinned) - set(current_pins))
    assert not added and not removed, (
        f"env files not pinned: {added}; pinned but gone: {removed}. "
        "Regenerate with MERXEN_UPDATE_GOLDEN=1 and review the diff."
    )


def test_env_file_text_is_unchanged(
    current_pins: dict[str, str], pinned: dict[str, str]
) -> None:
    """Every env file is byte-identical to its pin."""
    changed = sorted(
        key
        for key in set(current_pins) & set(pinned)
        if current_pins[key] != pinned[key]
    )
    assert not changed, (
        f"env file text changed: {changed}. Nextflow names conda envs after the "
        "file text (comments included), so the change builds a new env and "
        "recomputes every completed task that uses it. Only change it for an "
        "announced reason, keep the edit out of [docs] commits, then regenerate "
        "the pins with MERXEN_UPDATE_GOLDEN=1 and record the reason in this "
        "module's docstring."
    )
