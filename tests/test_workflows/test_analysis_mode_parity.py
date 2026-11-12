"""The three ``analysis_mode`` maps must agree on the modes and their platforms.

The mode of a samplesheet row is resolved in three places:

1. ``workflows/main.nf``: ``normalizeAnalysisMode`` (aliases) and
   ``activePlatformsForMode`` (platforms of a canonical mode);
2. ``merxen.io.samplesheet.required_platforms_for_mode``;
3. ``scripts/cleanup_assignment_tables_and_outputs.py``: ``active_platforms``.

The canonical modes must give the same ordered platforms everywhere, and each
map's alias set is pinned, including the divergence known today, so that a mode
added to one map alone fails here until all three have it (AT-4 added ``atera``
to all three).
The Groovy maps are read as text (the golden harness of
``test_platform_config_golden.py`` also evaluates them).
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from merxen.io.samplesheet import required_platforms_for_mode

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLEANUP_SCRIPT = _REPO_ROOT / "scripts" / "cleanup_assignment_tables_and_outputs.py"

CANONICAL: dict[str, tuple[str, ...]] = {
    "paired": ("MERSCOPE", "XENIUM"),
    "merscope": ("MERSCOPE",),
    "xenium": ("XENIUM",),
    "atera": ("ATERA",),
}

# Alias -> canonical mode of each map, as of today. The Groovy map also has the
# one-letter aliases and folds every non-alphanumeric run to "_"; samplesheet.py
# only folds "-" to "_"; the cleanup script does not normalise beyond case and
# surrounding whitespace.
GROOVY_ALIASES: dict[str, str] = {
    "paired": "paired",
    "pair": "paired",
    "both": "paired",
    "merscope": "merscope",
    "merfish": "merscope",
    "m": "merscope",
    "xenium": "xenium",
    "x": "xenium",
    "atera": "atera",
    "a": "atera",
}
SAMPLESHEET_ALIASES: dict[str, str] = {
    alias: mode
    for alias, mode in GROOVY_ALIASES.items()
    if alias not in {"m", "x", "a"}
}
CLEANUP_ALIASES: dict[str, str] = dict(SAMPLESHEET_ALIASES)


def _function_text(text: str, name: str) -> str:
    start = text.index(f"def {name}(")
    end = text.index("\n}\n", start)
    return text[start:end]


def _groovy_aliases() -> dict[str, str]:
    """Alias map of ``normalizeAnalysisMode`` in main.nf."""
    body = _function_text(
        (_REPO_ROOT / "workflows" / "main.nf").read_text(), "normalizeAnalysisMode"
    )
    block = body[
        body.index("def aliases = [") : body.index("]", body.index("def aliases = ["))
    ]
    return dict(re.findall(r'"(\w+)":\s*"(\w+)"', block))


def _groovy_active_platforms() -> dict[str, tuple[str, ...]]:
    """Mode arms of ``activePlatformsForMode`` in main.nf."""
    body = _function_text(
        (_REPO_ROOT / "workflows" / "main.nf").read_text(), "activePlatformsForMode"
    )
    arms = re.findall(r'if \(analysisMode == "(\w+)"\) \{\s*return \[([^\]]*)\]', body)
    assert arms, "no mode arm found in activePlatformsForMode"
    return {mode: tuple(re.findall(r'"(\w+)"', platforms)) for mode, platforms in arms}


@pytest.fixture(scope="module")
def cleanup_script() -> ModuleType:
    """The cleanup script, loaded as a module."""
    spec = importlib.util.spec_from_file_location(
        "cleanup_assignment_tables_and_outputs", _CLEANUP_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cleanup_platforms(script: ModuleType, mode: str) -> tuple[str, ...]:
    return tuple(script.active_platforms({"analysis_mode": mode, "pair_id": "PAIR_A"}))


def test_groovy_alias_map_is_pinned() -> None:
    assert _groovy_aliases() == GROOVY_ALIASES


def test_groovy_canonical_modes_map_to_ordered_platforms() -> None:
    assert _groovy_active_platforms() == CANONICAL
    assert set(GROOVY_ALIASES.values()) == set(CANONICAL)


def test_samplesheet_map_matches_canonical_modes_and_pinned_aliases() -> None:
    for alias, mode in SAMPLESHEET_ALIASES.items():
        assert required_platforms_for_mode(alias) == CANONICAL[mode], alias
    for alias in set(GROOVY_ALIASES) - set(SAMPLESHEET_ALIASES):
        with pytest.raises(ValueError, match="Unknown analysis_mode"):
            required_platforms_for_mode(alias)
    # Only "-" is folded to "_"; other separators are not.
    assert required_platforms_for_mode(" Paired ") == CANONICAL["paired"]
    with pytest.raises(ValueError, match="Unknown analysis_mode"):
        required_platforms_for_mode("both-")


def test_cleanup_map_matches_canonical_modes_and_pinned_aliases(
    cleanup_script: ModuleType,
) -> None:
    for alias, mode in CLEANUP_ALIASES.items():
        assert _cleanup_platforms(cleanup_script, alias) == CANONICAL[mode], alias
        assert (
            _cleanup_platforms(cleanup_script, f" {alias.upper()} ") == CANONICAL[mode]
        )
    for alias in set(GROOVY_ALIASES) - set(CLEANUP_ALIASES):
        with pytest.raises(ValueError, match="Unknown analysis_mode"):
            _cleanup_platforms(cleanup_script, alias)
    with pytest.raises(ValueError, match="Unknown analysis_mode"):
        _cleanup_platforms(cleanup_script, "both-")
    # An empty mode means paired, as in the workflow.
    assert _cleanup_platforms(cleanup_script, "") == CANONICAL["paired"]


@pytest.mark.parametrize("probe", ["atera", "multi", "merscope_xenium"])
def test_no_map_accepts_a_mode_the_others_reject(
    probe: str, cleanup_script: ModuleType
) -> None:
    """A new mode must land in all three maps at once (as AT-4's ``atera`` did)."""
    accepted = {
        "groovy": probe in _groovy_aliases(),
        "samplesheet": _accepts(required_platforms_for_mode, probe),
        "cleanup": _accepts(
            lambda mode: _cleanup_platforms(cleanup_script, mode), probe
        ),
    }
    assert len(set(accepted.values())) == 1, accepted


def _accepts(resolve: Callable[[str], object], mode: str) -> bool:
    try:
        resolve(mode)
    except ValueError:
        return False
    return True
