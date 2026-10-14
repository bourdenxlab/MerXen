"""Locations and aliases of ``scripts/annotation/remove_mapfirst_outputs.py``.

The results directories are named by the real pair id (R3), so ``--pair``
takes an alias and translates it; ``--outdir`` defaults to the ``.env``
variable ``MERXEN_RESULTS_ROOT``. The behaviour of the removal itself is
tested in ``test_remove_mapfirst_outputs_script.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("spatialdata")

from merxen import local_config, privacy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "annotation" / "remove_mapfirst_outputs.py"


@pytest.fixture(scope="module")
def remover() -> ModuleType:
    """Import the script as a module."""
    spec = importlib.util.spec_from_file_location(
        "remove_mapfirst_outputs_aliases", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path: Path, *pairs: str, outdir: bool = True) -> list[str]:
    argv = [item for pair in pairs for item in ("--pair", pair)]
    argv += ["--quarantine", str(tmp_path / "quarantine")]
    return [*argv, "--outdir", str(tmp_path / "results")] if outdir else argv


def test_an_alias_is_translated_to_the_real_pair_id(
    remover: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "aliases.tsv"
    table.write_text("HD1\thuman_paired_donor\tpair_id\ttoken\tQ1001\n")
    monkeypatch.setenv(privacy.ALIAS_FILE_VAR, str(table))
    privacy.clear_cache()
    args = remover.parse_args(_argv(tmp_path, "HD1", "Q2002"))
    # An alias translates; a real id (or any other name) passes through.
    assert args.pair == ["Q1001", "Q2002"]


def test_an_alias_passes_through_when_the_table_is_disabled(
    remover: ModuleType, tmp_path: Path
) -> None:
    args = remover.parse_args(_argv(tmp_path, "HD1"))
    assert args.pair == ["HD1"]


def test_the_results_root_defaults_to_its_variable(
    remover: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MERXEN_RESULTS_ROOT", str(tmp_path / "published"))
    local_config.clear_cache()
    args = remover.parse_args(_argv(tmp_path, "HD1", outdir=False))
    assert args.outdir == tmp_path / "published"
    # An explicit option wins over the variable.
    explicit = remover.parse_args(_argv(tmp_path, "HD1"))
    assert explicit.outdir == tmp_path / "results"


def test_the_results_root_is_required_without_its_variable(
    remover: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        remover.parse_args(_argv(tmp_path, "HD1", outdir=False))
