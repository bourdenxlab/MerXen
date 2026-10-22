"""Tests for the stale assignment-table cleanup script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from spatialdata.models import TableModel

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "cleanup_assignment_tables_and_outputs.py"
)


@pytest.fixture(scope="module")
def cleanup_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cleanup_assignment_tables_and_outputs", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _table() -> ad.AnnData:
    return TableModel.parse(
        ad.AnnData(
            X=np.ones((1, 1), dtype=np.float32),
            var=pd.DataFrame(index=pd.Index(["GeneA"])),
        )
    )


def _write_store(path: Path) -> Path:
    sd.SpatialData(tables={"table": _table(), "table_MOSAIK_proseg": _table()}).write(
        path
    )
    return path


def test_dry_run_keeps_derived_tables(
    tmp_path: Path, cleanup_script: ModuleType
) -> None:
    store = _write_store(tmp_path / "latest.zarr")

    cleanup_script.remove_derived_tables(store, apply=False)

    assert set(sd.read_zarr(store).tables) == {"table", "table_MOSAIK_proseg"}


def test_table_removal_serializes_with_another_store_writer(
    tmp_path: Path,
    cleanup_script: ModuleType,
    store_lock_contention: Any,
) -> None:
    """Applied removal waits for a writer holding the store lock through another
    staged symlink, and keeps the lock from its read until the metadata is
    pruned, so it cannot interleave with a pipeline writer."""
    store = _write_store(tmp_path / "latest.zarr")

    store_lock_contention.run(
        lambda staged: cleanup_script.remove_derived_tables(staged, apply=True),
        store,
        reads=[(cleanup_script.sd, "read_zarr")],
        persists=[(cleanup_script, "_prune_zarr_consolidated_metadata")],
    )

    assert set(sd.read_zarr(store).tables) == {"table"}
