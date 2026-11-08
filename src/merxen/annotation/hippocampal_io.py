"""Selective SpatialData reads for the explicit whole-panel annotation profile."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spatialdata import SpatialData

PROFILE = "human_hippocampus_whole_panel_v1"


def read_annotation_store(path: Path | str, table_key: str) -> SpatialData:
    """Read shapes and just the requested table, retaining the backing store.

    A whole-panel store can contain many multi-gigabyte expression tables.
    FINALIZE and MENDER need one table and native shapes; loading unrelated
    tables, images or transcript partitions adds neither validation nor data.
    All writes still use the normal store lock and single-element writer.
    A missing table remains absent, so the caller's existing check applies.
    """
    import anndata as ad
    import spatialdata as sd

    root = Path(path)
    store = sd.read_zarr(root, selection=("shapes",))
    table_path = root / "tables" / table_key
    if table_path.is_dir():
        store.tables[table_key] = ad.read_zarr(table_path)
    return store
