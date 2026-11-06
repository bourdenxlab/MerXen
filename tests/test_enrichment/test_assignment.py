"""Tests for per-shape table assignment helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from spatialdata.models import TableModel

import merxen.enrichment.assignment as assignment_module
from merxen.enrichment.assignment import (
    clone_table_for_region,
    compute_table_from_points_for_shape,
    ensure_shape_has_cell_id,
    run_per_shape_assignment_for_dataset,
    shape_to_existing_table_source,
)


def _source_table() -> ad.AnnData:
    """Build a SpatialData table with existing spatialdata_attrs metadata."""
    adata = ad.AnnData(
        X=np.array([[1, 2], [3, 4]], dtype=np.int64),
        obs=pd.DataFrame(
            {
                "cell": ["cell_a", "cell_b"],
                "region": pd.Categorical(["cell_boundaries", "cell_boundaries"]),
            },
            index=pd.Index(["cell_a", "cell_b"], name="cell"),
        ),
        var=pd.DataFrame(index=pd.Index(["GeneA", "GeneB"], name="gene")),
    )
    return TableModel.parse(
        adata,
        region="cell_boundaries",
        region_key="region",
        instance_key="cell",
    )


def test_clone_table_for_region_retargets_existing_spatialdata_table() -> None:
    """Cloning should update table metadata without changing the count matrix."""
    source = _source_table()

    cloned = clone_table_for_region(source, "MOSAIK_proseg")

    np.testing.assert_array_equal(np.asarray(cloned.X), np.asarray(source.X))
    assert cloned.obs_names.tolist() == ["1", "2"]
    assert cloned.var_names.equals(source.var_names)
    assert cloned.obs["instance_id"].tolist() == [1, 2]
    assert cloned.obs["source_cell_id"].tolist() == ["cell_a", "cell_b"]
    assert cloned.obs["region"].astype(str).tolist() == [
        "MOSAIK_proseg",
        "MOSAIK_proseg",
    ]
    assert cloned.uns["spatialdata_attrs"] == {
        "region": "MOSAIK_proseg",
        "region_key": "region",
        "instance_key": "instance_id",
    }
    assert source.uns["spatialdata_attrs"]["region"] == "cell_boundaries"


def test_compute_table_from_points_uses_positive_shape_ids() -> None:
    """Generated tables should preserve positive canonical shape IDs."""
    points = pd.DataFrame(
        {
            "x": [0.5, 2.5, 10.0],
            "y": [0.5, 0.5, 10.0],
            "gene": ["GeneA", "GeneA", "GeneA"],
        }
    )
    shapes = gpd.GeoDataFrame(
        {
            "instance_id": [1, 2],
            "geometry": [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
        },
        geometry="geometry",
    )

    table, summary = compute_table_from_points_for_shape(
        dataset_name="P1_MERSCOPE",
        points_obj=points,
        shape_gdf=shapes,
        shape_id_col="instance_id",
        shape_key="MOSAIK_proseg",
        gene_list=["GeneA"],
        chunk_rows=10,
    )

    assert summary["n_points_assigned"] == 2
    assert table.obs["instance_id"].tolist() == [1, 2]
    x_matrix = table.X.toarray() if hasattr(table.X, "toarray") else np.asarray(table.X)
    np.testing.assert_array_equal(np.asarray(x_matrix).ravel(), np.array([1, 1]))


@pytest.mark.parametrize(
    ("legacy_column", "legacy_ids", "expected_source_ids", "expected_instance_ids"),
    [
        ("cell", [0, 7], ["0", "7"], [1, 8]),
        (
            "cell_id",
            ["cellpose_2", "cellpose_9"],
            ["cellpose_2", "cellpose_9"],
            [2, 9],
        ),
    ],
)
def test_ensure_shape_has_cell_id_normalizes_legacy_identifiers(
    legacy_column: str,
    legacy_ids: list[object],
    expected_source_ids: list[str],
    expected_instance_ids: list[int],
) -> None:
    """Legacy ProSeg and Cellpose IDs should become stable positive labels."""
    shapes = gpd.GeoDataFrame(
        {
            legacy_column: legacy_ids,
            "geometry": [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
        },
        geometry="geometry",
    )
    sdata = SimpleNamespace(shapes={"legacy_cells": shapes})

    normalized, id_column = ensure_shape_has_cell_id(sdata, "legacy_cells")

    assert id_column == "instance_id"
    assert normalized["source_cell_id"].tolist() == expected_source_ids
    assert normalized["instance_id"].tolist() == expected_instance_ids
    assert normalized.index.tolist() == expected_instance_ids
    assert normalized.index.name == "instance_id"


def test_source_backed_clone_failure_does_not_use_spatial_join(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Known source-table layers should fail rather than fall back to sjoin."""
    sdata = SimpleNamespace(
        points={"transcripts": object()},
        shapes={
            "MOSAIK_proseg": gpd.GeoDataFrame(
                {"cell": ["cell_a"], "geometry": [box(0, 0, 1, 1)]},
                geometry="geometry",
            )
        },
        tables={"table": _source_table()},
    )

    monkeypatch.setattr(
        "merxen.enrichment.assignment.sd.read_zarr",
        lambda path: sdata,
    )

    def _raise_clone_error(*args: object, **kwargs: object) -> ad.AnnData:
        raise ValueError("simulated clone failure")

    def _unexpected_sjoin(*args: object, **kwargs: object) -> tuple[ad.AnnData, dict]:
        raise AssertionError("spatial-join fallback should not be used")

    monkeypatch.setattr(
        "merxen.enrichment.assignment.clone_table_for_region",
        _raise_clone_error,
    )
    monkeypatch.setattr(
        "merxen.enrichment.assignment.compute_table_from_points_for_shape",
        _unexpected_sjoin,
    )

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        run_per_shape_assignment_for_dataset("P1_MERSCOPE", tmp_path / "latest.zarr")


def test_hybrid_assignment_table_is_preserved_during_forced_reassignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generic per-shape assignment must not replace overlap-aware hybrid counts."""
    hybrid_table = object()
    sdata = SimpleNamespace(
        points={
            "transcripts": pd.DataFrame({"x": [0.5], "y": [0.5], "gene": ["GeneA"]})
        },
        shapes={
            "MOSAIK_proseg_hybrid": gpd.GeoDataFrame(
                {"cell_id": ["1"], "geometry": [box(0, 0, 1, 1)]},
                geometry="geometry",
            )
        },
        tables={
            "table": _source_table(),
            "table_MOSAIK_proseg_hybrid": hybrid_table,
        },
    )
    monkeypatch.setattr(
        "merxen.enrichment.assignment.sd.read_zarr",
        lambda path: sdata,
    )

    summaries = run_per_shape_assignment_for_dataset(
        "P1_MERSCOPE",
        tmp_path / "latest.zarr",
        force_rerun=True,
    )

    assert summaries == []
    assert sdata.tables["table_MOSAIK_proseg_hybrid"] is hybrid_table


class _RecordingStore(SimpleNamespace):
    """In-memory store whose element and metadata writes are recorded."""

    def write_element(self, key: str, overwrite: bool = False) -> None:
        del overwrite
        self.written.append(key)

    def write_metadata(self, write_attrs: bool = True) -> None:
        del write_attrs
        self.written.append("metadata")


def test_per_shape_assignment_serializes_with_another_store_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store_lock_contention: Any,
) -> None:
    """Assignment waits for a writer holding the store lock through another
    staged symlink, and keeps the lock from its read until the last write."""
    latest = tmp_path / "latest.zarr"
    latest.mkdir()
    sdata = _RecordingStore(
        points={
            "transcripts": pd.DataFrame({"x": [0.5], "y": [0.5], "gene": ["GeneA"]})
        },
        shapes={
            "MOSAIK_proseg": gpd.GeoDataFrame(
                {"cell": ["cell_a", "cell_b"]},
                geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
            )
        },
        tables={"table": _source_table()},
        attrs={},
        written=[],
    )
    monkeypatch.setattr(assignment_module.sd, "read_zarr", lambda path: sdata)

    summaries = store_lock_contention.run(
        lambda staged: run_per_shape_assignment_for_dataset("P1_MERSCOPE", staged),
        latest,
        reads=[(assignment_module.sd, "read_zarr")],
        persists=[
            (assignment_module, "write_or_replace_element"),
            (assignment_module, "_write_table_element"),
            (assignment_module, "write_spatialdata_metadata"),
        ],
    )

    assert [summary["table_key"] for summary in summaries] == ["table_MOSAIK_proseg"]
    assert sdata.written[-1] == "metadata"


@pytest.mark.parametrize(
    ("shape_key", "source"),
    [
        ("MOSAIK_proseg", "table"),
        ("cell_boundaries", "table"),
        ("merscope_cell_boundaries", "table_original"),
        ("xenium_cell_boundaries", "table_original"),
        ("xenium_nucleus", None),
        ("MOSAIK_cellpose", None),
        ("merscope_cell_boundaries_aligned_nonrigid", None),
        ("xenium_cell_boundaries_aligned_nonrigid", None),
    ],
)
def test_shape_to_existing_table_source(shape_key: str, source: str | None) -> None:
    """Only ProSeg and the vendor cell shapes reuse an existing table."""
    assert shape_to_existing_table_source(shape_key) == source


def _xenium_store() -> _RecordingStore:
    """In-memory XENIUM latest store: vendor cells, vendor nuclei, two tables."""
    shapes = {
        "xenium_cell_boundaries": gpd.GeoDataFrame(
            {"cell_id": ["cell_a", "cell_b"]},
            geometry=[box(0, 0, 2, 2), box(3, 0, 5, 2)],
        ),
        "xenium_nucleus": gpd.GeoDataFrame(
            {"cell_id": ["cell_a", "cell_b"]},
            geometry=[box(0.5, 0.5, 1.5, 1.5), box(3.5, 0.5, 4.5, 1.5)],
        ),
    }
    original = _source_table()
    return _RecordingStore(
        points={
            "transcripts": pd.DataFrame(
                {
                    "x": [1.0, 1.9, 4.0, 4.4, 9.0],
                    "y": [1.0, 1.9, 1.0, 1.2, 9.0],
                    "gene": ["GeneA", "GeneB", "GeneA", "GeneA", "GeneB"],
                }
            )
        },
        shapes=shapes,
        tables={"table": _source_table(), "table_original": original},
        attrs={},
        written=[],
    )


_XENIUM_SUMMARIES = [
    {
        "dataset": "P1_XENIUM",
        "shape_key": "xenium_cell_boundaries",
        "table_key": "table_xenium_cell_boundaries",
        "mode": "copied_from_table_original",
        "n_cells": 2,
        "n_genes": 2,
    },
    {
        "dataset": "P1_XENIUM",
        "shape_key": "xenium_nucleus",
        "n_cells": 2,
        "n_genes": 2,
        "n_points_input": 5,
        "n_points_used": 5,
        "n_points_assigned": 3,
        "pct_assigned": 60.0,
        "table_key": "table_xenium_nucleus",
        "mode": "computed_sjoin",
    },
]


@pytest.mark.parametrize(
    "native_arguments",
    [{}, {"platform": "XENIUM", "segmentation_source": "cellpose_proseg"}],
)
def test_xenium_assignment_still_clones_and_joins_its_nucleus_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_arguments: dict[str, str],
) -> None:
    """AT-6 regression: XENIUM keeps its clone and its xenium_nucleus sjoin."""
    sdata = _xenium_store()
    monkeypatch.setattr(assignment_module.sd, "read_zarr", lambda path: sdata)

    summaries = run_per_shape_assignment_for_dataset(
        "P1_XENIUM", tmp_path / "latest.zarr", **native_arguments
    )

    cleaned = [
        {key: value for key, value in summary.items() if not pd.isna(value)}
        for summary in summaries
    ]
    assert cleaned == _XENIUM_SUMMARIES
    assert sdata.written == [
        "xenium_cell_boundaries",
        "table_xenium_cell_boundaries",
        "xenium_nucleus",
        "table_xenium_nucleus",
        "metadata",
    ]
    nucleus_counts = sdata.tables["table_xenium_nucleus"].X.toarray()
    np.testing.assert_array_equal(nucleus_counts, [[1, 0], [2, 0]])
    registry = sdata.attrs["merxen_schema"]["segmentations"]
    assert registry["original"]["table"] == "table_original"
    assert registry["original"]["shape"] == "xenium_cell_boundaries"


def test_nucleus_row_shapes_are_never_assigned() -> None:
    """Only ATERA's per-nucleus shape is keyed by nucleus row."""
    assert frozenset({"atera_nucleus"}) == assignment_module.NUCLEUS_ROW_SHAPE_KEYS
