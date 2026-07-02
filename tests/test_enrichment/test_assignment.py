"""Tests for per-shape table assignment helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from spatialdata.models import TableModel

from merxen.enrichment.assignment import (
    clone_table_for_region,
    compute_table_from_points_for_shape,
    run_per_shape_assignment_for_dataset,
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
    assert cloned.obs_names.equals(source.obs_names)
    assert cloned.var_names.equals(source.var_names)
    assert cloned.obs["region"].astype(str).tolist() == [
        "MOSAIK_proseg",
        "MOSAIK_proseg",
    ]
    assert cloned.uns["spatialdata_attrs"] == {
        "region": "MOSAIK_proseg",
        "region_key": "region",
        "instance_key": "cell",
    }
    assert source.uns["spatialdata_attrs"]["region"] == "cell_boundaries"


def test_compute_table_from_points_keeps_zero_shape_id() -> None:
    """A geometric shape id of 0 is a real ProSeg cell, not unassigned."""
    points = pd.DataFrame(
        {
            "x": [0.5, 2.5, 10.0],
            "y": [0.5, 0.5, 10.0],
            "gene": ["GeneA", "GeneA", "GeneA"],
        }
    )
    shapes = gpd.GeoDataFrame(
        {
            "cell_id": ["0", "1"],
            "geometry": [box(0.0, 0.0, 1.0, 1.0), box(2.0, 0.0, 3.0, 1.0)],
        },
        geometry="geometry",
    )

    table, summary = compute_table_from_points_for_shape(
        dataset_name="P1_MERSCOPE",
        points_obj=points,
        shape_gdf=shapes,
        shape_id_col="cell_id",
        shape_key="MOSAIK_proseg",
        gene_list=["GeneA"],
        chunk_rows=10,
    )

    assert summary["n_points_assigned"] == 2
    assert table.obs["cell_id"].astype(str).tolist() == ["0", "1"]
    x_matrix = table.X.toarray() if hasattr(table.X, "toarray") else np.asarray(table.X)
    np.testing.assert_array_equal(np.asarray(x_matrix).ravel(), np.array([1, 1]))


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
