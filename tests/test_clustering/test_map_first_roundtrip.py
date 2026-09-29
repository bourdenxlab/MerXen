"""Every new map_first ``obs`` / ``uns`` field survives h5ad and zarr (R-eng M2).

The clustered H5AD is written by COMPUTE_CPU and copied into the SpatialData
table by FINALIZE (``write_clustered_spatialdata_table``, TableModel-parsed,
zarr). h5ad silently nests ``uns`` keys with ``/``, zarr raises on them, and
neither stores ``None``; categoricals, nullable integers and booleans must
keep their values and categories. These tests write both and compare.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from merxen.annotation.provenance import PROVENANCE_UNS_KEY, AnnotationProvenance
from merxen.clustering.map_first import (
    CLUSTERING_PARAMS_UNS_KEY,
    HIERARCHICAL_UNS_KEY,
    LEIDEN_PROVENANCE_UNS_KEY,
    label_obs_columns,
    run_map_first_hierarchy,
)
from merxen.table_keys import clustered_table_key

from .conftest import Section, make_config, make_section

SAMPLE = "P0001_MERSCOPE"
HIERARCHY_COLUMNS = (
    "broad_class",
    "broad_atlas_label",
    "broad_annotation_score",
    "broad_annotation_score_margin",
    "broad_annotation_n_markers",
    "neuron_split_label",
    "subcluster_label",
    "subcluster_status",
    "hierarchical_cluster",
    "leiden",
    "leiden_broad",
)
NEW_UNS_KEYS = (
    PROVENANCE_UNS_KEY,
    HIERARCHICAL_UNS_KEY,
    f"{CLUSTERING_PARAMS_UNS_KEY}_leiden",
    CLUSTERING_PARAMS_UNS_KEY,
    LEIDEN_PROVENANCE_UNS_KEY,
)


def _clustered(section: Section, tmp_path: Path) -> ad.AnnData:
    clustered, _ = run_map_first_hierarchy(
        section.adata,
        section.labels,
        make_config(tmp_path),
        tmp_path / "hierarchy",
        SAMPLE,
        provenance=section.provenance,
        plots=False,
        stability=True,
        stability_subsamples=2,
    )
    return clustered


def _new_columns(section: Section) -> list[str]:
    return [*label_obs_columns(section.labels.columns), *HIERARCHY_COLUMNS]


def _assert_obs_equal(
    before: pd.DataFrame, after: pd.DataFrame, columns: list[str]
) -> None:
    assert list(after.index) == list(before.index)
    for column in columns:
        want = before[column]
        got = after[column]
        if isinstance(want.dtype, pd.CategoricalDtype):
            assert isinstance(got.dtype, pd.CategoricalDtype), column
            assert list(got.cat.categories) == list(want.cat.categories), column
        else:
            assert got.dtype == want.dtype, (column, got.dtype, want.dtype)
        left = want.astype(object).where(want.notna(), None).tolist()
        right = got.astype(object).where(got.notna(), None).tolist()
        assert right == left, column


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _assert_uns_equal(before: dict[str, Any], after: dict[str, Any]) -> None:
    for key in NEW_UNS_KEYS:
        want = _plain(before[key])
        got = _plain(after[key])
        if isinstance(want, dict):
            assert set(got) == set(want), key
            for name, value in want.items():
                if isinstance(value, float) and math.isnan(value):
                    assert math.isnan(got[name]), (key, name)
                else:
                    assert got[name] == value, (key, name)
        else:
            assert got == want, key
    provenance = AnnotationProvenance.read_from_uns(after)
    assert provenance is not None
    assert provenance == AnnotationProvenance.read_from_uns(before)
    branches = json.loads(str(after[HIERARCHICAL_UNS_KEY]["branch_manifest_json"]))
    assert branches == json.loads(
        str(before[HIERARCHICAL_UNS_KEY]["branch_manifest_json"])
    )


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_h5ad_round_trip_keeps_every_new_field(species: str, tmp_path: Path) -> None:
    section = make_section(species)
    clustered = _clustered(section, tmp_path)
    path = tmp_path / f"{SAMPLE}_clustered.h5ad"
    clustered.write_h5ad(path)
    back = ad.read_h5ad(path)
    columns = _new_columns(section)
    assert set(columns) <= set(back.obs.columns)
    _assert_obs_equal(clustered.obs, back.obs, columns)
    _assert_uns_equal(dict(clustered.uns), dict(back.uns))
    for key in ("X_pca", "X_umap", "spatial"):
        np.testing.assert_allclose(back.obsm[key], clustered.obsm[key])
    assert (back.layers["counts"] != clustered.layers["counts"]).nnz == 0
    assert set(back.obsp) == {"connectivities", "distances"}


def _spatialdata_zarr(section: Section, path: Path) -> None:
    import spatialdata as sd
    from spatialdata.models import ShapesModel, TableModel

    coords = section.adata.obsm["spatial"]
    shapes = gpd.GeoDataFrame(
        {"radius": np.full(len(coords), 5.0)},
        geometry=[Point(x, y) for x, y in coords],
        index=section.adata.obs["instance_id"].to_numpy(),
    )
    table = ad.AnnData(
        X=section.adata.X.copy(),
        obs=section.adata.obs[["instance_id", "region"]].copy(),
        var=section.adata.var.copy(),
    )
    sdata = sd.SpatialData(
        shapes={"cell_boundaries": ShapesModel.parse(shapes)},
        tables={
            "table": TableModel.parse(
                table,
                region="cell_boundaries",
                region_key="region",
                instance_key="instance_id",
            )
        },
    )
    sdata.write(path)


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_tablemodel_zarr_round_trip_keeps_every_new_field(
    species: str, tmp_path: Path
) -> None:
    import spatialdata as sd

    from merxen.analysis.clustering_squidpy import write_clustered_spatialdata_table

    section = make_section(species)
    clustered = _clustered(section, tmp_path)
    # FINALIZE reads the clustered H5AD written by COMPUTE_CPU.
    h5ad_path = tmp_path / f"{SAMPLE}_clustered.h5ad"
    clustered.write_h5ad(h5ad_path)
    from_h5ad = ad.read_h5ad(h5ad_path)
    zarr_path = tmp_path / "latest_spatialdata.zarr"
    _spatialdata_zarr(section, zarr_path)
    written_path, table_key = write_clustered_spatialdata_table(
        zarr_path, from_h5ad, segmentation="proseg_hybrid", table_key_suffix="mapfirst"
    )
    assert written_path == zarr_path
    assert table_key == clustered_table_key("table", "proseg_hybrid", "mapfirst")
    assert table_key.endswith("_clustering_squidpy_mapfirst")
    table = sd.read_zarr(zarr_path).tables[table_key]
    table.obs_names = table.obs_names.astype(str)
    columns = _new_columns(section)
    assert set(columns) <= set(table.obs.columns)
    _assert_obs_equal(clustered.obs, table.obs, columns)
    _assert_uns_equal(dict(clustered.uns), dict(table.uns))
    assert table.uns["spatialdata_attrs"]["instance_key"] == "instance_id"
    # The zarr table's ids are the H5AD's, i.e. the parquet in_table ids.
    labels = section.labels
    in_table = set(labels.loc[labels["in_table"], "cell_id"])
    assert set(table.obs_names) == in_table


def _tree_digest(path: Path) -> dict[str, str]:
    import hashlib

    return {
        str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_map_first_table_never_overwrites_the_legacy_table(tmp_path: Path) -> None:
    """The suffixed key keeps the legacy clustered table byte-identical (R3)."""
    import spatialdata as sd

    from merxen.analysis.clustering_squidpy import write_clustered_spatialdata_table

    section = make_section("human")
    clustered = _clustered(section, tmp_path)
    zarr_path = tmp_path / "latest_spatialdata.zarr"
    _spatialdata_zarr(section, zarr_path)
    # A legacy clustered table under the unsuffixed key.
    legacy = ad.AnnData(
        X=section.adata.X.copy(),
        obs=section.adata.obs[["instance_id", "region", "cell_id"]].copy(),
        uns={
            "merxen_clustering_squidpy": {"table_key": "table"},
            "spatialdata_attrs": dict(section.adata.uns["spatialdata_attrs"]),
            "merxen_hierarchical_clustering": {"enabled": True},
        },
    )
    legacy.obs["broad_class"] = pd.Categorical(["Neurons"] * legacy.n_obs)
    legacy.uns["merxen_clustering_squidpy"]["shape_key"] = "cell_boundaries"
    _, legacy_key = write_clustered_spatialdata_table(
        zarr_path, legacy, segmentation="proseg_hybrid"
    )
    assert legacy_key == clustered_table_key("table", "proseg_hybrid")
    before = _tree_digest(zarr_path / "tables" / legacy_key)

    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_clustered_spatialdata_table(
            zarr_path, clustered, segmentation="proseg_hybrid"
        )
    _, map_first_key = write_clustered_spatialdata_table(
        zarr_path, clustered, segmentation="proseg_hybrid", table_key_suffix="mapfirst"
    )
    assert map_first_key == f"{legacy_key}_mapfirst"
    assert _tree_digest(zarr_path / "tables" / legacy_key) == before
    tables = sd.read_zarr(zarr_path).tables
    assert set(tables[legacy_key].obs["broad_class"].astype(str)) == {"Neurons"}
    assert "hierarchical_cluster" in tables[map_first_key].obs
