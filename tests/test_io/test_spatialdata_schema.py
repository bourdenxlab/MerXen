"""Tests for the versioned MerXen SpatialData storage contract."""

from __future__ import annotations

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from scipy import sparse
from shapely.geometry import box
from spatialdata.models import PointsModel, ShapesModel, TableModel

from merxen.io.spatialdata_io import (
    prepare_source_spatialdata_contract,
    upgrade_spatialdata_contract_in_memory,
)
from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    MERXEN_SCHEMA_ATTR,
    MERXEN_SCHEMA_VERSION,
    PROSEG_INTERNAL_ID_COLUMN,
    SpatialDataContractError,
    canonical_instance_series,
    choose_primary_points_key,
    validate_merxen_schema,
    with_stable_transcript_ids,
)


def _raw_proseg_spatialdata() -> sd.SpatialData:
    """Build a minimal zero-based ProSeg object before MerXen normalization."""
    points_df = pd.DataFrame(
        {
            "transcript_id": [0, 0, 0],
            "x": [0.5, 2.5, 2.7],
            "y": [0.5, 0.5, 0.7],
            "z": np.zeros(3, dtype=np.float32),
            "gene": ["GeneA", "GeneA", "GeneB"],
            "assignment": pd.Series([0, 1, pd.NA], dtype="UInt32"),
            "background": [False, False, True],
        }
    )
    points = PointsModel.parse(
        dd.from_pandas(points_df, npartitions=2),
        coordinates={"x": "x", "y": "y", "z": "z"},
        feature_key="gene",
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "cell": np.asarray([0, 1], dtype=np.uint32),
                "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1)],
            },
            geometry="geometry",
            index=pd.Index([0, 1], name="cell"),
        )
    )
    obs = pd.DataFrame(
        {
            "cell": np.asarray([0, 1], dtype=np.uint32),
            "original_cell_id": ["10", "20"],
            "region": pd.Categorical(["cell_boundaries", "cell_boundaries"]),
        },
        index=pd.Index(["0", "1"], name="cell_index"),
    )
    var = pd.DataFrame(index=pd.Index(["GeneA", "GeneB"], name="gene"))
    table = TableModel.parse(
        ad.AnnData(X=sparse.csr_matrix([[1, 0], [0, 1]]), obs=obs, var=var),
        region="cell_boundaries",
        region_key="region",
        instance_key="cell",
    )
    return sd.SpatialData(
        points={"transcripts": points},
        shapes={"cell_boundaries": shapes},
        tables={"table": table},
    )


def test_canonical_instance_series_rejects_reserved_zero() -> None:
    """Operational cell identifiers must leave zero available for raster background."""
    with pytest.raises(SpatialDataContractError, match="positive"):
        canonical_instance_series(pd.Series([0, 1], dtype="uint64"))


def test_primary_points_key_uses_schema_instead_of_mapping_order() -> None:
    """Consumers should resolve the registered native transcript element."""
    sdata_obj = type(
        "SpatialDataLike",
        (),
        {
            "attrs": {
                MERXEN_SCHEMA_ATTR: {
                    "primary_points": "transcripts",
                }
            },
            "points": {
                "transcripts_aligned_nonrigid": object(),
                "transcripts": object(),
            },
        },
    )()

    assert choose_primary_points_key(sdata_obj) == "transcripts"


def test_stable_transcript_ids_are_positive_and_preserve_source_ids() -> None:
    """Source transcript IDs should remain available beside stable row IDs."""
    points = dd.from_pandas(
        pd.DataFrame(
            {
                "transcript_id": [99, 98, 97],
                "x": [1.0, 2.0, 3.0],
            }
        ),
        npartitions=2,
    )

    result = with_stable_transcript_ids(
        points,
        preserve_existing=True,
    ).compute()

    assert result["transcript_id"].tolist() == [1, 2, 3]
    assert result["source_transcript_id"].tolist() == [99, 98, 97]


def test_upgrade_maps_proseg_ids_and_repairs_duplicate_transcript_ids() -> None:
    """A raw ProSeg object should gain one positive ID contract across elements."""
    sdata_obj = _raw_proseg_spatialdata()

    changed = upgrade_spatialdata_contract_in_memory(
        sdata_obj,
        platform="MERSCOPE",
        quality_column_alias="transcript_score",
    )

    assert changed
    points = sdata_obj.points["transcripts"].compute().sort_values("transcript_id")
    assert points["transcript_id"].tolist() == [1, 2, 3]
    assert points["assignment"].tolist() == [10, 20, pd.NA]
    assert points[PROSEG_INTERNAL_ID_COLUMN].tolist() == [0, 1, pd.NA]

    shapes = sdata_obj.shapes["cell_boundaries"]
    assert shapes.index.tolist() == [10, 20]
    assert shapes[INSTANCE_ID_COLUMN].tolist() == [10, 20]
    assert shapes[PROSEG_INTERNAL_ID_COLUMN].tolist() == [0, 1]

    table = sdata_obj.tables["table"]
    assert table.obs[INSTANCE_ID_COLUMN].tolist() == [10, 20]
    assert table.uns["spatialdata_attrs"]["instance_key"] == INSTANCE_ID_COLUMN
    assert (
        sdata_obj.attrs[MERXEN_SCHEMA_ATTR]["schema_version"] == MERXEN_SCHEMA_VERSION
    )
    validate_merxen_schema(sdata_obj, deep=True)


def test_upgrade_is_a_noop_for_current_schema() -> None:
    """Current stores should not be rewritten on every pipeline reuse."""
    sdata_obj = _raw_proseg_spatialdata()
    upgrade_spatialdata_contract_in_memory(sdata_obj)

    assert not upgrade_spatialdata_contract_in_memory(sdata_obj)


def test_source_contract_preserves_opaque_ids_and_quality_scores() -> None:
    """Instrument IDs remain provenance while operational joins use integers."""
    region = "merscope_polygons"
    points_df = pd.DataFrame(
        {
            "global_x": [0.5, 2.5, 5.0],
            "global_y": [0.5, 0.5, 5.0],
            "global_z": np.zeros(3, dtype=np.float32),
            "gene": ["GeneA", "GeneB", "GeneA"],
            "cell_id": ["entity-b", "entity-a", "0"],
            "transcript_score": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        }
    )
    points = PointsModel.parse(
        dd.from_pandas(points_df, npartitions=2),
        coordinates={
            "x": "global_x",
            "y": "global_y",
            "z": "global_z",
        },
        feature_key="gene",
        instance_key="cell_id",
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "EntityID": ["entity-b", "entity-a"],
                "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1)],
            },
            geometry="geometry",
            index=pd.Index(["entity-b", "entity-a"], name="EntityID"),
        )
    )
    obs = pd.DataFrame(
        {
            "EntityID": ["entity-b", "entity-a"],
            "region": pd.Categorical([region, region]),
        },
        index=pd.Index(["entity-b", "entity-a"]),
    )
    table = TableModel.parse(
        ad.AnnData(
            X=sparse.csr_matrix([[1], [1]]),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(["GeneA"], name="gene")),
        ),
        region=region,
        region_key="region",
        instance_key="EntityID",
    )
    sdata_obj = sd.SpatialData(
        points={"transcripts": points},
        shapes={region: shapes},
        tables={"table": table},
    )

    prepare_source_spatialdata_contract(sdata_obj, platform="MERSCOPE")

    normalized_points = sdata_obj.points["transcripts"].compute()
    assert normalized_points["transcript_id"].tolist() == [1, 2, 3]
    assert normalized_points["original_assignment"].tolist() == [2, 1, pd.NA]
    assert "cell_id" not in normalized_points.columns
    assert normalized_points["transcript_score"].tolist() == pytest.approx(
        [0.9, 0.8, 0.7]
    )
    assert sdata_obj.shapes[region][INSTANCE_ID_COLUMN].tolist() == [2, 1]
    assert sdata_obj.tables["table"].obs[INSTANCE_ID_COLUMN].tolist() == [2, 1]
    validate_merxen_schema(sdata_obj, deep=True)
