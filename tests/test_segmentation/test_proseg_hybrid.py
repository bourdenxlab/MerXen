"""Tests for transcript-supported ProSeg hybrid segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse
from shapely.geometry import Point, box
from spatialdata.models import PointsModel, ShapesModel, TableModel

from merxen.config import ProsegHybridConfig
from merxen.io.spatialdata_io import write_spatialdata_zarr
from merxen.segmentation.proseg_hybrid import (
    HYBRID_ASSIGNMENT_COLUMN,
    HYBRID_ASSIGNMENT_SOURCE_COLUMN,
    HYBRID_BACKGROUND_COLUMN,
    PROSEG_HYBRID_ALGORITHM,
    PROSEG_HYBRID_SHAPE_NAME,
    PROSEG_HYBRID_TABLE_NAME,
    assign_transcripts_to_hybrid_masks,
    build_hybrid_cell_geometry,
    build_local_convex_expansion_geometry,
    has_proseg_hybrid_refinement,
    run_proseg_hybrid_refinement,
    select_bulk_transcripts,
    smooth_growth_only_geometry,
)


def _bulk_points() -> np.ndarray:
    """Return a dense, asymmetric bulk transcript cloud."""
    return np.asarray(
        [(x, y) for x in np.linspace(1.0, 5.0, 5) for y in np.linspace(1.0, 4.0, 4)],
        dtype=np.float64,
    )


def test_bulk_selection_ignores_remote_low_count_component() -> None:
    """A small remote cluster should not extend the transcript hull."""
    bulk = _bulk_points()
    remote = np.asarray([[100.0, 100.0], [100.5, 100.0]], dtype=np.float64)

    selection = select_bulk_transcripts(
        np.vstack([bulk, remote]),
        box(0.0, 0.0, 6.0, 5.0),
    )

    assert selection.is_cellpose_anchored
    assert selection.outlier_count == 2
    assert len(selection.retained_points) == len(bulk)
    assert float(selection.retained_points[:, 0].max()) < 10.0


def test_bulk_selection_can_extend_a_cell_without_an_inside_anchor() -> None:
    """The largest dense component can extend a small Cellpose prior."""
    bulk = _bulk_points() + np.asarray([5.0, 0.0])
    points = np.vstack([bulk, [[100.0, 100.0]]])

    selection = select_bulk_transcripts(points, box(0.0, 0.0, 1.0, 1.0))

    assert not selection.is_cellpose_anchored
    assert len(selection.retained_points) == len(bulk)
    assert selection.outlier_count == 1


def test_bulk_selection_keeps_small_nearby_component() -> None:
    """Low-count components are outliers only when also remote from the bulk."""
    bulk = _bulk_points()
    nearby = np.asarray([[5.8, 2.0], [5.8, 3.0]], dtype=np.float64)

    selection = select_bulk_transcripts(
        np.vstack([bulk, nearby]),
        box(0.0, 0.0, 6.0, 5.0),
    )

    assert selection.outlier_count == 0
    assert len(selection.retained_points) == len(bulk) + len(nearby)


def test_local_convex_geometry_accepts_supported_chain_not_remote_outlier() -> None:
    """Supported external chains expand the mask but remote outliers do not."""
    cellpose = box(0.0, 0.0, 4.0, 4.0)
    points = np.vstack([_bulk_points(), [[100.0, 100.0]]])
    config = ProsegHybridConfig(min_transcripts=10)

    result = build_hybrid_cell_geometry(points, cellpose, config)

    assert result.fallback_reason == ""
    assert result.outlier_count == 1
    assert result.geometry.covers(cellpose)
    assert result.geometry.covers(Point(5.0, 4.0))
    assert not result.geometry.covers(Point(100.0, 100.0))
    assert result.supported_external == 4
    assert result.accepted_groups == 1


def test_low_transcript_cell_uses_cellpose_before_universal_smoothing() -> None:
    """Low-transcript cells do not create transcript-driven expansions."""
    cellpose = box(0.0, 0.0, 2.0, 2.0)
    result = build_local_convex_expansion_geometry(
        np.asarray([[0.5, 0.5], [1.0, 1.0]], dtype=np.float64),
        cellpose,
        ProsegHybridConfig(min_transcripts=10),
    )

    assert result.geometry.equals(cellpose)
    assert result.fallback_reason == "low_transcript_count"


def test_near_surface_singleton_can_expand_but_distant_singleton_cannot() -> None:
    """The safe zone is inclusive while distant expansion needs a group."""
    cellpose = box(0.0, 0.0, 4.0, 4.0)
    inside = np.asarray(
        [(x, y) for x in (1.0, 2.0, 3.0) for y in (1.0, 2.0, 3.0)],
        dtype=np.float64,
    )
    near = np.asarray([[4.2, 2.0]], dtype=np.float64)
    distant = np.asarray([[5.4, 3.5]], dtype=np.float64)
    config = ProsegHybridConfig(
        smoothing_radius_um=0.0,
        outward_rounding_um=0.0,
    )

    result = build_local_convex_expansion_geometry(
        np.vstack([inside, near, distant]),
        cellpose,
        config,
    )

    assert result.near_surface_accepted == 1
    assert result.geometry.covers(Point(near[0]))
    assert not result.geometry.covers(Point(distant[0]))


def test_growth_only_smoothing_contains_input_fills_holes_and_obeys_cap() -> None:
    """Closing may round concavities but can never shrink or escape the cap."""
    cellpose = box(0.0, 0.0, 4.0, 4.0)
    jagged_with_hole = (
        box(0.0, 0.0, 4.0, 4.0)
        .difference(box(1.0, 1.0, 2.0, 2.0))
        .union(box(4.0, 1.8, 5.5, 2.2))
    )
    config = ProsegHybridConfig(
        smoothing_radius_um=1.0,
        outward_rounding_um=0.2,
    )

    result = smooth_growth_only_geometry(
        jagged_with_hole,
        cellpose,
        config,
    )

    assert result.geometry.covers(jagged_with_hole)
    assert result.holes_filled == 1
    assert result.added_area > 1.0
    assert result.missing_original_area == 0.0
    assert result.cap_violation_area == 0.0
    assert (
        all(len(part.interiors) == 0 for part in result.geometry.geoms)
        if (result.geometry.geom_type == "MultiPolygon")
        else len(result.geometry.interiors) == 0
    )


def test_overlap_assignment_uses_proseg_only_inside_overlap() -> None:
    """Single-mask points use geometry; overlap points defer to ProSeg."""
    points = pd.DataFrame(
        {
            "x": [0.5, 1.5, 1.5, 1.5, 4.0],
            "y": [1.0, 1.0, 1.0, 1.0, 1.0],
            "assignment": pd.Series([pd.NA, 1, 0, pd.NA, 0], dtype="UInt32"),
            "background": [True, False, True, True, False],
        }
    )

    result = assign_transcripts_to_hybrid_masks(
        points,
        [box(0.0, 0.0, 2.0, 2.0), box(1.0, 0.0, 3.0, 2.0)],
        ["10", "20"],
        {0: "10", 1: "20"},
        x_col="x",
        y_col="y",
        assignment_col="assignment",
    )

    assert result[HYBRID_ASSIGNMENT_COLUMN].tolist() == [
        10,
        20,
        10,
        pd.NA,
        pd.NA,
    ]
    assert result[HYBRID_ASSIGNMENT_SOURCE_COLUMN].tolist() == [
        "single_mask",
        "proseg_overlap",
        "proseg_overlap",
        "ambiguous_overlap",
        "outside",
    ]
    assert result[HYBRID_BACKGROUND_COLUMN].tolist() == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_hybrid_refinement_roundtrips_spatialdata(
    tmp_path: Path,
) -> None:
    """Hybrid shapes, point assignments, and counts should persist together."""
    coordinates = [
        (x, y, assignment)
        for assignment, x_offset in [(0, 1.0), (1, 8.0)]
        for x in np.linspace(x_offset, x_offset + 3.0, 4)
        for y in np.linspace(1.0, 4.0, 4)
    ]
    points_df = pd.DataFrame(
        {
            "x": [item[0] for item in coordinates],
            "y": [item[1] for item in coordinates],
            "z": np.zeros(len(coordinates), dtype=np.float32),
            "gene": ["GeneA"] * len(coordinates),
            "assignment": pd.Series(
                [item[2] for item in coordinates],
                dtype="UInt32",
            ),
            "background": [False] * len(coordinates),
            "qv": np.linspace(0.8, 1.0, len(coordinates), dtype=np.float32),
        }
    )
    points = PointsModel.parse(
        dd.from_pandas(points_df, npartitions=2),
        coordinates={"x": "x", "y": "y", "z": "z"},
        feature_key="gene",
    )
    source_shape_gdf = gpd.GeoDataFrame(
        {
            "cell": [0, 1],
            "geometry": [box(0.5, 0.5, 5.5, 5.5), box(7.5, 0.5, 12.5, 5.5)],
        },
        geometry="geometry",
        index=pd.Index([0, 1], name="cell"),
    )
    source_shapes = ShapesModel.parse(source_shape_gdf)
    obs = pd.DataFrame(
        {
            "cell": [0, 1],
            "original_cell_id": ["1", "2"],
            "region": pd.Categorical(["cell_boundaries", "cell_boundaries"]),
        },
        index=pd.Index(["0", "1"], name="cell_index"),
    )
    var = pd.DataFrame(index=pd.Index(["GeneA"], name="gene"))
    var["gene"] = var.index
    table = TableModel.parse(
        ad.AnnData(X=sparse.csr_matrix([[16], [16]]), obs=obs, var=var),
        region="cell_boundaries",
        region_key="region",
        instance_key="cell",
    )
    zarr_path = tmp_path / "proseg.zarr"
    write_spatialdata_zarr(
        sd.SpatialData(
            points={"transcripts": points},
            shapes={"cell_boundaries": source_shapes},
            tables={"table": table},
        ),
        zarr_path,
    )

    masks = np.zeros((16, 16), dtype=np.uint32)
    masks[1:6, 1:6] = 1
    masks[1:6, 8:13] = 2
    mask_path = tmp_path / "cellpose.npy"
    np.save(mask_path, masks)
    transforms_path = tmp_path / "transforms.json"
    transforms_path.write_text(
        json.dumps(
            {
                "x_transform": [1.0, 0.0, 0.0],
                "y_transform": [0.0, 1.0, 0.0],
            }
        )
    )

    config = ProsegHybridConfig(min_transcripts=10)
    run_proseg_hybrid_refinement(
        zarr_path,
        mask_path,
        transforms_path,
        config,
    )

    assert has_proseg_hybrid_refinement(zarr_path, config)
    assert not has_proseg_hybrid_refinement(
        zarr_path,
        config.model_copy(update={"smoothing_radius_um": 2.0}),
    )
    result = sd.read_zarr(zarr_path)
    assert PROSEG_HYBRID_SHAPE_NAME in result.shapes
    assert PROSEG_HYBRID_TABLE_NAME in result.tables
    assert len(result.shapes[PROSEG_HYBRID_SHAPE_NAME]) == 2
    assert int(result.tables[PROSEG_HYBRID_TABLE_NAME].X.sum()) == 32
    assert (
        result.tables[PROSEG_HYBRID_TABLE_NAME].uns["proseg_hybrid"]["algorithm"]
        == PROSEG_HYBRID_ALGORITHM
    )
    augmented = result.points["transcripts"].compute()
    assert HYBRID_ASSIGNMENT_COLUMN in augmented.columns
    assert augmented[HYBRID_ASSIGNMENT_COLUMN].notna().all()
    assert "qv" in augmented.columns
