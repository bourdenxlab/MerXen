"""Tests for alignment stage SpatialData writing."""

from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.transformations import Identity, get_transformation

from merxen.alignment.pipeline import (
    MERXEN_ALIGNMENT_ATTR,
    _write_moving_alignment_to_zarr,
)
from merxen.alignment.register import TransformResult
from merxen.alignment.transforms import fit_nonrigid_transform
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    PROSEG_ID_NAMESPACE,
    register_segmentation_branch,
    stamp_merxen_schema,
)


def test_write_moving_aligned_zarr_adds_transforms_and_nonrigid_elements(
    tmp_path: Path,
) -> None:
    """Alignment output should preserve raw elements and add non-rigid elements."""
    input_zarr = tmp_path / "input.zarr"
    source_xy = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    affine = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    nonrigid_xy = source_xy + np.array([11.0, 22.0])

    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "instance_id": np.asarray([1, 2, 3], dtype=np.uint64),
                "geometry": [
                    box(0.0, 0.0, 1.0, 1.0),
                    box(2.0, 0.0, 3.0, 1.0),
                    box(0.0, 2.0, 1.0, 3.0),
                ],
            },
            index=pd.Index([1, 2, 3], dtype="uint64", name="instance_id"),
        ),
        transformations={"global": Identity()},
    )
    points = PointsModel.parse(
        dd.from_pandas(
            pd.DataFrame(
                {
                    "x": source_xy[:, 0],
                    "y": source_xy[:, 1],
                    "gene": ["A", "B", "C"],
                    "transcript_id": np.asarray([1, 2, 3], dtype=np.uint64),
                    "assignment": pd.Series([1, 2, 3], dtype="UInt64"),
                }
            ),
            npartitions=1,
        ),
        coordinates={"x": "x", "y": "y"},
        feature_key="gene",
        transformations={"global": Identity()},
    )
    source = SpatialData(shapes={"cells": shapes}, points={"transcripts": points})
    stamp_merxen_schema(source, primary_points_key="transcripts")
    register_segmentation_branch(
        source,
        "proseg",
        points_key="transcripts",
        assignment_column="assignment",
        shape_key="cells",
        table_key=None,
        id_namespace=PROSEG_ID_NAMESPACE,
    )
    source.write(input_zarr)

    transform = fit_nonrigid_transform(
        source_xy,
        nonrigid_xy,
        affine_matrix=affine,
        max_anchors=3,
    )
    result = TransformResult(
        merscope_to_common={
            "selected_mode": "nonrigid",
            "rigid_affine_matrix": affine.tolist(),
        },
        xenium_to_common={"type": "identity"},
        metadata={"pair_id": "example"},
        nonrigid_transform=transform,
    )

    _write_moving_alignment_to_zarr(input_zarr, result)
    _write_moving_alignment_to_zarr(input_zarr, result)

    aligned = sd.read_zarr(input_zarr)
    assert MERXEN_ALIGNMENT_ATTR in aligned.attrs
    assert "cells" in aligned.shapes
    assert "cells_aligned_nonrigid" in aligned.shapes
    assert "transcripts" in aligned.points
    assert "transcripts_aligned_nonrigid" in aligned.points
    registry = aligned.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    assert registry["proseg_aligned_nonrigid"]["points"] == (
        "transcripts_aligned_nonrigid"
    )
    assert registry["proseg_aligned_nonrigid"]["shape"] == ("cells_aligned_nonrigid")
    assert registry["proseg_aligned_nonrigid"]["coordinate_variant_of"] == "proseg"

    rigid = get_transformation(
        aligned.shapes["cells"],
        to_coordinate_system="merxen_xenium",
    )
    nonrigid = get_transformation(
        aligned.shapes["cells_aligned_nonrigid"],
        to_coordinate_system="merxen_xenium",
    )
    np.testing.assert_allclose(
        rigid.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        affine,
    )
    assert isinstance(nonrigid, Identity)

    point_df = aligned.points["transcripts_aligned_nonrigid"].compute()
    np.testing.assert_allclose(point_df[["x", "y"]].to_numpy(), nonrigid_xy)
    np.testing.assert_allclose(point_df[["raw_x", "raw_y"]].to_numpy(), source_xy)
