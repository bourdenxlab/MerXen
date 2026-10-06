"""Tests for alignment stage SpatialData writing."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import (
    Affine,
    Identity,
    get_transformation,
    set_transformation,
)

from merxen.alignment.bundle import ValisTransformBundle
from merxen.alignment.pipeline import (
    MERXEN_ALIGNMENT_ATTR,
    _transform_points,
    _transform_shapes,
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
    table = ad.AnnData(
        X=sparse.csr_matrix(np.eye(3, dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray([1, 2, 3], dtype=np.uint64),
                "region": pd.Categorical(["cells", "cells", "cells"]),
            },
            index=pd.Index(["1", "2", "3"], name="cell_index"),
        ),
        var=pd.DataFrame(index=pd.Index(["A", "B", "C"], name="gene")),
    )
    table.obsm["spatial"] = source_xy.copy()
    parsed_table = TableModel.parse(
        table,
        region="cells",
        region_key="region",
        instance_key="instance_id",
    )
    source = SpatialData(
        shapes={"cells": shapes},
        points={"transcripts": points},
        tables={"table": parsed_table},
    )
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

    # The native transcripts must carry the same rigid affine as the native
    # shapes; only their non-rigid copy is already in the alignment frame.
    native_points_rigid = get_transformation(
        aligned.points["transcripts"],
        to_coordinate_system="merxen_xenium",
    )
    np.testing.assert_allclose(
        native_points_rigid.to_affine_matrix(
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ),
        affine,
    )
    assert isinstance(
        get_transformation(
            aligned.points["transcripts_aligned_nonrigid"],
            to_coordinate_system="merxen_xenium",
        ),
        Identity,
    )

    point_df = aligned.points["transcripts_aligned_nonrigid"].compute()
    np.testing.assert_allclose(point_df[["x", "y"]].to_numpy(), nonrigid_xy)
    np.testing.assert_allclose(point_df[["raw_x", "raw_y"]].to_numpy(), source_xy)
    np.testing.assert_allclose(
        aligned.tables["table"].obsm["spatial_merxen_xenium"],
        nonrigid_xy,
    )


def _translation_result() -> TransformResult:
    affine = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    source_xy = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    return TransformResult(
        merscope_to_common={
            "selected_mode": "nonrigid",
            "rigid_affine_matrix": affine.tolist(),
        },
        xenium_to_common={"type": "identity"},
        metadata={"pair_id": "example"},
        nonrigid_transform=fit_nonrigid_transform(
            source_xy,
            source_xy + np.array([5.0, -3.0]),
            affine_matrix=affine,
            max_anchors=3,
        ),
    )


def test_aligned_copies_do_not_share_transformations_with_native() -> None:
    """Stamping an aligned copy must leave the native element's transforms intact."""
    rigid = Affine(
        np.array([[0.0, -1.0, 7.0], [1.0, 0.0, 2.0], [0.0, 0.0, 1.0]]),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    points = PointsModel.parse(
        dd.from_pandas(
            pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "gene": ["A", "B"]}),
            npartitions=1,
        ),
        coordinates={"x": "x", "y": "y"},
        feature_key="gene",
        transformations={"global": Identity(), "merxen_xenium": rigid},
    )
    # Points without recognised coordinate columns are passed through unchanged,
    # so the returned element must still be a detached copy.
    no_xy_points = dd.from_pandas(
        pd.DataFrame({"u": [0.0], "v": [0.0], "gene": ["A"]}),
        npartitions=1,
    )
    set_transformation(
        no_xy_points,
        {"global": Identity(), "merxen_xenium": rigid},
        set_all=True,
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame({"geometry": [box(0.0, 0.0, 1.0, 1.0)]}),
        transformations={"global": Identity(), "merxen_xenium": rigid},
    )
    result = _translation_result()

    derived = [
        (points, _transform_points(points, result)),
        (no_xy_points, _transform_points(no_xy_points, result)),
        (shapes, _transform_shapes(shapes, result)),
    ]
    for native, aligned in derived:
        set_transformation(aligned, Identity(), to_coordinate_system="merxen_xenium")
        assert get_transformation(native, to_coordinate_system="merxen_xenium") is rigid
        assert isinstance(
            get_transformation(aligned, to_coordinate_system="merxen_xenium"),
            Identity,
        )


def test_transform_points_dask_metadata_matches_p7113_shared_domain_order() -> None:
    """Derived transcript columns must match Dask metadata exactly."""
    source = pd.DataFrame(
        {
            "transcript_id": np.asarray([1, 2, 3, 4], dtype=np.uint64),
            "qv": np.asarray([40.0, 35.0, 30.0, 25.0], dtype=np.float32),
            "x": np.asarray([1.0, 2.0, np.nan, np.nan], dtype=np.float32),
            "y": np.asarray([3.0, 4.0, np.nan, np.nan], dtype=np.float32),
            "z": np.zeros(4, dtype=np.float32),
            "gene": pd.Categorical(["A", "B", "A", "B"]),
            "assignment": pd.Series([1, 2, pd.NA, pd.NA], dtype="UInt64"),
            "hybrid_assignment": pd.Series([1, 2, pd.NA, pd.NA], dtype="UInt64"),
            "hybrid_assignment_source": pd.Categorical(
                ["proseg", "cellpose", "background", "background"]
            ),
        }
    )
    points = dd.from_pandas(source, npartitions=2)
    identity = np.eye(3, dtype=np.float64)
    bundle = ValisTransformBundle(
        moving_dataset_to_image=identity,
        moving_image_to_registration=identity,
        pre_matrix=identity,
        global_matrix=np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 6.0], [0.0, 0.0, 1.0]]),
        fixed_image_to_registration=identity,
        fixed_dataset_to_image=identity,
        selected_mode="global",
    )
    result = TransformResult(
        merscope_to_common={
            "selected_mode": "global",
            "rigid_affine_matrix": bundle.global_dataset_matrix.tolist(),
        },
        xenium_to_common={"type": "identity"},
        metadata={
            "backend": "valis",
            "moving_platform": "MERSCOPE",
            "parameters": {"mark_shared_tissue_domain": True},
        },
        valis_transform=bundle,
        valid_domain_mask=np.ones((16, 16), dtype=np.uint8),
    )

    transformed = _transform_points(points, result)
    expected_columns = [
        *source.columns,
        "raw_x",
        "raw_y",
        "in_shared_tissue_domain",
    ]

    assert list(transformed._meta.columns) == expected_columns
    computed = transformed.compute()
    assert list(computed.columns) == expected_columns
    assert computed["x"].dtype == np.dtype("float64")
    assert computed["y"].dtype == np.dtype("float64")
    assert computed["raw_x"].dtype == np.dtype("float64")
    assert computed["raw_y"].dtype == np.dtype("float64")
    assert computed["in_shared_tissue_domain"].dtype == np.dtype("bool")
    np.testing.assert_allclose(
        computed.loc[[0, 1], ["x", "y"]].to_numpy(),
        np.asarray([[6.0, 9.0], [7.0, 10.0]]),
    )
    np.testing.assert_allclose(
        computed.loc[[0, 1], ["raw_x", "raw_y"]].to_numpy(),
        np.asarray([[1.0, 3.0], [2.0, 4.0]]),
    )
    assert computed.loc[[0, 1], "in_shared_tissue_domain"].all()
    assert not computed.loc[[2, 3], "in_shared_tissue_domain"].any()
