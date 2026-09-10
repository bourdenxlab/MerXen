"""Alignment materialization contract, inverse mapping, and raster tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
import xarray as xr
from scipy import sparse
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, get_transformation

from merxen.alignment.bundle import DisplacementField, ValisTransformBundle
from merxen.alignment.image_warp import warp_image_array_tiled
from merxen.alignment.manifest import (
    ALIGNMENT_MANIFEST_ATTR,
    ALIGNMENT_PAIR_REFERENCE_ATTR,
    invalidate_alignment_materialization,
)
from merxen.alignment.mapping import _load_legacy_transform, build_legacy_inverse_field
from merxen.alignment.materialize import (
    _assert_preserved_point_frame,
    _validate_roundtrip,
    materialize_alignment,
)
from merxen.alignment.transforms import NonRigidTransform
from merxen.config import AlignmentConfig
from merxen.io.spatialdata_io import (
    write_or_replace_element,
    write_spatialdata_metadata,
)
from merxen.io.spatialdata_schema import (
    PROSEG_ID_NAMESPACE,
    register_segmentation_branch,
    stamp_merxen_schema,
)
from merxen.masks import discover_masks, register_missing_mask_branches
from merxen.viewer_cache.format import (
    derived_image_pyramid_cache_key,
    derived_label_pyramid_cache_key,
)


def test_valis_bundle_maps_fixed_pixels_back_through_backward_field() -> None:
    identity = np.eye(3, dtype=np.float64)
    x = np.asarray([0.0, 10.0, 20.0])
    y = np.asarray([0.0, 10.0, 20.0])
    forward = np.zeros((3, 3, 2), dtype=np.float64)
    backward = np.zeros((3, 3, 2), dtype=np.float64)
    forward[..., 0] = 2.0
    backward[..., 0] = -2.0
    bundle = ValisTransformBundle(
        moving_dataset_to_image=identity,
        moving_image_to_registration=identity,
        pre_matrix=identity,
        global_matrix=identity,
        fixed_image_to_registration=identity,
        fixed_dataset_to_image=identity,
        selected_mode="non_rigid",
        forward_displacement=DisplacementField(x, y, forward),
        backward_displacement=DisplacementField(x, y, backward),
    )
    fixed = np.asarray([[4.0, 5.0], [12.0, 8.0], [18.0, 16.0]])

    moving = bundle.fixed_image_to_moving_image(fixed)

    np.testing.assert_allclose(moving, fixed - np.asarray([2.0, 0.0]))
    np.testing.assert_allclose(bundle.transform(moving), fixed)
    with pytest.raises(ValueError, match="backward displacement field"):
        replace(bundle, backward_displacement=None).fixed_image_to_moving_image(fixed)


def test_valis_bundle_roundtrips_scale_rotation_translation_and_displacement() -> None:
    theta = np.deg2rad(17.0)
    scale = 1.2
    linear = scale * np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    global_matrix = np.eye(3, dtype=np.float64)
    global_matrix[:2, :2] = linear
    global_matrix[:2, 2] = [8.0, -5.0]
    forward_offset = np.asarray([1.5, -0.75])
    backward_offset = -np.linalg.inv(linear) @ forward_offset
    axis = np.linspace(-100.0, 100.0, 9)
    forward = np.broadcast_to(forward_offset, (9, 9, 2)).copy()
    backward = np.broadcast_to(backward_offset, (9, 9, 2)).copy()
    identity = np.eye(3, dtype=np.float64)
    bundle = ValisTransformBundle(
        moving_dataset_to_image=identity,
        moving_image_to_registration=identity,
        pre_matrix=identity,
        global_matrix=global_matrix,
        fixed_image_to_registration=identity,
        fixed_dataset_to_image=identity,
        selected_mode="non_rigid",
        forward_displacement=DisplacementField(axis, axis, forward),
        backward_displacement=DisplacementField(axis, axis, backward),
    )
    moving = np.asarray([[2.0, 3.0], [10.0, 4.0], [15.0, 20.0]])
    fixed = bundle.transform(moving)

    np.testing.assert_allclose(
        bundle.fixed_dataset_to_moving_dataset(fixed),
        moving,
        atol=1e-8,
    )


def test_valis_roundtrip_uses_robust_and_field_resolution_limits(
    tmp_path: Path,
) -> None:
    identity = np.eye(3, dtype=np.float64)
    axis = np.asarray([0.0, 8.0])
    backward = DisplacementField(
        axis,
        axis,
        np.zeros((2, 2, 2), dtype=np.float64),
    )
    bundle = ValisTransformBundle(
        moving_dataset_to_image=identity,
        moving_image_to_registration=identity,
        pre_matrix=identity,
        global_matrix=identity,
        fixed_image_to_registration=identity,
        fixed_dataset_to_image=identity,
        selected_mode="non_rigid",
        backward_displacement=backward,
    )
    config = AlignmentConfig(
        pair_id="pair",
        merscope_zarr_path=tmp_path / "merscope.zarr",
        xenium_zarr_path=tmp_path / "xenium.zarr",
        output_dir=tmp_path / "align_out",
        materialization={"roundtrip_sample_spacing": 1},
    )
    errors = np.zeros(400, dtype=np.float64)
    errors[-1] = 3.9
    mapper = SimpleNamespace(
        backend="valis",
        valis_bundle=bundle,
        fixed_dataset_to_image_matrix=identity,
        roundtrip_error=lambda _xy: errors,
    )
    axis_coordinates = np.arange(20, dtype=np.float64)

    qc = _validate_roundtrip(
        cfg=config,
        mapper=mapper,
        fixed_x=axis_coordinates,
        fixed_y=axis_coordinates,
    )

    assert qc["maximum_tolerance_um"] == 4.0
    assert qc["maximum_error_um"] == 3.9

    errors[-1] = 4.1
    with pytest.raises(ValueError, match="maximum=4.1"):
        _validate_roundtrip(
            cfg=config,
            mapper=mapper,
            fixed_x=axis_coordinates,
            fixed_y=axis_coordinates,
        )

    errors[:] = 1.1
    with pytest.raises(ValueError, match="p95=1.1"):
        _validate_roundtrip(
            cfg=config,
            mapper=mapper,
            fixed_x=axis_coordinates,
            fixed_y=axis_coordinates,
        )


def test_legacy_inverse_field_roundtrips_affine_plus_rbf() -> None:
    anchors = np.asarray(
        [[0.0, 0.0], [30.0, 0.0], [0.0, 20.0], [30.0, 20.0]],
        dtype=np.float64,
    )
    transform = NonRigidTransform(
        affine_matrix=np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]),
        anchors=anchors,
        residuals=np.tile(np.asarray([[0.5, 0.25]]), (len(anchors), 1)),
        neighbors=4,
    )

    inverse = build_legacy_inverse_field(
        transform,
        fixed_shape_rc=(21, 31),
        fixed_dataset_to_image_matrix=np.eye(3),
        spacing_um=5.0,
        iterations=20,
        tolerance_um=1e-4,
    )
    fixed = np.asarray([[5.0, 5.0], [15.0, 10.0], [25.0, 15.0]])
    affine_source = fixed - np.asarray([3.0, -2.0])
    moving = affine_source + inverse.sample(fixed)

    np.testing.assert_allclose(transform.transform(moving), fixed, atol=1e-4)


def test_legacy_transform_reloads_from_version_one_zarr_metadata(
    tmp_path: Path,
) -> None:
    affine = np.asarray(
        [[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    config = AlignmentConfig(
        pair_id="legacy",
        merscope_zarr_path=tmp_path / "merscope.zarr",
        xenium_zarr_path=tmp_path / "xenium.zarr",
        output_dir=tmp_path / "missing-align-out",
        backend="legacy_spateo",
    )
    sdata_obj = SimpleNamespace(
        attrs={
            ALIGNMENT_MANIFEST_ATTR: {
                "version": 1,
                "selected_mode": "nonrigid",
                "nonrigid_transform": {
                    "affine_matrix": affine.tolist(),
                    "anchors": [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
                    "residuals": [[0.5, 0.25], [0.5, 0.25], [0.5, 0.25]],
                    "neighbors": 3,
                    "smoothing": 0.0,
                },
            }
        }
    )

    transform, selected_mode = _load_legacy_transform(config, sdata_obj)

    assert selected_mode == "nonrigid"
    np.testing.assert_allclose(
        transform.transform([[2.0, 4.0]]),
        [[5.5, 2.25]],
    )


def test_tiled_multichannel_uint16_warp_matches_single_tile_with_origins() -> None:
    y, x = np.indices((12, 15))
    image = xr.DataArray(
        np.stack([x + 10 * y, 1000 + 2 * x + y], axis=0).astype(np.uint16),
        dims=("c", "y", "x"),
        coords={"c": ["DAPI", "PolyT"]},
    )
    fixed_x = np.arange(10.0, 25.0)
    fixed_y = np.arange(20.0, 32.0)

    def fixed_to_source(xy: np.ndarray) -> np.ndarray:
        return xy - np.asarray([10.0, 20.0])

    tiled = warp_image_array_tiled(
        image,
        output_shape_rc=(12, 15),
        fixed_to_source_image_xy=fixed_to_source,
        tile_size=4,
        target_x_coordinates=fixed_x,
        target_y_coordinates=fixed_y,
    )
    single = warp_image_array_tiled(
        image,
        output_shape_rc=(12, 15),
        fixed_to_source_image_xy=fixed_to_source,
        tile_size=64,
        target_x_coordinates=fixed_x,
        target_y_coordinates=fixed_y,
    )

    assert tiled.dtype == np.uint16
    np.testing.assert_array_equal(tiled, single)
    np.testing.assert_array_equal(np.moveaxis(tiled, -1, 0), image.values)


def test_mask_inventory_registers_cellpose_and_xenium_nuclei() -> None:
    sdata_obj = SimpleNamespace(
        attrs={},
        points={"transcripts": SimpleNamespace(columns=["transcript_id"])},
        shapes={
            "MOSAIK_proseg": object(),
            "cell_boundaries": object(),
            "cellpose_nuclei": object(),
            "xenium_nucleus": object(),
        },
    )
    stamp_merxen_schema(sdata_obj, primary_points_key="transcripts")
    register_segmentation_branch(
        sdata_obj,
        "proseg",
        points_key="transcripts",
        assignment_column=None,
        shape_key="MOSAIK_proseg",
        table_key=None,
        id_namespace=PROSEG_ID_NAMESPACE,
    )
    register_segmentation_branch(
        sdata_obj,
        "legacy_proseg_alias",
        points_key="transcripts",
        assignment_column=None,
        shape_key="cell_boundaries",
        table_key=None,
        id_namespace=PROSEG_ID_NAMESPACE,
    )

    added = register_missing_mask_branches(sdata_obj)
    masks = discover_masks(sdata_obj)

    assert set(added) == {"cellpose_nuclei", "original_nucleus"}
    assert {(mask.shape_key, mask.boundary_type) for mask in masks} == {
        ("MOSAIK_proseg", "cell"),
        ("cellpose_nuclei", "nucleus"),
        ("xenium_nucleus", "nucleus"),
    }
    assert all(not mask.has_assignment for mask in masks)


def test_point_validation_ignores_unordered_categorical_dictionary_order() -> None:
    native = pd.DataFrame(
        {
            "hybrid_assignment_source": pd.Categorical(
                ["single_mask", "outside", "proseg_overlap"],
                categories=[
                    "ambiguous_overlap",
                    "single_mask",
                    "outside",
                    "proseg_overlap",
                ],
            )
        }
    )
    aligned = pd.DataFrame(
        {
            "hybrid_assignment_source": pd.Categorical(
                ["single_mask", "outside", "proseg_overlap"],
                categories=[
                    "ambiguous_overlap",
                    "outside",
                    "proseg_overlap",
                    "single_mask",
                ],
            )
        }
    )

    _assert_preserved_point_frame(native, aligned)

    ordered_native = native.astype(
        pd.CategoricalDtype(native.iloc[:, 0].cat.categories, ordered=True)
    )
    ordered_aligned = aligned.astype(
        pd.CategoricalDtype(aligned.iloc[:, 0].cat.categories, ordered=True)
    )
    with pytest.raises(AssertionError):
        _assert_preserved_point_frame(ordered_native, ordered_aligned)

    aligned.loc[1, "hybrid_assignment_source"] = "single_mask"
    with pytest.raises(AssertionError):
        _assert_preserved_point_frame(native, aligned)


def test_materializer_reconciles_vectors_tables_labels_image_and_manifest(
    tmp_path: Path,
) -> None:
    merscope_path, xenium_path, output_dir = _write_materialization_pair(tmp_path)
    config = _materialization_config(merscope_path, xenium_path, output_dir)
    no_reconcile = config.model_copy(
        update={
            "materialization": config.materialization.model_copy(
                update={"reconcile": False}
            )
        }
    )

    with pytest.raises(RuntimeError, match="reconciliation is disabled"):
        materialize_alignment(no_reconcile)
    assert ALIGNMENT_MANIFEST_ATTR not in sd.read_zarr(merscope_path).attrs

    first = materialize_alignment(config)
    (output_dir / "transform_chain.json").unlink()
    second = materialize_alignment(config)

    assert first["status"] == "complete"
    assert second["status"] == "current"
    interrupted = sd.read_zarr(merscope_path)
    interrupted_manifest = dict(interrupted.attrs[ALIGNMENT_MANIFEST_ATTR])
    interrupted_manifest["complete"] = False
    interrupted.attrs[ALIGNMENT_MANIFEST_ATTR] = interrupted_manifest
    write_spatialdata_metadata(interrupted, write_attrs=True)
    recovered_complete = materialize_alignment(config)
    assert recovered_complete["status"] == "complete"
    assert recovered_complete["recovered_incomplete_manifest"] is True

    moving = sd.read_zarr(merscope_path)
    fixed = sd.read_zarr(xenium_path)
    manifest = moving.attrs[ALIGNMENT_MANIFEST_ATTR]
    assert manifest["version"] == 2
    assert manifest["complete"] is True
    assert manifest["pair_id"] == "pair-1"
    assert fixed.attrs[ALIGNMENT_PAIR_REFERENCE_ATTR]["pair_id"] == "pair-1"
    assert "cells_aligned_nonrigid" in moving.shapes
    assert "transcripts_aligned_nonrigid" in moving.points
    assert "table_cells_aligned_nonrigid" in moving.tables
    assert "cells_aligned_nonrigid_labels" in moving.labels
    assert "cellpose_nuclei_aligned_nonrigid_labels" in moving.labels
    assert "MERSCOPE_z_projection_aligned_nonrigid" in moving.images
    np.testing.assert_array_equal(
        moving.tables["table_cells"].X.toarray(),
        moving.tables["table_cells_aligned_nonrigid"].X.toarray(),
    )
    transform = get_transformation(
        moving.images["MERSCOPE_z_projection_aligned_nonrigid"],
        to_coordinate_system="merxen_xenium",
    )
    np.testing.assert_allclose(
        transform.to_affine_matrix(
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ),
        np.asarray(manifest["fixed_grid"]["pixel_to_world_affine"]),
    )

    image_pyramid = derived_image_pyramid_cache_key(
        "MERSCOPE_z_projection_aligned_nonrigid",
        4,
    )
    label_pyramid = derived_label_pyramid_cache_key(
        "cells_aligned_nonrigid_labels",
        4,
    )
    get_transformation(
        moving.images[image_pyramid],
        to_coordinate_system="merxen_xenium",
    )
    get_transformation(
        moving.labels[label_pyramid],
        to_coordinate_system="merxen_xenium",
    )

    assert invalidate_alignment_materialization(moving, reason="hybrid columns added")
    assert moving.attrs[ALIGNMENT_MANIFEST_ATTR]["complete"] is False
    native_frame = moving.points["transcripts"].compute()
    native_frame.attrs.clear()
    native_frame["hybrid_assignment"] = pd.Series(
        [2, 1],
        index=native_frame.index,
        dtype="UInt64",
    )
    refreshed_points = PointsModel.parse(
        dd.from_pandas(native_frame, npartitions=1),
        coordinates={"x": "x", "y": "y"},
        feature_key="gene",
        transformations={"global": Identity()},
    )
    write_or_replace_element(
        moving,
        "transcripts",
        "points",
        refreshed_points,
        overwrite=True,
    )
    write_spatialdata_metadata(moving, write_attrs=True)

    reconciled = materialize_alignment(config)
    assert reconciled["status"] == "complete"
    refreshed = sd.read_zarr(merscope_path)
    aligned_frame = refreshed.points["transcripts_aligned_nonrigid"].compute()
    pd.testing.assert_series_equal(
        native_frame["hybrid_assignment"].reset_index(drop=True),
        aligned_frame["hybrid_assignment"].reset_index(drop=True),
        check_names=False,
    )

    del refreshed.labels["cells_aligned_nonrigid_labels"]
    refreshed.delete_element_from_disk("cells_aligned_nonrigid_labels")
    recovered = materialize_alignment(config)
    assert recovered["status"] == "complete"
    assert "cells_aligned_nonrigid_labels" in sd.read_zarr(merscope_path).labels


def _write_materialization_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    merscope_path = tmp_path / "merscope.zarr"
    xenium_path = tmp_path / "xenium.zarr"
    output_dir = tmp_path / "align_out"
    identity = np.eye(3, dtype=np.float64)
    rng = np.random.default_rng(4)
    dapi = rng.integers(1, 4000, size=(128, 128), dtype=np.uint16)
    source_image = Image2DModel.parse(
        xr.DataArray(
            np.stack([dapi, dapi // 2], axis=0),
            dims=("c", "y", "x"),
            coords={"c": ["DAPI", "PolyT"]},
        ),
        c_coords=["DAPI", "PolyT"],
        transformations={"global": Identity()},
    )
    fixed_image = Image2DModel.parse(
        xr.DataArray(
            dapi[np.newaxis, ...],
            dims=("c", "y", "x"),
            coords={"c": ["DAPI"]},
        ),
        c_coords=["DAPI"],
        transformations={"global": Identity()},
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "instance_id": np.asarray([1, 2], dtype=np.uint64),
                "geometry": [box(10, 10, 40, 40), box(60, 60, 100, 100)],
            },
            index=pd.Index([1, 2], dtype="uint64", name="instance_id"),
        ),
        transformations={"global": Identity()},
    )
    nuclei = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "instance_id": np.asarray([11, 12], dtype=np.uint64),
                "geometry": [box(15, 15, 25, 25), box(70, 70, 85, 85)],
            },
            index=pd.Index([11, 12], dtype="uint64", name="instance_id"),
        ),
        transformations={"global": Identity()},
    )
    points_frame = pd.DataFrame(
        {
            "x": [20.0, 70.0],
            "y": [20.0, 70.0],
            "gene": pd.Categorical(["A", "B"]),
            "transcript_id": np.asarray([1, 2], dtype=np.uint64),
            "assignment": pd.Series([1, 2], dtype="UInt64"),
        }
    )
    points = PointsModel.parse(
        dd.from_pandas(points_frame, npartitions=1),
        coordinates={"x": "x", "y": "y"},
        feature_key="gene",
        transformations={"global": Identity()},
    )
    table = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[2, 0], [0, 3]], dtype=np.int64)),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray([1, 2], dtype=np.uint64),
                "region": pd.Categorical(["cells", "cells"]),
            },
            index=pd.Index(["1", "2"], name="obs_id"),
        ),
        var=pd.DataFrame(index=pd.Index(["A", "B"], name="gene")),
    )
    table.obsm["spatial"] = np.asarray([[25.0, 25.0], [80.0, 80.0]])
    parsed_table = TableModel.parse(
        table,
        region="cells",
        region_key="region",
        instance_key="instance_id",
    )
    moving = SpatialData(
        images={"MERSCOPE_z_projection": source_image},
        shapes={"cells": shapes, "cellpose_nuclei": nuclei},
        points={"transcripts": points},
        tables={"table_cells": parsed_table},
    )
    stamp_merxen_schema(moving, primary_points_key="transcripts", platform="MERSCOPE")
    register_segmentation_branch(
        moving,
        "proseg",
        points_key="transcripts",
        assignment_column="assignment",
        shape_key="cells",
        table_key="table_cells",
        id_namespace=PROSEG_ID_NAMESPACE,
    )
    moving.write(merscope_path)
    SpatialData(images={"morphology_focus": fixed_image}).write(xenium_path)
    bundle = ValisTransformBundle(
        moving_dataset_to_image=identity,
        moving_image_to_registration=identity,
        pre_matrix=identity,
        global_matrix=identity,
        fixed_image_to_registration=identity,
        fixed_dataset_to_image=identity,
        selected_mode="global",
    )
    bundle.save(output_dir)
    return merscope_path, xenium_path, output_dir


def _materialization_config(
    merscope_path: Path,
    xenium_path: Path,
    output_dir: Path,
) -> AlignmentConfig:
    identity = np.eye(3).tolist()
    return AlignmentConfig.model_validate(
        {
            "pair_id": "pair-1",
            "merscope_zarr_path": str(merscope_path),
            "xenium_zarr_path": str(xenium_path),
            "output_dir": str(output_dir),
            "backend": "valis",
            "merscope_image": {
                "image_key": "MERSCOPE_z_projection",
                "dapi_channel": "DAPI",
                "dataset_to_image_matrix": identity,
            },
            "xenium_image": {
                "image_key": "morphology_focus",
                "dapi_channel": "DAPI",
                "dataset_to_image_matrix": identity,
            },
            "materialization": {
                "tile_size": 64,
                "chunk_size": 64,
                "pyramid_min_size": 64,
                "roundtrip_sample_spacing": 32,
            },
        }
    )
