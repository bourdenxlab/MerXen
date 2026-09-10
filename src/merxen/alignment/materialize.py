"""Idempotent alignment materialization into the moving MERSCOPE Zarr."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
import zarr
from spatialdata.transformations import (
    Affine,
    Identity,
    get_transformation,
    set_transformation,
)

from merxen.alignment.image_warp import materialize_warped_image
from merxen.alignment.manifest import (
    ALIGNMENT_MANIFEST_ATTR,
    ALIGNMENT_MANIFEST_VERSION,
    ALIGNMENT_PAIR_REFERENCE_ATTR,
    NONRIGID_ELEMENT_SUFFIX,
    alignment_pair_reference,
    initialize_alignment_manifest,
    native_store_revision,
    set_artifact_status,
    utc_timestamp,
)
from merxen.alignment.mapping import AlignmentCoordinateMapper, load_alignment_mapper
from merxen.alignment.models import TransformResult
from merxen.alignment.pipeline import _transform_points, _transform_shapes
from merxen.config import AlignmentConfig
from merxen.enrichment.assignment import clone_table_for_region
from merxen.io.image_source import image_to_cyx
from merxen.io.spatialdata_io import (
    spatialdata_write_lock,
    write_or_replace_element,
    write_spatialdata_metadata,
)
from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    MERXEN_SCHEMA_ATTR,
    TRANSCRIPT_ID_COLUMN,
    canonical_shape_instance_series,
    register_segmentation_branch,
    stamp_merxen_schema,
    validate_merxen_schema,
)
from merxen.io.transcript_io import first_existing_col
from merxen.masks import discover_masks, register_missing_mask_branches
from merxen.viewer_cache.build import (
    ViewerCacheParams,
    build_image_pyramid,
    build_mask_cache_on_grid,
)
from merxen.viewer_cache.format import (
    derived_image_pyramid_cache_key,
    derived_label_pyramid_cache_key,
    derived_outline_cache_key,
)
from merxen.viewer_cache.rasterize import coords_origin_step

logger = logging.getLogger(__name__)
VALIS_ROUNDTRIP_P95_TOLERANCE_UM = 1.0
VALIS_ROUNDTRIP_MAX_FIELD_STEP_FRACTION = 0.5


def materialize_alignment(config: AlignmentConfig) -> dict[str, Any]:
    """Reconcile aligned vectors, tables, labels, and imagery for one pair.

    Args:
        config: Alignment config pointing at both stores and a transform bundle.

    Returns:
        JSON-compatible materialization summary.

    Raises:
        ValueError: If pair metadata, transforms, or validation are inconsistent.
    """
    cfg = config
    params = cfg.materialization
    if not params.enabled:
        return {"pair_id": cfg.pair_id, "status": "disabled"}
    if cfg.fixed_platform != "XENIUM" or cfg.moving_platform != "MERSCOPE":
        raise ValueError(
            "Alignment materialization currently requires fixed_platform='XENIUM' "
            "and moving_platform='MERSCOPE'"
        )

    moving_path = Path(cfg.merscope_zarr_path)
    fixed_path = Path(cfg.xenium_zarr_path)
    with ExitStack() as stack:
        for path in sorted(
            {moving_path, fixed_path}, key=lambda item: str(item.resolve())
        ):
            stack.enter_context(spatialdata_write_lock(path))
        moving_sdata = sd.read_zarr(moving_path)
        fixed_sdata = sd.read_zarr(fixed_path)
        _validate_pair_reference(
            fixed_sdata,
            cfg.pair_id,
            moving_manifest=moving_sdata.attrs.get(ALIGNMENT_MANIFEST_ATTR),
        )
        register_missing_mask_branches(moving_sdata)
        stamp_merxen_schema(moving_sdata, platform="MERSCOPE")

        source_key = params.source_image_key
        fixed_key = params.fixed_image_key
        if source_key not in moving_sdata.images:
            raise KeyError(f"MERSCOPE source image {source_key!r} does not exist")
        if fixed_key not in fixed_sdata.images:
            raise KeyError(f"Xenium fixed image {fixed_key!r} does not exist")
        source_cyx = image_to_cyx(moving_sdata.images[source_key])
        fixed_cyx = image_to_cyx(fixed_sdata.images[fixed_key])
        fixed_shape_rc = (int(fixed_cyx.sizes["y"]), int(fixed_cyx.sizes["x"]))
        source_x = _axis_coordinates(source_cyx, "x")
        source_y = _axis_coordinates(source_cyx, "y")
        fixed_x = _axis_coordinates(fixed_cyx, "x")
        fixed_y = _axis_coordinates(fixed_cyx, "y")

        mapper = load_alignment_mapper(
            cfg,
            moving_sdata=moving_sdata,
            fixed_sdata=fixed_sdata,
            source_image_key=source_key,
            fixed_image_key=fixed_key,
            fixed_shape_rc=fixed_shape_rc,
            require_raster_inverse=(
                params.materialize_labels or params.materialize_image
            ),
        )
        native_revision = native_store_revision(moving_path, moving_sdata)
        existing_manifest = moving_sdata.attrs.get(ALIGNMENT_MANIFEST_ATTR)
        if (
            not params.reconcile
            and not params.force
            and not _is_compatible_manifest(existing_manifest, cfg)
        ):
            raise RuntimeError(
                "Alignment manifest is missing or incompatible and reconciliation "
                "is disabled"
            )
        manifest = _current_or_new_manifest(
            cfg=cfg,
            mapper=mapper,
            moving_sdata=moving_sdata,
            fixed_sdata=fixed_sdata,
            moving_path=moving_path,
        )
        if _can_reuse_complete_manifest(manifest, native_revision, params.force):
            try:
                qc = _validate_materialization(
                    cfg=cfg,
                    manifest=manifest,
                    mapper=mapper,
                    moving_path=moving_path,
                    fixed_sdata=fixed_sdata,
                    fixed_cyx=fixed_cyx,
                    fixed_x=fixed_x,
                    fixed_y=fixed_y,
                )
            except (KeyError, RuntimeError, ValueError, AssertionError) as exc:
                if not params.reconcile:
                    raise RuntimeError(
                        "Alignment outputs failed validation and reconciliation is "
                        "disabled"
                    ) from exc
                logger.warning(
                    "Complete alignment manifest failed reconciliation; rebuilding",
                    exc_info=True,
                )
            else:
                if _ensure_pair_reference(
                    fixed_sdata,
                    pair_id=cfg.pair_id,
                    native_fingerprint=str(native_revision["fingerprint"]),
                    transform_fingerprint=str(manifest["transform"]["fingerprint"]),
                ):
                    write_spatialdata_metadata(fixed_sdata, write_attrs=True)
                return {
                    "pair_id": cfg.pair_id,
                    "status": "current",
                    "native_fingerprint": native_revision["fingerprint"],
                    "qc": qc,
                }
        elif params.reconcile and _can_recover_complete_artifacts(
            manifest,
            native_revision,
            params.force,
        ):
            try:
                qc = _validate_materialization(
                    cfg=cfg,
                    manifest=manifest,
                    mapper=mapper,
                    moving_path=moving_path,
                    fixed_sdata=fixed_sdata,
                    fixed_cyx=fixed_cyx,
                    fixed_x=fixed_x,
                    fixed_y=fixed_y,
                )
            except (KeyError, RuntimeError, ValueError, AssertionError):
                logger.warning(
                    "Incomplete alignment manifest failed recovery validation; "
                    "rebuilding",
                    exc_info=True,
                )
            else:
                _finalize_materialization(
                    moving_sdata=moving_sdata,
                    fixed_sdata=fixed_sdata,
                    manifest=manifest,
                    qc=qc,
                    pair_id=cfg.pair_id,
                    native_revision=native_revision,
                )
                return {
                    "pair_id": cfg.pair_id,
                    "status": "complete",
                    "recovered_incomplete_manifest": True,
                    "native_fingerprint": native_revision["fingerprint"],
                    "artifacts": manifest["artifacts"],
                    "qc": qc,
                }
        elif not params.reconcile and not params.force:
            raise RuntimeError(
                "Alignment outputs are missing or stale and reconciliation is disabled"
            )

        coordinate_system = str(manifest["common_coordinate_system"])
        coordinate_to_world = np.linalg.inv(mapper.fixed_dataset_to_image_matrix)
        source_coordinate_to_index = np.linalg.inv(
            _index_to_coordinate_matrix(source_x, source_y)
        )
        pixel_to_world = coordinate_to_world @ _index_to_coordinate_matrix(
            fixed_x,
            fixed_y,
        )
        manifest["native_input"] = native_revision
        manifest["fixed_grid"] = {
            "image_key": fixed_key,
            "dimensions": {
                "width": fixed_shape_rc[1],
                "height": fixed_shape_rc[0],
            },
            "axis_order": ["c", "y", "x"],
            "pixel_to_world_affine": pixel_to_world.tolist(),
            "coordinate_to_world_affine": coordinate_to_world.tolist(),
            "coordinate_system": coordinate_system,
        }
        manifest["source_image"] = {
            "key": source_key,
            "dtype": str(source_cyx.dtype),
            "channels": _channel_labels(source_cyx),
            "axis_order": ["c", "y", "x"],
        }
        manifest["complete"] = False
        manifest["mappings"] = {
            "points": {},
            "shapes": {},
            "tables": {},
            "labels": {},
            "images": {},
        }
        manifest["artifacts"] = {}
        manifest["qc"] = {}
        moving_sdata.attrs[ALIGNMENT_MANIFEST_ATTR] = manifest
        write_spatialdata_metadata(moving_sdata, write_attrs=True)

        native_masks = discover_masks(moving_sdata)
        if params.materialize_vectors:
            _materialize_vectors_and_tables(
                moving_sdata=moving_sdata,
                mapper=mapper,
                manifest=manifest,
                coordinate_system=coordinate_system,
            )
            write_spatialdata_metadata(
                moving_sdata,
                write_attrs=True,
                write_transformations=True,
            )

        if params.materialize_labels:
            _materialize_labels(
                moving_sdata=moving_sdata,
                moving_path=moving_path,
                native_masks=native_masks,
                manifest=manifest,
                fixed_shape_rc=fixed_shape_rc,
                pixel_to_world=pixel_to_world,
                coordinate_system=coordinate_system,
                config=cfg,
            )
            write_spatialdata_metadata(moving_sdata, write_attrs=True)

        if params.materialize_image:
            image_result = materialize_warped_image(
                sdata_obj=moving_sdata,
                zarr_path=moving_path,
                source_image=moving_sdata.images[source_key],
                output_key=params.output_image_key,
                output_shape_rc=fixed_shape_rc,
                fixed_to_source_image_xy=lambda xy: _apply_affine(
                    mapper.fixed_image_to_moving_image(xy),
                    source_coordinate_to_index,
                ),
                transform=Affine(
                    pixel_to_world,
                    input_axes=("x", "y"),
                    output_axes=("x", "y"),
                ),
                coordinate_system=coordinate_system,
                tile_size=params.tile_size,
                chunk_size=params.chunk_size,
                interpolation=params.image_interpolation,
                fill_value=params.image_fill_value,
                target_x_coordinates=fixed_x,
                target_y_coordinates=fixed_y,
            )
            image_pyramid_key = derived_image_pyramid_cache_key(
                params.output_image_key,
                params.image_pyramid_downsample,
            )
            base_cyx = xr.DataArray(
                da.from_zarr(
                    str(moving_path / "images" / params.output_image_key / "s0")
                ),
                dims=("c", "y", "x"),
                coords={"c": list(image_result.channels), "y": fixed_y, "x": fixed_x},
            )
            pyramid_status = build_image_pyramid(
                sdata=moving_sdata,
                zarr_path=moving_path,
                image_key=params.output_image_key,
                base_cyx=base_cyx,
                channels=list(image_result.channels),
                transform=Affine(
                    pixel_to_world,
                    input_axes=("x", "y"),
                    output_axes=("x", "y"),
                ),
                params=ViewerCacheParams(
                    downsample=params.image_pyramid_downsample,
                    min_size=params.pyramid_min_size,
                    build_image_pyramid=True,
                    force=True,
                ),
                coordinate_system=coordinate_system,
            )
            set_artifact_status(
                manifest,
                kind="images",
                native_key=source_key,
                output_key=params.output_image_key,
                status="complete",
                qc={
                    "nonzero_fraction": image_result.nonzero_fraction,
                    "tile_seams_validated": image_result.tile_seams_validated,
                },
                source_dtype=str(source_cyx.dtype),
                output_dtype=image_result.dtype,
                source_channels=_channel_labels(source_cyx),
                output_channels=list(image_result.channels),
                chunks=list(image_result.chunks_cyx),
                pyramid_key=image_pyramid_key,
                pyramid_status=pyramid_status,
            )
            moving_sdata.attrs[ALIGNMENT_MANIFEST_ATTR] = manifest
            write_spatialdata_metadata(moving_sdata, write_attrs=True)

        qc = _validate_materialization(
            cfg=cfg,
            manifest=manifest,
            mapper=mapper,
            moving_path=moving_path,
            fixed_sdata=fixed_sdata,
            fixed_cyx=fixed_cyx,
            fixed_x=fixed_x,
            fixed_y=fixed_y,
        )
        _finalize_materialization(
            moving_sdata=moving_sdata,
            fixed_sdata=fixed_sdata,
            manifest=manifest,
            qc=qc,
            pair_id=cfg.pair_id,
            native_revision=native_revision,
        )

    return {
        "pair_id": cfg.pair_id,
        "status": "complete",
        "native_fingerprint": native_revision["fingerprint"],
        "artifacts": manifest["artifacts"],
        "qc": qc,
    }


def _current_or_new_manifest(
    *,
    cfg: AlignmentConfig,
    mapper: AlignmentCoordinateMapper,
    moving_sdata: Any,
    fixed_sdata: Any,
    moving_path: Path,
) -> dict[str, Any]:
    existing = moving_sdata.attrs.get(ALIGNMENT_MANIFEST_ATTR)
    if _is_compatible_manifest(existing, cfg):
        return dict(existing)
    result = _transform_result_for_mapper(cfg, mapper)
    return initialize_alignment_manifest(
        config=cfg,
        result=result,
        moving_sdata=moving_sdata,
        fixed_sdata=fixed_sdata,
        moving_zarr_path=moving_path,
    )


def _is_compatible_manifest(value: Any, cfg: AlignmentConfig) -> bool:
    if not isinstance(value, dict):
        return False
    transform = value.get("transform")
    return bool(
        value.get("version") == ALIGNMENT_MANIFEST_VERSION
        and value.get("pair_id") == cfg.pair_id
        and isinstance(transform, dict)
        and transform.get("backend") == cfg.backend
    )


def _transform_result_for_mapper(
    cfg: AlignmentConfig,
    mapper: AlignmentCoordinateMapper,
) -> TransformResult:
    if mapper.valis_bundle is not None:
        affine = mapper.valis_bundle.global_dataset_matrix
    elif mapper.legacy_transform is not None:
        affine = mapper.legacy_transform.affine_matrix
    else:
        raise RuntimeError("Mapper has no transform payload")
    return TransformResult(
        merscope_to_common={
            "selected_mode": mapper.selected_mode,
            "rigid_affine_matrix": np.asarray(affine).tolist(),
        },
        xenium_to_common={"type": "identity"},
        metadata={
            "pair_id": cfg.pair_id,
            "backend": cfg.backend,
            "moving_platform": cfg.moving_platform,
            "fixed_platform": cfg.fixed_platform,
            "coordinate_system_name": cfg.valis.coordinate_system_name,
            "parameters": cfg.valis.model_dump(mode="json"),
        },
        nonrigid_transform=mapper.legacy_transform,
        valis_transform=mapper.valis_bundle,
    )


def _materialize_vectors_and_tables(
    *,
    moving_sdata: Any,
    mapper: AlignmentCoordinateMapper,
    manifest: dict[str, Any],
    coordinate_system: str,
) -> None:
    result = _transform_result_for_mapper_from_manifest(mapper, manifest)

    native_shape_keys = sorted(
        str(key)
        for key in moving_sdata.shapes
        if not str(key).endswith(NONRIGID_ELEMENT_SUFFIX)
    )
    for shape_key in native_shape_keys:
        output_key = f"{shape_key}{NONRIGID_ELEMENT_SUFFIX}"
        aligned = _transform_shapes(moving_sdata.shapes[shape_key], result)
        set_transformation(aligned, Identity(), to_coordinate_system=coordinate_system)
        write_or_replace_element(
            moving_sdata,
            output_key,
            "shapes",
            aligned,
            overwrite=True,
        )
        set_artifact_status(
            manifest,
            kind="shapes",
            native_key=shape_key,
            output_key=output_key,
            status="complete",
            qc={"instance_ids_preserved": True},
            source_rows=int(len(moving_sdata.shapes[shape_key])),
            output_rows=int(len(aligned)),
        )

    native_point_keys = sorted(
        str(key)
        for key in moving_sdata.points
        if not str(key).endswith(NONRIGID_ELEMENT_SUFFIX)
        and _point_coordinates_are_eligible(moving_sdata.points[key])
    )
    for points_key in native_point_keys:
        output_key = f"{points_key}{NONRIGID_ELEMENT_SUFFIX}"
        aligned = _transform_points(moving_sdata.points[points_key], result)
        detached_attrs = deepcopy(dict(aligned.attrs))
        aligned.attrs.clear()
        aligned.attrs.update(detached_attrs)
        set_transformation(aligned, Identity(), to_coordinate_system=coordinate_system)
        write_or_replace_element(
            moving_sdata,
            output_key,
            "points",
            aligned,
            overwrite=True,
        )
        set_artifact_status(
            manifest,
            kind="points",
            native_key=points_key,
            output_key=output_key,
            status="complete",
            qc={"columns_preserved": True, "transcript_ids_preserved": True},
            source_columns={
                str(column): str(dtype)
                for column, dtype in moving_sdata.points[points_key].dtypes.items()
            },
            output_columns={
                str(column): str(dtype) for column, dtype in aligned.dtypes.items()
            },
        )

    schema = stamp_merxen_schema(moving_sdata)
    registry = dict(schema.get("segmentations", {}))
    native_branches = {
        str(branch): dict(entry)
        for branch, entry in registry.items()
        if isinstance(entry, dict) and entry.get("coordinate_variant_of") is None
    }
    schema["segmentations"] = {
        str(branch): dict(entry)
        for branch, entry in registry.items()
        if isinstance(entry, dict)
        and (
            entry.get("coordinate_variant_of") is None
            or not _is_alignment_schema_variant(str(branch), entry)
        )
    }
    moving_sdata.attrs[MERXEN_SCHEMA_ATTR] = schema
    written_tables: dict[str, str] = {}
    for branch, entry in native_branches.items():
        points_key = str(entry["points"])
        shape_key = str(entry["shape"])
        aligned_points = f"{points_key}{NONRIGID_ELEMENT_SUFFIX}"
        aligned_shape = f"{shape_key}{NONRIGID_ELEMENT_SUFFIX}"
        if (
            aligned_points not in moving_sdata.points
            or aligned_shape not in moving_sdata.shapes
        ):
            continue
        native_table = entry.get("table")
        aligned_table = None
        if native_table is not None:
            native_table = str(native_table)
            aligned_table = f"{native_table}{NONRIGID_ELEMENT_SUFFIX}"
            previous_region = written_tables.get(aligned_table)
            if previous_region is not None and previous_region != aligned_shape:
                raise ValueError(
                    f"Native table {native_table!r} is shared by incompatible shapes"
                )
            if previous_region is None:
                table = clone_table_for_region(
                    moving_sdata.tables[native_table],
                    aligned_shape,
                )
                if "spatial" in table.obsm:
                    aligned_centroids = mapper.moving_dataset_to_fixed_dataset(
                        np.asarray(table.obsm["spatial"], dtype=np.float64)[:, :2]
                    )
                    table.obsm["spatial"] = aligned_centroids
                    table.obsm[f"spatial_{coordinate_system}"] = (
                        aligned_centroids.copy()
                    )
                write_or_replace_element(
                    moving_sdata,
                    aligned_table,
                    "tables",
                    table,
                    overwrite=True,
                )
                written_tables[aligned_table] = aligned_shape
                set_artifact_status(
                    manifest,
                    kind="tables",
                    native_key=native_table,
                    output_key=aligned_table,
                    status="complete",
                    qc={"count_matrix_identical": True},
                    source_shape=list(moving_sdata.tables[native_table].shape),
                    output_shape=list(table.shape),
                )
        register_segmentation_branch(
            moving_sdata,
            f"{branch}{NONRIGID_ELEMENT_SUFFIX}",
            points_key=aligned_points,
            assignment_column=entry.get("assignment_column"),
            background_column=entry.get("background_column"),
            assignment_source_column=entry.get("assignment_source_column"),
            shape_key=aligned_shape,
            table_key=aligned_table,
            instance_key=str(entry.get("instance_key", INSTANCE_ID_COLUMN)),
            id_namespace=str(entry.get("id_namespace", branch)),
            coordinate_variant_of=branch,
        )
    moving_sdata.attrs[ALIGNMENT_MANIFEST_ATTR] = manifest


def _transform_result_for_mapper_from_manifest(
    mapper: AlignmentCoordinateMapper,
    manifest: dict[str, Any],
) -> TransformResult:
    coordinate_system = str(manifest["common_coordinate_system"])
    if mapper.valis_bundle is not None:
        affine = mapper.valis_bundle.global_dataset_matrix
    elif mapper.legacy_transform is not None:
        affine = np.asarray(mapper.legacy_transform.affine_matrix)
    else:
        raise RuntimeError("Alignment mapper has no transform payload")
    return TransformResult(
        merscope_to_common={
            "selected_mode": mapper.selected_mode,
            "rigid_affine_matrix": affine.tolist(),
        },
        xenium_to_common={"type": "identity"},
        metadata={
            "backend": mapper.backend,
            "moving_platform": "MERSCOPE",
            "coordinate_system_name": coordinate_system,
            "parameters": {"mark_shared_tissue_domain": False},
        },
        nonrigid_transform=mapper.legacy_transform,
        valis_transform=mapper.valis_bundle,
    )


def _materialize_labels(
    *,
    moving_sdata: Any,
    moving_path: Path,
    native_masks: tuple[Any, ...],
    manifest: dict[str, Any],
    fixed_shape_rc: tuple[int, int],
    pixel_to_world: np.ndarray,
    coordinate_system: str,
    config: AlignmentConfig,
) -> None:
    params = config.materialization
    chunk = min(params.chunk_size, *fixed_shape_rc)
    cache_params = ViewerCacheParams(
        downsample=params.label_pyramid_downsample,
        label_chunk_size=params.chunk_size,
        contour_width=params.outline_width,
        min_size=params.pyramid_min_size,
        build_image_pyramid=False,
        force=True,
    )
    for mask in native_masks:
        aligned_shape = f"{mask.shape_key}{NONRIGID_ELEMENT_SUFFIX}"
        if aligned_shape not in moving_sdata.shapes:
            raise ValueError(
                f"Registered mask {mask.branch!r} lacks aligned shape {aligned_shape!r}"
            )
        label_key = f"{aligned_shape}_labels"
        result = build_mask_cache_on_grid(
            sdata=moving_sdata,
            zarr_path=moving_path,
            platform="MERSCOPE",
            shape_key=aligned_shape,
            label_key=label_key,
            shape=fixed_shape_rc,
            chunks=(chunk, chunk),
            pixel_to_world_matrix=pixel_to_world,
            coordinate_system=coordinate_system,
            params=cache_params,
        )
        label_pyramid_key = derived_label_pyramid_cache_key(
            label_key,
            params.label_pyramid_downsample,
        )
        outline_key = derived_outline_cache_key(label_key, params.outline_width)
        label_array = zarr.open_array(
            str(moving_path / "labels" / label_key / "s0"),
            mode="r",
        )
        set_artifact_status(
            manifest,
            kind="labels",
            native_key=mask.shape_key,
            output_key=label_key,
            status="complete",
            qc={"instance_ids_preserved": True},
            source_dtype=str(
                moving_sdata.shapes[aligned_shape][INSTANCE_ID_COLUMN].dtype
            ),
            output_dtype=str(label_array.dtype),
            shape=list(label_array.shape),
            chunks=list(label_array.chunks),
            pyramid_key=label_pyramid_key,
            outline_key=outline_key,
            cache_status=result,
            boundary_type=mask.boundary_type,
            has_assignment=mask.has_assignment,
            has_table=mask.has_table,
        )
    moving_sdata.attrs[ALIGNMENT_MANIFEST_ATTR] = manifest


def _validate_materialization(
    *,
    cfg: AlignmentConfig,
    manifest: dict[str, Any],
    mapper: AlignmentCoordinateMapper,
    moving_path: Path,
    fixed_sdata: Any,
    fixed_cyx: Any,
    fixed_x: np.ndarray,
    fixed_y: np.ndarray,
) -> dict[str, Any]:
    moving_sdata = sd.read_zarr(moving_path)
    coverage_qc = _validate_mapping_coverage(cfg, moving_sdata, manifest)
    for kind, mappings in manifest.get("mappings", {}).items():
        container = getattr(moving_sdata, kind)
        for output_key in mappings.values():
            if output_key not in container:
                raise ValueError(f"Required {kind} artifact {output_key!r} is missing")

    points_qc = _validate_points(moving_sdata, manifest)
    shapes_qc = _validate_shapes(moving_sdata, manifest)
    tables_qc = _validate_tables(moving_sdata, manifest)
    raster_qc = _validate_rasters(
        cfg=cfg,
        moving_sdata=moving_sdata,
        fixed_sdata=fixed_sdata,
        fixed_cyx=fixed_cyx,
        manifest=manifest,
    )
    roundtrip_qc = _validate_roundtrip(
        cfg=cfg,
        mapper=mapper,
        fixed_x=fixed_x,
        fixed_y=fixed_y,
    )
    validate_merxen_schema(moving_sdata, deep=True)
    return {
        "coverage": coverage_qc,
        "points": points_qc,
        "shapes": shapes_qc,
        "tables": tables_qc,
        "rasters": raster_qc,
        "roundtrip": roundtrip_qc,
        "schema_valid": True,
    }


def _validate_mapping_coverage(
    cfg: AlignmentConfig,
    sdata_obj: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify that every enabled native input has a declared complete artifact."""
    mappings = manifest.get("mappings", {})
    expected: dict[str, set[str]] = {
        "points": set(),
        "shapes": set(),
        "tables": set(),
        "labels": set(),
        "images": set(),
    }
    if cfg.materialization.materialize_vectors:
        expected["points"] = {
            str(key)
            for key in sdata_obj.points
            if not str(key).endswith(NONRIGID_ELEMENT_SUFFIX)
            and _point_coordinates_are_eligible(sdata_obj.points[key])
        }
        expected["shapes"] = {
            str(key)
            for key in sdata_obj.shapes
            if not str(key).endswith(NONRIGID_ELEMENT_SUFFIX)
        }
        registry = dict(
            sdata_obj.attrs.get(MERXEN_SCHEMA_ATTR, {}).get("segmentations", {})
        )
        expected["tables"] = {
            str(entry["table"])
            for entry in registry.values()
            if isinstance(entry, dict)
            and entry.get("coordinate_variant_of") is None
            and entry.get("table") is not None
        }
    if cfg.materialization.materialize_labels:
        expected["labels"] = {mask.shape_key for mask in discover_masks(sdata_obj)}
    if cfg.materialization.materialize_image:
        expected["images"] = {cfg.materialization.source_image_key}

    for kind, required in expected.items():
        declared = set(map(str, mappings.get(kind, {})))
        missing = required - declared
        if missing:
            raise ValueError(
                f"Alignment manifest is missing {kind} mappings for {sorted(missing)}"
            )
    incomplete = sorted(
        artifact_key
        for artifact_key, artifact in manifest.get("artifacts", {}).items()
        if not isinstance(artifact, dict) or artifact.get("status") != "complete"
    )
    if incomplete:
        raise ValueError(f"Alignment artifacts are incomplete: {incomplete}")
    return {
        f"{kind}_native_inputs": len(required) for kind, required in expected.items()
    }


def _validate_points(sdata_obj: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    checked = 0
    transcript_id_elements = 0
    registry = dict(sdata_obj.attrs[MERXEN_SCHEMA_ATTR].get("segmentations", {}))
    assignment_columns = {
        str(entry[field])
        for entry in registry.values()
        if isinstance(entry, dict) and entry.get("coordinate_variant_of") is None
        for field in (
            "assignment_column",
            "background_column",
            "assignment_source_column",
        )
        if entry.get(field) is not None
    }
    for native_key, output_key in (
        manifest.get("mappings", {}).get("points", {}).items()
    ):
        native = sdata_obj.points[native_key]
        aligned = sdata_obj.points[output_key]
        if not set(native.columns).issubset(set(aligned.columns)):
            raise ValueError(f"Aligned points {output_key!r} dropped native columns")
        columns = sorted(assignment_columns & set(native.columns))
        if TRANSCRIPT_ID_COLUMN in native.columns:
            columns.insert(0, TRANSCRIPT_ID_COLUMN)
            transcript_id_elements += 1
        if columns:
            native_frame = _compute_frame(native[columns])
            aligned_frame = _compute_frame(aligned[columns])
            if TRANSCRIPT_ID_COLUMN in columns:
                native_frame = native_frame.sort_values(TRANSCRIPT_ID_COLUMN)
                aligned_frame = aligned_frame.sort_values(TRANSCRIPT_ID_COLUMN)
            _assert_preserved_point_frame(
                native_frame.reset_index(drop=True),
                aligned_frame.reset_index(drop=True),
            )
        checked += 1
    return {
        "elements_checked": checked,
        "transcript_id_elements_checked": transcript_id_elements,
        "ids_and_assignments_identical": True,
    }


def _validate_shapes(sdata_obj: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    checked = 0
    for native_key, output_key in (
        manifest.get("mappings", {}).get("shapes", {}).items()
    ):
        native_ids = canonical_shape_instance_series(
            sdata_obj.shapes[native_key],
            field_name=native_key,
        )
        aligned_ids = canonical_shape_instance_series(
            sdata_obj.shapes[output_key],
            field_name=output_key,
        )
        if set(map(int, native_ids)) != set(map(int, aligned_ids)):
            raise ValueError(f"Aligned shape {output_key!r} changed instance IDs")
        checked += 1
    return {"elements_checked": checked, "instance_ids_identical": True}


def _validate_tables(sdata_obj: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    checked = 0
    for native_key, output_key in (
        manifest.get("mappings", {}).get("tables", {}).items()
    ):
        native = sdata_obj.tables[native_key]
        aligned = sdata_obj.tables[output_key]
        if not _matrix_equal(native.X, aligned.X):
            raise ValueError(f"Aligned table {output_key!r} changed its count matrix")
        native_ids = np.asarray(native.obs[INSTANCE_ID_COLUMN], dtype=np.uint64)
        aligned_ids = np.asarray(aligned.obs[INSTANCE_ID_COLUMN], dtype=np.uint64)
        if not np.array_equal(native_ids, aligned_ids):
            raise ValueError(f"Aligned table {output_key!r} changed instance IDs")
        checked += 1
    return {"elements_checked": checked, "count_matrices_identical": True}


def _validate_rasters(
    *,
    cfg: AlignmentConfig,
    moving_sdata: Any,
    fixed_sdata: Any,
    fixed_cyx: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    expected_shape = (int(fixed_cyx.sizes["y"]), int(fixed_cyx.sizes["x"]))
    pixel_affine = np.asarray(manifest["fixed_grid"]["pixel_to_world_affine"])
    label_count = 0
    for output_key in manifest.get("mappings", {}).get("labels", {}).values():
        labels = image_to_cyx(moving_sdata.labels[output_key]).squeeze("c", drop=True)
        if tuple(map(int, labels.shape)) != expected_shape:
            raise ValueError(f"Aligned labels {output_key!r} do not match fixed grid")
        transformation = get_transformation(
            moving_sdata.labels[output_key],
            to_coordinate_system=manifest["common_coordinate_system"],
        )
        matrix = transformation.to_affine_matrix(
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        if not np.allclose(matrix, pixel_affine, atol=1e-8):
            raise ValueError(f"Aligned labels {output_key!r} have the wrong affine")
        label_count += 1

    image_qc: dict[str, Any] = {}
    output_key = cfg.materialization.output_image_key
    if cfg.materialization.materialize_image:
        aligned = image_to_cyx(moving_sdata.images[output_key])
        if tuple(map(int, aligned.shape[-2:])) != expected_shape:
            raise ValueError("Aligned image dimensions do not match the fixed image")
        transformation = get_transformation(
            moving_sdata.images[output_key],
            to_coordinate_system=manifest["common_coordinate_system"],
        )
        matrix = transformation.to_affine_matrix(
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )
        if not np.allclose(matrix, pixel_affine, atol=1e-8):
            raise ValueError("Aligned image has the wrong fixed-grid affine")
        source = image_to_cyx(moving_sdata.images[cfg.materialization.source_image_key])
        if aligned.dtype != source.dtype:
            raise ValueError("Aligned image dtype differs from the source image")
        if _channel_labels(aligned) != _channel_labels(source):
            raise ValueError("Aligned image channels differ from the source image")
        nonzero_fraction = float(
            manifest["artifacts"][f"images/{output_key}"]["qc"]["nonzero_fraction"]
        )
        if not 0.0 < nonzero_fraction <= 1.0:
            raise ValueError("Aligned image has no nonzero source support")
        image_qc = _dapi_agreement_qc(cfg, fixed_sdata, fixed_cyx, aligned)
        image_qc["nonzero_fraction"] = nonzero_fraction
        seams_validated = bool(
            manifest["artifacts"][f"images/{output_key}"]["qc"]["tile_seams_validated"]
        )
        if not seams_validated:
            raise ValueError("Aligned image tile-seam validation did not pass")
        image_qc["tile_seams_validated"] = seams_validated
    return {
        "labels_checked": label_count,
        "fixed_grid_shape": list(expected_shape),
        "image": image_qc,
    }


def _validate_roundtrip(
    *,
    cfg: AlignmentConfig,
    mapper: AlignmentCoordinateMapper,
    fixed_x: np.ndarray,
    fixed_y: np.ndarray,
) -> dict[str, Any]:
    spacing = max(1, int(cfg.materialization.roundtrip_sample_spacing))
    sample_x = np.unique(np.append(fixed_x[::spacing], fixed_x[-1]))
    sample_y = np.unique(np.append(fixed_y[::spacing], fixed_y[-1]))
    xx, yy = np.meshgrid(sample_x, sample_y)
    fixed_image_xy = np.column_stack([xx.ravel(), yy.ravel()])
    fixed_dataset_xy = _apply_affine(
        fixed_image_xy,
        np.linalg.inv(mapper.fixed_dataset_to_image_matrix),
    )
    errors = mapper.roundtrip_error(fixed_dataset_xy)
    maximum = float(np.max(errors)) if len(errors) else 0.0
    percentile_95 = float(np.percentile(errors, 95)) if len(errors) else 0.0
    typical_tolerance, maximum_tolerance = _roundtrip_tolerances(cfg, mapper)
    if (
        not np.isfinite(maximum)
        or not np.isfinite(percentile_95)
        or percentile_95 > typical_tolerance
        or maximum > maximum_tolerance
    ):
        raise ValueError(
            "Alignment inverse round-trip error exceeds tolerance: "
            f"p95={percentile_95:.6g} µm (limit {typical_tolerance:.6g} µm), "
            f"maximum={maximum:.6g} µm (limit {maximum_tolerance:.6g} µm)"
        )
    return {
        "sample_count": int(len(errors)),
        "maximum_error_um": maximum,
        "p95_error_um": percentile_95,
        "mean_error_um": float(np.mean(errors)) if len(errors) else 0.0,
        "p95_tolerance_um": typical_tolerance,
        "maximum_tolerance_um": maximum_tolerance,
    }


def _roundtrip_tolerances(
    cfg: AlignmentConfig,
    mapper: AlignmentCoordinateMapper,
) -> tuple[float, float]:
    if mapper.backend != "valis":
        tolerance = float(cfg.materialization.legacy_inverse_tolerance_um)
        return tolerance, tolerance
    bundle = mapper.valis_bundle
    if bundle is None or bundle.backward_displacement is None:
        return (
            VALIS_ROUNDTRIP_P95_TOLERANCE_UM,
            VALIS_ROUNDTRIP_P95_TOLERANCE_UM,
        )
    field = bundle.backward_displacement
    x_step = float(np.median(np.diff(np.asarray(field.x_coordinates))))
    y_step = float(np.median(np.diff(np.asarray(field.y_coordinates))))
    registration_to_dataset = np.asarray(
        bundle.fixed_dataset_from_registration_matrix,
        dtype=np.float64,
    )[:2, :2]
    x_step_um = float(
        np.linalg.norm(registration_to_dataset @ np.asarray([x_step, 0.0]))
    )
    y_step_um = float(
        np.linalg.norm(registration_to_dataset @ np.asarray([0.0, y_step]))
    )
    maximum_tolerance = max(
        VALIS_ROUNDTRIP_P95_TOLERANCE_UM,
        VALIS_ROUNDTRIP_MAX_FIELD_STEP_FRACTION * min(x_step_um, y_step_um),
    )
    return VALIS_ROUNDTRIP_P95_TOLERANCE_UM, maximum_tolerance


def _dapi_agreement_qc(
    cfg: AlignmentConfig,
    fixed_sdata: Any,
    fixed_cyx: Any,
    aligned_cyx: Any,
) -> dict[str, Any]:
    from merxen.alignment.qc import compute_dapi_metrics

    fixed_index = _channel_index(fixed_cyx, cfg.xenium_image.dapi_channel)
    aligned_index = _channel_index(aligned_cyx, cfg.merscope_image.dapi_channel)
    max_dimension = max(int(fixed_cyx.sizes["y"]), int(fixed_cyx.sizes["x"]))
    step = max(1, int(np.ceil(max_dimension / 512)))
    fixed = _compute_array(fixed_cyx.data[fixed_index, ::step, ::step])
    aligned = _compute_array(aligned_cyx.data[aligned_index, ::step, ::step])
    fixed_mask = np.isfinite(fixed) & (fixed != 0)
    aligned_mask = np.isfinite(aligned) & (aligned != 0)
    metrics = compute_dapi_metrics(fixed, aligned, fixed_mask, aligned_mask)
    nmi = float(metrics["normalized_mutual_information"])
    if not np.isfinite(nmi):
        raise ValueError("Aligned DAPI agreement metric is not finite")
    return {
        "downsample": step,
        "normalized_mutual_information": nmi,
        "density_correlation": float(metrics["density_correlation"]),
        "fixed_overlap_fraction": float(metrics["fixed_overlap_fraction"]),
        "moving_overlap_fraction": float(metrics["moving_overlap_fraction"]),
    }


def _can_reuse_complete_manifest(
    manifest: dict[str, Any],
    native_revision: dict[str, Any],
    force: bool,
) -> bool:
    return bool(
        not force
        and manifest.get("complete")
        and manifest.get("native_input", {}).get("fingerprint")
        == native_revision.get("fingerprint")
    )


def _can_recover_complete_artifacts(
    manifest: dict[str, Any],
    native_revision: dict[str, Any],
    force: bool,
) -> bool:
    artifacts = manifest.get("artifacts")
    return bool(
        not force
        and not manifest.get("complete")
        and manifest.get("native_input", {}).get("fingerprint")
        == native_revision.get("fingerprint")
        and isinstance(artifacts, dict)
        and artifacts
        and all(
            isinstance(artifact, dict) and artifact.get("status") == "complete"
            for artifact in artifacts.values()
        )
    )


def _assert_preserved_point_frame(
    native_frame: pd.DataFrame,
    aligned_frame: pd.DataFrame,
) -> None:
    pd.testing.assert_index_equal(native_frame.columns, aligned_frame.columns)
    for column in native_frame.columns:
        native = native_frame[column]
        aligned = aligned_frame[column]
        native_dtype = native.dtype
        aligned_dtype = aligned.dtype
        both_unordered_categorical = bool(
            isinstance(native_dtype, pd.CategoricalDtype)
            and isinstance(aligned_dtype, pd.CategoricalDtype)
            and not native_dtype.ordered
            and not aligned_dtype.ordered
        )
        pd.testing.assert_series_equal(
            native,
            aligned,
            check_dtype=True,
            check_categorical=True,
            check_category_order=not both_unordered_categorical,
        )


def _finalize_materialization(
    *,
    moving_sdata: Any,
    fixed_sdata: Any,
    manifest: dict[str, Any],
    qc: dict[str, Any],
    pair_id: str,
    native_revision: dict[str, Any],
) -> None:
    manifest["qc"] = qc
    manifest["complete"] = True
    manifest["completed_at"] = utc_timestamp()
    manifest.pop("invalidated_at", None)
    manifest.pop("invalidation_reason", None)
    moving_sdata.attrs[ALIGNMENT_MANIFEST_ATTR] = manifest
    transform_fingerprint = str(manifest["transform"]["fingerprint"])
    fixed_sdata.attrs[ALIGNMENT_PAIR_REFERENCE_ATTR] = alignment_pair_reference(
        pair_id=pair_id,
        counterpart_fingerprint=str(native_revision["fingerprint"]),
        transform_fingerprint=transform_fingerprint,
    )
    write_spatialdata_metadata(moving_sdata, write_attrs=True)
    write_spatialdata_metadata(fixed_sdata, write_attrs=True)


def _validate_pair_reference(
    fixed_sdata: Any,
    pair_id: str,
    *,
    moving_manifest: Any,
) -> None:
    reference = fixed_sdata.attrs.get(ALIGNMENT_PAIR_REFERENCE_ATTR)
    if isinstance(reference, dict) and reference.get("pair_id") not in (None, pair_id):
        raise ValueError(
            "Xenium store is stamped for alignment pair "
            f"{reference.get('pair_id')!r}, not {pair_id!r}"
        )
    if not isinstance(reference, dict) or not isinstance(moving_manifest, dict):
        return
    moving_transform = moving_manifest.get("transform", {})
    moving_fingerprint = (
        moving_transform.get("fingerprint")
        if isinstance(moving_transform, dict)
        else None
    )
    fixed_fingerprint = reference.get("transform_fingerprint")
    if (
        moving_fingerprint is not None
        and fixed_fingerprint is not None
        and moving_fingerprint != fixed_fingerprint
    ):
        raise ValueError(
            "Xenium and MERSCOPE stores reference different alignment transforms"
        )


def _ensure_pair_reference(
    fixed_sdata: Any,
    *,
    pair_id: str,
    native_fingerprint: str,
    transform_fingerprint: str,
) -> bool:
    reference = fixed_sdata.attrs.get(ALIGNMENT_PAIR_REFERENCE_ATTR)
    expected = {
        "pair_id": str(pair_id),
        "counterpart_native_fingerprint": str(native_fingerprint),
        "transform_fingerprint": str(transform_fingerprint),
    }
    if isinstance(reference, dict) and all(
        reference.get(key) == value for key, value in expected.items()
    ):
        return False
    fixed_sdata.attrs[ALIGNMENT_PAIR_REFERENCE_ATTR] = alignment_pair_reference(
        pair_id=pair_id,
        counterpart_fingerprint=native_fingerprint,
        transform_fingerprint=transform_fingerprint,
    )
    return True


def _axis_coordinates(image_cyx: Any, axis: str) -> np.ndarray:
    if axis in image_cyx.coords:
        values = np.asarray(image_cyx.coords[axis].values, dtype=np.float64)
    else:
        values = np.arange(int(image_cyx.sizes[axis]), dtype=np.float64)
    if values.ndim != 1 or len(values) != int(image_cyx.sizes[axis]):
        raise ValueError(f"Fixed image {axis!r} coordinates are invalid")
    return values


def _index_to_coordinate_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_origin, x_step = coords_origin_step(x)
    y_origin, y_step = coords_origin_step(y)
    return np.asarray(
        [[x_step, 0.0, x_origin], [0.0, y_step, y_origin], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _channel_labels(image_cyx: Any) -> list[str]:
    if "c" in image_cyx.coords:
        return [str(value) for value in image_cyx.coords["c"].values]
    return [f"c{index}" for index in range(int(image_cyx.sizes.get("c", 1)))]


def _channel_index(image_cyx: Any, requested: str) -> int:
    channels = _channel_labels(image_cyx)
    matches = [
        index
        for index, name in enumerate(channels)
        if name.lower() == requested.lower()
    ]
    if not matches and len(channels) == 1:
        return 0
    if len(matches) != 1:
        raise ValueError(
            f"DAPI channel {requested!r} is not unique in channels {channels}"
        )
    return matches[0]


def _compute_frame(value: Any) -> pd.DataFrame:
    return value.compute() if hasattr(value, "compute") else pd.DataFrame(value)


def _point_coordinates_are_eligible(points_obj: Any) -> bool:
    x_column = first_existing_col(
        points_obj,
        ["x", "x_micron", "x_location", "global_x", "x_global_px", "observed_x"],
    )
    y_column = first_existing_col(
        points_obj,
        ["y", "y_micron", "y_location", "global_y", "y_global_px", "observed_y"],
    )
    return x_column is not None and y_column is not None


def _is_alignment_schema_variant(branch: str, entry: dict[str, Any]) -> bool:
    return bool(
        str(branch).endswith(NONRIGID_ELEMENT_SUFFIX)
        or str(entry.get("points", "")).endswith(NONRIGID_ELEMENT_SUFFIX)
        or str(entry.get("shape", "")).endswith(NONRIGID_ELEMENT_SUFFIX)
    )


def _compute_array(value: Any) -> np.ndarray:
    return np.asarray(value.compute() if hasattr(value, "compute") else value)


def _matrix_equal(left: Any, right: Any) -> bool:
    if left.shape != right.shape:
        return False
    difference = left != right
    if hasattr(difference, "nnz"):
        return int(difference.nnz) == 0
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _apply_affine(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([xy, np.ones(len(xy), dtype=np.float64)])
    return (homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :2]
