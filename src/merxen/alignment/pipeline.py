"""CLI/pipeline orchestration for MerXen alignment."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from shapely.ops import transform as shapely_transform
from spatialdata.transformations import Affine, Identity, set_transformation

from merxen.alignment.register import (
    TransformResult,
    register_pair,
    transform_xy_for_result,
    write_valis_resume_manifest,
)
from merxen.alignment.transforms import apply_affine_matrix
from merxen.config import AlignmentConfig
from merxen.io.spatialdata_io import (
    write_or_replace_element,
    write_spatialdata_metadata,
)
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    register_segmentation_branch,
    stamp_merxen_schema,
)
from merxen.io.transcript_io import first_existing_col

MERXEN_ALIGNMENT_ATTR = "merxen_alignment"
ALIGNMENT_COORDINATE_SYSTEM = "merxen_xenium"
NONRIGID_ELEMENT_SUFFIX = "_aligned_nonrigid"
logger = logging.getLogger(__name__)


def run_alignment_pipeline(config: AlignmentConfig) -> dict[str, Path]:
    """Run paired-section alignment and write stage outputs."""
    cfg = config
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    xenium_sdata = sd.read_zarr(cfg.xenium_zarr_path)
    merscope_sdata = sd.read_zarr(cfg.merscope_zarr_path)
    try:
        result = register_pair(merscope_sdata, xenium_sdata, cfg)
    except Exception as exc:
        dependency_versions: dict[str, str] = {}
        if cfg.backend == "valis":
            from merxen.alignment.valis_register import (
                dependency_versions as collect_dependency_versions,
            )

            dependency_versions = collect_dependency_versions()
        failure_payload = {
            "pair_id": cfg.pair_id,
            "backend": cfg.backend,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "parameters": (
                cfg.valis.model_dump(mode="json")
                if cfg.backend == "valis"
                else cfg.legacy_spateo.model_dump(mode="json")
            ),
            "dependency_versions": dependency_versions,
        }
        failure_path = cfg.output_dir / "alignment_failure.json"
        failure_path.write_text(json.dumps(failure_payload, indent=2))
        summary_path = cfg.output_dir / "registration_summary.json"
        if not summary_path.exists():
            summary_path.write_text(json.dumps(failure_payload, indent=2))
        summary_csv = cfg.output_dir / "registration_summary.csv"
        if not summary_csv.exists():
            pd.json_normalize(
                {
                    key: value
                    for key, value in failure_payload.items()
                    if key != "traceback"
                },
                sep=".",
            ).to_csv(summary_csv, index=False)
        logger.exception("Alignment failed for %s", cfg.pair_id)
        raise

    coords_dir = cfg.output_dir / "alignment_coords"
    coords_dir.mkdir(parents=True, exist_ok=True)
    if result.coordinate_tables is not None:
        for name, table in result.coordinate_tables.items():
            table.to_csv(
                coords_dir / f"{cfg.pair_id}_{name}_alignment_coords.csv",
                index=False,
            )

    transform_json = cfg.output_dir / "alignment_transform.json"
    _write_transform_json(result, transform_json)

    if cfg.write_aligned_zarrs:
        moving_zarr_path = (
            cfg.merscope_zarr_path
            if cfg.moving_platform == "MERSCOPE"
            else cfg.xenium_zarr_path
        )
        _write_moving_alignment_to_zarr(moving_zarr_path, result)
    if cfg.backend == "valis":
        write_valis_resume_manifest(cfg)

    return {
        "merscope_zarr": cfg.merscope_zarr_path,
        "xenium_zarr": cfg.xenium_zarr_path,
        "transform_json": transform_json,
        "coords_dir": coords_dir,
    }


def _write_transform_json(result: TransformResult, path: Path) -> None:
    payload = {
        "merscope_to_common": result.merscope_to_common,
        "xenium_to_common": result.xenium_to_common,
        "nonrigid_transform": _nonrigid_transform_payload(result),
        "metadata": result.metadata,
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2))


def _write_moving_alignment_to_zarr(
    zarr_path: Path,
    result: TransformResult,
) -> None:
    sdata_obj = sd.read_zarr(zarr_path)
    sdata_obj.attrs[MERXEN_ALIGNMENT_ATTR] = _alignment_attrs_payload(result)

    coordinate_system = _alignment_coordinate_system(result)
    rigid = _rigid_affine_transformation(result)
    nonrigid_identity = Identity()

    shape_keys = (
        [
            key
            for key in list(sdata_obj.shapes.keys())
            if not key.endswith(NONRIGID_ELEMENT_SUFFIX)
        ]
        if _valis_setting(result, "transform_polygons", default=True)
        else []
    )
    for key in shape_keys:
        set_transformation(
            sdata_obj.shapes[key],
            rigid,
            to_coordinate_system=coordinate_system,
        )
        aligned_key = _nonrigid_element_key(key)
        aligned_shapes = _transform_shapes(sdata_obj.shapes[key], result)
        set_transformation(
            aligned_shapes,
            nonrigid_identity,
            to_coordinate_system=coordinate_system,
        )
        write_or_replace_element(
            sdata_obj,
            aligned_key,
            "shapes",
            aligned_shapes,
            overwrite=True,
        )

    point_keys = (
        [
            key
            for key in list(sdata_obj.points.keys())
            if not key.endswith(NONRIGID_ELEMENT_SUFFIX)
        ]
        if _valis_setting(result, "transform_transcripts", default=True)
        else []
    )
    for key in point_keys:
        set_transformation(
            sdata_obj.points[key],
            rigid,
            to_coordinate_system=coordinate_system,
        )
        aligned_key = _nonrigid_element_key(key)
        aligned_points = _transform_points(sdata_obj.points[key], result)
        set_transformation(
            aligned_points,
            nonrigid_identity,
            to_coordinate_system=coordinate_system,
        )
        write_or_replace_element(
            sdata_obj,
            aligned_key,
            "points",
            aligned_points,
            overwrite=True,
        )

    if _valis_setting(result, "transform_centroids", default=True):
        _add_registered_table_centroids(sdata_obj, result)
    stamp_merxen_schema(sdata_obj)
    schema = dict(sdata_obj.attrs.get(MERXEN_SCHEMA_ATTR, {}))
    native_branches = dict(schema.get("segmentations", {}))
    for branch, entry in native_branches.items():
        if entry.get("coordinate_variant_of") is not None:
            continue
        points_key = str(entry.get("points", ""))
        shape_key = str(entry.get("shape", ""))
        aligned_points_key = _nonrigid_element_key(points_key)
        aligned_shape_key = _nonrigid_element_key(shape_key)
        if (
            aligned_points_key not in sdata_obj.points
            or aligned_shape_key not in sdata_obj.shapes
        ):
            continue
        register_segmentation_branch(
            sdata_obj,
            f"{branch}{NONRIGID_ELEMENT_SUFFIX}",
            points_key=aligned_points_key,
            assignment_column=entry.get("assignment_column"),
            background_column=entry.get("background_column"),
            assignment_source_column=entry.get("assignment_source_column"),
            shape_key=aligned_shape_key,
            table_key=None,
            instance_key=str(entry.get("instance_key", "instance_id")),
            id_namespace=str(entry.get("id_namespace", branch)),
            coordinate_variant_of=str(branch),
        )

    write_spatialdata_metadata(
        sdata_obj,
        write_attrs=True,
        write_transformations=True,
    )


def _nonrigid_element_key(key: str) -> str:
    return f"{key}{NONRIGID_ELEMENT_SUFFIX}"


def _rigid_affine_transformation(result: TransformResult) -> Affine:
    moving_transform = _moving_transform_payload(result)
    matrix = np.asarray(
        moving_transform["rigid_affine_matrix"],
        dtype=np.float64,
    )
    return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))


def _alignment_attrs_payload(result: TransformResult) -> dict[str, Any]:
    payload = _jsonable(
        {
            "version": 1,
            "backend": result.metadata.get("backend", "legacy_spateo"),
            "alignment_coordinate_system": _alignment_coordinate_system(result),
            "nonrigid_element_suffix": NONRIGID_ELEMENT_SUFFIX,
            "selected_mode": _moving_transform_payload(result).get("selected_mode"),
            "rigid_affine_matrix": _moving_transform_payload(result).get(
                "rigid_affine_matrix"
            ),
            "nonrigid_transform": _nonrigid_transform_payload(result),
            "metadata": result.metadata,
        }
    )
    return cast(dict[str, Any], payload)


def _nonrigid_transform_payload(result: TransformResult) -> dict[str, Any] | None:
    if result.valis_transform is not None:
        return {
            "type": "valis_sampled_displacement_field",
            "selected_mode": result.valis_transform.selected_mode,
            "transform_chain": result.valis_transform.to_metadata(),
        }
    transform = result.nonrigid_transform
    if transform is None:
        return None
    return {
        "type": "affine_plus_rbf_residual",
        "affine_matrix": transform.affine_matrix,
        "anchors": transform.anchors,
        "residuals": transform.residuals,
        "neighbors": transform.neighbors,
        "smoothing": transform.smoothing,
        "support_radius": transform.support_radius,
    }


def _transform_shapes(
    shapes: gpd.GeoDataFrame,
    result: TransformResult,
) -> gpd.GeoDataFrame:
    gdf = shapes.copy()
    if "geometry" not in gdf.columns:
        gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry)

    def _xy_func(x: Any, y: Any, z: Any | None = None) -> Any:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        coords = np.column_stack([x_arr.ravel(), y_arr.ravel()])
        out = transform_xy_for_result(result, coords)
        ox = out[:, 0].reshape(x_arr.shape)
        oy = out[:, 1].reshape(y_arr.shape)
        if z is None:
            return ox, oy
        return ox, oy, z

    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: (
            shapely_transform(_xy_func, geom)
            if geom is not None and not geom.is_empty
            else geom
        )
    )
    if result.valid_domain_mask is not None and _valis_setting(
        result,
        "mark_shared_tissue_domain",
        default=True,
    ):
        representative = gdf.geometry.representative_point()
        xy = np.column_stack(
            [
                representative.x.to_numpy(dtype=float),
                representative.y.to_numpy(dtype=float),
            ]
        )
        gdf["in_shared_tissue_domain"] = _inside_valid_domain(result, xy)
    return gdf


def _transform_points(points_obj: Any, result: TransformResult) -> Any:
    x_col = first_existing_col(
        points_obj,
        ["x", "x_micron", "x_location", "global_x", "x_global_px", "observed_x"],
    )
    y_col = first_existing_col(
        points_obj,
        ["y", "y_micron", "y_location", "global_y", "y_global_px", "observed_y"],
    )
    if x_col is None or y_col is None:
        return points_obj

    def _part(part: pd.DataFrame) -> pd.DataFrame:
        out = part.copy()
        xy = out[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(xy).all(axis=1)
        if np.any(valid):
            aligned = transform_xy_for_result(result, xy[valid])
            out.loc[valid, f"raw_{x_col}"] = xy[valid, 0]
            out.loc[valid, f"raw_{y_col}"] = xy[valid, 1]
            out.loc[valid, x_col] = aligned[:, 0]
            out.loc[valid, y_col] = aligned[:, 1]
        return out

    if hasattr(points_obj, "map_partitions"):
        meta = points_obj._meta.copy()
        meta[f"raw_{x_col}"] = pd.Series(dtype="float64")
        meta[f"raw_{y_col}"] = pd.Series(dtype="float64")
        return points_obj.map_partitions(_part, meta=meta)
    return _part(points_obj)


def _add_registered_table_centroids(
    sdata_obj: Any,
    result: TransformResult,
) -> None:
    coordinate_system = _alignment_coordinate_system(result)
    for key in list(sdata_obj.tables.keys()):
        table = sdata_obj.tables[key].copy()
        if "spatial" not in table.obsm:
            continue
        coords = np.asarray(table.obsm["spatial"], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] < 2:
            continue
        aligned = transform_xy_for_result(result, coords[:, :2])
        table.obsm[f"spatial_{coordinate_system}"] = aligned
        if result.valid_domain_mask is not None and _valis_setting(
            result,
            "mark_shared_tissue_domain",
            default=True,
        ):
            table.obs["in_shared_tissue_domain"] = _inside_valid_domain(
                result,
                aligned,
            )
        write_or_replace_element(
            sdata_obj,
            key,
            "tables",
            table,
            overwrite=True,
        )


def _inside_valid_domain(
    result: TransformResult,
    fixed_dataset_xy: np.ndarray,
) -> np.ndarray:
    if result.valid_domain_mask is None or result.valis_transform is None:
        return np.zeros(len(fixed_dataset_xy), dtype=bool)
    fixed_to_registration = np.asarray(
        result.valis_transform.fixed_image_to_registration,
        dtype=np.float64,
    ) @ np.asarray(
        result.valis_transform.fixed_dataset_to_image,
        dtype=np.float64,
    )
    registration_xy = apply_affine_matrix(fixed_dataset_xy, fixed_to_registration)
    cols = np.rint(registration_xy[:, 0]).astype(np.int64)
    rows = np.rint(registration_xy[:, 1]).astype(np.int64)
    mask = np.asarray(result.valid_domain_mask) > 0
    valid = (rows >= 0) & (rows < mask.shape[0]) & (cols >= 0) & (cols < mask.shape[1])
    inside = np.zeros(len(registration_xy), dtype=bool)
    inside[valid] = mask[rows[valid], cols[valid]]
    return inside


def _alignment_coordinate_system(result: TransformResult) -> str:
    return str(
        result.metadata.get("coordinate_system_name", ALIGNMENT_COORDINATE_SYSTEM)
    )


def _moving_transform_payload(result: TransformResult) -> dict[str, Any]:
    moving_platform = str(result.metadata.get("moving_platform", "MERSCOPE"))
    return (
        result.merscope_to_common
        if moving_platform == "MERSCOPE"
        else result.xenium_to_common
    )


def _valis_setting(
    result: TransformResult,
    key: str,
    *,
    default: bool,
) -> bool:
    if result.metadata.get("backend") != "valis":
        return default
    parameters = result.metadata.get("parameters", {})
    if not isinstance(parameters, dict):
        return default
    return bool(parameters.get(key, default))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value
