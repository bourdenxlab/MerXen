"""CLI/pipeline orchestration for MerXen alignment."""

from __future__ import annotations

import json
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
)
from merxen.config import AlignmentConfig
from merxen.io.spatialdata_io import (
    write_or_replace_element,
    write_spatialdata_metadata,
)
from merxen.io.transcript_io import first_existing_col

MERXEN_ALIGNMENT_ATTR = "merxen_alignment"
ALIGNMENT_COORDINATE_SYSTEM = "merxen_xenium"
NONRIGID_ELEMENT_SUFFIX = "_aligned_nonrigid"


def run_alignment_pipeline(config: AlignmentConfig) -> dict[str, Path]:
    """Run paired-section alignment and write stage outputs."""
    cfg = config
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    xenium_sdata = sd.read_zarr(cfg.xenium_zarr_path)
    merscope_sdata = sd.read_zarr(cfg.merscope_zarr_path)
    result = register_pair(merscope_sdata, xenium_sdata, cfg)

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
        _write_moving_alignment_to_zarr(cfg.merscope_zarr_path, result)

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

    rigid = _rigid_affine_transformation(result)
    nonrigid_identity = Identity()

    for key in [
        k
        for k in list(sdata_obj.shapes.keys())
        if not k.endswith(NONRIGID_ELEMENT_SUFFIX)
    ]:
        set_transformation(
            sdata_obj.shapes[key],
            rigid,
            to_coordinate_system=ALIGNMENT_COORDINATE_SYSTEM,
        )
        aligned_key = _nonrigid_element_key(key)
        aligned_shapes = _transform_shapes(sdata_obj.shapes[key], result)
        set_transformation(
            aligned_shapes,
            nonrigid_identity,
            to_coordinate_system=ALIGNMENT_COORDINATE_SYSTEM,
        )
        write_or_replace_element(
            sdata_obj,
            aligned_key,
            "shapes",
            aligned_shapes,
            overwrite=True,
        )

    for key in [
        k
        for k in list(sdata_obj.points.keys())
        if not k.endswith(NONRIGID_ELEMENT_SUFFIX)
    ]:
        set_transformation(
            sdata_obj.points[key],
            rigid,
            to_coordinate_system=ALIGNMENT_COORDINATE_SYSTEM,
        )
        aligned_key = _nonrigid_element_key(key)
        aligned_points = _transform_points(sdata_obj.points[key], result)
        set_transformation(
            aligned_points,
            nonrigid_identity,
            to_coordinate_system=ALIGNMENT_COORDINATE_SYSTEM,
        )
        write_or_replace_element(
            sdata_obj,
            aligned_key,
            "points",
            aligned_points,
            overwrite=True,
        )

    write_spatialdata_metadata(
        sdata_obj,
        write_attrs=True,
        write_transformations=True,
    )


def _nonrigid_element_key(key: str) -> str:
    return f"{key}{NONRIGID_ELEMENT_SUFFIX}"


def _rigid_affine_transformation(result: TransformResult) -> Affine:
    matrix = np.asarray(
        result.merscope_to_common["rigid_affine_matrix"],
        dtype=np.float64,
    )
    return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))


def _alignment_attrs_payload(result: TransformResult) -> dict[str, Any]:
    payload = _jsonable(
        {
            "version": 1,
            "alignment_coordinate_system": ALIGNMENT_COORDINATE_SYSTEM,
            "nonrigid_element_suffix": NONRIGID_ELEMENT_SUFFIX,
            "selected_mode": result.merscope_to_common.get("selected_mode"),
            "rigid_affine_matrix": result.merscope_to_common.get("rigid_affine_matrix"),
            "nonrigid_transform": _nonrigid_transform_payload(result),
            "metadata": result.metadata,
        }
    )
    return cast(dict[str, Any], payload)


def _nonrigid_transform_payload(result: TransformResult) -> dict[str, Any] | None:
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
