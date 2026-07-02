"""Gradient-field streamline tracing for cortical-depth coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from merxen.cortical_depth.laplace import interpolate_scalar_field
from merxen.cortical_depth.ribbon import RibbonGrid, points_inside_mask


@dataclass(frozen=True)
class Streamline:
    """One pial-to-white-matter gradient streamline."""

    streamline_id: int
    points: np.ndarray
    tangential_position_um: float
    thickness_um: float
    reached_wm: bool
    near_side_boundary: bool
    qc_flag: str


def compute_normalized_gradient(phi: np.ndarray, grid: RibbonGrid) -> np.ndarray:
    """Compute a normalized 2D gradient vector field from a Laplace solution."""
    values = np.asarray(phi, dtype=float)
    valid = np.asarray(grid.mask, dtype=bool) & np.isfinite(values)
    if not valid.any():
        raise ValueError("Cannot compute gradient: phi has no finite ribbon values.")
    nearest = ndimage.distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )
    filled = values[tuple(nearest)]
    grad_y, grad_x = np.gradient(filled, grid.spec.step, grid.spec.step)
    gradient = np.stack([grad_x, grad_y], axis=-1).astype(np.float32, copy=False)
    norm = np.linalg.norm(gradient, axis=-1)
    keep = norm > 0
    gradient[keep] = gradient[keep] / norm[keep, None]
    gradient[~keep] = np.nan
    return gradient


def trace_streamlines(
    phi: np.ndarray,
    grid: RibbonGrid,
    *,
    spacing_um: float = 50.0,
    step_um: float | None = None,
    max_steps: int = 4000,
    resample_points: int = 101,
    terminate_depth: float = 0.995,
    side_boundary_distance_um: float = 25.0,
) -> list[Streamline]:
    """Trace streamlines from the pial boundary toward the white-matter boundary."""
    if spacing_um <= 0:
        raise ValueError("spacing_um must be positive.")
    if resample_points < 2:
        raise ValueError("resample_points must be at least 2.")
    step_um = float(
        step_um if step_um is not None else max(grid.spec.resolution_um / 2.0, 1.0)
    )
    if step_um <= 0:
        raise ValueError("step_um must be positive.")

    gradient = compute_normalized_gradient(phi, grid)
    gx = _field_interpolator(gradient[..., 0], grid)
    gy = _field_interpolator(gradient[..., 1], grid)
    valid_gradient = np.isfinite(gradient).all(axis=2) & grid.mask
    valid_rows, valid_cols = np.nonzero(valid_gradient)
    valid_points = grid.spec.indices_to_points(valid_rows, valid_cols)
    valid_tree = cKDTree(valid_points) if valid_points.size else None
    side_distance_um = (
        ndimage.distance_transform_edt(~grid.side_boundary) * grid.spec.resolution_um
        if grid.side_boundary.any()
        else np.full(grid.shape, np.inf, dtype=float)
    )
    step_units = step_um / grid.spec.coordinate_unit_um
    seeds = _pial_seed_points(grid, spacing_um=spacing_um)
    streamlines: list[Streamline] = []
    for streamline_id, (position_um, seed) in enumerate(seeds):
        path, reached_wm, flag = _integrate_one_streamline(
            seed,
            phi,
            grid,
            gx,
            gy,
            step_units=step_units,
            max_steps=int(max_steps),
            terminate_depth=float(terminate_depth),
            valid_tree=valid_tree,
            valid_points=valid_points,
            snap_distance_units=max(3.0 * grid.spec.step, 3.0 * step_units),
        )
        if path.shape[0] >= 2:
            resampled = resample_path(path, int(resample_points))
            thickness_um = path_length_um(path, grid.spec.coordinate_unit_um)
            near_side = _path_near_side(
                resampled,
                grid,
                side_distance_um,
                side_boundary_distance_um,
            )
        else:
            resampled = np.repeat(path[:1], int(resample_points), axis=0)
            thickness_um = 0.0
            near_side = True
            flag = "failed_short_path"
        if near_side and flag == "ok":
            flag = "near_side_boundary"
        streamlines.append(
            Streamline(
                streamline_id=streamline_id,
                points=resampled.astype(np.float32, copy=False),
                tangential_position_um=float(position_um),
                thickness_um=float(thickness_um),
                reached_wm=bool(reached_wm),
                near_side_boundary=bool(near_side),
                qc_flag=flag,
            )
        )
    return streamlines


def resample_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a polyline to a fixed number of equally spaced points."""
    arr = np.asarray(path, dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("Cannot resample an empty path.")
    if arr.shape[0] == 1:
        return np.repeat(arr, int(n_points), axis=0)
    segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total <= 0:
        return np.repeat(arr[:1], int(n_points), axis=0)
    target = np.linspace(0.0, total, int(n_points))
    x = np.interp(target, cumulative, arr[:, 0])
    y = np.interp(target, cumulative, arr[:, 1])
    return np.column_stack([x, y])


def path_length_um(path: np.ndarray, coordinate_unit_um: float) -> float:
    """Return polyline length in microns."""
    arr = np.asarray(path, dtype=float)
    if arr.shape[0] < 2:
        return 0.0
    return float(
        np.linalg.norm(np.diff(arr, axis=0), axis=1).sum() * coordinate_unit_um
    )


def streamlines_to_dataframe(streamlines: list[Streamline]) -> pd.DataFrame:
    """Flatten streamlines to a point-level dataframe."""
    rows: list[dict[str, Any]] = []
    for streamline in streamlines:
        n_points = int(streamline.points.shape[0])
        if n_points == 0:
            continue
        for point_index, (x_coord, y_coord) in enumerate(streamline.points):
            rows.append(
                {
                    "streamline_id": int(streamline.streamline_id),
                    "point_index": int(point_index),
                    "x": float(x_coord),
                    "y": float(y_coord),
                    "fraction": float(point_index / max(n_points - 1, 1)),
                    "tangential_position_um": float(streamline.tangential_position_um),
                    "streamline_thickness_um": float(streamline.thickness_um),
                    "reached_wm": bool(streamline.reached_wm),
                    "near_side_boundary": bool(streamline.near_side_boundary),
                    "qc_flag": streamline.qc_flag,
                }
            )
    return pd.DataFrame(rows)


def streamlines_to_geojson(streamlines: list[Streamline]) -> dict[str, Any]:
    """Serialize streamlines as a GeoJSON FeatureCollection."""
    features: list[dict[str, Any]] = []
    for streamline in streamlines:
        coordinates = [
            [float(x_coord), float(y_coord)] for x_coord, y_coord in streamline.points
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "streamline_id": int(streamline.streamline_id),
                    "tangential_position_um": float(streamline.tangential_position_um),
                    "streamline_thickness_um": float(streamline.thickness_um),
                    "reached_wm": bool(streamline.reached_wm),
                    "near_side_boundary": bool(streamline.near_side_boundary),
                    "qc_flag": streamline.qc_flag,
                },
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _field_interpolator(
    values: np.ndarray,
    grid: RibbonGrid,
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (grid.spec.y_centers, grid.spec.x_centers),
        np.asarray(values, dtype=float),
        bounds_error=False,
        fill_value=np.nan,
        method="linear",
    )


def _pial_seed_points(
    grid: RibbonGrid,
    *,
    spacing_um: float,
) -> list[tuple[float, np.ndarray]]:
    spacing_units = float(spacing_um) / grid.spec.coordinate_unit_um
    length = float(grid.pial_line.length)
    n_segments = max(1, int(np.ceil(length / spacing_units)))
    distances = np.linspace(0.0, length, n_segments + 1)
    seeds: list[tuple[float, np.ndarray]] = []
    for distance in distances:
        point = np.asarray(
            grid.pial_line.interpolate(float(distance)).coords[0][:2],
            dtype=float,
        )
        seeds.append((float(distance * grid.spec.coordinate_unit_um), point))
    return seeds


def _integrate_one_streamline(
    seed: np.ndarray,
    phi: np.ndarray,
    grid: RibbonGrid,
    gx: RegularGridInterpolator,
    gy: RegularGridInterpolator,
    *,
    step_units: float,
    max_steps: int,
    terminate_depth: float,
    valid_tree: cKDTree | None,
    valid_points: np.ndarray,
    snap_distance_units: float,
) -> tuple[np.ndarray, bool, str]:
    start = np.asarray(seed, dtype=float)
    points = [start]
    snapped = _snap_to_valid_gradient(
        start,
        valid_tree=valid_tree,
        valid_points=valid_points,
        max_distance_units=snap_distance_units,
    )
    if snapped is not None and np.linalg.norm(snapped - start) > 1e-9:
        points.append(snapped)
    reached_wm = False
    flag = "ok"
    for _step in range(max_steps):
        current = points[-1]
        query = np.asarray([[current[1], current[0]]], dtype=float)
        vector = np.asarray([gx(query)[0], gy(query)[0]], dtype=float)
        if not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            flag = "zero_or_nan_gradient"
            break
        vector = vector / np.linalg.norm(vector)
        new_point = current + float(step_units) * vector
        points.append(new_point)
        if _point_in_boundary(new_point, grid, grid.wm_boundary):
            reached_wm = True
            break
        depth = interpolate_scalar_field(phi, grid.spec, new_point.reshape(1, 2))[0]
        if np.isfinite(depth) and depth >= terminate_depth:
            reached_wm = True
            break
        if not points_inside_mask(new_point.reshape(1, 2), grid)[0]:
            flag = "left_ribbon"
            break
    else:
        flag = "max_steps"
    if reached_wm:
        flag = "ok"
    return np.asarray(points, dtype=float), reached_wm, flag


def _snap_to_valid_gradient(
    point: np.ndarray,
    *,
    valid_tree: cKDTree | None,
    valid_points: np.ndarray,
    max_distance_units: float,
) -> np.ndarray | None:
    if valid_tree is None or valid_points.size == 0:
        return None
    distance, index = valid_tree.query(np.asarray(point, dtype=float), k=1)
    if not np.isfinite(distance) or float(distance) > float(max_distance_units):
        return None
    return np.asarray(valid_points[int(index)], dtype=float)


def _path_near_side(
    path: np.ndarray,
    grid: RibbonGrid,
    side_distance_um: np.ndarray,
    threshold_um: float,
) -> bool:
    rows, cols = grid.spec.points_to_indices(np.asarray(path, dtype=float))
    keep = (
        (rows >= 0) & (rows < grid.spec.height) & (cols >= 0) & (cols < grid.spec.width)
    )
    if not keep.any():
        return True
    return bool(
        np.nanmin(side_distance_um[rows[keep], cols[keep]]) <= float(threshold_um)
    )


def _point_in_boundary(
    point: np.ndarray,
    grid: RibbonGrid,
    boundary: np.ndarray,
) -> bool:
    rows, cols = grid.spec.points_to_indices(
        np.asarray(point, dtype=float).reshape(1, 2)
    )
    row = int(rows[0])
    col = int(cols[0])
    if row < 0 or row >= grid.spec.height or col < 0 or col >= grid.spec.width:
        return False
    return bool(boundary[row, col])
