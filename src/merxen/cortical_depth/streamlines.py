"""Gradient-field streamline tracing for cortical-depth coordinates."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from merxen.cortical_depth.ribbon import RibbonGrid, points_inside_mask

logger = logging.getLogger(__name__)

# Below this many seeds, fork/IPC overhead outweighs any parallel speedup.
_PARALLEL_STREAMLINE_THRESHOLD = 16


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


@dataclass(frozen=True)
class _TracingContext:
    """Read-only shared state for tracing streamlines.

    Streamlines are independent, so each is integrated with an identical copy of
    this context. On ``fork`` platforms the context is inherited copy-on-write by
    worker processes, so the large interpolators and trees are never pickled.
    """

    grid: RibbonGrid
    gx: RegularGridInterpolator
    gy: RegularGridInterpolator
    phi_interp: RegularGridInterpolator
    seeds: list[tuple[float, np.ndarray]]
    step_units: float
    max_steps: int
    terminate_depth: float
    valid_tree: cKDTree | None
    valid_points: np.ndarray
    snap_distance_units: float
    side_distance_um: np.ndarray
    side_boundary_distance_um: float
    resample_points: int


# Set in the parent immediately before forking a worker pool; worker processes
# inherit it copy-on-write. ``None`` in the serial path.
_TRACING_CONTEXT: _TracingContext | None = None


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
    # Fill degenerate (zero-norm) nodes with the nearest valid unit vector.
    # These are almost entirely the flat, nearest-value-filled band just outside
    # the ribbon: leaving them NaN makes the linear gradient interpolator return
    # NaN for any streamline whose stencil straddles the pial boundary, which
    # kills a large fraction of seeds at their first step. Filling instead lets
    # streamlines step inward off the pia and terminate on the ribbon/WM checks.
    invalid = ~keep
    if invalid.any() and keep.any():
        nearest = ndimage.distance_transform_edt(
            invalid,
            return_distances=False,
            return_indices=True,
        )
        gradient = gradient[nearest[0], nearest[1]]
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
    n_jobs: int | None = None,
) -> list[Streamline]:
    """Trace streamlines from the pial boundary toward the white-matter boundary.

    Streamlines are mutually independent, so integration is parallelised across
    CPU cores when the seed count justifies the pool overhead. ``n_jobs`` sets the
    worker count; ``None`` resolves it from the ``MERXEN_CORTICAL_DEPTH_WORKERS``
    or ``OMP_NUM_THREADS`` environment variables, then the CPU count. The result
    is identical to serial tracing: seeds keep their input order and each
    streamline is integrated with the same arithmetic.
    """
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
    context = _TracingContext(
        grid=grid,
        gx=gx,
        gy=gy,
        # Building the depth interpolator once here (instead of rebuilding it on
        # every integration step, as the old per-step call did) is the single
        # biggest per-streamline speedup and does not change the interpolated
        # values.
        phi_interp=_field_interpolator(phi, grid),
        seeds=seeds,
        step_units=float(step_units),
        max_steps=int(max_steps),
        terminate_depth=float(terminate_depth),
        valid_tree=valid_tree,
        valid_points=valid_points,
        snap_distance_units=max(3.0 * grid.spec.step, 3.0 * step_units),
        side_distance_um=side_distance_um,
        side_boundary_distance_um=float(side_boundary_distance_um),
        resample_points=int(resample_points),
    )
    return _trace_all_streamlines(context, n_jobs=n_jobs)


def _trace_all_streamlines(
    context: _TracingContext,
    *,
    n_jobs: int | None,
) -> list[Streamline]:
    """Integrate every seed, in parallel when the seed count justifies it."""
    n_seeds = len(context.seeds)
    if n_seeds == 0:
        return []
    workers = _resolve_n_jobs(n_jobs)
    if workers > 1 and n_seeds >= _PARALLEL_STREAMLINE_THRESHOLD:
        parallel = _trace_streamlines_parallel(context, workers=workers)
        if parallel is not None:
            return parallel
    return [_trace_one_streamline(index, context) for index in range(n_seeds)]


def _trace_streamlines_parallel(
    context: _TracingContext,
    *,
    workers: int,
) -> list[Streamline] | None:
    """Trace streamlines across a ``fork`` worker pool, or ``None`` on failure.

    ``imap`` preserves seed order, so the returned list matches serial tracing.
    """
    global _TRACING_CONTEXT
    n_seeds = len(context.seeds)
    chunksize = max(1, n_seeds // (workers * 4))
    _TRACING_CONTEXT = context
    try:
        pool_context = mp.get_context("fork")
        with pool_context.Pool(processes=workers) as pool:
            return list(
                pool.imap(
                    _trace_one_streamline_pooled,
                    range(n_seeds),
                    chunksize=chunksize,
                )
            )
    except Exception:
        logger.warning(
            "Parallel streamline tracing failed; falling back to serial.",
            exc_info=True,
        )
        return None
    finally:
        _TRACING_CONTEXT = None


def _trace_one_streamline_pooled(index: int) -> Streamline:
    """Worker entry point; reads the fork-inherited tracing context."""
    context = _TRACING_CONTEXT
    if context is None:  # pragma: no cover - defensive
        raise RuntimeError("Streamline tracing context is not initialized.")
    return _trace_one_streamline(index, context)


def _trace_one_streamline(index: int, context: _TracingContext) -> Streamline:
    """Integrate and post-process a single seed into a ``Streamline``."""
    position_um, seed = context.seeds[index]
    path, reached_wm, flag = _integrate_one_streamline(
        seed,
        context.grid,
        context.gx,
        context.gy,
        context.phi_interp,
        step_units=context.step_units,
        max_steps=context.max_steps,
        terminate_depth=context.terminate_depth,
        valid_tree=context.valid_tree,
        valid_points=context.valid_points,
        snap_distance_units=context.snap_distance_units,
    )
    if path.shape[0] >= 2:
        resampled = resample_path(path, context.resample_points)
        thickness_um = path_length_um(path, context.grid.spec.coordinate_unit_um)
        near_side = _path_near_side(
            resampled,
            context.grid,
            context.side_distance_um,
            context.side_boundary_distance_um,
        )
    else:
        resampled = np.repeat(path[:1], context.resample_points, axis=0)
        thickness_um = 0.0
        near_side = True
        flag = "failed_short_path"
    if near_side and flag == "ok":
        flag = "near_side_boundary"
    return Streamline(
        streamline_id=index,
        points=resampled.astype(np.float32, copy=False),
        tangential_position_um=float(position_um),
        thickness_um=float(thickness_um),
        reached_wm=bool(reached_wm),
        near_side_boundary=bool(near_side),
        qc_flag=flag,
    )


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve the worker count from the argument, environment, or CPU count."""
    if n_jobs is not None:
        return max(1, int(n_jobs))
    for variable in ("MERXEN_CORTICAL_DEPTH_WORKERS", "OMP_NUM_THREADS"):
        value = os.environ.get(variable, "").strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    return max(1, os.cpu_count() or 1)


def select_valid_streamlines(streamlines: list[Streamline]) -> list[Streamline]:
    """Return streamlines that trace a real pia-to-white-matter path.

    Failed seeds (``zero_or_nan_gradient``/``left_ribbon``) are still resampled to
    the fixed point count and would otherwise pollute nearest-streamline lookups
    and area-ratio estimates with degenerate stubs clustered on the pia.
    """
    return [
        line
        for line in streamlines
        if line.reached_wm
        and float(line.thickness_um) > 0.0
        and np.asarray(line.points).shape[0] >= 2
    ]


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
    grid: RibbonGrid,
    gx: RegularGridInterpolator,
    gy: RegularGridInterpolator,
    phi_interp: RegularGridInterpolator,
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
        depth = float(
            phi_interp(np.asarray([[new_point[1], new_point[0]]], dtype=float))[0]
        )
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
