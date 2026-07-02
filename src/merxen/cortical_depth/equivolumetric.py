"""2D equivolumetric cortical depth via the Waehnert/Bok closed form.

Equidistant (Laplace) depth spaces layers by geometric distance; equivolumetric
depth instead preserves layer *volume* (area in 2D) as the cortex folds. Bok's
principle models an intermediate surface's area as varying linearly with the
equidistant coordinate between the pial and white-matter surface areas of the
local column, which yields the closed form used here (Waehnert et al., 2014).

For a point with equidistant coordinate ``phi`` (0 at pia, 1 at WM) whose column
has pial-side surface area ``A_pia`` and WM-side area ``A_wm``, let
``r = A_wm / A_pia``. The volume fraction from the pia to that point is::

    equivolumetric_depth = (2 * phi + (r - 1) * phi**2) / (1 + r)

This is monotone in ``phi`` for ``r > 0`` and reduces to ``phi`` when ``r == 1``
(uniform thickness). The surface areas are estimated per streamline column from
the tangential spacing of streamline seeds (``A_pia``) and the spacing of their
white-matter endpoints (``A_wm``); by flux conservation of the Laplace field
these are the true 2D "surface areas" of each column. The ratio field ``r`` is
smoothed and interpolated continuously across the ribbon, so the resulting depth
field is smooth rather than piecewise-constant per column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from merxen.cortical_depth.ribbon import RibbonGrid
from merxen.cortical_depth.streamlines import Streamline, select_valid_streamlines

_RATIO_CLIP = (0.1, 10.0)
_RATIO_SMOOTH_WINDOW = 5


@dataclass(frozen=True)
class EquivolumetricResult:
    """Raster equivolumetric depth field and column assignments."""

    depth: np.ndarray
    column_ids: np.ndarray
    column_summary: pd.DataFrame


def compute_equal_area_depth(
    laplace_depth: np.ndarray,
    grid: RibbonGrid,
    streamlines: list[Streamline],
) -> EquivolumetricResult:
    """Compute a 2D equivolumetric depth field via the Waehnert closed form.

    Each pixel's ``equivolumetric_depth`` is derived from its Laplace value and
    the local column area ratio ``r = A_wm / A_pia`` estimated from the valid
    streamlines. Unlike a per-column area ranking, the ratio is interpolated as a
    continuous field, so the output is smooth across column boundaries.
    """
    phi = np.asarray(laplace_depth, dtype=float)
    mask = np.asarray(grid.mask, dtype=bool)
    depth = np.full(phi.shape, np.nan, dtype=np.float32)
    column_ids = np.full(phi.shape, -1, dtype=np.int32)
    rows, cols = np.nonzero(mask & np.isfinite(phi))
    if rows.size == 0:
        return EquivolumetricResult(
            depth=depth,
            column_ids=column_ids,
            column_summary=pd.DataFrame(),
        )

    centers = grid.spec.indices_to_points(rows, cols)
    phi_values = np.clip(phi[rows, cols], 0.0, 1.0)

    valid = select_valid_streamlines(streamlines)
    if len(valid) < 2:
        # Without a spanning column set we cannot estimate area ratios; fall back
        # to the equidistant coordinate (r == 1), which is the r -> 1 limit.
        depth[rows, cols] = phi_values.astype(np.float32)
        return EquivolumetricResult(
            depth=depth,
            column_ids=column_ids,
            column_summary=pd.DataFrame(),
        )

    ratios = _streamline_area_ratios(valid)
    tree_points, point_ids, point_ratios = _streamline_point_arrays(valid, ratios)

    ratio_field = _interpolate_ratio_field(tree_points, point_ratios, centers)
    ratio_field = np.clip(ratio_field, *_RATIO_CLIP)

    eq_values = (2.0 * phi_values + (ratio_field - 1.0) * phi_values**2) / (
        1.0 + ratio_field
    )
    eq_values = np.clip(eq_values, 0.0, 1.0)
    depth[rows, cols] = eq_values.astype(np.float32)

    nearest = cKDTree(tree_points).query(centers, k=1)[1]
    assigned_ids = point_ids[nearest].astype(np.int32, copy=False)
    column_ids[rows, cols] = assigned_ids

    summary = _column_summary(
        assigned_ids=assigned_ids,
        phi_values=phi_values,
        ratios=ratios,
        resolution_um=grid.spec.resolution_um,
    )
    return EquivolumetricResult(
        depth=depth,
        column_ids=column_ids,
        column_summary=summary,
    )


def _streamline_area_ratios(streamlines: list[Streamline]) -> dict[int, float]:
    """Estimate ``A_wm / A_pia`` per streamline column, ordered along the pia."""
    order = np.argsort([line.tangential_position_um for line in streamlines])
    ordered = [streamlines[int(i)] for i in order]
    pia_positions = np.asarray(
        [line.tangential_position_um for line in ordered], dtype=float
    )
    wm_points = np.asarray([np.asarray(line.points)[-1] for line in ordered], float)

    pia_area = _central_spacing(pia_positions)
    wm_arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(wm_points, axis=0), axis=1))]
    )
    wm_area = _central_spacing(wm_arc)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(pia_area > 0, wm_area / pia_area, 1.0)
    ratios[~np.isfinite(ratios)] = 1.0
    ratios = _smooth(ratios, _RATIO_SMOOTH_WINDOW)
    ratios = np.clip(ratios, *_RATIO_CLIP)
    return {
        int(ordered[i].streamline_id): float(ratios[i]) for i in range(len(ordered))
    }


def _streamline_point_arrays(
    streamlines: list[Streamline],
    ratios: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for line in streamlines:
        pts = np.asarray(line.points, dtype=float)
        if pts.shape[0] == 0:
            continue
        points.append(pts)
        ids.append(np.full(pts.shape[0], int(line.streamline_id), dtype=np.int32))
        values.append(np.full(pts.shape[0], ratios[int(line.streamline_id)]))
    return np.vstack(points), np.concatenate(ids), np.concatenate(values)


def _interpolate_ratio_field(
    points: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Continuously interpolate column ratios onto query points.

    Linear interpolation gives a smooth field inside the streamline hull; pixels
    outside the hull (e.g. beyond the outermost streamline) fall back to nearest.
    """
    linear = griddata(points, values, query, method="linear")
    nearest = griddata(points, values, query, method="nearest")
    return np.where(np.isfinite(linear), linear, nearest)


def _central_spacing(values: np.ndarray) -> np.ndarray:
    """Local sample spacing of a sorted 1D coordinate (one-sided at the ends)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 1:
        return np.array([1.0], dtype=float)
    spacing = np.empty(n, dtype=float)
    spacing[1:-1] = (arr[2:] - arr[:-2]) / 2.0
    spacing[0] = arr[1] - arr[0]
    spacing[-1] = arr[-1] - arr[-2]
    return np.asarray(np.abs(spacing), dtype=float)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= window:
        return np.asarray(values, dtype=float)
    return np.asarray(
        ndimage.uniform_filter1d(
            np.asarray(values, dtype=float),
            size=int(window),
            mode="nearest",
        ),
        dtype=float,
    )


def _column_summary(
    *,
    assigned_ids: np.ndarray,
    phi_values: np.ndarray,
    ratios: dict[int, float],
    resolution_um: float,
) -> pd.DataFrame:
    summaries: list[dict[str, float | int]] = []
    for column_id in np.unique(assigned_ids):
        column_mask = assigned_ids == int(column_id)
        values = phi_values[column_mask]
        if values.size == 0:
            continue
        summaries.append(
            {
                "column_id": int(column_id),
                "area_ratio_wm_over_pia": float(ratios.get(int(column_id), 1.0)),
                "area_um2": float(values.size * resolution_um**2),
                "n_pixels": int(values.size),
                "min_laplace_depth": float(np.nanmin(values)),
                "max_laplace_depth": float(np.nanmax(values)),
            }
        )
    return pd.DataFrame(summaries)
