"""2D equal-area approximation to equivolumetric cortical depth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from merxen.cortical_depth.ribbon import RibbonGrid
from merxen.cortical_depth.streamlines import Streamline


@dataclass(frozen=True)
class EquivolumetricResult:
    """Raster equal-area depth field and column assignments."""

    depth: np.ndarray
    column_ids: np.ndarray
    column_summary: pd.DataFrame


def compute_equal_area_depth(
    laplace_depth: np.ndarray,
    grid: RibbonGrid,
    streamlines: list[Streamline],
) -> EquivolumetricResult:
    """Compute a 2D equal-area depth field within nearest-streamline columns.

    Pixels are assigned to the nearest sampled streamline. Within each resulting
    strip, ``equivolumetric_depth`` is the cumulative area fraction of pixels
    from pia to the pixel's Laplace value. In 2D this is an equal-area
    approximation to true 3D equivolumetric depth.
    """
    mask = np.asarray(grid.mask, dtype=bool)
    phi = np.asarray(laplace_depth, dtype=float)
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
    if streamlines:
        tree_points, tree_ids = _streamline_tree_points(streamlines)
        tree = cKDTree(tree_points)
        _distance, nearest = tree.query(centers, k=1)
        assigned_ids = tree_ids[nearest].astype(np.int32, copy=False)
    else:
        assigned_ids = np.zeros(rows.size, dtype=np.int32)

    column_ids[rows, cols] = assigned_ids
    pixel_values = phi[rows, cols]
    summaries: list[dict[str, float | int]] = []
    for column_id in np.unique(assigned_ids):
        column_mask = assigned_ids == int(column_id)
        values = pixel_values[column_mask]
        if values.size == 0:
            continue
        sorted_values = np.sort(values)
        ranks = np.searchsorted(sorted_values, values, side="right") - 1
        denom = max(sorted_values.size - 1, 1)
        eq_values = np.clip(ranks.astype(float) / float(denom), 0.0, 1.0)
        depth[rows[column_mask], cols[column_mask]] = eq_values.astype(np.float32)
        summaries.append(
            {
                "column_id": int(column_id),
                "area_um2": float(values.size * grid.spec.resolution_um**2),
                "n_pixels": int(values.size),
                "min_laplace_depth": float(np.nanmin(values)),
                "max_laplace_depth": float(np.nanmax(values)),
            }
        )

    return EquivolumetricResult(
        depth=depth,
        column_ids=column_ids,
        column_summary=pd.DataFrame(summaries),
    )


def _streamline_tree_points(
    streamlines: list[Streamline],
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for streamline in streamlines:
        if streamline.points.size == 0:
            continue
        points.append(np.asarray(streamline.points, dtype=float))
        ids.append(
            np.full(
                streamline.points.shape[0],
                int(streamline.streamline_id),
                dtype=np.int32,
            )
        )
    if not points:
        return np.zeros((1, 2), dtype=float), np.zeros(1, dtype=np.int32)
    return np.vstack(points), np.concatenate(ids)
