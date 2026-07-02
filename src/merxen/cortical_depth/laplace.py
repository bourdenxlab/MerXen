"""Sparse 2D Laplace solver for cortical-depth fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import linalg as sparse_linalg

from merxen.cortical_depth.ribbon import RasterSpec, RibbonGrid


@dataclass(frozen=True)
class LaplaceSolution:
    """Laplace scalar field and boundary masks."""

    phi: np.ndarray
    grid: RibbonGrid
    converged: bool
    residual: float


def solve_laplace_depth(grid: RibbonGrid) -> LaplaceSolution:
    """Solve ``del^2 phi = 0`` in the ribbon with pia=0 and WM=1."""
    mask = np.asarray(grid.mask, dtype=bool)
    pial = np.asarray(grid.pial_boundary, dtype=bool)
    wm = np.asarray(grid.wm_boundary, dtype=bool)
    unknown = mask & ~(pial | wm)
    labels = np.full(mask.shape, -1, dtype=np.int64)
    labels[unknown] = np.arange(int(np.count_nonzero(unknown)), dtype=np.int64)
    n_unknown = int(labels.max() + 1)

    phi = np.full(mask.shape, np.nan, dtype=np.float64)
    phi[pial] = 0.0
    phi[wm] = 1.0
    if n_unknown == 0:
        return LaplaceSolution(phi=phi, grid=grid, converged=True, residual=0.0)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n_unknown, dtype=np.float64)
    height, width = mask.shape
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row, col in np.argwhere(unknown):
        equation = int(labels[row, col])
        degree = 0
        for drow, dcol in neighbor_offsets:
            nrow = int(row + drow)
            ncol = int(col + dcol)
            if nrow < 0 or nrow >= height or ncol < 0 or ncol >= width:
                continue
            if not mask[nrow, ncol]:
                # Missing neighbours represent zero-normal-gradient boundaries.
                continue
            degree += 1
            neighbor_label = int(labels[nrow, ncol])
            if neighbor_label >= 0:
                rows.append(equation)
                cols.append(neighbor_label)
                data.append(-1.0)
            else:
                boundary_value = phi[nrow, ncol]
                if np.isfinite(boundary_value):
                    rhs[equation] += float(boundary_value)

        if degree <= 0:
            rows.append(equation)
            cols.append(equation)
            data.append(1.0)
            rhs[equation] = 0.5
        else:
            rows.append(equation)
            cols.append(equation)
            data.append(float(degree))

    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_unknown, n_unknown),
    ).tocsr()
    values = sparse_linalg.spsolve(matrix, rhs)
    phi[unknown] = values
    phi[mask] = np.clip(phi[mask], 0.0, 1.0)

    residual = matrix @ values - rhs
    residual_norm = float(np.linalg.norm(residual) / max(np.sqrt(n_unknown), 1.0))
    return LaplaceSolution(
        phi=phi.astype(np.float32, copy=False),
        grid=grid,
        converged=bool(np.isfinite(values).all()),
        residual=residual_norm,
    )


def interpolate_scalar_field(
    field: np.ndarray,
    spec: RasterSpec,
    points: np.ndarray,
    *,
    fill_value: float = np.nan,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a raster scalar field at source-coordinate points."""
    arr = np.asarray(points, dtype=float)
    interpolator = RegularGridInterpolator(
        (spec.y_centers, spec.x_centers),
        np.asarray(field, dtype=float),
        bounds_error=False,
        fill_value=fill_value,
        method=method,
    )
    return np.asarray(interpolator(arr[:, [1, 0]]), dtype=float)


def finite_mask_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return finite values from a raster restricted to a boolean mask."""
    arr = np.asarray(values, dtype=float)
    selected = arr[np.asarray(mask, dtype=bool)]
    return np.asarray(selected[np.isfinite(selected)], dtype=float)
