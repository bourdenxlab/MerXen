"""Coordinate transform helpers for cross-section alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


def as_xy_array(coords: Any) -> np.ndarray:
    """Return a finite ``(n, 2)`` float array."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected an (n, 2) coordinate array, got {arr.shape}")
    return arr


def fit_affine_matrix(source_xy: Any, target_xy: Any) -> np.ndarray:
    """Fit a 2D affine matrix mapping source coordinates to target coordinates."""
    src = as_xy_array(source_xy)
    dst = as_xy_array(target_xy)
    if src.shape != dst.shape:
        raise ValueError(f"Coordinate shape mismatch: {src.shape} vs {dst.shape}")
    if len(src) < 3:
        raise ValueError("At least three points are required for affine alignment")

    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[valid]
    dst = dst[valid]
    if len(src) < 3:
        raise ValueError("At least three finite point pairs are required")

    design = np.column_stack([src[:, 0], src[:, 1], np.ones(len(src))])
    params, *_ = np.linalg.lstsq(design, dst, rcond=None)
    matrix = np.array(
        [
            [params[0, 0], params[1, 0], params[2, 0]],
            [params[0, 1], params[1, 1], params[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix


def apply_affine_matrix(coords: Any, matrix: Any) -> np.ndarray:
    """Apply a 3x3 homogeneous affine matrix to xy coordinates."""
    xy = as_xy_array(coords)
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine matrix, got {mat.shape}")
    hom = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    out = hom @ mat.T
    return out[:, :2]


@dataclass
class NonRigidTransform:
    """Affine transform plus a residual RBF displacement field."""

    affine_matrix: np.ndarray
    anchors: np.ndarray
    residuals: np.ndarray
    neighbors: int = 64
    smoothing: float = 0.0
    support_radius: float | None = None

    def transform(self: NonRigidTransform, coords: Any) -> np.ndarray:
        """Map source xy coordinates into aligned coordinates."""
        xy = as_xy_array(coords)
        base = apply_affine_matrix(xy, self.affine_matrix)
        if len(self.anchors) < 3 or not np.any(np.isfinite(self.residuals)):
            return base

        n_neighbors = min(int(self.neighbors), len(self.anchors))
        try:
            rbf = RBFInterpolator(
                self.anchors,
                self.residuals,
                kernel="thin_plate_spline",
                neighbors=n_neighbors if n_neighbors < len(self.anchors) else None,
                smoothing=float(self.smoothing),
            )
            disp = rbf(xy)
        except Exception:  # noqa: BLE001
            rbf = RBFInterpolator(
                self.anchors,
                self.residuals,
                kernel="linear",
                neighbors=n_neighbors if n_neighbors < len(self.anchors) else None,
                smoothing=max(float(self.smoothing), 1e-6),
            )
            disp = rbf(xy)

        out = np.asarray(base + disp, dtype=np.float64)
        if self.support_radius is not None and np.isfinite(self.support_radius):
            tree = cKDTree(self.anchors)
            dist, _ = tree.query(xy, k=1)
            far = dist > float(self.support_radius)
            out[far] = base[far]
        return out

    def extrapolated_fraction(self: NonRigidTransform, coords: Any) -> float:
        """Return the fraction of coordinates outside the residual support radius."""
        if self.support_radius is None or len(self.anchors) == 0:
            return 0.0
        xy = as_xy_array(coords)
        tree = cKDTree(self.anchors)
        dist, _ = tree.query(xy, k=1)
        return float(np.mean(dist > float(self.support_radius)))


def fit_nonrigid_transform(
    source_xy: Any,
    target_xy: Any,
    *,
    affine_matrix: Any | None = None,
    neighbors: int = 64,
    smoothing: float = 0.0,
    max_anchors: int = 20_000,
) -> NonRigidTransform:
    """Fit an affine transform with a residual RBF displacement field."""
    src = as_xy_array(source_xy)
    dst = as_xy_array(target_xy)
    if src.shape != dst.shape:
        raise ValueError(f"Coordinate shape mismatch: {src.shape} vs {dst.shape}")
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src = src[valid]
    dst = dst[valid]
    if len(src) < 3:
        raise ValueError("At least three finite point pairs are required")

    if len(src) > int(max_anchors):
        keep = np.linspace(0, len(src) - 1, int(max_anchors), dtype=np.int64)
        src = src[keep]
        dst = dst[keep]

    affine = (
        fit_affine_matrix(src, dst)
        if affine_matrix is None
        else np.asarray(affine_matrix, dtype=np.float64)
    )
    residuals = dst - apply_affine_matrix(src, affine)
    support_radius = _estimate_support_radius(src)
    return NonRigidTransform(
        affine_matrix=affine,
        anchors=src,
        residuals=residuals,
        neighbors=int(neighbors),
        smoothing=float(smoothing),
        support_radius=support_radius,
    )


def transform_displacement_summary(source_xy: Any, target_xy: Any) -> dict[str, float]:
    """Summarize displacement magnitudes between source and target coordinates."""
    src = as_xy_array(source_xy)
    dst = as_xy_array(target_xy)
    disp = np.linalg.norm(dst - src, axis=1)
    return {
        "min": float(np.nanmin(disp)) if len(disp) else float("nan"),
        "p50": float(np.nanpercentile(disp, 50)) if len(disp) else float("nan"),
        "p95": float(np.nanpercentile(disp, 95)) if len(disp) else float("nan"),
        "max": float(np.nanmax(disp)) if len(disp) else float("nan"),
    }


def _estimate_support_radius(anchors: np.ndarray) -> float | None:
    if len(anchors) < 3:
        return None
    tree = cKDTree(anchors)
    dist, _ = tree.query(anchors, k=min(2, len(anchors)))
    nearest = dist[:, 1] if dist.ndim == 2 and dist.shape[1] > 1 else dist
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if len(nearest) == 0:
        return None
    return float(5.0 * np.nanmedian(nearest))
