"""Similarity fits of landmark pairs (Umeyama), leave-one-out residuals, trimming.

numpy only; the napari viewer mirrors this module and the golden vectors
(``tests/test_data/manual_registration/golden_v1.json``) pin both copies, so
"Preview fit" in the viewer equals the workflow fit when the outlier floor is
the same. Points are ``(x, y)`` level-0 pixels; matrices map moving to
reference (``docs/plans/manual-registration-plan.md``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from merxen.manual_registration.geometry import apply_matrix_xy, similarity_parameters
from merxen.manual_registration.schema import (
    DEFAULT_OUTLIER_FLOOR_UM,
    MAX_OUTLIER_FRACTION,
    MIN_USED_PAIRS,
    OUTLIER_K,
)


@dataclass(frozen=True)
class SimilarityFit:
    """A similarity fit of moving to reference landmarks."""

    matrix: np.ndarray  # 3x3 moving px -> reference px
    scale: float  # sqrt|det L|
    rotation_deg: float  # applied after the mirror, wrapped to (-180, 180]
    reflected: bool
    residuals_px: np.ndarray  # in-sample, reference px, (N,)


@dataclass(frozen=True)
class RobustFitOptions:
    """Options of ``robust_similarity``."""

    floor_um: float = DEFAULT_OUTLIER_FLOOR_UM
    k: float = OUTLIER_K
    max_outlier_fraction: float = MAX_OUTLIER_FRACTION
    min_used_pairs: int = MIN_USED_PAIRS
    trim_outliers: bool = True


@dataclass(frozen=True)
class RobustSimilarityFit:
    """A similarity fit after trimming outlier pairs, with diagnostics."""

    fit: SimilarityFit  # on used pairs
    used: np.ndarray  # bool (N,)
    outlier: np.ndarray  # bool (N,)
    residuals_um: np.ndarray  # every pair under fit.matrix
    loo_um: np.ndarray  # used: LOO within the used set; others: = residuals_um
    chosen_class_median_um: float  # median in-sample residual of used pairs
    other_class_median_um: float  # same for the opposite reflection class
    spread_ratio: float  # of the used moving points
    scale_ratio: float  # fit.scale / (moving_px / reference_px)


_DEFAULT_OPTIONS = RobustFitOptions()


def _as_pairs(
    moving_xy: ArrayLike, reference_xy: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(moving_xy, dtype=np.float64)
    dst = np.asarray(reference_xy, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape:
        raise ValueError(
            "moving_xy and reference_xy must be (N, 2) arrays of the same length"
        )
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("landmark coordinates must be finite")
    return src, dst


def umeyama_similarity(
    moving_xy: ArrayLike, reference_xy: ArrayLike, *, reflection: bool
) -> np.ndarray:
    """Least-squares similarity inside one reflection class (Umeyama 1991).

    Args:
        moving_xy: ``(N, 2)`` moving points (source).
        reference_xy: ``(N, 2)`` reference points (destination).
        reflection: Fit a reflecting (det < 0) similarity instead of a proper one.

    Returns:
        The 3x3 matrix mapping moving to reference.

    Raises:
        ValueError: If there are fewer than 2 pairs, the inputs are malformed or
            the moving points are all identical.
    """
    src, dst = _as_pairs(moving_xy, reference_xy)
    count = len(src)
    if count < 2:
        raise ValueError("a similarity fit needs at least 2 landmark pairs")
    mean_src, mean_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mean_src, dst - mean_dst
    var_src = float((src_c**2).sum() / count)
    if var_src == 0.0:
        raise ValueError("moving landmarks are all identical")
    covariance = dst_c.T @ src_c / count
    u, d, vt = np.linalg.svd(covariance)
    sign = float(np.sign(np.linalg.det(u) * np.linalg.det(vt)))
    if sign == 0.0:
        sign = 1.0
    reflect = np.diag([1.0, -sign if reflection else sign])
    rotation = u @ reflect @ vt
    scale = float(np.trace(np.diag(d) @ reflect) / var_src)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scale * rotation
    matrix[:2, 2] = mean_dst - scale * rotation @ mean_src
    return matrix


def fit_similarity(
    moving_xy: ArrayLike, reference_xy: ArrayLike, *, reflection: bool
) -> SimilarityFit:
    """Fit a similarity and report its in-sample residuals (reference px).

    ``reflected`` equals ``reflection`` except for a degenerate fit whose scale
    is not positive; the value of ``similarity_parameters`` is returned.
    """
    src, dst = _as_pairs(moving_xy, reference_xy)
    matrix = umeyama_similarity(src, dst, reflection=reflection)
    scale, angle, reflected = similarity_parameters(matrix)
    residuals = np.linalg.norm(apply_matrix_xy(matrix, src) - dst, axis=1)
    return SimilarityFit(
        matrix=matrix,
        scale=scale,
        rotation_deg=angle,
        reflected=reflected,
        residuals_px=residuals,
    )


def leave_one_out_residuals(
    moving_xy: ArrayLike, reference_xy: ArrayLike, *, reflection: bool
) -> np.ndarray:
    """Residual (reference px) of each pair under a fit that excludes it.

    Fewer than 3 pairs give all NaN; a degenerate sub-fit gives NaN for that pair.
    """
    src, dst = _as_pairs(moving_xy, reference_xy)
    count = len(src)
    result = np.full(count, np.nan)
    if count < 3:
        return result
    for index in range(count):
        keep = np.arange(count) != index
        try:
            matrix = umeyama_similarity(src[keep], dst[keep], reflection=reflection)
        except ValueError:
            continue
        result[index] = np.linalg.norm(apply_matrix_xy(matrix, src[index]) - dst[index])
    return result


def spread_ratio(points_xy: ArrayLike) -> float:
    """Return ``sqrt(lambda_min / lambda_max)`` of the points' covariance.

    1.0 for an isotropic cloud, 0.0 for collinear points or fewer than 2 points.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    covariance = np.cov(pts.T, bias=True)
    lambda_min, lambda_max = np.linalg.eigvalsh(covariance)
    if lambda_max <= 0.0:
        return 0.0
    return float(math.sqrt(max(float(lambda_min), 0.0) / float(lambda_max)))


def robust_similarity(
    moving_xy: ArrayLike,
    reference_xy: ArrayLike,
    *,
    reflection: bool,
    moving_pixel_size_um: float,
    reference_pixel_size_um: float,
    options: RobustFitOptions = _DEFAULT_OPTIONS,
) -> RobustSimilarityFit:
    """Fit a similarity after iteratively trimming outlier landmark pairs.

    A pair is flagged when its leave-one-out residual (within the pairs still
    used) exceeds both ``floor_um`` and ``k`` times the median of the finite
    residuals. The worst flagged pair is dropped, one at a time, until none is
    flagged, ``max_outlier_fraction * N`` pairs were dropped or dropping one more
    would leave fewer than ``min_used_pairs``. Deterministic (ties: lowest index).

    Args:
        moving_xy: ``(N, 2)`` moving landmark points, level-0 px.
        reference_xy: ``(N, 2)`` reference landmark points, level-0 px.
        reflection: The reflection class to fit (the coarse ``mirror_x``).
        moving_pixel_size_um: Moving level-0 pixel size.
        reference_pixel_size_um: Reference level-0 pixel size.
        options: Trimming options.

    Returns:
        The fit on the used pairs with per-pair diagnostics in micrometres.

    Raises:
        ValueError: On malformed input, fewer than 2 pairs or non-positive pixel sizes.
    """
    src, dst = _as_pairs(moving_xy, reference_xy)
    count = len(src)
    if count < 2:
        raise ValueError("a similarity fit needs at least 2 landmark pairs")
    pixel_ref, pixel_mov = reference_pixel_size_um, moving_pixel_size_um
    if not (pixel_ref > 0.0 and pixel_mov > 0.0):
        raise ValueError("pixel sizes must be positive")

    used = np.ones(count, dtype=bool)
    outlier: np.ndarray
    if options.trim_outliers:
        trimmed = 0
        max_trim = math.floor(options.max_outlier_fraction * count)
        while int(used.sum()) >= 3:
            idx = np.flatnonzero(used)
            loo = (
                leave_one_out_residuals(src[idx], dst[idx], reflection=reflection)
                * pixel_ref
            )
            finite = np.isfinite(loo)
            if not finite.any():
                break
            median = float(np.median(loo[finite]))
            flagged = finite & (loo > options.floor_um) & (loo > options.k * median)
            if (
                not flagged.any()
                or trimmed >= max_trim
                or int(used.sum()) - 1 < options.min_used_pairs
            ):
                break
            worst = idx[int(np.argmax(np.where(flagged, loo, -np.inf)))]
            used[worst] = False
            trimmed += 1
        outlier = ~used
    else:
        loo_all = leave_one_out_residuals(src, dst, reflection=reflection) * pixel_ref
        finite = np.isfinite(loo_all)
        median = float(np.median(loo_all[finite])) if finite.any() else math.nan
        outlier = finite & (loo_all > options.floor_um) & (loo_all > options.k * median)

    fit = fit_similarity(src[used], dst[used], reflection=reflection)
    residuals_um = np.linalg.norm(apply_matrix_xy(fit.matrix, src) - dst, axis=1)
    residuals_um = residuals_um * pixel_ref
    loo_um = residuals_um.copy()
    loo_um[used] = (
        leave_one_out_residuals(src[used], dst[used], reflection=reflection) * pixel_ref
    )
    chosen_median = float(np.median(residuals_um[used]))
    other = fit_similarity(src[used], dst[used], reflection=not reflection)
    other_median = float(np.median(other.residuals_px) * pixel_ref)
    return RobustSimilarityFit(
        fit=fit,
        used=used,
        outlier=outlier,
        residuals_um=residuals_um,
        loo_um=loo_um,
        chosen_class_median_um=chosen_median,
        other_class_median_um=other_median,
        spread_ratio=spread_ratio(src[used]),
        scale_ratio=fit.scale / (pixel_mov / pixel_ref),
    )
