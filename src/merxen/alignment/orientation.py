"""Arbitrary-angle DAPI pre-orientation before VALIS refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation
from skimage.transform import resize

from merxen.alignment.transforms import apply_affine_matrix, fit_rigid_matrix
from merxen.config import OrientationSearchConfig


@dataclass(frozen=True)
class OrientationResult:
    """Moving-to-fixed similarity pre-transform and its diagnostics."""

    matrix: np.ndarray
    method: str
    angle_degrees: float
    scale: float
    score: float
    metrics: dict[str, Any]
    moving_inlier_xy: np.ndarray | None = None
    fixed_inlier_xy: np.ndarray | None = None


def estimate_pre_orientation(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    *,
    config: OrientationSearchConfig,
) -> OrientationResult:
    """Estimate unrestricted moving-to-fixed rotation, scale, and translation."""
    fixed, moving, fmask, mmask, full_to_small = _orientation_inputs(
        fixed_image,
        moving_image,
        fixed_mask,
        moving_mask,
        max_dimension=int(config.max_dimension_px),
    )
    feature_candidates: list[OrientationResult] = []
    feature_result = _estimate_with_sift(
        fixed,
        moving,
        fmask,
        mmask,
        config=config,
    )
    if feature_result is not None:
        feature_candidates.append(feature_result)
    if config.allow_reflection:
        reflection = _horizontal_reflection_matrix(fixed.shape)
        reflected_moving = warp_image(
            moving,
            reflection,
            output_shape_rc=fixed.shape,
        )
        reflected_mask = warp_image(
            mmask,
            reflection,
            output_shape_rc=fixed.shape,
            interpolation=cv2.INTER_NEAREST,
        )
        reflected_result = _estimate_with_sift(
            fixed,
            reflected_moving,
            fmask,
            reflected_mask,
            config=config,
        )
        if reflected_result is not None:
            feature_candidates.append(
                _compose_orientation_pretransform(
                    reflected_result,
                    reflection,
                    reflected=True,
                )
            )
    if feature_candidates:
        return _to_full_resolution_result(
            _select_orientation_candidate(feature_candidates, config=config),
            full_to_small,
        )

    angular_result = _fallback_angular_search(
        fixed,
        moving,
        fmask,
        mmask,
        config=config,
    )
    return _to_full_resolution_result(angular_result, full_to_small)


def warp_image(
    image: Any,
    matrix: Any,
    *,
    output_shape_rc: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Warp an image with a forward xy homogeneous matrix."""
    arr = np.asarray(image)
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 warp matrix, got {mat.shape}")
    height, width = output_shape_rc
    return cv2.warpAffine(
        arr,
        mat[:2],
        (int(width), int(height)),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def tissue_dice(fixed_mask: Any, moving_mask: Any) -> float:
    """Return Dice overlap for two binary tissue masks."""
    fixed = np.asarray(fixed_mask) > 0
    moving = np.asarray(moving_mask) > 0
    denom = int(fixed.sum()) + int(moving.sum())
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(fixed, moving).sum() / denom)


def masked_normalized_mutual_information(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
) -> float:
    """Return normalized mutual information inside shared tissue."""
    fixed = np.asarray(fixed_image, dtype=np.float32)
    moving = np.asarray(moving_image, dtype=np.float32)
    overlap = (np.asarray(fixed_mask) > 0) & (np.asarray(moving_mask) > 0)
    if int(overlap.sum()) < 32:
        return float("nan")
    fixed_values = fixed[overlap]
    moving_values = moving[overlap]
    histogram, _, _ = np.histogram2d(fixed_values, moving_values, bins=32)
    total = float(histogram.sum())
    if total <= 0:
        return float("nan")
    joint = histogram / total
    fixed_probability = joint.sum(axis=1)
    moving_probability = joint.sum(axis=0)
    fixed_entropy = _entropy(fixed_probability)
    moving_entropy = _entropy(moving_probability)
    denominator = float(np.sqrt(fixed_entropy * moving_entropy))
    if denominator <= 0:
        return float("nan")
    independent = fixed_probability[:, None] * moving_probability[None, :]
    populated = joint > 0
    mutual_information = float(
        np.sum(joint[populated] * np.log(joint[populated] / independent[populated]))
    )
    return mutual_information / denominator


def masked_density_correlation(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    *,
    sigma_px: float = 8.0,
) -> float:
    """Correlate heavily smoothed DAPI density within shared tissue."""
    fixed = ndi.gaussian_filter(
        np.asarray(fixed_image, dtype=np.float32),
        sigma=float(sigma_px),
    )
    moving = ndi.gaussian_filter(
        np.asarray(moving_image, dtype=np.float32),
        sigma=float(sigma_px),
    )
    overlap = (np.asarray(fixed_mask) > 0) & (np.asarray(moving_mask) > 0)
    if int(overlap.sum()) < 32:
        return float("nan")
    x = fixed[overlap]
    y = moving[overlap]
    if float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _orientation_inputs(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    *,
    max_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fixed = np.asarray(fixed_image, dtype=np.uint8)
    moving = np.asarray(moving_image, dtype=np.uint8)
    fmask = (np.asarray(fixed_mask) > 0).astype(np.uint8) * 255
    mmask = (np.asarray(moving_mask) > 0).astype(np.uint8) * 255
    if (
        fixed.shape != moving.shape
        or fixed.shape != fmask.shape
        or fixed.shape != mmask.shape
    ):
        raise ValueError("Orientation images and masks must share one canvas shape")

    scale = min(1.0, float(max_dimension) / float(max(fixed.shape)))
    if scale < 1.0:
        target = (
            max(1, int(round(fixed.shape[0] * scale))),
            max(1, int(round(fixed.shape[1] * scale))),
        )
        fixed = resize(
            fixed,
            target,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
        moving = resize(
            moving,
            target,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
        fmask = (
            resize(fmask, target, order=0, preserve_range=True, anti_aliasing=False) > 0
        ).astype(np.uint8) * 255
        mmask = (
            resize(mmask, target, order=0, preserve_range=True, anti_aliasing=False) > 0
        ).astype(np.uint8) * 255

    full_to_small = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return fixed, moving, fmask, mmask, full_to_small


def _estimate_with_sift(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    config: OrientationSearchConfig,
) -> OrientationResult | None:
    if not hasattr(cv2, "SIFT_create"):
        return None
    sift = cv2.SIFT_create(nfeatures=int(config.sift_features))
    fixed_kp, fixed_desc = sift.detectAndCompute(fixed, fixed_mask)
    moving_kp, moving_desc = sift.detectAndCompute(moving, moving_mask)
    if fixed_desc is None or moving_desc is None:
        return None
    if len(fixed_desc) < 3 or len(moving_desc) < 3:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(moving_desc, fixed_desc, k=2)
    good = [
        first
        for first, second in pairs
        if first.distance < float(config.ratio_threshold) * second.distance
    ]
    if len(good) < max(3, int(config.minimum_inliers)):
        return None
    moving_xy = np.asarray(
        [moving_kp[item.queryIdx].pt for item in good], dtype=np.float32
    )
    fixed_xy = np.asarray(
        [fixed_kp[item.trainIdx].pt for item in good], dtype=np.float32
    )
    matrix_2x3, inlier_mask = cv2.estimateAffinePartial2D(
        moving_xy,
        fixed_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(config.ransac_threshold_px),
        maxIters=10_000,
        confidence=0.999,
        refineIters=25,
    )
    if matrix_2x3 is None or inlier_mask is None:
        return None
    inliers = inlier_mask.ravel().astype(bool)
    if int(inliers.sum()) < 3:
        return None
    matrix = fit_rigid_matrix(moving_xy[inliers], fixed_xy[inliers])
    for _ in range(2):
        residual = np.linalg.norm(
            apply_affine_matrix(moving_xy, matrix) - fixed_xy,
            axis=1,
        )
        refined_inliers = residual <= float(config.ransac_threshold_px)
        if int(refined_inliers.sum()) < 3:
            return None
        if np.array_equal(refined_inliers, inliers):
            break
        inliers = refined_inliers
        matrix = fit_rigid_matrix(moving_xy[inliers], fixed_xy[inliers])
    n_inliers = int(inliers.sum())
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
    determinant = float(np.linalg.det(matrix[:2, :2]))
    moving_coverage = _point_coverage(moving_xy[inliers], moving_mask)
    fixed_coverage = _point_coverage(fixed_xy[inliers], fixed_mask)
    coverage = min(moving_coverage, fixed_coverage)
    warped_mask = warp_image(
        moving_mask,
        matrix,
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    dice = tissue_dice(fixed_mask, warped_mask)
    if (
        n_inliers < int(config.minimum_inliers)
        or coverage < float(config.minimum_inlier_coverage)
        or scale < float(config.minimum_scale)
        or scale > float(config.maximum_scale)
        or dice < float(config.minimum_dice)
        or (determinant < 0 and not config.allow_reflection)
    ):
        return None

    warped_image = warp_image(moving, matrix, output_shape_rc=fixed.shape)
    nmi = masked_normalized_mutual_information(
        fixed,
        warped_image,
        fixed_mask,
        warped_mask,
    )
    score = _combined_score(
        dice=dice,
        nmi=nmi,
        inliers=n_inliers,
        coverage=coverage,
        config=config,
    )
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])) % 360.0)
    return OrientationResult(
        matrix=matrix,
        method="sift_ransac",
        angle_degrees=angle,
        scale=scale,
        score=score,
        metrics={
            "n_matches": int(len(good)),
            "n_inliers": n_inliers,
            "inlier_coverage": coverage,
            "dice": dice,
            "normalized_mutual_information": nmi,
            "reflection": bool(determinant < 0),
            "scale_locked_to_metadata": True,
        },
        moving_inlier_xy=moving_xy[inliers].astype(np.float64),
        fixed_inlier_xy=fixed_xy[inliers].astype(np.float64),
    )


def _fallback_angular_search(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    config: OrientationSearchConfig,
) -> OrientationResult:
    reflections = [False, True] if config.allow_reflection else [False]
    coarse_angles = np.arange(0.0, 360.0, float(config.coarse_step_degrees))
    candidates = [
        _score_angle(
            float(angle),
            reflected=reflected,
            fixed=fixed,
            moving=moving,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            config=config,
        )
        for reflected in reflections
        for angle in coarse_angles
    ]
    candidates = _best_distinct(candidates, int(config.candidates_to_refine))

    refine_candidates: list[OrientationResult] = []
    refine_radius = float(config.coarse_step_degrees)
    for candidate in candidates:
        reflected = bool(candidate.metrics["reflection"])
        for angle in _angle_neighborhood(
            candidate.angle_degrees,
            radius=refine_radius,
            step=float(config.refine_step_degrees),
        ):
            refine_candidates.append(
                _score_angle(
                    angle,
                    reflected=reflected,
                    fixed=fixed,
                    moving=moving,
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                    config=config,
                )
            )
    refine_candidates = _best_distinct(
        refine_candidates,
        int(config.candidates_to_refine),
    )

    final_candidates: list[OrientationResult] = []
    final_radius = float(config.refine_step_degrees)
    for candidate in refine_candidates:
        reflected = bool(candidate.metrics["reflection"])
        for angle in _angle_neighborhood(
            candidate.angle_degrees,
            radius=final_radius,
            step=float(config.final_step_degrees),
        ):
            final_candidates.append(
                _score_angle(
                    angle,
                    reflected=reflected,
                    fixed=fixed,
                    moving=moving,
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                    config=config,
                )
            )
    if not final_candidates:
        raise RuntimeError("Fallback angular search produced no candidates")
    return _select_orientation_candidate(final_candidates, config=config)


def _select_orientation_candidate(
    candidates: list[OrientationResult],
    *,
    config: OrientationSearchConfig,
) -> OrientationResult:
    """Select handedness conservatively and retain both candidates for QC."""
    best_by_reflection: dict[bool, OrientationResult] = {}
    for candidate in candidates:
        reflected = bool(candidate.metrics.get("reflection", False))
        current = best_by_reflection.get(reflected)
        if current is None or candidate.score > current.score:
            best_by_reflection[reflected] = candidate

    non_reflected = best_by_reflection.get(False)
    reflected = best_by_reflection.get(True)
    if non_reflected is None and reflected is None:
        raise RuntimeError("Orientation search produced no valid candidates")
    if non_reflected is None:
        selected = reflected
        reason = "only_reflected_candidate_valid"
    elif reflected is None:
        selected = non_reflected
        reason = "only_non_reflected_candidate_valid"
    else:
        improvement = float(reflected.score - non_reflected.score)
        if improvement >= float(config.reflection_minimum_score_improvement):
            selected = reflected
            reason = "reflected_candidate_exceeded_score_margin"
        else:
            selected = non_reflected
            reason = "reflection_did_not_exceed_score_margin"
    assert selected is not None

    metrics = dict(selected.metrics)
    metrics["selection_reason"] = reason
    metrics["reflection_score_improvement"] = (
        float("nan")
        if non_reflected is None or reflected is None
        else float(reflected.score - non_reflected.score)
    )
    metrics["candidate_comparison"] = {
        "non_reflected": _orientation_candidate_summary(non_reflected),
        "reflected": _orientation_candidate_summary(reflected),
    }
    return OrientationResult(
        matrix=selected.matrix,
        method=selected.method,
        angle_degrees=selected.angle_degrees,
        scale=selected.scale,
        score=selected.score,
        metrics=metrics,
        moving_inlier_xy=selected.moving_inlier_xy,
        fixed_inlier_xy=selected.fixed_inlier_xy,
    )


def _orientation_candidate_summary(
    candidate: OrientationResult | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "method": candidate.method,
        "angle_degrees": float(candidate.angle_degrees),
        "scale": float(candidate.scale),
        "score": float(candidate.score),
        "dice": float(candidate.metrics.get("dice", np.nan)),
        "normalized_mutual_information": float(
            candidate.metrics.get("normalized_mutual_information", np.nan)
        ),
        "n_inliers": int(candidate.metrics.get("n_inliers", 0)),
        "inlier_coverage": float(candidate.metrics.get("inlier_coverage", 0.0)),
    }


def _horizontal_reflection_matrix(shape_rc: tuple[int, int]) -> np.ndarray:
    """Return a centered left/right reflection on an image canvas."""
    _, width = shape_rc
    return np.array(
        [
            [-1.0, 0.0, float(width - 1)],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _compose_orientation_pretransform(
    result: OrientationResult,
    pretransform: np.ndarray,
    *,
    reflected: bool,
) -> OrientationResult:
    """Express a result estimated on a prewarped image in original pixels."""
    matrix = np.asarray(result.matrix) @ np.asarray(pretransform)
    moving_inliers = (
        None
        if result.moving_inlier_xy is None
        else _apply_affine(
            result.moving_inlier_xy,
            np.linalg.inv(pretransform),
        )
    )
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])) % 360.0)
    metrics = dict(result.metrics)
    metrics["reflection"] = bool(reflected)
    return OrientationResult(
        matrix=matrix,
        method=result.method,
        angle_degrees=angle,
        scale=result.scale,
        score=result.score,
        metrics=metrics,
        moving_inlier_xy=moving_inliers,
        fixed_inlier_xy=result.fixed_inlier_xy,
    )


def _score_angle(
    angle: float,
    *,
    reflected: bool,
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    config: OrientationSearchConfig,
) -> OrientationResult:
    height, width = fixed.shape
    center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    rotation = np.vstack(
        [cv2.getRotationMatrix2D(center, float(angle), 1.0), [0.0, 0.0, 1.0]]
    )
    if reflected:
        reflection = np.array(
            [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        base = rotation @ reflection
    else:
        base = rotation

    rotated_mask = warp_image(
        moving_mask,
        base,
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    fixed_distance = ndi.distance_transform_edt(fixed_mask > 0)
    moving_distance = ndi.distance_transform_edt(rotated_mask > 0)
    shift_rc, _, _ = phase_cross_correlation(
        fixed_distance,
        moving_distance,
        upsample_factor=1,
        normalization=None,
    )
    translation = np.array(
        [
            [1.0, 0.0, float(shift_rc[1])],
            [0.0, 1.0, float(shift_rc[0])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix = translation @ base
    warped_mask = warp_image(
        moving_mask,
        matrix,
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    warped_image = warp_image(moving, matrix, output_shape_rc=fixed.shape)
    dice = tissue_dice(fixed_mask, warped_mask)
    nmi = masked_normalized_mutual_information(
        fixed,
        warped_image,
        fixed_mask,
        warped_mask,
    )
    score = _combined_score(
        dice=dice,
        nmi=nmi,
        inliers=0,
        coverage=0.0,
        config=config,
    )
    return OrientationResult(
        matrix=matrix,
        method="angular_search",
        angle_degrees=float(angle % 360.0),
        scale=1.0,
        score=score,
        metrics={
            "n_matches": 0,
            "n_inliers": 0,
            "inlier_coverage": 0.0,
            "dice": dice,
            "normalized_mutual_information": nmi,
            "reflection": reflected,
            "translation_x_px": float(shift_rc[1]),
            "translation_y_px": float(shift_rc[0]),
        },
    )


def _combined_score(
    *,
    dice: float,
    nmi: float,
    inliers: int,
    coverage: float,
    config: OrientationSearchConfig,
) -> float:
    safe_nmi = float(nmi) if np.isfinite(nmi) else 0.0
    feature_score = np.log1p(max(0, int(inliers))) * max(0.0, float(coverage))
    return float(
        float(config.overlap_weight) * max(0.0, float(dice))
        + float(config.mutual_information_weight) * safe_nmi
        + float(config.feature_weight) * feature_score
    )


def _entropy(probabilities: np.ndarray) -> float:
    populated = np.asarray(probabilities, dtype=np.float64)
    populated = populated[populated > 0]
    if len(populated) == 0:
        return 0.0
    return max(0.0, float(-np.sum(populated * np.log(populated))))


def _point_coverage(points_xy: np.ndarray, tissue_mask: np.ndarray) -> float:
    if len(points_xy) == 0:
        return 0.0
    rows, cols = np.nonzero(tissue_mask > 0)
    if len(rows) == 0:
        return 0.0
    tissue_width = max(float(cols.max() - cols.min()), 1.0)
    tissue_height = max(float(rows.max() - rows.min()), 1.0)
    point_width = max(float(np.ptp(points_xy[:, 0])), 0.0)
    point_height = max(float(np.ptp(points_xy[:, 1])), 0.0)
    return float(
        min(1.0, (point_width * point_height) / (tissue_width * tissue_height))
    )


def _angle_neighborhood(center: float, *, radius: float, step: float) -> list[float]:
    offsets = np.arange(-radius, radius + step * 0.5, step)
    return [float((center + offset) % 360.0) for offset in offsets]


def _best_distinct(
    candidates: list[OrientationResult],
    count: int,
) -> list[OrientationResult]:
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    selected: list[OrientationResult] = []
    seen: set[tuple[float, bool]] = set()
    for item in ordered:
        key = (
            round(float(item.angle_degrees), 6),
            bool(item.metrics.get("reflection", False)),
        )
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= count:
            break
    return selected


def _to_full_resolution_result(
    result: OrientationResult,
    full_to_small: np.ndarray,
) -> OrientationResult:
    small_to_full = np.linalg.inv(full_to_small)
    full_matrix = small_to_full @ result.matrix @ full_to_small
    moving_inliers = (
        None
        if result.moving_inlier_xy is None
        else _apply_affine(result.moving_inlier_xy, small_to_full)
    )
    fixed_inliers = (
        None
        if result.fixed_inlier_xy is None
        else _apply_affine(result.fixed_inlier_xy, small_to_full)
    )
    return OrientationResult(
        matrix=full_matrix,
        method=result.method,
        angle_degrees=result.angle_degrees,
        scale=result.scale,
        score=result.score,
        metrics=dict(result.metrics),
        moving_inlier_xy=moving_inliers,
        fixed_inlier_xy=fixed_inliers,
    )


def _apply_affine(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points_xy, np.ones(len(points_xy))])
    return np.asarray((homogeneous @ matrix.T)[:, :2], dtype=np.float64)
