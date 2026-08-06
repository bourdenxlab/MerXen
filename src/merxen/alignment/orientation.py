"""Arbitrary-angle DAPI pre-orientation before VALIS refinement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
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


@dataclass(frozen=True)
class _LocalFinePeak:
    """One persistent local maximum from coarse and refined score grids."""

    candidate: OrientationResult
    coarse_index: tuple[int, int, int]
    coarse_score: float
    prominence: float
    coarse_interior: bool
    refinement_interior: bool
    refinement_iterations: int

    @property
    def is_stable(self: _LocalFinePeak) -> bool:
        """Return whether the maximum persists away from both grid boundaries."""
        return self.coarse_interior and self.refinement_interior


def estimate_pre_orientation(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    *,
    config: OrientationSearchConfig,
    pixel_size_um: float = 1.0,
    output_dir: Path | None = None,
) -> OrientationResult:
    """Estimate unrestricted moving-to-fixed rotation, scale, and translation."""
    fixed, moving, fmask, mmask, full_to_small = _orientation_inputs(
        fixed_image,
        moving_image,
        fixed_mask,
        moving_mask,
        max_dimension=int(config.max_dimension_px),
    )
    reflections = _requested_reflections(config)
    has_manual_seed = bool(
        config.initial_angle_degrees is not None
        or config.initial_translation_x_um is not None
    )
    feature_candidates: list[OrientationResult] = []
    if not has_manual_seed and False in reflections:
        feature_result = _estimate_with_sift(
            fixed,
            moving,
            fmask,
            mmask,
            config=config,
        )
        if feature_result is not None:
            feature_candidates.append(feature_result)
    if not has_manual_seed and True in reflections:
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
        selected = _select_orientation_candidate(feature_candidates, config=config)
        selected = _final_local_orientation_search(
            selected,
            fixed=fixed,
            moving=moving,
            fixed_mask=fmask,
            moving_mask=mmask,
            config=config,
            reduction_scale=float(full_to_small[0, 0]),
            pixel_size_um=float(pixel_size_um),
            output_dir=output_dir,
        )
        if output_dir is not None:
            _write_orientation_search_qc(
                Path(output_dir),
                fixed=fixed,
                moving=moving,
                candidates=feature_candidates,
                selected=selected,
            )
        return _to_full_resolution_result(
            selected,
            full_to_small,
        )

    angular_result = _fallback_angular_search(
        fixed,
        moving,
        fmask,
        mmask,
        config=config,
        reduction_scale=float(full_to_small[0, 0]),
        pixel_size_um=float(pixel_size_um),
        output_dir=output_dir,
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
    reduction_scale: float = 1.0,
    pixel_size_um: float = 1.0,
    output_dir: Path | None = None,
) -> OrientationResult:
    reflections = _requested_reflections(config)
    coarse_angles = list(
        np.arange(0.0, 360.0, float(config.coarse_step_degrees), dtype=float)
    )
    if config.initial_angle_degrees is not None:
        coarse_angles.append(float(config.initial_angle_degrees) % 360.0)
    override_translation = _override_translation_small_px(
        config,
        reduction_scale=reduction_scale,
        pixel_size_um=pixel_size_um,
    )
    coarse_evaluated = [
        candidate
        for reflected in reflections
        for angle in coarse_angles
        for candidate in _score_angle_translation_candidates(
            float(angle),
            reflected=reflected,
            fixed=fixed,
            moving=moving,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            config=config,
            translation_center_xy=None,
            translation_radius_px=float(config.coarse_translation_radius_px),
            override_translation_xy=override_translation,
            search_stage="coarse",
        )
    ]
    coarse_evaluated = _apply_candidate_eligibility(
        coarse_evaluated,
        config=config,
    )
    candidates = _best_distinct_by_handedness(
        coarse_evaluated,
        count=int(config.candidates_to_refine),
        reflections=reflections,
    )

    refine_evaluated: list[OrientationResult] = []
    refine_radius = float(config.coarse_step_degrees)
    for candidate in candidates:
        reflected = bool(candidate.metrics["reflection"])
        for angle in _angle_neighborhood(
            candidate.angle_degrees,
            radius=refine_radius,
            step=float(config.refine_step_degrees),
        ):
            refine_evaluated.extend(
                _score_angle_translation_candidates(
                    angle,
                    reflected=reflected,
                    fixed=fixed,
                    moving=moving,
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                    config=config,
                    translation_center_xy=(
                        float(candidate.metrics["translation_x_px"]),
                        float(candidate.metrics["translation_y_px"]),
                    ),
                    translation_radius_px=float(config.refine_translation_radius_px),
                    override_translation_xy=override_translation,
                    search_stage="refine",
                )
            )
    refine_evaluated = _apply_candidate_eligibility(
        refine_evaluated,
        config=config,
    )
    refine_candidates = _best_distinct_by_handedness(
        refine_evaluated,
        count=int(config.candidates_to_refine),
        reflections=reflections,
    )

    final_evaluated: list[OrientationResult] = []
    final_radius = float(config.refine_step_degrees)
    for candidate in refine_candidates:
        reflected = bool(candidate.metrics["reflection"])
        for angle in _angle_neighborhood(
            candidate.angle_degrees,
            radius=final_radius,
            step=float(config.final_step_degrees),
        ):
            final_evaluated.extend(
                _score_angle_translation_candidates(
                    angle,
                    reflected=reflected,
                    fixed=fixed,
                    moving=moving,
                    fixed_mask=fixed_mask,
                    moving_mask=moving_mask,
                    config=config,
                    translation_center_xy=(
                        float(candidate.metrics["translation_x_px"]),
                        float(candidate.metrics["translation_y_px"]),
                    ),
                    translation_radius_px=float(config.final_translation_radius_px),
                    override_translation_xy=override_translation,
                    search_stage="final",
                )
            )
    if not final_evaluated:
        raise RuntimeError("Fallback angular search produced no candidates")
    final_evaluated = _apply_candidate_eligibility(
        final_evaluated,
        config=config,
    )
    final_candidates = _best_distinct_by_handedness(
        final_evaluated,
        count=int(config.candidates_to_refine),
        reflections=reflections,
    )
    selected = _select_orientation_candidate(final_candidates, config=config)
    search_metrics = {
        "coarse_evaluated": len(coarse_evaluated),
        "refine_evaluated": len(refine_evaluated),
        "final_evaluated": len(final_evaluated),
        "candidates_retained_per_handedness": int(config.candidates_to_refine),
        "translation_candidates_per_angle": int(
            config.translation_candidates_per_angle
        ),
        "reflection_mode": config.reflection_mode,
        "manual_seed": {
            "angle_degrees": config.initial_angle_degrees,
            "translation_x_um": config.initial_translation_x_um,
            "translation_y_um": config.initial_translation_y_um,
        },
        "top_final_candidates": [
            _orientation_candidate_summary(candidate) for candidate in final_candidates
        ],
    }
    selected_metrics = dict(selected.metrics)
    selected_metrics["joint_search"] = search_metrics
    selected = OrientationResult(
        matrix=selected.matrix,
        method="joint_angular_translation_search",
        angle_degrees=selected.angle_degrees,
        scale=selected.scale,
        score=selected.score,
        metrics=selected_metrics,
        moving_inlier_xy=selected.moving_inlier_xy,
        fixed_inlier_xy=selected.fixed_inlier_xy,
    )
    selected = _final_local_orientation_search(
        selected,
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=config,
        reduction_scale=reduction_scale,
        pixel_size_um=pixel_size_um,
        output_dir=output_dir,
    )
    if output_dir is not None:
        _write_orientation_search_qc(
            Path(output_dir),
            fixed=fixed,
            moving=moving,
            candidates=final_candidates,
            selected=selected,
        )
    return selected


def _requested_reflections(config: OrientationSearchConfig) -> list[bool]:
    """Return handedness branches requested by automatic or manual configuration."""
    if config.reflection_mode == "force":
        return [True]
    if config.reflection_mode == "forbid" or not config.allow_reflection:
        return [False]
    return [False, True]


def _override_translation_small_px(
    config: OrientationSearchConfig,
    *,
    reduction_scale: float,
    pixel_size_um: float,
) -> tuple[float, float] | None:
    """Convert an optional full-resolution physical translation to search pixels."""
    if config.initial_translation_x_um is None:
        return None
    if config.initial_translation_y_um is None:
        raise ValueError(
            "initial_translation_x_um and initial_translation_y_um must be set together"
        )
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
    return (
        float(config.initial_translation_x_um) / pixel_size_um * reduction_scale,
        float(config.initial_translation_y_um) / pixel_size_um * reduction_scale,
    )


def _score_angle_translation_candidates(
    angle: float,
    *,
    reflected: bool,
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    config: OrientationSearchConfig,
    translation_center_xy: tuple[float, float] | None,
    translation_radius_px: float,
    override_translation_xy: tuple[float, float] | None,
    search_stage: str,
) -> list[OrientationResult]:
    """Score several translation initial conditions for one angular candidate."""
    base = _orientation_base_matrix(angle, reflected=reflected, shape_rc=fixed.shape)
    rotated_mask = warp_image(
        moving_mask,
        base,
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    seeds = _translation_seeds(
        fixed_mask,
        rotated_mask,
        center_xy=translation_center_xy,
        radius_px=translation_radius_px,
        override_xy=override_translation_xy,
    )
    preliminary: list[tuple[float, tuple[float, float]]] = []
    for translation_xy in seeds:
        matrix = _translation_matrix(*translation_xy) @ base
        warped_mask = warp_image(
            moving_mask,
            matrix,
            output_shape_rc=fixed.shape,
            interpolation=cv2.INTER_NEAREST,
        )
        preliminary.append((tissue_dice(fixed_mask, warped_mask), translation_xy))
    preliminary.sort(key=lambda item: item[0], reverse=True)
    retained = preliminary[: int(config.translation_candidates_per_angle)]
    if override_translation_xy is not None and all(
        not np.allclose(translation_xy, override_translation_xy)
        for _, translation_xy in retained
    ):
        override_score = next(
            score
            for score, translation_xy in preliminary
            if np.allclose(translation_xy, override_translation_xy)
        )
        retained[-1] = (override_score, override_translation_xy)
    return [
        _score_orientation_transform(
            angle,
            translation_xy=translation_xy,
            reflected=reflected,
            fixed=fixed,
            moving=moving,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            config=config,
            search_stage=search_stage,
            translation_seed_rank=rank,
        )
        for rank, (_, translation_xy) in enumerate(retained, start=1)
    ]


def _orientation_base_matrix(
    angle: float,
    *,
    reflected: bool,
    shape_rc: tuple[int, int],
) -> np.ndarray:
    """Return a centered rotation with an optional left/right reflection."""
    height, width = shape_rc
    center = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
    rotation = np.vstack(
        [cv2.getRotationMatrix2D(center, float(angle), 1.0), [0.0, 0.0, 1.0]]
    )
    return rotation @ _horizontal_reflection_matrix(shape_rc) if reflected else rotation


def _translation_seeds(
    fixed_mask: np.ndarray,
    rotated_mask: np.ndarray,
    *,
    center_xy: tuple[float, float] | None,
    radius_px: float,
    override_xy: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    """Generate distinct phase, centroid, manual, and local-grid translation seeds."""
    fixed_distance = ndi.distance_transform_edt(fixed_mask > 0)
    moving_distance = ndi.distance_transform_edt(rotated_mask > 0)
    shift_rc, _, _ = phase_cross_correlation(
        fixed_distance,
        moving_distance,
        upsample_factor=1,
        normalization=None,
    )
    phase_xy = (float(shift_rc[1]), float(shift_rc[0]))
    fixed_center = ndi.center_of_mass(fixed_mask > 0)
    moving_center = ndi.center_of_mass(rotated_mask > 0)
    centroid_xy = (
        float(fixed_center[1] - moving_center[1]),
        float(fixed_center[0] - moving_center[0]),
    )
    centers = [phase_xy, centroid_xy, (0.0, 0.0)]
    if center_xy is not None:
        centers.insert(0, center_xy)
    if override_xy is not None:
        centers.insert(0, override_xy)
    offsets = (-float(radius_px), 0.0, float(radius_px))
    seeds = [
        (center[0] + dx, center[1] + dy)
        for center in centers
        for dx in offsets
        for dy in offsets
    ]
    distinct: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in seeds:
        key = (round(x, 4), round(y, 4))
        if key not in seen:
            seen.add(key)
            distinct.append((x, y))
    return distinct


def _translation_matrix(x: float, y: float) -> np.ndarray:
    """Return a homogeneous xy translation matrix."""
    return np.array(
        [[1.0, 0.0, float(x)], [0.0, 1.0, float(y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _apply_candidate_eligibility(
    candidates: list[OrientationResult],
    *,
    config: OrientationSearchConfig,
) -> list[OrientationResult]:
    """Apply absolute and handedness-relative plausibility gates."""
    maximum_dice = {
        reflected: max(
            (
                float(candidate.metrics.get("dice", 0.0))
                for candidate in candidates
                if bool(candidate.metrics.get("reflection", False)) == reflected
            ),
            default=0.0,
        )
        for reflected in (False, True)
    }
    evaluated: list[OrientationResult] = []
    for candidate in candidates:
        metrics = dict(candidate.metrics)
        reasons: list[str] = []
        reflected = bool(metrics.get("reflection", False))
        dice = float(metrics.get("dice", np.nan))
        if not np.isfinite(dice) or dice < float(config.minimum_dice):
            reasons.append("dice_below_minimum")
        if dice < maximum_dice[reflected] * float(config.minimum_relative_dice):
            reasons.append("dice_below_handedness_relative_minimum")
        if float(metrics.get("fixed_overlap_fraction", 0.0)) < float(
            config.minimum_fixed_overlap_fraction
        ):
            reasons.append("fixed_overlap_below_minimum")
        if float(metrics.get("moving_overlap_fraction", 0.0)) < float(
            config.minimum_moving_overlap_fraction
        ):
            reasons.append("moving_overlap_below_minimum")
        if float(metrics.get("retained_moving_fraction", 0.0)) < float(
            config.minimum_retained_moving_fraction
        ):
            reasons.append("moving_tissue_clipped")
        metrics["eligible"] = not reasons
        metrics["eligibility_reasons"] = reasons
        evaluated.append(
            OrientationResult(
                matrix=candidate.matrix,
                method=candidate.method,
                angle_degrees=candidate.angle_degrees,
                scale=candidate.scale,
                score=candidate.score,
                metrics=metrics,
                moving_inlier_xy=candidate.moving_inlier_xy,
                fixed_inlier_xy=candidate.fixed_inlier_xy,
            )
        )
    return evaluated


def _best_distinct_by_handedness(
    candidates: list[OrientationResult],
    *,
    count: int,
    reflections: list[bool],
) -> list[OrientationResult]:
    """Retain an independent candidate beam for every requested handedness."""
    retained: list[OrientationResult] = []
    for reflected in reflections:
        branch = [
            candidate
            for candidate in candidates
            if bool(candidate.metrics.get("reflection", False)) == reflected
        ]
        eligible = [
            candidate
            for candidate in branch
            if bool(candidate.metrics.get("eligible", True))
        ]
        retained.extend(_best_distinct(eligible or branch, count))
    return retained


def _select_orientation_candidate(
    candidates: list[OrientationResult],
    *,
    config: OrientationSearchConfig,
) -> OrientationResult:
    """Select handedness conservatively and retain both candidates for QC."""
    comparison_by_reflection: dict[bool, OrientationResult] = {}
    for candidate in candidates:
        is_reflected = bool(candidate.metrics.get("reflection", False))
        current = comparison_by_reflection.get(is_reflected)
        if current is None or candidate.score > current.score:
            comparison_by_reflection[is_reflected] = candidate

    eligible_candidates = [
        candidate
        for candidate in candidates
        if bool(candidate.metrics.get("eligible", True))
    ]
    eligibility_fallback = not eligible_candidates
    selection_pool = eligible_candidates or candidates
    best_by_reflection: dict[bool, OrientationResult] = {}
    for candidate in selection_pool:
        is_reflected = bool(candidate.metrics.get("reflection", False))
        current = best_by_reflection.get(is_reflected)
        if current is None or candidate.score > current.score:
            best_by_reflection[is_reflected] = candidate

    non_reflected = best_by_reflection.get(False)
    reflected_candidate = best_by_reflection.get(True)
    if non_reflected is None and reflected_candidate is None:
        raise RuntimeError("Orientation search produced no valid candidates")
    ambiguous = False
    if non_reflected is None:
        selected = reflected_candidate
        reason = "only_reflected_candidate_valid"
    elif reflected_candidate is None:
        selected = non_reflected
        reason = "only_non_reflected_candidate_valid"
    else:
        improvement = float(reflected_candidate.score - non_reflected.score)
        margin = float(config.reflection_minimum_score_improvement)
        if abs(improvement) < margin:
            ambiguous = True
            selected = reflected_candidate if improvement > 0 else non_reflected
            reason = (
                "handedness_ambiguous_reflected_provisional"
                if improvement > 0
                else "handedness_ambiguous_non_reflected_provisional"
            )
        elif improvement > 0:
            selected = reflected_candidate
            reason = "reflected_candidate_exceeded_score_margin"
        else:
            selected = non_reflected
            reason = "non_reflected_candidate_exceeded_score_margin"
    assert selected is not None

    metrics = dict(selected.metrics)
    metrics["selection_reason"] = reason
    metrics["handedness_ambiguous"] = ambiguous
    metrics["eligibility_fallback"] = eligibility_fallback
    metrics["provisional_selection"] = ambiguous or eligibility_fallback
    metrics["reflection_score_improvement"] = (
        float("nan")
        if non_reflected is None or reflected_candidate is None
        else float(reflected_candidate.score - non_reflected.score)
    )
    metrics["candidate_comparison"] = {
        "non_reflected": _orientation_candidate_summary(
            comparison_by_reflection.get(False)
        ),
        "reflected": _orientation_candidate_summary(comparison_by_reflection.get(True)),
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


def _final_local_orientation_search(
    initial: OrientationResult,
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    config: OrientationSearchConfig,
    reduction_scale: float,
    pixel_size_um: float,
    output_dir: Path | None,
) -> OrientationResult:
    """Refine the selected handedness and characterize nearby score maxima."""
    if not config.local_fine_search_enabled:
        metrics = dict(initial.metrics)
        metrics["local_fine_search"] = {"enabled": False}
        return OrientationResult(
            matrix=initial.matrix,
            method=initial.method,
            angle_degrees=initial.angle_degrees,
            scale=initial.scale,
            score=initial.score,
            metrics=metrics,
            moving_inlier_xy=initial.moving_inlier_xy,
            fixed_inlier_xy=initial.fixed_inlier_xy,
        )
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
    if not np.isfinite(reduction_scale) or reduction_scale <= 0:
        raise ValueError(f"reduction_scale must be positive, got {reduction_scale}")

    reflected = bool(initial.metrics.get("reflection", False))
    center_angle, center_translation = _orientation_search_parameters(
        initial,
        reflected=reflected,
        shape_rc=fixed.shape,
    )
    search_px_to_um = float(pixel_size_um) / float(reduction_scale)
    angle_offsets = _inclusive_offsets(
        float(config.local_fine_angle_radius_degrees),
        float(config.local_fine_coarse_angle_step_degrees),
    )
    translation_offsets_um = _inclusive_offsets(
        float(config.local_fine_translation_radius_um),
        float(config.local_fine_coarse_translation_step_um),
    )
    translation_offsets_px = translation_offsets_um / search_px_to_um
    coarse_candidates, coarse_volume = _score_local_orientation_grid(
        center_angle=center_angle,
        center_translation_xy=center_translation,
        angle_offsets=angle_offsets,
        x_offsets_px=translation_offsets_px,
        y_offsets_px=translation_offsets_px,
        reflected=reflected,
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=config,
        search_stage="local_fine_coarse",
    )
    coarse_candidates = _apply_candidate_eligibility(
        coarse_candidates,
        config=config,
    )
    coarse_volume = _candidate_score_volume(
        coarse_candidates,
        coarse_volume.shape,
    )
    center_index = (
        int(np.argmin(np.abs(angle_offsets))),
        int(np.argmin(np.abs(translation_offsets_px))),
        int(np.argmin(np.abs(translation_offsets_px))),
    )
    center_candidate = coarse_candidates[
        np.ravel_multi_index(center_index, coarse_volume.shape)
    ]
    peak_indices = _local_maximum_indices(coarse_volume)
    if not peak_indices:
        raise RuntimeError("Final local orientation search produced no candidates")

    peaks: list[_LocalFinePeak] = []
    for coarse_index in peak_indices[: int(config.local_fine_maxima_to_refine)]:
        coarse_candidate = coarse_candidates[
            np.ravel_multi_index(coarse_index, coarse_volume.shape)
        ]
        refined_angle = float(coarse_candidate.angle_degrees)
        refined_translation = (
            float(coarse_candidate.metrics["translation_x_px"]),
            float(coarse_candidate.metrics["translation_y_px"]),
        )
        refinement_interior = False
        refinement_iterations = 0
        for iteration in range(1, 5):
            refinement_iterations = iteration
            angle_center_offset = _signed_angle_difference(
                refined_angle,
                center_angle,
            )
            x_center_offset_um = (
                refined_translation[0] - center_translation[0]
            ) * search_px_to_um
            y_center_offset_um = (
                refined_translation[1] - center_translation[1]
            ) * search_px_to_um
            refine_angle_offsets = _bounded_local_offsets(
                center_offset=angle_center_offset,
                global_radius=float(config.local_fine_angle_radius_degrees),
                local_radius=float(config.local_fine_coarse_angle_step_degrees),
                maximum_step=float(config.local_fine_refine_angle_step_degrees),
            )
            refine_x_offsets_um = _bounded_local_offsets(
                center_offset=x_center_offset_um,
                global_radius=float(config.local_fine_translation_radius_um),
                local_radius=float(config.local_fine_coarse_translation_step_um),
                maximum_step=float(config.local_fine_refine_translation_step_um),
            )
            refine_y_offsets_um = _bounded_local_offsets(
                center_offset=y_center_offset_um,
                global_radius=float(config.local_fine_translation_radius_um),
                local_radius=float(config.local_fine_coarse_translation_step_um),
                maximum_step=float(config.local_fine_refine_translation_step_um),
            )
            refined_candidates, refined_volume = _score_local_orientation_grid(
                center_angle=refined_angle,
                center_translation_xy=refined_translation,
                angle_offsets=refine_angle_offsets,
                x_offsets_px=refine_x_offsets_um / search_px_to_um,
                y_offsets_px=refine_y_offsets_um / search_px_to_um,
                reflected=reflected,
                fixed=fixed,
                moving=moving,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                config=config,
                search_stage="local_fine_refine",
            )
            refined_candidates = _apply_candidate_eligibility(
                refined_candidates,
                config=config,
            )
            refined_volume = _candidate_score_volume(
                refined_candidates,
                refined_volume.shape,
            )
            refined_index = np.unravel_index(
                int(np.nanargmax(refined_volume)),
                refined_volume.shape,
            )
            refined_candidate = refined_candidates[
                np.ravel_multi_index(refined_index, refined_volume.shape)
            ]
            refinement_interior = all(
                0 < index < size - 1
                for index, size in zip(
                    refined_index,
                    refined_volume.shape,
                    strict=True,
                )
            )
            if refinement_interior:
                break
            refined_angle = float(refined_candidate.angle_degrees)
            refined_translation = (
                float(refined_candidate.metrics["translation_x_px"]),
                float(refined_candidate.metrics["translation_y_px"]),
            )
            if _is_on_local_search_boundary(
                refined_candidate,
                center_angle=center_angle,
                center_translation_xy=center_translation,
                angle_radius_degrees=float(config.local_fine_angle_radius_degrees),
                translation_radius_um=float(config.local_fine_translation_radius_um),
                search_px_to_um=search_px_to_um,
            ):
                break
        coarse_interior = all(
            0 < index < size - 1
            for index, size in zip(coarse_index, coarse_volume.shape, strict=True)
        )
        peaks.append(
            _LocalFinePeak(
                candidate=refined_candidate,
                coarse_index=coarse_index,
                coarse_score=float(coarse_volume[coarse_index]),
                prominence=_local_peak_prominence(coarse_volume, coarse_index),
                coarse_interior=coarse_interior,
                refinement_interior=refinement_interior,
                refinement_iterations=refinement_iterations,
            )
        )

    peaks = _distinct_local_peaks(
        peaks,
        angle_tolerance_degrees=float(config.local_fine_coarse_angle_step_degrees),
        translation_tolerance_px=float(
            config.local_fine_coarse_translation_step_um / search_px_to_um
        ),
    )
    selected_peak = max(peaks, key=lambda peak: peak.candidate.score)
    selected = selected_peak.candidate
    stable_peaks = [peak for peak in peaks if peak.is_stable]
    nearby_stable_peaks = [peak for peak in stable_peaks if peak is not selected_peak]
    competing_stable_peaks = [
        peak
        for peak in nearby_stable_peaks
        if peak.candidate.score
        >= selected.score - float(config.local_fine_competing_score_margin)
    ]
    local_metrics = {
        "enabled": True,
        "search_window": {
            "angle_radius_degrees": float(config.local_fine_angle_radius_degrees),
            "translation_radius_um": float(config.local_fine_translation_radius_um),
        },
        "coarse_grid_shape": [int(size) for size in coarse_volume.shape],
        "coarse_evaluated": int(coarse_volume.size),
        "maxima_refined": len(peaks),
        "initial": _orientation_candidate_summary(initial),
        "selected": _orientation_candidate_summary(selected),
        "selected_delta": {
            "angle_degrees": _signed_angle_difference(
                selected.angle_degrees,
                center_angle,
            ),
            "translation_x_um": float(
                (selected.metrics["translation_x_px"] - center_translation[0])
                * search_px_to_um
            ),
            "translation_y_um": float(
                (selected.metrics["translation_y_px"] - center_translation[1])
                * search_px_to_um
            ),
            "score": float(selected.score - center_candidate.score),
        },
        "selected_coordinate_stable": selected_peak.is_stable,
        "selected_peak_on_search_boundary": not selected_peak.coarse_interior,
        "selected_peak_on_refinement_boundary": not selected_peak.refinement_interior,
        "stable_maxima_count": len(stable_peaks),
        "has_nearby_stable_maximum": bool(nearby_stable_peaks),
        "has_competing_stable_maximum": bool(competing_stable_peaks),
        "competing_score_margin": float(config.local_fine_competing_score_margin),
        "maxima": [
            _local_peak_summary(
                peak,
                center_angle=center_angle,
                center_translation_xy=center_translation,
                search_px_to_um=search_px_to_um,
            )
            for peak in peaks
        ],
    }
    metrics = dict(initial.metrics)
    metrics.update(selected.metrics)
    metrics["local_fine_search"] = local_metrics
    metrics["provisional_selection"] = bool(
        metrics.get("provisional_selection", False)
        or not selected_peak.is_stable
        or competing_stable_peaks
    )
    result = OrientationResult(
        matrix=selected.matrix,
        method=initial.method,
        angle_degrees=selected.angle_degrees,
        scale=selected.scale,
        score=selected.score,
        metrics=metrics,
        moving_inlier_xy=initial.moving_inlier_xy,
        fixed_inlier_xy=initial.fixed_inlier_xy,
    )
    if output_dir is not None:
        _write_local_fine_search_qc(
            Path(output_dir),
            fixed=fixed,
            moving=moving,
            initial=initial,
            selected=result,
            local_metrics=local_metrics,
            coarse_volume=coarse_volume,
            angle_offsets=angle_offsets,
            translation_offsets_um=translation_offsets_um,
            peaks=peaks,
            center_angle=center_angle,
            center_translation_xy=center_translation,
            search_px_to_um=search_px_to_um,
        )
    return result


def _orientation_search_parameters(
    result: OrientationResult,
    *,
    reflected: bool,
    shape_rc: tuple[int, int],
) -> tuple[float, tuple[float, float]]:
    """Return canonical search angle and post-rotation translation."""
    if "search_rotation_degrees" in result.metrics:
        angle = float(result.metrics["search_rotation_degrees"]) % 360.0
    else:
        matrix = np.asarray(result.matrix, dtype=np.float64)
        angle = float(
            np.degrees(
                np.arctan2(
                    matrix[0, 1],
                    -matrix[0, 0] if reflected else matrix[0, 0],
                )
            )
            % 360.0
        )
    if "translation_x_px" in result.metrics:
        return angle, (
            float(result.metrics["translation_x_px"]),
            float(result.metrics["translation_y_px"]),
        )
    base = _orientation_base_matrix(angle, reflected=reflected, shape_rc=shape_rc)
    residual = np.asarray(result.matrix, dtype=np.float64) @ np.linalg.inv(base)
    return angle, (float(residual[0, 2]), float(residual[1, 2]))


def _inclusive_offsets(radius: float, maximum_step: float) -> np.ndarray:
    """Return a symmetric grid containing both bounds and zero."""
    intervals = max(2, int(np.ceil(2.0 * float(radius) / float(maximum_step))))
    if intervals % 2:
        intervals += 1
    return np.linspace(-float(radius), float(radius), intervals + 1, dtype=float)


def _bounded_local_offsets(
    *,
    center_offset: float,
    global_radius: float,
    local_radius: float,
    maximum_step: float,
) -> np.ndarray:
    """Return local offsets clipped to the original outer search window."""
    lower = max(-float(local_radius), -float(global_radius) - float(center_offset))
    upper = min(float(local_radius), float(global_radius) - float(center_offset))
    intervals = max(1, int(np.ceil((upper - lower) / float(maximum_step))))
    offsets = np.linspace(lower, upper, intervals + 1, dtype=float)
    if lower < 0.0 < upper and not np.any(np.isclose(offsets, 0.0)):
        offsets = np.sort(np.append(offsets, 0.0))
    return offsets


def _is_on_local_search_boundary(
    candidate: OrientationResult,
    *,
    center_angle: float,
    center_translation_xy: tuple[float, float],
    angle_radius_degrees: float,
    translation_radius_um: float,
    search_px_to_um: float,
) -> bool:
    """Return whether a candidate has reached any original local-search bound."""
    tolerance = 1e-6
    angle_offset = abs(_signed_angle_difference(candidate.angle_degrees, center_angle))
    x_offset_um = abs(
        (float(candidate.metrics["translation_x_px"]) - center_translation_xy[0])
        * search_px_to_um
    )
    y_offset_um = abs(
        (float(candidate.metrics["translation_y_px"]) - center_translation_xy[1])
        * search_px_to_um
    )
    return bool(
        angle_offset >= angle_radius_degrees - tolerance
        or x_offset_um >= translation_radius_um - tolerance
        or y_offset_um >= translation_radius_um - tolerance
    )


def _score_local_orientation_grid(
    *,
    center_angle: float,
    center_translation_xy: tuple[float, float],
    angle_offsets: np.ndarray,
    x_offsets_px: np.ndarray,
    y_offsets_px: np.ndarray,
    reflected: bool,
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    config: OrientationSearchConfig,
    search_stage: str,
) -> tuple[list[OrientationResult], np.ndarray]:
    """Score a deterministic angle/X/Y grid in row-major order."""
    candidates = [
        _score_orientation_transform(
            center_angle + float(angle_offset),
            translation_xy=(
                center_translation_xy[0] + float(x_offset),
                center_translation_xy[1] + float(y_offset),
            ),
            reflected=reflected,
            fixed=fixed,
            moving=moving,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            config=config,
            search_stage=search_stage,
            translation_seed_rank=1,
        )
        for angle_offset in angle_offsets
        for x_offset in x_offsets_px
        for y_offset in y_offsets_px
    ]
    shape = (len(angle_offsets), len(x_offsets_px), len(y_offsets_px))
    return candidates, np.empty(shape, dtype=np.float64)


def _candidate_score_volume(
    candidates: list[OrientationResult],
    shape: tuple[int, ...],
) -> np.ndarray:
    """Reshape eligible candidate scores into their search-grid volume."""
    scores = [
        candidate.score if bool(candidate.metrics.get("eligible", True)) else -np.inf
        for candidate in candidates
    ]
    volume = np.asarray(scores, dtype=np.float64).reshape(shape)
    if not np.isfinite(volume).any():
        volume = np.asarray(
            [candidate.score for candidate in candidates],
            dtype=np.float64,
        ).reshape(shape)
    return volume


def _local_maximum_indices(volume: np.ndarray) -> list[tuple[int, int, int]]:
    """Return one representative from each 3D local-maximum plateau."""
    local_maximum = np.isfinite(volume) & (
        volume >= ndi.maximum_filter(volume, size=3, mode="constant", cval=-np.inf)
    )
    labels, count = ndi.label(local_maximum)
    indices: list[tuple[int, int, int]] = []
    for label in range(1, count + 1):
        locations = np.argwhere(labels == label)
        if locations.size == 0:
            continue
        scores = np.asarray([volume[tuple(location)] for location in locations])
        selected = locations[int(np.argmax(scores))]
        indices.append((int(selected[0]), int(selected[1]), int(selected[2])))
    return sorted(indices, key=lambda index: float(volume[index]), reverse=True)


def _local_peak_prominence(
    volume: np.ndarray,
    index: tuple[int, int, int],
) -> float:
    """Measure a peak against the median of its immediate finite neighborhood."""
    slices = tuple(
        slice(max(0, coordinate - 1), min(size, coordinate + 2))
        for coordinate, size in zip(index, volume.shape, strict=True)
    )
    neighborhood = np.asarray(volume[slices], dtype=np.float64)
    finite = neighborhood[np.isfinite(neighborhood)]
    if finite.size <= 1:
        return float("nan")
    return float(volume[index] - np.median(finite))


def _distinct_local_peaks(
    peaks: list[_LocalFinePeak],
    *,
    angle_tolerance_degrees: float,
    translation_tolerance_px: float,
) -> list[_LocalFinePeak]:
    """Merge coarse maxima that converge to the same refined maximum."""
    distinct: list[_LocalFinePeak] = []
    for peak in sorted(peaks, key=lambda item: item.candidate.score, reverse=True):
        if any(
            abs(
                _signed_angle_difference(
                    peak.candidate.angle_degrees,
                    retained.candidate.angle_degrees,
                )
            )
            <= angle_tolerance_degrees
            and np.hypot(
                float(peak.candidate.metrics["translation_x_px"])
                - float(retained.candidate.metrics["translation_x_px"]),
                float(peak.candidate.metrics["translation_y_px"])
                - float(retained.candidate.metrics["translation_y_px"]),
            )
            <= translation_tolerance_px
            for retained in distinct
        ):
            continue
        distinct.append(peak)
    return distinct


def _signed_angle_difference(angle: float, reference: float) -> float:
    """Return the signed shortest angular displacement in degrees."""
    return float((float(angle) - float(reference) + 180.0) % 360.0 - 180.0)


def _local_peak_summary(
    peak: _LocalFinePeak,
    *,
    center_angle: float,
    center_translation_xy: tuple[float, float],
    search_px_to_um: float,
) -> dict[str, Any]:
    """Return JSON-safe coordinates and persistence metadata for one peak."""
    summary = _orientation_candidate_summary(peak.candidate)
    assert summary is not None
    summary.update(
        {
            "offset_angle_degrees": _signed_angle_difference(
                peak.candidate.angle_degrees,
                center_angle,
            ),
            "offset_translation_x_um": float(
                (
                    float(peak.candidate.metrics["translation_x_px"])
                    - center_translation_xy[0]
                )
                * search_px_to_um
            ),
            "offset_translation_y_um": float(
                (
                    float(peak.candidate.metrics["translation_y_px"])
                    - center_translation_xy[1]
                )
                * search_px_to_um
            ),
            "coarse_score": peak.coarse_score,
            "coarse_prominence": peak.prominence,
            "coarse_interior": peak.coarse_interior,
            "refinement_interior": peak.refinement_interior,
            "refinement_iterations": peak.refinement_iterations,
            "stable": peak.is_stable,
        }
    )
    return summary


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
        "reflection": bool(candidate.metrics.get("reflection", False)),
        "search_rotation_degrees": float(
            candidate.metrics.get("search_rotation_degrees", candidate.angle_degrees)
        ),
        "matrix_diagnostic_angle_degrees": float(
            candidate.metrics.get("matrix_diagnostic_angle_degrees", np.nan)
        ),
        "equivalent_top_bottom_flip_rotation_degrees": candidate.metrics.get(
            "equivalent_top_bottom_flip_rotation_degrees"
        ),
        "reflection_axis": candidate.metrics.get("reflection_axis", "unknown"),
        "translation_x_px": float(candidate.metrics.get("translation_x_px", np.nan)),
        "translation_y_px": float(candidate.metrics.get("translation_y_px", np.nan)),
        "fixed_overlap_fraction": float(
            candidate.metrics.get("fixed_overlap_fraction", np.nan)
        ),
        "moving_overlap_fraction": float(
            candidate.metrics.get("moving_overlap_fraction", np.nan)
        ),
        "retained_moving_fraction": float(
            candidate.metrics.get("retained_moving_fraction", np.nan)
        ),
        "eligible": bool(candidate.metrics.get("eligible", True)),
        "eligibility_reasons": list(candidate.metrics.get("eligibility_reasons", [])),
        "n_inliers": int(candidate.metrics.get("n_inliers", 0)),
        "inlier_coverage": float(candidate.metrics.get("inlier_coverage", 0.0)),
    }


def _write_local_fine_search_qc(
    output_dir: Path,
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    initial: OrientationResult,
    selected: OrientationResult,
    local_metrics: dict[str, Any],
    coarse_volume: np.ndarray,
    angle_offsets: np.ndarray,
    translation_offsets_um: np.ndarray,
    peaks: list[_LocalFinePeak],
    center_angle: float,
    center_translation_xy: tuple[float, float],
    search_px_to_um: float,
) -> None:
    """Write local refinement metadata, before/after overlay, and score slices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "orientation_local_fine_search.json").write_text(
        json.dumps(local_metrics, indent=2, allow_nan=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, label, candidate in zip(
        axes,
        ("Before local fine search", "After local fine search"),
        (initial, selected),
        strict=True,
    ):
        warped = warp_image(moving, candidate.matrix, output_shape_rc=fixed.shape)
        axis.imshow(_overlay_rgb(fixed, warped))
        angle, translation = _orientation_search_parameters(
            candidate,
            reflected=bool(candidate.metrics.get("reflection", False)),
            shape_rc=fixed.shape,
        )
        axis.set_title(
            f"{label}\nangle={angle:.2f}°, "
            f"dx={translation[0]:+.2f}, dy={translation[1]:+.2f}\n"
            f"score={candidate.score:.5f}"
        )
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "orientation_local_fine_overlay.png", dpi=160)
    plt.close(fig)

    score_angle_x = np.max(coarse_volume, axis=2)
    score_angle_y = np.max(coarse_volume, axis=1)
    extent = [
        float(translation_offsets_um[0]),
        float(translation_offsets_um[-1]),
        float(angle_offsets[0]),
        float(angle_offsets[-1]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, surface, translation_axis in zip(
        axes,
        (score_angle_x, score_angle_y),
        ("X", "Y"),
        strict=True,
    ):
        image = axis.imshow(
            surface,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
        )
        for peak in peaks:
            summary = _local_peak_summary(
                peak,
                center_angle=center_angle,
                center_translation_xy=center_translation_xy,
                search_px_to_um=search_px_to_um,
            )
            translation_offset = float(
                summary[f"offset_translation_{translation_axis.lower()}_um"]
            )
            axis.scatter(
                translation_offset,
                float(summary["offset_angle_degrees"]),
                marker="*" if peak.is_stable else "x",
                s=90,
                edgecolors="white" if peak.is_stable else None,
                linewidths=0.8,
                color="red",
            )
        axis.set_title(
            f"Maximum over translation {('Y' if translation_axis == 'X' else 'X')}"
        )
        axis.set_xlabel(f"translation {translation_axis} offset (µm)")
        axis.set_ylabel("angle offset (degrees)")
        fig.colorbar(image, ax=axis, label="orientation score")
    fig.suptitle(
        "Final local orientation landscape\n"
        "stars = persistent interior maxima; crosses = boundary/unstable maxima"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "orientation_local_fine_landscape.png", dpi=160)
    plt.close(fig)


def _write_orientation_search_qc(
    output_dir: Path,
    *,
    fixed: np.ndarray,
    moving: np.ndarray,
    candidates: list[OrientationResult],
    selected: OrientationResult,
) -> None:
    """Write candidate metadata, overlays, and an angle/translation landscape."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected": _orientation_candidate_summary(selected),
        "selection_reason": selected.metrics.get("selection_reason"),
        "handedness_ambiguous": bool(
            selected.metrics.get("handedness_ambiguous", False)
        ),
        "provisional_selection": bool(
            selected.metrics.get("provisional_selection", False)
        ),
        "local_fine_search": selected.metrics.get("local_fine_search"),
        "candidates": [
            _orientation_candidate_summary(candidate) for candidate in candidates
        ],
    }
    (output_dir / "orientation_candidates.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True)
    )

    best_by_reflection = {
        reflected: max(
            (
                candidate
                for candidate in candidates
                if bool(candidate.metrics.get("reflection", False)) == reflected
            ),
            key=lambda candidate: candidate.score,
            default=None,
        )
        for reflected in (False, True)
    }
    displayed = [
        ("Best non-reflected", best_by_reflection[False]),
        ("Best reflected", best_by_reflection[True]),
        ("Selected", selected),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (label, candidate) in zip(axes, displayed, strict=True):
        if candidate is None:
            axis.text(0.5, 0.5, "Not searched", ha="center", va="center")
            axis.axis("off")
            continue
        warped = warp_image(
            moving,
            candidate.matrix,
            output_shape_rc=fixed.shape,
        )
        axis.imshow(_overlay_rgb(fixed, warped))
        axis.set_title(
            f"{label}\nreflection={candidate.metrics.get('reflection')}, "
            f"angle={candidate.angle_degrees:.1f}°, "
            f"dx={candidate.metrics.get('translation_x_px', np.nan):+.1f}, "
            f"dy={candidate.metrics.get('translation_y_px', np.nan):+.1f}\n"
            f"score={candidate.score:.4f}, "
            f"eligible={candidate.metrics.get('eligible', True)}"
        )
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "orientation_candidate_overlays.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, reflected in zip(axes, (False, True), strict=True):
        branch = [
            candidate
            for candidate in candidates
            if bool(candidate.metrics.get("reflection", False)) == reflected
        ]
        if branch:
            angles = [candidate.angle_degrees for candidate in branch]
            translations_x = [
                float(candidate.metrics.get("translation_x_px", np.nan))
                for candidate in branch
            ]
            scores = [candidate.score for candidate in branch]
            scatter = axis.scatter(
                angles,
                translations_x,
                c=scores,
                cmap="viridis",
                s=70,
            )
            for candidate in branch:
                axis.annotate(
                    f"dy={candidate.metrics.get('translation_y_px', np.nan):+.0f}",
                    (
                        candidate.angle_degrees,
                        float(candidate.metrics.get("translation_x_px", np.nan)),
                    ),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
            fig.colorbar(scatter, ax=axis, label="orientation score")
        axis.set_title("Reflected" if reflected else "Non-reflected")
        axis.set_xlabel("search rotation (degrees)")
        axis.set_ylabel("translation x (search pixels)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "orientation_angle_translation_landscape.png", dpi=160)
    plt.close(fig)


def _overlay_rgb(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Return a green/magenta normalized overlay for orientation QC."""
    fixed_normalized = _normalize_for_overlay(fixed)
    moving_normalized = _normalize_for_overlay(moving)
    rgb = np.zeros((*fixed_normalized.shape, 3), dtype=np.float32)
    rgb[..., 0] = moving_normalized
    rgb[..., 1] = fixed_normalized
    rgb[..., 2] = moving_normalized
    return rgb


def _normalize_for_overlay(image: np.ndarray) -> np.ndarray:
    """Normalize nonzero image intensities robustly for plotting."""
    array = np.asarray(image, dtype=np.float32)
    populated = array[array > 0]
    if populated.size == 0:
        return np.zeros(array.shape, dtype=np.float32)
    lower, upper = np.percentile(populated, (0.5, 99.5))
    return np.asarray(
        np.clip((array - lower) / max(float(upper - lower), 1e-6), 0.0, 1.0),
        dtype=np.float32,
    )


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
    base = _orientation_base_matrix(angle, reflected=reflected, shape_rc=fixed.shape)
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
    return _score_orientation_transform(
        angle,
        translation_xy=(float(shift_rc[1]), float(shift_rc[0])),
        reflected=reflected,
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=config,
        search_stage="single_phase",
        translation_seed_rank=1,
    )


def _score_orientation_transform(
    angle: float,
    *,
    translation_xy: tuple[float, float],
    reflected: bool,
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    config: OrientationSearchConfig,
    search_stage: str,
    translation_seed_rank: int,
) -> OrientationResult:
    """Score one explicit handedness, angle, and translation transform."""
    base = _orientation_base_matrix(angle, reflected=reflected, shape_rc=fixed.shape)
    matrix = _translation_matrix(*translation_xy) @ base
    warped_mask = warp_image(
        moving_mask,
        matrix,
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    warped_image = warp_image(moving, matrix, output_shape_rc=fixed.shape)
    dice = tissue_dice(fixed_mask, warped_mask)
    fixed_binary = np.asarray(fixed_mask) > 0
    moving_binary = np.asarray(moving_mask) > 0
    warped_binary = np.asarray(warped_mask) > 0
    overlap = fixed_binary & warped_binary
    fixed_overlap_fraction = float(overlap.sum() / max(1, fixed_binary.sum()))
    moving_overlap_fraction = float(overlap.sum() / max(1, warped_binary.sum()))
    retained_moving_fraction = float(warped_binary.sum() / max(1, moving_binary.sum()))
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
            "search_rotation_degrees": float(angle % 360.0),
            "matrix_diagnostic_angle_degrees": float(
                np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])) % 360.0
            ),
            "equivalent_top_bottom_flip_rotation_degrees": (
                float(((angle - 180.0 + 180.0) % 360.0) - 180.0) if reflected else None
            ),
            "reflection_axis": "left_right" if reflected else "none",
            "translation_x_px": float(translation_xy[0]),
            "translation_y_px": float(translation_xy[1]),
            "fixed_overlap_fraction": fixed_overlap_fraction,
            "moving_overlap_fraction": moving_overlap_fraction,
            "retained_moving_fraction": retained_moving_fraction,
            "search_stage": search_stage,
            "translation_seed_rank": int(translation_seed_rank),
            "eligible": True,
            "eligibility_reasons": [],
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
    seen: set[tuple[float, bool, float, float]] = set()
    for item in ordered:
        key = (
            round(float(item.angle_degrees), 6),
            bool(item.metrics.get("reflection", False)),
            round(float(item.metrics.get("translation_x_px", 0.0)), 3),
            round(float(item.metrics.get("translation_y_px", 0.0)), 3),
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
    metrics = dict(result.metrics)
    scale = float(small_to_full[0, 0])
    if "translation_x_px" in metrics and "translation_y_px" in metrics:
        metrics["translation_x_search_px"] = float(metrics["translation_x_px"])
        metrics["translation_y_search_px"] = float(metrics["translation_y_px"])
        metrics["translation_x_px"] = float(metrics["translation_x_px"]) * scale
        metrics["translation_y_px"] = float(metrics["translation_y_px"]) * scale
    return OrientationResult(
        matrix=full_matrix,
        method=result.method,
        angle_degrees=result.angle_degrees,
        scale=result.scale,
        score=result.score,
        metrics=metrics,
        moving_inlier_xy=moving_inliers,
        fixed_inlier_xy=fixed_inliers,
    )


def _apply_affine(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points_xy, np.ones(len(points_xy))])
    return np.asarray((homogeneous @ matrix.T)[:, :2], dtype=np.float64)
