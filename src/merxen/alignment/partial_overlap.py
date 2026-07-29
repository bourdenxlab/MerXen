"""Partial-overlap-aware rigid refinement for paired DAPI sections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage.registration import phase_cross_correlation
from skimage.transform import resize

from merxen.alignment.orientation import OrientationResult, warp_image
from merxen.config import PartialOverlapRigidConfig
from merxen.plotting import prepare_plot_output, save_figure


@dataclass(frozen=True)
class _Candidate:
    """One residual rigid candidate on the reduced registration canvas."""

    angle_degrees: float
    translation_x_px: float
    translation_y_px: float
    score: float
    metrics: dict[str, float | bool]
    search_stage: str = "evaluated"


class _PartialOverlapObjective:
    """Evaluate robust boundary and density agreement for rigid candidates."""

    def __init__(
        self: _PartialOverlapObjective,
        fixed_image: np.ndarray,
        moving_image: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
        fixed_valid: np.ndarray,
        moving_valid: np.ndarray,
        *,
        density_sigma_px: float,
        boundary_distance_scale_px: float,
        config: PartialOverlapRigidConfig,
    ) -> None:
        self.fixed_image = np.asarray(fixed_image, dtype=np.float32)
        self.moving_image = np.asarray(moving_image, dtype=np.float32)
        self.fixed_mask = np.asarray(fixed_mask, dtype=bool)
        self.moving_mask = np.asarray(moving_mask, dtype=bool)
        self.fixed_valid = np.asarray(fixed_valid, dtype=bool)
        self.moving_valid = np.asarray(moving_valid, dtype=bool)
        self.config = config
        self.shape_rc: tuple[int, int] = (
            int(self.fixed_image.shape[0]),
            int(self.fixed_image.shape[1]),
        )
        self.center_xy = (
            (self.shape_rc[1] - 1.0) / 2.0,
            (self.shape_rc[0] - 1.0) / 2.0,
        )
        sigma = max(0.5, float(density_sigma_px))
        self.fixed_density = ndi.gaussian_filter(
            self.fixed_image,
            sigma=sigma,
            mode="reflect",
        )
        self.moving_density = ndi.gaussian_filter(
            self.moving_image,
            sigma=sigma,
            mode="reflect",
        )
        self.fixed_boundary = _mask_boundary(self.fixed_mask) & self.fixed_valid
        self.fixed_boundary_distance = ndi.distance_transform_edt(~self.fixed_boundary)
        self.boundary_distance_scale_px = max(
            1.0,
            float(boundary_distance_scale_px),
        )

    def matrix(
        self: _PartialOverlapObjective,
        parameters: Any,
    ) -> np.ndarray:
        """Build a proper residual rotation about canvas center plus translation."""
        angle, translation_x, translation_y = np.asarray(
            parameters,
            dtype=np.float64,
        )
        matrix = np.vstack(
            [
                cv2.getRotationMatrix2D(
                    self.center_xy,
                    float(angle),
                    1.0,
                ),
                [0.0, 0.0, 1.0],
            ]
        )
        matrix[0, 2] += float(translation_x)
        matrix[1, 2] += float(translation_y)
        return np.asarray(matrix, dtype=np.float64)

    def evaluate(
        self: _PartialOverlapObjective,
        parameters: Any,
        *,
        search_stage: str = "evaluated",
    ) -> _Candidate:
        """Return the robust partial-overlap score for one candidate."""
        angle, translation_x, translation_y = np.asarray(
            parameters,
            dtype=np.float64,
        )
        matrix = self.matrix(parameters)
        moving_mask = (
            warp_image(
                self.moving_mask.astype(np.uint8),
                matrix,
                output_shape_rc=self.shape_rc,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        moving_valid = (
            warp_image(
                self.moving_valid.astype(np.uint8),
                matrix,
                output_shape_rc=self.shape_rc,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        fixed_tissue = self.fixed_mask & self.fixed_valid
        moving_tissue = moving_mask & moving_valid
        overlap = fixed_tissue & moving_tissue
        fixed_overlap = _safe_fraction(int(overlap.sum()), int(fixed_tissue.sum()))
        moving_overlap = _safe_fraction(
            int(overlap.sum()),
            int(moving_tissue.sum()),
        )

        common_valid = self.fixed_valid & moving_valid
        moving_boundary = _mask_boundary(moving_mask) & common_valid
        fixed_boundary = self.fixed_boundary & common_valid
        moving_to_fixed = self.fixed_boundary_distance[moving_boundary]
        if np.any(moving_boundary):
            moving_boundary_distance = ndi.distance_transform_edt(~moving_boundary)
            fixed_to_moving = moving_boundary_distance[fixed_boundary]
        else:
            fixed_to_moving = np.empty(0, dtype=np.float64)
        retained = float(self.config.retained_boundary_fraction)
        moving_distance = _trimmed_mean(moving_to_fixed, retained)
        fixed_distance = _trimmed_mean(fixed_to_moving, retained)
        if np.isfinite(moving_distance) and np.isfinite(fixed_distance):
            boundary_distance = 0.5 * (moving_distance + fixed_distance)
            boundary_score = float(
                np.exp(-boundary_distance / self.boundary_distance_scale_px)
            )
        else:
            boundary_distance = float("inf")
            boundary_score = 0.0

        moving_density = warp_image(
            self.moving_density,
            matrix,
            output_shape_rc=self.shape_rc,
        )
        density_correlation = _masked_correlation(
            self.fixed_density,
            moving_density,
            overlap & common_valid,
        )
        density_score = (
            max(0.0, float(density_correlation))
            if np.isfinite(density_correlation)
            else 0.0
        )
        weight_total = float(self.config.boundary_weight) + float(
            self.config.density_weight
        )
        score = (
            float(self.config.boundary_weight) * boundary_score
            + float(self.config.density_weight) * density_score
        ) / weight_total

        fixed_deficit = max(
            0.0,
            float(self.config.minimum_fixed_overlap_fraction) - fixed_overlap,
        )
        moving_deficit = max(
            0.0,
            float(self.config.minimum_moving_overlap_fraction) - moving_overlap,
        )
        score -= float(self.config.overlap_penalty_weight) * (
            fixed_deficit + moving_deficit
        )
        eligible = fixed_deficit == 0.0 and moving_deficit == 0.0
        return _Candidate(
            angle_degrees=float(angle),
            translation_x_px=float(translation_x),
            translation_y_px=float(translation_y),
            score=float(score),
            metrics={
                "score": float(score),
                "boundary_score": boundary_score,
                "trimmed_boundary_distance_px": float(boundary_distance),
                "density_correlation": float(density_correlation),
                "fixed_overlap_fraction": fixed_overlap,
                "moving_overlap_fraction": moving_overlap,
                "eligible": eligible,
            },
            search_stage=str(search_stage),
        )


def evaluate_aligned_partial_overlap_objective(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    *,
    config: PartialOverlapRigidConfig,
    pixel_size_um: float,
    fixed_valid_mask: Any | None = None,
    moving_valid_mask: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the robust objective at identity for already-aligned images."""
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")

    (
        fixed,
        moving,
        fixed_binary,
        moving_binary,
        fixed_valid,
        moving_valid,
        full_to_small,
    ) = _reduced_inputs(
        fixed_image,
        moving_image,
        fixed_mask,
        moving_mask,
        fixed_valid_mask=fixed_valid_mask,
        moving_valid_mask=moving_valid_mask,
        max_dimension=int(config.max_dimension_px),
    )
    reduction_scale = float(full_to_small[0, 0])
    objective = _PartialOverlapObjective(
        fixed,
        moving,
        fixed_binary,
        moving_binary,
        fixed_valid,
        moving_valid,
        density_sigma_px=(
            float(config.density_sigma_um) / float(pixel_size_um) * reduction_scale
        ),
        boundary_distance_scale_px=(
            float(config.boundary_distance_scale_um)
            / float(pixel_size_um)
            * reduction_scale
        ),
        config=config,
    )
    candidate = objective.evaluate(
        [0.0, 0.0, 0.0],
        search_stage="aligned_identity",
    )
    summary = _candidate_summary(
        candidate,
        reduction_scale=reduction_scale,
        pixel_size_um=float(pixel_size_um),
    )
    boundary_distance_px = float(summary["trimmed_boundary_distance_px"])
    summary["trimmed_boundary_distance_um"] = (
        boundary_distance_px * float(pixel_size_um) / reduction_scale
    )
    return summary


def refine_partial_overlap_rigid(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    *,
    initial: OrientationResult,
    config: PartialOverlapRigidConfig,
    pixel_size_um: float,
    fixed_valid_mask: Any | None = None,
    moving_valid_mask: Any | None = None,
    output_dir: Path | None = None,
) -> OrientationResult:
    """Jointly refine rotation and translation without rewarding extra overlap."""
    if not config.enabled:
        return initial
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")

    (
        fixed,
        moving,
        fixed_binary,
        moving_binary,
        fixed_valid,
        moving_valid,
        full_to_small,
    ) = _reduced_inputs(
        fixed_image,
        moving_image,
        fixed_mask,
        moving_mask,
        fixed_valid_mask=fixed_valid_mask,
        moving_valid_mask=moving_valid_mask,
        max_dimension=int(config.max_dimension_px),
    )
    initial_small = (
        full_to_small
        @ np.asarray(initial.matrix, dtype=np.float64)
        @ np.linalg.inv(full_to_small)
    )
    moving_pre = warp_image(
        moving,
        initial_small,
        output_shape_rc=fixed.shape,
    )
    moving_pre_mask = warp_image(
        moving_binary,
        initial_small,
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    moving_pre_valid = warp_image(
        moving_valid,
        initial_small,
        output_shape_rc=fixed.shape,
        interpolation=cv2.INTER_NEAREST,
    )
    reduction_scale = float(full_to_small[0, 0])
    density_sigma_px = (
        float(config.density_sigma_um) / float(pixel_size_um) * reduction_scale
    )
    boundary_distance_scale_px = (
        float(config.boundary_distance_scale_um)
        / float(pixel_size_um)
        * reduction_scale
    )
    maximum_translation_px = (
        float(config.maximum_translation_um) / float(pixel_size_um) * reduction_scale
    )
    objective = _PartialOverlapObjective(
        fixed,
        moving_pre,
        fixed_binary,
        moving_pre_mask,
        fixed_valid,
        moving_pre_valid,
        density_sigma_px=density_sigma_px,
        boundary_distance_scale_px=boundary_distance_scale_px,
        config=config,
    )

    baseline = objective.evaluate(
        [0.0, 0.0, 0.0],
        search_stage="baseline",
    )
    coarse_candidates = _coarse_candidates(
        objective,
        config=config,
        maximum_translation_px=maximum_translation_px,
    )
    starts = _best_distinct(
        [baseline, *coarse_candidates],
        count=int(config.candidates_to_refine),
        angle_separation=max(
            0.25,
            float(config.coarse_angle_step_degrees) / 2.0,
        ),
    )
    refined = [
        _refine_candidate(
            objective,
            candidate,
            config=config,
            maximum_translation_px=maximum_translation_px,
        )
        for candidate in starts
    ]
    ranked = sorted(
        [baseline, *coarse_candidates, *refined],
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    eligible = [
        candidate
        for candidate in ranked
        if bool(candidate.metrics.get("eligible", False))
    ]
    proposed = eligible[0] if eligible else baseline
    improvement = float(proposed.score - baseline.score)
    accepted = proposed is not baseline and improvement >= float(
        config.minimum_score_improvement
    )
    selected = proposed if accepted else baseline
    distinct_ranked = _best_distinct(
        ranked,
        count=max(2, int(config.candidates_to_refine)),
        angle_separation=0.1,
    )
    runner_up = next(
        (
            candidate
            for candidate in distinct_ranked
            if candidate is not selected
            and (
                abs(candidate.angle_degrees - selected.angle_degrees) > 0.05
                or np.hypot(
                    candidate.translation_x_px - selected.translation_x_px,
                    candidate.translation_y_px - selected.translation_y_px,
                )
                > 0.5
            )
        ),
        None,
    )
    score_margin = (
        float("nan") if runner_up is None else float(selected.score - runner_up.score)
    )
    ambiguous = bool(
        runner_up is not None and score_margin < float(config.ambiguity_score_margin)
    )

    residual_small = objective.matrix(
        [
            selected.angle_degrees,
            selected.translation_x_px,
            selected.translation_y_px,
        ]
    )
    residual_full = np.linalg.inv(full_to_small) @ residual_small @ full_to_small
    selected_matrix = residual_full @ np.asarray(initial.matrix, dtype=np.float64)
    determinant = float(np.linalg.det(selected_matrix[:2, :2]))
    scale = float(np.sqrt(abs(determinant)))
    angle = float(
        np.degrees(np.arctan2(selected_matrix[1, 0], selected_matrix[0, 0])) % 360.0
    )
    candidate_summaries = [
        _candidate_summary(
            candidate,
            reduction_scale=reduction_scale,
            pixel_size_um=float(pixel_size_um),
        )
        for candidate in distinct_ranked
    ]
    partial_metrics: dict[str, Any] = {
        "enabled": True,
        "accepted": accepted,
        "ambiguous": ambiguous,
        "selection_reason": (
            "partial_overlap_score_improved"
            if accepted
            else (
                "no_candidate_met_overlap_constraints"
                if not eligible
                else "improvement_below_minimum"
            )
        ),
        "baseline": _candidate_summary(
            baseline,
            reduction_scale=reduction_scale,
            pixel_size_um=float(pixel_size_um),
        ),
        "selected": _candidate_summary(
            selected,
            reduction_scale=reduction_scale,
            pixel_size_um=float(pixel_size_um),
        ),
        "score_improvement": improvement,
        "runner_up_score_margin": score_margin,
        "residual_matrix": residual_full.tolist(),
        "candidate_ranking": candidate_summaries,
    }
    metrics = dict(initial.metrics)
    metrics["partial_overlap"] = partial_metrics

    if output_dir is not None:
        _write_qc(
            Path(output_dir),
            fixed_full=np.asarray(fixed_image),
            moving_full=np.asarray(moving_image),
            initial_matrix=np.asarray(initial.matrix, dtype=np.float64),
            selected_matrix=selected_matrix,
            fixed_small=fixed,
            moving_pre_small=moving_pre,
            objective=objective,
            candidates=distinct_ranked,
            coarse_candidates=coarse_candidates,
            refined_candidates=refined,
            baseline=baseline,
            selected=selected,
            pixels_to_um=float(pixel_size_um) / reduction_scale,
            metrics=partial_metrics,
        )

    return OrientationResult(
        matrix=selected_matrix,
        method=(
            "partial_overlap_rigid"
            if accepted
            else f"{initial.method}_partial_overlap_retained"
        ),
        angle_degrees=angle,
        scale=scale,
        score=float(selected.score),
        metrics=metrics,
        moving_inlier_xy=initial.moving_inlier_xy,
        fixed_inlier_xy=initial.fixed_inlier_xy,
    )


def _reduced_inputs(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    *,
    fixed_valid_mask: Any | None,
    moving_valid_mask: Any | None,
    max_dimension: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    fixed = np.asarray(fixed_image, dtype=np.uint8)
    moving = np.asarray(moving_image, dtype=np.uint8)
    fixed_binary = (np.asarray(fixed_mask) > 0).astype(np.uint8)
    moving_binary = (np.asarray(moving_mask) > 0).astype(np.uint8)
    fixed_valid = (
        np.ones_like(fixed_binary)
        if fixed_valid_mask is None
        else (np.asarray(fixed_valid_mask) > 0).astype(np.uint8)
    )
    moving_valid = (
        np.ones_like(moving_binary)
        if moving_valid_mask is None
        else (np.asarray(moving_valid_mask) > 0).astype(np.uint8)
    )
    arrays = (
        fixed,
        moving,
        fixed_binary,
        moving_binary,
        fixed_valid,
        moving_valid,
    )
    if any(array.shape != fixed.shape for array in arrays):
        raise ValueError("Partial-overlap images and masks must share one canvas")
    scale = min(1.0, float(max_dimension) / float(max(fixed.shape)))
    if scale < 1.0:
        target_shape = (
            max(1, int(round(fixed.shape[0] * scale))),
            max(1, int(round(fixed.shape[1] * scale))),
        )
        fixed = _resize(fixed, target_shape, order=1).astype(np.uint8)
        moving = _resize(moving, target_shape, order=1).astype(np.uint8)
        fixed_binary = (_resize(fixed_binary, target_shape, order=0) > 0).astype(
            np.uint8
        )
        moving_binary = (_resize(moving_binary, target_shape, order=0) > 0).astype(
            np.uint8
        )
        fixed_valid = (_resize(fixed_valid, target_shape, order=0) > 0).astype(np.uint8)
        moving_valid = (_resize(moving_valid, target_shape, order=0) > 0).astype(
            np.uint8
        )
    full_to_small = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (
        fixed,
        moving,
        fixed_binary,
        moving_binary,
        fixed_valid,
        moving_valid,
        full_to_small,
    )


def _resize(image: np.ndarray, shape_rc: tuple[int, int], *, order: int) -> np.ndarray:
    return np.asarray(
        resize(
            image,
            shape_rc,
            order=order,
            preserve_range=True,
            anti_aliasing=order > 0,
        )
    )


def _coarse_candidates(
    objective: _PartialOverlapObjective,
    *,
    config: PartialOverlapRigidConfig,
    maximum_translation_px: float,
) -> list[_Candidate]:
    radius = float(config.angle_search_radius_degrees)
    step = float(config.coarse_angle_step_degrees)
    angles = np.arange(-radius, radius + step * 0.5, step)
    candidates: list[_Candidate] = []
    for angle in angles:
        rotation = objective.matrix([float(angle), 0.0, 0.0])
        rotated_density = warp_image(
            objective.moving_density,
            rotation,
            output_shape_rc=objective.shape_rc,
        )
        rotated_mask = (
            warp_image(
                objective.moving_mask.astype(np.uint8),
                rotation,
                output_shape_rc=objective.shape_rc,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        rotated_valid = (
            warp_image(
                objective.moving_valid.astype(np.uint8),
                rotation,
                output_shape_rc=objective.shape_rc,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        phase_translation = _phase_translation(
            objective.fixed_density,
            rotated_density,
            objective.fixed_mask & objective.fixed_valid,
            rotated_mask & rotated_valid,
        )
        centroid_translation = _centroid_translation(
            objective.fixed_mask & objective.fixed_valid,
            rotated_mask & rotated_valid,
        )
        translation_seeds = [
            ("coarse_zero", (0.0, 0.0)),
            ("coarse_phase", phase_translation),
            ("coarse_centroid", centroid_translation),
        ]
        for search_stage, (translation_x, translation_y) in translation_seeds:
            clipped_x = float(
                np.clip(
                    translation_x,
                    -maximum_translation_px,
                    maximum_translation_px,
                )
            )
            clipped_y = float(
                np.clip(
                    translation_y,
                    -maximum_translation_px,
                    maximum_translation_px,
                )
            )
            candidates.append(
                objective.evaluate(
                    [float(angle), clipped_x, clipped_y],
                    search_stage=search_stage,
                )
            )
    return candidates


def _refine_candidate(
    objective: _PartialOverlapObjective,
    candidate: _Candidate,
    *,
    config: PartialOverlapRigidConfig,
    maximum_translation_px: float,
) -> _Candidate:
    radius = float(config.angle_search_radius_degrees)
    result = minimize(
        lambda parameters: -objective.evaluate(parameters).score,
        x0=np.array(
            [
                candidate.angle_degrees,
                candidate.translation_x_px,
                candidate.translation_y_px,
            ],
            dtype=np.float64,
        ),
        method="Powell",
        bounds=[
            (-radius, radius),
            (-maximum_translation_px, maximum_translation_px),
            (-maximum_translation_px, maximum_translation_px),
        ],
        options={
            "maxiter": int(config.optimizer_max_iterations),
            "xtol": 0.05,
            "ftol": 1.0e-4,
        },
    )
    return objective.evaluate(result.x, search_stage="refined")


def _phase_translation(
    fixed_density: np.ndarray,
    moving_density: np.ndarray,
    fixed_tissue: np.ndarray,
    moving_tissue: np.ndarray,
) -> tuple[float, float]:
    fixed_values = np.asarray(fixed_density, dtype=np.float32) * fixed_tissue
    moving_values = np.asarray(moving_density, dtype=np.float32) * moving_tissue
    if not np.any(fixed_values) or not np.any(moving_values):
        return 0.0, 0.0
    shift_rc, _, _ = phase_cross_correlation(
        fixed_values,
        moving_values,
        upsample_factor=1,
        normalization=None,
    )
    if not np.isfinite(shift_rc).all():
        return 0.0, 0.0
    return float(shift_rc[1]), float(shift_rc[0])


def _centroid_translation(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> tuple[float, float]:
    fixed_rows, fixed_cols = np.nonzero(fixed_mask)
    moving_rows, moving_cols = np.nonzero(moving_mask)
    if len(fixed_rows) == 0 or len(moving_rows) == 0:
        return 0.0, 0.0
    return (
        float(np.mean(fixed_cols) - np.mean(moving_cols)),
        float(np.mean(fixed_rows) - np.mean(moving_rows)),
    )


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8))
    return np.asarray((binary > 0) & (eroded == 0), dtype=bool)


def _trimmed_mean(values: np.ndarray, retained_fraction: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 8:
        return float("inf")
    retained_count = min(
        finite.size,
        max(8, int(np.ceil(float(retained_fraction) * finite.size))),
    )
    retained = np.partition(finite, retained_count - 1)[:retained_count]
    return float(np.mean(retained))


def _masked_correlation(
    fixed: np.ndarray,
    moving: np.ndarray,
    mask: np.ndarray,
) -> float:
    if int(np.asarray(mask).sum()) < 64:
        return float("nan")
    fixed_values = np.asarray(fixed, dtype=np.float64)[mask]
    moving_values = np.asarray(moving, dtype=np.float64)[mask]
    if float(np.std(fixed_values)) <= 1.0e-8:
        return float("nan")
    if float(np.std(moving_values)) <= 1.0e-8:
        return float("nan")
    return float(np.corrcoef(fixed_values, moving_values)[0, 1])


def _safe_fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _best_distinct(
    candidates: list[_Candidate],
    *,
    count: int,
    angle_separation: float,
) -> list[_Candidate]:
    selected: list[_Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(
            abs(candidate.angle_degrees - current.angle_degrees)
            < float(angle_separation)
            and np.hypot(
                candidate.translation_x_px - current.translation_x_px,
                candidate.translation_y_px - current.translation_y_px,
            )
            < 2.0
            for current in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= int(count):
            break
    return selected


def _candidate_summary(
    candidate: _Candidate,
    *,
    reduction_scale: float,
    pixel_size_um: float,
) -> dict[str, Any]:
    pixels_to_um = float(pixel_size_um) / float(reduction_scale)
    return {
        "residual_angle_degrees": float(candidate.angle_degrees),
        "translation_x_px_reduced": float(candidate.translation_x_px),
        "translation_y_px_reduced": float(candidate.translation_y_px),
        "translation_x_um": float(candidate.translation_x_px * pixels_to_um),
        "translation_y_um": float(candidate.translation_y_px * pixels_to_um),
        **candidate.metrics,
    }


_COARSE_SEED_STYLES: tuple[tuple[str, str, str], ...] = (
    ("coarse_zero", "zero-translation seed", "o"),
    ("coarse_phase", "phase-correlation seed", "s"),
    ("coarse_centroid", "centroid seed", "^"),
)


def _local_objective_slice(
    objective: _PartialOverlapObjective,
    *,
    selected: _Candidate,
    pixels_to_um: float,
    half_width_um: float,
    grid_size: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an x/y score slice around the selected candidate in physical units."""
    if not np.isfinite(pixels_to_um) or pixels_to_um <= 0:
        raise ValueError("pixels_to_um must be positive")
    if not np.isfinite(half_width_um) or half_width_um <= 0:
        raise ValueError("half_width_um must be positive")
    if int(grid_size) < 3:
        raise ValueError("grid_size must be at least 3")

    half_width_px = float(half_width_um) / float(pixels_to_um)
    x_px = np.linspace(
        selected.translation_x_px - half_width_px,
        selected.translation_x_px + half_width_px,
        int(grid_size),
    )
    y_px = np.linspace(
        selected.translation_y_px - half_width_px,
        selected.translation_y_px + half_width_px,
        int(grid_size),
    )
    scores = np.empty((len(y_px), len(x_px)), dtype=np.float64)
    for row, translation_y_px in enumerate(y_px):
        for column, translation_x_px in enumerate(x_px):
            scores[row, column] = objective.evaluate(
                [
                    selected.angle_degrees,
                    translation_x_px,
                    translation_y_px,
                ]
            ).score
    return (
        np.asarray(x_px * float(pixels_to_um), dtype=np.float64),
        np.asarray(y_px * float(pixels_to_um), dtype=np.float64),
        scores,
    )


def _plot_objective_diagnostics(
    output_path: Path,
    *,
    objective: _PartialOverlapObjective,
    coarse_candidates: list[_Candidate],
    refined_candidates: list[_Candidate],
    ranked_candidates: list[_Candidate],
    baseline: _Candidate,
    selected: _Candidate,
    pixels_to_um: float,
) -> None:
    """Plot the sampled search and a true local objective slice."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))

    angles = sorted({candidate.angle_degrees for candidate in coarse_candidates})
    if angles:
        best_by_angle = [
            max(
                candidate.score
                for candidate in coarse_candidates
                if candidate.angle_degrees == angle
            )
            for angle in angles
        ]
        axes[0].plot(
            angles,
            best_by_angle,
            color="0.25",
            linewidth=1.5,
            label="best coarse seed",
            zorder=2,
        )
    for search_stage, label, marker in _COARSE_SEED_STYLES:
        stage_candidates = sorted(
            (
                candidate
                for candidate in coarse_candidates
                if candidate.search_stage == search_stage
            ),
            key=lambda candidate: candidate.angle_degrees,
        )
        if not stage_candidates:
            continue
        axes[0].plot(
            [candidate.angle_degrees for candidate in stage_candidates],
            [candidate.score for candidate in stage_candidates],
            marker=marker,
            markersize=3.5,
            linewidth=0.8,
            alpha=0.75,
            label=label,
        )
    if refined_candidates:
        axes[0].scatter(
            [candidate.angle_degrees for candidate in refined_candidates],
            [candidate.score for candidate in refined_candidates],
            marker="D",
            s=34,
            facecolors="white",
            edgecolors="black",
            linewidths=0.9,
            label="refined candidates",
            zorder=4,
        )
    axes[0].axhline(
        baseline.score,
        color="0.5",
        linestyle="--",
        linewidth=1.0,
        label="baseline",
    )
    axes[0].scatter(
        [selected.angle_degrees],
        [selected.score],
        marker="*",
        s=130,
        color="tab:red",
        edgecolors="white",
        linewidths=0.6,
        label="selected",
        zorder=6,
    )
    axes[0].set_title("Rotation search samples")
    axes[0].set_xlabel("residual rotation (degrees)")
    axes[0].set_ylabel("robust score")
    axes[0].legend(fontsize=7.5)

    plotted_candidates = [*coarse_candidates, *refined_candidates]
    finite_scores = np.asarray(
        [candidate.score for candidate in plotted_candidates],
        dtype=np.float64,
    )
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size:
        score_min = float(np.min(finite_scores))
        score_max = float(np.max(finite_scores))
    else:
        score_min, score_max = 0.0, 1.0
    if np.isclose(score_min, score_max):
        score_min -= 0.5
        score_max += 0.5

    score_scatter = None
    for search_stage, label, marker in _COARSE_SEED_STYLES:
        stage_candidates = [
            candidate
            for candidate in coarse_candidates
            if candidate.search_stage == search_stage
        ]
        if not stage_candidates:
            continue
        score_scatter = axes[1].scatter(
            [
                candidate.translation_x_px * pixels_to_um
                for candidate in stage_candidates
            ],
            [
                candidate.translation_y_px * pixels_to_um
                for candidate in stage_candidates
            ],
            c=[candidate.score for candidate in stage_candidates],
            marker=marker,
            s=25,
            cmap="viridis",
            vmin=score_min,
            vmax=score_max,
            label=label,
            alpha=0.85,
        )
    if refined_candidates:
        score_scatter = axes[1].scatter(
            [
                candidate.translation_x_px * pixels_to_um
                for candidate in refined_candidates
            ],
            [
                candidate.translation_y_px * pixels_to_um
                for candidate in refined_candidates
            ],
            c=[candidate.score for candidate in refined_candidates],
            marker="D",
            s=48,
            cmap="viridis",
            vmin=score_min,
            vmax=score_max,
            edgecolors="black",
            linewidths=0.8,
            label="refined candidates",
            zorder=4,
        )
    axes[1].scatter(
        [selected.translation_x_px * pixels_to_um],
        [selected.translation_y_px * pixels_to_um],
        marker="*",
        s=150,
        color="tab:red",
        edgecolors="white",
        linewidths=0.7,
        label="selected",
        zorder=6,
    )
    axes[1].set_title("Sampled translations (not a dense landscape)")
    axes[1].set_xlabel("translation x (µm)")
    axes[1].set_ylabel("translation y (µm; + down)")
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].legend(fontsize=7.5)
    if score_scatter is not None:
        fig.colorbar(score_scatter, ax=axes[1], label="robust score")

    local_half_width_um = max(
        150.0,
        min(500.0, 6.0 * float(pixels_to_um)),
    )
    local_x_um, local_y_um, local_scores = _local_objective_slice(
        objective,
        selected=selected,
        pixels_to_um=pixels_to_um,
        half_width_um=local_half_width_um,
    )
    local_image = axes[2].imshow(
        local_scores,
        extent=(
            float(local_x_um[0]),
            float(local_x_um[-1]),
            float(local_y_um[-1]),
            float(local_y_um[0]),
        ),
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        cmap="viridis",
    )
    for index, candidate in enumerate(ranked_candidates[:5], start=1):
        if abs(candidate.angle_degrees - selected.angle_degrees) > 0.1:
            continue
        candidate_x_um = candidate.translation_x_px * pixels_to_um
        candidate_y_um = candidate.translation_y_px * pixels_to_um
        if not (
            local_x_um[0] <= candidate_x_um <= local_x_um[-1]
            and local_y_um[0] <= candidate_y_um <= local_y_um[-1]
        ):
            continue
        axes[2].annotate(
            str(index),
            (candidate_x_um, candidate_y_um),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
        )
    axes[2].scatter(
        [selected.translation_x_px * pixels_to_um],
        [selected.translation_y_px * pixels_to_um],
        marker="*",
        s=150,
        color="tab:red",
        edgecolors="white",
        linewidths=0.7,
        label="selected",
        zorder=6,
    )
    axes[2].set_title(
        f"Local x/y objective slice\nrotation fixed at {selected.angle_degrees:+.2f}°"
    )
    axes[2].set_xlabel("translation x (µm)")
    axes[2].set_ylabel("translation y (µm; + down)")
    axes[2].legend(fontsize=7.5)
    fig.colorbar(local_image, ax=axes[2], label="robust score")

    fig.tight_layout()
    save_figure(fig, output_path, dpi=160)
    plt.close(fig)


def _write_qc(
    output_dir: Path,
    *,
    fixed_full: np.ndarray,
    moving_full: np.ndarray,
    initial_matrix: np.ndarray,
    selected_matrix: np.ndarray,
    fixed_small: np.ndarray,
    moving_pre_small: np.ndarray,
    objective: _PartialOverlapObjective,
    candidates: list[_Candidate],
    coarse_candidates: list[_Candidate],
    refined_candidates: list[_Candidate],
    baseline: _Candidate,
    selected: _Candidate,
    pixels_to_um: float,
    metrics: dict[str, Any],
) -> None:
    from merxen.alignment.qc import plot_registration_overlay

    output_dir.mkdir(parents=True, exist_ok=True)
    initial_image = warp_image(
        moving_full,
        initial_matrix,
        output_shape_rc=fixed_full.shape,
    )
    selected_image = warp_image(
        moving_full,
        selected_matrix,
        output_shape_rc=fixed_full.shape,
    )
    plot_registration_overlay(
        fixed_full,
        initial_image,
        output_dir / "partial_overlap_before.png",
        title="Before partial-overlap rigid refinement",
    )
    plot_registration_overlay(
        fixed_full,
        selected_image,
        output_dir / "partial_overlap_selected.png",
        title="Selected partial-overlap rigid refinement",
    )

    count = min(6, len(candidates))
    columns = 3
    rows = max(1, int(np.ceil(count / columns)))
    output_path = prepare_plot_output(output_dir / "partial_overlap_candidates.png")
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for index, axis in enumerate(axes_array):
        if index >= count:
            axis.axis("off")
            continue
        candidate = candidates[index]
        moving_candidate = warp_image(
            moving_pre_small,
            objective.matrix(
                [
                    candidate.angle_degrees,
                    candidate.translation_x_px,
                    candidate.translation_y_px,
                ]
            ),
            output_shape_rc=fixed_small.shape,
        )
        rgb = np.zeros((*fixed_small.shape, 3), dtype=np.uint8)
        rgb[..., 0] = moving_candidate
        rgb[..., 1] = fixed_small
        rgb[..., 2] = moving_candidate
        axis.imshow(rgb)
        axis.set_title(
            f"#{index + 1}: {candidate.angle_degrees:+.2f}°, "
            f"dx {candidate.translation_x_px * pixels_to_um:+.0f} µm, "
            f"dy {candidate.translation_y_px * pixels_to_um:+.0f} µm\n"
            f"score {candidate.score:.3f}"
        )
        axis.axis("off")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=160)
    plt.close(fig)

    objective_path = prepare_plot_output(output_dir / "partial_overlap_objective.png")
    _plot_objective_diagnostics(
        objective_path,
        objective=objective,
        coarse_candidates=coarse_candidates,
        refined_candidates=refined_candidates,
        ranked_candidates=candidates,
        baseline=baseline,
        selected=selected,
        pixels_to_um=pixels_to_um,
    )

    (output_dir / "partial_overlap_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True)
    )
