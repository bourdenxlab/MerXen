"""DAPI-only registration QC, selection, and pipeline-stage collation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merxen.alignment.bundle import DisplacementField
from merxen.alignment.orientation import (
    masked_density_correlation,
    masked_normalized_mutual_information,
    tissue_dice,
)
from merxen.config import AlignmentQCConfig, AlignmentQCThresholds
from merxen.plotting import prepare_plot_output, save_figure


def run_alignment_qc(config: AlignmentQCConfig) -> dict[str, Path]:
    """Collate the alignment-stage DAPI QC without using expression features."""
    cfg = config
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.transform_json_path is None or not Path(cfg.transform_json_path).exists():
        raise FileNotFoundError(
            "DAPI alignment QC requires the alignment transform JSON"
        )
    transform_path = Path(cfg.transform_json_path)
    payload = json.loads(transform_path.read_text())
    metadata = dict(payload.get("metadata", {}))
    backend = str(metadata.get("backend", "legacy_spateo"))
    if backend == "legacy_spateo":
        from merxen.alignment.legacy_qc import run_alignment_qc as run_legacy_qc

        return run_legacy_qc(config)

    metrics = dict(metadata.get("qc", {}))
    metrics["pair_id"] = cfg.pair_id
    metrics["backend"] = backend
    metrics["transform_json_path"] = str(transform_path)

    metrics_json = cfg.output_dir / f"{cfg.pair_id}_alignment_qc.json"
    metrics_json.write_text(json.dumps(_jsonable(metrics), indent=2))
    metrics_csv = cfg.output_dir / f"{cfg.pair_id}_alignment_qc_metrics.csv"
    pd.json_normalize(metrics, sep=".").to_csv(metrics_csv, index=False)

    overlay_png = cfg.output_dir / f"{cfg.pair_id}_alignment_overlay.png"
    preview = metadata.get("registered_preview_path")
    preview_path = (
        None
        if preview is None
        else _resolve_alignment_artifact(transform_path, Path(preview))
    )
    if preview_path is not None and preview_path.exists():
        shutil.copy2(preview_path, overlay_png)
        preview_pdf = preview_path.with_suffix(".pdf")
        if preview_pdf.exists():
            shutil.copy2(preview_pdf, overlay_png.with_suffix(".pdf"))
    else:
        _write_status_plot(
            overlay_png,
            pair_id=cfg.pair_id,
            status=str(metadata.get("status", "unknown")),
        )
    return {
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "overlay_png": overlay_png,
    }


def compute_grid_alignment_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call the legacy expression-grid QC for source compatibility."""
    from merxen.alignment.legacy_qc import compute_grid_alignment_metrics as legacy

    return legacy(*args, **kwargs)


def plot_alignment_overlay(*args: Any, **kwargs: Any) -> None:
    """Call the legacy centroid overlay for source compatibility."""
    from merxen.alignment.legacy_qc import plot_alignment_overlay as legacy

    legacy(*args, **kwargs)


def compute_dapi_metrics(
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    *,
    moving_feature_xy: np.ndarray | None = None,
    fixed_feature_xy: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute DAPI morphology QC within valid overlapping tissue."""
    fixed = np.asarray(fixed_image)
    moving = np.asarray(moving_image)
    fmask = np.asarray(fixed_mask) > 0
    mmask = np.asarray(moving_mask) > 0
    if (
        fixed.shape != moving.shape
        or fixed.shape != fmask.shape
        or fixed.shape != mmask.shape
    ):
        raise ValueError("DAPI QC images and masks must share one shape")
    overlap = fmask & mmask
    union = fmask | mmask
    metrics: dict[str, float | int] = {
        "tissue_dice": tissue_dice(fmask, mmask),
        "tissue_iou": (
            float(overlap.sum() / union.sum()) if int(union.sum()) > 0 else float("nan")
        ),
        "normalized_mutual_information": masked_normalized_mutual_information(
            fixed,
            moving,
            fmask,
            mmask,
        ),
        "density_correlation": masked_density_correlation(
            fixed,
            moving,
            fmask,
            mmask,
        ),
        "fixed_overlap_fraction": (
            float(overlap.sum() / fmask.sum()) if int(fmask.sum()) > 0 else float("nan")
        ),
        "moving_overlap_fraction": (
            float(overlap.sum() / mmask.sum()) if int(mmask.sum()) > 0 else float("nan")
        ),
    }
    if moving_feature_xy is None or fixed_feature_xy is None:
        metrics.update(
            {
                "feature_inliers": 0,
                "feature_residual_median_px": float("nan"),
                "feature_residual_p90_px": float("nan"),
                "feature_inlier_coverage": 0.0,
            }
        )
        return metrics

    moving_xy = np.asarray(moving_feature_xy, dtype=np.float64)
    fixed_xy = np.asarray(fixed_feature_xy, dtype=np.float64)
    if (
        moving_xy.shape != fixed_xy.shape
        or moving_xy.ndim != 2
        or moving_xy.shape[1] != 2
    ):
        raise ValueError("Feature coordinate arrays must have matching (n, 2) shapes")
    residual = np.linalg.norm(moving_xy - fixed_xy, axis=1)
    metrics.update(
        {
            "feature_inliers": int(len(residual)),
            "feature_residual_median_px": (
                float(np.median(residual)) if len(residual) else float("nan")
            ),
            "feature_residual_p90_px": (
                float(np.percentile(residual, 90)) if len(residual) else float("nan")
            ),
            "feature_inlier_coverage": _feature_coverage(
                fixed_xy,
                fixed_mask=fmask,
            ),
        }
    )
    return metrics


def affine_diagnostics(matrix: Any) -> dict[str, float]:
    """Decompose a 2D affine into interpretable rotation/scale/shear metrics."""
    affine = np.asarray(matrix, dtype=np.float64)
    if affine.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine matrix, got {affine.shape}")
    linear = affine[:2, :2]
    determinant = float(np.linalg.det(linear))
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale_x = float(np.linalg.norm(linear[:, 0]))
    normalized_x = linear[:, 0] / max(scale_x, np.finfo(float).eps)
    shear_projection = float(np.dot(normalized_x, linear[:, 1]))
    orthogonal_y = linear[:, 1] - shear_projection * normalized_x
    scale_y = float(np.linalg.norm(orthogonal_y))
    shear = shear_projection / max(scale_y, np.finfo(float).eps)
    rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    return {
        "rotation_degrees": rotation,
        "translation_x": float(affine[0, 2]),
        "translation_y": float(affine[1, 2]),
        "scale_x": scale_x,
        "scale_y": scale_y,
        "shear": shear,
        "determinant": determinant,
        "minimum_singular_value": float(np.min(singular_values)),
        "maximum_singular_value": float(np.max(singular_values)),
    }


def global_qc_passes(
    image_metrics: dict[str, Any],
    affine_metrics: dict[str, Any],
    *,
    thresholds: AlignmentQCThresholds,
) -> tuple[bool, list[str]]:
    """Validate global tissue agreement and affine plausibility."""
    reasons: list[str] = []
    _minimum_check(
        image_metrics,
        "tissue_dice",
        thresholds.minimum_global_dice,
        reasons,
    )
    _minimum_check(
        image_metrics,
        "normalized_mutual_information",
        thresholds.minimum_global_mutual_information,
        reasons,
    )
    if int(image_metrics.get("feature_inliers", 0)) < int(
        thresholds.minimum_global_inliers
    ):
        reasons.append("too_few_global_feature_inliers")
    if float(image_metrics.get("feature_inlier_coverage", 0.0)) < float(
        thresholds.minimum_inlier_coverage
    ):
        reasons.append("insufficient_global_feature_coverage")
    determinant = float(affine_metrics["determinant"])
    if determinant < float(thresholds.affine_minimum_determinant):
        reasons.append("affine_determinant_too_small")
    if determinant > float(thresholds.affine_maximum_determinant):
        reasons.append("affine_determinant_too_large")
    minimum_sv = float(affine_metrics["minimum_singular_value"])
    maximum_sv = float(affine_metrics["maximum_singular_value"])
    if minimum_sv < float(thresholds.affine_minimum_singular_value):
        reasons.append("affine_singular_value_too_small")
    if maximum_sv > float(thresholds.affine_maximum_singular_value):
        reasons.append("affine_singular_value_too_large")
    if abs(float(affine_metrics["shear"])) > float(thresholds.affine_maximum_shear):
        reasons.append("affine_shear_too_large")
    return len(reasons) == 0, reasons


def rigid_qc_passes(
    affine_metrics: dict[str, Any],
    *,
    thresholds: AlignmentQCThresholds,
) -> tuple[bool, list[str]]:
    """Require a global refinement to contain rotation and translation only."""
    reasons: list[str] = []
    determinant_deviation = abs(float(affine_metrics["determinant"]) - 1.0)
    if determinant_deviation > float(thresholds.rigid_maximum_determinant_deviation):
        reasons.append("rigid_determinant_deviates_from_one")
    singular_value_deviation = max(
        abs(float(affine_metrics["minimum_singular_value"]) - 1.0),
        abs(float(affine_metrics["maximum_singular_value"]) - 1.0),
    )
    if singular_value_deviation > float(
        thresholds.rigid_maximum_singular_value_deviation
    ):
        reasons.append("rigid_scale_detected")
    if abs(float(affine_metrics["shear"])) > float(thresholds.rigid_maximum_shear):
        reasons.append("rigid_shear_detected")
    return len(reasons) == 0, reasons


def morphology_supported_global_qc_passes(
    image_metrics: dict[str, Any],
    affine_metrics: dict[str, Any],
    *,
    preorientation_metrics: dict[str, Any],
    thresholds: AlignmentQCThresholds,
    trusted_coordinate_metadata: bool,
    reflection_selected: bool,
    authoritative_preorientation_locked: bool = False,
) -> tuple[bool, list[str]]:
    """Accept cross-platform DAPI by strong morphology under strict safeguards.

    Adjacent-section MERSCOPE/Xenium DAPI can have little pixelwise mutual
    information and few exact cellular feature matches. This fallback is only
    available when the physical frames came from authoritative metadata and
    either an explicitly searched reflection was selected or the independently
    accepted pre-orientation was locked as the authoritative global transform.
    It requires substantially stronger tissue agreement and a near-identity
    VALIS refinement.
    """
    reasons: list[str] = []
    if not bool(thresholds.morphology_fallback_enabled):
        reasons.append("morphology_fallback_disabled")
    if not trusted_coordinate_metadata:
        reasons.append("untrusted_coordinate_metadata")
    if not reflection_selected and not authoritative_preorientation_locked:
        reasons.append("reflection_not_selected")
    _minimum_check(
        image_metrics,
        "tissue_dice",
        thresholds.morphology_minimum_global_dice,
        reasons,
    )
    _minimum_check(
        image_metrics,
        "tissue_iou",
        thresholds.morphology_minimum_global_iou,
        reasons,
    )
    _minimum_check(
        image_metrics,
        "density_correlation",
        thresholds.morphology_minimum_density_correlation,
        reasons,
    )
    for key in ("fixed_overlap_fraction", "moving_overlap_fraction"):
        _minimum_check(
            image_metrics,
            key,
            thresholds.morphology_minimum_overlap_fraction,
            reasons,
        )

    pre_dice = float(preorientation_metrics.get("tissue_dice", np.nan))
    global_dice = float(image_metrics.get("tissue_dice", np.nan))
    if (
        not np.isfinite(pre_dice)
        or not np.isfinite(global_dice)
        or global_dice
        < pre_dice - float(thresholds.morphology_maximum_dice_degradation)
    ):
        reasons.append("global_dice_degraded_from_preorientation")

    minimum_sv = float(affine_metrics["minimum_singular_value"])
    maximum_sv = float(affine_metrics["maximum_singular_value"])
    if minimum_sv < float(thresholds.morphology_affine_minimum_singular_value):
        reasons.append("morphology_affine_singular_value_too_small")
    if maximum_sv > float(thresholds.morphology_affine_maximum_singular_value):
        reasons.append("morphology_affine_singular_value_too_large")
    if abs(float(affine_metrics["shear"])) > float(
        thresholds.morphology_affine_maximum_shear
    ):
        reasons.append("morphology_affine_shear_too_large")
    if float(affine_metrics["determinant"]) <= 0:
        # The searched reflection belongs in the pre-orientation matrix. VALIS
        # should only make a proper, near-identity refinement after that.
        reasons.append("unexpected_reflection_in_valis_refinement")
    return len(reasons) == 0, reasons


def displacement_diagnostics(
    field: DisplacementField,
    *,
    pixel_size_um: float,
) -> dict[str, float]:
    """Summarize displacement magnitude and Jacobian plausibility."""
    displacement = np.asarray(field.displacement_xy, dtype=np.float64)
    magnitude_um = np.linalg.norm(displacement, axis=-1) * float(pixel_size_um)
    x = np.asarray(field.x_coordinates, dtype=np.float64)
    y = np.asarray(field.y_coordinates, dtype=np.float64)
    du_dy, du_dx = np.gradient(displacement[..., 0], y, x)
    dv_dy, dv_dx = np.gradient(displacement[..., 1], y, x)
    jacobian = (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx
    return {
        "displacement_median_um": float(np.nanmedian(magnitude_um)),
        "displacement_p95_um": float(np.nanpercentile(magnitude_um, 95)),
        "displacement_maximum_um": float(np.nanmax(magnitude_um)),
        "jacobian_minimum": float(np.nanmin(jacobian)),
        "jacobian_p01": float(np.nanpercentile(jacobian, 1)),
        "jacobian_median": float(np.nanmedian(jacobian)),
        "jacobian_p99": float(np.nanpercentile(jacobian, 99)),
        "jacobian_maximum": float(np.nanmax(jacobian)),
        "jacobian_nonpositive_fraction": float(np.mean(jacobian <= 0.0)),
        "jacobian_extreme_fraction": float(
            np.mean((jacobian < 0.2) | (jacobian > 5.0))
        ),
    }


def select_non_rigid_result(
    global_metrics: dict[str, Any],
    non_rigid_metrics: dict[str, Any],
    deformation_metrics: dict[str, Any],
    *,
    thresholds: AlignmentQCThresholds,
) -> tuple[bool, list[str]]:
    """Select non-rigid only for meaningful, anatomically plausible improvement."""
    reasons: list[str] = []
    global_nmi = float(global_metrics.get("normalized_mutual_information", np.nan))
    non_rigid_nmi = float(
        non_rigid_metrics.get("normalized_mutual_information", np.nan)
    )
    improvement = non_rigid_nmi - global_nmi
    if not np.isfinite(improvement) or improvement < float(
        thresholds.non_rigid_minimum_nmi_improvement
    ):
        reasons.append("insufficient_non_rigid_nmi_improvement")

    global_density = float(global_metrics.get("density_correlation", np.nan))
    non_rigid_density = float(non_rigid_metrics.get("density_correlation", np.nan))
    if (
        np.isfinite(global_density)
        and np.isfinite(non_rigid_density)
        and non_rigid_density
        < global_density
        - float(thresholds.non_rigid_maximum_density_correlation_degradation)
    ):
        reasons.append("non_rigid_density_correlation_degraded")

    global_dice = float(global_metrics.get("tissue_dice", np.nan))
    non_rigid_dice = float(non_rigid_metrics.get("tissue_dice", np.nan))
    if (
        np.isfinite(global_dice)
        and np.isfinite(non_rigid_dice)
        and non_rigid_dice
        < global_dice - float(thresholds.non_rigid_maximum_tissue_dice_degradation)
    ):
        reasons.append("non_rigid_tissue_dice_degraded")

    global_robust_score = float(
        global_metrics.get("partial_overlap_robust_score", np.nan)
    )
    non_rigid_robust_score = float(
        non_rigid_metrics.get("partial_overlap_robust_score", np.nan)
    )
    if (
        np.isfinite(global_robust_score)
        and np.isfinite(non_rigid_robust_score)
        and non_rigid_robust_score
        < global_robust_score
        - float(thresholds.non_rigid_maximum_robust_score_degradation)
    ):
        reasons.append("non_rigid_robust_score_degraded")

    coherent_rotation = float(
        deformation_metrics.get("coherent_rotation_degrees", np.nan)
    )
    if np.isfinite(coherent_rotation) and abs(coherent_rotation) > float(
        thresholds.non_rigid_maximum_coherent_rotation_degrees
    ):
        reasons.append("non_rigid_coherent_rotation_too_large")
    coherent_translation = float(
        deformation_metrics.get("coherent_translation_magnitude_um", np.nan)
    )
    if np.isfinite(coherent_translation) and coherent_translation > float(
        thresholds.non_rigid_maximum_coherent_translation_um
    ):
        reasons.append("non_rigid_coherent_translation_too_large")

    if float(deformation_metrics["displacement_p95_um"]) > float(
        thresholds.non_rigid_maximum_p95_displacement_um
    ):
        reasons.append("non_rigid_displacement_too_large")
    if float(deformation_metrics["jacobian_nonpositive_fraction"]) > float(
        thresholds.non_rigid_maximum_nonpositive_jacobian_fraction
    ):
        reasons.append("non_rigid_folding")
    if float(deformation_metrics["jacobian_p01"]) < float(
        thresholds.non_rigid_minimum_jacobian
    ):
        reasons.append("non_rigid_extreme_contraction")
    if float(deformation_metrics["jacobian_p99"]) > float(
        thresholds.non_rigid_maximum_jacobian
    ):
        reasons.append("non_rigid_extreme_expansion")
    return len(reasons) == 0, reasons


def plot_registration_overlay(
    fixed_image: Any,
    moving_image: Any,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save a magenta/green DAPI overlay as PNG and PDF."""
    fixed = _normalize_uint8(fixed_image)
    moving = _normalize_uint8(moving_image)
    rgb = np.zeros((*fixed.shape, 3), dtype=np.uint8)
    rgb[..., 0] = moving
    rgb[..., 1] = fixed
    rgb[..., 2] = moving
    output_path = prepare_plot_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def plot_checkerboard(
    fixed_image: Any,
    moving_image: Any,
    output_path: Path,
    *,
    title: str,
    tile_size: int = 64,
) -> None:
    """Save a fixed/moving checkerboard comparison."""
    fixed = _normalize_uint8(fixed_image)
    moving = _normalize_uint8(moving_image)
    rows, cols = np.indices(fixed.shape)
    choose_moving = ((rows // tile_size) + (cols // tile_size)) % 2 == 1
    checker = fixed.copy()
    checker[choose_moving] = moving[choose_moving]
    output_path = prepare_plot_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(checker, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def plot_mask_overlap(
    fixed_mask: Any,
    moving_mask: Any,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save tissue-mask fixed/moving/overlap colors."""
    fixed = np.asarray(fixed_mask) > 0
    moving = np.asarray(moving_mask) > 0
    rgb = np.zeros((*fixed.shape, 3), dtype=np.uint8)
    rgb[..., 0] = moving.astype(np.uint8) * 255
    rgb[..., 1] = fixed.astype(np.uint8) * 255
    rgb[..., 2] = (fixed & moving).astype(np.uint8) * 128
    output_path = prepare_plot_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def plot_feature_matches(
    fixed_image: Any,
    moving_image: Any,
    moving_xy: np.ndarray | None,
    fixed_xy: np.ndarray | None,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save spatially distributed feature inliers used for QC."""
    fixed = _normalize_uint8(fixed_image)
    moving = _normalize_uint8(moving_image)
    output_path = prepare_plot_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(moving, cmap="gray")
    axes[1].imshow(fixed, cmap="gray")
    if moving_xy is not None and fixed_xy is not None:
        axes[0].scatter(moving_xy[:, 0], moving_xy[:, 1], s=4, c="cyan")
        axes[1].scatter(fixed_xy[:, 0], fixed_xy[:, 1], s=4, c="yellow")
    axes[0].set_title("moving")
    axes[1].set_title("fixed")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def plot_deformation_qc(
    field: DisplacementField,
    output_dir: Path,
) -> dict[str, Path]:
    """Save displacement magnitude, Jacobian, and deformation-grid images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    displacement = np.asarray(field.displacement_xy, dtype=np.float64)
    magnitude = np.linalg.norm(displacement, axis=-1)
    x = np.asarray(field.x_coordinates, dtype=np.float64)
    y = np.asarray(field.y_coordinates, dtype=np.float64)
    du_dy, du_dx = np.gradient(displacement[..., 0], y, x)
    dv_dy, dv_dx = np.gradient(displacement[..., 1], y, x)
    jacobian = (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx

    outputs: dict[str, Path] = {}
    for name, values, cmap, title in (
        ("displacement_magnitude", magnitude, "magma", "displacement magnitude (px)"),
        ("jacobian_determinant", jacobian, "coolwarm", "Jacobian determinant"),
    ):
        path = prepare_plot_output(output_dir / f"{name}.png")
        fig, ax = plt.subplots(figsize=(8, 8))
        image = ax.imshow(
            values,
            cmap=cmap,
            extent=(x.min(), x.max(), y.max(), y.min()),
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        save_figure(fig, path, dpi=180)
        plt.close(fig)
        outputs[name] = path

    grid_path = prepare_plot_output(output_dir / "deformation_grid.png")
    xx, yy = np.meshgrid(x, y)
    warped_x = xx + displacement[..., 0]
    warped_y = yy + displacement[..., 1]
    stride = max(1, min(len(x), len(y)) // 30)
    fig, ax = plt.subplots(figsize=(8, 8))
    for row in range(0, len(y), stride):
        ax.plot(warped_x[row], warped_y[row], color="black", linewidth=0.5)
    for col in range(0, len(x), stride):
        ax.plot(warped_x[:, col], warped_y[:, col], color="black", linewidth=0.5)
    ax.set_title("non-rigid deformation grid")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, grid_path, dpi=180)
    plt.close(fig)
    outputs["deformation_grid"] = grid_path
    return outputs


def _feature_coverage(points_xy: np.ndarray, *, fixed_mask: np.ndarray) -> float:
    if len(points_xy) == 0:
        return 0.0
    rows, cols = np.nonzero(fixed_mask)
    if len(rows) == 0:
        return 0.0
    tissue_area = max(
        float((cols.max() - cols.min()) * (rows.max() - rows.min())),
        1.0,
    )
    point_area = max(float(np.ptp(points_xy[:, 0]) * np.ptp(points_xy[:, 1])), 0.0)
    return float(min(1.0, point_area / tissue_area))


def _minimum_check(
    metrics: dict[str, Any],
    key: str,
    threshold: float,
    reasons: list[str],
) -> None:
    value = float(metrics.get(key, np.nan))
    if not np.isfinite(value) or value < float(threshold):
        reasons.append(f"{key}_below_threshold")


def _normalize_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.asarray(
        np.clip((arr - low) * 255.0 / (high - low), 0, 255),
        dtype=np.uint8,
    )


def _write_status_plot(output_path: Path, *, pair_id: str, status: str) -> None:
    output_path = prepare_plot_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        f"{pair_id}\nVALIS status: {status}",
        ha="center",
        va="center",
    )
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=180)
    plt.close(fig)


def _resolve_alignment_artifact(transform_path: Path, artifact: Path) -> Path:
    if artifact.is_absolute():
        return artifact
    if artifact.exists():
        return artifact
    resolved_root = transform_path.resolve().parent
    parts = list(artifact.parts)
    if parts and parts[0] == "align_out":
        parts = parts[1:]
    return resolved_root.joinpath(*parts)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
