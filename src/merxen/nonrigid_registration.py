"""Robust dense optical-flow refinement for post-acquisition image registration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage.registration import optical_flow_tvl1

_HIGHPASS_SIGMA_UM = 25.0
_LOCAL_CORRELATION_SIGMA_UM = 15.0
_MIN_LOCAL_SUPPORT = 0.25
_MIN_SMOOTHED_SUPPORT = 0.02
_FULL_SMOOTHED_SUPPORT = 0.25
_JACOBIAN_REPAIR_ITERATIONS = 3
_LOCAL_JACOBIAN_REPAIR_ITERATIONS = 24
_MIN_RESIDUAL_DISPLACEMENT_UM = 0.25
_LOCAL_RESIDUAL_TILE_UM = 220.0
_LOCAL_RESIDUAL_MAX_CORRELATION = 0.85
_LOCAL_RESIDUAL_MIN_VALID_FRACTION = 0.4
_LOCAL_RESIDUAL_MIN_CORRELATION_IMPROVEMENT = 0.03

REJECTION_OUTSIDE_OVERLAP = np.uint16(1 << 0)
REJECTION_LOW_SUPPORT = np.uint16(1 << 1)
REJECTION_LOW_CORRELATION = np.uint16(1 << 2)
REJECTION_CYCLE_INCONSISTENT = np.uint16(1 << 3)
REJECTION_EXCESSIVE_DISPLACEMENT = np.uint16(1 << 4)
REJECTION_UNSAFE_JACOBIAN = np.uint16(1 << 5)


@dataclass(frozen=True)
class NonrigidFlowResult:
    """A guarded fixed-grid to moving-grid displacement field."""

    status: str
    reason: str | None
    flow_yx: np.ndarray
    confidence: np.ndarray
    residual_confidence: np.ndarray
    rejection_reasons: np.ndarray
    local_correlation: np.ndarray
    cycle_error_px: np.ndarray
    jacobian_determinant: np.ndarray
    warped: np.ndarray
    stats: dict[str, Any]


def estimate_nonrigid_flow(
    reference: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
    *,
    pixel_size_um: float,
    attachment: float,
    tightness: float,
    num_warp: int,
    num_iter: int,
    smoothing_sigma_um: float,
    max_displacement_um: float,
    max_cycle_error_um: float,
    min_local_correlation: float,
    min_jacobian: float,
    max_jacobian: float,
    min_confident_fraction: float,
    residual_passes: int = 1,
    min_residual_correlation_improvement: float = 0.02,
    min_residual_initial_confidence: float = 0.75,
) -> NonrigidFlowResult:
    """Estimate and guard a TV-L1 pull displacement field.

    The returned ``flow_yx`` satisfies ``moving[p + flow(p)] ~= reference[p]``.
    Regions without reliable bidirectional support retain zero displacement, so
    the caller falls back to its affine mapping there.
    """
    reference_array = np.asarray(reference, dtype=np.float32)
    moving_array = np.asarray(moving, dtype=np.float32)
    valid_array = np.asarray(valid, dtype=bool)
    if residual_passes < 0:
        raise ValueError("residual_passes must be non-negative")
    if min_residual_correlation_improvement < 0.0:
        raise ValueError("min_residual_correlation_improvement must be non-negative")
    if not 0.0 <= min_residual_initial_confidence <= 1.0:
        raise ValueError("min_residual_initial_confidence must be between 0 and 1")
    if (
        reference_array.ndim != 2
        or moving_array.shape != reference_array.shape
        or valid_array.shape != reference_array.shape
    ):
        raise ValueError(
            "Non-rigid registration requires equally shaped 2D reference, moving, "
            f"and valid arrays; got {reference_array.shape}, {moving_array.shape}, "
            f"and {valid_array.shape}."
        )
    if not valid_array.any():
        return _rejected_result(
            reference_array,
            moving_array,
            valid_array,
            reason="no overlapping tissue pixels",
            pixel_size_um=pixel_size_um,
        )

    reference_prepared = _prepare_image(
        reference_array,
        valid_array,
        pixel_size_um=pixel_size_um,
    )
    moving_prepared = _prepare_image(
        moving_array,
        valid_array,
        pixel_size_um=pixel_size_um,
    )
    reference_input = np.where(valid_array, reference_prepared, 0.0).astype(
        np.float32,
        copy=False,
    )
    moving_input = np.where(valid_array, moving_prepared, 0.0).astype(
        np.float32,
        copy=False,
    )

    flow_kwargs = {
        "attachment": float(attachment),
        "tightness": float(tightness),
        "num_warp": int(num_warp),
        "num_iter": int(num_iter),
        "tol": 1.0e-4,
        "prefilter": True,
        "dtype": np.float32,
    }
    forward = np.asarray(
        optical_flow_tvl1(reference_input, moving_input, **flow_kwargs),
        dtype=np.float32,
    )
    backward = np.asarray(
        optical_flow_tvl1(moving_input, reference_input, **flow_kwargs),
        dtype=np.float32,
    )

    raw_warped = warp_with_flow(moving_prepared, forward)
    warped_valid = (
        warp_with_flow(
            valid_array.astype(np.float32),
            forward,
            interpolation=cv2.INTER_NEAREST,
        )
        > 0.5
    )
    comparison_valid = valid_array & warped_valid
    backward_at_forward = np.stack(
        [warp_with_flow(component, forward) for component in backward],
        axis=0,
    )
    cycle_error = np.linalg.norm(forward + backward_at_forward, axis=0)
    correlation, local_support = _local_correlation(
        reference_prepared,
        raw_warped,
        comparison_valid,
        sigma_px=max(1.0, _LOCAL_CORRELATION_SIGMA_UM / float(pixel_size_um)),
    )
    raw_jacobian = flow_jacobian_determinant(forward)
    raw_magnitude = np.linalg.norm(forward, axis=0)
    max_displacement_px = float(max_displacement_um) / float(pixel_size_um)
    max_cycle_error_px = float(max_cycle_error_um) / float(pixel_size_um)
    enough_support = local_support >= _MIN_LOCAL_SUPPORT
    correlation_ok = np.isfinite(correlation) & (
        correlation >= float(min_local_correlation)
    )
    cycle_ok = np.isfinite(cycle_error) & (cycle_error <= max_cycle_error_px)
    displacement_ok = np.isfinite(raw_magnitude) & (
        raw_magnitude <= max_displacement_px
    )
    jacobian_ok = (
        np.isfinite(raw_jacobian)
        & (raw_jacobian >= float(min_jacobian))
        & (raw_jacobian <= float(max_jacobian))
    )
    confidence = (
        comparison_valid
        & enough_support
        & correlation_ok
        & cycle_ok
        & displacement_ok
        & jacobian_ok
    )
    rejection_reasons = np.zeros(valid_array.shape, dtype=np.uint16)
    rejection_reasons[~comparison_valid] |= REJECTION_OUTSIDE_OVERLAP
    rejection_reasons[comparison_valid & ~enough_support] |= REJECTION_LOW_SUPPORT
    rejection_reasons[comparison_valid & ~correlation_ok] |= REJECTION_LOW_CORRELATION
    rejection_reasons[comparison_valid & ~cycle_ok] |= REJECTION_CYCLE_INCONSISTENT
    rejection_reasons[comparison_valid & ~displacement_ok] |= (
        REJECTION_EXCESSIVE_DISPLACEMENT
    )
    rejection_reasons[comparison_valid & ~jacobian_ok] |= REJECTION_UNSAFE_JACOBIAN
    valid_count = int(valid_array.sum())
    confident_fraction = float(confidence.sum() / max(valid_count, 1))
    if confident_fraction < float(min_confident_fraction):
        return _rejected_result(
            reference_array,
            moving_array,
            valid_array,
            reason=(
                f"confident fraction {confident_fraction:.3f} < "
                f"{float(min_confident_fraction):.3f}"
            ),
            pixel_size_um=pixel_size_um,
            confidence=confidence,
            rejection_reasons=rejection_reasons,
            local_correlation=correlation,
            cycle_error_px=cycle_error,
            extra_stats={
                "raw_flow": _flow_stats(
                    forward,
                    raw_jacobian,
                    valid_array,
                    pixel_size_um,
                ),
                "confident_fraction": confident_fraction,
            },
        )

    correlation_scale = max(
        1.0 - float(min_local_correlation),
        float(np.finfo(np.float32).eps),
    )
    correlation_weight = np.clip(
        (correlation - float(min_local_correlation)) / correlation_scale,
        0.0,
        1.0,
    )
    cycle_weight = np.clip(
        1.0 - cycle_error / max(max_cycle_error_px, np.finfo(np.float32).eps),
        0.0,
        1.0,
    )
    weights = np.where(
        confidence,
        np.maximum(correlation_weight, 0.05) * np.maximum(cycle_weight, 0.05),
        0.0,
    ).astype(np.float32)
    smoothing_sigma_px = max(
        0.5,
        float(smoothing_sigma_um) / float(pixel_size_um),
    )
    guarded, supported = _weighted_smooth_flow(
        forward,
        weights,
        sigma_px=smoothing_sigma_px,
    )
    guarded = _cap_flow_magnitude(guarded, max_displacement_px)
    for _ in range(_JACOBIAN_REPAIR_ITERATIONS):
        jacobian = flow_jacobian_determinant(guarded)
        unsafe = supported & (
            (jacobian < float(min_jacobian))
            | (jacobian > float(max_jacobian))
            | ~np.isfinite(jacobian)
        )
        if not unsafe.any():
            break
        weights[binary_dilation(unsafe, iterations=1)] = 0.0
        guarded, supported = _weighted_smooth_flow(
            forward,
            weights,
            sigma_px=smoothing_sigma_px,
        )
        guarded = _cap_flow_magnitude(guarded, max_displacement_px)
    guarded, jacobian_repair = _repair_jacobian_locally(
        guarded,
        np.ones(valid_array.shape, dtype=bool),
        min_jacobian=float(min_jacobian),
        max_jacobian=float(max_jacobian),
        sigma_px=smoothing_sigma_px,
    )

    final_jacobian = flow_jacobian_determinant(guarded)
    final_warped = warp_with_flow(moving_prepared, guarded)
    before_correlation = _masked_correlation(
        reference_prepared,
        moving_prepared,
        valid_array,
    )
    after_correlation = _masked_correlation(
        reference_prepared,
        final_warped,
        valid_array,
    )
    if not np.isfinite(after_correlation) or after_correlation <= before_correlation:
        return _rejected_result(
            reference_array,
            moving_array,
            valid_array,
            reason=(
                f"guarded flow did not improve DAPI correlation "
                f"({before_correlation:.4f} to {after_correlation:.4f})"
            ),
            pixel_size_um=pixel_size_um,
            confidence=confidence,
            rejection_reasons=rejection_reasons,
            local_correlation=correlation,
            cycle_error_px=cycle_error,
            extra_stats={
                "raw_flow": _flow_stats(
                    forward,
                    raw_jacobian,
                    valid_array,
                    pixel_size_um,
                ),
                "candidate_flow": _flow_stats(
                    guarded,
                    final_jacobian,
                    valid_array,
                    pixel_size_um,
                ),
                "confident_fraction": confident_fraction,
                "pearson_before": before_correlation,
                "pearson_after": after_correlation,
            },
        )

    residual_confidence = np.zeros(valid_array.shape, dtype=bool)
    residual_stats: list[dict[str, Any]] = []
    local_residual_stats: dict[str, Any] = {"status": "disabled"}
    if residual_passes:
        (
            guarded,
            final_warped,
            final_jacobian,
            after_correlation,
            residual_confidence,
            residual_stats,
        ) = _apply_residual_passes(
            reference_array=reference_array,
            moving_array=moving_array,
            reference_prepared=reference_prepared,
            moving_prepared=moving_prepared,
            valid_array=valid_array,
            initial_confidence=confidence,
            initial_flow=guarded,
            initial_warped=final_warped,
            initial_jacobian=final_jacobian,
            initial_correlation=after_correlation,
            residual_passes=residual_passes,
            min_residual_correlation_improvement=(min_residual_correlation_improvement),
            min_residual_initial_confidence=min_residual_initial_confidence,
            pixel_size_um=pixel_size_um,
            attachment=attachment,
            tightness=tightness,
            num_warp=num_warp,
            num_iter=num_iter,
            smoothing_sigma_um=smoothing_sigma_um,
            max_displacement_um=max_displacement_um,
            max_cycle_error_um=max_cycle_error_um,
            min_local_correlation=min_local_correlation,
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
            min_confident_fraction=min_confident_fraction,
        )
        (
            guarded,
            final_warped,
            final_jacobian,
            after_correlation,
            local_residual_confidence,
            local_residual_stats,
        ) = _apply_local_residual_tiles(
            reference_array=reference_array,
            moving_array=moving_array,
            reference_prepared=reference_prepared,
            moving_prepared=moving_prepared,
            valid_array=valid_array,
            initial_flow=guarded,
            initial_warped=final_warped,
            initial_jacobian=final_jacobian,
            initial_correlation=after_correlation,
            min_correlation_improvement=max(
                _LOCAL_RESIDUAL_MIN_CORRELATION_IMPROVEMENT,
                float(min_residual_correlation_improvement),
            ),
            pixel_size_um=pixel_size_um,
            attachment=attachment,
            tightness=tightness,
            num_warp=num_warp,
            num_iter=num_iter,
            smoothing_sigma_um=smoothing_sigma_um,
            max_displacement_um=max_displacement_um,
            max_cycle_error_um=max_cycle_error_um,
            min_local_correlation=min_local_correlation,
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
            min_confident_fraction=min_confident_fraction,
        )
        residual_confidence |= local_residual_confidence
    final_local_correlation, _ = _local_correlation(
        reference_prepared,
        final_warped,
        valid_array,
        sigma_px=max(1.0, _LOCAL_CORRELATION_SIGMA_UM / float(pixel_size_um)),
    )
    return NonrigidFlowResult(
        status="applied",
        reason=None,
        flow_yx=guarded.astype(np.float32, copy=False),
        confidence=confidence,
        residual_confidence=residual_confidence,
        rejection_reasons=rejection_reasons,
        local_correlation=final_local_correlation.astype(np.float32, copy=False),
        cycle_error_px=cycle_error.astype(np.float32, copy=False),
        jacobian_determinant=final_jacobian.astype(np.float32, copy=False),
        warped=final_warped.astype(np.float32, copy=False),
        stats={
            "pixel_size_um": float(pixel_size_um),
            "shape_yx": list(reference_array.shape),
            "valid_fraction": float(valid_array.mean()),
            "confident_fraction": confident_fraction,
            "supported_fraction": float(
                np.sum(supported & valid_array) / max(valid_count, 1)
            ),
            "pearson_before": before_correlation,
            "pearson_after": after_correlation,
            "jacobian_repair": jacobian_repair,
            "residual_passes": residual_stats,
            "local_residual_refinement": local_residual_stats,
            "raw_flow": _flow_stats(
                forward,
                raw_jacobian,
                valid_array,
                pixel_size_um,
            ),
            "final_flow": _flow_stats(
                guarded,
                final_jacobian,
                valid_array,
                pixel_size_um,
            ),
        },
    )


def warp_with_flow(
    image: np.ndarray,
    flow_yx: np.ndarray,
    *,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Pull-warp a 2D array with a fixed-grid ``(y, x)`` displacement field."""
    array = np.asarray(image)
    flow = np.asarray(flow_yx, dtype=np.float32)
    if flow.shape != (2, *array.shape):
        raise ValueError(
            f"Flow shape {flow.shape} does not match image shape {array.shape}."
        )
    rows, columns = np.meshgrid(
        np.arange(array.shape[0], dtype=np.float32),
        np.arange(array.shape[1], dtype=np.float32),
        indexing="ij",
    )
    return cv2.remap(
        array,
        columns + flow[1],
        rows + flow[0],
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def flow_jacobian_determinant(flow_yx: np.ndarray) -> np.ndarray:
    """Return the Jacobian determinant of ``p -> p + flow(p)``."""
    flow = np.asarray(flow_yx, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"Expected flow shape (2, y, x), got {flow.shape}.")
    dv_dy, dv_dx = np.gradient(flow[0])
    du_dy, du_dx = np.gradient(flow[1])
    return np.asarray(
        (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx,
        dtype=np.float32,
    )


def _prepare_image(
    image: np.ndarray,
    valid: np.ndarray,
    *,
    pixel_size_um: float,
) -> np.ndarray:
    logged = np.log1p(np.maximum(image, 0.0)).astype(np.float32, copy=False)
    sigma_px = max(1.0, _HIGHPASS_SIGMA_UM / float(pixel_size_um))
    highpass = logged - gaussian_filter(logged, sigma_px)
    values = highpass[valid]
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.asarray(
        np.clip((highpass - low) / (high - low), 0.0, 1.0),
        dtype=np.float32,
    )


def _local_correlation(
    reference: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
    *,
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(valid, dtype=np.float32)
    support = gaussian_filter(weights, sigma_px)
    denominator = np.maximum(support, np.finfo(np.float32).eps)
    reference_mean = gaussian_filter(reference * weights, sigma_px) / denominator
    moving_mean = gaussian_filter(moving * weights, sigma_px) / denominator
    reference_second = gaussian_filter(reference * reference * weights, sigma_px)
    moving_second = gaussian_filter(moving * moving * weights, sigma_px)
    cross = gaussian_filter(reference * moving * weights, sigma_px)
    reference_variance = np.maximum(
        reference_second / denominator - reference_mean * reference_mean,
        0.0,
    )
    moving_variance = np.maximum(
        moving_second / denominator - moving_mean * moving_mean,
        0.0,
    )
    covariance = cross / denominator - reference_mean * moving_mean
    scale = np.sqrt(reference_variance * moving_variance)
    correlation = np.divide(
        covariance,
        scale,
        out=np.full(reference.shape, np.nan, dtype=np.float32),
        where=scale > np.finfo(np.float32).eps,
    )
    correlation[support < _MIN_LOCAL_SUPPORT] = np.nan
    return correlation, support


def _weighted_smooth_flow(
    flow_yx: np.ndarray,
    weights: np.ndarray,
    *,
    sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = gaussian_filter(weights, sigma_px)
    supported = denominator >= _MIN_SMOOTHED_SUPPORT
    taper = np.clip(
        (denominator - _MIN_SMOOTHED_SUPPORT)
        / (_FULL_SMOOTHED_SUPPORT - _MIN_SMOOTHED_SUPPORT),
        0.0,
        1.0,
    )
    result = np.zeros_like(flow_yx, dtype=np.float32)
    for axis in range(2):
        numerator = gaussian_filter(flow_yx[axis] * weights, sigma_px)
        np.divide(
            numerator,
            denominator,
            out=result[axis],
            where=supported,
        )
        result[axis] *= taper
    return result, supported


def _cap_flow_magnitude(flow_yx: np.ndarray, limit_px: float) -> np.ndarray:
    result = np.asarray(flow_yx, dtype=np.float32).copy()
    magnitude = np.linalg.norm(result, axis=0)
    scale = np.minimum(1.0, float(limit_px) / np.maximum(magnitude, 1.0e-6))
    result *= scale
    return result


def _compose_pull_flows(
    first_flow_yx: np.ndarray,
    residual_flow_yx: np.ndarray,
) -> np.ndarray:
    """Compose pull fields applied first and then residually.

    If ``first`` maps ``p`` to ``p + first(p)`` and ``residual`` aligns the
    already-warped image, the combined source coordinate is
    ``p + residual(p) + first(p + residual(p))``.
    """
    first = np.asarray(first_flow_yx, dtype=np.float32)
    residual = np.asarray(residual_flow_yx, dtype=np.float32)
    if first.shape != residual.shape:
        raise ValueError(
            f"Cannot compose flow shapes {first.shape} and {residual.shape}."
        )
    sampled_first = np.stack(
        [warp_with_flow(component, residual) for component in first],
        axis=0,
    )
    return np.asarray(residual + sampled_first, dtype=np.float32)


def _unsafe_jacobian_mask(
    flow_yx: np.ndarray,
    domain: np.ndarray,
    *,
    min_jacobian: float,
    max_jacobian: float,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = flow_jacobian_determinant(flow_yx)
    unsafe = np.asarray(domain, dtype=bool) & (
        ~np.isfinite(jacobian)
        | (jacobian < float(min_jacobian))
        | (jacobian > float(max_jacobian))
    )
    return jacobian, unsafe


def _repair_jacobian_locally(
    flow_yx: np.ndarray,
    domain: np.ndarray,
    *,
    min_jacobian: float,
    max_jacobian: float,
    sigma_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repair unsafe flow gradients locally before using a global fallback."""
    original = np.asarray(flow_yx, dtype=np.float32)
    result = original.copy()
    domain_array = np.asarray(domain, dtype=bool)
    if not domain_array.any():
        return np.zeros_like(result), {
            "initial_unsafe_fraction": 0.0,
            "final_unsafe_fraction": 0.0,
            "locally_changed_fraction": 0.0,
            "global_fallback_scale": 1.0,
        }

    _jacobian, initial_unsafe = _unsafe_jacobian_mask(
        result,
        domain_array,
        min_jacobian=min_jacobian,
        max_jacobian=max_jacobian,
    )
    repair_sigma = max(0.75, float(sigma_px) / 2.0)
    dilation_iterations = max(1, int(math.ceil(float(sigma_px))))
    for iteration in range(_LOCAL_JACOBIAN_REPAIR_ITERATIONS):
        _jacobian, unsafe = _unsafe_jacobian_mask(
            result,
            domain_array,
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
        )
        if not unsafe.any():
            break
        repair_region = binary_dilation(unsafe, iterations=dilation_iterations)
        blend = gaussian_filter(repair_region.astype(np.float32), repair_sigma)
        blend = np.clip(blend * 1.5, 0.0, 1.0)
        smoothed = np.stack(
            [gaussian_filter(component, repair_sigma) for component in result],
            axis=0,
        ).astype(np.float32, copy=False)
        result = result * (1.0 - blend) + smoothed * blend
        if iteration >= _LOCAL_JACOBIAN_REPAIR_ITERATIONS // 2:
            result *= 1.0 - 0.1 * blend

    _jacobian, final_unsafe = _unsafe_jacobian_mask(
        result,
        domain_array,
        min_jacobian=min_jacobian,
        max_jacobian=max_jacobian,
    )
    fallback_scale = 1.0
    if final_unsafe.any():
        before_norm = float(np.linalg.norm(result))
        result = _scale_to_jacobian_limits(
            result,
            domain_array,
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
        )
        after_norm = float(np.linalg.norm(result))
        fallback_scale = after_norm / before_norm if before_norm else 1.0
        _jacobian, final_unsafe = _unsafe_jacobian_mask(
            result,
            domain_array,
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
        )

    changed = np.any(np.abs(result - original) > 1.0e-4, axis=0)
    domain_count = max(int(domain_array.sum()), 1)
    return np.asarray(result, dtype=np.float32), {
        "initial_unsafe_fraction": float(initial_unsafe.sum() / domain_count),
        "final_unsafe_fraction": float(final_unsafe.sum() / domain_count),
        "locally_changed_fraction": float(
            (changed & domain_array).sum() / domain_count
        ),
        "global_fallback_scale": float(fallback_scale),
    }


def _apply_residual_passes(
    *,
    reference_array: np.ndarray,
    moving_array: np.ndarray,
    reference_prepared: np.ndarray,
    moving_prepared: np.ndarray,
    valid_array: np.ndarray,
    initial_confidence: np.ndarray,
    initial_flow: np.ndarray,
    initial_warped: np.ndarray,
    initial_jacobian: np.ndarray,
    initial_correlation: float,
    residual_passes: int,
    min_residual_correlation_improvement: float,
    min_residual_initial_confidence: float,
    pixel_size_um: float,
    attachment: float,
    tightness: float,
    num_warp: int,
    num_iter: int,
    smoothing_sigma_um: float,
    max_displacement_um: float,
    max_cycle_error_um: float,
    min_local_correlation: float,
    min_jacobian: float,
    max_jacobian: float,
    min_confident_fraction: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    list[dict[str, Any]],
]:
    current_flow = np.asarray(initial_flow, dtype=np.float32)
    current_warped = np.asarray(initial_warped, dtype=np.float32)
    current_jacobian = np.asarray(initial_jacobian, dtype=np.float32)
    current_correlation = float(initial_correlation)
    residual_confidence = np.zeros(valid_array.shape, dtype=bool)
    pass_stats: list[dict[str, Any]] = []
    smoothing_sigma_px = max(
        0.5,
        float(smoothing_sigma_um) / float(pixel_size_um),
    )
    local_sigma_px = max(
        1.0,
        _LOCAL_CORRELATION_SIGMA_UM / float(pixel_size_um),
    )
    deep_invalid = ~binary_dilation(
        valid_array,
        iterations=max(1, int(math.ceil(2.0 * smoothing_sigma_px))),
    )
    initial_confidence_density = gaussian_filter(
        np.asarray(initial_confidence, dtype=np.float32),
        smoothing_sigma_px,
    )

    for pass_index in range(int(residual_passes)):
        current_original = warp_with_flow(moving_array, current_flow)
        current_footprint = (
            warp_with_flow(
                valid_array.astype(np.float32),
                current_flow,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0.5
        )
        pass_valid = valid_array & current_footprint
        residual = estimate_nonrigid_flow(
            reference_array,
            current_original,
            pass_valid,
            pixel_size_um=pixel_size_um,
            attachment=attachment,
            tightness=tightness,
            num_warp=num_warp,
            num_iter=num_iter,
            smoothing_sigma_um=smoothing_sigma_um,
            max_displacement_um=max_displacement_um,
            max_cycle_error_um=max_cycle_error_um,
            min_local_correlation=min_local_correlation,
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
            min_confident_fraction=min_confident_fraction,
            residual_passes=0,
            min_residual_correlation_improvement=(min_residual_correlation_improvement),
            min_residual_initial_confidence=min_residual_initial_confidence,
        )
        stats: dict[str, Any] = {
            "pass": pass_index + 1,
            "status": residual.status,
            "reason": residual.reason,
        }
        if residual.status != "applied":
            pass_stats.append(stats)
            break

        pass_reference = _prepare_image(
            reference_array,
            pass_valid,
            pixel_size_um=pixel_size_um,
        )
        pass_moving = _prepare_image(
            current_original,
            pass_valid,
            pixel_size_um=pixel_size_um,
        )
        proposed = warp_with_flow(pass_moving, residual.flow_yx)
        before_local, _ = _local_correlation(
            pass_reference,
            pass_moving,
            pass_valid,
            sigma_px=local_sigma_px,
        )
        after_local, _ = _local_correlation(
            pass_reference,
            proposed,
            pass_valid,
            sigma_px=local_sigma_px,
        )
        improvement = after_local - before_local
        residual_magnitude_um = np.linalg.norm(residual.flow_yx, axis=0) * float(
            pixel_size_um
        )
        accepted = (
            residual.confidence
            & (initial_confidence_density >= float(min_residual_initial_confidence))
            & np.isfinite(improvement)
            & (improvement >= float(min_residual_correlation_improvement))
            & (residual_magnitude_um >= _MIN_RESIDUAL_DISPLACEMENT_UM)
        )
        accepted_count = int(accepted.sum())
        stats["candidate_confident_fraction"] = float(
            residual.confidence.sum() / max(int(pass_valid.sum()), 1)
        )
        stats["accepted_fraction"] = float(
            accepted_count / max(int(pass_valid.sum()), 1)
        )
        if not accepted_count:
            stats["status"] = "skipped_no_local_improvement"
            pass_stats.append(stats)
            break

        improvement_weight = np.clip(
            improvement / max(0.25, float(min_residual_correlation_improvement)),
            0.05,
            1.0,
        )
        weights = np.where(accepted, improvement_weight, 0.0).astype(np.float32)
        residual_update, update_supported = _weighted_smooth_flow(
            residual.flow_yx,
            weights,
            sigma_px=smoothing_sigma_px,
        )
        candidate_flow = _compose_pull_flows(current_flow, residual_update)
        candidate_flow, repair_stats = _repair_jacobian_locally(
            candidate_flow,
            np.ones(valid_array.shape, dtype=bool),
            min_jacobian=min_jacobian,
            max_jacobian=max_jacobian,
            sigma_px=smoothing_sigma_px,
        )
        candidate_flow[:, deep_invalid] = 0.0
        candidate_warped = warp_with_flow(moving_prepared, candidate_flow)
        candidate_correlation = _masked_correlation(
            reference_prepared,
            candidate_warped,
            valid_array,
        )

        current_local, _ = _local_correlation(
            reference_prepared,
            current_warped,
            valid_array,
            sigma_px=local_sigma_px,
        )
        candidate_local, _ = _local_correlation(
            reference_prepared,
            candidate_warped,
            valid_array,
            sigma_px=local_sigma_px,
        )
        regressed = (
            update_supported
            & np.isfinite(current_local)
            & np.isfinite(candidate_local)
            & (
                candidate_local
                < current_local - float(min_residual_correlation_improvement)
            )
        )
        if regressed.any():
            weights[binary_dilation(regressed, iterations=1)] = 0.0
            residual_update, update_supported = _weighted_smooth_flow(
                residual.flow_yx,
                weights,
                sigma_px=smoothing_sigma_px,
            )
            candidate_flow = _compose_pull_flows(current_flow, residual_update)
            candidate_flow, repair_stats = _repair_jacobian_locally(
                candidate_flow,
                np.ones(valid_array.shape, dtype=bool),
                min_jacobian=min_jacobian,
                max_jacobian=max_jacobian,
                sigma_px=smoothing_sigma_px,
            )
            candidate_flow[:, deep_invalid] = 0.0
            candidate_warped = warp_with_flow(moving_prepared, candidate_flow)
            candidate_correlation = _masked_correlation(
                reference_prepared,
                candidate_warped,
                valid_array,
            )

        stats.update(
            {
                "pearson_before": current_correlation,
                "pearson_after": candidate_correlation,
                "supported_fraction": float(
                    (update_supported & valid_array).sum()
                    / max(int(valid_array.sum()), 1)
                ),
                "jacobian_repair": repair_stats,
            }
        )
        if (
            not np.isfinite(candidate_correlation)
            or candidate_correlation <= current_correlation
        ):
            stats["status"] = "rejected_no_global_improvement"
            pass_stats.append(stats)
            break

        stats["status"] = "applied"
        pass_stats.append(stats)
        current_flow = candidate_flow
        current_warped = candidate_warped
        current_jacobian = flow_jacobian_determinant(current_flow)
        current_correlation = candidate_correlation
        residual_confidence |= accepted & (weights > 0.0)

    return (
        current_flow,
        current_warped,
        current_jacobian,
        current_correlation,
        residual_confidence,
        pass_stats,
    )


def _apply_local_residual_tiles(
    *,
    reference_array: np.ndarray,
    moving_array: np.ndarray,
    reference_prepared: np.ndarray,
    moving_prepared: np.ndarray,
    valid_array: np.ndarray,
    initial_flow: np.ndarray,
    initial_warped: np.ndarray,
    initial_jacobian: np.ndarray,
    initial_correlation: float,
    min_correlation_improvement: float,
    pixel_size_um: float,
    attachment: float,
    tightness: float,
    num_warp: int,
    num_iter: int,
    smoothing_sigma_um: float,
    max_displacement_um: float,
    max_cycle_error_um: float,
    min_local_correlation: float,
    min_jacobian: float,
    max_jacobian: float,
    min_confident_fraction: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
    dict[str, Any],
]:
    """Refine low-correlation subregions using locally normalized TV-L1 tiles."""
    current_flow = np.asarray(initial_flow, dtype=np.float32)
    current_warped = np.asarray(initial_warped, dtype=np.float32)
    current_jacobian = np.asarray(initial_jacobian, dtype=np.float32)
    current_correlation = float(initial_correlation)
    no_confidence = np.zeros(valid_array.shape, dtype=bool)
    tile_size_px = max(32, int(round(_LOCAL_RESIDUAL_TILE_UM / pixel_size_um)))
    tile_size_px += tile_size_px % 2
    if min(valid_array.shape) < tile_size_px:
        return (
            current_flow,
            current_warped,
            current_jacobian,
            current_correlation,
            no_confidence,
            {
                "status": "skipped_image_smaller_than_tile",
                "tile_size_px": tile_size_px,
            },
        )

    current_original = warp_with_flow(moving_array, current_flow)
    current_footprint = (
        warp_with_flow(
            valid_array.astype(np.float32),
            current_flow,
            interpolation=cv2.INTER_NEAREST,
        )
        > 0.5
    )
    pass_valid = valid_array & current_footprint
    local_sigma_px = max(
        1.0,
        _LOCAL_CORRELATION_SIGMA_UM / float(pixel_size_um),
    )
    current_local, _ = _local_correlation(
        reference_prepared,
        current_warped,
        pass_valid,
        sigma_px=local_sigma_px,
    )
    tile_step = tile_size_px // 2
    window_axis = np.sin(np.linspace(0.0, np.pi, tile_size_px, dtype=np.float32)) ** 2
    window = np.outer(window_axis, window_axis).astype(np.float32, copy=False)
    numerator = np.zeros((2, *valid_array.shape), dtype=np.float32)
    denominator = np.zeros(valid_array.shape, dtype=np.float32)
    candidate_tiles = 0
    attempted_tiles = 0
    accepted_tiles = 0

    for row in _tile_starts(valid_array.shape[0], tile_size_px, tile_step):
        for column in _tile_starts(valid_array.shape[1], tile_size_px, tile_step):
            local = np.s_[
                row : row + tile_size_px,
                column : column + tile_size_px,
            ]
            tile_valid = pass_valid[local]
            if tile_valid.mean() < _LOCAL_RESIDUAL_MIN_VALID_FRACTION:
                continue
            tile_correlation = current_local[local]
            correlation_values = tile_correlation[
                tile_valid & np.isfinite(tile_correlation)
            ]
            if (
                not correlation_values.size
                or np.median(correlation_values) >= _LOCAL_RESIDUAL_MAX_CORRELATION
            ):
                continue
            candidate_tiles += 1
            tile_result = estimate_nonrigid_flow(
                reference_array[local],
                current_original[local],
                tile_valid,
                pixel_size_um=pixel_size_um,
                attachment=attachment,
                tightness=tightness,
                num_warp=num_warp,
                num_iter=num_iter,
                smoothing_sigma_um=smoothing_sigma_um,
                max_displacement_um=max_displacement_um,
                max_cycle_error_um=max_cycle_error_um,
                min_local_correlation=min_local_correlation,
                min_jacobian=min_jacobian,
                max_jacobian=max_jacobian,
                min_confident_fraction=min_confident_fraction,
                residual_passes=0,
            )
            attempted_tiles += 1
            if tile_result.status != "applied":
                continue
            tile_improvement = float(
                tile_result.stats["pearson_after"] - tile_result.stats["pearson_before"]
            )
            if tile_improvement < min_correlation_improvement:
                continue
            confidence_weight = np.where(tile_result.confidence, 1.0, 0.25).astype(
                np.float32
            )
            weight = window * confidence_weight * tile_improvement
            numerator[:, local[0], local[1]] += tile_result.flow_yx * weight
            denominator[local] += weight
            accepted_tiles += 1

    supported = denominator > np.finfo(np.float32).eps
    stats: dict[str, Any] = {
        "status": "skipped_no_accepted_tiles",
        "tile_size_px": tile_size_px,
        "tile_size_um": float(tile_size_px * pixel_size_um),
        "tile_step_px": tile_step,
        "candidate_tiles": candidate_tiles,
        "attempted_tiles": attempted_tiles,
        "accepted_tiles": accepted_tiles,
        "supported_fraction": float(
            (supported & valid_array).sum() / max(int(valid_array.sum()), 1)
        ),
        "pearson_before": current_correlation,
    }
    if not accepted_tiles:
        return (
            current_flow,
            current_warped,
            current_jacobian,
            current_correlation,
            no_confidence,
            stats,
        )

    residual_update = np.zeros_like(numerator)
    for axis in range(2):
        np.divide(
            numerator[axis],
            denominator,
            out=residual_update[axis],
            where=supported,
        )
    candidate_flow = _compose_pull_flows(current_flow, residual_update)
    smoothing_sigma_px = max(
        0.5,
        float(smoothing_sigma_um) / float(pixel_size_um),
    )
    candidate_flow, repair_stats = _repair_jacobian_locally(
        candidate_flow,
        np.ones(valid_array.shape, dtype=bool),
        min_jacobian=min_jacobian,
        max_jacobian=max_jacobian,
        sigma_px=smoothing_sigma_px,
    )
    deep_invalid = ~binary_dilation(
        valid_array,
        iterations=max(1, int(math.ceil(2.0 * smoothing_sigma_px))),
    )
    candidate_flow[:, deep_invalid] = 0.0
    candidate_warped = warp_with_flow(moving_prepared, candidate_flow)
    candidate_correlation = _masked_correlation(
        reference_prepared,
        candidate_warped,
        valid_array,
    )
    stats.update(
        {
            "pearson_after": candidate_correlation,
            "jacobian_repair": repair_stats,
        }
    )
    if (
        not np.isfinite(candidate_correlation)
        or candidate_correlation <= current_correlation
    ):
        stats["status"] = "rejected_no_global_improvement"
        return (
            current_flow,
            current_warped,
            current_jacobian,
            current_correlation,
            no_confidence,
            stats,
        )

    stats["status"] = "applied"
    return (
        candidate_flow,
        candidate_warped,
        flow_jacobian_determinant(candidate_flow),
        candidate_correlation,
        supported,
        stats,
    )


def _tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    final_start = length - tile_size
    starts = list(range(0, final_start + 1, step))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _scale_to_jacobian_limits(
    flow_yx: np.ndarray,
    supported: np.ndarray,
    *,
    min_jacobian: float,
    max_jacobian: float,
) -> np.ndarray:
    if not supported.any():
        return np.zeros_like(flow_yx, dtype=np.float32)

    def acceptable(scale: float) -> bool:
        jacobian = flow_jacobian_determinant(flow_yx * float(scale))
        values = jacobian[supported]
        return bool(
            np.isfinite(values).all()
            and values.min() >= min_jacobian
            and values.max() <= max_jacobian
        )

    if acceptable(1.0):
        return np.asarray(flow_yx, dtype=np.float32)
    lower, upper = 0.0, 1.0
    for _ in range(20):
        middle = (lower + upper) / 2.0
        if acceptable(middle):
            lower = middle
        else:
            upper = middle
    return np.asarray(flow_yx * lower, dtype=np.float32)


def _masked_correlation(
    reference: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
) -> float:
    first = np.asarray(reference[valid], dtype=np.float64)
    second = np.asarray(moving[valid], dtype=np.float64)
    if first.size < 3:
        return float("nan")
    first -= first.mean()
    second -= second.mean()
    denominator = math.sqrt(float(first @ first) * float(second @ second))
    return float(first @ second / denominator) if denominator > 0 else float("nan")


def _flow_stats(
    flow_yx: np.ndarray,
    jacobian: np.ndarray,
    valid: np.ndarray,
    pixel_size_um: float,
) -> dict[str, Any]:
    magnitude_um = np.linalg.norm(flow_yx, axis=0)[valid] * float(pixel_size_um)
    jacobian_values = np.asarray(jacobian)[valid]
    return {
        "median_displacement_um": float(np.median(magnitude_um)),
        "p90_displacement_um": float(np.percentile(magnitude_um, 90.0)),
        "p99_displacement_um": float(np.percentile(magnitude_um, 99.0)),
        "max_displacement_um": float(magnitude_um.max()),
        "min_jacobian": float(jacobian_values.min()),
        "p01_jacobian": float(np.percentile(jacobian_values, 1.0)),
        "median_jacobian": float(np.median(jacobian_values)),
        "p99_jacobian": float(np.percentile(jacobian_values, 99.0)),
        "max_jacobian": float(jacobian_values.max()),
        "nonpositive_jacobian_fraction": float(np.mean(jacobian_values <= 0.0)),
    }


def _rejected_result(
    reference: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
    *,
    reason: str,
    pixel_size_um: float,
    confidence: np.ndarray | None = None,
    rejection_reasons: np.ndarray | None = None,
    local_correlation: np.ndarray | None = None,
    cycle_error_px: np.ndarray | None = None,
    extra_stats: dict[str, Any] | None = None,
) -> NonrigidFlowResult:
    shape = reference.shape
    zero_flow = np.zeros((2, *shape), dtype=np.float32)
    return NonrigidFlowResult(
        status="rejected",
        reason=reason,
        flow_yx=zero_flow,
        confidence=(
            np.zeros(shape, dtype=bool)
            if confidence is None
            else np.asarray(confidence, dtype=bool)
        ),
        residual_confidence=np.zeros(shape, dtype=bool),
        rejection_reasons=(
            np.full(shape, REJECTION_OUTSIDE_OVERLAP, dtype=np.uint16)
            if rejection_reasons is None
            else np.asarray(rejection_reasons, dtype=np.uint16)
        ),
        local_correlation=(
            np.full(shape, np.nan, dtype=np.float32)
            if local_correlation is None
            else np.asarray(local_correlation, dtype=np.float32)
        ),
        cycle_error_px=(
            np.full(shape, np.nan, dtype=np.float32)
            if cycle_error_px is None
            else np.asarray(cycle_error_px, dtype=np.float32)
        ),
        jacobian_determinant=np.ones(shape, dtype=np.float32),
        warped=np.asarray(moving, dtype=np.float32),
        stats={
            "pixel_size_um": float(pixel_size_um),
            "shape_yx": list(shape),
            "valid_fraction": float(np.asarray(valid, dtype=bool).mean()),
            **(extra_stats or {}),
        },
    )
