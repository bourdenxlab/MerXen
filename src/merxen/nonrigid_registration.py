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


@dataclass(frozen=True)
class NonrigidFlowResult:
    """A guarded fixed-grid to moving-grid displacement field."""

    status: str
    reason: str | None
    flow_yx: np.ndarray
    confidence: np.ndarray
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
) -> NonrigidFlowResult:
    """Estimate and guard a TV-L1 pull displacement field.

    The returned ``flow_yx`` satisfies ``moving[p + flow(p)] ~= reference[p]``.
    Regions without reliable bidirectional support retain zero displacement, so
    the caller falls back to its affine mapping there.
    """
    reference_array = np.asarray(reference, dtype=np.float32)
    moving_array = np.asarray(moving, dtype=np.float32)
    valid_array = np.asarray(valid, dtype=bool)
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
    confidence = (
        comparison_valid
        & (local_support >= _MIN_LOCAL_SUPPORT)
        & np.isfinite(correlation)
        & (correlation >= float(min_local_correlation))
        & np.isfinite(cycle_error)
        & (cycle_error <= max_cycle_error_px)
        & np.isfinite(raw_magnitude)
        & (raw_magnitude <= max_displacement_px)
        & np.isfinite(raw_jacobian)
        & (raw_jacobian >= float(min_jacobian))
        & (raw_jacobian <= float(max_jacobian))
    )
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
    guarded = _scale_to_jacobian_limits(
        guarded,
        np.ones(valid_array.shape, dtype=bool),
        min_jacobian=float(min_jacobian),
        max_jacobian=float(max_jacobian),
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

    return NonrigidFlowResult(
        status="applied",
        reason=None,
        flow_yx=guarded.astype(np.float32, copy=False),
        confidence=confidence,
        local_correlation=correlation.astype(np.float32, copy=False),
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
