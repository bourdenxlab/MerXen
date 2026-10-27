"""Tests for guarded optical-flow refinement of registered images."""

from __future__ import annotations

import cv2
import numpy as np
from pytest import MonkeyPatch
from scipy.ndimage import gaussian_filter

from merxen.nonrigid_registration import (
    NonrigidFlowResult,
    _apply_local_residual_tiles,
    _masked_correlation,
    _prepare_image,
    _repair_jacobian_locally,
    estimate_nonrigid_flow,
    flow_jacobian_determinant,
    warp_with_flow,
)


def _nuclei_image(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(41)
    seeds = np.zeros((size, size), dtype=np.float32)
    ys = rng.integers(16, size - 16, 700)
    xs = rng.integers(16, size - 16, 700)
    seeds[ys, xs] = rng.uniform(500.0, 1500.0, len(ys))
    image = gaussian_filter(seeds, 1.8)
    image += gaussian_filter(rng.normal(0.0, 2.0, image.shape), 8.0)
    return np.maximum(image, 0.0).astype(np.float32)


def _flow_result(
    reference: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
) -> NonrigidFlowResult:
    return estimate_nonrigid_flow(
        reference,
        moving,
        valid,
        pixel_size_um=1.0,
        attachment=10.0,
        tightness=0.3,
        num_warp=5,
        num_iter=10,
        smoothing_sigma_um=5.0,
        max_displacement_um=12.0,
        max_cycle_error_um=3.0,
        min_local_correlation=0.2,
        min_jacobian=0.5,
        max_jacobian=2.0,
        min_confident_fraction=0.1,
    )


def test_guarded_flow_corrects_local_deformation_without_folding() -> None:
    reference = _nuclei_image()
    rows, columns = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    displacement_x = 4.0 * np.sin(2.0 * np.pi * rows / 160.0)
    displacement_y = 2.0 * np.sin(2.0 * np.pi * columns / 180.0)
    moving = cv2.remap(
        reference,
        columns - displacement_x,
        rows - displacement_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = np.ones(reference.shape, dtype=bool)
    valid[:12] = False
    valid[-12:] = False
    valid[:, :12] = False
    valid[:, -12:] = False

    result = _flow_result(reference, moving, valid)

    assert result.status == "applied"
    assert result.stats["pearson_after"] > result.stats["pearson_before"] + 0.05
    assert result.stats["final_flow"]["max_displacement_um"] <= 12.0
    assert result.jacobian_determinant[valid].min() >= 0.5
    assert result.jacobian_determinant[valid].max() <= 2.0
    assert np.corrcoef(result.flow_yx[1, valid], displacement_x[valid])[0, 1] > 0.9
    assert np.corrcoef(result.flow_yx[0, valid], displacement_y[valid])[0, 1] > 0.9


def test_damaged_and_nonoverlapping_regions_fall_back_to_affine() -> None:
    reference = _nuclei_image()
    rows, columns = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    displacement_x = 3.0 * np.sin(2.0 * np.pi * rows / 180.0)
    moving = cv2.remap(
        reference,
        columns - displacement_x,
        rows,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    moving[80:145, 90:165] = np.flipud(reference[80:145, 90:165])
    valid = np.ones(reference.shape, dtype=bool)
    valid[:, :45] = False

    result = _flow_result(reference, moving, valid)

    assert result.status == "applied"
    assert np.max(np.abs(result.flow_yx[:, :, :35])) == 0.0
    assert result.confidence[95:140, 100:155].mean() < 0.75
    damaged_magnitude = np.linalg.norm(result.flow_yx[:, 95:140, 100:155], axis=0)
    assert damaged_magnitude.max() < 6.0
    assert result.jacobian_determinant[valid].min() >= 0.5


def test_flow_warp_uses_fixed_to_moving_displacements() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[10:14, 20:24] = 1.0
    flow = np.zeros((2, 32, 32), dtype=np.float32)
    flow[1] = 5.0

    warped = warp_with_flow(image, flow)

    assert warped[10:14, 15:19].mean() == 1.0
    assert flow_jacobian_determinant(flow).min() == 1.0


def test_flow_is_rejected_without_overlap() -> None:
    image = _nuclei_image(64)

    result = _flow_result(image, image, np.zeros(image.shape, dtype=bool))

    assert result.status == "rejected"
    assert result.reason == "no overlapping tissue pixels"
    assert not result.flow_yx.any()


def test_jacobian_repair_is_local_instead_of_scaling_the_whole_field() -> None:
    flow = np.zeros((2, 128, 128), dtype=np.float32)
    flow[1] = 4.0
    local_columns = np.arange(28, dtype=np.float32) - 13.5
    flow[1, 50:78, 50:78] -= 0.8 * local_columns[None, :]

    repaired, stats = _repair_jacobian_locally(
        flow,
        np.ones((128, 128), dtype=bool),
        min_jacobian=0.5,
        max_jacobian=2.0,
        sigma_px=3.0,
    )

    jacobian = flow_jacobian_determinant(repaired)
    assert jacobian.min() >= 0.5
    assert jacobian.max() <= 2.0
    np.testing.assert_array_equal(repaired[:, :, :30], flow[:, :, :30])
    assert stats["global_fallback_scale"] == 1.0
    assert stats["locally_changed_fraction"] < 0.2


def test_residual_pass_improves_an_undercorrected_translation() -> None:
    reference = _nuclei_image(192)
    moving = cv2.warpAffine(
        reference,
        np.asarray([[1.0, 0.0, -7.0], [0.0, 1.0, 4.0]], dtype=np.float32),
        (192, 192),
    )
    valid = np.ones(reference.shape, dtype=bool)
    valid[:15] = False
    valid[-15:] = False
    valid[:, :15] = False
    valid[:, -15:] = False
    parameters = {
        "pixel_size_um": 1.0,
        "attachment": 10.0,
        "tightness": 0.3,
        "num_warp": 2,
        "num_iter": 3,
        "smoothing_sigma_um": 5.0,
        "max_displacement_um": 12.0,
        "max_cycle_error_um": 3.0,
        "min_local_correlation": 0.2,
        "min_jacobian": 0.5,
        "max_jacobian": 2.0,
        "min_confident_fraction": 0.1,
    }

    single_pass = estimate_nonrigid_flow(
        reference,
        moving,
        valid,
        residual_passes=0,
        **parameters,
    )
    refined = estimate_nonrigid_flow(
        reference,
        moving,
        valid,
        residual_passes=1,
        **parameters,
    )

    assert single_pass.status == "applied"
    assert refined.status == "applied"
    assert refined.stats["residual_passes"][0]["status"] == "applied"
    assert refined.stats["pearson_after"] > single_pass.stats["pearson_after"]
    assert refined.residual_confidence.any()
    assert refined.jacobian_determinant[valid].min() >= 0.5
    assert refined.jacobian_determinant[valid].max() <= 2.0


def test_local_residual_tiles_correct_only_the_shifted_subregion(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "merxen.nonrigid_registration._LOCAL_RESIDUAL_TILE_UM",
        64.0,
    )
    reference = _nuclei_image(256)
    shifted = cv2.warpAffine(
        reference,
        np.asarray([[1.0, 0.0, -5.0], [0.0, 1.0, 2.0]], dtype=np.float32),
        (256, 256),
    )
    moving = reference.copy()
    moving[:, 128:] = shifted[:, 128:]
    valid = np.ones(reference.shape, dtype=bool)
    valid[:16] = False
    valid[-16:] = False
    valid[:, :16] = False
    valid[:, -16:] = False
    reference_prepared = _prepare_image(reference, valid, pixel_size_um=1.0)
    moving_prepared = _prepare_image(moving, valid, pixel_size_um=1.0)
    initial_flow = np.zeros((2, *reference.shape), dtype=np.float32)
    initial_correlation = _masked_correlation(
        reference_prepared,
        moving_prepared,
        valid,
    )

    flow, warped, jacobian, correlation, confidence, stats = (
        _apply_local_residual_tiles(
            reference_array=reference,
            moving_array=moving,
            reference_prepared=reference_prepared,
            moving_prepared=moving_prepared,
            valid_array=valid,
            initial_flow=initial_flow,
            initial_warped=moving_prepared,
            initial_jacobian=np.ones(reference.shape, dtype=np.float32),
            initial_correlation=initial_correlation,
            min_correlation_improvement=0.03,
            pixel_size_um=1.0,
            attachment=10.0,
            tightness=0.3,
            num_warp=5,
            num_iter=10,
            smoothing_sigma_um=5.0,
            max_displacement_um=12.0,
            max_cycle_error_um=3.0,
            min_local_correlation=0.2,
            min_jacobian=0.5,
            max_jacobian=2.0,
            min_confident_fraction=0.1,
        )
    )

    assert stats["status"] == "applied"
    assert stats["accepted_tiles"] > 0
    assert correlation > initial_correlation
    assert confidence[:, 150:].any()
    assert np.median(np.linalg.norm(flow[:, 32:224, 32:96], axis=0)) < 0.25
    assert _masked_correlation(
        reference_prepared[:, 150:],
        warped[:, 150:],
        valid[:, 150:],
    ) > _masked_correlation(
        reference_prepared[:, 150:],
        moving_prepared[:, 150:],
        valid[:, 150:],
    )
    assert jacobian[valid].min() >= 0.5
    assert jacobian[valid].max() <= 2.0
