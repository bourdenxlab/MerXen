"""Tests for ``NonrigidTuning``: defaults, bit identity and the individual knobs."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import pytest
from pytest import MonkeyPatch
from scipy.ndimage import gaussian_filter

import merxen.nonrigid_registration as nr
from merxen.nonrigid_registration import (
    NonrigidFlowResult,
    NonrigidTuning,
    _apply_local_residual_tiles,
    _masked_correlation,
    _prepare_image,
    estimate_nonrigid_flow,
)

_PARAMETERS: dict[str, Any] = {
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


def _nuclei_image(size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(41)
    seeds = np.zeros((size, size), dtype=np.float32)
    ys = rng.integers(16, size - 16, 700)
    xs = rng.integers(16, size - 16, 700)
    seeds[ys, xs] = rng.uniform(500.0, 1500.0, len(ys))
    image = gaussian_filter(seeds, 1.8)
    image += gaussian_filter(rng.normal(0.0, 2.0, image.shape), 8.0)
    return np.maximum(image, 0.0).astype(np.float32)


def _translation_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return reference, moving, valid


def _run(
    tuning: NonrigidTuning | None, *, residual_passes: int = 1
) -> NonrigidFlowResult:
    reference, moving, valid = _translation_inputs()
    return estimate_nonrigid_flow(
        reference,
        moving,
        valid,
        residual_passes=residual_passes,
        tuning=tuning,
        **_PARAMETERS,
    )


def _tile_inputs(
    size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    reference = _nuclei_image(size)
    shifted = cv2.warpAffine(
        reference,
        np.asarray([[1.0, 0.0, -5.0], [0.0, 1.0, 2.0]], dtype=np.float32),
        (size, size),
    )
    moving = reference.copy()
    moving[:, size // 2 :] = shifted[:, size // 2 :]
    valid = np.ones(reference.shape, dtype=bool)
    valid[:16] = False
    valid[-16:] = False
    valid[:, :16] = False
    valid[:, -16:] = False
    reference_prepared = _prepare_image(reference, valid, pixel_size_um=1.0)
    moving_prepared = _prepare_image(moving, valid, pixel_size_um=1.0)
    correlation = _masked_correlation(reference_prepared, moving_prepared, valid)
    return reference, moving, valid, reference_prepared, moving_prepared, correlation


def _tile_stats(tuning: NonrigidTuning | None) -> dict[str, Any]:
    reference, moving, valid, reference_prepared, moving_prepared, correlation = (
        _tile_inputs()
    )
    *_, stats = _apply_local_residual_tiles(
        reference_array=reference,
        moving_array=moving,
        reference_prepared=reference_prepared,
        moving_prepared=moving_prepared,
        valid_array=valid,
        initial_flow=np.zeros((2, *reference.shape), dtype=np.float32),
        initial_warped=moving_prepared,
        initial_jacobian=np.ones(reference.shape, dtype=np.float32),
        initial_correlation=correlation,
        min_correlation_improvement=0.03,
        tuning=tuning,
        **{**_PARAMETERS, "num_warp": 5, "num_iter": 10},
    )
    return stats


def test_tuning_defaults_equal_module_constants() -> None:
    tuning = NonrigidTuning()

    assert (
        tuning.highpass_sigma_um,
        tuning.local_correlation_sigma_um,
        tuning.local_residual_tiles,
        tuning.local_residual_tile_um,
    ) == (25.0, 15.0, True, 220.0)
    assert tuning.highpass_sigma_um == nr._HIGHPASS_SIGMA_UM
    assert tuning.local_correlation_sigma_um == nr._LOCAL_CORRELATION_SIGMA_UM
    assert tuning.local_residual_tile_um == nr._LOCAL_RESIDUAL_TILE_UM


def test_default_tuning_is_bit_identical_to_no_tuning() -> None:
    implicit = _run(None)
    explicit = _run(NonrigidTuning())

    assert implicit.status == explicit.status
    for name in (
        "flow_yx",
        "confidence",
        "residual_confidence",
        "rejection_reasons",
        "jacobian_determinant",
        "warped",
    ):
        assert np.array_equal(getattr(implicit, name), getattr(explicit, name)), name
    for name in ("local_correlation", "cycle_error_px"):
        assert np.array_equal(
            getattr(implicit, name), getattr(explicit, name), equal_nan=True
        ), name
    assert json.dumps(implicit.stats, sort_keys=True) == json.dumps(
        explicit.stats, sort_keys=True
    )


def test_local_tiles_off_disables_only_the_tile_stage() -> None:
    with_tiles = _run(NonrigidTuning())
    without_tiles = _run(NonrigidTuning(local_residual_tiles=False))

    assert without_tiles.stats["local_residual_refinement"] == {"status": "disabled"}
    assert with_tiles.stats["local_residual_refinement"]["status"] != "disabled"
    assert with_tiles.stats["residual_passes"]
    assert json.dumps(
        with_tiles.stats["residual_passes"], sort_keys=True
    ) == json.dumps(without_tiles.stats["residual_passes"], sort_keys=True)


def test_tile_size_follows_tuning() -> None:
    small = _tile_stats(NonrigidTuning(local_residual_tile_um=64.0))
    huge = _tile_stats(NonrigidTuning(local_residual_tile_um=1000.0))

    assert small["tile_size_px"] == 64
    assert huge["status"] == "skipped_image_smaller_than_tile"


def test_highpass_sigma_changes_preparation() -> None:
    image = _nuclei_image(128)
    valid = np.ones(image.shape, dtype=bool)

    default = _prepare_image(image, valid, pixel_size_um=1.0)

    assert np.array_equal(
        default, _prepare_image(image, valid, pixel_size_um=1.0, highpass_sigma_um=25.0)
    )
    assert not np.array_equal(
        default,
        _prepare_image(image, valid, pixel_size_um=1.0, highpass_sigma_um=120.0),
    )


def test_monkeypatched_constant_still_applies_without_tuning(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr("merxen.nonrigid_registration._LOCAL_RESIDUAL_TILE_UM", 64.0)

    assert _tile_stats(None)["tile_size_px"] == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"highpass_sigma_um": 0},
        {"highpass_sigma_um": -1},
        {"highpass_sigma_um": float("nan")},
        {"local_residual_tile_um": float("inf")},
        {"local_residual_tiles": 1},
    ],
)
def test_invalid_tuning_is_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="NonrigidTuning"):
        NonrigidTuning(**kwargs)
