"""Tests for acquisition-footprint-aware DAPI registration preprocessing."""

from __future__ import annotations

import numpy as np

from merxen.alignment.dapi import (
    dapi_edge_artifact_metrics,
    derive_acquired_support_mask,
    process_dapi_image,
)
from merxen.alignment.frames import DapiFrame, _prepare_one_frame
from merxen.config import DAPIProcessingConfig


def _irregular_merscope_image() -> tuple[np.ndarray, np.ndarray]:
    height, width = 160, 180
    rows, columns = np.indices((height, width))
    support = np.zeros((height, width), dtype=bool)
    support[12:148, 18:170] = True
    support[12:55, 18:42] = False
    support[92:112, 18:36] = False

    rng = np.random.default_rng(41)
    image = np.zeros((height, width), dtype=np.float32)
    image[support] = (
        110.0 + 0.2 * columns[support] + rng.normal(0.0, 4.0, int(support.sum()))
    )
    for center_row, center_column, amplitude, width_squared in (
        (66, 70, 210.0, 450.0),
        (103, 124, 250.0, 680.0),
        (48, 132, 175.0, 300.0),
    ):
        image += np.where(
            support,
            amplitude
            * np.exp(
                -((columns - center_column) ** 2 + (rows - center_row) ** 2)
                / width_squared
            ),
            0.0,
        )
    image[72:75, 82:85] = 0.0
    return image, support


def _processing_config() -> DAPIProcessingConfig:
    return DAPIProcessingConfig(
        background_sigma_um=15.0,
        smoothing_sigma_um=1.0,
        edge_taper_um=20.0,
        edge_exclusion_um=10.0,
        mask_downsample=2,
        mask_smoothing_sigma_um=2.0,
        mask_closing_radius_um=2.0,
        mask_min_area_um2=30.0,
        mask_hole_area_um2=30.0,
        mask_dilation_um=0.0,
    )


def test_support_inference_is_merscope_specific_and_fills_dark_holes() -> None:
    """Only MERSCOPE zero padding should define an irregular acquired footprint."""
    image, expected = _irregular_merscope_image()

    merscope = derive_acquired_support_mask(image, platform="MERSCOPE") > 0
    xenium = derive_acquired_support_mask(image, platform="XENIUM") > 0

    assert np.mean(merscope == expected) > 0.999
    assert merscope[73, 83]
    assert not merscope[30, 25]
    assert not merscope[100, 25]
    assert np.all(xenium)


def test_mask_normalized_background_and_support_taper_remove_irregular_halo() -> None:
    """Unacquired zeros must not create a high-pass rim along a stepped footprint."""
    image, _ = _irregular_merscope_image()
    original = image.copy()
    support = derive_acquired_support_mask(image, platform="MERSCOPE")
    config = _processing_config()

    rectangular = process_dapi_image(
        image,
        pixel_size_um=1.0,
        config=config,
    )
    corrected = process_dapi_image(
        image,
        pixel_size_um=1.0,
        config=config,
        acquired_support_mask=support,
    )
    rectangular_metrics = dapi_edge_artifact_metrics(
        rectangular,
        acquired_support_mask=support,
    )
    corrected_metrics = dapi_edge_artifact_metrics(
        corrected,
        acquired_support_mask=support,
    )

    assert np.array_equal(image, original)
    assert np.all(corrected[support == 0] == 0)
    assert corrected_metrics["support_boundary_to_inner_p95_ratio"] < 0.35
    assert (
        corrected_metrics["support_boundary_to_inner_p95_ratio"]
        < 0.25 * rectangular_metrics["support_boundary_to_inner_p95_ratio"]
    )


def test_registration_frame_excludes_support_boundary_from_tissue() -> None:
    """Returned frame masks must keep invalid footprint edges out of objectives."""
    image, _ = _irregular_merscope_image()
    dapi = DapiFrame(
        platform="MERSCOPE",
        image_key="test_dapi",
        image=image,
        original_shape_rc=image.shape,
        dataset_to_image_matrix=np.eye(3, dtype=np.float64),
        pixel_size_xy_um=(1.0, 1.0),
        channel_name="DAPI",
        coordinate_metadata_source="test",
        coordinate_metadata_trusted=True,
    )

    frame = _prepare_one_frame(
        dapi,
        target_shape_rc=image.shape,
        canvas_shape_rc=(180, 200),
        target_pixel_size_um=1.0,
        processing=_processing_config(),
    )

    assert frame.support_mask.dtype == np.uint8
    assert frame.valid_mask.dtype == np.uint8
    assert not np.any((frame.tissue_mask > 0) & (frame.valid_mask == 0))
    assert np.all(frame.processed_image[frame.support_mask == 0] == 0)
    assert frame.valid_mask[10 + 15, 10 + 70] == 0
    assert frame.valid_mask[10 + 80, 10 + 90] == 255
    assert "support_boundary_to_inner_p95_ratio" in frame.edge_artifact_metrics
    assert frame.edge_artifact_metrics["support_fraction"] < 0.75
