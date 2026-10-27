"""Tests for per-cell registered-image quality measurements."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, shift

from merxen.registration_quality import compute_cell_registration_quality


def _synthetic_cells() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.zeros((96, 96), dtype=np.uint32)
    nuclei = np.zeros_like(labels)
    reference = np.zeros(labels.shape, dtype=np.float32)
    label = 1
    for row in range(3):
        for column in range(3):
            y0, x0 = 5 + row * 30, 5 + column * 30
            labels[y0 : y0 + 26, x0 : x0 + 26] = label
            nuclei[y0 + 7 : y0 + 19, x0 + 8 : x0 + 18] = label
            reference[y0 + 7 : y0 + 19, x0 + 8 : x0 + 18] = 100.0
            label += 1
    return reference, nuclei, labels


def test_quality_score_separates_aligned_and_cell_shifted_dapi() -> None:
    """A nuclear-scale displacement must lower the per-cell quality score."""
    reference, nuclei, cells = _synthetic_cells()
    reference = gaussian_filter(reference, 1.2)

    aligned = compute_cell_registration_quality(
        reference,
        reference,
        nuclei,
        cells,
        downsample=1,
        source_pixel_size_um=0.85,
        smoothing_sigma_um=0.68,
    ).metrics
    displaced = compute_cell_registration_quality(
        reference,
        shift(reference, shift=(0, 7), order=1, mode="constant"),
        nuclei,
        cells,
        downsample=1,
        source_pixel_size_um=0.85,
        smoothing_sigma_um=0.68,
    ).metrics

    assert aligned["valid"].all()
    assert aligned["score"].median() > 0.99
    assert displaced["score"].median() < 0.65
    assert displaced["dapi_pearson"].median() < aligned["dapi_pearson"].median()


def test_quality_score_marks_cells_without_nuclear_support_invalid() -> None:
    """Cells without enough nuclear-mask pixels should not receive a score."""
    reference, nuclei, cells = _synthetic_cells()
    nuclei[nuclei == 5] = 0

    metrics = compute_cell_registration_quality(
        reference,
        reference,
        nuclei,
        cells,
        downsample=1,
        source_pixel_size_um=0.85,
    ).metrics.set_index("cellpose_label")

    assert not bool(metrics.loc[5, "valid"])
    assert np.isnan(metrics.loc[5, "score"])
