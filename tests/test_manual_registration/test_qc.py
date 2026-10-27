"""Tissue-scale alignment QC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from merxen.image_registration import WorkingImage
from merxen.manual_registration.qc import (
    PullMapping,
    affine_pull,
    resample_working,
    tissue_alignment_qc,
    write_tissue_qc_figure,
)

SIZE = 400
FACTOR = 4.0


def _to_level0() -> np.ndarray:
    return np.array(
        [[FACTOR, 0, (FACTOR - 1) / 2], [0, FACTOR, (FACTOR - 1) / 2], [0, 0, 1]]
    )


def _pattern(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(size=(SIZE, SIZE)), 6.0)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    rows, cols = np.indices((SIZE, SIZE))
    disc = (rows - SIZE / 2) ** 2 + (cols - SIZE / 2) ** 2 < 150**2
    # A dim background and a contrasty tissue disc (log scale, like a DAPI image).
    return np.asarray(50.0 + (50.0 + 3000.0 * noise**2) * disc, dtype=np.float32)


def _image(pixels: np.ndarray) -> WorkingImage:
    return WorkingImage(pixels=pixels, to_level0=_to_level0())


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0, dx], [0, 1.0, dy], [0, 0, 1]])


def _qc(
    reference: WorkingImage, moving: WorkingImage, pulls: dict[str, PullMapping]
) -> tuple[dict[str, float], dict[str, float]]:
    result = tissue_alignment_qc(
        reference,
        moving,
        pulls,
        working_pixel_size_um=10.0,
        sigma_um=20.0,
        highpass_sigma_um=200.0,
    )
    return result.pearson, result.mask_fraction


def test_identical_images_correlate_perfectly() -> None:
    image = _image(_pattern())
    pearson, fraction = _qc(image, image, {"final": lambda xy: xy})
    assert pearson["final"] == pytest.approx(1.0, abs=1e-9)
    assert fraction["final"] > 0.3


def test_shift_lowers_and_undoing_it_restores_the_correlation() -> None:
    pixels = _pattern()
    reference = _image(pixels)
    shifted = _image(np.roll(pixels, 10, axis=1))  # content moved +10 working px in x
    pulls = {
        "coarse": lambda xy: xy,
        "final": affine_pull(_translation(-40.0, 0.0)),
    }
    pearson, _ = _qc(reference, shifted, pulls)
    assert list(pearson) == ["coarse", "final"]
    assert pearson["coarse"] < 0.8
    assert pearson["final"] > 0.99


def test_empty_overlap_is_nan() -> None:
    image = _image(_pattern())
    pearson, fraction = _qc(image, image, {"far": lambda xy: xy + 1e6})
    assert np.isnan(pearson["far"])
    assert fraction["far"] == 0.0


def test_resample_working_does_not_depend_on_the_block_size() -> None:
    reference = _image(_pattern(1))
    moving = _image(_pattern(2))
    pull = affine_pull(_translation(13.5, -7.25))
    first = resample_working(reference, moving, pull, block_rows=7)
    second = resample_working(reference, moving, pull, block_rows=256)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_affine_pull_inverts_the_matrix() -> None:
    matrix = np.array([[0.0, -2.0, 5.0], [2.0, 0.0, -3.0], [0, 0, 1]])
    pull = affine_pull(matrix)
    points = np.array([[1.0, 2.0], [-4.0, 7.5]])
    forward = points @ matrix[:2, :2].T + matrix[:2, 2]
    np.testing.assert_allclose(pull(forward), points, atol=1e-12)


def test_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    image = _image(_pattern())
    path = write_tissue_qc_figure(
        image,
        image,
        {"landmark": lambda xy: xy, "final": lambda xy: xy},
        tmp_path / "qc" / "tissue.png",
        title="SYN1: tissue alignment",
    )
    assert path == tmp_path / "qc" / "tissue.png"
    assert path.stat().st_size > 0
    assert path.with_suffix(".pdf").stat().st_size > 0
