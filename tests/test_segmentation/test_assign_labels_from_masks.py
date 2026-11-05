"""Tests for the micron-to-mask-pixel lookup in ``assign_labels_from_masks``."""

from __future__ import annotations

import numpy as np

from merxen.segmentation.cellpose import assign_labels_from_masks

# A dyadic pixel size keeps i * ps exact in floating point, so boundary cases
# test the rounding rule rather than representation error.
_PIXEL_SIZE = 0.25
_A_INV = np.diag([1.0 / _PIXEL_SIZE, 1.0 / _PIXEL_SIZE])
_B = np.zeros(2)


def _column_mask(width: int = 4, height: int = 2) -> np.ndarray:
    """Return a mask whose label is the column index plus one."""
    return np.tile(np.arange(1, width + 1, dtype=np.uint32), (height, 1))


def _labels_for_x(x_px: list[float], **kwargs: str) -> list[int]:
    """Look up labels for pixel x positions on row zero of the column mask."""
    x_um = np.asarray(x_px, dtype=np.float64) * _PIXEL_SIZE
    y_um = np.full_like(x_um, 0.25 * _PIXEL_SIZE)
    labels = assign_labels_from_masks(x_um, y_um, _column_mask(), _A_INV, _B, **kwargs)
    return labels.tolist()


def _rint_reference(
    x_um: np.ndarray,
    y_um: np.ndarray,
    masks: np.ndarray,
    a_inv: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Reproduce the original pixel-centre lookup for comparison."""
    pix = a_inv @ (np.vstack([x_um, y_um]) - b[:, None])
    x_px = np.rint(pix[0]).astype(np.int64)
    y_px = np.rint(pix[1]).astype(np.int64)
    height, width = masks.shape
    valid = (x_px >= 0) & (x_px < width) & (y_px >= 0) & (y_px < height)
    labels = np.zeros(len(x_um), dtype=np.int32)
    labels[valid] = masks[y_px[valid], x_px[valid]].astype(np.int32, copy=False)
    return labels


def test_default_lookup_rounds_to_the_nearest_pixel_centre() -> None:
    """The default maps a position to the pixel whose centre is nearest."""
    labels = _labels_for_x([0.0, 0.4, 0.6, 1.49, 2.51, 3.49, 3.51, -0.4, -0.6])

    assert labels == [1, 1, 2, 2, 4, 4, 0, 1, 0]


def test_default_lookup_rounds_half_pixels_to_even() -> None:
    """Exact half pixels follow ``np.rint``, which rounds half to even."""
    assert _labels_for_x([0.5, 1.5, 2.5]) == [1, 3, 3]


def test_default_lookup_matches_the_rint_reference_under_an_affine() -> None:
    """The default output is unchanged for a general affine and random points."""
    rng = np.random.default_rng(7)
    masks = rng.integers(0, 50, size=(37, 53)).astype(np.uint32)
    a_inv = np.array([[4.6, 0.3], [-0.2, 4.4]])
    b = np.array([-1.5, 2.25])
    x_um = rng.uniform(-3.0, 14.0, 5000)
    y_um = rng.uniform(-1.0, 12.0, 5000)

    labels = assign_labels_from_masks(x_um, y_um, masks, a_inv, b)

    expected = _rint_reference(x_um, y_um, masks, a_inv, b)
    assert labels.dtype == np.int32
    np.testing.assert_array_equal(labels, expected)
