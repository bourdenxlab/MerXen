"""Tests for the micron-to-mask-pixel lookup in ``assign_labels_from_masks``."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_explicit_center_convention_equals_the_default() -> None:
    """Passing ``"center"`` is the same as omitting the argument."""
    x_px = [0.0, 0.5, 1.5, 2.49, 2.51, -0.4, 3.6]

    assert _labels_for_x(x_px, pixel_convention="center") == _labels_for_x(x_px)


def test_corner_convention_includes_the_lower_pixel_edge() -> None:
    """Under ``"corner"`` the position i * ps belongs to pixel i."""
    assert _labels_for_x([0.0, 1.0, 2.0, 3.0], pixel_convention="corner") == [
        1,
        2,
        3,
        4,
    ]


def test_corner_convention_excludes_the_upper_pixel_edge() -> None:
    """A position just below (i + 1) * ps still belongs to pixel i."""
    eps = 1e-9
    x_px = [1.0 - eps, 2.0 - eps, 3.0 - eps, 4.0 - eps, 0.5, 2.5]

    assert _labels_for_x(x_px, pixel_convention="corner") == [1, 2, 3, 4, 1, 3]


def test_corner_convention_leaves_out_of_bounds_positions_unassigned() -> None:
    """Negative positions and positions at or past the far edge get label 0."""
    eps = 1e-9
    x_px = [-eps, -0.4, -0.6, -1.0, 4.0, 4.5]

    assert _labels_for_x(x_px, pixel_convention="corner") == [0, 0, 0, 0, 0, 0]


def test_corner_convention_applies_to_rows() -> None:
    """The floor rule also selects the row, including the bottom edge."""
    masks = np.array([[1, 1], [2, 2]], dtype=np.uint32)
    eps = 1e-9
    y_px = np.array([0.0, 1.0 - eps, 1.0, 2.0 - eps, 2.0, -eps])
    x_um = np.full(len(y_px), 0.5 * _PIXEL_SIZE)

    labels = assign_labels_from_masks(
        x_um, y_px * _PIXEL_SIZE, masks, _A_INV, _B, pixel_convention="corner"
    )

    assert labels.tolist() == [1, 1, 2, 2, 0, 0]


def test_corner_convention_matches_a_floor_lookup_at_a_vendor_pixel_size() -> None:
    """A pure-scale affine reproduces ``masks[floor(y / ps), floor(x / ps)]``."""
    pixel_size = 0.2125
    rng = np.random.default_rng(11)
    masks = rng.integers(0, 200, size=(64, 96)).astype(np.uint32)
    x_um = rng.uniform(-1.0, 96 * pixel_size + 1.0, 20_000)
    y_um = rng.uniform(-1.0, 64 * pixel_size + 1.0, 20_000)
    a_inv = np.diag([1.0 / pixel_size, 1.0 / pixel_size])

    labels = assign_labels_from_masks(
        x_um, y_um, masks, a_inv, np.zeros(2), pixel_convention="corner"
    )

    columns = np.floor(x_um / pixel_size).astype(np.int64)
    rows = np.floor(y_um / pixel_size).astype(np.int64)
    inside = (columns >= 0) & (columns < 96) & (rows >= 0) & (rows < 64)
    expected = np.zeros(len(x_um), dtype=np.int32)
    expected[inside] = masks[rows[inside], columns[inside]].astype(np.int32)
    np.testing.assert_array_equal(labels, expected)


def test_conventions_differ_only_in_the_upper_half_of_each_pixel() -> None:
    """Positions in the lower half of a pixel agree; the upper half moves by one."""
    lower_half = [0.1, 1.2, 2.4]
    upper_half = [0.6, 1.8, 2.9]

    assert _labels_for_x(lower_half, pixel_convention="corner") == _labels_for_x(
        lower_half
    )
    assert _labels_for_x(upper_half, pixel_convention="corner") == [1, 2, 3]
    assert _labels_for_x(upper_half) == [2, 3, 4]


def test_unknown_pixel_convention_raises() -> None:
    """A misspelt convention is an error rather than a silent default."""
    with pytest.raises(ValueError, match="pixel_convention"):
        _labels_for_x([0.0], pixel_convention="centre")
