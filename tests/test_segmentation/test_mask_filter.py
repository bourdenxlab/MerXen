"""Tests for regionprops-based mask filtering."""

from __future__ import annotations

import numpy as np

from merxen.segmentation.mask_filter import filter_cell_by_regionprops


def test_filter_cell_by_regionprops_drops_eccentric_region() -> None:
    """Elongated high-eccentricity components should be removed."""
    seg = np.zeros((32, 32), dtype=np.uint8)
    seg[2:10, 2:10] = 1
    seg[20:21, 4:28] = 1

    out = filter_cell_by_regionprops(
        seg,
        max_eccentricity=0.90,
        n_jobs=1,
        min_area_percentile=0.0,
    )

    labels = np.unique(out)
    assert labels.tolist() == [0, 1]
    assert int((out == 1).sum()) == 64


def test_filter_cell_by_regionprops_returns_empty_for_empty_input() -> None:
    """Empty masks should produce an all-zero output mask."""
    seg = np.zeros((16, 16), dtype=np.uint8)
    out = filter_cell_by_regionprops(seg, n_jobs=1)
    assert out.shape == seg.shape
    assert out.dtype == np.int32
    assert int(out.sum()) == 0
