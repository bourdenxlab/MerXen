"""Tests for regionprops-based mask filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from merxen.segmentation.mask_filter import (
    filter_cell_by_regionprops,
    filter_labeled_mask_by_area,
)


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


def test_filter_labeled_mask_by_area_removes_small_and_large_masks(
    tmp_path: Path,
) -> None:
    """Absolute micron-area filtering should stream, remove, and relabel masks."""
    mask = np.zeros((64, 64), dtype=np.uint32)
    mask[1:3, 1:3] = 1
    mask[5:10, 5:10] = 2
    mask[12:31, 12:31] = 3
    mask[36:58, 36:58] = 4

    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)

    stats = filter_labeled_mask_by_area(
        mask_path,
        pixel_area_um2=1.0,
        min_area_um2=5.0,
        max_area_um2=400.0,
        chunk_mb=1,
        show_progress=False,
    )

    out = np.load(mask_path)
    assert stats["n_labels"] == 4
    assert stats["n_kept"] == 2
    assert stats["n_removed_small"] == 1
    assert stats["n_removed_large"] == 1
    assert np.unique(out).tolist() == [0, 1, 2]
    assert int((out == 1).sum()) == 25
    assert int((out == 2).sum()) == 361
