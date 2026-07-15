"""Tests for the ported pyramid + outline builders."""

from __future__ import annotations

import dask.array as da
import numpy as np

from merxen.viewer_cache.pyramids import (
    label_outline_mask_chunk,
    lazy_coarsened_pyramid,
    lazy_label_pyramid,
    lazy_outline_pyramid,
)


def test_coarsened_pyramid_excludes_base_and_shrinks() -> None:
    base = da.zeros((640, 640), chunks=(128, 128), dtype=np.uint32)
    levels = lazy_coarsened_pyramid(base, step=4, reducer=np.max, min_size=32)
    # Base is excluded; each level is /4 while the larger axis stays > min_size.
    assert [tuple(int(s) for s in lvl.shape) for lvl in levels] == [
        (160, 160),
        (40, 40),
        (10, 10),
    ]
    # Strictly decreasing so the viewer's multiscale check accepts it.
    sizes = [lvl.shape[0] for lvl in levels]
    assert sizes == sorted(sizes, reverse=True)


def test_coarsened_pyramid_empty_when_base_below_min_size() -> None:
    base = da.zeros((16, 16), chunks=(16, 16), dtype=np.uint32)
    assert lazy_coarsened_pyramid(base, step=4, reducer=np.max, min_size=4096) == []


def test_label_pyramid_includes_base_and_halves() -> None:
    base = da.zeros((320, 320), chunks=(64, 64), dtype=np.uint32)
    levels = lazy_label_pyramid(base, min_size=32)
    shapes = [tuple(int(s) for s in lvl.shape) for lvl in levels]
    assert shapes[0] == (320, 320)  # base included as finest level
    assert shapes == [(320, 320), (160, 160), (80, 80), (40, 40), (20, 20)]


def test_label_pyramid_max_pooling_preserves_ids() -> None:
    arr = np.zeros((4, 4), dtype=np.uint32)
    arr[0, 0] = 7
    levels = lazy_label_pyramid(
        da.from_array(arr, chunks=(2, 2)), min_size=1, max_levels=3
    )
    coarse = np.asarray(levels[1].compute())
    # Max-pool keeps the id rather than averaging it away.
    assert coarse[0, 0] == 7


def test_outline_mask_marks_boundaries() -> None:
    arr = np.zeros((6, 6), dtype=np.uint32)
    arr[1:5, 1:5] = 3
    outline = label_outline_mask_chunk(arr, width=1)
    assert outline.dtype == np.uint8
    # Interior pixel is not outline; a border pixel of the region is.
    assert outline[2, 2] == 0
    assert outline[1, 1] == 1


def test_outline_pyramid_builds_uint8_levels() -> None:
    base = da.zeros((320, 320), chunks=(64, 64), dtype=np.uint32)
    levels = lazy_outline_pyramid(base, width=1, min_size=32)
    assert len(levels) == 5  # one outline per label-pyramid level (base + 4 coarser)
    assert all(lvl.dtype == np.uint8 for lvl in levels)
