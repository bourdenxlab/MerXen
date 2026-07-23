"""Tests for mask polygon extraction."""

from __future__ import annotations

import numpy as np

from merxen.segmentation.mask_geometry import (
    masks_to_labeled_polygons,
    masks_to_polygons,
)


def test_masks_to_polygons_extracts_expected_count() -> None:
    """Two disconnected labels should produce two polygons."""
    masks = np.zeros((10, 10), dtype=np.int32)
    masks[1:4, 1:4] = 1
    masks[6:9, 6:9] = 2

    polys = masks_to_polygons(masks, n_jobs=1)
    assert len(polys) == 2
    assert all(poly.is_valid and not poly.is_empty for poly in polys)


def test_masks_to_polygons_applies_scale_factor() -> None:
    """Scaling polygons by 2x should increase area by approximately 4x."""
    masks = np.zeros((8, 8), dtype=np.int32)
    masks[2:6, 2:6] = 1

    base = masks_to_polygons(masks, factor_rescale=0.0, n_jobs=1)[0]
    scaled = masks_to_polygons(masks, factor_rescale=2.0, n_jobs=1)[0]

    assert base.area > 0
    ratio = scaled.area / base.area
    assert np.isclose(ratio, 4.0, rtol=1e-3)


def test_masks_to_labeled_polygons_preserves_sparse_label_ids() -> None:
    """Polygon extraction should retain non-contiguous mask identifiers."""
    masks = np.zeros((10, 10), dtype=np.int32)
    masks[1:4, 1:4] = 2
    masks[6:9, 6:9] = 9

    labeled = masks_to_labeled_polygons(masks, n_jobs=1)

    assert [label_id for label_id, _ in labeled] == [2, 9]


def test_masks_to_labeled_polygons_preserves_disconnected_label_components() -> None:
    """One mask ID with two components should retain both pieces."""
    masks = np.zeros((12, 12), dtype=np.int32)
    masks[1:4, 1:4] = 5
    masks[8:11, 8:11] = 5

    [(label_id, geometry)] = masks_to_labeled_polygons(masks, n_jobs=1)

    assert label_id == 5
    assert geometry.geom_type == "MultiPolygon"
    assert len(geometry.geoms) == 2
