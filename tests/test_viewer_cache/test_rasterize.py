"""Tests for instance-id assignment and polygon rasterization."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from merxen.viewer_cache.rasterize import (
    label_ids_for_shapes,
    napari_affine_from_px_to_um,
    rasterize_geometries_chunk,
)


def _square(cx: float, cy: float, s: float) -> Polygon:
    return Polygon([(cx, cy), (cx + s, cy), (cx + s, cy + s), (cx, cy + s)])


def test_label_ids_use_true_instance_ids_not_positional() -> None:
    gdf = gpd.GeoDataFrame(
        geometry=[_square(0, 0, 1), _square(2, 0, 1), _square(4, 0, 1)],
        index=[2, 94831, 7],
    )
    series, dtype = label_ids_for_shapes(gdf)
    assert np.dtype(dtype) == np.uint32
    # The polygon at index 94831 keeps its id, not the positional +1 of 2.
    assert series.loc[94831] == 94831
    assert series.loc[2] == 2
    assert series.loc[7] == 7


def test_label_ids_promote_to_uint64_for_large_ids() -> None:
    big = 3715213700018100006  # merscope-style EntityID, exceeds uint32
    gdf = gpd.GeoDataFrame(geometry=[_square(0, 0, 1)], index=[big])
    series, dtype = label_ids_for_shapes(gdf)
    assert np.dtype(dtype) == np.uint64
    assert int(series.loc[big]) == big


def test_label_ids_fall_back_to_codes_for_noninteger_index() -> None:
    gdf = gpd.GeoDataFrame(
        geometry=[_square(0, 0, 1), _square(2, 0, 1)],
        index=["cellpose_1", "cellpose_2"],
    )
    series, dtype = label_ids_for_shapes(gdf)
    assert np.dtype(dtype) == np.uint32
    # Positional 1..N codes, never 0 (0 is the transparent background).
    assert sorted(series.tolist()) == [1, 2]


def test_rasterize_writes_instance_id_pixels() -> None:
    # Identity-ish affine: 1 um per pixel, no offset.
    napari_affine = napari_affine_from_px_to_um((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    inv_affine = np.linalg.inv(napari_affine)
    poly = _square(2.0, 3.0, 4.0)  # microns
    tile = rasterize_geometries_chunk(
        [poly],
        np.array([42], dtype=np.uint32),
        shape=(16, 16),
        inv_affine=inv_affine,
        dtype=np.uint32,
    )
    assert tile.dtype == np.uint32
    assert set(np.unique(tile).tolist()) == {0, 42}
    # The painted region sits at the polygon's micron -> pixel location.
    ys, xs = np.where(tile == 42)
    assert ys.min() >= 3 and xs.min() >= 2
