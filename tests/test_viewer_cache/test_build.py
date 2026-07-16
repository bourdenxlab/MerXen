"""End-to-end build test on a small synthetic SpatialData store.

Builds every artifact and asserts the on-disk keys + completion markers match
the exact format the viewer trusts, that the mask carries true instance ids, and
that the store still reloads through SpatialData.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
import zarr
from shapely.geometry import Polygon
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.transformations import Identity

from merxen.viewer_cache import format as fmt
from merxen.viewer_cache.build import ViewerCacheParams, build_viewer_caches


def _square(cx: float, cy: float, s: float) -> Polygon:
    return Polygon([(cx, cy), (cx + s, cy), (cx + s, cy + s), (cx, cy + s)])


@pytest.fixture
def synthetic_store(tmp_path: Path) -> Path:
    store = tmp_path / "synth.zarr"
    rng = np.random.default_rng(0)
    img = rng.integers(0, 500, size=(1, 160, 160), dtype=np.uint16)
    image = Image2DModel.parse(
        xr.DataArray(img, dims=("c", "y", "x")),
        transformations={"global": Identity()},
    )
    gdf = gpd.GeoDataFrame(
        geometry=[
            _square(1.0, 1.0, 3.0),
            _square(6.0, 6.0, 4.0),
            _square(11.0, 2.0, 2.0),
        ],
        index=[2, 94831, 7],  # non-contiguous integer instance ids
    )
    shapes = ShapesModel.parse(gdf)
    sd = SpatialData(
        images={"MERSCOPE_z_projection": image},
        shapes={"MOSAIK_proseg": shapes},
    )
    sd.write(store)
    return store


def _marker(store: Path, group: str, key: str, attr: str) -> dict:
    grp = zarr.open_group(str(store / group / key), mode="r")
    return dict(grp.attrs.get(attr, {}))


def test_build_produces_viewer_trusted_caches(synthetic_store: Path) -> None:
    params = ViewerCacheParams(
        downsample=4,
        label_chunk_size=64,
        contour_width=1,
        min_size=32,
        shape_keys=("MOSAIK_proseg",),
        build_image_pyramid=True,
    )
    summary = build_viewer_caches(
        synthetic_store,
        "MERSCOPE",
        original_data_path=synthetic_store,  # no transform file -> 0.108 fallback
        params=params,
    )
    assert summary["masks"]["MOSAIK_proseg"]["mask"] == "built"

    # Base mask: exact LABEL_CACHE marker + true instance ids + correct dtype.
    lm = _marker(
        synthetic_store, "labels", "MOSAIK_proseg_labels", fmt.LABEL_CACHE_ATTR
    )
    assert lm["version"] == fmt.LABEL_CACHE_VERSION
    assert lm["complete"] is True
    assert lm["source_shape_key"] == "MOSAIK_proseg"
    assert lm["shape"] == [160, 160]

    s0 = zarr.open(
        str(synthetic_store / "labels" / "MOSAIK_proseg_labels" / "s0"), mode="r"
    )[:]
    assert s0.dtype == np.uint32
    present = set(np.unique(s0).tolist()) - {0}
    assert present.issubset({2, 94831, 7})
    assert 94831 in present  # large instance id survived rasterization

    # Derived caches exist under their exact keys with matching markers.
    lp_key = fmt.derived_label_pyramid_cache_key("MOSAIK_proseg_labels", 4)
    lp = _marker(synthetic_store, "labels", lp_key, fmt.DERIVED_CACHE_ATTR)
    assert lp["kind"] == "label_pyramid"
    assert lp["version"] == fmt.VIEWER_DERIVED_CACHE_VERSION
    assert lp["downsample"] == 4

    ol_key = fmt.derived_outline_cache_key("MOSAIK_proseg_labels", 1)
    ol = _marker(synthetic_store, "labels", ol_key, fmt.DERIVED_CACHE_ATTR)
    assert ol["kind"] == "label_outline"
    assert ol["width"] == 1

    im_key = fmt.derived_image_pyramid_cache_key("MERSCOPE_z_projection", 4)
    im = _marker(synthetic_store, "images", im_key, fmt.DERIVED_CACHE_ATTR)
    assert im["kind"] == "image_pyramid"
    assert im["downsample"] == 4

    # Store still reloads with every element present.
    sd = read_zarr(synthetic_store)
    assert "MOSAIK_proseg_labels" in sd.labels
    assert lp_key in sd.labels
    assert ol_key in sd.labels
    assert im_key in sd.images


def test_build_is_idempotent(synthetic_store: Path) -> None:
    params = ViewerCacheParams(
        label_chunk_size=64, min_size=32, shape_keys=("MOSAIK_proseg",)
    )
    build_viewer_caches(
        synthetic_store, "MERSCOPE", original_data_path=synthetic_store, params=params
    )
    # A second run with matching markers skips every artifact.
    summary = build_viewer_caches(
        synthetic_store, "MERSCOPE", original_data_path=synthetic_store, params=params
    )
    assert summary["masks"]["MOSAIK_proseg"]["mask"] == "skipped"
    assert summary["masks"]["MOSAIK_proseg"]["label_pyramid"] == "skipped"
    assert summary["masks"]["MOSAIK_proseg"]["outline"] == "skipped"
    assert summary["image_pyramid"] == "skipped"
