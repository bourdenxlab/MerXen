"""Image-only viewer-cache mode: pyramids of the base images of a prepared store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale, get_transformation

from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.io.spatialdata_schema import stamp_merxen_schema
from merxen.viewer_cache import build as build_module
from merxen.viewer_cache.build import (
    ViewerCacheParams,
    build_image_pyramid_caches,
    build_image_pyramids_in_store,
)
from merxen.viewer_cache.format import DERIVED_CACHE_ATTR

PARAMS = ViewerCacheParams(min_size=256)
REFERENCE = np.random.default_rng(0).integers(0, 4000, (2, 1300, 1100), dtype=np.uint16)


def _parse(
    array: np.ndarray, names: list[str], pixel_size_um: float, **kwargs: Any
) -> Any:
    return Image2DModel.parse(
        array,
        dims=("c", "y", "x"),
        c_coords=names,
        transformations={"global": Scale([pixel_size_um] * 2, axes=("x", "y"))},
        **kwargs,
    )


@pytest.fixture
def image_only_store(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    sdata = SpatialData(
        images={
            "reference": _parse(REFERENCE, ["DAPI", "AF568"], 0.5),
            "moving": _parse(
                rng.integers(0, 4000, (3, 1200, 1250), dtype=np.uint16),
                ["DAPI", "AF568", "AF647"],
                0.4,
            ),
            "multi": _parse(
                rng.integers(0, 4000, (1, 600, 600), dtype=np.uint16),
                ["DAPI"],
                0.5,
                scale_factors=[2],
            ),
        }
    )
    sdata.attrs["merxen_manual_registration"] = {"revision": 3}
    stamp_merxen_schema(sdata)
    store = tmp_path / "store.zarr"
    sdata.write(store)
    return store


def _root_attributes(store: Path) -> dict[str, Any]:
    return json.loads((store / "zarr.json").read_text())["attributes"]


def _cache_key(image_key: str) -> str:
    return f"_napari_compare_imgpyr__{image_key}__ds4"


def test_builds_every_base_image_in_sorted_order(image_only_store: Path) -> None:
    result = build_image_pyramid_caches(image_only_store, params=PARAMS)

    assert list(result["image_pyramids"].items()) == [
        ("moving", "built"),
        ("multi", "stored_multiscale"),
        ("reference", "built"),
    ]
    for key in ("moving", "reference"):
        marker = zarr.open_group(
            str(image_only_store / "images" / _cache_key(key)), mode="r"
        ).attrs[DERIVED_CACHE_ATTR]
        assert marker["kind"] == "image_pyramid"
        assert marker["source_image_key"] == key
        assert marker["downsample"] == 4
        assert marker["min_size"] == 256

    sdata = read_zarr(image_only_store)
    cache = sdata.images[_cache_key("reference")]
    level0 = np.asarray(cache["scale0"]["image"].values)
    expected = np.rint(REFERENCE[:, :1300].reshape(2, 325, 4, 275, 4).mean((2, 4)))
    assert level0.shape == (2, 325, 275)
    np.testing.assert_array_equal(level0, expected.astype(np.uint16))
    transform = get_transformation(cache, "global")
    base = get_transformation(sdata.images["reference"], "global")
    np.testing.assert_allclose(
        transform.to_affine_matrix(("x", "y"), ("x", "y"))[:2, :2],
        base.to_affine_matrix(("x", "y"), ("x", "y"))[:2, :2],
    )


def test_root_attributes_are_untouched(image_only_store: Path) -> None:
    before = _root_attributes(image_only_store)
    build_image_pyramid_caches(image_only_store, params=PARAMS)
    assert _root_attributes(image_only_store) == before
    build_image_pyramid_caches(
        image_only_store, params=ViewerCacheParams(min_size=256, force=True)
    )
    assert _root_attributes(image_only_store) == before
    assert before["merxen_manual_registration"] == {"revision": 3}


def test_restrict_to_image_keys_and_reject_unknown(image_only_store: Path) -> None:
    result = build_image_pyramid_caches(
        image_only_store, image_keys=["reference"], params=PARAMS
    )
    assert result == {"image_pyramids": {"reference": "built"}}

    with pytest.raises(ValueError, match=r"\['moving', 'multi', 'reference'\]"):
        build_image_pyramid_caches(image_only_store, image_keys=["nope"], params=PARAMS)
    with pytest.raises(ValueError, match="not base image elements"):
        build_image_pyramid_caches(
            image_only_store, image_keys=[_cache_key("reference")], params=PARAMS
        )


def _cache_arrays(store: Path, key: str) -> list[np.ndarray]:
    cache = read_zarr(store).images[_cache_key(key)]
    return [np.asarray(cache[name]["image"].values) for name in sorted(cache.children)]


def test_rerun_skips_and_force_rebuilds(image_only_store: Path) -> None:
    build_image_pyramid_caches(image_only_store, params=PARAMS)
    first = _cache_arrays(image_only_store, "reference")

    second = build_image_pyramid_caches(image_only_store, params=PARAMS)
    assert second["image_pyramids"] == {
        "moving": "skipped",
        "multi": "stored_multiscale",
        "reference": "skipped",
    }

    forced = build_image_pyramid_caches(
        image_only_store, params=ViewerCacheParams(min_size=256, force=True)
    )
    assert forced["image_pyramids"]["moving"] == "built"
    assert forced["image_pyramids"]["reference"] == "built"
    again = _cache_arrays(image_only_store, "reference")
    assert len(first) == len(again)
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)


def test_disabled_reports_without_writing(image_only_store: Path) -> None:
    result = build_image_pyramid_caches(
        image_only_store, params=ViewerCacheParams(build_image_pyramid=False)
    )
    assert result["image_pyramids"] == {
        "moving": "disabled",
        "multi": "disabled",
        "reference": "disabled",
    }
    keys = [str(k) for k in read_zarr(image_only_store).images]
    assert not any(k.startswith("_napari") for k in keys)


def test_in_store_helper_does_not_take_the_lock(image_only_store: Path) -> None:
    with spatialdata_write_lock(image_only_store):
        statuses = build_image_pyramids_in_store(
            read_zarr(image_only_store, selection=("images",)),
            image_only_store,
            params=PARAMS,
        )
    assert statuses["reference"] == "built"


def test_image_only_build_serializes_with_another_store_writer(
    image_only_store: Path, store_lock_contention: Any
) -> None:
    result = store_lock_contention.run(
        lambda staged: build_image_pyramid_caches(staged, params=PARAMS),
        image_only_store,
        reads=[(build_module, "read_zarr")],
        persists=[
            (build_module, "write_or_replace_element"),
            (build_module, "_stamp_marker"),
        ],
    )
    assert result["image_pyramids"]["moving"] == "built"
    assert result["image_pyramids"]["reference"] == "built"
