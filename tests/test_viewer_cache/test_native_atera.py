"""Viewer caches of a native (vendor-segmentation) ATERA latest store.

The base masks of ``atera_cell_boundaries`` and ``atera_nucleus`` are the
vendor ``cell_labels`` and ``nucleus_labels`` (their values are the shapes'
instance ids): they are aliased, not rasterized, and the label pyramids and
outlines are built from them. No transcript cache is written.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import spatialdata as sd
import zarr
from spatialdata.transformations import get_transformation
from test_io.atera_native_fixture import make_native_reference

from merxen.viewer_cache import build as build_module
from merxen.viewer_cache import format as fmt
from merxen.viewer_cache.build import (
    VENDOR_LABEL_SOURCE,
    ViewerCacheParams,
    build_viewer_caches,
    vendor_label_keys,
)
from merxen.viewer_cache.transform import resolve_mask_affine

PARAMS = ViewerCacheParams(min_size=16, base_image_key="morphology_focus")
VENDOR = {
    "atera_cell_boundaries": "cell_labels",
    "atera_nucleus": "nucleus_labels",
}


@pytest.fixture(scope="module")
def reference_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _, latest = make_native_reference(tmp_path_factory.mktemp("native_ref"))
    return latest


@pytest.fixture
def latest(reference_store: Path, tmp_path: Path) -> Iterator[Path]:
    store = tmp_path / "latest_spatialdata.zarr"
    shutil.copytree(reference_store, store)
    yield store


def _files(root: Path) -> dict[str, tuple[int, int, bytes]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_ino,
            path.stat().st_size,
            path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _labels(store: Path, key: str) -> np.ndarray:
    return np.asarray(zarr.open_array(str(store / "labels" / key / "s0"), mode="r"))


def test_vendor_labels_become_the_base_masks(latest: Path) -> None:
    summary = build_viewer_caches(latest, "ATERA", latest, params=PARAMS)

    sdata = sd.read_zarr(latest)
    for shape_key, vendor_key in VENDOR.items():
        label_key = f"{shape_key}_labels"
        result = summary["masks"][shape_key]
        assert result["mask"] == "vendor_hardlink"
        assert result["source_label_key"] == vendor_key
        assert result["label_pyramid"] == "built"
        assert result["outline"] == "built"
        # Exactly the vendor labels, whose values are the shapes' instance ids.
        base = _labels(latest, label_key)
        np.testing.assert_array_equal(base, _labels(latest, vendor_key))
        ids = set(np.unique(base).tolist()) - {0}
        assert ids
        assert ids <= set(sdata.shapes[shape_key]["instance_id"].tolist())
        # Every scale level is aliased and the alias reads back through spatialdata.
        assert label_key in sdata.labels
        vendor_levels = sorted(
            p.name for p in (latest / "labels" / vendor_key).iterdir()
        )
        alias_levels = sorted(p.name for p in (latest / "labels" / label_key).iterdir())
        assert alias_levels == vendor_levels
        # The derived caches carry the vendor element's transform (global = pixels).
        vendor_transform = get_transformation(sdata.labels[vendor_key], get_all=True)
        for cache_key in (
            fmt.derived_label_pyramid_cache_key(label_key, PARAMS.downsample),
            fmt.derived_outline_cache_key(label_key, 1),
        ):
            assert cache_key in sdata.labels
            assert get_transformation(sdata.labels[cache_key], get_all=True) == (
                vendor_transform
            )
        marker = dict(
            zarr.open_group(str(latest / "labels" / label_key), mode="r").attrs[
                fmt.LABEL_CACHE_ATTR
            ]
        )
        assert marker["complete"] is True
        assert marker["version"] == fmt.LABEL_CACHE_VERSION
        assert marker["source_shape_key"] == shape_key
        assert marker["source"] == VENDOR_LABEL_SOURCE
        assert marker["source_label_key"] == vendor_key
        assert marker["link_mode"] == "hardlink"
        assert marker["shape"] == list(base.shape)
    # The vendor elements themselves are unchanged and keep their attrs.
    for vendor_key in VENDOR.values():
        attrs = zarr.open_group(str(latest / "labels" / vendor_key), mode="r").attrs
        assert fmt.LABEL_CACHE_ATTR not in attrs


def test_alias_shares_payload_inodes_but_not_metadata(latest: Path) -> None:
    before = _files(latest / "labels" / "cell_labels")

    build_viewer_caches(latest, "ATERA", latest, params=PARAMS)

    vendor = latest / "labels" / "cell_labels"
    alias = latest / "labels" / "atera_cell_boundaries_labels"
    assert _files(vendor) == before
    for relative, (inode, _, _) in before.items():
        alias_file = alias / relative
        if Path(relative).name == "zarr.json":
            assert alias_file.stat().st_ino != inode
        else:
            assert alias_file.stat().st_ino == inode


def test_points_are_untouched(latest: Path) -> None:
    points = latest / "points" / "transcripts"
    before = _files(points)

    build_viewer_caches(latest, "ATERA", latest, params=PARAMS)

    assert _files(points) == before
    assert sorted(p.name for p in (latest / "points").iterdir() if p.is_dir()) == [
        "transcripts"
    ]


def test_second_build_skips_and_force_rebuilds(latest: Path) -> None:
    build_viewer_caches(latest, "ATERA", latest, params=PARAMS)
    vendor_before = _files(latest / "labels" / "cell_labels")

    again = build_viewer_caches(latest, "ATERA", latest, params=PARAMS)
    assert {r["mask"] for r in again["masks"].values()} == {"skipped"}
    assert {r["label_pyramid"] for r in again["masks"].values()} == {"skipped"}

    forced = build_viewer_caches(
        latest,
        "ATERA",
        latest,
        params=ViewerCacheParams(
            min_size=16, base_image_key="morphology_focus", force=True
        ),
    )
    assert {r["mask"] for r in forced["masks"].values()} == {"vendor_hardlink"}
    assert _files(latest / "labels" / "cell_labels") == vendor_before
    np.testing.assert_array_equal(
        _labels(latest, "atera_cell_boundaries_labels"), _labels(latest, "cell_labels")
    )
    labels_attrs = dict(zarr.open_group(str(latest / "labels"), mode="r").attrs)
    names = labels_attrs["ome"]["labels"]
    assert len(names) == len(set(names))
    assert not any(p.name.startswith(".") for p in (latest / "labels").iterdir())


def test_alias_copies_when_links_fail(
    latest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(build_module, "_hardlink", refuse)
    summary = build_viewer_caches(latest, "ATERA", latest, params=PARAMS)

    assert summary["masks"]["atera_cell_boundaries"]["mask"] == "vendor_copy"
    np.testing.assert_array_equal(
        _labels(latest, "atera_cell_boundaries_labels"), _labels(latest, "cell_labels")
    )
    alias_chunk = next(
        p
        for p in (latest / "labels" / "atera_cell_boundaries_labels" / "s0").rglob("*")
        if p.is_file() and p.name != "zarr.json"
    )
    assert alias_chunk.stat().st_nlink == 1


def test_base_image_key_is_honoured(latest: Path) -> None:
    sdata = sd.read_zarr(latest)
    key, _, _, _, _ = build_module._resolve_base_image(sdata, "morphology_focus")
    assert key == "morphology_focus"
    with pytest.raises(KeyError, match="base_image_key"):
        build_module._resolve_base_image(sdata, "missing_image")
    with pytest.raises(KeyError, match="base_image_key"):
        build_viewer_caches(
            latest,
            "ATERA",
            latest,
            params=ViewerCacheParams(min_size=16, base_image_key="missing_image"),
        )


def test_atera_pixel_size_comes_from_the_store(latest: Path, tmp_path: Path) -> None:
    x_row, y_row = resolve_mask_affine("ATERA", latest)
    ps = 0.21249222
    assert x_row == pytest.approx((ps, 0.0, 0.0))
    assert y_row == pytest.approx((0.0, ps, 0.0))

    empty = tmp_path / "no_sidecars.zarr"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="refusing the fallback"):
        resolve_mask_affine("ATERA", empty)
    # XENIUM and MERSCOPE keep their fallback.
    assert resolve_mask_affine("XENIUM", empty) == (
        (0.2125, 0.0, 0.0),
        (0.0, 0.2125, 0.0),
    )


def test_vendor_label_keys_are_a_no_op_for_xenium_and_merscope(latest: Path) -> None:
    sdata = sd.read_zarr(latest)
    assert vendor_label_keys(sdata, "ATERA") == VENDOR
    assert vendor_label_keys(sdata, "XENIUM") == {}
    assert vendor_label_keys(sdata, "MERSCOPE") == {}


def test_link_count_of_payloads(latest: Path) -> None:
    build_viewer_caches(latest, "ATERA", latest, params=PARAMS)
    chunk = next(
        p
        for p in (latest / "labels" / "cell_labels" / "s0").rglob("*")
        if p.is_file() and p.name != "zarr.json"
    )
    assert os.stat(chunk).st_nlink == 2
