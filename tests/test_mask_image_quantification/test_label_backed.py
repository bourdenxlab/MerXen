"""MIQ with a label element of the store as the mask (vendor cells, native ATERA)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
import zarr
from spatialdata.models import Image2DModel
from test_io.atera_native_fixture import CELL_SHAPES_KEY, make_native_reference

import merxen.mask_image_quantification as miq
from merxen.config import MaskImageQuantificationConfig

TABLE = f"table_{CELL_SHAPES_KEY}_image_quantification"


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return make_native_reference(tmp_path_factory.mktemp("miq_native_ref"))


def _copy(latest: Path, destination: Path) -> Path:
    shutil.copytree(latest, destination)
    return destination


def _config(
    latest: Path, output_dir: Path, **fields: object
) -> MaskImageQuantificationConfig:
    base: dict[str, object] = {
        "dataset_name": "ATERA_SYNTH",
        "platform": "ATERA",
        "latest_zarr_path": latest,
        "output_dir": output_dir,
        "shape_key": CELL_SHAPES_KEY,
        "table_key": TABLE,
        "tile_size": 32,
    }
    base.update(fields)
    return MaskImageQuantificationConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tiled label counts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64, np.int32])
@pytest.mark.parametrize("tile_size", [1, 7, 64, 4096])
def test_tiled_label_counts_equal_bincount(dtype: type, tile_size: int) -> None:
    rng = np.random.default_rng(11)
    mask = rng.integers(0, 40, size=(53, 37)).astype(dtype)
    mask[mask > 30] = 0
    expected_ids, expected_counts = miq._foreground_label_counts(mask)
    for lazy in (da.from_array(mask, chunks=(16, 9)), zarr.array(mask)):
        ids, counts = miq._foreground_label_counts_tiled(lazy, tile_size=tile_size)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(counts, expected_counts)


def test_tiled_label_counts_refuse_bad_masks() -> None:
    with pytest.raises(ValueError, match="2D"):
        miq._foreground_label_counts_tiled(da.zeros((2, 2, 2)), tile_size=4)
    with pytest.raises(ValueError, match="non-negative"):
        miq._foreground_label_counts_tiled(
            da.from_array(np.array([[0, -1]], dtype=np.int32)), tile_size=4
        )


def test_tiled_counts_never_load_the_whole_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every read is at most one tile; no full-size int64 copy is made."""
    mask = da.from_array(np.arange(100 * 90, dtype=np.uint32).reshape(100, 90) % 7)
    sizes: list[int] = []
    original = miq._read_mask_tile

    def _spy(array: object, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        tile = original(array, y0, y1, x0, x1)
        sizes.append(tile.size)
        return tile

    monkeypatch.setattr(miq, "_read_mask_tile", _spy)
    miq._foreground_label_counts_tiled(mask, tile_size=32)
    assert max(sizes) <= 32 * 32


@pytest.mark.parametrize(
    ("chunks", "tile_size", "expected"),
    [
        ((4096, 4096), 2048, 4096),
        ((4096, 4096), 4096, 4096),
        ((4096, 4096), 5000, 8192),
        ((16, 24), 20, 48),
        ((97, 67), 32, 97),
    ],
)
def test_lazy_mask_tiles_are_whole_chunks(
    chunks: tuple[int, int], tile_size: int, expected: int
) -> None:
    """A tile never splits a label chunk, so each chunk is decoded once a pass."""
    shape = (2 * chunks[0], 2 * chunks[1])
    as_dask = da.zeros(shape, dtype=np.uint32, chunks=chunks)
    as_zarr = zarr.zeros(shape, dtype=np.uint32, chunks=chunks)
    for mask in (as_dask, as_zarr):
        assert miq._chunk_aligned_tile_size(mask, tile_size) == expected
    assert miq._chunk_aligned_tile_size(np.zeros((4, 4)), 3) == 3


def test_lazy_mask_reads_follow_the_label_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiles cover whole label chunks, so no chunk is split between reads."""
    labels = np.zeros((64, 48), dtype=np.uint32)
    labels[3:40, 5:30] = 1
    labels[41:60, 31:47] = 2
    mask = da.from_array(labels, chunks=(32, 24))
    image = np.arange(64 * 48, dtype=np.float32).reshape(1, 64, 48)
    sdata = SimpleNamespace(
        images={"morphology_focus": Image2DModel.parse(image, dims=("c", "y", "x"))},
        tables={},
    )
    reads: list[tuple[int, int]] = []
    original = miq._read_mask_tile

    def _spy(array: object, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        reads.append((y1 - y0, x1 - x0))
        return original(array, y0, y1, x0, x1)

    monkeypatch.setattr(miq, "_read_mask_tile", _spy)
    result = miq.build_mask_image_quantification_table(
        sdata, mask, "HA1", tile_size=8, mask_name="vendor 'cell_labels'"
    )
    assert result.summary["tile_size_px"] == 24 * 4  # lcm(32, 24) = 96
    assert reads == [(64, 48)]
    assert result.table.obs["mask_pixel_count"].tolist() == [
        int((labels == 1).sum()),
        int((labels == 2).sum()),
    ]


def test_an_empty_vendor_mask_names_the_label_element() -> None:
    mask = da.zeros((8, 8), dtype=np.uint32, chunks=(4, 4))
    with pytest.raises(ValueError, match="vendor 'cell_labels' mask contains no"):
        miq.build_mask_image_quantification_table(
            SimpleNamespace(images={}, tables={}),
            mask,
            "HA1",
            mask_name="vendor 'cell_labels'",
        )


# --------------------------------------------------------------------------- #
# Label-backed against .npy-backed on the native fixture
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_label_backed_quantification_equals_npy_backed(
    reference: tuple[Path, Path], tmp_path: Path
) -> None:
    _, latest = reference
    labels_store = _copy(latest, tmp_path / "labels.zarr")
    npy_store = _copy(latest, tmp_path / "npy.zarr")
    sdata = sd.read_zarr(latest)
    mask = np.asarray(sd.get_pyramid_levels(sdata.labels["cell_labels"], n=0))
    mask_path = tmp_path / "cell_labels.npy"
    np.save(mask_path, mask)

    miq.run_mask_image_quantification(
        _config(
            labels_store,
            tmp_path / "out_labels",
            mask_path=None,
            mask_label_key="cell_labels",
        )
    )
    miq.run_mask_image_quantification(
        _config(npy_store, tmp_path / "out_npy", mask_path=mask_path)
    )

    from_labels = sd.read_zarr(labels_store).tables[TABLE]
    from_npy = sd.read_zarr(npy_store).tables[TABLE]
    np.testing.assert_array_equal(np.asarray(from_labels.X), np.asarray(from_npy.X))
    pd.testing.assert_frame_equal(from_labels.var, from_npy.var)
    same = ["instance_id", "label_id", "mask_pixel_count", "region"]
    pd.testing.assert_frame_equal(from_labels.obs[same], from_npy.obs[same])

    # Rows are the vendor cells, named by their vendor source_cell_id.
    cells = sdata.shapes[CELL_SHAPES_KEY]
    assert from_labels.n_obs == len(cells)
    expected_ids = dict(
        zip(cells["instance_id"].astype(int), cells["source_cell_id"], strict=True)
    )
    assert from_labels.obs["source_cell_id"].tolist() == [
        expected_ids[int(i)] for i in from_labels.obs["instance_id"]
    ]
    assert from_npy.obs["source_cell_id"].str.startswith("cellpose_").all()
    assert from_labels.uns["mask_image_quantification"]["mask_label_key"] == (
        "cell_labels"
    )
    assert "mask_label_key" not in from_npy.uns["mask_image_quantification"]
    assert from_labels.uns["spatialdata_attrs"]["region"] == CELL_SHAPES_KEY

    summary = json.loads(
        (
            tmp_path
            / "out_labels"
            / "atera_synth_mask_image_quantification_summary.json"
        ).read_text()
    )
    assert summary["n_cells"] == len(cells)
    assert summary["mask_label_key"] == "cell_labels"
    assert summary["hybrid_join"]["status"] == "skipped"


@pytest.mark.integration
def test_label_backed_quantification_is_reused(
    reference: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, latest = reference
    store = _copy(latest, tmp_path / "store.zarr")
    config = _config(
        store, tmp_path / "out", mask_path=None, mask_label_key="cell_labels"
    )
    miq.run_mask_image_quantification(config)

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("must reuse the existing quantification")

    monkeypatch.setattr(miq, "build_mask_image_quantification_table", _fail)
    outputs = miq.run_mask_image_quantification(config)
    assert outputs["latest_zarr"] == store


@pytest.mark.integration
def test_a_missing_label_element_is_reported(
    reference: tuple[Path, Path], tmp_path: Path
) -> None:
    _, latest = reference
    store = _copy(latest, tmp_path / "store.zarr")
    with pytest.raises(KeyError, match="no_such_labels"):
        miq.run_mask_image_quantification(
            _config(
                store, tmp_path / "out", mask_path=None, mask_label_key="no_such_labels"
            )
        )
