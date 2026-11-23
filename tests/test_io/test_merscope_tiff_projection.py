"""Windowed TIFF decoding and lossless, verified MERSCOPE max projection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from merxen.io.builders import merscope


@pytest.mark.parametrize("compressed", [False, True])
def test_tiff_windows_preserve_pixels_and_edge_chunks(
    tmp_path: Path, compressed: bool
) -> None:
    expected = np.arange(83 * 107, dtype=np.uint16).reshape(83, 107)
    path = tmp_path / "mosaic_DAPI_z1.tif"
    tifffile.imwrite(
        path,
        expected,
        tile=(32, 32) if compressed else None,
        compression="zlib" if compressed else None,
    )
    before = path.read_bytes()
    actual = merscope._open_merscope_channel_plane_array(
        images_dir=tmp_path, stain="DAPI", z_layer=1, chunks=(19, 23)
    )
    np.testing.assert_array_equal(actual.compute(scheduler="single-threaded"), expected)
    np.testing.assert_array_equal(actual[71:, 92:].compute(), expected[71:, 92:])
    assert path.read_bytes() == before


def test_sparse_bigtiff_window_beyond_four_gigabytes_is_not_loaded_whole(
    tmp_path: Path,
) -> None:
    """A sparse fixture exercises real 64-bit TIFF offsets without allocating 6 GB."""
    path = tmp_path / "mosaic_DAPI_z1.tif"
    image = tifffile.memmap(path, shape=(50000, 60000), dtype=np.uint16, bigtiff=True)
    image[45000:45004, 50000:50008] = 12345
    image.flush()
    del image
    actual = merscope._open_merscope_channel_plane_array(
        images_dir=tmp_path, stain="DAPI", z_layer=1, chunks=(128, 128)
    )
    assert actual.shape == (50000, 60000)
    np.testing.assert_array_equal(
        actual[45000:45004, 50000:50008].compute(),
        np.full((4, 8), 12345, dtype=np.uint16),
    )


def _planes(path: Path) -> np.ndarray:
    rng = np.random.default_rng(9)
    values = rng.integers(0, 60000, (2, 3, 9, 13), dtype=np.uint16)
    for c, stain in enumerate(("DAPI", "PolyT")):
        for z in (1, 2, 3):
            tifffile.imwrite(path / f"mosaic_{stain}_z{z}.tif", values[c, z - 1])
    return values


def test_projection_equals_raw_maximum_including_partial_chunks(tmp_path: Path) -> None:
    values = _planes(tmp_path)
    result = merscope._build_iterative_projection_store(
        images_dir=tmp_path,
        stainings=["DAPI", "PolyT"],
        z_layers=[1, 2, 3],
        image_models_kwargs={"chunks": (1, 4, 5)},
        temp_root=tmp_path,
    )
    np.testing.assert_array_equal(result.image.data.compute(), values.max(axis=1))


def test_projection_refuses_a_corrupt_write_and_removes_partial_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _planes(tmp_path)
    original = merscope.zarr.open_array

    class CorruptWrite:
        def __init__(self, array: Any) -> None:
            self.array = array
            self.dtype = array.dtype

        def __setitem__(self, key: Any, value: np.ndarray) -> None:
            self.array[key] = np.zeros_like(value)

        def __getitem__(self, key: Any) -> np.ndarray:
            return np.asarray(self.array[key])

    def open_array(*args: Any, **kwargs: Any) -> Any:
        array = original(*args, **kwargs)
        return CorruptWrite(array) if kwargs.get("mode") == "w" else array

    monkeypatch.setattr(merscope.zarr, "open_array", open_array)
    with pytest.raises(OSError, match="projection read-back differs"):
        merscope._build_iterative_projection_store(
            images_dir=tmp_path,
            stainings=["DAPI", "PolyT"],
            z_layers=[1, 2, 3],
            image_models_kwargs={"chunks": (1, 4, 5)},
            temp_root=tmp_path,
        )
    assert list(tmp_path.glob("merscope_projection_*")) == []


def test_reader_refuses_multiplane_file_in_one_plane_slot(tmp_path: Path) -> None:
    tifffile.imwrite(
        tmp_path / "mosaic_DAPI_z1.tif",
        np.ones((2, 9, 13), dtype=np.uint16),
        photometric="minisblack",
    )
    with pytest.raises(ValueError, match="one 2D TIFF plane"):
        merscope._open_merscope_channel_plane_array(
            images_dir=tmp_path, stain="DAPI", z_layer=1, chunks=(4, 5)
        )
