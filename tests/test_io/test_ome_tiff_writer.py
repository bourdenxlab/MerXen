"""Tests for the streaming pyramidal OME-TIFF writer."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from merxen.io import ome_tiff_writer
from merxen.io.ome_tiff import open_ome_tiff_level, read_ome_tiff_info
from merxen.io.ome_tiff_writer import (
    OmeTiffWriteResult,
    downsample_2x2,
    pyramid_level_shapes,
    write_pyramidal_ome_tiff,
)

REPO = Path(__file__).resolve().parents[2]


def array_reader(array: np.ndarray) -> Callable[[int, int, int], np.ndarray]:
    return lambda c, y0, h: array[c, y0 : y0 + h].copy()


def ref_levels(array: np.ndarray, n_sub: int) -> list[np.ndarray]:
    """Independent reference pyramid: edge pad, (sum + 2) // 4 in uint64, recursive."""
    levels = [array]
    for _ in range(n_sub):
        out = []
        for plane in levels[-1]:
            h, w = plane.shape
            padded = np.pad(plane, ((0, h % 2), (0, w % 2)), mode="edge")
            total = np.zeros((padded.shape[0] // 2, padded.shape[1] // 2), np.uint64)
            for dy in (0, 1):
                for dx in (0, 1):
                    total += padded[dy::2, dx::2].astype(np.uint64)
            out.append(((total + 2) // 4).astype(array.dtype))
        levels.append(np.stack(out))
    return levels


def make_array(shape: tuple[int, int, int], dtype: str = "uint16") -> np.ndarray:
    rng = np.random.default_rng(0)
    info = np.iinfo(dtype)
    return rng.integers(0, int(info.max) + 1, size=shape).astype(dtype)


def small_write(path: Path, array: np.ndarray, **kwargs: Any) -> OmeTiffWriteResult:
    options = {
        "tile_size": 64,
        "band_height": 128,
        "min_level_size": 64,
        "pixel_size_um": 0.5,
        "image_name": "fixture_moving",
        "workers": 2,
        "channel_names": [f"ch{i}" for i in range(array.shape[0])],
    }
    options.update(kwargs)
    return write_pyramidal_ome_tiff(
        array_reader(array),
        shape_cyx=array.shape,
        dtype=array.dtype,
        output_path=path,
        **options,
    )


def test_pyramid_level_shapes() -> None:
    big = pyramid_level_shapes((55012, 76544), min_level_size=1024)
    assert len(big) == 8 and big[-1] == (430, 598)
    big = pyramid_level_shapes((61003, 80112), min_level_size=1024)
    assert len(big) == 8 and big[-1] == (477, 626)
    assert pyramid_level_shapes((300, 500), min_level_size=64) == (
        (300, 500),
        (150, 250),
        (75, 125),
        (38, 63),
    )
    assert pyramid_level_shapes((257, 129), min_level_size=16) == (
        (257, 129),
        (129, 65),
        (65, 33),
        (33, 17),
        (17, 9),
        (9, 5),
    )
    assert pyramid_level_shapes((40, 40), min_level_size=64) == ((40, 40),)


def test_downsample_2x2_values() -> None:
    u16 = np.uint16
    assert downsample_2x2(np.array([[1, 2], [3, 5]], u16)).tolist() == [[3]]
    assert downsample_2x2(np.array([[0, 1], [1, 1]], u16)).tolist() == [[1]]
    odd = np.arange(9, dtype=u16).reshape(3, 3)
    assert downsample_2x2(odd).tolist() == [[2, 4], [7, 8]]
    assert downsample_2x2(np.array([[1, 2], [3, 5]], np.float32)).tolist() == [[2.75]]
    out = downsample_2x2(np.array([[-3, -2], [-1, -1]], np.int16))
    assert out.tolist() == [[-2]] and out.dtype == np.int16
    full = downsample_2x2(np.full((2, 2), 65535, u16))
    assert full.tolist() == [[65535]] and full.dtype == u16


def test_round_trip_uint16(tmp_path: Path) -> None:
    array = make_array((2, 300, 500))
    out = tmp_path / "out.ome.tif"
    result = small_write(out, array, channel_names=["DAPI", "AF568"])
    expected_shapes = ((300, 500), (150, 250), (75, 125), (38, 63))
    assert result.level_shapes_yx == expected_shapes
    assert result.shape_cyx == (2, 300, 500)
    assert result.bytes_written == out.stat().st_size

    info = read_ome_tiff_info(out)
    assert info.axes == "CYX"
    assert info.level_shapes_cyx == tuple((2, *s) for s in expected_shapes)
    assert info.channel_names == ("DAPI", "AF568")
    assert info.pixel_size_um == pytest.approx((0.5, 0.5))
    assert info.dtype == "uint16"
    for k, expected in enumerate(ref_levels(array, 3)):
        got = open_ome_tiff_level(info, k).values
        assert np.array_equal(got, expected), f"level {k}"

    with tifffile.TiffFile(out) as tif:
        assert tif.is_bigtiff
        page = tif.pages[0]
        assert page.is_tiled
        assert page.tilewidth == page.tilelength == 64
        assert page.compression == 8
        xml = tif.ome_metadata
    assert 'Name="fixture_moving"' in xml
    assert 'Creator="merxen ' in xml
    assert not list(tmp_path.glob("*.partial-*"))
    assert not list(tmp_path.glob(".ome-tiff-levels-*"))


def test_odd_shape_single_channel(tmp_path: Path) -> None:
    array = make_array((1, 257, 129))
    out = tmp_path / "odd.ome.tif"
    result = small_write(out, array, tile_size=16, band_height=32, min_level_size=16)
    assert len(result.level_shapes_yx) == 6
    info = read_ome_tiff_info(out)
    assert info.level_shapes_cyx[0] == (1, 257, 129)
    for k, expected in enumerate(ref_levels(array, 5)):
        assert np.array_equal(open_ome_tiff_level(info, k).values, expected)


def test_float32_levels(tmp_path: Path) -> None:
    array = np.random.default_rng(1).random((1, 64, 96)).astype(np.float32)
    out = tmp_path / "f.ome.tif"
    small_write(out, array, tile_size=16, band_height=32, min_level_size=16)
    info = read_ome_tiff_info(out)
    level = array[0].astype(np.float64)
    for k in range(len(info.level_shapes_cyx)):
        got = open_ome_tiff_level(info, k).values[0]
        assert got.dtype == np.float32
        assert np.array_equal(got, level.astype(np.float32))
        level = level.reshape(level.shape[0] // 2, 2, level.shape[1] // 2, 2).mean(
            axis=(1, 3)
        )
        level = level.astype(np.float32).astype(np.float64)


def test_single_level(tmp_path: Path) -> None:
    array = make_array((1, 40, 40))
    out = tmp_path / "one.ome.tif"
    result = small_write(out, array, tile_size=16, band_height=32, min_level_size=64)
    assert result.level_shapes_yx == ((40, 40),)
    assert read_ome_tiff_info(out).is_pyramidal is False


def test_deterministic_bytes(tmp_path: Path) -> None:
    array = make_array((2, 200, 300))
    a, b, c = (tmp_path / f"{n}.ome.tif" for n in "abc")
    small_write(a, array, workers=1)
    small_write(b, array, workers=3)
    small_write(c, array, workers=3)
    assert a.read_bytes() == b.read_bytes() == c.read_bytes()
    with tifffile.TiffFile(a) as tif:
        assert "UUID" in tif.ome_metadata
        assert "urn:uuid:" in tif.ome_metadata


def test_atomic_replace(tmp_path: Path) -> None:
    first = make_array((1, 100, 100))
    second = (first // 2).astype(np.uint16)
    out = tmp_path / "out.ome.tif"
    small_write(out, first)
    small_write(out, second)
    info = read_ome_tiff_info(out)
    assert np.array_equal(open_ome_tiff_level(info, 0).values, second)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.ome.tif"]


def test_failure_mid_write_leaves_nothing(tmp_path: Path) -> None:
    array = make_array((1, 300, 100))
    calls = []

    def failing(c: int, y0: int, h: int) -> np.ndarray:
        calls.append((c, y0))
        if len(calls) == 3:
            raise RuntimeError("boom")
        return array[c, y0 : y0 + h].copy()

    out = tmp_path / "out.ome.tif"
    scratch = tmp_path / "t"
    with pytest.raises(RuntimeError, match="boom"):
        write_pyramidal_ome_tiff(
            failing,
            shape_cyx=array.shape,
            dtype=array.dtype,
            channel_names=["a"],
            pixel_size_um=0.5,
            image_name="x",
            output_path=out,
            tile_size=16,
            band_height=64,
            min_level_size=16,
            tmp_dir=scratch,
        )
    assert not out.exists()
    assert not list(tmp_path.glob("*.partial-*"))
    assert list(scratch.iterdir()) == []


def test_wrong_band_contract(tmp_path: Path) -> None:
    array = make_array((1, 100, 100))
    with pytest.raises(RuntimeError, match=r"expected \(100, 100\) uint16"):
        write_pyramidal_ome_tiff(
            lambda c, y0, h: np.zeros((h, 99), np.uint16),
            shape_cyx=array.shape,
            dtype="uint16",
            channel_names=["a"],
            pixel_size_um=0.5,
            image_name="x",
            output_path=tmp_path / "o.ome.tif",
            tile_size=16,
            band_height=128,
            min_level_size=16,
        )
    with pytest.raises(RuntimeError, match="uint16"):
        write_pyramidal_ome_tiff(
            lambda c, y0, h: np.zeros((h, 100), np.uint8),
            shape_cyx=array.shape,
            dtype="uint16",
            channel_names=["a"],
            pixel_size_um=0.5,
            image_name="x",
            output_path=tmp_path / "o.ome.tif",
            tile_size=16,
            band_height=128,
            min_level_size=16,
        )
    assert not (tmp_path / "o.ome.tif").exists()


def test_disk_space_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ome_tiff_writer, "_free_bytes", lambda p: 0)
    out = tmp_path / "new" / "out.ome.tif"
    with pytest.raises(OSError, match="not enough free space") as caught:
        small_write(out, make_array((1, 64, 64)))
    assert caught.value.errno == errno.ENOSPC
    assert not (tmp_path / "new").exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channel_names": ["a"]}, "channel name"),
        ({"channel_names": ["DAPI", "dapi"]}, "unique"),
        ({"band_height": 100}, "multiple of tile_size"),
        ({"tile_size": 24}, "multiple of 16"),
        ({"output_path": "out.tif"}, r"\.ome\.tif"),
        ({"compression": "lzw"}, "compression"),
        ({"compression_level": 10}, "0-9"),
        ({"channel_metadata": [{"Color": 1}, {"Color": 2}]}, "Color"),
        ({"pixel_size_um": 0.0}, "pixel_size_um"),
        ({"workers": 0}, "workers"),
    ],
)
def test_validation_errors(tmp_path: Path, kwargs: dict, match: str) -> None:
    array = make_array((2, 64, 64))
    kwargs = dict(kwargs)
    name = kwargs.pop("output_path", "out.ome.tif")
    with pytest.raises(ValueError, match=match):
        small_write(tmp_path / name, array, **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_band_height_must_cover_levels(tmp_path: Path) -> None:
    array = make_array((1, 257, 129))
    with pytest.raises(ValueError, match="multiple of 2\\*\\*5"):
        small_write(
            tmp_path / "o.ome.tif",
            array,
            tile_size=16,
            band_height=16,
            min_level_size=16,
        )


def test_channel_metadata(tmp_path: Path) -> None:
    array = make_array((2, 64, 64))
    out = tmp_path / "m.ome.tif"
    small_write(
        out,
        array,
        channel_names=["DAPI", "AF568"],
        channel_metadata=[
            {"Fluor": "DAPI", "EmissionWavelength": 465.0},
            {"Fluor": "Alexa Fluor 568", "EmissionWavelength": None},
        ],
    )
    with tifffile.TiffFile(out) as tif:
        xml = tif.ome_metadata
    assert 'Fluor="Alexa Fluor 568"' in xml
    assert "EmissionWavelength" not in xml

    full = tmp_path / "full.ome.tif"
    small_write(
        full,
        array,
        channel_names=["DAPI", "AF568"],
        channel_metadata=[
            {"Fluor": "DAPI", "EmissionWavelength": 465.0},
            {"Fluor": "Alexa Fluor 568", "EmissionWavelength": 603.0},
        ],
    )
    with tifffile.TiffFile(full) as tif:
        xml = tif.ome_metadata
    assert 'EmissionWavelength="465.0"' in xml


def test_compression_choices(tmp_path: Path) -> None:
    array = make_array((1, 64, 64))
    none = tmp_path / "n.ome.tif"
    small_write(none, array, compression="none")
    with tifffile.TiffFile(none) as tif:
        assert tif.pages[0].compression == 1
    zstd = tmp_path / "z.ome.tif"
    small_write(zstd, array, compression="zstd", compression_level=3)
    with tifffile.TiffFile(zstd) as tif:
        assert tif.pages[0].compression == 50000
    info = read_ome_tiff_info(zstd)
    assert np.array_equal(open_ome_tiff_level(info, 0).values, array)


def test_import_isolation() -> None:
    code = textwrap.dedent(
        """
        import sys
        BLOCKED = {"pydantic", "click", "dask", "xarray", "zarr", "ome_types",
                   "scipy", "pandas", "anndata", "spatialdata", "squidpy",
                   "torch", "czifile"}

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked import of {name}")
                return None

        sys.meta_path.insert(0, Blocker())
        import merxen.io.ome_tiff_writer
        import merxen.io.czi
        """
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
