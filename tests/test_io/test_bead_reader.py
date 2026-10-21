"""Tests of the memory-mapped mosaic plane reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from merxen.io.bead_removal import MosaicPlaneReader, list_mosaic_planes

tifffile = pytest.importorskip("tifffile")

STAINS = ("PolyT", "DAPI")


def _write_planes(
    folder: Path,
    stains: tuple[str, ...] = STAINS,
    planes: tuple[int, ...] = (0, 1),
    shape: tuple[int, int] = (100, 120),
    seed: int = 0,
) -> dict[tuple[str, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    written = {}
    for stain in stains:
        for z in planes:
            data = rng.integers(1, 5000, shape, dtype=np.uint16)
            tifffile.imwrite(folder / f"mosaic_{stain}_z{z}.tif", data)
            written[(stain, z)] = data
    return written


def test_zero_padding_and_window_content(tmp_path: Path) -> None:
    planes = _write_planes(tmp_path)
    with MosaicPlaneReader(tmp_path, ["PolyT", "DAPI"], [0, 1]) as reader:
        assert reader.shape == (100, 120)
        window = reader.read("PolyT", 1, -10, 20, -5, 10)
        assert window.shape == (30, 15)
        assert window.dtype == np.float32
        # Everything outside the mosaic is the invalid value 0.
        assert (window[:10] == 0).all()
        assert (window[:, :5] == 0).all()
        assert np.array_equal(window[10:, 5:], planes[("PolyT", 1)][0:20, 0:10])
        assert (reader.read("PolyT", 0, 200, 210, 0, 10) == 0).all()
        # A window beyond the far edges is cut and padded the same way.
        far = reader.read("DAPI", 0, 90, 110, 110, 130)
        assert far.shape == (20, 20)
        assert np.array_equal(far[:10, :10], planes[("DAPI", 0)][90:100, 110:120])
        assert (far[10:] == 0).all()
        assert (far[:, 10:] == 0).all()


def test_read_stack_returns_one_plane_per_z(tmp_path: Path) -> None:
    planes = _write_planes(tmp_path)
    with MosaicPlaneReader(tmp_path, ["PolyT"], [0, 1]) as reader:
        stack = reader.read_stack("PolyT", [1, 0], 5, 15, 5, 25)
        assert set(stack) == {0, 1}
        assert np.array_equal(stack[1], planes[("PolyT", 1)][5:15, 5:25])


def test_rows_and_overview(tmp_path: Path) -> None:
    planes = _write_planes(tmp_path)
    with MosaicPlaneReader(tmp_path, ["DAPI"], [0]) as reader:
        rows = reader.read_rows("DAPI", 0, 16)
        assert rows.shape == (7, 120)
        assert np.array_equal(rows, planes[("DAPI", 0)][::16].astype(np.float32))
        overview = reader.overview("DAPI", 0, 16)
        assert overview.shape == (7, 7)
        expected = rows[:, :112].reshape(7, 7, 16).mean(axis=2)
        assert np.allclose(overview, expected)


def test_missing_plane_is_refused(tmp_path: Path) -> None:
    _write_planes(tmp_path)
    with pytest.raises(FileNotFoundError, match="mosaic_PolyT_z5"):
        MosaicPlaneReader(tmp_path, ["PolyT"], [0, 5])


def test_non_2d_plane_is_refused(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "mosaic_PolyT_z0.tif", np.ones((2, 8, 8), np.uint16))
    with pytest.raises(ValueError, match="2-D"):
        MosaicPlaneReader(tmp_path, ["PolyT"], [0])


def test_close_drops_the_maps_and_reopens_on_use(tmp_path: Path) -> None:
    _write_planes(tmp_path)
    reader = MosaicPlaneReader(tmp_path, ["PolyT"], [0])
    first = reader.read("PolyT", 0, 0, 10, 0, 10)
    reader.close()
    assert not reader._maps
    assert np.array_equal(reader.read("PolyT", 0, 0, 10, 0, 10), first)
    reader.close()


def test_list_mosaic_planes_groups_by_stain(tmp_path: Path) -> None:
    _write_planes(tmp_path, stains=("PolyT", "Anti-X", "Cellbound1"), planes=(2, 0, 1))
    (tmp_path / "notes.txt").write_text("not a plane")
    (tmp_path / "mosaic_PolyT_z3.tiff").write_bytes(b"")
    found = list_mosaic_planes(tmp_path)
    assert found == {
        "PolyT": [0, 1, 2],
        "Anti-X": [0, 1, 2],
        "Cellbound1": [0, 1, 2],
    }


def test_stain_discovery_agrees_with_the_builder(tmp_path: Path) -> None:
    """The reader and the builder must name the stains of a folder alike."""
    from merxen.io.builders.merscope import _get_channel_names

    _write_planes(
        tmp_path,
        stains=("PolyT", "DAPI", "Anti-X", "Cellbound1", "Cellbound3"),
        planes=(0,),
        shape=(8, 8),
    )
    assert sorted(_get_channel_names(tmp_path)) == sorted(list_mosaic_planes(tmp_path))
