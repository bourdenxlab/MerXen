"""Cropped pyramid sources for the registration core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from merxen.io.ome_tiff import OmeTiffInfo, read_ome_tiff_info
from merxen.io.ome_tiff_writer import write_pyramidal_ome_tiff
from merxen.manual_registration.schema import CropRect
from merxen.manual_registration.sources import (
    crop_offset_affine,
    cropped_moving_image,
    cropped_reference_grid,
    pyramid_levels,
)

NAMES = ["a", "b"]
PIXEL_SIZE = 0.5


def _mean_2x2(array: np.ndarray) -> np.ndarray:
    """Integer 2x2 mean ``(sum + 2) // 4`` with edge replication at odd ends."""
    c, h, w = array.shape
    padded = np.pad(array, ((0, 0), (0, h % 2), (0, w % 2)), mode="edge").astype(
        np.uint64
    )
    total = (
        padded[:, 0::2, 0::2]
        + padded[:, 1::2, 0::2]
        + padded[:, 0::2, 1::2]
        + padded[:, 1::2, 1::2]
    )
    return ((total + 2) // 4).astype(array.dtype)


@pytest.fixture(scope="module")
def pyramid(tmp_path_factory: pytest.TempPathFactory) -> tuple[OmeTiffInfo, np.ndarray]:
    full = np.random.default_rng(0).integers(0, 65535, (2, 1300, 1500), dtype=np.uint16)
    path = tmp_path_factory.mktemp("sources") / "image.ome.tif"
    write_pyramidal_ome_tiff(
        lambda c, y0, h: full[c, y0 : y0 + h],
        shape_cyx=full.shape,
        dtype=np.uint16,
        channel_names=NAMES,
        pixel_size_um=PIXEL_SIZE,
        image_name="image",
        output_path=path,
        tile_size=256,
        band_height=512,
        min_level_size=256,
        workers=1,
    )
    return read_ome_tiff_info(path), full


def test_crop_is_bit_identical_to_means_of_the_cropped_level_zero(
    pyramid: tuple[OmeTiffInfo, np.ndarray],
) -> None:
    info, full = pyramid
    assert [shape[1:] for shape in info.level_shapes_cyx] == [
        (1300, 1500),
        (650, 750),
        (325, 375),
        (163, 188),
    ]
    crop = CropRect(x0=128, y0=64, x1=1500, y1=1280)
    levels, shapes, factors = pyramid_levels(
        info, channel_names=NAMES, crop=crop, exact_factors=True
    )
    expected = full[:, 64:1280, 128:1500]
    for index, level in enumerate(levels):
        assert np.array_equal(level.values, expected), index
        expected = _mean_2x2(expected)
    assert len(levels) == 4
    assert factors == ((1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0))
    assert shapes == tuple(level.shape[1:] for level in levels)
    assert [str(name) for name in levels[0].coords["c"].values] == NAMES


def test_inexact_factors_are_not_reported(
    pyramid: tuple[OmeTiffInfo, np.ndarray],
) -> None:
    info, _ = pyramid
    crop = CropRect(x0=128, y0=64, x1=1500, y1=1280)
    exact_levels, _, _ = pyramid_levels(
        info, channel_names=NAMES, crop=crop, exact_factors=True
    )
    levels, _, factors = pyramid_levels(
        info, channel_names=NAMES, crop=crop, exact_factors=False
    )
    assert factors is None
    for first, second in zip(levels, exact_levels, strict=True):
        assert np.array_equal(first.values, second.values)


def test_unsnapped_crop_is_rejected(pyramid: tuple[OmeTiffInfo, np.ndarray]) -> None:
    info, _ = pyramid
    with pytest.raises(ValueError, match="not snapped"):
        pyramid_levels(
            info,
            channel_names=NAMES,
            crop=CropRect(x0=100, y0=64, x1=1500, y1=1280),
            exact_factors=True,
        )


def test_levels_whose_factor_does_not_divide_64_are_dropped(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    base = rng.integers(0, 1000, (1, 1200, 1200), dtype=np.uint16)
    path = tmp_path / "odd.ome.tif"
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as writer:
        writer.write(
            base,
            subifds=2,
            tile=(256, 256),
            metadata={"axes": "CYX", "Channel": {"Name": ["a"]}},
        )
        for factor in (3, 4):
            size = 1200 // factor
            writer.write(
                base[:, ::factor, ::factor][:, :size, :size],
                subfiletype=1,
                tile=(256, 256),
            )
    info = read_ome_tiff_info(path)
    assert [shape[1:] for shape in info.level_shapes_cyx] == [
        (1200, 1200),
        (400, 400),
        (300, 300),
    ]
    levels, shapes, factors = pyramid_levels(
        info,
        channel_names=["a"],
        crop=CropRect(x0=0, y0=0, x1=1200, y1=1200),
        exact_factors=True,
    )
    assert factors == ((1.0, 1.0), (4.0, 4.0))
    assert shapes == ((1200, 1200), (300, 300))
    assert len(levels) == 2


def test_cropped_reference_grid(pyramid: tuple[OmeTiffInfo, np.ndarray]) -> None:
    info, _ = pyramid
    crop = CropRect(x0=128, y0=64, x1=1500, y1=1280)
    grid = cropped_reference_grid(
        info,
        channel_names=NAMES,
        crop=crop,
        pixel_size_um=PIXEL_SIZE,
        exact_factors=True,
    )
    assert grid.key == "reference"
    assert grid.shape_yx == (1216, 1372)
    matrix = grid.transformations["global"].to_affine_matrix(("x", "y"), ("x", "y"))
    np.testing.assert_allclose(
        matrix,
        [
            [PIXEL_SIZE, 0, PIXEL_SIZE * 128],
            [0, PIXEL_SIZE, PIXEL_SIZE * 64],
            [0, 0, 1],
        ],
    )
    assert grid.coordinate_system == "global"
    assert grid.level_factors_xy == ((1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0))


def test_crop_offset_affine_matches_the_documented_matrix() -> None:
    affine = crop_offset_affine(CropRect(x0=64, y0=128, x1=2048, y1=2048), 2.0)
    np.testing.assert_allclose(
        affine.to_affine_matrix(("x", "y"), ("x", "y")),
        [[2, 0, 128], [0, 2, 256], [0, 0, 1]],
    )


def test_cropped_moving_image(pyramid: tuple[OmeTiffInfo, np.ndarray]) -> None:
    info, _ = pyramid
    crop = CropRect(x0=128, y0=64, x1=1500, y1=1280)
    moving = cropped_moving_image(
        info,
        channel_names=NAMES,
        crop=crop,
        pixel_size_um=PIXEL_SIZE,
        exact_factors=True,
    )
    assert moving.image_key == "moving"
    assert [str(name) for name in moving.level0.coords["c"].values] == NAMES
    assert moving.open_level(2).shape == (2, 304, 343)
    assert moving.level_shapes_yx[2] == (304, 343)
    assert moving.level_shapes_yx[0] == (1216, 1372)
    assert moving.level_factors_xy == ((1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (8.0, 8.0))
