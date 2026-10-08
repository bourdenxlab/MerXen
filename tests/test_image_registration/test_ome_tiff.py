"""Tests for the lazy OME-TIFF reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from merxen.io.ome_tiff import open_ome_tiff_level, read_ome_tiff_info


def _write_ome(
    path: Path,
    data: np.ndarray,
    *,
    axes: str,
    channel_names: list[str] | None = None,
    pixel_size_um: float | None = 0.5,
    pyramid_levels: int = 0,
    photometric: str = "minisblack",
) -> Path:
    metadata: dict[str, object] = {"axes": axes}
    if channel_names is not None:
        metadata["Channel"] = {"Name": channel_names}
    if pixel_size_um is not None:
        metadata.update(
            {
                "PhysicalSizeX": pixel_size_um,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size_um,
                "PhysicalSizeYUnit": "µm",
            }
        )
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        tif.write(
            data,
            subifds=pyramid_levels,
            metadata=metadata,
            photometric=photometric,
            tile=(16, 16),
        )
        level = data
        for _ in range(pyramid_levels):
            level = level[..., ::2, ::2] if axes.endswith("YX") else level[::2, ::2]
            tif.write(level, subfiletype=1, photometric=photometric, tile=(16, 16))
    return path


def test_flat_multichannel_image_metadata_and_values(tmp_path: Path) -> None:
    """Channel names, pixel size, and pixels come from a flat OME-TIFF."""
    data = np.arange(3 * 40 * 48, dtype=np.uint16).reshape(3, 40, 48)
    path = _write_ome(
        tmp_path / "if.ome.tif",
        data,
        axes="CYX",
        channel_names=["DAPI", "AT8", "p62"],
        pixel_size_um=0.343,
    )

    info = read_ome_tiff_info(path)
    image = open_ome_tiff_level(info)

    assert info.shape_cyx == (3, 40, 48)
    assert not info.is_pyramidal
    assert info.channel_names == ("DAPI", "AT8", "p62")
    assert info.pixel_size_um == pytest.approx((0.343, 0.343))
    assert image.dims == ("c", "y", "x")
    assert list(image.coords["c"].values) == ["DAPI", "AT8", "p62"]
    np.testing.assert_array_equal(image.values, data)


def test_pyramidal_image_lists_levels_and_opens_reduced_level(tmp_path: Path) -> None:
    """Every SubIFD level is reported and readable without the full level."""
    data = np.random.default_rng(0).integers(0, 1000, (2, 64, 64), dtype=np.uint16)
    path = _write_ome(
        tmp_path / "pyr.ome.tif",
        data,
        axes="CYX",
        channel_names=["DAPI", "GFAP"],
        pyramid_levels=2,
    )

    info = read_ome_tiff_info(path)
    level_two = open_ome_tiff_level(info, 2)

    assert info.is_pyramidal
    assert info.level_shapes_cyx == ((2, 64, 64), (2, 32, 32), (2, 16, 16))
    np.testing.assert_array_equal(level_two.values, data[:, ::4, ::4])


def test_channel_names_can_be_overridden(tmp_path: Path) -> None:
    """Fluorophore names in the file can be replaced by marker names."""
    path = _write_ome(
        tmp_path / "if.ome.tif",
        np.zeros((2, 16, 16), dtype=np.uint16),
        axes="CYX",
        channel_names=["AF568", "AF647"],
    )
    info = read_ome_tiff_info(path)

    image = open_ome_tiff_level(info, channel_names=["p62", "AT8"])

    assert list(image.coords["c"].values) == ["p62", "AT8"]
    with pytest.raises(ValueError, match="channel name"):
        open_ome_tiff_level(info, channel_names=["only_one"])


def test_rgb_samples_become_channels(tmp_path: Path) -> None:
    """Interleaved RGB samples are exposed as three channels."""
    data = np.zeros((16, 16, 3), dtype=np.uint8)
    data[..., 1] = 7
    path = _write_ome(
        tmp_path / "he.ome.tif",
        data,
        axes="YXS",
        pixel_size_um=None,
        photometric="rgb",
    )

    info = read_ome_tiff_info(path)
    image = open_ome_tiff_level(info)

    assert info.shape_cyx == (3, 16, 16)
    assert info.pixel_size_um is None
    assert info.channel_names == ("c0", "c1", "c2")
    assert int(image.sel(c="c1").max()) == 7


def test_z_stack_is_rejected(tmp_path: Path) -> None:
    """Images with several Z planes must be projected before registration."""
    path = _write_ome(
        tmp_path / "stack.ome.tif",
        np.zeros((3, 2, 16, 16), dtype=np.uint16),
        axes="ZCYX",
    )

    with pytest.raises(ValueError, match="project Z-stacks"):
        read_ome_tiff_info(path)


def test_plain_multipage_tiff_pages_become_channels(tmp_path: Path) -> None:
    """A non-OME TIFF stores channels as pages; they must not read as Z/T."""
    data = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16)
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, data)

    info = read_ome_tiff_info(path)
    image = open_ome_tiff_level(info)

    assert info.shape_cyx == (2, 16, 16)
    assert info.pixel_size_um is None
    np.testing.assert_array_equal(image.values, data)
