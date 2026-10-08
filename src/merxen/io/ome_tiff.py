"""Lazy reader for flat or pyramidal OME-TIFF images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import ParseError

import dask.array as da
import tifffile
import xarray as xr
import zarr
from ome_types import from_xml

logger = logging.getLogger(__name__)

_SPATIAL_AXES = ("Y", "X")
# tifffile labels interleaved RGB samples "S"; both C and S are channel axes.
_CHANNEL_AXES = ("C", "S")
# Plain multi-page TIFFs only carry a generic page axis, read as channels when
# the file has no C or S axis.
_PAGE_AXES = ("I", "Q")
_MICRONS_PER_UNIT = {
    "pm": 1e-6,
    "nm": 1e-3,
    "µm": 1.0,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
}


@dataclass(frozen=True)
class OmeTiffInfo:
    """Shape, channel and calibration metadata of an OME-TIFF's first series.

    Attributes:
        path: Image file path.
        axes: tifffile axes string of the series, e.g. ``"CYX"`` or ``"YXS"``.
        level_shapes_cyx: ``(c, y, x)`` shape of every pyramid level, level 0 first.
        channel_names: One name per channel, from OME-XML or ``c0, c1, ...``.
        pixel_size_um: Level-0 ``(x, y)`` pixel size in microns, when recorded.
        dtype: Pixel data type.
    """

    path: Path
    axes: str
    level_shapes_cyx: tuple[tuple[int, int, int], ...]
    channel_names: tuple[str, ...]
    pixel_size_um: tuple[float, float] | None
    dtype: str

    @property
    def is_pyramidal(self) -> bool:
        """Whether the file stores reduced-resolution levels."""
        return len(self.level_shapes_cyx) > 1

    @property
    def shape_cyx(self) -> tuple[int, int, int]:
        """Level-0 ``(c, y, x)`` shape."""
        return self.level_shapes_cyx[0]


def read_ome_tiff_info(path: Path | str) -> OmeTiffInfo:
    """Read metadata of the first image series without loading pixels.

    Args:
        path: OME-TIFF file, flat or pyramidal (SubIFD or multi-series levels).

    Returns:
        Shape, channel, and pixel-size metadata.

    Raises:
        ValueError: If the image has a non-singleton Z/T axis or two channel axes.
    """
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes).upper()
        level_shapes = tuple(
            _shape_cyx(axes, tuple(int(size) for size in level.shape), path)
            for level in series.levels
        )
        dtype = str(series.dtype)
        ome_xml = tif.ome_metadata if tif.is_ome else None

    channel_names, pixel_size_um = _ome_channels_and_pixel_size(
        ome_xml,
        n_channels=level_shapes[0][0],
        path=path,
    )
    return OmeTiffInfo(
        path=path,
        axes=axes,
        level_shapes_cyx=level_shapes,
        channel_names=channel_names,
        pixel_size_um=pixel_size_um,
        dtype=dtype,
    )


def open_ome_tiff_level(
    info: OmeTiffInfo,
    level: int = 0,
    *,
    channel_names: list[str] | None = None,
) -> xr.DataArray:
    """Open one pyramid level as a lazy ``(c, y, x)`` DataArray.

    Pixels are read through tifffile's Zarr view, one TIFF tile per chunk, so
    only the regions that are computed are read from disk.

    Args:
        info: Metadata from :func:`read_ome_tiff_info`.
        level: Pyramid level; 0 is full resolution.
        channel_names: Optional names replacing the file's channel names.

    Returns:
        Dask-backed DataArray with dims ``("c", "y", "x")`` and ``c`` coordinates.

    Raises:
        ValueError: If the level is missing or the channel names do not match.
    """
    if not 0 <= int(level) < len(info.level_shapes_cyx):
        raise ValueError(
            f"{info.path} has {len(info.level_shapes_cyx)} level(s); "
            f"level {level} is unavailable"
        )
    store = tifffile.imread(info.path, aszarr=True, level=int(level))
    array = da.from_zarr(zarr.open(store, mode="r"))
    data = xr.DataArray(array, dims=tuple(info.axes))
    for dim in info.axes:
        if dim not in _SPATIAL_AXES and data.sizes[dim] == 1:
            data = data.isel({dim: 0}, drop=True)
    channel_dims = [dim for dim in data.dims if dim not in _SPATIAL_AXES]
    data = (
        data.rename({channel_dims[0]: "c"})
        if channel_dims
        else data.expand_dims("c", axis=0)
    )
    data = data.rename({"Y": "y", "X": "x"}).transpose("c", "y", "x")

    names = list(channel_names) if channel_names is not None else info.channel_names
    if len(names) != int(data.sizes["c"]):
        raise ValueError(
            f"{info.path} has {data.sizes['c']} channel(s) but {len(names)} "
            f"channel name(s) were given: {list(names)}"
        )
    return data.assign_coords(c=[str(name) for name in names])


def _shape_cyx(axes: str, shape: tuple[int, ...], path: Path) -> tuple[int, int, int]:
    if len(axes) != len(shape) or not all(axis in axes for axis in _SPATIAL_AXES):
        raise ValueError(f"{path} has unsupported axes {axes!r} for shape {shape}")
    sizes = dict(zip(axes, shape, strict=True))
    channel_axes = [axis for axis in _CHANNEL_AXES if sizes.get(axis, 1) > 1]
    if not channel_axes:
        channel_axes = [axis for axis in _PAGE_AXES if sizes.get(axis, 1) > 1]
    if len(channel_axes) > 1:
        raise ValueError(f"{path} has more than one channel-like axis ({axes!r})")
    extra = {
        axis: size
        for axis, size in sizes.items()
        if axis not in _SPATIAL_AXES and axis not in channel_axes and size > 1
    }
    if extra:
        raise ValueError(
            f"{path} has non-singleton axes {extra}; only 2D multichannel images "
            "are supported, so project Z-stacks and split time series first"
        )
    n_channels = sizes[channel_axes[0]] if channel_axes else 1
    return (int(n_channels), int(sizes["Y"]), int(sizes["X"]))


def _ome_channels_and_pixel_size(
    ome_xml: str | None,
    *,
    n_channels: int,
    path: Path,
) -> tuple[tuple[str, ...], tuple[float, float] | None]:
    fallback_names = tuple(f"c{index}" for index in range(int(n_channels)))
    if not ome_xml:
        return fallback_names, None
    try:
        pixels = from_xml(ome_xml).images[0].pixels
    except (ValueError, ParseError, IndexError) as exc:
        logger.warning("Could not parse OME-XML of %s: %s", path, exc)
        return fallback_names, None

    names = tuple(str(channel.name or "").strip() for channel in pixels.channels)
    if len(names) != int(n_channels) or not all(names):
        # RGB images record one OME channel holding three samples.
        names = fallback_names

    pixel_size_um = None
    if pixels.physical_size_x and pixels.physical_size_y:
        pixel_size_um = (
            _to_microns(pixels.physical_size_x, pixels.physical_size_x_unit),
            _to_microns(pixels.physical_size_y, pixels.physical_size_y_unit),
        )
    return names, pixel_size_um


def _to_microns(value: float, unit: object) -> float:
    unit_name = str(getattr(unit, "value", unit) or "µm")
    if unit_name not in _MICRONS_PER_UNIT:
        raise ValueError(f"Unsupported OME physical size unit: {unit_name!r}")
    return float(value) * _MICRONS_PER_UNIT[unit_name]
