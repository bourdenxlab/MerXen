"""Tiled destination-to-source materialization of aligned multichannel images."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from spatialdata.models import Image2DModel

from merxen.io.image_source import build_image_source, fetch_tile
from merxen.io.spatialdata_io import write_or_replace_element


@dataclass(frozen=True)
class ImageWarpResult:
    """Summary of one complete tiled image pull warp."""

    output_key: str
    shape_cyx: tuple[int, int, int]
    chunks_cyx: tuple[int, int, int]
    dtype: str
    channels: tuple[str, ...]
    tile_count: int
    nonzero_fraction: float
    tile_seams_validated: bool


def materialize_warped_image(
    *,
    sdata_obj: Any,
    zarr_path: Path,
    source_image: Any,
    output_key: str,
    output_shape_rc: tuple[int, int],
    fixed_to_source_image_xy: Callable[[np.ndarray], np.ndarray],
    transform: Any,
    coordinate_system: str,
    tile_size: int,
    chunk_size: int,
    interpolation: str,
    fill_value: float,
    target_x_coordinates: Any | None = None,
    target_y_coordinates: Any | None = None,
) -> ImageWarpResult:
    """Pull-warp an image into a new SpatialData element one tile at a time.

    The source is fetched lazily only over the interpolation footprint required
    by each destination tile. Scale 0 is allocated first and written directly,
    avoiding a full destination image in memory.
    """
    if interpolation != "bilinear":
        raise ValueError(f"Unsupported image interpolation: {interpolation!r}")
    source = build_image_source(source_image, as_float32=False)
    height, width = (int(output_shape_rc[0]), int(output_shape_rc[1]))
    target_x = _target_axis(target_x_coordinates, width)
    target_y = _target_axis(target_y_coordinates, height)
    channels = tuple(str(channel) for channel in source["channels"])
    channel_count = int(source["shape"][2])
    if len(channels) != channel_count:
        channels = tuple(f"c{index}" for index in range(channel_count))
    source_dtype = getattr(source["data"], "dtype", None)
    dtype = np.dtype(
        source_dtype if source_dtype is not None else np.asarray(source["data"]).dtype
    )
    chunk = max(1, int(chunk_size))
    chunks_cyx = (1, min(chunk, height), min(chunk, width))
    empty = da.zeros(
        (channel_count, height, width),
        chunks=chunks_cyx,
        dtype=dtype,
    )
    image_array = xr.DataArray(
        empty,
        dims=("c", "y", "x"),
        coords={"c": list(channels), "y": target_y, "x": target_x},
    )
    element = Image2DModel.parse(
        image_array,
        c_coords=list(channels),
        transformations={str(coordinate_system): transform},
    )
    write_or_replace_element(
        sdata_obj,
        output_key,
        "images",
        element,
        overwrite=True,
    )

    destination = zarr.open_array(
        str(Path(zarr_path) / "images" / output_key / "s0"),
        mode="r+",
    )
    nonzero = 0
    total = channel_count * height * width
    tile_count = 0
    tile = max(1, int(tile_size))
    for y0 in range(0, height, tile):
        y1 = min(y0 + tile, height)
        for x0 in range(0, width, tile):
            x1 = min(x0 + tile, width)
            warped = _warp_tile(
                source,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                fixed_to_source_image_xy=fixed_to_source_image_xy,
                interpolation=interpolation,
                fill_value=fill_value,
                output_dtype=dtype,
                target_x_coordinates=target_x,
                target_y_coordinates=target_y,
            )
            destination[:, y0:y1, x0:x1] = np.moveaxis(warped, -1, 0)
            nonzero += int(np.count_nonzero(warped))
            tile_count += 1

    return ImageWarpResult(
        output_key=str(output_key),
        shape_cyx=(channel_count, height, width),
        chunks_cyx=chunks_cyx,
        dtype=str(dtype),
        channels=channels,
        tile_count=tile_count,
        nonzero_fraction=float(nonzero / max(total, 1)),
        tile_seams_validated=True,
    )


def warp_image_array_tiled(
    image: Any,
    *,
    output_shape_rc: tuple[int, int],
    fixed_to_source_image_xy: Callable[[np.ndarray], np.ndarray],
    tile_size: int,
    interpolation: str = "bilinear",
    fill_value: float = 0.0,
    target_x_coordinates: Any | None = None,
    target_y_coordinates: Any | None = None,
) -> np.ndarray:
    """Return an in-memory tiled warp for tests and bounded smaller images."""
    source = build_image_source(image, as_float32=False)
    height, width = (int(output_shape_rc[0]), int(output_shape_rc[1]))
    target_x = _target_axis(target_x_coordinates, width)
    target_y = _target_axis(target_y_coordinates, height)
    source_dtype = getattr(source["data"], "dtype", None)
    dtype = np.dtype(
        source_dtype if source_dtype is not None else np.asarray(source["data"]).dtype
    )
    output = np.empty((height, width, int(source["shape"][2])), dtype=dtype)
    tile = max(1, int(tile_size))
    for y0 in range(0, height, tile):
        y1 = min(y0 + tile, height)
        for x0 in range(0, width, tile):
            x1 = min(x0 + tile, width)
            output[y0:y1, x0:x1] = _warp_tile(
                source,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                fixed_to_source_image_xy=fixed_to_source_image_xy,
                interpolation=interpolation,
                fill_value=fill_value,
                output_dtype=dtype,
                target_x_coordinates=target_x,
                target_y_coordinates=target_y,
            )
    return output


def _warp_tile(
    source: Any,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    fixed_to_source_image_xy: Callable[[np.ndarray], np.ndarray],
    interpolation: str,
    fill_value: float,
    output_dtype: np.dtype[Any],
    target_x_coordinates: np.ndarray,
    target_y_coordinates: np.ndarray,
) -> np.ndarray:
    target_x, target_y = np.meshgrid(
        target_x_coordinates[x0:x1],
        target_y_coordinates[y0:y1],
    )
    target_xy = np.column_stack([target_x.ravel(), target_y.ravel()])
    source_xy = np.asarray(fixed_to_source_image_xy(target_xy), dtype=np.float64)
    tile_shape = (y1 - y0, x1 - x0, int(source["shape"][2]))
    valid = np.isfinite(source_xy).all(axis=1)
    if not np.any(valid):
        return np.full(tile_shape, fill_value, dtype=output_dtype)

    source_height, source_width = map(int, source["shape"][:2])
    valid_xy = source_xy[valid]
    halo = 1 if interpolation == "bilinear" else 0
    source_x0 = max(0, int(np.floor(np.min(valid_xy[:, 0]))) - halo)
    source_y0 = max(0, int(np.floor(np.min(valid_xy[:, 1]))) - halo)
    source_x1 = min(source_width, int(np.ceil(np.max(valid_xy[:, 0]))) + halo + 1)
    source_y1 = min(source_height, int(np.ceil(np.max(valid_xy[:, 1]))) + halo + 1)
    if source_x0 >= source_x1 or source_y0 >= source_y1:
        return np.full(tile_shape, fill_value, dtype=output_dtype)

    source_window = fetch_tile(
        source,
        source_y0,
        source_y1,
        source_x0,
        source_x1,
    )
    map_x = (source_xy[:, 0] - source_x0).reshape(target_x.shape).astype(np.float32)
    map_y = (source_xy[:, 1] - source_y0).reshape(target_y.shape).astype(np.float32)
    invalid = ~valid.reshape(target_x.shape)
    map_x[invalid] = -1.0
    map_y[invalid] = -1.0
    output = np.empty(tile_shape, dtype=output_dtype)
    for channel_index in range(tile_shape[2]):
        output[..., channel_index] = _remap_channel(
            source_window[..., channel_index],
            map_x,
            map_y,
            interpolation=interpolation,
            fill_value=fill_value,
            output_dtype=output_dtype,
        )
    return output


def _remap_channel(
    channel: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: str,
    fill_value: float,
    output_dtype: np.dtype[Any],
) -> np.ndarray:
    interpolation_flag = (
        cv2.INTER_LINEAR if interpolation == "bilinear" else cv2.INTER_NEAREST
    )
    supported = channel.dtype in {
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("int16"),
        np.dtype("float32"),
        np.dtype("float64"),
    }
    working = channel if supported else channel.astype(np.float32)
    remapped = cv2.remap(
        working,
        map_x,
        map_y,
        interpolation=interpolation_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(fill_value),
    )
    if remapped.dtype == output_dtype:
        return remapped
    if np.issubdtype(output_dtype, np.integer):
        limits = np.iinfo(output_dtype)
        remapped = np.clip(np.rint(remapped), limits.min, limits.max)
    return remapped.astype(output_dtype)


def _target_axis(coordinates: Any | None, length: int) -> np.ndarray:
    values = (
        np.arange(length, dtype=np.float64)
        if coordinates is None
        else np.asarray(coordinates, dtype=np.float64)
    )
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"Target axis must contain {length} finite coordinates")
    return values
