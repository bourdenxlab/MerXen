"""Streaming writer for tiled, pyramidal BigTIFF OME-TIFF images.

The image is supplied band by band (``read_band``), so a mosaic of tens of
gigabytes never has to fit in memory. Level 0 is written tile by tile while
every reduced level is accumulated in a temporary memory map; the levels are
exact recursive 2x2 integer means (:func:`downsample_2x2`), stored as SubIFDs
of the level-0 pages, which is the layout :mod:`merxen.io.ome_tiff` reads.

The module imports only the standard library, numpy and tifffile (it also runs
in the slim CZI reader runtime) and writes byte-identical files for identical
inputs: the OME UUID is derived from the settings, not from the host.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

import merxen

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES: frozenset[str] = frozenset(
    {
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "float32",
        "float64",
    }
)
CHANNEL_METADATA_KEYS: frozenset[str] = frozenset(
    {"Fluor", "ExcitationWavelength", "EmissionWavelength"}
)

_COMPRESSIONS = ("zlib", "zstd", "none")
_SLACK_BYTES = 64 * 1024 * 1024
_COMPRESSED_FRACTION = 0.8


@dataclass(frozen=True)
class OmeTiffWriteResult:
    """What :func:`write_pyramidal_ome_tiff` wrote.

    Attributes:
        output_path: The final file.
        shape_cyx: Level-0 ``(c, y, x)`` shape.
        level_shapes_yx: ``(y, x)`` of every level, level 0 first.
        bytes_written: Size of the final file in bytes.
    """

    output_path: Path
    shape_cyx: tuple[int, int, int]
    level_shapes_yx: tuple[tuple[int, int], ...]
    bytes_written: int


def pyramid_level_shapes(
    shape_yx: tuple[int, int], *, min_level_size: int
) -> tuple[tuple[int, int], ...]:
    """Return the ``(y, x)`` shape of every pyramid level, level 0 first.

    Each level halves the previous one (rounding up) until its longer side is at
    most ``min_level_size``.

    Args:
        shape_yx: Level-0 ``(y, x)`` shape.
        min_level_size: Stop once the longer side is at most this many pixels.

    Returns:
        The level shapes.
    """
    shapes = [(int(shape_yx[0]), int(shape_yx[1]))]
    while max(shapes[-1]) > min_level_size:
        height, width = shapes[-1]
        shapes.append((math.ceil(height / 2), math.ceil(width / 2)))
    return tuple(shapes)


def downsample_2x2(array: np.ndarray) -> np.ndarray:
    """Reduce a 2-D array by 2 in both axes with an exact block mean.

    An odd last row or column is replicated first. Integers are rounded half
    up (``(sum + 2) // 4``) in 64-bit arithmetic, floats averaged in float64.

    Args:
        array: A 2-D array of a supported dtype.

    Returns:
        The reduced array, same dtype, shape ``(ceil(h / 2), ceil(w / 2))``.
    """
    height, width = array.shape
    if height % 2 or width % 2:
        array = np.pad(array, ((0, height % 2), (0, width % 2)), mode="edge")
    blocks = array.reshape(array.shape[0] // 2, 2, array.shape[1] // 2, 2)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        total = blocks.sum(axis=(1, 3), dtype=np.uint64)
        return np.asarray(((total + 2) // 4).astype(array.dtype))
    if np.issubdtype(array.dtype, np.signedinteger):
        signed = blocks.sum(axis=(1, 3), dtype=np.int64)
        return np.asarray(((signed + 2) // 4).astype(array.dtype))
    mean = blocks.mean(axis=(1, 3), dtype=np.float64)
    return np.asarray(mean.astype(array.dtype))


def _free_bytes(path: Path) -> int:
    """Return the free bytes of the file system holding ``path``."""
    return shutil.disk_usage(path).free


def _validate(
    *,
    shape_cyx: tuple[int, int, int],
    dtype: np.dtype,
    channel_names: Sequence[str],
    pixel_size_um: float,
    image_name: str,
    output_path: Path,
    tile_size: int,
    band_height: int,
    min_level_size: int,
    compression: str,
    compression_level: int,
    workers: int,
    channel_metadata: Sequence[Mapping[str, Any]] | None,
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Check every argument; return the level shapes and the sub-level count."""
    if len(shape_cyx) != 3 or any(int(n) <= 0 for n in shape_cyx):
        raise ValueError(f"shape_cyx must be three positive sizes, got {shape_cyx}")
    n_c, height, width = (int(n) for n in shape_cyx)
    if dtype.name not in SUPPORTED_DTYPES:
        raise ValueError(
            f"dtype {dtype.name} is not supported; "
            f"use one of {sorted(SUPPORTED_DTYPES)}"
        )
    names = [str(name).strip() for name in channel_names]
    if len(names) != n_c:
        raise ValueError(
            f"{len(names)} channel name(s) {list(channel_names)} given for "
            f"{n_c} channel(s)"
        )
    if any(not name for name in names):
        raise ValueError(f"channel names must be non-empty, got {list(channel_names)}")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(
            f"channel names must be unique ignoring case, got {list(channel_names)}"
        )
    if not (math.isfinite(pixel_size_um) and pixel_size_um > 0):
        raise ValueError(f"pixel_size_um must be finite and > 0, got {pixel_size_um}")
    if not str(image_name).strip():
        raise ValueError("image_name must not be empty")
    if tile_size < 16 or tile_size % 16:
        raise ValueError(
            f"tile_size must be a multiple of 16 and >= 16, got {tile_size}"
        )
    if band_height <= 0 or band_height % tile_size:
        raise ValueError(
            f"band_height {band_height} must be a positive multiple of "
            f"tile_size {tile_size}"
        )
    if min_level_size < 1:
        raise ValueError(f"min_level_size must be >= 1, got {min_level_size}")
    if compression not in _COMPRESSIONS:
        raise ValueError(
            f"compression must be one of {list(_COMPRESSIONS)}, got {compression!r}"
        )
    if compression == "zlib" and not 0 <= compression_level <= 9:
        raise ValueError(f"zlib compression_level must be 0-9, got {compression_level}")
    if compression == "zstd" and not 1 <= compression_level <= 22:
        raise ValueError(
            f"zstd compression_level must be 1-22, got {compression_level}"
        )
    if not output_path.name.lower().endswith((".ome.tif", ".ome.tiff")):
        raise ValueError(
            f"output file name {output_path.name!r} must end with .ome.tif or .ome.tiff"
        )
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if channel_metadata is not None:
        if len(channel_metadata) != n_c:
            raise ValueError(
                f"channel_metadata has {len(channel_metadata)} entries for "
                f"{n_c} channel(s)"
            )
        for entry in channel_metadata:
            unknown = set(entry) - CHANNEL_METADATA_KEYS
            if unknown:
                raise ValueError(
                    f"unsupported channel_metadata keys {sorted(unknown)}; "
                    f"allowed: {sorted(CHANNEL_METADATA_KEYS)}"
                )
    shapes = pyramid_level_shapes((height, width), min_level_size=min_level_size)
    n_sub = len(shapes) - 1
    if height > band_height and band_height % 2**n_sub:
        raise ValueError(
            f"band_height {band_height} must be a multiple of 2**{n_sub} = {2**n_sub} "
            f"for a {height}x{width} image; increase it"
        )
    return shapes, n_sub


def _check_disk_space(
    *,
    output_path: Path,
    tmp_parent: Path,
    shape_cyx: tuple[int, int, int],
    shapes: tuple[tuple[int, int], ...],
    dtype: np.dtype,
    compression: str,
) -> None:
    """Raise ``OSError(ENOSPC)`` when the output or temporary levels will not fit."""
    n_c, height, width = shape_cyx
    itemsize = dtype.itemsize
    raw = n_c * height * width * itemsize
    levels = sum(n_c * h * w * itemsize for h, w in shapes[1:])
    fraction = 1.0 if compression == "none" else _COMPRESSED_FRACTION
    out_need = (raw + levels) * fraction + _SLACK_BYTES
    tmp_need = levels + _SLACK_BYTES
    out_dir = output_path.parent
    probe_out = _existing_ancestor(out_dir)
    probe_tmp = _existing_ancestor(tmp_parent)
    if os.stat(probe_out).st_dev == os.stat(probe_tmp).st_dev:
        needs = {probe_out: out_need + tmp_need}
    else:
        needs = {probe_out: out_need, probe_tmp: tmp_need}
    for directory, need in needs.items():
        free = _free_bytes(directory)
        if free < need:
            raise OSError(
                errno.ENOSPC,
                f"not enough free space in {directory} to write {output_path.name}: "
                f"need {need / 1e9:.1f} GB, {free / 1e9:.1f} GB free",
            )


def _existing_ancestor(path: Path) -> Path:
    """Return ``path`` or its closest existing parent (the check runs before mkdir)."""
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _ome_metadata(
    *,
    shape_cyx: tuple[int, int, int],
    dtype: np.dtype,
    channel_names: Sequence[str],
    pixel_size_um: float,
    image_name: str,
    tile_size: int,
    compression: str,
    compression_level: int,
    min_level_size: int,
    channel_metadata: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Return the tifffile OME metadata (deterministic: no host-derived values)."""
    key = {
        "image_name": image_name,
        "shape_cyx": list(shape_cyx),
        "dtype": dtype.name,
        "channel_names": list(channel_names),
        "pixel_size_um": pixel_size_um,
        "tile_size": tile_size,
        "compression": compression,
        "compression_level": compression_level,
        "min_level_size": min_level_size,
    }
    digest = uuid.uuid5(
        uuid.NAMESPACE_URL, "merxen-ome-tiff:" + json.dumps(key, sort_keys=True)
    )
    channel: dict[str, Any] = {"Name": list(channel_names)}
    if channel_metadata:
        for field in sorted(CHANNEL_METADATA_KEYS):
            values = [entry.get(field) for entry in channel_metadata]
            if all(value is not None for value in values):
                channel[field] = (
                    [str(v) for v in values]
                    if field == "Fluor"
                    else [float(v) for v in values if v is not None]
                )
    return {
        "axes": "CYX",
        "Name": image_name,
        "UUID": f"urn:uuid:{digest}",
        "Creator": f"merxen {merxen.__version__}",
        "Channel": channel,
        "PhysicalSizeX": pixel_size_um,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_size_um,
        "PhysicalSizeYUnit": "µm",
    }


def _verify_written(
    path: Path,
    shapes: tuple[tuple[int, int], ...],
    n_c: int,
    dtype: np.dtype,
) -> None:
    """Read the file back with tifffile and compare the levels with what was asked."""
    expected = [(n_c, *shape) for shape in shapes]
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = []
        for level in series.levels:
            shape = tuple(int(n) for n in level.shape)
            if n_c == 1 and len(shape) == 2:
                shape = (1, *shape)
            levels.append(shape)
        problems = []
        if levels != expected:
            problems.append(f"levels {levels} instead of {expected}")
        if np.dtype(series.dtype) != dtype:
            problems.append(f"dtype {series.dtype} instead of {dtype}")
        if not tif.is_bigtiff:
            problems.append("not BigTIFF")
        if not tif.is_ome:
            problems.append("no OME metadata")
    if problems:
        raise RuntimeError(f"wrote {path.name} but read back " + "; ".join(problems))


def write_pyramidal_ome_tiff(
    read_band: Callable[[int, int, int], np.ndarray],
    *,
    shape_cyx: tuple[int, int, int],
    dtype: np.dtype | str,
    channel_names: Sequence[str],
    pixel_size_um: float,
    image_name: str,
    output_path: Path,
    tile_size: int = 1024,
    band_height: int = 2048,
    min_level_size: int = 1024,
    compression: str = "zlib",
    compression_level: int = 6,
    workers: int = 4,
    tmp_dir: Path | None = None,
    channel_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> OmeTiffWriteResult:
    """Write a tiled, pyramidal BigTIFF OME-TIFF from row bands.

    Args:
        read_band: ``read_band(channel, y0, height)`` returns the ``(height,
            width)`` rows ``[y0, y0 + height)`` of a channel, in ``dtype``. Bands
            are requested channel by channel, top to bottom.
        shape_cyx: Level-0 ``(c, y, x)`` shape.
        dtype: Pixel dtype (a name in :data:`SUPPORTED_DTYPES`).
        channel_names: One unique non-empty name per channel.
        pixel_size_um: Isotropic pixel size of level 0 in micrometres.
        image_name: OME image name.
        output_path: Destination, ending in ``.ome.tif`` or ``.ome.tiff``. It is
            replaced atomically once the file has been verified.
        tile_size: Tile edge in pixels (a multiple of 16).
        band_height: Rows per band (a multiple of ``tile_size``; for a taller
            image also of ``2**n_sublevels``).
        min_level_size: Reduced levels are added until the longer side is at
            most this size.
        compression: ``"zlib"``, ``"zstd"`` or ``"none"``.
        compression_level: Compression level for zlib (0-9) or zstd (1-22).
        workers: Threads used by tifffile for compression.
        tmp_dir: Directory for the temporary level memory maps (default: next
            to ``output_path``).
        channel_metadata: Optional per-channel ``Fluor`` / ``ExcitationWavelength``
            / ``EmissionWavelength`` written into the OME channels (a key is
            written only when present for every channel).

    Returns:
        The output path, shapes and file size.

    Raises:
        ValueError: On an invalid argument.
        OSError: With ``ENOSPC`` when the disk space check fails.
        RuntimeError: If ``read_band`` returns the wrong shape or dtype, or the
            written file does not read back as asked.
    """
    np_dtype = np.dtype(dtype)
    output_path = Path(output_path)
    shapes, n_sub = _validate(
        shape_cyx=shape_cyx,
        dtype=np_dtype,
        channel_names=channel_names,
        pixel_size_um=pixel_size_um,
        image_name=image_name,
        output_path=output_path,
        tile_size=tile_size,
        band_height=band_height,
        min_level_size=min_level_size,
        compression=compression,
        compression_level=compression_level,
        workers=workers,
        channel_metadata=channel_metadata,
    )
    n_c, height, width = (int(n) for n in shape_cyx)
    names = [str(name).strip() for name in channel_names]
    tmp_parent = Path(tmp_dir) if tmp_dir is not None else output_path.parent
    _check_disk_space(
        output_path=output_path,
        tmp_parent=tmp_parent,
        shape_cyx=(n_c, height, width),
        shapes=shapes,
        dtype=np_dtype,
        compression=compression,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".ome-tiff-levels-", dir=tmp_parent))
    partial = output_path.with_name(f"{output_path.name}.partial-{uuid.uuid4().hex}")
    levels: list[Any] = []
    try:
        for k in range(1, n_sub + 1):
            levels.append(
                np.lib.format.open_memmap(
                    work_dir / f"level{k}.npy",
                    mode="w+",
                    dtype=np_dtype,
                    shape=(n_c, *shapes[k]),
                )
            )

        def level0_tiles() -> Iterator[np.ndarray]:
            for c in range(n_c):
                for by in range(0, height, band_height):
                    bh = min(band_height, height - by)
                    band = np.asarray(read_band(c, by, bh))
                    if band.shape != (bh, width) or band.dtype != np_dtype:
                        raise RuntimeError(
                            f"read_band({c}, {by}, {bh}) returned {band.shape} "
                            f"{band.dtype}; expected ({bh}, {width}) {np_dtype}"
                        )
                    current, row = band, by
                    for k in range(1, n_sub + 1):
                        current, row = downsample_2x2(current), row // 2
                        levels[k - 1][c, row : row + current.shape[0], :] = current
                    for ty in range(0, bh, tile_size):
                        for tx in range(0, width, tile_size):
                            yield band[ty : ty + tile_size, tx : tx + tile_size]

        def stored_tiles(array: np.ndarray) -> Iterator[np.ndarray]:
            _, level_h, level_w = array.shape
            for c in range(n_c):
                for ty in range(0, level_h, tile_size):
                    for tx in range(0, level_w, tile_size):
                        yield np.asarray(
                            array[c, ty : ty + tile_size, tx : tx + tile_size]
                        )

        options: dict[str, Any] = {
            "tile": (tile_size, tile_size),
            "photometric": "minisblack",
            "maxworkers": max(1, workers),
            "compression": None if compression == "none" else compression,
            "compressionargs": (
                None if compression == "none" else {"level": compression_level}
            ),
        }
        metadata = _ome_metadata(
            shape_cyx=(n_c, height, width),
            dtype=np_dtype,
            channel_names=names,
            pixel_size_um=pixel_size_um,
            image_name=image_name,
            tile_size=tile_size,
            compression=compression,
            compression_level=compression_level,
            min_level_size=min_level_size,
            channel_metadata=channel_metadata,
        )
        with tifffile.TiffWriter(partial, bigtiff=True, ome=True) as tif:
            tif.write(
                level0_tiles(),
                shape=(n_c, height, width),
                dtype=np_dtype,
                subifds=n_sub or None,
                metadata=metadata,
                **options,
            )
            for k in range(1, n_sub + 1):
                level = levels[k - 1]
                level.flush()
                tif.write(
                    stored_tiles(level),
                    shape=level.shape,
                    dtype=np_dtype,
                    subfiletype=1,
                    metadata=None,
                    **options,
                )
        _verify_written(partial, shapes, n_c, np_dtype)
        os.replace(partial, output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        levels.clear()
        shutil.rmtree(work_dir, ignore_errors=True)
    return OmeTiffWriteResult(
        output_path=output_path,
        shape_cyx=(n_c, height, width),
        level_shapes_yx=shapes,
        bytes_written=output_path.stat().st_size,
    )
