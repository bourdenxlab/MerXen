"""Zeiss CZI scene to pyramidal OME-TIFF.

Reads one scene of a Zeiss CZI mosaic with ``czifile`` (imported lazily, so the
module imports without it) and streams it into a tiled, pyramidal BigTIFF
OME-TIFF through :mod:`merxen.io.ome_tiff_writer`. Level 0 is pixel-identical to
the CZI scene on its bounding-box grid (negative origins are fine, pixels
covered by no sub-block are 0, Z planes are max-projected).

The CZI XML holds user names, paths and instrument serial numbers. It is parsed
here but never written, logged or returned: only the fields of
:class:`CziChannelMeta`, :class:`CziSceneMeta` and :class:`CziFileInfo` leave
this module.

``czifile`` may live in a separate runtime (its ``imagecodecs`` requirement
conflicts with the base environment). :func:`run_read_czi_info` and
:func:`run_convert_czi_scene` call the functions in this module in-process when
``czifile`` is importable and otherwise through ``python -m merxen.io.czi`` in
the interpreter named by ``MERXEN_CZI_PYTHON``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np

from merxen.io.channel_names import ChannelNameCandidates
from merxen.io.ome_tiff_writer import write_pyramidal_ome_tiff

logger = logging.getLogger(__name__)

CZI_PYTHON_ENV = "MERXEN_CZI_PYTHON"
SUPPORTED_PIXEL_TYPES = frozenset({"GRAY8", "GRAY16", "GRAY32FLOAT"})
CZIFILE_VERSION_PIN = "2026.10.14"
IN_PROCESS = "in_process"

_RUNTIME_MESSAGE = (
    "Reading CZI files needs the czifile package, which is not installed in this "
    "Python environment, and MERXEN_CZI_PYTHON is not set. Create the CZI runtime "
    "from envs/environment.czi.yml (or install requirements/requirements.czi.lock "
    "into a Python 3.12 venv) and set MERXEN_CZI_PYTHON to its python executable "
    "in .env (or the environment); Nextflow runs take it from "
    "--manual_registration_czi_python (site config)."
)

_ZEN_PIXEL_TYPES = {
    "GRAY8": "Gray8",
    "GRAY16": "Gray16",
    "GRAY32FLOAT": "Gray32Float",
    "GRAY32": "Gray32",
    "GRAY64": "Gray64",
    "BGR24": "Bgr24",
    "BGR48": "Bgr48",
    "BGR96FLOAT": "Bgr96Float",
    "BGRA32": "Bgra32",
    "GRAY64COMPLEXFLOAT": "Gray64ComplexFloat",
    "BGR192COMPLEXFLOAT": "Bgr192ComplexFloat",
}
_ZEN_COMPRESSION = {
    "UNCOMPRESSED": "Uncompressed",
    "JPEG": "Jpg",
    "LZW": "Lzw",
    "JPEGXR": "JpegXr",
    "ZSTD": "Zstd0",
    "ZSTDHDR": "Zstd1",
}
_RANGE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*")


class CziRuntimeUnavailableError(ImportError):
    """``czifile`` is not importable here and no CZI runtime is configured."""


@dataclass(frozen=True)
class CziChannelMeta:
    """Name-like and spectral metadata of one CZI channel (``None`` when absent)."""

    index: int
    channel_id: str | None
    name: str | None
    fluor: str | None
    dye_name: str | None
    short_name: str | None
    excitation_nm: float | None
    emission_nm: float | None
    pixel_type: str | None
    contrast_method: str | None


@dataclass(frozen=True)
class CziSceneMeta:
    """One scene: its index, optional name and ``(x, y, w, h)`` pixel bounding box."""

    index: int
    name: str | None
    bbox_xywh: tuple[int, int, int, int]


@dataclass(frozen=True)
class CziFileInfo:
    """What a CZI holds, read without touching pixels.

    Attributes:
        path: The CZI file.
        scenes: Every scene with its bounding box.
        channels: Channel metadata sliced to the image's channel range, indexed
            ``0..C-1``.
        sizes: Every non-spatial dimension including size-1 ones.
        dtype: numpy dtype name, e.g. ``"uint16"``.
        pixel_size_um: ``(x, y)`` pixel size in micrometres, or ``None``.
        pixel_type: ZEN spelling, e.g. ``"Gray16"``.
        compression: ZEN spelling, e.g. ``"Uncompressed"`` or ``"JpegXr"``.
        czifile_version: Version of the ``czifile`` that read the file.
    """

    path: Path
    scenes: tuple[CziSceneMeta, ...]
    channels: tuple[CziChannelMeta, ...]
    sizes: dict[str, int]
    dtype: str
    pixel_size_um: tuple[float, float] | None
    pixel_type: str | None
    compression: str | None
    czifile_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation (see :meth:`from_dict`)."""
        return {
            "path": str(self.path),
            "scenes": [
                {
                    "index": s.index,
                    "name": s.name,
                    "bbox_xywh": list(s.bbox_xywh),
                }
                for s in self.scenes
            ],
            "channels": [dataclasses.asdict(c) for c in self.channels],
            "sizes": dict(self.sizes),
            "dtype": self.dtype,
            "pixel_size_um": (
                None if self.pixel_size_um is None else list(self.pixel_size_um)
            ),
            "pixel_type": self.pixel_type,
            "compression": self.compression,
            "czifile_version": self.czifile_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CziFileInfo:
        """Rebuild an instance from :meth:`to_dict` output."""
        pixel_size = data["pixel_size_um"]
        return cls(
            path=Path(data["path"]),
            scenes=tuple(
                CziSceneMeta(
                    int(s["index"]),
                    s["name"],
                    _int4(s["bbox_xywh"]),
                )
                for s in data["scenes"]
            ),
            channels=tuple(CziChannelMeta(**c) for c in data["channels"]),
            sizes={str(k): int(v) for k, v in data["sizes"].items()},
            dtype=str(data["dtype"]),
            pixel_size_um=(
                None
                if pixel_size is None
                else (float(pixel_size[0]), float(pixel_size[1]))
            ),
            pixel_type=data["pixel_type"],
            compression=data["compression"],
            czifile_version=str(data["czifile_version"]),
        )


@dataclass(frozen=True)
class CziConversionResult:
    """What :func:`convert_czi_scene` wrote."""

    output_path: Path
    shape_cyx: tuple[int, int, int]
    level_shapes_yx: tuple[tuple[int, int], ...]
    scene_index: int
    scene_name: str | None
    bbox_xywh: tuple[int, int, int, int]
    z_planes_projected: int
    pixel_type: str | None
    compression: str | None
    czifile_version: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation (see :meth:`from_dict`)."""
        return {
            "output_path": str(self.output_path),
            "shape_cyx": list(self.shape_cyx),
            "level_shapes_yx": [list(s) for s in self.level_shapes_yx],
            "scene_index": self.scene_index,
            "scene_name": self.scene_name,
            "bbox_xywh": list(self.bbox_xywh),
            "z_planes_projected": self.z_planes_projected,
            "pixel_type": self.pixel_type,
            "compression": self.compression,
            "czifile_version": self.czifile_version,
            "seconds": self.seconds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CziConversionResult:
        """Rebuild an instance from :meth:`to_dict` output."""
        c, h, w = (int(n) for n in data["shape_cyx"])
        return cls(
            output_path=Path(data["output_path"]),
            shape_cyx=(c, h, w),
            level_shapes_yx=tuple(
                (int(s[0]), int(s[1])) for s in data["level_shapes_yx"]
            ),
            scene_index=int(data["scene_index"]),
            scene_name=data["scene_name"],
            bbox_xywh=_int4(data["bbox_xywh"]),
            z_planes_projected=int(data["z_planes_projected"]),
            pixel_type=data["pixel_type"],
            compression=data["compression"],
            czifile_version=str(data["czifile_version"]),
            seconds=float(data["seconds"]),
        )


def _int4(values: Sequence[Any]) -> tuple[int, int, int, int]:
    a, b, c, d = (int(v) for v in values)
    return (a, b, c, d)


# ---------------------------------------------------------------------------
# Metadata parsing (pure)
# ---------------------------------------------------------------------------


def _text(element: ElementTree.Element | None, tag: str) -> str | None:
    if element is None:
        return None
    text = element.findtext(tag)
    if text is None:
        return None
    text = text.strip()
    return text or None


def _float(element: ElementTree.Element | None, tag: str) -> float | None:
    text = _text(element, tag)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _range_midpoint(text: str | None) -> float | None:
    if text is None:
        return None
    match = _RANGE.fullmatch(text)
    if match is None:
        return None
    return (float(match.group(1)) + float(match.group(2))) / 2.0


def parse_channel_metadata(xml_text: str) -> list[CziChannelMeta]:
    """Parse the channel metadata of a CZI XML document.

    Channels come from ``Information/Image/Dimensions/Channels``; the display
    settings (``ShortName``, ``DyeName``, ...) are matched by channel ``Id``.

    Args:
        xml_text: The CZI XML metadata (never logged).

    Returns:
        One entry per channel in document order, or ``[]`` without channels.
    """
    root = ElementTree.fromstring(xml_text)
    displays = {
        el.get("Id"): el for el in root.findall(".//DisplaySetting/Channels/Channel")
    }
    channels = []
    for index, el in enumerate(
        root.findall(".//Information/Image/Dimensions/Channels/Channel")
    ):
        channel_id = el.get("Id")
        disp = displays.get(channel_id) if channel_id is not None else None
        name = (el.get("Name") or "").strip() or _text(el, "Name")
        excitation = (
            _float(el, "ExcitationWavelength")
            or _float(el, "LightSourcesSettings/LightSourceSettings/Wavelength")
            or _float(disp, "DyeMaxExcitation")
        )
        emission = (
            _float(el, "EmissionWavelength")
            or _range_midpoint(_text(el, "DetectionWavelength/Ranges"))
            or _float(disp, "DyeMaxEmission")
        )
        channels.append(
            CziChannelMeta(
                index=index,
                channel_id=channel_id,
                name=name,
                fluor=_text(el, "Fluor"),
                dye_name=_text(el, "DyeName") or _text(disp, "DyeName"),
                short_name=_text(disp, "ShortName"),
                excitation_nm=excitation,
                emission_nm=emission,
                pixel_type=_text(el, "PixelType"),
                contrast_method=_text(el, "ContrastMethod"),
            )
        )
    return channels


def parse_scene_names(xml_text: str) -> dict[int, str]:
    """Return ``{scene index: name}`` for the named scenes of a CZI XML document."""
    root = ElementTree.fromstring(xml_text)
    names: dict[int, str] = {}
    for position, el in enumerate(
        root.findall(".//Information/Image/Dimensions/S/Scenes/Scene")
    ):
        index = int(el.get("Index", position))
        name = (el.get("Name") or el.findtext("Name") or "").strip()
        if name:
            names[index] = name
    return names


def describe_scenes(scenes: Sequence[CziSceneMeta]) -> str:
    """Describe scenes for messages: ``0 'Name' (76544×55012 px at x=-1, y=2); ...``."""
    parts = []
    for scene in scenes:
        x, y, w, h = scene.bbox_xywh
        geometry = f"({w}×{h} px at x={x}, y={y})"
        if scene.name:
            parts.append(f"{scene.index} {scene.name!r} {geometry}")
        else:
            parts.append(f"{scene.index} {geometry}")
    return "; ".join(parts)


def resolve_scene(scenes: Sequence[CziSceneMeta], requested: str | None) -> int:
    """Choose a scene index from an index or a name.

    Args:
        scenes: The scenes of the CZI.
        requested: A scene index, a scene name (case-insensitive) or ``None``.

    Returns:
        The scene index.

    Raises:
        ValueError: If there are no scenes, none was chosen among several, or the
            request matches none or several scenes.
    """
    if not scenes:
        raise ValueError("the CZI has no image scenes")
    if requested is None or not requested.strip():
        if len(scenes) == 1:
            return scenes[0].index
        raise ValueError(
            f"the CZI has {len(scenes)} scenes and none was chosen; set the scene "
            f"(index or name) to one of: {describe_scenes(scenes)}"
        )
    text = requested.strip()
    not_found = ValueError(
        f"scene {text!r} not found; available: {describe_scenes(scenes)}"
    )
    if re.fullmatch(r"-?\d+", text):
        for scene in scenes:
            if scene.index == int(text):
                return scene.index
        raise not_found
    matches = [s for s in scenes if s.name and s.name.casefold() == text.casefold()]
    if len(matches) == 1:
        return matches[0].index
    if len(matches) > 1:
        raise ValueError(
            f"scene name {text!r} is ambiguous; use the index: "
            f"{describe_scenes(scenes)}"
        )
    raise not_found


def czi_channel_candidates(info: CziFileInfo) -> list[ChannelNameCandidates]:
    """Return the name-like fields of every channel for ``resolve_channel_names``."""
    return [
        ChannelNameCandidates(
            name=ch.name,
            fluor=ch.fluor,
            dye_name=ch.dye_name,
            short_name=ch.short_name,
        )
        for ch in info.channels
    ]


def resolve_czi_pixel_size(
    info: CziFileInfo,
    override: float | None,
    *,
    override_hint: str = "--pixel-size-um",
) -> tuple[float, str]:
    """Choose the pixel size in micrometres.

    Args:
        info: The CZI's metadata.
        override: An explicit pixel size, or ``None`` to use the metadata.
        override_hint: Name of the setting to mention in error messages.

    Returns:
        ``(pixel size, "override" | "metadata")``.

    Raises:
        ValueError: If the override is invalid, the metadata has no pixel size
            or its X and Y sizes differ by more than 0.1 %.
    """
    name = info.path.name
    metadata = info.pixel_size_um
    if override is not None:
        if not (math.isfinite(override) and override > 0):
            raise ValueError(
                f"{name}: pixel size override must be finite and > 0, got {override}"
            )
        if metadata is not None and abs(override / metadata[0] - 1) > 0.01:
            logger.warning(
                "%s: pixel size override %.6g µm differs from the metadata "
                "%.6g µm by more than 1 %%",
                name,
                override,
                metadata[0],
            )
        return float(override), "override"
    if metadata is None:
        raise ValueError(
            f"{name}: no pixel size (Scaling) metadata; set {override_hint}"
        )
    x, y = metadata
    if abs(x - y) > 0.001 * max(x, y):
        raise ValueError(
            f"{name}: X and Y pixel sizes differ ({x:.6g} vs {y:.6g} µm); "
            f"set {override_hint} explicitly"
        )
    return float(x), "metadata"


# ---------------------------------------------------------------------------
# czifile access
# ---------------------------------------------------------------------------


def _import_czifile() -> Any:
    try:
        import czifile
    except ImportError as exc:
        raise CziRuntimeUnavailableError(_RUNTIME_MESSAGE) from exc
    logging.getLogger("czifile").setLevel(logging.ERROR)
    return czifile


def _open_czi(path: Path) -> Any:
    """Open a CZI (module-level so tests can replace it); not thread-safe."""
    return _import_czifile().CziFile(path, memmap=True, squeeze=False)


def read_czi_info(path: Path) -> CziFileInfo:
    """Read scenes, channels, sizes and calibration of a CZI.

    Args:
        path: The CZI file.

    Returns:
        The file's metadata.

    Raises:
        CziRuntimeUnavailableError: If ``czifile`` is not installed.
        ValueError: For multi-file CZIs, RGB or non-grey pixel types,
            sub-sampled/super-resolved acquisitions and any dimension other than
            C and Z larger than 1.
    """
    path = Path(path)
    czifile = _import_czifile()
    with _open_czi(path) as czi:
        header = czi.header
        if header.file_part != 0 or header.primary_file_guid != header.file_guid:
            raise ValueError(
                f"{path.name}: multi-file CZI (file part {header.file_part}) is not "
                "supported; save it as a single file in ZEN first"
            )
        xml_text = czi.metadata()
        names = parse_scene_names(xml_text)
        scenes = tuple(
            CziSceneMeta(
                int(key),
                names.get(int(key)),
                _int4(img.bbox),
            )
            for key, img in czi.scenes.items()
        )
        merged = czi.scenes()
        pixel_type = merged.pixeltype.name
        zen_type = _ZEN_PIXEL_TYPES.get(pixel_type, pixel_type)
        if pixel_type.startswith("BGR"):
            raise ValueError(
                f"{path.name}: pixel type {zen_type} "
                "(RGB) is not supported in v1; only grey-level CZIs (Gray8, Gray16, "
                "Gray32Float) are"
            )
        if pixel_type not in SUPPORTED_PIXEL_TYPES:
            raise ValueError(
                f"{path.name}: pixel type {zen_type} "
                "is not supported; only Gray8, Gray16 and Gray32Float are"
            )
        if merged.is_upsampled or merged.is_downsampled:
            raise ValueError(
                f"{path.name}: sub-sampled (Airyscan) or super-resolved (PALM) "
                "acquisitions are not supported"
            )
        sizes_all = {str(d): int(n) for d, n in merged.sizes.items()}
        extra = {
            d: n
            for d, n in sizes_all.items()
            if d not in {"C", "Z", "Y", "X", "S"} and n > 1
        }
        if extra:
            raise ValueError(
                f"{path.name}: unsupported dimensions {extra}; only C and Z may be "
                "larger than 1 (split time series, phases, rotations, views or "
                "blocks in ZEN first)"
            )
        start = dict(zip(merged.dims, merged.start, strict=True))
        dtype = np.dtype(merged.dtype).name
        mpp = merged.mpp
        compression_name = merged.compression.name
    channels = parse_channel_metadata(xml_text)
    n_c = sizes_all.get("C", 1)
    c0 = int(start.get("C", 0))
    channels = channels[c0 : c0 + n_c] if len(channels) >= c0 + n_c else channels[:n_c]
    channels = [dataclasses.replace(ch, index=i) for i, ch in enumerate(channels)]
    for i in range(len(channels), n_c):
        channels.append(
            CziChannelMeta(i, None, None, None, None, None, None, None, None, None)
        )
    sizes = {d: n for d, n in sizes_all.items() if d not in ("X", "Y")}
    return CziFileInfo(
        path=path,
        scenes=scenes,
        channels=tuple(channels),
        sizes=sizes,
        dtype=dtype,
        pixel_size_um=(float(mpp[0]), float(mpp[1])) if mpp else None,
        pixel_type=_ZEN_PIXEL_TYPES.get(pixel_type, pixel_type),
        compression=_ZEN_COMPRESSION.get(compression_name, compression_name),
        czifile_version=str(czifile.__version__),
    )


def convert_czi_scene(
    path: Path,
    output_path: Path,
    *,
    scene: int,
    channel_names: Sequence[str],
    pixel_size_um: float,
    image_name: str,
    tile_size: int = 1024,
    band_height: int = 2048,
    min_level_size: int = 1024,
    compression: str = "zlib",
    compression_level: int = 6,
    workers: int = 4,
    tmp_dir: Path | None = None,
) -> CziConversionResult:
    """Convert one CZI scene to a tiled, pyramidal OME-TIFF.

    OME-TIFF pixel ``(row, col)`` is CZI global pixel ``(y0 + row, x0 + col)`` of
    the scene's bounding box; pixels with no sub-block are 0 and Z planes are
    max-projected. Output channel ``i`` is the ``i``-th CZI channel.

    Args:
        path: The CZI file.
        output_path: Destination ``.ome.tif`` / ``.ome.tiff``.
        scene: Scene index (as in :class:`CziSceneMeta`).
        channel_names: One name per channel.
        pixel_size_um: Pixel size written to the OME metadata.
        image_name: OME image name.
        tile_size: Tile edge.
        band_height: Rows per band (a multiple of ``tile_size``).
        min_level_size: Smallest pyramid level's longer side.
        compression: ``"zlib"``, ``"zstd"`` or ``"none"``.
        compression_level: Compression level.
        workers: Reader and compression threads.
        tmp_dir: Directory for temporary pyramid levels.

    Returns:
        What was written and how long it took.

    Raises:
        ValueError: For a missing scene, bad channel names or pixel size, and
            every unsupported CZI feature (see :func:`read_czi_info`).
    """
    started = time.perf_counter()
    path = Path(path)
    czifile = _import_czifile()
    info = read_czi_info(path)
    meta = next((s for s in info.scenes if s.index == scene), None)
    if meta is None:
        raise ValueError(
            f"{path.name}: scene {scene} does not exist; available: "
            f"{describe_scenes(info.scenes)}"
        )
    n_c = info.sizes.get("C", 1)
    names = [str(n).strip() for n in channel_names]
    if (
        len(names) != n_c
        or any(not n for n in names)
        or len({n.casefold() for n in names}) != n_c
    ):
        raise ValueError(
            f"{path.name}: {len(channel_names)} channel name(s) {list(channel_names)} "
            f"given for {n_c} channel(s) (names must be non-empty and unique)"
        )
    if not (math.isfinite(pixel_size_um) and pixel_size_um > 0):
        raise ValueError(f"pixel_size_um must be finite and > 0, got {pixel_size_um}")

    with _open_czi(path) as czi:
        img = czi.scenes[scene]
        dims = tuple(img.dims)
        start = dict(zip(dims, img.start, strict=True))
        sizes = {str(d): int(n) for d, n in img.sizes.items()}
        x0, y0, width, height = (int(v) for v in img.bbox)
        dtype = np.dtype(img.dtype)
    c_start = int(start["C"]) if "C" in dims else None
    z_start = int(start["Z"]) if "Z" in dims else None
    n_z = sizes.get("Z", 1)
    n_strips = max(1, min(workers, width))
    edges = [width * i // n_strips for i in range(n_strips + 1)]
    strips = [
        (edges[i], edges[i + 1] - edges[i])
        for i in range(n_strips)
        if edges[i + 1] > edges[i]
    ]

    local = threading.local()
    handles: list[Any] = []
    handles_lock = threading.Lock()
    warned: set[tuple[int, int]] = set()

    def thread_image() -> Any:
        if not hasattr(local, "img"):
            handle = _open_czi(path)
            with handles_lock:
                handles.append(handle)
            local.img = handle.scenes[scene]
        return local.img

    def read_strip(c: int, z: int, sx: int, sw: int, by: int, bh: int) -> np.ndarray:
        selection: dict[str, int] = {}
        if c_start is not None:
            selection["C"] = c_start + c
        if z_start is not None:
            selection["Z"] = z_start + z
        try:
            arr = thread_image()(roi=(x0 + sx, y0 + by, sw, bh), **selection).asarray(
                fillvalue=0, maxworkers=1
            )
        except ValueError as exc:
            if "roi matches no subblocks" in str(exc):
                return np.zeros((bh, sw), dtype)
            if "selection matches no subblocks" in str(exc):
                with handles_lock:
                    first = (c, z) not in warned
                    warned.add((c, z))
                if first:
                    logger.warning(
                        "%s scene %d has no data for channel %d, plane %d; "
                        "written as 0",
                        path.name,
                        scene,
                        c,
                        z,
                    )
                return np.zeros((bh, sw), dtype)
            raise
        if arr.size != bh * sw:
            raise RuntimeError(f"czifile returned {arr.shape} for a {bh}x{sw} read")
        return np.asarray(arr.reshape(bh, sw).astype(dtype, copy=False))

    with ThreadPoolExecutor(max_workers=len(strips)) as pool:

        def read_band(c: int, by: int, bh: int) -> np.ndarray:
            band: np.ndarray | None = None
            for z in range(n_z):
                parts = list(
                    pool.map(
                        lambda s, c=c, z=z: read_strip(c, z, s[0], s[1], by, bh),
                        strips,
                    )
                )
                plane = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
                band = plane if band is None else np.maximum(band, plane, out=band)
            assert band is not None
            return band

        try:
            result = write_pyramidal_ome_tiff(
                read_band,
                shape_cyx=(n_c, height, width),
                dtype=dtype,
                channel_names=names,
                pixel_size_um=pixel_size_um,
                image_name=image_name,
                output_path=output_path,
                tile_size=tile_size,
                band_height=band_height,
                min_level_size=min_level_size,
                compression=compression,
                compression_level=compression_level,
                workers=workers,
                tmp_dir=tmp_dir,
                channel_metadata=[
                    {
                        "Fluor": ch.fluor,
                        "ExcitationWavelength": ch.excitation_nm,
                        "EmissionWavelength": ch.emission_nm,
                    }
                    for ch in info.channels
                ],
            )
        finally:
            for handle in handles:
                handle.close()
    return CziConversionResult(
        output_path=result.output_path,
        shape_cyx=result.shape_cyx,
        level_shapes_yx=result.level_shapes_yx,
        scene_index=scene,
        scene_name=meta.name,
        bbox_xywh=meta.bbox_xywh,
        z_planes_projected=n_z,
        pixel_type=info.pixel_type,
        compression=info.compression,
        czifile_version=str(czifile.__version__),
        seconds=round(time.perf_counter() - started, 3),
    )


# ---------------------------------------------------------------------------
# Runtime bridge
# ---------------------------------------------------------------------------


def czifile_available() -> bool:
    """Return whether ``czifile`` can be imported in this interpreter."""
    return importlib.util.find_spec("czifile") is not None


def czi_runtime_python(python: str | os.PathLike[str] | None = None) -> Path | None:
    """Return the interpreter that reads CZIs, or ``None`` for in-process.

    Args:
        python: An explicit interpreter (always used through a subprocess).

    Returns:
        ``None`` when ``czifile`` is importable here, else the interpreter named
        by ``MERXEN_CZI_PYTHON`` (environment first, then ``.env``).

    Raises:
        CziRuntimeUnavailableError: If neither is available.
    """
    if python is not None:
        return Path(python)
    if czifile_available():
        return None
    from merxen import local_config

    value = local_config.path(CZI_PYTHON_ENV)
    if value is not None and str(value) != IN_PROCESS:
        return value
    raise CziRuntimeUnavailableError(_RUNTIME_MESSAGE)


def run_read_czi_info(
    path: Path, *, python: str | os.PathLike[str] | None = None
) -> CziFileInfo:
    """Run :func:`read_czi_info` in-process or in the CZI runtime."""
    interpreter = czi_runtime_python(python)
    if interpreter is None:
        return read_czi_info(path)
    payload = _run_runtime(interpreter, "info", {"path": os.path.abspath(path)})
    return CziFileInfo.from_dict(payload["info"])


def run_convert_czi_scene(
    path: Path,
    output_path: Path,
    *,
    python: str | os.PathLike[str] | None = None,
    **kwargs: Any,
) -> CziConversionResult:
    """Run :func:`convert_czi_scene` in-process or in the CZI runtime.

    Args:
        path: The CZI file.
        output_path: Destination ``.ome.tif``.
        python: An explicit runtime interpreter.
        **kwargs: Keyword arguments of :func:`convert_czi_scene`.
    """
    interpreter = czi_runtime_python(python)
    if interpreter is None:
        return convert_czi_scene(path, output_path, **kwargs)
    # The runtime starts in a temporary directory: every path must be absolute
    # in the caller's working directory, or a relative output is written there
    # and deleted with it.
    request = {
        "path": os.path.abspath(path),
        "output_path": os.path.abspath(output_path),
        **{
            k: (
                os.path.abspath(v)
                if isinstance(v, os.PathLike) or k == "tmp_dir"
                else v
            )
            for k, v in kwargs.items()
        },
    }
    payload = _run_runtime(interpreter, "convert", request)
    return CziConversionResult.from_dict(payload["result"])


def _merxen_package_dir() -> Path:
    import merxen

    return Path(merxen.__file__).resolve().parent


_ERROR_TYPES: dict[str, type[Exception]] = {
    "ValueError": ValueError,
    "OSError": OSError,
    "FileNotFoundError": FileNotFoundError,
    "CziRuntimeUnavailableError": CziRuntimeUnavailableError,
}


def _run_runtime(
    interpreter: Path, command: str, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Run ``python -m merxen.io.czi <command>`` in ``interpreter``."""
    with tempfile.TemporaryDirectory(prefix="merxen-czi-") as tmp_name:
        tmp = Path(tmp_name)
        request_file = tmp / "request.json"
        result_file = tmp / "result.json"
        request_file.write_text(json.dumps(dict(request)), encoding="utf-8")
        # Expose only the merxen package to the runtime. The package's parent
        # directory may be the base environment's site-packages, whose numpy,
        # tifffile and imagecodecs would shadow the runtime's own.
        shim = tmp / "pypath"
        shim.mkdir()
        (shim / "merxen").symlink_to(_merxen_package_dir(), target_is_directory=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            v for v in (str(shim), env.get("PYTHONPATH")) if v
        )
        proc = subprocess.run(
            [
                str(interpreter),
                "-m",
                "merxen.io.czi",
                command,
                "--request",
                str(request_file),
                "--result",
                str(result_file),
            ],
            env=env,
            cwd=tmp,
            check=False,
        )
        if not result_file.exists():
            raise RuntimeError(
                f"the CZI runtime ({interpreter}) exited with code {proc.returncode} "
                "without a result; see its log above"
            )
        payload: dict[str, Any] = json.loads(result_file.read_text(encoding="utf-8"))
    if not payload.get("ok"):
        exc_type = _ERROR_TYPES.get(payload.get("error_type", ""))
        message = payload.get("message", "")
        if exc_type is None:
            raise RuntimeError(f"{payload.get('error_type')}: {message}")
        raise exc_type(message)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m merxen.io.czi`` (runs inside the CZI runtime)."""
    parser = argparse.ArgumentParser(
        prog="python -m merxen.io.czi", description="CZI reader runtime entry point."
    )
    parser.add_argument("command", choices=["info", "convert", "probe"])
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        payload: dict[str, Any]
        if args.command == "info":
            payload = {
                "ok": True,
                "info": read_czi_info(Path(request["path"])).to_dict(),
            }
        elif args.command == "convert":
            kwargs = {
                k: v for k, v in request.items() if k not in {"path", "output_path"}
            }
            if kwargs.get("tmp_dir") is not None:
                kwargs["tmp_dir"] = Path(kwargs["tmp_dir"])
            result = convert_czi_scene(
                Path(request["path"]), Path(request["output_path"]), **kwargs
            )
            payload = {"ok": True, "result": result.to_dict()}
        else:
            modules: dict[str, str | None] = {}
            for name in ("merxen", "numpy", "tifffile", "imagecodecs"):
                try:
                    module = importlib.import_module(name)
                except ImportError:
                    modules[name] = None
                else:
                    file = getattr(module, "__file__", None)
                    modules[name] = os.path.realpath(file) if file else None
            payload = {"ok": True, "modules": modules}
        code = 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        code = 2
    result_path = Path(args.result)
    partial = result_path.with_name(result_path.name + ".tmp")
    partial.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(partial, result_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
