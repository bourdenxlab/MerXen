"""Prepare step of manual landmark registration: images to a prepared store.

Each input image (a Zeiss CZI scene, or an OME-TIFF) becomes a tiled pyramidal
OME-TIFF (a CZI is converted; an OME-TIFF is validated and, when it is untiled,
has too coarse a pyramid or needs another pixel size, rewritten), and the two
images are wrapped in one image-only SpatialData "prepared store"
``<id>_prepared.zarr`` that the napari viewer opens for landmark labelling. The
viewer writes the labelling into one root attribute of that store, so prepare
must never destroy it.

Idempotence rules (details at ``_plan_store_update``): an unchanged re-run writes
nothing (the store is not even opened for writing); a re-run with the same pixel
grid rebuilds the store in a private sibling directory and carries the labelling
over; a re-run with a changed grid under a labelling is refused (before any
OME-TIFF is overwritten) unless forced, in which case the old store is kept as
``<name>.superseded-<UTC>``. A store is never modified in place: it is built in a
temporary sibling and renamed into place under the store's write lock.

Conversions and normalisations are skipped when the ``<id>_<role>.prepare.json``
provenance file next to the output matches the current inputs and settings. The
raw CZI XML is never written anywhere.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import dask
import numpy as np
import tifffile
import xarray as xr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

import merxen
from merxen.io.channel_names import (
    ChannelNameCandidates,
    ResolvedChannelNames,
    resolve_channel_names,
    resolve_registration_channel,
    shared_channel_subset,
    stain_key,
)
from merxen.io.czi import (
    CziFileInfo,
    czi_channel_candidates,
    resolve_scene,
    run_convert_czi_scene,
    run_read_czi_info,
)
from merxen.io.ome_tiff import (
    OmeTiffInfo,
    open_ome_tiff_level,
    read_ome_tiff_info,
)
from merxen.io.ome_tiff_writer import write_pyramidal_ome_tiff
from merxen.io.spatialdata_io import (
    spatialdata_write_lock,
    spatialdata_write_lock_path,
)
from merxen.io.spatialdata_schema import stamp_merxen_schema
from merxen.manual_registration.config import (
    ManualRegistrationImageInput,
    ManualRegistrationPrepareConfig,
)
from merxen.manual_registration.schema import (
    CONVENTIONS,
    CROP_SNAP_PX,
    LABELLING_ATTR,
    SCHEMA_VERSION,
    STORE_ATTR,
    STORE_SCHEMA,
    StoreAttrs,
    canonical_json,
    canonical_sha256,
    file_fingerprint,
    read_root_attrs,
    utc_now,
    write_root_attr,
)
from merxen.memory import log_status

logger = logging.getLogger(__name__)

Role = Literal["reference", "moving"]
ROLES: tuple[Role, Role] = ("reference", "moving")

PREPARE_ALGORITHM = "manual_registration_prepare_v1"
NORMALISE_CONVERTER = "normalise_ome_tiff_v1"
PROVENANCE_SCHEMA = "merxen.manual_registration.prepare_image"
SUMMARY_SCHEMA = "merxen.manual_registration.prepare_summary"
MAX_COARSEST_LEVEL_SIDE_PX = 2048
PIXEL_SIZE_OVERRIDE_TOLERANCE = 0.01  # relative; > 1 % -> warning / normalisation
PIXEL_SIZE_ANISOTROPY_TOLERANCE = 0.001  # relative X/Y difference needing an override
CZI_SUFFIXES = (".czi",)
TIFF_SUFFIXES = (".ome.tif", ".ome.tiff", ".tif", ".tiff")
STALE_TEMP_AGE_SECONDS = 24 * 3600
_GRID_FIELDS = ("source_fingerprint", "scene_index", "shape_yx", "pixel_size_um")


@dataclass(frozen=True)
class PreparedImage:
    """One prepared input image.

    Attributes:
        role: ``"reference"`` or ``"moving"``.
        ome_tiff_path: The OME-TIFF the store reads (written by prepare, or the
            user's file used in place).
        info: Metadata of ``ome_tiff_path``.
        written_by_prepare: Whether prepare wrote ``ome_tiff_path``.
        shape_yx: Level-0 ``(H, W)``.
        pixel_size_um: Isotropic pixel size, rounded to 10 significant digits.
        pixel_size_source: ``"metadata"`` or ``"override"``.
        dtype: Pixel dtype name.
        canonical_channel_names: Names of every channel, in file order.
        name_sources: Where each canonical name came from.
        source_channel_names: The names as recorded in the source file.
        registration_channel: The resolved registration channel.
        record: ``STORE_ATTR`` ``images.<role>`` minus ``channel_names``.
        warnings: Per-image warnings for the summary.
        converter_settings: Settings of the conversion that wrote the OME-TIFF
            (part of the store content digest), or ``None`` when used in place.
    """

    role: Role
    ome_tiff_path: Path
    info: OmeTiffInfo
    written_by_prepare: bool
    shape_yx: tuple[int, int]
    pixel_size_um: float
    pixel_size_source: Literal["metadata", "override"]
    dtype: str
    canonical_channel_names: tuple[str, ...]
    name_sources: tuple[str, ...]
    source_channel_names: tuple[str, ...]
    registration_channel: str
    record: dict[str, Any]
    warnings: tuple[str, ...] = ()
    converter_settings: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _round_pixel_size(value: float) -> float:
    """Round a pixel size to 10 significant digits (strips float noise)."""
    return float(f"{float(value):.10g}")


def _resolve_workers(config: ManualRegistrationPrepareConfig) -> int:
    """Return ``config.workers``, else ``$DASK_NUM_WORKERS``, else ``min(8, cpus)``."""
    if config.workers is not None:
        return int(config.workers)
    env = os.environ.get("DASK_NUM_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return min(8, os.cpu_count() or 1)


def _image_format(path: Path) -> Literal["czi", "ome_tiff"]:
    """Classify an input file by its extension.

    Raises:
        ValueError: If the extension is not supported.
    """
    name = path.name.lower()
    if name.endswith(CZI_SUFFIXES):
        return "czi"
    if name.endswith(TIFF_SUFFIXES):
        return "ome_tiff"
    raise ValueError(
        f"unsupported image file type {path.name!r}; expected .czi, .ome.tif, "
        ".ome.tiff, .tif or .tiff"
    )


def _output_ome_tiff_path(config: ManualRegistrationPrepareConfig, role: str) -> Path:
    return config.prepared_dir / f"{config.dataset_name}_{role}.ome.tif"


def _provenance_path(config: ManualRegistrationPrepareConfig, role: str) -> Path:
    return config.prepared_dir / f"{config.dataset_name}_{role}.prepare.json"


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically (temp file in the same folder, then replace).

    The write is skipped when the file already holds byte-identical text.
    """
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == text:
                return
        except OSError:
            pass
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _safe_root_attrs(store: Path) -> dict[str, Any]:
    """Root attributes of ``store``; ``{}`` when it is not a readable Zarr group."""
    try:
        return read_root_attrs(store)
    except (FileNotFoundError, ValueError, OSError):
        return {}


@contextlib.contextmanager
def _writer_tmp_dir(config: ManualRegistrationPrepareConfig) -> Iterator[Path]:
    """Yield the writer's scratch directory (removed afterwards unless configured)."""
    if config.tmp_dir is not None:
        config.tmp_dir.mkdir(parents=True, exist_ok=True)
        yield config.tmp_dir
        return
    tmp = config.prepared_dir / f".tmp-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True)
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pixel size and normalisation rules (pure)
# ---------------------------------------------------------------------------


def _resolve_pixel_size(
    metadata_xy: tuple[float, float] | None,
    override: float | None,
    *,
    role: str,
) -> tuple[float, Literal["metadata", "override"], list[str]]:
    """Choose the isotropic pixel size of an image.

    Args:
        metadata_xy: The ``(x, y)`` pixel size recorded in the file, if any.
        override: The samplesheet pixel size, if any.
        role: ``"reference"`` or ``"moving"`` (for messages).

    Returns:
        ``(pixel_size_um, source, warnings)``.

    Raises:
        ValueError: If the file records none (and no override) or X and Y differ
            (and no override).
    """
    warnings: list[str] = []
    metadata: float | None
    anisotropic = False
    mx = my = 0.0
    if metadata_xy is not None:
        mx, my = metadata_xy
        anisotropic = abs(mx - my) / max(mx, my) > PIXEL_SIZE_ANISOTROPY_TOLERANCE
        metadata = _round_pixel_size((mx + my) / 2)
    else:
        metadata = None
    if override is not None:
        value = _round_pixel_size(override)
        if (
            metadata is not None
            and abs(value / metadata - 1) > PIXEL_SIZE_OVERRIDE_TOLERANCE
        ):
            warnings.append(
                f"{role}: pixel size override {value:g} um differs from the file "
                f"metadata {metadata:g} um by more than 1 %"
            )
        return value, "override", warnings
    if metadata is None:
        raise ValueError(
            f"{role}: the image records no pixel size; set {role}_pixel_size_um "
            "in the samplesheet"
        )
    if anisotropic:
        raise ValueError(
            f"{role}: X and Y pixel sizes differ ({mx:g} vs {my:g} um); "
            f"set {role}_pixel_size_um in the samplesheet to choose one"
        )
    return metadata, "metadata", warnings


def _usable_levels(level_shapes_yx: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Levels whose integer downsampling factor divides ``CROP_SNAP_PX``."""
    h0, w0 = level_shapes_yx[0]
    usable = [(int(h0), int(w0))]
    for h, w in level_shapes_yx[1:]:
        fy, fx = h0 / h, w0 / w
        if (
            abs(fx - round(fx)) / fx < 0.01
            and abs(fy - round(fy)) / fy < 0.01
            and round(fx) == round(fy)
            and CROP_SNAP_PX % round(fx) == 0
        ):
            usable.append((int(h), int(w)))
    return usable


def _normalisation_reasons(
    *,
    is_tiled: bool,
    level_shapes_yx: Sequence[tuple[int, int]],
    metadata_pixel_size_um: float | None,
    override_pixel_size_um: float | None,
) -> list[str]:
    """Why an OME-TIFF input has to be rewritten (empty: use it in place).

    Args:
        is_tiled: Whether the level-0 page is tiled.
        level_shapes_yx: ``(h, w)`` of every level, level 0 first.
        metadata_pixel_size_um: Rounded mean of the file's X/Y pixel size, or
            ``None``.
        override_pixel_size_um: The rounded samplesheet pixel size, or ``None``.

    Returns:
        Any of ``untiled``, ``insufficient_pyramid``, ``pixel_size_missing`` and
        ``pixel_size_override``, in that order.
    """
    reasons: list[str] = []
    if not is_tiled:
        reasons.append("untiled")
    coarsest = _usable_levels(level_shapes_yx)[-1]
    if max(coarsest) > MAX_COARSEST_LEVEL_SIDE_PX:
        reasons.append("insufficient_pyramid")
    if metadata_pixel_size_um is None:
        reasons.append("pixel_size_missing")
    elif (
        override_pixel_size_um is not None
        and abs(override_pixel_size_um / metadata_pixel_size_um - 1)
        > PIXEL_SIZE_OVERRIDE_TOLERANCE
    ):
        reasons.append("pixel_size_override")
    return reasons


# ---------------------------------------------------------------------------
# Image probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ImageProbe:
    role: str
    source_path: Path
    source_format: Literal["czi", "ome_tiff"]
    source_fingerprint: str
    source_size_bytes: int
    scene_index: int | None
    shape_cyx: tuple[int, int, int]
    dtype: str
    metadata_pixel_size_xy: tuple[float, float] | None
    pixel_size_um: float
    pixel_size_source: Literal["metadata", "override"]
    names: ResolvedChannelNames
    source_channel_names: tuple[str, ...]
    warnings: tuple[str, ...]
    czi_info: CziFileInfo | None
    ome_info: OmeTiffInfo | None
    level0_is_tiled: bool | None
    level0_single_strip: bool | None


def _inspect_image(
    spec: ManualRegistrationImageInput,
    *,
    role: str,
    config: ManualRegistrationPrepareConfig,
) -> _ImageProbe:
    """Read an input's metadata (no pixels beyond its fingerprint).

    Raises:
        ValueError: On any problem, prefixed with the dataset id, role and file
            name.
    """
    path = Path(spec.path).expanduser().absolute()
    try:
        return _probe_image(spec, role=role, path=path)
    except Exception as exc:
        raise ValueError(
            f"[{config.dataset_name}] {role} image {path.name}: {exc}"
        ) from exc


def _probe_image(
    spec: ManualRegistrationImageInput, *, role: str, path: Path
) -> _ImageProbe:
    source_format = _image_format(path)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path.name}")
    fingerprint = file_fingerprint(path)
    size_bytes = path.stat().st_size

    czi_info: CziFileInfo | None = None
    ome_info: OmeTiffInfo | None = None
    scene_index: int | None = None
    is_tiled: bool | None = None
    single_strip: bool | None = None
    if source_format == "czi":
        czi_info = run_read_czi_info(path)
        scene_index = resolve_scene(czi_info.scenes, spec.scene)
        scene = next(s for s in czi_info.scenes if s.index == scene_index)
        candidates = czi_channel_candidates(czi_info)
        names = resolve_channel_names(candidates, override=spec.channel_names)
        source_channel_names = tuple(
            c.name or c.fluor or c.dye_name or c.short_name or f"c{i}"
            for i, c in enumerate(candidates)
        )
        metadata_xy = czi_info.pixel_size_um
        shape_cyx = (
            len(czi_info.channels),
            int(scene.bbox_xywh[3]),
            scene.bbox_xywh[2],
        )
        dtype = czi_info.dtype
    else:
        if spec.scene is not None:
            raise ValueError("scene is only valid for .czi inputs")
        ome_info = read_ome_tiff_info(path)
        with tifffile.TiffFile(path) as tif:
            keyframe = tif.series[0].levels[0].keyframe
            is_tiled = bool(keyframe.is_tiled)
            rows = int(keyframe.rowsperstrip)
            single_strip = (not is_tiled) and (
                rows == 0 or rows >= ome_info.shape_cyx[1]
            )
        names = resolve_channel_names(
            [ChannelNameCandidates(name=n) for n in ome_info.channel_names],
            override=spec.channel_names,
        )
        source_channel_names = tuple(ome_info.channel_names)
        metadata_xy = ome_info.pixel_size_um
        shape_cyx = ome_info.shape_cyx
        dtype = ome_info.dtype
    pixel_size, pixel_source, px_warnings = _resolve_pixel_size(
        metadata_xy, spec.pixel_size_um, role=role
    )
    resolve_registration_channel(names.names, spec.registration_channel)
    warnings = (*(f"{role}: {w}" for w in names.warnings), *px_warnings)
    return _ImageProbe(
        role=role,
        source_path=path,
        source_format=source_format,
        source_fingerprint=fingerprint,
        source_size_bytes=size_bytes,
        scene_index=scene_index,
        shape_cyx=(int(shape_cyx[0]), int(shape_cyx[1]), int(shape_cyx[2])),
        dtype=str(np.dtype(dtype)),
        metadata_pixel_size_xy=metadata_xy,
        pixel_size_um=pixel_size,
        pixel_size_source=pixel_source,
        names=names,
        source_channel_names=source_channel_names,
        warnings=warnings,
        czi_info=czi_info,
        ome_info=ome_info,
        level0_is_tiled=is_tiled,
        level0_single_strip=single_strip,
    )


# ---------------------------------------------------------------------------
# Conversion provenance
# ---------------------------------------------------------------------------


def _conversion_is_current(
    provenance_path: Path, output_path: Path, conversion: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return the stored provenance when it matches ``conversion`` and the output."""
    if not provenance_path.is_file() or not output_path.is_file():
        return None
    try:
        loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
        if loaded["schema_version"] != 1:
            return None
        if canonical_json(loaded["conversion"]) != canonical_json(conversion):
            return None
        if file_fingerprint(output_path) != loaded["output"]["fingerprint"]:
            return None
    except (ValueError, KeyError, TypeError, OSError) as exc:
        logger.warning("Ignoring unusable provenance %s: %s", provenance_path.name, exc)
        return None
    return dict(loaded)


def _write_provenance(
    config: ManualRegistrationPrepareConfig,
    role: str,
    *,
    conversion: Mapping[str, Any],
    output_path: Path,
    level_shapes_yx: Sequence[tuple[int, int]],
    czi_block: Mapping[str, Any] | None,
    seconds: float,
) -> str:
    """Write ``<id>_<role>.prepare.json`` and return the output fingerprint."""
    fingerprint = file_fingerprint(output_path)
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "schema_version": 1,
        "dataset_name": config.dataset_name,
        "role": role,
        "conversion": dict(conversion),
        "output": {
            "file_name": output_path.name,
            "fingerprint": fingerprint,
            "level_shapes_yx": [[int(h), int(w)] for h, w in level_shapes_yx],
            "size_bytes": output_path.stat().st_size,
        },
        "czi": None if czi_block is None else dict(czi_block),
        "created_at": utc_now(),
        "seconds": round(float(seconds), 1),
    }
    _write_json_atomic(_provenance_path(config, role), payload)
    return fingerprint


def _ome_band_reader(
    info: OmeTiffInfo, *, single_strip: bool
) -> Callable[[int, int, int], np.ndarray]:
    """Return ``read_band(c, y0, h)`` over level 0 of an OME-TIFF.

    With ``single_strip`` (one strip per plane) every plane is decoded once and
    cached, because the writer streams channel by channel.
    """
    level0 = open_ome_tiff_level(info, 0).data
    lock = threading.Lock()
    cached: dict[int, np.ndarray] = {}

    def read_band(c: int, y0: int, h: int) -> np.ndarray:
        if single_strip:
            with lock:
                if c not in cached:
                    cached.clear()
                    cached[c] = np.asarray(level0[c].compute(scheduler="synchronous"))
                plane = cached[c]
            return np.ascontiguousarray(plane[y0 : y0 + h])
        return np.asarray(level0[c, y0 : y0 + h, :].compute(scheduler="synchronous"))

    return read_band


def _conversion_settings(
    probe: _ImageProbe,
    config: ManualRegistrationPrepareConfig,
    *,
    converter: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "source_format": probe.source_format,
        "source_fingerprint": probe.source_fingerprint,
        "scene_index": probe.scene_index,
        "channel_names": list(probe.names.names),
        "pixel_size_um": probe.pixel_size_um,
        "image_name": f"{config.dataset_name}_{probe.role}",
        "tile_size": config.tile_size,
        "band_height": config.band_height,
        "min_level_size": config.min_level_size,
        "compression": config.compression,
        "compression_level": config.compression_level,
        "converter": converter,
        "reasons": list(reasons),
    }


def _level_shapes_yx(info: OmeTiffInfo) -> list[tuple[int, int]]:
    return [(h, w) for (_, h, w) in info.level_shapes_cyx]


# ---------------------------------------------------------------------------
# prepare_image
# ---------------------------------------------------------------------------


def prepare_image(
    spec: ManualRegistrationImageInput,
    *,
    role: str,
    config: ManualRegistrationPrepareConfig,
) -> PreparedImage:
    """Convert or validate one input image into a tiled pyramidal OME-TIFF.

    A CZI scene is converted to ``<prepared_dir>/<id>_<role>.ome.tif``. An
    OME-TIFF is used in place when it is tiled, has a usable coarse level and its
    pixel size needs no change; otherwise it is rewritten to the same path.
    Conversions are skipped when ``<id>_<role>.prepare.json`` matches.

    Args:
        spec: The input image of the samplesheet row.
        role: ``"reference"`` or ``"moving"``.
        config: The prepare configuration.

    Returns:
        The prepared image and its store record.

    Raises:
        ValueError: On any user error (the message names id, role and file).
        RuntimeError: If the written file does not have the expected shape,
            dtype, names or pixel size (a bug).
    """
    if role not in ROLES:
        raise ValueError(f"role must be 'reference' or 'moving', not {role!r}")
    probe = _inspect_image(spec, role=role, config=config)
    out = _output_ome_tiff_path(config, role)
    prov_path = _provenance_path(config, role)
    config.prepared_dir.mkdir(parents=True, exist_ok=True)
    warnings = list(probe.warnings)
    czi_block: dict[str, Any] | None
    converter_settings: dict[str, Any] | None
    fingerprint: str | None = None

    if probe.source_format == "czi":
        assert probe.czi_info is not None and probe.scene_index is not None
        conversion = _conversion_settings(
            probe,
            config,
            converter=f"czifile=={probe.czi_info.czifile_version}",
            reasons=[],
        )
        current = _conversion_is_current(prov_path, out, conversion)
        if current is None:
            with _writer_tmp_dir(config) as tmp:
                result = run_convert_czi_scene(
                    probe.source_path,
                    out,
                    scene=probe.scene_index,
                    channel_names=list(probe.names.names),
                    pixel_size_um=probe.pixel_size_um,
                    image_name=f"{config.dataset_name}_{role}",
                    tile_size=config.tile_size,
                    band_height=config.band_height,
                    min_level_size=config.min_level_size,
                    compression=config.compression,
                    compression_level=config.compression_level,
                    workers=_resolve_workers(config),
                    tmp_dir=tmp,
                )
            czi_block = {
                "scene_index": result.scene_index,
                "scene_name": result.scene_name,
                "bbox_xywh": list(result.bbox_xywh),
                "z_planes_projected": result.z_planes_projected,
                "pixel_type": result.pixel_type,
                "compression": result.compression,
            }
            fingerprint = _write_provenance(
                config,
                role,
                conversion=conversion,
                output_path=out,
                level_shapes_yx=[(h, w) for h, w in result.level_shapes_yx],
                czi_block=czi_block,
                seconds=result.seconds,
            )
            log_status(f"{config.dataset_name} {role}: converted the CZI scene")
        else:
            czi_block = dict(current["czi"])
            fingerprint = str(current["output"]["fingerprint"])
            logger.info(
                "%s %s: conversion is current; skipped", config.dataset_name, role
            )
        written_by_prepare = True
        converter_settings = _converter_settings(conversion)
    else:
        assert probe.ome_info is not None
        assert probe.level0_is_tiled is not None
        metadata_size = (
            _round_pixel_size(sum(probe.metadata_pixel_size_xy) / 2)
            if probe.metadata_pixel_size_xy is not None
            else None
        )
        override = (
            _round_pixel_size(spec.pixel_size_um)
            if spec.pixel_size_um is not None
            else None
        )
        reasons = _normalisation_reasons(
            is_tiled=probe.level0_is_tiled,
            level_shapes_yx=_level_shapes_yx(probe.ome_info),
            metadata_pixel_size_um=metadata_size,
            override_pixel_size_um=override,
        )
        czi_block = None
        if reasons:
            if probe.source_path.resolve() == out.resolve():
                raise ValueError(
                    f"[{config.dataset_name}] {role} image {probe.source_path.name}: "
                    f"it must be rewritten ({', '.join(reasons)}) but it is the "
                    "prepare output itself; point the samplesheet at the original "
                    "file"
                )
            conversion = _conversion_settings(
                probe, config, converter=NORMALISE_CONVERTER, reasons=reasons
            )
            current = _conversion_is_current(prov_path, out, conversion)
            if current is None:
                started = time.monotonic()
                with _writer_tmp_dir(config) as tmp:
                    result_w = write_pyramidal_ome_tiff(
                        _ome_band_reader(
                            probe.ome_info,
                            single_strip=bool(probe.level0_single_strip),
                        ),
                        shape_cyx=probe.shape_cyx,
                        dtype=np.dtype(probe.dtype),
                        channel_names=probe.names.names,
                        pixel_size_um=probe.pixel_size_um,
                        image_name=f"{config.dataset_name}_{role}",
                        output_path=out,
                        tile_size=config.tile_size,
                        band_height=config.band_height,
                        min_level_size=config.min_level_size,
                        compression=config.compression,
                        compression_level=config.compression_level,
                        workers=_resolve_workers(config),
                        tmp_dir=tmp,
                    )
                fingerprint = _write_provenance(
                    config,
                    role,
                    conversion=conversion,
                    output_path=out,
                    level_shapes_yx=list(result_w.level_shapes_yx),
                    czi_block=None,
                    seconds=time.monotonic() - started,
                )
                log_status(f"{config.dataset_name} {role}: normalised the OME-TIFF")
            else:
                fingerprint = str(current["output"]["fingerprint"])
                logger.info(
                    "%s %s: conversion is current; skipped", config.dataset_name, role
                )
            written_by_prepare = True
            converter_settings = _converter_settings(conversion)
            warnings.append(
                f"{role}: rewritten as a tiled pyramidal OME-TIFF "
                f"({', '.join(reasons)})"
            )
        else:
            out = probe.source_path
            written_by_prepare = False
            converter_settings = None

    info = read_ome_tiff_info(out)
    if info.shape_cyx != probe.shape_cyx:
        raise RuntimeError(
            f"converted image has shape {info.shape_cyx}, expected {probe.shape_cyx}"
        )
    if np.dtype(info.dtype) != np.dtype(probe.dtype):
        raise RuntimeError(
            f"converted image has dtype {info.dtype}, expected {probe.dtype}"
        )
    if written_by_prepare:
        if tuple(info.channel_names) != probe.names.names:
            raise RuntimeError(
                f"converted image has channel names {list(info.channel_names)}, "
                f"expected {list(probe.names.names)}"
            )
        if info.pixel_size_um is None or any(
            abs(value / probe.pixel_size_um - 1) > 1e-6 for value in info.pixel_size_um
        ):
            raise RuntimeError(
                f"converted image has pixel size {info.pixel_size_um}, "
                f"expected {probe.pixel_size_um}"
            )
    registration_channel = resolve_registration_channel(
        probe.names.names, spec.registration_channel
    )
    ome_fingerprint = (
        fingerprint
        if written_by_prepare and fingerprint is not None
        else probe.source_fingerprint
    )
    height, width = info.shape_cyx[1], info.shape_cyx[2]
    record: dict[str, Any] = {
        "element": role,
        "role": role,
        "shape_yx": [height, width],
        "dtype": str(np.dtype(info.dtype)),
        "pixel_size_um": probe.pixel_size_um,
        "pixel_size_source": probe.pixel_size_source,
        "registration_channel": registration_channel,
        "source_channel_names": list(probe.source_channel_names),
        "canonical_channel_names": list(probe.names.names),
        "name_sources": list(probe.names.sources),
        "ome_tiff": {
            "path": str(out.absolute()),
            "fingerprint": ome_fingerprint,
            "written_by_prepare": written_by_prepare,
            "level_shapes_yx": [[h, w] for (_, h, w) in info.level_shapes_cyx],
            "exact_level_factors": written_by_prepare,
        },
        "source": {
            "format": probe.source_format,
            "path": str(probe.source_path),
            "fingerprint": probe.source_fingerprint,
            "size_bytes": probe.source_size_bytes,
        },
        "czi": czi_block,
    }
    for warning in warnings:
        logger.warning("[%s] %s", config.dataset_name, warning)
    return PreparedImage(
        role=role,
        ome_tiff_path=out,
        info=info,
        written_by_prepare=written_by_prepare,
        shape_yx=(height, width),
        pixel_size_um=probe.pixel_size_um,
        pixel_size_source=probe.pixel_size_source,
        dtype=record["dtype"],
        canonical_channel_names=probe.names.names,
        name_sources=probe.names.sources,
        source_channel_names=probe.source_channel_names,
        registration_channel=registration_channel,
        record=record,
        warnings=tuple(warnings),
        converter_settings=converter_settings,
    )


def _converter_settings(conversion: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a conversion record that decides the OME-TIFF content."""
    keys = (
        "tile_size",
        "band_height",
        "min_level_size",
        "compression",
        "compression_level",
        "converter",
    )
    return {key: conversion[key] for key in keys}


# ---------------------------------------------------------------------------
# Channel matching and digests
# ---------------------------------------------------------------------------


def _channel_matching(
    reference: PreparedImage, moving: PreparedImage
) -> tuple[list[int], dict[str, list[str]]]:
    """Choose the reference channels kept in the store and describe the matching."""
    kept = shared_channel_subset(
        reference.canonical_channel_names,
        moving.canonical_channel_names,
        reference_registration_channel=reference.registration_channel,
    )
    ref_names = reference.canonical_channel_names
    reference_kept = [ref_names[i] for i in kept]
    kept_set = set(kept)
    reference_dropped = [n for i, n in enumerate(ref_names) if i not in kept_set]
    ref_keys = {stain_key(n) for n in ref_names}
    moving_unmatched = [
        n for n in moving.canonical_channel_names if stain_key(n) not in ref_keys
    ]
    return kept, {
        "reference_kept": reference_kept,
        "reference_dropped": reference_dropped,
        "moving_unmatched": moving_unmatched,
    }


def _grid_entry(
    *,
    source_fingerprint: str,
    scene_index: int | None,
    shape_yx: Sequence[int],
    pixel_size_um: float,
) -> dict[str, Any]:
    return {
        "source_fingerprint": source_fingerprint,
        "scene_index": scene_index,
        "shape_yx": [int(shape_yx[0]), int(shape_yx[1])],
        "pixel_size_um": float(pixel_size_um),
    }


def _grid_entries_from_records(
    images: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        record = images[role]
        czi = record.get("czi")
        entries[role] = _grid_entry(
            source_fingerprint=record["source"]["fingerprint"],
            scene_index=None if czi is None else czi["scene_index"],
            shape_yx=record["shape_yx"],
            pixel_size_um=record["pixel_size_um"],
        )
    return entries


def _grid_digest(entries: Mapping[str, Mapping[str, Any]]) -> str:
    return canonical_sha256({role: dict(entries[role]) for role in ROLES})


def _content_digest(
    *,
    grid_digest: str,
    images: Mapping[str, Mapping[str, Any]],
    prepared: Mapping[str, PreparedImage],
    chunk_size: int,
) -> str:
    return canonical_sha256(
        {
            "grid_digest": grid_digest,
            "images": {
                role: {
                    "format": prepared[role].record["source"]["format"],
                    "channel_names": list(prepared[role].canonical_channel_names),
                    "kept_channel_names": list(images[role]["channel_names"]),
                    "registration_channel": prepared[role].registration_channel,
                    "ome_tiff_fingerprint": prepared[role].record["ome_tiff"][
                        "fingerprint"
                    ],
                    # The recorded locations: an input that moved (same bytes)
                    # must rebuild so the store points at the new path.
                    "ome_tiff_path": prepared[role].record["ome_tiff"]["path"],
                    "source_path": prepared[role].record["source"]["path"],
                    "converter_settings": prepared[role].converter_settings
                    if prepared[role].written_by_prepare
                    else None,
                }
                for role in ROLES
            },
            "chunk_size": chunk_size,
            "version": PREPARE_ALGORITHM,
        }
    )


# ---------------------------------------------------------------------------
# Store update plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StorePlan:
    action: Literal["unchanged", "build", "refuse"]
    status: Literal["built", "unchanged", "rebuilt_kept_labelling", "forced"] | None
    carry_labelling: dict[str, Any] | None
    old_store: Literal["none", "delete", "supersede"]
    message: str | None
    warnings: tuple[str, ...] = ()


def _short_digest(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("sha256:"):
        return value[len("sha256:") : len("sha256:") + 12] + "…"
    return value


def _grid_changes(
    old_images: Mapping[str, Any], new_entries: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Describe each differing grid field as ``<role>.<field>: <old> -> <new>``."""
    old_entries: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        record = old_images.get(role) if isinstance(old_images, Mapping) else None
        record = record if isinstance(record, Mapping) else {}
        source = record.get("source")
        czi = record.get("czi")
        old_entries[role] = {
            "source_fingerprint": source.get("fingerprint")
            if isinstance(source, Mapping)
            else None,
            "scene_index": czi.get("scene_index") if isinstance(czi, Mapping) else None,
            "shape_yx": record.get("shape_yx"),
            "pixel_size_um": record.get("pixel_size_um"),
        }
    changes = []
    for role in ROLES:
        for field in _GRID_FIELDS:
            old, new = old_entries[role][field], new_entries[role][field]
            if old != new:
                changes.append(
                    f"{role}.{field}: {_short_digest(old)} -> {_short_digest(new)}"
                )
    return changes


def _shapes_and_pixel_sizes_match(
    old_images: Mapping[str, Any], new_entries: Mapping[str, Mapping[str, Any]]
) -> str | None:
    """Return the first role whose shape or pixel size changed, else ``None``."""
    for role in ROLES:
        record = old_images.get(role) if isinstance(old_images, Mapping) else None
        record = record if isinstance(record, Mapping) else {}
        old_shape = record.get("shape_yx")
        if (
            old_shape is None
            or list(old_shape) != list(new_entries[role]["shape_yx"])
            or record.get("pixel_size_um") != new_entries[role]["pixel_size_um"]
        ):
            return role
    return None


def _plan_store_update(
    *,
    dataset_name: str,
    store_path: Path,
    exists: bool,
    intact: bool,
    old_attrs: Mapping[str, Any],
    new_grid_entries: Mapping[str, Mapping[str, Any]],
    grid_digest: str,
    content_digest: str | None,
    force: bool,
) -> _StorePlan:
    """Decide what to do with an existing store (pure).

    The rules, top to bottom: no store: build. Not a prepared store, or written
    by a newer version: refuse (force: build, keep the old one as superseded).
    A labelling attribute that is not a mapping: refuse (force: build without
    it, keep the old store as superseded).
    Force: build, carry the labelling when both images keep shape and pixel size.
    Same content and intact: nothing to do. Same grid: rebuild, carry the
    labelling. Changed grid under a labelling: refuse. Changed grid, no
    labelling: rebuild.
    """
    name = store_path.name
    if not exists:
        return _StorePlan("build", "built", None, "none", None)

    store_attr = old_attrs.get(STORE_ATTR)
    labelling = old_attrs.get(LABELLING_ATTR)
    has_labelling = LABELLING_ATTR in old_attrs
    labelling_dict = dict(labelling) if isinstance(labelling, Mapping) else None

    if (
        not isinstance(store_attr, Mapping)
        or store_attr.get("schema") != STORE_SCHEMA
        or store_attr.get("kind") != "prepared"
    ):
        if not force:
            return _StorePlan(
                "refuse",
                None,
                None,
                "none",
                f"[{dataset_name}] {store_path} exists but is not a MerXen prepared "
                f'store (no "{STORE_ATTR}" attribute of kind "prepared"). Move it '
                "away, or set --manual_registration_force_prepare true (CLI: "
                f"--force) to keep it as {name}.superseded-<UTC> and build a new "
                "store.",
            )
        return _StorePlan("build", "forced", None, "supersede", None)

    def _version(value: Any) -> int:
        version = value.get("schema_version") if isinstance(value, Mapping) else None
        return version if isinstance(version, int) else 0

    newest = max(_version(store_attr), _version(labelling))
    if newest > SCHEMA_VERSION:
        if not force:
            return _StorePlan(
                "refuse",
                None,
                None,
                "none",
                f"[{dataset_name}] {store_path} was written by a newer "
                f"MerXen/viewer (schema_version {newest}); update MerXen, or "
                "force the rebuild.",
            )
        return _StorePlan("build", "forced", None, "supersede", None)

    if has_labelling and labelling_dict is None:
        # Unreadable labelling: never carry it silently, never delete the store.
        if not force:
            return _StorePlan(
                "refuse",
                None,
                None,
                "none",
                f"[{dataset_name}] {store_path} has a labelling attribute "
                f'("{LABELLING_ATTR}") that is not a mapping, so it cannot be '
                "carried over. Repair it, or set "
                "--manual_registration_force_prepare true (CLI: --force) to keep "
                f"the old store as {name}.superseded-<UTC> and build a new one "
                "without the labelling.",
            )
        return _StorePlan(
            "build",
            "forced",
            None,
            "supersede",
            None,
            (
                "labelling not carried over: it was not readable; it remains in "
                "{superseded}",
            ),
        )

    old_images = store_attr.get("images", {})
    if force:
        carry: dict[str, Any] | None = None
        warnings: tuple[str, ...] = ()
        if has_labelling:
            changed = _shapes_and_pixel_sizes_match(old_images, new_grid_entries)
            if changed is None:
                carry = labelling_dict
            else:
                warnings = (
                    f"labelling not carried over: the {changed} image changed "
                    "shape or pixel size; it remains in {superseded}",
                )
        return _StorePlan("build", "forced", carry, "supersede", None, warnings)

    if (
        content_digest is not None
        and content_digest == store_attr.get("content_digest")
        and intact
    ):
        return _StorePlan("unchanged", "unchanged", None, "none", None)

    if grid_digest == store_attr.get("grid_digest"):
        return _StorePlan(
            "build",
            "rebuilt_kept_labelling" if has_labelling else "built",
            labelling_dict if has_labelling else None,
            "delete",
            None,
        )

    if has_labelling:
        status = labelling_dict.get("status") if labelling_dict else None
        revision = labelling_dict.get("revision") if labelling_dict else None
        changes = "; ".join(_grid_changes(old_images, new_grid_entries)) or (
            "image grid digest changed"
        )
        return _StorePlan(
            "refuse",
            None,
            None,
            "none",
            f"[{dataset_name}] {store_path} holds landmark labelling (status "
            f"{status}, revision {revision}) made on a different image grid: "
            f"{changes}. Re-running prepare would invalidate those landmarks. "
            "Restore the original inputs and settings, or set "
            "--manual_registration_force_prepare true (CLI: --force) to keep the "
            f"old store as {name}.superseded-<UTC> and build a new one.",
        )
    return _StorePlan("build", "built", None, "delete", None)


def _store_is_intact(store_path: Path) -> bool:
    return (store_path / "images" / "reference").is_dir() and (
        store_path / "images" / "moving"
    ).is_dir()


# ---------------------------------------------------------------------------
# build_prepared_store
# ---------------------------------------------------------------------------


def _image_element(
    image: PreparedImage, channel_indices: Sequence[int], chunk_size: int
) -> xr.DataArray:
    level0 = open_ome_tiff_level(
        image.info, 0, channel_names=list(image.canonical_channel_names)
    )
    names = [image.canonical_channel_names[i] for i in channel_indices]
    element: xr.DataArray = Image2DModel.parse(
        level0.isel(c=list(channel_indices)).data,
        dims=("c", "y", "x"),
        c_coords=names,  # spatialdata drops the DataArray's channel coordinates
        transformations={
            "global": Scale([image.pixel_size_um, image.pixel_size_um], axes=("x", "y"))
        },
        chunks=(1, chunk_size, chunk_size),
    )
    return element


def _superseded_path(store_path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")  # no colons in file names
    candidate = store_path.with_name(f"{store_path.name}.superseded-{stamp}")
    n = 2
    while candidate.exists():
        candidate = store_path.with_name(f"{store_path.name}.superseded-{stamp}-{n}")
        n += 1
    return candidate


def _remove_stale_temporaries(store_path: Path) -> None:
    """Delete ``.<name>.tmp-*`` and ``.<name>.old-*`` siblings older than 24 h."""
    parent = store_path.parent
    if not parent.is_dir():
        return
    cutoff = time.time() - STALE_TEMP_AGE_SECONDS
    for pattern in (f".{store_path.name}.tmp-*", f".{store_path.name}.old-*"):
        for stale in parent.glob(pattern):
            try:
                if stale.stat().st_mtime >= cutoff:
                    continue
                if stale.is_dir() and not stale.is_symlink():
                    shutil.rmtree(stale)
                else:
                    stale.unlink()
                logger.info("Removed stale prepare leftover %s", stale.name)
            except OSError as exc:
                logger.warning(
                    "Could not remove stale leftover %s: %s", stale.name, exc
                )


def build_prepared_store(
    *,
    config: ManualRegistrationPrepareConfig,
    reference: PreparedImage,
    moving: PreparedImage,
    details_out: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    """Build (or keep) the image-only prepared store.

    The store is never modified in place: a new store is written to a private
    sibling and renamed into place under the store's write lock, after the
    labelling of the old store has been re-read and carried over.

    Args:
        config: The prepare configuration.
        reference: The prepared reference image.
        moving: The prepared moving image.
        details_out: When given, filled with ``warnings``, ``superseded_store``,
            ``grid_digest`` and ``content_digest``.

    Returns:
        ``(store_path, status)``; status is ``"built"``, ``"unchanged"``,
        ``"rebuilt_kept_labelling"`` or ``"forced"``.

    Raises:
        ValueError: If the existing store is refused (see ``_plan_store_update``).
    """
    store_path = config.store_path
    config.prepared_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_temporaries(store_path)

    kept_ref, matching = _channel_matching(reference, moving)
    images = {
        "reference": {**reference.record, "channel_names": matching["reference_kept"]},
        "moving": {
            **moving.record,
            "channel_names": list(moving.canonical_channel_names),
        },
    }
    prepared = {"reference": reference, "moving": moving}
    grid_entries = _grid_entries_from_records(images)
    grid_digest = _grid_digest(grid_entries)
    content_digest = _content_digest(
        grid_digest=grid_digest,
        images=images,
        prepared=prepared,
        chunk_size=config.chunk_size,
    )

    def plan_with(old_attrs: Mapping[str, Any]) -> _StorePlan:
        return _plan_store_update(
            dataset_name=config.dataset_name,
            store_path=store_path,
            exists=store_path.exists(),
            intact=_store_is_intact(store_path),
            old_attrs=old_attrs,
            new_grid_entries=grid_entries,
            grid_digest=grid_digest,
            content_digest=content_digest,
            force=config.force,
        )

    plan = plan_with(_safe_root_attrs(store_path) if store_path.exists() else {})
    if plan.action == "refuse":
        raise ValueError(plan.message)
    if details_out is not None:
        details_out.update(
            {
                "warnings": [],
                "superseded_store": None,
                "grid_digest": grid_digest,
                "content_digest": content_digest,
            }
        )
    if plan.action == "unchanged":
        logger.info("%s: the prepared store is up to date", config.dataset_name)
        return store_path, "unchanged"

    payload = {
        "schema": STORE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": "prepared",
        "dataset_name": config.dataset_name,
        "created_at": utc_now(),
        "merxen_version": merxen.__version__,
        "conventions": dict(CONVENTIONS),
        "images": images,
        "channel_matching": matching,
        "grid_digest": grid_digest,
        "content_digest": content_digest,
    }
    StoreAttrs.model_validate(payload)  # fail before writing anything

    tmp = config.prepared_dir / f".{store_path.name}.tmp-{uuid.uuid4().hex}"
    aside: Path | None = None
    status = plan.status or "built"
    replan = plan
    try:
        sdata = SpatialData(
            images={
                "reference": _image_element(reference, kept_ref, config.chunk_size),
                "moving": _image_element(
                    moving,
                    list(range(len(moving.canonical_channel_names))),
                    config.chunk_size,
                ),
            }
        )
        sdata.attrs[STORE_ATTR] = payload
        if plan.carry_labelling is not None:
            sdata.attrs[LABELLING_ATTR] = plan.carry_labelling
        stamp_merxen_schema(sdata)
        with dask.config.set(scheduler="threads", num_workers=_resolve_workers(config)):
            sdata.write(tmp)
        del sdata

        with spatialdata_write_lock(store_path):
            # read_root_attrs, not read_root_attrs_locked: the lock is held.
            fresh = _safe_root_attrs(store_path) if store_path.exists() else {}
            replan = plan_with(fresh)
            if replan.action == "refuse":
                raise ValueError(replan.message)
            if replan.action == "unchanged":
                status = "unchanged"
            else:
                status = replan.status or "built"
                fresh_labelling = replan.carry_labelling
                if fresh_labelling is not None and canonical_json(
                    fresh_labelling
                ) != canonical_json(plan.carry_labelling):
                    # The viewer saved meanwhile. write_root_attr needs the lock
                    # of the store it writes, so take the temporary store's own
                    # (a different lock file: nesting cannot deadlock).
                    with spatialdata_write_lock(tmp):
                        write_root_attr(tmp, LABELLING_ATTR, fresh_labelling)
                if store_path.exists():
                    aside = (
                        _superseded_path(store_path)
                        if replan.old_store == "supersede"
                        else store_path.with_name(
                            f".{store_path.name}.old-{uuid.uuid4().hex}"
                        )
                    )
                    os.rename(store_path, aside)
                os.rename(tmp, store_path)
        if aside is not None and replan.old_store == "delete":
            try:
                shutil.rmtree(aside, ignore_errors=False)
            except OSError as exc:
                logger.warning("Could not remove the replaced store %s: %s", aside, exc)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        spatialdata_write_lock_path(tmp).unlink(missing_ok=True)

    if details_out is not None:
        superseded = str(aside) if replan.old_store == "supersede" and aside else None
        details_out["superseded_store"] = superseded
        details_out["warnings"] = [
            w.replace("{superseded}", superseded or "the superseded store")
            for w in replan.warnings
        ]
    log_status(f"{config.dataset_name}: prepared store {status}")
    return store_path, status


# ---------------------------------------------------------------------------
# run_manual_registration_prepare
# ---------------------------------------------------------------------------


def run_manual_registration_prepare(
    config: ManualRegistrationPrepareConfig,
) -> dict[str, Path]:
    """Prepare both images and build the prepared store.

    A metadata-only preflight runs first, so a refused run never overwrites an
    OME-TIFF that a labelled store references.

    Args:
        config: The prepare configuration.

    Returns:
        ``{"store", "summary", "durable_summary", "reference_ome_tiff",
        "moving_ome_tiff"}``.

    Raises:
        ValueError: On any user error or a refused store update.
    """
    config.prepared_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = config.dataset_name

    probes = {
        role: _inspect_image(getattr(config, role), role=role, config=config)
        for role in ROLES
    }
    reference_source = probes["reference"].source_path
    if reference_source.resolve() == probes["moving"].source_path.resolve():
        raise ValueError(
            f"[{dataset}] reference and moving are the same file "
            f"({reference_source.name})"
        )
    grid_entries: dict[str, dict[str, Any]] = {
        role: _grid_entry(
            source_fingerprint=probe.source_fingerprint,
            scene_index=probe.scene_index,
            shape_yx=probe.shape_cyx[1:],
            pixel_size_um=probe.pixel_size_um,
        )
        for role, probe in probes.items()
    }
    store_path = config.store_path
    if store_path.exists():
        plan = _plan_store_update(
            dataset_name=dataset,
            store_path=store_path,
            exists=True,
            intact=_store_is_intact(store_path),
            old_attrs=_safe_root_attrs(store_path),
            new_grid_entries=grid_entries,
            grid_digest=_grid_digest(grid_entries),
            content_digest=None,
            force=config.force,
        )
        if plan.action == "refuse":
            raise ValueError(plan.message)

    reference = prepare_image(config.reference, role="reference", config=config)
    moving = prepare_image(config.moving, role="moving", config=config)

    details: dict[str, Any] = {}
    store_path, status = build_prepared_store(
        config=config, reference=reference, moving=moving, details_out=details
    )
    store_attr = read_root_attrs(store_path)[STORE_ATTR]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": 1,
        "dataset_name": dataset,
        "status": status,
        "store_path": str(store_path),
        "created_at": utc_now(),
        "images": store_attr["images"],
        "channel_matching": store_attr["channel_matching"],
        "grid_digest": store_attr["grid_digest"],
        "content_digest": store_attr["content_digest"],
        "superseded_store": details.get("superseded_store"),
        "warnings": [
            *reference.warnings,
            *moving.warnings,
            *_matching_warnings(store_attr["channel_matching"]),
            *details.get("warnings", []),
        ],
    }
    summary_path = config.output_dir / "prepare_summary.json"
    durable_path = config.prepared_dir / f"{dataset}_prepare_summary.json"
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(durable_path, summary)
    logger.info(
        "Open in the viewer: Dataset loader > Manual Registration > Load image "
        "data > %s",
        store_path,
    )
    return {
        "store": store_path,
        "summary": summary_path,
        "durable_summary": durable_path,
        "reference_ome_tiff": reference.ome_tiff_path,
        "moving_ome_tiff": moving.ome_tiff_path,
    }


def _matching_warnings(matching: Mapping[str, Sequence[str]]) -> list[str]:
    warnings = []
    if matching["reference_dropped"]:
        warnings.append(
            "reference channels not in the moving image are left out of the store: "
            f"{list(matching['reference_dropped'])}"
        )
    if matching["moving_unmatched"]:
        warnings.append(
            "moving channels not in the reference image: "
            f"{list(matching['moving_unmatched'])}"
        )
    return warnings
