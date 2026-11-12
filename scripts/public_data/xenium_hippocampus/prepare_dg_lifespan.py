"""Turn the dentate-gyrus lifespan downloads into MerXen-ready Xenium folders.

Input: ``$MERXEN_XENIUM_PUBLIC_ROOT/xenium_prime_5k_proto_human_dentate_gyrus_ff/raw``
as written by ``fetch_dg_lifespan.py`` (``raw/listing.json`` plus
``raw/<GSM>/<file>``). The raw files are only read, never changed.

Output: one folder per section, ``.../dg_lifespan/xenium/<pair_id>/``, that
the spatialdata-io 0.6 Xenium reader behind MerXen's ``build_spatialdata``
stage accepts with MerXen's fixed reader options:

- ``cell_feature_matrix.h5``: hard link to the download (symlink or copy with
  ``--link-mode``).
- ``cells.parquet``, ``cell_boundaries.parquet``,
  ``nucleus_boundaries.parquet``, ``transcripts.parquet``: decompressed from
  the ``.gz`` downloads (``pigz``/``gzip`` check the CRC; the size is checked
  against the gzip trailer).
- ``morphology_focus/morphology_focus_0000.ome.tif``: a 2-D projection of the
  deposited 3-D DAPI z-stack (``--projection mip``, the default, or
  ``focus``: the sharpest plane per tile). The pixel size is kept and the
  channel is named ``DAPI`` (the name spatialdata-io gives it anyway); tiled,
  zlib-compressed, with a SubIFD pyramid like the vendor files.
  The decompressed z-stack is deleted afterwards unless ``--keep-zstack``.
- ``cells.zarr.zip``: nucleus and cell label masks rasterised from the vendor
  polygons at the morphology resolution (vendor conventions: corner pixels,
  cell label = cell row + 1, nucleus label = nucleus-polygon row + 1), with
  ``cell_id``, ``polygon_sets/{0,1}/cell_index`` and ``cell_summary`` (the
  deposit has no ``cells.zarr.zip``, but MerXen reads the labels).
- ``experiment.xenium`` and ``gene_panel.json``: synthesised (none were
  deposited); ``merxen_value_sources`` in ``experiment.xenium`` names the
  source of every value. ``analysis_sw_version`` is ``xenium-3.0.0.0`` (the
  paper's prototype XOA 3.0.0.0), which puts the reader on its XOA 2/3 branch
  so the image channel is named ``DAPI`` (MerXen's nuclei pass needs that
  name).
- ``merxen_provenance.json``: what was derived, how, and from which files.

Each output is written under a temporary name and renamed when complete, so
a re-run skips finished outputs (``--force`` redoes them) and keeps their
provenance. A section is refused when its estimated new bytes plus
``--margin-gb`` do not fit. ``--dry-run`` reports the planned actions, output
sizes and free space and writes nothing.

Usage::

    python scripts/public_data/xenium_hippocampus/prepare_dg_lifespan.py --dry-run
    python scripts/public_data/xenium_hippocampus/prepare_dg_lifespan.py --sections teen
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from merxen import local_config

logger = logging.getLogger("prepare_dg_lifespan")

DATASET_KEY: Final = "dg_lifespan"
DATASET_DIR: Final = "xenium_prime_5k_proto_human_dentate_gyrus_ff"
LISTING_FILE: Final = "listing.json"
PROVENANCE_FILE: Final = "merxen_provenance.json"
WORK_DIR: Final = ".merxen_work"
#: Prototype XOA 3.0.0.0 (paper methods 4.2.1); parses as >= 2.0 in the reader.
SYNTHETIC_ANALYSIS_SW_VERSION: Final = "xenium-3.0.0.0"
PANEL_NAME: Final = (
    "Xenium Prime 5K Human Pan Tissue & Pathways (pre-commercial prototype)"
)
GENE_FEATURE_TYPE: Final = "Gene Expression"
NEGATIVE_PROBE_FEATURE_TYPE: Final = "Negative Control Probe"
CHANNEL_NAME: Final = "DAPI"
#: Nominal Xenium pixel size; only for dry-run estimates. The real value is
#: read from the deposited TIFF.
NOMINAL_PIXEL_SIZE_UM: Final = 0.2125
GZIP_ISIZE_MODULUS: Final = 2**32
#: Fixed-point bits for sub-pixel polygon vertices in cv2.fillPoly.
FILL_SHIFT: Final = 4
CELL_SUMMARY_COLUMNS: Final = (
    "cell_centroid_x",
    "cell_centroid_y",
    "cell_area",
    "nucleus_centroid_x",
    "nucleus_centroid_y",
    "nucleus_area",
    "z_level",
    "nucleus_count",
)
#: Decompressed outputs: role -> file name in the Xenium folder.
GUNZIP_OUTPUTS: Final[dict[str, str]] = {
    "cells": "cells.parquet",
    "cell_boundaries": "cell_boundaries.parquet",
    "nucleus_boundaries": "nucleus_boundaries.parquet",
    "transcripts": "transcripts.parquet",
}
REQUIRED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "cells.parquet": ("cell_id", "x_centroid", "y_centroid"),
    "cell_boundaries.parquet": ("cell_id", "vertex_x", "vertex_y"),
    "nucleus_boundaries.parquet": ("cell_id", "vertex_x", "vertex_y"),
    "transcripts.parquet": (
        "x_location",
        "y_location",
        "z_location",
        "feature_name",
        "cell_id",
        "qv",
    ),
}
FOCUS_FILE: Final = Path("morphology_focus") / "morphology_focus_0000.ome.tif"


class PrepareError(RuntimeError):
    """Raised when an input is missing or a conversion check fails."""


@dataclass
class SectionInputs:
    """The raw files of one section.

    Attributes:
        accession: GSM accession.
        title: GEO title.
        label: Short label.
        pair_id: MerXen pair id (output folder name).
        files: Local path per role.
        sizes: Expected byte size per role (from the listing).
        isizes: Gzip trailer per role, when the listing has it.
    """

    accession: str
    title: str
    label: str
    pair_id: str
    files: dict[str, Path]
    sizes: dict[str, int]
    isizes: dict[str, int | None] = field(default_factory=dict)


@dataclass
class ImageInfo:
    """The projected morphology image.

    Attributes:
        height: Rows.
        width: Columns.
        dtype: Pixel dtype name.
        pixel_size_um: Micrometres per pixel.
        channel_name: Channel name written to the OME-XML. Always ``DAPI``:
            spatialdata-io names a one-file XOA 2/3 ``morphology_focus/``
            channel ``DAPI`` whatever the OME-XML says, so the samplesheet's
            ``xenium_channels`` must be ``DAPI``.
        source_channel_name: Name of the projected channel in the source
            OME-XML (empty when it had none).
        pixel_size_source: Where ``pixel_size_um`` came from.
        method: Projection method.
        source_axes: Axes of the source series.
        source_shape: Shape of the source series.
        planes_used: Number of source planes projected.
    """

    height: int
    width: int
    dtype: str
    pixel_size_um: float
    channel_name: str = CHANNEL_NAME
    source_channel_name: str = ""
    pixel_size_source: str = ""
    method: str = "mip"
    source_axes: str = ""
    source_shape: tuple[int, ...] = ()
    planes_used: int = 0


@dataclass
class PolygonSet:
    """Vendor polygons as fixed-point pixel vertices.

    Attributes:
        owners: Owning cell row in ``cells.parquet`` per polygon.
        labels: Mask value per polygon: the owning cell row + 1 for cells,
            the nucleus-polygon row + 1 for nuclei (the XOA >= 2.0 vendor
            convention: the owner of nucleus ``k`` is
            ``polygon_sets/0/cell_index[k - 1]``).
        starts: First vertex index per polygon.
        ends: One past the last vertex per polygon.
        fx: Vertex x in pixels times ``2**FILL_SHIFT`` (pixel-centre origin).
        fy: Vertex y, likewise.
        n_unknown_vertices: Vertices whose cell id is not in cells.parquet.
        n_label_id_mismatches: Polygons whose deposited ``label_id`` differs
            from the mask value written (None when there is no ``label_id``).
    """

    owners: np.ndarray
    labels: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    n_unknown_vertices: int = 0
    n_label_id_mismatches: int | None = None


@dataclass
class PlannedAction:
    """One step of the plan for the dry-run report.

    Attributes:
        description: What is done.
        output: Output path.
        estimated_bytes: Estimated output size, or None if unknown.
        is_done: Whether the output already exists.
        is_transient: Whether the output is deleted again by the run.
    """

    description: str
    output: Path
    estimated_bytes: int | None
    is_done: bool
    is_transient: bool = False


@dataclass
class Options:
    """Prepare options.

    Attributes:
        projection: ``mip`` or ``focus``.
        workers: Threads reading z-planes.
        tile_size: Output TIFF tile edge (multiple of 16).
        mask_chunk: Label-mask chunk edge.
        keep_zstack: Keep the decompressed 3-D TIFF as ``morphology.ome.tif``.
        link_mode: ``hardlink``, ``symlink`` or ``copy`` for the h5.
        pixel_size_um: Override of the TIFF pixel size.
        force: Redo outputs that already exist.
        margin_bytes: Free space to keep on top of the section's estimated
            new bytes (persistent plus transient peak); a section is refused
            when the file system has less.
    """

    projection: str = "mip"
    workers: int = 4
    tile_size: int = 1024
    mask_chunk: int = 4096
    keep_zstack: bool = False
    link_mode: str = "hardlink"
    pixel_size_um: float | None = None
    force: bool = False
    margin_bytes: int = 0


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def dataset_dir(public_root: Path) -> Path:
    """Return ``<public root>/xenium_prime_5k_proto_human_dentate_gyrus_ff``."""
    return public_root / DATASET_DIR


def load_sections(raw_dir: Path) -> list[SectionInputs]:
    """Read ``raw/listing.json`` into per-section inputs.

    Args:
        raw_dir: ``.../dg_lifespan/raw``.

    Returns:
        One entry per listed section (files may not exist yet).

    Raises:
        PrepareError: If the listing is missing.
    """
    listing_path = raw_dir / LISTING_FILE
    if not listing_path.is_file():
        raise PrepareError(
            f"{listing_path} is missing: run fetch_dg_lifespan.py --metadata-only "
            "(or --download) first"
        )
    data = json.loads(listing_path.read_text(encoding="utf-8"))
    sections: list[SectionInputs] = []
    for section in data["sections"]:
        files: dict[str, Path] = {}
        sizes: dict[str, int] = {}
        isizes: dict[str, int | None] = {}
        for item in section["files"]:
            role = item.get("role")
            if not role:
                continue
            files[role] = raw_dir / section["accession"] / item["name"]
            sizes[role] = int(item["size"])
            isizes[role] = item.get("isize")
        sections.append(
            SectionInputs(
                accession=section["accession"],
                title=section["title"],
                label=section["label"],
                pair_id=section["pair_id"],
                files=files,
                sizes=sizes,
                isizes=isizes,
            )
        )
    return sections


def select_sections(
    sections: Sequence[SectionInputs], wanted: Sequence[str]
) -> list[SectionInputs]:
    """Return the sections named by accession, label or pair id (all if empty).

    Raises:
        PrepareError: If a name matches nothing.
    """
    keys = [w.strip().upper() for w in wanted if w.strip()]
    if not keys:
        return list(sections)
    chosen: list[SectionInputs] = []
    for key in keys:
        hits = [
            s
            for s in sections
            if key in {s.accession.upper(), s.label.upper(), s.pair_id.upper()}
        ]
        if not hits:
            raise PrepareError(f"unknown section {key!r}")
        chosen.extend(h for h in hits if h not in chosen)
    return [s for s in sections if s in chosen]


def missing_inputs(inputs: SectionInputs) -> list[str]:
    """Return the roles whose download is absent or incomplete."""
    problems = []
    for role, path in sorted(inputs.files.items()):
        if not path.is_file():
            problems.append(f"{role}: not downloaded")
        elif path.stat().st_size != inputs.sizes[role]:
            problems.append(
                f"{role}: {path.stat().st_size} of {inputs.sizes[role]} bytes"
            )
    return problems


# --------------------------------------------------------------------------- #
# gzip / linking
# --------------------------------------------------------------------------- #


def read_gzip_isize(path: Path) -> int:
    """Return the gzip ``ISIZE`` trailer (decompressed size mod 2**32)."""
    with path.open("rb") as handle:
        handle.seek(-4, os.SEEK_END)
        return int.from_bytes(handle.read(4), "little")


def estimate_decompressed_size(compressed_size: int, isize: int) -> int:
    """Resolve a gzip ``ISIZE`` to a full-size estimate (see the fetch script)."""
    floor = int(compressed_size * 0.99)
    wraps = max(0, -(-(floor - isize) // GZIP_ISIZE_MODULUS))
    return isize + wraps * GZIP_ISIZE_MODULUS


def gunzip_file(source: Path, destination: Path) -> None:
    """Decompress a single-member gzip file atomically and verify it.

    ``pigz``/``gzip`` verify the CRC32 and length; the output size is also
    compared with the trailer.

    Args:
        source: ``.gz`` file (only read).
        destination: Output file.

    Raises:
        PrepareError: If decompression fails or the size disagrees.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    tool = shutil.which("pigz") or shutil.which("gzip")
    logger.info("  gunzip %s -> %s", source.name, destination.name)
    if tool is not None:
        with partial.open("wb") as out:
            result = subprocess.run(  # noqa: S603
                [tool, "-dc", str(source)],
                stdout=out,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode != 0:
            partial.unlink(missing_ok=True)
            raise PrepareError(
                f"{source.name}: decompression failed: "
                + result.stderr.decode(errors="replace")
            )
    else:
        with gzip.open(source, "rb") as src, partial.open("wb") as out:
            shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
    expected = read_gzip_isize(source)
    actual = partial.stat().st_size
    if actual % GZIP_ISIZE_MODULUS != expected:
        partial.unlink(missing_ok=True)
        raise PrepareError(
            f"{source.name}: {actual} bytes out, trailer says {expected} mod 2**32"
        )
    partial.replace(destination)


def link_file(source: Path, destination: Path, mode: str) -> str:
    """Place ``source`` at ``destination`` without modifying the source.

    Args:
        source: Existing file.
        destination: Link or copy to create (replaced if present).
        mode: ``hardlink`` (falls back to ``symlink`` across file systems),
            ``symlink`` or ``copy``.

    Returns:
        The method used.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            mode = "symlink"
    if mode == "symlink":
        destination.symlink_to(source.resolve())
        return "symlink"
    shutil.copy2(source, destination)
    return "copy"


# --------------------------------------------------------------------------- #
# Checks on the vendor tables
# --------------------------------------------------------------------------- #


def read_h5_features(path: Path) -> pd.DataFrame:
    """Return ``id``, ``name``, ``feature_type`` of a 10x feature-matrix h5."""
    import h5py

    with h5py.File(path, "r") as handle:
        features = handle["matrix/features"]
        columns = {key: features[key][...] for key in ("id", "name", "feature_type")}
    return pd.DataFrame(
        {
            key: [v.decode() if isinstance(v, bytes) else str(v) for v in values]
            for key, values in columns.items()
        }
    )


def read_h5_barcodes(path: Path) -> list[str]:
    """Return the barcodes (cell ids) of a 10x feature-matrix h5."""
    import h5py

    with h5py.File(path, "r") as handle:
        values = handle["matrix/barcodes"][...]
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def control_name_problems(features: pd.DataFrame) -> list[str]:
    """Check that MerXen's name rule classifies every feature as its h5 type.

    The deposited transcripts have no ``is_gene`` / ``codeword_category``
    columns, so MerXen falls back to the control-name rule.

    Args:
        features: :func:`read_h5_features` output.

    Returns:
        One message per misclassified feature (empty when all agree).
    """
    from merxen.control_features import matches_control_name_pattern

    problems = []
    for name, feature_type in zip(
        features["name"], features["feature_type"], strict=True
    ):
        is_control = matches_control_name_pattern(name, "XENIUM")
        if is_control == (feature_type == GENE_FEATURE_TYPE):
            problems.append(
                f"{name} ({feature_type}) classified as "
                + ("control" if is_control else "gene")
            )
    return problems


def parquet_column_problems(path: Path) -> list[str]:
    """Return missing required columns (footer read only)."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    required = REQUIRED_COLUMNS.get(path.name, ())
    problems = [
        f"{path.name}: no column {c!r}" for c in required if c not in schema.names
    ]
    if "cell_id" in schema.names:
        cell_type = str(schema.field("cell_id").type)
        if "binary" in cell_type:
            problems.append(f"{path.name}: cell_id is {cell_type}, not a string")
    return problems


# --------------------------------------------------------------------------- #
# Morphology projection
# --------------------------------------------------------------------------- #


def _ome_channel_names(ome_xml: str | None) -> list[str]:
    if not ome_xml:
        return []
    return re.findall(r"<Channel\b[^>]*\bName=\"([^\"]*)\"", ome_xml)


def tiff_pixel_size_um(tiff: Any) -> float | None:
    """Return the pixel size of a TIFF in micrometres, or None.

    Prefers the OME ``PhysicalSizeX`` (in µm, or nm/mm when so stated), then
    the first page's ``XResolution`` with its unit (tifffile writes unit 5 for
    micrometre, 4 millimetre, 3 centimetre, 2 inch).

    Args:
        tiff: An open ``tifffile.TiffFile``.
    """
    ome = tiff.ome_metadata
    if ome:
        match = re.search(r"\bPhysicalSizeX=\"([0-9.eE+-]+)\"", ome)
        if match:
            unit_match = re.search(r"\bPhysicalSizeXUnit=\"([^\"]+)\"", ome)
            unit = unit_match.group(1) if unit_match else "µm"
            factor = {"nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6}.get(unit, 1.0)
            return float(match.group(1)) * factor
    page = tiff.pages.first
    try:
        x_resolution = page.resolution[0]
        unit = int(page.resolutionunit)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
    um_per_unit = {5: 1.0, 4: 1e3, 3: 1e4, 2: 25400.0}.get(unit)
    if not x_resolution or um_per_unit is None:
        return None
    return um_per_unit / float(x_resolution)


def tiff_pixel_size_source(tiff: Any) -> str:
    """Name the tag :func:`tiff_pixel_size_um` reads its value from."""
    ome = tiff.ome_metadata
    if ome and re.search(r"\bPhysicalSizeX=\"", ome):
        return "OME PhysicalSizeX of the deposited morphology.ome.tif"
    return "TIFF XResolution/ResolutionUnit of the deposited morphology.ome.tif"


def _plane_indices(
    axes: str, shape: Sequence[int], channel_names: Sequence[str]
) -> tuple[list[tuple[int, ...]], str]:
    """Return the leading-axis index of every plane to project, and the source
    name of the projected channel (empty when the OME-XML names none)."""
    lead_axes, lead_shape = axes[:-2], tuple(shape[:-2])
    channel_name = channel_names[0] if len(channel_names) == 1 else ""
    choices: list[list[int]] = []
    for axis, size in zip(lead_axes, lead_shape, strict=True):
        if axis == "C" and size > 1:
            folded = [n.casefold() for n in channel_names]
            if "dapi" not in folded:
                raise PrepareError(f"{size} channels {channel_names}; none is DAPI")
            index = folded.index("dapi")
            channel_name = channel_names[index]
            choices.append([index])
        else:
            choices.append(list(range(size)))
    return [tuple(idx) for idx in itertools.product(*choices)], channel_name


def _bin2(image: np.ndarray) -> np.ndarray:
    """Return a 2x2 mean-binned copy (odd edges padded by replication)."""
    height, width = image.shape
    padded = np.pad(image, ((0, height % 2), (0, width % 2)), mode="edge")
    binned = padded.reshape(padded.shape[0] // 2, 2, padded.shape[1] // 2, 2).mean(
        axis=(1, 3)
    )
    result: np.ndarray = binned.astype(image.dtype)
    return result


def pyramid_level_count(height: int, width: int, tile_size: int) -> int:
    """Return the number of 2x-downsampled levels until one tile covers the image."""
    count = 0
    while max(height, width) > tile_size:
        height, width = -(-height // 2), -(-width // 2)
        count += 1
    return count


def _focus_score(block: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(block.astype(np.float32), cv2.CV_32F).var())


def project_zstack(
    source: Path,
    destination: Path,
    *,
    method: str = "mip",
    workers: int = 4,
    tile_size: int = 1024,
    pixel_size_um: float | None = None,
) -> ImageInfo:
    """Project a 3-D OME-TIFF to a 2-D, tiled, pyramidal OME-TIFF.

    The source is read one strip of ``tile_size`` rows at a time over all
    planes (``workers`` planes in parallel), so memory stays near
    ``workers x tile_size x width`` pixels plus a half-resolution copy for the
    pyramid.

    Args:
        source: Decompressed 3-D TIFF (only read).
        destination: Output ``.ome.tif`` (written atomically).
        method: ``mip`` (maximum over planes) or ``focus`` (per output tile,
            the plane with the largest variance of the Laplacian).
        workers: Planes read concurrently.
        tile_size: Output tile edge (multiple of 16).
        pixel_size_um: Override; otherwise read from the TIFF.

    Returns:
        The image description.

    Raises:
        PrepareError: On an unexpected layout or a missing pixel size.
    """
    import tifffile
    import zarr

    if method not in {"mip", "focus"}:
        raise PrepareError(f"unknown projection {method!r}")
    with tifffile.TiffFile(source) as tiff:
        series = tiff.series[0]
        axes, shape = series.axes, tuple(series.shape)
        if axes[-2:] != "YX":
            raise PrepareError(f"{source.name}: series axes {axes!r} do not end in YX")
        planes, source_channel_name = _plane_indices(
            axes, shape, _ome_channel_names(tiff.ome_metadata)
        )
        found = tiff_pixel_size_um(tiff)
        if pixel_size_um is None:
            if found is None:
                raise PrepareError(f"{source.name}: no pixel size; pass --pixel-size")
            pixel_size_um = found
            pixel_size_source = tiff_pixel_size_source(tiff)
        else:
            pixel_size_source = f"--pixel-size override (the TIFF gives {found})"
            if found is not None and abs(found - pixel_size_um) > 0.01 * pixel_size_um:
                logger.warning(
                    "  --pixel-size %.5f overrides the TIFF's %.5f",
                    pixel_size_um,
                    found,
                )
        array: Any = zarr.open(series.aszarr(level=0), mode="r")
        height, width = shape[-2:]
        dtype = np.dtype(series.dtype)
        n_tiles_x = -(-width // tile_size)
        half = np.zeros((-(-height // 2), -(-width // 2)), dtype=dtype)

        def read_plane(args: tuple[tuple[int, ...], int, int]) -> np.ndarray:
            plane, y0, y1 = args
            return np.asarray(array[(*plane, slice(y0, y1), slice(None))])

        def strips() -> Iterator[tuple[int, np.ndarray]]:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for y0 in range(0, height, tile_size):
                    y1 = min(y0 + tile_size, height)
                    out = np.zeros((y1 - y0, width), dtype=dtype)
                    best = np.full(n_tiles_x, -np.inf)
                    for batch_start in range(0, len(planes), max(1, workers)):
                        batch = planes[batch_start : batch_start + max(1, workers)]
                        for data in pool.map(read_plane, [(p, y0, y1) for p in batch]):
                            if method == "mip":
                                np.maximum(out, data, out=out)
                                continue
                            for tx in range(n_tiles_x):
                                x0, x1 = (
                                    tx * tile_size,
                                    min((tx + 1) * tile_size, width),
                                )
                                score = _focus_score(data[:, x0:x1])
                                if score > best[tx]:
                                    best[tx] = score
                                    out[:, x0:x1] = data[:, x0:x1]
                    half[y0 // 2 : y0 // 2 + -(-(y1 - y0) // 2)] = _bin2(out)
                    yield y0, out

        def tiles() -> Iterator[np.ndarray]:
            for _, strip in strips():
                for tx in range(n_tiles_x):
                    yield strip[:, tx * tile_size : (tx + 1) * tile_size]

        n_levels = pyramid_level_count(height, width, tile_size)
        metadata: dict[str, Any] = {
            "axes": "YX",
            "PhysicalSizeX": pixel_size_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size_um,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": [CHANNEL_NAME]},
        }
        common: dict[str, Any] = {
            "tile": (tile_size, tile_size),
            "compression": "zlib",
            "photometric": "minisblack",
            "resolution": (1.0 / pixel_size_um, 1.0 / pixel_size_um),
            "resolutionunit": "MICROMETER",
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        logger.info(
            "  project %s %s %s planes=%d -> %dx%d px (%s)",
            source.name,
            axes,
            shape,
            len(planes),
            height,
            width,
            method,
        )
        with tifffile.TiffWriter(partial, bigtiff=True, ome=True) as writer:
            writer.write(
                tiles(),
                shape=(height, width),
                dtype=dtype,
                subifds=n_levels,
                metadata=metadata,
                **common,
            )
            # The pyramid comes from the half-resolution copy that was filled
            # while the full-resolution tiles streamed out.
            level = half
            for _ in range(n_levels):
                writer.write(level, subfiletype=1, **common)
                level = _bin2(level)
    partial.replace(destination)
    return ImageInfo(
        height=int(height),
        width=int(width),
        dtype=dtype.name,
        pixel_size_um=float(pixel_size_um),
        channel_name=CHANNEL_NAME,
        source_channel_name=source_channel_name,
        pixel_size_source=pixel_size_source,
        method=method,
        source_axes=axes,
        source_shape=tuple(int(s) for s in shape),
        planes_used=len(planes),
    )


def read_image_info(path: Path) -> ImageInfo:
    """Describe an existing projected focus image."""
    import tifffile

    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages.first
        height, width = page.shape[-2:]
        pixel_size = tiff_pixel_size_um(tiff)
        if pixel_size is None:
            raise PrepareError(f"{path}: no pixel size")
        return ImageInfo(
            height=int(height),
            width=int(width),
            dtype=np.dtype(page.dtype).name,
            pixel_size_um=pixel_size,
            pixel_size_source=f"{FOCUS_FILE.as_posix()} written by an earlier run",
            method="unknown (made by an earlier run)",
        )


# --------------------------------------------------------------------------- #
# cells.zarr.zip
# --------------------------------------------------------------------------- #


def read_polygons(
    path: Path,
    cell_index: pd.Index,
    pixel_size_um: float,
    *,
    label_by: str = "cell",
) -> PolygonSet:
    """Read vendor boundaries as fixed-point pixel polygons.

    A polygon is a run of rows with the same ``cell_id`` (and ``label_id`` when
    present, which separates the nuclei of one cell). Its mask value follows
    the XOA >= 2.0 convention: the owning cell's row in ``cells.parquet`` plus
    one for cells (``label_by="cell"``; the reader maps cell labels to table
    rows by position), the polygon's own row plus one for nuclei
    (``label_by="polygon"``). The deposited ``label_id`` is compared with that
    value and the disagreements are counted.

    Args:
        path: ``cell_boundaries.parquet`` or ``nucleus_boundaries.parquet``.
        cell_index: ``cells.parquet`` cell ids, in table order.
        pixel_size_um: Micrometres per pixel.
        label_by: ``cell`` or ``polygon`` (see above).

    Returns:
        The polygons (unknown cell ids dropped and counted).

    Raises:
        PrepareError: On an unknown ``label_by``.
    """
    if label_by not in {"cell", "polygon"}:
        raise PrepareError(f"unknown label_by {label_by!r}")
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    columns = ["cell_id", "vertex_x", "vertex_y"] + (
        ["label_id"] if "label_id" in names else []
    )
    table = pq.read_table(path, columns=columns)
    cell_ids = pd.Series(table.column("cell_id").to_pandas()).map(
        lambda v: v.decode() if isinstance(v, bytes) else str(v)
    )
    codes = cell_index.get_indexer(cell_ids.to_numpy())
    vx = table.column("vertex_x").to_numpy().astype(np.float64)
    vy = table.column("vertex_y").to_numpy().astype(np.float64)
    keep = codes >= 0
    n_unknown = int((~keep).sum())
    codes, vx, vy = codes[keep], vx[keep], vy[keep]
    split = codes[1:] != codes[:-1]
    label_id: np.ndarray | None = None
    if "label_id" in columns:
        label_id = table.column("label_id").to_numpy()[keep]
        split |= label_id[1:] != label_id[:-1]
    starts = (
        np.flatnonzero(np.r_[True, split])
        if codes.size
        else np.zeros(0, dtype=np.int64)
    )
    owners = codes[starts].astype(np.int64)
    if label_by == "cell":
        labels = owners + 1
    else:
        labels = np.arange(1, starts.size + 1, dtype=np.int64)
    mismatches = (
        int((label_id[starts].astype(np.int64) != labels).sum())
        if label_id is not None
        else None
    )
    ends = np.r_[starts[1:], codes.size].astype(np.int64)
    scale = float(1 << FILL_SHIFT)
    # Pixel (r, c) covers [c, c+1) in the reader's frame; cv2 puts integer
    # coordinates at pixel centres, hence the half-pixel shift.
    fx = np.round((vx / pixel_size_um - 0.5) * scale).astype(np.int64)
    fy = np.round((vy / pixel_size_um - 0.5) * scale).astype(np.int64)
    return PolygonSet(
        owners=owners,
        labels=labels,
        starts=starts.astype(np.int64),
        ends=ends,
        fx=fx,
        fy=fy,
        n_unknown_vertices=n_unknown,
        n_label_id_mismatches=mismatches,
    )


def _polygons_by_chunk(
    polygons: PolygonSet, shape: tuple[int, int], chunk: int
) -> dict[tuple[int, int], list[int]]:
    """Map each label-mask chunk to the polygons overlapping it."""
    height, width = shape
    if polygons.starts.size == 0:
        return {}
    shift = FILL_SHIFT
    min_x = np.minimum.reduceat(polygons.fx, polygons.starts) >> shift
    max_x = (np.maximum.reduceat(polygons.fx, polygons.starts) >> shift) + 1
    min_y = np.minimum.reduceat(polygons.fy, polygons.starts) >> shift
    max_y = (np.maximum.reduceat(polygons.fy, polygons.starts) >> shift) + 1
    by_chunk: dict[tuple[int, int], list[int]] = {}
    for index in range(polygons.starts.size):
        x0, x1 = max(int(min_x[index]), 0), min(int(max_x[index]), width - 1)
        y0, y1 = max(int(min_y[index]), 0), min(int(max_y[index]), height - 1)
        if x0 > x1 or y0 > y1:
            continue
        for cy in range(y0 // chunk, y1 // chunk + 1):
            for cx in range(x0 // chunk, x1 // chunk + 1):
                by_chunk.setdefault((cy, cx), []).append(index)
    return by_chunk


def rasterize_polygons(
    target: Any,
    polygons: PolygonSet,
    *,
    n_labels: int,
    chunk: int,
) -> np.ndarray:
    """Fill polygons into a 2-D label array chunk by chunk.

    Args:
        target: Writable 2-D array (zarr) of shape (height, width).
        polygons: The polygons (later ones overwrite earlier ones).
        n_labels: Largest label value.
        chunk: Chunk edge; writes are chunk-aligned.

    Returns:
        Pixel count per label value (index 0 is background).
    """
    import cv2

    height, width = target.shape
    counts = np.zeros(n_labels + 1, dtype=np.int64)
    by_chunk = _polygons_by_chunk(polygons, (height, width), chunk)
    for cy in range(-(-height // chunk)):
        for cx in range(-(-width // chunk)):
            y0, x0 = cy * chunk, cx * chunk
            y1, x1 = min(y0 + chunk, height), min(x0 + chunk, width)
            canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.int32)
            for index in by_chunk.get((cy, cx), []):
                start, end = polygons.starts[index], polygons.ends[index]
                points = np.stack(
                    [
                        polygons.fx[start:end] - (x0 << FILL_SHIFT),
                        polygons.fy[start:end] - (y0 << FILL_SHIFT),
                    ],
                    axis=1,
                ).astype(np.int32)
                cv2.fillPoly(
                    canvas,
                    [points.reshape(-1, 1, 2)],
                    color=(int(polygons.labels[index]),),
                    lineType=cv2.LINE_8,
                    shift=FILL_SHIFT,
                )
            counts += np.bincount(canvas.ravel(), minlength=n_labels + 1)[
                : n_labels + 1
            ]
            if canvas.any():
                target[y0:y1, x0:x1] = canvas.astype(np.uint32)
    return counts


def stamp_missing_labels(
    target: Any,
    counts: np.ndarray,
    seeds_yx: np.ndarray,
    *,
    chunk: int,
    radius: int = 8,
) -> list[int]:
    """Give every label without pixels one pixel near its seed.

    The reader maps label values to cells positionally, so each cell needs at
    least one pixel. A seed pixel is taken only from background or from a
    label that keeps at least one other pixel.

    Args:
        target: The label array (modified in place).
        counts: Pixel count per label (updated).
        seeds_yx: ``(n_labels, 2)`` seed row/column per label (label i+1).
        chunk: Chunk edge (one read/write per touched chunk).
        radius: Search radius around the seed.

    Returns:
        Labels that could not be placed.
    """
    height, width = target.shape
    missing = [int(v) for v in np.flatnonzero(counts == 0) if v > 0]
    by_chunk: dict[tuple[int, int], list[int]] = {}
    for label in missing:
        row = int(np.clip(seeds_yx[label - 1, 0], 0, height - 1))
        col = int(np.clip(seeds_yx[label - 1, 1], 0, width - 1))
        by_chunk.setdefault((row // chunk, col // chunk), []).append(label)
    failed: list[int] = []
    for (cy, cx), labels in by_chunk.items():
        y0, x0 = cy * chunk, cx * chunk
        y1, x1 = min(y0 + chunk, height), min(x0 + chunk, width)
        block = np.asarray(target[y0:y1, x0:x1]).copy()
        for label in labels:
            row = int(np.clip(seeds_yx[label - 1, 0], y0, y1 - 1)) - y0
            col = int(np.clip(seeds_yx[label - 1, 1], x0, x1 - 1)) - x0
            placed = False
            for r in range(radius + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        yy, xx = row + dy, col + dx
                        if not (0 <= yy < block.shape[0] and 0 <= xx < block.shape[1]):
                            continue
                        old = int(block[yy, xx])
                        if old == 0 or counts[old] > 1:
                            block[yy, xx] = label
                            counts[old] -= 1
                            counts[label] += 1
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                failed.append(label)
        target[y0:y1, x0:x1] = block
    return failed


def cell_id_pairs(cell_ids: Sequence[str]) -> np.ndarray:
    """Encode XOA cell ids (``aaaakneh-1``) as the ``(prefix, suffix)`` uint32 pairs.

    Raises:
        PrepareError: If the ids are not XOA ids or do not round-trip.
    """
    from spatialdata_io.readers.xenium import (
        cell_id_str_from_prefix_suffix_uint32,
        prefix_suffix_uint32_from_cell_id_str,
    )

    try:
        prefix, suffix = prefix_suffix_uint32_from_cell_id_str(np.asarray(cell_ids))
    except (KeyError, ValueError) as error:
        raise PrepareError(f"cell ids are not XOA ids: {error}") from error
    decoded = cell_id_str_from_prefix_suffix_uint32(prefix, suffix)
    if list(decoded) != list(cell_ids):
        raise PrepareError("cell ids do not round-trip through the XOA encoding")
    return np.stack([prefix, np.asarray(suffix, dtype=np.uint32)], axis=1).astype(
        np.uint32
    )


def build_cells_zarr(
    destination: Path,
    *,
    cells: pd.DataFrame,
    cell_boundaries: Path,
    nucleus_boundaries: Path,
    image: ImageInfo,
    chunk: int = 4096,
) -> dict[str, Any]:
    """Write ``cells.zarr.zip`` with label masks rasterised from the polygons.

    Layout (zarr v2, read by spatialdata-io's XOA >= 2.0 branch), following
    the vendor convention: ``masks/1`` cells (uint32, value = cell row + 1),
    ``masks/0`` nuclei (value = nucleus-polygon row + 1), background 0;
    ``cell_id`` ``(n, 2)`` uint32; ``polygon_sets/1/cell_index`` = ``0..n-1``;
    ``polygon_sets/0/cell_index`` = owning cell row per nucleus polygon; and
    ``cell_summary`` with ``column_names``.

    Args:
        destination: Output zip (written atomically).
        cells: ``cells.parquet`` table (``cell_id``, centroids, areas).
        cell_boundaries: Cell polygon parquet.
        nucleus_boundaries: Nucleus polygon parquet.
        image: The morphology image (shape and pixel size).
        chunk: Mask chunk edge.

    Returns:
        Statistics for the provenance file.

    Raises:
        PrepareError: If a cell ends up without pixels.
    """
    import zarr

    cell_ids = [str(c) for c in cells["cell_id"]]
    index = pd.Index(cell_ids)
    if not index.is_unique:
        raise PrepareError("cells.parquet cell ids are not unique")
    n_cells = len(cell_ids)
    shape = (image.height, image.width)
    work = destination.parent / WORK_DIR / "cells.zarr"
    if work.exists():
        shutil.rmtree(work)
    work.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(store=str(work), mode="w", zarr_format=2)
    root.attrs.update(
        {
            "major_version": 5,
            "minor_version": 0,
            "merxen_synthesized": True,
            "merxen_note": (
                "rasterised from the deposited vendor polygons by "
                "prepare_dg_lifespan.py"
            ),
        }
    )
    masks = root.create_group("masks")
    stats: dict[str, Any] = {"n_cells": n_cells, "mask_shape": list(shape)}
    nucleus_polygons = read_polygons(
        nucleus_boundaries, index, image.pixel_size_um, label_by="polygon"
    )
    cell_polygons = read_polygons(
        cell_boundaries, index, image.pixel_size_um, label_by="cell"
    )
    for mask_index, polygons in ((0, nucleus_polygons), (1, cell_polygons)):
        target = masks.create_array(
            str(mask_index),
            shape=shape,
            chunks=(chunk, chunk),
            dtype="uint32",
            fill_value=0,
        )
        logger.info(
            "  rasterise masks/%d: %d polygons", mask_index, polygons.starts.size
        )
        n_labels = n_cells if mask_index == 1 else int(polygons.starts.size)
        counts = rasterize_polygons(target, polygons, n_labels=n_labels, chunk=chunk)
        key = "nucleus" if mask_index == 0 else "cell"
        stats[f"{key}_polygons"] = int(polygons.starts.size)
        stats[f"{key}_unknown_vertices"] = int(polygons.n_unknown_vertices)
        stats[f"{key}_label_id_mismatches"] = polygons.n_label_id_mismatches
        stats[f"{key}_labels_without_pixels"] = int((counts[1:] == 0).sum())
        if mask_index == 1:
            seeds = np.stack(
                [
                    cells["y_centroid"].to_numpy() / image.pixel_size_um,
                    cells["x_centroid"].to_numpy() / image.pixel_size_um,
                ],
                axis=1,
            ).astype(np.int64)
            failed = stamp_missing_labels(target, counts, seeds, chunk=chunk)
            stats["cell_labels_stamped"] = stats["cell_labels_without_pixels"] - len(
                failed
            )
            if failed:
                raise PrepareError(
                    f"{len(failed)} cells have no mask pixel (e.g. label {failed[0]})"
                )
    root.create_array("cell_id", data=cell_id_pairs(cell_ids))
    polygon_sets = root.create_group("polygon_sets")
    polygon_sets.create_group("1").create_array(
        "cell_index", data=np.arange(n_cells, dtype=np.uint32)
    )
    polygon_sets.create_group("0").create_array(
        "cell_index", data=nucleus_polygons.owners.astype(np.uint32)
    )
    nucleus_count = np.bincount(nucleus_polygons.owners, minlength=n_cells)[:n_cells]
    summary = np.full((n_cells, len(CELL_SUMMARY_COLUMNS)), np.nan, dtype=np.float64)
    summary[:, 0] = cells["x_centroid"].to_numpy()
    summary[:, 1] = cells["y_centroid"].to_numpy()
    if "cell_area" in cells:
        summary[:, 2] = cells["cell_area"].to_numpy()
    if "nucleus_area" in cells:
        summary[:, 5] = cells["nucleus_area"].to_numpy()
    summary[:, 7] = nucleus_count
    summary_array = root.create_array("cell_summary", data=summary)
    summary_array.attrs["column_names"] = list(CELL_SUMMARY_COLUMNS)
    partial = destination.with_name(destination.name + ".partial")
    with zipfile.ZipFile(
        partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path in sorted(work.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(work).as_posix())
    partial.replace(destination)
    shutil.rmtree(work)
    return stats


# --------------------------------------------------------------------------- #
# Synthesised metadata
# --------------------------------------------------------------------------- #


def synthesize_experiment(
    *, image: ImageInfo, features: pd.DataFrame, n_cells: int, inputs: SectionInputs
) -> dict[str, Any]:
    """Return a minimal ``experiment.xenium`` for the reader and MerXen.

    Only ``pixel_size`` and ``analysis_sw_version`` are read by the reader;
    the panel count feeds MerXen's Atera guard (5K passes). Every value is
    sourced, and ``merxen_value_sources`` says from where. The split of the
    genes into predesigned and custom targets is not deposited (the paper's
    second batch added 15 custom genes, yet both slides' h5 files carry the
    same 5,011 genes), so all Gene Expression features are counted as
    ``panel_num_targets_predesigned`` and ``panel_num_targets_custom`` is
    left out (MerXen's guard reads a missing count as 0).
    """
    n_genes = int((features["feature_type"] == GENE_FEATURE_TYPE).sum())
    return {
        "analysis_sw_version": SYNTHETIC_ANALYSIS_SW_VERSION,
        "pixel_size": image.pixel_size_um,
        "panel_name": PANEL_NAME,
        "panel_organism": "Human",
        "panel_num_targets_predesigned": n_genes,
        "num_cells": n_cells,
        "preservation_method": "fresh frozen",
        "region_name": inputs.pair_id,
        "merxen_synthesized": True,
        "merxen_provenance": PROVENANCE_FILE,
        "merxen_value_sources": {
            "analysis_sw_version": (
                "paper methods 4.2.1: 'a prototype version of XOA 3.0.0.0' "
                "(bioRxiv 10.64898/2025.12.12.694021); written in the XOA "
                "'xenium-X.Y.Z.W' form so spatialdata-io takes its XOA 2/3 branch"
            ),
            "pixel_size": image.pixel_size_source,
            "panel_name": (
                "GEO GSE317381 / paper: pre-commercial prototype of the Xenium "
                "Prime 5K Human Pan Tissue & Pathways panel"
            ),
            "panel_organism": "GEO GSE317381 (Homo sapiens)",
            "panel_num_targets_predesigned": (
                f"count of 'Gene Expression' features in {inputs.accession}'s "
                "cell_feature_matrix.h5 (predesigned/custom split unknown)"
            ),
            "num_cells": f"rows of {inputs.accession}'s cells.parquet",
            "preservation_method": "paper methods: fresh frozen 10 um sections",
            "region_name": "MerXen pair id (no vendor region name deposited)",
        },
    }


def synthesize_gene_panel(features: pd.DataFrame) -> dict[str, Any]:
    """Return a ``gene_panel.json`` with the gene and negative-probe targets.

    The descriptors are those MerXen's panel reader maps; probe coverage is
    unknown, so ``info`` is left out.
    """
    descriptors = {
        GENE_FEATURE_TYPE: "gene",
        NEGATIVE_PROBE_FEATURE_TYPE: "negative_control",
    }
    targets = [
        {"type": {"descriptor": descriptors[t], "data": {"id": i, "name": n}}}
        for i, n, t in zip(
            features["id"], features["name"], features["feature_type"], strict=True
        )
        if t in descriptors
    ]
    return {
        "metadata": {
            "panel_name": PANEL_NAME,
            "merxen_synthesized": True,
            "merxen_note": (
                "built from cell_feature_matrix.h5 features; no probe coverage"
            ),
        },
        "payload": {"targets": targets},
    }


# --------------------------------------------------------------------------- #
# Planning and running
# --------------------------------------------------------------------------- #


def _size_estimate(inputs: SectionInputs, role: str) -> int | None:
    path = inputs.files.get(role)
    if (
        path is not None
        and path.is_file()
        and path.stat().st_size == inputs.sizes[role]
    ):
        return estimate_decompressed_size(inputs.sizes[role], read_gzip_isize(path))
    isize = inputs.isizes.get(role)
    if isize is None:
        return None
    return estimate_decompressed_size(inputs.sizes[role], int(isize))


def _image_shape_estimate(inputs: SectionInputs) -> tuple[int, int] | None:
    """Estimate the image shape from the vendor cell extent (dry run only)."""
    import pyarrow.parquet as pq

    path = inputs.files.get("cells")
    if path is None or not path.is_file():
        return None
    with gzip.open(path, "rb") as handle:
        table = pq.read_table(handle, columns=["x_centroid", "y_centroid"])
    width = (
        float(np.nanmax(table.column("x_centroid").to_numpy())) / NOMINAL_PIXEL_SIZE_UM
    )
    height = (
        float(np.nanmax(table.column("y_centroid").to_numpy())) / NOMINAL_PIXEL_SIZE_UM
    )
    return int(height * 1.01), int(width * 1.01)


def plan_section(
    inputs: SectionInputs, out_dir: Path, options: Options
) -> list[PlannedAction]:
    """Return the actions :func:`prepare_section` would take."""
    actions = [
        PlannedAction(
            f"{options.link_mode} {inputs.files['cell_feature_matrix'].name}",
            out_dir / "cell_feature_matrix.h5",
            0 if options.link_mode != "copy" else inputs.sizes["cell_feature_matrix"],
            (out_dir / "cell_feature_matrix.h5").exists(),
        )
    ]
    for role, name in GUNZIP_OUTPUTS.items():
        actions.append(
            PlannedAction(
                f"gunzip {inputs.files[role].name}",
                out_dir / name,
                _size_estimate(inputs, role),
                (out_dir / name).is_file(),
            )
        )
    zstack = out_dir / (
        "morphology.ome.tif" if options.keep_zstack else f"{WORK_DIR}/zstack.ome.tif"
    )
    focus = out_dir / FOCUS_FILE
    focus_done = focus.is_file()
    actions.append(
        PlannedAction(
            "gunzip z-stack "
            + (
                "(kept)"
                if options.keep_zstack
                else "(transient, deleted after projection)"
            ),
            zstack,
            _size_estimate(inputs, "morphology"),
            focus_done and (not options.keep_zstack or zstack.is_file()),
            is_transient=not options.keep_zstack,
        )
    )
    shape = _image_shape_estimate(inputs)
    pixels = shape[0] * shape[1] if shape else None
    actions.append(
        PlannedAction(
            f"{options.projection} projection -> 2-D DAPI OME-TIFF "
            "(zlib; size shown uncompressed + pyramid)",
            focus,
            int(pixels * 2 * 4 / 3) if pixels else None,
            focus_done,
        )
    )
    actions.append(
        PlannedAction(
            "rasterise vendor polygons -> cells.zarr.zip "
            "(size shown is the uncompressed bound)",
            out_dir / "cells.zarr.zip",
            pixels * 8 if pixels else None,
            (out_dir / "cells.zarr.zip").is_file(),
        )
    )
    for name in ("experiment.xenium", "gene_panel.json", PROVENANCE_FILE):
        actions.append(
            PlannedAction(
                f"synthesise {name}", out_dir / name, 0, (out_dir / name).is_file()
            )
        )
    return actions


def space_needed(actions: Sequence[PlannedAction]) -> tuple[int, int, int]:
    """Sum a plan's new bytes.

    Returns:
        ``(persistent, transient_peak, n_unknown)``: bytes of the outputs not
        yet done that stay, the largest transient output, and the number of
        outputs whose size cannot be estimated.
    """
    persistent, transient, unknown = 0, 0, 0
    for action in actions:
        if action.is_done:
            continue
        if action.estimated_bytes is None:
            unknown += 1
        elif action.is_transient:
            transient = max(transient, action.estimated_bytes)
        else:
            persistent += action.estimated_bytes
    return persistent, transient, unknown


def free_bytes(path: Path) -> int:
    """Return the free bytes on the file system that holds (or would hold) path."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def check_free_space(inputs: SectionInputs, out_dir: Path, options: Options) -> None:
    """Refuse a section when its estimated new bytes do not fit.

    Raises:
        PrepareError: If the free space is below the persistent estimate plus
            the transient peak plus ``options.margin_bytes``.
    """
    actions = plan_section(inputs, out_dir, options)
    if options.force:
        # --force rewrites existing outputs; count them as new (conservative).
        actions = [replace(action, is_done=False) for action in actions]
    persistent, transient, _ = space_needed(actions)
    needed = persistent + transient + options.margin_bytes
    free = free_bytes(out_dir)
    if free < needed:
        raise PrepareError(
            f"{inputs.pair_id}: not enough space: {_human(free)} free < "
            f"{_human(persistent)} new + {_human(transient)} transient + "
            f"{_human(options.margin_bytes)} margin"
        )


def prepare_section(
    inputs: SectionInputs, out_dir: Path, options: Options
) -> dict[str, Any]:
    """Build the MerXen-ready Xenium folder of one section.

    Args:
        inputs: Raw files (must be complete).
        out_dir: ``.../xenium/<pair_id>``.
        options: Prepare options.

    Returns:
        The provenance record that was written.

    Raises:
        PrepareError: If an input is missing or a check fails.
    """
    problems = missing_inputs(inputs)
    if problems:
        raise PrepareError(
            f"{inputs.pair_id}: incomplete download: {'; '.join(problems)}"
        )
    check_free_space(inputs, out_dir, options)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps: list[str] = []
    # A re-run keeps the record of outputs it does not redo.
    previous: dict[str, Any] = {}
    if (out_dir / PROVENANCE_FILE).is_file() and not options.force:
        previous = json.loads((out_dir / PROVENANCE_FILE).read_text(encoding="utf-8"))

    def is_done(path: Path) -> bool:
        return path.is_file() and not options.force

    h5 = out_dir / "cell_feature_matrix.h5"
    if not h5.exists() or options.force:
        steps.append(
            "cell_feature_matrix.h5: "
            + link_file(inputs.files["cell_feature_matrix"], h5, options.link_mode)
        )
    for role, name in GUNZIP_OUTPUTS.items():
        if not is_done(out_dir / name):
            gunzip_file(inputs.files[role], out_dir / name)
            steps.append(f"{name}: gunzip")

    checks: list[str] = []
    for name in GUNZIP_OUTPUTS.values():
        checks += parquet_column_problems(out_dir / name)
    features = read_h5_features(h5)
    checks += control_name_problems(features)
    cells = pd.read_parquet(out_dir / "cells.parquet")
    cells["cell_id"] = cells["cell_id"].map(
        lambda v: v.decode() if isinstance(v, bytes) else str(v)
    )
    if read_h5_barcodes(h5) != list(cells["cell_id"]):
        checks.append("cells.parquet cell_id order differs from the h5 barcodes")
    if checks:
        raise PrepareError(
            f"{inputs.pair_id}: checks failed: " + "; ".join(checks[:20])
        )

    focus = out_dir / FOCUS_FILE
    image_record: dict[str, Any] | None = None
    if is_done(focus):
        image = read_image_info(focus)
        earlier = previous.get("image")
        if isinstance(earlier, dict) and (
            earlier.get("height"),
            earlier.get("width"),
        ) == (image.height, image.width):
            image_record = earlier
            image.pixel_size_source = str(
                earlier.get("pixel_size_source") or image.pixel_size_source
            )
    else:
        if options.keep_zstack:
            zstack = out_dir / "morphology.ome.tif"
        else:
            zstack = out_dir / WORK_DIR / "zstack.ome.tif"
        if not zstack.is_file():
            gunzip_file(inputs.files["morphology"], zstack)
        image = project_zstack(
            zstack,
            focus,
            method=options.projection,
            workers=options.workers,
            tile_size=options.tile_size,
            pixel_size_um=options.pixel_size_um,
        )
        if not options.keep_zstack:
            zstack.unlink()
        steps.append(
            f"{FOCUS_FILE.as_posix()}: {options.projection} of the deposited z-stack"
        )

    zarr_path = out_dir / "cells.zarr.zip"
    zarr_stats: dict[str, Any] = dict(previous.get("cells_zarr") or {})
    if not is_done(zarr_path):
        zarr_stats = build_cells_zarr(
            zarr_path,
            cells=cells,
            cell_boundaries=out_dir / "cell_boundaries.parquet",
            nucleus_boundaries=out_dir / "nucleus_boundaries.parquet",
            image=image,
            chunk=options.mask_chunk,
        )
        steps.append("cells.zarr.zip: rasterised from vendor polygons")

    experiment = synthesize_experiment(
        image=image, features=features, n_cells=len(cells), inputs=inputs
    )
    (out_dir / "experiment.xenium").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "gene_panel.json").write_text(
        json.dumps(synthesize_gene_panel(features), indent=2) + "\n", encoding="utf-8"
    )
    work = out_dir / WORK_DIR
    if work.is_dir() and not any(work.iterdir()):
        work.rmdir()
    provenance = {
        "dataset": DATASET_KEY,
        "geo_accession": inputs.accession,
        "geo_title": inputs.title,
        "label": inputs.label,
        "pair_id": inputs.pair_id,
        "inputs": {
            role: {"file": p.name, "bytes": inputs.sizes[role]}
            for role, p in inputs.files.items()
        },
        "image": image_record or asdict(image),
        "cells_zarr": zarr_stats,
        "steps_this_run": steps,
        "steps_history": [*list(previous.get("steps_history") or []), steps],
        "notes": [
            "experiment.xenium is synthesised: the deposit has none. "
            f"analysis_sw_version {SYNTHETIC_ANALYSIS_SW_VERSION} stands for the "
            "prototype XOA 3.0.0.0 of the paper and puts spatialdata-io on its "
            "XOA 2/3 branch (channel named DAPI).",
            "morphology_focus_0000.ome.tif is a MerXen projection of the deposited "
            "3-D DAPI z-stack, not the vendor focus image; the boundary/interior "
            "stains were not deposited, so the folder is DAPI-only "
            "(xenium_channels=DAPI).",
            "cells.zarr.zip is rasterised from the deposited vendor polygons "
            "(corner pixel convention; cell label = cells.parquet row + 1, "
            "nucleus label = nucleus-polygon row + 1, as in XOA >= 2.0); cells "
            "without a pixel are stamped at their centroid; cell_summary z_level "
            "and nucleus centroids are NaN.",
            "gene_panel.json is synthesised from the h5 features (genes and "
            "negative control probes); probe coverage is unknown.",
            "transcripts.parquet has no is_gene/codeword_category; every h5 "
            "feature was checked to be classified correctly by MerXen's "
            "control-name rule.",
        ],
    }
    (out_dir / PROVENANCE_FILE).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def _human(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local_config.add_location_argument(
        parser,
        "--public-root",
        "MERXEN_XENIUM_PUBLIC_ROOT",
        "Root of the public validation datasets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the plan only.")
    parser.add_argument(
        "--sections", default="", help="Comma-separated GSMs, labels or pair ids."
    )
    parser.add_argument("--projection", choices=("mip", "focus"), default="mip")
    parser.add_argument(
        "--workers", type=int, default=4, help="Z-planes read in parallel."
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--mask-chunk", type=int, default=4096)
    parser.add_argument("--keep-zstack", action="store_true")
    parser.add_argument(
        "--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink"
    )
    parser.add_argument(
        "--pixel-size", type=float, default=None, help="Override in µm/px."
    )
    parser.add_argument("--force", action="store_true", help="Redo existing outputs.")
    parser.add_argument(
        "--margin-gb",
        type=float,
        default=20.0,
        help="Free space (GiB) to keep beyond a section's estimated new bytes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prepare step.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)
    if args.tile_size % 16:
        raise PrepareError("--tile-size must be a multiple of 16")
    root = dataset_dir(args.public_root)
    options = Options(
        projection=args.projection,
        workers=args.workers,
        tile_size=args.tile_size,
        mask_chunk=args.mask_chunk,
        keep_zstack=args.keep_zstack,
        link_mode=args.link_mode,
        pixel_size_um=args.pixel_size,
        force=args.force,
        margin_bytes=int(args.margin_gb * 1024**3),
    )
    sections = select_sections(load_sections(root / "raw"), args.sections.split(","))
    failures = 0
    for inputs in sections:
        out_dir = root / "xenium" / inputs.pair_id
        logger.info("\n%s (%s) -> %s", inputs.pair_id, inputs.accession, out_dir)
        if args.dry_run:
            for problem in missing_inputs(inputs):
                logger.info("  input %s", problem)
            actions = plan_section(inputs, out_dir, options)
            for action in actions:
                logger.info(
                    "  [%s] %-11s %s\n         -> %s",
                    "done" if action.is_done else "todo",
                    _human(action.estimated_bytes),
                    action.description,
                    action.output.relative_to(root),
                )
            total, transient, unknown = space_needed(actions)
            logger.info(
                "  new persistent bytes (upper bound): %s%s; transient peak: %s; "
                "free: %s (margin %s)",
                _human(total),
                f" + {unknown} output(s) not estimable yet" if unknown else "",
                _human(transient),
                _human(free_bytes(out_dir)),
                _human(options.margin_bytes),
            )
            continue
        try:
            prepare_section(inputs, out_dir, options)
            logger.info("  ready: %s", out_dir)
        except PrepareError as error:
            failures += 1
            logger.error("  %s", error)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PrepareError as error:
        logger.error("%s", error)
        sys.exit(1)
