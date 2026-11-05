"""Pure readers for 10x Atera output bundles (one extracted ROI directory).

An Atera ROI holds zipped zarr v2 stores (``transcripts.zarr.zip``,
``cells.zarr.zip``, ``cell_feature_matrix.zarr.zip``), a multi-file OME-TIFF
``morphology_2d/`` and JSON/CSV sidecars. No released reader opens it, so this
module reads each part into the element MerXen's source store needs; the
builder (``merxen.io.builders.atera``) assembles them. The format and every
decision below are documented in ``docs/plans/atera-support-plan.md`` (§2, §3).

Conventions the readers rely on, verified on the public dataset:

- Every zip is opened with ``zarr.storage.ZipStore(path, mode="r")``, one store
  per read task: ``ZipStore`` serialises reads behind a lock.
- Transcript tiles are sorted by gene; ``gene_offset`` holds end-exclusive
  ``[start, end)`` row ranges, so a row's gene comes from ``gene_offset``
  without decoding ``codeword_identity``.
- ``grid_number_objects`` follows the ``grid_keys`` attribute order, which
  fixes the vendor global ordinal and hence ``transcript_id`` (ordinal + 1).
- Cell mask label = cell row + 1; nucleus mask label = nucleus-polygon row + 1;
  the owning cell of nucleus ``k`` is ``polygon_sets/0/cell_index[k - 1]``.
- Pixel ``i`` covers ``[i * ps, (i + 1) * ps)`` microns (corner convention), so
  contour coordinates with pixel centres at integers map to microns as
  ``(c + 0.5) * ps``.
- Identifying manifest fields (run, slide, region, serial, UUIDs, customer)
  never leave this module unsanitised: see :func:`sanitize_experiment_spatial`.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
import threading
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

import anndata as ad
import dask
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tifffile
import xarray as xr
import zarr
from scipy import sparse
from spatialdata.models.pyramids_utils import dask_arrays_to_datatree
from spatialdata.transformations import Identity, set_transformation

from merxen.io.spatialdata_schema import INSTANCE_ID_COLUMN, SOURCE_CELL_ID_COLUMN
from merxen.io.tenx_cell_ids import (
    UNASSIGNED_CELL_ID_PART,
    cell_id_str_from_prefix_suffix_uint32,
    pack_cell_ids,
)
from merxen.segmentation.mask_geometry import MpStartMethod, masks_to_labeled_polygons

logger = logging.getLogger(__name__)

ATERA_SW_PREFIX: Final = "atera-"
SUPPORTED_EXPERIMENT_MAJOR_VERSIONS: Final = frozenset({1})
SUPPORTED_RNA_MAJOR_VERSIONS: Final = frozenset({6})
SUPPORTED_CELLS_MAJOR_VERSIONS: Final = frozenset({6})
RNA_DATASET_NAME: Final = "RnaDataset"
CELLS_DATASET_NAME: Final = "CellSegmentationDataset"
COORDINATE_SPACE: Final = "refined-final_global_micron"
MICRON_UNITS: Final = frozenset({"micron", "microns", "um", "µm"})
TRANSFORM_RTOL: Final = 1e-6
PIXEL_SIZE_RTOL: Final = 1e-6

GENE_EXPRESSION_TYPE: Final = "Gene Expression"
GENOMIC_CONTROL_TYPE: Final = "Genomic Control"
UNASSIGNED_CODEWORD_TYPE: Final = "Unassigned Codeword"
DEPRECATED_CODEWORD_TYPE: Final = "Deprecated Codeword"
#: Per-cell control counts the table derives from the vendor X columns.
DERIVED_CONTROL_COUNT_COLUMNS: Final = {
    "vendor_genomic_control_counts": GENOMIC_CONTROL_TYPE,
    "vendor_unassigned_codeword_counts": UNASSIGNED_CODEWORD_TYPE,
    "vendor_deprecated_codeword_counts": DEPRECATED_CODEWORD_TYPE,
}
#: Vendor ``obs`` columns copied with a ``vendor_`` prefix when present.
VENDOR_OBS_COLUMNS: Final = (
    "leiden_res_1.0",
    "azimuth_broad",
    "azimuth_coarse",
    "azimuth_fine",
    "azimuth_full_hierarchy",
    "azimuth_score",
    "filtered",
    "is_sketch",
    "total_counts",
    "transcript_counts",
    "control_probe_counts",
    "control_codeword_counts",
)

#: Column order of every transcripts partition.
TRANSCRIPT_COLUMNS: Final = (
    "x",
    "y",
    "z",
    "feature_name",
    "is_gene",
    "qv",
    "overlaps_nucleus",
    "original_assignment",
    "transcript_id",
    "source_tile",
)
CODEWORD_INDEX_COLUMN: Final = "codeword_index"
#: QV histograms count quarter-QV steps from 0 to 40 (vendor QV is 8-40 in 0.25 steps).
QV_HISTOGRAM_STEP: Final = 0.25
QV_HISTOGRAM_BINS: Final = 161
DENSITY_BIN_UM: Final = 100.0

SpatialOrder = Literal["row_range", "tile_blocks"]

#: ``experiment.spatial`` fields that identify a run, slide, ROI or instrument.
IDENTIFYING_EXPERIMENT_FIELDS: Final = frozenset(
    {
        "run_name",
        "run_start_time",
        "slide_name",
        "slide_id",
        "region_name",
        "instrument_sn",
        "analysis_uuid",
        "experiment_uuid",
        "cassette_uuid",
        "roi_uuid",
        "well_uuid",
        "calibration_uuid",
        "dataset_uuid",
    }
)
_EXPERIMENT_TEXT_FIELDS: Final = frozenset(
    {
        "analysis_sw_version",
        "instrument_sw_version",
        "chemistry_version",
        "segmentation_stain",
        "sample_type",
    }
)
_EXPERIMENT_NUMBER_FIELDS: Final = frozenset(
    {"major_version", "minor_version", "pixel_size", "z_step_size", "total_num_targets"}
)
# Summary metrics: numeric values under these name prefixes are kept.
_EXPERIMENT_METRIC_RE: Final = re.compile(
    r"^(num_|transcripts_per|genes_per|nuclear_transcripts|region_area|total_cell_area"
    r"|thickness_|non_zero_|fraction_|segmented_|imported_)"
)
_PANEL_FIELDS: Final = frozenset(
    {
        "panel_name",
        "panel_design_id",
        "panel_num_targets",
        "panel_organism",
        "panel_tissue_type",
    }
)
_PATH_GROUPS: Final = ("images", "explorer_files")
#: Sidecar name patterns; the first group of each is the neutral name.
_NEUTRAL_NAME_RES: Final = (
    re.compile(r"^(metrics_summary)(?:_.*)?(\.csv)$"),
    re.compile(r"^(analysis_summary)(?:_.*)?(\.html)$"),
    re.compile(r"^(?:.*_)?(he_imagealignment)(\.csv)$"),
    re.compile(r"^(?:.*_)?(keypoints)(\.csv)$"),
    re.compile(r"^(?:.*_)?(he_image)(\.ome\.tif)$"),
    re.compile(
        r"^(?:.*_)?(transcripts|binned_transcripts|cells|cell_feature_matrix"
        r"|csc_cell_feature_matrix)(\.zarr\.zip)$"
    ),
)
_SIDECAR_GLOBS: Final = (
    "panel_config.json",
    "metrics_summary*.csv",
    "morphology_fov_locations.json",
    "diffexp_genes/*.csv",
    "*he_imagealignment.csv",
    "*keypoints.csv",
)
#: ``metrics_summary`` columns kept only if not identifying.
_IDENTIFYING_METRIC_COLUMN_RE: Final = re.compile(
    r"^(run_name|instrument_sn|slide_name|region_name|slide_id|.*uuid.*)$"
)


class AteraFormatError(ValueError):
    """Raised when an Atera bundle does not match the supported format."""


# --------------------------------------------------------------------------- #
# Bundle resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AteraBundle:
    """Resolved files and header facts of one Atera ROI directory.

    Attributes:
        root: The ROI directory.
        experiment: Parsed ``experiment.spatial`` (or the metadata override).
            It holds identifying fields; publish only
            :func:`sanitize_experiment_spatial` of it.
        pixel_size_um: Microns per level-0 morphology pixel.
        analysis_sw_version: Onboard analysis version, ``atera-...``.
        transcripts_zip: ``transcripts.zarr.zip``.
        cells_zip: ``cells.zarr.zip``.
        cell_feature_matrix_zip: ``cell_feature_matrix.zarr.zip``.
        morphology_2d_files: One file per channel, ordered by OME ``FirstC``.
        morphology_channel_names: OME channel names in the same order.
        morphology_shape_yx: Level-0 ``(y, x)`` shape of the morphology image.
        morphology_3d_file: The 3D DAPI stack, if present (not ingested).
        panel_config_file: ``panel_config.json``, if present.
        analysis_summary_file: The vendor HTML report, if present; it embeds
            identifiers, so it is copied only on request.
        sidecars: Other small files worth keeping (sanitised before copying).
    """

    root: Path
    experiment: Mapping[str, Any]
    pixel_size_um: float
    analysis_sw_version: str
    transcripts_zip: Path
    cells_zip: Path
    cell_feature_matrix_zip: Path
    morphology_2d_files: tuple[Path, ...]
    morphology_channel_names: tuple[str, ...]
    morphology_shape_yx: tuple[int, int]
    morphology_3d_file: Path | None
    panel_config_file: Path | None
    analysis_summary_file: Path | None
    sidecars: tuple[Path, ...]


@contextmanager
def open_zip_group(path: Path) -> Iterator[Any]:
    """Open the root group of a zipped zarr store read-only.

    A plain string path fails for these zips; the store must be a
    ``ZipStore``. Each call opens its own store, so concurrent tasks do not
    share the store's lock.

    Args:
        path: Zipped zarr store.

    Yields:
        The root group; the store is closed on exit.
    """
    store = zarr.storage.ZipStore(path, mode="r")
    try:
        yield zarr.open_group(store, mode="r")
    finally:
        store.close()


def resolve_atera_bundle(
    input_path: Path,
    *,
    spec_override: Path | None = None,
    transcripts_path: Path | None = None,
    cells_path: Path | None = None,
    cfm_path: Path | None = None,
    species: str | None = None,
) -> AteraBundle:
    """Resolve and check the files of one extracted Atera ROI directory.

    Paths come from ``experiment.spatial`` (``images.*``, ``explorer_files.*``)
    relative to ``input_path``, then from the plain vendor names, then from the
    prefixed names of the public companion downloads (``*_transcripts.zarr.zip``).

    Args:
        input_path: The ROI directory.
        spec_override: A replacement ``experiment.spatial`` for metadata only;
            paths still resolve against ``input_path``.
        transcripts_path: Explicit ``transcripts.zarr.zip``.
        cells_path: Explicit ``cells.zarr.zip``.
        cfm_path: Explicit ``cell_feature_matrix.zarr.zip``.
        species: Expected species (``human`` or ``mouse``); checked against the
            panel organism when given.

    Returns:
        The resolved bundle.

    Raises:
        FileNotFoundError: If the directory or a required file is missing.
        AteraFormatError: If a version, unit or header check fails.
    """
    root = Path(input_path)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Atera input {root} is not a directory; extract the outs bundle first"
        )
    spec_path = (
        Path(spec_override)
        if spec_override is not None
        else root / ("experiment.spatial")
    )
    if not spec_path.is_file():
        raise FileNotFoundError(f"{spec_path.name} not found in the Atera input")
    experiment = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict):
        raise AteraFormatError(f"{spec_path.name} is not a JSON object")
    analysis_sw_version = _check_experiment(experiment)
    if species is not None:
        _check_species(experiment, species)
    pixel_size_um = float(experiment["pixel_size"])

    transcripts_zip = _resolve_member(
        root,
        transcripts_path,
        experiment,
        "transcripts_zarr_filepath",
        "transcripts.zarr.zip",
    )
    cells_zip = _resolve_member(
        root, cells_path, experiment, "cells_zarr_filepath", "cells.zarr.zip"
    )
    cfm_zip = _resolve_member(
        root,
        cfm_path,
        experiment,
        "cell_feature_zarr_filepath",
        "cell_feature_matrix.zarr.zip",
    )
    _check_rna_dataset(transcripts_zip)
    _check_cells_dataset(cells_zip)
    _check_cell_feature_matrix(cfm_zip)

    morphology_dir = _morphology_2d_dir(root, experiment)
    files, names, shape_yx, ome_pixel_size = _morphology_2d_channels(morphology_dir)
    if ome_pixel_size is not None and not math.isclose(
        ome_pixel_size, pixel_size_um, rel_tol=PIXEL_SIZE_RTOL
    ):
        raise AteraFormatError(
            f"experiment pixel_size {pixel_size_um} and OME PhysicalSizeX "
            f"{ome_pixel_size} differ"
        )
    images = experiment.get("images") or {}
    morphology_3d = _relative_member(root, images.get("morphology_3d_filepath"))
    panel_config = root / "panel_config.json"
    analysis_summary = sorted(root.glob("analysis_summary*.html"))
    sidecars: list[Path] = []
    for pattern in _SIDECAR_GLOBS:
        sidecars.extend(path for path in sorted(root.glob(pattern)) if path.is_file())
    return AteraBundle(
        root=root,
        experiment=experiment,
        pixel_size_um=pixel_size_um,
        analysis_sw_version=analysis_sw_version,
        transcripts_zip=transcripts_zip,
        cells_zip=cells_zip,
        cell_feature_matrix_zip=cfm_zip,
        morphology_2d_files=files,
        morphology_channel_names=names,
        morphology_shape_yx=shape_yx,
        morphology_3d_file=morphology_3d,
        panel_config_file=panel_config if panel_config.is_file() else None,
        analysis_summary_file=analysis_summary[0] if analysis_summary else None,
        sidecars=tuple(dict.fromkeys(sidecars)),
    )


def _check_experiment(experiment: Mapping[str, Any]) -> str:
    version = str(experiment.get("analysis_sw_version", ""))
    if not version.startswith(ATERA_SW_PREFIX):
        raise AteraFormatError(
            f"analysis_sw_version {version!r} is not an Atera version "
            f"(expected {ATERA_SW_PREFIX}*)"
        )
    major = experiment.get("major_version")
    if major not in SUPPORTED_EXPERIMENT_MAJOR_VERSIONS:
        raise AteraFormatError(
            f"experiment.spatial major_version {major!r} is not supported "
            f"(supported: {sorted(SUPPORTED_EXPERIMENT_MAJOR_VERSIONS)})"
        )
    pixel_size = experiment.get("pixel_size")
    if not isinstance(pixel_size, int | float) or not pixel_size > 0:
        raise AteraFormatError(f"experiment pixel_size {pixel_size!r} is not positive")
    return version


def panel_organisms(experiment: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the organisms of every panel named in ``experiment.spatial``.

    Args:
        experiment: Parsed ``experiment.spatial``.

    Returns:
        ``panel_organism`` of each ``panel_*`` entry, in file order.
    """
    return tuple(
        str(value["panel_organism"])
        for value in experiment.values()
        if isinstance(value, dict) and "panel_organism" in value
    )


def _check_species(experiment: Mapping[str, Any], species: str) -> None:
    organisms = panel_organisms(experiment)
    if not organisms:
        raise AteraFormatError("experiment.spatial names no panel organism")
    mismatched = [name for name in organisms if name.lower() != species.lower()]
    if mismatched:
        raise AteraFormatError(
            f"panel organism {mismatched} does not match species {species!r}"
        )


def _relative_member(root: Path, relative: object) -> Path | None:
    """Return ``root / relative`` if it is a safe relative path that exists."""
    if not isinstance(relative, str) or not relative:
        return None
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AteraFormatError(f"bundle path {relative!r} leaves the ROI directory")
    path = root.joinpath(*candidate.parts)
    return path if path.exists() else None


def _resolve_member(
    root: Path,
    explicit: Path | None,
    experiment: Mapping[str, Any],
    explorer_key: str,
    default_name: str,
) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{default_name} override {path} does not exist")
        return path
    explorer = experiment.get("explorer_files") or {}
    named = _relative_member(root, explorer.get(explorer_key))
    if named is not None:
        return named
    plain = root / default_name
    if plain.is_file():
        return plain
    prefixed = sorted(root.glob(f"*_{default_name}"))
    prefixed = [
        path for path in prefixed if not path.name.startswith(("csc_", "binned_"))
    ]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        raise AteraFormatError(
            f"several *_{default_name} files in the Atera input: "
            f"{[path.name for path in prefixed]}"
        )
    raise FileNotFoundError(f"{default_name} not found in the Atera input")


def _check_units(value: object, what: str) -> None:
    if str(value).lower() not in MICRON_UNITS:
        raise AteraFormatError(f"{what} spatial_units {value!r} is not microns")


def _check_rna_dataset(path: Path) -> None:
    with open_zip_group(path) as group:
        attrs = group.attrs.asdict()
    if attrs.get("name") != RNA_DATASET_NAME:
        raise AteraFormatError(f"{path.name} is not an {RNA_DATASET_NAME}")
    major = attrs.get("major_version")
    if major not in SUPPORTED_RNA_MAJOR_VERSIONS:
        raise AteraFormatError(
            f"{RNA_DATASET_NAME} major_version {major!r} is not supported "
            f"(supported: {sorted(SUPPORTED_RNA_MAJOR_VERSIONS)})"
        )
    if attrs.get("coordinate_space") != COORDINATE_SPACE:
        raise AteraFormatError(
            f"transcripts coordinate_space {attrs.get('coordinate_space')!r} "
            f"is not {COORDINATE_SPACE!r}"
        )
    _check_units(attrs.get("spatial_units"), "transcripts")


def _check_cells_dataset(path: Path) -> None:
    with open_zip_group(path) as group:
        attrs = group.attrs.asdict()
    if attrs.get("name") != CELLS_DATASET_NAME:
        raise AteraFormatError(f"{path.name} is not a {CELLS_DATASET_NAME}")
    major = attrs.get("major_version")
    if major not in SUPPORTED_CELLS_MAJOR_VERSIONS:
        raise AteraFormatError(
            f"{CELLS_DATASET_NAME} major_version {major!r} is not supported "
            f"(supported: {sorted(SUPPORTED_CELLS_MAJOR_VERSIONS)})"
        )
    _check_units(attrs.get("spatial_units"), "cells")


def _check_cell_feature_matrix(path: Path) -> None:
    with open_zip_group(path) as group:
        encoding = group.attrs.get("encoding-type")
        members = set(group.keys())
    if encoding != "anndata" or not {"X", "obs", "var"} <= members:
        raise AteraFormatError(f"{path.name} is not an AnnData zarr store")


# --------------------------------------------------------------------------- #
# Privacy: sanitised metadata and neutral names
# --------------------------------------------------------------------------- #


def neutral_sidecar_name(name: str) -> str:
    """Return a bundle file name without run, slide or dataset identifiers.

    ``metrics_summary_<run>.csv`` becomes ``metrics_summary.csv``,
    ``<dataset>_he_imagealignment.csv`` becomes ``he_imagealignment.csv`` and so
    on; names that carry no identifier are returned unchanged.

    Args:
        name: A file name (no directory).

    Returns:
        The neutral name.
    """
    for pattern in _NEUTRAL_NAME_RES:
        match = pattern.match(name)
        if match:
            return f"{match.group(1)}{match.group(2)}"
    return name


def _neutral_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parts = PurePosixPath(value).parts
    name = neutral_sidecar_name(parts[-1])
    if len(parts) > 1 and parts[-2] in ("morphology_2d", "morphology_3d"):
        return f"{parts[-2]}/{name}"
    return name


def sanitize_experiment_spatial(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the non-identifying fields of ``experiment.spatial``.

    The result is an allow-list: versions, pixel and z-step sizes, panel IDs,
    names, sizes and organisms, the segmentation stain, the numeric summary
    metrics, and the file paths with identifiers removed from file names. Run,
    slide, region, instrument and UUID fields, and anything not recognised,
    are dropped.

    Args:
        experiment: Parsed ``experiment.spatial``.

    Returns:
        A new JSON-serialisable dictionary.
    """
    sanitized: dict[str, Any] = {}
    for key, value in experiment.items():
        if key in IDENTIFYING_EXPERIMENT_FIELDS:
            continue
        if _is_allowed_scalar(key, value):
            sanitized[key] = value
        elif key == "panel_config" and isinstance(value, dict):
            if "panel_config_id" in value:
                sanitized[key] = {"panel_config_id": value["panel_config_id"]}
        elif key.startswith("panel_") and isinstance(value, dict):
            sanitized[key] = {k: v for k, v in value.items() if k in _PANEL_FIELDS}
        elif key in _PATH_GROUPS and isinstance(value, dict):
            paths = {k: _neutral_relative_path(v) for k, v in value.items()}
            sanitized[key] = {k: v for k, v in paths.items() if v is not None}
    return sanitized


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_allowed_scalar(key: str, value: object) -> bool:
    if key in _EXPERIMENT_TEXT_FIELDS:
        return isinstance(value, str)
    if key in _EXPERIMENT_NUMBER_FIELDS or _EXPERIMENT_METRIC_RE.match(key):
        return _is_number(value)
    return False


def sanitize_panel_config(panel_config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove customer and designer contact fields from ``panel_config.json``.

    Args:
        panel_config: Parsed ``panel_config.json``.

    Returns:
        A deep copy without ``payloads[].customer`` and
        ``payloads[].designer.designed_by``.
    """
    sanitized: dict[str, Any] = json.loads(json.dumps(panel_config))
    for payload in sanitized.get("payloads") or []:
        if not isinstance(payload, dict):
            continue
        payload.pop("customer", None)
        designer = payload.get("designer")
        if isinstance(designer, dict):
            designer.pop("designed_by", None)
    return sanitized


def sanitize_metrics_summary(text: str) -> str:
    """Drop the identifying columns of a vendor ``metrics_summary`` CSV.

    Args:
        text: The CSV text.

    Returns:
        The CSV text without run, instrument, slide, region and UUID columns.
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return text
    keep = [
        index
        for index, name in enumerate(rows[0])
        if not _IDENTIFYING_METRIC_COLUMN_RE.match(name.strip())
    ]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for row in rows:
        writer.writerow([row[index] for index in keep if index < len(row)])
    return out.getvalue()


# --------------------------------------------------------------------------- #
# Morphology
# --------------------------------------------------------------------------- #


def _morphology_2d_dir(root: Path, experiment: Mapping[str, Any]) -> Path:
    images = experiment.get("images") or {}
    named = _relative_member(root, images.get("morphology_2d_filepath"))
    directory = named.parent if named is not None else root / "morphology_2d"
    if not directory.is_dir():
        raise FileNotFoundError("morphology_2d/ not found in the Atera input")
    return directory


def _ome_namespace(element: ET.Element) -> dict[str, str]:
    return {"o": element.tag.split("}")[0].strip("{")} if "}" in element.tag else {}


def _ome_find(element: ET.Element, tag: str, ns: Mapping[str, str]) -> list[ET.Element]:
    return element.findall(f"o:{tag}" if ns else tag, dict(ns))


@dataclass(frozen=True)
class _OmeHeader:
    uuid: str | None
    channel_names: tuple[str, ...]
    files_by_channel: tuple[tuple[str, str | None], ...]
    pixel_size_um: float | None


def _parse_ome_header(xml: str, path: Path) -> _OmeHeader:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise AteraFormatError(f"{path.name} has unreadable OME-XML") from exc
    ns = _ome_namespace(root)
    images = _ome_find(root, "Image", ns)
    if len(images) != 1:
        raise AteraFormatError(f"{path.name} OME-XML has {len(images)} images, not 1")
    pixels_list = _ome_find(images[0], "Pixels", ns)
    if len(pixels_list) != 1:
        raise AteraFormatError(f"{path.name} OME-XML has no single Pixels element")
    pixels = pixels_list[0]
    size_c = int(pixels.get("SizeC", "1"))
    names = tuple(
        str(channel.get("Name", "")).strip()
        for channel in _ome_find(pixels, "Channel", ns)
    )
    if len(names) != size_c or not all(names):
        raise AteraFormatError(
            f"{path.name} OME-XML names {len(names)} channels for SizeC={size_c}"
        )
    mapping: dict[int, tuple[str, str | None]] = {}
    for tiff_data in _ome_find(pixels, "TiffData", ns):
        first_c = int(tiff_data.get("FirstC", "0"))
        if int(tiff_data.get("PlaneCount", "1")) != 1 or int(tiff_data.get("IFD", "0")):
            raise AteraFormatError(
                f"{path.name} TiffData for channel {first_c} is not one plane per file"
            )
        uuids = _ome_find(tiff_data, "UUID", ns)
        if len(uuids) != 1 or not uuids[0].get("FileName"):
            raise AteraFormatError(
                f"{path.name} TiffData for channel {first_c} names no file"
            )
        if first_c in mapping:
            raise AteraFormatError(f"{path.name} maps channel {first_c} twice")
        mapping[first_c] = (str(uuids[0].get("FileName")), uuids[0].text)
    if sorted(mapping) != list(range(size_c)):
        raise AteraFormatError(
            f"{path.name} TiffData covers channels {sorted(mapping)}, "
            f"not 0..{size_c - 1}"
        )
    pixel_size = None
    if pixels.get("PhysicalSizeX"):
        unit = pixels.get("PhysicalSizeXUnit", "µm")
        if str(unit).lower() not in MICRON_UNITS:
            raise AteraFormatError(f"{path.name} PhysicalSizeXUnit {unit!r} is not µm")
        pixel_size = float(pixels.get("PhysicalSizeX", "nan"))
    return _OmeHeader(
        uuid=root.get("UUID"),
        channel_names=names,
        files_by_channel=tuple(mapping[c] for c in range(size_c)),
        pixel_size_um=pixel_size,
    )


def _single_plane_levels(path: Path) -> tuple[str, tuple[tuple[int, int], ...], str]:
    """Return a file's OME-XML, its own plane's level shapes and dtype."""
    # _multifile=False reads only this file's plane and its SubIFD levels;
    # the multi-file OME series would zero-fill siblings it cannot find.
    with tifffile.TiffFile(path, _multifile=False) as tif:
        xml = tif.ome_metadata or ""
        series = tif.series[0]
        shapes = tuple(tuple(int(n) for n in level.shape) for level in series.levels)
        dtype = str(series.dtype)
    if any(len(shape) != 2 for shape in shapes):
        raise AteraFormatError(f"{path.name} is not one 2D plane per level: {shapes}")
    return xml, tuple((shape[0], shape[1]) for shape in shapes), dtype


def _morphology_2d_channels(
    directory: Path,
) -> tuple[tuple[Path, ...], tuple[str, ...], tuple[int, int], float | None]:
    """Map ``morphology_2d`` files to channels from the OME ``TiffData``."""
    candidates = sorted(directory.glob("*.ome.tif")) + sorted(
        directory.glob("*.ome.tiff")
    )
    if not candidates:
        raise FileNotFoundError("morphology_2d/ holds no OME-TIFF")
    headers = {}
    level_shapes = {}
    for path in candidates:
        xml, shapes, _ = _single_plane_levels(path)
        if not xml:
            raise AteraFormatError(f"{path.name} carries no OME-XML")
        headers[path.name] = _parse_ome_header(xml, path)
        level_shapes[path.name] = shapes
    reference = headers[candidates[0].name]
    files: list[Path] = []
    for channel, (file_name, file_uuid) in enumerate(reference.files_by_channel):
        if file_name not in headers:
            raise AteraFormatError(
                f"morphology_2d channel {channel} ({reference.channel_names[channel]}) "
                f"file {file_name} is missing; keep the sibling files together"
            )
        header = headers[file_name]
        if header.channel_names != reference.channel_names:
            raise AteraFormatError(f"{file_name} names different OME channels")
        if file_uuid and header.uuid and file_uuid != header.uuid:
            raise AteraFormatError(
                f"{file_name} UUID does not match its TiffData entry"
            )
        files.append(directory / file_name)
    unused = sorted(set(headers) - {path.name for path in files})
    if unused:
        logger.warning(
            "morphology_2d files not in the OME series are ignored: %s", unused
        )
    distinct_shapes = {level_shapes[path.name] for path in files}
    if len(distinct_shapes) != 1:
        raise AteraFormatError("morphology_2d files have different level shapes")
    return (
        tuple(files),
        reference.channel_names,
        level_shapes[files[0].name][0],
        reference.pixel_size_um,
    )


class _LazyTiffLevel:
    """Array-like view of one stored pyramid level of one OME-TIFF plane.

    Every ``__getitem__`` opens the file, reads the region and closes it, so a
    dask graph holds no open file handles.
    """

    def __init__(
        self, path: Path, level: int, shape: tuple[int, int], dtype: str
    ) -> None:
        self.path = path
        self.level = level
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.ndim = 2

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        with tifffile.TiffFile(self.path, _multifile=False) as tif:
            store = tif.series[0].aszarr(level=self.level)
            try:
                return np.asarray(zarr.open_array(store, mode="r")[key])
            finally:
                store.close()

    def __dask_tokenize__(self) -> tuple[str, int, float]:
        return (str(self.path), self.level, self.path.stat().st_mtime)


def read_atera_morphology_2d(
    bundle: AteraBundle,
    *,
    n_levels: int = 5,
    chunks: tuple[int, int] = (4096, 4096),
) -> xr.DataTree:
    """Read ``morphology_2d`` as a lazy multiscale ``(c, y, x)`` image.

    Each channel comes from its own file's stored pyramid (never from a
    multi-file series, which zero-fills missing siblings), in OME ``FirstC``
    order. Levels are taken as stored; spatialdata derives each level's scale
    from the shape ratios when writing, which handles odd sizes.

    Args:
        bundle: Resolved bundle.
        n_levels: Largest number of levels to read, level 0 included.
        chunks: ``(y, x)`` dask chunk size per channel.

    Returns:
        A DataTree ``scale0, scale1, ...`` with an Identity transform to
        ``global``.

    Raises:
        AteraFormatError: If the files disagree on level shapes or dtype.
    """
    level_lists = []
    dtypes = set()
    for path in bundle.morphology_2d_files:
        _, shapes, dtype = _single_plane_levels(path)
        level_lists.append(shapes)
        dtypes.add(dtype)
    if len(set(level_lists)) != 1 or len(dtypes) != 1:
        raise AteraFormatError("morphology_2d files differ in level shapes or dtype")
    shapes = level_lists[0][: max(1, int(n_levels))]
    dtype = dtypes.pop()
    arrays = []
    for level, shape in enumerate(shapes):
        level_chunks = (min(chunks[0], shape[0]), min(chunks[1], shape[1]))
        planes = [
            da.from_array(
                _LazyTiffLevel(path, level, shape, dtype),
                chunks=level_chunks,
                lock=False,
                asarray=True,
                fancy=False,
                name=f"atera-morphology-{path.name}-{level}-"
                + dask.base.tokenize(_LazyTiffLevel(path, level, shape, dtype)),
            )
            for path in bundle.morphology_2d_files
        ]
        arrays.append(da.stack(planes, axis=0))
    tree = dask_arrays_to_datatree(
        arrays, dims=("c", "y", "x"), channels=list(bundle.morphology_channel_names)
    )
    set_transformation(tree, {"global": Identity()}, set_all=True)
    return cast(xr.DataTree, tree)


# --------------------------------------------------------------------------- #
# Cells, labels and shapes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AteraCells:
    """Cell and nucleus records of ``cells.zarr.zip`` (no masks).

    Attributes:
        cell_id_prefix: uint32 cell ID prefix per cell row.
        dataset_suffix: uint32 dataset suffix per cell row.
        source_cell_id: 10x string ID per cell (CFM ``obs_names``).
        vendor_cell_id: Packed ``(suffix << 32) | prefix`` (CFM ``obs.cell_id``).
        summary: ``cell_summary`` with its named columns, one row per cell.
        cell_segmentation_method: Decoded method text per cell.
        nucleus_cell_index: Owning cell row per nucleus row.
        nucleus_segmentation_method: Decoded method text per nucleus.
        cell_vertices: Vendor cell polygons, ``[n, 50]`` (x, y pairs, µm).
        cell_num_vertices: Vertex count per cell polygon (closing vertex included).
        nucleus_vertices: Vendor nucleus polygons, ``[n_nuclei, 50]``.
        nucleus_num_vertices: Vertex count per nucleus polygon.
        mask_shape: ``(y, x)`` shape of both masks.
        pixel_size_um: Microns per mask pixel.
    """

    cell_id_prefix: np.ndarray
    dataset_suffix: np.ndarray
    source_cell_id: np.ndarray
    vendor_cell_id: np.ndarray
    summary: pd.DataFrame
    cell_segmentation_method: np.ndarray
    nucleus_cell_index: np.ndarray
    nucleus_segmentation_method: np.ndarray
    cell_vertices: np.ndarray
    cell_num_vertices: np.ndarray
    nucleus_vertices: np.ndarray
    nucleus_num_vertices: np.ndarray
    mask_shape: tuple[int, int]
    pixel_size_um: float

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return len(self.cell_id_prefix)

    @property
    def n_nuclei(self) -> int:
        """Number of nuclei."""
        return len(self.nucleus_cell_index)

    @property
    def instance_id(self) -> np.ndarray:
        """Canonical cell IDs: cell row + 1 = ``masks/1`` label (uint64)."""
        return np.arange(1, self.n_cells + 1, dtype=np.uint64)

    @property
    def nucleus_instance_id(self) -> np.ndarray:
        """Nucleus IDs: nucleus row + 1 = ``masks/0`` label (uint64)."""
        return np.arange(1, self.n_nuclei + 1, dtype=np.uint64)

    @property
    def nucleus_cell_instance_id(self) -> np.ndarray:
        """Owning cell's ``instance_id`` per nucleus (uint64)."""
        return self.nucleus_cell_index.astype(np.uint64) + np.uint64(1)


def check_homogeneous_transform(matrix: np.ndarray, pixel_size_um: float) -> None:
    """Check that the vendor µm-to-pixel transform is a pure ``1 / ps`` scale.

    Args:
        matrix: ``masks/homogeneous_transform`` (4 x 4, float32).
        pixel_size_um: Microns per pixel.

    Raises:
        AteraFormatError: If the matrix is not ``diag(1/ps, 1/ps, 1, 1)``
            within float32 precision (relative tolerance 1e-6).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    expected = np.diag([1.0 / pixel_size_um, 1.0 / pixel_size_um, 1.0, 1.0])
    if matrix.shape != (4, 4) or not np.allclose(
        matrix, expected, rtol=TRANSFORM_RTOL, atol=0.0
    ):
        raise AteraFormatError(
            "masks/homogeneous_transform is not a pure 1/pixel_size scale: "
            f"{matrix.tolist()}"
        )


def _decode_methods(codes: np.ndarray, methods: Sequence[str], what: str) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.size and (codes.min() < 0 or codes.max() >= len(methods)):
        raise AteraFormatError(f"{what} segmentation method codes out of range")
    return (
        np.asarray(methods, dtype=object)[codes]
        if codes.size
        else np.asarray([], dtype=object)
    )


def read_atera_cells(bundle: AteraBundle) -> AteraCells:
    """Read cell IDs, the cell summary and vendor polygons from ``cells.zarr.zip``.

    Args:
        bundle: Resolved bundle.

    Returns:
        The cell and nucleus records.

    Raises:
        AteraFormatError: If the records are inconsistent (cell count, cell
            index, nucleus count, transform, duplicate IDs).
    """
    with open_zip_group(bundle.cells_zip) as group:
        attrs = group.attrs.asdict()
        cell_id = np.asarray(group["cell_id"][:])
        summary_array = group["cell_summary"]
        columns = list(summary_array.attrs.get("column_names", []))
        summary_values = np.asarray(summary_array[:])
        polygons = {
            key: {
                name: np.asarray(group[f"polygon_sets/{key}/{name}"][:])
                for name in ("cell_index", "method", "num_vertices", "vertices")
            }
            for key in ("0", "1")
        }
        mask_shapes = {
            tuple(int(n) for n in group[f"masks/{key}"].shape) for key in ("0", "1")
        }
        transform = np.asarray(group["masks/homogeneous_transform"][:])
    n_cells = int(cell_id.shape[0])
    if attrs.get("number_cells", n_cells) != n_cells:
        raise AteraFormatError(
            f"number_cells {attrs.get('number_cells')} != {n_cells} cell_id rows"
        )
    if cell_id.ndim != 2 or cell_id.shape[1] != 2:
        raise AteraFormatError(f"cell_id has shape {cell_id.shape}, not (n, 2)")
    if len(columns) != summary_values.shape[1] or summary_values.shape[0] != n_cells:
        raise AteraFormatError("cell_summary shape does not match its column names")
    check_homogeneous_transform(transform, bundle.pixel_size_um)
    if len(mask_shapes) != 1:
        raise AteraFormatError(f"masks/0 and masks/1 differ in shape: {mask_shapes}")
    mask_shape = mask_shapes.pop()
    if len(mask_shape) != 2:
        raise AteraFormatError(f"masks are not 2D: {mask_shape}")

    cells_set, nuclei_set = polygons["1"], polygons["0"]
    if not np.array_equal(cells_set["cell_index"], np.arange(n_cells)):
        raise AteraFormatError("polygon_sets/1/cell_index is not the cell row order")
    nucleus_cell_index = nuclei_set["cell_index"].astype(np.int64)
    if nucleus_cell_index.size and (
        nucleus_cell_index.min() < 0 or nucleus_cell_index.max() >= n_cells
    ):
        raise AteraFormatError("polygon_sets/0/cell_index names unknown cells")
    summary = pd.DataFrame(summary_values, columns=columns)
    if "nucleus_count" in summary:
        observed = np.bincount(nucleus_cell_index, minlength=n_cells)
        if not np.array_equal(observed, summary["nucleus_count"].to_numpy()):
            raise AteraFormatError(
                "cell_summary.nucleus_count disagrees with polygon_sets/0/cell_index"
            )
    prefix = cell_id[:, 0].astype(np.uint32)
    suffix = cell_id[:, 1].astype(np.uint32)
    packed = pack_cell_ids(prefix, suffix)
    if len(np.unique(packed)) != n_cells:
        raise AteraFormatError("cells.zarr.zip holds duplicate cell IDs")
    methods = [str(name) for name in attrs.get("segmentation_methods", [])]
    return AteraCells(
        cell_id_prefix=prefix,
        dataset_suffix=suffix,
        source_cell_id=cell_id_str_from_prefix_suffix_uint32(prefix, suffix),
        vendor_cell_id=packed.astype(np.int64),
        summary=summary,
        cell_segmentation_method=_decode_methods(cells_set["method"], methods, "cell"),
        nucleus_cell_index=nucleus_cell_index,
        nucleus_segmentation_method=_decode_methods(
            nuclei_set["method"], methods, "nucleus"
        ),
        cell_vertices=cells_set["vertices"].astype(np.float32),
        cell_num_vertices=cells_set["num_vertices"].astype(np.int64),
        nucleus_vertices=nuclei_set["vertices"].astype(np.float32),
        nucleus_num_vertices=nuclei_set["num_vertices"].astype(np.int64),
        mask_shape=(int(mask_shape[0]), int(mask_shape[1])),
        pixel_size_um=bundle.pixel_size_um,
    )


class _LazyZipArray:
    """Array-like view of one array inside a zipped zarr store.

    Every ``__getitem__`` opens its own ``ZipStore``, because a store
    serialises reads behind a lock.
    """

    def __init__(self, zip_path: Path, member: str) -> None:
        self.zip_path = zip_path
        self.member = member
        with open_zip_group(zip_path) as group:
            array = group[member]
            self.shape = tuple(int(n) for n in array.shape)
            self.dtype = np.dtype(array.dtype)
            self.native_chunks = tuple(int(n) for n in array.chunks)
        self.ndim = len(self.shape)

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        with open_zip_group(self.zip_path) as group:
            return np.asarray(group[self.member][key])

    def __dask_tokenize__(self) -> tuple[str, str, float]:
        return (str(self.zip_path), self.member, self.zip_path.stat().st_mtime)


def _zip_dask_array(zip_path: Path, member: str, chunks: tuple[int, ...]) -> da.Array:
    source = _LazyZipArray(zip_path, member)
    chunks = tuple(min(c, n) for c, n in zip(chunks, source.shape, strict=True))
    array: da.Array = da.from_array(
        source,
        chunks=chunks,
        lock=False,
        asarray=True,
        fancy=False,
        name=f"atera-{member.replace('/', '-')}-{dask.base.tokenize(source)}",
    )
    return array


def read_atera_labels(
    bundle: AteraBundle, *, chunks: tuple[int, int] = (4096, 4096)
) -> dict[str, da.Array]:
    """Read the vendor masks as lazy label arrays.

    Args:
        bundle: Resolved bundle.
        chunks: ``(y, x)`` chunk size; aligned square chunks halve cleanly
            when the builder writes a label pyramid.

    Returns:
        ``{"cell_labels": masks/1, "nucleus_labels": masks/0}`` (uint32).

    Raises:
        AteraFormatError: If a mask's shape differs from the morphology image.
    """
    labels = {
        "cell_labels": _zip_dask_array(bundle.cells_zip, "masks/1", chunks),
        "nucleus_labels": _zip_dask_array(bundle.cells_zip, "masks/0", chunks),
    }
    for key, array in labels.items():
        if tuple(array.shape) != tuple(bundle.morphology_shape_yx):
            raise AteraFormatError(
                f"{key} shape {array.shape} != morphology shape "
                f"{bundle.morphology_shape_yx}"
            )
    return labels


def _mask_row_slabs(
    shape: tuple[int, ...], chunk_rows: int, target_bytes: int
) -> Iterator[slice]:
    rows_per_slab = max(
        chunk_rows, (target_bytes // max(1, shape[1] * 4)) // chunk_rows * chunk_rows
    )
    for start in range(0, shape[0], rows_per_slab):
        yield slice(start, min(shape[0], start + rows_per_slab))


def export_mask_npy(
    bundle: AteraBundle, member: str, path: Path, *, slab_bytes: int = 256 << 20
) -> Path:
    """Stream one vendor mask into an ``.npy`` file without holding it in memory.

    Args:
        bundle: Resolved bundle.
        member: ``"masks/0"`` (nuclei) or ``"masks/1"`` (cells).
        path: Output ``.npy`` file.
        slab_bytes: Approximate size of each row slab read from the zip.

    Returns:
        ``path``.
    """
    with open_zip_group(bundle.cells_zip) as group:
        source = group[member]
        shape = tuple(int(n) for n in source.shape)
        out = np.lib.format.open_memmap(
            path, mode="w+", dtype=source.dtype, shape=shape
        )
        try:
            for rows in _mask_row_slabs(shape, int(source.chunks[0]), slab_bytes):
                out[rows] = source[rows]
            out.flush()
        finally:
            del out
    return path


def atera_nucleus_area_total(
    bundle: AteraBundle, cells: AteraCells, *, slab_bytes: int = 256 << 20
) -> np.ndarray:
    """Sum the mask area of every nucleus of each cell, streaming ``masks/0``.

    ``cell_summary.nucleus_area`` describes only the largest nucleus of a
    multinucleate cell; this is the total.

    Args:
        bundle: Resolved bundle.
        cells: Cell records.
        slab_bytes: Approximate size of each row slab read from the zip.

    Returns:
        float64 area in µm² per cell, NaN for cells without a nucleus.

    Raises:
        AteraFormatError: If ``masks/0`` holds a label without a nucleus row.
    """
    counts = np.zeros(cells.n_nuclei + 1, dtype=np.int64)
    with open_zip_group(bundle.cells_zip) as group:
        source = group["masks/0"]
        shape = tuple(int(n) for n in source.shape)
        for rows in _mask_row_slabs(shape, int(source.chunks[0]), slab_bytes):
            slab = np.asarray(source[rows]).ravel()
            if slab.size and int(slab.max()) > cells.n_nuclei:
                raise AteraFormatError(
                    f"masks/0 label {int(slab.max())} exceeds {cells.n_nuclei} nuclei"
                )
            counts += np.bincount(slab, minlength=cells.n_nuclei + 1)
    per_cell = np.bincount(
        cells.nucleus_cell_index, weights=counts[1:], minlength=cells.n_cells
    ) * (bundle.pixel_size_um**2)
    has_nucleus = np.bincount(cells.nucleus_cell_index, minlength=cells.n_cells) > 0
    return np.where(has_nucleus, per_cell, np.nan)


def _pixel_polygons_to_microns(
    geometries: np.ndarray, pixel_size_um: float, simplify_px: float
) -> np.ndarray:
    """Simplify pixel-space contours, then map them to microns at ``(c + 0.5) ps``."""
    if simplify_px > 0:
        geometries = shapely.simplify(geometries, simplify_px, preserve_topology=True)
    shifted: np.ndarray = shapely.transform(
        geometries, lambda coords: (coords + 0.5) * pixel_size_um
    )
    return shifted


def _shape_frame(
    geometries: np.ndarray,
    instance_ids: np.ndarray,
    columns: Mapping[str, np.ndarray],
) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(
        {INSTANCE_ID_COLUMN: instance_ids.astype(np.uint64), **columns},
        geometry=list(geometries),
    )
    frame.index = pd.Index(
        instance_ids.astype(np.uint64), dtype="uint64", name=INSTANCE_ID_COLUMN
    )
    return frame


def _cell_shape_columns(cells: AteraCells) -> dict[str, np.ndarray]:
    return {
        SOURCE_CELL_ID_COLUMN: cells.source_cell_id,
        "segmentation_method": cells.cell_segmentation_method,
    }


def _nucleus_shape_columns(
    cells: AteraCells, rows: np.ndarray
) -> dict[str, np.ndarray]:
    owners = cells.nucleus_cell_index[rows]
    return {
        "cell_instance_id": owners.astype(np.uint64) + np.uint64(1),
        SOURCE_CELL_ID_COLUMN: cells.source_cell_id[owners],
        "segmentation_method": cells.nucleus_segmentation_method[rows],
    }


def atera_shapes_from_masks(
    bundle: AteraBundle,
    cells: AteraCells,
    *,
    n_jobs: int,
    workdir: Path,
    simplify_px: float = 0.5,
    mp_context: MpStartMethod = "forkserver",
) -> dict[str, gpd.GeoDataFrame]:
    """Polygonise the vendor masks into cell and nucleus shapes (microns).

    Each mask is streamed into a memory-mapped ``.npy`` in ``workdir`` and
    polygonised by a process pool that maps the file (no copies, no silent
    serial fallback). Contours, whose pixel centres sit at integers, are
    simplified topology-preservingly by ``simplify_px`` pixels and mapped to
    microns at ``(c + 0.5) * ps``, the vendor's corner convention. A label
    made of several components becomes a MultiPolygon.

    Call it before any thread pool starts when ``mp_context="fork"``; the
    default ``forkserver`` starts clean workers.

    Args:
        bundle: Resolved bundle.
        cells: Cell records.
        n_jobs: Worker processes for the polygonisation pool.
        workdir: Directory for the temporary ``.npy`` masks (deleted after).
        simplify_px: Simplification tolerance in pixels (0 disables it).
        mp_context: Multiprocessing start method of the pool.

    Returns:
        ``{"cell_boundaries": ..., "nucleus_boundaries": ...}``. Both carry
        ``instance_id`` as a uint64 column and index (cell row + 1, nucleus
        row + 1), ``source_cell_id`` (of the owning cell) and
        ``segmentation_method``; nuclei also carry ``cell_instance_id``.

    Raises:
        AteraFormatError: If a label yields no polygon or exceeds the record count.
        RuntimeError: If the polygonisation pool fails.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, gpd.GeoDataFrame] = {}
    for key, member, n_expected in (
        ("cell_boundaries", "masks/1", cells.n_cells),
        ("nucleus_boundaries", "masks/0", cells.n_nuclei),
    ):
        npy = export_mask_npy(
            bundle, member, workdir / f"atera_{member.replace('/', '_')}.npy"
        )
        try:
            mask = np.load(npy, mmap_mode="r")
            labeled = masks_to_labeled_polygons(
                mask, n_jobs=n_jobs, serial_fallback=False, mp_context=mp_context
            )
            del mask
        finally:
            npy.unlink(missing_ok=True)
        labels = np.asarray([label for label, _ in labeled], dtype=np.int64)
        if labels.size and labels.max() > n_expected:
            raise AteraFormatError(
                f"{member} label {labels.max()} exceeds {n_expected} records"
            )
        missing = np.setdiff1d(np.arange(1, n_expected + 1), labels)
        if missing.size:
            raise AteraFormatError(
                f"{member}: {missing.size} label(s) without a polygon, "
                f"e.g. {missing[:5].tolist()}"
            )
        order = np.argsort(labels)
        geometries = _pixel_polygons_to_microns(
            np.asarray([geometry for _, geometry in labeled], dtype=object)[order],
            bundle.pixel_size_um,
            simplify_px,
        )
        rows = labels[order] - 1
        columns = (
            _cell_shape_columns(cells)
            if key == "cell_boundaries"
            else _nucleus_shape_columns(cells, rows)
        )
        shapes[key] = _shape_frame(geometries, rows + 1, columns)
    return shapes


def atera_shapes_from_vendor_polygons(
    cells: AteraCells, pixel_size_um: float
) -> dict[str, gpd.GeoDataFrame]:
    """Build shapes from the vendor display polygons (fast fallback).

    The stored closing vertex is dropped and the ring closed again from the
    first vertex; rows with ``num_vertices == 0`` (no polygon) are skipped;
    vertices, which sit on pixel indices times ``ps``, move by ``+0.5 * ps`` to
    the corner convention. The polygons are simplified displays (median area
    about 0.95 of the mask), so prefer :func:`atera_shapes_from_masks`.

    Args:
        cells: Cell records.
        pixel_size_um: Microns per pixel.

    Returns:
        The same elements and columns as :func:`atera_shapes_from_masks`.
    """
    shapes: dict[str, gpd.GeoDataFrame] = {}
    for key, vertices, num_vertices in (
        ("cell_boundaries", cells.cell_vertices, cells.cell_num_vertices),
        ("nucleus_boundaries", cells.nucleus_vertices, cells.nucleus_num_vertices),
    ):
        rows = np.flatnonzero(num_vertices > 1)
        rings = [
            vertices[row, : 2 * (int(num_vertices[row]) - 1)]
            .reshape(-1, 2)
            .astype(np.float64)
            for row in rows
        ]
        geometries = np.asarray(
            [shapely.Polygon(ring + 0.5 * pixel_size_um) for ring in rings],
            dtype=object,
        )
        columns = (
            {name: values[rows] for name, values in _cell_shape_columns(cells).items()}
            if key == "cell_boundaries"
            else _nucleus_shape_columns(cells, rows)
        )
        shapes[key] = _shape_frame(geometries, rows + 1, columns)
    return shapes


def polygon_vertex_stats(shapes: gpd.GeoDataFrame) -> dict[str, float | int]:
    """Summarise the polygon vertex counts of a shapes element.

    Args:
        shapes: A shapes GeoDataFrame.

    Returns:
        Shape count, MultiPolygon count and the min / median / max vertices.
    """
    counts = shapely.get_num_coordinates(shapes.geometry.to_numpy())
    multi = int((shapes.geometry.geom_type == "MultiPolygon").sum())
    if counts.size == 0:
        return {"n_shapes": 0, "n_multipolygons": 0}
    return {
        "n_shapes": int(counts.size),
        "n_multipolygons": multi,
        "vertices_min": int(counts.min()),
        "vertices_median": float(np.median(counts)),
        "vertices_max": int(counts.max()),
    }


# --------------------------------------------------------------------------- #
# Cell-feature matrix
# --------------------------------------------------------------------------- #


def _plain(values: Any) -> Any:
    """Turn numpy ``StringDType`` arrays (anndata string arrays) into object arrays."""
    if isinstance(values, np.ndarray) and values.dtype.kind == "T":
        return values.astype(object)
    return values


def _read_columns(group: Any, names: Sequence[str]) -> dict[str, Any]:
    return {
        name: _plain(ad.io.read_elem(group[name])) for name in names if name in group
    }


def _cfm_index(group: Any) -> np.ndarray:
    index_name = group.attrs.get("_index")
    if not isinstance(index_name, str) or index_name not in group:
        raise AteraFormatError("cell_feature_matrix frame has no readable index")
    return np.asarray(_plain(ad.io.read_elem(group[index_name])), dtype=object)


def read_atera_features(bundle: AteraBundle) -> pd.DataFrame:
    """Read the feature table (``var``) of the cell-feature matrix.

    Only the columns MerXen uses are read; the ``de_*`` columns are skipped.

    Args:
        bundle: Resolved bundle.

    Returns:
        A frame indexed by ``feature_id`` with ``feature_name``,
        ``feature_type`` and, when present, ``genome``, ``filtered`` and
        ``highly_variable``, in vendor feature order.
    """
    with open_zip_group(bundle.cell_feature_matrix_zip) as group:
        var = group["var"]
        index = _cfm_index(var)
        columns = _read_columns(
            var,
            ("feature_name", "feature_type", "genome", "filtered", "highly_variable"),
        )
    missing = {"feature_name", "feature_type"} - set(columns)
    if missing:
        raise AteraFormatError(f"cell_feature_matrix var lacks {sorted(missing)}")
    frame = pd.DataFrame(
        columns, index=pd.Index(index, dtype=object, name="feature_id")
    )
    frame["feature_name"] = frame["feature_name"].astype(str)
    frame["feature_type"] = frame["feature_type"].astype(str)
    return frame


def _gene_coverage(panel_config_file: Path | None) -> dict[str, int]:
    if panel_config_file is None:
        return {}
    panel = json.loads(panel_config_file.read_text(encoding="utf-8"))
    coverage: dict[str, int] = {}
    for payload in panel.get("payloads") or []:
        for target in payload.get("targets") or []:
            identity = ((target.get("type") or {}).get("data") or {}).get("id")
            value = (target.get("info") or {}).get("gene_coverage")
            if identity is not None and value is not None:
                coverage[str(identity)] = int(value)
    return coverage


def read_atera_gene_names(bundle: AteraBundle) -> tuple[str, ...]:
    """Return the transcripts' ``gene_names`` (genes and controls, vendor order).

    Args:
        bundle: Resolved bundle.

    Returns:
        The feature names in vendor order.
    """
    with open_zip_group(bundle.transcripts_zip) as group:
        return tuple(str(name) for name in group.attrs["gene_names"])


def read_atera_table(
    bundle: AteraBundle,
    cells: AteraCells,
    *,
    gex_only: bool = True,
    keep_vendor_annotations: bool = True,
    nucleus_area_total: np.ndarray | None = None,
) -> ad.AnnData:
    """Read the cell-feature matrix as MerXen's cell table.

    Per-cell control counts are derived from the control columns before X is
    reduced to genes. ``var_names`` are the gene symbols (unique), with the
    Ensembl IDs in ``var["gene_ids"]``; vendor columns take a ``vendor_``
    prefix and are each optional; ``de_*`` columns are dropped. The vendor's
    ``filtered`` flags (True = kept) are copied, not applied.

    Args:
        bundle: Resolved bundle.
        cells: Cell records (row order must match the matrix).
        gex_only: Keep only ``Gene Expression`` columns in X.
        keep_vendor_annotations: Copy Leiden, Azimuth, filtered flags, vendor
            counts and the vendor PCA/UMAP.
        nucleus_area_total: Total nucleus area per cell; computed from
            ``masks/0`` (one streaming pass) when ``None``.

    Returns:
        AnnData with float32 CSR counts, ``obs_names`` = ``instance_id`` text,
        and ``obsm["spatial"]`` = cell centroids in µm.

    Raises:
        AteraFormatError: If cell or feature order disagrees with the other
            stores, or gene symbols are not unique.
    """
    features = read_atera_features(bundle)
    gene_names = read_atera_gene_names(bundle)
    if tuple(features["feature_name"]) != gene_names:
        raise AteraFormatError(
            "cell_feature_matrix features differ from transcripts gene_names"
        )
    with open_zip_group(bundle.cell_feature_matrix_zip) as group:
        matrix = ad.io.read_elem(group["X"])
        obs_group = group["obs"]
        barcodes = _cfm_index(obs_group)
        obs_columns = _read_columns(obs_group, ("cell_id", "cell_area", "nucleus_area"))
        vendor_columns = (
            _read_columns(obs_group, VENDOR_OBS_COLUMNS)
            if keep_vendor_annotations
            else {}
        )
        obsm = (
            _read_columns(group["obsm"], ("X_pca", "X_umap"))
            if keep_vendor_annotations and "obsm" in group
            else {}
        )
    matrix = sparse.csr_matrix(matrix)
    if matrix.shape != (cells.n_cells, len(features)):
        raise AteraFormatError(
            f"cell_feature_matrix X {matrix.shape} != "
            f"({cells.n_cells}, {len(features)})"
        )
    if not np.array_equal(barcodes, cells.source_cell_id):
        raise AteraFormatError(
            "cell_feature_matrix barcodes differ from cells.zarr order"
        )
    if "cell_id" in obs_columns and not np.array_equal(
        np.asarray(obs_columns["cell_id"], dtype=np.int64), cells.vendor_cell_id
    ):
        raise AteraFormatError(
            "cell_feature_matrix obs.cell_id differs from cells.zarr"
        )

    feature_types = features["feature_type"].to_numpy()
    obs = pd.DataFrame(index=pd.Index(cells.instance_id.astype(str)))
    obs[INSTANCE_ID_COLUMN] = cells.instance_id
    obs[SOURCE_CELL_ID_COLUMN] = cells.source_cell_id
    obs["vendor_cell_id"] = cells.vendor_cell_id
    summary = cells.summary
    obs["cell_area"] = summary["cell_area"].to_numpy(np.float64)
    obs["nucleus_area"] = summary["nucleus_area"].to_numpy(np.float64)
    if nucleus_area_total is None:
        nucleus_area_total = atera_nucleus_area_total(bundle, cells)
    obs["nucleus_area_total"] = np.asarray(nucleus_area_total, dtype=np.float64)
    obs["nucleus_count"] = summary["nucleus_count"].to_numpy().astype(np.int64)
    if "z_level" in summary:
        obs["vendor_z_level"] = summary["z_level"].to_numpy(np.float64)
    for column, feature_type in DERIVED_CONTROL_COUNT_COLUMNS.items():
        selected = np.flatnonzero(feature_types == feature_type)
        obs[column] = (
            np.asarray(matrix[:, selected].sum(axis=1)).ravel().astype(np.int64)
        )
    for name, values in vendor_columns.items():
        obs[f"vendor_{name}"] = (
            values.values if isinstance(values, pd.Series) else values
        )

    keep = (
        np.flatnonzero(feature_types == GENE_EXPRESSION_TYPE)
        if gex_only
        else np.arange(len(features))
    )
    kept = features.iloc[keep]
    symbols = kept["feature_name"].to_numpy()
    if len(set(symbols)) != len(symbols):
        raise AteraFormatError("gene symbols in the cell-feature matrix are not unique")
    coverage = _gene_coverage(bundle.panel_config_file)
    var = pd.DataFrame(index=pd.Index(symbols.astype(str)))
    var["gene_ids"] = kept.index.to_numpy().astype(str)
    var["feature_types"] = pd.Categorical(kept["feature_type"].to_numpy())
    if "genome" in kept:
        var["genome"] = pd.Categorical(kept["genome"].astype(str).to_numpy())
    if "filtered" in kept:
        var["vendor_filtered"] = kept["filtered"].to_numpy(dtype=bool)
    if "highly_variable" in kept:
        var["vendor_highly_variable"] = kept["highly_variable"].to_numpy(dtype=bool)
    if coverage:
        var["vendor_gene_coverage"] = pd.array(
            [coverage.get(gene_id) for gene_id in var["gene_ids"]], dtype="Int64"
        )
    table = ad.AnnData(
        X=matrix[:, keep].astype(np.float32),
        obs=obs,
        var=var,
    )
    table.obsm["spatial"] = summary[["cell_centroid_x", "cell_centroid_y"]].to_numpy(
        np.float64
    )
    for name, values in obsm.items():
        table.obsm[f"{name}_vendor"] = np.asarray(values)
    return table


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AteraTile:
    """One transcript tile of ``transcripts.zarr.zip/grid``.

    Attributes:
        key: Group name, e.g. ``"2,1"``.
        x_index: ``grid_x_loc`` of the key.
        y_index: ``grid_y_loc`` of the key.
        n_rows: Transcript count (``grid_number_objects``).
        global_offset: Vendor ordinal of the tile's first row: the row counts
            of the tiles before it in ``grid_keys`` attribute order.
        grid_size_um: Tile edge length in microns.
    """

    key: str
    x_index: int
    y_index: int
    n_rows: int
    global_offset: int
    grid_size_um: float

    @property
    def bounds_um(self) -> tuple[float, float, float, float]:
        """Nominal ``(x0, x1, y0, y1)``; edge rows may fall slightly outside it."""
        size = self.grid_size_um
        return (
            self.x_index * size,
            (self.x_index + 1) * size,
            self.y_index * size,
            (self.y_index + 1) * size,
        )


@dataclass(frozen=True)
class PartitionSpec:
    """Rows of one transcripts partition; a partition never crosses a tile.

    Attributes:
        tile_key: Tile group name.
        row_start: First tile row (inclusive).
        row_stop: Last tile row (exclusive).
        global_offset: Vendor ordinal of the tile's first row.
        block_bounds_um: For ``tile_blocks`` partitions, the ``(x0, x1, y0, y1)``
            block whose rows the partition keeps (outer edges are infinite);
            ``None`` keeps every row of the range.
    """

    tile_key: str
    row_start: int
    row_stop: int
    global_offset: int
    block_bounds_um: tuple[float, float, float, float] | None = None


def read_atera_tile_index(bundle: AteraBundle) -> list[AteraTile]:
    """List the transcript tiles in ``grid_keys`` attribute order.

    Keys are parsed through ``grid_key_names``; ``grid_number_objects`` is
    index-aligned with ``grid_keys`` and checked against each tile's
    ``location`` shape (metadata only).

    Args:
        bundle: Resolved bundle.

    Returns:
        The tiles with their global ordinal offsets.

    Raises:
        AteraFormatError: If the grid attributes are inconsistent.
    """
    with open_zip_group(bundle.transcripts_zip) as root:
        grid = root["grid"]
        attrs = grid.attrs.asdict()
        key_names = list(attrs.get("grid_key_names", []))
        keys = [str(key) for key in attrs.get("grid_keys", [])]
        counts = [int(n) for n in attrs.get("grid_number_objects", [])]
        if len(keys) != len(counts) or len(set(keys)) != len(keys):
            raise AteraFormatError("grid_keys and grid_number_objects do not align")
        for name in ("grid_x_loc", "grid_y_loc"):
            if name not in key_names:
                raise AteraFormatError(f"grid_key_names {key_names} lacks {name}")
        grid_size = float(attrs["grid_size"])
        tiles = []
        offset = 0
        for key, n_rows in zip(keys, counts, strict=True):
            parts = key.split(",")
            if len(parts) != len(key_names):
                raise AteraFormatError(f"grid key {key!r} does not match {key_names}")
            values = dict(zip(key_names, (int(part) for part in parts), strict=True))
            if key not in grid:
                raise AteraFormatError(f"grid key {key!r} has no tile group")
            stored = int(grid[key]["location"].shape[0])
            if stored != n_rows:
                raise AteraFormatError(
                    f"tile {key}: grid_number_objects {n_rows} != "
                    f"{stored} location rows"
                )
            tiles.append(
                AteraTile(
                    key=key,
                    x_index=values["grid_x_loc"],
                    y_index=values["grid_y_loc"],
                    n_rows=n_rows,
                    global_offset=offset,
                    grid_size_um=grid_size,
                )
            )
            offset += n_rows
    return tiles


def plan_transcript_partitions(
    tiles: Sequence[AteraTile],
    *,
    partition_rows: int,
    spatial_order: SpatialOrder = "row_range",
    block_um: float = 500.0,
    min_block_rows: int = 1_048_576,
    tile_allowlist: Sequence[str] | None = None,
) -> list[PartitionSpec]:
    """Split the tiles into partitions, in ``grid_keys`` order.

    ``row_range`` cuts each tile into consecutive row ranges of
    ``partition_rows`` (gene-sorted within a tile). ``tile_blocks`` makes one
    partition per ``block_um`` square of a tile (spatially coherent; each block
    decodes its whole tile, so the reader bounds concurrent decodes); a tile
    whose average block would hold fewer than ``min_block_rows`` rows stays one
    partition. Empty tiles get no partition.

    Args:
        tiles: Tile index from :func:`read_atera_tile_index`.
        partition_rows: Rows per ``row_range`` partition; multiples of
            1,048,576 align with the vendor chunks.
        spatial_order: ``"row_range"`` or ``"tile_blocks"``.
        block_um: Block edge for ``tile_blocks``.
        min_block_rows: Smallest average block before a tile is not split.
        tile_allowlist: Only these tile keys (pilot and debugging).

    Returns:
        The partition specs.

    Raises:
        ValueError: If an argument is invalid or an allow-listed key is unknown.
    """
    if partition_rows <= 0:
        raise ValueError(f"partition_rows must be positive, got {partition_rows}")
    if spatial_order not in ("row_range", "tile_blocks"):
        raise ValueError(f"unknown spatial_order {spatial_order!r}")
    if block_um <= 0:
        raise ValueError(f"block_um must be positive, got {block_um}")
    if tile_allowlist is not None:
        unknown = sorted(set(tile_allowlist) - {tile.key for tile in tiles})
        if unknown:
            raise ValueError(f"tile_allowlist names unknown tiles: {unknown}")
    selected = set(tile_allowlist) if tile_allowlist is not None else None
    specs: list[PartitionSpec] = []
    for tile in tiles:
        if tile.n_rows == 0 or (selected is not None and tile.key not in selected):
            continue
        if spatial_order == "tile_blocks":
            blocks_per_axis = max(1, math.ceil(tile.grid_size_um / block_um))
            n_blocks = blocks_per_axis**2
            if n_blocks > 1 and tile.n_rows / n_blocks >= min_block_rows:
                specs.extend(_block_specs(tile, blocks_per_axis, block_um))
                continue
            specs.append(PartitionSpec(tile.key, 0, tile.n_rows, tile.global_offset))
            continue
        for start in range(0, tile.n_rows, partition_rows):
            stop = min(tile.n_rows, start + partition_rows)
            specs.append(PartitionSpec(tile.key, start, stop, tile.global_offset))
    return specs


def _block_specs(
    tile: AteraTile, blocks_per_axis: int, block_um: float
) -> list[PartitionSpec]:
    x0, _, y0, _ = tile.bounds_um

    def edges(origin: float) -> list[float]:
        inner = [origin + k * block_um for k in range(1, blocks_per_axis)]
        return [-math.inf, *inner, math.inf]

    x_edges, y_edges = edges(x0), edges(y0)
    return [
        PartitionSpec(
            tile.key,
            0,
            tile.n_rows,
            tile.global_offset,
            block_bounds_um=(x_edges[i], x_edges[i + 1], y_edges[j], y_edges[j + 1]),
        )
        for j in range(blocks_per_axis)
        for i in range(blocks_per_axis)
    ]


@dataclass(frozen=True)
class CellKeyLookup:
    """Map 10x ``(prefix, suffix)`` cell IDs to canonical ``instance_id``.

    Attributes:
        sorted_keys: Packed cell IDs, sorted.
        order: Cell row of each sorted key.
    """

    sorted_keys: np.ndarray
    order: np.ndarray

    @classmethod
    def from_cell_ids(cls, prefix: np.ndarray, suffix: np.ndarray) -> CellKeyLookup:
        """Build the lookup from the cell ID columns of ``cells.zarr.zip``.

        Args:
            prefix: Cell ID prefix per cell row.
            suffix: Dataset suffix per cell row.

        Returns:
            The lookup.
        """
        keys = pack_cell_ids(prefix, suffix)
        order = np.argsort(keys, kind="stable")
        return cls(sorted_keys=keys[order], order=order.astype(np.int64))

    def instance_ids(
        self, prefix: np.ndarray, suffix: np.ndarray
    ) -> pd.arrays.IntegerArray:
        """Return ``cell row + 1`` per transcript, NA where unassigned.

        Args:
            prefix: Transcript cell ID prefixes.
            suffix: Transcript dataset suffixes.

        Returns:
            Nullable ``UInt64`` array.

        Raises:
            AteraFormatError: If only half of a pair is the unassigned
                sentinel, or a cell ID is not in ``cells.zarr.zip``.
        """
        prefix = np.asarray(prefix, dtype=np.uint32)
        suffix = np.asarray(suffix, dtype=np.uint32)
        prefix_unassigned = prefix == UNASSIGNED_CELL_ID_PART
        unassigned = prefix_unassigned & (suffix == UNASSIGNED_CELL_ID_PART)
        if not np.array_equal(
            unassigned, prefix_unassigned | (suffix == UNASSIGNED_CELL_ID_PART)
        ):
            raise AteraFormatError(
                "transcript cell_id has a half-unassigned (prefix, suffix)"
            )
        values = np.zeros(len(prefix), dtype=np.uint64)
        assigned = np.flatnonzero(~unassigned)
        if assigned.size == 0:
            return pd.arrays.IntegerArray(values, unassigned.copy())
        if len(self.sorted_keys) == 0:
            raise AteraFormatError(
                f"{assigned.size} transcripts are assigned, "
                "but cells.zarr.zip has no cells"
            )
        query = pack_cell_ids(prefix[assigned], suffix[assigned])
        position = np.minimum(
            np.searchsorted(self.sorted_keys, query), len(self.sorted_keys) - 1
        )
        found = self.sorted_keys[position] == query
        if not bool(found.all()):
            raise AteraFormatError(
                f"{int((~found).sum())} transcript cell IDs are not in cells.zarr.zip"
            )
        values[assigned] = self.order[position].astype(np.uint64) + np.uint64(1)
        return pd.arrays.IntegerArray(values, unassigned.copy())


@dataclass(frozen=True)
class PartitionStats:
    """Pre- and post-filter statistics of one transcripts partition.

    Attributes:
        tile_key: Tile of the partition.
        n_rows_read: Rows read (before the QV filter).
        n_rows_kept: Rows kept by the QV filter.
        qv_histogram: Rows read per quarter-QV step from 0 to 40.
        gene_counts_all_qv: Rows read per feature.
        gene_counts_kept: Rows kept per feature.
        bounds_um: ``(x_min, x_max, y_min, y_max, z_min, z_max)`` of rows read.
        density: Rows kept per ``DENSITY_BIN_UM`` bin (``[ny, nx]``).
    """

    tile_key: str
    n_rows_read: int
    n_rows_kept: int
    qv_histogram: np.ndarray
    gene_counts_all_qv: np.ndarray
    gene_counts_kept: np.ndarray
    bounds_um: tuple[float, ...]
    density: np.ndarray


@dataclass(frozen=True)
class TranscriptSummary:
    """Transcript statistics gathered in the same pass as the read.

    Attributes:
        min_qv: QV threshold of the filter (``None`` keeps every row).
        n_rows_read: Rows read over all partitions.
        n_rows_kept: Rows kept over all partitions.
        rows_read_per_tile: Rows read per tile.
        rows_kept_per_tile: Rows kept per tile.
        qv_histogram_per_tile: Rows read per quarter-QV step (0-40) per tile.
        gene_counts_all_qv: Rows read per feature (all QV); equals the vendor
            ``grid.gene_counts`` when every tile is read.
        gene_counts_kept: Rows kept per feature.
        bounds_um: Coordinate ranges of the rows read.
        density: Rows kept per ``density_bin_um`` square, ``[ny, nx]``.
        density_origin_um: ``(x, y)`` of the density grid's first bin.
        density_bin_um: Bin edge length.
    """

    min_qv: float | None
    n_rows_read: int
    n_rows_kept: int
    rows_read_per_tile: dict[str, int]
    rows_kept_per_tile: dict[str, int]
    qv_histogram_per_tile: dict[str, np.ndarray]
    gene_counts_all_qv: np.ndarray
    gene_counts_kept: np.ndarray
    bounds_um: dict[str, float]
    density: np.ndarray
    density_origin_um: tuple[float, float]
    density_bin_um: float

    def to_attrs(self) -> dict[str, Any]:
        """Return the summary as JSON-serialisable attrs."""
        return {
            "min_qv": self.min_qv,
            "n_rows_read": self.n_rows_read,
            "n_points_written": self.n_rows_kept,
            "rows_read_per_tile": dict(self.rows_read_per_tile),
            "rows_kept_per_tile": dict(self.rows_kept_per_tile),
            "qv_histogram_step": QV_HISTOGRAM_STEP,
            "qv_histogram_per_tile": {
                key: values.tolist()
                for key, values in self.qv_histogram_per_tile.items()
            },
            "gene_counts_all_qv": self.gene_counts_all_qv.tolist(),
            "gene_counts_kept": self.gene_counts_kept.tolist(),
            "bounds_um": dict(self.bounds_um),
            "density_bin_um": self.density_bin_um,
            "density_origin_um": list(self.density_origin_um),
            "density": self.density.tolist(),
        }


@dataclass(frozen=True)
class _DensityGrid:
    origin_x: float
    origin_y: float
    n_x: int
    n_y: int
    bin_um: float

    @classmethod
    def for_tiles(cls, tiles: Sequence[AteraTile], bin_um: float) -> _DensityGrid:
        if not tiles:
            return cls(0.0, 0.0, 1, 1, bin_um)
        size = tiles[0].grid_size_um
        x0 = min(tile.x_index for tile in tiles) * size
        y0 = min(tile.y_index for tile in tiles) * size
        x1 = (max(tile.x_index for tile in tiles) + 1) * size
        y1 = (max(tile.y_index for tile in tiles) + 1) * size
        return cls(
            x0,
            y0,
            max(1, math.ceil((x1 - x0) / bin_um)),
            max(1, math.ceil((y1 - y0) / bin_um)),
            bin_um,
        )

    def counts(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = np.clip(
            np.floor((x - self.origin_x) / self.bin_um), 0, self.n_x - 1
        ).astype(np.int64)
        iy = np.clip(
            np.floor((y - self.origin_y) / self.bin_um), 0, self.n_y - 1
        ).astype(np.int64)
        flat = np.bincount(iy * self.n_x + ix, minlength=self.n_x * self.n_y)
        return flat.reshape(self.n_y, self.n_x)


@dataclass
class _TranscriptContext:
    """Everything a partition read needs besides its spec.

    ``tile_semaphore`` bounds concurrent whole-tile decodes (``tile_blocks``);
    ``token`` names the read in dask task keys.
    """

    zip_path: Path
    gene_dtype: pd.CategoricalDtype
    tile_dtype: pd.CategoricalDtype
    is_gene: np.ndarray
    cell_lookup: CellKeyLookup
    min_qv: float | None
    emit_codeword_index: bool
    density_grid: _DensityGrid
    tile_semaphore: threading.BoundedSemaphore | None = None
    token: str = field(default="")


def _read_rows(tile: Any, name: str, rows: slice) -> np.ndarray:
    """Read one row range of one tile array (the only tile-array access)."""
    return np.asarray(tile[name][rows])


def _gene_codes(
    gene_offset: np.ndarray, start: int, stop: int, n_rows: int
) -> np.ndarray:
    """Gene code of rows ``[start, stop)`` from end-exclusive ``gene_offset``."""
    starts = gene_offset[:, 0].astype(np.int64)
    ends = gene_offset[:, 1].astype(np.int64)
    contiguous = (
        len(starts) > 0
        and starts[0] == 0
        and ends[-1] == n_rows
        and bool(np.all(starts[1:] == ends[:-1]))
        and bool(np.all(ends >= starts))
    )
    if not contiguous:
        raise AteraFormatError("gene_offset is not a contiguous [start, end) row cover")
    counts = np.clip(ends, start, stop) - np.clip(starts, start, stop)
    return np.repeat(np.arange(len(starts), dtype=np.int32), counts)


def _empty_frame(context: _TranscriptContext) -> pd.DataFrame:
    empty_codes = np.zeros(0, dtype=np.int32)
    columns: dict[str, Any] = {
        "x": np.zeros(0, dtype=np.float32),
        "y": np.zeros(0, dtype=np.float32),
        "z": np.zeros(0, dtype=np.float32),
        "feature_name": pd.Categorical.from_codes(
            empty_codes, dtype=context.gene_dtype
        ),
        "is_gene": np.zeros(0, dtype=bool),
        "qv": np.zeros(0, dtype=np.float32),
        "overlaps_nucleus": np.zeros(0, dtype=bool),
        "original_assignment": pd.array(np.zeros(0, dtype=np.uint64), dtype="UInt64"),
        "transcript_id": np.zeros(0, dtype=np.uint64),
        "source_tile": pd.Categorical.from_codes(empty_codes, dtype=context.tile_dtype),
    }
    if context.emit_codeword_index:
        columns[CODEWORD_INDEX_COLUMN] = np.zeros(0, dtype=np.uint32)
    return pd.DataFrame(columns)


def _frame_from_rows(
    context: _TranscriptContext,
    *,
    tile_key: str,
    location: np.ndarray,
    cell_id: np.ndarray,
    quality: np.ndarray,
    overlaps: np.ndarray,
    codes: np.ndarray,
    ordinals: np.ndarray,
    codewords: np.ndarray | None,
) -> pd.DataFrame:
    n_rows = len(codes)
    tile_code = context.tile_dtype.categories.get_loc(tile_key)
    columns: dict[str, Any] = {
        "x": location[:, 0].astype(np.float32, copy=False),
        "y": location[:, 1].astype(np.float32, copy=False),
        "z": location[:, 2].astype(np.float32, copy=False),
        "feature_name": pd.Categorical.from_codes(codes, dtype=context.gene_dtype),
        "is_gene": context.is_gene[codes],
        "qv": quality.astype(np.float32),
        "overlaps_nucleus": overlaps.astype(bool),
        "original_assignment": context.cell_lookup.instance_ids(
            cell_id[:, 0], cell_id[:, 1]
        ),
        "transcript_id": ordinals.astype(np.uint64) + np.uint64(1),
        "source_tile": pd.Categorical.from_codes(
            np.full(n_rows, tile_code, dtype=np.int32), dtype=context.tile_dtype
        ),
    }
    if context.emit_codeword_index and codewords is not None:
        columns[CODEWORD_INDEX_COLUMN] = codewords.astype(np.uint32)
    return pd.DataFrame(columns)


def _read_partition_with_stats(
    context: _TranscriptContext, spec: PartitionSpec
) -> tuple[pd.DataFrame, PartitionStats]:
    rows = slice(spec.row_start, spec.row_stop)
    semaphore = context.tile_semaphore
    if semaphore is not None:
        semaphore.acquire()
    try:
        with open_zip_group(context.zip_path) as root:
            tile = root["grid"][spec.tile_key]
            n_tile_rows = int(tile["location"].shape[0])
            location: np.ndarray = _read_rows(tile, "location", rows).reshape(-1, 3)
            cell_id: np.ndarray = _read_rows(tile, "cell_id", rows).reshape(-1, 2)
            quality: np.ndarray = _read_rows(tile, "quality_score", rows).reshape(-1)
            overlaps: np.ndarray = _read_rows(tile, "overlaps_nucleus", rows).ravel()
            codewords: np.ndarray | None = (
                _read_rows(tile, "codeword_identity", rows).reshape(-1)
                if context.emit_codeword_index
                else None
            )
            gene_offset = np.asarray(tile["gene_offset"][:])
    finally:
        if semaphore is not None:
            semaphore.release()
    codes: np.ndarray = _gene_codes(
        gene_offset, spec.row_start, spec.row_stop, n_tile_rows
    )
    ordinals: np.ndarray = spec.global_offset + np.arange(
        spec.row_start, spec.row_stop, dtype=np.int64
    )
    if spec.block_bounds_um is not None:
        x0, x1, y0, y1 = spec.block_bounds_um
        x, y = location[:, 0], location[:, 1]
        selected = np.flatnonzero((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
        location, cell_id, quality, overlaps = (
            location[selected],
            cell_id[selected],
            quality[selected],
            overlaps[selected],
        )
        codes, ordinals = codes[selected], ordinals[selected]
        codewords = codewords[selected] if codewords is not None else None

    n_genes = len(context.gene_dtype.categories)
    quality64 = quality.astype(np.float64)
    keep = (
        np.flatnonzero(quality64 >= context.min_qv)
        if context.min_qv is not None
        else np.arange(len(codes))
    )
    qv_bins = np.clip(np.rint(quality64 / QV_HISTOGRAM_STEP), 0, QV_HISTOGRAM_BINS - 1)
    bounds = (
        tuple(
            float(value)
            for axis in range(3)
            for value in (location[:, axis].min(), location[:, axis].max())
        )
        if len(location)
        else ()
    )
    location, cell_id, overlaps = location[keep], cell_id[keep], overlaps[keep]
    kept_codes, quality, ordinals = codes[keep], quality[keep], ordinals[keep]
    codewords = codewords[keep] if codewords is not None else None
    stats = PartitionStats(
        tile_key=spec.tile_key,
        n_rows_read=len(codes),
        n_rows_kept=len(kept_codes),
        qv_histogram=np.bincount(qv_bins.astype(np.int64), minlength=QV_HISTOGRAM_BINS),
        gene_counts_all_qv=np.bincount(codes, minlength=n_genes),
        gene_counts_kept=np.bincount(kept_codes, minlength=n_genes),
        bounds_um=bounds,
        density=context.density_grid.counts(
            location[:, 0].astype(np.float64), location[:, 1].astype(np.float64)
        ),
    )
    frame = _frame_from_rows(
        context,
        tile_key=spec.tile_key,
        location=location,
        cell_id=cell_id,
        quality=quality,
        overlaps=overlaps,
        codes=kept_codes,
        ordinals=ordinals,
        codewords=codewords,
    )
    return frame, stats


class TranscriptAggregates:
    """Transcript statistics recorded by the partition reads themselves.

    Every partition read stores its :class:`PartitionStats` here under its
    partition number, so whatever computes the points (a parquet write inside
    spatialdata, or a plain ``compute``) gathers the statistics in the same
    pass. Recording is idempotent: a re-executed task overwrites its own entry.
    """

    def __init__(
        self,
        n_partitions: int,
        *,
        tile_keys: Sequence[str],
        n_genes: int,
        min_qv: float | None,
        density_grid: _DensityGrid,
    ) -> None:
        """Prepare an empty record for ``n_partitions`` partitions.

        Args:
            n_partitions: Number of partitions that must be recorded.
            tile_keys: Every tile key, in ``grid_keys`` order.
            n_genes: Number of features.
            min_qv: QV threshold of the reads.
            density_grid: Grid of the kept-row density map.
        """
        self.n_partitions = n_partitions
        self._tile_keys = tuple(tile_keys)
        self._n_genes = n_genes
        self._min_qv = min_qv
        self._density_grid = density_grid
        self._stats: dict[int, PartitionStats] = {}
        self._lock = threading.Lock()
        # Distinguishes the graphs of two reads with the same parameters, so
        # dask never merges their tasks and fills only one record.
        self.token = uuid.uuid4().hex

    def record(self, index: int, stats: PartitionStats) -> None:
        """Store the statistics of partition ``index``."""
        with self._lock:
            self._stats[index] = stats

    @property
    def n_recorded(self) -> int:
        """Number of partitions recorded so far."""
        with self._lock:
            return len(self._stats)

    def summary(self) -> TranscriptSummary:
        """Combine the recorded partitions.

        Returns:
            The statistics over every partition.

        Raises:
            RuntimeError: If some partitions have not been read yet (compute
                or write the points first).
        """
        with self._lock:
            stats = [self._stats[i] for i in sorted(self._stats)]
        if len(stats) != self.n_partitions:
            raise RuntimeError(
                f"only {len(stats)} of {self.n_partitions} transcript partitions "
                "were read; compute or write the points before summarising them"
            )
        grid = self._density_grid
        read_per_tile = dict.fromkeys(self._tile_keys, 0)
        kept_per_tile = dict.fromkeys(self._tile_keys, 0)
        histograms = {
            key: np.zeros(QV_HISTOGRAM_BINS, dtype=np.int64) for key in self._tile_keys
        }
        all_qv = np.zeros(self._n_genes, dtype=np.int64)
        kept = np.zeros(self._n_genes, dtype=np.int64)
        density = np.zeros((grid.n_y, grid.n_x), dtype=np.int64)
        bounds = np.array([np.inf, -np.inf] * 3)
        for item in stats:
            read_per_tile[item.tile_key] += item.n_rows_read
            kept_per_tile[item.tile_key] += item.n_rows_kept
            histograms[item.tile_key] += item.qv_histogram
            all_qv += item.gene_counts_all_qv
            kept += item.gene_counts_kept
            density += item.density
            if item.bounds_um:
                values = np.asarray(item.bounds_um)
                bounds[0::2] = np.minimum(bounds[0::2], values[0::2])
                bounds[1::2] = np.maximum(bounds[1::2], values[1::2])
        names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        return TranscriptSummary(
            min_qv=self._min_qv,
            n_rows_read=int(sum(read_per_tile.values())),
            n_rows_kept=int(sum(kept_per_tile.values())),
            rows_read_per_tile=read_per_tile,
            rows_kept_per_tile=kept_per_tile,
            qv_histogram_per_tile=histograms,
            gene_counts_all_qv=all_qv,
            gene_counts_kept=kept,
            bounds_um={
                name: float(value)
                for name, value in zip(names, bounds, strict=True)
                if np.isfinite(value)
            },
            density=density,
            density_origin_um=(grid.origin_x, grid.origin_y),
            density_bin_um=grid.bin_um,
        )


class _PartitionReader:
    """``from_map`` function: read one partition and record its statistics."""

    def __init__(
        self, context: _TranscriptContext, aggregates: TranscriptAggregates
    ) -> None:
        self.context = context
        self.aggregates = aggregates

    def __call__(self, index: int, spec: PartitionSpec) -> pd.DataFrame:
        frame, stats = _read_partition_with_stats(self.context, spec)
        self.aggregates.record(index, stats)
        return frame

    def __dask_tokenize__(self) -> tuple[str, str]:
        return (self.context.token, self.aggregates.token)


def _category_column(group: Any, array: str, column: str) -> np.ndarray:
    """Select one named column of a ``*_category`` table (never by position)."""
    names = list(group[array].attrs.get("column_names", []))
    if column not in names:
        raise AteraFormatError(f"{array} has no {column!r} column (columns: {names})")
    return np.asarray(group[array][:, names.index(column)], dtype=bool)


def _transcript_context(
    bundle: AteraBundle,
    tiles: Sequence[AteraTile],
    *,
    min_qv: float | None,
    emit_codeword_index: bool,
    tile_semaphore: threading.BoundedSemaphore | None = None,
) -> _TranscriptContext:
    with open_zip_group(bundle.transcripts_zip) as root:
        gene_names = [str(name) for name in root.attrs["gene_names"]]
        is_gene = _category_column(root, "gene_category", "is_gene")
    if len(set(gene_names)) != len(gene_names):
        raise AteraFormatError("transcripts gene_names are not unique")
    if len(is_gene) != len(gene_names):
        raise AteraFormatError("gene_category rows differ from gene_names")
    with open_zip_group(bundle.cells_zip) as cells_root:
        cell_id = np.asarray(cells_root["cell_id"][:])
    context = _TranscriptContext(
        zip_path=bundle.transcripts_zip,
        gene_dtype=pd.CategoricalDtype(gene_names, ordered=False),
        tile_dtype=pd.CategoricalDtype([tile.key for tile in tiles], ordered=False),
        is_gene=is_gene,
        cell_lookup=CellKeyLookup.from_cell_ids(cell_id[:, 0], cell_id[:, 1]),
        min_qv=min_qv,
        emit_codeword_index=emit_codeword_index,
        density_grid=_DensityGrid.for_tiles(tiles, DENSITY_BIN_UM),
        tile_semaphore=tile_semaphore,
    )
    context.token = dask.base.tokenize(
        str(bundle.transcripts_zip),
        bundle.transcripts_zip.stat().st_mtime,
        min_qv,
        emit_codeword_index,
    )
    return context


def read_transcript_partition(
    bundle: AteraBundle,
    spec: PartitionSpec,
    *,
    gene_dtype: pd.CategoricalDtype,
    is_gene: np.ndarray,
    cell_lookup: CellKeyLookup,
    min_qv: float | None,
    tile_dtype: pd.CategoricalDtype,
    emit_codeword_index: bool = False,
) -> pd.DataFrame:
    """Read one transcripts partition into the points schema (plan §3.3).

    Columns: ``x``, ``y``, ``z`` (float32 µm), ``feature_name`` (known
    categorical over every feature, from ``gene_offset``), ``is_gene``,
    ``qv`` (float32), ``overlaps_nucleus``, ``original_assignment`` (nullable
    UInt64, cell row + 1), ``transcript_id`` (vendor ordinal + 1, assigned
    before the QV filter), ``source_tile`` and optionally ``codeword_index``.

    Args:
        bundle: Resolved bundle.
        spec: Rows to read.
        gene_dtype: Categorical dtype of ``feature_name``.
        is_gene: ``is_gene`` flag per feature code.
        cell_lookup: Cell ID lookup.
        min_qv: Keep rows with QV at least this (``None`` keeps all).
        tile_dtype: Categorical dtype of ``source_tile``.
        emit_codeword_index: Add ``codeword_index``.

    Returns:
        The partition frame.
    """
    context = _TranscriptContext(
        zip_path=bundle.transcripts_zip,
        gene_dtype=gene_dtype,
        tile_dtype=tile_dtype,
        is_gene=np.asarray(is_gene, dtype=bool),
        cell_lookup=cell_lookup,
        min_qv=min_qv,
        emit_codeword_index=emit_codeword_index,
        density_grid=_DensityGrid(0.0, 0.0, 1, 1, DENSITY_BIN_UM),
    )
    frame, _ = _read_partition_with_stats(context, spec)
    return frame


def read_atera_transcripts(
    bundle: AteraBundle,
    *,
    min_qv: float | None = 20.0,
    partition_rows: int = 4_194_304,
    spatial_order: SpatialOrder = "row_range",
    block_um: float = 500.0,
    min_block_rows: int = 1_048_576,
    max_tiles_in_flight: int = 4,
    emit_codeword_index: bool = False,
    tile_allowlist: Sequence[str] | None = None,
) -> tuple[dd.DataFrame, list[PartitionSpec], TranscriptAggregates]:
    """Read the transcripts lazily as one partition per :class:`PartitionSpec`.

    Each partition is one read task that also records its statistics in the
    returned :class:`TranscriptAggregates`, so writing the points once reads
    every tile slice once and gathers the statistics in the same pass. The
    frame's ``meta`` holds the full categorical dtypes, so
    ``PointsModel.parse(..., sort=True)`` triggers no computation.

    Args:
        bundle: Resolved bundle.
        min_qv: Keep rows with QV at least this (``None`` keeps all).
        partition_rows: Rows per ``row_range`` partition.
        spatial_order: ``"row_range"`` (default) or ``"tile_blocks"``.
        block_um: Block edge for ``tile_blocks``.
        min_block_rows: See :func:`plan_transcript_partitions`.
        max_tiles_in_flight: Concurrent whole-tile decodes in ``tile_blocks``.
        emit_codeword_index: Add ``codeword_index``.
        tile_allowlist: Only these tiles.

    Returns:
        ``(points, specs, aggregates)``: the dask frame, its partition specs in
        order, and the statistics record; call ``aggregates.summary()`` after
        the points have been computed or written.

    Raises:
        ValueError: If ``max_tiles_in_flight`` is not positive.
    """
    if max_tiles_in_flight <= 0:
        raise ValueError(
            f"max_tiles_in_flight must be positive, got {max_tiles_in_flight}"
        )
    tiles = read_atera_tile_index(bundle)
    specs = plan_transcript_partitions(
        tiles,
        partition_rows=partition_rows,
        spatial_order=spatial_order,
        block_um=block_um,
        min_block_rows=min_block_rows,
        tile_allowlist=tile_allowlist,
    )
    semaphore = (
        threading.BoundedSemaphore(max_tiles_in_flight)
        if spatial_order == "tile_blocks"
        else None
    )
    context = _transcript_context(
        bundle,
        tiles,
        min_qv=min_qv,
        emit_codeword_index=emit_codeword_index,
        tile_semaphore=semaphore,
    )
    aggregates = TranscriptAggregates(
        len(specs),
        tile_keys=[tile.key for tile in tiles],
        n_genes=len(context.gene_dtype.categories),
        min_qv=min_qv,
        density_grid=context.density_grid,
    )
    meta = _empty_frame(context)
    if not specs:
        return dd.from_pandas(meta, npartitions=1), specs, aggregates
    points = dd.from_map(
        _PartitionReader(context, aggregates),
        list(range(len(specs))),
        specs,
        meta=meta,
        divisions=None,
        label="atera-transcripts",
        enforce_metadata=False,
    )
    return points, specs, aggregates


def read_atera_transcripts_for_genes(
    bundle: AteraBundle, genes: Sequence[str], *, min_qv: float | None = None
) -> pd.DataFrame:
    """Read every transcript of a few genes, using ``gene_offset`` slices only.

    For QC and debugging; the build reads whole partitions.

    Args:
        bundle: Resolved bundle.
        genes: Feature names (genes or controls).
        min_qv: Keep rows with QV at least this (``None`` keeps all).

    Returns:
        A frame with the partition columns, tiles in ``grid_keys`` order.

    Raises:
        KeyError: If a name is not a feature of the bundle.
    """
    tiles = read_atera_tile_index(bundle)
    context = _transcript_context(
        bundle, tiles, min_qv=min_qv, emit_codeword_index=False
    )
    categories = context.gene_dtype.categories
    unknown = [gene for gene in genes if gene not in categories]
    if unknown:
        raise KeyError(f"unknown features: {unknown}")
    codes = sorted({int(categories.get_loc(gene)) for gene in genes})
    frames = [_empty_frame(context)]
    with open_zip_group(bundle.transcripts_zip) as root:
        for tile in tiles:
            if tile.n_rows == 0:
                continue
            group = root["grid"][tile.key]
            gene_offset = np.asarray(group["gene_offset"][:])
            for code in codes:
                start, stop = (int(v) for v in gene_offset[code])
                if stop <= start:
                    continue
                rows = slice(start, stop)
                quality = _read_rows(group, "quality_score", rows).reshape(-1)
                keep = (
                    quality.astype(np.float64) >= min_qv
                    if min_qv is not None
                    else np.ones(len(quality), dtype=bool)
                )
                frames.append(
                    _frame_from_rows(
                        context,
                        tile_key=tile.key,
                        location=_read_rows(group, "location", rows).reshape(-1, 3)[
                            keep
                        ],
                        cell_id=_read_rows(group, "cell_id", rows).reshape(-1, 2)[keep],
                        quality=quality[keep],
                        overlaps=_read_rows(group, "overlaps_nucleus", rows).reshape(
                            -1
                        )[keep],
                        codes=np.full(int(keep.sum()), code, dtype=np.int32),
                        ordinals=(tile.global_offset + np.arange(start, stop))[keep],
                        codewords=None,
                    )
                )
    return pd.concat(frames, ignore_index=True)
