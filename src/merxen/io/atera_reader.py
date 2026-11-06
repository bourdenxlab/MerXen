"""Pure readers for 10x Atera output bundles (one extracted ROI directory).

An Atera ROI holds zipped zarr v2 stores (``transcripts.zarr.zip``,
``cells.zarr.zip``, ``cell_feature_matrix.zarr.zip``), a multi-file OME-TIFF
``morphology_2d/`` and JSON/CSV sidecars. No released reader opens it, so this
module reads each part into the element MerXen's source store needs; the
builder (``merxen.io.builders.atera``) assembles them. The format and every
decision below are documented in ``docs/plans/atera-support-plan.md`` (§2, §3).

Conventions the readers rely on, verified on the public dataset:

- Every zip is opened with ``zarr.storage.ZipStore(path, mode="r")``. Lazy
  reads keep one store per thread and zip: ``ZipStore`` serialises reads
  behind a lock, and opening the large transcripts zip costs about 0.3 s.
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
- Vendor metadata is kept whole (plan §3.11): :func:`copy_atera_sidecars`
  copies the manifest, metrics, panel, reports and FOV files byte for byte
  under stable names, :func:`write_atera_embedded_metadata` keeps the metadata
  held inside the zipped zarr stores and the OME-TIFF headers verbatim, and
  :func:`atera_metadata_attrs` returns the full header for the store attrs.
  Run, slide and instrument values in it are sample names, so human-facing
  text (logs, errors, plot titles) never prints them unless through
  :func:`merxen.privacy.to_alias`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
import shutil
import threading
import uuid
import weakref
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from merxen.segmentation.mask_geometry import masks_to_labeled_polygons

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
    "centroid_column",
    "centroid_row",
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
#: QV of the per-gene totals recorded whatever ``min_qv`` is (the vendor rule).
Q20: Final = 20.0
#: Mask contour simplification in pixels. Below 0.5 px it removes only
#: collinear staircase vertices, so centroids and areas stay as unsimplified.
DEFAULT_SIMPLIFY_PX: Final = 0.25
#: Shapes per ``shapely.simplify`` call when simplifying across threads.
_SIMPLIFY_CHUNK: Final = 4096

SpatialOrder = Literal["row_range", "tile_blocks"]
#: Start methods safe after zarr or dask have started threads.
PolygonStartMethod = Literal["forkserver", "spawn"]

#: Bundle file name patterns; the groups of each make the stable name.
_STABLE_NAME_RES: Final = (
    re.compile(r"^(metrics_summary)(?:_.*)?(\.csv)$"),
    re.compile(r"^(analysis_summary)(?:_.*)?(\.html)$"),
    re.compile(r"^(?:.*_)?(he_imagealignment)(\.csv)$"),
    re.compile(r"^(?:.*_)?(keypoints)(\.csv)$"),
    re.compile(r"^(?:.*_)?(he_image)(\.ome\.tif)$"),
    # Lazy prefix, longer names first: binned_/csc_ belong to the viz stores.
    re.compile(
        r"^(?:.*?_)??(binned_transcripts|csc_cell_feature_matrix|transcripts|cells"
        r"|cell_feature_matrix)(\.zarr\.zip)$"
    ),
)
#: Vendor metadata and report files copied verbatim into the store (§3.11).
_SIDECAR_GLOBS: Final = (
    "experiment.spatial",
    "metrics_summary*.csv",
    "analysis_summary*.html",
    "qc_images.html",
    "panel_config.json",
    "morphology_fov_locations.json",
    "overview_scan_fov_locations.json",
    "overview_scan.png",
    "diffexp_genes/*.csv",
    "*he_imagealignment.csv",
    "*keypoints.csv",
)
#: Sidecar folder at the store root; it survives store rewrites.
ATERA_SIDECAR_DIR: Final = "atera"
#: Subfolder of ``atera/`` with the metadata members of each vendor zarr zip.
ZARR_METADATA_DIR: Final = "zarr_metadata"
#: Subfolder of ``atera/`` with the OME-XML header of each morphology TIFF.
OME_XML_DIR: Final = "ome_xml"
#: Zarr metadata member names (v2 and v3) kept from every vendor zip.
ZARR_METADATA_MEMBERS: Final = frozenset(
    {".zattrs", ".zgroup", ".zarray", ".zmetadata", "zarr.json"}
)
#: Vendor zarr root attributes larger than this (JSON bytes) stay in their
#: sidecar only, so the store's root attrs stay small enough to read on open.
ROOT_ATTR_INLINE_MAX_BYTES: Final = 32 << 10
#: Vendor visualisation stores that are not ingested; only their metadata is
#: kept, as ``(explorer_files key, stable name)``.
_VIZ_STORES: Final = (
    ("transcripts_viz_zarr_filepath", "binned_transcripts.zarr.zip"),
    ("cell_feature_viz_zarr_filepath", "csc_cell_feature_matrix.zarr.zip"),
)
#: Prefix of the differential-expression columns of the matrix ``var``.
DE_COLUMN_PREFIX: Final = "de_"
#: ``cell_summary`` columns the table maps to named obs columns or ``obsm``;
#: every other column is copied as ``vendor_<name>``.
_MAPPED_SUMMARY_COLUMNS: Final = frozenset(
    {"cell_centroid_x", "cell_centroid_y", "cell_area", "nucleus_area", "nucleus_count"}
)
#: Stable sidecar name of a ``spec_override`` manifest (the bundle's own
#: ``experiment.spatial`` keeps its name).
EXPERIMENT_OVERRIDE_SIDECAR: Final = "experiment_override.spatial"


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
        experiment: Parsed ``experiment.spatial`` (or the metadata override),
            complete; :func:`atera_metadata_attrs` stores all of it.
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
        sidecars: Every vendor metadata and report file, copied verbatim into
            the store by :func:`copy_atera_sidecars`.
        viz_stores: The vendor visualisation zips (binned transcripts, CSC
            matrix) found in the bundle; not ingested, but their metadata is
            kept by :func:`write_atera_embedded_metadata`.
        morphology_2d_annotations: The OME ``MapAnnotation`` key-value pairs
            of the morphology image (cycle, channel, purpose, observed and
            suggested intensity ranges, pixel value offset), in order.
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
    sidecars: tuple[AteraSidecar, ...]
    viz_stores: tuple[Path, ...] = ()
    morphology_2d_annotations: tuple[Mapping[str, Any], ...] = ()


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


def _close_stores(stores: list[Any], lock: threading.Lock) -> None:
    with lock:
        for store in stores:
            store.close()
        stores.clear()


class _ThreadZipGroups:
    """Root groups of zipped zarr stores, opened once per thread and zip.

    Opening a ``ZipStore`` parses the zip's central directory in Python
    (about 0.3 s, GIL-bound, for the 42,797-member transcripts zip), so read
    tasks reuse one store per thread instead of opening one per task. A store
    is never shared between threads, because it serialises reads behind a
    lock. The stores close when this object is garbage collected (or at
    interpreter exit); a pickled copy starts empty.
    """

    def __init__(self) -> None:
        self._start_empty()

    def _start_empty(self) -> None:
        self._local = threading.local()
        self._stores: list[Any] = []
        self._lock = threading.Lock()
        weakref.finalize(self, _close_stores, self._stores, self._lock)

    def group(self, path: Path) -> Any:
        """Return this thread's open root group of ``path``.

        Args:
            path: Zipped zarr store.

        Returns:
            The read-only root group.
        """
        groups: dict[Path, Any] | None = getattr(self._local, "groups", None)
        if groups is None:
            groups = {}
            self._local.groups = groups
        group = groups.get(path)
        if group is None:
            store = zarr.storage.ZipStore(path, mode="r")
            with self._lock:
                self._stores.append(store)
            group = zarr.open_group(store, mode="r")
            groups[path] = group
        return group

    def __getstate__(self) -> dict[str, Any]:
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._start_empty()


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
            "the Atera input is not a directory; extract the outs bundle first"
        )
    spec_path = (
        Path(spec_override)
        if spec_override is not None
        else root / "experiment.spatial"
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
    files, names, shape_yx, ome_pixel_size, annotations = _morphology_2d_channels(
        morphology_dir
    )
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
        sidecars=_bundle_sidecars(
            root, Path(spec_override) if spec_override is not None else None
        ),
        viz_stores=tuple(
            path
            for key, name in _VIZ_STORES
            if (path := _optional_member(root, experiment, key, name)) is not None
        ),
        morphology_2d_annotations=annotations,
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
            raise FileNotFoundError(f"the {default_name} override does not exist")
        return path
    explorer = experiment.get("explorer_files") or {}
    named = _relative_member(root, explorer.get(explorer_key))
    if named is not None:
        return named
    plain = root / default_name
    if plain.is_file():
        return plain
    # stable_sidecar_name tells <dataset>_binned_transcripts apart from
    # <dataset>_transcripts (and csc_ from the main matrix).
    prefixed = [
        path
        for path in sorted(root.glob(f"*_{default_name}"))
        if stable_sidecar_name(path.name) == default_name
    ]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        raise AteraFormatError(
            f"{len(prefixed)} *_{default_name} files in the Atera input; keep one"
        )
    raise FileNotFoundError(f"{default_name} not found in the Atera input")


def _optional_member(
    root: Path, experiment: Mapping[str, Any], explorer_key: str, default_name: str
) -> Path | None:
    """Like :func:`_resolve_member`, but ``None`` when the store is absent."""
    try:
        return _resolve_member(root, None, experiment, explorer_key, default_name)
    except FileNotFoundError:
        return None


def _check_units(value: object, what: str) -> None:
    if str(value).lower() not in MICRON_UNITS:
        raise AteraFormatError(f"{what} spatial_units {value!r} is not microns")


def _check_rna_dataset(path: Path) -> None:
    with open_zip_group(path) as group:
        attrs = group.attrs.asdict()
    if attrs.get("name") != RNA_DATASET_NAME:
        raise AteraFormatError(
            f"{stable_sidecar_name(path.name)} is not an {RNA_DATASET_NAME}"
        )
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
        raise AteraFormatError(
            f"{stable_sidecar_name(path.name)} is not a {CELLS_DATASET_NAME}"
        )
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
        raise AteraFormatError(
            f"{stable_sidecar_name(path.name)} is not an AnnData zarr store"
        )


# --------------------------------------------------------------------------- #
# Vendor metadata: verbatim sidecars and the full header
# --------------------------------------------------------------------------- #


def stable_sidecar_name(name: str) -> str:
    """Return the stable name of a bundle file, without its run or dataset part.

    ``metrics_summary_<run>.csv`` becomes ``metrics_summary.csv``,
    ``<dataset>_he_imagealignment.csv`` becomes ``he_imagealignment.csv`` and so
    on, so code finds a sidecar under one name whatever the run; names without
    such a part are returned unchanged. The original name is recorded with
    each copied sidecar.

    Args:
        name: A file name (no directory).

    Returns:
        The stable name.
    """
    for pattern in _STABLE_NAME_RES:
        match = pattern.match(name)
        if match:
            return f"{match.group(1)}{match.group(2)}"
    return name


@dataclass(frozen=True)
class AteraSidecar:
    """One vendor file to copy byte for byte into the store.

    Attributes:
        source: The file in the bundle (or the ``spec_override`` manifest).
        stored_path: POSIX path under the store's ``atera/`` folder, a stable
            name (:func:`stable_sidecar_name`) that keeps any subfolder.
        original_name: POSIX path of the file relative to the bundle (its
            file name for a ``spec_override`` outside the bundle).
    """

    source: Path
    stored_path: str
    original_name: str


def _bundle_sidecars(
    root: Path, spec_override: Path | None
) -> tuple[AteraSidecar, ...]:
    sidecars: dict[str, AteraSidecar] = {}

    def add(sidecar: AteraSidecar) -> None:
        if sidecar.stored_path in sidecars:
            raise AteraFormatError(
                f"several bundle files map to the sidecar {sidecar.stored_path}; "
                "keep one per ROI directory"
            )
        sidecars[sidecar.stored_path] = sidecar

    for pattern in _SIDECAR_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            stored = relative.with_name(stable_sidecar_name(path.name)).as_posix()
            add(AteraSidecar(path, stored, relative.as_posix()))
    if spec_override is not None and spec_override.resolve() != (
        (root / "experiment.spatial").resolve()
    ):
        name = spec_override.name
        add(AteraSidecar(spec_override, EXPERIMENT_OVERRIDE_SIDECAR, name))
    return tuple(sidecars.values())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_atera_sidecars(bundle: AteraBundle, store_root: Path) -> list[dict[str, Any]]:
    """Copy every vendor metadata file verbatim into ``<store_root>/atera/``.

    Nothing is redacted: ``experiment.spatial``, the metrics summary,
    ``panel_config.json`` (customer and designer included), the vendor HTML
    reports, the FOV files, ``diffexp_genes/`` and the H&E alignment are copied
    byte for byte under stable names (:func:`stable_sidecar_name`). The
    returned records keep each file's original name; store them with
    :func:`atera_metadata_attrs`.

    Args:
        bundle: The resolved bundle.
        store_root: Root of the SpatialData zarr store being written.

    Returns:
        One JSON-serialisable record per sidecar: ``path`` (relative to the
        store root), ``original_name``, ``size_bytes`` and ``sha256``.
    """
    records: list[dict[str, Any]] = []
    for sidecar in bundle.sidecars:
        target = Path(store_root, ATERA_SIDECAR_DIR, *sidecar.stored_path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sidecar.source, target)
        records.append(
            {
                "path": f"{ATERA_SIDECAR_DIR}/{sidecar.stored_path}",
                "original_name": sidecar.original_name,
                "size_bytes": target.stat().st_size,
                "sha256": _file_sha256(target),
            }
        )
    return records


#: A leading zero before another digit marks an identifier ("00123"), not a number.
_LEADING_ZERO_RE: Final = re.compile(r"^[+-]?0\d")


def _zarr_metadata_members(zip_path: Path) -> dict[str, str]:
    """Return the text of every zarr metadata member of a zip, in zip order."""
    members: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if PurePosixPath(info.filename).name not in ZARR_METADATA_MEMBERS:
                continue
            try:
                members[info.filename] = archive.read(info).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AteraFormatError(
                    f"{stable_sidecar_name(zip_path.name)} member {info.filename} "
                    "is not UTF-8 text"
                ) from exc
    return members


def _root_attrs(members: Mapping[str, str]) -> dict[str, Any]:
    if ".zattrs" in members:
        attrs = json.loads(members[".zattrs"])
    elif "zarr.json" in members:
        attrs = json.loads(members["zarr.json"]).get("attributes") or {}
    else:
        attrs = {}
    return attrs if isinstance(attrs, dict) else {}


def vendor_zarr_stores(bundle: AteraBundle) -> dict[str, Path]:
    """Return every vendor zarr zip of the bundle by its stable store name.

    Args:
        bundle: The resolved bundle.

    Returns:
        ``{"transcripts": ..., "cells": ..., "cell_feature_matrix": ...}`` plus
        ``binned_transcripts`` and ``csc_cell_feature_matrix`` when present.
    """
    paths = (
        bundle.transcripts_zip,
        bundle.cells_zip,
        bundle.cell_feature_matrix_zip,
        *bundle.viz_stores,
    )
    return {
        stable_sidecar_name(path.name).removesuffix(".zarr.zip"): path for path in paths
    }


def _write_text_sidecar(store_root: Path, relative: str, text: str) -> Path:
    target = Path(store_root, ATERA_SIDECAR_DIR, *relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))
    return target


def write_atera_embedded_metadata(
    bundle: AteraBundle, store_root: Path
) -> list[dict[str, Any]]:
    """Keep the metadata held inside the vendor stores and images verbatim.

    Two kinds of sidecar are written under ``<store_root>/atera/``:

    - ``zarr_metadata/<store>.json`` for every vendor zarr zip (the ingested
      stores and the visualisation stores): ``{"original_name", "members"}``,
      where ``members`` maps each ``.zattrs``/``.zgroup``/``.zarray``/
      ``.zmetadata``/``zarr.json`` member path to its exact text. This keeps
      the root attributes (dataset UUID, kit, codeword maps, segmentation
      methods, cell annotations, clusterings, matrix properties) and every
      array's attributes (column names and descriptions).
    - ``ome_xml/<relative path>.xml`` for every ``morphology_2d`` file and
      the 3D stack: the file's OME-XML header, exactly as stored (instrument,
      plate, channels, map annotations with intensity ranges and offsets).

    Args:
        bundle: The resolved bundle.
        store_root: Root of the SpatialData zarr store being written.

    Returns:
        One record per sidecar: ``kind`` (``zarr_metadata`` or ``ome_xml``),
        ``path`` (relative to the store root), ``original_name`` (bundle-relative
        source), ``size_bytes`` and ``sha256``; zarr records add ``store``,
        ``n_members`` and ``root_attrs`` (the parsed root attributes, for
        :func:`atera_metadata_attrs`).
    """
    records: list[dict[str, Any]] = []
    for store, zip_path in vendor_zarr_stores(bundle).items():
        members = _zarr_metadata_members(zip_path)
        original = bundle_relative_name(zip_path, bundle.root)
        text = json.dumps(
            {"original_name": original, "members": members}, ensure_ascii=False
        )
        target = _write_text_sidecar(
            store_root, f"{ZARR_METADATA_DIR}/{store}.json", text
        )
        records.append(
            {
                "kind": "zarr_metadata",
                "store": store,
                "path": f"{ATERA_SIDECAR_DIR}/{ZARR_METADATA_DIR}/{store}.json",
                "original_name": original,
                "n_members": len(members),
                "size_bytes": target.stat().st_size,
                "sha256": _file_sha256(target),
                "root_attrs": _root_attrs(members),
            }
        )
    images = [*bundle.morphology_2d_files]
    if bundle.morphology_3d_file is not None:
        images.append(bundle.morphology_3d_file)
    for path in images:
        with tifffile.TiffFile(path, _multifile=False) as tif:
            xml = tif.ome_metadata or ""
        if not xml:
            continue
        original = bundle_relative_name(path, bundle.root)
        relative = f"{OME_XML_DIR}/{original}.xml"
        target = _write_text_sidecar(store_root, relative, xml)
        records.append(
            {
                "kind": "ome_xml",
                "path": f"{ATERA_SIDECAR_DIR}/{relative}",
                "original_name": original,
                "size_bytes": target.stat().st_size,
                "sha256": _file_sha256(target),
            }
        )
    return records


def bundle_relative_name(path: Path, root: Path) -> str:
    """Return ``path`` relative to the bundle root, else its file name.

    Args:
        path: A bundle file (or an explicit override outside the bundle).
        root: The ROI directory.

    Returns:
        A POSIX path that names no host directory.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _vendor_store_attrs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Split a store's root attributes into inline values and sidecar-only keys."""
    inline: dict[str, Any] = {}
    sidecar_only: list[str] = []
    for key, value in record["root_attrs"].items():
        size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        if size > ROOT_ATTR_INLINE_MAX_BYTES:
            sidecar_only.append(key)
        else:
            inline[key] = value
    return {
        "sidecar": record["path"],
        "attrs": inline,
        "attrs_in_sidecar_only": sidecar_only,
    }


def _metric_value(text: str) -> int | float | str | None:
    value = text.strip()
    if not value:
        return None
    if _LEADING_ZERO_RE.match(value):
        return value
    for parse in (int, float):
        try:
            number = parse(value)
        except ValueError:
            continue
        # NaN and infinities are not JSON; keep their text.
        return number if math.isfinite(number) else value
    return value


def read_metrics_summary(path: Path) -> list[dict[str, int | float | str | None]]:
    """Read every column of a ``metrics_summary`` CSV as structured values.

    Args:
        path: The CSV file.

    Returns:
        One dictionary per data row, in column order: integers and finite
        floats as numbers, empty cells as ``None``, all other text verbatim.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    header = [name.strip() for name in rows[0]]
    return [
        {name: _metric_value(value) for name, value in zip(header, row, strict=False)}
        for row in rows[1:]
        if any(value.strip() for value in row)
    ]


def _panel_config_header(panel_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``panel_config.json`` without its per-target list (in the sidecar)."""
    header: dict[str, Any] = json.loads(json.dumps(panel_config))
    for payload in header.get("payloads") or []:
        if isinstance(payload, dict) and isinstance(payload.get("targets"), list):
            payload["num_targets"] = len(payload.pop("targets"))
    return header


def atera_metadata_attrs(
    bundle: AteraBundle,
    sidecars: Sequence[Mapping[str, Any]],
    embedded: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return the full vendor header for ``attrs["merxen_atera"]``.

    No field is dropped: the whole ``experiment.spatial`` (run, slide, region,
    instrument, UUIDs, versions, panels, counts), every metrics summary column,
    the ``panel_config.json`` header (all but its per-target list, which stays
    in the verbatim sidecar), the root attributes of every vendor zarr store
    (values over :data:`ROOT_ATTR_INLINE_MAX_BYTES`, such as the codeword
    maps, are named and kept in their sidecar only), the morphology map
    annotations, the bundle's store file names and the sidecar records.
    ``experiment_source`` says whether ``experiment`` is the bundle's own
    manifest or a ``spec_override``. Human-facing text that prints any of
    these values passes it through :func:`merxen.privacy.to_alias` first.

    Args:
        bundle: The resolved bundle.
        sidecars: Records from :func:`copy_atera_sidecars`.
        embedded: Records from :func:`write_atera_embedded_metadata`.

    Returns:
        A new JSON-serialisable dictionary.
    """
    stored = {sidecar.stored_path: sidecar.source for sidecar in bundle.sidecars}
    metrics = stored.get("metrics_summary.csv")
    panel = stored.get("panel_config.json")
    attrs: dict[str, Any] = {
        "experiment": json.loads(json.dumps(bundle.experiment)),
        "metrics_summary": read_metrics_summary(metrics) if metrics else [],
        "panel_config": (
            _panel_config_header(json.loads(panel.read_text(encoding="utf-8")))
            if panel
            else None
        ),
        "bundle_files": {
            "transcripts": bundle.transcripts_zip.name,
            "cells": bundle.cells_zip.name,
            "cell_feature_matrix": bundle.cell_feature_matrix_zip.name,
            "morphology_2d": [path.name for path in bundle.morphology_2d_files],
            "morphology_3d": (
                bundle.morphology_3d_file.name if bundle.morphology_3d_file else None
            ),
        },
        "sidecars": [dict(record) for record in sidecars],
        "experiment_source": (
            "override"
            if any(
                sidecar.stored_path == EXPERIMENT_OVERRIDE_SIDECAR
                for sidecar in bundle.sidecars
            )
            else "bundle"
        ),
        "morphology_2d_annotations": json.loads(
            json.dumps(list(bundle.morphology_2d_annotations))
        ),
        "embedded_metadata": [
            {key: value for key, value in record.items() if key != "root_attrs"}
            for record in embedded
        ],
        "vendor_store_attrs": {
            record["store"]: _vendor_store_attrs(record)
            for record in embedded
            if record.get("kind") == "zarr_metadata"
        },
    }
    return attrs


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
    annotations: tuple[Mapping[str, Any], ...]


def _map_annotations(
    root: ET.Element, ns: Mapping[str, str]
) -> tuple[dict[str, Any], ...]:
    """Return every OME ``MapAnnotation`` as ``{"id", "values": {key: text}}``."""
    annotations = []
    for block in _ome_find(root, "StructuredAnnotations", ns):
        for annotation in _ome_find(block, "MapAnnotation", ns):
            values = {
                str(item.get("K", "")): item.text or ""
                for value in _ome_find(annotation, "Value", ns)
                for item in _ome_find(value, "M", ns)
            }
            annotations.append({"id": annotation.get("ID"), "values": values})
    return tuple(annotations)


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
        annotations=_map_annotations(root, ns),
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
) -> tuple[
    tuple[Path, ...],
    tuple[str, ...],
    tuple[int, int],
    float | None,
    tuple[Mapping[str, Any], ...],
]:
    """Map ``morphology_2d`` files to channels from the OME ``TiffData``.

    Also returns the pixel size and the ``MapAnnotation`` entries of the first
    file's OME-XML, which every sibling repeats.
    """
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
        reference.annotations,
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

    Reads go through one ``ZipStore`` per thread (:class:`_ThreadZipGroups`),
    because a store serialises reads behind a lock.
    """

    def __init__(self, zip_path: Path, member: str) -> None:
        self.zip_path = zip_path
        self.member = member
        with open_zip_group(zip_path) as group:
            array = group[member]
            self.shape = tuple(int(n) for n in array.shape)
            self.dtype = np.dtype(array.dtype)
            self.chunks = tuple(int(n) for n in array.chunks)
        self.ndim = len(self.shape)
        self._groups = _ThreadZipGroups()

    def __getitem__(self, key: tuple[slice, ...]) -> np.ndarray:
        return np.asarray(self._groups.group(self.zip_path)[self.member][key])

    def __dask_tokenize__(self) -> tuple[str, str, float]:
        return (str(self.zip_path), self.member, self.zip_path.stat().st_mtime)


def _vendor_aligned_dask_array(
    zip_path: Path, member: str, chunks: tuple[int, ...]
) -> da.Array:
    """Read an array in whole vendor chunks, then rechunk to ``chunks``.

    Each read task covers a whole number of the vendor's chunks (the multiple
    closest to ``chunks``), so no vendor chunk is decoded twice; the rechunk
    only moves decoded blocks.
    """
    source = _LazyZipArray(zip_path, member)
    read_chunks = tuple(
        min(n, vendor * max(1, round(wanted / vendor)))
        for wanted, vendor, n in zip(chunks, source.chunks, source.shape, strict=True)
    )
    array: da.Array = da.from_array(
        source,
        chunks=read_chunks,
        lock=False,
        asarray=True,
        fancy=False,
        name=f"atera-{member.replace('/', '-')}-"
        + dask.base.tokenize(source, read_chunks),
    )
    rechunked: da.Array = array.rechunk(
        tuple(min(c, n) for c, n in zip(chunks, source.shape, strict=True))
    )
    return rechunked


def read_atera_labels(
    bundle: AteraBundle, *, chunks: tuple[int, int] = (4096, 4096)
) -> dict[str, da.Array]:
    """Read the vendor masks as lazy label arrays.

    The reads follow the vendor chunk grid (``(665, 1918)`` on the public
    dataset) and are rechunked once, which halves the decode work against
    reading square chunks straight from the zip.

    Args:
        bundle: Resolved bundle.
        chunks: ``(y, x)`` chunk size of the result; aligned square chunks
            halve cleanly when the builder writes a label pyramid.

    Returns:
        ``{"cell_labels": masks/1, "nucleus_labels": masks/0}`` (uint32).

    Raises:
        AteraFormatError: If a mask's shape differs from the morphology image.
    """
    labels = {
        "cell_labels": _vendor_aligned_dask_array(bundle.cells_zip, "masks/1", chunks),
        "nucleus_labels": _vendor_aligned_dask_array(
            bundle.cells_zip, "masks/0", chunks
        ),
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
    bundle: AteraBundle,
    member: str,
    path: Path,
    *,
    slab_bytes: int = 256 << 20,
    on_slab: Callable[[np.ndarray], None] | None = None,
) -> Path:
    """Stream one vendor mask into an ``.npy`` file without holding it in memory.

    Args:
        bundle: Resolved bundle.
        member: ``"masks/0"`` (nuclei) or ``"masks/1"`` (cells).
        path: Output ``.npy`` file.
        slab_bytes: Approximate size of each row slab read from the zip.
        on_slab: Called with every decoded row slab, so a statistic (such as
            :class:`MaskPixelCounter`) needs no second decode of the mask.

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
                slab = np.asarray(source[rows])
                out[rows] = slab
                if on_slab is not None:
                    on_slab(slab)
            out.flush()
        finally:
            del out
    return path


class MaskPixelCounter:
    """Count the pixels of every label of one mask from its streamed row slabs.

    Attributes:
        counts: int64 pixels per label value (index 0 is the background).
        n_pixels: Pixels counted so far.
    """

    def __init__(self, n_labels: int) -> None:
        """Start with zero counts.

        Args:
            n_labels: Largest valid label (the record count).
        """
        self.n_labels = n_labels
        self.counts = np.zeros(n_labels + 1, dtype=np.int64)
        self.n_pixels = 0

    def __call__(self, slab: np.ndarray) -> None:
        """Add one slab of the mask.

        Args:
            slab: Mask pixels (any shape).

        Raises:
            AteraFormatError: If the slab holds a label above ``n_labels``.
        """
        values = np.asarray(slab).ravel()
        if values.size and int(values.max()) > self.n_labels:
            raise AteraFormatError(
                f"mask label {int(values.max())} exceeds {self.n_labels} records"
            )
        self.counts += np.bincount(values, minlength=self.n_labels + 1)
        self.n_pixels += int(values.size)


def nucleus_area_total_from_counts(
    counts: np.ndarray, cells: AteraCells, pixel_size_um: float
) -> np.ndarray:
    """Sum the nucleus pixel counts of each cell into an area.

    Args:
        counts: Pixels per nucleus label (index 0 the background).
        cells: Cell records.
        pixel_size_um: Microns per pixel.

    Returns:
        float64 area in µm² per cell, NaN for cells without a nucleus.
    """
    per_cell = np.bincount(
        cells.nucleus_cell_index, weights=counts[1:], minlength=cells.n_cells
    ) * (pixel_size_um**2)
    has_nucleus = np.bincount(cells.nucleus_cell_index, minlength=cells.n_cells) > 0
    return np.where(has_nucleus, per_cell, np.nan)


def atera_nucleus_area_total(
    bundle: AteraBundle, cells: AteraCells, *, slab_bytes: int = 256 << 20
) -> np.ndarray:
    """Sum the mask area of every nucleus of each cell, streaming ``masks/0``.

    ``cell_summary.nucleus_area`` describes only the largest nucleus of a
    multinucleate cell; this is the total. The builder counts the pixels
    while it exports ``masks/0`` for polygonisation instead, so this second
    decode only runs without that export.

    Args:
        bundle: Resolved bundle.
        cells: Cell records.
        slab_bytes: Approximate size of each row slab read from the zip.

    Returns:
        float64 area in µm² per cell, NaN for cells without a nucleus.

    Raises:
        AteraFormatError: If ``masks/0`` holds a label without a nucleus row.
    """
    counter = MaskPixelCounter(cells.n_nuclei)
    with open_zip_group(bundle.cells_zip) as group:
        source = group["masks/0"]
        shape = tuple(int(n) for n in source.shape)
        for rows in _mask_row_slabs(shape, int(source.chunks[0]), slab_bytes):
            counter(np.asarray(source[rows]))
    return nucleus_area_total_from_counts(counter.counts, cells, bundle.pixel_size_um)


def _pixel_polygons_to_microns(
    geometries: np.ndarray,
    pixel_size_um: float,
    simplify_px: float,
    *,
    n_threads: int = 1,
) -> np.ndarray:
    """Simplify pixel-space contours, then map them to microns at ``(c + 0.5) ps``.

    ``shapely.simplify`` releases the GIL, so chunks run on ``n_threads``.
    """
    if simplify_px > 0:
        chunks = [
            geometries[start : start + _SIMPLIFY_CHUNK]
            for start in range(0, len(geometries), _SIMPLIFY_CHUNK)
        ]
        with ThreadPoolExecutor(max_workers=max(1, n_threads)) as pool:
            simplified = list(
                pool.map(
                    lambda chunk: shapely.simplify(
                        chunk, simplify_px, preserve_topology=True
                    ),
                    chunks,
                )
            )
        if simplified:
            geometries = np.concatenate(simplified)
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
    simplify_px: float = DEFAULT_SIMPLIFY_PX,
    mp_context: PolygonStartMethod = "forkserver",
    nucleus_counter: MaskPixelCounter | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """Polygonise the vendor masks into cell and nucleus shapes (microns).

    Each mask is streamed into a memory-mapped ``.npy`` in ``workdir`` and
    polygonised by a process pool that maps the file (no copies, no silent
    serial fallback). Contours, whose pixel centres sit at integers, are
    simplified topology-preservingly by ``simplify_px`` pixels (on ``n_jobs``
    threads) and mapped to microns at ``(c + 0.5) * ps``, the vendor's corner
    convention. A label made of several components becomes a MultiPolygon.

    The workers start clean (``forkserver`` or ``spawn``; ``fork`` is refused
    because reading the masks has already started zarr threads), so they
    import the caller's ``__main__``: a calling script must guard its entry
    point with ``if __name__ == "__main__":``. Every worker maps the whole
    ``.npy``, so the pages it touches count in its file-backed RSS (about
    110 GB summed over 16 workers on a 10 GB mask) although they are shared
    page cache: a memory watchdog must measure anonymous memory (``RssAnon``,
    USS or PSS), not ``rss``.

    Args:
        bundle: Resolved bundle.
        cells: Cell records.
        n_jobs: Worker processes for the polygonisation pool, and threads for
            the simplification.
        workdir: Directory for the temporary ``.npy`` masks (deleted after,
            also when the export or the pool fails).
        simplify_px: Simplification tolerance in pixels (0 disables it).
        mp_context: ``"forkserver"`` or ``"spawn"``.
        nucleus_counter: Counts the nucleus pixels while ``masks/0`` is
            exported (for :func:`nucleus_area_total_from_counts`).

    Returns:
        ``{"cell_boundaries": ..., "nucleus_boundaries": ...}``. Both carry
        ``instance_id`` as a uint64 column and index (cell row + 1, nucleus
        row + 1), ``source_cell_id`` (of the owning cell) and
        ``segmentation_method``; nuclei also carry ``cell_instance_id``.

    Raises:
        ValueError: If ``mp_context`` is not ``forkserver`` or ``spawn``.
        AteraFormatError: If a label yields no polygon or exceeds the record count.
        RuntimeError: If the polygonisation pool fails.
    """
    if mp_context not in ("forkserver", "spawn"):
        raise ValueError(
            f"mp_context must be 'forkserver' or 'spawn', got {mp_context!r}: "
            "zarr threads are already running, so fork is unsafe"
        )
    workdir.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, gpd.GeoDataFrame] = {}
    for key, member, n_expected in (
        ("cell_boundaries", "masks/1", cells.n_cells),
        ("nucleus_boundaries", "masks/0", cells.n_nuclei),
    ):
        npy = workdir / f"atera_{member.replace('/', '_')}.npy"
        try:
            export_mask_npy(
                bundle,
                member,
                npy,
                on_slab=nucleus_counter if member == "masks/0" else None,
            )
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
            n_threads=n_jobs,
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

    Raises:
        AteraFormatError: If a polygon has 1 to 3 vertices (no closed ring).
    """
    shapes: dict[str, gpd.GeoDataFrame] = {}
    for key, vertices, num_vertices in (
        ("cell_boundaries", cells.cell_vertices, cells.cell_num_vertices),
        ("nucleus_boundaries", cells.nucleus_vertices, cells.nucleus_num_vertices),
    ):
        degenerate = np.flatnonzero((num_vertices > 0) & (num_vertices < 4))
        if degenerate.size:
            raise AteraFormatError(
                f"{key}: {degenerate.size} vendor polygon(s) with 1-3 vertices, "
                f"e.g. row {int(degenerate[0])}"
            )
        rows = np.flatnonzero(num_vertices > 0)
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
    prefix and are each optional. ``cell_summary`` columns without a named
    place (``nucleus_centroid_x/y``, the centroid of the largest nucleus, and
    ``z_level``) become ``vendor_<name>``. With ``keep_vendor_annotations``,
    the ``de_*`` statistics of the kept features go to
    ``varm["vendor_de"]`` and the vendor category colours (``#rrggbbaa``) to
    ``uns["vendor_<column>_colors"]``. The vendor's ``filtered`` flags
    (True = kept by the vendor) are copied as ``vendor_filtered``, not
    applied.

    Args:
        bundle: Resolved bundle.
        cells: Cell records (row order must match the matrix).
        gex_only: Keep only ``Gene Expression`` columns in X.
        keep_vendor_annotations: Copy Leiden, Azimuth (with colours), filtered
            flags, vendor counts and centroids, the vendor PCA/UMAP and the
            differential-expression statistics.
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
        obs_columns = _read_columns(obs_group, ("cell_id",))
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
        colors = (
            _vendor_colors(obs_group, vendor_columns) if keep_vendor_annotations else {}
        )
        de_columns = (
            _read_columns(
                group["var"],
                [
                    name
                    for name in group["var"].attrs.get("column-order", [])
                    if str(name).startswith(DE_COLUMN_PREFIX)
                ],
            )
            if keep_vendor_annotations
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
    for name in summary.columns:
        if name not in _MAPPED_SUMMARY_COLUMNS:
            obs[f"vendor_{name}"] = summary[name].to_numpy(np.float64)
    # One product sums every control type: X @ (features x types indicator).
    indicator = np.column_stack(
        [feature_types == kind for kind in DERIVED_CONTROL_COUNT_COLUMNS.values()]
    ).astype(np.int64)
    control_counts = np.asarray(matrix @ indicator, dtype=np.int64)
    for index, column in enumerate(DERIVED_CONTROL_COUNT_COLUMNS):
        obs[column] = control_counts[:, index]
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
    counts = matrix[:, keep] if gex_only else matrix
    del matrix
    counts.data = counts.data.astype(np.float32)
    table = ad.AnnData(X=counts, obs=obs, var=var)
    table.obsm["spatial"] = summary[["cell_centroid_x", "cell_centroid_y"]].to_numpy(
        np.float64
    )
    for name, values in obsm.items():
        table.obsm[f"{name}_vendor"] = np.asarray(values)
    for name, values in colors.items():
        table.uns[f"vendor_{name}_colors"] = values
    if de_columns:
        table.varm["vendor_de"] = pd.DataFrame(
            {name: np.asarray(values)[keep] for name, values in de_columns.items()},
            index=table.var_names,
        )
    return table


def _vendor_colors(obs_group: Any, columns: Mapping[str, Any]) -> dict[str, list[str]]:
    """Read the ``color`` arrays of the vendor categorical columns.

    The vendor stores one packed ``0xRRGGBBAA`` integer per category beside
    ``codes`` and ``categories`` (``ad.io.read_elem`` ignores it).

    Args:
        obs_group: The matrix ``obs`` group.
        columns: The vendor columns that were read.

    Returns:
        ``#rrggbbaa`` strings per column, in category order.

    Raises:
        AteraFormatError: If a colour array does not match its categories.
    """
    colors: dict[str, list[str]] = {}
    for name, values in columns.items():
        if not isinstance(values, pd.Categorical) or name not in obs_group:
            continue
        member = obs_group[name]
        if not hasattr(member, "keys") or "color" not in member:
            continue
        packed = np.asarray(member["color"][:]).astype(np.int64) & 0xFFFFFFFF
        categories = values.categories
        if packed.shape != (len(categories),):
            raise AteraFormatError(
                f"obs/{name}/color has {packed.size} entries for "
                f"{len(categories)} categories"
            )
        colors[name] = [f"#{int(value):08x}" for value in packed]
    return colors


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
    decodes its whole tile, so a tile is read once per block, 16 times for
    500 µm blocks of a 2000 µm tile, and the reader bounds concurrent
    decodes); a tile
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
        qv_sum: Sum of the QV of the rows read.
        qv_histogram: Rows read per quarter-QV step from 0 to 40.
        gene_counts_all_qv: Rows read per feature.
        gene_counts_q20: Rows read with QV >= 20 per feature.
        gene_counts_kept: Rows kept per feature.
        bounds_um: ``(x_min, x_max, y_min, y_max, z_min, z_max)`` of rows read.
        density: Rows kept per ``DENSITY_BIN_UM`` bin (``[ny, nx]``).
    """

    tile_key: str
    n_rows_read: int
    n_rows_kept: int
    qv_sum: float
    qv_histogram: np.ndarray
    gene_counts_all_qv: np.ndarray
    gene_counts_q20: np.ndarray
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
        mean_qv_per_tile: Mean QV of the rows read per tile (``None`` if empty).
        gene_counts_all_qv: Rows read per feature (all QV); equals the vendor
            ``grid.gene_counts`` when every tile is read.
        gene_counts_q20: Rows read with QV >= 20 per feature, whatever
            ``min_qv`` is.
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
    mean_qv_per_tile: dict[str, float | None]
    gene_counts_all_qv: np.ndarray
    gene_counts_q20: np.ndarray
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
            "mean_qv_per_tile": dict(self.mean_qv_per_tile),
            "gene_counts_all_qv": self.gene_counts_all_qv.tolist(),
            "gene_counts_q20": self.gene_counts_q20.tolist(),
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

    ``tile_semaphore`` bounds concurrent whole-tile reads (``tile_blocks``);
    ``zip_groups`` holds each thread's open transcripts store; ``token`` names
    the read in dask task keys.
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
    zip_groups: _ThreadZipGroups = field(default_factory=_ThreadZipGroups)
    token: str = field(default="")


def _read_rows(tile: Any, name: str, rows: slice) -> np.ndarray:
    """Read one row range of one tile array (the only tile-array access)."""
    return np.asarray(tile[name][rows])


def _gene_codes(
    gene_offset: np.ndarray,
    start: int,
    stop: int,
    n_rows: int,
    n_genes: int,
    selected: np.ndarray | None = None,
) -> np.ndarray:
    """Gene code of rows ``[start, stop)`` from end-exclusive ``gene_offset``.

    Args:
        gene_offset: ``[n_genes, 2]`` row ranges of the tile.
        start: First row of the range.
        stop: End of the range (exclusive).
        n_rows: Rows in the tile.
        n_genes: Features of the bundle; ``gene_offset`` has one row each.
        selected: Positions within the range to code (all when ``None``).

    Returns:
        int32 codes, one per row (or per selected row).

    Raises:
        AteraFormatError: If ``gene_offset`` is not one contiguous range per
            feature covering the tile.
    """
    if gene_offset.shape != (n_genes, 2):
        raise AteraFormatError(
            f"gene_offset has shape {gene_offset.shape}, not ({n_genes}, 2)"
        )
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
    if selected is not None:
        # Row r belongs to the first gene whose end exceeds r.
        return np.searchsorted(ends, start + selected, side="right").astype(np.int32)
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


def _block_rows(
    location: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """Positions of the rows whose ``(x, y)`` lies in ``[x0, x1) x [y0, y1)``."""
    x0, x1, y0, y1 = bounds
    x, y = location[:, 0], location[:, 1]
    return np.flatnonzero((x >= x0) & (x < x1) & (y >= y0) & (y < y1))


def _read_selected(
    tile: Any, name: str, rows: slice, width: int, selected: np.ndarray | None
) -> np.ndarray:
    """Read one tile column range and keep only ``selected`` rows (all if None)."""
    values = _read_rows(tile, name, rows).reshape(-1, width)
    if selected is not None:
        values = values[selected]
    return values if width > 1 else values.reshape(-1)


def _read_partition_with_stats(
    context: _TranscriptContext, spec: PartitionSpec
) -> tuple[pd.DataFrame, PartitionStats]:
    rows = slice(spec.row_start, spec.row_stop)
    n_genes = len(context.gene_dtype.categories)
    semaphore = context.tile_semaphore
    if semaphore is not None:
        semaphore.acquire()
    try:
        # Whole-tile columns live only inside this block: each is cut to the
        # partition's spatial block as soon as it is read, so at most
        # max_tiles_in_flight whole-tile columns exist at once.
        tile = context.zip_groups.group(context.zip_path)["grid"][spec.tile_key]
        n_tile_rows = int(tile["location"].shape[0])
        gene_offset = np.asarray(tile["gene_offset"][:])
        location: np.ndarray = _read_rows(tile, "location", rows).reshape(-1, 3)
        selected = (
            _block_rows(location, spec.block_bounds_um)
            if spec.block_bounds_um is not None
            else None
        )
        if selected is not None:
            location = location[selected]
        cell_id = _read_selected(tile, "cell_id", rows, 2, selected)
        quality = _read_selected(tile, "quality_score", rows, 1, selected)
        overlaps = _read_selected(tile, "overlaps_nucleus", rows, 1, selected)
        codewords = (
            _read_selected(tile, "codeword_identity", rows, 1, selected)
            if context.emit_codeword_index
            else None
        )
    finally:
        if semaphore is not None:
            semaphore.release()
    codes = _gene_codes(
        gene_offset, spec.row_start, spec.row_stop, n_tile_rows, n_genes, selected
    )
    positions = (
        selected
        if selected is not None
        else np.arange(spec.row_stop - spec.row_start, dtype=np.int64)
    )
    ordinals = spec.global_offset + spec.row_start + positions.astype(np.int64)

    q20 = quality >= np.float64(Q20)
    keep = quality >= np.float64(context.min_qv) if context.min_qv is not None else None
    qv_bins = np.clip(
        np.rint(quality.astype(np.float32) / np.float32(QV_HISTOGRAM_STEP)),
        0,
        QV_HISTOGRAM_BINS - 1,
    ).astype(np.intp)
    bounds = (
        tuple(
            float(value)
            for axis in range(3)
            for value in (location[:, axis].min(), location[:, axis].max())
        )
        if len(location)
        else ()
    )
    n_rows_read = len(codes)
    gene_counts_all_qv = np.bincount(codes, minlength=n_genes)
    gene_counts_q20 = np.bincount(codes[q20], minlength=n_genes)
    qv_histogram = np.bincount(qv_bins, minlength=QV_HISTOGRAM_BINS)
    qv_sum = float(quality.sum(dtype=np.float64))
    del qv_bins, q20
    if keep is not None:
        location, cell_id, overlaps = location[keep], cell_id[keep], overlaps[keep]
        codes, quality, ordinals = codes[keep], quality[keep], ordinals[keep]
        codewords = codewords[keep] if codewords is not None else None
    stats = PartitionStats(
        tile_key=spec.tile_key,
        n_rows_read=n_rows_read,
        n_rows_kept=len(codes),
        qv_sum=qv_sum,
        qv_histogram=qv_histogram,
        gene_counts_all_qv=gene_counts_all_qv,
        gene_counts_q20=gene_counts_q20,
        gene_counts_kept=np.bincount(codes, minlength=n_genes),
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
        codes=codes,
        ordinals=ordinals,
        codewords=codewords,
    )
    return frame, stats


class TranscriptAggregates:
    """Transcript statistics recorded by the partition reads themselves.

    Every partition read adds its :class:`PartitionStats` to running totals,
    so whatever computes the points (a parquet write inside spatialdata, or a
    plain ``compute``) gathers the statistics in the same pass. Recording is
    idempotent: a re-executed task's partition is counted once (reads are
    deterministic). The record lives in this process, so the points must be
    computed by dask's threaded or synchronous scheduler.
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
        self._min_qv = min_qv
        self._density_grid = density_grid
        self._recorded: set[int] = set()
        self._read_per_tile = dict.fromkeys(self._tile_keys, 0)
        self._kept_per_tile = dict.fromkeys(self._tile_keys, 0)
        self._qv_sum_per_tile = dict.fromkeys(self._tile_keys, 0.0)
        self._histograms = {
            key: np.zeros(QV_HISTOGRAM_BINS, dtype=np.int64) for key in self._tile_keys
        }
        self._all_qv = np.zeros(n_genes, dtype=np.int64)
        self._q20 = np.zeros(n_genes, dtype=np.int64)
        self._kept = np.zeros(n_genes, dtype=np.int64)
        self._density = np.zeros((density_grid.n_y, density_grid.n_x), dtype=np.int64)
        self._bounds = np.array([np.inf, -np.inf] * 3)
        self._lock = threading.Lock()
        # Distinguishes the graphs of two reads with the same parameters, so
        # dask never merges their tasks and fills only one record.
        self.token = uuid.uuid4().hex

    def record(self, index: int, stats: PartitionStats) -> None:
        """Add the statistics of partition ``index`` (once per index)."""
        with self._lock:
            if index in self._recorded:
                return
            self._recorded.add(index)
            key = stats.tile_key
            self._read_per_tile[key] += stats.n_rows_read
            self._kept_per_tile[key] += stats.n_rows_kept
            self._qv_sum_per_tile[key] += stats.qv_sum
            self._histograms[key] += stats.qv_histogram
            self._all_qv += stats.gene_counts_all_qv
            self._q20 += stats.gene_counts_q20
            self._kept += stats.gene_counts_kept
            self._density += stats.density
            if stats.bounds_um:
                values = np.asarray(stats.bounds_um)
                self._bounds[0::2] = np.minimum(self._bounds[0::2], values[0::2])
                self._bounds[1::2] = np.maximum(self._bounds[1::2], values[1::2])

    def summary(self) -> TranscriptSummary:
        """Return the statistics over every partition.

        Returns:
            The combined statistics.

        Raises:
            RuntimeError: If some partitions have not been read yet (compute
                or write the points first).
        """
        with self._lock:
            n_recorded = len(self._recorded)
            if n_recorded != self.n_partitions:
                raise RuntimeError(
                    f"only {n_recorded} of {self.n_partitions} transcript "
                    "partitions were read; compute or write the points before "
                    "summarising them"
                )
            grid = self._density_grid
            names = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
            return TranscriptSummary(
                min_qv=self._min_qv,
                n_rows_read=int(sum(self._read_per_tile.values())),
                n_rows_kept=int(sum(self._kept_per_tile.values())),
                rows_read_per_tile=dict(self._read_per_tile),
                rows_kept_per_tile=dict(self._kept_per_tile),
                qv_histogram_per_tile={
                    key: values.copy() for key, values in self._histograms.items()
                },
                mean_qv_per_tile={
                    key: (total / self._read_per_tile[key])
                    if self._read_per_tile[key]
                    else None
                    for key, total in self._qv_sum_per_tile.items()
                },
                gene_counts_all_qv=self._all_qv.copy(),
                gene_counts_q20=self._q20.copy(),
                gene_counts_kept=self._kept.copy(),
                bounds_um={
                    name: float(value)
                    for name, value in zip(names, self._bounds, strict=True)
                    if np.isfinite(value)
                },
                density=self._density.copy(),
                density_origin_um=(grid.origin_x, grid.origin_y),
                density_bin_um=grid.bin_um,
            )


class _PartitionReader:
    """``from_map`` function: read one partition and record its statistics.

    It holds an in-process record, a tile semaphore and per-thread zip stores,
    so it refuses to be pickled: dask's ``processes`` scheduler or a
    distributed cluster would read in other processes, where the statistics
    and the decode bound are lost. Copies share the original's state.
    """

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

    def __copy__(self) -> _PartitionReader:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _PartitionReader:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        raise TypeError(
            "Atera transcript partitions record their statistics in this process; "
            "compute them with dask's threaded or synchronous scheduler, not the "
            "processes scheduler or a distributed cluster"
        )


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
    ``PointsModel.parse(..., sort=True)`` triggers no computation. Compute the
    points with dask's threaded (or synchronous) scheduler: the statistics and
    the ``tile_blocks`` decode bound live in this process, so the partition
    function refuses to be pickled for other processes.

    Args:
        bundle: Resolved bundle.
        min_qv: Keep rows with QV at least this (``None`` keeps all).
        partition_rows: Rows per ``row_range`` partition.
        spatial_order: ``"row_range"`` (default) or ``"tile_blocks"``.
        block_um: Block edge for ``tile_blocks``.
        min_block_rows: See :func:`plan_transcript_partitions`.
        max_tiles_in_flight: Concurrent whole-tile reads in ``tile_blocks``;
            each is cut to its block before the next read starts.
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
            if gene_offset.shape != (len(categories), 2):
                raise AteraFormatError(
                    f"tile {tile.key} gene_offset has shape {gene_offset.shape}, "
                    f"not ({len(categories)}, 2)"
                )
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
