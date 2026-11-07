"""Transcript spatial patterns over every transcript with label rasters.

The polygon engine (:mod:`merxen.analysis.transcript_spatial_patterns`) keeps
every retained transcript in memory and asks shape trees where each one lies.
That needs cortical tissue annotations and does not scale to whole-transcriptome
sections (more than a billion transcripts). This *label-raster engine* gives
the same compartment and signed-distance statistics without keeping points:

1. **Tissue support.** Hand annotations win when the sample has them (the
   polygon is rasterised on a 10 um grid; transcripts in grid bins that the
   boundary crosses get an exact point-in-polygon test). Otherwise the tissue
   is the all-transcript density at 10 um, smoothed, thresholded (log-Otsu,
   capped at a fraction of the dense-tissue level), morphologically closed,
   hole-filled and cleared of small fragments (:func:`density_tissue_mask`).
   The method and its parameters are recorded.
2. **Compartments.** Each transcript's pixel in the layer's cell label raster
   and the nucleus label raster (``PlatformSpec.mask_pixel_convention``, the
   vendor's float32 corner rule for ATERA) decides nuclear, cytoplasmic or
   extracellular. Label rasters cannot overlap, so the overlap counters of the
   polygon engine are 0 by construction.
3. **Signed distances.** A tiled exact Euclidean distance transform
   (``scipy.ndimage.distance_transform_edt``) of each label raster's boundary
   pixels, with a halo just above the largest distance any statistic needs
   (200 um by default), is stored as a ``uint16`` memmap in task scratch.
   A pixel's distance is the distance from its centre to the nearest pixel
   centre that touches a label change, plus half a pixel: within one pixel
   of the polygon engine's point-to-boundary distance. Positive inside a
   label, negative outside; distances beyond the halo are reported as beyond
   the outermost bin edge.
4. **Accumulators.** One streaming pass adds every transcript to per-gene
   compartment counts, signed-distance bin counts, fine distance histograms
   (for the medians) and the pericellular and membrane-proximal counters.
   The shared count-based statistics of the polygon engine then give the
   tables (:func:`merxen.analysis.transcript_spatial_patterns.
   summarize_gene_pattern_counts`).

Nothing is thinned or subsampled: ``n_transcripts_used`` is every finite,
in-tissue, non-control transcript. Pair correlation is not computed by this
engine (a user decision for AT-8; an all-transcript method is a follow-up),
which the summary records in ``paircorr_status``.

Memory is bounded by the per-gene accumulators (about 1.2 GB for 18,000
genes), the label block cache and one record batch per worker thread.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import ndimage
from shapely import box as shapely_box
from shapely import contains_xy, intersects, prepare
from shapely.geometry.base import BaseGeometry

from merxen.analysis.transcript_spatial_patterns import (
    COMPARTMENT_CODES,
    COMPARTMENTS,
    SIGNED_DISTANCE_BOUNDARIES,
    GenePatternCounts,
    add_signed_distance_summary,
    build_analysis_tissue_polygon,
    classify_gene_patterns,
    rank_transcript_patterns,
    signed_distance_enrichment_from_counts,
    summarize_gene_pattern_counts,
)
from merxen.config import SpatialGeneAnalysisConfig, SpatialGeneAnalysisSampleConfig
from merxen.control_features import control_transcript_mask
from merxen.enrichment.label_lookup import (
    DEFAULT_BLOCK_PX,
    BlockCache,
    PixelFrame,
    default_n_workers,
    lookup_labels,
    natural_partition_order,
    pixel_indices,
)
from merxen.io.points_aggregates import DEFAULT_POINTS_KEY, points_parquet_dir
from merxen.memory import log_status
from merxen.platforms import PixelConvention, find_um_per_px, platform_spec
from merxen.visualization.raster_density import bin_indices

logger = logging.getLogger(__name__)

ENGINE_NAME: Final = "label_raster"
"""Name of this engine in manifests and summaries."""

PAIRCORR_STATUS: Final = "not_computed_label_raster_engine"
"""``paircorr_status`` of every gene: pair correlation is off on this engine."""

PAIRCORR_COLUMNS: Final[tuple[str, ...]] = (
    "gene",
    "null_model",
    "band_index",
    "distance_min_um",
    "distance_max_um",
    "band_label",
    "n_transcripts_total",
    "n_transcripts_used",
    "thinning_fraction",
    "observed_pair_count",
    "null_pair_count_mean",
    "null_pair_count_low",
    "null_pair_count_high",
    "paircorr_enrichment",
    "pval_clustered",
    "pval_disperse",
    "pval_clustered_fdr_bh",
    "pval_disperse_fdr_bh",
)
"""Columns of the (empty) pair-correlation table this engine writes."""

DISTANCE_CLIP_CODE: Final = np.iinfo(np.uint16).max
"""``uint16`` code of a distance beyond the exact range (the halo)."""

MAX_DISTANCE_CODE: Final = DISTANCE_CLIP_CODE - 1
"""Largest code of an exact distance."""

DEFAULT_BATCH_ROWS: Final = 500_000
"""Rows per record batch when the config gives none."""

_ORIGINAL_SEGMENTATIONS: Final = frozenset({"original_seg", "original"})
_PROSEG_SEGMENTATIONS: Final = frozenset({"reseg", "proseg"})


@dataclass(frozen=True)
class RasterEngineSettings:
    """Parameters of the label-raster engine beyond the stage config.

    They are not fields of :class:`~merxen.config.SpatialGeneAnalysisConfig`
    on purpose: the Nextflow JSON never sets them, and ``merxen/config.py`` is
    part of the map_first hierarchy fingerprint, so a new field there would
    re-run every row's annotation ``COMPUTE_CPU`` task. Python callers may pass
    other values.

    Attributes:
        tissue_bin_um: Bin of the tissue grid (density and annotation masks).
        tissue_smoothing_sigma_um: Gaussian sigma of the density.
        tissue_closing_radius_um: Radius of the closing disk.
        tissue_min_component_area_um2: Smallest kept tissue component.
        tissue_max_threshold_fraction: Cap of the density threshold, as a
            fraction of the 99th percentile of the positive bins.
        distance_tile_px: Output tile edge of the distance transform.
        median_bin_um: Bin of the fine distance histograms (medians).
        diagnostic_bin_um: Bin of the densest-window diagnostic rasters.
        scratch_dir: Parent of the temporary distance memmaps; ``None`` puts
            them beside the output directory (the Nextflow task directory).
        n_workers: Threads; ``None`` uses ``DASK_NUM_WORKERS``
            (:func:`~merxen.enrichment.label_lookup.default_n_workers`).
        block_cache_bytes: Decoded label blocks kept across batches.
    """

    tissue_bin_um: float = 10.0
    tissue_smoothing_sigma_um: float = 10.0
    tissue_closing_radius_um: float = 50.0
    tissue_min_component_area_um2: float = 50_000.0
    tissue_max_threshold_fraction: float = 0.05
    distance_tile_px: int = 4096
    median_bin_um: float = 0.05
    diagnostic_bin_um: float = 1.0
    scratch_dir: Path | None = None
    n_workers: int | None = None
    block_cache_bytes: int = 3 * 1024**3

    def workers(self) -> int:
        """Return the worker threads to use."""
        return (
            int(self.n_workers) if self.n_workers is not None else default_n_workers()
        )


DEFAULT_SETTINGS: Final = RasterEngineSettings()
"""Settings the pipeline uses."""


# --------------------------------------------------------------------------
# Layer resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelLayer:
    """Label elements and pixel grid of one analysis layer of a store.

    Attributes:
        cell_label_key: Label element of the layer's cells.
        nucleus_label_key: Label element of the nuclei.
        um_per_px: Microns per label pixel.
        convention: Micron -> pixel rule of the lookups.
    """

    cell_label_key: str
    nucleus_label_key: str
    um_per_px: float
    convention: PixelConvention


def layer_label_keys(
    zarr_path: Path, platform: str, segmentation: str | None
) -> tuple[str, str] | None:
    """Return the cell and nucleus label elements of a layer, if the store has them.

    The original (vendor) layer uses ``PlatformSpec.original_label_keys``; the
    ProSeg layer (``reseg``) uses ``PlatformSpec.proseg_label_key`` for its
    cells and the vendor nucleus labels. Platforms without label elements
    (XENIUM, MERSCOPE) and layers without one answer ``None``.

    Args:
        zarr_path: Latest SpatialData store.
        platform: Registered platform name.
        segmentation: Analysis segmentation of the layer (``None`` means the
            original layer).

    Returns:
        ``(cell_label_key, nucleus_label_key)``, or ``None``.
    """
    spec = platform_spec(platform)
    if len(spec.original_label_keys) < 2:
        return None
    layer = "original_seg" if segmentation is None else str(segmentation)
    if layer in _ORIGINAL_SEGMENTATIONS:
        cell_key: str | None = spec.original_label_keys[0]
    elif layer in _PROSEG_SEGMENTATIONS:
        cell_key = spec.proseg_label_key
    else:
        cell_key = None
    if cell_key is None:
        return None
    nucleus_key = spec.original_label_keys[1]
    labels_dir = Path(zarr_path) / "labels"
    if not (labels_dir / cell_key).is_dir() or not (labels_dir / nucleus_key).is_dir():
        return None
    return cell_key, nucleus_key


def resolve_label_layer(
    zarr_path: Path,
    platform: str,
    segmentation: str | None,
    *,
    points_key: str = DEFAULT_POINTS_KEY,
) -> LabelLayer:
    """Return the label layer of a sample, with its pixel size checked.

    The pixel size comes from the platform's spec sidecars under the store
    and must agree with the ``Scale`` transform the points element records.

    Raises:
        ValueError: If the layer has no label elements or the two pixel sizes
            disagree.
    """
    keys = layer_label_keys(zarr_path, platform, segmentation)
    if keys is None:
        raise ValueError(
            f"{platform} layer {segmentation!r} of the store has no cell and "
            "nucleus label elements"
        )
    um_per_px = float(find_um_per_px(Path(zarr_path), platform))
    recorded = points_um_per_px(Path(zarr_path), points_key)
    if recorded is not None and not math.isclose(
        recorded, um_per_px, rel_tol=1e-6, abs_tol=0.0
    ):
        raise ValueError(
            f"pixel size {um_per_px} of the spec sidecar disagrees with the "
            f"points transform ({recorded} um per global unit)"
        )
    return LabelLayer(
        cell_label_key=keys[0],
        nucleus_label_key=keys[1],
        um_per_px=um_per_px,
        convention=platform_spec(platform).mask_pixel_convention,
    )


def points_um_per_px(zarr_path: Path, points_key: str) -> float | None:
    """Return the microns per ``global`` unit a points element records, if a scale."""
    meta = Path(zarr_path, "points", points_key, "zarr.json")
    if not meta.is_file():
        return None
    attrs = json.loads(meta.read_text()).get("attributes", {})
    transforms = attrs.get("coordinateTransformations") or []
    for transform in transforms:
        if transform.get("type") != "scale":
            continue
        output = (transform.get("output") or {}).get("name")
        if output not in (None, "global"):
            continue
        scale = [float(v) for v in transform.get("scale", [])]
        if len(scale) >= 2 and scale[0] > 0 and math.isclose(scale[0], scale[1]):
            return 1.0 / scale[0]
    return None


def open_label_level0(zarr_path: Path, key: str) -> Any:
    """Open the level-0 array of a label element lazily (zarr, read-only)."""
    import zarr

    element = Path(zarr_path, "labels", key)
    meta = element / "zarr.json"
    level = "s0"
    if meta.is_file():
        attrs = json.loads(meta.read_text()).get("attributes", {})
        multiscales = (attrs.get("ome") or attrs).get("multiscales") or []
        if multiscales and multiscales[0].get("datasets"):
            level = str(multiscales[0]["datasets"][0]["path"])
    elif not (element / level).exists() and (element / "0").exists():
        level = "0"
    return zarr.open_array(str(element / level), mode="r")


def points_files(zarr_path: Path, points_key: str = DEFAULT_POINTS_KEY) -> list[Path]:
    """Return the parquet files of a points element in natural order."""
    directory = points_parquet_dir(Path(zarr_path), points_key)
    if directory.is_file():
        return [directory]
    files = natural_partition_order(list(directory.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"points[{points_key}] of the store has no files")
    return files


# --------------------------------------------------------------------------
# Tissue support
# --------------------------------------------------------------------------


@dataclass
class TissueMask:
    """Tissue support on a regular micron grid.

    Attributes:
        origin_um: ``(x0, y0)`` of the first bin's corner.
        bin_um: Bin edge length.
        inside: Boolean grid (row = y): bins inside the tissue.
        method: ``"density"`` or ``"annotation"``.
        parameters: Method parameters and derived values (threshold, areas).
        boundary: Bins a hand-annotation boundary crosses; transcripts there
            get an exact point-in-polygon test (annotation masks only).
        polygon: The annotation polygon (annotation masks only).
        density: All-transcript counts per bin the mask came from (density
            masks only; kept for the tissue-support figure).
    """

    origin_um: tuple[float, float]
    bin_um: float
    inside: np.ndarray
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    boundary: np.ndarray | None = None
    polygon: BaseGeometry | None = None
    density: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_y, n_x)`` of the grid."""
        n_y, n_x = self.inside.shape
        return int(n_y), int(n_x)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return ``(left, right, bottom, top)`` for ``imshow``."""
        n_y, n_x = self.shape
        x0, y0 = self.origin_um
        return (x0, x0 + n_x * self.bin_um, y0, y0 + n_y * self.bin_um)

    @property
    def area_um2(self) -> float:
        """Return the tissue area in square microns."""
        if self.polygon is not None:
            return float(self.polygon.area)
        return float(np.count_nonzero(self.inside)) * self.bin_um**2

    def bins(self, x_um: np.ndarray, y_um: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the grid column and row of each coordinate (``-1`` outside)."""
        n_y, n_x = self.shape
        ix = bin_indices(x_um, self.origin_um[0], self.bin_um, n_x)
        iy = bin_indices(y_um, self.origin_um[1], self.bin_um, n_y)
        return ix, iy

    def contains(self, x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
        """Return whether each coordinate lies in the tissue."""
        x = np.asarray(x_um, dtype=np.float64)
        y = np.asarray(y_um, dtype=np.float64)
        ix, iy = self.bins(x, y)
        on_grid = (ix >= 0) & (iy >= 0)
        result = np.zeros(x.shape, dtype=bool)
        result[on_grid] = self.inside[iy[on_grid], ix[on_grid]]
        if self.boundary is not None and self.polygon is not None:
            exact = np.zeros(x.shape, dtype=bool)
            exact[on_grid] = self.boundary[iy[on_grid], ix[on_grid]]
            if exact.any():
                result[exact] = contains_xy(self.polygon, x[exact], y[exact])
        return result

    def support(self) -> np.ndarray:
        """Return the bins that may hold tissue (inside or on the boundary)."""
        if self.boundary is None:
            return self.inside
        merged: np.ndarray = self.inside | self.boundary
        return merged

    def overlaps_window(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Return whether any tissue bin meets the micron window."""
        n_y, n_x = self.shape
        ox, oy = self.origin_um
        c0 = max(0, int(math.floor((x0 - ox) / self.bin_um)) - 1)
        c1 = min(n_x, int(math.ceil((x1 - ox) / self.bin_um)) + 1)
        r0 = max(0, int(math.floor((y0 - oy) / self.bin_um)) - 1)
        r1 = min(n_y, int(math.ceil((y1 - oy) / self.bin_um)) + 1)
        if c0 >= c1 or r0 >= r1:
            return False
        return bool(self.support()[r0:r1, c0:c1].any())

    def record(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the mask."""
        return {
            "method": self.method,
            "bin_um": self.bin_um,
            "origin_um": list(self.origin_um),
            "shape": list(self.shape),
            "area_um2": self.area_um2,
            **self.parameters,
        }


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Return Otsu's threshold of a 1-D sample (between-class variance max)."""
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if not high > low:
        return low
    hist, edges = np.histogram(values, bins=n_bins, range=(low, high))
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(hist)
    weight_high = np.cumsum(hist[::-1])[::-1]
    mean_low = np.cumsum(hist * centres) / np.maximum(weight_low, 1)
    mean_high = (np.cumsum((hist * centres)[::-1]) / np.maximum(weight_high[::-1], 1))[
        ::-1
    ]
    between = weight_low[:-1] * weight_high[1:] * (mean_low[:-1] - mean_high[1:]) ** 2
    return float(centres[int(np.argmax(between))])


def density_tissue_mask(
    counts: np.ndarray,
    *,
    origin_um: tuple[float, float],
    bin_um: float,
    smoothing_sigma_um: float = 10.0,
    closing_radius_um: float = 50.0,
    min_component_area_um2: float = 50_000.0,
    max_threshold_fraction: float = 0.05,
) -> TissueMask:
    """Derive the tissue support from an all-transcript density raster.

    Steps, all recorded in the mask's ``parameters``:

    1. Gaussian smoothing (``smoothing_sigma_um``).
    2. Threshold: Otsu on ``log10`` of the positive smoothed bins (the sparse
       off-tissue background and the tissue form two modes), capped at
       ``max_threshold_fraction`` of the 99th percentile of the positive bins,
       so that a window holding no background (a crop inside the tissue) is
       not split into dense and sparse tissue.
    3. Binary closing with a disk of ``closing_radius_um`` (on a padded grid,
       so the grid edge does not erode the mask).
    4. Hole filling.
    5. Removal of connected components smaller than
       ``min_component_area_um2`` (debris).

    Args:
        counts: Transcripts per bin (row = y).
        origin_um: ``(x0, y0)`` of the grid.
        bin_um: Bin edge length.
        smoothing_sigma_um: Gaussian sigma; 0 disables smoothing.
        closing_radius_um: Radius of the closing disk; 0 disables closing.
        min_component_area_um2: Smallest kept component.
        max_threshold_fraction: Cap of the threshold, as a fraction of the
            99th percentile of the positive smoothed bins.

    Returns:
        The density-derived mask.

    Raises:
        ValueError: If the raster holds no transcript.
    """
    density = np.asarray(counts, dtype=np.float64)
    if not np.any(density > 0):
        raise ValueError("the density raster holds no transcript")
    sigma_bins = float(smoothing_sigma_um) / float(bin_um)
    smooth = (
        ndimage.gaussian_filter(density, sigma=sigma_bins, mode="constant")
        if sigma_bins > 0
        else density
    )
    positive = smooth[smooth > 0]
    # Smoothing spreads far-off counts into tiny values; they are background.
    positive = positive[positive >= 1e-3 * float(positive.max())]
    otsu = float(10.0 ** _otsu_threshold(np.log10(positive)))
    p99 = float(np.percentile(positive, 99.0))
    ceiling = float(max_threshold_fraction) * p99
    threshold = min(otsu, ceiling)
    rule = "otsu_log10" if otsu <= ceiling else "fraction_of_p99"
    mask = smooth >= threshold
    radius_bins = int(round(float(closing_radius_um) / float(bin_um)))
    if radius_bins > 0:
        yy, xx = np.mgrid[
            -radius_bins : radius_bins + 1, -radius_bins : radius_bins + 1
        ]
        disk = (xx**2 + yy**2) <= radius_bins**2
        padded = np.pad(mask, radius_bins + 1, constant_values=False)
        closed = ndimage.binary_closing(padded, structure=disk)
        mask = closed[
            radius_bins + 1 : -(radius_bins + 1), radius_bins + 1 : -(radius_bins + 1)
        ]
    mask = ndimage.binary_fill_holes(mask)
    labelled, n_components = ndimage.label(mask)
    min_bins = float(min_component_area_um2) / float(bin_um) ** 2
    removed_bins = 0
    n_removed = 0
    if n_components:
        sizes = np.bincount(labelled.ravel(), minlength=n_components + 1)
        small = sizes < min_bins
        small[0] = False
        n_removed = int(small.sum())
        removed_bins = int(sizes[small].sum())
        mask = mask & ~small[labelled]
        n_components -= n_removed
    if not mask.any():
        raise ValueError(
            "the density-derived tissue mask is empty; lower "
            "transcript_tissue_min_component_area_um2 or check the store"
        )
    total = float(density.sum())
    in_mask = float(density[mask].sum())
    return TissueMask(
        origin_um=(float(origin_um[0]), float(origin_um[1])),
        bin_um=float(bin_um),
        inside=np.asarray(mask, dtype=bool),
        method="density",
        parameters={
            "source": "all transcripts (controls included), every transcript binned",
            "smoothing_sigma_um": float(smoothing_sigma_um),
            "threshold_rule": rule,
            "threshold_transcripts_per_bin": threshold,
            "otsu_log10_threshold_transcripts_per_bin": otsu,
            "p99_positive_bin": p99,
            "max_threshold_fraction": float(max_threshold_fraction),
            "closing_radius_um": float(closing_radius_um),
            "fill_holes": True,
            "min_component_area_um2": float(min_component_area_um2),
            "n_components": int(n_components),
            "n_components_removed": n_removed,
            "removed_area_um2": removed_bins * float(bin_um) ** 2,
            "fraction_of_transcripts_in_mask": in_mask / total if total else 0.0,
        },
    )


def polygon_tissue_mask(
    polygon: BaseGeometry,
    *,
    origin_um: tuple[float, float],
    bin_um: float,
    shape: tuple[int, int],
) -> TissueMask:
    """Rasterise a hand-annotated tissue polygon for streaming lookups.

    A bin is inside when the polygon covers it, on the boundary when its box
    meets the polygon's boundary; transcripts in boundary bins get the exact
    ``contains_xy`` test, so the mask agrees with the polygon everywhere.
    """
    n_y, n_x = (int(v) for v in shape)
    x0, y0 = float(origin_um[0]), float(origin_um[1])
    xs = x0 + (np.arange(n_x) + 0.5) * bin_um
    ys = y0 + (np.arange(n_y) + 0.5) * bin_um
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = contains_xy(polygon, grid_x, grid_y)
    half = bin_um / 2.0
    boundary = np.zeros((n_y, n_x), dtype=bool)
    outline = polygon.boundary
    minx, miny, maxx, maxy = outline.bounds
    cols = np.flatnonzero((xs + half >= minx) & (xs - half <= maxx))
    rows = np.flatnonzero((ys + half >= miny) & (ys - half <= maxy))
    if cols.size and rows.size:
        prepare(outline)
        for row in rows:
            cells = shapely_box(
                xs[cols] - half, ys[row] - half, xs[cols] + half, ys[row] + half
            )
            boundary[row, cols] = intersects(cells, outline)
    inside &= ~boundary
    return TissueMask(
        origin_um=(x0, y0),
        bin_um=float(bin_um),
        inside=inside,
        method="annotation",
        parameters={
            "source": "hand tissue annotations (pia / tissue edge / exclusions)",
            "boundary_rule": "exact contains_xy in boundary bins",
        },
        boundary=boundary,
        polygon=polygon,
    )


def files_bounds(files: Sequence[Path]) -> tuple[float, float, float, float]:
    """Return the ``x``/``y`` bounds of parquet files from their footers.

    A file without min/max statistics is scanned.

    Raises:
        ValueError: If no finite coordinate exists.
    """
    low = [math.inf, math.inf]
    high = [-math.inf, -math.inf]
    for path in files:
        metadata = pq.read_metadata(path)
        names = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
        ranges: list[tuple[float, float]] | None = []
        for name in ("x", "y"):
            position = names.index(name)
            lo, hi = math.inf, -math.inf
            for group in range(metadata.num_row_groups):
                column = metadata.row_group(group).column(position)
                stats = column.statistics
                if column.num_values == 0:
                    continue
                if stats is None or not stats.has_min_max:
                    ranges = None
                    break
                lo = min(lo, float(stats.min))
                hi = max(hi, float(stats.max))
            if ranges is None:
                break
            ranges.append((lo, hi))
        if ranges is None:
            table = pq.read_table(path, columns=["x", "y"])
            ranges = []
            for name in ("x", "y"):
                values = table.column(name).to_numpy().astype(np.float64)
                values = values[np.isfinite(values)]
                ranges.append(
                    (float(values.min()), float(values.max()))
                    if values.size
                    else (math.inf, -math.inf)
                )
        for axis in range(2):
            low[axis] = min(low[axis], ranges[axis][0])
            high[axis] = max(high[axis], ranges[axis][1])
    if not all(math.isfinite(v) for v in (*low, *high)):
        raise ValueError("the points hold no finite coordinate")
    return low[0], low[1], high[0], high[1]


@dataclass(frozen=True)
class TissueGrid:
    """Regular micron grid covering every transcript."""

    origin_um: tuple[float, float]
    bin_um: float
    shape: tuple[int, int]


def tissue_grid(files: Sequence[Path], bin_um: float) -> TissueGrid:
    """Return a grid aligned to multiples of ``bin_um`` covering the points."""
    x0, y0, x1, y1 = files_bounds(files)
    gx0 = math.floor(x0 / bin_um) * bin_um
    gy0 = math.floor(y0 / bin_um) * bin_um
    n_x = max(1, int(math.floor((x1 - gx0) / bin_um)) + 1)
    n_y = max(1, int(math.floor((y1 - gy0) / bin_um)) + 1)
    return TissueGrid(
        origin_um=(float(gx0), float(gy0)), bin_um=float(bin_um), shape=(n_y, n_x)
    )


def _iter_batches(
    path: Path, columns: Sequence[str], batch_rows: int
) -> Iterator[pa.RecordBatch]:
    parquet = pq.ParquetFile(path)
    try:
        available = set(parquet.schema_arrow.names)
        wanted = [column for column in columns if column in available]
        yield from parquet.iter_batches(batch_size=int(batch_rows), columns=wanted)
    finally:
        parquet.close()


def _float_column(batch: pa.RecordBatch, name: str) -> np.ndarray:
    column = batch.column(batch.schema.get_field_index(name))
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)


def _map_files(
    files: Sequence[Path], work: Callable[[Path], None], n_workers: int
) -> None:
    if n_workers <= 1 or len(files) <= 1:
        for path in files:
            work(path)
        return
    with ThreadPoolExecutor(max_workers=min(n_workers, len(files))) as pool:
        for _ in pool.map(work, files):
            pass


def stream_density(
    files: Sequence[Path],
    grid: TissueGrid,
    *,
    batch_rows: int,
    n_workers: int,
) -> np.ndarray:
    """Bin every transcript of the files on the grid (``int64`` counts)."""
    n_y, n_x = grid.shape
    total = np.zeros(n_y * n_x, dtype=np.int64)
    lock = threading.Lock()

    def _one(path: Path) -> None:
        local = np.zeros(n_y * n_x, dtype=np.int64)
        for batch in _iter_batches(path, ["x", "y"], batch_rows):
            if batch.num_rows == 0:
                continue
            ix = bin_indices(
                _float_column(batch, "x"), grid.origin_um[0], grid.bin_um, n_x
            )
            iy = bin_indices(
                _float_column(batch, "y"), grid.origin_um[1], grid.bin_um, n_y
            )
            inside = (ix >= 0) & (iy >= 0)
            local += np.bincount(iy[inside] * n_x + ix[inside], minlength=n_y * n_x)
        with lock:
            total[:] += local

    _map_files(files, _one, n_workers)
    return total.reshape(n_y, n_x)


def build_tissue_mask(
    files: Sequence[Path],
    *,
    sample: SpatialGeneAnalysisSampleConfig,
    config: SpatialGeneAnalysisConfig,
    settings: RasterEngineSettings = DEFAULT_SETTINGS,
) -> TissueMask:
    """Return the sample's tissue support: annotations if any, else density."""
    grid = tissue_grid(files, float(settings.tissue_bin_um))
    if _has_tissue_annotations(sample):
        polygon = build_analysis_tissue_polygon(sample)
        return polygon_tissue_mask(
            polygon, origin_um=grid.origin_um, bin_um=grid.bin_um, shape=grid.shape
        )
    counts = stream_density(
        files, grid, batch_rows=_batch_rows(config), n_workers=settings.workers()
    )
    mask = density_tissue_mask(
        counts,
        origin_um=grid.origin_um,
        bin_um=grid.bin_um,
        smoothing_sigma_um=settings.tissue_smoothing_sigma_um,
        closing_radius_um=settings.tissue_closing_radius_um,
        min_component_area_um2=settings.tissue_min_component_area_um2,
        max_threshold_fraction=settings.tissue_max_threshold_fraction,
    )
    mask.parameters["n_transcripts_binned"] = int(counts.sum())
    mask.density = counts
    return mask


def _has_tissue_annotations(sample: SpatialGeneAnalysisSampleConfig) -> bool:
    return any(
        path is not None
        for path in (
            sample.annotation_path,
            sample.pial_boundary_path,
            sample.side_boundary_path,
            sample.ribbon_path,
        )
    )


# --------------------------------------------------------------------------
# Signed-distance rasters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DistanceEncoding:
    """``uint16`` encoding of boundary distances.

    Attributes:
        quantum_um: Microns per code unit.
        exact_range_um: Distances up to this value are exact; beyond it the
            code is :data:`DISTANCE_CLIP_CODE` (decoded as infinity).
        halo_px: Halo of each distance tile, in pixels (covers the range).
    """

    quantum_um: float
    exact_range_um: float
    halo_px: int

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Return distances in microns (``inf`` for clipped codes)."""
        values = np.asarray(codes, dtype=np.float64) * self.quantum_um
        values[np.asarray(codes) >= DISTANCE_CLIP_CODE] = np.inf
        return values


def distance_encoding(
    config: SpatialGeneAnalysisConfig, um_per_px: float
) -> DistanceEncoding:
    """Return the encoding covering every distance a statistic needs."""
    edges = np.asarray(config.signed_distance_edges_um, dtype=float)
    exact_range = float(
        max(
            abs(edges[0]),
            abs(edges[-1]),
            config.pericellular_distance_um,
            config.membrane_distance_um,
        )
    )
    quantum = max(0.01, exact_range / 60_000.0)
    halo_px = int(math.ceil(exact_range / um_per_px)) + 2
    return DistanceEncoding(
        quantum_um=quantum, exact_range_um=exact_range, halo_px=halo_px
    )


def boundary_pixels(labels: np.ndarray) -> np.ndarray:
    """Return pixels with an 8-neighbour of a different label (both sides).

    Diagonal neighbours count, so the pixels touching a label corner are edge
    pixels too; with them, the distance of every point to the nearest label
    change is within one pixel of the raster value (a 4-neighbour edge set
    overestimates distances diagonal to corners by up to 1.5 pixels).
    """
    window = np.asarray(labels)
    edges = np.zeros(window.shape, dtype=bool)
    # (pixel a, its neighbour b) slices for the right, down, down-right and
    # down-left neighbours; both pixels of a differing pair are edges.
    pairs = (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None))),
        ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1))),
    )
    for first, second in pairs:
        differs = window[first] != window[second]
        edges[first] |= differs
        edges[second] |= differs
    return edges


def build_distance_raster(
    labels: Any,
    out: np.ndarray,
    *,
    um_per_px: float,
    encoding: DistanceEncoding,
    tile_px: int,
    n_workers: int,
    tile_needed: Callable[[int, int, int, int], bool] | None = None,
) -> dict[str, int]:
    """Fill ``out`` with each pixel's encoded distance to the nearest label edge.

    Each output tile of ``tile_px`` reads its labels with a halo of
    ``encoding.halo_px`` and runs an exact Euclidean distance transform of
    the non-edge pixels there; the result is exact for every distance up to
    the halo (a nearer edge would lie inside the window). The distance of a
    pixel is the distance from its centre to the nearest edge pixel's centre
    plus half a pixel (an edge pixel touches the label change, half a pixel
    from its centre).

    Args:
        labels: Level-0 label array (numpy, zarr), ``[row, column]``.
        out: ``uint16`` array of the same shape (a memmap) to fill.
        um_per_px: Microns per pixel.
        encoding: Distance encoding.
        tile_px: Output tile edge.
        n_workers: Threads.
        tile_needed: ``(row0, row1, col0, col1) -> bool``; tiles answering
            ``False`` (no tissue) are filled with the clip code without work.

    Returns:
        Counters: tiles computed, skipped (no tissue) and without any edge.
    """
    height, width = (int(v) for v in labels.shape)
    if tuple(out.shape) != (height, width):
        raise ValueError(f"distance raster shape {out.shape} != labels {labels.shape}")
    halo = int(encoding.halo_px)
    tiles = [
        (r0, min(height, r0 + tile_px), c0, min(width, c0 + tile_px))
        for r0 in range(0, height, tile_px)
        for c0 in range(0, width, tile_px)
    ]
    counters = {
        "tiles": len(tiles),
        "computed": 0,
        "skipped_no_tissue": 0,
        "no_edges": 0,
    }
    lock = threading.Lock()

    def _one(tile: tuple[int, int, int, int]) -> None:
        r0, r1, c0, c1 = tile
        if tile_needed is not None and not tile_needed(r0, r1, c0, c1):
            out[r0:r1, c0:c1] = DISTANCE_CLIP_CODE
            with lock:
                counters["skipped_no_tissue"] += 1
            return
        wr0, wr1 = max(0, r0 - halo), min(height, r1 + halo)
        wc0, wc1 = max(0, c0 - halo), min(width, c1 + halo)
        window = np.asarray(labels[wr0:wr1, wc0:wc1])
        edges = boundary_pixels(window)
        del window
        if not edges.any():
            out[r0:r1, c0:c1] = DISTANCE_CLIP_CODE
            with lock:
                counters["no_edges"] += 1
            return
        distance = ndimage.distance_transform_edt(~edges)
        del edges
        core = distance[r0 - wr0 : r1 - wr0, c0 - wc0 : c1 - wc0]
        core_um = (core + 0.5) * float(um_per_px)
        del distance
        codes = np.minimum(np.rint(core_um / encoding.quantum_um), MAX_DISTANCE_CODE)
        codes = codes.astype(np.uint16)
        codes[core_um > encoding.exact_range_um] = DISTANCE_CLIP_CODE
        out[r0:r1, c0:c1] = codes
        with lock:
            counters["computed"] += 1

    if n_workers <= 1:
        for tile in tiles:
            _one(tile)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for _ in pool.map(_one, tiles):
                pass
    return counters


# --------------------------------------------------------------------------
# Streaming accumulation
# --------------------------------------------------------------------------


@dataclass
class RasterInputs:
    """Everything the streaming passes read.

    Attributes:
        files: Parquet files of the transcripts (``x``, ``y`` in microns, a
            feature column and optionally ``is_gene``).
        cell_labels: Level-0 cell label array.
        nucleus_labels: Level-0 nucleus label array (same grid).
        frame: Micron -> label pixel mapping.
        convention: Pixel rule.
        platform: Platform name (control-name rules).
        feature_column: Feature name column.
    """

    files: list[Path]
    cell_labels: Any
    nucleus_labels: Any
    frame: PixelFrame
    convention: PixelConvention
    platform: str
    feature_column: str = "feature_name"


class _GeneRegistry:
    """Thread-safe feature name -> code registry with control-name rules.

    Every feature name gets a code (controls too: their accumulators simply
    stay empty when controls are dropped). A row is a control when its
    ``is_gene`` value says so, else when its name matches a control rule of
    :mod:`merxen.control_features` (the order of
    :func:`~merxen.control_features.classify_control_transcripts`).
    """

    def __init__(self, platform: str, *, drop_controls: bool) -> None:
        self.platform = platform
        self.drop_controls = drop_controls
        self.names: list[str] = []
        self._info: dict[str, tuple[int, bool]] = {}
        self._lock = threading.Lock()

    def _entries(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return the codes and control-name flags of dictionary entries.

        New names are classified in one vectorised call and registered under
        the lock (first-seen order).
        """
        info = self._info
        new = [name for name in dict.fromkeys(names) if name not in info]
        if new:
            rules = control_transcript_mask(new, platform=self.platform)
            with self._lock:
                for name, rule in zip(new, rules, strict=True):
                    if name not in info:
                        info[name] = (len(self.names), bool(rule))
                        self.names.append(name)
        n_entries = max(1, len(names))
        codes = np.zeros(n_entries, dtype=np.int64)
        rules_out = np.zeros(n_entries, dtype=bool)
        for position, name in enumerate(names):
            codes[position], rules_out[position] = info[name]
        return codes, rules_out

    def code_of(self, name: str) -> int | None:
        """Return the code of a known name, else ``None``."""
        info = self._info.get(name)
        return None if info is None else info[0]

    def batch_codes(
        self, batch: pa.RecordBatch, feature_column: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(codes, control, missing)`` for each row of a batch.

        ``codes`` is ``-1`` for missing names and, when controls are dropped,
        for control rows.
        """
        column = batch.column(batch.schema.get_field_index(feature_column))
        if not pa.types.is_dictionary(column.type):
            column = column.dictionary_encode()
        indices_arrow = column.indices
        missing = np.asarray(indices_arrow.is_null().to_numpy(zero_copy_only=False))
        indices = np.asarray(
            indices_arrow.fill_null(0).to_numpy(zero_copy_only=False), dtype=np.int64
        )
        dictionary = [str(value) for value in column.dictionary.to_pylist()]
        entry_codes, entry_rule = self._entries(dictionary)
        codes = entry_codes[indices]
        control = entry_rule[indices]
        if "is_gene" in batch.schema.names:
            flags = batch.column(batch.schema.get_field_index("is_gene"))
            known = ~np.asarray(flags.is_null().to_numpy(zero_copy_only=False))
            is_gene = np.asarray(
                flags.fill_null(False).to_numpy(zero_copy_only=False), dtype=bool
            )
            control = np.where(known, ~is_gene, control)
        control &= ~missing
        if self.drop_controls:
            codes[control] = -1
        codes[missing] = -1
        return codes, control, missing


def lookup_many(
    column: np.ndarray,
    row: np.ndarray,
    arrays: dict[str, Any],
    *,
    cache: BlockCache | None = None,
    block_px: int = DEFAULT_BLOCK_PX,
) -> dict[str, np.ndarray]:
    """Look pixels up in several arrays of one grid, sorting by block once.

    Same values as :func:`~merxen.enrichment.label_lookup.lookup_labels` per
    array (0 off the grid), with one block ordering shared by every array.
    """
    shapes = {tuple(array.shape) for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"arrays differ in shape: {shapes}")
    height, width = (int(v) for v in next(iter(shapes)))
    out = {name: np.zeros(column.shape, dtype=np.int64) for name in arrays}
    inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    if not inside.any():
        return out
    positions = np.flatnonzero(inside)
    n_block_cols = (width + block_px - 1) // block_px
    block_ids = (row[positions] // block_px) * n_block_cols + (
        column[positions] // block_px
    )
    order = np.argsort(block_ids, kind="stable")
    sorted_ids = block_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    stops = np.r_[starts[1:], sorted_ids.size]
    for start, stop in zip(starts, stops, strict=True):
        by, bx = divmod(int(sorted_ids[start]), n_block_cols)
        y0, x0 = by * block_px, bx * block_px
        y1, x1 = min(height, y0 + block_px), min(width, x0 + block_px)
        members = positions[order[start:stop]]
        local_rows = row[members] - y0
        local_cols = column[members] - x0
        for name, array in arrays.items():
            if cache is None:
                block = np.asarray(array[y0:y1, x0:x1])
            else:
                block = cache.get(
                    (name, block_px, y0, x0),
                    partial(_read_array_block, array, y0, y1, x0, x1),
                )
            out[name][members] = block[local_rows, local_cols]
    return out


def _read_array_block(array: Any, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    return np.asarray(array[y0:y1, x0:x1])


@dataclass
class _Accumulators:
    """Per-gene streaming accumulators (grown as genes appear)."""

    n_distance_bins: int
    n_fine_bins: int
    capacity: int = 0
    compartments: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    distance_bins: dict[str, np.ndarray] = field(default_factory=dict)
    fine: dict[str, np.ndarray] = field(default_factory=dict)
    n_pericellular: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    n_membrane: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    totals: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for boundary in SIGNED_DISTANCE_BOUNDARIES:
            self.distance_bins[boundary] = np.zeros((0, self.n_distance_bins), np.int64)
            self.fine[boundary] = np.zeros((0, self.n_fine_bins), np.uint32)

    def ensure(self, n_genes: int) -> None:
        if n_genes <= self.capacity:
            return
        new = max(int(n_genes), int(self.capacity * 1.5))

        def _grow(array: np.ndarray) -> np.ndarray:
            grown = np.zeros((new, *array.shape[1:]), dtype=array.dtype)
            grown[: array.shape[0]] = array
            return grown

        self.compartments = _grow(self.compartments)
        self.n_pericellular = _grow(self.n_pericellular)
        self.n_membrane = _grow(self.n_membrane)
        for boundary in SIGNED_DISTANCE_BOUNDARIES:
            self.distance_bins[boundary] = _grow(self.distance_bins[boundary])
            self.fine[boundary] = _grow(self.fine[boundary])
        self.capacity = new

    def add_total(self, name: str, value: int) -> None:
        self.totals[name] = self.totals.get(name, 0) + int(value)


def _fine_bin_codes(
    distance_um: np.ndarray, low: float, bin_um: float, n_inner: int
) -> np.ndarray:
    """Return fine histogram bins: 0 underflow, 1..n_inner, n_inner + 1 overflow."""
    with np.errstate(invalid="ignore"):
        raw = np.floor((distance_um - low) / bin_um)
    raw = np.where(np.isfinite(raw), raw, np.where(distance_um > 0, n_inner, -1))
    return (np.clip(raw, -1, n_inner) + 1).astype(np.int64)


@dataclass
class _PassSettings:
    edges: np.ndarray
    fine_low: float
    fine_bin_um: float
    n_fine_inner: int
    pericellular_um: float
    membrane_um: float
    batch_rows: int


def _signed(
    codes: np.ndarray, inside: np.ndarray, encoding: DistanceEncoding
) -> np.ndarray:
    distance = encoding.decode(codes)
    return np.where(inside, distance, -distance)


def accumulate_gene_patterns(
    inputs: RasterInputs,
    *,
    tissue: TissueMask,
    cell_distance: np.ndarray,
    nucleus_distance: np.ndarray,
    encoding: DistanceEncoding,
    config: SpatialGeneAnalysisConfig,
    settings: RasterEngineSettings = DEFAULT_SETTINGS,
) -> tuple[_GeneRegistry, _Accumulators]:
    """Stream every transcript into the per-gene accumulators.

    Returns:
        The gene registry (codes in first-seen order) and the accumulators.
    """
    edges = np.asarray(config.signed_distance_edges_um, dtype=float)
    fine_bin = float(settings.median_bin_um)
    n_inner = int(math.ceil((edges[-1] - edges[0]) / fine_bin))
    pass_settings = _PassSettings(
        edges=edges,
        fine_low=float(edges[0]),
        fine_bin_um=fine_bin,
        n_fine_inner=n_inner,
        pericellular_um=float(config.pericellular_distance_um),
        membrane_um=float(config.membrane_distance_um),
        batch_rows=_batch_rows(config),
    )
    registry = _GeneRegistry(
        inputs.platform, drop_controls=config.drop_control_features
    )
    acc = _Accumulators(n_distance_bins=len(edges) + 1, n_fine_bins=n_inner + 2)
    lock = threading.Lock()
    cache = (
        BlockCache(settings.block_cache_bytes)
        if settings.block_cache_bytes > 0
        else None
    )
    arrays = {
        "cell": inputs.cell_labels,
        "nucleus": inputs.nucleus_labels,
        "cell_distance": cell_distance,
        "nucleus_distance": nucleus_distance,
    }
    height, width = (int(v) for v in inputs.cell_labels.shape)
    columns = ["x", "y", inputs.feature_column, "is_gene"]

    def _batch(batch: pa.RecordBatch) -> None:
        n_rows = batch.num_rows
        x = _float_column(batch, "x")
        y = _float_column(batch, "y")
        finite = np.isfinite(x) & np.isfinite(y)
        codes, control, missing = registry.batch_codes(batch, inputs.feature_column)
        in_tissue = np.zeros(n_rows, dtype=bool)
        in_tissue[finite] = tissue.contains(x[finite], y[finite])
        # Rows of analysed features (genes; controls too when they are kept).
        gene_rows = finite & (codes >= 0)
        valid = gene_rows & in_tissue
        column, row = pixel_indices(x, y, inputs.frame, convention=inputs.convention)
        lookups = lookup_many(column[gene_rows], row[gene_rows], arrays, cache=cache)
        on_grid = (
            (column[gene_rows] >= 0)
            & (column[gene_rows] < width)
            & (row[gene_rows] >= 0)
            & (row[gene_rows] < height)
        )
        for name in ("cell_distance", "nucleus_distance"):
            lookups[name][~on_grid] = DISTANCE_CLIP_CODE
        tissue_of_gene_rows = in_tissue[gene_rows]
        keep = valid[gene_rows]
        cell = lookups["cell"][keep]
        nucleus = lookups["nucleus"][keep]
        gene = codes[gene_rows][keep]
        compartment = np.full(gene.size, COMPARTMENT_CODES["extracellular"], np.int64)
        compartment[cell > 0] = COMPARTMENT_CODES["cytoplasmic"]
        compartment[nucleus > 0] = COMPARTMENT_CODES["nuclear"]
        signed = {
            "cell": _signed(lookups["cell_distance"][keep], cell > 0, encoding),
            "nucleus": _signed(
                lookups["nucleus_distance"][keep], nucleus > 0, encoding
            ),
        }
        n_genes = len(registry.names)
        partial_compartments = np.bincount(
            gene * 3 + compartment, minlength=n_genes * 3
        ).reshape(n_genes, 3)
        partial_bins: dict[str, np.ndarray] = {}
        partial_fine: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        nb = len(pass_settings.edges) + 1
        nf = pass_settings.n_fine_inner + 2
        for boundary in SIGNED_DISTANCE_BOUNDARIES:
            bins = np.searchsorted(pass_settings.edges, signed[boundary], side="right")
            partial_bins[boundary] = np.bincount(
                gene * nb + bins, minlength=n_genes * nb
            ).reshape(n_genes, nb)
            fine = _fine_bin_codes(
                signed[boundary],
                pass_settings.fine_low,
                pass_settings.fine_bin_um,
                pass_settings.n_fine_inner,
            )
            partial_fine[boundary] = np.unique(gene * nf + fine, return_counts=True)
        cell_signed = signed["cell"]
        pericellular = (cell_signed < 0) & (
            cell_signed >= -pass_settings.pericellular_um
        )
        membrane = (compartment == COMPARTMENT_CODES["cytoplasmic"]) & (
            cell_signed <= pass_settings.membrane_um
        )
        partial_peri = np.bincount(gene[pericellular], minlength=n_genes)
        partial_membrane = np.bincount(gene[membrane], minlength=n_genes)
        on_cell_all = lookups["cell"] > 0
        stats = {
            "n_input": n_rows,
            "n_invalid_coordinates": int((~finite).sum()),
            "n_missing_feature_names": int((finite & missing).sum()),
            "n_outside_tissue": int((finite & ~missing & ~in_tissue).sum()),
            "n_controls_excluded": int((finite & in_tissue & control).sum())
            if config.drop_control_features
            else 0,
            "n_control_transcripts_in_tissue": int(
                (finite & in_tissue & control).sum()
            ),
            "n_used": int(gene.size),
            "n_gene_transcripts_finite": int(gene_rows.sum()),
            "n_gene_transcripts_on_cell_label": int(on_cell_all.sum()),
            "n_gene_transcripts_on_cell_label_in_tissue": int(
                (on_cell_all & tissue_of_gene_rows).sum()
            ),
            "n_gene_transcripts_on_nucleus_label": int((lookups["nucleus"] > 0).sum()),
            "n_gene_transcripts_off_label_grid": int((~on_grid).sum()),
        }
        with lock:
            acc.ensure(n_genes)
            acc.compartments[:n_genes] += partial_compartments
            acc.n_pericellular[:n_genes] += partial_peri
            acc.n_membrane[:n_genes] += partial_membrane
            for boundary in SIGNED_DISTANCE_BOUNDARIES:
                acc.distance_bins[boundary][:n_genes] += partial_bins[boundary]
                keys, counts = partial_fine[boundary]
                flat = acc.fine[boundary].reshape(-1)
                flat[keys] += counts.astype(np.uint32)
            for name, value in stats.items():
                acc.add_total(name, value)

    def _file(path: Path) -> None:
        for batch in _iter_batches(path, columns, pass_settings.batch_rows):
            if batch.num_rows:
                _batch(batch)

    _map_files(inputs.files, _file, settings.workers())
    acc.ensure(len(registry.names))
    return registry, acc


def _fine_medians(
    fine: np.ndarray, low: float, bin_um: float, n_inner: int, high: float
) -> np.ndarray:
    """Return the median of each row of a fine histogram (bin centres).

    For an even count the two middle values are averaged, as ``np.median``
    does. Underflow and overflow bins report the range edges.
    """
    centres = np.empty(n_inner + 2, dtype=np.float64)
    centres[0] = low
    centres[1:-1] = low + (np.arange(n_inner) + 0.5) * bin_um
    centres[-1] = high
    medians = np.full(fine.shape[0], np.nan)
    chunk = 1024
    for start in range(0, fine.shape[0], chunk):
        block = np.cumsum(fine[start : start + chunk], axis=1, dtype=np.int64)
        totals = block[:, -1]
        has = totals > 0
        lower_rank = (totals - 1) // 2
        upper_rank = totals // 2
        lower = np.argmax(block > lower_rank[:, None], axis=1)
        upper = np.argmax(block > upper_rank[:, None], axis=1)
        values = (centres[lower] + centres[upper]) / 2.0
        medians[start : start + chunk] = np.where(has, values, np.nan)
    return medians


def counts_from_accumulators(
    registry: _GeneRegistry,
    acc: _Accumulators,
    *,
    config: SpatialGeneAnalysisConfig,
    settings: RasterEngineSettings = DEFAULT_SETTINGS,
) -> GenePatternCounts:
    """Turn the accumulators into sorted per-gene counts (genes seen in tissue)."""
    edges = np.asarray(config.signed_distance_edges_um, dtype=float)
    n_genes = len(registry.names)
    n_by_gene = acc.compartments[:n_genes].sum(axis=1)
    order = sorted(
        (code for code in range(n_genes) if n_by_gene[code] > 0),
        key=lambda code: registry.names[code],
    )
    index = np.asarray(order, dtype=np.int64)
    fine_bin = float(settings.median_bin_um)
    n_inner = acc.n_fine_bins - 2
    medians = {
        boundary: _fine_medians(
            acc.fine[boundary][index],
            float(edges[0]),
            fine_bin,
            n_inner,
            float(edges[-1]),
        )
        for boundary in SIGNED_DISTANCE_BOUNDARIES
    }
    zeros = np.zeros(index.size, dtype=np.int64)
    return GenePatternCounts(
        gene_names=[registry.names[code] for code in order],
        compartment_counts=acc.compartments[index].copy(),
        distance_bin_counts={
            boundary: acc.distance_bins[boundary][index].copy()
            for boundary in SIGNED_DISTANCE_BOUNDARIES
        },
        median_signed_cell_distance_um=medians["cell"],
        median_signed_nucleus_distance_um=medians["nucleus"],
        n_pericellular=acc.n_pericellular[index].copy(),
        n_membrane_proximal=acc.n_membrane[index].copy(),
        n_cell_overlap_ambiguous=zeros,
        n_nucleus_overlap_ambiguous=zeros.copy(),
    )


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class RasterTranscriptPatternResults:
    """Tables and records of one label-raster run.

    Attributes:
        summary: Per-gene summary (as the polygon engine, plus
            ``paircorr_status``).
        signed_distance: Long signed-distance table.
        paircorr: Empty pair-correlation table (engine does not compute it).
        rankings: Per-metric rankings.
        counts: The per-gene counts behind the tables.
        totals: Transcript counters of the streaming pass.
        tissue: Tissue support.
        record: Engine provenance for the manifest.
        inputs: Inputs of the passes (reused by the diagnostics).
        registry: Gene registry of the pass.
        settings: Engine settings of the run.
    """

    summary: pd.DataFrame
    signed_distance: pd.DataFrame
    paircorr: pd.DataFrame
    rankings: pd.DataFrame
    counts: GenePatternCounts
    totals: dict[str, int]
    tissue: TissueMask
    record: dict[str, Any]
    inputs: RasterInputs
    registry: _GeneRegistry
    settings: RasterEngineSettings = DEFAULT_SETTINGS


def _batch_rows(config: SpatialGeneAnalysisConfig) -> int:
    return int(config.transcript_chunk_rows or DEFAULT_BATCH_ROWS)


def empty_paircorr_table() -> pd.DataFrame:
    """Return the empty pair-correlation table of this engine."""
    return pd.DataFrame(
        {column: pd.Series(dtype="float64") for column in PAIRCORR_COLUMNS}
    ).astype({"gene": "object", "null_model": "object", "band_label": "object"})


def run_label_raster_patterns(
    inputs: RasterInputs,
    *,
    tissue: TissueMask,
    config: SpatialGeneAnalysisConfig,
    scratch_dir: Path,
    sample_id: str,
    settings: RasterEngineSettings = DEFAULT_SETTINGS,
) -> RasterTranscriptPatternResults:
    """Run the label-raster engine on prepared inputs.

    Args:
        inputs: Transcript files and label arrays.
        tissue: Tissue support.
        config: Analysis parameters.
        scratch_dir: Directory for the distance memmaps (removed by the
            caller).
        sample_id: Sample name for logs.
        settings: Engine settings.

    Returns:
        The results.
    """
    n_workers = settings.workers()
    um_per_px = float(inputs.frame.um_per_px)
    encoding = distance_encoding(config, um_per_px)
    height, width = (int(v) for v in inputs.cell_labels.shape)
    if tuple(inputs.nucleus_labels.shape) != (height, width):
        raise ValueError(
            f"cell labels {inputs.cell_labels.shape} and nucleus labels "
            f"{inputs.nucleus_labels.shape} differ in shape"
        )
    tile_px = int(settings.distance_tile_px)
    ox, oy = inputs.frame.origin_um

    def _tile_needed(r0: int, r1: int, c0: int, c1: int) -> bool:
        return tissue.overlaps_window(
            ox + c0 * um_per_px,
            oy + r0 * um_per_px,
            ox + c1 * um_per_px,
            oy + r1 * um_per_px,
        )

    scratch_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    distance_counters: dict[str, dict[str, int]] = {}
    memmaps: dict[str, np.ndarray] = {}
    for boundary, labels in (
        ("cell", inputs.cell_labels),
        ("nucleus", inputs.nucleus_labels),
    ):
        started = time.monotonic()
        path = scratch_dir / f"{boundary}_distance_u16.dat"
        memmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=(height, width))
        distance_counters[boundary] = build_distance_raster(
            labels,
            memmap,
            um_per_px=um_per_px,
            encoding=encoding,
            tile_px=tile_px,
            n_workers=n_workers,
            tile_needed=_tile_needed,
        )
        memmap.flush()
        memmaps[boundary] = memmap
        timings[f"{boundary}_distance_s"] = time.monotonic() - started
        log_status(
            f"[{sample_id}] {boundary} distance raster: {distance_counters[boundary]} "
            f"in {timings[f'{boundary}_distance_s']:.0f} s"
        )
    scratch_bytes = sum(path.stat().st_size for path in scratch_dir.glob("*.dat"))

    started = time.monotonic()
    registry, acc = accumulate_gene_patterns(
        inputs,
        tissue=tissue,
        cell_distance=memmaps["cell"],
        nucleus_distance=memmaps["nucleus"],
        encoding=encoding,
        config=config,
        settings=settings,
    )
    timings["accumulate_s"] = time.monotonic() - started
    totals = dict(acc.totals)
    log_status(f"[{sample_id}] label-raster pass: {totals}")
    if totals.get("n_used", 0) == 0:
        raise ValueError(f"[{sample_id}] No transcripts remained inside tissue.")
    memmaps.clear()

    counts = counts_from_accumulators(registry, acc, config=config, settings=settings)
    del acc
    summary = summarize_gene_pattern_counts(counts, config=config)
    signed_distance = signed_distance_enrichment_from_counts(counts, config=config)
    summary = add_signed_distance_summary(summary, signed_distance)
    summary["paircorr_status"] = PAIRCORR_STATUS
    summary["pattern_label"] = classify_gene_patterns(summary)
    rankings = rank_transcript_patterns(summary, top_n=config.top_n)
    edges = np.asarray(config.signed_distance_edges_um, dtype=float)
    record = {
        "engine": ENGINE_NAME,
        "paircorr": {
            "status": PAIRCORR_STATUS,
            "reason": (
                "pair correlation is off on the label-raster engine (AT-8 "
                "decision); an all-transcript method is a follow-up"
            ),
        },
        "pixel_convention": inputs.convention,
        "pixel_precision": inputs.frame.precision,
        "um_per_px": um_per_px,
        "label_shape": [height, width],
        "compartment_rule": (
            "nuclear if the nucleus label is non-zero, else cytoplasmic if the "
            "cell label is non-zero, else extracellular"
        ),
        "overlap_counts": "0 by construction (label rasters cannot overlap)",
        "distance_method": (
            "tiled exact EDT (scipy.ndimage.distance_transform_edt) to label-edge "
            "pixels + 0.5 px; positive on a label, negative off it"
        ),
        "distance_tile_px": tile_px,
        "distance_halo_px": encoding.halo_px,
        "distance_exact_range_um": encoding.exact_range_um,
        "distance_quantum_um": encoding.quantum_um,
        "distance_tiles": distance_counters,
        "median_bin_um": float(settings.median_bin_um),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in dataclasses.asdict(settings).items()
        },
        "median_range_um": [float(edges[0]), float(edges[-1])],
        "median_rule": (
            "fine-histogram bin centres; a median beyond the outermost edge is "
            "reported at that edge"
        ),
        "scratch_bytes": int(scratch_bytes),
        "n_workers": n_workers,
        "batch_rows": _batch_rows(config),
        "timings_s": timings,
        "tissue": tissue.record(),
        "subsampling": "none: every finite, in-tissue, non-control transcript is used",
    }
    return RasterTranscriptPatternResults(
        summary=summary,
        signed_distance=signed_distance,
        paircorr=empty_paircorr_table(),
        rankings=rankings,
        counts=counts,
        totals=totals,
        tissue=tissue,
        record=record,
        inputs=inputs,
        registry=registry,
        settings=settings,
    )


def run_raster_transcript_pattern_analysis(
    *,
    sample: SpatialGeneAnalysisSampleConfig,
    config: SpatialGeneAnalysisConfig,
    settings: RasterEngineSettings = DEFAULT_SETTINGS,
    points_key: str = DEFAULT_POINTS_KEY,
) -> RasterTranscriptPatternResults:
    """Run the label-raster engine on one sample's latest store.

    The store is only read. The distance memmaps live in a temporary
    directory under ``settings.scratch_dir`` (default: beside the output
    directory, i.e. the Nextflow task directory) and are removed at the end.

    Raises:
        ValueError: If the layer has no label elements.
    """
    zarr_path = Path(sample.zarr_path)
    layer = resolve_label_layer(
        zarr_path, sample.platform, sample.segmentation, points_key=points_key
    )
    files = points_files(zarr_path, points_key)
    started = time.monotonic()
    tissue = build_tissue_mask(files, sample=sample, config=config, settings=settings)
    tissue_s = time.monotonic() - started
    log_status(
        f"[{sample.sample_id}] tissue support ({tissue.method}): "
        f"{tissue.area_um2 / 1e6:.2f} mm2"
    )
    inputs = RasterInputs(
        files=files,
        cell_labels=open_label_level0(zarr_path, layer.cell_label_key),
        nucleus_labels=open_label_level0(zarr_path, layer.nucleus_label_key),
        frame=PixelFrame(layer.um_per_px),
        convention=layer.convention,
        platform=sample.platform,
    )
    scratch_root = Path(
        settings.scratch_dir
        if settings.scratch_dir is not None
        else Path(config.output_dir).resolve().parent
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(prefix="merxen_transcript_raster_", dir=scratch_root)
    )
    try:
        results = run_label_raster_patterns(
            inputs,
            tissue=tissue,
            config=config,
            scratch_dir=scratch,
            sample_id=sample.sample_id,
            settings=settings,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    results.record["timings_s"]["tissue_s"] = tissue_s
    results.record["cell_labels"] = layer.cell_label_key
    results.record["nucleus_labels"] = layer.nucleus_label_key
    results.record["points_key"] = points_key
    results.record["n_points_files"] = len(files)
    return results


# --------------------------------------------------------------------------
# Diagnostics (rasterised over every transcript of the selected genes)
# --------------------------------------------------------------------------


@dataclass
class GeneDiagnosticRaster:
    """Density rasters of one gene for its diagnostic figure.

    Attributes:
        gene: Gene name.
        tissue_counts: In-tissue transcripts per tissue-grid bin.
        window_origin_um: ``(x0, y0)`` of the densest window.
        window_um: Window edge.
        window_bin_um: Bin edge of the window rasters.
        window_counts: Compartment -> counts in the window (row = y).
        cell_edges: Label-edge pixels of the cell labels in the window.
        nucleus_edges: Label-edge pixels of the nucleus labels in the window.
        edges_extent: ``imshow`` extent of the edge rasters.
    """

    gene: str
    tissue_counts: np.ndarray
    window_origin_um: tuple[float, float]
    window_um: float
    window_bin_um: float
    window_counts: dict[str, np.ndarray]
    cell_edges: np.ndarray | None = None
    nucleus_edges: np.ndarray | None = None
    edges_extent: tuple[float, float, float, float] | None = None

    @property
    def window_extent(self) -> tuple[float, float, float, float]:
        """Return the ``imshow`` extent of the window rasters."""
        x0, y0 = self.window_origin_um
        n = self.window_counts[COMPARTMENTS[0]].shape[0]
        return (x0, x0 + n * self.window_bin_um, y0, y0 + n * self.window_bin_um)


def densest_window(counts: np.ndarray, window_bins: int) -> tuple[int, int]:
    """Return ``(row, column)`` of the top-left bin of the densest square window.

    Ties go to the first window in row-major order.
    """
    n_y, n_x = counts.shape
    w = max(1, min(int(window_bins), n_y, n_x))
    summed = np.zeros((n_y + 1, n_x + 1), dtype=np.int64)
    summed[1:, 1:] = np.cumsum(np.cumsum(counts.astype(np.int64), axis=0), axis=1)
    sums = summed[w:, w:] - summed[:-w, w:] - summed[w:, :-w] + summed[:-w, :-w]
    row, column = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(row), int(column)


def gene_diagnostic_rasters(
    results: RasterTranscriptPatternResults,
    genes: Sequence[str],
    *,
    config: SpatialGeneAnalysisConfig,
) -> dict[str, GeneDiagnosticRaster]:
    """Bin every in-tissue transcript of the selected genes for the figures.

    Two streaming passes: every file for the tissue-grid rasters (from which
    each gene's densest window is chosen), then only the files whose
    footers meet a window for the fine compartment rasters. No point is
    thinned or dropped.
    """
    inputs = results.inputs
    tissue = results.tissue
    wanted = [gene for gene in genes if gene in set(results.counts.gene_names)]
    if not wanted:
        return {}
    registry = results.registry
    slot_of_code = np.full(len(registry.names), -1, dtype=np.int64)
    for slot, gene in enumerate(wanted):
        code = registry.code_of(gene)
        if code is not None:
            slot_of_code[code] = slot
    n_slots = len(wanted)
    n_y, n_x = tissue.shape
    n_bins = n_y * n_x
    tissue_counts = np.zeros((n_slots, n_bins), dtype=np.int64)
    lock = threading.Lock()
    batch_rows = _batch_rows(config)
    n_workers = results.settings.workers()
    columns = ["x", "y", inputs.feature_column, "is_gene"]

    def _selected(batch: pa.RecordBatch) -> tuple[np.ndarray, ...] | None:
        codes, _, _ = registry.batch_codes(batch, inputs.feature_column)
        codes = np.where(codes < len(slot_of_code), codes, -1)
        slots = np.where(codes >= 0, slot_of_code[np.maximum(codes, 0)], -1)
        take = slots >= 0
        if not take.any():
            return None
        x = _float_column(batch, "x")[take]
        y = _float_column(batch, "y")[take]
        slots = slots[take]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y, slots = x[finite], y[finite], slots[finite]
        inside = tissue.contains(x, y)
        return x[inside], y[inside], slots[inside]

    def _tissue_pass(path: Path) -> None:
        local = np.zeros((n_slots, n_bins), dtype=np.int64)
        for batch in _iter_batches(path, columns, batch_rows):
            selected = _selected(batch) if batch.num_rows else None
            if selected is None:
                continue
            x, y, slots = selected
            ix, iy = tissue.bins(x, y)
            flat = slots * n_bins + iy * n_x + ix
            local += np.bincount(flat, minlength=n_slots * n_bins).reshape(
                n_slots, n_bins
            )
        with lock:
            tissue_counts[:] += local

    _map_files(inputs.files, _tissue_pass, n_workers)

    window_um = float(config.transcript_diagnostic_window_um)
    fine_um = float(results.settings.diagnostic_bin_um)
    n_fine = max(1, int(round(window_um / fine_um)))
    window_bins = max(1, int(round(window_um / tissue.bin_um)))
    origins = np.zeros((n_slots, 2), dtype=np.float64)
    for slot in range(n_slots):
        grid = tissue_counts[slot].reshape(n_y, n_x)
        row, column = densest_window(grid, window_bins)
        origins[slot] = (
            tissue.origin_um[0] + column * tissue.bin_um,
            tissue.origin_um[1] + row * tissue.bin_um,
        )
    window_counts = np.zeros((n_slots, len(COMPARTMENTS), n_fine * n_fine), np.int64)
    span = n_fine * fine_um
    window_files = [
        path for path in inputs.files if _file_meets_windows(path, origins, span)
    ]
    cache = BlockCache(512 * 1024**2)

    def _window_pass(path: Path) -> None:
        local = np.zeros_like(window_counts)
        for batch in _iter_batches(path, columns, batch_rows):
            selected = _selected(batch) if batch.num_rows else None
            if selected is None:
                continue
            x, y, slots = selected
            fx = np.floor((x - origins[slots, 0]) / fine_um).astype(np.int64)
            fy = np.floor((y - origins[slots, 1]) / fine_um).astype(np.int64)
            take = (fx >= 0) & (fx < n_fine) & (fy >= 0) & (fy < n_fine)
            if not take.any():
                continue
            x, y, slots, fx, fy = x[take], y[take], slots[take], fx[take], fy[take]
            column, row = pixel_indices(
                x, y, inputs.frame, convention=inputs.convention
            )
            cell = lookup_labels(
                column, row, inputs.cell_labels, cache=cache, cache_key="cell"
            )
            nucleus = lookup_labels(
                column, row, inputs.nucleus_labels, cache=cache, cache_key="nucleus"
            )
            compartment = np.full(x.size, COMPARTMENT_CODES["extracellular"], np.int64)
            compartment[cell > 0] = COMPARTMENT_CODES["cytoplasmic"]
            compartment[nucleus > 0] = COMPARTMENT_CODES["nuclear"]
            flat = (slots * len(COMPARTMENTS) + compartment) * (n_fine * n_fine) + (
                fy * n_fine + fx
            )
            local += np.bincount(flat, minlength=local.size).reshape(local.shape)
        with lock:
            window_counts[:] += local

    _map_files(window_files, _window_pass, n_workers)

    rasters: dict[str, GeneDiagnosticRaster] = {}
    for slot, gene in enumerate(wanted):
        x0, y0 = (float(v) for v in origins[slot])
        cell_edges, nucleus_edges, extent = _window_edges(inputs, x0, y0, span)
        rasters[gene] = GeneDiagnosticRaster(
            gene=gene,
            tissue_counts=tissue_counts[slot].reshape(n_y, n_x),
            window_origin_um=(x0, y0),
            window_um=span,
            window_bin_um=fine_um,
            window_counts={
                name: window_counts[slot, code].reshape(n_fine, n_fine)
                for code, name in enumerate(COMPARTMENTS)
            },
            cell_edges=cell_edges,
            nucleus_edges=nucleus_edges,
            edges_extent=extent,
        )
    return rasters


def _file_meets_windows(path: Path, origins: np.ndarray, span: float) -> bool:
    try:
        x0, y0, x1, y1 = files_bounds([path])
    except ValueError:
        return False
    return bool(
        np.any(
            (origins[:, 0] <= x1)
            & (origins[:, 0] + span >= x0)
            & (origins[:, 1] <= y1)
            & (origins[:, 1] + span >= y0)
        )
    )


def _window_edges(
    inputs: RasterInputs, x0: float, y0: float, span: float
) -> tuple[
    np.ndarray | None, np.ndarray | None, tuple[float, float, float, float] | None
]:
    """Return the label-edge pixels of both rasters inside a micron window."""
    ps = float(inputs.frame.um_per_px)
    ox, oy = inputs.frame.origin_um
    height, width = (int(v) for v in inputs.cell_labels.shape)
    c0 = max(0, int(math.floor((x0 - ox) / ps)))
    r0 = max(0, int(math.floor((y0 - oy) / ps)))
    c1 = min(width, int(math.ceil((x0 + span - ox) / ps)))
    r1 = min(height, int(math.ceil((y0 + span - oy) / ps)))
    if c0 >= c1 or r0 >= r1:
        return None, None, None
    cell = boundary_pixels(np.asarray(inputs.cell_labels[r0:r1, c0:c1]))
    nucleus = boundary_pixels(np.asarray(inputs.nucleus_labels[r0:r1, c0:c1]))
    extent = (ox + c0 * ps, ox + c1 * ps, oy + r0 * ps, oy + r1 * ps)
    return cell, nucleus, extent


def raster_transcript_counts(
    results: RasterTranscriptPatternResults,
) -> dict[str, int | float]:
    """Return the manifest ``transcript_counts`` of a label-raster run."""
    totals = results.totals
    compartment_totals = results.counts.compartment_counts.sum(axis=0)
    counts: dict[str, int | float] = {
        "n_transcripts_input": int(totals.get("n_input", 0)),
        "n_transcripts_in_tissue": int(totals.get("n_used", 0)),
        "n_transcripts_used": int(totals.get("n_used", 0)),
        "n_transcripts_outside_tissue": int(totals.get("n_outside_tissue", 0)),
        "n_transcripts_invalid_coordinates": int(
            totals.get("n_invalid_coordinates", 0)
        ),
        "n_transcripts_missing_feature_name": int(
            totals.get("n_missing_feature_names", 0)
        ),
        "n_control_transcripts_excluded": int(totals.get("n_controls_excluded", 0)),
        "n_transcript_genes": len(results.counts.gene_names),
        "n_cell_overlap_ambiguous": 0,
        "n_nucleus_overlap_ambiguous": 0,
        "tissue_area_um2": float(results.tissue.area_um2),
        "n_gene_transcripts_finite": int(totals.get("n_gene_transcripts_finite", 0)),
        "n_gene_transcripts_on_cell_label": int(
            totals.get("n_gene_transcripts_on_cell_label", 0)
        ),
        "n_gene_transcripts_on_cell_label_in_tissue": int(
            totals.get("n_gene_transcripts_on_cell_label_in_tissue", 0)
        ),
        "n_gene_transcripts_on_nucleus_label": int(
            totals.get("n_gene_transcripts_on_nucleus_label", 0)
        ),
        "n_gene_transcripts_off_label_grid": int(
            totals.get("n_gene_transcripts_off_label_grid", 0)
        ),
    }
    counts.update(
        {
            f"n_{compartment}_transcripts": int(compartment_totals[code])
            for code, compartment in enumerate(COMPARTMENTS)
        }
    )
    return counts


__all__ = [
    "DISTANCE_CLIP_CODE",
    "ENGINE_NAME",
    "PAIRCORR_STATUS",
    "DistanceEncoding",
    "GeneDiagnosticRaster",
    "LabelLayer",
    "RasterInputs",
    "RasterTranscriptPatternResults",
    "TissueMask",
    "accumulate_gene_patterns",
    "boundary_pixels",
    "lookup_many",
    "build_distance_raster",
    "build_tissue_mask",
    "counts_from_accumulators",
    "densest_window",
    "density_tissue_mask",
    "distance_encoding",
    "empty_paircorr_table",
    "gene_diagnostic_rasters",
    "layer_label_keys",
    "polygon_tissue_mask",
    "raster_transcript_counts",
    "resolve_label_layer",
    "run_label_raster_patterns",
    "run_raster_transcript_pattern_analysis",
]
