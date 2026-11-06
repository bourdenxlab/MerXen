"""Streaming transcript density rasters over every transcript of a store.

Whole-transcriptome sections hold more than a billion transcripts, so static
figures cannot scatter them and must not subsample them. This module bins the
points of a SpatialData store into a regular micron grid, datashader style:
it reads one parquet file of the points element at a time (only the columns
it needs, in record batches), drops the files whose indexed bounding box
misses the window (:mod:`merxen.io.points_aggregates`), and adds each batch to
integer accumulators with ``np.bincount``. Memory is bounded by one record
batch plus the grid, whatever the number of transcripts.

The bins follow ``np.histogram2d``: the edges are ``x0 + k * bin_um``, every bin
holds ``[edge_k, edge_k+1)`` and the last bin also holds its right edge.
Coordinates are the raw ``x``/``y`` micron columns of the points.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib import colors as mcolors

from merxen.io.points_aggregates import (
    DEFAULT_POINTS_KEY,
    BBox,
    points_parquet_dir,
    read_points_aggregates,
)
from merxen.io.transcript_io import assignment_mask

logger = logging.getLogger(__name__)

ALL_LAYER: Final = "all"
"""Layer name of a raster without a category column."""

ASSIGNED_LAYER: Final = "assigned"
UNASSIGNED_LAYER: Final = "unassigned"

BATCH_ROWS: Final = 4_194_304
"""Rows per parquet record batch; bounds the transient memory of a pass."""

MAX_BINS: Final = 64_000_000
"""Largest grid a raster may allocate (about 512 MB of uint64 per layer)."""

CategoryMode = Literal["assignment", "values"]
"""How a category column splits the rows into layers.

``"assignment"`` gives the layers ``assigned`` and ``unassigned`` with
:func:`merxen.io.transcript_io.assignment_mask`; ``"values"`` gives one layer
per distinct value of the column.
"""


@dataclass
class DensityRaster:
    """Transcript counts binned on a regular micron grid.

    Attributes:
        origin_um: ``(x0, y0)`` of the first bin's corner.
        bin_um: Edge length of a bin.
        shape: ``(n_y, n_x)`` of the grid.
        counts: Layer name to ``uint64`` counts of shape ``shape`` (row = y).
        n_rows_read: Rows read from the files that were scanned.
        n_files_read: Parquet files scanned.
        n_files_total: Parquet files of the points element.
        pruned: Whether the files were selected through the points index.
    """

    origin_um: tuple[float, float]
    bin_um: float
    shape: tuple[int, int]
    counts: dict[str, np.ndarray] = field(default_factory=dict)
    n_rows_read: int = 0
    n_files_read: int = 0
    n_files_total: int = 0
    pruned: bool = False

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return the ``(left, right, bottom, top)`` extent for ``imshow``."""
        n_y, n_x = self.shape
        x0, y0 = self.origin_um
        return (x0, x0 + n_x * self.bin_um, y0, y0 + n_y * self.bin_um)

    @property
    def x_edges(self) -> np.ndarray:
        """Return the ``n_x + 1`` bin edges along x."""
        return _edges(self.origin_um[0], self.bin_um, self.shape[1])

    @property
    def y_edges(self) -> np.ndarray:
        """Return the ``n_y + 1`` bin edges along y."""
        return _edges(self.origin_um[1], self.bin_um, self.shape[0])

    def total(self, layer: str | None = None) -> int:
        """Return the transcripts binned in one layer, or in every layer.

        Args:
            layer: Layer name; ``None`` sums every layer.

        Returns:
            The number of transcripts inside the grid.
        """
        if layer is not None:
            values = self.counts.get(layer)
            return 0 if values is None else int(values.sum(dtype=np.uint64))
        return int(sum(int(v.sum(dtype=np.uint64)) for v in self.counts.values()))

    def combined(self) -> np.ndarray:
        """Return the sum of every layer (``uint64``, shape ``shape``)."""
        out = np.zeros(self.shape, dtype=np.uint64)
        for values in self.counts.values():
            out += values
        return out


def _edges(origin: float, bin_um: float, n_bins: int) -> np.ndarray:
    return float(origin) + float(bin_um) * np.arange(n_bins + 1, dtype=np.float64)


def grid_shape(bbox: BBox, bin_um: float) -> tuple[int, int]:
    """Return the ``(n_y, n_x)`` grid covering a window with square bins.

    Args:
        bbox: Window ``(x0, y0, x1, y1)`` in microns.
        bin_um: Bin edge length in microns.

    Returns:
        The grid shape; the last bins may extend past ``x1``/``y1``.

    Raises:
        ValueError: If the window is empty, the bin size is not positive or
            the grid would exceed :data:`MAX_BINS`.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if not (math.isfinite(bin_um) and bin_um > 0):
        raise ValueError(f"bin_um must be a positive number, got {bin_um}")
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"empty raster window {bbox}")
    n_x = max(1, math.ceil((x1 - x0) / bin_um - 1e-6))
    n_y = max(1, math.ceil((y1 - y0) / bin_um - 1e-6))
    if n_x * n_y > MAX_BINS:
        raise ValueError(
            f"a {bin_um} um raster of {bbox} needs {n_x * n_y:,} bins "
            f"(limit {MAX_BINS:,}); use larger bins"
        )
    return n_y, n_x


def bin_indices(
    values: np.ndarray, origin: float, bin_um: float, n_bins: int
) -> np.ndarray:
    """Return the bin of each value, or ``-1`` outside the grid.

    Matches ``np.histogram``: bin ``k`` holds ``[edge_k, edge_k+1)`` and the
    last bin also holds the right edge. The floor estimate is corrected
    against the float64 edges, so values on an edge land where
    ``np.histogram2d`` puts them.

    Args:
        values: Coordinates in microns (any float dtype; NaN gives ``-1``).
        origin: First edge.
        bin_um: Bin edge length.
        n_bins: Number of bins.

    Returns:
        ``int64`` bin indices.
    """
    coords = np.asarray(values, dtype=np.float64)
    edges = _edges(origin, bin_um, n_bins)
    with np.errstate(invalid="ignore"):
        estimate = np.floor((coords - edges[0]) / float(bin_um))
    valid = np.isfinite(coords) & (coords >= edges[0]) & (coords <= edges[-1])
    index = np.where(valid, np.clip(estimate, 0, n_bins - 1), 0).astype(np.int64)
    # Correct floating-point disagreements with the edges in both directions.
    index -= (coords < edges[index]) & (index > 0)
    index += (coords >= edges[np.minimum(index + 1, n_bins)]) & (index < n_bins - 1)
    index[~valid] = -1
    return index


def points_files(
    zarr_path: Path,
    points_key: str = DEFAULT_POINTS_KEY,
    bbox: BBox | None = None,
    *,
    use_index: bool = True,
) -> tuple[list[Path], int, bool]:
    """Return the parquet files to scan for a window.

    Args:
        zarr_path: SpatialData store.
        points_key: Points element.
        bbox: Window ``(x0, y0, x1, y1)``; ``None`` keeps every file.
        use_index: Prune through the points index when it matches the files.

    Returns:
        ``(files, n_files_total, pruned)``.

    Raises:
        FileNotFoundError: If the points element has no parquet files.
    """
    directory = points_parquet_dir(Path(zarr_path), points_key)
    if directory.is_file():
        every = [directory]
    else:
        every = sorted(directory.glob("*.parquet"), key=_natural_key)
    if not every:
        raise FileNotFoundError(
            f"points[{points_key}] of the store has no parquet files"
        )
    if bbox is None or not use_index:
        return every, len(every), False
    aggregates = read_points_aggregates(Path(zarr_path), points_key)
    if aggregates is None or not aggregates.fingerprint_matches():
        return every, len(every), False
    selected = aggregates.files_intersecting(bbox)
    return selected, len(every), True


def _natural_key(path: Path) -> tuple[object, ...]:
    parts = path.stem.split(".")
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _iter_batches(path: Path, columns: Sequence[str]) -> Iterator[pa.RecordBatch]:
    parquet = pq.ParquetFile(path)
    try:
        yield from parquet.iter_batches(batch_size=BATCH_ROWS, columns=list(columns))
    finally:
        parquet.close()


def _column_numpy(batch: pa.RecordBatch, name: str) -> np.ndarray:
    column = batch.column(batch.schema.get_field_index(name))
    return np.asarray(column.to_numpy(zero_copy_only=False))


def _category_codes(
    batch: pa.RecordBatch,
    category: str,
    mode: CategoryMode,
    layers: list[str],
) -> np.ndarray:
    """Map each row of a batch to a layer index, extending ``layers``."""
    series = batch.column(batch.schema.get_field_index(category)).to_pandas()
    if mode == "assignment":
        assigned = assignment_mask(pd.Series(series)).to_numpy(dtype=bool)
        return np.where(assigned, 0, 1).astype(np.int64)
    values = pd.Series(series).astype("string").fillna("<NA>")
    codes = np.empty(len(values), dtype=np.int64)
    uniques, inverse = np.unique(values.to_numpy(dtype=object), return_inverse=True)
    for position, value in enumerate(uniques):
        name = str(value)
        if name not in layers:
            layers.append(name)
        codes[inverse == position] = layers.index(name)
    return codes


def stream_hist2d(
    zarr_path: Path | str,
    points_key: str = DEFAULT_POINTS_KEY,
    bbox: BBox | None = None,
    bin_um: float = 10.0,
    *,
    category: str | None = None,
    category_mode: CategoryMode = "values",
    x_col: str = "x",
    y_col: str = "y",
    use_index: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> DensityRaster:
    """Bin every transcript of a points element in a window.

    Args:
        zarr_path: SpatialData store whose ``points/<points_key>`` is read.
        points_key: Points element.
        bbox: Window ``(x0, y0, x1, y1)`` in microns. ``None`` takes the build
            aggregates' bounds, else the parquet column statistics.
        bin_um: Bin edge length in microns.
        category: Column splitting the rows into layers; ``None`` gives the
            single layer ``"all"``.
        category_mode: How ``category`` maps rows to layers
            (:data:`CategoryMode`).
        x_col: Micron x column.
        y_col: Micron y column.
        use_index: Read only the files whose indexed bbox meets the window
            when the points index still matches the store.
        progress: Called with ``(files_done, files_total)`` after each file.

    Returns:
        The raster. With ``category_mode="assignment"`` both layers are always
        present, ``assigned`` first.

    Raises:
        FileNotFoundError: If the points element has no parquet files.
        KeyError: If a requested column is missing from a file.
        ValueError: If the window or bin size is invalid.
    """
    zarr_path = Path(zarr_path)
    if bbox is None:
        bbox = points_bounds(zarr_path, points_key, x_col=x_col, y_col=y_col)
    n_y, n_x = grid_shape(bbox, bin_um)
    x0, y0 = float(bbox[0]), float(bbox[1])
    files, n_total, pruned = points_files(
        zarr_path, points_key, bbox, use_index=use_index
    )
    columns = [x_col, y_col] + ([category] if category is not None else [])
    if category is None:
        layers = [ALL_LAYER]
    elif category_mode == "assignment":
        layers = [ASSIGNED_LAYER, UNASSIGNED_LAYER]
    else:
        layers = []
    n_bins = n_x * n_y
    accumulators: list[np.ndarray] = []
    raster = DensityRaster(
        origin_um=(x0, y0),
        bin_um=float(bin_um),
        shape=(n_y, n_x),
        n_files_total=n_total,
        pruned=pruned,
    )
    for done, path in enumerate(files, start=1):
        for batch in _iter_batches(path, columns):
            raster.n_rows_read += batch.num_rows
            if batch.num_rows == 0:
                continue
            ix = bin_indices(_column_numpy(batch, x_col), x0, bin_um, n_x)
            iy = bin_indices(_column_numpy(batch, y_col), y0, bin_um, n_y)
            inside = (ix >= 0) & (iy >= 0)
            if not np.any(inside):
                continue
            flat = iy[inside] * n_x + ix[inside]
            if category is None:
                codes = np.zeros(flat.shape, dtype=np.int64)
            else:
                codes = _category_codes(batch, category, category_mode, layers)[inside]
            while len(accumulators) < len(layers):
                accumulators.append(np.zeros(n_bins, dtype=np.uint64))
            counts = np.bincount(
                codes * n_bins + flat, minlength=len(layers) * n_bins
            ).astype(np.uint64, copy=False)
            for layer_index, accumulator in enumerate(accumulators):
                accumulator += counts[layer_index * n_bins : (layer_index + 1) * n_bins]
        raster.n_files_read += 1
        if progress is not None:
            progress(done, len(files))
    while len(accumulators) < len(layers):
        accumulators.append(np.zeros(n_bins, dtype=np.uint64))
    raster.counts = {
        name: accumulator.reshape(n_y, n_x)
        for name, accumulator in zip(layers, accumulators, strict=True)
    }
    logger.info(
        "Binned %d of %d rows from %d/%d parquet files of points[%s] at %.3g um "
        "(pruned=%s)",
        raster.total(),
        raster.n_rows_read,
        raster.n_files_read,
        n_total,
        points_key,
        bin_um,
        pruned,
    )
    return raster


def points_bounds(
    zarr_path: Path | str,
    points_key: str = DEFAULT_POINTS_KEY,
    *,
    x_col: str = "x",
    y_col: str = "y",
) -> BBox:
    """Return ``(x0, y0, x1, y1)`` of the transcripts of a store.

    Uses the build aggregates when they still match the files; otherwise the
    parquet row-group statistics; a file without statistics is streamed.

    Args:
        zarr_path: SpatialData store.
        points_key: Points element.
        x_col: Micron x column.
        y_col: Micron y column.

    Returns:
        The bounding box of every transcript.

    Raises:
        FileNotFoundError: If the points element has no parquet files.
        ValueError: If no finite coordinate exists.
    """
    zarr_path = Path(zarr_path)
    aggregates = read_points_aggregates(zarr_path, points_key)
    if aggregates is not None and aggregates.fingerprint_matches():
        bounds = aggregates.bounds_um
        keys = ("x_min", "y_min", "x_max", "y_max")
        if all(key in bounds for key in keys):
            return (
                bounds["x_min"],
                bounds["y_min"],
                bounds["x_max"],
                bounds["y_max"],
            )
    files, _, _ = points_files(zarr_path, points_key, None, use_index=False)
    low = [math.inf, math.inf]
    high = [-math.inf, -math.inf]
    for path in files:
        stats = _file_column_range(path, (x_col, y_col))
        if stats is None:
            stats = _scan_column_range(path, (x_col, y_col))
        if stats is None:
            continue
        for axis in range(2):
            low[axis] = min(low[axis], stats[axis][0])
            high[axis] = max(high[axis], stats[axis][1])
    if not all(math.isfinite(v) for v in (*low, *high)):
        raise ValueError(f"points[{points_key}] has no finite coordinates")
    return (low[0], low[1], high[0], high[1])


def _file_column_range(
    path: Path, columns: Sequence[str]
) -> list[tuple[float, float]] | None:
    """Return per-column ``(min, max)`` from the row-group statistics."""
    metadata = pq.read_metadata(path)
    names = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
    ranges: list[tuple[float, float]] = []
    for name in columns:
        if name not in names:
            raise KeyError(f"column {name!r} missing from {path.name}")
        position = names.index(name)
        low, high = math.inf, -math.inf
        for group in range(metadata.num_row_groups):
            column = metadata.row_group(group).column(position)
            stats = column.statistics
            if column.num_values == 0:
                continue
            if stats is None or not stats.has_min_max:
                return None
            low = min(low, float(stats.min))
            high = max(high, float(stats.max))
        ranges.append((low, high))
    return ranges


def _scan_column_range(
    path: Path, columns: Sequence[str]
) -> list[tuple[float, float]] | None:
    low = [math.inf] * len(columns)
    high = [-math.inf] * len(columns)
    for batch in _iter_batches(path, columns):
        for axis, name in enumerate(columns):
            values = _column_numpy(batch, name).astype(np.float64, copy=False)
            finite = values[np.isfinite(values)]
            if finite.size:
                low[axis] = min(low[axis], float(finite.min()))
                high[axis] = max(high[axis], float(finite.max()))
    if not all(math.isfinite(v) for v in low):
        return None
    return list(zip(low, high, strict=True))


def percentile_vmax(counts: np.ndarray, percentile: float = 99.5) -> float:
    """Return a display maximum from the non-empty bins of a raster.

    Args:
        counts: Binned counts.
        percentile: Percentile of the non-zero bins.

    Returns:
        The percentile, at least 1.
    """
    values = np.asarray(counts)
    positive = values[values > 0]
    if positive.size == 0:
        return 1.0
    return max(1.0, float(np.percentile(positive.astype(np.float64), percentile)))


def composite_rgba(
    layers: Sequence[tuple[np.ndarray, str]],
    *,
    vmax: Mapping[int, float] | None = None,
    percentile: float = 99.5,
    max_alpha: float = 0.9,
) -> np.ndarray:
    """Composite count layers into one RGBA image with log alpha.

    Each layer is drawn in its own colour with an opacity of
    ``log1p(count) / log1p(vmax)`` (clipped to ``max_alpha``), where ``vmax``
    is the layer's ``percentile`` of non-zero bins unless given. Layers are
    stacked with the "over" operator in order, the last on top. Empty bins
    stay fully transparent.

    Args:
        layers: ``(counts, colour)`` pairs, bottom first.
        vmax: Optional per-layer (by position) display maximum.
        percentile: Percentile used for a layer without ``vmax``.
        max_alpha: Opacity of a bin at or above ``vmax``.

    Returns:
        ``float32`` RGBA of shape ``counts.shape + (4,)``, rows as in the
        counts (draw with ``origin="lower"``).

    Raises:
        ValueError: If no layer is given or the shapes differ.
    """
    if not layers:
        raise ValueError("composite_rgba needs at least one layer")
    shape = np.asarray(layers[0][0]).shape
    out_rgb = np.zeros((*shape, 3), dtype=np.float32)
    out_alpha = np.zeros(shape, dtype=np.float32)
    for position, (counts, colour) in enumerate(layers):
        values = np.asarray(counts, dtype=np.float64)
        if values.shape != shape:
            raise ValueError(f"layer shapes differ: {values.shape} != {shape}")
        top = (vmax or {}).get(position)
        top = percentile_vmax(values, percentile) if top is None else float(top)
        alpha = np.clip(np.log1p(values) / np.log1p(max(top, 1.0)), 0.0, 1.0)
        alpha = (alpha * max_alpha).astype(np.float32)
        rgb = np.asarray(mcolors.to_rgb(colour), dtype=np.float32)
        new_alpha = alpha + out_alpha * (1.0 - alpha)
        with np.errstate(invalid="ignore", divide="ignore"):
            blended = (
                rgb * alpha[..., None]
                + out_rgb * (out_alpha * (1.0 - alpha))[..., None]
            ) / new_alpha[..., None]
        out_rgb = np.where(new_alpha[..., None] > 0, blended, 0.0).astype(np.float32)
        out_alpha = new_alpha
    return np.concatenate([out_rgb, out_alpha[..., None]], axis=-1)


def single_colour_cmap(colour: str, name: str = "merxen_density") -> mcolors.Colormap:
    """Return a colormap from white to one colour, for density panels.

    Args:
        colour: Matplotlib colour of the densest bins.
        name: Colormap name.

    Returns:
        A linear segmented colormap; masked (empty) bins are transparent.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(name, ["#ffffff", colour])
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    return cmap
