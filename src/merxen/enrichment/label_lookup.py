"""Streaming transcript lookups into label masks (vendor corner convention).

A native (vendor-segmentation) store keeps the vendor cell and nucleus label
masks as label elements in the ``global`` pixel frame (``Identity``), while the
transcripts are stored in microns. A transcript at ``(x, y)`` microns lies on
mask pixel ``(floor(y / ps), floor(x / ps))`` under the vendor's corner
convention (pixel ``i`` covers ``[i * ps, (i + 1) * ps)``; plan §3.8). The
centre convention (``rint``) of the Cellpose masks MerXen writes disagrees with
the vendor on about 1.35% of transcripts, so the lookups here default to
``"corner"``.

Nothing here loads a whole mask or the whole points element. The points are
read one partition (parquet file) at a time, only the requested columns; the
label mask is read block by block, only the blocks the partition's
transcripts touch. Decoded blocks are kept in a small LRU shared by the
partitions (bounded in bytes), so neighbouring partitions of one tile decode
each block once. Partitions run in a thread pool bounded by
``DASK_NUM_WORKERS``, and each one is reduced to a small result by the
caller's ``reduce`` function, so memory is bounded by a few partitions and
the block cache however large the section is.
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PixelConvention = Literal["center", "corner"]

DEFAULT_BLOCK_PX: Final = 4096
"""Edge of the label blocks read at once (the Atera label chunk size)."""

DEFAULT_BLOCK_CACHE_BYTES: Final = 3 * 1024**3
"""Bytes of decoded label blocks kept for reuse across partitions.

About 48 uint32 blocks of 4096 x 4096: the cell and nucleus blocks of the two
or three tiles that the partitions in flight belong to.
"""

_T = TypeVar("_T")

PixelPrecision = Literal["float32", "float64"]


@dataclass(frozen=True)
class PixelFrame:
    """Axis-aligned micron -> mask pixel mapping.

    Attributes:
        um_per_px: Microns per mask pixel (isotropic).
        origin_um: Micron coordinates ``(x, y)`` of the corner of pixel
            ``(0, 0)``.
        precision: Arithmetic of the mapping. ``"float32"`` is the vendor's
            rule, ``float32(x) * float32(1 / ps)``: it reproduces the 10x
            vendor assignment exactly, including the transcripts that sit
            within a thousandth of a pixel of a pixel edge. On the real crop,
            float64 division ``x / ps`` puts 552 of 40.6 million such
            transcripts (1.4e-5) on the neighbouring pixel. ``"float64"`` is
            ``x / ps`` in float64.
    """

    um_per_px: float
    origin_um: tuple[float, float] = (0.0, 0.0)
    precision: PixelPrecision = "float32"

    def __post_init__(self) -> None:
        """Reject a pixel size that is not a finite positive number."""
        if not np.isfinite(self.um_per_px) or float(self.um_per_px) <= 0:
            raise ValueError(f"um_per_px must be positive, got {self.um_per_px!r}")
        if self.precision not in ("float32", "float64"):
            raise ValueError(
                f"precision must be 'float32' or 'float64', got {self.precision!r}"
            )

    def to_pixels(self, x_um: np.ndarray, y_um: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the continuous pixel coordinates ``(x_px, y_px)``."""
        x = np.asarray(x_um)
        y = np.asarray(y_um)
        if self.precision == "float32":
            scale = np.float32(1.0 / float(self.um_per_px))
            ox, oy = (np.float32(value) for value in self.origin_um)
            x32 = x.astype(np.float32, copy=False)
            y32 = y.astype(np.float32, copy=False)
            if ox != 0:
                x32 = x32 - ox
            if oy != 0:
                y32 = y32 - oy
            return x32 * scale, y32 * scale
        ps = float(self.um_per_px)
        x_px = (x.astype(np.float64, copy=False) - float(self.origin_um[0])) / ps
        y_px = (y.astype(np.float64, copy=False) - float(self.origin_um[1])) / ps
        return x_px, y_px


def pixel_indices(
    x_um: np.ndarray,
    y_um: np.ndarray,
    frame: PixelFrame,
    *,
    convention: PixelConvention = "corner",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mask column and row of each micron coordinate.

    The arithmetic follows ``frame.precision``; the default float32 rule is
    the vendor's, so a transcript lands on the pixel the vendor assigned it
    to even within a thousandth of a pixel of an edge.

    Args:
        x_um: X coordinates in microns.
        y_um: Y coordinates in microns.
        frame: Micron -> pixel mapping.
        convention: ``"corner"`` (floor; vendor masks) or ``"center"``
            (round half to even; MerXen's Cellpose masks).

    Returns:
        ``(column, row)`` as int64 arrays. Non-finite coordinates map to
        ``-1`` (outside every mask).

    Raises:
        ValueError: If ``convention`` is unknown.
    """
    if convention not in ("center", "corner"):
        raise ValueError(
            f"pixel_convention must be 'center' or 'corner', got {convention!r}"
        )
    x_px, y_px = frame.to_pixels(x_um, y_um)
    rounder = np.floor if convention == "corner" else np.rint
    finite = np.isfinite(x_px) & np.isfinite(y_px)
    column = np.full(x_px.shape, -1, dtype=np.int64)
    row = np.full(y_px.shape, -1, dtype=np.int64)
    column[finite] = rounder(x_px[finite]).astype(np.int64)
    row[finite] = rounder(y_px[finite]).astype(np.int64)
    return column, row


def _read_block(labels: Any, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Read one rectangle of a numpy, zarr or dask label array."""
    block = labels[y0:y1, x0:x1]
    if hasattr(block, "compute"):
        # The caller's thread pool is the parallelism; a nested dask pool
        # per block would oversubscribe it.
        block = block.compute(scheduler="synchronous")
    return np.asarray(block)


class BlockCache:
    """Thread-safe LRU of decoded label blocks, bounded in bytes.

    A block requested by several threads at once is decoded once: the first
    thread decodes it and the others wait for that result. Blocks larger than
    the budget are returned without being kept.

    Attributes:
        max_bytes: Budget of the kept blocks.
        hits: Requests served from the cache (or from a decode in flight).
        misses: Requests that decoded a block.
    """

    def __init__(self, max_bytes: int) -> None:
        """Create an empty cache.

        Args:
            max_bytes: Budget of the kept blocks; 0 keeps nothing.

        Raises:
            ValueError: If ``max_bytes`` is negative.
        """
        if int(max_bytes) < 0:
            raise ValueError(f"max_bytes must be >= 0, got {max_bytes!r}")
        self.max_bytes = int(max_bytes)
        self.hits = 0
        self.misses = 0
        self._bytes = 0
        self._blocks: OrderedDict[Hashable, np.ndarray] = OrderedDict()
        self._pending: dict[Hashable, Future[np.ndarray]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable, load: Callable[[], np.ndarray]) -> np.ndarray:
        """Return the block of ``key``, decoding it with ``load`` on a miss."""
        with self._lock:
            block = self._blocks.get(key)
            if block is not None:
                self._blocks.move_to_end(key)
                self.hits += 1
                return block
            pending = self._pending.get(key)
            if pending is None:
                future: Future[np.ndarray] = Future()
                self._pending[key] = future
                self.misses += 1
            else:
                self.hits += 1
        if pending is not None:
            return pending.result()
        try:
            block = load()
        except BaseException as exc:
            with self._lock:
                del self._pending[key]
            future.set_exception(exc)
            raise
        with self._lock:
            del self._pending[key]
            if 0 < block.nbytes <= self.max_bytes:
                self._blocks[key] = block
                self._bytes += block.nbytes
                while self._bytes > self.max_bytes:
                    _, evicted = self._blocks.popitem(last=False)
                    self._bytes -= evicted.nbytes
        future.set_result(block)
        return block


def lookup_labels(
    column: np.ndarray,
    row: np.ndarray,
    labels: Any,
    *,
    block_px: int = DEFAULT_BLOCK_PX,
    cache: BlockCache | None = None,
    cache_key: Hashable = None,
) -> np.ndarray:
    """Return the label under each pixel, reading only the blocks touched.

    Args:
        column: Pixel columns (int64; any value, out-of-range ones get 0).
        row: Pixel rows, aligned with ``column``.
        labels: Two-dimensional label array (numpy, zarr or dask) indexed
            ``[row, column]``.
        block_px: Edge of the blocks read at once.
        cache: Decoded blocks shared with other lookups, or ``None``.
        cache_key: Name of ``labels`` in ``cache`` (one per mask).

    Returns:
        The label of each pixel as int64; 0 for background and for pixels
        outside the mask (negative coordinates included).
    """
    height, width = (int(size) for size in labels.shape)
    out = np.zeros(column.shape, dtype=np.int64)
    inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    if not inside.any():
        return out
    positions = np.flatnonzero(inside)
    block_row = row[positions] // block_px
    block_col = column[positions] // block_px
    n_block_cols = (width + block_px - 1) // block_px
    block_ids = block_row * n_block_cols + block_col
    order = np.argsort(block_ids, kind="stable")
    sorted_ids = block_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    stops = np.r_[starts[1:], sorted_ids.size]
    for start, stop in zip(starts, stops, strict=True):
        block_id = int(sorted_ids[start])
        by, bx = divmod(block_id, n_block_cols)
        y0, x0 = by * block_px, bx * block_px
        y1, x1 = min(height, y0 + block_px), min(width, x0 + block_px)
        if cache is None:
            block = _read_block(labels, y0, y1, x0, x1)
        else:
            block = cache.get(
                (cache_key, block_px, y0, x0),
                partial(_read_block, labels, y0, y1, x0, x1),
            )
        members = positions[order[start:stop]]
        out[members] = block[row[members] - y0, column[members] - x0].astype(
            np.int64, copy=False
        )
    return out


def default_n_workers() -> int:
    """Return ``DASK_NUM_WORKERS`` when set, else ``min(8, CPU count)``.

    Raises:
        ValueError: If ``DASK_NUM_WORKERS`` is set but not a positive integer.
    """
    raw = os.environ.get("DASK_NUM_WORKERS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(
                f"DASK_NUM_WORKERS must be a positive integer, got {raw!r}"
            ) from None
        if value <= 0:
            raise ValueError(f"DASK_NUM_WORKERS must be positive, got {raw!r}")
        return value
    return max(1, min(8, os.cpu_count() or 1))


def natural_partition_order(paths: Sequence[Path]) -> list[Path]:
    """Return parquet partitions in natural name order (``part.2`` < ``part.10``).

    The points writer numbers the partitions tile by tile, so in this order
    neighbouring partitions (the ones in flight together in the thread pool)
    touch the same label blocks and share the :class:`BlockCache`.

    Args:
        paths: Parquet files of one points element.

    Returns:
        The same files, sorted.
    """

    def _key(path: Path) -> tuple[tuple[int, int, str], ...]:
        return tuple(
            (0, int(token), "") if token.isdigit() else (1, 0, token)
            for token in re.split(r"(\d+)", path.name)
            if token
        )

    return sorted((Path(path) for path in paths), key=_key)


def partition_readers(
    points: Sequence[Path] | Any, columns: Sequence[str]
) -> list[Callable[[], pd.DataFrame]]:
    """Return one reader per points partition, each reading only ``columns``.

    Args:
        points: Parquet files (read with pyarrow), a Dask DataFrame (one reader
            per partition) or a pandas DataFrame (one partition).
        columns: Columns to read.

    Returns:
        The readers, in partition order.
    """
    wanted = list(columns)
    if isinstance(points, pd.DataFrame):
        frame = points[wanted]
        return [lambda: frame]
    if hasattr(points, "to_delayed"):
        lazy: Any = points
        delayed_parts = lazy[wanted].to_delayed()

        def _delayed_reader(part: Any) -> Callable[[], pd.DataFrame]:
            def _read() -> pd.DataFrame:
                frame: pd.DataFrame = part.compute(scheduler="synchronous")
                return frame

            return _read

        return [_delayed_reader(part) for part in delayed_parts]
    paths = [Path(path) for path in points]

    def _reader(path: Path) -> Callable[[], pd.DataFrame]:
        def _read() -> pd.DataFrame:
            table = pq.read_table(path, columns=wanted, use_threads=False)
            return table.to_pandas()

        return _read

    return [_reader(path) for path in paths]


def stream_label_lookup(
    points: Sequence[Path] | Any,
    labels_level0: Mapping[str, Any],
    um_per_px: float,
    *,
    columns: Sequence[str],
    reduce: Callable[[pd.DataFrame, dict[str, np.ndarray]], _T],
    convention: PixelConvention = "corner",
    origin_um: tuple[float, float] = (0.0, 0.0),
    precision: PixelPrecision = "float32",
    x_column: str = "x",
    y_column: str = "y",
    block_px: int = DEFAULT_BLOCK_PX,
    block_cache_bytes: int = DEFAULT_BLOCK_CACHE_BYTES,
    n_workers: int | None = None,
) -> list[_T]:
    """Look every transcript up in one or more label masks, partition by partition.

    For each partition, only ``columns`` (plus the coordinates) are read, the
    pixel of every row is computed once (``convention``), every mask of
    ``labels_level0`` is read block by block where the rows fall, and
    ``reduce(frame, labels)`` turns the partition into a small result. The
    partitions run in a thread pool; results are returned in partition order.

    Args:
        points: Parquet files, a Dask DataFrame or a pandas DataFrame.
        labels_level0: Name -> level-0 label array, all on the same pixel grid.
        um_per_px: Microns per mask pixel.
        columns: Further columns ``reduce`` needs.
        reduce: Called once per partition with the partition's columns and the
            label of each row per mask (int64; 0 off-mask).
        convention: Pixel convention (``"corner"`` for vendor masks).
        origin_um: Micron position of the corner of pixel ``(0, 0)``.
        precision: Arithmetic of the micron -> pixel mapping
            (:class:`PixelFrame`; ``"float32"`` is the vendor's rule).
        x_column: X coordinate column (microns).
        y_column: Y coordinate column (microns).
        block_px: Edge of the label blocks read at once.
        block_cache_bytes: Bytes of decoded blocks kept for the following
            partitions (:class:`BlockCache`); 0 decodes every block per
            partition.
        n_workers: Threads; ``None`` uses :func:`default_n_workers`.

    Returns:
        ``reduce``'s result for every partition, in partition order.

    Raises:
        ValueError: If the masks do not share one shape.
    """
    shapes = {name: tuple(array.shape) for name, array in labels_level0.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"label masks differ in shape: {shapes}")
    frame = PixelFrame(
        float(um_per_px),
        (float(origin_um[0]), float(origin_um[1])),
        precision=precision,
    )
    wanted = list(dict.fromkeys([x_column, y_column, *columns]))
    readers = partition_readers(points, wanted)
    cache = BlockCache(block_cache_bytes) if block_cache_bytes > 0 else None

    def _one(reader: Callable[[], pd.DataFrame]) -> _T:
        part = reader()
        column, row = pixel_indices(
            part[x_column].to_numpy(),
            part[y_column].to_numpy(),
            frame,
            convention=convention,
        )
        looked_up = {
            name: lookup_labels(
                column, row, array, block_px=block_px, cache=cache, cache_key=name
            )
            for name, array in labels_level0.items()
        }
        return reduce(part, looked_up)

    workers = int(n_workers) if n_workers is not None else default_n_workers()
    if workers <= 1 or len(readers) <= 1:
        return [_one(reader) for reader in readers]
    with ThreadPoolExecutor(max_workers=min(workers, len(readers))) as pool:
        return list(pool.map(_one, readers))


NATIVE_COUNT_COLUMNS: Final = {
    "cell": "merxen_cell_transcript_counts",
    "nucleus": "merxen_nucleus_transcript_counts",
    "cell_gene": "merxen_cell_gene_counts",
    "nucleus_gene": "merxen_nucleus_gene_counts",
}
"""``table_original`` obs columns written by the native label check.

``cell``: transcripts whose pixel lies on the cell's ``cell_labels`` label.
``nucleus``: transcripts on any ``nucleus_labels`` label of a nucleus the cell
owns (through ``cell_instance_id``). The ``*_gene`` columns count only rows
with ``is_gene`` (written when the points carry that column).
"""

DEFAULT_MISMATCH_TOLERANCE: Final = 1e-6
"""Largest tolerated fraction of rows whose cell label disagrees with the vendor."""


@dataclass
class _NativeTally:
    """Running sums of the native label check over partitions."""

    n_bins: int
    has_overlaps: bool
    has_gene_flag: bool
    n_partitions: int = 0
    n_rows: int = 0
    n_vendor_assigned: int = 0
    n_on_cell_label: int = 0
    n_mismatch_label_differs: int = 0
    n_mismatch_vendor_unassigned: int = 0
    n_cell_labels_outside_table: int = 0
    n_on_nucleus_label: int = 0
    n_unknown_nucleus_labels: int = 0
    n_nucleus_outside_own_cell: int = 0
    n_overlaps_vendor: int = 0
    n_overlaps_agree: int = 0
    n_overlaps_vendor_only: int = 0
    n_overlaps_lookup_only: int = 0
    per_cell: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Allocate the per-cell (label-indexed) count arrays."""
        names = ["cell", "nucleus"]
        if self.has_gene_flag:
            names += ["cell_gene", "nucleus_gene"]
        self.per_cell = {name: np.zeros(self.n_bins, dtype=np.int64) for name in names}


def _label_ids(series: pd.Series) -> np.ndarray:
    """Return a nullable/float label column as int64 with missing -> 0."""
    values = pd.to_numeric(series, errors="coerce")
    ids: np.ndarray = values.fillna(0).to_numpy(dtype=np.int64)
    return ids


def native_label_check(
    points: Sequence[Path] | Any,
    *,
    cell_labels: Any,
    nucleus_labels: Any | None,
    nucleus_ids: np.ndarray | None,
    nucleus_cell_ids: np.ndarray | None,
    table_cell_ids: np.ndarray,
    um_per_px: float,
    assignment_column: str = "original_assignment",
    overlaps_column: str | None = "overlaps_nucleus",
    gene_flag_column: str | None = "is_gene",
    available_columns: Sequence[str] | None = None,
    tolerance: float = DEFAULT_MISMATCH_TOLERANCE,
    precision: PixelPrecision = "float32",
    n_workers: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Check vendor assignments against the label masks and count per cell.

    One streaming pass over the points (corner convention):

    - the ``cell_labels`` label of every row is compared with the vendor's
      ``assignment_column`` (0 or missing = unassigned);
    - the ``nucleus_labels`` label is mapped to its owning cell
      (``nucleus_cell_ids[nucleus_ids == label]``; nucleus ids are not cell
      ids) and counted for that cell;
    - when present, the vendor ``overlaps_column`` is compared with "on a
      nucleus label".

    The check passes only when every entry of ``summary["checks"]`` holds:
    ``cell_assignment`` (at most ``tolerance`` of the rows disagree with the
    vendor), ``cell_labels_in_table`` (no cell label above the largest table
    id), ``nucleus_labels_known`` (every nucleus label has a nucleus row with
    an owning cell), ``nuclei_inside_own_cell`` (every nuclear transcript lies
    on its owning cell's label: a broken ``cell_instance_id`` join fails here)
    and ``nucleus_overlaps`` (at most ``tolerance`` of the rows disagree with
    the vendor's nucleus flag). ``summary["failures"]`` names the failing ones.

    Args:
        points: Parquet files, a Dask DataFrame or a pandas DataFrame.
        cell_labels: Level-0 cell label mask (``[row, column]``).
        nucleus_labels: Level-0 nucleus label mask on the same grid, or
            ``None`` to skip the nucleus attribution.
        nucleus_ids: Nucleus ``instance_id`` (= ``nucleus_labels`` value).
        nucleus_cell_ids: Owning cell ``instance_id`` of each nucleus.
        table_cell_ids: Cell ``instance_id`` of every table row.
        um_per_px: Microns per mask pixel.
        assignment_column: Vendor cell assignment column of the points.
        overlaps_column: Vendor "on a nucleus" flag column, or ``None``.
        gene_flag_column: Boolean "is a gene" column, or ``None``.
        available_columns: Columns of the points; optional columns missing
            from it are skipped. ``None`` assumes all are present.
        tolerance: Largest tolerated fraction of mismatching rows.
        precision: Arithmetic of the micron -> pixel mapping; the vendor's
            float32 rule reproduces its assignment exactly.
        n_workers: Threads; ``None`` uses :func:`default_n_workers`.

    Returns:
        ``(summary, per_cell)``. ``summary`` is JSON-serialisable and has
        ``passed``, ``checks`` and ``failures``; ``per_cell`` maps each
        :data:`NATIVE_COUNT_COLUMNS` key to counts aligned with
        ``table_cell_ids``.

    Raises:
        ValueError: If the nucleus inputs are inconsistent.
    """
    available = None if available_columns is None else set(available_columns)

    def _has(column: str | None) -> bool:
        return column is not None and (available is None or column in available)

    has_overlaps = _has(overlaps_column) and nucleus_labels is not None
    has_gene_flag = _has(gene_flag_column)
    table_ids = np.asarray(table_cell_ids, dtype=np.int64)
    if table_ids.size and table_ids.min() < 1:
        raise ValueError("table cell instance ids must be positive")

    nucleus_to_cell = np.zeros(1, dtype=np.int64)
    if nucleus_labels is not None:
        if nucleus_ids is None or nucleus_cell_ids is None:
            raise ValueError("nucleus_labels needs nucleus_ids and nucleus_cell_ids")
        nuc_ids = np.asarray(nucleus_ids, dtype=np.int64)
        owner_ids = np.asarray(nucleus_cell_ids, dtype=np.int64)
        if nuc_ids.shape != owner_ids.shape:
            raise ValueError("nucleus_ids and nucleus_cell_ids differ in length")
        if nuc_ids.size:
            if nuc_ids.min() < 1 or len(np.unique(nuc_ids)) != nuc_ids.size:
                raise ValueError("nucleus instance ids must be positive and unique")
            nucleus_to_cell = np.zeros(int(nuc_ids.max()) + 1, dtype=np.int64)
            nucleus_to_cell[nuc_ids] = owner_ids

    max_cell = int(max(table_ids.max(initial=0), nucleus_to_cell.max(initial=0)))
    tally = _NativeTally(
        n_bins=max_cell + 1, has_overlaps=has_overlaps, has_gene_flag=has_gene_flag
    )
    lock = threading.Lock()
    columns = [assignment_column]
    if has_overlaps and overlaps_column is not None:
        columns.append(overlaps_column)
    if has_gene_flag and gene_flag_column is not None:
        columns.append(gene_flag_column)
    labels: dict[str, Any] = {"cell": cell_labels}
    if nucleus_labels is not None:
        labels["nucleus"] = nucleus_labels

    def _reduce(part: pd.DataFrame, looked_up: dict[str, np.ndarray]) -> None:
        cell = looked_up["cell"]
        vendor = _label_ids(part[assignment_column])
        differs = (vendor > 0) & (cell != vendor)
        vendor_unassigned = (vendor == 0) & (cell > 0)
        inside_table = cell < tally.n_bins
        cell_in = np.where(inside_table, cell, 0)
        is_gene = (
            part[gene_flag_column].fillna(False).to_numpy(dtype=bool)
            if has_gene_flag and gene_flag_column is not None
            else None
        )
        counts = {"cell": np.bincount(cell_in, minlength=tally.n_bins)}
        if is_gene is not None:
            counts["cell_gene"] = np.bincount(
                cell_in, weights=is_gene, minlength=tally.n_bins
            ).astype(np.int64)
        stats = {
            "n_rows": len(part),
            "n_vendor_assigned": int((vendor > 0).sum()),
            "n_on_cell_label": int((cell > 0).sum()),
            "n_mismatch_label_differs": int(differs.sum()),
            "n_mismatch_vendor_unassigned": int(vendor_unassigned.sum()),
            "n_cell_labels_outside_table": int((~inside_table).sum()),
        }
        if "nucleus" in looked_up:
            nucleus = looked_up["nucleus"]
            known = nucleus < nucleus_to_cell.size
            owner = np.where(known, nucleus_to_cell[np.where(known, nucleus, 0)], 0)
            unknown = (nucleus > 0) & ((~known) | (owner == 0))
            owner = np.where(owner < tally.n_bins, owner, 0)
            counts["nucleus"] = np.bincount(owner, minlength=tally.n_bins)
            if is_gene is not None:
                counts["nucleus_gene"] = np.bincount(
                    owner, weights=is_gene, minlength=tally.n_bins
                ).astype(np.int64)
            on_nucleus = nucleus > 0
            stats["n_on_nucleus_label"] = int(on_nucleus.sum())
            stats["n_unknown_nucleus_labels"] = int(unknown.sum())
            stats["n_nucleus_outside_own_cell"] = int(
                ((owner > 0) & (owner != cell)).sum()
            )
            if has_overlaps and overlaps_column is not None:
                vendor_overlaps = (
                    part[overlaps_column].fillna(False).to_numpy().astype(bool)
                )
                stats["n_overlaps_vendor"] = int(vendor_overlaps.sum())
                stats["n_overlaps_agree"] = int((vendor_overlaps == on_nucleus).sum())
                stats["n_overlaps_vendor_only"] = int(
                    (vendor_overlaps & ~on_nucleus).sum()
                )
                stats["n_overlaps_lookup_only"] = int(
                    (on_nucleus & ~vendor_overlaps).sum()
                )
        with lock:
            tally.n_partitions += 1
            for name, total in stats.items():
                setattr(tally, name, getattr(tally, name) + int(total))
            assert tally.per_cell is not None
            for name, partial in counts.items():
                tally.per_cell[name] += partial[: tally.n_bins].astype(np.int64)

    stream_label_lookup(
        points,
        labels,
        um_per_px,
        columns=columns,
        reduce=_reduce,
        convention="corner",
        precision=precision,
        n_workers=n_workers,
    )

    n_mismatch = tally.n_mismatch_label_differs + tally.n_mismatch_vendor_unassigned
    allowed = int(np.floor(float(tolerance) * tally.n_rows))
    checks: dict[str, bool] = {
        "cell_assignment": bool(n_mismatch <= allowed),
        "cell_labels_in_table": tally.n_cell_labels_outside_table == 0,
    }
    if nucleus_labels is not None:
        checks["nucleus_labels_known"] = tally.n_unknown_nucleus_labels == 0
        checks["nuclei_inside_own_cell"] = tally.n_nucleus_outside_own_cell == 0
        if has_overlaps:
            n_overlaps_disagree = (
                tally.n_overlaps_vendor_only + tally.n_overlaps_lookup_only
            )
            checks["nucleus_overlaps"] = bool(n_overlaps_disagree <= allowed)
    summary: dict[str, Any] = {
        "pixel_convention": "corner",
        "pixel_precision": precision,
        "um_per_px": float(um_per_px),
        "assignment_column": assignment_column,
        "n_partitions": tally.n_partitions,
        "n_rows": tally.n_rows,
        "n_vendor_assigned": tally.n_vendor_assigned,
        "n_on_cell_label": tally.n_on_cell_label,
        "n_mismatch": n_mismatch,
        "n_mismatch_label_differs": tally.n_mismatch_label_differs,
        "n_mismatch_vendor_unassigned": tally.n_mismatch_vendor_unassigned,
        "mismatch_tolerance_fraction": float(tolerance),
        "mismatch_allowed": allowed,
        "n_cell_labels_outside_table": tally.n_cell_labels_outside_table,
        "n_table_cells": int(table_ids.size),
        "checks": checks,
        "failures": sorted(name for name, ok in checks.items() if not ok),
        "passed": all(checks.values()),
    }
    if nucleus_labels is not None:
        summary["nucleus"] = {
            "n_nuclei": int(np.count_nonzero(nucleus_to_cell)),
            "n_on_nucleus_label": tally.n_on_nucleus_label,
            "n_unknown_nucleus_labels": tally.n_unknown_nucleus_labels,
            "n_nucleus_outside_own_cell": tally.n_nucleus_outside_own_cell,
        }
        if has_overlaps:
            agreement = tally.n_overlaps_agree / tally.n_rows if tally.n_rows else None
            summary["nucleus"]["overlaps_column"] = overlaps_column
            summary["nucleus"]["n_overlaps_vendor"] = tally.n_overlaps_vendor
            summary["nucleus"]["n_overlaps_agree"] = tally.n_overlaps_agree
            summary["nucleus"]["n_overlaps_vendor_only"] = tally.n_overlaps_vendor_only
            summary["nucleus"]["n_overlaps_lookup_only"] = tally.n_overlaps_lookup_only
            summary["nucleus"]["overlaps_agreement"] = agreement
    assert tally.per_cell is not None
    per_cell = {key: values[table_ids] for key, values in tally.per_cell.items()}
    if "nucleus" in per_cell:
        summary["nucleus"]["n_cells_with_nuclear_transcripts"] = int(
            np.count_nonzero(per_cell["nucleus"])
        )
    return summary, per_cell
