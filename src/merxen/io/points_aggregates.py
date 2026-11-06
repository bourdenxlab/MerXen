"""Read-only access to the transcript aggregates an Atera build records.

The Atera builder gathers transcript statistics in the same pass that writes
the points (``attrs["merxen_atera"]["transcripts"]``) and indexes the written
parquet files by bounding box (``merxen_atera_points_index.json`` at the store
root). With 1.4 billion rows per section, these replace full passes over the
points for counts, bounds, per-gene totals and the partition pruning of
windowed reads.

They describe the parquet files as the build wrote them, so a caller trusts
them only while :meth:`PointsAggregates.fingerprint_matches` holds: the store's
parquet listing and footers still agree with the index. Otherwise the caller
streams the points with dask (never ``to_pandas``).

XENIUM and MERSCOPE stores carry neither, so :func:`read_points_aggregates`
returns ``None`` for them and their callers keep their current path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

ATERA_ATTRS_KEY: Final = "merxen_atera"
"""Root attrs key of the Atera build header (``io.builders.atera``)."""

POINTS_INDEX_NAME: Final = "merxen_atera_points_index.json"
"""Store-root file name of the points index (``io.builders.atera``)."""

DEFAULT_POINTS_KEY: Final = "transcripts"

BBox = tuple[float, float, float, float]
"""A window ``(x0, y0, x1, y1)`` in microns, as elsewhere in MerXen."""


@dataclass(frozen=True)
class PointsFile:
    """One indexed transcript parquet file.

    Attributes:
        name: File name under the points directory, e.g. ``part.3.parquet``.
        n_rows: Rows the build wrote to the file.
        bbox_um: ``(x0, y0, x1, y1)`` of the file's rows from the parquet
            column statistics, or ``None`` when the footer had none. The index
            stores it as ``[xmin, xmax, ymin, ymax]``; it is reordered here.
    """

    name: str
    n_rows: int
    bbox_um: BBox | None


@dataclass(frozen=True)
class PointsAggregates:
    """Transcript aggregates and points index of one Atera store.

    Attributes:
        zarr_path: The store the aggregates were read from.
        points_key: Points element they describe.
        n_points_written: Rows written to the points element (the rows kept by
            the build's QV filter).
        feature_names: Feature names in ``gene_counts_kept`` order.
        gene_counts_kept: Rows written per feature.
        bounds_um: ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min``,
            ``z_max`` of the rows the build read (before the QV filter).
        density: Rows written per ``density_bin_um`` square, ``[ny, nx]``;
            rows outside the grid (negative coordinates) are counted in the
            nearest edge bin, so the grid sums to ``n_points_written``.
        density_bin_um: Edge length of a density bin.
        density_origin_um: ``(x, y)`` of the first density bin's corner.
        files: The indexed parquet files, in index order.
        index_fingerprint: ``n_files`` and ``n_rows`` the index records.
    """

    zarr_path: Path
    points_key: str
    n_points_written: int
    feature_names: tuple[str, ...]
    gene_counts_kept: np.ndarray
    bounds_um: dict[str, float]
    density: np.ndarray
    density_bin_um: float
    density_origin_um: tuple[float, float]
    files: tuple[PointsFile, ...] = field(default_factory=tuple)
    index_fingerprint: dict[str, int] = field(default_factory=dict)

    @property
    def points_dir(self) -> Path:
        """Directory holding the points element's parquet files."""
        return points_parquet_dir(self.zarr_path, self.points_key)

    def gene_counts(self) -> dict[str, int]:
        """Return ``feature name -> rows written`` for every feature."""
        return {
            name: int(count)
            for name, count in zip(
                self.feature_names, self.gene_counts_kept, strict=True
            )
        }

    def density_extent_um(self) -> BBox:
        """Return the ``(x0, y0, x1, y1)`` covered by the density grid."""
        n_y, n_x = self.density.shape
        x0, y0 = self.density_origin_um
        return (
            x0,
            y0,
            x0 + n_x * self.density_bin_um,
            y0 + n_y * self.density_bin_um,
        )

    def fingerprint_matches(self) -> bool:
        """Whether the store's parquet files still match the index.

        Compares the file listing of the points directory with the index
        (names, count) and each footer's row count with the index entry, and
        checks that the index total equals ``n_points_written``. Only footers
        are read.

        Returns:
            ``True`` when the aggregates describe the files on disk.
        """
        return _fingerprint_problem(self) is None

    def files_intersecting(self, bbox: BBox) -> list[Path]:
        """Return the parquet files whose rows may lie in a window.

        A file without a recorded bbox is always returned. Call this only when
        :meth:`fingerprint_matches` holds.

        Args:
            bbox: Window ``(x0, y0, x1, y1)`` in microns (edges inclusive).

        Returns:
            Paths of the overlapping files, in index order.
        """
        x0, y0, x1, y1 = (float(value) for value in bbox)
        if x1 < x0 or y1 < y0:
            raise ValueError(f"empty window {bbox}: need x0 <= x1 and y0 <= y1")
        selected = []
        for entry in self.files:
            box = entry.bbox_um
            if box is None or (
                box[0] <= x1 and box[2] >= x0 and box[1] <= y1 and box[3] >= y0
            ):
                selected.append(self.points_dir / entry.name)
        return selected


def points_parquet_dir(zarr_path: Path, points_key: str = DEFAULT_POINTS_KEY) -> Path:
    """Return the parquet directory of a points element of a store."""
    return Path(zarr_path, "points", points_key, "points.parquet")


def read_root_attrs(zarr_path: Path) -> dict[str, Any]:
    """Return the root attrs of a zarr store without opening it with zarr.

    Reads ``zarr.json`` (zarr v3) or ``.zattrs`` (zarr v2).

    Args:
        zarr_path: The store.

    Returns:
        The root attributes, or ``{}`` when the store has none.
    """
    root = Path(zarr_path)
    v3 = root / "zarr.json"
    if v3.is_file():
        attrs = json.loads(v3.read_text()).get("attributes")
        return dict(attrs) if isinstance(attrs, dict) else {}
    v2 = root / ".zattrs"
    if v2.is_file():
        attrs = json.loads(v2.read_text())
        return dict(attrs) if isinstance(attrs, dict) else {}
    return {}


def _index_files(
    zarr_path: Path, points_key: str
) -> tuple[tuple[PointsFile, ...], dict[str, int]] | None:
    path = Path(zarr_path) / POINTS_INDEX_NAME
    if not path.is_file():
        return None
    index = json.loads(path.read_text())
    element = (index.get("points") or {}).get(points_key)
    if not isinstance(element, dict):
        return None
    files = []
    for entry in element.get("files") or []:
        box = entry.get("bbox_um")
        bbox: BBox | None = None
        if isinstance(box, list) and len(box) == 4 and None not in box:
            xmin, xmax, ymin, ymax = (float(value) for value in box)
            bbox = (xmin, ymin, xmax, ymax)
        files.append(
            PointsFile(
                name=str(entry["file"]), n_rows=int(entry["n_rows"]), bbox_um=bbox
            )
        )
    fingerprint = {
        key: int(value) for key, value in dict(element.get("fingerprint") or {}).items()
    }
    return tuple(files), fingerprint


def read_points_aggregates(
    zarr_path: Path, points_key: str = DEFAULT_POINTS_KEY
) -> PointsAggregates | None:
    """Read the build aggregates and points index of a store.

    Args:
        zarr_path: A built Atera store or a latest store derived from one.
        points_key: Points element the aggregates describe.

    Returns:
        The aggregates, or ``None`` when the store records none (XENIUM and
        MERSCOPE stores, or a build without transcripts or index). The caller
        checks :meth:`PointsAggregates.fingerprint_matches` before using them.
    """
    zarr_path = Path(zarr_path)
    header = read_root_attrs(zarr_path).get(ATERA_ATTRS_KEY)
    if not isinstance(header, dict):
        return None
    transcripts = header.get("transcripts")
    if not isinstance(transcripts, dict) or "n_points_written" not in transcripts:
        return None
    if transcripts.get("points_index", POINTS_INDEX_NAME) != POINTS_INDEX_NAME:
        return None
    indexed = _index_files(zarr_path, points_key)
    if indexed is None:
        return None
    files, fingerprint = indexed
    origin = transcripts.get("density_origin_um") or (0.0, 0.0)
    return PointsAggregates(
        zarr_path=zarr_path,
        points_key=points_key,
        n_points_written=int(transcripts["n_points_written"]),
        feature_names=tuple(str(name) for name in transcripts.get("feature_names", ())),
        gene_counts_kept=np.asarray(
            transcripts.get("gene_counts_kept", ()), dtype=np.int64
        ),
        bounds_um={
            str(key): float(value)
            for key, value in dict(transcripts.get("bounds_um") or {}).items()
        },
        density=np.asarray(transcripts.get("density", ()), dtype=np.int64).reshape(
            _density_shape(transcripts.get("density"))
        ),
        density_bin_um=float(transcripts.get("density_bin_um", 0.0)),
        density_origin_um=(float(origin[0]), float(origin[1])),
        files=files,
        index_fingerprint=fingerprint,
    )


def _density_shape(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, list) or not raw:
        return (0, 0)
    first = raw[0]
    return (len(raw), len(first) if isinstance(first, list) else 0)


def _fingerprint_problem(aggregates: PointsAggregates) -> str | None:
    """Return why the aggregates no longer describe the files, or ``None``."""
    expected_files = aggregates.index_fingerprint.get("n_files")
    expected_rows = aggregates.index_fingerprint.get("n_rows")
    if expected_files is None or expected_rows is None:
        return "the index records no fingerprint"
    if expected_rows != aggregates.n_points_written:
        return (
            f"the index holds {expected_rows} rows, the build attrs "
            f"{aggregates.n_points_written}"
        )
    if len(aggregates.files) != expected_files:
        return f"the index lists {len(aggregates.files)} of {expected_files} files"
    directory = aggregates.points_dir
    if not directory.is_dir():
        return f"no points directory {directory.name}"
    on_disk = sorted(path.name for path in directory.glob("*.parquet"))
    indexed = sorted(entry.name for entry in aggregates.files)
    if on_disk != indexed:
        return f"{len(on_disk)} parquet files on disk, {len(indexed)} indexed"
    total = 0
    for entry in aggregates.files:
        try:
            n_rows = pq.read_metadata(directory / entry.name).num_rows
        except (OSError, ValueError) as error:
            return f"unreadable footer of {entry.name}: {error}"
        if n_rows != entry.n_rows:
            return f"{entry.name} holds {n_rows} rows, the index {entry.n_rows}"
        total += n_rows
    if total != expected_rows:
        return f"{total} rows on disk, {expected_rows} indexed"
    return None


def fingerprint_matches(zarr_path: Path, points_key: str = DEFAULT_POINTS_KEY) -> bool:
    """Whether a store has aggregates that still describe its parquet files.

    Args:
        zarr_path: The store.
        points_key: Points element.

    Returns:
        ``False`` for a store without aggregates or with a stale index.
    """
    aggregates = read_points_aggregates(zarr_path, points_key)
    if aggregates is None:
        return False
    problem = _fingerprint_problem(aggregates)
    if problem is not None:
        logger.info("Points aggregates of the store are stale: %s", problem)
        return False
    return True


def files_intersecting(
    zarr_path: Path, bbox: BBox, points_key: str = DEFAULT_POINTS_KEY
) -> list[Path] | None:
    """Return the parquet files of a store that may hold rows in a window.

    Args:
        zarr_path: The store.
        bbox: Window ``(x0, y0, x1, y1)`` in microns.
        points_key: Points element.

    Returns:
        The overlapping files, or ``None`` when the store has no index or the
        index is stale (the caller then reads every file).
    """
    aggregates = read_points_aggregates(zarr_path, points_key)
    if aggregates is None or not aggregates.fingerprint_matches():
        return None
    return aggregates.files_intersecting(bbox)
