#!/usr/bin/env python
"""Milestone-1 acceptance checks of an Atera SpatialData store (plan §1.1, §7.4).

The store is re-read from disk and checked against the raw bundle it was built
from, one criterion at a time. The streamed checks (schema, labels) hold
bounded memory; the matrix and table checks hold the vendor matrix and the
points counts in memory, which grows with the matrix nonzeros (about 21 GB on
the public dataset).

``build_options``
    The store's recorded build options are the §1.1 ones: ``min_qv`` 20,
    ``boundary_simplify_px`` 0.25 and no ``tile_allowlist`` (each can be
    changed on the command line for a deliberate non-§1.1 build).
``schema``
    ``validate_merxen_schema(deep=False)`` and the streaming deep check
    (``validate_merxen_schema_streaming``): unique ``transcript_id``,
    ``original_assignment`` within the cell shape IDs, table IDs equal to shape
    IDs, and points rows per source tile equal to the raw tile rows that pass
    the build's QV filter (counted here from the raw ``quality_score``).
``points_vs_matrix``
    A COO accumulation of the points (``qv >= min_qv``, assigned) over every
    feature, compared with the vendor matrix ``X`` in int64: equal on every
    column with transcripts (vendor all-QV ``gene_counts`` above 0). The only
    excess allowed is on columns with no transcripts at all, and every such
    column must be a ``DeprecatedCodeword_*`` one (the vendor quirk recorded in
    the plan); ``--expected-vendor-excess`` pins its total and is required
    unless ``--allow-unpinned-excess`` is given.
``labels_vs_cell_summary``
    ``cell_labels`` pixel count x ps² against ``cell_summary.cell_area``
    (largest difference at most 1e-4 µm²), label centroids at
    ``(mean index + 0.5) * ps`` against the vendor centroids (0.005 µm), the
    largest label equal to the cell count, no empty label, and every pixel
    equal to the vendor ``masks/1``.
``nucleus_containment``
    Every ``nucleus_labels`` pixel lies on its owning cell's label
    (``polygon_sets/0/cell_index[label - 1] + 1``); the largest nucleus label
    equals the nucleus count, no nucleus label is empty, and every pixel equals
    the vendor ``masks/0``.
``shape_centroids``
    Exactly one ``cell_boundaries`` shape per cell, indexed by its label, with
    the area-weighted centroid within 0.01 µm of ``cell_summary`` (the
    half-pixel check); nucleus shapes indexed by nucleus row + 1 and owned by
    ``cell_instance_id`` = owning cell row + 1; the vertex statistics.
``reopen``, ``um_per_px``
    ``sd.read_zarr`` opens the store and ``find_um_per_px(store, "ATERA")``
    returns the bundle's pixel size.
``table_vs_vendor``
    The table against the raw matrix and ``cell_summary``, read here straight
    from the zips: ``instance_id`` = vendor row + 1, ``X`` equal to the vendor
    gene columns, every vendor ``obs`` column (``vendor_<name>``), the
    ``cell_summary`` values (areas, ``obsm["spatial"]``, nucleus centroids as
    ``vendor_<name>``), ``obsm["X_pca_vendor"/"X_umap_vendor"]``,
    ``varm["vendor_de"]`` and the category colours.
``metadata``
    Every vendor sidecar byte-identical to the bundle file with its SHA-256
    recorded; the full header (``experiment.spatial``, every metrics column
    against the CSV text, the ``panel_config.json`` header, the vendor zarr root
    attributes, the map annotations) in ``attrs["merxen_atera"]``; the zarr
    metadata members (listed here from the zip) and the OME-XML kept verbatim.
``build_marker``
    The completion marker exists, its fingerprint equals the header's and
    one recomputed from the bundle and the recorded options, and no
    ``.partial-*`` build folder is left beside the store.

The JSON report holds pass/fail, details, wall time, the process peak RSS
after each criterion and the git revision of this checkout; it names the
dataset only by ``--alias`` and holds no host path or vendor manifest value.
Exit status 0 when every criterion passed.

Example::

    python scripts/acceptance/atera_build.py \\
        --store $MERXEN_RESULTS_ROOT/atera_pilot/<alias>/source.zarr \\
        --bundle $MERXEN_ATERA_RAW_ROOT/<dataset>/outs --alias ATERA_PUBLIC \\
        --report $MERXEN_EVIDENCE_ROOT/atera_at5/<alias>_acceptance.json \\
        --expected-vendor-excess 84
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import resource
import subprocess
import sys
import time
import traceback
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import spatialdata as sd
import tifffile
import zarr
from scipy import sparse

from merxen.config import AteraBuildConfig
from merxen.io import atera_reader as ar
from merxen.io.builders import atera as atera_builder
from merxen.io.spatialdata_schema import (
    INSTANCE_ID_COLUMN,
    SOURCE_CELL_ID_COLUMN,
    validate_merxen_schema,
    validate_merxen_schema_streaming,
)
from merxen.platforms import find_um_per_px

logger = logging.getLogger("atera_acceptance")

#: Tolerances of plan §1.1 item 4.
AREA_TOLERANCE_UM2 = 1e-4
LABEL_CENTROID_TOLERANCE_UM = 0.005
SHAPE_CENTROID_TOLERANCE_UM = 0.01
#: Feature-name prefix of the only columns the vendor matrix may count
#: without any transcript (plan §1.1 item 3).
DEPRECATED_CODEWORD_PREFIX = "DeprecatedCodeword_"
#: Points rows accumulated before the per-file count matrices are merged.
DEFAULT_MERGE_ROWS = 1 << 26
#: Mismatching entries listed in a report (the counts are always complete).
MAX_LISTED = 20
#: Build options of plan §1.1 (as amended: 0.25 px simplification).
EXPECTED_MIN_QV = 20.0
EXPECTED_BOUNDARY_SIMPLIFY_PX = 0.25
#: Zarr metadata member names (v2 and v3), listed independently of the reader.
ZARR_METADATA_NAMES = frozenset(
    {".zattrs", ".zgroup", ".zarray", ".zmetadata", "zarr.json"}
)
#: Vendor ``obs`` columns the table does not copy: the index and the float32
#: areas (the table's areas come from ``cell_summary``).
VENDOR_OBS_NOT_COPIED = frozenset({"cell_area", "nucleus_area"})
#: ``cell_summary`` columns with a named place in the table.
SUMMARY_NAMED_COLUMNS = {
    "cell_area": "cell_area",
    "nucleus_area": "nucleus_area",
    "nucleus_count": "nucleus_count",
}
#: Matrix rows compared at once in the table check.
TABLE_ROW_BLOCK = 16_384

CRITERIA = (
    "build_options",
    "reopen",
    "schema",
    "points_vs_matrix",
    "labels_vs_cell_summary",
    "nucleus_containment",
    "shape_centroids",
    "um_per_px",
    "table_vs_vendor",
    "metadata",
    "build_marker",
)


class CheckFailedError(AssertionError):
    """A criterion did not hold; ``details`` says by how much."""

    def __init__(self, message: str, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9


@dataclass
class Context:
    """What the checks share: the store, the bundle and lazily read records."""

    store: Path
    bundle: ar.AteraBundle
    header: dict[str, Any]
    build_config: AteraBuildConfig
    n_workers: int
    merge_rows: int = DEFAULT_MERGE_ROWS
    _cells: ar.AteraCells | None = field(default=None, repr=False)
    _sdata: Any = field(default=None, repr=False)

    @property
    def min_qv(self) -> float | None:
        """QV filter the store was built with."""
        min_qv: float | None = self.build_config.min_qv
        return min_qv

    @property
    def cells(self) -> ar.AteraCells:
        """Raw cell records of the bundle."""
        if self._cells is None:
            self._cells = ar.read_atera_cells(self.bundle)
        return self._cells

    @property
    def sdata(self) -> Any:
        """The store, opened lazily with ``sd.read_zarr``."""
        if self._sdata is None:
            self._sdata = sd.read_zarr(self.store)
        return self._sdata


# --------------------------------------------------------------------------- #
# Raw bundle reads
# --------------------------------------------------------------------------- #


def raw_rows_per_tile(
    bundle: ar.AteraBundle,
    min_qv: float | None,
    *,
    n_workers: int,
    rows_per_read: int = 1 << 22,
) -> dict[str, int]:
    """Count each raw tile's rows that pass the QV filter, streaming ``quality_score``.

    Args:
        bundle: The raw bundle.
        min_qv: QV filter (``None`` keeps every row).
        n_workers: Tiles read at once.
        rows_per_read: Rows decoded per read.

    Returns:
        Rows kept per tile key (tiles without rows included as 0).
    """
    tiles = ar.read_atera_tile_index(bundle)

    def count(tile: ar.AteraTile) -> tuple[str, int]:
        if min_qv is None or tile.n_rows == 0:
            return tile.key, tile.n_rows
        kept = 0
        with ar.open_zip_group(bundle.transcripts_zip) as root:
            quality = root["grid"][tile.key]["quality_score"]
            for start in range(0, tile.n_rows, rows_per_read):
                block = np.asarray(quality[start : start + rows_per_read]).ravel()
                kept += int(np.count_nonzero(block >= min_qv))
        return tile.key, kept

    with ThreadPoolExecutor(max(1, n_workers)) as pool:
        return dict(pool.map(count, tiles))


def raw_gene_counts_all_qv(bundle: ar.AteraBundle) -> np.ndarray:
    """Vendor all-QV transcripts per feature (``grid`` ``gene_counts``)."""
    with ar.open_zip_group(bundle.transcripts_zip) as root:
        return np.asarray(root["grid"].attrs["gene_counts"], dtype=np.int64)


def raw_matrix(bundle: ar.AteraBundle) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Vendor matrix ``X`` (int64 CSR) and its row barcodes."""
    with ar.open_zip_group(bundle.cell_feature_matrix_zip) as group:
        matrix = sparse.csr_matrix(ad.io.read_elem(group["X"]))
        barcodes = ar._cfm_index(group["obs"])
    matrix.data = matrix.data.astype(np.int64, copy=False)
    return matrix, np.asarray(barcodes, dtype=object)


# --------------------------------------------------------------------------- #
# Criterion: schema and per-tile reconciliation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExpectedOptions:
    """Build options the store must have been built with (plan §1.1)."""

    min_qv: float | None = EXPECTED_MIN_QV
    boundary_simplify_px: float = EXPECTED_BOUNDARY_SIMPLIFY_PX
    allow_tile_allowlist: bool = False


def check_build_options(ctx: Context, expected: ExpectedOptions) -> dict[str, Any]:
    """The recorded build options are the §1.1 ones.

    The other checks take the options from the store's own header, so a store
    simplified at another tolerance or built from a tile subset would pass
    them; this check pins the options.
    """
    recorded = dict(ctx.header.get("build", {}).get("options") or {})
    config = ctx.build_config
    details = {
        "recorded": {
            "min_qv": recorded.get("min_qv"),
            "boundary_simplify_px": recorded.get("boundary_simplify_px"),
            "tile_allowlist": recorded.get("tile_allowlist"),
        },
        "expected": {
            "min_qv": expected.min_qv,
            "boundary_simplify_px": expected.boundary_simplify_px,
            "tile_allowlist_allowed": expected.allow_tile_allowlist,
        },
    }
    failures = []
    if not recorded:
        failures.append("the store records no build options")
    if config.min_qv != expected.min_qv:
        failures.append(f"min_qv is {config.min_qv}, not {expected.min_qv}")
    if not math.isclose(
        config.boundary_simplify_px, expected.boundary_simplify_px, abs_tol=1e-12
    ):
        failures.append(
            f"boundary_simplify_px is {config.boundary_simplify_px}, "
            f"not {expected.boundary_simplify_px}"
        )
    if config.tile_allowlist is not None and not expected.allow_tile_allowlist:
        failures.append("the store was built from a tile allow-list")
    if failures:
        raise CheckFailedError("; ".join(failures), details)
    return details


def check_reopen(ctx: Context) -> dict[str, Any]:
    """``sd.read_zarr`` opens the store; list its elements."""
    sdata = ctx.sdata
    return {
        kind: sorted(map(str, getattr(sdata, kind)))
        for kind in ("images", "labels", "points", "shapes", "tables")
    }


def check_schema(ctx: Context) -> dict[str, Any]:
    """Shallow schema, streaming deep check and per-tile row reconciliation."""
    validate_merxen_schema(ctx.sdata, deep=False)
    expected = raw_rows_per_tile(ctx.bundle, ctx.min_qv, n_workers=ctx.n_workers)
    allowlist = ctx.build_config.tile_allowlist
    if allowlist is not None:
        expected = {key: rows for key, rows in expected.items() if key in allowlist}
    summaries = validate_merxen_schema_streaming(
        ctx.sdata,
        group_column="source_tile",
        expected_group_rows={atera_builder.POINTS_KEY: expected},
    )
    summary = summaries[atera_builder.POINTS_KEY]
    n_expected = sum(expected.values())
    if summary.n_rows != n_expected:
        raise CheckFailedError(
            "points rows differ from the raw rows passing the QV filter",
            {"points_rows": summary.n_rows, "raw_rows_kept": n_expected},
        )
    recorded = ctx.header.get("transcripts", {}).get("n_points_written")
    if recorded is not None and int(recorded) != n_expected:
        raise CheckFailedError(
            "attrs n_points_written differs from the points rows",
            {"n_points_written": recorded, "points_rows": summary.n_rows},
        )
    return {
        "points_rows": summary.n_rows,
        "raw_rows_kept": n_expected,
        "n_partitions": len(summary.partition_rows),
        "n_tiles_with_rows": sum(1 for rows in expected.values() if rows),
        "transcript_id_range": [summary.transcript_id_min, summary.transcript_id_max],
        "reread_partitions": summary.reread_partitions,
        "rows_per_tile": dict(sorted(expected.items())),
    }


# --------------------------------------------------------------------------- #
# Criterion: points reproduce the vendor matrix
# --------------------------------------------------------------------------- #


def points_files(store: Path) -> list[Path]:
    """Parquet files of the transcripts points element, in name order."""
    directory = store / "points" / atera_builder.POINTS_KEY / "points.parquet"
    files = sorted(
        directory.glob("*.parquet"),
        key=lambda path: (len(path.name), path.name),
    )
    if not files:
        raise CheckFailedError("the store has no transcripts parquet files")
    return files


def _file_counts(
    path: Path,
    *,
    feature_index: pd.Index,
    instance_rows: np.ndarray,
    min_qv: float | None,
    shape: tuple[int, int],
) -> tuple[sparse.csr_matrix, int]:
    """Count one parquet file's kept, assigned points per cell row x feature."""
    table = pq.read_table(path, columns=["feature_name", "qv", "original_assignment"])
    names = table.column("feature_name").combine_chunks()
    if hasattr(names, "dictionary"):
        lookup = feature_index.get_indexer(names.dictionary.to_pylist())
        if (lookup < 0).any():
            raise CheckFailedError(
                "points feature names absent from the vendor matrix",
                {"file": path.name},
            )
        codes = names.indices.to_numpy(zero_copy_only=False)
        features = lookup[codes.astype(np.int64)]
    else:
        features = feature_index.get_indexer(names.to_pylist())
        if (features < 0).any():
            raise CheckFailedError(
                "points feature names absent from the vendor matrix",
                {"file": path.name},
            )
    assignment = table.column("original_assignment").combine_chunks()
    assigned = ~assignment.is_null().to_numpy(zero_copy_only=False)
    keep = assigned.copy()
    if min_qv is not None:
        qv = table.column("qv").to_numpy()
        keep &= qv >= min_qv
    ids = assignment.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)[keep]
    if ids.size and (ids.max() >= len(instance_rows) or ids.min() < 1):
        raise CheckFailedError(
            "points original_assignment outside the table instance IDs",
            {"file": path.name},
        )
    rows = instance_rows[ids]
    if (rows < 0).any():
        raise CheckFailedError(
            "points original_assignment names no vendor matrix row",
            {"file": path.name},
        )
    counts = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.int64), (rows, features[keep])), shape=shape
    ).tocsr()
    counts.sum_duplicates()
    return counts, int(rows.size)


def accumulate_point_counts(
    files: Sequence[Path],
    *,
    feature_index: pd.Index,
    instance_rows: np.ndarray,
    min_qv: float | None,
    shape: tuple[int, int],
    n_workers: int,
    merge_rows: int = DEFAULT_MERGE_ROWS,
) -> sparse.csr_matrix:
    """Stream the points files into an int64 cell x feature count matrix.

    Files are counted on ``n_workers`` threads; their small matrices are merged
    into the running total once ``merge_rows`` points are pending, so memory is
    the total's nonzeros plus a bounded batch.

    Args:
        files: Points parquet files.
        feature_index: Vendor feature names (matrix column order).
        instance_rows: Matrix row of each table ``instance_id`` (-1: none).
        min_qv: QV filter.
        shape: Matrix shape.
        n_workers: Files read at once.
        merge_rows: Points pending before a merge.

    Returns:
        The counts as canonical int64 CSR.
    """
    total = sparse.csr_matrix(shape, dtype=np.int64)
    pending: list[sparse.csr_matrix] = []
    pending_rows = 0

    def flush() -> None:
        nonlocal total, pending, pending_rows
        if pending:
            batch = pending[0]
            for part in pending[1:]:
                batch = batch + part
            total = total + batch
            pending = []
            pending_rows = 0

    def count(path: Path) -> tuple[sparse.csr_matrix, int]:
        return _file_counts(
            path,
            feature_index=feature_index,
            instance_rows=instance_rows,
            min_qv=min_qv,
            shape=shape,
        )

    with ThreadPoolExecutor(max(1, n_workers)) as pool:
        for counts, n_rows in pool.map(count, files):
            pending.append(counts)
            pending_rows += n_rows
            if pending_rows >= merge_rows:
                flush()
    flush()
    total.sum_duplicates()
    return total


@dataclass(frozen=True)
class MatrixComparison:
    """Points counts against the vendor matrix."""

    passed: bool
    details: dict[str, Any]


def compare_matrix_counts(
    vendor: sparse.csr_matrix,
    points: sparse.csr_matrix,
    *,
    feature_names: Sequence[str],
    feature_types: Sequence[str],
    gene_counts_all_qv: np.ndarray,
    expected_excess: int | None = None,
) -> MatrixComparison:
    """Compare the points counts with the vendor matrix in int64.

    Columns with transcripts (vendor all-QV ``gene_counts`` above 0) must be
    equal entry by entry. Columns without any transcript cannot be counted
    from points, so the vendor counts there are the excess; it is allowed only
    on ``DeprecatedCodeword_*`` columns, and must equal ``expected_excess``
    when that is given.

    Args:
        vendor: Vendor ``X`` (int64 CSR).
        points: Counts from the points.
        feature_names: Column names.
        feature_types: Column feature types.
        gene_counts_all_qv: Vendor all-QV transcripts per column.
        expected_excess: Expected total vendor count on columns without
            transcripts.

    Returns:
        Pass/fail with the totals and any mismatch.
    """
    names = np.asarray(feature_names, dtype=object)
    types = np.asarray(feature_types, dtype=object)
    no_transcripts = np.asarray(gene_counts_all_qv) == 0
    diff = (vendor.astype(np.int64) - points.astype(np.int64)).tocoo()
    nonzero = diff.data != 0
    rows, cols, values = diff.row[nonzero], diff.col[nonzero], diff.data[nonzero]
    on_tx = ~no_transcripts[cols]
    excess_cols = cols[~on_tx]
    excess_values = values[~on_tx]
    excess_by_column: dict[str, int] = {}
    for column in np.unique(excess_cols):
        excess_by_column[str(names[column])] = int(
            excess_values[excess_cols == column].sum()
        )
    excess_total = int(excess_values.sum())
    not_deprecated = sorted(
        name
        for name in excess_by_column
        if not name.startswith(DEPRECATED_CODEWORD_PREFIX)
    )
    vendor_col_totals = np.asarray(vendor.sum(axis=0, dtype=np.int64)).ravel()
    points_col_totals = np.asarray(points.sum(axis=0, dtype=np.int64)).ravel()
    is_gene = types == ar.GENE_EXPRESSION_TYPE
    points_total = int(points_col_totals.sum())
    tx_cols = ~no_transcripts
    details: dict[str, Any] = {
        "vendor_total": int(vendor_col_totals.sum()),
        "vendor_gene_total": int(vendor_col_totals[is_gene].sum()),
        "vendor_total_on_columns_with_transcripts": int(
            vendor_col_totals[tx_cols].sum()
        ),
        "points_total": points_total,
        "points_gene_total": int(points_col_totals[is_gene].sum()),
        "n_columns": int(len(names)),
        "n_columns_without_transcripts": int(no_transcripts.sum()),
        "vendor_nnz": int(vendor.nnz),
        "points_nnz": int(points.nnz),
        "mismatched_entries_on_columns_with_transcripts": int(on_tx.sum()),
        "mismatched_count_on_columns_with_transcripts": int(
            np.abs(values[on_tx]).sum()
        ),
        "excess_on_columns_without_transcripts": excess_total,
        "excess_by_column": excess_by_column,
        "negative_excess_entries": int((excess_values < 0).sum()),
        "excess_on_non_deprecated_columns": not_deprecated,
        "expected_excess": expected_excess,
    }
    if on_tx.any():
        details["first_mismatches"] = [
            {
                "matrix_row": int(r),
                "feature": str(names[c]),
                "vendor_minus_points": int(v),
            }
            for r, c, v in list(
                zip(rows[on_tx], cols[on_tx], values[on_tx], strict=True)
            )[:MAX_LISTED]
        ]
    passed = (
        not on_tx.any()
        and not not_deprecated
        and details["negative_excess_entries"] == 0
        and (expected_excess is None or excess_total == expected_excess)
    )
    return MatrixComparison(passed, details)


def table_instance_rows(sdata: Any, barcodes: np.ndarray) -> np.ndarray:
    """Vendor matrix row of each table ``instance_id`` (index), -1 when none.

    Rows are matched by the 10x cell ID string, not by position.
    """
    obs = sdata.tables["table"].obs
    instance_ids = obs[INSTANCE_ID_COLUMN].to_numpy(dtype=np.int64)
    position = pd.Index(barcodes).get_indexer(obs[SOURCE_CELL_ID_COLUMN].astype(str))
    if (position < 0).any():
        raise CheckFailedError(
            "table cells absent from the vendor matrix",
            {"n_missing": int((position < 0).sum())},
        )
    lookup = np.full(int(instance_ids.max(initial=0)) + 1, -1, dtype=np.int64)
    lookup[instance_ids] = position
    return lookup


def check_points_vs_matrix(
    ctx: Context,
    expected_excess: int | None = None,
    *,
    allow_unpinned_excess: bool = False,
) -> dict[str, Any]:
    """Points (QV filter, assigned, all features) reproduce the vendor matrix.

    The deprecated-codeword excess must be pinned (``expected_excess``) unless
    ``allow_unpinned_excess``; an unpinned run fails after reporting it.
    """
    features = ar.read_atera_features(ctx.bundle)
    feature_names = features["feature_name"].astype(str).tolist()
    feature_index = pd.Index(feature_names)
    if not feature_index.is_unique:
        raise CheckFailedError("vendor feature names are not unique")
    vendor, barcodes = raw_matrix(ctx.bundle)
    instance_rows = table_instance_rows(ctx.sdata, barcodes)
    shape = vendor.shape
    del vendor
    points = accumulate_point_counts(
        points_files(ctx.store),
        feature_index=feature_index,
        instance_rows=instance_rows,
        min_qv=ctx.min_qv,
        shape=shape,
        n_workers=ctx.n_workers,
        merge_rows=ctx.merge_rows,
    )
    vendor, _ = raw_matrix(ctx.bundle)
    comparison = compare_matrix_counts(
        vendor,
        points,
        feature_names=feature_names,
        feature_types=features["feature_type"].astype(str).tolist(),
        gene_counts_all_qv=raw_gene_counts_all_qv(ctx.bundle),
        expected_excess=expected_excess,
    )
    if not comparison.passed:
        raise CheckFailedError(
            "points do not reproduce the vendor matrix", comparison.details
        )
    if expected_excess is None and not allow_unpinned_excess:
        raise CheckFailedError(
            "the vendor excess is not pinned: pass --expected-vendor-excess "
            "(or --allow-unpinned-excess)",
            comparison.details,
        )
    return comparison.details


# --------------------------------------------------------------------------- #
# Criteria: labels, nucleus containment, shapes
# --------------------------------------------------------------------------- #


def _level0_array(store: Path, labels_key: str) -> Any:
    """Level-0 array of a labels element, read with zarr."""
    group = zarr.open_group(store / "labels" / labels_key, mode="r")
    attrs: dict[str, Any] = dict(group.attrs.asdict())
    multiscales: list[Any] = (attrs.get("ome") or attrs).get("multiscales") or []
    path = multiscales[0]["datasets"][0]["path"] if multiscales else "0"
    return group[path]


def _blocks(
    shape: tuple[int, int], chunks: tuple[int, int]
) -> Iterator[tuple[slice, slice]]:
    for row in range(0, shape[0], chunks[0]):
        for col in range(0, shape[1], chunks[1]):
            yield (
                slice(row, min(shape[0], row + chunks[0])),
                slice(col, min(shape[1], col + chunks[1])),
            )


@dataclass
class LabelStats:
    """Streaming pixel statistics of the two label images."""

    cell_counts: np.ndarray
    cell_row_sums: np.ndarray
    cell_col_sums: np.ndarray
    nucleus_counts: np.ndarray
    containment_violations: int
    max_cell_label: int
    max_nucleus_label: int
    #: Pixels that differ from the vendor ``masks/1`` / ``masks/0`` (``None``
    #: when the masks were not compared).
    cell_vendor_mismatch: int | None = None
    nucleus_vendor_mismatch: int | None = None


def stream_label_stats(
    store: Path,
    *,
    n_cells: int,
    nucleus_owner: np.ndarray,
    n_workers: int,
    block: tuple[int, int] | None = None,
    strip_rows: int = 512,
    vendor_cells_zip: Path | None = None,
) -> LabelStats:
    """One pass over both level-0 label images, block by block.

    Args:
        store: The store.
        n_cells: Cell records (the largest valid cell label).
        nucleus_owner: Owning cell row of each nucleus row.
        n_workers: Blocks processed at once.
        block: Block shape; defaults to the cell labels' chunk shape.
        strip_rows: Rows of a block counted at once (bounds the working
            copies to about 40 B per pixel of a strip).
        vendor_cells_zip: The bundle's ``cells.zarr.zip``; when given, every
            block is also compared pixel by pixel with the vendor ``masks/1``
            (cells) and ``masks/0`` (nuclei).

    Returns:
        Per-label pixel counts and index sums, nucleus pixel counts, the
        number of nucleus pixels off their owning cell's label and, with
        ``vendor_cells_zip``, the pixels that differ from the vendor masks.
    """
    cell_array = _level0_array(store, "cell_labels")
    nucleus_array = _level0_array(store, "nucleus_labels")
    shape = (int(cell_array.shape[0]), int(cell_array.shape[1]))
    if tuple(int(n) for n in nucleus_array.shape) != shape:
        raise CheckFailedError(
            "cell and nucleus labels differ in shape",
            {"cell": list(shape), "nucleus": list(nucleus_array.shape)},
        )
    step = block or (int(cell_array.chunks[0]), int(cell_array.chunks[1]))
    n_nuclei = len(nucleus_owner)
    expected_cell = np.concatenate([[0], nucleus_owner.astype(np.int64) + 1])
    if vendor_cells_zip is not None:
        with ar.open_zip_group(vendor_cells_zip) as group:
            vendor_shapes = {
                tuple(int(n) for n in group[f"masks/{key}"].shape) for key in "01"
            }
        if vendor_shapes != {shape}:
            raise CheckFailedError(
                "the vendor masks differ in shape from the labels",
                {"labels": list(shape), "vendor": sorted(map(list, vendor_shapes))},
            )

    def vendor_mismatch(
        window: tuple[slice, slice], cells: Any, nuclei: Any
    ) -> list[int]:
        # Each block opens its own zip store (no shared store lock).
        with ar.open_zip_group(vendor_cells_zip) as group:
            return [
                int(np.count_nonzero(np.asarray(group[member][window]) != ours))
                for member, ours in (("masks/1", cells), ("masks/0", nuclei))
            ]

    def one(window: tuple[slice, slice]) -> tuple[np.ndarray, ...]:
        cell_block = np.asarray(cell_array[window])
        nucleus_block = np.asarray(nucleus_array[window])
        mismatch = (
            vendor_mismatch(window, cell_block, nucleus_block)
            if vendor_cells_zip is not None
            else [0, 0]
        )
        top_cell = int(cell_block.max(initial=0))
        top_nucleus = int(nucleus_block.max(initial=0))
        size = max(n_cells, top_cell) + 1
        counts = np.zeros(size, dtype=np.int64)
        row_sums = np.zeros(size)
        col_sums = np.zeros(size)
        nucleus_counts = np.zeros(max(n_nuclei, top_nucleus) + 1, dtype=np.int64)
        violations = 0
        width = cell_block.shape[1]
        cols = np.arange(window[1].start, window[1].start + width, dtype=np.float64)
        # Row strips keep the int64 and float64 working copies small.
        for start in range(0, cell_block.shape[0], strip_rows):
            cells = cell_block[start : start + strip_rows].astype(np.int64).ravel()
            nuclei = nucleus_block[start : start + strip_rows].astype(np.int64).ravel()
            height = len(cells) // max(1, width)
            rows = np.repeat(
                np.arange(height, dtype=np.float64) + window[0].start + start, width
            )
            counts += np.bincount(cells, minlength=size)
            row_sums += np.bincount(cells, weights=rows, minlength=size)
            col_sums += np.bincount(
                cells, weights=np.tile(cols, height), minlength=size
            )
            nucleus_counts += np.bincount(nuclei, minlength=len(nucleus_counts))
            on_nucleus = nuclei > 0
            labels = nuclei[on_nucleus]
            valid = labels <= n_nuclei
            owner = np.zeros(labels.shape, dtype=np.int64)
            owner[valid] = expected_cell[labels[valid]]
            violations += int(
                np.count_nonzero(cells[on_nucleus] != owner) + np.count_nonzero(~valid)
            )
        return (
            counts,
            row_sums,
            col_sums,
            nucleus_counts,
            np.asarray([violations, top_cell, top_nucleus, *mismatch]),
        )

    cell_counts = np.zeros(n_cells + 1, dtype=np.int64)
    row_sums = np.zeros(n_cells + 1)
    col_sums = np.zeros(n_cells + 1)
    nucleus_counts = np.zeros(n_nuclei + 1, dtype=np.int64)
    violations = top_cell = top_nucleus = cell_mismatch = nucleus_mismatch = 0

    def grow(values: np.ndarray, size: int) -> np.ndarray:
        return (
            values if len(values) >= size else np.pad(values, (0, size - len(values)))
        )

    with ThreadPoolExecutor(max(1, n_workers)) as pool:
        for counts, rs, cs, ncounts, scalars in pool.map(one, _blocks(shape, step)):
            cell_counts = grow(cell_counts, len(counts))
            row_sums = grow(row_sums, len(counts))
            col_sums = grow(col_sums, len(counts))
            cell_counts[: len(counts)] += counts
            row_sums[: len(counts)] += rs
            col_sums[: len(counts)] += cs
            nucleus_counts = grow(nucleus_counts, len(ncounts))
            nucleus_counts[: len(ncounts)] += ncounts
            violations += int(scalars[0])
            top_cell = max(top_cell, int(scalars[1]))
            top_nucleus = max(top_nucleus, int(scalars[2]))
            cell_mismatch += int(scalars[3])
            nucleus_mismatch += int(scalars[4])
    compared = vendor_cells_zip is not None
    return LabelStats(
        cell_counts=cell_counts,
        cell_row_sums=row_sums,
        cell_col_sums=col_sums,
        nucleus_counts=nucleus_counts,
        containment_violations=violations,
        max_cell_label=top_cell,
        max_nucleus_label=top_nucleus,
        cell_vendor_mismatch=cell_mismatch if compared else None,
        nucleus_vendor_mismatch=nucleus_mismatch if compared else None,
    )


def _label_stats(ctx: Context, cache: dict[str, Any]) -> LabelStats:
    if "labels" not in cache:
        cache["labels"] = stream_label_stats(
            ctx.store,
            n_cells=ctx.cells.n_cells,
            nucleus_owner=ctx.cells.nucleus_cell_index,
            n_workers=ctx.n_workers,
            vendor_cells_zip=ctx.bundle.cells_zip,
        )
    stats: LabelStats = cache["labels"]
    return stats


def check_labels(ctx: Context, cache: dict[str, Any]) -> dict[str, Any]:
    """Cell label areas and centroids against ``cell_summary``."""
    stats = _label_stats(ctx, cache)
    cells = ctx.cells
    n_cells = cells.n_cells
    ps = ctx.bundle.pixel_size_um
    counts = stats.cell_counts[1 : n_cells + 1]
    summary = cells.summary
    area = counts * ps**2
    area_diff = np.abs(area - summary["cell_area"].to_numpy(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = (stats.cell_col_sums[1 : n_cells + 1] / counts + 0.5) * ps
        cy = (stats.cell_row_sums[1 : n_cells + 1] / counts + 0.5) * ps
    centroid_diff = np.hypot(
        cx - summary["cell_centroid_x"].to_numpy(np.float64),
        cy - summary["cell_centroid_y"].to_numpy(np.float64),
    )
    empty = int(np.count_nonzero(counts == 0))
    details = {
        "n_cells": n_cells,
        "max_cell_label": stats.max_cell_label,
        "empty_cell_labels": empty,
        "labelled_pixels": int(counts.sum()),
        "area_max_abs_diff_um2": float(np.nanmax(area_diff)) if n_cells else 0.0,
        "centroid_max_diff_um": float(np.nanmax(centroid_diff)) if n_cells else 0.0,
        "area_over_tolerance": int(
            np.count_nonzero(~(area_diff <= AREA_TOLERANCE_UM2))
        ),
        "centroid_over_tolerance": int(
            np.count_nonzero(~(centroid_diff <= LABEL_CENTROID_TOLERANCE_UM))
        ),
        "area_tolerance_um2": AREA_TOLERANCE_UM2,
        "centroid_tolerance_um": LABEL_CENTROID_TOLERANCE_UM,
        "pixels_differing_from_vendor_mask": stats.cell_vendor_mismatch,
    }
    failures = []
    if stats.cell_vendor_mismatch:
        failures.append("cell labels differ from the vendor masks/1")
    if stats.max_cell_label != n_cells:
        failures.append("the largest cell label is not the cell count")
    if empty:
        failures.append("some cell labels are empty")
    if details["area_over_tolerance"]:
        failures.append("cell label areas differ from cell_summary")
    if details["centroid_over_tolerance"]:
        failures.append("cell label centroids differ from cell_summary")
    if failures:
        raise CheckFailedError("; ".join(failures), details)
    return details


def check_nucleus_containment(ctx: Context, cache: dict[str, Any]) -> dict[str, Any]:
    """Every nucleus pixel lies on its owning cell's label."""
    stats = _label_stats(ctx, cache)
    n_nuclei = ctx.cells.n_nuclei
    counts = stats.nucleus_counts[1 : n_nuclei + 1]
    empty = int(np.count_nonzero(counts == 0))
    details = {
        "n_nuclei": n_nuclei,
        "max_nucleus_label": stats.max_nucleus_label,
        "empty_nucleus_labels": empty,
        "nucleus_pixels": int(counts.sum()),
        "containment_violations": stats.containment_violations,
        "pixels_differing_from_vendor_mask": stats.nucleus_vendor_mismatch,
    }
    failures = []
    if stats.nucleus_vendor_mismatch:
        failures.append("nucleus labels differ from the vendor masks/0")
    if stats.containment_violations:
        failures.append("nucleus pixels lie off their owning cell")
    if stats.max_nucleus_label != n_nuclei:
        failures.append("the largest nucleus label is not the nucleus count")
    if empty:
        failures.append("some nucleus labels are empty")
    if failures:
        raise CheckFailedError("; ".join(failures), details)
    return details


def _int_index(frame: Any) -> np.ndarray | None:
    """A shapes index as int64, or ``None`` when it is not integral."""
    try:
        return np.asarray(frame.index).astype(np.int64)
    except (TypeError, ValueError):
        return None


def check_shape_centroids(ctx: Context) -> dict[str, Any]:
    """One shape per cell, indexed by its label, centroid within 0.01 µm.

    The centroid tolerance is the half-pixel check against ``cell_summary``;
    nucleus shapes must be indexed by nucleus row + 1 and owned by
    ``cell_instance_id`` = owning cell row + 1.
    """
    cells = ctx.cells
    shapes = ctx.sdata.shapes[atera_builder.CELL_SHAPES_KEY]
    position = pd.Index(cells.source_cell_id).get_indexer(
        shapes[SOURCE_CELL_ID_COLUMN].astype(str)
    )
    if (position < 0).any():
        raise CheckFailedError(
            "cell shapes name unknown cells", {"n_unknown": int((position < 0).sum())}
        )
    centroids = shapely.centroid(shapes.geometry.to_numpy())
    x, y = shapely.get_x(centroids), shapely.get_y(centroids)
    summary = cells.summary
    diff = np.hypot(
        x - summary["cell_centroid_x"].to_numpy(np.float64)[position],
        y - summary["cell_centroid_y"].to_numpy(np.float64)[position],
    )
    details: dict[str, Any] = {
        "n_shapes": int(len(shapes)),
        "n_cells": cells.n_cells,
        "centroid_max_diff_um": float(np.nanmax(diff)) if len(diff) else 0.0,
        "centroid_median_diff_um": float(np.nanmedian(diff)) if len(diff) else 0.0,
        "over_tolerance": int(np.count_nonzero(~(diff <= SHAPE_CENTROID_TOLERANCE_UM))),
        "tolerance_um": SHAPE_CENTROID_TOLERANCE_UM,
        "cell_vertex_stats": ar.polygon_vertex_stats(shapes),
    }
    nucleus_key = atera_builder.NUCLEUS_SHAPES_KEY
    if nucleus_key in ctx.sdata.shapes:
        nuclei = ctx.sdata.shapes[nucleus_key]
        details["nucleus_vertex_stats"] = ar.polygon_vertex_stats(nuclei)
        # Informational: single-nucleus cells against the vendor nucleus centroid.
        owner = nuclei["cell_instance_id"].to_numpy(np.int64) - 1
        single = summary["nucleus_count"].to_numpy()[owner] == 1
        if single.any():
            ncent = shapely.centroid(nuclei.geometry.to_numpy()[single])
            ndiff = np.hypot(
                shapely.get_x(ncent)
                - summary["nucleus_centroid_x"].to_numpy(np.float64)[owner[single]],
                shapely.get_y(ncent)
                - summary["nucleus_centroid_y"].to_numpy(np.float64)[owner[single]],
            )
            details["single_nucleus_centroid_max_diff_um"] = float(np.nanmax(ndiff))
    failures = []
    if len(shapes) != cells.n_cells or len(np.unique(position)) != cells.n_cells:
        failures.append("not every cell has exactly one shape")
    index = _int_index(shapes)
    if index is None or not np.array_equal(index, position + 1):
        failures.append("cell shapes are not indexed by their label (row + 1)")
    if details["over_tolerance"]:
        failures.append("cell shape centroids differ from cell_summary")
    if nucleus_key in ctx.sdata.shapes:
        nuclei = ctx.sdata.shapes[nucleus_key]
        nucleus_index = _int_index(nuclei)
        owner_expected = cells.nucleus_cell_index.astype(np.int64) + 1
        details["n_nucleus_shapes"] = int(len(nuclei))
        if nucleus_index is None or not np.array_equal(
            np.sort(nucleus_index), np.arange(1, cells.n_nuclei + 1)
        ):
            failures.append("nucleus shapes are not one per nucleus row + 1")
        elif not np.array_equal(
            nuclei["cell_instance_id"].to_numpy(np.int64),
            owner_expected[nucleus_index - 1],
        ):
            failures.append("nucleus shapes name the wrong owning cell")
    if failures:
        raise CheckFailedError("; ".join(failures), details)
    return details


# --------------------------------------------------------------------------- #
# Criteria: pixel size, metadata, completion marker
# --------------------------------------------------------------------------- #


def _same_values(ours: Any, vendor: Any) -> bool:
    """Equal values, NaN-aware for floats; categories compared as text."""
    if isinstance(ours, pd.Series):
        ours = ours.to_numpy()
    if isinstance(vendor, pd.Series):
        vendor = vendor.to_numpy()
    a = np.asarray(ours.astype(object) if isinstance(ours, pd.Categorical) else ours)
    b = np.asarray(
        vendor.astype(object) if isinstance(vendor, pd.Categorical) else vendor
    )
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
        return bool(
            np.array_equal(a.astype(np.float64), b.astype(np.float64), equal_nan=True)
        )
    return bool(np.array_equal(a.astype(str), b.astype(str)))


def _read_elem(group: Any, name: str) -> Any:
    value = ad.io.read_elem(group[name])
    if isinstance(value, np.ndarray) and value.dtype.kind == "T":
        return value.astype(object)
    return value


def read_vendor_table_parts(bundle: ar.AteraBundle) -> dict[str, Any]:
    """The raw matrix pieces the table must keep, read straight from the zips."""
    parts: dict[str, Any] = {}
    with ar.open_zip_group(bundle.cell_feature_matrix_zip) as group:
        matrix = sparse.csr_matrix(ad.io.read_elem(group["X"]))
        obs = group["obs"]
        index_name = obs.attrs["_index"]
        parts["barcodes"] = np.asarray(_read_elem(obs, index_name), dtype=object)
        parts["obs"] = {
            name: _read_elem(obs, name)
            for name in obs.attrs.get("column-order", [])
            if name != index_name
        }
        parts["colors"] = {
            name: [
                f"#{int(value) & 0xFFFFFFFF:08x}"
                for value in np.asarray(obs[name]["color"][:]).astype(np.int64)
            ]
            for name in obs.attrs.get("column-order", [])
            if name in obs and hasattr(obs[name], "keys") and "color" in obs[name]
        }
        var = group["var"]
        parts["var"] = {
            name: _read_elem(var, name) for name in var.attrs.get("column-order", [])
        }
        parts["obsm"] = (
            {
                name: np.asarray(_read_elem(group["obsm"], name))
                for name in group["obsm"]
            }
            if "obsm" in group
            else {}
        )
    matrix.data = matrix.data.astype(np.int64, copy=False)
    parts["X"] = matrix
    with ar.open_zip_group(bundle.cells_zip) as group:
        summary = group["cell_summary"]
        parts["summary"] = pd.DataFrame(
            np.asarray(summary[:]), columns=list(summary.attrs["column_names"])
        )
    return parts


def check_table_vs_vendor(ctx: Context) -> dict[str, Any]:
    """The table keeps the vendor matrix, ``obs``, ``obsm``, DE and colours."""
    table = ctx.sdata.tables["table"]
    raw = read_vendor_table_parts(ctx.bundle)
    failures: list[str] = []
    n_vendor = len(raw["barcodes"])
    position = pd.Index(raw["barcodes"]).get_indexer(
        table.obs[SOURCE_CELL_ID_COLUMN].astype(str)
    )
    instance_ids = table.obs[INSTANCE_ID_COLUMN].to_numpy(np.int64)
    details: dict[str, Any] = {"n_cells": int(table.n_obs), "n_vendor_cells": n_vendor}
    if table.n_obs != n_vendor or (position < 0).any():
        raise CheckFailedError(
            "table cells differ from the vendor matrix rows", details
        )
    if not np.array_equal(instance_ids, position + 1):
        failures.append("table instance_id is not the vendor matrix row + 1")
    if not np.array_equal(
        table.obs_names.to_numpy().astype(str), instance_ids.astype(str)
    ):
        failures.append("table obs_names are not the instance_id text")

    # X: the vendor gene columns, row block by row block.
    feature_type = np.asarray(raw["var"]["feature_type"]).astype(str)
    feature_name = np.asarray(raw["var"]["feature_name"]).astype(str)
    genes = np.flatnonzero(feature_type == ar.GENE_EXPRESSION_TYPE)
    if not np.array_equal(table.var_names.to_numpy().astype(str), feature_name[genes]):
        failures.append("table var_names are not the vendor gene features")
    ours = sparse.csr_matrix(table.X)
    vendor = raw.pop("X")
    differing = 0
    vendor_total = table_total = 0
    for start in range(0, n_vendor, TABLE_ROW_BLOCK):
        rows = position[start : start + TABLE_ROW_BLOCK]
        expected = vendor[rows][:, genes]
        got = ours[start : start + TABLE_ROW_BLOCK]
        if not np.array_equal(got.data, np.round(got.data)):
            failures.append("table X holds non-integer counts")
            break
        diff = expected - got.astype(np.int64)
        diff.eliminate_zeros()
        differing += int(diff.nnz)
        vendor_total += int(expected.sum())
        table_total += int(got.astype(np.int64).sum())
    del vendor, ours
    details.update(
        x_differing_entries=differing,
        x_vendor_gene_total=vendor_total,
        x_table_total=table_total,
    )
    if differing:
        failures.append("table X differs from the vendor gene columns")

    # Vendor obs columns, as vendor_<name>.
    obs_checked = []
    for name, values in raw["obs"].items():
        if name in VENDOR_OBS_NOT_COPIED:
            continue
        column = f"vendor_{name}"
        if column not in table.obs:
            failures.append(f"vendor obs column {name} is not kept")
            continue
        vendor_values = (
            values[position]
            if not isinstance(values, pd.Categorical)
            else values.take(position)
        )
        if not _same_values(table.obs[column], vendor_values):
            failures.append(f"vendor obs column {name} differs")
        obs_checked.append(name)
    details["vendor_obs_columns_checked"] = obs_checked

    # cell_summary: named columns, centroids in obsm["spatial"], others vendor_<name>.
    summary = raw["summary"].iloc[position].reset_index(drop=True)
    summary_checked = []
    for name in summary.columns:
        if name in ("cell_centroid_x", "cell_centroid_y"):
            continue
        column = SUMMARY_NAMED_COLUMNS.get(name, f"vendor_{name}")
        if column not in table.obs:
            failures.append(f"cell_summary column {name} is not kept")
        elif not _same_values(table.obs[column], summary[name]):
            failures.append(f"cell_summary column {name} differs from obs {column}")
        else:
            summary_checked.append(name)
    if not _same_values(
        np.asarray(table.obsm["spatial"]),
        summary[["cell_centroid_x", "cell_centroid_y"]].to_numpy(),
    ):
        failures.append("obsm spatial differs from the cell_summary centroids")
    details["cell_summary_columns_checked"] = summary_checked

    # obsm, varm DE statistics and category colours.
    for name, values in raw["obsm"].items():
        key = f"{name}_vendor"
        if key not in table.obsm:
            failures.append(f"vendor obsm {name} is not kept")
        elif not _same_values(np.asarray(table.obsm[key]), values[position]):
            failures.append(f"vendor obsm {name} differs")
    de_names = [name for name in raw["var"] if name.startswith("de_")]
    if de_names:
        de = table.varm.get("vendor_de")
        if de is None or sorted(map(str, de.columns)) != sorted(de_names):
            failures.append("varm vendor_de does not hold every de_* column")
        else:
            bad = [
                name
                for name in de_names
                if not _same_values(de[name], np.asarray(raw["var"][name])[genes])
            ]
            if bad:
                failures.append(f"varm vendor_de columns differ: {bad[:MAX_LISTED]}")
    for name, colors in raw["colors"].items():
        kept = table.uns.get(f"vendor_{name}_colors")
        if kept is None or [str(value) for value in kept] != colors:
            failures.append(f"vendor colours of {name} differ")
    details.update(
        n_de_columns=len(de_names),
        obsm_checked=sorted(raw["obsm"]),
        colors_checked=sorted(raw["colors"]),
        failures=failures,
    )
    if failures:
        raise CheckFailedError("the table does not keep the vendor data", details)
    return details


def check_um_per_px(ctx: Context) -> dict[str, Any]:
    """``find_um_per_px(store, "ATERA")`` returns the bundle pixel size."""
    found = float(find_um_per_px(ctx.store, "ATERA"))
    expected = float(ctx.bundle.pixel_size_um)
    details = {"um_per_px": found, "expected": expected}
    if not math.isclose(found, expected, rel_tol=1e-9):
        raise CheckFailedError(
            "find_um_per_px differs from the bundle pixel size", details
        )
    return details


def zip_metadata_members(zip_path: Path) -> dict[str, str]:
    """Every zarr metadata member of a zip as text, listed from its name list."""
    with zipfile.ZipFile(zip_path) as archive:
        return {
            name: archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.rsplit("/", 1)[-1] in ZARR_METADATA_NAMES
        }


def zip_root_attrs(members: Mapping[str, str]) -> dict[str, Any]:
    """Root attributes of a zipped zarr store (v2 ``.zattrs`` or v3 ``zarr.json``)."""
    if ".zattrs" in members:
        attrs = json.loads(members[".zattrs"])
    elif "zarr.json" in members:
        attrs = json.loads(members["zarr.json"]).get("attributes") or {}
    else:
        attrs = {}
    return attrs if isinstance(attrs, dict) else {}


def metric_matches(stored: Any, text: str) -> bool:
    """Whether a stored metrics value is the CSV cell ``text``.

    Empty cells are ``None``; numbers must equal the number the text spells;
    anything else must be the text itself.
    """
    value = text.strip()
    if not value:
        return stored is None
    if isinstance(stored, bool) or stored is None:
        return False
    if isinstance(stored, int):
        try:
            return int(value) == stored
        except ValueError:
            return False
    if isinstance(stored, float):
        try:
            return float(value) == stored
        except ValueError:
            return False
    return bool(stored == value)


def compare_metrics(stored: Any, csv_path: Path) -> list[str]:
    """Differences between ``attrs`` metrics rows and the raw CSV text."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle)]
    header = [name.strip() for name in rows[0]] if rows else []
    data = [row for row in rows[1:] if any(cell.strip() for cell in row)]
    if not isinstance(stored, list) or len(stored) != len(data):
        return ["metrics rows differ in number from the CSV"]
    problems = []
    for index, (record, row) in enumerate(zip(stored, data, strict=True)):
        if list(record) != header[: len(row)]:
            problems.append(f"metrics row {index} columns differ from the CSV header")
            continue
        bad = [
            name
            for name, text in zip(header, row, strict=False)
            if not metric_matches(record[name], text)
        ]
        if bad:
            problems.append(f"metrics row {index} values differ: {bad[:MAX_LISTED]}")
    return problems


def panel_header(panel: Mapping[str, Any]) -> dict[str, Any]:
    """``panel_config.json`` with each ``targets`` list replaced by its length."""
    header: dict[str, Any] = json.loads(json.dumps(panel))
    for payload in header.get("payloads") or []:
        if isinstance(payload, dict) and isinstance(payload.get("targets"), list):
            payload["num_targets"] = len(payload.pop("targets"))
    return header


def check_metadata(ctx: Context) -> dict[str, Any]:
    """Sidecars byte-identical; the full header and embedded metadata kept."""
    header = ctx.header
    bundle = ctx.bundle
    failures: list[str] = []
    records = {record["original_name"]: record for record in header.get("sidecars", [])}
    for sidecar in bundle.sidecars:
        record = records.get(sidecar.original_name)
        if record is None:
            failures.append(f"sidecar {sidecar.stored_path} is not recorded")
            continue
        stored = ctx.store / record["path"]
        if not stored.is_file() or stored.read_bytes() != sidecar.source.read_bytes():
            failures.append(f"sidecar {sidecar.stored_path} differs from the bundle")
        elif record.get("sha256") != ar._file_sha256(sidecar.source):
            failures.append(f"sidecar {sidecar.stored_path} SHA-256 differs")
        elif record.get("size_bytes") != sidecar.source.stat().st_size:
            failures.append(f"sidecar {sidecar.stored_path} size differs")
    if len(records) != len(bundle.sidecars):
        failures.append("the store records sidecars the bundle does not have")

    if header.get("experiment") != json.loads(json.dumps(bundle.experiment)):
        failures.append("attrs experiment differs from experiment.spatial")
    stored_paths = {sidecar.stored_path: sidecar.source for sidecar in bundle.sidecars}
    metrics = stored_paths.get("metrics_summary.csv")
    n_metrics_columns = 0
    if metrics is not None:
        problems = compare_metrics(header.get("metrics_summary"), metrics)
        failures.extend(f"attrs metrics_summary: {item}" for item in problems)
        with metrics.open(newline="", encoding="utf-8") as handle:
            n_metrics_columns = len(next(csv.reader(handle), []))
    elif header.get("metrics_summary"):
        failures.append("attrs metrics_summary without a metrics CSV")
    panel = stored_paths.get("panel_config.json")
    if panel is not None and header.get("panel_config") != panel_header(
        json.loads(panel.read_text(encoding="utf-8"))
    ):
        failures.append("attrs panel_config differs from panel_config.json")
    if header.get("morphology_2d_annotations") != json.loads(
        json.dumps(list(bundle.morphology_2d_annotations))
    ):
        failures.append("attrs morphology_2d_annotations differ from the OME-XML")

    embedded = {
        (record["kind"], record["original_name"]): record
        for record in header.get("embedded_metadata", [])
    }
    n_zarr_members = 0
    for store_name, zip_path in ar.vendor_zarr_stores(bundle).items():
        original = ar.bundle_relative_name(zip_path, bundle.root)
        record = embedded.get(("zarr_metadata", original))
        if record is None:
            failures.append(f"zarr metadata of {store_name} is not recorded")
            continue
        kept = json.loads((ctx.store / record["path"]).read_text(encoding="utf-8"))
        members = zip_metadata_members(zip_path)
        n_zarr_members += len(members)
        if kept.get("members") != members:
            failures.append(f"zarr metadata of {store_name} differs from the zip")
        root_attrs = zip_root_attrs(members)
        vendor = header.get("vendor_store_attrs", {}).get(store_name, {})
        inline = vendor.get("attrs", {})
        if set(inline) | set(vendor.get("attrs_in_sidecar_only", [])) != set(
            root_attrs
        ):
            failures.append(f"root attrs of {store_name} are not all kept")
        elif any(inline[key] != root_attrs[key] for key in inline):
            failures.append(f"root attrs of {store_name} differ from the zip")
    images = [*bundle.morphology_2d_files]
    if bundle.morphology_3d_file is not None:
        images.append(bundle.morphology_3d_file)
    for path in images:
        original = ar.bundle_relative_name(path, bundle.root)
        with tifffile.TiffFile(path, _multifile=False) as tif:
            xml = tif.ome_metadata or ""
        record = embedded.get(("ome_xml", original))
        if record is None:
            failures.append(f"OME-XML of {path.name} is not recorded")
        elif (ctx.store / record["path"]).read_text(encoding="utf-8") != xml:
            failures.append(f"OME-XML of {path.name} differs from the TIFF")
    for record in header.get("embedded_metadata", []):
        if record.get("sha256") != ar._file_sha256(ctx.store / record["path"]):
            failures.append(f"{record['path']} SHA-256 differs from its record")
    details = {
        "n_sidecars": len(bundle.sidecars),
        "n_zarr_metadata_members": n_zarr_members,
        "n_ome_xml": len(images),
        "n_experiment_fields": len(bundle.experiment),
        "n_metrics_columns": n_metrics_columns,
        "failures": failures,
    }
    if failures:
        raise CheckFailedError("vendor metadata is incomplete", details)
    return details


def check_build_marker(ctx: Context, atera_spec: Path | None = None) -> dict[str, Any]:
    """Completion marker present; fingerprints agree with a recomputed one."""
    marker = atera_builder.read_build_marker(ctx.store)
    if marker is None:
        raise CheckFailedError("the store has no completion marker")
    recorded = marker.get("build_fingerprint")
    in_attrs = ctx.header.get("build", {}).get("build_fingerprint")
    bundle = atera_builder.resolve_atera_input(
        ctx.bundle.root, ctx.build_config, atera_spec_path=atera_spec
    )
    recomputed = atera_builder.atera_build_fingerprint(bundle, ctx.build_config)
    details = {
        "marker_fingerprint": recorded,
        "attrs_fingerprint": in_attrs,
        "recomputed_fingerprint": recomputed,
        "format_version": marker.get("format_version"),
        "build_elapsed_s": marker.get("elapsed_s"),
        "build_peak_anonymous_rss_gb": marker.get("peak_anonymous_rss_gb"),
    }
    if not (recorded == in_attrs == recomputed):
        raise CheckFailedError("the build fingerprints disagree", details)
    partials = sorted(ctx.store.parent.glob(f"{ctx.store.name}.partial-*"))
    details["leftover_partials"] = len(partials)
    if partials:
        raise CheckFailedError(
            "partial build folders are left beside the store", details
        )
    return details


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def element_sizes(store: Path) -> dict[str, int]:
    """On-disk bytes of every element folder and of the ``atera/`` sidecars."""
    sizes: dict[str, int] = {}
    for kind in ("images", "labels", "points", "shapes", "tables"):
        folder = store / kind
        if not folder.is_dir():
            continue
        for element in sorted(path for path in folder.iterdir() if path.is_dir()):
            sizes[f"{kind}/{element.name}"] = _folder_bytes(element)
    sidecars = store / ar.ATERA_SIDECAR_DIR
    if sidecars.is_dir():
        sizes[ar.ATERA_SIDECAR_DIR] = _folder_bytes(sidecars)
    sizes["total"] = _folder_bytes(store)
    return sizes


def _folder_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def code_revision() -> dict[str, Any]:
    """Git revision of this script's checkout, and whether it had local changes."""
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(here), *args],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def build_config_from_header(
    header: Mapping[str, Any], overrides: Mapping[str, Any] | None = None
) -> AteraBuildConfig:
    """Rebuild the store's ``AteraBuildConfig`` from its recorded options."""
    options = dict(header.get("build", {}).get("options") or {})
    options.update(overrides or {})
    return AteraBuildConfig(**options)


def run_acceptance(
    store: Path,
    bundle_root: Path,
    *,
    alias: str,
    n_workers: int = 8,
    expected_vendor_excess: int | None = None,
    allow_unpinned_excess: bool = False,
    expected_options: ExpectedOptions | None = None,
    atera_spec: Path | None = None,
    input_overrides: Mapping[str, Any] | None = None,
    criteria: Sequence[str] = CRITERIA,
    merge_rows: int = DEFAULT_MERGE_ROWS,
) -> dict[str, Any]:
    """Run the acceptance criteria and return the report.

    Args:
        store: The built store.
        bundle_root: The raw bundle (extracted ROI folder) it was built from.
        alias: Dataset alias for the report.
        n_workers: Threads for the streamed reads.
        expected_vendor_excess: Expected vendor matrix count on columns with no
            transcripts (84 on the public dataset).
        allow_unpinned_excess: Pass ``points_vs_matrix`` without
            ``expected_vendor_excess`` (any deprecated-codeword excess).
        expected_options: The build options the store must record (the §1.1
            ones by default).
        atera_spec: The ``experiment.spatial`` override the build used, if any.
        input_overrides: Explicit input paths the build used
            (``transcripts_path`` etc.), as ``AteraBuildConfig`` fields.
        criteria: Criteria to run, a subset of :data:`CRITERIA`.
        merge_rows: Points pending before a count merge.

    Returns:
        The JSON-serialisable report.
    """
    started = time.monotonic()
    store = Path(store)
    root_attrs: dict[str, Any] = dict(zarr.open_group(store, mode="r").attrs)
    header: dict[str, Any] = dict(root_attrs.get(atera_builder.ATERA_ATTRS_KEY) or {})
    build_config = build_config_from_header(header, input_overrides)
    bundle = atera_builder.resolve_atera_input(
        Path(bundle_root), build_config, atera_spec_path=atera_spec
    )
    ctx = Context(
        store=store,
        bundle=bundle,
        header=header,
        build_config=build_config,
        n_workers=n_workers,
        merge_rows=merge_rows,
    )
    cache: dict[str, Any] = {}
    options = expected_options or ExpectedOptions()
    checks: dict[str, Callable[[], dict[str, Any]]] = {
        "build_options": lambda: check_build_options(ctx, options),
        "reopen": lambda: check_reopen(ctx),
        "schema": lambda: check_schema(ctx),
        "points_vs_matrix": lambda: check_points_vs_matrix(
            ctx, expected_vendor_excess, allow_unpinned_excess=allow_unpinned_excess
        ),
        "labels_vs_cell_summary": lambda: check_labels(ctx, cache),
        "nucleus_containment": lambda: check_nucleus_containment(ctx, cache),
        "shape_centroids": lambda: check_shape_centroids(ctx),
        "um_per_px": lambda: check_um_per_px(ctx),
        "table_vs_vendor": lambda: check_table_vs_vendor(ctx),
        "metadata": lambda: check_metadata(ctx),
        "build_marker": lambda: check_build_marker(ctx, atera_spec),
    }
    unknown = sorted(set(criteria) - set(checks))
    if unknown:
        raise ValueError(f"unknown criteria: {unknown}")
    results: dict[str, Any] = {}
    for name in CRITERIA:
        if name not in criteria:
            continue
        logger.info("[%s] checking %s", alias, name)
        t0 = time.monotonic()
        entry: dict[str, Any]
        try:
            entry = {"passed": True, "details": checks[name]()}
        except CheckFailedError as exc:
            entry = {"passed": False, "error": str(exc), "details": exc.details}
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            entry = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }
        entry["seconds"] = round(time.monotonic() - t0, 2)
        entry["peak_rss_gb_after"] = round(peak_rss_gb(), 3)
        results[name] = entry
        logger.info(
            "[%s] %s: %s (%.1f s, peak RSS %.2f GB)",
            alias,
            name,
            "PASS" if entry["passed"] else "FAIL",
            entry["seconds"],
            entry["peak_rss_gb_after"],
        )
    return {
        "alias": alias,
        "store_name": store.name,
        "code_revision": code_revision(),
        "passed": all(entry["passed"] for entry in results.values()),
        "criteria": results,
        "build_options": atera_builder.build_options(build_config),
        "pixel_size_um": bundle.pixel_size_um,
        "shape_stats": header.get("shapes"),
        "element_sizes_bytes": element_sizes(store),
        "n_workers": n_workers,
        "wall_seconds": round(time.monotonic() - started, 2),
        "peak_rss_gb": round(peak_rss_gb(), 3),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_report(report: Mapping[str, Any], path: Path) -> None:
    """Write the report as indented JSON (non-finite floats become null)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        atera_builder._json_safe(json.loads(json.dumps(report, default=_json_default))),
        indent=1,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--alias", required=True, help="Dataset alias for the report")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--expected-vendor-excess",
        type=int,
        default=None,
        help="Vendor matrix counts on columns without transcripts (84 on the "
        "public dataset); required by points_vs_matrix",
    )
    parser.add_argument(
        "--allow-unpinned-excess",
        action="store_true",
        help="Accept any deprecated-codeword excess without --expected-vendor-excess",
    )
    parser.add_argument(
        "--expect-min-qv",
        type=float,
        default=EXPECTED_MIN_QV,
        help="min_qv the store must record (plan §1.1: 20)",
    )
    parser.add_argument(
        "--expect-boundary-simplify-px",
        type=float,
        default=EXPECTED_BOUNDARY_SIMPLIFY_PX,
        help="boundary_simplify_px the store must record (plan §1.1: 0.25)",
    )
    parser.add_argument(
        "--allow-tile-allowlist",
        action="store_true",
        help="Accept a store built from a tile allow-list",
    )
    parser.add_argument("--atera-spec", type=Path, default=None)
    parser.add_argument("--transcripts-path", type=Path, default=None)
    parser.add_argument("--cells-path", type=Path, default=None)
    parser.add_argument("--cell-feature-matrix-path", type=Path, default=None)
    parser.add_argument(
        "--criteria", nargs="+", default=list(CRITERIA), choices=list(CRITERIA)
    )
    args = parser.parse_args(argv)
    if (
        "points_vs_matrix" in args.criteria
        and args.expected_vendor_excess is None
        and not args.allow_unpinned_excess
    ):
        parser.error(
            "points_vs_matrix needs --expected-vendor-excess "
            "(or --allow-unpinned-excess)"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    overrides = {
        key: value
        for key, value in {
            "transcripts_path": args.transcripts_path,
            "cells_path": args.cells_path,
            "cell_feature_matrix_path": args.cell_feature_matrix_path,
        }.items()
        if value is not None
    }
    report = run_acceptance(
        args.store,
        args.bundle,
        alias=args.alias,
        n_workers=args.workers,
        expected_vendor_excess=args.expected_vendor_excess,
        allow_unpinned_excess=args.allow_unpinned_excess,
        expected_options=ExpectedOptions(
            min_qv=args.expect_min_qv,
            boundary_simplify_px=args.expect_boundary_simplify_px,
            allow_tile_allowlist=args.allow_tile_allowlist,
        ),
        atera_spec=args.atera_spec,
        input_overrides=overrides,
        criteria=args.criteria,
    )
    write_report(report, args.report)
    logger.info(
        "[%s] acceptance %s in %.0f s (peak RSS %.2f GB)",
        args.alias,
        "PASSED" if report["passed"] else "FAILED",
        report["wall_seconds"],
        report["peak_rss_gb"],
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
