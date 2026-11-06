"""ATERA_PROSEG_STITCH: write the stitched ProSeg layer into the latest store.

Inputs are the tile plan, every tile result the plan runs and the persistent
latest store of the row (built by NATIVE_LATEST from the same source build).
The stitch writes, in place under the store's write lock:

- ``shapes/MOSAIK_proseg``: every owned ProSeg polygon, unclipped, in microns,
  with the vendor shapes' ``Scale(1 / ps)`` transform to ``global``;
- ``labels/proseg_cell_labels``: the polygons rasterised on the morphology
  grid (pixel-centre rule, ascending ``instance_id``, last wins), with the
  vendor labels' pyramid and ``Identity`` transform;
- ``tables/table_MOSAIK_proseg``: cells x genes foreground counts (genes only,
  ``var`` copied from ``table_original``), the ProSeg cell metadata and the
  per-cell control counts in ``obs``;
- ``points/transcripts``: every points file rewritten with two appended
  columns, ``assignment`` (nullable uint64, the vendor label of the ProSeg
  cell) and ``background``; file names, row counts and row order are kept, so
  the build's points index and aggregates still describe the files;
- the registry branch ``proseg`` (legacy alias ``reseg``) and the provenance
  ``attrs["merxen_atera_proseg"]``.

A transcript claimed by one tile gets that tile's assignment; claimed by two
tiles for different cells it goes to the tile whose core holds it, else to the
first tile in plan order (conflicts are counted). Unclaimed rows (vendor and
ProSeg background, rows ProSeg dropped, controls) have no assignment and are
background. Control rows are attributed to cells geometrically: their pixel's
value in ``proseg_cell_labels``.

The stitch records a fingerprint of the plan and the tile manifests; a store
that already carries it (and every element) is left as it is. A new stitch
deletes the tables derived from the previous ProSeg cells (every table of
region ``MOSAIK_proseg`` other than ``table_MOSAIK_proseg``: the image
quantification, clustering), so no later stage reuses them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial as partial_function
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from merxen.config import AteraProsegStitchConfig
from merxen.io.points_aggregates import points_parquet_dir, read_root_attrs
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    ORIGINAL_ID_NAMESPACE,
    register_segmentation_branch,
)
from merxen.path_utils import remove_path, stage_existing_output
from merxen.segmentation.atera_proseg import (
    ASSIGNMENT_COLUMN,
    BACKGROUND_COLUMN,
    PRIOR_NAME,
    PROSEG_BRANCH,
    PROSEG_LABEL_KEY,
    PROSEG_SHAPES_KEY,
    PROSEG_TABLE_KEY,
    PROVENANCE_ATTR,
    STITCH_FORMAT_VERSION,
    TILE_CELLS_NAME,
    TILE_CLAIMS_DIR,
    AteraProsegError,
)
from merxen.segmentation.atera_proseg.plan import _core_index, read_plan
from merxen.segmentation.atera_proseg.tile import read_tile_manifest

logger = logging.getLogger(__name__)

POINTS_KEY = "transcripts"
ORIGINAL_TABLE_KEY = "table_original"
NATIVE_MARKER_NAME = "merxen_native_latest.json"
LABEL_BLOCK_PX = 4096
CONTROL_COUNT_PREFIXES: dict[str, tuple[str, ...]] = {
    "genomic_control_counts": ("Intergenic_Region",),
    "unassigned_codeword_counts": ("UnassignedCodeword", "BLANK_"),
    "deprecated_codeword_counts": ("DeprecatedCodeword",),
    "control_codeword_counts": ("NegControlCodeword",),
    "control_probe_counts": ("NegControlProbe", "antisense_"),
}
"""``obs`` control count columns and the feature-name prefixes they count.

The first three mirror the vendor's derived counts of ``table_original``; the
negative controls mirror its ``vendor_control_*_counts``. Controls of no known
prefix are counted in ``other_control_counts``.
"""
OTHER_CONTROL_COLUMN = "other_control_counts"
CONTROL_ASSIGNMENT_RULE = "proseg_label_raster_corner"


# --------------------------------------------------------------------------- #
# Claim resolution (pure; unit-tested)
# --------------------------------------------------------------------------- #


@dataclass
class ClaimSet:
    """The claims one tile makes on the rows of one points file.

    Attributes:
        tile_id: The tile.
        plan_index: Position of the tile in the plan (lower wins a tie).
        grid: ``(column, row)`` of the tile's core.
        rows: Row positions in the points file.
        instance_id: Claimed cell of each row.
        background: ProSeg's background flag of each row.
    """

    tile_id: str
    plan_index: int
    grid: tuple[int, int]
    rows: np.ndarray
    instance_id: np.ndarray
    background: np.ndarray


@dataclass
class Resolution:
    """Final assignment of the rows of one points file.

    Attributes:
        assignment: Vendor label of the assigned cell, 0 when unassigned.
        background: Background flag (True when unclaimed).
        claimed: Whether any tile claimed the row.
        conflicts: Rows two tiles claimed differently, by ``"A|B"`` pair
            (winner first).
        losing_cells: Rows a cell lost in a conflict, by ``instance_id``.
    """

    assignment: np.ndarray
    background: np.ndarray
    claimed: np.ndarray
    conflicts: Counter[str] = field(default_factory=Counter)
    losing_cells: Counter[int] = field(default_factory=Counter)


def resolve_claims(
    n_rows: int,
    x: np.ndarray,
    y: np.ndarray,
    tile_um: float,
    claim_sets: list[ClaimSet],
) -> Resolution:
    """Resolve the claims of every tile on one file's rows.

    Args:
        n_rows: Rows of the file.
        x: X microns of every row.
        y: Y microns of every row.
        tile_um: Core edge (to find the core holding each row).
        claim_sets: The tiles' claims.

    Returns:
        The resolution.
    """
    assignment = np.zeros(n_rows, dtype=np.uint64)
    background = np.ones(n_rows, dtype=bool)
    owner = np.full(n_rows, -1, dtype=np.int64)
    owner_grid = np.zeros((n_rows, 2), dtype=np.int64)
    result = Resolution(
        assignment=assignment, background=background, claimed=owner >= 0
    )
    if not claim_sets:
        return result
    core_col = _core_index(np.asarray(x, dtype=np.float64), tile_um)
    core_row = _core_index(np.asarray(y, dtype=np.float64), tile_um)
    by_index = {claims.plan_index: claims.tile_id for claims in claim_sets}
    for claims in sorted(claim_sets, key=lambda item: item.plan_index):
        rows = np.asarray(claims.rows, dtype=np.int64)
        inst = np.asarray(claims.instance_id, dtype=np.uint64)
        flag = np.asarray(claims.background, dtype=bool)
        first = owner[rows] < 0
        set_rows = rows[first]
        assignment[set_rows] = inst[first]
        background[set_rows] = flag[first]
        owner[set_rows] = claims.plan_index
        owner_grid[set_rows] = claims.grid
        again = ~first
        if not again.any():
            continue
        rows_again = rows[again]
        differs = (assignment[rows_again] != inst[again]) | (
            background[rows_again] != flag[again]
        )
        if not differs.any():
            continue
        rows_conflict = rows_again[differs]
        inst_conflict = inst[again][differs]
        flag_conflict = flag[again][differs]
        in_new_core = (core_col[rows_conflict] == claims.grid[0]) & (
            core_row[rows_conflict] == claims.grid[1]
        )
        in_old_core = (core_col[rows_conflict] == owner_grid[rows_conflict, 0]) & (
            core_row[rows_conflict] == owner_grid[rows_conflict, 1]
        )
        # The core tile wins; otherwise the earlier tile (already set) stays.
        new_wins = in_new_core & ~in_old_core
        cells_differ = assignment[rows_conflict] != inst_conflict
        for old_index, wins, cell_old, cell_new, differ in zip(
            owner[rows_conflict],
            new_wins,
            assignment[rows_conflict],
            inst_conflict,
            cells_differ,
            strict=True,
        ):
            winner, loser = (
                (claims.tile_id, by_index[int(old_index)])
                if wins
                else (by_index[int(old_index)], claims.tile_id)
            )
            result.conflicts[f"{winner}|{loser}"] += 1
            if differ:
                result.losing_cells[int(cell_old if wins else cell_new)] += 1
        win_rows = rows_conflict[new_wins]
        assignment[win_rows] = inst_conflict[new_wins]
        background[win_rows] = flag_conflict[new_wins]
        owner[win_rows] = claims.plan_index
        owner_grid[win_rows] = claims.grid
    result.claimed = owner >= 0
    return result


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass
class _Tiles:
    plan: dict[str, Any]
    manifests: dict[str, dict[str, Any]]
    dirs: dict[str, Path]
    order: list[str]
    grid: dict[str, tuple[int, int]]


def _load_tiles(cfg: AteraProsegStitchConfig) -> _Tiles:
    plan = read_plan(cfg.plan_path)
    if plan.get("tile_subset") is not None:
        raise AteraProsegError(
            f"[{cfg.dataset_name}] The plan runs a tile subset "
            f"({', '.join(plan['tile_subset'])}); a subset is a check run and "
            "cannot be stitched"
        )
    expected = [tile["tile_id"] for tile in plan["tiles"] if tile["run"]]
    manifests: dict[str, dict[str, Any]] = {}
    dirs: dict[str, Path] = {}
    for tile_dir in cfg.tile_dirs:
        manifest = read_tile_manifest(Path(tile_dir))
        tid = str(manifest["tile_id"])
        if tid in manifests:
            raise AteraProsegError(f"[{cfg.dataset_name}] Tile {tid} delivered twice")
        manifests[tid] = manifest
        dirs[tid] = Path(tile_dir)
    missing = sorted(set(expected) - set(manifests))
    extra = sorted(set(manifests) - set(expected))
    if missing or extra:
        raise AteraProsegError(
            f"[{cfg.dataset_name}] The delivered tiles do not match the plan: "
            f"missing {missing or 'none'}, not in the plan {extra or 'none'}"
        )
    for tid, manifest in manifests.items():
        if manifest["source_build_fingerprint"] != plan["source_build_fingerprint"]:
            raise AteraProsegError(
                f"[{cfg.dataset_name}] Tile {tid} was run on another source build"
            )
    grid = {
        str(tile["tile_id"]): (int(tile["grid"][0]), int(tile["grid"][1]))
        for tile in plan["tiles"]
    }
    return _Tiles(plan, manifests, dirs, expected, grid)


def stitch_fingerprint(tiles: _Tiles) -> str:
    """Return the SHA-256 of the plan, the tile manifests and the stitch version."""
    payload = {
        "stitch_format_version": STITCH_FORMAT_VERSION,
        "plan": tiles.plan,
        "tiles": [tiles.manifests[tid] for tid in sorted(tiles.manifests)],
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_latest_store(latest: Path, plan: dict[str, Any], dataset: str) -> None:
    try:
        marker = json.loads((latest / NATIVE_MARKER_NAME).read_text("utf-8"))
    except (OSError, ValueError) as error:
        raise AteraProsegError(
            f"[{dataset}] The latest store is not a NATIVE_LATEST store "
            f"({NATIVE_MARKER_NAME} unreadable)"
        ) from error
    if marker.get("source_build_fingerprint") != plan["source_build_fingerprint"]:
        raise AteraProsegError(
            f"[{dataset}] The latest store was built from another source build "
            "than the tile plan; re-run NATIVE_LATEST and the tiles"
        )


def _load_cells(tiles: _Tiles) -> pd.DataFrame:
    frames = []
    for tid in tiles.order:
        frame = pq.read_table(tiles.dirs[tid] / TILE_CELLS_NAME).to_pandas()
        frame["proseg_tile"] = tid
        frames.append(frame)
    cells = pd.concat(frames, ignore_index=True)
    if cells["instance_id"].duplicated().any():
        raise AteraProsegError("Two tiles own the same vendor label")
    return cells.sort_values("instance_id", kind="stable").reset_index(drop=True)


def _claim_files(tiles: _Tiles) -> dict[str, list[tuple[str, Path]]]:
    """Return ``points file name -> [(tile id, claims file)]``."""
    found: dict[str, list[tuple[str, Path]]] = {}
    for tid in tiles.order:
        for name in tiles.manifests[tid]["rows"]["claims_per_file"]:
            path = tiles.dirs[tid] / TILE_CLAIMS_DIR / f"{Path(name).stem}.parquet"
            if not path.is_file():
                raise AteraProsegError(f"Tile {tid} lacks its claims for {name}")
            found.setdefault(name, []).append((tid, path))
    return found


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


@dataclass
class RasterOverlap:
    """Pixels claimed by two different polygons while rasterising.

    Attributes:
        total: Every such pixel.
        cross_group: Those whose two polygons belong to different groups
            (cells of two tiles: an overlap across a tile seam).
        pixel_counts: Pixels each geometry holds once every polygon is drawn
            (input order). Zero for a polygon that holds no pixel centre or
            whose pixels later polygons took: such a cell has no pixel in the
            label raster, so mask-based stages (image quantification) cannot
            see it.
    """

    total: int = 0
    cross_group: int = 0
    pixel_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def within_group(self) -> int:
        """Overlap pixels of two polygons of one group (one tile)."""
        return self.total - self.cross_group


def rasterise_cells(
    geometries: list[Any],
    instance_ids: np.ndarray,
    shape: tuple[int, int],
    pixel_size_um: float,
    out: np.ndarray,
    groups: np.ndarray | None = None,
) -> RasterOverlap:
    """Rasterise polygons (microns) into ``out`` by the pixel-centre rule.

    Pixel ``(i, j)`` takes the label of the polygon containing its centre
    ``((j + 0.5) ps, (i + 0.5) ps)``; polygons are drawn in the given order
    (last wins) and holes are left to other cells.

    Args:
        geometries: Shapely polygons or multipolygons, in microns.
        instance_ids: Label of each geometry (unique).
        shape: ``(n_rows, n_cols)`` of the grid.
        pixel_size_um: Microns per pixel.
        out: uint32 array of ``shape`` (zeros), written in place.
        groups: Integer group of each geometry (its tile); ``None`` puts every
            geometry in one group.

    Returns:
        The overlap pixels, in total and across groups.
    """
    from skimage.draw import polygon as draw_polygon

    n_rows, n_cols = shape
    ps = float(pixel_size_um)
    overlap = RasterOverlap()
    ids = np.asarray(instance_ids, dtype=np.uint64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    group_of = (
        np.zeros(ids.size, dtype=np.int64)
        if groups is None
        else np.asarray(groups, dtype=np.int64)
    )
    sorted_groups = group_of[order]
    sorted_counts = np.zeros(ids.size, dtype=np.int64)
    for geometry, label, group in zip(geometries, ids, group_of, strict=True):
        if geometry is None or geometry.is_empty:
            continue
        x0, y0, x1, y1 = geometry.bounds
        r0 = max(int(np.floor(y0 / ps - 0.5)), 0)
        c0 = max(int(np.floor(x0 / ps - 0.5)), 0)
        r1 = min(int(np.ceil(y1 / ps - 0.5)) + 1, n_rows)
        c1 = min(int(np.ceil(x1 / ps - 0.5)) + 1, n_cols)
        if r1 <= r0 or c1 <= c0:
            continue
        local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        parts = geometry.geoms if geometry.geom_type == "MultiPolygon" else (geometry,)
        for part in parts:
            for ring, value in (
                (part.exterior, True),
                *((interior, False) for interior in part.interiors),
            ):
                coords = np.asarray(ring.coords, dtype=np.float64)
                rows = coords[:, 1] / ps - 0.5 - r0
                cols = coords[:, 0] / ps - 0.5 - c0
                rr, cc = draw_polygon(rows, cols, shape=local.shape)
                local[rr, cc] = value
        window = out[r0:r1, c0:c1]
        taken = local & (window != 0) & (window != label)
        if taken.any():
            overlap.total += int(taken.sum())
            previous = window[taken].astype(np.uint64)
            position = np.searchsorted(sorted_ids, previous)
            overlap.cross_group += int((sorted_groups[position] != group).sum())
            lost, n_lost = np.unique(position, return_counts=True)
            sorted_counts[lost] -= n_lost
        sorted_counts[np.searchsorted(sorted_ids, label)] += int(
            (local & (window != label)).sum()
        )
        window[local] = np.uint32(label)
    counts = np.zeros(ids.size, dtype=np.int64)
    counts[order] = sorted_counts
    overlap.pixel_counts = counts
    return overlap


def _label_levels(latest: Path) -> int:
    root = latest / "labels" / "cell_labels"
    levels = [path for path in root.iterdir() if path.is_dir()]
    return max(len(levels), 1)


# --------------------------------------------------------------------------- #
# Points pass
# --------------------------------------------------------------------------- #


@dataclass
class _FileResult:
    name: str
    n_rows: int
    n_claimed: int
    n_assigned: int
    n_foreground: int
    n_controls: int
    n_genes_unclaimed: int
    conflicts: Counter[str]
    losing_cells: Counter[int]
    gene_keys: np.ndarray
    gene_counts: np.ndarray
    control_keys: np.ndarray
    control_counts: np.ndarray


def _control_category(name: str) -> int:
    for index, prefixes in enumerate(CONTROL_COUNT_PREFIXES.values()):
        if name.startswith(prefixes):
            return index
    return len(CONTROL_COUNT_PREFIXES)


def _pandas_metadata(schema: pa.Schema, added: list[tuple[str, str, str]]) -> bytes:
    """Return pandas metadata that also describes the appended columns."""
    raw = (schema.metadata or {}).get(b"pandas")
    if raw is None:
        return b""
    meta = json.loads(raw)
    columns = [
        column
        for column in meta.get("columns", [])
        if column.get("name") not in {name for name, _, _ in added}
    ]
    index_names = set(meta.get("index_columns", []))
    data = [column for column in columns if column.get("name") not in index_names]
    index = [column for column in columns if column.get("name") in index_names]
    for name, pandas_type, numpy_type in added:
        data.append(
            {
                "name": name,
                "field_name": name,
                "pandas_type": pandas_type,
                "numpy_type": numpy_type,
                "metadata": None,
            }
        )
    meta["columns"] = data + index
    return json.dumps(meta).encode("utf-8")


def _rewrite_points_file(
    source: Path,
    destination: Path,
    *,
    claims: list[tuple[str, Path]],
    tiles: _Tiles,
    tile_um: float,
    cell_ids_sorted: np.ndarray,
    gene_position: dict[str, int],
    labels: np.ndarray,
    pixel_size_um: float,
) -> _FileResult:
    from merxen.enrichment.label_lookup import PixelFrame, pixel_indices

    parquet = pq.ParquetFile(source)
    metadata = parquet.metadata
    table = parquet.read()
    parquet.close()
    drop = [
        name
        for name in (ASSIGNMENT_COLUMN, BACKGROUND_COLUMN)
        if name in table.column_names
    ]
    if drop:
        table = table.drop_columns(drop)
    n_rows = table.num_rows
    ids = table.column("transcript_id").to_numpy().astype(np.uint64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    x = table.column("x").to_numpy()
    y = table.column("y").to_numpy()
    claim_sets = []
    for tid, path in claims:
        frame = pq.read_table(path)
        claim_ids = frame.column("transcript_id").to_numpy().astype(np.uint64)
        position = np.searchsorted(sorted_ids, claim_ids)
        position = np.minimum(position, max(n_rows - 1, 0))
        if n_rows == 0 or not np.array_equal(sorted_ids[position], claim_ids):
            raise AteraProsegError(
                f"Tile {tid} claims transcripts that are not in {source.name}"
            )
        claim_sets.append(
            ClaimSet(
                tile_id=tid,
                plan_index=tiles.order.index(tid),
                grid=tiles.grid[tid],
                rows=order[position],
                instance_id=frame.column("instance_id").to_numpy().astype(np.uint64),
                background=frame.column("background").to_numpy().astype(bool),
            )
        )
    resolution = resolve_claims(n_rows, x, y, tile_um, claim_sets)
    claimed = resolution.claimed
    is_gene = table.column("is_gene").to_numpy(zero_copy_only=False).astype(bool)

    # Gene foreground counts per (cell, gene).
    feature = table.column("feature_name").combine_chunks()
    if pa.types.is_dictionary(feature.type):
        names = [str(v) for v in feature.dictionary.to_pylist()]
        codes = feature.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    else:
        uniques, codes = np.unique(
            np.asarray(feature.to_pylist(), dtype=object), return_inverse=True
        )
        names = [str(v) for v in uniques]
    code_to_gene = np.asarray([gene_position.get(name, -1) for name in names], np.int64)
    counted = claimed & ~resolution.background & is_gene
    cell_row = np.searchsorted(cell_ids_sorted, resolution.assignment[counted])
    if cell_row.size and (
        (cell_row >= cell_ids_sorted.size).any()
        or not np.array_equal(cell_ids_sorted[cell_row], resolution.assignment[counted])
    ):
        raise AteraProsegError(f"{source.name}: a claimed cell is not a stitched cell")
    gene = code_to_gene[codes[counted]]
    if (gene < 0).any():
        raise AteraProsegError(
            f"{source.name}: a claimed gene is not in table_original"
        )
    n_genes = len(gene_position)
    gene_keys, gene_counts = np.unique(
        cell_row.astype(np.int64) * n_genes + gene, return_counts=True
    )

    # Controls: geometric attribution through the ProSeg label raster.
    control = ~is_gene
    control_keys = np.zeros(0, dtype=np.int64)
    control_counts = np.zeros(0, dtype=np.int64)
    if control.any():
        column, row = pixel_indices(
            x[control], y[control], PixelFrame(um_per_px=pixel_size_um)
        )
        n_label_rows, n_label_cols = labels.shape
        inside = (
            (column >= 0) & (row >= 0) & (column < n_label_cols) & (row < n_label_rows)
        )
        label_value = np.zeros(column.size, dtype=np.uint64)
        label_value[inside] = labels[row[inside], column[inside]]
        on_cell = label_value > 0
        category_of_code = np.asarray(
            [_control_category(name) for name in names], np.int64
        )
        category = category_of_code[codes[control]][on_cell]
        rows_of_cells = np.searchsorted(cell_ids_sorted, label_value[on_cell])
        n_categories = len(CONTROL_COUNT_PREFIXES) + 1
        control_keys, control_counts = np.unique(
            rows_of_cells.astype(np.int64) * n_categories + category, return_counts=True
        )

    assignment = pa.array(resolution.assignment, pa.uint64(), mask=~claimed)
    background = pa.array(resolution.background, pa.bool_())
    columns = list(table.column_names)
    index_column = "__null_dask_index__"
    insert_at = columns.index(index_column) if index_column in columns else len(columns)
    table = table.add_column(insert_at, ASSIGNMENT_COLUMN, assignment)
    table = table.add_column(insert_at + 1, BACKGROUND_COLUMN, background)
    pandas_meta = _pandas_metadata(
        table.schema,
        [(ASSIGNMENT_COLUMN, "uint64", "UInt64"), (BACKGROUND_COLUMN, "bool", "bool")],
    )
    schema_meta = dict(table.schema.metadata or {})
    if pandas_meta:
        schema_meta[b"pandas"] = pandas_meta
    table = table.replace_schema_metadata(schema_meta)
    compression = (
        metadata.row_group(0).column(0).compression.lower()
        if metadata.num_row_groups
        else "snappy"
    )
    row_group = metadata.row_group(0).num_rows if metadata.num_row_groups else None
    pq.write_table(
        table,
        destination,
        compression=compression if compression != "uncompressed" else None,
        row_group_size=row_group or None,
    )
    os.chmod(destination, 0o644)
    return _FileResult(
        name=source.name,
        n_rows=n_rows,
        n_claimed=int(claimed.sum()),
        n_assigned=int(claimed.sum()),
        n_foreground=int((claimed & ~resolution.background).sum()),
        n_controls=int(control.sum()),
        n_genes_unclaimed=int((~claimed & is_gene).sum()),
        conflicts=resolution.conflicts,
        losing_cells=resolution.losing_cells,
        gene_keys=gene_keys,
        gene_counts=gene_counts,
        control_keys=control_keys,
        control_counts=control_counts,
    )


def _rewrite_one(
    path: Path,
    *,
    partial: Path,
    claim_files: dict[str, list[tuple[str, Path]]],
    **kwargs: Any,
) -> _FileResult:
    return _rewrite_points_file(
        path, partial / path.name, claims=claim_files.get(path.name, []), **kwargs
    )


def _swap_points_directory(directory: Path, partial: Path) -> None:
    """Replace the points directory by the rewritten one (two renames)."""
    old = directory.with_name(f"{directory.name}.old-{uuid.uuid4().hex}")
    os.replace(directory, old)
    os.replace(partial, directory)
    remove_path(old)


def _remove_leftovers(directory: Path) -> None:
    """Clean up after a stitch killed while it rewrote the points.

    A ``.partial-*`` folder is an unfinished rewrite and is removed. A kill
    between the two renames of :func:`_swap_points_directory` leaves no points
    directory and the previous one as ``.old-*``, which is put back.
    """
    for path in sorted(directory.parent.glob(f"{directory.name}.partial-*")):
        remove_path(path)
    old = sorted(directory.parent.glob(f"{directory.name}.old-*"))
    if old and not directory.exists():
        os.replace(old[-1], directory)
        old = old[:-1]
    for path in old:
        remove_path(path)


# --------------------------------------------------------------------------- #
# Stitch
# --------------------------------------------------------------------------- #


def _layer_complete(latest: Path) -> bool:
    return all(
        (latest / kind / name / "zarr.json").is_file()
        or (latest / kind / name / ".zgroup").is_file()
        for kind, name in (
            ("shapes", PROSEG_SHAPES_KEY),
            ("tables", PROSEG_TABLE_KEY),
            ("labels", PROSEG_LABEL_KEY),
        )
    )


def _points_have_assignment(latest: Path) -> bool:
    directory = points_parquet_dir(latest, POINTS_KEY)
    files = sorted(directory.glob("*.parquet"))
    return bool(files) and all(
        {ASSIGNMENT_COLUMN, BACKGROUND_COLUMN} <= set(pq.read_schema(path).names)
        for path in files
    )


def _table_region(table_dir: Path) -> Any:
    """Return the ``region`` attribute of an on-disk table (zarr v3 or v2)."""
    for name in ("zarr.json", ".zattrs"):
        path = table_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        attrs = data.get("attributes", data) if name == "zarr.json" else data
        return attrs.get("region")
    return None


def derived_proseg_tables(latest: Path) -> list[str]:
    """Return the tables computed from the ProSeg cells (not the layer's own).

    Args:
        latest: The latest store.

    Returns:
        Sorted names of the tables of region ``MOSAIK_proseg`` other than
        ``table_MOSAIK_proseg`` (image quantification, clustering).
    """
    tables_dir = Path(latest) / "tables"
    if not tables_dir.is_dir():
        return []
    found = []
    for table_dir in sorted(tables_dir.iterdir()):
        if not table_dir.is_dir() or table_dir.name == PROSEG_TABLE_KEY:
            continue
        region = _table_region(table_dir)
        regions = region if isinstance(region, list) else [region]
        if PROSEG_SHAPES_KEY in regions:
            found.append(table_dir.name)
    return found


def stitch(cfg: AteraProsegStitchConfig) -> dict[str, Any]:
    """Stitch the tiles of a plan into the latest store (or skip if current).

    Args:
        cfg: The stitch configuration.

    Returns:
        The stitch report (also written to ``atera_proseg_report.json``).

    Raises:
        AteraProsegError: On a subset plan, missing or foreign tiles, or a
            latest store of another source build.
    """
    from merxen.io.spatialdata_io import spatialdata_write_lock

    target = cfg.persistent_latest_zarr_path or cfg.latest_zarr_path
    latest = Path(os.path.realpath(target))
    tiles = _load_tiles(cfg)
    _check_latest_store(latest, tiles.plan, cfg.dataset_name)
    fingerprint = stitch_fingerprint(tiles)
    with spatialdata_write_lock(latest):
        recorded = dict(read_root_attrs(latest).get(PROVENANCE_ATTR) or {})
        if (
            not cfg.force
            and recorded.get("fingerprint") == fingerprint
            and _layer_complete(latest)
            and _points_have_assignment(latest)
        ):
            logger.info(
                "[%s] ProSeg layer already stitched from these tiles (%s); skipping",
                cfg.dataset_name,
                fingerprint[:12],
            )
            report = {
                "skipped": True,
                "fingerprint": fingerprint,
                **recorded.get("stitch", {}),
            }
        else:
            try:
                report = _stitch_into(cfg, latest, tiles, fingerprint)
            finally:
                # The label raster (n_rows x n_cols uint32, about 10 GB for a
                # full section) is scratch whether or not the stitch succeeds.
                remove_path(cfg.output_dir / "scratch")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stage_existing_output(latest, cfg.output_dir / "proseg_latest.zarr")
    (cfg.output_dir / "atera_proseg_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def _stitch_into(
    cfg: AteraProsegStitchConfig,
    latest: Path,
    tiles: _Tiles,
    fingerprint: str,
) -> dict[str, Any]:
    import anndata as ad
    import dask.array as da
    import geopandas as gpd
    import shapely
    import spatialdata as sd
    from spatialdata.models import Labels2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity, Scale

    from merxen.io.spatialdata_io import delete_element, write_or_replace_element

    plan = tiles.plan
    ps = float(plan["pixel_size_um"])
    tile_um = float(plan["grid"]["tile_um"])
    n_rows_px, n_cols_px = (int(v) for v in plan["mask_shape_px"])
    cells = _load_cells(tiles)
    instance_ids = cells["instance_id"].to_numpy(dtype=np.uint64)
    geometries = list(shapely.from_wkb(cells["geometry_wkb"].to_numpy()))
    n_workers = cfg.n_workers or int(os.environ.get("DASK_NUM_WORKERS", "4") or 4)

    # Table identity from table_original (genes and vendor cell ids).
    group_path = latest / "tables" / ORIGINAL_TABLE_KEY
    original = ad.read_zarr(group_path)
    var = original.var.copy()
    gene_position = {str(name): i for i, name in enumerate(original.var_names)}
    source_ids = pd.Series(
        original.obs["source_cell_id"].astype(str).to_numpy()
        if "source_cell_id" in original.obs
        else original.obs["instance_id"].astype(str).to_numpy(),
        index=original.obs["instance_id"].to_numpy(dtype=np.uint64),
    )
    del original

    # Labels first: the control attribution reads them.
    scratch = cfg.output_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    raster_path = scratch / f"proseg_cell_labels_s0-{uuid.uuid4().hex}.npy"
    labels = np.lib.format.open_memmap(
        raster_path, mode="w+", dtype=np.uint32, shape=(n_rows_px, n_cols_px)
    )
    tile_codes = pd.Categorical(cells["proseg_tile"], categories=tiles.order).codes
    overlap = rasterise_cells(
        geometries,
        instance_ids,
        (n_rows_px, n_cols_px),
        ps,
        labels,
        groups=np.asarray(tile_codes, dtype=np.int64),
    )
    labels.flush()
    logger.info(
        "[%s] Rasterised %d ProSeg cells (%d overlap pixels, %d across tiles)",
        cfg.dataset_name,
        len(geometries),
        overlap.total,
        overlap.cross_group,
    )
    without_pixels = sorted(int(v) for v in instance_ids[overlap.pixel_counts == 0])
    if without_pixels:
        # Tiny polygons or polygons covered by later cells: they keep their
        # table row and transcripts but have no pixel in the label raster, so
        # mask-based stages (image quantification) and the control
        # attribution do not see them.
        logger.warning(
            "[%s] %d ProSeg cells have no pixel in %s (tiny or covered by "
            "other cells); mask-based stages leave them out",
            cfg.dataset_name,
            len(without_pixels),
            PROSEG_LABEL_KEY,
        )

    # Points: rewrite every file with the resolved assignment.
    directory = points_parquet_dir(latest, POINTS_KEY)
    _remove_leftovers(directory)
    partial = directory.with_name(f"{directory.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir()
    claim_files = _claim_files(tiles)
    files = sorted(directory.glob("*.parquet"))
    unknown = sorted(set(claim_files) - {path.name for path in files})
    if unknown:
        raise AteraProsegError(f"Claims name points files the store lacks: {unknown}")
    read_only = np.load(raster_path, mmap_mode="r")
    rewrite = partial_function(
        _rewrite_one,
        partial=partial,
        claim_files=claim_files,
        tiles=tiles,
        tile_um=tile_um,
        cell_ids_sorted=instance_ids,
        gene_position=gene_position,
        labels=read_only,
        pixel_size_um=ps,
    )
    try:
        with ThreadPoolExecutor(max_workers=max(n_workers, 1)) as pool:
            results = list(pool.map(rewrite, files))
        for name in os.listdir(directory):
            if not name.endswith(".parquet"):
                shutil.copy2(directory / name, partial / name)
        _swap_points_directory(directory, partial)
    except BaseException:
        remove_path(partial)
        raise

    n_genes = len(gene_position)
    n_cells = instance_ids.size
    keys = np.concatenate(
        [result.gene_keys for result in results] or [np.zeros(0, np.int64)]
    )
    counts = np.concatenate(
        [result.gene_counts for result in results] or [np.zeros(0, np.int64)]
    )
    matrix = sparse.coo_matrix(
        (counts.astype(np.float32), (keys // n_genes, keys % n_genes)),
        shape=(n_cells, n_genes),
    ).tocsr()
    matrix.sum_duplicates()
    n_categories = len(CONTROL_COUNT_PREFIXES) + 1
    control_matrix = np.zeros((n_cells, n_categories), dtype=np.int64)
    for result in results:
        if result.control_keys.size:
            np.add.at(
                control_matrix,
                (
                    result.control_keys // n_categories,
                    result.control_keys % n_categories,
                ),
                result.control_counts,
            )
    transcript_counts = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    zero = transcript_counts == 0

    obs = pd.DataFrame(
        {
            "instance_id": instance_ids,
            "region": pd.Categorical([PROSEG_SHAPES_KEY] * n_cells),
            "source_cell_id": source_ids.reindex(instance_ids).fillna("").to_numpy(),
            "proseg_internal_id": cells["proseg_internal_id"].to_numpy(np.uint32),
            "proseg_tile": pd.Categorical(cells["proseg_tile"].to_numpy()),
            "centroid_x": cells["centroid_x"].to_numpy(np.float32),
            "centroid_y": cells["centroid_y"].to_numpy(np.float32),
            "centroid_z": cells["centroid_z"].to_numpy(np.float32),
            "volume": cells["volume"].to_numpy(np.float32),
            "surface_area": cells["surface_area"].to_numpy(),
            "scale": cells["scale"].to_numpy(np.float32),
            "proseg_cluster": cells["proseg_cluster"].to_numpy(),
            "cell_area": np.asarray(shapely.area(geometries), dtype=np.float64),
            "transcript_counts": transcript_counts,
            "proseg_zero_transcripts": zero,
        },
        index=pd.Index([str(int(v)) for v in instance_ids], name="cell_id"),
    )
    for index, column in enumerate([*CONTROL_COUNT_PREFIXES, OTHER_CONTROL_COLUMN]):
        obs[column] = control_matrix[:, index]
    without_cell = sorted(
        int(v)
        for m in tiles.manifests.values()
        for v in m["cells"]["owned_labels_without_proseg_cell"]
    )
    without_polygon = sorted(
        int(v)
        for m in tiles.manifests.values()
        for v in m["cells"]["owned_cells_without_polygon"]
    )
    if without_cell or without_polygon:
        # Their transcripts are background in the layer: vendor cells lost.
        logger.warning(
            "[%s] %d vendor labels have no ProSeg cell and %d owned ProSeg cells "
            "have no polygon; they are not in %s",
            cfg.dataset_name,
            len(without_cell),
            len(without_polygon),
            PROSEG_TABLE_KEY,
        )
    conflicts: Counter[str] = Counter()
    losing: Counter[int] = Counter()
    for result in results:
        conflicts.update(result.conflicts)
        losing.update(result.losing_cells)
    stitch_record = {
        "format_version": STITCH_FORMAT_VERSION,
        "stitched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_cells": int(n_cells),
        "n_zero_transcript_cells": int(zero.sum()),
        "zero_transcript_cells": [int(v) for v in instance_ids[zero]],
        "n_vendor_labels_without_proseg_cell": len(without_cell),
        "vendor_labels_without_proseg_cell": without_cell,
        "n_owned_cells_without_polygon": len(without_polygon),
        "owned_cells_without_polygon": without_polygon,
        "n_points_files": len(results),
        "n_rows": int(sum(result.n_rows for result in results)),
        "n_rows_assigned": int(sum(result.n_assigned for result in results)),
        "n_rows_foreground": int(sum(result.n_foreground for result in results)),
        "n_rows_unclaimed_genes": int(sum(r.n_genes_unclaimed for r in results)),
        "n_controls": int(sum(result.n_controls for result in results)),
        "n_claim_conflicts": int(sum(conflicts.values())),
        "claim_conflicts_by_tile_pair": dict(sorted(conflicts.items())),
        "n_cells_losing_conflicts": len(losing),
        "cells_losing_most_conflicts": [
            [int(cell), int(count)] for cell, count in losing.most_common(20)
        ],
        "n_cells_without_label_pixels": len(without_pixels),
        "cells_without_label_pixels": without_pixels,
        "overlap_pixels_within_tile": int(overlap.within_group),
        "overlap_pixels_across_tiles": int(overlap.cross_group),
        "owned_cells_near_interior_window_edge": int(
            sum(
                len(m["edge_check"]["owned_cells_near_interior_edge"])
                for m in tiles.manifests.values()
            )
        ),
        "control_assignment": CONTROL_ASSIGNMENT_RULE,
    }

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm["spatial"] = np.column_stack(
        [obs["centroid_x"].to_numpy(np.float64), obs["centroid_y"].to_numpy(np.float64)]
    )
    first = next(iter(tiles.manifests.values()))
    provenance = {
        "version": STITCH_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "source_build_fingerprint": plan["source_build_fingerprint"],
        "proseg_version": first["proseg"]["version"],
        "proseg_git_rev": first["proseg"]["git_rev"],
        "command_template": first["proseg"]["arguments"],
        "model_params": first["model"],
        "grid": plan["grid"],
        "tiles": [_compact_manifest(tiles.manifests[tid]) for tid in tiles.order],
        "stitch": stitch_record,
    }
    adata.uns[PROVENANCE_ATTR] = {
        key: provenance[key]
        for key in ("version", "fingerprint", "proseg_version", "model_params", "grid")
    }
    adata.uns[PROVENANCE_ATTR]["control_assignment"] = CONTROL_ASSIGNMENT_RULE
    table = TableModel.parse(
        adata, region=PROSEG_SHAPES_KEY, region_key="region", instance_key="instance_id"
    )

    gdf = gpd.GeoDataFrame(
        {
            "instance_id": instance_ids,
            "source_cell_id": obs["source_cell_id"].to_numpy(),
            "proseg_internal_id": cells["proseg_internal_id"].to_numpy(np.uint32),
            "proseg_tile": cells["proseg_tile"].to_numpy(),
        },
        geometry=geometries,
        index=pd.Index(instance_ids, name="instance_id"),
    )
    shapes = ShapesModel.parse(
        gdf, transformations={"global": Scale([1 / ps, 1 / ps], axes=("x", "y"))}
    )
    n_levels = _label_levels(latest)
    label_element = Labels2DModel.parse(
        da.from_array(read_only, chunks=(LABEL_BLOCK_PX, LABEL_BLOCK_PX)),
        dims=("y", "x"),
        scale_factors=[2] * (n_levels - 1) if n_levels > 1 else None,
        transformations={"global": Identity()},
    )

    sdata = sd.read_zarr(latest, selection=("shapes", "tables", "labels"))
    write_or_replace_element(sdata, PROSEG_SHAPES_KEY, "shapes", shapes)
    write_or_replace_element(sdata, PROSEG_LABEL_KEY, "labels", label_element)
    write_or_replace_element(sdata, PROSEG_TABLE_KEY, "tables", table)
    stale = derived_proseg_tables(latest)
    for key in stale:
        delete_element(sdata, key, "tables")
    if stale:
        logger.info(
            "[%s] Deleted %d tables of the previous ProSeg cells: %s",
            cfg.dataset_name,
            len(stale),
            ", ".join(stale),
        )
    stitch_record["deleted_derived_tables"] = stale
    register_segmentation_branch(
        sdata,
        PROSEG_BRANCH,
        points_key=POINTS_KEY,
        assignment_column=ASSIGNMENT_COLUMN,
        background_column=BACKGROUND_COLUMN,
        shape_key=PROSEG_SHAPES_KEY,
        table_key=PROSEG_TABLE_KEY,
        id_namespace=ORIGINAL_ID_NAMESPACE,
        legacy_aliases=("reseg",),
        label_key=PROSEG_LABEL_KEY,
        prior=PRIOR_NAME,
    )
    schema = dict(sdata.attrs[MERXEN_SCHEMA_ATTR])
    registry = dict(schema["segmentations"])
    registry[PROSEG_BRANCH] = {
        **registry[PROSEG_BRANCH],
        "model_identity": {
            "tool": "proseg",
            "version": provenance["proseg_version"],
            "git_rev": provenance["proseg_git_rev"],
        },
    }
    schema["segmentations"] = registry
    sdata.attrs[MERXEN_SCHEMA_ATTR] = schema
    sdata.attrs[PROVENANCE_ATTR] = provenance
    sdata.write_attrs()
    sdata.write_consolidated_metadata()
    del sdata, read_only, labels
    remove_path(scratch)
    logger.info(
        "[%s] Stitched %d ProSeg cells from %d tiles: %d assigned rows, %d "
        "conflicts, %d cells with no transcripts",
        cfg.dataset_name,
        n_cells,
        len(tiles.order),
        stitch_record["n_rows_assigned"],
        stitch_record["n_claim_conflicts"],
        stitch_record["n_zero_transcript_cells"],
    )
    return {"skipped": False, "fingerprint": fingerprint, **stitch_record}


def _compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a tile manifest without its bulky per-cell lists."""
    compact: dict[str, Any] = json.loads(json.dumps(manifest, default=str))
    cells = compact.get("cells", {})
    for key in (
        "owned_labels_without_proseg_cell",
        "owned_cells_without_polygon",
        "owned_cells_zero_transcripts",
    ):
        values = cells.get(key, [])
        cells[f"n_{key}"] = len(values)
        cells.pop(key, None)
    edge = compact.get("edge_check", {})
    near = edge.pop("owned_cells_near_interior_edge", [])
    edge["n_owned_cells_near_interior_edge"] = len(near)
    compact.get("export", {}).pop("rows_per_file", None)
    compact.get("rows", {}).pop("claims_per_file", None)
    compact.pop("model", None)
    return compact


# --------------------------------------------------------------------------- #
# Layer check (ENRICH of a native_proseg row)
# --------------------------------------------------------------------------- #


def _check_points_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the foreground gene counts per assigned cell of one file."""
    table = pq.read_table(
        path, columns=[ASSIGNMENT_COLUMN, BACKGROUND_COLUMN, "is_gene"]
    )
    assignment = table.column(ASSIGNMENT_COLUMN).combine_chunks()
    valid = assignment.is_valid().to_numpy(zero_copy_only=False)
    values = assignment.fill_null(0).to_numpy().astype(np.uint64)
    background = table.column(BACKGROUND_COLUMN).to_numpy(zero_copy_only=False)
    is_gene = table.column("is_gene").to_numpy(zero_copy_only=False).astype(bool)
    counted = valid & ~background.astype(bool) & is_gene
    cells, counts = np.unique(values[counted], return_counts=True)
    assigned_ids = np.unique(values[valid])
    return cells, counts, assigned_ids


def check_proseg_layer(
    latest_path: Path, n_workers: int | None = None
) -> dict[str, Any]:
    """Check the stitched ProSeg layer of a latest store in one streaming pass.

    Checks that the table and shape ids are equal, every non-null
    ``assignment`` is a table id, the per-cell foreground gene counts of the
    points equal the table's row sums (int64), and the registry entry is the
    expected one.

    Args:
        latest_path: The latest store.
        n_workers: Threads over the points files.

    Returns:
        ``{"passed": bool, "failures": [...], ...}`` with the counts checked.

    Raises:
        AteraProsegError: If the store has no stitched ProSeg layer.
    """
    import anndata as ad
    import geopandas as gpd
    import zarr

    latest = Path(latest_path)
    if not _layer_complete(latest) or not _points_have_assignment(latest):
        raise AteraProsegError(
            "native_proseg row without a stitched ProSeg layer: the latest store "
            f"lacks {PROSEG_SHAPES_KEY}, {PROSEG_TABLE_KEY}, {PROSEG_LABEL_KEY} or "
            "the points assignment; run the segment stage (ATERA_PROSEG_STITCH)"
        )
    failures: list[str] = []
    group = zarr.open_group(str(latest / "tables" / PROSEG_TABLE_KEY), mode="r")
    obs = ad.io.read_elem(group["obs"])
    x_matrix = ad.io.read_elem(group["X"])
    table_ids = obs["instance_id"].to_numpy(dtype=np.uint64)
    row_sums = np.asarray(sparse.csr_matrix(x_matrix).sum(axis=1)).ravel()
    row_sums = np.rint(row_sums).astype(np.int64)
    shapes = gpd.read_parquet(latest / "shapes" / PROSEG_SHAPES_KEY / "shapes.parquet")
    shape_ids = shapes["instance_id"].to_numpy(dtype=np.uint64)
    if set(table_ids.tolist()) != set(shape_ids.tolist()):
        failures.append("table and shape ids differ")
    order = np.argsort(table_ids)
    sorted_ids = table_ids[order]
    counted = np.zeros(table_ids.size, dtype=np.int64)
    unknown = 0
    files = sorted(points_parquet_dir(latest, POINTS_KEY).glob("*.parquet"))
    workers = max(n_workers or int(os.environ.get("DASK_NUM_WORKERS", "4") or 4), 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cells, counts, assigned in pool.map(_check_points_file, files):
            position = np.searchsorted(sorted_ids, assigned)
            position = np.minimum(position, max(sorted_ids.size - 1, 0))
            if sorted_ids.size == 0:
                unknown += int(assigned.size)
            else:
                unknown += int((sorted_ids[position] != assigned).sum())
            if cells.size:
                rows = order[np.searchsorted(sorted_ids, cells)]
                np.add.at(counted, rows, counts.astype(np.int64))
    if unknown:
        failures.append(f"{unknown} assigned cell ids are not table ids")
    differing = int((counted != row_sums).sum())
    if differing:
        failures.append(
            f"{differing} cells: points foreground counts != table row sums"
        )
    entry = (
        dict(read_root_attrs(latest).get(MERXEN_SCHEMA_ATTR, {}) or {})
        .get("segmentations", {})
        .get(PROSEG_BRANCH, {})
    )
    expected = {
        "shape": PROSEG_SHAPES_KEY,
        "table": PROSEG_TABLE_KEY,
        "assignment_column": ASSIGNMENT_COLUMN,
        "background_column": BACKGROUND_COLUMN,
        "id_namespace": ORIGINAL_ID_NAMESPACE,
        "label": PROSEG_LABEL_KEY,
        "prior": PRIOR_NAME,
    }
    wrong = sorted(key for key, value in expected.items() if entry.get(key) != value)
    if wrong:
        failures.append(f"registry entry {PROSEG_BRANCH} differs in {', '.join(wrong)}")
    provenance = dict(read_root_attrs(latest).get(PROVENANCE_ATTR) or {})
    return {
        "passed": not failures,
        "failures": failures,
        "fingerprint": provenance.get("fingerprint"),
        "n_cells": int(table_ids.size),
        "n_points_files": len(files),
        "n_foreground_gene_rows": int(counted.sum()),
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
