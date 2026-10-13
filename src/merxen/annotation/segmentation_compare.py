"""Helpers for comparing two segmentations of one section (M8b; pre-reg §19).

The M8b comparison (``scripts/acceptance/segmentation_comparison.py``) sets
reseg (ProSeg's own transcript assignment) against proseg_hybrid (every
transcript inside a boundary drawn around ProSeg's assignment). Both share
their cell ids, so most of the work is joins on cell id. The pure parts that
decide a number are here, on plain arrays, so they are unit tested on small
synthetic inputs and mutation-checked:

- **Depth matching** (``thin_to_targets``): each cell's counts thinned to a
  target total by a draw without replacement from its own counts
  (multivariate hypergeometric). The generator of a cell is seeded from the
  run seed and the cell id (``cell_key``), so a cell's draw does not depend
  on row order or on the other cells.
- **Transcripts in nuclei** (``points_in_polygons``, ``nucleus_holders``,
  ``transcript_cells``): which nucleus polygon a transcript lies in, which
  cell holds each nucleus (largest overlap) and which cell a transcript
  belongs to (the nucleus holder for an in-nucleus transcript, else its
  assigned table cell).
- **Assignment tallies and shares** (``loss_tally``, ``loss_shares``): per
  group and gene, the decoded transcripts, those assigned under each
  segmentation, those in a nucleus and those in a nucleus that reseg leaves
  unassigned; the assigned shares, the in-nucleus unassigned share and the
  loss ``L = 1 - A_reseg / A_hybrid``.
- **Neighbour distances** (``nearest_other_class_distance``,
  ``distance_bin_codes``): the distance from a cell to the nearest confident
  cell of another class, binned on the pre-registered edges.
- **Spatial block bootstrap of a paired ratio** (``tile_bootstrap_ratio``):
  per group, the arm-B minus arm-A difference of a mean over cells, with a
  percentile CI from resampling 500 µm tiles (one draw for both arms).
- **Tests** (``benjamini_hochberg``, ``spearman_bootstrap``,
  ``mann_whitney``, ``wilcoxon_paired``): p values with their effect sizes.
- **Pseudobulk shares** (``pseudobulk_shares``, ``reference_broad_shares``)
  and the combined labelling (``combined_labels``).
- **Gene covariates** (``longest_transcript_lengths``,
  ``xenium_probe_counts``).

The module needs numpy, pandas and scipy; shapely only for the polygon
helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import sparse, stats

THINNING_SEED: Final = 0
# Pre-registration §19 analysis 3: bins 0-10, 10-15, 15-20, 20-30, 30-50 and
# > 50 µm. A bin holds [lower, upper): a distance of exactly 10 µm is in
# 10-15, of exactly 50 µm in > 50.
DISTANCE_BIN_EDGES_UM: Final[tuple[float, ...]] = (
    0.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    math.inf,
)
DISTANCE_BIN_LABELS: Final[tuple[str, ...]] = (
    "0-10",
    "10-15",
    "15-20",
    "20-30",
    "30-50",
    ">50",
)
N_TILE_BOOTSTRAP: Final = 200
N_GENE_BOOTSTRAP: Final = 1000
BOOTSTRAP_SEED: Final = 0
CI_PERCENTILES: Final = (2.5, 97.5)
PSEUDOCOUNT: Final = 1.0
# Fields of a loss tally, per group and gene.
TALLY_FIELDS: Final[tuple[str, ...]] = (
    "n_decoded",
    "n_assigned_reseg",
    "n_assigned_hybrid",
    "n_assigned_reseg_same_cell",
    "n_in_nucleus",
    "n_in_nucleus_unassigned_reseg",
    "n_in_nucleus_background_reseg",
    "n_in_nucleus_nontable_reseg",
)
PRIMARY_CHROMOSOMES: Final[frozenset[str]] = frozenset(
    {f"chr{index}" for index in range(1, 23)} | {"chrX", "chrY", "chrM"}
)


# --------------------------------------------------------------------------
# Depth matching


def cell_key(cell_id: object) -> int:
    """Return a cell id's 64-bit seed key (the first 8 bytes of its sha256).

    The key depends on the id's string only, so it is the same on every
    host, in every run and whatever the row order.

    Args:
        cell_id: A cell id (any value; its ``str`` is hashed).

    Returns:
        A non-negative integer below ``2**64``.
    """
    digest = hashlib.sha256(str(cell_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def cell_generator(cell_id: object, seed: int = THINNING_SEED) -> np.random.Generator:
    """Return the random generator of one cell (seeded by run seed and cell id)."""
    sequence = np.random.SeedSequence([int(seed), cell_key(cell_id)])
    return np.random.Generator(np.random.PCG64(sequence))


@dataclass(frozen=True)
class ThinningResult:
    """Counts thinned to per-cell targets.

    Attributes:
        counts: The thinned ``(n_cells, n_genes)`` counts (CSR, int64).
        thinned: Whether each cell was thinned (total above its target).
        totals_before: Each cell's total before thinning.
        totals_after: Each cell's total after thinning.
    """

    counts: sparse.csr_matrix
    thinned: np.ndarray
    totals_before: np.ndarray
    totals_after: np.ndarray


def thin_to_targets(
    counts: sparse.spmatrix | np.ndarray,
    cell_ids: Sequence[object],
    targets: Sequence[int] | np.ndarray,
    *,
    seed: int = THINNING_SEED,
) -> ThinningResult:
    """Thin each cell's counts to its target total, without replacement.

    A cell whose total is at most its target keeps its counts. Any other
    cell keeps a multivariate hypergeometric draw of ``target`` of its own
    transcripts (each transcript equally likely), from the generator
    ``cell_generator(cell_id, seed)`` over its non-zero genes in column order.

    Args:
        counts: ``(n_cells, n_genes)`` non-negative integer counts.
        cell_ids: One id per row (the seed key; ids should be unique).
        targets: One non-negative integer target per row.
        seed: The run seed (0 in the pre-registration).

    Returns:
        The thinned counts and per-cell totals.

    Raises:
        ValueError: If the shapes disagree, a target is negative or a count
            is negative or not an integer.
    """
    matrix = sparse.csr_matrix(counts, copy=True)
    matrix.sort_indices()
    n_cells = matrix.shape[0]
    goal = np.asarray(targets)
    if len(cell_ids) != n_cells or goal.shape != (n_cells,):
        raise ValueError("cell_ids and targets need one entry per row")
    if np.any(goal < 0) or not np.all(np.equal(np.mod(goal, 1), 0)):
        raise ValueError("targets must be non-negative integers")
    data = matrix.data
    if np.any(data < 0) or not np.all(np.equal(np.mod(data, 1), 0)):
        raise ValueError("counts must be non-negative integers")
    data = data.astype(np.int64)
    goal = goal.astype(np.int64)
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)
    thinned = totals > goal
    out = data.copy()
    for row in np.flatnonzero(thinned):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        generator = cell_generator(cell_ids[row], seed)
        out[start:stop] = generator.multivariate_hypergeometric(
            data[start:stop], int(goal[row])
        )
    result = sparse.csr_matrix(
        (out, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape
    )
    result.eliminate_zeros()
    after = np.asarray(result.sum(axis=1)).ravel().astype(np.int64)
    return ThinningResult(
        counts=result,
        thinned=thinned,
        totals_before=totals,
        totals_after=after,
    )


# --------------------------------------------------------------------------
# Transcripts in nuclei


def points_in_polygons(
    x: np.ndarray,
    y: np.ndarray,
    polygons: Sequence[Any] | np.ndarray,
    *,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """Return the polygon each point lies in (``-1`` for none).

    A point on a polygon's boundary lies in it (``intersects``). Where
    polygons overlap, the point takes the lowest polygon index.

    Args:
        x: Point x coordinates.
        y: Point y coordinates (same frame as the polygons).
        polygons: Shapely polygons.
        chunk_size: Points per spatial query (bounds memory).

    Returns:
        ``int64`` polygon index per point.
    """
    import shapely

    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same shape")
    geometries = np.asarray(polygons, dtype=object)
    found = np.full(len(xs), -1, dtype=np.int64)
    if not len(geometries) or not len(xs):
        return found
    tree = shapely.STRtree(geometries)
    for start in range(0, len(xs), chunk_size):
        stop = min(start + chunk_size, len(xs))
        points = shapely.points(xs[start:stop], ys[start:stop])
        point_index, polygon_index = tree.query(points, predicate="intersects")
        if not len(point_index):
            continue
        order = np.lexsort((polygon_index, point_index))
        point_index, polygon_index = point_index[order], polygon_index[order]
        first = np.ones(len(point_index), dtype=bool)
        first[1:] = point_index[1:] != point_index[:-1]
        found[start + point_index[first]] = polygon_index[first]
    return found


def nucleus_holders(
    nuclei: Sequence[Any] | np.ndarray,
    cells: Sequence[Any] | np.ndarray,
    cell_ids: Sequence[object],
) -> np.ndarray:
    """Return the cell that holds each nucleus: its largest overlap.

    Args:
        nuclei: Nucleus polygons.
        cells: Cell polygons (same frame).
        cell_ids: One id per cell polygon.

    Returns:
        ``object`` array, one cell id per nucleus; ``None`` where no cell
        overlaps the nucleus with a positive area. Ties go to the cell that
        comes first in ``cells``.
    """
    import shapely

    nucleus_geometries = np.asarray(nuclei, dtype=object)
    cell_geometries = np.asarray(cells, dtype=object)
    ids = np.asarray(list(cell_ids), dtype=object)
    if len(ids) != len(cell_geometries):
        raise ValueError("cell_ids needs one id per cell polygon")
    holders = np.full(len(nucleus_geometries), None, dtype=object)
    if not len(nucleus_geometries) or not len(cell_geometries):
        return holders
    tree = shapely.STRtree(cell_geometries)
    nucleus_index, cell_index = tree.query(nucleus_geometries, predicate="intersects")
    if not len(nucleus_index):
        return holders
    areas = shapely.area(
        shapely.intersection(
            nucleus_geometries[nucleus_index], cell_geometries[cell_index]
        )
    )
    keep = areas > 0
    nucleus_index, cell_index, areas = (
        nucleus_index[keep],
        cell_index[keep],
        areas[keep],
    )
    # Largest area first, then the lowest cell index (ties).
    order = np.lexsort((cell_index, -areas, nucleus_index))
    nucleus_index, cell_index = nucleus_index[order], cell_index[order]
    first = np.ones(len(nucleus_index), dtype=bool)
    first[1:] = nucleus_index[1:] != nucleus_index[:-1]
    holders[nucleus_index[first]] = ids[cell_index[first]]
    return holders


def transcript_cells(
    nucleus_index: np.ndarray,
    holders: np.ndarray,
    assigned_cell: np.ndarray,
) -> np.ndarray:
    """Return the cell each transcript belongs to (pre-registration §19).

    An in-nucleus transcript belongs to the cell that holds its nucleus
    (none if no cell holds it); any other transcript to the table cell it is
    assigned to (``None`` if none).

    Args:
        nucleus_index: Nucleus per transcript (``-1``: not in a nucleus).
        holders: Holder cell id per nucleus (``nucleus_holders``).
        assigned_cell: The assigned table cell per transcript (``None``, or
            a sentinel id for an integer array).

    Returns:
        The cell id per transcript, with ``assigned_cell``'s dtype.
    """
    index = np.asarray(nucleus_index, dtype=np.int64)
    assigned = np.asarray(assigned_cell)
    if index.shape != assigned.shape:
        raise ValueError("nucleus_index and assigned_cell must align")
    out = assigned.copy()
    inside = index >= 0
    out[inside] = np.asarray(holders, dtype=assigned.dtype)[index[inside]]
    return out


# --------------------------------------------------------------------------
# Assignment tallies and shares


def loss_tally(
    gene_codes: np.ndarray,
    n_genes: int,
    *,
    assigned_reseg: np.ndarray,
    assigned_hybrid: np.ndarray,
    assigned_reseg_same_cell: np.ndarray,
    in_nucleus: np.ndarray,
    background_reseg: np.ndarray,
    group_codes: np.ndarray | None = None,
    n_groups: int = 1,
) -> np.ndarray:
    """Count transcripts per group and gene for every ``TALLY_FIELDS`` entry.

    Args:
        gene_codes: Gene index per transcript (``0 .. n_genes - 1``).
        n_genes: Number of genes.
        assigned_reseg: Assigned to a reseg table cell (``assignment`` set,
            ``background`` false, a table cell).
        assigned_hybrid: Assigned to a proseg_hybrid table cell.
        assigned_reseg_same_cell: Assigned under reseg to the table cell of
            the same id as the transcript's proseg_hybrid cell.
        in_nucleus: In a nucleus polygon.
        background_reseg: ProSeg's ``background`` flag.
        group_codes: Group per transcript (``-1``: in no group); ``None``
            puts every transcript in group 0.
        n_groups: Number of groups.

    Returns:
        ``(n_groups, n_genes, len(TALLY_FIELDS))`` int64 counts. An
        in-nucleus transcript that reseg leaves unassigned is either
        background or assigned to a non-table cell.
    """
    genes = np.asarray(gene_codes, dtype=np.int64)
    groups = (
        np.zeros(len(genes), dtype=np.int64)
        if group_codes is None
        else np.asarray(group_codes, dtype=np.int64)
    )
    flags = [
        np.asarray(item, dtype=bool)
        for item in (
            assigned_reseg,
            assigned_hybrid,
            assigned_reseg_same_cell,
            in_nucleus,
            background_reseg,
        )
    ]
    if any(item.shape != genes.shape for item in flags) or groups.shape != genes.shape:
        raise ValueError("every per-transcript array must align")
    if len(genes) and (genes.min() < 0 or genes.max() >= n_genes):
        raise ValueError("gene codes out of range")
    reseg, hybrid, same, nucleus, background = flags
    keep = (groups >= 0) & (groups < n_groups)
    flat = groups[keep] * n_genes + genes[keep]
    size = n_groups * n_genes
    unassigned_nucleus = nucleus & ~reseg
    columns = (
        np.ones(len(genes), dtype=bool),
        reseg,
        hybrid,
        same & reseg,
        nucleus,
        unassigned_nucleus,
        unassigned_nucleus & background,
        unassigned_nucleus & ~background,
    )
    out = np.zeros((size, len(TALLY_FIELDS)), dtype=np.int64)
    for position, column in enumerate(columns):
        out[:, position] = np.bincount(
            flat, weights=column[keep].astype(np.float64), minlength=size
        ).astype(np.int64)
    return out.reshape(n_groups, n_genes, len(TALLY_FIELDS))


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = numerator.astype(np.float64)
    bottom = denominator.astype(np.float64)
    return (top / bottom.where(bottom > 0)).astype(np.float64)


def loss_shares(tally: pd.DataFrame) -> pd.DataFrame:
    """Add the pre-registered shares to a tally frame (one row per gene).

    - ``A_reseg`` / ``A_hybrid``: assigned share of the decoded transcripts;
    - ``N_bg``: share of the in-nucleus transcripts that reseg leaves
      unassigned (with its background / non-table parts);
    - ``L = 1 - A_reseg / A_hybrid`` (NaN when ``A_hybrid`` is 0);
    - ``A_reseg_same_cell``: share assigned under reseg to the same cell id.

    Args:
        tally: Columns ``TALLY_FIELDS``.

    Returns:
        A copy with the share columns.
    """
    out = tally.copy()
    out["A_reseg"] = _ratio(out["n_assigned_reseg"], out["n_decoded"])
    out["A_hybrid"] = _ratio(out["n_assigned_hybrid"], out["n_decoded"])
    out["A_reseg_same_cell"] = _ratio(
        out["n_assigned_reseg_same_cell"], out["n_decoded"]
    )
    out["N_bg"] = _ratio(out["n_in_nucleus_unassigned_reseg"], out["n_in_nucleus"])
    out["N_bg_background"] = _ratio(
        out["n_in_nucleus_background_reseg"], out["n_in_nucleus"]
    )
    out["N_bg_nontable"] = _ratio(
        out["n_in_nucleus_nontable_reseg"], out["n_in_nucleus"]
    )
    hybrid = out["A_hybrid"].where(out["A_hybrid"] > 0)
    out["L"] = 1.0 - out["A_reseg"] / hybrid
    return out


# --------------------------------------------------------------------------
# Neighbour distances


def nearest_other_class_distance(
    xy: np.ndarray,
    classes: Sequence[object] | np.ndarray,
    query: np.ndarray | None = None,
) -> np.ndarray:
    """Return each cell's distance to the nearest cell of another class.

    Args:
        xy: ``(n, 2)`` cell centroids (µm).
        classes: Confident class per cell (``None`` / NaN: not confident;
            such cells are never a neighbour and get NaN).
        query: Cells to measure (default: all); others get NaN.

    Returns:
        Distances (NaN where the cell has no class, no other class exists or
        the cell is not queried).
    """
    from scipy.spatial import cKDTree

    points = np.asarray(xy, dtype=np.float64)
    labels = np.asarray(classes, dtype=object)
    if points.ndim != 2 or points.shape[1] != 2 or len(labels) != len(points):
        raise ValueError("xy must be (n, 2) with one class per cell")
    wanted = (
        np.ones(len(points), dtype=bool)
        if query is None
        else np.asarray(query, dtype=bool)
    )
    has = np.asarray(pd.notna(labels), dtype=bool)
    finite = np.isfinite(points).all(axis=1)
    out = np.full(len(points), math.nan)
    usable = has & finite
    as_str = np.array(
        [str(value) if ok else "" for value, ok in zip(labels, has, strict=True)]
    )
    present = sorted(set(as_str[usable]))
    if len(present) < 2:
        return out
    nearest = np.full((len(points), len(present)), math.inf)
    targets = np.flatnonzero(wanted & usable)
    for column, name in enumerate(present):
        members = usable & (as_str == name)
        tree = cKDTree(points[members])
        distance, _ = tree.query(points[targets], k=1)
        nearest[targets, column] = distance
    for column, name in enumerate(present):
        own = targets[as_str[targets] == name]
        nearest[own, column] = math.inf
    best = nearest[targets].min(axis=1)
    out[targets] = np.where(np.isfinite(best), best, math.nan)
    return out


def distance_bin_codes(
    distances: np.ndarray, edges: Sequence[float] = DISTANCE_BIN_EDGES_UM
) -> np.ndarray:
    """Return the bin of each distance on ``[edges[i], edges[i + 1])``.

    Args:
        distances: Distances (µm).
        edges: Increasing bin edges; the first is 0, the last may be inf.

    Returns:
        ``int64`` bin index; ``-1`` for NaN, a distance below ``edges[0]``
        or one at or above a finite last edge.
    """
    values = np.asarray(distances, dtype=np.float64)
    bounds = np.asarray(edges, dtype=np.float64)
    if np.any(np.diff(bounds) <= 0):
        raise ValueError("edges must increase")
    codes = np.searchsorted(bounds[1:-1], values, side="right").astype(np.int64)
    invalid = np.isnan(values) | (values < bounds[0])
    if math.isfinite(bounds[-1]):
        invalid |= values >= bounds[-1]
    codes[invalid] = -1
    return codes


# --------------------------------------------------------------------------
# Spatial block bootstrap of a paired mean


@dataclass(frozen=True)
class PairedTileBootstrap:
    """Per-group means of two arms, their difference and its tile CI.

    Attributes:
        mean_a: Arm A's mean per group (NaN without cells).
        mean_b: Arm B's mean per group.
        n_a: Arm A's cells per group.
        n_b: Arm B's cells per group.
        difference: ``mean_b - mean_a``.
        ci_low: 2.5th percentile of the replicate differences.
        ci_high: 97.5th percentile.
        n_tiles: Non-empty tiles resampled.
        n_reps: Replicates.
    """

    mean_a: np.ndarray
    mean_b: np.ndarray
    n_a: np.ndarray
    n_b: np.ndarray
    difference: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_tiles: int
    n_reps: int


def _group_tile_sums(
    values: np.ndarray, groups: np.ndarray, tiles: np.ndarray, n_tiles: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(values) & (groups >= 0) & (groups < n) & (tiles >= 0)
    flat = tiles[keep] * n + groups[keep]
    size = n_tiles * n
    sums = np.bincount(flat, weights=values[keep], minlength=size).reshape(n_tiles, n)
    counts = np.bincount(flat, minlength=size).reshape(n_tiles, n).astype(np.float64)
    return sums, counts


def tile_bootstrap_ratio(
    values_a: np.ndarray,
    groups_a: np.ndarray,
    values_b: np.ndarray,
    groups_b: np.ndarray,
    tiles: np.ndarray,
    n_groups: int,
    *,
    n_reps: int = N_TILE_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> PairedTileBootstrap:
    """Bootstrap the per-group difference of two arms' means over tiles.

    Each cell has one tile (shared by both arms) and, per arm, a value and a
    group (e.g. a distance bin; ``-1`` / non-finite values are left out).
    The mean of a group is the sum of its values over its cell count. Each
    replicate draws the non-empty tiles with replacement (multinomial
    weights, as ``composition.block_bootstrap_jsd``) and applies the same
    weights to both arms, so the CI is that of the paired difference.

    Args:
        values_a: Arm A value per cell.
        groups_a: Arm A group per cell.
        values_b: Arm B value per cell.
        groups_b: Arm B group per cell.
        tiles: Tile id per cell (``composition.tile_codes``; ``-1``: none).
        n_groups: Number of groups.
        n_reps: Replicates (200).
        seed: RNG seed (0).

    Returns:
        Means, difference and percentile CI per group.
    """
    tile_codes = np.asarray(tiles, dtype=np.int64)
    arrays = [
        np.asarray(values_a, dtype=np.float64),
        np.asarray(groups_a, dtype=np.int64),
        np.asarray(values_b, dtype=np.float64),
        np.asarray(groups_b, dtype=np.int64),
    ]
    if any(item.shape != tile_codes.shape for item in arrays):
        raise ValueError("every per-cell array must align")
    n_tiles = (
        int(tile_codes.max()) + 1 if len(tile_codes) and tile_codes.max() >= 0 else 0
    )
    sum_a, count_a = _group_tile_sums(
        arrays[0], arrays[1], tile_codes, n_tiles, n_groups
    )
    sum_b, count_b = _group_tile_sums(
        arrays[2], arrays[3], tile_codes, n_tiles, n_groups
    )
    nonempty = (count_a.sum(axis=1) + count_b.sum(axis=1)) > 0
    sum_a, count_a = sum_a[nonempty], count_a[nonempty]
    sum_b, count_b = sum_b[nonempty], count_b[nonempty]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_a = sum_a.sum(axis=0) / count_a.sum(axis=0)
        mean_b = sum_b.sum(axis=0) / count_b.sum(axis=0)
    n_kept = int(nonempty.sum())
    low = np.full(n_groups, math.nan)
    high = np.full(n_groups, math.nan)
    if n_kept:
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(n_kept, np.full(n_kept, 1 / n_kept), n_reps).astype(
            np.float64
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            reps = (weights @ sum_b) / (weights @ count_b) - (weights @ sum_a) / (
                weights @ count_a
            )
        for group in range(n_groups):
            finite = reps[:, group][np.isfinite(reps[:, group])]
            if len(finite):
                low[group], high[group] = np.percentile(finite, CI_PERCENTILES)
    return PairedTileBootstrap(
        mean_a=mean_a,
        mean_b=mean_b,
        n_a=count_a.sum(axis=0).astype(np.int64),
        n_b=count_b.sum(axis=0).astype(np.int64),
        difference=mean_b - mean_a,
        ci_low=low,
        ci_high=high,
        n_tiles=n_kept,
        n_reps=int(n_reps),
    )


# --------------------------------------------------------------------------
# Tests with effect sizes


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p values (q values).

    NaN p values are left out of the family and stay NaN.

    Args:
        p_values: The family's p values.

    Returns:
        q values in input order, monotone in p and capped at 1.
    """
    values = np.asarray(p_values, dtype=np.float64)
    out = np.full(values.shape, math.nan)
    finite = np.flatnonzero(np.isfinite(values))
    m = len(finite)
    if not m:
        return out
    order = finite[np.argsort(values[finite], kind="mergesort")]
    ranked = values[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(adjusted, 1.0)
    return out


@dataclass(frozen=True)
class TestResult:
    """One test: statistic, p value and effect size.

    Attributes:
        test: Test name.
        statistic: Test statistic.
        p_value: Two-sided p value.
        effect: Effect size (see ``effect_name``).
        effect_name: What ``effect`` is.
        ci_low: Lower 95% bound of the effect, if any.
        ci_high: Upper 95% bound.
        n: Observations (genes or gene pairs).
        n_other: Second group's size (Mann-Whitney).
        median_difference: Median of group 1 minus median of group 2
            (Mann-Whitney) or median paired difference (Wilcoxon).
        note: Why the test did not run, if it did not.
    """

    test: str
    statistic: float
    p_value: float
    effect: float
    effect_name: str
    ci_low: float = math.nan
    ci_high: float = math.nan
    n: int = 0
    n_other: int = 0
    median_difference: float = math.nan
    note: str = ""

    def to_row(self: TestResult) -> dict[str, Any]:
        """Return the result as a flat record."""
        return dict(self.__dict__)


MIN_TEST_N: Final = 3


def spearman_bootstrap(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    n_reps: int = N_GENE_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> TestResult:
    """Spearman rho with a percentile CI from resampling observations.

    Args:
        x: First variable (pairs with a non-finite value are dropped).
        y: Second variable.
        n_reps: Bootstrap replicates (1,000).
        seed: RNG seed (0).

    Returns:
        rho, its two-sided p value and the 95% bootstrap CI of rho.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("x and y must align")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = len(a)
    if n < MIN_TEST_N or np.ptp(a) == 0 or np.ptp(b) == 0:
        return TestResult(
            "spearman",
            math.nan,
            math.nan,
            math.nan,
            "rho",
            n=n,
            note="fewer than 3 finite pairs or a constant variable",
        )
    result = stats.spearmanr(a, b)
    rho, p_value = float(result.statistic), float(result.pvalue)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_reps, n))
    ranks_a = stats.rankdata(a[draws], axis=1)
    ranks_b = stats.rankdata(b[draws], axis=1)
    centred_a = ranks_a - ranks_a.mean(axis=1, keepdims=True)
    centred_b = ranks_b - ranks_b.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        reps = (centred_a * centred_b).sum(axis=1) / np.sqrt(
            (centred_a**2).sum(axis=1) * (centred_b**2).sum(axis=1)
        )
    reps = reps[np.isfinite(reps)]
    low, high = (
        np.percentile(reps, CI_PERCENTILES) if len(reps) else (math.nan, math.nan)
    )
    return TestResult(
        "spearman",
        rho,
        p_value,
        rho,
        "rho",
        ci_low=float(low),
        ci_high=float(high),
        n=n,
    )


def mann_whitney(
    first: Sequence[float] | np.ndarray, second: Sequence[float] | np.ndarray
) -> TestResult:
    """Two-sided Mann-Whitney U test with the rank-biserial correlation.

    Args:
        first: Group 1 values (e.g. list members; non-finite dropped).
        second: Group 2 values.

    Returns:
        U of group 1, p, rank-biserial ``2 U / (n1 n2) - 1`` (positive when
        group 1 tends to be larger) and the median difference.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_TEST_N or len(b) < MIN_TEST_N:
        return TestResult(
            "mann_whitney",
            math.nan,
            math.nan,
            math.nan,
            "rank_biserial",
            n=len(a),
            n_other=len(b),
            note="fewer than 3 finite values in a group",
        )
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    u = float(result.statistic)
    return TestResult(
        "mann_whitney",
        u,
        float(result.pvalue),
        2.0 * u / (len(a) * len(b)) - 1.0,
        "rank_biserial",
        n=len(a),
        n_other=len(b),
        median_difference=float(np.median(a) - np.median(b)),
    )


def wilcoxon_paired(
    first: Sequence[float] | np.ndarray, second: Sequence[float] | np.ndarray
) -> TestResult:
    """Two-sided Wilcoxon signed-rank test of paired values.

    Args:
        first: Values of the first condition per item (e.g. gene).
        second: Values of the second condition (same items; pairs with a
            non-finite value are dropped; zero differences are dropped, the
            ``wilcox`` convention).

    Returns:
        The statistic, p, the matched-pairs rank-biserial correlation
        ``(R+ - R-) / (R+ + R-)`` (positive when ``first`` tends to be
        larger) and the median paired difference.
    """
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired values must align")
    keep = np.isfinite(a) & np.isfinite(b)
    difference = a[keep] - b[keep]
    nonzero = difference[difference != 0]
    if len(nonzero) < MIN_TEST_N:
        return TestResult(
            "wilcoxon_signed_rank",
            math.nan,
            math.nan,
            math.nan,
            "rank_biserial",
            n=int(keep.sum()),
            median_difference=(
                float(np.median(difference)) if len(difference) else math.nan
            ),
            note="fewer than 3 non-zero paired differences",
        )
    result = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return TestResult(
        "wilcoxon_signed_rank",
        float(result.statistic),
        float(result.pvalue),
        (positive - negative) / (positive + negative),
        "rank_biserial",
        n=int(keep.sum()),
        median_difference=float(np.median(difference)),
    )


# --------------------------------------------------------------------------
# Pseudobulk shares and the combined labelling


def pseudobulk_shares(
    counts: sparse.spmatrix | np.ndarray,
    labels: Sequence[object] | np.ndarray,
    classes: Sequence[str],
    *,
    include: np.ndarray | None = None,
    pseudocount: float = PSEUDOCOUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-class pseudobulk shares of each gene.

    ``p(g, c) = (sum of counts of g + pseudocount) / (sum of counts of all
    genes + pseudocount * n_genes)`` over the cells of class ``c``.

    Args:
        counts: ``(n_cells, n_genes)`` counts (the panel genes).
        labels: Class per cell (cells outside ``classes`` are left out).
        classes: Classes, in row order of the result.
        include: Cells to use (default: all).
        pseudocount: Added per gene (1).

    Returns:
        ``(shares, n_cells)``: ``(len(classes), n_genes)`` and cells per class.
    """
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    names = np.asarray(labels, dtype=object)
    if len(names) != matrix.shape[0]:
        raise ValueError("labels needs one entry per cell")
    keep = (
        np.ones(len(names), dtype=bool)
        if include is None
        else np.asarray(include, dtype=bool)
    )
    index = {str(name): position for position, name in enumerate(classes)}
    rows = np.array(
        [
            index.get(str(name), -1) if (name is not None and keep[i]) else -1
            for i, name in enumerate(names)
        ],
        dtype=np.int64,
    )
    used = rows >= 0
    indicator = sparse.csr_matrix(
        (np.ones(int(used.sum())), (rows[used], np.flatnonzero(used))),
        shape=(len(classes), matrix.shape[0]),
    )
    sums = np.asarray((indicator @ matrix).todense(), dtype=np.float64)
    n_genes = matrix.shape[1]
    shares = (sums + pseudocount) / (
        sums.sum(axis=1, keepdims=True) + pseudocount * n_genes
    )
    n_cells = np.bincount(rows[used], minlength=len(classes)).astype(np.int64)
    return shares, n_cells


def reference_broad_shares(
    profiles: pd.DataFrame,
    level: str,
    broad_of: Mapping[str, str],
    gene_ids: Sequence[str],
    classes: Sequence[str],
) -> pd.DataFrame:
    """Aggregate a bundle's node profiles to broad-class gene shares.

    Per broad class, the node ``mean_cpm`` profiles are averaged with the
    nodes' ``n_cells`` as weights (the class's mean CPM per cell) and
    renormalised over ``gene_ids``; a gene the profiles lack has share 0.

    Args:
        profiles: ``profiles.parquet`` (``level``, ``node_name``,
            ``n_cells``, ``gene_id``, ``mean_cpm``).
        level: The profile level (e.g. ``CCN202210140_SUPC``).
        broad_of: Node name to broad class (sinks map elsewhere).
        gene_ids: The genes (columns), in order.
        classes: The broad classes (rows), in order.

    Returns:
        ``classes x gene_ids`` shares (rows sum to 1; a class without a node
        is NaN).
    """
    rows = profiles[profiles["level"].astype(str) == level].copy()
    rows["broad"] = rows["node_name"].astype(str).map(dict(broad_of))
    rows = rows[rows["broad"].isin(list(classes))]
    rows["weighted"] = rows["mean_cpm"].astype(np.float64) * rows["n_cells"].astype(
        np.float64
    )
    numerator = rows.pivot_table(
        index="broad", columns="gene_id", values="weighted", aggfunc="sum"
    )
    cells = (
        rows.drop_duplicates(["broad", "node_name"])
        .groupby("broad")["n_cells"]
        .sum()
        .astype(np.float64)
    )
    mean = numerator.div(cells, axis=0)
    mean = mean.reindex(index=list(classes), columns=[str(g) for g in gene_ids])
    present = mean.index.isin(cells.index)
    mean.loc[present] = mean.loc[present].fillna(0.0)
    totals = mean.sum(axis=1)
    shares = mean.div(totals.where(totals > 0), axis=0)
    shares.loc[~present] = math.nan
    return shares


def combined_labels(
    hybrid_table_ids: Sequence[object],
    reseg_labels: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Label the proseg_hybrid table cells with the reseg cell of the same id.

    Args:
        hybrid_table_ids: The proseg_hybrid table cells' ids.
        reseg_labels: The reseg label table (``cell_id``, ``in_table`` and
            ``columns``), every object.
        columns: Label columns to carry over (e.g. ``ct_broad_status``).

    Returns:
        One row per proseg_hybrid table cell (index ``cell_id`` as str, in
        input order) with ``has_reseg_table_cell`` and the reseg columns;
        a cell without a reseg table cell has missing values (unlabelled).
    """
    table = reseg_labels[reseg_labels["in_table"].to_numpy(bool)].copy()
    table.index = pd.Index(table["cell_id"].astype(str), name="cell_id")
    if table.index.has_duplicates:
        raise ValueError("reseg label table has duplicate cell ids")
    ids = pd.Index([str(value) for value in hybrid_table_ids], name="cell_id")
    if ids.has_duplicates:
        raise ValueError("proseg_hybrid table ids must be unique")
    out = table.reindex(ids)[list(columns)].astype(object)
    out = out.where(pd.notna(out), None)
    out.insert(0, "has_reseg_table_cell", ids.isin(table.index))
    return out


# --------------------------------------------------------------------------
# Gene covariates


def _attribute(attributes: str, key: str) -> str | None:
    marker = f'{key} "'
    start = attributes.find(marker)
    if start < 0:
        return None
    start += len(marker)
    stop = attributes.find('"', start)
    return attributes[start:stop] if stop > start else None


def longest_transcript_lengths(
    gtf_lines: Iterable[str], gene_ids: Iterable[str]
) -> pd.DataFrame:
    """Return each gene's longest annotated transcript (sum of its exons).

    Only exons on the primary chromosomes (chr1-22, X, Y, M) count; gene
    ids are compared without their version suffix.

    Args:
        gtf_lines: Lines of a GENCODE / Ensembl GTF.
        gene_ids: Ensembl gene ids wanted (versions ignored).

    Returns:
        One row per wanted gene found: ``gene_id``, ``gene_name``,
        ``transcript_id``, ``transcript_length``.
    """
    wanted = {str(gene).split(".")[0] for gene in gene_ids}
    lengths: dict[tuple[str, str], int] = {}
    names: dict[str, str] = {}
    for line in gtf_lines:
        if not line or line.startswith("#"):
            continue
        fields = line.rstrip("\n").split("\t")
        if (
            len(fields) < 9
            or fields[2] != "exon"
            or fields[0] not in PRIMARY_CHROMOSOMES
        ):
            continue
        gene = _attribute(fields[8], "gene_id")
        if gene is None:
            continue
        gene = gene.split(".")[0]
        if gene not in wanted:
            continue
        transcript = _attribute(fields[8], "transcript_id")
        if transcript is None:
            continue
        key = (gene, transcript.split(".")[0])
        lengths[key] = lengths.get(key, 0) + int(fields[4]) - int(fields[3]) + 1
        name = _attribute(fields[8], "gene_name")
        if name is not None:
            names.setdefault(gene, name)
    best: dict[str, tuple[str, int]] = {}
    for (gene, transcript), length in sorted(lengths.items()):
        if gene not in best or length > best[gene][1]:
            best[gene] = (transcript, length)
    return pd.DataFrame(
        [
            {
                "gene_id": gene,
                "gene_name": names.get(gene),
                "transcript_id": transcript,
                "transcript_length": length,
            }
            for gene, (transcript, length) in sorted(best.items())
        ],
        columns=["gene_id", "gene_name", "transcript_id", "transcript_length"],
    )


def xenium_probe_counts(gene_panel: Path | Mapping[str, Any]) -> pd.DataFrame:
    """Return the probe count per gene of a Xenium ``gene_panel.json``.

    The count is the target's ``info.gene_coverage`` (the number of probes
    designed against the gene in 10x's panel file).

    Args:
        gene_panel: The file, or its parsed JSON.

    Returns:
        ``gene_id``, ``gene_name``, ``probe_count`` for the gene targets.
    """
    data = (
        json.loads(Path(gene_panel).read_text(encoding="utf-8"))
        if not isinstance(gene_panel, Mapping)
        else gene_panel
    )
    rows = []
    for target in data["payload"]["targets"]:
        kind = target.get("type") or {}
        if kind.get("descriptor") != "gene":
            continue
        info = target.get("info") or {}
        coverage = info.get("gene_coverage")
        rows.append(
            {
                "gene_id": str((kind.get("data") or {}).get("id", "")).split(".")[0],
                "gene_name": (kind.get("data") or {}).get("name"),
                "probe_count": math.nan if coverage is None else float(coverage),
            }
        )
    return pd.DataFrame(rows, columns=["gene_id", "gene_name", "probe_count"])


__all__ = [
    "BOOTSTRAP_SEED",
    "DISTANCE_BIN_EDGES_UM",
    "DISTANCE_BIN_LABELS",
    "N_GENE_BOOTSTRAP",
    "N_TILE_BOOTSTRAP",
    "PairedTileBootstrap",
    "TALLY_FIELDS",
    "THINNING_SEED",
    "TestResult",
    "ThinningResult",
    "benjamini_hochberg",
    "cell_generator",
    "cell_key",
    "combined_labels",
    "distance_bin_codes",
    "longest_transcript_lengths",
    "loss_shares",
    "loss_tally",
    "mann_whitney",
    "nearest_other_class_distance",
    "nucleus_holders",
    "points_in_polygons",
    "pseudobulk_shares",
    "reference_broad_shares",
    "spearman_bootstrap",
    "thin_to_targets",
    "tile_bootstrap_ratio",
    "transcript_cells",
    "wilcoxon_paired",
    "xenium_probe_counts",
]
