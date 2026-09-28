"""Soft composition, Jensen-Shannon distance and spatial block bootstrap (§5.5).

All cross-platform and cross-dataset composition statistics of the
annotation use one estimator (plan §5.5):

- **Soft vectors** (per table cell): the WHB bootstrap probability of the
  assigned supercluster plus its runner-ups', aggregated to the seven broad
  classes through the vocab; sinks, nodes outside the vocab and the residual
  ``1 - sum`` go to ``unallocated``, so every row sums to 1. RESOLVE writes
  them as ``soft_broad_<token>`` label-table columns (parquet only, §4.1);
  mouse uses the WMB classes (``soft_class_<token>``, M6).
- **Composition** = the sum of a section's rows; the **JSD** is the base-2
  Jensen-Shannon distance (as ``scipy.spatial.distance.jensenshannon`` and
  E1 ``real_jsd.csv``) of the renormalised seven-class vectors, with the
  unallocated share reported beside it.
- **95% CI**: a spatial block bootstrap over 500 µm tiles with 200
  replicates. Co-registered sections (the MERSCOPE ``*_aligned_nonrigid``
  coordinates are in the Xenium frame) share one grid and are resampled
  jointly (the registered H1 method); otherwise each section's tiles are
  resampled on their own (a sensitivity).
- **Also reported:** whole section vs the shared tissue mask, the
  depth-stratified composition (cells with >= 30 counts) and the
  confident-only composition (secondary; expected to differ more).

The same soft mass, on WHB supercluster labels per depth bin, is the
dataset composition RESOLVE reweights the resolvability tables to
(``dataset_type_composition``, §8.3).

The module imports numpy and pandas only (resolvability's
``DatasetComposition`` inside the function that builds one).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

import numpy as np
import pandas as pd

from merxen.annotation.schema import Columns, safe_token
from merxen.annotation.vocab import (
    HUMAN_BROAD_CLASSES,
    UNASSIGNED_LABEL,
    VocabTable,
    primary_vocab,
)

if TYPE_CHECKING:
    from merxen.annotation.resolvability import DatasetComposition

logger = logging.getLogger(__name__)

UNALLOCATED: Final = "unallocated"
COMPOSITION_COLUMNS: Final[tuple[str, ...]] = (*HUMAN_BROAD_CLASSES, UNALLOCATED)
TILE_UM: Final = 500.0
N_BOOTSTRAP: Final = 200
BOOTSTRAP_SEED: Final = 0
DEPTH_STRATUM_MIN_COUNTS: Final = 30
N_RUNNERS_UP: Final = 5
# Composition kinds reported for every section (§5.5): the soft estimator
# (primary), its depth stratum, the confident-only and the argmax variants.
COMPOSITION_KINDS: Final[tuple[str, ...]] = ("soft", "soft_ge30", "confident", "argmax")
WHOLE_SECTION: Final = "whole_section"
SHARED_MASK: Final = "shared_mask"


def _object_array(values: Sequence[object] | np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    missing = pd.isna(array)
    if missing.any():
        array = array.copy()
        array[missing] = None
    return array


# --------------------------------------------------------------------------
# Composition and JSD


def broad_class_index(
    names: Sequence[object] | np.ndarray | pd.Series,
    broad_of: Mapping[str, str],
) -> np.ndarray:
    """Return the ``COMPOSITION_COLUMNS`` index of each node's broad class.

    Args:
        names: Node names (``None`` / NaN for missing).
        broad_of: Node name to human broad class.

    Returns:
        Integer indices; nodes outside the seven broad classes (sinks,
        unknown or missing names) get the ``unallocated`` index.
    """
    position = {name: index for index, name in enumerate(HUMAN_BROAD_CLASSES)}
    unallocated = len(HUMAN_BROAD_CLASSES)
    lookup = {
        str(name): position.get(str(broad), unallocated)
        for name, broad in broad_of.items()
    }
    return np.array(
        [
            unallocated if value is None else lookup.get(str(value), unallocated)
            for value in _object_array(names)
        ],
        dtype=np.int64,
    )


def whb_broad_of(vocab: VocabTable | None = None) -> dict[str, str]:
    """Return the WHB supercluster name to broad class map (sinks unassigned)."""
    table = vocab or primary_vocab("human")
    return {
        name: (UNASSIGNED_LABEL if table.is_sink(name) else table.broad_class(name))
        for name in table.names
    }


def soft_broad_matrix(
    names: np.ndarray,
    probabilities: np.ndarray,
    *,
    broad_of: Mapping[str, str],
) -> np.ndarray:
    """Return per-cell soft broad-class mass (plan §5.5).

    Args:
        names: ``(n, k)`` node names: the assigned supercluster, then its
            runner-ups (``None`` / NaN where absent).
        probabilities: ``(n, k)`` bootstrap probabilities of those nodes.
        broad_of: Node name to broad class.

    Returns:
        ``(n, 8)`` mass over ``COMPOSITION_COLUMNS``; mass on nodes outside
        the seven classes and the residual ``1 - sum`` go to ``unallocated``,
        so every row sums to 1 (cells without a call are all unallocated).

    Raises:
        ValueError: If the shapes differ.
    """
    names = np.asarray(names, dtype=object)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if names.ndim != 2 or names.shape != probabilities.shape:
        raise ValueError("names and probabilities must be (n, k) arrays of one shape")
    n_cells, n_nodes = names.shape
    matrix = np.zeros((n_cells, len(COMPOSITION_COLUMNS)), dtype=np.float64)
    rows = np.arange(n_cells)
    n_classes = len(HUMAN_BROAD_CLASSES)
    for column in range(n_nodes):
        index = broad_class_index(names[:, column], broad_of)
        values = np.nan_to_num(probabilities[:, column], nan=0.0)
        values = np.clip(values, 0.0, 1.0)
        allocated = index < n_classes
        np.add.at(matrix, (rows[allocated], index[allocated]), values[allocated])
    total = matrix[:, :n_classes].sum(axis=1)
    over = total > 1.0
    if over.any():
        matrix[over, :n_classes] /= total[over, None]
        total = np.minimum(total, 1.0)
    matrix[:, n_classes] = 1.0 - total
    return matrix


def one_hot_broad_matrix(
    broad_names: Sequence[object] | np.ndarray | pd.Series,
    *,
    include: np.ndarray | None = None,
) -> np.ndarray:
    """Return per-cell one-hot broad-class rows (argmax or confident-only).

    Args:
        broad_names: Broad class per cell (anything outside the seven classes,
            incl. ``Mixed/Unknown`` and missing, is ``unallocated``).
        include: Cells that count; excluded cells get an all-zero row
            (confident-only composition).

    Returns:
        ``(n, 8)`` rows over ``COMPOSITION_COLUMNS``.
    """
    identity = {name: name for name in HUMAN_BROAD_CLASSES}
    index = broad_class_index(broad_names, identity)
    matrix = np.zeros((len(index), len(COMPOSITION_COLUMNS)), dtype=np.float64)
    matrix[np.arange(len(index)), index] = 1.0
    if include is not None:
        matrix[~np.asarray(include, dtype=bool)] = 0.0
    return matrix


def jensen_shannon_distance(
    p: np.ndarray, q: np.ndarray, *, columns: Sequence[int] | None = None
) -> float:
    """Return the base-2 Jensen-Shannon distance of two compositions.

    Both vectors are restricted to the seven broad classes (the first seven
    entries when ``unallocated`` is appended), or to ``columns``, and
    renormalised, as the plan's JSD (§5.5) and
    ``scipy.spatial.distance.jensenshannon(p, q, base=2)``.

    Args:
        p: Class masses (7 or 8 entries; rows of a matrix are vectorised).
        q: Class masses of the same length.
        columns: Entries to compare (e.g. the glial classes); default the
            first seven.

    Returns:
        The distance in [0, 1]; NaN if either vector has no mass there.
    """
    n_classes = len(HUMAN_BROAD_CLASSES)
    selected = (
        np.arange(n_classes) if columns is None else np.asarray(columns, dtype=int)
    )
    first = np.asarray(p, dtype=np.float64)[..., selected]
    second = np.asarray(q, dtype=np.float64)[..., selected]
    first_total = first.sum(axis=-1, keepdims=True)
    second_total = second.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        first = first / first_total
        second = second / second_total
        middle = 0.5 * (first + second)
        left = np.where(first > 0, first * np.log2(first / middle), 0.0)
        right = np.where(second > 0, second * np.log2(second / middle), 0.0)
    divergence = 0.5 * (left.sum(axis=-1) + right.sum(axis=-1))
    distance = np.sqrt(np.clip(divergence, 0.0, None))
    empty = (first_total[..., 0] <= 0) | (second_total[..., 0] <= 0)
    distance = np.where(empty, np.nan, distance)
    return float(distance) if np.ndim(distance) == 0 else distance  # type: ignore[return-value]


def composition_shares(matrix: np.ndarray) -> dict[str, float]:
    """Return the class shares of a per-cell matrix, over all its mass.

    Args:
        matrix: ``(n, 8)`` rows over ``COMPOSITION_COLUMNS``.

    Returns:
        Share per column (the seven classes and ``unallocated``) of the total
        mass; the seven-class shares are not renormalised.
    """
    totals = np.asarray(matrix, dtype=np.float64).sum(axis=0)
    mass = float(totals.sum())
    if mass <= 0:
        return {name: math.nan for name in COMPOSITION_COLUMNS}
    return {
        name: float(value / mass)
        for name, value in zip(COMPOSITION_COLUMNS, totals, strict=True)
    }


# --------------------------------------------------------------------------
# Spatial block bootstrap


def tile_codes(xy: np.ndarray, tile_um: float = TILE_UM) -> np.ndarray:
    """Return the square-tile id of each cell (plan §5.5: 500 µm tiles).

    Args:
        xy: ``(n, 2)`` coordinates in µm.
        tile_um: Tile edge.

    Returns:
        Dense integer tile ids ``0..n_tiles-1``; cells with non-finite
        coordinates get ``-1``.

    Raises:
        ValueError: If ``tile_um`` is not positive or ``xy`` is not ``(n, 2)``.
    """
    if tile_um <= 0:
        raise ValueError("tile_um must be positive")
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy must be an (n, 2) array")
    finite = np.isfinite(points).all(axis=1)
    codes = np.full(len(points), -1, dtype=np.int64)
    if not finite.any():
        return codes
    cells = np.floor(points[finite] / tile_um).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    codes[finite] = inverse.reshape(-1)
    return codes


def shared_tile_codes(
    xy_a: np.ndarray, xy_b: np.ndarray, tile_um: float = TILE_UM
) -> tuple[np.ndarray, np.ndarray, int]:
    """Tile two co-registered sections on one grid (plan §5.5).

    Both sections must be in one frame (the MERSCOPE ``*_aligned_nonrigid``
    coordinates are in the Xenium frame of ``registration_summary.json``), so
    tile ``t`` of section A and tile ``t`` of section B cover the same
    tissue, and a bootstrap can resample tile locations for both at once
    (``block_bootstrap_jsd(resampling="joint")``).

    Args:
        xy_a: ``(n_a, 2)`` coordinates of section A in µm.
        xy_b: ``(n_b, 2)`` coordinates of section B, same frame.
        tile_um: Tile edge.

    Returns:
        ``(codes_a, codes_b, n_tiles)``: tile ids on the shared grid
        (``-1`` for non-finite coordinates) and the number of tiles.
    """
    points_a = np.asarray(xy_a, dtype=np.float64)
    points_b = np.asarray(xy_b, dtype=np.float64)
    if points_a.ndim != 2 or points_b.ndim != 2:
        raise ValueError("xy must be (n, 2) arrays")
    codes = tile_codes(np.vstack([points_a, points_b]), tile_um)
    n_tiles = int(codes.max()) + 1 if len(codes) and codes.max() >= 0 else 0
    return codes[: len(points_a)], codes[len(points_a) :], n_tiles


def tile_sums(
    matrix: np.ndarray, codes: np.ndarray, n_tiles: int | None = None
) -> np.ndarray:
    """Sum per-cell composition rows by tile.

    Args:
        matrix: ``(n, k)`` per-cell rows.
        codes: Tile id per cell (``tile_codes``); ``-1`` cells are dropped.
        n_tiles: Rows of the result (``shared_tile_codes``: the shared grid's
            tile count, so both sections' tables have the same rows);
            default: the largest id + 1.

    Returns:
        ``(n_tiles, k)`` sums.
    """
    codes = np.asarray(codes, dtype=np.int64)
    values = np.asarray(matrix, dtype=np.float64)
    keep = codes >= 0
    if n_tiles is None:
        n_tiles = int(codes[keep].max()) + 1 if keep.any() else 0
    sums = np.zeros((n_tiles, values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        sums[:, column] = np.bincount(
            codes[keep], weights=values[keep, column], minlength=n_tiles
        )
    return sums


Resampling = Literal["joint", "independent"]


@dataclass(frozen=True)
class BootstrapJsd:
    """A Jensen-Shannon distance with its spatial block-bootstrap CI.

    Attributes:
        jsd: Point estimate on all cells.
        ci_low: 2.5th percentile of the replicates.
        ci_high: 97.5th percentile of the replicates.
        n_reps: Replicates.
        n_tiles_a: Non-empty tiles of the first section.
        n_tiles_b: Non-empty tiles of the second section.
        replicates: Replicate distances.
        resampling: ``joint`` (one draw of shared tile locations for both
            sections) or ``independent`` (each section's tiles on its own).
        n_locations: Tile locations resampled (``joint``: tiles non-empty in
            either section; ``independent``: ``n_tiles_a + n_tiles_b``).
    """

    jsd: float
    ci_low: float
    ci_high: float
    n_reps: int
    n_tiles_a: int
    n_tiles_b: int
    replicates: np.ndarray
    resampling: Resampling = "joint"
    n_locations: int = 0


def _bootstrap_weights(
    rng: np.random.Generator, n_items: int, n_reps: int
) -> np.ndarray:
    return rng.multinomial(n_items, np.full(n_items, 1 / n_items), n_reps)


def block_bootstrap_jsd(
    tiles_a: np.ndarray,
    tiles_b: np.ndarray,
    *,
    n_reps: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    columns: Sequence[int] | None = None,
    resampling: Resampling = "joint",
) -> BootstrapJsd:
    """Bootstrap the JSD of two sections by resampling 500 µm tiles (§5.5).

    - ``joint`` (the registered H1 method): the two sections are tiled on one
      grid in the shared frame (``shared_tile_codes``), so row ``t`` of both
      tables is the same tissue. Each replicate draws the tile locations
      (non-empty in either section) with replacement and applies the same
      multinomial weights to both sections, so anatomy stays matched
      between them.
    - ``independent`` (a sensitivity, or sections without a shared frame):
      each section's non-empty tiles are drawn on their own; this mismatches
      anatomy between adjacent sections and widens the interval.

    Args:
        tiles_a: ``(n_tiles, k)`` tile sums of the first section.
        tiles_b: ``(n_tiles, k)`` tile sums of the second section (same
            rows as ``tiles_a`` for ``joint``).
        n_reps: Replicates (200 in the plan).
        seed: RNG seed.
        columns: Classes compared (``jensen_shannon_distance``).
        resampling: ``joint`` or ``independent``.

    Returns:
        Point estimate, percentile CI and replicates.

    Raises:
        ValueError: If a section has no tile, or ``joint`` tables differ in
            rows.
    """
    first = np.asarray(tiles_a, dtype=np.float64)
    second = np.asarray(tiles_b, dtype=np.float64)
    nonempty_a = first.sum(axis=1) > 0
    nonempty_b = second.sum(axis=1) > 0
    if not nonempty_a.any() or not nonempty_b.any():
        raise ValueError("each section needs at least one non-empty tile")
    point = jensen_shannon_distance(
        first.sum(axis=0), second.sum(axis=0), columns=columns
    )
    rng = np.random.default_rng(seed)
    if resampling == "joint":
        if first.shape[0] != second.shape[0]:
            raise ValueError(
                "joint resampling needs both sections on one tile grid "
                "(shared_tile_codes)"
            )
        keep = nonempty_a | nonempty_b
        first, second = first[keep], second[keep]
        weights = _bootstrap_weights(rng, len(first), n_reps)
        replicate_a, replicate_b = weights @ first, weights @ second
        n_locations = len(first)
    elif resampling == "independent":
        first, second = first[nonempty_a], second[nonempty_b]
        replicate_a = _bootstrap_weights(rng, len(first), n_reps) @ first
        replicate_b = _bootstrap_weights(rng, len(second), n_reps) @ second
        n_locations = len(first) + len(second)
    else:
        raise ValueError(f"unknown resampling {resampling!r}")
    replicates = np.asarray(
        jensen_shannon_distance(replicate_a, replicate_b, columns=columns),
        dtype=np.float64,
    ).reshape(-1)
    low, high = np.nanpercentile(replicates, [2.5, 97.5])
    return BootstrapJsd(
        jsd=float(point),
        ci_low=float(low),
        ci_high=float(high),
        n_reps=int(n_reps),
        n_tiles_a=int(nonempty_a.sum()),
        n_tiles_b=int(nonempty_b.sum()),
        replicates=replicates,
        resampling=resampling,
        n_locations=int(n_locations),
    )


# --------------------------------------------------------------------------
# Production: soft vectors, the reweighting composition and section statistics


def level_candidates(
    frame: pd.DataFrame,
    *,
    field_name: Literal["name", "assignment"] = "name",
    n_runners_up: int = N_RUNNERS_UP,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the assigned node and its runner-ups with their probabilities.

    Args:
        frame: One level of a tidy MMC table (``level_frame``), one row per
            cell.
        field_name: ``name`` (node names, the vocab key of the soft vectors)
            or ``assignment`` (node labels, the key of the resolvability
            tables' ``truth_leaf``).
        n_runners_up: Runner-ups to include (those present in the frame).

    Returns:
        ``(names, probabilities)``, both ``(n, k)``: column 0 is the assigned
        node with its bootstrap probability, then runner-ups 1..k-1 with
        theirs (``None`` / NaN where absent).
    """
    names = [_object_array(frame[field_name])]
    probabilities = [frame["bp"].to_numpy(np.float64)]
    for rank in range(1, n_runners_up + 1):
        name_column = f"runner_up_{rank}_{field_name}"
        probability_column = f"runner_up_{rank}_probability"
        if name_column not in frame.columns or probability_column not in frame.columns:
            continue
        names.append(_object_array(frame[name_column]))
        probabilities.append(
            pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(
                np.float64
            )
        )
    return np.column_stack(names), np.column_stack(probabilities)


def soft_broad_from_level(
    frame: pd.DataFrame,
    *,
    broad_of: Mapping[str, str] | None = None,
    n_runners_up: int = N_RUNNERS_UP,
) -> np.ndarray:
    """Return the soft broad rows of a WHB supercluster level (§5.5).

    Args:
        frame: The supercluster level of a WHB tidy table (``level_frame``).
        broad_of: Supercluster name to broad class (default: the packaged
            WHB vocab, sinks unassigned).
        n_runners_up: Runner-ups to aggregate.

    Returns:
        ``(n, 8)`` rows over ``COMPOSITION_COLUMNS``, each summing to 1.
    """
    names, probabilities = level_candidates(frame, n_runners_up=n_runners_up)
    return soft_broad_matrix(names, probabilities, broad_of=broad_of or whb_broad_of())


def soft_broad_columns(
    matrix: np.ndarray, rows: np.ndarray, n_objects: int
) -> dict[str, np.ndarray]:
    """Return the ``soft_broad_*`` label-table columns (§4.1).

    Args:
        matrix: ``(n_rows, 8)`` soft rows (``soft_broad_matrix``).
        rows: Object position of each row (the table cells).
        n_objects: Objects of the label table.

    Returns:
        ``soft_broad_<token>`` per broad class and ``soft_broad_unallocated``,
        float32, NaN for objects without a row (outside the table).
    """
    values = np.asarray(matrix, dtype=np.float64)
    positions = np.asarray(rows, dtype=np.int64)
    if values.shape != (len(positions), len(COMPOSITION_COLUMNS)):
        raise ValueError("matrix must have one row per position and 8 columns")
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(COMPOSITION_COLUMNS):
        column = np.full(n_objects, np.nan, dtype=np.float32)
        column[positions] = np.clip(values[:, index], 0.0, 1.0).astype(np.float32)
        key = (
            Columns.SOFT_BROAD_UNALLOCATED
            if name == UNALLOCATED
            else f"{Columns.SOFT_BROAD_PREFIX}{safe_token(name)}"
        )
        columns[key] = column
    return columns


def dataset_type_composition(
    frame: pd.DataFrame,
    counts: np.ndarray | Sequence[float],
    grid: Sequence[int],
    *,
    min_bin_mass: float = 50.0,
    n_runners_up: int = N_RUNNERS_UP,
) -> DatasetComposition:
    """Return the dataset composition the resolvability tables are reweighted to.

    The soft mass of §5.5 on the primary leaf's node **labels** (the
    resolvability tables key truth types by label, e.g. ``CS202210140_476``):
    each table cell contributes the bootstrap probability of its assigned
    node and of each runner-up, overall and per depth bin of the bundle's
    grid (``resolvability.DatasetComposition.from_cells``; §8.3 composition
    reweighting, M3b review).

    Args:
        frame: The primary leaf level of the table cells (``level_frame``).
        counts: Total counts per row of ``frame``.
        grid: The bundle's depth grid.
        min_bin_mass: Bins with less mass follow the overall composition
            (``composition_min_bin_cells``).
        n_runners_up: Runner-ups to include.

    Returns:
        The composition.
    """
    from merxen.annotation.resolvability import DatasetComposition

    labels, probabilities = level_candidates(
        frame, field_name="assignment", n_runners_up=n_runners_up
    )
    depth = np.asarray(counts, dtype=np.float64)
    if len(depth) != len(labels):
        raise ValueError("counts must have one value per row of the level frame")
    n_columns = labels.shape[1]
    return DatasetComposition.from_cells(
        labels.reshape(-1).tolist(),
        np.clip(np.nan_to_num(probabilities.reshape(-1), nan=0.0), 0.0, 1.0),
        np.repeat(depth, n_columns),
        list(grid),
        min_bin_mass=min_bin_mass,
    )


@dataclass(frozen=True)
class SectionComposition:
    """One section's per-cell composition rows (table cells; §5.5).

    Attributes:
        sample_id: Sample id.
        platform: ``MERSCOPE`` or ``XENIUM``.
        matrices: ``(n, 8)`` rows per kind (``COMPOSITION_KINDS``).
        xy: ``(n, 2)`` coordinates in µm, if known.
        aligned_frame: Whether ``xy`` is in the pair's shared (fixed) frame.
    """

    sample_id: str
    platform: str
    matrices: Mapping[str, np.ndarray]
    xy: np.ndarray | None = None
    aligned_frame: bool = False

    @property
    def n_cells(self) -> int:
        """Return the number of table cells."""
        first = next(iter(self.matrices.values()))
        return int(first.shape[0])

    def shares(self, kind: str, keep: np.ndarray | None = None) -> dict[str, Any]:
        """Return one kind's composition over the kept cells.

        Args:
            kind: A ``COMPOSITION_KINDS`` entry.
            keep: Cells to include (default: all).

        Returns:
            ``n_cells``, ``mass``, the share of every column over all mass
            (``share_<token>``, the seven classes and ``unallocated``) and the
            renormalised seven-class shares (``share7_<token>``).
        """
        matrix = self.matrices[kind]
        selected = matrix if keep is None else matrix[np.asarray(keep, dtype=bool)]
        shares = composition_shares(selected)
        classes = selected[:, : len(HUMAN_BROAD_CLASSES)].sum(axis=0)
        total = float(classes.sum())
        record: dict[str, Any] = {
            "n_cells": int(len(selected)),
            "mass": float(selected.sum()),
        }
        for name in COMPOSITION_COLUMNS:
            record[f"share_{safe_token(name)}"] = _finite(shares[name])
        for name, value in zip(HUMAN_BROAD_CLASSES, classes, strict=True):
            record[f"share7_{safe_token(name)}"] = (
                _finite(float(value / total)) if total > 0 else None
            )
        return record


def _finite(value: float) -> float | None:
    return None if not math.isfinite(value) else round(float(value), 6)


def section_composition(
    sample_id: str,
    platform: str,
    soft: np.ndarray,
    *,
    total_counts: np.ndarray | Sequence[float],
    argmax_broad: Sequence[object] | np.ndarray,
    confident_broad: Sequence[object] | np.ndarray,
    confident: np.ndarray | Sequence[bool],
    xy: np.ndarray | None = None,
    aligned_frame: bool = False,
    depth_min_counts: int = DEPTH_STRATUM_MIN_COUNTS,
) -> SectionComposition:
    """Return a section's composition rows of every kind (§5.5).

    Args:
        sample_id: Sample id.
        platform: Platform.
        soft: ``(n, 8)`` soft rows of the table cells.
        total_counts: Their counts.
        argmax_broad: Broad class of each cell's assigned supercluster.
        confident_broad: ``ct_broad_name`` of each cell.
        confident: Whether each cell's broad level is confident.
        xy: ``(n, 2)`` coordinates, if known.
        aligned_frame: Whether ``xy`` is in the pair's shared frame.
        depth_min_counts: Depth of the stratified kind (30 counts).

    Returns:
        The section composition.
    """
    rows = np.asarray(soft, dtype=np.float64)
    counts = np.asarray(total_counts, dtype=np.float64)
    if len(counts) != len(rows):
        raise ValueError("total_counts must have one value per soft row")
    matrices = {
        "soft": rows,
        "soft_ge30": np.where((counts >= depth_min_counts)[:, None], rows, 0.0),
        "confident": one_hot_broad_matrix(
            confident_broad, include=np.asarray(confident, dtype=bool)
        ),
        "argmax": one_hot_broad_matrix(argmax_broad),
    }
    points = None if xy is None else np.asarray(xy, dtype=np.float64)[:, :2]
    if points is not None and len(points) != len(rows):
        raise ValueError("xy must have one row per cell")
    return SectionComposition(
        sample_id=sample_id,
        platform=str(platform).upper(),
        matrices=matrices,
        xy=points,
        aligned_frame=bool(aligned_frame) and points is not None,
    )


class TissueMask(Protocol):
    """What the pair statistics need of ``panel.SharedTissueMask``."""

    fixed_platform: str

    def contains(self, xy: np.ndarray) -> np.ndarray:
        """Return whether points fall inside the shared tissue."""
        ...


@dataclass(frozen=True)
class JsdRecord:
    """One pair JSD with its block-bootstrap CI (§5.5; H1).

    Attributes:
        kind: Composition kind.
        region: ``whole_section`` or ``shared_mask``.
        jsd: Point estimate.
        ci_low: 2.5th percentile (``resampling``), ``None`` without tiles.
        ci_high: 97.5th percentile.
        resampling: ``joint`` (shared grid) or ``independent``; ``None``
            when no bootstrap ran.
        ci_low_independent: Independent-resampling sensitivity.
        ci_high_independent: Independent-resampling sensitivity.
        n_reps: Replicates.
        tile_um: Tile edge.
        n_tile_locations: Tile locations resampled.
        n_tiles_a: Non-empty tiles of the first section.
        n_tiles_b: Non-empty tiles of the second section.
        n_cells_a: Cells of the first section in the region.
        n_cells_b: Cells of the second section in the region.
        unallocated_a: Unallocated share of the first section.
        unallocated_b: Unallocated share of the second section.
        note: Why the bootstrap or the region is limited, if it is.
    """

    kind: str
    region: str
    jsd: float | None
    ci_low: float | None
    ci_high: float | None
    resampling: str | None
    ci_low_independent: float | None
    ci_high_independent: float | None
    n_reps: int
    tile_um: float
    n_tile_locations: int
    n_tiles_a: int
    n_tiles_b: int
    n_cells_a: int
    n_cells_b: int
    unallocated_a: float | None
    unallocated_b: float | None
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return the record as JSON (finite floats, ``None`` otherwise)."""
        record: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, float):
                record[key] = _finite(value)
            else:
                record[key] = value
        return record


def shared_mask_regions(
    first: SectionComposition,
    second: SectionComposition,
    mask: TissueMask | None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str]:
    """Return the pair's regions: the whole sections and, if usable, the mask.

    The shared tissue mask is in the fixed platform's frame, so it applies
    only when both sections' coordinates are in that frame.

    Args:
        first: First section.
        second: Second section.
        mask: The pair's shared tissue mask, if any.

    Returns:
        ``(regions, note)``: region name to the kept cells of each section,
        and why the mask was not applied (``"applied"`` when it was).
    """
    regions = {
        WHOLE_SECTION: (
            np.ones(first.n_cells, dtype=bool),
            np.ones(second.n_cells, dtype=bool),
        )
    }
    if mask is None:
        return regions, "no shared tissue mask"
    if first.xy is None or second.xy is None:
        return regions, "no coordinates"
    if not (first.aligned_frame and second.aligned_frame):
        return regions, "coordinates not in the mask's fixed frame"
    regions[SHARED_MASK] = (mask.contains(first.xy), mask.contains(second.xy))
    return regions, "applied"


def pair_jsd(
    first: SectionComposition,
    second: SectionComposition,
    *,
    mask: TissueMask | None = None,
    kinds: Sequence[str] = COMPOSITION_KINDS,
    tile_um: float = TILE_UM,
    n_reps: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[list[JsdRecord], str]:
    """Return the pair's JSD with block-bootstrap CIs per kind and region (§5.5).

    Args:
        first: First section (MERSCOPE).
        second: Second section (XENIUM).
        mask: The pair's shared tissue mask.
        kinds: Composition kinds.
        tile_um: Tile edge (500 µm).
        n_reps: Replicates (200).
        seed: Bootstrap seed.

    Returns:
        ``(records, mask_note)``.
    """
    regions, note = shared_mask_regions(first, second, mask)
    records: list[JsdRecord] = []
    for region, (keep_a, keep_b) in regions.items():
        tiles: tuple[np.ndarray, np.ndarray, int | None, bool] | None = None
        region_note = "" if region == WHOLE_SECTION else note
        if first.xy is not None and second.xy is not None:
            if first.aligned_frame and second.aligned_frame:
                grid_a, grid_b, grid_tiles = shared_tile_codes(
                    first.xy[keep_a], second.xy[keep_b], tile_um
                )
                tiles = (grid_a, grid_b, grid_tiles, True)
            else:
                tiles = (
                    tile_codes(first.xy[keep_a], tile_um),
                    tile_codes(second.xy[keep_b], tile_um),
                    None,
                    False,
                )
                region_note = region_note or "no shared frame: independent resampling"
        else:
            region_note = region_note or "no coordinates: no bootstrap"
        for kind in kinds:
            matrix_a = first.matrices[kind][keep_a]
            matrix_b = second.matrices[kind][keep_b]
            shares_a = composition_shares(matrix_a)
            shares_b = composition_shares(matrix_b)
            point = jensen_shannon_distance(matrix_a.sum(axis=0), matrix_b.sum(axis=0))
            joint = independent = None
            n_classes = len(HUMAN_BROAD_CLASSES)
            has_mass = (
                matrix_a[:, :n_classes].sum() > 0 and matrix_b[:, :n_classes].sum() > 0
            )
            if tiles is not None and has_mass:
                codes_a, codes_b, n_tiles, is_joint = tiles
                sums_a = tile_sums(matrix_a, codes_a, n_tiles)
                sums_b = tile_sums(matrix_b, codes_b, n_tiles)
                independent = block_bootstrap_jsd(
                    sums_a, sums_b, n_reps=n_reps, seed=seed, resampling="independent"
                )
                joint = (
                    block_bootstrap_jsd(sums_a, sums_b, n_reps=n_reps, seed=seed)
                    if is_joint
                    else independent
                )
            records.append(
                JsdRecord(
                    kind=kind,
                    region=region,
                    jsd=_finite(float(point)),
                    ci_low=None if joint is None else _finite(joint.ci_low),
                    ci_high=None if joint is None else _finite(joint.ci_high),
                    resampling=None if joint is None else joint.resampling,
                    ci_low_independent=(
                        None if independent is None else _finite(independent.ci_low)
                    ),
                    ci_high_independent=(
                        None if independent is None else _finite(independent.ci_high)
                    ),
                    n_reps=0 if joint is None else joint.n_reps,
                    tile_um=float(tile_um),
                    n_tile_locations=0 if joint is None else joint.n_locations,
                    n_tiles_a=0 if joint is None else joint.n_tiles_a,
                    n_tiles_b=0 if joint is None else joint.n_tiles_b,
                    n_cells_a=int(np.count_nonzero(keep_a)),
                    n_cells_b=int(np.count_nonzero(keep_b)),
                    unallocated_a=_finite(shares_a[UNALLOCATED]),
                    unallocated_b=_finite(shares_b[UNALLOCATED]),
                    note=region_note,
                )
            )
    return records, note


def section_shares(
    section: SectionComposition,
    *,
    regions: Mapping[str, np.ndarray] | None = None,
    kinds: Sequence[str] = COMPOSITION_KINDS,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return a section's composition per region and kind (provenance, summary).

    Args:
        section: The section.
        regions: Region name to kept cells (default: the whole section).
        kinds: Composition kinds.

    Returns:
        ``{region: {kind: SectionComposition.shares}}``.
    """
    selected = dict(regions or {WHOLE_SECTION: np.ones(section.n_cells, dtype=bool)})
    return {
        region: {kind: section.shares(kind, keep) for kind in kinds}
        for region, keep in selected.items()
    }


__all__ = [
    "BOOTSTRAP_SEED",
    "COMPOSITION_COLUMNS",
    "COMPOSITION_KINDS",
    "DEPTH_STRATUM_MIN_COUNTS",
    "N_BOOTSTRAP",
    "SHARED_MASK",
    "TILE_UM",
    "UNALLOCATED",
    "WHOLE_SECTION",
    "BootstrapJsd",
    "JsdRecord",
    "Resampling",
    "SectionComposition",
    "TissueMask",
    "block_bootstrap_jsd",
    "broad_class_index",
    "composition_shares",
    "dataset_type_composition",
    "jensen_shannon_distance",
    "level_candidates",
    "one_hot_broad_matrix",
    "pair_jsd",
    "section_composition",
    "section_shares",
    "shared_mask_regions",
    "shared_tile_codes",
    "soft_broad_columns",
    "soft_broad_from_level",
    "soft_broad_matrix",
    "tile_codes",
    "tile_sums",
    "whb_broad_of",
]
