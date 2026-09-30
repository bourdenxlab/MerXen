"""Metric functions of the annotation report (plan §9; M7).

Every number the report plots or writes to ``acceptance_metrics.json`` that
is not copied from an upstream record (RESOLVE's summary, the provenance,
the bundle) is computed here, on plain arrays, so the definitions are unit
tested on synthetic inputs and mutation-checked:

- **Composition with spatial block-bootstrap CIs** (§5.5, item 2): shares of
  a per-cell mass matrix over all its mass and renormalised over the classes,
  with a percentile interval from resampling 500 µm tiles
  (``block_bootstrap_shares``); soft mass at any level from an engine's
  assignment and its runner-ups (``soft_level_matrix``).
- **Aligned-bin concordance** (item 7): per-type density correlation of two
  co-registered sections in 200 µm bins of the fixed (Xenium) frame
  (``aligned_bin_density_correlation``).
- **Cortical depth** (item 9; H12): medians with block-bootstrap CIs
  (``median_block_ci``), the strict ordering with non-overlapping CIs
  (``depth_ordering``), its replication across platforms
  (``depth_replication``), a class-share contrast between two regions
  (``share_contrast``; oligodendrocytes WM > GM) and depth profiles.
- **Platform factors** (item 7): per-gene log2 ratios within a label
  (``platform_gene_log2_ratios``) and pseudobulk correlations.
- Method agreement by depth quantile, 2D histograms, tile maps and the
  self-thinning eligibility of item 1.

The module imports numpy and pandas only, so it loads in any environment
that can read a label table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

TILE_UM: Final = 500.0
DENSITY_BIN_UM: Final = 200.0
N_BOOTSTRAP: Final = 200
BOOTSTRAP_SEED: Final = 0
CI_PERCENTILES: Final = (2.5, 97.5)
UNALLOCATED: Final = "unallocated"
SELF_THINNING_MIN_COUNTS: Final = 200
SELF_THINNING_MIN_CELLS: Final = 500
# A self-thinning diagnostic whose deep cells are dominated by one class says
# little about the others (plan §9 item 1: P1212_M's 509 such cells are 89%
# Vascular -> unreliable). The report marks it unreliable above this share.
SELF_THINNING_MAX_DOMINANT_SHARE: Final = 0.5
GENE_RATIO_PSEUDOCOUNT: Final = 0.001


# --------------------------------------------------------------------------
# Tiles and weights


def grid_codes(xy: np.ndarray, bin_um: float) -> np.ndarray:
    """Return a dense square-bin id per point (``-1`` for non-finite points).

    Args:
        xy: ``(n, 2)`` coordinates in µm.
        bin_um: Bin edge in µm.

    Returns:
        Integer ids ``0..n_bins-1`` (row-major over the occupied bins).

    Raises:
        ValueError: If ``bin_um`` is not positive or ``xy`` is not ``(n, 2)``.
    """
    if bin_um <= 0:
        raise ValueError("bin_um must be positive")
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy must be an (n, 2) array")
    codes = np.full(len(points), -1, dtype=np.int64)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return codes
    cells = np.floor(points[finite] / bin_um).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    codes[finite] = inverse.reshape(-1)
    return codes


def _tile_weights(rng: np.random.Generator, n_tiles: int, n_reps: int) -> np.ndarray:
    """Return ``(n_reps, n_tiles)`` multinomial resampling multiplicities."""
    return rng.multinomial(n_tiles, np.full(n_tiles, 1.0 / n_tiles), size=n_reps)


def _dense_codes(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Return codes renumbered ``0..k-1`` over the used tiles (``-1`` kept)."""
    values = np.asarray(codes, dtype=np.int64)
    dense = np.full(len(values), -1, dtype=np.int64)
    keep = values >= 0
    if not keep.any():
        return dense, 0
    unique, inverse = np.unique(values[keep], return_inverse=True)
    dense[keep] = inverse
    return dense, len(unique)


def _percentile_ci(replicates: np.ndarray) -> tuple[float, float]:
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return math.nan, math.nan
    low, high = np.percentile(finite, CI_PERCENTILES)
    return float(low), float(high)


# --------------------------------------------------------------------------
# Composition


def soft_level_matrix(
    names: Sequence[object] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    runner_up_names: Sequence[Sequence[object] | np.ndarray] = (),
    runner_up_probabilities: Sequence[Sequence[float] | np.ndarray] = (),
    *,
    categories: Sequence[str],
    mapping: Mapping[str, str] | None = None,
) -> np.ndarray:
    """Return per-cell soft mass over categories, residual to ``unallocated``.

    The §5.5 soft estimator at any level: a cell's mass is the bootstrap
    probability of its assignment plus its runner-ups', each added to the
    category it maps to (``mapping``, default the name itself); names outside
    ``categories`` and the residual ``1 - sum`` go to the last column. A row
    whose probabilities sum above 1 (float rounding) is rescaled to 1.

    Args:
        names: Assigned node name per cell (``None``/NaN: no assignment).
        probabilities: Its bootstrap probability.
        runner_up_names: Runner-up names, one sequence per rank.
        runner_up_probabilities: Their probabilities, one sequence per rank.
        categories: Output categories, in column order.
        mapping: Node name to category (e.g. supercluster -> broad class).

    Returns:
        ``(n, len(categories) + 1)`` rows summing to 1 (``unallocated`` last).

    Raises:
        ValueError: If the runner-up sequences differ in number or length.
    """
    if len(runner_up_names) != len(runner_up_probabilities):
        raise ValueError("runner-up names and probabilities differ in rank count")
    index = {name: position for position, name in enumerate(categories)}
    lookup = dict(index)
    for source, target in (mapping or {}).items():
        lookup[str(source)] = index.get(str(target), -1)
    first = np.asarray(names, dtype=object)
    n_cells = len(first)
    matrix = np.zeros((n_cells, len(categories) + 1), dtype=np.float64)
    columns = [(first, np.asarray(probabilities, dtype=np.float64))]
    for rank_names, rank_probabilities in zip(
        runner_up_names, runner_up_probabilities, strict=True
    ):
        columns.append(
            (
                np.asarray(rank_names, dtype=object),
                np.asarray(rank_probabilities, dtype=np.float64),
            )
        )
    for rank_names, rank_probabilities in columns:
        if len(rank_names) != n_cells or len(rank_probabilities) != n_cells:
            raise ValueError("every rank needs one value per cell")
        values = np.where(np.isfinite(rank_probabilities), rank_probabilities, 0.0)
        values = np.clip(values, 0.0, None)
        targets = _category_index(rank_names, lookup)
        known = targets >= 0
        np.add.at(matrix, (np.flatnonzero(known), targets[known]), values[known])
    allocated = matrix[:, :-1].sum(axis=1)
    over = allocated > 1.0
    if over.any():
        matrix[over, :-1] /= allocated[over, None]
        allocated = np.where(over, 1.0, allocated)
    matrix[:, -1] = np.clip(1.0 - allocated, 0.0, None)
    return matrix


def _category_index(
    names: Sequence[object] | np.ndarray, lookup: Mapping[str, int]
) -> np.ndarray:
    """Return each name's column in ``lookup`` (``-1`` if absent or missing)."""
    series = pd.Series(np.asarray(names, dtype=object), dtype=object)
    mapped = series.map(
        lambda value: lookup.get(str(value), -1) if _is_label(value) else -1
    )
    return np.asarray(mapped.to_numpy(dtype=np.int64))


def _is_label(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value) not in ("", "nan", "None", "<NA>")


def one_hot_matrix(
    labels: Sequence[object] | np.ndarray,
    categories: Sequence[str],
    *,
    include: np.ndarray | None = None,
) -> np.ndarray:
    """Return one-hot rows over categories with an ``unallocated`` last column.

    Args:
        labels: Label per cell.
        categories: Output categories.
        include: Cells that carry mass (others get an all-zero row).

    Returns:
        ``(n, len(categories) + 1)``; labels outside ``categories`` go to the
        last column.
    """
    index = {name: position for position, name in enumerate(categories)}
    values = np.asarray(labels, dtype=object)
    matrix = np.zeros((len(values), len(categories) + 1), dtype=np.float64)
    targets = _category_index(values, index)
    targets[targets < 0] = len(categories)
    matrix[np.arange(len(values)), targets] = 1.0
    if include is not None:
        matrix[~np.asarray(include, dtype=bool)] = 0.0
    return matrix


@dataclass(frozen=True)
class ShareRecord:
    """One category's share of a section with its block-bootstrap CI.

    Attributes:
        category: Category (the last one is ``unallocated``).
        share: Share of all mass.
        ci_low: 2.5th percentile of the replicate shares.
        ci_high: 97.5th percentile.
        share_renormalised: Share of the classes' mass (``unallocated``
            excluded; NaN for ``unallocated`` itself).
        renormalised_ci_low: Its 2.5th percentile.
        renormalised_ci_high: Its 97.5th percentile.
        mass: The category's mass.
    """

    category: str
    share: float
    ci_low: float
    ci_high: float
    share_renormalised: float
    renormalised_ci_low: float
    renormalised_ci_high: float
    mass: float


@dataclass(frozen=True)
class ShareResult:
    """A section's composition with CIs (plan §5.5; report item 2).

    Attributes:
        records: One record per category.
        n_cells: Cells counted.
        n_tiles: Non-empty tiles resampled (0 without coordinates).
        n_reps: Replicates (0 without a bootstrap).
    """

    records: tuple[ShareRecord, ...]
    n_cells: int
    n_tiles: int
    n_reps: int

    def frame(self) -> pd.DataFrame:
        """Return the records as a data frame."""
        return pd.DataFrame([record.__dict__ for record in self.records])

    def share(self, category: str) -> float:
        """Return one category's share of all mass (NaN if absent)."""
        for record in self.records:
            if record.category == category:
                return record.share
        return math.nan


def block_bootstrap_shares(
    matrix: np.ndarray,
    categories: Sequence[str],
    codes: np.ndarray | None,
    *,
    n_reps: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    keep: np.ndarray | None = None,
) -> ShareResult:
    """Return a section's composition with a spatial block-bootstrap CI.

    Shares are column sums over all mass; the renormalised shares drop the
    last (``unallocated``) column, as the JSD's seven-class vectors do. The
    interval resamples the section's non-empty tiles with replacement
    (percentile, ``n_reps`` replicates, seed ``seed``), as the pair JSD's
    independent resampling does (§5.5).

    Args:
        matrix: ``(n, k)`` per-cell mass; the last column is ``unallocated``.
        categories: ``k`` column names.
        codes: Tile id per cell (``grid_codes``), or ``None`` for no CI.
        n_reps: Replicates.
        seed: RNG seed.
        keep: Cells to include (default: all).

    Returns:
        The composition.

    Raises:
        ValueError: If the shapes disagree.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(categories):
        raise ValueError("matrix must be (n, len(categories))")
    selected = np.ones(len(values), dtype=bool) if keep is None else np.asarray(keep)
    values = values[selected]
    totals = values.sum(axis=0)
    point = _shares(totals)
    point_renorm = _shares_renormalised(totals)
    low = np.full(len(categories), math.nan)
    high = np.full(len(categories), math.nan)
    low_r = np.full(len(categories), math.nan)
    high_r = np.full(len(categories), math.nan)
    n_tiles = 0
    reps = 0
    if codes is not None and len(values):
        dense, n_tiles = _dense_codes(np.asarray(codes)[selected])
        if n_tiles > 0:
            tile_totals = np.zeros((n_tiles, values.shape[1]))
            used = dense >= 0
            np.add.at(tile_totals, dense[used], values[used])
            rng = np.random.default_rng(seed)
            weights = _tile_weights(rng, n_tiles, n_reps)
            replicate_totals = weights @ tile_totals
            replicate = np.vstack([_shares(row) for row in replicate_totals])
            replicate_r = np.vstack(
                [_shares_renormalised(row) for row in replicate_totals]
            )
            for column in range(len(categories)):
                low[column], high[column] = _percentile_ci(replicate[:, column])
                low_r[column], high_r[column] = _percentile_ci(replicate_r[:, column])
            reps = int(n_reps)
    records = tuple(
        ShareRecord(
            category=str(name),
            share=float(point[column]),
            ci_low=float(low[column]),
            ci_high=float(high[column]),
            share_renormalised=float(point_renorm[column]),
            renormalised_ci_low=float(low_r[column]),
            renormalised_ci_high=float(high_r[column]),
            mass=float(totals[column]),
        )
        for column, name in enumerate(categories)
    )
    return ShareResult(
        records=records, n_cells=int(len(values)), n_tiles=int(n_tiles), n_reps=reps
    )


def _shares(totals: np.ndarray) -> np.ndarray:
    mass = float(np.sum(totals))
    if mass <= 0:
        return np.full(len(totals), math.nan)
    return np.asarray(totals, dtype=np.float64) / mass


def _shares_renormalised(totals: np.ndarray) -> np.ndarray:
    classes = np.asarray(totals[:-1], dtype=np.float64)
    mass = float(classes.sum())
    out = np.full(len(totals), math.nan)
    if mass > 0:
        out[:-1] = classes / mass
    return out


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Return the base-2 Jensen-Shannon distance of two mass vectors.

    Both are renormalised first (as ``scipy.spatial.distance.jensenshannon``
    with ``base=2``).

    Args:
        p: Non-negative masses.
        q: Non-negative masses of the same length.

    Returns:
        Distance in [0, 1]; NaN if either vector has no mass.
    """
    first = np.asarray(p, dtype=np.float64)
    second = np.asarray(q, dtype=np.float64)
    if first.sum() <= 0 or second.sum() <= 0:
        return math.nan
    first = first / first.sum()
    second = second / second.sum()
    middle = 0.5 * (first + second)
    with np.errstate(divide="ignore", invalid="ignore"):
        left = np.where(first > 0, first * np.log2(first / middle), 0.0)
        right = np.where(second > 0, second * np.log2(second / middle), 0.0)
    return float(math.sqrt(max(0.0, 0.5 * (left.sum() + right.sum()))))


# --------------------------------------------------------------------------
# Aligned-bin concordance (item 7)


def aligned_bin_density_correlation(
    xy_a: np.ndarray,
    matrix_a: np.ndarray,
    xy_b: np.ndarray,
    matrix_b: np.ndarray,
    categories: Sequence[str],
    *,
    bin_um: float = DENSITY_BIN_UM,
    min_cells_per_bin: int = 5,
) -> pd.DataFrame:
    """Correlate per-type densities of two sections in aligned bins (item 7).

    Both sections must be in one frame (the MERSCOPE ``*_aligned_nonrigid``
    coordinates are in the Xenium frame). The two sections are binned on one
    ``bin_um`` grid; only bins holding at least ``min_cells_per_bin`` cells
    in **both** sections are compared (the tissue they share). A type's
    density in a bin is its mass (cells, or soft mass) per mm². The Pearson
    and Spearman correlations across the shared bins measure whether the two
    platforms place the type in the same tissue, independently of their
    overall calling rates.

    Args:
        xy_a: ``(n_a, 2)`` coordinates of the first section (µm).
        matrix_a: ``(n_a, k)`` per-cell mass over ``categories``.
        xy_b: ``(n_b, 2)`` coordinates of the second section, same frame.
        matrix_b: ``(n_b, k)`` per-cell mass.
        categories: ``k`` category names.
        bin_um: Bin edge (200 µm in the plan).
        min_cells_per_bin: Cells each section needs in a bin to compare it.

    Returns:
        One row per category: ``category``, ``n_bins``, ``pearson_r``,
        ``spearman_r``, ``density_a_per_mm2``, ``density_b_per_mm2`` (means
        over the shared bins), ``bin_um``.

    Raises:
        ValueError: If shapes disagree.
    """
    first_xy = np.asarray(xy_a, dtype=np.float64)
    second_xy = np.asarray(xy_b, dtype=np.float64)
    first = np.asarray(matrix_a, dtype=np.float64)
    second = np.asarray(matrix_b, dtype=np.float64)
    if first.shape != (len(first_xy), len(categories)) or second.shape != (
        len(second_xy),
        len(categories),
    ):
        raise ValueError("matrices must be (n, len(categories)) per section")
    codes = grid_codes(np.vstack([first_xy, second_xy]), bin_um)
    codes_a, codes_b = codes[: len(first_xy)], codes[len(first_xy) :]
    n_bins = int(codes.max()) + 1 if len(codes) and codes.max() >= 0 else 0
    area_mm2 = (bin_um / 1000.0) ** 2
    rows: list[dict[str, object]] = []
    if n_bins == 0:
        return pd.DataFrame(
            [
                _density_row(name, 0, math.nan, math.nan, math.nan, math.nan, bin_um)
                for name in categories
            ]
        )
    cells_a = np.bincount(codes_a[codes_a >= 0], minlength=n_bins)
    cells_b = np.bincount(codes_b[codes_b >= 0], minlength=n_bins)
    shared = (cells_a >= min_cells_per_bin) & (cells_b >= min_cells_per_bin)
    for column, name in enumerate(categories):
        mass_a = _bin_sum(first[:, column], codes_a, n_bins)[shared] / area_mm2
        mass_b = _bin_sum(second[:, column], codes_b, n_bins)[shared] / area_mm2
        rows.append(
            _density_row(
                name,
                int(shared.sum()),
                _pearson(mass_a, mass_b),
                _spearman(mass_a, mass_b),
                float(mass_a.mean()) if mass_a.size else math.nan,
                float(mass_b.mean()) if mass_b.size else math.nan,
                bin_um,
            )
        )
    return pd.DataFrame(rows)


def _density_row(
    name: str,
    n_bins: int,
    pearson: float,
    spearman: float,
    density_a: float,
    density_b: float,
    bin_um: float,
) -> dict[str, object]:
    return {
        "category": name,
        "n_bins": n_bins,
        "pearson_r": pearson,
        "spearman_r": spearman,
        "density_a_per_mm2": density_a,
        "density_b_per_mm2": density_b,
        "bin_um": float(bin_um),
    }


def _bin_sum(values: np.ndarray, codes: np.ndarray, n_bins: int) -> np.ndarray:
    keep = codes >= 0
    return np.bincount(codes[keep], weights=values[keep], minlength=n_bins)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    a = pd.Series(np.asarray(first, dtype=np.float64)).rank().to_numpy()
    b = pd.Series(np.asarray(second, dtype=np.float64)).rank().to_numpy()
    return _pearson(a, b)


def pearson_r(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Pearson correlation (NaN for < 3 points or no variance)."""
    return _pearson(first, second)


def spearman_r(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Spearman rank correlation (NaN like ``pearson_r``)."""
    return _spearman(first, second)


# --------------------------------------------------------------------------
# Cortical depth (item 9; H12)


@dataclass(frozen=True)
class MedianCi:
    """A median with its spatial block-bootstrap CI.

    Attributes:
        median: Point median.
        ci_low: 2.5th percentile of the replicate medians.
        ci_high: 97.5th percentile.
        n_cells: Values used.
        n_tiles: Tiles resampled.
    """

    median: float
    ci_low: float
    ci_high: float
    n_cells: int
    n_tiles: int


def _weighted_median(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return math.nan
    cumulative = np.cumsum(weights)
    position = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


def median_block_ci(
    values: np.ndarray,
    codes: np.ndarray | None,
    *,
    n_reps: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> MedianCi:
    """Return the median of ``values`` with a block-bootstrap percentile CI.

    Replicates resample the tiles (``codes``) with replacement; a replicate
    median weights each value by its tile's multiplicity. Non-finite values
    are dropped first.

    Args:
        values: Per-cell values (e.g. cortical depth).
        codes: Tile id per cell, or ``None`` for no CI.
        n_reps: Replicates.
        seed: RNG seed.

    Returns:
        The median and CI (NaN without values).
    """
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    data = data[finite]
    if data.size == 0:
        return MedianCi(math.nan, math.nan, math.nan, 0, 0)
    point = float(np.median(data))
    if codes is None:
        return MedianCi(point, math.nan, math.nan, int(data.size), 0)
    dense, n_tiles = _dense_codes(np.asarray(codes)[finite])
    used = dense >= 0
    if n_tiles == 0:
        return MedianCi(point, math.nan, math.nan, int(data.size), 0)
    order = np.argsort(data[used], kind="stable")
    sorted_values = data[used][order]
    sorted_codes = dense[used][order]
    rng = np.random.default_rng(seed)
    weights = _tile_weights(rng, n_tiles, n_reps)
    replicates = np.array(
        [_weighted_median(sorted_values, row[sorted_codes]) for row in weights]
    )
    low, high = _percentile_ci(replicates)
    return MedianCi(point, low, high, int(data.size), int(n_tiles))


@dataclass(frozen=True)
class OrderingResult:
    """Whether group medians follow a strict order with separated CIs (H12).

    Attributes:
        order: Groups in the expected order (shallow to deep).
        medians: Their medians (NaN when a group has no cells).
        ordered: Point medians strictly increase along ``order``.
        separated: Each group's CI upper bound lies below the next group's
            lower bound.
        passes: ``ordered and separated`` with every group present.
        missing: Groups without cells (or without a CI).
    """

    order: tuple[str, ...]
    medians: tuple[float, ...]
    ordered: bool
    separated: bool
    passes: bool
    missing: tuple[str, ...] = field(default_factory=tuple)


def depth_ordering(
    medians: Mapping[str, MedianCi], order: Sequence[str], *, min_cells: int = 1
) -> OrderingResult:
    """Test ``order[0] < order[1] < ...`` on medians with non-overlapping CIs.

    Args:
        medians: Group to its ``MedianCi``.
        order: Expected order, shallow (pia) to deep (white matter).
        min_cells: Cells a group needs to count as present.

    Returns:
        The ordering result; a missing group or CI fails it.
    """
    groups = tuple(order)
    present = [
        group
        for group in groups
        if group in medians
        and medians[group].n_cells >= min_cells
        and math.isfinite(medians[group].median)
    ]
    missing = tuple(group for group in groups if group not in present)
    values = tuple(
        medians[group].median if group in present else math.nan for group in groups
    )
    ordered = not missing and all(
        values[index] < values[index + 1] for index in range(len(groups) - 1)
    )
    separated = not missing and all(
        math.isfinite(medians[groups[index]].ci_high)
        and math.isfinite(medians[groups[index + 1]].ci_low)
        and medians[groups[index]].ci_high < medians[groups[index + 1]].ci_low
        for index in range(len(groups) - 1)
    )
    return OrderingResult(
        order=groups,
        medians=values,
        ordered=bool(ordered),
        separated=bool(separated),
        passes=bool(ordered and separated),
        missing=missing,
    )


@dataclass(frozen=True)
class ReplicationResult:
    """Whether the depth ordering replicates across the platforms of a pair.

    Attributes:
        platforms: Platforms compared.
        passes_per_platform: The ordering verdict per platform.
        replicated: The ordering passes on every platform (at least two).
        n_groups: Groups with a median on every platform.
        spearman_r: Rank correlation of the group medians between the first
            two platforms (NaN below 3 shared groups).
        max_abs_difference: Largest absolute median difference of a shared
            group.
    """

    platforms: tuple[str, ...]
    passes_per_platform: tuple[bool, ...]
    replicated: bool
    n_groups: int
    spearman_r: float
    max_abs_difference: float


def depth_replication(
    medians: Mapping[str, Mapping[str, MedianCi]],
    orderings: Mapping[str, OrderingResult],
) -> ReplicationResult:
    """Return the replication of the depth profile across platforms (H12).

    Args:
        medians: Platform to group medians (every group with a median, not
            only the ordered ones).
        orderings: Platform to its ordering result.

    Returns:
        The replication record.
    """
    platforms = tuple(sorted(orderings))
    passes = tuple(bool(orderings[name].passes) for name in platforms)
    replicated = len(platforms) >= 2 and all(passes)
    spearman = math.nan
    max_diff = math.nan
    n_groups = 0
    if len(platforms) >= 2:
        first = medians.get(platforms[0], {})
        second = medians.get(platforms[1], {})
        shared = sorted(
            group
            for group in set(first) & set(second)
            if math.isfinite(first[group].median)
            and math.isfinite(second[group].median)
        )
        n_groups = len(shared)
        if shared:
            a = np.array([first[group].median for group in shared])
            b = np.array([second[group].median for group in shared])
            max_diff = float(np.max(np.abs(a - b)))
            spearman = _spearman(a, b) if n_groups >= 3 else math.nan
    return ReplicationResult(
        platforms=platforms,
        passes_per_platform=passes,
        replicated=bool(replicated),
        n_groups=n_groups,
        spearman_r=spearman,
        max_abs_difference=max_diff,
    )


@dataclass(frozen=True)
class ShareContrast:
    """A class's share in region A vs region B with a block-bootstrap CI.

    Attributes:
        share_a: Class share of region A's cells.
        share_b: Class share of region B's cells.
        difference: ``share_a - share_b``.
        ci_low: 2.5th percentile of the replicate differences.
        ci_high: 97.5th percentile.
        n_a: Cells in region A.
        n_b: Cells in region B.
        greater: ``share_a > share_b``.
        ci_excludes_zero: ``ci_low > 0``.
    """

    share_a: float
    share_b: float
    difference: float
    ci_low: float
    ci_high: float
    n_a: int
    n_b: int
    greater: bool
    ci_excludes_zero: bool


def share_contrast(
    is_class: np.ndarray,
    in_a: np.ndarray,
    in_b: np.ndarray,
    codes: np.ndarray | None,
    *,
    n_reps: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> ShareContrast:
    """Return a class's share in two regions and their difference (H12).

    Used for "oligodendrocytes WM > GM": the share of oligodendrocytes among
    the white-matter cells vs among the grey-matter cells. Replicates
    resample tiles (a tile may hold both regions).

    Args:
        is_class: Whether each cell is of the class.
        in_a: Whether each cell lies in region A (e.g. white matter).
        in_b: Whether each cell lies in region B (e.g. grey matter).
        codes: Tile id per cell, or ``None`` for no CI.
        n_reps: Replicates.
        seed: RNG seed.

    Returns:
        The contrast (NaN shares for an empty region).
    """
    cls = np.asarray(is_class, dtype=bool)
    a = np.asarray(in_a, dtype=bool)
    b = np.asarray(in_b, dtype=bool)
    n_a, n_b = int(a.sum()), int(b.sum())
    share_a = float(cls[a].mean()) if n_a else math.nan
    share_b = float(cls[b].mean()) if n_b else math.nan
    difference = share_a - share_b
    low = high = math.nan
    if codes is not None and n_a and n_b:
        dense, n_tiles = _dense_codes(np.asarray(codes))
        used = dense >= 0
        if n_tiles:
            sums = np.zeros((n_tiles, 4))
            np.add.at(sums[:, 0], dense[used], (cls & a)[used])
            np.add.at(sums[:, 1], dense[used], a[used])
            np.add.at(sums[:, 2], dense[used], (cls & b)[used])
            np.add.at(sums[:, 3], dense[used], b[used])
            rng = np.random.default_rng(seed)
            totals = _tile_weights(rng, n_tiles, n_reps) @ sums
            with np.errstate(divide="ignore", invalid="ignore"):
                replicate = totals[:, 0] / totals[:, 1] - totals[:, 2] / totals[:, 3]
            low, high = _percentile_ci(replicate)
    return ShareContrast(
        share_a=share_a,
        share_b=share_b,
        difference=difference,
        ci_low=low,
        ci_high=high,
        n_a=n_a,
        n_b=n_b,
        greater=bool(math.isfinite(difference) and difference > 0),
        ci_excludes_zero=bool(math.isfinite(low) and low > 0),
    )


def depth_profile(
    depth: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    categories: Sequence[str],
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return each category's share of cells per depth bin (item 9 gradients).

    Args:
        depth: Normalised depth per cell (0 = pia, 1 = white matter).
        labels: Label per cell.
        categories: Categories to profile.
        n_bins: Equal-width bins over [0, 1].

    Returns:
        Rows ``depth_bin``, ``depth_low``, ``depth_high``, ``category``,
        ``n_cells`` (bin total), ``n_category``, ``share``.
    """
    values = np.asarray(depth, dtype=np.float64)
    names = np.asarray(labels, dtype=object).astype(str)
    finite = np.isfinite(values)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(values[finite], edges[1:-1]), 0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        in_bin = bins == index
        total = int(in_bin.sum())
        for name in categories:
            count = int(np.count_nonzero(names[finite][in_bin] == name))
            rows.append(
                {
                    "depth_bin": index,
                    "depth_low": float(edges[index]),
                    "depth_high": float(edges[index + 1]),
                    "category": name,
                    "n_cells": total,
                    "n_category": count,
                    "share": count / total if total else math.nan,
                }
            )
    return pd.DataFrame(rows)


def profile_gradient(profile: pd.DataFrame) -> pd.DataFrame:
    """Return the Spearman correlation of each category's share with depth.

    Args:
        profile: ``depth_profile`` output.

    Returns:
        Rows ``category``, ``spearman_r`` (share vs bin centre),
        ``share_superficial`` (first bin), ``share_deep`` (last bin).
    """
    rows = []
    for name, part in profile.groupby("category", sort=False):
        ordered = part.sort_values("depth_bin")
        centre = 0.5 * (ordered["depth_low"] + ordered["depth_high"]).to_numpy()
        share = ordered["share"].to_numpy(dtype=np.float64)
        keep = np.isfinite(share)
        rows.append(
            {
                "category": name,
                "spearman_r": _spearman(centre[keep], share[keep]),
                "share_superficial": float(share[0]) if len(share) else math.nan,
                "share_deep": float(share[-1]) if len(share) else math.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Platform factors and pseudobulks (items 4, 7)


def platform_gene_log2_ratios(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    *,
    pseudocount: float = GENE_RATIO_PSEUDOCOUNT,
    centre: bool = True,
) -> tuple[np.ndarray, float]:
    """Return per-gene ``log2((mean_b + pc) / (mean_a + pc))``, median-centred.

    The E5 set-c definition (median-centred log2 of the Xenium over the
    MERSCOPE per-cell mean counts within a confident label).

    Args:
        mean_a: Per-gene mean counts per cell of the first platform.
        mean_b: The same genes on the second platform.
        pseudocount: Added to both means.
        centre: Subtract the median ratio (over finite genes).

    Returns:
        ``(ratios, median)``: the (centred) ratios and the median removed
        (the uncentred median when ``centre`` is false).

    Raises:
        ValueError: If lengths differ or the pseudocount is not positive.
    """
    a = np.asarray(mean_a, dtype=np.float64)
    b = np.asarray(mean_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("mean_a and mean_b must have the same shape")
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    ratios = np.log2((b + pseudocount) / (a + pseudocount))
    finite = np.isfinite(ratios)
    median = float(np.median(ratios[finite])) if finite.any() else math.nan
    if centre and math.isfinite(median):
        ratios = ratios - median
    return ratios, median


def group_mean_counts(
    matrix: object, groups: Sequence[object] | np.ndarray, group_order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-group mean counts per cell and group sizes.

    Args:
        matrix: ``(n, g)`` counts (numpy array or scipy sparse matrix).
        groups: Group per cell (cells outside ``group_order`` are ignored).
        group_order: Groups, in row order of the result.

    Returns:
        ``(means, n_cells)``: ``(len(group_order), g)`` means and cell counts.
    """
    labels = np.asarray(groups, dtype=object).astype(str)
    n_groups = len(group_order)
    index = {name: position for position, name in enumerate(group_order)}
    rows = np.array([index.get(label, -1) for label in labels], dtype=np.int64)
    keep = rows >= 0
    n_cells = np.bincount(rows[keep], minlength=n_groups).astype(np.int64)
    n_genes = int(matrix.shape[1])  # type: ignore[attr-defined]
    indicator = np.zeros((n_groups, len(labels)), dtype=np.float64)
    indicator[rows[keep], np.flatnonzero(keep)] = 1.0
    sums = np.asarray(_dense(indicator @ matrix)).reshape(n_groups, n_genes)  # type: ignore[operator]
    with np.errstate(divide="ignore", invalid="ignore"):
        means = sums / n_cells[:, None]
    means[n_cells == 0] = np.nan
    return means, n_cells


def _dense(value: object) -> np.ndarray:
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    return np.asarray(value)


def log_cpm(means: np.ndarray) -> np.ndarray:
    """Return ``log2(CPM + 1)`` of per-gene mean counts (rows = profiles)."""
    values = np.asarray(means, dtype=np.float64)
    totals = np.nansum(values, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpm = values / totals * 1e6
    return np.asarray(np.log2(cpm + 1.0), dtype=np.float64)


def nearest_centroid(
    profile: np.ndarray, centroids: np.ndarray, names: Sequence[str]
) -> tuple[str | None, float]:
    """Return the reference centroid best correlated with a pseudobulk.

    A correlation re-mapping of a label's pseudobulk (item 4): the Pearson
    correlation of ``log2(CPM + 1)`` profiles over the shared genes.

    Args:
        profile: ``(g,)`` pseudobulk log profile.
        centroids: ``(m, g)`` reference log profiles.
        names: ``m`` centroid names.

    Returns:
        ``(name, r)`` of the best centroid (``(None, NaN)`` if none has a
        finite correlation).
    """
    best: tuple[str | None, float] = (None, math.nan)
    for name, row in zip(names, np.asarray(centroids), strict=True):
        keep = np.isfinite(row) & np.isfinite(profile)
        r = _pearson(profile[keep], row[keep])
        if math.isfinite(r) and (not math.isfinite(best[1]) or r > best[1]):
            best = (str(name), r)
    return best


# --------------------------------------------------------------------------
# Method agreement, histograms, maps (items 1, 3, 6, 10)


def agreement_by_quantile(
    first: Sequence[object] | np.ndarray,
    second: Sequence[object] | np.ndarray,
    counts: np.ndarray,
    *,
    classes: Sequence[str],
    unassigned: str,
    n_quantiles: int = 4,
    mask: np.ndarray | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Return the share of cells whose two labels agree, per depth quantile.

    As the H3 definition: labels outside ``classes`` count as one
    ``unassigned`` label, so two out-of-class labels agree unless ``strict``.

    Args:
        first: Labels of the first method.
        second: Labels of the second method.
        counts: Total counts per cell.
        classes: The label vocabulary compared.
        unassigned: The out-of-class label.
        n_quantiles: Count quantiles (4: quartiles).
        mask: Cells to score (default: all).
        strict: Count out-of-class labels as disagreeing.

    Returns:
        Rows ``quantile`` (1-based), ``counts_low``, ``counts_high``,
        ``n_cells``, ``agreement``, plus an ``all`` row (quantile 0).
    """
    a = _normalised_labels(first, classes, unassigned)
    b = _normalised_labels(second, classes, unassigned)
    same = a == b
    if strict:
        same &= a != unassigned
    values = np.asarray(counts, dtype=np.float64)
    keep = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask, bool)
    rows = [
        {
            "quantile": 0,
            "counts_low": float(np.nanmin(values[keep])) if keep.any() else math.nan,
            "counts_high": float(np.nanmax(values[keep])) if keep.any() else math.nan,
            "n_cells": int(keep.sum()),
            "agreement": float(same[keep].mean()) if keep.any() else math.nan,
        }
    ]
    if keep.any():
        edges = np.quantile(values[keep], np.linspace(0, 1, n_quantiles + 1))
        bins = np.clip(
            np.searchsorted(edges[1:-1], values, side="right"), 0, n_quantiles - 1
        )
        for index in range(n_quantiles):
            selected = keep & (bins == index)
            rows.append(
                {
                    "quantile": index + 1,
                    "counts_low": float(edges[index]),
                    "counts_high": float(edges[index + 1]),
                    "n_cells": int(selected.sum()),
                    "agreement": (
                        float(same[selected].mean()) if selected.any() else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _normalised_labels(
    labels: Sequence[object] | np.ndarray, classes: Sequence[str], unassigned: str
) -> np.ndarray:
    values = pd.Series(np.asarray(labels, dtype=object), dtype=object).astype(str)
    return np.where(values.isin(list(classes)), values, unassigned).astype(object)


def histogram_2d(
    x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> pd.DataFrame:
    """Return a 2D histogram as a long table (item 3).

    Args:
        x: First coordinate per cell (e.g. log10 total counts).
        y: Second coordinate (e.g. bootstrap probability).
        x_edges: Bin edges of ``x``.
        y_edges: Bin edges of ``y``.

    Returns:
        Rows ``x_low``, ``x_high``, ``y_low``, ``y_high``, ``n``.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    counts, _, _ = np.histogram2d(a[keep], b[keep], bins=[x_edges, y_edges])
    rows = []
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            rows.append(
                {
                    "x_low": float(x_edges[i]),
                    "x_high": float(x_edges[i + 1]),
                    "y_low": float(y_edges[j]),
                    "y_high": float(y_edges[j + 1]),
                    "n": int(counts[i, j]),
                }
            )
    return pd.DataFrame(rows)


def tile_mean_map(
    xy: np.ndarray, values: np.ndarray, *, tile_um: float = TILE_UM
) -> pd.DataFrame:
    """Return the mean of a per-cell value per square tile (spatial maps).

    Args:
        xy: ``(n, 2)`` coordinates (µm).
        values: Per-cell values (bools are rates; NaN ignored).
        tile_um: Tile edge.

    Returns:
        Rows ``tile_x``, ``tile_y`` (tile index), ``x_um``, ``y_um`` (tile
        centre), ``n_cells``, ``value`` (mean of finite values).
    """
    points = np.asarray(xy, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(points).all(axis=1) & np.isfinite(data)
    if not keep.any():
        return pd.DataFrame(
            columns=["tile_x", "tile_y", "x_um", "y_um", "n_cells", "value"]
        )
    tiles = np.floor(points[keep] / tile_um).astype(np.int64)
    frame = pd.DataFrame(
        {"tile_x": tiles[:, 0], "tile_y": tiles[:, 1], "value": data[keep]}
    )
    grouped = (
        frame.groupby(["tile_x", "tile_y"], sort=True)["value"]
        .agg(["size", "mean"])
        .reset_index()
        .rename(columns={"size": "n_cells", "mean": "value"})
    )
    grouped["x_um"] = (grouped["tile_x"] + 0.5) * tile_um
    grouped["y_um"] = (grouped["tile_y"] + 0.5) * tile_um
    return grouped[["tile_x", "tile_y", "x_um", "y_um", "n_cells", "value"]]


@dataclass(frozen=True)
class SelfThinningEligibility:
    """Whether a dataset supports the self-thinning diagnostic (item 1).

    Attributes:
        n_deep: Table cells with at least ``min_counts`` counts.
        eligible: ``n_deep >= min_cells``.
        composition: Share of each label among the deep cells.
        dominant_label: The commonest label among them.
        dominant_share: Its share.
        reliable: Eligible and no label above ``max_dominant_share``.
        min_counts: Depth threshold (200).
        min_cells: Cells required (500).
    """

    n_deep: int
    eligible: bool
    composition: dict[str, float]
    dominant_label: str | None
    dominant_share: float
    reliable: bool
    min_counts: int
    min_cells: int


def self_thinning_eligibility(
    total_counts: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    *,
    min_counts: int = SELF_THINNING_MIN_COUNTS,
    min_cells: int = SELF_THINNING_MIN_CELLS,
    max_dominant_share: float = SELF_THINNING_MAX_DOMINANT_SHARE,
) -> SelfThinningEligibility:
    """Return the self-thinning eligibility and its truth composition.

    The diagnostic re-maps thinned deep cells, whose full-depth labels are
    the truth (§5.4, E2 verdict 2); it is informative only when ≥ 500 cells
    have ≥ 200 counts, and only for the classes those cells hold.

    Args:
        total_counts: Counts per table cell.
        labels: Their full-depth labels (e.g. the confident broad class).
        min_counts: Depth a cell needs.
        min_cells: Deep cells needed.
        max_dominant_share: Largest single-label share for a reliable run.

    Returns:
        The eligibility record.
    """
    counts = np.asarray(total_counts, dtype=np.float64)
    names = np.asarray(labels, dtype=object).astype(str)
    deep = counts >= min_counts
    n_deep = int(deep.sum())
    composition: dict[str, float] = {}
    dominant: str | None = None
    dominant_share = math.nan
    if n_deep:
        values, freq = np.unique(names[deep], return_counts=True)
        order = np.argsort(-freq, kind="stable")
        composition = {str(values[i]): float(freq[i] / n_deep) for i in order}
        dominant = str(values[order[0]])
        dominant_share = float(freq[order[0]] / n_deep)
    eligible = n_deep >= min_cells
    return SelfThinningEligibility(
        n_deep=n_deep,
        eligible=bool(eligible),
        composition=composition,
        dominant_label=dominant,
        dominant_share=dominant_share,
        reliable=bool(eligible and dominant_share <= max_dominant_share),
        min_counts=int(min_counts),
        min_cells=int(min_cells),
    )


__all__ = [
    "BOOTSTRAP_SEED",
    "DENSITY_BIN_UM",
    "GENE_RATIO_PSEUDOCOUNT",
    "N_BOOTSTRAP",
    "SELF_THINNING_MAX_DOMINANT_SHARE",
    "SELF_THINNING_MIN_CELLS",
    "SELF_THINNING_MIN_COUNTS",
    "TILE_UM",
    "UNALLOCATED",
    "MedianCi",
    "OrderingResult",
    "ReplicationResult",
    "SelfThinningEligibility",
    "ShareContrast",
    "ShareRecord",
    "ShareResult",
    "agreement_by_quantile",
    "aligned_bin_density_correlation",
    "block_bootstrap_shares",
    "depth_ordering",
    "depth_profile",
    "depth_replication",
    "grid_codes",
    "group_mean_counts",
    "histogram_2d",
    "jensen_shannon_distance",
    "log_cpm",
    "median_block_ci",
    "nearest_centroid",
    "one_hot_matrix",
    "pearson_r",
    "platform_gene_log2_ratios",
    "profile_gradient",
    "self_thinning_eligibility",
    "share_contrast",
    "soft_level_matrix",
    "spearman_r",
    "tile_mean_map",
]
