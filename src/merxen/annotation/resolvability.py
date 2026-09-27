"""Resolvability self-map: which levels a panel resolves, where and how (§8.3).

Before any real cell is labelled, reference cells held out from the mapping
reference are restricted to the panel, thinned to the depth grid, perturbed
with the fixed ``R1_contam_HO`` recipe (E2) and mapped with the production
configuration. From the calls this module derives, per (level, parent class,
depth bin):

* **local thresholds** (E2 verdict 3; moved from M10 into v1): an isotonic
  map ``g(bp)`` of correctness is fitted on one half of the test cells and
  ``t* = min{t >= default : g(t) >= target}`` (capped) is checked on the
  other half. Thresholds are raise-only: never below the v1 raw default.
  The set-level rule ("lowest threshold whose accepted set reaches the
  target") is never used; ``set_level_threshold`` exists only so tests and
  reports can show where it fails;
* **emission**: a bin is emitted when ``t*`` exists and the check half holds
  at least ``min_confident_n`` confident calls with a Wilson 95% lower bound
  of their precision at least ``target - wilson_margin`` and coverage at
  least ``min_coverage``. Bins deeper than ``D_max(c)`` (the deepest grid
  value with ``min_cells_per_bin`` test cells of class ``c``; only cells whose
  native panel counts reach a depth are thinned to it) inherit the decision
  of ``D_max(c)`` and are marked extrapolated; a class without such a bin is
  never emitted (no pooling across classes). Everything not emitted is
  ``not_resolvable``;
* **floors**: the smallest emitted depth, combined with the known floors by
  the max rule for panels without real-data validation (§5.4);
* **trust constraints**: ``refused`` when broad fails the local rule (base
  target) at every depth up to ``trust_max_depth``; ``broad_only`` when the
  leaf level is resolvable for fewer than half of the classes with enough
  test cells at every such depth (§8.2).

Three decision regimes are tabulated, because the trust state is known only
after diagnostics: ``validated`` (real-data-validated panels: base targets,
the pre-registered default applied, raised local thresholds only reported;
H18), ``provisional`` (panels without real-data validation and families
validated by simulation: targets + the provisional margins, local
thresholds applied) and ``trust`` (base targets with the local rule; the
§8.2 refusal and broad-only tests).

PREP writes ``resolvability.parquet`` (bins, precision-coverage curves,
isotonic knots, per-node F1, confusion, decisions), ``resolvability_cells
.parquet`` (per simulated cell x level: parent class, depth, truth, call, bp,
half) and ``resolvability_summary.json`` into the bundle. RESOLVE refits the
decisions with the simulated cells reweighted to each dataset's composition
(``composition_weights``, ``decide``): PREP's unweighted tables set only the
panel's trust constraints.

This module needs numpy, pandas and scipy only; MapMyCells runs through the
``map_fn`` a builder passes to ``run_resolvability``.
"""

from __future__ import annotations

import json
import logging
import math
import time
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import pandas as pd

from merxen.annotation.schema import meets_threshold

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.config import (
        AnnotationResolvabilityConfig,
        AnnotationThresholds,
    )

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants

# Bumped when the simulation, tabulation or decision logic changes what the
# resolvability files contain (it enters build_hash with the recipe).
RESOLVABILITY_VERSION: Final = 1
SUMMARY_SCHEMA_VERSION: Final = 1
RESOLVABILITY_FILE: Final = "resolvability.parquet"
RESOLVABILITY_CELLS_FILE: Final = "resolvability_cells.parquet"
RESOLVABILITY_SUMMARY_FILE: Final = "resolvability_summary.json"
TEST_CELLS_FILE: Final = "test_cells.h5ad"
TEST_CELLS_OBS_FILE: Final = "test_cells.parquet"

DECISION_RECIPE: Final = "R1_contam_HO"
CLEAN_RECIPE: Final = "clean"
RECIPE_VERSIONS: Final[dict[str, int]] = {DECISION_RECIPE: 1, CLEAN_RECIPE: 1}

# Test-cell columns (``test_cells.parquet`` / ``test_cells.h5ad`` obs).
TRUTH_PREFIX: Final = "truth__"
TRUTH_LEAF_COLUMN: Final = "truth_leaf"
SPILL_GROUP_COLUMN: Final = "spill_group"
NATIVE_COUNTS_COLUMN: Final = "native_counts"

# Threshold grid of the local rule (E2 thresholds2.py: 201 points on [0, 1]).
THRESHOLD_GRID: Final = np.round(np.linspace(0.0, 1.0, 201), 3)
# Precision-coverage curve thresholds (§8.3: 0.50-0.99).
CURVE_THRESHOLDS: Final = np.round(np.arange(0.50, 0.995, 0.01), 2)
WILSON_Z: Final = 1.959963984540054
_TOLERANCE: Final = 1e-9

Regime = Literal["validated", "provisional", "trust"]
REGIMES: Final[tuple[Regime, ...]] = ("validated", "provisional", "trust")
LevelRole = Literal["lineage", "broad", "nt", "class", "leaf", "fine", "other"]
STATUS_EMITTED: Final = "emitted"
STATUS_NOT_RESOLVABLE: Final = "not_resolvable"

CELLS_COLUMNS: Final[tuple[str, ...]] = (
    "recipe",
    "seed",
    "level",
    "sim_id",
    "cell_id",
    "depth",
    "half",
    "parent",
    "call",
    "bp",
    "corr",
    "truth",
    "truth_parent",
    TRUTH_LEAF_COLUMN,
    "correct",
    "total_counts",
)


class ResolvabilityError(RuntimeError):
    """The self-map cannot run on its inputs."""


# --------------------------------------------------------------------------
# Small statistics


def wilson_lower_bound(precision: float, n: float, z: float = WILSON_Z) -> float:
    """Return the Wilson score lower bound of a proportion.

    Args:
        precision: Observed proportion in [0, 1].
        n: Number of trials (a Kish effective n for weighted sets).
        z: Normal quantile (95% two-sided by default).

    Returns:
        The lower bound; ``nan`` when ``n`` is not positive.
    """
    if not n > 0 or not math.isfinite(precision):
        return math.nan
    p = min(max(float(precision), 0.0), 1.0)
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    spread = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - spread) / denominator)


def kish_effective_n(weights: np.ndarray) -> float:
    """Return Kish's effective sample size ``(sum w)^2 / sum w^2``.

    Args:
        weights: Non-negative weights.

    Returns:
        The effective n (0 for no positive weight).
    """
    values = np.asarray(weights, dtype=np.float64)
    squared = float(np.sum(values * values))
    if squared <= 0.0:
        return 0.0
    return float(np.sum(values)) ** 2 / squared


@dataclass(frozen=True)
class IsotonicFit:
    """A non-decreasing isotonic map ``g(bp)`` of correctness.

    Attributes:
        x: Distinct fitted bp values, ascending.
        y: Fitted correctness at ``x`` (non-decreasing).
        n: Cells in the fit.
    """

    x: np.ndarray
    y: np.ndarray
    n: int

    def predict(self, values: np.ndarray | Sequence[float] | float) -> np.ndarray:
        """Return ``g`` at some bp values.

        Linear between knots and constant beyond them, as scikit-learn's
        ``IsotonicRegression(out_of_bounds="clip")`` that E2 used.

        Args:
            values: bp values.

        Returns:
            ``g(values)``.
        """
        return np.asarray(
            np.interp(np.asarray(values, dtype=np.float64), self.x, self.y),
            dtype=np.float64,
        )


def isotonic_fit(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> IsotonicFit:
    """Fit a weighted non-decreasing isotonic regression (pool adjacent violators).

    Ties in ``x`` are merged into their weighted mean first, as
    scikit-learn does, so ``predict`` matches ``IsotonicRegression``.

    Args:
        x: bp values.
        y: Correctness (0 / 1, or any values).
        weights: Non-negative weights (default 1).

    Returns:
        The fit.

    Raises:
        ValueError: If there is no point with positive weight.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    ws = (
        np.ones(len(xs), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    keep = np.isfinite(xs) & np.isfinite(ys) & (ws > 0)
    xs, ys, ws = xs[keep], ys[keep], ws[keep]
    if len(xs) == 0:
        raise ValueError("isotonic_fit needs at least one point with positive weight")
    unique, inverse = np.unique(xs, return_inverse=True)
    block_weight = np.bincount(inverse, weights=ws)
    block_sum = np.bincount(inverse, weights=ws * ys)
    means: list[float] = []
    totals: list[float] = []
    counts: list[int] = []
    for value, weight in zip(block_sum / block_weight, block_weight, strict=True):
        means.append(float(value))
        totals.append(float(weight))
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1] + 1e-15:
            weight_sum = totals[-2] + totals[-1]
            merged = (means[-2] * totals[-2] + means[-1] * totals[-1]) / weight_sum
            means[-2:] = [merged]
            totals[-2:] = [weight_sum]
            counts[-2:] = [counts[-2] + counts[-1]]
    fitted = np.repeat(np.asarray(means), counts)
    return IsotonicFit(x=unique, y=np.clip(fitted, 0.0, 1.0), n=int(len(xs)))


def local_threshold(
    fit: IsotonicFit | None,
    *,
    default: float,
    target: float,
    cap: float,
) -> float | None:
    """Return the local threshold ``t* = min{t >= default : g(t) >= target}``.

    Args:
        fit: The isotonic fit (``None``: too few cells).
        default: The v1 raw threshold (raise-only floor of ``t*``).
        target: Precision target.
        cap: Largest allowed threshold.

    Returns:
        ``t*`` on ``THRESHOLD_GRID`` (never below ``default``), or ``None``
        when ``g`` never reaches the target at or below ``cap``.
    """
    if fit is None:
        return None
    grid = THRESHOLD_GRID[default - _TOLERANCE <= THRESHOLD_GRID]
    grid = grid[grid <= cap + _TOLERANCE]
    if grid.size == 0:
        return None
    values = fit.predict(grid)
    hits = np.flatnonzero(values >= target - _TOLERANCE)
    if hits.size == 0:
        return None
    return float(max(grid[hits[0]], default))


def set_level_threshold(
    bp: np.ndarray,
    correct: np.ndarray,
    *,
    target: float,
    floor: float = 0.0,
    weights: np.ndarray | None = None,
) -> float | None:
    """Return the set-level threshold (not used for decisions; E2 verdict 3).

    The lowest threshold ``t >= floor`` whose accepted set ``{bp >= t}``
    reaches the target precision. E2 found it misses the target in 37-50%
    of dataset x depth combinations, because a precise set above ``t`` can
    hide an error band just above ``t``.

    Args:
        bp: bp values.
        correct: Correctness.
        target: Precision target.
        floor: Lowest threshold considered.
        weights: Optional weights.

    Returns:
        The threshold, or ``None`` when no accepted set reaches the target.
    """
    values = np.asarray(bp, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    ws = np.ones(len(values)) if weights is None else np.asarray(weights, np.float64)
    candidates = np.unique(values[values >= floor - _TOLERANCE])
    for threshold in candidates:
        accepted = values >= threshold - _TOLERANCE
        total = float(ws[accepted].sum())
        if (
            total > 0
            and float((ws * ok)[accepted].sum()) / total >= target - _TOLERANCE
        ):
            return float(threshold)
    return None


def cell_split_half(cell_ids: Iterable[str]) -> np.ndarray:
    """Return each test cell's split half (0 = fit, 1 = check).

    ``crc32(cell id) mod 2`` as E2's ``thresholds2.py``, so every depth and
    recipe of one test cell falls in the same half.

    Args:
        cell_ids: Test cell ids.

    Returns:
        ``int8`` halves.
    """
    return np.array(
        [zlib.crc32(str(cell_id).encode("utf-8")) % 2 for cell_id in cell_ids],
        dtype=np.int8,
    )


def depth_bin(counts: np.ndarray | Sequence[float], grid: Sequence[int]) -> np.ndarray:
    """Return the depth bin of each cell: the largest grid value <= its counts.

    Args:
        counts: Total counts per cell.
        grid: The bundle's depth grid (ascending).

    Returns:
        Bins as float (``nan`` below the smallest grid value).
    """
    values = np.asarray(counts, dtype=np.float64)
    edges = np.asarray(sorted(grid), dtype=np.float64)
    index = np.searchsorted(edges, values, side="right") - 1
    result = np.full(values.shape, np.nan)
    inside = index >= 0
    result[inside] = edges[index[inside]]
    return result


# --------------------------------------------------------------------------
# Simulation recipe and thinning


@dataclass(frozen=True)
class SimulationRecipe:
    """One simulation recipe of the self-map (§8.3 step 3).

    Attributes:
        name: ``R1_contam_HO`` (decisions) or ``clean`` (upper bound).
        version: Recipe version.
        gene_efficiency_sigma: Sigma of the per-gene LogNormal(0, sigma)
            efficiency (median-normalised); ``None`` for none.
        spill_fraction: Foreign-class spill as a fraction of the depth.
        seed: Simulation seed.
    """

    name: str
    version: int
    gene_efficiency_sigma: float | None
    spill_fraction: float
    seed: int

    def to_json(self) -> dict[str, Any]:
        """Return the recipe as JSON-native values."""
        return {
            "name": self.name,
            "version": self.version,
            "gene_efficiency_sigma": self.gene_efficiency_sigma,
            "spill_fraction": self.spill_fraction,
            "seed": self.seed,
        }


def simulation_recipes(
    config: AnnotationResolvabilityConfig, *, seed: int = 0
) -> list[SimulationRecipe]:
    """Return the recipes of a self-map: ``R1_contam_HO`` then ``clean``.

    Args:
        config: Resolvability settings (sigma, spill).
        seed: Simulation seed.

    Returns:
        The decision recipe and the clean upper bound.
    """
    return [
        SimulationRecipe(
            name=config.recipe,
            version=RECIPE_VERSIONS[config.recipe],
            gene_efficiency_sigma=config.gene_efficiency_sigma,
            spill_fraction=config.spill_fraction,
            seed=seed,
        ),
        SimulationRecipe(
            name=CLEAN_RECIPE,
            version=RECIPE_VERSIONS[CLEAN_RECIPE],
            gene_efficiency_sigma=None,
            spill_fraction=0.0,
            seed=seed,
        ),
    ]


def gene_efficiency(n_genes: int, sigma: float | None, seed: int) -> np.ndarray:
    """Return per-gene detection efficiencies, LogNormal(0, sigma) / median.

    Args:
        n_genes: Genes.
        sigma: LogNormal sigma (``None``: all ones).
        seed: Seed (one draw per self-map, fixed across depths and cells, as
            E2's ``gene_efficiency.npy``).

    Returns:
        Efficiencies with median 1.
    """
    if sigma is None or n_genes == 0:
        return np.ones(n_genes, dtype=np.float64)
    rng = np.random.default_rng([int(seed), 0x6566])
    values = np.exp(rng.normal(0.0, float(sigma), n_genes))
    return np.asarray(values / np.median(values), dtype=np.float64)


def _thin_rows(
    matrix: sparse.csr_matrix,
    targets: np.ndarray,
    efficiency: np.ndarray,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    """Thin each row binomially to an expected total (E2 ``thin``).

    ``p_g = min(1, D * e_g / sum_g(x_g * e_g))`` per row, then
    ``Binomial(x_g, p_g)``; rows are never raised.
    """
    from scipy import sparse as sp

    work = sp.csr_matrix(matrix, dtype=np.float64, copy=True)
    rows = np.repeat(np.arange(work.shape[0]), np.diff(work.indptr))
    weighted = np.asarray(
        work.multiply(efficiency[None, :]).sum(axis=1), dtype=np.float64
    ).ravel()
    scale = np.asarray(targets, dtype=np.float64) / np.maximum(weighted, 1e-12)
    probability = np.clip(scale[rows] * efficiency[work.indices], 0.0, 1.0)
    counts = work.data.astype(np.int64)
    thinned = rng.binomial(counts, probability)
    result = sp.csr_matrix(
        (thinned.astype(np.float64), work.indices.copy(), work.indptr.copy()),
        shape=work.shape,
    )
    result.eliminate_zeros()
    return result


@dataclass
class HeldOutCells:
    """Held-out reference cells restricted to the panel (native counts).

    Attributes:
        counts: Cells x genes raw counts (CSR).
        genes: Gene IDs (columns).
        obs: One row per cell, indexed by cell id: the truth node per engine
            level (``truth__<level>``), ``truth_leaf`` (the composition key),
            ``spill_group`` (broad class; spill partners come from another),
            ``native_counts``.
    """

    counts: sparse.csr_matrix
    genes: list[str]
    obs: pd.DataFrame

    def __post_init__(self) -> None:
        n_cells, n_genes = self.counts.shape
        if n_cells != len(self.obs) or n_genes != len(self.genes):
            raise ResolvabilityError(
                f"test cells: counts {self.counts.shape} do not fit {len(self.obs)} "
                f"cells x {len(self.genes)} genes"
            )
        missing = [
            column
            for column in (TRUTH_LEAF_COLUMN, SPILL_GROUP_COLUMN)
            if column not in self.obs.columns
        ]
        if missing:
            raise ResolvabilityError(f"test cells lack the columns {missing}")
        if NATIVE_COUNTS_COLUMN not in self.obs.columns:
            self.obs[NATIVE_COUNTS_COLUMN] = np.asarray(self.counts.sum(axis=1)).ravel()

    @property
    def native_counts(self) -> np.ndarray:
        """Native panel counts per cell."""
        return np.asarray(self.obs[NATIVE_COUNTS_COLUMN], dtype=np.float64)

    def truth(self, engine_level: str) -> pd.Series:
        """Return the truth node of each cell at an engine level.

        Args:
            engine_level: A taxonomy level, e.g. ``CCN202210140_SUPC``.

        Returns:
            Truth labels indexed by cell id.

        Raises:
            ResolvabilityError: If the test cells carry no such truth.
        """
        column = f"{TRUTH_PREFIX}{engine_level}"
        if column not in self.obs.columns:
            raise ResolvabilityError(f"test cells carry no truth for {engine_level!r}")
        return self.obs[column].astype(object)

    def restrict_genes(self, genes: Sequence[str]) -> HeldOutCells:
        """Return the cells on a subset of their genes (order of ``genes``).

        Native counts are recomputed on the kept genes.

        Args:
            genes: Genes to keep (each must be a test-cell gene).

        Returns:
            The restricted cells.
        """
        position = {gene: index for index, gene in enumerate(self.genes)}
        missing = [gene for gene in genes if gene not in position]
        if missing:
            raise ResolvabilityError(f"test cells lack genes {missing[:10]}")
        columns = [position[gene] for gene in genes]
        counts = self.counts[:, columns].tocsr()
        obs = self.obs.copy()
        obs[NATIVE_COUNTS_COLUMN] = np.asarray(counts.sum(axis=1)).ravel()
        return HeldOutCells(counts=counts, genes=list(genes), obs=obs)


@dataclass
class SimulatedQuery:
    """Simulated cells of one recipe over the depth grid.

    Attributes:
        recipe: The recipe.
        counts: Simulated cells x genes counts (CSR).
        genes: Gene IDs.
        obs: Indexed by simulated cell id (``<cell id>|D<depth>``): ``cell_id``,
            ``depth``, ``partner_id`` (spill donor or empty), ``host_counts``,
            ``spill_counts``, ``total_counts``.
        n_by_depth: Test cells simulated at each depth (native counts >= D).
    """

    recipe: SimulationRecipe
    counts: sparse.csr_matrix
    genes: list[str]
    obs: pd.DataFrame
    n_by_depth: dict[int, int]


def simulated_id(cell_id: str, depth: int) -> str:
    """Return the id of a test cell thinned to a depth."""
    return f"{cell_id}|D{int(depth)}"


def thin_and_contaminate(
    test: HeldOutCells,
    depths: Sequence[int],
    recipe: SimulationRecipe,
) -> SimulatedQuery:
    """Simulate test cells at each grid depth with one recipe (§8.3 steps 2-3).

    Only cells whose native panel counts are at least ``D`` are thinned to
    ``D`` (thinning cannot raise counts). ``R1_contam_HO`` (E2
    ``research/insilico/02_make_queries.py``): per-gene efficiency applied
    inside the thinning, then ``spill_fraction * D`` counts thinned from a
    random cell of another spill group (broad class) whose native counts
    reach that amount; truth stays the host cell's. ``clean`` thins only.

    Args:
        test: Test cells.
        depths: Depth grid.
        recipe: Recipe.

    Returns:
        The simulated query.

    Raises:
        ResolvabilityError: If a spill recipe has cells of a single group.
    """
    from scipy import sparse as sp

    efficiency = gene_efficiency(
        len(test.genes), recipe.gene_efficiency_sigma, recipe.seed
    )
    rng = np.random.default_rng([int(recipe.seed), zlib.crc32(recipe.name.encode())])
    native = test.native_counts
    groups = test.obs[SPILL_GROUP_COLUMN].astype(str).to_numpy()
    cell_ids = test.obs.index.astype(str).to_numpy()
    counts = sp.csr_matrix(test.counts)
    blocks: list[sp.csr_matrix] = []
    frames: list[pd.DataFrame] = []
    n_by_depth: dict[int, int] = {}
    for depth in sorted(int(value) for value in depths):
        hosts = np.flatnonzero(native >= depth)
        n_by_depth[depth] = int(len(hosts))
        if len(hosts) == 0:
            continue
        host_counts = _thin_rows(
            counts[hosts], np.full(len(hosts), float(depth)), efficiency, rng
        )
        partner_ids = np.full(len(hosts), "", dtype=object)
        spill = sp.csr_matrix(host_counts.shape, dtype=np.float64)
        if recipe.spill_fraction > 0:
            amount = recipe.spill_fraction * depth
            donors = np.flatnonzero(native >= amount)
            partners = np.empty(len(hosts), dtype=np.int64)
            for group in np.unique(groups[hosts]):
                is_host = groups[hosts] == group
                candidates = donors[groups[donors] != group]
                if len(candidates) == 0:
                    raise ResolvabilityError(
                        f"no spill donor outside group {group!r} at depth {depth}"
                    )
                partners[is_host] = candidates[
                    rng.integers(0, len(candidates), int(is_host.sum()))
                ]
            spill = _thin_rows(
                counts[partners], np.full(len(hosts), amount), efficiency, rng
            )
            partner_ids = cell_ids[partners].astype(object)
        simulated = (host_counts + spill).tocsr()
        blocks.append(simulated)
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": cell_ids[hosts],
                    "depth": np.full(len(hosts), depth, dtype=np.int32),
                    "partner_id": partner_ids,
                    "host_counts": np.asarray(host_counts.sum(axis=1)).ravel(),
                    "spill_counts": np.asarray(spill.sum(axis=1)).ravel(),
                    "total_counts": np.asarray(simulated.sum(axis=1)).ravel(),
                },
                index=[simulated_id(cell_id, depth) for cell_id in cell_ids[hosts]],
            )
        )
    if blocks:
        matrix = sp.vstack(blocks).tocsr()
        obs = pd.concat(frames)
    else:
        matrix = sp.csr_matrix((0, len(test.genes)), dtype=np.float64)
        obs = pd.DataFrame(
            columns=[
                "cell_id",
                "depth",
                "partner_id",
                "host_counts",
                "spill_counts",
                "total_counts",
            ]
        )
    return SimulatedQuery(
        recipe=recipe,
        counts=matrix,
        genes=list(test.genes),
        obs=obs,
        n_by_depth=n_by_depth,
    )


def select_test_cells(
    obs: pd.DataFrame,
    *,
    stratum: str,
    max_per_stratum: int,
    n_max: int | None,
    seed: int,
    eligible: np.ndarray | None = None,
) -> pd.Index:
    """Select test cells stratified by type (§8.3 step 1).

    Each stratum gives at most ``max_per_stratum`` cells (all cells of rarer
    strata); when that exceeds ``n_max`` the per-stratum cap is lowered
    (water filling) until the total fits, so rare types keep every cell.

    Args:
        obs: Candidate cells (index = cell id).
        stratum: Column to stratify on (human supercluster).
        max_per_stratum: Largest share of one stratum (1,000).
        n_max: Largest total (``n_test_cells``), or ``None``.
        seed: Sampling seed.
        eligible: Optional boolean mask of usable candidates.

    Returns:
        Selected cell ids, in ``obs`` order.
    """
    frame = obs if eligible is None else obs[np.asarray(eligible, dtype=bool)]
    sizes = frame.groupby(stratum, observed=True).size()
    cap = int(max_per_stratum)
    if n_max is not None:
        while cap > 0 and int(np.minimum(sizes, cap).sum()) > n_max:
            cap -= 1
    rng = np.random.default_rng(int(seed))
    chosen: list[str] = []
    for _, group in frame.groupby(stratum, observed=True, sort=True):
        take = min(len(group), cap)
        if take == 0:
            continue
        picked = rng.choice(len(group), size=take, replace=False)
        chosen.extend(str(value) for value in group.index[np.sort(picked)])
    keep = set(chosen)
    return pd.Index([str(value) for value in obs.index if str(value) in keep])


def write_test_cells(test: HeldOutCells, directory: Path) -> dict[str, Any]:
    """Write test cells as ``test_cells.h5ad`` and ``test_cells.parquet``.

    Args:
        test: Test cells.
        directory: Bundle work directory.

    Returns:
        File names and counts for ``bundle.json``.
    """
    import anndata as ad

    directory.mkdir(parents=True, exist_ok=True)
    obs = test.obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    for column in obs.columns:
        if obs[column].dtype == object:
            obs[column] = obs[column].astype(str)
    adata = ad.AnnData(
        X=test.counts.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(test.genes, name="gene_id")),
    )
    adata.write_h5ad(directory / TEST_CELLS_FILE)
    obs.reset_index().to_parquet(directory / TEST_CELLS_OBS_FILE, index=False)
    return {
        "file": TEST_CELLS_FILE,
        "obs_file": TEST_CELLS_OBS_FILE,
        "n_cells": int(len(obs)),
        "n_genes": len(test.genes),
    }


def load_test_cells(directory: Path | str) -> HeldOutCells:
    """Read the test cells of a resolvability test-set bundle.

    Args:
        directory: Bundle directory holding ``test_cells.h5ad``.

    Returns:
        The test cells.
    """
    import anndata as ad
    from scipy import sparse as sp

    adata = ad.read_h5ad(Path(directory) / TEST_CELLS_FILE)
    obs = adata.obs.copy()
    for column in obs.columns:
        if isinstance(obs[column].dtype, pd.CategoricalDtype):
            obs[column] = obs[column].astype(str)
    obs.index = obs.index.astype(str)
    return HeldOutCells(
        counts=sp.csr_matrix(adata.X).astype(np.float64),
        genes=[str(gene) for gene in adata.var_names],
        obs=obs,
    )


# --------------------------------------------------------------------------
# Levels and calls


@dataclass(frozen=True)
class LevelMeta:
    """What the decisions need to know about one level (JSON-serialisable).

    Attributes:
        level: Level name (``broad``, ``supercluster``, ``subclass``, ...).
        engine_level: Taxonomy level the calls come from.
        role: ``broad`` (refusal test), ``leaf`` (broad-only test), ``fine``
            (report-only, OD-E4) or another descriptive role.
        default_threshold: The v1 raw threshold (raise-only floor).
        base_target: Precision target of real-data-validated panels.
        floor_level: Level of the packaged floor table (``None``: hard floor).
    """

    level: str
    engine_level: str
    role: LevelRole
    default_threshold: float
    base_target: float
    floor_level: str | None

    def to_json(self) -> dict[str, Any]:
        """Return the metadata as JSON."""
        return {
            "level": self.level,
            "engine_level": self.engine_level,
            "role": self.role,
            "default_threshold": self.default_threshold,
            "base_target": self.base_target,
            "floor_level": self.floor_level,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> LevelMeta:
        """Rebuild the metadata from ``to_json`` output."""
        return cls(
            level=str(payload["level"]),
            engine_level=str(payload["engine_level"]),
            role=payload["role"],
            default_threshold=float(payload["default_threshold"]),
            base_target=float(payload["base_target"]),
            floor_level=payload.get("floor_level"),
        )


CallFunction = Callable[[pd.DataFrame], pd.DataFrame]
TruthFunction = Callable[[HeldOutCells], pd.DataFrame]


@dataclass(frozen=True)
class LevelSpec:
    """How one level's calls and truths are read (see ``level_specs_*``).

    Attributes:
        meta: The level's decision metadata.
        calls: MMC tidy table -> per simulated cell ``call``, ``bp``,
            ``corr``, ``parent`` (class key of the call; ``None`` excludes
            the call: sinks, region-implausible nodes, not applicable).
        truth: Test cells -> per test cell ``truth`` and ``truth_parent``.
    """

    meta: LevelMeta
    calls: CallFunction
    truth: TruthFunction


def _tidy_level(tidy: pd.DataFrame, engine_level: str) -> pd.DataFrame:
    from merxen.annotation.mapmycells_engine import level_frame

    return level_frame(tidy, engine_level)


def node_level_calls(
    engine_level: str, parent_of: Mapping[str, str | None]
) -> CallFunction:
    """Return a call reader for a level that is the engine level's node itself.

    Args:
        engine_level: Taxonomy level.
        parent_of: Class key per node label (missing / ``None``: excluded).

    Returns:
        The reader.
    """

    def read(tidy: pd.DataFrame) -> pd.DataFrame:
        frame = _tidy_level(tidy, engine_level)
        nodes = frame["assignment"].astype(object).to_numpy()
        return pd.DataFrame(
            {
                "call": nodes,
                "bp": frame["bp"].to_numpy(np.float64),
                "corr": frame["avg_correlation"].to_numpy(np.float64),
                "parent": [parent_of.get(str(node)) for node in nodes],
            },
            index=frame.index.astype(str),
        )

    return read


def group_level_calls(
    engine_level: str,
    group_of: Mapping[str, str | None],
    parent_of: Mapping[str, str | None],
) -> CallFunction:
    """Return a call reader for a level aggregated from an engine level.

    The coarse probability of the assigned node's group is its bp plus that
    of same-group runner-ups (E1 / E2; ``aggregate_parent_probability``).

    Args:
        engine_level: Taxonomy level.
        group_of: Group per node label (``None``: counts toward no group).
        parent_of: Class key per node label.

    Returns:
        The reader.
    """
    from merxen.annotation.mapmycells_engine import aggregate_parent_probability

    def read(tidy: pd.DataFrame) -> pd.DataFrame:
        frame = _tidy_level(tidy, engine_level)
        aggregated = aggregate_parent_probability(frame, group_of)
        nodes = frame["assignment"].astype(object).to_numpy()
        parents = [
            parent_of.get(str(node)) if group is not None else None
            for node, group in zip(nodes, aggregated.classes, strict=True)
        ]
        return pd.DataFrame(
            {
                "call": aggregated.classes,
                "bp": aggregated.probability,
                "corr": frame["avg_correlation"].to_numpy(np.float64),
                "parent": parents,
            },
            index=frame.index.astype(str),
        )

    return read


def mapped_truth(
    engine_level: str,
    value_of: Mapping[str, str | None] | None,
    parent_of: Mapping[str, str | None],
    *,
    missing_value: str | None = None,
    parent_level: str | None = None,
) -> TruthFunction:
    """Return a truth reader mapping each test cell's truth node.

    Args:
        engine_level: Truth level (``truth__<level>`` column).
        value_of: Truth value per truth node (``None``: the node itself).
        parent_of: Truth class key per truth node (of ``parent_level``).
        missing_value: Truth value of nodes ``value_of`` maps to ``None``
            (e.g. ``"not_neuron"`` for NT).
        parent_level: Truth level ``parent_of`` is keyed by (default
            ``engine_level``), e.g. the supercluster of a cluster truth.

    Returns:
        The reader.
    """

    def read(test: HeldOutCells) -> pd.DataFrame:
        nodes = test.truth(engine_level).astype(str).to_numpy()
        parent_nodes = (
            nodes
            if parent_level is None
            else test.truth(parent_level).astype(str).to_numpy()
        )
        if value_of is None:
            values: list[str | None] = [str(node) for node in nodes]
        else:
            values = [
                value_of.get(node) if value_of.get(node) is not None else missing_value
                for node in nodes
            ]
        return pd.DataFrame(
            {
                "truth": values,
                "truth_parent": [parent_of.get(node) for node in parent_nodes],
            },
            index=test.obs.index.astype(str),
        )

    return read


# --------------------------------------------------------------------------
# Engine-specific levels (WHB, SEA-AD, WMB)

WHB_SUPC: Final = "CCN202210140_SUPC"
WHB_CLUS: Final = "CCN202210140_CLUS"
SEAAD_CLASS: Final = "CCN20260630_LEVEL_0"
SEAAD_SUBCLASS: Final = "CCN20260630_LEVEL_1"
SEAAD_SUPERTYPE: Final = "CCN20260630_LEVEL_2"
WMB_CLAS: Final = "CCN20230722_CLAS"
WMB_SUBC: Final = "CCN20230722_SUBC"
WMB_SUPT: Final = "CCN20230722_SUPT"
WMB_CLUS: Final = "CCN20230722_CLUS"
NOT_NEURON: Final = "not_neuron"


def _vocab_rows(vocab: pd.DataFrame, level: str) -> pd.DataFrame:
    frame = vocab.reset_index(drop=True)
    return frame[frame["level"].astype(str) == level]


def _vocab_values(rows: pd.DataFrame, column: str) -> dict[str, str | None]:
    if column not in rows.columns:
        return {str(node): None for node in rows["node"]}
    return {
        str(node): _clean_label(value)
        for node, value in zip(rows["node"], rows[column], strict=True)
    }


def _vocab_flags(rows: pd.DataFrame, column: str, default: bool) -> dict[str, bool]:
    return {
        node: default if value is None else value.strip().lower() == "true"
        for node, value in _vocab_values(rows, column).items()
    }


def whb_level_specs(
    vocab: pd.DataFrame,
    thresholds: AnnotationThresholds,
    *,
    region: str = "frontal_cortex",
    include_fine: bool = True,
) -> list[LevelSpec]:
    """Return the WHB levels: lineage, broad, NT, supercluster (+ cluster).

    Coarse levels aggregate the supercluster bp over the assigned node's
    group (E1 / E2). Calls to sinks and region-implausible superclusters are
    excluded (never confident in production, §5.2). Class keys are the E2
    floor classes (neurons by NT; ``COP`` at supercluster level).

    Args:
        vocab: The mapped bundle's ``vocab_snapshot.csv`` (``MmcBundle.vocab``).
        thresholds: Raw thresholds and targets.
        region: Anatomical region of the plausibility column.
        include_fine: Add the report-only cluster level (OD-E4).

    Returns:
        The level specs.
    """
    from merxen.annotation.vocab import (
        COP_SUPERCLUSTER,
        NEURONS,
        UNASSIGNED_LABEL,
        human_floor_class,
    )

    rows = _vocab_rows(vocab, WHB_SUPC)
    broad = _vocab_values(rows, "broad_class")
    nt = _vocab_values(rows, "nt")
    lineage = _vocab_values(rows, "lineage")
    names = _vocab_values(rows, "key_name")
    sink = _vocab_flags(rows, "sink", False)
    plausible = _vocab_flags(rows, f"region_plausible_{region}", True)
    broad_group = {
        node: (value if value not in (None, UNASSIGNED_LABEL) else None)
        for node, value in broad.items()
    }
    lineage_group = {
        node: (value if value not in (None, UNASSIGNED_LABEL) else None)
        for node, value in lineage.items()
    }
    nt_group = {
        node: (nt[node] or "Other") if broad_group[node] == NEURONS else None
        for node in broad
    }
    truth_broad_class = {
        node: human_floor_class(broad_group[node], nt[node]) for node in broad
    }
    truth_supc_class = {
        node: "COP" if names.get(node) == COP_SUPERCLUSTER else value
        for node, value in truth_broad_class.items()
    }
    excluded = {node for node in broad if sink[node] or not plausible[node]}
    call_broad_class = {
        node: None if node in excluded else value
        for node, value in truth_broad_class.items()
    }
    call_supc_class = {
        node: None if node in excluded else value
        for node, value in truth_supc_class.items()
    }
    specs = [
        LevelSpec(
            meta=LevelMeta(
                "lineage",
                WHB_SUPC,
                "lineage",
                thresholds.whb_broad,
                thresholds.target_lineage,
                None,
            ),
            calls=group_level_calls(WHB_SUPC, lineage_group, call_broad_class),
            truth=mapped_truth(WHB_SUPC, lineage_group, truth_broad_class),
        ),
        LevelSpec(
            meta=LevelMeta(
                "broad",
                WHB_SUPC,
                "broad",
                thresholds.whb_broad,
                thresholds.target_broad,
                "broad",
            ),
            calls=group_level_calls(WHB_SUPC, broad_group, call_broad_class),
            truth=mapped_truth(WHB_SUPC, broad_group, truth_broad_class),
        ),
        LevelSpec(
            meta=LevelMeta(
                "nt",
                WHB_SUPC,
                "nt",
                thresholds.whb_broad,
                thresholds.target_nt,
                "broad",
            ),
            calls=group_level_calls(WHB_SUPC, nt_group, call_broad_class),
            truth=mapped_truth(
                WHB_SUPC, nt_group, truth_broad_class, missing_value=NOT_NEURON
            ),
        ),
        LevelSpec(
            meta=LevelMeta(
                "supercluster",
                WHB_SUPC,
                "leaf",
                thresholds.whb_supercluster,
                thresholds.target_supercluster,
                "supercluster",
            ),
            calls=node_level_calls(WHB_SUPC, call_supc_class),
            truth=mapped_truth(WHB_SUPC, None, truth_supc_class),
        ),
    ]
    if include_fine:
        cluster_rows = _vocab_rows(vocab, WHB_CLUS)
        cluster_key = _vocab_values(cluster_rows, "key_label")
        cluster_class = {
            node: call_supc_class.get(key) if key is not None else None
            for node, key in cluster_key.items()
        }
        specs.append(
            LevelSpec(
                meta=LevelMeta(
                    "cluster",
                    WHB_CLUS,
                    "fine",
                    thresholds.whb_supercluster,
                    thresholds.target_supercluster,
                    "supercluster",
                ),
                calls=node_level_calls(WHB_CLUS, cluster_class),
                truth=mapped_truth(
                    WHB_CLUS, None, truth_supc_class, parent_level=WHB_SUPC
                ),
            )
        )
    return specs


def seaad_level_specs(thresholds: AnnotationThresholds) -> list[LevelSpec]:
    """Return SEA-AD's 7-class level measured on WHB test cells (§8.3 step 1).

    SEA-AD's broad call and probability follow E2's definition
    (``shadow.seaad_broad_calls`` with the class level; the v1
    ``seaad_broad`` threshold 0.68 was derived on it); truth is the WHB test
    cell's broad class through the WHB vocab. SEA-AD subclass has no
    in-silico truth (E2: real data only) and is not tabulated.

    Args:
        thresholds: Raw thresholds and targets.

    Returns:
        The level spec.
    """
    from merxen.annotation.mapmycells_engine import level_frame
    from merxen.annotation.shadow import seaad_broad_calls
    from merxen.annotation.vocab import (
        UNASSIGNED_LABEL,
        human_floor_class,
        load_vocab,
    )

    sea_vocab = load_vocab("seaad_mr_subclass")
    whb_vocab = load_vocab("whb_supercluster")

    def calls(tidy: pd.DataFrame) -> pd.DataFrame:
        subclass = level_frame(tidy, SEAAD_SUBCLASS)
        supertype = level_frame(tidy, SEAAD_SUPERTYPE)
        class_level = level_frame(tidy, SEAAD_CLASS)
        broad = seaad_broad_calls(subclass, supertype, class_level=class_level)
        nts = [
            sea_vocab.nt(str(name)) if str(name) in sea_vocab else None
            for name in broad["subclass"].astype(object)
        ]
        labels = broad["broad"].astype(object).to_numpy()
        parents = [
            None if label == UNASSIGNED_LABEL else human_floor_class(str(label), value)
            for label, value in zip(labels, nts, strict=True)
        ]
        return pd.DataFrame(
            {
                "call": [
                    None if label == UNASSIGNED_LABEL else label for label in labels
                ],
                "bp": broad["broad_raw"].to_numpy(np.float64),
                "corr": subclass["avg_correlation"]
                .reindex(broad.index)
                .to_numpy(np.float64),
                "parent": parents,
            },
            index=broad.index.astype(str),
        )

    label_to_name = whb_vocab.label_to_name()
    truth_broad: dict[str, str | None] = {}
    truth_class: dict[str, str | None] = {}
    for label, name in label_to_name.items():
        value = whb_vocab.broad_class(name)
        truth_broad[label] = None if value == UNASSIGNED_LABEL else value
        truth_class[label] = human_floor_class(truth_broad[label], whb_vocab.nt(name))
    return [
        LevelSpec(
            meta=LevelMeta(
                "broad",
                SEAAD_SUBCLASS,
                "broad",
                thresholds.seaad_broad,
                thresholds.target_broad,
                "broad",
            ),
            calls=calls,
            truth=mapped_truth(WHB_SUPC, truth_broad, truth_class),
        )
    ]


def wmb_level_specs(
    vocab: pd.DataFrame,
    thresholds: AnnotationThresholds,
    *,
    supertype_of_cluster: Mapping[str, str] | None = None,
) -> list[LevelSpec]:
    """Return the WMB levels: broad, class, NT, subclass (+ supertype).

    Class keys are WMB class names (the floor classes). The report-only
    supertype level (OD-E4) aggregates the cluster call to its supertype,
    because the mapping drops SUPT.

    Args:
        vocab: The mapped bundle's ``vocab_snapshot.csv``.
        thresholds: Raw thresholds and targets.
        supertype_of_cluster: Supertype per cluster label (from the full
            mapping precompute tree); ``None`` omits the supertype level.

    Returns:
        The level specs.
    """
    from merxen.annotation.vocab import NEURONS, UNASSIGNED_LABEL

    class_rows = _vocab_rows(vocab, WMB_CLAS)
    class_name = _vocab_values(class_rows, "key_name")
    broad = _vocab_values(class_rows, "broad_class")
    nt = _vocab_values(class_rows, "nt")
    broad_group = {
        node: (value if value not in (None, UNASSIGNED_LABEL) else None)
        for node, value in broad.items()
    }
    nt_group = {
        node: (nt[node] or "Other") if broad_group[node] == NEURONS else None
        for node in broad
    }
    subclass_rows = _vocab_rows(vocab, WMB_SUBC)
    subclass_class = _vocab_values(subclass_rows, "key_name")
    specs = [
        LevelSpec(
            meta=LevelMeta(
                "broad",
                WMB_CLAS,
                "broad",
                thresholds.wmb_class,
                thresholds.target_broad,
                "class",
            ),
            calls=group_level_calls(WMB_CLAS, broad_group, class_name),
            truth=mapped_truth(WMB_CLAS, broad_group, class_name),
        ),
        LevelSpec(
            meta=LevelMeta(
                "class",
                WMB_CLAS,
                "class",
                thresholds.wmb_class,
                thresholds.target_class,
                "class",
            ),
            calls=node_level_calls(WMB_CLAS, class_name),
            truth=mapped_truth(WMB_CLAS, None, class_name),
        ),
        LevelSpec(
            meta=LevelMeta(
                "nt",
                WMB_CLAS,
                "nt",
                thresholds.wmb_class,
                thresholds.target_nt,
                "class",
            ),
            calls=group_level_calls(WMB_CLAS, nt_group, class_name),
            truth=mapped_truth(
                WMB_CLAS, nt_group, class_name, missing_value=NOT_NEURON
            ),
        ),
        LevelSpec(
            meta=LevelMeta(
                "subclass",
                WMB_SUBC,
                "leaf",
                thresholds.wmb_subclass,
                thresholds.target_subclass,
                "subclass",
            ),
            calls=node_level_calls(WMB_SUBC, subclass_class),
            truth=mapped_truth(WMB_SUBC, None, class_name, parent_level=WMB_CLAS),
        ),
    ]
    if supertype_of_cluster:
        cluster_rows = _vocab_rows(vocab, WMB_CLUS)
        cluster_class = _vocab_values(cluster_rows, "key_name")
        specs.append(
            LevelSpec(
                meta=LevelMeta(
                    "supertype",
                    WMB_CLUS,
                    "fine",
                    thresholds.wmb_subclass,
                    thresholds.target_subclass,
                    "subclass",
                ),
                calls=group_level_calls(
                    WMB_CLUS, dict(supertype_of_cluster), cluster_class
                ),
                truth=mapped_truth(WMB_SUPT, None, class_name, parent_level=WMB_CLAS),
            )
        )
    return specs


def level_cells(
    tidy: pd.DataFrame,
    query: SimulatedQuery,
    test: HeldOutCells,
    specs: Sequence[LevelSpec],
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Return the per simulated cell x level table of one mapped recipe.

    Args:
        tidy: MMC tidy table of the simulated query.
        query: The simulated query.
        test: The test cells.
        specs: Levels to read.
        seed: Mapping seed of the run (recorded).

    Returns:
        ``CELLS_COLUMNS`` rows; ``correct`` is ``call == truth``.
    """
    sim = query.obs
    halves = pd.Series(
        cell_split_half(test.obs.index), index=test.obs.index.astype(str)
    )
    leaves = test.obs[TRUTH_LEAF_COLUMN].astype(str)
    leaves.index = leaves.index.astype(str)
    parts: list[pd.DataFrame] = []
    for spec in specs:
        calls = spec.calls(tidy).reindex(sim.index)
        truth = spec.truth(test).reindex(sim["cell_id"].astype(str).to_numpy())
        call_values = calls["call"].astype(object).to_numpy()
        truth_values = truth["truth"].astype(object).to_numpy()
        correct = np.array(
            [
                call is not None
                and not (isinstance(call, float) and math.isnan(call))
                and value is not None
                and str(call) == str(value)
                for call, value in zip(call_values, truth_values, strict=True)
            ],
            dtype=bool,
        )
        cell_ids = sim["cell_id"].astype(str).to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "recipe": query.recipe.name,
                    "seed": np.int32(seed),
                    "level": spec.meta.level,
                    "sim_id": sim.index.astype(str),
                    "cell_id": cell_ids,
                    "depth": sim["depth"].to_numpy(np.int32),
                    "half": halves.reindex(cell_ids).to_numpy(np.int8),
                    "parent": calls["parent"].astype(object).to_numpy(),
                    "call": call_values,
                    "bp": calls["bp"].to_numpy(np.float64),
                    "corr": calls["corr"].to_numpy(np.float64),
                    "truth": truth_values,
                    "truth_parent": truth["truth_parent"].astype(object).to_numpy(),
                    TRUTH_LEAF_COLUMN: leaves.reindex(cell_ids).to_numpy(),
                    "correct": correct,
                    "total_counts": sim["total_counts"].to_numpy(np.float64),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=list(CELLS_COLUMNS))
    return pd.concat(parts, ignore_index=True)


def _clean_label(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    return None if text in {"", "nan", "None", "<NA>"} else text


def coerce_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """Return a cells table with parquet-friendly dtypes (categories, floats).

    Args:
        cells: ``level_cells`` output (possibly concatenated).

    Returns:
        A copy with category label columns, float32 bp / corr / counts.
    """
    frame = cells.copy()
    for column in ("recipe", "level", "parent", "call", "truth", "truth_parent"):
        frame[column] = pd.Categorical(
            [_clean_label(value) for value in frame[column].astype(object)]
        )
    frame[TRUTH_LEAF_COLUMN] = frame[TRUTH_LEAF_COLUMN].astype("category")
    for column in ("bp", "corr", "total_counts"):
        frame[column] = frame[column].astype(np.float32)
    frame["depth"] = frame["depth"].astype(np.int32)
    frame["half"] = frame["half"].astype(np.int8)
    frame["seed"] = frame["seed"].astype(np.int32)
    frame["correct"] = frame["correct"].astype(bool)
    return frame


# --------------------------------------------------------------------------
# Decisions


@dataclass(frozen=True)
class RuleSettings:
    """Settings of the local rule, emission and trust tests (§3.7, §8.2, §8.3).

    Attributes:
        min_cells_per_bin: Test cells of a class a depth bin needs (D_max).
        min_confident_n: Confident calls an emitted bin needs (check half).
        wilson_margin: The Wilson bound may sit this far below the target.
        min_coverage: Coverage an emitted bin needs.
        threshold_cap: Largest local threshold.
        split_halves: Fit on one half, check on the other.
        provisional_margin: Target margin at >= 60 counts (provisional).
        provisional_margin_below60: Target margin below 60 counts.
        provisional_cap: Cap of margin-raised targets.
        below_counts: The margin switch depth (60).
        provisional_mouse_subclass_floor: Mouse subclass floor while
            provisional.
        trust_max_depth: Deepest depth the trust tests consider (250).
        broad_only_min_class_share: Share of classes whose leaf must be
            resolvable at some depth, else ``broad_only``.
        hard_floor: The hard count floor (``min_counts``).
    """

    min_cells_per_bin: int = 50
    min_confident_n: int = 50
    wilson_margin: float = 0.02
    min_coverage: float = 0.2
    threshold_cap: float = 0.99
    split_halves: bool = True
    provisional_margin: float = 0.05
    provisional_margin_below60: float = 0.10
    provisional_cap: float = 0.97
    below_counts: int = 60
    provisional_mouse_subclass_floor: int = 60
    trust_max_depth: int = 250
    broad_only_min_class_share: float = 0.5
    hard_floor: int = 10

    @classmethod
    def from_config(
        cls,
        resolvability: AnnotationResolvabilityConfig,
        thresholds: AnnotationThresholds,
        *,
        trust_max_depth: int = 250,
        broad_only_min_class_share: float = 0.5,
        hard_floor: int = 10,
    ) -> RuleSettings:
        """Return the settings of an annotation config.

        Args:
            resolvability: ``AnnotationConfig.resolvability``.
            thresholds: ``AnnotationConfig.thresholds``.
            trust_max_depth: ``AnnotationPanelConfig.trust_max_depth``.
            broad_only_min_class_share: The panel config's broad-only share.
            hard_floor: The hard floor (``min_counts``).

        Returns:
            The settings.
        """
        return cls(
            min_cells_per_bin=resolvability.min_cells_per_bin,
            min_confident_n=resolvability.min_confident_n,
            wilson_margin=resolvability.wilson_margin,
            min_coverage=resolvability.min_coverage,
            threshold_cap=resolvability.threshold_cap,
            split_halves=resolvability.split_halves,
            provisional_margin=thresholds.provisional_target_margin,
            provisional_margin_below60=thresholds.provisional_target_margin_below60,
            provisional_cap=thresholds.provisional_target_cap,
            below_counts=thresholds.second_vote_below_counts,
            provisional_mouse_subclass_floor=thresholds.provisional_mouse_subclass_floor,
            trust_max_depth=trust_max_depth,
            broad_only_min_class_share=broad_only_min_class_share,
            hard_floor=hard_floor,
        )

    def target(self, regime: Regime, base_target: float, depth: int) -> float:
        """Return a regime's precision target at a depth bin.

        Args:
            regime: Decision regime.
            base_target: Level target of real-data-validated panels.
            depth: Depth bin.

        Returns:
            The base target (``validated``, ``trust``) or the base target +
            the provisional margin (+0.10 below 60 counts), capped.
        """
        if regime != "provisional":
            return base_target
        margin = (
            self.provisional_margin_below60
            if depth < self.below_counts
            else self.provisional_margin
        )
        return min(base_target + margin, self.provisional_cap)

    def to_json(self) -> dict[str, Any]:
        """Return the settings as JSON."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class CheckStats:
    """The check-half statistics of one threshold in one bin."""

    threshold: float | None
    n_called: int
    n_confident: int
    n_effective: float
    precision: float
    wilson_lb: float
    coverage: float


def check_threshold(
    bp: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray,
    threshold: float | None,
) -> CheckStats:
    """Return precision, Wilson bound and coverage of the calls a threshold keeps.

    Args:
        bp: Check-half bp values of the bin's calls.
        correct: Their correctness.
        weights: Their weights.
        threshold: The threshold (``None``: nothing is confident).

    Returns:
        The statistics (Wilson bound on the Kish effective n).
    """
    n_called = int(len(bp))
    if threshold is None or n_called == 0:
        return CheckStats(threshold, n_called, 0, 0.0, math.nan, math.nan, 0.0)
    accepted = meets_threshold(bp, threshold)
    accepted_weights = weights[accepted]
    total = float(accepted_weights.sum())
    called_total = float(weights.sum())
    if total <= 0:
        return CheckStats(threshold, n_called, 0, 0.0, math.nan, math.nan, 0.0)
    precision = float((accepted_weights * correct[accepted]).sum()) / total
    n_effective = kish_effective_n(accepted_weights)
    return CheckStats(
        threshold=threshold,
        n_called=n_called,
        n_confident=int(accepted.sum()),
        n_effective=n_effective,
        precision=precision,
        wilson_lb=wilson_lower_bound(precision, n_effective),
        coverage=total / called_total if called_total > 0 else 0.0,
    )


def _rule_pass(stats: CheckStats, target: float, settings: RuleSettings) -> str | None:
    """Return why a checked threshold fails the emission rule (``None``: passes)."""
    if stats.threshold is None:
        return "no_local_threshold"
    if stats.n_confident < settings.min_confident_n:
        return "too_few_confident_calls"
    if not stats.wilson_lb >= target - settings.wilson_margin - _TOLERANCE:
        return "wilson_bound_below_target"
    if stats.coverage < settings.min_coverage - _TOLERANCE:
        return "coverage_below_minimum"
    return None


def class_test_counts(cells: pd.DataFrame) -> pd.DataFrame:
    """Return test cells per (level, truth class, depth) of the decision rows.

    Args:
        cells: Decision-recipe rows of the cells table.

    Returns:
        ``level``, ``class``, ``depth``, ``n_test`` (distinct test cells).
    """
    frame = cells[cells["truth_parent"].notna()]
    counts = (
        frame.groupby(["level", "truth_parent", "depth"], observed=True)["cell_id"]
        .nunique()
        .reset_index()
        .rename(columns={"truth_parent": "class", "cell_id": "n_test"})
    )
    counts["class"] = counts["class"].astype(str)
    counts["level"] = counts["level"].astype(str)
    return counts


def d_max_table(
    n_test: pd.DataFrame, *, min_cells_per_bin: int
) -> dict[tuple[str, str], int | None]:
    """Return ``D_max`` per (level, class): the deepest depth with enough test cells.

    Args:
        n_test: ``class_test_counts`` output.
        min_cells_per_bin: Test cells a bin needs (50).

    Returns:
        ``D_max`` per (level, class), ``None`` when no bin qualifies.
    """
    result: dict[tuple[str, str], int | None] = {}
    for (level, cls), group in n_test.groupby(["level", "class"], observed=True):
        qualified = group[group["n_test"] >= min_cells_per_bin]["depth"]
        result[(str(level), str(cls))] = (
            int(qualified.max()) if len(qualified) else None
        )
    return result


def decide(
    cells: pd.DataFrame,
    levels: Sequence[LevelMeta],
    depths: Sequence[int],
    settings: RuleSettings,
    *,
    weights: np.ndarray | None = None,
    recipe: str = DECISION_RECIPE,
    seed: int = 0,
) -> pd.DataFrame:
    """Return the per (regime, level, class, depth) decisions with D_max inheritance.

    For each bin with ``depth <= D_max(class)``: the isotonic fit on the fit
    half, the local thresholds for the base and provisional targets, and the
    check-half statistics at the applied threshold (``validated``: the
    default; ``provisional`` / ``trust``: ``t*``). Deeper bins inherit the
    ``D_max`` bin's decision (``extrapolated``); classes without ``D_max``
    are never emitted.

    Args:
        cells: The cells table (``level_cells`` rows).
        levels: Level metadata (defaults, targets, roles).
        depths: The depth grid.
        settings: Rule settings.
        weights: Optional per-row weights (composition reweighting, RESOLVE).
        recipe: Recipe the decisions use.
        seed: Mapping seed the decisions use.

    Returns:
        One row per (regime, level, class, depth): ``status`` (``emitted`` or
        ``not_resolvable``), ``threshold`` (applied), ``t_star``,
        ``would_raise``, statistics, ``target``, ``d_max``, ``extrapolated``,
        ``reason``.
    """
    row_weights = (
        np.ones(len(cells), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if len(row_weights) != len(cells):
        raise ResolvabilityError("weights must have one value per cells row")
    selected = (cells["recipe"].astype(str) == recipe).to_numpy() & (
        cells["seed"].to_numpy() == seed
    )
    frame = cells[selected].copy()
    frame["_weight"] = row_weights[selected]
    grid = sorted(int(depth) for depth in depths)
    n_test = class_test_counts(frame)
    dmax = d_max_table(n_test, min_cells_per_bin=settings.min_cells_per_bin)
    n_test_index = {
        (str(row.level), str(row["class"]), int(row.depth)): int(row.n_test)
        for _, row in n_test.iterrows()
    }
    records: list[dict[str, Any]] = []
    for meta in levels:
        level_rows = frame[frame["level"].astype(str) == meta.level]
        called = level_rows[level_rows["parent"].notna()]
        groups = {
            (str(cls), int(depth)): group
            for (cls, depth), group in called.groupby(
                [called["parent"].astype(str), "depth"], observed=True
            )
        }
        classes = sorted(
            {str(cls) for cls in called["parent"].dropna().astype(str)}
            | {cls for (level, cls) in dmax if level == meta.level}
        )
        for cls in classes:
            class_dmax = dmax.get((meta.level, cls))
            direct: dict[tuple[Regime, int], dict[str, Any]] = {}
            for depth in grid:
                group = groups.get((cls, depth))
                base = {
                    "level": meta.level,
                    "class": cls,
                    "depth": depth,
                    "n_test": n_test_index.get((meta.level, cls, depth), 0),
                    "d_max": class_dmax,
                    "default_threshold": meta.default_threshold,
                }
                if class_dmax is None or depth > class_dmax:
                    continue
                bin_records = _bin_decisions(group, meta, depth, settings, base)
                for record in bin_records:
                    direct[(record["regime"], depth)] = record
            for depth in grid:
                for regime in REGIMES:
                    if class_dmax is None:
                        records.append(
                            {
                                "regime": regime,
                                "level": meta.level,
                                "class": cls,
                                "depth": depth,
                                "n_test": n_test_index.get((meta.level, cls, depth), 0),
                                "d_max": None,
                                "default_threshold": meta.default_threshold,
                                "target": settings.target(
                                    regime, meta.base_target, depth
                                ),
                                "status": STATUS_NOT_RESOLVABLE,
                                "extrapolated": False,
                                "reason": "too_few_test_cells",
                            }
                        )
                    elif depth <= class_dmax:
                        records.append(direct[(regime, depth)])
                    else:
                        inherited = dict(direct[(regime, class_dmax)])
                        inherited.update(
                            depth=depth,
                            n_test=n_test_index.get((meta.level, cls, depth), 0),
                            extrapolated=True,
                            inherited_from=class_dmax,
                        )
                        records.append(inherited)
    columns = [
        "regime",
        "level",
        "class",
        "depth",
        "status",
        "threshold",
        "t_star",
        "would_raise",
        "target",
        "default_threshold",
        "n_test",
        "n_called",
        "n_fit",
        "n_confident",
        "n_effective",
        "precision",
        "wilson_lb",
        "coverage",
        "g_at_default",
        "precision_at_default",
        "d_max",
        "extrapolated",
        "inherited_from",
        "reason",
    ]
    table = pd.DataFrame.from_records(records)
    for column in columns:
        if column not in table.columns:
            table[column] = np.nan if column not in {"status", "reason"} else None
    table = table[columns]
    table["extrapolated"] = _bool_column(table["extrapolated"])
    table["would_raise"] = _bool_column(table["would_raise"])
    return table


def _bool_column(values: pd.Series) -> pd.Series:
    """Return a boolean column with missing values as ``False``."""
    return pd.Series(
        [
            bool(value) if isinstance(value, bool | np.bool_) else False
            for value in values
        ],
        index=values.index,
        dtype=bool,
    )


def _bin_decisions(
    group: pd.DataFrame | None,
    meta: LevelMeta,
    depth: int,
    settings: RuleSettings,
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    """Decide one (level, class, depth) bin under every regime."""
    if group is None or len(group) == 0:
        return [
            {
                **base,
                "regime": regime,
                "target": settings.target(regime, meta.base_target, depth),
                "status": STATUS_NOT_RESOLVABLE,
                "extrapolated": False,
                "n_called": 0,
                "n_fit": 0,
                "reason": "no_calls",
            }
            for regime in REGIMES
        ]
    bp = group["bp"].to_numpy(np.float64)
    correct = group["correct"].to_numpy(bool).astype(np.float64)
    weights = group["_weight"].to_numpy(np.float64)
    half = group["half"].to_numpy(np.int8)
    if settings.split_halves:
        fit_mask = half == 0
        check_mask = half == 1
    else:
        fit_mask = np.ones(len(bp), dtype=bool)
        check_mask = fit_mask
    fit_mask &= np.isfinite(bp) & (weights > 0)
    fit: IsotonicFit | None = None
    if int(fit_mask.sum()) >= settings.min_cells_per_bin:
        fit = isotonic_fit(bp[fit_mask], correct[fit_mask], weights[fit_mask])
    check_bp = bp[check_mask]
    check_correct = correct[check_mask]
    check_weights = weights[check_mask]
    at_default = check_threshold(
        check_bp, check_correct, check_weights, meta.default_threshold
    )
    g_default = (
        float(fit.predict(meta.default_threshold)) if fit is not None else math.nan
    )
    records = []
    for regime in REGIMES:
        target = settings.target(regime, meta.base_target, depth)
        t_star = local_threshold(
            fit,
            default=meta.default_threshold,
            target=target,
            cap=settings.threshold_cap,
        )
        if regime == "validated":
            applied: float | None = meta.default_threshold
            stats = at_default
        else:
            applied = t_star
            stats = check_threshold(check_bp, check_correct, check_weights, t_star)
        reason = _rule_pass(stats, target, settings)
        if fit is None and regime != "validated":
            reason = "too_few_fit_cells"
        records.append(
            {
                **base,
                "regime": regime,
                "target": target,
                "threshold": applied,
                "t_star": t_star,
                "would_raise": bool(
                    t_star is None or t_star > meta.default_threshold + _TOLERANCE
                ),
                "n_called": stats.n_called,
                "n_fit": int(fit_mask.sum()),
                "n_confident": stats.n_confident,
                "n_effective": stats.n_effective,
                "precision": stats.precision,
                "wilson_lb": stats.wilson_lb,
                "coverage": stats.coverage,
                "g_at_default": g_default,
                "precision_at_default": at_default.precision,
                "status": STATUS_EMITTED if reason is None else STATUS_NOT_RESOLVABLE,
                "extrapolated": False,
                "reason": reason,
            }
        )
    return records


def emission_lookup(
    decisions: pd.DataFrame, regime: Regime
) -> dict[tuple[str, str, int], tuple[str, float | None, bool]]:
    """Return ``(status, threshold, extrapolated)`` per (level, class, depth).

    Args:
        decisions: ``decide`` output.
        regime: The regime to read.

    Returns:
        The lookup.
    """
    frame = decisions[decisions["regime"] == regime]
    result: dict[tuple[str, str, int], tuple[str, float | None, bool]] = {}
    for record in frame.to_dict("records"):
        threshold = record["threshold"]
        result[(str(record["level"]), str(record["class"]), int(record["depth"]))] = (
            str(record["status"]),
            None if threshold is None or pd.isna(threshold) else float(threshold),
            bool(record["extrapolated"]),
        )
    return result


def cell_emission(
    decisions: pd.DataFrame,
    regime: Regime,
    level: str,
    parents: Sequence[str | None],
    counts: np.ndarray | Sequence[float],
    grid: Sequence[int],
) -> pd.DataFrame:
    """Apply an emission table to real cells (RESOLVE; §8.3).

    Args:
        decisions: ``decide`` output (reweighted in RESOLVE).
        regime: The panel's regime.
        level: The level.
        parents: Each cell's class key at the level (its called class).
        counts: Each cell's total counts.
        grid: Depth grid.

    Returns:
        Per cell: ``depth_bin``, ``emitted`` (bool), ``threshold``,
        ``resolvability_extrapolated``; cells below the grid or of classes
        absent from the table are not emitted.
    """
    lookup = emission_lookup(decisions, regime)
    bins = depth_bin(counts, grid)
    emitted = np.zeros(len(bins), dtype=bool)
    threshold = np.full(len(bins), np.nan)
    extrapolated = np.zeros(len(bins), dtype=bool)
    for index, (parent, value) in enumerate(zip(parents, bins, strict=True)):
        label = _clean_label(parent)
        if label is None or not np.isfinite(value):
            continue
        entry = lookup.get((level, label, int(value)))
        if entry is None:
            continue
        emitted[index] = entry[0] == STATUS_EMITTED
        threshold[index] = np.nan if entry[1] is None else entry[1]
        extrapolated[index] = entry[2]
    return pd.DataFrame(
        {
            "depth_bin": bins,
            "emitted": emitted,
            "threshold": threshold,
            "resolvability_extrapolated": extrapolated,
        }
    )


REASON_FINE_NOT_ENABLED: Final = "fine_level_not_enabled"
REASON_FINE_SEED_UNSTABLE: Final = "fine_level_seed_unstable"
REASON_NOT_TABULATED: Final = "level_not_tabulated"
REASON_NOT_RESOLVABLE: Final = "not_resolvable_for_class_and_depth"


def level_emission(
    decisions: pd.DataFrame,
    levels: Sequence[LevelMeta],
    level: str,
    parents: Sequence[str | None],
    counts: np.ndarray | Sequence[float],
    grid: Sequence[int],
    *,
    regime: Regime,
    allow_fine_levels: bool = False,
    fine_seed_stability: Mapping[str, float] | None = None,
    seed_stability_max_change: float = 0.02,
) -> pd.DataFrame:
    """Return which cells a level may emit: resolvability replaces ``never_emit``.

    Nothing about emission is hard-coded (plan §8.1): a level is emitted for a
    cell only where the bundle's resolvability table emits it for the
    cell's class at its depth bin (``cell_emission``). The report-only fine
    levels (WHB cluster, WMB supertype; OD-E4) additionally need
    ``allow_fine_levels`` and a seed-1 re-simulation that changes at most
    ``seed_stability_max_change`` of their confident labels; otherwise they
    are ``not_resolvable`` with the reason recorded (§4.2).

    Args:
        decisions: ``decide`` output (RESOLVE: reweighted to the dataset).
        levels: The bundle's level metadata (``ResolvabilityTables.levels``).
        level: The level.
        parents: Each cell's class key at the level (its called class).
        counts: Each cell's total counts.
        grid: The bundle's depth grid.
        regime: ``validated`` (real-data-validated family) or
            ``provisional`` (any other panel, simulation-validated families).
        allow_fine_levels: ``AnnotationThresholds.allow_fine_levels``.
        fine_seed_stability: ``resolvability_summary.json``
            ``fine_level_seed_stability`` (changed share per fine level).
        seed_stability_max_change: ``seed_stability_max_change`` (0.02).

    Returns:
        ``cell_emission`` columns plus ``reason`` (``None`` where emitted).
    """
    meta = next((item for item in levels if item.level == level), None)
    n_cells = len(parents)
    if meta is None:
        return pd.DataFrame(
            {
                "depth_bin": depth_bin(counts, grid),
                "emitted": np.zeros(n_cells, dtype=bool),
                "threshold": np.full(n_cells, np.nan),
                "resolvability_extrapolated": np.zeros(n_cells, dtype=bool),
                "reason": REASON_NOT_TABULATED,
            }
        )
    table = cell_emission(decisions, regime, level, parents, counts, grid)
    # An object array: None where emitted (numpy's stubs reject None here).
    reasons = np.where(table["emitted"].to_numpy(), None, REASON_NOT_RESOLVABLE)  # type: ignore[call-overload]
    if meta.role == "fine":
        stability = (fine_seed_stability or {}).get(level)
        blocked: str | None = None
        if not allow_fine_levels:
            blocked = REASON_FINE_NOT_ENABLED
        elif stability is None or stability > seed_stability_max_change + _TOLERANCE:
            blocked = REASON_FINE_SEED_UNSTABLE
        if blocked is not None:
            table["emitted"] = False
            reasons = np.full(n_cells, blocked, dtype=object)
    table["reason"] = reasons
    return table


def composition_weights(
    cells: pd.DataFrame,
    composition: Mapping[str, float],
    *,
    key: str = TRUTH_LEAF_COLUMN,
) -> np.ndarray:
    """Return per-row weights that reweight the test cells to a composition.

    Within each (recipe, seed, level, depth) the test cells' truth types are
    reweighted to the dataset's composition (§8.3 composition reweighting,
    §5.5): ``w = q(type) / p_depth(type)``. Types absent from the
    composition get weight 0; composition mass on types without test cells
    at a depth cannot be represented and is dropped there.

    Args:
        cells: The cells table.
        composition: Dataset share per truth type, keyed like ``truth_leaf``
            (human: WHB supercluster labels, e.g. ``CS202210140_476``;
            mouse: WMB subclass labels); any scale.
        key: Column holding the truth type.

    Returns:
        Weights aligned with ``cells``.
    """
    total = float(sum(max(float(value), 0.0) for value in composition.values()))
    if total <= 0:
        raise ResolvabilityError("the composition has no positive share")
    share = {
        str(name): max(float(value), 0.0) / total for name, value in composition.items()
    }
    types = cells[key].astype(str).to_numpy()
    weights = np.zeros(len(cells), dtype=np.float64)
    group_keys = ["recipe", "seed", "level", "depth"]
    for _, index in cells.groupby(group_keys, observed=True).indices.items():
        group_types = types[index]
        present, counts = np.unique(group_types, return_counts=True)
        p = dict(zip(present, counts / counts.sum(), strict=True))
        weights[index] = [share.get(value, 0.0) / p[value] for value in group_types]
    return weights


def simulated_floors(
    decisions: pd.DataFrame, regime: Regime
) -> dict[tuple[str, str], int | None]:
    """Return the smallest emitted depth per (level, class) (``derive_floors``).

    Args:
        decisions: ``decide`` output.
        regime: The regime.

    Returns:
        The floor, or ``None`` when the level is never emitted for the class.
    """
    frame = decisions[decisions["regime"] == regime]
    floors: dict[tuple[str, str], int | None] = {}
    for (level, cls), group in frame.groupby(["level", "class"], observed=True):
        emitted = group[group["status"] == STATUS_EMITTED]["depth"]
        floors[(str(level), str(cls))] = int(emitted.min()) if len(emitted) else None
    return floors


def known_floor(
    floor_table: pd.DataFrame | None,
    floor_level: str | None,
    cls: str,
    *,
    hard_floor: int,
) -> int:
    """Return the known floor of a class: the maximum over platforms and panels.

    Args:
        floor_table: The packaged floors (``load_floor_table``), or ``None``.
        floor_level: The floor table's level (``None``: the hard floor).
        cls: Floor class.
        hard_floor: The hard floor.

    Returns:
        ``max(hard_floor, the class's packaged floors)``.
    """
    if floor_table is None or floor_level is None:
        return hard_floor
    rows = floor_table[
        (floor_table["level"] == floor_level) & (floor_table["floor_class"] == cls)
    ]
    if rows.empty:
        return hard_floor
    return max(hard_floor, int(rows["min_counts"].max()))


def combined_floors(
    decisions: pd.DataFrame,
    levels: Sequence[LevelMeta],
    settings: RuleSettings,
    *,
    floor_table: pd.DataFrame | None,
    species: str,
) -> pd.DataFrame:
    """Return the floors per regime: known, simulated and the one applied.

    ``validated``: the packaged (real-data) floors, simulated reported beside
    them. ``provisional``: max(known, simulated) (§5.4 unknown-panel rule),
    and at least ``provisional_mouse_subclass_floor`` for the mouse subclass
    level. A level never emitted for a class has no floor (``None``): it is
    ``not_resolvable`` there anyway.

    Args:
        decisions: ``decide`` output.
        levels: Level metadata.
        settings: Rule settings.
        floor_table: Packaged floors, or ``None``.
        species: ``human`` or ``mouse``.

    Returns:
        ``regime``, ``level``, ``class``, ``known_floor``, ``simulated_floor``,
        ``floor``, ``floor_source``.
    """
    meta_of = {meta.level: meta for meta in levels}
    records = []
    for regime in ("validated", "provisional"):
        simulated = simulated_floors(decisions, regime)
        for (level, cls), value in sorted(simulated.items()):
            meta = meta_of.get(level)
            if meta is None:
                continue
            known = known_floor(
                floor_table, meta.floor_level, cls, hard_floor=settings.hard_floor
            )
            if regime == "validated":
                floor: int | None = known
                source = "real_e2" if meta.floor_level is not None else "hard_floor"
            elif value is None:
                floor, source = None, "not_resolvable"
            else:
                floor = max(known, value)
                if species == "mouse" and level == "subclass":
                    floor = max(floor, settings.provisional_mouse_subclass_floor)
                source = "unknown_panel"
            records.append(
                {
                    "regime": regime,
                    "level": level,
                    "class": cls,
                    "known_floor": known,
                    "simulated_floor": value,
                    "floor": floor,
                    "floor_source": source,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "regime",
            "level",
            "class",
            "known_floor",
            "simulated_floor",
            "floor",
            "floor_source",
        ],
    )


@dataclass(frozen=True)
class TrustConstraint:
    """What resolvability alone implies for the trust state (§8.2).

    Attributes:
        state: ``refused``, ``broad_only`` or ``None`` (no constraint; the
            diagnostics and ``validated_panels.csv`` decide between
            ``provisional`` and ``validated``).
        reasons: Why.
        broad_emitted_bins: Broad bins emitted up to ``trust_max_depth``.
        leaf_share_by_depth: Share of eligible classes whose leaf level is
            emitted, per depth up to ``trust_max_depth``.
        leaf_classes: The eligible classes (enough test cells).
    """

    state: Literal["refused", "broad_only"] | None
    reasons: list[str]
    broad_emitted_bins: int
    leaf_share_by_depth: dict[int, float]
    leaf_classes: list[str]

    def to_json(self) -> dict[str, Any]:
        """Return the constraint as JSON."""
        return {
            "state": self.state,
            "reasons": list(self.reasons),
            "broad_emitted_bins": self.broad_emitted_bins,
            "leaf_share_by_depth": {
                str(depth): round(share, 6)
                for depth, share in self.leaf_share_by_depth.items()
            },
            "leaf_classes": list(self.leaf_classes),
        }


def trust_constraint(
    decisions: pd.DataFrame,
    levels: Sequence[LevelMeta],
    settings: RuleSettings,
) -> TrustConstraint:
    """Return the resolvability part of the trust state (``trust`` regime).

    ``refused`` when the broad level is emitted for no class at any depth up
    to ``trust_max_depth``; ``broad_only`` when, at every such depth, the
    leaf level is emitted for fewer than ``broad_only_min_class_share`` of
    the classes with ``D_max`` (enough test cells).

    Args:
        decisions: ``decide`` output.
        levels: Level metadata (roles ``broad`` and ``leaf``).
        settings: Rule settings.

    Returns:
        The constraint.
    """
    frame = decisions[
        (decisions["regime"] == "trust")
        & (decisions["depth"] <= settings.trust_max_depth)
    ]
    reasons: list[str] = []
    state: Literal["refused", "broad_only"] | None = None
    broad_levels = [meta.level for meta in levels if meta.role == "broad"]
    broad_bins = int(
        (frame["level"].isin(broad_levels) & (frame["status"] == STATUS_EMITTED)).sum()
    )
    if broad_levels and broad_bins == 0:
        state = "refused"
        reasons.append(
            "broad_unresolvable: the broad level fails the local emission rule "
            f"at every depth <= {settings.trust_max_depth}"
        )
    leaf_levels = [meta.level for meta in levels if meta.role == "leaf"]
    shares: dict[int, float] = {}
    eligible: list[str] = []
    if leaf_levels:
        leaf = frame[frame["level"].isin(leaf_levels)]
        eligible = sorted(
            {str(cls) for cls in leaf[leaf["d_max"].notna()]["class"].astype(str)}
        )
        for depth, group in leaf.groupby("depth"):
            if not eligible:
                shares[int(depth)] = 0.0
                continue
            emitted = {
                str(cls)
                for cls in group[
                    (group["status"] == STATUS_EMITTED)
                    & group["class"].astype(str).isin(eligible)
                ]["class"].astype(str)
            }
            shares[int(depth)] = len(emitted) / len(eligible)
        best = max(shares.values()) if shares else 0.0
        if state is None and best < settings.broad_only_min_class_share - _TOLERANCE:
            state = "broad_only"
            reasons.append(
                f"leaf_unresolvable: the leaf level is emitted for at most {best:.2f} "
                f"of the {len(eligible)} classes with enough test cells at every "
                f"depth <= {settings.trust_max_depth} (needs "
                f"{settings.broad_only_min_class_share})"
            )
    return TrustConstraint(
        state=state,
        reasons=reasons,
        broad_emitted_bins=broad_bins,
        leaf_share_by_depth=shares,
        leaf_classes=eligible,
    )


# --------------------------------------------------------------------------
# Tabulation (resolvability.parquet)

TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "kind",
    "recipe",
    "seed",
    "regime",
    "level",
    "class",
    "depth",
    "threshold",
    "node",
    "truth",
    "call",
    "n_test",
    "n_called",
    "n_correct",
    "n_confident",
    "n_effective",
    "precision",
    "coverage",
    "recall",
    "f1",
    "wilson_lb",
    "t_star",
    "target",
    "default_threshold",
    "d_max",
    "extrapolated",
    "status",
    "reason",
    "x",
    "y",
)


def resolvability_table(
    cells: pd.DataFrame,
    decisions: pd.DataFrame,
    levels: Sequence[LevelMeta],
    settings: RuleSettings,
    *,
    extra: Sequence[pd.DataFrame] = (),
) -> pd.DataFrame:
    """Return ``resolvability.parquet``: one long table, ``kind`` per row type.

    Kinds: ``bin`` (per recipe x level x class x depth: n, precision and
    coverage at the default threshold, the local thresholds), ``curve``
    (precision and coverage at thresholds 0.50-0.99), ``isotonic`` (the
    fit-half knots), ``node`` (per-node precision, recall, F1), ``confusion``
    (truth x call counts within the called class), ``decision`` (``decide``
    output) and any ``extra`` rows (e.g. gene efficiencies).

    Args:
        cells: The cells table.
        decisions: ``decide`` output (decision recipe).
        levels: Level metadata.
        settings: Rule settings.
        extra: Further rows in the same columns.

    Returns:
        The table (``TABLE_COLUMNS``).
    """
    meta_of = {meta.level: meta for meta in levels}
    rows: list[dict[str, Any]] = []
    for (recipe, seed, level, cls, depth), group in cells[
        cells["parent"].notna()
    ].groupby(["recipe", "seed", "level", "parent", "depth"], observed=True):
        meta = meta_of.get(str(level))
        if meta is None:
            continue
        bp = group["bp"].to_numpy(np.float64)
        correct = group["correct"].to_numpy(bool)
        ones = np.ones(len(bp))
        at_default = check_threshold(
            bp, correct.astype(float), ones, meta.default_threshold
        )
        fit_mask = (
            (group["half"].to_numpy() == 0)
            if settings.split_halves
            else np.ones(len(bp), bool)
        )
        fit_mask &= np.isfinite(bp)
        common = {
            "recipe": str(recipe),
            "seed": int(seed),
            "level": str(level),
            "class": str(cls),
            "depth": int(depth),
        }
        fit = (
            isotonic_fit(bp[fit_mask], correct[fit_mask].astype(float))
            if int(fit_mask.sum()) >= settings.min_cells_per_bin
            else None
        )
        rows.append(
            {
                "kind": "bin",
                **common,
                "n_called": int(len(bp)),
                "n_correct": int(correct.sum()),
                "n_confident": at_default.n_confident,
                "precision": at_default.precision,
                "coverage": at_default.coverage,
                "wilson_lb": at_default.wilson_lb,
                "default_threshold": meta.default_threshold,
                "t_star": local_threshold(
                    fit,
                    default=meta.default_threshold,
                    target=meta.base_target,
                    cap=settings.threshold_cap,
                ),
                "target": meta.base_target,
            }
        )
        order = np.argsort(-bp, kind="mergesort")
        sorted_bp = bp[order]
        sorted_ok = correct[order]
        cumulative = np.cumsum(sorted_ok)
        for threshold in CURVE_THRESHOLDS:
            n_keep = int(
                np.searchsorted(-sorted_bp, -threshold + _TOLERANCE, side="right")
            )
            rows.append(
                {
                    "kind": "curve",
                    **common,
                    "threshold": float(threshold),
                    "n_called": int(len(bp)),
                    "n_confident": n_keep,
                    "precision": float(cumulative[n_keep - 1] / n_keep)
                    if n_keep
                    else math.nan,
                    "coverage": n_keep / len(bp) if len(bp) else 0.0,
                }
            )
        if fit is not None:
            for x_value, y_value in zip(fit.x, fit.y, strict=True):
                rows.append(
                    {
                        "kind": "isotonic",
                        **common,
                        "x": float(x_value),
                        "y": float(y_value),
                    }
                )
    rows.extend(_node_rows(cells))
    table = pd.DataFrame.from_records(rows)
    decision_rows = decisions.copy()
    decision_rows.insert(0, "kind", "decision")
    decision_rows["recipe"] = DECISION_RECIPE
    frames = [table, decision_rows, *extra]
    combined = pd.concat([frame for frame in frames if len(frame)], ignore_index=True)
    for column in TABLE_COLUMNS:
        if column not in combined.columns:
            combined[column] = None
    combined = combined[list(TABLE_COLUMNS)]
    for column in (
        "kind",
        "recipe",
        "regime",
        "level",
        "class",
        "node",
        "truth",
        "call",
        "status",
        "reason",
    ):
        combined[column] = pd.Categorical(
            [_clean_label(value) for value in combined[column].astype(object)]
        )
    for column in (
        "threshold",
        "n_effective",
        "precision",
        "coverage",
        "recall",
        "f1",
        "wilson_lb",
        "t_star",
        "target",
        "default_threshold",
        "x",
        "y",
    ):
        combined[column] = pd.to_numeric(combined[column], errors="coerce").astype(
            np.float64
        )
    for column in (
        "seed",
        "depth",
        "n_test",
        "n_called",
        "n_correct",
        "n_confident",
        "d_max",
    ):
        combined[column] = pd.to_numeric(combined[column], errors="coerce").astype(
            "Int64"
        )
    combined["extrapolated"] = combined["extrapolated"].astype("boolean")
    return combined


def _node_rows(cells: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-node F1 and within-class confusion rows of the cells table."""
    rows: list[dict[str, Any]] = []
    frame = cells.copy()
    frame["_truth"] = frame["truth"].astype(object).map(_clean_label)
    frame["_call"] = frame["call"].astype(object).map(_clean_label)
    frame.loc[frame["parent"].isna(), "_call"] = None
    for (recipe, seed, level, depth), group in frame.groupby(
        ["recipe", "seed", "level", "depth"], observed=True
    ):
        common = {
            "recipe": str(recipe),
            "seed": int(seed),
            "level": str(level),
            "depth": int(depth),
        }
        truth_counts = group["_truth"].value_counts()
        call_counts = group["_call"].value_counts()
        hits = group[group["_truth"] == group["_call"]]["_truth"].value_counts()
        for node in sorted(set(truth_counts.index) | set(call_counts.index)):
            n_truth = int(truth_counts.get(node, 0))
            n_called = int(call_counts.get(node, 0))
            n_hit = int(hits.get(node, 0))
            precision = n_hit / n_called if n_called else math.nan
            recall = n_hit / n_truth if n_truth else math.nan
            f1 = (
                2 * precision * recall / (precision + recall)
                if n_called and n_truth and (precision + recall) > 0
                else math.nan
            )
            rows.append(
                {
                    "kind": "node",
                    **common,
                    "node": node,
                    "n_test": n_truth,
                    "n_called": n_called,
                    "n_correct": n_hit,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        called = group[group["_call"].notna()]
        pairs = called.groupby(
            [called["parent"].astype(str), "_truth", "_call"], dropna=False
        ).size()
        for (cls, truth, call), count in pairs.items():
            rows.append(
                {
                    "kind": "confusion",
                    **common,
                    "class": cls,
                    "truth": truth,
                    "call": call,
                    "n_called": int(count),
                }
            )
    return rows


# --------------------------------------------------------------------------
# Orchestration


MapFunction = Callable[[SimulatedQuery, str, int], pd.DataFrame]


@dataclass
class ResolvabilityResult:
    """Everything one self-map produced.

    Attributes:
        cells: ``resolvability_cells.parquet`` content.
        table: ``resolvability.parquet`` content.
        decisions: ``decide`` output (unweighted, decision recipe, seed 0).
        floors: ``combined_floors`` output.
        trust: The resolvability trust constraint.
        summary: ``resolvability_summary.json`` content.
    """

    cells: pd.DataFrame
    table: pd.DataFrame
    decisions: pd.DataFrame
    floors: pd.DataFrame
    trust: TrustConstraint
    summary: dict[str, Any]

    def write(self, directory: Path) -> dict[str, str]:
        """Write the three resolvability files.

        Args:
            directory: The bundle work directory.

        Returns:
            File name per output.
        """
        directory.mkdir(parents=True, exist_ok=True)
        coerce_cells(self.cells).to_parquet(
            directory / RESOLVABILITY_CELLS_FILE, index=False
        )
        self.table.to_parquet(directory / RESOLVABILITY_FILE, index=False)
        (directory / RESOLVABILITY_SUMMARY_FILE).write_text(
            json.dumps(_json_native(self.summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {
            "table": RESOLVABILITY_FILE,
            "cells": RESOLVABILITY_CELLS_FILE,
            "summary": RESOLVABILITY_SUMMARY_FILE,
        }

    def bundle_output(self) -> dict[str, Any]:
        """Return the compact record ``bundle.json`` keeps."""
        return {
            "files": {
                "table": RESOLVABILITY_FILE,
                "cells": RESOLVABILITY_CELLS_FILE,
                "summary": RESOLVABILITY_SUMMARY_FILE,
            },
            "recipe": self.summary["recipes"][0],
            "resolvability_version": RESOLVABILITY_VERSION,
            "n_test_cells": self.summary["test_set"]["n_cells"],
            "n_simulated_cells": self.summary["n_simulated_cells"],
            "emitted": self.summary["emitted"],
            "trust": self.trust.to_json(),
            "runtime_s": self.summary["runtime_s"],
        }


def _optional_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for missing values."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def emitted_summary(
    decisions: pd.DataFrame, regime: Regime
) -> dict[str, dict[str, list[int]]]:
    """Return the emitted depths per level and class (for reports and bundle.json).

    Args:
        decisions: ``decide`` output.
        regime: The regime.

    Returns:
        ``{level: {class: [emitted depths]}}`` (empty lists included).
    """
    frame = decisions[decisions["regime"] == regime]
    result: dict[str, dict[str, list[int]]] = {}
    for (level, cls), group in frame.groupby(["level", "class"], observed=True):
        result.setdefault(str(level), {})[str(cls)] = sorted(
            int(depth) for depth in group[group["status"] == STATUS_EMITTED]["depth"]
        )
    return result


def seed_stability(
    first: pd.DataFrame, second: pd.DataFrame, decisions: pd.DataFrame, level: str
) -> float:
    """Return the share of confident labels a seed change alters (fine levels, §8.3).

    Args:
        first: Cells rows of one seed (one level).
        second: Cells rows of the other seed (same simulated cells).
        decisions: Decisions of the first seed (thresholds).
        level: The level.

    Returns:
        Changed share of the first seed's confident labels (0 if none).
    """
    lookup = emission_lookup(decisions, "provisional")
    rows = first[first["level"].astype(str) == level]
    other = second[second["level"].astype(str) == level].set_index("sim_id")
    changed = 0
    confident = 0
    for row in rows.itertuples(index=False):
        if row.parent is None or pd.isna(row.parent):
            continue
        entry = lookup.get((level, str(row.parent), int(row.depth)))
        if entry is None or entry[0] != STATUS_EMITTED or entry[1] is None:
            continue
        if not bool(meets_threshold(np.array([row.bp]), entry[1])[0]):
            continue
        confident += 1
        if row.sim_id not in other.index or str(other.loc[row.sim_id, "call"]) != str(
            row.call
        ):
            changed += 1
    return changed / confident if confident else 0.0


def run_resolvability(
    test: HeldOutCells,
    *,
    specs: Sequence[LevelSpec],
    depths: Sequence[int],
    recipes: Sequence[SimulationRecipe],
    map_fn: MapFunction,
    settings: RuleSettings,
    species: str,
    floor_table: pd.DataFrame | None = None,
    fine_seed_check: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> ResolvabilityResult:
    """Run the self-map: simulate, map, tabulate, decide (§8.3).

    Args:
        test: The test cells (already restricted to the engine's genes).
        specs: Levels to evaluate (calls and truths).
        depths: The depth grid.
        recipes: Recipes; the first is the decision recipe.
        map_fn: ``(query, tag, seed) -> MMC tidy table`` with the production
            configuration.
        settings: Rule settings.
        species: ``human`` or ``mouse``.
        floor_table: Packaged floors for the max rule.
        fine_seed_check: Re-map the decision recipe with seed 1 and record
            the fine levels' seed stability (``allow_fine_levels``).
        provenance: Extra summary fields (engine, test set, bundle ids).

    Returns:
        The result.
    """
    if not recipes:
        raise ResolvabilityError("run_resolvability needs at least one recipe")
    started = time.monotonic()
    timings: dict[str, float] = {}
    frames: list[pd.DataFrame] = []
    queries: dict[str, SimulatedQuery] = {}
    for recipe in recipes:
        step = time.monotonic()
        query = thin_and_contaminate(test, depths, recipe)
        queries[recipe.name] = query
        timings[f"simulate_{recipe.name}"] = round(time.monotonic() - step, 3)
        step = time.monotonic()
        tidy = map_fn(query, recipe.name, 0)
        timings[f"map_{recipe.name}"] = round(time.monotonic() - step, 3)
        frames.append(level_cells(tidy, query, test, specs, seed=0))
    decision_recipe = recipes[0]
    fine_levels = [spec.meta.level for spec in specs if spec.meta.role == "fine"]
    stability: dict[str, float] = {}
    cells = pd.concat(frames, ignore_index=True)
    levels = [spec.meta for spec in specs]
    step = time.monotonic()
    decisions = decide(cells, levels, depths, settings, recipe=decision_recipe.name)
    timings["decide"] = round(time.monotonic() - step, 3)
    if fine_seed_check and fine_levels:
        step = time.monotonic()
        tidy = map_fn(queries[decision_recipe.name], f"{decision_recipe.name}_seed1", 1)
        second = level_cells(
            tidy,
            queries[decision_recipe.name],
            test,
            [spec for spec in specs if spec.meta.role == "fine"],
            seed=1,
        )
        timings["map_seed1"] = round(time.monotonic() - step, 3)
        first = cells[(cells["recipe"] == decision_recipe.name) & (cells["seed"] == 0)]
        for level in fine_levels:
            stability[level] = seed_stability(first, second, decisions, level)
        cells = pd.concat([cells, second], ignore_index=True)
    floors = combined_floors(
        decisions, levels, settings, floor_table=floor_table, species=species
    )
    trust = trust_constraint(decisions, levels, settings)
    efficiency = gene_efficiency(
        len(test.genes), decision_recipe.gene_efficiency_sigma, decision_recipe.seed
    )
    extra = [
        pd.DataFrame(
            {
                "kind": "gene_efficiency",
                "recipe": decision_recipe.name,
                "node": test.genes,
                "x": efficiency,
            }
        )
    ]
    step = time.monotonic()
    table = resolvability_table(cells, decisions, levels, settings, extra=extra)
    timings["tabulate"] = round(time.monotonic() - step, 3)
    summary = build_summary(
        test=test,
        levels=levels,
        depths=depths,
        recipes=recipes,
        settings=settings,
        decisions=decisions,
        floors=floors,
        trust=trust,
        queries=queries,
        stability=stability,
        species=species,
        runtime={**timings, "total": round(time.monotonic() - started, 3)},
        provenance=provenance or {},
    )
    return ResolvabilityResult(
        cells=cells,
        table=table,
        decisions=decisions,
        floors=floors,
        trust=trust,
        summary=summary,
    )


def build_summary(
    *,
    test: HeldOutCells,
    levels: Sequence[LevelMeta],
    depths: Sequence[int],
    recipes: Sequence[SimulationRecipe],
    settings: RuleSettings,
    decisions: pd.DataFrame,
    floors: pd.DataFrame,
    trust: TrustConstraint,
    queries: Mapping[str, SimulatedQuery],
    stability: Mapping[str, float],
    species: str,
    runtime: Mapping[str, float],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return ``resolvability_summary.json`` (decisions RESOLVE and reports read).

    Args:
        test: Test cells.
        levels: Level metadata.
        depths: Depth grid.
        recipes: Recipes (decision recipe first).
        settings: Rule settings.
        decisions: ``decide`` output.
        floors: ``combined_floors`` output.
        trust: Trust constraint.
        queries: Simulated queries by recipe.
        stability: Fine-level seed stability.
        species: Species.
        runtime: Step timings in seconds.
        provenance: Extra fields.

    Returns:
        The summary.
    """
    decision = queries[recipes[0].name]
    native = test.native_counts
    spill_groups = test.obs[SPILL_GROUP_COLUMN].astype(str)
    emission: dict[str, Any] = {}
    for regime in REGIMES:
        frame = decisions[decisions["regime"] == regime]
        per_level: dict[str, Any] = {}
        for record in frame.to_dict("records"):
            per_level.setdefault(str(record["level"]), {}).setdefault(
                str(record["class"]), {}
            )[str(int(record["depth"]))] = {
                "status": record["status"],
                "threshold": _optional_float(record["threshold"]),
                "t_star": _optional_float(record["t_star"]),
                "extrapolated": bool(record["extrapolated"]),
                "reason": record["reason"],
            }
        emission[regime] = per_level
    d_max: dict[str, dict[str, int | None]] = {}
    for record in decisions[decisions["regime"] == "trust"].to_dict("records"):
        value = _optional_float(record["d_max"])
        d_max.setdefault(str(record["level"]), {})[str(record["class"])] = (
            None if value is None else int(value)
        )
    raised = decisions[
        (decisions["regime"] == "validated")
        & (~decisions["extrapolated"])
        & _bool_column(decisions["would_raise"])
        & (decisions["n_fit"].fillna(0) > 0)
    ]
    floors_json: dict[str, Any] = {}
    for record in floors.to_dict("records"):
        simulated = _optional_float(record["simulated_floor"])
        floor = _optional_float(record["floor"])
        floors_json.setdefault(str(record["regime"]), {}).setdefault(
            str(record["level"]), {}
        )[str(record["class"])] = {
            "known": int(record["known_floor"]),
            "simulated": None if simulated is None else int(simulated),
            "floor": None if floor is None else int(floor),
            "source": record["floor_source"],
        }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "resolvability_version": RESOLVABILITY_VERSION,
        "species": species,
        "decision_recipe": recipes[0].name,
        "recipes": [recipe.to_json() for recipe in recipes],
        "depth_grid": [int(depth) for depth in sorted(depths)],
        "levels": [meta.to_json() for meta in levels],
        "settings": settings.to_json(),
        "regimes": list(REGIMES),
        "test_set": {
            "n_cells": int(len(test.obs)),
            "n_genes": len(test.genes),
            "per_spill_group": {
                str(key): int(value)
                for key, value in spill_groups.value_counts().items()
            },
            "native_counts_quantiles": {
                str(q): float(np.quantile(native, q)) if len(native) else None
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "n_by_depth": {str(depth): n for depth, n in decision.n_by_depth.items()},
        },
        "n_simulated_cells": {
            name: int(len(query.obs)) for name, query in queries.items()
        },
        "d_max": d_max,
        "emission": emission,
        "emitted": {regime: emitted_summary(decisions, regime) for regime in REGIMES},
        "floors": floors_json,
        "validated_thresholds_would_raise": [
            {
                "level": str(record["level"]),
                "class": str(record["class"]),
                "depth": int(record["depth"]),
                "default": float(record["default_threshold"]),
                "t_star": _optional_float(record["t_star"]),
            }
            for record in raised.to_dict("records")
        ],
        "fine_level_seed_stability": dict(stability),
        "trust": trust.to_json(),
        "runtime_s": dict(runtime),
        **dict(provenance),
    }


# --------------------------------------------------------------------------
# Loading for RESOLVE and reports


@dataclass
class ResolvabilityTables:
    """The resolvability outputs of one bundle (RESOLVE's inputs).

    Attributes:
        summary: ``resolvability_summary.json``.
        cells: ``resolvability_cells.parquet``.
        levels: Level metadata.
        settings: The rule settings the bundle used.
    """

    summary: dict[str, Any]
    cells: pd.DataFrame
    levels: list[LevelMeta]
    settings: RuleSettings

    @property
    def depth_grid(self) -> list[int]:
        """The bundle's depth grid."""
        return [int(depth) for depth in self.summary["depth_grid"]]

    def decisions(
        self,
        *,
        composition: Mapping[str, float] | None = None,
        settings: RuleSettings | None = None,
    ) -> pd.DataFrame:
        """Return decisions, reweighted to a dataset's composition when given.

        Args:
            composition: Dataset share per truth type (human supercluster,
                mouse subclass), or ``None`` for PREP's unweighted decisions.
            settings: Rule settings (default: the bundle's).

        Returns:
            ``decide`` output.
        """
        weights = (
            None
            if composition is None
            else composition_weights(self.cells, composition)
        )
        return decide(
            self.cells,
            self.levels,
            self.depth_grid,
            settings or self.settings,
            weights=weights,
            recipe=str(self.summary["decision_recipe"]),
        )


def load_resolvability(directory: Path | str) -> ResolvabilityTables | None:
    """Read a bundle's resolvability outputs (``None`` when it has none).

    Args:
        directory: Bundle directory.

    Returns:
        The tables, or ``None``.
    """
    root = Path(directory)
    summary_path = root / RESOLVABILITY_SUMMARY_FILE
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cells = pd.read_parquet(root / RESOLVABILITY_CELLS_FILE)
    for column in ("recipe", "level", "parent", "call", "truth", "truth_parent"):
        cells[column] = cells[column].astype(object).where(cells[column].notna(), None)
    fields = RuleSettings.__dataclass_fields__
    settings = RuleSettings(
        **{key: value for key, value in summary["settings"].items() if key in fields}
    )
    return ResolvabilityTables(
        summary=summary,
        cells=cells,
        levels=[LevelMeta.from_json(item) for item in summary["levels"]],
        settings=settings,
    )


# --------------------------------------------------------------------------
# Gate-P machinery (M13 scores NP3-NP7 with these; §14 evaluation rules)


def frozen_threshold_eval(
    cells: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    regime: Regime = "provisional",
    recipe: str | None = None,
) -> pd.DataFrame:
    """Apply frozen thresholds to replicate cells, per (level, class, depth).

    Args:
        cells: Replicate cells table (another donor, draw, seed or recipe).
        decisions: Frozen ``decide`` output of the base run.
        regime: Regime whose thresholds are frozen.
        recipe: Restrict the replicate rows to one recipe.

    Returns:
        ``level``, ``class``, ``depth``, ``threshold``, ``n_called``,
        ``n_confident``, ``precision``, ``wilson_lb``, ``coverage``.
    """
    lookup = emission_lookup(decisions, regime)
    frame = cells[cells["parent"].notna()]
    if recipe is not None:
        frame = frame[frame["recipe"].astype(str) == recipe]
    records = []
    for (level, cls, depth), group in frame.groupby(
        [frame["level"].astype(str), frame["parent"].astype(str), "depth"],
        observed=True,
    ):
        entry = lookup.get((str(level), str(cls), int(depth)))
        threshold = None if entry is None or entry[0] != STATUS_EMITTED else entry[1]
        stats = check_threshold(
            group["bp"].to_numpy(np.float64),
            group["correct"].to_numpy(bool).astype(float),
            np.ones(len(group)),
            threshold,
        )
        records.append(
            {
                "level": level,
                "class": cls,
                "depth": int(depth),
                "threshold": threshold,
                "n_called": stats.n_called,
                "n_confident": stats.n_confident,
                "precision": stats.precision,
                "wilson_lb": stats.wilson_lb,
                "coverage": stats.coverage,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "level",
            "class",
            "depth",
            "threshold",
            "n_called",
            "n_confident",
            "precision",
            "wilson_lb",
            "coverage",
        ],
    )


@dataclass(frozen=True)
class GatePTestedSet:
    """One gate-P tested set of a (level, class) (§14 evaluation rules).

    Attributes:
        level: Level.
        cls: Class.
        depths: Depth bins in the set (one bin, or a pooled ">= D_P" set).
        pooled: Whether bins were pooled from the deep end.
        n_confident: Confident calls (each test cell once, at its deepest
            bin of the set).
        precision: Their precision.
        wilson_lb: Wilson lower bound.
    """

    level: str
    cls: str
    depths: tuple[int, ...]
    pooled: bool
    n_confident: int
    precision: float
    wilson_lb: float


def gate_p_tested_sets(
    cells: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    min_confident_n: int = 200,
    regime: Regime = "provisional",
) -> dict[tuple[str, str], list[GatePTestedSet] | None]:
    """Return the gate-P tested sets per (level, class) (``None``: not evaluable).

    From the deep end, bins are pooled into a ">= d" set (each test cell
    counted once, at its deepest bin in the set) until it holds
    ``min_confident_n`` confident calls; its shallowest bin is ``D_P``.
    Shallower bins are tested on their own when they hold enough calls.

    Args:
        cells: Pooled held-out cells (frozen-threshold replicates).
        decisions: Frozen decisions.
        min_confident_n: ``gate_p_min_confident_n`` (200).
        regime: Regime of the thresholds.

    Returns:
        Tested sets per (level, class), deepest first; ``None`` when the
        class has fewer than ``min_confident_n`` confident calls in total.
    """
    lookup = emission_lookup(decisions, regime)
    frame = cells[cells["parent"].notna()].copy()
    thresholds = [
        lookup.get((str(level), str(cls), int(depth)))
        for level, cls, depth in zip(
            frame["level"].astype(str),
            frame["parent"].astype(str),
            frame["depth"],
            strict=True,
        )
    ]
    confident = np.array(
        [
            entry is not None
            and entry[0] == STATUS_EMITTED
            and entry[1] is not None
            and bp >= entry[1] - 1e-9
            for entry, bp in zip(
                thresholds, frame["bp"].to_numpy(np.float64), strict=True
            )
        ],
        dtype=bool,
    )
    frame = frame[confident]
    result: dict[tuple[str, str], list[GatePTestedSet] | None] = {}
    for (level, cls), group in frame.groupby(
        [frame["level"].astype(str), frame["parent"].astype(str)], observed=True
    ):
        depths = sorted({int(depth) for depth in group["depth"]}, reverse=True)
        sets: list[GatePTestedSet] = []
        pooled: list[int] = []
        remaining = list(depths)
        while remaining:
            pooled.append(remaining.pop(0))
            subset = group[group["depth"].isin(pooled)]
            deepest = subset.sort_values("depth", ascending=False).drop_duplicates(
                "cell_id"
            )
            if len(deepest) >= min_confident_n:
                precision = float(deepest["correct"].mean())
                sets.append(
                    GatePTestedSet(
                        level=str(level),
                        cls=str(cls),
                        depths=tuple(sorted(pooled)),
                        pooled=len(pooled) > 1,
                        n_confident=int(len(deepest)),
                        precision=precision,
                        wilson_lb=wilson_lower_bound(precision, len(deepest)),
                    )
                )
                break
        if not sets:
            result[(str(level), str(cls))] = None
            continue
        for depth in remaining:
            subset = group[group["depth"] == depth]
            if len(subset) >= min_confident_n:
                precision = float(subset["correct"].mean())
                sets.append(
                    GatePTestedSet(
                        level=str(level),
                        cls=str(cls),
                        depths=(depth,),
                        pooled=False,
                        n_confident=int(len(subset)),
                        precision=precision,
                        wilson_lb=wilson_lower_bound(precision, len(subset)),
                    )
                )
        result[(str(level), str(cls))] = sets
    return result


def gate_p_class_set(
    truth_classes: pd.Series | Sequence[str],
    *,
    min_test_cells: int = 700,
    min_share: float = 0.9,
) -> tuple[list[str], float, bool]:
    """Return the gate-P class set C_P, fixed from the test-cell table.

    Args:
        truth_classes: Pooled test cells' truth classes.
        min_test_cells: ``gate_p_class_min_test_cells`` (700).
        min_share: Share of pooled test cells C_P must hold (0.9).

    Returns:
        ``(C_P, share of test cells in C_P, share >= min_share)``.
    """
    counts = pd.Series(list(truth_classes), dtype=object).value_counts()
    members = sorted(
        str(cls) for cls, count in counts.items() if count >= min_test_cells
    )
    total = int(counts.sum())
    share = float(counts[counts.index.isin(members)].sum()) / total if total else 0.0
    return members, share, share >= min_share - _TOLERANCE
