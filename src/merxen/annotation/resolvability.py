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
* **emission**: a tested set (a bin, or a pooled deep set) is emitted when
  ``t*`` exists and the check half holds at least ``min_confident_n``
  confident calls with a Wilson 95% lower bound of their precision (on the
  Kish n) at least ``target - wilson_margin``, a point precision at least
  ``target`` and coverage at least ``min_coverage`` (one rule with the
  gate-P evaluation rules, §14). The ``validated`` regime applies the
  pre-registered default, fitted on no test cell, so it checks the same
  rule on every call of the bin (both halves; M3b review). Only cells whose
  native panel counts reach a depth are thinned to it, so deep bins run
  short of calls: a bin with fewer than ``min_confident_n`` confident calls
  takes the verdict of the deep-end pool (user decision 2026-09-27; bins
  pooled from the deep end, each test cell once at its deepest bin, until
  the set holds ``min_confident_n`` calls; ``decide``) and is marked
  extrapolated when it lies deeper than the pool's shallowest bin ``D_P``,
  while a bin judged on its own keeps its own verdict (withdrawn when the
  pool fails); a class that never reaches them is ``insufficient_calls``,
  and a class without ``D_max(c)`` (no grid value with
  ``min_cells_per_bin`` test cells of class ``c``) is never emitted (no
  pooling across classes). Everything not emitted is ``not_resolvable``;
* **floors**: the smallest emitted depth, combined with the known floors by
  the max rule for panels without real-data validation (§5.4); validated
  panels keep the packaged per-platform floors;
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
decisions with the simulated cells of each depth bin reweighted to the
dataset's composition in that bin (``DatasetComposition``,
``composition_weights``: rare types pooled at broad-class level; ``decide``
trims the weights within each judged set): PREP's unweighted tables set only
the panel's trust constraints. Production rules that decide confidence
outside the level's own bp (the WHB COP rule, ``whb_cells_rules``) are
applied to the cells table before the decisions.

This module needs numpy, pandas and scipy only; MapMyCells runs through the
``map_fn`` a builder passes to ``run_resolvability``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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
# 2 (M3b review): the validated regime checks the default on both halves;
# composition weights per depth bin, rare types pooled at broad-class level,
# trimmed; Kish n and the largest weight share per decision; the WHB COP
# rule on the broad level; per-platform packaged floors in the summary.
# 3 (H18 follow-up, user decisions 2026-09-27): pooled deep bins (the gate-P
# pooling with n_min = min_confident_n replaces the D_max inheritance;
# insufficient_calls), the point precision joins the Wilson rule, and the
# human held-out test set gains other-region non-neuronal cells.
# 4 (M3b review 2, 2026-09-28): composition weights trimmed within each
# judged set (``RuleSettings.weight_trim_scope``), not against the median of
# the whole depth bin; only positive-weight calls count towards
# ``min_confident_n``; every bin judged on its own at or above ``D_P`` keeps
# its own verdict (withdrawn when the pool fails) and only unjudged bins take
# the pooled one; ``would_raise`` only where a fit exists
# (``would_raise_evaluable``); PREP decides on the stored (float32) cells;
# the human held-out test set drops truth superclusters no production call
# can name (sinks, no floor class, region-implausible; E2).
# 5 (M3b final follow-up, 2026-09-28): every per-cell simulation draw is
# keyed by what it simulates (``draw_key``: the seed, the recipe name and
# version, the cell id, the depth; version 5 also keyed the gene efficiency
# by the gene id) instead of one stream over all test cells, and the spill
# partner is chosen by
# rendezvous hashing, so adding or removing test cells no longer redraws
# every other cell (review 2: the redraw moved 0-7 validated bins per
# dataset); the human held-out test set's other-region cells come only from
# clusters of the held-out training reference (user decision 2026-09-27).
# 6 (M3b review 3, 2026-09-28): the per-gene efficiency is again the
# pre-registered version-4 draw (one LogNormal(0, sigma) draw over the
# panel's gene order from the seed); version 5 had keyed it per gene id,
# which neither requested change needed (the version-4 vector never
# depended on the test cells) and which replaced the pre-registered seed-0
# realisation. The per-cell thinning, spill and partner keys of version 5
# stay.
RESOLVABILITY_VERSION: Final = 6
SUMMARY_SCHEMA_VERSION: Final = 1
RESOLVABILITY_FILE: Final = "resolvability.parquet"
RESOLVABILITY_CELLS_FILE: Final = "resolvability_cells.parquet"
RESOLVABILITY_SUMMARY_FILE: Final = "resolvability_summary.json"
TEST_CELLS_FILE: Final = "test_cells.h5ad"
TEST_CELLS_OBS_FILE: Final = "test_cells.parquet"

DECISION_RECIPE: Final = "R1_contam_HO"
CLEAN_RECIPE: Final = "clean"
# Version-7 recipes (M3c, plan §8.3 v7.3-v7.4): the measured-efficiency member
# and the cross-tissue human stress recipe; version 6 uses only the two above.
R3_RECIPE: Final = "R3_measured_HO"
LUNG_STRESS_RECIPE: Final = "R1_xtissue_lung_stress"
RECIPE_VERSIONS: Final[dict[str, int]] = {
    DECISION_RECIPE: 1,
    CLEAN_RECIPE: 1,
    R3_RECIPE: 1,
    LUNG_STRESS_RECIPE: 1,
}
# Keyed simulation draws (version 5; ``draw_key``): the streams of one
# simulated cell (its thinning, its spill partner and the spill's thinning).
DRAW_KEY_BYTES: Final = 16
DRAW_STREAM_THIN: Final = "thin"
DRAW_STREAM_PARTNER: Final = "partner"
DRAW_STREAM_SPILL: Final = "spill"
DRAW_STREAM_CANDIDATE: Final = "candidate"
# The per-gene efficiency generator (the pre-registered version-4 draw).
_GENE_EFFICIENCY_STREAM: Final = 0x6566
# Host x candidate scores per block of the rendezvous partner choice.
_RENDEZVOUS_BLOCK_SCORES: Final = 1 << 22

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
    """One simulation recipe of the self-map (§8.3 step 3; v7.3-v7.4).

    Attributes:
        name: ``R1_contam_HO`` (decisions) or ``clean`` (upper bound);
            version 7 adds ``R3_measured_HO`` and ``R1_xtissue_lung_stress``.
        version: Recipe version.
        gene_efficiency_sigma: Sigma of the per-gene LogNormal(0, sigma)
            efficiency (median-normalised); ``None`` for none.
        spill_fraction: Foreign-class spill as a fraction of the depth.
        seed: Simulation seed.
        efficiency_source: ``lognormal`` (versions 1-6 and the R1 members),
            ``measured`` (R3: a ``member`` factor table) or
            ``xtissue_stress`` (the R1 draw times a ``stress`` ratio table).
        efficiency_table: The table's simulation-input asset id
            (``merxen.annotation.sim_inputs``), for ``measured`` and
            ``xtissue_stress``.
        efficiency_table_sha256: That asset's sha256 (it enters a version-7
            ``build_hash``; the recipe refuses another table).
        table_rule: R3 table rule (``restricted``; ``all_measured`` for the
            D3 regression only).
        residual_sd_log2: R3 residual SD of measured genes (0.20 log2).
    """

    name: str
    version: int
    gene_efficiency_sigma: float | None
    spill_fraction: float
    seed: int
    efficiency_source: str = "lognormal"
    efficiency_table: str | None = None
    efficiency_table_sha256: str | None = None
    table_rule: str | None = None
    residual_sd_log2: float | None = None

    @property
    def member(self) -> str:
        """The ensemble member name, ``<name>@<seed>`` (e.g. ``R1_contam_HO@1``)."""
        return f"{self.name}@{int(self.seed)}"

    def to_json(self) -> dict[str, Any]:
        """Return the recipe as JSON-native values.

        A ``lognormal`` recipe keeps the five keys of versions 1-6, so its
        record (and every ``build_hash`` holding it) is unchanged; the
        version-7 efficiency fields are added only for other sources.
        """
        record: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "gene_efficiency_sigma": self.gene_efficiency_sigma,
            "spill_fraction": self.spill_fraction,
            "seed": self.seed,
        }
        if self.efficiency_source != "lognormal":
            record.update(
                {
                    "efficiency_source": self.efficiency_source,
                    "efficiency_table": self.efficiency_table,
                    "efficiency_table_sha256": self.efficiency_table_sha256,
                    "table_rule": self.table_rule,
                    "residual_sd_log2": self.residual_sd_log2,
                }
            )
        return record


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


def draw_key(*parts: object) -> int:
    """Return the stable 128-bit key of a simulation draw (BLAKE2b of ``parts``).

    The key depends only on the text of ``parts`` (never on Python's hash
    seed, the platform or the other test cells), so a draw keyed by what it
    simulates is the same in every build that simulates it.

    Args:
        parts: What identifies the draw (seed, recipe name and version, cell
            id, depth, stream).

    Returns:
        A non-negative integer below ``2**128`` (a ``numpy`` seed).
    """
    text = "\x1f".join(str(part) for part in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=DRAW_KEY_BYTES)
    return int.from_bytes(digest.digest(), "big")


def cell_draw_key(
    recipe: SimulationRecipe, cell_id: str, depth: int, stream: str
) -> int:
    """Return the key of one stream of one simulated cell (version 5).

    Args:
        recipe: The recipe (its seed, name and version enter the key).
        cell_id: The test cell.
        depth: The grid value the cell is thinned to.
        stream: ``thin``, ``partner`` or ``spill``.

    Returns:
        The ``draw_key``.
    """
    return draw_key(
        int(recipe.seed), recipe.name, int(recipe.version), cell_id, int(depth), stream
    )


def _key64(key: int) -> np.uint64:
    return np.uint64(key & 0xFFFFFFFFFFFFFFFF)


def _mix64(values: np.ndarray) -> np.ndarray:
    """Return the splitmix64 finaliser of uint64 values (vectorised)."""
    with np.errstate(over="ignore"):
        mixed = values + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return np.asarray(mixed ^ (mixed >> np.uint64(31)), dtype=np.uint64)


def rendezvous_choice(host_keys: np.ndarray, candidate_keys: np.ndarray) -> np.ndarray:
    """Choose one candidate per host by rendezvous (highest-random-weight) hashing.

    Each (host, candidate) pair gets a pseudo-random score from the two
    keys alone and a host takes the candidate with the highest score, so
    the choice is uniform over the candidates and does not depend on their
    order; removing a candidate changes only the hosts that had chosen it,
    and adding one changes only the hosts it outranks.

    Args:
        host_keys: One uint64 key per host.
        candidate_keys: One uint64 key per candidate (at least one).

    Returns:
        Per host, the position of the chosen candidate in ``candidate_keys``.

    Raises:
        ResolvabilityError: If there is no candidate.
    """
    hosts = np.asarray(host_keys, dtype=np.uint64)
    candidates = np.asarray(candidate_keys, dtype=np.uint64)
    if len(candidates) == 0:
        raise ResolvabilityError("rendezvous_choice needs at least one candidate")
    chosen = np.empty(len(hosts), dtype=np.int64)
    block = max(1, _RENDEZVOUS_BLOCK_SCORES // len(candidates))
    mixed_candidates = _mix64(candidates)
    for start in range(0, len(hosts), block):
        scores = _mix64(hosts[start : start + block, None] ^ mixed_candidates[None, :])
        chosen[start : start + block] = np.argmax(scores, axis=1)
    return chosen


def gene_efficiency(n_genes: int, sigma: float | None, seed: int) -> np.ndarray:
    """Return per-gene detection efficiencies, LogNormal(0, sigma) / median.

    One draw per gene, fixed across depths, cells and recipes, as E2's
    ``gene_efficiency.npy``: it models a platform's per-gene detection bias,
    which every cell shares. The draw is the pre-registered one (resolvability
    versions 1-4 and 6): ``n_genes`` normals from one generator seeded by
    ``[seed, 0x6566]``, in the panel's gene order (the test cells' columns,
    sorted Ensembl ids). It depends on the seed and the panel only, never on
    the test cells, so changing the test set leaves it unchanged; a recipe
    with another sigma (a gate-P stress recipe) scales the same normals.
    Version 5 keyed each gene's normal by the gene id instead, which redrew
    the pre-registered seed-0 realisation; version 6 restores it.

    Args:
        n_genes: Genes (the test cells' columns).
        sigma: LogNormal sigma (``None``: all ones).
        seed: The simulation seed.

    Returns:
        Efficiencies with median 1, in the panel's gene order.
    """
    if sigma is None or n_genes == 0:
        return np.ones(n_genes, dtype=np.float64)
    rng = np.random.default_rng([int(seed), _GENE_EFFICIENCY_STREAM])
    values = np.exp(rng.normal(0.0, float(sigma), n_genes))
    return np.asarray(values / np.median(values), dtype=np.float64)


def _thin_rows(
    matrix: sparse.csr_matrix,
    targets: np.ndarray,
    efficiency: np.ndarray,
    keys: Sequence[int],
) -> sparse.csr_matrix:
    """Thin each row binomially to an expected total (E2 ``thin``).

    ``p_g = min(1, D * e_g / sum_g(x_g * e_g))`` per row, then
    ``Binomial(x_g, p_g)``; rows are never raised. Row ``i`` is drawn from
    its own generator seeded by ``keys[i]`` over its entries in gene order,
    so its result depends on nothing but the row, its target and its key.
    """
    from scipy import sparse as sp

    work = sp.csr_matrix(matrix, dtype=np.float64, copy=True)
    work.sum_duplicates()
    work.eliminate_zeros()
    if len(keys) != work.shape[0]:
        raise ResolvabilityError(
            f"thinning: {len(keys)} draw keys for {work.shape[0]} rows"
        )
    rows = np.repeat(np.arange(work.shape[0]), np.diff(work.indptr))
    weighted = np.asarray(
        work.multiply(efficiency[None, :]).sum(axis=1), dtype=np.float64
    ).ravel()
    scale = np.asarray(targets, dtype=np.float64) / np.maximum(weighted, 1e-12)
    probability = np.clip(scale[rows] * efficiency[work.indices], 0.0, 1.0)
    counts = work.data.astype(np.int64)
    thinned = np.zeros(len(counts), dtype=np.int64)
    indptr = work.indptr
    for row, key in enumerate(keys):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if stop > start:
            thinned[start:stop] = np.random.default_rng(int(key)).binomial(
                counts[start:stop], probability[start:stop]
            )
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

    Every per-cell draw is keyed by what it simulates (resolvability version
    5; ``cell_draw_key``: the recipe's seed, name and version, the host cell
    id and ``D``), and the gene efficiency depends on the seed and the panel
    only (``gene_efficiency``): the host's thinning and the spill's thinning
    each come from
    the host's own generator, and the spill partner is the eligible cell
    with the highest rendezvous score (``rendezvous_choice``; uniform over
    the eligible cells). A simulated cell therefore does not depend on the
    other test cells or their order: removing or adding test cells changes
    it only when its partner is removed or an added eligible cell outranks
    that partner (``clean``: never).

    Args:
        test: Test cells (unique cell ids).
        depths: Depth grid.
        recipe: Recipe.

    Returns:
        The simulated query.

    Raises:
        ResolvabilityError: If the cell ids are not unique, or a spill recipe
            has cells of a single group.
    """
    from scipy import sparse as sp

    if not test.obs.index.is_unique:
        raise ResolvabilityError(
            "test cell ids must be unique (they key the simulation draws)"
        )
    efficiency = gene_efficiency(
        len(test.genes), recipe.gene_efficiency_sigma, recipe.seed
    )
    native = test.native_counts
    groups = test.obs[SPILL_GROUP_COLUMN].astype(str).to_numpy()
    cell_ids = test.obs.index.astype(str).to_numpy()
    counts = sp.csr_matrix(test.counts)
    candidate_keys = np.array(
        [_key64(draw_key(DRAW_STREAM_CANDIDATE, cell_id)) for cell_id in cell_ids],
        dtype=np.uint64,
    )
    blocks: list[sp.csr_matrix] = []
    frames: list[pd.DataFrame] = []
    n_by_depth: dict[int, int] = {}
    for depth in sorted(int(value) for value in depths):
        hosts = np.flatnonzero(native >= depth)
        n_by_depth[depth] = int(len(hosts))
        if len(hosts) == 0:
            continue
        host_ids = [str(cell_id) for cell_id in cell_ids[hosts]]
        host_counts = _thin_rows(
            counts[hosts],
            np.full(len(hosts), float(depth)),
            efficiency,
            [
                cell_draw_key(recipe, cell_id, depth, DRAW_STREAM_THIN)
                for cell_id in host_ids
            ],
        )
        partner_ids = np.full(len(hosts), "", dtype=object)
        spill = sp.csr_matrix(host_counts.shape, dtype=np.float64)
        if recipe.spill_fraction > 0:
            amount = recipe.spill_fraction * depth
            donors = np.flatnonzero(native >= amount)
            partners = np.empty(len(hosts), dtype=np.int64)
            host_keys = np.array(
                [
                    _key64(cell_draw_key(recipe, cell_id, depth, DRAW_STREAM_PARTNER))
                    for cell_id in host_ids
                ],
                dtype=np.uint64,
            )
            for group in np.unique(groups[hosts]):
                is_host = groups[hosts] == group
                candidates = donors[groups[donors] != group]
                if len(candidates) == 0:
                    raise ResolvabilityError(
                        f"no spill donor outside group {group!r} at depth {depth}"
                    )
                partners[is_host] = candidates[
                    rendezvous_choice(host_keys[is_host], candidate_keys[candidates])
                ]
            spill = _thin_rows(
                counts[partners],
                np.full(len(hosts), amount),
                efficiency,
                [
                    cell_draw_key(recipe, cell_id, depth, DRAW_STREAM_SPILL)
                    for cell_id in host_ids
                ],
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
    cap = stratum_cap(sizes, max_per_stratum, n_max)
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


def stratum_cap(
    sizes: pd.Series | Mapping[str, int], max_per_stratum: int, n_max: int | None
) -> int:
    """Return the per-stratum cap of ``select_test_cells`` (water filling).

    Args:
        sizes: Candidate cells per stratum.
        max_per_stratum: Largest share of one stratum (1,000).
        n_max: Largest total, or ``None``.

    Returns:
        ``max_per_stratum``, lowered until the capped strata fit ``n_max``.
    """
    values = np.asarray(list(dict(sizes).values()), dtype=np.int64)
    cap = int(max_per_stratum)
    if n_max is not None:
        while cap > 0 and int(np.minimum(values, cap).sum()) > n_max:
            cap -= 1
    return cap


def top_up_test_cells(
    candidates: pd.DataFrame,
    *,
    stratum: str,
    have: Mapping[str, int],
    cap: int,
    room: int | None,
    seed: int,
) -> pd.Index:
    """Top strata of a test set up to a cap from extra candidates (§8.3 step 1).

    The human held-out donor holds few cells of the thin non-neuronal
    superclusters (e.g. 23 Vascular and 5 Fibroblast on set a), so they are
    topped up from cells of other dissections (user decision 2026-09-27; E2
    drew non-neuronal test cells from neocortex outside the frontal set),
    capped per stratum like the rest: a stratum holding ``have`` test cells
    gains at most ``cap - have``. When the top-up would exceed ``room``, the
    cap of the topped-up strata is lowered (water filling) until it fits.

    Args:
        candidates: Extra candidate cells (index = cell id).
        stratum: Column to stratify on (human supercluster).
        have: Test cells each stratum already holds.
        cap: The per-stratum cap of the test set (``stratum_cap``).
        room: Cells the test set may still take, or ``None``.
        seed: Sampling seed.

    Returns:
        Selected candidate ids, in ``candidates`` order.
    """
    available = candidates.groupby(stratum, observed=True).size()

    def taken(limit: int) -> dict[str, int]:
        return {
            str(name): min(int(count), max(0, limit - int(have.get(str(name), 0))))
            for name, count in available.items()
        }

    limit = int(cap)
    if room is not None:
        while limit > 0 and sum(taken(limit).values()) > max(0, int(room)):
            limit -= 1
    takes = taken(limit)
    rng = np.random.default_rng(int(seed))
    chosen: set[str] = set()
    for name, group in candidates.groupby(stratum, observed=True, sort=True):
        take = takes.get(str(name), 0)
        if take <= 0:
            continue
        picked = rng.choice(len(group), size=take, replace=False)
        chosen.update(str(value) for value in group.index[np.sort(picked)])
    return pd.Index([str(value) for value in candidates.index if str(value) in chosen])


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


CellsRule = Callable[[pd.DataFrame], pd.DataFrame]
COP_CLASS: Final = "COP"


def whb_cop_rule(
    thresholds: AnnotationThresholds,
    *,
    min_depth: int | None = None,
    level: str = "broad",
    source_level: str = "supercluster",
) -> CellsRule:
    """Return the §5.2 COP rule for the WHB broad level of a cells table.

    Production gives a WHB COP (committed oligodendrocyte precursor) call
    broad OPC only with at least the COP supercluster floor (120 counts) and
    supercluster bp >= ``whb_supercluster``; otherwise the cell stays at
    lineage (``flag_cop_suppressed``). The self-map applies the same rule,
    so broad OPC is not charged for COP errors production suppresses (M3b
    review): a broad call whose supercluster call is COP and fails the rule
    is excluded from the level (``parent`` set to ``None``), as calls to
    sinks and region-implausible superclusters are, because production
    never gives it a broad label whatever its bp; it counts neither as a
    confident call nor in the coverage. A simulated cell stands for the
    production cells of its depth bin, so the bin (``depth``) is compared
    with the floor. SEA-AD's confident-OPC rescue is not applied (the WHB
    self-map has no SEA-AD call), so broad OPC is measured conservatively.

    Args:
        thresholds: Thresholds (``whb_supercluster``).
        min_depth: The COP count floor (default: the packaged human
            supercluster COP floor, the maximum over platforms).
        level: The level the rule suppresses.
        source_level: The level holding the supercluster calls.

    Returns:
        ``cells -> cells`` (a copy when anything changes).
    """
    if min_depth is None:
        from merxen.annotation.vocab import load_floor_table

        min_depth = known_floor(
            load_floor_table("human"), "supercluster", COP_CLASS, hard_floor=0
        )
    floor = int(min_depth)
    keys = ["recipe", "seed", "sim_id"]

    def apply(cells: pd.DataFrame) -> pd.DataFrame:
        levels = cells["level"].astype(str)
        source = cells[
            (levels == source_level) & (cells["parent"].astype(str) == COP_CLASS)
        ]
        if source.empty or not (levels == level).any():
            return cells
        passes = (source["depth"].to_numpy(np.int64) >= floor) & meets_threshold(
            np.nan_to_num(source["bp"].to_numpy(np.float64), nan=-1.0),
            thresholds.whb_supercluster,
        )
        failed = source.loc[~passes, keys].astype(str)
        if failed.empty:
            return cells
        suppressed = pd.MultiIndex.from_frame(failed)
        target = cells[keys].astype(str)
        hit = (levels == level).to_numpy() & pd.MultiIndex.from_frame(target).isin(
            suppressed
        )
        if not hit.any():
            return cells
        frame = cells.copy()
        parents = frame["parent"].astype(object).to_numpy(copy=True)
        parents[hit] = None
        frame["parent"] = parents
        return frame

    return apply


def whb_cells_rules(thresholds: AnnotationThresholds) -> list[CellsRule]:
    """Return the production rules the WHB self-map applies to its cells table."""
    return [whb_cop_rule(thresholds)]


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


LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "recipe",
    "level",
    "parent",
    "call",
    "truth",
    "truth_parent",
)


def restore_labels(cells: pd.DataFrame) -> pd.DataFrame:
    """Return a stored cells table with object label columns (``None`` if missing).

    Args:
        cells: ``coerce_cells`` output, e.g. read from
            ``resolvability_cells.parquet``.

    Returns:
        The table with its label columns as objects (in place and returned).
    """
    for column in LABEL_COLUMNS:
        cells[column] = cells[column].astype(object).where(cells[column].notna(), None)
    return cells


def as_stored(cells: pd.DataFrame) -> pd.DataFrame:
    """Return a cells table exactly as RESOLVE reads it back from the bundle.

    PREP decides on this, so its summary and RESOLVE's re-derived decisions
    see identical inputs (float32 ``bp``: the isotonic ties and knots follow
    the stored values; M3b review 2).

    Args:
        cells: ``level_cells`` rows (possibly concatenated).

    Returns:
        ``restore_labels(coerce_cells(cells))``.
    """
    return restore_labels(coerce_cells(cells))


# --------------------------------------------------------------------------
# Decisions


WeightTrimScope = Literal["judged_set", "bin"]
WEIGHT_TRIM_SCOPES: Final[tuple[WeightTrimScope, ...]] = ("judged_set", "bin")
# The trimming unit of bundles whose summary settings predate the field
# (resolvability versions 2-3).
LEGACY_WEIGHT_TRIM_SCOPE: Final[WeightTrimScope] = "bin"


def trim_weights(weights: np.ndarray, factor: float) -> np.ndarray:
    """Cap weights at ``factor`` x their median positive weight.

    Args:
        weights: Non-negative weights of one trimming unit.
        factor: The cap in medians (``<= 0``: no trim).

    Returns:
        The capped weights (a copy; unchanged when there is nothing to cap).
    """
    values = np.asarray(weights, dtype=np.float64)
    positive = values[values > 0]
    if factor <= 0 or not len(positive):
        return values.copy()
    return np.minimum(values, factor * float(np.median(positive)))


@dataclass(frozen=True)
class RuleSettings:
    """Settings of the local rule, emission and trust tests (§3.7, §8.2, §8.3).

    Attributes:
        min_cells_per_bin: Test cells of a class a depth bin needs (D_max).
        min_confident_n: Confident calls a tested set needs (a bin, else the
            pooled deep set it takes the verdict of; check half).
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
        weight_min_type_cells: Test cells a truth type needs in a depth bin
            to be reweighted on its own (``composition_weights``).
        weight_trim_factor: Composition weights are capped at this multiple
            of the median positive weight of their trimming unit
            (``weight_trim_scope``).
        weight_trim_scope: The trimming unit: ``judged_set`` (resolvability
            version 4, M3b review 2: each tested set, i.e. a (level, class,
            depth) bin's calls or a pooled deep set, trimmed by ``decide``)
            or ``bin`` (versions 2-3: every row of a (recipe, seed, level,
            depth) bin, trimmed by ``composition_weights``; kept to re-derive
            older bundles). Trimming against the whole bin let the common
            types of the bin (neurons) set the cap of every rarer type, so
            deep glial types all hit the same cap and reweighting did nothing
            inside a called glial class.
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
    weight_min_type_cells: int = 20
    weight_trim_factor: float = 10.0
    weight_trim_scope: WeightTrimScope = "judged_set"

    def __post_init__(self) -> None:
        """Check the trimming unit."""
        if self.weight_trim_scope not in WEIGHT_TRIM_SCOPES:
            raise ResolvabilityError(
                f"weight_trim_scope must be one of {WEIGHT_TRIM_SCOPES}, "
                f"got {self.weight_trim_scope!r}"
            )

    @property
    def bin_trim_factor(self) -> float:
        """The ``composition_weights`` trim factor (0 unless trimming per bin)."""
        return self.weight_trim_factor if self.weight_trim_scope == "bin" else 0.0

    @property
    def set_trim_factor(self) -> float:
        """The per-tested-set trim factor ``decide`` applies (0: none)."""
        return (
            self.weight_trim_factor if self.weight_trim_scope == "judged_set" else 0.0
        )

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
            weight_min_type_cells=resolvability.weight_min_type_cells,
            weight_trim_factor=resolvability.weight_trim_factor,
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
    """The checked statistics of one threshold in one bin.

    Attributes:
        threshold: The threshold (``None``: nothing is confident).
        n_called: Calls checked (with a positive weight).
        n_confident: Calls at or above the threshold (with a positive weight:
            a zero-weight call is of a type the dataset lacks and adds
            nothing to the precision, so it cannot count towards
            ``min_confident_n`` either; M3b review 2).
        n_effective: Kish effective n of the confident calls' weights (the n
            of the Wilson bound).
        precision: Weighted precision of the confident calls.
        wilson_lb: Wilson 95% lower bound of that precision.
        coverage: Weighted share of the checked calls that are confident.
        max_weight_share: The largest single confident call's share of the
            confident weight (1 / n_confident when unweighted).
    """

    threshold: float | None
    n_called: int
    n_confident: int
    n_effective: float
    precision: float
    wilson_lb: float
    coverage: float
    max_weight_share: float = math.nan


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
        The statistics (Wilson bound on the Kish effective n; calls counted
        only with a positive weight, as the isotonic fit uses them).
    """
    positive = np.asarray(weights, dtype=np.float64) > 0
    n_called = int(positive.sum())
    if threshold is None or n_called == 0:
        return CheckStats(threshold, n_called, 0, 0.0, math.nan, math.nan, 0.0)
    accepted = meets_threshold(bp, threshold) & positive
    accepted_weights = weights[accepted]
    total = float(accepted_weights.sum())
    called_total = float(weights[positive].sum())
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
        max_weight_share=float(accepted_weights.max()) / total,
    )


REASON_NO_LOCAL_THRESHOLD: Final = "no_local_threshold"
REASON_TOO_FEW_CONFIDENT: Final = "too_few_confident_calls"
REASON_WILSON: Final = "wilson_bound_below_target"
REASON_POINT: Final = "point_precision_below_target"
REASON_COVERAGE: Final = "coverage_below_minimum"
REASON_TOO_FEW_TEST_CELLS: Final = "too_few_test_cells"
# A class whose calls never reach min_confident_n, even with every depth bin
# pooled (user decision 2026-09-27, H18 follow-up).
REASON_INSUFFICIENT_CALLS: Final = "insufficient_calls"
# Prefix of the reason of a bin whose own test passed but whose deep pool
# (the ">= D_P" set it completes) failed.
POOL_REASON_PREFIX: Final = "pool_"


def _rule_pass(stats: CheckStats, target: float, settings: RuleSettings) -> str | None:
    """Return why a tested set fails the emission rule (``None``: passes).

    One rule for single bins and pooled deep sets, as the gate-P evaluation
    rules (§14; user decision 2026-09-27): a threshold, at least
    ``min_confident_n`` confident calls, the Wilson lower bound (on the Kish
    effective n) at least ``target - wilson_margin``, the point precision at
    least ``target`` and the coverage at least ``min_coverage``. The Wilson
    test comes first, so ``point_precision_below_target`` marks exactly the
    sets the point estimate alone rejects.
    """
    if stats.threshold is None:
        return REASON_NO_LOCAL_THRESHOLD
    if stats.n_confident < settings.min_confident_n:
        return REASON_TOO_FEW_CONFIDENT
    if not stats.wilson_lb >= target - settings.wilson_margin - _TOLERANCE:
        return REASON_WILSON
    if not stats.precision >= target - _TOLERANCE:
        return REASON_POINT
    if stats.coverage < settings.min_coverage - _TOLERANCE:
        return REASON_COVERAGE
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


def deepest_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return each test cell's deepest simulated row (one row per ``cell_id``).

    A test cell is simulated at every grid depth its native counts reach, so
    a ">= d" set of deep bins holds each test cell once, at its deepest bin
    (§14 gate-P evaluation rules, §8.3 pooled deep bins). The deepest row is
    taken over every row of a level, whatever the call, before any class or
    confidence filter: a cell called into another class (or a sink) at its
    deepest bin is not a call of the class there, and a cell unconfident at
    its deepest bin is not rescued by a confident shallower row.

    Args:
        rows: Cells-table rows of one level (one recipe and seed).

    Returns:
        One row per test cell, in depth order.
    """
    if rows.empty:
        return rows
    order = np.argsort(rows["depth"].to_numpy(np.int64), kind="mergesort")
    ordered = rows.iloc[order]
    return ordered[~ordered["cell_id"].astype(str).duplicated(keep="last").to_numpy()]


def _judged(record: Mapping[str, Any], settings: RuleSettings) -> bool:
    """Whether a tested set holds enough calls for the rule to decide it.

    Enough confident calls (``min_confident_n``), or a fitted regime whose
    isotonic fit exists and never reaches the target on at least
    ``min_confident_n`` checked calls (``no_local_threshold``: more calls
    would not change the verdict). Only calls with a positive weight count
    (``check_threshold``): a set of zero-weight calls (types the dataset
    lacks) is not judged and takes the pooled verdict. There is no fallback
    to the unweighted verdict: a set without weight has no dataset cell
    behind it, so its verdict reaches no cell.
    """
    n_confident = record.get("n_confident")
    if (
        n_confident is not None
        and not pd.isna(n_confident)
        and int(n_confident) >= settings.min_confident_n
    ):
        return True
    n_called = record.get("n_called")
    return (
        record.get("reason") == REASON_NO_LOCAL_THRESHOLD
        and n_called is not None
        and not pd.isna(n_called)
        and int(n_called) >= settings.min_confident_n
    )


def _pool_min_depth(
    own: Mapping[int, Mapping[str, dict[str, Any]]],
    pooled_at: Callable[[int], Mapping[str, dict[str, Any]]],
    regime: Regime,
    grid: Sequence[int],
    settings: RuleSettings,
) -> tuple[int | None, bool]:
    """Return ``(D_P, insufficient)`` of one (regime, level, class).

    ``D_P`` is ``None`` when the deepest grid bin is judged on its own (no
    pool); otherwise bins are pooled from the deep end until the ">= d" set
    is judged, and ``D_P`` is its shallowest bin. ``insufficient`` is true
    when even the set of every bin is not judged.
    """
    if _judged(own[grid[-1]][regime], settings):
        return None, False
    for depth in reversed(grid):
        if _judged(pooled_at(depth)[regime], settings):
            return depth, False
    return None, True


def decide(
    cells: pd.DataFrame,
    levels: Sequence[LevelMeta],
    depths: Sequence[int],
    settings: RuleSettings,
    *,
    weights: np.ndarray | None = None,
    pool_weights: PoolWeights | None = None,
    recipe: str = DECISION_RECIPE,
    seed: int = 0,
    saturated_bp_share: float | None = None,
) -> pd.DataFrame:
    """Return the per (regime, level, class, depth) decisions with pooled deep bins.

    Every bin is tested on its own: the isotonic fit on the fit half, the
    local thresholds for the base and provisional targets, and the rule
    (``_rule_pass``) at the applied threshold (``validated``: the default,
    checked on every call of the bin; ``provisional`` / ``trust``: ``t*``,
    checked on the check half). A bin with at least ``min_confident_n``
    confident calls keeps its own verdict (``_judged``).

    Pooled deep bins (user decision 2026-09-27; the gate-P evaluation rules
    of §14 with ``n_min = min_confident_n``): when the deepest bin is not
    judged on its own, bins are pooled from the deep end into a ">= d" set,
    each test cell counted once at its deepest bin (``deepest_rows``), until
    the set is judged; its shallowest bin is ``D_P``. The set is tested with
    the same rule (its own fit and ``t*``, the target of ``D_P``, the Wilson
    bound on the Kish n). Every bin at or above ``D_P`` that is judged on
    its own keeps its own verdict and is emitted only if the set passes too
    (a pooled pass never overrides a bin's own failure, a pooled failure
    withdraws its pass: ``pool_<reason>``); every bin at or above ``D_P``
    that is not judged on its own takes the set's verdict and is marked
    ``pooled``, and ``extrapolated`` when it lies deeper than ``D_P`` (M3b
    review 2; §8.3, §14). A class whose calls are not judged even with
    every bin pooled is ``not_resolvable`` (``insufficient_calls``) in every
    bin not judged on its own. Classes without ``D_max`` (no bin with
    ``min_cells_per_bin`` test cells) are never emitted
    (``too_few_test_cells``; no pooling across classes).

    Weighted rows (RESOLVE) enter the precision with their weight and the
    Wilson bound with the Kish effective n; only positive-weight calls count
    towards ``min_confident_n``; ``max_weight_share`` records the largest
    single call's share of a set's confident weight. With
    ``weight_trim_scope`` ``judged_set`` each tested set's weights are
    capped at ``weight_trim_factor`` x its median positive weight before the
    fit and the checks. A pooled set is reweighted as one set when
    ``pool_weights`` is given (``pooled_composition_weights``; RESOLVE),
    else its rows keep their per-bin weights. ``would_raise`` is set only
    where the fit half holds ``min_cells_per_bin`` calls
    (``would_raise_evaluable``).

    Args:
        cells: The cells table (``level_cells`` rows).
        levels: Level metadata (defaults, targets, roles).
        depths: The depth grid.
        settings: Rule settings.
        weights: Optional per-row weights (composition reweighting, RESOLVE).
        pool_weights: Optional weights of pooled deep sets (RESOLVE).
        recipe: Recipe the decisions use.
        seed: Mapping seed the decisions use.
        saturated_bp_share: Resolvability version 7 only (v7.8, the
            saturated-bp rule): a fitted set without ``t*`` whose
            positive-weight fit-half calls hold more than this share at
            ``bp = 1`` is judged at ``threshold_cap``; the table then adds
            ``threshold_source``, ``saturated_bp`` and ``saturated_share``.
            ``None`` (version 6) leaves every row and column unchanged.

    Returns:
        One row per (regime, level, class, depth): ``status`` (``emitted`` or
        ``not_resolvable``), ``threshold`` (applied), ``t_star``,
        ``would_raise``, ``would_raise_evaluable``, ``check_set`` (``all`` or
        ``check_half``),
        statistics (of the pooled set for pooled bins), ``target``,
        ``d_max``, ``extrapolated``, ``pooled``, ``pool_min_depth``
        (``D_P``), ``own_status`` / ``own_reason`` / ``own_n_confident`` (the
        bin's own test), ``reason``.
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
        pools = _LevelPools(deepest_rows(level_rows), pool_weights)
        classes = sorted(
            {str(cls) for cls in called["parent"].dropna().astype(str)}
            | {cls for (level, cls) in dmax if level == meta.level}
        )
        for cls in classes:
            records.extend(
                _class_decisions(
                    meta,
                    cls,
                    grid,
                    settings,
                    groups=groups,
                    pools=pools,
                    class_dmax=dmax.get((meta.level, cls)),
                    n_test_index=n_test_index,
                    saturated_bp_share=saturated_bp_share,
                )
            )
    columns = list(DECISION_COLUMNS)
    text_columns = set(DECISION_TEXT_COLUMNS)
    if saturated_bp_share is not None:
        columns += list(SATURATED_COLUMNS)
        text_columns.add("threshold_source")
    table = pd.DataFrame.from_records(records)
    for column in columns:
        if column not in table.columns:
            table[column] = np.nan if column not in text_columns else None
    table = table[columns]
    table["extrapolated"] = _bool_column(table["extrapolated"])
    table["would_raise"] = _bool_column(table["would_raise"])
    table["would_raise_evaluable"] = _bool_column(table["would_raise_evaluable"])
    table["pooled"] = _bool_column(table["pooled"])
    if saturated_bp_share is not None:
        table["saturated_bp"] = _bool_column(table["saturated_bp"])
    return table


# ``decide`` output columns (version 6; version 7 adds ``SATURATED_COLUMNS``
# and ``V7_DECISION_COLUMNS``).
DECISION_COLUMNS: Final[tuple[str, ...]] = (
    "regime",
    "level",
    "class",
    "depth",
    "status",
    "threshold",
    "t_star",
    "would_raise",
    "would_raise_evaluable",
    "target",
    "default_threshold",
    "n_test",
    "check_set",
    "n_called",
    "n_fit",
    "n_confident",
    "n_effective",
    "max_weight_share",
    "precision",
    "wilson_lb",
    "coverage",
    "g_at_default",
    "precision_at_default",
    "d_max",
    "extrapolated",
    "pooled",
    "pool_min_depth",
    "own_status",
    "own_reason",
    "own_n_confident",
    "reason",
)
DECISION_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "status",
    "reason",
    "check_set",
    "own_status",
    "own_reason",
)


class _LevelPools:
    """The pooled ">= d" sets of one level (``decide``): each test cell once.

    Holds the level's deepest rows (every call, ``deepest_rows``) and, when
    ``pool_weights`` is given, reweights each ">= d" set as one set (cached
    per ``d``); a class's pooled set is its called rows there.
    """

    def __init__(self, deepest: pd.DataFrame, pool_weights: PoolWeights | None) -> None:
        self.deepest = deepest
        self.pool_weights = pool_weights
        self._by_depth: dict[int, pd.DataFrame] = {}

    def rows(self, cls: str, depth: int) -> pd.DataFrame:
        """Return the pooled ">= depth" rows called ``cls``."""
        if depth not in self._by_depth:
            subset = self.deepest[self.deepest["depth"].to_numpy(np.int64) >= depth]
            if self.pool_weights is not None:
                subset = subset.copy()
                subset["_weight"] = self.pool_weights(subset, depth)
            self._by_depth[depth] = subset[subset["parent"].notna().to_numpy()]
        subset = self._by_depth[depth]
        return subset[(subset["parent"].astype(str) == cls).to_numpy()]


def _class_decisions(
    meta: LevelMeta,
    cls: str,
    grid: Sequence[int],
    settings: RuleSettings,
    *,
    groups: Mapping[tuple[str, int], pd.DataFrame],
    pools: _LevelPools,
    class_dmax: int | None,
    n_test_index: Mapping[tuple[str, str, int], int],
    saturated_bp_share: float | None = None,
) -> list[dict[str, Any]]:
    """Decide every (regime, depth) bin of one (level, class) (``decide``)."""

    def base(depth: int) -> dict[str, Any]:
        return {
            "level": meta.level,
            "class": cls,
            "depth": depth,
            "n_test": n_test_index.get((meta.level, cls, depth), 0),
            "d_max": class_dmax,
            "default_threshold": meta.default_threshold,
        }

    if class_dmax is None:
        return [
            {
                **base(depth),
                "regime": regime,
                "target": settings.target(regime, meta.base_target, depth),
                "status": STATUS_NOT_RESOLVABLE,
                "extrapolated": False,
                "pooled": False,
                "reason": REASON_TOO_FEW_TEST_CELLS,
            }
            for depth in grid
            for regime in REGIMES
        ]
    own: dict[int, dict[str, dict[str, Any]]] = {}
    for depth in grid:
        own[depth] = {
            record["regime"]: record
            for record in _bin_decisions(
                groups.get((cls, depth)),
                meta,
                depth,
                settings,
                base(depth),
                saturated_bp_share=saturated_bp_share,
            )
        }
    pooled: dict[int, dict[str, dict[str, Any]]] = {}

    def pooled_at(depth: int) -> dict[str, dict[str, Any]]:
        if depth not in pooled:
            pooled[depth] = {
                record["regime"]: record
                for record in _bin_decisions(
                    pools.rows(cls, depth),
                    meta,
                    depth,
                    settings,
                    base(depth),
                    saturated_bp_share=saturated_bp_share,
                )
            }
        return pooled[depth]

    records: list[dict[str, Any]] = []
    for regime in REGIMES:
        pool_min, insufficient = _pool_min_depth(own, pooled_at, regime, grid, settings)
        for depth in grid:
            mine = own[depth][regime]
            own_fields = {
                "own_status": mine["status"],
                "own_reason": mine["reason"],
                "own_n_confident": mine.get("n_confident"),
                "pool_min_depth": pool_min,
            }
            judged = _judged(mine, settings)
            if insufficient and not judged:
                records.append(
                    {
                        **mine,
                        **own_fields,
                        "status": STATUS_NOT_RESOLVABLE,
                        "reason": REASON_INSUFFICIENT_CALLS,
                        "pooled": False,
                    }
                )
            elif pool_min is None or depth < pool_min:
                records.append({**mine, **own_fields, "pooled": False})
            elif judged:
                # Judged on its own at or above D_P: its own verdict, withdrawn
                # when the pool it belongs to fails (a pooled pass never
                # overrides its own failure; M3b review 2).
                verdict = pooled_at(pool_min)[regime]
                record = {**mine, **own_fields, "pooled": False}
                if mine["status"] == STATUS_EMITTED and (
                    verdict["status"] != STATUS_EMITTED
                ):
                    record.update(
                        status=STATUS_NOT_RESOLVABLE,
                        reason=f"{POOL_REASON_PREFIX}{verdict['reason']}",
                    )
                records.append(record)
            else:
                # Not judged on its own: the pool's verdict; only bins deeper
                # than D_P are extrapolated (§14).
                verdict = pooled_at(pool_min)[regime]
                records.append(
                    {
                        **verdict,
                        **own_fields,
                        "depth": depth,
                        "n_test": mine["n_test"],
                        "extrapolated": depth > pool_min,
                        "pooled": True,
                    }
                )
    return records


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
    *,
    saturated_bp_share: float | None = None,
) -> list[dict[str, Any]]:
    """Decide one (level, class, depth) bin under every regime.

    ``provisional`` and ``trust`` fit ``t*`` on the fit half and check it on
    the other half (out of sample). ``validated`` applies the pre-registered
    default, which is not fitted on these cells, so it is checked on every
    call of the bin (both halves): halving it would only cost power
    (M3b review). Its fit-half ``t*`` is reported (``would_raise``), never
    applied. With ``saturated_bp_share`` (version 7, v7.8) a fitted set
    without ``t*`` whose fit-half calls are saturated is judged at the cap
    (``saturated_cap``).
    """
    if group is None or len(group) == 0:
        empty = [
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
        if saturated_bp_share is not None:
            for record in empty:
                record.update(
                    threshold_source=None, saturated_bp=False, saturated_share=math.nan
                )
        return empty
    bp = group["bp"].to_numpy(np.float64)
    correct = group["correct"].to_numpy(bool).astype(np.float64)
    weights = trim_weights(
        group["_weight"].to_numpy(np.float64), settings.set_trim_factor
    )
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
    at_default = check_threshold(bp, correct, weights, meta.default_threshold)
    g_default = (
        float(fit.predict(meta.default_threshold)) if fit is not None else math.nan
    )
    saturated_share = math.nan
    if saturated_bp_share is not None and fit is not None:
        saturated_share = saturated_bp_fraction(bp[fit_mask])
    records = []
    for regime in REGIMES:
        target = settings.target(regime, meta.base_target, depth)
        t_star = local_threshold(
            fit,
            default=meta.default_threshold,
            target=target,
            cap=settings.threshold_cap,
        )
        saturated = False
        if regime == "validated":
            applied: float | None = meta.default_threshold
            stats = at_default
            check_set = "all"
        else:
            applied = t_star
            saturated = (
                saturated_bp_share is not None
                and t_star is None
                and fit is not None
                and saturated_share > saturated_bp_share
            )
            if saturated:
                applied = settings.threshold_cap
            stats = check_threshold(check_bp, check_correct, check_weights, applied)
            check_set = "check_half" if settings.split_halves else "all"
        reason = _rule_pass(stats, target, settings)
        if fit is None and regime != "validated":
            reason = "too_few_fit_cells"
        extra: dict[str, Any] = {}
        if saturated_bp_share is not None:
            extra = {
                "threshold_source": _threshold_source(regime, applied, saturated),
                "saturated_bp": bool(saturated),
                "saturated_share": saturated_share,
            }
        records.append(
            {
                **base,
                "regime": regime,
                "target": target,
                "threshold": applied,
                "t_star": t_star,
                # Without a fit (fewer than min_cells_per_bin fit-half calls)
                # t* is unknown, not "above the default" (M3b review 2).
                "would_raise": bool(
                    fit is not None
                    and (t_star is None or t_star > meta.default_threshold + _TOLERANCE)
                ),
                "would_raise_evaluable": fit is not None,
                "check_set": check_set,
                "n_called": stats.n_called,
                "n_fit": int(fit_mask.sum()),
                "n_confident": stats.n_confident,
                "n_effective": stats.n_effective,
                "max_weight_share": stats.max_weight_share,
                "precision": stats.precision,
                "wilson_lb": stats.wilson_lb,
                "coverage": stats.coverage,
                "g_at_default": g_default,
                "precision_at_default": at_default.precision,
                "status": STATUS_EMITTED if reason is None else STATUS_NOT_RESOLVABLE,
                "extrapolated": False,
                "reason": reason,
                **extra,
            }
        )
    return records


# The saturated-bp rule (resolvability version 7, v7.8; pre-registration
# §14.3: share 0.90 and cap 0.99, only tightenable). A call counts as
# saturated at bp >= 1 - 1e-6 (the float32 tolerance of the stored bp).
SATURATED_BP_TOLERANCE: Final = 1e-6
SATURATED_COLUMNS: Final[tuple[str, ...]] = (
    "threshold_source",
    "saturated_bp",
    "saturated_share",
)
THRESHOLD_SOURCE_DEFAULT: Final = "default"
THRESHOLD_SOURCE_LOCAL: Final = "resolvability_local"
THRESHOLD_SOURCE_SATURATED: Final = "saturated_cap"
THRESHOLD_SOURCE_INHERITED: Final = "monotone_inherited"


def saturated_bp_fraction(bp: np.ndarray) -> float:
    """Return the share of calls at ``bp = 1`` (``nan`` for no call; v7.8).

    Args:
        bp: The fit-half bp values of a tested set's positive-weight calls.

    Returns:
        The share with ``bp >= 1 - SATURATED_BP_TOLERANCE``.
    """
    values = np.asarray(bp, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    return float(np.mean(values >= 1.0 - SATURATED_BP_TOLERANCE))


def _threshold_source(
    regime: Regime, applied: float | None, saturated: bool
) -> str | None:
    """Where a version-7 set's applied threshold comes from."""
    if regime == "validated":
        return THRESHOLD_SOURCE_DEFAULT
    if saturated:
        return THRESHOLD_SOURCE_SATURATED
    return None if applied is None else THRESHOLD_SOURCE_LOCAL


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


@dataclass(frozen=True)
class DatasetComposition:
    """A dataset's soft composition over truth types, overall and per depth bin.

    RESOLVE reweights the simulated cells of each depth bin to the dataset's
    cells in the same bin (``composition_weights``): low-count cells are
    often called into a few sink-like types (e.g. COP), so one dataset-wide
    composition would over-weight those types at every depth (M3b review).

    Attributes:
        overall: Share (any scale) per truth type over all table cells.
        by_depth: Share per truth type of the cells in each depth bin.
        bin_mass: Total mass (cells) per depth bin.
        min_bin_mass: Bins with less mass use ``overall``.
    """

    overall: Mapping[str, float]
    by_depth: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    bin_mass: Mapping[int, float] = field(default_factory=dict)
    min_bin_mass: float = 50.0

    def at(self, depth: int) -> Mapping[str, float]:
        """Return the composition the simulated cells of a depth bin follow.

        Args:
            depth: Grid depth.

        Returns:
            The bin's composition, or ``overall`` when the bin holds fewer
            than ``min_bin_mass`` cells.
        """
        mass = float(self.bin_mass.get(int(depth), 0.0))
        shares = self.by_depth.get(int(depth))
        if shares is None or mass < self.min_bin_mass - _TOLERANCE:
            return self.overall
        return shares

    def at_least(self, depth: int) -> Mapping[str, float]:
        """Return the composition a pooled ">= depth" set follows.

        The dataset's cells of every bin at or above ``depth``: each bin's
        composition normalised and weighted by its mass (a pooled deep set
        stands for those cells; ``decide``).

        Args:
            depth: The pool's shallowest bin (``D_P``).

        Returns:
            The pooled composition, or ``overall`` when those bins hold fewer
            than ``min_bin_mass`` cells.
        """
        total: dict[str, float] = {}
        mass = 0.0
        for bin_depth, shares in self.by_depth.items():
            if int(bin_depth) < int(depth):
                continue
            bin_mass = float(self.bin_mass.get(int(bin_depth), 0.0))
            mass += bin_mass
            for name, value in _normalised(shares).items():
                total[name] = total.get(name, 0.0) + value * bin_mass
        if not total or mass < self.min_bin_mass - _TOLERANCE:
            return self.overall
        return total

    @classmethod
    def from_cells(
        cls,
        types: Sequence[object],
        mass: np.ndarray | Sequence[float],
        counts: np.ndarray | Sequence[float],
        grid: Sequence[int],
        *,
        min_bin_mass: float = 50.0,
    ) -> DatasetComposition:
        """Sum soft type mass over a dataset's cells, overall and per depth bin.

        Args:
            types: Truth type of each contribution (one cell may contribute
                several: its assigned node and runner-ups, §5.5).
            mass: The contribution (e.g. bootstrap probability).
            counts: The contributing cell's total counts.
            grid: The bundle's depth grid.
            min_bin_mass: ``DatasetComposition.min_bin_mass``.

        Returns:
            The composition (``bin_mass`` counts cells as their total mass).
        """
        labels = np.asarray([_clean_label(value) for value in types], dtype=object)
        values = np.nan_to_num(np.asarray(mass, dtype=np.float64), nan=0.0)
        bins = depth_bin(counts, grid)
        keep = np.array([label is not None for label in labels]) & (values > 0)
        frame = pd.DataFrame(
            {"type": labels[keep], "mass": values[keep], "bin": bins[keep]}
        )
        overall = {
            str(name): float(value)
            for name, value in frame.groupby("type")["mass"].sum().items()
        }
        by_depth: dict[int, dict[str, float]] = {}
        bin_mass: dict[int, float] = {}
        inside = frame[np.isfinite(frame["bin"].to_numpy(np.float64))]
        for value, rows in inside.groupby("bin"):
            depth = int(value)
            by_depth[depth] = {
                str(name): float(total)
                for name, total in rows.groupby("type")["mass"].sum().items()
            }
            bin_mass[depth] = float(rows["mass"].sum())
        return cls(
            overall=overall,
            by_depth=by_depth,
            bin_mass=bin_mass,
            min_bin_mass=min_bin_mass,
        )


BROAD_LEVEL: Final = "broad"


def leaf_class_map(
    cells: pd.DataFrame, *, key: str = TRUTH_LEAF_COLUMN
) -> dict[str, str]:
    """Return each truth type's broad class (the pooling unit of rare types).

    Read from the ``broad`` level's truth classes (human: E2 floor classes,
    COP inside OPC; mouse: WMB classes), else from any level's.

    Args:
        cells: The cells table.
        key: Column holding the truth type.

    Returns:
        Broad class per truth type (types without one are left out).
    """
    frame = cells[cells["truth_parent"].notna()]
    broad = frame[frame["level"].astype(str) == BROAD_LEVEL]
    source = broad if len(broad) else frame
    pairs = source[[key, "truth_parent"]].astype(str).drop_duplicates(key)
    return dict(zip(pairs[key], pairs["truth_parent"], strict=True))


def composition_weights(
    cells: pd.DataFrame,
    composition: Mapping[str, float] | DatasetComposition,
    *,
    key: str = TRUTH_LEAF_COLUMN,
    class_of: Mapping[str, str] | None = None,
    min_type_cells: int = 20,
    trim_factor: float = 0.0,
) -> np.ndarray:
    """Return per-row weights that reweight the test cells to a composition.

    Within each (recipe, seed, level, depth) the test cells' truth types are
    reweighted to the dataset's composition of that depth bin (§8.3
    composition reweighting, §5.5), written down in the M3b review:

    1. ``q``: the dataset's composition in the bin (``DatasetComposition.at``;
       a plain mapping is used for every bin);
    2. a type with at least ``min_type_cells`` test cells in the bin gets
       ``w = q(type) / p(type)`` (``p``: its share of the bin's test cells);
    3. a rarer type is weighted at its broad class's level, so a handful of
       test cells cannot stand for a large composition share on their own:
       it takes the weight of the class's common types,
       ``sum q / sum p`` over them, or, when the class has no common type
       in the bin, ``q(class) / p(class)`` over its types present;
    4. only when ``trim_factor > 0`` (``weight_trim_scope`` ``bin``,
       resolvability versions 2-3): weights above ``trim_factor`` x the
       bin's median positive weight are capped; from version 4 ``decide``
       trims within each judged set instead (``RuleSettings``), because the
       bin's median is set by its most common test types (neurons) and gave
       every deep glial type the same cap (M3b review 2);
    5. the bin is renormalised to a mean weight of 1.

    Types absent from the composition get weight 0; composition mass on
    types without test cells in a bin cannot be represented and is dropped
    there. Decisions record the Kish n and the largest single call's weight
    share of each bin (``decide``).

    Args:
        cells: The cells table.
        composition: Dataset share per truth type, keyed like ``truth_leaf``
            (human: WHB supercluster labels, e.g. ``CS202210140_476``;
            mouse: WMB subclass labels), any scale; a
            ``DatasetComposition`` gives one per depth bin.
        key: Column holding the truth type.
        class_of: Broad class per truth type (default ``leaf_class_map``).
        min_type_cells: Test cells a type needs in a bin to be weighted on
            its own.
        trim_factor: Cap on a weight, in bin medians (``<= 0``, the default:
            no trim here; ``RuleSettings.bin_trim_factor``).

    Returns:
        Weights aligned with ``cells``.

    Raises:
        ResolvabilityError: If the composition has no positive share.
    """
    per_depth = (
        composition
        if isinstance(composition, DatasetComposition)
        else DatasetComposition(overall=dict(composition))
    )
    if not sum(max(float(value), 0.0) for value in per_depth.overall.values()) > 0:
        raise ResolvabilityError("the composition has no positive share")
    classes = dict(class_of) if class_of is not None else leaf_class_map(cells, key=key)
    types = cells[key].astype(str).to_numpy()
    weights = np.zeros(len(cells), dtype=np.float64)
    group_keys = ["recipe", "seed", "level", "depth"]
    for group_key, index in cells.groupby(group_keys, observed=True).indices.items():
        share = _normalised(per_depth.at(int(group_key[3])))
        group_types = types[index]
        present, counts = np.unique(group_types, return_counts=True)
        test_share = dict(zip(present, counts / counts.sum(), strict=True))
        n_cells = dict(zip(present, counts, strict=True))
        # Per broad class: (q, p) of its common types and of all its types.
        common: dict[str, list[float]] = {}
        pooled: dict[str, list[float]] = {}
        for name in present:
            unit = classes.get(str(name), str(name))
            q_p = (share.get(str(name), 0.0), float(test_share[name]))
            totals = pooled.setdefault(unit, [0.0, 0.0])
            totals[0] += q_p[0]
            totals[1] += q_p[1]
            if n_cells[name] >= min_type_cells:
                totals = common.setdefault(unit, [0.0, 0.0])
                totals[0] += q_p[0]
                totals[1] += q_p[1]
        type_weight: dict[str, float] = {}
        for name in present:
            if n_cells[name] >= min_type_cells:
                type_weight[name] = share.get(str(name), 0.0) / test_share[name]
                continue
            unit = classes.get(str(name), str(name))
            q_class, p_class = common.get(unit, pooled[unit])
            type_weight[name] = q_class / p_class if p_class > 0 else 0.0
        values = trim_weights(
            np.array([type_weight[name] for name in group_types]), trim_factor
        )
        total = float(values.sum())
        if total > 0:
            values = values * (len(values) / total)
        weights[index] = values
    return weights


PoolWeights = Callable[[pd.DataFrame, int], np.ndarray]


def pooled_composition_weights(
    composition: Mapping[str, float] | DatasetComposition,
    *,
    key: str = TRUTH_LEAF_COLUMN,
    class_of: Mapping[str, str] | None = None,
    min_type_cells: int = 20,
    trim_factor: float = 0.0,
) -> PoolWeights:
    """Return the weights of pooled deep sets (``decide``, RESOLVE).

    A ">= D_P" set holds each test cell once, at its deepest bin, so its
    rows come from several depth bins whose weights ``composition_weights``
    normalised separately; mixing them lets a deep bin where the dataset
    (almost) lacks a type carry a whole set (on ag7, 69 confident calls of
    one class had a Kish n of 1). The set is therefore reweighted as one set
    (gate-P evaluation rules, §14: "reweighted sets") to the dataset's cells
    at depths ``>= D_P`` (``DatasetComposition.at_least``), with the same
    rare-type pooling (and bin trimming, ``trim_factor``) as a single bin;
    ``decide`` trims each class's pooled set like any judged set.

    Args:
        composition: The dataset composition (per depth bin, or one).
        key: Column holding the truth type.
        class_of: Broad class per truth type (computed on the whole cells
            table, so a level without broad rows keeps the broad classes).
        min_type_cells: ``composition_weights`` ``min_type_cells``.
        trim_factor: ``composition_weights`` ``trim_factor``.

    Returns:
        ``(rows, D_P) -> weights``: ``rows`` are a level's deepest rows at
        depths ``>= D_P`` (every class, one recipe and seed).
    """

    def weigh(rows: pd.DataFrame, depth: int) -> np.ndarray:
        if rows.empty:
            return np.zeros(0, dtype=np.float64)
        target = (
            composition.at_least(depth)
            if isinstance(composition, DatasetComposition)
            else composition
        )
        frame = rows.copy()
        frame["depth"] = int(depth)
        return composition_weights(
            frame,
            target,
            key=key,
            class_of=class_of,
            min_type_cells=min_type_cells,
            trim_factor=trim_factor,
        )

    return weigh


def _normalised(composition: Mapping[str, float]) -> dict[str, float]:
    """Return a composition scaled to sum 1 (negative shares count as 0)."""
    total = float(sum(max(float(value), 0.0) for value in composition.values()))
    if total <= 0:
        return {}
    return {
        str(name): max(float(value), 0.0) / total for name, value in composition.items()
    }


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


def packaged_floors(
    floor_table: pd.DataFrame | None, floor_level: str | None, cls: str
) -> list[dict[str, Any]]:
    """Return a class's packaged floors, one per (platform, panel family).

    Args:
        floor_table: The packaged floors (``load_floor_table``), or ``None``.
        floor_level: The floor table's level (``None``: none).
        cls: Floor class.

    Returns:
        ``platform``, ``panel_family``, ``min_counts`` and ``floor_source``
        per row, sorted.
    """
    if floor_table is None or floor_level is None:
        return []
    rows = floor_table[
        (floor_table["level"] == floor_level) & (floor_table["floor_class"] == cls)
    ]
    records = [
        {
            "platform": str(row["platform"]),
            "panel_family": str(row["panel_family"]),
            "min_counts": int(row["min_counts"]),
            "floor_source": str(row.get("floor_source", "packaged")),
        }
        for _, row in rows.iterrows()
    ]
    return sorted(records, key=lambda item: (item["platform"], item["panel_family"]))


def combined_floors(
    decisions: pd.DataFrame,
    levels: Sequence[LevelMeta],
    settings: RuleSettings,
    *,
    floor_table: pd.DataFrame | None,
    species: str,
) -> pd.DataFrame:
    """Return the floors per regime: known, simulated and the one applied.

    ``validated``: the packaged (real-data) floors apply per (level, class,
    platform, panel family) straight from ``floors_<species>.csv``
    (``floor_source`` ``packaged``, ``floor`` left empty, the per-platform
    rows in ``packaged``), the simulated floor reported beside them; a
    level without a packaged table falls back to the hard floor.
    ``provisional``: max(known, simulated) (§5.4 unknown-panel rule; known
    = the maximum over platforms and panels), and at least
    ``provisional_mouse_subclass_floor`` for the mouse subclass level. A
    level never emitted for a class has no provisional floor (``None``): it
    is ``not_resolvable`` there anyway.

    Args:
        decisions: ``decide`` output.
        levels: Level metadata.
        settings: Rule settings.
        floor_table: Packaged floors, or ``None``.
        species: ``human`` or ``mouse``.

    Returns:
        ``regime``, ``level``, ``class``, ``known_floor``, ``simulated_floor``,
        ``floor``, ``floor_source``, ``packaged``.
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
            packaged = packaged_floors(floor_table, meta.floor_level, cls)
            if regime == "validated":
                floor: int | None
                if packaged:
                    floor, source = None, "packaged"
                else:
                    floor, source = settings.hard_floor, "hard_floor"
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
                    "packaged": packaged if regime == "validated" else [],
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
            "packaged",
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
    "pooled",
    "pool_min_depth",
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
        "pool_min_depth",
    ):
        combined[column] = pd.to_numeric(combined[column], errors="coerce").astype(
            "Int64"
        )
    combined["extrapolated"] = combined["extrapolated"].astype("boolean")
    combined["pooled"] = combined["pooled"].astype("boolean")
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


def pooled_sets(decisions: pd.DataFrame, regime: Regime) -> dict[str, dict[str, Any]]:
    """Return the pooled deep sets per level and class (reports, bundle summary).

    Args:
        decisions: ``decide`` output.
        regime: The regime.

    Returns:
        ``{level: {class: record}}`` for the (level, class) pairs with a pool:
        ``min_depth`` (``D_P``), ``depths`` (the bins taking its verdict),
        the set's statistics, threshold, ``t_star``, ``would_raise``,
        ``target``, ``status`` and ``reason``.
    """
    frame = decisions[(decisions["regime"] == regime) & decisions["pooled"]]
    result: dict[str, dict[str, Any]] = {}
    for (level, cls), group in frame.groupby(["level", "class"], observed=True):
        first = group.sort_values("depth").iloc[0]
        result.setdefault(str(level), {})[str(cls)] = {
            "min_depth": None
            if pd.isna(first["pool_min_depth"])
            else int(first["pool_min_depth"]),
            "depths": sorted(int(depth) for depth in group["depth"]),
            "n_called": _optional_float(first["n_called"]),
            "n_confident": _optional_float(first["n_confident"]),
            "n_effective": _optional_float(first["n_effective"]),
            "precision": _optional_float(first["precision"]),
            "wilson_lb": _optional_float(first["wilson_lb"]),
            "coverage": _optional_float(first["coverage"]),
            "threshold": _optional_float(first["threshold"]),
            "t_star": _optional_float(first["t_star"]),
            "would_raise": bool(first["would_raise"]),
            "would_raise_evaluable": bool(first["would_raise_evaluable"]),
            "target": _optional_float(first["target"]),
            "status": str(first["status"]),
            "reason": _clean_label(first["reason"]),
        }
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
    cells_rules: Sequence[CellsRule] = (),
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
        cells_rules: Production rules applied to each recipe's cells table
            before the decisions (WHB: ``whb_cells_rules``, the COP rule).

    Returns:
        The result.
    """
    if not recipes:
        raise ResolvabilityError("run_resolvability needs at least one recipe")
    started = time.monotonic()
    timings: dict[str, float] = {}
    frames: list[pd.DataFrame] = []
    queries: dict[str, SimulatedQuery] = {}

    def tabulate_calls(
        tidy: pd.DataFrame,
        query: SimulatedQuery,
        level_specs: Sequence[LevelSpec],
        seed: int,
    ) -> pd.DataFrame:
        frame = level_cells(tidy, query, test, level_specs, seed=seed)
        for rule in cells_rules:
            frame = rule(frame)
        return frame

    for recipe in recipes:
        step = time.monotonic()
        query = thin_and_contaminate(test, depths, recipe)
        queries[recipe.name] = query
        timings[f"simulate_{recipe.name}"] = round(time.monotonic() - step, 3)
        logger.info(
            "resolvability %s: simulated %d cells from %d test cells in %.1f s "
            "(per depth: %s)",
            recipe.name,
            len(query.obs),
            len(test.obs),
            timings[f"simulate_{recipe.name}"],
            ", ".join(f"{d}: {n}" for d, n in sorted(query.n_by_depth.items())),
        )
        step = time.monotonic()
        tidy = map_fn(query, recipe.name, 0)
        timings[f"map_{recipe.name}"] = round(time.monotonic() - step, 3)
        logger.info(
            "resolvability %s: mapped in %.1f s",
            recipe.name,
            timings[f"map_{recipe.name}"],
        )
        frames.append(tabulate_calls(tidy, query, specs, 0))
    decision_recipe = recipes[0]
    fine_levels = [spec.meta.level for spec in specs if spec.meta.role == "fine"]
    stability: dict[str, float] = {}
    # Decide on the cells as the bundle stores them (float32 bp), so the
    # summary equals what RESOLVE re-derives from the stored table.
    cells = as_stored(pd.concat(frames, ignore_index=True))
    levels = [spec.meta for spec in specs]
    step = time.monotonic()
    decisions = decide(cells, levels, depths, settings, recipe=decision_recipe.name)
    timings["decide"] = round(time.monotonic() - step, 3)
    for regime in REGIMES:
        frame = decisions[decisions["regime"] == regime]
        logger.info(
            "resolvability %s decisions in %.1f s: %s",
            regime,
            timings["decide"],
            "; ".join(
                f"{level} {int((rows['status'] == STATUS_EMITTED).sum())} emitted / "
                f"{int((rows['status'] != STATUS_EMITTED).sum())} not_resolvable"
                for level, rows in frame.groupby("level", sort=False)
            ),
        )
    if fine_seed_check and fine_levels:
        step = time.monotonic()
        tidy = map_fn(queries[decision_recipe.name], f"{decision_recipe.name}_seed1", 1)
        second = tabulate_calls(
            tidy,
            queries[decision_recipe.name],
            [spec for spec in specs if spec.meta.role == "fine"],
            1,
        )
        timings["map_seed1"] = round(time.monotonic() - step, 3)
        second = as_stored(second)
        first = cells[(cells["recipe"] == decision_recipe.name) & (cells["seed"] == 0)]
        for level in fine_levels:
            stability[level] = seed_stability(first, second, decisions, level)
        cells = as_stored(pd.concat([cells, second], ignore_index=True))
    floors = combined_floors(
        decisions, levels, settings, floor_table=floor_table, species=species
    )
    trust = trust_constraint(decisions, levels, settings)
    logger.info(
        "resolvability trust constraint: %s%s",
        trust.state or "none",
        f" ({'; '.join(trust.reasons)})" if trust.reasons else "",
    )
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


def would_raise_bins(
    decisions: pd.DataFrame, settings: RuleSettings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the validated bins whose local rule would raise the default.

    Only a bin's own test counts (not ``pooled``). A bin whose fit half holds
    fewer than ``min_cells_per_bin`` calls has no isotonic fit, so its ``t*``
    is unknown: it is listed as not evaluable, never as a raise (M3b review
    2: most earlier "would raise" bins were such bins).

    Args:
        decisions: ``decide`` output.
        settings: Rule settings (``min_cells_per_bin``).

    Returns:
        ``(raised, not_evaluable)``: the evaluable bins with ``would_raise``,
        and the bins with some fit-half calls but no fit.
    """
    own = decisions[
        (decisions["regime"] == "validated")
        & ~_bool_column(decisions["pooled"])
        & ~_bool_column(decisions["extrapolated"])
    ]
    n_fit = own["n_fit"].fillna(0)
    evaluable = _bool_column(own["would_raise_evaluable"]) & (
        n_fit >= settings.min_cells_per_bin
    )
    raised = own[evaluable & _bool_column(own["would_raise"])]
    not_evaluable = own[~evaluable & (n_fit > 0)]
    return raised, not_evaluable


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
                "pooled": bool(record["pooled"]),
                "reason": record["reason"],
            }
        emission[regime] = per_level
    d_max: dict[str, dict[str, int | None]] = {}
    for record in decisions[decisions["regime"] == "trust"].to_dict("records"):
        value = _optional_float(record["d_max"])
        d_max.setdefault(str(record["level"]), {})[str(record["class"])] = (
            None if value is None else int(value)
        )
    raised, unevaluable = would_raise_bins(decisions, settings)
    floors_json: dict[str, Any] = {}
    for record in floors.to_dict("records"):
        simulated = _optional_float(record["simulated_floor"])
        floor = _optional_float(record["floor"])
        entry: dict[str, Any] = {
            "known": int(record["known_floor"]),
            "simulated": None if simulated is None else int(simulated),
            "floor": None if floor is None else int(floor),
            "source": record["floor_source"],
        }
        packaged = record.get("packaged")
        if isinstance(packaged, list) and packaged:
            entry["packaged"] = [dict(item) for item in packaged]
        floors_json.setdefault(str(record["regime"]), {}).setdefault(
            str(record["level"]), {}
        )[str(record["class"])] = entry
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
        "pooled_sets": {regime: pooled_sets(decisions, regime) for regime in REGIMES},
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
        "validated_thresholds_not_evaluable": [
            {
                "level": str(record["level"]),
                "class": str(record["class"]),
                "depth": int(record["depth"]),
                "n_fit": int(record["n_fit"]),
                "status": str(record["status"]),
            }
            for record in unevaluable.to_dict("records")
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

    @property
    def version(self) -> int | None:
        """The bundle's resolvability version (``None`` if unrecorded)."""
        value = self.summary.get("resolvability_version")
        return int(value) if value is not None else None

    def decisions(
        self,
        *,
        composition: Mapping[str, float] | DatasetComposition | None = None,
        settings: RuleSettings | None = None,
    ) -> pd.DataFrame:
        """Return decisions, reweighted to a dataset's composition when given.

        A version-7 bundle re-derives its ensemble decisions
        (``ensemble_decisions``).

        Args:
            composition: Dataset share per truth type (human supercluster,
                mouse subclass), per depth bin (``DatasetComposition``, what
                RESOLVE passes) or one for every bin; ``None`` for PREP's
                unweighted decisions.
            settings: Rule settings (default: the bundle's).

        Returns:
            ``decide`` output (version 7: ``ensemble_decide`` decisions).
        """
        if self.version == RESOLVABILITY_VERSION_V7:
            return self.ensemble_decisions(
                composition=composition, settings=settings
            ).decisions
        rule = settings or self.settings
        if composition is None:
            return decide(
                self.cells,
                self.levels,
                self.depth_grid,
                rule,
                recipe=str(self.summary["decision_recipe"]),
            )
        class_of = leaf_class_map(self.cells)
        weights = composition_weights(
            self.cells,
            composition,
            class_of=class_of,
            min_type_cells=rule.weight_min_type_cells,
            trim_factor=rule.bin_trim_factor,
        )
        return decide(
            self.cells,
            self.levels,
            self.depth_grid,
            rule,
            weights=weights,
            pool_weights=pooled_composition_weights(
                composition,
                class_of=class_of,
                min_type_cells=rule.weight_min_type_cells,
                trim_factor=rule.bin_trim_factor,
            ),
            recipe=str(self.summary["decision_recipe"]),
        )

    def ensemble_decisions(
        self,
        *,
        composition: Mapping[str, float] | DatasetComposition | None = None,
        settings: RuleSettings | None = None,
        ensemble: EnsembleSettings | None = None,
    ) -> EnsembleDecisions:
        """Re-derive a version-7 bundle's ensemble decisions (§8.3 v7.7-v7.9).

        With a composition each member's cells are reweighted separately
        (``composition_weights`` per member; pooled deep sets per member)
        and the ensemble re-run with the weights, as RESOLVE will (M4
        follow-up; plan §12 M3c).

        Args:
            composition: Dataset composition, or ``None`` (PREP's decisions).
            settings: Rule settings (default: the bundle's).
            ensemble: Ensemble settings (default: the bundle's).

        Returns:
            The ensemble decisions.

        Raises:
            ResolvabilityError: For a bundle that is not version 7.
        """
        if self.version != RESOLVABILITY_VERSION_V7:
            raise ResolvabilityError(
                f"ensemble decisions need a version-7 bundle (this is {self.version})"
            )
        rule = settings or self.settings
        rules = ensemble or EnsembleSettings.from_json(
            self.summary.get("ensemble_settings")
        )
        members = [str(name) for name in self.summary.get("emission_members") or []]
        reported = [
            str(item["member"])
            for item in self.summary.get("members") or []
            if item.get("role") != "emission"
        ]
        weights: np.ndarray | None = None
        pool_weights: PoolWeights | None = None
        if composition is not None:
            class_of = leaf_class_map(self.cells)
            weights = np.zeros(len(self.cells), dtype=np.float64)
            names = self.cells[MEMBER_COLUMN].astype(str).to_numpy()
            for name in dict.fromkeys(names):
                mask = names == name
                weights[mask] = composition_weights(
                    self.cells[mask],
                    composition,
                    class_of=class_of,
                    min_type_cells=rule.weight_min_type_cells,
                    trim_factor=rule.bin_trim_factor,
                )
            pool_weights = pooled_composition_weights(
                composition,
                class_of=class_of,
                min_type_cells=rule.weight_min_type_cells,
                trim_factor=rule.bin_trim_factor,
            )
        return ensemble_decide(
            self.cells,
            self.levels,
            self.depth_grid,
            rule,
            rules,
            members=members,
            reported=reported,
            weights=weights,
            pool_weights=pool_weights,
            neuronal=dict(self.summary.get("neuronal_classes") or {}),
        )


def load_resolvability(
    directory: Path | str, *, allow_version_7: bool = False
) -> ResolvabilityTables | None:
    """Read a bundle's resolvability outputs (``None`` when it has none).

    A version-7 bundle (M3c) is read only by a caller that declares support
    (``allow_version_7``): its decisions come from the ensemble, its cells
    carry several members of one recipe, and it adds the monotone fill and
    the non-neuronal high-depth marker, so a consumer written for version 6
    (M4's RESOLVE until its follow-up, plan §12 M3c) refuses it loudly
    instead of misreading it.

    Args:
        directory: Bundle directory.
        allow_version_7: The caller handles version-7 bundles.

    Returns:
        The tables, or ``None``.

    Raises:
        ResolvabilityError: For a version-7 bundle without
            ``allow_version_7``, or an unknown later version.
    """
    root = Path(directory)
    summary_path = root / RESOLVABILITY_SUMMARY_FILE
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    version = summary.get("resolvability_version")
    if version is not None and int(version) > RESOLVABILITY_VERSION_V7:
        raise ResolvabilityError(
            f"{root}: resolvability version {version} is newer than this code "
            f"({RESOLVABILITY_VERSION_V7})"
        )
    if (
        version is not None
        and int(version) == RESOLVABILITY_VERSION_V7
        and not (allow_version_7)
    ):
        raise ResolvabilityError(
            f"{root}: a resolvability version-7 bundle (ensemble decisions, "
            "monotone fill; plan §8.3 v7) needs a consumer that declares "
            "version-7 support (load_resolvability(..., allow_version_7=True)); "
            "RESOLVE accepts it after its M3c follow-up (plan §12 M3c)"
        )
    cells = restore_labels(pd.read_parquet(root / RESOLVABILITY_CELLS_FILE))
    if MEMBER_COLUMN in cells.columns:
        cells[MEMBER_COLUMN] = cells[MEMBER_COLUMN].astype(str)
    if MEMBER_ROLE_COLUMN in cells.columns:
        cells[MEMBER_ROLE_COLUMN] = cells[MEMBER_ROLE_COLUMN].astype(str)
    fields = RuleSettings.__dataclass_fields__
    stored = {key: value for key, value in summary["settings"].items() if key in fields}
    # Bundles of resolvability versions 2-3 trimmed composition weights per
    # depth bin and did not record the unit.
    stored.setdefault("weight_trim_scope", LEGACY_WEIGHT_TRIM_SCOPE)
    settings = RuleSettings(**stored)
    return ResolvabilityTables(
        summary=summary,
        cells=cells,
        levels=[LevelMeta.from_json(item) for item in summary["levels"]],
        settings=settings,
    )


# --------------------------------------------------------------------------
# Gate-P machinery (M13 scores NP3-NP7 with these; §14 evaluation rules)


def replicate_rows(
    cells: pd.DataFrame,
    *,
    recipe: str | None,
    seed: int | None,
    member: str | None = None,
) -> pd.DataFrame:
    """Return one replicate's rows: a cells table restricted to a recipe and seed.

    A bundle's cells table holds several recipes (``R1_contam_HO`` and
    ``clean``) and, with the fine-level check, two seeds of the same
    simulated cells; pooling or counting them together would mix recipes
    and count test cells twice (M3b review 2). The rows are filtered on
    ``recipe`` and ``seed`` when given, and each (level, cell, depth) must
    then occur once (pooled donors with distinct cell ids are fine).

    Args:
        cells: A cells table.
        recipe: The recipe to keep (``None``: all).
        seed: The mapping seed to keep (``None``: all).
        member: The version-7 ensemble member to keep (``None``: all); the
            R1 members of an ensemble share their recipe and mapping seed.

    Returns:
        The rows.

    Raises:
        ResolvabilityError: If a (level, cell_id, depth) occurs more than
            once after the filters (pass ``recipe`` / ``seed`` / ``member``).
    """
    frame = cells
    if member is not None:
        if MEMBER_COLUMN not in frame.columns:
            raise ResolvabilityError(f"member {member!r}: the cells have no members")
        frame = frame[(frame[MEMBER_COLUMN].astype(str) == member).to_numpy()]
    if recipe is not None:
        frame = frame[(frame["recipe"].astype(str) == recipe).to_numpy()]
    if seed is not None:
        frame = frame[(frame["seed"].to_numpy() == seed)]
    key = pd.DataFrame(
        {
            "level": frame["level"].astype(str).to_numpy(),
            "cell_id": frame["cell_id"].astype(str).to_numpy(),
            "depth": frame["depth"].to_numpy(np.int64),
        }
    )
    duplicated = key.duplicated()
    if bool(duplicated.any()):
        mixed = sorted(
            {
                f"{recipe_value}/seed {seed_value}"
                for recipe_value, seed_value in zip(
                    frame["recipe"].astype(str), frame["seed"], strict=True
                )
            }
        )
        raise ResolvabilityError(
            f"{int(duplicated.sum())} (level, cell, depth) rows occur more than "
            f"once ({', '.join(mixed)}): pass recipe= and seed= to select one "
            "replicate"
        )
    return frame


def frozen_threshold_eval(
    cells: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    regime: Regime = "provisional",
    recipe: str | None = None,
    seed: int | None = None,
    member: str | None = None,
) -> pd.DataFrame:
    """Apply frozen thresholds to replicate cells, per (level, class, depth).

    Args:
        cells: Replicate cells table (another donor, draw, seed or recipe).
        decisions: Frozen ``decide`` output of the base run (version 7: the
            ensemble's).
        regime: Regime whose thresholds are frozen.
        recipe: Restrict the replicate rows to one recipe.
        seed: Restrict the replicate rows to one mapping seed.
        member: Restrict the replicate rows to one ensemble member.

    Returns:
        ``level``, ``class``, ``depth``, ``threshold``, ``n_called``,
        ``n_confident``, ``precision``, ``wilson_lb``, ``coverage``.

    Raises:
        ResolvabilityError: If the rows mix replicates (``replicate_rows``).
    """
    lookup = emission_lookup(decisions, regime)
    frame = replicate_rows(cells, recipe=recipe, seed=seed, member=member)
    frame = frame[frame["parent"].notna()]
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
    recipe: str | None = DECISION_RECIPE,
    seed: int | None = 0,
    member: str | None = None,
) -> dict[tuple[str, str], list[GatePTestedSet] | None]:
    """Return the gate-P tested sets per (level, class) (``None``: not evaluable).

    The pooling ``decide`` applies with ``min_confident_n`` (one rule for
    resolvability and gate P, §14; user decision 2026-09-27): a bin holding
    ``min_confident_n`` confident calls is tested on its own; when the
    deepest bin does not, bins are pooled from the deep end into a ">= d"
    set (each test cell counted once, at its deepest bin, ``deepest_rows``)
    until it holds ``min_confident_n`` confident calls; its shallowest bin
    is ``D_P``, whose own test (when it holds enough calls) is a tested set
    too. Confidence uses the frozen threshold of each call's bin.

    Args:
        cells: Pooled held-out cells (frozen-threshold replicates).
        decisions: Frozen decisions.
        min_confident_n: ``gate_p_min_confident_n`` (200).
        regime: Regime of the thresholds.
        recipe: The recipe of the tested rows (``None``: the table must hold
            one; ``replicate_rows``).
        seed: The mapping seed of the tested rows (``None``: likewise).
        member: The version-7 ensemble member of the tested rows (gate P
            scores NP3-NP7 in every emission member; ``gate_p_member_sets``).

    Returns:
        Tested sets per (level, class), the pooled set first, then the bins
        tested on their own, deepest first; ``None`` when even the set of
        every bin has fewer than ``min_confident_n`` confident calls.

    Raises:
        ResolvabilityError: If the rows mix replicates (``replicate_rows``).
    """
    lookup = emission_lookup(decisions, regime)
    frame = replicate_rows(cells, recipe=recipe, seed=seed, member=member).copy()
    thresholds = np.array(
        [
            _frozen_threshold(lookup.get((str(level), str(cls), int(depth))))
            if cls is not None and not pd.isna(cls)
            else math.nan
            for level, cls, depth in zip(
                frame["level"].astype(str),
                frame["parent"].astype(object),
                frame["depth"],
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    bp = frame["bp"].to_numpy(np.float64)
    frame["_confident"] = np.isfinite(thresholds) & (
        np.nan_to_num(bp, nan=-1.0) >= thresholds - 1e-9
    )
    result: dict[tuple[str, str], list[GatePTestedSet] | None] = {}
    for level, level_rows in frame.groupby(frame["level"].astype(str), observed=True):
        deepest = deepest_rows(level_rows)
        called = level_rows[level_rows["parent"].notna()]
        for cls, group in called.groupby(called["parent"].astype(str), observed=True):
            key = (str(level), str(cls))
            confident = group[group["_confident"].to_numpy(bool)]
            own = confident.groupby("depth").size()
            grid = sorted({int(depth) for depth in group["depth"]}, reverse=True)
            class_deepest = deepest[
                (deepest["parent"].astype(object) == cls).to_numpy()
                & deepest["_confident"].to_numpy(bool)
            ]
            sets: list[GatePTestedSet] = []
            pool_min: int | None = None
            if int(own.get(grid[0], 0)) < min_confident_n:
                for depth in grid:
                    pooled = class_deepest[class_deepest["depth"] >= depth]
                    if len(pooled) >= min_confident_n:
                        pool_min = depth
                        members = sorted(d for d in grid if d >= depth)
                        sets.append(_tested_set(key, pooled, members))
                        break
                if pool_min is None:
                    result[key] = None
                    continue
            for depth in grid:
                if int(own.get(depth, 0)) >= min_confident_n:
                    subset = confident[confident["depth"] == depth]
                    sets.append(_tested_set(key, subset, [depth]))
            result[key] = sets or None
    return result


def _frozen_threshold(entry: tuple[str, float | None, bool] | None) -> float:
    """The frozen threshold of an emitted bin (``nan``: nothing is confident)."""
    if entry is None or entry[0] != STATUS_EMITTED or entry[1] is None:
        return math.nan
    return float(entry[1])


def _tested_set(
    key: tuple[str, str], rows: pd.DataFrame, depths: Sequence[int]
) -> GatePTestedSet:
    """A tested set of confident rows (unweighted precision and Wilson bound)."""
    precision = float(rows["correct"].to_numpy(bool).mean())
    return GatePTestedSet(
        level=key[0],
        cls=key[1],
        depths=tuple(int(depth) for depth in depths),
        pooled=len(depths) > 1,
        n_confident=int(len(rows)),
        precision=precision,
        wilson_lb=wilson_lower_bound(precision, len(rows)),
    )


def gate_p_member_sets(
    cells: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    members: Sequence[str],
    min_confident_n: int = 200,
    regime: Regime = "provisional",
    seed: int | None = 0,
) -> dict[str, dict[tuple[str, str], list[GatePTestedSet] | None]]:
    """Return each emission member's gate-P tested sets (version 7, §14).

    Gate P freezes the ensemble's thresholds and emission (``decisions``: the
    ensemble decisions of the default donor or draw, monotone-filled bins
    included) and scores NP3-NP7 separately in every emission member: each
    member's own simulation of the held-out calls at the frozen thresholds.

    Args:
        cells: Pooled held-out version-7 cells (a ``member`` column).
        decisions: Frozen ensemble decisions.
        members: The emission members.
        min_confident_n: ``gate_p_min_confident_n`` (200).
        regime: Regime of the thresholds.
        seed: The mapping seed of the tested rows.

    Returns:
        Per member, ``gate_p_tested_sets`` of its rows.
    """
    return {
        member: gate_p_tested_sets(
            cells,
            decisions,
            min_confident_n=min_confident_n,
            regime=regime,
            recipe=None,
            seed=seed,
            member=member,
        )
        for member in members
    }


GATE_P_PASSED: Final = "passed"
GATE_P_FAILED: Final = "failed"
GATE_P_NOT_EVALUABLE: Final = "not_evaluable"


def every_member_verdict(
    verdicts: Mapping[str, Mapping[tuple[str, str], bool | None]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Combine per-member gate-P verdicts: a (level, class) passes only in all.

    Version-7 families (plan §14 "Version-7 families"): a (level, class) is
    validated only if NP3-NP7 pass in every emission member. A member that
    fails it fails it; a member where it is not evaluable (``None`` or
    missing) leaves it not evaluable.

    Args:
        verdicts: Per member, per (level, class): ``True`` (passes), ``False``
            (fails) or ``None`` (not evaluable).

    Returns:
        Per (level, class): ``status`` (``passed``, ``failed`` or
        ``not_evaluable``), ``failed_members`` and ``unevaluable_members``.
    """
    keys = sorted({key for values in verdicts.values() for key in values})
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key in keys:
        failed = sorted(
            member for member, values in verdicts.items() if values.get(key) is False
        )
        unevaluable = sorted(
            member for member, values in verdicts.items() if values.get(key) is None
        )
        if failed:
            status = GATE_P_FAILED
        elif unevaluable or not verdicts:
            status = GATE_P_NOT_EVALUABLE
        else:
            status = GATE_P_PASSED
        result[key] = {
            "status": status,
            "failed_members": failed,
            "unevaluable_members": unevaluable,
        }
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


# --------------------------------------------------------------------------
# Resolvability version 7: families, members and simulation (M3c; §8.3 v7)
#
# Version 7 is additive: nothing above changes, so version 6 stays
# byte-identical for the families of ``validated_panels.csv`` (and the pins of
# ``resolvability_v6_pins.csv``), whose decisions are pre-registered for gates
# H and M (the M3c scope decision; pre-registration §14 (i)). Version 7 (every
# other family) simulates each ensemble member with exact-total thinning, on
# TOTAL counts: a grid value D is the simulated cell's total, host D / (1 + s)
# plus spill s D / (1 + s), because real cells are binned by their totals.

RESOLVABILITY_VERSION_V6: Final = 6
RESOLVABILITY_VERSION_V7: Final = 7
RESOLVABILITY_VERSIONS: Final[tuple[int, ...]] = (6, 7)
V6_PINS_FILE: Final = "resolvability_v6_pins.csv"
EFFICIENCY_SOURCES: Final[tuple[str, ...]] = ("lognormal", "measured", "xtissue_stress")
# Exact-total thinning (v7.2): at most 30 fixed-point steps, stopping per row
# at |ratio - 1| < 1e-6.
THIN_MAX_ITER: Final = 30
THIN_TOLERANCE: Final = 1e-6
# Panels above 1,000 genes (v7.2): 13 values, neighbours <= 1.67x apart above
# 100 counts, 3,000 reaching the real 5K q95 (3,330).
V7_LARGE_PANEL_GRID: Final[tuple[int, ...]] = (
    10,
    20,
    50,
    100,
    150,
    250,
    350,
    500,
    700,
    1000,
    1400,
    2000,
    3000,
)
V7_LARGE_PANEL_GENES: Final = 1000
# Ensemble members (v7.3 as amended on 2026-09-29: orchestrator decision D1 (a),
# pending the user's confirmation; pre-registration §15.3, fixed there and only
# tightenable). Eight emission members per version-7 family: R1 x 6 + R3 x 2
# where the species x chemistry has a measured factor table, else R1 x 8.
# R1@0 is the pre-registered realisation (gate P's NP3 base); seeds 1-5 of R1
# and 0-1 of R3 were drawn for the ensembles A and B of the failed stage-D test
# (iii) and are not re-used.
V7_R1_SEEDS_WITH_TABLE: Final[tuple[int, ...]] = (0, 6, 7, 8, 9, 10)
V7_R1_SEEDS_WITHOUT_TABLE: Final[tuple[int, ...]] = (0, 6, 7, 8, 9, 10, 11, 12)
V7_R3_SEEDS: Final[tuple[int, ...]] = (2, 3)
V7_EMISSION_MEMBERS: Final = 8
# The comparator of the amended re-test of pre-registration §14 (iii) (§15.4):
# disjoint from every production and stage-D member; never production.
V7_COMPARATOR_R1_SEEDS_WITH_TABLE: Final[tuple[int, ...]] = (20, 21, 22, 23, 24, 25)
V7_COMPARATOR_R1_SEEDS_WITHOUT_TABLE: Final[tuple[int, ...]] = (
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
)
V7_COMPARATOR_R3_SEEDS: Final[tuple[int, ...]] = (20, 21)
# The ensembles of stages A-D (the bundles built before the amendment keep them).
V7_STAGE_D_R1_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
V7_STAGE_D_R3_SEEDS: Final[tuple[int, ...]] = (0,)
V7_STRESS_SEED: Final = 0
MemberRole = Literal["emission", "reported", "stress"]
MEMBER_ROLES: Final[tuple[str, ...]] = ("emission", "reported", "stress")


def load_v6_pins(path: Path | str | None = None) -> pd.DataFrame:
    """Return the families pinned to resolvability version 6 (OD-E20).

    Args:
        path: The pins CSV (default: the packaged
            ``assets/annotation/resolvability_v6_pins.csv``).

    Returns:
        ``family_id, panel_hash, species, reason, date`` rows.

    Raises:
        ResolvabilityError: If a column is missing or a row lacks a reason.
    """
    from merxen.annotation.vocab import ASSET_DIR

    location = Path(path) if path is not None else ASSET_DIR / V6_PINS_FILE
    table = pd.read_csv(location, dtype=str, keep_default_na=False)
    required = ("family_id", "panel_hash", "species", "reason", "date")
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ResolvabilityError(f"{location.name}: columns {missing} are missing")
    if (table["reason"].str.strip() == "").any():
        raise ResolvabilityError(f"{location.name}: every pin needs its reason")
    return table


def v6_family_ids(
    validated: Any | None = None, pins: pd.DataFrame | None = None
) -> set[str]:
    """Return the families that keep resolvability version 6 (v7.1).

    The real-data families of ``validated_panels.csv`` (the families whose
    decisions are pre-registered for gates H and M: human_set_a with its set
    c, mouse_ag7, mouse_vzg2) and the families of ``resolvability_v6_pins
    .csv``. A family validated later by simulation (gate P, M13) was
    validated on version 7 and stays there.

    Args:
        validated: A ``diagnostics.ValidatedPanelTable`` (default: the
            packaged tables).
        pins: ``load_v6_pins`` output (default: the packaged pins).

    Returns:
        Family ids.
    """
    from merxen.annotation.diagnostics import load_validated_panels

    table = validated if validated is not None else load_validated_panels()
    families = {
        str(record.family_id)
        for record in table.records
        if str(getattr(record, "validation_basis", "real_data")) == "real_data"
    }
    pinned = pins if pins is not None else load_v6_pins()
    return families | {str(value) for value in pinned["family_id"]}


def resolvability_version_for(
    family_id: str | None,
    panel_hash: str | None = None,
    *,
    validated: Any | None = None,
    pins: pd.DataFrame | None = None,
) -> int:
    """Return the resolvability version of a panel family (plan §8.3 v7.1).

    Args:
        family_id: The panel's family after trust inheritance
            (``AnnotationPanel.panel_family.family_id``); a listed or
            inherited panel carries its validated family's id.
        panel_hash: The panel hash (a pin may name it).
        validated: A ``ValidatedPanelTable`` (default: packaged).
        pins: ``load_v6_pins`` output (default: packaged).

    Returns:
        6 for the real-data families of ``validated_panels.csv`` (by family
        id, or by a listed panel hash when the family is not known) and the
        pinned families (or pinned panel hashes), else 7.
    """
    from merxen.annotation.diagnostics import load_validated_panels

    table = validated if validated is not None else load_validated_panels()
    pinned = pins if pins is not None else load_v6_pins()
    if family_id is not None and family_id in v6_family_ids(table, pinned):
        return RESOLVABILITY_VERSION_V6
    if panel_hash is not None:
        if panel_hash in set(pinned["panel_hash"]):
            return RESOLVABILITY_VERSION_V6
        record = table.record_for_hash(panel_hash)
        if (
            record is not None
            and str(getattr(record, "validation_basis", "real_data")) == "real_data"
        ):
            return RESOLVABILITY_VERSION_V6
    return RESOLVABILITY_VERSION_V7


def v7_depth_grid(
    species: str, n_panel_genes: int | None, explicit: Sequence[int] | None = None
) -> list[int]:
    """Return a version-7 family's depth grid (v7.2).

    Args:
        species: ``human`` or ``mouse``.
        n_panel_genes: Declared panel size.
        explicit: An explicit grid (``AnnotationReferenceSpec.depth_grid``).

    Returns:
        The explicit grid; above 1,000 genes the 13-value grid; otherwise
        the version-6 grid of the species and size.
    """
    from merxen.annotation.config import default_depth_grid

    if explicit is not None:
        return [int(value) for value in explicit]
    if n_panel_genes is not None and n_panel_genes > V7_LARGE_PANEL_GENES:
        return list(V7_LARGE_PANEL_GRID)
    return default_depth_grid(species, n_panel_genes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class EnsembleMember:
    """One member of a version-7 draw ensemble (v7.3): a recipe and a seed.

    Attributes:
        recipe: The simulation recipe (its ``seed`` is the member seed).
        role: ``emission`` (decides emission), ``reported`` (``clean``: the
            upper bound) or ``stress`` (reported only; gate P's NP6).
    """

    recipe: SimulationRecipe
    role: MemberRole

    @property
    def name(self) -> str:
        """``<recipe>@<seed>``."""
        return self.recipe.member

    def to_json(self) -> dict[str, Any]:
        """Return the member as JSON-native values (hashed for version 7)."""
        return {"member": self.name, "role": self.role, "recipe": self.recipe.to_json()}


def member_recipe(
    name: str,
    seed: int,
    config: AnnotationResolvabilityConfig,
    *,
    table: Any | None = None,
    table_rule: str = "restricted",
    residual_sd_log2: float | None = None,
) -> SimulationRecipe:
    """Return the recipe of one version-7 member.

    Args:
        name: ``R1_contam_HO``, ``clean``, ``R3_measured_HO`` or
            ``R1_xtissue_lung_stress``.
        seed: The member seed (efficiency and per-cell keys).
        config: Resolvability settings (sigma, spill fraction).
        table: The ``sim_inputs.SimInputAsset`` of an R3 (``member``) or
            stress (``stress``) recipe.
        table_rule: R3 table rule.
        residual_sd_log2: R3 residual SD (default 0.20 log2).

    Returns:
        The recipe.

    Raises:
        ResolvabilityError: For an unknown recipe or a missing table.
    """
    from merxen.annotation import sim_inputs as si

    if name not in RECIPE_VERSIONS:
        raise ResolvabilityError(f"unknown simulation recipe {name!r}")
    version = RECIPE_VERSIONS[name]
    if name == CLEAN_RECIPE:
        return SimulationRecipe(CLEAN_RECIPE, version, None, 0.0, int(seed))
    if name == DECISION_RECIPE:
        return SimulationRecipe(
            DECISION_RECIPE,
            version,
            config.gene_efficiency_sigma,
            config.spill_fraction,
            int(seed),
        )
    if table is None:
        raise ResolvabilityError(f"recipe {name} needs its simulation-input table")
    if name == R3_RECIPE:
        if table.role != "member":
            raise ResolvabilityError(f"{name} needs a member table, not {table.role}")
        if table_rule not in si.TABLE_RULES:
            raise ResolvabilityError(f"unknown R3 table rule {table_rule!r}")
        return SimulationRecipe(
            R3_RECIPE,
            version,
            None,
            config.spill_fraction,
            int(seed),
            efficiency_source="measured",
            efficiency_table=table.asset_id,
            efficiency_table_sha256=table.sha256,
            table_rule=table_rule,
            residual_sd_log2=si.R3_RESIDUAL_SD_LOG2
            if residual_sd_log2 is None
            else float(residual_sd_log2),
        )
    if table.role != "stress":
        raise ResolvabilityError(f"{name} needs a stress table, not {table.role}")
    return SimulationRecipe(
        LUNG_STRESS_RECIPE,
        version,
        config.gene_efficiency_sigma,
        config.spill_fraction,
        int(seed),
        efficiency_source="xtissue_stress",
        efficiency_table=table.asset_id,
        efficiency_table_sha256=table.sha256,
    )


def default_member_seeds(
    has_table: bool, *, comparator: bool = False
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the ``(R1 seeds, R3 seeds)`` of a version-7 ensemble (v7.3).

    As amended on 2026-09-29 (pre-registration §15.3): eight emission
    members, R1 x 6 + R3 x 2 where the family's species x chemistry has a
    measured factor table, else R1 x 8. ``comparator`` returns the
    comparator of the amended re-test of §14 (iii) instead, disjoint from
    every production and stage-D member (never production).

    Args:
        has_table: Whether a measured ``member`` table exists (an R3 member).
        comparator: Return the re-test's comparator seeds.

    Returns:
        R1 seeds and R3 seeds (the R3 seeds are empty without a table).
    """
    if comparator:
        r1 = (
            V7_COMPARATOR_R1_SEEDS_WITH_TABLE
            if has_table
            else V7_COMPARATOR_R1_SEEDS_WITHOUT_TABLE
        )
        return r1, V7_COMPARATOR_R3_SEEDS if has_table else ()
    r1 = V7_R1_SEEDS_WITH_TABLE if has_table else V7_R1_SEEDS_WITHOUT_TABLE
    return r1, V7_R3_SEEDS if has_table else ()


def ensemble_members(
    config: AnnotationResolvabilityConfig,
    *,
    species: str,
    chemistry: str,
    member_table: Any | None = None,
    stress_table: Any | None = None,
    r1_seeds: Sequence[int] | None = None,
    r3_seeds: Sequence[int] | None = None,
    table_rule: str = "restricted",
    residual_sd_log2: float | None = None,
) -> list[EnsembleMember]:
    """Return a version-7 family's members (plan §8.3 v7.3 table, as amended).

    Emission: eight members (amendment of 2026-09-29, pre-registration
    §15.3) -- ``R1_contam_HO@0``, ``@6``-``@10`` plus ``R3_measured_HO@2``,
    ``@3`` when the family's species x chemistry has a measured ``member``
    table (Xenium Prime 5K mouse), else ``R1_contam_HO@0``, ``@6``-``@12``;
    reported: ``clean@0``; stress (human Prime families with the lung ratio
    table, reported only and never an emission member):
    ``R1_xtissue_lung_stress@0``. R3 is one member, never the base (user
    decision 3); ``R1@0`` is the pre-registered realisation. The version-6
    families' diagnostic uses the same rule (no table: R1 x 8).

    Args:
        config: Resolvability settings.
        species: The family's species.
        chemistry: ``sim_inputs.resolve_chemistry`` result.
        member_table: The family's ``member`` asset (``None``: no R3).
        stress_table: The human lung ``stress`` asset (``None``: no stress).
        r1_seeds: R1 member seeds (``None``: ``default_member_seeds``).
        r3_seeds: R3 member seeds (``None``: ``default_member_seeds``);
            ignored without a table.
        table_rule: R3 table rule.
        residual_sd_log2: R3 residual SD (``None``: 0.20 log2).

    Returns:
        Members: emission first (R1 seeds, then R3 seeds), then reported,
        then stress.

    Raises:
        ResolvabilityError: For a member table of another species or
            repeated seeds.
    """
    default_r1, default_r3 = default_member_seeds(member_table is not None)
    r1 = tuple(int(seed) for seed in (default_r1 if r1_seeds is None else r1_seeds))
    r3 = tuple(int(seed) for seed in (default_r3 if r3_seeds is None else r3_seeds))
    for label, seeds in (("R1", r1), ("R3", r3)):
        if len(set(seeds)) != len(seeds):
            raise ResolvabilityError(f"{label} member seeds repeat: {list(seeds)}")
    members = [
        EnsembleMember(member_recipe(DECISION_RECIPE, seed, config), "emission")
        for seed in r1
    ]
    if member_table is not None:
        if member_table.species != species:
            raise ResolvabilityError(
                f"{member_table.asset_id} is a {member_table.species} table; "
                f"factors never cross species ({species})"
            )
        members.extend(
            EnsembleMember(
                member_recipe(
                    R3_RECIPE,
                    seed,
                    config,
                    table=member_table,
                    table_rule=table_rule,
                    residual_sd_log2=residual_sd_log2,
                ),
                "emission",
            )
            for seed in r3
        )
    members.append(EnsembleMember(member_recipe(CLEAN_RECIPE, 0, config), "reported"))
    if (
        stress_table is not None
        and species == "human"
        and chemistry == "xenium_prime"
        and stress_table.species == "human"
    ):
        members.append(
            EnsembleMember(
                member_recipe(
                    LUNG_STRESS_RECIPE, V7_STRESS_SEED, config, table=stress_table
                ),
                "stress",
            )
        )
    return members


def parse_member_name(name: str) -> tuple[str, int]:
    """Return ``(recipe, seed)`` of a member name ``<recipe>@<seed>``.

    Raises:
        ResolvabilityError: If the name has no ``@<seed>`` suffix.
    """
    recipe, separator, seed = str(name).rpartition("@")
    if not separator or not recipe or not seed.lstrip("-").isdigit():
        raise ResolvabilityError(f"{name!r} is not a member name <recipe>@<seed>")
    return recipe, int(seed)


def members_from_names(
    names: Sequence[str],
    config: AnnotationResolvabilityConfig,
    *,
    member_table: Any | None = None,
    stress_table: Any | None = None,
    table_rule: str = "restricted",
    residual_sd_log2: float | None = None,
    role: MemberRole = "emission",
) -> list[EnsembleMember]:
    """Return the members a bundle records by name (e.g. its emission members).

    Profile mode and the diagnostics re-simulate a bundle's own members, so a
    bundle built before the amendment of 2026-09-29 keeps its stage-D
    members.

    Args:
        names: Member names (``resolvability_summary.json``
            ``emission_members``).
        config: Resolvability settings.
        member_table: The ``member`` asset of R3 members.
        stress_table: The ``stress`` asset of the lung stress member.
        table_rule: R3 table rule.
        residual_sd_log2: R3 residual SD.
        role: The members' role.

    Returns:
        The members, in the order given.
    """
    members = []
    for name in names:
        recipe, seed = parse_member_name(name)
        table = (
            member_table
            if recipe == R3_RECIPE
            else stress_table
            if recipe == LUNG_STRESS_RECIPE
            else None
        )
        members.append(
            EnsembleMember(
                member_recipe(
                    recipe,
                    seed,
                    config,
                    table=table,
                    table_rule=table_rule,
                    residual_sd_log2=residual_sd_log2,
                ),
                role,
            )
        )
    return members


def member_efficiency(
    recipe: SimulationRecipe,
    genes: Sequence[str],
    *,
    registry: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return a member's per-gene efficiency on a panel (test-cell columns).

    ``lognormal``: ``gene_efficiency`` (the pre-registered draw of the seed);
    ``measured``: ``sim_inputs.r3_efficiency`` on the recipe's table and
    rule; ``xtissue_stress``: the lognormal draw of the seed times the lung
    ratio (``sim_inputs.xtissue_stress_efficiency``).

    Raises:
        ResolvabilityError: For an unknown source, or a table whose sha256
            differs from the recipe's.
    """
    from merxen.annotation import sim_inputs as si

    names = [str(gene) for gene in genes]
    if recipe.efficiency_source == "lognormal":
        return gene_efficiency(len(names), recipe.gene_efficiency_sigma, recipe.seed)
    if recipe.efficiency_source not in EFFICIENCY_SOURCES:
        raise ResolvabilityError(
            f"unknown efficiency source {recipe.efficiency_source!r}"
        )
    asset = si.get_asset(str(recipe.efficiency_table), registry)
    if recipe.efficiency_table_sha256 not in (None, asset.sha256):
        raise ResolvabilityError(
            f"{recipe.member}: table {asset.asset_id} has sha256 {asset.sha256[:16]}, "
            f"the recipe was built on {str(recipe.efficiency_table_sha256)[:16]}"
        )
    if recipe.efficiency_source == "measured":
        result = si.r3_efficiency(
            names,
            si.efficiency_table(asset),
            rule=str(recipe.table_rule or "restricted"),
            seed=int(recipe.seed),
            asset_id=asset.asset_id,
            residual_sd_log2=float(
                si.R3_RESIDUAL_SD_LOG2
                if recipe.residual_sd_log2 is None
                else recipe.residual_sd_log2
            ),
        )
        return result.efficiency
    base = gene_efficiency(len(names), recipe.gene_efficiency_sigma, recipe.seed)
    efficiency, _measured = si.xtissue_stress_efficiency(
        names, base, si.ratio_table(asset), seed=int(recipe.seed)
    )
    return efficiency


def _exact_probabilities(
    work: sparse.csr_matrix,
    targets: np.ndarray | Sequence[float],
    efficiency: np.ndarray,
    *,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each entry's row and keep probability ``min(1, s_row e_g)``.

    The row scale ``s`` solves ``sum_g x_g min(1, s e_g) = min(target,
    native)`` by fixed-point iteration (``s <- s * goal / expected``), each
    row stopping once ``|ratio - 1| < tolerance`` or after ``max_iter``
    steps; a row whose target reaches its native counts keeps every count.
    """
    n_rows = work.shape[0]
    rows = np.repeat(np.arange(n_rows), np.diff(work.indptr))
    counts = work.data
    gene_eff = np.asarray(efficiency, dtype=np.float64)[work.indices]
    native = np.bincount(rows, weights=counts, minlength=n_rows)
    goal = np.minimum(np.asarray(targets, dtype=np.float64), native)
    weighted = np.bincount(rows, weights=counts * gene_eff, minlength=n_rows)
    scale = goal / np.maximum(weighted, 1e-12)
    keep_all = goal >= native
    active = ~keep_all & (goal > 0)
    for _ in range(int(max_iter)):
        if not active.any():
            break
        expected = np.bincount(
            rows,
            weights=counts * np.minimum(1.0, scale[rows] * gene_eff),
            minlength=n_rows,
        )
        ratio = np.where(expected > 0, goal / np.maximum(expected, 1e-12), 1.0)
        active &= ~(np.abs(ratio - 1.0) < tolerance)
        scale = np.where(active, scale * ratio, scale)
    # The fixed point contracts by the clipped share of the goal per step, so a
    # row whose efficient genes carry most of its target may not converge in
    # max_iter steps; solve its piecewise-linear equation exactly instead.
    for row in np.flatnonzero(active):
        start, stop = int(work.indptr[row]), int(work.indptr[row + 1])
        scale[row] = _exact_row_scale(
            counts[start:stop], gene_eff[start:stop], float(goal[row])
        )
    probability = np.clip(scale[rows] * gene_eff, 0.0, 1.0)
    probability[keep_all[rows]] = 1.0
    return rows, probability


def _exact_row_scale(counts: np.ndarray, efficiency: np.ndarray, goal: float) -> float:
    """Return ``s`` with ``sum_g x_g min(1, s e_g) = goal`` (0 < goal < sum x).

    ``E(s)`` is piecewise linear and increasing: at ``s``, the genes with
    ``s e_g >= 1`` keep every count. The breakpoints ``1 / e_g`` are sorted,
    ``E`` is evaluated at each and the root is interpolated on its segment.
    """
    breakpoints = 1.0 / efficiency
    order = np.argsort(breakpoints, kind="stable")
    x = counts[order]
    e = efficiency[order]
    b = breakpoints[order]
    total_weight = float(np.sum(x * e))
    clipped = np.cumsum(x)
    unclipped_weight = total_weight - np.cumsum(x * e)
    at_breakpoint = clipped + b * np.maximum(unclipped_weight, 0.0)
    k = int(np.searchsorted(at_breakpoint, goal, side="left"))
    if k >= len(x):
        return float(b[-1])
    fixed = float(clipped[k - 1]) if k > 0 else 0.0
    weight = total_weight - (float(np.sum(x[:k] * e[:k])) if k > 0 else 0.0)
    return (goal - fixed) / max(weight, 1e-300)


def _exact_work(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    from scipy import sparse as sp

    work = sp.csr_matrix(matrix, dtype=np.float64, copy=True)
    work.sum_duplicates()
    work.eliminate_zeros()
    return work


def thin_rows_exact(
    matrix: sparse.csr_matrix,
    targets: np.ndarray | Sequence[float],
    efficiency: np.ndarray,
    keys: Sequence[int],
    *,
    max_iter: int = THIN_MAX_ITER,
    tolerance: float = THIN_TOLERANCE,
) -> sparse.csr_matrix:
    """Thin each row binomially to an exact expected total (v7.2).

    The row scale ``s`` solves ``sum_g x_g min(1, s e_g) = min(target,
    native)`` by fixed-point iteration (``s <- s * goal / expected``, at most
    ``max_iter`` steps, each row stopping once ``|ratio - 1| < tolerance``);
    then ``Binomial(x_g, min(1, s e_g))`` from the row's own generator seeded
    by its key, over its entries in gene order. Version 6's ``_thin_rows``
    uses the unclipped scale and falls 3-8% short once efficient genes clip
    at ``p = 1`` (``5k_real/sim/REPORT.txt`` §1). A row whose target reaches
    its native counts keeps every count. A row's result depends on nothing
    but the row, its target and its key; counts are never raised.

    Args:
        matrix: Rows x genes counts.
        targets: Expected total per row.
        efficiency: Per-gene efficiency (the matrix's columns).
        keys: One ``draw_key`` per row.
        max_iter: Fixed-point steps.
        tolerance: Relative stopping tolerance.

    Returns:
        The thinned counts (CSR, float64).

    Raises:
        ResolvabilityError: If the keys do not match the rows.
    """
    from scipy import sparse as sp

    work = _exact_work(matrix)
    if len(keys) != work.shape[0]:
        raise ResolvabilityError(
            f"thinning: {len(keys)} draw keys for {work.shape[0]} rows"
        )
    _rows, probability = _exact_probabilities(
        work, targets, efficiency, max_iter=max_iter, tolerance=tolerance
    )
    counts = work.data.astype(np.int64)
    thinned = np.zeros(len(counts), dtype=np.int64)
    indptr = work.indptr
    for row, key in enumerate(keys):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if stop > start:
            thinned[start:stop] = np.random.default_rng(int(key)).binomial(
                counts[start:stop], probability[start:stop]
            )
    result = sp.csr_matrix(
        (thinned.astype(np.float64), work.indices.copy(), work.indptr.copy()),
        shape=work.shape,
    )
    result.eliminate_zeros()
    return result


def expected_thinned_totals(
    matrix: sparse.csr_matrix,
    targets: np.ndarray | Sequence[float],
    efficiency: np.ndarray,
    *,
    max_iter: int = THIN_MAX_ITER,
    tolerance: float = THIN_TOLERANCE,
) -> np.ndarray:
    """Return the expected totals ``thin_rows_exact`` draws (no sampling).

    Args:
        matrix: Rows x genes counts.
        targets: Expected total per row.
        efficiency: Per-gene efficiency.
        max_iter: Fixed-point steps.
        tolerance: Relative stopping tolerance.

    Returns:
        ``sum_g x_g p_g`` per row.
    """
    work = _exact_work(matrix)
    rows, probability = _exact_probabilities(
        work, targets, efficiency, max_iter=max_iter, tolerance=tolerance
    )
    return np.bincount(rows, weights=work.data * probability, minlength=work.shape[0])


def host_target(depth: float, spill_fraction: float) -> float:
    """Return the host share of a version-7 grid total: ``D / (1 + s)``."""
    return float(depth) / (1.0 + float(spill_fraction))


def thin_and_contaminate_v7(
    test: HeldOutCells,
    depths: Sequence[int],
    recipe: SimulationRecipe,
    *,
    efficiency: np.ndarray | None = None,
    registry: Mapping[str, Any] | None = None,
) -> SimulatedQuery:
    """Simulate test cells on a version-7 grid of TOTAL counts (v7.2).

    For grid value ``D`` and spill fraction ``s``: the host target is ``h =
    D / (1 + s)`` and the spill ``s h`` (``s = 0.25``: 0.8 D + 0.2 D); hosts
    are the test cells whose native counts reach ``h`` (``clean``: ``D``);
    the spill partner is the rendezvous choice among the cells of another
    spill group whose native counts reach ``s h`` (version 5's keys: the
    recipe's seed, name and version, the host cell id and ``D``); host and
    spill are thinned with ``thin_rows_exact``. A simulated cell's bin is
    ``D``; its realised total is recorded (``total_counts``). Version 6's
    ``thin_and_contaminate`` is unchanged.

    Args:
        test: Test cells (unique cell ids).
        depths: The version-7 grid.
        recipe: The member's recipe.
        efficiency: Per-gene efficiency (default: ``member_efficiency``).
        registry: Simulation-input registry for table recipes.

    Returns:
        The simulated query (obs adds ``host_target`` and ``member``).

    Raises:
        ResolvabilityError: If the cell ids are not unique, or a spill recipe
            has cells of a single group.
    """
    from scipy import sparse as sp

    if not test.obs.index.is_unique:
        raise ResolvabilityError(
            "test cell ids must be unique (they key the simulation draws)"
        )
    gene_eff = (
        member_efficiency(recipe, test.genes, registry=registry)
        if efficiency is None
        else np.asarray(efficiency, dtype=np.float64)
    )
    if len(gene_eff) != len(test.genes):
        raise ResolvabilityError(
            f"{recipe.member}: {len(gene_eff)} efficiencies for {len(test.genes)} genes"
        )
    native = test.native_counts
    groups = test.obs[SPILL_GROUP_COLUMN].astype(str).to_numpy()
    cell_ids = test.obs.index.astype(str).to_numpy()
    counts = sp.csr_matrix(test.counts)
    candidate_keys = np.array(
        [_key64(draw_key(DRAW_STREAM_CANDIDATE, cell_id)) for cell_id in cell_ids],
        dtype=np.uint64,
    )
    spill_fraction = float(recipe.spill_fraction)
    blocks: list[sp.csr_matrix] = []
    frames: list[pd.DataFrame] = []
    n_by_depth: dict[int, int] = {}
    for depth in sorted(int(value) for value in depths):
        target = host_target(depth, spill_fraction)
        hosts = np.flatnonzero(native >= target)
        n_by_depth[depth] = int(len(hosts))
        if len(hosts) == 0:
            continue
        host_ids = [str(cell_id) for cell_id in cell_ids[hosts]]
        host_counts = thin_rows_exact(
            counts[hosts],
            np.full(len(hosts), target),
            gene_eff,
            [
                cell_draw_key(recipe, cell_id, depth, DRAW_STREAM_THIN)
                for cell_id in host_ids
            ],
        )
        partner_ids = np.full(len(hosts), "", dtype=object)
        spill = sp.csr_matrix(host_counts.shape, dtype=np.float64)
        if spill_fraction > 0:
            amount = spill_fraction * target
            donors = np.flatnonzero(native >= amount)
            partners = np.empty(len(hosts), dtype=np.int64)
            host_keys = np.array(
                [
                    _key64(cell_draw_key(recipe, cell_id, depth, DRAW_STREAM_PARTNER))
                    for cell_id in host_ids
                ],
                dtype=np.uint64,
            )
            for group in np.unique(groups[hosts]):
                is_host = groups[hosts] == group
                candidates = donors[groups[donors] != group]
                if len(candidates) == 0:
                    raise ResolvabilityError(
                        f"no spill donor outside group {group!r} at depth {depth}"
                    )
                partners[is_host] = candidates[
                    rendezvous_choice(host_keys[is_host], candidate_keys[candidates])
                ]
            spill = thin_rows_exact(
                counts[partners],
                np.full(len(hosts), amount),
                gene_eff,
                [
                    cell_draw_key(recipe, cell_id, depth, DRAW_STREAM_SPILL)
                    for cell_id in host_ids
                ],
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
                    "host_target": np.full(len(hosts), target),
                    "member": recipe.member,
                },
                index=[simulated_id(cell_id, depth) for cell_id in cell_ids[hosts]],
            )
        )
    columns = [
        "cell_id",
        "depth",
        "partner_id",
        "host_counts",
        "spill_counts",
        "total_counts",
        "host_target",
        "member",
    ]
    if blocks:
        matrix = sp.vstack(blocks).tocsr()
        obs = pd.concat(frames)
    else:
        matrix = sp.csr_matrix((0, len(test.genes)), dtype=np.float64)
        obs = pd.DataFrame(columns=columns)
    return SimulatedQuery(
        recipe=recipe,
        counts=matrix,
        genes=list(test.genes),
        obs=obs,
        n_by_depth=n_by_depth,
    )


def v7_simulation_payload(
    *,
    members: Sequence[EnsembleMember],
    assets: Iterable[Any],
    chemistry: Mapping[str, Any] | str,
    depth_grid: Sequence[int],
    top_up: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return what a version-7 self-map's ``build_hash`` holds (v7.1).

    Every version-7 input: the version, the ensemble rule's version
    (``ENSEMBLE_RULE_VERSION``: 2 since the amendment of 2026-09-29, so no
    bundle decided by the stage A-D rule is reused), the members (recipes
    with their table sha256), every simulation-input asset used (id,
    version, sha256), the chemistry, the grid, the top-up rule and the
    simulation conventions. A version-6 bundle's payload never holds it, so
    version-6 ``build_hash`` values are unchanged (pre-registration §14 (i)).

    Args:
        members: The ensemble members.
        assets: The ``SimInputAsset`` objects used (tables, profile, lists).
        chemistry: ``ChemistryResolution.to_json()`` or the chemistry name.
        depth_grid: The version-7 grid.
        top_up: The test-set top-up rule (v7.6), when it applies.

    Returns:
        JSON-native payload.
    """
    from merxen.annotation import sim_inputs as si

    return {
        "resolvability_version": RESOLVABILITY_VERSION_V7,
        "ensemble_rule_version": ENSEMBLE_RULE_VERSION,
        "members": [member.to_json() for member in members],
        "assets": si.asset_hashes(assets),
        "chemistry": dict(chemistry) if isinstance(chemistry, Mapping) else chemistry,
        "depth_grid": [int(value) for value in depth_grid],
        "top_up": None if top_up is None else dict(top_up),
        "conventions": {
            "thinning": "exact_total",
            "thin_max_iter": THIN_MAX_ITER,
            "thin_tolerance": THIN_TOLERANCE,
            "grid_values": "total_counts",
            "host_target": "D / (1 + spill_fraction)",
        },
    }


# --------------------------------------------------------------------------
# Resolvability version 7: ensemble decisions (M3c; §8.3 v7.7-v7.10)
#
# Every emission member is decided on its own by the version-6 rule plus the
# saturated-bp rule (``decide`` with ``saturated_bp_share``); the ensemble then
# judges the union of the members' rows of each (level, class, depth) (or the
# ensemble deep pool) once, counting each test cell once (E1), and requires the
# members to agree (E2: unanimous, or the member spread within sampling
# error). The monotone fill (v7.9) runs last; floors and trust come from the
# decisions before it.

MEMBER_COLUMN: Final = "member"
MEMBER_ROLE_COLUMN: Final = "member_role"
ENSEMBLE_RECIPE: Final = "ensemble"
CLASS_DEPTH_FILE: Final = "resolvability_class_depth.parquet"
REASON_ENSEMBLE_PREFIX: Final = "ensemble_"
REASON_ENSEMBLE_SPREAD: Final = "ensemble_spread"
# The spread route's margin (amendment of 2026-09-29, pre-registration §15.3):
# E1 and the spread limit pass, the pooled Wilson bound does not clear its
# limit by the margin.
REASON_ENSEMBLE_SPREAD_MARGIN: Final = "ensemble_spread_margin"
REASONS_ENSEMBLE_E2: Final[tuple[str, ...]] = (
    REASON_ENSEMBLE_SPREAD,
    REASON_ENSEMBLE_SPREAD_MARGIN,
)
# The ensemble rule's version in a version-7 build hash: 1 = stages A-D; 2 =
# the amendment of 2026-09-29 (eight members, the spread margin).
ENSEMBLE_RULE_VERSION: Final = 2
# The spread margin of a bundle whose ensemble settings do not record one
# (built before the amendment): it re-derives its decisions as built.
SPREAD_MARGIN_BEFORE_AMENDMENT: Final = 0.0
REASON_TOO_FEW_FIT_CELLS: Final = "too_few_fit_cells"
RULE_UNANIMOUS: Final = "unanimous"
RULE_SPREAD: Final = "spread"
FILL_OWN: Final = "own"
FILL_POOL: Final = "pool"
V7_DECISION_COLUMNS: Final[tuple[str, ...]] = (
    "ensemble_rule",
    "ensemble_status",
    "ensemble_reason",
    "member_emitted",
    "member_statuses",
    "member_t_star",
    "member_precision_min",
    "member_precision_max",
    "member_coverage_min",
    "member_coverage_max",
    "member_min_n",
    "member_spread",
    "spread_limit",
    "spread_ok",
    "spread_se",
    "wilson_clearance",
    "spread_margin_ok",
    "n_rows",
    "monotone_filled",
    "fill_source",
    "fill_precision",
    "fill_coverage",
    "neuronal",
    "nonneuronal_high_depth",
)
V7_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "threshold_source",
    "ensemble_rule",
    "ensemble_status",
    "ensemble_reason",
    "member_statuses",
    "member_t_star",
    "fill_source",
)
V7_BOOL_COLUMNS: Final[tuple[str, ...]] = (
    "saturated_bp",
    "member_emitted",
    "spread_ok",
    "spread_margin_ok",
    "monotone_filled",
    "nonneuronal_high_depth",
)
ENSEMBLE_DECISION_COLUMNS: Final[tuple[str, ...]] = (
    *DECISION_COLUMNS,
    *SATURATED_COLUMNS,
    *V7_DECISION_COLUMNS,
)

NeuronalOf = Mapping[str, bool | None] | Callable[[str], bool | None]


@dataclass(frozen=True)
class EnsembleSettings:
    """The version-7 ensemble rules (§3.7; pre-registration §14.3, tightenable).

    Attributes:
        spread_floor: E2's smallest allowed member spread (0.03).
        spread_se_multiplier: E2's spread limit in standard errors (3.5).
        member_min_confident: Confident check-half calls (Kish n) each member
            needs for the spread test (10).
        saturated_bp_share: Fit-half share at ``bp = 1`` above which a set
            without ``t*`` is judged at the cap (0.90; v7.8).
        monotone_depth: Apply the monotone fill (v7.9).
        nonneuronal_monotone_max_depth: Non-neuronal bins at or above this
            depth are never filled (1,000).
        spread_wilson_margin_se: The spread route of E2 needs the pooled
            Wilson bound to clear ``target - wilson_margin`` by this many
            standard errors of the pooled precision (1; amendment of
            2026-09-29, pre-registration §15.3; 0: no margin, the rule of
            stages A-D).
    """

    spread_floor: float = 0.03
    spread_se_multiplier: float = 3.5
    member_min_confident: int = 10
    saturated_bp_share: float = 0.90
    monotone_depth: bool = True
    nonneuronal_monotone_max_depth: int = 1000
    spread_wilson_margin_se: float = 1.0

    def spread_limit(self, p_bar: float, n_bar: float) -> float:
        """Return ``max(floor, multiplier * sqrt(p (1 - p) / n))`` (NP4's statistic).

        Args:
            p_bar: Mean member precision.
            n_bar: Mean member Kish n.

        Returns:
            The limit (the floor when ``n_bar`` is not positive).
        """
        if not n_bar > 0 or not math.isfinite(p_bar):
            return float(self.spread_floor)
        p = min(max(float(p_bar), 0.0), 1.0)
        return max(
            float(self.spread_floor),
            float(self.spread_se_multiplier) * math.sqrt(p * (1.0 - p) / n_bar),
        )

    @classmethod
    def from_config(cls, resolvability: Any) -> EnsembleSettings:
        """Return the settings of an ``AnnotationResolvabilityConfig``."""
        defaults = cls()
        return cls(
            spread_floor=float(
                getattr(resolvability, "ensemble_spread_floor", defaults.spread_floor)
            ),
            spread_se_multiplier=float(
                getattr(
                    resolvability,
                    "ensemble_spread_se_multiplier",
                    defaults.spread_se_multiplier,
                )
            ),
            member_min_confident=int(
                getattr(
                    resolvability,
                    "ensemble_member_min_confident",
                    defaults.member_min_confident,
                )
            ),
            saturated_bp_share=float(
                getattr(
                    resolvability, "saturated_bp_share", defaults.saturated_bp_share
                )
            ),
            monotone_depth=bool(
                getattr(resolvability, "monotone_depth", defaults.monotone_depth)
            ),
            nonneuronal_monotone_max_depth=int(
                getattr(
                    resolvability,
                    "nonneuronal_monotone_max_depth",
                    defaults.nonneuronal_monotone_max_depth,
                )
            ),
            spread_wilson_margin_se=float(
                getattr(
                    resolvability,
                    "ensemble_spread_wilson_margin_se",
                    defaults.spread_wilson_margin_se,
                )
            ),
        )

    @classmethod
    def from_json(cls, payload: Mapping[str, Any] | None) -> EnsembleSettings:
        """Rebuild the settings from ``to_json`` output (defaults for gaps).

        A bundle built before the amendment of 2026-09-29 records no spread
        margin and gets ``SPREAD_MARGIN_BEFORE_AMENDMENT`` (0), so its
        decisions re-derive as built.
        """
        fields_ = cls.__dataclass_fields__
        values = {k: v for k, v in dict(payload or {}).items() if k in fields_}
        values.setdefault("spread_wilson_margin_se", SPREAD_MARGIN_BEFORE_AMENDMENT)
        return cls(**values)

    def spread_margin_ok(self, wilson_clearance: float, standard_error: float) -> bool:
        """Return whether a set may use E2's spread route (pre-registration §15.3).

        Args:
            wilson_clearance: ``L - (target - wilson_margin)``, the pooled
                Wilson bound's clearance of its E1 limit.
            standard_error: ``pooled_standard_error`` of the set.

        Returns:
            True without a margin (0); else whether the clearance reaches
            ``spread_wilson_margin_se`` standard errors (float tolerance).
        """
        if not self.spread_wilson_margin_se > 0:
            return True
        if not (math.isfinite(wilson_clearance) and math.isfinite(standard_error)):
            return False
        return bool(
            wilson_clearance
            >= float(self.spread_wilson_margin_se) * standard_error - _TOLERANCE
        )

    def to_json(self) -> dict[str, Any]:
        """Return the settings as JSON."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _neuronal(neuronal: NeuronalOf | None, cls: str) -> bool | None:
    if neuronal is None:
        return None
    if isinstance(neuronal, Mapping):
        value = neuronal.get(str(cls))
    else:
        value = neuronal(str(cls))
    return None if value is None else bool(value)


@dataclass(frozen=True)
class _SetArrays:
    """The rows of one tested set as arrays (every member's calls)."""

    bp: np.ndarray
    correct: np.ndarray
    weight: np.ndarray
    half: np.ndarray
    member: np.ndarray
    cell: np.ndarray

    @classmethod
    def of(cls, frame: pd.DataFrame | None, trim_factor: float) -> _SetArrays | None:
        if frame is None or len(frame) == 0:
            return None
        return cls(
            bp=frame["bp"].to_numpy(np.float64),
            correct=frame["correct"].to_numpy(bool).astype(np.float64),
            weight=trim_weights(frame["_weight"].to_numpy(np.float64), trim_factor),
            half=frame["half"].to_numpy(np.int8),
            member=frame["_member"].to_numpy(np.int64),
            cell=frame["_cell"].to_numpy(np.int64),
        )

    def __len__(self) -> int:
        return len(self.bp)

    def select(self, mask: np.ndarray) -> _SetArrays:
        return _SetArrays(
            bp=self.bp[mask],
            correct=self.correct[mask],
            weight=self.weight[mask],
            half=self.half[mask],
            member=self.member[mask],
            cell=self.cell[mask],
        )

    def check_mask(self, regime: Regime, settings: RuleSettings) -> np.ndarray:
        """The rows a regime checks: all (validated), else the check half."""
        if regime == "validated" or not settings.split_halves:
            return np.ones(len(self.bp), dtype=bool)
        return np.asarray(self.half == 1, dtype=bool)

    def fit_mask(self, settings: RuleSettings) -> np.ndarray:
        mask = (
            self.half == 0
            if settings.split_halves
            else np.ones(len(self.bp), dtype=bool)
        )
        return np.asarray(mask & np.isfinite(self.bp) & (self.weight > 0), dtype=bool)


def distinct_cell_stats(
    bp: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray,
    cells: np.ndarray,
    threshold: float | None,
) -> CheckStats:
    """Return a pooled set's statistics with each test cell counted once (v7.7).

    The members re-simulate the same test cells, which add no test-cell
    information: ``n_called`` and ``n_confident`` count distinct test cells,
    the precision and coverage are weighted over every (member, cell) row,
    and the Wilson bound uses ``n_effective = Kish n x distinct / rows`` (the
    distinct count when unweighted).

    Args:
        bp: bp of the set's rows.
        correct: Their correctness (0 / 1).
        weights: Their weights (trimmed).
        cells: The test cell of each row (any hashable codes).
        threshold: The applied threshold (``None``: nothing is confident).

    Returns:
        The statistics.
    """
    positive = np.asarray(weights, dtype=np.float64) > 0
    n_called = int(np.unique(cells[positive]).size) if positive.any() else 0
    if threshold is None or n_called == 0:
        return CheckStats(threshold, n_called, 0, 0.0, math.nan, math.nan, 0.0)
    accepted = meets_threshold(bp, threshold) & positive
    accepted_weights = weights[accepted]
    total = float(accepted_weights.sum())
    called_total = float(weights[positive].sum())
    if total <= 0:
        return CheckStats(threshold, n_called, 0, 0.0, math.nan, math.nan, 0.0)
    distinct = int(np.unique(cells[accepted]).size)
    rows = int(accepted.sum())
    precision = float((accepted_weights * correct[accepted]).sum()) / total
    n_effective = kish_effective_n(accepted_weights) * distinct / rows
    return CheckStats(
        threshold=threshold,
        n_called=n_called,
        n_confident=distinct,
        n_effective=n_effective,
        precision=precision,
        wilson_lb=wilson_lower_bound(precision, n_effective),
        coverage=total / called_total if called_total > 0 else 0.0,
        max_weight_share=float(accepted_weights.max()) / total,
    )


def pooled_standard_error(precision: float, n_effective: float) -> float:
    """Return ``sqrt(p (1 - p) / n_eff)``, the SE of a pooled set's precision.

    The binomial standard error of the pooled point precision on the
    effective n of its Wilson bound (Kish n x distinct test cells / rows; the
    distinct confidently called test cells when unweighted), NP4's statistic
    applied to the pooled set (amendment of 2026-09-29, pre-registration
    §15.3). 0 when ``p`` is 0 or 1.

    Args:
        precision: The pooled point precision at the applied threshold.
        n_effective: Its effective n.

    Returns:
        The standard error (``nan`` without a precision or a positive n).
    """
    if not (math.isfinite(precision) and n_effective > 0):
        return math.nan
    p = min(max(float(precision), 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / float(n_effective))


def member_spread(
    precisions: Sequence[float],
    effective_n: Sequence[float],
    ensemble: EnsembleSettings,
) -> tuple[float, float, bool]:
    """Return ``(spread, limit, within)`` of E2 (v7.7).

    Args:
        precisions: Each emission member's check-half precision at the
            ensemble's threshold (``nan``: no confident call).
        effective_n: Each member's Kish n of those confident calls.
        ensemble: The ensemble settings.

    Returns:
        ``max p - min p``, the limit ``max(floor, k * sqrt(p (1 - p) / n))``
        on the means, and whether every member has ``member_min_confident``
        calls and the spread is within the limit.
    """
    p = np.asarray(precisions, dtype=np.float64)
    n = np.asarray(effective_n, dtype=np.float64)
    if not len(p) or not np.all(np.isfinite(p)):
        return math.nan, float(ensemble.spread_floor), False
    spread = float(p.max() - p.min())
    limit = ensemble.spread_limit(float(p.mean()), float(n.mean()))
    enough = bool(np.all(n >= ensemble.member_min_confident - _TOLERANCE))
    return spread, limit, enough and spread <= limit + _TOLERANCE


def _set_records(
    arrays: _SetArrays | None,
    meta: LevelMeta,
    depth: int,
    settings: RuleSettings,
    ensemble: EnsembleSettings,
    base: dict[str, Any],
    n_members: int,
) -> dict[str, dict[str, Any]]:
    """Judge one ensemble tested set (a bin's pooled rows or a deep pool) per regime.

    E1's statistics (``distinct_cell_stats``) at ``t*_pool`` (the isotonic
    fit on every member's fit-half rows; the saturated cap, v7.8; the default
    in the validated regime) and each member's precision, coverage and Kish
    n there (E2's spread). The E2 unanimity is per bin (``_verdict``).
    """
    records: dict[str, dict[str, Any]] = {}
    if arrays is None:
        for regime in REGIMES:
            records[regime] = {
                **base,
                "regime": regime,
                "target": settings.target(regime, meta.base_target, depth),
                "threshold": None,
                "t_star": None,
                "threshold_source": None,
                "saturated_bp": False,
                "saturated_share": math.nan,
                "n_called": 0,
                "n_fit": 0,
                "n_confident": 0,
                "n_rows": 0,
                "e1_reason": "no_calls",
                "spread_ok": False,
                "spread_se": math.nan,
                "wilson_clearance": math.nan,
                "spread_margin_ok": False,
                "would_raise": False,
                "would_raise_evaluable": False,
            }
        return records
    fit_mask = arrays.fit_mask(settings)
    n_fit = int(np.unique(arrays.cell[fit_mask]).size) if fit_mask.any() else 0
    fit: IsotonicFit | None = None
    if n_fit >= settings.min_cells_per_bin:
        fit = isotonic_fit(
            arrays.bp[fit_mask], arrays.correct[fit_mask], arrays.weight[fit_mask]
        )
    saturated_share = (
        saturated_bp_fraction(arrays.bp[fit_mask]) if fit is not None else math.nan
    )
    at_default = distinct_cell_stats(
        arrays.bp, arrays.correct, arrays.weight, arrays.cell, meta.default_threshold
    )
    g_default = (
        float(fit.predict(meta.default_threshold)) if fit is not None else math.nan
    )
    for regime in REGIMES:
        target = settings.target(regime, meta.base_target, depth)
        t_star = local_threshold(
            fit,
            default=meta.default_threshold,
            target=target,
            cap=settings.threshold_cap,
        )
        saturated = False
        if regime == "validated":
            applied: float | None = meta.default_threshold
        else:
            applied = t_star
            saturated = (
                t_star is None
                and fit is not None
                and saturated_share > ensemble.saturated_bp_share
            )
            if saturated:
                applied = settings.threshold_cap
        check = arrays.select(arrays.check_mask(regime, settings))
        stats = distinct_cell_stats(
            check.bp, check.correct, check.weight, check.cell, applied
        )
        e1_reason = _rule_pass(stats, target, settings)
        if fit is None and regime != "validated":
            e1_reason = REASON_TOO_FEW_FIT_CELLS
        precisions: list[float] = []
        effective: list[float] = []
        coverages: list[float] = []
        for code in range(n_members):
            mine = check.select(check.member == code)
            member_stats = check_threshold(mine.bp, mine.correct, mine.weight, applied)
            precisions.append(member_stats.precision)
            effective.append(member_stats.n_effective)
            coverages.append(member_stats.coverage)
        spread, limit, within = member_spread(precisions, effective, ensemble)
        # The spread route's margin (pre-registration §15.3): the pooled
        # Wilson bound's clearance of its E1 limit, in SE of the precision.
        standard_error = pooled_standard_error(stats.precision, stats.n_effective)
        clearance = float(stats.wilson_lb) - (target - settings.wilson_margin)
        finite_p = [value for value in precisions if math.isfinite(value)]
        n_rows = (
            int((meets_threshold(check.bp, applied) & (check.weight > 0)).sum())
            if applied is not None
            else 0
        )
        records[regime] = {
            **base,
            "regime": regime,
            "target": target,
            "threshold": applied,
            "t_star": t_star,
            "threshold_source": _threshold_source(regime, applied, saturated),
            "saturated_bp": bool(saturated),
            "saturated_share": saturated_share,
            "would_raise": bool(
                fit is not None
                and (t_star is None or t_star > meta.default_threshold + _TOLERANCE)
            ),
            "would_raise_evaluable": fit is not None,
            "check_set": "all"
            if regime == "validated" or not settings.split_halves
            else "check_half",
            "n_called": stats.n_called,
            "n_fit": n_fit,
            "n_confident": stats.n_confident,
            "n_effective": stats.n_effective,
            "max_weight_share": stats.max_weight_share,
            "precision": stats.precision,
            "wilson_lb": stats.wilson_lb,
            "coverage": stats.coverage,
            "g_at_default": g_default,
            "precision_at_default": at_default.precision,
            "n_rows": n_rows,
            "e1_reason": e1_reason,
            "member_precision_min": min(finite_p) if finite_p else math.nan,
            "member_precision_max": max(finite_p) if finite_p else math.nan,
            "member_coverage_min": float(np.min(coverages)) if coverages else math.nan,
            "member_coverage_max": float(np.max(coverages)) if coverages else math.nan,
            "member_min_n": float(np.min(effective)) if effective else math.nan,
            "member_spread": spread,
            "spread_limit": limit,
            "spread_ok": bool(within),
            "spread_se": standard_error,
            "wilson_clearance": clearance,
            "spread_margin_ok": ensemble.spread_margin_ok(clearance, standard_error),
        }
    return records


def _ensemble_judged(record: Mapping[str, Any], settings: RuleSettings) -> bool:
    """``_judged`` for an ensemble set: distinct test cells, E1's reason."""
    if int(record.get("n_confident") or 0) >= settings.min_confident_n:
        return True
    return (
        record.get("e1_reason") == REASON_NO_LOCAL_THRESHOLD
        and int(record.get("n_called") or 0) >= settings.min_confident_n
    )


def _verdict(record: Mapping[str, Any], unanimous: bool) -> dict[str, Any]:
    """E1 and E2 of one tested set: its status, reason and ensemble rule (v7.7).

    E1 first; then E2's unanimous route (every emission member emits the
    bin) or its spread route, which needs the spread within its limit and,
    since the amendment of 2026-09-29 (pre-registration §15.3), the pooled
    Wilson bound clearing its limit by the margin (``spread_margin_ok``,
    computed by ``_set_records``). Used for a bin's own set, the pool a
    pooled bin takes and the pool whose failure withdraws the judged bins.
    """
    e1 = record.get("e1_reason")
    if e1 is not None:
        return {
            "status": STATUS_NOT_RESOLVABLE,
            "reason": f"{REASON_ENSEMBLE_PREFIX}{e1}",
            "ensemble_rule": None,
        }
    if unanimous:
        return {
            "status": STATUS_EMITTED,
            "reason": None,
            "ensemble_rule": RULE_UNANIMOUS,
        }
    if not record.get("spread_ok"):
        return {
            "status": STATUS_NOT_RESOLVABLE,
            "reason": REASON_ENSEMBLE_SPREAD,
            "ensemble_rule": None,
        }
    if not record.get("spread_margin_ok"):
        return {
            "status": STATUS_NOT_RESOLVABLE,
            "reason": REASON_ENSEMBLE_SPREAD_MARGIN,
            "ensemble_rule": None,
        }
    return {"status": STATUS_EMITTED, "reason": None, "ensemble_rule": RULE_SPREAD}


class _EnsemblePools:
    """The ensemble deep pools of one level: each (member, test cell) once."""

    def __init__(
        self,
        level_rows: pd.DataFrame,
        n_members: int,
        pool_weights: PoolWeights | None,
        trim_factor: float,
    ) -> None:
        parts = [
            deepest_rows(level_rows[(level_rows["_member"] == code).to_numpy()])
            for code in range(n_members)
        ]
        self.deepest = [part for part in parts if len(part)]
        self.pool_weights = pool_weights
        self.trim_factor = trim_factor
        self._by_depth: dict[int, pd.DataFrame] = {}
        self._arrays: dict[tuple[str, int], _SetArrays | None] = {}

    def frame(self, cls: str, depth: int) -> pd.DataFrame:
        if depth not in self._by_depth:
            subsets = []
            for part in self.deepest:
                subset = part[part["depth"].to_numpy(np.int64) >= depth]
                if self.pool_weights is not None and len(subset):
                    subset = subset.copy()
                    subset["_weight"] = self.pool_weights(subset, depth)
                subsets.append(subset[subset["parent"].notna().to_numpy()])
            self._by_depth[depth] = (
                pd.concat(subsets, ignore_index=True) if subsets else pd.DataFrame()
            )
        frame = self._by_depth[depth]
        if frame.empty:
            return frame
        return frame[(frame["parent"].astype(str) == cls).to_numpy()]

    def arrays(self, cls: str, depth: int) -> _SetArrays | None:
        key = (cls, depth)
        if key not in self._arrays:
            self._arrays[key] = _SetArrays.of(self.frame(cls, depth), self.trim_factor)
        return self._arrays[key]


def _member_text(values: Mapping[str, Any]) -> str:
    return ";".join(f"{name}={values[name]}" for name in values)


def _ensemble_class_decisions(
    meta: LevelMeta,
    cls: str,
    grid: Sequence[int],
    settings: RuleSettings,
    ensemble: EnsembleSettings,
    *,
    groups: Mapping[tuple[str, int], pd.DataFrame],
    pools: _EnsemblePools,
    class_dmax: int | None,
    n_test_index: Mapping[tuple[str, str, int], int],
    member_view: Mapping[tuple[str, str, str, int], Mapping[str, tuple[str, Any]]],
    members: Sequence[str],
    neuronal: bool | None,
) -> list[dict[str, Any]]:
    """Decide every (regime, depth) bin of one (level, class) of the ensemble."""
    high_depth = ensemble.nonneuronal_monotone_max_depth

    def base(depth: int) -> dict[str, Any]:
        return {
            "level": meta.level,
            "class": cls,
            "depth": depth,
            "n_test": n_test_index.get((meta.level, cls, depth), 0),
            "d_max": class_dmax,
            "default_threshold": meta.default_threshold,
            "neuronal": neuronal,
            "nonneuronal_high_depth": bool(
                neuronal is not True and depth >= high_depth
            ),
        }

    def member_fields(regime: str, depth: int) -> tuple[bool, dict[str, Any]]:
        view = member_view.get((regime, meta.level, cls, depth), {})
        statuses = {name: (view.get(name) or (None, None))[0] for name in members}
        t_stars = {
            name: _optional_float((view.get(name) or (None, None))[1])
            for name in members
        }
        unanimous = bool(members) and all(
            value == STATUS_EMITTED for value in statuses.values()
        )
        return unanimous, {
            "member_emitted": unanimous,
            "member_statuses": _member_text(statuses),
            "member_t_star": _member_text(
                {k: "" if v is None else f"{v:.3f}" for k, v in t_stars.items()}
            ),
        }

    records: list[dict[str, Any]] = []
    if class_dmax is None:
        for depth in grid:
            for regime in REGIMES:
                _unanimous, fields_ = member_fields(regime, depth)
                records.append(
                    {
                        **base(depth),
                        **fields_,
                        "regime": regime,
                        "target": settings.target(regime, meta.base_target, depth),
                        "status": STATUS_NOT_RESOLVABLE,
                        "extrapolated": False,
                        "pooled": False,
                        "reason": REASON_TOO_FEW_TEST_CELLS,
                        "ensemble_status": STATUS_NOT_RESOLVABLE,
                        "ensemble_reason": REASON_TOO_FEW_TEST_CELLS,
                        "monotone_filled": False,
                    }
                )
        return records
    trim = settings.set_trim_factor
    own_arrays = {
        depth: _SetArrays.of(groups.get((cls, depth)), trim) for depth in grid
    }
    own = {
        depth: _set_records(
            own_arrays[depth],
            meta,
            depth,
            settings,
            ensemble,
            base(depth),
            len(members),
        )
        for depth in grid
    }
    pooled: dict[int, dict[str, dict[str, Any]]] = {}

    def pooled_at(depth: int) -> dict[str, dict[str, Any]]:
        if depth not in pooled:
            pooled[depth] = _set_records(
                pools.arrays(cls, depth),
                meta,
                depth,
                settings,
                ensemble,
                base(depth),
                len(members),
            )
        return pooled[depth]

    for regime in REGIMES:
        pool_min: int | None = None
        insufficient = False
        if not _ensemble_judged(own[grid[-1]][regime], settings):
            insufficient = True
            for depth in reversed(grid):
                if _ensemble_judged(pooled_at(depth)[regime], settings):
                    pool_min, insufficient = depth, False
                    break
        regime_records: list[dict[str, Any]] = []
        for depth in grid:
            mine = own[depth][regime]
            unanimous, fields_ = member_fields(regime, depth)
            own_verdict = _verdict(mine, unanimous)
            own_fields = {
                **fields_,
                "own_status": own_verdict["status"],
                "own_reason": own_verdict["reason"],
                "own_n_confident": mine.get("n_confident"),
                "pool_min_depth": pool_min,
            }
            judged = _ensemble_judged(mine, settings)
            if insufficient and not judged:
                record = {
                    **mine,
                    **own_fields,
                    "status": STATUS_NOT_RESOLVABLE,
                    "reason": REASON_INSUFFICIENT_CALLS,
                    "ensemble_rule": None,
                    "pooled": False,
                    "extrapolated": False,
                }
            elif pool_min is None or depth < pool_min:
                record = {
                    **mine,
                    **own_fields,
                    **own_verdict,
                    "pooled": False,
                    "extrapolated": False,
                }
            elif judged:
                record = {
                    **mine,
                    **own_fields,
                    **own_verdict,
                    "pooled": False,
                    "extrapolated": False,
                }
                pool_unanimous, _ = member_fields(regime, pool_min)
                pool_verdict = _verdict(pooled_at(pool_min)[regime], pool_unanimous)
                if (
                    record["status"] == STATUS_EMITTED
                    and pool_verdict["status"] != STATUS_EMITTED
                ):
                    record.update(
                        status=STATUS_NOT_RESOLVABLE,
                        reason=f"{POOL_REASON_PREFIX}{pool_verdict['reason']}",
                        ensemble_rule=None,
                    )
            else:
                verdict_record = pooled_at(pool_min)[regime]
                record = {
                    **verdict_record,
                    **own_fields,
                    **_verdict(verdict_record, unanimous),
                    **base(depth),
                    "regime": regime,
                    "target": verdict_record["target"],
                    "extrapolated": depth > pool_min,
                    "pooled": True,
                }
            record["ensemble_status"] = record["status"]
            record["ensemble_reason"] = record["reason"]
            record["monotone_filled"] = False
            record.pop("e1_reason", None)
            regime_records.append(record)
        if ensemble.monotone_depth:
            _monotone_fill(
                regime_records,
                regime=regime,
                own=own,
                own_arrays=own_arrays,
                pool_arrays=None if pool_min is None else pools.arrays(cls, pool_min),
                pool_min=pool_min,
                settings=settings,
                ensemble=ensemble,
                neuronal=neuronal,
            )
        records.extend(regime_records)
    return records


def _monotone_fill(
    records: list[dict[str, Any]],
    *,
    regime: Regime,
    own: Mapping[int, Mapping[str, Mapping[str, Any]]],
    own_arrays: Mapping[int, _SetArrays | None],
    pool_arrays: _SetArrays | None,
    pool_min: int | None,
    settings: RuleSettings,
    ensemble: EnsembleSettings,
    neuronal: bool | None,
) -> None:
    """Fill bins deeper than the shallowest emitted one (v7.9; in place).

    A bin v7.7 did not emit is filled when, at ``t_d`` (its own ``t*_pool``
    or saturated cap, else the applied threshold of the nearest shallower
    emitted bin), its point precision reaches its target and its coverage
    ``min_coverage``, measured on its own pooled calls when it has any, else
    on the ensemble deep pool it belongs to; only the power conditions and a
    missing threshold are waived. Non-neuronal (or unknown-lineage) classes
    are never filled at ``nonneuronal_monotone_max_depth`` or deeper. The
    caller has checked that the class has ``D_max``.
    """
    emitted = [int(r["depth"]) for r in records if r["status"] == STATUS_EMITTED]
    if not emitted:
        return
    shallowest = min(emitted)
    inherited: float | None = None
    for record in records:
        depth = int(record["depth"])
        if record["status"] == STATUS_EMITTED:
            inherited = _optional_float(record["threshold"])
            continue
        if depth <= shallowest:
            continue
        if neuronal is not True and depth >= ensemble.nonneuronal_monotone_max_depth:
            continue
        own_record = own[depth][regime]
        threshold = _optional_float(own_record.get("threshold"))
        source = own_record.get("threshold_source")
        if threshold is None:
            threshold, source = inherited, THRESHOLD_SOURCE_INHERITED
        if threshold is None:
            continue
        measured: tuple[str, _SetArrays] | None = None
        arrays = own_arrays.get(depth)
        if arrays is not None:
            check = arrays.select(arrays.check_mask(regime, settings))
            if bool((check.weight > 0).any()):
                measured = (FILL_OWN, check)
        if (
            measured is None
            and pool_arrays is not None
            and pool_min is not None
            and depth >= pool_min
        ):
            check = pool_arrays.select(pool_arrays.check_mask(regime, settings))
            if bool((check.weight > 0).any()):
                measured = (FILL_POOL, check)
        if measured is None:
            continue
        stats = distinct_cell_stats(
            measured[1].bp,
            measured[1].correct,
            measured[1].weight,
            measured[1].cell,
            threshold,
        )
        # The bin's own target (a pooled bin carries its pool's, D_P's).
        target = float(own_record["target"])
        if not (
            math.isfinite(stats.precision)
            and stats.precision >= target - _TOLERANCE
            and stats.coverage >= settings.min_coverage - _TOLERANCE
        ):
            continue
        record.update(
            status=STATUS_EMITTED,
            reason=None,
            threshold=threshold,
            threshold_source=source,
            monotone_filled=True,
            fill_source=measured[0],
            fill_precision=stats.precision,
            fill_coverage=stats.coverage,
            extrapolated=True,
        )
        inherited = threshold


@dataclass
class EnsembleDecisions:
    """The version-7 decisions of one self-map (v7.7-v7.9).

    Attributes:
        decisions: One row per (regime, level, class, depth) after the
            monotone fill (``ENSEMBLE_DECISION_COLUMNS``).
        member_decisions: Each member's own ``decide`` rows (saturated-bp rule
            included) with ``member`` and ``member_role``.
        members: The emission members, in order.
    """

    decisions: pd.DataFrame
    member_decisions: pd.DataFrame
    members: list[str]

    def unfilled(self) -> pd.DataFrame:
        """Return the decisions before the monotone fill (floors and trust)."""
        return unfilled_decisions(self.decisions)


def unfilled_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Return ensemble decisions with the monotone fill undone (v7.9).

    Filled bins take no part in the trust tests or the floors: they get back
    their v7.7 status and reason.

    Args:
        decisions: ``ensemble_decide`` decisions.

    Returns:
        A copy with ``status`` = ``ensemble_status`` and ``reason`` =
        ``ensemble_reason`` on filled rows.
    """
    frame = decisions.copy()
    if "monotone_filled" not in frame.columns:
        return frame
    filled = _bool_column(frame["monotone_filled"]).to_numpy()
    if filled.any():
        frame.loc[filled, "status"] = frame.loc[filled, "ensemble_status"]
        frame.loc[filled, "reason"] = frame.loc[filled, "ensemble_reason"]
        frame.loc[filled, "extrapolated"] = False
    return frame


def member_names(cells: pd.DataFrame) -> list[str]:
    """Return the members of a version-7 cells table in first-seen order."""
    if MEMBER_COLUMN not in cells.columns:
        raise ResolvabilityError("a version-7 cells table needs a member column")
    return list(dict.fromkeys(cells[MEMBER_COLUMN].astype(str)))


def ensemble_decide(
    cells: pd.DataFrame,
    levels: Sequence[LevelMeta],
    depths: Sequence[int],
    settings: RuleSettings,
    ensemble: EnsembleSettings,
    *,
    members: Sequence[str],
    reported: Sequence[str] = (),
    weights: np.ndarray | None = None,
    pool_weights: PoolWeights | None = None,
    neuronal: NeuronalOf | None = None,
) -> EnsembleDecisions:
    """Return the version-7 ensemble decisions (plan §8.3 v7.7-v7.9).

    Each emission member (and each ``reported`` member, for the record) is
    decided on its own by ``decide`` with the saturated-bp rule. Per
    (regime, level, class, depth) the ensemble's tested set is the union of
    the emission members' rows (each test cell counted once: n, Kish n x
    distinct / rows), or, when the bin holds fewer than ``min_confident_n``
    distinct confident test cells, the ensemble deep pool (from the deep end,
    each (member, test cell) once at its deepest bin) exactly as version 6
    pools. The bin is emitted iff (E1) the set passes the §8.3 rule at
    ``t*_pool`` (the saturated cap, v7.8; the default in the validated
    regime) and (E2) every emission member emits the bin by its own decision
    or the member spread there is within ``max(floor, k * SE)`` with every
    member holding ``member_min_confident`` calls and, since the amendment of
    2026-09-29 (pre-registration §15.3), the set's Wilson bound clears
    ``target - wilson_margin`` by ``spread_wilson_margin_se`` standard errors
    of its precision (``pooled_standard_error``). Then the monotone fill
    (v7.9). A failure takes E1's reason prefixed ``ensemble_``,
    ``ensemble_spread`` or ``ensemble_spread_margin``.

    Args:
        cells: A version-7 cells table (``member`` column).
        levels: Level metadata.
        depths: The depth grid.
        settings: Rule settings.
        ensemble: Ensemble settings.
        members: The emission members (``<recipe>@<seed>``).
        reported: Further members decided for the record only (``clean``,
            stress).
        weights: Optional per-row weights (RESOLVE; per member).
        pool_weights: Optional weights of pooled deep sets, applied per member.
        neuronal: Per class, whether it is neuronal (v7.9); unknown classes
            are treated as non-neuronal.

    Returns:
        The decisions.

    Raises:
        ResolvabilityError: Without emission members or a member column.
    """
    if not members:
        raise ResolvabilityError("ensemble_decide needs at least one emission member")
    row_weights = (
        np.ones(len(cells), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if len(row_weights) != len(cells):
        raise ResolvabilityError("weights must have one value per cells row")
    names = cells[MEMBER_COLUMN].astype(str).to_numpy() if len(cells) else np.array([])
    member_frames: list[pd.DataFrame] = []
    for role, group in (("emission", members), ("reported", reported)):
        for name in group:
            mask = names == name
            rows = cells[mask]
            recipe = str(rows["recipe"].iloc[0]) if len(rows) else DECISION_RECIPE
            decided = decide(
                rows,
                levels,
                depths,
                settings,
                weights=row_weights[mask],
                pool_weights=pool_weights,
                recipe=recipe,
                seed=0,
                saturated_bp_share=ensemble.saturated_bp_share,
            )
            decided.insert(0, MEMBER_ROLE_COLUMN, role)
            decided.insert(0, MEMBER_COLUMN, name)
            member_frames.append(decided)
    # Object-typed before the concat: a member's all-missing column (no
    # threshold anywhere) must not decide the column's dtype.
    member_decisions = pd.concat(
        [frame.astype(object) for frame in member_frames], ignore_index=True
    ).infer_objects()
    member_view: dict[tuple[str, str, str, int], dict[str, tuple[str, Any]]] = {}
    emission_rows = member_decisions[member_decisions[MEMBER_ROLE_COLUMN] == "emission"]
    for member, regime, level, cls, depth, status, t_star in zip(
        emission_rows[MEMBER_COLUMN],
        emission_rows["regime"],
        emission_rows["level"],
        emission_rows["class"],
        emission_rows["depth"],
        emission_rows["status"],
        emission_rows["t_star"],
        strict=True,
    ):
        member_view.setdefault((str(regime), str(level), str(cls), int(depth)), {})[
            str(member)
        ] = (str(status), t_star)
    code_of = {name: code for code, name in enumerate(members)}
    selected = np.isin(names, list(members)) & (
        cells["seed"].to_numpy() == 0 if len(cells) else np.zeros(0, dtype=bool)
    )
    frame = cells[selected].copy()
    frame["_weight"] = row_weights[selected]
    frame["_member"] = np.array(
        [code_of[name] for name in names[selected]], dtype=np.int64
    )
    frame["_cell"] = pd.factorize(frame["cell_id"].astype(str))[0].astype(np.int64)
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
        pools = _EnsemblePools(
            level_rows, len(members), pool_weights, settings.set_trim_factor
        )
        classes = sorted(
            {str(cls) for cls in called["parent"].dropna().astype(str)}
            | {cls for (level, cls) in dmax if level == meta.level}
        )
        for cls in classes:
            records.extend(
                _ensemble_class_decisions(
                    meta,
                    cls,
                    grid,
                    settings,
                    ensemble,
                    groups=groups,
                    pools=pools,
                    class_dmax=dmax.get((meta.level, cls)),
                    n_test_index=n_test_index,
                    member_view=member_view,
                    members=list(members),
                    neuronal=_neuronal(neuronal, cls),
                )
            )
    return EnsembleDecisions(
        decisions=_ensemble_table(records),
        member_decisions=member_decisions,
        members=list(members),
    )


def _ensemble_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return ensemble records as a typed table (``ENSEMBLE_DECISION_COLUMNS``)."""
    columns = list(ENSEMBLE_DECISION_COLUMNS)
    text = set(DECISION_TEXT_COLUMNS) | set(V7_TEXT_COLUMNS)
    table = pd.DataFrame.from_records(list(records))
    for column in columns:
        if column not in table.columns:
            table[column] = None if column in text else np.nan
    table = table[columns].copy()
    for column in ("extrapolated", "would_raise", "would_raise_evaluable", "pooled"):
        table[column] = _bool_column(table[column])
    for column in V7_BOOL_COLUMNS:
        table[column] = _bool_column(table[column])
    table["neuronal"] = pd.array(
        [
            None if value is None or pd.isna(value) else bool(value)
            for value in table["neuronal"]
        ],
        dtype="boolean",
    )
    return table


# --------------------------------------------------------------------------
# Version 7: per-class depth tables, summaries and churn (v7.5, v7.10)

CLASS_DEPTH_COLUMNS: Final[tuple[str, ...]] = (
    "regime",
    "level",
    "class",
    "depth",
    "status",
    "threshold",
    "threshold_source",
    "ensemble_rule",
    "pooled",
    "extrapolated",
    "monotone_filled",
    "nonneuronal_high_depth",
    "neuronal",
    "n_test",
    "n_confident",
    "precision",
    "coverage",
    "member_precision_min",
    "member_precision_max",
    "member_coverage_min",
    "member_coverage_max",
    "profile_source",
    "profile_share",
    "predicted_coverage_term",
)


def class_depth_table(
    decisions: pd.DataFrame,
    *,
    profile: Any | None = None,
    grid: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Return ``resolvability_class_depth.parquet`` (plan §8.3 v7.5).

    One row per (regime, level L, class c, depth bin d) of the ensemble
    decisions, the schema RESOLVE consumes (M4 follow-up): ``status`` and the
    applied ``threshold`` (with ``threshold_source``: ``default``,
    ``resolvability_local``, ``saturated_cap`` or ``monotone_inherited``),
    how the bin was decided (``ensemble_rule``, ``pooled``, ``extrapolated``,
    ``monotone_filled``, ``nonneuronal_high_depth``, ``neuronal``), the
    ensemble's ``precision`` and ``coverage`` there (a filled bin: its fill
    measurement; a pooled bin: its pool's), ``n_test`` and ``n_confident``
    (distinct test cells), and the member minimum and maximum of precision
    and coverage at the applied threshold. With a depth ``profile`` of the
    family's species x chemistry (``sim_inputs.DepthProfile``):
    ``profile_source`` (the class, or the pool it falls back to),
    ``profile_share`` ``s_c(d)`` (the share of the profile's class-c cells in
    bin ``d``) and ``predicted_coverage_term`` ``s_c(d) cov(L, c, d)`` on
    emitted bins (0 elsewhere). RESOLVE sums the same terms with the
    dataset's own ``s_c(d)`` (its cells called ``c``); the predicted coverage
    of (L, c) is ``sum_d s_c(d) cov(L, c, d)`` and its resolvable share
    ``sum_d s_c(d)`` over emitted bins.

    Args:
        decisions: ``ensemble_decide`` decisions.
        profile: Optional ``sim_inputs.DepthProfile``.
        grid: The depth grid (default: the decisions' depths).

    Returns:
        ``CLASS_DEPTH_COLUMNS`` rows.
    """
    if decisions.empty:
        return pd.DataFrame(columns=list(CLASS_DEPTH_COLUMNS))
    frame = decisions.copy()
    filled = _bool_column(frame["monotone_filled"]).to_numpy()
    precision = frame["precision"].to_numpy(np.float64).copy()
    coverage = frame["coverage"].to_numpy(np.float64).copy()
    precision[filled] = frame["fill_precision"].to_numpy(np.float64)[filled]
    coverage[filled] = frame["fill_coverage"].to_numpy(np.float64)[filled]
    frame["precision"] = precision
    frame["coverage"] = coverage
    depths = sorted(
        {int(value) for value in (grid or frame["depth"].astype(int).unique())}
    )
    shares: dict[str, dict[int, float]] = {}
    sources: list[str | None] = []
    profile_share: list[float] = []
    if profile is not None:
        for cls, depth in zip(frame["class"].astype(str), frame["depth"], strict=True):
            source = str(profile.source(cls))
            if source not in shares:
                bins = depth_bin(profile.values(source), depths)
                shares[source] = {
                    value: float(np.mean(bins == float(value))) for value in depths
                }
            sources.append(source)
            profile_share.append(shares[source].get(int(depth), 0.0))
        frame["profile_source"] = sources
        frame["profile_share"] = profile_share
        emitted = (frame["status"] == STATUS_EMITTED).to_numpy()
        values = np.nan_to_num(coverage, nan=0.0)
        frame["predicted_coverage_term"] = np.where(
            emitted, np.asarray(profile_share) * values, 0.0
        )
    else:
        frame["profile_source"] = None
        frame["profile_share"] = np.nan
        frame["predicted_coverage_term"] = np.nan
    table = frame[list(CLASS_DEPTH_COLUMNS)].reset_index(drop=True)
    return table


def profile_prediction(
    class_depth: pd.DataFrame, profile: Any | None
) -> dict[str, Any] | None:
    """Return the summary's ``profile_prediction`` (plan §8.3 v7.5).

    Per regime and level: the resolvable share and the predicted coverage per
    class (sums over the class's bins) and, weighted by the profile's own
    class shares (classes of the profile with their own totals), the level's
    resolvable share and predicted coverage.

    Args:
        class_depth: ``class_depth_table`` output (with a profile).
        profile: The profile (``None``: no prediction).

    Returns:
        The record, or ``None`` without a profile.
    """
    if profile is None or class_depth.empty:
        return None
    n_total = sum(len(values) for values in profile.by_class.values())
    result: dict[str, Any] = {
        "profile": {
            "label": profile.label,
            "asset": profile.asset,
            "sha256": profile.sha256,
            "n_cells": int(profile.n_cells),
        },
        "regimes": {},
    }
    emitted = class_depth["status"] == STATUS_EMITTED
    frame = class_depth.assign(
        _resolvable=np.where(emitted, class_depth["profile_share"], 0.0)
    )
    for regime, regime_rows in frame.groupby("regime", sort=True):
        levels: dict[str, Any] = {}
        for level, rows in regime_rows.groupby("level", sort=True):
            per_class = rows.groupby("class", sort=True).agg(
                resolvable_share=("_resolvable", "sum"),
                predicted_coverage=("predicted_coverage_term", "sum"),
            )
            weights = np.array(
                [
                    len(profile.by_class.get(str(cls), ())) / n_total
                    if n_total
                    else 0.0
                    for cls in per_class.index
                ]
            )
            mass = float(weights.sum())
            levels[str(level)] = {
                "profile_share_covered": round(mass, 6),
                "resolvable_share": None
                if mass <= 0
                else round(
                    float(np.sum(weights * per_class["resolvable_share"]) / mass), 6
                ),
                "predicted_coverage": None
                if mass <= 0
                else round(
                    float(np.sum(weights * per_class["predicted_coverage"]) / mass), 6
                ),
                "classes": {
                    str(cls): {
                        "resolvable_share": round(float(row.resolvable_share), 6),
                        "predicted_coverage": round(float(row.predicted_coverage), 6),
                    }
                    for cls, row in per_class.iterrows()
                },
            }
        result["regimes"][str(regime)] = levels
    return result


def emitted_triples(
    decisions: pd.DataFrame,
    regime: Regime = "provisional",
    *,
    levels: Iterable[str] | None = None,
) -> set[tuple[str, str, int]]:
    """Return the emitted (level, class, depth) triples of a regime.

    Args:
        decisions: ``decide`` or ``ensemble_decide`` decisions.
        regime: The regime.
        levels: Restrict to these levels (default: all).

    Returns:
        The triples.
    """
    frame = decisions[
        (decisions["regime"] == regime) & (decisions["status"] == STATUS_EMITTED)
    ]
    if levels is not None:
        frame = frame[frame["level"].astype(str).isin(list(levels))]
    return {
        (str(level), str(cls), int(depth))
        for level, cls, depth in zip(
            frame["level"], frame["class"], frame["depth"], strict=True
        )
    }


def triple_churn(
    first: set[tuple[str, str, int]], second: set[tuple[str, str, int]]
) -> dict[str, Any]:
    """Return the churn ``(|A \\ B| + |B \\ A|) / |A u B|`` of emitted triples.

    Args:
        first: Triples of A.
        second: Triples of B.

    Returns:
        ``n_first``, ``n_second``, ``lost`` (in A only), ``gained`` (in B
        only), ``union``, ``churn`` (0 for two empty sets) and ``per_level``.
    """
    union = first | second
    lost = first - second
    gained = second - first
    per_level: dict[str, dict[str, Any]] = {}
    for level in sorted({triple[0] for triple in union}):
        a = {triple for triple in first if triple[0] == level}
        b = {triple for triple in second if triple[0] == level}
        both = a | b
        per_level[level] = {
            "n_first": len(a),
            "n_second": len(b),
            "lost": len(a - b),
            "gained": len(b - a),
            "churn": (len(a - b) + len(b - a)) / len(both) if both else 0.0,
        }
    return {
        "n_first": len(first),
        "n_second": len(second),
        "lost": len(lost),
        "gained": len(gained),
        "union": len(union),
        "churn": (len(lost) + len(gained)) / len(union) if union else 0.0,
        "lost_triples": sorted(lost),
        "gained_triples": sorted(gained),
        "per_level": per_level,
    }


def ensemble_summary(
    result: EnsembleDecisions, ensemble: EnsembleSettings
) -> dict[str, Any]:
    """Return the summary's ``ensemble`` record (v7.7-v7.10).

    Per regime: bins, emitted bins by rule (unanimous, spread, filled),
    E1, spread and spread-margin failures, saturated and non-neuronal
    high-depth bins, the
    member spread's distribution where it was evaluated, how often the
    members agree, and each member's emitted count; per level the same
    counts.

    Args:
        result: ``ensemble_decide`` output.
        ensemble: The settings.

    Returns:
        The record.
    """
    decisions = result.decisions
    members = result.member_decisions
    record: dict[str, Any] = {
        "members": list(result.members),
        "settings": ensemble.to_json(),
        "regimes": {},
    }
    for regime in REGIMES:
        frame = decisions[decisions["regime"] == regime]
        emitted = frame["status"] == STATUS_EMITTED
        filled = _bool_column(frame["monotone_filled"])
        reason = frame["reason"].astype(object)
        # The spread of the bins whose E1 passed (where E2 decided).
        tested = (
            frame["ensemble_rule"].notna()
            | frame["ensemble_reason"].isin(REASONS_ENSEMBLE_E2)
        ).to_numpy()
        spread = frame["member_spread"].to_numpy(np.float64)[tested]
        finite = spread[np.isfinite(spread)]
        mine = members[
            (members["regime"] == regime) & (members[MEMBER_ROLE_COLUMN] == "emission")
        ]
        statuses = mine.pivot_table(
            index=["level", "class", "depth"],
            columns=MEMBER_COLUMN,
            values="status",
            aggfunc="first",
        )
        agree = float((statuses.nunique(axis=1) == 1).mean()) if len(statuses) else None

        def counts(rows: pd.DataFrame) -> dict[str, int]:
            is_emitted = rows["status"] == STATUS_EMITTED
            is_filled = _bool_column(rows["monotone_filled"])
            return {
                "n_bins": int(len(rows)),
                "n_emitted": int(is_emitted.sum()),
                "n_unanimous": int((rows["ensemble_rule"] == RULE_UNANIMOUS).sum()),
                "n_spread": int((rows["ensemble_rule"] == RULE_SPREAD).sum()),
                "n_filled": int(is_filled.sum()),
            }

        record["regimes"][regime] = {
            **counts(frame),
            "n_spread_failed": int((reason == REASON_ENSEMBLE_SPREAD).sum()),
            "n_spread_margin_failed": int(
                (reason == REASON_ENSEMBLE_SPREAD_MARGIN).sum()
            ),
            "n_ensemble_e1_failed": int(
                reason.map(
                    lambda value: (
                        isinstance(value, str)
                        and value.startswith(REASON_ENSEMBLE_PREFIX)
                        and value not in REASONS_ENSEMBLE_E2
                    )
                ).sum()
            ),
            "n_saturated_emitted": int(
                (
                    emitted & (frame["threshold_source"] == THRESHOLD_SOURCE_SATURATED)
                ).sum()
            ),
            "n_saturated_filled": int(
                (
                    filled & (frame["threshold_source"] == THRESHOLD_SOURCE_SATURATED)
                ).sum()
            ),
            "n_nonneuronal_high_depth_emitted": int(
                (emitted & _bool_column(frame["nonneuronal_high_depth"])).sum()
            ),
            "member_spread": {
                "n": int(len(finite)),
                "median": float(np.median(finite)) if len(finite) else None,
                "q90": float(np.quantile(finite, 0.9)) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
            },
            "member_status_agreement": agree,
            "member_emitted": {
                str(name): int((rows["status"] == STATUS_EMITTED).sum())
                for name, rows in mine.groupby(MEMBER_COLUMN, sort=False)
            },
            "per_level": {
                str(level): counts(rows)
                for level, rows in frame.groupby("level", sort=False)
            },
        }
    return record


V7_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    *TABLE_COLUMNS,
    MEMBER_COLUMN,
    MEMBER_ROLE_COLUMN,
    *SATURATED_COLUMNS,
    *V7_DECISION_COLUMNS,
)


def resolvability_table_v7(
    cells: pd.DataFrame,
    result: EnsembleDecisions,
    levels: Sequence[LevelMeta],
    settings: RuleSettings,
    *,
    efficiencies: Mapping[str, tuple[Sequence[str], np.ndarray]] | None = None,
    roles: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return a version-7 ``resolvability.parquet`` (plan §8.3 v7.10).

    The version-6 row kinds per member (``bin``, ``curve``, ``isotonic``,
    ``node``, ``confusion``; ``member`` and ``recipe`` set), each member's
    own decisions (``kind = member_decision``), the ensemble's
    (``kind = decision``, ``recipe = ensemble``) with the version-7 columns,
    and each member's gene efficiency (``kind = gene_efficiency``).

    Args:
        cells: The version-7 cells table.
        result: ``ensemble_decide`` output.
        levels: Level metadata.
        settings: Rule settings.
        efficiencies: Per member, the genes and their efficiency.
        roles: Per member, its role (``emission``, ``reported``, ``stress``).

    Returns:
        The table (``V7_TABLE_COLUMNS``).
    """
    recipe_of = dict(
        zip(
            cells[MEMBER_COLUMN].astype(str),
            cells["recipe"].astype(str),
            strict=True,
        )
    )
    by_member = cells.assign(recipe=cells[MEMBER_COLUMN].astype(str))
    empty = pd.DataFrame(columns=list(DECISION_COLUMNS))
    base = resolvability_table(by_member, empty, levels, settings)
    base = base.astype({"recipe": object})
    base[MEMBER_COLUMN] = base["recipe"]
    base["recipe"] = base[MEMBER_COLUMN].map(lambda name: recipe_of.get(str(name)))
    member_rows = result.member_decisions.copy()
    member_rows.insert(0, "kind", "member_decision")
    member_rows["recipe"] = member_rows[MEMBER_COLUMN].map(
        lambda name: recipe_of.get(str(name))
    )
    decision_rows = result.decisions.copy()
    decision_rows.insert(0, "kind", "decision")
    decision_rows["recipe"] = ENSEMBLE_RECIPE
    extra: list[pd.DataFrame] = []
    for name, (genes, values) in (efficiencies or {}).items():
        extra.append(
            pd.DataFrame(
                {
                    "kind": "gene_efficiency",
                    "recipe": recipe_of.get(name),
                    MEMBER_COLUMN: name,
                    "node": list(genes),
                    "x": np.asarray(values, dtype=np.float64),
                }
            )
        )
    frames = [base, member_rows, decision_rows, *extra]
    combined = pd.concat(
        [frame.astype(object) for frame in frames if len(frame)], ignore_index=True
    )
    if roles:
        missing = (
            combined[MEMBER_ROLE_COLUMN].isna()
            if MEMBER_ROLE_COLUMN in combined
            else None
        )
        if missing is not None:
            combined.loc[missing, MEMBER_ROLE_COLUMN] = combined.loc[
                missing, MEMBER_COLUMN
            ].map(lambda name: None if name is None else roles.get(str(name)))
    return _typed_v7_table(combined)


def _typed_v7_table(combined: pd.DataFrame) -> pd.DataFrame:
    for column in V7_TABLE_COLUMNS:
        if column not in combined.columns:
            combined[column] = None
    combined = combined[list(V7_TABLE_COLUMNS)].copy()
    text = (
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
        MEMBER_COLUMN,
        MEMBER_ROLE_COLUMN,
        *V7_TEXT_COLUMNS,
    )
    for column in text:
        combined[column] = pd.Categorical(
            [_clean_label(value) for value in combined[column].astype(object)]
        )
    floats = (
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
        "saturated_share",
        "member_precision_min",
        "member_precision_max",
        "member_coverage_min",
        "member_coverage_max",
        "member_min_n",
        "member_spread",
        "spread_limit",
        "spread_se",
        "wilson_clearance",
        "fill_precision",
        "fill_coverage",
    )
    for column in floats:
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
        "pool_min_depth",
        "n_rows",
    ):
        combined[column] = pd.to_numeric(combined[column], errors="coerce").astype(
            "Int64"
        )
    for column in ("extrapolated", "pooled", "neuronal", *V7_BOOL_COLUMNS):
        combined[column] = pd.array(
            [
                None
                if value is None
                or (isinstance(value, float) and math.isnan(value))
                or value is pd.NA
                else bool(value)
                for value in combined[column]
            ],
            dtype="boolean",
        )
    return combined


def coerce_cells_v7(cells: pd.DataFrame) -> pd.DataFrame:
    """Return a version-7 cells table with parquet-friendly dtypes.

    ``coerce_cells`` plus the ``member`` and ``member_role`` columns as
    categories.
    """
    frame = coerce_cells(cells)
    for column in (MEMBER_COLUMN, MEMBER_ROLE_COLUMN):
        if column in frame.columns:
            frame[column] = frame[column].astype(str).astype("category")
    return frame


# --------------------------------------------------------------------------
# Version 7: orchestration (PREP and the diagnostic of version-6 families)


@dataclass(frozen=True)
class _QueryStub:
    """What ``build_summary`` reads of a simulated query (no counts kept)."""

    obs: pd.DataFrame
    n_by_depth: dict[int, int]


@dataclass
class EnsembleSimulation:
    """The mapped members of one version-7 self-map (no counts kept).

    Attributes:
        cells: The cells table of every member (``member``, ``member_role``).
        n_simulated: Simulated cells per member.
        n_by_depth: Test cells simulated per depth, per member.
        efficiencies: Per member, the genes and their efficiency.
        timings: Seconds per step.
    """

    cells: pd.DataFrame
    n_simulated: dict[str, int]
    n_by_depth: dict[str, dict[int, int]]
    efficiencies: dict[str, tuple[list[str], np.ndarray]]
    timings: dict[str, float]


def member_tag(member: EnsembleMember) -> str:
    """Return a file-name-safe tag of a member (``R1_contam_HO_seed0``)."""
    return f"{member.recipe.name}_seed{int(member.recipe.seed)}"


def simulate_members(
    test: HeldOutCells,
    *,
    specs: Sequence[LevelSpec],
    depths: Sequence[int],
    members: Sequence[EnsembleMember],
    map_fn: MapFunction,
    cells_rules: Sequence[CellsRule] = (),
    registry: Mapping[str, Any] | None = None,
    mapping_seed: int = 0,
    tag_prefix: str = "",
) -> EnsembleSimulation:
    """Simulate, map and tabulate each member in turn (v7.2-v7.4).

    One member's simulated counts and MapMyCells table are held at a time
    and dropped once its calls are tabulated (``as_stored``), so the parent
    process keeps only the cells tables (the phase-1 prototype's largest
    process reached 13.2 GB by holding every draw).

    Args:
        test: The test cells.
        specs: Levels to read.
        depths: The version-7 grid.
        members: Every member (emission, reported, stress).
        map_fn: ``(query, tag, seed) -> MMC tidy table``.
        cells_rules: Production rules applied to each member's cells.
        registry: Simulation-input registry (table recipes).
        mapping_seed: Mapping seed (0; 1 for the fine-level check).
        tag_prefix: Prefix of the mapping tags (diagnostics).

    Returns:
        The simulation.
    """
    frames: list[pd.DataFrame] = []
    n_simulated: dict[str, int] = {}
    n_by_depth: dict[str, dict[int, int]] = {}
    efficiencies: dict[str, tuple[list[str], np.ndarray]] = {}
    timings: dict[str, float] = {}
    for member in members:
        step = time.monotonic()
        efficiency = member_efficiency(member.recipe, test.genes, registry=registry)
        query = thin_and_contaminate_v7(
            test, depths, member.recipe, efficiency=efficiency
        )
        timings[f"simulate_{member.name}"] = round(time.monotonic() - step, 3)
        n_simulated[member.name] = int(len(query.obs))
        n_by_depth[member.name] = dict(query.n_by_depth)
        efficiencies[member.name] = (list(test.genes), efficiency)
        logger.info(
            "resolvability v7 %s: simulated %d cells from %d test cells in %.1f s",
            member.name,
            len(query.obs),
            len(test.obs),
            timings[f"simulate_{member.name}"],
        )
        step = time.monotonic()
        tidy = map_fn(query, f"{tag_prefix}{member_tag(member)}", mapping_seed)
        timings[f"map_{member.name}"] = round(time.monotonic() - step, 3)
        frame = level_cells(tidy, query, test, specs, seed=mapping_seed)
        del tidy, query
        for rule in cells_rules:
            frame = rule(frame)
        frame = as_stored(frame)
        frame[MEMBER_COLUMN] = member.name
        frame[MEMBER_ROLE_COLUMN] = member.role
        frames.append(frame)
        logger.info(
            "resolvability v7 %s: mapped in %.1f s",
            member.name,
            timings[f"map_{member.name}"],
        )
    cells = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=[*CELLS_COLUMNS, MEMBER_COLUMN, MEMBER_ROLE_COLUMN])
    )
    return EnsembleSimulation(
        cells=cells,
        n_simulated=n_simulated,
        n_by_depth=n_by_depth,
        efficiencies=efficiencies,
        timings=timings,
    )


def neuronal_classes(
    cells: pd.DataFrame, species: str, neuronal: NeuronalOf | None = None
) -> dict[str, bool | None]:
    """Return whether each class of a cells table is neuronal (v7.9).

    Args:
        cells: A cells table (its ``parent`` and ``truth_parent`` classes).
        species: ``human`` or ``mouse`` (``sim_inputs.is_neuronal_class``).
        neuronal: An explicit mapping or function (overrides the species).

    Returns:
        Class -> ``True`` / ``False`` / ``None`` (unknown).
    """
    from merxen.annotation import sim_inputs as si

    names = sorted(
        {str(value) for value in cells["parent"].dropna().astype(str)}
        | {str(value) for value in cells["truth_parent"].dropna().astype(str)}
    )
    if neuronal is not None:
        return {name: _neuronal(neuronal, name) for name in names}
    return {name: si.is_neuronal_class(name, species) for name in names}


@dataclass
class ResolvabilityResultV7(ResolvabilityResult):
    """Everything one version-7 self-map produced (plan §8.3 v7.10).

    Attributes:
        member_decisions: Each member's own decisions.
        class_depth: ``resolvability_class_depth.parquet`` content.
    """

    member_decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    class_depth: pd.DataFrame = field(default_factory=pd.DataFrame)

    def write(self, directory: Path) -> dict[str, str]:
        """Write the four version-7 resolvability files.

        Args:
            directory: The bundle work directory (or a diagnostic directory).

        Returns:
            File name per output.
        """
        directory.mkdir(parents=True, exist_ok=True)
        coerce_cells_v7(self.cells).to_parquet(
            directory / RESOLVABILITY_CELLS_FILE, index=False
        )
        self.table.to_parquet(directory / RESOLVABILITY_FILE, index=False)
        self.class_depth.to_parquet(directory / CLASS_DEPTH_FILE, index=False)
        (directory / RESOLVABILITY_SUMMARY_FILE).write_text(
            json.dumps(_json_native(self.summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return {
            "table": RESOLVABILITY_FILE,
            "cells": RESOLVABILITY_CELLS_FILE,
            "summary": RESOLVABILITY_SUMMARY_FILE,
            "class_depth": CLASS_DEPTH_FILE,
        }

    def bundle_output(self) -> dict[str, Any]:
        """Return the compact record ``bundle.json`` keeps (version 7)."""
        record = super().bundle_output()
        record["files"]["class_depth"] = CLASS_DEPTH_FILE
        record["resolvability_version"] = RESOLVABILITY_VERSION_V7
        record["decision_recipe"] = ENSEMBLE_RECIPE
        record["members"] = self.summary["members"]
        record["ensemble"] = {
            regime: {
                key: value
                for key, value in stats.items()
                if key
                in (
                    "n_bins",
                    "n_emitted",
                    "n_unanimous",
                    "n_spread",
                    "n_filled",
                    "n_spread_failed",
                )
            }
            for regime, stats in self.summary["ensemble"]["regimes"].items()
        }
        return record


def build_summary_v7(
    *,
    test: HeldOutCells,
    levels: Sequence[LevelMeta],
    depths: Sequence[int],
    members: Sequence[EnsembleMember],
    settings: RuleSettings,
    ensemble: EnsembleSettings,
    result: EnsembleDecisions,
    floors: pd.DataFrame,
    trust: TrustConstraint,
    simulation: EnsembleSimulation,
    stability: Mapping[str, float],
    species: str,
    neuronal: Mapping[str, bool | None],
    class_depth: pd.DataFrame,
    profile: Any | None,
    runtime: Mapping[str, float],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a version-7 ``resolvability_summary.json`` (plan §8.3 v7.10).

    The version-6 summary of the ensemble decisions (``emission``, ``emitted``,
    pooled sets, floors, trust from the decisions before the fill), plus the
    version, the members, the ensemble settings and statistics (member
    spread and agreement), the emitted bins before the fill, each class's
    lineage, the class-depth file and the profile prediction; ``provenance``
    adds the assets, chemistry, top-up and engine records.
    """
    emission_members = [member for member in members if member.role == "emission"]
    first = emission_members[0]
    stub = _QueryStub(
        obs=pd.DataFrame(index=pd.RangeIndex(simulation.n_simulated[first.name])),
        n_by_depth=simulation.n_by_depth[first.name],
    )
    summary = build_summary(
        test=test,
        levels=levels,
        depths=depths,
        recipes=[first.recipe],
        settings=settings,
        decisions=result.decisions,
        floors=floors,
        trust=trust,
        queries={first.recipe.name: stub},  # type: ignore[dict-item]
        stability=stability,
        species=species,
        runtime=runtime,
        provenance={},
    )
    decisions = result.decisions
    for record in decisions.to_dict("records"):
        entry = summary["emission"][str(record["regime"])][str(record["level"])][
            str(record["class"])
        ][str(int(record["depth"]))]
        entry.update(
            {
                "threshold_source": record["threshold_source"],
                "ensemble_rule": record["ensemble_rule"],
                "monotone_filled": bool(record["monotone_filled"]),
                "nonneuronal_high_depth": bool(record["nonneuronal_high_depth"]),
                "saturated_bp": bool(record["saturated_bp"]),
                "member_spread": _optional_float(record["member_spread"]),
                "spread_limit": _optional_float(record["spread_limit"]),
                "spread_se": _optional_float(record["spread_se"]),
                "wilson_clearance": _optional_float(record["wilson_clearance"]),
            }
        )
    unfilled = unfilled_decisions(decisions)
    summary.update(
        {
            "resolvability_version": RESOLVABILITY_VERSION_V7,
            "decision_recipe": ENSEMBLE_RECIPE,
            "recipes": [member.recipe.to_json() for member in members],
            "members": [member.to_json() for member in members],
            "emission_members": [member.name for member in emission_members],
            "n_simulated_cells": dict(simulation.n_simulated),
            "n_by_depth_per_member": {
                name: {str(depth): n for depth, n in values.items()}
                for name, values in simulation.n_by_depth.items()
            },
            "ensemble_settings": ensemble.to_json(),
            "ensemble": ensemble_summary(result, ensemble),
            "emitted_before_fill": {
                regime: emitted_summary(unfilled, regime) for regime in REGIMES
            },
            "neuronal_classes": dict(neuronal),
            "class_depth_file": CLASS_DEPTH_FILE,
            "profile_prediction": profile_prediction(class_depth, profile),
        }
    )
    summary.update(dict(provenance))
    return summary


TRUST_GUARD_VERSION: Final = 1
_TRUST_SEVERITY: Final[dict[str | None, int]] = {"refused": 0, "broad_only": 1, None: 2}


def member_uses_assets(member: EnsembleMember) -> bool:
    """Whether a member's efficiency comes from a simulation-input asset.

    ``R3_measured_HO`` (a ``member`` factor table) and the stress recipe
    ``R1_xtissue_lung_stress`` (a ``stress`` ratio table) do; the R1 and
    clean recipes (a keyed LogNormal draw, or none) do not.
    """
    return member.recipe.efficiency_table is not None


def lower_trust_constraint(
    first: TrustConstraint, second: TrustConstraint
) -> TrustConstraint:
    """Return the more severe of two trust constraints (refused < broad_only < none).

    Ties keep ``first`` (its reasons and statistics).
    """
    if _TRUST_SEVERITY[second.state] < _TRUST_SEVERITY[first.state]:
        return second
    return first


def asset_free_trust(
    cells: pd.DataFrame,
    levels: Sequence[LevelMeta],
    depths: Sequence[int],
    settings: RuleSettings,
    ensemble: EnsembleSettings,
    *,
    members: Sequence[EnsembleMember],
    neuronal: Mapping[str, bool | None],
    trust: TrustConstraint,
) -> tuple[TrustConstraint, dict[str, Any]]:
    """Guard the trust constraint against simulation-input assets (v7; §14 (v)).

    Simulation inputs never promote trust (OD-E1 as amended, user decision 1).
    When an emission member uses an asset (``R3_measured_HO``), the ensemble
    of the asset-free emission members (the R1 draws) is decided on the same
    cells and the trust constraint is the more severe of the two, so that for
    every set of assets the trust state is at most its state without any
    asset: an asset can lower the constraint (as real data can), never raise
    it. Emission and floors keep the full ensemble.

    Args:
        cells: The cells table of every member.
        levels: Level metadata.
        depths: The depth grid.
        settings: Rule settings.
        ensemble: Ensemble settings.
        members: Every member.
        neuronal: Class lineage.
        trust: The full ensemble's trust constraint.

    Returns:
        ``(guarded constraint, summary record)``.
    """
    emission = [member for member in members if member.role == "emission"]
    with_assets = [member.name for member in emission if member_uses_assets(member)]
    free = [member.name for member in emission if not member_uses_assets(member)]
    record: dict[str, Any] = {
        "version": TRUST_GUARD_VERSION,
        "rule": (
            "the more severe of the full ensemble's constraint and the "
            "asset-free emission members' ensemble constraint"
        ),
        "asset_members": with_assets,
        "asset_free_members": free,
        "ensemble_state": trust.state,
        "applied": False,
        "asset_free_state": trust.state,
        "state": trust.state,
    }
    if not with_assets:
        record["note"] = "no emission member uses a simulation-input asset"
        return trust, record
    if not free:
        # Every emission member uses an asset: nothing is free of them, so
        # the family keeps no resolvability-based trust beyond broad-only.
        guarded = TrustConstraint(
            state="broad_only",
            reasons=["every emission member uses a simulation-input asset"],
            broad_emitted_bins=trust.broad_emitted_bins,
            leaf_share_by_depth=dict(trust.leaf_share_by_depth),
            leaf_classes=list(trust.leaf_classes),
        )
        guarded = lower_trust_constraint(trust, guarded)
        record.update(
            {"applied": True, "asset_free_state": None, "state": guarded.state}
        )
        return guarded, record
    rows = cells[cells[MEMBER_COLUMN].astype(str).isin(free)]
    decided = ensemble_decide(
        rows,
        levels,
        depths,
        settings,
        ensemble,
        members=free,
        reported=[],
        neuronal=neuronal,
    )
    free_trust = trust_constraint(decided.unfilled(), levels, settings)
    guarded = lower_trust_constraint(trust, free_trust)
    record.update(
        {
            "applied": True,
            "asset_free_state": free_trust.state,
            "asset_free_reasons": list(free_trust.reasons),
            "state": guarded.state,
        }
    )
    return guarded, record


def trust_and_floors(
    result: EnsembleDecisions,
    levels: Sequence[LevelMeta],
    settings: RuleSettings,
    *,
    floor_table: pd.DataFrame | None,
    species: str,
) -> tuple[TrustConstraint, pd.DataFrame]:
    """Return the trust constraint and floors of ensemble decisions (v7.7, v7.9).

    Both come from the decisions before the monotone fill (filled bins take
    no part in the trust tests or the floors), exactly as version 6 derives
    them from its decision recipe.

    Args:
        result: ``ensemble_decide`` output.
        levels: Level metadata.
        settings: Rule settings.
        floor_table: Packaged floors.
        species: ``human`` or ``mouse``.

    Returns:
        ``(trust constraint, combined floors)``.
    """
    unfilled = result.unfilled()
    floors = combined_floors(
        unfilled, levels, settings, floor_table=floor_table, species=species
    )
    return trust_constraint(unfilled, levels, settings), floors


def decide_v7(
    cells: pd.DataFrame,
    levels: Sequence[LevelMeta],
    depths: Sequence[int],
    settings: RuleSettings,
    ensemble: EnsembleSettings,
    *,
    members: Sequence[EnsembleMember],
    neuronal: Mapping[str, bool | None],
) -> EnsembleDecisions:
    """Return the ensemble decisions of a cells table (emission and reported)."""
    return ensemble_decide(
        cells,
        levels,
        depths,
        settings,
        ensemble,
        members=[member.name for member in members if member.role == "emission"],
        reported=[member.name for member in members if member.role != "emission"],
        neuronal=neuronal,
    )


def run_resolvability_v7(
    test: HeldOutCells,
    *,
    specs: Sequence[LevelSpec],
    depths: Sequence[int],
    members: Sequence[EnsembleMember],
    map_fn: MapFunction,
    settings: RuleSettings,
    ensemble: EnsembleSettings,
    species: str,
    floor_table: pd.DataFrame | None = None,
    fine_seed_check: bool = False,
    provenance: Mapping[str, Any] | None = None,
    cells_rules: Sequence[CellsRule] = (),
    registry: Mapping[str, Any] | None = None,
    profile: Any | None = None,
    neuronal: NeuronalOf | None = None,
) -> ResolvabilityResultV7:
    """Run a version-7 self-map: members, ensemble, fill, tables (§8.3 v7).

    Args:
        test: The test cells (topped up, v7.6).
        specs: Levels to evaluate.
        depths: The version-7 grid.
        members: Every member (``ensemble_members``).
        map_fn: ``(query, tag, seed) -> MMC tidy table``.
        settings: Rule settings.
        ensemble: Ensemble settings.
        species: ``human`` or ``mouse``.
        floor_table: Packaged floors for the max rule.
        fine_seed_check: Re-map the first emission member with seed 1 and
            record the fine levels' seed stability.
        provenance: Extra summary fields (assets, chemistry, top-up, engine).
        cells_rules: Production rules applied to each member's cells.
        registry: Simulation-input registry.
        profile: Optional ``sim_inputs.DepthProfile`` of the family's species
            x chemistry (class-depth predictions).
        neuronal: Class lineage override (default: the species vocabularies).

    Returns:
        The result.

    Raises:
        ResolvabilityError: Without an emission member.
    """
    emission = [member for member in members if member.role == "emission"]
    if not emission:
        raise ResolvabilityError("run_resolvability_v7 needs an emission member")
    started = time.monotonic()
    simulation = simulate_members(
        test,
        specs=specs,
        depths=depths,
        members=members,
        map_fn=map_fn,
        cells_rules=cells_rules,
        registry=registry,
    )
    timings = dict(simulation.timings)
    cells = simulation.cells
    levels = [spec.meta for spec in specs]
    lineage = neuronal_classes(cells, species, neuronal)
    step = time.monotonic()
    result = decide_v7(
        cells, levels, depths, settings, ensemble, members=members, neuronal=lineage
    )
    timings["decide"] = round(time.monotonic() - step, 3)
    fine_levels = [spec.meta.level for spec in specs if spec.meta.role == "fine"]
    stability: dict[str, float] = {}
    if fine_seed_check and fine_levels:
        step = time.monotonic()
        second = simulate_members(
            test,
            specs=[spec for spec in specs if spec.meta.role == "fine"],
            depths=depths,
            members=[emission[0]],
            map_fn=map_fn,
            cells_rules=cells_rules,
            registry=registry,
            mapping_seed=1,
            tag_prefix="seed1_",
        ).cells
        timings["map_seed1"] = round(time.monotonic() - step, 3)
        first_rows = cells[
            (cells[MEMBER_COLUMN].astype(str) == emission[0].name)
            & (cells["seed"] == 0)
        ]
        for level in fine_levels:
            stability[level] = seed_stability(
                first_rows, second, result.decisions, level
            )
        cells = pd.concat([cells, second], ignore_index=True)
    trust, floors = trust_and_floors(
        result, levels, settings, floor_table=floor_table, species=species
    )
    step = time.monotonic()
    trust, trust_guard = asset_free_trust(
        cells[cells["seed"] == 0] if "seed" in cells.columns else cells,
        levels,
        depths,
        settings,
        ensemble,
        members=members,
        neuronal=lineage,
        trust=trust,
    )
    timings["trust_guard"] = round(time.monotonic() - step, 3)
    logger.info(
        "resolvability v7 trust constraint: %s%s",
        trust.state or "none",
        f" ({'; '.join(trust.reasons)})" if trust.reasons else "",
    )
    step = time.monotonic()
    table = resolvability_table_v7(
        cells,
        result,
        levels,
        settings,
        efficiencies=simulation.efficiencies,
        roles={member.name: member.role for member in members},
    )
    class_depth = class_depth_table(result.decisions, profile=profile, grid=depths)
    timings["tabulate"] = round(time.monotonic() - step, 3)
    summary = build_summary_v7(
        test=test,
        levels=levels,
        depths=depths,
        members=members,
        settings=settings,
        ensemble=ensemble,
        result=result,
        floors=floors,
        trust=trust,
        simulation=simulation,
        stability=stability,
        species=species,
        neuronal=lineage,
        class_depth=class_depth,
        profile=profile,
        runtime={**timings, "total": round(time.monotonic() - started, 3)},
        provenance=provenance or {},
    )
    summary["trust_asset_guard"] = trust_guard
    for regime, stats in summary["ensemble"]["regimes"].items():
        logger.info(
            "resolvability v7 %s: %d of %d bins emitted (%d unanimous, %d by the "
            "spread, %d filled; %d spread failures)",
            regime,
            stats["n_emitted"],
            stats["n_bins"],
            stats["n_unanimous"],
            stats["n_spread"],
            stats["n_filled"],
            stats["n_spread_failed"],
        )
    return ResolvabilityResultV7(
        cells=cells,
        table=table,
        decisions=result.decisions,
        floors=floors,
        trust=trust,
        summary=summary,
        member_decisions=result.member_decisions,
        class_depth=class_depth,
    )


def v7_diagnostic_comparison(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    regimes: Sequence[Regime] = ("validated", "provisional"),
) -> dict[str, Any]:
    """Compare a bundle's decisions with version-7 decisions (diagnostic only).

    Used for the version-6 families (plan §8.3 v7.1; pre-registration §14
    (vii): version 6 vs version 7 emitted triples lost and gained per level)
    and for fresh ensemble draws (§14 (iii)). Nothing is applied.

    Args:
        reference: The bundle's stored decisions (version 6 or 7).
        candidate: The version-7 decisions computed beside them.
        regimes: Regimes to compare.

    Returns:
        Per regime: ``triple_churn(reference, candidate)``.
    """
    return {
        regime: triple_churn(
            emitted_triples(reference, regime), emitted_triples(candidate, regime)
        )
        for regime in regimes
    }


# --------------------------------------------------------------------------
# Version 7: test-set top-up (v7.6; user decision 5)

TOP_UP_VERSION: Final = 1
TOP_UP_STREAM: Final = "class_top_up"
# Pre-registration §14.3: 200 = 4 x min_confident_n, only tightenable.
TOP_UP_MIN_CLASS_TEST_CELLS: Final = 200


def water_fill(available: Mapping[str, int], need: int) -> dict[str, int]:
    """Split ``need`` cells over strata as equally as their sizes allow.

    Every stratum gets ``min(available, t)`` for the largest level ``t``
    that fits; the cells left over go one each to the strata with more
    available, in name order (deterministic).

    Args:
        available: Candidate cells per stratum.
        need: Cells to take.

    Returns:
        Cells to take per stratum (every stratum listed).
    """
    names = sorted(str(name) for name in available)
    sizes = {name: max(0, int(available[name])) for name in names}
    if need <= 0:
        return {name: 0 for name in names}
    if sum(sizes.values()) <= need:
        return dict(sizes)
    low, high = 0, max(sizes.values())
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(size, middle) for size in sizes.values()) <= need:
            low = middle
        else:
            high = middle - 1
    take = {name: min(size, low) for name, size in sizes.items()}
    rest = need - sum(take.values())
    for name in names:
        if rest <= 0:
            break
        if sizes[name] > take[name]:
            take[name] += 1
            rest -= 1
    return take


def _top_up_order(seed: int, cell_ids: Sequence[str]) -> np.ndarray:
    """Return candidate positions ranked by their keyed draw (lowest first)."""
    keys = [draw_key(int(seed), TOP_UP_STREAM, str(cell)) for cell in cell_ids]
    return np.array(sorted(range(len(keys)), key=lambda index: keys[index]), dtype=int)


def class_top_up(
    candidates: pd.DataFrame,
    *,
    class_column: str,
    leaf_column: str,
    have: Mapping[str, int],
    target: int = TOP_UP_MIN_CLASS_TEST_CELLS,
    seed: int = 0,
    pool_column: str | None = None,
    pool_order: Sequence[str] | None = None,
    cluster_column: str | None = None,
    cluster_cap: Mapping[str, int] | None = None,
    pool_classes: Mapping[str, Iterable[str]] | None = None,
) -> tuple[pd.Index, dict[str, Any]]:
    """Top every class with fewer than ``target`` test cells up to it (v7.6).

    Before anything is mapped, a class of the leaf's parent level (mouse WMB
    class; human leaf-level class) holding fewer than ``target`` test cells
    gains up to ``target - have`` candidates, stratified by leaf type
    (``water_fill``), drawn pool by pool in ``pool_order`` (human: the
    held-out donor's unused cells, then the other-region pool; mouse: one
    pool). Within a leaf the candidates with the lowest keyed draw
    (``draw_key(seed, "class_top_up", cell id)``) are taken, so the draw
    does not depend on the candidates' order. ``cluster_cap`` bounds the
    cells taken from one cluster (mouse: 5% of its 10Xv3 cells, the gate-P
    bound); ``pool_classes`` restricts a pool to some classes (human: the
    other-region pool serves non-neuronal classes only). The caller passes
    only admissible candidates (not test cells, not marker-training cells,
    training clusters only, never another frontal donor). No mapping result
    enters: the rule is a count fixed before mapping.

    Args:
        candidates: Admissible candidates (index = cell id).
        class_column: Their class.
        leaf_column: Their leaf type (the stratum).
        have: Test cells per class before the top-up (classes with none are
            not topped up).
        target: The class minimum (200).
        seed: The test-set seed.
        pool_column: Column naming each candidate's pool (``None``: one).
        pool_order: Pools in priority order (default: sorted names).
        cluster_column: Column naming each candidate's cluster.
        cluster_cap: Largest number of cells taken per cluster.
        pool_classes: Classes a pool may serve (default: every class).

    Returns:
        ``(chosen ids in id order, record)``: the record holds, per class,
        the test cells before and after, what each pool held and gave,
        whether the pools ran out, and the rule.
    """
    frame = candidates.copy()
    frame.index = frame.index.astype(str)
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    needs = {
        str(cls): int(target) - int(count)
        for cls, count in have.items()
        if 0 < int(count) < int(target)
    }
    frame = frame[frame[class_column].astype(str).isin(list(needs))]
    pools = (
        frame[pool_column].astype(str)
        if pool_column is not None
        else pd.Series("pool", index=frame.index)
    )
    order = list(pool_order) if pool_order is not None else sorted(set(pools))
    ranked = frame.iloc[_top_up_order(seed, list(frame.index))] if len(frame) else frame
    capped_out = 0
    if cluster_column is not None and cluster_cap is not None and len(ranked):
        clusters = ranked[cluster_column].astype(str)
        rank = clusters.groupby(clusters, sort=False).cumcount().to_numpy()
        caps = clusters.map(lambda name: int(cluster_cap.get(name, 0))).to_numpy()
        within = rank < caps
        capped_out = int((~within).sum())
        ranked = ranked[within]
    chosen: list[str] = []
    per_class: dict[str, Any] = {}
    for cls in sorted(needs):
        need = needs[cls]
        rows = ranked[ranked[class_column].astype(str) == cls]
        record: dict[str, Any] = {
            "before": int(have[cls]),
            "need": int(need),
            "pools": {},
        }
        for pool in order:
            allowed = pool_classes.get(pool) if pool_classes is not None else None
            if allowed is not None and cls not in {str(item) for item in allowed}:
                record["pools"][pool] = {"available": 0, "taken": 0, "serves": False}
                continue
            in_pool = rows[pools.reindex(rows.index).to_numpy() == pool]
            leaves = in_pool[leaf_column].astype(str)
            available = leaves.value_counts().to_dict()
            take = water_fill(available, need)
            picked = [
                cell
                for leaf, count in take.items()
                if count > 0
                for cell in in_pool.index[(leaves == leaf).to_numpy()][:count]
            ]
            chosen.extend(picked)
            need -= len(picked)
            record["pools"][pool] = {
                "available": int(len(in_pool)),
                "taken": int(len(picked)),
                "per_leaf": {leaf: int(count) for leaf, count in take.items() if count},
            }
        record["after"] = int(have[cls]) + int(needs[cls] - need)
        record["ran_out"] = bool(need > 0)
        per_class[cls] = record
    selected = pd.Index(sorted(set(chosen)))
    return selected, {
        "version": TOP_UP_VERSION,
        "rule": (
            f"every class with fewer than {int(target)} test cells is topped up to "
            f"{int(target)} where its pool allows, stratified by leaf type "
            "(water filling), lowest keyed draw first, before any mapping"
        ),
        "target": int(target),
        "seed": int(seed),
        "pool_order": order,
        "n_candidates": int(len(frame)),
        "n_capped_by_cluster": capped_out,
        "n_cells": int(len(selected)),
        "per_class": per_class,
    }
