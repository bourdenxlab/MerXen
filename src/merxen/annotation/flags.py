"""Per-cell report flags of RESOLVE (plan §4.3, §5.6; human v1, mouse hooks).

Every flag is report-only (a DE covariate or a sensitivity), never a status.
Each has a continuous companion and a realised rate per (class, platform)
stratum; a stratum whose realised rate is too high is marked uninformative
and its flag is null (H16 transparency):

- **Contamination** (§5.6): ``contamination_score`` = counts on the assigned
  broad class's negative genes / total counts; ``neg_counts`` the numerator.
  Negative genes are those detected in < 1% of the class's cells in **both**
  WHB frontal and SEA-AD Multiregion, minus ``state_genes_human.csv``
  (``NegativeGeneSet.from_table`` re-derives them from the bundle's
  per-reference detection columns). The null is **dataset-empirical**: per
  (class, platform) a beta-binomial fitted by maximum likelihood to the
  negative counts of the class's confident cells in its top depth quartile
  (>= 30 cells, else no null and the stratum is uninformative); a cell is
  flagged at upper-tail p < 0.01 with >= 3 negative counts. It finds cells
  more contaminated than typical deep cells of their class in the same
  dataset, not an absolute false-positive rate (a reference null fired on
  17-75% of cells). A stratum whose realised rate over its confident broad
  calls exceeds 15% is uninformative (``flag_contaminated`` null).
- **Diffuse profile** (§5.6): ``expected_genes_q95`` = the 95th percentile of
  distinct genes in 200 multinomial draws at the cell's query depth from the
  class profile (the bundle's ``profiles.parquet``, cell-weighted over the
  class's superclusters; a geometric grid of depths, interpolated);
  ``flag_diffuse_profile`` = more distinct genes than that. Native cells
  carry 30-45% more genes than thinned ones (E2), so high rates are
  expected: a stratum above 30% is uninformative (§4.3, §3.7
  ``diffuse_rate_uninformative_above``). H16's 15% marking is reported
  beside it (``informative_h16``), not applied (M3 open question, resolved
  here: the per-cell switch follows §4.3 and the config; H16 is reporting).
- **OOD** (§4.3): ``ood_z`` = robust z (median / 1.4826 MAD) of the primary
  leaf's ``avg_correlation`` within class x platform x depth bin (strata of
  >= 30 cells with a positive MAD); ``flag_ood`` = z < -3 [L]. No null
  switch is registered for it; its rates are reported with the H16 marking.
- **Microglial spill-over** (§5.6, §8.6): human off (OD-C5, decided), so the
  columns are null with a recorded reason; mouse (M6) takes the result of
  ``mouse_flags.microglial_spillover`` (E3's specific-gene test with the
  panel's derived genes, the astrocyte false-positive check and the
  held-out check). Its realised rate per class is reported with the H16
  marking; the flag's own switches are the gene minimum and the FPR check.
- **Astro-Epen low count** (§7.4, mouse): ``astro_lowcount_flags``.
- **Region coherence F1** (E7 §3, mouse): ``mouse_flags.region_incoherent``
  (``flag_region_incoherent``, ``region_coherence``); null without
  coordinates.
- ``discovery_caution`` = any of contaminated, diffuse, OOD, method
  disagreement, microglial spill-over or Astro-Epen low count (null flags
  count as false); ``exclude_hard`` comes from the consensus (§4.3).

The module needs numpy and pandas; scipy is imported inside the functions
that fit or test the beta-binomial.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from merxen.annotation.schema import Columns, safe_token

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.config import AnnotationFlagsConfig
    from merxen.annotation.mouse_flags import CoherenceResult, SpilloverResult
    from merxen.annotation.provenance import FlagProvenance

logger = logging.getLogger(__name__)


def _object_array(values: Sequence[object] | np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    missing = pd.isna(array)
    if missing.any():
        array = array.copy()
        array[missing] = None
    return array


def _masked_objects(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return ``values`` as objects with ``None`` where ``mask`` is false."""
    output = np.asarray(values, dtype=object).copy()
    output[~np.asarray(mask, dtype=bool)] = None
    return output


# --------------------------------------------------------------------------
# Primitives (the M3 prototype, ``scripts/acceptance/shadow_flags.py``)

CONTAMINATION_ALPHA: Final = 0.01
CONTAMINATION_MIN_NEGATIVE: Final = 3
CONTAMINATION_NULL_QUANTILE: Final = 0.75
CONTAMINATION_MIN_NULL_CELLS: Final = 30
CONTAMINATION_MAX_INFORMATIVE: Final = 0.15
DIFFUSE_QUANTILE: Final = 0.95
DIFFUSE_N_SIMULATIONS: Final = 200
DIFFUSE_MAX_INFORMATIVE: Final = 0.30


def negative_gene_mask(
    negatives: pd.DataFrame,
    gene_ids: Sequence[str],
    classes: Sequence[str],
) -> np.ndarray:
    """Return the negative genes of each class as a boolean matrix.

    Args:
        negatives: ``negative_genes.parquet`` (``broad_class``, ``gene_id``,
            ``negative``).
        gene_ids: Query genes (columns).
        classes: Broad classes (rows).

    Returns:
        ``classes x genes`` mask.
    """
    column = {str(gene): index for index, gene in enumerate(gene_ids)}
    row = {str(cls): index for index, cls in enumerate(classes)}
    mask = np.zeros((len(classes), len(gene_ids)), dtype=bool)
    chosen = negatives[negatives["negative"].astype(bool)]
    for record in chosen.itertuples(index=False):
        cls, gene = str(record.broad_class), str(record.gene_id)
        if cls in row and gene in column:
            mask[row[cls], column[gene]] = True
    return mask


def negative_counts(
    counts: sparse.spmatrix | np.ndarray,
    labels: Sequence[object] | np.ndarray,
    mask: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Return each cell's counts on its class's negative genes.

    Args:
        counts: Cells x query genes.
        labels: Broad class per cell.
        mask: ``negative_gene_mask`` output.
        classes: Its rows.

    Returns:
        Counts per cell; NaN for cells outside ``classes``.
    """
    from scipy import sparse as sp

    matrix = sp.csr_matrix(counts, dtype=np.float64)
    per_class = np.asarray(matrix @ mask.T.astype(np.float64))
    names = _object_array(labels)
    index = {cls: position for position, cls in enumerate(classes)}
    rows = np.array(
        [index.get(str(name), -1) if name is not None else -1 for name in names]
    )
    values = np.full(len(names), math.nan)
    good = rows >= 0
    values[good] = per_class[np.flatnonzero(good), rows[good]]
    return values


@dataclass(frozen=True)
class BetaBinomialFit:
    """A beta-binomial null of negative-gene counts.

    Attributes:
        alpha: First shape parameter.
        beta: Second shape parameter.
        n_cells: Cells fitted.
        mean_rate: Pooled negative rate of those cells.
    """

    alpha: float
    beta: float
    n_cells: int
    mean_rate: float


def fit_beta_binomial(successes: np.ndarray, trials: np.ndarray) -> BetaBinomialFit:
    """Fit a beta-binomial by maximum likelihood (method-of-moments start).

    Args:
        successes: Negative counts per cell.
        trials: Total counts per cell.

    Returns:
        The fit.

    Raises:
        ValueError: If there is no cell or no trial.
    """
    from scipy.optimize import minimize
    from scipy.stats import betabinom

    k = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    if k.size == 0 or n.sum() <= 0:
        raise ValueError("a beta-binomial fit needs cells with counts")
    mean = float(np.clip(k.sum() / n.sum(), 1e-6, 1 - 1e-6))
    rates = k / np.maximum(n, 1.0)
    variance = float(np.var(rates))
    binomial_part = float(np.mean(mean * (1 - mean) / np.maximum(n, 1.0)))
    extra = max(variance - binomial_part, 1e-8)
    rho = float(np.clip(extra / (mean * (1 - mean)), 1e-4, 0.5))
    start_total = (1 - rho) / rho
    start = np.log([mean * start_total, (1 - mean) * start_total])
    counts_k = k.astype(np.int64)
    counts_n = n.astype(np.int64)

    def negative_log_likelihood(log_params: np.ndarray) -> float:
        a, b = np.exp(log_params)
        return float(-betabinom.logpmf(counts_k, counts_n, a, b).sum())

    result = minimize(negative_log_likelihood, start, method="Nelder-Mead")
    a, b = np.exp(result.x if result.success else start)
    return BetaBinomialFit(
        alpha=float(a), beta=float(b), n_cells=int(k.size), mean_rate=mean
    )


def beta_binomial_upper_tail(
    successes: np.ndarray, trials: np.ndarray, fit: BetaBinomialFit
) -> np.ndarray:
    """Return ``P(K >= k)`` under the fitted beta-binomial, per cell."""
    from scipy.stats import betabinom

    k = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    return np.asarray(
        betabinom.sf(k - 1, n.astype(np.int64), fit.alpha, fit.beta), dtype=np.float64
    )


@dataclass(frozen=True)
class ContaminationFlags:
    """The §5.6 contamination flag of one dataset (prototype).

    Attributes:
        score: Negative counts / total counts per cell (NaN outside the
            classes).
        negative: Negative counts per cell.
        p_value: Upper-tail p-value under the class null (NaN where the class
            has no null).
        flag: Tail p < alpha with >= the minimum negative counts (False where
            there is no null).
        fits: Beta-binomial null per class (``None`` when too few cells).
    """

    score: np.ndarray
    negative: np.ndarray
    p_value: np.ndarray
    flag: np.ndarray
    fits: dict[str, BetaBinomialFit | None]


def contamination_flags(
    negative: np.ndarray,
    total_counts: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    null_cells: np.ndarray,
    classes: Sequence[str],
    *,
    alpha: float = CONTAMINATION_ALPHA,
    min_negative: int = CONTAMINATION_MIN_NEGATIVE,
    null_quantile: float = CONTAMINATION_NULL_QUANTILE,
    min_null_cells: int = CONTAMINATION_MIN_NULL_CELLS,
) -> ContaminationFlags:
    """Flag cells more contaminated than typical deep cells of their class.

    Per class (one dataset = one platform), the null is a beta-binomial fitted
    on the negative counts of the class's ``null_cells`` (its confident cells)
    in the top depth quartile; a cell is flagged if its upper-tail p-value is
    below ``alpha`` and it has at least ``min_negative`` negative counts
    (§5.6).

    Args:
        negative: Negative counts per cell (``negative_counts``).
        total_counts: Total counts per cell.
        labels: Broad class per cell.
        null_cells: Cells eligible for the null (confident calls).
        classes: Broad classes.
        alpha: Tail threshold.
        min_negative: Minimum negative counts of a flagged cell.
        null_quantile: Depth quantile above which null cells are fitted.
        min_null_cells: Minimum null cells to fit a class.

    Returns:
        Scores, p-values, flags and the fits.
    """
    k = np.asarray(negative, dtype=np.float64)
    n = np.asarray(total_counts, dtype=np.float64)
    names = _object_array(labels)
    eligible = np.asarray(null_cells, dtype=bool)
    score = np.where(np.isfinite(k), k / np.maximum(n, 1.0), math.nan)
    p_value = np.full(len(k), math.nan)
    fits: dict[str, BetaBinomialFit | None] = {}
    for cls in classes:
        members = (names == cls) & np.isfinite(k)
        candidates = members & eligible
        if not candidates.any():
            fits[cls] = None
            continue
        depth_cut = np.quantile(n[candidates], null_quantile)
        fitted = candidates & (n >= depth_cut)
        if fitted.sum() < min_null_cells:
            fits[cls] = None
            continue
        fit = fit_beta_binomial(k[fitted], n[fitted])
        fits[cls] = fit
        p_value[members] = beta_binomial_upper_tail(k[members], n[members], fit)
    flag = np.nan_to_num(p_value, nan=1.0) < alpha
    flag &= np.nan_to_num(k, nan=0.0) >= min_negative
    return ContaminationFlags(
        score=score, negative=k, p_value=p_value, flag=flag, fits=fits
    )


def class_profiles(
    profiles: pd.DataFrame,
    level: str,
    gene_ids: Sequence[str],
    node_class: Mapping[str, str],
    classes: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return each class's expected fractions (cell-weighted over its nodes).

    Args:
        profiles: ``profiles.parquet``.
        level: Level whose nodes are pooled.
        gene_ids: Query genes.
        node_class: Node name to class.
        classes: Classes to build.

    Returns:
        Class to its expected fraction over ``gene_ids`` (classes without a
        node are left out).
    """
    rows = profiles[profiles["level"].astype(str) == level]
    wide = (
        rows.pivot_table(
            index="node_name", columns="gene_id", values="mean_cpm", aggfunc="first"
        )
        .reindex(columns=[str(gene) for gene in gene_ids])
        .fillna(0.0)
    )
    weights = rows.drop_duplicates("node_name").set_index("node_name")["n_cells"]
    output: dict[str, np.ndarray] = {}
    for cls in classes:
        nodes = [name for name in wide.index if node_class.get(str(name)) == cls]
        if not nodes:
            continue
        w = weights.reindex(nodes).to_numpy(np.float64)
        mean = (wide.loc[nodes].to_numpy(np.float64) * w[:, None]).sum(axis=0) / w.sum()
        total = mean.sum()
        if total > 0:
            output[cls] = mean / total
    return output


def depth_grid(max_depth: int, *, n_points: int = 48, start: int = 1) -> np.ndarray:
    """Return a geometric grid of integer depths from ``start`` to ``max_depth``."""
    top = max(int(max_depth), start)
    grid: np.ndarray = np.unique(
        np.rint(np.geomspace(start, top, n_points)).astype(np.int64)
    )
    return grid[grid >= start]


def distinct_gene_quantiles(
    profile: np.ndarray,
    depths: np.ndarray,
    *,
    n_simulations: int = DIFFUSE_N_SIMULATIONS,
    quantile: float = DIFFUSE_QUANTILE,
    seed: int = 0,
) -> np.ndarray:
    """Simulate the distinct-gene quantile of multinomial draws at each depth.

    Args:
        profile: Expected fraction per gene (sums to 1).
        depths: Integer depths.
        n_simulations: Draws per depth (200 in §5.6).
        quantile: Quantile of the distinct-gene count.
        seed: RNG seed.

    Returns:
        The quantile per depth.
    """
    rng = np.random.default_rng(seed)
    probabilities = np.asarray(profile, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    values = np.empty(len(depths))
    for index, depth in enumerate(np.asarray(depths, dtype=np.int64)):
        draws = rng.multinomial(int(depth), probabilities, size=n_simulations)
        values[index] = np.quantile((draws > 0).sum(axis=1), quantile)
    return values


def expected_genes_quantile(
    depth: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    profiles_by_class: Mapping[str, np.ndarray],
    *,
    n_simulations: int = DIFFUSE_N_SIMULATIONS,
    quantile: float = DIFFUSE_QUANTILE,
    seed: int = 0,
    n_grid: int = 48,
) -> np.ndarray:
    """Return each cell's simulated distinct-gene quantile at its depth (§5.6).

    Args:
        depth: Counts per cell on the profiled (query) genes.
        labels: Class per cell (a key of ``profiles_by_class``).
        profiles_by_class: Class to expected fractions over the query genes.
        n_simulations: Multinomial draws per grid depth.
        quantile: Quantile (0.95: ``expected_genes_q95``).
        seed: RNG seed.
        n_grid: Grid depths (interpolated linearly in between).

    Returns:
        The quantile per cell; NaN for cells without a profile.
    """
    counts = np.asarray(depth, dtype=np.float64)
    names = _object_array(labels)
    values = np.full(len(counts), math.nan)
    if len(counts) == 0:
        return values
    grid = depth_grid(int(max(np.nanmax(counts), 1)), n_points=n_grid)
    for offset, (cls, profile) in enumerate(sorted(profiles_by_class.items())):
        members = names == cls
        if not members.any():
            continue
        curve = distinct_gene_quantiles(
            profile,
            grid,
            n_simulations=n_simulations,
            quantile=quantile,
            seed=seed + offset,
        )
        values[members] = np.interp(counts[members], grid, curve)
    return values


def realised_rates(
    flags: np.ndarray,
    labels: Sequence[object] | np.ndarray,
    classes: Sequence[str],
    *,
    include: np.ndarray | None = None,
    max_informative: float,
) -> pd.DataFrame:
    """Return the realised flag rate per class (H16).

    Args:
        flags: Flag per cell.
        labels: Class per cell.
        classes: Classes to report.
        include: Cells counted (default: all).
        max_informative: A class whose rate exceeds it is uninformative.

    Returns:
        ``class, n_cells, n_flagged, rate, informative`` rows.
    """
    names = _object_array(labels)
    flagged = np.asarray(flags, dtype=bool)
    keep = (
        np.ones(len(names), dtype=bool)
        if include is None
        else np.asarray(include, bool)
    )
    rows = []
    for cls in classes:
        members = keep & (names == cls)
        n_cells = int(members.sum())
        n_flagged = int((flagged & members).sum())
        rate = n_flagged / n_cells if n_cells else math.nan
        rows.append(
            {
                "class": cls,
                "n_cells": n_cells,
                "n_flagged": n_flagged,
                "rate": rate,
                "informative": bool(n_cells and rate <= max_informative),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Production flags (RESOLVE)

H16_MAX_INFORMATIVE: Final = 0.15
OOD_MIN_CELLS: Final = 30
MAD_SCALE: Final = 1.4826
FLAG_CONTAMINATED: Final = "contaminated"
FLAG_DIFFUSE: Final = "diffuse_profile"
FLAG_OOD: Final = "ood"
FLAG_MICROGLIAL_SPILLOVER: Final = "microglial_spillover"
FLAG_ASTRO_LOWCOUNT: Final = "astro_lowcount"
FLAG_REGION_INCOHERENT: Final = "region_incoherent"
REASON_HUMAN_SPILLOVER_OFF: Final = (
    "disabled: human microglial spill-over flag is off (OD-C5)"
)
REASON_HUMAN_SPILLOVER_NOT_IMPLEMENTED: Final = (
    "not_run: the spill-over test is mouse-only (E3); OD-C5 keeps it off for human"
)
REASON_MOUSE_NOT_COMPUTED: Final = "not_computed: no spill-over result given"
REASON_COHERENCE_NOT_COMPUTED: Final = "not_computed: no coherence result given"
DETECTION_PREFIX: Final = "detection_"


@dataclass(frozen=True)
class NegativeGeneSet:
    """The negative genes of each broad class on a query's genes (§5.6).

    Attributes:
        classes: Broad classes (rows of ``mask``).
        gene_ids: Query genes (columns of ``mask``).
        mask: ``classes x genes``: the gene is negative for the class.
        references: References whose detection decided it (human: WHB
            frontal and SEA-AD Multiregion).
        state_gene_ids: State genes never counted as negative.
        max_fraction: A gene is negative below this detection fraction in
            every reference.
        n_disagree_with_stored: (class, gene) pairs where the re-derived
            decision differs from the bundle's ``negative`` column.
    """

    classes: tuple[str, ...]
    gene_ids: tuple[str, ...]
    mask: np.ndarray
    references: tuple[str, ...] = ()
    state_gene_ids: tuple[str, ...] = ()
    max_fraction: float = 0.01
    n_disagree_with_stored: int = 0

    @classmethod
    def from_table(
        cls,
        negatives: pd.DataFrame,
        gene_ids: Sequence[str],
        *,
        classes: Sequence[str],
        state_gene_ids: Iterable[str] = (),
        max_fraction: float = 0.01,
    ) -> NegativeGeneSet:
        """Re-derive the negative genes from a bundle's ``negative_genes.parquet``.

        A gene is negative for a class when every reference column
        ``detection_<reference>`` is finite and below ``max_fraction`` and
        the gene is not a state gene (by Ensembl ID). A table without
        detection columns falls back to its ``negative`` column (still minus
        the state genes). Genes outside ``gene_ids`` are left out.

        Args:
            negatives: ``negative_genes.parquet``.
            gene_ids: The query genes (the dataset's panel genes).
            classes: Broad classes.
            state_gene_ids: Ensembl IDs of the curated state genes.
            max_fraction: Detection fraction below which a gene is negative.

        Returns:
            The negative gene set.
        """
        states = {str(gene) for gene in state_gene_ids}
        references = tuple(
            sorted(
                str(column).removeprefix(DETECTION_PREFIX)
                for column in negatives.columns
                if str(column).startswith(DETECTION_PREFIX)
            )
        )
        frame = negatives.copy()
        is_state = frame["gene_id"].astype(str).isin(states).to_numpy()
        if "is_state_gene" in frame.columns:
            is_state |= frame["is_state_gene"].astype(bool).to_numpy()
        if references:
            detection = frame[[f"{DETECTION_PREFIX}{ref}" for ref in references]]
            values = detection.to_numpy(dtype=np.float64)
            below = np.isfinite(values).all(axis=1) & (values < max_fraction).all(
                axis=1
            )
            derived = below & ~is_state
        else:
            derived = frame["negative"].astype(bool).to_numpy() & ~is_state
        disagree = 0
        if "negative" in frame.columns:
            disagree = int((frame["negative"].astype(bool).to_numpy() != derived).sum())
            if disagree:
                logger.warning(
                    "negative genes: %d (class, gene) pairs differ from the "
                    "bundle's negative column after re-deriving them",
                    disagree,
                )
        frame["derived"] = derived
        column = {str(gene): index for index, gene in enumerate(gene_ids)}
        row = {str(name): index for index, name in enumerate(classes)}
        mask = np.zeros((len(classes), len(gene_ids)), dtype=bool)
        chosen = frame[frame["derived"].to_numpy()]
        for broad, gene in zip(
            chosen["broad_class"].astype(str),
            chosen["gene_id"].astype(str),
            strict=True,
        ):
            if broad in row and gene in column:
                mask[row[broad], column[gene]] = True
        return cls(
            classes=tuple(str(name) for name in classes),
            gene_ids=tuple(str(gene) for gene in gene_ids),
            mask=mask,
            references=references,
            state_gene_ids=tuple(sorted(states)),
            max_fraction=float(max_fraction),
            n_disagree_with_stored=disagree,
        )

    def genes(self, cls: str) -> list[str]:
        """Return the negative genes of one class (query order)."""
        if cls not in self.classes:
            return []
        row = self.mask[self.classes.index(cls)]
        return [gene for gene, keep in zip(self.gene_ids, row, strict=True) if keep]

    def n_genes(self) -> dict[str, int]:
        """Return the number of negative genes per class."""
        return {
            cls: int(self.mask[index].sum()) for index, cls in enumerate(self.classes)
        }


@dataclass(frozen=True)
class FlagStratum:
    """The realised rate of one flag in one (class, platform) stratum (H16).

    Attributes:
        flag: Flag name (``contaminated``, ``diffuse_profile``, ``ood``, ...).
        cls: Broad class.
        platform: Platform.
        n_basis: Confident broad calls of the class with a flag value (the
            rate's basis, as the H16 prototype).
        n_flagged_basis: Of those, flagged.
        rate: ``n_flagged_basis / n_basis`` (``None`` without basis cells).
        n_cells: Table cells of the class the flag was evaluated on.
        n_flagged: Of those, flagged.
        rate_all: ``n_flagged / n_cells``.
        informative: The flag's own switch: false makes the stratum's flag
            null (contamination above 15%, diffuse above 30%, no null).
        informative_h16: ``rate <= 0.15`` (H16 marking).
        reason: Why the stratum is uninformative, if it is.
    """

    flag: str
    cls: str
    platform: str
    n_basis: int
    n_flagged_basis: int
    rate: float | None
    n_cells: int
    n_flagged: int
    rate_all: float | None
    informative: bool
    informative_h16: bool
    reason: str | None = None

    @property
    def token(self) -> str:
        """Return the provenance key ``<class>__<platform>`` (safe tokens)."""
        return f"{safe_token(self.cls)}__{safe_token(self.platform)}"

    def to_json(self) -> dict[str, Any]:
        """Return the stratum as JSON."""
        return {
            "flag": self.flag,
            "class": self.cls,
            "platform": self.platform,
            "n_basis": self.n_basis,
            "n_flagged_basis": self.n_flagged_basis,
            "rate": _rounded(self.rate),
            "n_cells": self.n_cells,
            "n_flagged": self.n_flagged,
            "rate_all": _rounded(self.rate_all),
            "informative": self.informative,
            "informative_h16": self.informative_h16,
            "reason": self.reason,
        }


def _rounded(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 6)


def _rate(flags: np.ndarray, members: np.ndarray) -> tuple[int, int, float | None]:
    n = int(members.sum())
    flagged = int((flags & members).sum())
    return n, flagged, (flagged / n if n else None)


def _stratum(
    flag: str,
    cls: str,
    platform: str,
    raw: np.ndarray,
    evaluated: np.ndarray,
    basis: np.ndarray,
    *,
    max_informative: float | None,
    unavailable: str | None = None,
) -> FlagStratum:
    """One stratum's rates and switch (``max_informative=None``: no switch)."""
    n_basis, flagged_basis, rate = _rate(raw, basis & evaluated)
    n_cells, flagged, rate_all = _rate(raw, evaluated)
    reason = unavailable
    informative = unavailable is None
    if informative and max_informative is not None:
        if rate is None:
            informative, reason = False, "no_confident_cells"
        elif rate > max_informative + 1e-12:
            informative = False
            reason = f"realised_rate_{rate:.3f}_above_{max_informative}"
    return FlagStratum(
        flag=flag,
        cls=cls,
        platform=platform,
        n_basis=n_basis,
        n_flagged_basis=flagged_basis,
        rate=rate,
        n_cells=n_cells,
        n_flagged=flagged,
        rate_all=rate_all,
        informative=informative,
        informative_h16=rate is not None and rate <= H16_MAX_INFORMATIVE + 1e-12,
        reason=reason,
    )


def nullable_flags(values: np.ndarray, defined: np.ndarray) -> pd.arrays.BooleanArray:
    """Return a nullable boolean column: ``values`` where ``defined``, else NA.

    Args:
        values: Boolean flag per object.
        defined: Where the flag has a value.

    Returns:
        A pandas ``boolean`` array.
    """
    data = np.asarray(values, dtype=bool)
    mask = ~np.asarray(defined, dtype=bool)
    return pd.arrays.BooleanArray(data & ~mask, mask)


@dataclass(frozen=True)
class ContaminationResult:
    """The contamination flag of one sample (§5.6).

    Attributes:
        score: ``contamination_score`` per object (NaN where undefined).
        neg_counts: Negative counts per object (NaN where undefined).
        p_value: Upper-tail p-value under the stratum's null.
        raw_flag: p < alpha with enough negative counts (before the switch).
        defined: Where ``flag_contaminated`` has a value (informative strata).
        fits: Beta-binomial null per class (``None``: too few null cells).
        strata: Realised rates per class.
    """

    score: np.ndarray
    neg_counts: np.ndarray
    p_value: np.ndarray
    raw_flag: np.ndarray
    defined: np.ndarray
    fits: dict[str, BetaBinomialFit | None]
    strata: list[FlagStratum]

    @property
    def flag(self) -> pd.arrays.BooleanArray:
        """Return ``flag_contaminated`` (NA outside informative strata)."""
        return nullable_flags(self.raw_flag, self.defined)


def contamination_result(
    neg_counts: np.ndarray,
    total_counts: np.ndarray,
    classes: Sequence[object] | np.ndarray,
    confident: np.ndarray,
    *,
    platform: str,
    class_names: Sequence[str],
    alpha: float = CONTAMINATION_ALPHA,
    min_neg_counts: int = CONTAMINATION_MIN_NEGATIVE,
    null_quantile: float = CONTAMINATION_NULL_QUANTILE,
    min_null_cells: int = CONTAMINATION_MIN_NULL_CELLS,
    max_informative: float = CONTAMINATION_MAX_INFORMATIVE,
) -> ContaminationResult:
    """Flag contaminated cells against the dataset-empirical null (§5.6).

    Args:
        neg_counts: Negative counts per object (NaN where the object has no
            assigned class or is outside the table).
        total_counts: Total counts per object.
        classes: Assigned broad class per object.
        confident: Whether the object's broad level is confident (null
            cells and the rate basis).
        platform: The sample's platform (one sample = one platform stratum).
        class_names: Broad classes.
        alpha: Tail probability.
        min_neg_counts: Negative counts a flagged cell needs.
        null_quantile: Depth quantile above which confident cells fit the null.
        min_null_cells: Null cells a class needs.
        max_informative: Realised rate above which a stratum is uninformative.

    Returns:
        The flag.
    """
    k = np.asarray(neg_counts, dtype=np.float64)
    n = np.asarray(total_counts, dtype=np.float64)
    names = _object_array(classes)
    confident_cells = np.asarray(confident, dtype=bool)
    raw = contamination_flags(
        k,
        n,
        names,
        confident_cells,
        class_names,
        alpha=alpha,
        min_negative=min_neg_counts,
        null_quantile=null_quantile,
        min_null_cells=min_null_cells,
    )
    defined = np.zeros(len(k), dtype=bool)
    strata: list[FlagStratum] = []
    for cls in class_names:
        members = (names == cls) & np.isfinite(k)
        fit = raw.fits.get(cls)
        stratum = _stratum(
            FLAG_CONTAMINATED,
            cls,
            platform,
            raw.flag,
            members,
            confident_cells,
            max_informative=max_informative,
            unavailable=None if fit is not None else "no_null",
        )
        strata.append(stratum)
        if stratum.informative:
            defined |= members
    return ContaminationResult(
        score=raw.score,
        neg_counts=k,
        p_value=raw.p_value,
        raw_flag=raw.flag & defined,
        defined=defined,
        fits=dict(raw.fits),
        strata=strata,
    )


@dataclass(frozen=True)
class DiffuseResult:
    """The diffuse-profile flag of one sample (§5.6).

    Attributes:
        expected: ``expected_genes_q95`` per object (NaN where undefined).
        detected: Distinct query genes per object.
        raw_flag: ``detected > expected`` (before the switch).
        defined: Where ``flag_diffuse_profile`` has a value.
        strata: Realised rates per class.
    """

    expected: np.ndarray
    detected: np.ndarray
    raw_flag: np.ndarray
    defined: np.ndarray
    strata: list[FlagStratum]

    @property
    def flag(self) -> pd.arrays.BooleanArray:
        """Return ``flag_diffuse_profile`` (NA outside informative strata)."""
        return nullable_flags(self.raw_flag, self.defined)


def diffuse_result(
    query_depth: np.ndarray,
    detected: np.ndarray,
    classes: Sequence[object] | np.ndarray,
    confident: np.ndarray,
    profiles_by_class: Mapping[str, np.ndarray],
    *,
    platform: str,
    class_names: Sequence[str],
    evaluate: np.ndarray | None = None,
    n_simulations: int = DIFFUSE_N_SIMULATIONS,
    quantile: float = DIFFUSE_QUANTILE,
    max_informative: float = DIFFUSE_MAX_INFORMATIVE,
    seed: int = 0,
) -> DiffuseResult:
    """Flag cells with more distinct genes than their class profile predicts.

    Args:
        query_depth: Counts per object on the profiled (query) genes.
        detected: Distinct query genes per object.
        classes: Assigned broad class per object.
        confident: Whether the object's broad level is confident (rate basis).
        profiles_by_class: Class to expected fraction over the query genes.
        platform: Platform.
        class_names: Broad classes.
        evaluate: Objects to evaluate (default: every object with a class).
        n_simulations: Multinomial draws per grid depth.
        quantile: Quantile of the simulated distinct genes.
        max_informative: Realised rate above which a stratum is uninformative.
        seed: RNG seed.

    Returns:
        The flag.
    """
    depth = np.asarray(query_depth, dtype=np.float64)
    genes = np.asarray(detected, dtype=np.float64)
    names = _object_array(classes)
    keep = (
        np.ones(len(depth), dtype=bool)
        if evaluate is None
        else np.asarray(evaluate, dtype=bool)
    )
    labels = _masked_objects(keep, names)
    expected = expected_genes_quantile(
        depth,
        labels,
        profiles_by_class,
        n_simulations=n_simulations,
        quantile=quantile,
        seed=seed,
    )
    evaluated = np.isfinite(expected)
    raw = evaluated & (genes > expected)
    defined = np.zeros(len(depth), dtype=bool)
    confident_cells = np.asarray(confident, dtype=bool)
    strata: list[FlagStratum] = []
    for cls in class_names:
        members = (labels == cls) & evaluated
        stratum = _stratum(
            FLAG_DIFFUSE,
            cls,
            platform,
            raw,
            members,
            confident_cells,
            max_informative=max_informative,
            unavailable=None if cls in profiles_by_class else "no_profile",
        )
        strata.append(stratum)
        if stratum.informative:
            defined |= members
    return DiffuseResult(
        expected=expected,
        detected=genes,
        raw_flag=raw & defined,
        defined=defined,
        strata=strata,
    )


@dataclass(frozen=True)
class OodResult:
    """The out-of-distribution flag of one sample (§4.3).

    Attributes:
        z: Robust z of ``avg_correlation`` per object (NaN where undefined).
        defined: Where ``flag_ood`` has a value.
        raw_flag: ``z < threshold``.
        strata: Realised rates per class (no null switch; H16 marking).
    """

    z: np.ndarray
    defined: np.ndarray
    raw_flag: np.ndarray
    strata: list[FlagStratum]

    @property
    def flag(self) -> pd.arrays.BooleanArray:
        """Return ``flag_ood``."""
        return nullable_flags(self.raw_flag, self.defined)


def robust_z(values: np.ndarray, *, min_cells: int = OOD_MIN_CELLS) -> np.ndarray:
    """Return the robust z (median / 1.4826 MAD) of finite values.

    Args:
        values: Values of one stratum.
        min_cells: Finite values a stratum needs.

    Returns:
        z per value; NaN for non-finite values, and everywhere when the
        stratum is too small or its MAD is zero.
    """
    array = np.asarray(values, dtype=np.float64)
    output = np.full(len(array), np.nan)
    finite = np.isfinite(array)
    if int(finite.sum()) < min_cells:
        return output
    median = float(np.median(array[finite]))
    mad = float(np.median(np.abs(array[finite] - median)))
    scale = MAD_SCALE * mad
    if scale <= 0:
        return output
    output[finite] = (array[finite] - median) / scale
    return output


def ood_result(
    corr: np.ndarray,
    classes: Sequence[object] | np.ndarray,
    depth_bins: np.ndarray,
    confident: np.ndarray,
    *,
    platform: str,
    class_names: Sequence[str],
    threshold: float = -3.0,
    min_cells: int = OOD_MIN_CELLS,
) -> OodResult:
    """Robust z of ``avg_correlation`` within class x platform x depth bin.

    Args:
        corr: The primary leaf's ``avg_correlation`` per object.
        classes: Assigned broad class per object (``None``: not evaluated).
        depth_bins: Grid depth bin per object (NaN below the grid: not
            evaluated).
        confident: Whether the object's broad level is confident (rate basis).
        platform: Platform.
        class_names: Broad classes.
        threshold: ``ood_robust_z`` (-3).
        min_cells: Cells a (class, depth bin) stratum needs.

    Returns:
        The flag.
    """
    values = np.asarray(corr, dtype=np.float64)
    names = _object_array(classes)
    bins = np.asarray(depth_bins, dtype=np.float64)
    z = np.full(len(values), np.nan)
    for cls in class_names:
        members = names == cls
        for depth in np.unique(bins[members & np.isfinite(bins)]):
            stratum = members & (bins == depth)
            z[stratum] = robust_z(values[stratum], min_cells=min_cells)
    defined = np.isfinite(z)
    raw = defined & (z < threshold)
    confident_cells = np.asarray(confident, dtype=bool)
    strata = [
        _stratum(
            FLAG_OOD,
            cls,
            platform,
            raw,
            (names == cls) & defined,
            confident_cells,
            max_informative=None,
        )
        for cls in class_names
    ]
    return OodResult(z=z, defined=defined, raw_flag=raw, strata=strata)


def astro_lowcount_flags(
    classes: Sequence[object] | np.ndarray,
    total_counts: np.ndarray,
    in_table: np.ndarray,
    *,
    below: int = 100,
    astro_class: str = "Astro-Epen",
) -> np.ndarray:
    """Return ``flag_astro_lowcount``: mouse Astro-Epen calls below 100 counts.

    Args:
        classes: WMB class per object.
        total_counts: Counts per object.
        in_table: Table cells.
        below: ``astro_lowcount_below``.
        astro_class: The WMB class name matched (by suffix, e.g.
            ``"30 Astro-Epen"``).

    Returns:
        Boolean per object (§7.4, E3 residual sink).
    """
    names = _object_array(classes)
    is_astro = np.array(
        [name is not None and str(name).endswith(astro_class) for name in names],
        dtype=bool,
    )
    counts = np.asarray(total_counts, dtype=np.float64)
    flags: np.ndarray = is_astro & (counts < below) & np.asarray(in_table, dtype=bool)
    return flags


def discovery_caution(
    flags: Iterable[np.ndarray | pd.arrays.BooleanArray | pd.Series],
) -> np.ndarray:
    """Return ``discovery_caution``: any of the given flags (NA counts false).

    Args:
        flags: The contaminated, diffuse, OOD, method-disagreement,
            microglial spill-over and Astro-Epen low-count flags.

    Returns:
        Boolean per object.
    """
    total: np.ndarray | None = None
    for values in flags:
        series = pd.Series(values)
        current = series.fillna(False).to_numpy(dtype=bool)
        total = current if total is None else (total | current)
    if total is None:
        raise ValueError("discovery_caution needs at least one flag")
    return total


@dataclass(frozen=True)
class FlagInputs:
    """What ``compute_flags`` reads for one sample (every segmented object).

    Attributes:
        species: ``"human"`` or ``"mouse"``.
        platform: Platform.
        total_counts: Counts per object after control removal.
        in_table: Table cells.
        assigned_class: The primary call's broad class per object (``None``
            without a call, for sinks and outside the vocabulary classes).
        confident: Whether the object's broad level is confident.
        method_disagree: ``flag_method_disagree``.
        corr: The primary leaf's ``avg_correlation`` per object.
        depth_bin: Grid depth bin per object (NaN below the grid).
        exclude_hard: ``exclude_hard`` (from the consensus).
        query_counts: Table cells x query genes (the dataset's panel genes).
        query_rows: Object position of each ``query_counts`` row.
        gene_ids: Query genes.
        negatives: Negative genes on the query genes (``None``: the bundle
            has none; the contamination flag is null).
        profiles_by_class: Class profiles on the query genes (``None``: the
            diffuse flag is null).
        wmb_class: Mouse WMB class per object (Astro-Epen low count).
        spillover: Mouse spill-over result on the table cells
            (``mouse_flags.microglial_spillover``, rows = ``query_rows``).
        coherence: Mouse F1 on the table cells
            (``mouse_flags.region_incoherent``, rows = ``query_rows``).
    """

    species: str
    platform: str
    total_counts: np.ndarray
    in_table: np.ndarray
    assigned_class: np.ndarray
    confident: np.ndarray
    method_disagree: np.ndarray
    corr: np.ndarray
    depth_bin: np.ndarray
    exclude_hard: np.ndarray
    query_counts: sparse.spmatrix | np.ndarray | None = None
    query_rows: np.ndarray | None = None
    gene_ids: tuple[str, ...] = ()
    negatives: NegativeGeneSet | None = None
    profiles_by_class: Mapping[str, np.ndarray] | None = None
    wmb_class: np.ndarray | None = None
    spillover: SpilloverResult | None = None
    coherence: CoherenceResult | None = None


@dataclass
class FlagSet:
    """The flag columns, realised rates and provenance of one sample.

    Attributes:
        columns: Label-table columns (§4.3; schema dtypes).
        strata: Realised rates per flag x class x platform (H16).
        thresholds: The flag settings applied.
        gene_sets: Gene sets per flag (negative genes per class).
        null_reasons: Why a flag (or a stratum) is null.
        fits: The contamination null per class.
    """

    columns: dict[str, Any]
    strata: list[FlagStratum] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    gene_sets: dict[str, list[str]] = field(default_factory=dict)
    null_reasons: dict[str, str] = field(default_factory=dict)
    fits: dict[str, dict[str, Any] | None] = field(default_factory=dict)

    def rates_frame(self) -> pd.DataFrame:
        """Return the realised rates as a table (one row per stratum)."""
        return pd.DataFrame([stratum.to_json() for stratum in self.strata])

    def provenance(self) -> FlagProvenance:
        """Return ``AnnotationProvenance.flags`` (§4.6)."""
        from merxen.annotation.provenance import FlagProvenance

        rates: dict[str, dict[str, float]] = {}
        informative: dict[str, dict[str, bool]] = {}
        for stratum in self.strata:
            informative.setdefault(stratum.flag, {})[stratum.token] = (
                stratum.informative
            )
            if stratum.rate is not None:
                rates.setdefault(stratum.flag, {})[stratum.token] = round(
                    float(stratum.rate), 6
                )
        return FlagProvenance(
            thresholds=dict(self.thresholds),
            gene_sets={key: list(value) for key, value in self.gene_sets.items()},
            realised_rates=rates,
            informative=informative,
            null_reasons=dict(self.null_reasons),
        )

    def summary(self) -> dict[str, Any]:
        """Return the resolve-summary record (rates, nulls, fits)."""
        return {
            "strata": [stratum.to_json() for stratum in self.strata],
            "null_reasons": dict(self.null_reasons),
            "thresholds": dict(self.thresholds),
            "contamination_null": self.fits,
            "n_negative_genes": {
                key.removeprefix("negative_"): len(value)
                for key, value in self.gene_sets.items()
                if key.startswith("negative_")
            },
        }


def _float32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).astype(np.float32)


def _row_sums(matrix: sparse.spmatrix | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy import sparse as sp

    csr = sp.csr_matrix(matrix)
    depth = np.asarray(csr.sum(axis=1)).reshape(-1).astype(np.float64)
    detected = np.diff(csr.indptr).astype(np.float64)
    if csr.nnz and (csr.data == 0).any():
        detected = np.asarray((csr != 0).sum(axis=1)).reshape(-1).astype(np.float64)
    return depth, detected


def compute_flags(
    inputs: FlagInputs,
    config: AnnotationFlagsConfig,
    *,
    class_names: Sequence[str],
    seed: int = 0,
) -> FlagSet:
    """Return every §4.3 flag column of one sample (report-only).

    Args:
        inputs: The sample's calls, counts and bundle tables.
        config: ``AnnotationConfig.flags``.
        class_names: The species' broad classes.
        seed: RNG seed of the diffuse simulation.

    Returns:
        The flag set.
    """
    n = len(inputs.total_counts)
    table = np.asarray(inputs.in_table, dtype=bool)
    counts = np.asarray(inputs.total_counts, dtype=np.float64)
    classes = _masked_objects(table, _object_array(inputs.assigned_class))
    confident = np.asarray(inputs.confident, dtype=bool) & table
    platform = str(inputs.platform).upper()
    null_reasons: dict[str, str] = {}
    gene_sets: dict[str, list[str]] = {}
    strata: list[FlagStratum] = []
    fits: dict[str, dict[str, Any] | None] = {}
    thresholds = {
        "contamination_alpha": float(config.contamination_alpha),
        "contamination_min_neg_counts": float(config.contamination_min_neg_counts),
        "contamination_null_depth_quantile": float(
            config.contamination_null_depth_quantile
        ),
        "contamination_min_null_cells": float(config.contamination_min_null_cells),
        "negative_gene_max_fraction": float(config.negative_gene_max_fraction),
        "flag_rate_uninformative_above": float(config.flag_rate_uninformative_above),
        "diffuse_quantile": float(config.diffuse_quantile),
        "diffuse_n_simulations": float(config.diffuse_n_simulations),
        "diffuse_rate_uninformative_above": float(
            config.diffuse_rate_uninformative_above
        ),
        "ood_robust_z": float(config.ood_robust_z),
        "ood_min_cells": float(config.ood_min_cells),
        "h16_uninformative_above": float(H16_MAX_INFORMATIVE),
    }
    columns: dict[str, Any] = {}

    # Query counts per object (table rows only).
    query_depth = np.full(n, np.nan)
    detected = np.full(n, np.nan)
    have_query = inputs.query_counts is not None and inputs.query_rows is not None
    if have_query:
        assert inputs.query_counts is not None and inputs.query_rows is not None
        rows = np.asarray(inputs.query_rows, dtype=np.int64)
        depth_rows, detected_rows = _row_sums(inputs.query_counts)
        query_depth[rows] = depth_rows
        detected[rows] = detected_rows

    # Contamination.
    neg = np.full(n, np.nan)
    if inputs.negatives is not None and have_query:
        assert inputs.query_counts is not None and inputs.query_rows is not None
        rows = np.asarray(inputs.query_rows, dtype=np.int64)
        neg[rows] = negative_counts(
            inputs.query_counts,
            classes[rows],
            inputs.negatives.mask,
            list(inputs.negatives.classes),
        )
        for cls in inputs.negatives.classes:
            gene_sets[f"negative_{safe_token(cls)}"] = inputs.negatives.genes(cls)
    else:
        null_reasons[FLAG_CONTAMINATED] = (
            "no negative genes in the primary bundle"
            if inputs.negatives is None
            else "no query counts"
        )
    contamination = contamination_result(
        neg,
        counts,
        classes,
        confident,
        platform=platform,
        class_names=class_names,
        alpha=config.contamination_alpha,
        min_neg_counts=config.contamination_min_neg_counts,
        null_quantile=config.contamination_null_depth_quantile,
        min_null_cells=config.contamination_min_null_cells,
        max_informative=config.flag_rate_uninformative_above,
    )
    strata += contamination.strata
    for cls, fit in contamination.fits.items():
        fits[safe_token(cls)] = (
            None
            if fit is None
            else {
                "alpha": round(fit.alpha, 6),
                "beta": round(fit.beta, 6),
                "n_cells": fit.n_cells,
                "mean_rate": round(fit.mean_rate, 6),
            }
        )
    score = np.clip(contamination.score, 0.0, 1.0)
    columns[Columns.CONTAMINATION_SCORE] = _float32(score)
    columns[Columns.NEG_COUNTS] = pd.array(
        [None if not math.isfinite(value) else int(round(value)) for value in neg],
        dtype=pd.Int32Dtype(),
    )
    columns[Columns.FLAG_CONTAMINATED] = contamination.flag

    # Diffuse profile.
    if inputs.profiles_by_class and have_query:
        diffuse = diffuse_result(
            query_depth,
            detected,
            classes,
            confident,
            inputs.profiles_by_class,
            platform=platform,
            class_names=class_names,
            evaluate=table & np.isfinite(query_depth),
            n_simulations=config.diffuse_n_simulations,
            quantile=config.diffuse_quantile,
            max_informative=config.diffuse_rate_uninformative_above,
            seed=seed,
        )
        strata += diffuse.strata
        columns[Columns.EXPECTED_GENES_Q95] = _float32(diffuse.expected)
        columns[Columns.FLAG_DIFFUSE_PROFILE] = diffuse.flag
        diffuse_flag: Any = diffuse.flag
    else:
        null_reasons[FLAG_DIFFUSE] = (
            "no class profiles in the primary bundle"
            if not inputs.profiles_by_class
            else "no query counts"
        )
        columns[Columns.EXPECTED_GENES_Q95] = np.full(n, np.nan, dtype=np.float32)
        diffuse_flag = nullable_flags(np.zeros(n, bool), np.zeros(n, bool))
        columns[Columns.FLAG_DIFFUSE_PROFILE] = diffuse_flag

    # OOD.
    ood = ood_result(
        np.where(table, np.asarray(inputs.corr, dtype=np.float64), np.nan),
        classes,
        np.asarray(inputs.depth_bin, dtype=np.float64),
        confident,
        platform=platform,
        class_names=class_names,
        threshold=config.ood_robust_z,
        min_cells=config.ood_min_cells,
    )
    strata += ood.strata
    columns[Columns.OOD_Z] = _float32(ood.z)
    columns[Columns.FLAG_OOD] = ood.flag

    # Microglial spill-over (human off, OD-C5; mouse M6) and the mouse flags.
    enabled = config.microglial_spillover_enabled
    if enabled is None:
        enabled = inputs.species == "mouse"
    empty = nullable_flags(np.zeros(n, bool), np.zeros(n, bool))
    columns[Columns.MICROGLIA_STAT] = np.full(n, np.nan, dtype=np.float32)
    columns[Columns.MICROGLIA_WEIGHT] = np.full(n, np.nan, dtype=np.float32)
    columns[Columns.FLAG_MICROGLIAL_SPILLOVER] = empty
    spill = inputs.spillover if enabled else None
    rows = (
        np.asarray(inputs.query_rows, dtype=np.int64)
        if inputs.query_rows is not None
        else np.flatnonzero(table)
    )
    if spill is not None and len(spill.raw_flag) == len(rows):
        stat = np.full(n, np.nan)
        weight = np.full(n, np.nan)
        raw = np.zeros(n, dtype=bool)
        stat[rows] = spill.statistic
        weight[rows] = spill.weight
        raw[rows] = spill.raw_flag
        defined = np.zeros(n, dtype=bool)
        if spill.defined:
            defined[rows] = True
        columns[Columns.MICROGLIA_STAT] = _float32(stat)
        columns[Columns.MICROGLIA_WEIGHT] = np.clip(weight, 0.0, 1.0).astype(np.float32)
        columns[Columns.FLAG_MICROGLIAL_SPILLOVER] = nullable_flags(raw, defined)
        gene_sets[f"{FLAG_MICROGLIAL_SPILLOVER}_genes"] = list(spill.genes.gene_ids)
        if spill.astro_genes is not None:
            gene_sets[f"{FLAG_MICROGLIAL_SPILLOVER}_astrocyte_genes"] = list(
                spill.astro_genes.gene_ids
            )
        if spill.null_reason is not None:
            null_reasons[FLAG_MICROGLIAL_SPILLOVER] = spill.null_reason
        thresholds.update(
            {
                "microglia_stat_min": float(config.microglia_stat_min),
                "microglia_weight_min": float(config.microglia_weight_min),
                "microglia_fpr_max": float(config.microglia_fpr_max),
                "specific_gene_ratio": float(config.specific_gene_ratio),
                "specific_gene_min_share": float(config.specific_gene_min_share),
                "min_specific_genes": float(config.min_specific_genes),
            }
        )
        strata += [
            _stratum(
                FLAG_MICROGLIAL_SPILLOVER,
                cls,
                platform,
                raw,
                defined & (classes == cls),
                confident,
                max_informative=None,
                unavailable=None if spill.defined else spill.null_reason,
            )
            for cls in class_names
        ]
    elif enabled:
        null_reasons[FLAG_MICROGLIAL_SPILLOVER] = (
            REASON_MOUSE_NOT_COMPUTED
            if inputs.species == "mouse"
            else REASON_HUMAN_SPILLOVER_NOT_IMPLEMENTED
        )
    else:
        null_reasons[FLAG_MICROGLIAL_SPILLOVER] = REASON_HUMAN_SPILLOVER_OFF
    astro = np.zeros(n, dtype=bool)
    if inputs.species == "mouse":
        columns.update(mouse_flag_columns(n))
        coherence = inputs.coherence
        if coherence is not None and len(coherence.raw_flag) == len(rows):
            values = np.full(n, np.nan)
            values[rows] = coherence.coherence
            raw = np.zeros(n, dtype=bool)
            raw[rows] = coherence.raw_flag
            defined = np.zeros(n, dtype=bool)
            if coherence.defined:
                defined[rows] = True
            columns[Columns.REGION_COHERENCE] = np.clip(values, 0.0, 1.0).astype(
                np.float32
            )
            columns[Columns.FLAG_REGION_INCOHERENT] = nullable_flags(raw, defined)
            gene_sets[f"{FLAG_REGION_INCOHERENT}_classes"] = list(
                coherence.restricted_classes
            )
            if coherence.null_reason is not None:
                null_reasons[FLAG_REGION_INCOHERENT] = coherence.null_reason
        else:
            null_reasons[FLAG_REGION_INCOHERENT] = REASON_COHERENCE_NOT_COMPUTED
        if inputs.wmb_class is not None:
            astro = astro_lowcount_flags(
                inputs.wmb_class, counts, table, below=config.astro_lowcount_below
            )
        columns[Columns.FLAG_ASTRO_LOWCOUNT] = astro

    columns[Columns.DISCOVERY_CAUTION] = (
        discovery_caution(
            [
                contamination.flag,
                diffuse_flag,
                ood.flag,
                np.asarray(inputs.method_disagree, dtype=bool),
                columns[Columns.FLAG_MICROGLIAL_SPILLOVER],
                astro,
            ]
        )
        & table
    )
    for stratum in strata:
        if not stratum.informative and stratum.reason:
            null_reasons[f"{stratum.flag}__{stratum.token}"] = stratum.reason
    logger.info(
        "flags %s: contaminated %d, diffuse %d, ood %d, discovery_caution %d of %d "
        "table cells; uninformative strata %s",
        platform,
        int(contamination.raw_flag.sum()),
        int(pd.Series(diffuse_flag).fillna(False).sum()),
        int(ood.raw_flag.sum()),
        int(columns[Columns.DISCOVERY_CAUTION].sum()),
        int(table.sum()),
        ", ".join(
            f"{stratum.flag}:{stratum.cls}"
            for stratum in strata
            if not stratum.informative
        )
        or "none",
    )
    return FlagSet(
        columns=columns,
        strata=strata,
        thresholds=thresholds,
        gene_sets=gene_sets,
        null_reasons=null_reasons,
        fits=fits,
    )


def mouse_flag_columns(n: int) -> dict[str, Any]:
    """Return the mouse-only flag columns as nulls (filled when F1 is given).

    Args:
        n: Objects.

    Returns:
        ``flag_region_incoherent`` (NA) and ``region_coherence`` (NaN).
    """
    return {
        Columns.FLAG_REGION_INCOHERENT: nullable_flags(
            np.zeros(n, bool), np.zeros(n, bool)
        ),
        Columns.REGION_COHERENCE: np.full(n, np.nan, dtype=np.float32),
    }


def profiles_for_classes(
    profiles: pd.DataFrame,
    *,
    level: str,
    gene_ids: Sequence[str],
    node_class: Mapping[str, str],
    classes: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return the class profiles of a bundle on the query genes (diffuse flag).

    Args:
        profiles: ``profiles.parquet``.
        level: The primary leaf level (e.g. ``CCN202210140_SUPC``).
        gene_ids: Query genes.
        node_class: Node name to broad class.
        classes: Broad classes.

    Returns:
        ``class_profiles`` output (classes without a node are left out).
    """
    return class_profiles(profiles, level, gene_ids, node_class, classes)
