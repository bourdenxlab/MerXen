"""Mouse report flags: microglial spill-over and region coherence (plan §4.3, §7.4).

Both flags are report-only: they never change a label or a status.

- **Microglial spill-over** (E3 ``09b_specific_flag.py``; plan §7.4, §8.6).
  The specific genes are derived from the primary bundle per panel: genes
  whose microglia profile (WMB subclass ``334 Microglia NN``; the Immune
  class when the tree has no microglia subclass) is at least
  ``specific_gene_ratio`` (20) times the largest non-Immune class profile
  and at least ``specific_gene_min_share`` (1/1000) of the microglia
  profile, immediate-early genes excluded. On the ag7 panel this gives E3's
  five genes (Csf1r, Cx3cr1, Aif1, Blnk, C1qa), on VZG2 its 13. Fewer than
  ``min_specific_genes`` (3) make the flag null with reason
  ``insufficient_panel_genes``. Per cell a binomial likelihood-ratio test
  on the counts ``k`` in those genes of ``n`` total counts compares
  ``q(w) = (1 - w) p0 + w p1`` over a grid of spill weights ``w`` with
  ``w = 0``: ``p1`` is the genes' share of the microglia profile, ``p0``
  0.9 x their share of the cell's best non-Immune subclass (multinomial
  argmax over the subclass profiles, each floored 1% towards the mean
  profile, as E3's reference) + 0.1 x their share of the dataset's counts
  (ambient). ``microglia_stat`` = the statistic, ``microglia_weight`` = the
  maximising ``w``; flagged at statistic >= ``microglia_stat_min`` (10) and
  weight >= ``microglia_weight_min`` (0.05). The flag measures spill-over
  **prevalence**, never a microglia count (8.6% of VZG2 cells vs 1.5%
  microglia in MERFISH). E3 used supertype profiles; the bundle's are
  subclass-level (supertype is dropped from the mapping tree), and on ag7 /
  VZG2 the two agree on 99.8% of cells (M6 stage B).
- **False-positive check** (§8.6, MO5): the flag rate among marker
  astrocytes must not exceed ``microglia_fpr_max`` (0.5%); above it the
  flag is null with reason ``astrocyte_fpr``. Marker astrocytes are E3's
  AST set (``04_metrics.py``) with the panel's derived Astro-Epen genes
  instead of Aqp4 alone: >= 2 counts and >= 10 per 1,000 on them, and <= 3
  per 1,000 on the **purity genes**, E3's MG_loose genes (Cx3cr1, Csf1r,
  C1qa) when all three are on the panel, else the three most specific
  derived microglia genes. The purity genes must leave some flag genes
  free: conditioning on low counts in every flag gene caps a cell's
  microglial share at 3 per 1,000, below any flagged spill weight, so the
  rate would be 0 by construction; the check is then not evaluated (a
  recorded reason) and the flag stays defined.
- **Held-out check** (§8.6, needs >= 6 genes): the genes are split into
  two halves by specificity rank (odd / even), the flag is rebuilt on the
  first half and the enrichment of the second half (counts per 1,000 in
  flagged vs unflagged cells) is reported (E3 VZG2: 35-40x; MO5 >= 10x).
- **Region coherence F1** (E7 §3): ``region_coherence`` = the share of a
  table cell's 30 nearest table-cell neighbours with its WMB class;
  ``flag_region_incoherent`` = a region-restricted class
  (``wmb_region_restricted_classes.csv``), coherence < ``coherence_min``
  (0.1) and class bootstrap probability < ``coherence_max_conf`` (0.8).
  Null without coordinates.

The module needs numpy and pandas; scipy is imported inside the functions
that use it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.config import AnnotationFlagsConfig, MouseRegionConfig

logger = logging.getLogger(__name__)

# E3 09b_specific_flag.py IEG (verbatim): never specific genes.
IMMEDIATE_EARLY_GENES: Final[frozenset[str]] = frozenset(
    {
        "Fos",
        "Fosb",
        "Jun",
        "Junb",
        "Egr1",
        "Egr2",
        "Egr3",
        "Ier2",
        "Nr4a1",
        "Nr4a3",
        "Tnf",
        "Dusp1",
        "Arc",
        "Npas4",
        "Atf3",
    }
)
# E3 09b_specific_flag.py WG and the reference of e3lib.load_ref.
SPILL_WEIGHT_GRID: Final[tuple[float, ...]] = (
    0.0,
    0.02,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
)
PROFILE_FLOOR: Final = 0.01
AMBIENT_WEIGHT: Final = 0.1
PROBABILITY_CLIP: Final = 1e-9
# E3 04_metrics.py AST: marker astrocytes of the false-positive check.
ASTRO_MIN_COUNTS: Final = 2
ASTRO_MIN_PER_1000: Final = 10.0
ASTRO_MAX_MICROGLIA_PER_1000: Final = 3.0
# E3 04_metrics.py MG_loose genes: the purity genes of the AST set.
MG_LOOSE_SYMBOLS: Final[tuple[str, ...]] = ("Cx3cr1", "Csf1r", "C1qa")
PURITY_BASIS_MG_LOOSE: Final = "e3_mg_loose"
PURITY_BASIS_DERIVED: Final = "derived_top3"
# Plan §8.6: the held-out check needs >= 6 genes (build on half, test on half).
MIN_HELDOUT_GENES: Final = 6
IMMUNE_CLASS_SUFFIX: Final = "Immune"
ASTRO_CLASS_SUFFIX: Final = "Astro-Epen"
MICROGLIA_TOKEN: Final = "Microglia"
REASON_INSUFFICIENT_GENES: Final = "insufficient_panel_genes"
REASON_ASTRO_FPR: Final = "astrocyte_fpr"
REASON_NO_PROFILES: Final = "no_profiles"
REASON_NO_QUERY: Final = "no_query_counts"
REASON_NO_COORDINATES: Final = "no_coordinates"
CHUNK_ROWS: Final = 20_000


def _ends_with(name: object, suffix: str) -> bool:
    return name is not None and str(name).endswith(suffix)


@dataclass(frozen=True)
class MouseFlagProfiles:
    """WMB class and subclass profiles on a query's genes (``profiles.parquet``).

    Attributes:
        gene_ids: Query genes (columns of the profiles).
        symbols: Their symbols (for the immediate-early-gene exclusion and
            reporting).
        class_names: WMB class names (rows of ``class_profiles``).
        class_profiles: Expected fraction per class x gene.
        subclass_names: WMB subclass names (rows of ``subclass_profiles``).
        subclass_class: Class name of each subclass.
        subclass_profiles: Expected fraction per subclass x gene.
    """

    gene_ids: tuple[str, ...]
    symbols: tuple[str, ...]
    class_names: tuple[str, ...]
    class_profiles: np.ndarray
    subclass_names: tuple[str, ...]
    subclass_class: tuple[str, ...]
    subclass_profiles: np.ndarray

    @classmethod
    def from_table(
        cls,
        profiles: pd.DataFrame,
        *,
        gene_ids: Sequence[str],
        symbols: Sequence[str],
        class_level: str,
        subclass_level: str,
        subclass_class: Mapping[str, str],
    ) -> MouseFlagProfiles:
        """Build the profiles from a bundle's ``profiles.parquet``.

        Args:
            profiles: The long table (``level``, ``node_name``, ``gene_id``,
                ``expected_fraction``).
            gene_ids: Query genes; genes the table lacks get zero.
            symbols: Their symbols.
            class_level: Class level key (``CCN20230722_CLAS``).
            subclass_level: Subclass level key (``CCN20230722_SUBC``).
            subclass_class: Subclass name to class name (mapping tree).

        Returns:
            The profiles.

        Raises:
            ValueError: If a level is missing or the symbols do not fit.
        """
        if len(symbols) != len(gene_ids):
            raise ValueError("symbols must have one entry per gene")
        needed = {"level", "node_name", "gene_id", "expected_fraction"}
        missing = sorted(needed - set(profiles.columns))
        if missing:
            raise ValueError(f"profiles table lacks columns {missing}")
        genes = [str(gene) for gene in gene_ids]

        def matrix(level: str) -> tuple[tuple[str, ...], np.ndarray]:
            rows = profiles[profiles["level"].astype(str) == level]
            if rows.empty:
                raise ValueError(f"profiles table has no {level} rows")
            wide = rows.pivot_table(
                index="node_name",
                columns="gene_id",
                values="expected_fraction",
                aggfunc="first",
                observed=True,
            )
            wide.index = wide.index.astype(str)
            wide = wide.sort_index().reindex(columns=genes).fillna(0.0)
            return tuple(wide.index), wide.to_numpy(np.float64)

        class_names, class_matrix = matrix(class_level)
        subclass_names, subclass_matrix = matrix(subclass_level)
        return cls(
            gene_ids=tuple(genes),
            symbols=tuple(str(symbol) for symbol in symbols),
            class_names=class_names,
            class_profiles=class_matrix,
            subclass_names=subclass_names,
            subclass_class=tuple(
                str(subclass_class.get(name, "")) for name in subclass_names
            ),
            subclass_profiles=subclass_matrix,
        )

    def class_row(self, suffix: str) -> int | None:
        """Return the row of the class whose name ends with ``suffix``."""
        rows = [
            index
            for index, name in enumerate(self.class_names)
            if _ends_with(name, suffix)
        ]
        return rows[0] if len(rows) == 1 else None

    def microglia_profile(self) -> np.ndarray | None:
        """Return the microglia target profile (subclass, else Immune class)."""
        rows = [
            index
            for index, name in enumerate(self.subclass_names)
            if MICROGLIA_TOKEN in name
            and _ends_with(self.subclass_class[index], IMMUNE_CLASS_SUFFIX)
        ]
        if rows:
            profile: np.ndarray = self.subclass_profiles[rows].mean(axis=0)
            return profile
        row = self.class_row(IMMUNE_CLASS_SUFFIX)
        return None if row is None else self.class_profiles[row]


@dataclass(frozen=True)
class SpecificGenes:
    """A target's specific genes on a panel (E3 rule; §8.6).

    Attributes:
        target: ``microglia`` or ``astrocyte``.
        indices: Gene positions (query order), most specific first.
        gene_ids: Their IDs.
        symbols: Their symbols.
        ratios: Target / largest non-target class profile per gene.
    """

    target: str
    indices: np.ndarray
    gene_ids: tuple[str, ...]
    symbols: tuple[str, ...]
    ratios: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.indices)


def specific_genes(
    target_profile: np.ndarray,
    other_profiles: np.ndarray,
    *,
    gene_ids: Sequence[str],
    symbols: Sequence[str],
    ratio: float,
    min_share: float,
    target: str,
    exclude_symbols: frozenset[str] = IMMEDIATE_EARLY_GENES,
) -> SpecificGenes:
    """Return the genes specific to a target profile (E3 ``spec_genes``).

    Args:
        target_profile: Expected fraction per gene of the target.
        other_profiles: Non-target class profiles (rows) on the same genes.
        gene_ids: Gene IDs.
        symbols: Gene symbols.
        ratio: Minimum target / largest non-target profile.
        min_share: Minimum target share.
        target: Name recorded with the set.
        exclude_symbols: Symbols never selected (immediate-early genes).

    Returns:
        The set, most specific gene first (ties by gene ID).
    """
    target_values = np.asarray(target_profile, dtype=np.float64)
    others = np.asarray(other_profiles, dtype=np.float64)
    largest = others.max(axis=0) if len(others) else np.zeros_like(target_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(largest > 0, target_values / largest, np.inf)
    excluded = np.array([str(symbol) in exclude_symbols for symbol in symbols])
    keep = (ratios >= ratio) & (target_values >= min_share) & ~excluded
    positions = np.flatnonzero(keep)
    order = sorted(positions, key=lambda index: (-ratios[index], str(gene_ids[index])))
    chosen = np.asarray(order, dtype=np.int64)
    return SpecificGenes(
        target=target,
        indices=chosen,
        gene_ids=tuple(str(gene_ids[index]) for index in chosen),
        symbols=tuple(str(symbols[index]) for index in chosen),
        ratios=tuple(float(ratios[index]) for index in chosen),
    )


def microglia_genes(
    profiles: MouseFlagProfiles, config: AnnotationFlagsConfig
) -> SpecificGenes:
    """Return the panel's microglia-specific genes (§8.6).

    Args:
        profiles: Class and subclass profiles.
        config: ``specific_gene_ratio`` and ``specific_gene_min_share``.

    Returns:
        The set (empty when the tree has no microglia node).
    """
    target = profiles.microglia_profile()
    others = np.array(
        [not _ends_with(name, IMMUNE_CLASS_SUFFIX) for name in profiles.class_names],
        dtype=bool,
    )
    if target is None:
        return _empty_set("microglia")
    return specific_genes(
        target,
        profiles.class_profiles[others],
        gene_ids=profiles.gene_ids,
        symbols=profiles.symbols,
        ratio=config.specific_gene_ratio,
        min_share=config.specific_gene_min_share,
        target="microglia",
    )


def astrocyte_genes(
    profiles: MouseFlagProfiles, config: AnnotationFlagsConfig
) -> SpecificGenes:
    """Return the panel's Astro-Epen-specific genes (the FPR check's set, §8.6).

    Args:
        profiles: Class profiles.
        config: The specificity rule.

    Returns:
        The set (empty without an Astro-Epen class).
    """
    row = profiles.class_row(ASTRO_CLASS_SUFFIX)
    if row is None:
        return _empty_set("astrocyte")
    others = np.ones(len(profiles.class_names), dtype=bool)
    others[row] = False
    return specific_genes(
        profiles.class_profiles[row],
        profiles.class_profiles[others],
        gene_ids=profiles.gene_ids,
        symbols=profiles.symbols,
        ratio=config.specific_gene_ratio,
        min_share=config.specific_gene_min_share,
        target="astrocyte",
    )


def _empty_set(target: str) -> SpecificGenes:
    return SpecificGenes(
        target=target,
        indices=np.zeros(0, dtype=np.int64),
        gene_ids=(),
        symbols=(),
        ratios=(),
    )


def floored_profiles(matrix: np.ndarray, floor: float = PROFILE_FLOOR) -> np.ndarray:
    """Return row-normalised profiles floored towards their mean (e3lib ``load_ref``).

    Args:
        matrix: Profiles (rows) on the query genes.
        floor: Weight of the mean profile.

    Returns:
        ``(1 - floor) P + floor mean(P)`` with each row of ``P`` summing to 1.
    """
    values = np.asarray(matrix, dtype=np.float64)
    totals = values.sum(axis=1, keepdims=True)
    normalised = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    floored: np.ndarray = (1.0 - floor) * normalised + floor * normalised.mean(
        axis=0, keepdims=True
    )
    return floored


def spillover_statistic(
    counts: sparse.spmatrix | np.ndarray,
    gene_indices: np.ndarray,
    base_profiles: np.ndarray,
    target_profile: np.ndarray,
    *,
    weight_grid: Sequence[float] = SPILL_WEIGHT_GRID,
    ambient_weight: float = AMBIENT_WEIGHT,
    ambient_share: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return E3's binomial likelihood-ratio statistic and spill weight per cell.

    Args:
        counts: Cells x query genes (raw counts).
        gene_indices: The specific genes.
        base_profiles: Non-target profiles (rows, floored, each summing to
            1); a cell's host is the one with the largest multinomial
            log-likelihood.
        target_profile: The target profile (summing to 1).
        weight_grid: Spill weights tested (must start with 0).
        ambient_weight: Weight of the dataset's ambient share in ``p0``.
        ambient_share: The genes' share of the dataset's counts (default:
            from ``counts``).

    Returns:
        ``(statistic, weight)``: ``2 [max_w LL(w) - LL(0)]`` and the
        maximising ``w`` (0 for cells without counts).

    Raises:
        ValueError: If the grid does not start with 0 or the shapes differ.
    """
    from scipy import sparse as sp

    grid = np.asarray(weight_grid, dtype=np.float64)
    if not len(grid) or grid[0] != 0.0:
        raise ValueError("weight_grid must start with 0")
    matrix = sp.csr_matrix(counts, dtype=np.float64)
    base = np.asarray(base_profiles, dtype=np.float64)
    target = np.asarray(target_profile, dtype=np.float64)
    if base.shape[1] != matrix.shape[1] or len(target) != matrix.shape[1]:
        raise ValueError("profiles and counts must share the genes")
    genes = np.asarray(gene_indices, dtype=np.int64)
    n_cells = matrix.shape[0]
    total = np.asarray(matrix.sum(axis=1)).reshape(-1)
    k = np.asarray(matrix[:, genes].sum(axis=1)).reshape(-1)
    if ambient_share is None:
        ambient_share = float(k.sum() / total.sum()) if total.sum() > 0 else 0.0
    log_base = np.log(np.clip(base, PROBABILITY_CLIP, None)).T
    host = np.zeros(n_cells, dtype=np.int64)
    for start in range(0, n_cells, CHUNK_ROWS):
        block = matrix[start : start + CHUNK_ROWS] @ log_base
        host[start : start + CHUNK_ROWS] = np.asarray(block).argmax(axis=1)
    base_share = base[:, genes].sum(axis=1)
    p0 = (1.0 - ambient_weight) * base_share[host] + ambient_weight * ambient_share
    p1 = float(target[genes].sum())
    q = (1.0 - grid[None, :]) * p0[:, None] + grid[None, :] * p1
    q = np.clip(q, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    log_likelihood = k[:, None] * np.log(q) + (total - k)[:, None] * np.log1p(-q)
    best = log_likelihood.argmax(axis=1)
    statistic = 2.0 * (log_likelihood[np.arange(n_cells), best] - log_likelihood[:, 0])
    return np.maximum(statistic, 0.0), grid[best]


def astro_purity_genes(
    profiles: MouseFlagProfiles, microglia: SpecificGenes
) -> tuple[SpecificGenes, str]:
    """Return the microglial genes that exclude marker astrocytes (E3 AST).

    Args:
        profiles: The panel's profiles (gene symbols).
        microglia: The derived microglia set (the flag's genes).

    Returns:
        ``(genes, basis)``: E3's MG_loose genes (Cx3cr1, Csf1r, C1qa) when all
        three are on the panel (``e3_mg_loose``), else the three most
        specific derived microglia genes (``derived_top3``).
    """
    position = {symbol: index for index, symbol in enumerate(profiles.symbols)}
    if all(symbol in position for symbol in MG_LOOSE_SYMBOLS):
        indices = np.array(
            [position[symbol] for symbol in MG_LOOSE_SYMBOLS], dtype=np.int64
        )
        return (
            SpecificGenes(
                target="astrocyte_purity",
                indices=indices,
                gene_ids=tuple(profiles.gene_ids[index] for index in indices),
                symbols=MG_LOOSE_SYMBOLS,
                ratios=tuple(math.nan for _ in indices),
            ),
            PURITY_BASIS_MG_LOOSE,
        )
    return (
        SpecificGenes(
            target="astrocyte_purity",
            indices=microglia.indices[:3],
            gene_ids=microglia.gene_ids[:3],
            symbols=microglia.symbols[:3],
            ratios=microglia.ratios[:3],
        ),
        PURITY_BASIS_DERIVED,
    )


def marker_astrocytes(
    counts: sparse.spmatrix | np.ndarray,
    astro: SpecificGenes,
    purity: SpecificGenes,
) -> np.ndarray:
    """Return E3's marker astrocytes (AST) with the panel's derived sets.

    Args:
        counts: Cells x query genes.
        astro: Astro-Epen genes.
        purity: The microglial purity genes (``astro_purity_genes``).

    Returns:
        Boolean per cell: >= 2 astrocyte counts and >= 10 per 1,000 on
        them, and <= 3 per 1,000 on the purity genes.
    """
    from scipy import sparse as sp

    matrix = sp.csr_matrix(counts, dtype=np.float64)
    total = np.maximum(np.asarray(matrix.sum(axis=1)).reshape(-1), 1.0)
    astro_counts = np.asarray(matrix[:, astro.indices].sum(axis=1)).reshape(-1)
    purity_counts = np.asarray(matrix[:, purity.indices].sum(axis=1)).reshape(-1)
    selected: np.ndarray = (
        (astro_counts >= ASTRO_MIN_COUNTS)
        & (1000.0 * astro_counts / total >= ASTRO_MIN_PER_1000)
        & (1000.0 * purity_counts / total <= ASTRO_MAX_MICROGLIA_PER_1000)
    )
    return selected


@dataclass
class SpilloverResult:
    """``microglia_stat``, ``microglia_weight`` and the flag of the table cells.

    Attributes:
        statistic: Statistic per table cell (NaN when null).
        weight: Spill weight per table cell (NaN when null).
        raw_flag: The rule's decision per table cell (before the null
            switches).
        defined: Whether the flag is defined (false: null).
        genes: The microglia set.
        astro_genes: The Astro-Epen set of the false-positive check.
        null_reason: Why the flag is null, if it is.
        checks: The false-positive and held-out checks (JSON).
    """

    statistic: np.ndarray
    weight: np.ndarray
    raw_flag: np.ndarray
    defined: bool
    genes: SpecificGenes
    astro_genes: SpecificGenes | None = None
    null_reason: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def rate(self) -> float | None:
        """Flag rate over the table cells (``None`` when null)."""
        if not self.defined or not len(self.raw_flag):
            return None
        return float(np.mean(self.raw_flag))


def null_spillover(
    n_cells: int, reason: str, genes: SpecificGenes | None = None
) -> SpilloverResult:
    """Return a null spill-over result with a reason.

    Args:
        n_cells: Table cells.
        reason: The null reason.
        genes: The (insufficient) microglia set, if derived.

    Returns:
        The result.
    """
    return SpilloverResult(
        statistic=np.full(n_cells, np.nan),
        weight=np.full(n_cells, np.nan),
        raw_flag=np.zeros(n_cells, dtype=bool),
        defined=False,
        genes=genes if genes is not None else _empty_set("microglia"),
        null_reason=reason,
    )


def spillover_reference(profiles: MouseFlagProfiles) -> tuple[np.ndarray, np.ndarray]:
    """Return the host profiles and the microglia profile of the LRT (E3).

    As e3lib ``load_ref``: every subclass profile is normalised and floored
    1% towards the mean over all subclasses; the hosts are the non-Immune
    subclasses, the target the mean of the microglia subclasses (the
    floored Immune class profile when the tree has none).

    Args:
        profiles: Class and subclass profiles.

    Returns:
        ``(base, target)``: host rows and the target, each summing to 1.

    Raises:
        ValueError: If there is no microglia or Immune node, or no host.
    """
    floored = floored_profiles(profiles.subclass_profiles)
    immune = np.array(
        [_ends_with(name, IMMUNE_CLASS_SUFFIX) for name in profiles.subclass_class],
        dtype=bool,
    )
    microglia = (
        np.array(
            [MICROGLIA_TOKEN in name for name in profiles.subclass_names], dtype=bool
        )
        & immune
    )
    if not (~immune).any():
        raise ValueError("no non-Immune subclass profile to host spill-over")
    if microglia.any():
        target: np.ndarray = floored[microglia].mean(axis=0)
    else:
        row = profiles.class_row(IMMUNE_CLASS_SUFFIX)
        if row is None:
            raise ValueError("no microglia subclass or Immune class profile")
        raw = profiles.class_profiles[row]
        total = float(raw.sum())
        normalised = raw / total if total > 0 else np.zeros_like(raw)
        mean = (
            profiles.subclass_profiles
            / np.maximum(profiles.subclass_profiles.sum(axis=1, keepdims=True), 1e-300)
        ).mean(axis=0)
        target = (1.0 - PROFILE_FLOOR) * normalised + PROFILE_FLOOR * mean
    return floored[~immune], target


def _flag_cells(
    counts: sparse.spmatrix | np.ndarray,
    genes: SpecificGenes,
    base: np.ndarray,
    target: np.ndarray,
    config: AnnotationFlagsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    statistic, weight = spillover_statistic(counts, genes.indices, base, target)
    flag = (statistic >= config.microglia_stat_min) & (
        weight >= config.microglia_weight_min
    )
    return statistic, weight, flag


def _heldout_check(
    counts: sparse.spmatrix | np.ndarray,
    genes: SpecificGenes,
    base: np.ndarray,
    target: np.ndarray,
    config: AnnotationFlagsConfig,
) -> dict[str, Any]:
    """Rebuild the flag on half of the genes; enrichment of the other half."""
    from scipy import sparse as sp

    if len(genes) < MIN_HELDOUT_GENES:
        return {
            "evaluated": False,
            "reason": f"{len(genes)} microglia genes < {MIN_HELDOUT_GENES}",
        }
    build = SpecificGenes(
        target="microglia_build_half",
        indices=genes.indices[0::2],
        gene_ids=genes.gene_ids[0::2],
        symbols=genes.symbols[0::2],
        ratios=genes.ratios[0::2],
    )
    held = genes.indices[1::2]
    held_symbols = list(genes.symbols[1::2])
    if set(build.indices.tolist()) & set(held.tolist()):
        raise ValueError("the held-out genes overlap the build genes")
    _, _, flag = _flag_cells(counts, build, base, target, config)
    matrix = sp.csr_matrix(counts, dtype=np.float64)
    total = np.maximum(np.asarray(matrix.sum(axis=1)).reshape(-1), 1.0)
    held_counts = np.asarray(matrix[:, held].sum(axis=1)).reshape(-1)
    per_1000 = 1000.0 * held_counts / total
    flagged = float(per_1000[flag].mean()) if flag.any() else math.nan
    unflagged = float(per_1000[~flag].mean()) if (~flag).any() else math.nan
    enrichment = (
        flagged / unflagged
        if math.isfinite(flagged) and math.isfinite(unflagged) and unflagged > 0
        else None
    )
    return {
        "evaluated": True,
        "build_genes": list(build.symbols),
        "heldout_genes": held_symbols,
        "build_flag_rate": round(float(flag.mean()), 6),
        "heldout_per_1000_flagged": _rounded(flagged),
        "heldout_per_1000_unflagged": _rounded(unflagged),
        "enrichment": _rounded(enrichment),
        "heldout_detected_flagged": _rounded(
            float((held_counts[flag] > 0).mean()) if flag.any() else math.nan
        ),
        "heldout_detected_unflagged": _rounded(
            float((held_counts[~flag] > 0).mean()) if (~flag).any() else math.nan
        ),
    }


def _rounded(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 6)


def microglial_spillover(
    counts: sparse.spmatrix | np.ndarray | None,
    profiles: MouseFlagProfiles | None,
    config: AnnotationFlagsConfig,
    *,
    heldout: bool = True,
) -> SpilloverResult:
    """Return the microglial spill-over flag of a sample's table cells (§7.4).

    Args:
        counts: Table cells x query genes (``profiles.gene_ids`` order), or
            ``None`` without query counts.
        profiles: The primary bundle's profiles on the query genes.
        config: ``AnnotationConfig.flags``.
        heldout: Run the held-out check (reported only).

    Returns:
        The result (null with a reason when the flag cannot be trusted).
    """
    if counts is None:
        return null_spillover(0, REASON_NO_QUERY)
    n_cells = counts.shape[0]
    if profiles is None:
        return null_spillover(n_cells, REASON_NO_PROFILES)
    genes = microglia_genes(profiles, config)
    if len(genes) < config.min_specific_genes:
        logger.warning(
            "microglial spill-over: %d specific gene(s) on the panel (< %d): null (%s)",
            len(genes),
            config.min_specific_genes,
            REASON_INSUFFICIENT_GENES,
        )
        return null_spillover(n_cells, REASON_INSUFFICIENT_GENES, genes)
    base, target = spillover_reference(profiles)
    statistic, weight, flag = _flag_cells(counts, genes, base, target, config)
    astro = astrocyte_genes(profiles, config)
    checks: dict[str, Any] = {"n_genes": len(genes)}
    defined = True
    reason = None
    purity, basis = astro_purity_genes(profiles, genes)
    free = sorted(set(genes.gene_ids) - set(purity.gene_ids))
    if not len(astro):
        checks["astrocyte_fpr"] = {"evaluated": False, "reason": "no Astro-Epen genes"}
    elif not free:
        checks["astrocyte_fpr"] = {
            "evaluated": False,
            "reason": (
                "the purity genes cover every flag gene (the rate would be 0 by "
                "construction)"
            ),
            "astro_genes": list(astro.symbols),
            "purity_genes": list(purity.symbols),
            "purity_basis": basis,
        }
    else:
        ast = marker_astrocytes(counts, astro, purity)
        n_ast = int(ast.sum())
        fpr = float(flag[ast].mean()) if n_ast else None
        checks["astrocyte_fpr"] = {
            "evaluated": True,
            "astro_genes": list(astro.symbols),
            "purity_genes": list(purity.symbols),
            "purity_basis": basis,
            "n_flag_genes_outside_purity": len(free),
            "n_marker_astrocytes": n_ast,
            "n_flagged": int(flag[ast].sum()),
            "fpr": _rounded(fpr),
            "max": config.microglia_fpr_max,
        }
        if fpr is not None and fpr > config.microglia_fpr_max:
            defined = False
            reason = (
                f"{REASON_ASTRO_FPR}: {fpr:.4f} of {n_ast} marker astrocytes "
                f"flagged > {config.microglia_fpr_max}"
            )
            logger.warning("microglial spill-over: null (%s)", reason)
    if heldout:
        checks["heldout"] = _heldout_check(counts, genes, base, target, config)
    result = SpilloverResult(
        statistic=statistic,
        weight=weight,
        raw_flag=flag,
        defined=defined,
        genes=genes,
        astro_genes=astro,
        null_reason=reason,
        checks=checks,
    )
    logger.info(
        "microglial spill-over: %d genes (%s), flag rate %.4f of %d table cells%s",
        len(genes),
        ", ".join(genes.symbols),
        float(flag.mean()) if n_cells else math.nan,
        n_cells,
        "" if defined else f" (null: {reason})",
    )
    return result


# --------------------------------------------------------------------------
# Region coherence F1 (E7 §3)


def region_coherence(
    xy: np.ndarray, labels: Sequence[object], *, k: int = 30
) -> np.ndarray:
    """Return the share of each cell's ``k`` nearest neighbours with its label.

    Args:
        xy: ``(n, 2)`` coordinates.
        labels: Label per cell (``None`` never matches).
        k: Neighbours (E7: 30).

    Returns:
        The coherence per cell (NaN with fewer than ``k + 1`` cells).
    """
    from scipy.spatial import cKDTree

    points = np.asarray(xy, dtype=np.float64)
    names = np.array(
        [None if value is None else str(value) for value in labels], dtype=object
    )
    n = len(points)
    if n <= k:
        return np.full(n, np.nan)
    _, neighbours = cKDTree(points).query(points, k=k + 1)
    same = names[neighbours[:, 1:]] == names[:, None]
    same &= np.array([value is not None for value in names])[:, None]
    coherence: np.ndarray = same.mean(axis=1)
    return coherence


def class_coherence_summary(
    coherence: np.ndarray,
    classes: Sequence[object],
    reference: Mapping[str, float],
) -> dict[str, dict[str, float | int | None]]:
    """Return per class the section's coherence next to the intrinsic one (§7.2).

    Plan §7.2 step 5's QC item "class coherence vs intrinsic whole-brain
    coherence": the median ``region_coherence`` of the table cells called to
    each class, E7's ``coh_home_median`` of the class on MERFISH
    (``vocab.load_class_home_coherence``) and their ratio. MERFISH is denser
    than most query sections, so the ratio is a relative QC reading, not a
    test.

    Args:
        coherence: ``region_coherence`` per table cell (NaN when undefined).
        classes: WMB class per table cell (``None``: no call).
        reference: Class name to its MERFISH ``coh_home_median``.

    Returns:
        Class name to ``n_cells``, ``median_coherence``,
        ``merfish_home_median`` and ``ratio`` (classes with at least one
        called table cell with a coherence, sorted by name).
    """
    values = np.asarray(coherence, dtype=np.float64)
    names = pd.Series(
        [None if value is None else str(value) for value in classes], dtype=object
    )
    frame = pd.DataFrame({"class": names, "coherence": values})
    frame = frame[frame["class"].notna() & np.isfinite(frame["coherence"])]
    summary: dict[str, dict[str, float | int | None]] = {}
    for name, rows in sorted(frame.groupby("class")["coherence"], key=lambda x: x[0]):
        median = float(rows.median())
        expected = reference.get(str(name))
        summary[str(name)] = {
            "n_cells": len(rows),
            "median_coherence": _rounded(median),
            "merfish_home_median": _rounded(expected),
            "ratio": _rounded(median / expected) if expected else None,
        }
    return summary


@dataclass
class CoherenceResult:
    """F1 of the table cells.

    Attributes:
        coherence: ``region_coherence`` per table cell (NaN when undefined).
        raw_flag: The rule per table cell.
        defined: Whether the flag is defined.
        restricted_classes: The region-restricted classes used.
        null_reason: Why it is null, if it is.
    """

    coherence: np.ndarray
    raw_flag: np.ndarray
    defined: bool
    restricted_classes: tuple[str, ...]
    null_reason: str | None = None

    @property
    def rate(self) -> float | None:
        """Flag rate over the table cells (``None`` when null)."""
        if not self.defined or not len(self.raw_flag):
            return None
        return float(np.mean(self.raw_flag))


def region_incoherent(
    xy: np.ndarray | None,
    classes: Sequence[object],
    class_bp: np.ndarray,
    config: MouseRegionConfig,
    *,
    restricted_classes: Sequence[str] | None = None,
) -> CoherenceResult:
    """Return F1 of a sample's table cells (E7 §3).

    Args:
        xy: Table-cell coordinates (``None``: the flag is null).
        classes: WMB class per table cell (the pruned primary call).
        class_bp: Its bootstrap probability.
        config: ``coherence_k``, ``coherence_min`` and ``coherence_max_conf``.
        restricted_classes: Region-restricted classes (default: the packaged
            ``wmb_region_restricted_classes.csv``).

    Returns:
        The result.
    """
    from merxen.annotation.vocab import load_region_restricted_classes

    restricted = tuple(
        restricted_classes
        if restricted_classes is not None
        else load_region_restricted_classes()
    )
    n = len(classes)
    if xy is None or not np.isfinite(np.asarray(xy, dtype=np.float64)).all():
        return CoherenceResult(
            coherence=np.full(n, np.nan),
            raw_flag=np.zeros(n, dtype=bool),
            defined=False,
            restricted_classes=restricted,
            null_reason=REASON_NO_COORDINATES,
        )
    coherence = region_coherence(xy, classes, k=config.coherence_k)
    in_restricted = np.array(
        [value is not None and str(value) in set(restricted) for value in classes],
        dtype=bool,
    )
    bp = np.asarray(class_bp, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        flag = (
            in_restricted
            & (coherence < config.coherence_min)
            & (np.nan_to_num(bp, nan=0.0) < config.coherence_max_conf)
        )
    return CoherenceResult(
        coherence=coherence,
        raw_flag=np.asarray(flag, dtype=bool),
        defined=bool(np.isfinite(coherence).all()),
        restricted_classes=restricted,
        null_reason=None if np.isfinite(coherence).all() else "too few cells",
    )
