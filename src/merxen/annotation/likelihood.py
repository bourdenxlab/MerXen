"""Count-likelihood typer with a spill-over mixture term (LL variant vii).

A port of E1's ``LLmix_wbctx+platform`` typer (``exp/E1/14_ll_contam.py`` on
``exp/E1/02_ll_runs.py`` and ``research/insilico/04_likelihood_and_eval_lib.py``)
onto the reference bundles' ``profiles.parquet``. The M3 shadow programme uses
it for the LL value test (plan §12 M3 item 6, OD-B8) and the OD-B13 trigger;
M10 builds the production v1.1 typer on it (plan §5.1, §5.3).

Model (E1 (vii)):

- **Leaves** are the reference's leaf nodes (WHB frontal: clusters), each with a
  multinomial profile over the query genes: the node's expected CPM
  (``profiles.parquet``, the conditional log-normal reconstruction) renormalised
  over the query genes, scaled to ``panel_counts_per_cell`` pseudo-counts per
  reference cell, smoothed with a pseudocount and mixed with the global
  background (``background``). Leaves with fewer than ``min_cells`` reference
  cells are dropped.
- **Prior:** uniform over parents (superclusters), then uniform over a parent's
  leaves.
- **Spill-over mixture:** a cell is either pure leaf ``k`` (prior
  ``pure_prior``) or ``(1 - a) P_k + a Q_j`` for a contaminant broad-class
  profile ``Q_j`` (the unweighted mean of that class's leaf profiles) and
  ``a`` in ``alphas``; the leaf posterior marginalises ``j`` and ``a``.
- **Platform term:** per-gene factors re-estimated from the query by five EM
  iterations on the plain (no-mixture) model, applied to every profile. M10's
  specification caps them at +/- 2 log2 after centring on the median
  (``cap_log2``); ``None`` keeps E1's uncapped factors.

Posteriors are aggregated to parent (supercluster), broad class and lineage
through the reference's vocabulary; the typer is uncalibrated (E2 verdict 4).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from merxen.annotation.vocab import (
    HUMAN_BROAD_CLASSES,
    HUMAN_LINEAGE_OF_BROAD_CLASS,
    UNASSIGNED_LABEL,
)

if TYPE_CHECKING:
    from scipy import sparse

    from merxen.annotation.reference import TaxonomyTreeView

logger = logging.getLogger(__name__)

PANEL_COUNTS_PER_CELL: Final = 246.0
PSEUDOCOUNT: Final = 0.5
BACKGROUND: Final = 0.05
MIN_LEAF_CELLS: Final = 20
ALPHAS: Final[tuple[float, ...]] = (0.15, 0.3)
PURE_PRIOR: Final = 0.5
PLATFORM_ITERATIONS: Final = 5
FACTOR_CAP_LOG2: Final = 2.0
CHUNK_SIZE: Final = 5000


@dataclass(frozen=True)
class LikelihoodReference:
    """Leaf profiles and hierarchy of the likelihood typer.

    Attributes:
        gene_ids: Gene IDs of the profile columns (query order).
        leaves: Leaf node labels.
        leaf_names: Leaf node names.
        parents: Parent node names, sorted.
        parent_of_leaf: Index into ``parents`` per leaf.
        broad_of_parent: Broad class per parent (``Mixed/Unknown`` for sinks
            and nodes outside the broad classes).
        profiles: ``leaves x genes`` smoothed multinomial probabilities.
        log_prior: Log prior per leaf.
        contaminant_classes: Broad classes with a contaminant profile.
        contaminants: ``classes x genes`` contaminant profiles.
    """

    gene_ids: tuple[str, ...]
    leaves: tuple[str, ...]
    leaf_names: tuple[str, ...]
    parents: tuple[str, ...]
    parent_of_leaf: np.ndarray
    broad_of_parent: tuple[str, ...]
    profiles: np.ndarray
    log_prior: np.ndarray
    contaminant_classes: tuple[str, ...]
    contaminants: np.ndarray

    @property
    def n_leaves(self) -> int:
        """Return the number of leaves."""
        return len(self.leaves)


def reference_from_profiles(
    profiles: pd.DataFrame,
    tree: TaxonomyTreeView,
    *,
    leaf_level: str,
    parent_level: str,
    broad_of_parent: Mapping[str, str],
    gene_ids: Sequence[str] | None = None,
    panel_counts_per_cell: float = PANEL_COUNTS_PER_CELL,
    pseudocount: float = PSEUDOCOUNT,
    background: float = BACKGROUND,
    min_cells: int = MIN_LEAF_CELLS,
    broad_classes: Sequence[str] = HUMAN_BROAD_CLASSES,
) -> LikelihoodReference:
    """Build the typer's reference from a bundle's ``profiles.parquet``.

    Args:
        profiles: Long profile table (``level, node, node_name, n_cells,
            gene_id, mean_cpm``).
        tree: The bundle's mapping tree (leaf-to-parent membership).
        leaf_level: Level of the leaves (WHB: ``CCN202210140_CLUS``).
        parent_level: Level of the parents (WHB: ``CCN202210140_SUPC``).
        broad_of_parent: Broad class per parent node **label**.
        gene_ids: Query genes (profile columns, in this order); genes the
            profiles lack are an error. ``None`` uses every profiled gene.
        panel_counts_per_cell: Pseudo-counts per reference cell (E1: the
            median Siletti panel counts per nucleus, 246).
        pseudocount: Added to every leaf x gene pseudo-count.
        background: Weight of the global background profile.
        min_cells: Leaves with fewer reference cells are dropped.
        broad_classes: Classes that get a contaminant profile.

    Returns:
        The reference.

    Raises:
        ValueError: If a query gene or a leaf's parent is missing, or no leaf
            remains.
    """
    leaf_rows = profiles[profiles["level"].astype(str) == leaf_level]
    if leaf_rows.empty:
        raise ValueError(f"profiles have no rows at level {leaf_level!r}")
    wide = leaf_rows.pivot_table(
        index="node", columns="gene_id", values="mean_cpm", aggfunc="first"
    )
    genes = (
        [str(gene) for gene in wide.columns]
        if gene_ids is None
        else [str(gene) for gene in gene_ids]
    )
    missing = [gene for gene in genes if gene not in wide.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} query gene(s) have no profile, e.g. {missing[:5]}"
        )
    meta = leaf_rows.drop_duplicates("node").set_index("node")
    ancestors = tree.ancestors_of_leaves().get(parent_level)
    if ancestors is None and parent_level != leaf_level:
        raise ValueError(f"level {parent_level!r} is not above {leaf_level!r}")
    assert ancestors is not None
    leaves: list[str] = []
    leaf_names: list[str] = []
    parent_labels: list[str] = []
    cpm_rows: list[np.ndarray] = []
    n_cells: list[float] = []
    for node in sorted(wide.index.astype(str)):
        count = float(meta.at[node, "n_cells"])
        if count < min_cells:
            continue
        if node not in ancestors:
            raise ValueError(f"leaf {node!r} has no {parent_level} parent")
        leaves.append(node)
        leaf_names.append(str(meta.at[node, "node_name"]))
        parent_labels.append(ancestors[node])
        cpm_rows.append(
            np.nan_to_num(wide.loc[node, genes].to_numpy(np.float64), nan=0.0)
        )
        n_cells.append(count)
    if not leaves:
        raise ValueError(f"no leaf has >= {min_cells} reference cells")
    cpm = np.vstack(cpm_rows)
    totals = cpm.sum(axis=1, keepdims=True)
    fraction = np.divide(cpm, totals, out=np.zeros_like(cpm), where=totals > 0)
    pseudo = fraction * (np.asarray(n_cells)[:, None] * panel_counts_per_cell)
    smoothed = (pseudo + pseudocount) / (pseudo + pseudocount).sum(
        axis=1, keepdims=True
    )
    global_profile = pseudo.sum(axis=0) / max(float(pseudo.sum()), 1e-300)
    probabilities = (1.0 - background) * smoothed + background * global_profile[None, :]
    parent_names = [tree.name(parent_level, label) for label in parent_labels]
    parents = sorted(set(parent_names))
    parent_of_leaf = np.array([parents.index(name) for name in parent_names])
    name_to_label = dict(zip(parent_names, parent_labels, strict=True))
    parent_broad = tuple(
        str(broad_of_parent.get(name_to_label[name], UNASSIGNED_LABEL))
        for name in parents
    )
    n_children = np.bincount(parent_of_leaf, minlength=len(parents))
    log_prior = -np.log(len(parents)) - np.log(n_children[parent_of_leaf])
    leaf_broad = np.array([parent_broad[index] for index in parent_of_leaf])
    classes = tuple(cls for cls in broad_classes if (leaf_broad == cls).any())
    contaminants = np.vstack(
        [probabilities[leaf_broad == cls].mean(axis=0) for cls in classes]
    )
    return LikelihoodReference(
        gene_ids=tuple(genes),
        leaves=tuple(leaves),
        leaf_names=tuple(leaf_names),
        parents=tuple(parents),
        parent_of_leaf=parent_of_leaf,
        broad_of_parent=parent_broad,
        profiles=probabilities,
        log_prior=log_prior,
        contaminant_classes=classes,
        contaminants=contaminants,
    )


def _csr(counts: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    from scipy import sparse as sp

    return sp.csr_matrix(counts, dtype=np.float64)


def platform_log2_factors(
    counts: sparse.spmatrix | np.ndarray,
    reference: LikelihoodReference,
    *,
    n_iter: int = PLATFORM_ITERATIONS,
    cap_log2: float | None = FACTOR_CAP_LOG2,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-gene platform factors by EM on the plain model (E1).

    Each iteration computes the leaf posterior of every cell, the expected
    gene totals ``sum_c n_c * sum_k post_ck P_kg`` and the factor
    ``(observed + 1) / (expected + 1)``, then rescales and renormalises the
    profiles; the returned factor is the product over iterations.

    Args:
        counts: Cells x query genes.
        reference: The reference (same gene order).
        n_iter: EM iterations (E1: 5).
        cap_log2: Cap on the median-centred log2 factor (M10: 2); ``None``
            returns E1's uncapped factors (centred).

    Returns:
        ``(log2_factors, uncapped_log2_factors)``, both median-centred (a
        common factor cancels when the profiles are renormalised).

    Raises:
        ValueError: If the counts do not have one column per reference gene.
    """
    matrix = _csr(counts)
    if matrix.shape[1] != len(reference.gene_ids):
        raise ValueError("counts must have one column per reference gene")
    profiles = reference.profiles.copy()
    log_factor = np.zeros(matrix.shape[1])
    depth = np.asarray(matrix.sum(axis=1)).ravel()
    observed = np.asarray(matrix.sum(axis=0)).ravel()
    for _ in range(n_iter):
        loglik = np.asarray(matrix @ np.log(profiles).T) + reference.log_prior[None, :]
        loglik -= loglik.max(axis=1, keepdims=True)
        posterior = np.exp(loglik)
        posterior /= posterior.sum(axis=1, keepdims=True)
        expected = (posterior * depth[:, None]).sum(axis=0) @ profiles
        factor = (observed + 1.0) / (expected + 1.0)
        log_factor += np.log2(factor)
        profiles = profiles * factor[None, :]
        profiles /= profiles.sum(axis=1, keepdims=True)
    uncapped = log_factor - np.median(log_factor)
    capped = uncapped if cap_log2 is None else np.clip(uncapped, -cap_log2, cap_log2)
    return capped, uncapped


def _scaled(profiles: np.ndarray, log2_factors: np.ndarray | None) -> np.ndarray:
    if log2_factors is None:
        return profiles
    scaled: np.ndarray = profiles * np.power(2.0, log2_factors)[None, :]
    normalised: np.ndarray = scaled / scaled.sum(axis=1, keepdims=True)
    return normalised


@dataclass(frozen=True)
class MixturePosterior:
    """Posterior of the mixture typer.

    Attributes:
        leaf: ``cells x leaves`` posterior (mixture state marginalised).
        mix_fraction: Posterior mean spill-over fraction ``a`` per cell.
        mix_partner: Most probable contaminant class given contamination
            (``None`` for cells with no contaminant mass).
    """

    leaf: np.ndarray
    mix_fraction: np.ndarray
    mix_partner: np.ndarray


def mixture_posterior(
    counts: sparse.spmatrix | np.ndarray,
    reference: LikelihoodReference,
    *,
    log2_factors: np.ndarray | None = None,
    alphas: Sequence[float] = ALPHAS,
    pure_prior: float = PURE_PRIOR,
    chunk_size: int = CHUNK_SIZE,
) -> MixturePosterior:
    """Return the leaf posterior of the spill-over mixture model (E1 (vii)).

    Args:
        counts: Cells x query genes.
        reference: The reference (same gene order).
        log2_factors: Platform factors applied to every profile (``None``:
            no platform term).
        alphas: Spill-over fractions.
        pure_prior: Prior of the pure (uncontaminated) state.
        chunk_size: Cells per block.

    Returns:
        The posterior.

    Raises:
        ValueError: If the counts do not fit the reference or ``pure_prior``
            is not in (0, 1).
    """
    if not 0.0 < pure_prior < 1.0:
        raise ValueError("pure_prior must be in (0, 1)")
    matrix = _csr(counts)
    if matrix.shape[1] != len(reference.gene_ids):
        raise ValueError("counts must have one column per reference gene")
    profiles = _scaled(reference.profiles, log2_factors)
    contaminants = _scaled(reference.contaminants, log2_factors)
    n_leaves = reference.n_leaves
    n_classes = len(reference.contaminant_classes)
    components = [np.log(profiles)]
    priors = [np.log(pure_prior) + reference.log_prior]
    component_alpha = [0.0]
    component_class = [-1]
    mixed_prior = np.log((1.0 - pure_prior) / (len(alphas) * n_classes))
    for alpha in alphas:
        for index in range(n_classes):
            components.append(
                np.log((1.0 - alpha) * profiles + alpha * contaminants[index][None, :])
            )
            priors.append(mixed_prior + reference.log_prior)
            component_alpha.append(float(alpha))
            component_class.append(index)
    log_profiles = np.vstack(components)
    log_priors = np.concatenate(priors)
    n_components = len(components)
    alpha_of = np.asarray(component_alpha)
    class_of = np.asarray(component_class)
    leaf_blocks, fraction_blocks, partner_blocks = [], [], []
    for start in range(0, matrix.shape[0], chunk_size):
        block = np.asarray(matrix[start : start + chunk_size] @ log_profiles.T)
        block = block + log_priors[None, :]
        block = block.reshape(block.shape[0], n_components, n_leaves)
        peak = block.max(axis=(1, 2), keepdims=True)
        weights = np.exp(block - peak)
        total = weights.sum(axis=(1, 2), keepdims=True)
        weights /= total
        leaf_blocks.append(weights.sum(axis=1))
        per_component = weights.sum(axis=2)
        fraction_blocks.append(per_component @ alpha_of)
        by_class = np.zeros((per_component.shape[0], max(n_classes, 1)))
        for component in range(1, n_components):
            by_class[:, class_of[component]] += per_component[:, component]
        partner = np.full(per_component.shape[0], None, dtype=object)
        has_mass = by_class.sum(axis=1) > 0
        if n_classes:
            best = by_class.argmax(axis=1)
            names = np.asarray(reference.contaminant_classes, dtype=object)
            partner[has_mass] = names[best[has_mass]]
        partner_blocks.append(partner)
    if not leaf_blocks:
        return MixturePosterior(
            leaf=np.zeros((0, n_leaves)),
            mix_fraction=np.zeros(0),
            mix_partner=np.zeros(0, dtype=object),
        )
    return MixturePosterior(
        leaf=np.vstack(leaf_blocks),
        mix_fraction=np.concatenate(fraction_blocks),
        mix_partner=np.concatenate(partner_blocks),
    )


def summarize_posterior(
    posterior: np.ndarray,
    reference: LikelihoodReference,
    *,
    lineage_of_broad: Mapping[str, str] = HUMAN_LINEAGE_OF_BROAD_CLASS,
) -> pd.DataFrame:
    """Aggregate a leaf posterior to parent, broad class and lineage.

    Args:
        posterior: ``cells x leaves`` posterior.
        reference: The reference.
        lineage_of_broad: Lineage per broad class.

    Returns:
        One row per cell: ``ll_cluster_name`` / ``_post`` (argmax leaf),
        ``ll_supercluster_name`` / ``_post``, ``ll_broad_name`` / ``_post``
        and ``ll_lineage_name`` / ``_post`` (the argmax of the aggregated
        posterior at each level; parents outside the broad classes count as
        ``Mixed/Unknown``).
    """
    n_parents = len(reference.parents)
    parent_post = np.zeros((posterior.shape[0], n_parents))
    for index in range(n_parents):
        parent_post[:, index] = posterior[:, reference.parent_of_leaf == index].sum(
            axis=1
        )
    broad_names = sorted(set(reference.broad_of_parent))
    broad_index = np.array(
        [broad_names.index(name) for name in reference.broad_of_parent]
    )
    broad_post = np.zeros((posterior.shape[0], len(broad_names)))
    for index in range(len(broad_names)):
        broad_post[:, index] = parent_post[:, broad_index == index].sum(axis=1)
    lineage_names = sorted(
        {lineage_of_broad.get(name, UNASSIGNED_LABEL) for name in broad_names}
    )
    lineage_index = np.array(
        [
            lineage_names.index(lineage_of_broad.get(name, UNASSIGNED_LABEL))
            for name in broad_names
        ]
    )
    lineage_post = np.zeros((posterior.shape[0], len(lineage_names)))
    for index in range(len(lineage_names)):
        lineage_post[:, index] = broad_post[:, lineage_index == index].sum(axis=1)
    leaf_best = posterior.argmax(axis=1)
    parent_best = parent_post.argmax(axis=1)
    broad_best = broad_post.argmax(axis=1)
    lineage_best = lineage_post.argmax(axis=1)
    rows = np.arange(posterior.shape[0])
    return pd.DataFrame(
        {
            "ll_cluster_name": np.asarray(reference.leaf_names, dtype=object)[
                leaf_best
            ],
            "ll_cluster_post": posterior[rows, leaf_best],
            "ll_supercluster_name": np.asarray(reference.parents, dtype=object)[
                parent_best
            ],
            "ll_supercluster_post": parent_post[rows, parent_best],
            "ll_broad_name": np.asarray(broad_names, dtype=object)[broad_best],
            "ll_broad_post": broad_post[rows, broad_best],
            "ll_lineage_name": np.asarray(lineage_names, dtype=object)[lineage_best],
            "ll_lineage_post": lineage_post[rows, lineage_best],
        }
    )


@dataclass(frozen=True)
class LikelihoodResult:
    """Calls of the LL (vii) typer on one query.

    Attributes:
        calls: ``summarize_posterior`` output plus ``ll_mix_fraction`` and
            ``ll_mix_partner``, indexed like the query cells.
        log2_factors: Platform factors applied (capped).
        uncapped_log2_factors: The factors before the cap.
    """

    calls: pd.DataFrame
    log2_factors: np.ndarray
    uncapped_log2_factors: np.ndarray


def run_ll_vii(
    counts: sparse.spmatrix | np.ndarray,
    reference: LikelihoodReference,
    *,
    cell_ids: Sequence[str] | pd.Index | None = None,
    cap_log2: float | None = FACTOR_CAP_LOG2,
    chunk_size: int = CHUNK_SIZE,
) -> LikelihoodResult:
    """Type a query with LL (vii): platform factors, then the mixture model.

    Args:
        counts: Cells x query genes (the reference's gene order).
        reference: The reference.
        cell_ids: Index of the result (default: positions).
        cap_log2: Factor cap (``None``: E1's uncapped factors).
        chunk_size: Cells per block.

    Returns:
        The calls and the factors.
    """
    factors, uncapped = platform_log2_factors(counts, reference, cap_log2=cap_log2)
    posterior = mixture_posterior(
        counts, reference, log2_factors=factors, chunk_size=chunk_size
    )
    calls = summarize_posterior(posterior.leaf, reference)
    calls["ll_mix_fraction"] = posterior.mix_fraction
    calls["ll_mix_partner"] = posterior.mix_partner
    if cell_ids is not None:
        calls.index = pd.Index([str(cell) for cell in cell_ids], name="cell_id")
    n_capped = int(np.count_nonzero(np.abs(uncapped) > np.abs(factors) + 1e-12))
    logger.info(
        "LL (vii): %d cells x %d genes, %d leaves; %d factor(s) capped at %s log2",
        calls.shape[0],
        len(reference.gene_ids),
        reference.n_leaves,
        n_capped,
        cap_log2,
    )
    return LikelihoodResult(
        calls=calls, log2_factors=factors, uncapped_log2_factors=uncapped
    )
