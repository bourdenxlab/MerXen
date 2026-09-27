"""Per-parent marker prefilter for large panels (plan §8.7; M3b stage D).

Above ``large_panel_genes`` (1,000; e.g. Xenium Prime 5K) cell_type_mapper's
marker steps grow with the number of candidate genes: the WMB reference-marker
file is linear in genes (5.9 / 9.4 GB at 500 / 815 genes) and the
query-marker step loads and copies it (peak RSS 20 / 27 GB at 500 / 815
genes). The prefilter restricts marker discovery to a candidate set of at most
``cap`` (2,000) genes, chosen **per parent** (sibling set), never by a global
between-leaf variance ranking, which would drop genes that mark only a few of
the WMB leaves (rare types).

Method (``PREFILTER_VERSION`` 1):

1. For every parent of the marker tree (the root and every node with at
   least two children, after ``--drop_level``), each child's profile is the
   cell-weighted mean of log2(CPM + 1) and, when the precompute holds it,
   the detection fraction (``gt0 / n_cells``) over the child's leaves.
2. For every pair of siblings a gene scores ``|mean_a - mean_b|`` scaled by
   cell_type_mapper's penetrance terms (detection of the higher sibling
   against 0.5, the relative detection contrast against 0.7), and counts
   only above cell_type_mapper's minimum marker thresholds (log2 fold 0.8,
   detection 0.1, contrast 0.1). Each pair keeps its ``2 x markers_per_pair``
   best genes as candidates and needs ``markers_per_pair`` (``n_per_utility``,
   30) of them, as cell_type_mapper's query markers need that many per pair.
3. The parent's genes are ordered by max-min greedy pair coverage: each
   step takes the gene that serves the most of the pairs with the fewest
   chosen candidates (ties: the most pairs still short of
   ``markers_per_pair``, then the total score), so pairs of a rare child are
   served before genes that split many siblings pile up; then by total
   score. At most ``cap`` genes are ordered by coverage.
4. ``k`` is the largest integer such that the union over parents of each
   parent's first ``k`` genes holds at most ``cap`` genes; that union is the
   candidate set passed to ``reference_markers`` (as the panel stub).

The result, with ``k``, the candidate set and its sha256, is written to the
bundle (``marker_prefilter.json``) and the method, cap and version enter
``build_hash``. Whether it is good enough is measured, not assumed:
``annotation-panel-simulate`` maps the resolvability test cells with the
prefiltered and the unfiltered lookup and requires agreement >= 0.95 per
emitted level and class with >= 50 confident calls, and no parent below 5
markers (plan §8.7, NP9).

Opt-in since the M3b 5K measurement (``large_panel_marker_prefilter``
default ``"none"``): on the Xenium Prime 5K Mouse panel (5,006 genes, WMB)
version 1 kept class-level calls (agreement >= 0.994 per class) but not
subclass calls (0.920-0.949 in 13 of 34 classes), and it saved neither
memory nor time (query markers 37.7 GB prefiltered vs 21.3 GB unfiltered,
reference markers 47 min either way). It remains for panels whose
unfiltered marker steps would not fit the PREP reserve.

This module is pure numpy / scipy, so the bundle builders (``reference.py``)
and the simulation (``simulate.py``) share it without import cycles.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

PREFILTER_METHOD: Final = "per_parent_topk_union"
PREFILTER_VERSION: Final = 1
MARKER_PREFILTER_FILE: Final = "marker_prefilter.json"
# Pairs scored per chunk (bounds the pair x gene score matrices).
PAIR_CHUNK: Final = 512


@dataclass(frozen=True)
class PrefilterSettings:
    """Settings of the per-parent prefilter.

    Attributes:
        cap: Largest candidate set (``large_panel_prefilter_cap``).
        markers_per_pair: Markers each sibling pair needs (``n_per_utility``);
            it keeps twice as many candidates.
        min_log2_fold: Smallest ``|mean_a - mean_b|`` (log2(CPM + 1)).
        min_high_detection: Smallest detection of the higher sibling.
        min_detection_contrast: Smallest ``(q_high - q_low) / q_high``.
        full_high_detection: Detection at which the penetrance term saturates.
        full_detection_contrast: Contrast at which the contrast term saturates.
    """

    cap: int = 2000
    markers_per_pair: int = 30
    min_log2_fold: float = 0.8
    min_high_detection: float = 0.1
    min_detection_contrast: float = 0.1
    full_high_detection: float = 0.5
    full_detection_contrast: float = 0.7

    def to_json(self) -> dict[str, Any]:
        """Return the settings as a JSON object."""
        return {
            "cap": self.cap,
            "markers_per_pair": self.markers_per_pair,
            "min_log2_fold": self.min_log2_fold,
            "min_high_detection": self.min_high_detection,
            "min_detection_contrast": self.min_detection_contrast,
            "full_high_detection": self.full_high_detection,
            "full_detection_contrast": self.full_detection_contrast,
        }


@dataclass(frozen=True)
class SiblingSet:
    """The children of one parent and their leaf rows in the precompute.

    Attributes:
        key: Lookup key of the parent (``"None"`` or ``"<level>/<node>"``).
        children: Child labels.
        leaf_rows: Precompute rows of each child's leaves.
    """

    key: str
    children: tuple[str, ...]
    leaf_rows: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ParentOrder:
    """One parent's gene order.

    Attributes:
        key: Lookup key of the parent.
        n_children: Children of the parent.
        n_pairs: Sibling pairs scored.
        n_covering: Genes of the greedy pair-coverage phase.
        order: Gene column indices, best first (all genes).
    """

    key: str
    n_children: int
    n_pairs: int
    n_covering: int
    order: np.ndarray


@dataclass(frozen=True)
class PrefilterResult:
    """The candidate genes of marker discovery.

    Attributes:
        genes: Candidate gene IDs (sorted).
        k: Genes taken from each parent's order.
        n_input_genes: Panel genes the prefilter chose from.
        settings: The settings.
        n_parents: Parents scored.
        per_parent: Per parent: children, pairs, coverage genes and how
            many of its first ``k`` genes are unique to it.
        applied: Whether the candidate set is smaller than the input.
    """

    genes: list[str]
    k: int
    n_input_genes: int
    settings: PrefilterSettings
    n_parents: int
    per_parent: dict[str, dict[str, int]] = field(default_factory=dict)
    applied: bool = True

    @property
    def genes_sha256(self) -> str:
        """sha256 of the newline-joined sorted candidate genes."""
        return hashlib.sha256("\n".join(self.genes).encode("utf-8")).hexdigest()

    def to_json(self) -> dict[str, Any]:
        """Return the record written to ``marker_prefilter.json``."""
        return {
            "method": PREFILTER_METHOD,
            "version": PREFILTER_VERSION,
            "settings": self.settings.to_json(),
            "n_input_genes": self.n_input_genes,
            "n_genes": len(self.genes),
            "k": self.k,
            "applied": self.applied,
            "n_parents": self.n_parents,
            "genes_sha256": self.genes_sha256,
            "genes": list(self.genes),
            "per_parent": dict(self.per_parent),
        }

    def summary(self) -> dict[str, Any]:
        """Return the compact record kept in ``bundle.json``."""
        record = self.to_json()
        record.pop("genes")
        record.pop("per_parent")
        record["file"] = MARKER_PREFILTER_FILE
        return record


def child_profiles(
    n_cells: np.ndarray,
    total: np.ndarray,
    detected: np.ndarray | None,
    sibling_set: SiblingSet,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return the children's mean log2(CPM + 1) and detection per gene.

    Args:
        n_cells: Cells per precompute row.
        total: ``rows x genes`` sum of log2(CPM + 1).
        detected: ``rows x genes`` cells with a count above zero, if known.
        sibling_set: The parent's children and their leaf rows.

    Returns:
        ``children x genes`` means and detection fractions (``None`` without
        ``detected``).
    """
    means = np.zeros((len(sibling_set.children), total.shape[1]), dtype=np.float64)
    detection = None if detected is None else np.zeros_like(means)
    for index, rows in enumerate(sibling_set.leaf_rows):
        selected = np.asarray(rows, dtype=np.int64)
        cells = float(n_cells[selected].sum())
        if cells <= 0:
            continue
        means[index] = total[selected].sum(axis=0) / cells
        if detection is not None and detected is not None:
            detection[index] = detected[selected].sum(axis=0) / cells
    return means, detection


def pair_scores(
    means: np.ndarray,
    detection: np.ndarray | None,
    first: np.ndarray,
    second: np.ndarray,
    settings: PrefilterSettings,
) -> np.ndarray:
    """Return the ``pairs x genes`` marker scores of some sibling pairs.

    Args:
        means: ``children x genes`` mean log2(CPM + 1).
        detection: ``children x genes`` detection fractions, if known.
        first: First child of each pair.
        second: Second child of each pair.
        settings: Thresholds.

    Returns:
        Non-negative scores; zero where a gene is no candidate for the pair.
    """
    fold = np.abs(means[first] - means[second])
    valid = fold >= settings.min_log2_fold
    score = fold
    if detection is not None:
        higher_first = means[first] >= means[second]
        high = np.where(higher_first, detection[first], detection[second])
        low = np.where(higher_first, detection[second], detection[first])
        contrast = (high - low) / np.maximum(high, 1e-12)
        valid &= (high >= settings.min_high_detection) & (
            contrast >= settings.min_detection_contrast
        )
        score = (
            fold
            * np.clip(high / settings.full_high_detection, 0.0, 1.0)
            * np.clip(contrast / settings.full_detection_contrast, 0.0, 1.0)
        )
    return np.where(valid, score, 0.0)


def parent_gene_order(
    key: str,
    means: np.ndarray,
    detection: np.ndarray | None,
    settings: PrefilterSettings,
) -> ParentOrder:
    """Order the genes of one sibling set (greedy pair coverage, then score).

    Args:
        key: Lookup key of the parent.
        means: ``children x genes`` mean log2(CPM + 1).
        detection: ``children x genes`` detection fractions, if known.
        settings: Thresholds and candidates per pair.

    Returns:
        The parent's order over all genes.
    """
    import scipy.sparse as sp

    n_children, n_genes = means.shape
    first, second = np.triu_indices(n_children, 1)
    n_pairs = len(first)
    per_pair = min(2 * settings.markers_per_pair, n_genes)
    totals = np.zeros(n_genes, dtype=np.float64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for start in range(0, n_pairs, PAIR_CHUNK):
        stop = min(start + PAIR_CHUNK, n_pairs)
        score = pair_scores(
            means, detection, first[start:stop], second[start:stop], settings
        )
        totals += score.sum(axis=0)
        if per_pair <= 0:
            continue
        top = np.argpartition(-score, per_pair - 1, axis=1)[:, :per_pair]
        pair_index = np.repeat(np.arange(start, stop), per_pair)
        top_flat = top.ravel()
        keep = score[pair_index - start, top_flat] > 0
        rows.append(pair_index[keep])
        cols.append(top_flat[keep])
    covering: list[int] = []
    if rows and sum(len(part) for part in rows):
        pair_ids = np.concatenate(rows)
        gene_ids = np.concatenate(cols)
        candidates = sp.csr_matrix(
            (np.ones(len(pair_ids)), (pair_ids, gene_ids)), shape=(n_pairs, n_genes)
        )
        by_gene = candidates.tocsc()
        remaining = np.asarray(candidates.sum(axis=1)).ravel()
        need = np.minimum(remaining, settings.markers_per_pair)
        coverage = np.zeros(n_pairs, dtype=np.float64)
        is_open = need > 0
        open_counts = np.asarray(candidates.sum(axis=0)).ravel()
        chosen_mask = np.zeros(n_genes, dtype=bool)
        # Ties: larger total score, then the lower column (deterministic).
        scale = float(n_pairs + 1)
        tie = totals / (totals.max() + 1.0) / scale**2
        while len(covering) < settings.cap and is_open.any():
            # Max-min coverage: serve the pairs with the fewest chosen
            # candidates first (a rare child's pairs are not starved by genes
            # that split many siblings), then the most open pairs.
            lowest = coverage[is_open].min()
            target = np.flatnonzero(is_open & (coverage == lowest))
            target_counts = np.asarray(candidates[target].sum(axis=0)).ravel()
            priority = target_counts + open_counts / scale + tie
            priority[chosen_mask | (target_counts <= 0)] = -1.0
            best = int(np.argmax(priority))
            if priority[best] <= 0:
                break
            covering.append(best)
            chosen_mask[best] = True
            served = by_gene.indices[by_gene.indptr[best] : by_gene.indptr[best + 1]]
            served = served[is_open[served]]
            coverage[served] += 1
            remaining[served] -= 1
            closed = served[
                (coverage[served] >= need[served]) | (remaining[served] <= 0)
            ]
            if len(closed):
                is_open[closed] = False
                open_counts -= np.asarray(candidates[closed].sum(axis=0)).ravel()
    chosen = np.zeros(n_genes, dtype=bool)
    chosen[covering] = True
    rest = np.lexsort((np.arange(n_genes), -totals))
    rest = rest[~chosen[rest]]
    order = np.concatenate([np.asarray(covering, dtype=np.int64), rest]).astype(
        np.int64
    )
    return ParentOrder(
        key=key,
        n_children=int(n_children),
        n_pairs=int(n_pairs),
        n_covering=len(covering),
        order=order,
    )


def union_of_top_k(orders: Sequence[ParentOrder], k: int) -> set[int]:
    """Return the union of each parent's first ``k`` genes (column indices)."""
    selected: set[int] = set()
    for item in orders:
        selected.update(int(value) for value in item.order[:k])
    return selected


def largest_k(orders: Sequence[ParentOrder], cap: int) -> int:
    """Return the largest ``k`` whose union of per-parent top-k is <= ``cap``.

    Args:
        orders: Every parent's order.
        cap: Largest candidate set.

    Returns:
        ``k`` (0 when even one gene per parent exceeds the cap).
    """
    if not orders:
        return 0
    low, high = 0, max(len(item.order) for item in orders)
    while low < high:
        middle = (low + high + 1) // 2
        if len(union_of_top_k(orders, middle)) <= cap:
            low = middle
        else:
            high = middle - 1
    return low


def per_parent_topk_union(
    genes: Sequence[str],
    n_cells: np.ndarray,
    total: np.ndarray,
    detected: np.ndarray | None,
    sibling_sets: Sequence[SiblingSet],
    settings: PrefilterSettings | None = None,
) -> PrefilterResult:
    """Return the prefiltered candidate genes of a marker precompute.

    Args:
        genes: Gene IDs of the columns of ``total`` / ``detected``.
        n_cells: Cells per precompute row.
        total: ``rows x genes`` sum of log2(CPM + 1).
        detected: ``rows x genes`` cells detecting each gene, if known.
        sibling_sets: The parents of the marker tree (``sibling_sets`` in
            ``reference.py``); parents with fewer than two children are
            skipped.
        settings: Prefilter settings (default ``PrefilterSettings()``).

    Returns:
        The candidate set; all genes (``applied`` false) when the input has
        at most ``cap`` genes.
    """
    rule = settings or PrefilterSettings()
    gene_list = [str(gene) for gene in genes]
    if total.shape[1] != len(gene_list):
        raise ValueError("total has a column per gene")
    if len(gene_list) <= rule.cap:
        return PrefilterResult(
            genes=sorted(gene_list),
            k=0,
            n_input_genes=len(gene_list),
            settings=rule,
            n_parents=0,
            applied=False,
        )
    orders: list[ParentOrder] = []
    for sibling_set in sibling_sets:
        if len(sibling_set.children) < 2:
            continue
        means, detection = child_profiles(n_cells, total, detected, sibling_set)
        orders.append(parent_gene_order(sibling_set.key, means, detection, rule))
    k = largest_k(orders, rule.cap)
    selected = union_of_top_k(orders, k)
    owners: dict[int, int] = {}
    for item in orders:
        for value in item.order[:k]:
            owners[int(value)] = owners.get(int(value), 0) + 1
    per_parent = {
        item.key: {
            "n_children": item.n_children,
            "n_pairs": item.n_pairs,
            "n_covering": item.n_covering,
            "n_unique_in_top_k": sum(
                1 for value in item.order[:k] if owners[int(value)] == 1
            ),
        }
        for item in orders
    }
    chosen = sorted(gene_list[index] for index in selected)
    logger.info(
        "marker prefilter: %d of %d genes (k = %d over %d parents, cap %d)",
        len(chosen),
        len(gene_list),
        k,
        len(orders),
        rule.cap,
    )
    return PrefilterResult(
        genes=chosen,
        k=k,
        n_input_genes=len(gene_list),
        settings=rule,
        n_parents=len(orders),
        per_parent=per_parent,
        applied=len(chosen) < len(gene_list),
    )


def settings_from_config(cap: int, markers_per_pair: int) -> PrefilterSettings:
    """Return the prefilter settings of a build (cap, ``n_per_utility``)."""
    return PrefilterSettings(cap=int(cap), markers_per_pair=int(markers_per_pair))


def prefilter_payload(cap: int, markers_per_pair: int) -> dict[str, Any]:
    """Return what the prefilter's output depends on, for ``build_hash``."""
    return {
        "method": PREFILTER_METHOD,
        "version": PREFILTER_VERSION,
        "settings": settings_from_config(cap, markers_per_pair).to_json(),
    }


def read_prefilter_genes(record: Mapping[str, Any]) -> list[str]:
    """Return the candidate genes of a ``marker_prefilter.json`` record."""
    return [str(gene) for gene in record.get("genes", [])]
