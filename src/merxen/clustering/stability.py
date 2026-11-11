"""Subsample-stability diagnostic of the whole-section QC Leiden (plan §6.3).

The map_first hierarchy never lets a clustering resolution decide anything:
the whole-section Leiden at 0.5 is for UMAP and QC only. This module reports
how stable that partition is: Leiden is re-run on ``n_subsamples`` random
subsamples of ``fraction`` of the cells (their own kNN graph on the same
embedding) and each subsample partition is compared with the full partition
restricted to the same cells by the adjusted Rand index (ARI). The result is
reported, **never** used to choose a resolution: the "largest resolution
with ARI >= 0.8" rule overshoots the corrected optimum by 0-4 grid steps
(E6).

The ARI comes from the contingency table of the two partitions, so one
comparison costs O(N + T), where T is the number of non-empty cells of the
table, instead of the O(N^2) pair enumeration (the RESOLUTE prototype,
``research/resolute_eval/fast_stability.py``); it is exact
(``adjusted_rand_index`` equals the pair-counting definition).

Only numpy and pandas are imported at module level; scanpy and anndata are
imported inside ``subsample_ari``, so ``merxen.clustering`` stays importable
without them (plan §11.2).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

STABILITY_N_SUBSAMPLES: Final = 5
STABILITY_FRACTION: Final = 0.8
STABILITY_SEED: Final = 0
# Same CPU Leiden as the QC partition (plan §6.1: one Leiden engine).
STABILITY_LEIDEN_FLAVOR: Final = "igraph"
STABILITY_LEIDEN_N_ITERATIONS: Final = 2


@dataclass(frozen=True)
class Contingency:
    """Contingency table of two partitions of the same objects.

    Attributes:
        counts: Non-empty cell counts ``n_ij`` (one entry per observed
            ``(i, j)`` pair).
        row_sums: Sizes ``a_i`` of the first partition's groups.
        col_sums: Sizes ``b_j`` of the second partition's groups.
        n_objects: Number of objects.
    """

    counts: np.ndarray
    row_sums: np.ndarray
    col_sums: np.ndarray
    n_objects: int


def contingency(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> Contingency:
    """Return the sparse contingency table of two labelings.

    Args:
        labels_a: Group label per object (any hashable values; NaN is a
            group of its own).
        labels_b: Group label per object, aligned with ``labels_a``.

    Returns:
        The non-empty table cells and the marginal group sizes.

    Raises:
        ValueError: If the labelings differ in length.
    """
    codes_a, uniques_a = pd.factorize(np.asarray(labels_a, dtype=object))
    codes_b, uniques_b = pd.factorize(np.asarray(labels_b, dtype=object))
    if codes_a.shape != codes_b.shape:
        raise ValueError(
            f"labelings differ in length: {codes_a.size} vs {codes_b.size}"
        )
    # pd.factorize gives -1 to missing values; keep them as one extra group
    # so every object is counted exactly once.
    n_a = len(uniques_a) + 1
    n_b = len(uniques_b) + 1
    codes_a = np.where(codes_a < 0, n_a - 1, codes_a).astype(np.int64)
    codes_b = np.where(codes_b < 0, n_b - 1, codes_b).astype(np.int64)
    if codes_a.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Contingency(empty, empty, empty, 0)
    cells = np.bincount(codes_a * n_b + codes_b, minlength=n_a * n_b)
    rows = np.bincount(codes_a, minlength=n_a)
    cols = np.bincount(codes_b, minlength=n_b)
    return Contingency(
        counts=cells[cells > 0].astype(np.int64),
        row_sums=rows[rows > 0].astype(np.int64),
        col_sums=cols[cols > 0].astype(np.int64),
        n_objects=int(codes_a.size),
    )


def _sum_squares(values: np.ndarray) -> int:
    """Return the exact sum of squares of non-negative integers."""
    return sum(int(value) * int(value) for value in values)


def adjusted_rand_index(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> float:
    """Return the adjusted Rand index of two partitions in O(N + T).

    Uses the pair confusion counts derived from the contingency table (the
    formula of ``sklearn.metrics.adjusted_rand_score``), with exact integer
    arithmetic, so it equals the O(N^2) pair-counting definition.

    Args:
        labels_a: Group label per object.
        labels_b: Group label per object, aligned with ``labels_a``.

    Returns:
        The ARI; 1.0 when the partitions put exactly the same pairs together
        (including two single-group or two all-singleton partitions and an
        empty input).
    """
    table = contingency(labels_a, labels_b)
    n = table.n_objects
    sum_cells = _sum_squares(table.counts)
    sum_rows = _sum_squares(table.row_sums)
    sum_cols = _sum_squares(table.col_sums)
    # Ordered pairs of distinct objects: together in both, only in b, only
    # in a, and in neither (sklearn's pair_confusion_matrix).
    together_both = sum_cells - n
    only_b = sum_cols - sum_cells
    only_a = sum_rows - sum_cells
    neither = n * n - sum_rows - sum_cols + sum_cells
    if only_a == 0 and only_b == 0:
        return 1.0
    numerator = 2 * (together_both * neither - only_a * only_b)
    denominator = (together_both + only_a) * (only_a + neither) + (
        together_both + only_b
    ) * (only_b + neither)
    return float(numerator / denominator)


def cluster_pair_consistency(
    reference: Sequence[Any], subsample_labels: Sequence[Any]
) -> tuple[pd.Series, pd.Series]:
    """Return, per reference group, its within-group pairs kept together.

    Args:
        reference: Reference group per object of the subsample.
        subsample_labels: Subsample partition of the same objects.

    Returns:
        ``(kept_pairs, total_pairs)`` indexed by reference group (as str):
        ordered pairs of distinct objects of the group that the subsample
        partition also puts together, and all ordered pairs of the group.
        Summing both over subsamples gives the ratio-of-sums estimator of
        the RESOLUTE prototype.
    """
    frame = pd.DataFrame(
        {
            "reference": pd.Series(np.asarray(reference, dtype=object)).astype(str),
            "subsample": pd.Series(np.asarray(subsample_labels, dtype=object)).astype(
                str
            ),
        }
    )
    cells = frame.groupby(["reference", "subsample"], sort=True).size()
    kept = (cells * (cells - 1)).groupby(level="reference").sum()
    sizes = frame.groupby("reference", sort=True).size()
    total = sizes * (sizes - 1)
    return kept.reindex(total.index, fill_value=0).astype(np.int64), total.astype(
        np.int64
    )


@dataclass(frozen=True)
class SubsampleStability:
    """Stability of a partition across random subsamples (plan §6.3).

    Attributes:
        reference_key: ``obs`` column of the full-data partition.
        use_rep: Embedding the subsample kNN graphs were built on.
        resolution: Leiden resolution of both partitions.
        n_subsamples: Number of subsamples.
        fraction: Share of the cells in each subsample.
        seed: Seed of the subsample draws and of each re-clustering.
        n_neighbors: kNN size of the subsample graphs.
        n_objects: Cells in the full partition.
        n_objects_per_subsample: Cells in each subsample.
        n_clusters_reference: Groups of the full partition.
        aris: ARI of each subsample partition with the full partition on the
            same cells.
        n_clusters_subsample: Groups of each subsample partition.
        cluster_pair_consistency: Per full-partition group (as str), the
            share of its within-group ordered cell pairs that the subsample
            partitions also put together, pooled over subsamples (ratio of
            sums); groups with fewer than two cells in every subsample are
            omitted.
        wall_time_s: Seconds spent re-clustering.
        role: What the diagnostic is for (always report-only).
    """

    reference_key: str
    use_rep: str
    resolution: float
    n_subsamples: int
    fraction: float
    seed: int
    n_neighbors: int
    n_objects: int
    n_objects_per_subsample: int
    n_clusters_reference: int
    aris: tuple[float, ...]
    n_clusters_subsample: tuple[int, ...]
    cluster_pair_consistency: dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0
    role: str = "report_only"

    @property
    def mean_ari(self) -> float:
        """Return the mean subsample ARI (NaN without subsamples)."""
        return float(np.mean(self.aris)) if self.aris else float("nan")

    @property
    def min_ari(self) -> float:
        """Return the smallest subsample ARI (NaN without subsamples)."""
        return float(np.min(self.aris)) if self.aris else float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dict with the summary statistics added.

        Returns:
            All fields plus ``mean_ari`` and ``min_ari`` (``None`` when there
            is no subsample, as JSON has no NaN).
        """
        payload = asdict(self)
        payload["aris"] = [float(value) for value in self.aris]
        payload["n_clusters_subsample"] = [int(v) for v in self.n_clusters_subsample]
        payload["mean_ari"] = self.mean_ari if self.aris else None
        payload["min_ari"] = self.min_ari if self.aris else None
        return payload

    def to_json(self) -> str:
        """Return the canonical JSON string (safe for h5ad and zarr ``uns``).

        Returns:
            Compact JSON with sorted keys.
        """
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def subsample_indices(
    n_objects: int, *, n_subsamples: int, fraction: float, seed: int
) -> list[np.ndarray]:
    """Return the sorted cell indices of each subsample.

    Args:
        n_objects: Number of cells.
        n_subsamples: Number of subsamples.
        fraction: Share of the cells per subsample, in (0, 1].
        seed: Seed of the draws (one ``numpy`` generator for all of them).

    Returns:
        One sorted index array of ``round(fraction * n_objects)`` cells (at
        least 2 when there are 2 cells) per subsample.

    Raises:
        ValueError: On a fraction outside (0, 1] or a negative count.
    """
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if n_subsamples < 0:
        raise ValueError(f"n_subsamples must be >= 0, got {n_subsamples}")
    size = min(n_objects, max(2, int(round(float(fraction) * n_objects))))
    rng = np.random.default_rng(int(seed))
    return [
        np.sort(rng.choice(n_objects, size=size, replace=False))
        for _ in range(int(n_subsamples))
    ]


def subsample_ari(
    adata: ad.AnnData,
    reference_key: str = "leiden",
    *,
    resolution: float = 0.5,
    n_subsamples: int = STABILITY_N_SUBSAMPLES,
    fraction: float = STABILITY_FRACTION,
    use_rep: str = "X_pca",
    n_neighbors: int = 30,
    seed: int = STABILITY_SEED,
) -> SubsampleStability:
    """Measure the stability of a Leiden partition over random subsamples.

    Each subsample gets its own kNN graph on ``obsm[use_rep]`` and a CPU
    igraph Leiden run (``n_iterations=2``, the QC engine) at ``resolution``;
    its partition is compared with ``obs[reference_key]`` on the same cells.

    Args:
        adata: Cells with the full partition and the embedding.
        reference_key: ``obs`` column of the full-data partition.
        resolution: Leiden resolution (the one that made the reference).
        n_subsamples: Number of subsamples (5 in production).
        fraction: Share of the cells per subsample (0.8 in production).
        use_rep: ``obsm`` embedding for the subsample graphs.
        n_neighbors: kNN size (clipped to the subsample size).
        seed: Seed of the draws and of each re-clustering.

    Returns:
        The stability record (report-only).

    Raises:
        KeyError: If the partition or the embedding is missing.
        ValueError: If there are fewer than 3 cells.
    """
    import anndata as ad
    import scanpy as sc

    if reference_key not in adata.obs:
        raise KeyError(f"adata.obs has no partition {reference_key!r}")
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm has no embedding {use_rep!r}")
    n_objects = int(adata.n_obs)
    if n_objects < 3:
        raise ValueError(f"subsample_ari needs at least 3 cells, got {n_objects}")
    reference = adata.obs[reference_key].astype(str).to_numpy()
    embedding = np.asarray(adata.obsm[use_rep])
    started = time.perf_counter()
    aris: list[float] = []
    n_clusters: list[int] = []
    kept_total: pd.Series | None = None
    pairs_total: pd.Series | None = None
    draws = subsample_indices(
        n_objects, n_subsamples=n_subsamples, fraction=fraction, seed=seed
    )
    size = int(draws[0].size) if draws else 0
    effective_neighbors = max(2, min(int(n_neighbors), max(size - 1, 2)))
    for position, index in enumerate(draws):
        subsample = ad.AnnData(
            obs=pd.DataFrame(index=pd.Index([str(i) for i in range(index.size)])),
            obsm={use_rep: embedding[index]},
        )
        sc.pp.neighbors(
            subsample,
            n_neighbors=effective_neighbors,
            use_rep=use_rep,
            random_state=int(seed),
        )
        sc.tl.leiden(
            subsample,
            resolution=float(resolution),
            random_state=int(seed),
            key_added="subsample",
            flavor=STABILITY_LEIDEN_FLAVOR,
            n_iterations=STABILITY_LEIDEN_N_ITERATIONS,
            directed=False,
        )
        labels = subsample.obs["subsample"].astype(str).to_numpy()
        aris.append(adjusted_rand_index(reference[index], labels))
        n_clusters.append(int(pd.unique(labels).size))
        kept, pairs = cluster_pair_consistency(reference[index], labels)
        kept_total = kept if kept_total is None else kept_total.add(kept, fill_value=0)
        pairs_total = (
            pairs if pairs_total is None else pairs_total.add(pairs, fill_value=0)
        )
        logger.info(
            "stability subsample %d/%d: %d cells, %d clusters, ARI %.3f",
            position + 1,
            len(draws),
            index.size,
            n_clusters[-1],
            aris[-1],
        )
    consistency: dict[str, float] = {}
    if kept_total is not None and pairs_total is not None:
        for group, pairs in pairs_total.items():
            if pairs > 0:
                consistency[str(group)] = float(kept_total[group] / pairs)
    return SubsampleStability(
        reference_key=str(reference_key),
        use_rep=str(use_rep),
        resolution=float(resolution),
        n_subsamples=int(n_subsamples),
        fraction=float(fraction),
        seed=int(seed),
        n_neighbors=int(effective_neighbors),
        n_objects=n_objects,
        n_objects_per_subsample=size,
        n_clusters_reference=int(pd.unique(reference).size),
        aris=tuple(aris),
        n_clusters_subsample=tuple(n_clusters),
        cluster_pair_consistency=consistency,
        wall_time_s=float(time.perf_counter() - started),
    )
