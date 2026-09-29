"""Whole-section QC embedding and branch embeddings for map_first (plan §6.3).

In map_first mode no clustering resolution decides the hierarchy: branches
and leaves come from the reference mapping (``map_first``). The embedding
here is for UMAP, QC and the stability diagnostic only:

- ``normalize_table_cells`` repeats legacy ``run_scanpy_clustering``'s
  preprocessing of the table cells (``min_cells`` gene filter, raw counts in
  ``layers["counts"]``, ``normalize_total``, ``log1p``);
- ``whole_section_qc`` computes PCA, the kNN graph, UMAP and one Leiden
  partition at 0.5 with the CPU engine (scanpy ``flavor="igraph"``,
  ``n_iterations=2``, recorded seed; plan §6.1), written to
  ``obs["leiden"]`` and ``obs["leiden_broad"]``. Panels above
  ``large_panel_genes`` (5K) run PCA on the top ``n_top_genes`` highly
  variable genes;
- ``branch_embedding`` gives one branch its own kNN graph and UMAP on the
  whole-section PCA, for per-branch QC plots.

Only numpy and pandas are imported at module level; scanpy and anndata are
imported inside the functions, so ``merxen.clustering`` stays importable
without them (plan §11.2, test-enforced).
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

QC_LEIDEN_KEY: Final = "leiden"
QC_BROAD_LEIDEN_KEY: Final = "leiden_broad"
QC_LEIDEN_RESOLUTION: Final = 0.5
COUNTS_LAYER: Final = "counts"
# The CPU engine of legacy clustering (clustering_squidpy.CPU_LEIDEN_*): one
# Leiden engine for every partition that reaches outputs (plan §6.1).
CPU_LEIDEN_ENGINE: Final = "scanpy-igraph"
CPU_EMBEDDING_ENGINE: Final = "scanpy"
CPU_LEIDEN_FLAVOR: Final = "igraph"
CPU_LEIDEN_N_ITERATIONS: Final = 2
CPU_ENGINE_LIBRARIES: Final[tuple[str, ...]] = ("scanpy", "igraph")
# Panels above this many genes (AnnotationPanelConfig.large_panel_genes) run
# the QC PCA on the top LARGE_PANEL_HVG_GENES highly variable genes [L].
LARGE_PANEL_GENES: Final = 1000
LARGE_PANEL_HVG_GENES: Final = 2000


@dataclass(frozen=True)
class EmbeddingParams:
    """Settings of the QC embedding (defaults = ``ClusteringSquidpyConfig``).

    Attributes:
        min_cells: Minimum cells per gene (``filter_genes``).
        n_pcs: Principal components (clipped to the data).
        n_neighbors: kNN size (clipped to the data).
        umap_min_dist: UMAP minimum distance.
        umap_spread: UMAP spread.
        normalize_target_sum: ``normalize_total`` target (``None`` = median).
        normalize_exclude_highly_expressed: ``normalize_total`` option.
        normalize_max_fraction: ``normalize_total`` option.
        large_panel_genes: Above this many genes the PCA uses an HVG subset.
        n_top_genes: Size of that HVG subset.
    """

    min_cells: int = 5
    n_pcs: int = 60
    n_neighbors: int = 30
    umap_min_dist: float = 0.3
    umap_spread: float = 1.0
    normalize_target_sum: float | None = None
    normalize_exclude_highly_expressed: bool = False
    normalize_max_fraction: float = 0.05
    large_panel_genes: int = LARGE_PANEL_GENES
    n_top_genes: int = LARGE_PANEL_HVG_GENES

    @classmethod
    def from_config(cls: type[EmbeddingParams], config: Any) -> EmbeddingParams:
        """Read the embedding settings of a ``ClusteringSquidpyConfig``.

        The top-level settings are used (not the legacy per-round overrides),
        as legacy's whole-section rounds inherit them.

        Args:
            config: A ``ClusteringSquidpyConfig`` (or any object with its
                attribute names).

        Returns:
            The embedding settings.
        """
        return cls(
            min_cells=int(config.min_cells),
            n_pcs=int(config.n_pcs),
            n_neighbors=int(config.n_neighbors),
            umap_min_dist=float(config.umap_min_dist),
            umap_spread=float(config.umap_spread),
            normalize_target_sum=config.normalize_target_sum,
            normalize_exclude_highly_expressed=bool(
                config.normalize_exclude_highly_expressed
            ),
            normalize_max_fraction=float(config.normalize_max_fraction),
        )


@dataclass(frozen=True)
class QcLeidenProvenance:
    """Engine and effective settings of the QC Leiden partition.

    The fields mirror legacy ``LeidenProvenance`` (M0d), so the QC partition
    is recorded exactly like a legacy one.

    Attributes:
        engine: ``"scanpy-igraph"``.
        engine_versions: Library versions keyed by import name.
        embedding_engine: ``"scanpy"``.
        flavor: ``"igraph"``.
        n_iterations: Leiden iteration cap (2).
        random_state: Seed of PCA, neighbors, UMAP and Leiden.
        resolution: Leiden resolution.
        n_pcs_used: Principal components used (0 = no PCA).
        n_neighbors_used: Effective kNN size.
        n_genes_used: Genes the PCA used.
        hvg_subset: Whether the PCA used an HVG subset (large panels).
        n_clusters: Groups of the partition.
        gpu_requested: Always ``False`` (map_first runs on CPU).
        gpu_used: Always ``False``.
        wall_time_s: Seconds for PCA, neighbors, UMAP and Leiden.
    """

    engine: str
    engine_versions: dict[str, str | None]
    embedding_engine: str
    flavor: str
    n_iterations: int
    random_state: int
    resolution: float
    n_pcs_used: int
    n_neighbors_used: int
    n_genes_used: int
    hvg_subset: bool
    n_clusters: int
    gpu_requested: bool = False
    gpu_used: bool = False
    wall_time_s: float = 0.0

    def as_uns_params(self) -> dict[str, str | int | float | bool]:
        """Return the scalars recorded in ``merxen_clustering_params_leiden``.

        Returns:
            h5ad/zarr-safe scalars with the legacy key names; library
            versions as a JSON string.
        """
        return {
            "engine": self.engine,
            "engine_versions": json.dumps(self.engine_versions, sort_keys=True),
            "embedding_engine": self.embedding_engine,
            "leiden_flavor": self.flavor,
            "leiden_n_iterations": int(self.n_iterations),
            "leiden_random_state": int(self.random_state),
            "n_pcs_used": int(self.n_pcs_used),
            "n_neighbors_used": int(self.n_neighbors_used),
            "n_genes_used": int(self.n_genes_used),
            "hvg_subset": bool(self.hvg_subset),
            "gpu_requested": bool(self.gpu_requested),
        }

    def to_json(self, **context: str) -> str:
        """Serialise the provenance plus context as one JSON string.

        Args:
            **context: Extra string fields, e.g. ``key_added``.

        Returns:
            A JSON object string with sorted keys.
        """
        return json.dumps({**asdict(self), **context}, sort_keys=True)


def library_versions(import_names: tuple[str, ...]) -> dict[str, str | None]:
    """Return installed versions for import names, without importing them.

    Args:
        import_names: Top-level import names (``"igraph"`` is resolved through
            its ``python-igraph`` / ``igraph`` distributions).

    Returns:
        Version per name, ``None`` when unknown.
    """
    distributions = {"igraph": ("igraph", "python-igraph")}
    versions: dict[str, str | None] = {}
    for name in import_names:
        version: str | None = None
        for distribution in distributions.get(name, (name,)):
            try:
                version = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        versions[name] = version
    return versions


def normalize_table_cells(adata: ad.AnnData, params: EmbeddingParams) -> None:
    """Preprocess table cells in place, as legacy ``run_scanpy_clustering``.

    Drops genes detected in fewer than ``min_cells`` cells, keeps the raw
    counts in ``layers["counts"]``, then ``normalize_total`` and ``log1p``.

    Args:
        adata: Table cells (controls removed), raw counts in ``X``.
        params: Embedding settings.

    Raises:
        ValueError: If fewer than 3 cells or 2 genes remain.
    """
    import scanpy as sc

    sc.pp.filter_genes(adata, min_cells=int(params.min_cells))
    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError(
            "Too few cells/genes remain after filtering: "
            f"n_obs={adata.n_obs}, n_vars={adata.n_vars}"
        )
    adata.layers[COUNTS_LAYER] = adata.X.copy()
    sc.pp.normalize_total(
        adata,
        target_sum=params.normalize_target_sum,
        exclude_highly_expressed=bool(params.normalize_exclude_highly_expressed),
        max_fraction=float(params.normalize_max_fraction),
        inplace=True,
    )
    sc.pp.log1p(adata)


def pca_gene_mask(adata: ad.AnnData, params: EmbeddingParams) -> np.ndarray:
    """Return the genes the QC PCA uses.

    All genes for panels up to ``large_panel_genes``; above it, the top
    ``n_top_genes`` highly variable genes of the log-normalised data
    (scanpy ``flavor="seurat"``, which needs no extra dependency).

    Args:
        adata: Log-normalised table cells.
        params: Embedding settings.

    Returns:
        Boolean mask over ``var_names``.
    """
    n_genes = int(adata.n_vars)
    if n_genes <= int(params.large_panel_genes) or n_genes <= params.n_top_genes:
        return np.ones(n_genes, dtype=bool)
    import scanpy as sc

    table = sc.pp.highly_variable_genes(
        adata,
        n_top_genes=int(params.n_top_genes),
        flavor="seurat",
        inplace=False,
    )
    return np.asarray(table["highly_variable"].to_numpy(), dtype=bool)


def whole_section_qc(
    adata: ad.AnnData,
    resolution: float = QC_LEIDEN_RESOLUTION,
    seed: int = 0,
    *,
    params: EmbeddingParams | None = None,
) -> QcLeidenProvenance:
    """Compute the whole-section QC embedding and Leiden partition in place.

    PCA (HVG subset for large panels), kNN graph, UMAP and Leiden with the
    CPU igraph engine (``n_iterations=2``, ``directed=False``), all seeded
    with ``seed``. Writes ``obsm["X_pca"]``, ``obsm["X_umap"]``, the
    ``obsp`` graph, ``obs["leiden"]`` and ``obs["leiden_broad"]`` (the same
    partition under the legacy broad-round key). Used for UMAP and QC only,
    never for routing (plan §6.3).

    Args:
        adata: Log-normalised table cells (``normalize_table_cells``).
        resolution: Leiden resolution (``qc_leiden_resolution``, 0.5).
        seed: Random seed.
        params: Embedding settings (defaults when ``None``).

    Returns:
        The engine provenance of the partition.

    Raises:
        ValueError: If fewer than 3 cells or 2 genes are given.
    """
    import scanpy as sc

    settings = params or EmbeddingParams()
    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError(
            "whole_section_qc needs at least 3 cells and 2 genes: "
            f"n_obs={adata.n_obs}, n_vars={adata.n_vars}"
        )
    started = time.perf_counter()
    mask = pca_gene_mask(adata, settings)
    n_genes_used = int(mask.sum())
    max_pcs = min(int(settings.n_pcs), adata.n_obs - 1, n_genes_used - 1)
    effective_neighbors = max(2, min(int(settings.n_neighbors), adata.n_obs - 1))
    if max_pcs > 0:
        sc.pp.pca(
            adata,
            n_comps=max_pcs,
            random_state=int(seed),
            mask_var=None if bool(mask.all()) else mask,
        )
    sc.pp.neighbors(
        adata,
        n_neighbors=effective_neighbors,
        n_pcs=max_pcs if max_pcs > 0 else None,
        random_state=int(seed),
    )
    sc.tl.umap(
        adata,
        min_dist=float(settings.umap_min_dist),
        spread=float(settings.umap_spread),
        random_state=int(seed),
    )
    sc.tl.leiden(
        adata,
        resolution=float(resolution),
        random_state=int(seed),
        key_added=QC_LEIDEN_KEY,
        flavor=CPU_LEIDEN_FLAVOR,
        n_iterations=CPU_LEIDEN_N_ITERATIONS,
        directed=False,
    )
    labels = adata.obs[QC_LEIDEN_KEY].astype(str)
    adata.obs[QC_LEIDEN_KEY] = pd.Categorical(labels.to_numpy())
    adata.obs[QC_BROAD_LEIDEN_KEY] = pd.Categorical(labels.to_numpy())
    provenance = QcLeidenProvenance(
        engine=CPU_LEIDEN_ENGINE,
        engine_versions=library_versions(CPU_ENGINE_LIBRARIES),
        embedding_engine=CPU_EMBEDDING_ENGINE,
        flavor=CPU_LEIDEN_FLAVOR,
        n_iterations=CPU_LEIDEN_N_ITERATIONS,
        random_state=int(seed),
        resolution=float(resolution),
        n_pcs_used=max(int(max_pcs), 0),
        n_neighbors_used=int(effective_neighbors),
        n_genes_used=n_genes_used,
        hvg_subset=not bool(mask.all()),
        n_clusters=int(labels.nunique()),
        wall_time_s=float(time.perf_counter() - started),
    )
    logger.info(
        "QC Leiden: engine=%s flavor=%s n_iterations=%d seed=%d resolution=%s "
        "n_pcs=%d n_neighbors=%d genes=%d%s clusters=%d (%.1f s)",
        provenance.engine,
        provenance.flavor,
        provenance.n_iterations,
        provenance.random_state,
        provenance.resolution,
        provenance.n_pcs_used,
        provenance.n_neighbors_used,
        provenance.n_genes_used,
        " (HVG subset)" if provenance.hvg_subset else "",
        provenance.n_clusters,
        provenance.wall_time_s,
    )
    return provenance


def branch_embedding(
    adata: ad.AnnData,
    cell_mask: np.ndarray,
    *,
    use_rep: str = "X_pca",
    n_neighbors: int = 30,
    seed: int = 0,
    umap_min_dist: float = 0.3,
    umap_spread: float = 1.0,
) -> ad.AnnData | None:
    """Return one branch's cells with their own kNN graph and UMAP.

    The branch reuses the whole-section embedding ``obsm[use_rep]`` (no new
    PCA), so the per-branch view is cheap and on the same axes as the QC
    embedding. For per-branch QC plots only.

    Args:
        adata: Cells with ``obsm[use_rep]``.
        cell_mask: Boolean mask of the branch's cells.
        use_rep: Embedding the branch graph is built on.
        n_neighbors: kNN size (clipped to the branch size).
        seed: Random seed of the graph and UMAP.
        umap_min_dist: UMAP minimum distance.
        umap_spread: UMAP spread.

    Returns:
        A copy of the branch's cells with ``obsm["X_umap"]`` replaced by the
        branch UMAP, or ``None`` if the branch has fewer than 3 cells.

    Raises:
        KeyError: If ``obsm[use_rep]`` is missing.
        ValueError: If the mask does not match the cells.
    """
    import scanpy as sc

    mask = np.asarray(cell_mask, dtype=bool)
    if mask.shape != (adata.n_obs,):
        raise ValueError(f"cell_mask has shape {mask.shape}, expected ({adata.n_obs},)")
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm has no embedding {use_rep!r}")
    if int(mask.sum()) < 3:
        return None
    branch = adata[mask].copy()
    for key in list(branch.obsp.keys()):
        del branch.obsp[key]
    branch.uns.pop("neighbors", None)
    branch.uns.pop("umap", None)
    sc.pp.neighbors(
        branch,
        n_neighbors=max(2, min(int(n_neighbors), branch.n_obs - 1)),
        use_rep=use_rep,
        random_state=int(seed),
    )
    sc.tl.umap(
        branch,
        min_dist=float(umap_min_dist),
        spread=float(umap_spread),
        random_state=int(seed),
    )
    return branch
