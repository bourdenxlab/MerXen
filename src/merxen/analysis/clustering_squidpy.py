"""Scanpy/Squidpy clustering shim for enriched MerXen SpatialData outputs."""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any, cast

import anndata as ad
import geopandas as gpd
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spatialdata as sd
import squidpy as sq
from scipy import sparse

from merxen.config import ClusteringSquidpyConfig
from merxen.io.transcript_io import first_existing_col
from merxen.memory import force_release, log_status

logger = logging.getLogger(__name__)

GPU_SPARSE_PCA_CHUNK_SIZE = 2048
EXPECTED_PANEL_GENE_COUNT = 300
ENSEMBL_ID_COLUMN = "ensembl_id"
GENE_ID_COLUMN_CANDIDATES = (
    ENSEMBL_ID_COLUMN,
    "gene_ids",
    "gene_id",
    "feature_id",
)
GENE_SYMBOL_COLUMN_CANDIDATES = (
    "gene",
    "feature_name",
    "feature",
    "name",
)
CONTROL_TOKENS = (
    "blank",
    "control",
    "negative",
    "negcontrol",
    "unassigned",
    "deprecated",
)
CONTROL_OUTPUT_COLUMNS = {
    "control_counts",
    "pct_control_counts",
    "control_obs_counts",
    "control_feature_counts",
    "control_obsm_counts",
}
QC_COLUMNS = [
    "total_counts",
    "transcript_counts",
    "n_genes_by_counts",
    "cell_area",
    "nucleus_area",
    "nucleus_ratio",
    "control_counts",
    "pct_control_counts",
    "control_obs_counts",
    "control_feature_counts",
    "control_obsm_counts",
]


def load_spatialdata_adata(
    zarr_path: Path | str,
    *,
    platform: str,
    table_key: str | None = None,
    shape_key: str | None = None,
    gene_id_lookup: dict[str, str] | None = None,
) -> ad.AnnData:
    """Load a SpatialData zarr and return a Squidpy-ready AnnData table.

    The returned object is a copy of the selected table with
    ``.obsm["spatial"]`` populated from the best matching shape centroids when
    needed. If aligned MERSCOPE shapes are present, those centroids are
    preferred so spatial plots use the Xenium reference coordinate system.

    Args:
        zarr_path: Enriched/latest SpatialData zarr path.
        platform: Platform name, used for shape selection metadata.
        table_key: Optional explicit table key. Defaults to ``table`` when
            present.
        shape_key: Optional explicit shape key for spatial coordinates/area.
        gene_id_lookup: Optional shared mapping from gene symbols to Ensembl
            IDs. When present, the returned AnnData gets
            ``.var["ensembl_id"]`` for downstream reference mapping.

    Returns:
        An AnnData object ready for Scanpy/Squidpy analysis.
    """
    zarr_path = Path(zarr_path)
    log_status(f"[{platform}] Loading SpatialData for clustering: {zarr_path}")
    sdata_obj = sd.read_zarr(zarr_path)
    try:
        adata = adata_from_spatialdata(
            sdata_obj,
            platform=platform,
            table_key=table_key,
            shape_key=shape_key,
            gene_id_lookup=gene_id_lookup,
        )
    finally:
        del sdata_obj
        force_release(note=f"after loading clustering input {platform}")
    return adata


def adata_from_spatialdata(
    sdata_obj: Any,
    *,
    platform: str,
    table_key: str | None = None,
    shape_key: str | None = None,
    gene_id_lookup: dict[str, str] | None = None,
) -> ad.AnnData:
    """Extract and annotate an AnnData table from an open SpatialData object."""
    resolved_table_key = _choose_table_key(sdata_obj, table_key)
    table = sdata_obj.tables[resolved_table_key]
    resolved_shape_key = _choose_shape_key(
        sdata_obj,
        platform=platform,
        table=table,
        preferred=shape_key,
    )

    adata = table.copy()
    _normalize_var_names(adata)
    _apply_ensembl_id_metadata(
        adata,
        _merge_gene_id_lookups(
            gene_id_lookup or {},
            _extract_gene_id_lookup_from_spatialdata(sdata_obj),
        ),
    )

    if resolved_shape_key is not None:
        shape_metrics = _shape_metrics(sdata_obj.shapes[resolved_shape_key])
        _apply_shape_metrics(adata, shape_metrics, shape_key=resolved_shape_key)

    nucleus_shape_key = _choose_nucleus_shape_key(sdata_obj)
    if nucleus_shape_key is not None:
        nucleus_metrics = _shape_metrics(sdata_obj.shapes[nucleus_shape_key])
        _apply_area_metric(adata, nucleus_metrics, column="nucleus_area")

    if "spatial" not in adata.obsm:
        raise KeyError(
            "Could not populate adata.obsm['spatial']. "
            f"table_key={resolved_table_key!r}, shape_key={resolved_shape_key!r}"
        )

    add_qc_metrics(adata)
    adata.uns["merxen_clustering_squidpy"] = {
        **dict(adata.uns.get("merxen_clustering_squidpy", {})),
        "platform": str(platform).upper(),
        "table_key": resolved_table_key,
        "shape_key": resolved_shape_key,
        "nucleus_shape_key": nucleus_shape_key,
    }
    return adata


def add_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """Add basic and control-probe QC metrics to ``adata.obs`` in place.

    Scanpy's standard ``total_counts`` and ``n_genes_by_counts`` metrics are
    computed first. Platform controls are then summarized from available
    ``obs`` columns, control-like variables, and MERSCOPE-style ``obsm["blank"]``
    matrices. Missing nucleus measurements are represented by ``NaN`` so the
    MERSCOPE path remains valid until nucleus metrics are added upstream.

    Args:
        adata: AnnData object to annotate.

    Returns:
        The same AnnData object, for convenient notebook chaining.
    """
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None)
    total = _obs_numeric(adata, "total_counts")

    sources: list[str] = []
    control_parts: list[np.ndarray] = []

    obs_cols = _control_obs_columns(adata)
    if obs_cols:
        obs_counts = np.zeros(adata.n_obs, dtype=float)
        for col in obs_cols:
            obs_counts += _obs_numeric(adata, col)
        adata.obs["control_obs_counts"] = obs_counts
        control_parts.append(obs_counts)
        sources.extend([f"obs:{col}" for col in obs_cols])

    feature_mask = _control_feature_mask(adata)
    if feature_mask.any():
        feature_counts = _sum_matrix_rows(adata[:, feature_mask].X)
        adata.obs["control_feature_counts"] = feature_counts
        control_parts.append(feature_counts)
        sources.append("var:control_like_features")

    obsm_counts = _control_obsm_counts(adata)
    if obsm_counts is not None:
        adata.obs["control_obsm_counts"] = obsm_counts
        control_parts.append(obsm_counts)
        sources.append("obsm:blank_or_control")

    if control_parts:
        control_counts = np.sum(np.vstack(control_parts), axis=0)
    else:
        control_counts = np.full(adata.n_obs, np.nan, dtype=float)
    adata.obs["control_counts"] = control_counts
    adata.obs["pct_control_counts"] = np.divide(
        100.0 * control_counts,
        total,
        out=np.full(adata.n_obs, np.nan, dtype=float),
        where=np.isfinite(total) & (total > 0),
    )

    if "cell_area" not in adata.obs:
        adata.obs["cell_area"] = np.nan
    if "nucleus_area" not in adata.obs:
        adata.obs["nucleus_area"] = np.nan
    adata.obs["nucleus_ratio"] = np.divide(
        _obs_numeric(adata, "nucleus_area"),
        _obs_numeric(adata, "cell_area"),
        out=np.full(adata.n_obs, np.nan, dtype=float),
        where=_obs_numeric(adata, "cell_area") > 0,
    )

    adata.uns["merxen_clustering_squidpy"] = {
        **dict(adata.uns.get("merxen_clustering_squidpy", {})),
        "control_qc_sources": sources,
    }
    return adata


def run_scanpy_clustering(
    adata: ad.AnnData,
    *,
    drop_control_features: bool = True,
    min_counts: int = 10,
    min_cells: int = 5,
    normalize_target_sum: float | None = None,
    normalize_exclude_highly_expressed: bool = False,
    normalize_max_fraction: float = 0.05,
    n_pcs: int = 60,
    n_neighbors: int = 30,
    leiden_resolution: float = 0.5,
    umap_min_dist: float = 0.3,
    umap_spread: float = 1.0,
    random_seed: int = 0,
    use_gpu: bool = True,
) -> ad.AnnData:
    """Run the Scanpy preprocessing and clustering workflow.

    Args:
        adata: Input AnnData object.
        drop_control_features: Remove blank/negative/control variables before
            cell, gene, normalization, PCA, and clustering steps.
        min_counts: Minimum transcript counts per cell.
        min_cells: Minimum cells per gene.
        normalize_target_sum: Target sum for normalization (None = median).
        normalize_exclude_highly_expressed: Exclude highly expressed genes from
            size-factor calculation.
        normalize_max_fraction: Max fraction a gene can occupy before being
            excluded (only relevant when exclude_highly_expressed is True).
        n_pcs: Number of principal components.
        n_neighbors: Number of neighbors for the kNN graph.
        leiden_resolution: Leiden clustering resolution.
        umap_min_dist: UMAP minimum distance.
        umap_spread: UMAP spread.
        random_seed: Random seed for reproducibility.
        use_gpu: Use rapids-singlecell GPU-accelerated PCA, neighbors, UMAP,
            and Leiden. Falls back to CPU if rapids-singlecell is not installed.

    Returns:
        Clustered AnnData with ``leiden`` labels in ``.obs``.
    """
    clustered = adata.copy()
    if drop_control_features:
        clustered = remove_control_features(clustered)
    else:
        _record_control_feature_filter(
            clustered,
            removed_features=[],
            n_features_before=clustered.n_vars,
            enabled=False,
        )

    sc.pp.filter_cells(clustered, min_counts=int(min_counts))
    sc.pp.filter_genes(clustered, min_cells=int(min_cells))
    if clustered.n_obs < 3 or clustered.n_vars < 2:
        raise ValueError(
            "Too few cells/genes remain after filtering: "
            f"n_obs={clustered.n_obs}, n_vars={clustered.n_vars}"
        )
    if clustered.n_vars != EXPECTED_PANEL_GENE_COUNT:
        logger.warning(
            "Expected %d genes after control-feature and min-cell filtering; "
            "observed %d.",
            EXPECTED_PANEL_GENE_COUNT,
            clustered.n_vars,
        )
    filter_summary = dict(
        clustered.uns.get("merxen_clustering_squidpy", {}).get(
            "control_feature_filter", {}
        )
    )
    if filter_summary:
        filter_summary["n_features_after_min_cell_filter"] = int(clustered.n_vars)
        filter_summary["has_expected_panel_gene_count_after_min_cell_filter"] = (
            int(clustered.n_vars) == EXPECTED_PANEL_GENE_COUNT
        )
        clustered.uns["merxen_clustering_squidpy"] = {
            **dict(clustered.uns.get("merxen_clustering_squidpy", {})),
            "control_feature_filter": filter_summary,
        }

    clustered.layers["counts"] = clustered.X.copy()
    sc.pp.normalize_total(
        clustered,
        target_sum=normalize_target_sum,
        exclude_highly_expressed=bool(normalize_exclude_highly_expressed),
        max_fraction=float(normalize_max_fraction),
        inplace=True,
    )
    sc.pp.log1p(clustered)

    max_pcs = min(int(n_pcs), clustered.n_obs - 1, clustered.n_vars - 1)
    n_pcs_for_neighbors: int | None = max_pcs if max_pcs > 0 else None
    effective_neighbors = max(2, min(int(n_neighbors), clustered.n_obs - 1))

    gpu_used = False
    if use_gpu:
        gpu_used = _run_gpu_clustering(
            clustered,
            max_pcs=max_pcs,
            n_pcs_for_neighbors=n_pcs_for_neighbors,
            effective_neighbors=effective_neighbors,
            umap_min_dist=float(umap_min_dist),
            umap_spread=float(umap_spread),
            leiden_resolution=float(leiden_resolution),
            random_seed=int(random_seed),
        )

    if not gpu_used:
        if max_pcs > 0:
            sc.pp.pca(clustered, n_comps=max_pcs, random_state=int(random_seed))
        sc.pp.neighbors(
            clustered,
            n_neighbors=effective_neighbors,
            n_pcs=n_pcs_for_neighbors,
            random_state=int(random_seed),
        )
        sc.tl.umap(
            clustered,
            min_dist=float(umap_min_dist),
            spread=float(umap_spread),
            random_state=int(random_seed),
        )
        sc.tl.leiden(
            clustered,
            resolution=float(leiden_resolution),
            random_state=int(random_seed),
            key_added="leiden",
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

    clustered.uns["merxen_clustering_params"] = {
        "drop_control_features": bool(drop_control_features),
        "min_counts": int(min_counts),
        "min_cells": int(min_cells),
        "normalize_target_sum": normalize_target_sum,
        "normalize_exclude_highly_expressed": bool(normalize_exclude_highly_expressed),
        "normalize_max_fraction": float(normalize_max_fraction),
        "n_pcs": int(n_pcs),
        "n_neighbors": int(n_neighbors),
        "effective_neighbors": int(effective_neighbors),
        "leiden_resolution": float(leiden_resolution),
        "umap_min_dist": float(umap_min_dist),
        "umap_spread": float(umap_spread),
        "random_seed": int(random_seed),
        "gpu_used": gpu_used,
    }
    return clustered


def remove_control_features(adata: ad.AnnData) -> ad.AnnData:
    """Return a copy with blank/negative/control variables removed."""
    control_mask = _control_feature_mask(adata)
    removed_features = [str(x) for x in adata.var_names[control_mask]]
    n_features_before = int(adata.n_vars)
    filtered = adata[:, ~control_mask].copy() if control_mask.any() else adata.copy()
    _record_control_feature_filter(
        filtered,
        removed_features=removed_features,
        n_features_before=n_features_before,
        enabled=True,
    )
    return filtered


def _record_control_feature_filter(
    adata: ad.AnnData,
    *,
    removed_features: list[str],
    n_features_before: int,
    enabled: bool,
) -> None:
    retained_features = [str(x) for x in adata.var_names]
    adata.uns["merxen_clustering_squidpy"] = {
        **dict(adata.uns.get("merxen_clustering_squidpy", {})),
        "control_feature_filter": {
            "enabled": bool(enabled),
            "n_features_before": int(n_features_before),
            "n_control_features_removed": len(removed_features),
            "n_features_after_control_filter": int(adata.n_vars),
            "expected_panel_gene_count": EXPECTED_PANEL_GENE_COUNT,
            "has_expected_panel_gene_count": (
                int(adata.n_vars) == EXPECTED_PANEL_GENE_COUNT
            ),
            "removed_control_features": removed_features,
            "retained_features": retained_features,
        },
    }


def _run_gpu_clustering(
    adata: ad.AnnData,
    *,
    max_pcs: int,
    n_pcs_for_neighbors: int | None,
    effective_neighbors: int,
    umap_min_dist: float,
    umap_spread: float,
    leiden_resolution: float,
    random_seed: int,
) -> bool:
    """Run PCA, neighbors, UMAP, and Leiden on GPU via rapids-singlecell.

    Returns True when GPU steps completed successfully, False when
    rapids-singlecell is unavailable and the caller should use CPU instead.
    """
    try:
        import rapids_singlecell as rsc
    except ImportError:
        logger.warning(
            "rapids_singlecell not installed; falling back to CPU clustering. "
            "Install with: pip install -e '.[gpu]' --extra-index-url=https://pypi.nvidia.com"
        )
        return False

    rsc.get.anndata_to_GPU(adata)
    try:
        if max_pcs > 0:
            pca_kwargs = _gpu_pca_kwargs(adata, max_pcs=max_pcs)
            rsc.pp.pca(
                adata,
                n_comps=max_pcs,
                random_state=random_seed,
                **pca_kwargs,
            )
        use_rep = "X_pca" if max_pcs > 0 else None
        rsc.pp.neighbors(
            adata,
            n_neighbors=effective_neighbors,
            n_pcs=n_pcs_for_neighbors,
            use_rep=use_rep,
            random_state=random_seed,
        )
        rsc.tl.umap(
            adata,
            min_dist=umap_min_dist,
            spread=umap_spread,
            random_state=random_seed,
        )
        rsc.tl.leiden(
            adata,
            resolution=leiden_resolution,
            random_state=random_seed,
            key_added="leiden",
        )
    finally:
        rsc.get.anndata_to_CPU(adata)
    return True


def _gpu_pca_kwargs(adata: ad.AnnData, *, max_pcs: int) -> dict[str, int | bool]:
    """Return rapids-singlecell PCA kwargs for the current matrix layout."""
    if not _is_sparse_matrix(adata.X):
        return {}

    chunk_size = min(
        adata.n_obs,
        max(GPU_SPARSE_PCA_CHUNK_SIZE, int(max_pcs) * 4, int(max_pcs) + 1),
    )
    return {"chunked": True, "chunk_size": int(chunk_size)}


def _is_sparse_matrix(matrix: Any) -> bool:
    """Return True for SciPy and CuPy sparse matrices."""
    if sparse.issparse(matrix):
        return True
    try:
        from cupyx.scipy import sparse as cupy_sparse
    except ImportError:
        return False
    return bool(cupy_sparse.issparse(matrix))


def plot_qc_histograms(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    sample_label: str,
    platform: str,
    dpi: int = 160,
) -> Path:
    """Plot transcript/gene/geometry/control QC histograms."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("total_counts", "Transcripts per cell"),
        ("n_genes_by_counts", "Genes per cell"),
        ("cell_area", "Cell area"),
        ("nucleus_ratio", "Nucleus ratio"),
        ("control_counts", "Control/blank counts"),
        ("pct_control_counts", "Control/blank percent"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    for ax, (column, title) in zip(axes.ravel(), panels, strict=True):
        values = _obs_numeric(adata, column)
        finite = values[np.isfinite(values)]
        ax.set_title(title)
        if finite.size == 0:
            ax.text(
                0.5,
                0.5,
                "not available",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            sns.histplot(finite, kde=False, ax=ax)
            ax.set_xlim(1, None)
            ax.set_xlabel(column)
    fig.suptitle(f"{sample_label} ({platform.upper()}) QC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_umap(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    color: list[str] | None = None,
    dpi: int = 160,
) -> Path:
    """Save a Scanpy UMAP plot for the clustered AnnData object."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = color or ["total_counts", "n_genes_by_counts", "leiden"]
    colors = [c for c in colors if c in adata.obs or c in adata.var_names]
    fig = sc.pl.umap(
        adata,
        color=colors,
        wspace=0.4,
        show=False,
        return_fig=True,
    )
    if fig is None:
        fig = plt.gcf()
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_spatial_scatter(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    color: str = "leiden",
    point_size: float = 2.0,
    alpha: float = 0.6,
    dpi: int = 160,
) -> Path:
    """Save a Squidpy spatial scatter plot for the clustered AnnData object."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "spatial" not in adata.obsm:
        raise KeyError("Expected adata.obsm['spatial'] for Squidpy spatial plot.")

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter_kwargs = {
        "shape": None,
        "color": [color],
        "library_id": "",
        "size": float(point_size),
        "edgecolors": "none",
        "linewidths": 0,
        "img": False,
        "ax": ax,
        "return_ax": True,
    }
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No data for colormapping provided via 'c'.*",
                category=UserWarning,
            )
            sq.pl.spatial_scatter(adata, alpha=alpha, **scatter_kwargs)
    except TypeError:
        scatter_kwargs.pop("edgecolors", None)
        scatter_kwargs.pop("linewidths", None)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No data for colormapping provided via 'c'.*",
                category=UserWarning,
            )
            sq.pl.spatial_scatter(adata, **scatter_kwargs)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_qc_metrics(adata: ad.AnnData, output_path: Path | str) -> Path:
    """Write selected per-cell QC metrics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [col for col in QC_COLUMNS if col in adata.obs]
    df = adata.obs.loc[:, columns].copy()
    df.insert(0, "obs_name", adata.obs_names.astype(str))
    if "cell_id" in adata.obs and "cell_id" not in df.columns:
        df.insert(1, "cell_id", adata.obs["cell_id"].astype(str).to_numpy())
    df.to_csv(output_path, index=False)
    return output_path


def save_clustered_adata(adata: ad.AnnData, output_path: Path | str) -> Path:
    """Write the clustered AnnData object to ``.h5ad``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    return output_path


def run_clustering_squidpy(
    config: ClusteringSquidpyConfig,
) -> dict[str, dict[str, Path]]:
    """Run the clustering_squidpy stage for every sample in a pair."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Path]] = {}
    gene_id_lookup = collect_gene_id_lookup_for_samples(config)

    for sample in config.samples:
        sample_dir = config.output_dir / sample.platform.lower()
        sample_dir.mkdir(parents=True, exist_ok=True)
        log_status(
            f"[{sample.sample_id}] Starting clustering_squidpy "
            f"(platform={sample.platform})"
        )
        adata = load_spatialdata_adata(
            sample.zarr_path,
            platform=sample.platform,
            table_key=sample.table_key,
            shape_key=sample.shape_key,
            gene_id_lookup=gene_id_lookup,
        )

        qc_plot = plot_qc_histograms(
            adata,
            sample_dir / f"{sample.sample_id}_qc_histograms.png",
            sample_label=sample.sample_id,
            platform=sample.platform,
            dpi=config.figure_dpi,
        )
        qc_csv = save_qc_metrics(
            adata,
            sample_dir / f"{sample.sample_id}_qc_metrics.csv",
        )

        clustered = run_scanpy_clustering(
            adata,
            drop_control_features=config.drop_control_features,
            min_counts=config.min_counts,
            min_cells=config.min_cells,
            normalize_target_sum=config.normalize_target_sum,
            normalize_exclude_highly_expressed=(
                config.normalize_exclude_highly_expressed
            ),
            normalize_max_fraction=config.normalize_max_fraction,
            n_pcs=config.n_pcs,
            n_neighbors=config.n_neighbors,
            leiden_resolution=config.leiden_resolution,
            umap_min_dist=config.umap_min_dist,
            umap_spread=config.umap_spread,
            random_seed=config.random_seed,
            use_gpu=config.use_gpu,
        )
        umap_plot = plot_umap(
            clustered,
            sample_dir / f"{sample.sample_id}_umap.png",
            dpi=config.figure_dpi,
        )
        spatial_plot = plot_spatial_scatter(
            clustered,
            sample_dir / f"{sample.sample_id}_spatial_scatter_leiden.png",
            point_size=config.spatial_point_size,
            dpi=config.figure_dpi,
        )
        h5ad = save_clustered_adata(
            clustered,
            sample_dir / f"{sample.sample_id}_clustered.h5ad",
        )

        results[sample.sample_id] = {
            "qc_plot": qc_plot,
            "qc_csv": qc_csv,
            "umap_plot": umap_plot,
            "spatial_plot": spatial_plot,
            "h5ad": h5ad,
        }
        del adata, clustered
        force_release(note=f"after clustering_squidpy {sample.sample_id}")

    return results


def collect_gene_id_lookup_for_samples(
    config: ClusteringSquidpyConfig,
) -> dict[str, str]:
    """Collect a shared gene-symbol to Ensembl-ID lookup from sample zarrs.

    Xenium source tables retain Ensembl IDs in ``var["gene_ids"]`` while the
    downstream enriched tables used for clustering can carry gene symbols only.
    Because paired MerXen datasets use the same panel, one platform can provide
    the lookup used to annotate both clustered outputs.
    """
    combined: dict[str, str] = {}
    for sample in config.samples:
        zarr_path = Path(sample.zarr_path)
        if not zarr_path.exists():
            logger.warning(
                "[%s] Cannot inspect gene IDs; zarr path is missing: %s",
                sample.sample_id,
                zarr_path,
            )
            continue
        sdata_obj = sd.read_zarr(zarr_path)
        try:
            sample_lookup = _extract_gene_id_lookup_from_spatialdata(sdata_obj)
        finally:
            del sdata_obj
            force_release(note=f"after collecting gene IDs {sample.sample_id}")
        combined = _merge_gene_id_lookups(combined, sample_lookup)

    if combined:
        logger.info("Collected %d gene symbol -> Ensembl ID mappings.", len(combined))
    else:
        logger.warning("No Ensembl ID metadata found in clustering input zarrs.")
    return combined


def _choose_table_key(sdata_obj: Any, preferred: str | None) -> str:
    if preferred is not None:
        if preferred not in sdata_obj.tables:
            raise KeyError(
                f"Requested table_key={preferred!r} not found. "
                f"Available tables: {list(sdata_obj.tables.keys())}"
            )
        return preferred
    for candidate in ["table", "table_MOSAIK_proseg", "table_cell_boundaries"]:
        if candidate in sdata_obj.tables:
            return candidate
    if len(sdata_obj.tables) == 0:
        raise RuntimeError("SpatialData object has no AnnData tables.")
    return str(list(sdata_obj.tables.keys())[0])


def _choose_shape_key(
    sdata_obj: Any,
    *,
    platform: str,
    table: ad.AnnData,
    preferred: str | None,
) -> str | None:
    if len(sdata_obj.shapes) == 0:
        return None
    if preferred is not None:
        if preferred not in sdata_obj.shapes:
            raise KeyError(
                f"Requested shape_key={preferred!r} not found. "
                f"Available shapes: {list(sdata_obj.shapes.keys())}"
            )
        return preferred

    table_region = _table_region(table)
    if table_region is not None:
        aligned_region = f"{table_region}_aligned_nonrigid"
        if platform.upper() == "MERSCOPE" and aligned_region in sdata_obj.shapes:
            return aligned_region
        if table_region in sdata_obj.shapes:
            return table_region

    candidates = [
        "MOSAIK_proseg_aligned_nonrigid",
        "cell_boundaries_aligned_nonrigid",
        "merscope_cell_boundaries_aligned_nonrigid",
        "MOSAIK_proseg",
        "cell_boundaries",
        "merscope_cell_boundaries",
        "xenium_cell_boundaries",
    ]
    for candidate in candidates:
        if candidate in sdata_obj.shapes:
            return candidate
    for key in sdata_obj.shapes:
        if str(key).endswith("_aligned_nonrigid"):
            return str(key)
    return str(list(sdata_obj.shapes.keys())[0])


def _choose_nucleus_shape_key(sdata_obj: Any) -> str | None:
    if len(sdata_obj.shapes) == 0:
        return None
    candidates = [
        "xenium_nucleus_aligned_nonrigid",
        "nucleus_boundaries_aligned_nonrigid",
        "xenium_nucleus",
        "nucleus_boundaries",
    ]
    for candidate in candidates:
        if candidate in sdata_obj.shapes:
            return candidate
    return None


def _table_region(table: ad.AnnData) -> str | None:
    attrs = dict(table.uns.get("spatialdata_attrs", {}))
    region = attrs.get("region")
    if isinstance(region, str):
        return region
    if isinstance(region, list | tuple) and len(region) > 0:
        return str(region[0])
    return None


def _normalize_var_names(adata: ad.AnnData) -> None:
    if "gene" in adata.var.columns:
        adata.var_names = pd.Index(adata.var["gene"].astype(str), name="gene")
    else:
        adata.var_names = pd.Index(adata.var_names.astype(str), name="gene")
    adata.var_names_make_unique()
    adata.var["gene"] = adata.var_names.astype(str)


def _extract_gene_id_lookup_from_spatialdata(sdata_obj: Any) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for table_key, table in sdata_obj.tables.items():
        table_lookup = _extract_gene_id_lookup_from_var(table.var, table.var_names)
        if table_lookup:
            logger.info(
                "Found %d Ensembl ID mappings in SpatialData table %r.",
                len(table_lookup),
                table_key,
            )
        lookup = _merge_gene_id_lookups(lookup, table_lookup)
    return lookup


def _extract_gene_id_lookup_from_var(
    var: pd.DataFrame,
    var_names: pd.Index,
) -> dict[str, str]:
    id_col = _first_existing_column(var, GENE_ID_COLUMN_CANDIDATES)
    if id_col is None:
        return {}

    gene_ids = var[id_col].astype(str).to_numpy()
    symbol_arrays: list[np.ndarray] = [var_names.astype(str).to_numpy()]
    for col in GENE_SYMBOL_COLUMN_CANDIDATES:
        if col in var.columns:
            symbol_arrays.append(var[col].astype(str).to_numpy())

    lookup: dict[str, str] = {}
    for idx, raw_gene_id in enumerate(gene_ids):
        gene_id = str(raw_gene_id).strip()
        if not gene_id.startswith("ENSG"):
            continue
        for symbols in symbol_arrays:
            symbol = str(symbols[idx]).strip()
            if not symbol or symbol.lower() in {"nan", "none"}:
                continue
            lookup.setdefault(symbol, gene_id)
    return lookup


def _apply_ensembl_id_metadata(
    adata: ad.AnnData,
    gene_id_lookup: dict[str, str],
) -> None:
    existing_col = _first_existing_column(adata.var, GENE_ID_COLUMN_CANDIDATES)
    existing_values = (
        adata.var[existing_col].astype(str).to_numpy()
        if existing_col is not None
        else None
    )
    if existing_values is not None and any(
        str(value).startswith("ENSG") for value in existing_values
    ):
        adata.var[ENSEMBL_ID_COLUMN] = existing_values
        return

    if not gene_id_lookup:
        return

    symbols = (
        adata.var["gene"].astype(str).to_numpy()
        if "gene" in adata.var.columns
        else adata.var_names.astype(str).to_numpy()
    )
    ensembl_ids = [gene_id_lookup.get(str(symbol), "") for symbol in symbols]
    n_mapped = sum(bool(value) for value in ensembl_ids)
    if n_mapped == 0:
        return

    adata.var[ENSEMBL_ID_COLUMN] = pd.Series(
        ensembl_ids,
        index=adata.var_names,
        dtype="object",
    )
    adata.uns["merxen_clustering_squidpy"] = {
        **dict(adata.uns.get("merxen_clustering_squidpy", {})),
        "ensembl_id_mapping": {
            "n_features": int(adata.n_vars),
            "n_mapped": int(n_mapped),
            "column": ENSEMBL_ID_COLUMN,
        },
    }


def _merge_gene_id_lookups(
    left: dict[str, str],
    right: dict[str, str],
) -> dict[str, str]:
    merged = dict(left)
    for symbol, gene_id in right.items():
        if symbol not in merged:
            merged[symbol] = gene_id
        elif merged[symbol] != gene_id:
            logger.warning(
                "Conflicting Ensembl IDs for gene %s: keeping %s, ignoring %s.",
                symbol,
                merged[symbol],
                gene_id,
            )
    return merged


def _first_existing_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _shape_metrics(shapes: gpd.GeoDataFrame) -> pd.DataFrame:
    gdf = shapes.copy()
    if "geometry" not in gdf.columns:
        gdf = gpd.GeoDataFrame({"geometry": gdf.geometry}, index=gdf.index)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    id_col = first_existing_col(
        gdf,
        ["cell_id", "cell", "cells", "cell_ID", "region", "label_id", "EntityID"],
    )
    ids = gdf.index.astype(str) if id_col is None else gdf[id_col].astype(str)
    centroids = _robust_centroid_xy(gdf)
    metrics = pd.DataFrame(
        {
            "cell_id": ids.astype(str).to_numpy(),
            "spatial_x": centroids[0],
            "spatial_y": centroids[1],
            "cell_area": gdf.geometry.area.to_numpy(float),
        },
        index=pd.Index(ids.astype(str), name="cell_id"),
    )
    metrics = metrics[np.isfinite(metrics["spatial_x"])]
    metrics = metrics[np.isfinite(metrics["spatial_y"])]
    return metrics[~metrics.index.duplicated(keep="first")]


def _robust_centroid_xy(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    cent = gdf.geometry.centroid
    bounds = gdf.geometry.bounds
    x = cent.x.to_numpy(float)
    y = cent.y.to_numpy(float)
    minx = bounds["minx"].to_numpy(float)
    miny = bounds["miny"].to_numpy(float)
    maxx = bounds["maxx"].to_numpy(float)
    maxy = bounds["maxy"].to_numpy(float)
    bad = (
        ~np.isfinite(x)
        | ~np.isfinite(y)
        | (x < minx)
        | (x > maxx)
        | (y < miny)
        | (y > maxy)
    )
    if bad.any():
        reps = gdf.geometry.representative_point()
        rx = reps.x.to_numpy(float)
        ry = reps.y.to_numpy(float)
        use_rep = (
            bad
            & np.isfinite(rx)
            & np.isfinite(ry)
            & (rx >= minx)
            & (rx <= maxx)
            & (ry >= miny)
            & (ry <= maxy)
        )
        x[use_rep] = rx[use_rep]
        y[use_rep] = ry[use_rep]
        fallback = bad & ~use_rep
        x[fallback] = (minx[fallback] + maxx[fallback]) / 2.0
        y[fallback] = (miny[fallback] + maxy[fallback]) / 2.0
    return x, y


def _apply_shape_metrics(
    adata: ad.AnnData,
    metrics: pd.DataFrame,
    *,
    shape_key: str,
) -> None:
    table_ids = _table_cell_ids(adata)
    common = pd.Index(table_ids).intersection(metrics.index)

    if len(common) > 0:
        pos = pd.Series(np.arange(adata.n_obs), index=table_ids).loc[common]
        coords = np.full((adata.n_obs, 2), np.nan, dtype=float)
        areas = np.full(adata.n_obs, np.nan, dtype=float)
        coords[pos.to_numpy(), :] = metrics.loc[
            common, ["spatial_x", "spatial_y"]
        ].to_numpy(float)
        areas[pos.to_numpy()] = metrics.loc[common, "cell_area"].to_numpy(float)
        valid = np.isfinite(coords).all(axis=1)
        if valid.all():
            adata.obsm["spatial"] = coords
        elif "spatial" not in adata.obsm:
            raise ValueError(
                f"Only {int(valid.sum())}/{adata.n_obs} cells in table matched "
                f"shape_key={shape_key!r}; cannot populate spatial coordinates."
            )
        if "cell_area" not in adata.obs or adata.obs["cell_area"].isna().all():
            adata.obs["cell_area"] = areas
        if "cell_id" not in adata.obs:
            adata.obs["cell_id"] = table_ids.astype(str).to_numpy()
        return

    if len(metrics) == adata.n_obs:
        adata.obsm["spatial"] = metrics[["spatial_x", "spatial_y"]].to_numpy(float)
        if "cell_area" not in adata.obs or adata.obs["cell_area"].isna().all():
            adata.obs["cell_area"] = metrics["cell_area"].to_numpy(float)
        if "cell_id" not in adata.obs:
            adata.obs["cell_id"] = metrics.index.astype(str).to_numpy()


def _apply_area_metric(
    adata: ad.AnnData,
    metrics: pd.DataFrame,
    *,
    column: str,
) -> None:
    table_ids = _table_cell_ids(adata)
    common = pd.Index(table_ids).intersection(metrics.index)
    if len(common) == 0:
        return

    values = np.full(adata.n_obs, np.nan, dtype=float)
    pos = pd.Series(np.arange(adata.n_obs), index=table_ids).loc[common]
    values[pos.to_numpy()] = metrics.loc[common, "cell_area"].to_numpy(float)
    if column not in adata.obs or adata.obs[column].isna().all():
        adata.obs[column] = values


def _table_cell_ids(adata: ad.AnnData) -> pd.Index:
    for col in ["cell_id", "cell", "cells", "cell_ID", "EntityID"]:
        if col in adata.obs.columns:
            return pd.Index(adata.obs[col].astype(str))
    return pd.Index(adata.obs_names.astype(str))


def _control_obs_columns(adata: ad.AnnData) -> list[str]:
    columns: list[str] = []
    for col in adata.obs.columns:
        col_str = str(col)
        lower = col_str.lower()
        if lower in CONTROL_OUTPUT_COLUMNS:
            continue
        if any(token in lower for token in CONTROL_TOKENS) and "count" in lower:
            columns.append(col_str)
    return columns


def _control_feature_mask(adata: ad.AnnData) -> np.ndarray:
    mask = np.zeros(adata.n_vars, dtype=bool)
    values = [pd.Series(adata.var_names.astype(str), index=adata.var_names)]
    for col in ["gene", "feature_name", "feature_types", "feature_type", "gene_ids"]:
        if col in adata.var.columns:
            values.append(adata.var[col].astype(str))
    for series in values:
        lower = series.astype(str).str.lower()
        mask |= lower.apply(lambda x: any(t in x for t in CONTROL_TOKENS)).to_numpy()
    return mask


def _control_obsm_counts(adata: ad.AnnData) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    for key, value in adata.obsm.items():
        lower = str(key).lower()
        if not any(token in lower for token in CONTROL_TOKENS):
            continue
        parts.append(_sum_obsm_rows(value, n_obs=adata.n_obs))
    if not parts:
        return None
    return _float_array(np.sum(np.vstack(parts), axis=0))


def _sum_obsm_rows(value: Any, *, n_obs: int) -> np.ndarray:
    if isinstance(value, pd.DataFrame):
        numeric = value.apply(pd.to_numeric, errors="coerce").fillna(0)
        return _float_array(numeric.sum(axis=1).to_numpy(dtype=float))
    arr = value
    if sparse.issparse(arr):
        out = _float_array(np.asarray(arr.sum(axis=1)).ravel())
    else:
        out = _float_array(arr)
        if out.ndim == 1:
            return out
        out = _float_array(np.nansum(out, axis=1))
    if len(out) != n_obs:
        raise ValueError(
            f"Control obsm row count mismatch: expected {n_obs}, got {len(out)}"
        )
    return out


def _sum_matrix_rows(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return _float_array(np.asarray(matrix.sum(axis=1)).ravel())
    arr = _float_array(matrix)
    if arr.ndim == 1:
        return arr
    return _float_array(np.nansum(arr, axis=1))


def _float_array(value: Any) -> np.ndarray:
    """Coerce array-like values to a float NumPy array for typed helpers."""
    return cast(np.ndarray, np.asarray(value, dtype=float))


def _obs_numeric(adata: ad.AnnData, column: str) -> np.ndarray:
    if column not in adata.obs:
        return np.full(adata.n_obs, np.nan, dtype=float)
    return _float_array(pd.to_numeric(adata.obs[column], errors="coerce"))
