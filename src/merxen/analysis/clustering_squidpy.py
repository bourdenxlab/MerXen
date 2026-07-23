"""Scanpy/Squidpy clustering shim for enriched MerXen SpatialData outputs."""

# ruff: noqa: E402

from __future__ import annotations

import os

os.environ["MPLCONFIGDIR"] = "./tmp/mpl"
os.environ["NUMBA_CACHE_DIR"] = "./tmp/numba"

import fcntl
import json
import logging
import re
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata as ad
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg", force=True)

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.cluster.hierarchy import leaves_list, linkage

from merxen.config import ClusteringSquidpyConfig
from merxen.io.transcript_io import first_existing_col
from merxen.memory import force_release, log_status
from merxen.plotting import prepare_plot_output, save_figure

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
NEUROTRANSMITTER_LEVEL = "CCN202210140_NEUR"
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
HIERARCHICAL_UNS_KEY = "merxen_hierarchical_clustering"
BROAD_CLUSTER_KEY = "leiden_broad"
BROAD_ATLAS_LABEL_KEY = "broad_atlas_label"
BROAD_CLASS_KEY = "broad_class"
NEURON_SPLIT_KEY = "neuron_split_label"
SUBCLUSTER_LABEL_KEY = "subcluster_label"
HIERARCHICAL_CLUSTER_KEY = "hierarchical_cluster"
UNKNOWN_LABEL = "Mixed/Unknown"
NEURON_CLASS = "Neurons"
NON_NEURON_SUPERCLUSTER_CLASS_MAP = {
    "Oligodendrocyte": "Oligodendrocytes",
    "Committed oligodendrocyte precursor": "Oligodendrocyte precursors",
    "Oligodendrocyte precursor": "Oligodendrocyte precursors",
    "Astrocyte": "Astrocytes",
    "Microglia": "Microglia",
    "Fibroblast": "Fibroblasts",
    "Vascular": "Vascular cells",
}
EXTRA_SUPERCLUSTER_LABELS = {
    "Bergmann glia",
    "Choroid plexus",
    "Ependymal",
    "Miscellaneous",
    "Splatter",
}
NEURON_SUPERCLUSTER_LABELS = {
    "Upper-layer intratelencephalic",
    "Deep-layer intratelencephalic",
    "Deep-layer near-projecting",
    "Deep-layer corticothalamic and 6b",
    "MGE interneuron",
    "CGE interneuron",
    "LAMP5-LHX6 and Chandelier",
    "Hippocampal CA1-3",
    "Hippocampal CA4",
    "Hippocampal dentate gyrus",
    "Amygdala excitatory",
    "Medium spiny neuron",
    "Eccentric medium spiny neuron",
    "Mammillary body",
    "Thalamic excitatory",
    "Midbrain-derived inhibitory",
    "Upper rhombic lip",
    "Cerebellar inhibitory",
    "Lower rhombic lip",
}
INHIBITORY_SUPERCLUSTER_TOKENS = (
    "inhibitory",
    "interneuron",
    "chandelier",
    "medium spiny",
)
EXCITATORY_SUPERCLUSTER_TOKENS = (
    "excitatory",
    "intratelencephalic",
    "corticothalamic",
    "near-projecting",
    "hippocampal",
    "mammillary",
    "rhombic lip",
)
ASSIGNMENT_COLUMNS = [
    "cluster",
    "atlas_label",
    "broad_class",
    "score",
    "runner_up_label",
    "runner_up_score",
    "score_margin",
    "n_markers",
]
SCORE_COLUMNS = [
    "cluster",
    "label_id",
    "atlas_label",
    "broad_class",
    "score",
    "n_markers",
    "resolved_markers",
]
MARKER_COLUMNS = [
    "label_id",
    "label_name",
    "broad_class",
    "neuron_split",
    "n_reference_markers",
    "n_resolved_markers",
    "resolved_markers",
]


@dataclass(frozen=True)
class AtlasMarkerSet:
    """Resolved metadata for one atlas marker set."""

    level: str
    label_id: str
    label_name: str
    broad_class: str
    marker_ids: tuple[str, ...]
    neuron_split: str = ""


@dataclass(frozen=True)
class RoundParams:
    """Effective clustering parameters for one hierarchical round."""

    min_counts: int
    min_cells: int
    n_pcs: int
    n_neighbors: int
    leiden_resolution: float
    umap_min_dist: float
    umap_spread: float


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
    import spatialdata as sd

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
    key_added: str = "leiden",
    input_layer: str | None = None,
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
        key_added: ``obs`` column for Leiden labels.
        input_layer: Optional layer copied into ``X`` before filtering and
            normalization, used when reclustering subsets from raw counts.

    Returns:
        Clustered AnnData with Leiden labels in ``.obs[key_added]``.
    """
    clustered = adata.copy()
    if input_layer is not None:
        if input_layer not in clustered.layers:
            raise KeyError(
                f"Requested input_layer={input_layer!r} not found. "
                f"Available layers: {list(clustered.layers.keys())}"
            )
        clustered.X = _copy_matrix(clustered.layers[input_layer])

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
            key_added=key_added,
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
            key_added=key_added,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

    params = {
        "drop_control_features": bool(drop_control_features),
        "min_counts": int(min_counts),
        "min_cells": int(min_cells),
        "input_layer": input_layer,
        "key_added": key_added,
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
    params_key = f"merxen_clustering_params_{key_added}"
    clustered.uns[params_key] = params
    if key_added == "leiden":
        clustered.uns["merxen_clustering_params"] = params
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
    key_added: str = "leiden",
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
            "Run this command in MerXen's dedicated clustering GPU environment."
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
            key_added=key_added,
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
    output_path = prepare_plot_output(output_path)
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
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
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
    output_path = prepare_plot_output(output_path)
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
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_spatial_scatter(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    color: str = "leiden",
    point_size: float = 2.0,
    alpha: float = 0.6,
    scale_bar_um: float | None = 200.0,
    dpi: int = 160,
) -> Path:
    """Save a spatial scatter plot without requiring SpatialData/Squidpy."""
    output_path = prepare_plot_output(output_path)
    if "spatial" not in adata.obsm:
        raise KeyError("Expected adata.obsm['spatial'] for Squidpy spatial plot.")

    fig, ax = plt.subplots(figsize=(7, 7))
    try:
        import squidpy as sq
    except ImportError:
        sq = None
    if sq is not None:
        import warnings

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
        _clean_spatial_axis(ax)
        if scale_bar_um is not None:
            _add_spatial_scale_bar(ax, length_um=float(scale_bar_um))
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        return output_path

    coordinates = np.asarray(adata.obsm["spatial"])
    values = pd.Series(adata.obs[color], index=adata.obs_names)
    is_categorical = isinstance(values.dtype, pd.CategoricalDtype)
    if is_categorical or not pd.api.types.is_numeric_dtype(values):
        categories = pd.Categorical(values.astype(str))
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=categories.codes,
            cmap="tab20",
            s=float(point_size),
            alpha=float(alpha),
            edgecolors="none",
        )
        category_colormap = plt.get_cmap("tab20")
        category_scale = max(len(categories.categories) - 1, 1)
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=to_hex(category_colormap(index / category_scale)),
                label=str(label),
                markersize=4,
            )
            for index, label in enumerate(categories.categories)
        ]
        if handles:
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=pd.to_numeric(values, errors="coerce"),
            cmap="viridis",
            s=float(point_size),
            alpha=float(alpha),
            edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax, label=color)
    _clean_spatial_axis(ax)
    if scale_bar_um is not None:
        _add_spatial_scale_bar(ax, length_um=float(scale_bar_um))
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def _clean_spatial_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_spatial_scale_bar(
    ax: plt.Axes,
    *,
    length_um: float = 200.0,
    label: str | None = None,
    color: str = "white",
    outline_color: str = "black",
    linewidth: float = 3.0,
    fontsize: float = 9.0,
) -> None:
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    x_min, x_max = min(x_limits), max(x_limits)
    y_min, y_max = min(y_limits), max(y_limits)
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span <= 0 or y_span <= 0 or length_um <= 0:
        return

    pad_x = x_span * 0.06
    pad_y = y_span * 0.06
    text_offset = y_span * 0.025
    x_end = x_max - pad_x
    x_start = x_end - float(length_um)
    if x_start < x_min + pad_x:
        x_start = x_min + pad_x
        x_end = min(x_max - pad_x, x_start + float(length_um))

    y_inverted = y_limits[0] > y_limits[1]
    if y_inverted:
        y_bar = y_max - pad_y
        y_text = y_bar - text_offset
        va = "bottom"
    else:
        y_bar = y_min + pad_y
        y_text = y_bar + text_offset
        va = "bottom"

    outline = [
        path_effects.Stroke(linewidth=linewidth + 2.2, foreground=outline_color),
        path_effects.Normal(),
    ]
    ax.plot(
        [x_start, x_end],
        [y_bar, y_bar],
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
        path_effects=outline,
        zorder=20,
    )
    ax.text(
        (x_start + x_end) / 2.0,
        y_text,
        label or f"{length_um:g} um",
        ha="center",
        va=va,
        color=color,
        fontsize=fontsize,
        path_effects=[
            path_effects.Stroke(linewidth=2.2, foreground=outline_color),
            path_effects.Normal(),
        ],
        zorder=21,
    )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)


def plot_spatial_cluster_grid(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    color: str = "leiden",
    point_size_background: float = 0.08,
    point_size_highlight: float = 0.45,
    alpha_background: float = 0.32,
    alpha_highlight: float = 0.82,
    dpi: int = 160,
) -> Path:
    """Save a spatial small-multiple grid highlighting each de novo cluster."""
    output_path = prepare_plot_output(output_path)
    if "spatial" not in adata.obsm:
        raise KeyError("Expected adata.obsm['spatial'] for spatial cluster grid.")
    if color not in adata.obs:
        raise KeyError(f"Expected adata.obs[{color!r}] for spatial cluster grid.")

    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            "Expected adata.obsm['spatial'] to have at least two columns; "
            f"found shape {coords.shape}."
        )
    labels = pd.Series(adata.obs[color], index=adata.obs_names).astype("string")
    labels = labels.fillna("unassigned")
    categories = [
        str(label)
        for label in labels.value_counts().index
        if str(label) != "unassigned"
    ]
    if not categories:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        _plot_empty_spatial_grid_axis(ax, f"No {color} labels were available.")
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        return output_path

    n_categories = len(categories)
    n_cols = min(4, int(np.ceil(np.sqrt(n_categories))))
    n_rows = int(np.ceil(n_categories / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    x = coords[:, 0]
    y = coords[:, 1]
    x_limits = _padded_limits(x)
    y_limits = _padded_limits(y)
    label_values = labels.astype(str).to_numpy()

    for ax, category in zip(axes.ravel(), categories, strict=False):
        mask = label_values == category
        ax.scatter(
            x,
            y,
            s=float(point_size_background),
            c="#c7c7c7",
            alpha=float(alpha_background),
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            x[mask],
            y[mask],
            s=float(point_size_highlight),
            c="#d7191c",
            alpha=float(alpha_highlight),
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(_wrapped_cluster_title(f"{color} {category}"), fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[n_categories:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.5)
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_group_gene_summary(
    adata: ad.AnnData,
    genes: list[str],
    *,
    groupby: str,
    group_order: list[str] | None = None,
    layer: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return mean expression and fraction expressing for genes grouped by obs."""
    if groupby not in adata.obs:
        raise KeyError(f"Expected adata.obs[{groupby!r}] for gene summary.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Expected adata.layers[{layer!r}] for gene summary.")

    if group_order is None:
        groups = adata.obs[groupby]
        observed = set(groups.astype(str))
        if isinstance(groups.dtype, pd.CategoricalDtype):
            group_order = [
                str(category)
                for category in groups.cat.categories
                if str(category) in observed
            ]
        else:
            group_order = [str(group) for group in pd.unique(groups.astype(str))]
    group_order = [str(group) for group in group_order]

    marker_lookup = _feature_marker_lookup(adata)
    gene_indices: list[int] = []
    resolved_genes: list[str] = []
    seen_indices: set[int] = set()
    for gene in dict.fromkeys(str(gene) for gene in genes):
        idx = marker_lookup.get(_normalize_marker_id(gene))
        if idx is None or idx in seen_indices:
            continue
        gene_indices.append(idx)
        resolved_genes.append(str(adata.var_names[idx]))
        seen_indices.add(idx)
    if not gene_indices:
        raise ValueError("No requested genes were present in the AnnData object.")

    matrix = adata.layers[layer] if layer is not None else adata.X
    expression = matrix[:, gene_indices]
    if sparse.issparse(expression):
        expression = expression.tocsr()
    else:
        expression = np.asarray(expression)

    labels = adata.obs[groupby].astype(str).to_numpy()
    means: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    for group in group_order:
        mask = labels == str(group)
        if not np.any(mask):
            means.append(np.zeros(len(resolved_genes), dtype=float))
            fractions.append(np.zeros(len(resolved_genes), dtype=float))
            continue
        group_expression = expression[mask]
        if sparse.issparse(group_expression):
            means.append(np.asarray(group_expression.mean(axis=0)).ravel())
            fractions.append(np.asarray((group_expression > 0).mean(axis=0)).ravel())
        else:
            means.append(np.asarray(group_expression.mean(axis=0)).ravel())
            fractions.append(np.asarray((group_expression > 0).mean(axis=0)).ravel())

    mean_expression = pd.DataFrame(
        means,
        index=pd.Index(group_order, name=groupby),
        columns=pd.Index(resolved_genes, name="gene"),
    )
    fraction_expression = pd.DataFrame(
        fractions,
        index=pd.Index(group_order, name=groupby),
        columns=pd.Index(resolved_genes, name="gene"),
    )
    return mean_expression, fraction_expression


def plot_group_gene_dotplot(
    mean_expression: pd.DataFrame,
    fraction_expression: pd.DataFrame,
    output_path: Path | str,
    *,
    gene_order: list[str] | None = None,
    cluster_genes: bool = True,
    standardize_mean_by_gene: bool = True,
    z_score_clip: tuple[float, float] = (-2.0, 2.0),
    min_dot_size: float = 2.0,
    max_dot_size: float = 55.0,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] | None = None,
    x_axis_label: str = "Subcluster",
    y_axis_label: str = "Panel genes",
    title: str = "Panel gene expression by subcluster",
    x_label_fontsize: float = 10.0,
    y_label_fontsize: float = 4.5,
    title_fontsize: float = 12.0,
    dpi: int = 220,
) -> Path:
    """Save a dotplot of mean expression and fraction expressing by group."""
    output_path = prepare_plot_output(output_path)
    if mean_expression.empty or fraction_expression.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.text(
            0.5,
            0.5,
            "No gene expression summary was available.",
            ha="center",
            va="center",
        )
        ax.axis("off")
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        return output_path

    if gene_order is None:
        gene_order = (
            _cluster_gene_order(mean_expression)
            if cluster_genes
            else list(mean_expression.columns)
        )

    group_order = list(mean_expression.index.astype(str))
    mean_plot = mean_expression.loc[group_order, gene_order]
    fraction_plot = fraction_expression.loc[group_order, gene_order]

    color_values = mean_plot.copy()
    if standardize_mean_by_gene:
        gene_std = color_values.std(axis=0).replace(0, np.nan)
        color_values = (
            color_values.sub(color_values.mean(axis=0), axis=1)
            .div(gene_std, axis=1)
            .fillna(0)
            .clip(lower=z_score_clip[0], upper=z_score_clip[1])
        )
        vmin, vmax = z_score_clip
        colorbar_label = "Mean expression z-score"
    else:
        vmin = 0.0
        vmax = float(np.nanpercentile(color_values.to_numpy(), 99.5))
        if vmax <= 0:
            vmax = 1.0
        colorbar_label = "Mean expression"

    if figsize is None:
        figsize = (
            max(8.0, len(group_order) * 0.95 + 2.5),
            max(10.0, len(gene_order) * 0.09 + 2.0),
        )

    fig, ax = plt.subplots(figsize=figsize)
    x_positions, y_positions = np.meshgrid(
        np.arange(len(group_order)),
        np.arange(len(gene_order)),
    )
    dot_sizes = min_dot_size + fraction_plot.T.to_numpy().ravel() * (
        max_dot_size - min_dot_size
    )
    scatter = ax.scatter(
        x_positions.ravel(),
        y_positions.ravel(),
        c=color_values.T.to_numpy().ravel(),
        s=dot_sizes,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="lightgray",
        linewidths=0.15,
    )

    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(
        group_order,
        rotation=35,
        ha="right",
        fontsize=x_label_fontsize,
    )
    ax.set_yticks(np.arange(len(gene_order)))
    ax.set_yticklabels(gene_order, fontsize=y_label_fontsize)
    ax.set_xlim(-0.5, len(group_order) - 0.5)
    ax.set_ylim(len(gene_order) - 0.5, -0.5)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title, fontsize=title_fontsize)
    ax.grid(axis="x", color="lightgray", linewidth=0.35, alpha=0.5)
    ax.grid(axis="y", color="lightgray", linewidth=0.25, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.018, pad=0.008)
    colorbar.set_label(colorbar_label)

    legend_fractions = [0.25, 0.5, 0.75, 1.0]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="none",
            markerfacecolor="white",
            markeredgecolor="gray",
            markersize=np.sqrt(min_dot_size + fraction * (max_dot_size - min_dot_size)),
            label=f"{fraction:.0%}",
        )
        for fraction in legend_fractions
    ]
    ax.legend(
        handles=legend_handles,
        title="Fraction expressing",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def _cluster_gene_order(
    mean_expression: pd.DataFrame,
    *,
    method: str = "average",
    metric: str = "euclidean",
) -> list[str]:
    if mean_expression.shape[1] <= 2:
        return list(mean_expression.columns)

    profiles = mean_expression.T.to_numpy(dtype=float)
    profiles = np.nan_to_num(profiles, copy=False)
    profile_means = profiles.mean(axis=1, keepdims=True)
    profile_stds = profiles.std(axis=1, keepdims=True)
    profiles = (profiles - profile_means) / np.where(
        profile_stds == 0,
        1.0,
        profile_stds,
    )
    if np.allclose(profiles, 0):
        return list(mean_expression.columns)

    gene_linkage = linkage(
        profiles,
        method=method,
        metric=metric,
        optimal_ordering=True,
    )
    return [str(mean_expression.columns[index]) for index in leaves_list(gene_linkage)]


def save_qc_metrics(adata: ad.AnnData, output_path: Path | str) -> Path:
    """Write selected per-cell QC metrics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [col for col in QC_COLUMNS if col in adata.obs]
    df = adata.obs.loc[:, columns].copy()
    df.insert(0, "obs_name", adata.obs_names.astype(str))
    if "instance_id" in adata.obs and "instance_id" not in df.columns:
        df.insert(1, "instance_id", adata.obs["instance_id"].to_numpy())
    if "cell_id" in adata.obs and "cell_id" not in df.columns:
        position = 2 if "instance_id" in df.columns else 1
        df.insert(position, "cell_id", adata.obs["cell_id"].astype(str).to_numpy())
    df.to_csv(output_path, index=False)
    return output_path


def save_clustered_adata(adata: ad.AnnData, output_path: Path | str) -> Path:
    """Write the clustered AnnData object to ``.h5ad``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    return output_path


def _clustered_spatialdata_table_key(
    source_table_key: str,
    segmentation: str | None,
) -> str:
    """Return the derived SpatialData table key for a clustered AnnData table."""
    segmentation_key = "" if segmentation is None else str(segmentation).strip().lower()
    source_key = str(source_table_key)
    if segmentation_key == "reseg" or source_key == "table_MOSAIK_proseg":
        return "table_MOSAIK_proseg_clustering_squidpy"
    if segmentation_key == "original_seg" or source_key == "table_original":
        return "table_original_clustering_squidpy"
    return f"{source_key}_clustering_squidpy"


def build_clustered_spatialdata_table(
    adata: ad.AnnData,
    *,
    output_table_key: str,
    output_region: str,
    source_table_key: str,
    source_region: str | None,
) -> ad.AnnData:
    """Return a SpatialData table for the final clustered AnnData object."""
    table = adata.copy()
    spatial_attrs = dict(table.uns.get("spatialdata_attrs", {}))
    region_key = str(spatial_attrs.get("region_key", "region"))
    instance_key = spatial_attrs.get("instance_key")

    if not isinstance(instance_key, str) or instance_key not in table.obs.columns:
        instance_key = _first_existing_column(
            table.obs,
            (
                "instance_id",
                "cell_id",
                "cell",
                "cells",
                "cell_ID",
                "EntityID",
                "cell_labels",
            ),
        )
    if instance_key is None:
        instance_key = "cell_id"
        table.obs[instance_key] = table.obs_names.astype(str)

    if region_key not in table.obs.columns:
        table.obs[region_key] = str(output_region)
    table.obs[region_key] = pd.Categorical(
        [str(output_region)] * table.n_obs,
        categories=[str(output_region)],
    )

    clustering_meta = dict(table.uns.get("merxen_clustering_squidpy", {}))
    clustering_meta.update(
        {
            "source_table_key": str(source_table_key),
            "source_region": None if source_region is None else str(source_region),
            "written_table_key": str(output_table_key),
            "written_region": str(output_region),
            "spatialdata_table_key": str(output_table_key),
            "spatialdata_region": str(output_region),
        }
    )
    table.uns["merxen_clustering_squidpy"] = clustering_meta
    table.uns.pop("spatialdata_attrs", None)

    from spatialdata.models import TableModel

    return TableModel.parse(
        table,
        region=str(output_region),
        region_key=region_key,
        instance_key=str(instance_key),
    )


def write_clustered_spatialdata_table(
    zarr_path: Path | str,
    adata: ad.AnnData,
    *,
    segmentation: str | None,
) -> tuple[Path, str]:
    """Attach the final clustered AnnData object as a SpatialData table."""
    zarr_path = Path(zarr_path)
    clustering_meta = dict(adata.uns.get("merxen_clustering_squidpy", {}))
    source_table_key = str(clustering_meta.get("table_key") or "table")
    spatial_attrs = dict(adata.uns.get("spatialdata_attrs", {}))
    source_region = _region_as_string(spatial_attrs.get("region"))
    output_region = _region_as_string(clustering_meta.get("shape_key")) or source_region
    if output_region is None:
        raise ValueError(
            "Cannot write clustered SpatialData table without a source or plotting "
            "region in AnnData metadata."
        )

    output_table_key = _clustered_spatialdata_table_key(
        source_table_key,
        segmentation,
    )
    parsed_table = build_clustered_spatialdata_table(
        adata,
        output_table_key=output_table_key,
        output_region=output_region,
        source_table_key=source_table_key,
        source_region=source_region,
    )

    import spatialdata as sd

    from merxen.io.spatialdata_io import write_or_replace_element

    with _spatialdata_zarr_write_lock(zarr_path):
        log_status(
            f"Writing clustered SpatialData table '{output_table_key}' to {zarr_path}"
        )
        sdata_obj = sd.read_zarr(zarr_path)
        try:
            write_or_replace_element(
                sdata_obj,
                output_table_key,
                "tables",
                parsed_table,
                overwrite=True,
            )
        finally:
            del sdata_obj
            force_release(note=f"after writing clustered table {output_table_key}")

    return zarr_path, output_table_key


@contextmanager
def _spatialdata_zarr_write_lock(zarr_path: Path | str) -> Iterator[None]:
    """Serialize side-effectful writes to one SpatialData zarr path."""
    lock_path = Path(f"{Path(zarr_path)}.clustering_squidpy.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _region_as_string(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, list | tuple) and len(value) > 0:
        return _region_as_string(value[0])
    return None


def run_hierarchical_scanpy_clustering(
    adata: ad.AnnData,
    config: ClusteringSquidpyConfig,
    *,
    output_dir: Path | str,
    sample_id: str,
) -> tuple[ad.AnnData, dict[str, Path]]:
    """Run broad annotation and branch-level subclustering."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    broad_params = _effective_round_params(config, config.broad_round)
    clustered = _run_configured_round(
        adata,
        config,
        broad_params,
        key_added=BROAD_CLUSTER_KEY,
        input_layer=None,
        drop_control_features=config.drop_control_features,
    )
    clustered.obs["leiden"] = clustered.obs[BROAD_CLUSTER_KEY].astype(str)

    marker_sets = _load_configured_marker_sets(config)
    marker_alias_lookup = _load_configured_marker_alias_lookup(config)
    broad_assignments, broad_scores, broad_markers = score_clusters_by_atlas_markers(
        clustered,
        cluster_key=BROAD_CLUSTER_KEY,
        marker_sets=marker_sets,
        marker_alias_lookup=marker_alias_lookup,
        min_marker_overlap=config.broad_annotation.min_marker_overlap,
        max_markers_per_label=config.broad_annotation.max_markers_per_label,
        score_margin_threshold=config.broad_annotation.score_margin_threshold,
        unknown_label=config.broad_annotation.unknown_label,
    )
    _apply_cluster_annotations(
        clustered,
        cluster_key=BROAD_CLUSTER_KEY,
        assignments=broad_assignments,
        atlas_label_key=BROAD_ATLAS_LABEL_KEY,
        broad_class_key=BROAD_CLASS_KEY,
        metric_prefix="broad",
    )
    artifacts.update(
        _write_annotation_artifacts(
            output_dir,
            prefix=f"{sample_id}_broad",
            assignments=broad_assignments,
            scores=broad_scores,
            markers=broad_markers,
            heatmap_title=f"{sample_id} broad atlas marker scores",
            dpi=config.figure_dpi,
        )
    )
    artifacts.update(
        _save_round_plots(
            clustered,
            output_dir=output_dir,
            prefix=f"{sample_id}_broad",
            colors=["total_counts", "n_genes_by_counts", BROAD_CLUSTER_KEY],
            spatial_color=BROAD_CLASS_KEY,
            grid_color=BROAD_CLASS_KEY,
            spatial_point_size=config.spatial_scatter_point_size,
            grid_point_size=config.spatial_point_size,
            dpi=config.figure_dpi,
        )
    )

    _initialize_hierarchical_obs(clustered)
    branch_manifest: dict[str, Any] = {}
    for broad_class in _ordered_obs_values(clustered, BROAD_CLASS_KEY):
        branch_mask = (
            clustered.obs[BROAD_CLASS_KEY].astype(str).to_numpy() == broad_class
        )
        branch_cells = list(clustered.obs_names[branch_mask].astype(str))
        branch_token = _safe_token(broad_class)
        if len(branch_cells) < config.min_branch_cells:
            _assign_unclustered_branch(
                clustered,
                cell_ids=branch_cells,
                broad_class=broad_class,
                reason="too_few_cells",
            )
            branch_manifest[broad_class] = {
                "n_cells": len(branch_cells),
                "clustered": False,
                "reason": "too_few_cells",
            }
            continue

        if broad_class == NEURON_CLASS:
            branch_manifest[broad_class] = _run_neuron_hierarchy(
                clustered,
                cell_ids=branch_cells,
                config=config,
                marker_sets=marker_sets,
                marker_alias_lookup=marker_alias_lookup,
                output_dir=output_dir / f"branch_{branch_token}",
                sample_id=sample_id,
                artifacts=artifacts,
            )
        else:
            branch_manifest[broad_class] = _run_leaf_branch_subclustering(
                clustered,
                cell_ids=branch_cells,
                config=config,
                broad_class=broad_class,
                output_dir=output_dir / f"branch_{branch_token}",
                sample_id=sample_id,
                artifacts=artifacts,
            )

    clustered.obs[SUBCLUSTER_LABEL_KEY] = clustered.obs[SUBCLUSTER_LABEL_KEY].astype(
        "category"
    )
    clustered.obs[HIERARCHICAL_CLUSTER_KEY] = clustered.obs[
        HIERARCHICAL_CLUSTER_KEY
    ].astype("category")
    clustered.obs[NEURON_SPLIT_KEY] = clustered.obs[NEURON_SPLIT_KEY].astype("category")
    manifest_path = output_dir / f"{sample_id}_hierarchical_manifest.json"
    artifacts["hierarchical_manifest"] = manifest_path
    clustered.uns[HIERARCHICAL_UNS_KEY] = {
        "enabled": True,
        "broad_cluster_key": BROAD_CLUSTER_KEY,
        "broad_atlas_label_key": BROAD_ATLAS_LABEL_KEY,
        "broad_class_key": BROAD_CLASS_KEY,
        "subcluster_label_key": SUBCLUSTER_LABEL_KEY,
        "hierarchical_cluster_key": HIERARCHICAL_CLUSTER_KEY,
        "neuron_split_key": NEURON_SPLIT_KEY,
        "min_branch_cells": int(config.min_branch_cells),
        "branch_manifest": branch_manifest,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    manifest_path.write_text(
        json.dumps(clustered.uns[HIERARCHICAL_UNS_KEY], indent=2) + "\n"
    )
    return clustered, artifacts


def _load_configured_marker_sets(
    config: ClusteringSquidpyConfig,
) -> list[AtlasMarkerSet]:
    annotation = config.broad_annotation
    marker_lookup_path = annotation.marker_lookup_path
    taxonomy_metadata_path = annotation.taxonomy_metadata_path
    if taxonomy_metadata_path is None and annotation.reference_cache_dir is not None:
        taxonomy_metadata_path = _find_cached_taxonomy_metadata(
            annotation.reference_cache_dir
        )
    if marker_lookup_path is None or taxonomy_metadata_path is None:
        raise ValueError(
            "hierarchical clustering requires broad_annotation.marker_lookup_path "
            "and broad_annotation.taxonomy_metadata_path or reference_cache_dir"
        )
    marker_sets = load_atlas_marker_sets(
        marker_lookup_path,
        taxonomy_metadata_path,
        marker_level=annotation.marker_level,
        cluster_membership_path=annotation.cluster_membership_path,
    )
    if not marker_sets:
        raise ValueError(
            "No atlas marker sets were loaded from "
            f"{marker_lookup_path} at marker_level={annotation.marker_level!r}"
        )
    return marker_sets


def _find_cached_taxonomy_metadata(cache_dir: Path | str) -> Path | None:
    candidates = sorted(
        Path(cache_dir).glob(
            "abc_whb/metadata/WHB-taxonomy/*/cluster_annotation_term.csv"
        )
    )
    return candidates[-1] if candidates else None


def _load_configured_marker_alias_lookup(
    config: ClusteringSquidpyConfig,
) -> dict[str, str]:
    annotation = config.broad_annotation
    paths = list(annotation.reference_gene_metadata_paths)
    if not paths and annotation.reference_cache_dir is not None:
        paths = _find_cached_reference_gene_metadata_paths(
            annotation.reference_cache_dir
        )
    if not paths:
        return {}

    lookup: dict[str, str] = {}
    for path in paths:
        lookup.update(_reference_gene_symbol_lookup_from_h5ad(path))
    if lookup:
        logger.info(
            "Loaded %d marker ID aliases from reference gene metadata.",
            len(lookup),
        )
    return lookup


def _find_cached_reference_gene_metadata_paths(cache_dir: Path | str) -> list[Path]:
    return sorted(
        Path(cache_dir).glob(
            "abc_whb/expression_matrices/WHB-10Xv3/*/WHB-10Xv3-*-raw.h5ad"
        )
    )


def _reference_gene_symbol_lookup_from_h5ad(path: Path | str) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        logger.warning("Reference gene metadata H5AD does not exist: %s", path)
        return {}

    try:
        reference = ad.read_h5ad(path, backed="r")
    except Exception as exc:  # pragma: no cover - defensive around large H5AD IO
        logger.warning("Could not read reference gene metadata from %s: %s", path, exc)
        return {}
    try:
        if "gene_symbol" not in reference.var:
            return {}
        lookup: dict[str, str] = {}
        ensembl_ids = reference.var_names.astype(str)
        symbols = reference.var["gene_symbol"].astype(str).to_numpy()
        for ensembl_id, symbol in zip(ensembl_ids, symbols, strict=True):
            ensembl = str(ensembl_id).strip()
            gene_symbol = str(symbol).strip()
            if not ensembl or not gene_symbol or gene_symbol.lower() in {"nan", "none"}:
                continue
            lookup.setdefault(_normalize_marker_id(ensembl), gene_symbol)
            lookup.setdefault(_normalize_marker_id(gene_symbol), ensembl)
        return lookup
    finally:
        reference.file.close()


def _effective_round_params(
    config: ClusteringSquidpyConfig,
    round_config: Any,
    *,
    leiden_resolution: float | None = None,
) -> RoundParams:
    return RoundParams(
        min_counts=int(
            config.min_counts
            if round_config.min_counts is None
            else round_config.min_counts
        ),
        min_cells=int(
            config.min_cells
            if round_config.min_cells is None
            else round_config.min_cells
        ),
        n_pcs=int(config.n_pcs if round_config.n_pcs is None else round_config.n_pcs),
        n_neighbors=int(
            config.n_neighbors
            if round_config.n_neighbors is None
            else round_config.n_neighbors
        ),
        leiden_resolution=float(
            round_config.leiden_resolution
            if leiden_resolution is None
            else leiden_resolution
        ),
        umap_min_dist=float(
            config.umap_min_dist
            if round_config.umap_min_dist is None
            else round_config.umap_min_dist
        ),
        umap_spread=float(
            config.umap_spread
            if round_config.umap_spread is None
            else round_config.umap_spread
        ),
    )


def _run_configured_round(
    adata: ad.AnnData,
    config: ClusteringSquidpyConfig,
    params: RoundParams,
    *,
    key_added: str,
    input_layer: str | None,
    drop_control_features: bool,
) -> ad.AnnData:
    return run_scanpy_clustering(
        adata,
        drop_control_features=drop_control_features,
        min_counts=params.min_counts,
        min_cells=params.min_cells,
        normalize_target_sum=config.normalize_target_sum,
        normalize_exclude_highly_expressed=config.normalize_exclude_highly_expressed,
        normalize_max_fraction=config.normalize_max_fraction,
        n_pcs=params.n_pcs,
        n_neighbors=params.n_neighbors,
        leiden_resolution=params.leiden_resolution,
        umap_min_dist=params.umap_min_dist,
        umap_spread=params.umap_spread,
        random_seed=config.random_seed,
        use_gpu=config.use_gpu,
        key_added=key_added,
        input_layer=input_layer,
    )


def _apply_cluster_annotations(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    assignments: pd.DataFrame,
    atlas_label_key: str,
    broad_class_key: str,
    metric_prefix: str,
) -> None:
    mapping = assignments.set_index("cluster", drop=False)
    cluster_values = pd.Series(
        adata.obs[cluster_key],
        index=adata.obs_names,
    ).astype(str)
    adata.obs[atlas_label_key] = pd.Categorical(
        cluster_values.map(mapping["atlas_label"]).fillna(UNKNOWN_LABEL)
    )
    adata.obs[broad_class_key] = pd.Categorical(
        cluster_values.map(mapping["broad_class"]).fillna(UNKNOWN_LABEL)
    )
    for source, target_suffix in [
        ("score", "score"),
        ("score_margin", "score_margin"),
        ("n_markers", "n_markers"),
    ]:
        adata.obs[f"{metric_prefix}_annotation_{target_suffix}"] = cluster_values.map(
            mapping[source]
        ).to_numpy()


def _write_annotation_artifacts(
    output_dir: Path,
    *,
    prefix: str,
    assignments: pd.DataFrame,
    scores: pd.DataFrame,
    markers: pd.DataFrame,
    heatmap_title: str,
    dpi: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / f"{prefix}_cluster_annotation.csv"
    scores_path = output_dir / f"{prefix}_annotation_scores.csv"
    markers_path = output_dir / f"{prefix}_resolved_markers.csv"
    heatmap_path = _plot_output_dir(output_dir, "annotation") / (
        f"{prefix}_annotation_score_heatmap.png"
    )
    assignments.to_csv(assignment_path, index=False)
    scores.to_csv(scores_path, index=False)
    markers.to_csv(markers_path, index=False)
    plot_annotation_score_heatmap(scores, heatmap_path, title=heatmap_title, dpi=dpi)
    return {
        f"{prefix}_cluster_annotation": assignment_path,
        f"{prefix}_annotation_scores": scores_path,
        f"{prefix}_resolved_markers": markers_path,
        f"{prefix}_annotation_score_heatmap": heatmap_path,
    }


def _save_round_plots(
    adata: ad.AnnData,
    *,
    output_dir: Path,
    prefix: str,
    colors: list[str],
    spatial_color: str,
    grid_color: str,
    spatial_point_size: float,
    grid_point_size: float,
    dpi: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        f"{prefix}_umap": plot_umap(
            adata,
            _plot_output_dir(output_dir, "umap") / f"{prefix}_umap.png",
            color=colors,
            dpi=dpi,
        ),
        f"{prefix}_spatial": plot_spatial_scatter(
            adata,
            _plot_output_dir(output_dir, "spatial") / f"{prefix}_spatial.png",
            color=spatial_color,
            point_size=spatial_point_size,
            dpi=dpi,
        ),
        f"{prefix}_spatial_grid": plot_spatial_cluster_grid(
            adata,
            _plot_output_dir(output_dir, "spatial_grid") / f"{prefix}_spatial_grid.png",
            color=grid_color,
            point_size_highlight=grid_point_size,
            dpi=dpi,
        ),
    }
    return paths


def _save_branch_gene_dotplot(
    adata: ad.AnnData,
    *,
    output_dir: Path,
    prefix: str,
    group_key: str,
    dpi: int,
) -> dict[str, Path]:
    genes = _panel_genes_for_dotplot(adata)
    mean_expression, fraction_expression = compute_group_gene_summary(
        adata,
        genes,
        groupby=group_key,
    )
    gene_order = _cluster_gene_order(mean_expression)
    table_dir = _table_output_dir(output_dir, "dotplot")
    plot_dir = _plot_output_dir(output_dir, "dotplot")
    mean_path = table_dir / f"{prefix}_gene_mean_expression.csv"
    fraction_path = table_dir / f"{prefix}_gene_fraction_expressing.csv"
    dotplot_path = plot_dir / f"{prefix}_gene_dotplot.png"
    mean_expression.loc[:, gene_order].to_csv(mean_path)
    fraction_expression.loc[:, gene_order].to_csv(fraction_path)
    plot_group_gene_dotplot(
        mean_expression,
        fraction_expression,
        dotplot_path,
        gene_order=gene_order,
        cluster_genes=False,
        x_axis_label=group_key,
        y_axis_label="Panel genes",
        title=f"{prefix} panel gene expression by subcluster",
        dpi=max(int(dpi), 220),
    )
    return {
        f"{prefix}_gene_dotplot": dotplot_path,
        f"{prefix}_gene_mean_expression": mean_path,
        f"{prefix}_gene_fraction_expressing": fraction_path,
    }


def _panel_genes_for_dotplot(adata: ad.AnnData) -> list[str]:
    if adata.n_vars == 0:
        return []
    control_mask = _control_feature_mask(adata)
    genes = [
        str(gene)
        for gene, is_control in zip(
            adata.var_names.astype(str),
            control_mask,
            strict=True,
        )
        if not bool(is_control)
    ]
    return genes or [str(gene) for gene in adata.var_names.astype(str)]


def _plot_output_dir(output_dir: Path, plot_type: str) -> Path:
    path = output_dir / "plots" / plot_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def _table_output_dir(output_dir: Path, table_type: str) -> Path:
    path = output_dir / "tables" / table_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def _initialize_hierarchical_obs(adata: ad.AnnData) -> None:
    adata.obs[SUBCLUSTER_LABEL_KEY] = pd.Series(
        "unassigned",
        index=adata.obs_names,
        dtype="object",
    )
    adata.obs[HIERARCHICAL_CLUSTER_KEY] = pd.Series(
        "unassigned",
        index=adata.obs_names,
        dtype="object",
    )
    adata.obs[NEURON_SPLIT_KEY] = pd.Series(
        "not_neuron",
        index=adata.obs_names,
        dtype="object",
    )


def _ordered_obs_values(adata: ad.AnnData, key: str) -> list[str]:
    values = pd.Series(adata.obs[key], index=adata.obs_names).astype(str)
    return [
        str(value)
        for value in values.value_counts(sort=False).index
        if str(value) != "nan"
    ]


def _assign_unclustered_branch(
    adata: ad.AnnData,
    *,
    cell_ids: list[str],
    broad_class: str,
    reason: str,
) -> None:
    label = f"{broad_class}:not_subclustered"
    subcluster_label = (
        "not_subclustered"
        if reason == "too_few_cells"
        else f"not_subclustered_{reason}"
    )
    adata.obs.loc[cell_ids, SUBCLUSTER_LABEL_KEY] = subcluster_label
    adata.obs.loc[cell_ids, HIERARCHICAL_CLUSTER_KEY] = label
    if broad_class == NEURON_CLASS:
        adata.obs.loc[cell_ids, NEURON_SPLIT_KEY] = reason


def _run_leaf_branch_subclustering(
    clustered: ad.AnnData,
    *,
    cell_ids: list[str],
    config: ClusteringSquidpyConfig,
    broad_class: str,
    output_dir: Path,
    sample_id: str,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    branch = clustered[cell_ids, :].copy()
    resolution = config.subcluster_resolution_overrides.get(
        broad_class,
        config.subcluster_round.leiden_resolution,
    )
    params = _effective_round_params(
        config,
        config.subcluster_round,
        leiden_resolution=resolution,
    )
    try:
        branch = _run_configured_round(
            branch,
            config,
            params,
            key_added="leiden_subcluster",
            input_layer="counts",
            drop_control_features=False,
        )
    except ValueError as exc:
        logger.warning(
            "Skipping subclustering for %s after filtering failed: %s",
            broad_class,
            exc,
        )
        _assign_unclustered_branch(
            clustered,
            cell_ids=cell_ids,
            broad_class=broad_class,
            reason="filtering_failed",
        )
        return {
            "n_cells": len(cell_ids),
            "clustered": False,
            "reason": "filtering_failed",
            "error": str(exc),
        }
    filtered_out_cells = _missing_cell_ids(cell_ids, branch)
    if filtered_out_cells:
        _assign_unclustered_branch(
            clustered,
            cell_ids=filtered_out_cells,
            broad_class=broad_class,
            reason="filtered_out",
        )
    prefix = f"{sample_id}_{_safe_token(broad_class)}_subcluster"
    artifacts.update(
        _save_round_plots(
            branch,
            output_dir=output_dir,
            prefix=prefix,
            colors=["total_counts", "n_genes_by_counts", "leiden_subcluster"],
            spatial_color="leiden_subcluster",
            grid_color="leiden_subcluster",
            spatial_point_size=config.spatial_scatter_point_size,
            grid_point_size=config.spatial_point_size,
            dpi=config.figure_dpi,
        )
    )
    artifacts.update(
        _save_branch_gene_dotplot(
            branch,
            output_dir=output_dir,
            prefix=prefix,
            group_key="leiden_subcluster",
            dpi=config.figure_dpi,
        )
    )
    h5ad_path = save_clustered_adata(branch, output_dir / f"{prefix}.h5ad")
    artifacts[f"{prefix}_h5ad"] = h5ad_path
    _assign_branch_labels(
        clustered,
        branch=branch,
        broad_class=broad_class,
        cluster_key="leiden_subcluster",
    )
    return {
        "n_cells": len(cell_ids),
        "n_clustered_cells": int(branch.n_obs),
        "n_filtered_out_cells": len(filtered_out_cells),
        "clustered": True,
        "leiden_resolution": float(params.leiden_resolution),
        "n_subclusters": int(branch.obs["leiden_subcluster"].nunique()),
    }


def _assign_branch_labels(
    target: ad.AnnData,
    *,
    branch: ad.AnnData,
    broad_class: str,
    cluster_key: str,
    neuron_split: str | None = None,
) -> None:
    labels = pd.Series(branch.obs[cluster_key], index=branch.obs_names).astype(str)
    for obs_name, local_label in labels.items():
        subcluster = str(local_label)
        if neuron_split is None:
            hierarchical = f"{broad_class}:{subcluster}"
        else:
            hierarchical = f"{broad_class}/{neuron_split}:{subcluster}"
            target.obs.at[obs_name, NEURON_SPLIT_KEY] = neuron_split
        target.obs.at[obs_name, SUBCLUSTER_LABEL_KEY] = subcluster
        target.obs.at[obs_name, HIERARCHICAL_CLUSTER_KEY] = hierarchical


def _run_neuron_hierarchy(
    clustered: ad.AnnData,
    *,
    cell_ids: list[str],
    config: ClusteringSquidpyConfig,
    marker_sets: list[AtlasMarkerSet],
    marker_alias_lookup: dict[str, str],
    output_dir: Path,
    sample_id: str,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    neuron_branch = clustered[cell_ids, :].copy()
    split_params = _effective_round_params(config, config.neuron_split_round)
    try:
        neuron_branch = _run_configured_round(
            neuron_branch,
            config,
            split_params,
            key_added="leiden_neuron_split",
            input_layer="counts",
            drop_control_features=False,
        )
    except ValueError as exc:
        logger.warning("Skipping neuron hierarchy after filtering failed: %s", exc)
        _assign_unclustered_branch(
            clustered,
            cell_ids=cell_ids,
            broad_class=NEURON_CLASS,
            reason="filtering_failed",
        )
        return {
            "n_cells": len(cell_ids),
            "clustered": False,
            "reason": "filtering_failed",
            "error": str(exc),
        }
    split_marker_sets = _make_neuron_split_marker_sets(marker_sets)
    split_assignments, split_scores, split_markers = score_clusters_by_atlas_markers(
        neuron_branch,
        cluster_key="leiden_neuron_split",
        marker_sets=split_marker_sets,
        marker_alias_lookup=marker_alias_lookup,
        min_marker_overlap=config.broad_annotation.min_marker_overlap,
        max_markers_per_label=config.broad_annotation.max_markers_per_label,
        score_margin_threshold=config.broad_annotation.score_margin_threshold,
        unknown_label="Other",
    )
    _apply_cluster_annotations(
        neuron_branch,
        cluster_key="leiden_neuron_split",
        assignments=split_assignments,
        atlas_label_key="neuron_split_atlas_label",
        broad_class_key=NEURON_SPLIT_KEY,
        metric_prefix="neuron_split",
    )
    prefix = f"{sample_id}_neurons_split"
    artifacts.update(
        _write_annotation_artifacts(
            output_dir,
            prefix=prefix,
            assignments=split_assignments,
            scores=split_scores,
            markers=split_markers,
            heatmap_title=f"{sample_id} neuron split marker scores",
            dpi=config.figure_dpi,
        )
    )
    artifacts.update(
        _save_round_plots(
            neuron_branch,
            output_dir=output_dir,
            prefix=prefix,
            colors=["total_counts", "n_genes_by_counts", "leiden_neuron_split"],
            spatial_color=NEURON_SPLIT_KEY,
            grid_color=NEURON_SPLIT_KEY,
            spatial_point_size=config.spatial_scatter_point_size,
            grid_point_size=config.spatial_point_size,
            dpi=config.figure_dpi,
        )
    )
    split_h5ad = save_clustered_adata(neuron_branch, output_dir / f"{prefix}.h5ad")
    artifacts[f"{prefix}_h5ad"] = split_h5ad

    split_manifest: dict[str, Any] = {}
    filtered_out_cells = _missing_cell_ids(cell_ids, neuron_branch)
    if filtered_out_cells:
        _assign_unclustered_neuron_split(
            clustered,
            cell_ids=filtered_out_cells,
            split_label="filtered_out",
            reason="neuron_split_filtering",
        )
        split_manifest["filtered_out"] = {
            "n_cells": len(filtered_out_cells),
            "clustered": False,
            "reason": "neuron_split_filtering",
        }
    for split_label in _ordered_obs_values(neuron_branch, NEURON_SPLIT_KEY):
        split_mask = neuron_branch.obs[NEURON_SPLIT_KEY].astype(str).to_numpy() == (
            split_label
        )
        split_cells = list(neuron_branch.obs_names[split_mask].astype(str))
        if len(split_cells) < config.min_branch_cells:
            _assign_unclustered_neuron_split(
                clustered,
                cell_ids=split_cells,
                split_label=split_label,
                reason="too_few_cells",
            )
            split_manifest[split_label] = {
                "n_cells": len(split_cells),
                "clustered": False,
                "reason": "too_few_cells",
            }
            continue

        split_manifest[split_label] = _run_neuron_split_subclustering(
            clustered,
            neuron_branch=neuron_branch,
            cell_ids=split_cells,
            split_label=split_label,
            config=config,
            output_dir=output_dir / f"split_{_safe_token(split_label)}",
            sample_id=sample_id,
            artifacts=artifacts,
        )
    return {
        "n_cells": len(cell_ids),
        "n_split_cells": int(neuron_branch.n_obs),
        "n_filtered_out_cells": len(filtered_out_cells),
        "clustered": True,
        "split_leiden_resolution": float(split_params.leiden_resolution),
        "splits": split_manifest,
    }


def _make_neuron_split_marker_sets(
    marker_sets: list[AtlasMarkerSet],
) -> list[AtlasMarkerSet]:
    grouped: dict[str, list[str]] = {"Excitatory": [], "Inhibitory": [], "Other": []}
    for marker_set in marker_sets:
        if marker_set.broad_class != NEURON_CLASS:
            continue
        split = marker_set.neuron_split or _neuron_split_for_supercluster(
            marker_set.label_name
        )
        grouped[split].extend(marker_set.marker_ids)

    split_sets: list[AtlasMarkerSet] = []
    for split, markers in grouped.items():
        unique_markers = tuple(dict.fromkeys(markers))
        split_sets.append(
            AtlasMarkerSet(
                level="neuron_split",
                label_id=split,
                label_name=split,
                broad_class=split,
                marker_ids=unique_markers,
            )
        )
    return split_sets


def _neuron_split_for_supercluster(label_name: str) -> str:
    label = str(label_name).strip().lower()
    if any(token in label for token in INHIBITORY_SUPERCLUSTER_TOKENS):
        return "Inhibitory"
    if any(token in label for token in EXCITATORY_SUPERCLUSTER_TOKENS):
        return "Excitatory"
    return "Other"


def _assign_unclustered_neuron_split(
    adata: ad.AnnData,
    *,
    cell_ids: list[str],
    split_label: str,
    reason: str,
) -> None:
    label = f"{NEURON_CLASS}/{split_label}:not_subclustered"
    adata.obs.loc[cell_ids, NEURON_SPLIT_KEY] = split_label
    adata.obs.loc[cell_ids, SUBCLUSTER_LABEL_KEY] = f"not_subclustered_{reason}"
    adata.obs.loc[cell_ids, HIERARCHICAL_CLUSTER_KEY] = label


def _run_neuron_split_subclustering(
    clustered: ad.AnnData,
    *,
    neuron_branch: ad.AnnData,
    cell_ids: list[str],
    split_label: str,
    config: ClusteringSquidpyConfig,
    output_dir: Path,
    sample_id: str,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    branch = neuron_branch[cell_ids, :].copy()
    resolution = config.subcluster_resolution_overrides.get(
        f"{NEURON_CLASS}/{split_label}",
        config.subcluster_resolution_overrides.get(
            split_label,
            config.neuron_subcluster_round.leiden_resolution,
        ),
    )
    params = _effective_round_params(
        config,
        config.neuron_subcluster_round,
        leiden_resolution=resolution,
    )
    try:
        branch = _run_configured_round(
            branch,
            config,
            params,
            key_added="leiden_neuron_subcluster",
            input_layer="counts",
            drop_control_features=False,
        )
    except ValueError as exc:
        logger.warning(
            "Skipping neuron subclustering for %s after filtering failed: %s",
            split_label,
            exc,
        )
        _assign_unclustered_neuron_split(
            clustered,
            cell_ids=cell_ids,
            split_label=split_label,
            reason="filtering_failed",
        )
        return {
            "n_cells": len(cell_ids),
            "clustered": False,
            "reason": "filtering_failed",
            "error": str(exc),
        }
    filtered_out_cells = _missing_cell_ids(cell_ids, branch)
    if filtered_out_cells:
        _assign_unclustered_neuron_split(
            clustered,
            cell_ids=filtered_out_cells,
            split_label=split_label,
            reason="filtered_out",
        )
    prefix = f"{sample_id}_neurons_{_safe_token(split_label)}_subcluster"
    artifacts.update(
        _save_round_plots(
            branch,
            output_dir=output_dir,
            prefix=prefix,
            colors=[
                "total_counts",
                "n_genes_by_counts",
                "leiden_neuron_subcluster",
            ],
            spatial_color="leiden_neuron_subcluster",
            grid_color="leiden_neuron_subcluster",
            spatial_point_size=config.spatial_scatter_point_size,
            grid_point_size=config.spatial_point_size,
            dpi=config.figure_dpi,
        )
    )
    artifacts.update(
        _save_branch_gene_dotplot(
            branch,
            output_dir=output_dir,
            prefix=prefix,
            group_key="leiden_neuron_subcluster",
            dpi=config.figure_dpi,
        )
    )
    h5ad_path = save_clustered_adata(branch, output_dir / f"{prefix}.h5ad")
    artifacts[f"{prefix}_h5ad"] = h5ad_path
    _assign_branch_labels(
        clustered,
        branch=branch,
        broad_class=NEURON_CLASS,
        cluster_key="leiden_neuron_subcluster",
        neuron_split=split_label,
    )
    return {
        "n_cells": len(cell_ids),
        "n_clustered_cells": int(branch.n_obs),
        "n_filtered_out_cells": len(filtered_out_cells),
        "clustered": True,
        "leiden_resolution": float(params.leiden_resolution),
        "n_subclusters": int(branch.obs["leiden_neuron_subcluster"].nunique()),
    }


def _missing_cell_ids(input_cell_ids: list[str], adata: ad.AnnData) -> list[str]:
    retained = set(adata.obs_names.astype(str))
    return [cell_id for cell_id in input_cell_ids if cell_id not in retained]


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "value"


def _cluster_loaded_adata(
    adata: ad.AnnData,
    config: ClusteringSquidpyConfig,
    *,
    sample: Any,
    sample_dir: Path,
) -> tuple[ad.AnnData, dict[str, Path | str]]:
    """Cluster one prepared AnnData object and write non-SpatialData artifacts."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    qc_plot = plot_qc_histograms(
        adata,
        _plot_output_dir(sample_dir, "qc") / f"{sample.sample_id}_qc_histograms.png",
        sample_label=sample.sample_id,
        platform=sample.platform,
        dpi=config.figure_dpi,
    )
    qc_csv = save_qc_metrics(adata, sample_dir / f"{sample.sample_id}_qc_metrics.csv")

    hierarchical_artifacts: dict[str, Path] = {}
    if config.hierarchical_enabled:
        clustered, hierarchical_artifacts = run_hierarchical_scanpy_clustering(
            adata,
            config,
            output_dir=sample_dir / f"{sample.sample_id}_hierarchical",
            sample_id=sample.sample_id,
        )
        umap_colors = [
            "total_counts",
            "n_genes_by_counts",
            BROAD_CLUSTER_KEY,
            BROAD_CLASS_KEY,
        ]
        spatial_color = BROAD_CLASS_KEY
    else:
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
        umap_colors = ["total_counts", "n_genes_by_counts", "leiden"]
        spatial_color = "leiden"

    umap_plot = plot_umap(
        clustered,
        _plot_output_dir(sample_dir, "umap") / f"{sample.sample_id}_umap.png",
        color=umap_colors,
        dpi=config.figure_dpi,
    )
    spatial_plot = plot_spatial_scatter(
        clustered,
        _plot_output_dir(sample_dir, "spatial")
        / f"{sample.sample_id}_spatial_scatter_leiden.png",
        color=spatial_color,
        point_size=config.spatial_scatter_point_size,
        dpi=config.figure_dpi,
    )
    spatial_cluster_grid = plot_spatial_cluster_grid(
        clustered,
        _plot_output_dir(sample_dir, "spatial_grid")
        / f"{sample.sample_id}_spatial_scatter_leiden_grid.png",
        color=spatial_color,
        point_size_highlight=config.spatial_point_size,
        dpi=config.figure_dpi,
    )
    h5ad = save_clustered_adata(
        clustered,
        sample_dir / f"{sample.sample_id}_clustered.h5ad",
    )
    return clustered, {
        "qc_plot": qc_plot,
        "qc_csv": qc_csv,
        "umap_plot": umap_plot,
        "spatial_plot": spatial_plot,
        "spatial_cluster_grid": spatial_cluster_grid,
        "h5ad": h5ad,
        **hierarchical_artifacts,
    }


def prepare_clustering_squidpy(
    config: ClusteringSquidpyConfig,
    output_dir: Path | str,
) -> Path:
    """Export SpatialData-backed samples as portable H5AD compute inputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_id_lookup = collect_gene_id_lookup_for_samples(config)
    manifest: dict[str, str] = {}
    for sample in config.samples:
        adata = load_spatialdata_adata(
            sample.zarr_path,
            platform=sample.platform,
            table_key=sample.table_key,
            shape_key=sample.shape_key,
            gene_id_lookup=gene_id_lookup,
        )
        relative_path = Path(sample.platform.lower()) / (
            f"{sample.sample_id}_prepared.h5ad"
        )
        save_clustered_adata(adata, output_dir / relative_path)
        manifest[sample.sample_id] = str(relative_path)
        del adata
        force_release(note=f"after preparing clustering input {sample.sample_id}")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": manifest}, indent=2) + "\n")
    return manifest_path


def compute_clustering_squidpy(
    config: ClusteringSquidpyConfig,
    prepared_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, dict[str, Path | str]]:
    """Run clustering from H5AD inputs without importing SpatialData."""
    prepared_dir = Path(prepared_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((prepared_dir / "manifest.json").read_text())["samples"]
    results: dict[str, dict[str, Path | str]] = {}
    for sample in config.samples:
        log_status(f"[{sample.sample_id}] Starting isolated clustering compute")
        adata = ad.read_h5ad(prepared_dir / manifest[sample.sample_id])
        clustered, artifacts = _cluster_loaded_adata(
            adata,
            config,
            sample=sample,
            sample_dir=output_dir / sample.platform.lower(),
        )
        results[sample.sample_id] = artifacts
        del adata, clustered
        force_release(note=f"after clustering compute {sample.sample_id}")
    return results


def finalize_clustering_squidpy(
    config: ClusteringSquidpyConfig,
    computed_dir: Path | str,
) -> dict[str, dict[str, Path | str]]:
    """Publish compute artifacts and write clustered tables with SpatialData."""
    import shutil

    computed_dir = Path(computed_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(computed_dir, config.output_dir, dirs_exist_ok=True)
    results: dict[str, dict[str, Path | str]] = {}
    for sample in config.samples:
        h5ad_path = (
            config.output_dir
            / sample.platform.lower()
            / f"{sample.sample_id}_clustered.h5ad"
        )
        sample_results: dict[str, Path | str] = {"h5ad": h5ad_path}
        if config.write_spatialdata_table:
            clustered = ad.read_h5ad(h5ad_path)
            zarr_path, table_key = write_clustered_spatialdata_table(
                sample.zarr_path,
                clustered,
                segmentation=sample.segmentation,
            )
            sample_results.update(
                spatialdata_zarr=zarr_path,
                spatialdata_table_key=table_key,
            )
            del clustered
        results[sample.sample_id] = sample_results
        force_release(note=f"after clustering finalize {sample.sample_id}")
    return results


def run_clustering_squidpy(
    config: ClusteringSquidpyConfig,
) -> dict[str, dict[str, Path | str]]:
    """Run the legacy single-process clustering path for direct CLI callers."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Path | str]] = {}
    gene_id_lookup = collect_gene_id_lookup_for_samples(config)
    for sample in config.samples:
        adata = load_spatialdata_adata(
            sample.zarr_path,
            platform=sample.platform,
            table_key=sample.table_key,
            shape_key=sample.shape_key,
            gene_id_lookup=gene_id_lookup,
        )
        clustered, artifacts = _cluster_loaded_adata(
            adata,
            config,
            sample=sample,
            sample_dir=config.output_dir / sample.platform.lower(),
        )
        if config.write_spatialdata_table:
            zarr_path, table_key = write_clustered_spatialdata_table(
                sample.zarr_path,
                clustered,
                segmentation=sample.segmentation,
            )
            artifacts.update(
                spatialdata_zarr=zarr_path,
                spatialdata_table_key=table_key,
            )
        results[sample.sample_id] = artifacts
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
    import spatialdata as sd

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


def collapse_atlas_label_to_broad_class(label_name: str) -> str:
    """Collapse an Allen WHB supercluster label to MerXen's broad classes."""
    label = str(label_name).strip()
    if label in NON_NEURON_SUPERCLUSTER_CLASS_MAP:
        return NON_NEURON_SUPERCLUSTER_CLASS_MAP[label]
    if label in EXTRA_SUPERCLUSTER_LABELS:
        return label
    if label in NEURON_SUPERCLUSTER_LABELS:
        return NEURON_CLASS
    lower = label.lower()
    neuron_tokens = (
        "neuron",
        "interneuron",
        "excitatory",
        "inhibitory",
        "intratelencephalic",
        "corticothalamic",
    )
    if any(token in lower for token in neuron_tokens):
        return NEURON_CLASS
    return label


def load_atlas_marker_sets(
    marker_lookup_path: Path | str,
    taxonomy_metadata_path: Path | str,
    *,
    marker_level: str = "CCN202210140_SUPC",
    cluster_membership_path: Path | str | None = None,
) -> list[AtlasMarkerSet]:
    """Read MapMyCells query markers and Allen taxonomy labels.

    Args:
        marker_lookup_path: JSON lookup produced by MapMyCells QueryMarkerRunner.
        taxonomy_metadata_path: Allen ``cluster_annotation_term.csv`` file.
        marker_level: Atlas term set to extract, usually WHB supercluster.
        cluster_membership_path: Optional Allen
            ``cluster_to_cluster_annotation_membership.csv`` path used to infer
            neuronal Exc/Inh/Other splits from neurotransmitter metadata.

    Returns:
        Marker sets with atlas labels and MerXen broad-class mapping.
    """
    marker_lookup = _read_marker_lookup(marker_lookup_path)
    taxonomy = _read_taxonomy_metadata(taxonomy_metadata_path)
    taxonomy = taxonomy[
        taxonomy["cluster_annotation_term_set_label"].astype(str) == marker_level
    ].copy()
    label_to_name = dict(
        zip(
            taxonomy["label"].astype(str),
            taxonomy["name"].astype(str),
            strict=False,
        )
    )
    split_lookup = (
        _supercluster_neuron_split_lookup(
            cluster_membership_path,
            supercluster_level=marker_level,
        )
        if cluster_membership_path is not None
        else {}
    )

    marker_sets: list[AtlasMarkerSet] = []
    for key, marker_ids in marker_lookup.items():
        if "/" not in key or not isinstance(marker_ids, list):
            continue
        level, label_id = key.split("/", 1)
        if level != marker_level or label_id not in label_to_name:
            continue
        label_name = label_to_name[label_id]
        cleaned_markers = tuple(
            str(marker).strip() for marker in marker_ids if str(marker).strip()
        )
        marker_sets.append(
            AtlasMarkerSet(
                level=level,
                label_id=label_id,
                label_name=label_name,
                broad_class=collapse_atlas_label_to_broad_class(label_name),
                marker_ids=cleaned_markers,
                neuron_split=split_lookup.get(
                    label_id,
                    (
                        _neuron_split_for_supercluster(label_name)
                        if collapse_atlas_label_to_broad_class(label_name)
                        == NEURON_CLASS
                        else ""
                    ),
                ),
            )
        )
    return marker_sets


def _supercluster_neuron_split_lookup(
    cluster_membership_path: Path | str,
    *,
    supercluster_level: str,
) -> dict[str, str]:
    path = Path(cluster_membership_path)
    if not path.exists():
        logger.warning("Allen cluster membership metadata does not exist: %s", path)
        return {}
    membership = pd.read_csv(path)
    required = {
        "cluster_annotation_term_label",
        "cluster_annotation_term_set_label",
        "cluster_alias",
        "cluster_annotation_term_name",
    }
    missing = required.difference(membership.columns)
    if missing:
        logger.warning(
            "Allen cluster membership metadata missing columns %s: %s",
            sorted(missing),
            path,
        )
        return {}

    term_set = membership["cluster_annotation_term_set_label"].astype(str)
    supercluster_rows = membership.loc[
        term_set == supercluster_level,
        ["cluster_alias", "cluster_annotation_term_label"],
    ].rename(columns={"cluster_annotation_term_label": "supercluster_label"})
    neurotransmitter_rows = membership.loc[
        term_set == NEUROTRANSMITTER_LEVEL,
        ["cluster_alias", "cluster_annotation_term_name"],
    ].rename(columns={"cluster_annotation_term_name": "neurotransmitter"})
    joined = supercluster_rows.merge(
        neurotransmitter_rows,
        on="cluster_alias",
        how="inner",
    )
    if joined.empty:
        return {}

    split_lookup: dict[str, str] = {}
    for label_id, group in joined.groupby("supercluster_label", sort=False):
        splits = [
            _neuron_split_for_neurotransmitter(value)
            for value in group["neurotransmitter"].astype(str)
        ]
        counts = pd.Series(splits, dtype="object").value_counts()
        if counts.empty:
            continue
        best_split = str(counts.index[0])
        if len(counts) > 1 and int(counts.iloc[0]) == int(counts.iloc[1]):
            best_split = "Other"
        split_lookup[str(label_id)] = best_split
    return split_lookup


def _neuron_split_for_neurotransmitter(value: str) -> str:
    label = str(value).upper()
    has_inhibitory = "GABA" in label or "GLY" in label
    has_excitatory = "VGLUT" in label
    if has_inhibitory and not has_excitatory:
        return "Inhibitory"
    if has_excitatory and not has_inhibitory:
        return "Excitatory"
    return "Other"


def score_clusters_by_atlas_markers(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    marker_sets: list[AtlasMarkerSet],
    marker_alias_lookup: dict[str, str] | None = None,
    min_marker_overlap: int = 3,
    max_markers_per_label: int | None = 80,
    score_margin_threshold: float = 0.0,
    unknown_label: str = UNKNOWN_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score de novo clusters against atlas marker sets.

    Scores are mean gene-wise z-scores over available marker genes. Z-scores are
    computed across de novo cluster means, so each marker contributes relative
    enrichment rather than absolute abundance.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"Expected adata.obs[{cluster_key!r}] for atlas scoring.")

    mean_expression = _mean_expression_by_cluster(adata, cluster_key)
    if mean_expression.empty:
        return (
            pd.DataFrame(columns=ASSIGNMENT_COLUMNS),
            pd.DataFrame(columns=SCORE_COLUMNS),
            pd.DataFrame(columns=MARKER_COLUMNS),
        )

    z_expression = _gene_zscore(mean_expression)
    marker_lookup = _feature_marker_lookup(adata)
    score_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []

    for marker_set in marker_sets:
        marker_indices, resolved_markers = _resolve_marker_indices(
            adata,
            marker_set.marker_ids,
            marker_lookup=marker_lookup,
            marker_alias_lookup=marker_alias_lookup,
            max_markers=max_markers_per_label,
        )
        marker_rows.append(
            {
                "label_id": marker_set.label_id,
                "label_name": marker_set.label_name,
                "broad_class": marker_set.broad_class,
                "neuron_split": marker_set.neuron_split,
                "n_reference_markers": len(marker_set.marker_ids),
                "n_resolved_markers": len(resolved_markers),
                "resolved_markers": ";".join(resolved_markers),
            }
        )
        if len(marker_indices) < int(min_marker_overlap):
            continue
        scores = z_expression.iloc[:, marker_indices].mean(axis=1)
        for cluster, score in scores.items():
            score_rows.append(
                {
                    "cluster": str(cluster),
                    "label_id": marker_set.label_id,
                    "atlas_label": marker_set.label_name,
                    "broad_class": marker_set.broad_class,
                    "score": float(score),
                    "n_markers": len(marker_indices),
                    "resolved_markers": ";".join(resolved_markers),
                }
            )

    score_table = pd.DataFrame(score_rows, columns=SCORE_COLUMNS)
    marker_table = pd.DataFrame(marker_rows, columns=MARKER_COLUMNS)
    assignments = _best_marker_assignments(
        clusters=[str(cluster) for cluster in mean_expression.index],
        score_table=score_table,
        score_margin_threshold=score_margin_threshold,
        unknown_label=unknown_label,
    )
    return assignments, score_table, marker_table


def plot_annotation_score_heatmap(
    score_table: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str,
    dpi: int = 180,
) -> Path:
    """Write a cluster-by-atlas-label score heatmap."""
    output_path = prepare_plot_output(output_path)
    if score_table.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        _plot_empty_spatial_grid_axis(ax, "No atlas marker scores were available.")
        ax.set_title(title)
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        return output_path

    score_matrix = score_table.pivot_table(
        index="cluster",
        columns="atlas_label",
        values="score",
        aggfunc="max",
    ).fillna(0.0)
    height = max(3.5, 0.35 * len(score_matrix.index) + 1.5)
    width = max(7.0, 0.26 * len(score_matrix.columns) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(score_matrix, cmap="vlag", center=0.0, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Atlas label")
    ax.set_ylabel("De novo cluster")
    ax.tick_params(axis="x", labelrotation=75, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def _read_marker_lookup(marker_lookup_path: Path | str) -> dict[str, Any]:
    path = Path(marker_lookup_path)
    if not path.exists():
        raise FileNotFoundError(f"Atlas marker lookup does not exist: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Atlas marker lookup is not a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _read_taxonomy_metadata(taxonomy_metadata_path: Path | str) -> pd.DataFrame:
    path = Path(taxonomy_metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Allen taxonomy metadata does not exist: {path}")
    taxonomy = pd.read_csv(path)
    required = {"label", "name", "cluster_annotation_term_set_label"}
    missing = required.difference(taxonomy.columns)
    if missing:
        raise KeyError(
            f"Allen taxonomy metadata missing columns {sorted(missing)}: {path}"
        )
    return taxonomy


def _best_marker_assignments(
    *,
    clusters: list[str],
    score_table: pd.DataFrame,
    score_margin_threshold: float,
    unknown_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_scores = score_table[score_table["cluster"].astype(str) == cluster]
        if cluster_scores.empty:
            rows.append(_unknown_assignment_row(cluster, unknown_label))
            continue

        ordered = cluster_scores.sort_values("score", ascending=False).reset_index(
            drop=True
        )
        best = ordered.iloc[0]
        runner_up = ordered.iloc[1] if len(ordered) > 1 else None
        runner_score = (
            float(runner_up["score"]) if runner_up is not None else float("nan")
        )
        margin = (
            float(best["score"]) - runner_score
            if np.isfinite(runner_score)
            else float("inf")
        )
        if margin < float(score_margin_threshold):
            rows.append(_unknown_assignment_row(cluster, unknown_label))
            continue

        rows.append(
            {
                "cluster": cluster,
                "atlas_label": str(best["atlas_label"]),
                "broad_class": str(best["broad_class"]),
                "score": float(best["score"]),
                "runner_up_label": (
                    str(runner_up["atlas_label"]) if runner_up is not None else ""
                ),
                "runner_up_score": runner_score,
                "score_margin": margin,
                "n_markers": int(best["n_markers"]),
            }
        )
    return pd.DataFrame(rows)


def _unknown_assignment_row(cluster: str, unknown_label: str) -> dict[str, Any]:
    return {
        "cluster": cluster,
        "atlas_label": unknown_label,
        "broad_class": unknown_label,
        "score": float("nan"),
        "runner_up_label": "",
        "runner_up_score": float("nan"),
        "score_margin": float("nan"),
        "n_markers": 0,
    }


def _mean_expression_by_cluster(
    adata: ad.AnnData,
    cluster_key: str,
) -> pd.DataFrame:
    labels = pd.Series(adata.obs[cluster_key], index=adata.obs_names)
    labels = labels.astype("string").fillna("unassigned")
    categories = [str(label) for label in pd.unique(labels)]
    matrix = _matrix_to_dense(adata.X).astype(float, copy=False)
    means: list[np.ndarray] = []
    label_values = labels.astype(str).to_numpy()
    for category in categories:
        mask = label_values == category
        if mask.any():
            means.append(np.asarray(matrix[mask, :].mean(axis=0)).reshape(-1))
        else:
            means.append(np.zeros(adata.n_vars, dtype=float))
    return pd.DataFrame(
        np.vstack(means) if means else np.empty((0, adata.n_vars)),
        index=pd.Index(categories, name=cluster_key),
        columns=pd.Index(adata.var_names.astype(str), name="gene"),
    )


def _gene_zscore(mean_expression: pd.DataFrame) -> pd.DataFrame:
    values = mean_expression.to_numpy(dtype=float)
    centered = values - np.nanmean(values, axis=0, keepdims=True)
    scale = np.nanstd(centered, axis=0, keepdims=True)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    z_values = centered / scale
    z_values[~np.isfinite(z_values)] = 0.0
    return pd.DataFrame(
        z_values,
        index=mean_expression.index,
        columns=mean_expression.columns,
    )


def _feature_marker_lookup(adata: ad.AnnData) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for column in GENE_ID_COLUMN_CANDIDATES:
        if column not in adata.var:
            continue
        values = adata.var[column].astype(str).to_numpy()
        for idx, value in enumerate(values):
            cleaned = str(value).strip()
            if cleaned and cleaned.lower() not in {"nan", "none"}:
                lookup.setdefault(_normalize_marker_id(cleaned), idx)

    for idx, var_name in enumerate(adata.var_names.astype(str)):
        lookup.setdefault(_normalize_marker_id(var_name), idx)

    for column in GENE_SYMBOL_COLUMN_CANDIDATES:
        if column not in adata.var:
            continue
        values = adata.var[column].astype(str).to_numpy()
        for idx, value in enumerate(values):
            cleaned = str(value).strip()
            if cleaned and cleaned.lower() not in {"nan", "none"}:
                lookup.setdefault(_normalize_marker_id(cleaned), idx)
    return lookup


def _resolve_marker_indices(
    adata: ad.AnnData,
    marker_ids: tuple[str, ...],
    *,
    marker_lookup: dict[str, int],
    marker_alias_lookup: dict[str, str] | None,
    max_markers: int | None,
) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    names: list[str] = []
    seen_indices: set[int] = set()
    aliases = marker_alias_lookup or {}
    for marker_id in marker_ids:
        normalized_marker = _normalize_marker_id(marker_id)
        candidate_ids = [marker_id]
        alias = aliases.get(normalized_marker)
        if alias is not None:
            candidate_ids.append(alias)
        idx = None
        for candidate_id in candidate_ids:
            idx = marker_lookup.get(_normalize_marker_id(candidate_id))
            if idx is not None:
                break
        if idx is None or idx in seen_indices:
            continue
        indices.append(idx)
        names.append(str(adata.var_names[idx]))
        seen_indices.add(idx)
        if max_markers is not None and len(indices) >= int(max_markers):
            break
    return indices, names


def _normalize_marker_id(value: str) -> str:
    return str(value).strip().upper()


def _matrix_to_dense(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _copy_matrix(matrix: Any) -> Any:
    return matrix.copy() if hasattr(matrix, "copy") else np.array(matrix, copy=True)


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    min_value = float(finite.min())
    max_value = float(finite.max())
    span = max_value - min_value
    padding = 0.02 * span if span > 0 else 1.0
    return (min_value - padding, max_value + padding)


def _plot_empty_spatial_grid_axis(ax: plt.Axes, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _wrapped_cluster_title(value: str, *, width: int = 34) -> str:
    wrapped = textwrap.wrap(str(value), width=width)
    return "\n".join(wrapped) if wrapped else str(value)


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
        aligned_preferred = f"{preferred}_aligned_nonrigid"
        if platform.upper() == "MERSCOPE" and aligned_preferred in sdata_obj.shapes:
            return aligned_preferred
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


def _shape_metrics(shapes: Any) -> pd.DataFrame:
    gdf = shapes.copy()
    if "geometry" not in gdf.columns:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame({"geometry": gdf.geometry}, index=gdf.index)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    id_col = first_existing_col(
        gdf,
        [
            "instance_id",
            "cell_id",
            "cell",
            "cells",
            "cell_ID",
            "region",
            "label_id",
            "EntityID",
        ],
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


def _robust_centroid_xy(gdf: Any) -> tuple[np.ndarray, np.ndarray]:
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
        if "instance_id" not in adata.obs and "cell_id" not in adata.obs:
            adata.obs["cell_id"] = table_ids.astype(str).to_numpy()
        return

    if len(metrics) == adata.n_obs:
        adata.obsm["spatial"] = metrics[["spatial_x", "spatial_y"]].to_numpy(float)
        if "cell_area" not in adata.obs or adata.obs["cell_area"].isna().all():
            adata.obs["cell_area"] = metrics["cell_area"].to_numpy(float)
        if "instance_id" not in adata.obs and "cell_id" not in adata.obs:
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
    attrs = dict(adata.uns.get("spatialdata_attrs", {}))
    instance_key = attrs.get("instance_key")
    if isinstance(instance_key, str) and instance_key in adata.obs.columns:
        return pd.Index(adata.obs[instance_key].astype(str))
    for col in ["instance_id", "cell_id", "cell", "cells", "cell_ID", "EntityID"]:
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
