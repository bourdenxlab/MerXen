"""Per-gene spatial autocorrelation analysis with Squidpy."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from scipy import sparse

from merxen.analysis.clustering_squidpy import (
    load_spatialdata_adata,
    remove_control_features,
)
from merxen.config import SpatialGeneAnalysisConfig
from merxen.memory import force_release, log_status
from merxen.plotting import prepare_plot_output, save_figure

logger = logging.getLogger(__name__)


def run_spatial_gene_analysis(
    config: SpatialGeneAnalysisConfig,
) -> dict[str, dict[str, Path]]:
    """Run per-gene Moran's I and Geary's C analysis for every sample."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Path]] = {}

    for sample in config.samples:
        sample_dir = config.output_dir / sample.platform.lower()
        sample_dir.mkdir(parents=True, exist_ok=True)
        log_status(
            f"[{sample.sample_id}] Starting spatial_gene_analysis "
            f"(platform={sample.platform})"
        )

        adata = load_spatialdata_adata(
            sample.zarr_path,
            platform=sample.platform,
            table_key=sample.table_key,
            shape_key=sample.shape_key,
        )
        analysis_adata = prepare_spatial_autocorr_adata(
            adata,
            drop_control_features=config.drop_control_features,
            min_counts=config.min_counts,
            min_cells=config.min_cells,
            normalize_target_sum=config.normalize_target_sum,
            normalize_exclude_highly_expressed=(
                config.normalize_exclude_highly_expressed
            ),
            normalize_max_fraction=config.normalize_max_fraction,
        )
        add_spatial_neighbors(analysis_adata, n_neighbors=config.n_neighbors)

        metrics = compute_spatial_autocorrelation(analysis_adata)
        metrics_csv = save_spatial_autocorr_metrics(
            metrics,
            sample_dir / f"{sample.sample_id}_spatial_gene_autocorrelation.csv",
        )
        ranking_table = ranked_spatial_autocorr_genes(metrics, top_n=config.top_n)
        ranking_csv = save_spatial_autocorr_rankings(
            ranking_table,
            sample_dir
            / f"{sample.sample_id}_spatial_gene_autocorrelation_rankings.csv",
        )
        distribution_plot = plot_autocorr_distributions(
            metrics,
            sample_dir
            / "plots"
            / "distributions"
            / f"{sample.sample_id}_spatial_autocorrelation_distribution.png",
            sample_label=sample.sample_id,
            dpi=config.figure_dpi,
        )
        gene_plots = plot_ranked_spatial_genes(
            analysis_adata,
            ranking_table,
            output_dir=sample_dir / "plots" / "spatial_genes",
            sample_id=sample.sample_id,
            point_size=config.spatial_point_size,
            dpi=config.figure_dpi,
        )
        manifest = write_spatial_gene_analysis_manifest(
            sample_dir / f"{sample.sample_id}_spatial_gene_analysis_manifest.json",
            sample_id=sample.sample_id,
            platform=sample.platform,
            metrics_csv=metrics_csv,
            rankings_csv=ranking_csv,
            distribution_plot=distribution_plot,
            gene_plots=gene_plots,
            n_cells=analysis_adata.n_obs,
            n_genes=analysis_adata.n_vars,
            config=config,
        )

        results[sample.sample_id] = {
            "metrics_csv": metrics_csv,
            "rankings_csv": ranking_csv,
            "distribution_plot": distribution_plot,
            "manifest": manifest,
        }
        del adata, analysis_adata
        force_release(note=f"after spatial_gene_analysis {sample.sample_id}")

    return results


def prepare_spatial_autocorr_adata(
    adata: ad.AnnData,
    *,
    drop_control_features: bool = True,
    min_counts: int = 0,
    min_cells: int = 5,
    normalize_target_sum: float | None = None,
    normalize_exclude_highly_expressed: bool = False,
    normalize_max_fraction: float = 0.05,
) -> ad.AnnData:
    """Return a filtered, log-normalized AnnData for gene autocorrelation."""
    prepared = remove_control_features(adata) if drop_control_features else adata.copy()
    if min_counts > 0:
        sc.pp.filter_cells(prepared, min_counts=int(min_counts))
    sc.pp.filter_genes(prepared, min_cells=int(min_cells))
    if prepared.n_obs < 3 or prepared.n_vars < 1:
        raise ValueError(
            "Too few cells/genes remain for spatial gene analysis: "
            f"n_obs={prepared.n_obs}, n_vars={prepared.n_vars}"
        )
    sc.pp.normalize_total(
        prepared,
        target_sum=normalize_target_sum,
        exclude_highly_expressed=bool(normalize_exclude_highly_expressed),
        max_fraction=float(normalize_max_fraction),
        inplace=True,
    )
    sc.pp.log1p(prepared)
    return prepared


def add_spatial_neighbors(adata: ad.AnnData, *, n_neighbors: int = 6) -> ad.AnnData:
    """Add a generic-coordinate spatial neighbor graph for Squidpy."""
    if "spatial" not in adata.obsm:
        raise KeyError("Expected adata.obsm['spatial'] for spatial gene analysis.")
    if adata.n_obs < 2:
        raise ValueError("Spatial neighbor graph requires at least two cells.")
    effective_neighbors = max(1, min(int(n_neighbors), adata.n_obs - 1))
    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        n_neighs=effective_neighbors,
    )
    return adata


def compute_spatial_autocorrelation(adata: ad.AnnData) -> pd.DataFrame:
    """Compute Moran's I and Geary's C for every retained gene."""
    sq.gr.spatial_autocorr(adata, mode="moran", genes=list(adata.var_names))
    sq.gr.spatial_autocorr(adata, mode="geary", genes=list(adata.var_names))

    moran = _autocorr_frame_from_uns(adata, key="moranI", score_column="I")
    geary = _autocorr_frame_from_uns(adata, key="gearyC", score_column="C")

    metrics = pd.DataFrame(index=adata.var_names.astype(str))
    metrics.index.name = "gene"
    metrics["moran_i"] = moran["score"]
    metrics["moran_i_pval_norm"] = moran.get("pval_norm")
    metrics["moran_i_pval_fdr_bh"] = moran.get("pval_fdr_bh")
    metrics["geary_c"] = geary["score"]
    metrics["geary_c_pval_norm"] = geary.get("pval_norm")
    metrics["geary_c_pval_fdr_bh"] = geary.get("pval_fdr_bh")
    return metrics.reset_index()


def ranked_spatial_autocorr_genes(
    metrics: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top and bottom ranked genes for Moran's I and Geary's C."""
    rankings: list[pd.DataFrame] = []
    ranking_specs = [
        ("moran_i", "top", False),
        ("moran_i", "bottom", True),
        ("geary_c", "top", False),
        ("geary_c", "bottom", True),
    ]
    for metric, direction, ascending in ranking_specs:
        ranked = (
            metrics.dropna(subset=[metric])
            .sort_values(metric, ascending=ascending)
            .head(int(top_n))
            .copy()
        )
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.insert(0, "direction", direction)
        ranked.insert(0, "metric", metric)
        rankings.append(ranked)
    if not rankings:
        return pd.DataFrame(
            columns=[
                "metric",
                "direction",
                "rank",
                *metrics.columns.to_list(),
            ]
        )
    return pd.concat(rankings, ignore_index=True)


def save_spatial_autocorr_metrics(metrics: pd.DataFrame, output_path: Path) -> Path:
    """Write the full per-gene autocorrelation metric table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    return output_path


def save_spatial_autocorr_rankings(rankings: pd.DataFrame, output_path: Path) -> Path:
    """Write the ranked top/bottom gene table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output_path, index=False)
    return output_path


def plot_autocorr_distributions(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    sample_label: str,
    dpi: int = 180,
) -> Path:
    """Plot the distributions of Moran's I and Geary's C values."""
    output_path = prepare_plot_output(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    plot_specs = [
        ("moran_i", "Moran's I", "#3a6ea5"),
        ("geary_c", "Geary's C", "#c25b45"),
    ]
    for ax, (column, label, color) in zip(axes, plot_specs, strict=True):
        values = pd.to_numeric(metrics[column], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) == 0:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#555555",
            )
        else:
            sns.histplot(
                values,
                bins=min(40, max(8, len(values) // 4)),
                ax=ax,
                color=color,
            )
            ax.axvline(values.median(), color="#222222", linewidth=1.0, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Genes")
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.7)

    fig.suptitle(f"{sample_label} spatial autocorrelation")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ranked_spatial_genes(
    adata: ad.AnnData,
    rankings: pd.DataFrame,
    *,
    output_dir: Path,
    sample_id: str,
    point_size: float = 2.0,
    dpi: int = 180,
) -> dict[str, Path]:
    """Save one spatial expression plot for each ranked gene row."""
    output_paths: dict[str, Path] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in rankings.itertuples(index=False):
        metric = str(row.metric)
        direction = str(row.direction)
        gene = str(row.gene)
        key = (metric, direction, gene)
        if key in seen:
            continue
        seen.add(key)
        if gene not in adata.var_names:
            logger.warning("[%s] Ranked gene %s is not in AnnData.", sample_id, gene)
            continue
        out_path = (
            output_dir
            / metric
            / direction
            / f"{sample_id}_{metric}_{direction}_{_safe_filename(gene)}.png"
        )
        output_paths[f"{metric}_{direction}_{gene}"] = plot_spatial_gene_expression(
            adata,
            gene,
            out_path,
            title=f"{gene} ({metric} {direction})",
            point_size=point_size,
            dpi=dpi,
        )
    return output_paths


def plot_spatial_gene_expression(
    adata: ad.AnnData,
    gene: str,
    output_path: Path,
    *,
    title: str | None = None,
    point_size: float = 2.0,
    dpi: int = 180,
) -> Path:
    """Plot one gene's expression over spatial coordinates."""
    output_path = prepare_plot_output(output_path)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Expected adata.obsm['spatial'] with at least two columns.")
    expression = _gene_expression_vector(adata, gene)
    finite_expr = expression[np.isfinite(expression)]
    vmax = float(np.percentile(finite_expr, 99)) if finite_expr.size else 1.0
    vmin = float(np.percentile(finite_expr, 1)) if finite_expr.size else 0.0
    if np.isclose(vmin, vmax):
        vmin = float(finite_expr.min()) if finite_expr.size else 0.0
        vmax = float(finite_expr.max()) if finite_expr.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=expression,
        s=float(point_size),
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=0.9,
    )
    ax.set_title(title or gene)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("log-normalized expression")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_spatial_gene_analysis_manifest(
    output_path: Path,
    *,
    sample_id: str,
    platform: str,
    metrics_csv: Path,
    rankings_csv: Path,
    distribution_plot: Path,
    gene_plots: dict[str, Path],
    n_cells: int,
    n_genes: int,
    config: SpatialGeneAnalysisConfig,
) -> Path:
    """Write a compact JSON manifest for one sample's outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sample_id": sample_id,
        "platform": platform,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "parameters": {
            "drop_control_features": config.drop_control_features,
            "min_counts": config.min_counts,
            "min_cells": config.min_cells,
            "normalize_target_sum": config.normalize_target_sum,
            "normalize_exclude_highly_expressed": (
                config.normalize_exclude_highly_expressed
            ),
            "normalize_max_fraction": config.normalize_max_fraction,
            "n_neighbors": config.n_neighbors,
            "top_n": config.top_n,
            "spatial_point_size": config.spatial_point_size,
            "figure_dpi": config.figure_dpi,
        },
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "rankings_csv": str(rankings_csv),
            "distribution_plot": str(distribution_plot),
            "gene_plots": {key: str(value) for key, value in gene_plots.items()},
        },
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path


def _autocorr_frame_from_uns(
    adata: ad.AnnData,
    *,
    key: str,
    score_column: str,
) -> pd.DataFrame:
    if key not in adata.uns:
        raise KeyError(f"Squidpy did not write adata.uns[{key!r}].")
    raw = pd.DataFrame(adata.uns[key]).copy()
    if score_column not in raw.columns:
        raise KeyError(
            f"Expected Squidpy autocorrelation column {score_column!r} "
            f"in adata.uns[{key!r}]. Found {list(raw.columns)}."
        )
    if "genes" in raw.columns:
        raw = raw.set_index("genes")
    raw.index = raw.index.astype(str)
    renamed = pd.DataFrame(index=raw.index)
    renamed["score"] = pd.to_numeric(raw[score_column], errors="coerce")
    for pval_column in ["pval_norm", "pval_fdr_bh"]:
        if pval_column in raw.columns:
            renamed[pval_column] = pd.to_numeric(raw[pval_column], errors="coerce")
    return renamed


def _gene_expression_vector(adata: ad.AnnData, gene: str) -> np.ndarray:
    matrix = adata[:, [gene]].X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=float).reshape(-1)


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "gene"
