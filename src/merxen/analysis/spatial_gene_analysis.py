"""Per-gene spatial autocorrelation analysis with Squidpy."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

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
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from merxen.analysis.clustering_squidpy import (
    load_spatialdata_adata,
    remove_control_features,
)
from merxen.analysis.transcript_spatial_patterns import (
    COMPARTMENTS,
    TranscriptPatternResults,
    run_transcript_pattern_analysis,
    transcript_indices_for_gene,
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
        n_cells = analysis_adata.n_obs
        n_genes = analysis_adata.n_vars
        del adata, analysis_adata
        force_release(note=f"before transcript spatial analysis {sample.sample_id}")

        transcript_outputs: dict[str, Path] = {}
        transcript_counts: dict[str, int | float] = {}
        transcript_gene_plots: dict[str, Path] = {}
        if config.transcript_analysis_enabled:
            import spatialdata as sd

            sdata_obj = sd.read_zarr(sample.zarr_path)
            try:
                transcript_results = run_transcript_pattern_analysis(
                    sdata_obj=sdata_obj,
                    sample=sample,
                    config=config,
                )
                summary_path = (
                    sample_dir / f"{sample.sample_id}_transcript_spatial_patterns.csv"
                )
                paircorr_path = (
                    sample_dir
                    / f"{sample.sample_id}_transcript_pair_correlation.parquet"
                )
                signed_distance_path = (
                    sample_dir
                    / f"{sample.sample_id}_transcript_signed_distance.parquet"
                )
                transcript_rankings_path = (
                    sample_dir
                    / f"{sample.sample_id}_transcript_spatial_pattern_rankings.csv"
                )
                transcript_results.summary.to_csv(summary_path, index=False)
                transcript_results.signed_distance.to_parquet(
                    signed_distance_path,
                    index=False,
                )
                transcript_results.paircorr.to_parquet(paircorr_path, index=False)
                transcript_results.rankings.to_csv(
                    transcript_rankings_path,
                    index=False,
                )
                transcript_plot_dir = sample_dir / "plots" / "transcript_patterns"
                transcript_gene_plots = plot_transcript_pattern_diagnostics(
                    transcript_results,
                    cell_shapes=sdata_obj.shapes[sample.shape_key],
                    nuclei_shapes=sdata_obj.shapes[sample.nuclei_shape_key],
                    output_dir=transcript_plot_dir,
                    sample_id=sample.sample_id,
                    top_n=config.transcript_diagnostic_top_n,
                    max_genes=config.transcript_diagnostic_max_genes,
                    window_um=config.transcript_diagnostic_window_um,
                    max_points=config.transcript_plot_max_points,
                    dpi=config.figure_dpi,
                )
                transcript_outputs = {
                    "transcript_summary_csv": summary_path,
                    "signed_distance_parquet": signed_distance_path,
                    "paircorr_parquet": paircorr_path,
                    "transcript_rankings_csv": transcript_rankings_path,
                    "transcript_diagnostics_dir": transcript_plot_dir,
                }
                transcript_counts = {
                    "n_transcripts_input": transcript_results.data.n_input,
                    "n_transcripts_in_tissue": len(transcript_results.data.coordinates),
                    "n_transcripts_outside_tissue": (
                        transcript_results.data.n_outside_tissue
                    ),
                    "n_transcripts_invalid_coordinates": (
                        transcript_results.data.n_invalid_coordinates
                    ),
                    "n_control_transcripts_excluded": (
                        transcript_results.data.n_controls_excluded
                    ),
                    "n_transcript_genes": len(transcript_results.data.gene_names),
                    "n_cell_overlap_ambiguous": int(
                        np.count_nonzero(transcript_results.data.cell_overlap_count > 1)
                    ),
                    "n_nucleus_overlap_ambiguous": int(
                        np.count_nonzero(
                            transcript_results.data.nucleus_overlap_count > 1
                        )
                    ),
                    "tissue_area_um2": float(transcript_results.tissue_polygon.area),
                }
                compartment_counts = np.bincount(
                    transcript_results.data.compartments,
                    minlength=len(COMPARTMENTS),
                )
                transcript_counts.update(
                    {
                        f"n_{compartment}_transcripts": int(compartment_counts[code])
                        for code, compartment in enumerate(COMPARTMENTS)
                    }
                )
                del transcript_results
            finally:
                del sdata_obj
                force_release(
                    note=f"after transcript spatial analysis {sample.sample_id}"
                )
        manifest = write_spatial_gene_analysis_manifest(
            sample_dir / f"{sample.sample_id}_spatial_gene_analysis_manifest.json",
            sample_id=sample.sample_id,
            platform=sample.platform,
            metrics_csv=metrics_csv,
            rankings_csv=ranking_csv,
            distribution_plot=distribution_plot,
            gene_plots=gene_plots,
            n_cells=n_cells,
            n_genes=n_genes,
            config=config,
            transcript_outputs=transcript_outputs,
            transcript_counts=transcript_counts,
            transcript_gene_plots=transcript_gene_plots,
        )

        results[sample.sample_id] = {
            "metrics_csv": metrics_csv,
            "rankings_csv": ranking_csv,
            "distribution_plot": distribution_plot,
            "manifest": manifest,
            **transcript_outputs,
        }
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


def plot_transcript_pattern_diagnostics(
    results: TranscriptPatternResults,
    *,
    cell_shapes: object,
    nuclei_shapes: object,
    output_dir: Path,
    sample_id: str,
    top_n: int = 3,
    max_genes: int = 30,
    window_um: float = 250.0,
    max_points: int = 20_000,
    dpi: int = 180,
) -> dict[str, Path]:
    """Plot tissue and subcellular views for representative ranked genes."""
    genes = _select_transcript_diagnostic_genes(
        results.rankings,
        top_n=top_n,
        max_genes=max_genes,
    )
    if not genes:
        return {}

    cell_geometries = _plot_geometries(cell_shapes)
    nuclei_geometries = _plot_geometries(nuclei_shapes)
    cell_tree = STRtree(cell_geometries)
    nucleus_tree = STRtree(nuclei_geometries)
    summary_by_gene = results.summary.set_index("gene")
    paths: dict[str, Path] = {}
    for gene in genes:
        indices = transcript_indices_for_gene(results.data, gene)
        if len(indices) == 0:
            continue
        coordinates = results.data.coordinates[indices]
        compartments = results.data.compartments[indices]
        tissue_take = _even_sample_indices(len(indices), max_points)
        center = _densest_window_center(
            coordinates,
            window_um=window_um,
            max_points=min(max_points, 5_000),
        )
        half = float(window_um) / 2.0
        local = (
            (coordinates[:, 0] >= center[0] - half)
            & (coordinates[:, 0] <= center[0] + half)
            & (coordinates[:, 1] >= center[1] - half)
            & (coordinates[:, 1] <= center[1] + half)
        )

        fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
        _plot_transcript_map(
            axes[0, 0],
            coordinates[tissue_take],
            compartments[tissue_take],
            title="Tissue-wide transcript locations",
        )
        _plot_geometry_outline(
            axes[0, 0],
            results.tissue_polygon,
            color="#262626",
            linewidth=0.8,
        )

        local_box = box(
            center[0] - half,
            center[1] - half,
            center[0] + half,
            center[1] + half,
        )
        _plot_local_shape_outlines(
            axes[0, 1],
            tree=cell_tree,
            geometries=cell_geometries,
            window=local_box,
            color="#777777",
            linewidth=0.45,
        )
        _plot_local_shape_outlines(
            axes[0, 1],
            tree=nucleus_tree,
            geometries=nuclei_geometries,
            window=local_box,
            color="#2b6cb0",
            linewidth=0.7,
        )
        _plot_transcript_map(
            axes[0, 1],
            coordinates[local],
            compartments[local],
            title=f"Densest {window_um:g} µm window",
        )
        axes[0, 1].set_xlim(center[0] - half, center[0] + half)
        axes[0, 1].set_ylim(center[1] - half, center[1] + half)

        _plot_signed_distance_profile(axes[1, 0], results, gene)
        _plot_paircorr_profile(axes[1, 1], results, gene)
        pattern = str(summary_by_gene.loc[gene, "pattern_label"])
        fig.suptitle(
            f"{sample_id}: {gene} · {pattern} · n={len(indices):,}",
            fontsize=14,
        )
        fig.tight_layout()
        out_path = prepare_plot_output(
            output_dir / f"{sample_id}_{_safe_filename(gene)}.png"
        )
        save_figure(fig, out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths[gene] = out_path
    return paths


def _select_transcript_diagnostic_genes(
    rankings: pd.DataFrame,
    *,
    top_n: int,
    max_genes: int,
) -> list[str]:
    if rankings.empty:
        return []
    categories = (
        lambda metric: not metric.startswith(("paircorr_", "signed_distance_")),
        lambda metric: metric.startswith("paircorr_compartment_"),
        lambda metric: metric.startswith("signed_distance_"),
    )
    selected: list[str] = []
    quota = max(1, int(max_genes) // len(categories))
    for belongs in categories:
        category_genes: list[str] = []
        for row in rankings.itertuples(index=False):
            if int(row.rank) > int(top_n) or not belongs(str(row.metric)):
                continue
            gene = str(row.gene)
            if gene not in selected and gene not in category_genes:
                category_genes.append(gene)
            if len(category_genes) >= quota:
                break
        selected.extend(category_genes)
    if len(selected) < max_genes:
        for row in rankings.itertuples(index=False):
            if int(row.rank) > int(top_n):
                continue
            gene = str(row.gene)
            if gene not in selected:
                selected.append(gene)
            if len(selected) >= max_genes:
                break
    return selected[: int(max_genes)]


def _plot_geometries(shapes: Any) -> np.ndarray:
    geometries = np.asarray(shapes.geometry, dtype=object)
    return np.asarray(
        [geom for geom in geometries if geom is not None and not geom.is_empty],
        dtype=object,
    )


def _even_sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=int(maximum), dtype=np.int64)


def _densest_window_center(
    coordinates: np.ndarray,
    *,
    window_um: float,
    max_points: int,
) -> np.ndarray:
    take = _even_sample_indices(len(coordinates), max_points)
    sampled = np.asarray(coordinates[take], dtype=float)
    if len(sampled) == 1:
        return np.asarray(sampled[0], dtype=float)
    tree = cKDTree(sampled)
    counts = tree.query_ball_point(
        sampled,
        r=float(window_um) / 2.0,
        return_length=True,
    )
    return np.asarray(sampled[int(np.argmax(counts))], dtype=float)


def _plot_transcript_map(
    ax: plt.Axes,
    coordinates: np.ndarray,
    compartments: np.ndarray,
    *,
    title: str,
) -> None:
    colors = ("#805ad5", "#dd6b20", "#319795")
    for code, (label, color) in enumerate(zip(COMPARTMENTS, colors, strict=True)):
        take = compartments == code
        if take.any():
            ax.scatter(
                coordinates[take, 0],
                coordinates[take, 1],
                s=3.0,
                c=color,
                alpha=0.65,
                linewidths=0,
                label=label,
                rasterized=True,
            )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.legend(loc="best", markerscale=2.5, frameon=False)


def _plot_local_shape_outlines(
    ax: plt.Axes,
    *,
    tree: STRtree,
    geometries: np.ndarray,
    window: BaseGeometry,
    color: str,
    linewidth: float,
) -> None:
    indices = np.asarray(tree.query(window), dtype=np.int64)[:2_000]
    for index in indices:
        _plot_geometry_outline(
            ax,
            geometries[index],
            color=color,
            linewidth=linewidth,
        )


def _plot_geometry_outline(
    ax: plt.Axes,
    geometry: BaseGeometry,
    *,
    color: str,
    linewidth: float,
) -> None:
    boundary = geometry.boundary
    parts = getattr(boundary, "geoms", (boundary,))
    for part in parts:
        coords = np.asarray(part.coords, dtype=float)
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=0.8,
        )


def _plot_signed_distance_profile(
    ax: plt.Axes,
    results: TranscriptPatternResults,
    gene: str,
) -> None:
    rows = results.signed_distance[results.signed_distance["gene"] == gene]
    for boundary, color in (("cell", "#dd6b20"), ("nucleus", "#2b6cb0")):
        subset = rows[rows["boundary"] == boundary].sort_values("bin_index")
        ax.plot(
            subset["bin_index"],
            subset["enrichment_log2_odds"],
            marker="o",
            markersize=3.5,
            color=color,
            label=boundary,
        )
    labels = (
        rows[rows["boundary"] == "cell"]
        .sort_values("bin_index")["bin_label"]
        .astype(str)
    )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=70, ha="right")
    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_title("Signed-distance enrichment")
    ax.set_ylabel("log2 odds vs all transcripts")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)


def _plot_paircorr_profile(
    ax: plt.Axes,
    results: TranscriptPatternResults,
    gene: str,
) -> None:
    rows = (
        results.paircorr[results.paircorr["gene"] == gene]
        if "gene" in results.paircorr.columns
        else pd.DataFrame()
    )
    if rows.empty:
        ax.text(0.5, 0.5, "Below pair-correlation count threshold", ha="center")
        ax.set_axis_off()
        return
    colors = {"global": "#4a5568", "compartment_stratified": "#c53030"}
    for null_model, subset in rows.groupby("null_model", sort=False):
        subset = subset.sort_values("band_index")
        ax.plot(
            subset["band_index"],
            subset["paircorr_enrichment"],
            marker="o",
            color=colors[str(null_model)],
            label=str(null_model).replace("_", " "),
        )
    labels = (
        rows[rows["null_model"] == "global"]
        .sort_values("band_index")["band_label"]
        .astype(str)
    )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_title("Multiscale same-gene pair enrichment")
    ax.set_ylabel("observed / random-label mean")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.6)


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
    transcript_outputs: dict[str, Path] | None = None,
    transcript_counts: dict[str, int | float] | None = None,
    transcript_gene_plots: dict[str, Path] | None = None,
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
            "transcript_analysis_enabled": config.transcript_analysis_enabled,
            "transcript_min_count": config.transcript_min_count,
            "paircorr_min_count": config.paircorr_min_count,
            "paircorr_max_transcripts_per_gene": (
                config.paircorr_max_transcripts_per_gene
            ),
            "paircorr_distance_edges_um": config.paircorr_distance_edges_um,
            "paircorr_permutations": config.paircorr_permutations,
            "paircorr_seed": config.paircorr_seed,
            "paircorr_n_jobs": config.paircorr_n_jobs,
            "pericellular_distance_um": config.pericellular_distance_um,
            "membrane_distance_um": config.membrane_distance_um,
            "signed_distance_edges_um": config.signed_distance_edges_um,
            "transcript_diagnostic_top_n": config.transcript_diagnostic_top_n,
            "transcript_diagnostic_max_genes": (config.transcript_diagnostic_max_genes),
            "transcript_diagnostic_window_um": (config.transcript_diagnostic_window_um),
            "transcript_plot_max_points": config.transcript_plot_max_points,
        },
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "rankings_csv": str(rankings_csv),
            "distribution_plot": str(distribution_plot),
            "gene_plots": {key: str(value) for key, value in gene_plots.items()},
            "transcript_gene_plots": {
                key: str(value) for key, value in (transcript_gene_plots or {}).items()
            },
            **{key: str(value) for key, value in (transcript_outputs or {}).items()},
        },
        "transcript_counts": transcript_counts or {},
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
