"""Mutually exclusive co-expression rate (MECR) analysis.

The implementation follows Hartman and Satija's paper-standard definition:
reference markers are detected in more than 25% of one broad class and less
than 1% of all other retained classes, genes qualifying for multiple classes
are removed, and spatial co-expression is the binary intersection divided by
the binary union for every cross-class marker pair.
"""

from __future__ import annotations

import json
import logging
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
from scipy import sparse

from merxen.analysis.clustering_squidpy import (
    collapse_atlas_label_to_broad_class,
)
from merxen.config import MecrConfig, MecrReferenceConfig, MecrSampleConfig
from merxen.memory import force_release, log_status
from merxen.plotting import prepare_plot_output, save_figure

logger = logging.getLogger(__name__)

MECR_REFERENCE_STATS_NAME = "mecr_reference_gene_statistics.csv"
MECR_REFERENCE_MARKERS_NAME = "mecr_reference_markers.csv"
MECR_REFERENCE_PANEL_NAME = "mecr_reference_panel_genes.csv"
MECR_REFERENCE_MANIFEST_NAME = "mecr_reference_manifest.json"
MECR_REFERENCE_PAIRS_NAME = "mecr_reference_pairs.csv"
MECR_REFERENCE_DISTRIBUTION_NAME = "mecr_reference_distribution.png"
PAIR_ID_COLUMNS = ["gene_1", "class_1", "gene_2", "class_2"]
BARNYARD_CANONICAL_PAIRS = (
    ("SLC17A7", "GFAP"),
    ("GAD1", "GFAP"),
    ("SLC17A7", "GAD1"),
)
PLATFORM_COLORS = {"MERSCOPE": "#2878B5", "XENIUM": "#D95F02"}


def run_mecr_reference(config: MecrReferenceConfig) -> dict[str, Path]:
    """Discover mutually exclusive broad-class markers in the WHB reference.

    Args:
        config: Validated WHB reference and marker-discovery configuration.

    Returns:
        Paths to the marker table, full gene statistics, panel genes, and
        reproducibility manifest.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    panel_genes = collect_spatial_panel_genes(config.samples)
    if not panel_genes:
        raise ValueError("No spatial panel genes were found for MECR reference prep")

    panel_path = config.output_dir / MECR_REFERENCE_PANEL_NAME
    pd.DataFrame({"gene": panel_genes}).to_csv(panel_path, index=False)
    log_status(
        f"Preparing WHB MECR reference for {len(panel_genes):,} spatial panel genes"
    )
    reference = load_whb_panel_reference(config, panel_genes=panel_genes)
    try:
        statistics, markers = discover_reference_markers(
            reference,
            class_key="broad_class",
            target_classes=config.target_broad_classes,
            min_target_fraction=config.marker_min_target_fraction,
            max_other_fraction=config.marker_max_other_fraction,
            tie_correct=config.wilcoxon_tie_correct,
        )
        stats_path = config.output_dir / MECR_REFERENCE_STATS_NAME
        markers_path = config.output_dir / MECR_REFERENCE_MARKERS_NAME
        reference_pairs_path = config.output_dir / MECR_REFERENCE_PAIRS_NAME
        reference_distribution_path = (
            config.output_dir / MECR_REFERENCE_DISTRIBUTION_NAME
        )
        statistics.to_csv(stats_path, index=False)
        markers.to_csv(markers_path, index=False)
        reference_pairs = compute_mecr_pair_metrics(reference, markers)
        reference_pairs.to_csv(reference_pairs_path, index=False)
        plot_reference_mecr_histogram(
            reference_pairs,
            reference_distribution_path,
            dpi=config.figure_dpi,
        )

        class_counts = {
            str(label): int(count)
            for label, count in reference.obs["broad_class"].value_counts().items()
        }
        manifest_path = config.output_dir / MECR_REFERENCE_MANIFEST_NAME
        manifest = {
            "method": "paper_standard_mecr",
            "reference": "Allen Whole Human Brain WHB-10Xv3",
            "neurons_h5ad_path": str(config.neurons_h5ad_path),
            "nonneurons_h5ad_path": str(config.nonneurons_h5ad_path),
            "cell_metadata_path": str(config.cell_metadata_path),
            "taxonomy_metadata_path": str(config.taxonomy_metadata_path),
            "cluster_membership_path": str(config.cluster_membership_path),
            "taxonomy_level": config.taxonomy_level,
            "gene_symbol_column": config.gene_symbol_column,
            "target_broad_classes": list(config.target_broad_classes),
            "marker_min_target_fraction_strictly_greater_than": (
                config.marker_min_target_fraction
            ),
            "marker_max_other_fraction_strictly_less_than": (
                config.marker_max_other_fraction
            ),
            "normalization": {
                "method": "full_library_normalize_total_log1p",
                "target_sum": config.normalize_target_sum,
            },
            "wilcoxon": {
                "implementation": "scanpy.tl.rank_genes_groups",
                "tie_correct": config.wilcoxon_tie_correct,
                "cells": "all retained whole-brain reference cells",
            },
            "n_panel_genes": len(panel_genes),
            "n_reference_panel_genes": reference.n_vars,
            "n_reference_cells": reference.n_obs,
            "reference_class_counts": class_counts,
            "n_unique_markers": len(markers),
            "n_reference_marker_pairs": len(reference_pairs),
            "outputs": {
                "panel_genes_csv": str(panel_path),
                "gene_statistics_csv": str(stats_path),
                "markers_csv": str(markers_path),
                "reference_pairs_csv": str(reference_pairs_path),
                "reference_distribution_plot": str(reference_distribution_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        del reference
        force_release(note="after WHB MECR reference preparation")

    return {
        "markers_csv": markers_path,
        "statistics_csv": stats_path,
        "panel_genes_csv": panel_path,
        "reference_pairs_csv": reference_pairs_path,
        "reference_distribution_plot": reference_distribution_path,
        "manifest": manifest_path,
    }


def run_mecr(config: MecrConfig) -> dict[str, Path]:
    """Score paper-standard MECR for every configured spatial sample.

    Args:
        config: Validated spatial MECR configuration.

    Returns:
        Paths to pair metrics, summaries, plots, and the stage manifest.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    markers = pd.read_csv(config.reference_markers_path)
    _validate_marker_table(markers)

    all_pairs: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    output_paths: dict[str, Path] = {}
    for sample in config.samples:
        sample_dir = config.output_dir / sample.platform.lower()
        sample_dir.mkdir(parents=True, exist_ok=True)
        log_status(
            f"[{sample.sample_id}] Computing MECR "
            f"(platform={sample.platform}, segmentation={config.segmentation})"
        )
        spatial_adata = load_spatial_count_table(sample)
        try:
            pair_metrics = compute_mecr_pair_metrics(spatial_adata, markers)
            summary = summarize_mecr_pair_metrics(
                pair_metrics,
                sample_id=sample.sample_id,
                platform=sample.platform,
                segmentation=config.segmentation,
                n_cells=spatial_adata.n_obs,
                n_panel_genes=spatial_adata.n_vars,
                n_reference_markers=len(markers),
            )
        finally:
            del spatial_adata
            force_release(note=f"after MECR scoring {sample.sample_id}")

        pair_metrics.insert(0, "segmentation", config.segmentation)
        pair_metrics.insert(0, "platform", sample.platform)
        pair_metrics.insert(0, "sample_id", sample.sample_id)
        pair_path = sample_dir / f"{sample.sample_id}_mecr_pairs.csv"
        summary_path = sample_dir / f"{sample.sample_id}_mecr_summary.json"
        pair_metrics.to_csv(pair_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        all_pairs.append(pair_metrics)
        summaries.append(summary)
        output_paths[f"{sample.platform.lower()}_pairs_csv"] = pair_path
        output_paths[f"{sample.platform.lower()}_summary_json"] = summary_path

    combined_pairs = pd.concat(all_pairs, ignore_index=True)
    combined_pairs_path = config.output_dir / f"{config.pair_id}_mecr_pairs.csv"
    combined_summary_path = config.output_dir / f"{config.pair_id}_mecr_summary.csv"
    combined_pairs.to_csv(combined_pairs_path, index=False)
    pd.DataFrame(summaries).to_csv(combined_summary_path, index=False)
    distribution_path = plot_mecr_distributions(
        combined_pairs,
        config.output_dir / f"{config.pair_id}_mecr_distribution.png",
        title=f"{config.pair_id} MECR ({config.segmentation})",
        dpi=config.figure_dpi,
    )
    platform_comparison_path = plot_platform_mecr_comparison(
        combined_pairs,
        config.output_dir / f"{config.pair_id}_mecr_platform_comparison.png",
        title=f"{config.pair_id} shared-pair MECR ({config.segmentation})",
        dpi=config.figure_dpi,
    )
    class_heatmap_path = plot_class_pair_mecr_heatmaps(
        combined_pairs,
        config.output_dir / f"{config.pair_id}_mecr_class_pair_heatmap.png",
        title=f"{config.pair_id} broad-class pair MECR ({config.segmentation})",
        dpi=config.figure_dpi,
    )
    barnyard_pairs = select_barnyard_pairs(
        combined_pairs,
        top_n=config.barnyard_top_n_pairs,
    )
    barnyard_selection_path = (
        config.output_dir / f"{config.pair_id}_mecr_barnyard_pairs.csv"
    )
    barnyard_pairs.to_csv(barnyard_selection_path, index=False)
    barnyard_dir = config.output_dir / "plots" / "barnyard"
    barnyard_paths = plot_barnyard_pairs(
        config.samples,
        barnyard_pairs,
        barnyard_dir,
        title_prefix=f"{config.pair_id} ({config.segmentation})",
        dpi=config.figure_dpi,
        max_points=config.barnyard_max_points,
        random_seed=config.barnyard_random_seed,
        log1p_counts=config.barnyard_log1p,
    )
    manifest_path = config.output_dir / f"{config.pair_id}_mecr_manifest.json"
    manifest = {
        "pair_id": config.pair_id,
        "segmentation": config.segmentation,
        "reference_markers_path": str(config.reference_markers_path),
        "formula": "cells_both / cells_either",
        "aggregate": "unweighted arithmetic mean across finite pair MECR values",
        "zero_union_policy": "record NaN and exclude from aggregate mean",
        "samples": summaries,
        "outputs": {
            "combined_pairs_csv": str(combined_pairs_path),
            "combined_summary_csv": str(combined_summary_path),
            "distribution_plot": str(distribution_path),
            "platform_comparison_plot": str(platform_comparison_path),
            "class_pair_heatmap": str(class_heatmap_path),
            "barnyard_selection_csv": str(barnyard_selection_path),
            "barnyard_plots": [str(path) for path in barnyard_paths],
            **{key: str(value) for key, value in output_paths.items()},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "combined_pairs_csv": combined_pairs_path,
        "combined_summary_csv": combined_summary_path,
        "distribution_plot": distribution_path,
        "platform_comparison_plot": platform_comparison_path,
        "class_pair_heatmap": class_heatmap_path,
        "barnyard_selection_csv": barnyard_selection_path,
        "barnyard_plot_dir": barnyard_dir,
        "manifest": manifest_path,
        **output_paths,
    }


def collect_spatial_panel_genes(samples: list[MecrSampleConfig]) -> list[str]:
    """Return the sorted union of gene names in configured SpatialData tables."""
    genes: dict[str, str] = {}
    for sample in samples:
        import spatialdata as sd

        sdata_obj = sd.read_zarr(sample.zarr_path)
        try:
            table_key = _resolve_table_key(sdata_obj, sample.table_key)
            table = sdata_obj.tables[table_key]
            for gene in table.var_names.astype(str):
                token = _normalize_gene(gene)
                if token:
                    genes.setdefault(token, str(gene))
        finally:
            del sdata_obj
    return sorted(genes.values(), key=str.casefold)


def load_whb_panel_reference(
    config: MecrReferenceConfig,
    *,
    panel_genes: list[str],
) -> ad.AnnData:
    """Load a panel-restricted, normalized WHB reference in bounded row chunks."""
    class_by_cell = _load_whb_broad_classes(config)
    panel_lookup = {_normalize_gene(gene): str(gene) for gene in panel_genes}
    matrices: list[sparse.csr_matrix] = []
    labels: list[np.ndarray] = []
    resolved_gene_order: list[str] | None = None
    missing_metadata = 0

    for h5ad_path in (config.neurons_h5ad_path, config.nonneurons_h5ad_path):
        if not h5ad_path.exists():
            raise FileNotFoundError(f"WHB reference H5AD does not exist: {h5ad_path}")
        reference = ad.read_h5ad(h5ad_path, backed="r")
        try:
            if config.gene_symbol_column not in reference.var:
                raise KeyError(f"{h5ad_path} lacks var[{config.gene_symbol_column!r}]")
            symbols = reference.var[config.gene_symbol_column].astype(str).to_numpy()
            selected_indices, selector, gene_order = _reference_panel_selector(
                symbols,
                panel_lookup,
            )
            if resolved_gene_order is None:
                resolved_gene_order = gene_order
            elif resolved_gene_order != gene_order:
                raise ValueError(
                    "WHB neuron/non-neuron references resolved different panels"
                )

            for start in range(0, reference.n_obs, config.reference_chunk_rows):
                stop = min(start + config.reference_chunk_rows, reference.n_obs)
                obs_names = reference.obs_names[start:stop].astype(str)
                chunk_labels = class_by_cell.reindex(obs_names)
                missing_metadata += int(chunk_labels.isna().sum())
                keep = chunk_labels.isin(config.target_broad_classes).to_numpy()
                if not np.any(keep):
                    continue

                raw = sparse.csr_matrix(reference.X[start:stop])
                raw = raw[keep]
                totals = np.asarray(raw.sum(axis=1)).ravel().astype(np.float64)
                panel = raw[:, selected_indices].astype(np.float32)
                panel = sparse.csr_matrix(panel @ selector)
                scale = np.divide(
                    float(config.normalize_target_sum),
                    totals,
                    out=np.zeros_like(totals),
                    where=totals > 0,
                )
                panel = sparse.diags(scale.astype(np.float32)) @ panel
                panel = sparse.csr_matrix(panel)
                panel.data = np.log1p(panel.data)
                panel.eliminate_zeros()
                matrices.append(panel)
                labels.append(chunk_labels.to_numpy(dtype=str)[keep])
                logger.info(
                    "Loaded WHB MECR rows %s:%s from %s (%s retained)",
                    start,
                    stop,
                    h5ad_path.name,
                    int(np.count_nonzero(keep)),
                )
        finally:
            reference.file.close()

    if missing_metadata:
        raise ValueError(
            f"WHB cell metadata was missing for {missing_metadata:,} reference cells"
        )
    if not matrices or resolved_gene_order is None:
        raise ValueError("No WHB cells or panel genes remained for MECR")
    matrix = sparse.vstack(matrices, format="csr", dtype=np.float32)
    obs = pd.DataFrame(
        {"broad_class": pd.Categorical(np.concatenate(labels))},
        index=pd.RangeIndex(matrix.shape[0]).astype(str),
    )
    var = pd.DataFrame(index=pd.Index(resolved_gene_order, name=None))
    return ad.AnnData(X=matrix, obs=obs, var=var)


def discover_reference_markers(
    reference: ad.AnnData,
    *,
    class_key: str,
    target_classes: list[str],
    min_target_fraction: float = 0.25,
    max_other_fraction: float = 0.01,
    tie_correct: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Python Wilcoxon and apply the paper's strict detection filters.

    Marker eligibility is based only on the stated detection fractions. The
    Wilcoxon statistics are retained for auditability and ranking, but no
    undocumented p-value or fold-change threshold is added.
    """
    if class_key not in reference.obs:
        raise KeyError(f"Reference lacks obs[{class_key!r}]")
    classes = reference.obs[class_key].astype(str)
    missing = [label for label in target_classes if label not in set(classes)]
    if missing:
        raise ValueError(f"Reference is missing target broad classes: {missing}")

    sc.tl.rank_genes_groups(
        reference,
        groupby=class_key,
        groups=target_classes,
        reference="rest",
        method="wilcoxon",
        n_genes=reference.n_vars,
        use_raw=False,
        tie_correct=bool(tie_correct),
        pts=True,
        key_added="mecr_wilcoxon",
    )
    binary = sparse.csr_matrix(reference.X)
    binary.data = np.ones(binary.nnz, dtype=np.int64)
    binary.eliminate_zeros()

    rows: list[dict[str, Any]] = []
    for broad_class in target_classes:
        target_mask = classes.to_numpy() == broad_class
        n_target = int(np.count_nonzero(target_mask))
        n_other = int(len(target_mask) - n_target)
        if n_target == 0 or n_other == 0:
            raise ValueError(f"Invalid one-vs-rest group size for {broad_class}")
        target_detected = np.asarray(binary[target_mask].sum(axis=0)).ravel()
        other_detected = np.asarray(binary[~target_mask].sum(axis=0)).ravel()
        wilcoxon = sc.get.rank_genes_groups_df(
            reference,
            group=broad_class,
            key="mecr_wilcoxon",
        ).set_index("names")
        for gene_index, gene in enumerate(reference.var_names.astype(str)):
            target_fraction = float(target_detected[gene_index] / n_target)
            other_fraction = float(other_detected[gene_index] / n_other)
            stat = wilcoxon.loc[gene]
            rows.append(
                {
                    "gene": gene,
                    "broad_class": broad_class,
                    "target_fraction": target_fraction,
                    "other_fraction": other_fraction,
                    "n_target_cells": n_target,
                    "n_other_cells": n_other,
                    "wilcoxon_score": float(stat.get("scores", np.nan)),
                    "logfoldchange": float(stat.get("logfoldchanges", np.nan)),
                    "p_value": float(stat.get("pvals", np.nan)),
                    "p_value_adjusted": float(stat.get("pvals_adj", np.nan)),
                    "passes_detection_thresholds": (
                        target_fraction > float(min_target_fraction)
                        and other_fraction < float(max_other_fraction)
                    ),
                }
            )

    statistics = pd.DataFrame(rows)
    candidates = statistics[statistics["passes_detection_thresholds"]].copy()
    qualifying_class_counts = candidates.groupby("gene")["broad_class"].nunique()
    unique_genes = qualifying_class_counts[qualifying_class_counts == 1].index
    statistics["qualifying_class_count"] = (
        statistics["gene"].map(qualifying_class_counts).fillna(0).astype(int)
    )
    statistics["is_unique_marker"] = (
        statistics["gene"].isin(unique_genes)
        & statistics["passes_detection_thresholds"]
    )
    markers = statistics[statistics["is_unique_marker"]].copy()
    markers = markers.sort_values(
        ["broad_class", "target_fraction", "other_fraction", "gene"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return statistics, markers


def compute_mecr_pair_metrics(
    spatial_adata: ad.AnnData,
    markers: pd.DataFrame,
) -> pd.DataFrame:
    """Compute intersection-over-union MECR for all eligible marker pairs."""
    _validate_marker_table(markers)
    marker_rows = markers.drop_duplicates("gene", keep=False).copy()
    var_lookup: dict[str, int] = {}
    for index, gene in enumerate(spatial_adata.var_names.astype(str)):
        token = _normalize_gene(gene)
        if token in var_lookup:
            raise ValueError(f"Spatial table contains duplicate gene symbol {gene!r}")
        var_lookup[token] = index

    marker_rows["spatial_index"] = marker_rows["gene"].map(
        lambda gene: var_lookup.get(_normalize_gene(gene))
    )
    marker_rows = marker_rows.dropna(subset=["spatial_index"]).copy()
    marker_rows["spatial_index"] = marker_rows["spatial_index"].astype(int)
    marker_rows = marker_rows.sort_values(
        "gene", key=lambda values: values.str.casefold()
    )
    marker_rows = marker_rows.reset_index(drop=True)
    output_columns = [
        "gene_1",
        "class_1",
        "gene_2",
        "class_2",
        "cells_gene_1",
        "cells_gene_2",
        "cells_both",
        "cells_either",
        "mecr",
    ]
    if len(marker_rows) < 2:
        return pd.DataFrame(columns=output_columns)

    counts = sparse.csr_matrix(
        spatial_adata.X[:, marker_rows["spatial_index"].to_numpy()]
    )
    binary = counts.astype(bool).astype(np.int64)
    cooccurrence = np.asarray((binary.T @ binary).toarray(), dtype=np.int64)
    detected = np.diag(cooccurrence)
    classes = marker_rows["broad_class"].astype(str).to_numpy()
    genes = marker_rows["gene"].astype(str).to_numpy()
    first, second = np.triu_indices(len(marker_rows), k=1)
    cross_class = classes[first] != classes[second]
    first = first[cross_class]
    second = second[cross_class]
    both = cooccurrence[first, second]
    either = detected[first] + detected[second] - both
    rates = np.divide(
        both.astype(float),
        either.astype(float),
        out=np.full(len(either), np.nan, dtype=float),
        where=either > 0,
    )
    return pd.DataFrame(
        {
            "gene_1": genes[first],
            "class_1": classes[first],
            "gene_2": genes[second],
            "class_2": classes[second],
            "cells_gene_1": detected[first],
            "cells_gene_2": detected[second],
            "cells_both": both,
            "cells_either": either,
            "mecr": rates,
        },
        columns=output_columns,
    )


def summarize_mecr_pair_metrics(
    pair_metrics: pd.DataFrame,
    *,
    sample_id: str,
    platform: str,
    segmentation: str,
    n_cells: int,
    n_panel_genes: int,
    n_reference_markers: int,
) -> dict[str, Any]:
    """Summarize pair-level MECR values using the paper's unweighted mean."""
    finite = pd.to_numeric(pair_metrics.get("mecr"), errors="coerce")
    finite = finite[np.isfinite(finite)]
    genes = set(pair_metrics.get("gene_1", pd.Series(dtype=str)).astype(str))
    genes.update(pair_metrics.get("gene_2", pd.Series(dtype=str)).astype(str))
    return {
        "sample_id": sample_id,
        "platform": platform,
        "segmentation": segmentation,
        "mecr": float(finite.mean()) if len(finite) else None,
        "median_pair_mecr": float(finite.median()) if len(finite) else None,
        "n_cells": int(n_cells),
        "n_panel_genes": int(n_panel_genes),
        "n_reference_markers": int(n_reference_markers),
        "n_panel_reference_markers": len(genes),
        "n_eligible_pairs": int(len(pair_metrics)),
        "n_scored_pairs": int(len(finite)),
        "n_zero_union_pairs": int(len(pair_metrics) - len(finite)),
    }


def load_spatial_count_table(sample: MecrSampleConfig) -> ad.AnnData:
    """Load the selected raw cell-count table from a SpatialData zarr."""
    import spatialdata as sd

    sdata_obj = sd.read_zarr(sample.zarr_path)
    try:
        table_key = _resolve_table_key(sdata_obj, sample.table_key)
        return sdata_obj.tables[table_key].copy()
    finally:
        del sdata_obj


def plot_mecr_distributions(
    pair_metrics: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str,
    dpi: int = 180,
) -> Path:
    """Plot pair-level MECR distributions for each platform."""
    output_path = prepare_plot_output(output_path)
    finite = pair_metrics[np.isfinite(pd.to_numeric(pair_metrics["mecr"]))].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if finite.empty:
        ax.text(0.5, 0.5, "No finite MECR pairs", ha="center", va="center")
        ax.set_axis_off()
    else:
        sns.boxplot(data=finite, x="mecr", y="platform", ax=ax, color="#8db3d3")
        ax.set_xlabel("Mutually exclusive co-expression rate (MECR)")
        ax.set_ylabel("")
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reference_mecr_histogram(
    pair_metrics: pd.DataFrame,
    output_path: Path | str,
    *,
    dpi: int = 180,
) -> Path:
    """Plot the WHB MECR distribution for eligible cross-class marker pairs."""
    output_path = prepare_plot_output(output_path)
    values = pd.to_numeric(pair_metrics.get("mecr"), errors="coerce")
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    if values.empty:
        ax.text(0.5, 0.5, "No finite reference MECR pairs", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(values, bins=100, color="#7089AC", edgecolor="none")
        mean_value = float(values.mean())
        median_value = float(values.median())
        ax.axvline(
            mean_value,
            color="#B23A48",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean = {mean_value:.4f}",
        )
        ax.axvline(
            median_value,
            color="#2A9D8F",
            linestyle=":",
            linewidth=1.5,
            label=f"Median = {median_value:.4f}",
        )
        ax.legend(frameon=False)
        ax.set_xlabel("WHB mutually exclusive co-expression rate")
        ax.set_ylabel("Gene-pair count")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax.set_title("WHB reference MECR distribution")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_platform_mecr_comparison(
    pair_metrics: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str,
    dpi: int = 180,
) -> Path:
    """Compare MECR for gene pairs shared by MERSCOPE and Xenium."""
    output_path = prepare_plot_output(output_path)
    finite = pair_metrics.copy()
    finite["mecr"] = pd.to_numeric(finite["mecr"], errors="coerce")
    finite = finite[np.isfinite(finite["mecr"])]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if finite.empty:
        shared = pd.DataFrame()
    else:
        pivot = finite.pivot_table(
            index=PAIR_ID_COLUMNS,
            columns="platform",
            values="mecr",
            aggfunc="mean",
        )
        shared = (
            pivot.dropna(subset=["MERSCOPE", "XENIUM"])
            if {"MERSCOPE", "XENIUM"}.issubset(pivot.columns)
            else pd.DataFrame()
        )
    if shared.empty:
        ax.text(
            0.5,
            0.5,
            "No finite gene pairs shared by MERSCOPE and Xenium",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.set_axis_off()
    else:
        x_values = shared["MERSCOPE"].to_numpy(dtype=float)
        y_values = shared["XENIUM"].to_numpy(dtype=float)
        limit = max(float(np.max(x_values)), float(np.max(y_values)), 0.01)
        ax.scatter(
            x_values,
            y_values,
            s=18,
            alpha=0.65,
            color="#5B6F9B",
            edgecolors="none",
        )
        ax.plot([0, limit], [0, limit], color="#555555", linestyle="--", linewidth=1)
        differences = np.abs(y_values - x_values)
        for position in np.argsort(differences)[-min(8, len(shared)) :]:
            gene_1, _class_1, gene_2, _class_2 = shared.index[position]
            ax.annotate(
                f"{gene_1}–{gene_2}",
                (x_values[position], y_values[position]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        correlation = (
            float(np.corrcoef(x_values, y_values)[0, 1]) if len(shared) > 1 else np.nan
        )
        subtitle = f"n={len(shared):,} shared pairs"
        if np.isfinite(correlation):
            subtitle += f"; Pearson r={correlation:.3f}"
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, ha="left", va="top")
        ax.set_xlim(-0.01 * limit, 1.05 * limit)
        ax.set_ylim(-0.01 * limit, 1.05 * limit)
        ax.set_xlabel("MERSCOPE pair MECR")
        ax.set_ylabel("Xenium pair MECR")
        ax.grid(color="#e5e5e5", linewidth=0.5)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_class_pair_mecr_heatmaps(
    pair_metrics: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str,
    dpi: int = 180,
) -> Path:
    """Plot median pair MECR for every broad-class combination by platform."""
    output_path = prepare_plot_output(output_path)
    finite = pair_metrics.copy()
    finite["mecr"] = pd.to_numeric(finite["mecr"], errors="coerce")
    finite = finite[np.isfinite(finite["mecr"])]
    platforms = _ordered_platforms(pair_metrics.get("platform", pd.Series(dtype=str)))
    if not platforms:
        platforms = ["MECR"]
    fig, axes = plt.subplots(
        1,
        len(platforms),
        figsize=(6.0 * len(platforms), 5.4),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    if finite.empty:
        for ax in axes_flat:
            ax.text(
                0.5, 0.5, "No finite class-pair MECR values", ha="center", va="center"
            )
            ax.set_axis_off()
    else:
        finite["class_a"] = np.where(
            finite["class_1"].astype(str) <= finite["class_2"].astype(str),
            finite["class_1"].astype(str),
            finite["class_2"].astype(str),
        )
        finite["class_b"] = np.where(
            finite["class_1"].astype(str) <= finite["class_2"].astype(str),
            finite["class_2"].astype(str),
            finite["class_1"].astype(str),
        )
        classes = sorted(
            set(finite["class_a"].astype(str)) | set(finite["class_b"].astype(str)),
            key=str.casefold,
        )
        vmax = max(float(finite["mecr"].max()), 0.01)
        for index, (platform, ax) in enumerate(zip(platforms, axes_flat, strict=True)):
            subset = finite[finite["platform"].astype(str) == platform]
            matrix = pd.DataFrame(np.nan, index=classes, columns=classes)
            grouped = subset.groupby(["class_a", "class_b"])["mecr"].median()
            for (class_a, class_b), value in grouped.items():
                matrix.loc[str(class_a), str(class_b)] = float(value)
                matrix.loc[str(class_b), str(class_a)] = float(value)
            sns.heatmap(
                matrix,
                mask=matrix.isna(),
                cmap="mako",
                vmin=0.0,
                vmax=vmax,
                annot=True,
                fmt=".3f",
                square=True,
                linewidths=0.5,
                cbar=index == len(platforms) - 1,
                cbar_kws={"label": "Median pair MECR"},
                ax=ax,
            )
            ax.set_title(platform)
            ax.set_xlabel("")
            ax.set_ylabel("")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def select_barnyard_pairs(
    pair_metrics: pd.DataFrame,
    *,
    top_n: int = 6,
) -> pd.DataFrame:
    """Select canonical, high-MECR, and highly detected pairs for barnyard plots."""
    output_columns = [
        *PAIR_ID_COLUMNS,
        "mean_mecr",
        "max_mecr",
        "total_cells_either",
        "selection_reason",
    ]
    if top_n <= 0 or pair_metrics.empty:
        return pd.DataFrame(columns=output_columns)
    finite = pair_metrics.copy()
    finite["mecr"] = pd.to_numeric(finite["mecr"], errors="coerce")
    finite = finite[np.isfinite(finite["mecr"])]
    if finite.empty:
        return pd.DataFrame(columns=output_columns)
    aggregated = (
        finite.groupby(PAIR_ID_COLUMNS, as_index=False)
        .agg(
            mean_mecr=("mecr", "mean"),
            max_mecr=("mecr", "max"),
            total_cells_either=("cells_either", "sum"),
        )
        .sort_values(PAIR_ID_COLUMNS)
        .reset_index(drop=True)
    )
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add_row(row: pd.Series, reason: str) -> None:
        key = (
            str(row["gene_1"]),
            str(row["class_1"]),
            str(row["gene_2"]),
            str(row["class_2"]),
        )
        if key not in selected and len(selected) >= top_n:
            return
        if key not in selected:
            selected[key] = {column: row[column] for column in output_columns[:-1]}
            selected[key]["_reasons"] = []
        reasons = selected[key]["_reasons"]
        if reason not in reasons:
            reasons.append(reason)

    canonical_lookup = {
        frozenset((_normalize_gene(gene_1), _normalize_gene(gene_2)))
        for gene_1, gene_2 in BARNYARD_CANONICAL_PAIRS
    }
    for _, row in aggregated.iterrows():
        pair_key = frozenset(
            (_normalize_gene(row["gene_1"]), _normalize_gene(row["gene_2"]))
        )
        if pair_key in canonical_lookup:
            add_row(row, "canonical")
    high_mecr_count = max(1, (top_n + 1) // 2)
    for _, row in aggregated.nlargest(high_mecr_count, "mean_mecr").iterrows():
        add_row(row, "highest_mean_mecr")
    for _, row in aggregated.nlargest(len(aggregated), "total_cells_either").iterrows():
        add_row(row, "most_detected")
        if len(selected) >= top_n:
            break
    records = []
    for record in selected.values():
        reasons = record.pop("_reasons")
        record["selection_reason"] = ";".join(reasons)
        records.append(record)
    return pd.DataFrame(records, columns=output_columns)


def plot_barnyard_pairs(
    samples: list[MecrSampleConfig],
    selected_pairs: pd.DataFrame,
    output_dir: Path | str,
    *,
    title_prefix: str,
    dpi: int = 180,
    max_points: int = 50_000,
    random_seed: int = 0,
    log1p_counts: bool = False,
) -> list[Path]:
    """Plot cell counts for selected mutually exclusive gene pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if selected_pairs.empty:
        return []
    pair_records = selected_pairs.to_dict(orient="records")
    plot_data: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
        (str(row["gene_1"]), str(row["gene_2"])): {} for row in pair_records
    }
    rng = np.random.default_rng(random_seed)
    for sample in samples:
        spatial_adata = load_spatial_count_table(sample)
        try:
            var_lookup = {
                _normalize_gene(gene): index
                for index, gene in enumerate(spatial_adata.var_names.astype(str))
            }
            for row in pair_records:
                gene_1 = str(row["gene_1"])
                gene_2 = str(row["gene_2"])
                index_1 = var_lookup.get(_normalize_gene(gene_1))
                index_2 = var_lookup.get(_normalize_gene(gene_2))
                if index_1 is None or index_2 is None:
                    continue
                counts_1 = _dense_gene_counts(spatial_adata.X, index_1)
                counts_2 = _dense_gene_counts(spatial_adata.X, index_2)
                detected_1 = counts_1 > 0
                detected_2 = counts_2 > 0
                both = int(np.count_nonzero(detected_1 & detected_2))
                either = int(np.count_nonzero(detected_1 | detected_2))
                indices = np.arange(len(counts_1))
                if len(indices) > max_points:
                    indices = np.sort(
                        rng.choice(indices, size=max_points, replace=False)
                    )
                plot_data[(gene_1, gene_2)][sample.sample_id] = {
                    "platform": sample.platform,
                    "counts_1": counts_1[indices],
                    "counts_2": counts_2[indices],
                    "n_cells": len(counts_1),
                    "cells_both": both,
                    "cells_either": either,
                    "mecr": float(both / either) if either else np.nan,
                }
        finally:
            del spatial_adata
            force_release(note=f"after MECR barnyard extraction {sample.sample_id}")

    output_paths: list[Path] = []
    for row in pair_records:
        gene_1 = str(row["gene_1"])
        gene_2 = str(row["gene_2"])
        panels = plot_data[(gene_1, gene_2)]
        fig, axes = plt.subplots(
            1,
            max(1, len(samples)),
            figsize=(4.6 * max(1, len(samples)), 4.4),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for sample, ax in zip(samples, axes.ravel(), strict=True):
            panel = panels.get(sample.sample_id)
            if panel is None:
                ax.text(0.5, 0.5, "Pair not present", ha="center", va="center")
                ax.set_axis_off()
                continue
            x_values = np.clip(panel["counts_1"], a_min=0, a_max=None)
            y_values = np.clip(panel["counts_2"], a_min=0, a_max=None)
            axis_prefix = ""
            if log1p_counts:
                x_values = np.log1p(x_values)
                y_values = np.log1p(y_values)
                axis_prefix = "log1p "
            ax.scatter(
                x_values,
                y_values,
                s=5,
                alpha=0.22,
                color=PLATFORM_COLORS.get(str(panel["platform"]), "#555555"),
                edgecolors="none",
            )
            mecr_label = f"{panel['mecr']:.4f}" if np.isfinite(panel["mecr"]) else "NaN"
            ax.set_title(
                f"{panel['platform']}\nMECR={mecr_label}; "
                f"both/either={panel['cells_both']:,}/{panel['cells_either']:,}"
            )
            ax.set_xlabel(f"{axis_prefix}{gene_1} counts")
            ax.set_ylabel(f"{axis_prefix}{gene_2} counts")
            ax.grid(color="#eeeeee", linewidth=0.5)
        reason = str(row.get("selection_reason", "selected"))
        fig.suptitle(f"{title_prefix}: {gene_1} vs {gene_2} [{reason}]")
        fig.tight_layout()
        output_path = (
            output_dir / f"{_safe_filename(gene_1)}--{_safe_filename(gene_2)}.png"
        )
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def _ordered_platforms(values: pd.Series) -> list[str]:
    platforms = [str(value) for value in pd.unique(values.astype(str)) if str(value)]
    preferred = [
        platform for platform in ("MERSCOPE", "XENIUM") if platform in platforms
    ]
    return preferred + sorted(set(platforms).difference(preferred), key=str.casefold)


def _dense_gene_counts(matrix: Any, index: int) -> np.ndarray:
    column = matrix[:, index]
    if sparse.issparse(column):
        return np.asarray(column.toarray()).ravel()
    return np.asarray(column).ravel()


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    return cleaned.strip("_") or "gene"


def _load_whb_broad_classes(config: MecrReferenceConfig) -> pd.Series:
    for path in (
        config.cell_metadata_path,
        config.taxonomy_metadata_path,
        config.cluster_membership_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"WHB metadata file does not exist: {path}")
    taxonomy = pd.read_csv(config.taxonomy_metadata_path)
    taxonomy = taxonomy[
        taxonomy["cluster_annotation_term_set_label"].astype(str)
        == config.taxonomy_level
    ]
    label_to_name = dict(
        zip(taxonomy["label"].astype(str), taxonomy["name"].astype(str), strict=False)
    )
    membership = pd.read_csv(
        config.cluster_membership_path,
        usecols=[
            "cluster_annotation_term_label",
            "cluster_annotation_term_set_label",
            "cluster_alias",
        ],
        dtype={"cluster_alias": "string"},
    )
    membership = membership[
        membership["cluster_annotation_term_set_label"].astype(str)
        == config.taxonomy_level
    ].copy()
    membership["broad_class"] = (
        membership["cluster_annotation_term_label"]
        .astype(str)
        .map(label_to_name)
        .map(collapse_atlas_label_to_broad_class)
    )
    alias_to_class = membership.drop_duplicates("cluster_alias").set_index(
        "cluster_alias"
    )["broad_class"]
    cell_metadata = pd.read_csv(
        config.cell_metadata_path,
        usecols=["cell_label", "cluster_alias"],
        dtype={"cell_label": "string", "cluster_alias": "string"},
    )
    cell_metadata["broad_class"] = cell_metadata["cluster_alias"].map(alias_to_class)
    if cell_metadata["broad_class"].isna().any():
        missing = int(cell_metadata["broad_class"].isna().sum())
        raise ValueError(f"Could not map {missing:,} WHB cells to a broad class")
    return cell_metadata.set_index("cell_label")["broad_class"]


def _reference_panel_selector(
    reference_symbols: np.ndarray,
    panel_lookup: dict[str, str],
) -> tuple[np.ndarray, sparse.csr_matrix, list[str]]:
    resolved: dict[str, list[int]] = {}
    for index, symbol in enumerate(reference_symbols):
        token = _normalize_gene(symbol)
        if token in panel_lookup:
            resolved.setdefault(token, []).append(index)
    if not resolved:
        raise ValueError("No spatial panel genes matched WHB gene symbols")
    ordered_tokens = sorted(resolved, key=lambda token: panel_lookup[token].casefold())
    selected_indices: list[int] = []
    selector_rows: list[int] = []
    selector_cols: list[int] = []
    for output_index, token in enumerate(ordered_tokens):
        for reference_index in resolved[token]:
            selector_rows.append(len(selected_indices))
            selector_cols.append(output_index)
            selected_indices.append(reference_index)
    selector = sparse.csr_matrix(
        (
            np.ones(len(selected_indices), dtype=np.float32),
            (selector_rows, selector_cols),
        ),
        shape=(len(selected_indices), len(ordered_tokens)),
    )
    gene_order = [panel_lookup[token] for token in ordered_tokens]
    return np.asarray(selected_indices, dtype=int), selector, gene_order


def _resolve_table_key(sdata_obj: Any, preferred: str | None) -> str:
    if preferred is not None:
        if preferred not in sdata_obj.tables:
            raise KeyError(
                f"SpatialData table {preferred!r} not found; "
                f"available={list(sdata_obj.tables)}"
            )
        return preferred
    if "table" in sdata_obj.tables:
        return "table"
    keys = list(sdata_obj.tables)
    if len(keys) == 1:
        return str(keys[0])
    raise KeyError(f"Could not choose SpatialData table from {keys}")


def _validate_marker_table(markers: pd.DataFrame) -> None:
    required = {"gene", "broad_class"}
    missing = required.difference(markers.columns)
    if missing:
        raise KeyError(f"MECR marker table lacks columns: {sorted(missing)}")
    if markers.empty:
        raise ValueError("MECR reference marker table is empty")


def _normalize_gene(value: object) -> str:
    return str(value).strip().casefold()
