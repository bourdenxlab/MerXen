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
import multiprocessing
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import sparse

from merxen import palette
from merxen.analysis.clustering_squidpy import (
    collapse_atlas_label_to_broad_class,
)
from merxen.config import MecrConfig, MecrReferenceConfig, MecrSampleConfig
from merxen.control_features import is_registered_control_name
from merxen.gene_sets import WHOLE_TRANSCRIPTOME_MIN_GENES, is_whole_transcriptome
from merxen.memory import force_release, log_status
from merxen.platforms import PLATFORM_SPECS
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
PLATFORM_COLORS = {
    "MERSCOPE": "#2878B5",
    "XENIUM": "#D95F02",
    "ATERA": palette.ATERA,
}
# Platforms whose barnyard plots draw every cell as a binned density instead
# of a random scatter subsample (no subsampling in static plots, Atera plan
# AT-8). The XENIUM/MERSCOPE plots keep their scatter subsample.
BARNYARD_DENSITY_PLATFORMS = frozenset({"ATERA"})
REFERENCE_MODE_IN_MEMORY = "in_memory"
REFERENCE_MODE_STREAMING = "streaming"


def run_mecr_reference(config: MecrReferenceConfig) -> dict[str, Path]:
    """Discover mutually exclusive broad-class markers in a whole-brain reference.

    Args:
        config: Validated reference and marker-discovery configuration.

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
    reference_name = _reference_name(config)
    reference_mode, mode_reason = resolve_reference_mode(config, panel_genes)
    log_status(
        f"Preparing {reference_name} MECR reference for "
        f"{len(panel_genes):,} spatial panel genes "
        f"(reference_mode={reference_mode}; {mode_reason})"
    )
    reference: ad.AnnData | None = None
    streamed: StreamingReference | None = None
    try:
        if reference_mode == REFERENCE_MODE_STREAMING:
            streamed = stream_atlas_panel_reference(config, panel_genes=panel_genes)
            statistics, markers = discover_streaming_reference_markers(
                streamed,
                min_target_fraction=config.marker_min_target_fraction,
                max_other_fraction=config.marker_max_other_fraction,
                tie_correct=config.wilcoxon_tie_correct,
            )
            detection = streamed.detection
            pair_reference = streamed.candidates
            class_counts = _class_count_record(detection.classes, detection.class_sizes)
            n_reference_panel_genes = len(detection.genes)
            n_reference_cells = int(detection.class_sizes.sum())
            n_missing_metadata = detection.n_missing_metadata
            n_tested = 0 if pair_reference is None else int(pair_reference.n_vars)
            wilcoxon_genes = {
                "genes": (
                    "panel genes passing the detection thresholds in at least "
                    "one class; NaN statistics for the other genes"
                ),
                "n_tested_genes": n_tested,
                "p_value_adjusted": "Benjamini-Hochberg over the tested genes",
            }
        else:
            reference = load_atlas_panel_reference(config, panel_genes=panel_genes)
            statistics, markers = discover_reference_markers(
                reference,
                class_key="broad_class",
                target_classes=config.target_broad_classes,
                min_target_fraction=config.marker_min_target_fraction,
                max_other_fraction=config.marker_max_other_fraction,
                tie_correct=config.wilcoxon_tie_correct,
            )
            pair_reference = reference
            class_counts = {
                str(label): int(count)
                for label, count in reference.obs["broad_class"].value_counts().items()
            }
            n_reference_panel_genes = int(reference.n_vars)
            n_reference_cells = int(reference.n_obs)
            n_missing_metadata = int(
                reference.uns.get("merxen_mecr_reference", {}).get(
                    "n_raw_cells_without_taxonomy_metadata",
                    0,
                )
            )
            wilcoxon_genes = {
                "genes": "all panel genes",
                "n_tested_genes": n_reference_panel_genes,
                "p_value_adjusted": "Benjamini-Hochberg over the tested genes",
            }
        stats_path = config.output_dir / MECR_REFERENCE_STATS_NAME
        markers_path = config.output_dir / MECR_REFERENCE_MARKERS_NAME
        reference_pairs_path = config.output_dir / MECR_REFERENCE_PAIRS_NAME
        reference_distribution_path = (
            config.output_dir / MECR_REFERENCE_DISTRIBUTION_NAME
        )
        statistics.to_csv(stats_path, index=False)
        markers.to_csv(markers_path, index=False)
        if pair_reference is None:
            # No detection candidate, so no marker (as the in-memory path).
            _validate_marker_table(markers)
            raise ValueError("MECR reference has no detection candidate genes")
        reference_pairs = compute_mecr_pair_metrics(pair_reference, markers)
        reference_pairs.to_csv(reference_pairs_path, index=False)
        plot_reference_mecr_histogram(
            reference_pairs,
            reference_distribution_path,
            dpi=config.figure_dpi,
        )

        manifest_path = config.output_dir / MECR_REFERENCE_MANIFEST_NAME
        manifest = {
            "method": "paper_standard_mecr",
            "reference": (
                "Allen Whole Mouse Brain WMB-10X"
                if config.reference_atlas == "wmb"
                else "Allen Whole Human Brain WHB-10Xv3"
            ),
            "reference_atlas": config.reference_atlas,
            "reference_h5ad_paths": [
                str(path) for path in _mecr_reference_h5ad_paths(config)
            ],
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
            "reference_mode": {
                "requested": config.reference_mode,
                "resolved": reference_mode,
                "reason": mode_reason,
                "n_non_control_panel_genes": count_non_control_genes(panel_genes),
                "whole_transcriptome_min_genes": WHOLE_TRANSCRIPTOME_MIN_GENES,
                "reference_workers": config.reference_workers,
                "reference_chunk_rows": config.reference_chunk_rows,
            },
            "wilcoxon": {
                "implementation": "scanpy.tl.rank_genes_groups",
                "tie_correct": config.wilcoxon_tie_correct,
                "cells": "all retained whole-brain reference cells",
                **wilcoxon_genes,
            },
            "n_panel_genes": len(panel_genes),
            "n_reference_panel_genes": n_reference_panel_genes,
            "n_reference_cells": n_reference_cells,
            "n_raw_cells_without_taxonomy_metadata": int(n_missing_metadata),
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
        del reference, streamed
        force_release(note=f"after {reference_name} MECR reference preparation")

    return {
        "markers_csv": markers_path,
        "statistics_csv": stats_path,
        "panel_genes_csv": panel_path,
        "reference_pairs_csv": reference_pairs_path,
        "reference_distribution_plot": reference_distribution_path,
        "manifest": manifest_path,
    }


def _class_count_record(classes: list[str], sizes: np.ndarray) -> dict[str, int]:
    """Class sizes, largest first, as ``value_counts`` orders them."""
    counts = pd.Series(
        [int(size) for size in sizes], index=list(classes), dtype=np.int64
    )
    counts = counts[counts > 0].sort_values(ascending=False, kind="stable")
    return {str(label): int(count) for label, count in counts.items()}


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
        for gene in _spatial_table_var_names(sample):
            token = _normalize_gene(gene)
            if token:
                genes.setdefault(token, str(gene))
    return sorted(genes.values(), key=str.casefold)


def _spatial_table_group(sample: MecrSampleConfig) -> Path | None:
    """The zarr group of the sample's table, when it can be read on its own.

    Reading one table directly skips parsing every element of the store (on a
    whole-transcriptome store, minutes of labels, shapes and other tables).
    ``None`` defers to SpatialData, which also reports a missing table.
    """
    tables = Path(sample.zarr_path) / "tables"
    if not tables.is_dir():
        return None
    if sample.table_key is not None:
        candidate = tables / sample.table_key
    elif (tables / "table").is_dir():
        candidate = tables / "table"
    else:
        keys = [path for path in tables.iterdir() if path.is_dir()]
        if len(keys) != 1:
            return None
        candidate = keys[0]
    is_group = (candidate / "zarr.json").is_file() or (candidate / ".zgroup").is_file()
    return candidate if is_group else None


def _spatial_table_var_names(sample: MecrSampleConfig) -> list[str]:
    group_path = _spatial_table_group(sample)
    if group_path is not None:
        import zarr
        from anndata.io import read_elem

        var = read_elem(zarr.open_group(str(group_path), mode="r")["var"])
        return [str(gene) for gene in var.index]
    import spatialdata as sd

    sdata_obj = sd.read_zarr(sample.zarr_path)
    try:
        table_key = _resolve_table_key(sdata_obj, sample.table_key)
        return [str(gene) for gene in sdata_obj.tables[table_key].var_names]
    finally:
        del sdata_obj


def _mecr_reference_h5ad_paths(config: MecrReferenceConfig) -> list[Path]:
    if config.reference_h5ad_paths:
        return list(config.reference_h5ad_paths)
    return [
        path
        for path in (config.neurons_h5ad_path, config.nonneurons_h5ad_path)
        if path is not None
    ]


def _resolve_mecr_reference_inputs(config: MecrReferenceConfig) -> list[Path]:
    h5ad_paths = _mecr_reference_h5ad_paths(config)
    metadata_paths = (
        config.cell_metadata_path,
        config.taxonomy_metadata_path,
        config.cluster_membership_path,
    )
    if (
        config.reference_atlas == "wmb"
        and config.auto_download_reference
        and (not h5ad_paths or any(path is None for path in metadata_paths))
    ):
        from merxen.analysis.mapmycells import ensure_wmb_mecr_reference_inputs

        downloaded = ensure_wmb_mecr_reference_inputs(config.reference_cache_dir)
        if not h5ad_paths:
            config.reference_h5ad_paths = sorted(
                path
                for key, path in downloaded.items()
                if key.startswith("expression:")
            )
            h5ad_paths = list(config.reference_h5ad_paths)
        config.cell_metadata_path = (
            config.cell_metadata_path or downloaded["cell_metadata"]
        )
        config.taxonomy_metadata_path = (
            config.taxonomy_metadata_path or downloaded["cluster_annotation_term"]
        )
        config.cluster_membership_path = (
            config.cluster_membership_path
            or downloaded["cluster_to_cluster_annotation_membership"]
        )

    if not h5ad_paths:
        raise ValueError(
            "MECR requires reference_h5ad_paths (or the legacy WHB neuron and "
            "non-neuron paths). Enable auto_download_reference for WMB."
        )
    missing_h5ads = [path for path in h5ad_paths if not path.is_file()]
    if missing_h5ads:
        raise FileNotFoundError(
            "Missing MECR reference H5AD files: "
            + ", ".join(str(path) for path in missing_h5ads)
        )
    for label, path in (
        ("cell metadata", config.cell_metadata_path),
        ("taxonomy metadata", config.taxonomy_metadata_path),
        ("cluster membership metadata", config.cluster_membership_path),
    ):
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing MECR {label}: {path}")
    return h5ad_paths


@dataclass(frozen=True)
class _ReferenceShard:
    """One reference expression H5AD, with its symbols and per-cell classes."""

    path: Path
    n_obs: int
    symbols: np.ndarray
    # Index into ``target_broad_classes`` per cell; -1 = not retained.
    class_codes: np.ndarray


@dataclass(frozen=True)
class _ChunkTask:
    """A row chunk of one reference shard, as sent to a decode worker."""

    path: str
    start: int
    stop: int
    class_codes: np.ndarray
    selected_indices: np.ndarray
    selector: sparse.csr_matrix
    normalize_target_sum: float
    n_classes: int
    # "count": per-class detection counts; "collect": the normalised chunk.
    mode: str


@dataclass(frozen=True)
class ReferenceDetection:
    """Per-class detection counts of every panel gene (streaming pass 1).

    Attributes:
        genes: Panel genes resolved in the reference, in output order.
        classes: Target broad classes, in configuration order.
        detected: ``(n_classes, n_genes)`` count of cells of each class in
            which each gene is detected.
        class_sizes: Retained cells per class.
        n_missing_metadata: Raw cells without taxonomy metadata.
        rest_detected: Detections in retained cells outside every target
            class (zeros for an atlas reference, which keeps target classes
            only).
        n_rest_cells: Retained cells outside every target class.
    """

    genes: list[str]
    classes: list[str]
    detected: np.ndarray
    class_sizes: np.ndarray
    n_missing_metadata: int
    rest_detected: np.ndarray | None = None
    n_rest_cells: int = 0


@dataclass(frozen=True)
class StreamingReference:
    """What the streaming reference keeps instead of the full panel matrix.

    Attributes:
        detection: Pass-1 detection counts over every panel gene.
        candidates: Pass-2 reference restricted to the genes that pass the
            detection thresholds in at least one class (``None`` when no gene
            does). Its ``X`` is CSC, for the per-gene Wilcoxon ranking.
    """

    detection: ReferenceDetection
    candidates: ad.AnnData | None


def count_non_control_genes(genes: list[str]) -> int:
    """Count panel genes that are not control features of any platform."""
    return sum(1 for gene in genes if not is_registered_control_name(str(gene)))


def resolve_reference_mode(
    config: MecrReferenceConfig,
    panel_genes: list[str],
) -> tuple[str, str]:
    """Return the reference mode to run and why.

    ``auto`` streams when the panel has more than
    :data:`~merxen.gene_sets.WHOLE_TRANSCRIPTOME_MIN_GENES` non-control genes
    (Atera plan AT-8, Q22), whatever the platform.
    """
    if config.reference_mode != "auto":
        return config.reference_mode, "requested"
    n_genes = count_non_control_genes(panel_genes)
    if is_whole_transcriptome(n_genes):
        return (
            REFERENCE_MODE_STREAMING,
            f"auto: {n_genes:,} non-control panel genes > "
            f"{WHOLE_TRANSCRIPTOME_MIN_GENES:,}",
        )
    return (
        REFERENCE_MODE_IN_MEMORY,
        f"auto: {n_genes:,} non-control panel genes <= "
        f"{WHOLE_TRANSCRIPTOME_MIN_GENES:,}",
    )


def load_atlas_panel_reference(
    config: MecrReferenceConfig,
    *,
    panel_genes: list[str],
) -> ad.AnnData:
    """Load a panel-restricted whole-brain reference in bounded row chunks.

    The whole panel-restricted matrix of every retained cell is kept; whole-
    transcriptome panels use :func:`stream_atlas_panel_reference` instead.
    """
    shards, missing_metadata = _open_reference_shards(config)
    return _collect_panel_reference(
        config,
        shards,
        panel_genes=panel_genes,
        missing_metadata=missing_metadata,
    )


def stream_atlas_panel_reference(
    config: MecrReferenceConfig,
    *,
    panel_genes: list[str],
) -> StreamingReference:
    """Read the reference twice without keeping the panel matrix.

    Pass 1 counts, per broad class, the cells in which each panel gene is
    detected. Pass 2 keeps only the genes that pass the marker detection
    thresholds in at least one class (normalised as in
    :func:`load_atlas_panel_reference`), for the Wilcoxon test and the
    reference marker co-occurrence.
    """
    reference_name = _reference_name(config)
    shards, missing_metadata = _open_reference_shards(config)
    detection = _count_reference_detection(
        config,
        shards,
        panel_genes=panel_genes,
        missing_metadata=missing_metadata,
    )
    candidate_genes = detection_candidate_genes(
        detection,
        min_target_fraction=config.marker_min_target_fraction,
        max_other_fraction=config.marker_max_other_fraction,
    )
    log_status(
        f"{reference_name} MECR streaming pass 1: {len(detection.genes):,} panel "
        f"genes, {int(detection.class_sizes.sum()):,} cells, "
        f"{len(candidate_genes):,} detection candidates"
    )
    if not candidate_genes:
        return StreamingReference(detection=detection, candidates=None)
    candidates = _collect_panel_reference(
        config,
        shards,
        panel_genes=candidate_genes,
        missing_metadata=missing_metadata,
    )
    if list(candidates.var_names.astype(str)) != candidate_genes:
        raise ValueError(
            f"{reference_name} streaming pass 2 resolved a different gene order"
        )
    candidates.X = sparse.csc_matrix(candidates.X)
    return StreamingReference(detection=detection, candidates=candidates)


def detection_candidate_genes(
    detection: ReferenceDetection,
    *,
    min_target_fraction: float,
    max_other_fraction: float,
) -> list[str]:
    """Genes that pass the marker detection thresholds in at least one class."""
    target_fraction, other_fraction = _detection_fractions(detection)
    passes = (target_fraction > float(min_target_fraction)) & (
        other_fraction < float(max_other_fraction)
    )
    keep = np.any(passes, axis=0)
    return [gene for gene, kept in zip(detection.genes, keep, strict=True) if kept]


def load_whb_panel_reference(
    config: MecrReferenceConfig,
    *,
    panel_genes: list[str],
) -> ad.AnnData:
    """Backward-compatible alias for loading a configured MECR reference."""
    return load_atlas_panel_reference(config, panel_genes=panel_genes)


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
    classes = reference.obs[class_key].astype(str).to_numpy()
    missing = [label for label in target_classes if label not in set(classes)]
    if missing:
        raise ValueError(f"Reference is missing target broad classes: {missing}")

    wilcoxon = _wilcoxon_by_class(
        reference,
        class_key=class_key,
        target_classes=target_classes,
        tie_correct=tie_correct,
    )
    binary = sparse.csr_matrix(reference.X)
    binary.data = np.ones(binary.nnz, dtype=np.int64)
    binary.eliminate_zeros()
    # Cells outside every target class count as "other" for every class.
    n_classes = len(target_classes)
    codes = np.full(len(classes), n_classes, dtype=np.int64)
    for index, label in enumerate(target_classes):
        codes[classes == label] = index
    detected, class_sizes = _class_detection_counts(
        binary, codes, n_classes=n_classes + 1
    )
    detection = ReferenceDetection(
        genes=list(reference.var_names.astype(str)),
        classes=list(target_classes),
        detected=detected[:n_classes],
        class_sizes=class_sizes[:n_classes],
        n_missing_metadata=0,
        rest_detected=detected[n_classes],
        n_rest_cells=int(class_sizes[n_classes]),
    )
    return reference_marker_tables(
        detection,
        wilcoxon,
        min_target_fraction=min_target_fraction,
        max_other_fraction=max_other_fraction,
    )


def discover_streaming_reference_markers(
    streamed: StreamingReference,
    *,
    min_target_fraction: float = 0.25,
    max_other_fraction: float = 0.01,
    tie_correct: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marker tables of a streaming reference.

    Detection fractions cover every panel gene. The Wilcoxon test runs on the
    detection candidates only, so its columns are NaN for the other genes and
    ``p_value_adjusted`` is Benjamini-Hochberg over the candidates. Marker
    eligibility uses detection only, so the markers equal the in-memory ones.
    """
    detection = streamed.detection
    missing = [
        label
        for label, size in zip(detection.classes, detection.class_sizes, strict=True)
        if int(size) == 0
    ]
    if missing:
        raise ValueError(f"Reference is missing target broad classes: {missing}")
    wilcoxon: dict[str, pd.DataFrame] = {}
    if streamed.candidates is not None:
        wilcoxon = _wilcoxon_by_class(
            streamed.candidates,
            class_key="broad_class",
            target_classes=detection.classes,
            tie_correct=tie_correct,
        )
    return reference_marker_tables(
        detection,
        wilcoxon,
        min_target_fraction=min_target_fraction,
        max_other_fraction=max_other_fraction,
    )


def reference_marker_tables(
    detection: ReferenceDetection,
    wilcoxon: dict[str, pd.DataFrame],
    *,
    min_target_fraction: float,
    max_other_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the gene statistics and unique-marker tables from detection counts.

    Args:
        detection: Per-class detection counts of every panel gene.
        wilcoxon: Per class, the scanpy one-vs-rest table indexed by gene;
            genes it lacks get NaN Wilcoxon columns.
        min_target_fraction: Strict lower bound on the target-class fraction.
        max_other_fraction: Strict upper bound on the other-class fraction.

    Returns:
        ``(statistics, markers)`` with one statistics row per class and gene.
    """
    genes = pd.Index(detection.genes, dtype=object)
    target_fraction, other_fraction = _detection_fractions(detection)
    total_cells = int(detection.class_sizes.sum()) + int(detection.n_rest_cells)
    frames: list[pd.DataFrame] = []
    for class_index, broad_class in enumerate(detection.classes):
        n_target = int(detection.class_sizes[class_index])
        n_other = total_cells - n_target
        if n_target == 0 or n_other == 0:
            raise ValueError(f"Invalid one-vs-rest group size for {broad_class}")
        table = wilcoxon.get(broad_class)
        stats = (
            table.reindex(genes)
            if table is not None
            else pd.DataFrame(index=genes, dtype=float)
        )

        def column(name: str, stats: pd.DataFrame = stats) -> np.ndarray:
            if name not in stats:
                return np.full(len(genes), np.nan, dtype=np.float64)
            return np.asarray(
                stats[name].to_numpy(dtype=np.float64, na_value=np.nan),
                dtype=np.float64,
            )

        target = target_fraction[class_index]
        other = other_fraction[class_index]
        frames.append(
            pd.DataFrame(
                {
                    "gene": genes.to_numpy(),
                    "broad_class": broad_class,
                    "target_fraction": target,
                    "other_fraction": other,
                    "n_target_cells": n_target,
                    "n_other_cells": n_other,
                    "wilcoxon_score": column("scores"),
                    "logfoldchange": column("logfoldchanges"),
                    "p_value": column("pvals"),
                    "p_value_adjusted": column("pvals_adj"),
                    "passes_detection_thresholds": (
                        (target > float(min_target_fraction))
                        & (other < float(max_other_fraction))
                    ),
                }
            )
        )

    statistics = pd.concat(frames, ignore_index=True)
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


def _detection_fractions(
    detection: ReferenceDetection,
) -> tuple[np.ndarray, np.ndarray]:
    detected = np.asarray(detection.detected, dtype=np.int64)
    sizes = np.asarray(detection.class_sizes, dtype=np.int64)
    total_detected = detected.sum(axis=0)
    if detection.rest_detected is not None:
        total_detected = total_detected + np.asarray(
            detection.rest_detected, dtype=np.int64
        )
    other_detected = total_detected[None, :] - detected
    other_sizes = int(sizes.sum()) + int(detection.n_rest_cells) - sizes
    with np.errstate(divide="ignore", invalid="ignore"):
        target_fraction = detected / sizes[:, None]
        other_fraction = other_detected / other_sizes[:, None]
    return target_fraction, other_fraction


def _wilcoxon_by_class(
    reference: ad.AnnData,
    *,
    class_key: str,
    target_classes: list[str],
    tie_correct: bool,
) -> dict[str, pd.DataFrame]:
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
    return {
        broad_class: sc.get.rank_genes_groups_df(
            reference,
            group=broad_class,
            key="mecr_wilcoxon",
        ).set_index("names")
        for broad_class in target_classes
    }


def _class_detection_counts(
    binary: sparse.csr_matrix,
    codes: np.ndarray,
    *,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class detected-cell counts of a binary cell x gene matrix."""
    keep = codes >= 0
    rows = np.flatnonzero(keep)
    indicator = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (codes[keep], rows)),
        shape=(n_classes, binary.shape[0]),
    )
    detected = np.asarray((indicator @ binary).toarray(), dtype=np.int64)
    class_sizes = np.bincount(codes[keep], minlength=n_classes).astype(np.int64)
    return detected, class_sizes


def _reference_name(config: MecrReferenceConfig) -> str:
    return "WMB" if config.reference_atlas == "wmb" else "WHB"


def _open_reference_shards(
    config: MecrReferenceConfig,
) -> tuple[list[_ReferenceShard], int]:
    """Resolve the reference shards, their symbols and per-cell class codes."""
    reference_name = _reference_name(config)
    h5ad_paths = _resolve_mecr_reference_inputs(config)
    class_by_cell = _load_atlas_broad_classes(config)
    class_index = {
        label: index for index, label in enumerate(config.target_broad_classes)
    }
    shards: list[_ReferenceShard] = []
    missing_metadata = 0
    for h5ad_path in h5ad_paths:
        if not h5ad_path.exists():
            raise FileNotFoundError(
                f"{reference_name} reference H5AD does not exist: {h5ad_path}"
            )
        reference = ad.read_h5ad(h5ad_path, backed="r")
        try:
            if config.gene_symbol_column not in reference.var:
                raise KeyError(f"{h5ad_path} lacks var[{config.gene_symbol_column!r}]")
            symbols = reference.var[config.gene_symbol_column].astype(str).to_numpy()
            labels = class_by_cell.reindex(reference.obs_names.astype(str))
            missing_metadata += int(labels.isna().sum())
            codes = (
                labels.map(class_index).fillna(-1).to_numpy(dtype=np.int16)
                if len(labels)
                else np.zeros(0, dtype=np.int16)
            )
            shards.append(
                _ReferenceShard(
                    path=Path(h5ad_path),
                    n_obs=int(reference.n_obs),
                    symbols=symbols,
                    class_codes=codes,
                )
            )
        finally:
            reference.file.close()
    if missing_metadata and config.reference_atlas == "whb":
        raise ValueError(
            f"{reference_name} cell metadata was missing for "
            f"{missing_metadata:,} reference cells"
        )
    if missing_metadata:
        logger.warning(
            "Skipped %d WMB raw reference cells absent from taxonomy metadata",
            missing_metadata,
        )
    return shards, missing_metadata


def _chunk_tasks(
    config: MecrReferenceConfig,
    shards: list[_ReferenceShard],
    *,
    panel_genes: list[str],
    mode: str,
) -> tuple[list[_ChunkTask], list[str]]:
    """Row-chunk tasks over every shard, and the resolved panel gene order."""
    reference_name = _reference_name(config)
    panel_lookup = {_normalize_gene(gene): str(gene) for gene in panel_genes}
    resolved_gene_order: list[str] | None = None
    tasks: list[_ChunkTask] = []
    for shard in shards:
        selected_indices, selector, gene_order = _reference_panel_selector(
            shard.symbols,
            panel_lookup,
        )
        if resolved_gene_order is None:
            resolved_gene_order = gene_order
        elif resolved_gene_order != gene_order:
            raise ValueError(
                f"{reference_name} reference shards resolved different panels"
            )
        for start in range(0, shard.n_obs, config.reference_chunk_rows):
            stop = min(start + config.reference_chunk_rows, shard.n_obs)
            codes = shard.class_codes[start:stop]
            if not np.any(codes >= 0):
                continue
            tasks.append(
                _ChunkTask(
                    path=str(shard.path),
                    start=start,
                    stop=stop,
                    class_codes=codes,
                    selected_indices=selected_indices,
                    selector=selector,
                    normalize_target_sum=float(config.normalize_target_sum),
                    n_classes=len(config.target_broad_classes),
                    mode=mode,
                )
            )
    if resolved_gene_order is None:
        raise ValueError(f"No {reference_name} cells or panel genes remained for MECR")
    return tasks, resolved_gene_order


def _count_reference_detection(
    config: MecrReferenceConfig,
    shards: list[_ReferenceShard],
    *,
    panel_genes: list[str],
    missing_metadata: int,
) -> ReferenceDetection:
    tasks, gene_order = _chunk_tasks(
        config, shards, panel_genes=panel_genes, mode="count"
    )
    n_classes = len(config.target_broad_classes)
    detected = np.zeros((n_classes, len(gene_order)), dtype=np.int64)
    class_sizes = np.zeros(n_classes, dtype=np.int64)
    for task, (chunk_detected, chunk_sizes) in zip(
        tasks,
        _map_reference_chunks(tasks, workers=config.reference_workers),
        strict=True,
    ):
        detected += chunk_detected
        class_sizes += chunk_sizes
        logger.info(
            "Counted %s MECR rows %s:%s from %s (%s retained)",
            _reference_name(config),
            task.start,
            task.stop,
            Path(task.path).name,
            int(chunk_sizes.sum()),
        )
    if not class_sizes.sum():
        raise ValueError(
            f"No {_reference_name(config)} cells or panel genes remained for MECR"
        )
    return ReferenceDetection(
        genes=gene_order,
        classes=list(config.target_broad_classes),
        detected=detected,
        class_sizes=class_sizes,
        n_missing_metadata=int(missing_metadata),
    )


def _collect_panel_reference(
    config: MecrReferenceConfig,
    shards: list[_ReferenceShard],
    *,
    panel_genes: list[str],
    missing_metadata: int,
) -> ad.AnnData:
    reference_name = _reference_name(config)
    tasks, gene_order = _chunk_tasks(
        config, shards, panel_genes=panel_genes, mode="collect"
    )
    target_classes = np.asarray(config.target_broad_classes, dtype=object)
    matrices: list[sparse.csr_matrix] = []
    labels: list[np.ndarray] = []
    for task, (panel, codes) in zip(
        tasks,
        _map_reference_chunks(tasks, workers=config.reference_workers),
        strict=True,
    ):
        matrices.append(panel)
        labels.append(target_classes[codes].astype(str))
        logger.info(
            "Loaded %s MECR rows %s:%s from %s (%s retained)",
            reference_name,
            task.start,
            task.stop,
            Path(task.path).name,
            len(codes),
        )
    if not matrices:
        raise ValueError(f"No {reference_name} cells or panel genes remained for MECR")
    matrix = sparse.vstack(matrices, format="csr", dtype=np.float32)
    obs = pd.DataFrame(
        {"broad_class": pd.Categorical(np.concatenate(labels))},
        index=pd.RangeIndex(matrix.shape[0]).astype(str),
    )
    var = pd.DataFrame(index=pd.Index(gene_order, name=None))
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.uns["merxen_mecr_reference"] = {
        "reference_atlas": config.reference_atlas,
        "n_raw_cells_without_taxonomy_metadata": int(missing_metadata),
    }
    return result


def _map_reference_chunks(
    tasks: list[_ChunkTask],
    *,
    workers: int,
) -> Iterator[Any]:
    """Decode chunk tasks in order, in-process or in ``workers`` processes."""
    if workers <= 1 or len(tasks) <= 1:
        reader = _ReferenceRowReader()
        try:
            for task in tasks:
                yield _process_reference_chunk(task, reader)
        finally:
            reader.close()
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(workers), len(tasks)),
        mp_context=context,
    ) as executor:
        yield from executor.map(_process_reference_chunk_in_worker, tasks)


class _ReferenceRowReader:
    """Read CSR row ranges of reference H5ADs, keeping each file open."""

    def __init__(self) -> None:
        self._handles: dict[str, Any] = {}

    def rows(self, path: str, start: int, stop: int) -> sparse.csr_matrix:
        import h5py
        from anndata.io import sparse_dataset

        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        matrix = handle["X"]
        if isinstance(matrix, h5py.Group):
            return sparse.csr_matrix(sparse_dataset(matrix)[start:stop])
        return sparse.csr_matrix(matrix[start:stop])

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


_WORKER_READER: _ReferenceRowReader | None = None


def _process_reference_chunk_in_worker(task: _ChunkTask) -> Any:
    global _WORKER_READER
    if _WORKER_READER is None:
        _WORKER_READER = _ReferenceRowReader()
    return _process_reference_chunk(task, _WORKER_READER)


def _process_reference_chunk(task: _ChunkTask, reader: _ReferenceRowReader) -> Any:
    """Normalise one row chunk; count detections or return the panel chunk."""
    keep = task.class_codes >= 0
    raw = reader.rows(task.path, task.start, task.stop)
    raw = raw[keep]
    codes = task.class_codes[keep].astype(np.int64)
    totals = np.asarray(raw.sum(axis=1)).ravel().astype(np.float64)
    panel = raw[:, task.selected_indices].astype(np.float32)
    panel = sparse.csr_matrix(panel @ task.selector)
    scale = np.divide(
        float(task.normalize_target_sum),
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )
    panel = sparse.diags(scale.astype(np.float32)) @ panel
    panel = sparse.csr_matrix(panel)
    panel.data = np.log1p(panel.data)
    panel.eliminate_zeros()
    if task.mode == "collect":
        return panel, codes
    binary = panel
    binary.data = np.ones(binary.nnz, dtype=np.int64)
    return _class_detection_counts(binary, codes, n_classes=task.n_classes)


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
    group_path = _spatial_table_group(sample)
    if group_path is not None:
        return ad.read_zarr(group_path)
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
    """Plot the reference MECR distribution for eligible marker pairs."""
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
        ax.set_xlabel("Reference mutually exclusive co-expression rate")
        ax.set_ylabel("Gene-pair count")
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax.set_title("Whole-brain reference MECR distribution")
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
    """Plot cell counts for selected mutually exclusive gene pairs.

    XENIUM and MERSCOPE panels scatter at most ``max_points`` randomly chosen
    cells. Platforms in :data:`BARNYARD_DENSITY_PLATFORMS` (ATERA) draw every
    cell as a binned density with a logarithmic colour scale instead.
    """
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
                density = sample.platform in BARNYARD_DENSITY_PLATFORMS
                if not density and len(indices) > max_points:
                    indices = np.sort(
                        rng.choice(indices, size=max_points, replace=False)
                    )
                plot_data[(gene_1, gene_2)][sample.sample_id] = {
                    "platform": sample.platform,
                    "density": density,
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
            color = PLATFORM_COLORS.get(str(panel["platform"]), "#555555")
            if panel["density"]:
                _draw_barnyard_density(
                    fig,
                    ax,
                    x_values,
                    y_values,
                    color=color,
                    integer_counts=not log1p_counts,
                )
            else:
                ax.scatter(
                    x_values,
                    y_values,
                    s=5,
                    alpha=0.22,
                    color=color,
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


def _draw_barnyard_density(
    fig: Any,
    ax: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    color: str,
    integer_counts: bool,
    max_bins: int = 120,
) -> None:
    """Draw every cell as a 2D histogram of the two genes' counts.

    Integer counts get one bin per count up to ``max_bins`` bins per axis
    (wider bins beyond); log1p values get ``max_bins`` equal bins. Bins hold
    cell counts on a log colour scale; empty bins stay blank.
    """
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    upper = max(
        float(np.max(x_values, initial=0.0)), float(np.max(y_values, initial=0.0))
    )
    edges: np.ndarray
    if integer_counts and upper + 1 <= max_bins:
        edges = np.arange(-0.5, upper + 1.5, 1.0)
    elif integer_counts:
        edges = np.linspace(-0.5, upper + 0.5, max_bins + 1)
    else:
        edges = np.linspace(0.0, max(upper, 1e-6), max_bins + 1)
    histogram, x_edges, y_edges = np.histogram2d(
        x_values, y_values, bins=[edges, edges]
    )
    masked = np.ma.masked_equal(histogram.T, 0)
    cmap = LinearSegmentedColormap.from_list(
        "merxen_barnyard_density", ["#F2F2F2", color, "#1A1A1A"]
    )
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        masked,
        cmap=cmap,
        norm=LogNorm(vmin=1.0, vmax=max(float(histogram.max()), 2.0)),
        rasterized=True,
    )
    fig.colorbar(mesh, ax=ax, label="cells", fraction=0.046, pad=0.04)
    ax.text(
        0.98,
        0.98,
        f"all {len(x_values):,} cells",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
    )


def _ordered_platforms(values: pd.Series) -> list[str]:
    platforms = [str(value) for value in pd.unique(values.astype(str)) if str(value)]
    # Registry order (MERSCOPE, XENIUM, ATERA), then any other label.
    preferred = [platform for platform in PLATFORM_SPECS if platform in platforms]
    return preferred + sorted(set(platforms).difference(preferred), key=str.casefold)


def _dense_gene_counts(matrix: Any, index: int) -> np.ndarray:
    column = matrix[:, index]
    if sparse.issparse(column):
        return np.asarray(column.toarray()).ravel()
    return np.asarray(column).ravel()


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    return cleaned.strip("_") or "gene"


def _load_atlas_broad_classes(config: MecrReferenceConfig) -> pd.Series:
    metadata_paths = (
        config.cell_metadata_path,
        config.taxonomy_metadata_path,
        config.cluster_membership_path,
    )
    if any(path is None for path in metadata_paths):
        raise ValueError("MECR reference metadata paths have not been resolved")
    cell_metadata_path, taxonomy_metadata_path, cluster_membership_path = (
        path for path in metadata_paths if path is not None
    )
    taxonomy_level = config.taxonomy_level
    if taxonomy_level is None:
        raise ValueError("MECR taxonomy_level has not been resolved")
    taxonomy = pd.read_csv(taxonomy_metadata_path)
    taxonomy = taxonomy[
        taxonomy["cluster_annotation_term_set_label"].astype(str) == taxonomy_level
    ]
    label_to_name = dict(
        zip(taxonomy["label"].astype(str), taxonomy["name"].astype(str), strict=False)
    )
    membership = pd.read_csv(
        cluster_membership_path,
        usecols=[
            "cluster_annotation_term_label",
            "cluster_annotation_term_set_label",
            "cluster_alias",
        ],
        dtype={"cluster_alias": "string"},
    )
    membership = membership[
        membership["cluster_annotation_term_set_label"].astype(str) == taxonomy_level
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
        cell_metadata_path,
        usecols=["cell_label", "cluster_alias"],
        dtype={"cell_label": "string", "cluster_alias": "string"},
    )
    cell_metadata["broad_class"] = cell_metadata["cluster_alias"].map(alias_to_class)
    if cell_metadata["broad_class"].isna().any():
        missing = int(cell_metadata["broad_class"].isna().sum())
        atlas_name = config.reference_atlas.upper()
        raise ValueError(
            f"Could not map {missing:,} {atlas_name} cells to a broad class"
        )
    return cell_metadata.set_index("cell_label")["broad_class"]


def _load_whb_broad_classes(config: MecrReferenceConfig) -> pd.Series:
    """Backward-compatible alias for configured atlas class loading."""
    return _load_atlas_broad_classes(config)


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
        raise ValueError("No spatial panel genes matched reference gene symbols")
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
