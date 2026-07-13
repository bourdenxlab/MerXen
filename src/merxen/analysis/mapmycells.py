"""Local MapMyCells annotation for clustered MerXen AnnData outputs."""

from __future__ import annotations

import fcntl
import json
import logging
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import anndata as ad
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from scipy import sparse

from merxen.config import MapMyCellsConfig
from merxen.memory import force_release, log_status
from merxen.plotting import prepare_plot_output, save_figure

logger = logging.getLogger(__name__)

MAPMYCELLS_PREFIX = "mapmycells_"
MAPMYCELLS_WMB_PREFIX = "mapmycells_wmb_"
MAPMYCELLS_REGION_PREFIX_TEMPLATE = "mapmycells_region_{region_name}_"
MAPMYCELLS_WMB_REGION_PREFIX_TEMPLATE = "mapmycells_wmb_region_{region_name}_"
MAPMYCELLS_UNS_KEY = "merxen_mapmycells"
MAPMYCELLS_WMB_UNS_KEY = "merxen_mapmycells_wmb"
MAPMYCELLS_REGION_UNS_KEY_TEMPLATE = "merxen_mapmycells_region_{region_name}"
MAPMYCELLS_WMB_REGION_UNS_KEY_TEMPLATE = "merxen_mapmycells_wmb_region_{region_name}"
MAPMYCELLS_ASSIGNMENT_COLOR_SUFFIX_CANDIDATES = (
    "subcluster_name",
    "cluster_name",
    "supercluster_name",
    "class_name",
    "subclass_name",
    "type_name",
    "cell_type",
)
MAPMYCELLS_MAX_LEGEND_CATEGORIES = 64
MAPMYCELLS_QC_LEVELS = ("supercluster", "cluster")
MAPMYCELLS_QC_MAX_ASSIGNMENT_CATEGORIES = 40
MAPMYCELLS_LEVEL_FIELD_SUFFIXES = (
    "bootstrapping_probability",
    "correlation_coefficient",
    "aggregate_probability",
    "avg_correlation",
    "directly_assigned",
    "alias",
    "label",
    "name",
)
WHB_MANIFEST_URL = (
    "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
    "releases/20250531/manifest.json"
)
WHB_DATASET_DIRECTORY = "WHB-10Xv3"
WHB_TAXONOMY_DIRECTORY = "WHB-taxonomy"
WHB_HIERARCHY = [
    "CCN202210140_SUPC",
    "CCN202210140_CLUS",
    "CCN202210140_SUBC",
]
WHB_EXPRESSION_MATRIX_NAMES = (
    "WHB-10Xv3-Neurons",
    "WHB-10Xv3-Nonneurons",
)
WMB_DATASET_DIRECTORY = "WMB-10X"
WMB_TAXONOMY_DIRECTORY = "WMB-taxonomy"
WMB_EXPRESSION_DIRECTORIES = ("WMB-10Xv2", "WMB-10Xv3", "WMB-10XMulti")
WMB_HIERARCHY = [
    "CCN20230722_CLAS",
    "CCN20230722_SUBC",
    "CCN20230722_SUPT",
    "CCN20230722_CLUS",
]
WMB_REGION_COLUMN = "region_of_interest_acronym"
WMB_FEATURE_MATRIX_COLUMN = "feature_matrix_label"
MMC_GENE_MAPPER_URL = (
    "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
    "mapmycells/mmc-gene-mapper/20250630/mmc_gene_mapper.2025-08-04.db"
)
MMC_GENE_MAPPER_SIZE = 16_219_131_904

FULL_REFERENCE_MANIFEST_KEYS = {
    "whb": {
        "directory": WHB_DATASET_DIRECTORY,
        "marker": ("query_markers.n10.20240221800", "json"),
        "stats": ("precomputed_stats.siletti.training", "h5"),
    },
    "wmb": {
        "directory": WMB_DATASET_DIRECTORY,
        "marker": ("mouse_markers_230821", "json"),
        "stats": ("precomputed_stats_ABC_revision_230821", "h5"),
    },
}


@dataclass(frozen=True)
class MapMyCellsReference:
    """One reference taxonomy to map each MerXen sample against."""

    name: str
    output_dir: Path
    marker_lookup_path: Path
    precomputed_stats_path: Path
    column_prefix: str
    uns_key: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class RegionReferenceArtifacts:
    """Cached files generated for a strict WHB ROI reference."""

    marker_lookup_path: Path
    precomputed_stats_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def prepare_mapmycells_query(
    input_h5ad: Path | str,
    output_h5ad: Path | str,
    *,
    query_layer: str | None = "counts",
    gene_id_column: str | None = None,
    obs_id_column: str | None = None,
) -> Path:
    """Write a MapMyCells-ready H5AD query file.

    The Squidpy clustering stage leaves normalized/log-transformed values in
    ``X`` and preserves raw counts in ``layers["counts"]``. MapMyCells expects
    the query matrix in ``X``, so this helper copies the selected layer into
    ``X`` before writing a local query file.

    Args:
        input_h5ad: Clustered AnnData from ``clustering_squidpy``.
        output_h5ad: Destination H5AD consumed by MapMyCells.
        query_layer: AnnData layer to copy into ``X``. Use ``None`` to keep the
            current ``X`` matrix.
        gene_id_column: Optional ``var`` column to use as gene identifiers.
        obs_id_column: Optional ``obs`` column to use as cell identifiers.

    Returns:
        Path to the written query H5AD.
    """
    input_h5ad = Path(input_h5ad)
    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_h5ad)
    try:
        if query_layer is not None:
            if query_layer not in adata.layers:
                raise KeyError(
                    f"Requested query_layer={query_layer!r} not found in {input_h5ad}. "
                    f"Available layers: {list(adata.layers.keys())}"
                )
            adata.X = _copy_matrix(adata.layers[query_layer])

        if gene_id_column is not None:
            if gene_id_column not in adata.var.columns:
                raise KeyError(
                    f"Requested gene_id_column={gene_id_column!r} not found in "
                    f"{input_h5ad}. Available var columns: {list(adata.var.columns)}"
                )
            adata.var_names = _index_from_column_with_fallback(
                adata.var,
                column=gene_id_column,
                fallback=adata.var_names,
            )

        if obs_id_column is not None:
            if obs_id_column not in adata.obs.columns:
                raise KeyError(
                    f"Requested obs_id_column={obs_id_column!r} not found in "
                    f"{input_h5ad}. Available obs columns: {list(adata.obs.columns)}"
                )
            adata.obs_names = _index_from_column_with_fallback(
                adata.obs,
                column=obs_id_column,
                fallback=adata.obs_names,
            )

        adata.var_names = pd.Index(adata.var_names.astype(str), name=None)
        adata.obs_names = pd.Index(adata.obs_names.astype(str), name=None)
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adata.var.index.name = None
        adata.obs.index.name = None
        adata.write_h5ad(output_h5ad)
    finally:
        del adata
        force_release(note=f"after preparing MapMyCells query {input_h5ad.name}")

    return output_h5ad


def run_mapmycells(config: MapMyCellsConfig) -> dict[str, dict[str, dict[str, Path]]]:
    """Run local MapMyCells assignment for every sample in a pair.

    Args:
        config: Validated MapMyCells stage configuration.

    Returns:
        Mapping from sample ID to output artifact paths.
    """
    if config.tmp_dir is not None:
        config.tmp_dir.mkdir(parents=True, exist_ok=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    references = _build_mapmycells_references(config)
    results: dict[str, dict[str, dict[str, Path]]] = {}

    for sample in config.samples:
        log_status(
            f"[{sample.sample_id}] Starting MapMyCells "
            f"(platform={sample.platform}, reference_mode={config.reference_mode}, "
            f"bootstrap_factor={config.bootstrap_factor}, "
            f"plots_only={config.plots_only})"
        )
        _require_existing_file(
            sample.anndata_path, f"clustered AnnData for {sample.sample_id}"
        )
        results[sample.sample_id] = {}
        for reference in references:
            results[sample.sample_id][reference.name] = _run_mapmycells_reference(
                config,
                sample=sample,
                reference=reference,
            )
        force_release(note=f"after MapMyCells {sample.sample_id}")

    manifest_path = config.output_dir / f"{config.pair_id}_mapmycells_manifest.json"
    _write_results_manifest(manifest_path, config, references, results)
    return results


def _build_mapmycells_references(config: MapMyCellsConfig) -> list[MapMyCellsReference]:
    references: list[MapMyCellsReference] = []
    if (
        not config.plots_only
        and config.reference_atlas == "wmb"
        and config.query_species == "human"
    ):
        config.gene_mapping_db_path = _resolve_gene_mapping_db(config)
    if config.reference_mode in {"whole_brain", "both"}:
        if config.plots_only:
            marker_lookup_path = config.marker_lookup_path or Path("")
            precomputed_stats_path = config.precomputed_stats_path or Path("")
            reference_manifest = {
                "reference_type": "whole_brain",
                "reference_atlas": config.reference_atlas,
                "plots_only": True,
            }
        else:
            (
                marker_lookup_path,
                precomputed_stats_path,
                reference_manifest,
            ) = _resolve_full_reference_artifacts(config)
        references.append(
            MapMyCellsReference(
                name="whole_brain" if config.reference_atlas == "whb" else "wmb",
                output_dir=(
                    config.output_dir
                    if config.reference_atlas == "whb"
                    else config.output_dir / "wmb"
                ),
                marker_lookup_path=marker_lookup_path,
                precomputed_stats_path=precomputed_stats_path,
                column_prefix=(
                    MAPMYCELLS_PREFIX
                    if config.reference_atlas == "whb"
                    else MAPMYCELLS_WMB_PREFIX
                ),
                uns_key=(
                    MAPMYCELLS_UNS_KEY
                    if config.reference_atlas == "whb"
                    else MAPMYCELLS_WMB_UNS_KEY
                ),
                manifest=reference_manifest,
            )
        )
    if config.reference_mode in {"region", "both"}:
        region_name = _sanitize_token(config.region_name)
        if config.reference_atlas == "whb":
            reference_name = f"region_{region_name}"
            column_prefix = MAPMYCELLS_REGION_PREFIX_TEMPLATE.format(
                region_name=region_name
            )
            uns_key = MAPMYCELLS_REGION_UNS_KEY_TEMPLATE.format(region_name=region_name)
        else:
            reference_name = f"wmb_region_{region_name}"
            column_prefix = MAPMYCELLS_WMB_REGION_PREFIX_TEMPLATE.format(
                region_name=region_name
            )
            uns_key = MAPMYCELLS_WMB_REGION_UNS_KEY_TEMPLATE.format(
                region_name=region_name
            )
        if config.plots_only:
            region_artifacts = RegionReferenceArtifacts(
                marker_lookup_path=Path(""),
                precomputed_stats_path=Path(""),
                manifest_path=Path(""),
                manifest={
                    "reference_type": "region",
                    "config": {
                        "region_name": region_name,
                        "region_labels": list(config.region_labels),
                    },
                    "plots_only": True,
                },
            )
        else:
            region_artifacts = prepare_region_mapmycells_reference(config)
        references.append(
            MapMyCellsReference(
                name=reference_name,
                output_dir=config.output_dir / reference_name,
                marker_lookup_path=region_artifacts.marker_lookup_path,
                precomputed_stats_path=region_artifacts.precomputed_stats_path,
                column_prefix=column_prefix,
                uns_key=uns_key,
                manifest=region_artifacts.manifest,
            )
        )
    return references


def _resolve_full_reference_artifacts(
    config: MapMyCellsConfig,
) -> tuple[Path, Path, dict[str, Any]]:
    marker_path = config.marker_lookup_path
    stats_path = config.precomputed_stats_path
    if marker_path is not None:
        _require_existing_file(marker_path, "MapMyCells marker lookup")
    if stats_path is not None:
        _require_existing_file(stats_path, "MapMyCells precomputed stats")

    downloaded = False
    if marker_path is None or stats_path is None:
        if not config.auto_download_references:
            raise ValueError(
                "Whole-brain MapMyCells reference paths are missing and automatic "
                "downloads are disabled."
            )
        manifest = _load_abc_manifest()
        keys = FULL_REFERENCE_MANIFEST_KEYS[config.reference_atlas]
        directory = str(keys["directory"])
        cache_dir = config.region_cache_dir / "abc_atlas"
        if marker_path is None:
            marker_name, marker_suffix = cast(tuple[str, str], keys["marker"])
            marker_info = _manifest_file_info(
                manifest,
                directory,
                "mapmycells",
                marker_name,
                "files",
                marker_suffix,
            )
            marker_path = _ensure_manifest_file(marker_info, cache_dir)
            downloaded = True
        if stats_path is None:
            stats_name, stats_suffix = cast(tuple[str, str], keys["stats"])
            stats_info = _manifest_file_info(
                manifest,
                directory,
                "mapmycells",
                stats_name,
                "files",
                stats_suffix,
            )
            stats_path = _ensure_manifest_file(stats_info, cache_dir)
            downloaded = True

    if marker_path is None or stats_path is None:
        raise RuntimeError("Could not resolve the MapMyCells full reference artifacts.")
    return (
        marker_path,
        stats_path,
        {
            "reference_type": "whole_brain",
            "reference_atlas": config.reference_atlas,
            "query_species": config.query_species,
            "marker_lookup_path": str(marker_path),
            "precomputed_stats_path": str(stats_path),
            "gene_mapping_db_path": _path_as_str(config.gene_mapping_db_path),
            "downloaded": downloaded,
        },
    )


def _resolve_gene_mapping_db(config: MapMyCellsConfig) -> Path:
    if config.gene_mapping_db_path is not None:
        _require_existing_file(config.gene_mapping_db_path, "MapMyCells gene mapper DB")
        return config.gene_mapping_db_path
    if not config.auto_download_references:
        raise ValueError(
            "Human-to-WMB mapping requires gene_mapping_db_path when automatic "
            "reference downloads are disabled."
        )
    output_path = (
        config.region_cache_dir
        / "abc_atlas"
        / "mapmycells"
        / "mmc-gene-mapper"
        / "20250630"
        / "mmc_gene_mapper.2025-08-04.db"
    )
    return _ensure_url_file(
        MMC_GENE_MAPPER_URL,
        output_path,
        expected_size=MMC_GENE_MAPPER_SIZE,
    )


def _run_mapmycells_reference(
    config: MapMyCellsConfig,
    *,
    sample: Any,
    reference: MapMyCellsReference,
) -> dict[str, Path]:
    sample_dir = reference.output_dir / sample.platform.lower()
    sample_dir.mkdir(parents=True, exist_ok=True)

    query_h5ad = sample_dir / f"{sample.sample_id}_mapmycells_query.h5ad"
    extended_json = sample_dir / f"{sample.sample_id}_mapmycells_extended.json"
    csv_path = sample_dir / f"{sample.sample_id}_mapmycells.csv"
    log_path = sample_dir / f"{sample.sample_id}_mapmycells.log"
    stdout_path = sample_dir / f"{sample.sample_id}_mapmycells_stdout.log"
    stderr_path = sample_dir / f"{sample.sample_id}_mapmycells_stderr.log"
    command_path = sample_dir / f"{sample.sample_id}_mapmycells_command.json"
    annotated_h5ad = sample_dir / f"{sample.sample_id}_mapmycells_annotated.h5ad"
    umap_plot = sample_dir / f"{sample.sample_id}_mapmycells_umap.png"
    spatial_plot = sample_dir / f"{sample.sample_id}_mapmycells_spatial.png"
    umap_cluster_by_supercluster_dir = (
        sample_dir / f"{sample.sample_id}_mapmycells_umap_cluster_by_supercluster"
    )
    quality_scatter_plot = (
        sample_dir / f"{sample.sample_id}_mapmycells_quality_scatter.png"
    )
    supercluster_qc_plot = (
        sample_dir / f"{sample.sample_id}_mapmycells_supercluster_assignment_qc.png"
    )
    cluster_qc_plot = (
        sample_dir / f"{sample.sample_id}_mapmycells_cluster_assignment_qc.png"
    )
    spatial_supercluster_grid_plot = (
        sample_dir / f"{sample.sample_id}_mapmycells_spatial_supercluster_grid.png"
    )

    if config.plots_only:
        _require_existing_file(
            csv_path,
            f"existing MapMyCells CSV for {sample.sample_id}",
        )
        _require_existing_file(
            extended_json,
            f"existing MapMyCells extended JSON for {sample.sample_id}",
        )
    else:
        query_h5ad = prepare_mapmycells_query(
            sample.anndata_path,
            query_h5ad,
            query_layer=sample.query_layer,
            gene_id_column=sample.gene_id_column,
            obs_id_column=sample.obs_id_column,
        )
        command = build_mapmycells_command(
            config,
            query_h5ad=query_h5ad,
            extended_json=extended_json,
            csv_path=csv_path,
            log_path=log_path,
            marker_lookup_path=reference.marker_lookup_path,
            precomputed_stats_path=reference.precomputed_stats_path,
        )
        _write_command_manifest(command_path, command)
        _run_command(command, stdout_path=stdout_path, stderr_path=stderr_path)

    annotate_h5ad_with_mapmycells(
        sample.anndata_path,
        csv_path,
        annotated_h5ad,
        extended_json_path=extended_json,
        command_path=command_path,
        log_path=log_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        umap_plot_path=umap_plot,
        spatial_plot_path=spatial_plot,
        umap_cluster_by_supercluster_dir=umap_cluster_by_supercluster_dir,
        quality_scatter_plot_path=quality_scatter_plot,
        supercluster_qc_plot_path=supercluster_qc_plot,
        cluster_qc_plot_path=cluster_qc_plot,
        spatial_supercluster_grid_plot_path=spatial_supercluster_grid_plot,
        column_prefix=reference.column_prefix,
        uns_key=reference.uns_key,
        reference_metadata=reference.manifest,
    )

    outputs = {
        "query_h5ad": query_h5ad,
        "extended_json": extended_json,
        "csv": csv_path,
        "log": log_path,
        "stdout_log": stdout_path,
        "stderr_log": stderr_path,
        "command_json": command_path,
        "annotated_h5ad": annotated_h5ad,
        "umap_plot": umap_plot,
        "spatial_plot": spatial_plot,
    }
    for key, path in {
        "umap_cluster_by_supercluster_dir": umap_cluster_by_supercluster_dir,
        "quality_scatter_plot": quality_scatter_plot,
        "supercluster_qc_plot": supercluster_qc_plot,
        "cluster_qc_plot": cluster_qc_plot,
        "spatial_supercluster_grid_plot": spatial_supercluster_grid_plot,
    }.items():
        if path.exists():
            outputs[key] = path
    return outputs


def build_mapmycells_command(
    config: MapMyCellsConfig,
    *,
    query_h5ad: Path,
    extended_json: Path,
    csv_path: Path,
    log_path: Path,
    marker_lookup_path: Path | None = None,
    precomputed_stats_path: Path | None = None,
) -> list[str]:
    """Build the ``cell_type_mapper`` command-line invocation."""
    marker_lookup_path = marker_lookup_path or config.marker_lookup_path
    precomputed_stats_path = precomputed_stats_path or config.precomputed_stats_path
    if marker_lookup_path is None or precomputed_stats_path is None:
        raise ValueError("MapMyCells marker lookup and precomputed stats are required.")
    command = [
        sys.executable,
        "-m",
        "merxen.analysis.mapmycells_entrypoint",
        "--query_path",
        str(query_h5ad),
        "--extended_result_path",
        str(extended_json),
        "--csv_result_path",
        str(csv_path),
        "--log_path",
        str(log_path),
        "--cloud_safe",
        _bool_arg(config.cloud_safe),
        "--query_markers.serialized_lookup",
        str(marker_lookup_path),
        "--precomputed_stats.path",
        str(precomputed_stats_path),
        "--type_assignment.normalization",
        config.normalization,
        "--type_assignment.bootstrap_iteration",
        str(config.bootstrap_iteration),
        "--type_assignment.bootstrap_factor",
        str(config.bootstrap_factor),
        "--type_assignment.n_processors",
        str(config.n_processors),
        "--flatten",
        _bool_arg(config.flatten),
    ]
    if config.drop_level is not None:
        command.extend(["--drop_level", config.drop_level])
    if config.chunk_size is not None:
        command.extend(["--type_assignment.chunk_size", str(config.chunk_size)])
    if config.rng_seed is not None:
        command.extend(["--type_assignment.rng_seed", str(config.rng_seed)])
    if config.max_gb is not None:
        command.extend(["--max_gb", str(config.max_gb)])
    if config.tmp_dir is not None:
        command.extend(["--tmp_dir", str(config.tmp_dir)])
    if config.gene_mapping_db_path is not None:
        command.extend(["--gene_mapping.db_path", str(config.gene_mapping_db_path)])
    if config.verbose_csv:
        command.extend(["--verbose_csv", _bool_arg(config.verbose_csv)])
    command.extend(config.extra_args)
    return command


def annotate_h5ad_with_mapmycells(
    input_h5ad: Path | str,
    csv_path: Path | str,
    output_h5ad: Path | str,
    *,
    extended_json_path: Path | str | None = None,
    command_path: Path | str | None = None,
    log_path: Path | str | None = None,
    stdout_path: Path | str | None = None,
    stderr_path: Path | str | None = None,
    umap_plot_path: Path | str | None = None,
    spatial_plot_path: Path | str | None = None,
    umap_cluster_by_supercluster_dir: Path | str | None = None,
    quality_scatter_plot_path: Path | str | None = None,
    supercluster_qc_plot_path: Path | str | None = None,
    cluster_qc_plot_path: Path | str | None = None,
    spatial_supercluster_grid_plot_path: Path | str | None = None,
    column_prefix: str = MAPMYCELLS_PREFIX,
    uns_key: str = MAPMYCELLS_UNS_KEY,
    reference_metadata: dict[str, Any] | None = None,
) -> Path:
    """Attach MapMyCells CSV assignments to ``adata.obs`` and write H5AD."""
    input_h5ad = Path(input_h5ad)
    csv_path = Path(csv_path)
    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    assignments = read_mapmycells_csv(csv_path)
    adata = ad.read_h5ad(input_h5ad)
    try:
        indexed = assignments.set_index(assignments.columns[0], drop=False)
        indexed.index = indexed.index.astype(str)
        indexed = indexed[~indexed.index.duplicated(keep="first")]
        aligned = indexed.reindex(adata.obs_names.astype(str))
        assignment_columns: list[str] = []
        for column in aligned.columns:
            target = f"{column_prefix}{column}"
            adata.obs[target] = aligned[column].to_numpy()
            assignment_columns.append(target)

        matched = int(aligned.iloc[:, 0].notna().sum()) if not aligned.empty else 0
        plot_column = choose_mapmycells_assignment_column(
            adata,
            column_prefix=column_prefix,
        )
        plot_paths: dict[str, Any] = {}
        if umap_plot_path is not None:
            plot_paths["umap"] = str(
                plot_mapmycells_umap(
                    adata,
                    umap_plot_path,
                    color=plot_column,
                    column_prefix=column_prefix,
                )
            )
        if spatial_plot_path is not None:
            plot_paths["spatial"] = str(
                plot_mapmycells_spatial(
                    adata,
                    spatial_plot_path,
                    color=plot_column,
                    column_prefix=column_prefix,
                )
            )
        if quality_scatter_plot_path is not None and extended_json_path is not None:
            plot_paths["quality_scatter"] = str(
                plot_mapmycells_quality_scatter(
                    adata,
                    extended_json_path,
                    quality_scatter_plot_path,
                    column_prefix=column_prefix,
                )
            )
        if umap_cluster_by_supercluster_dir is not None:
            umap_cluster_paths = (
                _plot_mapmycells_umap_clusters_by_supercluster_if_available(
                    adata,
                    umap_cluster_by_supercluster_dir,
                    column_prefix=column_prefix,
                )
            )
            if umap_cluster_paths:
                plot_paths["umap_cluster_by_supercluster"] = {
                    supercluster: str(path)
                    for supercluster, path in umap_cluster_paths.items()
                }
        if supercluster_qc_plot_path is not None:
            level_path = _plot_mapmycells_assignment_qc_if_available(
                adata,
                supercluster_qc_plot_path,
                level="supercluster",
                column_prefix=column_prefix,
            )
            if level_path is not None:
                plot_paths["supercluster_assignment_qc"] = str(level_path)
        if cluster_qc_plot_path is not None:
            level_path = _plot_mapmycells_assignment_qc_if_available(
                adata,
                cluster_qc_plot_path,
                level="cluster",
                column_prefix=column_prefix,
            )
            if level_path is not None:
                plot_paths["cluster_assignment_qc"] = str(level_path)
        if spatial_supercluster_grid_plot_path is not None:
            grid_path = _plot_mapmycells_spatial_supercluster_grid_if_available(
                adata,
                spatial_supercluster_grid_plot_path,
                column_prefix=column_prefix,
            )
            if grid_path is not None:
                plot_paths["spatial_supercluster_grid"] = str(grid_path)

        adata.uns[uns_key] = {
            "csv_path": str(csv_path),
            "csv_header_comments": _read_comment_header(csv_path),
            "column_prefix": column_prefix,
            "assignment_columns": assignment_columns,
            "plot_assignment_column": plot_column,
            "plot_paths": plot_paths,
            "n_assignments": int(len(assignments)),
            "n_obs": int(adata.n_obs),
            "n_matched_obs": matched,
            "extended_json_path": _path_as_str(extended_json_path),
            "extended_json_text": _read_text_if_present(extended_json_path),
            "command_json_path": _path_as_str(command_path),
            "command_json_text": _read_text_if_present(command_path),
            "log_path": _path_as_str(log_path),
            "log_text": _read_text_if_present(log_path),
            "stdout_log_path": _path_as_str(stdout_path),
            "stdout_log_text": _read_text_if_present(stdout_path),
            "stderr_log_path": _path_as_str(stderr_path),
            "stderr_log_text": _read_text_if_present(stderr_path),
            "reference_metadata": reference_metadata or {},
        }
        adata.write_h5ad(output_h5ad)
    finally:
        del adata
        force_release(note=f"after annotating MapMyCells output {input_h5ad.name}")

    return output_h5ad


def choose_mapmycells_assignment_column(
    adata: ad.AnnData,
    *,
    column_prefix: str = MAPMYCELLS_PREFIX,
    max_categories: int = MAPMYCELLS_MAX_LEGEND_CATEGORIES,
) -> str:
    """Choose the most specific MapMyCells label column that remains plottable."""
    preferred = [
        f"{column_prefix}{suffix}"
        for suffix in MAPMYCELLS_ASSIGNMENT_COLOR_SUFFIX_CANDIDATES
        if f"{column_prefix}{suffix}" in adata.obs
    ]
    name_columns = [
        str(column)
        for column in adata.obs.columns
        if str(column).startswith(column_prefix) and str(column).endswith("_name")
    ]
    label_columns = [
        str(column)
        for column in adata.obs.columns
        if str(column).startswith(column_prefix) and str(column).endswith("_label")
    ]
    candidates = list(dict.fromkeys([*preferred, *name_columns, *label_columns]))
    if not candidates and column_prefix != MAPMYCELLS_PREFIX:
        candidates = [
            column
            for column in adata.obs.columns
            if str(column).startswith(column_prefix)
        ]
    if not candidates:
        preferred = [
            f"{MAPMYCELLS_PREFIX}{suffix}"
            for suffix in MAPMYCELLS_ASSIGNMENT_COLOR_SUFFIX_CANDIDATES
            if f"{MAPMYCELLS_PREFIX}{suffix}" in adata.obs
        ]
        candidates = list(preferred)
    if not candidates:
        raise KeyError(
            f"No MapMyCells assignment columns with prefix {column_prefix!r} "
            "were found in adata.obs."
        )

    category_counts = {
        column: int(pd.Series(adata.obs[column]).nunique(dropna=True))
        for column in candidates
    }
    for column in candidates:
        if 0 < category_counts[column] <= max_categories:
            return column
    return min(candidates, key=lambda column: category_counts[column])


def plot_mapmycells_umap(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    color: str | None = None,
    column_prefix: str = MAPMYCELLS_PREFIX,
    point_size: float = 1.0,
    alpha: float = 0.65,
    dpi: int = 180,
) -> Path:
    """Plot the existing clustering UMAP colored by MapMyCells assignments."""
    color = color or choose_mapmycells_assignment_column(
        adata,
        column_prefix=column_prefix,
    )
    return _plot_mapmycells_embedding(
        adata,
        output_path,
        basis="X_umap",
        color=color,
        title="MapMyCells assignment UMAP",
        column_prefix=column_prefix,
        point_size=point_size,
        alpha=alpha,
        dpi=dpi,
    )


def plot_mapmycells_spatial(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    color: str | None = None,
    column_prefix: str = MAPMYCELLS_PREFIX,
    point_size: float = 0.25,
    alpha: float = 0.65,
    dpi: int = 180,
) -> Path:
    """Plot spatial coordinates colored by MapMyCells assignments."""
    color = color or choose_mapmycells_assignment_column(
        adata,
        column_prefix=column_prefix,
    )
    return _plot_mapmycells_embedding(
        adata,
        output_path,
        basis="spatial",
        color=color,
        title="MapMyCells assignment spatial plot",
        column_prefix=column_prefix,
        point_size=point_size,
        alpha=alpha,
        dpi=dpi,
    )


def plot_mapmycells_umap_clusters_by_supercluster(
    adata: ad.AnnData,
    output_dir: Path | str,
    *,
    column_prefix: str = MAPMYCELLS_PREFIX,
    point_size_background: float = 0.35,
    point_size_highlight: float = 1.4,
    alpha_background: float = 0.24,
    alpha_highlight: float = 0.78,
    dpi: int = 180,
) -> dict[str, Path]:
    """Plot one UMAP per supercluster, coloring member cells by cluster."""
    if "X_umap" not in adata.obsm:
        raise KeyError("Expected adata.obsm['X_umap'] for MapMyCells UMAP plots.")
    _, supercluster_columns = _resolve_mapmycells_level_columns(
        adata,
        level="supercluster",
        column_prefix=column_prefix,
    )
    _, cluster_columns = _resolve_mapmycells_level_columns(
        adata,
        level="cluster",
        column_prefix=column_prefix,
    )
    supercluster_labels = _mapmycells_level_label_series(adata, supercluster_columns)
    cluster_labels = _mapmycells_level_label_series(adata, cluster_columns)
    coords = np.asarray(adata.obsm["X_umap"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            "Expected adata.obsm['X_umap'] to have at least two columns; "
            f"found shape {coords.shape}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x = coords[:, 0]
    y = coords[:, 1]
    x_limits = _axis_limits(x)
    y_limits = _axis_limits(y)
    supercluster_values = supercluster_labels.astype(str).to_numpy()
    cluster_values = cluster_labels.astype(str).to_numpy()
    superclusters = [
        str(label)
        for label in supercluster_labels.value_counts().index
        if str(label) != "unassigned"
    ]

    paths: dict[str, Path] = {}
    used_tokens: set[str] = set()
    for supercluster in superclusters:
        mask = supercluster_values == supercluster
        if not mask.any():
            continue
        clusters = [
            str(label)
            for label in pd.Series(cluster_values[mask]).value_counts().index
            if str(label) != "unassigned"
        ]
        if not clusters:
            continue
        token = _unique_plot_token(supercluster, used=used_tokens)
        output_path = output_dir / f"supercluster_{token}.png"
        cmap = _categorical_cmap(len(clusters))
        cluster_to_color = {cluster: cmap(idx) for idx, cluster in enumerate(clusters)}

        fig, ax = plt.subplots(figsize=(7.0, 6.5))
        ax.scatter(
            x[~mask],
            y[~mask],
            s=float(point_size_background),
            c="#c7c7c7",
            alpha=float(alpha_background),
            linewidths=0,
            rasterized=True,
        )
        for cluster in clusters:
            cluster_mask = mask & (cluster_values == cluster)
            ax.scatter(
                x[cluster_mask],
                y[cluster_mask],
                s=float(point_size_highlight),
                c=[cluster_to_color[cluster]],
                alpha=float(alpha_highlight),
                linewidths=0,
                rasterized=True,
                label=cluster,
            )
        ax.set_title(_wrapped_title(f"{supercluster}\nMapMyCells clusters on UMAP"))
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        if len(clusters) <= MAPMYCELLS_MAX_LEGEND_CATEGORIES:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=cluster_to_color[cluster],
                    markeredgewidth=0,
                    markersize=4,
                    label=cluster,
                )
                for cluster in clusters
            ]
            ax.legend(
                handles=handles,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
                fontsize=5,
                title=f"{len(clusters)} clusters",
                title_fontsize=6,
            )
        else:
            ax.text(
                0.02,
                0.98,
                f"{len(clusters)} clusters",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.8,
                },
            )
        fig.tight_layout()
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        paths[supercluster] = output_path
    return paths


def plot_mapmycells_assignment_qc(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    level: str,
    column_prefix: str = MAPMYCELLS_PREFIX,
    max_categories: int = MAPMYCELLS_QC_MAX_ASSIGNMENT_CATEGORIES,
    dpi: int = 180,
) -> Path:
    """Plot assignment counts and confidence summaries for one taxonomy level."""
    output_path = prepare_plot_output(output_path)
    resolved_level, columns = _resolve_mapmycells_level_columns(
        adata,
        level=level,
        column_prefix=column_prefix,
    )
    labels = _mapmycells_level_label_series(adata, columns)
    confidence, confidence_label = _mapmycells_level_confidence_series(adata, columns)

    counts = labels.value_counts()
    categories = [str(value) for value in counts.index[: int(max_categories)]]
    if not categories:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        _plot_empty_axis(ax, f"No {level} assignments were available.")
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        return output_path

    plot_df = pd.DataFrame(
        {
            "assignment": labels.astype(str).to_numpy(),
            "confidence": confidence.to_numpy(dtype=float),
        }
    )
    plot_df = plot_df[plot_df["assignment"].isin(categories)].copy()
    y_positions = np.arange(len(categories), dtype=float)
    height = max(4.5, 0.22 * len(categories) + 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, height), sharey=True)

    _plot_assignment_count_panel(
        axes[0],
        plot_df,
        categories=categories,
        confidence_label=confidence_label,
    )
    _plot_assignment_confidence_panel(
        axes[1],
        plot_df,
        categories=categories,
        y_positions=y_positions,
        confidence_label=confidence_label,
    )
    _plot_assignment_low_confidence_panel(
        axes[2],
        plot_df,
        categories=categories,
        confidence_label=confidence_label,
    )

    for ax in axes:
        ax.set_yticks(y_positions)
        ax.set_yticklabels(categories, fontsize=5)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.5)
    if len(counts) > len(categories):
        axes[0].text(
            0.01,
            -0.08,
            f"Showing top {len(categories)} of {len(counts)} assignments by count",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )

    fig.suptitle(
        f"MapMyCells {_format_level_display(resolved_level)} assignment QC",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_mapmycells_spatial_supercluster_grid(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    column_prefix: str = MAPMYCELLS_PREFIX,
    point_size_background: float = 0.08,
    point_size_highlight: float = 0.45,
    alpha_background: float = 0.32,
    alpha_highlight: float = 0.82,
    dpi: int = 180,
) -> Path:
    """Plot a spatial small-multiple grid highlighting each supercluster."""
    output_path = prepare_plot_output(output_path)
    if "spatial" not in adata.obsm:
        raise KeyError("Expected adata.obsm['spatial'] for MapMyCells spatial grid.")
    _, columns = _resolve_mapmycells_level_columns(
        adata,
        level="supercluster",
        column_prefix=column_prefix,
    )
    labels = _mapmycells_level_label_series(adata, columns)
    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            "Expected adata.obsm['spatial'] to have at least two columns; "
            f"found shape {coords.shape}."
        )

    label_counts = labels.value_counts()
    categories = [
        str(label) for label in label_counts.index if str(label) != "unassigned"
    ]
    if not categories:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        _plot_empty_axis(ax, "No supercluster assignments were available.")
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        return output_path

    n_categories = len(categories)
    n_cols = min(4, int(np.ceil(np.sqrt(n_categories))))
    n_rows = int(np.ceil(n_categories / n_cols))
    fig_width = 3.2 * n_cols
    fig_height = 3.2 * n_rows
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    x = coords[:, 0]
    y = coords[:, 1]
    x_pad = _axis_padding(x)
    y_pad = _axis_padding(y)
    x_limits = (float(np.nanmin(x) - x_pad), float(np.nanmax(x) + x_pad))
    y_limits = (float(np.nanmin(y) - y_pad), float(np.nanmax(y) + y_pad))

    for ax, category in zip(axes.ravel(), categories, strict=False):
        mask = labels.astype(str).to_numpy() == category
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
        ax.set_title(_wrapped_title(category), fontsize=7)
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


def plot_mapmycells_quality_scatter(
    adata: ad.AnnData,
    extended_json_path: Path | str,
    output_path: Path | str,
    *,
    levels: tuple[str, ...] = MAPMYCELLS_QC_LEVELS,
    column_prefix: str = MAPMYCELLS_PREFIX,
    point_size: float = 2.0,
    alpha: float = 0.28,
    dpi: int = 180,
) -> Path:
    """Plot JSON-backed MapMyCells quality metrics at selected taxonomy levels."""
    output_path = prepare_plot_output(output_path)
    qc = read_mapmycells_extended_qc(extended_json_path)
    selected_levels = _select_extended_qc_levels(qc, levels=levels)
    n_rows = max(1, len(selected_levels))
    n_cols = 5
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    if qc.empty or not selected_levels:
        _plot_empty_axis(
            axes[0, 0],
            "No extended JSON quality metrics were available.",
        )
        for ax in axes.ravel()[1:]:
            ax.set_visible(False)
        save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        return output_path

    complexity_label, complexity = _mapmycells_cell_complexity(adata)
    cell_ids = _mapmycells_cell_ids_for_obs(adata, column_prefix=column_prefix)
    obs_qc = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "cell_complexity": complexity,
        }
    )
    obs_qc = obs_qc.drop_duplicates("cell_id", keep="first")
    qc = qc.merge(obs_qc, on="cell_id", how="left")

    for row_idx, (level_token, level_display) in enumerate(selected_levels):
        row_axes = axes[row_idx]
        level_df = qc[qc["level_token"] == level_token].copy()
        _plot_scatter_metric(
            row_axes[0],
            level_df["cell_complexity"],
            level_df["avg_correlation"],
            xlabel=complexity_label,
            ylabel="Average correlation",
            title=f"{level_display}: correlation vs cell QC",
            point_size=point_size,
            alpha=alpha,
            x_log=True,
        )
        _plot_scatter_metric(
            row_axes[1],
            level_df["cell_complexity"],
            level_df["bootstrapping_probability"],
            xlabel=complexity_label,
            ylabel="Bootstrapping probability",
            title=f"{level_display}: bootstrap vs cell QC",
            point_size=point_size,
            alpha=alpha,
            x_log=True,
            y_limits=(0.0, 1.02),
        )
        _plot_scatter_metric(
            row_axes[2],
            level_df["avg_correlation"],
            level_df["bootstrapping_probability"],
            xlabel="Average correlation",
            ylabel="Bootstrapping probability",
            title=f"{level_display}: bootstrap vs correlation",
            point_size=point_size,
            alpha=alpha,
            y_limits=(0.0, 1.02),
        )
        _plot_hist_metric(
            row_axes[3],
            level_df["aggregate_probability"],
            xlabel="Aggregate probability",
            title=f"{level_display}: aggregate probability",
            x_limits=(0.0, 1.0),
        )
        _plot_hist_metric(
            row_axes[4],
            level_df["runner_up_margin"],
            xlabel="Assigned minus runner-up probability",
            title=f"{level_display}: runner-up margin",
            x_limits=(-1.0, 1.0),
        )
        direct_fraction = pd.to_numeric(
            level_df["directly_assigned"], errors="coerce"
        ).mean()
        if np.isfinite(direct_fraction):
            row_axes[4].text(
                0.02,
                0.95,
                f"directly assigned: {100.0 * direct_fraction:.1f}%",
                transform=row_axes[4].transAxes,
                ha="left",
                va="top",
                fontsize=7,
            )

    fig.suptitle("MapMyCells assignment quality metrics", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def read_mapmycells_extended_qc(
    extended_json_path: Path | str,
) -> pd.DataFrame:
    """Read MapMyCells extended JSON quality metrics into a tidy table."""
    path = Path(extended_json_path)
    if not path.exists():
        return _empty_extended_qc_frame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Could not decode MapMyCells extended JSON: %s", path)
        return _empty_extended_qc_frame()
    if not isinstance(payload, dict):
        return _empty_extended_qc_frame()

    taxonomy_tree = payload.get("taxonomy_tree")
    taxonomy_tree = taxonomy_tree if isinstance(taxonomy_tree, dict) else {}
    hierarchy_mapper = taxonomy_tree.get("hierarchy_mapper")
    hierarchy_mapper = hierarchy_mapper if isinstance(hierarchy_mapper, dict) else {}
    name_mapper = taxonomy_tree.get("name_mapper")
    name_mapper = name_mapper if isinstance(name_mapper, dict) else {}
    results = payload.get("results")
    if not isinstance(results, list):
        return _empty_extended_qc_frame()

    rows: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        cell_id = str(result.get("cell_id", ""))
        if not cell_id:
            continue
        for level_key, metrics in result.items():
            if level_key == "cell_id" or not isinstance(metrics, dict):
                continue
            level_display = str(hierarchy_mapper.get(level_key, level_key))
            level_token = _normalize_level_token(level_display)
            assignment = metrics.get("assignment")
            assignment = "" if assignment is None else str(assignment)
            runner_up_probability = _first_numeric(metrics.get("runner_up_probability"))
            bootstrap = _numeric_or_nan(metrics.get("bootstrapping_probability"))
            assignment_name = _taxonomy_assignment_name(
                name_mapper,
                level_key=level_key,
                assignment=assignment,
            )
            rows.append(
                {
                    "cell_id": cell_id,
                    "level_key": str(level_key),
                    "level": level_display,
                    "level_token": level_token,
                    "assignment": assignment,
                    "assignment_name": assignment_name,
                    "bootstrapping_probability": bootstrap,
                    "aggregate_probability": _numeric_or_nan(
                        metrics.get("aggregate_probability")
                    ),
                    "avg_correlation": _numeric_or_nan(metrics.get("avg_correlation")),
                    "directly_assigned": _bool_or_nan(metrics.get("directly_assigned")),
                    "runner_up_probability": runner_up_probability,
                    "runner_up_margin": bootstrap - runner_up_probability,
                }
            )

    if not rows:
        return _empty_extended_qc_frame()
    return pd.DataFrame(rows)


def read_mapmycells_csv(csv_path: Path | str) -> pd.DataFrame:
    """Read the comment-prefixed MapMyCells CSV output."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, comment="#", converters={0: str})
    if df.empty:
        raise ValueError(f"MapMyCells CSV is empty: {csv_path}")
    first_col = str(df.columns[0])
    df[first_col] = df[first_col].astype(str)
    return df


def _plot_mapmycells_assignment_qc_if_available(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    level: str,
    column_prefix: str,
) -> Path | None:
    try:
        return plot_mapmycells_assignment_qc(
            adata,
            output_path,
            level=level,
            column_prefix=column_prefix,
        )
    except KeyError:
        logger.info("Skipping MapMyCells %s QC plot; level is unavailable.", level)
        return None


def _plot_mapmycells_spatial_supercluster_grid_if_available(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    column_prefix: str,
) -> Path | None:
    try:
        return plot_mapmycells_spatial_supercluster_grid(
            adata,
            output_path,
            column_prefix=column_prefix,
        )
    except KeyError as exc:
        logger.info("Skipping MapMyCells supercluster spatial grid: %s", exc)
        return None


def _plot_mapmycells_umap_clusters_by_supercluster_if_available(
    adata: ad.AnnData,
    output_dir: Path | str,
    *,
    column_prefix: str,
) -> dict[str, Path]:
    try:
        return plot_mapmycells_umap_clusters_by_supercluster(
            adata,
            output_dir,
            column_prefix=column_prefix,
        )
    except KeyError as exc:
        logger.info("Skipping MapMyCells UMAP cluster-by-supercluster plots: %s", exc)
        return {}


def _mapmycells_level_columns(
    adata: ad.AnnData,
    *,
    column_prefix: str,
) -> dict[str, dict[str, str]]:
    levels: dict[str, dict[str, str]] = {}
    for column in adata.obs.columns:
        column_name = str(column)
        if not column_name.startswith(column_prefix):
            continue
        suffix_part = column_name[len(column_prefix) :]
        for field in MAPMYCELLS_LEVEL_FIELD_SUFFIXES:
            field_suffix = f"_{field}"
            if not suffix_part.endswith(field_suffix):
                continue
            level = suffix_part[: -len(field_suffix)]
            if level:
                levels.setdefault(level, {})[field] = column_name
            break
    return levels


def _resolve_mapmycells_level_columns(
    adata: ad.AnnData,
    *,
    level: str,
    column_prefix: str,
) -> tuple[str, dict[str, str]]:
    levels = _mapmycells_level_columns(adata, column_prefix=column_prefix)
    for candidate in _matching_level_names(levels, level):
        columns = levels[candidate]
        if "name" in columns or "label" in columns:
            return candidate, columns
    raise KeyError(
        f"No MapMyCells {level!r} assignment columns with prefix "
        f"{column_prefix!r} were found in adata.obs."
    )


def _matching_level_names(
    levels: dict[str, dict[str, str]] | list[str] | tuple[str, ...],
    requested: str,
) -> list[str]:
    names = list(levels.keys()) if isinstance(levels, dict) else list(levels)
    requested_token = _normalize_level_token(requested)
    synonyms = {
        "supercluster": {"supercluster", "super_cluster", "supc"},
        "cluster": {"cluster", "clus"},
    }
    wanted = synonyms.get(requested_token, {requested_token})
    exact = [name for name in names if _normalize_level_token(name) in wanted]
    if exact:
        return exact

    matches: list[str] = []
    for name in names:
        token = _normalize_level_token(name)
        is_supercluster_match = (
            requested_token == "supercluster" and "supercluster" in token
        )
        is_cluster_match = (
            requested_token == "cluster"
            and (
                token.endswith("cluster")
                or token.endswith("clus")
                or token.startswith("clus_")
            )
            and "supercluster" not in token
            and "subcluster" not in token
        )
        if is_supercluster_match or is_cluster_match:
            matches.append(name)
    return matches


def _mapmycells_level_label_series(
    adata: ad.AnnData,
    columns: dict[str, str],
) -> pd.Series:
    label_column = columns.get("name") or columns.get("label")
    if label_column is None:
        raise KeyError("No MapMyCells label/name column was available.")
    labels = pd.Series(adata.obs[label_column], index=adata.obs_names).astype("string")
    labels = labels.fillna("unassigned")
    labels = labels.mask(
        labels.str.strip().eq("") | labels.str.lower().isin({"nan", "none"}),
        "unassigned",
    )
    return labels


def _mapmycells_level_confidence_series(
    adata: ad.AnnData,
    columns: dict[str, str],
) -> tuple[pd.Series, str]:
    for field, label in (
        ("bootstrapping_probability", "Bootstrapping probability"),
        ("correlation_coefficient", "Correlation coefficient"),
    ):
        column = columns.get(field)
        if column is not None:
            values = pd.to_numeric(adata.obs[column], errors="coerce")
            return pd.Series(values, index=adata.obs_names), label
    empty = pd.Series(np.full(adata.n_obs, np.nan), index=adata.obs_names)
    return empty, "Confidence"


def _plot_assignment_count_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    categories: list[str],
    confidence_label: str,
) -> None:
    counts = plot_df["assignment"].value_counts().reindex(categories).fillna(0)
    medians = plot_df.groupby("assignment")["confidence"].median().reindex(categories)
    y_positions = np.arange(len(categories), dtype=float)
    if medians.notna().any():
        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=0.0, vmax=1.0)
        colors = cmap(norm(medians.fillna(0.0).clip(0.0, 1.0).to_numpy()))
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(
            mappable,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label=f"Median {confidence_label.lower()}",
        )
    else:
        colors = "#4c78a8"
    ax.barh(y_positions, counts.to_numpy(dtype=float), color=colors)
    ax.set_title("Assigned cells")
    ax.set_xlabel("Cell count")


def _plot_assignment_confidence_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    categories: list[str],
    y_positions: np.ndarray,
    confidence_label: str,
) -> None:
    has_values = False
    for y_position, category in zip(y_positions, categories, strict=True):
        values = (
            pd.to_numeric(
                plot_df.loc[plot_df["assignment"] == category, "confidence"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        if values.size == 0:
            continue
        has_values = True
        q10, q25, q50, q75, q90 = np.nanquantile(
            values,
            [0.10, 0.25, 0.50, 0.75, 0.90],
        )
        ax.plot([q10, q90], [y_position, y_position], color="#4d4d4d", linewidth=1)
        ax.barh(
            y_position,
            q75 - q25,
            left=q25,
            height=0.55,
            color="#9ecae1",
            edgecolor="#2b5c7a",
            linewidth=0.5,
        )
        ax.scatter([q50], [y_position], s=12, color="#08306b", zorder=3)
    if not has_values:
        _plot_empty_axis(ax, "No confidence values available.")
        return
    ax.set_title("Confidence distribution")
    ax.set_xlabel(confidence_label)
    if "probability" in confidence_label.lower():
        ax.set_xlim(0.0, 1.02)


def _plot_assignment_low_confidence_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    categories: list[str],
    confidence_label: str,
) -> None:
    confidence = pd.to_numeric(plot_df["confidence"], errors="coerce")
    if confidence.notna().sum() == 0:
        _plot_empty_axis(ax, "No confidence values available.")
        return
    threshold = 0.7 if "probability" in confidence_label.lower() else 0.3
    work = plot_df.assign(is_low=confidence < threshold)
    low_fraction = (
        work.groupby("assignment")["is_low"].mean().reindex(categories).fillna(0.0)
    )
    y_positions = np.arange(len(categories), dtype=float)
    ax.barh(y_positions, 100.0 * low_fraction.to_numpy(dtype=float), color="#e15759")
    ax.set_title(f"Below {threshold:g}")
    ax.set_xlabel("Cells below threshold (%)")
    ax.set_xlim(0.0, 100.0)


def _select_extended_qc_levels(
    qc: pd.DataFrame,
    *,
    levels: tuple[str, ...],
) -> list[tuple[str, str]]:
    if qc.empty or "level_token" not in qc.columns:
        return []
    available = [
        str(token)
        for token in dict.fromkeys(qc["level_token"].dropna().astype(str).tolist())
    ]
    selected: list[str] = []
    for level in levels:
        for match in _matching_level_names(available, level):
            token = _normalize_level_token(match)
            if token not in selected:
                selected.append(token)
                break
    if not selected:
        selected = available[: len(levels)]

    display_by_token = (
        qc.dropna(subset=["level_token", "level"])
        .drop_duplicates("level_token")
        .set_index("level_token")["level"]
        .astype(str)
        .to_dict()
    )
    return [
        (token, _format_level_display(display_by_token.get(token, token)))
        for token in selected
    ]


def _mapmycells_cell_ids_for_obs(
    adata: ad.AnnData,
    *,
    column_prefix: str,
) -> np.ndarray:
    cell_id_column = f"{column_prefix}cell_id"
    if cell_id_column in adata.obs:
        cell_ids = pd.Series(adata.obs[cell_id_column], index=adata.obs_names).astype(
            "string"
        )
        fallback = pd.Series(adata.obs_names.astype(str), index=adata.obs_names)
        cell_ids = cell_ids.mask(
            cell_ids.isna()
            | cell_ids.str.strip().eq("")
            | cell_ids.str.lower().isin({"nan", "none"}),
            fallback,
        )
        return cast(np.ndarray, cell_ids.astype(str).to_numpy())
    return cast(np.ndarray, adata.obs_names.astype(str).to_numpy())


def _mapmycells_cell_complexity(adata: ad.AnnData) -> tuple[str, np.ndarray]:
    for column, label in (
        ("n_genes_by_counts", "Genes detected"),
        ("total_counts", "Total counts"),
        ("transcript_counts", "Transcript counts"),
    ):
        if column in adata.obs:
            values = pd.to_numeric(adata.obs[column], errors="coerce").to_numpy(float)
            return label, values
    matrix = adata.layers.get("counts", adata.X)
    return "Non-zero genes", _matrix_nonzero_counts(matrix)


def _matrix_nonzero_counts(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return cast(np.ndarray, np.asarray(matrix.getnnz(axis=1), dtype=float))
    array = np.asarray(matrix)
    return cast(np.ndarray, np.count_nonzero(array > 0, axis=1).astype(float))


def _plot_scatter_metric(
    ax: plt.Axes,
    x_values: pd.Series | np.ndarray,
    y_values: pd.Series | np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    point_size: float,
    alpha: float,
    x_log: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    x = pd.to_numeric(pd.Series(x_values), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y_values), errors="coerce").to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if x_log:
        finite &= x > 0
    if not finite.any():
        _plot_empty_axis(ax, "No finite values available.")
        ax.set_title(title)
        return
    ax.scatter(
        x[finite],
        y[finite],
        s=float(point_size),
        alpha=float(alpha),
        color="#4c78a8",
        linewidths=0,
        rasterized=True,
    )
    if x_log:
        ax.set_xscale("log")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#e5e5e5", linewidth=0.5)


def _plot_hist_metric(
    ax: plt.Axes,
    values: pd.Series | np.ndarray,
    *,
    xlabel: str,
    title: str,
    x_limits: tuple[float, float] | None = None,
) -> None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        _plot_empty_axis(ax, "No finite values available.")
        ax.set_title(title)
        return
    ax.hist(finite, bins=40, color="#59a14f", edgecolor="white", linewidth=0.25)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cells")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)


def _plot_empty_axis(ax: plt.Axes, message: str) -> None:
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


def _empty_extended_qc_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cell_id",
            "level_key",
            "level",
            "level_token",
            "assignment",
            "assignment_name",
            "bootstrapping_probability",
            "aggregate_probability",
            "avg_correlation",
            "directly_assigned",
            "runner_up_probability",
            "runner_up_margin",
        ]
    )


def _taxonomy_assignment_name(
    name_mapper: dict[Any, Any],
    *,
    level_key: str,
    assignment: str,
) -> str:
    level_mapper = name_mapper.get(level_key)
    if not isinstance(level_mapper, dict):
        return assignment
    assignment_mapper = level_mapper.get(assignment)
    if (
        isinstance(assignment_mapper, dict)
        and assignment_mapper.get("name") is not None
    ):
        return str(assignment_mapper["name"])
    return assignment


def _numeric_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first_numeric(value: Any) -> float:
    if isinstance(value, list) and value:
        return _numeric_or_nan(value[0])
    return _numeric_or_nan(value)


def _bool_or_nan(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    return _numeric_or_nan(value)


def _normalize_level_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return token.strip("_")


def _format_level_display(value: str) -> str:
    return _normalize_level_token(value).replace("_", " ").title()


def _axis_padding(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    span = float(finite.max() - finite.min())
    return 0.02 * span if span > 0 else 1.0


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    padding = _axis_padding(finite)
    return (float(finite.min() - padding), float(finite.max() + padding))


def _unique_plot_token(value: str, *, used: set[str]) -> str:
    try:
        token = _sanitize_token(value)
    except ValueError:
        token = "unlabeled"
    candidate = token
    suffix = 2
    while candidate in used:
        candidate = f"{token}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _wrapped_title(value: str, *, width: int = 34) -> str:
    wrapped = textwrap.wrap(str(value), width=width)
    return "\n".join(wrapped) if wrapped else str(value)


def prepare_region_mapmycells_reference(
    config: MapMyCellsConfig,
) -> RegionReferenceArtifacts:
    """Download/cache and generate a strict WHB or WMB ROI reference."""
    region_name = _sanitize_token(config.region_name)
    reference_prefix = "region" if config.reference_atlas == "whb" else "wmb_region"
    reference_dir = (
        config.region_cache_dir / "references" / f"{reference_prefix}_{region_name}"
    )
    precompute_dir = reference_dir / "precompute"
    reference_marker_dir = reference_dir / "reference_markers"
    query_marker_dir = reference_dir / "query_markers"
    manifest_path = reference_dir / "region_reference_manifest.json"
    region_cell_metadata_path = reference_dir / "region_cell_metadata.csv"
    precomputed_stats_path = precompute_dir / "precomputed_stats.h5"
    query_marker_path = (
        query_marker_dir
        / f"query_markers.n{config.region_query_markers_n_per_utility}.json"
    )
    expected_config = _region_reference_config_payload(config, region_name)

    if (
        not config.region_force_rebuild
        and precomputed_stats_path.exists()
        and query_marker_path.exists()
        and manifest_path.exists()
    ):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config") == expected_config:
            return RegionReferenceArtifacts(
                marker_lookup_path=query_marker_path,
                precomputed_stats_path=precomputed_stats_path,
                manifest_path=manifest_path,
                manifest=manifest,
            )

    if config.region_force_rebuild and reference_dir.exists():
        shutil.rmtree(reference_dir)
    elif reference_dir.exists():
        for stale_dir in (precompute_dir, reference_marker_dir, query_marker_dir):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
    for directory in (precompute_dir, reference_marker_dir, query_marker_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if config.reference_atlas == "whb":
        reference_inputs = _ensure_whb_reference_inputs(
            config.region_cache_dir,
            force_download=False,
        )
        region_column = "region_of_interest_label"
        roi_map_path = reference_inputs["region_of_interest_structure_map"]
        hierarchy = WHB_HIERARCHY
    else:
        reference_inputs = _ensure_wmb_reference_metadata_inputs(
            config.region_cache_dir,
            force_download=False,
        )
        region_column = WMB_REGION_COLUMN
        roi_map_path = reference_inputs["region_of_interest_metadata"]
        hierarchy = WMB_HIERARCHY
    filtered_summary = _write_region_cell_metadata(
        cell_metadata_path=reference_inputs["cell_metadata"],
        output_path=region_cell_metadata_path,
        region_labels=config.region_labels,
        min_cells_per_leaf=config.region_min_cells_per_leaf,
        roi_map_path=roi_map_path,
        region_column=region_column,
    )
    if config.reference_atlas == "whb":
        h5ad_path_list = [
            reference_inputs["WHB-10Xv3-Neurons_raw"],
            reference_inputs["WHB-10Xv3-Nonneurons_raw"],
        ]
    else:
        matrix_labels = filtered_summary.get("feature_matrix_labels", [])
        expression_inputs = _ensure_wmb_expression_inputs(
            config.region_cache_dir,
            matrix_labels=cast(list[str], matrix_labels),
            force_download=False,
        )
        reference_inputs.update(expression_inputs)
        h5ad_path_list = list(expression_inputs.values())

    scratch_dir = _reference_scratch_dir(config, reference_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    _run_precomputation_abc(
        {
            "output_path": str(precomputed_stats_path),
            "hierarchy": hierarchy,
            "h5ad_path_list": [str(path) for path in h5ad_path_list],
            "cell_metadata_path": str(region_cell_metadata_path),
            "cluster_annotation_path": str(reference_inputs["cluster_annotation_term"]),
            "cluster_membership_path": str(
                reference_inputs["cluster_to_cluster_annotation_membership"]
            ),
            "n_processors": config.n_processors,
            "split_by_dataset": False,
            "do_pruning": True,
            "tmp_dir": str(scratch_dir),
            "clobber": True,
            "normalization": "raw",
        }
    )

    reference_config: dict[str, Any] = {
        "precomputed_path_list": [str(precomputed_stats_path)],
        "output_dir": str(reference_marker_dir),
        "tmp_dir": str(scratch_dir),
        "n_processors": config.n_processors,
        "clobber": True,
    }
    if config.max_gb is not None:
        reference_config["max_gb"] = int(config.max_gb)
    if config.drop_level is not None:
        reference_config["drop_level"] = config.drop_level
    _run_reference_markers(reference_config)

    reference_marker_path_list = [
        str(path) for path in sorted(reference_marker_dir.iterdir()) if path.is_file()
    ]
    if not reference_marker_path_list:
        raise RuntimeError(
            f"No reference marker files were generated in {reference_marker_dir}"
        )
    if query_marker_path.exists():
        query_marker_path.unlink()
    query_config: dict[str, Any] = {
        "output_path": str(query_marker_path),
        "reference_marker_path_list": reference_marker_path_list,
        "n_processors": config.n_processors,
        "tmp_dir": str(scratch_dir),
        "n_per_utility": config.region_query_markers_n_per_utility,
        "search_for_stats_file": False,
    }
    if config.drop_level is not None:
        query_config["drop_level"] = config.drop_level
    _run_query_markers(query_config)

    manifest = {
        "reference_type": "region",
        "config": expected_config,
        "manifest_url": WHB_MANIFEST_URL,
        "region_cell_metadata_path": str(region_cell_metadata_path),
        "precomputed_stats_path": str(precomputed_stats_path),
        "reference_marker_path_list": reference_marker_path_list,
        "marker_lookup_path": str(query_marker_path),
        "filtering_summary": filtered_summary,
        "input_paths": {key: str(value) for key, value in reference_inputs.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return RegionReferenceArtifacts(
        marker_lookup_path=query_marker_path,
        precomputed_stats_path=precomputed_stats_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _region_reference_config_payload(
    config: MapMyCellsConfig,
    region_name: str,
) -> dict[str, Any]:
    return {
        "region_name": region_name,
        "region_labels": list(config.region_labels),
        "region_min_cells_per_leaf": config.region_min_cells_per_leaf,
        "region_query_markers_n_per_utility": (
            config.region_query_markers_n_per_utility
        ),
        "reference_atlas": config.reference_atlas,
        "query_species": config.query_species,
        "hierarchy": (
            WHB_HIERARCHY if config.reference_atlas == "whb" else WMB_HIERARCHY
        ),
        "normalization": "raw",
        "manifest_url": WHB_MANIFEST_URL,
        "drop_level": config.drop_level,
    }


def _reference_scratch_dir(config: MapMyCellsConfig, reference_dir: Path) -> Path:
    if config.tmp_dir is not None:
        return config.tmp_dir / "mapmycells_region_reference"
    return reference_dir / "scratch"


def _ensure_whb_reference_inputs(
    cache_dir: Path,
    *,
    force_download: bool = False,
) -> dict[str, Path]:
    manifest = _load_abc_manifest()
    abc_cache_dir = cache_dir / "abc_whb"
    inputs: dict[str, Path] = {}
    for file_key in (
        "cell_metadata",
        "region_of_interest_structure_map",
        "anatomical_division_structure_map",
    ):
        info = _manifest_file_info(
            manifest,
            WHB_DATASET_DIRECTORY,
            "metadata",
            file_key,
            "files",
            "csv",
        )
        inputs[file_key] = _ensure_manifest_file(
            info,
            abc_cache_dir,
            force_download=force_download,
        )

    for file_key in (
        "cluster",
        "cluster_annotation_term",
        "cluster_annotation_term_set",
        "cluster_to_cluster_annotation_membership",
    ):
        info = _manifest_file_info(
            manifest,
            WHB_TAXONOMY_DIRECTORY,
            "metadata",
            file_key,
            "files",
            "csv",
        )
        inputs[file_key] = _ensure_manifest_file(
            info,
            abc_cache_dir,
            force_download=force_download,
        )

    for matrix_name in WHB_EXPRESSION_MATRIX_NAMES:
        info = _manifest_file_info(
            manifest,
            WHB_DATASET_DIRECTORY,
            "expression_matrices",
            matrix_name,
            "raw",
            "files",
            "h5ad",
        )
        inputs[f"{matrix_name}_raw"] = _ensure_manifest_file(
            info,
            abc_cache_dir,
            force_download=force_download,
        )
    return inputs


def _ensure_wmb_reference_metadata_inputs(
    cache_dir: Path,
    *,
    force_download: bool = False,
) -> dict[str, Path]:
    manifest = _load_abc_manifest()
    abc_cache_dir = cache_dir / "abc_atlas"
    inputs: dict[str, Path] = {}
    for file_key in ("cell_metadata", "region_of_interest_metadata"):
        info = _manifest_file_info(
            manifest,
            WMB_DATASET_DIRECTORY,
            "metadata",
            file_key,
            "files",
            "csv",
        )
        inputs[file_key] = _ensure_manifest_file(
            info,
            abc_cache_dir,
            force_download=force_download,
        )
    for file_key in (
        "cluster_annotation_term",
        "cluster_to_cluster_annotation_membership",
    ):
        info = _manifest_file_info(
            manifest,
            WMB_TAXONOMY_DIRECTORY,
            "metadata",
            file_key,
            "files",
            "csv",
        )
        inputs[file_key] = _ensure_manifest_file(
            info,
            abc_cache_dir,
            force_download=force_download,
        )
    return inputs


def _ensure_wmb_expression_inputs(
    cache_dir: Path,
    *,
    matrix_labels: list[str],
    force_download: bool = False,
) -> dict[str, Path]:
    if not matrix_labels:
        raise ValueError(
            "The filtered WMB metadata did not contain any feature_matrix_label "
            "values, so expression matrices could not be selected."
        )
    manifest = _load_abc_manifest()
    abc_cache_dir = cache_dir / "abc_atlas"
    inputs: dict[str, Path] = {}
    for matrix_label in sorted(set(matrix_labels)):
        match: tuple[str, dict[str, Any]] | None = None
        for directory in WMB_EXPRESSION_DIRECTORIES:
            matrices = manifest["file_listing"][directory]["expression_matrices"]
            if matrix_label in matrices:
                match = (directory, dict(matrices[matrix_label]))
                break
        if match is None:
            raise KeyError(
                f"WMB feature_matrix_label {matrix_label!r} was not found in "
                f"{WMB_EXPRESSION_DIRECTORIES!r}."
            )
        directory, _ = match
        info = _manifest_file_info(
            manifest,
            directory,
            "expression_matrices",
            matrix_label,
            "raw",
            "files",
            "h5ad",
        )
        inputs[f"{matrix_label}_raw"] = _ensure_manifest_file(
            info,
            abc_cache_dir,
            force_download=force_download,
        )
    return inputs


def _load_abc_manifest() -> dict[str, Any]:
    with urllib.request.urlopen(WHB_MANIFEST_URL) as response:
        manifest: object = json.loads(response.read().decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Allen WHB manifest is not a JSON object: {WHB_MANIFEST_URL}")
    return cast(dict[str, Any], manifest)


def _manifest_file_info(manifest: dict[str, Any], *keys: str) -> dict[str, Any]:
    node: Any = manifest["file_listing"]
    for key in keys:
        node = node[key]
    return dict(node)


def _ensure_manifest_file(
    info: dict[str, Any],
    cache_dir: Path,
    *,
    force_download: bool = False,
) -> Path:
    relative_path = Path(str(info["relative_path"]))
    output_path = cache_dir / relative_path
    expected_size = int(info.get("size", 0) or 0)
    if (
        output_path.exists()
        and not force_download
        and _file_size_matches(output_path, expected_size)
    ):
        return output_path

    return _ensure_url_file(
        str(info["url"]),
        output_path,
        expected_size=expected_size,
        force_download=force_download,
    )


def _ensure_url_file(
    url: str,
    output_path: Path,
    *,
    expected_size: int = 0,
    force_download: bool = False,
) -> Path:
    if (
        output_path.exists()
        and not force_download
        and _file_size_matches(output_path, expected_size)
    ):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if (
            output_path.exists()
            and not force_download
            and _file_size_matches(output_path, expected_size)
        ):
            return output_path
        if force_download:
            tmp_path.unlink(missing_ok=True)

        offset = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        logger.info(
            "Downloading Allen reference file: %s (resume_offset=%d)", url, offset
        )
        with urllib.request.urlopen(request) as response:
            is_partial = getattr(response, "status", None) == 206
            mode = "ab" if offset and is_partial else "wb"
            with tmp_path.open(mode) as out:
                while True:
                    chunk = response.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    out.write(chunk)
        actual_size = tmp_path.stat().st_size
        if expected_size and actual_size != expected_size:
            if actual_size > expected_size:
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded {url} to {tmp_path}, but size was "
                f"{actual_size} bytes; expected {expected_size} bytes. "
                "The partial file was retained for a resumed download."
            )
        tmp_path.replace(output_path)
    return output_path


def _file_size_matches(path: Path, expected_size: int) -> bool:
    return expected_size <= 0 or path.stat().st_size == expected_size


def _write_region_cell_metadata(
    *,
    cell_metadata_path: Path,
    output_path: Path,
    region_labels: list[str],
    min_cells_per_leaf: int,
    roi_map_path: Path,
    region_column: str = "region_of_interest_label",
) -> dict[str, Any]:
    if not region_labels:
        raise ValueError("At least one region label is required for region mapping.")
    cell_metadata = pd.read_csv(cell_metadata_path)
    if region_column not in cell_metadata.columns:
        raise KeyError(
            f"{cell_metadata_path} does not contain {region_column}. "
            f"Available columns: {list(cell_metadata.columns)}"
        )
    if "cluster_alias" not in cell_metadata.columns:
        raise KeyError(
            f"{cell_metadata_path} does not contain cluster_alias. "
            f"Available columns: {list(cell_metadata.columns)}"
        )
    region_mask = cell_metadata[region_column].isin(region_labels)
    region_cells = cell_metadata.loc[region_mask].copy()
    if region_cells.empty:
        available = _available_roi_labels(roi_map_path, region_column=region_column)
        raise ValueError(
            f"No {region_column} cells matched mapmycells_region_labels="
            f"{region_labels!r}. Available ROI labels include: {available[:20]}"
        )

    leaf_counts = region_cells["cluster_alias"].value_counts()
    valid_leaf_aliases = leaf_counts[leaf_counts >= min_cells_per_leaf].index
    filtered = region_cells[
        region_cells["cluster_alias"].isin(valid_leaf_aliases)
    ].copy()
    if filtered.empty:
        raise ValueError(
            "No WHB cells remained after applying "
            f"region_min_cells_per_leaf={min_cells_per_leaf} to "
            f"region_labels={region_labels!r}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    dropped_counts = leaf_counts[leaf_counts < min_cells_per_leaf]
    return {
        "requested_region_labels": list(region_labels),
        "region_column": region_column,
        "matched_region_labels": sorted(
            str(value) for value in region_cells[region_column].unique()
        ),
        "feature_matrix_labels": sorted(
            str(value)
            for value in filtered.get(WMB_FEATURE_MATRIX_COLUMN, pd.Series(dtype=str))
            .dropna()
            .unique()
        ),
        "n_cells_before_region_filter": int(len(cell_metadata)),
        "n_cells_after_region_filter": int(len(region_cells)),
        "n_cells_after_min_leaf_filter": int(len(filtered)),
        "n_leaf_aliases_before_min_leaf_filter": int(len(leaf_counts)),
        "n_leaf_aliases_after_min_leaf_filter": int(len(valid_leaf_aliases)),
        "min_cells_per_leaf": int(min_cells_per_leaf),
        "dropped_leaf_aliases": {
            str(alias): int(count) for alias, count in dropped_counts.items()
        },
    }


def _available_roi_labels(
    roi_map_path: Path,
    *,
    region_column: str = "region_of_interest_label",
) -> list[str]:
    roi_map = pd.read_csv(roi_map_path)
    if region_column not in roi_map.columns:
        return []
    return sorted(str(value) for value in roi_map[region_column].unique())


def _run_precomputation_abc(config: dict[str, Any]) -> None:
    from cell_type_mapper.cli.precompute_stats_abc import PrecomputationABCRunner

    runner = PrecomputationABCRunner(args=[], input_data=config)
    runner.run()


def _run_reference_markers(config: dict[str, Any]) -> None:
    from cell_type_mapper.cli.reference_markers import ReferenceMarkerRunner

    runner = ReferenceMarkerRunner(args=[], input_data=config)
    runner.run()


def _run_query_markers(config: dict[str, Any]) -> None:
    from cell_type_mapper.cli.query_markers import QueryMarkerRunner

    runner = QueryMarkerRunner(args=[], input_data=config)
    runner.run()


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    token = token.strip("_")
    if not token:
        raise ValueError(f"Could not create a safe token from {value!r}")
    return token


def _copy_matrix(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        return matrix.copy()
    return np.array(matrix, copy=True)


def _plot_mapmycells_embedding(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    basis: str,
    color: str,
    title: str,
    column_prefix: str,
    point_size: float,
    alpha: float,
    dpi: int,
) -> Path:
    if basis not in adata.obsm:
        raise KeyError(f"Expected adata.obsm[{basis!r}] for MapMyCells plot.")
    if color not in adata.obs:
        raise KeyError(f"Expected adata.obs[{color!r}] for MapMyCells plot.")

    output_path = prepare_plot_output(output_path)
    coords = np.asarray(adata.obsm[basis])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"Expected adata.obsm[{basis!r}] to have at least two columns; "
            f"found shape {coords.shape}."
        )

    labels = pd.Series(adata.obs[color].astype("string"), index=adata.obs_names)
    labels = labels.fillna("unassigned")
    label_counts = labels.value_counts()
    categories = [str(label) for label in label_counts.index]
    categorical = pd.Categorical(labels.astype(str), categories=categories)
    codes = categorical.codes
    n_categories = len(categories)
    cmap = _categorical_cmap(n_categories)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        cmap=cmap,
        s=float(point_size),
        alpha=float(alpha),
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(f"{title}\ncolored by {color.replace(column_prefix, '')}")
    ax.set_xlabel(f"{basis} 1")
    ax.set_ylabel(f"{basis} 2")
    ax.set_aspect("equal" if basis == "spatial" else "auto")
    if 0 < n_categories <= MAPMYCELLS_MAX_LEGEND_CATEGORIES:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=cmap(idx),
                markeredgewidth=0,
                markersize=4,
                label=label,
            )
            for idx, label in enumerate(categories)
        ]
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
            fontsize=5,
            title=f"{n_categories} labels",
            title_fontsize=6,
        )
    else:
        ax.text(
            0.02,
            0.98,
            f"{n_categories} labels",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "none",
                "alpha": 0.8,
            },
        )
    fig.tight_layout()
    save_figure(fig, output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output_path


def _categorical_cmap(n_categories: int) -> ListedColormap:
    if n_categories <= 0:
        return ListedColormap(["#bdbdbd"])
    base = plt.get_cmap("turbo", n_categories)
    return ListedColormap([base(i) for i in range(n_categories)])


def _path_as_str(path: Path | str | None) -> str | None:
    return None if path is None else str(path)


def _read_text_if_present(path: Path | str | None) -> str:
    if path is None:
        return ""
    resolved = Path(path)
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8", errors="replace")


def _read_comment_header(path: Path | str) -> list[str]:
    comments: list[str] = []
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            comments.append(line.rstrip("\n"))
    return comments


def _index_from_column_with_fallback(
    df: pd.DataFrame,
    *,
    column: str,
    fallback: pd.Index,
) -> pd.Index:
    values = df[column].astype(str)
    fallback_values = fallback.astype(str)
    cleaned = values.mask(
        values.str.strip().eq("") | values.str.lower().isin({"nan", "none"}),
        fallback_values,
    )
    return pd.Index(cleaned.astype(str), name=None)


def _run_command(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    logger.info("Running MapMyCells command: %s", " ".join(command))
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with (
            stdout_path.open("w") as stdout_handle,
            stderr_path.open("w") as stderr_handle,
        ):
            stdout_handle.write("$ " + " ".join(command) + "\n\n")
            stdout_handle.flush()
            completed = subprocess.run(
                command,
                check=False,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not start MapMyCells. Install the Allen Institute "
            "cell_type_mapper package in the active environment."
        ) from exc

    if completed.returncode == 0:
        return

    message = [
        f"MapMyCells failed with exit code {completed.returncode}",
        f"stdout log: {stdout_path}",
        f"stderr log: {stderr_path}",
    ]
    stdout_tail = _tail_text(stdout_path)
    stderr_tail = _tail_text(stderr_path)
    if stdout_tail:
        message.extend(["stdout tail:", stdout_tail])
    if stderr_tail:
        message.extend(["stderr tail:", stderr_tail])
    raise RuntimeError("\n".join(message))


def _tail_text(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _require_existing_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _write_command_manifest(path: Path, command: list[str]) -> None:
    path.write_text(json.dumps({"command": command}, indent=2) + "\n")


def _write_results_manifest(
    path: Path,
    config: MapMyCellsConfig,
    references: list[MapMyCellsReference],
    results: dict[str, dict[str, dict[str, Path]]],
) -> None:
    payload = {
        "pair_id": config.pair_id,
        "reference_mode": config.reference_mode,
        "reference_atlas": config.reference_atlas,
        "query_species": config.query_species,
        "auto_download_references": config.auto_download_references,
        "marker_lookup_path": _path_as_str(config.marker_lookup_path),
        "precomputed_stats_path": _path_as_str(config.precomputed_stats_path),
        "gene_mapping_db_path": _path_as_str(config.gene_mapping_db_path),
        "region_name": config.region_name,
        "region_labels": list(config.region_labels),
        "region_cache_dir": str(config.region_cache_dir),
        "region_min_cells_per_leaf": config.region_min_cells_per_leaf,
        "region_query_markers_n_per_utility": (
            config.region_query_markers_n_per_utility
        ),
        "bootstrap_factor": config.bootstrap_factor,
        "bootstrap_iteration": config.bootstrap_iteration,
        "plots_only": bool(config.plots_only),
        "n_processors": config.n_processors,
        "references": {reference.name: reference.manifest for reference in references},
        "samples": {
            sample_id: {
                reference_name: {key: str(value) for key, value in paths.items()}
                for reference_name, paths in reference_results.items()
            }
            for sample_id, reference_results in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"
