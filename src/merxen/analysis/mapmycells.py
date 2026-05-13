"""Local MapMyCells annotation for clustered MerXen AnnData outputs."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import urllib.request
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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy import sparse

from merxen.config import MapMyCellsConfig
from merxen.memory import force_release, log_status

logger = logging.getLogger(__name__)

MAPMYCELLS_PREFIX = "mapmycells_"
MAPMYCELLS_REGION_PREFIX_TEMPLATE = "mapmycells_region_{region_name}_"
MAPMYCELLS_UNS_KEY = "merxen_mapmycells"
MAPMYCELLS_REGION_UNS_KEY_TEMPLATE = "merxen_mapmycells_region_{region_name}"
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
            f"bootstrap_factor={config.bootstrap_factor})"
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
    if config.reference_mode in {"whole_brain", "both"}:
        if config.marker_lookup_path is None or config.precomputed_stats_path is None:
            raise ValueError(
                "Whole-brain MapMyCells requested but marker/stat paths are missing."
            )
        _require_existing_file(config.marker_lookup_path, "MapMyCells marker lookup")
        _require_existing_file(
            config.precomputed_stats_path, "MapMyCells precomputed stats"
        )
        references.append(
            MapMyCellsReference(
                name="whole_brain",
                output_dir=config.output_dir,
                marker_lookup_path=config.marker_lookup_path,
                precomputed_stats_path=config.precomputed_stats_path,
                column_prefix=MAPMYCELLS_PREFIX,
                uns_key=MAPMYCELLS_UNS_KEY,
                manifest={
                    "reference_type": "whole_brain",
                    "marker_lookup_path": str(config.marker_lookup_path),
                    "precomputed_stats_path": str(config.precomputed_stats_path),
                },
            )
        )
    if config.reference_mode in {"region", "both"}:
        region_name = _sanitize_token(config.region_name)
        region_artifacts = prepare_region_mapmycells_reference(config)
        references.append(
            MapMyCellsReference(
                name=f"region_{region_name}",
                output_dir=config.output_dir / f"region_{region_name}",
                marker_lookup_path=region_artifacts.marker_lookup_path,
                precomputed_stats_path=region_artifacts.precomputed_stats_path,
                column_prefix=MAPMYCELLS_REGION_PREFIX_TEMPLATE.format(
                    region_name=region_name
                ),
                uns_key=MAPMYCELLS_REGION_UNS_KEY_TEMPLATE.format(
                    region_name=region_name
                ),
                manifest=region_artifacts.manifest,
            )
        )
    return references


def _run_mapmycells_reference(
    config: MapMyCellsConfig,
    *,
    sample: Any,
    reference: MapMyCellsReference,
) -> dict[str, Path]:
    sample_dir = reference.output_dir / sample.platform.lower()
    sample_dir.mkdir(parents=True, exist_ok=True)

    query_h5ad = prepare_mapmycells_query(
        sample.anndata_path,
        sample_dir / f"{sample.sample_id}_mapmycells_query.h5ad",
        query_layer=sample.query_layer,
        gene_id_column=sample.gene_id_column,
        obs_id_column=sample.obs_id_column,
    )
    extended_json = sample_dir / f"{sample.sample_id}_mapmycells_extended.json"
    csv_path = sample_dir / f"{sample.sample_id}_mapmycells.csv"
    log_path = sample_dir / f"{sample.sample_id}_mapmycells.log"
    stdout_path = sample_dir / f"{sample.sample_id}_mapmycells_stdout.log"
    stderr_path = sample_dir / f"{sample.sample_id}_mapmycells_stderr.log"
    command_path = sample_dir / f"{sample.sample_id}_mapmycells_command.json"
    annotated_h5ad = sample_dir / f"{sample.sample_id}_mapmycells_annotated.h5ad"
    umap_plot = sample_dir / f"{sample.sample_id}_mapmycells_umap.png"
    spatial_plot = sample_dir / f"{sample.sample_id}_mapmycells_spatial.png"

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
        column_prefix=reference.column_prefix,
        uns_key=reference.uns_key,
        reference_metadata=reference.manifest,
    )

    return {
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
        plot_paths: dict[str, str] = {}
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


def read_mapmycells_csv(csv_path: Path | str) -> pd.DataFrame:
    """Read the comment-prefixed MapMyCells CSV output."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, comment="#", converters={0: str})
    if df.empty:
        raise ValueError(f"MapMyCells CSV is empty: {csv_path}")
    first_col = str(df.columns[0])
    df[first_col] = df[first_col].astype(str)
    return df


def prepare_region_mapmycells_reference(
    config: MapMyCellsConfig,
) -> RegionReferenceArtifacts:
    """Download/cache and generate a strict WHB ROI-specific MapMyCells reference."""
    region_name = _sanitize_token(config.region_name)
    reference_dir = config.region_cache_dir / "references" / f"region_{region_name}"
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

    reference_inputs = _ensure_whb_reference_inputs(
        config.region_cache_dir,
        force_download=False,
    )
    filtered_summary = _write_region_cell_metadata(
        cell_metadata_path=reference_inputs["cell_metadata"],
        output_path=region_cell_metadata_path,
        region_labels=config.region_labels,
        min_cells_per_leaf=config.region_min_cells_per_leaf,
        roi_map_path=reference_inputs["region_of_interest_structure_map"],
    )

    scratch_dir = _reference_scratch_dir(config, reference_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    _run_precomputation_abc(
        {
            "output_path": str(precomputed_stats_path),
            "hierarchy": WHB_HIERARCHY,
            "h5ad_path_list": [
                str(reference_inputs["WHB-10Xv3-Neurons_raw"]),
                str(reference_inputs["WHB-10Xv3-Nonneurons_raw"]),
            ],
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
        "hierarchy": WHB_HIERARCHY,
        "normalization": "raw",
        "manifest_url": WHB_MANIFEST_URL,
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


def _load_abc_manifest() -> dict[str, Any]:
    with urllib.request.urlopen(WHB_MANIFEST_URL) as response:
        return json.loads(response.read().decode("utf-8"))


def _manifest_file_info(manifest: dict[str, Any], *keys: str) -> dict[str, Any]:
    node: Any = manifest["file_listing"]
    for key in keys:
        node = node[key]
    return dict(node)


def _ensure_manifest_file(
    info: dict[str, Any],
    cache_dir: Path,
    *,
    force_download: bool,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    url = str(info["url"])
    logger.info("Downloading Allen WHB reference file: %s", url)
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024 * 16)
            if not chunk:
                break
            out.write(chunk)
    actual_size = tmp_path.stat().st_size
    if expected_size and actual_size != expected_size:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {url} to {tmp_path}, but size was "
            f"{actual_size} bytes; expected {expected_size} bytes."
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
) -> dict[str, Any]:
    if not region_labels:
        raise ValueError("At least one region label is required for region mapping.")
    cell_metadata = pd.read_csv(cell_metadata_path)
    if "region_of_interest_label" not in cell_metadata.columns:
        raise KeyError(
            f"{cell_metadata_path} does not contain region_of_interest_label. "
            f"Available columns: {list(cell_metadata.columns)}"
        )
    if "cluster_alias" not in cell_metadata.columns:
        raise KeyError(
            f"{cell_metadata_path} does not contain cluster_alias. "
            f"Available columns: {list(cell_metadata.columns)}"
        )
    region_mask = cell_metadata["region_of_interest_label"].isin(region_labels)
    region_cells = cell_metadata.loc[region_mask].copy()
    if region_cells.empty:
        available = _available_roi_labels(roi_map_path)
        raise ValueError(
            "No WHB cells matched mapmycells_region_labels="
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
        "matched_region_labels": sorted(
            str(value) for value in region_cells["region_of_interest_label"].unique()
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


def _available_roi_labels(roi_map_path: Path) -> list[str]:
    roi_map = pd.read_csv(roi_map_path)
    if "region_of_interest_label" not in roi_map.columns:
        return []
    return sorted(str(value) for value in roi_map["region_of_interest_label"].unique())


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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
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
        "marker_lookup_path": _path_as_str(config.marker_lookup_path),
        "precomputed_stats_path": _path_as_str(config.precomputed_stats_path),
        "region_name": config.region_name,
        "region_labels": list(config.region_labels),
        "region_cache_dir": str(config.region_cache_dir),
        "region_min_cells_per_leaf": config.region_min_cells_per_leaf,
        "region_query_markers_n_per_utility": (
            config.region_query_markers_n_per_utility
        ),
        "bootstrap_factor": config.bootstrap_factor,
        "bootstrap_iteration": config.bootstrap_iteration,
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
