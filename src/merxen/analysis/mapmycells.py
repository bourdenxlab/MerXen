"""Local MapMyCells annotation for clustered MerXen AnnData outputs."""

from __future__ import annotations

import json
import logging
import subprocess
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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from scipy import sparse

from merxen.config import MapMyCellsConfig
from merxen.memory import force_release, log_status

logger = logging.getLogger(__name__)

MAPMYCELLS_PREFIX = "mapmycells_"
MAPMYCELLS_ASSIGNMENT_COLOR_CANDIDATES = (
    "mapmycells_subcluster_name",
    "mapmycells_cluster_name",
    "mapmycells_supercluster_name",
    "mapmycells_class_name",
    "mapmycells_subclass_name",
    "mapmycells_type_name",
    "mapmycells_cell_type",
)
MAPMYCELLS_MAX_LEGEND_CATEGORIES = 64


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


def run_mapmycells(config: MapMyCellsConfig) -> dict[str, dict[str, Path]]:
    """Run local MapMyCells assignment for every sample in a pair.

    Args:
        config: Validated MapMyCells stage configuration.

    Returns:
        Mapping from sample ID to output artifact paths.
    """
    _require_existing_file(config.marker_lookup_path, "MapMyCells marker lookup")
    _require_existing_file(
        config.precomputed_stats_path, "MapMyCells precomputed stats"
    )
    if config.tmp_dir is not None:
        config.tmp_dir.mkdir(parents=True, exist_ok=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Path]] = {}

    for sample in config.samples:
        log_status(
            f"[{sample.sample_id}] Starting MapMyCells "
            f"(platform={sample.platform}, bootstrap_factor={config.bootstrap_factor})"
        )
        sample_dir = config.output_dir / sample.platform.lower()
        sample_dir.mkdir(parents=True, exist_ok=True)
        _require_existing_file(
            sample.anndata_path, f"clustered AnnData for {sample.sample_id}"
        )

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
        )

        results[sample.sample_id] = {
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
        force_release(note=f"after MapMyCells {sample.sample_id}")

    manifest_path = config.output_dir / f"{config.pair_id}_mapmycells_manifest.json"
    _write_results_manifest(manifest_path, config, results)
    return results


def build_mapmycells_command(
    config: MapMyCellsConfig,
    *,
    query_h5ad: Path,
    extended_json: Path,
    csv_path: Path,
    log_path: Path,
) -> list[str]:
    """Build the ``cell_type_mapper`` command-line invocation."""
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
        str(config.marker_lookup_path),
        "--precomputed_stats.path",
        str(config.precomputed_stats_path),
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
            target = f"{MAPMYCELLS_PREFIX}{column}"
            adata.obs[target] = aligned[column].to_numpy()
            assignment_columns.append(target)

        matched = int(aligned.iloc[:, 0].notna().sum()) if not aligned.empty else 0
        plot_column = choose_mapmycells_assignment_column(adata)
        plot_paths: dict[str, str] = {}
        if umap_plot_path is not None:
            plot_paths["umap"] = str(
                plot_mapmycells_umap(
                    adata,
                    umap_plot_path,
                    color=plot_column,
                )
            )
        if spatial_plot_path is not None:
            plot_paths["spatial"] = str(
                plot_mapmycells_spatial(
                    adata,
                    spatial_plot_path,
                    color=plot_column,
                )
            )

        adata.uns["merxen_mapmycells"] = {
            "csv_path": str(csv_path),
            "csv_header_comments": _read_comment_header(csv_path),
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
        }
        adata.write_h5ad(output_h5ad)
    finally:
        del adata
        force_release(note=f"after annotating MapMyCells output {input_h5ad.name}")

    return output_h5ad


def choose_mapmycells_assignment_column(
    adata: ad.AnnData,
    *,
    max_categories: int = MAPMYCELLS_MAX_LEGEND_CATEGORIES,
) -> str:
    """Choose the most specific MapMyCells label column that remains plottable."""
    preferred = [
        column
        for column in MAPMYCELLS_ASSIGNMENT_COLOR_CANDIDATES
        if column in adata.obs
    ]
    name_columns = [
        str(column)
        for column in adata.obs.columns
        if str(column).startswith(MAPMYCELLS_PREFIX) and str(column).endswith("_name")
    ]
    label_columns = [
        str(column)
        for column in adata.obs.columns
        if str(column).startswith(MAPMYCELLS_PREFIX) and str(column).endswith("_label")
    ]
    candidates = list(dict.fromkeys([*preferred, *name_columns, *label_columns]))
    if not candidates:
        raise KeyError("No MapMyCells assignment columns were found in adata.obs.")

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
    point_size: float = 1.0,
    alpha: float = 0.65,
    dpi: int = 180,
) -> Path:
    """Plot the existing clustering UMAP colored by MapMyCells assignments."""
    color = color or choose_mapmycells_assignment_column(adata)
    return _plot_mapmycells_embedding(
        adata,
        output_path,
        basis="X_umap",
        color=color,
        title="MapMyCells assignment UMAP",
        point_size=point_size,
        alpha=alpha,
        dpi=dpi,
    )


def plot_mapmycells_spatial(
    adata: ad.AnnData,
    output_path: Path | str,
    *,
    color: str | None = None,
    point_size: float = 0.25,
    alpha: float = 0.65,
    dpi: int = 180,
) -> Path:
    """Plot spatial coordinates colored by MapMyCells assignments."""
    color = color or choose_mapmycells_assignment_column(adata)
    return _plot_mapmycells_embedding(
        adata,
        output_path,
        basis="spatial",
        color=color,
        title="MapMyCells assignment spatial plot",
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
    ax.set_title(f"{title}\ncolored by {color.replace(MAPMYCELLS_PREFIX, '')}")
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
    results: dict[str, dict[str, Path]],
) -> None:
    payload = {
        "pair_id": config.pair_id,
        "marker_lookup_path": str(config.marker_lookup_path),
        "precomputed_stats_path": str(config.precomputed_stats_path),
        "bootstrap_factor": config.bootstrap_factor,
        "bootstrap_iteration": config.bootstrap_iteration,
        "n_processors": config.n_processors,
        "samples": {
            sample_id: {key: str(value) for key, value in paths.items()}
            for sample_id, paths in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"
