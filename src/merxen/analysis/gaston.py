"""GASTON terminal analysis with native SpatialData coordinates.

The module deliberately separates SpatialData-aware preparation/import from the
pinned GASTON runtime. The files exchanged between those environments are
plain NumPy/SciPy/Parquet/JSON artifacts and are therefore portable across the
Dwight workstation, Apptainer, and scheduler-managed GPU nodes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import resource
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from merxen.config import GastonConfig
from merxen.memory import force_release

logger = logging.getLogger(__name__)

GASTON_COMMIT = "79577c45111b3442b808c8e620711b822965493b"
GASTON_VERSION = "0.0.2"
GLMPCA_CONVERGENCE_TOLERANCE = 1.0e-4
# ``glmpca`` itself defaults to 1,000 iterations. The MerXen setting retains
# the requested tutorial-scale initial budget, but whole acquisitions are
# allowed to continue to the library default so a healthy, slowly converging
# fit is not rejected after 30 iterations.
GLMPCA_LIBRARY_DEFAULT_MAX_ITERATIONS = 1_000
GASTON_OWNED_COLUMNS = (
    "gaston_domain",
    "gaston_isodepth",
    "gaston_model_seed",
    "gaston_num_domains",
)
CONTROL_TOKENS = (
    "blank",
    "control",
    "negative",
    "negcontrol",
    "unassigned",
    "deprecated",
)
ISODEPTH_ORIENTATION_NOTE = (
    "GASTON isodepth direction is unanchored and arbitrary; values are raw and "
    "have not been reversed, scaled, or anatomically oriented."
)


@dataclass(frozen=True)
class SeedResult:
    """Validated summary for one GASTON neural-network restart."""

    seed: int
    minimum_loss: float
    output_dir: Path
    status: str


def prepare_gaston_input(
    config: GastonConfig,
    output_dir: Path | str,
) -> Path:
    """Create a validated native-coordinate bundle for one clustered sample.

    Args:
        config: One sample/platform/segmentation GASTON configuration.
        output_dir: Directory in which the portable input bundle is written.

    Returns:
        Path to ``input_manifest.json``.

    Raises:
        ValueError: If identities, coordinates, or count data violate the
            GASTON input contract.
        KeyError: If required AnnData layers or SpatialData elements are absent.
    """
    import spatialdata as sd

    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clustered = ad.read_h5ad(config.clustered_h5ad_path)
    sdata_obj = sd.read_zarr(config.latest_zarr_path)
    try:
        if config.clustered_table_key not in sdata_obj.tables:
            raise KeyError(
                f"Missing clustered SpatialData table {config.clustered_table_key!r} "
                f"in {config.latest_zarr_path}"
            )
        if config.shape_key not in sdata_obj.shapes:
            raise KeyError(
                f"Missing native SpatialData shape {config.shape_key!r} in "
                f"{config.latest_zarr_path}"
            )

        spatial_table = sdata_obj.tables[config.clustered_table_key]
        table_attrs = dict(spatial_table.uns.get("spatialdata_attrs", {}))
        instance_key = table_attrs.get("instance_key")
        if not isinstance(instance_key, str) or not instance_key:
            raise ValueError(
                f"SpatialData table {config.clustered_table_key!r} has no "
                "instance_key metadata"
            )
        table_ids = _identifiers_from_obs(
            spatial_table,
            instance_key,
            label=f"SpatialData table {config.clustered_table_key!r}",
        )
        cell_ids = _identifiers_from_obs(
            clustered,
            instance_key,
            label=f"clustered H5AD {config.clustered_h5ad_path}",
        )
        _require_same_identifiers(
            expected=cell_ids,
            observed=table_ids,
            expected_label="clustered H5AD",
            observed_label=f"SpatialData table {config.clustered_table_key!r}",
        )

        native_shapes = sdata_obj.shapes[config.shape_key]
        if hasattr(native_shapes, "compute"):
            native_shapes = native_shapes.compute()
        shape_ids = _identifiers_from_shapes(
            native_shapes,
            instance_key,
            shape_key=config.shape_key,
        )
        coordinates = _native_centroids_in_order(
            native_shapes,
            shape_ids=shape_ids,
            cell_ids=cell_ids,
            shape_key=config.shape_key,
        )
    finally:
        del sdata_obj
        force_release(note=f"after GASTON native input read {config.sample_id}")

    counts, gene_names, gene_indices = _eligible_counts(
        clustered,
        max_genes=config.max_genes,
    )
    n_control_genes = int(np.count_nonzero(_control_feature_mask(clustered)))
    n_eligible_genes = int(clustered.n_vars - n_control_genes)
    _validate_count_matrix(counts)
    connectivity = spatial_connectivity_diagnostic(coordinates)
    cell_totals = np.asarray(counts.sum(axis=1)).ravel()
    gene_totals = np.asarray(counts.sum(axis=0)).ravel()
    genes_per_cell = np.asarray(counts.getnnz(axis=1)).ravel()
    cells_per_gene = np.asarray(counts.getnnz(axis=0)).ravel()
    nearest_distances = cKDTree(coordinates).query(
        coordinates,
        k=min(2, len(coordinates)),
    )[0]
    nearest_distances = (
        np.asarray(nearest_distances[:, 1], dtype=float)
        if len(coordinates) > 1
        else np.asarray([0.0])
    )
    if connectivity["n_substantial_components"] > 1:
        logger.warning(
            "[%s:%s] Native coordinates contain %d substantial spatial "
            "components; continuing with the complete acquisition.",
            config.sample_id,
            config.segmentation,
            connectivity["n_substantial_components"],
        )

    counts_path = output_dir / "counts.npz"
    coordinates_path = output_dir / "coordinates.npy"
    cell_ids_path = output_dir / "cell_ids.tsv"
    gene_names_path = output_dir / "gene_names.tsv"
    sparse.save_npz(counts_path, counts, compressed=True)
    np.save(coordinates_path, coordinates, allow_pickle=False)
    pd.DataFrame({"cell_id": cell_ids}).to_csv(cell_ids_path, sep="\t", index=False)
    pd.DataFrame({"gene_name": gene_names}).to_csv(
        gene_names_path,
        sep="\t",
        index=False,
    )

    bundle_files = [counts_path, coordinates_path, cell_ids_path, gene_names_path]
    manifest = {
        "schema_version": 1,
        "pair_id": config.pair_id,
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "clustered_h5ad_path": str(Path(config.clustered_h5ad_path).resolve()),
        "latest_zarr_path": str(Path(config.latest_zarr_path).resolve()),
        "source_table_key": config.source_table_key,
        "clustered_table_key": config.clustered_table_key,
        "shape_key": config.shape_key,
        "instance_key": instance_key,
        "coordinate_source": "native_shape_centroids",
        "coordinates_are_native": True,
        "aligned_coordinate_elements_used": [],
        "isodepth_orientation": ISODEPTH_ORIENTATION_NOTE,
        "n_cells": int(counts.shape[0]),
        "n_genes": int(counts.shape[1]),
        "n_nonzero_counts": int(counts.nnz),
        "total_counts": int(counts.sum()),
        "max_genes": int(config.max_genes),
        "n_genes_before_filter": int(clustered.n_vars),
        "n_control_genes_excluded": n_control_genes,
        "n_eligible_genes_before_maximum": n_eligible_genes,
        "n_eligible_genes_truncated": int(n_eligible_genes - counts.shape[1]),
        "selected_gene_indices": [int(value) for value in gene_indices],
        "coordinate_bounds": {
            "x_min": float(coordinates[:, 0].min()),
            "x_max": float(coordinates[:, 0].max()),
            "y_min": float(coordinates[:, 1].min()),
            "y_max": float(coordinates[:, 1].max()),
        },
        "spatial_connectivity": connectivity,
        "qc": {
            "cell_total_counts": _numeric_summary(cell_totals),
            "genes_detected_per_cell": _numeric_summary(genes_per_cell),
            "gene_total_counts": _numeric_summary(gene_totals),
            "cells_detected_per_gene": _numeric_summary(cells_per_gene),
            "native_nearest_neighbor_distance": _numeric_summary(nearest_distances),
        },
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_rss_bytes": _peak_rss_bytes(),
        "files": {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in bundle_files
        },
    }
    manifest_path = output_dir / "input_manifest.json"
    _write_json(manifest_path, manifest)
    del clustered
    force_release(note=f"after GASTON input preparation {config.sample_id}")
    return manifest_path


def spatial_connectivity_diagnostic(coordinates: np.ndarray) -> dict[str, Any]:
    """Summarize disconnected pieces without filtering any cells.

    A six-nearest-neighbor graph is pruned at a robust distance threshold. The
    graph is diagnostic only: disconnected or substantial components generate a
    warning in preparation but never change the exported acquisition.

    Args:
        coordinates: Finite ``N x 2`` native centroid array.

    Returns:
        JSON-compatible component counts, sizes, and distance threshold.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape N x 2")
    n_cells = int(coordinates.shape[0])
    if n_cells == 0:
        raise ValueError("Cannot diagnose an empty coordinate matrix")
    if n_cells == 1:
        return {
            "method": "pruned_6_nearest_neighbor_graph",
            "distance_threshold": None,
            "n_components": 1,
            "component_sizes": [1],
            "substantial_component_min_cells": 1,
            "n_substantial_components": 1,
        }

    n_neighbors = min(7, n_cells)
    distances, neighbors = cKDTree(coordinates).query(coordinates, k=n_neighbors)
    nearest = np.asarray(distances[:, 1], dtype=float)
    finite_nearest = nearest[np.isfinite(nearest)]
    if finite_nearest.size == 0:
        raise ValueError("Native coordinates have no finite nearest-neighbor distances")
    positive = finite_nearest[finite_nearest > 0]
    baseline = float(np.median(positive)) if positive.size else 1.0
    threshold = max(
        float(np.quantile(finite_nearest, 0.99)) * 3.0,
        baseline * 10.0,
        np.finfo(float).eps,
    )
    rows: list[int] = []
    cols: list[int] = []
    for row_index in range(n_cells):
        for distance, col_index in zip(
            distances[row_index, 1:],
            neighbors[row_index, 1:],
            strict=True,
        ):
            if np.isfinite(distance) and float(distance) <= threshold:
                rows.extend((row_index, int(col_index)))
                cols.extend((int(col_index), row_index))
    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(n_cells, n_cells),
    ).tocsr()
    n_components, component_labels = csgraph.connected_components(
        graph,
        directed=False,
    )
    sizes = np.bincount(component_labels, minlength=n_components)
    sorted_sizes = sorted((int(size) for size in sizes), reverse=True)
    substantial_minimum = max(100, int(math.ceil(n_cells * 0.01)))
    if n_cells < 100:
        substantial_minimum = max(1, int(math.ceil(n_cells * 0.1)))
    return {
        "method": "pruned_6_nearest_neighbor_graph",
        "distance_threshold": float(threshold),
        "n_components": int(n_components),
        "component_sizes": sorted_sizes,
        "substantial_component_min_cells": int(substantial_minimum),
        "n_substantial_components": int(np.count_nonzero(sizes >= substantial_minimum)),
    }


def run_gaston_glmpca(
    config: GastonConfig,
    bundle_dir: Path | str,
    output_dir: Path | str,
) -> Path:
    """Compute tutorial-compatible Poisson GLM-PCA features on CPU.

    Args:
        config: GASTON configuration.
        bundle_dir: Portable input bundle directory.
        output_dir: Directory for GLM-PCA features and convergence metadata.

    Returns:
        Path to ``glmpca_features.npy``.

    Raises:
        RuntimeError: If GLM-PCA reaches the effective convergence ceiling
            without convergence or produces non-finite features/deviance.
    """
    from glmpca import glmpca

    started = time.monotonic()
    bundle_dir = Path(bundle_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = sparse.load_npz(bundle_dir / "counts.npz").tocsr()
    _validate_count_matrix(counts)
    max_dimensions = min(counts.shape[0] - 1, counts.shape[1] - 1)
    if config.glmpca_dimensions > max_dimensions:
        raise ValueError(
            "GASTON GLM-PCA dimensions must be smaller than both matrix axes: "
            f"dimensions={config.glmpca_dimensions}, shape={counts.shape}"
        )
    dense_gene_by_cell = np.asarray(counts.transpose().toarray())
    # The inspected GLM-PCA implementation uses NumPy's global generator for
    # its small random initialization and has no seed argument. Each process is
    # isolated, so fixing it here makes the portable feature bundle reproducible.
    np.random.seed(0)
    configured_iterations = int(config.glmpca_iterations)
    maximum_iterations = max(
        configured_iterations,
        GLMPCA_LIBRARY_DEFAULT_MAX_ITERATIONS,
    )
    if maximum_iterations > configured_iterations:
        print(
            "GASTON GLM-PCA: extending the configured iteration budget "
            f"from {configured_iterations} to the {maximum_iterations}-iteration "
            "convergence ceiling.",
            flush=True,
        )
    result = glmpca.glmpca(
        dense_gene_by_cell,
        int(config.glmpca_dimensions),
        fam="poi",
        ctl={
            "maxIter": maximum_iterations,
            "eps": GLMPCA_CONVERGENCE_TOLERANCE,
            "optimizeTheta": True,
        },
        penalty=float(config.glmpca_penalty),
        verbose=True,
    )
    features = np.asarray(result["factors"], dtype=float)
    deviance = np.asarray(result.get("dev", []), dtype=float)
    if features.shape != (counts.shape[0], config.glmpca_dimensions):
        raise RuntimeError(
            "GLM-PCA returned an unexpected factor shape: "
            f"{features.shape}, expected "
            f"{(counts.shape[0], config.glmpca_dimensions)}"
        )
    if not np.isfinite(features).all() or not np.isfinite(deviance).all():
        raise RuntimeError("GLM-PCA produced NaN or infinite output")
    relative_changes = np.abs(np.diff(deviance)) / (0.1 + np.abs(deviance[:-1]))
    # glmpca checks this condition only from iteration index 5 onward. Derive
    # convergence from the returned trajectory instead of using its length:
    # convergence can occur on the final allowed iteration, for which
    # ``len(dev) == maxIter`` is otherwise ambiguous.
    converged = bool(
        len(deviance) >= 6
        and relative_changes.size > 0
        and relative_changes[-1] < GLMPCA_CONVERGENCE_TOLERANCE
    )
    final_relative_change = (
        float(relative_changes[-1]) if relative_changes.size else None
    )
    metadata = {
        "schema_version": 1,
        "dimensions": int(config.glmpca_dimensions),
        "penalty": float(config.glmpca_penalty),
        "configured_iterations": configured_iterations,
        "maximum_iterations": maximum_iterations,
        "iteration_budget_auto_extended": bool(
            maximum_iterations > configured_iterations
        ),
        "convergence_tolerance": GLMPCA_CONVERGENCE_TOLERANCE,
        "random_seed": 0,
        "iterations_completed": int(len(deviance)),
        "converged": bool(converged),
        "termination_reason": (
            "relative_deviance_tolerance" if converged else "maximum_iterations"
        ),
        "final_relative_deviance_change": final_relative_change,
        "deviance_monotonic_nonincreasing": bool(np.all(np.diff(deviance) <= 0.0)),
        "deviance": [float(value) for value in deviance],
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    metadata_path = output_dir / "glmpca_manifest.json"
    _write_json(metadata_path, metadata)
    if not converged:
        raise RuntimeError(
            "GLM-PCA did not meet the relative-deviance convergence tolerance "
            f"{GLMPCA_CONVERGENCE_TOLERANCE:g} within {maximum_iterations} "
            "effective iterations "
            f"(configured initial budget={configured_iterations}, "
            f"final relative change={final_relative_change}); diagnostics: "
            f"{metadata_path}"
        )
    features_path = output_dir / "glmpca_features.npy"
    np.save(features_path, features, allow_pickle=False)
    metadata["features_sha256"] = sha256_file(features_path)
    _write_json(metadata_path, metadata)
    return features_path


def run_gaston_training(
    config: GastonConfig,
    bundle_dir: Path | str,
    glmpca_dir: Path | str,
    seed: int,
    output_dir: Path | str,
) -> Path:
    """Train one resumable GASTON neural-network restart.

    Args:
        config: GASTON configuration.
        bundle_dir: Portable input directory containing native coordinates.
        glmpca_dir: Directory containing ``glmpca_features.npy``.
        seed: Restart seed.
        output_dir: Seed-specific output directory.

    Returns:
        Path to the seed manifest.
    """
    import torch
    from gaston import neural_net

    if seed < 0 or seed >= config.n_restarts:
        raise ValueError(f"Seed {seed} is outside 0..{config.n_restarts - 1}")
    device = "cuda" if config.use_gpu else "cpu"
    if config.use_gpu and not torch.cuda.is_available():
        raise RuntimeError("GASTON_TRAIN requested CUDA but torch.cuda is unavailable")

    started = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = np.load(Path(bundle_dir) / "coordinates.npy", allow_pickle=False)
    features = np.load(
        Path(glmpca_dir) / "glmpca_features.npy",
        allow_pickle=False,
    )
    if not np.isfinite(coordinates).all() or not np.isfinite(features).all():
        raise ValueError("GASTON training input contains non-finite values")
    spatial_tensor, expression_tensor = neural_net.load_rescale_input_data(
        coordinates,
        features,
    )
    _model, losses = neural_net.train(
        spatial_tensor,
        expression_tensor,
        S_hidden_list=list(config.hidden_spatial),
        A_hidden_list=list(config.hidden_expression),
        epochs=int(config.epochs),
        checkpoint=int(config.checkpoint_interval),
        save_dir=str(output_dir),
        optim=config.optimizer,
        seed=int(seed),
        save_final=True,
        device=device,
    )
    losses = np.asarray(losses, dtype=float)
    finite_losses = losses[np.isfinite(losses)]
    if finite_losses.size == 0:
        raise RuntimeError(f"GASTON seed {seed} produced no finite losses")
    model_path = output_dir / "final_model.pt"
    if not model_path.exists():
        raise RuntimeError(f"GASTON seed {seed} did not write {model_path}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "seed": int(seed),
        "minimum_loss": float(finite_losses.min()),
        "final_loss": float(losses[-1]),
        "epochs": int(config.epochs),
        "checkpoint_interval": int(config.checkpoint_interval),
        "hidden_spatial": list(config.hidden_spatial),
        "hidden_expression": list(config.hidden_expression),
        "optimizer": config.optimizer,
        "device": device,
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if device == "cuda"
            else None
        ),
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_rss_bytes": _peak_rss_bytes(),
        "model_sha256": sha256_file(model_path),
    }
    manifest_path = output_dir / "seed_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def rank_seed_results(seed_dirs: list[Path | str]) -> tuple[pd.DataFrame, SeedResult]:
    """Rank completed restarts and select the finite minimum-loss model.

    Args:
        seed_dirs: Seed output directories containing optional manifests/models.

    Returns:
        A ranking table and the best valid seed result.

    Raises:
        RuntimeError: If no completed seed has both a finite loss and model.
    """
    rows: list[dict[str, Any]] = []
    valid: list[SeedResult] = []
    for raw_dir in seed_dirs:
        seed_dir = Path(raw_dir)
        manifest_path = seed_dir / "seed_manifest.json"
        status = "missing_manifest"
        seed = _seed_from_path(seed_dir)
        minimum_loss = math.nan
        model_exists = (seed_dir / "final_model.pt").exists()
        error: str | None = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                seed = int(manifest.get("seed", seed))
                status = str(manifest.get("status", "complete"))
                raw_loss = manifest.get("minimum_loss")
                minimum_loss = math.nan if raw_loss is None else float(raw_loss)
                error = (
                    str(manifest["error"])
                    if manifest.get("error") is not None
                    else None
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                status = "invalid_manifest"
                error = str(exc)
        is_valid = status == "complete" and np.isfinite(minimum_loss) and model_exists
        rows.append(
            {
                "seed": int(seed),
                "status": status,
                "minimum_loss": minimum_loss,
                "model_exists": bool(model_exists),
                "valid": bool(is_valid),
                "error": error,
                "seed_dir": str(seed_dir),
            }
        )
        if is_valid:
            valid.append(
                SeedResult(
                    seed=int(seed),
                    minimum_loss=float(minimum_loss),
                    output_dir=seed_dir,
                    status=status,
                )
            )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        ranking = pd.DataFrame(
            columns=[
                "seed",
                "status",
                "minimum_loss",
                "model_exists",
                "valid",
                "error",
                "seed_dir",
            ]
        )
    ranking = ranking.sort_values(
        ["valid", "minimum_loss", "seed"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    if not valid:
        raise RuntimeError("No GASTON restart produced a finite loss and final model")
    best = min(valid, key=lambda result: (result.minimum_loss, result.seed))
    return ranking, best


def select_num_domains(
    likelihoods: pd.DataFrame,
    *,
    domain_mode: str,
    num_domains: int | None,
    auto_k_fallback: int | None,
) -> tuple[int, str]:
    """Select fixed or Kneedle-derived domain count reproducibly.

    Args:
        likelihoods: Table with integer ``num_domains`` and numeric
            ``negative_log_likelihood`` columns.
        domain_mode: ``auto`` or ``fixed``.
        num_domains: Direct override; required in fixed mode.
        auto_k_fallback: Optional explicit fallback when Kneedle finds no knee.

    Returns:
        Selected domain count and selection-source label.

    Raises:
        RuntimeError: If automatic mode finds no knee and no fallback exists.
    """
    if num_domains is not None:
        return int(num_domains), "num_domains_override"
    if domain_mode == "fixed":
        raise ValueError("num_domains is required when domain_mode='fixed'")
    from kneed import KneeLocator

    x = likelihoods["num_domains"].to_numpy(dtype=int)
    y = likelihoods["negative_log_likelihood"].to_numpy(dtype=float)
    knee = KneeLocator(x, y, curve="convex", direction="decreasing").knee
    if knee is not None:
        return int(knee), "kneedle"
    if auto_k_fallback is not None:
        return int(auto_k_fallback), "auto_k_fallback"
    raise RuntimeError(
        "Automatic GASTON domain selection found no Kneedle knee and "
        "gaston_auto_k_fallback is not configured"
    )


def postprocess_gaston(
    config: GastonConfig,
    bundle_dir: Path | str,
    glmpca_dir: Path | str,
    seed_dirs: list[Path | str],
    output_dir: Path | str,
) -> Path:
    """Select the best restart, domains, annotations, models, and plots.

    Args:
        config: GASTON configuration.
        bundle_dir: Portable prepared input directory.
        glmpca_dir: GLM-PCA output directory.
        seed_dirs: All restart output directories.
        output_dir: Standalone per-platform GASTON result directory.

    Returns:
        Path to the portable per-cell Parquet annotations.
    """
    from gaston import dp_related, model_selection

    started = time.monotonic()
    bundle_dir = Path(bundle_dir)
    glmpca_dir = Path(glmpca_dir)
    output_dir = Path(output_dir)
    input_output_dir = output_dir / "input"
    model_dir = output_dir / "model"
    plots_dir = output_dir / "plots"
    input_output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for path in bundle_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, input_output_dir / path.name)

    ranking, best = rank_seed_results([Path(path) for path in seed_dirs])
    seed_losses_path = model_dir / "seed_losses.tsv"
    ranking.to_csv(seed_losses_path, sep="\t", index=False)
    model = _torch_load(best.output_dir / "final_model.pt", map_location="cpu")
    model = model.cpu()
    model.eval()
    scaled_expression = (
        _torch_load(
            best.output_dir / "Atorch.pt",
            map_location="cpu",
        )
        .detach()
        .cpu()
        .numpy()
    )
    scaled_coordinates = (
        _torch_load(
            best.output_dir / "Storch.pt",
            map_location="cpu",
        )
        .detach()
        .cpu()
        .numpy()
    )
    likelihood_values = np.asarray(
        model_selection.get_ll_list(
            model,
            scaled_expression,
            scaled_coordinates,
            num_buckets=int(config.domain_buckets),
            kmax=int(config.max_domains),
        ),
        dtype=float,
    )
    likelihoods = pd.DataFrame(
        {
            "num_domains": np.arange(1, len(likelihood_values) + 1, dtype=int),
            "negative_log_likelihood": likelihood_values,
        }
    )
    likelihoods = likelihoods[
        likelihoods["num_domains"].between(
            int(config.min_domains),
            int(config.max_domains),
        )
    ].reset_index(drop=True)
    likelihood_path = model_dir / "domain_likelihoods.tsv"
    likelihoods.to_csv(likelihood_path, sep="\t", index=False)
    if (
        likelihoods.empty
        or not np.isfinite(likelihoods["negative_log_likelihood"].to_numpy(float)).all()
    ):
        raise RuntimeError(
            f"GASTON domain likelihood curve is empty or non-finite: {likelihood_path}"
        )
    num_domains, selection_source = select_num_domains(
        likelihoods,
        domain_mode=config.domain_mode,
        num_domains=config.num_domains,
        auto_k_fallback=config.auto_k_fallback,
    )
    isodepth, labels = dp_related.get_isodepth_labels(
        model,
        scaled_expression,
        scaled_coordinates,
        int(num_domains),
        num_buckets=int(config.domain_buckets),
    )
    isodepth = np.asarray(isodepth, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if not np.isfinite(isodepth).all():
        raise RuntimeError("GASTON produced non-finite raw isodepth values")
    cell_ids = pd.read_csv(bundle_dir / "cell_ids.tsv", sep="\t")["cell_id"].astype(str)
    if len(cell_ids) != len(isodepth) or len(labels) != len(cell_ids):
        raise RuntimeError(
            "GASTON annotation length does not match the prepared cell identifiers"
        )
    annotations = pd.DataFrame(
        {
            "cell_id": cell_ids.to_numpy(),
            "gaston_domain": labels,
            "gaston_isodepth": isodepth,
            "gaston_model_seed": np.full(len(cell_ids), best.seed, dtype=np.int64),
            "gaston_num_domains": np.full(
                len(cell_ids),
                num_domains,
                dtype=np.int64,
            ),
        }
    )
    if annotations["cell_id"].duplicated().any():
        raise RuntimeError("GASTON postprocessing produced duplicate cell identifiers")
    cells_path = output_dir / f"{config.sample_id}_gaston_cells.parquet"
    annotations.to_parquet(cells_path, index=False)

    best_model_path = model_dir / "best_model.pt"
    shutil.copy2(best.output_dir / "final_model.pt", best_model_path)
    _copy_retained_seed_outputs(config, seed_dirs, best, model_dir)
    _copy_gpu_vram_logs(seed_dirs, model_dir)
    coordinates = np.load(bundle_dir / "coordinates.npy", allow_pickle=False)
    _write_annotation_plots(
        coordinates=coordinates,
        isodepth=isodepth,
        labels=labels,
        likelihoods=likelihoods,
        selected_num_domains=num_domains,
        seed_dirs=[Path(path) for path in seed_dirs],
        plots_dir=plots_dir,
        dpi=config.figure_dpi,
    )
    selection_manifest = {
        "schema_version": 1,
        "gaston_commit": GASTON_COMMIT,
        "gaston_version": GASTON_VERSION,
        "best_seed": int(best.seed),
        "best_minimum_loss": float(best.minimum_loss),
        "n_configured_restarts": int(config.n_restarts),
        "n_ranked_restarts": int(len(ranking)),
        "num_domains": int(num_domains),
        "domain_selection_source": selection_source,
        "domain_mode": config.domain_mode,
        "domain_search_min": int(config.min_domains),
        "domain_search_max": int(config.max_domains),
        "domain_buckets": int(config.domain_buckets),
        "isodepth_orientation": ISODEPTH_ORIENTATION_NOTE,
        "isodepth_transformed": False,
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_rss_bytes": _peak_rss_bytes(),
        "best_model_sha256": sha256_file(best_model_path),
        "annotations_sha256": sha256_file(cells_path),
        "glmpca_manifest": json.loads(
            (glmpca_dir / "glmpca_manifest.json").read_text()
        ),
        "configuration": _jsonable_config(config),
    }
    _write_json(model_dir / "model_selection.json", selection_manifest)
    return cells_path


def import_gaston_annotations(
    config: GastonConfig,
    standalone_dir: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Write an annotated H5AD and merge only GASTON-owned table columns.

    Args:
        config: GASTON configuration.
        standalone_dir: Completed standalone per-platform GASTON directory.
        output_dir: Final per-platform output directory.

    Returns:
        Paths to the annotated H5AD and SpatialData import manifest.
    """
    standalone_dir = Path(standalone_dir)
    output_dir = Path(output_dir)
    shutil.copytree(standalone_dir, output_dir, dirs_exist_ok=True)
    cells_path = output_dir / f"{config.sample_id}_gaston_cells.parquet"
    annotations = pd.read_parquet(cells_path)
    _validate_annotation_table(annotations)
    clustered = ad.read_h5ad(config.clustered_h5ad_path)
    instance_key = _instance_key_from_adata(
        clustered,
        label=f"clustered H5AD {config.clustered_h5ad_path}",
    )
    cell_ids = _identifiers_from_obs(
        clustered,
        instance_key,
        label=f"clustered H5AD {config.clustered_h5ad_path}",
    )
    ordered_annotations = _annotations_in_order(annotations, cell_ids)
    _apply_annotation_columns(clustered, ordered_annotations)
    model_selection = json.loads(
        (output_dir / "model" / "model_selection.json").read_text()
    )
    clustered.uns["merxen_gaston"] = {
        "schema_version": 1,
        "gaston_commit": GASTON_COMMIT,
        "gaston_version": GASTON_VERSION,
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "source_table_key": config.source_table_key,
        "clustered_table_key": config.clustered_table_key,
        "native_shape_key": config.shape_key,
        "native_coordinates": True,
        "model_seed": int(model_selection["best_seed"]),
        "num_domains": int(model_selection["num_domains"]),
        "isodepth_orientation": ISODEPTH_ORIENTATION_NOTE,
        "configuration": _jsonable_config(config),
    }
    annotated_h5ad = output_dir / f"{config.sample_id}_gaston_annotated.h5ad"
    clustered.write_h5ad(annotated_h5ad)

    import_manifest: dict[str, Any] = {
        "schema_version": 1,
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "latest_zarr_path": str(Path(config.latest_zarr_path).resolve()),
        "clustered_table_key": config.clustered_table_key,
        "owned_columns": list(GASTON_OWNED_COLUMNS),
        "n_annotations": int(len(annotations)),
        "write_spatialdata_table": bool(config.write_spatialdata_table),
        "status": "not_requested",
        "isodepth_orientation": ISODEPTH_ORIENTATION_NOTE,
        "annotated_h5ad_sha256": sha256_file(annotated_h5ad),
    }
    if config.write_spatialdata_table:
        _merge_gaston_into_spatialdata(config, annotations)
        import_manifest["status"] = "imported"
    import_manifest_path = output_dir / "spatialdata_import_manifest.json"
    _write_json(import_manifest_path, import_manifest)
    del clustered
    force_release(note=f"after GASTON import {config.sample_id}")
    return {
        "annotated_h5ad": annotated_h5ad,
        "cells": cells_path,
        "spatialdata_import_manifest": import_manifest_path,
    }


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_gaston_into_spatialdata(
    config: GastonConfig,
    annotations: pd.DataFrame,
) -> None:
    import spatialdata as sd
    from spatialdata.models import TableModel

    from merxen.io.spatialdata_io import (
        spatialdata_write_lock,
        write_or_replace_element,
    )

    with spatialdata_write_lock(config.latest_zarr_path):
        sdata_obj = sd.read_zarr(config.latest_zarr_path)
        try:
            if config.clustered_table_key not in sdata_obj.tables:
                raise KeyError(
                    f"Missing clustered SpatialData table "
                    f"{config.clustered_table_key!r} in {config.latest_zarr_path}"
                )
            source_table = sdata_obj.tables[config.clustered_table_key]
            attrs = dict(source_table.uns.get("spatialdata_attrs", {}))
            instance_key = attrs.get("instance_key")
            region_key = attrs.get("region_key")
            region = attrs.get("region")
            if (
                not isinstance(instance_key, str)
                or instance_key not in source_table.obs
            ):
                raise ValueError(
                    f"SpatialData table {config.clustered_table_key!r} has an "
                    "invalid instance_key"
                )
            if not isinstance(region_key, str) or region_key not in source_table.obs:
                raise ValueError(
                    f"SpatialData table {config.clustered_table_key!r} has an "
                    "invalid region_key"
                )
            if region is None:
                raise ValueError(
                    f"SpatialData table {config.clustered_table_key!r} has no region"
                )
            table_ids = _identifiers_from_obs(
                source_table,
                instance_key,
                label=f"SpatialData table {config.clustered_table_key!r}",
            )
            ordered = _annotations_in_order(annotations, table_ids)
            updated = source_table.copy()
            _apply_annotation_columns(updated, ordered)
            updated.uns["merxen_gaston"] = {
                "schema_version": 1,
                "gaston_commit": GASTON_COMMIT,
                "sample_id": config.sample_id,
                "segmentation": config.segmentation,
                "native_shape_key": config.shape_key,
                "native_coordinates": True,
                "isodepth_orientation": ISODEPTH_ORIENTATION_NOTE,
            }
            updated.uns.pop("spatialdata_attrs", None)
            parsed = TableModel.parse(
                updated,
                region=region,
                region_key=region_key,
                instance_key=instance_key,
            )
            write_or_replace_element(
                sdata_obj,
                config.clustered_table_key,
                "tables",
                parsed,
                overwrite=True,
            )
        finally:
            del sdata_obj
            force_release(note=f"after locked GASTON table import {config.sample_id}")


def _eligible_counts(
    clustered: ad.AnnData,
    *,
    max_genes: int,
) -> tuple[sparse.csr_matrix, list[str], list[int]]:
    if "counts" not in clustered.layers:
        raise KeyError(
            "Clustered H5AD is missing required raw-count layer layers['counts']"
        )
    control_mask = _control_feature_mask(clustered)
    eligible_indices = np.flatnonzero(~control_mask)
    if eligible_indices.size == 0:
        raise ValueError("No eligible non-control genes remain for GASTON")
    eligible_indices = eligible_indices[: int(max_genes)]
    matrix = clustered.layers["counts"][:, eligible_indices]
    counts = sparse.csr_matrix(matrix)
    counts.eliminate_zeros()
    gene_names = [str(clustered.var_names[index]) for index in eligible_indices]
    if pd.Index(gene_names).duplicated().any():
        duplicated = pd.Index(gene_names)[pd.Index(gene_names).duplicated()].unique()
        raise ValueError(f"GASTON gene names are duplicated: {list(duplicated[:5])}")
    return counts, gene_names, [int(index) for index in eligible_indices]


def _control_feature_mask(adata: ad.AnnData) -> np.ndarray:
    mask = np.zeros(adata.n_vars, dtype=bool)
    candidates = [pd.Series(adata.var_names.astype(str), index=adata.var_names)]
    for column in (
        "gene",
        "feature_name",
        "feature_types",
        "feature_type",
        "gene_ids",
    ):
        if column in adata.var:
            candidates.append(adata.var[column].astype(str))
    for values in candidates:
        lower = values.astype(str).str.lower()
        mask |= lower.apply(
            lambda value: any(token in value for token in CONTROL_TOKENS)
        ).to_numpy()
    return mask


def _validate_count_matrix(counts: sparse.spmatrix) -> None:
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError(f"GASTON count matrix is empty: shape={counts.shape}")
    values = np.asarray(counts.data)
    if not np.isfinite(values).all():
        raise ValueError("GASTON raw counts contain NaN or infinite values")
    if np.any(values < 0):
        raise ValueError("GASTON raw counts contain negative values")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("GASTON raw counts contain non-integer values")
    if counts.nnz == 0 or float(counts.sum()) <= 0:
        raise ValueError("GASTON count matrix contains no expression")


def _instance_key_from_adata(adata: ad.AnnData, *, label: str) -> str:
    attrs = dict(adata.uns.get("spatialdata_attrs", {}))
    instance_key = attrs.get("instance_key")
    if not isinstance(instance_key, str) or instance_key not in adata.obs:
        raise ValueError(f"{label} has no usable SpatialData instance_key")
    return instance_key


def _identifiers_from_obs(
    table: ad.AnnData,
    instance_key: str,
    *,
    label: str,
) -> pd.Index:
    if instance_key not in table.obs:
        raise ValueError(f"{label} lacks instance_key column {instance_key!r}")
    values = table.obs[instance_key]
    if values.isna().any():
        raise ValueError(f"{label} instance identifiers contain missing values")
    identifiers = pd.Index(values.astype(str), name="cell_id")
    _refuse_duplicate_ids(identifiers, label=label)
    return identifiers


def _identifiers_from_shapes(
    shapes: Any,
    instance_key: str,
    *,
    shape_key: str,
) -> pd.Index:
    if instance_key not in shapes.columns:
        raise ValueError(
            f"Native shape {shape_key!r} lacks SpatialData instance_key column "
            f"{instance_key!r}"
        )
    values = shapes[instance_key]
    if values.isna().any():
        raise ValueError(f"Native shape {shape_key!r} contains missing identifiers")
    identifiers = pd.Index(values.astype(str), name="cell_id")
    _refuse_duplicate_ids(identifiers, label=f"native shape {shape_key!r}")
    return identifiers


def _refuse_duplicate_ids(identifiers: pd.Index, *, label: str) -> None:
    duplicate_mask = identifiers.duplicated(keep=False)
    if duplicate_mask.any():
        examples = identifiers[duplicate_mask].unique()[:5].tolist()
        raise ValueError(f"{label} contains duplicate cell IDs: {examples}")


def _require_same_identifiers(
    *,
    expected: pd.Index,
    observed: pd.Index,
    expected_label: str,
    observed_label: str,
) -> None:
    expected_set = set(expected)
    observed_set = set(observed)
    missing = [value for value in expected if value not in observed_set]
    extra = [value for value in observed if value not in expected_set]
    if missing or extra:
        raise ValueError(
            f"Cell-ID mismatch between {expected_label} and {observed_label}: "
            f"missing={len(missing)} examples={missing[:5]}, "
            f"extra={len(extra)} examples={extra[:5]}"
        )


def _native_centroids_in_order(
    shapes: Any,
    *,
    shape_ids: pd.Index,
    cell_ids: pd.Index,
    shape_key: str,
) -> np.ndarray:
    missing = [cell_id for cell_id in cell_ids if cell_id not in set(shape_ids)]
    if missing:
        raise ValueError(
            f"Native shape {shape_key!r} is missing coordinate matches for "
            f"{len(missing)}/{len(cell_ids)} clustered cells: {missing[:5]}"
        )
    geometry = shapes.geometry
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Geometry is in a geographic CRS.*",
            category=UserWarning,
        )
        centroids = geometry.centroid
    centroid_table = pd.DataFrame(
        {
            "x": np.asarray(centroids.x, dtype=float),
            "y": np.asarray(centroids.y, dtype=float),
        },
        index=shape_ids,
    )
    coordinates = centroid_table.loc[cell_ids, ["x", "y"]].to_numpy(dtype=float)
    if coordinates.shape != (len(cell_ids), 2):
        raise RuntimeError("Native centroid join produced an unexpected shape")
    if not np.isfinite(coordinates).all():
        invalid = int(np.count_nonzero(~np.isfinite(coordinates).all(axis=1)))
        raise ValueError(
            f"Native shape {shape_key!r} produced non-finite coordinates for "
            f"{invalid} cells"
        )
    return coordinates


def _validate_annotation_table(annotations: pd.DataFrame) -> None:
    required = {"cell_id", *GASTON_OWNED_COLUMNS}
    missing = sorted(required.difference(annotations.columns))
    if missing:
        raise ValueError(f"GASTON annotations are missing columns: {missing}")
    if annotations.empty:
        raise ValueError("GASTON annotation table is empty")
    identifiers = pd.Index(annotations["cell_id"].astype(str))
    _refuse_duplicate_ids(identifiers, label="GASTON annotations")
    if not np.isfinite(annotations["gaston_isodepth"].to_numpy(float)).all():
        raise ValueError("GASTON annotations contain non-finite isodepth")


def _annotations_in_order(
    annotations: pd.DataFrame,
    cell_ids: pd.Index,
) -> pd.DataFrame:
    annotation_ids = pd.Index(annotations["cell_id"].astype(str))
    _require_same_identifiers(
        expected=cell_ids,
        observed=annotation_ids,
        expected_label="clustered cells",
        observed_label="GASTON annotations",
    )
    indexed = annotations.copy()
    indexed["cell_id"] = indexed["cell_id"].astype(str)
    indexed = indexed.set_index("cell_id", drop=False)
    return indexed.loc[cell_ids].reset_index(drop=True)


def _apply_annotation_columns(
    table: ad.AnnData,
    ordered_annotations: pd.DataFrame,
) -> None:
    domains = ordered_annotations["gaston_domain"].to_numpy(dtype=np.int64)
    table.obs["gaston_domain"] = pd.Categorical(domains, ordered=True)
    table.obs["gaston_isodepth"] = ordered_annotations["gaston_isodepth"].to_numpy(
        dtype=float
    )
    table.obs["gaston_model_seed"] = ordered_annotations["gaston_model_seed"].to_numpy(
        dtype=np.int64
    )
    table.obs["gaston_num_domains"] = ordered_annotations[
        "gaston_num_domains"
    ].to_numpy(dtype=np.int64)


def _write_annotation_plots(
    *,
    coordinates: np.ndarray,
    isodepth: np.ndarray,
    labels: np.ndarray,
    likelihoods: pd.DataFrame,
    selected_num_domains: int,
    seed_dirs: list[Path],
    plots_dir: Path,
    dpi: int,
) -> None:
    orientation_title = "Raw GASTON isodepth (unanchored; direction arbitrary)"
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=isodepth,
        cmap="viridis",
        s=2,
        linewidths=0,
    )
    ax.set_title(orientation_title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.colorbar(scatter, ax=ax, label="Raw isodepth; orientation arbitrary")
    _save_plot_pair(fig, plots_dir / "isodepth", dpi=dpi)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab20",
        s=2,
        linewidths=0,
    )
    ax.set_title(
        f"GASTON domains (K={selected_num_domains}; isodepth direction arbitrary)"
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.colorbar(scatter, ax=ax, label="GASTON domain")
    _save_plot_pair(fig, plots_dir / "domains", dpi=dpi)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        likelihoods["num_domains"],
        likelihoods["negative_log_likelihood"],
        marker="o",
    )
    ax.axvline(selected_num_domains, color="0.4", linestyle="--")
    ax.set_xlabel("Number of domains")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("GASTON automatic domain selection")
    _save_plot_pair(fig, plots_dir / "domain_likelihood", dpi=dpi)

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = 0
    for seed_dir in sorted(seed_dirs, key=_seed_from_path):
        loss_path = seed_dir / "loss_list.txt"
        if not loss_path.exists():
            continue
        losses = np.atleast_1d(np.loadtxt(loss_path, dtype=float))
        ax.plot(losses, linewidth=0.7, alpha=0.55, label=str(_seed_from_path(seed_dir)))
        plotted += 1
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"GASTON training losses ({plotted} restarts)")
    if 0 < plotted <= 12:
        ax.legend(title="Seed", ncol=2, fontsize=7)
    _save_plot_pair(fig, plots_dir / "training_losses", dpi=dpi)


def _save_plot_pair(fig: plt.Figure, stem: Path, *, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _copy_retained_seed_outputs(
    config: GastonConfig,
    seed_dirs: list[Path | str],
    best: SeedResult,
    model_dir: Path,
) -> None:
    retain_seeds = config.keep_seed_models or config.keep_checkpoints == "all"
    for raw_seed_dir in seed_dirs:
        seed_dir = Path(raw_seed_dir)
        is_best = seed_dir.resolve() == best.output_dir.resolve()
        keep_checkpoints = config.keep_checkpoints == "all" or (
            config.keep_checkpoints == "best" and is_best
        )
        if not retain_seeds and not keep_checkpoints:
            continue
        destination = model_dir / "seeds" / f"seed_{_seed_from_path(seed_dir)}"
        destination.mkdir(parents=True, exist_ok=True)
        for source in seed_dir.iterdir():
            if source.name.startswith("model_epoch_") and not keep_checkpoints:
                continue
            if source.name == "final_model.pt" and not config.keep_seed_models:
                continue
            if source.is_file():
                shutil.copy2(source, destination / source.name)


def _copy_gpu_vram_logs(seed_dirs: list[Path | str], model_dir: Path) -> None:
    """Publish the lightweight VRAM trace for every monitored restart."""
    for raw_seed_dir in seed_dirs:
        seed_dir = Path(raw_seed_dir)
        source = seed_dir / "gpu_vram"
        if not source.is_dir():
            continue
        destination = model_dir / "gpu_vram" / f"seed_{_seed_from_path(seed_dir)}"
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _seed_from_path(path: Path | str) -> int:
    digits = "".join(character for character in Path(path).name if character.isdigit())
    return int(digits) if digits else -1


def _torch_load(path: Path, *, map_location: str) -> Any:
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB while macOS reports bytes.
    return peak * 1024 if sys.platform.startswith("linux") else peak


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Cannot summarize empty or non-finite GASTON QC values")
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _jsonable_config(config: GastonConfig) -> dict[str, Any]:
    return json.loads(config.model_dump_json())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
