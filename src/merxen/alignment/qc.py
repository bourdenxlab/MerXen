"""SABench-style alignment QC metrics for MerXen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse
from scipy.spatial import cKDTree

from merxen.alignment.features import build_alignment_adata, shared_gene_subset
from merxen.config import AlignmentQCConfig


def run_alignment_qc(config: AlignmentQCConfig) -> dict[str, Path]:
    """Compute post-alignment QC outputs for one pair."""
    cfg = config
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    xenium_sdata = sd.read_zarr(cfg.xenium_zarr_path)
    merscope_sdata = sd.read_zarr(cfg.merscope_zarr_path)
    fixed = build_alignment_adata(xenium_sdata, platform="XENIUM")
    moving = build_alignment_adata(merscope_sdata, platform="MERSCOPE")
    fixed, moving = shared_gene_subset(fixed, moving)

    metrics: dict[str, float | int | str] = dict(
        compute_grid_alignment_metrics(
            fixed,
            moving,
            grid_shape=(cfg.grid_rows, cfg.grid_cols),
        )
    )
    metrics["pair_id"] = cfg.pair_id
    if cfg.transform_json_path is not None and Path(cfg.transform_json_path).exists():
        metrics["transform_json_path"] = str(cfg.transform_json_path)

    metrics_json = cfg.output_dir / f"{cfg.pair_id}_alignment_qc.json"
    metrics_json.write_text(json.dumps(_jsonable(metrics), indent=2))

    metrics_csv = cfg.output_dir / f"{cfg.pair_id}_alignment_qc_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    overlay_png = cfg.output_dir / f"{cfg.pair_id}_alignment_overlay.png"
    plot_alignment_overlay(fixed, moving, overlay_png, title=f"{cfg.pair_id} alignment")

    return {
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "overlay_png": overlay_png,
    }


def compute_grid_alignment_metrics(
    fixed: ad.AnnData,
    moving: ad.AnnData,
    *,
    grid_shape: tuple[int, int] = (10, 10),
) -> dict[str, float | int]:
    """Compute SABench-style grid expression and point-overlap metrics."""
    fixed_xy = np.asarray(fixed.obsm["spatial"], dtype=float)
    moving_xy = np.asarray(moving.obsm["spatial"], dtype=float)
    fixed_grid, moving_grid, n_grids = _shared_overlap_grid(
        fixed_xy,
        moving_xy,
        grid_shape=grid_shape,
    )
    fixed_expr = _dense(fixed.X)
    moving_expr = _dense(moving.X)

    fixed_means, moving_means = _grid_means(
        fixed_expr,
        moving_expr,
        fixed_grid,
        moving_grid,
        n_grids=n_grids,
    )
    valid_grid = np.isfinite(fixed_means).all(axis=1) & np.isfinite(moving_means).all(
        axis=1
    )
    fixed_means = fixed_means[valid_grid]
    moving_means = moving_means[valid_grid]

    return {
        "grid_rows": int(grid_shape[0]),
        "grid_cols": int(grid_shape[1]),
        "n_overlap_grids": int(len(fixed_means)),
        "n_fixed_cells": int(fixed.n_obs),
        "n_moving_cells": int(moving.n_obs),
        "gene_grid_pearson": _mean_gene_pearson(fixed_means, moving_means),
        "grid_cosine": _mean_grid_cosine(fixed_means, moving_means),
        "grid_mutual_information": _mean_gene_mi(fixed_means, moving_means),
        "centroid_assd": _assd(fixed_xy, moving_xy),
    }


def plot_alignment_overlay(
    fixed: ad.AnnData,
    moving: ad.AnnData,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save a centroid overlay plot for aligned sections."""
    fixed_xy = np.asarray(fixed.obsm["spatial"], dtype=float)
    moving_xy = np.asarray(moving.obsm["spatial"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 8))
    if len(fixed_xy):
        ax.scatter(fixed_xy[:, 0], fixed_xy[:, 1], s=1, alpha=0.35, label="Xenium")
    if len(moving_xy):
        ax.scatter(moving_xy[:, 0], moving_xy[:, 1], s=1, alpha=0.35, label="MERSCOPE")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _shared_overlap_grid(
    fixed_xy: np.ndarray,
    moving_xy: np.ndarray,
    *,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    all_xy = np.vstack([fixed_xy, moving_xy])
    xmin, ymin = np.nanmin(all_xy, axis=0)
    xmax, ymax = np.nanmax(all_xy, axis=0)
    rows, cols = grid_shape
    x_edges = np.linspace(xmin, xmax, cols + 1)
    y_edges = np.linspace(ymin, ymax, rows + 1)
    return (
        _grid_ids(fixed_xy, x_edges, y_edges, cols),
        _grid_ids(moving_xy, x_edges, y_edges, cols),
        rows * cols,
    )


def _grid_ids(
    xy: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cols: int,
) -> np.ndarray:
    x_bin = np.searchsorted(x_edges, xy[:, 0], side="right") - 1
    y_bin = np.searchsorted(y_edges, xy[:, 1], side="right") - 1
    x_bin = np.clip(x_bin, 0, len(x_edges) - 2)
    y_bin = np.clip(y_bin, 0, len(y_edges) - 2)
    return y_bin * cols + x_bin


def _grid_means(
    fixed_expr: np.ndarray,
    moving_expr: np.ndarray,
    fixed_grid: np.ndarray,
    moving_grid: np.ndarray,
    *,
    n_grids: int,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_means = np.full((n_grids, fixed_expr.shape[1]), np.nan, dtype=float)
    moving_means = np.full((n_grids, moving_expr.shape[1]), np.nan, dtype=float)
    for grid_id in range(n_grids):
        fmask = fixed_grid == grid_id
        mmask = moving_grid == grid_id
        if fmask.any() and mmask.any():
            fixed_means[grid_id] = np.nanmean(fixed_expr[fmask], axis=0)
            moving_means[grid_id] = np.nanmean(moving_expr[mmask], axis=0)
    return fixed_means, moving_means


def _mean_gene_pearson(fixed_means: np.ndarray, moving_means: np.ndarray) -> float:
    if fixed_means.shape[0] < 2:
        return float("nan")
    values = []
    for i in range(fixed_means.shape[1]):
        x = fixed_means[:, i]
        y = moving_means[:, i]
        if np.nanstd(x) > 0 and np.nanstd(y) > 0:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.nanmean(values)) if values else float("nan")


def _mean_grid_cosine(fixed_means: np.ndarray, moving_means: np.ndarray) -> float:
    if len(fixed_means) == 0:
        return float("nan")
    denom = np.linalg.norm(fixed_means, axis=1) * np.linalg.norm(moving_means, axis=1)
    num = np.sum(fixed_means * moving_means, axis=1)
    vals = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom > 0)
    return float(np.nanmean(vals))


def _mean_gene_mi(fixed_means: np.ndarray, moving_means: np.ndarray) -> float:
    if fixed_means.shape[0] < 3:
        return float("nan")
    vals = [
        _mutual_information(fixed_means[:, i], moving_means[:, i])
        for i in range(fixed_means.shape[1])
    ]
    return float(np.nanmean(vals)) if vals else float("nan")


def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 16) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3:
        return float("nan")
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist / np.sum(hist)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / np.outer(px, py)[nz])))


def _assd(fixed_xy: np.ndarray, moving_xy: np.ndarray) -> float:
    if len(fixed_xy) == 0 or len(moving_xy) == 0:
        return float("nan")
    f_tree = cKDTree(fixed_xy)
    m_tree = cKDTree(moving_xy)
    d_m_to_f, _ = f_tree.query(moving_xy, k=1)
    d_f_to_m, _ = m_tree.query(fixed_xy, k=1)
    return float((np.nanmean(d_m_to_f) + np.nanmean(d_f_to_m)) / 2.0)


def _dense(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray(), dtype=float)
    return np.asarray(x, dtype=float)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value
