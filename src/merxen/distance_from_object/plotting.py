"""QC plotting for distance-from-object analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from merxen.distance_from_object.annotations import ObjectAnnotation


def plot_cell_distances(
    path: Path | str,
    cells: pd.DataFrame,
    annotations: list[ObjectAnnotation],
    *,
    max_distance_um: float,
) -> Path:
    """Plot cell centroids colored by clipped nearest-edge distance."""
    output_path = Path(path)
    figure, axis = plt.subplots(figsize=(9, 9))
    values = pd.to_numeric(
        cells["distance_to_object_edge_um"], errors="coerce"
    ).to_numpy(float)
    scatter = axis.scatter(
        cells["x"],
        cells["y"],
        c=np.clip(values, 0.0, float(max_distance_um)),
        s=1.5,
        alpha=0.75,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    for annotation in annotations:
        xy = np.asarray(annotation.geometry.exterior.coords, dtype=float)
        axis.plot(xy[:, 0], xy[:, 1], color="#ff2da1", linewidth=1.0)
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.75)
    colorbar.set_label(
        f"Distance to nearest object edge (µm; clipped at {max_distance_um:g})"
    )
    axis.set_title("Cell distance from annotated objects")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    _save_png_and_pdf(figure, output_path)
    return output_path


def plot_proximity_counts(path: Path | str, cells: pd.DataFrame) -> Path:
    """Plot counts of cells by tissue annotation and proximity class."""
    output_path = Path(path)
    counts = (
        cells.groupby(
            ["cortical_depth_annotation", "object_proximity"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
    )
    order = ["near", "middle", "far", "beyond_max"]
    counts = counts.reindex(columns=[value for value in order if value in counts])
    figure, axis = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", stacked=True, ax=axis, width=0.8)
    axis.set_title("Cells by tissue annotation and object proximity")
    axis.set_xlabel("Tissue annotation")
    axis.set_ylabel("Cells")
    axis.legend(title="Proximity", frameon=False)
    axis.tick_params(axis="x", rotation=25)
    _save_png_and_pdf(figure, output_path)
    return output_path


def plot_volcano(path: Path | str, results: pd.DataFrame) -> Path:
    """Plot paired near-vs-far differential-expression results."""
    output_path = Path(path)
    frame = results.copy()
    adjusted = pd.to_numeric(frame["padj"], errors="coerce").fillna(1.0)
    adjusted = adjusted.clip(lower=1e-300)
    fold_change = pd.to_numeric(frame["log2FoldChange"], errors="coerce")
    significant = adjusted.lt(0.05)
    colors = np.where(
        significant & fold_change.gt(0),
        "#d73027",
        np.where(significant & fold_change.lt(0), "#4575b4", "#9e9e9e"),
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(
        fold_change,
        -np.log10(adjusted),
        c=colors,
        s=10,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    axis.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title("Paired pseudobulk near vs far")
    axis.set_xlabel("log2 fold change (near / far)")
    axis.set_ylabel("-log10 adjusted p-value")
    _save_png_and_pdf(figure, output_path)
    return output_path


def _save_png_and_pdf(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
