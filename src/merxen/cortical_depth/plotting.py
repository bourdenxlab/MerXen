"""QC plotting and GeoJSON export helpers for cortical depth."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from skimage import measure

from merxen.cortical_depth.ribbon import RibbonGrid
from merxen.cortical_depth.streamlines import Streamline

# Human-readable y-axis labels for the depth columns produced upstream.
DEPTH_AXIS_LABELS = {
    "laplace_depth": "Laplace depth (pia = 0, WM = 1)",
    "equivolumetric_depth": "Equivolumetric depth (pia = 0, WM = 1)",
}

# Cluster labels that represent unassigned/ambiguous cells; ordered last.
_UNKNOWN_CLUSTER_LABELS = frozenset(
    {"Mixed/Unknown", "Unknown", "unknown", "unclustered", "nan", "None", ""}
)


def depth_contours_to_geojson(
    depth: np.ndarray,
    grid: RibbonGrid,
    *,
    levels: list[float],
    property_name: str = "laplace_depth",
) -> dict[str, Any]:
    """Convert raster depth contours to GeoJSON LineString features."""
    features: list[dict[str, Any]] = []
    field = np.asarray(depth, dtype=float)
    for level in levels:
        contours = measure.find_contours(field, float(level), mask=grid.mask)
        for contour_index, contour in enumerate(contours):
            if contour.shape[0] < 2:
                continue
            rows = contour[:, 0]
            cols = contour[:, 1]
            coords = grid.spec.indices_to_points(rows, cols)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        property_name: float(level),
                        "contour_index": int(contour_index),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [float(x_coord), float(y_coord)]
                            for x_coord, y_coord in coords
                        ],
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def write_geojson(data: dict[str, Any], path: Path | str) -> Path:
    """Write a GeoJSON dictionary to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    return output_path


def plot_depth_overlay(
    path: Path | str,
    grid: RibbonGrid,
    laplace_depth: np.ndarray,
    streamlines: list[Streamline],
    *,
    contour_levels: list[float],
) -> Path:
    """Save a QC overlay with ribbon, boundaries, depth contours, and streamlines."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    extent = _extent(grid)
    masked = np.ma.masked_invalid(np.where(grid.mask, laplace_depth, np.nan))
    ax.imshow(masked, origin="lower", extent=extent, cmap="viridis", alpha=0.55)
    _plot_line(ax, grid.pial_line, color="#2c7fb8", linewidth=2.0, label="pia")
    if grid.wm_line is not None:
        _plot_line(ax, grid.wm_line, color="#d95f0e", linewidth=2.0, label="WM")
    for side_line in grid.side_lines:
        _plot_line(ax, side_line, color="#666666", linewidth=1.0, linestyle="--")
    ax.contour(
        grid.spec.x_centers,
        grid.spec.y_centers,
        np.where(grid.mask, laplace_depth, np.nan),
        levels=contour_levels,
        colors="white",
        linewidths=0.6,
        alpha=0.8,
    )
    for streamline in streamlines:
        pts = np.asarray(streamline.points)
        if pts.shape[0] >= 2:
            color = "#d7301f" if streamline.near_side_boundary else "#111111"
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=0.5, alpha=0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", frameon=False)
    _save_png_pdf(fig, output_path)
    return output_path


def plot_cells_by_depth(
    path: Path | str,
    cells: pd.DataFrame,
    grid: RibbonGrid,
    *,
    value_column: str,
    cmap: str = "viridis",
) -> Path:
    """Save a cell scatter plot colored by one depth column."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(
        np.where(grid.mask, 1.0, np.nan),
        origin="lower",
        extent=_extent(grid),
        cmap="Greys",
        alpha=0.15,
    )
    _plot_line(ax, grid.pial_line, color="#2c7fb8", linewidth=1.5)
    if grid.wm_line is not None:
        _plot_line(ax, grid.wm_line, color="#d95f0e", linewidth=1.5)
    valid = (
        np.isfinite(pd.to_numeric(cells.get("x"), errors="coerce"))
        & np.isfinite(pd.to_numeric(cells.get("y"), errors="coerce"))
        & np.isfinite(pd.to_numeric(cells.get(value_column), errors="coerce"))
    )
    if valid.any():
        scatter = ax.scatter(
            cells.loc[valid, "x"],
            cells.loc[valid, "y"],
            c=pd.to_numeric(cells.loc[valid, value_column], errors="coerce"),
            s=2,
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.7, label=value_column)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _save_png_pdf(fig, output_path)
    return output_path


def plot_depth_difference(
    path: Path | str,
    grid: RibbonGrid,
    laplace_depth: np.ndarray,
    equivolumetric_depth: np.ndarray,
) -> Path:
    """Save a raster plot of Laplace minus equivolumetric depth."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    difference = np.asarray(laplace_depth, dtype=float) - np.asarray(
        equivolumetric_depth, dtype=float
    )
    masked = np.ma.masked_invalid(np.where(grid.mask, difference, np.nan))
    finite = np.asarray(masked.compressed(), dtype=float)
    limit = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=_extent(grid),
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    _plot_line(ax, grid.pial_line, color="#2c7fb8", linewidth=1.5)
    if grid.wm_line is not None:
        _plot_line(ax, grid.wm_line, color="#d95f0e", linewidth=1.5)
    for side_line in grid.side_lines:
        _plot_line(ax, side_line, color="#666666", linewidth=1.0, linestyle="--")
    fig.colorbar(
        image,
        ax=ax,
        shrink=0.7,
        label="laplace_depth - equivolumetric_depth",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _save_png_pdf(fig, output_path)
    return output_path


def plot_cells_by_annotation(
    path: Path | str,
    cells: pd.DataFrame,
    grids: list[RibbonGrid],
    *,
    category_column: str = "cortical_depth_annotation",
) -> Path:
    """Save a whole-sample cell scatter plot colored by tissue annotation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    for grid in grids:
        ax.imshow(
            np.where(grid.mask, 1.0, np.nan),
            origin="lower",
            extent=_extent(grid),
            cmap="Greys",
            alpha=0.08,
        )
        _plot_line(ax, grid.pial_line, color="#2c7fb8", linewidth=1.0)
        if grid.wm_line is not None:
            _plot_line(ax, grid.wm_line, color="#d95f0e", linewidth=1.0)
        for side_line in grid.side_lines:
            _plot_line(ax, side_line, color="#666666", linewidth=0.6, linestyle="--")

    valid = np.isfinite(pd.to_numeric(cells.get("x"), errors="coerce")) & np.isfinite(
        pd.to_numeric(cells.get("y"), errors="coerce")
    )
    category_values = (
        cells[category_column].astype(str)
        if category_column in cells.columns
        else pd.Series("outside_brain", index=cells.index)
    )
    categories = {
        "outside_brain": ("outside brain", "#9aa7b2"),
        "white_matter": ("white matter", "#f7f7f2"),
        "grey_matter": ("grey matter", "#4d4d4d"),
        "excluded": ("excluded", "#c44e52"),
    }
    for category, (_label, color) in categories.items():
        take = valid & (category_values == category)
        if not take.any():
            continue
        edgecolors = "#777777" if category == "white_matter" else "none"
        ax.scatter(
            cells.loc[take, "x"],
            cells.loc[take, "y"],
            s=2,
            c=color,
            linewidths=0.15,
            edgecolors=edgecolors,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="#777777" if category == "white_matter" else color,
            markersize=5,
            label=label,
        )
        for category, (label, color) in categories.items()
        if (valid & (category_values == category)).any()
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=False)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    _save_png_pdf(fig, output_path)
    return output_path


def plot_depth_violins_by_broad_class(
    path: Path | str,
    cells: pd.DataFrame,
    *,
    depth_column: str,
    class_column: str = "broad_class",
    depth_label: str | None = None,
    title: str | None = None,
) -> Path:
    """Save violin distributions of one depth column per broad cell-type cluster.

    Args:
        path: Output PNG path. A PDF copy is written alongside it.
        cells: Per-cell table with the depth column and a broad-class column.
        depth_column: Depth column to distribute (``laplace_depth`` or
            ``equivolumetric_depth``).
        class_column: Column holding the broad cell-type cluster label.
        depth_label: Optional y-axis label. Defaults to a readable label for
            the depth column.
        title: Optional plot title.

    Returns:
        The written PNG path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = _depth_violin_frame(
        cells, depth_column=depth_column, group_columns=[class_column]
    )
    order = _ordered_cluster_labels(cells, class_column, plot_df)
    axis_label = depth_label or DEPTH_AXIS_LABELS.get(depth_column, depth_column)

    width = max(6.0, 1.1 * max(len(order), 1))
    fig, ax = plt.subplots(figsize=(width, 5.0), constrained_layout=True)
    if plot_df.empty or not order:
        _empty_axis(ax, "No cells with depth values and cluster labels")
    else:
        sns.violinplot(
            data=plot_df,
            x=class_column,
            y=depth_column,
            order=order,
            hue=class_column,
            hue_order=order,
            palette="tab20",
            legend=False,
            cut=0,
            density_norm="width",
            ax=ax,
        )
        _format_depth_axis(ax, axis_label, ylim=_depth_ylim(plot_df[depth_column]))
        ax.set_xlabel("Broad cell-type cluster")
        ax.set_title(
            title or f"{axis_label} by broad cell-type cluster",
            fontsize=10,
        )
        _rotate_xticklabels(ax)
    _save_png_pdf(fig, output_path)
    return output_path


def plot_depth_violins_by_subcluster(
    path: Path | str,
    cells: pd.DataFrame,
    *,
    depth_column: str,
    class_column: str = "broad_class",
    subcluster_column: str = "subcluster_label",
    depth_label: str | None = None,
    max_columns: int = 3,
) -> Path:
    """Save a grid of subplots, one broad class each, with per-subcluster violins.

    Every subplot corresponds to one broad cell class and shows violin
    distributions of the depth column for each subclustered annotation within
    that class.

    Args:
        path: Output PNG path. A PDF copy is written alongside it.
        cells: Per-cell table with depth, broad-class, and subcluster columns.
        depth_column: Depth column to distribute (``laplace_depth`` or
            ``equivolumetric_depth``).
        class_column: Column holding the broad cell-type cluster label.
        subcluster_column: Column holding the subclustered annotation label.
        depth_label: Optional y-axis label. Defaults to a readable label for
            the depth column.
        max_columns: Maximum number of subplot columns in the grid.

    Returns:
        The written PNG path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis_label = depth_label or DEPTH_AXIS_LABELS.get(depth_column, depth_column)
    plot_df = _depth_violin_frame(
        cells,
        depth_column=depth_column,
        group_columns=[class_column, subcluster_column],
    )
    class_order = _ordered_cluster_labels(cells, class_column, plot_df)

    if plot_df.empty or not class_order:
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
        _empty_axis(ax, "No cells with depth values and subcluster labels")
        _save_png_pdf(fig, output_path)
        return output_path

    ylim = _depth_ylim(plot_df[depth_column])
    n_panels = len(class_order)
    n_cols = max(1, min(max_columns, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(4.5 * n_cols, 6.0), max(4.5 * n_rows, 4.5)),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()
    for index, broad_class in enumerate(class_order):
        ax = flat_axes[index]
        class_df = plot_df[plot_df[class_column] == broad_class]
        subcluster_order = _ordered_cluster_labels(cells, subcluster_column, class_df)
        if class_df.empty or not subcluster_order:
            _empty_axis(ax, "No labelled subclusters")
            ax.set_title(str(broad_class), fontsize=9)
            continue
        sns.violinplot(
            data=class_df,
            x=subcluster_column,
            y=depth_column,
            order=subcluster_order,
            hue=subcluster_column,
            hue_order=subcluster_order,
            palette="tab20",
            legend=False,
            cut=0,
            density_norm="width",
            ax=ax,
        )
        _format_depth_axis(ax, axis_label, ylim=ylim)
        ax.set_xlabel("Subcluster")
        ax.set_title(str(broad_class), fontsize=9)
        _rotate_xticklabels(ax)

    for ax in flat_axes[n_panels:]:
        ax.set_axis_off()

    fig.suptitle(f"{axis_label} by subcluster within broad class", fontsize=11)
    _save_png_pdf(fig, output_path)
    return output_path


def _depth_violin_frame(
    cells: pd.DataFrame,
    *,
    depth_column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """Return long-form rows with finite depth and non-null group labels."""
    required = [depth_column, *group_columns]
    if any(column not in cells.columns for column in required):
        return pd.DataFrame(columns=required)
    frame = cells[required].copy()
    frame[depth_column] = pd.to_numeric(frame[depth_column], errors="coerce")
    valid = np.isfinite(frame[depth_column].to_numpy())
    for column in group_columns:
        labels = frame[column].astype(str)
        valid &= frame[column].notna().to_numpy()
        valid &= ~labels.isin(_UNKNOWN_CLUSTER_LABELS).to_numpy()
        frame[column] = labels
    return frame.loc[valid].reset_index(drop=True)


def _ordered_cluster_labels(
    cells: pd.DataFrame,
    column: str,
    plot_df: pd.DataFrame,
) -> list[str]:
    """Order labels by source categorical order, otherwise sorted; unknowns last."""
    present = (
        set(plot_df[column].astype(str).unique())
        if column in plot_df.columns and not plot_df.empty
        else set()
    )
    if not present:
        return []
    ordered: list[str] = []
    if column in cells.columns and isinstance(cells[column].dtype, pd.CategoricalDtype):
        ordered = [
            str(category)
            for category in cells[column].cat.categories
            if str(category) in present
        ]
    remaining = sorted(present.difference(ordered))
    ordered.extend(remaining)
    return [label for label in ordered if label not in _UNKNOWN_CLUSTER_LABELS]


def _depth_ylim(values: pd.Series) -> tuple[float, float] | None:
    finite = pd.to_numeric(values, errors="coerce").to_numpy()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    low = float(min(0.0, np.min(finite)))
    high = float(max(1.0, np.max(finite)))
    pad = 0.02 * (high - low or 1.0)
    return (low - pad, high + pad)


def _format_depth_axis(
    ax: plt.Axes,
    label: str,
    *,
    ylim: tuple[float, float] | None,
) -> None:
    ax.set_ylabel(label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # Pia (depth 0) at the top, white matter (depth 1) at the bottom.
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def _rotate_xticklabels(ax: plt.Axes) -> None:
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(45)
        tick_label.set_horizontalalignment("right")


def _empty_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9)
    ax.set_axis_off()


def _plot_line(
    ax: plt.Axes,
    line: Any,
    *,
    color: str,
    linewidth: float,
    label: str | None = None,
    linestyle: str = "-",
) -> None:
    coords = np.asarray(line.coords, dtype=float)
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=color,
        linewidth=linewidth,
        label=label,
        linestyle=linestyle,
    )


def _extent(grid: RibbonGrid) -> tuple[float, float, float, float]:
    return (
        float(grid.spec.x_centers[0]),
        float(grid.spec.x_centers[-1]),
        float(grid.spec.y_centers[0]),
        float(grid.spec.y_centers[-1]),
    )


def _save_png_pdf(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=180)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=180)
    plt.close(fig)
