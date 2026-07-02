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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from skimage import measure

from merxen.cortical_depth.ribbon import RibbonGrid
from merxen.cortical_depth.streamlines import Streamline


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
