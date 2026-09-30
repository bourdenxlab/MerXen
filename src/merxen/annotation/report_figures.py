"""Figures of the annotation report (plan §9): PNG + PDF + one CSV each.

Every figure is drawn from the data frame written beside it
(``<stem>.csv``), so a figure can be re-drawn or checked from its CSV.
Figures use ``matplotlib.figure.Figure`` directly (no pyplot state, the Agg
canvas), a fixed size and DPI, and no creation timestamps in the files, so
identical inputs give identical PNG and PDF bytes.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from merxen.palette import MERSCOPE, OPTIONAL, PAIRED, XENIUM
from merxen.plotting import configure_matplotlib_for_pdf

logger = logging.getLogger(__name__)
# The PDF backend subsets fonts through fontTools, which logs every table.
logging.getLogger("fontTools").setLevel(logging.WARNING)

DPI: Final = 110
PLATFORM_COLOURS: Final[dict[str, str]] = {"MERSCOPE": MERSCOPE, "XENIUM": XENIUM}
CYCLE: Final[tuple[str, ...]] = (
    PAIRED,
    OPTIONAL,
    "#4C9F38",
    "#B8336A",
    "#6B6B6B",
    "#C9A227",
    "#2E4057",
    "#8C564B",
    "#17BECF",
    "#9467BD",
)
PNG_METADATA: Final[dict[str, Any]] = {"Software": None}
PDF_METADATA: Final[dict[str, Any]] = {
    "Creator": None,
    "Producer": None,
    "CreationDate": None,
}


@dataclass(frozen=True)
class FigureRecord:
    """One written figure.

    Attributes:
        stem: File stem (``<stem>.png``, ``.pdf``, ``.csv``).
        png: The PNG path.
        pdf: The PDF path.
        csv: The data CSV path.
        caption: What the figure shows.
    """

    stem: str
    png: Path
    pdf: Path
    csv: Path
    caption: str


def colour_for(key: str, index: int = 0) -> str:
    """Return a platform colour, or a cycle colour for anything else."""
    upper = str(key).upper()
    if upper in PLATFORM_COLOURS:
        return PLATFORM_COLOURS[upper]
    return CYCLE[index % len(CYCLE)]


def new_figure(width: float, height: float) -> Figure:
    """Return a figure with an Agg canvas (no pyplot state)."""
    configure_matplotlib_for_pdf()
    figure = Figure(figsize=(width, height), dpi=DPI)
    FigureCanvasAgg(figure)
    return figure


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    """Write a data frame deterministically (fixed float format, no index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    return path


def save_report_figure(
    figure: Figure, directory: Path, stem: str, data: pd.DataFrame, caption: str
) -> FigureRecord:
    """Save a figure as PNG and PDF, and its data as CSV.

    Args:
        figure: The figure.
        directory: Output directory.
        stem: File stem.
        data: The data drawn.
        caption: Caption for the report.

    Returns:
        The record of the three files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    csv = directory / f"{stem}.csv"
    for collection in (item for ax in figure.axes for item in ax.collections):
        offsets = np.asarray(collection.get_offsets())
        if offsets.ndim and len(offsets) > 500:
            collection.set_rasterized(True)
    figure.savefig(png, dpi=DPI, metadata=PNG_METADATA)
    figure.savefig(pdf, dpi=DPI, metadata=PDF_METADATA)
    write_csv(data, csv)
    return FigureRecord(stem=stem, png=png, pdf=pdf, csv=csv, caption=caption)


def placeholder(message: str, *, width: float = 6.0, height: float = 1.6) -> Figure:
    """Return a figure that states why there is nothing to draw."""
    figure = new_figure(width, height)
    ax = figure.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=9)
    return figure


def grouped_bars(
    frame: pd.DataFrame,
    *,
    category: str,
    group: str,
    value: str,
    low: str | None = None,
    high: str | None = None,
    facet: str | None = None,
    ylabel: str = "",
    title: str = "",
    horizontal_line: float | None = None,
) -> Figure:
    """Grouped bar chart, one group per colour, optional CI whiskers and facets.

    Args:
        frame: Long table.
        category: Column on the x axis.
        group: Column mapped to colour.
        value: Bar height.
        low: CI lower bound column.
        high: CI upper bound column.
        facet: Column splitting panels.
        ylabel: Y label.
        title: Figure title.
        horizontal_line: A reference line (e.g. 0).

    Returns:
        The figure.
    """
    if frame.empty:
        return placeholder(f"{title}: no data")
    facets = [None] if facet is None else list(dict.fromkeys(frame[facet].astype(str)))
    categories = list(dict.fromkeys(frame[category].astype(str)))
    groups = list(dict.fromkeys(frame[group].astype(str)))
    width = max(5.0, min(16.0, 1.0 + 0.45 * len(categories) * max(1, len(groups)) / 2))
    figure = new_figure(width, 2.6 * len(facets) + 0.6)
    for row, facet_value in enumerate(facets):
        ax = figure.add_subplot(len(facets), 1, row + 1)
        part = (
            frame
            if facet_value is None
            else frame[frame[facet].astype(str) == facet_value]
        )
        n_groups = max(1, len(groups))
        bar_width = 0.8 / n_groups
        positions = np.arange(len(categories))
        for index, name in enumerate(groups):
            sub = part[part[group].astype(str) == name]
            keys = sub[category].astype(str).to_numpy()
            heights = _lookup(keys, sub[value], categories)
            offsets = positions - 0.4 + bar_width * (index + 0.5)
            ax.bar(
                offsets,
                heights,
                width=bar_width,
                color=colour_for(name, index),
                label=name,
            )
            if low is not None and high is not None:
                lows = _lookup(keys, sub[low], categories)
                highs = _lookup(keys, sub[high], categories)
                ok = np.isfinite(lows) & np.isfinite(highs) & np.isfinite(heights)
                if ok.any():
                    ax.errorbar(
                        offsets[ok],
                        heights[ok],
                        yerr=np.vstack(
                            [heights[ok] - lows[ok], highs[ok] - heights[ok]]
                        ).clip(0),
                        fmt="none",
                        ecolor="#222222",
                        elinewidth=0.8,
                        capsize=2,
                    )
        if horizontal_line is not None:
            ax.axhline(horizontal_line, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        if facet_value is not None:
            ax.set_title(str(facet_value), fontsize=8, loc="left")
        if row == 0:
            ax.legend(fontsize=7, frameon=False, ncol=min(4, n_groups))
    if title:
        figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    return figure


def _lookup(
    keys: np.ndarray, values: pd.Series, categories: Sequence[str]
) -> np.ndarray:
    """Return ``values`` ordered by ``categories`` (NaN where absent)."""
    mapping = dict(
        zip(keys, pd.to_numeric(values, errors="coerce").to_numpy(), strict=True)
    )
    return np.array(
        [float(mapping.get(item, math.nan)) for item in categories], dtype=float
    )


def stacked_bars(
    frame: pd.DataFrame,
    *,
    bar: str,
    segment: str,
    value: str,
    title: str = "",
    xlabel: str = "share",
) -> Figure:
    """Horizontal stacked bars (e.g. status breakdown per level × platform)."""
    if frame.empty:
        return placeholder(f"{title}: no data")
    bars = list(dict.fromkeys(frame[bar].astype(str)))
    segments = list(dict.fromkeys(frame[segment].astype(str)))
    figure = new_figure(7.5, 0.35 * len(bars) + 1.6)
    ax = figure.add_subplot(1, 1, 1)
    left = np.zeros(len(bars))
    for index, name in enumerate(segments):
        part = frame[frame[segment].astype(str) == name]
        lookup = dict(
            zip(part[bar].astype(str), part[value].astype(float), strict=True)
        )
        widths = np.array([lookup.get(item, 0.0) for item in bars])
        widths = np.nan_to_num(widths)
        ax.barh(
            np.arange(len(bars)),
            widths,
            left=left,
            color=colour_for(name, index),
            label=name,
        )
        left += widths
    ax.set_yticks(np.arange(len(bars)))
    ax.set_yticklabels(bars, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.legend(
        fontsize=6,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
    )
    if title:
        ax.set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


def line_facets(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    row: str | None = None,
    column: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    max_panels: int = 24,
) -> Figure:
    """Lines per hue in a grid of facets (e.g. precision-coverage curves)."""
    if frame.empty:
        return placeholder(f"{title}: no data")
    rows = [None] if row is None else list(dict.fromkeys(frame[row].astype(str)))
    columns = (
        [None] if column is None else list(dict.fromkeys(frame[column].astype(str)))
    )
    if len(rows) * len(columns) > max_panels:
        columns = columns[: max(1, max_panels // max(1, len(rows)))]
    hues = list(dict.fromkeys(frame[hue].astype(str)))
    figure = new_figure(1.9 * len(columns) + 1.5, 1.7 * len(rows) + 0.8)
    for i, row_value in enumerate(rows):
        for j, column_value in enumerate(columns):
            ax = figure.add_subplot(len(rows), len(columns), i * len(columns) + j + 1)
            part = frame
            if row_value is not None:
                part = part[part[row].astype(str) == row_value]
            if column_value is not None:
                part = part[part[column].astype(str) == column_value]
            for index, name in enumerate(hues):
                sub = part[part[hue].astype(str) == name].sort_values(x)
                if len(sub):
                    ax.plot(
                        sub[x],
                        sub[y],
                        color=colour_for(name, index),
                        linewidth=0.9,
                        label=name,
                    )
            ax.tick_params(labelsize=6)
            label = " / ".join(
                str(v) for v in (row_value, column_value) if v is not None
            )
            if label:
                ax.set_title(label, fontsize=7)
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel, fontsize=7)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=7)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            fontsize=6,
            frameon=False,
            loc="lower center",
            ncol=min(6, len(labels)),
        )
    if title:
        figure.suptitle(title, fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    return figure


def hist2d_panels(
    panels: Mapping[str, pd.DataFrame],
    *,
    xlabel: str,
    ylabel: str,
    thresholds: Mapping[str, Sequence[float]] | None = None,
    title: str = "",
) -> Figure:
    """Draw pre-binned 2D histograms (``report_metrics.histogram_2d`` tables)."""
    if not panels:
        return placeholder(f"{title}: no data")
    names = list(panels)
    n_columns = min(2, len(names))
    n_rows = int(math.ceil(len(names) / n_columns))
    figure = new_figure(4.2 * n_columns, 3.0 * n_rows + 0.4)
    for index, name in enumerate(names):
        table = panels[name]
        ax = figure.add_subplot(n_rows, n_columns, index + 1)
        if table.empty:
            ax.axis("off")
            continue
        x_edges = np.unique(np.concatenate([table["x_low"], table["x_high"]]))
        y_edges = np.unique(np.concatenate([table["y_low"], table["y_high"]]))
        grid = np.zeros((len(x_edges) - 1, len(y_edges) - 1))
        xi = np.searchsorted(x_edges, table["x_low"].to_numpy())
        yi = np.searchsorted(y_edges, table["y_low"].to_numpy())
        grid[xi, yi] = table["n"].to_numpy()
        image = ax.pcolormesh(
            x_edges, y_edges, np.log10(grid.T + 1.0), cmap="viridis", shading="flat"
        )
        figure.colorbar(image, ax=ax, label="log10(cells + 1)").ax.tick_params(
            labelsize=6
        )
        for value in (thresholds or {}).get(name, ()):
            ax.axhline(value, color="#FF4040", linewidth=0.8, linestyle="--")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
    if title:
        figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    return figure


def dotplot(
    frame: pd.DataFrame,
    *,
    label: str,
    gene: str,
    size_columns: tuple[str, str],
    colour_columns: tuple[str, str],
    panel_titles: tuple[str, str] = ("observed", "reference"),
    title: str = "",
) -> Figure:
    """Side-by-side dotplots: observed vs reference (size = fraction, colour = mean)."""
    if frame.empty:
        return placeholder(f"{title}: no data")
    labels = list(dict.fromkeys(frame[label].astype(str)))
    genes = list(dict.fromkeys(frame[gene].astype(str)))
    figure = new_figure(
        max(6.0, 0.25 * len(genes) * 2 + 3.0), max(2.5, 0.28 * len(labels) + 1.8)
    )
    colour_values = pd.concat(
        [frame[colour_columns[0]], frame[colour_columns[1]]]
    ).astype(float)
    vmax = float(np.nanmax(colour_values)) if np.isfinite(colour_values).any() else 1.0
    for index in range(2):
        ax = figure.add_subplot(1, 2, index + 1)
        x = np.array([genes.index(str(value)) for value in frame[gene]])
        y = np.array([labels.index(str(value)) for value in frame[label]])
        sizes = (
            np.nan_to_num(frame[size_columns[index]].astype(float).to_numpy()) * 60 + 1
        )
        colours = np.nan_to_num(frame[colour_columns[index]].astype(float).to_numpy())
        points = ax.scatter(
            x, y, s=sizes, c=colours, cmap="magma_r", vmin=0, vmax=vmax, linewidths=0
        )
        ax.set_xticks(np.arange(len(genes)))
        ax.set_xticklabels(genes, rotation=90, fontsize=5)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels if index == 0 else [""] * len(labels), fontsize=6)
        ax.set_xlim(-0.6, len(genes) - 0.4)
        ax.set_ylim(len(labels) - 0.4, -0.6)
        ax.set_title(panel_titles[index], fontsize=8)
    figure.colorbar(
        points, ax=figure.axes, label="mean log2(CPM + 1)", shrink=0.6
    ).ax.tick_params(labelsize=6)
    if title:
        figure.suptitle(title, fontsize=9)
    return figure


def box_summary(
    frame: pd.DataFrame,
    *,
    category: str,
    group: str,
    title: str = "",
    ylabel: str = "",
) -> Figure:
    """Boxes from pre-computed quantiles (``q10, q25, q50, q75, q90``)."""
    if frame.empty:
        return placeholder(f"{title}: no data")
    categories = list(dict.fromkeys(frame[category].astype(str)))
    groups = list(dict.fromkeys(frame[group].astype(str)))
    figure = new_figure(max(5.0, 0.7 * len(categories) * len(groups) / 1.5 + 1.5), 3.2)
    ax = figure.add_subplot(1, 1, 1)
    width = 0.8 / max(1, len(groups))
    for index, name in enumerate(groups):
        part = frame[frame[group].astype(str) == name]
        stats = []
        positions = []
        for position, cat in enumerate(categories):
            row = part[part[category].astype(str) == cat]
            if row.empty:
                continue
            record = row.iloc[0]
            stats.append(
                {
                    "whislo": float(record["q10"]),
                    "q1": float(record["q25"]),
                    "med": float(record["q50"]),
                    "q3": float(record["q75"]),
                    "whishi": float(record["q90"]),
                    "fliers": [],
                    "label": cat,
                }
            )
            positions.append(position - 0.4 + width * (index + 0.5))
        if stats:
            boxes = ax.bxp(
                stats,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
            )
            for patch in boxes["boxes"]:
                patch.set_facecolor(colour_for(name, index))
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    handles = [
        Patch(color=colour_for(name, index), label=name)
        for index, name in enumerate(groups)
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


def tile_maps(
    frame: pd.DataFrame,
    *,
    panel: str,
    x: str = "x_um",
    y: str = "y_um",
    value: str = "value",
    categorical: bool = False,
    title: str = "",
    colour_label: str = "",
    tile_um: float | None = None,
) -> Figure:
    """Draw tile maps (one panel per ``panel`` value)."""
    if frame.empty:
        return placeholder(f"{title}: no data")
    names = list(dict.fromkeys(frame[panel].astype(str)))
    figure = new_figure(4.2 * len(names), 4.0)
    categories = list(dict.fromkeys(frame[value].astype(str))) if categorical else []
    for index, name in enumerate(names):
        ax = figure.add_subplot(1, len(names), index + 1)
        part = frame[frame[panel].astype(str) == name]
        size = 4.0 if tile_um is None else max(1.0, 3000.0 / max(1.0, len(part)) ** 0.5)
        if categorical:
            for cat_index, category in enumerate(categories):
                sub = part[part[value].astype(str) == category]
                if len(sub):
                    ax.scatter(
                        sub[x],
                        sub[y],
                        s=size,
                        marker="s",
                        color=colour_for(category, cat_index),
                        label=category,
                        linewidths=0,
                    )
            ax.legend(fontsize=5, frameon=False, markerscale=2, loc="upper right")
        else:
            points = ax.scatter(
                part[x],
                part[y],
                s=size,
                marker="s",
                c=part[value].astype(float),
                cmap="viridis",
                linewidths=0,
            )
            figure.colorbar(
                points, ax=ax, label=colour_label, shrink=0.7
            ).ax.tick_params(labelsize=6)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
    if title:
        figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    return figure


def forest(
    frame: pd.DataFrame,
    *,
    label: str,
    group: str,
    value: str,
    low: str,
    high: str,
    title: str = "",
    xlabel: str = "",
) -> Figure:
    """Point estimates with CIs per label, one colour per group."""
    if frame.empty:
        return placeholder(f"{title}: no data")
    labels = list(dict.fromkeys(frame[label].astype(str)))
    groups = list(dict.fromkeys(frame[group].astype(str)))
    figure = new_figure(6.5, 0.3 * len(labels) + 1.4)
    ax = figure.add_subplot(1, 1, 1)
    step = 0.6 / max(1, len(groups))
    for index, name in enumerate(groups):
        part = frame[frame[group].astype(str) == name]
        y = (
            np.array([labels.index(str(v)) for v in part[label]])
            - 0.3
            + step * (index + 0.5)
        )
        centre = part[value].astype(float).to_numpy()
        lows = part[low].astype(float).to_numpy()
        highs = part[high].astype(float).to_numpy()
        ok = np.isfinite(lows) & np.isfinite(highs)
        ax.errorbar(
            centre[ok],
            y[ok],
            xerr=np.vstack([centre[ok] - lows[ok], highs[ok] - centre[ok]]).clip(0),
            fmt="o",
            color=colour_for(name, index),
            markersize=3,
            elinewidth=0.9,
            label=name,
        )
        ax.plot(
            centre[~ok],
            y[~ok],
            "o",
            color=colour_for(name, index),
            markersize=3,
            fillstyle="none",
        )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    if title:
        ax.set_title(title, fontsize=9)
    figure.tight_layout()
    return figure


__all__ = [
    "DPI",
    "FigureRecord",
    "box_summary",
    "colour_for",
    "dotplot",
    "forest",
    "grouped_bars",
    "hist2d_panels",
    "line_facets",
    "new_figure",
    "placeholder",
    "save_report_figure",
    "stacked_bars",
    "tile_maps",
    "write_csv",
]
