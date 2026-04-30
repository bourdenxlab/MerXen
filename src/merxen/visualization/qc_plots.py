"""QC plotting utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

PLATFORM_COLORS: dict[str, str] = {
    "MERSCOPE": "#1f77b4",
    "XENIUM": "#d62728",
}
PLATFORM_ORDER: tuple[str, str] = ("XENIUM", "MERSCOPE")
GEOMETRY_METRIC_ORDER: tuple[str, str, str] = ("area", "eccentricity", "aspect_ratio")
CELL_METRIC_LABELS: dict[str, str] = {
    "transcripts_per_cell": "Transcripts per cell",
    "genes_per_cell": "Genes per cell",
}


def plot_geometry_histograms(
    geometry_metrics: pd.DataFrame,
    output_path: Path | str,
    *,
    bins: int = 50,
) -> Path:
    """Plot geometry metric histograms for area/eccentricity/aspect ratio."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        c for c in ["area", "eccentricity", "aspect_ratio"] if c in geometry_metrics
    ]
    ncols = max(1, len(cols))
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for ax, col in zip(axes, cols, strict=False):
        vals = pd.to_numeric(geometry_metrics[col], errors="coerce").dropna()
        ax.hist(vals, bins=bins, color="#1f77b4", alpha=0.8)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_geometry_histograms_comparison(
    geometry_metrics_by_dataset: Mapping[str, pd.DataFrame],
    output_path: Path | str,
    *,
    bins: int = 50,
) -> Path:
    """Plot overlaid geometry histograms for Xenium and MERSCOPE."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        col
        for col in GEOMETRY_METRIC_ORDER
        if any(col in df for df in geometry_metrics_by_dataset.values())
    ]
    if not cols:
        raise ValueError("No supported geometry metric columns were found.")

    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols, strict=False):
        values_by_dataset = _numeric_values_by_dataset(
            geometry_metrics_by_dataset,
            col,
        )
        combined = np.concatenate(list(values_by_dataset.values()))
        if combined.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax.set_axis_off()
            continue

        bin_edges = np.histogram_bin_edges(combined, bins=bins)
        for dataset_name in _ordered_dataset_names(values_by_dataset):
            vals = values_by_dataset[dataset_name]
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=bin_edges,
                histtype="step",
                linewidth=1.8,
                color=PLATFORM_COLORS.get(dataset_name.upper()),
                label=dataset_name,
            )

        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_cell_metrics_violin(
    cell_metrics: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Plot violin distributions of transcripts/cell and genes/cell."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [c for c in ["transcripts_per_cell", "genes_per_cell"] if c in cell_metrics]
    if not cols:
        raise ValueError(
            "cell_metrics is missing required columns for violin plotting."
        )

    melted = cell_metrics[cols].melt(var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=melted, x="metric", y="value", fill=True, ax=ax)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Per-cell QC Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_cell_metrics_violin_comparison(
    cell_metrics_by_dataset: Mapping[str, pd.DataFrame],
    output_path: Path | str,
) -> Path:
    """Plot side-by-side per-cell transcript and gene violins by platform."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = [
        metric
        for metric in CELL_METRIC_LABELS
        if _has_metric(metric, cell_metrics_by_dataset)
    ]
    if not metrics:
        raise ValueError("No supported cell metric columns were found.")

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics, strict=False):
        plot_df = _cell_metric_long_df(cell_metrics_by_dataset, metric)
        if plot_df.empty:
            ax.text(0.5, 0.5, "No positive values", ha="center", va="center")
            ax.set_axis_off()
            continue

        order = _ordered_dataset_names_from_series(plot_df["dataset"])
        palette = {name: PLATFORM_COLORS.get(name.upper(), "#4b5563") for name in order}
        sns.violinplot(
            data=plot_df,
            x="dataset",
            y="value",
            hue="dataset",
            order=order,
            hue_order=order,
            palette=palette,
            legend=False,
            fill=True,
            cut=0,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(CELL_METRIC_LABELS[metric])
        ax.set_title(CELL_METRIC_LABELS[metric])

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_assignment_bar(
    assignment_summaries: pd.DataFrame,
    output_path: Path | str,
    *,
    dataset_col: str = "dataset",
    pct_col: str = "pct_assigned",
) -> Path:
    """Plot per-dataset assignment rate bar chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_col not in assignment_summaries or pct_col not in assignment_summaries:
        raise KeyError(
            f"Expected columns '{dataset_col}' and '{pct_col}' in assignment_summaries."
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=assignment_summaries, x=dataset_col, y=pct_col, ax=ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Assigned Transcripts (%)")
    ax.set_xlabel("Dataset")
    ax.set_title("Transcript Assignment Rate")
    for patch in ax.patches:
        if not isinstance(patch, Rectangle):
            continue
        height = patch.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _numeric_values_by_dataset(
    frames_by_dataset: Mapping[str, pd.DataFrame],
    column: str,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for dataset_name, df in frames_by_dataset.items():
        if column not in df:
            values[str(dataset_name)] = np.array([], dtype=np.float64)
            continue
        arr = pd.to_numeric(df[column], errors="coerce").to_numpy(np.float64)
        values[str(dataset_name)] = arr[np.isfinite(arr)]
    return values


def _has_metric(
    metric: str,
    frames_by_dataset: Mapping[str, pd.DataFrame],
) -> bool:
    return any(metric in df for df in frames_by_dataset.values())


def _cell_metric_long_df(
    frames_by_dataset: Mapping[str, pd.DataFrame],
    metric: str,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for dataset_name, df in frames_by_dataset.items():
        if metric not in df:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce")
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.empty:
            continue
        records.append(
            pd.DataFrame(
                {
                    "dataset": str(dataset_name),
                    "value": vals.to_numpy(np.float64),
                }
            )
        )
    if not records:
        return pd.DataFrame(columns=["dataset", "value"])
    return pd.concat(records, ignore_index=True)


def _ordered_dataset_names(values_by_dataset: Mapping[str, np.ndarray]) -> list[str]:
    present = list(values_by_dataset.keys())
    ordered = [name for name in PLATFORM_ORDER if name in present]
    ordered.extend(name for name in present if name not in ordered)
    return ordered


def _ordered_dataset_names_from_series(series: pd.Series) -> list[str]:
    present = list(dict.fromkeys(series.astype(str).tolist()))
    ordered = [name for name in PLATFORM_ORDER if name in present]
    ordered.extend(name for name in present if name not in ordered)
    return ordered
