"""CLI command for generating visualization artifacts."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import spatialdata as sd

from merxen.config import VisualizationConfig, load_config_from_json
from merxen.qc.gene_comparison import compute_gene_comparison_from_paths
from merxen.qc.metrics import compute_dataset_qc
from merxen.visualization.density_overview import plot_transcript_overview
from merxen.visualization.gene_scatter import plot_gene_scatter
from merxen.visualization.qc_plots import (
    plot_assignment_bar,
    plot_cell_metrics_violin_comparison,
    plot_geometry_histograms_comparison,
)
from merxen.visualization.sanity_plots import plot_pair_sanity_crops


@click.command(name="visualize")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against VisualizationConfig.",
)
def visualize_command(config_path: Path) -> None:
    """Generate visualization artifacts for a paired dataset comparison."""
    cfg = load_config_from_json(config_path, VisualizationConfig)
    assert isinstance(cfg, VisualizationConfig)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Gene-level scatter plots from comparison tables.
    comparison = compute_gene_comparison_from_paths(
        xenium_zarr_path=cfg.xenium_zarr_path,
        merscope_zarr_path=cfg.merscope_zarr_path,
    )
    total_scatter = cfg.output_dir / f"{cfg.pair_id}_gene_scatter_total_normalized.png"
    assigned_scatter = (
        cfg.output_dir / f"{cfg.pair_id}_gene_scatter_assigned_normalized.png"
    )
    plot_gene_scatter(
        comparison["total_normalized_df"],
        total_scatter,
        title=f"{cfg.pair_id} Total Gene Counts (normalized)",
        x_label="Xenium (normalized)",
        y_label="MERSCOPE (normalized)",
        log_scale=True,
    )
    plot_gene_scatter(
        comparison["assigned_normalized_df"],
        assigned_scatter,
        title=f"{cfg.pair_id} Assigned Gene Counts (normalized)",
        x_label="Xenium (normalized)",
        y_label="MERSCOPE (normalized)",
        log_scale=True,
    )

    qc_records: list[dict[str, float | str]] = []
    qc_by_dataset = {}
    dataset_items = [
        ("XENIUM", cfg.xenium_zarr_path),
        ("MERSCOPE", cfg.merscope_zarr_path),
    ]
    for dataset_name, zarr_path in dataset_items:
        qc = compute_dataset_qc(zarr_path, dataset_name=dataset_name)
        qc_by_dataset[dataset_name] = qc
        qc_records.append(
            {
                "dataset": dataset_name,
                "pct_assigned": float(qc["summary"]["pct_assigned"]),
            }
        )

    geom_plot = cfg.output_dir / f"{cfg.pair_id}_geometry_hist.png"
    cell_plot = cfg.output_dir / f"{cfg.pair_id}_cell_violin.png"
    plot_geometry_histograms_comparison(
        {
            dataset_name: qc["geometry_metrics"]
            for dataset_name, qc in qc_by_dataset.items()
        },
        geom_plot,
    )
    plot_cell_metrics_violin_comparison(
        {
            dataset_name: qc["cell_metrics"]
            for dataset_name, qc in qc_by_dataset.items()
        },
        cell_plot,
    )

    merscope_sdata = sd.read_zarr(cfg.merscope_zarr_path)
    xenium_sdata = sd.read_zarr(cfg.xenium_zarr_path)
    overlay_plot = cfg.output_dir / f"{cfg.pair_id}_sanity_overlay.png"
    plot_pair_sanity_crops(
        merscope_sdata,
        xenium_sdata,
        overlay_plot,
        merscope_zarr_path=cfg.merscope_zarr_path,
        xenium_zarr_path=cfg.xenium_zarr_path,
    )
    transcript_overview_plot = cfg.output_dir / f"{cfg.pair_id}_transcript_overview.png"
    plot_transcript_overview(
        merscope_sdata,
        xenium_sdata,
        transcript_overview_plot,
    )

    assignment_df = pd.DataFrame(qc_records)
    assign_plot = cfg.output_dir / f"{cfg.pair_id}_assignment_rate_bar.png"
    plot_assignment_bar(assignment_df, assign_plot)

    click.echo("Visualization complete:")
    click.echo(f"- {total_scatter}")
    click.echo(f"- {assigned_scatter}")
    click.echo(f"- {geom_plot}")
    click.echo(f"- {cell_plot}")
    click.echo(f"- {overlay_plot}")
    click.echo(f"- {transcript_overview_plot}")
    click.echo(f"- {assign_plot}")
