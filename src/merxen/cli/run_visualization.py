"""CLI command for generating visualization artifacts."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import spatialdata as sd

from merxen.config import (
    VisualizationConfig,
    VisualizationSampleConfig,
    load_config_from_json,
)
from merxen.qc.gene_comparison import (
    compute_gene_comparison_from_paths,
    compute_gene_summary_from_path,
)
from merxen.qc.metrics import compute_dataset_qc
from merxen.visualization.density_overview import (
    plot_single_transcript_overview,
    plot_transcript_overview,
)
from merxen.visualization.gene_scatter import plot_gene_abundance, plot_gene_scatter
from merxen.visualization.qc_plots import (
    plot_assignment_bar,
    plot_cell_metrics_violin,
    plot_cell_metrics_violin_comparison,
    plot_geometry_histograms,
    plot_geometry_histograms_comparison,
)
from merxen.visualization.sanity_plots import (
    plot_pair_sanity_crops,
    plot_single_sanity_crop,
)


@click.command(name="visualize")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against VisualizationConfig.",
)
def visualize_command(config_path: Path) -> None:
    """Generate visualization artifacts for paired or single-platform datasets."""
    cfg = load_config_from_json(config_path, VisualizationConfig)
    assert isinstance(cfg, VisualizationConfig)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    samples_by_platform: dict[str, VisualizationSampleConfig] = {
        sample.platform: sample for sample in cfg.samples
    }
    if {"MERSCOPE", "XENIUM"}.issubset(samples_by_platform):
        paths = _write_paired_visualizations(cfg, samples_by_platform)
    else:
        paths = []
        for sample in cfg.samples:
            paths.extend(_write_single_visualizations(cfg, sample))

    click.echo("Visualization complete:")
    for path in paths:
        click.echo(f"- {path}")


def _write_paired_visualizations(
    cfg: VisualizationConfig,
    samples_by_platform: dict[str, VisualizationSampleConfig],
) -> list[Path]:
    merscope_sample = samples_by_platform["MERSCOPE"]
    xenium_sample = samples_by_platform["XENIUM"]
    paths: list[Path] = []

    comparison = compute_gene_comparison_from_paths(
        xenium_zarr_path=xenium_sample.zarr_path,
        merscope_zarr_path=merscope_sample.zarr_path,
        xenium_table_key=xenium_sample.table_key,
        merscope_table_key=merscope_sample.table_key,
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
    paths.extend([total_scatter, assigned_scatter])

    qc_records: list[dict[str, float | str]] = []
    qc_by_dataset = {}
    dataset_items = [
        ("XENIUM", xenium_sample),
        ("MERSCOPE", merscope_sample),
    ]
    for dataset_name, sample in dataset_items:
        qc = compute_dataset_qc(
            sample.zarr_path,
            dataset_name=dataset_name,
            table_key=sample.table_key,
            shape_key=sample.shape_key,
        )
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
    paths.extend([geom_plot, cell_plot])

    merscope_sdata = sd.read_zarr(merscope_sample.zarr_path)
    xenium_sdata = sd.read_zarr(xenium_sample.zarr_path)
    overlay_plot = cfg.output_dir / f"{cfg.pair_id}_sanity_overlay.png"
    crop_location_plot = overlay_plot.with_name(
        f"{overlay_plot.stem}_crop_location{overlay_plot.suffix}"
    )
    plot_pair_sanity_crops(
        merscope_sdata,
        xenium_sdata,
        overlay_plot,
        merscope_zarr_path=merscope_sample.zarr_path,
        xenium_zarr_path=xenium_sample.zarr_path,
        merscope_assignment_shape_key=merscope_sample.shape_key,
        xenium_assignment_shape_key=xenium_sample.shape_key,
    )
    transcript_overview_plot = cfg.output_dir / f"{cfg.pair_id}_transcript_overview.png"
    plot_transcript_overview(
        merscope_sdata,
        xenium_sdata,
        transcript_overview_plot,
    )
    paths.extend([overlay_plot, crop_location_plot, transcript_overview_plot])

    assignment_df = pd.DataFrame(qc_records)
    assign_plot = cfg.output_dir / f"{cfg.pair_id}_assignment_rate_bar.png"
    plot_assignment_bar(assignment_df, assign_plot)
    paths.append(assign_plot)
    return paths


def _write_single_visualizations(
    cfg: VisualizationConfig,
    sample: VisualizationSampleConfig,
) -> list[Path]:
    dataset_name = sample.platform
    sample_id = sample.sample_id
    paths: list[Path] = []

    gene_summary = compute_gene_summary_from_path(
        sample.zarr_path,
        dataset_name=dataset_name,
        table_key=sample.table_key,
    )
    total_gene_plot = (
        cfg.output_dir / f"{sample_id}_gene_abundance_total_normalized.png"
    )
    assigned_gene_plot = (
        cfg.output_dir / f"{sample_id}_gene_abundance_assigned_normalized.png"
    )
    plot_gene_abundance(
        gene_summary["total_counts_df"],
        total_gene_plot,
        title=f"{sample_id} Total Gene Abundance (normalized)",
    )
    plot_gene_abundance(
        gene_summary["assigned_counts_df"],
        assigned_gene_plot,
        title=f"{sample_id} Assigned Gene Abundance (normalized)",
    )
    paths.extend([total_gene_plot, assigned_gene_plot])

    qc = compute_dataset_qc(
        sample.zarr_path,
        dataset_name=dataset_name,
        table_key=sample.table_key,
        shape_key=sample.shape_key,
    )
    geom_plot = cfg.output_dir / f"{sample_id}_geometry_hist.png"
    cell_plot = cfg.output_dir / f"{sample_id}_cell_violin.png"
    plot_geometry_histograms(qc["geometry_metrics"], geom_plot)
    plot_cell_metrics_violin(qc["cell_metrics"], cell_plot)
    paths.extend([geom_plot, cell_plot])

    sdata_obj = sd.read_zarr(sample.zarr_path)
    overlay_plot = cfg.output_dir / f"{sample_id}_sanity_overlay.png"
    crop_location_plot = overlay_plot.with_name(
        f"{overlay_plot.stem}_crop_location{overlay_plot.suffix}"
    )
    plot_single_sanity_crop(
        sdata_obj,
        dataset_name,
        overlay_plot,
        zarr_path=sample.zarr_path,
        assignment_shape_key=sample.shape_key,
    )
    transcript_overview_plot = cfg.output_dir / f"{sample_id}_transcript_overview.png"
    plot_single_transcript_overview(
        sdata_obj,
        dataset_name,
        transcript_overview_plot,
    )
    paths.extend([overlay_plot, crop_location_plot, transcript_overview_plot])

    assignment_df = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "pct_assigned": float(qc["summary"]["pct_assigned"]),
            }
        ]
    )
    assign_plot = cfg.output_dir / f"{sample_id}_assignment_rate_bar.png"
    plot_assignment_bar(assignment_df, assign_plot)
    paths.append(assign_plot)
    return paths
