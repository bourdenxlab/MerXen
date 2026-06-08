"""CLI command for cross-platform gene comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from merxen.config import ComparisonConfig, load_config_from_json
from merxen.qc.gene_comparison import compute_gene_comparison_from_paths


def _to_serializable_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON-serializable metrics from comparison output."""
    return {
        "totals": result["totals"],
        "table_keys": result.get("table_keys", {}),
        "fits": result["fits"],
        "n_genes": {
            "total_counts": int(len(result["total_counts_df"])),
            "assigned_counts": int(len(result["assigned_counts_df"])),
            "total_normalized": int(len(result["total_normalized_df"])),
            "assigned_normalized": int(len(result["assigned_normalized_df"])),
        },
    }


@click.command(name="compare")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against ComparisonConfig.",
)
def compare_command(config_path: Path) -> None:
    """Run cross-platform gene-level comparison and write summary artifacts."""
    cfg = load_config_from_json(config_path, ComparisonConfig)
    assert isinstance(cfg, ComparisonConfig)

    result = compute_gene_comparison_from_paths(
        xenium_zarr_path=cfg.xenium_zarr_path,
        merscope_zarr_path=cfg.merscope_zarr_path,
        xenium_table_key=cfg.xenium_table_key,
        merscope_table_key=cfg.merscope_table_key,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    total_counts_csv = cfg.output_dir / f"{cfg.pair_id}_total_counts_compare.csv"
    assigned_counts_csv = cfg.output_dir / f"{cfg.pair_id}_assigned_counts_compare.csv"
    total_norm_csv = cfg.output_dir / f"{cfg.pair_id}_total_normalized_compare.csv"
    assigned_norm_csv = (
        cfg.output_dir / f"{cfg.pair_id}_assigned_normalized_compare.csv"
    )
    metrics_json = cfg.output_dir / f"{cfg.pair_id}_comparison_metrics.json"

    result["total_counts_df"].to_csv(total_counts_csv, index=False)
    result["assigned_counts_df"].to_csv(assigned_counts_csv, index=False)
    result["total_normalized_df"].to_csv(total_norm_csv, index=False)
    result["assigned_normalized_df"].to_csv(assigned_norm_csv, index=False)
    metrics_json.write_text(json.dumps(_to_serializable_metrics(result), indent=2))

    click.echo("Comparison complete:")
    click.echo(f"- total_counts: {total_counts_csv}")
    click.echo(f"- assigned_counts: {assigned_counts_csv}")
    click.echo(f"- total_normalized: {total_norm_csv}")
    click.echo(f"- assigned_normalized: {assigned_norm_csv}")
    click.echo(f"- metrics: {metrics_json}")
