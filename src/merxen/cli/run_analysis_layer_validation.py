"""CLI command for selected downstream analysis-layer validation."""

from __future__ import annotations

import json
from pathlib import Path

import click

from merxen.analysis_layers import validate_analysis_layer


@click.command(name="validate-analysis-layer")
@click.option(
    "--zarr",
    "zarr_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option("--platform", required=True)
@click.option("--segmentation", required=True)
@click.option("--table-key", required=True)
@click.option("--shape-key", required=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
)
def validate_analysis_layer_command(
    zarr_path: Path,
    platform: str,
    segmentation: str,
    table_key: str,
    shape_key: str,
    output_path: Path,
) -> None:
    """Fail early when a selected downstream layer is incomplete."""
    summary = validate_analysis_layer(
        zarr_path,
        platform=platform,
        segmentation=segmentation,
        table_key=table_key,
        shape_key=shape_key,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    click.echo(f"Validated {platform}:{segmentation} ({summary['n_cells']:,} cells)")
