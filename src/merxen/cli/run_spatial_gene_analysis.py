"""CLI command for per-gene Squidpy spatial autocorrelation analysis."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.analysis.spatial_gene_analysis import run_spatial_gene_analysis
from merxen.config import SpatialGeneAnalysisConfig, load_config_from_json


@click.command(name="spatial-gene-analysis")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against SpatialGeneAnalysisConfig.",
)
def spatial_gene_analysis_command(config_path: Path) -> None:
    """Run per-gene Moran's I and Geary's C spatial analysis."""
    cfg = load_config_from_json(config_path, SpatialGeneAnalysisConfig)
    assert isinstance(cfg, SpatialGeneAnalysisConfig)

    results = run_spatial_gene_analysis(cfg)

    click.echo("spatial_gene_analysis complete:")
    for sample_id, paths in results.items():
        click.echo(f"- {sample_id}")
        for key, value in paths.items():
            click.echo(f"  - {key}: {value}")
