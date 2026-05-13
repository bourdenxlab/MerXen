"""CLI command for local MapMyCells cell type assignment."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.analysis.mapmycells import run_mapmycells
from merxen.config import MapMyCellsConfig, load_config_from_json


@click.command(name="mapmycells")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against MapMyCellsConfig.",
)
def mapmycells_command(config_path: Path) -> None:
    """Run local MapMyCells annotation on clustered AnnData outputs."""
    cfg = load_config_from_json(config_path, MapMyCellsConfig)
    assert isinstance(cfg, MapMyCellsConfig)

    results = run_mapmycells(cfg)

    click.echo("mapmycells complete:")
    for sample_id, reference_results in results.items():
        click.echo(f"- {sample_id}")
        for reference_name, paths in reference_results.items():
            click.echo(f"  - {reference_name}")
            for key, value in paths.items():
                click.echo(f"    - {key}: {value}")
