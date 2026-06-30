"""CLI command for cortical-depth coordinate computation."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.config import CorticalDepthConfig, load_config_from_json
from merxen.cortical_depth.pipeline import run_cortical_depth


@click.command(name="compute-cortical-depth")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against CorticalDepthConfig.",
)
def cortical_depth_command(config_path: Path) -> None:
    """Compute cortical-depth coordinates for one SpatialData sample."""
    cfg = load_config_from_json(config_path, CorticalDepthConfig)
    assert isinstance(cfg, CorticalDepthConfig)

    paths = run_cortical_depth(cfg)
    click.echo("Cortical-depth computation complete:")
    for key, value in paths.items():
        click.echo(f"- {key}: {value}")
