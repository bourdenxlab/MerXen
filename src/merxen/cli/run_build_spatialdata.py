"""CLI command for building platform-specific SpatialData zarrs."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.config import SpatialDataBuildConfig, load_config_from_json
from merxen.io.builders.pipeline import build_spatialdata_artifact


@click.command(name="build-spatialdata")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against SpatialDataBuildConfig.",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Rebuild the SpatialData object even when an existing zarr is available.",
)
def build_spatialdata_command(config_path: Path, force_rerun: bool) -> None:
    """Build or reuse SpatialData from MERSCOPE, VZG2, or Xenium input."""
    cfg = load_config_from_json(config_path, SpatialDataBuildConfig)
    assert isinstance(cfg, SpatialDataBuildConfig)
    output_path = build_spatialdata_artifact(cfg, force_rerun=force_rerun)
    click.echo(f"SpatialData build complete: {output_path}")
