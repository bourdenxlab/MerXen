"""CLI commands for distance-from-object analysis."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.config import (
    DistanceFromObjectCohortConfig,
    DistanceFromObjectConfig,
    load_config_from_json,
)
from merxen.distance_from_object.pipeline import (
    run_distance_from_object,
    run_distance_from_object_cohort,
)


@click.command(name="distance-from-object")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against DistanceFromObjectConfig.",
)
def distance_from_object_command(config_path: Path) -> None:
    """Assign cell distances and create pair-level pseudobulk counts."""
    config = load_config_from_json(config_path, DistanceFromObjectConfig)
    assert isinstance(config, DistanceFromObjectConfig)
    paths = run_distance_from_object(config)
    click.echo("Distance-from-object annotation complete:")
    for key, value in paths.items():
        click.echo(f"- {key}: {value}")


@click.command(name="distance-from-object-cohort")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against DistanceFromObjectCohortConfig.",
)
def distance_from_object_cohort_command(config_path: Path) -> None:
    """Run cohort-level paired PyDESeq2 near-vs-far analysis."""
    config = load_config_from_json(config_path, DistanceFromObjectCohortConfig)
    assert isinstance(config, DistanceFromObjectCohortConfig)
    paths = run_distance_from_object_cohort(config)
    click.echo("Distance-from-object cohort analysis complete:")
    for key, value in paths.items():
        click.echo(f"- {key}: {value}")
