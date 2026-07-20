"""CLI commands for mutually exclusive co-expression rate analysis."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.analysis.mecr import run_mecr, run_mecr_reference
from merxen.config import (
    MecrConfig,
    MecrReferenceConfig,
    load_config_from_json,
)


@click.command(name="mecr-reference")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against MecrReferenceConfig.",
)
def mecr_reference_command(config_path: Path) -> None:
    """Discover paper-standard mutually exclusive WHB marker genes."""
    config = load_config_from_json(config_path, MecrReferenceConfig)
    assert isinstance(config, MecrReferenceConfig)
    outputs = run_mecr_reference(config)
    click.echo("MECR reference preparation complete:")
    for name, path in outputs.items():
        click.echo(f"- {name}: {path}")


@click.command(name="mecr")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against MecrConfig.",
)
def mecr_command(config_path: Path) -> None:
    """Score mutually exclusive co-expression in spatial cell tables."""
    config = load_config_from_json(config_path, MecrConfig)
    assert isinstance(config, MecrConfig)
    outputs = run_mecr(config)
    click.echo("MECR analysis complete:")
    for name, path in outputs.items():
        click.echo(f"- {name}: {path}")
