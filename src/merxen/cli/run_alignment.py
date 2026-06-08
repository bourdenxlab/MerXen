"""CLI commands for cross-section alignment."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.alignment.dependencies import check_alignment_dependencies
from merxen.alignment.pipeline import run_alignment_pipeline
from merxen.alignment.qc import run_alignment_qc
from merxen.config import AlignmentConfig, AlignmentQCConfig, load_config_from_json


@click.command(name="align")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against AlignmentConfig.",
)
def align_command(config_path: Path) -> None:
    """Align a MERSCOPE section into paired Xenium xy coordinates."""
    cfg = load_config_from_json(config_path, AlignmentConfig)
    assert isinstance(cfg, AlignmentConfig)
    paths = run_alignment_pipeline(cfg)

    click.echo("Alignment complete:")
    for key, value in paths.items():
        click.echo(f"- {key}: {value}")


@click.command(name="alignment-qc")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against AlignmentQCConfig.",
)
def alignment_qc_command(config_path: Path) -> None:
    """Compute post-alignment QC metrics and overlays."""
    cfg = load_config_from_json(config_path, AlignmentQCConfig)
    assert isinstance(cfg, AlignmentQCConfig)
    paths = run_alignment_qc(cfg)

    click.echo("Alignment QC complete:")
    for key, value in paths.items():
        click.echo(f"- {key}: {value}")


@click.command(name="check-alignment-deps")
def check_alignment_deps_command() -> None:
    """Verify that optional Spateo alignment dependencies import."""
    status = check_alignment_dependencies()
    if not status.ok:
        raise click.ClickException(status.message)

    click.echo(status.message)
    for package, package_version in status.versions.items():
        click.echo(f"- {package}: {package_version}")
