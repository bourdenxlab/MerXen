"""CLI commands for cross-section alignment."""

from __future__ import annotations

import json
from pathlib import Path

import click

from merxen.alignment.dependencies import check_alignment_dependencies
from merxen.alignment.materialize import materialize_alignment
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


@click.command(name="materialize-alignment")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against AlignmentConfig.",
)
@click.option(
    "--summary",
    "summary_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Optional path for a machine-readable JSON summary.",
)
def materialize_alignment_command(
    config_path: Path,
    summary_path: Path | None,
) -> None:
    """Reconcile aligned vectors and fixed-grid rasters from a saved transform."""
    cfg = load_config_from_json(config_path, AlignmentConfig)
    assert isinstance(cfg, AlignmentConfig)
    summary = materialize_alignment(cfg)
    encoded = json.dumps(summary, indent=2)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(f"{encoded}\n")
    click.echo(encoded)


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
@click.option(
    "--backend",
    type=click.Choice(["valis", "legacy_spateo"], case_sensitive=False),
    default="valis",
    show_default=True,
)
def check_alignment_deps_command(backend: str) -> None:
    """Verify that the selected alignment backend dependencies import."""
    status = check_alignment_dependencies(backend)
    if not status.ok:
        raise click.ClickException(status.message)

    click.echo(status.message)
    for package, package_version in status.versions.items():
        click.echo(f"- {package}: {package_version}")
