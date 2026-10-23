"""CLI for preparing species- and region-matched pciSeq references."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.config import PciSeqReferenceConfig, load_config_from_json


@click.command(name="prepare-pciseq-reference")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="JSON configuration validated against PciSeqReferenceConfig.",
)
def prepare_pciseq_reference_command(config_path: Path) -> None:
    """Build reusable cell-type profiles from an Allen atlas or custom matrix."""
    from merxen.analysis.pciseq_reference import prepare_pciseq_reference

    config = load_config_from_json(config_path, PciSeqReferenceConfig)
    assert isinstance(config, PciSeqReferenceConfig)
    results = prepare_pciseq_reference(config, config.output_dir)
    click.echo("pciSeq reference prepared:")
    for key, path in results.items():
        click.echo(f"- {key}: {path}")
