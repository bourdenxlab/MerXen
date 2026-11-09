"""Pre-segmentation Atera transcript-density command."""

from pathlib import Path

import click

from merxen.config import AteraTranscriptDensityConfig, load_config_from_json


@click.command(name="atera-transcript-density")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
def atera_transcript_density_command(config_path: Path) -> None:
    """Generate an independent native-grid gene-density image.

    Args:
        config_path: Source store, new output directory and density settings.
    """
    from merxen.segmentation.transcript_density import create_transcript_density

    config = load_config_from_json(config_path, AteraTranscriptDensityConfig)
    assert isinstance(config, AteraTranscriptDensityConfig)
    result = create_transcript_density(config)
    click.echo(f"Atera transcript density written: {result}")
