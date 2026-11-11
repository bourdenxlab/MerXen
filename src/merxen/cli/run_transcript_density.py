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
@click.option(
    "--embed/--no-embed",
    default=True,
    show_default=True,
    help="Persist the density image and viewer pyramids in the source SpatialData.",
)
def atera_transcript_density_command(config_path: Path, embed: bool) -> None:
    """Generate native-grid density and persist it in SpatialData by default.

    Args:
        config_path: Source store, new output directory and density settings.
        embed: Also install the complete image and display pyramids in the source.
    """
    from merxen.segmentation.transcript_density import create_transcript_density

    config = load_config_from_json(config_path, AteraTranscriptDensityConfig)
    assert isinstance(config, AteraTranscriptDensityConfig)
    if embed:
        from merxen.segmentation.density_storage import prepare_embedded_density

        result = prepare_embedded_density(config)
    else:
        result = create_transcript_density(config)
    click.echo(f"Atera transcript density written: {result}")


@click.command(name="embed-transcript-density")
@click.option(
    "--spatialdata",
    "target",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--density",
    "density_source",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
def embed_transcript_density_command(target: Path, density_source: Path) -> None:
    """Install an existing matching raster and viewer pyramids in SpatialData.

    Args:
        target: Completed native SpatialData store to update additively.
        density_source: Existing matching raster; transcript generation is skipped.
    """
    from merxen.segmentation.density_storage import embed_transcript_density

    result = embed_transcript_density(target, density_source)
    click.echo(f"Permanent transcript density ready: {result}")
