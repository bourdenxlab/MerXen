"""CLI command for Cellpose-mask image-channel quantification."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.config import MaskImageQuantificationConfig, load_config_from_json
from merxen.mask_image_quantification import run_mask_image_quantification


@click.command(name="mask-image-quantification")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against MaskImageQuantificationConfig.",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Recompute image quantification even when existing outputs are present.",
)
def mask_image_quantification_command(
    config_path: Path,
    force_rerun: bool,
) -> None:
    """Quantify all image channels over final Cellpose masks."""
    cfg = load_config_from_json(config_path, MaskImageQuantificationConfig)
    assert isinstance(cfg, MaskImageQuantificationConfig)

    outputs = run_mask_image_quantification(cfg, force_rerun=force_rerun)
    click.echo("Mask image quantification complete:")
    for key, value in outputs.items():
        click.echo(f"- {key}: {value}")
