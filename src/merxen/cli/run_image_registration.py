"""CLI command for registering external images onto the mask grid."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.config import ImageRegistrationConfig, load_config_from_json
from merxen.image_registration import run_image_registration


@click.command(name="register-images")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against ImageRegistrationConfig.",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Re-register images even when an identical registration exists.",
)
def register_images_command(config_path: Path, force_rerun: bool) -> None:
    """Register OME-TIFF images with Xenium Explorer matrices onto the mask grid."""
    cfg = load_config_from_json(config_path, ImageRegistrationConfig)
    assert isinstance(cfg, ImageRegistrationConfig)

    outputs = run_image_registration(cfg, force_rerun=force_rerun)
    click.echo("Image registration complete:")
    for key, value in outputs.items():
        click.echo(f"- {key}: {value}")
