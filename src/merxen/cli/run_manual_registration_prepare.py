"""CLI command for the prepare step of manual landmark registration."""

from __future__ import annotations

import json
from pathlib import Path

import click

from merxen.config import load_config_from_json
from merxen.manual_registration.config import ManualRegistrationPrepareConfig
from merxen.manual_registration.prepare import run_manual_registration_prepare


@click.command(name="manual-registration-prepare")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against ManualRegistrationPrepareConfig.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Rebuild even when the image grid changed under existing landmarks; the "
        "old store is kept as <name>.superseded-<UTC>."
    ),
)
def manual_registration_prepare_command(config_path: Path, force: bool) -> None:
    """Convert or validate both images and build the image-only prepared store."""
    cfg = load_config_from_json(config_path, ManualRegistrationPrepareConfig)
    assert isinstance(cfg, ManualRegistrationPrepareConfig)
    if force:
        cfg = cfg.model_copy(update={"force": True})
    try:
        outputs = run_manual_registration_prepare(cfg)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    summary = json.loads(outputs["summary"].read_text())
    click.echo(f"Prepared store ({summary['status']}): {outputs['store']}")
    for role in ("reference", "moving"):
        record = summary["images"][role]
        how = (
            "written by prepare"
            if record["ome_tiff"]["written_by_prepare"]
            else "used in place"
        )
        click.echo(
            f"- {role}: {record['ome_tiff']['path']} ({how}); "
            f"channels {record['channel_names']}"
        )
    for warning in summary["warnings"]:
        click.echo(f"Warning: {warning}")
    click.echo(f"Summary: {outputs['summary']}")
    click.echo(
        "Open in the viewer: Dataset loader > Manual Registration > Load image "
        f"data > {outputs['store']}"
    )
