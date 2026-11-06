"""CLI commands of the ATERA ProSeg route (``segmentation_source=native_proseg``)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from merxen.config import (
    AteraProsegPlanConfig,
    AteraProsegStitchConfig,
    AteraProsegTileConfig,
    load_config_from_json,
)

_CONFIG = click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to the JSON config of the task.",
)


@click.command(name="atera-proseg-plan")
@_CONFIG
def atera_proseg_plan_command(config_path: Path) -> None:
    """Write the tile plan of an ATERA section (ATERA_PROSEG_PLAN)."""
    from merxen.segmentation.atera_proseg.plan import write_plan

    cfg = load_config_from_json(config_path, AteraProsegPlanConfig)
    assert isinstance(cfg, AteraProsegPlanConfig)
    path = write_plan(cfg)
    click.echo(f"Tile plan written: {path}")


@click.command(name="atera-proseg-tile")
@_CONFIG
@click.option(
    "--proseg-binary",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="The ProSeg 3.2.0 executable.",
)
def atera_proseg_tile_command(config_path: Path, proseg_binary: Path) -> None:
    """Run ProSeg on one tile and reduce it to its owned cells."""
    from merxen.segmentation.atera_proseg.tile import run_tile

    cfg = load_config_from_json(config_path, AteraProsegTileConfig)
    assert isinstance(cfg, AteraProsegTileConfig)
    manifest = run_tile(cfg, proseg_binary)
    click.echo(
        f"Tile {cfg.tile_id}: {manifest['cells']['n_owned_proseg_cells']} owned cells"
    )


@click.command(name="atera-proseg-stitch")
@_CONFIG
def atera_proseg_stitch_command(config_path: Path) -> None:
    """Stitch the tiles into the latest store's ProSeg layer."""
    from merxen.segmentation.atera_proseg.stitch import stitch

    cfg = load_config_from_json(config_path, AteraProsegStitchConfig)
    assert isinstance(cfg, AteraProsegStitchConfig)
    report = stitch(cfg)
    summary = {
        key: report.get(key)
        for key in (
            "skipped",
            "n_cells",
            "n_zero_transcript_cells",
            "n_rows_assigned",
            "n_claim_conflicts",
        )
    }
    click.echo(f"ProSeg layer: {json.dumps(summary)}")
