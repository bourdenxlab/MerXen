"""CLI commands for Cellpose and ProSeg segmentation."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from merxen.config import SegmentationConfig, load_config_from_json
from merxen.segmentation.pipeline import (
    run_cellpose_segmentation,
    run_proseg_segmentation,
    run_segmentation_pipeline,
)

logger = logging.getLogger(__name__)


@click.command(name="segment")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against SegmentationConfig.",
)
@click.option(
    "--force-rerun",
    is_flag=True,
    default=False,
    help="Recompute outputs even when existing zarr outputs are present.",
)
def segment_command(config_path: Path, force_rerun: bool) -> None:
    """Run unified segmentation for one dataset."""
    cfg = load_config_from_json(config_path, SegmentationConfig)
    assert isinstance(cfg, SegmentationConfig)
    outputs = run_segmentation_pipeline(cfg, force_rerun=force_rerun)
    click.echo("Segmentation complete:")
    for key, value in outputs.items():
        click.echo(f"- {key}: {value}")


@click.command(name="cellpose-segment")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against SegmentationConfig.",
)
@click.option("--force-rerun", is_flag=True, default=False)
def cellpose_segment_command(config_path: Path, force_rerun: bool) -> None:
    """Run Cellpose segmentation and prepare the ProSeg inputs."""
    cfg = load_config_from_json(config_path, SegmentationConfig)
    assert isinstance(cfg, SegmentationConfig)
    outputs = run_cellpose_segmentation(cfg, force_rerun=force_rerun)
    click.echo("Cellpose segmentation complete:")
    for key, value in outputs.items():
        click.echo(f"- {key}: {value}")


@click.command(name="proseg-segment")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to JSON config validated against SegmentationConfig.",
)
@click.option(
    "--transcripts-csv",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--cellpose-mask",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--cellpose-transforms",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--proseg-binary",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option("--force-rerun", is_flag=True, default=False)
def proseg_segment_command(
    config_path: Path,
    transcripts_csv: Path,
    cellpose_mask: Path,
    cellpose_transforms: Path,
    proseg_binary: Path | None,
    force_rerun: bool,
) -> None:
    """Run CPU-only ProSeg refinement from prepared Cellpose artifacts."""
    cfg = load_config_from_json(config_path, SegmentationConfig)
    assert isinstance(cfg, SegmentationConfig)
    outputs = run_proseg_segmentation(
        cfg,
        transcripts_csv=transcripts_csv,
        cellpose_mask_path=cellpose_mask,
        transforms_path=cellpose_transforms,
        proseg_binary=proseg_binary,
        force_rerun=force_rerun,
    )
    click.echo("ProSeg segmentation complete:")
    for key, value in outputs.items():
        click.echo(f"- {key}: {value}")
