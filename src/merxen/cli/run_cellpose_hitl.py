"""Commands for exporting human-correctable segmentation pilots."""

from __future__ import annotations

from pathlib import Path

import click

from merxen.config import load_config_from_json
from merxen.segmentation.hitl.export import (
    HitlExportConfig,
    export_annotation_pilot,
    prepare_viewer_roi_context,
)


@click.command(name="cellpose-hitl-export")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="JSON source, mask, tile-grid and annotation-pilot settings.",
)
def cellpose_hitl_export_command(config_path: Path) -> None:
    """Export a tissue-spanning pilot for correction in the Cellpose GUI.

    Args:
        config_path: JSON configuration validated before reading source arrays.
    """
    config = load_config_from_json(config_path, HitlExportConfig)
    assert isinstance(config, HitlExportConfig)
    result = export_annotation_pilot(config)
    click.echo(f"Exported {result.crop_count} annotation crops: {result.package_path}")


@click.command(name="cellpose-hitl-prepare-rois")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
def cellpose_hitl_prepare_rois_command(config_path: Path, output_path: Path) -> None:
    """Prepare a portable context file for choosing crop regions in napari.

    Args:
        config_path: Source and original grid/crop settings.
        output_path: New context file; the source store stays read only.
    """
    config = load_config_from_json(config_path, HitlExportConfig)
    assert isinstance(config, HitlExportConfig)
    prepare_viewer_roi_context(config, output_path)
    click.echo(f"Viewer ROI context ready: {output_path}")
