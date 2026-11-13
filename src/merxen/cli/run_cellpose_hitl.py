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


@click.command(name="cellpose-hitl-prepare-viewer")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
def cellpose_hitl_prepare_viewer_command(config_path: Path) -> None:
    """Add viewing density and crop setup while retaining existing Xenium masks.

    Args:
        config_path: Existing review store, original build and baseline tile grid.
    """
    from merxen.segmentation.hitl.preparation import (
        HitlViewerPreparationConfig,
        prepare_hitl_viewer,
    )

    config = load_config_from_json(config_path, HitlViewerPreparationConfig)
    assert isinstance(config, HitlViewerPreparationConfig)
    result = prepare_hitl_viewer(config)
    click.echo(f"HITL viewer preparation complete: {result}")


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


@click.command(name="cellpose-hitl-train")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option("--output", "output_dir", type=click.Path(path_type=Path), required=True)
@click.option("--model-name", default=None)
@click.option("--species", type=click.Choice(["human", "mouse"]), default=None)
def cellpose_hitl_train_command(
    config_path: Path, output_dir: Path, model_name: str | None, species: str | None
) -> None:
    """Validate returned crops and publish a uniquely named candidate model.

    Args:
        config_path: Server root, package IDs, model name and training recipe.
        output_dir: Directory for the downstream model selector.
        model_name: Unique name overriding the viewer's empty template field.
        species: Species when using the viewer's portable template directly.
    """
    import json

    from merxen.segmentation.hitl.training import (
        CellposeTrainingConfig,
        train_cellpose_model,
    )

    values = json.loads(config_path.read_text())
    if model_name is not None:
        values["model_name"] = model_name
    if species is not None:
        values["species"] = species
    config = CellposeTrainingConfig.model_validate(values)
    result = train_cellpose_model(config, output_dir)
    click.echo(f"Candidate model {result.model_name}: {result.bundle_path}")


@click.command(name="cellpose-hitl-seed")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
def cellpose_hitl_seed_command(config_path: Path) -> None:
    """Copy processed inputs into a fresh custom-model run.

    Args:
        config_path: Original processed store, fresh destination and protected roots.
    """
    from merxen.segmentation.hitl.seed_inputs import HitlSeedConfig, seed_hitl_inputs

    result = seed_hitl_inputs(
        HitlSeedConfig.model_validate_json(config_path.read_text())
    )
    click.echo(f"Independent HITL inputs ready: {result.source_zarr}")


@click.command(name="cellpose-hitl-import")
@click.option("--hitl-root", type=click.Path(path_type=Path), default=None)
@click.option("--package", "package_ids", multiple=True, required=True)
def cellpose_hitl_import_command(
    hitl_root: Path | None, package_ids: tuple[str, ...]
) -> None:
    """Validate returned numeric masks without starting GPU training.

    Args:
        hitl_root: Configured HITL root, or MERXEN_HITL_ROOT from the project loader.
        package_ids: Package folder names below annotations/.
    """
    import json
    import re

    from merxen import local_config
    from merxen.segmentation.hitl.import_annotations import import_annotation_packages

    root = hitl_root or local_config.require_path("MERXEN_HITL_ROOT")
    if any(not re.fullmatch(r"hitl-[A-Za-z0-9_-]{1,80}", name) for name in package_ids):
        raise click.BadParameter("Use package IDs from the viewer completion message.")
    corpus = import_annotation_packages(
        [root / "annotations" / name for name in package_ids]
    )
    click.echo(
        json.dumps(
            {
                "packages": list(corpus.package_ids),
                "crops": len(corpus.crops),
                "cells": corpus.cell_count,
                "training_set_sha256": corpus.fingerprint,
            },
            indent=2,
        )
    )
