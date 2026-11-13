"""Add native density and portable HITL setup to existing Xenium review stores."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import spatialdata as sd
from pydantic import BaseModel, ConfigDict, Field
from spatialdata.transformations import get_transformation

from merxen.config import TranscriptDensityConfig
from merxen.io.image_source import image_to_cyx
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.platforms import platform_spec
from merxen.segmentation.density_storage import (
    embed_transcript_density,
    has_embedded_density,
)
from merxen.segmentation.hitl.export import (
    HitlExportConfig,
    prepare_viewer_roi_context,
)
from merxen.segmentation.transcript_density import (
    _file_tree_identity,
    create_transcript_density,
    density_profile,
    read_density_record,
)
from merxen.viewer_cache.build import build_image_pyramid_caches


class HitlViewerPreparationConfig(BaseModel):
    """Identify an existing review store and its original native build.

    Original builds retain detected micron transcripts, gene/control flags and
    microscope metadata. Processed stores can instead contain ProSeg coordinates
    in pixels, so density is generated from the original build and installed in
    the matching review store. Existing masks, points and annotations are kept.
    """

    model_config = ConfigDict(extra="forbid")

    platform: Literal["XENIUM"] = "XENIUM"
    dataset_alias: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    source_zarr: Path
    native_source_zarr: Path
    baseline_stats_path: Path
    output_dir: Path
    sigma_um: float = Field(default=0.15, gt=0, allow_inf_nan=False)
    min_qv: float = Field(default=20, ge=0, allow_inf_nan=False)
    bin_size_px: int = Field(default=1, ge=1, le=8)
    margin_px: int = Field(default=656, ge=0)


def _install_setup_file(original: Path, destination: Path) -> None:
    if destination.exists():
        if destination.read_bytes() != original.read_bytes():
            raise ValueError(
                "Existing HITL setup differs; preserve it and use a new review store"
            )
        return
    shutil.copy2(original, destination)


def prepare_hitl_viewer(config: HitlViewerPreparationConfig) -> Path:
    """Prepare a Xenium store for local crop selection without resegmentation.

    Args:
        config: Existing review store, original build, baseline grid and output.

    Returns:
        Completion manifest recording density, setup and protected data checks.

    Raises:
        ValueError: Inputs differ, an annotation setup would be replaced, or
            existing masks, annotations or transcript/table payloads change.
    """
    source = config.source_zarr.resolve()
    native_source = config.native_source_zarr.resolve()
    output = config.output_dir.resolve()
    if any(
        output.is_relative_to(root) or root.is_relative_to(output)
        for root in (source, native_source)
    ):
        raise ValueError("Preparation output must be separate from both input stores")
    if not source.is_dir() or not native_source.is_dir():
        raise ValueError("Preparation requires existing review and native stores")
    protected = ("shapes", "labels", "tables", "points")
    complete = output / "preparation.json"
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["config"] != config.model_dump(mode="json"):
            raise ValueError("Preparation output belongs to a different configuration")
        current = {key: _file_tree_identity(source / key) for key in protected}
        if current != saved["protected_data_after"]:
            raise ValueError("Protected review data changed after preparation")
        if read_density_record(source)["fingerprint"] != saved["density_fingerprint"]:
            raise ValueError("Review density changed after preparation")
        return complete
    output.mkdir(parents=True, exist_ok=True)
    before = {key: _file_tree_identity(source / key) for key in protected}
    backup = output / "annotation_backup"
    if not backup.exists():
        backup.mkdir()
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, backup / path.name)
        if (source / "shapes").is_dir():
            shutil.copytree(
                source / "shapes", backup / "shapes", copy_function=shutil.copy2
            )
        shutil.copy2(
            config.baseline_stats_path, backup / "cellpose_stitching_stats.json"
        )
    spec = native_source / "experiment.xenium"
    if not spec.is_file():
        raise ValueError("The native Xenium build must retain experiment.xenium")
    _install_setup_file(spec, source / spec.name)
    # Old processing copies rewrote chunk timestamps. Compare coordinate grids
    # rather than treating those copies as immutable raster payloads.
    image_key = platform_spec(config.platform).native_image_key
    native_image = sd.read_zarr(native_source, selection=("images",)).images[image_key]
    review_image = sd.read_zarr(source, selection=("images",)).images[image_key]
    native_grid = get_transformation(native_image, to_coordinate_system="global")
    review_grid = get_transformation(review_image, to_coordinate_system="global")
    if image_to_cyx(native_image).shape != image_to_cyx(
        review_image
    ).shape or not np.allclose(
        native_grid.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        review_grid.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Native transcript source and review image grids differ")
    density_config = TranscriptDensityConfig(
        source_zarr_path=source,
        transcripts_source_zarr_path=native_source,
        output_dir=output / "density",
        platform="XENIUM",
        sigma_um=config.sigma_um,
        min_qv=config.min_qv,
        bin_size_px=config.bin_size_px,
    )
    if has_embedded_density(source):
        record = read_density_record(source)
        if (
            record["profile"] != density_profile(density_config)
            or record.get("platform") != config.platform
        ):
            raise ValueError("Existing density has a different platform or recipe")
        embed_transcript_density(source, source)
    else:
        raster = density_config.output_dir / "transcript_density.zarr"
        if not raster.exists():
            raster = create_transcript_density(density_config)
        embed_transcript_density(source, raster)
        record = read_density_record(source)
        # Only this task's temporary raster is removed; the permanent payload
        # is now an independent verified image inside the review store.
        shutil.rmtree(raster)
    cache_summary = build_image_pyramid_caches(source)
    setup = output / "setup"
    setup.mkdir(exist_ok=True)
    roi_path = setup / "roi_context.json"
    export_config = HitlExportConfig(
        source_zarr=source,
        baseline_stats_path=config.baseline_stats_path,
        output_dir=output / "annotation_export",
        protected_roots=[source, native_source],
        platform="XENIUM",
        dataset_alias=config.dataset_alias,
        mask_key="MOSAIK_cellpose_labels",
        channels=["DAPI", "18S"],
        margin_px=config.margin_px,
    )
    if not roi_path.exists():
        context = prepare_viewer_roi_context(export_config, roi_path)
    else:
        from merxen.segmentation.hitl.regions import RoiContext

        context = RoiContext(**json.loads(roi_path.read_text()))
        metadata = next(
            source / name
            for name in ("zarr.json", ".zattrs")
            if (source / name).is_file()
        )
        if (
            hashlib.sha256(metadata.read_bytes()).hexdigest()
            != context.source_metadata_sha256
        ):
            raise ValueError(
                "Review metadata changed after crop setup; retain the old setup"
            )
    with spatialdata_write_lock(source):
        _install_setup_file(roi_path, source / "merxen_hitl_roi_context.json")
        _install_setup_file(
            setup / "hitl_export_settings.json", source / "hitl_export_settings.json"
        )
    after = {key: _file_tree_identity(source / key) for key in protected}
    if after != before:
        raise ValueError(
            "Existing masks, annotations, tables or transcript data changed"
        )
    manifest = {
        "schema_version": 1,
        "config": config.model_dump(mode="json"),
        "context": asdict(context),
        "model_channels": ["DAPI", "18S"],
        "density_usage": "viewer_only",
        "density_fingerprint": record["fingerprint"],
        "density_totals": record["totals"],
        "image_caches": cache_summary,
        "protected_data_before": before,
        "protected_data_after": after,
        "annotation_backup": str(backup),
    }
    complete.write_text(json.dumps(manifest, indent=2) + "\n")
    return complete
