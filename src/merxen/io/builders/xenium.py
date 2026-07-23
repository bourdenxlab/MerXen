"""Xenium raw-folder to SpatialData writer."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from merxen.config import XeniumBuildConfig
from merxen.io.spatialdata_io import (
    prepare_source_spatialdata_contract,
    write_spatialdata_zarr,
)

logger = logging.getLogger(__name__)


def write_xenium_spatialdata(
    *,
    input_path: Path,
    output_path: Path,
    build_config: XeniumBuildConfig,
    xenium_spec_path: Path | None = None,
) -> Path:
    """Build a Xenium SpatialData zarr from a raw Xenium output folder.

    Args:
        input_path: Xenium output directory containing the raw files.
        output_path: Destination zarr path.
        build_config: Platform-specific reader options.
        xenium_spec_path: Optional explicit spec file to copy into the output
            zarr for later transform inference.

    Returns:
        The written zarr path.
    """
    from spatialdata_io import xenium as xenium_reader

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[XENIUM] Building SpatialData from %s", input_path)
    sdata = xenium_reader(
        input_path,
        cells_boundaries=build_config.cells_boundaries,
        nucleus_boundaries=build_config.nucleus_boundaries,
        cells_as_circles=build_config.cells_as_circles,
        cells_labels=build_config.cells_labels,
        nucleus_labels=build_config.nucleus_labels,
        transcripts=build_config.transcripts,
        morphology_mip=build_config.morphology_mip,
        morphology_focus=build_config.morphology_focus,
        aligned_images=build_config.aligned_images,
        cells_table=build_config.cells_table,
    )
    prepare_source_spatialdata_contract(sdata, platform="XENIUM")
    write_spatialdata_zarr(sdata, output_path, overwrite=True)
    _copy_xenium_sidecars(
        input_path=input_path,
        output_path=output_path,
        xenium_spec_path=xenium_spec_path,
    )
    return output_path


def _copy_xenium_sidecars(
    *,
    input_path: Path,
    output_path: Path,
    xenium_spec_path: Path | None,
) -> None:
    """Copy transform/spec metadata into the built zarr for downstream reuse."""
    candidates: list[tuple[Path, Path]] = []
    if xenium_spec_path is not None:
        candidate = Path(xenium_spec_path)
        if candidate.exists():
            if candidate.suffix.lower() == ".json":
                candidates.append((candidate, output_path / "experiment.xenium"))
            else:
                candidates.append((candidate, output_path / candidate.name))

    defaults = [
        (input_path / "experiment.xenium", output_path / "experiment.xenium"),
        (input_path / "specs.json", output_path / "specs.json"),
        (input_path / "specs" / "specs.json", output_path / "specs" / "specs.json"),
    ]
    candidates.extend(defaults)

    for source_path, dest_path in candidates:
        if not source_path.exists():
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
