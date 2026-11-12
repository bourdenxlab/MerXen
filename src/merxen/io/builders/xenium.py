"""Xenium raw-folder to SpatialData writer."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from merxen.config import XeniumBuildConfig
from merxen.io.spatialdata_io import (
    prepare_source_spatialdata_contract,
    write_spatialdata_zarr,
)

if TYPE_CHECKING:
    from packaging.version import Version

logger = logging.getLogger(__name__)

#: Gene panels above this size are whole-transcriptome (Atera WTA has 18,049
#: genes); the Xenium path has none of the scale guards they need. Xenium
#: Prime 5K panels (about 5,000 genes plus add-ons) stay below it.
MAX_XENIUM_PANEL_GENES: Final = 10_000

#: Channel name of the nuclear stain, the only stain of pre-2.0 Xenium images.
DAPI_CHANNEL: Final = "DAPI"

#: Images that pre-2.0 Xenium writes as single-channel DAPI without names.
PRE_V2_MORPHOLOGY_IMAGES: Final = ("morphology_focus", "morphology_mip")


def check_not_atera_preview(input_path: Path) -> None:
    """Refuse an Atera "preview" bundle laid out like a Xenium output.

    Such bundles carry ``experiment.xenium`` but are Atera runs: the onboard
    analysis version starts with ``atera-``, or the panel is whole-transcriptome.
    They need the ATERA build (plan §3.10, open question Q16).

    Args:
        input_path: Xenium output directory.

    Raises:
        ValueError: If the bundle is an Atera run.
    """
    try:
        experiment = json.loads((Path(input_path) / "experiment.xenium").read_text())
    except (OSError, ValueError):
        return
    if not isinstance(experiment, dict):
        return
    version = str(experiment.get("analysis_sw_version", ""))
    n_genes = 0
    for key in ("panel_num_targets_predesigned", "panel_num_targets_custom"):
        value = experiment.get(key)
        if isinstance(value, int | float):
            n_genes += int(value)
    if version.startswith("atera-") or n_genes > MAX_XENIUM_PANEL_GENES:
        raise ValueError(
            "This Xenium-layout bundle is an Atera run "
            f"(analysis_sw_version {version!r}, {n_genes} panel targets). "
            "Atera preview bundles are not supported yet; use analysis_mode "
            "atera with an Atera ROI folder (plan open question Q16)."
        )


def _xenium_analyzer_version(input_path: Path) -> Version | None:
    """Return the onboard-analysis version as spatialdata-io parses it.

    The reader's own parser is used so the version (including a Xenium Ranger
    re-segmentation version, which takes precedence) matches the branch the
    reader took when it named the image channels.

    Args:
        input_path: Xenium output directory.

    Returns:
        The parsed version, or ``None`` when ``experiment.xenium`` is missing,
        unreadable or has a version the reader cannot parse. For ``None`` the
        reader assumes the latest layout for labels and tables but reads the
        pre-2.0 single-file morphology images.
    """
    from spatialdata_io.readers.xenium import _parse_version_of_xenium_analyzer

    try:
        specs = json.loads((Path(input_path) / "experiment.xenium").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(specs, dict):
        return None
    try:
        version: Version | None = _parse_version_of_xenium_analyzer(specs)
    except (KeyError, TypeError):
        return None
    return version


def name_pre_v2_dapi_channels(sdata: Any, input_path: Path) -> list[str]:
    """Name the single channel of pre-2.0 morphology images ``DAPI``.

    Xenium Onboard Analysis before 2.0 writes one DAPI-only
    ``morphology_focus.ome.tif`` (and ``morphology_mip.ome.tif``) without
    channel names, so spatialdata-io labels the channel ``0``. MerXen selects
    the nuclear channel by name (``DAPI``), so the channel is renamed here. An
    image with several channels, or one already named otherwise, is left as
    read; version 2.0 and later name their channels themselves. An
    unparsable version is treated as pre-2.0, because spatialdata-io 0.6 then
    reads the pre-2.0 single-file images too.

    Args:
        sdata: SpatialData object returned by the Xenium reader; changed in
            place.
        input_path: Xenium output directory it was read from.

    Returns:
        The image keys that were renamed.
    """
    version = _xenium_analyzer_version(input_path)
    if version is not None and version.major >= 2:
        return []
    from spatialdata.models import get_channel_names, set_channel_names

    renamed: list[str] = []
    for key in PRE_V2_MORPHOLOGY_IMAGES:
        if key not in sdata.images:
            continue
        names = list(get_channel_names(sdata.images[key]))
        if len(names) != 1 or str(names[0]) != "0":
            continue
        sdata.images[key] = set_channel_names(sdata.images[key], [DAPI_CHANNEL])
        renamed.append(key)
    if renamed:
        logger.info(
            "[XENIUM] Named the single channel of %s %r (analysis version %s)",
            ", ".join(renamed),
            DAPI_CHANNEL,
            version,
        )
    return renamed


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

    check_not_atera_preview(input_path)
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
    name_pre_v2_dapi_channels(sdata, input_path)
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
