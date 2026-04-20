"""Orchestration for building reusable SpatialData artifacts."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from merxen.config import SpatialDataBuildConfig

logger = logging.getLogger(__name__)


def build_spatialdata_artifact(
    config: SpatialDataBuildConfig,
    *,
    force_rerun: bool = False,
) -> Path:
    """Build or reuse one platform-specific SpatialData zarr.

    Args:
        config: Validated build configuration.
        force_rerun: When True, rebuild from raw input even if an existing
            SpatialData artifact is available.

    Returns:
        Path to the staged SpatialData zarr that downstream workflow steps
        should consume.

    Raises:
        FileNotFoundError: If neither a reusable SpatialData artifact nor a raw
            input folder exists.
        ValueError: If a force rebuild is requested but only an existing zarr
            was provided and no raw input folder is available.
    """
    output_path = Path(config.output_path)
    persistent_output_path = (
        Path(config.persistent_output_path)
        if config.persistent_output_path is not None
        else None
    )
    input_path = Path(config.input_path)

    reusable_source = _find_reusable_source(
        input_path=input_path,
        persistent_output_path=persistent_output_path,
    )
    if reusable_source is not None and not force_rerun:
        logger.info(
            "[%s] Reusing existing SpatialData artifact: %s",
            config.dataset_name,
            reusable_source,
        )
        _stage_existing_output(reusable_source, output_path)
        return output_path

    if output_path.exists() and not force_rerun:
        logger.info(
            "[%s] Reusing staged SpatialData artifact: %s",
            config.dataset_name,
            output_path,
        )
        return output_path

    raw_input_path = _resolve_raw_input_path(input_path)
    if raw_input_path is None:
        if reusable_source is None:
            raise FileNotFoundError(
                f"[{config.dataset_name}] Could not find raw input folder or existing "
                f"SpatialData zarr. input_path={input_path!s}, "
                f"persistent_output_path={persistent_output_path!s}"
            )
        raise ValueError(
            f"[{config.dataset_name}] --force-rerun was requested but only an existing "
            f"SpatialData zarr is available ({reusable_source}). Provide the raw input "
            "folder to rebuild from source data."
        )

    target_path = persistent_output_path or output_path
    if target_path != output_path:
        _remove_path(output_path)
    _remove_path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    platform = config.platform.upper()
    if platform == "MERSCOPE":
        from merxen.io.builders.merscope import write_merscope_spatialdata

        write_merscope_spatialdata(
            input_path=raw_input_path,
            output_path=target_path,
            build_config=config.merscope,
            transform_path_override=config.merscope_transform_path,
        )
    elif platform == "XENIUM":
        from merxen.io.builders.xenium import write_xenium_spatialdata

        write_xenium_spatialdata(
            input_path=raw_input_path,
            output_path=target_path,
            build_config=config.xenium,
            xenium_spec_path=config.xenium_spec_path,
        )
    else:
        raise ValueError(f"Unsupported platform: {config.platform}")

    if target_path != output_path:
        _stage_existing_output(target_path, output_path)
        return output_path
    return target_path


def _find_reusable_source(
    *,
    input_path: Path,
    persistent_output_path: Path | None,
) -> Path | None:
    """Return the first existing SpatialData zarr that can be reused."""
    candidates: list[Path] = []
    if persistent_output_path is not None:
        candidates.append(persistent_output_path)
    if input_path.suffix == ".zarr":
        candidates.append(input_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_raw_input_path(input_path: Path) -> Path | None:
    """Resolve the raw input directory if one was supplied."""
    if input_path.exists() and input_path.is_dir() and input_path.suffix != ".zarr":
        return input_path
    return None


def _stage_existing_output(source_path: Path, output_path: Path) -> None:
    """Create a local staged view of an existing SpatialData zarr."""
    output_path = Path(output_path)
    source_path = Path(source_path)
    if output_path == source_path:
        return

    _remove_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.symlink_to(source_path, target_is_directory=True)
    except OSError:
        shutil.copytree(source_path, output_path, symlinks=True)


def _remove_path(path: Path) -> None:
    """Remove a file, symlink, or directory if it exists."""
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
