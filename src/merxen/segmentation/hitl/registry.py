"""Validate custom Cellpose model bundles before inference or output mutation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from merxen.config import CellposeConfig, MaskFilterConfig, SegmentationConfig
from merxen.segmentation.model_identity import (
    CPSAM_V2_SHA256,
    CellposeInferenceContext,
    CellposeModelIdentity,
    find_model_identity_path,
    read_model_identity,
    require_cellpose_runtime,
)

logger = logging.getLogger(__name__)


class CellposeModelMetadata(BaseModel):
    """Version-one model.json contract; registry creation comes with training."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    platform: Literal["MERSCOPE", "XENIUM"]
    species: Literal["human", "mouse"]
    channels: list[str] = Field(min_length=1, max_length=3)
    image_key: Literal["morphology_focus", "MERSCOPE_z_projection"]
    pixel_size_um: float = Field(gt=0)
    tile_size: int = Field(gt=0)
    normalization_route: Literal["A"]
    annotation_style: str = Field(min_length=1)
    annotation_guide_version: str = Field(min_length=1)
    license_note: str = Field(min_length=1)
    status: Literal["candidate", "accepted", "rejected"]
    cellpose_version: str
    torch_version: str
    evaluated_cellpose: CellposeConfig
    evaluated_mask_filter: MaskFilterConfig
    evaluation_summary: str | None = None

    @model_validator(mode="after")
    def _require_native_image_and_unique_channels(self) -> CellposeModelMetadata:
        expected = (
            "morphology_focus" if self.platform == "XENIUM" else "MERSCOPE_z_projection"
        )
        if self.image_key != expected:
            raise ValueError("Model image_key must name its platform's native image")
        if any(not channel.strip() for channel in self.channels) or len(
            set(self.channels)
        ) != len(self.channels):
            raise ValueError("Model channels must be non-empty and unique")
        return self


def has_custom_model(config: SegmentationConfig) -> bool:
    """Return whether model selection requires the custom-model contract.

    Args:
        config: Requested segmentation settings.

    Returns:
        Whether an explicit checkpoint, checksum or metadata path is selected.
    """
    return (
        config.cellpose.pretrained_model != "cpsam_v2"
        or config.cellpose.model_sha256 is not None
        or config.cellpose.model_metadata_path is not None
    )


def _protected_paths(roots: list[Path]) -> set[Path]:
    """Include published symlink targets without traversing zarr chunk trees."""
    protected: set[Path] = set()
    for root in roots:
        protected.update((Path(os.path.abspath(root)), root.resolve()))
        for directory, names, files in os.walk(root, followlinks=False):
            for name in names + files:
                path = Path(directory) / name
                if path.is_symlink():
                    protected.add(path.resolve())
            names[:] = [
                name
                for name in names
                if not name.endswith(".zarr")
                and not (Path(directory) / name).is_symlink()
            ]
    return protected


def require_safe_custom_outputs(
    config: SegmentationConfig, *, include_source: bool = True
) -> None:
    """Reject custom-arm outputs under any protected root, including symlinks.

    Args:
        config: Effective segmentation configuration.
        include_source: Protect mutable pipeline inputs too; read-only diagnostics
            may explicitly exclude their source from this check.

    Raises:
        ValueError: If protection is unspecified or an output resolves into it.
    """
    if not has_custom_model(config):
        return
    roots = config.cellpose_model_policy.protected_output_roots
    if not roots:
        raise ValueError("Custom models require non-empty protected_output_roots")
    dataset = config.dataset
    paths = [dataset.output_dir] + [
        value
        for name in type(dataset).model_fields
        if name.startswith("persistent_")
        and (value := getattr(dataset, name)) is not None
    ]
    if include_source:
        paths.append(dataset.data_path)
    protected = _protected_paths(roots)
    for path in paths:
        absolute = Path(os.path.abspath(path))
        resolved = path.resolve()
        resolved_parents = tuple(parent.resolve() for parent in absolute.parents)
        if any(
            absolute.is_relative_to(root)
            or resolved.is_relative_to(root)
            or root.is_relative_to(resolved)
            or any(parent.is_relative_to(root) for parent in resolved_parents)
            for root in protected
        ):
            raise ValueError(f"Custom-model arm targets a protected baseline: {path}")


def preflight_custom_model(
    config: SegmentationConfig,
    *,
    transform_matrix: np.ndarray,
    image_key: str,
    allow_readonly_source: bool = False,
) -> CellposeModelIdentity:
    """Validate a bundle against the actual native image and runtime.

    Args:
        config: Effective model, image, filter and output configuration.
        transform_matrix: Native micron-to-pixel affine, including rotation.
        image_key: Image selected by the production source loader.
        allow_readonly_source: Permit baseline source reads in a read-only tile
            diagnostic. Pipeline entry points always protect their source too.

    Returns:
        Expected inference identity, also used to validate existing masks.

    Raises:
        ValueError: If metadata, weights, runtime or image settings disagree.
        FileNotFoundError: If model weights or model.json are missing.
    """
    require_safe_custom_outputs(config, include_source=not allow_readonly_source)
    cellpose = config.cellpose
    metadata_path = cellpose.model_metadata_path
    if metadata_path is None or cellpose.model_sha256 is None:
        raise ValueError("Custom models require model_metadata_path and model_sha256")
    metadata = CellposeModelMetadata.model_validate_json(metadata_path.read_text())
    weights = Path(cellpose.pretrained_model)
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata.weights_sha256 or digest != cellpose.model_sha256:
        raise ValueError("Custom model checksum does not match model.json")
    policy = config.cellpose_model_policy
    if metadata.status == "rejected" or (
        metadata.status == "candidate" and not policy.allow_candidate_model
    ):
        raise ValueError("Model must be accepted, or candidate with explicit opt-in")
    if (
        metadata.platform != config.dataset.platform
        or metadata.species != policy.species
    ):
        raise ValueError("Model platform or species does not match this dataset")
    if metadata.channels != config.dataset.channels or metadata.image_key != image_key:
        raise ValueError(
            "Model channels/order or native image does not match this dataset"
        )
    pixel_to_um = np.linalg.inv(transform_matrix)
    pixel_sizes = tuple(float(np.linalg.norm(pixel_to_um[:2, axis])) for axis in (0, 1))
    if not all(
        abs(size - metadata.pixel_size_um) <= metadata.pixel_size_um * 0.01 + 1e-12
        for size in pixel_sizes
    ):
        raise ValueError(
            "Model pixel size differs from the native affine by more than 1%"
        )
    if config.tiling.tile_size_candidates != [metadata.tile_size] or (
        metadata.tile_size < config.tiling.min_tile_size
    ):
        raise ValueError(
            "Custom models require the evaluated tile size as a single candidate"
        )
    cellpose_version, torch_version = require_cellpose_runtime()
    if (cellpose_version, torch_version) != (
        metadata.cellpose_version,
        metadata.torch_version,
    ):
        raise ValueError(
            "Model Cellpose/torch runtime does not match the loaded runtime"
        )
    for field in ("flow_threshold", "cellprob_threshold", "diameter", "use_bfloat16"):
        if getattr(cellpose, field) != getattr(metadata.evaluated_cellpose, field):
            logger.warning(
                "Explicit %s differs from the model's evaluation settings", field
            )
    for field in ("final_min_area_um2", "final_max_area_um2", "max_eccentricity"):
        if getattr(config.mask_filter, field) != getattr(
            metadata.evaluated_mask_filter, field
        ):
            logger.warning(
                "Explicit %s differs from the model's evaluation settings", field
            )
    context = CellposeInferenceContext(
        platform=config.dataset.platform,
        species=policy.species,
        channels=config.dataset.channels,
        image_key=image_key,
        pixel_size_um=(pixel_sizes[0], pixel_sizes[1]),
        mask_filter=config.mask_filter,
        tiling=config.tiling,
    )
    return CellposeModelIdentity(
        status="verified",
        model_name=cellpose.pretrained_model,
        weights_sha256=digest,
        cellpose_version=cellpose_version,
        torch_version=torch_version,
        cellpose_config=cellpose,
        inference_context=context,
    )


def require_matching_cached_model(
    mask_path: Path,
    config: SegmentationConfig,
    expected: CellposeModelIdentity | None = None,
) -> None:
    """Refuse model/configuration mismatches without changing cached artifacts.

    Args:
        mask_path: Existing or prospective mask path.
        config: Requested segmentation settings.
        expected: Full preflight identity for a custom model.

    Raises:
        ValueError: If a custom mask is unverified, or recorded settings differ.
    """
    if not mask_path.exists():
        return
    recorded = read_model_identity(mask_path)
    if has_custom_model(config):
        if expected is None or recorded != expected:
            raise ValueError(
                "Cached custom-model identity mismatch; use a separate outdir"
            )
        sidecar_path = find_model_identity_path(mask_path)
        assert sidecar_path is not None
        stats_path = config.dataset.persistent_cellpose_stitching_stats_path or (
            sidecar_path.with_name("cellpose_stitching_stats.json")
        )
        if not stats_path.exists() or json.loads(stats_path.read_text()).get(
            "model_identity"
        ) != expected.model_dump(mode="json"):
            raise ValueError("Cached custom-model statistics identity mismatch")
        return
    # The agreed compatibility policy preserves unknown historical stock output.
    if recorded.status == "unverified":
        return
    if recorded.weights_sha256 != CPSAM_V2_SHA256 or (
        recorded.cellpose_config != config.cellpose
    ):
        raise ValueError(
            "Cached Cellpose model/configuration mismatch; use a separate outdir"
        )
    if recorded.inference_context is not None and (
        recorded.inference_context.mask_filter != config.mask_filter
        or recorded.inference_context.tiling != config.tiling
    ):
        raise ValueError(
            "Cached Cellpose filtering/tiling mismatch; use a separate outdir"
        )
