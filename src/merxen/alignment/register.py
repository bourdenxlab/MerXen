"""Dispatch DAPI-only VALIS registration or the explicit legacy Spateo path."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

from merxen.alignment.bundle import ValisTransformBundle
from merxen.alignment.frames import (
    prepare_registration_frames,
    resolve_dapi_frame,
)
from merxen.alignment.models import TransformResult
from merxen.alignment.orientation import estimate_pre_orientation
from merxen.alignment.partial_overlap import refine_partial_overlap_rigid
from merxen.alignment.tissue import (
    AlignmentTissueAnnotation,
    annotation_sha256,
    load_alignment_tissue_annotation,
)
from merxen.alignment.transforms import apply_affine_matrix
from merxen.alignment.valis_register import run_valis_registration
from merxen.config import AlignmentConfig

logger = logging.getLogger(__name__)


def register_pair(
    merscope_sdata: Any,
    xenium_sdata: Any,
    config: Any,
    *,
    spateo_runner: Any = None,
    valis_runner: Callable[..., Any] | None = None,
) -> TransformResult:
    """Register a paired section with VALIS by default or legacy Spateo."""
    cfg = _coerce_alignment_config(config)
    if cfg.backend == "legacy_spateo" or spateo_runner is not None:
        from merxen.alignment.legacy_spateo import register_pair as legacy_register

        return cast(
            TransformResult,
            legacy_register(
                merscope_sdata,
                xenium_sdata,
                cfg,
                spateo_runner=spateo_runner,
            ),
        )

    tissue_annotations = load_required_valis_tissue_annotations(cfg)
    resumed_result = _load_completed_valis_result(cfg)
    if resumed_result is not None:
        return resumed_result
    _set_registration_seed(cfg.valis.random_seed)

    platform_sdata = {
        "MERSCOPE": merscope_sdata,
        "XENIUM": xenium_sdata,
    }
    platform_image_config = {
        "MERSCOPE": cfg.merscope_image,
        "XENIUM": cfg.xenium_image,
    }
    fixed_dapi = resolve_dapi_frame(
        platform_sdata[cfg.fixed_platform],
        platform=cfg.fixed_platform,
        config=platform_image_config[cfg.fixed_platform],
    )
    moving_dapi = resolve_dapi_frame(
        platform_sdata[cfg.moving_platform],
        platform=cfg.moving_platform,
        config=platform_image_config[cfg.moving_platform],
    )
    registration_image_dir = cfg.output_dir / "registration_inputs"
    fixed_frame, moving_frame = prepare_registration_frames(
        fixed_dapi,
        moving_dapi,
        config=cfg.valis,
        output_dir=registration_image_dir,
        fixed_tissue_annotation=tissue_annotations[cfg.fixed_platform],
        moving_tissue_annotation=tissue_annotations[cfg.moving_platform],
    )
    pre_orientation = estimate_pre_orientation(
        fixed_frame.processed_image,
        moving_frame.processed_image,
        fixed_frame.tissue_mask,
        moving_frame.tissue_mask,
        fixed_valid_mask=fixed_frame.valid_mask,
        moving_valid_mask=moving_frame.valid_mask,
        config=cfg.valis.orientation,
        pixel_size_um=fixed_frame.registration_pixel_size_um,
        output_dir=cfg.output_dir / "qc" / "orientation",
    )
    pre_orientation = refine_partial_overlap_rigid(
        fixed_frame.processed_image,
        moving_frame.processed_image,
        fixed_frame.tissue_mask,
        moving_frame.tissue_mask,
        initial=pre_orientation,
        config=cfg.valis.partial_overlap,
        pixel_size_um=fixed_frame.registration_pixel_size_um,
        fixed_valid_mask=fixed_frame.valid_mask,
        moving_valid_mask=moving_frame.valid_mask,
        output_dir=cfg.output_dir / "qc" / "partial_overlap",
    )
    runner = run_valis_registration if valis_runner is None else valis_runner
    valis_result = runner(
        fixed_frame,
        moving_frame,
        pre_orientation,
        config=cfg.valis,
        output_dir=cfg.output_dir,
    )

    return _build_valis_transform_result(
        cfg,
        bundle=valis_result.bundle,
        metadata=valis_result.metadata,
        valid_domain_mask=valis_result.shared_tissue_mask,
    )


def _build_valis_transform_result(
    cfg: AlignmentConfig,
    *,
    bundle: ValisTransformBundle,
    metadata: dict[str, Any],
    valid_domain_mask: np.ndarray,
) -> TransformResult:
    moving_transform = {
        "type": "valis_dapi_transform_chain",
        "selected_mode": (
            "nonrigid" if bundle.selected_mode == "non_rigid" else "rigid"
        ),
        "rigid_affine_matrix": bundle.global_dataset_matrix.tolist(),
        "transform_chain_path": str(cfg.output_dir / "transform_chain.json"),
    }
    identity = {
        "type": "identity",
        "affine_matrix": np.eye(3, dtype=float).tolist(),
    }
    if cfg.moving_platform == "MERSCOPE":
        merscope_transform = moving_transform
        xenium_transform = identity
    else:
        merscope_transform = identity
        xenium_transform = moving_transform

    complete_metadata = dict(metadata)
    complete_metadata.update(
        {
            "pair_id": cfg.pair_id,
            "fixed_platform": cfg.fixed_platform,
            "moving_platform": cfg.moving_platform,
            "coordinate_system_name": cfg.valis.coordinate_system_name,
        }
    )
    return TransformResult(
        merscope_to_common=merscope_transform,
        xenium_to_common=xenium_transform,
        metadata=complete_metadata,
        valis_transform=bundle,
        valid_domain_mask=valid_domain_mask,
    )


def _load_completed_valis_result(cfg: AlignmentConfig) -> TransformResult | None:
    """Reload a complete compatible registration when direct-CLI resume is enabled."""
    if not cfg.valis.resume:
        return None
    output_dir = Path(cfg.output_dir)
    chain_path = output_dir / "transform_chain.json"
    summary_path = output_dir / "registration_summary.json"
    mask_path = output_dir / "shared_tissue_mask_registration.npy"
    resume_manifest_path = output_dir / "resume_manifest.json"
    if not all(
        path.exists()
        for path in (
            chain_path,
            summary_path,
            mask_path,
            resume_manifest_path,
        )
    ):
        return None
    try:
        metadata = json.loads(summary_path.read_text())
        resume_manifest = json.loads(resume_manifest_path.read_text())
        expected_parameters = cfg.valis.model_dump(mode="json")
        coordinate_frames = metadata.get("coordinate_frames", {})
        compatible = (
            metadata.get("backend") == "valis"
            and metadata.get("parameters") == expected_parameters
            and coordinate_frames.get("fixed_platform") == cfg.fixed_platform
            and coordinate_frames.get("moving_platform") == cfg.moving_platform
            and resume_manifest == _resume_manifest_payload(cfg)
        )
        if not compatible:
            logger.info(
                "Existing VALIS artifacts do not match the requested configuration; "
                "starting a fresh registration"
            )
            return None
        bundle = ValisTransformBundle.load(chain_path)
        if bundle.selected_mode == "non_rigid" and bundle.forward_displacement is None:
            logger.warning(
                "Ignoring incomplete resumed VALIS transform without its forward field"
            )
            return None
        valid_domain_mask = np.asarray(np.load(mask_path), dtype=np.uint8)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Could not resume existing VALIS artifacts: %s", exc)
        return None

    logger.info("Resuming completed VALIS registration from %s", output_dir)
    return _build_valis_transform_result(
        cfg,
        bundle=bundle,
        metadata=metadata,
        valid_domain_mask=valid_domain_mask,
    )


def write_valis_resume_manifest(cfg: AlignmentConfig) -> Path:
    """Record the exact inputs and configuration required for safe direct resume."""
    manifest_path = Path(cfg.output_dir) / "resume_manifest.json"
    manifest_path.write_text(json.dumps(_resume_manifest_payload(cfg), indent=2))
    return manifest_path


def _resume_manifest_payload(cfg: AlignmentConfig) -> dict[str, Any]:
    annotation_paths = {
        "MERSCOPE": cfg.merscope_image.tissue_annotation_path,
        "XENIUM": cfg.xenium_image.tissue_annotation_path,
    }
    annotations = {
        platform: {
            "path": str(Path(cast(Path, path)).expanduser().resolve()),
            "sha256": annotation_sha256(cast(Path, path)),
        }
        for platform, path in annotation_paths.items()
        if path is not None
    }
    return {
        "version": 2,
        "backend": cfg.backend,
        "pair_id": cfg.pair_id,
        "merscope_zarr_path": str(Path(cfg.merscope_zarr_path).resolve()),
        "xenium_zarr_path": str(Path(cfg.xenium_zarr_path).resolve()),
        "fixed_platform": cfg.fixed_platform,
        "moving_platform": cfg.moving_platform,
        "tissue_annotations": annotations,
        "parameters": cfg.valis.model_dump(mode="json"),
    }


def load_required_valis_tissue_annotations(
    cfg: AlignmentConfig,
) -> dict[str, AlignmentTissueAnnotation]:
    """Load both platform-specific annotations required by the VALIS backend."""
    if cfg.backend != "valis":
        return {}
    image_configs = {
        "MERSCOPE": cfg.merscope_image,
        "XENIUM": cfg.xenium_image,
    }
    loaded: dict[str, AlignmentTissueAnnotation] = {}
    for platform, image_config in image_configs.items():
        path = image_config.tissue_annotation_path
        if path is None:
            raise ValueError(
                f"VALIS alignment for pair {cfg.pair_id!r} requires the "
                f"{platform} tissue annotation. Set "
                f"{platform.lower()}_image.tissue_annotation_path to the "
                "combined GeoJSON containing pial boundary piece(s) and "
                "exactly one shared tissue-edge boundary."
            )
        loaded[platform] = load_alignment_tissue_annotation(
            path,
            platform=platform,
        )
    return loaded


def _set_registration_seed(seed: int) -> None:
    """Seed NumPy, OpenCV, Python, and PyTorch registration components."""
    normalized_seed = int(seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    try:
        import cv2

        cv2.setRNGSeed(normalized_seed)
    except (ImportError, AttributeError):
        logger.debug("OpenCV RNG seeding is unavailable")
    try:
        import torch

        torch.manual_seed(normalized_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(normalized_seed)
    except ImportError:
        logger.debug("PyTorch RNG seeding is unavailable")


def transform_xy_for_result(result: TransformResult, coords: Any) -> np.ndarray:
    """Transform moving-platform dataset coordinates with the selected result."""
    if result.valis_transform is not None:
        return result.valis_transform.transform(coords, chunk_size=250_000)
    selected = result.merscope_to_common.get("selected_mode", "nonrigid")
    if selected == "nonrigid" and result.nonrigid_transform is not None:
        return result.nonrigid_transform.transform(coords)
    matrix = result.merscope_to_common["rigid_affine_matrix"]
    return apply_affine_matrix(coords, matrix)


def _coerce_alignment_config(config: Any) -> AlignmentConfig:
    if isinstance(config, AlignmentConfig):
        return config
    if isinstance(config, dict):
        return cast(AlignmentConfig, AlignmentConfig.model_validate(config))
    raise TypeError(
        "register_pair expects an AlignmentConfig or dict. "
        f"Got {type(config).__name__}."
    )


# Backward-compatible imports intentionally point at the marked legacy module.
from merxen.alignment.legacy_spateo import (  # noqa: E402
    _apply_spateo_import_shims,
    _resolve_device,
    _spateo_pairwise_kwargs,
    run_spateo_alignment,
)

__all__ = [
    "TransformResult",
    "_apply_spateo_import_shims",
    "_resolve_device",
    "_spateo_pairwise_kwargs",
    "register_pair",
    "load_required_valis_tissue_annotations",
    "run_spateo_alignment",
    "transform_xy_for_result",
    "write_valis_resume_manifest",
]
