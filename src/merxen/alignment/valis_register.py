"""VALIS 1.2 execution and transform extraction for two DAPI images."""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from skimage.transform import EuclideanTransform

from merxen.alignment.bundle import DisplacementField, ValisTransformBundle
from merxen.alignment.dapi import configured_dapi_processor_class
from merxen.alignment.frames import RegistrationFrame
from merxen.alignment.orientation import OrientationResult, warp_image
from merxen.alignment.partial_overlap import (
    evaluate_aligned_partial_overlap_objective,
)
from merxen.alignment.qc import (
    affine_diagnostics,
    compute_dapi_metrics,
    displacement_diagnostics,
    global_qc_passes,
    morphology_supported_global_qc_passes,
    plot_checkerboard,
    plot_deformation_qc,
    plot_feature_matches,
    plot_mask_overlap,
    plot_registration_overlay,
    rigid_qc_passes,
    select_non_rigid_result,
)
from merxen.alignment.transforms import (
    apply_affine_matrix,
    fit_rigid_matrix,
)
from merxen.alignment.valis_compat import apply_valis_numpy_compatibility
from merxen.config import ValisAlignmentConfig

logger = logging.getLogger(__name__)


class _LockedIdentityTransform(EuclideanTransform):
    """Identity-only transformer used to prevent hidden VALIS rigid fitting.

    ``Valis(do_rigid=False)`` is documented to install identity transforms, but
    VALIS still constructs a serial rigid registrar and retains a transformer
    object for feature matching and transform extraction. Keeping that object
    identity-only makes the accepted MerXen partial-overlap transform
    authoritative even if a future VALIS code path calls ``estimate``.
    """

    def estimate(
        self: _LockedIdentityTransform,
        src: Any,
        dst: Any,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        del src, dst, args, kwargs
        self.params = np.eye(3, dtype=np.float64)
        return True


@dataclass(frozen=True)
class ValisRegistrationResult:
    """Selected VALIS transform bundle and DAPI-only QC artifacts."""

    bundle: ValisTransformBundle
    status: str
    metadata: dict[str, Any]
    registrar_path: Path
    shared_tissue_mask: np.ndarray
    registered_preview_path: Path


def run_valis_registration(
    fixed: RegistrationFrame,
    moving: RegistrationFrame,
    pre_orientation: OrientationResult,
    *,
    config: ValisAlignmentConfig,
    output_dir: Path,
) -> ValisRegistrationResult:
    """Run VALIS refinement and select global or non-rigid DAPI registration."""
    output_dir = Path(output_dir)
    valis_input_dir = output_dir / "registration_images"
    valis_output_dir = output_dir / "valis"
    qc_dir = output_dir / "qc"
    for directory in (valis_input_dir, valis_output_dir, qc_dir):
        directory.mkdir(parents=True, exist_ok=True)

    moving_pre_image = warp_image(
        moving.processed_image,
        pre_orientation.matrix,
        output_shape_rc=fixed.registration_shape_rc,
    )
    moving_pre_mask = warp_image(
        moving.tissue_mask,
        pre_orientation.matrix,
        output_shape_rc=fixed.registration_shape_rc,
        interpolation=cv2.INTER_NEAREST,
    )
    moving_pre_valid = warp_image(
        moving.valid_mask,
        pre_orientation.matrix,
        output_shape_rc=fixed.registration_shape_rc,
        interpolation=cv2.INTER_NEAREST,
    )
    shared_non_rigid_mask = _shared_non_rigid_domain(
        fixed_tissue_mask=fixed.tissue_mask,
        moving_tissue_mask=moving_pre_mask,
        fixed_valid_mask=fixed.valid_mask,
        moving_valid_mask=moving_pre_valid,
    )
    shared_non_rigid_weight = _feather_binary_mask(
        shared_non_rigid_mask,
        taper_px=float(config.preprocessing.edge_taper_um)
        / float(fixed.registration_pixel_size_um),
    )
    fixed_path = valis_input_dir / "0_fixed_dapi.tif"
    moving_path = valis_input_dir / "1_moving_dapi_preoriented.tif"
    _write_tiff(fixed_path, fixed.processed_image)
    _write_tiff(moving_path, moving_pre_image)
    locked_input_dir = valis_input_dir / "locked_shared_domain"
    locked_input_dir.mkdir(parents=True, exist_ok=True)
    locked_fixed_path = locked_input_dir / fixed_path.name
    locked_moving_path = locked_input_dir / moving_path.name
    _write_tiff(
        locked_fixed_path,
        _weight_registration_image(fixed.processed_image, shared_non_rigid_weight),
    )
    _write_tiff(
        locked_moving_path,
        _weight_registration_image(moving_pre_image, shared_non_rigid_weight),
    )
    shared_non_rigid_mask_path = valis_input_dir / "shared_non_rigid_mask.tif"
    shared_non_rigid_feather_path = valis_input_dir / "shared_non_rigid_feather.tif"
    _write_tiff(shared_non_rigid_mask_path, shared_non_rigid_mask)
    _write_tiff(
        shared_non_rigid_feather_path,
        np.round(shared_non_rigid_weight * 255.0).astype(np.uint8),
    )

    initial_metrics = compute_dapi_metrics(
        fixed.processed_image,
        moving.processed_image,
        fixed.tissue_mask,
        moving.tissue_mask,
    )
    pre_metrics = compute_dapi_metrics(
        fixed.processed_image,
        moving_pre_image,
        fixed.tissue_mask,
        moving_pre_mask,
        moving_feature_xy=(
            None
            if pre_orientation.moving_inlier_xy is None
            else apply_affine_matrix(
                pre_orientation.moving_inlier_xy,
                pre_orientation.matrix,
            )
        ),
        fixed_feature_xy=pre_orientation.fixed_inlier_xy,
    )
    plot_registration_overlay(
        fixed.processed_image,
        moving.processed_image,
        qc_dir / "original_overlay.png",
        title="Original DAPI overlay",
    )
    plot_registration_overlay(
        fixed.processed_image,
        moving_pre_image,
        qc_dir / "preoriented_overlay.png",
        title="Coarse pre-oriented DAPI overlay",
    )

    attempts: list[dict[str, Any]] = []
    global_selection_reason = (
        "accepted partial-overlap transform locked; VALIS optimized only the "
        "local non-rigid deformation"
    )
    attempt = _execute_valis(
        fixed_path=locked_fixed_path,
        moving_path=locked_moving_path,
        fixed_frame=fixed,
        moving_pre_image=moving_pre_image,
        moving_pre_mask=moving_pre_mask,
        moving_pre_valid=moving_pre_valid,
        shared_non_rigid_mask=shared_non_rigid_mask,
        shared_non_rigid_weight=shared_non_rigid_weight,
        config=config,
        output_dir=valis_output_dir / "locked_non_rigid",
    )
    attempts.append(attempt.metadata)

    global_pass, global_reasons, global_acceptance_mode = _evaluate_global_attempt(
        attempt,
        preorientation_metrics=pre_metrics,
        config=config,
        trusted_coordinate_metadata=(
            fixed.coordinate_metadata_trusted and moving.coordinate_metadata_trusted
        ),
        reflection_selected=bool(pre_orientation.metrics.get("reflection", False)),
    )

    if not global_pass:
        preorientation_attempt = _preorientation_fallback_attempt(
            moving_pre_image=moving_pre_image,
            moving_pre_mask=moving_pre_mask,
            preorientation_metrics=pre_metrics,
            moving_feature_xy=(
                None
                if pre_orientation.moving_inlier_xy is None
                else apply_affine_matrix(
                    pre_orientation.moving_inlier_xy,
                    pre_orientation.matrix,
                )
            ),
            fixed_feature_xy=pre_orientation.fixed_inlier_xy,
            registrar_path=attempt.registrar_path,
            failed_valis_reasons=global_reasons,
        )
        (
            preorientation_pass,
            preorientation_reasons,
            preorientation_acceptance_mode,
        ) = _evaluate_global_attempt(
            preorientation_attempt,
            preorientation_metrics=pre_metrics,
            config=config,
            trusted_coordinate_metadata=(
                fixed.coordinate_metadata_trusted and moving.coordinate_metadata_trusted
            ),
            reflection_selected=bool(pre_orientation.metrics.get("reflection", False)),
        )
        attempts.append(preorientation_attempt.metadata)
        if preorientation_pass:
            logger.warning(
                "VALIS refinements failed DAPI QC (%s); retaining the validated "
                "coarse pre-orientation",
                ", ".join(global_reasons),
            )
            attempt = preorientation_attempt
            global_pass = True
            global_reasons = []
            global_acceptance_mode = preorientation_acceptance_mode
            global_selection_reason = (
                "VALIS refinements failed QC; retained validated pre-orientation"
            )
        else:
            raise RuntimeError(
                "VALIS global registration failed DAPI QC: "
                + ", ".join(preorientation_reasons)
            )

    selected_mode = "global"
    status = "global_only"
    selection_reasons: list[str] = []
    if (
        config.non_rigid.enabled
        and attempt.forward_displacement is not None
        and attempt.non_rigid_metrics is not None
        and attempt.deformation_metrics is not None
    ):
        use_non_rigid, selection_reasons = select_non_rigid_result(
            attempt.global_metrics,
            attempt.non_rigid_metrics,
            attempt.deformation_metrics,
            thresholds=config.qc,
        )
        if use_non_rigid:
            selected_mode = "non_rigid"
            status = "non_rigid_pass"
        else:
            status = "global_only"

    selected_image = (
        attempt.non_rigid_image
        if selected_mode == "non_rigid" and attempt.non_rigid_image is not None
        else attempt.global_image
    )
    selected_mask = (
        attempt.non_rigid_mask
        if selected_mode == "non_rigid" and attempt.non_rigid_mask is not None
        else attempt.global_mask
    )
    shared_tissue_registration = (
        (np.asarray(fixed.tissue_mask) > 0) & (np.asarray(selected_mask) > 0)
    ).astype(np.uint8)
    shared_tissue_registration_path = output_dir / "shared_tissue_mask_registration.npy"
    np.save(shared_tissue_registration_path, shared_tissue_registration)
    _write_tiff(
        output_dir / "shared_tissue_mask_registration.tif",
        shared_tissue_registration * 255,
    )
    shared_tissue_fixed_image = warp_image(
        shared_tissue_registration,
        np.linalg.inv(fixed.original_to_registration_matrix),
        output_shape_rc=fixed.original_shape_rc,
        interpolation=cv2.INTER_NEAREST,
    )
    shared_tissue_path = output_dir / "shared_tissue_mask.npy"
    np.save(shared_tissue_path, shared_tissue_fixed_image)
    _write_tiff(output_dir / "shared_tissue_mask.tif", shared_tissue_fixed_image * 255)

    bundle = ValisTransformBundle(
        moving_dataset_to_image=moving.dataset_to_image_matrix,
        moving_image_to_registration=moving.original_to_registration_matrix,
        pre_matrix=pre_orientation.matrix,
        global_matrix=attempt.global_matrix,
        fixed_image_to_registration=fixed.original_to_registration_matrix,
        fixed_dataset_to_image=fixed.dataset_to_image_matrix,
        selected_mode=selected_mode,
        forward_displacement=attempt.forward_displacement,
        backward_displacement=attempt.backward_displacement,
    )
    transform_outputs = bundle.save(output_dir)

    plot_registration_overlay(
        fixed.processed_image,
        attempt.global_image,
        qc_dir / "global_overlay.png",
        title="Global VALIS DAPI overlay",
    )
    plot_checkerboard(
        fixed.processed_image,
        attempt.global_image,
        qc_dir / "global_checkerboard.png",
        title="Global VALIS checkerboard",
    )
    if attempt.non_rigid_image is not None:
        plot_registration_overlay(
            fixed.processed_image,
            attempt.non_rigid_image,
            qc_dir / "non_rigid_overlay.png",
            title="Non-rigid VALIS DAPI overlay",
        )
        plot_checkerboard(
            fixed.processed_image,
            attempt.non_rigid_image,
            qc_dir / "non_rigid_checkerboard.png",
            title="Non-rigid VALIS checkerboard",
        )
    plot_mask_overlap(
        fixed.tissue_mask,
        selected_mask,
        qc_dir / "tissue_mask_overlap.png",
        title="Selected registration tissue overlap",
    )
    plot_feature_matches(
        fixed.processed_image,
        moving_pre_image,
        attempt.moving_feature_xy,
        attempt.fixed_feature_xy,
        qc_dir / "feature_inliers.png",
        title="VALIS global feature inliers",
    )
    if attempt.forward_displacement is not None:
        plot_deformation_qc(attempt.forward_displacement, qc_dir)

    preview_path = qc_dir / "registered_dapi_preview.png"
    plot_registration_overlay(
        fixed.processed_image,
        selected_image,
        preview_path,
        title=f"Selected VALIS registration ({status})",
    )

    metadata = {
        "backend": "valis",
        "method": "valis_dapi",
        "status": status,
        "selected_mode": selected_mode,
        "selected_global_transform": attempt.transform_name,
        "configured_global_transform_deprecated": config.global_transform,
        "global_acceptance_mode": global_acceptance_mode,
        "global_selection_reason": global_selection_reason,
        "non_rigid_selection_reasons": selection_reasons,
        "pre_orientation": {
            "method": pre_orientation.method,
            "angle_degrees": pre_orientation.angle_degrees,
            "scale": pre_orientation.scale,
            "score": pre_orientation.score,
            "matrix": pre_orientation.matrix.tolist(),
            "metrics": pre_orientation.metrics,
        },
        "qc": {
            "initial": initial_metrics,
            "preoriented": pre_metrics,
            "global": attempt.global_metrics,
            "non_rigid": attempt.non_rigid_metrics,
            "affine": attempt.affine_metrics,
            "composed_dataset_affine": affine_diagnostics(bundle.global_dataset_matrix),
            "deformation": attempt.deformation_metrics,
            "thresholds": config.qc.model_dump(),
        },
        "attempts": attempts,
        "dependency_versions": dependency_versions(),
        "parameters": config.model_dump(mode="json"),
        "coordinate_frames": {
            "fixed_platform": fixed.platform,
            "moving_platform": moving.platform,
            "fixed_image_key": fixed.image_key,
            "moving_image_key": moving.image_key,
            "registration_shape_rc": list(fixed.registration_shape_rc),
            "registration_pixel_size_um": fixed.registration_pixel_size_um,
            "fixed_coordinate_metadata_source": fixed.coordinate_metadata_source,
            "moving_coordinate_metadata_source": moving.coordinate_metadata_source,
            "fixed_coordinate_metadata_trusted": (fixed.coordinate_metadata_trusted),
            "moving_coordinate_metadata_trusted": (moving.coordinate_metadata_trusted),
            "fixed_dapi_edge_artifact_metrics": fixed.edge_artifact_metrics,
            "moving_dapi_edge_artifact_metrics": moving.edge_artifact_metrics,
            "fixed_dataset_to_image_matrix": (fixed.dataset_to_image_matrix.tolist()),
            "moving_dataset_to_image_matrix": (moving.dataset_to_image_matrix.tolist()),
        },
        "transform_chain_path": str(transform_outputs["transform_chain"]),
        "shared_tissue_mask_path": str(shared_tissue_path),
        "shared_tissue_registration_mask_path": str(shared_tissue_registration_path),
        "non_rigid_shared_domain_path": str(shared_non_rigid_mask_path),
        "non_rigid_shared_feather_path": str(shared_non_rigid_feather_path),
        "registered_preview_path": str(preview_path),
        "registrar_path": str(attempt.registrar_path),
    }
    (output_dir / "registration_summary.json").write_text(
        json.dumps(_jsonable(metadata), indent=2)
    )
    pd.json_normalize(metadata["qc"], sep=".").to_csv(
        output_dir / "registration_summary.csv",
        index=False,
    )
    return ValisRegistrationResult(
        bundle=bundle,
        status=status,
        metadata=metadata,
        registrar_path=attempt.registrar_path,
        shared_tissue_mask=shared_tissue_registration,
        registered_preview_path=preview_path,
    )


@dataclass(frozen=True)
class _ValisAttempt:
    global_matrix: np.ndarray
    forward_displacement: DisplacementField | None
    backward_displacement: DisplacementField | None
    global_image: np.ndarray
    global_mask: np.ndarray
    non_rigid_image: np.ndarray | None
    non_rigid_mask: np.ndarray | None
    global_metrics: dict[str, Any]
    non_rigid_metrics: dict[str, Any] | None
    affine_metrics: dict[str, Any]
    deformation_metrics: dict[str, Any] | None
    moving_feature_xy: np.ndarray | None
    fixed_feature_xy: np.ndarray | None
    transform_name: str
    registrar_path: Path
    metadata: dict[str, Any]


def _evaluate_global_attempt(
    attempt: _ValisAttempt,
    *,
    preorientation_metrics: dict[str, Any],
    config: ValisAlignmentConfig,
    trusted_coordinate_metadata: bool,
    reflection_selected: bool,
) -> tuple[bool, list[str], str]:
    """Apply image, morphology, and rigid-model QC to one global result."""
    standard_pass, standard_reasons = global_qc_passes(
        attempt.global_metrics,
        attempt.affine_metrics,
        thresholds=config.qc,
    )
    rigid_pass, rigid_reasons = rigid_qc_passes(
        attempt.affine_metrics,
        thresholds=config.qc,
    )
    if standard_pass and rigid_pass:
        return True, [], "standard"

    morphology_pass, morphology_reasons = morphology_supported_global_qc_passes(
        attempt.global_metrics,
        attempt.affine_metrics,
        preorientation_metrics=preorientation_metrics,
        thresholds=config.qc,
        trusted_coordinate_metadata=trusted_coordinate_metadata,
        reflection_selected=reflection_selected,
    )
    if morphology_pass and rigid_pass:
        return True, [], "morphology_supported"
    reasons = list(standard_reasons) + list(rigid_reasons)
    reasons.extend(f"morphology:{reason}" for reason in morphology_reasons)
    return False, reasons, "rejected"


def _preorientation_fallback_attempt(
    *,
    moving_pre_image: np.ndarray,
    moving_pre_mask: np.ndarray,
    preorientation_metrics: dict[str, Any],
    moving_feature_xy: np.ndarray | None,
    fixed_feature_xy: np.ndarray | None,
    registrar_path: Path,
    failed_valis_reasons: list[str],
) -> _ValisAttempt:
    """Represent a validated coarse orientation as an identity refinement.

    Cross-platform adjacent sections can have strong tissue agreement but too
    few cell-exact features for VALIS to estimate a safe refinement. In that
    case the already-scaled, explicitly reflected pre-orientation is preferable
    to accepting a sheared or morphology-degrading affine.
    """
    identity = np.eye(3, dtype=np.float64)
    affine_metrics = affine_diagnostics(identity)
    metadata = {
        "transform": "PreorientationIdentity",
        "global_qc": preorientation_metrics,
        "affine": affine_metrics,
        "non_rigid_qc": None,
        "deformation": None,
        "valis_summary": None,
        "fallback_after_valis_qc_reasons": list(failed_valis_reasons),
    }
    return _ValisAttempt(
        global_matrix=identity,
        forward_displacement=None,
        backward_displacement=None,
        global_image=np.asarray(moving_pre_image),
        global_mask=np.asarray(moving_pre_mask),
        non_rigid_image=None,
        non_rigid_mask=None,
        global_metrics=dict(preorientation_metrics),
        non_rigid_metrics=None,
        affine_metrics=affine_metrics,
        deformation_metrics=None,
        moving_feature_xy=moving_feature_xy,
        fixed_feature_xy=fixed_feature_xy,
        transform_name="PreorientationIdentity",
        registrar_path=Path(registrar_path),
        metadata=metadata,
    )


def _shared_non_rigid_domain(
    *,
    fixed_tissue_mask: Any,
    moving_tissue_mask: Any,
    fixed_valid_mask: Any,
    moving_valid_mask: Any,
) -> np.ndarray:
    """Return the acquired, valid tissue shared by both pre-aligned inputs."""
    arrays = [
        np.asarray(value) > 0
        for value in (
            fixed_tissue_mask,
            moving_tissue_mask,
            fixed_valid_mask,
            moving_valid_mask,
        )
    ]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(
            "Shared non-rigid tissue/support masks must have one common shape; "
            f"got {sorted(shapes)}"
        )
    shared = np.logical_and.reduce(arrays)
    if not np.any(shared):
        raise RuntimeError(
            "The accepted partial-overlap transform has no shared valid tissue "
            "for VALIS non-rigid registration"
        )
    return np.asarray(shared.astype(np.uint8) * 255, dtype=np.uint8)


def _feather_binary_mask(mask: Any, *, taper_px: float) -> np.ndarray:
    """Cosine-feather a binary domain inward without expanding its support."""
    binary = np.asarray(mask) > 0
    if not np.any(binary):
        raise ValueError("Cannot feather an empty non-rigid registration mask")
    if float(taper_px) <= 0.0:
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(
        binary.astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    phase = np.clip(distance / float(taper_px), 0.0, 1.0)
    weight = 0.5 - 0.5 * np.cos(np.pi * phase)
    weight[~binary] = 0.0
    return np.asarray(weight, dtype=np.float32)


def _weight_registration_image(image: Any, weight: Any) -> np.ndarray:
    """Apply a registration-only spatial weight without changing source data."""
    arr = np.asarray(image)
    weights = np.asarray(weight, dtype=np.float32)
    if arr.shape != weights.shape:
        raise ValueError(
            f"Registration image and weight shapes differ: {arr.shape} vs "
            f"{weights.shape}"
        )
    weighted = np.asarray(arr, dtype=np.float32) * weights
    if np.issubdtype(arr.dtype, np.integer):
        limits = np.iinfo(arr.dtype)
        weighted = np.clip(np.rint(weighted), limits.min, limits.max)
    return weighted.astype(arr.dtype, copy=False)


def _add_partial_overlap_metrics(
    metrics: dict[str, Any],
    *,
    fixed_image: Any,
    moving_image: Any,
    fixed_mask: Any,
    moving_mask: Any,
    fixed_valid_mask: Any,
    moving_valid_mask: Any,
    config: ValisAlignmentConfig,
    pixel_size_um: float,
) -> None:
    """Attach the same robust score used to choose the locked pretransform."""
    objective = evaluate_aligned_partial_overlap_objective(
        fixed_image,
        moving_image,
        fixed_mask,
        moving_mask,
        config=config.partial_overlap,
        pixel_size_um=float(pixel_size_um),
        fixed_valid_mask=fixed_valid_mask,
        moving_valid_mask=moving_valid_mask,
    )
    metrics["partial_overlap_robust_score"] = float(objective["score"])
    metrics["partial_overlap_objective"] = objective


def _shared_mask_processor_class(
    processor_cls: type[Any],
    *,
    mask_by_path: dict[Path, np.ndarray],
) -> type[Any]:
    """Wrap a VALIS DAPI processor with the explicit shared registration mask."""
    masks = {
        Path(path).name: np.asarray(mask, dtype=np.uint8)
        for path, mask in mask_by_path.items()
    }

    class LockedSharedMaskDapiImageProcesser(processor_cls):
        def create_mask(
            self: LockedSharedMaskDapiImageProcesser,
        ) -> np.ndarray:
            key = Path(str(self.src_f)).name
            if key not in masks:
                raise RuntimeError(
                    f"No locked shared-tissue mask configured for VALIS input {key!r}"
                )
            mask = masks[key]
            image = np.asarray(self.image)
            target_shape = image.shape[:2]
            if mask.shape != target_shape:
                mask = np.asarray(
                    cv2.resize(
                        mask,
                        (int(target_shape[1]), int(target_shape[0])),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    dtype=np.uint8,
                )
            return np.asarray((mask > 0).astype(np.uint8) * 255, dtype=np.uint8)

    LockedSharedMaskDapiImageProcesser.__name__ = "LockedSharedMaskDapiImageProcesser"
    return LockedSharedMaskDapiImageProcesser


def _assert_valis_rigid_map_is_identity(
    moving_slide: Any,
    fixed_slide: Any,
    *,
    sample_xy: np.ndarray,
    shape_rc: tuple[int, int],
    tolerance_px: float = 1e-3,
) -> dict[str, float]:
    """Fail closed if VALIS changes the locked global coordinate frame."""
    source = np.asarray(sample_xy, dtype=np.float64)
    target = np.asarray(
        moving_slide.warp_xy_from_to(
            source,
            fixed_slide,
            src_pt_level=shape_rc,
            dst_slide_level=shape_rc,
            non_rigid=False,
        ),
        dtype=np.float64,
    )
    if target.shape != source.shape or not np.isfinite(target).all():
        raise RuntimeError(
            "VALIS returned invalid coordinates while verifying its locked "
            "identity rigid map"
        )
    residual = np.linalg.norm(target - source, axis=1)
    diagnostics = {
        "maximum_residual_px": float(np.max(residual)),
        "median_residual_px": float(np.median(residual)),
        "tolerance_px": float(tolerance_px),
    }
    if diagnostics["maximum_residual_px"] > float(tolerance_px):
        raise RuntimeError(
            "VALIS changed the locked global transform despite do_rigid=False: "
            f"maximum identity residual "
            f"{diagnostics['maximum_residual_px']:.6g}px exceeds "
            f"{float(tolerance_px):.6g}px"
        )
    return diagnostics


def _execute_valis(
    *,
    fixed_path: Path,
    moving_path: Path,
    fixed_frame: RegistrationFrame,
    moving_pre_image: np.ndarray,
    moving_pre_mask: np.ndarray,
    moving_pre_valid: np.ndarray,
    shared_non_rigid_mask: np.ndarray,
    shared_non_rigid_weight: np.ndarray,
    config: ValisAlignmentConfig,
    output_dir: Path,
) -> _ValisAttempt:
    apply_valis_numpy_compatibility()
    try:
        from valis import (
            feature_detectors,
            feature_matcher,
            non_rigid_registrars,
            registration,
        )
    except ImportError as exc:
        raise RuntimeError(
            "VALIS alignment requires valis-wsi==1.2.0 in the dedicated "
            "alignment environment"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    processor_cls = configured_dapi_processor_class(
        pixel_size_um=fixed_frame.registration_pixel_size_um,
        config=config.preprocessing,
        input_is_preprocessed=True,
    )
    processor_cls = _shared_mask_processor_class(
        processor_cls,
        mask_by_path={
            fixed_path: shared_non_rigid_mask,
            moving_path: shared_non_rigid_mask,
        },
    )
    processor_dict = {
        str(fixed_path): [processor_cls, {"input_is_preprocessed": True}],
        str(moving_path): [processor_cls, {"input_is_preprocessed": True}],
    }
    device = _resolve_torch_device(config.features.device)
    detector = _create_disk_detector(
        feature_detectors,
        num_features=int(config.features.num_features),
        device=device,
    )
    matcher = feature_matcher.LightGlueMatcher(
        feature_detector=detector,
        match_filter_method="USAC_MAGSAC",
        ransac_thresh=float(config.features.ransac_threshold_px),
        device=device,
    )
    non_rigid_cls, non_rigid_params = _non_rigid_backend(
        config,
        non_rigid_registrars,
    )
    norm_method = None if config.norm_method == "none" else config.norm_method

    registrar = registration.Valis(
        str(fixed_path.parent),
        str(output_dir),
        name="merxen_locked_non_rigid",
        image_type="fluorescence",
        feature_detector_cls=detector,
        transformer_cls=_LockedIdentityTransform,
        affine_optimizer_cls=None,
        matcher=matcher,
        matcher_for_sorting=matcher,
        imgs_ordered=True,
        non_rigid_registrar_cls=non_rigid_cls,
        non_rigid_reg_params=non_rigid_params,
        compose_non_rigid=False,
        img_list={
            str(fixed_path): "0_fixed_dapi",
            str(moving_path): "1_moving_dapi_preoriented",
        },
        # VALIS indexes ``name_dict`` by the exact keys supplied through the
        # explicit ``img_list`` mapping.  Supplying only the basename passes
        # constructor validation but later fails in ``rigid_register_partial``.
        reference_img_f=str(fixed_path),
        align_to_reference=True,
        do_rigid=False,
        crop="reference",
        create_masks=True,
        # Both inputs already occupy the exact fixed registration canvas.
        # Cropping here can introduce an implicit origin shift even when VALIS
        # reports an identity rigid transform.
        crop_for_rigid_reg=False,
        # MerXen has already searched and applied handedness in
        # ``pre_orientation``. Asking VALIS to search reflections again both
        # risks cancelling the selected flip and enters a VALIS 1.2 LightGlue
        # path that promotes reflected keypoints to float64. The refinement
        # here must therefore remain a proper (positive-determinant) transform.
        check_for_reflections=False,
        resolution_xyu=(
            float(fixed_frame.registration_pixel_size_um),
            float(fixed_frame.registration_pixel_size_um),
            "µm",
        ),
        max_processed_image_dim_px=int(config.max_processed_image_dim_px),
        max_non_rigid_registration_dim_px=int(config.max_non_rigid_registration_dim_px),
        thumbnail_size=int(config.thumbnail_size),
        norm_method=norm_method,
        micro_rigid_registrar_cls=None,
    )
    if not hasattr(registrar, "non_rigid_reg_kwargs"):
        # VALIS 1.2 omits this attribute when non-rigid registration is
        # disabled, but unconditionally dereferences it during cleanup.
        registrar.non_rigid_reg_kwargs = {registration.NON_RIGID_REG_CLASS_KEY: None}
    try:
        with (
            _valis_locked_global_compatibility(registration.Valis),
            _valis_lightglue_cuda_compatibility(
                feature_matcher.LightGlueMatcher,
            ),
        ):
            rigid_registrar, _, summary = registrar.register(
                if_processing_cls=processor_cls,
                if_processing_kwargs={"input_is_preprocessed": True},
                processor_dict=processor_dict,
            )
        if rigid_registrar is None or rigid_registrar is False or summary is None:
            raise RuntimeError("VALIS locked non-rigid registration returned no result")
        moving_slide = registrar.get_slide(str(moving_path))
        fixed_slide = registrar.get_slide(str(fixed_path))
        shape_rc = fixed_frame.registration_shape_rc
        sample_xy = _sample_grid_xy(shape_rc, spacing=64)
        identity_diagnostics = _assert_valis_rigid_map_is_identity(
            moving_slide,
            fixed_slide,
            sample_xy=sample_xy,
            shape_rc=shape_rc,
        )
        # Do not fit a matrix to numerically near-identity samples. The exact
        # identity is a deliberate coordinate-frame invariant: the accepted
        # partial-overlap transform is the complete global transform.
        global_matrix = np.eye(3, dtype=np.float64)
        global_image = np.asarray(moving_pre_image)
        global_mask = np.asarray(moving_pre_mask)
        moving_features, fixed_features = _match_registered_features(
            fixed_frame.processed_image,
            global_image,
            fixed_frame.tissue_mask,
            global_mask,
            num_features=int(config.features.num_features),
            residual_threshold_px=float(config.features.ransac_threshold_px) * 2.0,
        )
        global_metrics = compute_dapi_metrics(
            fixed_frame.processed_image,
            global_image,
            fixed_frame.tissue_mask,
            global_mask,
            moving_feature_xy=moving_features,
            fixed_feature_xy=fixed_features,
        )
        _add_partial_overlap_metrics(
            global_metrics,
            fixed_image=fixed_frame.processed_image,
            moving_image=global_image,
            fixed_mask=fixed_frame.tissue_mask,
            moving_mask=global_mask,
            fixed_valid_mask=fixed_frame.valid_mask,
            moving_valid_mask=moving_pre_valid,
            config=config,
            pixel_size_um=fixed_frame.registration_pixel_size_um,
        )
        affine_metrics = affine_diagnostics(global_matrix)

        forward_field = None
        backward_field = None
        non_rigid_image = None
        non_rigid_mask = None
        non_rigid_metrics = None
        deformation_metrics = None
        if (
            config.non_rigid.enabled
            and getattr(moving_slide, "fwd_dxdy", None) is not None
        ):
            forward_field = _sample_forward_field(
                moving_slide,
                fixed_slide,
                global_matrix,
                shape_rc=shape_rc,
                spacing=int(config.non_rigid.field_sample_spacing_px),
            )
            raw_forward_field = forward_field
            backward_field = _sample_backward_field(
                moving_slide,
                fixed_slide,
                global_matrix,
                shape_rc=shape_rc,
                spacing=int(config.non_rigid.field_sample_spacing_px),
            )
            drift_metrics = _coherent_euclidean_drift_diagnostics(
                raw_forward_field,
                active_mask=shared_non_rigid_mask,
                pixel_size_um=fixed_frame.registration_pixel_size_um,
            )
            forward_field = _taper_displacement_field(
                raw_forward_field,
                shared_non_rigid_weight,
            )
            backward_field = _taper_displacement_field(
                backward_field,
                shared_non_rigid_weight,
            )
            non_rigid_image = _warp_with_backward_displacement(
                moving_pre_image,
                backward_field,
                output_shape_rc=shape_rc,
                interpolation=cv2.INTER_LINEAR,
            )
            non_rigid_mask = _warp_with_backward_displacement(
                moving_pre_mask,
                backward_field,
                output_shape_rc=shape_rc,
                interpolation=cv2.INTER_NEAREST,
            )
            non_rigid_valid = _warp_with_backward_displacement(
                moving_pre_valid,
                backward_field,
                output_shape_rc=shape_rc,
                interpolation=cv2.INTER_NEAREST,
            )
            non_rigid_metrics = compute_dapi_metrics(
                fixed_frame.processed_image,
                non_rigid_image,
                fixed_frame.tissue_mask,
                non_rigid_mask,
            )
            _add_partial_overlap_metrics(
                non_rigid_metrics,
                fixed_image=fixed_frame.processed_image,
                moving_image=non_rigid_image,
                fixed_mask=fixed_frame.tissue_mask,
                moving_mask=non_rigid_mask,
                fixed_valid_mask=fixed_frame.valid_mask,
                moving_valid_mask=non_rigid_valid,
                config=config,
                pixel_size_um=fixed_frame.registration_pixel_size_um,
            )
            deformation_metrics = displacement_diagnostics(
                forward_field,
                pixel_size_um=fixed_frame.registration_pixel_size_um,
            )
            deformation_metrics.update(drift_metrics)

        registrar_path = _ensure_registrar_pickle(registrar, output_dir)
        attempt_metadata = {
            "transform": "LockedIdentityNonRigid",
            "global_transform_locked": True,
            "valis_do_rigid": False,
            "valis_rigid_identity": identity_diagnostics,
            "shared_non_rigid_domain": {
                "pixels": int(np.count_nonzero(shared_non_rigid_mask)),
                "fraction_of_canvas": float(
                    np.count_nonzero(shared_non_rigid_mask) / shared_non_rigid_mask.size
                ),
                "feather_um": float(config.preprocessing.edge_taper_um),
                "input_mask_source": (
                    "fixed.valid_mask & moving_pre.valid_mask & "
                    "fixed.tissue_mask & moving_pre.tissue_mask"
                ),
            },
            "global_qc": global_metrics,
            "affine": affine_metrics,
            "non_rigid_qc": non_rigid_metrics,
            "deformation": deformation_metrics,
            "valis_summary": summary.to_dict(orient="records"),
        }
        return _ValisAttempt(
            global_matrix=global_matrix,
            forward_displacement=forward_field,
            backward_displacement=backward_field,
            global_image=global_image,
            global_mask=global_mask,
            non_rigid_image=non_rigid_image,
            non_rigid_mask=non_rigid_mask,
            global_metrics=global_metrics,
            non_rigid_metrics=non_rigid_metrics,
            affine_metrics=affine_metrics,
            deformation_metrics=deformation_metrics,
            moving_feature_xy=moving_features,
            fixed_feature_xy=fixed_features,
            transform_name="LockedIdentityNonRigid",
            registrar_path=registrar_path,
            metadata=attempt_metadata,
        )
    finally:
        # VALIS starts Bio-Formats' JVM lazily. Its shutdown is safe when the JVM
        # was never needed, and ensures repeated Nextflow tasks do not leak it.
        registration.kill_jvm()


def _non_rigid_backend(
    config: ValisAlignmentConfig,
    module: Any,
) -> tuple[type[Any] | None, dict[str, Any] | None]:
    if not config.non_rigid.enabled:
        return None, None
    if config.non_rigid.backend == "optical_flow":
        return (
            module.OpticalFlowWarper,
            {
                "smoothing_method": "gauss",
                "sigma_ratio": float(config.non_rigid.smoothing_sigma_ratio),
            },
        )

    import SimpleITK

    if not hasattr(SimpleITK, "ElastixImageFilter"):
        raise RuntimeError(
            "non_rigid.backend='simple_elastix' requires a SimpleITK build with "
            "ElastixImageFilter; choose 'optical_flow' explicitly otherwise"
        )
    params = module.SimpleElastixWarper.get_default_params(
        (config.max_non_rigid_registration_dim_px,) * 2,
        grid_spacing_ratio=float(config.non_rigid.grid_spacing_ratio),
    )
    params["MaximumNumberOfIterations"] = [
        str(int(config.non_rigid.maximum_iterations))
    ]
    return module.SimpleElastixWarper, {"params": params}


def _sample_forward_field(
    moving_slide: Any,
    fixed_slide: Any,
    global_matrix: np.ndarray,
    *,
    shape_rc: tuple[int, int],
    spacing: int,
) -> DisplacementField:
    x, y, xy = _field_grid(shape_rc, spacing)
    non_rigid_xy = moving_slide.warp_xy_from_to(
        xy,
        fixed_slide,
        src_pt_level=shape_rc,
        dst_slide_level=shape_rc,
        non_rigid=True,
    )
    global_xy = apply_affine_matrix(xy, global_matrix)
    return DisplacementField(
        x_coordinates=x,
        y_coordinates=y,
        displacement_xy=(non_rigid_xy - global_xy).reshape(len(y), len(x), 2),
    )


def _sample_backward_field(
    moving_slide: Any,
    fixed_slide: Any,
    global_matrix: np.ndarray,
    *,
    shape_rc: tuple[int, int],
    spacing: int,
) -> DisplacementField:
    x, y, xy = _field_grid(shape_rc, spacing)
    non_rigid_xy = fixed_slide.warp_xy_from_to(
        xy,
        moving_slide,
        src_pt_level=shape_rc,
        dst_slide_level=shape_rc,
        non_rigid=True,
    )
    global_inverse_xy = apply_affine_matrix(xy, np.linalg.inv(global_matrix))
    return DisplacementField(
        x_coordinates=x,
        y_coordinates=y,
        displacement_xy=(non_rigid_xy - global_inverse_xy).reshape(
            len(y),
            len(x),
            2,
        ),
    )


def _taper_displacement_field(
    field: DisplacementField,
    weight_image: Any,
) -> DisplacementField:
    """Taper a sampled field to zero outside the shared valid-tissue domain."""
    weight = np.asarray(weight_image, dtype=np.float32)
    xx, yy = np.meshgrid(
        np.asarray(field.x_coordinates, dtype=np.float32),
        np.asarray(field.y_coordinates, dtype=np.float32),
    )
    sampled_weight = cv2.remap(
        weight,
        xx,
        yy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return DisplacementField(
        x_coordinates=np.asarray(field.x_coordinates, dtype=np.float64),
        y_coordinates=np.asarray(field.y_coordinates, dtype=np.float64),
        displacement_xy=(
            np.asarray(field.displacement_xy, dtype=np.float64)
            * sampled_weight[..., np.newaxis]
        ),
    )


def _coherent_euclidean_drift_diagnostics(
    field: DisplacementField,
    *,
    active_mask: Any,
    pixel_size_um: float,
) -> dict[str, Any]:
    """Measure broad rotation/translation encoded in a nominally local field."""
    x = np.asarray(field.x_coordinates, dtype=np.float64)
    y = np.asarray(field.y_coordinates, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    source = np.column_stack([xx.ravel(), yy.ravel()])
    displacement = np.asarray(field.displacement_xy, dtype=np.float64).reshape(-1, 2)
    mask = np.asarray(active_mask) > 0
    x_idx = np.clip(np.rint(source[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y_idx = np.clip(np.rint(source[:, 1]).astype(int), 0, mask.shape[0] - 1)
    keep = (
        mask[y_idx, x_idx]
        & np.isfinite(source).all(axis=1)
        & np.isfinite(displacement).all(axis=1)
    )
    if int(np.count_nonzero(keep)) < 2:
        raise RuntimeError(
            "Too few sampled displacement vectors fall within the shared "
            "non-rigid registration domain"
        )
    source = source[keep]
    target = source + displacement[keep]
    coherent_matrix = fit_rigid_matrix(source, target)
    coherent_target = apply_affine_matrix(source, coherent_matrix)
    coherent_displacement = coherent_target - source
    local_residual = target - coherent_target
    center = np.mean(source, axis=0, keepdims=True)
    center_displacement = apply_affine_matrix(center, coherent_matrix)[0] - center[0]
    rotation_degrees = float(
        np.degrees(
            np.arctan2(
                coherent_matrix[1, 0],
                coherent_matrix[0, 0],
            )
        )
    )
    coherent_magnitude_um = np.linalg.norm(coherent_displacement, axis=1) * float(
        pixel_size_um
    )
    residual_magnitude_um = np.linalg.norm(local_residual, axis=1) * float(
        pixel_size_um
    )
    translation_um = center_displacement * float(pixel_size_um)
    return {
        "coherent_rotation_degrees": rotation_degrees,
        "coherent_translation_x_um": float(translation_um[0]),
        "coherent_translation_y_um": float(translation_um[1]),
        "coherent_translation_magnitude_um": float(np.linalg.norm(translation_um)),
        "coherent_drift_p95_um": float(np.percentile(coherent_magnitude_um, 95)),
        "local_residual_p95_um": float(np.percentile(residual_magnitude_um, 95)),
        "coherent_euclidean_matrix": coherent_matrix.tolist(),
        "coherent_fit_sample_count": int(len(source)),
    }


def _warp_with_backward_displacement(
    image: Any,
    backward_field: DisplacementField,
    *,
    output_shape_rc: tuple[int, int],
    interpolation: int,
    block_rows: int = 256,
) -> np.ndarray:
    """Warp an image using the sampled fixed-to-moving field in bounded memory."""
    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError(
            f"Displacement preview warping requires a 2D image, got {source.shape}"
        )
    height, width = (int(output_shape_rc[0]), int(output_shape_rc[1]))
    output = np.empty((height, width), dtype=source.dtype)
    x = np.arange(width, dtype=np.float64)
    rows_per_block = max(1, int(block_rows))
    for row_start in range(0, height, rows_per_block):
        row_stop = min(height, row_start + rows_per_block)
        y = np.arange(row_start, row_stop, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        destination_xy = np.column_stack([xx.ravel(), yy.ravel()])
        displacement = backward_field.sample(destination_xy)
        source_xy = destination_xy + displacement
        map_x = source_xy[:, 0].reshape(xx.shape).astype(np.float32)
        map_y = source_xy[:, 1].reshape(yy.shape).astype(np.float32)
        output[row_start:row_stop] = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return output


def _field_grid(
    shape_rc: tuple[int, int],
    spacing: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape_rc
    x = np.unique(
        np.append(np.arange(0, width, max(1, spacing), dtype=float), width - 1.0)
    )
    y = np.unique(
        np.append(np.arange(0, height, max(1, spacing), dtype=float), height - 1.0)
    )
    xx, yy = np.meshgrid(x, y)
    return x, y, np.column_stack([xx.ravel(), yy.ravel()])


def _sample_grid_xy(
    shape_rc: tuple[int, int],
    *,
    spacing: int,
) -> np.ndarray:
    _, _, xy = _field_grid(shape_rc, spacing)
    return xy


def _match_registered_features(
    fixed_image: np.ndarray,
    moving_image: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    *,
    num_features: int,
    residual_threshold_px: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not hasattr(cv2, "SIFT_create"):
        return None, None
    sift = cv2.SIFT_create(nfeatures=int(num_features))
    fixed_kp, fixed_desc = sift.detectAndCompute(
        np.asarray(fixed_image, dtype=np.uint8),
        (np.asarray(fixed_mask) > 0).astype(np.uint8) * 255,
    )
    moving_kp, moving_desc = sift.detectAndCompute(
        np.asarray(moving_image, dtype=np.uint8),
        (np.asarray(moving_mask) > 0).astype(np.uint8) * 255,
    )
    if fixed_desc is None or moving_desc is None:
        return None, None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(moving_desc, fixed_desc, k=2)
    matches = [
        first for first, second in pairs if first.distance < 0.75 * second.distance
    ]
    if not matches:
        return None, None
    moving_xy = np.asarray(
        [moving_kp[match.queryIdx].pt for match in matches],
        dtype=np.float64,
    )
    fixed_xy = np.asarray(
        [fixed_kp[match.trainIdx].pt for match in matches],
        dtype=np.float64,
    )
    residual = np.linalg.norm(moving_xy - fixed_xy, axis=1)
    keep = residual <= float(residual_threshold_px)
    if not np.any(keep):
        return None, None
    return moving_xy[keep], fixed_xy[keep]


def _create_disk_detector(
    feature_detectors: Any,
    *,
    num_features: int,
    device: Any,
) -> Any:
    """Construct VALIS DISK with CUDA-safe conversion to NumPy."""
    import torch
    from valis import preprocessing

    class MerxenDiskFD(feature_detectors.DiskFD):
        def _detect_and_compute(
            self: MerxenDiskFD,
            image: Any,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[np.ndarray, np.ndarray]:
            del args, kwargs
            tensor_image = preprocessing.img_to_tensor(image)
            with torch.inference_mode():
                result = self.disk(
                    tensor_image.to(self.device).float(),
                    n=self.num_features,
                    pad_if_not_divisible=True,
                )[0]
            return (
                result.keypoints.detach().cpu().numpy(),
                result.descriptors.detach().cpu().numpy(),
            )

    MerxenDiskFD.__name__ = "DiskFD"
    return MerxenDiskFD(num_features=int(num_features), device=device)


class _HostTensorResultProxy:
    """Copy tensor results to host memory before VALIS converts them to NumPy."""

    def __init__(self: _HostTensorResultProxy, wrapped: Any) -> None:
        self._wrapped = wrapped

    def __call__(
        self: _HostTensorResultProxy,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return _move_tensor_results_to_cpu(self._wrapped(*args, **kwargs))

    def __getattr__(self: _HostTensorResultProxy, name: str) -> Any:
        return getattr(self._wrapped, name)


def _move_tensor_results_to_cpu(value: Any) -> Any:
    """Recursively detach torch tensors and copy them to CPU."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_move_tensor_results_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_move_tensor_results_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensor_results_to_cpu(item) for key, item in value.items()}
    return value


@contextmanager
def _valis_lightglue_cuda_compatibility(
    matcher_cls: type[Any],
) -> Iterator[None]:
    """Make VALIS 1.2's CPU-only NumPy conversion safe for CUDA LightGlue."""
    original_match_images = matcher_cls.match_images

    def match_images_on_host(self: Any, *args: Any, **kwargs: Any) -> Any:
        original_lightglue = self.lg_matcher
        self.lg_matcher = _HostTensorResultProxy(original_lightglue)
        try:
            return original_match_images(self, *args, **kwargs)
        finally:
            self.lg_matcher = original_lightglue

    matcher_cls.match_images = match_images_on_host
    try:
        yield
    finally:
        matcher_cls.match_images = original_match_images


@contextmanager
def _valis_locked_global_compatibility(
    valis_cls: type[Any],
) -> Iterator[None]:
    """Complete VALIS 1.2's identity-only bookkeeping during ``register``.

    ``do_rigid=False`` creates identity matrices through
    ``rigid_register_partial``, but VALIS 1.2 later expects that registrar to
    carry a ``transformer`` attribute that only the full rigid path normally
    installs. Temporarily adding an identity-only transformer lets the
    original, pickle-safe VALIS class finish without estimating a global map.
    """
    original_rigid_register_partial = valis_cls.rigid_register_partial

    def locked_rigid_register_partial(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        rigid_registrar = original_rigid_register_partial(self, *args, **kwargs)
        if rigid_registrar is not False and rigid_registrar is not None:
            rigid_registrar.transformer = _LockedIdentityTransform()
        return rigid_registrar

    valis_cls.rigid_register_partial = locked_rigid_register_partial
    try:
        yield
    finally:
        valis_cls.rigid_register_partial = original_rigid_register_partial


def _ensure_registrar_pickle(registrar: Any, output_dir: Path) -> Path:
    existing = Path(getattr(registrar, "reg_f", ""))
    if existing.exists():
        return existing
    registrar_path = output_dir / "valis_registrar.pickle"
    with registrar_path.open("wb") as handle:
        pickle.dump(registrar, handle)
    return registrar_path


def _resolve_torch_device(device: str) -> Any:
    import torch

    normalized = str(device).strip().lower()
    if normalized in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def dependency_versions() -> dict[str, str]:
    """Return exact versions of VALIS and its registration stack."""
    packages = (
        "valis-wsi",
        "opencv-contrib-python-headless",
        "opencv-python",
        "scikit-image",
        "torch",
        "torchvision",
        "kornia",
        "SimpleITK",
        "scipy",
        "numpy",
        "pyvips",
        "jpype1",
        "scyjava",
    )
    versions = {package: _package_version(package) for package in packages}
    versions["opencv-cv2-runtime"] = str(cv2.__version__)
    return versions


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _coerce_shape(image: np.ndarray, shape_rc: tuple[int, int]) -> np.ndarray:
    arr = np.squeeze(np.asarray(image))
    if arr.shape == shape_rc:
        return arr
    output = np.zeros(shape_rc, dtype=arr.dtype)
    height = min(shape_rc[0], arr.shape[0])
    width = min(shape_rc[1], arr.shape[1])
    output[:height, :width] = arr[:height, :width]
    return output


def _write_tiff(path: Path, image: Any) -> None:
    import tifffile

    tifffile.imwrite(
        path,
        np.asarray(image),
        photometric="minisblack",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
