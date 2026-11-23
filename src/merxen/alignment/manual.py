"""Adapt the cortex robust landmark field to the paired alignment contract.

The GP maps reference to moving micrometres in the labelling package. Its pull
map and a numerically inverted push map are sampled into the same portable
bundle used by automatic alignment. No AT8/MAP2 settings or standalone presets
are changed here: this explicitly DAPI-only route fits clicks alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spatialdata as sd
from scipy.ndimage import map_coordinates

from merxen.alignment.bundle import DisplacementField, ValisTransformBundle
from merxen.alignment.dapi import create_dapi_tissue_mask
from merxen.alignment.frames import resolve_dapi_frame
from merxen.alignment.landmark_prepare import PAIRED_LANDMARK_ATTR, dapi_frame_signature
from merxen.alignment.models import TransformResult
from merxen.alignment.qc import plot_registration_overlay
from merxen.alignment.transforms import apply_affine_matrix
from merxen.config import AlignmentConfig
from merxen.io.image_source import image_to_cyx
from merxen.manual_registration.check import read_labelling_file
from merxen.manual_registration.field_fit import build_landmark_field, field_summary
from merxen.manual_registration.geometry import snap_crop
from merxen.manual_registration.landmark_field import LandmarkField
from merxen.manual_registration.presets import resolve_settings
from merxen.manual_registration.register import _read_prepared_store, fit_landmarks
from merxen.manual_registration.schema import (
    STORE_ATTR,
    canonical_sha256,
    labelling_content_sha256,
    read_root_attrs,
)


def _grid(
    shape_yx: tuple[int, int],
    pixel: float,
    spacing: float,
    *,
    extra_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape_yx
    low = np.minimum([0, 0], extra_points.min(axis=0)) - spacing
    high = (
        np.maximum(
            [(width - 1) * pixel, (height - 1) * pixel], extra_points.max(axis=0)
        )
        + spacing
    )
    x = np.linspace(
        low[0], high[0], max(2, int(np.ceil((high[0] - low[0]) / spacing)) + 1)
    )
    y = np.linspace(
        low[1], high[1], max(2, int(np.ceil((high[1] - low[1]) / spacing)) + 1)
    )
    xx, yy = np.meshgrid(x, y)
    return x, y, np.column_stack([xx.ravel(), yy.ravel()])


def _inverse(
    field: LandmarkField, moving_um: np.ndarray, config: AlignmentConfig
) -> np.ndarray:
    reference = apply_affine_matrix(moving_um, np.linalg.inv(field.trend_matrix()))
    for _ in range(config.manual.inverse_iterations):
        residual = field.mean(reference, chunk=8192) - moving_um
        if (
            np.max(np.linalg.norm(residual, axis=1), initial=0)
            <= config.manual.inverse_tolerance_um
        ):
            return reference
        # Full Newton updates also work for reflected/rotated sections. A fixed
        # point update in raw xy would fail even for an exact 180-degree rotation.
        jacobian = field.jacobian(reference)
        reference -= np.linalg.solve(jacobian, residual[..., None])[..., 0]
    raise ValueError(
        "Manual field inversion did not converge; inspect the landmark deformation"
    )


def register_manual_pair(
    merscope_sdata: Any, xenium_sdata: Any, config: AlignmentConfig
) -> TransformResult:
    """Fit finished landmarks, validate both directions, and save a paired bundle.

    Args:
        merscope_sdata: Current native moving data (not a previously warped image).
        xenium_sdata: Current native reference data.
        config: Alignment configuration selecting ``manual_landmarks``.

    Returns:
        A transform consumable by the usual in-place alignment materializer.

    Raises:
        ValueError: For stale/wrong/draft landmarks, changed native frames, a
            folded field, or an inconsistent inverse. No analysis store is
            modified until the caller initializes the validated transform.
    """
    if config.manual.prepared_store is None:
        raise ValueError("manual_landmarks requires manual.prepared_store")
    path = Path(config.manual.prepared_store)
    attrs = read_root_attrs(path)
    package = attrs.get(PAIRED_LANDMARK_ATTR, {})
    if package.get("version") != 1 or package.get("pair_id") != config.pair_id:
        raise ValueError(
            "Manual alignment needs the DAPI package prepared for this pair"
        )
    labelling_override = None
    if config.manual.labelling_path is not None:
        labelling_override, uploaded_store = read_labelling_file(
            config.manual.labelling_path
        )
        if uploaded_store is not None and (
            uploaded_store.get("dataset_name") != config.pair_id
            or uploaded_store.get("content_digest")
            != attrs[STORE_ATTR]["content_digest"]
        ):
            raise ValueError("Uploaded landmarks belong to a different DAPI package")
    store, labelling, raw, warnings = _read_prepared_store(
        path,
        run_id=config.pair_id,
        dataset_name=config.pair_id,
        check_fingerprints=False,
        require_finished=True,
        labelling_payload=labelling_override,
    )
    if labelling.store_grid_digest != store.grid_digest:
        raise ValueError("Landmarks must reference this package's grid digest")
    frames = {}
    for role, platform, sdata, image_config in (
        ("reference", "XENIUM", xenium_sdata, config.xenium_image),
        ("moving", "MERSCOPE", merscope_sdata, config.merscope_image),
    ):
        frame = resolve_dapi_frame(sdata, platform=platform, config=image_config)
        if dapi_frame_signature(frame) != package["sources"][role]["native"]:
            raise ValueError(
                f"Native {platform} DAPI changed since landmark package preparation"
            )
        frames[role] = frame
    # Resolve against the actual cortex preset, then specialize only this route.
    # Image-based evidence is intentionally unavailable for DAPI-only packages.
    settings = resolve_settings(
        "cortex_serial",
        reference_pixel_size_um=store.images["reference"].pixel_size_um,
        reference_channel="DAPI",
        row_settings=config.manual.settings,
    )
    if (
        settings.field_model != "robust_gp"
        or settings.refine_affine
        or settings.nonrigid_enabled
    ):
        raise ValueError(
            "manual_landmarks requires the robust cortex field without image refinement"
        )
    settings = settings.model_copy(
        update={"field_image_check": False, "field_window_observations": False}
    )
    similarity = fit_landmarks(labelling, store, settings, run_id=config.pair_id)
    crops = {
        role: snap_crop(
            getattr(labelling.crops, role),
            store.images[role].shape_yx,
            multiple=settings.crop_snap_px,
            min_size_px=settings.crop_min_size_px,
        )
        for role in ("reference", "moving")
    }
    report, _ = build_landmark_field(
        similarity=similarity,
        store=store,
        settings=settings,
        roles=None,
        crops=crops,
        image_check=False,
        windows=False,
        run_id=config.pair_id,
    )
    field = report.field
    if not field.converged:
        raise ValueError("Robust landmark field did not converge; review the landmarks")
    global_matrix = np.linalg.inv(field.trend_matrix())

    def image_to_package_um(role: str) -> np.ndarray:
        pixel = store.images[role].pixel_size_um
        return np.diag([pixel, pixel, 1]) @ np.linalg.inv(
            np.asarray(package["sources"][role]["package_to_native_image"])
        )

    native_corners = {}
    for role in ("reference", "moving"):
        height, width = frames[role].original_shape_rc
        native_corners[role] = apply_affine_matrix(
            [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
            image_to_package_um(role),
        )
    x, y, fixed = _grid(
        store.images["reference"].shape_yx,
        store.images["reference"].pixel_size_um,
        config.manual.grid_spacing_um,
        extra_points=native_corners["reference"],
    )
    jacobians = np.linalg.det(field.jacobian(fixed)) * np.sign(
        np.linalg.det(field.trend_matrix()[:2, :2])
    )
    if (
        not np.isfinite(jacobians).all()
        or jacobians.min() < config.manual.minimum_jacobian
        or jacobians.max() > config.manual.maximum_jacobian
    ):
        raise ValueError(
            "Manual deformation exceeds Jacobian limits; inspect the landmark fit"
        )
    pulled_fixed = field.mean(fixed, chunk=8192)
    backward = DisplacementField(
        x,
        y,
        (
            pulled_fixed - apply_affine_matrix(fixed, np.linalg.inv(global_matrix))
        ).reshape(len(y), len(x), 2),
    )
    # Partial overlap can pull fixed pixels outside the moving image. The push
    # grid must cover those positions too, or zero-fill outside its bounds gives
    # a discontinuous inverse at image edges. Include native block-mean margins.
    mx, my, moving = _grid(
        store.images["moving"].shape_yx,
        store.images["moving"].pixel_size_um,
        config.manual.grid_spacing_um,
        extra_points=np.vstack([native_corners["moving"], pulled_fixed]),
    )
    pushed = _inverse(field, moving, config)
    forward = DisplacementField(
        mx,
        my,
        (pushed - apply_affine_matrix(moving, global_matrix)).reshape(
            len(my), len(mx), 2
        ),
    )

    bundle = ValisTransformBundle(
        moving_dataset_to_image=frames["moving"].dataset_to_image_matrix,
        moving_image_to_registration=image_to_package_um("moving"),
        pre_matrix=np.eye(3),
        global_matrix=global_matrix,
        fixed_image_to_registration=image_to_package_um("reference"),
        fixed_dataset_to_image=frames["reference"].dataset_to_image_matrix,
        selected_mode="non_rigid",
        forward_displacement=forward,
        backward_displacement=backward,
    )
    # Off-node round trips catch inverse, interpolation and frame mistakes before
    # any mask/transcript is warped, including the native-image border pixels.
    probes = fixed[:: max(1, len(fixed) // 2000)] + config.manual.grid_spacing_um * 0.37
    probes = np.vstack([probes, native_corners["reference"]])
    pulled = field.mean(probes)
    inside = (
        (pulled[:, 0] >= mx[0])
        & (pulled[:, 0] <= mx[-1])
        & (pulled[:, 1] >= my[0])
        & (pulled[:, 1] <= my[-1])
        & (probes[:, 0] <= x[-1])
        & (probes[:, 1] <= y[-1])
    )
    probes = probes[inside]
    if len(probes) < 8:
        raise ValueError("Too little shared DAPI image area for a round-trip check")
    returned = apply_affine_matrix(
        probes, np.linalg.inv(global_matrix)
    ) + backward.sample(probes)
    returned = apply_affine_matrix(returned, global_matrix) + forward.sample(returned)
    errors = np.linalg.norm(returned - probes, axis=1)
    if not np.isfinite(errors).all() or errors.max() > max(
        2.0, 4 * config.manual.inverse_tolerance_um
    ):
        raise ValueError(
            "Manual push/pull round trip failed; refine the grid or landmarks"
        )
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    bundle.save(out)
    (out / "landmark_field.json").write_text(
        json.dumps(field.to_dict(), indent=2) + "\n"
    )
    (out / "landmarks.json").write_text(json.dumps(raw, indent=2) + "\n")
    pd.DataFrame(
        {
            "id": [p.id for p in similarity.pairs],
            "enabled": similarity.enabled,
            "class": report.pair_class,
            "weight": report.weight,
            "loo_um": report.loo_reference_um,
        }
    ).to_csv(out / "landmark_diagnostics.csv", index=False)
    preview, mask_to_package_um = _write_preview(path, field, store, config)
    qc = {
        "roundtrip_max_um": float(errors.max()),
        "jacobian_min": float(jacobians.min()),
        "jacobian_max": float(jacobians.max()),
        "landmarks": field_summary(
            report, outlier_p=settings.field_outlier_p, sd_summary=None
        ),
        "image_checks": "disabled_dapi_only",
    }
    metadata = {
        "pair_id": config.pair_id,
        "backend": "manual_landmarks",
        "status": "complete",
        "fixed_platform": "XENIUM",
        "moving_platform": "MERSCOPE",
        "coordinate_system_name": config.valis.coordinate_system_name,
        "transform_chain_path": str(out / "transform_chain.json"),
        "registered_preview_path": str(preview.name),
        "shared_tissue_mask_path": "shared_tissue_mask.npy",
        "coordinate_frames": {
            "fixed_platform": "XENIUM",
            "moving_platform": "MERSCOPE",
            "fixed_dataset_to_image_matrix": (
                np.linalg.inv(mask_to_package_um)
                @ image_to_package_um("reference")
                @ frames["reference"].dataset_to_image_matrix
            ).tolist(),
            "shared_tissue_mask_grid": "DAPI overview pixels",
        },
        "qc": qc,
        "parameters": settings.model_dump(mode="json"),
        "package_digest": store.content_digest,
        "landmark_digest": labelling_content_sha256(raw),
        "settings_digest": canonical_sha256(settings.model_dump(mode="json")),
        "warnings": [*warnings, *report.warnings],
    }
    (out / "registration_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return TransformResult(
        merscope_to_common={
            "type": "manual_landmark_field",
            "selected_mode": "nonrigid",
            "rigid_affine_matrix": bundle.global_dataset_matrix.tolist(),
        },
        xenium_to_common={"type": "identity", "affine_matrix": np.eye(3).tolist()},
        metadata=metadata,
        valis_transform=bundle,
    )


def _write_preview(
    path: Path, field: LandmarkField, store: Any, config: AlignmentConfig
) -> tuple[Path, np.ndarray]:
    output = config.output_dir
    images = sd.read_zarr(path, selection=("images",)).images
    reference = image_to_cyx(images["reference"])
    moving = image_to_cyx(images["moving"])
    step = max(1, int(np.ceil(max(reference.shape[-2:]) / 1200)))
    moving_step = max(1, int(np.ceil(max(moving.shape[-2:]) / 1600)))
    fixed = np.asarray(reference.data[0, ::step, ::step].compute())
    source = np.asarray(moving.data[0, ::moving_step, ::moving_step].compute())
    yy, xx = np.indices(fixed.shape)
    fixed_um = (
        np.column_stack([xx.ravel(), yy.ravel()])
        * step
        * store.images["reference"].pixel_size_um
    )
    moving_px = field.mean(fixed_um, chunk=8192) / (
        store.images["moving"].pixel_size_um * moving_step
    )
    warped = map_coordinates(
        source, [moving_px[:, 1], moving_px[:, 0]], order=1, mode="constant"
    ).reshape(fixed.shape)
    # Reports and paired annotation need shared tissue coverage. Estimate it at
    # tissue scale from each native DAPI, then pull only the moving mask. This
    # does not enable the standalone AT8/MAP2 checks or nucleus-scale refinement.
    fixed_mask = (
        create_dapi_tissue_mask(
            fixed,
            pixel_size_um=step * store.images["reference"].pixel_size_um,
            config=config.valis.preprocessing,
        )
        > 0
    )
    source_mask = (
        create_dapi_tissue_mask(
            source,
            pixel_size_um=moving_step * store.images["moving"].pixel_size_um,
            config=config.valis.preprocessing,
        )
        > 0
    )
    warped_mask = (
        map_coordinates(
            source_mask.astype(np.uint8),
            [moving_px[:, 1], moving_px[:, 0]],
            order=0,
            mode="constant",
        ).reshape(fixed.shape)
        > 0
    )
    shared = fixed_mask & warped_mask
    if not shared.any():
        raise ValueError("Manual alignment has no shared DAPI tissue coverage")
    np.save(output / "shared_tissue_mask.npy", shared.astype(np.uint8))
    result = output / "manual_dapi_overlay.png"
    plot_registration_overlay(
        fixed,
        warped,
        result,
        title="Manual landmarks: Xenium reference / warped MERSCOPE DAPI",
    )
    pixel = step * store.images["reference"].pixel_size_um
    return result, np.diag([pixel, pixel, 1])
