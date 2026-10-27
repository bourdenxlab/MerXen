"""Register step of manual landmark registration.

Turns the landmark pairs saved by the viewer into a registration of the moving
image onto the reference image: a robust similarity fit of the landmarks, then the
registration core (affine refinement on the DAPI channel and a guarded optical
flow), a result store with the moving image resampled onto the (cropped)
reference grid, tissue-scale and nucleus-scale QC, and CSV and JSON summaries.

Conventions (``docs/plans/manual-registration-plan.md``):

* A point is ``(x, y)`` in level-0 pixels of the **full, uncropped** image it
  belongs to unless a name says ``crop``; an integer is a pixel centre.
* Every matrix maps moving to reference: ``p_ref = M p_mov``. ``*_full`` matrices
  act on full frames, ``*_crop`` matrices on the cropped frames (origins are the
  effective crop origins ``o_r`` and ``o_m``).
* Residuals and errors are in reference micrometres.

Register never writes the prepared store. The result store is built in a private
sibling and renamed into place under the final path's write lock.
"""

from __future__ import annotations

import json
import logging
import math
import os
import resource
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spatialdata as sd
from pydantic import ValidationError
from spatialdata.models import Image2DModel

import merxen
from merxen.image_registration import (
    NONRIGID_REGISTRATION_ALGORITHM,
    REGISTRATION_ALGORITHM,
    RegistrationOutcome,
    RegistrationTuning,
    check_matrix_scale,
    measure_registration_overlay,
    push_points_through_registration,
    read_working_channel,
    register_onto_reference,
    registration_output_paths,
    write_alignment_matrix,
)
from merxen.io.ome_tiff import read_ome_tiff_info
from merxen.io.spatialdata_io import (
    spatialdata_write_lock,
    spatialdata_write_lock_path,
    write_spatialdata_metadata,
)
from merxen.io.spatialdata_schema import stamp_merxen_schema
from merxen.manual_registration.config import (
    ManualRegistrationRunConfig,
    ManualRegistrationSettings,
)
from merxen.manual_registration.geometry import (
    apply_matrix_xy,
    coarse_matrix_xy,
    crop_matrix,
    full_matrix,
    points_outside_crop,
    snap_crop,
    wrap_deg,
)
from merxen.manual_registration.landmark_fit import (
    RobustFitOptions,
    RobustSimilarityFit,
    robust_similarity,
)
from merxen.manual_registration.presets import (
    PRESET_VERSIONS,
    registration_tuning,
    resolve_settings,
)
from merxen.manual_registration.qc import (
    affine_pull,
    tissue_alignment_qc,
    write_tissue_qc_figure,
)
from merxen.manual_registration.schema import (
    CONVENTIONS,
    LABELLING_ATTR,
    RESULT_ATTR,
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    STORE_ATTR,
    STORE_SCHEMA,
    CropRect,
    LabellingAttrs,
    LandmarkPair,
    StoreAttrs,
    UnsupportedSchemaVersionError,
    canonical_sha256,
    file_fingerprint,
    labelling_content_sha256,
    parse_labelling_attrs,
    parse_store_attrs,
    read_root_attrs_locked,
    utc_now,
)
from merxen.manual_registration.sources import (
    crop_offset_affine,
    cropped_moving_image,
    cropped_reference_grid,
)
from merxen.memory import log_status
from merxen.viewer_cache.build import ViewerCacheParams, build_image_pyramids_in_store

logger = logging.getLogger(__name__)

LANDMARK_FIT_ALGORITHM = "landmark_similarity_v1"
SPREAD_WARNING_RATIO = 0.2
ROTATION_WARNING_DEG = 20.0
ALGORITHM_BLOCK = {
    "landmark_fit": LANDMARK_FIT_ALGORITHM,
    "refinement": REGISTRATION_ALGORITHM,
    "nonrigid": NONRIGID_REGISTRATION_ALGORITHM,
}
LANDMARK_CSV_COLUMNS = [
    "id",
    "colour",
    "enabled",
    "used_in_fit",
    "outlier",
    "outside_crop",
    "reference_x_px",
    "reference_y_px",
    "moving_x_px",
    "moving_y_px",
    "landmark_x_px",
    "landmark_y_px",
    "affine_x_px",
    "affine_y_px",
    "final_x_px",
    "final_y_px",
    "residual_landmark_um",
    "residual_loo_um",
    "residual_affine_um",
    "residual_final_um",
    "final_includes_flow",
    "push_converged",
]
_STAGES = ("landmark", "affine", "final")


class LandmarkFitError(ValueError):
    """The landmark pairs do not give an acceptable fit.

    Attributes:
        report: The fit diagnostics when the fit itself ran (``None`` when the
            input was rejected before fitting).
    """

    def __init__(
        self, message: str, *, report: LandmarkFitReport | None = None
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class LandmarkFitReport:
    """Result of ``fit_landmarks``.

    All ``(N,)`` arrays are aligned with ``pairs`` (labelling order) and cover
    disabled pairs too. Matrices and residuals are in full-image frames and
    reference micrometres.
    """

    robust: RobustSimilarityFit  # fit on the pairs passed to the robust fit (subset)
    pairs: list[LandmarkPair]  # all pairs, labelling order
    enabled: np.ndarray  # bool (N,)
    outside_crop: np.ndarray  # bool (N,), against the REQUESTED crops
    matrix_full: np.ndarray  # = robust.fit.matrix (full frames)
    warnings: list[str]
    fit_index: np.ndarray  # int indices of the pairs passed to robust_similarity
    used: np.ndarray  # bool, used in the final fit
    outlier: np.ndarray  # bool, flagged
    residuals_um: np.ndarray  # |M m - r| p_r for every pair (incl. disabled)
    loo_um: np.ndarray  # used pairs: robust LOO; all others: = residuals_um


def _ids(pairs: list[LandmarkPair], mask: np.ndarray) -> str:
    return ", ".join(str(pairs[i].id) for i in np.flatnonzero(mask))


def fit_landmarks(
    labelling: LabellingAttrs,
    store: StoreAttrs,
    settings: ManualRegistrationSettings,
    *,
    run_id: str | None = None,
) -> LandmarkFitReport:
    """Fit a robust similarity to the enabled landmark pairs and check it.

    The checks run in a fixed order (pair count, crop policy, fit, outlier
    policy, used count, spread, flip, scale, rotation, viewer floor); the first
    failing one raises. Warnings are returned in order of occurrence.

    Args:
        labelling: The viewer's labelling (pairs, coarse state, requested crops).
        store: The prepared store's attrs (pixel sizes).
        settings: Run settings (landmark thresholds and policies).
        run_id: Prefix of every message (default: the dataset name).

    Returns:
        The fit with per-pair diagnostics.

    Raises:
        LandmarkFitError: If the landmarks are unusable or inconsistent.
    """
    pfx = f"[{run_id or store.dataset_name}]"
    pairs = list(labelling.landmarks.pairs)
    count = len(pairs)
    moving = np.array([p.moving_xy_px for p in pairs], dtype=np.float64).reshape(-1, 2)
    reference = np.array([p.reference_xy_px for p in pairs], dtype=np.float64).reshape(
        -1, 2
    )
    enabled = np.array([p.enabled for p in pairs], dtype=bool)
    pixel_ref = store.images["reference"].pixel_size_um
    pixel_mov = store.images["moving"].pixel_size_um
    outside = points_outside_crop(
        reference, labelling.crops.reference
    ) | points_outside_crop(moving, labelling.crops.moving)
    warnings: list[str] = []

    n_enabled = int(enabled.sum())
    if n_enabled < settings.landmark_min_pairs:
        raise LandmarkFitError(
            f"{pfx} {n_enabled} enabled landmark pair(s); at least "
            f"{settings.landmark_min_pairs} are required (landmark_min_pairs). "
            "Add pairs in the viewer and Finish labelling again."
        )

    outside_enabled = enabled & outside
    candidates = enabled.copy()
    if outside_enabled.any():
        ids = _ids(pairs, outside_enabled)
        n_out = int(outside_enabled.sum())
        policy = settings.landmarks_outside_crop
        if policy == "use":
            warnings.append(
                f"{n_out} enabled landmark pair(s) outside a crop are used in the "
                f"fit: {ids}"
            )
        elif policy == "exclude":
            candidates &= ~outside
            warnings.append(
                f"{n_out} enabled landmark pair(s) outside a crop are excluded from "
                f"the fit: {ids}"
            )
        else:
            raise LandmarkFitError(
                f"{pfx} landmark pair(s) {ids} lie outside a crop "
                "(landmarks_outside_crop=fail); move the crop or the pairs"
            )
    if int(candidates.sum()) < 3:
        raise LandmarkFitError(
            f"{pfx} fewer than 3 pairs left for the fit "
            f"({int(candidates.sum())} after the crop policy)"
        )

    fit_index = np.flatnonzero(candidates)
    options = RobustFitOptions(
        floor_um=settings.landmark_outlier_floor_um,
        k=settings.landmark_outlier_k,
        max_outlier_fraction=settings.landmark_max_outlier_fraction,
        min_used_pairs=settings.landmark_min_used_pairs,
        trim_outliers=settings.landmark_outlier_policy == "exclude",
    )
    try:
        robust = robust_similarity(
            moving[fit_index],
            reference[fit_index],
            reflection=labelling.coarse.mirror_x,
            moving_pixel_size_um=pixel_mov,
            reference_pixel_size_um=pixel_ref,
            options=options,
        )
    except ValueError as exc:
        raise LandmarkFitError(f"{pfx} the landmark fit failed: {exc}") from exc

    matrix = robust.fit.matrix
    residuals = (
        np.linalg.norm(apply_matrix_xy(matrix, moving) - reference, axis=1) * pixel_ref
    )
    loo = residuals.copy()
    loo[fit_index] = robust.loo_um
    used = np.zeros(count, dtype=bool)
    used[fit_index] = robust.used
    outlier = np.zeros(count, dtype=bool)
    outlier[fit_index] = robust.outlier
    report = LandmarkFitReport(
        robust=robust,
        pairs=pairs,
        enabled=enabled,
        outside_crop=outside,
        matrix_full=matrix,
        warnings=warnings,
        fit_index=fit_index,
        used=used,
        outlier=outlier,
        residuals_um=residuals,
        loo_um=loo,
    )

    if outlier.any():
        ids = _ids(pairs, outlier)
        n_out = int(outlier.sum())
        outlier_policy = settings.landmark_outlier_policy
        if outlier_policy == "fail":
            raise LandmarkFitError(
                f"{pfx} landmark pair(s) flagged as outliers: {ids} "
                "(landmark_outlier_policy=fail)",
                report=report,
            )
        verb = "excluded" if outlier_policy == "exclude" else "kept"
        warnings.append(
            f"{n_out} landmark pair(s) flagged as outlier and {verb}: {ids}"
        )

    n_used = int(used.sum())
    if n_used < settings.landmark_min_used_pairs:
        raise LandmarkFitError(
            f"{pfx} only {n_used} landmark pair(s) are used in the fit; at least "
            f"{settings.landmark_min_used_pairs} are required "
            "(landmark_min_used_pairs)",
            report=report,
        )

    spread = robust.spread_ratio
    if spread < settings.landmark_min_spread_ratio:
        raise LandmarkFitError(
            f"{pfx} the used landmarks are nearly collinear (spread ratio "
            f"{spread:.3f} < {settings.landmark_min_spread_ratio}); add pairs across "
            "the whole tissue",
            report=report,
        )
    if spread < SPREAD_WARNING_RATIO:
        warnings.append(
            f"landmark spread ratio {spread:.2f} < {SPREAD_WARNING_RATIO}; add pairs "
            "away from the line through the existing ones"
        )

    if (
        robust.other_class_median_um
        < settings.landmark_reflection_check_ratio * robust.chosen_class_median_um
    ):
        mirrored = labelling.coarse.mirror_x
        raise LandmarkFitError(
            f"{pfx} the landmarks say the moving image is "
            f"{'not ' if mirrored else ''}mirrored but the coarse flip says otherwise "
            f"(median residual {robust.chosen_class_median_um:.1f} um with the "
            f"current flip, {robust.other_class_median_um:.1f} um flipped); fix the "
            "flip in the viewer",
            report=report,
        )

    try:
        check_matrix_scale(
            matrix,
            image_pixel_size_um=pixel_mov,
            reference_pixel_size_um=pixel_ref,
            tolerance=settings.matrix_scale_tolerance,
        )
    except ValueError as exc:
        expected = pixel_mov / pixel_ref
        observed = robust.fit.scale
        raise LandmarkFitError(
            f"{pfx} landmark fit scale {observed:.5f} differs from the pixel-size "
            f"ratio {expected:.5f} by {observed / expected - 1:+.1%} "
            f"(matrix_scale_tolerance {settings.matrix_scale_tolerance:.0%}); check "
            "the pixel sizes in the samplesheet or the landmark pairs; for adjacent "
            "serial sections run with same_section, set "
            "registration_settings=same_section:matrix_scale_tolerance=0.05",
            report=report,
        ) from exc

    difference = wrap_deg(robust.fit.rotation_deg - labelling.coarse.rotation_deg)
    if abs(difference) > ROTATION_WARNING_DEG:
        warnings.append(
            f"landmark fit rotation {robust.fit.rotation_deg:.1f} deg differs from "
            f"the coarse rotation {labelling.coarse.rotation_deg:.1f} deg by "
            f"{difference:.1f} deg"
        )

    viewer_floor = labelling.outlier_floor_um
    if viewer_floor is not None and not math.isclose(
        viewer_floor, settings.landmark_outlier_floor_um, rel_tol=1e-9
    ):
        warnings.append(
            f"the viewer's outlier floor was {viewer_floor:g} um, this fit uses "
            f"{settings.landmark_outlier_floor_um:g} um; outlier flags can differ "
            "from the viewer's Preview fit"
        )
    return report


def fit_summary(
    report: LandmarkFitReport,
    *,
    store: StoreAttrs,
    settings: ManualRegistrationSettings,
    labelling: LabellingAttrs | None = None,
) -> dict[str, Any]:
    """Return the ``fit`` block of the result attrs (plain Python numbers).

    Args:
        report: The fit report.
        store: The prepared store's attrs.
        settings: The run settings.
        labelling: The parsed labelling (for the viewer's outlier floor).
    """
    robust = report.robust
    fit = robust.fit
    used_residuals = report.residuals_um[report.used]
    pixel_ref = store.images["reference"].pixel_size_um
    pixel_mov = store.images["moving"].pixel_size_um
    viewer_floor = labelling.outlier_floor_um if labelling is not None else None
    return {
        "model": "similarity",
        "reflected": bool(fit.reflected),
        "scale": float(fit.scale),
        "expected_scale": float(pixel_mov / pixel_ref),
        "scale_ratio": float(robust.scale_ratio),
        "rotation_deg": float(fit.rotation_deg),
        "rms_um": float(np.sqrt(np.mean(used_residuals**2))),
        "median_um": float(np.median(used_residuals)),
        "n_pairs": int(len(report.pairs)),
        "n_enabled": int(report.enabled.sum()),
        "n_used": int(report.used.sum()),
        "n_outliers": int(report.outlier.sum()),
        "n_outside_crop": int((report.outside_crop & report.enabled).sum()),
        "spread_ratio": float(robust.spread_ratio),
        "chosen_class_median_um": float(robust.chosen_class_median_um),
        "other_class_median_um": float(robust.other_class_median_um),
        "outlier_floor_um": float(settings.landmark_outlier_floor_um),
        "viewer_outlier_floor_um": (
            None if viewer_floor is None else float(viewer_floor)
        ),
        "outlier_policy": settings.landmark_outlier_policy,
    }


# ---------------------------------------------------------------------------
# Validation of the prepared store
# ---------------------------------------------------------------------------


def _wrap_parse_error(pfx: str, what: str, exc: Exception) -> Exception:
    if isinstance(exc, UnsupportedSchemaVersionError):
        return UnsupportedSchemaVersionError(f"{pfx} {exc}")
    return ValueError(f"{pfx} {what} is not valid: {exc}")


def _read_prepared_store(
    store_path: Path,
    *,
    run_id: str,
    dataset_name: str,
    check_fingerprints: bool,
    require_finished: bool,
) -> tuple[StoreAttrs, LabellingAttrs, dict[str, Any], list[str]]:
    pfx = f"[{run_id}]"
    path = Path(store_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{pfx} prepared store not found: {path}; run "
            "workflows/manual_registration_prepare.nf"
        )
    attrs = read_root_attrs_locked(path)
    if STORE_ATTR not in attrs:
        raise ValueError(f"{pfx} {path} is not a manual-registration store")
    try:
        store = parse_store_attrs(attrs[STORE_ATTR])
    except (UnsupportedSchemaVersionError, ValidationError) as exc:
        raise _wrap_parse_error(pfx, STORE_ATTR, exc) from exc
    if store.kind != "prepared":
        raise ValueError(
            f"{pfx} {path} is a result store; give the prepared store "
            "(<id>_prepared.zarr)"
        )
    if store.dataset_name != dataset_name:
        raise ValueError(
            f"{pfx} the store belongs to dataset {store.dataset_name!r}, but the "
            f"run is for {dataset_name!r}"
        )
    hint = (
        "open it in the viewer ▸ Manual Registration ▸ Finish labelling and run again"
    )
    if LABELLING_ATTR not in attrs:
        raise ValueError(f"{pfx} no landmarks in {path}; {hint}")
    raw_labelling = attrs[LABELLING_ATTR]
    try:
        labelling = parse_labelling_attrs(raw_labelling)
    except (UnsupportedSchemaVersionError, ValidationError) as exc:
        raise _wrap_parse_error(pfx, LABELLING_ATTR, exc) from exc
    if require_finished and labelling.status != "finished":
        raise ValueError(f"{pfx} the labelling is a draft; {hint}")

    warnings: list[str] = []
    for role in ("reference", "moving"):
        image = store.images[role]
        labelled = getattr(labelling.images, role)
        if tuple(labelled.shape_yx) != tuple(image.shape_yx) or not math.isclose(
            labelled.pixel_size_um, image.pixel_size_um, rel_tol=1e-9
        ):
            raise ValueError(
                f"{pfx} landmarks were placed on a different {role} image "
                "(shape/pixel size differ); re-label in the viewer"
            )
    if (
        labelling.store_grid_digest is not None
        and labelling.store_grid_digest != store.grid_digest
    ):
        warnings.append(
            f"labelling was saved against grid digest {labelling.store_grid_digest}; "
            "shapes and pixel sizes still match"
        )
    if check_fingerprints:
        for role in ("reference", "moving"):
            record = store.images[role].ome_tiff
            rerun = "re-run prepare (workflows/manual_registration_prepare.nf)"
            if record is None:
                raise ValueError(f"{pfx} the {role} image has no OME-TIFF record")
            file = Path(record.path)
            if not file.exists():
                raise FileNotFoundError(
                    f"{pfx} the {role} OME-TIFF {file} is missing: {rerun}"
                )
            if file_fingerprint(file) != record.fingerprint:
                raise ValueError(
                    f"{pfx} the {role} OME-TIFF changed since prepare "
                    f"(fingerprint mismatch): {rerun}"
                )
    try:
        recomputed = labelling_content_sha256(raw_labelling)
    except ValueError as exc:
        raise ValueError(f"{pfx} {LABELLING_ATTR} is not valid: {exc}") from exc
    if labelling.content_sha256 is not None and labelling.content_sha256 != recomputed:
        warnings.append(
            "stored labelling content_sha256 differs from the recomputed one; using "
            "the recomputed value"
        )
    return store, labelling, dict(raw_labelling), warnings


def _validate_prepared_store(
    store_path: Path,
    *,
    run_id: str,
    dataset_name: str,
    check_fingerprints: bool = True,
    require_finished: bool = True,
) -> tuple[StoreAttrs, LabellingAttrs, list[str]]:
    """Read and validate a prepared store and its labelling, writing nothing.

    Args:
        store_path: The prepared store.
        run_id: Prefix of every message.
        dataset_name: The dataset the run is for.
        check_fingerprints: Also compare the OME-TIFF fingerprints.
        require_finished: Refuse a ``draft`` labelling.

    Returns:
        ``(store attrs, labelling, warnings)``.

    Raises:
        FileNotFoundError: If the store or an OME-TIFF is missing.
        UnsupportedSchemaVersionError: If an attrs payload is from a newer writer.
        ValueError: On any other problem.
    """
    store, labelling, _raw, warnings = _read_prepared_store(
        store_path,
        run_id=run_id,
        dataset_name=dataset_name,
        check_fingerprints=check_fingerprints,
        require_finished=require_finished,
    )
    return store, labelling, warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _provenance_digest(
    *,
    preset: str,
    run_label: str | None,
    settings: ManualRegistrationSettings,
    labelling_sha: str,
    store: StoreAttrs,
) -> str:
    """Digest of everything that determines a run's result (not the MerXen version)."""
    fingerprints = {}
    for role in ("reference", "moving"):
        record = store.images[role].ome_tiff
        fingerprints[role] = None if record is None else record.fingerprint
    return canonical_sha256(
        {
            "schema_version": 1,
            "preset": preset,
            "preset_version": PRESET_VERSIONS[preset],
            "run_label": run_label,
            "settings": settings.model_dump(mode="json"),
            "algorithm": ALGORITHM_BLOCK,
            "labelling_content_sha256": labelling_sha,
            "prepared_content_digest": store.content_digest,
            "ome_tiff_fingerprints": fingerprints,
        }
    )


def _plain_json(value: Any) -> Any:
    """Convert to plain JSON data: numpy to Python, non-finite floats to ``None``."""
    if isinstance(value, dict):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        return _plain_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _publish(src: Path, dst: Path) -> Path:
    """Copy ``src`` to ``dst`` atomically (temporary sibling, then ``os.replace``)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(f"{dst.name}.partial-{uuid.uuid4().hex}")
    try:
        shutil.copy2(src, partial)
        os.replace(partial, dst)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return dst


def _xy(point: np.ndarray) -> list[float]:
    return [float(point[0]), float(point[1])]


def _landmark_records(
    report: LandmarkFitReport,
    *,
    reference_pixel_size_um: float,
    reference_origin_xy: tuple[int, int],
    moving_origin_xy: tuple[int, int],
    landmark_full: np.ndarray,
    final_affine_crop: np.ndarray,
    flow_yx: np.ndarray | None,
    flow_working_to_reference: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Place every moving landmark in the reference at each registration stage.

    Positions are full reference pixels. ``landmark`` is the landmark similarity,
    ``affine`` the refined affine and ``final`` the affine plus flow, the latter
    pushed through the flow in the crop frame and shifted back. Residuals are
    reference micrometres.
    """
    pairs = report.pairs
    moving = np.array([p.moving_xy_px for p in pairs], dtype=np.float64).reshape(-1, 2)
    reference = np.array([p.reference_xy_px for p in pairs], dtype=np.float64).reshape(
        -1, 2
    )
    origin_r = np.asarray(reference_origin_xy, dtype=np.float64)
    origin_m = np.asarray(moving_origin_xy, dtype=np.float64)
    affine_full = full_matrix(
        final_affine_crop,
        reference_origin_xy=reference_origin_xy,
        moving_origin_xy=moving_origin_xy,
    )
    landmark = apply_matrix_xy(landmark_full, moving)
    affine = apply_matrix_xy(affine_full, moving)
    converged = np.ones(len(pairs), dtype=bool)
    includes_flow = np.zeros(len(pairs), dtype=bool)
    if flow_yx is not None and flow_working_to_reference is not None:
        pushed, converged = push_points_through_registration(
            moving - origin_m,
            final_matrix=final_affine_crop,
            flow_yx=flow_yx,
            working_to_reference=flow_working_to_reference,
        )
        final = pushed + origin_r
        grid_h, grid_w = flow_yx.shape[1:]
        q = apply_matrix_xy(
            np.linalg.inv(flow_working_to_reference) @ final_affine_crop,
            moving - origin_m,
        )
        includes_flow = (
            (q[:, 0] >= 0)
            & (q[:, 0] <= grid_w - 1)
            & (q[:, 1] >= 0)
            & (q[:, 1] <= grid_h - 1)
        )
    else:
        final = affine
    stage_positions = {"landmark": landmark, "affine": affine, "final": final}
    residual = {
        stage: np.linalg.norm(position - reference, axis=1) * reference_pixel_size_um
        for stage, position in stage_positions.items()
    }
    records: list[dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        records.append(
            {
                "id": int(pair.id),
                "colour": pair.colour,
                "enabled": bool(pair.enabled),
                "used_in_fit": bool(report.used[i]),
                "outlier": bool(report.outlier[i]),
                "outside_crop": bool(report.outside_crop[i]),
                "reference_xy_px": _xy(reference[i]),
                "moving_xy_px": _xy(moving[i]),
                "moving_in_reference_xy_px": {
                    stage: _xy(position[i])
                    for stage, position in stage_positions.items()
                },
                "residual_um": {
                    "landmark": float(residual["landmark"][i]),
                    "loo": float(report.loo_um[i]),
                    "affine": float(residual["affine"][i]),
                    "final": float(residual["final"][i]),
                },
                "final_includes_flow": bool(includes_flow[i]),
                "push_converged": bool(converged[i]),
            }
        )
    return records


def _residual_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Median, 90th percentile and maximum residual per group and stage.

    Groups: ``used`` (used in the fit) and ``held_out`` (disabled pairs).
    """
    groups = {
        "used": [r for r in records if r["used_in_fit"]],
        "held_out": [r for r in records if not r["enabled"]],
    }
    summary: dict[str, Any] = {}
    for name, members in groups.items():
        summary[name] = {}
        for stage in _STAGES:
            values = np.array([r["residual_um"][stage] for r in members], dtype=float)
            if values.size == 0:
                summary[name][stage] = {
                    "n": 0,
                    "median": None,
                    "p90": None,
                    "max": None,
                }
            else:
                summary[name][stage] = {
                    "n": int(values.size),
                    "median": float(np.median(values)),
                    "p90": float(np.percentile(values, 90)),
                    "max": float(values.max()),
                }
    return summary


def _landmarks_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "id": r["id"],
                "colour": r["colour"],
                "enabled": r["enabled"],
                "used_in_fit": r["used_in_fit"],
                "outlier": r["outlier"],
                "outside_crop": r["outside_crop"],
                "reference_x_px": r["reference_xy_px"][0],
                "reference_y_px": r["reference_xy_px"][1],
                "moving_x_px": r["moving_xy_px"][0],
                "moving_y_px": r["moving_xy_px"][1],
                "landmark_x_px": r["moving_in_reference_xy_px"]["landmark"][0],
                "landmark_y_px": r["moving_in_reference_xy_px"]["landmark"][1],
                "affine_x_px": r["moving_in_reference_xy_px"]["affine"][0],
                "affine_y_px": r["moving_in_reference_xy_px"]["affine"][1],
                "final_x_px": r["moving_in_reference_xy_px"]["final"][0],
                "final_y_px": r["moving_in_reference_xy_px"]["final"][1],
                "residual_landmark_um": r["residual_um"]["landmark"],
                "residual_loo_um": r["residual_um"]["loo"],
                "residual_affine_um": r["residual_um"]["affine"],
                "residual_final_um": r["residual_um"]["final"],
                "final_includes_flow": r["final_includes_flow"],
                "push_converged": r["push_converged"],
            }
        )
    return pd.DataFrame(rows, columns=LANDMARK_CSV_COLUMNS)


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(matrix, dtype=np.float64)]


def _result_store_attr(
    *,
    store: StoreAttrs,
    config: ManualRegistrationRunConfig,
    store_path: Path,
    crop_reference: CropRect,
    crop_moving: CropRect,
    kept_reference_names: list[str],
) -> dict[str, Any]:
    """Build and validate the ``STORE_ATTR`` payload of a result store."""
    ref = store.images["reference"]
    mov = store.images["moving"]
    payload = {
        "schema": STORE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": "result",
        "dataset_name": store.dataset_name,
        "created_at": utc_now(),
        "merxen_version": merxen.__version__,
        "conventions": dict(store.conventions or CONVENTIONS),
        "grid_digest": store.grid_digest,
        "content_digest": store.content_digest,
        "channel_matching": (
            None
            if store.channel_matching is None
            else store.channel_matching.model_dump()
        ),
        "run_id": config.run_id,
        "preset": config.preset,
        "run_label": config.run_label,
        "prepared_store": {
            "path": str(store_path),
            "content_digest": store.content_digest,
            "grid_digest": store.grid_digest,
        },
        "images": {
            "reference": {
                "element": "reference",
                "role": "reference",
                "shape_yx": [crop_reference.height(), crop_reference.width()],
                "full_shape_yx": list(ref.shape_yx),
                "crop": crop_reference.model_dump(),
                "pixel_size_um": ref.pixel_size_um,
                "dtype": ref.dtype,
                "channel_names": kept_reference_names,
                "registration_channel": ref.registration_channel,
            },
            "moving": {
                "element": "moving_registered",
                "role": "moving",
                "shape_yx": [crop_reference.height(), crop_reference.width()],
                "pixel_size_um": ref.pixel_size_um,
                "dtype": mov.dtype,
                "channel_names": list(mov.channel_names),
                "registration_channel": mov.registration_channel,
                "source_shape_yx": list(mov.shape_yx),
                "source_pixel_size_um": mov.pixel_size_um,
                "source_crop": crop_moving.model_dump(),
            },
        },
    }
    plain: dict[str, Any] = _plain_json(payload)
    parse_store_attrs(plain)
    return plain


def _superseded_path(final: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = final.with_name(f"{final.name}.superseded-{stamp}")
    n = 2
    while candidate.exists():
        candidate = final.with_name(f"{final.name}.superseded-{stamp}-{n}")
        n += 1
    return candidate


def _stored_result(final: Path) -> dict[str, Any] | None:
    """Read the ``RESULT_ATTR`` of an existing result store (``None`` if unreadable)."""
    if not final.exists():
        return None
    try:
        payload = read_root_attrs_locked(final).get(RESULT_ATTR)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _outputs_from_stored(
    stored: dict[str, Any], run_id: str, out: Path
) -> dict[str, Path]:
    outputs = dict(stored.get("outputs", {}))
    paths = {
        key: Path(outputs[key])
        for key in (
            "result_store",
            "landmark_matrix_csv",
            "final_matrix_csv",
            "landmarks_csv",
            "summary_json",
            "tissue_qc_png",
        )
        if key in outputs
    }
    paths["task_summary_json"] = out / f"{run_id}_manual_registration_summary.json"
    return paths


def _reusable(stored: dict[str, Any] | None, digest: str) -> bool:
    """Whether the stored result and its durable files are this run's, complete.

    The durable CSVs and the summary are published after the store swap, the
    summary last. A store that carries ``digest`` therefore proves nothing about
    the files until the durable summary carries it too (a failed or killed
    publish leaves the previous run's files next to the new store). The core QC
    files the result lists must exist as well: a run deletes them before the
    core and a failure in the core leaves them missing.
    """
    if stored is None:
        return False
    if stored.get("status") != "complete" or stored.get("provenance_digest") != digest:
        return False
    outputs = dict(stored.get("outputs", {}))
    needed = (
        "landmark_matrix_csv",
        "final_matrix_csv",
        "landmarks_csv",
        "summary_json",
    )
    if not all(key in outputs and Path(outputs[key]).exists() for key in needed):
        return False
    # The core QC files are deleted before the core runs, so a run that fails in
    # the core leaves the previous result current with its listed files missing.
    core_qc = outputs.get("core_qc") or {}
    if not all(Path(path).exists() for path in dict(core_qc).values()):
        logger.info(
            "The result store is current but core QC files it lists are missing "
            "(an earlier run failed in the core?); registering again"
        )
        return False
    try:
        durable = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        not isinstance(durable, dict)
        or durable.get("status") != "complete"
        or durable.get("provenance_digest") != digest
    ):
        logger.info(
            "The result store is current but its durable files are from another "
            "run (an interrupted publish?); registering again"
        )
        return False
    return True


# (outputs.core_qc key, registration_output_paths key) of the files the core writes.
_CORE_QC_FILES = (
    ("registration_qc_png", "qc_plot"),
    ("registration_local_shifts_csv", "local_shifts"),
    ("nonrigid_flow_npz", "nonrigid_flow"),
    ("nonrigid_qc_png", "nonrigid_qc_plot"),
)


def _remove_core_qc_files(qc_paths: dict[str, Path]) -> None:
    """Delete this run id's core QC files (PNG and PDF twins) from earlier runs.

    The core writes the flow and the non-rigid plot only when a flow was computed
    and the refinement plot and CSV only when the refinement measured something,
    so a file that is still there after a run may belong to an earlier run (and to
    another crop frame).
    """
    for _name, key in _CORE_QC_FILES:
        path = Path(qc_paths[key])
        path.unlink(missing_ok=True)
        if path.suffix == ".png":
            path.with_suffix(".pdf").unlink(missing_ok=True)


def _core_qc_listing(
    qc_paths: dict[str, Path], outcome: RegistrationOutcome
) -> dict[str, str]:
    """The core QC files written by this run, decided from the outcome.

    The refinement plot and CSV exist when the refinement measured its windows
    (``refinement.before``); the flow archive and the non-rigid plot when a flow
    was computed (``nonrigid.result``), applied or rejected.
    """
    refinement, nonrigid = outcome.refinement, outcome.nonrigid
    written = {
        "qc_plot": refinement.before is not None
        and refinement.working_to_reference is not None,
        "local_shifts": refinement.before is not None
        and refinement.working_to_reference is not None,
        "nonrigid_flow": nonrigid.result is not None
        and nonrigid.working_to_reference is not None,
        "nonrigid_qc_plot": nonrigid.result is not None
        and nonrigid.working_to_reference is not None,
    }
    return {
        name: str(qc_paths[key])
        for name, key in _CORE_QC_FILES
        if written[key] and Path(qc_paths[key]).exists()
    }


def _remove_superseded_siblings(final: Path, *, keep: Path | None = None) -> None:
    """Remove ``<final>.superseded-*`` stores left by a run that died mid-swap."""
    for stale in final.parent.glob(f"{final.name}.superseded-*"):
        if keep is not None and stale == keep:
            continue
        try:
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
            else:
                stale.unlink()
            logger.info("Removed superseded result store %s", stale.name)
        except OSError as exc:
            logger.warning("Could not remove superseded store %s: %s", stale.name, exc)


def _remove_stale_siblings(result_dir: Path, run_id: str) -> None:
    for stale in result_dir.glob(f".{run_id}_registered.zarr.partial-*"):
        try:
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
            else:
                stale.unlink()
            logger.info("Removed stale register leftover %s", stale.name)
        except OSError as exc:
            logger.warning("Could not remove stale leftover %s: %s", stale.name, exc)


# ---------------------------------------------------------------------------
# run_manual_registration
# ---------------------------------------------------------------------------


def run_manual_registration(
    config: ManualRegistrationRunConfig, *, force_rerun: bool = False
) -> dict[str, Path]:
    """Register the moving image onto the reference from the saved landmarks.

    Steps: validate the prepared store, resolve the settings, fit the landmarks,
    skip when an identical result exists, write the cropped reference into a
    private temporary result store, run the registration core, place every
    landmark at each stage, compute the QC, write CSVs and the summary, and swap
    the store into place.

    Args:
        config: The run configuration.
        force_rerun: Re-register even when an identical result exists (also
            ``config.force``).

    Returns:
        ``result_store``, ``landmark_matrix_csv``, ``final_matrix_csv``,
        ``landmarks_csv``, ``summary_json`` and ``tissue_qc_png`` (durable
        paths) and ``task_summary_json`` (copy in ``config.output_dir``).

    Raises:
        FileNotFoundError: If the prepared store or an OME-TIFF is missing.
        LandmarkFitError: If the landmarks are unusable.
        ValueError: On any other invalid input.
    """
    started = time.perf_counter()
    force = config.force or force_rerun
    run_id = config.run_id
    pfx = f"[{run_id}]"
    result_dir = Path(config.result_dir).resolve()
    final = Path(config.result_store_path).resolve()
    out = Path(config.output_dir)

    # Step 1: validate everything before any write.
    store_path = Path(config.prepared_store_path).resolve()
    store, labelling, raw_labelling, run_warnings = _read_prepared_store(
        store_path,
        run_id=run_id,
        dataset_name=config.dataset_name,
        check_fingerprints=True,
        require_finished=True,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    ref = store.images["reference"]
    mov = store.images["moving"]
    pixel_ref, pixel_mov = ref.pixel_size_um, mov.pixel_size_um
    labelling_sha = labelling_content_sha256(raw_labelling)

    # Step 2: settings.
    settings = resolve_settings(
        config.preset,
        reference_pixel_size_um=pixel_ref,
        reference_channel=ref.registration_channel,
        global_settings=config.global_settings,
        row_settings=config.row_settings,
    )
    tuning = registration_tuning(settings)

    # Step 3: fit and crops.
    report = fit_landmarks(labelling, store, settings, run_id=run_id)
    crops: dict[str, CropRect] = {}
    for role, record in (("reference", ref), ("moving", mov)):
        try:
            crops[role] = snap_crop(
                getattr(labelling.crops, role),
                record.shape_yx,
                multiple=settings.crop_snap_px,
                min_size_px=settings.crop_min_size_px,
            )
        except ValueError as exc:
            raise ValueError(f"{pfx} {role} crop: {exc}") from exc
    crop_r, crop_m = crops["reference"], crops["moving"]
    origin_r = (crop_r.x0, crop_r.y0)
    origin_m = (crop_m.x0, crop_m.y0)
    matrix_coarse_full = coarse_matrix_xy(
        labelling.coarse,
        moving_shape_yx=mov.shape_yx,
        reference_shape_yx=ref.shape_yx,
        moving_pixel_size_um=pixel_mov,
        reference_pixel_size_um=pixel_ref,
    )
    matrix_landmark_full = report.matrix_full
    matrix_landmark_crop = crop_matrix(
        matrix_landmark_full, reference_origin_xy=origin_r, moving_origin_xy=origin_m
    )
    matrix_coarse_crop = crop_matrix(
        matrix_coarse_full, reference_origin_xy=origin_r, moving_origin_xy=origin_m
    )

    # Step 4: skip when nothing changed.
    digest = _provenance_digest(
        preset=config.preset,
        run_label=config.run_label,
        settings=settings,
        labelling_sha=labelling_sha,
        store=store,
    )
    stored = _stored_result(final)
    if not force and stored is not None and _reusable(stored, digest):
        task_summary = out / f"{run_id}_manual_registration_summary.json"
        task_summary.write_text(
            json.dumps(_plain_json({**stored, "reused": True}), indent=2) + "\n",
            encoding="utf-8",
        )
        log_status(f"{pfx} already registered; skipping")
        return _outputs_from_stored(stored, run_id, out)

    # Step 5: temporary result store holding the cropped reference.
    _remove_stale_siblings(result_dir, run_id)
    tmp = result_dir / f".{run_id}_registered.zarr.partial-{uuid.uuid4().hex}"
    try:
        kept_names = list(ref.channel_names)
        prepared = sd.read_zarr(store_path, selection=("images",))
        ref_element = prepared.images["reference"]
        data = ref_element.data[
            :, crop_r.y0 : crop_r.y1, crop_r.x0 : crop_r.x1
        ].rechunk((1, settings.chunk_size, settings.chunk_size))
        element = Image2DModel.parse(
            data,
            dims=("c", "y", "x"),
            c_coords=kept_names,
            transformations={"global": crop_offset_affine(crop_r, pixel_ref)},
        )
        sdata_tmp = sd.SpatialData(images={"reference": element})
        sdata_tmp.attrs[STORE_ATTR] = _result_store_attr(
            store=store,
            config=config,
            store_path=store_path,
            crop_reference=crop_r,
            crop_moving=crop_m,
            kept_reference_names=kept_names,
        )
        stamp_merxen_schema(sdata_tmp)
        sdata_tmp.write(tmp)
        del sdata_tmp, prepared, element, data

        with spatialdata_write_lock(tmp):
            payload, files = _register_into_store(
                tmp=tmp,
                config=config,
                store=store,
                labelling=labelling,
                raw_labelling=raw_labelling,
                labelling_sha=labelling_sha,
                settings=settings,
                tuning=tuning,
                report=report,
                digest=digest,
                crop_r=crop_r,
                crop_m=crop_m,
                matrix_coarse_full=matrix_coarse_full,
                matrix_coarse_crop=matrix_coarse_crop,
                matrix_landmark_full=matrix_landmark_full,
                matrix_landmark_crop=matrix_landmark_crop,
                result_dir=result_dir,
                final=final,
                out=out,
                run_warnings=run_warnings,
                started=started,
                store_path=store_path,
            )
    except BaseException:
        if tmp.exists():
            logger.error(
                "%s registration failed; the unfinished result store %s is left "
                "for the next run to remove",
                pfx,
                tmp,
            )
        raise

    # Step 10: swap the store into place, then publish the files.
    tmp_lock = spatialdata_write_lock_path(tmp)
    superseded: Path | None = None
    with spatialdata_write_lock(final):
        if final.exists():
            superseded = _superseded_path(final)
            os.rename(final, superseded)
        try:
            os.rename(tmp, final)
        except BaseException:
            if superseded is not None:
                os.rename(superseded, final)
            raise
        # A run that died between the two renames left its superseded store behind.
        _remove_superseded_siblings(final, keep=superseded)
    if superseded is not None:
        shutil.rmtree(superseded, ignore_errors=True)
    tmp_lock.unlink(missing_ok=True)
    for path in files:
        _publish(path, result_dir / path.name)
    log_status(f"{pfx} manual registration complete")
    outputs = {
        key: Path(value)
        for key, value in payload["outputs"].items()
        if key != "core_qc"
    }
    outputs["task_summary_json"] = out / f"{run_id}_manual_registration_summary.json"
    return outputs


def _register_into_store(
    *,
    tmp: Path,
    config: ManualRegistrationRunConfig,
    store: StoreAttrs,
    labelling: LabellingAttrs,
    raw_labelling: dict[str, Any],
    labelling_sha: str,
    settings: ManualRegistrationSettings,
    tuning: RegistrationTuning,
    report: LandmarkFitReport,
    digest: str,
    crop_r: CropRect,
    crop_m: CropRect,
    matrix_coarse_full: np.ndarray,
    matrix_coarse_crop: np.ndarray,
    matrix_landmark_full: np.ndarray,
    matrix_landmark_crop: np.ndarray,
    result_dir: Path,
    final: Path,
    out: Path,
    run_warnings: list[str],
    started: float,
    store_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Steps 6 to 9: register, QC and write the payload into the temporary store.

    The caller holds the temporary store's write lock. Returns the payload and
    the files written to ``out`` that are published after the swap.
    """
    run_id = config.run_id
    ref = store.images["reference"]
    mov = store.images["moving"]
    pixel_ref, pixel_mov = ref.pixel_size_um, mov.pixel_size_um
    origin_r = (crop_r.x0, crop_r.y0)
    origin_m = (crop_m.x0, crop_m.y0)
    assert ref.ome_tiff is not None and mov.ome_tiff is not None
    ref_canonical = list(ref.canonical_channel_names or ref.channel_names)
    mov_canonical = list(mov.canonical_channel_names or mov.channel_names)

    # Step 6: register.
    sdata = sd.read_zarr(tmp, selection=("images",))
    ref_info = read_ome_tiff_info(ref.ome_tiff.path)
    mov_info = read_ome_tiff_info(mov.ome_tiff.path)
    ref_grid = cropped_reference_grid(
        ref_info,
        channel_names=ref_canonical,
        crop=crop_r,
        pixel_size_um=pixel_ref,
        exact_factors=ref.ome_tiff.exact_level_factors,
    )
    mov_img = cropped_moving_image(
        mov_info,
        channel_names=mov_canonical,
        crop=crop_m,
        pixel_size_um=pixel_mov,
        exact_factors=mov.ome_tiff.exact_level_factors,
    )
    qc_dir = result_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    qc_paths = registration_output_paths(qc_dir, run_id, "moving_registered")
    _remove_core_qc_files(qc_paths)
    outcome = register_onto_reference(
        sdata_obj=sdata,
        zarr_path=tmp,
        reference=ref_grid,
        moving=mov_img,
        output_key="moving_registered",
        matrix=matrix_landmark_crop,
        registration_channel=mov.registration_channel,
        settings=settings,
        paths=qc_paths,
        dataset_name=run_id,
        tuning=tuning,
    )
    build_image_pyramids_in_store(
        sdata, tmp, image_keys=["reference"], params=ViewerCacheParams(force=True)
    )

    # Step 7: landmark stage positions.
    matrix_affine_crop = outcome.final_matrix
    matrix_affine_full = full_matrix(
        matrix_affine_crop, reference_origin_xy=origin_r, moving_origin_xy=origin_m
    )
    records = _landmark_records(
        report,
        reference_pixel_size_um=pixel_ref,
        reference_origin_xy=origin_r,
        moving_origin_xy=origin_m,
        landmark_full=matrix_landmark_full,
        final_affine_crop=matrix_affine_crop,
        flow_yx=outcome.flow_yx,
        flow_working_to_reference=outcome.flow_working_to_reference,
    )
    warnings = [*run_warnings, *report.warnings]
    not_converged = [str(r["id"]) for r in records if not r["push_converged"]]
    if not_converged:
        warnings.append(
            "push through the flow did not converge for pair(s): "
            + ", ".join(not_converged)
        )
    summary_residuals = _residual_summary(records)

    # Step 8: QC.
    ref_work = read_working_channel(
        open_level=lambda k: ref_grid.levels[k],
        level_shapes_yx=ref_grid.level_shapes_yx,
        channel_index=ref_canonical.index(ref.registration_channel),
        target_factor=settings.tissue_qc_pixel_size_um / pixel_ref,
        level_factors_xy=ref_grid.level_factors_xy,
    )
    mov_work = read_working_channel(
        open_level=mov_img.open_level,
        level_shapes_yx=mov_img.level_shapes_yx,
        channel_index=mov_canonical.index(mov.registration_channel),
        target_factor=settings.tissue_qc_pixel_size_um / pixel_mov,
        level_factors_xy=mov_img.level_factors_xy,
    )
    working_px = pixel_ref * float(np.mean(np.diag(ref_work.to_level0)[:2]))
    pulls = {
        "coarse": affine_pull(matrix_coarse_crop),
        "landmark": affine_pull(matrix_landmark_crop),
        "affine": affine_pull(matrix_affine_crop),
        "final": outcome.pull_mapping,
    }
    tissue = tissue_alignment_qc(
        ref_work,
        mov_work,
        pulls,
        working_pixel_size_um=working_px,
        sigma_um=settings.tissue_qc_sigma_um,
        highpass_sigma_um=settings.tissue_qc_highpass_sigma_um,
    )
    tissue_png = out / f"{run_id}_tissue_qc.png"
    write_tissue_qc_figure(
        ref_work, mov_work, pulls, tissue_png, title=f"{run_id}: tissue alignment"
    )

    nucleus_qc: dict[str, Any]
    if settings.nucleus_qc:
        try:
            stats = measure_registration_overlay(
                reference=ref_grid,
                moving=mov_img,
                registration_channel=mov.registration_channel,
                matrix=matrix_affine_crop,
                settings=settings.model_copy(
                    update={
                        "registration_pixel_size_um": settings.nucleus_qc_pixel_size_um,
                        "refinement_window_um": settings.nucleus_qc_window_um,
                    }
                ),
                tuning=tuning,
            )
            nucleus_qc = {
                "status": "measured",
                "informative_only": True,
                "pixel_size_um": stats.get("working_pixel_size_um"),
                "window_um": settings.nucleus_qc_window_um,
                "n_windows": stats.get("n_windows"),
                "median_um": stats.get("median_um"),
                "p90_um": stats.get("p90_um"),
                "pearson_log_registration_channel": stats.get(
                    "pearson_log_registration_channel"
                ),
            }
        except (ValueError, MemoryError) as exc:
            nucleus_qc = {
                "status": "failed",
                "informative_only": True,
                "reason": str(exc),
            }
            warnings.append(f"nucleus-scale QC failed (informative only): {exc}")
            logger.warning("%s nucleus QC failed: %s", run_id, exc)
    else:
        nucleus_qc = {"status": "disabled"}

    # Step 9: payload, CSVs, attrs.
    refinement = outcome.refinement
    correction = dict(refinement.correction)
    for key in ("n_windows", "n_inlier_windows"):
        if key in correction:
            correction[key] = int(correction[key])
    nonrigid = outcome.nonrigid
    if refinement.status not in ("applied", "disabled"):
        warnings.append(f"refinement {refinement.status}: {refinement.reason}")
    if nonrigid.status not in ("applied", "disabled"):
        warnings.append(f"non-rigid {nonrigid.status}: {nonrigid.reason}")

    core_qc = _core_qc_listing(qc_paths, outcome)
    durable = {
        "landmark_matrix_csv": result_dir / f"{run_id}_landmark_matrix.csv",
        "final_matrix_csv": result_dir / f"{run_id}_final_matrix.csv",
        "landmarks_csv": result_dir / f"{run_id}_landmarks.csv",
        "summary_json": result_dir / f"{run_id}_manual_registration_summary.json",
        "tissue_qc_png": result_dir / tissue_png.name,
    }
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "run_id": run_id,
        "dataset_name": config.dataset_name,
        "preset": config.preset,
        "preset_version": PRESET_VERSIONS[config.preset],
        "run_label": config.run_label,
        "algorithm": dict(ALGORITHM_BLOCK),
        "settings": settings.model_dump(mode="json"),
        "pixel_size_um": {"reference": pixel_ref, "moving": pixel_mov},
        "crops": {
            "reference": {
                "requested": (
                    None
                    if labelling.crops.reference is None
                    else labelling.crops.reference.model_dump()
                ),
                "effective": crop_r.model_dump(),
            },
            "moving": {
                "requested": (
                    None
                    if labelling.crops.moving is None
                    else labelling.crops.moving.model_dump()
                ),
                "effective": crop_m.model_dump(),
            },
        },
        "matrices": {
            "coarse_full": _matrix_list(matrix_coarse_full),
            "landmark_full": _matrix_list(matrix_landmark_full),
            "landmark_crop": _matrix_list(matrix_landmark_crop),
            "final_affine_crop": _matrix_list(matrix_affine_crop),
            "final_affine_full": _matrix_list(matrix_affine_full),
        },
        "fit": fit_summary(report, store=store, settings=settings, labelling=labelling),
        "refinement": {
            "status": refinement.status,
            "reason": refinement.reason,
            **correction,
        },
        "refinement_qc": outcome.overlay_summary(),
        "nonrigid": {
            "enabled": settings.nonrigid_enabled,
            "algorithm": NONRIGID_REGISTRATION_ALGORITHM,
            "status": nonrigid.status,
            "reason": nonrigid.reason,
            **(nonrigid.result.stats if nonrigid.result is not None else {}),
        },
        "tissue_qc": {
            "pixel_size_um": working_px,
            "sigma_um": settings.tissue_qc_sigma_um,
            "highpass_sigma_um": settings.tissue_qc_highpass_sigma_um,
            "pearson": tissue.pearson,
            "mask_fraction": tissue.mask_fraction,
        },
        "nucleus_qc": nucleus_qc,
        "landmarks": records,
        "residual_summary_um": summary_residuals,
        "warnings": warnings,
        "labelling": {
            "revision": labelling.revision,
            "content_sha256": labelling_sha,
            "snapshot": raw_labelling,
        },
        "prepared_store": {
            "path": str(store_path),
            "content_digest": store.content_digest,
            "grid_digest": store.grid_digest,
        },
        "provenance_digest": digest,
        "resources": {},
        "outputs": {
            **{key: str(path) for key, path in durable.items()},
            "result_store": str(final),
            "core_qc": core_qc,
        },
    }

    files: list[Path] = []
    matrix_landmark_csv = write_alignment_matrix(
        out / durable["landmark_matrix_csv"].name, matrix_landmark_full
    )
    matrix_final_csv = write_alignment_matrix(
        out / durable["final_matrix_csv"].name, matrix_affine_full
    )
    landmarks_csv = out / durable["landmarks_csv"].name
    _landmarks_dataframe(records).to_csv(
        landmarks_csv, index=False, float_format="%.6g"
    )
    files += [matrix_landmark_csv, matrix_final_csv, landmarks_csv, tissue_png]
    if tissue_png.with_suffix(".pdf").exists():
        files.append(tissue_png.with_suffix(".pdf"))

    payload["resources"] = {
        "wall_time_s": time.perf_counter() - started,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    payload = _plain_json(payload)
    summary_path = out / durable["summary_json"].name
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files.append(summary_path)

    sdata.attrs[RESULT_ATTR] = payload
    write_spatialdata_metadata(sdata, write_attrs=True)
    del sdata
    return payload, files
