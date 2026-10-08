"""Register externally acquired images (e.g. post-Xenium IF) onto the mask grid.

Each image is placed with the 3x3 affine exported by Xenium Explorer, which maps
image level-0 pixels to reference (``morphology_focus``) level-0 pixels. The
matrix is optionally refined by a robust affine fit to local DAPI-to-DAPI shifts,
then the image is resampled tile by tile onto the reference level-0 grid. That
grid is the Cellpose mask grid, so mask image quantification measures the new
channels over the same masks as the instrument channels.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata as sd
import xarray as xr
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation
from spatialdata.transformations import get_transformation, set_transformation

from merxen.alignment.image_warp import materialize_warped_image
from merxen.config import ImageRegistrationConfig, RegisteredImageSpec
from merxen.io.image_source import REGISTERED_IMAGES_ATTR, image_to_cyx
from merxen.io.ome_tiff import OmeTiffInfo, open_ome_tiff_level, read_ome_tiff_info
from merxen.io.spatialdata_io import (
    delete_element,
    spatialdata_write_lock,
    write_spatialdata_metadata,
)
from merxen.memory import force_release, log_status
from merxen.plotting import prepare_plot_output, save_figure
from merxen.viewer_cache.build import ViewerCacheParams, build_image_pyramid
from merxen.viewer_cache.format import derived_image_pyramid_cache_key

logger = logging.getLogger(__name__)

REGISTRATION_ALGORITHM = "explorer_affine_dapi_refine_v1"
# OpenCV remapping indexes pixels with signed 16-bit integers.
_MAX_CV2_DIMENSION = 32767
_HIGHPASS_SIGMA_PX = 8.0
_TISSUE_SIGMA_PX = 4.0
_MIN_WINDOW_PX = 32
_MIN_WINDOW_VALID_FRACTION = 0.6
_MIN_WINDOW_QUALITY = 0.3
_PHASE_UPSAMPLE_FACTOR = 20
_MAX_QC_DISPLAY_PX = 2000
_FACTOR_TOLERANCE = 0.01


@dataclass(frozen=True)
class MatrixScaleCheck:
    """Agreement between the matrix scale and the two images' pixel sizes."""

    status: str
    observed_scale: float
    expected_scale: float | None = None
    relative_error: float | None = None


@dataclass(frozen=True)
class LocalShifts:
    """Per-window residual shifts between a reference and an overlaid image.

    A shift is the displacement that moves overlaid content onto the reference:
    content drawn at ``center`` belongs at ``center + shift``. Both are ``(x, y)``
    in working-grid pixels.
    """

    centers_xy: np.ndarray
    shifts_xy: np.ndarray
    quality: np.ndarray
    n_candidate_windows: int


@dataclass(frozen=True)
class _WorkingImage:
    """One channel resampled near the registration pixel size."""

    pixels: np.ndarray
    to_level0: np.ndarray


@dataclass(frozen=True)
class _Overlay:
    shifts: LocalShifts
    pearson_log: float
    warped: np.ndarray
    valid: np.ndarray


@dataclass
class _Refinement:
    matrix: np.ndarray
    status: str
    reason: str | None = None
    correction: dict[str, float] = field(default_factory=dict)
    before: _Overlay | None = None
    after: _Overlay | None = None
    working_pixel_size_um: float | None = None
    working_to_reference: np.ndarray | None = None
    reference_pixels: np.ndarray | None = None


@dataclass(frozen=True)
class _ReferenceGrid:
    key: str
    levels: list[xr.DataArray]
    transformations: dict[str, Any]
    coordinate_system: str

    @property
    def shape_yx(self) -> tuple[int, int]:
        return (int(self.levels[0].sizes["y"]), int(self.levels[0].sizes["x"]))


def load_alignment_matrix(path: Path | str) -> np.ndarray:
    """Read and validate a 3x3 affine alignment matrix CSV.

    Args:
        path: Comma- or whitespace-separated file with three numeric rows, as
            exported by Xenium Explorer.

    Returns:
        The matrix as a ``(3, 3)`` float array.

    Raises:
        ValueError: If the file is not a finite, invertible 2D affine matrix.
    """
    path = Path(path)
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    try:
        rows = [[float(value) for value in re.split(r"[,\s]+", line)] for line in lines]
    except ValueError as exc:
        raise ValueError(
            f"Alignment matrix {path} contains non-numeric values"
        ) from exc
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"Alignment matrix {path} must be 3x3, got rows {rows}")
    matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Alignment matrix {path} contains non-finite values")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(
            f"Alignment matrix {path} is not affine: last row is {matrix[2].tolist()}, "
            "expected [0, 0, 1]"
        )
    if abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
        raise ValueError(f"Alignment matrix {path} is singular")
    return matrix


def check_matrix_scale(
    matrix: np.ndarray,
    *,
    image_pixel_size_um: float | None,
    reference_pixel_size_um: float,
    tolerance: float,
) -> MatrixScaleCheck:
    """Check that the matrix maps image pixels to reference pixels.

    An image-pixel to reference-pixel affine must scale lengths by
    ``image_pixel_size_um / reference_pixel_size_um``. Inverted matrices or
    matrices expressed in microns fail this check by a large factor.

    Args:
        matrix: ``(3, 3)`` affine from image level-0 to reference level-0 pixels.
        image_pixel_size_um: Image level-0 pixel size, or None when unknown.
        reference_pixel_size_um: Reference level-0 pixel size.
        tolerance: Accepted relative deviation of the observed scale.

    Returns:
        The check outcome; ``status`` is ``"ok"`` or ``"skipped_no_pixel_size"``.

    Raises:
        ValueError: If the scale disagrees with the pixel sizes.
    """
    observed = float(math.sqrt(abs(np.linalg.det(np.asarray(matrix)[:2, :2]))))
    if image_pixel_size_um is None:
        return MatrixScaleCheck(status="skipped_no_pixel_size", observed_scale=observed)

    expected = float(image_pixel_size_um) / float(reference_pixel_size_um)
    relative_error = observed / expected - 1.0
    if abs(relative_error) <= tolerance:
        return MatrixScaleCheck(
            status="ok",
            observed_scale=observed,
            expected_scale=expected,
            relative_error=float(relative_error),
        )

    conventions = {
        "reference-pixel to image-pixel (inverted)": 1.0 / expected,
        "image-pixel to micron": float(image_pixel_size_um),
        "micron to reference-pixel": 1.0 / float(reference_pixel_size_um),
    }
    closest = min(
        conventions,
        key=lambda name: abs(math.log(observed / conventions[name])),
    )
    hint = ""
    if abs(observed / conventions[closest] - 1.0) <= tolerance:
        hint = f" Its scale matches a {closest} matrix instead."
    raise ValueError(
        f"Alignment matrix scale {observed:.5f} does not match the expected "
        f"image-to-reference pixel scale {expected:.5f} (image "
        f"{image_pixel_size_um:.6g} um/px, reference {reference_pixel_size_um:.6g} "
        f"um/px, tolerance {tolerance:.0%}).{hint} Export the matrix from Xenium "
        "Explorer after aligning this exact image to this dataset."
    )


def measure_local_shifts(
    reference: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
    *,
    window_px: int,
    min_valid_fraction: float = _MIN_WINDOW_VALID_FRACTION,
    min_quality: float = _MIN_WINDOW_QUALITY,
) -> LocalShifts:
    """Measure residual shifts between two co-registered images in windows.

    Windows overlap by half their size. A window is used when enough of it is
    valid (tissue covered by both images), and its shift is kept when the
    shift-corrected normalized cross-correlation reaches ``min_quality``.

    Args:
        reference: 2D reference image, ideally high-pass filtered.
        moving: 2D image on the same grid as ``reference``.
        valid: Boolean mask of pixels where both images carry tissue.
        window_px: Square window size in pixels.
        min_valid_fraction: Minimum valid fraction for a window to be measured.
        min_quality: Minimum shift-corrected correlation for a window to count.

    Returns:
        Window centres, shifts, and correlation scores of accepted windows.

    Raises:
        ValueError: If the three arrays differ in shape.
    """
    if reference.shape != moving.shape or valid.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape}, moving {moving.shape}, "
            f"valid {valid.shape}"
        )
    window = int(window_px)
    step = max(1, window // 2)
    margin = max(1, window // 32)
    height, width = reference.shape
    centers: list[tuple[float, float]] = []
    shifts: list[tuple[float, float]] = []
    quality: list[float] = []
    n_candidates = 0
    for y0 in range(0, height - window + 1, step):
        for x0 in range(0, width - window + 1, step):
            if valid[y0 : y0 + window, x0 : x0 + window].mean() < min_valid_fraction:
                continue
            n_candidates += 1
            ref_window = reference[y0 : y0 + window, x0 : x0 + window]
            mov_window = moving[y0 : y0 + window, x0 : x0 + window]
            if ref_window.std() == 0 or mov_window.std() == 0:
                continue
            shift_yx, _, _ = phase_cross_correlation(
                ref_window,
                mov_window,
                upsample_factor=_PHASE_UPSAMPLE_FACTOR,
                normalization=None,
            )
            rolled = np.roll(
                mov_window,
                (int(round(shift_yx[0])), int(round(shift_yx[1]))),
                axis=(0, 1),
            )
            score = _normalized_correlation(
                ref_window[margin:-margin, margin:-margin],
                rolled[margin:-margin, margin:-margin],
            )
            if not np.isfinite(score) or score < min_quality:
                continue
            centers.append((x0 + (window - 1) / 2.0, y0 + (window - 1) / 2.0))
            shifts.append((float(shift_yx[1]), float(shift_yx[0])))
            quality.append(score)
    return LocalShifts(
        centers_xy=np.asarray(centers, dtype=np.float64).reshape(-1, 2),
        shifts_xy=np.asarray(shifts, dtype=np.float64).reshape(-1, 2),
        quality=np.asarray(quality, dtype=np.float64),
        n_candidate_windows=n_candidates,
    )


def fit_affine_displacement(
    centers_xy: np.ndarray,
    shifts_xy: np.ndarray,
    *,
    max_iterations: int = 5,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly fit an affine correction to a field of window shifts.

    Least squares is repeated after dropping windows whose residual exceeds the
    median by three robust standard deviations, so local deformation and bad
    windows do not drag the global fit.

    Args:
        centers_xy: ``(n, 2)`` window centres.
        shifts_xy: ``(n, 2)`` shifts measured at those centres.
        max_iterations: Maximum number of trimming rounds.
        min_points: Minimum number of windows to keep.

    Returns:
        A ``(3, 3)`` correction mapping a point ``q`` to ``q + shift(q)``, and the
        boolean inlier mask used for the final fit.

    Raises:
        ValueError: If fewer than ``min_points`` windows are given.
    """
    centers = np.asarray(centers_xy, dtype=np.float64)
    shifts = np.asarray(shifts_xy, dtype=np.float64)
    if len(centers) < max(int(min_points), 3):
        raise ValueError(f"Need at least {min_points} windows, got {len(centers)}")
    design = np.column_stack([centers, np.ones(len(centers))])
    inliers = np.ones(len(centers), dtype=bool)
    for _ in range(int(max_iterations)):
        coef, *_ = np.linalg.lstsq(design[inliers], shifts[inliers], rcond=None)
        residual = np.linalg.norm(shifts - design @ coef, axis=1)
        median = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - median)))
        threshold = max(median + 3.0 * 1.4826 * mad, 0.5)
        updated = residual <= threshold
        if updated.sum() < min_points or np.array_equal(updated, inliers):
            break
        inliers = updated
    coef, *_ = np.linalg.lstsq(design[inliers], shifts[inliers], rcond=None)
    correction = np.eye(3)
    correction[:2, :2] += coef[:2].T
    correction[:2, 2] = coef[2]
    return correction, inliers


def run_image_registration(
    config: ImageRegistrationConfig,
    *,
    force_rerun: bool = False,
) -> dict[str, Path]:
    """Register every configured image into the SpatialData zarr and write QC.

    Registered images are tracked in ``sdata.attrs[REGISTERED_IMAGES_ATTR]``.
    An image is skipped when its source file, matrix, and settings are unchanged;
    previously registered images that are no longer configured are removed so
    downstream quantification never measures stale channels.

    Args:
        config: Validated registration configuration.
        force_rerun: Re-register images even when an identical result exists.

    Returns:
        Paths of the updated zarr and of the per-image QC summaries.
    """
    latest_path = Path(config.latest_zarr_path)
    if not latest_path.exists():
        raise FileNotFoundError(f"[{config.dataset_name}] Missing zarr: {latest_path}")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {"latest_zarr": latest_path}

    # Lock the resolved store: Nextflow stages see it through per-task symlinks.
    with spatialdata_write_lock(latest_path.resolve()):
        sdata_obj = sd.read_zarr(latest_path, selection=("images",))
        try:
            reference = _reference_grid(sdata_obj, config)
            registry = _registry(sdata_obj)
            removed = _remove_stale_images(
                sdata_obj,
                registry,
                keep={spec.image_key for spec in config.images},
            )
            summaries = []
            for spec in config.images:
                summary = _register_image(
                    sdata_obj=sdata_obj,
                    zarr_path=latest_path,
                    spec=spec,
                    reference=reference,
                    registry=registry,
                    config=config,
                    force_rerun=force_rerun,
                )
                summaries.append(summary)
                outputs[f"{spec.image_key}_summary"] = Path(summary["summary_path"])
        finally:
            del sdata_obj
            force_release(note=f"after {config.dataset_name} image registration")

    combined_path = (
        output_dir / f"{_prefix(config.dataset_name)}_registered_images.json"
    )
    combined_path.write_text(
        json.dumps(
            {
                "dataset_name": config.dataset_name,
                "removed_image_keys": removed,
                "images": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    outputs["summary"] = combined_path
    return outputs


def _register_image(
    *,
    sdata_obj: Any,
    zarr_path: Path,
    spec: RegisteredImageSpec,
    reference: _ReferenceGrid,
    registry: dict[str, Any],
    config: ImageRegistrationConfig,
    force_rerun: bool,
) -> dict[str, Any]:
    key = spec.image_key
    paths = _output_paths(Path(config.output_dir), config.dataset_name, key)
    info = read_ome_tiff_info(spec.image_path)
    channel_names = (
        list(spec.channel_names)
        if spec.channel_names is not None
        else list(info.channel_names)
    )
    if len(channel_names) != info.shape_cyx[0]:
        raise ValueError(
            f"[{config.dataset_name}] {key}: {len(channel_names)} channel name(s) "
            f"given for an image with {info.shape_cyx[0]} channel(s)"
        )
    matrix = load_alignment_matrix(spec.alignment_matrix_path)
    image_pixel_size_um = (
        float(np.mean(info.pixel_size_um)) if info.pixel_size_um is not None else None
    )
    scale_check = check_matrix_scale(
        matrix,
        image_pixel_size_um=image_pixel_size_um,
        reference_pixel_size_um=config.reference_pixel_size_um,
        tolerance=config.matrix_scale_tolerance,
    )
    if image_pixel_size_um is None:
        logger.warning(
            "[%s] %s has no OME pixel size; skipping the matrix scale check and "
            "assuming the matrix maps image pixels to reference pixels.",
            config.dataset_name,
            key,
        )
        image_pixel_size_um = (
            scale_check.observed_scale * config.reference_pixel_size_um
        )

    if key in sdata_obj.images and key not in registry:
        raise ValueError(
            f"[{config.dataset_name}] image_key {key!r} already names an image this "
            "stage did not register; choose another key so it is not overwritten"
        )
    provenance = _provenance(spec, matrix, channel_names, config)
    existing = registry.get(key, {})
    if (
        not force_rerun
        and existing.get("status") == "complete"
        and _json_equal(existing.get("provenance"), provenance)
        and key in sdata_obj.images
        and isinstance(existing.get("summary"), dict)
    ):
        log_status(f"[{config.dataset_name}] {key} already registered; skipping.")
        return _reemit_summary(existing["summary"], paths)

    registry[key] = {"status": "in_progress", "provenance": provenance}
    _write_registry(sdata_obj, registry)

    log_status(
        f"[{config.dataset_name}] Registering {key}: {info.path.name} "
        f"{info.shape_cyx} channels={channel_names} pyramidal={info.is_pyramidal}"
    )
    refinement = _refine_registration(
        reference=reference,
        info=info,
        spec=spec,
        channel_names=channel_names,
        matrix=matrix,
        image_pixel_size_um=image_pixel_size_um,
        config=config,
    )
    _write_qc_outputs(refinement, paths, config.dataset_name, key)

    source = open_ome_tiff_level(info, 0, channel_names=channel_names)
    warp = materialize_warped_image(
        sdata_obj=sdata_obj,
        zarr_path=zarr_path,
        source_image=source,
        output_key=key,
        output_shape_rc=reference.shape_yx,
        fixed_to_source_image_xy=_affine_function(np.linalg.inv(refinement.matrix)),
        transform=reference.transformations[reference.coordinate_system],
        coordinate_system=reference.coordinate_system,
        tile_size=config.tile_size,
        chunk_size=config.chunk_size,
        interpolation="bilinear",
        fill_value=0.0,
    )
    if len(reference.transformations) > 1:
        set_transformation(
            sdata_obj.images[key],
            reference.transformations,
            set_all=True,
            write_to_sdata=sdata_obj,
        )
    pyramid_status = "disabled"
    if config.build_viewer_pyramid:
        base_cyx = xr.DataArray(
            da.from_zarr(str(zarr_path / "images" / key / "s0")),
            dims=("c", "y", "x"),
            coords={"c": list(warp.channels)},
        )
        pyramid_status = build_image_pyramid(
            sdata=sdata_obj,
            zarr_path=zarr_path,
            image_key=key,
            base_cyx=base_cyx,
            channels=list(warp.channels),
            transform=reference.transformations[reference.coordinate_system],
            params=ViewerCacheParams(force=True),
            coordinate_system=reference.coordinate_system,
        )

    summary = {
        "dataset_name": config.dataset_name,
        "image_key": key,
        "image_path": str(info.path),
        "image_shape_cyx": list(info.shape_cyx),
        "image_is_pyramidal": info.is_pyramidal,
        "image_pixel_size_um": image_pixel_size_um,
        "source_channel_names": list(info.channel_names),
        "channel_names": channel_names,
        "registration_channel": spec.registration_channel,
        "reference_image_key": reference.key,
        "reference_shape_yx": list(reference.shape_yx),
        "reference_pixel_size_um": config.reference_pixel_size_um,
        "explorer_matrix": matrix.tolist(),
        "final_matrix": refinement.matrix.tolist(),
        "matrix_scale_check": _jsonable(scale_check.__dict__),
        "refinement": {
            "status": refinement.status,
            "reason": refinement.reason,
            **refinement.correction,
        },
        "residuals_before": _overlay_stats(refinement.before, refinement),
        "residuals_after": _overlay_stats(refinement.after, refinement),
        "output": {
            "shape_cyx": list(warp.shape_cyx),
            "dtype": warp.dtype,
            "nonzero_fraction": warp.nonzero_fraction,
            "viewer_pyramid": pyramid_status,
        },
        "summary_path": str(paths["summary"]),
        # QC files exist only when a registration channel was compared.
        "outputs": {
            name: str(path)
            for name, path in paths.items()
            if name == "summary" or path.exists()
        },
    }
    paths["summary"].write_text(json.dumps(_jsonable(summary), indent=2) + "\n")

    registry[key] = {
        "status": "complete",
        "provenance": provenance,
        "final_matrix": refinement.matrix.tolist(),
        "refinement_status": refinement.status,
        "channel_names": channel_names,
        # Kept in the zarr so a rerun in a fresh output directory (a new
        # Nextflow work dir) can skip the warp and still report its result.
        "summary": _jsonable(summary),
    }
    _write_registry(sdata_obj, registry)
    log_status(
        f"[{config.dataset_name}] Registered {key}: refinement={refinement.status}, "
        f"residual median {summary['residuals_after'].get('median_um')} um"
    )
    return summary


def _refine_registration(
    *,
    reference: _ReferenceGrid,
    info: OmeTiffInfo,
    spec: RegisteredImageSpec,
    channel_names: list[str],
    matrix: np.ndarray,
    image_pixel_size_um: float,
    config: ImageRegistrationConfig,
) -> _Refinement:
    if spec.registration_channel is None:
        return _Refinement(matrix=matrix, status="skipped_no_registration_channel")

    reference_channel = _channel_index(
        _channel_names(reference.levels[0]),
        config.reference_channel,
        label=f"reference image {reference.key!r}",
    )
    moving_channel = _channel_index(
        channel_names,
        spec.registration_channel,
        label=f"image {spec.image_key!r}",
    )
    reference_work = _read_working_channel(
        open_level=lambda level: reference.levels[level],
        level_shapes_yx=[
            (int(level.sizes["y"]), int(level.sizes["x"])) for level in reference.levels
        ],
        channel_index=reference_channel,
        target_factor=config.registration_pixel_size_um
        / config.reference_pixel_size_um,
    )
    moving_work = _read_working_channel(
        open_level=lambda level: open_ome_tiff_level(info, level),
        level_shapes_yx=[(shape[1], shape[2]) for shape in info.level_shapes_cyx],
        channel_index=moving_channel,
        target_factor=config.registration_pixel_size_um / image_pixel_size_um,
    )
    working_px_um = config.reference_pixel_size_um * float(
        np.mean(np.diag(reference_work.to_level0)[:2])
    )
    window_px = max(
        _MIN_WINDOW_PX, int(round(config.refinement_window_um / working_px_um))
    )
    reference_hp = _highpass_log(reference_work.pixels)
    tissue = _tissue_mask(reference_work.pixels)

    def overlay(candidate: np.ndarray) -> _Overlay:
        return _measure_overlay(
            reference_work,
            reference_hp,
            tissue,
            moving_work,
            candidate,
            window_px,
        )

    before = overlay(matrix)
    result = _Refinement(
        matrix=matrix,
        status="disabled",
        before=before,
        after=before,
        working_pixel_size_um=working_px_um,
        working_to_reference=reference_work.to_level0,
        reference_pixels=reference_work.pixels,
    )
    n_windows = len(before.shifts.centers_xy)
    result.correction = {"n_windows": float(n_windows)}
    if not config.refine_affine:
        return result
    if n_windows < config.min_refinement_windows:
        result.status = "skipped_insufficient_windows"
        result.reason = (
            f"{n_windows} usable windows < min_refinement_windows "
            f"({config.min_refinement_windows})"
        )
        logger.warning("%s: %s", spec.image_key, result.reason)
        return result

    correction, inliers = fit_affine_displacement(
        before.shifts.centers_xy,
        before.shifts.shifts_xy,
    )
    diagnostics = _correction_diagnostics(
        correction,
        before.shifts.centers_xy[inliers],
        working_px_um,
    )
    result.correction = {
        **diagnostics,
        "n_windows": float(n_windows),
        "n_inlier_windows": float(inliers.sum()),
    }
    violations = []
    if diagnostics["max_shift_um"] > config.max_refinement_shift_um:
        violations.append(
            f"shift {diagnostics['max_shift_um']:.2f} um > "
            f"{config.max_refinement_shift_um} um"
        )
    if abs(diagnostics["rotation_deg"]) > config.max_refinement_rotation_deg:
        violations.append(
            f"rotation {diagnostics['rotation_deg']:.3f} deg > "
            f"{config.max_refinement_rotation_deg} deg"
        )
    if abs(diagnostics["scale_change"]) > config.max_refinement_scale_change:
        violations.append(
            f"scale change {diagnostics['scale_change']:.4f} > "
            f"{config.max_refinement_scale_change}"
        )
    if violations:
        result.status = "rejected_exceeds_limits"
        result.reason = "; ".join(violations)
        logger.warning(
            "%s: refinement rejected, keeping the Explorer matrix (%s)",
            spec.image_key,
            result.reason,
        )
        return result

    to_level0 = reference_work.to_level0
    refined = to_level0 @ correction @ np.linalg.inv(to_level0) @ matrix
    result.matrix = refined
    result.status = "applied"
    result.after = overlay(refined)
    return result


def _measure_overlay(
    reference: _WorkingImage,
    reference_hp: np.ndarray,
    tissue: np.ndarray,
    moving: _WorkingImage,
    matrix: np.ndarray,
    window_px: int,
) -> _Overlay:
    height, width = reference.pixels.shape
    moving_to_reference = np.linalg.inv(reference.to_level0) @ matrix @ moving.to_level0
    warped = cv2.warpAffine(
        moving.pixels,
        moving_to_reference[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    footprint = cv2.warpAffine(
        np.ones(moving.pixels.shape, dtype=np.uint8),
        moving_to_reference[:2],
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    valid = tissue & footprint
    shifts = measure_local_shifts(
        reference_hp,
        _highpass_log(warped),
        valid,
        window_px=window_px,
    )
    return _Overlay(
        shifts=shifts,
        pearson_log=_masked_log_pearson(reference.pixels, warped, valid),
        warped=warped,
        valid=valid,
    )


def _read_working_channel(
    *,
    open_level: Callable[[int], Any],
    level_shapes_yx: list[tuple[int, int]],
    channel_index: int,
    target_factor: float,
) -> _WorkingImage:
    """Read one channel near ``target_factor`` x the level-0 pixel size.

    Uses the coarsest stored level that is not coarser than the target, then
    block-averages by an integer factor, so non-pyramidal files never need a
    full-resolution in-memory copy.
    """
    height0, width0 = level_shapes_yx[0]
    level = 0
    # Pyramid levels of odd-sized images are slightly more than 2x apart
    # (51313 / 12828 = 4.00008), so factors are compared with a 1% tolerance.
    for index, (_height, width) in enumerate(level_shapes_yx):
        if width0 / width <= target_factor * (1.0 + _FACTOR_TOLERANCE):
            level = index
    height, width = level_shapes_yx[level]
    level_factor_x, level_factor_y = width0 / width, height0 / height
    block = max(1, int(math.floor(target_factor / level_factor_x + _FACTOR_TOLERANCE)))
    while max(height, width) / block >= _MAX_CV2_DIMENSION:
        block += 1

    channel = image_to_cyx(open_level(level)).isel(c=channel_index).data
    array = da.asarray(channel)
    if block > 1:
        array = da.coarsen(np.mean, array, {0: block, 1: block}, trim_excess=True)
    pixels = np.asarray(array.compute(), dtype=np.float32)
    to_level0 = _pixel_centre_scale(
        level_factor_x, level_factor_y
    ) @ _pixel_centre_scale(block, block)
    return _WorkingImage(pixels=pixels, to_level0=to_level0)


def _pixel_centre_scale(factor_x: float, factor_y: float) -> np.ndarray:
    """Map coarse pixel indices to fine indices, keeping pixel centres aligned."""
    return np.array(
        [
            [factor_x, 0.0, (factor_x - 1.0) / 2.0],
            [0.0, factor_y, (factor_y - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _correction_diagnostics(
    correction: np.ndarray,
    centers_xy: np.ndarray,
    working_px_um: float,
) -> dict[str, float]:
    linear = correction[:2, :2]
    rotation = math.degrees(
        math.atan2(linear[1, 0] - linear[0, 1], linear[0, 0] + linear[1, 1])
    )
    displacement = centers_xy @ (linear - np.eye(2)).T + correction[:2, 2]
    max_shift_px = float(np.linalg.norm(displacement, axis=1).max())
    return {
        "rotation_deg": float(rotation),
        "scale_change": float(math.sqrt(abs(np.linalg.det(linear))) - 1.0),
        "max_shift_um": max_shift_px * float(working_px_um),
    }


def _highpass_log(image: np.ndarray) -> np.ndarray:
    logged = np.log1p(np.maximum(image, 0.0)).astype(np.float32, copy=False)
    return np.asarray(logged - gaussian_filter(logged, _HIGHPASS_SIGMA_PX))


def _tissue_mask(image: np.ndarray) -> np.ndarray:
    smoothed = gaussian_filter(np.log1p(np.maximum(image, 0.0)), _TISSUE_SIGMA_PX)
    sample = smoothed[::4, ::4]
    if float(sample.max()) <= float(sample.min()):
        return np.zeros(image.shape, dtype=bool)
    return np.asarray(smoothed > threshold_otsu(sample))


def _masked_log_pearson(
    reference: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
) -> float:
    mask = valid[::2, ::2]
    if mask.sum() < 3:
        return float("nan")
    ref_values = np.log1p(np.maximum(reference[::2, ::2][mask], 0.0))
    mov_values = np.log1p(np.maximum(moving[::2, ::2][mask], 0.0))
    return _normalized_correlation(ref_values, mov_values)


def _normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def _affine_function(matrix: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    linear = np.asarray(matrix[:2, :2], dtype=np.float64)
    offset = np.asarray(matrix[:2, 2], dtype=np.float64)

    def apply(xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=np.float64) @ linear.T + offset

    return apply


def _channel_names(image_cyx: xr.DataArray) -> list[str]:
    if "c" in image_cyx.coords:
        return [str(name) for name in image_cyx.coords["c"].values]
    return [f"c{index}" for index in range(int(image_cyx.sizes["c"]))]


def _channel_index(channels: list[str], wanted: str, *, label: str) -> int:
    lowered = [str(name).lower() for name in channels]
    if str(wanted).lower() not in lowered:
        raise ValueError(
            f"Channel {wanted!r} not found in {label}; channels: {channels}"
        )
    return lowered.index(str(wanted).lower())


def _reference_grid(sdata_obj: Any, config: ImageRegistrationConfig) -> _ReferenceGrid:
    key = config.reference_image_key
    if key not in sdata_obj.images:
        raise KeyError(
            f"[{config.dataset_name}] Reference image {key!r} not found; images: "
            f"{sorted(map(str, sdata_obj.images))}"
        )
    image_obj = sdata_obj.images[key]
    if hasattr(image_obj, "children"):
        scale_names = sorted(
            image_obj.children,
            key=lambda name: int(re.sub(r"\D", "", str(name)) or 0),
        )
        levels = [image_to_cyx(image_obj[name].ds["image"]) for name in scale_names]
    else:
        levels = [image_to_cyx(image_obj)]
    transformations = get_transformation(image_obj, get_all=True)
    coordinate_system = (
        "global" if "global" in transformations else sorted(transformations)[0]
    )
    return _ReferenceGrid(
        key=key,
        levels=levels,
        transformations=dict(transformations),
        coordinate_system=coordinate_system,
    )


def _registry(sdata_obj: Any) -> dict[str, Any]:
    registry = sdata_obj.attrs.get(REGISTERED_IMAGES_ATTR, {})
    return dict(registry) if isinstance(registry, dict) else {}


def _write_registry(sdata_obj: Any, registry: dict[str, Any]) -> None:
    sdata_obj.attrs[REGISTERED_IMAGES_ATTR] = _jsonable(registry)
    write_spatialdata_metadata(sdata_obj, write_attrs=True)


def _remove_stale_images(
    sdata_obj: Any,
    registry: dict[str, Any],
    *,
    keep: set[str],
) -> list[str]:
    removed = []
    for key in sorted(set(registry) - keep):
        pyramid_prefix = derived_image_pyramid_cache_key(key, 0)[: -len("0")]
        for element_key in list(map(str, sdata_obj.images)):
            if element_key == key or element_key.startswith(pyramid_prefix):
                delete_element(sdata_obj, element_key, "images")
        registry.pop(key)
        removed.append(key)
        log_status(f"Removed previously registered image {key!r}")
    if removed:
        _write_registry(sdata_obj, registry)
    return removed


def _provenance(
    spec: RegisteredImageSpec,
    matrix: np.ndarray,
    channel_names: list[str],
    config: ImageRegistrationConfig,
) -> dict[str, Any]:
    image_path = Path(spec.image_path).resolve()
    stat = image_path.stat()
    settings = config.model_dump(
        mode="json",
        exclude={
            "dataset_name",
            "platform",
            "latest_zarr_path",
            "output_dir",
            "images",
            "tile_size",
            "chunk_size",
            "build_viewer_pyramid",
        },
    )
    provenance: dict[str, Any] = _jsonable(
        {
            "algorithm": REGISTRATION_ALGORITHM,
            "image_path": str(image_path),
            "image_size_bytes": int(stat.st_size),
            "image_mtime_ns": int(stat.st_mtime_ns),
            "explorer_matrix": matrix.tolist(),
            "channel_names": list(channel_names),
            "registration_channel": spec.registration_channel,
            "settings": settings,
        }
    )
    return provenance


def _overlay_stats(overlay: _Overlay | None, refinement: _Refinement) -> dict[str, Any]:
    if overlay is None or refinement.working_pixel_size_um is None:
        return {}
    magnitudes = (
        np.linalg.norm(overlay.shifts.shifts_xy, axis=1)
        * refinement.working_pixel_size_um
    )
    stats: dict[str, Any] = {
        "n_windows": int(len(magnitudes)),
        "n_candidate_windows": int(overlay.shifts.n_candidate_windows),
        "pearson_log_registration_channel": overlay.pearson_log,
    }
    if len(magnitudes):
        stats.update(
            {
                "median_um": float(np.median(magnitudes)),
                "p90_um": float(np.percentile(magnitudes, 90)),
                "p99_um": float(np.percentile(magnitudes, 99)),
                "max_um": float(magnitudes.max()),
            }
        )
    return stats


def _reemit_summary(
    stored: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Write a skipped image's stored summary, and its QC files, to ``paths``."""
    outputs = {"summary": str(paths["summary"])}
    for name, previous in dict(stored.get("outputs", {})).items():
        source, target = Path(previous), paths.get(name)
        if name == "summary" or target is None or not source.exists():
            continue
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
            # save_figure writes a PDF twin next to every PNG.
            if source.with_suffix(".pdf").exists():
                shutil.copy2(source.with_suffix(".pdf"), target.with_suffix(".pdf"))
        outputs[name] = str(target)
    summary = {
        **stored,
        "reused": True,
        "summary_path": str(paths["summary"]),
        "outputs": outputs,
    }
    paths["summary"].write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    return summary


def _write_qc_outputs(
    refinement: _Refinement,
    paths: dict[str, Path],
    dataset_name: str,
    image_key: str,
) -> None:
    if refinement.before is None or refinement.working_to_reference is None:
        return
    pixel_um = float(refinement.working_pixel_size_um or 1.0)
    frames = []
    for stage, overlay in (("before", refinement.before), ("after", refinement.after)):
        if overlay is None:
            continue
        shifts = overlay.shifts
        frames.append(
            pd.DataFrame(
                {
                    "stage": stage,
                    "center_x_working_px": shifts.centers_xy[:, 0],
                    "center_y_working_px": shifts.centers_xy[:, 1],
                    "shift_x_um": shifts.shifts_xy[:, 0] * pixel_um,
                    "shift_y_um": shifts.shifts_xy[:, 1] * pixel_um,
                    "shift_um": np.linalg.norm(shifts.shifts_xy, axis=1) * pixel_um,
                    "quality": shifts.quality,
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(paths["local_shifts"], index=False)
    _plot_registration_qc(refinement, paths["qc_plot"], f"{dataset_name} {image_key}")


def _plot_registration_qc(refinement: _Refinement, path: Path, title: str) -> None:
    after = refinement.after or refinement.before
    before = refinement.before
    assert after is not None and before is not None
    pixel_um = float(refinement.working_pixel_size_um or 1.0)
    prepare_plot_output(path)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    assert refinement.reference_pixels is not None
    reference_display = _normalize_for_display(refinement.reference_pixels, after.valid)
    stride = max(1, math.ceil(max(after.warped.shape) / _MAX_QC_DISPLAY_PX))
    moving_display = _normalize_for_display(after.warped, after.valid)
    rgb = np.dstack([moving_display, reference_display, moving_display])[
        ::stride, ::stride
    ]
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Reference (green) / registered image (magenta)")
    axes[0].set_axis_off()

    magnitudes = np.linalg.norm(after.shifts.shifts_xy, axis=1) * pixel_um
    axes[1].imshow(reference_display[::stride, ::stride], cmap="gray", alpha=0.5)
    if len(magnitudes):
        quiver = axes[1].quiver(
            after.shifts.centers_xy[:, 0] / stride,
            after.shifts.centers_xy[:, 1] / stride,
            after.shifts.shifts_xy[:, 0],
            -after.shifts.shifts_xy[:, 1],
            magnitudes,
            cmap="viridis",
        )
        fig.colorbar(quiver, ax=axes[1], label="residual shift (um)", fraction=0.046)
    axes[1].set_title(f"Residual shift per window ({refinement.status})")
    axes[1].set_axis_off()

    for label, overlay in (("Explorer matrix", before), ("final matrix", after)):
        values = np.linalg.norm(overlay.shifts.shifts_xy, axis=1) * pixel_um
        if len(values):
            axes[2].hist(
                values,
                bins=40,
                histtype="step",
                linewidth=1.5,
                label=f"{label} (median {np.median(values):.2f} um)",
            )
    axes[2].set_xlabel("residual shift (um)")
    axes[2].set_ylabel("windows")
    axes[2].legend()
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, path, dpi=110)
    plt.close(fig)


def _normalize_for_display(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.log1p(np.maximum(image, 0.0))
    sample = values[valid] if valid.any() else values.ravel()
    low, high = np.percentile(sample, [1.0, 99.5]) if sample.size else (0.0, 1.0)
    if high <= low:
        return np.zeros(values.shape, dtype=np.float32)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return np.asarray(scaled, dtype=np.float32)


def _output_paths(
    output_dir: Path, dataset_name: str, image_key: str
) -> dict[str, Path]:
    # Absolute, because the registry stores these paths and Nextflow reruns
    # resolve the relative output_dir inside a different work directory.
    output_dir = Path(output_dir).resolve()
    prefix = f"{_prefix(dataset_name)}_{image_key}"
    return {
        "summary": output_dir / f"{prefix}_registration_summary.json",
        "qc_plot": output_dir / f"{prefix}_registration_qc.png",
        "local_shifts": output_dir / f"{prefix}_registration_local_shifts.csv",
    }


def _prefix(dataset_name: str) -> str:
    return str(dataset_name).lower()


def _json_equal(first: Any, second: Any) -> bool:
    return json.dumps(_jsonable(first), sort_keys=True) == json.dumps(
        _jsonable(second), sort_keys=True
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
