"""Register externally acquired images (e.g. post-Xenium IF) onto the mask grid.

Each image is placed with the 3x3 affine exported by Xenium Explorer, which maps
image level-0 pixels to reference (``morphology_focus``) level-0 pixels. The
matrix is optionally refined by a robust affine fit to local DAPI-to-DAPI shifts,
then the image is resampled tile by tile onto the reference level-0 grid. That
grid is the Cellpose mask grid, so mask image quantification measures the new
channels over the same masks as the instrument channels.

The registration core (``register_onto_reference``) is independent of where the
reference and moving images come from; ``REGISTER_IMAGES`` and the manual
landmark registration workflow both call it.
"""

from __future__ import annotations

import json
import logging
import math
import numbers
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
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
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation
from spatialdata.transformations import get_transformation, set_transformation

from merxen.alignment.image_warp import ImageWarpResult, materialize_warped_image
from merxen.config import (
    ImageRegistrationConfig,
    RegisteredImageSpec,
    RegistrationSettings,
)
from merxen.io.image_source import REGISTERED_IMAGES_ATTR, image_to_cyx
from merxen.io.ome_tiff import OmeTiffInfo, open_ome_tiff_level, read_ome_tiff_info
from merxen.io.spatialdata_io import (
    delete_element,
    spatialdata_write_lock,
    write_spatialdata_metadata,
)
from merxen.memory import force_release, log_status
from merxen.nonrigid_registration import (
    NonrigidFlowResult,
    NonrigidTuning,
    estimate_nonrigid_flow,
)
from merxen.plotting import prepare_plot_output, save_figure
from merxen.registration_quality import score_registered_image_in_spatialdata
from merxen.viewer_cache.build import ViewerCacheParams, build_image_pyramid
from merxen.viewer_cache.format import derived_image_pyramid_cache_key

logger = logging.getLogger(__name__)

REGISTRATION_ALGORITHM = "explorer_affine_dapi_refine_v1"
NONRIGID_REGISTRATION_ALGORITHM = "guarded_tvl1_v3"
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


@dataclass
class _NonrigidRefinement:
    status: str
    reason: str | None = None
    result: NonrigidFlowResult | None = None
    working_to_reference: np.ndarray | None = None
    reference_pixels: np.ndarray | None = None
    affine_warped_pixels: np.ndarray | None = None
    valid: np.ndarray | None = None


WorkingImage = _WorkingImage  # public alias; fields (pixels, to_level0)


def _check_level_factors(
    factors: Sequence[Sequence[float]] | None,
    n_levels: int,
    *,
    label: str,
) -> None:
    """Validate exact per-level ``(fx, fy)`` factors (None passes)."""
    if factors is None:
        return
    message = (
        f"{label}: level_factors_xy must give one positive (fx, fy) per level with "
        f"level 0 = (1, 1); got {factors}"
    )
    if len(factors) != n_levels:
        raise ValueError(message)
    for pair in factors:
        if len(pair) != 2 or not all(
            isinstance(value, numbers.Real | np.integer | np.floating)
            and not isinstance(value, bool | np.bool_)
            and math.isfinite(float(value))
            and value > 0
            for value in pair
        ):
            raise ValueError(message)
    if (float(factors[0][0]), float(factors[0][1])) != (1.0, 1.0):
        raise ValueError(message)


@dataclass(frozen=True)
class ReferenceGrid:
    """Level-0 grid that registered images are resampled onto.

    Attributes:
        key: Element name of the reference image (used in labels and errors).
        levels: ``(c, y, x)`` DataArrays, level 0 first, with ``c`` coordinates.
        transformations: SpatialData transformations of the reference element, by
            coordinate system; copied onto the registered element.
        coordinate_system: Coordinate system whose transformation places the output.
        level_factors_xy: Exact ``(fx, fy)`` level-0 pixels per level pixel for every
            level, level 0 = ``(1.0, 1.0)``; None derives factors from the level
            shapes.
    """

    key: str
    levels: list[xr.DataArray]
    transformations: dict[str, Any]
    coordinate_system: str
    level_factors_xy: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError(f"ReferenceGrid {self.key!r} has no levels")
        if self.coordinate_system not in self.transformations:
            raise ValueError(
                f"ReferenceGrid {self.key!r}: coordinate system "
                f"{self.coordinate_system!r} not in transformations "
                f"{sorted(self.transformations)}"
            )
        _check_level_factors(
            self.level_factors_xy,
            len(self.levels),
            label=f"reference image {self.key!r}",
        )

    @property
    def shape_yx(self) -> tuple[int, int]:
        return (int(self.levels[0].sizes["y"]), int(self.levels[0].sizes["x"]))

    @property
    def level_shapes_yx(self) -> tuple[tuple[int, int], ...]:
        """``(y, x)`` sizes of every level."""
        return tuple(
            (int(level.sizes["y"]), int(level.sizes["x"])) for level in self.levels
        )


_ReferenceGrid = ReferenceGrid  # backward-compatible private name


@dataclass(frozen=True)
class MovingImage:
    """An image to register, independent of its storage.

    Attributes:
        image_key: Name used in labels, logs and errors (``f"image {image_key!r}"``).
        channel_names: Names used to find the registration channel (by index).
        pixel_size_um: Level-0 pixel size in um.
        level_shapes_yx: ``(y, x)`` size of every stored level, level 0 first.
        open_level: Returns level ``i`` as a lazy ``(c, y, x)`` DataArray; channels
            are selected by index, so its ``c`` coordinates are not used.
        level0: Level 0 as a lazy ``(c, y, x)`` DataArray whose ``c`` coordinates
            are ``channel_names``; the source of the final warp.
        level_factors_xy: Exact per-level factors as in :class:`ReferenceGrid`, or
            None.
    """

    image_key: str
    channel_names: tuple[str, ...]
    pixel_size_um: float
    level_shapes_yx: tuple[tuple[int, int], ...]
    open_level: Callable[[int], xr.DataArray]
    level0: xr.DataArray
    level_factors_xy: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        label = f"image {self.image_key!r}"
        if not self.level_shapes_yx:
            raise ValueError(f"{label}: level_shapes_yx is empty")
        if not (math.isfinite(self.pixel_size_um) and self.pixel_size_um > 0):
            raise ValueError(
                f"{label}: pixel_size_um must be finite and > 0, "
                f"got {self.pixel_size_um!r}"
            )
        if len(self.channel_names) != int(self.level0.sizes["c"]):
            raise ValueError(
                f"{label}: {len(self.channel_names)} channel name(s) given for "
                f"level 0 with {int(self.level0.sizes['c'])} channel(s)"
            )
        level0_yx = (int(self.level0.sizes["y"]), int(self.level0.sizes["x"]))
        if level0_yx != tuple(self.level_shapes_yx[0]):
            raise ValueError(
                f"{label}: level 0 has (y, x) = {level0_yx} but level_shapes_yx[0] "
                f"is {tuple(self.level_shapes_yx[0])}"
            )
        _check_level_factors(
            self.level_factors_xy, len(self.level_shapes_yx), label=label
        )


@dataclass(frozen=True)
class RegistrationTuning:
    """Knobs of the registration core that are not fields of ``RegistrationSettings``.

    Defaults reproduce ``REGISTER_IMAGES``.

    Attributes:
        refinement_min_window_quality: Smallest shift-corrected correlation for a
            refinement window to count.
        refinement_require_moving_tissue: Also require tissue in the warped moving
            image for a window to be valid (holes and torn tissue carry no windows).
        refinement_min_correlation_gain: Smallest rise of the registration-channel
            log Pearson correlation (after minus before, both measured on the
            refinement overlay) for a refinement inside the limits to be applied.
            A refinement that raises the correlation by less (or whose correlation
            cannot be measured) is rejected as ``rejected_no_correlation_gain`` and
            the input matrix is kept. None applies every refinement inside the
            limits, as ``REGISTER_IMAGES`` does, and leaves the summary unchanged;
            a number also records ``pearson_before`` and ``pearson_candidate`` in
            the refinement summary.
        nonrigid: Length scales of the guarded TV-L1 flow.
        qc_reference_label: Reference name in the non-rigid QC figure title.
        qc_moving_label: Moving name in the non-rigid QC figure title.
        qc_initial_matrix_label: Name of the input matrix in the QC histogram legend
            and in the rejection log line.
    """

    refinement_min_window_quality: float = _MIN_WINDOW_QUALITY
    refinement_require_moving_tissue: bool = False
    refinement_min_correlation_gain: float | None = None
    nonrigid: NonrigidTuning = field(default_factory=NonrigidTuning)
    qc_reference_label: str = "Xenium DAPI"
    qc_moving_label: str = "IF DAPI"
    qc_initial_matrix_label: str = "Explorer matrix"

    def __post_init__(self) -> None:
        quality = self.refinement_min_window_quality
        if not (
            isinstance(quality, int | float)
            and math.isfinite(quality)
            and -1.0 <= quality <= 1.0
        ):
            raise ValueError(
                "RegistrationTuning.refinement_min_window_quality must be a finite "
                f"number in [-1, 1], got {quality!r}"
            )
        gain = self.refinement_min_correlation_gain
        if gain is not None and not (
            isinstance(gain, int | float)
            and not isinstance(gain, bool)
            and math.isfinite(gain)
            and -2.0 <= gain <= 2.0
        ):
            raise ValueError(
                "RegistrationTuning.refinement_min_correlation_gain must be None or "
                f"a finite number in [-2, 2], got {gain!r}"
            )
        for name in (
            "qc_reference_label",
            "qc_moving_label",
            "qc_initial_matrix_label",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"RegistrationTuning.{name} must be a non-empty string, "
                    f"got {value!r}"
                )


@dataclass
class RegistrationOutcome:
    """Everything ``register_onto_reference`` computed and wrote.

    Attributes:
        refinement: Affine refinement result (matrix, status, overlays).
        nonrigid: Guarded optical-flow result.
        warp: Description of the resampled image written to the store.
        pyramid_status: Viewer pyramid status (``"disabled"`` when not built).
        pull_mapping: Maps reference level-0 ``(x, y)`` to moving level-0 ``(x, y)``.
    """

    refinement: _Refinement
    nonrigid: _NonrigidRefinement
    warp: ImageWarpResult
    pyramid_status: str
    pull_mapping: Callable[[np.ndarray], np.ndarray]

    @property
    def final_matrix(self) -> np.ndarray:
        """Moving to reference level-0 affine after refinement."""
        return self.refinement.matrix

    @property
    def flow_yx(self) -> np.ndarray | None:
        """Pull flow in working pixels, only when the non-rigid stage was applied."""
        if self.nonrigid.status == "applied" and self.nonrigid.result is not None:
            return self.nonrigid.result.flow_yx
        return None

    @property
    def flow_working_to_reference(self) -> np.ndarray | None:
        """Working-grid to reference level-0 affine of the flow, when it applies."""
        if self.nonrigid.status == "applied":
            return self.nonrigid.working_to_reference
        return None

    def overlay_summary(self) -> dict[str, Any]:
        """Working pixel size and the residual statistics before and after."""
        return {
            "working_pixel_size_um": self.refinement.working_pixel_size_um,
            "before": _overlay_stats(self.refinement.before, self.refinement),
            "after": _overlay_stats(self.refinement.after, self.refinement),
        }


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


def write_alignment_matrix(path: Path, matrix: np.ndarray) -> Path:
    """Write a 3x3 affine matrix CSV that ``load_alignment_matrix`` reads back exactly.

    The matrix is validated in memory, written to a temporary sibling with 17
    significant digits (an exact float64 round trip), read back and compared, and
    only then moved into place, so ``path`` is never left half written.

    Args:
        path: Destination CSV; parent directories are created.
        matrix: ``(3, 3)`` finite, invertible affine matrix with last row 0, 0, 1.

    Returns:
        ``path``.

    Raises:
        ValueError: If the matrix is invalid or does not survive the round trip.
    """
    path = Path(path)
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(
            f"Alignment matrix for {path} must be 3x3, got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Alignment matrix for {path} contains non-finite values")
    if not np.allclose(array[2], [0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(
            f"Alignment matrix for {path} is not affine: last row is "
            f"{array[2].tolist()}, expected [0, 0, 1]"
        )
    if abs(np.linalg.det(array[:2, :2])) < 1e-12:
        raise ValueError(f"Alignment matrix for {path} is singular")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        np.savetxt(tmp, array, delimiter=",", fmt="%.17g")
        loaded = load_alignment_matrix(tmp)
        if not np.array_equal(loaded, array):
            raise ValueError(
                f"Alignment matrix for {path} did not survive a write/read round trip"
            )
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


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
        selection = (
            ("images", "labels", "tables")
            if config.registration_quality_enabled
            else ("images",)
        )
        sdata_obj = sd.read_zarr(latest_path, selection=selection)
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
                if config.registration_quality_enabled:
                    summary = _add_registration_quality(
                        sdata_obj=sdata_obj,
                        spec=spec,
                        reference=reference,
                        registry=registry,
                        config=config,
                        summary=summary,
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


def _add_registration_quality(
    *,
    sdata_obj: Any,
    spec: RegisteredImageSpec,
    reference: ReferenceGrid,
    registry: dict[str, Any],
    config: ImageRegistrationConfig,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Score the final registered DAPI and record the result with the image."""
    paths = _output_paths(Path(config.output_dir), config.dataset_name, spec.image_key)
    if spec.registration_channel is None:
        quality = {
            "status": "skipped_no_registration_channel",
            "reason": "registration quality requires a moving DAPI channel",
        }
    else:
        log_status(
            f"[{config.dataset_name}] Scoring per-cell registration quality for "
            f"{spec.image_key}"
        )
        quality = score_registered_image_in_spatialdata(
            sdata_obj,
            image_key=spec.image_key,
            reference_image_key=reference.key,
            reference_channel=config.reference_channel,
            moving_channel=spec.registration_channel,
            cell_label_key=config.registration_quality_cell_label_key,
            nuclei_label_key=config.registration_quality_nuclei_label_key,
            source_pixel_size_um=config.reference_pixel_size_um,
            target_pixel_size_um=config.registration_quality_pixel_size_um,
            smoothing_sigma_um=config.registration_quality_smoothing_sigma_um,
            output_path=paths["registration_quality"],
        )
    updated = {**summary, "registration_quality": quality}
    outputs = dict(updated.get("outputs", {}))
    if quality.get("status") == "complete":
        outputs["registration_quality"] = str(paths["registration_quality"])
    updated["outputs"] = outputs
    updated["summary_path"] = str(paths["summary"])
    paths["summary"].write_text(json.dumps(_jsonable(updated), indent=2) + "\n")

    entry = dict(registry.get(spec.image_key, {}))
    entry["registration_quality"] = _jsonable(quality)
    entry["summary"] = _jsonable(updated)
    registry[spec.image_key] = entry
    _write_registry(sdata_obj, registry)
    return updated


def _register_image(
    *,
    sdata_obj: Any,
    zarr_path: Path,
    spec: RegisteredImageSpec,
    reference: ReferenceGrid,
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
    moving = moving_image_from_ome_tiff(
        info,
        image_key=key,
        channel_names=channel_names,
        pixel_size_um=image_pixel_size_um,
    )
    outcome = register_onto_reference(
        sdata_obj=sdata_obj,
        zarr_path=zarr_path,
        reference=reference,
        moving=moving,
        output_key=key,
        matrix=matrix,
        registration_channel=spec.registration_channel,
        settings=config,
        paths=paths,
        dataset_name=config.dataset_name,
    )
    refinement, nonrigid, warp = outcome.refinement, outcome.nonrigid, outcome.warp
    pyramid_status = outcome.pyramid_status
    overlay = outcome.overlay_summary()

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
        "nonrigid": {
            "enabled": config.nonrigid_enabled,
            "algorithm": NONRIGID_REGISTRATION_ALGORITHM,
            "status": nonrigid.status,
            "reason": nonrigid.reason,
            **(nonrigid.result.stats if nonrigid.result is not None else {}),
        },
        "residuals_before": overlay["before"],
        "residuals_after": overlay["after"],
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
        "nonrigid_status": nonrigid.status,
        "channel_names": channel_names,
        # Kept in the zarr so a rerun in a fresh output directory (a new
        # Nextflow work dir) can skip the warp and still report its result.
        "summary": _jsonable(summary),
    }
    _write_registry(sdata_obj, registry)
    log_status(
        f"[{config.dataset_name}] Registered {key}: refinement={refinement.status}, "
        f"nonrigid={nonrigid.status}, residual median "
        f"{summary['residuals_after'].get('median_um')} um"
    )
    return summary


def moving_image_from_ome_tiff(
    info: OmeTiffInfo,
    *,
    image_key: str,
    channel_names: Sequence[str],
    pixel_size_um: float,
) -> MovingImage:
    """Describe an OME-TIFF file as a :class:`MovingImage`.

    Args:
        info: Metadata of the file from ``read_ome_tiff_info``.
        image_key: Name used in labels, logs and errors.
        channel_names: One name per channel of the file.
        pixel_size_um: Level-0 pixel size in um.

    Returns:
        The lazily read moving image (levels are opened on demand).
    """
    return MovingImage(
        image_key=image_key,
        channel_names=tuple(str(name) for name in channel_names),
        pixel_size_um=float(pixel_size_um),
        level_shapes_yx=tuple((int(s[1]), int(s[2])) for s in info.level_shapes_cyx),
        open_level=lambda level: open_ome_tiff_level(info, level),
        level0=open_ome_tiff_level(info, 0, channel_names=list(channel_names)),
        level_factors_xy=None,
    )


def register_onto_reference(
    *,
    sdata_obj: Any,
    zarr_path: Path,
    reference: ReferenceGrid,
    moving: MovingImage,
    output_key: str,
    matrix: np.ndarray,
    registration_channel: str | None,
    settings: RegistrationSettings,
    paths: Mapping[str, Path],
    dataset_name: str,
    tuning: RegistrationTuning | None = None,
) -> RegistrationOutcome:
    """Refine, resample and store a moving image on a reference grid.

    ``matrix`` maps moving level-0 pixels to reference level-0 pixels. The affine is
    refined on the registration channel, an optional guarded optical flow is
    estimated, the moving image is resampled tile by tile onto the reference level-0
    grid as ``output_key`` in the store behind ``sdata_obj`` at ``zarr_path``, and
    the viewer pyramid is built when ``settings.build_viewer_pyramid`` is set. QC
    files are written to the ``qc_plot``, ``local_shifts``, ``nonrigid_flow`` and
    ``nonrigid_qc_plot`` entries of ``paths``.

    The caller must hold ``spatialdata_write_lock`` of ``zarr_path``; this function
    never takes it. ``tuning=None`` reproduces ``REGISTER_IMAGES``.

    Args:
        sdata_obj: SpatialData object read from ``zarr_path``.
        zarr_path: Store the output image is written to.
        reference: Grid the moving image is resampled onto.
        moving: Image to register.
        output_key: Element name of the registered image.
        matrix: ``(3, 3)`` moving to reference level-0 affine.
        registration_channel: Moving channel compared with the reference channel,
            or None to skip refinement and the non-rigid stage.
        settings: Registration algorithm settings.
        paths: Output files, as from ``registration_output_paths``.
        dataset_name: Dataset label used in QC titles.
        tuning: Knobs beyond ``settings``; None uses the defaults.

    Returns:
        The refinement and flow results, the warp description and the pull mapping.

    Raises:
        ValueError: If ``paths`` lacks a required entry, or the registration channel
            is not found.
    """
    missing = {"qc_plot", "local_shifts", "nonrigid_flow", "nonrigid_qc_plot"} - set(
        paths
    )
    if missing:
        raise ValueError(f"paths is missing {sorted(missing)}")
    # None keeps the call-time module defaults of the non-rigid stage.
    nonrigid_tuning = None if tuning is None else tuning.nonrigid
    tuning = RegistrationTuning() if tuning is None else tuning
    for name in ("qc_plot", "local_shifts", "nonrigid_flow", "nonrigid_qc_plot"):
        Path(paths[name]).parent.mkdir(parents=True, exist_ok=True)

    refinement = _refine_registration(
        reference=reference,
        moving=moving,
        registration_channel=registration_channel,
        matrix=matrix,
        settings=settings,
        tuning=tuning,
    )
    _write_qc_outputs(
        refinement,
        paths,
        dataset_name,
        output_key,
        initial_matrix_label=tuning.qc_initial_matrix_label,
    )
    nonrigid = _refine_nonrigid_registration(
        reference=reference,
        moving=moving,
        registration_channel=registration_channel,
        matrix=refinement.matrix,
        settings=settings,
        tuning=tuning,
        nonrigid_tuning=nonrigid_tuning,
    )
    _write_nonrigid_outputs(
        nonrigid,
        paths,
        dataset_name,
        output_key,
        reference_label=tuning.qc_reference_label,
        moving_label=tuning.qc_moving_label,
    )

    pull_mapping = _affine_function(np.linalg.inv(refinement.matrix))
    if nonrigid.status == "applied":
        assert nonrigid.result is not None
        assert nonrigid.working_to_reference is not None
        pull_mapping = _nonrigid_pull_function(
            refinement.matrix,
            nonrigid.result.flow_yx,
            nonrigid.working_to_reference,
        )
    warp = materialize_warped_image(
        sdata_obj=sdata_obj,
        zarr_path=zarr_path,
        source_image=moving.level0,
        output_key=output_key,
        output_shape_rc=reference.shape_yx,
        fixed_to_source_image_xy=pull_mapping,
        transform=reference.transformations[reference.coordinate_system],
        coordinate_system=reference.coordinate_system,
        tile_size=settings.tile_size,
        chunk_size=settings.chunk_size,
        interpolation="bilinear",
        fill_value=0.0,
    )
    if len(reference.transformations) > 1:
        set_transformation(
            sdata_obj.images[output_key],
            reference.transformations,
            set_all=True,
            write_to_sdata=sdata_obj,
        )
    pyramid_status = "disabled"
    if settings.build_viewer_pyramid:
        base_cyx = xr.DataArray(
            da.from_zarr(str(zarr_path / "images" / output_key / "s0")),
            dims=("c", "y", "x"),
            coords={"c": list(warp.channels)},
        )
        pyramid_status = build_image_pyramid(
            sdata=sdata_obj,
            zarr_path=zarr_path,
            image_key=output_key,
            base_cyx=base_cyx,
            channels=list(warp.channels),
            transform=reference.transformations[reference.coordinate_system],
            params=ViewerCacheParams(force=True),
            coordinate_system=reference.coordinate_system,
        )
    return RegistrationOutcome(
        refinement=refinement,
        nonrigid=nonrigid,
        warp=warp,
        pyramid_status=pyramid_status,
        pull_mapping=pull_mapping,
    )


def measure_registration_overlay(
    *,
    reference: ReferenceGrid,
    moving: MovingImage,
    registration_channel: str,
    matrix: np.ndarray,
    settings: RegistrationSettings,
    tuning: RegistrationTuning | None = None,
) -> dict[str, Any]:
    """Measure the residual shifts of ``matrix`` without refining or writing anything.

    Args:
        reference: Reference grid.
        moving: Image to measure.
        registration_channel: Moving channel compared with the reference channel.
        matrix: ``(3, 3)`` moving to reference level-0 affine.
        settings: Registration settings; ``registration_pixel_size_um`` and
            ``refinement_window_um`` set the scale of the measurement.
        tuning: Knobs beyond ``settings``; None uses the defaults.

    Returns:
        ``working_pixel_size_um``, ``n_windows``, ``n_candidate_windows`` and
        ``pearson_log_registration_channel``, plus ``median_um``, ``p90_um``,
        ``p99_um`` and ``max_um`` when at least one window was accepted.

    Raises:
        ValueError: If ``registration_channel`` is None or not found.
    """
    if registration_channel is None:
        raise ValueError(
            "measure_registration_overlay needs a registration channel; got None"
        )
    tuning = RegistrationTuning() if tuning is None else tuning
    refinement = _refine_registration(
        reference=reference,
        moving=moving,
        registration_channel=registration_channel,
        matrix=matrix,
        settings=settings.model_copy(update={"refine_affine": False}),
        tuning=tuning,
    )
    return {
        "working_pixel_size_um": refinement.working_pixel_size_um,
        **_overlay_stats(refinement.before, refinement),
    }


def _validated_points(points: np.ndarray, *, name: str = "points") -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(
            f"{name} must be a finite (N, 2) array of (x, y), got shape {array.shape}"
        )
    return array


def _validate_flow_inputs(
    flow_yx: np.ndarray | None, working_to_reference: np.ndarray | None
) -> None:
    if (flow_yx is None) != (working_to_reference is None):
        raise ValueError("flow_yx and working_to_reference must be given together")
    if flow_yx is None or working_to_reference is None:
        return
    if np.ndim(flow_yx) != 3 or np.shape(flow_yx)[0] != 2:
        raise ValueError(f"flow_yx must have shape (2, h, w), got {np.shape(flow_yx)}")
    if np.shape(working_to_reference) != (3, 3):
        raise ValueError(
            "working_to_reference must have shape (3, 3), "
            f"got {np.shape(working_to_reference)}"
        )


def pull_points_through_registration(
    points_reference_xy: np.ndarray,
    *,
    final_matrix: np.ndarray,
    flow_yx: np.ndarray | None = None,
    working_to_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Map reference points to moving points exactly as the resampling does.

    Points are ``(N, 2)`` ``(x, y)`` in level-0 pixel-centre coordinates of the
    respective grid. This is the pull mapping used by ``materialize_warped_image``:
    ``M^-1 r`` without a flow, ``M^-1 W (w + d(w))`` with ``w = W^-1 r`` and the
    flow ``d = (flow[1], flow[0])`` sampled bilinearly (zero outside the grid).

    Args:
        points_reference_xy: ``(N, 2)`` reference points.
        final_matrix: ``(3, 3)`` moving to reference affine.
        flow_yx: ``(2, h, w)`` pull displacement in working pixels, or None.
        working_to_reference: ``(3, 3)`` working grid to reference affine; required
            with ``flow_yx``.

    Returns:
        ``(N, 2)`` float64 moving points.

    Raises:
        ValueError: If the points or the flow inputs are malformed.
    """
    points = _validated_points(points_reference_xy)
    _validate_flow_inputs(flow_yx, working_to_reference)
    matrix = np.asarray(final_matrix, dtype=np.float64)
    if flow_yx is None or working_to_reference is None:
        return _affine_function(np.linalg.inv(matrix))(points)
    return _nonrigid_pull_function(matrix, flow_yx, working_to_reference)(points)


def push_points_through_registration(
    points_moving_xy: np.ndarray,
    *,
    final_matrix: np.ndarray,
    flow_yx: np.ndarray | None = None,
    working_to_reference: np.ndarray | None = None,
    max_iterations: int = 50,
    tolerance_px: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Map moving points to reference points by inverting the pull mapping.

    Without a flow this is the affine ``M m``. With a flow the pull of ``r = W w`` is
    ``M^-1 W (w + d(w))``, so the working coordinates solve ``w = q - d(w)`` with
    ``q = W^-1 M m``; the fixed point is found by iteration from ``w = q``. Points
    outside the flow grid see ``d = 0`` and reduce to the affine. The iteration
    converges when the flow is a contraction, which the Jacobian guards ensure in
    practice.

    Args:
        points_moving_xy: ``(N, 2)`` moving points.
        final_matrix: ``(3, 3)`` moving to reference affine.
        flow_yx: ``(2, h, w)`` pull displacement in working pixels, or None.
        working_to_reference: ``(3, 3)`` working grid to reference affine; required
            with ``flow_yx``.
        max_iterations: Largest number of fixed-point iterations.
        tolerance_px: Step size, in working pixels, below which a point has converged.

    Returns:
        ``(points_reference_xy, converged)``; ``converged`` is a boolean ``(N,)``
        array. Points that did not converge keep their last iterate.

    Raises:
        ValueError: If the points, the flow inputs or the iteration limits are
            malformed.
    """
    points = _validated_points(points_moving_xy, name="points")
    _validate_flow_inputs(flow_yx, working_to_reference)
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if not (math.isfinite(tolerance_px) and tolerance_px > 0):
        raise ValueError(f"tolerance_px must be finite and > 0, got {tolerance_px}")
    matrix = np.asarray(final_matrix, dtype=np.float64)
    if flow_yx is None or working_to_reference is None:
        return (
            _affine_function(matrix)(points),
            np.ones(len(points), dtype=bool),
        )

    working = np.asarray(working_to_reference, dtype=np.float64)
    target = _affine_function(np.linalg.inv(working) @ matrix)(points)
    current = target.copy()
    converged = np.zeros(len(points), dtype=bool)
    active = np.ones(len(points), dtype=bool)
    flow = np.asarray(flow_yx, dtype=np.float32)
    for _ in range(int(max_iterations)):
        if not active.any():
            break
        at = current[active]
        coordinates = [at[:, 1], at[:, 0]]
        displacement_x = map_coordinates(
            flow[1], coordinates, order=1, mode="constant", cval=0.0, prefilter=False
        )
        displacement_y = map_coordinates(
            flow[0], coordinates, order=1, mode="constant", cval=0.0, prefilter=False
        )
        updated = target[active] - np.column_stack([displacement_x, displacement_y])
        step = np.linalg.norm(updated - at, axis=1)
        current[active] = updated
        done = step < tolerance_px
        indices = np.flatnonzero(active)
        converged[indices[done]] = True
        active[indices[done]] = False
    return _affine_function(working)(current), converged


def _refine_registration(
    *,
    reference: ReferenceGrid,
    moving: MovingImage,
    registration_channel: str | None,
    matrix: np.ndarray,
    settings: RegistrationSettings,
    tuning: RegistrationTuning,
) -> _Refinement:
    if registration_channel is None:
        return _Refinement(matrix=matrix, status="skipped_no_registration_channel")

    reference_channel = _channel_index(
        _channel_names(reference.levels[0]),
        settings.reference_channel,
        label=f"reference image {reference.key!r}",
    )
    moving_channel = _channel_index(
        list(moving.channel_names),
        registration_channel,
        label=f"image {moving.image_key!r}",
    )
    reference_work = read_working_channel(
        open_level=lambda level: reference.levels[level],
        level_shapes_yx=reference.level_shapes_yx,
        channel_index=reference_channel,
        target_factor=settings.registration_pixel_size_um
        / settings.reference_pixel_size_um,
        level_factors_xy=reference.level_factors_xy,
    )
    moving_work = read_working_channel(
        open_level=moving.open_level,
        level_shapes_yx=moving.level_shapes_yx,
        channel_index=moving_channel,
        target_factor=settings.registration_pixel_size_um / moving.pixel_size_um,
        level_factors_xy=moving.level_factors_xy,
    )
    working_px_um = settings.reference_pixel_size_um * float(
        np.mean(np.diag(reference_work.to_level0)[:2])
    )
    window_px = max(
        _MIN_WINDOW_PX, int(round(settings.refinement_window_um / working_px_um))
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
            min_quality=tuning.refinement_min_window_quality,
            require_moving_tissue=tuning.refinement_require_moving_tissue,
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
    if not settings.refine_affine:
        return result
    if n_windows < settings.min_refinement_windows:
        result.status = "skipped_insufficient_windows"
        result.reason = (
            f"{n_windows} usable windows < min_refinement_windows "
            f"({settings.min_refinement_windows})"
        )
        logger.warning("%s: %s", moving.image_key, result.reason)
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
    if diagnostics["max_shift_um"] > settings.max_refinement_shift_um:
        violations.append(
            f"shift {diagnostics['max_shift_um']:.2f} um > "
            f"{settings.max_refinement_shift_um} um"
        )
    if abs(diagnostics["rotation_deg"]) > settings.max_refinement_rotation_deg:
        violations.append(
            f"rotation {diagnostics['rotation_deg']:.3f} deg > "
            f"{settings.max_refinement_rotation_deg} deg"
        )
    if abs(diagnostics["scale_change"]) > settings.max_refinement_scale_change:
        violations.append(
            f"scale change {diagnostics['scale_change']:.4f} > "
            f"{settings.max_refinement_scale_change}"
        )
    if violations:
        result.status = "rejected_exceeds_limits"
        result.reason = "; ".join(violations)
        logger.warning(
            "%s: refinement rejected, keeping the %s (%s)",
            moving.image_key,
            tuning.qc_initial_matrix_label,
            result.reason,
        )
        return result

    to_level0 = reference_work.to_level0
    refined = to_level0 @ correction @ np.linalg.inv(to_level0) @ matrix
    candidate = overlay(refined)
    min_gain = tuning.refinement_min_correlation_gain
    if min_gain is not None:
        gain = candidate.pearson_log - before.pearson_log
        result.correction = {
            **result.correction,
            "pearson_before": before.pearson_log,
            "pearson_candidate": candidate.pearson_log,
        }
        # "not >=" also fails a NaN: an unmeasurable correlation cannot show a gain.
        if not gain >= min_gain:
            result.status = "rejected_no_correlation_gain"
            result.reason = _correlation_gain_reason(
                registration_channel,
                before.pearson_log,
                candidate.pearson_log,
                gain,
                min_gain,
            )
            logger.warning(
                "%s: refinement rejected, keeping the %s (%s)",
                moving.image_key,
                tuning.qc_initial_matrix_label,
                result.reason,
            )
            return result
    result.matrix = refined
    result.status = "applied"
    result.after = candidate
    return result


def _correlation_gain_reason(
    channel: str, before: float, candidate: float, gain: float, min_gain: float
) -> str:
    """Describe a refinement that did not raise the registration-channel correlation.

    Args:
        channel: Name of the registration channel the correlation was measured on.
        before: Correlation of the overlay with the input matrix.
        candidate: Correlation of the overlay with the refined matrix.
        gain: ``candidate - before``.
        min_gain: The smallest gain that would have been accepted.

    Returns:
        The reason text recorded in the refinement summary and the log.
    """
    if math.isfinite(gain):
        return (
            f"{channel} correlation {before:.4f} -> {candidate:.4f} "
            f"(gain {gain:.4f} < {min_gain})"
        )
    return (
        f"{channel} correlation {before:.4f} -> {candidate:.4f} "
        f"(gain undefined, required >= {min_gain})"
    )


def _refine_nonrigid_registration(
    *,
    reference: ReferenceGrid,
    moving: MovingImage,
    registration_channel: str | None,
    matrix: np.ndarray,
    settings: RegistrationSettings,
    tuning: RegistrationTuning,
    nonrigid_tuning: NonrigidTuning | None,
) -> _NonrigidRefinement:
    if not settings.nonrigid_enabled:
        return _NonrigidRefinement(status="disabled")
    if registration_channel is None:
        return _NonrigidRefinement(
            status="skipped_no_registration_channel",
            reason="non-rigid refinement requires a registration channel",
        )

    reference_channel = _channel_index(
        _channel_names(reference.levels[0]),
        settings.reference_channel,
        label=f"reference image {reference.key!r}",
    )
    moving_channel = _channel_index(
        list(moving.channel_names),
        registration_channel,
        label=f"image {moving.image_key!r}",
    )
    reference_work = read_working_channel(
        open_level=lambda level: reference.levels[level],
        level_shapes_yx=reference.level_shapes_yx,
        channel_index=reference_channel,
        target_factor=settings.nonrigid_pixel_size_um
        / settings.reference_pixel_size_um,
        level_factors_xy=reference.level_factors_xy,
    )
    moving_work = read_working_channel(
        open_level=moving.open_level,
        level_shapes_yx=moving.level_shapes_yx,
        channel_index=moving_channel,
        target_factor=settings.nonrigid_pixel_size_um / moving.pixel_size_um,
        level_factors_xy=moving.level_factors_xy,
    )
    working_pixel_size_um = settings.reference_pixel_size_um * float(
        np.mean(np.diag(reference_work.to_level0)[:2])
    )
    affine_warped, footprint = _warp_working_image_to_reference(
        reference_work,
        moving_work,
        matrix,
    )
    valid = (
        _tissue_mask(reference_work.pixels) & _tissue_mask(affine_warped) & footprint
    )
    result = estimate_nonrigid_flow(
        reference_work.pixels,
        affine_warped,
        valid,
        pixel_size_um=working_pixel_size_um,
        attachment=settings.nonrigid_attachment,
        tightness=settings.nonrigid_tightness,
        num_warp=settings.nonrigid_num_warp,
        num_iter=settings.nonrigid_num_iter,
        smoothing_sigma_um=settings.nonrigid_smoothing_sigma_um,
        max_displacement_um=settings.nonrigid_max_displacement_um,
        max_cycle_error_um=settings.nonrigid_max_cycle_error_um,
        min_local_correlation=settings.nonrigid_min_local_correlation,
        min_jacobian=settings.nonrigid_min_jacobian,
        max_jacobian=settings.nonrigid_max_jacobian,
        min_confident_fraction=settings.nonrigid_min_confident_fraction,
        residual_passes=settings.nonrigid_residual_passes,
        min_residual_correlation_improvement=(
            settings.nonrigid_min_residual_correlation_improvement
        ),
        min_residual_initial_confidence=(
            settings.nonrigid_min_residual_initial_confidence
        ),
        tuning=nonrigid_tuning,
    )
    if result.status != "applied":
        logger.warning(
            "%s: non-rigid refinement rejected, keeping the affine registration (%s)",
            moving.image_key,
            result.reason,
        )
    return _NonrigidRefinement(
        status=result.status,
        reason=result.reason,
        result=result,
        working_to_reference=reference_work.to_level0,
        reference_pixels=reference_work.pixels,
        affine_warped_pixels=affine_warped,
        valid=valid,
    )


def _warp_working_image_to_reference(
    reference: _WorkingImage,
    moving: _WorkingImage,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    return warped, footprint


def _measure_overlay(
    reference: _WorkingImage,
    reference_hp: np.ndarray,
    tissue: np.ndarray,
    moving: _WorkingImage,
    matrix: np.ndarray,
    window_px: int,
    *,
    min_quality: float,
    require_moving_tissue: bool,
) -> _Overlay:
    warped, footprint = _warp_working_image_to_reference(reference, moving, matrix)
    valid = tissue & footprint
    if require_moving_tissue:
        # LCM holes and torn tissue carry no windows.
        valid = valid & _tissue_mask(warped)
    shifts = measure_local_shifts(
        reference_hp,
        _highpass_log(warped),
        valid,
        window_px=window_px,
        min_quality=min_quality,
    )
    return _Overlay(
        shifts=shifts,
        pearson_log=_masked_log_pearson(reference.pixels, warped, valid),
        warped=warped,
        valid=valid,
    )


def read_working_channel(
    *,
    open_level: Callable[[int], Any],
    level_shapes_yx: Sequence[tuple[int, int]],
    channel_index: int,
    target_factor: float,
    level_factors_xy: Sequence[tuple[float, float]] | None = None,
) -> WorkingImage:
    """Read one channel near ``target_factor`` x the level-0 pixel size.

    Uses the coarsest stored level that is not coarser than the target, then
    block-averages by an integer factor, so non-pyramidal files never need a
    full-resolution in-memory copy.

    Args:
        open_level: Returns level ``i`` as a ``(c, y, x)`` array.
        level_shapes_yx: ``(y, x)`` size of every level, level 0 first.
        channel_index: Channel to read.
        target_factor: Wanted working pixel size in level-0 pixels.
        level_factors_xy: Exact ``(fx, fy)`` level-0 pixels per level pixel for
            every level (level 0 = ``(1, 1)``). None derives them from the level
            shapes, which is only approximate for odd-sized pyramids and for crops.

    Returns:
        The working pixels and the pixel-centre affine from working pixels to
        level-0 pixels.

    Raises:
        ValueError: If ``level_factors_xy`` is inconsistent with the levels.
    """
    height0, width0 = level_shapes_yx[0]
    level = 0
    if level_factors_xy is None:
        # Pyramid levels of odd-sized images are slightly more than 2x apart
        # (51313 / 12828 = 4.00008), so factors are compared with a 1% tolerance.
        for index, (_height, width) in enumerate(level_shapes_yx):
            if width0 / width <= target_factor * (1.0 + _FACTOR_TOLERANCE):
                level = index
        height, width = level_shapes_yx[level]
        level_factor_x, level_factor_y = width0 / width, height0 / height
    else:
        _check_level_factors(
            level_factors_xy, len(level_shapes_yx), label="read_working_channel"
        )
        for index, (factor_x, _factor_y) in enumerate(level_factors_xy):
            if factor_x <= target_factor * (1.0 + _FACTOR_TOLERANCE):
                level = index
        height, width = level_shapes_yx[level]
        level_factor_x, level_factor_y = level_factors_xy[level]
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
    return WorkingImage(pixels=pixels, to_level0=to_level0)


_read_working_channel = read_working_channel


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


def _nonrigid_pull_function(
    image_to_reference_matrix: np.ndarray,
    flow_yx: np.ndarray,
    working_to_reference: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Compose a coarse fixed-grid pull flow with the inverse image affine."""
    flow = np.asarray(flow_yx, dtype=np.float32)
    reference_to_working = np.linalg.inv(working_to_reference)
    working_to_source = np.linalg.inv(image_to_reference_matrix) @ working_to_reference
    to_working = _affine_function(reference_to_working)
    to_source = _affine_function(working_to_source)

    def apply(reference_xy: np.ndarray) -> np.ndarray:
        working_xy = to_working(reference_xy)
        coordinates = [working_xy[:, 1], working_xy[:, 0]]
        displacement_y = map_coordinates(
            flow[0],
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        displacement_x = map_coordinates(
            flow[1],
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        displaced = working_xy.copy()
        displaced[:, 0] += displacement_x
        displaced[:, 1] += displacement_y
        return to_source(displaced)

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


def reference_grid(
    sdata_obj: Any, image_key: str, *, dataset_name: str
) -> ReferenceGrid:
    """Read the reference image of a SpatialData object as a registration grid.

    Args:
        sdata_obj: SpatialData object holding the reference image.
        image_key: Element name of the reference image.
        dataset_name: Dataset label used in error messages.

    Returns:
        The reference levels, transformations and output coordinate system.

    Raises:
        KeyError: If ``image_key`` is not an image of ``sdata_obj``.
    """
    key = image_key
    if key not in sdata_obj.images:
        raise KeyError(
            f"[{dataset_name}] Reference image {key!r} not found; images: "
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
    return ReferenceGrid(
        key=key,
        levels=levels,
        transformations=dict(transformations),
        coordinate_system=coordinate_system,
    )


def _reference_grid(sdata_obj: Any, config: ImageRegistrationConfig) -> ReferenceGrid:
    return reference_grid(
        sdata_obj, config.reference_image_key, dataset_name=config.dataset_name
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
    settings = {
        key: value
        for key, value in settings.items()
        if not str(key).startswith("registration_quality_")
    }
    if not config.nonrigid_enabled:
        settings = {
            key: value
            for key, value in settings.items()
            if not str(key).startswith("nonrigid_")
        }
    provenance: dict[str, Any] = _jsonable(
        {
            "algorithm": (
                f"{REGISTRATION_ALGORITHM}+{NONRIGID_REGISTRATION_ALGORITHM}"
                if config.nonrigid_enabled
                else REGISTRATION_ALGORITHM
            ),
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
    paths: Mapping[str, Path],
    dataset_name: str,
    image_key: str,
    *,
    initial_matrix_label: str,
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
    _plot_registration_qc(
        refinement,
        paths["qc_plot"],
        f"{dataset_name} {image_key}",
        initial_matrix_label=initial_matrix_label,
    )


def _write_nonrigid_outputs(
    refinement: _NonrigidRefinement,
    paths: Mapping[str, Path],
    dataset_name: str,
    image_key: str,
    *,
    reference_label: str,
    moving_label: str,
) -> None:
    if refinement.result is None or refinement.working_to_reference is None:
        return
    result = refinement.result
    np.savez_compressed(
        paths["nonrigid_flow"],
        flow_yx=result.flow_yx.astype(np.float32, copy=False),
        confidence=result.confidence.astype(np.uint8, copy=False),
        residual_confidence=result.residual_confidence.astype(np.uint8, copy=False),
        rejection_reasons=result.rejection_reasons.astype(np.uint16, copy=False),
        local_correlation=result.local_correlation.astype(np.float32, copy=False),
        cycle_error_px=result.cycle_error_px.astype(np.float32, copy=False),
        jacobian_determinant=result.jacobian_determinant.astype(
            np.float32,
            copy=False,
        ),
        working_to_reference=np.asarray(
            refinement.working_to_reference,
            dtype=np.float64,
        ),
        algorithm=np.asarray(NONRIGID_REGISTRATION_ALGORITHM),
    )
    _plot_nonrigid_qc(
        refinement,
        paths["nonrigid_qc_plot"],
        f"{dataset_name} {image_key}",
        reference_label=reference_label,
        moving_label=moving_label,
    )


def _plot_nonrigid_qc(
    refinement: _NonrigidRefinement,
    path: Path,
    title: str,
    *,
    reference_label: str,
    moving_label: str,
) -> None:
    result = refinement.result
    reference = refinement.reference_pixels
    affine_warped = refinement.affine_warped_pixels
    valid = refinement.valid
    assert result is not None
    assert reference is not None
    assert affine_warped is not None
    assert valid is not None
    prepare_plot_output(path)
    stride = max(1, math.ceil(max(reference.shape) / _MAX_QC_DISPLAY_PX))
    reference_display = _normalize_for_display(reference, valid)
    affine_display = _normalize_for_display(affine_warped, valid)
    warped_display = _normalize_for_display(result.warped, valid)
    before_rgb = np.dstack([affine_display, reference_display, affine_display])[
        ::stride, ::stride
    ]
    after_rgb = np.dstack([warped_display, reference_display, warped_display])[
        ::stride, ::stride
    ]
    pixel_size_um = float(result.stats.get("pixel_size_um", 1.0))
    magnitude_um = np.linalg.norm(result.flow_yx, axis=0) * pixel_size_um

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes[0, 0].imshow(before_rgb, interpolation="nearest")
    axes[0, 0].set_title(
        f"Affine: {reference_label} (green) / {moving_label} (magenta)"
    )
    axes[0, 0].set_axis_off()
    axes[0, 1].imshow(after_rgb, interpolation="nearest")
    axes[0, 1].set_title(f"Non-rigid result ({result.status})")
    axes[0, 1].set_axis_off()

    magnitude_plot = axes[1, 0].imshow(
        magnitude_um[::stride, ::stride],
        cmap="viridis",
        interpolation="nearest",
    )
    axes[1, 0].contour(
        result.confidence[::stride, ::stride],
        levels=[0.5],
        colors="white",
        linewidths=0.4,
    )
    axes[1, 0].set_title("Applied displacement; white = confident raw flow")
    axes[1, 0].set_axis_off()
    fig.colorbar(magnitude_plot, ax=axes[1, 0], label="displacement (um)")

    correlation_plot = axes[1, 1].imshow(
        result.local_correlation[::stride, ::stride],
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1, 1].set_title("Local DAPI correlation after final guarded flow")
    axes[1, 1].set_axis_off()
    fig.colorbar(correlation_plot, ax=axes[1, 1], label="local correlation")
    reason = f"; {result.reason}" if result.reason else ""
    fig.suptitle(f"{title}: {result.status}{reason}")
    fig.tight_layout()
    save_figure(fig, path, dpi=110)
    plt.close(fig)


def _plot_registration_qc(
    refinement: _Refinement,
    path: Path,
    title: str,
    *,
    initial_matrix_label: str,
) -> None:
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

    for label, overlay in ((initial_matrix_label, before), ("final matrix", after)):
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


def registration_output_paths(
    output_dir: Path, dataset_name: str, image_key: str
) -> dict[str, Path]:
    """Names of the files written for one registered image.

    Args:
        output_dir: Directory of the outputs (resolved to an absolute path).
        dataset_name: Dataset label; lower-cased in the file prefix.
        image_key: Name of the registered image.

    Returns:
        Absolute paths keyed ``summary``, ``qc_plot``, ``local_shifts``,
        ``nonrigid_flow``, ``nonrigid_qc_plot`` and ``registration_quality``.
    """
    # Absolute, because the registry stores these paths and Nextflow reruns
    # resolve the relative output_dir inside a different work directory.
    output_dir = Path(output_dir).resolve()
    prefix = f"{_prefix(dataset_name)}_{image_key}"
    return {
        "summary": output_dir / f"{prefix}_registration_summary.json",
        "qc_plot": output_dir / f"{prefix}_registration_qc.png",
        "local_shifts": output_dir / f"{prefix}_registration_local_shifts.csv",
        "nonrigid_flow": output_dir / f"{prefix}_nonrigid_flow.npz",
        "nonrigid_qc_plot": output_dir / f"{prefix}_nonrigid_qc.png",
        "registration_quality": (output_dir / f"{prefix}_registration_quality.csv.gz"),
    }


_output_paths = registration_output_paths


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
