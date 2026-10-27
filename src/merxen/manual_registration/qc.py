"""Tissue-scale alignment QC of a manual registration, per stage.

The tissue QC compares the reference and the moving registration channel at a
coarse working pixel size (about 10 um) after a band-pass that keeps tissue-scale
structure (layers, vessels, boundaries) and drops both nuclei and slow shading.
It is computed for the coarse, landmark, affine and final mappings, so a
registration that fails against the tissue shows up whatever the nuclei do.

Frames: working images come from ``read_working_channel`` of the cropped images;
their ``to_level0`` maps working pixels to crop-frame level-0 pixels. A *pull
mapping* maps crop-frame reference level-0 ``(x, y)`` to crop-frame moving level-0
``(x, y)``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.filters import threshold_otsu

from merxen.image_registration import WorkingImage
from merxen.plotting import prepare_plot_output, save_figure

logger = logging.getLogger(__name__)

PullMapping = Callable[[np.ndarray], np.ndarray]
_MIN_MASK_PIXELS = 50
_MAX_FIGURE_PX = 2000


@dataclass(frozen=True)
class TissueQcResult:
    """Tissue-scale agreement of the registration stages.

    Attributes:
        pixel_size_um: Working pixel size the QC was computed at.
        pearson: Stage name to band-passed Pearson correlation inside the tissue
            mask (NaN when the mask has fewer than 50 pixels).
        mask_fraction: Stage name to the mask's fraction of the reference image.
    """

    pixel_size_um: float
    pearson: dict[str, float]
    mask_fraction: dict[str, float]


def affine_pull(matrix: np.ndarray) -> PullMapping:
    """Return the pull mapping of a moving-to-reference affine: ``xy -> inv(M) xy``."""
    inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    linear = inverse[:2, :2]
    offset = inverse[:2, 2]

    def pull(xy: np.ndarray) -> np.ndarray:
        return np.asarray(np.asarray(xy, dtype=np.float64) @ linear.T + offset)

    return pull


def _apply(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.asarray(xy, dtype=np.float64) @ matrix[:2, :2].T + matrix[:2, 2]
    )


def resample_working(
    reference: WorkingImage,
    moving: WorkingImage,
    pull: PullMapping,
    *,
    block_rows: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample the moving working image onto the reference working grid.

    Args:
        reference: Reference working image (defines the output grid).
        moving: Moving working image.
        pull: Pull mapping from crop-frame reference level-0 to crop-frame moving
            level-0 pixels.
        block_rows: Rows of the reference grid processed at a time.

    Returns:
        ``(warped, footprint)``: float32 pixels and the boolean mask of reference
        pixels whose source lies inside the moving working image, both of the
        reference working shape.
    """
    height, width = reference.pixels.shape
    moving_height, moving_width = moving.pixels.shape
    moving_inverse = np.linalg.inv(np.asarray(moving.to_level0, dtype=np.float64))
    to_level0 = np.asarray(reference.to_level0, dtype=np.float64)
    warped = np.zeros((height, width), dtype=np.float32)
    footprint = np.zeros((height, width), dtype=bool)
    columns = np.arange(width, dtype=np.float64)
    for start in range(0, height, block_rows):
        stop = min(start + block_rows, height)
        jj, ii = np.meshgrid(columns, np.arange(start, stop, dtype=np.float64))
        level0_reference = _apply(to_level0, np.column_stack([jj.ravel(), ii.ravel()]))
        working = _apply(moving_inverse, pull(level0_reference))
        # Rounding keeps exact-integer coordinates exact (no 1e-13 spill past an edge).
        mx = np.round(working[:, 0], 9).reshape(jj.shape)
        my = np.round(working[:, 1], 9).reshape(jj.shape)
        warped[start:stop] = map_coordinates(
            moving.pixels,
            [my, mx],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        footprint[start:stop] = (
            (mx >= 0) & (mx <= moving_width - 1) & (my >= 0) & (my <= moving_height - 1)
        )
    return warped, footprint


def _masked_smooth(values: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    weight = mask.astype(np.float32)
    numerator = gaussian_filter(values * weight, sigma, mode="constant", cval=0.0)
    denominator = gaussian_filter(weight, sigma, mode="constant", cval=0.0)
    return np.asarray(numerator / np.maximum(denominator, 1e-6), dtype=np.float32)


def _log_image(pixels: np.ndarray) -> np.ndarray:
    return np.asarray(np.log1p(np.maximum(pixels, 0.0)), dtype=np.float32)


def _bandpass(
    pixels: np.ndarray, mask: np.ndarray, low_sigma: float, high_sigma: float
) -> np.ndarray:
    logged = _log_image(pixels)
    band = _masked_smooth(logged, mask, low_sigma) - _masked_smooth(
        logged, mask, high_sigma
    )
    return np.where(mask, band, 0.0).astype(np.float32, copy=False)


def _tissue(pixels: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    smoothed = _masked_smooth(_log_image(pixels), mask, max(1.0, sigma))
    sample = smoothed[mask][::4]
    if sample.size == 0 or float(sample.max()) <= float(sample.min()):
        return np.zeros(pixels.shape, dtype=bool)
    return np.asarray((smoothed > threshold_otsu(sample)) & mask)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def tissue_alignment_qc(
    reference: WorkingImage,
    moving: WorkingImage,
    pull_mappings: Mapping[str, PullMapping],
    *,
    working_pixel_size_um: float,
    sigma_um: float,
    highpass_sigma_um: float,
) -> TissueQcResult:
    """Measure tissue-scale agreement of each registration stage.

    Both images are log-transformed and band-passed (Gaussian ``sigma_um`` minus
    Gaussian ``highpass_sigma_um``, masked-normalised so empty borders do not
    leak in). The Pearson correlation is taken over pixels that are tissue in the
    reference and in the warped moving image and inside the moving footprint.

    Args:
        reference: Reference working image.
        moving: Moving working image.
        pull_mappings: Stage name to pull mapping, in the order to report.
        working_pixel_size_um: Pixel size of the working grid in um.
        sigma_um: Band-pass lower length scale.
        highpass_sigma_um: Band-pass upper length scale.

    Returns:
        Pearson and mask fraction per stage (NaN Pearson with fewer than 50
        mask pixels).
    """
    low = sigma_um / working_pixel_size_um
    high = highpass_sigma_um / working_pixel_size_um
    ones = np.ones(reference.pixels.shape, dtype=bool)
    reference_band = _bandpass(reference.pixels, ones, low, high)
    reference_tissue = _tissue(reference.pixels, ones, low)
    pearson: dict[str, float] = {}
    mask_fraction: dict[str, float] = {}
    for stage, pull in pull_mappings.items():
        warped, footprint = resample_working(reference, moving, pull)
        mask = reference_tissue & _tissue(warped, footprint, low) & footprint
        if int(mask.sum()) >= _MIN_MASK_PIXELS:
            warped_band = _bandpass(warped, footprint, low, high)
            pearson[stage] = _correlation(reference_band[mask], warped_band[mask])
        else:
            pearson[stage] = float("nan")
        mask_fraction[stage] = float(mask.mean())
        logger.info(
            "Tissue QC %s: Pearson %.3f over %.1f%% of the reference",
            stage,
            pearson[stage],
            100.0 * mask_fraction[stage],
        )
    return TissueQcResult(
        pixel_size_um=float(working_pixel_size_um),
        pearson=pearson,
        mask_fraction=mask_fraction,
    )


def _normalised(pixels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    logged = _log_image(pixels)
    values = logged[valid]
    if values.size == 0:
        return np.zeros(pixels.shape, dtype=np.float32)
    low, high = np.percentile(values, [1.0, 99.5])
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.float32)
    scaled = np.clip((logged - low) / (high - low), 0.0, 1.0)
    return np.where(valid, scaled, 0.0).astype(np.float32)


def write_tissue_qc_figure(
    reference: WorkingImage,
    moving: WorkingImage,
    pull_mappings: Mapping[str, PullMapping],
    path: Path,
    title: str,
) -> Path:
    """Write the tissue QC figure (PNG and PDF twin) and return the PNG path.

    Panels: reference (green) and warped moving (magenta) overlays for the
    ``landmark`` and ``final`` stages, and a checkerboard of the ``final`` stage.
    A panel whose stage is missing is skipped.
    """
    path = prepare_plot_output(path)
    height, width = reference.pixels.shape
    stride = max(1, int(np.ceil(max(height, width) / _MAX_FIGURE_PX)))
    ones = np.ones(reference.pixels.shape, dtype=bool)
    reference_image = _normalised(reference.pixels, ones)
    panels: list[tuple[str, np.ndarray]] = []
    final_warped: np.ndarray | None = None
    for stage in ("landmark", "final"):
        if stage not in pull_mappings:
            continue
        warped, footprint = resample_working(reference, moving, pull_mappings[stage])
        moving_image = _normalised(warped, footprint)
        overlay = np.dstack([moving_image, reference_image, moving_image])
        panels.append((f"{stage}: reference green, moving magenta", overlay))
        if stage == "final":
            final_warped = moving_image
    if final_warped is not None:
        square = max(8, min(height, width) // 16)
        rows, cols = np.indices(reference.pixels.shape)
        board = ((rows // square) + (cols // square)) % 2 == 0
        checker = np.where(board, reference_image, final_warped)
        panels.append(("final: checkerboard", np.dstack([checker] * 3)))
    figure, axes = plt.subplots(1, 3, figsize=(21, 7), squeeze=False)
    for axis in axes[0]:
        axis.set_axis_off()
    for axis, (label, image) in zip(axes[0], panels, strict=False):
        axis.imshow(image[::stride, ::stride], interpolation="nearest")
        axis.set_title(label)
    figure.suptitle(title)
    figure.tight_layout()
    save_figure(figure, path, dpi=110)
    plt.close(figure)
    return path
