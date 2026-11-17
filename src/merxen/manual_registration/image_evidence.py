"""Image evidence for serial-section registration of cortex pairs.

Band-pass representations of the AT8, MAP2 and DAPI channels at 2.76 um/px give two
kinds of evidence: a per-click image check (confirm, contradict, ridge, inconclusive)
and tissue-scale window observations calibrated against an empirical null.

Frames (contract C2): every image's working grid maps to full-frame level-0 px by
``to_level0`` and full-frame um are level-0 px times the pixel size. ``pull_um`` maps
reference full um to moving full um and is never assumed to be affine. Positions
are returned in full um; covariances and directions in the moving frame.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def bandpass_log(
    pixels: np.ndarray,
    mask: np.ndarray,
    pixel_size_um: float,
    *,
    highpass_sigma_um: float,
    smooth_sigma_um: float,
) -> np.ndarray:
    """Band-pass of the log image: tissue structure, no illumination or nuclei.

    Outside the mask the image is set to its in-mask median before the high-pass (so
    the tissue edge does not ring) and to 0 afterwards.

    Args:
        pixels: Raw 2-D working image.
        mask: Tissue mask.
        pixel_size_um: Working pixel size.
        highpass_sigma_um: Gaussian sigma of the background that is subtracted.
        smooth_sigma_um: Gaussian sigma of the final smoothing.

    Returns:
        float32 image, 0 outside the mask.
    """
    image = np.log1p(np.clip(np.asarray(pixels, dtype=np.float32), 0.0, None))
    fill = float(np.median(image[mask])) if mask.any() else 0.0
    image = np.where(mask, image, fill).astype(np.float32)
    high = image - ndimage.gaussian_filter(image, highpass_sigma_um / pixel_size_um)
    band = ndimage.gaussian_filter(high, smooth_sigma_um / pixel_size_um)
    return np.where(mask, band, 0.0).astype(np.float32)


def parabolic_peak(values: np.ndarray, index: int) -> float:
    """Sub-pixel position of the peak at integer ``index`` of a 1-D profile."""
    if 0 < index < len(values) - 1:
        left, centre, right = (float(values[i]) for i in (index - 1, index, index + 1))
        denominator = left - 2.0 * centre + right
        if denominator != 0.0:
            return index + 0.5 * (left - right) / denominator
    return float(index)


def match_window(
    region: np.ndarray, window: np.ndarray, *, exclusion_px: int
) -> tuple[float, float, float, float, bool]:
    """Locate a window in a larger region by normalised cross-correlation.

    Args:
        region: Search region (``window`` plus the search margin on every side).
        window: Template.
        exclusion_px: Half-side of the square around the peak that the second peak
            ignores.

    Returns:
        ``(peak, second_peak, x, y, on_border)``: the correlation maximum, the
        highest value outside the exclusion square (``-1`` when nothing is left),
        the sub-pixel position of the maximum as the template's top-left offset in
        the region, and whether the integer maximum lies on the border of the
        correlation map (the true shift may exceed the search).
    """
    scores = cv2.matchTemplate(region, window, cv2.TM_CCOEFF_NORMED)
    return summarise_scores(scores, exclusion_px=exclusion_px)


def summarise_scores(
    scores: np.ndarray, *, exclusion_px: int
) -> tuple[float, float, float, float, bool]:
    """Peak, second peak, sub-pixel position and border flag of a score map.

    Args:
        scores: 2-D correlation map.
        exclusion_px: Half-side of the square around the peak that the second peak
            ignores.

    Returns:
        ``(peak, second_peak, x, y, on_border)`` as in :func:`match_window`.
    """
    _, peak, _, (peak_x, peak_y) = cv2.minMaxLoc(scores)
    others = scores.copy()
    others[
        max(0, peak_y - exclusion_px) : peak_y + exclusion_px + 1,
        max(0, peak_x - exclusion_px) : peak_x + exclusion_px + 1,
    ] = -1.0
    height, width = scores.shape
    on_border = peak_x in (0, width - 1) or peak_y in (0, height - 1)
    return (
        float(peak),
        float(others.max()),
        parabolic_peak(scores[peak_y, :], peak_x),
        parabolic_peak(scores[:, peak_x], peak_y),
        bool(on_border),
    )
