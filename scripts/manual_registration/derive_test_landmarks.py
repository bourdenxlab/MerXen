#!/usr/bin/env python
"""Derive test landmarks for a prepared manual-registration store, without a human.

The real-data validation of the manual landmark registration needs a labelling
for pairs that have no ground truth (a Zeiss CZI pair: a serial section and a
mislabelled Xenium section). This script finds the coarse rotation, the mirror
state and a robust similarity between the two registration channels of a prepared
store (tissue-mask search, then SIFT matches gated by the coarse result and a
RANSAC similarity fit), picks landmark pairs spread over the tissue and writes
them as a JSON file outside the repository. With ``--inject`` the labelling is
written into the store the way the viewer's save does.

The derived pairs are a test input of the workflow, never a result: nothing here
is used by the pipeline.

Example::

    runs="$MERXEN_EVIDENCE_ROOT/manual_registration/runs"
    python scripts/manual_registration/derive_test_landmarks.py \\
        --store "$runs/out/LCM1/prepared/LCM1_prepared.zarr" \\
        --out "$runs/LCM1_test_landmarks.json" --inject

``--inject`` never replaces a labelling that was placed by hand in the viewer
(any labelling whose ``written_by.software`` is not the simulate module's): it
stops with a message. ``--inject --force`` replaces it after saving the old one
as ``<out stem>.previous_labelling.<UTC>.json`` next to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk

from merxen import local_config
from merxen.image_registration import WorkingImage, read_working_channel
from merxen.io.ome_tiff import open_ome_tiff_level, read_ome_tiff_info
from merxen.manual_registration.geometry import (
    coarse_from_matrix,
    points_outside_crop,
    snap_crop,
    wrap_deg,
)
from merxen.manual_registration.schema import (
    LABELLING_ATTR,
    STORE_ATTR,
    CoarseTransform,
    CropRect,
    LandmarkPair,
    labelling_content_sha256,
    pair_colour,
    read_root_attrs,
)
from merxen.manual_registration.simulate import inject_labelling

logger = logging.getLogger(__name__)

SCHEMA = "merxen.manual_registration.test_landmarks"
MAX_WORKING_SIDE_PX = 8192
RATIO_TEST = 0.75
ROLES = ("reference", "moving")
# ``written_by.software`` of the labellings ``inject_labelling`` writes.
SIMULATE_SOFTWARE = "merxen.manual_registration.simulate"


@dataclass(frozen=True)
class DerivedLandmarks:
    """The derived alignment of a prepared store.

    Attributes:
        matrix_full: 3x3 similarity from moving level-0 px to reference level-0 px
            (maybe reflected).
        coarse: Coarse state equivalent to ``matrix_full``.
        scale_ratio: Matrix scale over the pixel-size ratio.
        crops: Effective-crop proposals per role (``None``: no crop).
        pairs: Landmark pairs (ids ``1..n``, every third one disabled).
        diagnostics: Search and SIFT numbers (JSON-able).
    """

    matrix_full: np.ndarray
    coarse: CoarseTransform
    scale_ratio: float
    crops: dict[str, CropRect | None]
    pairs: list[LandmarkPair]
    diagnostics: dict[str, Any]


# ---------------------------------------------------------------------------
# Matrices (3 x 3 homogeneous, (x, y) column vectors; MD 2.1-2.2 conventions)
# ---------------------------------------------------------------------------


def _translation(x: float, y: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])


def _rotation(theta_deg: float) -> np.ndarray:
    """Rotation, positive = counter-clockwise on screen (y down)."""
    theta = math.radians(theta_deg)
    cos, sin = math.cos(theta), math.sin(theta)
    return np.array([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]])


def _mirror(mirror_x: bool) -> np.ndarray:
    return np.diag([-1.0 if mirror_x else 1.0, 1.0, 1.0])


def _scale(scale: float) -> np.ndarray:
    return np.diag([scale, scale, 1.0])


def _apply(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    return pts @ matrix[:2, :2].T + matrix[:2, 2]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_working_dapi(
    store: Path, role: str, *, pixel_size_um: float
) -> tuple[WorkingImage, float]:
    """Read the registration channel of one role near a working pixel size.

    Args:
        store: A prepared store.
        role: ``"reference"`` or ``"moving"``.
        pixel_size_um: Wanted working pixel size.

    Returns:
        ``(working image, level-0 pixel size of the role)``; the working image's
        ``to_level0`` maps working px to level-0 px of the full image.

    Raises:
        ValueError: If the store is not a prepared store or the registration
            channel is not in the OME-TIFF.
    """
    attrs = read_root_attrs(Path(store)).get(STORE_ATTR)
    if not attrs or attrs.get("kind") != "prepared":
        raise ValueError(f"{store} is not a prepared manual-registration store")
    record = attrs["images"][role]
    info = read_ome_tiff_info(record["ome_tiff"]["path"])
    names = list(record.get("canonical_channel_names") or info.channel_names)
    channel = record["registration_channel"]
    if channel not in names:
        raise ValueError(f"registration channel {channel!r} not in {names}")
    pixel_size = float(record["pixel_size_um"])
    working = read_working_channel(
        open_level=lambda level: open_ome_tiff_level(info, level),
        level_shapes_yx=[(shape[1], shape[2]) for shape in info.level_shapes_cyx],
        channel_index=names.index(channel),
        target_factor=pixel_size_um / pixel_size,
        level_factors_xy=None,
    )
    return working, pixel_size


def _working_pixel_size(working: WorkingImage, pixel_size_um: float) -> float:
    """Working pixel size of an image (level-0 size times the matrix scale)."""
    return pixel_size_um * math.sqrt(abs(np.linalg.det(working.to_level0[:2, :2])))


def tissue_mask(
    pixels: np.ndarray, *, sigma_px: float = 2.0, min_component_fraction: float = 0.05
) -> np.ndarray:
    """Tissue mask of a working image: smoothed log image above its Otsu level.

    Args:
        pixels: 2-D working image.
        sigma_px: Gaussian sigma in working pixels.
        min_component_fraction: Components smaller than this fraction of the largest
            one are dropped.

    Returns:
        Boolean mask.

    Raises:
        ValueError: If the image has no positive pixels or no tissue is found.
    """
    log_image = np.log1p(np.clip(np.asarray(pixels, dtype=np.float32), 0.0, None))
    smoothed = gaussian(log_image, sigma=sigma_px, preserve_range=True)
    positive = smoothed[smoothed > 0]
    if positive.size == 0 or float(positive.max()) == float(positive.min()):
        raise ValueError("the working image has no signal to find tissue in")
    mask = smoothed > threshold_otsu(positive)
    mask = ndimage.binary_opening(mask, structure=disk(2))
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("no tissue found in the working image")
    areas = np.bincount(labels.ravel())[1:]
    keep = np.nonzero(areas >= min_component_fraction * areas.max())[0] + 1
    return np.asarray(np.isin(labels, keep), dtype=bool)


def _centroid_xy(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return np.array([cols.mean(), rows.mean()])


# ---------------------------------------------------------------------------
# Coarse search
# ---------------------------------------------------------------------------


def _coarse_matrix(
    theta_deg: float,
    mirror_x: bool,
    scale: float,
    mu_moving: np.ndarray,
    target_xy: np.ndarray,
) -> np.ndarray:
    return (
        _translation(*target_xy)
        @ _rotation(theta_deg)
        @ _mirror(mirror_x)
        @ _scale(scale)
        @ _translation(*(-mu_moving))
    )


def _score_candidate(
    ref_mask: np.ndarray,
    mov_u8: np.ndarray,
    theta_deg: float,
    mirror_x: bool,
    scale: float,
    mu_moving: np.ndarray,
    mu_reference: np.ndarray,
) -> tuple[float, np.ndarray]:
    """IoU and matrix of one rotation/mirror candidate after the best shift."""
    height, width = ref_mask.shape
    size = (width, height)
    start = _coarse_matrix(theta_deg, mirror_x, scale, mu_moving, mu_reference)
    warped = cv2.warpAffine(mov_u8, start[:2], size, flags=cv2.INTER_NEAREST)
    if not warped.any():
        return 0.0, start
    reference_f = ref_mask.astype(np.float32)
    warped_f = (warped > 0).astype(np.float32)
    correlation = fftconvolve(reference_f, warped_f[::-1, ::-1], mode="full")
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift = np.array([peak_x - (width - 1), peak_y - (height - 1)], dtype=np.float64)
    matrix = _coarse_matrix(theta_deg, mirror_x, scale, mu_moving, mu_reference + shift)
    shifted = cv2.warpAffine(mov_u8, matrix[:2], size, flags=cv2.INTER_NEAREST) > 0
    union = np.logical_or(ref_mask, shifted).sum()
    iou = float(np.logical_and(ref_mask, shifted).sum() / union) if union else 0.0
    return iou, matrix


def coarse_search(
    ref_mask: np.ndarray,
    mov_mask: np.ndarray,
    *,
    scale: float,
    step_deg: float = 15.0,
    fine_step_deg: float = 1.0,
    fine_radius_deg: float = 15.0,
) -> dict[str, Any]:
    """Find rotation, mirror and shift that overlay the moving mask on the reference.

    Both mirror states and every rotation in ``step_deg`` steps are scored by the
    IoU of the masks after the best integer shift (FFT cross-correlation); the best
    candidate is refined in ``fine_step_deg`` steps within ``fine_radius_deg``.

    Args:
        ref_mask: Reference tissue mask (working px).
        mov_mask: Moving tissue mask (working px).
        scale: Moving working pixel size over the reference's.
        step_deg: Coarse rotation step.
        fine_step_deg: Fine rotation step.
        fine_radius_deg: Fine search radius around the best coarse angle.

    Returns:
        ``rotation_deg``, ``mirror_x``, ``translation`` (reference working px),
        ``matrix_working`` (moving working px to reference working px), ``iou``,
        ``runner_up_iou`` (best IoU of the other mirror state) and the best IoU of
        each state in ``iou_by_mirror``.
    """
    mu_reference = _centroid_xy(ref_mask)
    mu_moving = _centroid_xy(mov_mask)
    mov_u8 = mov_mask.astype(np.uint8)
    angles = np.arange(0.0, 360.0, step_deg)
    best: dict[bool, tuple[float, float, np.ndarray]] = {}
    for mirror_x in (False, True):
        results = [
            (
                *_score_candidate(
                    ref_mask,
                    mov_u8,
                    float(angle),
                    mirror_x,
                    scale,
                    mu_moving,
                    mu_reference,
                ),
                float(angle),
            )
            for angle in angles
        ]
        iou, matrix, angle = max(results, key=lambda item: item[0])
        best[mirror_x] = (iou, angle, matrix)
    chosen = max(best, key=lambda key: best[key][0])
    best_iou, centre, best_matrix = best[chosen]
    best_angle = centre
    n_steps = int(round(fine_radius_deg / fine_step_deg))
    for step in range(-n_steps, n_steps + 1):
        angle = centre + step * fine_step_deg
        candidate_iou, candidate = _score_candidate(
            ref_mask, mov_u8, angle, chosen, scale, mu_moving, mu_reference
        )
        if candidate_iou > best_iou:
            best_iou, best_angle, best_matrix = candidate_iou, angle, candidate
    iou, final_angle, matrix = best_iou, best_angle, best_matrix
    return {
        "rotation_deg": float(wrap_deg(final_angle)),
        "mirror_x": bool(chosen),
        "translation": [float(matrix[0, 2]), float(matrix[1, 2])],
        "matrix_working": matrix,
        "iou": float(iou),
        "runner_up_iou": float(best[not chosen][0]),
        "iou_by_mirror": {
            "false": float(best[False][0]),
            "true": float(best[True][0]),
        },
    }


# ---------------------------------------------------------------------------
# SIFT + RANSAC
# ---------------------------------------------------------------------------


def _to_uint8(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Log image clipped to its 0.5/99.5 percentiles inside tissue, CLAHE-equalised."""
    log_image = np.log1p(np.clip(np.asarray(pixels, dtype=np.float32), 0.0, None))
    inside = log_image[mask] if mask.any() else log_image.ravel()
    low, high = np.percentile(inside, [0.5, 99.5])
    scaled = np.clip((log_image - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    image = np.round(scaled * 255.0).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def _detect(
    image: np.ndarray, mask: np.ndarray
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    eroded = ndimage.binary_erosion(mask, structure=disk(3))
    sift = cv2.SIFT_create(nfeatures=20000)
    return sift.detectAndCompute(image, mask=eroded.astype(np.uint8) * 255)


def sift_similarity(
    ref_img: np.ndarray,
    mov_img: np.ndarray,
    ref_mask: np.ndarray,
    mov_mask: np.ndarray,
    *,
    coarse: dict[str, Any],
    mirror: bool,
    scale: float,
    gate_px: float,
    ransac_threshold_px: float,
    seed: int,
) -> dict[str, Any]:
    """Fit a similarity by SIFT matches gated by the coarse result and RANSAC.

    Args:
        ref_img: Reference working image (raw intensities).
        mov_img: Moving working image (raw intensities).
        ref_mask: Reference tissue mask.
        mov_mask: Moving tissue mask.
        coarse: Result of :func:`coarse_search` (its ``matrix_working`` gates).
        mirror: Whether the moving image is mirrored (it is flipped horizontally
            before matching, so the fitted similarity is proper).
        scale: Moving working pixel size over the reference's (informative).
        gate_px: Matches farther than this from the coarse prediction are dropped.
        ransac_threshold_px: RANSAC reprojection threshold.
        seed: Seed of OpenCV's random generator.

    Returns:
        ``matrix_working`` (moving working px, unflipped, to reference working px;
        ``None`` without a fit), ``n_inliers``, the inlier points ``moving_xy`` and
        ``reference_xy`` (moving in the unflipped frame) and the counts
        ``n_keypoints_reference``, ``n_keypoints_moving``, ``n_ratio_matches``,
        ``n_gated``.
    """
    del scale  # the coarse matrix already carries the scale
    cv2.setRNGSeed(int(seed))
    reference_u8 = _to_uint8(ref_img, ref_mask)
    moving_u8 = _to_uint8(mov_img, mov_mask)
    width = mov_img.shape[1]
    flip = np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    if mirror:
        moving_u8 = np.ascontiguousarray(moving_u8[:, ::-1])
        mov_mask = np.ascontiguousarray(mov_mask[:, ::-1])
    kp_ref, des_ref = _detect(reference_u8, ref_mask)
    kp_mov, des_mov = _detect(moving_u8, mov_mask)
    result: dict[str, Any] = {
        "matrix_working": None,
        "n_inliers": 0,
        "moving_xy": np.zeros((0, 2)),
        "reference_xy": np.zeros((0, 2)),
        "n_keypoints_reference": len(kp_ref),
        "n_keypoints_moving": len(kp_mov),
        "n_ratio_matches": 0,
        "n_gated": 0,
    }
    if des_ref is None or des_mov is None or len(kp_ref) < 3 or len(kp_mov) < 3:
        return result
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_mov, des_ref, k=2)
    good = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance
    ]
    result["n_ratio_matches"] = len(good)
    if not good:
        return result
    moving_flipped = np.array([kp_mov[m.queryIdx].pt for m in good], dtype=np.float64)
    reference = np.array([kp_ref[m.trainIdx].pt for m in good], dtype=np.float64)
    moving_unflipped = _apply(flip, moving_flipped) if mirror else moving_flipped
    predicted = _apply(np.asarray(coarse["matrix_working"]), moving_unflipped)
    keep = np.linalg.norm(predicted - reference, axis=1) <= gate_px
    result["n_gated"] = int(keep.sum())
    if keep.sum() < 3:
        return result
    transform, inliers = cv2.estimateAffinePartial2D(
        moving_flipped[keep].astype(np.float32),
        reference[keep].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_px),
        maxIters=20000,
        confidence=0.999,
        refineIters=10,
    )
    if transform is None or inliers is None:
        return result
    proper = np.vstack([transform, [0.0, 0.0, 1.0]])
    matrix = proper @ flip if mirror else proper
    flags = inliers.ravel().astype(bool)
    result.update(
        {
            "matrix_working": matrix,
            "n_inliers": int(flags.sum()),
            "moving_xy": moving_unflipped[keep][flags],
            "reference_xy": reference[keep][flags],
        }
    )
    return result


# ---------------------------------------------------------------------------
# Pair selection and crops
# ---------------------------------------------------------------------------


def farthest_point_order(points_xy: np.ndarray) -> np.ndarray:
    """Order points so that every prefix is spread out (farthest-point sampling).

    Starts at the point nearest the centroid, then repeatedly adds the point whose
    distance to the chosen set is largest (ties: lowest index).

    Args:
        points_xy: ``(N, 2)`` points.

    Returns:
        ``(N,)`` indices.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(0, dtype=int)
    first = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    order = [first]
    nearest = np.linalg.norm(pts - pts[first], axis=1)
    nearest[first] = -1.0
    while len(order) < len(pts):
        index = int(np.argmax(nearest))
        order.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(pts - pts[index], axis=1))
        nearest[order] = -1.0
    return np.asarray(order, dtype=int)


def _tissue_crop(
    mask: np.ndarray, to_level0: np.ndarray, shape_yx: tuple[int, int]
) -> CropRect | None:
    """Crop proposal: the tissue bounding box, inset by 1 %, snapped inward."""
    rows, cols = np.nonzero(mask)
    x0w, x1w = cols.min() - 0.5, cols.max() + 0.5
    y0w, y1w = rows.min() - 0.5, rows.max() + 0.5
    corners = _apply(
        to_level0, np.array([[x0w, y0w], [x1w, y0w], [x1w, y1w], [x0w, y1w]])
    )
    height, width = shape_yx
    left = math.floor(corners[:, 0].min())
    right = math.ceil(corners[:, 0].max())
    top = math.floor(corners[:, 1].min())
    bottom = math.ceil(corners[:, 1].max())
    inset_x = 0.01 * (right - left)
    inset_y = 0.01 * (bottom - top)
    x0 = max(0, math.ceil(left + inset_x))
    x1 = min(width, math.floor(right - inset_x))
    y0 = max(0, math.ceil(top + inset_y))
    y1 = min(height, math.floor(bottom - inset_y))
    try:
        rect = CropRect(x0=x0, y0=y0, x1=x1, y1=y1)
        snap_crop(rect, shape_yx)
    except ValueError as exc:
        logger.warning("dropping the crop proposal: %s", exc)
        return None
    return rect


def _disabled_ids(n_pairs: int, n_disabled: int) -> set[int]:
    return set(list(range(3, n_pairs + 1, 3))[:n_disabled])


def _estimate(
    store: Path,
    *,
    working_pixel_size_um: float,
    seed: int,
) -> dict[str, Any]:
    """Coarse search and SIFT fit at one working pixel size."""
    ref, p_ref = load_working_dapi(
        store, "reference", pixel_size_um=working_pixel_size_um
    )
    mov, p_mov = load_working_dapi(store, "moving", pixel_size_um=working_pixel_size_um)
    q_ref = _working_pixel_size(ref, p_ref)
    q_mov = _working_pixel_size(mov, p_mov)
    scale = q_mov / q_ref
    ref_mask = tissue_mask(ref.pixels)
    mov_mask = tissue_mask(mov.pixels)
    search = coarse_search(ref_mask, mov_mask, scale=scale)
    logger.info(
        "coarse search at %.1f um: rotation %.1f deg, mirror %s, IoU %.3f "
        "(other mirror state %.3f)",
        q_ref,
        search["rotation_deg"],
        search["mirror_x"],
        search["iou"],
        search["runner_up_iou"],
    )
    fit = sift_similarity(
        ref.pixels,
        mov.pixels,
        ref_mask,
        mov_mask,
        coarse=search,
        mirror=bool(search["mirror_x"]),
        scale=scale,
        gate_px=500.0 / q_ref,
        ransac_threshold_px=max(2.0, 30.0 / q_ref),
        seed=seed,
    )
    logger.info(
        "SIFT: %d/%d keypoints, %d ratio matches, %d gated, %d inliers",
        fit["n_keypoints_reference"],
        fit["n_keypoints_moving"],
        fit["n_ratio_matches"],
        fit["n_gated"],
        fit["n_inliers"],
    )
    return {
        "ref": ref,
        "mov": mov,
        "p_ref": p_ref,
        "p_mov": p_mov,
        "q_ref": q_ref,
        "scale": scale,
        "ref_mask": ref_mask,
        "mov_mask": mov_mask,
        "search": search,
        "fit": fit,
    }


def derive(
    store: Path,
    *,
    n_pairs: int = 24,
    n_disabled: int = 8,
    working_pixel_size_um: float = 10.0,
    min_inliers: int = 24,
    seed: int = 0,
) -> DerivedLandmarks:
    """Derive an alignment, crops and landmark pairs for a prepared store.

    Args:
        store: A prepared store.
        n_pairs: Pairs to derive (ids ``1..n_pairs``, farthest-point order).
        n_disabled: Pairs to disable (ids 3, 6, 9, ...), as the held-out set.
        working_pixel_size_um: Working pixel size of the search.
        min_inliers: RANSAC inliers needed; fewer retries once at half the pixel size.
        seed: Seed of the RANSAC random generator.

    Returns:
        The derived alignment.

    Raises:
        RuntimeError: If no usable alignment is found or fewer than ``n_pairs``
            pairs lie inside both crops.
    """
    store = Path(store)
    attrs = read_root_attrs(store)[STORE_ATTR]
    shapes = {role: tuple(attrs["images"][role]["shape_yx"]) for role in ROLES}
    used_pixel_size = working_pixel_size_um
    state = _estimate(
        store,
        working_pixel_size_um=used_pixel_size,
        seed=seed,
    )
    retried_at: float | None = None
    if state["fit"]["n_inliers"] < min_inliers:
        longest_um = max(
            max(shapes[role]) * float(attrs["images"][role]["pixel_size_um"])
            for role in ROLES
        )
        finer = max(working_pixel_size_um / 2.0, longest_um / MAX_WORKING_SIDE_PX)
        if finer < working_pixel_size_um:
            logger.info(
                "only %d inliers; retrying at %.2f um", state["fit"]["n_inliers"], finer
            )
            used_pixel_size = retried_at = finer
            state = _estimate(
                store,
                working_pixel_size_um=finer,
                seed=seed,
            )
    fit = state["fit"]
    if fit["n_inliers"] < min_inliers or fit["matrix_working"] is None:
        raise RuntimeError(
            f"only {fit['n_inliers']} RANSAC inliers (keypoints ref/mov "
            f"{fit['n_keypoints_reference']}/{fit['n_keypoints_moving']}, ratio "
            f"matches {fit['n_ratio_matches']}, gated {fit['n_gated']}); place "
            "landmarks by hand in the viewer"
        )
    w_ref = state["ref"].to_level0
    w_mov = state["mov"].to_level0
    matrix_full = w_ref @ np.asarray(fit["matrix_working"]) @ np.linalg.inv(w_mov)
    mirror = bool(state["search"]["mirror_x"])
    if (np.linalg.det(matrix_full[:2, :2]) < 0) != mirror:
        raise RuntimeError(
            "the fitted similarity disagrees with the searched mirror state"
        )
    crops: dict[str, CropRect | None] = {
        "reference": _tissue_crop(state["ref_mask"], w_ref, shapes["reference"]),
        "moving": _tissue_crop(state["mov_mask"], w_mov, shapes["moving"]),
    }
    moving_pts = _apply(w_mov, fit["moving_xy"])
    reference_pts = _apply(w_ref, fit["reference_xy"])
    inside = ~(
        points_outside_crop(moving_pts, crops["moving"])
        | points_outside_crop(reference_pts, crops["reference"])
    )
    moving_pts, reference_pts = moving_pts[inside], reference_pts[inside]
    if len(moving_pts) < n_pairs:
        raise RuntimeError(
            f"only {len(moving_pts)} inlier matches lie inside both crops, "
            f"{n_pairs} are needed; place landmarks by hand in the viewer"
        )
    order = farthest_point_order(moving_pts)[:n_pairs]
    disabled = _disabled_ids(n_pairs, n_disabled)
    pairs = [
        LandmarkPair(
            id=number,
            colour=pair_colour(number, 0),
            enabled=number not in disabled,
            reference_xy_px=(
                round(float(reference_pts[index, 0]), 3),
                round(float(reference_pts[index, 1]), 3),
            ),
            moving_xy_px=(
                round(float(moving_pts[index, 0]), 3),
                round(float(moving_pts[index, 1]), 3),
            ),
        )
        for number, index in enumerate(order, start=1)
    ]
    coarse, scale_ratio = coarse_from_matrix(
        matrix_full,
        moving_shape_yx=shapes["moving"],
        reference_shape_yx=shapes["reference"],
        moving_pixel_size_um=state["p_mov"],
        reference_pixel_size_um=state["p_ref"],
    )
    search = state["search"]
    diagnostics = {
        "working_pixel_size_um": {
            "reference": state["q_ref"],
            "moving": state["q_ref"] * state["scale"],
        },
        "coarse_search": {
            key: search[key]
            for key in (
                "rotation_deg",
                "mirror_x",
                "translation",
                "iou",
                "runner_up_iou",
                "iou_by_mirror",
            )
        },
        "sift": {
            "n_keypoints_reference": fit["n_keypoints_reference"],
            "n_keypoints_moving": fit["n_keypoints_moving"],
            "n_ratio_matches": fit["n_ratio_matches"],
            "n_gated": fit["n_gated"],
            "n_inliers": fit["n_inliers"],
            "ransac_threshold_um": max(2.0, 30.0 / state["q_ref"]) * state["q_ref"],
            "retried_at_um": retried_at,
        },
        "requested_working_pixel_size_um": working_pixel_size_um,
        "used_working_pixel_size_um": used_pixel_size,
    }
    return DerivedLandmarks(
        matrix_full=matrix_full,
        coarse=coarse,
        scale_ratio=float(scale_ratio),
        crops=crops,
        pairs=pairs,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _output_document(store: Path, derived: DerivedLandmarks) -> dict[str, Any]:
    attrs = read_root_attrs(store)[STORE_ATTR]
    diagnostics = derived.diagnostics
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "store_grid_digest": attrs.get("grid_digest"),
        "working_pixel_size_um": diagnostics["working_pixel_size_um"],
        "coarse_search": diagnostics["coarse_search"],
        "sift": diagnostics["sift"],
        "matrix_full": derived.matrix_full.tolist(),
        "coarse": derived.coarse.model_dump(),
        "scale_ratio": derived.scale_ratio,
        "crops": {
            role: None if crop is None else crop.model_dump()
            for role, crop in derived.crops.items()
        },
        "pairs": [pair.model_dump() for pair in derived.pairs],
    }


class ForeignLabellingError(RuntimeError):
    """The store holds a labelling that this script did not write."""


def _planned_content_sha256(derived: DerivedLandmarks) -> str:
    return labelling_content_sha256(
        {
            "coarse": {
                "rotation_deg": round(derived.coarse.rotation_deg, 4),
                "mirror_x": derived.coarse.mirror_x,
                "offset_um": [
                    round(derived.coarse.offset_um[0], 4),
                    round(derived.coarse.offset_um[1], 4),
                ],
            },
            "crops": {
                role: None if crop is None else crop.model_dump()
                for role, crop in derived.crops.items()
            },
            "landmarks": {"pairs": [p.model_dump() for p in derived.pairs]},
        }
    )


def is_foreign_labelling(labelling: dict[str, Any] | None) -> bool:
    """Whether a stored labelling was not written by ``simulate`` (a person's own)."""
    if not labelling:
        return False
    written_by = labelling.get("written_by")
    software = written_by.get("software") if isinstance(written_by, dict) else None
    return software != SIMULATE_SOFTWARE


def _check_overwrite(store: Path, *, force: bool) -> dict[str, Any] | None:
    """Return the labelling that a forced injection would replace, or raise.

    Raises:
        ForeignLabellingError: When the store holds a labelling that was not
            written by ``simulate`` and ``force`` is false.
    """
    current = read_root_attrs(store).get(LABELLING_ATTR)
    if not is_foreign_labelling(current):
        return None
    assert current is not None
    if not force:
        landmarks = current.get("landmarks")
        pairs = landmarks.get("pairs", []) if isinstance(landmarks, dict) else []
        raise ForeignLabellingError(
            f"{store} holds a labelling that was not written by "
            f"{SIMULATE_SOFTWARE} (revision {current.get('revision')}, status "
            f"{current.get('status')!r}, {len(pairs)} pairs): it is not replaced; "
            "pass --force to replace it (the old labelling is saved next to --out "
            "first)"
        )
    return current


def _save_previous_labelling(previous: dict[str, Any], out: Path) -> Path:
    """Write the labelling about to be replaced next to ``out``; return the path."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = out.with_name(f"{out.stem}.previous_labelling.{stamp}.json")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    return backup


def _inject(
    store: Path, derived: DerivedLandmarks, out: Path, *, force: bool = False
) -> int | None:
    """Write the derived labelling into the store.

    The labelling is left alone when the store already holds this content. A
    labelling that ``simulate`` wrote may be replaced. Any other one (a person's
    landmarks from the viewer) is replaced only with ``force``, after it has been
    saved to ``<out stem>.previous_labelling.<UTC>.json`` next to ``out``.

    Args:
        store: The prepared store.
        derived: The derived labelling.
        out: The output JSON path (the backup goes next to it).
        force: Replace a labelling that ``simulate`` did not write.

    Returns:
        The new revision, or ``None`` when the store already held this labelling.

    Raises:
        ForeignLabellingError: On a labelling that ``simulate`` did not write
            when ``force`` is false.
    """
    current = read_root_attrs(store).get(LABELLING_ATTR)
    if current and current.get("status") == "finished":
        try:
            existing = labelling_content_sha256(current)
        except ValueError:
            existing = None
        if existing == _planned_content_sha256(derived):
            logger.info("the store already holds this labelling: unchanged")
            return None
    previous = _check_overwrite(store, force=force)
    if previous is not None:
        backup = _save_previous_labelling(previous, out)
        logger.warning(
            "replacing a labelling that was not written by %s; the old one is "
            "saved in %s",
            SIMULATE_SOFTWARE,
            backup,
        )
    revision = inject_labelling(
        store,
        derived.pairs,
        coarse=derived.coarse,
        crops=derived.crops,
        status="finished",
    )
    logger.info("labelling written to the store (revision %d)", revision)
    return revision


def _refuse_inside_repository(out: Path) -> None:
    root = local_config.checkout_root()
    resolved = out.resolve()
    if root is not None and resolved.is_relative_to(root.resolve()):
        raise SystemExit(
            f"--out must not be inside the repository checkout ({root}): the "
            "derived landmarks belong to private data"
        )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Derive test landmarks for a prepared manual-registration store."
    )
    parser.add_argument("--store", type=Path, required=True, help="Prepared store.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON (outside the repository).",
    )
    parser.add_argument("--n-pairs", type=int, default=24)
    parser.add_argument("--n-disabled", type=int, default=8)
    parser.add_argument("--working-pixel-size-um", type=float, default=10.0)
    parser.add_argument("--min-inliers", type=int, default=24)
    parser.add_argument(
        "--inject",
        action="store_true",
        help=(
            "Write the labelling (status finished) into --store. A labelling that "
            "this script did not write (landmarks placed in the viewer) is not "
            "replaced unless --force is given."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "With --inject: replace a labelling that was not written by "
            "merxen.manual_registration.simulate. The old labelling is saved first "
            "as <out stem>.previous_labelling.<UTC>.json next to --out."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the script.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        The exit status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.force and not args.inject:
        parser.error("--force needs --inject")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _refuse_inside_repository(args.out)
    if args.inject:
        # Fail before the derivation (minutes) when the injection would be refused.
        try:
            _check_overwrite(args.store, force=args.force)
        except ForeignLabellingError as exc:
            raise SystemExit(str(exc)) from exc
    derived = derive(
        args.store,
        n_pairs=args.n_pairs,
        n_disabled=args.n_disabled,
        working_pixel_size_um=args.working_pixel_size_um,
        min_inliers=args.min_inliers,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_output_document(args.store, derived), indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %s", args.out)
    if args.inject:
        try:
            _inject(args.store, derived, args.out, force=args.force)
        except ForeignLabellingError as exc:
            raise SystemExit(str(exc)) from exc
    print(
        f"rotation {derived.coarse.rotation_deg:.2f} deg, mirror "
        f"{derived.coarse.mirror_x}, scale ratio {derived.scale_ratio:.4f}, "
        f"{len(derived.pairs)} pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
