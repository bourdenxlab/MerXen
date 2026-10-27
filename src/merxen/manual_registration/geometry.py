"""Coarse transform, crop and frame maths of manual landmark registration.

numpy only; the napari viewer mirrors these functions and the golden vectors
(``tests/test_data/manual_registration/golden_v1.json``) pin both copies.

Conventions (see ``docs/plans/manual-registration-plan.md``):

* A point is ``(x, y)`` (x = column, y = row) in level-0 pixels of the full,
  uncropped image it belongs to; an integer value is a pixel centre.
* Every matrix is 3x3 with last row ``[0, 0, 1]`` and maps moving to reference:
  ``p_ref = M @ p_mov``.
* ``rotation_matrix_xy(theta)`` is ``[[cos, sin], [-sin, cos]]``; positive
  angles are counter-clockwise on screen (y down).
* The coarse matrix is ``T(c_r + offset / p_r) R(theta) F(mirror) S(s) T(-c_m)``
  with ``c`` the image centre ``((W - 1) / 2, (H - 1) / 2)``.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike

from merxen.manual_registration.schema import (
    CROP_MIN_SIZE_PX,
    CROP_SNAP_PX,
    CoarseTransform,
    CropRect,
    wrap_angle_deg,
)

# Converts (x, y) <-> napari (row, col).
SWAP: Final[np.ndarray] = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle to (-180, 180] degrees.

    Raises:
        ValueError: If the angle is not finite.
    """
    return wrap_angle_deg(angle_deg)


def rotation_matrix_xy(angle_deg: float) -> np.ndarray:
    """Return the 3x3 rotation ``R(theta)``; positive is counter-clockwise on screen."""
    radians = math.radians(angle_deg)
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array(
        [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def translation_xy(dx: float, dy: float) -> np.ndarray:
    """Return the 3x3 translation ``T(dx, dy)``."""
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 2] = dx
    matrix[1, 2] = dy
    return matrix


def _scale_matrix(scale: float) -> np.ndarray:
    return np.diag([scale, scale, 1.0]).astype(np.float64)


def _mirror_matrix(mirror_x: bool) -> np.ndarray:
    return np.diag([-1.0 if mirror_x else 1.0, 1.0, 1.0]).astype(np.float64)


def image_centre_xy(shape_yx: tuple[int, int]) -> tuple[float, float]:
    """Return the centre ``((W - 1) / 2, (H - 1) / 2)`` of an image of shape ``(H, W)``.

    Raises:
        ValueError: If a dimension is not positive.
    """
    height, width = shape_yx
    if height <= 0 or width <= 0:
        raise ValueError(f"image shape must be positive, got {tuple(shape_yx)}")
    return ((width - 1) / 2.0, (height - 1) / 2.0)


def apply_matrix_xy(matrix: np.ndarray, points_xy: ArrayLike) -> np.ndarray:
    """Apply a 3x3 matrix to ``(x, y)`` points of shape ``(2,)`` or ``(N, 2)``.

    Raises:
        ValueError: If the matrix is not 3x3 or the points are not 2-D points.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {mat.shape}")
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim not in (1, 2) or pts.shape[-1] != 2:
        raise ValueError(f"points must have shape (2,) or (N, 2), got {pts.shape}")
    return np.asarray(pts @ mat[:2, :2].T + mat[:2, 2], dtype=np.float64)


def similarity_parameters(matrix_xy: np.ndarray) -> tuple[float, float, bool]:
    """Decompose the linear part of a similarity matrix.

    Returns:
        ``(scale, rotation_deg, reflected)`` with ``scale = sqrt(|det|)``,
        the rotation applied after the mirror, wrapped to (-180, 180], and
        ``reflected = det < 0``. Exact for similarity matrices.

    Raises:
        ValueError: If the matrix is singular or not finite.
    """
    linear = np.asarray(matrix_xy, dtype=np.float64)[:2, :2]
    det = float(np.linalg.det(linear))
    if det == 0.0 or not math.isfinite(det):
        raise ValueError("matrix is singular")
    reflected = det < 0.0
    scale = math.sqrt(abs(det))
    unit = linear / scale
    rotation = unit @ np.diag([-1.0, 1.0]) if reflected else unit
    angle = wrap_deg(math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])))
    return scale, angle, reflected


def _positive(name: str, value: float) -> None:
    if not value > 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def coarse_matrix_xy(
    coarse: CoarseTransform,
    *,
    moving_shape_yx: tuple[int, int],
    reference_shape_yx: tuple[int, int],
    moving_pixel_size_um: float,
    reference_pixel_size_um: float,
) -> np.ndarray:
    """Build the moving-to-reference matrix of a coarse transform state.

    ``M = T(c_r + offset / p_r) R(theta) F(mirror) S(p_m / p_r) T(-c_m)`` with
    ``c_m`` and ``c_r`` the centres of the full moving and reference images.
    """
    _positive("moving_pixel_size_um", moving_pixel_size_um)
    _positive("reference_pixel_size_um", reference_pixel_size_um)
    scale = moving_pixel_size_um / reference_pixel_size_um
    c_ref = image_centre_xy(reference_shape_yx)
    c_mov = image_centre_xy(moving_shape_yx)
    offset_x, offset_y = coarse.offset_um
    return np.asarray(
        translation_xy(
            c_ref[0] + offset_x / reference_pixel_size_um,
            c_ref[1] + offset_y / reference_pixel_size_um,
        )
        @ rotation_matrix_xy(coarse.rotation_deg)
        @ _mirror_matrix(coarse.mirror_x)
        @ _scale_matrix(scale)
        @ translation_xy(-c_mov[0], -c_mov[1]),
        dtype=np.float64,
    )


def coarse_from_matrix(
    matrix_xy: np.ndarray,
    *,
    moving_shape_yx: tuple[int, int],
    reference_shape_yx: tuple[int, int],
    moving_pixel_size_um: float,
    reference_pixel_size_um: float,
) -> tuple[CoarseTransform, float]:
    """Recover the coarse state of a similarity matrix.

    Returns:
        ``(coarse, scale_ratio)``. ``scale_ratio`` is the matrix scale divided
        by ``p_m / p_r`` (informative: the scale is not part of the coarse state).
    """
    _positive("moving_pixel_size_um", moving_pixel_size_um)
    _positive("reference_pixel_size_um", reference_pixel_size_um)
    mat = np.asarray(matrix_xy, dtype=np.float64)
    scale, angle, mirror = similarity_parameters(mat)
    c_ref = image_centre_xy(reference_shape_yx)
    c_mov = image_centre_xy(moving_shape_yx)
    mapped = apply_matrix_xy(mat, c_mov)
    offset = (
        float((mapped[0] - c_ref[0]) * reference_pixel_size_um),
        float((mapped[1] - c_ref[1]) * reference_pixel_size_um),
    )
    coarse = CoarseTransform(rotation_deg=angle, mirror_x=mirror, offset_um=offset)
    return coarse, scale / (moving_pixel_size_um / reference_pixel_size_um)


def flipped_horizontal(coarse: CoarseTransform) -> CoarseTransform:
    """Mirror the displayed moving image about the vertical axis (its centre)."""
    return CoarseTransform(
        rotation_deg=wrap_deg(-coarse.rotation_deg),
        mirror_x=not coarse.mirror_x,
        offset_um=coarse.offset_um,
    )


def flipped_vertical(coarse: CoarseTransform) -> CoarseTransform:
    """Mirror the displayed moving image about the horizontal axis (its centre)."""
    return CoarseTransform(
        rotation_deg=wrap_deg(180.0 - coarse.rotation_deg),
        mirror_x=not coarse.mirror_x,
        offset_um=coarse.offset_um,
    )


def rotated(coarse: CoarseTransform, delta_deg: float) -> CoarseTransform:
    """Rotate the displayed moving image by ``delta_deg`` about its centre."""
    return CoarseTransform(
        rotation_deg=wrap_deg(coarse.rotation_deg + delta_deg),
        mirror_x=coarse.mirror_x,
        offset_um=coarse.offset_um,
    )


def shifted(coarse: CoarseTransform, dx_um: float, dy_um: float) -> CoarseTransform:
    """Shift the displayed moving image by ``(dx_um, dy_um)`` reference micrometres."""
    return CoarseTransform(
        rotation_deg=coarse.rotation_deg,
        mirror_x=coarse.mirror_x,
        offset_um=(coarse.offset_um[0] + dx_um, coarse.offset_um[1] + dy_um),
    )


def snap_crop(
    crop: CropRect | None,
    shape_yx: tuple[int, int],
    *,
    multiple: int = CROP_SNAP_PX,
    min_size_px: int = CROP_MIN_SIZE_PX,
) -> CropRect:
    """Snap a crop inward to multiples of ``multiple`` pixels.

    The origin is rounded up and the far edge down, except an edge that lies on
    the image border, which stays there. ``None`` gives the full image (no size
    check: a full image cannot be enlarged).

    Raises:
        ValueError: If ``multiple < 1``, the crop exceeds the image or the snapped
            crop is smaller than ``min_size_px`` in either direction.
    """
    if multiple < 1:
        raise ValueError(f"multiple must be at least 1, got {multiple}")
    height, width = shape_yx
    if crop is None:
        return CropRect(x0=0, y0=0, x1=width, y1=height)
    crop.validate_within((height, width))
    x0 = -(-crop.x0 // multiple) * multiple
    y0 = -(-crop.y0 // multiple) * multiple
    x1 = width if crop.x1 == width else (crop.x1 // multiple) * multiple
    y1 = height if crop.y1 == height else (crop.y1 // multiple) * multiple
    if x1 - x0 < min_size_px or y1 - y0 < min_size_px:
        raise ValueError(
            f"crop {crop.as_tuple()} snapped inward to multiples of {multiple} px "
            f"is {max(x1 - x0, 0)} x {max(y1 - y0, 0)} px, below the minimum "
            f"{min_size_px} px; enlarge the crop"
        )
    return CropRect(x0=x0, y0=y0, x1=x1, y1=y1)


def crop_matrix(
    matrix_full: np.ndarray,
    *,
    reference_origin_xy: tuple[float, float],
    moving_origin_xy: tuple[float, float],
) -> np.ndarray:
    """Express a full-frame matrix in the frames of the cropped images.

    ``M_crop = T(-o_r) M_full T(o_m)`` with ``o`` the effective crop origins.
    """
    return np.asarray(
        translation_xy(-reference_origin_xy[0], -reference_origin_xy[1])
        @ np.asarray(matrix_full, dtype=np.float64)
        @ translation_xy(moving_origin_xy[0], moving_origin_xy[1]),
        dtype=np.float64,
    )


def full_matrix(
    matrix_crop: np.ndarray,
    *,
    reference_origin_xy: tuple[float, float],
    moving_origin_xy: tuple[float, float],
) -> np.ndarray:
    """Inverse of ``crop_matrix``: ``M_full = T(o_r) M_crop T(-o_m)``."""
    return np.asarray(
        translation_xy(reference_origin_xy[0], reference_origin_xy[1])
        @ np.asarray(matrix_crop, dtype=np.float64)
        @ translation_xy(-moving_origin_xy[0], -moving_origin_xy[1]),
        dtype=np.float64,
    )


def points_outside_crop(points_xy: ArrayLike, crop: CropRect | None) -> np.ndarray:
    """Return a boolean ``(N,)`` mask of points outside a crop (``None``: none).

    A point is inside when ``x0 - 0.5 <= x < x1 - 0.5`` and likewise for y.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if crop is None:
        return np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    inside = (
        (x >= crop.x0 - 0.5)
        & (x < crop.x1 - 0.5)
        & (y >= crop.y0 - 0.5)
        & (y < crop.y1 - 0.5)
    )
    return np.asarray(~inside, dtype=bool)


def napari_affine_rc(
    reference_base_rc: np.ndarray, matrix_xy: np.ndarray
) -> np.ndarray:
    """Return the napari ``(row, col)`` affine of a moving layer.

    ``A_mov = A_ref @ SWAP @ M @ SWAP`` where ``A_ref`` is the reference layer's
    base affine (scale and translate) in napari ``(row, col)`` order.
    """
    return np.asarray(
        np.asarray(reference_base_rc, dtype=np.float64)
        @ SWAP
        @ np.asarray(matrix_xy, dtype=np.float64)
        @ SWAP,
        dtype=np.float64,
    )
