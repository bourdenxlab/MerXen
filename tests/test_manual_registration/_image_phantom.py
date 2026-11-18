"""A numpy image phantom for the image evidence tests (2.76 um/px, three channels).

The reference shows random vessel-like tubes and ellipses (widths 15-25 um) inside an
elliptical tissue outline, in three channels with different mixing weights, plus fine
spots and noise that are independent per image. The moving image is the reference
structure seen through a known warp (a 35 degree rotation about the image centre
at scale 1.02, plus a sinusoid of 40 um amplitude and 1.5 mm wavelength) with its
own fine spots and noise. Everything is deterministic in the seed.

Vessel profiles (MAP2- and AT8-dark lumens) can be drawn into one image only, so a
profile can sit elsewhere in the moving image than the warp puts it: an obliquely
cut vessel whose profile translates along its axis while the tissue around it
follows the warp.

Both images sit on a level-0 grid of 1.38 um with a working factor of 2 (contract
C2's frames). The reference crop starts at the full-frame origin; the moving crop
starts at ``MOVING_ORIGIN_LEVEL0_PX`` level-0 px, so a moving full-frame um is the
crop-local um plus ``MOVING_ORIGIN_UM`` and a wrong grid conversion is visible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from merxen.manual_registration.image_evidence import RoleImages, _um_to_grid

PIXEL_UM = 2.76
LEVEL0_PIXEL_UM = 1.38
LEVEL0_FACTOR = 2
MOVING_ORIGIN_LEVEL0_PX = (300.0, 220.0)
MOVING_ORIGIN_UM = np.array(MOVING_ORIGIN_LEVEL0_PX) * LEVEL0_PIXEL_UM
ROTATION_DEG = 35.0
SIZE_PX = 2048
_PAD_PX = 300
_ROLE_ORDER = ("at8", "map2", "dapi")
# Log-intensity offsets of a vessel lumen (at8, map2, dapi): no neuropil staining.
VESSEL_WEIGHTS = np.array([-1.2, -2.0, -0.3], dtype=np.float32)


@dataclass(frozen=True, eq=False)
class Phantom:
    """A reference/moving phantom pair with the known warp.

    Attributes:
        reference: Reference role images.
        moving: Moving role images.
        pull: Exact map of reference full-frame um to moving full-frame um.
        prior: The affine part of the warp (what a coarse alignment would know).
        tubes_um: Explicit tubes ``(x0, y0, x1, y1)`` in reference um.
    """

    reference: RoleImages
    moving: RoleImages
    pull: Callable[[np.ndarray], np.ndarray]
    prior: Callable[[np.ndarray], np.ndarray]
    tubes_um: tuple[tuple[float, float, float, float], ...]


def _to_level0(origin_level0_px: tuple[float, float]) -> np.ndarray:
    """Working px to level-0 px: a factor of 2 and the crop origin."""
    return np.array(
        [
            [LEVEL0_FACTOR, 0.0, origin_level0_px[0]],
            [0.0, LEVEL0_FACTOR, origin_level0_px[1]],
            [0.0, 0.0, 1.0],
        ]
    )


_Patch = tuple[int, int, int, int, np.ndarray]


def _segment_patch(
    size: int, start: np.ndarray, end: np.ndarray, sigma_px: float
) -> _Patch | None:
    """Gaussian-profile line segment (px coordinates): its box and values."""
    low = np.floor(np.minimum(start, end) - 4 * sigma_px).astype(int)
    high = np.ceil(np.maximum(start, end) + 4 * sigma_px).astype(int) + 1
    x0, y0 = max(int(low[0]), 0), max(int(low[1]), 0)
    x1, y1 = min(int(high[0]), size), min(int(high[1]), size)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    direction = end - start
    length_sq = float(direction @ direction)
    t = ((xx - start[0]) * direction[0] + (yy - start[1]) * direction[1]) / max(
        length_sq, 1e-9
    )
    t = np.clip(t, 0.0, 1.0)
    dist_sq = (xx - start[0] - t * direction[0]) ** 2 + (
        yy - start[1] - t * direction[1]
    ) ** 2
    return y0, y1, x0, x1, np.exp(-dist_sq / (2 * sigma_px**2)).astype(np.float32)


def _ellipse_patch(
    size: int, centre: np.ndarray, axes_px: tuple[float, float], angle: float
) -> _Patch | None:
    """A ring (a vessel in cross-section) with a hollow lumen: its box and values."""
    reach = int(math.ceil(max(axes_px) * 2.5))
    x0, x1 = max(int(centre[0]) - reach, 0), min(int(centre[0]) + reach + 1, size)
    y0, y1 = max(int(centre[1]) - reach, 0), min(int(centre[1]) + reach + 1, size)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - centre[0], yy - centre[1]
    u = (dx * math.cos(angle) + dy * math.sin(angle)) / axes_px[0]
    v = (-dx * math.sin(angle) + dy * math.cos(angle)) / axes_px[1]
    radius = np.sqrt(u * u + v * v)
    values = np.exp(-((radius - 1.0) ** 2) / 0.18) - 0.4 * np.exp(-(radius**2) / 0.3)
    return y0, y1, x0, x1, values.astype(np.float32)


def _structure(
    rng: np.random.Generator,
    n_tubes: int,
    n_ellipses: int,
    extra_tubes_um: Sequence[tuple[float, float, float, float, float]],
    clear_boxes_um: Sequence[tuple[float, float, float, float]],
    tissue_scale: float,
    texture_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-role structure canvases (log units) and the tissue map, canvas frame.

    Every feature has its own sign and its own mixing weights over the roles, so
    unrelated features correlate poorly, as they do in tissue.
    """
    size = SIZE_PX + 2 * _PAD_PX
    canvas = np.zeros((3, size, size), dtype=np.float32)

    def cleared(point_px: np.ndarray, reach_px: float) -> bool:
        """Whether a feature that reaches ``reach_px`` from its centre meets a box."""
        x_um = (point_px[0] - _PAD_PX) * PIXEL_UM
        y_um = (point_px[1] - _PAD_PX) * PIXEL_UM
        reach_um = reach_px * PIXEL_UM
        return any(
            x0 - reach_um <= x_um <= x1 + reach_um
            and y0 - reach_um <= y_um <= y1 + reach_um
            for x0, y0, x1, y1 in clear_boxes_um
        )

    def add(patch: _Patch | None, weights: np.ndarray) -> None:
        if patch is not None:
            y0, y1, x0, x1, values = patch
            canvas[:, y0:y1, x0:x1] += weights[:, None, None] * values[None]

    def draw_weights() -> np.ndarray:
        sign = rng.choice([-1.0, 1.0])
        weights = sign * rng.uniform(0.6, 1.2) * rng.uniform(0.3, 1.0, 3)
        # DAPI shows tissue structure weakly, so that it still outlines the tissue.
        weights[2] *= 0.4
        return weights.astype(np.float32)

    for _ in range(n_tubes):
        centre = rng.uniform(_PAD_PX, size - _PAD_PX, 2)
        angle = rng.uniform(0, math.pi)
        length_px = rng.uniform(60, 300)
        half = 0.5 * length_px * np.array([math.cos(angle), math.sin(angle)])
        sigma_px = rng.uniform(15, 25) / 2.355 / PIXEL_UM
        weights = draw_weights()
        if not cleared(centre, length_px / 2 + 4 * sigma_px):
            add(_segment_patch(size, centre - half, centre + half, sigma_px), weights)
    for _ in range(n_ellipses):
        centre = rng.uniform(_PAD_PX, size - _PAD_PX, 2)
        axes = (rng.uniform(8, 25), rng.uniform(8, 25))
        angle = rng.uniform(0, math.pi)
        weights = draw_weights()
        if not cleared(centre, 2.5 * max(axes)):
            add(_ellipse_patch(size, centre, axes, angle), weights)
    for x0, y0, x1, y1, width_um in extra_tubes_um:
        start = np.array([x0 / PIXEL_UM + _PAD_PX, y0 / PIXEL_UM + _PAD_PX])
        end = np.array([x1 / PIXEL_UM + _PAD_PX, y1 / PIXEL_UM + _PAD_PX])
        sigma_px = width_um / 2.355 / PIXEL_UM
        add(_segment_patch(size, start, end, sigma_px), np.array([1.0, 0.8, 0.6]))
    # A shared fine texture (neuropil-like, 20-60 um) keeps unrelated windows from
    # correlating through a few dominant features; it is absent from cleared boxes.
    texture = rng.standard_normal((3, size, size)).astype(np.float32)
    for index in range(3):
        blurred = cv2.GaussianBlur(texture[index], (0, 0), 7.0)
        texture[index] = (
            blurred / blurred.std() * (0.3 if index == 2 else 0.6) * texture_scale
        )
    for x0, y0, x1, y1 in clear_boxes_um:
        texture[
            :,
            int(y0 / PIXEL_UM) + _PAD_PX : int(y1 / PIXEL_UM) + _PAD_PX,
            int(x0 / PIXEL_UM) + _PAD_PX : int(x1 / PIXEL_UM) + _PAD_PX,
        ] = 0.0
    canvas += texture
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    centre_px = size / 2.0
    angle = rng.uniform(0, math.pi)
    axis_a, axis_b = 0.43 * SIZE_PX * tissue_scale, 0.37 * SIZE_PX * tissue_scale
    dx, dy = xx - centre_px, yy - centre_px
    u = (dx * math.cos(angle) + dy * math.sin(angle)) / axis_a
    v = (-dx * math.sin(angle) + dy * math.cos(angle)) / axis_b
    radius = np.sqrt(u * u + v * v)
    tissue = 1.0 / (1.0 + np.exp(np.clip((radius - 1.0) / 0.03, -50.0, 50.0)))
    return canvas, tissue.astype(np.float32)


def _with_vessels(
    structure: np.ndarray,
    vessels_um: Sequence[tuple[float, float, float, float, float]],
) -> np.ndarray:
    """A copy of the structure canvases with vessel lumens ``(x0, y0, x1, y1, w)``."""
    out = structure.copy()
    size = out.shape[1]
    for x0, y0, x1, y1, width_um in vessels_um:
        start = np.array([x0 / PIXEL_UM + _PAD_PX, y0 / PIXEL_UM + _PAD_PX])
        end = np.array([x1 / PIXEL_UM + _PAD_PX, y1 / PIXEL_UM + _PAD_PX])
        patch = _segment_patch(size, start, end, width_um / 2.355 / PIXEL_UM)
        if patch is not None:
            r0, r1, c0, c1, values = patch
            out[:, r0:r1, c0:c1] += VESSEL_WEIGHTS[:, None, None] * values[None]
    return out


def _fine_detail(rng: np.random.Generator, size: int) -> np.ndarray:
    """Independent fine spots (nuclei-like) and noise, per role."""
    out = np.empty((3, size, size), dtype=np.float32)
    for index in range(3):
        noise = rng.normal(0.0, 0.12, (size, size)).astype(np.float32)
        spots = np.zeros((size, size), dtype=np.float32)
        count = 2500
        xs = rng.integers(3, size - 3, count)
        ys = rng.integers(3, size - 3, count)
        spots[ys, xs] = rng.uniform(0.5, 1.5, count)
        spots = _blur(spots, 1.8) * 12.0
        out[index] = noise + (spots if _ROLE_ORDER[index] != "map2" else 0.0)
    return out


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return np.asarray(cv2.GaussianBlur(image, (0, 0), sigma), dtype=np.float32)


def _render(
    structure: np.ndarray, tissue: np.ndarray, detail: np.ndarray
) -> np.ndarray:
    """Intensities from log structure: glass outside, tissue inside."""
    log_glass = math.log(60.0)
    log_tissue = math.log(500.0)
    span = np.float32(log_tissue - log_glass)
    log_image = np.float32(log_glass) + span * tissue[None]
    log_image = log_image + tissue[None] * (structure + detail)
    return np.exp(log_image, dtype=np.float32)


def make_phantom(
    seed: int = 0,
    *,
    n_tubes: int = 700,
    n_ellipses: int = 900,
    warp_amplitude_um: float = 40.0,
    warp_wavelength_um: float = 1500.0,
    extra_tubes_um: Sequence[tuple[float, float, float, float, float]] = (),
    clear_boxes_um: Sequence[tuple[float, float, float, float]] = (),
    moving_structure_seed: int | None = None,
    tissue_scale: float = 1.0,
    fine_detail: bool = True,
    reference_vessels_um: Sequence[tuple[float, float, float, float, float]] = (),
    moving_vessels_um: Sequence[tuple[float, float, float, float, float]] = (),
    texture_scale: float = 1.0,
) -> Phantom:
    """Render a phantom pair.

    Args:
        seed: Seed of the reference structure and all noise.
        n_tubes: Number of random tubes.
        n_ellipses: Number of random ring-like ellipses.
        warp_amplitude_um: Amplitude of the sinusoidal part of the warp.
        warp_wavelength_um: Its wavelength.
        extra_tubes_um: Explicit tubes ``(x0, y0, x1, y1, width_um)`` in reference
            um, added to the structure.
        clear_boxes_um: Boxes ``(x0, y0, x1, y1)`` in reference um where no random
            feature is centred.
        moving_structure_seed: When given, the moving image shows an unrelated
            structure drawn from this seed (independent sections).
        tissue_scale: Scale of the tissue outline.
        fine_detail: Whether the independent fine spots and noise are drawn (off:
            the moving image is exactly the warped reference).
        reference_vessels_um: Vessel lumens ``(x0, y0, x1, y1, width_um)`` in
            reference um drawn into the reference image only.
        moving_vessels_um: Vessel lumens in reference um drawn into the moving
            image only (before the warp, so they travel with it).
        texture_scale: Scale of the shared fine texture (1: the default).

    Returns:
        The phantom with its exact pull and its affine prior.
    """
    rng = np.random.default_rng(seed)
    structure, tissue = _structure(
        rng,
        n_tubes,
        n_ellipses,
        extra_tubes_um,
        clear_boxes_um,
        tissue_scale,
        texture_scale,
    )
    size = SIZE_PX + 2 * _PAD_PX
    zeros = np.zeros((3, size, size), dtype=np.float32)
    reference_full = _render(
        _with_vessels(structure, reference_vessels_um),
        tissue,
        _fine_detail(rng, size) if fine_detail else zeros,
    )
    reference_images = reference_full[:, _PAD_PX:-_PAD_PX, _PAD_PX:-_PAD_PX]

    if moving_structure_seed is not None:
        other = np.random.default_rng(moving_structure_seed)
        structure_m, tissue_m = _structure(
            other, n_tubes, n_ellipses, (), (), tissue_scale
        )
    else:
        structure_m, tissue_m = structure, tissue
    structure_m = _with_vessels(structure_m, moving_vessels_um)
    # Moving px (canvas frame) -> reference canvas px, then resample the canvas.
    angle = math.radians(ROTATION_DEG)
    scale = 1.02
    linear = scale * np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    offset_um = np.array([60.0, -40.0])
    centre_um = np.full(2, 0.5 * SIZE_PX * PIXEL_UM)
    phase = (0.7, 1.9)

    def push(moving_um: np.ndarray) -> np.ndarray:
        m = np.asarray(moving_um, dtype=float)
        wave = warp_amplitude_um * np.stack(
            [
                np.sin(2 * math.pi * m[..., 1] / warp_wavelength_um + phase[0]),
                np.sin(2 * math.pi * m[..., 0] / warp_wavelength_um + phase[1]),
            ],
            axis=-1,
        )
        return (m - centre_um) @ linear.T + centre_um + offset_um + wave

    def pull(reference_um: np.ndarray) -> np.ndarray:
        # Local (moving crop) um, found by fixed-point iteration, then shifted to
        # the full frame.
        r = np.asarray(reference_um, dtype=float)
        m = r.copy()
        for _ in range(60):
            m = m - (push(m) - r)
        return m + MOVING_ORIGIN_UM

    linear_inverse = np.linalg.inv(linear)

    def prior(reference_um: np.ndarray) -> np.ndarray:
        r = np.asarray(reference_um, dtype=float)
        return (
            (r - offset_um - centre_um) @ linear_inverse.T
            + centre_um
            + MOVING_ORIGIN_UM
        )

    grid = np.stack(
        np.meshgrid(np.arange(SIZE_PX), np.arange(SIZE_PX)), axis=-1
    ).astype(float)
    reference_px = push(grid * PIXEL_UM) / PIXEL_UM + _PAD_PX
    map_x = reference_px[..., 0].astype(np.float32)
    map_y = reference_px[..., 1].astype(np.float32)
    detail_moving = (
        _fine_detail(np.random.default_rng(seed + 1000), size)[
            :, _PAD_PX:-_PAD_PX, _PAD_PX:-_PAD_PX
        ]
        if fine_detail
        else zeros[:, _PAD_PX:-_PAD_PX, _PAD_PX:-_PAD_PX]
    )
    moving_full = _render(structure_m, tissue_m, zeros)
    warped = np.stack(
        [
            cv2.remap(moving_full[i], map_x, map_y, cv2.INTER_LINEAR, borderValue=0.0)
            for i in range(3)
        ]
    )
    tissue_warped = cv2.remap(
        tissue_m.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderValue=0.0
    )
    # The fine detail belongs to the moving frame: it is added after the warp, so
    # it does not travel with the structure.
    moving_images = np.asarray(
        warped * np.exp(tissue_warped[None] * detail_moving), dtype=np.float32
    )

    def role_images(array: np.ndarray, origin: tuple[float, float]) -> RoleImages:
        return RoleImages(
            images={role: array[i] for i, role in enumerate(_ROLE_ORDER)},
            to_level0=_to_level0(origin),
            pixel_size_um=PIXEL_UM,
            level0_pixel_size_um=LEVEL0_PIXEL_UM,
        )

    return Phantom(
        reference=role_images(reference_images, (0.0, 0.0)),
        moving=role_images(moving_images, MOVING_ORIGIN_LEVEL0_PX),
        pull=pull,
        prior=prior,
        tubes_um=tuple((x0, y0, x1, y1) for x0, y0, x1, y1, _ in extra_tubes_um),
    )


def replace_with_texture(
    images: RoleImages, box_um: tuple[float, float, float, float], seed: int = 99
) -> RoleImages:
    """Return a copy with a box (full-frame um of ``images``) replaced by texture."""
    other = make_phantom(seed, warp_amplitude_um=0.0).reference
    return _replace_box(images, box_um, lambda role, window: other.images[role][window])


def shift_patch(
    images: RoleImages,
    box_um: tuple[float, float, float, float],
    shift_um: tuple[float, float],
) -> RoleImages:
    """Return a copy whose box shows the image content shifted by ``shift_um``.

    The box (full-frame um of ``images``) is filled from the same image displaced
    by ``shift_um``, so a window inside it matches the reference at that offset.
    """
    shift_px = (round(shift_um[0] / PIXEL_UM), round(shift_um[1] / PIXEL_UM))

    def source(role: str, window: tuple[slice, slice]) -> np.ndarray:
        rows, columns = window
        return images.images[role][
            rows.start + shift_px[1] : rows.stop + shift_px[1],
            columns.start + shift_px[0] : columns.stop + shift_px[0],
        ]

    return _replace_box(images, box_um, source)


def _replace_box(
    images: RoleImages,
    box_um: tuple[float, float, float, float],
    source: Callable[[str, tuple[slice, slice]], np.ndarray],
) -> RoleImages:
    corners = _um_to_grid(
        images.to_level0,
        images.level0_pixel_size_um,
        np.array([[box_um[0], box_um[1]], [box_um[2], box_um[3]]]),
    )
    x0, y0 = (int(round(v)) for v in corners[0])
    x1, y1 = (int(round(v)) for v in corners[1])
    window = (slice(y0, y1), slice(x0, x1))
    out = {role: image.copy() for role, image in images.images.items()}
    for role, image in out.items():
        image[window] = source(role, window)
    return RoleImages(
        images=out,
        to_level0=images.to_level0,
        pixel_size_um=images.pixel_size_um,
        level0_pixel_size_um=images.level0_pixel_size_um,
    )
