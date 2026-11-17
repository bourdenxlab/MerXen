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
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage
from skimage.filters import threshold_otsu

from merxen.image_registration import read_working_channel
from merxen.io.ome_tiff import read_ome_tiff_info
from merxen.manual_registration.schema import CropRect, StoreAttrs
from merxen.manual_registration.sources import (
    cropped_moving_image,
    cropped_reference_grid,
)

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


# ---------------------------------------------------------------------------
# Channel roles (contract C6)
# ---------------------------------------------------------------------------

IMAGE_EVIDENCE_ALGORITHM = "image_evidence_bp20_v1"
ROLES: tuple[str, ...] = ("at8", "map2", "dapi")
WORKING_PIXEL_SIZE_UM = 2.76
_SCOPES: tuple[str, ...] = ("reference", "moving")
_ROLE_ERROR = "channel roles: "


@dataclass(frozen=True)
class ChannelRoles:
    """Which canonical channel plays each role in each image.

    Attributes:
        reference: Role to canonical channel name of the reference image.
        moving: Role to canonical channel name of the moving image.
    """

    reference: dict[str, str]
    moving: dict[str, str]

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Return the roles as ``{"reference": {...}, "moving": {...}}``."""
        return {
            "reference": dict(sorted(self.reference.items())),
            "moving": dict(sorted(self.moving.items())),
        }


def parse_channel_roles(text: str) -> dict[str | None, dict[str, str]]:
    """Parse the channel-roles grammar ``[scope:]role=channel;...``.

    Items are separated by ``;`` (empty items are skipped). A scope is
    ``reference`` or ``moving``; an unscoped item applies to both images. A role is
    ``at8``, ``map2`` or ``dapi``; a channel is non-empty and contains no ``;``,
    ``=`` or ``:``. Whitespace around every part is stripped.

    Args:
        text: The roles text.

    Returns:
        ``{None: {...}, "reference": {...}, "moving": {...}}`` with only the
        non-empty scopes.

    Raises:
        ValueError: On a malformed item, an unknown scope or role, an empty
            channel or a role given twice in one scope. The message starts with
            ``channel roles: ``.
    """
    parsed: dict[str | None, dict[str, str]] = {}
    for raw_item in text.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if item.count("=") != 1:
            raise ValueError(
                f"{_ROLE_ERROR}item {item!r} must have the form "
                "[reference:|moving:]role=channel with exactly one '='"
            )
        left, channel = (part.strip() for part in item.split("=", 1))
        if left.count(":") > 1:
            raise ValueError(f"{_ROLE_ERROR}item {item!r} has more than one ':' scope")
        scope: str | None = None
        role = left
        if ":" in left:
            scope, role = (part.strip() for part in left.split(":", 1))
            if scope not in _SCOPES:
                raise ValueError(
                    f"{_ROLE_ERROR}unknown scope {scope!r} in {item!r}; "
                    f"use one of {list(_SCOPES)}"
                )
        if role not in ROLES:
            raise ValueError(
                f"{_ROLE_ERROR}unknown role {role!r} in {item!r}; "
                f"use one of {list(ROLES)}"
            )
        if not channel:
            raise ValueError(f"{_ROLE_ERROR}item {item!r} has an empty channel name")
        if ":" in channel:
            raise ValueError(
                f"{_ROLE_ERROR}channel {channel!r} contains a character that is "
                "not allowed (';', '=' or ':')"
            )
        scoped = parsed.setdefault(scope, {})
        if role in scoped:
            raise ValueError(
                f"{_ROLE_ERROR}role {role!r} is given twice "
                f"({'unscoped' if scope is None else scope})"
            )
        scoped[role] = channel
    return parsed


def resolve_channel_roles(
    text: str,
    *,
    reference_channels: Sequence[str],
    moving_channels: Sequence[str],
    required: Sequence[str] = ROLES,
) -> ChannelRoles:
    """Resolve the roles text against the channel names of both images.

    Unscoped items apply to both images, then scoped items override them. Names
    are exact and case-sensitive.

    Args:
        text: The roles text (see :func:`parse_channel_roles`).
        reference_channels: Canonical channel names of the reference image.
        moving_channels: Canonical channel names of the moving image.
        required: Roles that must be resolved for both images.

    Returns:
        The resolved roles.

    Raises:
        ValueError: On a parse error, a name that is not a channel of its image or
            a missing required role. The message starts with ``channel roles: ``.
    """
    parsed = parse_channel_roles(text)
    resolved: dict[str, dict[str, str]] = {"reference": {}, "moving": {}}
    for scope in (None, "reference", "moving"):
        for role, channel in parsed.get(scope, {}).items():
            for image in _SCOPES if scope is None else (scope,):
                resolved[image][role] = channel
    names = {"reference": list(reference_channels), "moving": list(moving_channels)}
    for image in _SCOPES:
        missing = [role for role in required if role not in resolved[image]]
        if missing:
            raise ValueError(
                f"{_ROLE_ERROR}no channel for role(s) {missing} of the {image} image"
            )
        for role, channel in resolved[image].items():
            if channel not in names[image]:
                raise ValueError(
                    f"{_ROLE_ERROR}channel {channel!r} (role {role}) is not a "
                    f"channel of the {image} image; its channels are {names[image]}"
                )
    return ChannelRoles(reference=resolved["reference"], moving=resolved["moving"])


# ---------------------------------------------------------------------------
# Role images: loading, coarse images, warping
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RoleImages:
    """The role channels of one image on a working grid.

    Attributes:
        images: Role to float32 ``(H, W)`` image.
        to_level0: 3x3 pixel-centre affine of working px to full-frame level-0 px.
        pixel_size_um: Working pixel size.
        level0_pixel_size_um: Pixel size of the full-frame level-0 grid.
    """

    images: dict[str, np.ndarray]
    to_level0: np.ndarray
    pixel_size_um: float
    level0_pixel_size_um: float

    @property
    def shape_yx(self) -> tuple[int, int]:
        """Return the ``(H, W)`` of the working grid."""
        first = next(iter(self.images.values()))
        return int(first.shape[0]), int(first.shape[1])


def _grid_to_um(
    to_level0: np.ndarray, level0_pixel_size_um: float, xy_px: np.ndarray
) -> np.ndarray:
    points = np.asarray(xy_px, dtype=np.float64)
    flat = points.reshape(-1, 2)
    level0 = flat @ to_level0[:2, :2].T + to_level0[:2, 2]
    return np.asarray((level0 * level0_pixel_size_um).reshape(points.shape))


def _um_to_grid(
    to_level0: np.ndarray, level0_pixel_size_um: float, xy_um: np.ndarray
) -> np.ndarray:
    points = np.asarray(xy_um, dtype=np.float64)
    flat = points.reshape(-1, 2) / level0_pixel_size_um
    inverse = np.linalg.inv(to_level0)
    grid = flat @ inverse[:2, :2].T + inverse[:2, 2]
    return np.asarray(grid.reshape(points.shape))


def _read_roles(
    store: StoreAttrs,
    image_key: str,
    channels: Mapping[str, str],
    crop: CropRect,
    working_pixel_size_um: float,
) -> RoleImages:
    record = store.images[image_key]
    if record.ome_tiff is None:
        raise ValueError(f"image {image_key!r} of the store has no OME-TIFF record")
    canonical = list(record.canonical_channel_names or record.channel_names)
    pixel_size = float(record.pixel_size_um)
    info = read_ome_tiff_info(record.ome_tiff.path)
    exact = record.ome_tiff.exact_level_factors
    if image_key == "reference":
        grid = cropped_reference_grid(
            info,
            channel_names=canonical,
            crop=crop,
            pixel_size_um=pixel_size,
            exact_factors=exact,
        )
        levels = grid.levels
        shapes = tuple((int(v.sizes["y"]), int(v.sizes["x"])) for v in levels)
        factors = grid.level_factors_xy
    else:
        moving = cropped_moving_image(
            info,
            channel_names=canonical,
            crop=crop,
            pixel_size_um=pixel_size,
            exact_factors=exact,
        )
        levels = [moving.open_level(k) for k in range(len(moving.level_shapes_yx))]
        shapes = moving.level_shapes_yx
        factors = moving.level_factors_xy
    origin = np.array(
        [[1.0, 0.0, crop.x0], [0.0, 1.0, crop.y0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    images: dict[str, np.ndarray] = {}
    to_level0: np.ndarray | None = None
    for role, channel in channels.items():
        working = read_working_channel(
            open_level=lambda k: levels[k],
            level_shapes_yx=shapes,
            channel_index=canonical.index(channel),
            target_factor=working_pixel_size_um / pixel_size,
            level_factors_xy=factors,
        )
        images[role] = np.asarray(working.pixels, dtype=np.float32)
        to_level0 = origin @ working.to_level0
    assert to_level0 is not None
    return RoleImages(
        images=images,
        to_level0=to_level0,
        pixel_size_um=float(to_level0[0, 0]) * pixel_size,
        level0_pixel_size_um=pixel_size,
    )


def load_role_images(
    store: StoreAttrs,
    roles: ChannelRoles,
    *,
    crops: Mapping[str, CropRect],
    working_pixel_size_um: float = WORKING_PIXEL_SIZE_UM,
) -> dict[str, RoleImages]:
    """Read the role channels of both images at the working pixel size.

    Args:
        store: Attrs of the prepared store.
        roles: The channel of each role in each image.
        crops: Effective (snapped) crops by ``"reference"`` and ``"moving"``, in
            full-frame level-0 px.
        working_pixel_size_um: Wanted working pixel size.

    Returns:
        ``{"reference": RoleImages, "moving": RoleImages}``; ``to_level0`` includes
        the crop origin.
    """
    return {
        key: _read_roles(
            store, key, getattr(roles, key), crops[key], working_pixel_size_um
        )
        for key in _SCOPES
    }


def coarse_role_images(
    store: StoreAttrs,
    roles: ChannelRoles | None,
    *,
    crops: Mapping[str, CropRect],
    pixel_size_um: float = 22.0,
) -> dict[str, RoleImages]:
    """Read the role channels at a coarse pixel size (target weights, overviews).

    Args:
        store: Attrs of the prepared store.
        roles: The channel of each role, or None to read the registration channel
            of each image as role ``dapi``.
        crops: Effective crops as in :func:`load_role_images`.
        pixel_size_um: Wanted coarse pixel size.

    Returns:
        ``{"reference": RoleImages, "moving": RoleImages}``.
    """
    result: dict[str, RoleImages] = {}
    for key in _SCOPES:
        if roles is None:
            channels = {"dapi": store.images[key].registration_channel}
        else:
            channels = getattr(roles, key)
        result[key] = _read_roles(store, key, channels, crops[key], pixel_size_um)
    return result


def _interpolation_matrix(size: int, nodes: np.ndarray) -> np.ndarray:
    """Dense linear-interpolation weights ``(size, len(nodes))`` onto pixels."""
    matrix = np.zeros((size, len(nodes)), dtype=np.float32)
    positions = np.arange(size, dtype=np.float64)
    if len(nodes) == 1:
        matrix[:, 0] = 1.0
        return matrix
    index = np.clip(
        np.searchsorted(nodes, positions, side="right") - 1, 0, len(nodes) - 2
    )
    span = nodes[index + 1] - nodes[index]
    fraction = (positions - nodes[index]) / span
    rows = np.arange(size)
    matrix[rows, index] = 1.0 - fraction
    matrix[rows, index + 1] = fraction
    return matrix


def warp_role_images(
    moving: RoleImages,
    *,
    reference: RoleImages,
    pull_um: Callable[[np.ndarray], np.ndarray],
    lattice_px: int = 16,
) -> RoleImages:
    """Resample the moving role images onto the reference grid through a pull map.

    ``pull_um`` is evaluated on a lattice of reference working px (including the
    far edges), converted to moving working px and upsampled bilinearly to every
    reference pixel; each role is then resampled bilinearly with border value 0.

    Args:
        moving: Moving role images.
        reference: Reference role images (defines the output grid).
        pull_um: Maps ``(M, 2)`` reference full um to moving full um.
        lattice_px: Spacing of the lattice in reference working px.

    Returns:
        The moving roles on the reference grid, carrying the reference
        ``to_level0``.
    """
    height, width = reference.shape_yx

    def nodes(size: int) -> np.ndarray:
        values = [float(v) for v in range(0, size, lattice_px)]
        if values[-1] != size - 1:
            values.append(float(size - 1))
        return np.asarray(values, dtype=np.float64)

    node_y, node_x = nodes(height), nodes(width)
    grid_x, grid_y = np.meshgrid(node_x, node_y)
    reference_um = _grid_to_um(
        reference.to_level0,
        reference.level0_pixel_size_um,
        np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2),
    )
    moving_um = np.asarray(pull_um(reference_um), dtype=np.float64)
    moving_px = _um_to_grid(
        moving.to_level0, moving.level0_pixel_size_um, moving_um
    ).reshape(grid_x.shape + (2,))
    matrix_y = _interpolation_matrix(height, node_y)
    matrix_x = _interpolation_matrix(width, node_x)
    maps = [
        np.ascontiguousarray(
            (matrix_y @ moving_px[..., axis].astype(np.float32)) @ matrix_x.T,
            dtype=np.float32,
        )
        for axis in (0, 1)
    ]
    warped = {
        role: cv2.remap(
            np.ascontiguousarray(image, dtype=np.float32),
            maps[0],
            maps[1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        for role, image in moving.images.items()
    }
    return RoleImages(
        images=warped,
        to_level0=reference.to_level0,
        pixel_size_um=reference.pixel_size_um,
        level0_pixel_size_um=reference.level0_pixel_size_um,
    )


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

_MAX_OTSU_SAMPLE = 1_000_000


@dataclass(frozen=True)
class RepresentationParams:
    """Parameters of the band-pass and orientation representations."""

    dog_sigmas_um: tuple[float, float] = (20.0, 60.0)
    tissue_sigma_um: float = 10.0
    tissue_opening_um: float = 20.0
    tissue_min_area_mm2: float = 0.05
    grey_matter_sigma_um: float = 50.0
    orientation_gradient_sigma_um: float = 2.76
    orientation_integration_sigma_um: float = 25.0
    soma_mask: bool = True
    soma_tophat_sigma_um: float = 50.0
    soma_quantile: float = 0.995
    soma_dilation_um: float = 5.0


@dataclass(frozen=True, eq=False)
class Representation:
    """Band-pass bands, MAP2 orientation and masks of one image on its grid.

    Attributes:
        bands: float32 ``(3, H, W)`` in ``ROLES`` order, unit robust SD over tissue.
        orientation: float32 ``(2, H, W)``, the MAP2 double-angle vector times
            coherence.
        tissue: Tissue mask.
        covered: Scanned area.
        grey_matter: Grey-matter mask (within tissue).
        pixel_size_um: Working pixel size.
        to_level0: 3x3 working px to full-frame level-0 px.
        level0_pixel_size_um: Level-0 pixel size.
    """

    bands: np.ndarray
    orientation: np.ndarray
    tissue: np.ndarray
    covered: np.ndarray
    grey_matter: np.ndarray
    pixel_size_um: float
    to_level0: np.ndarray
    level0_pixel_size_um: float

    def um_to_px(self, xy_um: ArrayLike) -> np.ndarray:
        """Map full-frame um to working px ``(x, y)`` (fractional)."""
        return _um_to_grid(
            self.to_level0, self.level0_pixel_size_um, np.asarray(xy_um, dtype=float)
        )

    def px_to_um(self, xy_px: ArrayLike) -> np.ndarray:
        """Map working px ``(x, y)`` to full-frame um."""
        return _grid_to_um(
            self.to_level0, self.level0_pixel_size_um, np.asarray(xy_px, dtype=float)
        )

    @property
    def shape_yx(self) -> tuple[int, int]:
        """Return the ``(H, W)`` of the grid."""
        return int(self.tissue.shape[0]), int(self.tissue.shape[1])


def _blur(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur; large sigmas through a block resize, as the prototype did."""
    image = np.ascontiguousarray(image, dtype=np.float32)
    if sigma_px <= 12.0:
        return cv2.GaussianBlur(image, (0, 0), sigma_px, borderType=cv2.BORDER_REFLECT)
    factor = int(sigma_px // 4)
    height, width = image.shape
    small = cv2.resize(
        image,
        (max(width // factor, 1), max(height // factor, 1)),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(
        small, (0, 0), sigma_px / factor, borderType=cv2.BORDER_REFLECT
    )
    return np.asarray(
        cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR),
        dtype=np.float32,
    )


def _otsu(values: np.ndarray) -> float | None:
    """Otsu threshold of a sample of ``values``; None when it is degenerate."""
    if values.size < 2:
        return None
    if values.size > _MAX_OTSU_SAMPLE:
        values = values[:: int(math.ceil(values.size / _MAX_OTSU_SAMPLE))]
    if float(values.max()) <= float(values.min()):
        return None
    return float(threshold_otsu(values))


def _disk_kernel(radius_px: int) -> np.ndarray:
    size = 2 * max(radius_px, 0) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _masks(
    images: RoleImages,
    params: RepresentationParams,
    *,
    min_sigma_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covered area, tissue and grey matter of a set of role images."""
    pixel = images.pixel_size_um

    def sigma(sigma_um: float) -> float:
        return max(sigma_um / pixel, min_sigma_px)

    dapi = images.images["dapi"]
    covered = cv2.erode((dapi > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    smooth = _blur(np.log1p(np.clip(dapi, 0.0, None)), sigma(params.tissue_sigma_um))
    threshold = _otsu(smooth[covered])
    if threshold is None:
        tissue = np.zeros_like(covered)
    else:
        tissue = covered & (smooth > threshold)
        radius = max(int(round(params.tissue_opening_um / pixel)), 1)
        tissue = (
            cv2.morphologyEx(
                tissue.astype(np.uint8), cv2.MORPH_OPEN, _disk_kernel(radius)
            )
            > 0
        )
        labels, count = ndimage.label(tissue)
        if count:
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            minimum = params.tissue_min_area_mm2 * 1e6 / (pixel * pixel)
            keep = sizes >= minimum
            keep[0] = False
            tissue = keep[labels]
    grey_matter = np.zeros_like(tissue)
    if "map2" in images.images and tissue.any():
        smooth_map2 = _blur(
            np.log1p(np.clip(images.images["map2"], 0.0, None)),
            sigma(params.grey_matter_sigma_um),
        )
        threshold = _otsu(smooth_map2[tissue])
        if threshold is not None:
            grey_matter = tissue & (smooth_map2 > threshold)
    return covered, tissue, grey_matter


def _glass_level(
    log_image: np.ndarray, covered: np.ndarray, tissue: np.ndarray
) -> float:
    glass = covered & ~tissue
    if glass.any():
        return float(np.median(log_image[glass]))
    if covered.any():
        return float(np.percentile(log_image[covered], 1.0))
    return 0.0


def _band(
    images: RoleImages,
    role: str,
    params: RepresentationParams,
    covered: np.ndarray,
    tissue: np.ndarray,
) -> np.ndarray:
    pixel = images.pixel_size_um
    log_image = np.log1p(np.clip(images.images[role], 0.0, None)).astype(np.float32)
    log_image[~covered] = _glass_level(log_image, covered, tissue)
    if role == "at8" and params.soma_mask and tissue.any():
        background = _blur(log_image, params.soma_tophat_sigma_um / pixel)
        tophat = log_image - background
        soma = tissue & (tophat > np.quantile(tophat[tissue], params.soma_quantile))
        radius = int(round(params.soma_dilation_um / pixel))
        if radius > 0:
            soma = cv2.dilate(soma.astype(np.uint8), _disk_kernel(radius)) > 0
        log_image[soma] = background[soma]
    low, high = params.dog_sigmas_um
    band = _blur(log_image, low / pixel) - _blur(log_image, high / pixel)
    reference_region = tissue if tissue.any() else covered
    if reference_region.any():
        values = band[reference_region]
        scale = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    else:
        scale = 0.0
    return np.asarray(band / (scale if scale > 1e-12 else 1.0), dtype=np.float32)


def _orientation(
    map2: np.ndarray,
    tissue: np.ndarray,
    params: RepresentationParams,
    pixel_size_um: float,
) -> np.ndarray:
    log_image = np.log1p(np.clip(map2, 0.0, None)).astype(np.float32)
    smooth = _blur(log_image, params.orientation_gradient_sigma_um / pixel_size_um)
    grad_y, grad_x = np.gradient(smooth)
    sigma = params.orientation_integration_sigma_um / pixel_size_um
    jxx = _blur(grad_x * grad_x, sigma)
    jyy = _blur(grad_y * grad_y, sigma)
    jxy = _blur(grad_x * grad_y, sigma)
    trace = jxx + jyy
    # The floor keeps flat tissue from turning noise into a confident orientation.
    floor = 1e-3 * float(trace[tissue].mean()) if tissue.any() else 1e-12
    denominator = trace + max(floor, 1e-12)
    vector = np.stack([(jxx - jyy) / denominator, 2.0 * jxy / denominator])
    vector[:, ~tissue] = 0.0
    return np.asarray(vector, dtype=np.float32)


def compute_representation(
    images: RoleImages, params: RepresentationParams | None = None
) -> Representation:
    """Compute the band-pass bands, MAP2 orientation and masks of an image.

    Args:
        images: Role images (all of ``ROLES``).
        params: Representation parameters (defaults when None).

    Returns:
        The representation on the grid of ``images``.

    Raises:
        ValueError: If a role is missing.
    """
    params = params or RepresentationParams()
    missing = [role for role in ROLES if role not in images.images]
    if missing:
        raise ValueError(f"role images lack {missing}")
    covered, tissue, grey_matter = _masks(images, params, min_sigma_px=0.0)
    bands = np.stack(
        [_band(images, role, params, covered, tissue) for role in ROLES]
    ).astype(np.float32)
    orientation = _orientation(
        images.images["map2"], tissue, params, images.pixel_size_um
    )
    return Representation(
        bands=bands,
        orientation=orientation,
        tissue=tissue,
        covered=covered,
        grey_matter=grey_matter,
        pixel_size_um=images.pixel_size_um,
        to_level0=images.to_level0,
        level0_pixel_size_um=images.level0_pixel_size_um,
    )


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


def ncc_map(template: np.ndarray, search: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of a template over a search image.

    The correlation is joint over the channels (OpenCV ``TM_CCOEFF_NORMED``).

    Args:
        template: ``(C, h, w)`` with ``C <= 4``.
        search: ``(C, H, W)``.

    Returns:
        float32 ``(H - h + 1, W - w + 1)``; NaN becomes -1.
    """
    if template.ndim != 3 or search.ndim != 3 or template.shape[0] != search.shape[0]:
        raise ValueError("template and search must be (C, h, w) and (C, H, W)")
    if template.shape[0] > 4:
        raise ValueError("ncc_map supports at most 4 channels")
    template_cl = np.ascontiguousarray(np.moveaxis(template, 0, -1), dtype=np.float32)
    search_cl = np.ascontiguousarray(np.moveaxis(search, 0, -1), dtype=np.float32)
    scores = cv2.matchTemplate(search_cl, template_cl, cv2.TM_CCOEFF_NORMED)
    return np.asarray(
        np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0), dtype=np.float32
    )
