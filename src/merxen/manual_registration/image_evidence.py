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

import itertools
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import ndimage
from scipy.spatial import cKDTree
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

# v2: the click check's gates became 0.2 / 0.6 (v1: 0.5 / 0.9), which changes verdicts;
# the version is part of the register provenance digest, so v1 results are not reused.
# v3: elongated vessel profiles are masked (normalised convolution and masked NCC)
# and every click check carries the vessel profile under the click.
IMAGE_EVIDENCE_ALGORITHM = "image_evidence_bp20_v3"
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
class VesselMaskParams:
    """Parameters of the vessel mask and of the vessel profile under a click.

    Non-neuropil pixels are MAP2-poor (below the glass level plus ``map2_fraction``
    of the way to the local neuropil level), MAP2-reduced (``map2_reduced_fraction``)
    and AT8-dark (a lumen lacks the neuropil staining of both channels), or bright
    in both AT8 and DAPI (erythrocytes and vessel walls autofluoresce; tangles are
    AT8-bright but not DAPI-bright). Their connected components (holes of the
    tissue mask included) are vessel profiles when their area lies in
    ``[min_area_um2, max_area_um2]`` (larger ones are tissue gaps and tears), their
    elongation is at least ``min_elongation`` and their length at least
    ``min_length_um``; components that run along the tissue edge for more than half
    their length are pial strips, not vessels. Length and width are
    ``4 sqrt(lambda)`` of the pixel second moments.

    Attributes:
        smooth_sigma_um: Gaussian sigma applied to the log channels first.
        neuropil_sigma_um: Sigma of the local MAP2 (neuropil) level.
        map2_fraction: Fraction of the way from glass to neuropil below which a
            pixel is MAP2-poor.
        map2_reduced_fraction: The same fraction for the MAP2-reduced, AT8-dark
            rule.
        background_sigma_um: Sigma of the local background of the AT8 and DAPI
            z-scores.
        at8_dark_z: Robust AT8 z-score below which a pixel is AT8-dark.
        at8_z: Robust AT8 z-score of the autofluorescence rule.
        dapi_z: Robust DAPI z-score of the autofluorescence rule.
        opening_um: Radius of the opening that cuts thin bridges between dark
            spots first.
        closing_um: Radius of the closing that bridges nuclei inside lumens.
        fill_holes_max_um2: Holes of a component up to this area are filled.
        min_area_um2: Smallest component area.
        max_area_um2: Largest component area (larger: a gap, never a vessel).
        min_elongation: Smallest ``sqrt(lmax / lmin)`` of an elongated profile.
        min_length_um: Smallest length of an elongated profile.
        dilation_um: Dilation of the masked profiles (walls, perivascular space).
        click_search_um: A click sits on a profile when the profile comes this
            close to it.
        click_clip_um: Radius around the click within which the profile's local
            axis is measured.
    """

    smooth_sigma_um: float = 3.0
    neuropil_sigma_um: float = 50.0
    map2_fraction: float = 0.5
    map2_reduced_fraction: float = 0.8
    background_sigma_um: float = 50.0
    at8_dark_z: float = -2.0
    at8_z: float = 2.0
    dapi_z: float = 1.0
    opening_um: float = 2.76
    closing_um: float = 3.0
    fill_holes_max_um2: float = 2500.0
    min_area_um2: float = 150.0
    max_area_um2: float = 30000.0
    min_elongation: float = 2.0
    min_length_um: float = 40.0
    dilation_um: float = 10.0
    click_search_um: float = 15.0
    click_clip_um: float = 200.0


@dataclass(frozen=True)
class RepresentationParams:
    """Parameters of the band-pass and orientation representations.

    With ``vessel_mask`` the elongated vessel profiles (see
    :class:`VesselMaskParams`) are excluded: they never feed the band-pass and
    orientation filters (normalised convolution) and the correlations ignore them
    (masked NCC). The profiles are detected either way, for the click check.
    """

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
    vessel_mask: bool = True
    vessel: VesselMaskParams = VesselMaskParams()


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
        structure: Non-neuropil pixels (vessel candidates), or None.
        vessel: The masked vessel profiles (dilated), or None.
        valid: Pixels that feed the filters and the correlations; None means all
            (no vessel mask).
        vessel_params: Parameters the vessel profiles were found with.
    """

    bands: np.ndarray
    orientation: np.ndarray
    tissue: np.ndarray
    covered: np.ndarray
    grey_matter: np.ndarray
    pixel_size_um: float
    to_level0: np.ndarray
    level0_pixel_size_um: float
    structure: np.ndarray | None = None
    vessel: np.ndarray | None = None
    valid: np.ndarray | None = None
    vessel_params: VesselMaskParams | None = None

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


def _normalised_blur(
    weighted: np.ndarray, weight: np.ndarray, sigma_px: float
) -> np.ndarray:
    """Normalised convolution: the blur of ``image * weight`` over that of ``weight``.

    Pixels of zero weight never feed the result; where no weighted pixel is near,
    the value is meaningless (the caller masks it).
    """
    return np.asarray(
        _blur(weighted, sigma_px) / np.maximum(_blur(weight, sigma_px), 1e-6),
        dtype=np.float32,
    )


def _band(
    images: RoleImages,
    role: str,
    params: RepresentationParams,
    covered: np.ndarray,
    tissue: np.ndarray,
    valid: np.ndarray | None = None,
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
    if valid is None:
        band = _blur(log_image, low / pixel) - _blur(log_image, high / pixel)
    else:
        weight = valid.astype(np.float32)
        weighted = log_image * weight
        band = _normalised_blur(weighted, weight, low / pixel) - _normalised_blur(
            weighted, weight, high / pixel
        )
        band[~valid] = 0.0
        tissue = tissue & valid
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
    valid: np.ndarray | None = None,
) -> np.ndarray:
    log_image = np.log1p(np.clip(map2, 0.0, None)).astype(np.float32)
    gradient_sigma = params.orientation_gradient_sigma_um / pixel_size_um
    sigma = params.orientation_integration_sigma_um / pixel_size_um
    if valid is None:
        smooth = _blur(log_image, gradient_sigma)
        grad_y, grad_x = np.gradient(smooth)
        jxx = _blur(grad_x * grad_x, sigma)
        jyy = _blur(grad_y * grad_y, sigma)
        jxy = _blur(grad_x * grad_y, sigma)
    else:
        # Masked pixels are filled from their neighbours before the gradient and
        # carry no weight in the structure tensor, so a vessel's outline never
        # sets the orientation of the tissue around it.
        weight = valid.astype(np.float32)
        smooth = _normalised_blur(log_image * weight, weight, gradient_sigma)
        grad_y, grad_x = np.gradient(smooth)
        jxx = _normalised_blur(grad_x * grad_x * weight, weight, sigma)
        jyy = _normalised_blur(grad_y * grad_y * weight, weight, sigma)
        jxy = _normalised_blur(grad_x * grad_y * weight, weight, sigma)
        tissue = tissue & valid
    trace = jxx + jyy
    # The floor keeps flat tissue from turning noise into a confident orientation.
    floor = 1e-3 * float(trace[tissue].mean()) if tissue.any() else 1e-12
    denominator = trace + max(floor, 1e-12)
    vector = np.stack([(jxx - jyy) / denominator, 2.0 * jxy / denominator])
    vector[:, ~tissue] = 0.0
    return np.asarray(vector, dtype=np.float32)


# ---------------------------------------------------------------------------
# Vessel profiles
# ---------------------------------------------------------------------------


def _component_covariances(labels: np.ndarray, count: int) -> np.ndarray:
    """Pixel covariance ``(count + 1, 2, 2)`` of every label (px^2; 0 unused)."""
    rows, columns = np.nonzero(labels)
    label = labels[rows, columns]
    xs, ys = columns.astype(np.float64), rows.astype(np.float64)
    n = np.maximum(np.bincount(label, minlength=count + 1).astype(np.float64), 1.0)
    mean_x = np.bincount(label, xs, count + 1) / n
    mean_y = np.bincount(label, ys, count + 1) / n
    # 1/12 px^2 is the variance of one uniform pixel: it keeps a one-pixel-wide
    # component's elongation finite.
    cov = np.empty((count + 1, 2, 2))
    cov[:, 0, 0] = np.bincount(label, xs * xs, count + 1) / n - mean_x**2 + 1 / 12
    cov[:, 1, 1] = np.bincount(label, ys * ys, count + 1) / n - mean_y**2 + 1 / 12
    cov[:, 0, 1] = np.bincount(label, xs * ys, count + 1) / n - mean_x * mean_y
    cov[:, 1, 0] = cov[:, 0, 1]
    return cov


def _small_holes(mask: np.ndarray, max_area_px: float) -> np.ndarray:
    """Background components of ``mask`` that miss the border and are small."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~mask).astype(np.uint8), connectivity=4
    )
    height, width = mask.shape
    x0, y0 = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    x1 = x0 + stats[:, cv2.CC_STAT_WIDTH]
    y1 = y0 + stats[:, cv2.CC_STAT_HEIGHT]
    enclosed = (x0 > 0) & (y0 > 0) & (x1 < width) & (y1 < height)
    keep = enclosed & (stats[:, cv2.CC_STAT_AREA] <= max_area_px)
    keep[0] = False  # label 0 is ``mask`` itself
    holes = keep[labels]
    return np.asarray(holes)


def _robust_z(values: np.ndarray, region: np.ndarray) -> np.ndarray:
    sample = values[region]
    median = float(np.median(sample))
    scale = 1.4826 * float(np.median(np.abs(sample - median)))
    return np.asarray((values - median) / max(scale, 1e-6), dtype=np.float32)


def _vessel_masks(
    images: RoleImages,
    covered: np.ndarray,
    tissue: np.ndarray,
    params: VesselMaskParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-neuropil structure and the dilated elongated vessel profiles.

    Returns:
        ``(structure, vessel)`` boolean masks on the grid of ``images``.
    """
    empty = np.zeros_like(tissue)
    if not tissue.any():
        return empty, empty.copy()
    pixel = images.pixel_size_um
    area_px = pixel * pixel
    # The tissue mask (DAPI) leaves large lumens out as holes; they belong to the
    # vessels around them.
    domain = (tissue | _small_holes(tissue, params.max_area_um2 / area_px)) & covered
    weight = tissue.astype(np.float32)

    def smooth(role: str) -> np.ndarray:
        log_image = np.log1p(np.clip(images.images[role], 0.0, None))
        return _blur(log_image.astype(np.float32), params.smooth_sigma_um / pixel)

    def z_score(role: str) -> np.ndarray:
        image = smooth(role)
        background = _normalised_blur(
            image * weight, weight, params.background_sigma_um / pixel
        )
        return _robust_z(image - background, tissue)

    # One full-size float image at a time: these grids are ~200 MB each.
    map2 = smooth("map2")
    neuropil = _normalised_blur(map2 * weight, weight, params.neuropil_sigma_um / pixel)
    glass = _glass_level(map2, covered, tissue)
    level = (map2 - glass) / np.maximum(neuropil - glass, 1e-6)
    del map2, neuropil
    at8_z = z_score("at8")
    poor = (level < params.map2_fraction) | (
        (level < params.map2_reduced_fraction) & (at8_z < params.at8_dark_z)
    )
    del level
    bright = at8_z > params.at8_z
    del at8_z
    bright &= z_score("dapi") > params.dapi_z
    structure = (poor | bright) & domain
    for operation, radius_um in (
        (cv2.MORPH_OPEN, params.opening_um),
        (cv2.MORPH_CLOSE, params.closing_um),
    ):
        radius = int(round(radius_um / pixel))
        if radius > 0:
            structure = (
                cv2.morphologyEx(
                    structure.astype(np.uint8), operation, _disk_kernel(radius)
                )
                > 0
            )
    structure = (
        structure | _small_holes(structure, params.fill_holes_max_um2 / area_px)
    ) & domain
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        structure.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return structure, empty
    values = np.clip(
        np.linalg.eigvalsh(_component_covariances(labels, count - 1)), 0.0, None
    )
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64) * area_px
    length = 4.0 * np.sqrt(values[:, 1]) * pixel
    elongation = np.sqrt(values[:, 1] / np.maximum(values[:, 0], 1e-9))
    # A strip along the tissue edge (pia, a torn border) is no vessel: it touches
    # the outside over much of its length.
    outside = (
        cv2.dilate((~domain).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ) & structure
    rim_um = np.bincount(labels[outside], minlength=count).astype(np.float64) * pixel
    selected = (
        (area >= params.min_area_um2)
        & (area <= params.max_area_um2)
        & (elongation >= params.min_elongation)
        & (length >= params.min_length_um)
        & (rim_um <= 0.5 * length)
    )
    selected[0] = False
    vessel = selected[labels]
    radius = int(round(params.dilation_um / pixel))
    if radius > 0 and vessel.any():
        vessel = cv2.dilate(vessel.astype(np.uint8), _disk_kernel(radius)) > 0
    return structure, np.asarray(vessel & covered)


def compute_representation(
    images: RoleImages, params: RepresentationParams | None = None
) -> Representation:
    """Compute the band-pass bands, MAP2 orientation and masks of an image.

    The elongated vessel profiles are always found; with ``params.vessel_mask``
    (the default) they are excluded from the bands, the orientation and every
    correlation made with the representation.

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
    structure, vessel = _vessel_masks(images, covered, tissue, params.vessel)
    # Without a masked pixel the masked filters equal the plain ones; the plain
    # path is faster and leaves vessel-free results bit-identical.
    valid = ~vessel if params.vessel_mask and vessel.any() else None
    bands = np.stack(
        [_band(images, role, params, covered, tissue, valid) for role in ROLES]
    ).astype(np.float32)
    orientation = _orientation(
        images.images["map2"], tissue, params, images.pixel_size_um, valid
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
        structure=structure,
        vessel=vessel,
        valid=valid,
        vessel_params=params.vessel,
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


# A masked correlation needs at least this fraction of the template's valid pixels to
# overlap valid search pixels at a shift, and a template at least this valid.
_MASKED_MIN_OVERLAP = 0.5
_MASKED_MIN_TEMPLATE_VALID = 0.25


def _ccorr(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Plain cross-correlation sums (channels summed), float64."""
    return np.asarray(
        cv2.matchTemplate(
            np.ascontiguousarray(image, dtype=np.float32),
            np.ascontiguousarray(template, dtype=np.float32),
            cv2.TM_CCORR,
        ),
        dtype=np.float64,
    )


def masked_ncc_map(
    template: np.ndarray,
    template_valid: np.ndarray,
    search: np.ndarray,
    search_valid: np.ndarray,
    *,
    min_overlap: float = _MASKED_MIN_OVERLAP,
    min_template_valid: float = _MASKED_MIN_TEMPLATE_VALID,
) -> np.ndarray:
    """Masked normalised cross-correlation (Padfield 2012), joint over the channels.

    At each shift only the pixels valid in both the template and the search
    image count: the means, the variances and the products are taken over that
    overlap, per channel, and summed over the channels as :func:`ncc_map` does.
    With every pixel valid the result equals :func:`ncc_map`.

    Args:
        template: ``(C, h, w)``.
        template_valid: ``(h, w)`` boolean.
        search: ``(C, H, W)``.
        search_valid: ``(H, W)`` boolean.
        min_overlap: A shift whose overlap is below this fraction of the
            template's valid pixels scores -1.
        min_template_valid: A template valid on less than this fraction of its
            area scores -1 everywhere.

    Returns:
        float32 ``(H - h + 1, W - w + 1)`` in ``[-1, 1]``.
    """
    if template.ndim != 3 or search.ndim != 3 or template.shape[0] != search.shape[0]:
        raise ValueError("template and search must be (C, h, w) and (C, H, W)")
    if template_valid.shape != template.shape[1:] or (
        search_valid.shape != search.shape[1:]
    ):
        raise ValueError("the valid masks must match the images")
    if template_valid.all() and search_valid.all():
        return ncc_map(template, search)
    out_shape = (
        search.shape[1] - template.shape[1] + 1,
        search.shape[2] - template.shape[2] + 1,
    )
    template_count = float(template_valid.sum())
    if template_count < min_template_valid * template_valid.size:
        return np.full(out_shape, -1.0, dtype=np.float32)
    template_mask = template_valid.astype(np.float32)
    search_mask = search_valid.astype(np.float32)
    masked_template = template.astype(np.float32) * template_mask
    masked_search = search.astype(np.float32) * search_mask
    overlap = _ccorr(search_mask, template_mask)
    template_sums = [_ccorr(search_mask, channel) for channel in masked_template]
    search_sums = [_ccorr(channel, template_mask) for channel in masked_search]
    template_squares = _ccorr(search_mask, (masked_template**2).sum(axis=0))
    search_squares = _ccorr((masked_search**2).sum(axis=0), template_mask)
    products = _ccorr(
        np.moveaxis(masked_search, 0, -1), np.moveaxis(masked_template, 0, -1)
    )
    safe = np.maximum(overlap, 1.0)
    numerator = (
        products
        - sum(t * f for t, f in zip(template_sums, search_sums, strict=True)) / safe
    )
    template_var = template_squares - sum(t * t for t in template_sums) / safe
    search_var = search_squares - sum(f * f for f in search_sums) / safe
    # The sums come from float32 correlations: a variance within their rounding of
    # zero is a flat overlap, not a signal.
    template_floor = 1e-6 * max(float(np.abs(template_squares).max()), 1e-12)
    search_floor = 1e-6 * max(float(np.abs(search_squares).max()), 1e-12)
    usable = (
        (overlap >= min_overlap * template_count)
        & (template_var > template_floor)
        & (search_var > search_floor)
    )
    denominator = np.sqrt(
        np.clip(template_var, 1e-30, None) * np.clip(search_var, 1e-30, None)
    )
    scores = np.where(usable, numerator / denominator, -1.0)
    return np.asarray(np.clip(scores, -1.0, 1.0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Peak shape and covariance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PeakShape:
    """Second-moment shape of a correlation peak region.

    Attributes:
        elongation: ``sqrt(lmax / lmin)`` of the weighted second moments.
        angle_rad: Angle of the major axis from +x towards +y.
        cov_um2: ``(2, 2)`` weighted covariance in um^2.
    """

    elongation: float
    angle_rad: float
    cov_um2: np.ndarray


def _disk_window(
    scores: np.ndarray, centre_xy_px: tuple[float, float], radius_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values and ``(x, y)`` px coordinates of ``scores`` within a disk."""
    height, width = scores.shape
    x0 = max(int(math.floor(centre_xy_px[0] - radius_px)), 0)
    x1 = min(int(math.ceil(centre_xy_px[0] + radius_px)) + 1, width)
    y0 = max(int(math.floor(centre_xy_px[1] - radius_px)), 0)
    y1 = min(int(math.ceil(centre_xy_px[1] + radius_px)) + 1, height)
    if x0 >= x1 or y0 >= y1:
        empty = np.zeros(0)
        return empty, empty, empty
    rows, columns = np.mgrid[y0:y1, x0:x1]
    inside = (columns - centre_xy_px[0]) ** 2 + (
        rows - centre_xy_px[1]
    ) ** 2 <= radius_px**2
    return (
        scores[y0:y1, x0:x1][inside].astype(np.float64),
        columns[inside].astype(np.float64),
        rows[inside].astype(np.float64),
    )


def _elongation_and_angle(cov: np.ndarray) -> tuple[float, float]:
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, 1]
    angle = math.atan2(float(major[1]), float(major[0]))
    ratio = max(float(values[1]), 1e-9) / max(float(values[0]), 1e-9)
    return math.sqrt(ratio), angle


def peak_shape(
    scores: np.ndarray,
    centre_xy_px: ArrayLike,
    *,
    pixel_size_um: float,
    radius_um: float = 150.0,
) -> PeakShape:
    """Ridge detection: second moments of a score map around a centre.

    Each value within ``radius_um`` of the centre is weighted by
    ``(C - median(C))_+ ** 2``.

    Args:
        scores: 2-D correlation map.
        centre_xy_px: Centre ``(x, y)`` in map px.
        pixel_size_um: Pixel size of the map.
        radius_um: Radius of the region.

    Returns:
        The elongation, major-axis angle and covariance (um^2).
    """
    centre_array = np.asarray(centre_xy_px, dtype=float)
    centre = (float(centre_array[0]), float(centre_array[1]))
    values, xs, ys = _disk_window(scores, centre, radius_um / pixel_size_um)
    if values.size == 0:
        return PeakShape(1.0, 0.0, np.zeros((2, 2)))
    weights = np.clip(values - float(np.median(scores)), 0.0, None) ** 2
    total = float(weights.sum())
    if total <= 0.0:
        return PeakShape(1.0, 0.0, np.zeros((2, 2)))
    mean_x = float((weights * xs).sum() / total)
    mean_y = float((weights * ys).sum() / total)
    dx, dy = xs - mean_x, ys - mean_y
    cov = (
        np.array(
            [
                [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
                [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
            ]
        )
        / total
        * pixel_size_um**2
    )
    elongation, angle = _elongation_and_angle(cov)
    return PeakShape(elongation, angle, cov)


def peak_covariance(
    scores: np.ndarray,
    peak_xy_px: ArrayLike,
    *,
    pixel_size_um: float,
    drop: float = 0.05,
    radius_um: float = 100.0,
) -> np.ndarray:
    """Noise covariance of a window match: the correlation peak region's spread.

    Each value within ``radius_um`` of the peak is weighted by
    ``(C - (C_peak - drop))_+``; the result is the weighted second moment of the
    offsets from the peak.

    Args:
        scores: 2-D correlation map.
        peak_xy_px: Peak position ``(x, y)`` in map px (may be fractional).
        pixel_size_um: Pixel size of the map.
        drop: How far below the peak value the region reaches.
        radius_um: Radius of the region.

    Returns:
        ``(2, 2)`` covariance in um^2; zeros when every weight is zero or the map
        is flat within the radius (there is no peak to measure).
    """
    peak_array = np.asarray(peak_xy_px, dtype=float)
    centre = (float(peak_array[0]), float(peak_array[1]))
    values, xs, ys = _disk_window(scores, centre, radius_um / pixel_size_um)
    if values.size == 0 or float(values.max() - values.min()) < 1e-9:
        return np.zeros((2, 2))
    weights = np.clip(values - (float(values.max()) - drop), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.zeros((2, 2))
    dx, dy = xs - centre[0], ys - centre[1]
    return np.asarray(
        np.array(
            [
                [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
                [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
            ]
        )
        / total
        * pixel_size_um**2
    )


# ---------------------------------------------------------------------------
# Matching machinery shared by the null, the click check and the windows
# ---------------------------------------------------------------------------

_REPRESENTATIONS = ("bp20", "orientation", "fused")
_DEFAULT_EXCLUSION_UM = 50.0


@dataclass(frozen=True)
class _Geometry:
    """Pixel geometry of a window of a given size and search."""

    half: int
    search: int

    @property
    def side(self) -> int:
        return 2 * self.half + 1

    @property
    def margin(self) -> int:
        return self.half + self.search


def _geometry(window_um: float, search_um: float, pixel_size_um: float) -> _Geometry:
    return _Geometry(
        half=max(int(round(window_um / 2.0 / pixel_size_um)), 1),
        search=max(int(round(search_um / pixel_size_um)), 1),
    )


def _box_sum_table(mask: np.ndarray) -> np.ndarray:
    """Integral image (zero-padded) of a boolean mask."""
    table = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    table[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int32), axis=0), axis=1)
    return table


def _box_fraction(
    table: np.ndarray, cx: np.ndarray, cy: np.ndarray, half: int
) -> np.ndarray:
    """Fraction of a box ``[c - half, c + half]`` that is set (boxes inside)."""
    x0, x1 = cx - half, cx + half + 1
    y0, y1 = cy - half, cy + half + 1
    total = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    return np.asarray(total / float((2 * half + 1) ** 2))


def _inside(
    shape_yx: tuple[int, int], cx: np.ndarray, cy: np.ndarray, margin: int
) -> np.ndarray:
    height, width = shape_yx
    return (
        (cx >= margin) & (cy >= margin) & (cx + margin < width) & (cy + margin < height)
    )


def _correlate(
    reference: Representation,
    moving: Representation,
    reference_xy: tuple[int, int],
    moving_xy: tuple[int, int],
    geometry: _Geometry,
    *,
    use_bands: bool = True,
    use_orientation: bool = True,
) -> dict[str, np.ndarray]:
    """Score maps of the reference window at one centre over a warped region.

    The template is the reference window centred at ``reference_xy`` (px); the
    search region is the moving window plus the search margin, centred at
    ``moving_xy``. The fused map is the mean of the two others. When either
    representation masks vessels the correlation is the masked NCC over the
    pixels valid in both.
    """
    half, margin = geometry.half, geometry.margin
    template_slice = (
        slice(reference_xy[1] - half, reference_xy[1] + half + 1),
        slice(reference_xy[0] - half, reference_xy[0] + half + 1),
    )
    region_slice = (
        slice(moving_xy[1] - margin, moving_xy[1] + margin + 1),
        slice(moving_xy[0] - margin, moving_xy[0] + margin + 1),
    )
    masked = reference.valid is not None or moving.valid is not None
    template_valid = region_valid = None
    if masked:
        template_valid = (
            np.ones_like(reference.tissue[template_slice])
            if reference.valid is None
            else reference.valid[template_slice]
        )
        region_valid = (
            np.ones_like(moving.tissue[region_slice])
            if moving.valid is None
            else moving.valid[region_slice]
        )

    def correlate(template: np.ndarray, search: np.ndarray) -> np.ndarray:
        if template_valid is None or region_valid is None:
            return ncc_map(template, search)
        return masked_ncc_map(template, template_valid, search, region_valid)

    maps: dict[str, np.ndarray] = {}
    if use_bands:
        maps["bp20"] = correlate(
            reference.bands[(slice(None), *template_slice)],
            moving.bands[(slice(None), *region_slice)],
        )
    if use_orientation:
        maps["orientation"] = correlate(
            reference.orientation[(slice(None), *template_slice)],
            moving.orientation[(slice(None), *region_slice)],
        )
    if use_bands and use_orientation:
        maps["fused"] = 0.5 * (maps["bp20"] + maps["orientation"])
    return maps


def _map_threads(
    function: Callable[[Any], Any], items: Sequence[Any], workers: int | None
) -> list[Any]:
    """Apply ``function`` to ``items`` on threads, results in input order.

    OpenCV's own threading is switched off while the pool runs (each worker then
    uses one thread) and restored afterwards.
    """
    count = workers if workers is not None else (os.cpu_count() or 1)
    if count <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=count) as pool:
            return list(pool.map(function, items))
    finally:
        cv2.setNumThreads(previous)


def _check_same_grid(reference: Representation, moving: Representation) -> None:
    if reference.shape_yx != moving.shape_yx or not np.allclose(
        reference.to_level0, moving.to_level0
    ):
        raise ValueError(
            "the warped moving representation must be on the reference grid"
        )


# ---------------------------------------------------------------------------
# Empirical null
# ---------------------------------------------------------------------------


# Gates of the tissue-window observations (and of their nulls). The per-click
# check has its own, looser gates (``ClickCheckParams``).
_WINDOW_MIN_TISSUE_FRACTION = 0.5
_WINDOW_MIN_VALID_FRACTION = 0.9


@dataclass(frozen=True)
class NullModel:
    """Peak statistics of unrelated window pairs.

    Attributes:
        representation: ``bp20``, ``orientation`` or ``fused``.
        window_um: Window side.
        n: Pairs whose peak was not on the border.
        peak_p95: 95th percentile of the peak.
        peak_p99: 99th percentile of the peak.
        distinct_p95: 95th percentile of peak minus the second peak.
    """

    representation: str
    window_um: float
    n: int
    peak_p95: float
    peak_p99: float
    distinct_p95: float


def _null_models(
    reference: Representation,
    moving: Representation,
    *,
    representations: Sequence[str],
    window_um: float,
    search_um: float,
    n_pairs: int,
    min_distance_um: float,
    seed: int,
    exclusion_um: float = _DEFAULT_EXCLUSION_UM,
    workers: int | None = None,
    min_tissue_fraction: float = _WINDOW_MIN_TISSUE_FRACTION,
    min_valid_fraction: float = _WINDOW_MIN_VALID_FRACTION,
) -> dict[str, NullModel]:
    _check_same_grid(reference, moving)
    pixel = reference.pixel_size_um
    geometry = _geometry(window_um, search_um, pixel)
    height, width = reference.shape_yx
    step = max(int(round(200.0 / pixel)), 1)
    xs, ys = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    cx, cy = xs.ravel(), ys.ravel()
    inside = _inside((height, width), cx, cy, geometry.margin)
    cx, cy = cx[inside], cy[inside]
    ref_ok = (
        _box_fraction(_box_sum_table(reference.tissue), cx, cy, geometry.half)
        >= min_tissue_fraction
    )
    mov_ok = (
        _box_fraction(_box_sum_table(moving.tissue), cx, cy, geometry.half)
        >= min_tissue_fraction
    ) & (
        _box_fraction(_box_sum_table(moving.covered), cx, cy, geometry.margin)
        >= min_valid_fraction
    )
    ref_points = np.stack([cx[ref_ok], cy[ref_ok]], axis=1)
    mov_points = np.stack([cx[mov_ok], cy[mov_ok]], axis=1)
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    if len(ref_points) and len(mov_points):
        minimum_px = min_distance_um / pixel
        attempts = 0
        while len(pairs) < n_pairs and attempts < 50 * n_pairs:
            attempts += 1
            i = int(rng.integers(len(ref_points)))
            j = int(rng.integers(len(mov_points)))
            if np.hypot(*(ref_points[i] - mov_points[j])) >= minimum_px:
                pairs.append((i, j))
    exclusion_px = max(int(round(exclusion_um / pixel)), 1)
    use_orientation = any(name in ("orientation", "fused") for name in representations)
    use_bands = any(name in ("bp20", "fused") for name in representations)

    def run(pair: tuple[int, int]) -> dict[str, tuple[float, float, bool]]:
        maps = _correlate(
            reference,
            moving,
            (int(ref_points[pair[0]][0]), int(ref_points[pair[0]][1])),
            (int(mov_points[pair[1]][0]), int(mov_points[pair[1]][1])),
            geometry,
            use_bands=use_bands,
            use_orientation=use_orientation,
        )
        out: dict[str, tuple[float, float, bool]] = {}
        for name in representations:
            peak, second, _x, _y, on_border = summarise_scores(
                maps[name], exclusion_px=exclusion_px
            )
            out[name] = (peak, peak - second, on_border)
        return out

    rows = _map_threads(run, pairs, workers)
    models: dict[str, NullModel] = {}
    for name in representations:
        kept = [(row[name][0], row[name][1]) for row in rows if not row[name][2]]
        if not kept:
            logger.warning(
                "empirical null (%s, %g um) has no usable pair; nothing will be "
                "confident",
                name,
                window_um,
            )
            models[name] = NullModel(name, window_um, 0, 1.0, 1.0, 1.0)
            continue
        peaks = np.array([k[0] for k in kept])
        distinct = np.array([k[1] for k in kept])
        models[name] = NullModel(
            representation=name,
            window_um=window_um,
            n=len(kept),
            peak_p95=float(np.percentile(peaks, 95)),
            peak_p99=float(np.percentile(peaks, 99)),
            distinct_p95=float(np.percentile(distinct, 95)),
        )
    return models


def empirical_null(
    reference: Representation,
    moving: Representation,
    *,
    representation: str,
    window_um: float,
    search_um: float,
    n_pairs: int,
    min_distance_um: float = 2000.0,
    seed: int = 0,
    exclusion_um: float = _DEFAULT_EXCLUSION_UM,
    workers: int | None = None,
) -> NullModel:
    """Peak statistics of a reference window matched against a distant region.

    Pairs of tissue lattice points (every 200 um, window tissue fraction at least
    0.5) at least ``min_distance_um`` apart are drawn with ``rng(seed)``; the
    reference window at the first is matched against the warped region at the
    second.

    Args:
        reference: Reference representation.
        moving: Moving representation warped onto the reference grid.
        representation: ``bp20``, ``orientation`` or ``fused``.
        window_um: Window side.
        search_um: Search margin on each side.
        n_pairs: Number of pairs to draw.
        min_distance_um: Minimum distance between the two centres.
        seed: Random seed.
        exclusion_um: Half-side of the square the second peak ignores.
        workers: Threads (None: the CPU count).

    Returns:
        The null model over the pairs whose peak is not on the border.
    """
    if representation not in _REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    return _null_models(
        reference,
        moving,
        representations=(representation,),
        window_um=window_um,
        search_um=search_um,
        n_pairs=n_pairs,
        min_distance_um=min_distance_um,
        seed=seed,
        exclusion_um=exclusion_um,
        workers=workers,
    )[representation]


# ---------------------------------------------------------------------------
# Click checks (verdicts of contract C7)
# ---------------------------------------------------------------------------

_JACOBIAN_STEP_UM = 5.0


@dataclass(frozen=True)
class ClickCheckParams:
    """Parameters of the per-click image check."""

    windows_um: tuple[float, ...] = (300.0, 600.0)
    search_um: float = 250.0
    confirm_um: float = 15.0
    contradict_um: float = 50.0
    agree_um: float = 25.0
    ridge_elongation: float = 2.0
    ridge_drop: float = 0.1
    # Looser than the tissue windows (0.5 / 0.9): edge and pia clicks carry most of
    # the signal shared by serial sections. bp20 is not zeroed off tissue (only the
    # orientation map is), so the background part of a mostly-background template
    # carries section-independent texture; the null of these checks is drawn with
    # the same gates and calibrates that contribution statistically.
    min_tissue_fraction: float = 0.2
    min_valid_fraction: float = 0.6
    null_pairs: int = 300
    seed: int = 0


@dataclass(frozen=True, eq=False)
class ClickEstimate:
    """One image estimate of where a click's structure is in the moving image.

    Attributes:
        representation: ``bp20``, ``orientation`` or ``fused``.
        window_um: Window side.
        peak: Correlation maximum.
        distinct: Peak minus the second peak.
        on_border: Whether the maximum is on the border of the search.
        confident: Peak at least the null p99, distinct at least the null distinct
            p95 and not on the border.
        elongation: Elongation of the correlation peak region (ridge detection).
        image_moving_xy_um: Estimated moving position (full um).
    """

    representation: str
    window_um: float
    peak: float
    distinct: float
    on_border: bool
    confident: bool
    elongation: float
    image_moving_xy_um: np.ndarray


@dataclass(frozen=True, eq=False)
class ClickProfile:
    """The vessel profile a click sits on.

    An obliquely cut vessel shows as an elongated profile that translates along
    its own axis between serial sections, so a click on it (typically on its end)
    is precise across the axis and not along it.

    Attributes:
        elongated: The profile is a masked vessel profile and, within
            ``click_clip_um`` of the click, elongated (``min_elongation``) and long
            (``min_length_um``) with an area in the vessel range.
        elongation: ``sqrt(lmax / lmin)`` of the profile within the clip radius.
        length_um: Its length (``4 sqrt(lmax)``).
        width_um: Its width (``4 sqrt(lmin)``).
        area_um2: Area of the whole profile (within the search box).
        axis_reference_xy: Unit major axis in the reference frame.
        axis_moving_xy: Unit major axis in the moving frame (through the pull's
            Jacobian at the click).
        masked: At least half of the profile within ``click_clip_um`` lies in
            the vessel mask.
    """

    elongated: bool
    elongation: float
    length_um: float
    width_um: float
    area_um2: float
    axis_reference_xy: np.ndarray
    axis_moving_xy: np.ndarray
    masked: bool


@dataclass(frozen=True, eq=False)
class ClickCheck:
    """The image verdict of one click.

    Attributes:
        pair_id: Landmark pair id.
        verdict: ``confirmed``, ``contradicted``, ``ridge``, ``inconclusive`` or
            ``not_checked``.
        image_moving_xy_um: Best confident fused estimate (the best confirming one
            for a confirmed click; the snap position of a contradicted click
            without one), or None.
        image_offset_um: Its distance from the click, or None.
        prediction_offset_um: Its distance from the caller's prediction, or None.
        snap_moving_xy_um: The contradiction's snap position (contradicted only).
        ridge_direction_moving_xy: Unit ridge direction in the moving frame (ridge
            only).
        elongation: Elongation of the ridge (ridge only).
        estimates: Every estimate made.
        reason: Short reason code.
        strength: ``strong`` or ``weak`` for a confirmed verdict, else None.
        profile: The vessel profile under the click, or None when there is none.
    """

    pair_id: int
    verdict: str
    image_moving_xy_um: np.ndarray | None
    image_offset_um: float | None
    prediction_offset_um: float | None
    snap_moving_xy_um: np.ndarray | None
    ridge_direction_moving_xy: np.ndarray | None
    elongation: float | None
    estimates: tuple[ClickEstimate, ...]
    reason: str
    strength: str | None = None
    profile: ClickProfile | None = None

    @property
    def on_elongated_profile(self) -> bool:
        """Whether the click sits on an elongated vessel profile."""
        return self.profile is not None and self.profile.elongated


def _not_checked(pair_id: int, reason: str) -> ClickCheck:
    return ClickCheck(
        pair_id=pair_id,
        verdict="not_checked",
        image_moving_xy_um=None,
        image_offset_um=None,
        prediction_offset_um=None,
        snap_moving_xy_um=None,
        ridge_direction_moving_xy=None,
        elongation=None,
        estimates=(),
        reason=reason,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.hypot(vector[0], vector[1]))
    return np.asarray(vector / norm if norm > 0.0 else vector, dtype=np.float64)


def click_profile(
    reference: Representation, r_um: ArrayLike, jacobian: np.ndarray
) -> ClickProfile | None:
    """The vessel profile under a reference click, if any.

    The non-neuropil component with the most pixels within ``click_search_um``
    of the click is the profile; its local axis comes from the second moments of
    its pixels within ``click_clip_um``.

    Args:
        reference: Reference representation (with its vessel structure).
        r_um: Reference click (full um).
        jacobian: ``(2, 2)`` Jacobian of the pull at the click (moving um per
            reference um).

    Returns:
        The profile, or None without a structure mask, outside the image or with
        no component near the click.
    """
    params = reference.vessel_params
    structure = reference.structure
    if params is None or structure is None:
        return None
    pixel = reference.pixel_size_um
    height, width = reference.shape_yx
    x, y = (float(v) for v in reference.um_to_px(np.asarray(r_um, dtype=float)))
    if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
        return None
    half = int(math.ceil((params.click_clip_um + params.click_search_um) / pixel)) + 1
    ix, iy = int(round(x)), int(round(y))
    x0, x1 = max(ix - half, 0), min(ix + half + 1, width)
    y0, y1 = max(iy - half, 0), min(iy + half + 1, height)
    local = structure[y0:y1, x0:x1]
    if not local.any():
        return None
    _, labels = cv2.connectedComponents(local.astype(np.uint8), connectivity=8)
    rows, columns = np.mgrid[y0:y1, x0:x1]
    distance_um = np.hypot(columns - x, rows - y) * pixel
    near = (distance_um <= params.click_search_um) & (labels > 0)
    if not near.any():
        return None
    ids, counts = np.unique(labels[near], return_counts=True)
    component = labels == int(ids[np.argmax(counts)])
    area_um2 = float(component.sum()) * pixel * pixel
    clipped = component & (distance_um <= params.click_clip_um)
    cov = _component_covariances(clipped.astype(np.int32), 1)[1]
    values = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    elongation, angle = _elongation_and_angle(cov)
    length_um = 4.0 * math.sqrt(values[1]) * pixel
    linear = reference.to_level0[:2, :2] * reference.level0_pixel_size_um
    axis_reference = _unit(linear @ np.array([math.cos(angle), math.sin(angle)]))
    axis_moving = _unit(np.asarray(jacobian, dtype=np.float64) @ axis_reference)
    # The profile itself must be a masked one: a nearby masked vessel does not make
    # an unmasked strip or gap under the click a vessel.
    vessel = reference.vessel
    masked = bool(vessel is not None and vessel[y0:y1, x0:x1][clipped].mean() >= 0.5)
    elongated = (
        masked
        and elongation >= params.min_elongation
        and length_um >= params.min_length_um
        and params.min_area_um2 <= area_um2 <= params.max_area_um2
    )
    return ClickProfile(
        elongated=bool(elongated),
        elongation=float(elongation),
        length_um=float(length_um),
        width_um=float(4.0 * math.sqrt(values[0]) * pixel),
        area_um2=area_um2,
        axis_reference_xy=axis_reference,
        axis_moving_xy=axis_moving,
        masked=masked,
    )


def _jacobians(
    pull_um: Callable[[np.ndarray], np.ndarray], points_um: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pull at the points and its central-difference Jacobians ``(n, 2, 2)``."""
    step = _JACOBIAN_STEP_UM
    offsets = np.array([[0, 0], [step, 0], [-step, 0], [0, step], [0, -step]], float)
    stacked = (points_um[:, None, :] + offsets[None]).reshape(-1, 2)
    pulled = np.asarray(pull_um(stacked), dtype=np.float64).reshape(-1, 5, 2)
    jacobian = np.empty((len(points_um), 2, 2))
    jacobian[:, :, 0] = (pulled[:, 1] - pulled[:, 2]) / (2.0 * step)
    jacobian[:, :, 1] = (pulled[:, 3] - pulled[:, 4]) / (2.0 * step)
    return pulled[:, 0], jacobian


def _bilinear(scores: np.ndarray, x: float, y: float) -> float:
    return float(
        ndimage.map_coordinates(scores, [[y], [x]], order=1, mode="nearest")[0]
    )


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(*(np.asarray(a) - np.asarray(b))))


def _assess(
    pair_id: int,
    estimates: Sequence[ClickEstimate],
    m_um: np.ndarray,
    q_um: np.ndarray,
    params: ClickCheckParams,
) -> ClickCheck:
    """Apply the verdict rules to a click's estimates, short of the ridge test.

    A result of ``inconclusive`` whose reason is not ``conflicting`` is the case
    where neither condition holds, which the caller may still turn into a ridge.
    """
    smallest = min(params.windows_um)
    confident_estimates = [e for e in estimates if e.confident]
    confirming = [
        e
        for e in confident_estimates
        if e.representation == "fused"
        and _distance(e.image_moving_xy_um, m_um) <= params.confirm_um
    ]
    voters = [
        e
        for e in confident_estimates
        if e.representation in ("bp20", "orientation")
        and e.elongation < params.ridge_elongation
        and _distance(e.image_moving_xy_um, m_um) > params.contradict_um
    ]
    best_group: list[ClickEstimate] = []
    best_mean: np.ndarray | None = None
    near_miss = ""
    # Spec 5.8: the estimates must agree within agree_um of EACH OTHER, so a
    # chain of voters that agree only through a middle one is not a group. There
    # are at most a handful of voters (two representations by the window sizes).
    for size in range(2, len(voters) + 1):
        for group_tuple in itertools.combinations(voters, size):
            group = list(group_tuple)
            if any(
                _distance(a.image_moving_xy_um, b.image_moving_xy_um) > params.agree_um
                for a, b in itertools.combinations(group, 2)
            ):
                continue
            if not any(e.representation == "bp20" for e in group):
                near_miss = "orientation_only"
                continue
            if not any(e.window_um == smallest for e in group):
                near_miss = near_miss or "large_window_only"
                continue
            mean = np.mean([e.image_moving_xy_um for e in group], axis=0)
            if (
                not best_group
                or len(group) > len(best_group)
                or (
                    len(group) == len(best_group)
                    and best_mean is not None
                    and _distance(mean, q_um) < _distance(best_mean, q_um)
                )
            ):
                best_group, best_mean = group, mean
    contradicted = bool(best_group)
    confirmed = bool(confirming)

    def blank(verdict: str, reason: str) -> ClickCheck:
        return ClickCheck(
            pair_id=pair_id,
            verdict=verdict,
            image_moving_xy_um=None,
            image_offset_um=None,
            prediction_offset_um=None,
            snap_moving_xy_um=None,
            ridge_direction_moving_xy=None,
            elongation=None,
            estimates=tuple(estimates),
            reason=reason,
        )

    def with_image(check: ClickCheck, estimate: ClickEstimate | None) -> ClickCheck:
        position = (
            estimate.image_moving_xy_um
            if estimate is not None
            else check.snap_moving_xy_um
        )
        if position is None:
            return check
        return replace(
            check,
            image_moving_xy_um=position,
            image_offset_um=_distance(position, m_um),
            prediction_offset_um=_distance(position, q_um),
        )

    fused_confident = [e for e in confident_estimates if e.representation == "fused"]
    best_fused = max(fused_confident, key=lambda e: e.peak, default=None)
    if confirmed and contradicted:
        return with_image(blank("inconclusive", "conflicting"), best_fused)
    if confirmed:
        best = max(confirming, key=lambda e: e.peak)
        strong = any(e.window_um == smallest for e in confirming) and not any(
            e.representation == "bp20"
            and e.elongation < params.ridge_elongation
            and _distance(e.image_moving_xy_um, m_um) > params.contradict_um
            for e in confident_estimates
        )
        check = blank("confirmed", "strong" if strong else "weak")
        check = replace(check, strength="strong" if strong else "weak")
        return with_image(check, best)
    if contradicted:
        snap = np.mean([e.image_moving_xy_um for e in best_group], axis=0)
        check = replace(blank("contradicted", "contradicted"), snap_moving_xy_um=snap)
        return with_image(check, best_fused)
    reason = near_miss or (
        "no_confident_estimate" if not confident_estimates else "no_agreement"
    )
    return with_image(blank("inconclusive", reason), best_fused)


def _check_one(
    pair_id: int,
    reference: Representation,
    moving: Representation,
    *,
    r_um: np.ndarray,
    m_um: np.ndarray,
    q_um: np.ndarray,
    pulled_r: np.ndarray,
    jacobian: np.ndarray,
    pull_um: Callable[[np.ndarray], np.ndarray],
    params: ClickCheckParams,
    nulls: Mapping[tuple[str, float], NullModel],
    tables: tuple[np.ndarray, np.ndarray],
) -> ClickCheck:
    pixel = reference.pixel_size_um
    height, width = reference.shape_yx
    r_px = reference.um_to_px(r_um)
    if not (0 <= r_px[0] <= width - 1 and 0 <= r_px[1] <= height - 1):
        return _not_checked(pair_id, "outside_image")
    ix, iy = int(round(r_px[0])), int(round(r_px[1]))
    if not reference.covered[iy, ix]:
        return _not_checked(pair_id, "outside_coverage")
    determinant = float(np.linalg.det(jacobian))
    if not np.isfinite(determinant) or abs(determinant) < 1e-9:
        return _not_checked(pair_id, "singular_jacobian")
    rc_um = r_um + np.linalg.solve(jacobian, m_um - pulled_r)
    rc_px = reference.um_to_px(rc_um)
    if not (0 <= rc_px[0] <= width - 1 and 0 <= rc_px[1] <= height - 1):
        return _not_checked(pair_id, "outside_coverage")
    if not moving.covered[int(round(rc_px[1])), int(round(rc_px[0]))]:
        return _not_checked(pair_id, "outside_coverage")

    linear = reference.to_level0[:2, :2] * reference.level0_pixel_size_um
    cx_array, cy_array = np.array([ix]), np.array([iy])
    skipped: list[str] = []
    raw: list[tuple[str, float, float, float, bool, np.ndarray, float]] = []
    kept_maps: dict[float, dict[str, np.ndarray]] = {}
    for window_um in sorted(params.windows_um):
        geometry = _geometry(window_um, params.search_um, pixel)
        if not _inside(reference.shape_yx, cx_array, cy_array, geometry.margin)[0]:
            skipped.append("outside_image")
            continue
        tissue_fraction = _box_fraction(tables[0], cx_array, cy_array, geometry.half)[0]
        if tissue_fraction < params.min_tissue_fraction:
            skipped.append("no_tissue")
            continue
        valid_fraction = _box_fraction(tables[1], cx_array, cy_array, geometry.margin)[
            0
        ]
        if valid_fraction < params.min_valid_fraction:
            skipped.append("outside_coverage")
            continue
        maps = _correlate(reference, moving, (ix, iy), (ix, iy), geometry)
        kept_maps[window_um] = maps
        exclusion_px = max(int(round(_DEFAULT_EXCLUSION_UM / pixel)), 1)
        for name in _REPRESENTATIONS:
            peak, second, x, y, on_border = summarise_scores(
                maps[name], exclusion_px=exclusion_px
            )
            shape = peak_shape(maps[name], (x, y), pixel_size_um=pixel)
            delta = linear @ np.array([x - geometry.search, y - geometry.search])
            raw.append(
                (
                    name,
                    window_um,
                    peak,
                    peak - second,
                    on_border,
                    delta,
                    shape.elongation,
                )
            )
    if not raw:
        reason = skipped[0] if skipped else "no_estimates"
        return ClickCheck(
            pair_id=pair_id,
            verdict="inconclusive",
            image_moving_xy_um=None,
            image_offset_um=None,
            prediction_offset_um=None,
            snap_moving_xy_um=None,
            ridge_direction_moving_xy=None,
            elongation=None,
            estimates=(),
            reason=reason,
        )
    image_positions = np.asarray(
        pull_um(r_um[None, :] + np.array([entry[5] for entry in raw])),
        dtype=np.float64,
    )
    estimates: list[ClickEstimate] = []
    for entry, position in zip(raw, image_positions, strict=True):
        name, window_um, peak, distinct, on_border, _delta, elongation = entry
        null = nulls[(name, window_um)]
        confident = (
            peak >= null.peak_p99 and distinct >= null.distinct_p95 and not on_border
        )
        estimates.append(
            ClickEstimate(
                representation=name,
                window_um=window_um,
                peak=peak,
                distinct=distinct,
                on_border=on_border,
                confident=bool(confident),
                elongation=elongation,
                image_moving_xy_um=position,
            )
        )

    assessed = _assess(pair_id, estimates, m_um, q_um, params)
    if assessed.verdict != "inconclusive" or assessed.reason == "conflicting":
        return assessed

    # Ridge: the largest window's fused map is elongated through the click's own
    # position, which correlates nearly as well as the best peak.
    largest = max(kept_maps)
    fused_map = kept_maps[largest]["fused"]
    geometry = _geometry(largest, params.search_um, pixel)
    map_x = geometry.search + (rc_px[0] - ix)
    map_y = geometry.search + (rc_px[1] - iy)
    if 0 <= map_x <= fused_map.shape[1] - 1 and 0 <= map_y <= fused_map.shape[0] - 1:
        peak_value = float(fused_map.max())
        shape = peak_shape(fused_map, (map_x, map_y), pixel_size_um=pixel)
        null = nulls[("fused", largest)]
        if (
            shape.elongation >= params.ridge_elongation
            and _bilinear(fused_map, map_x, map_y) >= peak_value - params.ridge_drop
            and peak_value >= null.peak_p95
        ):
            direction = jacobian @ np.array(
                [math.cos(shape.angle_rad), math.sin(shape.angle_rad)]
            )
            norm = float(np.hypot(*direction))
            return replace(
                assessed,
                verdict="ridge",
                reason="ridge",
                ridge_direction_moving_xy=direction / norm if norm > 0 else direction,
                elongation=float(shape.elongation),
            )
    return assessed


def check_clicks(
    reference: Representation,
    moving_warped: Representation,
    *,
    pair_ids: Sequence[int],
    reference_xy_um: ArrayLike,
    moving_xy_um: ArrayLike,
    predicted_moving_xy_um: ArrayLike,
    pull_um: Callable[[np.ndarray], np.ndarray],
    params: ClickCheckParams | None = None,
    nulls: Mapping[tuple[str, float], NullModel] | None = None,
    workers: int | None = None,
) -> list[ClickCheck]:
    """Check each landmark click against the image evidence.

    For each click the reference window is matched against the moving image
    warped by ``pull_um`` (band-pass, orientation and fused representations, at
    each window size). The rules of the verdicts are those of contract C7: a click
    is confirmed by a confident fused estimate close to it, contradicted by at
    least two agreeing confident non-ridge estimates (with a band-pass estimate
    and one at the smallest window) farther away, ridge when only an elongated
    peak region passes through it, else inconclusive; a click outside either image
    is not checked. A confirmation is ``strong`` (the confirming estimate is at the
    smallest window and no confident non-ridge band-pass estimate disagrees) or
    ``weak``. Every check also carries the vessel profile under the click
    (:func:`click_profile`); with the vessel mask on, the vessel itself never
    feeds an estimate, so it cannot confirm a click on it.

    Args:
        reference: Reference representation.
        moving_warped: Moving representation on the reference grid.
        pair_ids: Landmark pair ids.
        reference_xy_um: ``(N, 2)`` clicked reference positions (full um).
        moving_xy_um: ``(N, 2)`` clicked moving positions (full um).
        predicted_moving_xy_um: ``(N, 2)`` leave-one-out predictions (full um).
        pull_um: Maps reference full um to moving full um (never assumed affine).
        params: Check parameters.
        nulls: Null models by ``(representation, window_um)``; missing ones are
            built here.
        workers: Threads for the nulls (None: the CPU count).

    Returns:
        One :class:`ClickCheck` per click, in input order.
    """
    params = params or ClickCheckParams()
    _check_same_grid(reference, moving_warped)
    ids = [int(v) for v in pair_ids]
    reference_xy = np.asarray(reference_xy_um, dtype=np.float64).reshape(-1, 2)
    moving_xy = np.asarray(moving_xy_um, dtype=np.float64).reshape(-1, 2)
    predicted_xy = np.asarray(predicted_moving_xy_um, dtype=np.float64).reshape(-1, 2)
    if not len(ids) == len(reference_xy) == len(moving_xy) == len(predicted_xy):
        raise ValueError("pair_ids and the position arrays must have the same length")
    if not ids:
        return []
    null_models: dict[tuple[str, float], NullModel] = dict(nulls or {})
    for window_um in params.windows_um:
        if any((name, window_um) not in null_models for name in _REPRESENTATIONS):
            built = _null_models(
                reference,
                moving_warped,
                representations=_REPRESENTATIONS,
                window_um=window_um,
                search_um=params.search_um,
                n_pairs=params.null_pairs,
                min_distance_um=2000.0,
                seed=params.seed,
                workers=workers,
                min_tissue_fraction=params.min_tissue_fraction,
                min_valid_fraction=params.min_valid_fraction,
            )
            for name, model in built.items():
                null_models.setdefault((name, window_um), model)
    pulled, jacobians = _jacobians(pull_um, reference_xy)
    tables = (_box_sum_table(reference.tissue), _box_sum_table(moving_warped.covered))

    def run(index: int) -> ClickCheck:
        check = _check_one(
            ids[index],
            reference,
            moving_warped,
            r_um=reference_xy[index],
            m_um=moving_xy[index],
            q_um=predicted_xy[index],
            pulled_r=pulled[index],
            jacobian=jacobians[index],
            pull_um=pull_um,
            params=params,
            nulls=null_models,
            tables=tables,
        )
        profile = click_profile(reference, reference_xy[index], jacobians[index])
        return replace(check, profile=profile)

    return list(_map_threads(run, list(range(len(ids))), workers))


# ---------------------------------------------------------------------------
# Tissue-scale window observations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowParams:
    """Parameters of the tissue-scale window observations."""

    sizes_um: tuple[float, ...] = (600.0, 800.0)
    search_um: float = 300.0
    step_um: float = 200.0
    min_tissue_fraction: float = 0.5
    min_valid_fraction: float = 0.9
    exclusion_um: float = 50.0
    null_pairs: int = 700
    null_min_distance_um: float = 2000.0
    neighbour_min_um: float = 250.0
    neighbour_max_um: float = 450.0
    neighbour_min: int = 2
    neighbour_tolerance_um: float = 30.0
    merge_radius_um: float = 300.0
    thinning_um: float = 400.0
    peak_drop: float = 0.05
    peak_radius_um: float = 100.0
    edge_distance_um: float = 250.0
    grey_matter_fraction: float = 0.5
    workers: int | None = None
    seed: int = 1


@dataclass(frozen=True, eq=False)
class WindowObservations:
    """Calibrated, consistent and thinned window displacements.

    Attributes:
        reference_xy_um: ``(K, 2)`` window centres in the reference (full um).
        moving_xy_um: ``(K, 2)`` matched positions in the moving image (full um).
        peak_cov_um2: ``(K, 2, 2)`` NCC peak-region covariance in the moving frame
            (the fixed term F of contract C2).
        elongation: ``(K,)`` ``sqrt(lmax / lmin)`` of ``peak_cov_um2``.
        window_um: ``(K,)`` window sides.
        peak: ``(K,)`` correlation maxima.
        distinct: ``(K,)`` peak minus the second peak.
        region: ``grey_matter``, ``white_matter`` or ``edge`` per window.
        residual_um: ``(K, 2)`` matched position minus the prior's, moving um.
        candidates: Every candidate window (QC and JSON).
        nulls: Null model by window side.
        summary: Counts, null quantiles, region fractions, peak SDs and runtime.
    """

    reference_xy_um: np.ndarray
    moving_xy_um: np.ndarray
    peak_cov_um2: np.ndarray
    elongation: np.ndarray
    window_um: np.ndarray
    peak: np.ndarray
    distinct: np.ndarray
    region: tuple[str, ...]
    residual_um: np.ndarray
    candidates: pd.DataFrame
    nulls: dict[float, NullModel]
    summary: dict[str, Any]


def _clipped_box_fraction(
    table: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    half: int,
    shape_yx: tuple[int, int],
) -> np.ndarray:
    """Fraction of a box set, with the box clipped to the image."""
    height, width = shape_yx
    x0, x1 = np.clip(cx - half, 0, width), np.clip(cx + half + 1, 0, width)
    y0, y1 = np.clip(cy - half, 0, height), np.clip(cy + half + 1, 0, height)
    total = table[y1, x1] - table[y0, x1] - table[y1, x0] + table[y0, x0]
    area = np.maximum((x1 - x0) * (y1 - y0), 1)
    return np.asarray(total / area)


def _covariance_to_moving(cov: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    moved = jacobian @ cov @ jacobian.T
    return np.asarray(0.5 * (moved + moved.T))


def _elongation(cov: np.ndarray) -> float:
    values = np.linalg.eigvalsh(cov)
    return math.sqrt(max(float(values[1]), 1e-9) / max(float(values[0]), 1e-9))


def tissue_window_observations(
    reference: Representation,
    moving_warped: Representation,
    *,
    pull_um: Callable[[np.ndarray], np.ndarray],
    params: WindowParams | None = None,
) -> WindowObservations:
    """Window displacements calibrated against a null, consistent and thinned.

    Reference windows of each size on a lattice over the tissue are matched
    (band-pass, joint over the three channels) against the moving image warped by
    ``pull_um``. A window is confident when its peak reaches the null's p99 and is
    not on the search border; it is accepted when it agrees within the tolerance
    with the median displacement of at least ``neighbour_min`` confident
    neighbours of the same size at ``neighbour_min_um`` to ``neighbour_max_um``.
    Larger sizes win the merge, then windows are thinned greedily by peak.

    Args:
        reference: Reference representation.
        moving_warped: Moving representation on the reference grid.
        pull_um: Maps reference full um to moving full um (never assumed affine).
        params: Window parameters.

    Returns:
        The observations; the NCC peak-region covariance of each is in the moving
        frame and carries no noise floor (the caller adds it, contract C2).
    """
    params = params or WindowParams()
    started = time.perf_counter()
    _check_same_grid(reference, moving_warped)
    pixel = reference.pixel_size_um
    shape_yx = reference.shape_yx
    height, width = shape_yx
    linear = reference.to_level0[:2, :2] * reference.level0_pixel_size_um
    step = max(int(round(params.step_um / pixel)), 1)
    grid_x, grid_y = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    lattice_x, lattice_y = grid_x.ravel(), grid_y.ravel()
    tissue_table = _box_sum_table(reference.tissue)
    covered_table = _box_sum_table(moving_warped.covered)
    exclusion_px = max(int(round(params.exclusion_um / pixel)), 1)

    sizes = sorted(params.sizes_um, reverse=True)
    records: list[dict[str, Any]] = []
    nulls: dict[float, NullModel] = {}
    for size in sizes:
        geometry = _geometry(size, params.search_um, pixel)
        usable = _inside(shape_yx, lattice_x, lattice_y, geometry.margin)
        cx, cy = lattice_x[usable], lattice_y[usable]
        usable = (
            _box_fraction(tissue_table, cx, cy, geometry.half)
            >= params.min_tissue_fraction
        ) & (
            _box_fraction(covered_table, cx, cy, geometry.margin)
            >= params.min_valid_fraction
        )
        cx, cy = cx[usable], cy[usable]
        nulls[size] = _null_models(
            reference,
            moving_warped,
            representations=("bp20",),
            window_um=size,
            search_um=params.search_um,
            n_pairs=params.null_pairs,
            min_distance_um=params.null_min_distance_um,
            seed=params.seed,
            exclusion_um=params.exclusion_um,
            workers=params.workers,
            min_tissue_fraction=params.min_tissue_fraction,
            min_valid_fraction=params.min_valid_fraction,
        )["bp20"]

        def run(
            index: int,
            cx: np.ndarray = cx,
            cy: np.ndarray = cy,
            geometry: _Geometry = geometry,
        ) -> tuple[float, float, bool, np.ndarray, np.ndarray]:
            centre = (int(cx[index]), int(cy[index]))
            scores = _correlate(
                reference,
                moving_warped,
                centre,
                centre,
                geometry,
                use_orientation=False,
            )["bp20"]
            peak, second, x, y, on_border = summarise_scores(
                scores, exclusion_px=exclusion_px
            )
            cov = peak_covariance(
                scores,
                (x, y),
                pixel_size_um=pixel,
                drop=params.peak_drop,
                radius_um=params.peak_radius_um,
            )
            delta = linear @ np.array([x - geometry.search, y - geometry.search])
            return peak, peak - second, on_border, delta, cov

        results = _map_threads(run, list(range(len(cx))), params.workers)
        for index, (peak, distinct, on_border, delta, cov) in enumerate(results):
            records.append(
                {
                    "size": size,
                    "cx": int(cx[index]),
                    "cy": int(cy[index]),
                    "peak": peak,
                    "distinct": distinct,
                    "on_border": on_border,
                    "delta": delta,
                    "cov": cov,
                }
            )

    count = len(records)
    size_array = np.array([r["size"] for r in records], dtype=np.float64)
    centre_px = np.array([[r["cx"], r["cy"]] for r in records], dtype=np.float64)
    centre_px = centre_px.reshape(-1, 2)
    centre_um = reference.px_to_um(centre_px) if count else np.zeros((0, 2))
    delta_um = np.array([r["delta"] for r in records]).reshape(-1, 2)
    peak = np.array([r["peak"] for r in records], dtype=np.float64)
    distinct = np.array([r["distinct"] for r in records], dtype=np.float64)
    on_border = np.array([r["on_border"] for r in records], dtype=bool)
    cov_reference = np.array([r["cov"] for r in records]).reshape(-1, 2, 2)
    if count:
        base = np.asarray(pull_um(centre_um), dtype=np.float64)
        matched = np.asarray(pull_um(centre_um + delta_um), dtype=np.float64)
    else:
        base = matched = np.zeros((0, 2))
    residual = matched - base
    p99 = np.array([nulls[s].peak_p99 for s in size_array]) if count else np.zeros(0)
    confident = (peak >= p99) & ~on_border

    # Neighbour consistency, per size.
    accepted = np.zeros(count, dtype=bool)
    n_neighbours = np.zeros(count, dtype=int)
    deviation = np.full(count, np.nan)
    for size in sizes:
        members = np.flatnonzero((size_array == size) & confident)
        if len(members) == 0:
            continue
        tree = cKDTree(centre_um[members])
        for local, neighbours in enumerate(
            tree.query_ball_point(centre_um[members], r=params.neighbour_max_um)
        ):
            distances = np.hypot(
                *(centre_um[members][neighbours] - centre_um[members][local]).T
            )
            chosen = [
                n
                for n, d in zip(neighbours, distances, strict=True)
                if d >= params.neighbour_min_um
            ]
            index = members[local]
            n_neighbours[index] = len(chosen)
            if len(chosen) < params.neighbour_min:
                continue
            median = np.median(residual[members[chosen]], axis=0)
            deviation[index] = float(np.hypot(*(residual[index] - median)))
            accepted[index] = deviation[index] <= params.neighbour_tolerance_um

    # Merge: larger windows win.
    merged = np.zeros(count, dtype=bool)
    larger_accepted: list[int] = []
    for size in sizes:
        members = np.flatnonzero((size_array == size) & accepted)
        if larger_accepted:
            tree = cKDTree(centre_um[larger_accepted])
            nearest, _ = tree.query(centre_um[members])
            members = members[nearest > params.merge_radius_um]
        merged[members] = True
        larger_accepted.extend(np.flatnonzero((size_array == size) & accepted).tolist())

    # Thinning: greedy by peak.
    kept = np.zeros(count, dtype=bool)
    kept_positions: list[np.ndarray] = []
    for index in np.argsort(-np.where(merged, peak, -np.inf), kind="stable"):
        if not merged[index]:
            break
        if kept_positions and (
            np.hypot(*(np.array(kept_positions) - centre_um[index]).T).min()
            < params.thinning_um
        ):
            continue
        kept[index] = True
        kept_positions.append(centre_um[index])
    selected = np.flatnonzero(kept)

    # Noise covariance in the moving frame.
    if len(selected):
        _, jacobians = _jacobians(pull_um, centre_um[selected])
        cov_moving = np.array(
            [
                _covariance_to_moving(cov_reference[i], jac)
                for i, jac in zip(selected, jacobians, strict=True)
            ]
        )
    else:
        cov_moving = np.zeros((0, 2, 2))
    elongation = np.array([_elongation(c) for c in cov_moving])

    # Regions of every candidate.
    if count:
        distance_to_edge = (
            cv2.distanceTransform(reference.tissue.astype(np.uint8), cv2.DIST_L2, 5)[
                centre_px[:, 1].astype(int), centre_px[:, 0].astype(int)
            ]
            * pixel
        )
        grey_fraction = _clipped_box_fraction(
            _box_sum_table(reference.grey_matter),
            centre_px[:, 0].astype(int),
            centre_px[:, 1].astype(int),
            max(int(round(250.0 / pixel)), 1),
            shape_yx,
        )
    else:
        distance_to_edge = grey_fraction = np.zeros(0)
    region_all = np.where(
        distance_to_edge < params.edge_distance_um,
        "edge",
        np.where(
            grey_fraction >= params.grey_matter_fraction, "grey_matter", "white_matter"
        ),
    )

    candidates = pd.DataFrame(
        {
            "window_um": size_array,
            "reference_x_um": centre_um[:, 0],
            "reference_y_um": centre_um[:, 1],
            "peak": peak,
            "distinct": distinct,
            "on_border": on_border,
            "delta_x_um": delta_um[:, 0],
            "delta_y_um": delta_um[:, 1],
            "residual_x_um": residual[:, 0],
            "residual_y_um": residual[:, 1],
            "confident": confident,
            "n_neighbours": n_neighbours,
            "neighbour_deviation_um": deviation,
            "accepted": accepted,
            "merged": merged,
            "kept": kept,
            "region": region_all,
        }
    )
    by_size: dict[str, dict[str, Any]] = {}
    for size in sizes:
        mask = size_array == size
        by_size[f"{size:g}"] = {
            "candidates": int(mask.sum()),
            "confident": int((mask & confident).sum()),
            "accepted": int((mask & accepted).sum()),
            "accepted_fraction": (
                float((mask & accepted).sum() / mask.sum()) if mask.any() else 0.0
            ),
            "null_p95": nulls[size].peak_p95,
            "null_p99": nulls[size].peak_p99,
        }
    region_fraction = {
        name: (
            float((accepted & (region_all == name)).sum() / (region_all == name).sum())
            if (region_all == name).any()
            else None
        )
        for name in ("grey_matter", "white_matter", "edge")
    }
    if len(selected):
        eigenvalues = np.array([np.linalg.eigvalsh(c) for c in cov_moving])
        minor = np.sqrt(np.clip(eigenvalues[:, 0], 0.0, None))
        major = np.sqrt(np.clip(eigenvalues[:, 1], 0.0, None))
        sd: dict[str, dict[str, float | None]] = {
            "minor": {
                "median": float(np.median(minor)),
                "p95": float(np.percentile(minor, 95)),
            },
            "major": {
                "median": float(np.median(major)),
                "p95": float(np.percentile(major, 95)),
            },
        }
    else:
        sd = {
            "minor": {"median": None, "p95": None},
            "major": {"median": None, "p95": None},
        }
    summary: dict[str, Any] = {
        "algorithm": IMAGE_EVIDENCE_ALGORITHM,
        "by_size": by_size,
        "n_merged": int(merged.sum()),
        "n_thinned": int(len(selected)),
        "accepted_fraction_by_region": region_fraction,
        "peak_sd_um": sd,
        "runtime_s": time.perf_counter() - started,
    }
    logger.info(
        "window observations: %d candidates, %d accepted, %d merged, %d thinned "
        "in %.1f s",
        count,
        int(accepted.sum()),
        summary["n_merged"],
        summary["n_thinned"],
        summary["runtime_s"],
    )
    return WindowObservations(
        reference_xy_um=centre_um[selected],
        moving_xy_um=matched[selected],
        peak_cov_um2=cov_moving,
        elongation=elongation,
        window_um=size_array[selected],
        peak=peak[selected],
        distinct=distinct[selected],
        region=tuple(str(v) for v in region_all[selected]),
        residual_um=residual[selected],
        candidates=candidates,
        nulls=nulls,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Target weights for suggestions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TargetWeights:
    """Candidate positions for new landmarks and how much each is worth.

    Attributes:
        centres_um: ``(M, 2)`` reference full um.
        weights: ``(M,)`` weights.
        region: ``grey_matter`` or ``other`` per centre.
    """

    centres_um: np.ndarray
    weights: np.ndarray
    region: tuple[str, ...]


def target_weight_map(
    reference: RoleImages,
    moving: RoleImages,
    *,
    pull_um: Callable[[np.ndarray], np.ndarray],
    step_um: float = 100.0,
    grey_matter_weight: float = 1.0,
    other_weight: float = 0.25,
) -> TargetWeights:
    """Weight candidate landmark positions by the tissue they sit in.

    Candidates lie every ``step_um`` on the shared tissue (reference tissue whose
    pull falls in moving tissue); grey matter, found from MAP2, weighs more. With
    no MAP2 image every candidate is ``other``.

    Args:
        reference: Coarse reference role images.
        moving: Coarse moving role images.
        pull_um: Maps reference full um to moving full um.
        step_um: Candidate spacing.
        grey_matter_weight: Weight in grey matter.
        other_weight: Weight elsewhere.

    Returns:
        The candidates with weights and regions.
    """
    params = RepresentationParams()
    _, reference_tissue, reference_grey = _masks(reference, params, min_sigma_px=1.0)
    _, moving_tissue, _ = _masks(moving, params, min_sigma_px=1.0)
    height, width = reference.shape_yx
    spacing = max(step_um / reference.pixel_size_um, 1.0)
    grid_x, grid_y = np.meshgrid(
        np.arange(0.0, width, spacing), np.arange(0.0, height, spacing)
    )
    points_px = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    nearest = np.rint(points_px).astype(int)
    nearest[:, 0] = np.clip(nearest[:, 0], 0, width - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, height - 1)
    in_tissue = reference_tissue[nearest[:, 1], nearest[:, 0]]
    points_px, nearest = points_px[in_tissue], nearest[in_tissue]
    if len(points_px) == 0:
        return TargetWeights(np.zeros((0, 2)), np.zeros(0), ())
    centres_um = _grid_to_um(
        reference.to_level0, reference.level0_pixel_size_um, points_px
    )
    moving_px = _um_to_grid(
        moving.to_level0,
        moving.level0_pixel_size_um,
        np.asarray(pull_um(centres_um), dtype=np.float64),
    )
    moving_nearest = np.rint(moving_px).astype(int)
    mov_height, mov_width = moving.shape_yx
    inside = (
        (moving_nearest[:, 0] >= 0)
        & (moving_nearest[:, 0] < mov_width)
        & (moving_nearest[:, 1] >= 0)
        & (moving_nearest[:, 1] < mov_height)
    )
    shared = np.zeros(len(points_px), dtype=bool)
    shared[inside] = moving_tissue[moving_nearest[inside, 1], moving_nearest[inside, 0]]
    centres_um, nearest = centres_um[shared], nearest[shared]
    is_grey = reference_grey[nearest[:, 1], nearest[:, 0]]
    return TargetWeights(
        centres_um=centres_um,
        weights=np.where(is_grey, grey_matter_weight, other_weight).astype(np.float64),
        region=tuple("grey_matter" if g else "other" for g in is_grey),
    )
