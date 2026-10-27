"""Cropped views of prepared OME-TIFF pyramids as registration-core inputs.

The register step never reads the prepared store's pixels for registration: it
opens the OME-TIFF pyramids recorded in the store attrs, crops every pyramid level
to the effective crop and hands them to the registration core as a
``ReferenceGrid`` / ``MovingImage``. Crops are snapped to multiples of 64 level-0
pixels, so every level whose factor divides 64 crops exactly and the crop-frame
pixel-centre scale of level ``k`` is exactly ``f_k``.

Frames: all crops here are **effective** crops in level-0 pixels of the full image;
a cropped level pixel ``j`` is at ``j * f + (f - 1) / 2`` in the crop frame.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import xarray as xr
from spatialdata.transformations import Affine

from merxen.image_registration import MovingImage, ReferenceGrid
from merxen.io.ome_tiff import OmeTiffInfo, open_ome_tiff_level
from merxen.manual_registration.schema import CROP_SNAP_PX, CropRect

logger = logging.getLogger(__name__)

# Level factors must divide this (== CROP_SNAP_PX) for exact cropping.
CROP_LEVEL_FACTOR_LIMIT = CROP_SNAP_PX
_FACTOR_TOLERANCE = 0.01


def pyramid_levels(
    info: OmeTiffInfo,
    *,
    channel_names: Sequence[str],
    crop: CropRect,
    exact_factors: bool,
) -> tuple[
    list[xr.DataArray],
    tuple[tuple[int, int], ...],
    tuple[tuple[float, float], ...] | None,
]:
    """Open the pyramid levels of an OME-TIFF cropped to an effective crop.

    Only levels whose integer factor ``f`` divides 64 are kept (level 0 always),
    each cropped to ``[y0 / f, ceil(y1 / f))`` by ``[x0 / f, ceil(x1 / f))``.

    Args:
        info: Metadata of the OME-TIFF.
        channel_names: One name per channel of the file.
        crop: The effective (snapped) crop in level-0 pixels of the full image.
        exact_factors: Whether the file's levels are exact integer means of level 0
            (written by prepare); then exact factors are returned.

    Returns:
        ``(levels, level_shapes_yx, level_factors_xy)``; the factors are ``None``
        when ``exact_factors`` is false (the core derives them from the shapes).

    Raises:
        ValueError: If the crop origin is not a multiple of a kept level's factor.
    """
    height0, width0 = info.level_shapes_cyx[0][1:]
    kept: list[tuple[int, xr.DataArray]] = []
    for index, (_, height, width) in enumerate(info.level_shapes_cyx):
        factor_x, factor_y = width0 / width, height0 / height
        factor = round(factor_x)
        if (
            factor < 1
            or abs(factor_x - factor) / factor >= _FACTOR_TOLERANCE
            or abs(factor_y - factor) / factor >= _FACTOR_TOLERANCE
        ):
            logger.debug("Skipping level %d of %s: odd factor", index, info.path)
            continue
        if CROP_LEVEL_FACTOR_LIMIT % factor != 0:
            logger.debug(
                "Skipping level %d of %s: factor %d does not divide %d",
                index,
                info.path,
                factor,
                CROP_LEVEL_FACTOR_LIMIT,
            )
            continue
        if any(factor == existing for existing, _ in kept):
            continue
        if crop.x0 % factor != 0 or crop.y0 % factor != 0:
            raise ValueError(
                f"crop {crop.as_tuple()} is not snapped: its origin is not a "
                f"multiple of the level factor {factor}"
            )
        rows = slice(crop.y0 // factor, min(math.ceil(crop.y1 / factor), height))
        cols = slice(crop.x0 // factor, min(math.ceil(crop.x1 / factor), width))
        level = open_ome_tiff_level(info, index, channel_names=list(channel_names))
        kept.append((factor, level.isel(y=rows, x=cols)))
    levels = [level for _, level in kept]
    shapes = tuple((int(level.sizes["y"]), int(level.sizes["x"])) for level in levels)
    factors = (
        tuple((float(factor), float(factor)) for factor, _ in kept)
        if exact_factors
        else None
    )
    return levels, shapes, factors


def crop_offset_affine(crop: CropRect, pixel_size_um: float) -> Affine:
    """Return the placement of a cropped image in the full image's micrometre frame.

    ``[[p, 0, p * x0], [0, p, p * y0], [0, 0, 1]]`` from cropped level-0 pixels
    to micrometres of the full image.
    """
    p = float(pixel_size_um)
    return Affine(
        np.array(
            [[p, 0.0, p * crop.x0], [0.0, p, p * crop.y0], [0.0, 0.0, 1.0]],
            dtype=float,
        ),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )


def cropped_reference_grid(
    info: OmeTiffInfo,
    *,
    channel_names: Sequence[str],
    crop: CropRect,
    pixel_size_um: float,
    exact_factors: bool,
) -> ReferenceGrid:
    """Describe the cropped reference OME-TIFF as the core's ``ReferenceGrid``.

    The grid's ``global`` transformation is the crop-offset affine, so a result
    resampled onto it lines up with the full reference frame.
    """
    levels, _shapes, factors = pyramid_levels(
        info, channel_names=channel_names, crop=crop, exact_factors=exact_factors
    )
    return ReferenceGrid(
        key="reference",
        levels=levels,
        transformations={"global": crop_offset_affine(crop, pixel_size_um)},
        coordinate_system="global",
        level_factors_xy=factors,
    )


def cropped_moving_image(
    info: OmeTiffInfo,
    *,
    channel_names: Sequence[str],
    crop: CropRect,
    pixel_size_um: float,
    exact_factors: bool,
) -> MovingImage:
    """Describe the cropped moving OME-TIFF as the core's ``MovingImage``."""
    levels, shapes, factors = pyramid_levels(
        info, channel_names=channel_names, crop=crop, exact_factors=exact_factors
    )
    return MovingImage(
        image_key="moving",
        channel_names=tuple(channel_names),
        pixel_size_um=pixel_size_um,
        level_shapes_yx=shapes,
        open_level=lambda k: levels[k],
        level0=levels[0],
        level_factors_xy=factors,
    )
