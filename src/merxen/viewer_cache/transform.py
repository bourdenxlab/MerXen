"""Resolve the pixel->micron affine used to place pre-built label masks.

A pre-built mask is displayed by the viewer via its OWN stored transform, so the
transform written here must map the morphology-image pixel grid into the micron
coordinate system the shapes live in. The primary source is the same transform
enrichment used to build the shapes (guaranteeing alignment); if that file is
unavailable at cache-build time we fall back to the viewer's own defaults
(``resolve_dataset_mask_affine``): the platform's isotropic
``PlatformSpec.fallback_um_per_px`` (0.108 um/px for MERSCOPE, 0.2125 for XENIUM).
"""

from __future__ import annotations

import logging
from pathlib import Path

from merxen.enrichment.enrich import resolve_px_to_um_transform
from merxen.platforms import platform_spec

logger = logging.getLogger(__name__)

AffineRows = tuple[tuple[float, float, float], tuple[float, float, float]]


def resolve_mask_affine(
    platform: str,
    original_data_path: Path,
    transform_path: Path | None = None,
) -> AffineRows:
    """Return the pixel->micron affine ``(x_transform, y_transform)`` for a mask.

    Tries the enrichment transform (matches the shapes exactly); on a missing
    transform file falls back to the viewer's isotropic default so the stage
    still produces a usable, viewer-consistent mask.
    """
    platform = str(platform).upper()
    try:
        return resolve_px_to_um_transform(
            platform=platform,
            original_data_path=Path(original_data_path),
            transform_path=Path(transform_path) if transform_path is not None else None,
        )
    except FileNotFoundError:
        # resolve_px_to_um_transform rejects an unknown platform with a
        # ValueError first, so only registered platforms get here.
        mpp = platform_spec(platform).fallback_um_per_px
        logger.warning(
            "[%s] No transform file found; falling back to %.4f um/px isotropic "
            "(matches the viewer's resolve_dataset_mask_affine default).",
            platform,
            mpp,
        )
        return (mpp, 0.0, 0.0), (0.0, mpp, 0.0)
