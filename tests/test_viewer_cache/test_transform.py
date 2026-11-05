"""Characterisation of the viewer-cache mask affine fallbacks."""

from __future__ import annotations

from pathlib import Path

import pytest

from merxen.viewer_cache.transform import resolve_mask_affine


@pytest.mark.parametrize(
    ("platform", "um_per_px"),
    [("MERSCOPE", 0.108), ("merscope", 0.108), ("XENIUM", 0.2125), ("xenium", 0.2125)],
)
def test_missing_transform_falls_back_to_the_viewer_default(
    tmp_path: Path, platform: str, um_per_px: float
) -> None:
    """Without a transform file the mask gets the isotropic viewer default."""
    assert resolve_mask_affine(platform, tmp_path) == (
        (um_per_px, 0.0, 0.0),
        (0.0, um_per_px, 0.0),
    )


def test_xenium_spec_wins_over_the_fallback(tmp_path: Path) -> None:
    """The enrichment transform (here experiment.xenium) is used when present."""
    (tmp_path / "experiment.xenium").write_text('{"pixel_size": 0.25}')

    assert resolve_mask_affine("XENIUM", tmp_path) == (
        (0.25, 0.0, 0.0),
        (0.0, 0.25, 0.0),
    )


def test_unknown_platform_raises_before_the_fallback(tmp_path: Path) -> None:
    """An unknown platform is a ValueError, never a silent fallback."""
    with pytest.raises(ValueError, match="Unknown platform"):
        resolve_mask_affine("COSMX", tmp_path)
