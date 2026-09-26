"""Tests for the MERSCOPE mask-to-micron transform used by segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from merxen.config import DatasetConfig, SegmentationConfig
from merxen.segmentation.pipeline import _load_merscope_transform_matrix

_BUILT = np.array([[9.2595, 0.0, 202.44], [0.0, 9.2593, 1034.98], [0.0, 0.0, 1.0]])


def _config(zarr_path: Path, transform_path: Path | None) -> SegmentationConfig:
    return SegmentationConfig(
        dataset=DatasetConfig(
            name="S1_MERSCOPE",
            platform="MERSCOPE",
            data_path=zarr_path,
            channels=["DAPI"],
            output_dir=zarr_path.parent / "segment_out",
            transform_path=transform_path,
        )
    )


def _built_zarr(tmp_path: Path, matrix: np.ndarray | None = _BUILT) -> Path:
    zarr_path = tmp_path / "source.zarr"
    zarr_path.mkdir()
    if matrix is not None:
        np.savetxt(zarr_path / "micron_to_mosaic_pixel_transform.csv", matrix)
    return zarr_path


def test_uses_build_sidecar_without_override(tmp_path: Path) -> None:
    matrix = _load_merscope_transform_matrix(_config(_built_zarr(tmp_path), None))

    np.testing.assert_allclose(matrix, _BUILT)


def test_accepts_override_matching_the_build(tmp_path: Path) -> None:
    override = tmp_path / "override.csv"
    # Vizgen's CSV holds float32 values; the build sidecar may hold float64.
    np.savetxt(override, _BUILT.astype(np.float32))

    matrix = _load_merscope_transform_matrix(_config(_built_zarr(tmp_path), override))

    np.testing.assert_allclose(matrix, _BUILT, atol=1e-3)


def test_rejects_override_that_changed_after_the_build(tmp_path: Path) -> None:
    """A corrected CSV must not be mixed with a zarr built from the stale one."""
    stale = _BUILT.copy()
    stale[:2, 2] = 0.0
    zarr_path = _built_zarr(tmp_path, stale)
    override = tmp_path / "override.csv"
    np.savetxt(override, _BUILT)

    with pytest.raises(ValueError, match="force_spatialdata_build"):
        _load_merscope_transform_matrix(_config(zarr_path, override))


def test_uses_override_when_the_zarr_has_no_sidecar(tmp_path: Path) -> None:
    override = tmp_path / "override.csv"
    np.savetxt(override, _BUILT)

    matrix = _load_merscope_transform_matrix(
        _config(_built_zarr(tmp_path, None), override)
    )

    np.testing.assert_allclose(matrix, _BUILT)
