"""Tests for SpatialData write helpers."""

from __future__ import annotations

from pathlib import Path

import spatialdata as sd
from spatialdata import datasets

from merxen.io.spatialdata_io import write_spatialdata_zarr


def test_write_spatialdata_zarr_writes_blobs_dataset(tmp_path: Path) -> None:
    """Writing a multiscale SpatialData object should succeed without local shims."""
    out = tmp_path / "blobs.zarr"

    write_spatialdata_zarr(datasets.blobs(), out)

    assert out.exists()
    reloaded = sd.read_zarr(out)
    assert "blobs_image" in reloaded.images
    assert "blobs_labels" in reloaded.labels


def test_write_spatialdata_zarr_supports_overwrite(tmp_path: Path) -> None:
    """The helper should pass through SpatialData's overwrite flag."""
    out = tmp_path / "blobs.zarr"

    write_spatialdata_zarr(datasets.blobs(), out)
    write_spatialdata_zarr(datasets.blobs(), out, overwrite=True)

    assert out.exists()
