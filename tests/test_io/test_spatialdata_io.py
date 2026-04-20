"""Tests for SpatialData write helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


def test_write_spatialdata_zarr_passes_overwrite_flag(tmp_path: Path) -> None:
    """write_spatialdata_zarr should forward the overwrite kwarg when supplied."""
    sdata = MagicMock()
    out = tmp_path / "out.zarr"

    write_spatialdata_zarr(sdata, out, overwrite=True)

    sdata.write.assert_called_once_with(out, overwrite=True)


def test_write_spatialdata_zarr_omits_overwrite_when_none(tmp_path: Path) -> None:
    """write_spatialdata_zarr should not pass overwrite when it is None."""
    sdata = MagicMock()
    out = tmp_path / "out.zarr"

    write_spatialdata_zarr(sdata, out, overwrite=None)

    sdata.write.assert_called_once_with(out)
