"""Tests for MERSCOPE projection image selection during segmentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from merxen.config import SegmentationConfig
from merxen.io.image_source import MERSCOPE_ZPROJ_IMAGE_NAME
from merxen.segmentation.pipeline import (
    _load_dataset_sdata,
    _require_transcript_points,
)


def _image(values: np.ndarray, channels: list[str]) -> xr.DataArray:
    """Build a small channel-last image."""
    return xr.DataArray(
        values.astype(np.uint16),
        dims=("y", "x", "c"),
        coords={"c": channels},
    )


def _config(tmp_path: Path, channels: list[str]) -> SegmentationConfig:
    """Build a minimal MERSCOPE segmentation config."""
    return SegmentationConfig.model_validate(
        {
            "dataset": {
                "name": "P1_MERSCOPE",
                "platform": "MERSCOPE",
                "data_path": str(tmp_path / "source.zarr"),
                "channels": channels,
                "output_dir": str(tmp_path / "out"),
                "z_range": [0, 6],
            }
        }
    )


def test_load_merscope_prefers_projection_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Projection-only source zarrs should be tiled directly."""
    arr = np.zeros((3, 4, 3), dtype=np.uint16)
    arr[..., 0] = 1
    arr[..., 1] = 2
    arr[..., 2] = 9
    sdata = SimpleNamespace(
        images={MERSCOPE_ZPROJ_IMAGE_NAME: _image(arr, ["DAPI", "PolyT", "AT8"])},
        points={"transcripts": pd.DataFrame({"x": [1.0], "y": [2.0], "gene": ["A"]})},
    )

    monkeypatch.setattr("merxen.segmentation.pipeline.sd.read_zarr", lambda _: sdata)
    monkeypatch.setattr(
        "merxen.segmentation.pipeline._load_merscope_transform_matrix",
        lambda _: np.eye(3),
    )

    _, fetch_tile_fn, height, width, _, points = _load_dataset_sdata(
        _config(tmp_path, ["DAPI", "AT8"])
    )

    assert (height, width) == (3, 4)
    assert points is sdata.points["transcripts"]
    tile = fetch_tile_fn(0, 2, 0, 2)
    assert tile.shape == (2, 2, 2)
    np.testing.assert_array_equal(tile[..., 0], np.ones((2, 2), dtype=np.float32))
    np.testing.assert_array_equal(tile[..., 1], np.full((2, 2), 9, dtype=np.float32))


def test_load_merscope_falls_back_to_legacy_z_planes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy source zarrs with z-plane layers should still max-project tiles."""
    z0 = _image(np.ones((3, 4, 2), dtype=np.uint16), ["DAPI", "PolyT"])
    z1 = _image(np.full((3, 4, 2), 7, dtype=np.uint16), ["DAPI", "PolyT"])
    sdata = SimpleNamespace(
        images={"run_z0": z0, "run_z1": z1},
        points={"transcripts": pd.DataFrame({"x": [1.0], "y": [2.0], "gene": ["A"]})},
    )

    monkeypatch.setattr("merxen.segmentation.pipeline.sd.read_zarr", lambda _: sdata)
    monkeypatch.setattr(
        "merxen.segmentation.pipeline._load_merscope_transform_matrix",
        lambda _: np.eye(3),
    )

    _, fetch_tile_fn, height, width, _, _ = _load_dataset_sdata(
        _config(tmp_path, ["DAPI", "PolyT"])
    )

    assert (height, width) == (3, 4)
    tile = fetch_tile_fn(0, 2, 0, 2)
    np.testing.assert_array_equal(
        tile,
        np.full((2, 2, 2), 7, dtype=np.float32),
    )


def test_vzg2_without_transcript_points_has_specific_error() -> None:
    """VZG2-only segmentation failures should explain the archive limitation."""
    sdata = SimpleNamespace(
        points={},
        attrs={"merxen_vzg2": {"transcript_points_available": False}},
    )

    with pytest.raises(RuntimeError, match="built from .vzg2 only"):
        _require_transcript_points(sdata, "P1_MERSCOPE")
