"""Validation checks for optional pciSeq acquisition settings."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from merxen.config import PciSeqConfig


def _config(**overrides: Any) -> PciSeqConfig:
    fields = {
        "pair_id": "pair",
        "sample_id": "sample",
        "platform": "XENIUM",
        "segmentation": "reseg",
        "source_h5ad": "clustered.h5ad",
        "spatialdata_path": "latest_spatialdata.zarr",
        "reference_path": "reference.csv",
    }
    return PciSeqConfig.model_validate(fields | overrides)


def test_pciseq_config_requires_explicit_reference() -> None:
    with pytest.raises(ValidationError, match="reference_path"):
        PciSeqConfig.model_validate(_config().model_dump(exclude={"reference_path"}))


def test_pciseq_default_geometry_records_2d_projection() -> None:
    config = _config()
    assert config.geometry_mode == "project_2d"
    assert config.pixel_size_um == 1.0
    assert config.overlap_policy == "background"
    assert config.labels_path is None
    assert config.options == {}


@pytest.mark.parametrize(
    "overrides",
    [
        {"pixel_size_um": 0},
        {"pixel_size_um": float("nan")},
        {"max_pixels": 0},
        {"voxel_size_um": [1, 0, 1]},
        {"origin_um": [0, float("inf"), 0]},
        {"max_iter": 0},
        {"n_neighbors": 0},
        {"geometry_mode": "labels_3d"},
        {
            "geometry_mode": "labels_3d",
            "labels_path": "cells.npy",
            "voxel_size_um": [0.25, 0.5, 1.5],
        },
        {"labels_path": "cells.npy"},
        {"unknown_option": True},
        {"overlap_policy": "overwrite"},
    ],
)
def test_pciseq_rejects_invalid_geometry_and_model_settings(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _config(**overrides)


@pytest.mark.parametrize(
    "option",
    ["save_data", "realtime_viewer", "exclude_genes", "voxel_size", "nNeighbors"],
)
def test_pciseq_rejects_options_that_break_headless_or_id_contract(option: str) -> None:
    with pytest.raises(ValidationError, match="integration-managed"):
        _config(options={option: True})


def test_pciseq_accepts_explicit_real_3d_grid() -> None:
    config = _config(
        geometry_mode="labels_3d",
        labels_path="cells.npy",
        origin_um=[-10, 5, 0],
        voxel_size_um=[0.25, 0.25, 1.5],
        options={"Inefficiency": 0.3},
    )
    assert config.voxel_size_um == (0.25, 0.25, 1.5)
    assert config.options == {"Inefficiency": 0.3}
