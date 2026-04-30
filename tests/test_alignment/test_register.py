"""Tests for alignment registration helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from merxen.alignment.register import (
    _resolve_device,
    _spateo_pairwise_kwargs,
    register_pair,
    transform_xy_for_result,
)
from merxen.config import SpateoAlignmentConfig


def _fake_sdata(offset: tuple[float, float]) -> SimpleNamespace:
    coords = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
        ]
    )
    ox, oy = offset
    shapes = gpd.GeoDataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "geometry": [
                box(x + ox - 1, y + oy - 1, x + ox + 1, y + oy + 1) for x, y in coords
            ],
        }
    )
    table = ad.AnnData(
        X=np.array(
            [
                [10.0, 0.0, 1.0],
                [0.0, 10.0, 1.0],
                [2.0, 8.0, 1.0],
                [8.0, 2.0, 1.0],
            ]
        ),
        obs=pd.DataFrame({"cell_id": ["a", "b", "c", "d"]}, index=["a", "b", "c", "d"]),
        var=pd.DataFrame({"gene": ["A", "B", "C"]}, index=["A", "B", "C"]),
    )
    return SimpleNamespace(shapes={"cells": shapes}, tables={"table": table})


def test_register_pair_fits_transform_with_injected_spateo_runner() -> None:
    """Registration should fit usable affine/RBF transforms from Spateo outputs."""
    xenium = _fake_sdata((0.0, 0.0))
    merscope = _fake_sdata((5.0, -2.0))

    def runner(
        fixed: ad.AnnData,
        moving: ad.AnnData,
        config: object,
    ) -> tuple[ad.AnnData, ad.AnnData]:
        del config
        fixed = fixed.copy()
        moving = moving.copy()
        fixed.obsm["align_spatial"] = fixed.obsm["spatial"].copy()
        fixed.obsm["align_spatial_rigid"] = fixed.obsm["spatial"].copy()
        moving.obsm["align_spatial_rigid"] = moving.obsm["spatial"] + np.array(
            [-5.0, 2.0]
        )
        moving.obsm["align_spatial_nonrigid"] = moving.obsm[
            "align_spatial_rigid"
        ].copy()
        moving.obsm["align_spatial"] = moving.obsm["align_spatial_nonrigid"].copy()
        return fixed, moving

    cfg = {
        "pair_id": "pair1",
        "merscope_zarr_path": "merscope.zarr",
        "xenium_zarr_path": "xenium.zarr",
        "output_dir": "align_out",
    }
    result = register_pair(merscope, xenium, cfg, spateo_runner=runner)
    out = transform_xy_for_result(result, np.array([[5.0, -2.0], [15.0, 8.0]]))

    assert result.metadata["method"] == "spateo_morpho_align"
    assert result.coordinate_tables is not None
    assert set(result.coordinate_tables) == {"merscope", "xenium"}
    np.testing.assert_allclose(out, np.array([[0.0, 0.0], [10.0, 10.0]]), atol=1e-6)


def test_register_pair_runs_opt_in_param_grid() -> None:
    """Spateo parameter tuning should evaluate every configured candidate."""
    xenium = _fake_sdata((0.0, 0.0))
    merscope = _fake_sdata((5.0, -2.0))
    calls: list[int] = []

    def runner(
        fixed: ad.AnnData,
        moving: ad.AnnData,
        config: object,
    ) -> tuple[ad.AnnData, ad.AnnData]:
        calls.append(config.partial_robust_level)  # type: ignore[attr-defined]
        fixed = fixed.copy()
        moving = moving.copy()
        fixed.obsm["align_spatial"] = fixed.obsm["spatial"].copy()
        fixed.obsm["align_spatial_rigid"] = fixed.obsm["spatial"].copy()
        moving.obsm["align_spatial_rigid"] = moving.obsm["spatial"] + np.array(
            [-5.0, 2.0]
        )
        moving.obsm["align_spatial_nonrigid"] = moving.obsm[
            "align_spatial_rigid"
        ].copy()
        moving.obsm["align_spatial"] = moving.obsm["align_spatial_nonrigid"].copy()
        return fixed, moving

    cfg = {
        "pair_id": "pair1",
        "merscope_zarr_path": "merscope.zarr",
        "xenium_zarr_path": "xenium.zarr",
        "output_dir": "align_out",
        "spateo": {
            "tune": True,
            "param_grid": [{}, {"partial_robust_level": 75}],
        },
    }
    result = register_pair(merscope, xenium, cfg, spateo_runner=runner)

    assert calls == [50, 75]
    assert len(result.metadata["tuning"]) == 2


def test_spateo_kwargs_map_sampling_to_installed_api() -> None:
    """The wrapper should adapt SABench-style sampling params to Spateo APIs."""

    class PairwiseWithBatchSize:
        def __init__(
            self: PairwiseWithBatchSize,
            *,
            batch_size: int,
            beta: float,
        ) -> None:
            del batch_size, beta

    cfg = SpateoAlignmentConfig(n_sampling=123, beta=2.0)
    kwargs = _spateo_pairwise_kwargs(cfg, PairwiseWithBatchSize)

    assert kwargs == {"beta": 2.0, "batch_size": 123}


def test_spateo_device_aliases_use_cuda_visible_device_ids() -> None:
    """Spateo expects CUDA_VISIBLE_DEVICES-style GPU ids, not torch aliases."""
    assert _resolve_device("cuda") == "0"
    assert _resolve_device("cuda:0") == "0"
    assert _resolve_device("CUDA:1") == "1"
    assert _resolve_device(" 0 ") == "0"
    assert _resolve_device("cpu") == "cpu"


def test_spateo_auto_device_uses_gpu_index_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto device selection should pass Spateo a usable GPU id."""
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert _resolve_device("auto") == "0"


def test_spateo_alignment_defaults_are_gpu_conservative() -> None:
    """Default alignment settings should fit large pairs more comfortably."""
    cfg = SpateoAlignmentConfig()

    assert cfg.dtype == "float32"
    assert cfg.n_sampling == 1000
    assert cfg.chunk_capacity == 1
    assert cfg.n_top_genes == 100
    assert cfg.k == 50
    assert cfg.max_nonrigid_anchors == 5000
