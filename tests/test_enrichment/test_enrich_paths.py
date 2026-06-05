"""Path-handling tests for enrichment staging helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from merxen.config import EnrichmentConfig
from merxen.enrichment.enrich import (
    MERSCOPE_OLD_SHAPE_NAME,
    MERSCOPE_ZPROJ_IMAGE_NAME,
    MOSAIK_CELLPOSE_SHAPE_NAME,
    MOSAIK_PROSEG_SHAPE_NAME,
    ORIGINAL_TABLE_NAME,
    _is_already_enriched,
    _remove_path,
    enrich_single_latest,
)


def test_remove_path_unlinks_directory_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    """Removing a staged symlink should not delete the upstream directory."""
    target = tmp_path / "target.zarr"
    target.mkdir()
    (target / "marker.txt").write_text("keep me")

    staged = tmp_path / "staged.zarr"
    staged.symlink_to(target, target_is_directory=True)

    _remove_path(staged)

    assert not staged.exists()
    assert not staged.is_symlink()
    assert target.exists()
    assert (target / "marker.txt").read_text() == "keep me"


def test_remove_path_deletes_real_directory_tree(tmp_path: Path) -> None:
    """Real directories should still be removed recursively."""
    out_dir = tmp_path / "out.zarr"
    out_dir.mkdir()
    (out_dir / "marker.txt").write_text("remove me")

    _remove_path(out_dir)

    assert not out_dir.exists()


def test_remove_path_ignores_missing_entries_during_rmtree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A partially-disappeared tree should not crash cleanup."""
    out_dir = tmp_path / "out.zarr"
    out_dir.mkdir()

    def _fake_rmtree(path: Path) -> None:
        raise FileNotFoundError("simulated race while removing a child entry")

    monkeypatch.setattr(shutil, "rmtree", _fake_rmtree)

    _remove_path(out_dir)


def test_is_already_enriched_checks_platform_specific_merscope_layers() -> None:
    """MERSCOPE completeness should depend on platform-specific shapes/images."""
    sdata = SimpleNamespace(
        shapes={
            MOSAIK_PROSEG_SHAPE_NAME: object(),
            MOSAIK_CELLPOSE_SHAPE_NAME: object(),
            MERSCOPE_OLD_SHAPE_NAME: object(),
        },
        tables={ORIGINAL_TABLE_NAME: object()},
        images={MERSCOPE_ZPROJ_IMAGE_NAME: object()},
    )

    assert _is_already_enriched(sdata, "MERSCOPE")

    del sdata.images[MERSCOPE_ZPROJ_IMAGE_NAME]

    assert not _is_already_enriched(sdata, "MERSCOPE")


def test_enrich_single_latest_writes_elements_in_place(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Enrichment should update latest in place without temp full-zarr rewrites."""
    target = tmp_path / "results" / "latest" / "latest_spatialdata.zarr"
    target.mkdir(parents=True)
    latest = tmp_path / "latest_input.zarr"
    latest.symlink_to(target, target_is_directory=True)
    original = tmp_path / "source.zarr"
    original.mkdir()
    mask = tmp_path / "mask.npy"
    np.save(mask, np.ones((4, 4), dtype=np.uint32))

    proseg_shape = gpd.GeoDataFrame(
        {"cell_id": ["1"], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        geometry="geometry",
    )
    original_shape = gpd.GeoDataFrame(
        {"cell_id": ["old"], "geometry": [box(1.0, 1.0, 2.0, 2.0)]},
        geometry="geometry",
    )
    original_table = ad.AnnData(
        X=np.ones((1, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["old"]),
        var=pd.DataFrame(index=["gene"]),
    )
    projection = xr.DataArray(
        np.ones((1, 4, 4), dtype=np.uint16),
        dims=("c", "y", "x"),
        coords={"c": ["DAPI"]},
    )
    dst = SimpleNamespace(
        shapes={"cell_boundaries": proseg_shape},
        images={},
        tables={},
    )
    src = SimpleNamespace(
        shapes={"original_cells": original_shape},
        images={MERSCOPE_ZPROJ_IMAGE_NAME: projection},
        tables={"table": original_table},
    )

    def _fake_read_zarr(path: Path) -> SimpleNamespace:
        resolved = Path(path).resolve()
        if resolved == target.resolve():
            return dst
        if resolved == original.resolve():
            return src
        raise AssertionError(f"unexpected read_zarr path: {path}")

    monkeypatch.setattr("merxen.enrichment.enrich.sd.read_zarr", _fake_read_zarr)
    monkeypatch.setattr(
        "merxen.enrichment.enrich._dataset_cellpose_transform",
        lambda config: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    monkeypatch.setattr(
        "merxen.enrichment.enrich._cellpose_gdf_from_mask",
        lambda *args, **kwargs: gpd.GeoDataFrame(
            {"cell_id": ["cp"], "geometry": [box(2.0, 2.0, 3.0, 3.0)]},
            geometry="geometry",
        ),
    )

    cfg = EnrichmentConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        latest_zarr_path=latest,
        mask_path=mask,
        original_data_path=original,
        output_dir=tmp_path / "enrich_out",
        persistent_output_path=target,
    )

    out = enrich_single_latest(cfg)

    assert out == target
    assert latest.is_symlink()
    assert latest.resolve() == target.resolve()
    assert MOSAIK_PROSEG_SHAPE_NAME in dst.shapes
    assert MOSAIK_CELLPOSE_SHAPE_NAME in dst.shapes
    assert MERSCOPE_OLD_SHAPE_NAME in dst.shapes
    assert MERSCOPE_ZPROJ_IMAGE_NAME in dst.images
    assert ORIGINAL_TABLE_NAME in dst.tables
    assert not any("__enrich_tmp" in path.name for path in target.parent.iterdir())
    assert not any("pre_enrich_backup" in path.name for path in target.parent.iterdir())
