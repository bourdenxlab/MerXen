"""Path-handling tests for enrichment staging helpers."""

from __future__ import annotations

import json
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
    CELLPOSE_NUCLEI_SHAPE_NAME,
    MERSCOPE_OLD_SHAPE_NAME,
    MERSCOPE_ZPROJ_IMAGE_NAME,
    MOSAIK_CELLPOSE_SHAPE_NAME,
    MOSAIK_PROSEG_SHAPE_NAME,
    ORIGINAL_TABLE_NAME,
    _cellpose_gdf_from_mask,
    _is_already_enriched,
    _read_latest_zarr_for_enrichment,
    _remove_partial_enrichment_artifacts_from_zarr_path,
    _remove_path,
    enrich_single_latest,
)


def test_cellpose_polygon_export_preserves_sparse_mask_labels(
    tmp_path: Path,
) -> None:
    """Filtered masks must retain their original positive label identities."""
    mask = np.zeros((8, 8), dtype=np.uint32)
    mask[1:3, 1:3] = 2
    mask[5:7, 5:7] = 9
    mask_path = tmp_path / "cellpose.npy"
    np.save(mask_path, mask)

    gdf = _cellpose_gdf_from_mask(
        mask_path,
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        "P1_MERSCOPE",
        polygon_n_jobs=1,
        polygon_show_progress=False,
    )

    assert gdf.index.tolist() == [2, 9]
    assert gdf["instance_id"].tolist() == [2, 9]
    assert gdf["source_cell_id"].tolist() == ["cellpose_2", "cellpose_9"]


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
            CELLPOSE_NUCLEI_SHAPE_NAME: object(),
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
        nuclei_mask_path=mask,
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
    assert CELLPOSE_NUCLEI_SHAPE_NAME in dst.shapes
    assert MERSCOPE_OLD_SHAPE_NAME in dst.shapes
    assert MERSCOPE_ZPROJ_IMAGE_NAME in dst.images
    assert ORIGINAL_TABLE_NAME in dst.tables
    assert not any("__enrich_tmp" in path.name for path in target.parent.iterdir())
    assert not any("pre_enrich_backup" in path.name for path in target.parent.iterdir())


def test_enrich_single_latest_adds_missing_nuclei_without_rebuilding_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A legacy enriched zarr should gain nuclei without losing derived tables."""
    target = tmp_path / "results" / "latest" / "latest_spatialdata.zarr"
    target.mkdir(parents=True)
    latest = tmp_path / "latest_input.zarr"
    latest.symlink_to(target, target_is_directory=True)
    mask = tmp_path / "nuclei.npy"
    np.save(mask, np.ones((4, 4), dtype=np.uint32))

    proseg_shape = gpd.GeoDataFrame(
        {"cell_id": ["1"], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        geometry="geometry",
    )
    preserved_tables = {
        ORIGINAL_TABLE_NAME: object(),
        "table_MOSAIK_proseg": object(),
        "table_MOSAIK_proseg_clustering_squidpy": object(),
    }
    dst = SimpleNamespace(
        shapes={
            MOSAIK_PROSEG_SHAPE_NAME: proseg_shape,
            MOSAIK_CELLPOSE_SHAPE_NAME: object(),
            MERSCOPE_OLD_SHAPE_NAME: object(),
        },
        images={MERSCOPE_ZPROJ_IMAGE_NAME: object()},
        tables=dict(preserved_tables),
    )
    writes: list[tuple[str, str, bool]] = []

    monkeypatch.setattr(
        "merxen.enrichment.enrich.sd.read_zarr",
        lambda path: dst,
    )
    monkeypatch.setattr(
        "merxen.enrichment.enrich._dataset_cellpose_transform",
        lambda config: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    monkeypatch.setattr(
        "merxen.enrichment.enrich._cellpose_gdf_from_mask",
        lambda *args, **kwargs: gpd.GeoDataFrame(
            {"cell_id": ["nucleus_1"], "geometry": [box(0.1, 0.1, 0.9, 0.9)]},
            geometry="geometry",
        ),
    )

    def _fake_write_or_replace(
        sdata_obj: SimpleNamespace,
        key: str,
        element_type: str,
        value: object,
        *,
        overwrite: bool = True,
    ) -> bool:
        writes.append((key, element_type, overwrite))
        getattr(sdata_obj, element_type)[key] = value
        return True

    monkeypatch.setattr(
        "merxen.enrichment.enrich.write_or_replace_element",
        _fake_write_or_replace,
    )
    monkeypatch.setattr(
        "merxen.enrichment.enrich._remove_per_shape_assignment_tables",
        lambda sdata: pytest.fail("incremental nuclei enrichment rebuilt tables"),
    )
    monkeypatch.setattr(
        "merxen.enrichment.enrich._load_original_source",
        lambda config: pytest.fail("incremental nuclei enrichment reloaded source"),
    )

    cfg = EnrichmentConfig(
        dataset_name="P7513_IF_MERSCOPE",
        platform="MERSCOPE",
        latest_zarr_path=latest,
        mask_path=mask,
        nuclei_mask_path=mask,
        original_data_path=tmp_path / "unused_source.zarr",
        output_dir=tmp_path / "enrich_out",
        persistent_output_path=target,
    )

    out = enrich_single_latest(cfg)

    assert out == target
    assert writes == [(CELLPOSE_NUCLEI_SHAPE_NAME, "shapes", False)]
    assert CELLPOSE_NUCLEI_SHAPE_NAME in dst.shapes
    assert dst.tables == preserved_tables


def test_incomplete_enrichment_rebuilds_partial_elements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Incomplete enrichment should overwrite partial artifacts, not preserve them."""
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
        shapes={
            "cell_boundaries": proseg_shape,
            MOSAIK_PROSEG_SHAPE_NAME: object(),
            MOSAIK_CELLPOSE_SHAPE_NAME: object(),
            CELLPOSE_NUCLEI_SHAPE_NAME: object(),
            MERSCOPE_OLD_SHAPE_NAME: object(),
        },
        images={MERSCOPE_ZPROJ_IMAGE_NAME: object()},
        tables={"table_MOSAIK_proseg": object()},
    )
    src = SimpleNamespace(
        shapes={"original_cells": original_shape},
        images={MERSCOPE_ZPROJ_IMAGE_NAME: projection},
        tables={"table": original_table},
    )
    writes: list[tuple[str, str, bool]] = []

    def _fake_read_zarr(path: Path) -> SimpleNamespace:
        resolved = Path(path).resolve()
        if resolved == target.resolve():
            return dst
        if resolved == original.resolve():
            return src
        raise AssertionError(f"unexpected read_zarr path: {path}")

    def _fake_write_or_replace(
        sdata_obj: SimpleNamespace,
        key: str,
        element_type: str,
        value: object,
        *,
        overwrite: bool = True,
    ) -> bool:
        writes.append((key, element_type, overwrite))
        getattr(sdata_obj, element_type)[key] = value
        return True

    monkeypatch.setattr("merxen.enrichment.enrich.sd.read_zarr", _fake_read_zarr)
    monkeypatch.setattr(
        "merxen.enrichment.enrich.write_or_replace_element",
        _fake_write_or_replace,
    )
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
        nuclei_mask_path=mask,
        original_data_path=original,
        output_dir=tmp_path / "enrich_out",
        persistent_output_path=target,
    )

    enrich_single_latest(cfg)

    assert "table_MOSAIK_proseg" not in dst.tables
    assert ORIGINAL_TABLE_NAME in dst.tables
    assert writes
    assert all(overwrite for _, _, overwrite in writes)


def test_partial_enrichment_cleanup_removes_rebuildable_zarr_artifacts(
    tmp_path: Path,
) -> None:
    """Corrupt partial enrichment elements should be deleted with stale metadata."""
    zarr_path = tmp_path / "latest_spatialdata.zarr"
    for rel_path in [
        f"images/{MERSCOPE_ZPROJ_IMAGE_NAME}/s0",
        f"shapes/{MOSAIK_CELLPOSE_SHAPE_NAME}",
        f"shapes/{CELLPOSE_NUCLEI_SHAPE_NAME}",
        f"shapes/{MOSAIK_PROSEG_SHAPE_NAME}",
        f"shapes/{MERSCOPE_OLD_SHAPE_NAME}",
        "shapes/cell_boundaries",
        f"tables/{ORIGINAL_TABLE_NAME}",
        "tables/table_MOSAIK_cellpose",
        "tables/table",
    ]:
        (zarr_path / rel_path).mkdir(parents=True)

    root_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "consolidated_metadata": {
            "kind": "inline",
            "must_understand": False,
            "metadata": {
                f"images/{MERSCOPE_ZPROJ_IMAGE_NAME}": {"attributes": {}},
                f"images/{MERSCOPE_ZPROJ_IMAGE_NAME}/s0": {"attributes": {}},
                f"shapes/{MOSAIK_CELLPOSE_SHAPE_NAME}": {"attributes": {}},
                f"shapes/{CELLPOSE_NUCLEI_SHAPE_NAME}": {"attributes": {}},
                f"shapes/{MOSAIK_PROSEG_SHAPE_NAME}": {"attributes": {}},
                f"shapes/{MERSCOPE_OLD_SHAPE_NAME}": {"attributes": {}},
                "shapes/cell_boundaries": {"attributes": {"keep": True}},
                f"tables/{ORIGINAL_TABLE_NAME}": {"attributes": {}},
                "tables/table_MOSAIK_cellpose": {"attributes": {}},
                "tables/table": {"attributes": {"keep": True}},
            },
        },
    }
    (zarr_path / "zarr.json").write_text(json.dumps(root_metadata))

    _remove_partial_enrichment_artifacts_from_zarr_path(zarr_path, "MERSCOPE")

    assert not (zarr_path / "images" / MERSCOPE_ZPROJ_IMAGE_NAME).exists()
    assert not (zarr_path / "shapes" / MOSAIK_CELLPOSE_SHAPE_NAME).exists()
    assert not (zarr_path / "shapes" / CELLPOSE_NUCLEI_SHAPE_NAME).exists()
    assert not (zarr_path / "shapes" / MOSAIK_PROSEG_SHAPE_NAME).exists()
    assert not (zarr_path / "shapes" / MERSCOPE_OLD_SHAPE_NAME).exists()
    assert not (zarr_path / "tables" / ORIGINAL_TABLE_NAME).exists()
    assert not (zarr_path / "tables" / "table_MOSAIK_cellpose").exists()
    assert (zarr_path / "shapes" / "cell_boundaries").exists()
    assert (zarr_path / "tables" / "table").exists()

    metadata = json.loads((zarr_path / "zarr.json").read_text())[
        "consolidated_metadata"
    ]["metadata"]
    assert f"images/{MERSCOPE_ZPROJ_IMAGE_NAME}" not in metadata
    assert f"images/{MERSCOPE_ZPROJ_IMAGE_NAME}/s0" not in metadata
    assert f"shapes/{MOSAIK_CELLPOSE_SHAPE_NAME}" not in metadata
    assert "tables/table_MOSAIK_cellpose" not in metadata
    assert "shapes/cell_boundaries" in metadata
    assert "tables/table" in metadata


def test_read_latest_zarr_recovers_from_partial_enrichment_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unreadable partial enrichment metadata should be cleaned before retrying."""
    target = tmp_path / "latest_spatialdata.zarr"
    target.mkdir()
    recovered = object()
    read_calls: list[Path] = []
    cleanup_calls: list[tuple[Path, str]] = []

    def _fake_read_zarr(path: Path) -> object:
        read_calls.append(Path(path))
        if len(read_calls) == 1:
            raise KeyError("ome")
        return recovered

    def _fake_cleanup(path: Path, platform: str) -> None:
        cleanup_calls.append((Path(path), platform))

    monkeypatch.setattr("merxen.enrichment.enrich.sd.read_zarr", _fake_read_zarr)
    monkeypatch.setattr(
        "merxen.enrichment.enrich._remove_partial_enrichment_artifacts_from_zarr_path",
        _fake_cleanup,
    )

    result = _read_latest_zarr_for_enrichment(target, "MERSCOPE", "P1_MERSCOPE")

    assert result is recovered
    assert read_calls == [target, target]
    assert cleanup_calls == [(target, "MERSCOPE")]
