"""Tests for Cellpose-mask image-channel quantification."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
import xarray as xr
from shapely.geometry import box
from spatialdata.models import Image2DModel, ShapesModel, TableModel

import merxen.mask_image_quantification as quantification_module
from merxen.config import MaskImageQuantificationConfig, MerscopeBeadRemovalConfig
from merxen.io.bead_removal.provenance import (
    IMAGE_PREPROCESSING_ATTR,
    build_image_preprocessing_record,
)
from merxen.mask_image_quantification import (
    HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX,
    HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY,
    IMAGE_QUANTIFICATION_STATS,
    MASK_IMAGE_QUANTIFICATION_TABLE_KEY,
    build_mask_image_quantification_table,
    join_cellpose_image_quantification_to_hybrid,
    run_mask_image_quantification,
)


def _image(values: np.ndarray, channels: list[str]) -> xr.DataArray:
    return xr.DataArray(values, dims=("y", "x", "c"), coords={"c": channels})


def test_quantifies_exact_stats_for_tiny_mask() -> None:
    """The core table should contain exact per-label/channel summary stats."""
    mask = np.array(
        [
            [0, 1, 1, 2],
            [0, 1, 2, 2],
            [3, 3, 0, 2],
        ],
        dtype=np.uint32,
    )
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    values = np.stack([base, base + 100.0], axis=-1)
    sdata = SimpleNamespace(images={"image": _image(values, ["DAPI", "RNA"])})

    result = build_mask_image_quantification_table(
        sdata,
        mask,
        "P1_MERSCOPE",
        tile_size=2,
    )
    table = result.table

    assert table.obs_names.tolist() == ["1", "2", "3"]
    assert table.obs["instance_id"].tolist() == [1, 2, 3]
    assert table.obs["source_cell_id"].tolist() == [
        "cellpose_1",
        "cellpose_2",
        "cellpose_3",
    ]
    assert table.obs["label_id"].tolist() == [1, 2, 3]
    assert table.obs["mask_pixel_count"].tolist() == [3, 4, 2]

    df = table.to_df()
    np.testing.assert_allclose(
        df.loc["1", "image__DAPI__min"],
        1.0,
    )
    np.testing.assert_allclose(
        df.loc["1", "image__DAPI__median"],
        2.0,
    )
    np.testing.assert_allclose(
        df.loc["1", "image__DAPI__mean"],
        (1.0 + 2.0 + 5.0) / 3.0,
    )
    np.testing.assert_allclose(
        df.loc["1", "image__DAPI__max"],
        5.0,
    )
    np.testing.assert_allclose(
        df.loc["1", "image__DAPI__iqr"],
        2.0,
    )
    np.testing.assert_allclose(
        df.loc["1", "image__RNA__median"],
        102.0,
    )


def test_ignores_nan_pixels_for_statistics() -> None:
    """NaN image pixels should be ignored without changing mask pixel counts."""
    mask = np.array([[1, 1], [2, 2]], dtype=np.uint32)
    values = np.array(
        [
            [[1.0], [np.nan]],
            [[10.0], [14.0]],
        ],
        dtype=np.float64,
    )
    sdata = SimpleNamespace(images={"img": _image(values, ["DAPI"])})

    table = build_mask_image_quantification_table(sdata, mask, "P1").table
    df = table.to_df()

    assert table.obs["mask_pixel_count"].tolist() == [2, 2]
    np.testing.assert_allclose(df.loc["1", "img__DAPI__mean"], 1.0)
    np.testing.assert_allclose(df.loc["2", "img__DAPI__median"], 12.0)


def test_multiple_images_and_channels_are_wide_with_metadata() -> None:
    """Every image element/channel/stat should become a wide matrix feature."""
    mask = np.array([[1, 2], [1, 2]], dtype=np.uint32)
    img1 = _image(
        np.stack(
            [
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[5.0, 6.0], [7.0, 8.0]]),
            ],
            axis=-1,
        ),
        ["DAPI", "RNA"],
    )
    img2 = _image(np.ones((2, 2, 1), dtype=np.float64), ["IF"])
    sdata = SimpleNamespace(images={"morphology": img1, "if_panel": img2})

    table = build_mask_image_quantification_table(sdata, mask, "P1").table

    assert table.n_vars == 15
    assert "morphology__RNA__iqr" in table.var_names
    assert "if_panel__IF__max" in table.var_names
    assert table.var.loc["morphology__RNA__iqr", "image_key"] == "morphology"
    assert table.var.loc["morphology__RNA__iqr", "channel"] == "RNA"
    assert table.var.loc["morphology__RNA__iqr", "statistic"] == "iqr"


def test_private_viewer_pyramids_are_excluded_from_quantification() -> None:
    """Generated viewer pyramids must not be compared with full-size masks."""
    mask = np.array([[1, 2], [1, 2]], dtype=np.uint32)
    source = _image(
        np.arange(4, dtype=np.float64).reshape(2, 2, 1),
        ["DAPI"],
    )
    viewer_pyramid = _image(
        np.ones((1, 1, 1), dtype=np.float64),
        ["DAPI"],
    )
    sdata = SimpleNamespace(
        images={
            "_napari_compare_imgpyr__morphology_focus__ds4": viewer_pyramid,
            "morphology_focus": source,
        }
    )

    result = build_mask_image_quantification_table(sdata, mask, "P7113_XENIUM")

    assert result.table.n_vars == len(IMAGE_QUANTIFICATION_STATS)
    assert result.table.var["image_key"].unique().tolist() == ["morphology_focus"]
    assert [image["image_key"] for image in result.summary["images"]] == [
        "morphology_focus"
    ]


def test_shape_mismatch_fails_fast() -> None:
    """Image elements must match the final Cellpose mask pixel grid."""
    mask = np.ones((2, 2), dtype=np.uint32)
    sdata = SimpleNamespace(
        images={"bad": _image(np.ones((3, 2, 1), dtype=np.float64), ["DAPI"])}
    )

    with pytest.raises(ValueError, match="does not match Cellpose mask shape"):
        build_mask_image_quantification_table(sdata, mask, "P1")


def test_run_mask_image_quantification_writes_table_and_sidecars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The run wrapper should persist a table and sidecar files."""
    latest = tmp_path / "latest.zarr"
    latest.mkdir()
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.array([[1, 1], [2, 0]], dtype=np.uint32))

    sdata = SimpleNamespace(
        images={
            "img": _image(np.arange(4, dtype=np.float64).reshape(2, 2, 1), ["DAPI"])
        },
        tables={},
    )
    writes: list[tuple[str, str]] = []

    def _fake_write(
        sdata_obj: SimpleNamespace,
        key: str,
        element_type: str,
        value: object,
        *,
        overwrite: bool = True,
    ) -> bool:
        del overwrite
        writes.append((key, element_type))
        getattr(sdata_obj, element_type)[key] = value
        return True

    monkeypatch.setattr(
        "merxen.mask_image_quantification.sd.read_zarr", lambda _: sdata
    )
    monkeypatch.setattr(
        "merxen.mask_image_quantification.TableModel.parse",
        lambda table, **_: table,
    )
    monkeypatch.setattr(
        "merxen.mask_image_quantification.write_or_replace_element",
        _fake_write,
    )

    cfg = MaskImageQuantificationConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        latest_zarr_path=latest,
        mask_path=mask_path,
        output_dir=tmp_path / "quant_out",
    )

    outputs = run_mask_image_quantification(cfg)

    assert writes == [(MASK_IMAGE_QUANTIFICATION_TABLE_KEY, "tables")]
    assert MASK_IMAGE_QUANTIFICATION_TABLE_KEY in sdata.tables
    assert outputs["wide_matrix"].exists()
    assert outputs["feature_metadata"].exists()
    assert outputs["summary"].exists()
    wide = pd.read_parquet(outputs["wide_matrix"])
    assert wide.index.tolist() == ["1", "2"]


def test_cellpose_image_features_join_to_hybrid_by_instance_id() -> None:
    image_table = ad.AnnData(
        X=np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=float),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray([2, 1], dtype=np.uint64),
                "mask_pixel_count": [8, 5],
            },
            index=["2", "1"],
        ),
        var=pd.DataFrame(
            index=[
                "morphology_focus__ATP1A1/CD45/E-Cadherin__mean",
                "img__DAPI__max",
            ]
        ),
    )
    image_table.uns["spatialdata_attrs"] = {"instance_key": "instance_id"}
    hybrid_table = ad.AnnData(
        X=np.asarray([[1], [2], [3]], dtype=np.int64),
        obs=pd.DataFrame(
            {"instance_id": np.asarray([1, 2, 3], dtype=np.uint64)},
            index=["1", "2", "3"],
        ),
        var=pd.DataFrame(index=["GeneA"]),
    )
    hybrid_table.uns["spatialdata_attrs"] = {"instance_key": "instance_id"}

    result = join_cellpose_image_quantification_to_hybrid(
        image_table,
        hybrid_table,
    )

    np.testing.assert_array_equal(result.table.X, hybrid_table.X)
    joined = result.table.obsm[HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY]
    np.testing.assert_allclose(joined[0], [20.0, 21.0])
    np.testing.assert_allclose(joined[1], [10.0, 11.0])
    assert np.isnan(joined[2]).all()
    assert result.table.obs[
        f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}"
        "morphology_focus__ATP1A1_CD45_E-Cadherin__mean"
    ].tolist()[:2] == [20.0, 10.0]
    marker = result.table.uns[HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY]
    assert marker["feature_names"][0] == (
        "morphology_focus__ATP1A1/CD45/E-Cadherin__mean"
    )
    assert marker["obs_columns"][0] == (
        f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}"
        "morphology_focus__ATP1A1_CD45_E-Cadherin__mean"
    )
    assert result.summary["n_matched_cells"] == 2
    assert result.summary["n_unmatched_hybrid_cells"] == 1

    parsed_input = result.table.copy()
    parsed_input.uns.pop("spatialdata_attrs", None)
    parsed_input.obs["region"] = pd.Categorical(
        ["MOSAIK_proseg_hybrid"] * parsed_input.n_obs
    )
    parsed = TableModel.parse(
        parsed_input,
        region="MOSAIK_proseg_hybrid",
        region_key="region",
        instance_key="instance_id",
    )
    assert HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY in parsed.obsm


def test_hybrid_image_obs_column_sanitization_avoids_collisions() -> None:
    """Sanitized channel names remain unique if slash replacement collides."""
    image_table = ad.AnnData(
        X=np.asarray([[10.0, 20.0]], dtype=float),
        obs=pd.DataFrame(
            {"instance_id": np.asarray([1], dtype=np.uint64)},
            index=["1"],
        ),
        var=pd.DataFrame(index=["img__A/B__mean", "img__A_B__mean"]),
    )
    hybrid_table = ad.AnnData(
        X=np.asarray([[1]], dtype=np.int64),
        obs=pd.DataFrame(
            {"instance_id": np.asarray([1], dtype=np.uint64)},
            index=["1"],
        ),
        var=pd.DataFrame(index=["GeneA"]),
    )

    result = join_cellpose_image_quantification_to_hybrid(
        image_table,
        hybrid_table,
    )

    marker = result.table.uns[HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY]
    assert marker["obs_columns"] == [
        f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}img__A_B__mean",
        f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}img__A_B__mean_1",
    ]
    assert len(set(result.table.obs.columns)) == len(result.table.obs.columns)


def test_rejoining_hybrid_image_features_replaces_stale_obs_columns() -> None:
    """A repeated join replaces its reserved columns without suffixing names."""
    first_image_table = ad.AnnData(
        X=np.asarray([[10.0]], dtype=float),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray([1], dtype=np.uint64),
                "mask_pixel_count": [5],
            },
            index=["1"],
        ),
        var=pd.DataFrame(index=["morphology_focus__DAPI__mean"]),
    )
    hybrid_table = ad.AnnData(
        X=np.asarray([[1]], dtype=np.int64),
        obs=pd.DataFrame(
            {"instance_id": np.asarray([1], dtype=np.uint64)},
            index=["1"],
        ),
        var=pd.DataFrame(index=["GeneA"]),
    )
    first_join = join_cellpose_image_quantification_to_hybrid(
        first_image_table,
        hybrid_table,
    ).table
    first_join.obs[
        f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}removed_image__p62__mean"
    ] = 99.0

    second_image_table = ad.AnnData(
        X=np.asarray([[20.0, 30.0]], dtype=float),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray([1], dtype=np.uint64),
                "mask_pixel_count": [7],
            },
            index=["1"],
        ),
        var=pd.DataFrame(
            index=[
                "morphology_focus__DAPI__mean",
                "post_xenium_if__p62__mean",
            ]
        ),
    )
    result = join_cellpose_image_quantification_to_hybrid(
        second_image_table,
        first_join,
    )

    marker = result.table.uns[HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY]
    prefixed_columns = [
        str(column)
        for column in result.table.obs.columns
        if str(column).startswith(HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX)
    ]
    assert prefixed_columns == marker["obs_columns"]
    assert prefixed_columns == [
        f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}morphology_focus__DAPI__mean",
        f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}post_xenium_if__p62__mean",
    ]
    assert result.table.obs[prefixed_columns].iloc[0].tolist() == [20.0, 30.0]
    assert result.table.obs["cellpose_image_mask_pixel_count"].tolist() == [7.0]


def test_hybrid_join_completeness_rejects_orphaned_obs_columns() -> None:
    """A marker cannot hide stale columns left by an earlier image join."""
    image_table = ad.AnnData(
        X=np.asarray([[10.0]], dtype=float),
        obs=pd.DataFrame(
            {"instance_id": np.asarray([1], dtype=np.uint64)},
            index=["1"],
        ),
        var=pd.DataFrame(index=["morphology_focus__DAPI__mean"]),
    )
    hybrid_table = ad.AnnData(
        X=np.asarray([[1]], dtype=np.int64),
        obs=pd.DataFrame(
            {"instance_id": np.asarray([1], dtype=np.uint64)},
            index=["1"],
        ),
        var=pd.DataFrame(index=["GeneA"]),
    )
    joined = join_cellpose_image_quantification_to_hybrid(
        image_table,
        hybrid_table,
    ).table
    sdata = SimpleNamespace(
        tables={
            MASK_IMAGE_QUANTIFICATION_TABLE_KEY: image_table,
            "table_MOSAIK_proseg_hybrid": joined,
        }
    )

    assert quantification_module._hybrid_image_join_is_complete(
        sdata,
        image_table_key=MASK_IMAGE_QUANTIFICATION_TABLE_KEY,
    )
    joined.obs[
        f"{HYBRID_IMAGE_QUANTIFICATION_OBS_PREFIX}morphology_focus__DAPI__mean_1"
    ] = 10.0
    assert not quantification_module._hybrid_image_join_is_complete(
        sdata,
        image_table_key=MASK_IMAGE_QUANTIFICATION_TABLE_KEY,
    )


def test_run_quantification_persists_hybrid_feature_join(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    latest = tmp_path / "latest.zarr"
    latest.mkdir()
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.asarray([[1, 1], [2, 0]], dtype=np.uint32))
    hybrid = ad.AnnData(
        X=np.asarray([[5], [7]], dtype=np.int64),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray([1, 2], dtype=np.uint64),
                "region": pd.Categorical(["MOSAIK_proseg_hybrid"] * 2),
            },
            index=["1", "2"],
        ),
        var=pd.DataFrame(index=["GeneA"]),
    )
    hybrid.uns["spatialdata_attrs"] = {"instance_key": "instance_id"}
    sdata = SimpleNamespace(
        images={
            "img": _image(
                np.arange(4, dtype=np.float64).reshape(2, 2, 1),
                ["DAPI"],
            )
        },
        tables={"table_MOSAIK_proseg_hybrid": hybrid},
    )
    writes: list[str] = []

    def _fake_write(
        sdata_obj: SimpleNamespace,
        key: str,
        element_type: str,
        value: object,
        *,
        overwrite: bool = True,
    ) -> bool:
        del overwrite
        writes.append(key)
        getattr(sdata_obj, element_type)[key] = value
        return True

    monkeypatch.setattr(
        "merxen.mask_image_quantification.sd.read_zarr",
        lambda _: sdata,
    )
    monkeypatch.setattr(
        "merxen.mask_image_quantification.TableModel.parse",
        lambda table, **_: table,
    )
    monkeypatch.setattr(
        "merxen.mask_image_quantification.write_or_replace_element",
        _fake_write,
    )

    run_mask_image_quantification(
        MaskImageQuantificationConfig(
            dataset_name="P1_MERSCOPE",
            platform="MERSCOPE",
            latest_zarr_path=latest,
            mask_path=mask_path,
            output_dir=tmp_path / "quant_out",
        )
    )

    assert writes == [
        MASK_IMAGE_QUANTIFICATION_TABLE_KEY,
        "table_MOSAIK_proseg_hybrid",
    ]
    updated = sdata.tables["table_MOSAIK_proseg_hybrid"]
    assert HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY in updated.obsm
    np.testing.assert_array_equal(updated.X, np.asarray([[5], [7]]))


def _patch_in_memory_zarr(
    monkeypatch: pytest.MonkeyPatch,
    sdata: SimpleNamespace,
) -> list[str]:
    writes: list[str] = []

    def _fake_write(
        sdata_obj: SimpleNamespace,
        key: str,
        element_type: str,
        value: object,
        *,
        overwrite: bool = True,
    ) -> bool:
        del overwrite
        writes.append(key)
        getattr(sdata_obj, element_type)[key] = value
        return True

    monkeypatch.setattr(
        "merxen.mask_image_quantification.sd.read_zarr", lambda _: sdata
    )
    monkeypatch.setattr(
        "merxen.mask_image_quantification.TableModel.parse",
        lambda table, **_: table,
    )
    monkeypatch.setattr(
        "merxen.mask_image_quantification.write_or_replace_element",
        _fake_write,
    )
    return writes


def test_existing_quantification_is_reused_when_images_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A rerun with the same source images must not rewrite the table."""
    latest = tmp_path / "latest.zarr"
    latest.mkdir()
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.array([[1, 1], [2, 0]], dtype=np.uint32))
    sdata = SimpleNamespace(
        images={"img": _image(np.ones((2, 2, 1)), ["DAPI"])},
        tables={},
    )
    writes = _patch_in_memory_zarr(monkeypatch, sdata)
    cfg = MaskImageQuantificationConfig(
        dataset_name="P1_XENIUM",
        platform="XENIUM",
        latest_zarr_path=latest,
        mask_path=mask_path,
        output_dir=tmp_path / "quant_out",
    )

    run_mask_image_quantification(cfg)
    run_mask_image_quantification(cfg)

    assert writes == [MASK_IMAGE_QUANTIFICATION_TABLE_KEY]


def test_quantification_serializes_with_another_store_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store_lock_contention: Any,
) -> None:
    """Quantification waits for a writer holding the store lock through another
    staged symlink, and keeps the lock from its read until the table is written."""
    latest = tmp_path / "latest.zarr"
    latest.mkdir()
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.array([[1, 1], [2, 0]], dtype=np.uint32))
    sdata = SimpleNamespace(
        images={"img": _image(np.ones((2, 2, 1)), ["DAPI"])},
        tables={},
    )
    writes = _patch_in_memory_zarr(monkeypatch, sdata)

    store_lock_contention.run(
        lambda staged: run_mask_image_quantification(
            MaskImageQuantificationConfig(
                dataset_name="P1_XENIUM",
                platform="XENIUM",
                latest_zarr_path=staged,
                mask_path=mask_path,
                output_dir=tmp_path / "quant_out",
            )
        ),
        latest,
        reads=[(quantification_module.sd, "read_zarr")],
        persists=[(quantification_module, "write_or_replace_element")],
    )

    assert writes == [MASK_IMAGE_QUANTIFICATION_TABLE_KEY]


def test_newly_registered_image_triggers_requantification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Channels of an image added after quantification must not be skipped."""
    latest = tmp_path / "latest.zarr"
    latest.mkdir()
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.array([[1, 1], [2, 0]], dtype=np.uint32))
    sdata = SimpleNamespace(
        images={"morphology_focus": _image(np.ones((2, 2, 1)), ["DAPI"])},
        tables={},
    )
    writes = _patch_in_memory_zarr(monkeypatch, sdata)
    cfg = MaskImageQuantificationConfig(
        dataset_name="P1_XENIUM",
        platform="XENIUM",
        latest_zarr_path=latest,
        mask_path=mask_path,
        output_dir=tmp_path / "quant_out",
    )
    run_mask_image_quantification(cfg)

    sdata.images["post_xenium_if"] = _image(
        np.arange(4, dtype=np.float64).reshape(2, 2, 1),
        ["p62"],
    )
    run_mask_image_quantification(cfg)

    assert writes == [MASK_IMAGE_QUANTIFICATION_TABLE_KEY] * 2
    table = sdata.tables[MASK_IMAGE_QUANTIFICATION_TABLE_KEY]
    assert set(table.var["image_key"]) == {"morphology_focus", "post_xenium_if"}
    assert "post_xenium_if__p62__mean" in table.var_names


def test_aligned_counterpart_image_is_not_quantified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An aligned image lives on the counterpart grid, not this mask's grid.

    Alignment materializes ``MERSCOPE_z_projection_aligned_nonrigid`` into the
    MERSCOPE store after quantification has run. A rerun must neither treat it
    as a newly registered image nor try to quantify it over the native masks.
    """
    latest = tmp_path / "latest.zarr"
    latest.mkdir()
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.array([[1, 1], [2, 0]], dtype=np.uint32))
    sdata = SimpleNamespace(
        images={"MERSCOPE_z_projection": _image(np.ones((2, 2, 1)), ["DAPI"])},
        tables={},
    )
    writes = _patch_in_memory_zarr(monkeypatch, sdata)
    cfg = MaskImageQuantificationConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        latest_zarr_path=latest,
        mask_path=mask_path,
        output_dir=tmp_path / "quant_out",
    )
    run_mask_image_quantification(cfg)

    sdata.images["MERSCOPE_z_projection_aligned_nonrigid"] = _image(
        np.ones((3, 3, 1)),
        ["DAPI"],
    )
    run_mask_image_quantification(cfg)
    assert writes == [MASK_IMAGE_QUANTIFICATION_TABLE_KEY]

    run_mask_image_quantification(cfg, force_rerun=True)
    table = sdata.tables[MASK_IMAGE_QUANTIFICATION_TABLE_KEY]
    assert set(table.var["image_key"]) == {"MERSCOPE_z_projection"}


def _bead_preprocessing(*, enabled: bool = True) -> dict[str, Any]:
    return build_image_preprocessing_record(
        z_layers=(1, 2, 3),
        stainings=["PolyT", "DAPI"],
        bead_removal_cfg=MerscopeBeadRemovalConfig(enabled=True) if enabled else None,
        calibration={"calibration_sha256": "cal-1"} if enabled else None,
    )


def _bead_sdata(record: dict[str, Any] | None) -> SimpleNamespace:
    return SimpleNamespace(
        images={"img": _image(np.ones((2, 2, 1)), ["DAPI"])},
        tables={},
        attrs={} if record is None else {IMAGE_PREPROCESSING_ATTR: record},
    )


def _bead_config(tmp_path: Path) -> MaskImageQuantificationConfig:
    latest = tmp_path / "latest.zarr"
    latest.mkdir(exist_ok=True)
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.array([[1, 1], [2, 0]], dtype=np.uint32))
    return MaskImageQuantificationConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        latest_zarr_path=latest,
        mask_path=mask_path,
        output_dir=tmp_path / "quant_out",
    )


def test_bead_corrected_source_is_flagged_in_table_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A table from a bead-corrected projection says which stains were corrected."""
    sdata = _bead_sdata(_bead_preprocessing())
    _patch_in_memory_zarr(monkeypatch, sdata)

    outputs = run_mask_image_quantification(_bead_config(tmp_path))

    expected = {
        "bead_corrected_stains": ["DAPI", "PolyT"],
        "bead_removal_version": 1,
        "bead_removal_fill": False,
    }
    info = sdata.tables[MASK_IMAGE_QUANTIFICATION_TABLE_KEY].uns[
        "mask_image_quantification"
    ]
    assert {key: info[key] for key in expected} == expected
    summary = json.loads(outputs["summary"].read_text())
    assert summary["bead_removal"] == expected


@pytest.mark.parametrize("record", [None, "off"])
def test_source_without_bead_removal_leaves_the_table_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record: str | None,
) -> None:
    """Default runs keep the table and summary exactly as before."""
    source = None if record is None else _bead_preprocessing(enabled=False)
    sdata = _bead_sdata(source)
    _patch_in_memory_zarr(monkeypatch, sdata)

    outputs = run_mask_image_quantification(_bead_config(tmp_path))

    info = sdata.tables[MASK_IMAGE_QUANTIFICATION_TABLE_KEY].uns[
        "mask_image_quantification"
    ]
    assert set(info) == {"table_key", "shape_key", "statistics"}
    assert "bead_removal" not in json.loads(outputs["summary"].read_text())


def test_table_without_the_flag_is_requantified_for_a_corrected_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A table written before the store recorded bead removal is not reused."""
    sdata = _bead_sdata(None)
    writes = _patch_in_memory_zarr(monkeypatch, sdata)
    cfg = _bead_config(tmp_path)
    run_mask_image_quantification(cfg)

    sdata.attrs[IMAGE_PREPROCESSING_ATTR] = _bead_preprocessing()
    run_mask_image_quantification(cfg)
    run_mask_image_quantification(cfg)

    assert writes == [MASK_IMAGE_QUANTIFICATION_TABLE_KEY] * 2
    info = sdata.tables[MASK_IMAGE_QUANTIFICATION_TABLE_KEY].uns[
        "mask_image_quantification"
    ]
    assert info["bead_corrected_stains"] == ["DAPI", "PolyT"]


def test_flagged_table_is_requantified_when_bead_removal_is_off_again(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale flag must not describe a source that no longer had the correction."""
    sdata = _bead_sdata(_bead_preprocessing())
    writes = _patch_in_memory_zarr(monkeypatch, sdata)
    cfg = _bead_config(tmp_path)
    run_mask_image_quantification(cfg)

    sdata.attrs[IMAGE_PREPROCESSING_ATTR] = _bead_preprocessing(enabled=False)
    run_mask_image_quantification(cfg)

    assert writes == [MASK_IMAGE_QUANTIFICATION_TABLE_KEY] * 2
    info = sdata.tables[MASK_IMAGE_QUANTIFICATION_TABLE_KEY].uns[
        "mask_image_quantification"
    ]
    assert "bead_corrected_stains" not in info


def test_hybrid_join_carries_the_bead_removal_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The hybrid table's feature marker repeats the image table's flag."""
    hybrid = ad.AnnData(
        X=np.asarray([[5], [7]], dtype=np.int64),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray([1, 2], dtype=np.uint64),
                "region": pd.Categorical(["MOSAIK_proseg_hybrid"] * 2),
            },
            index=["1", "2"],
        ),
        var=pd.DataFrame(index=["GeneA"]),
    )
    hybrid.uns["spatialdata_attrs"] = {"instance_key": "instance_id"}
    sdata = _bead_sdata(_bead_preprocessing())
    sdata.tables["table_MOSAIK_proseg_hybrid"] = hybrid
    _patch_in_memory_zarr(monkeypatch, sdata)

    run_mask_image_quantification(_bead_config(tmp_path))

    marker = sdata.tables["table_MOSAIK_proseg_hybrid"].uns[
        HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY
    ]
    assert marker["bead_corrected_stains"] == ["DAPI", "PolyT"]
    assert marker["bead_removal_version"] == 1
    assert marker["bead_removal_fill"] is False


def test_bead_removal_metadata_roundtrips_through_a_real_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The flag survives the zarr write and lets a second run reuse the table."""
    image = Image2DModel.parse(
        np.ones((1, 2, 2), dtype=np.uint16), dims=("c", "y", "x"), c_coords=["DAPI"]
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, index=["1"])
    )
    store = tmp_path / "latest.zarr"
    sd.SpatialData(images={"img": image}, shapes={"MOSAIK_cellpose": shapes}).write(
        store
    )
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.array([[1, 1], [1, 0]], dtype=np.uint32))
    loaded = sd.read_zarr(store)
    loaded.attrs[IMAGE_PREPROCESSING_ATTR] = _bead_preprocessing()
    loaded.write_metadata(write_attrs=True)
    cfg = MaskImageQuantificationConfig(
        dataset_name="P1_MERSCOPE",
        platform="MERSCOPE",
        latest_zarr_path=store,
        mask_path=mask_path,
        output_dir=tmp_path / "quant_out",
    )
    writes: list[str] = []
    real_write = quantification_module.write_or_replace_element

    def _recording_write(sdata_obj: Any, key: str, *args: Any, **kwargs: Any) -> bool:
        writes.append(key)
        return bool(real_write(sdata_obj, key, *args, **kwargs))

    monkeypatch.setattr(
        quantification_module, "write_or_replace_element", _recording_write
    )

    run_mask_image_quantification(cfg)
    run_mask_image_quantification(cfg)

    info = (
        sd.read_zarr(store)
        .tables[MASK_IMAGE_QUANTIFICATION_TABLE_KEY]
        .uns["mask_image_quantification"]
    )
    assert [str(stain) for stain in info["bead_corrected_stains"]] == [
        "DAPI",
        "PolyT",
    ]
    assert int(info["bead_removal_version"]) == 1
    assert not bool(info["bead_removal_fill"])
    assert writes == [MASK_IMAGE_QUANTIFICATION_TABLE_KEY]
