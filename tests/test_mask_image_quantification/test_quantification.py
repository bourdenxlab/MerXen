"""Tests for Cellpose-mask image-channel quantification."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from spatialdata.models import TableModel

from merxen.config import MaskImageQuantificationConfig
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
