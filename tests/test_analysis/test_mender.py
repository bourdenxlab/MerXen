"""Tests for the isolated MENDER analysis stages."""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import Point

from merxen.analysis.mender import (
    clustering_request,
    finalize_mender,
    import_mender_spatialdata,
    prepare_mender,
    resolve_cell_ids,
    spatialdata_write_lock,
)
from merxen.config import MenderConfig
from merxen.mender_compute import (
    build_minimal_anndata,
    run_mender_compute,
    validate_mender_result,
)


def _config(tmp_path: Path, **overrides: Any) -> MenderConfig:
    values: dict[str, Any] = {
        "pair_id": "pair1",
        "sample_id": "pair1_MERSCOPE",
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
        "source_h5ad": tmp_path / "clustered.h5ad",
        "spatialdata_path": tmp_path / "latest.zarr",
        "source_spatialdata_table": ("table_MOSAIK_proseg_hybrid_clustering_squidpy"),
        "native_shape_key": "MOSAIK_proseg_hybrid",
        "output_dir": tmp_path / "mender_out",
    }
    values.update(overrides)
    return MenderConfig.model_validate(values)


def _clustered_adata(
    cell_ids: list[str],
    states: pd.Categorical | list[str],
    *,
    state_key: str = "hierarchical_cluster",
) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "cell_id": cell_ids,
            state_key: states,
        },
        index=pd.Index([f"row_{index}" for index in range(len(cell_ids))]),
    )
    adata = ad.AnnData(X=sparse.csr_matrix((len(cell_ids), 2)), obs=obs)
    adata.uns["spatialdata_attrs"] = {
        "region": "MOSAIK_proseg_hybrid",
        "region_key": "region",
        "instance_key": "cell_id",
    }
    adata.obsm["spatial"] = np.full((len(cell_ids), 2), 999.0)
    return adata


def test_mender_config_defaults_and_clustering_modes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.cell_state_key == "hierarchical_cluster"
    assert config.radius_um == 20.0
    assert config.n_scales == 5
    assert config.count_rep == "s"
    assert config.include_self is False
    assert config.random_seed == 666
    assert clustering_request(config) == -0.8

    target = _config(tmp_path, clustering_mode="target_k", target_k=7)
    assert clustering_request(target) == 7
    with pytest.raises(ValueError, match="target_k is required"):
        _config(tmp_path, clustering_mode="target_k")
    with pytest.raises(ValueError):
        _config(tmp_path, clustering_mode="target_k", target_k=1)


def test_minimal_anndata_has_no_expression_columns_and_native_coordinates() -> None:
    portable = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c"],
            "native_x": [1.0, 2.0, 3.0],
            "native_y": [4.0, 5.0, 6.0],
            "cell_state": pd.Categorical(["x", "x", "y"]),
        }
    )
    adata = build_minimal_anndata(portable)
    assert adata.shape == (3, 0)
    assert list(adata.obs_names) == ["a", "b", "c"]
    assert isinstance(adata.obs["cell_state"].dtype, pd.CategoricalDtype)
    np.testing.assert_array_equal(
        adata.obsm["spatial"],
        portable[["native_x", "native_y"]].to_numpy(),
    )


@pytest.mark.parametrize(
    ("states", "error_type", "message"),
    [
        (["a", "b"], TypeError, "must be categorical"),
        (pd.Categorical(["a", None]), ValueError, "missing or empty"),
        (pd.Categorical(["a", ""]), ValueError, "missing or empty"),
    ],
)
def test_prepare_rejects_non_categorical_or_missing_states_before_spatialdata(
    tmp_path: Path,
    states: pd.Categorical | list[str],
    error_type: type[Exception],
    message: str,
) -> None:
    config = _config(tmp_path)
    _clustered_adata(["1", "2"], states).write_h5ad(config.source_h5ad)
    with pytest.raises(error_type, match=message):
        prepare_mender(config, tmp_path / "prepared")


def test_prepare_uses_custom_state_and_explicit_native_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, cell_state_key="custom_state")
    cell_ids = ["c1", "c2", "c3"]
    clustered = _clustered_adata(
        cell_ids,
        pd.Categorical(["glia", "neuron", "glia"]),
        state_key="custom_state",
    )
    clustered.write_h5ad(config.source_h5ad)
    table = clustered.copy()
    shapes = gpd.GeoDataFrame(
        {"cell_id": ["c3", "c1", "c2"]},
        geometry=[Point(30, 3), Point(10, 1), Point(20, 2)],
    )
    fake_sdata = types.SimpleNamespace(
        tables={config.source_spatialdata_table: table},
        shapes={config.native_shape_key: shapes},
    )
    spatialdata_module = types.SimpleNamespace(read_zarr=lambda _path: fake_sdata)
    monkeypatch.setitem(sys.modules, "spatialdata", spatialdata_module)

    manifest_path = prepare_mender(config, tmp_path / "prepared")
    portable = pd.read_parquet(manifest_path.parent / "mender_input.parquet")
    np.testing.assert_array_equal(portable["native_x"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(portable["native_y"], [1.0, 2.0, 3.0])
    assert not np.any(portable[["native_x", "native_y"]].to_numpy() == 999.0)
    assert list(portable["cell_state"].astype(str)) == ["glia", "neuron", "glia"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["native_shape_key"] == config.native_shape_key
    assert manifest["cell_state_key"] == "custom_state"
    assert manifest["settings"]["radius_um"] == 20.0
    assert manifest["settings"]["n_scales"] == 5
    assert manifest["settings"]["count_rep"] == "s"
    assert manifest["settings"]["include_self"] is False
    assert manifest["settings"]["clustering_request"] == -0.8


class _ResultModel:
    def __init__(self: _ResultModel, context: ad.AnnData) -> None:
        self.adata_MENDER = context


def _context_result(
    cell_ids: list[str],
    domains: list[str] | None,
) -> ad.AnnData:
    context = ad.AnnData(
        X=np.ones((len(cell_ids), 3)),
        obs=pd.DataFrame(index=pd.Index(cell_ids)),
    )
    context.obsm["spatial"] = np.column_stack(
        [np.arange(len(cell_ids)), np.arange(len(cell_ids)) + 10]
    ).astype(float)
    if domains is not None:
        context.obs["MENDER"] = pd.Categorical(domains)
    return context


@pytest.mark.parametrize(
    ("result_ids", "domains", "message"),
    [
        (["a", "b"], None, "missing obs"),
        (["a"], ["0"], "do not round-trip"),
        (["a", "b"], ["0", "0"], "fewer than two"),
        (["a", "b"], ["0", None], "missing domains"),
    ],
)
def test_compute_postconditions_reject_invalid_domains(
    result_ids: list[str],
    domains: list[str] | None,
    message: str,
) -> None:
    source = _context_result(["a", "b"], ["x", "y"])
    result = _context_result(result_ids, domains)
    with pytest.raises(RuntimeError, match=message):
        validate_mender_result(_ResultModel(result), source)


class _FakeMenderSingle:
    requests: list[float | int] = []

    def __init__(
        self: _FakeMenderSingle,
        adata: ad.AnnData,
        ct_obs: str,
        random_seed: int,
    ) -> None:
        assert ct_obs == "cell_state"
        assert random_seed == 666
        self.adata = adata

    def set_MENDER_para(  # noqa: N802
        self: _FakeMenderSingle,
        **kwargs: Any,
    ) -> None:
        assert kwargs == {
            "nn_mode": "radius",
            "nn_para": 20.0,
            "count_rep": "s",
            "include_self": False,
            "n_scales": 5,
        }

    def run_representation(self: _FakeMenderSingle) -> None:
        for scale in range(5):
            self.adata.obsm[f"scale{scale}"] = np.ones((self.adata.n_obs, 2))
        self.adata_MENDER = ad.AnnData(
            X=np.ones((self.adata.n_obs, 10)),
            obs=self.adata.obs.copy(),
        )
        self.adata_MENDER.obsm["spatial"] = self.adata.obsm["spatial"].copy()
        self.adata_MENDER.obsm["X_MENDERMAP2D"] = self.adata.obsm["spatial"].copy()

    def run_clustering_normal(
        self: _FakeMenderSingle,
        request: float | int,
        run_umap: bool,
    ) -> None:
        self.requests.append(request)
        assert run_umap is True
        labels = [str(index % 2) for index in range(self.adata.n_obs)]
        self.adata_MENDER.obs["MENDER"] = pd.Categorical(labels)


@pytest.mark.parametrize(
    ("mode", "target_k", "expected"),
    [("resolution", None, -0.8), ("target_k", 3, 3)],
)
def test_synthetic_compute_smoke_uses_defaults_and_signed_clustering_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    target_k: int | None,
    expected: float | int,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    grid = pd.DataFrame(
        {
            "cell_id": [f"cell_{index}" for index in range(6)],
            "native_x": [0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
            "native_y": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "cell_state": pd.Categorical(["a", "a", "b", "a", "b", "b"]),
        }
    )
    grid.to_parquet(prepared / "mender_input.parquet", index=False)
    config = {
        "sample_id": "pair1_MERSCOPE",
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
        "clustering_mode": mode,
        "target_k": target_k,
        "random_seed": 666,
        "nn_mode": "radius",
        "count_rep": "s",
        "run_umap": True,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setitem(
        sys.modules,
        "MENDER",
        types.SimpleNamespace(MENDER_single=_FakeMenderSingle),
    )
    _FakeMenderSingle.requests.clear()

    outputs = run_mender_compute(config_path, prepared, tmp_path / "computed")
    assert _FakeMenderSingle.requests == [expected]
    assert outputs["context_h5ad"].exists()
    domains = pd.read_parquet(outputs["domains"])
    assert set(domains["cell_id"]) == set(grid["cell_id"])
    assert domains["mender_domain"].nunique() == 2


def test_finalize_round_trips_reordered_domain_rows_and_writes_qc(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    cell_ids = ["c1", "c2", "c3", "c4"]
    source = _clustered_adata(
        cell_ids,
        pd.Categorical(["a", "a", "b", "b"]),
    )
    source.write_h5ad(config.source_h5ad)
    prepared = tmp_path / "prepared"
    computed = tmp_path / "computed"
    prepared.mkdir()
    computed.mkdir()
    portable = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "native_x": [0.0, 1.0, 2.0, 3.0],
            "native_y": [4.0, 5.0, 6.0, 7.0],
            "cell_state": pd.Categorical(["a", "a", "b", "b"]),
        }
    )
    portable.to_parquet(prepared / "mender_input.parquet", index=False)
    (prepared / "input_manifest.json").write_text(
        json.dumps(
            {
                "n_cells": 4,
                "state_counts": {"a": 2, "b": 2},
            }
        )
    )
    pd.DataFrame(
        {
            "cell_id": ["c4", "c2", "c1", "c3"],
            "mender_domain": pd.Categorical(["1", "0", "1", "0"]),
        }
    ).to_parquet(computed / "mender_domains.parquet", index=False)
    context = ad.AnnData(
        X=np.ones((4, 3)),
        obs=pd.DataFrame(index=pd.Index(cell_ids)),
    )
    context.obsm["spatial"] = portable[["native_x", "native_y"]].to_numpy()
    context.obsm["X_MENDERMAP2D"] = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    context.write_h5ad(computed / "mender_context.h5ad")
    pd.DataFrame(
        {
            "scale": range(6),
            "cell_count_min": [1] * 6,
            "cell_count_median": [2] * 6,
            "cell_count_mean": [2] * 6,
            "cell_count_max": [3] * 6,
        }
    ).to_csv(computed / "scale_neighbour_summary.tsv", sep="\t", index=False)

    outputs = finalize_mender(config, prepared, computed, tmp_path / "final")
    annotated = ad.read_h5ad(outputs["annotated_h5ad"])
    assert list(annotated.obs["mender_domain"].astype(str)) == ["1", "0", "0", "1"]
    assert "merxen_mender" in annotated.uns
    sample_dir = outputs["annotated_h5ad"].parent
    for relative in [
        "pair1_MERSCOPE_mender_context.h5ad",
        "pair1_MERSCOPE_mender_cells.parquet",
        "input/input_manifest.json",
        "tables/domain_sizes.tsv",
        "tables/state_by_domain.tsv",
        "tables/scale_neighbour_summary.tsv",
        "plots/spatial_domains.png",
        "plots/spatial_domains.pdf",
        "plots/context_umap.png",
        "plots/context_umap.pdf",
        "plots/state_domain_heatmap.png",
        "plots/state_domain_heatmap.pdf",
        "mender_manifest.json",
    ]:
        assert (sample_dir / relative).exists(), relative


def test_resolve_cell_ids_rejects_duplicates() -> None:
    adata = _clustered_adata(
        ["duplicate", "duplicate"],
        pd.Categorical(["a", "b"]),
    )
    with pytest.raises(ValueError, match="not unique"):
        resolve_cell_ids(adata)


def test_spatialdata_import_preserves_existing_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.spatialdata_path.mkdir()
    finalized = tmp_path / "finalized"
    sample_dir = finalized / "merscope"
    sample_dir.mkdir(parents=True)
    annotated = _clustered_adata(
        ["c1", "c2", "c3"],
        pd.Categorical(["a", "a", "b"]),
    )
    annotated.obs["mender_domain"] = pd.Categorical(["1", "0", "1"])
    annotated.uns["merxen_mender"] = {"radius_um": 15.0, "n_scales": 6}
    annotated.write_h5ad(sample_dir / "pair1_MERSCOPE_mender_annotated.h5ad")

    table = _clustered_adata(
        ["c3", "c1", "c2"],
        pd.Categorical(["b", "a", "a"]),
    )
    table.obs["region"] = pd.Categorical(["MOSAIK_proseg_hybrid"] * 3)
    table.obs["gaston_domain"] = pd.Categorical(["g2", "g1", "g1"])
    table.obs["cortical_depth"] = [0.8, 0.1, 0.4]
    table.obs["mapmycells_label"] = pd.Categorical(["x", "y", "z"])
    table.uns["merxen_gaston"] = {"preserve": True}
    fake_sdata = types.SimpleNamespace(tables={config.source_spatialdata_table: table})
    spatialdata_module = types.ModuleType("spatialdata")
    spatialdata_module.read_zarr = lambda _path: fake_sdata  # type: ignore[attr-defined]
    models_module = types.ModuleType("spatialdata.models")
    models_module.TableModel = types.SimpleNamespace(  # type: ignore[attr-defined]
        parse=lambda parsed_table, **_kwargs: parsed_table
    )
    monkeypatch.setitem(sys.modules, "spatialdata", spatialdata_module)
    monkeypatch.setitem(sys.modules, "spatialdata.models", models_module)
    captured: dict[str, Any] = {}

    def fake_write(
        _sdata: Any,
        key: str,
        element_type: str,
        value: ad.AnnData,
        *,
        overwrite: bool,
    ) -> bool:
        captured.update(
            key=key,
            element_type=element_type,
            value=value,
            overwrite=overwrite,
        )
        return True

    monkeypatch.setattr("merxen.analysis.mender.write_or_replace_element", fake_write)
    manifest = import_mender_spatialdata(
        config,
        finalized,
        tmp_path / "spatialdata_import_manifest.json",
    )
    imported = captured["value"]
    assert captured["key"] == config.source_spatialdata_table
    assert captured["element_type"] == "tables"
    assert captured["overwrite"] is True
    assert list(imported.obs["mender_domain"].astype(str)) == ["1", "1", "0"]
    for column in ["gaston_domain", "cortical_depth", "mapmycells_label"]:
        assert column in imported.obs
    assert imported.uns["merxen_gaston"] == {"preserve": True}
    assert imported.uns["merxen_mender"]["radius_um"] == 15.0
    assert json.loads(manifest.read_text())["imported"] is True


def test_shared_spatialdata_lock_serializes_contending_writers(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "latest.zarr"
    first_acquired = threading.Event()
    allow_first_release = threading.Event()
    second_acquired = threading.Event()

    def first_writer() -> None:
        with spatialdata_write_lock(zarr_path):
            first_acquired.set()
            allow_first_release.wait(timeout=5)

    def second_writer() -> None:
        first_acquired.wait(timeout=5)
        with spatialdata_write_lock(zarr_path):
            second_acquired.set()

    first = threading.Thread(target=first_writer)
    second = threading.Thread(target=second_writer)
    first.start()
    second.start()
    assert first_acquired.wait(timeout=2)
    time.sleep(0.05)
    assert not second_acquired.is_set()
    allow_first_release.set()
    assert second_acquired.wait(timeout=2)
    first.join(timeout=2)
    second.join(timeout=2)
    assert Path(f"{zarr_path}.merxen-write.lock").exists()
