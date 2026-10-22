"""Tests for portable GASTON preparation, selection, and import contracts."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import box

from merxen.analysis.gaston import (
    GASTON_OWNED_COLUMNS,
    import_gaston_annotations,
    postprocess_gaston,
    prepare_gaston_input,
    rank_seed_results,
    run_gaston_glmpca,
    run_gaston_training,
    select_num_domains,
)
from merxen.config import GastonConfig
from merxen.io.spatialdata_io import (
    spatialdata_write_lock,
    spatialdata_write_lock_path,
)


def _config(tmp_path: Path, **updates: object) -> GastonConfig:
    values: dict[str, object] = {
        "pair_id": "pair1",
        "sample_id": "pair1_MERSCOPE",
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
        "clustered_h5ad_path": tmp_path / "clustered.h5ad",
        "latest_zarr_path": tmp_path / "latest_spatialdata.zarr",
        "source_table_key": "table_MOSAIK_proseg_hybrid",
        "clustered_table_key": ("table_MOSAIK_proseg_hybrid_clustering_squidpy"),
        "shape_key": "MOSAIK_proseg_hybrid",
        "use_gpu": False,
        "n_restarts": 3,
        "epochs": 2,
        "checkpoint_interval": 1,
        "glmpca_dimensions": 2,
        "glmpca_iterations": 4,
        "min_domains": 2,
        "max_domains": 4,
        "max_genes": 10,
    }
    values.update(updates)
    return GastonConfig(**values)


def _clustered_adata(cell_ids: list[str]) -> ad.AnnData:
    counts = sparse.csr_matrix(
        np.array(
            [
                [5, 0, 1],
                [0, 4, 2],
                [3, 2, 0],
            ][: len(cell_ids)],
            dtype=np.int64,
        )
    )
    obs = pd.DataFrame(
        {"instance_id": cell_ids, "existing_cluster": range(len(cell_ids))},
        index=[f"obs_{index}" for index in range(len(cell_ids))],
    )
    var = pd.DataFrame(index=["GeneA", "Blank-1", "GeneB"])
    table = ad.AnnData(X=counts.astype(float), obs=obs, var=var)
    table.layers["counts"] = counts
    table.uns["spatialdata_attrs"] = {
        "region": "MOSAIK_proseg_hybrid",
        "region_key": "region",
        "instance_key": "instance_id",
    }
    table.obs["region"] = "MOSAIK_proseg_hybrid"
    return table


def _fake_spatialdata(cell_ids: list[str]) -> SimpleNamespace:
    table = _clustered_adata(cell_ids)
    native = gpd.GeoDataFrame(
        {
            "instance_id": ["c1", "c2", "c3"][: len(cell_ids)],
            "geometry": [
                box(0, 0, 2, 2),
                box(10, 0, 12, 2),
                box(20, 0, 22, 2),
            ][: len(cell_ids)],
        },
        geometry="geometry",
    )
    aligned = native.copy()
    aligned["geometry"] = aligned.translate(xoff=1_000, yoff=2_000)
    return SimpleNamespace(
        tables={"table_MOSAIK_proseg_hybrid_clustering_squidpy": table},
        shapes={
            "MOSAIK_proseg_hybrid": native,
            "MOSAIK_proseg_hybrid_aligned_nonrigid": aligned,
        },
    )


def test_prepare_uses_native_shape_and_reorders_by_instance_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Aligned shapes must be ignored and native centroids follow H5AD order."""
    clustered = _clustered_adata(["c2", "c1", "c3"])
    config = _config(tmp_path)
    clustered.write_h5ad(config.clustered_h5ad_path)
    fake = _fake_spatialdata(["c1", "c2", "c3"])
    monkeypatch.setattr("spatialdata.read_zarr", lambda _path: fake)

    manifest_path = prepare_gaston_input(config, tmp_path / "bundle")

    coordinates = np.load(tmp_path / "bundle" / "coordinates.npy")
    np.testing.assert_allclose(coordinates, [[11, 1], [1, 1], [21, 1]])
    counts = sparse.load_npz(tmp_path / "bundle" / "counts.npz")
    assert counts.shape == (3, 2)
    assert pd.read_csv(tmp_path / "bundle" / "gene_names.tsv", sep="\t")[
        "gene_name"
    ].tolist() == ["GeneA", "GeneB"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["coordinates_are_native"] is True
    assert manifest["shape_key"] == "MOSAIK_proseg_hybrid"
    assert manifest["aligned_coordinate_elements_used"] == []
    assert manifest["n_cells"] == 3
    assert manifest["n_genes"] == 2
    assert manifest["qc"]["cell_total_counts"]["maximum"] == 6.0
    assert manifest["qc"]["native_nearest_neighbor_distance"]["minimum"] > 0
    assert set(manifest["files"]) == {
        "counts.npz",
        "coordinates.npy",
        "cell_ids.tsv",
        "gene_names.tsv",
    }


@pytest.mark.parametrize(
    ("h5ad_ids", "shape_ids", "message"),
    [
        (["c1", "c1"], ["c1", "c2"], "duplicate cell IDs"),
        (["c1", "c2"], ["c1"], "missing coordinate matches"),
    ],
)
def test_prepare_refuses_duplicate_and_partial_cell_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    h5ad_ids: list[str],
    shape_ids: list[str],
    message: str,
) -> None:
    clustered = _clustered_adata(h5ad_ids)
    config = _config(tmp_path)
    clustered.write_h5ad(config.clustered_h5ad_path)
    table = _clustered_adata(list(dict.fromkeys(h5ad_ids)))
    shapes = gpd.GeoDataFrame(
        {
            "instance_id": shape_ids,
            "geometry": [
                box(index, 0, index + 1, 1) for index in range(len(shape_ids))
            ],
        },
        geometry="geometry",
    )
    fake = SimpleNamespace(
        tables={config.clustered_table_key: table},
        shapes={config.shape_key: shapes},
    )
    monkeypatch.setattr("spatialdata.read_zarr", lambda _path: fake)

    with pytest.raises(ValueError, match=message):
        prepare_gaston_input(config, tmp_path / "bundle")


def test_prepare_refuses_noninteger_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clustered = _clustered_adata(["c1", "c2"])
    clustered.layers["counts"] = sparse.csr_matrix([[1.5, 0, 1], [0, 2, 1]])
    config = _config(tmp_path)
    clustered.write_h5ad(config.clustered_h5ad_path)
    monkeypatch.setattr(
        "spatialdata.read_zarr",
        lambda _path: _fake_spatialdata(["c1", "c2"]),
    )

    with pytest.raises(ValueError, match="non-integer"):
        prepare_gaston_input(config, tmp_path / "bundle")


def test_glmpca_loads_sparse_bundle_and_records_convergence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sparse.save_npz(
        bundle / "counts.npz",
        sparse.csr_matrix([[1, 0, 2], [0, 3, 1], [2, 1, 0]], dtype=np.int64),
    )
    captured: dict[str, Any] = {}

    def _glmpca(matrix: np.ndarray, dimensions: int, **kwargs: object) -> dict:
        captured["shape"] = matrix.shape
        captured["dimensions"] = dimensions
        captured["kwargs"] = kwargs
        return {
            "factors": np.arange(6, dtype=float).reshape(3, 2),
            "dev": np.array([10.0, 8.0, 7.0, 6.0, 5.0, 4.9996]),
        }

    monkeypatch.setitem(
        sys.modules,
        "glmpca",
        SimpleNamespace(glmpca=SimpleNamespace(glmpca=_glmpca)),
    )
    config = _config(tmp_path)

    features_path = run_gaston_glmpca(config, bundle, tmp_path / "glmpca")

    assert captured["shape"] == (3, 3)
    assert captured["dimensions"] == 2
    assert captured["kwargs"]["ctl"] == {
        "maxIter": 1_000,
        "eps": 1.0e-4,
        "optimizeTheta": True,
    }
    assert np.load(features_path).shape == (3, 2)
    manifest = json.loads((tmp_path / "glmpca" / "glmpca_manifest.json").read_text())
    assert manifest["converged"] is True
    assert manifest["configured_iterations"] == 4
    assert manifest["maximum_iterations"] == 1_000
    assert manifest["iteration_budget_auto_extended"] is True
    assert manifest["termination_reason"] == "relative_deviance_tolerance"
    assert manifest["iterations_completed"] == 6


def test_glmpca_accepts_convergence_on_the_final_allowed_iteration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sparse.save_npz(
        bundle / "counts.npz",
        sparse.csr_matrix([[1, 0, 2], [0, 3, 1], [2, 1, 0]], dtype=np.int64),
    )
    deviance = np.linspace(100.0, 90.0, 1_000)
    deviance[-1] = deviance[-2] * (1.0 - 5.0e-5)

    def _glmpca(_matrix: np.ndarray, _dimensions: int, **_kwargs: object) -> dict:
        return {
            "factors": np.arange(6, dtype=float).reshape(3, 2),
            "dev": deviance,
        }

    monkeypatch.setitem(
        sys.modules,
        "glmpca",
        SimpleNamespace(glmpca=SimpleNamespace(glmpca=_glmpca)),
    )
    output_dir = tmp_path / "glmpca"
    features_path = run_gaston_glmpca(_config(tmp_path), bundle, output_dir)
    assert features_path.is_file()
    manifest = json.loads((output_dir / "glmpca_manifest.json").read_text())
    assert manifest["converged"] is True
    assert manifest["iterations_completed"] == 1_000
    assert manifest["termination_reason"] == "relative_deviance_tolerance"


def test_glmpca_fails_after_effective_convergence_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sparse.save_npz(
        bundle / "counts.npz",
        sparse.csr_matrix([[1, 0, 2], [0, 3, 1], [2, 1, 0]], dtype=np.int64),
    )

    def _glmpca(_matrix: np.ndarray, _dimensions: int, **_kwargs: object) -> dict:
        return {
            "factors": np.arange(6, dtype=float).reshape(3, 2),
            "dev": np.arange(2_000.0, 1_000.0, -1.0),
        }

    monkeypatch.setitem(
        sys.modules,
        "glmpca",
        SimpleNamespace(glmpca=SimpleNamespace(glmpca=_glmpca)),
    )
    output_dir = tmp_path / "glmpca"
    with pytest.raises(RuntimeError, match="within 1000 effective iterations"):
        run_gaston_glmpca(_config(tmp_path), bundle, output_dir)
    manifest = json.loads((output_dir / "glmpca_manifest.json").read_text())
    assert manifest["converged"] is False
    assert manifest["termination_reason"] == "maximum_iterations"


def _write_seed(
    root: Path,
    seed: int,
    loss: float,
    *,
    status: str = "complete",
) -> Path:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    (seed_dir / "seed_manifest.json").write_text(
        json.dumps({"seed": seed, "minimum_loss": loss, "status": status})
    )
    (seed_dir / "final_model.pt").write_bytes(b"model")
    return seed_dir


def test_best_seed_ignores_failed_and_nonfinite_restarts(tmp_path: Path) -> None:
    seeds = [
        _write_seed(tmp_path, 0, 2.0),
        _write_seed(tmp_path, 1, float("nan")),
        _write_seed(tmp_path, 2, 1.0),
        _write_seed(tmp_path, 3, 0.5, status="failed"),
    ]

    ranking, best = rank_seed_results(seeds)

    assert best.seed == 2
    assert best.minimum_loss == 1.0
    assert ranking.loc[0, "seed"] == 2
    assert ranking["valid"].sum() == 2


def test_training_cli_materializes_a_failed_seed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(_config(tmp_path, n_restarts=1).model_dump_json())
    bundle = tmp_path / "bundle"
    glmpca = tmp_path / "glmpca"
    bundle.mkdir()
    glmpca.mkdir()
    monkeypatch.setattr(
        "merxen.gaston_stages.run_gaston_training",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad seed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merxen.gaston_stages",
            "train",
            "--config",
            str(config_path),
            "--bundle-dir",
            str(bundle),
            "--glmpca-dir",
            str(glmpca),
            "--seed",
            "0",
            "--output-dir",
            str(tmp_path / "seed_0"),
        ],
    )
    from merxen.gaston_stages import main

    main()
    manifest = json.loads((tmp_path / "seed_0" / "seed_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] == "bad seed"
    ranking, _best = rank_seed_results(
        [tmp_path / "seed_0", _write_seed(tmp_path, 1, 1.0)]
    )
    assert ranking.loc[ranking["seed"] == 0, "status"].item() == "failed"


def test_domain_selection_fixed_and_direct_override() -> None:
    likelihoods = pd.DataFrame(
        {"num_domains": [2, 3, 4], "negative_log_likelihood": [10.0, 6.0, 5.5]}
    )
    assert select_num_domains(
        likelihoods,
        domain_mode="fixed",
        num_domains=3,
        auto_k_fallback=None,
    ) == (3, "num_domains_override")
    assert select_num_domains(
        likelihoods,
        domain_mode="auto",
        num_domains=4,
        auto_k_fallback=None,
    ) == (4, "num_domains_override")


def test_domain_selection_automatic_k_and_no_knee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    likelihoods = pd.DataFrame(
        {"num_domains": [2, 3, 4], "negative_log_likelihood": [10.0, 6.0, 5.5]}
    )
    fake_kneed = ModuleType("kneed")
    monkeypatch.setattr(
        fake_kneed,
        "KneeLocator",
        lambda *_args, **_kwargs: SimpleNamespace(knee=3),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "kneed", fake_kneed)
    assert select_num_domains(
        likelihoods,
        domain_mode="auto",
        num_domains=None,
        auto_k_fallback=None,
    ) == (3, "kneedle")

    monkeypatch.setattr(
        fake_kneed,
        "KneeLocator",
        lambda *_args, **_kwargs: SimpleNamespace(knee=None),
    )
    assert select_num_domains(
        likelihoods,
        domain_mode="auto",
        num_domains=None,
        auto_k_fallback=4,
    ) == (4, "auto_k_fallback")
    with pytest.raises(RuntimeError, match="found no Kneedle knee"):
        select_num_domains(
            likelihoods,
            domain_mode="auto",
            num_domains=None,
            auto_k_fallback=None,
        )


def test_import_preserves_existing_clustering_and_mender_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    Path(config.latest_zarr_path).mkdir()
    clustered = _clustered_adata(["c2", "c1"])
    clustered.write_h5ad(config.clustered_h5ad_path)
    source_table = clustered.copy()
    source_table.obs["mender_domain"] = [7, 8]
    fake = SimpleNamespace(tables={config.clustered_table_key: source_table})
    standalone = tmp_path / "standalone"
    (standalone / "model").mkdir(parents=True)
    annotations = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "gaston_domain": [0, 1],
            "gaston_isodepth": [0.25, -0.5],
            "gaston_model_seed": [2, 2],
            "gaston_num_domains": [2, 2],
        }
    )
    annotations.to_parquet(
        standalone / f"{config.sample_id}_gaston_cells.parquet",
        index=False,
    )
    (standalone / "model" / "model_selection.json").write_text(
        json.dumps({"best_seed": 2, "num_domains": 2})
    )
    captured: dict[str, ad.AnnData] = {}
    monkeypatch.setattr("spatialdata.read_zarr", lambda _path: fake)
    monkeypatch.setattr(
        "spatialdata.models.TableModel.parse",
        lambda table, **_kwargs: table,
    )

    def _capture_write(
        _sdata: object,
        _key: str,
        _element_type: str,
        value: ad.AnnData,
        **_kwargs: object,
    ) -> bool:
        captured["table"] = value
        return True

    monkeypatch.setattr(
        "merxen.io.spatialdata_io.write_or_replace_element",
        _capture_write,
    )

    results = import_gaston_annotations(config, standalone, tmp_path / "final")

    annotated = ad.read_h5ad(results["annotated_h5ad"])
    assert annotated.obs["gaston_domain"].dtype.name == "category"
    assert annotated.obs["gaston_domain"].astype(int).tolist() == [1, 0]
    imported = captured["table"]
    assert imported.obs["existing_cluster"].tolist() == [0, 1]
    assert imported.obs["mender_domain"].tolist() == [7, 8]
    assert set(GASTON_OWNED_COLUMNS).issubset(imported.obs.columns)
    assert imported.obs["gaston_isodepth"].tolist() == [-0.5, 0.25]
    for column in GASTON_OWNED_COLUMNS:
        assert (
            annotated.obs[column].astype(str).tolist()
            == imported.obs[column].astype(str).tolist()
        )


def test_shared_spatialdata_lock_serializes_two_writers(tmp_path: Path) -> None:
    zarr_path = tmp_path / "latest_spatialdata.zarr"
    zarr_path.mkdir()
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def _first_writer() -> None:
        with spatialdata_write_lock(zarr_path):
            first_acquired.set()
            assert release_first.wait(timeout=2)

    def _second_writer() -> None:
        assert first_acquired.wait(timeout=2)
        with spatialdata_write_lock(zarr_path):
            second_acquired.set()

    first = threading.Thread(target=_first_writer)
    second = threading.Thread(target=_second_writer)
    first.start()
    second.start()
    assert first_acquired.wait(timeout=2)
    time.sleep(0.05)
    assert not second_acquired.is_set()
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert second_acquired.is_set()
    assert spatialdata_write_lock_path(zarr_path) == Path(
        f"{zarr_path}.merxen-write.lock"
    )


def test_fixed_mode_requires_num_domains(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_domains is required"):
        _config(tmp_path, domain_mode="fixed", num_domains=None)


def test_postprocess_retains_all_gpu_monitor_logs(tmp_path: Path) -> None:
    from merxen.analysis.gaston import _copy_gpu_vram_logs

    seeds = []
    for seed in range(2):
        seed_dir = tmp_path / f"seed_{seed}"
        monitor_dir = seed_dir / "gpu_vram"
        monitor_dir.mkdir(parents=True)
        (monitor_dir / "summary.json").write_text(json.dumps({"seed": seed}))
        seeds.append(seed_dir)
    model_dir = tmp_path / "model"
    _copy_gpu_vram_logs(seeds, model_dir)
    assert json.loads(
        (model_dir / "gpu_vram" / "seed_0" / "summary.json").read_text()
    ) == {"seed": 0}
    assert json.loads(
        (model_dir / "gpu_vram" / "seed_1" / "summary.json").read_text()
    ) == {"seed": 1}


@pytest.mark.slow
def test_reduced_gaston_training_smoke(tmp_path: Path) -> None:
    """Exercise reduced GLM-PCA, restart, and postprocessing end to end."""
    pytest.importorskip("gaston")
    pytest.importorskip("glmpca")
    bundle = tmp_path / "bundle"
    glmpca = tmp_path / "glmpca"
    bundle.mkdir()
    random = np.random.default_rng(4)
    counts = sparse.csr_matrix(random.poisson(3, size=(30, 8)).astype(np.int64))
    sparse.save_npz(bundle / "counts.npz", counts)
    coordinates = np.column_stack(
        (np.arange(30, dtype=float), np.sin(np.arange(30, dtype=float) / 4.0))
    )
    np.save(bundle / "coordinates.npy", coordinates)
    pd.DataFrame({"cell_id": [f"cell_{index}" for index in range(30)]}).to_csv(
        bundle / "cell_ids.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame({"gene_name": [f"gene_{index}" for index in range(8)]}).to_csv(
        bundle / "gene_names.tsv",
        sep="\t",
        index=False,
    )
    (bundle / "input_manifest.json").write_text(json.dumps({"n_cells": 30}))
    config = _config(
        tmp_path,
        n_restarts=2,
        epochs=2,
        checkpoint_interval=1,
        glmpca_iterations=30,
        domain_mode="fixed",
        num_domains=2,
        min_domains=2,
        max_domains=2,
        domain_buckets=10,
        figure_dpi=72,
    )
    features = run_gaston_glmpca(config, bundle, glmpca)
    assert np.load(features).shape == (30, 2)
    seed_dirs = []
    for seed in range(2):
        seed_dir = tmp_path / f"seed_{seed}"
        manifest = run_gaston_training(
            config,
            bundle,
            glmpca,
            seed,
            seed_dir,
        )
        assert json.loads(manifest.read_text())["status"] == "complete"
        seed_dirs.append(seed_dir)
    cells = postprocess_gaston(
        config,
        bundle,
        glmpca,
        seed_dirs,
        tmp_path / "gaston_out",
    )
    annotations = pd.read_parquet(cells)
    assert len(annotations) == 30
    assert annotations["cell_id"].is_unique
    assert set(annotations["gaston_num_domains"]) == {2}
    for relative_path in (
        "input/input_manifest.json",
        "model/best_model.pt",
        "model/model_selection.json",
        "model/seed_losses.tsv",
        "model/domain_likelihoods.tsv",
    ):
        assert (tmp_path / "gaston_out" / relative_path).is_file()
    for stem in ("isodepth", "domains", "domain_likelihood", "training_losses"):
        assert (tmp_path / "gaston_out" / "plots" / f"{stem}.png").is_file()
        assert (tmp_path / "gaston_out" / "plots" / f"{stem}.pdf").is_file()
