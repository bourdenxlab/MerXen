"""Tests for the Scanpy/Squidpy clustering shim."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import box

from merxen.analysis.clustering_squidpy import (
    _run_gpu_clustering,
    adata_from_spatialdata,
    plot_spatial_scatter,
    remove_control_features,
    run_scanpy_clustering,
)


def test_adata_from_spatialdata_adds_spatial_area_and_control_metrics() -> None:
    """SpatialData table extraction should add Squidpy-ready coordinates."""
    obs = pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "control_probe_counts": [1, 0, 2, 0],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    var = pd.DataFrame(index=["GeneA", "Blank-1", "GeneB", "NegControlProbe-1"])
    adata = ad.AnnData(
        X=np.array(
            [
                [10, 1, 0, 2],
                [0, 0, 12, 1],
                [3, 4, 5, 0],
                [6, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
        obs=obs,
        var=var,
    )
    adata.obsm["blank"] = pd.DataFrame(
        {"Blank-A": [5, 0, 1, 0]},
        index=adata.obs_names,
    )
    adata.uns["spatialdata_attrs"] = {"region": "MOSAIK_proseg"}

    gdf = gpd.GeoDataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "geometry": [
                box(0, 0, 1, 1),
                box(2, 0, 3, 1),
                box(0, 2, 1, 3),
                box(2, 2, 3, 3),
            ],
        },
        geometry="geometry",
    )
    aligned_gdf = gdf.copy()
    aligned_gdf["geometry"] = aligned_gdf.geometry.translate(xoff=10.0)
    fake_sdata = SimpleNamespace(
        tables={"table": adata},
        shapes={
            "MOSAIK_proseg": gdf,
            "MOSAIK_proseg_aligned_nonrigid": aligned_gdf,
        },
    )

    out = adata_from_spatialdata(fake_sdata, platform="MERSCOPE")

    assert out.uns["merxen_clustering_squidpy"]["shape_key"] == (
        "MOSAIK_proseg_aligned_nonrigid"
    )
    np.testing.assert_allclose(out.obsm["spatial"][0], [10.5, 0.5])
    np.testing.assert_allclose(out.obs["cell_area"].to_numpy(float), 1.0)
    np.testing.assert_allclose(
        out.obs["control_counts"].to_numpy(float),
        [9.0, 1.0, 7.0, 0.0],
    )
    assert out.obs["nucleus_ratio"].isna().all()


def test_adata_from_spatialdata_adds_xenium_nucleus_ratio_from_shapes() -> None:
    """Xenium nucleus shapes should fill nucleus_area when tables lack it."""
    obs = pd.DataFrame(
        {"cell_id": ["x1", "x2"]},
        index=["x1", "x2"],
    )
    adata = ad.AnnData(
        X=np.array([[10, 1], [2, 8]], dtype=np.int64),
        obs=obs,
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    adata.uns["spatialdata_attrs"] = {"region": "xenium_cell_boundaries"}
    cell_gdf = gpd.GeoDataFrame(
        {
            "cell_id": ["x1", "x2"],
            "geometry": [box(0, 0, 2, 2), box(4, 0, 6, 2)],
        },
        geometry="geometry",
    )
    nucleus_gdf = gpd.GeoDataFrame(
        {
            "cell_id": ["x1", "x2"],
            "geometry": [box(0, 0, 1, 1), box(4, 0, 5, 1)],
        },
        geometry="geometry",
    )
    fake_sdata = SimpleNamespace(
        tables={"table": adata},
        shapes={
            "xenium_cell_boundaries": cell_gdf,
            "xenium_nucleus": nucleus_gdf,
        },
    )

    out = adata_from_spatialdata(fake_sdata, platform="XENIUM")

    np.testing.assert_allclose(out.obs["cell_area"].to_numpy(float), [4.0, 4.0])
    np.testing.assert_allclose(out.obs["nucleus_area"].to_numpy(float), [1.0, 1.0])
    np.testing.assert_allclose(out.obs["nucleus_ratio"].to_numpy(float), [0.25, 0.25])


def test_adata_from_spatialdata_adds_ensembl_ids_from_original_table() -> None:
    """Gene IDs from one SpatialData table should annotate clustering tables."""
    obs = pd.DataFrame({"cell_id": ["x1", "x2"]}, index=["x1", "x2"])
    adata = ad.AnnData(
        X=np.array([[10, 1], [2, 8]], dtype=np.int64),
        obs=obs,
        var=pd.DataFrame({"gene": ["GeneA", "GeneB"]}, index=["GeneA", "GeneB"]),
    )
    adata.uns["spatialdata_attrs"] = {"region": "xenium_cell_boundaries"}
    original = ad.AnnData(
        X=np.ones((1, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["cell0"]),
        var=pd.DataFrame(
            {
                "gene_ids": ["ENSG000001", "ENSG000002"],
                "feature_types": ["Gene Expression", "Gene Expression"],
            },
            index=["GeneA", "GeneB"],
        ),
    )
    cell_gdf = gpd.GeoDataFrame(
        {
            "cell_id": ["x1", "x2"],
            "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        },
        geometry="geometry",
    )
    fake_sdata = SimpleNamespace(
        tables={"table": adata, "table_original": original},
        shapes={"xenium_cell_boundaries": cell_gdf},
    )

    out = adata_from_spatialdata(fake_sdata, platform="XENIUM")

    assert list(out.var["ensembl_id"]) == ["ENSG000001", "ENSG000002"]
    assert out.uns["merxen_clustering_squidpy"]["ensembl_id_mapping"] == {
        "n_features": 2,
        "n_mapped": 2,
        "column": "ensembl_id",
    }


def test_run_scanpy_clustering_adds_umap_and_leiden() -> None:
    """The gentle Scanpy workflow should produce expected clustering fields."""
    rng = np.random.default_rng(1)
    adata = ad.AnnData(
        X=rng.poisson(lam=4, size=(12, 6)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(12)]),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(6)]),
    )
    adata.obsm["spatial"] = rng.normal(size=(12, 2))

    out = run_scanpy_clustering(
        adata,
        min_counts=1,
        min_cells=1,
        normalize_exclude_highly_expressed=False,
        n_pcs=3,
        n_neighbors=3,
        umap_min_dist=0.2,
        umap_spread=1.5,
        random_seed=1,
        use_gpu=False,
    )

    assert "counts" in out.layers
    assert "X_umap" in out.obsm
    assert "leiden" in out.obs
    assert out.uns["merxen_clustering_params"]["umap_min_dist"] == 0.2
    assert out.uns["merxen_clustering_params"]["umap_spread"] == 1.5


def test_run_scanpy_clustering_preserves_ensembl_ids() -> None:
    """Filtering and clustering should retain gene IDs needed by MapMyCells."""
    rng = np.random.default_rng(2)
    adata = ad.AnnData(
        X=rng.poisson(lam=4, size=(12, 4)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(12)]),
        var=pd.DataFrame(
            {
                "gene": ["GeneA", "GeneB", "Blank-1", "GeneC"],
                "ensembl_id": ["ENSG000001", "ENSG000002", "", "ENSG000003"],
            },
            index=["GeneA", "GeneB", "Blank-1", "GeneC"],
        ),
    )
    adata.obsm["spatial"] = rng.normal(size=(12, 2))

    out = run_scanpy_clustering(
        adata,
        min_counts=1,
        min_cells=1,
        n_pcs=2,
        n_neighbors=3,
        random_seed=2,
        use_gpu=False,
    )

    assert list(out.var["ensembl_id"]) == [
        "ENSG000001",
        "ENSG000002",
        "ENSG000003",
    ]


def test_remove_control_features_drops_blank_negative_and_unassigned() -> None:
    """Control-like features should be excluded from clustering inputs."""
    adata = ad.AnnData(
        X=np.ones((4, 5), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(
            index=[
                "GeneA",
                "Blank-1",
                "NegControlProbe_00001",
                "UnassignedCodeword_0001",
                "GeneB",
            ]
        ),
    )

    filtered = remove_control_features(adata)

    assert list(filtered.var_names) == ["GeneA", "GeneB"]
    summary = filtered.uns["merxen_clustering_squidpy"]["control_feature_filter"]
    assert summary["n_features_before"] == 5
    assert summary["n_control_features_removed"] == 3
    assert summary["removed_control_features"] == [
        "Blank-1",
        "NegControlProbe_00001",
        "UnassignedCodeword_0001",
    ]


def test_run_gpu_clustering_uses_chunked_pca_for_sparse_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse GPU PCA should avoid rapids-singlecell's fragile sparse helper."""
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.ones((20, 6), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(6)]),
    )
    calls: dict[str, object] = {}

    fake_get = SimpleNamespace(
        anndata_to_GPU=lambda data: calls.setdefault("to_gpu", data),
        anndata_to_CPU=lambda data: calls.setdefault("to_cpu", data),
    )

    def fake_pca(data: ad.AnnData, **kwargs: object) -> None:
        calls["pca"] = kwargs
        data.obsm["X_pca"] = np.ones((data.n_obs, 3), dtype=np.float32)

    def fake_neighbors(data: ad.AnnData, **kwargs: object) -> None:
        calls["neighbors"] = kwargs

    fake_pp = SimpleNamespace(pca=fake_pca, neighbors=fake_neighbors)
    fake_tl = SimpleNamespace(
        umap=lambda data, **kwargs: calls.setdefault("umap", kwargs),
        leiden=lambda data, **kwargs: calls.setdefault("leiden", kwargs),
    )
    fake_rsc = SimpleNamespace(get=fake_get, pp=fake_pp, tl=fake_tl)
    monkeypatch.setitem(sys.modules, "rapids_singlecell", fake_rsc)

    gpu_used = _run_gpu_clustering(
        adata,
        max_pcs=3,
        n_pcs_for_neighbors=3,
        effective_neighbors=5,
        umap_min_dist=0.4,
        umap_spread=1.2,
        leiden_resolution=0.8,
        random_seed=7,
    )

    assert gpu_used is True
    assert calls["to_gpu"] is adata
    assert calls["to_cpu"] is adata
    assert calls["pca"] == {
        "n_comps": 3,
        "random_state": 7,
        "chunked": True,
        "chunk_size": adata.n_obs,
    }
    assert calls["neighbors"] == {
        "n_neighbors": 5,
        "n_pcs": 3,
        "use_rep": "X_pca",
        "random_state": 7,
    }


def test_plot_spatial_scatter_suppresses_squidpy_noise(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Image-less spatial scatter should not emit Squidpy library warnings."""
    adata = ad.AnnData(
        X=np.ones((5, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"leiden": pd.Categorical(["0", "1", "0", "1", "2"])},
            index=[f"cell{i}" for i in range(5)],
        ),
        var=pd.DataFrame(index=["Gene0", "Gene1"]),
    )
    adata.obsm["spatial"] = np.column_stack(
        [np.arange(adata.n_obs), np.arange(adata.n_obs)]
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        output_path = plot_spatial_scatter(
            adata,
            tmp_path / "spatial.png",
            point_size=0.2,
        )

    captured = capsys.readouterr()
    warning_text = "\n".join(str(item.message) for item in recorded)
    output_text = f"{captured.out}\n{captured.err}"
    assert output_path.exists()
    assert "No data for colormapping provided via 'c'" not in warning_text
    assert "Please specify a valid `library_id`" not in output_text
