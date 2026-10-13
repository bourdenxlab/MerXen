"""Tests for the whole-section QC embedding (plan §6.3)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from merxen.clustering.representation import (
    CPU_LEIDEN_ENGINE,
    CPU_LEIDEN_FLAVOR,
    CPU_LEIDEN_N_ITERATIONS,
    QC_BROAD_LEIDEN_KEY,
    QC_LEIDEN_KEY,
    EmbeddingParams,
    branch_embedding,
    library_versions,
    normalize_table_cells,
    pca_gene_mask,
    whole_section_qc,
)

from .conftest import Section, make_config, make_section


def _table_cells(section: Section) -> object:
    adata = section.adata
    counts = np.asarray(adata.X.sum(axis=1)).ravel()
    return adata[counts >= 10].copy()


def test_embedding_params_follow_the_clustering_config(tmp_path: object) -> None:
    config = make_config(tmp_path, n_pcs=12, n_neighbors=9, min_cells=3)  # type: ignore[arg-type]
    params = EmbeddingParams.from_config(config)
    assert params.n_pcs == 12
    assert params.n_neighbors == 9
    assert params.min_cells == 3
    assert params.umap_min_dist == config.umap_min_dist
    assert params.normalize_target_sum == config.normalize_target_sum


def test_normalize_table_cells_keeps_raw_counts_and_log_normalises(
    human_section: Section,
) -> None:
    adata = _table_cells(human_section)
    dense = adata.X.toarray()
    dense[:, 0] = 0  # gene 0 detected in no cell: dropped by min_cells
    adata.X = sparse.csr_matrix(dense)
    raw = adata.X.copy()
    normalize_table_cells(adata, EmbeddingParams(min_cells=1))
    assert "GENE000" not in adata.var_names
    kept = [int(name[4:]) for name in adata.var_names]
    np.testing.assert_array_equal(
        adata.layers["counts"].toarray(), raw[:, kept].toarray()
    )
    per_cell = np.expm1(adata.X.toarray()).sum(axis=1)
    np.testing.assert_allclose(per_cell, np.median(per_cell), rtol=1e-4)


def test_normalize_rejects_too_few_genes(human_section: Section) -> None:
    adata = _table_cells(human_section)[:, :1].copy()
    with pytest.raises(ValueError, match="Too few"):
        normalize_table_cells(adata, EmbeddingParams(min_cells=1))


def test_whole_section_qc_is_seeded_cpu_igraph_and_finds_structure(
    human_section: Section,
) -> None:
    adata = _table_cells(human_section)
    params = EmbeddingParams(n_pcs=10, n_neighbors=15)
    normalize_table_cells(adata, params)
    again = adata.copy()
    provenance = whole_section_qc(adata, resolution=0.5, seed=0, params=params)
    whole_section_qc(again, resolution=0.5, seed=0, params=params)

    assert provenance.engine == CPU_LEIDEN_ENGINE
    assert provenance.flavor == CPU_LEIDEN_FLAVOR
    assert provenance.n_iterations == CPU_LEIDEN_N_ITERATIONS == 2
    assert provenance.random_state == 0
    assert provenance.resolution == 0.5
    assert provenance.n_pcs_used == 10
    assert provenance.n_neighbors_used == 15
    assert provenance.hvg_subset is False
    assert provenance.gpu_used is False
    assert {"X_pca", "X_umap"} <= set(adata.obsm)
    assert {"connectivities", "distances"} <= set(adata.obsp)
    for key in (QC_LEIDEN_KEY, QC_BROAD_LEIDEN_KEY):
        assert isinstance(adata.obs[key].dtype, pd.CategoricalDtype)
    assert (
        adata.obs[QC_LEIDEN_KEY]
        .astype(str)
        .equals(adata.obs[QC_BROAD_LEIDEN_KEY].astype(str))
    )
    assert (
        adata.obs[QC_LEIDEN_KEY]
        .astype(str)
        .equals(again.obs[QC_LEIDEN_KEY].astype(str))
    )
    assert provenance.n_clusters == adata.obs[QC_LEIDEN_KEY].nunique()
    # What scanpy actually ran, not only what the provenance claims.
    ran = adata.uns[QC_LEIDEN_KEY]["params"]
    assert ran["n_iterations"] == 2
    assert ran["random_state"] == 0
    assert ran["resolution"] == 0.5
    assert adata.uns["pca"]["params"]["mask_var"] is None
    blocks = human_section.type_of_cell.loc[adata.obs_names]
    assert adjusted_rand_score(blocks, adata.obs[QC_LEIDEN_KEY]) > 0.5
    params_record = provenance.as_uns_params()
    assert params_record["engine"] == CPU_LEIDEN_ENGINE
    assert params_record["leiden_n_iterations"] == 2
    assert json.loads(str(params_record["engine_versions"]))["scanpy"]
    assert json.loads(provenance.to_json(key_added="leiden"))["key_added"] == "leiden"


def test_large_panels_use_an_hvg_subset(human_section: Section) -> None:
    adata = _table_cells(human_section)
    params = EmbeddingParams(
        n_pcs=8, n_neighbors=10, large_panel_genes=40, n_top_genes=20
    )
    normalize_table_cells(adata, params)
    mask = pca_gene_mask(adata, params)
    assert mask.sum() == 20
    provenance = whole_section_qc(adata, seed=0, params=params)
    assert provenance.hvg_subset is True
    assert provenance.n_genes_used == 20
    assert int(np.asarray(adata.uns["pca"]["params"]["mask_var"]).sum()) == 20
    small = EmbeddingParams(large_panel_genes=1000)
    assert pca_gene_mask(adata, small).all()


def test_whole_section_qc_rejects_tiny_inputs(human_section: Section) -> None:
    adata = _table_cells(human_section)[:2].copy()
    with pytest.raises(ValueError, match="at least 3 cells"):
        whole_section_qc(adata)


def test_branch_embedding_reuses_the_pca_and_skips_tiny_branches(
    human_section: Section,
) -> None:
    adata = _table_cells(human_section)
    params = EmbeddingParams(n_pcs=10, n_neighbors=15)
    normalize_table_cells(adata, params)
    whole_section_qc(adata, seed=0, params=params)
    mask = (human_section.type_of_cell.loc[adata.obs_names] == "ul_it").to_numpy()
    branch = branch_embedding(adata, mask, n_neighbors=10, seed=0)
    assert branch is not None
    assert branch.n_obs == int(mask.sum())
    np.testing.assert_array_equal(branch.obsm["X_pca"], adata.obsm["X_pca"][mask])
    assert not np.array_equal(branch.obsm["X_umap"], adata.obsm["X_umap"][mask])
    assert branch.obsp["connectivities"].shape == (branch.n_obs, branch.n_obs)
    tiny = np.zeros(adata.n_obs, dtype=bool)
    tiny[:2] = True
    assert branch_embedding(adata, tiny) is None
    with pytest.raises(ValueError, match="shape"):
        branch_embedding(adata, mask[:-1])
    del adata.obsm["X_pca"]
    with pytest.raises(KeyError, match="X_pca"):
        branch_embedding(adata, mask)


def test_library_versions_resolve_igraph_distribution() -> None:
    versions = library_versions(("scanpy", "igraph", "not_a_package_xyz"))
    assert versions["scanpy"]
    assert versions["igraph"]
    assert versions["not_a_package_xyz"] is None


def test_dense_counts_are_supported(human_section: Section) -> None:
    adata = _table_cells(human_section)
    adata.X = adata.X.toarray()
    params = EmbeddingParams(n_pcs=5, n_neighbors=10)
    normalize_table_cells(adata, params)
    assert not sparse.issparse(adata.layers["counts"])
    provenance = whole_section_qc(adata, seed=1, params=params)
    assert provenance.random_state == 1


def test_section_fixture_has_table_and_low_count_objects() -> None:
    section = make_section("mouse", n_low=5)
    in_table = section.labels["in_table"].to_numpy()
    assert int((~in_table).sum()) == 5
