"""Tests for the report input readers (``report_inputs``; plan §3.6)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation.report_inputs import (
    ReportInputError,
    load_heldout,
    panel_gene_lookup,
    read_clustered_table,
    resolved_gene_ids,
)


def test_panel_gene_lookup_reads_every_panel_file(tmp_path: Path) -> None:
    (tmp_path / "panel_genes.json").write_text(
        json.dumps(
            {
                "ensembl_ids": ["ENSMUSG01", "ENSMUSG02"],
                "symbols": ["Gfap", "Aqp4"],
                "symbols_by_platform": {"MERSCOPE": ["GFAP", "Aqp4"]},
            }
        )
    )
    (tmp_path / "panel_genes_setc.json").write_text(
        json.dumps({"ensembl_ids": ["ENSMUSG03"], "symbols": ["Olig2"]})
    )
    (tmp_path / "panel_genes_bad.json").write_text(
        json.dumps({"ensembl_ids": ["X"], "symbols": ["a", "b"]})
    )
    lookup = panel_gene_lookup([tmp_path, None, tmp_path / "missing"])
    assert lookup == {"gfap": "ENSMUSG01", "aqp4": "ENSMUSG02", "olig2": "ENSMUSG03"}
    assert resolved_gene_ids(["Gfap", "ENSMUSG00000099999", "Unknown"], lookup) == [
        "ENSMUSG01",
        "ENSMUSG00000099999",
        "Unknown",
    ]


def test_read_clustered_table_reads_ids_counts_and_coordinates(tmp_path: Path) -> None:
    obs = pd.DataFrame(index=pd.Index(["c1", "c2", "c3"]))
    var = pd.DataFrame(
        {"ensembl_id": pd.Categorical(["ENSG01", "", "ENSG03"])},
        index=pd.Index(["A", "B", "C"]),
    )
    counts = sparse.csr_matrix(np.array([[1, 0, 2], [0, 0, 1], [3, 1, 0]]))
    adata = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.obsm["spatial"] = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    adata.uns["merxen_clustering_squidpy"] = {"shape_key": "MOSAIK_x_aligned_nonrigid"}
    adata.uns["merxen_hierarchical_clustering"] = {"mode": "map_first", "n_leaves": 4}
    path = tmp_path / "clustered.h5ad"
    adata.write_h5ad(path)
    table = read_clustered_table(path)
    assert list(table.obs_names) == ["c1", "c2", "c3"]
    assert table.gene_ids == ["ENSG01", "B", "ENSG03"]
    assert table.gene_symbols == ["A", "B", "C"]
    np.testing.assert_array_equal(table.counts.toarray(), counts.toarray())
    np.testing.assert_allclose(table.xy[:, 1], [1.0, 3.0, 5.0])
    assert table.shape_key == "MOSAIK_x_aligned_nonrigid"
    assert table.hierarchy == {"mode": "map_first", "n_leaves": 4}
    light = read_clustered_table(path, with_counts=False)
    assert light.counts is None and light.xy is not None


def test_load_heldout_keeps_the_pair_and_checks_columns(tmp_path: Path) -> None:
    path = tmp_path / "heldout.csv"
    pd.DataFrame(
        {
            "pair": ["P1", "P2"],
            "platform": ["MERSCOPE", "XENIUM"],
            "broad_class": ["Neurons", "Neurons"],
            "fold": [10.0, 3.0],
            "auroc": [0.9, 0.7],
        }
    ).to_csv(path, index=False)
    frame = load_heldout(path, "P1")
    assert frame is not None and list(frame["pair"]) == ["P1"]
    assert load_heldout(path, "P9") is None
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"pair": ["P1"]}).to_csv(bad, index=False)
    with pytest.raises(ReportInputError, match="lacks the columns"):
        load_heldout(bad, "P1")
