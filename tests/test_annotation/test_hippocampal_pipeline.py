"""Full-panel queries and stage handoffs preserve raw data and abstentions."""

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation.hippocampal_pipeline import (
    full_panel_counts,
    load_resources,
    report_resolved,
    write_json,
)
from merxen.clustering.map_first import is_unassigned_state


def test_full_query_reorders_gene_ids_without_losing_zero_columns_or_counts() -> None:
    genes = [f"ENSG{i:011d}" for i in range(18049)]
    matrix = sparse.csr_matrix(([3, 7], ([0, 1], [18048, 0])), shape=(3, 18049))
    data = ad.AnnData(matrix, var=pd.DataFrame({"gene_ids": genes[::-1]}))
    panel = SimpleNamespace(ensembl_ids=genes, n_genes=len(genes))
    result = full_panel_counts(data, panel)
    assert result.shape == (3, 18049)
    assert result[0, 0] == 3 and result[1, 18048] == 7
    np.testing.assert_array_equal(np.asarray(result.sum(axis=1)).ravel(), [3, 7, 0])
    with pytest.raises(ValueError, match="every full-panel gene"):
        full_panel_counts(data[:, :-1], panel)
    data.var.iloc[0, 0] = data.var.iloc[1, 0]
    with pytest.raises(ValueError, match="exactly once"):
        full_panel_counts(data, panel)


@pytest.mark.parametrize("value", [-1, 0.5, float("nan")])
def test_normalized_negative_and_nonfinite_queries_are_refused(value: float) -> None:
    data = ad.AnnData(
        sparse.csr_matrix([[value, 0]]), var=pd.DataFrame(index=["a", "b"])
    )
    panel = SimpleNamespace(ensembl_ids=["a", "b"], n_genes=2)
    with pytest.raises(ValueError, match="raw integer counts"):
        full_panel_counts(data, panel)


def test_staged_resource_symlink_resolves_relative_paths_at_original_manifest(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    reference = dict(
        reference_id="example",
        taxonomy_id="regional",
        regions=["HIP"],
        precompute="stats.h5",
        lookup="lookup.json",
        tree="tree.json",
        policy="policy.json",
        qualification="qualification.json",
    )
    write_json(
        original / "resources.json",
        dict(
            profile="human_hippocampus_whole_panel_v1",
            panel="panel.json",
            seaad=reference,
            whb=reference,
            ctm_version="1.7.2",
        ),
    )
    staged = tmp_path / "staged.json"
    staged.symlink_to(original / "resources.json")
    resources = load_resources(staged)
    assert resources.panel == original / "panel.json"
    assert resources.seaad.tree == original / "tree.json"


def test_requested_mender_without_completion_is_a_report_failure(
    tmp_path: Path,
) -> None:
    write_json(tmp_path / "manifest.json", {"samples": {}})
    with pytest.raises(ValueError, match="MENDER was requested"):
        report_resolved(tmp_path, tmp_path / "report", expect_mender=True)
    report_resolved(tmp_path, tmp_path / "report")
    assert (tmp_path / "report/annotation_report.html").exists()


def test_unknown_hippocampal_state_is_excluded_by_existing_mender_policy() -> None:
    assert is_unassigned_state("Mixed/Unknown:unresolved")
    assert not is_unassigned_state("Glutamatergic:unresolved")


def test_declared_gene_release_resolution_must_reproduce_its_digest(
    tmp_path: Path,
) -> None:
    from merxen.annotation.gene_ids import (
        GeneIdSources,
        load_gene_table,
        load_overrides,
        resolve_gene_ids,
    )
    from merxen.annotation.panel import DeclaredResolution

    table_path = tmp_path / "genes.csv"
    table_path.write_text(
        "gene_symbol,gene_identifier\nGENEA,ENSG00000000001\nGENEB,ENSG00000000002\n"
    )
    table = load_gene_table(table_path, "human")
    features = pd.DataFrame(
        {
            "name": ["GENEA", "GENEB"],
            "symbol": ["GENEA", "GENEB"],
            "native_value": ["ENSG00000000003", "ENSG00000000002"],
        }
    )
    resolution = resolve_gene_ids(
        features,
        "human",
        GeneIdSources(
            species="human",
            gene_tables={"human": table},
            overrides=load_overrides("human"),
        ),
        native_id_column="gene_ids",
    )
    record = DeclaredResolution(
        resolution_table_sha256=resolution.table_sha256(),
        gene_tables={"human": str(table_path)},
    )
    panel = SimpleNamespace(
        ensembl_ids=["ENSG00000000001", "ENSG00000000002"],
        n_genes=2,
        declared_resolutions={"atera": record},
    )
    data = ad.AnnData(
        sparse.csr_matrix([[4, 9]]),
        var=pd.DataFrame(
            {"gene_ids": features.native_value.tolist()}, index=features.name
        ),
    )
    counts = full_panel_counts(data, panel)
    np.testing.assert_array_equal(counts.toarray(), [[4, 9]])
    panel.declared_resolutions["atera"] = record.model_copy(
        update={"resolution_table_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="declared panel digest"):
        full_panel_counts(data, panel)


def test_depth_and_cross_atlas_tables_keep_the_full_object_denominator(
    tmp_path: Path,
) -> None:
    from merxen.annotation.hippocampal_report import outcome_tables

    labels = pd.DataFrame(
        {
            "n_counts": [0, 25, 6000],
            "ct_final_level": ["none", "class", "subclass"],
            "ct_final_name": ["Mixed/Unknown", "Glutamatergic", "Dentate Gyrus"],
            "ct_seaad_class_name": [None, "Glutamatergic", "Glutamatergic"],
            "ct_whb_supercluster_name": [None, None, "DG"],
        }
    )
    tables = outcome_tables(labels, tmp_path)
    assert all(table.n_objects.sum() == 3 for table in tables.values())


def test_compute_preserves_id_alignment_and_withheld_categorical_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.annotation.hippocampal_pipeline import compute_prepared
    from merxen.clustering import representation

    def qc(data: ad.AnnData, **kwargs: object) -> None:
        data.obsm["X_umap"] = np.zeros((data.n_obs, 2))

    monkeypatch.setattr(representation, "whole_section_qc", qc)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    data = ad.AnnData(
        sparse.csr_matrix([[3, 5, 1], [0, 0, 0], [4, 9, 2], [2, 2, 2]]),
        obs=pd.DataFrame(index=["a", "empty", "b", "c"]),
    )
    data.obsm["spatial"] = np.zeros((4, 2))
    data.write_h5ad(prepared / "cells.h5ad")
    write_json(prepared / "manifest.json", {"samples": {"TEST": "cells.h5ad"}})
    folder = tmp_path / "resolved/TEST"
    folder.mkdir(parents=True)
    labels = pd.DataFrame(
        {
            "cell_id": ["b", "empty", "a", "c"],
            "n_counts": [15, 0, 9, 6],
            "ct_branch": [
                "Glutamatergic",
                "Mixed/Unknown",
                "Glutamatergic",
                "Glutamatergic",
            ],
            "ct_leaf": ["unresolved", "unresolved", "DG", "unresolved"],
            "ct_seaad_subclass_name": pd.Categorical([None, None, "DG", None]),
            "ct_seaad_subclass_accepted": [False, False, True, False],
        }
    )
    labels.to_parquet(folder / "celltype_labels.parquet", index=False)
    (folder / "annotation_manifest.json").write_text("{}")
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "pair_id": "TEST",
            "output_dir": "unused",
            "samples": [
                {
                    "sample_id": "TEST",
                    "platform": "ATERA",
                    "zarr_path": "unused",
                    "anatomical_region": "hippocampus",
                    "table_key": "table_MOSAIK_cellpose",
                    "shape_key": "MOSAIK_cellpose",
                }
            ],
            "min_counts": 1,
            "min_cells": 1,
        },
    )
    compute_prepared(config, prepared, folder.parent, tmp_path / "computed")
    result = ad.read_h5ad(tmp_path / "computed/atera/TEST_clustered.h5ad")
    assert result.obs_names.tolist() == ["a", "b", "c"]
    assert result.obs.ct_leaf.astype(str).tolist() == ["DG", "unresolved", "unresolved"]
    assert not bool(result.obs.loc["b", "ct_seaad_subclass_accepted"])
    assert result.layers["counts"].sum() == 30
    assert (
        result.uns["merxen_clustering_squidpy"]["table_key"] == "table_MOSAIK_cellpose"
    )
    labels.loc[labels.cell_id.eq("b"), "n_counts"] = 16
    labels.to_parquet(folder / "celltype_labels.parquet", index=False)
    with pytest.raises(ValueError, match="Prepared counts differ"):
        compute_prepared(config, prepared, folder.parent, tmp_path / "invalid")


def test_hippocampal_domain_provenance_reaches_mender_without_frontal_trust() -> None:
    from merxen.clustering.map_first import downstream_annotation_summary

    summary = downstream_annotation_summary(
        {
            "merxen_hierarchical_clustering": {
                "mode": "map_first",
                "profile": "human_hippocampus_whole_panel_v1",
            }
        }
    )
    assert summary["panel_trust"] == "provisional"
    assert summary["evidence_basis"] == "existing_atlas_diagnostics_reused"
    assert summary["leaf_level"] == "seaad_subclass"
    assert summary["spatial_ground_truth_validated"] is False
    assert downstream_annotation_summary({}) is None


def test_selective_store_read_and_write_preserve_unrelated_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import spatialdata as sd
    from spatialdata.models import TableModel

    from merxen.annotation.hippocampal_io import read_annotation_store
    from merxen.io.spatialdata_io import write_or_replace_element

    root = tmp_path / "store.zarr"
    chosen = TableModel.parse(ad.AnnData(sparse.csr_matrix([[3, 4], [5, 6]])))
    other = TableModel.parse(ad.AnnData(sparse.csr_matrix([[7, 8]])))
    sd.SpatialData(tables={"chosen": chosen, "unrelated": other}).write(root)
    actual_read = ad.read_zarr
    read_paths = []

    def read_table(path: Path) -> ad.AnnData:
        read_paths.append(Path(path).name)
        return actual_read(path)

    monkeypatch.setattr(ad, "read_zarr", read_table)
    selected = read_annotation_store(root, "chosen")
    assert read_paths == ["chosen"]
    assert set(selected.tables) == {"chosen"}
    assert selected.path == root
    changed = selected.tables["chosen"].copy()
    changed.obs["new_annotation"] = pd.Categorical(["A", "B"])
    write_or_replace_element(selected, "chosen", "tables", changed)
    checked = sd.read_zarr(root)
    np.testing.assert_array_equal(checked.tables["unrelated"].X.toarray(), [[7, 8]])
    assert checked.tables["chosen"].obs.new_annotation.tolist() == ["A", "B"]
    assert not read_annotation_store(root, "missing").tables
