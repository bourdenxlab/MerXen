"""Known linear means, atlas joins, and cache integrity for pciSeq references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.analysis import pciseq_reference as reference_module
from merxen.analysis.allen_reference_assets import AtlasAssets
from merxen.analysis.pciseq_reference import prepare_pciseq_reference
from merxen.config import PciSeqReferenceConfig


def _assets(tmp_path: Path, source: str) -> AtlasAssets:
    folder = tmp_path / source
    folder.mkdir()
    paths: dict[str, Path] = {}

    def save(key: str, frame: pd.DataFrame) -> None:
        paths[key] = folder / f"{key}.csv"
        frame.to_csv(paths[key], index=False)

    frame = pd.DataFrame(
        {
            "cell_label": ["c1", "c2", "c3", "c4", "c5", "c7"],
            "cluster_alias": ["1", "1", "1", "2", "2", "1"],
            "donor_label": ["d1", "d1", "d2", "d3", "d3", "d2"],
            "library_label": ["l1", "l1", "l2", "l3", "l4", "l2"],
            "region_of_interest_label": ["R1", "R1", "R1", "R2", "R1", "R1"],
        }
    )
    if source == "wmb":
        frame = frame.rename(
            columns={"region_of_interest_label": "region_of_interest_acronym"}
        )
        save(
            "region_of_interest_metadata",
            pd.DataFrame(
                {
                    "label": ["WMB-R1", "WMB-R2"],
                    "acronym": ["R1", "R2"],
                    "name": ["Region one", "Region two"],
                }
            ),
        )
        levels = ["class", "subclass", "cluster", "neurotransmitter"]
    elif source == "whb":
        save(
            "region_of_interest_structure_map",
            pd.DataFrame(
                {
                    "region_of_interest_label": ["R1", "R2"],
                    "structure_symbol": ["r_one", "r_two"],
                    "structure_name": ["Region one", "Region two"],
                }
            ),
        )
        levels = ["supercluster", "cluster", "subcluster", "neurotransmitter"]
    else:
        save("cell_to_cluster_membership", frame[["cell_label", "cluster_alias"]])
        frame = frame.drop(columns=["cluster_alias", "region_of_interest_label"])
        save(
            "library",
            pd.DataFrame(
                {
                    "library_label": ["l1", "l2", "l3", "l4"],
                    "donor_label": ["d1", "d2", "d3", "d3"],
                    "region_of_interest_label": ["R1", "R1", "R2", "R1"],
                    "region_of_interest_name": [
                        "Region one",
                        "Region one",
                        "Region two",
                        "Region one",
                    ],
                }
            ),
        )
        save(
            "disease",
            pd.DataFrame(
                {
                    "donor_label": ["d1", "d2", "d3"],
                    "Overall AD neuropathological Change": ["Not AD", "High", "Low"],
                    "Cognitive Status": ["No dementia", "Dementia", "No dementia"],
                }
            ),
        )
        levels = ["Class", "Subclass", "Supertype"]
    save("cell_metadata", frame)
    save(
        "gene",
        pd.DataFrame(
            {"gene_identifier": ["e1", "e2", "e3"], "gene_symbol": ["A", "A", "B"]}
        ),
    )
    rows = []
    for order, level in enumerate(levels):
        for alias, name in [("1", "type_t"), ("2", "type_u")]:
            rows.append(
                {
                    "cluster_alias": alias,
                    "cluster_annotation_term_set_name": level,
                    "cluster_annotation_term_set_label": f"native_{level}",
                    "cluster_annotation_term_label": f"{level}_{alias}",
                    "cluster_annotation_term_name": name,
                    "term_set_order": order,
                }
            )
    membership = pd.DataFrame(rows)
    save(
        "cluster_to_cluster_annotation_membership",
        membership.drop(columns="term_set_order"),
    )
    save("cluster_annotation_term", membership)
    counts = sparse.csr_matrix(
        np.array(
            [
                [10, 0, 90],
                [30, 0, 70],
                [90, 0, 10],
                [0, 100, 0],
                [0, 0, 100],
                [2, 3, 5],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    expression = folder / "expression-raw.h5ad"
    matrix = ad.AnnData(
        counts,
        obs=pd.DataFrame(
            index=pd.Index(
                ["c1", "c2", "c3", "c4", "c5", "orphan", "c7"], name="cell_label"
            )
        ),
        var=pd.DataFrame(index=pd.Index(["e1", "e2", "e3"], name="gene_identifier")),
    )
    matrix.uns["normalization"] = "raw"
    matrix.write_h5ad(expression)
    return AtlasAssets(
        source,
        tmp_path,
        "fixture_release",
        folder / "manifest.json",
        (expression,),
        paths,
        {"source_fingerprint": "fixture_hash", "files": {}},
    )


def _config(tmp_path: Path, source: str, **changes: object) -> PciSeqReferenceConfig:
    values: dict[str, object] = {
        "species": "mouse" if source == "wmb" else "human",
        "source": source,
        "reference_root": tmp_path,
        "region_labels": ["R1"],
        "min_cells_per_type": 1,
        "chunk_rows": 2,
        "gene_names": ["A"],
        "output_dir": tmp_path / "output",
    }
    values.update(changes)
    return PciSeqReferenceConfig.model_validate(values)


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, **changes: object
) -> tuple[dict[str, Path], dict[str, object]]:
    assets = _assets(tmp_path, source)
    monkeypatch.setattr(
        reference_module, "ensure_atlas_assets", lambda *args, **kwargs: assets
    )
    paths = prepare_pciseq_reference(_config(tmp_path, source, **changes))
    return paths, json.loads(paths["manifest"].read_text())


@pytest.mark.parametrize(
    "source,native",
    [
        ("wmb", "native_subclass"),
        ("whb", "native_cluster"),
        ("seaad", "native_Subclass"),
    ],
)
def test_all_atlases_compute_donor_balanced_linear_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, native: str
) -> None:
    paths, manifest = _run(tmp_path, monkeypatch, source)
    profile = pd.read_csv(paths["reference_csv"], index_col=0)
    assert profile.columns.tolist() == ["type_t"]
    assert profile.loc["A", "type_t"] == pytest.approx(5500)
    assert manifest["resolved_taxonomy_level"]["native_label"] == native
    diagnostic = manifest["diagnostics"]
    assert diagnostic["raw_without_metadata"] == 1
    assert diagnostic["empty_library_cells"] == 1
    assert diagnostic["donor_balance"] is True
    support = pd.read_csv(paths["support"])
    assert support.set_index("type_name").loc["type_t", "n_cells"] == 3
    assert support.set_index("type_name").loc["type_t", "n_donors"] == 2
    assert support.set_index("type_name").loc["type_u", "status"] == "zero_expression"
    expected = hashlib.sha256(paths["reference_csv"].read_bytes()).hexdigest()
    assert manifest["reference_sha256"] == expected


def test_unbalanced_profile_weights_cells_equally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _run(tmp_path, monkeypatch, "wmb", donor_balance=False)
    profile = pd.read_csv(paths["reference_csv"], index_col=0)
    assert profile.loc["A", "type_t"] == pytest.approx(13000 / 3)


def test_complete_gene_means_sum_duplicate_symbols_and_keep_other_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _run(tmp_path, monkeypatch, "whb", gene_names=None)
    profile = pd.read_csv(paths["reference_csv"], index_col=0)
    assert profile.index.tolist() == ["A", "B"]
    assert profile.loc["B", "type_t"] == pytest.approx(4500)
    assert profile.loc["B", "type_u"] == pytest.approx(10000)
    np.testing.assert_allclose(profile.sum(axis=0), 10000)


def test_ensembl_mode_keeps_distinct_gene_identifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _run(
        tmp_path, monkeypatch, "wmb", gene_names=None, gene_id_mode="ensembl"
    )
    assert pd.read_csv(paths["reference_csv"], index_col=0).index.tolist() == [
        "e1",
        "e2",
        "e3",
    ]


@pytest.mark.parametrize(
    "cohort,expected,donors", [("all", 5500, 3), ("ad", 9000, 1), ("control", 2000, 1)]
)
def test_seaad_cohort_definitions_and_counts_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cohort: str,
    expected: int,
    donors: int,
) -> None:
    paths, manifest = _run(tmp_path, monkeypatch, "seaad", seaad_cohort=cohort)
    assert pd.read_csv(paths["reference_csv"], index_col=0).loc[
        "A", "type_t"
    ] == pytest.approx(expected)
    assert manifest["diagnostics"]["eligible_donor_count"] == donors
    assert manifest["diagnostics"]["cohort_definition"]


def test_roi_alias_is_explicit_and_unknown_roi_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _assets(tmp_path, "wmb")
    monkeypatch.setattr(
        reference_module, "ensure_atlas_assets", lambda *args, **kwargs: assets
    )
    paths = prepare_pciseq_reference(
        _config(tmp_path, "wmb", region_labels=["Region one"])
    )
    assert json.loads(paths["manifest"].read_text())["diagnostics"][
        "resolved_regions"
    ] == ["R1"]
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        prepare_pciseq_reference(_config(tmp_path, "wmb", region_labels=["unknown"]))


def test_nonexistent_semantic_level_is_not_cross_atlas_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _assets(tmp_path, "whb")
    monkeypatch.setattr(
        reference_module, "ensure_atlas_assets", lambda *args, **kwargs: assets
    )
    with pytest.raises(ValueError, match="available native levels"):
        prepare_pciseq_reference(_config(tmp_path, "whb", level="subclass"))
    paths = prepare_pciseq_reference(_config(tmp_path, "whb", level="finest"))
    assert (
        json.loads(paths["manifest"].read_text())["resolved_taxonomy_level"]["name"]
        == "subcluster"
    )


def test_minimum_type_support_is_applied_after_actual_roi_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _run(tmp_path, monkeypatch, "wmb", min_cells_per_type=2, gene_names=None)
    support = pd.read_csv(paths["support"]).set_index("type_name")
    assert support.loc["type_u", "status"] == "below_min_cells"
    assert pd.read_csv(paths["reference_csv"], index_col=0).columns.tolist() == [
        "type_t"
    ]


@pytest.mark.parametrize("invalid", ["fractional", "log_declared", "duplicate_cell"])
def test_invalid_raw_inputs_fail_without_changing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    assets = _assets(tmp_path, "wmb")
    path = assets.expression_paths[0]
    matrix = ad.read_h5ad(path)
    if invalid == "fractional":
        matrix.X.data[0] = 1.5
    elif invalid == "log_declared":
        matrix.uns["normalization"] = "log2(CPM+1)"
    else:
        matrix.obs_names = ["c1", "c1", "c3", "c4", "c5", "orphan", "c7"]
    matrix.write_h5ad(path)
    before = path.read_bytes()
    monkeypatch.setattr(
        reference_module, "ensure_atlas_assets", lambda *args, **kwargs: assets
    )
    with pytest.raises(ValueError, match="raw integer|normalization|duplicated"):
        prepare_pciseq_reference(_config(tmp_path, "wmb"))
    assert path.read_bytes() == before


def test_cached_reference_is_reused_and_complete_checksums_repair_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _assets(tmp_path, "wmb")
    monkeypatch.setattr(
        reference_module, "ensure_atlas_assets", lambda *args, **kwargs: assets
    )
    config = _config(tmp_path, "wmb")
    first = prepare_pciseq_reference(config)
    manifest = json.loads(first["manifest"].read_text())
    cache = tmp_path / "pciseq_prepared" / manifest["cache_key"]
    original = reference_module._aggregate
    monkeypatch.setattr(
        reference_module,
        "_aggregate",
        lambda *args: pytest.fail("valid cache should be reused"),
    )
    prepare_pciseq_reference(config, tmp_path / "second")
    broken = json.loads((cache / "reference_manifest.json").read_text())
    broken["output_checksums"] = {}
    (cache / "reference_manifest.json").write_text(json.dumps(broken))
    monkeypatch.setattr(reference_module, "_aggregate", original)
    repaired = prepare_pciseq_reference(config)
    assert json.loads(repaired["manifest"].read_text())["output_checksums"][
        "reference.csv"
    ]


def test_source_fingerprint_changes_invalidate_profile_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _assets(tmp_path, "wmb")
    monkeypatch.setattr(
        reference_module, "ensure_atlas_assets", lambda *args, **kwargs: assets
    )
    config = _config(tmp_path, "wmb")
    first = json.loads(prepare_pciseq_reference(config)["manifest"].read_text())[
        "cache_key"
    ]
    assets.provenance["source_fingerprint"] = "changed_source_hash"
    second = json.loads(prepare_pciseq_reference(config)["manifest"].read_text())[
        "cache_key"
    ]
    assert first != second


def test_custom_profile_roi_allowlist_and_linear_values_preserved(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "custom.csv"
    pd.DataFrame(
        {"type_a": [10.0, 20.0], "type_b": [30.0, 40.0]}, index=["A", "B"]
    ).to_csv(matrix)
    metadata = tmp_path / "custom_regions.csv"
    pd.DataFrame({"cell_type": ["type_a", "type_b"], "region": ["R1", "R2"]}).to_csv(
        metadata, index=False
    )
    config = PciSeqReferenceConfig(
        species="human",
        source="custom",
        reference_root=tmp_path,
        reference_path=matrix,
        custom_metadata_path=metadata,
        region_labels=["R1"],
        output_dir=tmp_path / "output",
    )
    paths = prepare_pciseq_reference(config)
    profile = pd.read_csv(paths["reference_csv"], index_col=0)
    assert profile.columns.tolist() == ["type_a"]
    assert profile.loc["A", "type_a"] == 10
    assert (
        "Supplied precomputed linear"
        in json.loads(paths["manifest"].read_text())["diagnostics"]["normalization"]
    )


@pytest.mark.parametrize(
    "broken_checksums", [{}, {"reference.csv": "0" * 64}, None, []]
)
def test_cache_rejects_incomplete_or_malformed_checksum_metadata(
    tmp_path: Path, broken_checksums: object
) -> None:
    cache = tmp_path / "cache_key"
    cache.mkdir()
    (cache / "reference.csv").write_text("gene,type\nA,1\n")
    (cache / "class_support.csv").write_text("type_id\ntype\n")
    (cache / "reference_manifest.json").write_text(
        json.dumps(
            {
                "algorithm_version": reference_module._ALGORITHM_VERSION,
                "cache_key": "cache_key",
                "reference_sha256": "0" * 64,
                "output_checksums": broken_checksums,
            }
        )
    )
    assert not reference_module._valid_cache(cache)


def test_types_with_no_expression_rows_remain_in_support_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _assets(tmp_path, "wmb")
    metadata_path = assets.metadata_paths["cell_metadata"]
    metadata = pd.read_csv(metadata_path)
    metadata.loc[metadata.cluster_alias == 2, "cell_label"] = [
        "unavailable_one",
        "unavailable_two",
    ]
    metadata.to_csv(metadata_path, index=False)
    monkeypatch.setattr(
        reference_module, "ensure_atlas_assets", lambda *args, **kwargs: assets
    )
    paths = prepare_pciseq_reference(_config(tmp_path, "wmb"))
    support = pd.read_csv(paths["support"]).set_index("type_name")
    assert support.loc["type_u", "status"] == "no_expression_cells"
    assert support.loc["type_u", "n_cells"] == 0


def test_custom_output_cannot_overwrite_source_profile(tmp_path: Path) -> None:
    matrix = tmp_path / "reference.csv"
    pd.DataFrame({"type_a": [10.0]}, index=["A"]).to_csv(matrix)
    original = matrix.read_bytes()
    config = PciSeqReferenceConfig(
        species="mouse",
        source="custom",
        reference_root=tmp_path,
        reference_path=matrix,
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="overwrite an input"):
        prepare_pciseq_reference(config)
    assert matrix.read_bytes() == original
