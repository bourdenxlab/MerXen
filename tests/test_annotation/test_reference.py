"""Tests for the reference bundle builders (plan §3.2, §5.6, §7.1, §12 M2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from merxen.annotation import reference
from merxen.annotation.config import AnnotationConfig, AnnotationReferenceSpec
from merxen.annotation.panel import AnnotationPanel, compute_panel_hash
from merxen.annotation.reference import (
    CollapsedParent,
    LookupValidationError,
    PinnedFile,
    PinnedFileError,
    PrecomputedStats,
    ReferenceBuildError,
    SourceOptions,
    TaxonomyTreeView,
    broad_class_detection,
    builder_for,
    ensure_pinned_file,
    filter_lookup_to_tree,
    lognormal_mean_cpm,
    lookup_sha256,
    mapping_tree,
    merfish_region,
    negative_gene_table,
    node_vocab_table,
    prepare_reference_spec,
    read_precomputed_stats,
    reference_profiles_from_stats,
    region_share_tables,
    sample_wmb_training_cells,
    validate_lookup,
    write_panel_stub_h5ad,
)
from merxen.annotation.store import (
    BUNDLE_MANIFEST_NAME,
    ReferenceStore,
    resolve_builder,
)
from merxen.cli import main as cli_main

# --------------------------------------------------------------------------
# Synthetic taxonomies and precomputes

# WHB (CCN202210140): three superclusters with real vocab labels.
UL_IT = "CS202210140_476"  # Upper-layer intratelencephalic (Neurons)
ASTRO = "CS202210140_470"  # Astrocyte
MICRO = "CS202210140_464"  # Microglia
SUPC, CLUS, SUBC = "CCN202210140_SUPC", "CCN202210140_CLUS", "CCN202210140_SUBC"
WHB_TREE: dict[str, Any] = {
    "hierarchy": [SUPC, CLUS, SUBC],
    SUPC: {UL_IT: ["c1", "c2"], ASTRO: ["c3"], MICRO: ["c4", "c5"]},
    CLUS: {
        "c1": ["s1", "s2"],
        "c2": ["s3"],
        "c3": ["s4", "s5"],
        "c4": ["s6"],
        "c5": ["s7"],
    },
    SUBC: {f"s{index}": [] for index in range(1, 8)},
    "name_mapper": {
        SUPC: {
            UL_IT: {"name": "Upper-layer intratelencephalic"},
            ASTRO: {"name": "Astrocyte"},
            MICRO: {"name": "Microglia"},
        },
        CLUS: {f"c{index}": {"name": f"cluster {index}"} for index in range(1, 6)},
    },
}
# SEA-AD Multiregion (CCN20260630).
L0, L1, L2 = "CCN20260630_LEVEL_0", "CCN20260630_LEVEL_1", "CCN20260630_LEVEL_2"
SEAAD_TREE: dict[str, Any] = {
    "hierarchy": [L0, L1, L2],
    L0: {
        "CS20260630_CLAS_002": ["CS20260630_SCLA_010"],
        "CS20260630_CLAS_003": [
            "CS20260630_SCLA_023",
            "CS20260630_SCLA_028",
            "CS20260630_SCLA_029",
        ],
    },
    L1: {
        "CS20260630_SCLA_010": ["t_it"],
        "CS20260630_SCLA_023": ["t_astro"],
        "CS20260630_SCLA_028": ["t_vlmc", "t_peri"],
        "CS20260630_SCLA_029": ["t_micro"],
    },
    L2: {name: [] for name in ("t_it", "t_astro", "t_vlmc", "t_peri", "t_micro")},
    "name_mapper": {
        L1: {
            "CS20260630_SCLA_010": {"name": "L2/3 IT"},
            "CS20260630_SCLA_023": {"name": "Astrocyte"},
            "CS20260630_SCLA_028": {"name": "VLMC & Perivascular"},
            "CS20260630_SCLA_029": {"name": "Immune"},
        },
        L2: {
            "t_it": {"name": "L2/3 IT_1"},
            "t_astro": {"name": "Astro_1"},
            "t_vlmc": {"name": "VLMC_1"},
            "t_peri": {"name": "Pericyte_1"},
            "t_micro": {"name": "Micro-PVM_1"},
        },
    },
}
GENES = [f"ENSG{index:011d}" for index in range(1, 11)]
ABSENT_GENE = "ENSG99999999999"


def leaves_of(tree: Mapping[str, Any]) -> list[str]:
    """Return the leaf labels of a ctm-style tree dict, sorted."""
    return sorted(tree[tree["hierarchy"][-1]])


def write_precompute(
    path: Path,
    tree: Mapping[str, Any],
    genes: Sequence[str],
    *,
    n_cells: Mapping[str, int],
    detect: Callable[[str, int], float] | None = None,
    level: Callable[[str, int], float] | None = None,
    with_detection: bool = True,
) -> Path:
    """Write a synthetic precompute with consistent sum / sumsq / gt0.

    Each leaf x gene has ``gt0 = round(detect * n)`` detecting cells whose
    log2(CPM + 1) equals ``level`` exactly (variance 0).
    """
    leaves = leaves_of(tree)
    detect = detect or (lambda leaf, gene: 0.5)
    level = level or (lambda leaf, gene: 2.0 + 0.1 * gene)
    n = np.array([n_cells[leaf] for leaf in leaves], dtype=np.int64)
    gt0 = np.array(
        [
            [round(detect(leaf, g) * n_cells[leaf]) for g in range(len(genes))]
            for leaf in leaves
        ],
        dtype=np.int64,
    )
    values = np.array(
        [[level(leaf, g) for g in range(len(genes))] for leaf in leaves], dtype=float
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("taxonomy_tree", data=json.dumps(tree).encode("utf-8"))
        handle.create_dataset("col_names", data=json.dumps(list(genes)).encode("utf-8"))
        handle.create_dataset(
            "cluster_to_row",
            data=json.dumps({leaf: row for row, leaf in enumerate(leaves)}).encode(),
        )
        handle.create_dataset("n_cells", data=n)
        handle.create_dataset("sum", data=gt0 * values)
        if with_detection:
            handle.create_dataset("sumsq", data=gt0 * values**2)
            handle.create_dataset("gt0", data=gt0)
    return path


def make_panel(
    ids: Sequence[str], *, species: str = "human", kind: str = "intersection"
) -> AnnotationPanel:
    """Return a declared panel on the given IDs."""
    ordered = sorted(ids)
    return AnnotationPanel(
        name=kind,
        kind=kind,  # type: ignore[arg-type]
        species=species,  # type: ignore[arg-type]
        platforms=["MERSCOPE", "XENIUM"],
        sample_ids=["S_M", "S_X"],
        panel_mode="intersection",
        panel_hash=compute_panel_hash(ordered),
        n_genes=len(ordered),
        ensembl_ids=ordered,
        symbols=[f"SYM{gene[-3:]}" for gene in ordered],
    )


def whb_view() -> TaxonomyTreeView:
    return TaxonomyTreeView.from_tree_dict(WHB_TREE)


# --------------------------------------------------------------------------
# Taxonomy tree view


def test_tree_view_reads_levels_children_and_names() -> None:
    tree = whb_view()
    assert tree.leaf_level == SUBC
    assert tree.nodes(SUPC) == (MICRO, ASTRO, UL_IT)
    assert tree.nodes(SUBC) == tuple(f"s{index}" for index in range(1, 8))
    assert tree.children_of(None, None) == tree.roots
    assert tree.children_of(CLUS, "c3") == ("s4", "s5")
    assert tree.children_of(SUBC, "s1") == ()
    assert tree.name(SUPC, ASTRO) == "Astrocyte"
    assert tree.name(SUBC, "s1") == "s1"
    assert tree.parents()[0] == (None, None)
    assert tree.leaves_under(SUPC, UL_IT) == ["s1", "s2", "s3"]
    assert tree.ancestors_of_leaves()[SUPC]["s6"] == MICRO
    assert tree.ancestors_of_leaves()[CLUS]["s5"] == "c3"
    assert tree.node_counts() == {SUPC: 3, CLUS: 5, SUBC: 7}


def test_tree_view_drop_level_moves_grandchildren_up() -> None:
    dropped = whb_view().drop_level(CLUS)
    assert dropped.hierarchy == (SUPC, SUBC)
    assert dropped.children_of(SUPC, UL_IT) == ("s1", "s2", "s3")
    top = whb_view().drop_level(SUPC)
    assert top.roots == ("c1", "c2", "c3", "c4", "c5")
    with pytest.raises(ValueError, match="leaf level"):
        whb_view().drop_level(SUBC)


def test_tree_view_drop_nodes_prunes_empty_ancestors() -> None:
    pruned = whb_view().drop_nodes([(CLUS, "c3")])
    assert ASTRO not in pruned.roots
    assert "s4" not in pruned.nodes(SUBC)
    assert pruned.nodes(CLUS) == ("c1", "c2", "c4", "c5")
    via_mapping = mapping_tree(whb_view(), nodes_to_drop=[f"{SUBC}/s6"])
    assert "c4" not in via_mapping.nodes(CLUS)
    with pytest.raises(ValueError, match="not in the tree"):
        whb_view().drop_nodes([(SUPC, "nope")])


def test_tree_view_json_round_trip() -> None:
    tree = whb_view()
    again = TaxonomyTreeView.from_tree_dict(tree.to_json())
    assert again == tree


# --------------------------------------------------------------------------
# Lookups: filter and auto-collapse


def whb_lookup() -> dict[str, Any]:
    return {
        "None": [GENES[0], GENES[1], GENES[2]],
        f"{SUPC}/{UL_IT}": [GENES[3], GENES[4]],
        f"{SUPC}/{ASTRO}": [],  # one child: trivial
        f"{SUPC}/{MICRO}": [ABSENT_GENE],  # no marker in the query
        f"{CLUS}/c1": [GENES[5]],
        f"{CLUS}/c3": [GENES[6], GENES[7]],
        f"{CLUS}/c4": [],
        f"{CLUS}/c9": [GENES[8]],  # node absent from the tree
        "metadata": {"config": {"n_per_utility": 30}},
        "log": ["noise"],
    }


def test_filter_lookup_to_tree_drops_absent_nodes_and_log() -> None:
    filtered, dropped = filter_lookup_to_tree(whb_lookup(), whb_view())
    assert dropped == [f"{CLUS}/c9"]
    assert "log" not in filtered
    assert filtered["metadata"] == {"config": {"n_per_utility": 30}}
    assert filtered["None"] == [GENES[0], GENES[1], GENES[2]]


def test_filter_lookup_drops_levels_absent_after_drop_level() -> None:
    tree = whb_view().drop_level(CLUS)
    filtered, dropped = filter_lookup_to_tree(whb_lookup(), tree)
    assert all(key.startswith(CLUS) for key in dropped)
    assert set(filtered) == {
        "None",
        f"{SUPC}/{UL_IT}",
        f"{SUPC}/{ASTRO}",
        f"{SUPC}/{MICRO}",
        "metadata",
    }


def test_validate_lookup_collapses_parents_without_query_markers() -> None:
    validation = validate_lookup(whb_lookup(), whb_view(), GENES[:8])
    (collapsed,) = validation.collapsed
    assert isinstance(collapsed, CollapsedParent)
    assert collapsed.key == f"{SUPC}/{MICRO}"
    assert collapsed.hidden_leaves == ("s6", "s7")
    assert collapsed.n_children == 2
    # Descendant keys are removed; the collapsed key stays, empty.
    assert f"{CLUS}/c4" not in validation.lookup
    assert validation.lookup[f"{SUPC}/{MICRO}"] == []
    assert validation.hidden_keys == [f"{CLUS}/c4", f"{CLUS}/c5"]
    # Trivial parents (one child) are neither counted nor collapsed.
    assert f"{SUPC}/{ASTRO}" not in validation.markers_per_parent
    assert validation.markers_per_parent == {
        "None": 3,
        f"{SUPC}/{UL_IT}": 2,
        f"{SUPC}/{MICRO}": 0,
        f"{CLUS}/c1": 1,
        f"{CLUS}/c3": 2,
    }
    summary = validation.summary(weak_parent_markers=2)
    assert summary["collapsed_parents"] == [f"{SUPC}/{MICRO}"]
    assert summary["markers_per_parent_min"] == 0
    assert summary["markers_per_parent_median"] == 2.0
    assert summary["weak_parents"] == [f"{CLUS}/c1", f"{SUPC}/{MICRO}"]
    assert summary["n_hidden_leaves"] == 2
    assert summary["dropped_keys"] == [f"{CLUS}/c9"]


def test_validate_lookup_restricts_markers_to_query_genes() -> None:
    validation = validate_lookup(
        whb_lookup(), whb_view(), [GENES[0], GENES[3], GENES[6]]
    )
    assert validation.lookup["None"] == [GENES[0]]
    assert validation.lookup[f"{CLUS}/c3"] == [GENES[6]]
    collapsed = {parent.key for parent in validation.collapsed}
    # c1 lost its only marker and collapses; MICRO has none in the query.
    assert collapsed == {f"{CLUS}/c1", f"{SUPC}/{MICRO}"}
    assert validation.n_marker_genes == 3


def test_validate_lookup_collapses_a_parent_missing_from_the_lookup() -> None:
    lookup = whb_lookup()
    del lookup[f"{CLUS}/c3"]
    validation = validate_lookup(lookup, whb_view(), GENES)
    assert f"{CLUS}/c3" in {parent.key for parent in validation.collapsed}


def test_validate_lookup_fails_only_when_the_root_has_no_markers() -> None:
    lookup = whb_lookup()
    lookup["None"] = [ABSENT_GENE]
    with pytest.raises(LookupValidationError, match="root"):
        validate_lookup(lookup, whb_view(), GENES)
    # Every other parent without markers is collapsed, not fatal.
    empty_children = {
        key: [] if key not in ("None", "metadata", "log") else value
        for key, value in whb_lookup().items()
    }
    validation = validate_lookup(empty_children, whb_view(), GENES)
    assert {parent.key for parent in validation.collapsed} == {
        f"{SUPC}/{UL_IT}",
        f"{SUPC}/{MICRO}",
        f"{CLUS}/c3",
    }


def test_lookup_sha256_ignores_metadata_and_log() -> None:
    lookup = whb_lookup()
    other = dict(lookup, metadata={"timestamp": "later"}, log=["other"])
    assert lookup_sha256(lookup) == lookup_sha256(other)
    other["None"] = [GENES[0]]
    assert lookup_sha256(lookup) != lookup_sha256(other)


# --------------------------------------------------------------------------
# Profiles and negative genes


def tiny_stats() -> PrecomputedStats:
    tree = TaxonomyTreeView.from_tree_dict(
        {
            "hierarchy": ["A", "B"],
            "A": {"a1": ["b1", "b2"], "a2": ["b3"]},
            "B": {"b1": [], "b2": [], "b3": []},
        }
    )
    # log2(CPM + 1) of detecting cells: b1 gene0 = 1 or 3 (mean 2, var 1).
    return PrecomputedStats(
        path=Path("tiny.h5"),
        leaves=["b1", "b2", "b3"],
        genes=["g0", "g1"],
        n_cells=np.array([4.0, 2.0, 10.0]),
        sum=np.array([[4.0, 0.0], [2.0, 2.0], [10.0, 0.0]]),
        sumsq=np.array([[10.0, 0.0], [2.0, 2.0], [10.0, 0.0]]),
        gt0=np.array([[2.0, 0.0], [2.0, 2.0], [10.0, 0.0]]),
        tree=tree,
    )


def test_lognormal_mean_cpm_matches_the_validated_formula() -> None:
    stats = tiny_stats()
    expected = lognormal_mean_cpm(stats)
    # b1 gene0: g/n = 0.5, m = 2, v = 1 -> 0.5 * (2 ** (2 + ln2 / 2) - 1).
    assert expected[0, 0] == pytest.approx(0.5 * (2 ** (2 + np.log(2) / 2) - 1))
    # b2 gene1: all detecting at 1 -> 1 * (2 ** 1 - 1) = 1.
    assert expected[1, 1] == pytest.approx(1.0)
    assert expected[2, 1] == 0.0


def test_reference_profiles_aggregate_leaves_weighted_by_cells() -> None:
    stats = tiny_stats()
    profiles = reference_profiles_from_stats(
        stats, levels=["A", "B"], gene_symbols={"g0": "G0", "g1": "G1"}
    )
    a1 = profiles[(profiles.level == "A") & (profiles.node == "a1")].set_index(
        "gene_id"
    )
    leaf = lognormal_mean_cpm(stats)
    assert a1.loc["g0", "mean_cpm"] == pytest.approx(
        (4 * leaf[0, 0] + 2 * leaf[1, 0]) / 6
    )
    assert a1.loc["g0", "detection_fraction"] == pytest.approx(4 / 6)
    assert a1.loc["g1", "mean_log2cpm"] == pytest.approx(2 / 6)
    assert a1["n_cells"].tolist() == [6, 6]
    assert a1["gene_symbol"].tolist() == ["G0", "G1"]
    for _key, node in profiles.groupby(["level", "node"]):
        assert node["expected_fraction"].sum() == pytest.approx(1.0)
    kept = reference_profiles_from_stats(stats, levels=["B"], keep_nodes={"B": ["b3"]})
    assert set(kept["node"]) == {"b3"}


def test_negative_genes_need_low_detection_in_every_reference() -> None:
    genes = ["g_neg", "g_one_ref", "g_state", "g_high"]
    classes = ["Astrocytes", "Neurons"]
    first = pd.DataFrame(
        [[0.001, 0.002, 0.0, 0.5], [0.3, 0.3, 0.3, 0.3]], index=classes, columns=genes
    )
    second = pd.DataFrame([[0.005, 0.2, 0.0, 0.4]], index=["Astrocytes"], columns=genes)
    table = negative_gene_table(
        {
            "ref_a": (first, pd.Series([100.0, 200.0], index=classes)),
            "ref_b": (second, pd.Series([50.0], index=["Astrocytes"])),
        },
        genes=genes,
        gene_symbols={"g_state": "GFAP"},
        state_genes=["Gfap"],
        max_fraction=0.01,
    )
    astro = table[table.broad_class == "Astrocytes"].set_index("gene_id")
    assert astro.loc["g_neg", "negative"]
    assert not astro.loc["g_one_ref", "negative"]
    assert (
        astro.loc["g_state", "is_state_gene"] and not astro.loc["g_state", "negative"]
    )
    assert not astro.loc["g_high", "negative"]
    # A class missing from one reference is never negative.
    neurons = table[table.broad_class == "Neurons"]
    assert not neurons["negative"].any()
    assert np.isnan(neurons["detection_ref_b"]).all()


def test_broad_class_detection_and_leaf_classes_from_the_vocab(tmp_path: Path) -> None:
    path = write_precompute(
        tmp_path / "seaad.h5",
        SEAAD_TREE,
        GENES[:3],
        n_cells={"t_it": 10, "t_astro": 20, "t_vlmc": 5, "t_peri": 5, "t_micro": 10},
        detect=lambda leaf, gene: 0.0 if (leaf == "t_astro" and gene == 0) else 0.6,
    )
    stats = read_precomputed_stats(path, GENES[:3])
    classes = reference.leaf_broad_classes(stats, reference.SEAAD_TAXONOMY_ID)
    assert classes == {
        "t_it": "Neurons",
        "t_astro": "Astrocytes",
        "t_vlmc": "Fibroblasts",
        "t_peri": "Vascular cells",
        "t_micro": "Microglia",
    }
    detection, n_cells = broad_class_detection(stats, classes)
    assert detection.loc["Astrocytes", GENES[0]] == 0.0
    assert detection.loc["Neurons", GENES[0]] == pytest.approx(0.6)
    assert n_cells["Fibroblasts"] == 5


def test_node_vocab_table_maps_every_level() -> None:
    table = node_vocab_table(whb_view(), reference.WHB_TAXONOMY_ID).set_index("node")
    assert table.loc[UL_IT, "broad_class"] == "Neurons"
    assert table.loc[UL_IT, "nt"] == "Excitatory"
    assert table.loc["c3", "broad_class"] == "Astrocytes"
    assert table.loc["s6", "broad_class"] == "Microglia"
    assert table.loc["s6", "key_label"] == MICRO
    assert bool(table.loc["s1", "region_plausible_frontal_cortex"])
    seaad = node_vocab_table(
        TaxonomyTreeView.from_tree_dict(SEAAD_TREE), reference.SEAAD_TAXONOMY_ID
    ).set_index("node")
    # The subclass is mixed; its supertypes follow the overrides.
    assert seaad.loc["CS20260630_SCLA_028", "broad_class"] == "Mixed/Unknown"
    assert seaad.loc["t_vlmc", "broad_class"] == "Fibroblasts"
    assert seaad.loc["CS20260630_CLAS_002", "broad_class"] == "Neurons"
    assert seaad.loc["CS20260630_CLAS_003", "broad_class"] == "Mixed/Unknown"


def test_panel_stub_has_no_cells_and_the_panel_genes(tmp_path: Path) -> None:
    import anndata as ad

    path = write_panel_stub_h5ad(
        GENES[:3], tmp_path / "stub.h5ad", gene_symbols=["A", "B", "C"]
    )
    stub = ad.read_h5ad(path)
    assert stub.n_obs == 0
    assert list(stub.var_names) == GENES[:3]
    assert list(stub.var["gene_symbol"]) == ["A", "B", "C"]
    with pytest.raises(ValueError, match="distinct"):
        write_panel_stub_h5ad([GENES[0], GENES[0]], tmp_path / "dup.h5ad")


# --------------------------------------------------------------------------
# Pinned downloads and source preparation


def pinned_for(tmp_path: Path, content: bytes) -> PinnedFile:
    return PinnedFile(
        key="precomputed_stats",
        url="https://example.invalid/stats.h5",
        relative_path="mapmycells/test/stats.h5",
        size=len(content),
        md5=hashlib.md5(content).hexdigest(),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_ensure_pinned_file_uses_a_verified_seed_copy(tmp_path: Path) -> None:
    content = b"stats" * 100
    pinned = pinned_for(tmp_path, content)
    seed = tmp_path / "archive" / "stats.h5"
    seed.parent.mkdir()
    seed.write_bytes(content)
    target = ensure_pinned_file(pinned, tmp_path / "cache", seed=seed)
    assert target.read_bytes() == content
    assert not target.is_symlink()
    assert seed.is_file()
    # Cached now: no seed needed.
    assert ensure_pinned_file(pinned, tmp_path / "cache") == target


def test_ensure_pinned_file_never_deletes_a_mismatching_cache(tmp_path: Path) -> None:
    pinned = pinned_for(tmp_path, b"expected")
    target = tmp_path / "cache" / pinned.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"tampered")
    with pytest.raises(PinnedFileError, match="left in place"):
        ensure_pinned_file(pinned, tmp_path / "cache", auto_download=True)
    assert target.read_bytes() == b"tampered"


def test_ensure_pinned_file_ignores_a_bad_seed_and_respects_no_download(
    tmp_path: Path,
) -> None:
    pinned = pinned_for(tmp_path, b"expected")
    seed = tmp_path / "bad_seed.h5"
    seed.write_bytes(b"other bytes")
    with pytest.raises(PinnedFileError, match="downloads are off"):
        ensure_pinned_file(pinned, tmp_path / "cache", seed=seed)


def test_ensure_pinned_file_downloads_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"downloaded"
    pinned = pinned_for(tmp_path, content)
    calls: list[str] = []

    def fake_ensure_url_file(url: str, output_path: Path, **_: Any) -> Path:
        calls.append(url)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)
        return output_path

    monkeypatch.setattr(
        "merxen.analysis.mapmycells._ensure_url_file", fake_ensure_url_file
    )
    target = ensure_pinned_file(pinned, tmp_path / "cache", auto_download=True)
    assert target.read_bytes() == content and calls == [pinned.url]
    bad = PinnedFile(
        **{**pinned.__dict__, "relative_path": "other.h5", "sha256": "0" * 64}
    )
    with pytest.raises(PinnedFileError, match="kept at"):
        ensure_pinned_file(bad, tmp_path / "cache", auto_download=True)
    assert (tmp_path / "cache" / "other.h5.download").is_file()


def test_seaad_pins_match_the_allen_manifest_md5() -> None:
    stats = reference.SEAAD_STATS_PIN
    assert stats.md5.startswith("9b1d5f50a412")
    assert stats.size == 443_095_208
    assert len(stats.sha256) == 64
    assert {pinned.key for pinned in reference.SEAAD_MULTIREGION_FILES} == {
        "precomputed_stats",
        "cluster",
        "cluster_annotation_term",
        "cluster_annotation_term_set",
        "cluster_to_cluster_annotation_membership",
    }


def whb_spec(**sources: Path) -> AnnotationReferenceSpec:
    return AnnotationReferenceSpec(
        reference_id="whb_frontal_supc_clus",
        species="human",
        role="primary",
        hierarchy=[SUPC, CLUS],
        sources=sources,
    )


def test_prepare_reference_spec_expands_a_region_reference_directory(
    tmp_path: Path,
) -> None:
    region = tmp_path / "region_frontal"
    (region / "precompute").mkdir(parents=True)
    (region / "precompute" / "precomputed_stats.h5").write_bytes(b"x")
    (region / "region_reference_manifest.json").write_text("{}")
    seaad = tmp_path / "seaad.h5"
    seaad.write_bytes(b"s")
    prepared = prepare_reference_spec(
        whb_spec(region_precompute=region, seaad_precomputed_stats=seaad)
    )
    assert prepared.sources["region_precompute"] == (
        region / "precompute" / "precomputed_stats.h5"
    )
    assert (
        prepared.sources["region_manifest"] == region / "region_reference_manifest.json"
    )


def test_prepare_reference_spec_requires_the_seaad_negatives_source(
    tmp_path: Path,
) -> None:
    precompute = tmp_path / "stats.h5"
    precompute.write_bytes(b"x")
    with pytest.raises(ReferenceBuildError, match="seaad_precomputed_stats"):
        prepare_reference_spec(whb_spec(region_precompute=precompute))
    # With a download cache, the pinned file comes from the cache or a seed.
    cache = tmp_path / "cache"
    target = cache / reference.SEAAD_STATS_PIN.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the pinned file")
    with pytest.raises(PinnedFileError):
        prepare_reference_spec(
            whb_spec(region_precompute=precompute), SourceOptions(download_dir=cache)
        )


def test_prepare_reference_spec_expands_whb_rebuild_directories(tmp_path: Path) -> None:
    metadata = tmp_path / "abc_whb" / "metadata"
    for relative in (
        "WHB-10Xv3/20241115/cell_metadata.csv",
        "WHB-10Xv3/20241115/region_of_interest_structure_map.csv",
        "WHB-taxonomy/20240330/cluster_annotation_term.csv",
        "WHB-taxonomy/20240330/cluster_to_cluster_annotation_membership.csv",
    ):
        (metadata / relative).parent.mkdir(parents=True, exist_ok=True)
        (metadata / relative).write_text("x\n")
    h5ads = tmp_path / "abc_whb" / "expression_matrices" / "WHB-10Xv3" / "20240330"
    h5ads.mkdir(parents=True)
    for name in ("WHB-10Xv3-Neurons-raw.h5ad", "WHB-10Xv3-Nonneurons-raw.h5ad"):
        (h5ads / name).write_bytes(b"h5")
    seaad = tmp_path / "seaad.h5"
    seaad.write_bytes(b"s")
    prepared = prepare_reference_spec(
        whb_spec(
            whb_metadata_dir=metadata,
            whb_h5ad_dir=h5ads,
            seaad_precomputed_stats=seaad,
        )
    )
    assert "whb_metadata_dir" not in prepared.sources
    assert (
        prepared.sources["whb_roi_map"].name == "region_of_interest_structure_map.csv"
    )
    assert prepared.sources["whb_neurons_h5ad"].name == "WHB-10Xv3-Neurons-raw.h5ad"


def test_prepare_reference_spec_reports_missing_wmb_sources(tmp_path: Path) -> None:
    spec = AnnotationReferenceSpec(
        reference_id="wmb_panel",
        species="mouse",
        role="primary",
        drop_level="CCN20230722_SUPT",
        sources={"wmb_h5ad_dir": tmp_path},
    )
    with pytest.raises(ReferenceBuildError, match="wmb_mapping_stats"):
        prepare_reference_spec(spec)


def test_registered_builders_carry_prepare_spec_and_hashed_params() -> None:
    spec = whb_spec()
    builder = resolve_builder(spec, None)
    assert builder.name == "whb_frontal_supc_clus"
    assert builder.prepare_spec is prepare_reference_spec
    assert builder.params["roi_labels"] == list(reference.WHB_FRONTAL_ROI_LABELS)
    assert "state_genes_human.csv" in builder.params["state_genes"]
    region = AnnotationReferenceSpec(
        reference_id="wmb_region_share", species="mouse", role="region_share"
    )
    assert builder_for(region).uses_panel is False


# --------------------------------------------------------------------------
# Builders with the cell_type_mapper runners monkeypatched


def truncate_precompute(
    input_path: Path, output_path: Path, hierarchy: list[str]
) -> None:
    """Aggregate a synthetic precompute's leaves to a shallower hierarchy."""
    with h5py.File(input_path, "r") as handle:
        tree = json.loads(handle["taxonomy_tree"][()].decode())
        genes = json.loads(handle["col_names"][()].decode())
        rows = json.loads(handle["cluster_to_row"][()].decode())
        arrays = {name: handle[name][()] for name in ("n_cells", "sum", "sumsq", "gt0")}
    view = TaxonomyTreeView.from_tree_dict(tree)
    new_leaf = hierarchy[-1]
    new_leaves = sorted(view.nodes(new_leaf))
    ancestors = view.ancestors_of_leaves()[new_leaf]
    out: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        out[name] = np.stack(
            [
                values[
                    [rows[leaf] for leaf, node in ancestors.items() if node == target]
                ].sum(axis=0)
                for target in new_leaves
            ]
        )
    new_tree: dict[str, Any] = {"hierarchy": hierarchy}
    for level in hierarchy[:-1]:
        new_tree[level] = {
            node: list(view.children_of(level, node)) for node in view.nodes(level)
        }
    new_tree[new_leaf] = {leaf: [] for leaf in new_leaves}
    new_tree["name_mapper"] = {
        level: names
        for level, names in tree.get("name_mapper", {}).items()
        if level in hierarchy
    }
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("taxonomy_tree", data=json.dumps(new_tree).encode())
        handle.create_dataset("col_names", data=json.dumps(genes).encode())
        handle.create_dataset(
            "cluster_to_row",
            data=json.dumps(
                {leaf: row for row, leaf in enumerate(new_leaves)}
            ).encode(),
        )
        for name, values in out.items():
            handle.create_dataset(name, data=values)


class FakeCtm:
    """Monkeypatched ctm runners that record their configs."""

    def __init__(self, lookup: Callable[[list[str]], dict[str, Any]]) -> None:
        self.lookup = lookup
        self.calls: dict[str, list[dict[str, Any]]] = {
            "truncate": [],
            "precompute": [],
            "reference": [],
            "query": [],
            "drop_nodes": [],
        }
        self.stub_genes: list[str] = []
        self.precompute: Callable[[dict[str, Any]], None] | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeCtm:
        monkeypatch.setattr(reference, "_run_ctm_truncate", self.truncate)
        monkeypatch.setattr(reference, "_run_ctm_precompute_abc", self.precompute_abc)
        monkeypatch.setattr(
            reference, "_run_ctm_reference_markers", self.reference_markers
        )
        monkeypatch.setattr(reference, "_run_ctm_query_markers", self.query_markers)
        monkeypatch.setattr(reference, "_run_ctm_drop_nodes", self.drop_nodes)
        return self

    def truncate(self, config: dict[str, Any], **_: Any) -> None:
        self.calls["truncate"].append(config)
        truncate_precompute(
            Path(config["input_path"]),
            Path(config["output_path"]),
            config["new_hierarchy"],
        )

    def precompute_abc(self, config: dict[str, Any], **_: Any) -> None:
        self.calls["precompute"].append(config)
        assert self.precompute is not None
        self.precompute(config)

    def reference_markers(self, config: dict[str, Any], **_: Any) -> None:
        import anndata as ad

        self.calls["reference"].append(config)
        self.stub_genes = list(ad.read_h5ad(config["query_path"]).var_names)
        (Path(config["output_dir"]) / "reference_markers.h5").write_bytes(b"markers")

    def query_markers(self, config: dict[str, Any], **_: Any) -> None:
        self.calls["query"].append(config)
        Path(config["output_path"]).write_text(json.dumps(self.lookup(self.stub_genes)))

    def drop_nodes(
        self, src: Path, dst: Path, nodes: list[tuple[str, str]], **_: Any
    ) -> None:
        self.calls["drop_nodes"].append({"src": src, "dst": dst, "nodes": nodes})
        truncate_precompute(
            src,
            dst,
            list(
                json.loads(h5py.File(src, "r")["taxonomy_tree"][()].decode())[
                    "hierarchy"
                ]
            ),
        )


@pytest.fixture
def small_resources() -> Any:
    previous = reference._PREP_RESOURCES
    reference.set_prep_resources(n_processors=2, max_gb=3)
    yield
    reference._PREP_RESOURCES = previous


WHB_N_CELLS = {"s1": 10, "s2": 12, "s3": 8, "s4": 20, "s5": 5, "s6": 7, "s7": 9}


def whb_detect(leaf: str, gene: int) -> float:
    # Gene 9 is never detected in astrocytes (s4, s5): a negative gene.
    if gene == 9 and leaf in ("s4", "s5"):
        return 0.0
    return 0.2 + 0.05 * gene


def seaad_detect(leaf: str, gene: int) -> float:
    if gene == 9 and leaf == "t_astro":
        return 0.0
    return 0.3


def write_whb_sources(
    tmp_path: Path, *, manifest_cells: int | None = None
) -> dict[str, Path]:
    region = tmp_path / "ssd1" / "region_frontal"
    stats = write_precompute(
        region / "precompute" / "precomputed_stats.h5",
        WHB_TREE,
        GENES,
        n_cells=WHB_N_CELLS,
        detect=whb_detect,
    )
    n_cells = sum(WHB_N_CELLS.values())
    (region / "region_reference_manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "region_labels": list(reference.WHB_FRONTAL_ROI_LABELS),
                    "region_min_cells_per_leaf": 10,
                },
                "filtering_summary": {
                    "n_cells_after_min_leaf_filter": manifest_cells or n_cells,
                    "n_leaf_aliases_after_min_leaf_filter": 7,
                },
            }
        )
    )
    seaad = write_precompute(
        tmp_path / "seaad" / "seaad_multiregion.h5",
        SEAAD_TREE,
        GENES,
        n_cells={"t_it": 30, "t_astro": 25, "t_vlmc": 5, "t_peri": 5, "t_micro": 10},
        detect=seaad_detect,
    )
    return {"region_dir": region, "stats": stats, "seaad": seaad}


def whb_truncated_lookup(stub: list[str]) -> dict[str, Any]:
    return {
        "None": stub[:4],
        f"{SUPC}/{UL_IT}": stub[4:6],
        f"{SUPC}/{ASTRO}": [],
        f"{SUPC}/{MICRO}": [ABSENT_GENE],
        f"{CLUS}/c1": [stub[0]],  # c1 is a leaf of the truncated tree
        "metadata": {"config": {"n_per_utility": 30}},
        "log": ["x"],
    }


def build_whb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    panel: AnnotationPanel,
    sources: Mapping[str, Path],
) -> tuple[ReferenceStore, Any, FakeCtm]:
    fake = FakeCtm(whb_truncated_lookup).install(monkeypatch)
    spec = prepare_reference_spec(
        whb_spec(
            region_precompute=sources["region_dir"],
            seaad_precomputed_stats=sources["seaad"],
        )
    )
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    (tmp_path / "scratch").mkdir(exist_ok=True)
    config = AnnotationConfig(species="human")
    bundle = store.get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    return store, bundle, fake


def test_whb_frontal_builder_copies_truncates_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_whb_sources(tmp_path)
    panel = make_panel([*GENES, ABSENT_GENE])
    store, bundle, fake = build_whb(tmp_path, monkeypatch, panel, sources)
    bundle_dir = Path(bundle.path)
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    output = manifest["builder_output"]
    # The region precompute is copied (not linked) with its checksum.
    copy = bundle_dir / reference.SOURCE_PRECOMPUTE_FILE
    assert not copy.is_symlink()
    assert reference.file_sha256(copy) == reference.file_sha256(sources["stats"])
    assert output["source_precompute"]["mode"] == "copied"
    verification = output["source_precompute"]["verification"]
    assert verification["manifest"]["agrees"] is True
    assert (
        verification["n_leaves"] == 7
        and verification["matches_validated_frontal"] is False
    )
    # Truncated to supercluster -> cluster.
    (truncate,) = fake.calls["truncate"]
    assert truncate["new_hierarchy"] == [SUPC, CLUS]
    assert output["levels"] == [SUPC, CLUS] and output["n_leaves"] == 5
    # Markers on the panel stub (genes present in the reference), resources explicit.
    assert fake.stub_genes == GENES
    (reference_config,) = fake.calls["reference"]
    assert reference_config["n_processors"] == 2 and reference_config["max_gb"] == 3
    assert "drop_level" not in reference_config
    (query_config,) = fake.calls["query"]
    assert query_config["n_per_utility"] == 30
    assert query_config["search_for_stats_file"] is False
    # Lookup filtered to the truncated tree; MICRO has no marker and collapses.
    filtered = json.loads(
        (bundle_dir / reference.QUERY_MARKERS_FILTERED_FILE).read_text()
    )
    assert f"{CLUS}/c1" not in filtered and "log" not in filtered
    assert filtered[f"{SUPC}/{MICRO}"] == []
    assert output["collapsed_parents"] == [f"{SUPC}/{MICRO}"]
    assert output["markers"]["dropped_keys"] == [f"{CLUS}/c1"]
    assert output["markers"]["collapsed"][0]["hidden_leaves"] == ["c4", "c5"]
    assert output["markers"]["markers_per_parent_min"] == 0
    assert output["markers"]["root_markers"] == 4
    coverage = output["panel_coverage"]
    assert coverage["n_panel_genes"] == 11 and coverage["n_query_genes_used"] == 10
    assert coverage["absent_from_reference"][0]["gene_id"] == ABSENT_GENE
    # Reference markers stay in scratch; only their checksum is recorded.
    assert not (bundle_dir / reference.REFERENCE_MARKERS_DIR).exists()
    assert output["markers"]["reference_markers"][0]["kept"] is False
    assert not list((tmp_path / "scratch").iterdir())
    # Profiles on the mapping levels, from the subcluster leaves.
    profiles = pd.read_parquet(bundle_dir / reference.PROFILES_FILE)
    assert set(profiles["level"]) == {SUPC, CLUS}
    assert len(profiles) == (3 + 5) * 10
    # Negative genes need both WHB frontal and SEA-AD.
    negatives = pd.read_parquet(bundle_dir / reference.NEGATIVE_GENES_FILE)
    astro = negatives[negatives.broad_class == "Astrocytes"].set_index("gene_id")
    assert astro.loc[GENES[9], "negative"]
    assert not astro.loc[GENES[0], "negative"]
    assert output["negative_genes"]["references"] == ["seaad_mr", "whb_frontal"]
    vocab = pd.read_csv(bundle_dir / reference.VOCAB_SNAPSHOT_FILE)
    assert set(vocab["level"]) == {SUPC, CLUS}
    depth = json.loads((bundle_dir / reference.DEPTH_GRID_FILE).read_text())
    assert depth["depth_grid"] == [10, 15, 30, 60, 120, 250]
    tree = json.loads((bundle_dir / reference.MAPPING_TREE_FILE).read_text())
    assert tree["hierarchy"] == [SUPC, CLUS]
    assert set(output["timings_s"]) >= {"truncate_taxonomy", "reference_markers"}
    # A second call reuses the bundle without running ctm again.
    config = AnnotationConfig(species="human")
    again = store.get_or_build(
        prepare_reference_spec(
            whb_spec(
                region_precompute=sources["region_dir"],
                seaad_precomputed_stats=sources["seaad"],
            )
        ),
        panel,
        builder=builder_for(whb_spec(), config),
        config=config,
    )
    assert again.reused and again.build_hash == bundle.build_hash
    assert len(fake.calls["reference"]) == 1


def test_whb_set_c_bundle_is_the_same_builder_on_the_set_c_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_whb_sources(tmp_path)
    set_a = make_panel(GENES)
    set_c = make_panel(GENES[:8], kind="setc").model_copy(
        update={"parent_panel_hash": set_a.panel_hash, "excluded_ids": GENES[8:]}
    )
    _store, bundle_a, _ = build_whb(tmp_path, monkeypatch, set_a, sources)
    _store, bundle_c, fake = build_whb(tmp_path, monkeypatch, set_c, sources)
    assert bundle_a.build_hash != bundle_c.build_hash
    assert Path(bundle_a.path).parent == Path(bundle_c.path).parent
    output = json.loads((Path(bundle_c.path) / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    assert output["panel_kind"] == "setc"
    assert output["parent_panel_hash"] == set_a.panel_hash
    assert fake.stub_genes == GENES[:8]


def test_whb_builder_refuses_a_precompute_that_disagrees_with_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_whb_sources(tmp_path, manifest_cells=999)
    with pytest.raises(ReferenceBuildError, match="disagrees"):
        build_whb(tmp_path, monkeypatch, make_panel(GENES), sources)
    store_root = tmp_path / "store"
    assert not list((store_root / "whb_frontal_supc_clus").glob("[0-9a-f]*"))
    assert list(store_root.glob(".failed-*"))


def write_seaad_metadata(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for key in reference.SEAAD_TAXONOMY_KEYS:
        (directory / f"{key}.csv").write_text(f"{key}\n")
    return directory


def test_seaad_builder_copies_the_precompute_and_taxonomy_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    stats = write_precompute(
        tmp_path / "seaad.h5",
        SEAAD_TREE,
        GENES,
        n_cells={"t_it": 30, "t_astro": 25, "t_vlmc": 5, "t_peri": 5, "t_micro": 10},
    )

    def lookup(stub: list[str]) -> dict[str, Any]:
        return {
            "None": stub[:3],
            f"{L0}/CS20260630_CLAS_003": stub[3:5],
            f"{L1}/CS20260630_SCLA_028": stub[5:6],
            f"{L1}/CS20260630_SCLA_010": [],
        }

    fake = FakeCtm(lookup).install(monkeypatch)
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id="seaad_mr_panel",
            species="human",
            role="secondary",
            sources={
                "seaad_precomputed_stats": stats,
                "seaad_metadata_dir": write_seaad_metadata(tmp_path / "meta"),
            },
        )
    )
    assert set(reference.SEAAD_TAXONOMY_SOURCES.values()) <= set(spec.sources)
    store = ReferenceStore(tmp_path / "store")
    panel = make_panel(GENES[:6])
    bundle = store.get_or_build(spec, panel, builder=builder_for(spec))
    bundle_dir = Path(bundle.path)
    output = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    precompute = output["mapping_precompute"]
    assert precompute["matches_pinned_release"] is False
    assert precompute["pinned_md5"] == reference.SEAAD_STATS_PIN.md5
    assert (bundle_dir / "taxonomy" / "cluster.csv").read_text() == "cluster\n"
    assert output["levels"] == [L0, L1, L2]
    assert output["collapsed_parents"] == []
    assert not (bundle_dir / reference.NEGATIVE_GENES_FILE).exists()
    profiles = pd.read_parquet(bundle_dir / reference.PROFILES_FILE)
    assert set(profiles["level"]) == {L0, L1, L2}
    assert fake.stub_genes == GENES[:6]


# WMB (CCN20230722) synthetic taxonomy: the mapping means cover clusters 1-7,
# the 10Xv3 marker cells only aliases 1-4.
CLAS, SUBC_W, SUPT, CLUS_W = (
    "CCN20230722_CLAS",
    "CCN20230722_SUBC",
    "CCN20230722_SUPT",
    "CCN20230722_CLUS",
)
WMB_TREE: dict[str, Any] = {
    "hierarchy": [CLAS, SUBC_W, SUPT, CLUS_W],
    CLAS: {
        "CS20230722_CLAS_01": ["CS20230722_SUBC_001", "CS20230722_SUBC_002"],
        "CS20230722_CLAS_30": ["CS20230722_SUBC_319", "CS20230722_SUBC_320"],
    },
    SUBC_W: {
        "CS20230722_SUBC_001": ["CS20230722_SUPT_0001"],
        "CS20230722_SUBC_002": ["CS20230722_SUPT_0002"],
        "CS20230722_SUBC_319": ["CS20230722_SUPT_1160"],
        "CS20230722_SUBC_320": ["CS20230722_SUPT_1161"],
    },
    SUPT: {
        "CS20230722_SUPT_0001": ["CS20230722_CLUS_0001", "CS20230722_CLUS_0002"],
        "CS20230722_SUPT_0002": ["CS20230722_CLUS_0003"],
        "CS20230722_SUPT_1160": ["CS20230722_CLUS_0004", "CS20230722_CLUS_0005"],
        "CS20230722_SUPT_1161": ["CS20230722_CLUS_0006", "CS20230722_CLUS_0007"],
    },
    CLUS_W: {f"CS20230722_CLUS_{index:04d}": [] for index in range(1, 8)},
    "name_mapper": {
        SUBC_W: {"CS20230722_SUBC_320": {"name": "320 Multiome-only Gaba"}},
    },
}
ALIAS_TO_CLUSTER = {alias: f"CS20230722_CLUS_{alias:04d}" for alias in range(1, 5)}
MOUSE_GENES = [f"ENSMUSG{index:011d}" for index in range(1, 9)]


def subtree(tree: Mapping[str, Any], leaves: set[str]) -> dict[str, Any]:
    view = TaxonomyTreeView.from_tree_dict(tree)
    drop = [
        (view.leaf_level, leaf)
        for leaf in view.nodes(view.leaf_level)
        if leaf not in leaves
    ]
    pruned = view.drop_nodes(drop).to_json()
    pruned["name_mapper"] = tree.get("name_mapper", {})
    return pruned


def write_wmb_sources(tmp_path: Path) -> dict[str, Path]:
    import anndata as ad
    import scipy.sparse as sp

    h5ad_dir = tmp_path / "WMB-10Xv3"
    h5ad_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for matrix in ("WMB-10Xv3-AAA", "WMB-10Xv3-BBB"):
        labels = [f"{matrix}-cell{index}" for index in range(12)]
        counts = sp.csr_matrix(
            rng.poisson(2.0, size=(12, len(MOUSE_GENES))).astype(np.float32)
        )
        ad.AnnData(
            X=counts,
            obs=pd.DataFrame(index=labels),
            var=pd.DataFrame(index=MOUSE_GENES),
        ).write_h5ad(h5ad_dir / f"{matrix}-raw.h5ad")
        for index, label in enumerate(labels):
            rows.append(
                {
                    "cell_label": label,
                    "feature_matrix_label": matrix,
                    "library_method": "10Xv3",
                    "cluster_alias": 1 + index % 4,
                }
            )
    rows += [
        {
            "cell_label": "v2cell",
            "feature_matrix_label": "WMB-10Xv2-X",
            "library_method": "10Xv2",
            "cluster_alias": 1,
        },
        {
            "cell_label": "nocluster",
            "feature_matrix_label": "WMB-10Xv3-AAA",
            "library_method": "10Xv3",
            "cluster_alias": None,
        },
        {
            "cell_label": "elsewhere",
            "feature_matrix_label": "WMB-10Xv3-CCC",
            "library_method": "10Xv3",
            "cluster_alias": 2,
        },
    ]
    metadata = tmp_path / "wmb_metadata"
    metadata.mkdir()
    pd.DataFrame(rows).to_csv(metadata / "cell_metadata.csv", index=False)
    (metadata / "cluster_annotation_term.csv").write_text("label\n")
    (metadata / "cluster_to_cluster_annotation_membership.csv").write_text("label\n")
    truth = tmp_path / "truth.csv"
    pd.DataFrame(
        {"class": ["x", "x"]},
        index=pd.Index(
            ["WMB-10Xv3-AAA-cell0", "WMB-10Xv3-BBB-cell1"], name="cell_label"
        ),
    ).to_csv(truth)
    mapping = write_precompute(
        tmp_path / "allen" / "precomputed_stats_ABC_revision_230821.h5",
        WMB_TREE,
        MOUSE_GENES,
        n_cells={leaf: 1 for leaf in leaves_of(WMB_TREE)},
        with_detection=False,
    )
    return {
        "h5ad_dir": h5ad_dir,
        "metadata": metadata,
        "truth": truth,
        "mapping": mapping,
    }


def wmb_marker_precompute(config: dict[str, Any]) -> None:
    import anndata as ad

    cells = pd.read_csv(config["cell_metadata_path"])
    genes = list(ad.read_h5ad(config["h5ad_path_list"][0], backed="r").var_names)
    counts = cells["cluster_alias"].map(ALIAS_TO_CLUSTER).value_counts().to_dict()
    tree = subtree(WMB_TREE, set(counts))
    astro_leaves = {"CS20230722_CLUS_0004"}
    write_precompute(
        Path(config["output_path"]),
        tree,
        genes,
        n_cells=counts,
        detect=lambda leaf, gene: 0.0 if (leaf in astro_leaves and gene == 0) else 0.5,
    )


def wmb_lookup(stub: list[str]) -> dict[str, Any]:
    return {
        "None": stub[:3],
        f"{CLAS}/CS20230722_CLAS_01": stub[3:5],
        f"{CLAS}/CS20230722_CLAS_30": stub[1:2],
        f"{SUBC_W}/CS20230722_SUBC_001": stub[4:6],
        f"{SUBC_W}/CS20230722_SUBC_319": [],
        f"{SUPT}/CS20230722_SUPT_0001": stub[:2],
    }


def wmb_spec(sources: Mapping[str, Path], **updates: Any) -> AnnotationReferenceSpec:
    return AnnotationReferenceSpec(
        reference_id="wmb_panel",
        species="mouse",
        role="primary",
        drop_level="CCN20230722_SUPT",
        max_cells_per_cluster=3,
        sources={
            "wmb_h5ad_dir": sources["h5ad_dir"],
            "wmb_metadata_dir": sources["metadata"],
            "wmb_mapping_stats": sources["mapping"],
            "wmb_selfmap_test_cells": sources["truth"],
        },
        **updates,
    )


def test_wmb_builder_samples_marker_cells_and_records_uncovered_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_wmb_sources(tmp_path)
    fake = FakeCtm(wmb_lookup).install(monkeypatch)
    fake.precompute = wmb_marker_precompute
    spec = prepare_reference_spec(wmb_spec(sources))
    assert spec.sources["wmb_cell_metadata"].name == "cell_metadata.csv"
    panel = make_panel([*MOUSE_GENES[:6], "ENSMUSG99999999999"], species="mouse")
    store = ReferenceStore(tmp_path / "store")
    bundle = store.get_or_build(spec, panel, builder=builder_for(spec))
    bundle_dir = Path(bundle.path)
    output = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    # At most 3 cells per cluster, 10Xv3 in local matrices, test cells excluded.
    training = pd.read_csv(bundle_dir / reference.MARKER_TRAINING_CELLS_FILE)
    assert training.groupby("cluster_alias").size().max() <= 3
    assert not set(training["cell_label"]) & {
        "WMB-10Xv3-AAA-cell0",
        "WMB-10Xv3-BBB-cell1",
        "v2cell",
        "nocluster",
        "elsewhere",
    }
    sampling = output["marker_precompute"]["sampling"]
    assert sampling["n_excluded_test_cells"] == 2
    assert sampling["n_sampled_cells"] == 12
    # Marker precompute restricted to the panel genes present in WMB.
    (precompute,) = fake.calls["precompute"]
    assert precompute["hierarchy"] == [CLAS, SUBC_W, SUPT, CLUS_W]
    assert precompute["n_processors"] == 2
    universe = output["marker_precompute"]["gene_universe"]
    assert universe["n_genes"] == 6 and universe["panel_genes_absent"] == [
        "ENSMUSG99999999999"
    ]
    # Both marker steps drop the supertype level.
    assert fake.calls["reference"][0]["drop_level"] == SUPT
    assert fake.calls["query"][0]["drop_level"] == SUPT
    # The Allen means are copied as the mapping precompute.
    mapping = output["mapping_precompute"]
    assert mapping["sha256"] == reference.file_sha256(sources["mapping"])
    assert mapping["matches_validated_release"] is False
    assert not (bundle_dir / reference.MAPPING_PRECOMPUTE_FILE).is_symlink()
    # Mapping tree without SUPT; uncovered clusters and subclasses recorded.
    assert output["levels"] == [CLAS, SUBC_W, CLUS_W]
    assert [entry["node"] for entry in output["uncovered_clusters"]] == [
        "CS20230722_CLUS_0005",
        "CS20230722_CLUS_0006",
        "CS20230722_CLUS_0007",
    ]
    assert output["uncovered_subclasses"] == [
        {"node": "CS20230722_SUBC_320", "name": "320 Multiome-only Gaba"}
    ]
    assert output["known_limitations"] == [reference.WMB_KNOWN_LIMITATION]
    filtered = json.loads(
        (bundle_dir / reference.QUERY_MARKERS_FILTERED_FILE).read_text()
    )
    assert f"{SUPT}/CS20230722_SUPT_0001" not in filtered
    assert output["markers"]["dropped_keys"] == [f"{SUPT}/CS20230722_SUPT_0001"]
    # Subclasses whose clusters lack markers in the mapping tree collapse.
    assert output["collapsed_parents"] == [
        f"{SUBC_W}/CS20230722_SUBC_319",
        f"{SUBC_W}/CS20230722_SUBC_320",
    ]
    profiles = pd.read_parquet(bundle_dir / reference.PROFILES_FILE)
    assert set(profiles["level"]) == {CLAS, SUBC_W, CLUS_W}
    assert "CS20230722_CLUS_0005" not in set(profiles["node"])
    negatives = pd.read_parquet(bundle_dir / reference.NEGATIVE_GENES_FILE)
    astro = negatives[negatives.broad_class == "Astrocytes/Ependymal"].set_index(
        "gene_id"
    )
    assert astro.loc[MOUSE_GENES[0], "negative"]
    assert output["negative_genes"]["references"] == ["wmb_panel"]
    depth = json.loads((bundle_dir / reference.DEPTH_GRID_FILE).read_text())
    assert depth["depth_grid"] == [10, 20, 50, 100, 250, 500, 1000, 2000]


def test_wmb_sampling_reproduces_the_validated_recipe(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    meta = pd.DataFrame(
        {
            "cell_label": [f"c{index}" for index in range(400)],
            "feature_matrix_label": rng.choice(["WMB-10Xv3-A", "WMB-10Xv3-B"], 400),
            "library_method": rng.choice(["10Xv3", "10Xv2"], 400, p=[0.8, 0.2]),
            "cluster_alias": rng.integers(1, 30, 400),
        }
    )
    path = tmp_path / "cell_metadata.csv"
    meta.to_csv(path, index=False)
    exclude = {"c1", "c2", "c3"}
    sampled, summary = sample_wmb_training_cells(
        path,
        ["WMB-10Xv3-A", "WMB-10Xv3-B"],
        max_cells_per_cluster=5,
        exclude_cells=exclude,
    )
    # research/selfmap/build_marker_ref.py, verbatim.
    recipe = pd.read_csv(
        path,
        usecols=[
            "cell_label",
            "feature_matrix_label",
            "library_method",
            "cluster_alias",
        ],
        dtype={"cluster_alias": "Int64"},
    )
    recipe = recipe[recipe.library_method == "10Xv3"].dropna(subset=["cluster_alias"])
    recipe = recipe[recipe.feature_matrix_label.isin({"WMB-10Xv3-A", "WMB-10Xv3-B"})]
    recipe = recipe[~recipe.cell_label.isin(exclude)]
    with pytest.warns(FutureWarning):
        expected = (
            recipe.groupby("cluster_alias", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), 5), random_state=1))
            .reset_index(drop=True)
        )
    assert sampled["cell_label"].tolist() == expected["cell_label"].tolist()
    assert summary["n_sampled_cells"] == len(expected)


def test_wmb_builder_refuses_a_panel_without_wmb_genes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_wmb_sources(tmp_path)
    fake = FakeCtm(wmb_lookup).install(monkeypatch)
    fake.precompute = wmb_marker_precompute
    spec = prepare_reference_spec(wmb_spec(sources))
    panel = make_panel(["ENSMUSG99999999998", "ENSMUSG99999999999"], species="mouse")
    with pytest.raises(ReferenceBuildError, match="no panel gene"):
        ReferenceStore(tmp_path / "store").get_or_build(
            spec, panel, builder=builder_for(spec)
        )
    assert fake.calls["precompute"] == []


def merfish_cells() -> pd.DataFrame:
    rows = []

    def add(
        section: str,
        ap: float,
        division: str,
        structure: str,
        cls: str,
        sub: str,
        n: int,
    ) -> None:
        rows.extend(
            {
                "brain_section_label": section,
                "class": cls,
                "subclass": sub,
                "x_ccf": ap,
                "parcellation_division": division,
                "parcellation_structure": structure,
            }
            for _ in range(n)
        )

    add("S.1", 2.0, "OLF", "MOB", "05 OB-IMN GABA", "045 OB-STR-CTX Inh IMN", 6)
    add(
        "S.1",
        2.0,
        "OLF",
        "OLF-unassigned",
        "05 OB-IMN GABA",
        "045 OB-STR-CTX Inh IMN",
        2,
    )
    add("S.1", 2.0, "Isocortex", "MOp", "01 IT-ET Glut", "007 L2/3 IT CTX Glut", 4)
    add("S.2", 5.0, "OLF", "OLF-unassigned", "01 IT-ET Glut", "007 L2/3 IT CTX Glut", 3)
    add("S.2", 5.0, "STR", "CP", "09 CNU-LGE GABA", "061 STR D1 Gaba", 10)
    add("S.2", 5.0, "lfbs", "cc", "31 OPC-Oligo", "327 Oligo NN", 5)
    add("S.3", 8.0, "HPF", "CA1", "01 IT-ET Glut", "016 CA1-ProS Glut", 8)
    add("S.3", 8.0, "VL", "VL", "30 Astro-Epen", "325 CHOR NN", 2)
    add(
        "S.3",
        8.0,
        "brain-unassigned",
        "unassigned",
        "30 Astro-Epen",
        "319 Astro-TE NN",
        1,
    )
    add("S.3", 8.0, "OLF", "In", "05 OB-IMN GABA", "045 OB-STR-CTX Inh IMN", 1)
    return pd.DataFrame(rows)


def test_merfish_region_splits_ob_from_olf() -> None:
    cells = merfish_cells()
    region = merfish_region(
        cells["parcellation_division"], cells["parcellation_structure"], cells["x_ccf"]
    )
    counts = region.value_counts().to_dict()
    # MOB (6) + anterior OLF-unassigned (2) + olfactory nerve layer "In" (1).
    assert counts["OB"] == 9
    assert counts["OLF"] == 3  # OLF-unassigned posterior of AP 2.5 stays OLF
    assert counts["fibre"] == 5 and counts["VS"] == 2 and counts["unassigned"] == 1


def test_region_share_tables_cover_every_section_with_windows() -> None:
    cells = merfish_cells()
    cells["region"] = merfish_region(
        cells["parcellation_division"], cells["parcellation_structure"], cells["x_ccf"]
    )
    tables = region_share_tables(cells)
    shares = tables["region_share"]
    for (_level, _node), group in shares.groupby(["level", "node_name"]):
        assert group["share"].sum() == pytest.approx(1.0)
    it = shares[(shares.level == "class") & (shares.node_name == "01 IT-ET Glut")]
    assert int(it["n_grey"].iloc[0]) == 15
    oligo = shares[shares.node_name == "31 OPC-Oligo"]
    assert oligo.empty  # fibre-tract cells only: not in the grey shares
    home = tables["region_home"].set_index(["level", "node_name"])
    assert home.loc[("subclass", "045 OB-STR-CTX Inh IMN"), "home"] == "OB"
    windows = tables["ap_windows"]
    assert windows["window"].nunique() == 3
    first = windows[windows.window == "S.1"].iloc[0]
    assert first["sections"] == "S.1;S.2" and first["n_sections"] == 2
    middle = windows[windows.window == "S.2"].iloc[0]
    assert middle["sections"] == "S.1;S.2;S.3"
    for (_window, _level), group in windows.groupby(["window", "level"]):
        assert group["freq"].sum() == pytest.approx(1.0)
    composition = tables["section_composition"]
    s2 = composition[(composition.section == "S.2") & (composition.level == "class")]
    assert int(s2["n_cells"].iloc[0]) == 18  # all cells, fibre included
    assert set(tables["section_regions"]["region"]) <= set(reference.GREY_REGIONS)


def test_cli_builds_the_region_share_bundle_then_reuses_it(tmp_path: Path) -> None:
    csv = tmp_path / "cell_metadata_with_parcellation_annotation.csv"
    merfish_cells().to_csv(csv, index=False)
    args = [
        "annotation-reference-prep",
        "--reference-id",
        "wmb_region_share",
        "--species",
        "mouse",
        "--store",
        str(tmp_path / "store"),
        "--source",
        f"merfish_ccf_metadata={csv}",
        "--output",
        str(tmp_path / "bundle_ref.json"),
    ]
    runner = CliRunner()
    first = runner.invoke(cli_main, args)
    assert first.exit_code == 0, first.output
    built = json.loads((tmp_path / "bundle_ref.json").read_text())
    assert built["reused"] is False and built["panel_hash"] is None
    bundle_dir = Path(built["path"])
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    output = manifest["builder_output"]
    assert output["n_sections"] == 3 and output["n_windows"] == 3
    assert output["region_scheme"] == "e7_v2"
    assert output["source_matches_allen_release"] is False
    for name in (
        reference.REGION_SHARE_FILE,
        reference.REGION_HOME_FILE,
        reference.SECTION_COMPOSITION_FILE,
        reference.SECTION_REGIONS_FILE,
        reference.AP_WINDOWS_FILE,
    ):
        assert (bundle_dir / name).is_file()
    second = runner.invoke(cli_main, args)
    assert second.exit_code == 0, second.output
    assert json.loads((tmp_path / "bundle_ref.json").read_text())["reused"] is True


def test_whole_ctx_builder_is_refused_above_1000_genes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeCtm(whb_truncated_lookup).install(monkeypatch)
    stats = write_precompute(
        tmp_path / "whole.h5", WHB_TREE, GENES, n_cells=WHB_N_CELLS
    )
    spec = AnnotationReferenceSpec(
        reference_id="whb_whole_ctx_panel",
        species="human",
        role="sensitivity",
        sources={"whb_whole_precompute": stats},
    )
    panel = make_panel([f"ENSG{index:011d}" for index in range(1001)])
    with pytest.raises(ReferenceBuildError, match="refused above 1000"):
        ReferenceStore(tmp_path / "store").get_or_build(
            prepare_reference_spec(spec), panel, builder=builder_for(spec)
        )


def test_cortex_implausible_superclusters_are_the_16_pruned_in_e1() -> None:
    nodes = reference.cortex_implausible_superclusters()
    assert len(nodes) == 16
    assert f"{SUPC}/CS202210140_475" in nodes  # Hippocampal CA1-3
    assert f"{SUPC}/{UL_IT}" not in nodes


# --------------------------------------------------------------------------
# Slow: a tiny real bundle through the real cell_type_mapper runners

REAL_CLASSES = {
    "CS20230722_CLAS_01": (
        "01 IT-ET Glut",
        ["CS20230722_SUBC_001", "CS20230722_SUBC_002"],
    ),
    "CS20230722_CLAS_30": (
        "30 Astro-Epen",
        ["CS20230722_SUBC_319", "CS20230722_SUBC_320"],
    ),
}
REAL_SUBCLASS_CLUSTERS = {
    "CS20230722_SUBC_001": [1, 2],
    "CS20230722_SUBC_002": [3, 4],
    "CS20230722_SUBC_319": [5, 6],
    "CS20230722_SUBC_320": [7, 8, 9],
}


def write_abc_taxonomy(directory: Path) -> tuple[Path, Path]:
    """Write ABC-style ``cluster_annotation_term`` and membership tables."""
    terms: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []

    def term(
        label: str,
        name: str,
        level: str,
        parent: str | None,
        parent_level: str | None,
        set_order: int,
        order: int,
        set_name: str,
    ) -> None:
        terms.append(
            {
                "label": label,
                "name": name,
                "cluster_annotation_term_set_label": level,
                "parent_term_label": parent or "",
                "parent_term_set_label": parent_level or "",
                "term_set_order": set_order,
                "term_order": order,
                "cluster_annotation_term_set_name": set_name,
                "color_hex_triplet": "#000000",
            }
        )

    order = 0
    for class_label, (class_name, subclasses) in REAL_CLASSES.items():
        term(class_label, class_name, CLAS, None, None, 0, order, "class")
        for subclass in subclasses:
            term(
                subclass,
                f"{subclass[-3:]} sub",
                SUBC_W,
                class_label,
                CLAS,
                1,
                order,
                "subclass",
            )
            supertype = subclass.replace("SUBC", "SUPT")
            term(
                supertype,
                f"{subclass[-3:]} supt",
                SUPT,
                subclass,
                SUBC_W,
                2,
                order,
                "supertype",
            )
            for alias in REAL_SUBCLASS_CLUSTERS[subclass]:
                cluster = f"CS20230722_CLUS_{alias:04d}"
                term(
                    cluster,
                    f"{alias:04d} clus",
                    CLUS_W,
                    supertype,
                    SUPT,
                    3,
                    order,
                    "cluster",
                )
                chain = [
                    (class_label, CLAS, class_name, "class"),
                    (subclass, SUBC_W, f"{subclass[-3:]} sub", "subclass"),
                    (supertype, SUPT, f"{subclass[-3:]} supt", "supertype"),
                    (cluster, CLUS_W, f"{alias:04d} clus", "cluster"),
                ]
                for label, level, name, set_name in chain:
                    members.append(
                        {
                            "cluster_annotation_term_label": label,
                            "cluster_annotation_term_set_label": level,
                            "cluster_alias": alias,
                            "cluster_annotation_term_name": name,
                            "cluster_annotation_term_set_name": set_name,
                            "number_of_cells": 25,
                            "color_hex_triplet": "#000000",
                        }
                    )
                order += 1
    directory.mkdir(parents=True, exist_ok=True)
    term_path = directory / "cluster_annotation_term.csv"
    member_path = directory / "cluster_to_cluster_annotation_membership.csv"
    pd.DataFrame(terms).to_csv(term_path, index=False)
    pd.DataFrame(members).to_csv(member_path, index=False)
    return term_path, member_path


def write_real_wmb_sources(tmp_path: Path) -> dict[str, Path]:
    """~200 WMB-like 10Xv3 cells in 9 clusters with planted markers."""
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(7)
    n_genes = 40
    genes = [f"ENSMUSG{index:011d}" for index in range(1, n_genes + 1)]
    subclass_of = {
        alias: subclass
        for subclass, aliases in REAL_SUBCLASS_CLUSTERS.items()
        for alias in aliases
    }
    class_of = {
        subclass: class_label
        for class_label, (_name, subclasses) in REAL_CLASSES.items()
        for subclass in subclasses
    }
    subclass_index = {
        subclass: index for index, subclass in enumerate(REAL_SUBCLASS_CLUSTERS)
    }
    class_index = {class_label: index for index, class_label in enumerate(REAL_CLASSES)}
    rows, blocks = [], []
    for alias in range(1, 10):
        base = rng.gamma(2.0, 1.0, n_genes)
        base[alias - 1] += 40.0  # cluster markers: genes 0-8
        base[10 + subclass_index[subclass_of[alias]]] += 40.0  # subclass: 10-13
        base[20 + class_index[class_of[subclass_of[alias]]]] += 60.0  # class: 20-21
        counts = rng.poisson(base, size=(25, n_genes)).astype(np.float32)
        blocks.append(counts)
        matrix = "WMB-10Xv3-AAA" if alias <= 4 else "WMB-10Xv3-BBB"
        rows.extend(
            {
                "cell_label": f"cell-{alias}-{index}",
                "feature_matrix_label": matrix,
                "library_method": "10Xv3",
                "cluster_alias": alias,
            }
            for index in range(25)
        )
    meta = pd.DataFrame(rows)
    matrix_all = np.vstack(blocks)
    h5ad_dir = tmp_path / "WMB-10Xv3"
    h5ad_dir.mkdir()
    for matrix, group in meta.groupby("feature_matrix_label"):
        ad.AnnData(
            X=sp.csr_matrix(matrix_all[group.index.to_numpy()]),
            obs=pd.DataFrame(index=group["cell_label"].to_numpy()),
            var=pd.DataFrame(index=genes),
        ).write_h5ad(h5ad_dir / f"{matrix}-raw.h5ad")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    meta.to_csv(metadata / "cell_metadata.csv", index=False)
    term_path, member_path = write_abc_taxonomy(metadata)
    # Cluster 9's cells are all self-map test cells: covered by the mapping
    # means, absent from the marker precompute.
    truth = tmp_path / "truth.csv"
    meta.loc[meta.cluster_alias == 9, ["cell_label"]].set_index("cell_label").assign(
        subclass="x"
    ).to_csv(truth)
    mapping = tmp_path / "allen" / "precomputed_stats_ABC_revision_230821.h5"
    mapping.parent.mkdir()
    reference._run_ctm_precompute_abc(
        {
            "output_path": str(mapping),
            "hierarchy": [CLAS, SUBC_W, SUPT, CLUS_W],
            "h5ad_path_list": [str(path) for path in sorted(h5ad_dir.glob("*.h5ad"))],
            "cell_metadata_path": str(metadata / "cell_metadata.csv"),
            "cluster_annotation_path": str(term_path),
            "cluster_membership_path": str(member_path),
            "n_processors": 2,
            "tmp_dir": str(tmp_path),
            "clobber": True,
            "normalization": "raw",
            "do_pruning": True,
        },
        log_dir=tmp_path / "mapping_logs",
    )
    return {
        "h5ad_dir": h5ad_dir,
        "metadata": metadata,
        "truth": truth,
        "mapping": mapping,
        "genes": Path(json.dumps(genes)),
    }


@pytest.mark.slow
def test_tiny_real_wmb_bundle_through_the_real_ctm_runners(
    tmp_path: Path, small_resources: Any
) -> None:
    pytest.importorskip("cell_type_mapper")
    sources = write_real_wmb_sources(tmp_path)
    genes = json.loads(str(sources["genes"]))
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id="wmb_panel",
            species="mouse",
            role="primary",
            drop_level=SUPT,
            max_cells_per_cluster=20,
            sources={
                "wmb_h5ad_dir": sources["h5ad_dir"],
                "wmb_metadata_dir": sources["metadata"],
                "wmb_mapping_stats": sources["mapping"],
                "wmb_selfmap_test_cells": sources["truth"],
            },
        )
    )
    panel = make_panel(genes[:30], species="mouse")
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    bundle = store.get_or_build(spec, panel, builder=builder_for(spec))
    bundle_dir = Path(bundle.path)
    output = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    assert output["levels"] == [CLAS, SUBC_W, CLUS_W]
    assert output["node_counts"] == {CLAS: 2, SUBC_W: 4, CLUS_W: 9}
    assert output["marker_precompute"]["sampling"]["n_sampled_cells"] == 8 * 20
    assert [entry["node"] for entry in output["uncovered_clusters"]] == [
        "CS20230722_CLUS_0009"
    ]
    assert output["uncovered_subclasses"] == []
    lookup = json.loads(
        (bundle_dir / reference.QUERY_MARKERS_FILTERED_FILE).read_text()
    )
    panel_genes = set(panel.ensembl_ids)
    assert lookup["None"] and set(lookup["None"]) <= panel_genes
    # The planted class markers separate the root.
    assert {genes[20], genes[21]} & set(lookup["None"])
    assert not [key for key in lookup if key.startswith(SUPT)]
    markers = output["markers"]
    assert markers["root_markers"] > 0 and markers["markers_per_parent_min"] >= 0
    assert output["panel_coverage"]["n_query_genes_used"] == 30
    profiles = pd.read_parquet(bundle_dir / reference.PROFILES_FILE)
    ul = profiles[
        (profiles.level == CLUS_W) & (profiles.node == "CS20230722_CLUS_0001")
    ]
    top = ul.sort_values("expected_fraction").iloc[-1]
    assert top["gene_id"] == genes[20]  # class marker (the strongest) leads
    negatives = pd.read_parquet(bundle_dir / reference.NEGATIVE_GENES_FILE)
    assert set(negatives["broad_class"]) == {"Neurons", "Astrocytes/Ependymal"}
    assert output["mapping_precompute"]["sha256"] == reference.file_sha256(
        sources["mapping"]
    )
    for name in (
        reference.MAPPING_PRECOMPUTE_FILE,
        reference.MARKER_PRECOMPUTE_FILE,
        reference.QUERY_MARKERS_FILE,
        reference.VOCAB_SNAPSHOT_FILE,
        reference.DEPTH_GRID_FILE,
        reference.MAPPING_TREE_FILE,
    ):
        path = bundle_dir / name
        assert path.is_file() and not path.is_symlink()
    assert not list((tmp_path / "scratch").iterdir())


def test_prepare_reference_spec_fills_seaad_sources_from_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins, seeds = [], {}
    for pinned in reference.SEAAD_MULTIREGION_FILES:
        content = f"synthetic {pinned.key}".encode()
        seed = tmp_path / "archive" / f"{pinned.key}.bin"
        seed.parent.mkdir(exist_ok=True)
        seed.write_bytes(content)
        seeds[pinned.key] = seed
        pins.append(
            PinnedFile(
                key=pinned.key,
                url=pinned.url,
                relative_path=pinned.relative_path,
                size=len(content),
                md5=hashlib.md5(content).hexdigest(),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    monkeypatch.setattr(reference, "SEAAD_MULTIREGION_FILES", tuple(pins))
    monkeypatch.setattr(reference, "SEAAD_STATS_PIN", pins[0])
    spec = AnnotationReferenceSpec(
        reference_id="seaad_mr_panel", species="human", role="secondary"
    )
    cache = tmp_path / "store" / ".downloads"
    prepared = prepare_reference_spec(
        spec, SourceOptions(download_dir=cache, seeds=seeds)
    )
    stats = prepared.sources["seaad_precomputed_stats"]
    assert stats == cache / pins[0].relative_path
    assert stats.read_bytes() == b"synthetic precomputed_stats"
    assert set(reference.SEAAD_TAXONOMY_SOURCES.values()) <= set(prepared.sources)
    assert all(not path.is_symlink() for path in prepared.sources.values())


def test_ctm_steps_run_in_a_fresh_single_threaded_interpreter(tmp_path: Path) -> None:
    environment = reference.ctm_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert all(environment[name] == "1" for name in reference.CTM_SINGLE_THREAD_ENV)
    script = (
        "import os, sys; print(os.environ['OMP_NUM_THREADS']); "
        "sys.exit(int(sys.argv[1]))"
    )
    metrics = reference.run_python_step("ok", ["-c", script, "0"], log_dir=tmp_path)
    assert metrics["returncode"] == 0 and metrics["peak_rss_gb"] > 0
    assert (tmp_path / "ok.stdout.log").read_text().strip() == "1"
    with pytest.raises(ReferenceBuildError, match="exit code 3"):
        reference.run_python_step("bad", ["-c", script, "3"], log_dir=tmp_path)
