"""Tests for the reference bundle builders (plan §3.2, §5.6, §7.1, §12 M2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from merxen.annotation import reference
from merxen.annotation import resolvability as res
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
    LargePanelRefusedError,
    ReferenceStore,
    large_panel_refusal,
    resolve_builder,
)
from merxen.annotation.vocab import load_state_gene_ids, load_state_genes
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
    profiles = reference_profiles_from_stats(stats, levels=["A", "B"])
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
    # Bundles are keyed by IDs; symbols come from the run's panel.
    assert "gene_symbol" not in profiles.columns
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
        state_gene_ids=["g_state"],
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
    assert "gene_symbol" not in table.columns


def test_state_genes_are_matched_by_id_whatever_the_panel_symbols() -> None:
    # CDKN1A (ENSG00000124762) declared under an alias, or by ID only.
    ids = load_state_gene_ids("human")
    assert ids["ENSG00000124762"] == "CDKN1A"
    assert set(load_state_gene_ids("mouse").values()) == set(load_state_genes("mouse"))
    genes = ["ENSG00000124762", "ENSG00000000001"]
    frame = pd.DataFrame([[0.0, 0.0]], index=["Astrocytes"], columns=genes)
    table = negative_gene_table(
        {"ref": (frame, pd.Series([100.0], index=["Astrocytes"]))},
        genes=genes,
        state_gene_ids=ids,
    ).set_index("gene_id")
    assert table.loc["ENSG00000124762", "is_state_gene"]
    assert not table.loc["ENSG00000124762", "negative"]
    assert table.loc["ENSG00000000001", "negative"]


def test_state_gene_assets_resolve_every_symbol_to_one_id() -> None:
    for species, prefix in (("human", "ENSG"), ("mouse", "ENSMUSG")):
        ids = load_state_gene_ids(species)
        assert len(ids) == len(load_state_genes(species))
        assert all(gene.startswith(prefix) for gene in ids)


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


def test_merfish_pin_matches_the_region_share_release_checks() -> None:
    pinned = reference.MERFISH_CCF_METADATA_PIN
    assert pinned.md5 == reference.MERFISH_CCF_METADATA_MD5
    assert pinned.size == reference.MERFISH_CCF_METADATA_SIZE
    assert pinned.relative_path.endswith(
        "MERFISH-C57BL6J-638850-CCF/20231215/views/"
        "cell_metadata_with_parcellation_annotation.csv"
    )
    assert pinned.url.endswith(pinned.relative_path)


def test_prepare_reference_spec_fetches_the_pinned_merfish_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = AnnotationReferenceSpec(
        reference_id="wmb_region_share", species="mouse", role="region_share"
    )
    with pytest.raises(ReferenceBuildError, match="merfish_ccf_metadata"):
        prepare_reference_spec(spec)
    content = b"cell_label,parcellation_division\n"
    pinned = pinned_for(tmp_path, content)
    monkeypatch.setattr(reference, "MERFISH_CCF_METADATA_PIN", pinned)
    seed = tmp_path / "archived.csv"
    seed.write_bytes(content)
    cache = tmp_path / "cache"
    with pytest.raises(PinnedFileError, match="downloads are off"):
        prepare_reference_spec(spec, SourceOptions(download_dir=cache))
    prepared = prepare_reference_spec(
        spec, SourceOptions(download_dir=cache, seeds={pinned.key: seed})
    )
    target = cache / pinned.relative_path
    assert prepared.sources["merfish_ccf_metadata"] == target
    assert target.read_bytes() == content and not target.is_symlink()
    # An explicit source wins over the cache.
    explicit = spec.model_copy(update={"sources": {"merfish_ccf_metadata": seed}})
    assert (
        prepare_reference_spec(explicit, SourceOptions(download_dir=cache)).sources[
            "merfish_ccf_metadata"
        ]
        == seed
    )


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


def without_resolvability(species: str) -> AnnotationConfig:
    """A config whose builds skip the resolvability self-map (tested apart)."""
    return AnnotationConfig(species=species, resolvability={"enabled": False})


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
    config = without_resolvability("human")
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
    assert coverage["absent_from_reference"] == [ABSENT_GENE]
    assert output["marker_unsupported_nodes"]["nodes"] == [
        f"{CLUS}/c4",
        f"{CLUS}/c5",
    ]
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
    config = without_resolvability("human")
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
    manifest = json.loads((Path(bundle_c.path) / BUNDLE_MANIFEST_NAME).read_text())
    assert manifest["built_from_panel"]["kind"] == "setc"
    assert manifest["built_from_panel"]["parent_panel_hash"] == set_a.panel_hash
    assert "panel_kind" not in manifest["builder_output"]
    assert fake.stub_genes == GENES[:8]


def test_whb_bundle_is_shared_by_panels_that_differ_only_in_symbols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # A gene list (one symbol per ID, or IDs only) and a prepared pair panel
    # (MERSCOPE H2AX, Xenium H2AFX) with the same IDs share one bundle.
    sources = write_whb_sources(tmp_path)
    prepared = make_panel(GENES)
    gene_list = prepared.model_copy(
        update={
            "name": "gene_list",
            "kind": "gene_list",
            "symbols": list(prepared.ensembl_ids),
            "sample_ids": [],
        }
    )
    store, bundle, fake = build_whb(tmp_path, monkeypatch, prepared, sources)
    config = without_resolvability("human")
    again = store.get_or_build(
        prepare_reference_spec(
            whb_spec(
                region_precompute=sources["region_dir"],
                seaad_precomputed_stats=sources["seaad"],
            )
        ),
        gene_list,
        builder=builder_for(whb_spec(), config),
        config=config,
    )
    assert again.reused and again.build_hash == bundle.build_hash
    assert len(fake.calls["reference"]) == 1


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
    bundle = store.get_or_build(
        spec,
        panel,
        builder=builder_for(spec, without_resolvability("human")),
        config=without_resolvability("human"),
    )
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
    bundle = store.get_or_build(
        spec,
        panel,
        builder=builder_for(spec, without_resolvability("mouse")),
        config=without_resolvability("mouse"),
    )
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
    # Synthetic genes lie outside the validated ag7 | VZG2 universe.
    assert universe["universe_source"] == "panel"
    assert universe["validated_universe"] is False
    assert output["selfmap_test_cells_excluded"] is True
    assert output["selfmap_test_cells"]["matches_validated_test_set"] is False
    assert output["validated_configuration"]["all"] is False
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
    # Uncovered nodes and the leaves under collapsed parents have no marker
    # support; ctm can still emit them, so RESOLVE gets the list.
    unsupported = output["marker_unsupported_nodes"]
    assert unsupported["nodes"] == [
        f"{CLUS_W}/CS20230722_CLUS_0004",
        f"{CLUS_W}/CS20230722_CLUS_0005",
        f"{CLUS_W}/CS20230722_CLUS_0006",
        f"{CLUS_W}/CS20230722_CLUS_0007",
        f"{SUBC_W}/CS20230722_SUBC_320",
    ]
    reasons = {entry["node"]: entry for entry in unsupported["details"]}
    assert reasons["CS20230722_SUBC_320"]["reason"] == "no_marker_training_cell"
    assert reasons["CS20230722_CLUS_0004"]["reason"] == "below_collapsed_parent"
    assert reasons["CS20230722_CLUS_0006"]["also"] == "no_marker_training_cell"
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


def test_wmb_panel_inside_the_validated_universe_uses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # ag7 (500) and VZG2 (815) lie inside the 899-gene union their validated
    # lookups were built on; raw-CPM normalisation over the union reproduces
    # them exactly (plan §7.1, R28).
    import anndata as ad

    sources = write_wmb_sources(tmp_path)
    fake = FakeCtm(wmb_lookup).install(monkeypatch)
    trained_genes: list[str] = []

    def precompute(config: dict[str, Any]) -> None:
        path = config["h5ad_path_list"][0]
        trained_genes.extend(ad.read_h5ad(path, backed="r").var_names)
        wmb_marker_precompute(config)

    fake.precompute = precompute
    monkeypatch.setattr(
        reference, "validated_wmb_marker_universe", lambda: list(MOUSE_GENES)
    )
    spec = prepare_reference_spec(wmb_spec(sources))
    panel = make_panel(MOUSE_GENES[:6], species="mouse")
    bundle = ReferenceStore(tmp_path / "store").get_or_build(
        spec,
        panel,
        builder=builder_for(spec, without_resolvability("mouse")),
        config=without_resolvability("mouse"),
    )
    output = json.loads((Path(bundle.path) / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    universe = output["marker_precompute"]["gene_universe"]
    assert universe["universe_source"] == reference.WMB_UNIVERSE_VALIDATED
    assert universe["n_genes"] == 8 and universe["n_extra_universe_genes"] == 2
    assert trained_genes == sorted(MOUSE_GENES)
    # Markers are still found on the panel genes only.
    assert fake.stub_genes == MOUSE_GENES[:6]
    assert output["validated_configuration"]["marker_universe"] is True


def test_the_packaged_wmb_universe_is_the_validated_union() -> None:
    universe = reference.validated_wmb_marker_universe()
    assert len(universe) == 899 and universe == sorted(set(universe))
    table = pd.read_csv(reference.ASSET_DIR / reference.WMB_MARKER_UNIVERSE_FILE)
    assert int(table["in_ag7"].sum()) == 500 and int(table["in_vzg2"].sum()) == 815
    assert all(gene.startswith("ENSMUSG") for gene in universe)


def test_wmb_builder_requires_the_test_cells_while_resolvability_is_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_wmb_sources(tmp_path)
    fake = FakeCtm(wmb_lookup).install(monkeypatch)
    fake.precompute = wmb_marker_precompute
    base = wmb_spec(sources)
    spec = prepare_reference_spec(
        base.model_copy(
            update={
                "sources": {
                    k: v
                    for k, v in base.sources.items()
                    if k != "wmb_selfmap_test_cells"
                }
            }
        )
    )
    panel = make_panel(MOUSE_GENES[:6], species="mouse")
    store = ReferenceStore(tmp_path / "store")
    with pytest.raises(ReferenceBuildError, match="self-map test cells"):
        store.get_or_build(spec, panel, builder=builder_for(spec))
    assert fake.calls["precompute"] == []
    config = AnnotationConfig(species="mouse")
    config = config.model_copy(
        update={
            "resolvability": config.resolvability.model_copy(update={"enabled": False})
        }
    )
    bundle = store.get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    output = json.loads((Path(bundle.path) / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    assert output["selfmap_test_cells_excluded"] is False


def test_wmb_matrices_ignore_lock_and_staging_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_wmb_sources(tmp_path)
    fake = FakeCtm(wmb_lookup).install(monkeypatch)
    fake.precompute = wmb_marker_precompute
    spec = prepare_reference_spec(wmb_spec(sources))
    panel = make_panel(MOUSE_GENES[:6], species="mouse")
    store = ReferenceStore(tmp_path / "store")
    before = store.prepare_request(
        spec, panel, builder=builder_for(spec, without_resolvability("mouse"))
    )
    # What the legacy downloader leaves in the shared ABC cache.
    (sources["h5ad_dir"] / "WMB-10Xv3-AAA-raw.h5ad.lock").write_bytes(b"")
    (sources["h5ad_dir"] / "WMB-10Xv3-CCC-raw.h5ad.tmp").write_bytes(b"part")
    after = store.prepare_request(
        spec, panel, builder=builder_for(spec, without_resolvability("mouse"))
    )
    assert after.build_hash == before.build_hash
    names = [Path(f.path).name for f in after.sources["wmb_h5ad_dir"].files]
    assert names == ["WMB-10Xv3-AAA-raw.h5ad", "WMB-10Xv3-BBB-raw.h5ad"]
    bundle = store.get_or_build(
        spec,
        panel,
        builder=builder_for(spec, without_resolvability("mouse")),
        config=without_resolvability("mouse"),
    )
    output = json.loads((Path(bundle.path) / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    assert output["marker_precompute"]["sampling"]["matrices"] == [
        "WMB-10Xv3-AAA",
        "WMB-10Xv3-BBB",
    ]


def test_prepare_reference_spec_seeds_the_pinned_test_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sources = write_wmb_sources(tmp_path)
    truth = sources["truth"]
    pin = PinnedFile(
        key="wmb_selfmap_test_cells",
        url="",
        relative_path="local/wmb_selfmap/truth.csv",
        size=truth.stat().st_size,
        md5="",
        sha256=reference.file_sha256(truth),
    )
    monkeypatch.setattr(reference, "WMB_SELFMAP_TEST_CELLS_PIN", pin)
    downloads = tmp_path / "store" / ".downloads"
    options = SourceOptions(download_dir=downloads)
    spec = prepare_reference_spec(wmb_spec(sources), options)
    cached = downloads / "local" / "wmb_selfmap" / "truth.csv"
    # The bundle hashes the stable cached copy, not the archive path.
    assert spec.sources["wmb_selfmap_test_cells"] == cached
    assert reference.file_sha256(cached) == pin.sha256
    assert cached.stat().st_mode & 0o222 == 0
    # Once seeded, the cache serves builds that name no list.
    base = wmb_spec(sources)
    without = base.model_copy(
        update={
            "sources": {
                k: v for k, v in base.sources.items() if k != "wmb_selfmap_test_cells"
            }
        }
    )
    assert (
        prepare_reference_spec(without, options).sources["wmb_selfmap_test_cells"]
        == cached
    )
    # Another list is used as given, with a warning.
    other = tmp_path / "other_truth.csv"
    other.write_text("cell_label\nX\n")
    changed = base.model_copy(
        update={"sources": {**base.sources, "wmb_selfmap_test_cells": other}}
    )
    with caplog.at_level("WARNING"):
        kept = prepare_reference_spec(changed, options)
    assert kept.sources["wmb_selfmap_test_cells"] == other
    assert "not the validated self-map test set" in caplog.text


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
    assert "annotation-reference-prep: built" in first.output
    built = json.loads((tmp_path / "bundle_ref.json").read_text())
    assert built["panel_hash"] is None and "reused" not in built
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
    assert "annotation-reference-prep: reused" in second.output
    assert json.loads((tmp_path / "bundle_ref.json").read_text()) == built


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
    store = ReferenceStore(tmp_path / "store")
    with pytest.raises(LargePanelRefusedError, match="refused above 1000 genes"):
        store.get_or_build(
            prepare_reference_spec(spec), panel, builder=builder_for(spec)
        )
    # Refused before a build directory exists.
    assert store.list() == []


def large_config(
    species: str = "human",
    *,
    limit: int = 5,
    cap: int = 6,
    prefilter: str | None = None,
) -> AnnotationConfig:
    """A config whose "large" panels start above ``limit`` genes (no self-map)."""
    config = without_resolvability(species)
    update: dict[str, Any] = {
        "large_panel_genes": limit,
        "large_panel_prefilter_cap": cap,
    }
    if prefilter is not None:
        update["large_panel_marker_prefilter"] = prefilter
    return config.model_copy(update={"panel": config.panel.model_copy(update=update)})


def test_large_panels_build_and_the_prefilter_is_mandatory_above_the_reserve(
    small_resources: Any,
) -> None:
    # Default since the M3b 5K measurement: no prefilter.
    config = AnnotationConfig(species="mouse")
    assert config.panel.large_panel_marker_prefilter == "none"
    wmb = builder_for(
        AnnotationReferenceSpec(
            reference_id="wmb_panel", species="mouse", role="primary"
        ),
        config,
    )
    five_k = make_panel(
        [f"ENSMUSG{index:011d}" for index in range(5006)], species="mouse"
    )
    huge = make_panel(
        [f"ENSMUSG{index:011d}" for index in range(20000)], species="mouse"
    )
    # The measured envelope: the largest measured peak up to 5,006 genes,
    # scaled beyond.
    assert reference.predicted_wmb_query_marker_peak_gb(815) == pytest.approx(37.7)
    assert reference.predicted_wmb_query_marker_peak_gb(5006) == pytest.approx(37.7)
    assert reference.predicted_wmb_query_marker_peak_gb(10012) == pytest.approx(75.4)
    # A 3 GB --max-gb gives a 4.8 GB reserve: the unfiltered 5K panel is
    # refused, the prefilter is then mandatory (OD-E8).
    reason = large_panel_refusal(five_k, config, wmb)
    assert reason is not None and "mandatory" in reason and "OD-E8" in reason
    prefiltered = large_config(
        "mouse", limit=1000, cap=2000, prefilter="per_parent_topk_union"
    )
    assert large_panel_refusal(five_k, prefiltered, wmb) is None
    # The standard reserve (--max-gb 40 = 64 GB) holds the measured 5K peak.
    reference.set_prep_resources(max_gb=40)
    assert large_panel_refusal(five_k, config, wmb) is None
    assert large_panel_refusal(None, config, wmb) is None
    # Beyond the measured range the scaled envelope can exceed it.
    assert large_panel_refusal(huge, config, wmb) is not None
    assert large_panel_refusal(huge, prefiltered, wmb) is None
    # The refusal keeps the OD-E8 margin (+30%): a panel whose predicted peak
    # fits the 64 GB reserve but not with the margin is refused.
    assert pytest.approx(1.3) == reference.PREP_MEMORY_MARGIN
    margin_panel = make_panel(
        [f"ENSMUSG{index:011d}" for index in range(7000)], species="mouse"
    )
    predicted = reference.predicted_wmb_query_marker_peak_gb(7000)
    assert predicted < reference.prep_memory_reserve_gb() < predicted * 1.3
    reason = large_panel_refusal(margin_panel, config, wmb)
    assert reason is not None and "+ 30%" in reason
    assert "annotation_prep_large_memory" in reason and "--annotation-config" in reason
    fits = make_panel(
        [f"ENSMUSG{index:011d}" for index in range(6500)], species="mouse"
    )
    assert large_panel_refusal(fits, config, wmb) is None
    # Small panels never need the prefilter.
    reference.set_prep_resources(max_gb=3)
    small = make_panel(
        [f"ENSMUSG{index:011d}" for index in range(815)], species="mouse"
    )
    assert large_panel_refusal(small, config, wmb) is None
    # The whole-WHB bundle stays refused above 1,000 genes.
    whole = builder_for(
        AnnotationReferenceSpec(
            reference_id="whb_whole_ctx_panel", species="human", role="sensitivity"
        )
    )
    human = make_panel([f"ENSG{index:011d}" for index in range(1001)])
    assert "refused above 1000" in (large_panel_refusal(human, None, whole) or "")
    assert large_panel_refusal(make_panel(GENES), None, whole) is None


def test_cli_refuses_a_large_wmb_panel_without_the_prefilter_before_building(
    tmp_path: Path,
) -> None:
    panel = make_panel(
        [f"ENSMUSG{index:011d}" for index in range(20000)], species="mouse"
    )
    panel_file = tmp_path / "panel_genes.json"
    panel.write(panel_file)
    config_file = tmp_path / "annotation_config.json"
    config_file.write_text(AnnotationConfig(species="mouse").model_dump_json())
    store = tmp_path / "store"
    result = CliRunner().invoke(
        cli_main,
        [
            "annotation-reference-prep",
            "--reference-id",
            "wmb_panel",
            "--species",
            "mouse",
            "--panel-genes",
            str(panel_file),
            "--store",
            str(store),
            "--annotation-config",
            str(config_file),
            "--max-gb",
            "40",
            "--output",
            str(tmp_path / "bundle_ref.json"),
        ],
    )
    assert result.exit_code != 0
    assert "prefilter is mandatory above the reserve" in result.output
    assert not (tmp_path / "bundle_ref.json").exists()
    # Nothing was hashed or built: no bundle, temporary or failed directory.
    assert not store.exists() or not any(store.rglob("*.json"))


def test_large_whb_panel_is_prefiltered_kept_and_stored_in_the_large_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    fake = FakeCtm(whb_truncated_lookup).install(monkeypatch)
    sources = write_whb_sources(tmp_path)
    panel = make_panel([*GENES, ABSENT_GENE])
    config = large_config("human", limit=5, cap=6, prefilter="per_parent_topk_union")
    spec = prepare_reference_spec(
        whb_spec(
            region_precompute=sources["region_dir"],
            seaad_precomputed_stats=sources["seaad"],
        )
    )
    (tmp_path / "scratch").mkdir()
    store = ReferenceStore(
        tmp_path / "ssd",
        large_root=tmp_path / "large",
        large_panel_genes=5,
        scratch_root=tmp_path / "scratch",
    )
    bundle = store.get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    assert bundle_dir.is_relative_to(tmp_path / "large")
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    output = manifest["builder_output"]
    # Marker discovery ran on the prefiltered candidates only.
    record = json.loads((bundle_dir / reference.MARKER_PREFILTER_FILE).read_text())
    assert record["method"] == "per_parent_topk_union" and record["version"] == 1
    assert len(fake.stub_genes) == record["n_genes"] <= 6
    assert sorted(fake.stub_genes) == record["genes"] and set(record["genes"]) < set(
        GENES
    )
    assert record["n_input_genes"] == 10 and record["applied"] is True
    assert output["markers"]["prefilter"]["genes_sha256"] == record["genes_sha256"]
    assert output["markers"]["n_candidate_genes"] == record["n_genes"]
    assert "marker_prefilter" in output["timings_s"]
    # Profiles still cover every panel gene in the reference.
    assert output["panel_coverage"]["n_query_genes_used"] == 10
    # The family's reference markers are kept in the large bundle.
    kept = bundle_dir / reference.REFERENCE_MARKERS_DIR / "reference_markers.h5"
    assert kept.is_file()
    assert output["markers"]["reference_markers"][0]["kept"] is True
    # The prefilter (method, version, settings) is in build_hash.
    prefilter = manifest["build_hash_payload"]["large_panel_prefilter"]
    assert prefilter["method"] == "per_parent_topk_union"
    assert prefilter["settings"]["cap"] == 6
    assert prefilter["settings"]["markers_per_pair"] == 30


def test_small_panel_reference_markers_are_deleted_once_query_markers_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_whb_sources(tmp_path)
    seen: dict[str, bool] = {}
    fake = FakeCtm(whb_truncated_lookup)
    original_query = fake.query_markers

    def query_markers(config: dict[str, Any], **kwargs: Any) -> None:
        # The reference markers exist while the query markers are found ...
        seen["during_query"] = all(
            Path(path).is_file() for path in config["reference_marker_path_list"]
        )
        seen["paths"] = config["reference_marker_path_list"]
        original_query(config, **kwargs)

    fake.query_markers = query_markers  # type: ignore[method-assign]
    fake.install(monkeypatch)
    original_find = reference.find_panel_markers
    right_after: dict[str, Any] = {}

    def find_panel_markers(*args: Any, **kwargs: Any) -> Any:
        markers = original_find(*args, **kwargs)
        # Checked as find_panel_markers returns, before the long self-map
        # and before the store removes the build's scratch directory.
        paths = [Path(path) for path in seen["paths"]]
        right_after["gone"] = [not path.exists() for path in paths]
        right_after["scratch_exists"] = all(path.parent.is_dir() for path in paths)
        right_after["kept"] = [item.get("kept") for item in markers.reference_markers]
        return markers

    monkeypatch.setattr(reference, "find_panel_markers", find_panel_markers)
    spec = prepare_reference_spec(
        whb_spec(
            region_precompute=sources["region_dir"],
            seaad_precomputed_stats=sources["seaad"],
        )
    )
    (tmp_path / "scratch").mkdir()
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    config = without_resolvability("human")
    bundle = store.get_or_build(
        spec, make_panel(GENES), builder=builder_for(spec, config), config=config
    )
    output = json.loads((Path(bundle.path) / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    # ... and are gone right after (<= 1,000 genes), their sha256 recorded.
    assert seen["during_query"] is True
    assert right_after["gone"] and all(right_after["gone"])
    assert right_after["scratch_exists"] is True
    assert right_after["kept"] == [False] * len(right_after["gone"])
    assert not any(Path(path).exists() for path in seen["paths"])
    record = output["markers"]["reference_markers"][0]
    assert record["kept"] is False and len(record["sha256"]) == 64
    assert output["markers"]["prefilter"] is None
    assert not (Path(bundle.path) / reference.MARKER_PREFILTER_FILE).exists()


def test_sibling_sets_follow_the_marker_tree_with_the_dropped_level() -> None:
    tree = TaxonomyTreeView.from_tree_dict(
        {
            "hierarchy": ["CLAS", "SUBC", "SUPT", "CLUS"],
            "CLAS": {"a": ["a1", "a2"], "b": ["b1"]},
            "SUBC": {"a1": ["t1"], "a2": ["t2", "t3"], "b1": ["t4"]},
            "SUPT": {"t1": ["c1", "c2"], "t2": ["c3"], "t3": ["c4"], "t4": ["c5"]},
            "CLUS": {f"c{index}": [] for index in range(1, 6)},
        }
    ).drop_level("SUPT")
    leaves = ["c5", "c4", "c3", "c2", "c1"]
    sets = {item.key: item for item in reference.sibling_sets(tree, leaves)}
    assert list(sets) == ["None", "CLAS/a", "CLAS/b", "SUBC/a1", "SUBC/a2", "SUBC/b1"]
    root = sets["None"]
    assert root.children == ("a", "b")
    # Rows follow the precompute's row order (c1 is row 4).
    assert root.leaf_rows == ((1, 2, 3, 4), (0,))
    assert sets["SUBC/a1"].children == ("c1", "c2")
    assert sets["SUBC/a1"].leaf_rows == ((4,), (3,))
    assert sets["CLAS/b"].children == ("b1",)


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
    sources = write_real_wmb_inputs(tmp_path)
    mapping = tmp_path / "allen" / "precomputed_stats_ABC_revision_230821.h5"
    mapping.parent.mkdir()
    reference._run_ctm_precompute_abc(
        {
            "output_path": str(mapping),
            "hierarchy": [CLAS, SUBC_W, SUPT, CLUS_W],
            "h5ad_path_list": [
                str(path) for path in sorted(sources["h5ad_dir"].glob("*.h5ad"))
            ],
            "cell_metadata_path": str(sources["metadata"] / "cell_metadata.csv"),
            "cluster_annotation_path": str(sources["term"]),
            "cluster_membership_path": str(sources["membership"]),
            "n_processors": 2,
            "tmp_dir": str(tmp_path),
            "clobber": True,
            "normalization": "raw",
            "do_pruning": True,
        },
        log_dir=tmp_path / "mapping_logs",
    )
    return {**sources, "mapping": mapping}


def write_real_wmb_inputs(tmp_path: Path) -> dict[str, Path]:
    """The h5ads, metadata, ABC taxonomy and self-map list of the tiny WMB set."""
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
    return {
        "h5ad_dir": h5ad_dir,
        "metadata": metadata,
        "truth": truth,
        "term": term_path,
        "membership": member_path,
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
    bundle = store.get_or_build(
        spec,
        panel,
        builder=builder_for(spec, without_resolvability("mouse")),
        config=without_resolvability("mouse"),
    )
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


def test_whb_builder_rebuilds_the_region_precompute_without_a_copy_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_whb_sources(tmp_path)
    metadata = tmp_path / "abc_whb" / "metadata"
    (metadata / "WHB-10Xv3" / "v").mkdir(parents=True)
    (metadata / "WHB-taxonomy" / "v").mkdir(parents=True)
    labels = [*reference.WHB_FRONTAL_ROI_LABELS, "Human MoAN"]
    cells = pd.DataFrame(
        {
            "cell_label": [f"cell{index}" for index in range(60)],
            "region_of_interest_label": [labels[index % 5] for index in range(60)],
            # alias 1 has 12 frontal cells, alias 2 only 9 (dropped).
            "cluster_alias": [
                1 if index < 15 else 2 if index < 26 else 3 for index in range(60)
            ],
        }
    )
    cells.to_csv(metadata / "WHB-10Xv3" / "v" / "cell_metadata.csv", index=False)
    pd.DataFrame({"region_of_interest_label": labels}).to_csv(
        metadata / "WHB-10Xv3" / "v" / "region_of_interest_structure_map.csv",
        index=False,
    )
    for name in ("cluster_annotation_term", "cluster_to_cluster_annotation_membership"):
        (metadata / "WHB-taxonomy" / "v" / f"{name}.csv").write_text("label\n")
    h5ads = tmp_path / "abc_whb" / "expression"
    h5ads.mkdir()
    for name in ("WHB-10Xv3-Neurons-raw.h5ad", "WHB-10Xv3-Nonneurons-raw.h5ad"):
        (h5ads / name).write_bytes(b"h5ad")
    fake = FakeCtm(whb_truncated_lookup).install(monkeypatch)

    def rebuilt(config: dict[str, Any]) -> None:
        region_cells = pd.read_csv(config["cell_metadata_path"])
        assert set(region_cells["region_of_interest_label"]) <= set(
            reference.WHB_FRONTAL_ROI_LABELS
        )
        assert set(region_cells["cluster_alias"]) == {1, 3}
        write_precompute(
            Path(config["output_path"]), WHB_TREE, GENES, n_cells=WHB_N_CELLS
        )

    fake.precompute = rebuilt
    spec = prepare_reference_spec(
        whb_spec(
            whb_metadata_dir=metadata,
            whb_h5ad_dir=h5ads,
            seaad_precomputed_stats=sources["seaad"],
        )
    )
    bundle = ReferenceStore(tmp_path / "store").get_or_build(
        spec,
        make_panel(GENES),
        builder=builder_for(spec, without_resolvability("human")),
        config=without_resolvability("human"),
    )
    output = json.loads((Path(bundle.path) / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    (precompute,) = fake.calls["precompute"]
    assert precompute["hierarchy"] == list(reference.WHB_SOURCE_HIERARCHY)
    assert precompute["do_pruning"] is True and precompute["normalization"] == "raw"
    assert [Path(path).name for path in precompute["h5ad_path_list"]] == [
        "WHB-10Xv3-Neurons-raw.h5ad",
        "WHB-10Xv3-Nonneurons-raw.h5ad",
    ]
    source = output["source_precompute"]
    assert source["mode"] == "rebuilt"
    assert source["filtering_summary"]["dropped_leaf_aliases"] == {"2": 9}
    assert source["verification"]["n_leaves"] == 7


# --------------------------------------------------------------------------
# Resolvability test sets and the self-map (M3b; plan §3.2, §8.3)

HO_DONORS = {"H_big": 60, "H_mid": 45, "H_small": 30}
SUBC_TO_CLUS = {
    "s1": "c1",
    "s2": "c1",
    "s3": "c2",
    "s4": "c3",
    "s5": "c3",
    "s6": "c4",
    "s7": "c5",
}
CLUS_TO_SUPC = {"c1": UL_IT, "c2": UL_IT, "c3": ASTRO, "c4": MICRO, "c5": MICRO}
MARKER_OF_SUPC = {UL_IT: 0, ASTRO: 1, MICRO: 2}
SEA_MARKERS = {
    "CS20260630_CLAS_002": 0,
    "CS20260630_CLAS_003": 1,
    "CS20260630_SCLA_023": 1,
    "CS20260630_SCLA_029": 2,
}


# Other-region WHB cells of the fixture (``HO_OTHER_REGION``): drawn only
# when non-neuronal (of a training supercluster) and of an E2 dissection;
# clusters are not restricted (c5 has no training cluster).
HO_OTHER_REGION = (
    # (label prefix, n, cluster alias, matrix, dissection, donor)
    ("mtg_astro", 4, 4, "WHB-10Xv3-Nonneurons", "Human MTG", "H_big"),
    ("v1c_micro", 3, 6, "WHB-10Xv3-Nonneurons", "Human V1C", "H_other"),
    ("mtg_c5", 2, 7, "WHB-10Xv3-Nonneurons", "Human MTG", "H_big"),  # cluster c5
    ("mtg_neuron", 2, 1, "WHB-10Xv3-Nonneurons", "Human MTG", "H_big"),  # neuron
    ("mtg_neuron_matrix", 2, 4, "WHB-10Xv3-Neurons", "Human MTG", "H_big"),
    ("a25_astro", 2, 4, "WHB-10Xv3-Nonneurons", "Human A25", "H_big"),  # not E2
)


def write_ho_sources(tmp_path: Path) -> dict[str, Path]:
    """Frontal-like WHB cells of three donors, raw h5ads and taxonomy tables.

    The WHB cell metadata holds the frontal cells and the other-region
    cells of ``HO_OTHER_REGION`` (whose counts come from a separate seed, so
    the frontal cells' counts do not depend on them).
    """
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(3)
    rows = []
    for donor, n_cells in HO_DONORS.items():
        for index in range(n_cells):
            # s7 (cluster c5) has only three training-donor cells: dropped.
            if donor == "H_small":
                alias = 1 + index % 7
            elif donor == "H_big" and index < 3:
                alias = 7
            else:
                alias = 1 + index % 6
            supercluster = CLUS_TO_SUPC[SUBC_TO_CLUS[f"s{alias}"]]
            rows.append(
                {
                    "cell_label": f"{donor}-{index}",
                    "feature_matrix_label": "WHB-10Xv3-Neurons"
                    if supercluster == UL_IT
                    else "WHB-10Xv3-Nonneurons",
                    "donor_label": donor,
                    "cluster_alias": alias,
                    "region_of_interest_label": "Human A46",
                }
            )
    meta = pd.DataFrame(rows)
    region = tmp_path / "region_ho"
    region.mkdir()
    meta.to_csv(region / reference.REGION_CELL_METADATA_FILE, index=False)
    metadata = tmp_path / "whb_metadata"
    metadata.mkdir()
    other = pd.DataFrame(
        [
            {
                "cell_label": f"{prefix}-{index}",
                "feature_matrix_label": matrix,
                "donor_label": donor,
                "cluster_alias": alias,
                "region_of_interest_label": roi,
            }
            for prefix, n_cells, alias, matrix, roi, donor in HO_OTHER_REGION
            for index in range(n_cells)
        ]
    )
    pd.concat([meta, other], ignore_index=True).assign(
        anatomical_division_label="Cerebral cortex"
    ).to_csv(metadata / "cell_metadata.csv", index=False)
    members = []
    for alias in range(1, 8):
        subcluster = f"s{alias}"
        cluster = SUBC_TO_CLUS[subcluster]
        for label, level in (
            (subcluster, SUBC),
            (cluster, CLUS),
            (CLUS_TO_SUPC[cluster], SUPC),
        ):
            members.append(
                {
                    "cluster_annotation_term_label": label,
                    "cluster_annotation_term_set_label": level,
                    "cluster_alias": alias,
                    "cluster_annotation_term_name": label,
                    "cluster_annotation_term_set_name": level,
                }
            )
    pd.DataFrame(members).to_csv(
        metadata / "cluster_to_cluster_annotation_membership.csv", index=False
    )
    (metadata / "cluster_annotation_term.csv").write_text("label,name\n")
    h5ad_dir = tmp_path / "WHB-10Xv3"
    h5ad_dir.mkdir()
    genes = [*GENES, "ENSG00000000999"]
    other_rng = np.random.default_rng(4)

    def cell_counts(aliases: Iterable[int], generator: np.random.Generator) -> list:
        counts = []
        for alias in aliases:
            base = generator.poisson(12.0, len(genes)).astype(float)
            supercluster = CLUS_TO_SUPC[SUBC_TO_CLUS[f"s{alias}"]]
            base[MARKER_OF_SUPC[supercluster]] += 150
            counts.append(base * generator.uniform(0.5, 2.0))
        return counts

    for matrix, group in meta.groupby("feature_matrix_label"):
        extra = other[other["feature_matrix_label"] == matrix]
        counts = cell_counts(group["cluster_alias"], rng) + cell_counts(
            extra["cluster_alias"], other_rng
        )
        ad.AnnData(
            X=sp.csr_matrix(np.round(np.asarray(counts)).astype(np.float32)),
            obs=pd.DataFrame(
                index=[*group["cell_label"].to_numpy(), *extra["cell_label"].to_numpy()]
            ),
            var=pd.DataFrame(index=genes),
        ).write_h5ad(h5ad_dir / f"{matrix}-raw.h5ad")
    return {"region_dir": region, "metadata": metadata, "h5ad_dir": h5ad_dir}


def ho_precompute(config: dict[str, Any]) -> None:
    import anndata as ad

    cells = pd.read_csv(config["cell_metadata_path"])
    genes = list(ad.read_h5ad(config["h5ad_path_list"][0], backed="r").var_names)
    counts = cells["cluster_alias"].map(lambda alias: f"s{alias}").value_counts()
    write_precompute(
        Path(config["output_path"]),
        subtree(WHB_TREE, set(counts.index)),
        genes,
        n_cells=counts.to_dict(),
    )


def ho_lookup(stub: list[str]) -> dict[str, Any]:
    return {
        "None": stub[:3],
        f"{SUPC}/{UL_IT}": stub[3:5],
        f"{SUPC}/{ASTRO}": [],
        f"{SUPC}/{MICRO}": stub[5:6],
    }


def marker_mapper(markers: Mapping[str, str]) -> Callable[..., pd.DataFrame]:
    """A hierarchical marker-count mapper standing in for MapMyCells."""
    from merxen.annotation.mapmycells_engine import N_RUNNERS_UP, runner_up_column

    def map_onto(engine: Any, query: Any) -> pd.DataFrame:
        tree = engine.tree()
        genes = list(query.genes)
        counts = query.counts.toarray()
        records = []
        for row, cell_id in enumerate(query.obs.index):
            parent: str | None = None
            for depth, level in enumerate(tree.hierarchy):
                children = list(
                    tree.nodes(level)
                    if depth == 0
                    else tree.children_of(tree.hierarchy[depth - 1], parent)
                )
                scores = np.array(
                    [
                        counts[row, genes.index(markers[child])] + 1.0
                        if child in markers and markers[child] in genes
                        else 1.0
                        for child in children
                    ]
                )
                share = scores / scores.sum()
                order = np.argsort(-share, kind="mergesort")
                best = children[order[0]]
                record: dict[str, Any] = {
                    "cell_id": cell_id,
                    "level": level,
                    "level_name": level.lower(),
                    "assignment": best,
                    "name": tree.name(level, best),
                    "bp": round(float(share[order[0]]), 2),
                    "aggregate_probability": round(float(share[order[0]]), 2),
                    "avg_correlation": 0.5,
                    "directly_assigned": True,
                    "n_runners_up": len(children) - 1,
                }
                for rank in range(1, N_RUNNERS_UP + 1):
                    if rank < len(children):
                        other = children[order[rank]]
                        record[runner_up_column(rank, "assignment")] = other
                        record[runner_up_column(rank, "name")] = tree.name(level, other)
                        record[runner_up_column(rank, "probability")] = round(
                            float(share[order[rank]]), 2
                        )
                        record[runner_up_column(rank, "correlation")] = 0.1
                    else:
                        record[runner_up_column(rank, "assignment")] = None
                        record[runner_up_column(rank, "name")] = None
                        record[runner_up_column(rank, "probability")] = np.nan
                        record[runner_up_column(rank, "correlation")] = np.nan
                records.append(record)
                parent = best
        return pd.DataFrame.from_records(records)

    return map_onto


def install_self_map(
    monkeypatch: pytest.MonkeyPatch, markers: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Replace the self-map's MapMyCells runs by ``marker_mapper``."""
    calls: list[dict[str, Any]] = []
    mapper = marker_mapper(markers)

    def factory(context: Any, engine: Any, runs: list[dict[str, Any]]) -> Any:
        def map_query(query: Any, tag: str, seed: int) -> pd.DataFrame:
            calls.append(
                {
                    "engine": engine.reference_id,
                    "engine_path": str(engine.path),
                    "tag": tag,
                    "seed": seed,
                    "n_cells": len(query.obs),
                }
            )
            runs.append({"tag": tag, "engine": engine.reference_id, "wall_s": 0.0})
            return mapper(engine, query)

        return map_query

    monkeypatch.setattr(reference, "_mmc_map_function", factory)
    return calls


def ho_spec_sources(sources: Mapping[str, Path]) -> dict[str, Path]:
    return {
        "whb_region_dir": sources["region_dir"],
        "whb_h5ad_dir": sources["h5ad_dir"],
        "whb_metadata_dir": sources["metadata"],
    }


def test_holdout_donor_auto_is_the_donor_with_the_fewest_cells() -> None:
    donors = pd.Series(["H18.30.002"] * 5 + ["H19.30.001"] * 4 + ["H19.30.002"] * 3)
    assert reference.holdout_donor(donors, "auto") == "H19.30.002"
    assert reference.holdout_donor(donors, "H18.30.002") == "H18.30.002"
    with pytest.raises(ReferenceBuildError, match="has no frontal"):
        reference.holdout_donor(donors, "H00.00.000")


def test_holdout_bundle_trains_without_the_donor_and_keeps_its_cells_as_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_ho_sources(tmp_path)
    fake = FakeCtm(ho_lookup).install(monkeypatch)
    trained: list[pd.DataFrame] = []

    def recording_precompute(config: dict[str, Any]) -> None:
        trained.append(pd.read_csv(config["cell_metadata_path"]))
        ho_precompute(config)

    fake.precompute = recording_precompute
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id=reference.HO_REFERENCE_ID,
            species="human",
            role="resolvability",
            hierarchy=[SUPC, CLUS],
            sources=ho_spec_sources(sources),
        )
    )
    assert set(spec.sources) == {
        "whb_region_cell_metadata",
        "whb_cell_metadata",
        "whb_neurons_h5ad",
        "whb_nonneurons_h5ad",
        "whb_cluster_annotation_term",
        "whb_cluster_membership",
    }
    panel = make_panel(GENES)
    # The version-6 test set (M3b); version 7 adds the class top-up (M3c).
    config = AnnotationConfig(
        species="human", resolvability={"n_test_cells": 20, "version": 6}
    )
    store = ReferenceStore(tmp_path / "store")
    bundle = store.get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    output = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]
    test_set = output["test_set"]
    assert test_set["holdout_donor"] == "H_small"
    assert test_set["donor_cells"] == HO_DONORS
    # Training: the two other donors, clusters with >= 5 cells (c5 has 3).
    assert test_set["training_clusters_dropped"] == ["c5"]
    (precompute,) = fake.calls["precompute"]
    (training,) = trained
    assert not training["cell_label"].str.startswith("H_small").any()
    assert 7 not in set(training["cluster_alias"])
    assert precompute["hierarchy"] == list(reference.WHB_SOURCE_HIERARCHY)
    (truncate,) = fake.calls["truncate"]
    assert truncate["new_hierarchy"] == [SUPC, CLUS]
    assert output["levels"] == [SUPC, CLUS]
    # Test cells: the held-out donor, capped at n_test_cells, all superclusters.
    test = res.load_test_cells(bundle_dir)
    # Water filling: one cap per supercluster (6 here) keeps the total <= 20.
    per_supercluster = test.obs[f"{res.TRUTH_PREFIX}{SUPC}"].value_counts()
    assert len(test.obs) <= 20 and test_set["n_test_cells_requested"] == 20
    assert set(per_supercluster) == {6}
    # The cap (6) is reached by the donor: no other-region top-up.
    assert set(test.obs["donor_label"]) == {"H_small"}
    assert set(test.obs[reference.TEST_SOURCE_COLUMN]) == {"holdout_donor"}
    assert test_set["other_region"]["n_cells"] == 0
    assert set(test.obs[f"{res.TRUTH_PREFIX}{SUPC}"]) == {UL_IT, ASTRO, MICRO}
    # Every truth supercluster here can be named by a call: none left out.
    exclusion = test_set["truth_exclusion"]
    assert exclusion["version"] == reference.HO_TRUTH_EXCLUSION_VERSION
    assert exclusion["region"] == "frontal_cortex"
    assert exclusion["excluded_superclusters"] == {}
    assert exclusion["n_pool_cells_before"] == test_set["n_pool_cells"]
    assert test.genes == GENES  # panel genes present in WHB
    assert set(test.obs[res.SPILL_GROUP_COLUMN]) == {
        "Neurons",
        "Astrocytes",
        "Microglia",
    }
    import anndata as ad

    raw = ad.read_h5ad(sources["h5ad_dir"] / "WHB-10Xv3-Neurons-raw.h5ad")
    first = test.obs.index[test.obs["feature_matrix_label"] == "WHB-10Xv3-Neurons"][0]
    expected = raw[first, GENES].X.toarray().ravel()
    got = test.counts[list(test.obs.index).index(first)].toarray().ravel()
    np.testing.assert_allclose(got, expected)
    # A mappable bundle: precompute, lookup and vocab snapshot.
    from merxen.annotation.mapmycells_engine import MmcBundle

    mmc = MmcBundle.from_dir(bundle_dir)
    assert mmc.levels == (SUPC, CLUS)
    assert (bundle_dir / res.TEST_CELLS_OBS_FILE).is_file()


def test_holdout_test_set_tops_up_nonneuronal_superclusters_from_other_regions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_ho_sources(tmp_path)
    fake = FakeCtm(ho_lookup).install(monkeypatch)
    trained: list[pd.DataFrame] = []

    def recording_precompute(config: dict[str, Any]) -> None:
        trained.append(pd.read_csv(config["cell_metadata_path"]))
        ho_precompute(config)

    fake.precompute = recording_precompute
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id=reference.HO_REFERENCE_ID,
            species="human",
            role="resolvability",
            hierarchy=[SUPC, CLUS],
            sources=ho_spec_sources(sources),
        )
    )
    config = AnnotationConfig(species="human", resolvability={"n_test_cells": 1000})
    bundle = ReferenceStore(tmp_path / "store").get_or_build(
        spec, make_panel(GENES), builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    test_set = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]["test_set"]
    test = res.load_test_cells(bundle_dir)
    source = test.obs[reference.TEST_SOURCE_COLUMN]
    other = test.obs[source == reference.TEST_SOURCE_OTHER_REGION]
    # Every held-out donor cell (cap 1,000) plus the eligible other-region
    # cells: non-neuronal nuclei of E2 dissections of clusters the held-out
    # training reference holds (the planted cluster c5 is never drawn).
    assert (source == reference.TEST_SOURCE_DONOR).sum() == HO_DONORS["H_small"]
    assert sorted({label.rsplit("-", 1)[0] for label in other.index}) == [
        "mtg_astro",
        "v1c_micro",
    ]
    assert len(other) == 7
    assert set(other["region_of_interest_label"]) <= set(
        reference.HO_OTHER_REGION_ROI_LABELS
    )
    # Disjoint from every reference cell: the frontal cells (training and
    # held-out donor) and the training cells the precompute was built from.
    frontal = pd.read_csv(sources["region_dir"] / reference.REGION_CELL_METADATA_FILE)
    (training,) = trained
    assert not set(other.index) & set(frontal["cell_label"])
    assert not set(other.index) & set(training["cell_label"])
    # Recorded in the bundle: dissections, counts per class, exclusions.
    record = test_set["other_region"]
    assert record["roi_labels"] == list(reference.HO_OTHER_REGION_ROI_LABELS)
    assert record["feature_matrix"] == "WHB-10Xv3-Nonneurons"
    assert record["version"] == reference.HO_OTHER_REGION_VERSION == 2
    assert record["n_cells"] == 7
    assert record["per_supercluster"] == {MICRO: 3, ASTRO: 4}
    assert record["per_broad_class"] == {"Astrocytes": 4, "Microglia": 3}
    assert record["per_region"] == {"Human MTG": 4, "Human V1C": 3}
    assert record["per_donor"] == {"H_big": 4, "H_other": 3}
    # The cluster rule and what it left out are recorded.
    assert record["cluster_rule"] == reference.HO_OTHER_REGION_RULE
    assert record["min_training_cells_per_cluster"] == (
        reference.HO_MIN_TRAINING_CELLS_PER_CLUSTER
    )
    assert record["n_training_clusters"] == test_set["n_training_clusters"]
    assert record["n_excluded_cluster_not_in_training"] == 2
    assert record["n_excluded_clusters"] == 1
    assert record["excluded_cluster_not_in_training_per_supercluster"] == {MICRO: 2}
    assert record["n_cluster_not_in_training"] == 0
    assert record["n_excluded_reference_cells"] == 0
    assert record["disjoint_from_reference_cells"] is True
    assert set(record["eligible_superclusters"]) == {ASTRO, MICRO}
    assert test_set["per_source"] == {"holdout_donor": 30, "other_region": 7}
    assert test_set["per_supercluster"][ASTRO] == 8 + 4
    # Their counts are the raw WHB counts of the panel genes.
    import anndata as ad

    raw = ad.read_h5ad(sources["h5ad_dir"] / "WHB-10Xv3-Nonneurons-raw.h5ad")
    label = other.index[0]
    np.testing.assert_allclose(
        test.counts[list(test.obs.index).index(label)].toarray().ravel(),
        raw[label, GENES].X.toarray().ravel(),
    )


def test_held_out_truths_no_call_can_name_are_excluded() -> None:
    # E2 (02_make_queries.py) kept neither the sinks and Mixed/Unknown
    # superclusters nor Amygdala excitatory (implausible in frontal cortex).
    labels = [
        "CS202210140_463",  # Miscellaneous (sink, Mixed/Unknown)
        "CS202210140_483",  # Splatter (sink, Mixed/Unknown)
        "CS202210140_478",  # Amygdala excitatory
        "CS202210140_471",  # Ependymal (Mixed/Unknown, not a sink)
        UL_IT,
        ASTRO,
        "CS202210140_468",  # COP: a floor class (OPC), plausible
    ]
    assert reference.ho_truth_exclusions(labels, "frontal_cortex") == {
        "CS202210140_463": reference.HO_EXCLUDED_SINK,
        "CS202210140_471": reference.HO_EXCLUDED_NO_FLOOR_CLASS,
        "CS202210140_478": reference.HO_EXCLUDED_REGION_IMPLAUSIBLE,
        "CS202210140_483": reference.HO_EXCLUDED_SINK,
    }
    assert reference.ho_truth_exclusions(["CS000_unknown"], "frontal_cortex") == {
        "CS000_unknown": reference.HO_EXCLUDED_NO_FLOOR_CLASS
    }


def test_holdout_test_set_leaves_out_excluded_truth_superclusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_ho_sources(tmp_path)
    FakeCtm(ho_lookup).install(monkeypatch).precompute = ho_precompute
    seen: list[tuple[set[str], str]] = []

    def microglia_as_sink(labels: Iterable[str], region: str) -> dict[str, str]:
        seen.append(({str(label) for label in labels}, region))
        return {MICRO: reference.HO_EXCLUDED_SINK}

    monkeypatch.setattr(reference, "ho_truth_exclusions", microglia_as_sink)
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id=reference.HO_REFERENCE_ID,
            species="human",
            role="resolvability",
            hierarchy=[SUPC, CLUS],
            sources=ho_spec_sources(sources),
        )
    )
    config = AnnotationConfig(species="human", resolvability={"n_test_cells": 1000})
    bundle = ReferenceStore(tmp_path / "store").get_or_build(
        spec, make_panel(GENES), builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    test_set = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]["test_set"]
    test = res.load_test_cells(bundle_dir)
    donor = test.obs[test.obs[reference.TEST_SOURCE_COLUMN] == "holdout_donor"]
    # The pool's truths are checked; the excluded ones never become tests.
    assert seen == [({UL_IT, ASTRO, MICRO}, "frontal_cortex")]
    assert MICRO not in set(donor[f"{res.TRUTH_PREFIX}{SUPC}"])
    exclusion = test_set["truth_exclusion"]
    n_micro = exclusion["excluded_superclusters"][MICRO]["n_pool_cells"]
    assert n_micro > 0
    assert exclusion["excluded_superclusters"][MICRO]["reason"] == "sink"
    assert exclusion["excluded_superclusters"][MICRO]["name"] == "Microglia"
    assert exclusion["n_excluded_pool_cells"] == n_micro
    assert test_set["n_pool_cells"] == exclusion["n_pool_cells_before"] - n_micro
    assert len(donor) == HO_DONORS["H_small"] - n_micro


def test_other_region_draw_never_takes_a_reference_cell() -> None:
    labels = pd.DataFrame(
        {SUPC: [ASTRO, MICRO], CLUS: ["c3", "c4"], SUBC: ["s4", "s6"]},
        index=pd.Index([4, 6], name="cluster_alias"),
    )
    metadata = pd.DataFrame(
        {
            "cell_label": [f"o-{index}" for index in range(8)],
            "feature_matrix_label": "WHB-10Xv3-Nonneurons",
            "donor_label": "H_x",
            "cluster_alias": [4, 4, 4, 4, 6, 6, 6, 6],
            "region_of_interest_label": "Human MTG",
        }
    ).join(labels, on="cluster_alias")
    # A candidate listed as a reference cell (a planted overlap) is dropped.
    rows, record = reference.other_region_test_cells(
        metadata,
        reference_cells={"o-0", "frontal-1"},
        training_superclusters=[ASTRO, MICRO, UL_IT],
        training_clusters=["c3", "c4"],
        have={ASTRO: 1},
        cap=3,
        room=None,
    )
    assert record["n_excluded_cluster_not_in_training"] == 0
    assert "o-0" not in set(rows["cell_label"])
    assert record["n_excluded_reference_cells"] == 1
    assert record["disjoint_from_reference_cells"] is True
    # Capped per supercluster like the donor cells: Astro 1 + 2, Micro 0 + 3.
    assert record["per_supercluster"] == {MICRO: 3, ASTRO: 2}
    assert record["candidates_per_supercluster"] == {MICRO: 4, ASTRO: 3}
    # Reproducible draw.
    again, _ = reference.other_region_test_cells(
        metadata,
        reference_cells={"o-0"},
        training_superclusters=[ASTRO, MICRO, UL_IT],
        training_clusters=["c3", "c4"],
        have={ASTRO: 1},
        cap=3,
        room=None,
    )
    assert again["cell_label"].tolist() == rows["cell_label"].tolist()


def test_other_region_draw_takes_only_clusters_of_the_training_reference() -> None:
    # User decision 2026-09-27 (HO_OTHER_REGION_VERSION 2): a candidate of a
    # cluster the held-out training reference lacks is never drawn, even with
    # room to spare, and is counted.
    labels = pd.DataFrame(
        {
            SUPC: [ASTRO, MICRO, MICRO],
            CLUS: ["c3", "c4", "c_unseen"],
            SUBC: ["s4", "s6", "s7"],
        },
        index=pd.Index([4, 6, 9], name="cluster_alias"),
    )
    metadata = pd.DataFrame(
        {
            "cell_label": [f"o-{index}" for index in range(7)] + ["planted"],
            "feature_matrix_label": "WHB-10Xv3-Nonneurons",
            "donor_label": "H_x",
            "cluster_alias": [4, 4, 4, 6, 6, 9, 9, 9],
            "region_of_interest_label": "Human MTG",
        }
    ).join(labels, on="cluster_alias")
    for seed in range(5):
        rows, record = reference.other_region_test_cells(
            metadata,
            reference_cells=set(),
            training_superclusters=[ASTRO, MICRO],
            training_clusters=["c3", "c4"],
            have={},
            cap=10,
            room=None,
            seed=seed,
        )
        drawn = set(rows["cell_label"])
        assert "planted" not in drawn and not drawn & {"o-5", "o-6"}
        assert drawn == {"o-0", "o-1", "o-2", "o-3", "o-4"}
    assert record["n_excluded_cluster_not_in_training"] == 3
    assert record["n_excluded_clusters"] == 1
    assert record["excluded_cluster_not_in_training_per_supercluster"] == {MICRO: 3}
    assert record["n_cluster_not_in_training"] == 0
    assert record["candidates_per_supercluster"] == {MICRO: 2, ASTRO: 3}
    assert record["n_training_clusters"] == 2
    assert record["cluster_rule"] == reference.HO_OTHER_REGION_RULE
    # A supercluster whose candidates are all of unseen clusters gains none.
    rows, record = reference.other_region_test_cells(
        metadata,
        reference_cells=set(),
        training_superclusters=[ASTRO, MICRO],
        training_clusters=["c3"],
        have={},
        cap=10,
        room=None,
    )
    assert set(rows[SUPC]) == {ASTRO}
    assert record["n_excluded_cluster_not_in_training"] == 5
    assert record["excluded_cluster_not_in_training_per_supercluster"] == {MICRO: 5}


def whb_resolvability_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Path], dict[str, Path], FakeCtm]:
    sources = write_whb_sources(tmp_path)
    ho = write_ho_sources(tmp_path)
    fake = FakeCtm(whb_truncated_lookup).install(monkeypatch)
    fake.precompute = ho_precompute
    return sources, ho, fake


def test_whb_primary_self_maps_the_held_out_cells_onto_the_held_out_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources, ho, fake = whb_resolvability_setup(tmp_path, monkeypatch)
    markers = {node: GENES[index] for node, index in MARKER_OF_SUPC.items()}
    calls = install_self_map(monkeypatch, markers)
    spec = prepare_reference_spec(
        whb_spec(
            region_precompute=sources["region_dir"],
            seaad_precomputed_stats=sources["seaad"],
            whb_h5ad_dir=ho["h5ad_dir"],
            whb_metadata_dir=ho["metadata"],
            whb_region_cell_metadata=ho["region_dir"]
            / reference.REGION_CELL_METADATA_FILE,
        )
    )
    panel = make_panel(GENES)
    config = AnnotationConfig(species="human", resolvability={"version": 6})
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    bundle = store.get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    output = manifest["builder_output"]["resolvability"]
    # The held-out bundle was built in the same store and mapped onto.
    (ho_dir,) = (tmp_path / "store" / reference.HO_REFERENCE_ID).glob("[0-9a-f]*")
    assert output["test_set_bundle"]["build_hash"] == ho_dir.name
    assert {call["engine"] for call in calls} == {reference.HO_REFERENCE_ID}
    assert [call["tag"] for call in calls] == ["R1_contam_HO", "clean"]
    for name in (
        res.RESOLVABILITY_FILE,
        res.RESOLVABILITY_CELLS_FILE,
        res.RESOLVABILITY_SUMMARY_FILE,
    ):
        assert (bundle_dir / name).is_file()
    summary = json.loads((bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    assert summary["engine"]["reference_id"] == reference.HO_REFERENCE_ID
    assert summary["engine"]["self"] is False
    assert summary["depth_grid"] == [10, 15, 30, 60, 120, 250]
    assert [level["level"] for level in summary["levels"]] == [
        "lineage",
        "broad",
        "nt",
        "supercluster",
        "cluster",
    ]
    cells = pd.read_parquet(bundle_dir / res.RESOLVABILITY_CELLS_FILE)
    # Only cells whose native counts reach a depth are simulated at it.
    test = res.load_test_cells(ho_dir)
    native = test.native_counts
    for depth in (30, 250):
        simulated = cells[(cells["depth"] == depth) & (cells["level"] == "broad")]
        simulated = simulated[simulated["recipe"] == "R1_contam_HO"]
        assert set(simulated["cell_id"]) == set(test.obs.index[native >= depth])
    assert manifest["recorded_settings"]["resolvability"]["outputs_in_bundle"]
    payload = manifest["build_hash_payload"]["builder_params"]["resolvability"]
    assert payload["enabled"] is True
    assert payload["test_set"]["reference_id"] == reference.HO_REFERENCE_ID
    assert payload["recipes"][0]["spill_fraction"] == 0.25
    # SEA-AD maps the same held-out cells onto itself, reusing the test set.
    seaad_spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id="seaad_mr_panel",
            species="human",
            role="secondary",
            sources={
                "seaad_precomputed_stats": sources["seaad"],
                **ho_spec_sources(ho),
            },
        )
    )
    sea_markers = {label: GENES[index] for label, index in SEA_MARKERS.items()}
    sea_calls = install_self_map(monkeypatch, sea_markers)
    n_precompute = len(fake.calls["precompute"])
    sea = store.get_or_build(
        seaad_spec, panel, builder=builder_for(seaad_spec, config), config=config
    )
    assert len(fake.calls["precompute"]) == n_precompute  # held-out bundle reused
    assert {call["engine"] for call in sea_calls} == {"seaad_mr_panel"}
    sea_summary = json.loads(
        (Path(sea.path) / res.RESOLVABILITY_SUMMARY_FILE).read_text()
    )
    assert sea_summary["engine"]["self"] is True
    assert [level["level"] for level in sea_summary["levels"]] == ["broad"]
    assert sea_summary["test_set_bundle"]["build_hash"] == ho_dir.name


COP = "CS202210140_468"  # Committed oligodendrocyte precursor
OPC_SUPC = "CS202210140_467"  # Oligodendrocyte precursor


def _held_out_cells(rows: Sequence[tuple[str, str, str]]) -> res.HeldOutCells:
    """Held-out test cells from (cell id, truth supercluster, test source)."""
    from scipy import sparse

    ids = [row[0] for row in rows]
    truth = [row[1] for row in rows]
    obs = pd.DataFrame(
        {
            f"{res.TRUTH_PREFIX}{SUPC}": truth,
            res.TRUTH_LEAF_COLUMN: truth,
            res.SPILL_GROUP_COLUMN: ["g"] * len(rows),
            reference.TEST_SOURCE_COLUMN: [row[2] for row in rows],
        },
        index=pd.Index(ids, name="cell_id"),
    )
    counts = sparse.csr_matrix(
        np.arange(1, 2 * len(rows) + 1, dtype=float).reshape(-1, 2)
    )
    return res.HeldOutCells(counts=counts, genes=["g1", "g2"], obs=obs)


def test_self_map_test_cells_leave_out_only_other_region_cop_cells() -> None:
    donor, other = reference.TEST_SOURCE_DONOR, reference.TEST_SOURCE_OTHER_REGION
    test = _held_out_cells(
        [
            ("donor_cop", COP, donor),
            ("other_cop_1", COP, other),
            ("donor_opc", OPC_SUPC, donor),
            ("other_opc", OPC_SUPC, other),
            ("other_cop_2", COP, other),
            ("other_astro", ASTRO, other),
        ]
    )
    kept, record = reference.self_map_test_cells(test, reference.HO_REFERENCE_ID)
    # M8 D1: the other-region COP cells go; the donor's own COP cells and the
    # other-region cells of every other supercluster stay.
    assert list(kept.obs.index) == [
        "donor_cop",
        "donor_opc",
        "other_opc",
        "other_astro",
    ]
    np.testing.assert_array_equal(
        kept.counts.toarray(), test.counts.toarray()[[0, 2, 3, 5]]
    )
    np.testing.assert_array_equal(kept.native_counts, test.native_counts[[0, 2, 3, 5]])
    assert record is not None
    assert record["revision"] == reference.HO_SELF_MAP_TEST_SET_REVISION == 1
    assert record["rule"] == reference.HO_SELF_MAP_EXCLUSION_RULE
    assert record["test_source"] == other
    assert record["excluded_superclusters"] == ["Committed oligodendrocyte precursor"]
    assert record["excluded_labels"] == [COP]
    assert record["n_test_cells_in_bundle"] == 6
    assert record["n_excluded"] == 2
    assert record["n_test_cells"] == 4
    assert record["excluded_per_supercluster"] == {COP: 2}
    assert record["kept_per_source_of_excluded_superclusters"] == {donor: 1}
    # The input is untouched (the held-out bundle's cells stay as built).
    assert len(test.obs) == 6


def test_self_map_test_cells_keep_other_test_sets_and_donor_only_sets() -> None:
    donor, other = reference.TEST_SOURCE_DONOR, reference.TEST_SOURCE_OTHER_REGION
    test = _held_out_cells([("a", COP, other), ("b", ASTRO, donor)])
    # The mouse test set has no exclusion (and no record).
    same, record = reference.self_map_test_cells(
        test, reference.WMB_TESTSET_REFERENCE_ID
    )
    assert same is test and record is None
    # A held-out set without other-region COP cells is returned as it is.
    donor_only = _held_out_cells([("a", COP, donor), ("b", ASTRO, other)])
    same, record = reference.self_map_test_cells(donor_only, reference.HO_REFERENCE_ID)
    assert same is donor_only
    assert record is not None and record["n_excluded"] == 0
    # A held-out set built before the top-up has no test_source: all donor.
    legacy = _held_out_cells([("a", COP, donor)])
    legacy.obs.drop(columns=[reference.TEST_SOURCE_COLUMN], inplace=True)
    same, record = reference.self_map_test_cells(legacy, reference.HO_REFERENCE_ID)
    assert same is legacy and record is not None and record["n_excluded"] == 0


def test_whb_self_map_leaves_the_excluded_other_region_cells_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    # The fixture's other-region cells are Astrocytes and Microglia (no COP),
    # so exclude Astrocyte to see the rule act end to end.
    monkeypatch.setattr(reference, "HO_SELF_MAP_EXCLUDED_SUPERCLUSTERS", ("Astrocyte",))
    sources, ho, _ = whb_resolvability_setup(tmp_path, monkeypatch)
    markers = {node: GENES[index] for node, index in MARKER_OF_SUPC.items()}
    calls = install_self_map(monkeypatch, markers)
    spec = prepare_reference_spec(
        whb_spec(
            region_precompute=sources["region_dir"],
            seaad_precomputed_stats=sources["seaad"],
            whb_h5ad_dir=ho["h5ad_dir"],
            whb_metadata_dir=ho["metadata"],
            whb_region_cell_metadata=ho["region_dir"]
            / reference.REGION_CELL_METADATA_FILE,
        )
    )
    config = AnnotationConfig(species="human")
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    bundle = store.get_or_build(
        spec, make_panel(GENES), builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    (ho_dir,) = (tmp_path / "store" / reference.HO_REFERENCE_ID).glob("[0-9a-f]*")
    test = res.load_test_cells(ho_dir)
    source = test.obs[reference.TEST_SOURCE_COLUMN]
    truth = test.obs[res.TRUTH_LEAF_COLUMN]
    dropped = set(
        test.obs.index[
            (source == reference.TEST_SOURCE_OTHER_REGION) & (truth == ASTRO)
        ]
    )
    assert len(dropped) == 4  # the held-out bundle keeps them
    cells = pd.read_parquet(bundle_dir / res.RESOLVABILITY_CELLS_FILE)
    simulated = set(cells["cell_id"].astype(str))
    assert not simulated & dropped
    # The donor's Astrocytes and the other-region Microglia are simulated.
    assert (
        set(test.obs.index[(truth == ASTRO) & (source == "holdout_donor")]) <= simulated
    )
    assert (
        set(test.obs.index[(truth == MICRO) & (source != "holdout_donor")]) <= simulated
    )
    # Every mapped query holds only kept cells.
    assert all(call["n_cells"] > 0 for call in calls)
    record = {
        **reference.ho_self_map_exclusion_params(),
        "excluded_labels": [ASTRO],
    }
    summary = json.loads((bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    recorded = summary["test_set_exclusion"]
    assert {key: recorded[key] for key in record} == record
    assert recorded["n_excluded"] == 4
    assert recorded["n_test_cells"] == len(test.obs) - 4
    assert recorded["excluded_per_supercluster"] == {ASTRO: 4}
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    output = manifest["builder_output"]["resolvability"]
    assert output["test_set_exclusion"]["n_excluded"] == 4
    payload = manifest["build_hash_payload"]["builder_params"]["resolvability"]
    assert payload["test_set"]["self_map_exclusion"] == (
        reference.ho_self_map_exclusion_params()
    )


def test_self_map_test_set_revision_changes_only_the_self_map_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AnnotationConfig(species="human")
    spec = whb_spec(region_precompute=tmp_path)
    seaad = AnnotationReferenceSpec(
        reference_id="seaad_mr_panel", species="human", role="secondary"
    )
    ho_spec = AnnotationReferenceSpec(
        reference_id=reference.HO_REFERENCE_ID,
        species="human",
        role="resolvability",
        hierarchy=[SUPC, CLUS],
    )
    wmb = AnnotationReferenceSpec(
        reference_id="wmb_panel", species="mouse", role="primary"
    )
    mouse = AnnotationConfig(species="mouse")

    def params() -> tuple[Any, ...]:
        return (
            builder_for(spec, config).params,
            builder_for(seaad, config).params,
            builder_for(ho_spec, config).params,
            builder_for(wmb, mouse).params,
        )

    before = params()
    hashed = before[0]["resolvability"]["test_set"]["self_map_exclusion"]
    assert hashed["revision"] == reference.HO_SELF_MAP_TEST_SET_REVISION
    assert hashed["excluded_superclusters"] == ["Committed oligodendrocyte precursor"]
    monkeypatch.setattr(reference, "HO_SELF_MAP_TEST_SET_REVISION", 0)
    after = params()
    # The WHB and SEA-AD self-maps get new hashes; the held-out bundle (its
    # test cells are unchanged) and the mouse bundles keep theirs.
    assert after[0]["resolvability"] != before[0]["resolvability"]
    assert after[1]["resolvability"] != before[1]["resolvability"]
    assert after[2] == before[2]
    assert after[3] == before[3]
    assert "self_map_exclusion" not in before[3]["resolvability"]["test_set"]
    # The resolvability algorithm version is not bumped (7 is M3c's).
    assert before[0]["resolvability"]["resolvability_version"] == 6


def test_self_map_settings_that_change_its_output_change_the_build_hash(
    tmp_path: Path,
) -> None:
    spec = whb_spec(region_precompute=tmp_path)
    base = builder_for(spec, AnnotationConfig(species="human")).params
    spill = builder_for(
        spec, AnnotationConfig(species="human", resolvability={"spill_fraction": 0.3})
    ).params
    knob = builder_for(
        spec, AnnotationConfig(species="human", resolvability={"min_coverage": 0.4})
    ).params
    off = builder_for(spec, without_resolvability("human")).params
    assert spill["resolvability"] != base["resolvability"]
    assert knob["resolvability"] == base["resolvability"]  # a RESOLVE-time knob
    assert off["resolvability"] == {"enabled": False}
    # The simulation logic and the other-region rule are hashed by version.
    hashed = base["resolvability"]
    assert hashed["resolvability_version"] == res.RESOLVABILITY_VERSION
    assert hashed["test_set"]["other_region"]["version"] == (
        reference.HO_OTHER_REGION_VERSION
    )


def test_other_region_rule_version_changes_the_held_out_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = AnnotationConfig(species="human")
    spec = whb_spec(region_precompute=tmp_path)
    ho_spec = AnnotationReferenceSpec(
        reference_id=reference.HO_REFERENCE_ID,
        species="human",
        role="resolvability",
        hierarchy=[SUPC, CLUS],
    )
    before = (builder_for(spec, config).params, builder_for(ho_spec, config).params)
    monkeypatch.setattr(reference, "HO_OTHER_REGION_VERSION", 1)
    after = (builder_for(spec, config).params, builder_for(ho_spec, config).params)
    # The primary's self-map and the held-out test-set bundle both change.
    assert after[0]["resolvability"] != before[0]["resolvability"]
    assert after[1] != before[1]


def test_resolvability_needs_the_test_set_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_whb_sources(tmp_path)
    ctm = FakeCtm(whb_truncated_lookup).install(monkeypatch)
    spec = prepare_reference_spec(
        whb_spec(
            region_precompute=sources["region_dir"],
            seaad_precomputed_stats=sources["seaad"],
        )
    )
    config = AnnotationConfig(species="human")
    with pytest.raises(ReferenceBuildError, match="test-set source"):
        ReferenceStore(tmp_path / "store").get_or_build(
            spec, make_panel(GENES), builder=builder_for(spec, config), config=config
        )
    # It fails before the marker steps, not after them.
    assert ctm.calls["reference"] == [] and ctm.calls["query"] == []
    assert ctm.calls["truncate"] == []


def test_prepare_keeps_the_held_out_sources_only_while_resolvability_is_on(
    tmp_path: Path,
) -> None:
    ho = write_ho_sources(tmp_path)
    sources = write_whb_sources(tmp_path)
    (sources["region_dir"] / reference.REGION_CELL_METADATA_FILE).write_text(
        (ho["region_dir"] / reference.REGION_CELL_METADATA_FILE).read_text()
    )
    spec = whb_spec(
        region_precompute=sources["region_dir"],
        seaad_precomputed_stats=sources["seaad"],
        whb_h5ad_dir=ho["h5ad_dir"],
        whb_metadata_dir=ho["metadata"],
    )
    kept = prepare_reference_spec(spec)
    assert {
        "whb_region_cell_metadata",
        "whb_neurons_h5ad",
        "whb_nonneurons_h5ad",
        "whb_cluster_membership",
    } <= set(kept.sources)
    dropped = prepare_reference_spec(spec, SourceOptions(resolvability=False))
    assert not {"whb_neurons_h5ad", "whb_region_cell_metadata"} & set(dropped.sources)
    assert "whb_h5ad_dir" not in dropped.sources


def write_wmb_testset_sources(tmp_path: Path) -> dict[str, Path]:
    """The tiny WMB inputs with a synthetic mapping precompute (no ctm)."""
    sources = write_real_wmb_inputs(tmp_path)
    # Self-map test cells of both classes: cluster 9 and part of 1 and 3.
    meta = pd.read_csv(sources["metadata"] / "cell_metadata.csv")
    picked = pd.concat(
        [
            meta[meta.cluster_alias == 9],
            meta[meta.cluster_alias == 1].head(10),
            meta[meta.cluster_alias == 3].head(10),
        ]
    )
    picked[["cell_label"]].assign(subclass="x").to_csv(sources["truth"], index=False)
    genes = json.loads(str(sources["genes"]))
    tree: dict[str, Any] = {"hierarchy": [CLAS, SUBC_W, SUPT, CLUS_W]}
    tree[CLAS] = {label: subs for label, (_, subs) in REAL_CLASSES.items()}
    tree[SUBC_W] = {
        subclass: [subclass.replace("SUBC", "SUPT")]
        for subclass in REAL_SUBCLASS_CLUSTERS
    }
    tree[SUPT] = {
        subclass.replace("SUBC", "SUPT"): [
            f"CS20230722_CLUS_{alias:04d}" for alias in aliases
        ]
        for subclass, aliases in REAL_SUBCLASS_CLUSTERS.items()
    }
    tree[CLUS_W] = {f"CS20230722_CLUS_{alias:04d}": [] for alias in range(1, 10)}
    tree["name_mapper"] = {
        CLAS: {label: {"name": name} for label, (name, _) in REAL_CLASSES.items()}
    }
    mapping = write_precompute(
        tmp_path / "allen" / "precomputed_stats_ABC_revision_230821.h5",
        tree,
        genes,
        n_cells={leaf: 25 for leaf in tree[CLUS_W]},
        with_detection=False,
    )
    return {**sources, "mapping": mapping, "tree": Path(json.dumps(tree))}


def test_wmb_testset_adds_non_neuronal_cells_the_marker_build_did_not_use(
    tmp_path: Path,
) -> None:
    sources = write_wmb_testset_sources(tmp_path)
    genes = json.loads(str(sources["genes"]))
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id=reference.WMB_TESTSET_REFERENCE_ID,
            species="mouse",
            role="resolvability",
            max_cells_per_cluster=20,
            sources={
                "wmb_h5ad_dir": sources["h5ad_dir"],
                "wmb_metadata_dir": sources["metadata"],
                "wmb_selfmap_test_cells": sources["truth"],
            },
        )
    )
    config = AnnotationConfig(
        species="mouse", resolvability={"mouse_nonneuronal_extra_per_supertype": 2}
    )
    panel = make_panel(genes[:30], species="mouse")
    bundle = ReferenceStore(tmp_path / "store").get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    test = res.load_test_cells(bundle_dir)
    output = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())[
        "builder_output"
    ]["test_set"]
    base = set(pd.read_csv(sources["truth"])["cell_label"])
    assert base <= set(test.obs.index)
    extra = test.obs[test.obs["test_source"] == "nonneuronal_extra"]
    # Two per non-neuronal supertype, never a marker-training cell.
    assert set(extra[f"{res.TRUTH_PREFIX}{CLAS}"]) == {"CS20230722_CLAS_30"}
    assert extra.groupby(f"{res.TRUTH_PREFIX}{SUPT}").size().max() <= 2
    sampled, _ = sample_wmb_training_cells(
        sources["metadata"] / "cell_metadata.csv",
        {"WMB-10Xv3-AAA": Path(), "WMB-10Xv3-BBB": Path()},
        max_cells_per_cluster=20,
        exclude_cells=base,
    )
    assert not set(extra.index) & set(sampled["cell_label"])
    assert output["marker_training_sample"]["disjoint"] is True
    assert output["n_nonneuronal_extra"] == len(extra)
    assert set(test.obs[res.SPILL_GROUP_COLUMN]) == {
        "Neurons",
        "Astrocytes/Ependymal",
    }
    assert test.genes == sorted(genes[:30])


def test_wmb_primary_self_maps_its_test_set_onto_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    sources = write_wmb_testset_sources(tmp_path)
    genes = json.loads(str(sources["genes"]))
    tree = json.loads(str(sources["tree"]))

    def marker_precompute(config: dict[str, Any]) -> None:
        import anndata as ad

        cells = pd.read_csv(config["cell_metadata_path"])
        h5ad_genes = list(
            ad.read_h5ad(config["h5ad_path_list"][0], backed="r").var_names
        )
        counts = (
            cells["cluster_alias"]
            .map(lambda a: f"CS20230722_CLUS_{a:04d}")
            .value_counts()
        )
        write_precompute(
            Path(config["output_path"]),
            subtree(tree, set(counts.index)),
            h5ad_genes,
            n_cells=counts.to_dict(),
        )

    def lookup(stub: list[str]) -> dict[str, Any]:
        return {"None": stub[:4], f"{CLAS}/CS20230722_CLAS_01": stub[4:6]}

    fake = FakeCtm(lookup).install(monkeypatch)
    fake.precompute = marker_precompute
    markers = {"CS20230722_CLAS_01": genes[20], "CS20230722_CLAS_30": genes[21]}
    for index, subclass in enumerate(REAL_SUBCLASS_CLUSTERS):
        markers[subclass] = genes[10 + index]
    calls = install_self_map(monkeypatch, markers)
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
    config = AnnotationConfig(species="mouse")
    panel = make_panel(genes[:30], species="mouse")
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    bundle = store.get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    summary = json.loads((bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    assert summary["engine"]["self"] is True
    assert {call["engine"] for call in calls} == {"wmb_panel"}
    assert summary["depth_grid"] == [10, 20, 50, 100, 250, 500, 1000, 2000]
    levels = [level["level"] for level in summary["levels"]]
    assert levels == ["broad", "class", "nt", "subclass", "supertype"]
    assert (tmp_path / "store" / reference.WMB_TESTSET_REFERENCE_ID).is_dir()
    cells = pd.read_parquet(bundle_dir / res.RESOLVABILITY_CELLS_FILE)
    assert set(cells["parent"].dropna()) <= {"01 IT-ET Glut", "30 Astro-Epen"}
    # Class keys are WMB class names; D_max stops where test cells run out.
    d_max = summary["d_max"]["class"]
    assert set(d_max) <= {"01 IT-ET Glut", "30 Astro-Epen"}
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    assert manifest["builder_output"]["resolvability"]["n_test_cells"] >= 25


def test_mmc_map_function_runs_mapmycells_with_the_production_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    from merxen.annotation import mapmycells_engine

    recorded: dict[str, Any] = {}

    def fake_run_mmc(query_h5ad: Path, bundle: Any, params: Any, **kwargs: Any) -> Any:
        recorded.update(params=params, kwargs=kwargs, query=Path(query_h5ad))
        tidy = pd.DataFrame({column: [] for column in mapmycells_engine.TIDY_COLUMNS})
        output = Path(kwargs["output_parquet"])
        mapmycells_engine.write_tidy_parquet(tidy, output, {})

        class Result:
            parquet = output
            n_cells = 2
            n_query_genes = 3
            wall_s = 1.5
            peak_rss_gb = 0.1

        return Result()

    monkeypatch.setattr(mapmycells_engine, "run_mmc", fake_run_mmc)
    monkeypatch.setattr(
        mapmycells_engine,
        "restrict_lookup",
        lambda bundle, genes, output: mapmycells_engine.RestrictedLookup(
            path=None, validation=None, lookup_sha256=None
        ),
    )

    class Context:
        spec = whb_spec().model_copy(update={"rng_seed": 3})
        config = AnnotationConfig(species="human")
        scratch_dir = tmp_path / "scratch"
        work_dir = tmp_path / "work"

    class Engine:
        reference_id = "whb_frontal_supc_clus_ho"
        build_hash = "abc"

    runs: list[dict[str, Any]] = []
    map_query = reference._mmc_map_function(Context(), Engine(), runs)

    class Query:
        counts = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]])
        obs = pd.DataFrame(index=["a|D10", "b|D10"])
        genes = GENES[:3]

    map_query(Query(), "R1_contam_HO", 1)
    params = recorded["params"]
    assert params.rng_seed == 4 and params.bootstrap_factor == 0.5
    assert params.bootstrap_iteration == 100 and params.n_processors == 2
    assert recorded["kwargs"]["expected_ctm_version"] == "1.7.2"
    assert runs[0]["rng_seed"] == 4 and runs[0]["wall_s"] == 1.5
    assert not recorded["query"].exists()  # the query is removed after mapping


@pytest.mark.slow
def test_tiny_real_wmb_bundle_self_maps_through_real_mapmycells(
    tmp_path: Path, small_resources: Any
) -> None:
    pytest.importorskip("cell_type_mapper")
    sources = write_real_wmb_sources(tmp_path)
    meta = pd.read_csv(sources["metadata"] / "cell_metadata.csv")
    picked = pd.concat(
        [meta[meta.cluster_alias == alias].head(8) for alias in (1, 3, 5, 7, 9)]
    )
    picked[["cell_label"]].assign(subclass="x").to_csv(sources["truth"], index=False)
    genes = json.loads(str(sources["genes"]))
    spec = prepare_reference_spec(
        AnnotationReferenceSpec(
            reference_id="wmb_panel",
            species="mouse",
            role="primary",
            drop_level=SUPT,
            max_cells_per_cluster=10,
            bootstrap_iteration=20,
            depth_grid=[10, 50, 200],
            sources={
                "wmb_h5ad_dir": sources["h5ad_dir"],
                "wmb_metadata_dir": sources["metadata"],
                "wmb_mapping_stats": sources["mapping"],
                "wmb_selfmap_test_cells": sources["truth"],
            },
        )
    )
    # The version-6 path (R1 + clean) through real MapMyCells; a tiny
    # unlisted family would otherwise get version 7's ensemble (M3c).
    config = AnnotationConfig(
        species="mouse",
        resolvability={"min_cells_per_bin": 5, "min_confident_n": 5, "version": 6},
    )
    panel = make_panel(genes[:30], species="mouse")
    store = ReferenceStore(tmp_path / "store", scratch_root=tmp_path / "scratch")
    (tmp_path / "scratch").mkdir()
    bundle = store.get_or_build(
        spec, panel, builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    summary = json.loads((bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    runs = summary["mapping_runs"]
    assert [run["tag"] for run in runs] == ["R1_contam_HO", "clean"]
    assert all(run["engine"] == "wmb_panel" for run in runs)
    tables = res.load_resolvability(bundle_dir)
    assert tables is not None
    cells = tables.cells
    assert set(cells["level"]) == {"broad", "class", "nt", "subclass", "supertype"}
    class_rows = cells[(cells["level"] == "class") & (cells["recipe"] == "clean")]
    # The planted class markers make deep clean class calls correct.
    deep = class_rows[class_rows["depth"] == 200]
    assert len(deep) and deep["correct"].mean() > 0.9
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    assert manifest["builder_output"]["resolvability"]["n_test_cells"] == len(
        res.load_test_cells(
            tmp_path
            / "store"
            / reference.WMB_TESTSET_REFERENCE_ID
            / summary["test_set_bundle"]["build_hash"]
        ).obs
    )


def test_large_panels_build_unfiltered_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_resources: Any
) -> None:
    fake = FakeCtm(whb_truncated_lookup).install(monkeypatch)
    sources = write_whb_sources(tmp_path)
    config = large_config("human", limit=5, cap=6)
    spec = prepare_reference_spec(
        whb_spec(
            region_precompute=sources["region_dir"],
            seaad_precomputed_stats=sources["seaad"],
        )
    )
    (tmp_path / "scratch").mkdir()
    store = ReferenceStore(
        tmp_path / "ssd",
        large_root=tmp_path / "large",
        large_panel_genes=5,
        scratch_root=tmp_path / "scratch",
    )
    bundle = store.get_or_build(
        spec, make_panel(GENES), builder=builder_for(spec, config), config=config
    )
    bundle_dir = Path(bundle.path)
    # Every panel gene is a candidate; the reference markers are kept.
    assert fake.stub_genes == GENES
    assert not (bundle_dir / reference.MARKER_PREFILTER_FILE).exists()
    assert (bundle_dir / reference.REFERENCE_MARKERS_DIR).is_dir()
    manifest = json.loads((bundle_dir / BUNDLE_MANIFEST_NAME).read_text())
    assert manifest["build_hash_payload"]["large_panel_prefilter"] is None
    assert manifest["builder_output"]["markers"]["n_candidate_genes"] == 10
