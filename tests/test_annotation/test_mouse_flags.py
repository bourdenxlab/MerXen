"""Tests for the mouse spill-over and region-coherence flags (plan §7.4, §8.6; M6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation.config import AnnotationFlagsConfig, MouseRegionConfig
from merxen.annotation.mouse_flags import (
    IMMEDIATE_EARLY_GENES,
    PURITY_BASIS_DERIVED,
    PURITY_BASIS_MG_LOOSE,
    REASON_ASTRO_FPR,
    REASON_INSUFFICIENT_GENES,
    REASON_NO_COORDINATES,
    MouseFlagProfiles,
    astro_purity_genes,
    astrocyte_genes,
    floored_profiles,
    marker_astrocytes,
    microglia_genes,
    microglial_spillover,
    region_coherence,
    region_incoherent,
    specific_genes,
    spillover_reference,
    spillover_statistic,
)
from merxen.annotation.vocab import load_region_restricted_classes

# Ten panel genes: 0-4 microglial (0-2 are E3's MG_loose genes), 5-6
# astrocytic, 7-9 neuronal / shared.
GENES = [f"ENSMUSG{index:011d}" for index in range(10)]
SYMBOLS = [
    "Cx3cr1",
    "Csf1r",
    "C1qa",
    "Aif1",
    "P2ry12",
    "Aqp4",
    "Gfap",
    "Slc17a7",
    "Snap25",
    "Fos",
]
MICROGLIA_GENES = [0, 1, 2, 3, 4]
FREE_FLAG_GENES = [3, 4]  # flag genes outside the MG_loose purity genes
CLASSES = ("01 IT-ET Glut", "30 Astro-Epen", "34 Immune")
SUBCLASSES = (
    "007 L2/3 IT CTX Glut",
    "319 Astro-TE NN",
    "334 Microglia NN",
    "338 BAM NN",
)
SUBCLASS_CLASS = {
    "007 L2/3 IT CTX Glut": "01 IT-ET Glut",
    "319 Astro-TE NN": "30 Astro-Epen",
    "334 Microglia NN": "34 Immune",
    "338 BAM NN": "34 Immune",
}


def _profile(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array / array.sum()


NEURON = _profile([0.0005] * 5 + [0.001, 0.001, 0.5, 0.4, 0.097])
ASTRO = _profile([0.0005] * 5 + [0.5, 0.4, 0.05, 0.048, 0.0005])
MICROGLIA = _profile([0.18] * 5 + [0.001, 0.001, 0.05, 0.047, 0.2])
BAM = _profile([0.1, 0.4, 0.2, 0.05, 0.05, 0.001, 0.001, 0.1, 0.1, 0.098])


def _profiles_table() -> pd.DataFrame:
    rows = []
    for level, names, matrix in (
        ("CCN20230722_CLAS", CLASSES, [NEURON, ASTRO, (MICROGLIA + BAM) / 2]),
        ("CCN20230722_SUBC", SUBCLASSES, [NEURON, ASTRO, MICROGLIA, BAM]),
    ):
        for name, profile in zip(names, matrix, strict=True):
            for gene, value in zip(GENES, profile, strict=True):
                rows.append(
                    {
                        "level": level,
                        "node_name": name,
                        "gene_id": gene,
                        "expected_fraction": float(value),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def profiles() -> MouseFlagProfiles:
    return MouseFlagProfiles.from_table(
        _profiles_table(),
        gene_ids=GENES,
        symbols=SYMBOLS,
        class_level="CCN20230722_CLAS",
        subclass_level="CCN20230722_SUBC",
        subclass_class=SUBCLASS_CLASS,
    )


def _cells(
    profile: np.ndarray, n: int, depth: int, rng: np.random.Generator
) -> np.ndarray:
    return rng.multinomial(depth, profile, size=n)


def test_specific_genes_apply_ratio_share_and_the_ieg_exclusion() -> None:
    target = np.array([0.30, 0.05, 0.0005, 0.20, 0.10])
    others = np.array([[0.01, 0.004, 0.0, 0.005, 0.2], [0.001, 0.001, 0.0, 0.0, 0.0]])
    genes = specific_genes(
        target,
        others,
        gene_ids=["g0", "g1", "g2", "g3", "g4"],
        symbols=["A", "B", "C", "Fos", "E"],
        ratio=20.0,
        min_share=0.001,
        target="t",
    )
    # g0: 30x; g1: 12.5x (< 20); g2: share < 1/1000; g3: an IEG; g4: 0.5x.
    assert genes.gene_ids == ("g0",)
    assert genes.ratios == (pytest.approx(30.0),)
    assert "Fos" in IMMEDIATE_EARLY_GENES


def test_specific_genes_order_by_ratio_and_count_absent_elsewhere() -> None:
    genes = specific_genes(
        np.array([0.1, 0.2, 0.3]),
        np.array([[0.0, 0.001, 0.003]]),
        gene_ids=["a", "b", "c"],
        symbols=["a", "b", "c"],
        ratio=20.0,
        min_share=0.001,
        target="t",
    )
    assert genes.gene_ids == ("a", "b", "c")  # inf, 200x, 100x


def test_microglia_and_astrocyte_sets_follow_the_e3_rule(
    profiles: MouseFlagProfiles,
) -> None:
    config = AnnotationFlagsConfig()
    microglia = microglia_genes(profiles, config)
    # Equal ratios: gene-ID order; Fos (an immediate-early gene) is excluded.
    assert microglia.symbols == ("Cx3cr1", "Csf1r", "C1qa", "Aif1", "P2ry12")
    astro = astrocyte_genes(profiles, config)
    assert set(astro.symbols) == {"Aqp4", "Gfap"}
    assert np.allclose(floored_profiles(profiles.subclass_profiles).sum(axis=1), 1.0)


def test_spillover_flags_planted_microglial_counts_only(
    profiles: MouseFlagProfiles,
) -> None:
    rng = np.random.default_rng(0)
    clean_neurons = _cells(NEURON, 300, 400, rng)
    clean_astro = _cells(ASTRO, 300, 400, rng)
    planted = _cells(NEURON, 300, 400, rng) + _cells(MICROGLIA, 300, 60, rng)
    counts = sparse.csr_matrix(np.vstack([clean_neurons, clean_astro, planted]))

    result = microglial_spillover(counts, profiles, AnnotationFlagsConfig())

    assert result.defined and result.null_reason is None
    flag = result.raw_flag
    assert flag[:600].mean() <= 0.01
    assert flag[600:].mean() >= 0.95
    assert np.all(result.weight[600:][flag[600:]] >= 0.05)
    assert result.statistic[600:].min() > result.statistic[:300].max()
    fpr = result.checks["astrocyte_fpr"]
    assert fpr["evaluated"] is True and fpr["n_marker_astrocytes"] >= 250
    assert fpr["purity_genes"] == ["Cx3cr1", "Csf1r", "C1qa"]
    assert fpr["purity_basis"] == PURITY_BASIS_MG_LOOSE
    assert fpr["fpr"] == 0.0
    assert result.checks["heldout"]["evaluated"] is False  # 5 genes < 6


def test_spillover_statistic_is_zero_without_the_genes(
    profiles: MouseFlagProfiles,
) -> None:
    base, target = spillover_reference(profiles)
    counts = np.zeros((3, len(GENES)))
    counts[0, 7] = 50  # neuron counts only
    counts[1, 0] = 40  # microglial counts only
    statistic, weight = spillover_statistic(counts, np.array([0, 1, 2]), base, target)
    assert statistic[0] == 0.0 and weight[0] == 0.0
    assert statistic[2] == 0.0  # no counts at all
    assert statistic[1] > 10.0 and weight[1] == 1.0


def test_spillover_needs_the_minimum_weight(profiles: MouseFlagProfiles) -> None:
    """Deep cells with a 2% microglial component are significant but too small."""
    rng = np.random.default_rng(4)
    deep = _cells(NEURON, 200, 20000, rng) + _cells(MICROGLIA, 200, 400, rng)
    counts = sparse.csr_matrix(deep)
    result = microglial_spillover(counts, profiles, AnnotationFlagsConfig())
    assert np.median(result.weight) == pytest.approx(0.02)
    assert (result.statistic >= 10).mean() > 0.9
    assert result.raw_flag.mean() < 0.1
    permissive = microglial_spillover(
        counts, profiles, AnnotationFlagsConfig(microglia_weight_min=0.0)
    )
    assert permissive.raw_flag.mean() > 0.9


def test_spillover_null_includes_the_ambient_share(profiles: MouseFlagProfiles) -> None:
    base, target = spillover_reference(profiles)
    genes = np.array([0, 1, 2])
    counts = np.zeros((1, len(GENES)))
    counts[0, 7] = 990
    counts[0, 0] = 12  # 1.2% microglial counts, within a 10% ambient share
    with_ambient, _ = spillover_statistic(
        counts, genes, base, target, ambient_share=0.10
    )
    without, weight = spillover_statistic(
        counts, genes, base, target, ambient_share=0.0
    )
    assert with_ambient[0] == 0.0
    assert without[0] > 10.0 and weight[0] > 0.0


def test_spillover_weight_grid_must_start_at_zero(profiles: MouseFlagProfiles) -> None:
    base, target = spillover_reference(profiles)
    with pytest.raises(ValueError, match="start with 0"):
        spillover_statistic(
            np.ones((1, len(GENES))), np.array([0]), base, target, weight_grid=[0.1]
        )


def test_spillover_is_null_with_too_few_specific_genes(
    profiles: MouseFlagProfiles,
) -> None:
    counts = sparse.csr_matrix(np.ones((4, len(GENES))))
    result = microglial_spillover(
        counts, profiles, AnnotationFlagsConfig(min_specific_genes=6)
    )
    assert not result.defined
    assert result.null_reason == REASON_INSUFFICIENT_GENES
    assert np.isnan(result.statistic).all()
    assert result.rate is None


def test_spillover_is_null_when_marker_astrocytes_are_flagged(
    profiles: MouseFlagProfiles,
) -> None:
    rng = np.random.default_rng(1)
    # Marker astrocytes (<= 3 MG_loose counts per 1,000); a rule that flags
    # every cell fails the false-positive check, so the flag is null.
    astro = _cells(ASTRO, 400, 1000, rng)
    astro[:, MICROGLIA_GENES] = 0
    counts = sparse.csr_matrix(astro)
    loose = AnnotationFlagsConfig(microglia_stat_min=0.0, microglia_weight_min=0.0)
    result = microglial_spillover(counts, profiles, loose)
    purity, _ = astro_purity_genes(profiles, result.genes)
    assert marker_astrocytes(counts, result.astro_genes, purity).sum() > 300
    assert not result.defined
    assert result.null_reason is not None
    assert result.null_reason.startswith(REASON_ASTRO_FPR)


def test_astrocyte_fpr_counts_spill_on_the_flag_genes_outside_the_purity_set(
    profiles: MouseFlagProfiles,
) -> None:
    """E3's AST rule excludes on MG_loose only, so the check can fire (MO5)."""
    rng = np.random.default_rng(5)
    astro = _cells(ASTRO, 400, 1000, rng)
    astro[:, MICROGLIA_GENES] = 0
    # 4-8 per 1,000 flag-gene counts on every astrocyte (Aif1, P2ry12: not
    # purity genes) ...
    astro[:, FREE_FLAG_GENES] = 3
    # ... and a real spill-over component on 20 of them.
    astro[:20, FREE_FLAG_GENES] = 25
    counts = sparse.csr_matrix(astro)

    result = microglial_spillover(counts, profiles, AnnotationFlagsConfig())

    fpr = result.checks["astrocyte_fpr"]
    assert fpr["evaluated"] is True
    assert fpr["n_marker_astrocytes"] == 400
    assert fpr["n_flagged"] == 20
    assert fpr["fpr"] == pytest.approx(20 / 400)
    assert not result.defined
    assert result.null_reason is not None
    assert result.null_reason.startswith(REASON_ASTRO_FPR)
    # Excluding on the flag's own genes (the pre-review rule) keeps none of
    # these astrocytes: the rate is 0 by construction.
    assert not marker_astrocytes(counts, result.astro_genes, result.genes).any()


def test_astrocyte_purity_genes_fall_back_to_the_top_derived_genes() -> None:
    """Without all MG_loose genes: the 3 most specific derived genes."""
    keep = [index for index in range(len(GENES)) if SYMBOLS[index] != "C1qa"]
    table = MouseFlagProfiles.from_table(
        _profiles_table(),
        gene_ids=[GENES[index] for index in keep],
        symbols=[SYMBOLS[index] for index in keep],
        class_level="CCN20230722_CLAS",
        subclass_level="CCN20230722_SUBC",
        subclass_class=SUBCLASS_CLASS,
    )
    microglia = microglia_genes(table, AnnotationFlagsConfig())
    purity, basis = astro_purity_genes(table, microglia)
    assert basis == PURITY_BASIS_DERIVED
    assert purity.symbols == microglia.symbols[:3]
    counts = sparse.csr_matrix(np.ones((50, len(keep))))
    result = microglial_spillover(counts, table, AnnotationFlagsConfig())
    assert result.checks["astrocyte_fpr"]["n_flag_genes_outside_purity"] == 1


def test_astrocyte_fpr_is_not_evaluated_when_purity_covers_the_flag_genes() -> None:
    """A flag of only MG_loose genes cannot be tested on MG_loose-clean cells."""
    keep = [index for index in range(len(GENES)) if index not in FREE_FLAG_GENES]
    table = MouseFlagProfiles.from_table(
        _profiles_table(),
        gene_ids=[GENES[index] for index in keep],
        symbols=[SYMBOLS[index] for index in keep],
        class_level="CCN20230722_CLAS",
        subclass_level="CCN20230722_SUBC",
        subclass_class=SUBCLASS_CLASS,
    )
    counts = sparse.csr_matrix(np.ones((50, len(keep))))
    loose = AnnotationFlagsConfig(microglia_stat_min=0.0, microglia_weight_min=0.0)
    result = microglial_spillover(counts, table, loose)
    check = result.checks["astrocyte_fpr"]
    assert result.genes.symbols == ("Cx3cr1", "Csf1r", "C1qa")
    assert check["evaluated"] is False
    assert "cover every flag gene" in check["reason"]
    assert result.defined  # a check that cannot run never nulls the flag


def _heldout_profiles(
    genes: list[str],
) -> tuple[MouseFlagProfiles, np.ndarray, np.ndarray]:
    microglia = _profile([0.12] * 6 + [0.01, 0.01, 0.2, 0.2])
    neuron = _profile([0.0001] * 6 + [0.4, 0.3, 0.2, 0.1])
    rows = []
    for level, names, matrix in (
        ("CCN20230722_CLAS", ("01 IT-ET Glut", "34 Immune"), [neuron, microglia]),
        ("CCN20230722_SUBC", ("007 L2/3", "334 Microglia NN"), [neuron, microglia]),
    ):
        for name, profile in zip(names, matrix, strict=True):
            rows += [
                {
                    "level": level,
                    "node_name": name,
                    "gene_id": gene,
                    "expected_fraction": value,
                }
                for gene, value in zip(genes, profile, strict=True)
            ]
    table = MouseFlagProfiles.from_table(
        pd.DataFrame(rows),
        gene_ids=genes,
        symbols=genes,
        class_level="CCN20230722_CLAS",
        subclass_level="CCN20230722_SUBC",
        subclass_class={"007 L2/3": "01 IT-ET Glut", "334 Microglia NN": "34 Immune"},
    )
    return table, neuron, microglia


def test_heldout_check_rebuilds_the_flag_on_half_of_the_genes() -> None:
    genes = [f"g{index}" for index in range(10)]
    table, neuron, microglia = _heldout_profiles(genes)
    rng = np.random.default_rng(2)
    counts = np.vstack(
        [
            _cells(neuron, 400, 300, rng),
            _cells(neuron, 200, 300, rng) + _cells(microglia, 200, 60, rng),
        ]
    )
    result = microglial_spillover(
        sparse.csr_matrix(counts), table, AnnotationFlagsConfig()
    )
    heldout = result.checks["heldout"]
    assert heldout["evaluated"] is True
    assert heldout["build_genes"] == ["g0", "g2", "g4"]
    assert heldout["heldout_genes"] == ["g1", "g3", "g5"]
    assert set(heldout["build_genes"]).isdisjoint(heldout["heldout_genes"])
    assert heldout["enrichment"] > 10.0


def test_heldout_check_gives_no_enrichment_without_heldout_signal() -> None:
    """Control arm: spill only on the build genes -> held-out enrichment ~1."""
    genes = [f"g{index}" for index in range(10)]
    table, neuron, microglia = _heldout_profiles(genes)
    rng = np.random.default_rng(6)
    build_only = microglia.copy()
    build_only[[1, 3, 5]] = 0.0
    build_only /= build_only.sum()
    counts = np.vstack(
        [
            _cells(neuron, 400, 300, rng),
            _cells(neuron, 200, 300, rng) + _cells(build_only, 200, 60, rng),
        ]
    )
    # Background on the held-out genes, equal in every cell.
    counts[:, [1, 3, 5]] += rng.poisson(2.0, size=(len(counts), 3))
    result = microglial_spillover(
        sparse.csr_matrix(counts), table, AnnotationFlagsConfig()
    )
    heldout = result.checks["heldout"]
    assert heldout["build_flag_rate"] > 0.25  # the build half still flags
    assert heldout["enrichment"] is not None
    assert 0.7 < heldout["enrichment"] < 1.4


def test_region_coherence_counts_same_class_neighbours() -> None:
    xy = np.array([[float(i), 0.0] for i in range(10)])
    labels = ["A"] * 5 + ["B"] * 5
    coherence = region_coherence(xy, labels, k=2)
    assert coherence[0] == 1.0 and coherence[9] == 1.0
    assert coherence[4] == 0.5  # neighbours 3 (A) and 5 (B)
    assert np.isnan(region_coherence(xy[:2], labels[:2], k=2)).all()


def test_f1_needs_a_restricted_class_low_coherence_and_low_confidence() -> None:
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 1000, size=(400, 2))
    classes = np.array(["01 IT-ET Glut"] * 400, dtype=object)
    stray = np.arange(0, 400, 40)
    classes[stray] = "24 MY Glut"
    bp = np.full(400, 0.95)
    bp[stray[:5]] = 0.5
    config = MouseRegionConfig()
    result = region_incoherent(xy, list(classes), bp, config)
    assert result.defined
    flagged = np.flatnonzero(result.raw_flag)
    assert set(flagged) == set(stray[:5])  # confident strays stay unflagged
    astro = np.where(classes == "24 MY Glut", "30 Astro-Epen", classes)
    assert not region_incoherent(xy, list(astro), bp, config).raw_flag.any()


def test_f1_is_null_without_coordinates() -> None:
    result = region_incoherent(
        None, ["01 IT-ET Glut"], np.array([0.5]), MouseRegionConfig()
    )
    assert not result.defined and result.null_reason == REASON_NO_COORDINATES
    assert result.rate is None


def test_packaged_region_restricted_classes_are_e7s_23() -> None:
    classes = load_region_restricted_classes()
    assert len(classes) == 23
    assert "24 MY Glut" in classes and "32 OEC" in classes
    for scattered in ("06 CTX-CGE GABA", "30 Astro-Epen", "34 Immune", "33 Vascular"):
        assert scattered not in classes


@pytest.mark.slow
def test_derived_microglia_set_on_ag7_equals_e3s() -> None:
    """The store's ag7 panel bundle gives E3's five genes (and VZG2's 13)."""
    store = Path("/media/mathieubo/SSD1/MerXen/annotation_references/wmb_panel")
    gene_table = Path(
        "/media/mathieubo/SSD1/MerXen/mapmycells/abc_atlas/metadata/WMB-10X/"
        "20241115/gene.csv"
    )
    bundles = {
        "ag7": "5a032858b05d71419f69e52b76ea97aa0ed4ddc88973c6d38d71876bdea38a88",
        "vzg2": "daa8c4a6611e75c5eaba6cb45180f05b468b3ef2b87cc06c85a897319f246781",
    }
    expected = {
        "ag7": {"Csf1r", "Cx3cr1", "Aif1", "Blnk", "C1qa"},
        "vzg2": {
            "Trem2",
            "Csf1r",
            "C1qa",
            "C1qb",
            "Cx3cr1",
            "Tmem119",
            "Gpr34",
            "Ctss",
            "Blnk",
            "P2ry12",
            "Siglech",
            "Ptprc",
            "Selplg",
        },
    }
    if not gene_table.is_file() or not all(
        (store / b).is_dir() for b in bundles.values()
    ):
        pytest.skip("the dwight reference store is not mounted")
    from merxen.annotation.mouse_regions import WmbTaxonomy
    from merxen.annotation.reference import TaxonomyTreeView

    genes = pd.read_csv(gene_table)
    symbol_of = dict(zip(genes["gene_identifier"], genes["gene_symbol"], strict=True))
    for key, build in bundles.items():
        bundle = store / build
        table = pd.read_parquet(bundle / "profiles.parquet")
        gene_ids = sorted(table["gene_id"].unique())
        tree = TaxonomyTreeView.from_tree_dict(
            json.loads((bundle / "mapping_tree.json").read_text())
        )
        taxonomy = WmbTaxonomy.from_tree(
            tree, class_level="CCN20230722_CLAS", subclass_level="CCN20230722_SUBC"
        )
        profiles = MouseFlagProfiles.from_table(
            table,
            gene_ids=gene_ids,
            symbols=[symbol_of[gene] for gene in gene_ids],
            class_level="CCN20230722_CLAS",
            subclass_level="CCN20230722_SUBC",
            subclass_class={
                subclass: cls
                for cls, subclasses in taxonomy.subclasses.items()
                for subclass in subclasses
            },
        )
        derived = microglia_genes(profiles, AnnotationFlagsConfig())
        assert set(derived.symbols) == expected[key], key


def test_region_restricted_loader_checks_the_e7_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from merxen.annotation import vocab

    table = vocab.load_asset_table(vocab.REGION_RESTRICTED_FILE)
    table.loc[table["class"] == "06 CTX-CGE GABA", "region_restricted"] = "true"
    monkeypatch.setattr(vocab, "load_asset_table", lambda name: table.copy())
    with pytest.raises(ValueError, match="06 CTX-CGE GABA"):
        vocab.load_region_restricted_classes()
