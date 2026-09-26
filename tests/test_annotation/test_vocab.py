"""Tests for the packaged vocabularies (merxen.annotation.vocab)."""

from __future__ import annotations

import tomllib
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from merxen.annotation import vocab
from merxen.annotation.vocab import (
    BROAD_CLASSES,
    FINAL_LEVELS,
    HUMAN_BROAD_CLASSES,
    HUMAN_FLOOR_CLASSES,
    HUMAN_LINEAGE_OF_BROAD_CLASS,
    HUMAN_LINEAGES,
    MAP_FIRST_BROAD_LABELS,
    MOUSE_BROAD_CLASSES,
    NT_CLASSES,
    OLIGODENDROCYTE_LINEAGE,
    UNASSIGNED_LABEL,
    broad_class_for,
    broad_class_for_map_first,
    broad_classes_for_map_first,
    classify_neurotransmitter,
    floor_class_for,
    is_region_plausible,
    is_sink,
    lineage_for,
    load_floor_table,
    load_heldout_markers,
    load_state_genes,
    load_vocab,
    nt_for,
    seaad_broad_class,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# exp/E1/e1lib.py DROP_IDS: the 16 superclusters pruned in E1 (ii).
E1_DROP_IDS = {
    f"CS202210140_{number}"
    for number in (475, 479, 480, 478, 481, 482, 490, 491, 492, 487, 488, 489)
    + (493, 472, 471, 483)
}
SEAAD_VLMC_SUPERTYPES = {
    "VLMC_1": "Fibroblasts",
    "VLMC_2-SEAAD": "Fibroblasts",
    "Pericyte_1": "Vascular cells",
    "Pericyte_2-SEAAD": "Vascular cells",
    "SMC-SEAAD": "Vascular cells",
}


def test_tables_have_the_planned_sizes() -> None:
    whb = load_vocab("whb_supercluster")
    seaad = load_vocab("seaad_mr_subclass")
    wmb = load_vocab("wmb_class")
    assert len(whb.names) == 31
    assert len(seaad.names) == 29
    assert len(seaad.supertype_overrides) == 3
    assert len(wmb.names) == 34
    assert whb.species == seaad.species == "human"
    assert wmb.species == "mouse"
    assert load_vocab("whb_supercluster") is whb


def test_unassigned_label_matches_legacy() -> None:
    from merxen.config import ClusteringSquidpyAnnotationConfig

    assert UNASSIGNED_LABEL == "Mixed/Unknown"
    assert ClusteringSquidpyAnnotationConfig().unknown_label == UNASSIGNED_LABEL


def test_every_whb_supercluster_maps() -> None:
    whb = load_vocab("whb_supercluster")
    allowed_broad = {*HUMAN_BROAD_CLASSES, UNASSIGNED_LABEL}
    for name in whb.names:
        broad = whb.broad_class(name)
        lineage = whb.lineage(name)
        assert broad in allowed_broad, name
        assert lineage in {*HUMAN_LINEAGES, UNASSIGNED_LABEL}, name
        if broad != UNASSIGNED_LABEL:
            assert HUMAN_LINEAGE_OF_BROAD_CLASS[broad] == lineage, name
    assert set(whb.label_to_name()) == set(whb.frame["label"])


def test_every_seaad_subclass_maps() -> None:
    seaad = load_vocab("seaad_mr_subclass")
    allowed_broad = {*HUMAN_BROAD_CLASSES, UNASSIGNED_LABEL}
    for name in seaad.names:
        assert seaad.broad_class(name) in allowed_broad, name
        assert seaad.lineage(name) in {*HUMAN_LINEAGES, UNASSIGNED_LABEL}, name
    for supertype, expected in SEAAD_VLMC_SUPERTYPES.items():
        assert seaad_broad_class("VLMC & Perivascular", supertype) == expected
        assert seaad.lineage("VLMC & Perivascular", supertype) == expected
    assert seaad_broad_class("VLMC & Perivascular") == UNASSIGNED_LABEL
    assert seaad_broad_class("Endothelial", "Endo_1") == "Vascular cells"
    assert seaad_broad_class("Immune") == "Microglia"
    assert seaad_broad_class("OPC") == "Oligodendrocyte precursors"


def test_every_wmb_class_maps() -> None:
    wmb = load_vocab("wmb_class")
    for name in wmb.names:
        assert wmb.broad_class(name) in MOUSE_BROAD_CLASSES, name
    assert broad_class_for("30 Astro-Epen", "mouse") == "Astrocytes/Ependymal"
    assert broad_class_for("31 OPC-Oligo", "mouse") == OLIGODENDROCYTE_LINEAGE
    assert broad_class_for("34 Immune", "mouse") == "Microglia"
    with pytest.raises(KeyError, match="lineage"):
        wmb.lineage("01 IT-ET Glut")


def test_vocab_broad_classes_match_the_legacy_collapse() -> None:
    from merxen.analysis.clustering_squidpy import (
        collapse_atlas_label_to_broad_class,
    )

    whb = load_vocab("whb_supercluster")
    for name in whb.names:
        broad = whb.broad_class(name)
        if broad in HUMAN_BROAD_CLASSES:
            assert collapse_atlas_label_to_broad_class(name) == broad, name
    wmb = load_vocab("wmb_class")
    for name in wmb.names:
        assert collapse_atlas_label_to_broad_class(name) == wmb.broad_class(name)


def test_nodes_outside_the_seven_classes_stay_unallocated() -> None:
    """E1's scheme: glia outside the seven classes carry no broad class."""
    whb = load_vocab("whb_supercluster")
    for name in ("Bergmann glia", "Ependymal", "Choroid plexus"):
        assert whb.broad_class(name) == UNASSIGNED_LABEL, name
        assert whb.lineage(name) == UNASSIGNED_LABEL, name
        assert whb.broad_classes_any(name) == (), name
        assert not whb.is_region_plausible(name, "frontal_cortex"), name
    assert whb.broad_classes_any("Astrocyte") == ("Astrocytes",)
    assert whb.broad_classes_any("Splatter") == ()


def test_seaad_broad_classes_any_keep_the_vlmc_mass() -> None:
    """Without a supertype, VLMC & Perivascular counts for both classes (E1)."""
    vlmc = "VLMC & Perivascular"
    assert vocab.seaad_broad_classes_any(vlmc) == ("Vascular cells", "Fibroblasts")
    assert vocab.seaad_broad_classes_any(vlmc, "VLMC_2-SEAAD") == ("Fibroblasts",)
    assert vocab.seaad_broad_classes_any(vlmc, "Pericyte_1") == ("Vascular cells",)
    assert vocab.seaad_broad_classes_any(vlmc, "SMC-SEAAD") == ("Vascular cells",)
    assert vocab.seaad_broad_classes_any("Ependymal") == ()
    assert vocab.seaad_broad_classes_any("Astrocyte") == ("Astrocytes",)
    assert vocab.seaad_broad_classes_any("L2/3 IT") == ("Neurons",)
    seaad = load_vocab("seaad_mr_subclass")
    for name in seaad.names:
        broad = seaad.broad_class(name)
        expected = () if broad == UNASSIGNED_LABEL else (broad,)
        if name != vlmc:
            assert seaad.broad_classes_any(name) == expected, name
    assert load_vocab("wmb_class").broad_classes_any("31 OPC-Oligo") == (
        OLIGODENDROCYTE_LINEAGE,
    )


@pytest.mark.parametrize("table_id", ["whb_supercluster", "seaad_mr_subclass"])
def test_nt_covers_every_human_neuronal_node(table_id: str) -> None:
    table = load_vocab(table_id)  # type: ignore[arg-type]
    for name in table.names:
        if table.lineage(name) == "Neurons":
            assert table.nt(name) in NT_CLASSES, name
        else:
            assert table.nt(name) is None, name


def test_nt_covers_every_mouse_neuronal_class() -> None:
    wmb = load_vocab("wmb_class")
    for name in wmb.names:
        if wmb.broad_class(name) == "Neurons":
            assert wmb.nt(name) in NT_CLASSES, name
        else:
            assert wmb.nt(name) is None, name
    assert nt_for("21 MB Dopa", "mouse") == "Other"
    assert nt_for("05 OB-IMN GABA", "mouse") == "Inhibitory"


def test_nt_of_cortical_superclusters() -> None:
    for name in (
        "Upper-layer intratelencephalic",
        "Deep-layer intratelencephalic",
        "Deep-layer near-projecting",
        "Deep-layer corticothalamic and 6b",
    ):
        assert nt_for(name) == "Excitatory"
    for name in ("MGE interneuron", "CGE interneuron", "LAMP5-LHX6 and Chandelier"):
        assert nt_for(name) == "Inhibitory"
    assert nt_for("Splatter") == "Other"
    assert nt_for("Astrocyte") is None


def test_sinks_and_frontal_plausibility() -> None:
    whb = load_vocab("whb_supercluster")
    sinks = {name for name in whb.names if whb.is_sink(name)}
    assert sinks == {"Splatter", "Miscellaneous"}
    implausible = {
        whb.row(name)["label"]
        for name in whb.names
        if not whb.is_region_plausible(name, "frontal_cortex")
    }
    assert implausible == E1_DROP_IDS
    assert is_sink("Miscellaneous")
    assert is_region_plausible("Miscellaneous", "frontal_cortex")
    assert not is_region_plausible("Splatter", "frontal_cortex")
    assert not is_region_plausible("Ependymal", "frontal_cortex")
    assert is_region_plausible("Upper-layer intratelencephalic", "frontal_cortex")
    for name in sinks:
        assert whb.broad_class(name) == UNASSIGNED_LABEL
    assert not is_sink("30 Astro-Epen", "mouse")


def test_region_plausibility_refuses_unvalidated_regions() -> None:
    with pytest.raises(ValueError, match="OD-C7"):
        is_region_plausible("Astrocyte", "temporal_cortex")
    with pytest.raises(ValueError, match="mouse"):
        is_region_plausible("30 Astro-Epen", "frontal_cortex", "mouse")
    assert load_vocab("whb_supercluster").regions == ("frontal_cortex",)


def test_seaad_sinks_and_plausibility() -> None:
    seaad = load_vocab("seaad_mr_subclass")
    assert {name for name in seaad.names if seaad.is_sink(name)} == {"Ependymal"}
    implausible = {
        name
        for name in seaad.names
        if not seaad.is_region_plausible(name, "frontal_cortex")
    }
    assert implausible == {"Sub-CA1", "CA2-4", "DG", "EC IT"}


def test_never_drop_classes() -> None:
    wmb = load_vocab("wmb_class")
    never_drop = {name for name in wmb.names if wmb.is_never_drop(name)}
    assert never_drop == {
        "25 Pineal Glut",
        "30 Astro-Epen",
        "31 OPC-Oligo",
        "33 Vascular",
        "34 Immune",
    }
    assert not load_vocab("whb_supercluster").is_never_drop("Astrocyte")


def test_lineage_for_is_human_only() -> None:
    assert lineage_for("Oligodendrocyte") == OLIGODENDROCYTE_LINEAGE
    assert lineage_for("Committed oligodendrocyte precursor") == OLIGODENDROCYTE_LINEAGE
    with pytest.raises(ValueError, match="no lineage level"):
        lineage_for("31 OPC-Oligo", "mouse")
    with pytest.raises(ValueError, match="species"):
        lineage_for("Astrocyte", "rat")  # type: ignore[arg-type]


def test_unknown_nodes_and_tables_raise() -> None:
    with pytest.raises(KeyError, match="not a node"):
        broad_class_for("Purkinje cell")
    with pytest.raises(ValueError, match="table_id"):
        load_vocab("whb_cluster")  # type: ignore[arg-type]
    assert "Astrocyte" in load_vocab("whb_supercluster")
    assert 3 not in load_vocab("whb_supercluster")


@pytest.mark.parametrize(
    ("term", "expected"),
    [
        ("VGLUT1", "Excitatory"),
        ("VGLUT1 VGLUT2", "Excitatory"),
        ("DA VGLUT2", "Excitatory"),
        ("GABA", "Inhibitory"),
        ("GLY", "Inhibitory"),
        ("GABA VGLUT3", "Other"),
        ("HDC", "Other"),
        ("Glut", "Excitatory"),
        ("GABA-Glyc", "Inhibitory"),
        ("Glut-GABA", "Other"),
        ("Dopa", "Other"),
        (None, None),
        (float("nan"), None),
        ("  ", None),
        (pd.NA, None),
        (np.float32("nan"), None),
        (np.float64("nan"), None),
    ],
)
def test_classify_neurotransmitter(term: str | None, expected: str | None) -> None:
    assert classify_neurotransmitter(term) == expected


def test_classify_neurotransmitter_over_nullable_columns() -> None:
    """Unannotated entries of nullable columns stay unannotated, not 'Other'."""
    for dtype in ("string", "category", object):
        values = pd.Series(["GABA", None, "VGLUT1"], dtype=dtype)
        assert [classify_neurotransmitter(value) for value in values] == [
            "Inhibitory",
            None,
            "Excitatory",
        ]


@pytest.mark.parametrize("species", ["human", "mouse"])
def test_broad_class_for_map_first_never_returns_a_sink(species: str) -> None:
    primary = vocab.primary_vocab(species)  # type: ignore[arg-type]
    labels = MAP_FIRST_BROAD_LABELS[species]
    sink_names = {name for name in primary.names if primary.is_sink(name)}
    candidates = [*primary.names, *BROAD_CLASSES[species], None, "", "Splatter"]
    candidates += [OLIGODENDROCYTE_LINEAGE, "Ependymal", "Bergmann glia"]
    for level in FINAL_LEVELS[species]:
        for broad in candidates:
            for lineage in candidates:
                result = broad_class_for_map_first(
                    level,
                    broad,
                    lineage,
                    species=species,  # type: ignore[arg-type]
                )
                assert result in labels, (level, broad, lineage)
                assert result not in sink_names
                assert result not in {"Splatter", "Miscellaneous", "Ependymal"}
    for name in sink_names:
        assert (
            broad_class_for_map_first(
                "supercluster",
                name,
                "Neurons",
                species=species,  # type: ignore[arg-type]
            )
            == UNASSIGNED_LABEL
        )


def test_broad_class_for_map_first_follows_plan_rules() -> None:
    assert (
        broad_class_for_map_first("broad", "Astrocytes", "Astrocytes", species="human")
        == "Astrocytes"
    )
    assert (
        broad_class_for_map_first("supercluster", "Neurons", "Neurons", species="human")
        == "Neurons"
    )
    assert (
        broad_class_for_map_first(
            "lineage",
            "Oligodendrocyte precursors",
            OLIGODENDROCYTE_LINEAGE,
            species="human",
        )
        == OLIGODENDROCYTE_LINEAGE
    )
    assert (
        broad_class_for_map_first("lineage", None, "Neurons", species="human")
        == "Neurons"
    )
    # §4.5 allows only the two lineage fallbacks: a single-class lineage whose
    # broad level is not confident gives Mixed/Unknown.
    for lineage in ("Astrocytes", "Microglia", "Vascular cells", "Fibroblasts"):
        assert (
            broad_class_for_map_first("lineage", None, lineage, species="human")
            == UNASSIGNED_LABEL
        ), lineage
        assert (
            broad_class_for_map_first("lineage", lineage, lineage, species="human")
            == UNASSIGNED_LABEL
        ), lineage
    assert (
        broad_class_for_map_first("none", "Astrocytes", "Astrocytes", species="human")
        == UNASSIGNED_LABEL
    )
    assert (
        broad_class_for_map_first("broad", "Mixed/Unknown", None, species="human")
        == UNASSIGNED_LABEL
    )
    # A node name is mapped through the vocabulary.
    assert (
        broad_class_for_map_first(
            "supercluster", "Oligodendrocyte", None, species="human"
        )
        == "Oligodendrocytes"
    )
    assert (
        broad_class_for_map_first("class", "30 Astro-Epen", None, species="mouse")
        == "Astrocytes/Ependymal"
    )
    assert (
        broad_class_for_map_first("subclass", "Microglia", None, species="mouse")
        == "Microglia"
    )
    with pytest.raises(ValueError, match="not a mouse level"):
        broad_class_for_map_first("lineage", None, "Neurons", species="mouse")


def test_vectorised_map_first_matches_scalar() -> None:
    index = pd.Index(["a", "b", "c", "d"])
    levels = pd.Series(["none", "broad", "lineage", "supercluster"], index=index)
    broads = ["Astrocytes", "Astrocytes", None, "Splatter"]
    lineages = [None, "Astrocytes", OLIGODENDROCYTE_LINEAGE, "Neurons"]
    result = broad_classes_for_map_first(levels, broads, lineages, species="human")
    assert list(result.index) == list(index)
    assert list(result.cat.categories) == list(MAP_FIRST_BROAD_LABELS["human"])
    expected = [
        broad_class_for_map_first(level, broad, lineage, species="human")
        for level, broad, lineage in zip(levels, broads, lineages, strict=True)
    ]
    assert result.tolist() == expected
    mouse = broad_classes_for_map_first(["broad"], ["Microglia"], species="mouse")
    assert mouse.tolist() == ["Microglia"]
    with pytest.raises(ValueError, match="lengths differ"):
        broad_classes_for_map_first(["broad"], [], species="human")


def test_floor_classes_cover_every_broad_class() -> None:
    whb = load_vocab("whb_supercluster")
    floors = load_floor_table("human")
    keys = set(
        zip(floors["level"], floors["floor_class"], floors["platform"], strict=True)
    )
    for name in whb.names:
        for level in ("broad", "supercluster"):
            floor_class = floor_class_for(name, species="human", level=level)
            if whb.broad_class(name) == UNASSIGNED_LABEL:
                assert floor_class is None, name
                continue
            assert floor_class in HUMAN_FLOOR_CLASSES, name
            for platform in ("MERSCOPE", "XENIUM"):
                assert (level, floor_class, platform) in keys, (name, level)
    assert (
        floor_class_for("Committed oligodendrocyte precursor", species="human") == "OPC"
    )
    assert (
        floor_class_for(
            "Committed oligodendrocyte precursor", species="human", level="supercluster"
        )
        == "COP"
    )
    assert floor_class_for("Microglia", species="human") == "Immune"
    assert floor_class_for("MGE interneuron", species="human") == "Inh"
    assert floor_class_for("34 Immune", species="mouse") == "34 Immune"


def test_human_floors_match_plan_values() -> None:
    floors = load_floor_table("human").set_index(["level", "floor_class", "platform"])
    expected = {
        ("broad", "Immune", "XENIUM"): 15,
        ("broad", "Inh", "XENIUM"): 30,
        ("broad", "Inh", "MERSCOPE"): 10,
        ("broad", "Oligo", "XENIUM"): 10,
        ("broad", "OtherNeuron", "MERSCOPE"): 60,
        ("supercluster", "Inh", "MERSCOPE"): 30,
        ("supercluster", "Inh", "XENIUM"): 10,
        ("supercluster", "Vascular", "XENIUM"): 30,
        ("supercluster", "Fibroblast", "XENIUM"): 30,
        ("supercluster", "OPC", "MERSCOPE"): 120,
        ("supercluster", "COP", "XENIUM"): 120,
        ("supercluster", "Astro", "XENIUM"): 10,
    }
    for key, value in expected.items():
        assert int(floors.loc[key, "min_counts"]) == value, key
    assert floors.loc[("broad", "Oligo", "XENIUM"), "inherited_from"] == "MERSCOPE"
    assert set(floors["panel_family"]) == {"human_set_a"}
    assert set(floors["floor_source"]) == {"real_e2"}
    assert len(floors) == 38


def test_mouse_floors_cover_every_class() -> None:
    floors = load_floor_table("mouse")
    wmb = load_vocab("wmb_class")
    for family in ("mouse_ag7", "mouse_vzg2"):
        subset = floors[floors["panel_family"] == family]
        for level, value in (("class", 20), ("subclass", 50)):
            rows = subset[subset["level"] == level]
            assert set(rows["floor_class"]) == set(wmb.names)
            assert set(rows["min_counts"]) == {value}
    assert set(floors["platform"]) == {"MERSCOPE"}


def test_state_genes_and_heldout_markers() -> None:
    assert load_state_genes("human") == (
        "CTSD",
        "SERPINA3",
        "SOX2",
        "CDKN1A",
        "APOE",
        "GFAP",
        "CD44",
        "C3",
        "HSPA1A",
        "HSP90AA1",
    )
    assert set(load_state_genes("mouse")) == {
        "Apoe",
        "Gfap",
        "C4b",
        "Serpina3n",
        "H2-K1",
    }
    for species in ("human", "mouse"):
        rationale = vocab.load_asset_table(vocab.STATE_GENE_FILES[species])["rationale"]
        assert (rationale.str.len() > 10).all()
    human = load_heldout_markers("human")
    assert set(human["marker_class"]) <= set(HUMAN_FLOOR_CLASSES)
    assert (human.groupby("marker_class").size() >= 2).all()
    assert "SST" not in set(human["gene_symbol"])
    avoid = human.set_index("gene_symbol")["avoid_platforms"]
    assert avoid["GAD2"] == "MERSCOPE"
    assert avoid["P2RY12"] == "MERSCOPE"
    mouse = load_heldout_markers("mouse")
    assert set(mouse["marker_class"]) <= set(load_vocab("wmb_class").names)
    assert {"P2ry12", "Tmem119", "Siglech"} <= set(mouse["gene_symbol"])


def test_asset_tables_are_fresh_copies() -> None:
    first = vocab.load_asset_table("wmb_class_vocab.csv")
    first.loc[0, "class"] = "changed"
    assert vocab.load_asset_table("wmb_class_vocab.csv").loc[0, "class"] != "changed"
    with pytest.raises(FileNotFoundError):
        vocab.asset_path("missing.csv")


def test_committed_tables_agree_with_overrides() -> None:
    overrides = yaml.safe_load((vocab.ASSET_DIR / "overrides.yaml").read_text())
    whb = load_vocab("whb_supercluster")
    for name, mapping in overrides["whb"]["superclusters"].items():
        assert whb.broad_class(name) == mapping["broad_class"], name
        assert whb.lineage(name) == mapping["lineage"], name
    assert {name for name in whb.names if whb.is_sink(name)} == set(
        overrides["whb"]["sinks"]
    )
    assert set(overrides["whb"]["regions"]["frontal_cortex"]["implausible_labels"]) == (
        E1_DROP_IDS
    )
    wmb = load_vocab("wmb_class")
    assert {name for name in wmb.names if wmb.is_never_drop(name)} == set(
        overrides["wmb"]["never_drop"]
    )


def test_assets_are_packaged_and_small() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    patterns = pyproject["tool"]["setuptools"]["package-data"]["merxen"]
    package_root = REPO_ROOT / "src" / "merxen"
    assets = sorted(path for path in vocab.ASSET_DIR.iterdir() if path.is_file())
    assert {path.name for path in assets} >= {
        "whb_supercluster_vocab.csv",
        "seaad_mr_subclass_vocab.csv",
        "wmb_class_vocab.csv",
        "floors_human.csv",
        "floors_mouse.csv",
        "state_genes_human.csv",
        "heldout_markers_human.csv",
        "overrides.yaml",
        "NOTICE",
    }
    for path in assets:
        relative = path.relative_to(package_root).as_posix()
        assert any(fnmatch(relative, pattern) for pattern in patterns), relative
        assert path.stat().st_size < 500 * 1024, relative
