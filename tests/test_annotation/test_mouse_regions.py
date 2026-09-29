"""Tests for mouse region inference and two-tier pruning (plan §7.2; M6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.config import MOUSE_CCF_REGIONS, MouseRegionConfig
from merxen.annotation.mapmycells_engine import coerce_tidy_dtypes
from merxen.annotation.mouse_regions import (
    REGION_CELL_COLUMNS,
    DroppedNode,
    MouseRegionError,
    MouseRegionOutputs,
    MouseRegionRecord,
    RegionPlan,
    RegionShareBundle,
    RegionShares,
    SectionRegionsRequest,
    WmbTaxonomy,
    changed_cells,
    check_rule_variant,
    confident_neurons,
    count_dropped_subclasses,
    dropped_levels,
    infer_regions,
    load_region_outputs,
    merge_pruned_tidy,
    plan_region_step,
    pruned_lookup,
    pruned_tree,
    read_region_cells,
    region_cells_frame,
    two_tier_drop_list,
    write_region_cells,
)
from merxen.annotation.reference import TaxonomyTreeView, lookup_key

CLAS = "CCN20230722_CLAS"
SUBC = "CCN20230722_SUBC"
CLUS = "CCN20230722_CLUS"

# A small WMB-like taxonomy: class name -> {subclass name: MERFISH grey cells
# per division}. Division order follows MOUSE_CCF_REGIONS.
TAXONOMY: dict[str, dict[str, dict[str, int]]] = {
    "01 IT-ET Glut": {
        "007 L2/3 IT CTX Glut": {"Isocortex": 1000},
        "008 L2/3 IT ENT Glut": {"Isocortex": 30, "OLF": 970},  # share .03
        "009 L4/5 IT CTX Glut": {"Isocortex": 900, "HPF": 100},
        "010 IT tiny Glut": {"OLF": 10},  # < 20 cells, class present: kept
        "011 CA1 Glut": {"HPF": 500},
    },
    "19 MB Glut": {"190 MB Glut A": {"MB": 500}, "191 MB Glut B": {"MB": 400}},
    "24 MY Glut": {
        "240 MY Glut A": {"MY": 480, "MB": 20},
        "241 MY Glut B": {"MY": 300, "MB": 100},  # share .25 < .3: dropped
        "242 MY Glut C": {"MY": 60, "MB": 40},  # share .40: kept
        "243 MY Glut tiny": {"MY": 5},  # < 20 cells, absent class: follows
    },
    "27 MY GABA": {"270 MY GABA A": {"MY": 900}, "271 MY GABA B": {"MY": 50}},
    # 55 grey cells (< 100): never "absent", so the default 0.1 rule applies.
    "15 HY Gnrh1 Glut": {
        "150 HY Gnrh1 Glut": {"HY": 40, "MB": 5},  # share .11
        "151 HY Gnrh1 tiny": {"HY": 10},
    },
    "30 Astro-Epen": {"300 Astro MY": {"MY": 1000}},  # never dropped
    "99 Not In MERFISH": {"990 Unmapped": {}},  # no MERFISH grey cells
}
PRESENT = ("Isocortex", "HPF", "MB")


def _label(name: str, level: str) -> str:
    return f"{level}_{name.split()[0]}"


def _tree() -> TaxonomyTreeView:
    classes: dict[str, list[str]] = {}
    subclasses: dict[str, list[str]] = {}
    names: dict[str, dict[str, dict[str, str]]] = {CLAS: {}, SUBC: {}, CLUS: {}}
    for class_name, members in TAXONOMY.items():
        class_label = _label(class_name, CLAS)
        names[CLAS][class_label] = {"name": class_name}
        classes[class_label] = []
        for subclass_name in members:
            sub_label = _label(subclass_name, SUBC)
            names[SUBC][sub_label] = {"name": subclass_name}
            classes[class_label].append(sub_label)
            leaf = f"{sub_label}_leaf"
            subclasses[sub_label] = [leaf]
            names[CLUS][leaf] = {"name": f"{subclass_name} leaf"}
    tree = {
        "hierarchy": [CLAS, SUBC, CLUS],
        CLAS: classes,
        SUBC: subclasses,
        CLUS: {leaf: [] for kids in subclasses.values() for leaf in kids},
        "name_mapper": names,
    }
    return TaxonomyTreeView.from_tree_dict(tree)


def _taxonomy() -> WmbTaxonomy:
    return WmbTaxonomy.from_tree(_tree(), class_level=CLAS, subclass_level=SUBC)


def _share_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    share_rows: list[dict[str, Any]] = []
    home_rows: list[dict[str, Any]] = []
    for class_name, members in TAXONOMY.items():
        class_counts: dict[str, int] = {}
        for subclass_name, counts in members.items():
            for region, value in counts.items():
                class_counts[region] = class_counts.get(region, 0) + value
            total = sum(counts.values())
            if not total:
                continue
            for region in MOUSE_CCF_REGIONS:
                share_rows.append(
                    {
                        "level": "subclass",
                        "node_name": subclass_name,
                        "region": region,
                        "n": counts.get(region, 0),
                        "share": counts.get(region, 0) / total,
                        "n_grey": total,
                        "n_all": total,
                    }
                )
            home = max(counts, key=lambda region: counts[region])
            home_rows.append(
                {
                    "level": "subclass",
                    "node_name": subclass_name,
                    "home": home,
                    "home_share": counts[home] / total,
                    "n_grey": total,
                }
            )
        total = sum(class_counts.values())
        if not total:
            continue
        for region in MOUSE_CCF_REGIONS:
            share_rows.append(
                {
                    "level": "class",
                    "node_name": class_name,
                    "region": region,
                    "n": class_counts.get(region, 0),
                    "share": class_counts.get(region, 0) / total,
                    "n_grey": total,
                    "n_all": total,
                }
            )
    return pd.DataFrame(share_rows), pd.DataFrame(home_rows)


def _shares() -> RegionShares:
    return RegionShares.from_tables(*_share_tables())


def _names(nodes: list[DroppedNode]) -> list[str]:
    return [node.name for node in nodes]


# --------------------------------------------------------------------------
# Request and settings


def test_section_regions_request_parses_auto_none_and_overrides() -> None:
    auto = SectionRegionsRequest.parse(" AUTO ")
    assert (auto.source, auto.value, auto.regions) == ("auto", "auto", ())
    assert auto.needs_region_shares
    none = SectionRegionsRequest.parse("None")
    assert none.source == "none" and not none.needs_region_shares
    override = SectionRegionsRequest.parse("isocortex; hpf;TH;hpf")
    assert override.source == "override"
    assert override.regions == ("Isocortex", "HPF", "TH")
    assert override.value == "Isocortex;HPF;TH"
    with pytest.raises(ValueError, match="unknown mouse section region"):
        SectionRegionsRequest.parse("Isocortex;Striatum")
    with pytest.raises(ValueError, match="must not be blank"):
        SectionRegionsRequest.parse(" ")


def test_unimplemented_rule_variants_are_refused() -> None:
    check_rule_variant(MouseRegionConfig())
    with pytest.raises(MouseRegionError, match="coupled_regions"):
        check_rule_variant(MouseRegionConfig(coupled_regions={"OB": ["OLF"]}))
    with pytest.raises(MouseRegionError, match="rule variant"):
        check_rule_variant(MouseRegionConfig(rule_variant="a"))


def test_region_shares_sum_the_present_grey_divisions() -> None:
    shares = _shares()
    assert list(shares.class_share.columns) == list(MOUSE_CCF_REGIONS)
    assert shares.class_n_grey["15 HY Gnrh1 Glut"] == 55
    present = shares.present_share("subclass", PRESENT)
    assert present["241 MY Glut B"] == pytest.approx(0.25)
    assert present["008 L2/3 IT ENT Glut"] == pytest.approx(0.03)
    assert present["011 CA1 Glut"] == pytest.approx(1.0)
    assert shares.present_share("class", ())["01 IT-ET Glut"] == 0.0
    assert shares.subclass_home["240 MY Glut A"] == "MY"
    assert "990 Unmapped" not in shares.subclass_home.index


def test_wmb_taxonomy_reads_class_and_subclass_levels() -> None:
    taxonomy = _taxonomy()
    assert taxonomy.class_label["24 MY Glut"] == "CCN20230722_CLAS_24"
    assert taxonomy.subclasses["19 MB Glut"] == ("190 MB Glut A", "191 MB Glut B")
    with pytest.raises(MouseRegionError, match="no"):
        WmbTaxonomy.from_tree(_tree(), class_level=CLAS, subclass_level=CLUS)


# --------------------------------------------------------------------------
# Two-tier drop rule, never-drop and small-subclass inheritance


def test_two_tier_rule_drops_by_the_class_share() -> None:
    nodes = two_tier_drop_list(PRESENT, _shares(), _taxonomy(), MouseRegionConfig())

    by_name = {node.name: node for node in nodes}
    assert _names(nodes) == [
        "008 L2/3 IT ENT Glut",
        "240 MY Glut A",
        "241 MY Glut B",
        "243 MY Glut tiny",
        "27 MY GABA",
    ]
    # Class 01 present (share .72): only 008 (share .03 < .1) is dropped; its
    # tiny subclass (< 20 cells) follows the present class and stays.
    assert by_name["008 L2/3 IT ENT Glut"].rule == "low_share"
    assert by_name["008 L2/3 IT ENT Glut"].kind == "subclass"
    assert by_name["008 L2/3 IT ENT Glut"].node == "CCN20230722_SUBC_008"
    # Class 24 absent (share 160 / 1005 < .2): the .3 rule, and its tiny
    # subclass follows the class; 242 (share .40) is kept, so no class node.
    assert by_name["241 MY Glut B"].rule == "absent_class_low_share"
    assert by_name["241 MY Glut B"].present_share == pytest.approx(0.25)
    assert by_name["243 MY Glut tiny"].rule == "absent_class_small_subclass"
    # Class 27: every subclass dropped -> one class node.
    assert by_name["27 MY GABA"].kind == "class"
    assert by_name["27 MY GABA"].level == CLAS
    assert by_name["27 MY GABA"].node == "CCN20230722_CLAS_27"
    assert by_name["27 MY GABA"].rule == "all_subclasses_dropped"


def test_never_drop_and_unmapped_classes_are_kept() -> None:
    nodes = two_tier_drop_list(PRESENT, _shares(), _taxonomy(), MouseRegionConfig())

    classes = {node.class_name for node in nodes}
    assert "30 Astro-Epen" not in classes  # never dropped, though all in MY
    assert "99 Not In MERFISH" not in classes  # no MERFISH grey cells
    # 15 HY Gnrh1 Glut has 55 < 100 cells: never "absent", so 150 (share
    # .11) passes the default rule and the tiny 151 stays with its class.
    assert "15 HY Gnrh1 Glut" not in classes
    smaller = two_tier_drop_list(
        PRESENT,
        _shares(),
        _taxonomy(),
        MouseRegionConfig(min_merfish_cells_class=50),
    )
    assert "15 HY Gnrh1 Glut" in _names(smaller)  # absent: both subclasses go
    more = MouseRegionConfig(never_drop_classes=["27 MY GABA"])
    assert "27 MY GABA" not in _names(
        two_tier_drop_list(PRESENT, _shares(), _taxonomy(), more)
    )
    assert "30 Astro-Epen" in _names(
        two_tier_drop_list(PRESENT, _shares(), _taxonomy(), more)
    )


def test_small_subclasses_follow_their_class() -> None:
    config = MouseRegionConfig()
    nodes = _names(two_tier_drop_list(PRESENT, _shares(), _taxonomy(), config))
    assert "243 MY Glut tiny" in nodes  # absent class: follows it (dropped)
    assert "010 IT tiny Glut" not in nodes  # present class: kept
    # With the whole taxonomy present nothing is dropped, tiny subclasses too.
    everything = two_tier_drop_list(MOUSE_CCF_REGIONS, _shares(), _taxonomy(), config)
    assert everything == []
    # Raising the small-subclass limit makes 242 (100 cells) follow class 24.
    bigger = MouseRegionConfig(min_merfish_cells_subclass=101)
    assert "24 MY Glut" in _names(
        two_tier_drop_list(PRESENT, _shares(), _taxonomy(), bigger)
    )


def test_drop_list_nodes_never_overlap_and_count_subclasses() -> None:
    taxonomy = _taxonomy()
    nodes = two_tier_drop_list(PRESENT, _shares(), taxonomy, MouseRegionConfig())
    class_nodes = {node.name for node in nodes if node.kind == "class"}
    assert not [
        node
        for node in nodes
        if node.kind == "subclass" and node.class_name in class_nodes
    ]
    # 008, then 240 / 241 / 243, then class 27's two subclasses.
    assert count_dropped_subclasses(nodes, taxonomy) == 1 + 3 + 2
    tree = pruned_tree(_tree(), nodes)
    assert "CCN20230722_CLAS_27" not in tree.nodes(CLAS)
    assert "CCN20230722_SUBC_241" not in tree.nodes(SUBC)
    assert "CCN20230722_SUBC_242" in tree.nodes(SUBC)


# --------------------------------------------------------------------------
# Region inference on synthetic tiles


def _section(
    blocks: dict[str, list[tuple[int, int]]],
    *,
    per_tile: int = 3,
    tile_um: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confident neurons at tile centres: ``per_tile`` of a home per tile."""
    subclass_of = {
        "Isocortex": "007 L2/3 IT CTX Glut",
        "HPF": "011 CA1 Glut",
        "MB": "190 MB Glut A",
        "MY": "240 MY Glut A",
        "HY": "150 HY Gnrh1 Glut",
    }
    xy, names = [], []
    for region, tiles in blocks.items():
        for i, j in tiles:
            for k in range(per_tile):
                xy.append(((i + 0.2 + 0.2 * k) * tile_um, (j + 0.5) * tile_um))
                names.append(subclass_of[region])
    return np.array(xy), np.array(names, dtype=object), np.ones(len(xy), bool)


def _rect(i0: int, i1: int, j0: int, j1: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(i0, i1) for j in range(j0, j1)]


def test_one_percent_rule_and_connected_components() -> None:
    # 400 assigned tiles: Isocortex 300, MB 88, HPF 4 connected (1%, present),
    # MY 4 scattered (1% but no 3-tile component), HY 3 connected (< 1%).
    blocks = {
        "Isocortex": _rect(0, 20, 0, 15),
        "MB": [*_rect(0, 20, 15, 19), *_rect(0, 8, 19, 20)],
        "HPF": [(10, 19), (11, 19), (12, 19), (13, 19)],
        "MY": [(15, 19), (17, 19), (19, 19), (9, 19)],
        "HY": [(14, 20), (15, 20), (16, 20)],
    }
    xy, names, confident = _section(blocks)
    inference = infer_regions(xy, names, confident, _shares(), MouseRegionConfig())

    assert inference.n_assigned_tiles == 399
    assert inference.tile_counts["HPF"] == 4
    assert inference.largest_component["MY"] == 1
    assert inference.largest_component["HPF"] == 4
    assert inference.inferred_regions == ("Isocortex", "HPF", "MB")
    # 1% of 399 tiles is 3.99: HY's 3 connected tiles are below it; lowering
    # the fraction admits HY, and a 5-tile component rule removes HPF.
    lower = MouseRegionConfig(min_tile_fraction=0.005)
    assert (
        "HY" in infer_regions(xy, names, confident, _shares(), lower).inferred_regions
    )
    strict = MouseRegionConfig(min_component_tiles=5)
    assert (
        "HPF"
        not in infer_regions(xy, names, confident, _shares(), strict).inferred_regions
    )
    # Cells know their tile's majority division.
    assert inference.cell_tile_region[0] == "Isocortex"
    assert set(inference.tiles.columns) == {"tile_i", "tile_j", "region", "n_neurons"}


def test_exactly_one_percent_of_tiles_is_present() -> None:
    blocks = {"Isocortex": _rect(0, 20, 0, 15), "HPF": [(0, 15), (1, 15), (2, 15)]}
    xy, names, confident = _section(blocks)
    inference = infer_regions(xy, names, confident, _shares(), MouseRegionConfig())
    assert inference.n_assigned_tiles == 303
    assert inference.inferred_regions == ("Isocortex",)  # 3 / 303 < 1%
    config = MouseRegionConfig(min_tile_fraction=3 / 303)
    assert infer_regions(xy, names, confident, _shares(), config).inferred_regions == (
        "Isocortex",
        "HPF",
    )


def test_tiles_need_enough_confident_neurons_and_voters() -> None:
    blocks = {"Isocortex": _rect(0, 10, 0, 10)}
    xy, names, confident = _section(blocks, per_tile=2)
    inference = infer_regions(xy, names, confident, _shares(), MouseRegionConfig())
    assert inference.n_assigned_tiles == 0 and inference.inferred_regions == ()
    assert inference.n_voting_neurons == 200
    two = MouseRegionConfig(min_neurons_per_tile=2)
    assert infer_regions(xy, names, confident, _shares(), two).n_assigned_tiles == 100
    # Non-confident cells and subclasses without a MERFISH home do not vote.
    xy3, names3, confident3 = _section(blocks, per_tile=3)
    confident3[::3] = False
    assert (
        infer_regions(
            xy3, names3, confident3, _shares(), MouseRegionConfig()
        ).n_assigned_tiles
        == 0
    )
    names3[:] = "990 Unmapped"
    none = infer_regions(xy3, names3, np.ones(len(xy3), bool), _shares(), two)
    assert none.n_voting_neurons == 0 and none.n_assigned_tiles == 0


def test_majority_ties_go_to_the_first_division_in_sorted_order() -> None:
    xy = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    names = np.array(
        ["190 MB Glut A", "190 MB Glut A", "011 CA1 Glut", "011 CA1 Glut"],
        dtype=object,
    )
    inference = infer_regions(
        xy, names, np.ones(4, bool), _shares(), MouseRegionConfig(min_component_tiles=1)
    )
    assert inference.tiles["region"].tolist() == ["HPF"]  # "HPF" < "MB"


def test_confident_neurons_use_the_vocab_and_float32_bp() -> None:
    bp = np.array([0.9, 0.9, 0.89, 0.95, 1.0], dtype=np.float32)
    mask = confident_neurons(
        ["01 IT-ET Glut", "30 Astro-Epen", "01 IT-ET Glut", "01 IT-ET Glut", None],
        bp,
        np.array([0.9, 1.0, 1.0, 0.85, 1.0], dtype=np.float32),
        min_bp=0.9,
    )
    assert mask.tolist() == [True, False, False, False, False]


# --------------------------------------------------------------------------
# The per-section plan: guard, override and none


def _plan_inputs(
    blocks: dict[str, list[tuple[int, int]]],
) -> dict[str, Any]:
    xy, names, _ = _section(blocks)
    class_of = {
        "007 L2/3 IT CTX Glut": "01 IT-ET Glut",
        "011 CA1 Glut": "01 IT-ET Glut",
        "190 MB Glut A": "19 MB Glut",
        "240 MY Glut A": "24 MY Glut",
        "150 HY Gnrh1 Glut": "15 HY Gnrh1 Glut",
    }
    classes = np.array([class_of[name] for name in names], dtype=object)
    return {
        "cell_ids": [f"c{index}" for index in range(len(xy))],
        "class_names": classes,
        "class_labels": np.array([_label(name, CLAS) for name in classes]),
        "class_bp": np.ones(len(xy)),
        "subclass_names": names,
        "subclass_labels": np.array([_label(name, SUBC) for name in names]),
        "subclass_bp": np.ones(len(xy)),
        "xy": xy,
    }


# 200 assigned tiles: Isocortex, MB and one MY tile (0.5%: not present).
BLOCKS_200 = {
    "Isocortex": _rect(0, 10, 0, 10),
    "MB": [*_rect(10, 20, 0, 10)][:99],
    "MY": [(19, 9)],
}


def test_auto_plan_prunes_and_marks_the_dropped_cells() -> None:
    inputs = _plan_inputs(BLOCKS_200)
    plan = plan_region_step(
        SectionRegionsRequest.parse("auto"),
        shares=_shares(),
        taxonomy=_taxonomy(),
        config=MouseRegionConfig(),
        **inputs,
    )
    assert plan.status == "pruned"
    assert plan.inference is not None and plan.inference.n_assigned_tiles == 200
    assert plan.present_regions == ("Isocortex", "MB")
    names = {node.name for node in plan.nodes}
    assert {"240 MY Glut A", "27 MY GABA", "008 L2/3 IT ENT Glut"} <= names
    assert "011 CA1 Glut" in names  # HPF is not present in this section
    # The MY cells (subclass 240 dropped; class 24 keeps 242) are re-mapped.
    levels = pd.Series(plan.dropped_level, index=plan.cell_ids)
    my_cells = inputs["class_names"] == "24 MY Glut"
    assert set(levels[my_cells]) == {"subclass"}
    assert set(levels[~my_cells]) == {None}
    assert list(plan.remap_cell_ids) == list(plan.cell_ids[my_cells])
    assert plan.confident.all()


def test_fewer_than_200_assigned_tiles_skip_pruning() -> None:
    blocks = {**BLOCKS_200, "MB": BLOCKS_200["MB"][:98]}
    plan = plan_region_step(
        SectionRegionsRequest.parse("auto"),
        shares=_shares(),
        taxonomy=_taxonomy(),
        config=MouseRegionConfig(),
        **_plan_inputs(blocks),
    )
    assert plan.status == "skipped_few_tiles"
    assert plan.nodes == () and plan.present_regions == ()
    assert "199 assigned tiles" in plan.reasons[0]
    assert all(value is None for value in plan.dropped_level)
    assert plan.inference is not None  # kept for QC


def test_auto_without_coordinates_skips_pruning() -> None:
    inputs = {**_plan_inputs(BLOCKS_200), "xy": None}
    plan = plan_region_step(
        SectionRegionsRequest.parse("auto"),
        shares=_shares(),
        taxonomy=_taxonomy(),
        config=MouseRegionConfig(),
        **inputs,
    )
    assert plan.status == "skipped_no_coordinates"
    assert plan.inference is None and not plan.nodes


def test_override_uses_the_listed_divisions_without_coordinates() -> None:
    inputs = {**_plan_inputs(BLOCKS_200), "xy": None}
    plan = plan_region_step(
        SectionRegionsRequest.parse("Isocortex;HPF;MB;MY"),
        shares=_shares(),
        taxonomy=_taxonomy(),
        config=MouseRegionConfig(),
        **inputs,
    )
    assert plan.status == "pruned"
    assert plan.present_regions == ("Isocortex", "HPF", "MB", "MY")
    names = {node.name for node in plan.nodes}
    assert names == {"008 L2/3 IT ENT Glut"}  # MY listed: class 24 / 27 stay
    # With coordinates the inference still runs and the difference is noted.
    with_xy = plan_region_step(
        SectionRegionsRequest.parse("Isocortex;HPF;MB;MY"),
        shares=_shares(),
        taxonomy=_taxonomy(),
        config=MouseRegionConfig(),
        **_plan_inputs(BLOCKS_200),
    )
    assert with_xy.inference is not None
    assert "differs from the inferred divisions" in with_xy.reasons[0]
    # Everything present: nothing to drop.
    full = plan_region_step(
        SectionRegionsRequest.parse(";".join(MOUSE_CCF_REGIONS)),
        shares=_shares(),
        taxonomy=_taxonomy(),
        config=MouseRegionConfig(),
        **inputs,
    )
    assert full.status == "no_nodes_dropped" and full.nodes == ()


def test_none_disables_pruning_and_needs_no_shares() -> None:
    plan = plan_region_step(
        SectionRegionsRequest.parse("none"),
        shares=None,
        taxonomy=_taxonomy(),
        config=MouseRegionConfig(),
        **_plan_inputs(BLOCKS_200),
    )
    assert plan.status == "disabled" and plan.nodes == ()
    assert plan.inference is None
    with pytest.raises(MouseRegionError, match="needs the wmb_region_share"):
        plan_region_step(
            SectionRegionsRequest.parse("Isocortex"),
            shares=None,
            taxonomy=_taxonomy(),
            config=MouseRegionConfig(),
            **_plan_inputs(BLOCKS_200),
        )
    with pytest.raises(MouseRegionError, match="coupled_regions"):
        plan_region_step(
            SectionRegionsRequest.parse("none"),
            shares=None,
            taxonomy=_taxonomy(),
            config=MouseRegionConfig(coupled_regions={"OB": ["OLF"]}),
            **_plan_inputs(BLOCKS_200),
        )


def test_dropped_levels_prefer_the_class() -> None:
    nodes = two_tier_drop_list(PRESENT, _shares(), _taxonomy(), MouseRegionConfig())
    levels = dropped_levels(
        ["CCN20230722_CLAS_27", "CCN20230722_CLAS_24", "CCN20230722_CLAS_24", None],
        ["CCN20230722_SUBC_270", "CCN20230722_SUBC_241", "CCN20230722_SUBC_242", None],
        nodes,
    )
    assert levels.tolist() == ["class", "subclass", None, None]


# --------------------------------------------------------------------------
# Lookup filter + nodes_to_drop without a zero-marker error


def _lookup() -> dict[str, Any]:
    tree = _tree()
    lookup: dict[str, Any] = {"metadata": {"source": "test"}, "log": ["x"]}
    for index, (level, node) in enumerate(tree.parents()):
        key = lookup_key(level, node)
        # The class-27 parent's markers are genes the query lacks.
        lookup[key] = (
            ["gMissing1", "gMissing2"]
            if node == "CCN20230722_CLAS_27"
            else [f"g{index}", f"g{index + 1}"]
        )
    return lookup


def test_pruned_lookup_removes_the_dropped_keys() -> None:
    nodes = two_tier_drop_list(PRESENT, _shares(), _taxonomy(), MouseRegionConfig())
    lookup = _lookup()
    genes = sorted(
        {
            gene
            for key, values in lookup.items()
            if key not in {"metadata", "log"}
            for gene in values
            if not gene.startswith("gMissing")
        }
    )
    validation = pruned_lookup(lookup, _tree(), nodes, genes)

    keys = set(validation.lookup)
    assert "CCN20230722_CLAS/CCN20230722_CLAS_27" not in keys
    assert "CCN20230722_SUBC/CCN20230722_SUBC_270" not in keys
    assert "CCN20230722_SUBC/CCN20230722_SUBC_241" not in keys
    assert "CCN20230722_CLAS/CCN20230722_CLAS_24" in keys
    assert "CCN20230722_SUBC/CCN20230722_SUBC_242" in keys
    assert "CCN20230722_CLAS/CCN20230722_CLAS_27" in validation.dropped_keys
    assert "log" not in keys and "metadata" in keys
    assert validation.collapsed == []
    assert all(
        set(values) <= set(genes)
        for key, values in validation.lookup.items()
        if key != "metadata"
    )


def _ctm_tree(tree: TaxonomyTreeView) -> Any:
    from cell_type_mapper.taxonomy.taxonomy_tree import TaxonomyTree

    payload = tree.to_json()
    payload.pop("name_mapper", None)
    leaf = tree.leaf_level
    payload[leaf] = {name: [f"cell_{name}"] for name in payload[leaf]}
    return TaxonomyTree(data=payload)


def test_ctm_accepts_the_pruned_lookup_but_not_the_unfiltered_one(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cell_type_mapper")
    from cell_type_mapper.type_assignment.marker_cache_v2 import (
        create_marker_cache_from_specified_markers,
    )

    nodes = two_tier_drop_list(PRESENT, _shares(), _taxonomy(), MouseRegionConfig())
    lookup = _lookup()
    genes = sorted(
        {
            gene
            for key, values in lookup.items()
            if key not in {"metadata", "log"}
            for gene in values
            if not gene.startswith("gMissing")
        }
    )
    reference_genes = [*genes, "gMissing1", "gMissing2"]
    # ctm's own --nodes_to_drop tree (drop_node_batch) equals ours.
    ctm_pruned = _ctm_tree(_tree()).drop_node_batch([node.as_pair() for node in nodes])
    ours = pruned_tree(_tree(), nodes)
    assert sorted(ctm_pruned.nodes_at_level(SUBC)) == sorted(ours.nodes(SUBC))
    assert sorted(ctm_pruned.nodes_at_level(CLAS)) == sorted(ours.nodes(CLAS))

    unfiltered = {key: value for key, value in lookup.items() if key != "log"}
    with pytest.raises(RuntimeError, match="No markers at parent node"):
        create_marker_cache_from_specified_markers(
            marker_lookup=unfiltered,
            reference_gene_names=reference_genes,
            query_gene_names=genes,
            output_cache_path=tmp_path / "unfiltered.h5",
            log=None,
            taxonomy_tree=ctm_pruned,
            min_markers=1,
        )
    validation = pruned_lookup(lookup, _tree(), nodes, genes)
    create_marker_cache_from_specified_markers(
        marker_lookup=validation.lookup,
        reference_gene_names=reference_genes,
        query_gene_names=genes,
        output_cache_path=tmp_path / "pruned.h5",
        log=None,
        taxonomy_tree=ctm_pruned,
        min_markers=1,
    )
    assert (tmp_path / "pruned.h5").is_file()


# --------------------------------------------------------------------------
# Tidy tables and the per-cell parquet


def _tidy(cells: dict[str, tuple[str, str]], probability: float = 1.0) -> pd.DataFrame:
    rows = []
    for cell, (class_label, subclass_label) in cells.items():
        for level, name, label in (
            (CLAS, "class", class_label),
            (SUBC, "subclass", subclass_label),
        ):
            rows.append(
                {
                    "cell_id": cell,
                    "level": level,
                    "level_name": name,
                    "assignment": label,
                    "name": f"name {label}",
                    "bp": probability,
                    "aggregate_probability": probability,
                    "avg_correlation": 0.5,
                    "directly_assigned": True,
                    "n_runners_up": 0,
                }
            )
    return coerce_tidy_dtypes(pd.DataFrame(rows))


def test_merge_pruned_tidy_replaces_only_the_remapped_cells() -> None:
    unpruned = _tidy({"a": ("C24", "S241"), "b": ("C01", "S007"), "c": ("C27", "S270")})
    remap = _tidy({"c": ("C19", "S190"), "a": ("C24", "S242")}, probability=0.5)

    merged = merge_pruned_tidy(unpruned, remap)

    assert merged["cell_id"].tolist() == ["a", "a", "b", "b", "c", "c"]
    assert merged["assignment"].astype(str).tolist() == [
        "C24",
        "S242",
        "C01",
        "S007",
        "C19",
        "S190",
    ]
    assert merged["bp"].tolist() == [0.5, 0.5, 1.0, 1.0, 0.5, 0.5]
    assert merged["bp"].dtype == np.float32
    assert isinstance(merged["assignment"].dtype, pd.CategoricalDtype)
    class_changed, subclass_changed = changed_cells(
        unpruned,
        merged,
        class_level=CLAS,
        subclass_level=SUBC,
        cell_ids=pd.Index(["a", "b", "c"]),
    )
    assert class_changed.tolist() == [False, False, True]
    assert subclass_changed.tolist() == [True, False, True]
    assert merge_pruned_tidy(unpruned, remap.iloc[0:0]).equals(unpruned)
    with pytest.raises(MouseRegionError, match="absent from the unpruned"):
        merge_pruned_tidy(unpruned, _tidy({"z": ("C19", "S190")}))


def _plan(n: int = 3) -> RegionPlan:
    blocks = {"Isocortex": _rect(0, 1, 0, 1)}
    xy, _, _ = _section(blocks, per_tile=n)
    inference = infer_regions(
        xy,
        np.array(["007 L2/3 IT CTX Glut"] * n, dtype=object),
        np.ones(n, bool),
        _shares(),
        MouseRegionConfig(min_component_tiles=1),
    )
    return RegionPlan(
        request=SectionRegionsRequest.parse("auto"),
        status="pruned",
        reasons=(),
        inference=inference,
        present_regions=("Isocortex",),
        nodes=(),
        cell_ids=pd.Index([f"c{index}" for index in range(n)]),
        dropped_level=np.array(["class", None, "subclass"][:n], dtype=object),
        confident=np.array([True, False, True][:n]),
    )


def test_region_cells_round_trip(tmp_path: Path) -> None:
    plan = _plan()
    cells = region_cells_frame(
        plan,
        class_changed=np.array([True, False, False]),
        subclass_changed=np.array([True, False, True]),
    )
    assert tuple(cells.columns) == REGION_CELL_COLUMNS
    path = write_region_cells(cells, tmp_path / "s_mouse_regions.parquet", {"a": 1})
    read, metadata = read_region_cells(path)
    assert metadata == {"a": 1, "schema_version": 1}
    assert read["inferred_region"].astype(str).tolist() == ["Isocortex"] * 3
    assert read["region_dropped_level"].astype(object).tolist()[1] is None or pd.isna(
        read["region_dropped_level"].iloc[1]
    )
    assert read["region_pruned_changed"].tolist() == [True, False, True]
    assert read["confident_neuron"].tolist() == [True, False, True]
    assert read["tile_i"].tolist() == [0, 0, 0]
    no_xy = RegionPlan(**{**plan.__dict__, "inference": None})
    empty = region_cells_frame(no_xy)
    assert empty["tile_i"].isna().all() and empty["inferred_region"].isna().all()


def test_region_outputs_give_the_label_table_columns(tmp_path: Path) -> None:
    unpruned = _tidy(
        {"c0": ("C24", "S241"), "c1": ("C01", "S007"), "c2": ("C27", "S270")}
    )
    remap = _tidy({"c0": ("C19", "S190"), "c2": ("C19", "S191")}, probability=0.4)
    cells = region_cells_frame(
        _plan(),
        class_changed=np.array([True, False, True]),
        subclass_changed=np.array([True, False, True]),
    )
    from merxen.annotation.mapmycells_engine import write_tidy_parquet
    from merxen.annotation.store import file_sha256

    cells_path = write_region_cells(cells, tmp_path / "s_mouse_regions.parquet", {})
    remap_path = write_tidy_parquet(
        remap, tmp_path / "s_mmc_wmb_panel_pruned.parquet", {}
    )
    record = MouseRegionRecord(
        rule_variant="v1",
        run_id="wmb_panel",
        requested="auto",
        source="auto",
        status="pruned",
        n_table_cells=3,
        cells_parquet=cells_path.name,
        cells_parquet_sha256=file_sha256(cells_path),
        remap={
            "run_id": "wmb_panel_pruned",
            "reference_id": "wmb_panel",
            "build_hash": "0" * 64,
            "bundle_path": "/b",
            "parquet": remap_path.name,
            "parquet_sha256": file_sha256(remap_path),
            "n_cells": 2,
            "n_query_genes": 5,
            "query_fingerprint": "f",
            "lookup_sha256": "l",
            "nodes_to_drop": [[CLAS, "C24"]],
            "engine_params": {},
            "ctm_version": "1.7.2",
        },
    )
    assert MouseRegionRecord.model_validate_json(record.model_dump_json()) == record
    outputs = load_region_outputs(tmp_path, record, unpruned)
    assert isinstance(outputs, MouseRegionOutputs)
    assert outputs.pruned["assignment"].astype(str).tolist()[:2] == ["C19", "S190"]

    obs = pd.Index(["c0", "c1", "c2", "offtable"])
    columns = outputs.engine_columns(
        obs, prefix="mmc_wmb", class_level=CLAS, subclass_level=SUBC
    )
    assert columns["mmc_wmb_unpruned_class_name"].astype(object).tolist()[:3] == [
        "name C24",
        "name C01",
        "name C27",
    ]
    assert pd.isna(columns["mmc_wmb_unpruned_class_name"].iloc[3])
    assert columns["mmc_wmb_unpruned_subclass_bp"].dtype == np.float32
    assert columns["region_pruned_changed"].tolist() == [True, False, True, False]
    assert columns["inferred_region"].astype(object).tolist()[:3] == ["Isocortex"] * 3
    assert outputs.dropped_level(obs).tolist() == ["class", None, "subclass", None]

    cells_path.write_bytes(cells_path.read_bytes() + b"x")
    with pytest.raises(MouseRegionError, match="differs from the sha256"):
        load_region_outputs(tmp_path, record, unpruned)


# --------------------------------------------------------------------------
# The region-share bundle


def _write_bundle(root: Path, **overrides: Any) -> Path:
    share, home = _share_tables()
    bundle_dir = root / "wmb_region_share" / ("a" * 64)
    bundle_dir.mkdir(parents=True)
    share.to_parquet(bundle_dir / "region_share.parquet")
    home.to_parquet(bundle_dir / "region_home.parquet")
    files = [
        {"path": path.name, "size": path.stat().st_size}
        for path in sorted(bundle_dir.iterdir())
    ]
    manifest = {
        "status": "complete",
        "reference_id": "wmb_region_share",
        "build_hash": "a" * 64,
        "files": files,
        **overrides,
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(manifest))
    return bundle_dir


def test_region_share_bundle_reads_a_complete_bundle(tmp_path: Path) -> None:
    bundle = RegionShareBundle.from_dir(_write_bundle(tmp_path / "ok"))
    assert bundle.build_hash == "a" * 64
    assert bundle.provenance()["build_hash"] == "a" * 64
    assert bundle.shares.subclass_home["190 MB Glut A"] == "MB"
    with pytest.raises(MouseRegionError, match="not complete"):
        RegionShareBundle.from_dir(_write_bundle(tmp_path / "p", status="building"))
    with pytest.raises(MouseRegionError, match="not wmb_region_share"):
        RegionShareBundle.from_dir(
            _write_bundle(tmp_path / "w", reference_id="wmb_panel")
        )
    changed = _write_bundle(tmp_path / "c")
    (changed / "region_home.parquet").write_bytes(b"truncated")
    with pytest.raises(MouseRegionError, match="not intact"):
        RegionShareBundle.from_dir(changed)
