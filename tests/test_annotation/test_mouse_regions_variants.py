"""Tests for the M6b region-rule variants on synthetic tiles (plan §7.8).

Variants (a) OB-OLF coupling with structure-level AON homes, (b) and (c)
conditional never-drop nodes, (d) an OB-only tile fraction and (f) the tile
parameters, behind ``MouseRegionConfig`` with v1 as the default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from merxen.annotation.config import (
    MOUSE_CCF_REGIONS,
    ConditionalNeverDrop,
    MouseRegionConfig,
)
from merxen.annotation.mouse_gate import t2_subclasses
from merxen.annotation.mouse_regions import (
    REGION_STRUCTURE_HOME_FILE,
    MouseRegionError,
    RegionShareBundle,
    RegionShares,
    SectionRegionsRequest,
    WmbTaxonomy,
    check_region_shares,
    check_rule_variant,
    effective_never_drop,
    infer_regions,
    plan_region_step,
    two_tier_drop_list,
)
from merxen.annotation.reference import TaxonomyTreeView

CLAS = "CCN20230722_CLAS"
SUBC = "CCN20230722_SUBC"
CLUS = "CCN20230722_CLUS"

# class -> subclass -> (MERFISH grey cells per division, structure home).
TAXONOMY: dict[str, dict[str, tuple[dict[str, int], str]]] = {
    "01 IT-ET Glut": {
        "007 L2/3 IT CTX Glut": ({"Isocortex": 1000}, "MOp"),
        "010 IT AON-TT-DP Glut": ({"OLF": 1000}, "AON"),
        "013 PIR Glut": ({"OLF": 1000}, "PIR"),
        # Share in Isocortex + STR: 0.05 < 0.1 (dropped without (c)).
        "025 CA2-FC-IG Glut": ({"HPF": 950, "Isocortex": 30, "STR": 20}, "CA2"),
    },
    "05 OB-IMN GABA": {
        "050 OB-IMN A": ({"OB": 900, "OLF": 50, "STR": 50}, "MOB"),
        "051 OB-IMN B": ({"OB": 1000}, "MOB"),
    },
    "09 CNU-LGE GABA": {"060 STR D1 Gaba": ({"STR": 1000}, "CP")},
    "30 Astro-Epen": {"300 Astro": ({"STR": 500, "OB": 500}, "CP")},
}
SUBCLASS_CLASS = {
    subclass: class_name
    for class_name, members in TAXONOMY.items()
    for subclass in members
}
TILE = 150.0
M6B = "m6b"
KEEP_B = {
    "level": "class",
    "name": "05 OB-IMN GABA",
    "when_any_present": ["STR", "OLF"],
}
KEEP_C = {
    "level": "subclass",
    "name": "025 CA2-FC-IG Glut",
    "when_all_present": ["Isocortex", "STR"],
}


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
    return TaxonomyTreeView.from_tree_dict(
        {
            "hierarchy": [CLAS, SUBC, CLUS],
            CLAS: classes,
            SUBC: subclasses,
            CLUS: {leaf: [] for kids in subclasses.values() for leaf in kids},
            "name_mapper": names,
        }
    )


def _taxonomy() -> WmbTaxonomy:
    return WmbTaxonomy.from_tree(_tree(), class_level=CLAS, subclass_level=SUBC)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    share_rows: list[dict[str, Any]] = []
    home_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []

    def add(level: str, name: str, counts: dict[str, int]) -> None:
        total = sum(counts.values())
        for region in MOUSE_CCF_REGIONS:
            share_rows.append(
                {
                    "level": level,
                    "node_name": name,
                    "region": region,
                    "n": counts.get(region, 0),
                    "share": counts.get(region, 0) / total,
                    "n_grey": total,
                    "n_all": total,
                }
            )

    for class_name, members in TAXONOMY.items():
        class_counts: dict[str, int] = {}
        for subclass_name, (counts, structure) in members.items():
            for region, value in counts.items():
                class_counts[region] = class_counts.get(region, 0) + value
            add("subclass", subclass_name, counts)
            home = max(counts, key=lambda region: counts[region])
            total = sum(counts.values())
            home_rows.append(
                {
                    "level": "subclass",
                    "node_name": subclass_name,
                    "home": home,
                    "home_share": counts[home] / total,
                    "n_grey": total,
                }
            )
            structure_rows.append(
                {
                    "level": "subclass",
                    "node_name": subclass_name,
                    "home": structure,
                    "home_share": 1.0,
                    "n_grey": total,
                }
            )
        add("class", class_name, class_counts)
    return (
        pd.DataFrame(share_rows),
        pd.DataFrame(home_rows),
        pd.DataFrame(structure_rows),
    )


def _shares(*, structures: bool = True) -> RegionShares:
    share, home, structure = _tables()
    return RegionShares.from_tables(share, home, structure if structures else None)


def _rect(i0: int, i1: int, j0: int, j1: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(i0, i1) for j in range(j0, j1)]


def _section(
    blocks: dict[str, list[tuple[int, int]]],
    *,
    per_tile: int = 3,
    tile_um: float = TILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``per_tile`` confident neurons of a subclass in each listed tile."""
    xy, names = [], []
    for subclass, tiles in blocks.items():
        for i, j in tiles:
            for k in range(per_tile):
                xy.append(((i + 0.1 + 0.25 * k) * tile_um, (j + 0.5) * tile_um))
                names.append(subclass)
    return np.array(xy), np.array(names, dtype=object), np.ones(len(xy), bool)


def _config(**settings: Any) -> MouseRegionConfig:
    return MouseRegionConfig.model_validate({"rule_variant": M6B, **settings})


def _names(nodes: list[Any]) -> set[str]:
    return {node.name for node in nodes}


# --------------------------------------------------------------------------
# Config and variant checks


def test_v1_is_the_default_and_refuses_m6b_fields() -> None:
    v1 = MouseRegionConfig()
    assert v1.rule_variant == "v1"
    assert v1.uses_m6b_fields() == []
    assert v1.coupling_home_structures == {"OB": ["AON"]}
    assert v1.tile_fraction_for("OB") == 0.01
    check_rule_variant(v1)
    for settings in (
        {"coupled_regions": {"OB": ["OLF"]}},
        {"conditional_never_drop": [KEEP_B]},
        {"region_min_tile_fraction": {"OB": 0.005}},
    ):
        config = MouseRegionConfig.model_validate(settings)
        (field,) = config.uses_m6b_fields()
        with pytest.raises(MouseRegionError, match=field):
            check_rule_variant(config)
        check_rule_variant(_config(**settings))  # m6b accepts it
    # The tile parameters (f) are v1 fields.
    check_rule_variant(MouseRegionConfig(tile_um=100.0, min_neurons_per_tile=2))
    with pytest.raises(MouseRegionError, match="not implemented"):
        check_rule_variant(MouseRegionConfig(rule_variant="v2"))


def test_variant_settings_are_validated() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ConditionalNeverDrop(level="class", name="05 OB-IMN GABA")
    with pytest.raises(ValidationError, match="exactly one"):
        ConditionalNeverDrop.model_validate({**KEEP_B, "when_all_present": ["STR"]})
    with pytest.raises(ValidationError, match="unknown regions"):
        ConditionalNeverDrop.model_validate({**KEEP_B, "when_any_present": ["CP"]})
    with pytest.raises(ValidationError, match="unknown regions"):
        MouseRegionConfig(region_min_tile_fraction={"AON": 0.005})
    with pytest.raises(ValidationError, match="must lie in"):
        MouseRegionConfig(region_min_tile_fraction={"OB": 1.5})
    with pytest.raises(ValidationError, match="coupling_home_structures"):
        MouseRegionConfig(coupled_regions={"OB": ["OLF"]}, coupling_home_structures={})
    with pytest.raises(ValidationError, match="other divisions"):
        MouseRegionConfig(coupled_regions={"OB": ["OB"]})
    config = _config(region_min_tile_fraction={"OB": 0.005})
    assert config.tile_fraction_for("OB") == 0.005
    assert config.tile_fraction_for("HY") == 0.01


def test_conditional_nodes_apply_on_their_regions() -> None:
    any_node = ConditionalNeverDrop.model_validate(KEEP_B)
    all_node = ConditionalNeverDrop.model_validate(KEEP_C)
    assert any_node.applies({"OLF"}) and any_node.applies(["STR", "MB"])
    assert not any_node.applies({"Isocortex", "OB"})
    assert all_node.applies({"Isocortex", "STR", "PAL"})
    assert not all_node.applies({"Isocortex"})
    config = _config(conditional_never_drop=[KEEP_B, KEEP_C])
    classes, subclasses = effective_never_drop(config, {"Isocortex", "STR"})
    assert "05 OB-IMN GABA" in classes and "30 Astro-Epen" in classes
    assert subclasses == {"025 CA2-FC-IG Glut"}
    classes, subclasses = effective_never_drop(config, {"Isocortex"})
    assert "05 OB-IMN GABA" not in classes and not subclasses


# --------------------------------------------------------------------------
# (a) coupling with structure-level AON homes


# 300 Isocortex tiles, 6 AON-home OLF tiles (2%), 1 OB tile (not present).
AON_BLOCKS = {
    "007 L2/3 IT CTX Glut": _rect(0, 20, 0, 15),
    "010 IT AON-TT-DP Glut": _rect(0, 6, 15, 16),
    "051 OB-IMN B": [(10, 15)],
}


def test_coupling_adds_ob_with_olf_and_aon_home_tiles() -> None:
    xy, names, confident = _section(AON_BLOCKS)
    v1 = infer_regions(xy, names, confident, _shares(), MouseRegionConfig())
    assert v1.inferred_regions == ("Isocortex", "OLF")
    assert v1.coupled_regions == () and v1.coupling_tile_counts == {}

    coupled = infer_regions(
        xy, names, confident, _shares(), _config(coupled_regions={"OB": ["OLF"]})
    )
    assert coupled.n_assigned_tiles == 307
    assert coupled.inferred_regions == ("Isocortex", "OLF", "OB")
    assert coupled.coupled_regions == ("OB",)
    assert coupled.coupling_tile_counts == {"OB": 6}
    # The tile majorities themselves are unchanged (division level).
    assert coupled.tile_counts == v1.tile_counts


def test_coupling_needs_aon_tiles_and_a_present_partner() -> None:
    config = _config(coupled_regions={"OB": ["OLF"]})
    # OLF present through piriform tiles only: 2 AON tiles of 308 (< 1%).
    blocks = {
        "007 L2/3 IT CTX Glut": _rect(0, 20, 0, 15),
        "013 PIR Glut": _rect(0, 6, 15, 16),
        "010 IT AON-TT-DP Glut": [(10, 15), (11, 15)],
    }
    xy, names, confident = _section(blocks)
    inference = infer_regions(xy, names, confident, _shares(), config)
    assert "OLF" in inference.inferred_regions
    assert "OB" not in inference.inferred_regions
    assert inference.coupling_tile_counts == {"OB": 2}
    # Exactly 1% of the assigned tiles is enough (>=).
    blocks["010 IT AON-TT-DP Glut"] = [(10, 15), (11, 15), (12, 15)]
    xy, names, confident = _section(blocks)
    inference = infer_regions(xy, names, confident, _shares(), config)
    assert inference.n_assigned_tiles == 309
    assert "OB" not in inference.inferred_regions  # 3 / 309 < 1%
    loose = _config(coupled_regions={"OB": ["OLF"]}, coupling_min_tile_fraction=3 / 309)
    assert "OB" in infer_regions(xy, names, confident, _shares(), loose).coupled_regions
    # AON tiles but no present partner: OLF below 1% of the tiles.
    small = {
        "007 L2/3 IT CTX Glut": _rect(0, 20, 0, 15),
        "010 IT AON-TT-DP Glut": [(0, 15), (1, 15)],
    }
    xy, names, confident = _section(small)
    inference = infer_regions(xy, names, confident, _shares(), config)
    assert inference.inferred_regions == ("Isocortex",)
    assert inference.coupling_tile_counts == {}


def test_structure_tiles_need_their_own_voters() -> None:
    # Six OLF tiles, each with two AON-home neurons and one piriform neuron
    # whose subclass has no structure-level home: the tiles are assigned
    # (three division voters) but have only two structure voters (< 3).
    xy, names, confident = _section({"007 L2/3 IT CTX Glut": _rect(0, 20, 0, 15)})
    extra_xy, extra_names = [], []
    for i in range(6):
        for k, name in enumerate(
            ["010 IT AON-TT-DP Glut", "010 IT AON-TT-DP Glut", "013 PIR Glut"]
        ):
            extra_xy.append(((i + 0.1 + 0.25 * k) * TILE, 15.5 * TILE))
            extra_names.append(name)
    xy = np.vstack([xy, np.array(extra_xy)])
    names = np.concatenate([names, np.array(extra_names, dtype=object)])
    confident = np.ones(len(xy), bool)
    share, home, structure = _tables()
    shares = RegionShares.from_tables(
        share, home, structure[structure["node_name"] != "013 PIR Glut"]
    )
    config = _config(coupled_regions={"OB": ["OLF"]})
    inference = infer_regions(xy, names, confident, shares, config)
    assert inference.tile_counts["OLF"] == 6
    assert "OLF" in inference.inferred_regions
    assert inference.coupling_tile_counts == {"OB": 0}
    assert "OB" not in inference.inferred_regions
    # Two structure neurons per tile suffice at a 2-neuron minimum.
    two = _config(coupled_regions={"OB": ["OLF"]}, min_neurons_per_tile=2)
    assert infer_regions(xy, names, confident, shares, two).coupled_regions == ("OB",)


def test_coupling_refuses_shares_without_structure_homes() -> None:
    xy, names, confident = _section(AON_BLOCKS)
    config = _config(coupled_regions={"OB": ["OLF"]})
    with pytest.raises(MouseRegionError, match="structure-level"):
        infer_regions(xy, names, confident, _shares(structures=False), config)
    with pytest.raises(MouseRegionError, match="structure-level"):
        check_region_shares(config, _shares(structures=False))
    check_region_shares(MouseRegionConfig(), _shares(structures=False))


def test_coupled_plan_keeps_the_ob_types_of_an_aon_section() -> None:
    xy, names, confident = _section(AON_BLOCKS)
    classes = np.array([SUBCLASS_CLASS[name] for name in names], dtype=object)
    inputs: dict[str, Any] = {
        "cell_ids": [f"c{index}" for index in range(len(xy))],
        "class_names": classes,
        "class_labels": np.array([_label(name, CLAS) for name in classes]),
        "class_bp": np.ones(len(xy)),
        "subclass_names": names,
        "subclass_labels": np.array([_label(name, SUBC) for name in names]),
        "subclass_bp": np.ones(len(xy)),
        "xy": xy,
    }
    common = {"shares": _shares(), "taxonomy": _taxonomy(), **inputs}
    request = SectionRegionsRequest.parse("auto")
    v1 = plan_region_step(request, config=MouseRegionConfig(), **common)
    assert v1.present_regions == ("Isocortex", "OLF")
    assert "05 OB-IMN GABA" in _names(list(v1.nodes))
    ob_cells = names == "051 OB-IMN B"
    assert all(level == "class" for level in v1.dropped_level[ob_cells])
    coupled = plan_region_step(
        request, config=_config(coupled_regions={"OB": ["OLF"]}), **common
    )
    assert coupled.present_regions == ("Isocortex", "OLF", "OB")
    assert "05 OB-IMN GABA" not in _names(list(coupled.nodes))
    assert not any(level is not None for level in coupled.dropped_level[ob_cells])


# --------------------------------------------------------------------------
# (b), (c) conditional never-drop


def test_ob_imn_is_kept_when_str_or_olf_is_present() -> None:
    taxonomy = _taxonomy()
    v1 = two_tier_drop_list(
        ("Isocortex", "STR"), _shares(), taxonomy, MouseRegionConfig()
    )
    assert "05 OB-IMN GABA" in _names(v1)  # absent class, both subclasses < .3
    keep = _config(conditional_never_drop=[KEEP_B])
    for present in (("Isocortex", "STR"), ("Isocortex", "OLF")):
        nodes = two_tier_drop_list(present, _shares(), taxonomy, keep)
        assert not {"05 OB-IMN GABA", "050 OB-IMN A", "051 OB-IMN B"} & _names(nodes)
    # Neither STR nor OLF: the condition fails and the class is dropped.
    nodes = two_tier_drop_list(("Isocortex", "HPF"), _shares(), taxonomy, keep)
    assert "05 OB-IMN GABA" in _names(nodes)
    # Everything else is unchanged.
    assert _names(v1) - {"05 OB-IMN GABA"} == _names(
        two_tier_drop_list(("Isocortex", "STR"), _shares(), taxonomy, keep)
    )


def test_ig_subclass_is_kept_when_isocortex_and_str_are_present() -> None:
    taxonomy = _taxonomy()
    present = ("Isocortex", "STR")
    v1 = two_tier_drop_list(present, _shares(), taxonomy, MouseRegionConfig())
    assert "025 CA2-FC-IG Glut" in _names(v1)
    keep = _config(conditional_never_drop=[KEEP_C])
    kept = two_tier_drop_list(present, _shares(), taxonomy, keep)
    assert "025 CA2-FC-IG Glut" not in _names(kept)
    assert _names(v1) - {"025 CA2-FC-IG Glut"} == _names(kept)
    # Isocortex without STR: dropped.
    assert "025 CA2-FC-IG Glut" in _names(
        two_tier_drop_list(("Isocortex", "OLF"), _shares(), taxonomy, keep)
    )


def test_a_kept_subclass_stops_its_class_being_dropped_whole() -> None:
    taxonomy = _taxonomy()
    keep = _config(
        conditional_never_drop=[
            {"level": "subclass", "name": "051 OB-IMN B", "when_any_present": ["STR"]}
        ]
    )
    nodes = two_tier_drop_list(("Isocortex", "STR"), _shares(), taxonomy, keep)
    names = _names(nodes)
    assert "05 OB-IMN GABA" not in names  # not every subclass is dropped
    assert "050 OB-IMN A" in names and "051 OB-IMN B" not in names


def test_t2_can_leave_the_kept_nodes_out() -> None:
    # G3 and MO2 pass the static list (kept nodes count); the M6b scoring's
    # diagnostic T2 leaves the section's kept nodes out as well.
    shares = _shares()
    present = ("Isocortex", "STR")
    base = t2_subclasses(
        present,
        shares,
        subclass_class=SUBCLASS_CLASS,
        never_drop_classes=["30 Astro-Epen"],
    )
    assert {"025 CA2-FC-IG Glut", "050 OB-IMN A", "051 OB-IMN B"} <= base
    config = _config(conditional_never_drop=[KEEP_B, KEEP_C])
    classes, subclasses = effective_never_drop(config, present)
    kept = t2_subclasses(
        present,
        shares,
        subclass_class=SUBCLASS_CLASS,
        never_drop_classes=sorted(classes),
        never_drop_subclasses=sorted(subclasses),
    )
    assert base - kept == {"025 CA2-FC-IG Glut", "050 OB-IMN A", "051 OB-IMN B"}


# --------------------------------------------------------------------------
# (d) OB-only tile fraction; (f) tile parameters


def test_ob_only_tile_fraction() -> None:
    # 400 tiles: OB and PIR (OLF) each hold 3 connected tiles (0.75%).
    blocks = {
        "007 L2/3 IT CTX Glut": [*_rect(0, 20, 0, 19), *_rect(0, 14, 19, 20)],
        "051 OB-IMN B": [(14, 19), (15, 19), (16, 19)],
        "013 PIR Glut": [(17, 19), (18, 19), (19, 19)],
    }
    xy, names, confident = _section(blocks)
    v1 = infer_regions(xy, names, confident, _shares(), MouseRegionConfig())
    assert v1.n_assigned_tiles == 400
    assert v1.inferred_regions == ("Isocortex",)
    ob = infer_regions(
        xy, names, confident, _shares(), _config(region_min_tile_fraction={"OB": 0.005})
    )
    assert ob.inferred_regions == ("Isocortex", "OB")  # OLF keeps the 1% rule
    # The component rule still applies to OB.
    scattered = dict(blocks, **{"051 OB-IMN B": [(14, 19), (16, 19), (18, 19)]})
    scattered["013 PIR Glut"] = [(15, 19), (17, 19), (19, 19)]
    xy, names, confident = _section(scattered)
    assert infer_regions(
        xy, names, confident, _shares(), _config(region_min_tile_fraction={"OB": 0.005})
    ).inferred_regions == ("Isocortex",)


@pytest.mark.parametrize(
    ("tile_um", "min_neurons", "expected"),
    [
        (100.0, 2, 400),  # one 100 µm tile per planted tile, 2 neurons each
        (100.0, 3, 0),
        (200.0, 3, 100),  # 2 x 2 planted tiles, 8 neurons each
        (200.0, 2, 100),
        (150.0, 3, None),  # uneven: counted directly below
        (150.0, 2, None),
    ],
)
def test_tile_parameters_change_the_assigned_tiles(
    tile_um: float, min_neurons: int, expected: int | None
) -> None:
    # 20 x 20 planted 100 µm tiles with two confident neurons each.
    xy, names, confident = _section(
        {"007 L2/3 IT CTX Glut": _rect(0, 20, 0, 20)}, per_tile=2, tile_um=100.0
    )
    _, per_tile = np.unique(
        np.floor(xy / tile_um).astype(int), axis=0, return_counts=True
    )
    counted = int((per_tile >= min_neurons).sum())
    if expected is not None:
        assert counted == expected
    config = MouseRegionConfig(tile_um=tile_um, min_neurons_per_tile=min_neurons)
    inference = infer_regions(xy, names, confident, _shares(), config)
    assert inference.n_assigned_tiles == counted
    assert inference.inferred_regions == (("Isocortex",) if counted else ())
    assert int(inference.tiles["n_neurons"].sum()) == int(
        per_tile[per_tile >= min_neurons].sum()
    )


# --------------------------------------------------------------------------
# Bundle reading


def test_region_share_bundle_reads_structure_homes_when_listed(tmp_path: Path) -> None:
    share, home, structure = _tables()
    directory = tmp_path / "wmb_region_share" / ("f" * 64)
    directory.mkdir(parents=True)
    files = []
    for name, frame in (
        ("region_share.parquet", share),
        ("region_home.parquet", home),
        (REGION_STRUCTURE_HOME_FILE, structure),
    ):
        frame.to_parquet(directory / name)
        files.append({"path": name, "size": (directory / name).stat().st_size})

    def write(listed: list[dict[str, Any]]) -> None:
        (directory / "bundle.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "reference_id": "wmb_region_share",
                    "build_hash": "f" * 64,
                    "files": listed,
                }
            )
        )

    write(files)
    bundle = RegionShareBundle.from_dir(directory)
    assert bundle.shares.subclass_structure_home is not None
    assert bundle.shares.subclass_structure_home["010 IT AON-TT-DP Glut"] == "AON"
    write(files[:2])  # a v1 bundle: no structure homes
    assert RegionShareBundle.from_dir(directory).shares.subclass_structure_home is None
