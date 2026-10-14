"""Tests for the M6b AP-axis scoring script (pre-registration §24)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.config import MOUSE_CCF_REGIONS
from merxen.annotation.mouse_regions import (
    DroppedNode,
    MouseRegionRecord,
    RegionShares,
    WmbTaxonomy,
)
from merxen.annotation.reference import (
    REGION_HOME_FILE,
    REGION_SHARE_FILE,
    TaxonomyTreeView,
    region_share_tables,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "acceptance"
CLAS = "CCN20230722_CLAS"
SUBC = "CCN20230722_SUBC"
CLUS = "CCN20230722_CLUS"


def _load() -> ModuleType:
    name = "mouse_ap_axis"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Candidates


def test_candidates_are_v1_and_the_95_combinations() -> None:
    module = _load()
    candidates = module.all_candidates()
    assert len(candidates) == 96
    assert candidates[0].label == "v1" and candidates[0].is_v1
    labels = [candidate.label for candidate in candidates]
    assert len(set(labels)) == 96
    assert sum(not candidate.is_v1 for candidate in candidates) == 95
    make = module.Candidate
    assert make(("a", "b"), 150.0, 3).label == "a+b"
    assert make(("a", "b"), 100.0, 3).label == "a+b+f:tile100"
    assert make((), 200.0, 2).label == "f:tile200,nn2"
    assert make(("a", "b", "c", "d"), 100.0, 2).n_changes == 6
    assert make((), 150.0, 2).n_changes == 1
    assert module.candidate_by_label("a+b+f:tile100") == make(("a", "b"), 100.0, 3)
    with pytest.raises(KeyError):
        module.candidate_by_label("e")


def test_candidate_configs_set_the_m6b_fields() -> None:
    from merxen.annotation.config import MouseRegionConfig

    module = _load()
    v1 = module.candidate_by_label("v1").config()
    assert v1.rule_variant == "v1" and v1.uses_m6b_fields() == []
    # The candidates are built on the v1 rule, not on the defaults, which
    # are the selected candidate since M6b-R.
    assert v1 == MouseRegionConfig.v1()
    assert (v1.tile_um, v1.min_neurons_per_tile) == (150.0, 3)
    assert module.candidate_by_label("a+f:tile200,nn2").config() == MouseRegionConfig()
    tiles = module.candidate_by_label("f:tile200,nn2").config()
    assert (tiles.tile_um, tiles.min_neurons_per_tile, tiles.rule_variant) == (
        200.0,
        2,
        "v1",
    )
    full = module.candidate_by_label("a+b+c+d").config()
    assert full.rule_variant == "m6b"
    assert full.coupled_regions == {"OB": ["OLF"]}
    assert full.coupling_home_structures == {"OB": ["AON"]}
    assert full.region_min_tile_fraction == {"OB": 0.005}
    assert [(node.level, node.name) for node in full.conditional_never_drop] == [
        ("class", "05 OB-IMN GABA"),
        ("subclass", "025 CA2-FC-IG Glut"),
    ]
    assert full.conditional_never_drop[0].when_any_present == ["STR", "OLF"]
    assert full.conditional_never_drop[1].when_all_present == ["Isocortex", "STR"]
    # Every other rule parameter keeps its M6 value.
    assert full.min_tile_fraction == 0.01 and full.min_component_tiles == 3
    assert full.subclass_share_default == 0.1


# --------------------------------------------------------------------------
# A planted-sliver section (§7.8: the missed OB sliver next to the AON)

# class -> subclass -> (MERFISH grey cells per division, structure home).
TAXONOMY: dict[str, dict[str, tuple[dict[str, int], str]]] = {
    "01 IT-ET Glut": {
        "007 L2/3 IT CTX Glut": ({"Isocortex": 1000}, "MOp"),
        "010 IT AON-TT-DP Glut": ({"OLF": 1000}, "AON"),
    },
    "05 OB-IMN GABA": {
        "050 OB-IMN A": ({"OB": 900, "OLF": 50, "STR": 50}, "MOB"),
        "051 OB-IMN B": ({"OB": 1000}, "MOB"),
    },
    "24 MY Glut": {"240 MY Glut A": ({"MY": 1000}, "MY-unassigned")},
    "30 Astro-Epen": {"300 Astro NN": ({"Isocortex": 500, "OB": 500}, "MOp")},
}


def _label(name: str, level: str) -> str:
    return f"{level}_{name.split()[0]}"


def _taxonomy() -> WmbTaxonomy:
    classes: dict[str, list[str]] = {}
    subclasses: dict[str, list[str]] = {}
    names: dict[str, dict[str, dict[str, str]]] = {CLAS: {}, SUBC: {}, CLUS: {}}
    for class_name, members in TAXONOMY.items():
        label = _label(class_name, CLAS)
        names[CLAS][label] = {"name": class_name}
        classes[label] = []
        for sub in members:
            sub_label = _label(sub, SUBC)
            names[SUBC][sub_label] = {"name": sub}
            classes[label].append(sub_label)
            subclasses[sub_label] = [f"{sub_label}_leaf"]
            names[CLUS][f"{sub_label}_leaf"] = {"name": f"{sub} leaf"}
    tree = TaxonomyTreeView.from_tree_dict(
        {
            "hierarchy": [CLAS, SUBC, CLUS],
            CLAS: classes,
            SUBC: subclasses,
            CLUS: {leaf: [] for kids in subclasses.values() for leaf in kids},
            "name_mapper": names,
        }
    )
    return WmbTaxonomy.from_tree(tree, class_level=CLAS, subclass_level=SUBC)


def _shares() -> RegionShares:
    share_rows, home_rows, structure_rows = [], [], []
    for class_name, members in TAXONOMY.items():
        class_counts: dict[str, int] = {}
        for sub, (counts, structure) in members.items():
            total = sum(counts.values())
            for region, value in counts.items():
                class_counts[region] = class_counts.get(region, 0) + value
            for region in MOUSE_CCF_REGIONS:
                share_rows.append(
                    {
                        "level": "subclass",
                        "node_name": sub,
                        "region": region,
                        "n": counts.get(region, 0),
                        "share": counts.get(region, 0) / total,
                        "n_grey": total,
                        "n_all": total,
                    }
                )
            home = max(counts, key=lambda region: counts[region])
            home_rows.append(
                {"level": "subclass", "node_name": sub, "home": home, "n_grey": total}
            )
            structure_rows.append(
                {"level": "subclass", "node_name": sub, "home": structure}
            )
        total = sum(class_counts.values())
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
    return RegionShares.from_tables(
        pd.DataFrame(share_rows), pd.DataFrame(home_rows), pd.DataFrame(structure_rows)
    )


def _sliver_dataset(module: ModuleType, *, cortex_columns: int = 20) -> Any:
    """300 Isocortex tiles, 6 AON tiles holding OB-IMN cells, a 1-tile OB sliver."""
    tile = 150.0
    cells: list[tuple[float, float, str]] = []
    for i in range(cortex_columns):
        for j in range(15):
            cells += [
                ((i + 0.2 + 0.25 * k) * tile, (j + 0.5) * tile, "007 L2/3 IT CTX Glut")
                for k in range(3)
            ]
    for i in range(6):
        cells += [
            ((i + 0.2 + 0.25 * k) * tile, 15.5 * tile, "010 IT AON-TT-DP Glut")
            for k in range(3)
        ]
        # Two OB-IMN neuroblasts per AON tile (bp 0.5: they do not vote).
        cells += [((i + 0.5) * tile, 15.2 * tile, "050 OB-IMN A") for _ in range(2)]
    cells += [((10.2 + 0.25 * k) * tile, 15.5 * tile, "051 OB-IMN B") for k in range(3)]
    cells += [((3.5) * tile, (3.5) * tile, "300 Astro NN") for _ in range(10)]
    # Five MY sink calls in cortex (class bp 0.5: no harm, no vote), dropped
    # under the inferred and the CCF regions alike (not oracle harm).
    cells += [((5.5) * tile, (5.5) * tile, "240 MY Glut A") for _ in range(5)]
    sub_class = {sub: cls for cls, subs in TAXONOMY.items() for sub in subs}
    subclasses = np.array([name for _, _, name in cells], dtype=object)
    classes = np.array([sub_class[name] for name in subclasses], dtype=object)
    bp = np.where(np.isin(subclasses, ["050 OB-IMN A", "240 MY Glut A"]), 0.5, 1.0)
    n = len(cells)
    return module.Dataset(
        key="merfish500/S57",
        kind="merfish",
        gene_set="merfish500",
        section="S57",
        map_dir="",
        bundle_path="",
        cell_ids=pd.Index([f"c{index}" for index in range(n)]),
        class_names=classes,
        class_labels=np.array([_label(name, CLAS) for name in classes], dtype=object),
        class_bp=np.select(
            [subclasses == "050 OB-IMN A", subclasses == "240 MY Glut A"],
            [0.95, 0.5],
            1.0,
        ),
        subclass_names=subclasses,
        subclass_labels=np.array(
            [_label(name, SUBC) for name in subclasses], dtype=object
        ),
        subclass_bp=bp,
        xy=np.array([(x, y) for x, y, _ in cells]),
        allen_class=classes.astype(str),
        allen_subclass=subclasses.astype(str),
        acs=np.full(n, 0.7),
    )


def test_direct_harm_counts_the_missed_sliver_and_coupling_removes_it() -> None:
    module = _load()
    data = _sliver_dataset(module)
    n = len(data.cell_ids)
    ccf = {"Isocortex", "OLF", "OB"}
    v1_row, v1_nodes, v1_remap = module.evaluate_candidate(
        data,
        _shares(),
        _taxonomy(),
        module.candidate_by_label("v1"),
        ccf_present=ccf,
        ccf_any=ccf,
    )
    assert v1_row["status"] == "pruned"
    assert v1_row["inferred"] == "Isocortex;OLF"
    assert v1_row["missed"] == "OB" and v1_row["extra"] == ""
    assert "05 OB-IMN GABA" in {node.name for node in v1_nodes}
    # 12 neuroblasts + 3 sliver cells, all confident Allen cells (bp >= .9).
    assert v1_row["n_harm"] == 15
    assert v1_row["harm"] == pytest.approx(15 / n)
    # The five MY calls (class bp 0.5) are not confident: the confident-cell
    # denominator leaves them out.
    assert v1_row["n_confident_allen"] == n - 5
    assert v1_row["harm_confident"] == pytest.approx(15 / (n - 5))
    assert v1_row["n_remap"] == 20 and len(v1_remap) == 20  # + the MY calls
    # The CCF (oracle) regions keep OB: the whole class is oracle harm too.
    assert v1_row["oracle_harm"] == pytest.approx(15 / n)
    assert v1_row["drop_sha"] == module.drop_sha(v1_nodes)
    coupled, nodes, remap = module.evaluate_candidate(
        data,
        _shares(),
        _taxonomy(),
        module.candidate_by_label("a"),
        ccf_present=ccf,
        ccf_any=ccf,
    )
    assert coupled["inferred"] == "Isocortex;OLF;OB" and coupled["coupled"] == "OB"
    assert json.loads(coupled["coupling_tiles"]) == {"OB": 6}
    assert coupled["missed"] == "" and coupled["harm"] == 0.0
    assert coupled["oracle_harm"] == 0.0 and len(remap) == 5
    # "extra" is inferred minus the > 0.2% CCF regions, "missed" the >= 1%
    # CCF regions minus inferred.
    row, _, _ = module.evaluate_candidate(
        data,
        _shares(),
        _taxonomy(),
        module.candidate_by_label("v1"),
        ccf_present={"Isocortex", "OB"},
        ccf_any={"Isocortex", "OLF", "OB"},
    )
    assert (row["missed"], row["extra"]) == ("OB", "")
    row, _, _ = module.evaluate_candidate(
        data,
        _shares(),
        _taxonomy(),
        module.candidate_by_label("v1"),
        ccf_present={"Isocortex", "OLF", "OB"},
        ccf_any={"Isocortex", "OB"},
    )
    assert (row["missed"], row["extra"]) == ("OB", "OLF")
    # Harm needs our class bp >= 0.9 as well as an Allen correlation >= 0.6.
    data.class_bp = np.where(
        data.subclass_names == "050 OB-IMN A",
        0.8,
        np.where(data.subclass_names == "240 MY Glut A", 0.5, 1.0),
    )
    row, _, _ = module.evaluate_candidate(
        data,
        _shares(),
        _taxonomy(),
        module.candidate_by_label("v1"),
        ccf_present=ccf,
        ccf_any=ccf,
    )
    assert row["n_harm"] == 3
    data.acs[: n // 2] = 0.1
    row, _, _ = module.evaluate_candidate(
        data,
        _shares(),
        _taxonomy(),
        module.candidate_by_label("v1"),
        ccf_present=ccf,
        ccf_any=ccf,
    )
    # Not confident: 12 neuroblasts (class bp 0.8) and 5 MY calls (0.5).
    assert row["n_confident_allen"] == int((data.acs >= 0.6).sum() - 17)


def test_drop_sha_matches_the_region_record() -> None:
    module = _load()
    nodes = [
        DroppedNode(
            level=CLAS,
            node="CL_05",
            name="05 OB-IMN GABA",
            kind="class",
            class_name="05 OB-IMN GABA",
            present_share=0.0,
            n_merfish_grey=10,
            rule="all_subclasses_dropped",
        )
    ]
    record = MouseRegionRecord(
        rule_variant="v1",
        run_id="wmb_panel",
        requested="auto",
        source="auto",
        status="pruned",
        n_table_cells=1,
        nodes_to_drop=nodes,
        cells_parquet="x",
        cells_parquet_sha256="y",
    )
    assert module.drop_sha(nodes) == record.drop_list_sha256


# --------------------------------------------------------------------------
# Per-bin summary on a planted-sliver set


def _section_rows(
    label: str, harms: dict[str, tuple[str, float, float, str]]
) -> list[dict[str, Any]]:
    return [
        {
            "gene_set": "merfish500",
            "candidate": label,
            "section": section,
            "ap_bin": ap_bin,
            "harm": harm,
            "t2_after": t2,
            "missed": missed,
            # The planted anterior sliver section also has an extra region.
            "extra": "HPF" if section == "S56" else "",
            "status": "skipped_few_tiles" if section == "S40" else "pruned",
            "dataset": f"merfish500/{section}",
        }
        for section, (ap_bin, harm, t2, missed) in harms.items()
    ]


PLANTED = {
    # section: (AP bin, harm, T2 after, missed regions)
    "S61": ("2.4-3.0", 0.0, 0.0002, ""),
    "S60": ("2.4-3.0", 0.001, 0.0, ""),  # exactly 0.1%: not "above"
    "S58": ("3.0-4.5", 0.0001, 0.0004, ""),
    "S57": ("3.0-4.5", 0.0070, 0.0001, "OB"),  # the planted OB sliver
    "S56": ("3.0-4.5", 0.0002, 0.0003, "CTXsp"),
    "S40": ("6.0-7.5", 0.0011, 0.0, ""),
}


def test_per_bin_summary_on_a_planted_sliver_set() -> None:
    module = _load()
    frame = pd.DataFrame(_section_rows("v1", PLANTED))
    bins = module.per_bin_summary(frame, ["harm", "t2_after"])
    assert bins["ap_bin"].tolist() == ["2.4-3.0", "3.0-4.5", "6.0-7.5"]
    assert bins.set_index("ap_bin").loc["2.4-3.0", "harm_n_above"] == 0
    anterior = bins.set_index("ap_bin").loc["3.0-4.5"]
    assert anterior["n_sections"] == 3
    assert anterior["n_sections_missing_region"] == 2
    assert anterior["n_sections_extra_region"] == 1
    assert anterior["n_sections_skipped"] == 0
    assert anterior["harm_mean"] == pytest.approx((0.0001 + 0.0070 + 0.0002) / 3)
    assert anterior["harm_max"] == pytest.approx(0.0070)
    assert anterior["harm_n_above"] == 1  # > 0.1%
    assert anterior["t2_after_max"] == pytest.approx(0.0004)
    posterior = bins.set_index("ap_bin").loc["6.0-7.5"]
    assert (
        posterior["harm_n_above"] == 1 and posterior["n_sections_missing_region"] == 0
    )
    assert posterior["n_sections_skipped"] == 1
    # MO10 fails on the planted sliver (bin mean, section max, missed + harm).
    verdict = module.mo10_verdict(frame.assign(oracle_harm=0.0), bins)
    entry = verdict["gene_sets"]["merfish500"]
    assert not entry["bin_mean_harm_ok"] and not entry["section_max_harm_ok"]
    assert not entry["missed_only_where_harm_small"] and not verdict["pass"]
    assert [row["section"] for row in entry["missed_with_harm_above"]] == ["S57"]
    fixed = dict(
        PLANTED, S57=("3.0-4.5", 0.0003, 0.0001, ""), S40=("6.0-7.5", 0.0009, 0.0, "")
    )
    frame = pd.DataFrame(_section_rows("a", fixed))
    bins = module.per_bin_summary(frame, ["harm", "t2_after"])
    verdict = module.mo10_verdict(frame.assign(oracle_harm=0.0), bins)
    assert verdict["pass"]
    # A missed region is allowed where the section's harm is <= 0.1%.
    assert verdict["gene_sets"]["merfish500"]["missed_only_where_harm_small"]


# --------------------------------------------------------------------------
# Candidate table and the pre-registered selection


def _nodes(*pairs: tuple[str, str]) -> str:
    return json.dumps(
        [
            {"level": level, "node": node, "name": node, "kind": "class"}
            for level, node in pairs
        ]
    )


def _merfish(
    label: str, harm: dict[str, float], nodes: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": f"merfish500/{section}",
            "gene_set": "merfish500",
            "section": section,
            "candidate": label,
            "harm": value,
            "oracle_harm": 0.0,
            "status": "pruned",
            "missed": "",
            "nodes": nodes[section],
            "remap_missing": False,
        }
        for section, value in harm.items()
    ]


def test_candidate_table_and_selection_follow_the_preregistration() -> None:
    module = _load()
    v1_nodes = {"S1": _nodes((CLAS, "A"), (CLAS, "B")), "S2": _nodes((CLAS, "A"))}
    fewer = {"S1": _nodes((CLAS, "A")), "S2": _nodes((CLAS, "A"))}
    rows = (
        _merfish("v1", {"S1": 0.007, "S2": 0.001}, v1_nodes)
        + _merfish("a", {"S1": 0.001, "S2": 0.0005}, fewer)
        + _merfish("b", {"S1": 0.001, "S2": 0.0002}, v1_nodes)
        + _merfish("a+b", {"S1": 0.0005, "S2": 0.0001}, fewer)
        + _merfish("f:tile100", {"S1": 0.001, "S2": 0.0}, v1_nodes)
    )
    merfish = pd.DataFrame(rows)
    bins = pd.DataFrame(
        [
            {
                "candidate": label,
                "gene_set": "merfish500",
                "ap_bin": "3.0-4.5",
                "t2_after_mean": t2,
            }
            for label, t2 in (
                ("v1", 0.0),
                ("a", 0.0),
                ("b", 0.001),
                ("a+b", 0.002),
                ("f:tile100", 0.0),
            )
        ]
    )
    real = pd.DataFrame(
        [
            {
                "candidate": label,
                "section": name,
                "t2_after": 0.0,
                "class_changed_outside": 0.0,
                "subclass_changed_outside": 0.02
                if label == "f:tile100" and name == "ag7"
                else 0.0,
                "inferred": "Isocortex;OLF;HPF;CTXsp;TH;HY;MB",
            }
            for label in ("v1", "a", "b", "a+b", "f:tile100")
            for name in ("ag7", "vzg2")
        ]
    )
    table = module.candidate_table(
        merfish, real, bins, t2_reading="registered"
    ).set_index("candidate")
    assert table.loc["v1", "is_reference"]
    assert table.loc["a", "S_max_harm"] == 0.001
    assert table.loc["a", "S_at"] == "merfish500/S1"
    assert not table.loc["a+b", "t2_ok"] and not table.loc["a+b", "admissible"]
    assert not table.loc["f:tile100", "mo2_ag7"] and table.loc["f:tile100", "mo2_vzg2"]
    # T2 exactly at 0.1% is allowed; above it is not.
    assert table.loc["b", "t2_ok"] and table.loc["b", "max_t2_bin_mean"] == 0.001
    # a drops one node fewer than v1 in S1; b drops the same nodes.
    assert table.loc["a", "n_drop_decisions_differing"] == 1
    assert table.loc["b", "n_drop_decisions_differing"] == 0
    selection = module.select_candidate(table.reset_index())
    # a and b tie at S = 0.001 with one change each; b differs from v1 less.
    assert selection["selected"] == "b"
    assert selection["ties"] == ["b", "a"]
    # v1 is never selected, even when it is best.
    table.loc["v1", "S_max_harm"] = 0.0
    assert module.select_candidate(table.reset_index())["selected"] == "b"
    none = table.reset_index().assign(admissible=False)
    assert module.select_candidate(none)["selected"] is None


def _t2_bins(values: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Bin rows of one gene set: candidate -> (T2 after, Allen-label T2)."""
    return pd.DataFrame(
        [
            {
                "candidate": label,
                "gene_set": "merfish500",
                "ap_bin": "3.0-4.5",
                "t2_after_mean": after,
                "t2_allen_mean": allen,
            }
            for label, (after, allen) in values.items()
        ]
    )


def test_the_two_t2_readings_of_constraint_ii() -> None:
    module = _load()
    bins = _t2_bins(
        {
            "at_limit": (0.001, 0.0005),
            "above_limit_below_allen": (0.004, 0.005),
            "equal_to_allen": (0.004, 0.004),
            "allen_plus_rounding": (0.004 + 5e-13, 0.004),
            "above_allen": (0.004 + 2e-12, 0.004),
            "allen_missing": (0.0, float("nan")),
            "after_missing": (float("nan"), 0.01),
        }
    ).set_index("candidate")

    def check(label: str) -> tuple[bool, bool]:
        part = bins.loc[[label]]
        return module.t2_ok_registered(part), module.t2_ok_allen(part)

    # Registered: the bin mean <= 0.1% (inclusive); a missing value fails.
    # Allen-relative (§24 amendment of 2026-10-14): the bin mean <= the Allen
    # labels' bin mean with the same regions, 1e-12 for rounding only; a
    # missing value on either side fails.
    assert check("at_limit") == (True, False)
    assert check("above_limit_below_allen") == (False, True)
    assert check("equal_to_allen") == (False, True)
    assert check("allen_plus_rounding") == (False, True)
    assert check("above_allen") == (False, False)
    assert check("allen_missing") == (True, False)
    assert check("after_missing") == (False, False)
    # Bins without the Allen-label column cannot meet the Allen bound.
    assert not module.t2_ok_allen(bins.loc[["equal_to_allen"]][["t2_after_mean"]])
    assert module.T2_READINGS == ("allen-relative", "registered")


def test_the_allen_relative_reading_replaces_constraint_ii() -> None:
    module = _load()
    nodes = {"S1": _nodes((CLAS, "A")), "S2": _nodes((CLAS, "A"))}
    merfish = pd.DataFrame(
        _merfish("v1", {"S1": 0.007, "S2": 0.001}, nodes)
        + _merfish("a", {"S1": 0.001, "S2": 0.0}, nodes)
        + _merfish("b", {"S1": 0.002, "S2": 0.0}, nodes)
        + _merfish("c", {"S1": 0.0005, "S2": 0.0}, nodes)
    )
    # Every rule is above 0.1%; a and b stay at or below the Allen labels'
    # own T2, c does not.
    bins = _t2_bins(
        {
            "v1": (0.004, 0.006),
            "a": (0.003, 0.003),
            "b": (0.002, 0.003),
            "c": (0.0031, 0.003),
        }
    )
    real = pd.DataFrame(
        [
            {
                "candidate": label,
                "section": name,
                "t2_after": 0.0,
                "class_changed_outside": 0.0,
                "subclass_changed_outside": 0.0,
                "inferred": "Isocortex",
            }
            for label in ("v1", "a", "b", "c")
            for name in ("ag7", "vzg2")
        ]
    )
    registered = module.candidate_table(
        merfish, real, bins, t2_reading="registered"
    ).set_index("candidate")
    assert not registered["admissible"].any()
    assert module.select_candidate(registered.reset_index())["selected"] is None
    table = module.candidate_table(merfish, real, bins).set_index("candidate")
    # The approved reading is the default; both verdicts are reported.
    assert set(table["t2_reading"]) == {"allen-relative"}
    assert not table["t2_ok_registered"].any()
    assert table["t2_ok_allen"].to_dict() == {
        "v1": True,
        "a": True,
        "b": True,
        "c": False,
    }
    assert table["t2_ok"].equals(table["t2_ok_allen"])
    assert table.loc["c", "max_t2_excess_over_allen"] == pytest.approx(0.0001)
    assert table.loc["c", "n_t2_bins_above_allen"] == 1
    assert table.loc["a", "n_t2_bins_above_allen"] == 0
    # c has the lowest S but fails the Allen bound; a is selected (S 0.001).
    selection = module.select_candidate(table.reset_index())
    assert selection["selected"] == "a" and selection["ties"] == ["a"]
    with pytest.raises(ValueError, match="T2 reading"):
        module.candidate_table(merfish, real, bins, t2_reading="excess")


def test_t2_columns_score_the_allen_labels_with_the_rules_regions() -> None:
    module = _load()
    taxonomy = _taxonomy()
    regions = ["Isocortex", "OLF"]
    # 050 / 051 OB-IMN and 240 MY are T2 with Isocortex + OLF present;
    # 300 Astro NN is in a never-drop class.
    calls = np.array(["007 L2/3 IT CTX Glut"] * 9 + ["051 OB-IMN B"], dtype=object)
    allen = np.array(
        ["050 OB-IMN A"] * 2 + ["300 Astro NN"] * 3 + ["007 L2/3 IT CTX Glut"] * 5,
        dtype=object,
    )
    v1 = module.candidate_by_label("v1").config()
    row = module.t2_columns(calls, allen, regions, _shares(), taxonomy, v1)
    assert row["t2_after"] == pytest.approx(0.1)
    assert row["t2_allen"] == pytest.approx(0.2)
    # The Allen side keeps the static never-drop classes only: (b)'s
    # conditional keep of 05 OB-IMN GABA (OLF present) does not change it.
    keep = module.candidate_by_label("b").config()
    kept = module.t2_columns(calls, allen, regions, _shares(), taxonomy, keep)
    assert kept["t2_allen"] == pytest.approx(0.2)
    assert kept["t2_after"] == pytest.approx(0.1)
    assert kept["t2_after_kept_excluded"] == 0.0
    # A missing re-map leaves T2 after pruning missing, not the Allen side.
    missing = module.t2_columns(
        calls, allen, regions, _shares(), taxonomy, v1, remap_missing=True
    )
    assert np.isnan(missing["t2_after"]) and np.isnan(missing["t2_after_kept_excluded"])
    assert missing["t2_allen"] == pytest.approx(0.2)
    # Real data has no Allen labels.
    real = module.t2_columns(calls, None, regions, _shares(), taxonomy, v1)
    assert np.isnan(real["t2_allen"]) and real["t2_after"] == pytest.approx(0.1)


def test_a_taxonomy_override_must_fit_every_call_and_allen_label() -> None:
    module = _load()
    data = _sliver_dataset(module)
    module.check_taxonomy_fits(data, _taxonomy())
    data.allen_class = data.allen_class.copy()
    data.allen_class[0] = "24 MY Glut"
    with pytest.raises(ValueError, match="do not fit"):
        module.check_taxonomy_fits(data, _taxonomy())
    data = _sliver_dataset(module)
    data.class_names = data.class_names.copy()
    data.class_names[-1] = "01 IT-ET Glut"  # a 240 MY Glut A call
    with pytest.raises(ValueError, match="do not fit"):
        module.check_taxonomy_fits(data, _taxonomy())


def test_mo10_refuses_before_the_selection_is_recorded(tmp_path: Path) -> None:
    module = _load()
    summary = tmp_path / "summary"
    summary.mkdir()
    (summary / "selection.json").write_text(json.dumps({"selected": "a+b"}))
    frame = pd.DataFrame(_section_rows("a+b", PLANTED)).assign(oracle_harm=0.0)
    frame.to_parquet(summary / "sections.parquet")
    module.per_bin_summary(frame, ["harm", "t2_after"]).to_csv(
        summary / "bins.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "candidate": "a+b",
                "section": "ag7",
                "t2_after": 0.0,
                "class_changed_outside": 0.0,
                "subclass_changed_outside": 0.0,
                "inferred": "Isocortex;OLF;HPF;CTXsp;TH;HY;MB",
            }
        ]
    ).to_csv(summary / "real.csv", index=False)
    prereg = tmp_path / "prereg.md"
    prereg.write_text("## 24\nno selection yet\n")
    args = module.build_parser().parse_args(
        ["mo10", "--out", str(tmp_path), "--preregistration", str(prereg)]
    )
    with pytest.raises(SystemExit, match="commit the selection first"):
        module.mo10_command(args)
    prereg.write_text("## 24\nSelected candidate (M6b): `a+b` (2026-09-30)\n")
    assert module.mo10_command(args) == 0
    verdict = json.loads((tmp_path / "mo10" / "mo10.json").read_text())
    assert verdict["candidate"] == "a+b" and not verdict["pass"]
    assert verdict["mo3"] == {"ag7": True}
    # A selection without a recorded reading was made under the registered one.
    assert verdict["t2_reading"] == "registered"
    assert verdict["gene_sets"]["merfish500"]["t2_bin_mean_ok_allen"] is None


# section: (AP bin, harm, T2 after, missed regions), and the Allen-label T2.
ALLEN_T2 = {
    "S61": ("2.4-3.0", 0.0, 0.0002, ""),
    "S60": ("2.4-3.0", 0.0002, 0.0, ""),
    "S58": ("3.0-4.5", 0.0001, 0.004, ""),
    "S57": ("3.0-4.5", 0.0003, 0.002, ""),
    "S56": ("3.0-4.5", 0.0002, 0.003, "CTXsp"),
    "S40": ("6.0-7.5", 0.0009, 0.0, ""),
}
ALLEN_LABEL_T2 = {
    "S61": 0.0003,
    "S60": 0.0,
    "S58": 0.006,
    "S57": 0.002,
    "S56": 0.001,
    "S40": 0.0,
}


def _write_mo10_inputs(out: Path, rows: dict[str, Any], selection: dict) -> None:
    summary = out / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "selection.json").write_text(json.dumps(selection))
    frame = pd.DataFrame(_section_rows("a+b", rows)).assign(oracle_harm=0.0)
    frame["t2_allen"] = frame["section"].map(ALLEN_LABEL_T2)
    frame.to_parquet(summary / "sections.parquet")
    module = _load()
    module.per_bin_summary(frame, ["harm", "t2_after", "t2_allen"]).to_csv(
        summary / "bins.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "candidate": "a+b",
                "section": "vzg2",
                "t2_after": 0.0,
                "class_changed_outside": 0.0,
                "subclass_changed_outside": 0.0,
                "inferred": "Isocortex;OLF;HPF;CTXsp;TH;HY;MB",
            }
        ]
    ).to_csv(summary / "real.csv", index=False)


def test_mo10_scores_t2_under_the_allen_relative_reading(tmp_path: Path) -> None:
    module = _load()
    _write_mo10_inputs(
        tmp_path, ALLEN_T2, {"selected": "a+b", "t2_reading": "allen-relative"}
    )
    prereg = tmp_path / "prereg.md"
    heading = "### Amendment of 2026-10-14 (U1): the T2 reading approved by the user"
    # The selection under the approved reading must be recorded after the
    # amendment that approved it.
    prereg.write_text(f"## 24\nSelected candidate (M6b): `a+b`\n\n{heading}\n")
    args = module.build_parser().parse_args(
        ["mo10", "--out", str(tmp_path), "--preregistration", str(prereg)]
    )
    with pytest.raises(SystemExit, match="after"):
        module.mo10_command(args)
    prereg.write_text(
        "## 24\nSelected candidate (M6b): `v1`\n\n"
        f"{heading}\n\nSelected candidate (M6b): `a+b`\n"
    )
    assert module.mo10_command(args) == 0
    verdict = json.loads((tmp_path / "mo10" / "mo10.json").read_text())
    assert verdict["t2_reading"] == "allen-relative"
    entry = verdict["gene_sets"]["merfish500"]
    # The 3.0-4.5 bin: T2 0.30% against the Allen labels' 0.30%.
    assert entry["t2_bin_mean_ok_allen"] and not entry["t2_bin_mean_ok_registered"]
    assert entry["t2_bin_mean_ok"] and entry["pass"]
    assert verdict["pass"] and not verdict["pass_registered_t2"]
    assert verdict["failing_parts"] == []
    assert [
        (part["part"], part["where"]) for part in verdict["failing_parts_registered_t2"]
    ] == [("t2_bin_mean", "3.0-4.5")]
    bins = {row["ap_bin"]: row for row in entry["t2_bins"]}
    assert bins["3.0-4.5"]["t2_allen_mean"] == pytest.approx(0.003)
    assert (tmp_path / "mo10" / "MO10.txt").is_file()
    # A missed region where the section's harm is above 0.1% fails MO10
    # under either reading and is listed.
    rows = dict(ALLEN_T2, S57=("3.0-4.5", 0.0014, 0.002, "STR"))
    _write_mo10_inputs(
        tmp_path, rows, {"selected": "a+b", "t2_reading": "allen-relative"}
    )
    assert module.mo10_command(args) == 0
    verdict = json.loads((tmp_path / "mo10" / "mo10.json").read_text())
    assert not verdict["pass"]
    assert [
        (part["part"], part["where"], part["value"])
        for part in verdict["failing_parts"]
    ] == [("missed_region", "S57", pytest.approx(0.0014))]
    assert verdict["failing_parts"][0]["missed"] == "STR"


# --------------------------------------------------------------------------
# The tables check against a bundle


def test_compare_tables_matches_a_bundle_written_by_region_share_tables(
    tmp_path: Path,
) -> None:
    from merxen.annotation.mouse_ap import MerfishRegionCounts

    module = _load()
    rng = np.random.default_rng(0)
    n = 2000
    cells = pd.DataFrame(
        {
            "brain_section_label": rng.choice(["S1", "S2", "S3"], n),
            "class": rng.choice(["01 A Glut", "30 Astro-Epen"], n),
            "x_ccf": rng.uniform(1, 5, n),
            "region": rng.choice(["Isocortex", "OLF", "OB", "fibre"], n),
            "parcellation_structure": rng.choice(["MOp", "AON", "MOB"], n),
        }
    )
    cells["subclass"] = cells["class"] + " " + rng.choice(["x", "y"], n)
    tables = region_share_tables(cells)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    tables["region_share"].to_parquet(bundle / REGION_SHARE_FILE)
    tables["region_home"].to_parquet(bundle / REGION_HOME_FILE)
    ours = MerfishRegionCounts.from_cells(cells).tables(())
    assert module.compare_tables(ours, bundle)["equal"]
    tampered = tables["region_share"].copy()
    tampered.loc[0, "n"] += 1
    tampered.to_parquet(bundle / REGION_SHARE_FILE)
    assert not module.compare_tables(ours, bundle)["equal"]


def test_section_shares_leave_the_section_and_its_neighbours_out() -> None:
    from merxen.annotation.mouse_ap import MerfishRegionCounts

    module = _load()
    rows = []
    for index, ap in enumerate((1.0, 2.0, 3.0, 4.0)):
        for _ in range(5 + index):
            rows.append(
                {
                    "brain_section_label": f"S{index}",
                    "x_ccf": ap,
                    "class": "01 A Glut",
                    "subclass": f"00{index} only S{index} Glut",
                    "region": "Isocortex",
                    "parcellation_structure": "MOp",
                }
            )
    counts = MerfishRegionCounts.from_cells(pd.DataFrame(rows))
    sections = pd.DataFrame(
        {"excluded": ["S0;S1", "S0;S1;S2", "S1;S2;S3", "S2;S3"]},
        index=pd.Index(["S0", "S1", "S2", "S3"], name="section"),
    )
    shares = module.section_shares(counts, sections, "S1")
    assert list(shares.subclass_n_grey.index) == ["003 only S3 Glut"]
    assert shares.class_n_grey["01 A Glut"] == 8
    sections.loc["S1", "excluded"] = "S1"
    with pytest.raises(ValueError, match="neighbourhood"):
        module.section_shares(counts, sections, "S1")


def test_a_skipped_section_scores_t2_on_its_inferred_regions() -> None:
    module = _load()
    # 12 x 15 Isocortex tiles + 6 AON + 1 OB = 187 assigned tiles (< 200).
    data = _sliver_dataset(module, cortex_columns=12)
    ccf = {"Isocortex", "OLF", "OB"}
    row, nodes, remap = module.evaluate_candidate(
        data,
        _shares(),
        _taxonomy(),
        module.candidate_by_label("v1"),
        ccf_present=ccf,
        ccf_any=ccf,
    )
    assert row["status"] == "skipped_few_tiles" and row["n_assigned_tiles"] == 187
    assert row["present"] == "" and not nodes and not remap and row["harm"] == 0.0
    assert row["t2_regions"] == "Isocortex;OLF" == row["inferred"]
    # Against the inferred regions the unpruned OB-IMN calls are T2.
    assert row["t2_unpruned"] == pytest.approx(20 / len(data.cell_ids))


def test_estimate_row_leaves_the_section_out() -> None:
    module = _load()
    subclasses = [f"{index:03d} T Glut" for index in range(60)]
    centres = np.linspace(0.5, 11.5, 60)
    rows, table = [], []
    for index, ap in enumerate(np.linspace(1.0, 10.6, 20)):
        weights = np.exp(-0.5 * ((centres - ap) / 0.6) ** 2) + 1e-4
        counts = np.round(4000 * weights / weights.sum()).astype(int)
        section = f"C57BL6J-638850.{70 - index}"
        rows += [
            {
                "section": section,
                "ap_ccf_mm": ap,
                "level": "subclass",
                "node_name": name,
                "n": n,
            }
            for name, n in zip(subclasses, counts, strict=True)
            if n
        ]
        table.append(
            {"section": section, "ap_ccf_mm": ap, "in_scope": 2.4 <= ap <= 10.4}
        )
    composition = pd.DataFrame(rows)
    sections = pd.DataFrame(table).set_index("section")
    steps = pd.Series(0.51, index=sections.index)
    section = "C57BL6J-638850.60"
    part = composition[composition["section"] == section]
    names = np.repeat(part["node_name"].to_numpy(), part["n"].to_numpy())
    row = module._estimate_row("allen", section, names, composition, sections, steps)
    assert row["estimate_section"] in {"C57BL6J-638850.61", "C57BL6J-638850.59"}
    assert row["within_step"] and row["abs_error_mm"] > 0
    few = module._estimate_row(
        "allen", section, names[:10], composition, sections, steps
    )
    assert "confident neurons" in few["error"]


def test_t2_counts_the_kept_nodes_and_reports_them_apart() -> None:
    module = _load()
    taxonomy = _taxonomy()
    calls = ["050 OB-IMN A"] * 3 + ["007 L2/3 IT CTX Glut"] * 97
    regions = ["Isocortex", "OLF"]
    v1 = module.candidate_by_label("v1").config()
    assert module.t2_values(calls, regions, _shares(), taxonomy, v1) == (0.03, 0.03)
    keep = module.candidate_by_label("b").config()  # OLF present: 05 kept
    static, kept_excluded = module.t2_values(calls, regions, _shares(), taxonomy, keep)
    assert static == pytest.approx(0.03) and kept_excluded == 0.0
    # With OB present the OB-IMN calls are plausible (no T2 either way).
    assert module.t2_values(calls, [*regions, "OB"], _shares(), taxonomy, keep) == (
        0.0,
        0.0,
    )
