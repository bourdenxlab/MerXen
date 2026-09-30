"""Tests for the mouse items of the annotation report (``report_mouse``; §9 item 10).

The end-to-end test runs the synthetic mouse section of ``conftest``
(``mouse_setup``) through MAP (region inference, pruning, the re-map) and
RESOLVE, then builds the report from those outputs. M6b fields (AP estimate,
rule variant) are absent there and must be reported as pending.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.report import build_annotation_report
from merxen.annotation.report_inputs import ReportSources
from merxen.annotation.report_model import ReportOptions
from merxen.annotation.report_mouse import (
    PENDING,
    ap_fields,
    subclass_region_table,
    window_composition,
)

from .conftest import MOUSE_SID, map_mouse
from .test_pipeline_resolve_mouse import PASSING, _add_profiles, _resolve

MakeTrust = Callable[..., Any]


def test_ap_fields_are_pending_without_m6b() -> None:
    fields = ap_fields({"rule_variant": "v1", "params": {"rule_variant": "v1"}}, {})
    assert fields["ap_status"] == PENDING and fields["ap_estimate_mm"] is None
    assert fields["rule_variant"] == "v1" and fields["rule_variant_status"] == PENDING
    assert fields["ap_in_scope"] is None and fields["warnings"] == []


def test_ap_fields_read_an_m6b_estimate_and_warn_outside_the_scope() -> None:
    inside = ap_fields(
        {"ap_estimate_mm": 8.2, "ap_bin": "7.5-9.0", "m6b_rule_variant": "a"}, None
    )
    assert inside["ap_status"] == "measured" and inside["ap_in_scope"] is True
    assert inside["rule_variant_status"] == "selected" and inside["ap_bin"] == "7.5-9.0"
    outside = ap_fields({}, {"ap_estimate_mm": 11.0})
    assert outside["ap_in_scope"] is False
    assert "outside the validated coronal scope" in outside["warnings"][0]


def test_subclass_region_table_compares_with_renormalised_merfish_shares() -> None:
    subclass = np.array(["S1", "S1", "S1", "S2", "", "S2"])
    region = np.array(["CTX", "CTX", "TH", "TH", "TH", ""])
    merfish = pd.DataFrame(
        {
            "level": ["subclass"] * 4 + ["class"],
            "node_name": ["S1", "S1", "S1", "S2", "S1"],
            "region": ["CTX", "TH", "MB", "TH", "CTX"],
            "share": [0.5, 0.25, 0.25, 1.0, 0.9],
        }
    )
    table = subclass_region_table(subclass, region, merfish, ["CTX", "TH"]).set_index(
        ["subclass", "region"]
    )
    assert table.loc[("S1", "CTX"), "n_cells"] == 2
    assert table.loc[("S1", "CTX"), "share"] == pytest.approx(2 / 3)
    # MB is not present: S1's MERFISH shares renormalise over CTX + TH.
    assert table.loc[("S1", "CTX"), "merfish_share"] == pytest.approx(0.5 / 0.75)
    assert table.loc[("S2", "TH"), "merfish_share"] == pytest.approx(1.0)
    assert subclass_region_table(np.array([""]), np.array([""]), None, []).empty


def test_window_composition_pools_the_window_sections(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "section": ["s1", "s1", "s2", "s3"],
            "level": ["class"] * 4,
            "node_name": ["A", "B", "A", "B"],
            "n": [10, 30, 60, 1000],
        }
    ).to_parquet(tmp_path / "section_composition.parquet")
    frame = window_composition(tmp_path, ["s1", "s2"]).set_index("class")
    assert frame.loc["A", "merfish_share"] == pytest.approx(70 / 100)
    assert window_composition(tmp_path, []) is None
    assert window_composition(None, ["s1"]) is None


def test_mouse_report_builds_item_10_on_a_synthetic_section(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    _add_profiles(mouse_setup["bundle"].path)
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    _resolve(
        mouse_setup,
        map_dir,
        tmp_path / "resolve",
        make_trust,
        registration={MOUSE_SID: PASSING},
    )
    summary = json.loads(
        next((tmp_path / "resolve").glob("*_resolve_summary.json")).read_text()
    )
    sources = ReportSources(
        species="mouse",
        pair_id=str(summary["pair_id"]),
        segmentation="proseg_hybrid",
        resolve_dir=tmp_path / "resolve",
        map_dir=map_dir,
        clustered_h5ad={MOUSE_SID: mouse_setup["sample"].h5ad_path},
        store_root=tmp_path / "store",
    )
    result = build_annotation_report(
        sources,
        tmp_path / "report",
        options=ReportOptions(n_bootstrap=10),
        strict=True,
        make_figures=False,
    )
    statuses = {item.number: item.status for item in result.items}
    assert statuses[10] == "ok"
    assert statuses[6] == statuses[7] == statuses[9] == statuses[11] == "not_applicable"
    metrics = result.metrics
    (regions,) = metrics.find("MO3", "inferred_regions")
    assert regions.value
    (ap,) = metrics.find("MO11", "ap_estimate_mm")
    assert ap.value is None and ap.note == PENDING and ap.status == "not_available"
    (variant,) = metrics.find("MO10", "m6b_rule_variant")
    assert variant.value is None and variant.note == PENDING
    (relabelled,) = metrics.find("MO2", "n_relabelled")
    assert relabelled.value > 0
    (outside,) = metrics.find("MO2", "class_changes_outside_dropped")
    assert outside.value == 0.0
    (in_dropped,) = metrics.find("MO2", "pruned_calls_in_dropped_nodes")
    assert in_dropped.value == 0
    assert metrics.find("MO6", "f1_region_incoherent_rate")
    (level,) = metrics.find("MO7", "mouse_gate_level")
    assert level.value in ("full", "broad_only", "failed")
    assert metrics.find("MO4", "astro_epen_soft_share")
    assert metrics.find("MO5", "spillover_prevalence")
    report = tmp_path / "report"
    tiles = pd.read_csv(report / "figures" / "item10_mouse_regions_tile_map.csv")
    assert len(tiles) and set(tiles["value"]) <= set(regions.value.split(";")) | {"nan"}
    table = pd.read_csv(report / "tables" / "item10_mouse_regions__regions.csv")
    assert table.iloc[0]["ap_status"] == PENDING
    genes = pd.read_csv(
        report / "tables" / "item10_mouse_regions__spillover_gene_sets.csv"
    )
    assert list(genes.columns) == ["sample_id", "set", "gene"]
    drops = pd.read_csv(report / "tables" / "item10_mouse_regions__drop_list.csv")
    assert len(drops)
    item = next(item for item in result.items if item.number == 10)
    assert any("pending (M6b)" in note for note in item.notes)
    assert any("157 RN Spp1 Glut" in note for note in item.notes)
    coverage = {row.criterion: row for row in metrics.criteria}
    assert coverage["MO7"].n_records >= 1 and coverage["MO10"].source.startswith(
        "script"
    )
