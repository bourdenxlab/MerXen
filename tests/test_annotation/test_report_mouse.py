"""Tests for the mouse items of the annotation report (``report_mouse``; §9 item 10).

The end-to-end test runs the synthetic mouse section of ``conftest``
(``mouse_setup``) through MAP (region inference, pruning, the re-map) and
RESOLVE, then builds the report from those outputs. That section has no AP
estimate (its region-share bundle has no section composition), which the
report must show as pending; a second test gives it one and reads it back.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation.mouse_regions import RegionShareBundle
from merxen.annotation.report import build_annotation_report
from merxen.annotation.report_inputs import ReportSources
from merxen.annotation.report_model import ReportOptions
from merxen.annotation.report_mouse import (
    PENDING,
    ap_fields,
    subclass_region_table,
    window_composition,
)

from .conftest import MOUSE_SID, map_mouse, mouse_region_share_bundle
from .test_pipeline_resolve_mouse import (
    PASSING,
    _add_profiles,
    _ap_composition,
    _resolve,
)

MakeTrust = Callable[..., Any]


def test_ap_fields_are_pending_without_an_estimate() -> None:
    fields = ap_fields({"rule_variant": "v1", "params": {"rule_variant": "v1"}}, {})
    assert fields["ap_status"] == PENDING and fields["ap_estimate_mm"] is None
    assert fields["ap_bin"] is None and fields["ap_in_scope"] is None
    assert fields["warnings"] == []
    # The recorded rule variant is the M6b variant (v1 is the production rule).
    assert fields["rule_variant"] == "v1" and fields["m6b_rule_variant"] == "v1"
    assert fields["rule_variant_status"] == "selected"
    nothing = ap_fields(None, None)
    assert nothing["ap_status"] == nothing["rule_variant_status"] == PENDING
    assert nothing["ap_estimate_mm"] is None and nothing["m6b_rule_variant"] is None


def test_ap_fields_read_an_m6b_estimate_and_warn_outside_the_scope() -> None:
    inside = ap_fields(
        {"ap_estimate_mm": 8.2, "ap_bin": "7.5-9.0", "m6b_rule_variant": "a"}, None
    )
    assert inside["ap_status"] == "measured" and inside["ap_in_scope"] is True
    assert inside["rule_variant_status"] == "selected" and inside["ap_bin"] == "7.5-9.0"
    assert inside["m6b_rule_variant"] == "a"
    outside = ap_fields({}, {"ap_estimate_mm": 11.0})
    assert outside["ap_in_scope"] is False
    assert "outside the validated coronal scope" in outside["warnings"][0]


def test_ap_fields_read_the_nested_gate_estimate() -> None:
    """RESOLVE writes ``mouse_gate["ap_estimate"]`` (``ApEstimate.to_json``)."""
    nested = {
        "section": "S2",
        "ap_mm": 5.0,
        "ap_bin": "4.5-6.0",
        "in_scope": True,
        "window_sections": ["S1", "S2", "S3"],
    }
    fields = ap_fields(
        {"rule_variant": "m6b", "params": {"rule_variant": "m6b"}},
        {"level": "full", "ap_estimate": nested},
    )
    assert fields["ap_estimate_mm"] == 5.0 and fields["ap_bin"] == "4.5-6.0"
    assert fields["ap_status"] == "measured" and fields["ap_in_scope"] is True
    assert fields["m6b_rule_variant"] == "m6b" and fields["warnings"] == []
    # An estimate outside the coronal scope warns; one estimated nothing
    # (``ap_estimate`` null: an explicit window, or the estimate switched off)
    # stays pending; an explicit flat key wins over the nested one.
    outside = ap_fields({}, {"ap_estimate": {**nested, "ap_mm": 11.5, "ap_bin": None}})
    assert outside["ap_in_scope"] is False and outside["ap_bin"] is None
    assert "11.50 mm is outside" in outside["warnings"][0]
    assert ap_fields({}, {"ap_estimate": None})["ap_status"] == PENDING
    flat = ap_fields({"ap_estimate_mm": 7.0}, {"ap_estimate": nested})
    assert flat["ap_estimate_mm"] == 7.0 and flat["ap_bin"] == "4.5-6.0"


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
    assert variant.value == "v1" and variant.status == "measured"
    (relabelled,) = metrics.find("MO2", "n_relabelled")
    assert relabelled.value > 0
    (outside,) = metrics.find("MO2", "class_changes_outside_dropped")
    assert outside.value == 0.0
    (in_dropped,) = metrics.find("MO2", "pruned_calls_in_dropped_nodes")
    assert in_dropped.value == 0
    assert metrics.find("MO6", "f1_region_incoherent_rate")
    (level,) = metrics.find("MO7", "mouse_gate_level")
    sample_summary = summary["samples"][MOUSE_SID]
    gate = (
        sample_summary.get("mouse_gate")
        or (sample_summary.get("resolution") or {}).get("gate")
        or {}
    )
    assert level.value == gate["level"]
    # The dataset gate records of item 1 are MO7's for a mouse section.
    gate_records = metrics.find("MO7", "gate_level")
    assert (
        gate_records
        and gate_records[0].value == (sample_summary["resolution"]["gate"]["level"])
    )
    assert not metrics.find("H8")
    # MO4 shares are over the allocated soft mass (G4's definition).
    labels = pd.read_parquet(
        next((tmp_path / "resolve").rglob(f"{MOUSE_SID}_celltype_labels.parquet"))
    )
    table = labels[labels["in_table"]]
    soft = table[[c for c in table.columns if c.startswith("soft_class_")]]
    allocated = soft.to_numpy(float)
    (astro,) = metrics.find("MO4", "astro_epen_soft_share")
    expected = allocated[:, soft.columns.get_loc("soft_class_30_astro_epen")].sum()
    assert astro.value == pytest.approx(expected / allocated.sum(), rel=1e-5)
    (unallocated,) = metrics.find("MO4", "unallocated_soft_share")
    residual = np.clip(1.0 - allocated.sum(axis=1), 0.0, None).sum()
    assert unallocated.value == pytest.approx(residual / len(table), rel=1e-4)
    shares = metrics.find("MO4", "soft_share")
    assert shares and "unallocated" not in {record.group for record in shares}
    assert sum(record.value for record in shares) == pytest.approx(1.0, abs=1e-4)
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
    assert not [banner for banner in item.banners if banner.code.startswith("anter")]
    assert any("is anterior" in note for note in item.notes)
    # An M6b AP estimate below 6.0 mm: the MO11 banner (plan §14).
    manifest_path = map_dir / "map_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    samples = manifest["samples"]
    record = (
        samples[MOUSE_SID]
        if isinstance(samples, dict)
        else next(entry for entry in samples if entry["sample_id"] == MOUSE_SID)
    )
    record.setdefault("mouse_regions", {})["ap_estimate_mm"] = 5.2
    manifest_path.write_text(json.dumps(manifest))
    # A gate that is not full must be reported as such (not a fixed level).
    summary_path = next((tmp_path / "resolve").glob("*_resolve_summary.json"))
    edited = json.loads(summary_path.read_text())
    entry = edited["samples"][MOUSE_SID]
    target = entry.get("mouse_gate") or entry["resolution"]["gate"]
    target["level"] = "broad_only"
    summary_path.write_text(json.dumps(edited))
    anterior = build_annotation_report(
        sources,
        tmp_path / "report_anterior",
        options=ReportOptions(n_bootstrap=10),
        strict=True,
        make_figures=False,
        items=[10],
    )
    item10 = next(item for item in anterior.items if item.number == 10)
    (banner,) = [b for b in item10.banners if b.code == "anterior_section_mo11"]
    assert "MO11 not yet passed" in banner.text and "5.20 mm" in banner.text
    assert "MO11 not yet passed" in anterior.html.read_text()
    (gated,) = anterior.metrics.find("MO7", "mouse_gate_level")
    assert gated.value == "broad_only"


def test_report_reads_the_ap_estimate_that_resolve_writes(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    """G4's AP estimate fills MO11's fields and MO10's variant in the report.

    RESOLVE records the estimate under ``mouse_gate["ap_estimate"]`` (``ap_mm``,
    ``ap_bin``); the report used to look for flat ``ap_estimate_mm`` keys no
    stage writes, which left MO10 and MO11 pending after M6b.
    """
    _add_profiles(mouse_setup["bundle"].path)
    region_dir = mouse_region_share_bundle(
        tmp_path / "store_ap", build_hash="c" * 64, composition=_ap_composition()
    )
    map_dir = tmp_path / "map"
    map_mouse(
        mouse_setup, map_dir, region_shares=RegionShareBundle.from_dir(region_dir)
    )
    _resolve(
        mouse_setup,
        map_dir,
        tmp_path / "resolve",
        make_trust,
        registration={MOUSE_SID: PASSING},
    )
    summary_path = next((tmp_path / "resolve").glob("*_resolve_summary.json"))
    summary = json.loads(summary_path.read_text())
    entry = summary["samples"][MOUSE_SID]
    written = (entry.get("mouse_gate") or entry["resolution"]["gate"])["ap_estimate"]
    # Half IT-ET and half MB neurons: the MERFISH section at AP 5.0 mm.
    assert written["ap_mm"] == 5.0 and written["ap_bin"] == "4.5-6.0"
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
        items=[10],
    )
    (ap,) = result.metrics.find("MO11", "ap_estimate_mm")
    assert ap.value == 5.0 and ap.status == "measured" and ap.note == ""
    (variant,) = result.metrics.find("MO10", "m6b_rule_variant")
    assert variant.value == "v1" and variant.status == "measured"
    item = next(item for item in result.items if item.number == 10)
    # AP 5.0 mm < 6.0 mm: the MO11 banner; nothing is reported as pending.
    (banner,) = [b for b in item.banners if b.code == "anterior_section_mo11"]
    assert "5.00 mm" in banner.text
    assert not any("pending (M6b)" in note for note in item.notes)
    regions = pd.read_csv(
        tmp_path / "report" / "tables" / "item10_mouse_regions__regions.csv"
    )
    row = regions.iloc[0]
    assert row["ap_status"] == "measured" and row["ap_bin"] == "4.5-6.0"
    assert row["ap_estimate_mm"] == 5.0 and row["m6b_rule_variant"] == "v1"


def test_mo4_shares_renormalise_the_residual_away() -> None:
    from merxen.annotation.report_mouse import allocated_class_shares

    soft = np.array([[0.6, 0.2, 0.2], [0.3, 0.3, 0.4]])  # two classes + residual
    shares, unallocated = allocated_class_shares(soft)
    np.testing.assert_allclose(shares, [0.9 / 1.4, 0.5 / 1.4])
    assert unallocated == pytest.approx(0.6 / 2.0)
    assert shares.sum() == pytest.approx(1.0)
