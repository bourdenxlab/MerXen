"""Tests of the real-data QC in mouse RESOLVE (plan §8.8; M13b chunk C22).

Mouse RESOLVE runs the QC on every sample (``real_qc.enabled``), warn-only:
the outcomes go to the sample's ``real_qc`` summary block (with the G1-G5
records of the mouse gate and G4's AP estimate) and to
``PanelProvenance.real_data_qc``; the pair summary gains ``real_qc_config``.
Nothing else moves: the label table, the mouse gate and every other record
equal those of the QC-free run (``real_qc.enabled`` false).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from merxen.annotation import real_qc as qc
from merxen.annotation.config import AnnotationConfig
from merxen.annotation.mouse_regions import RegionShareBundle
from merxen.annotation.pipeline import ResolveResult, read_label_table

from .conftest import MOUSE_SID as SID
from .conftest import map_mouse, mouse_region_share_bundle
from .test_pipeline_resolve_mouse import (
    PASSING,
    _add_profiles,
    _ap_composition,
    _config,
    _resolve,
)

MakeTrust = Callable[..., Any]
# The checks a version-6, <= 1,000-gene, single-platform mouse section
# records (ag7 and VZG2): the gate's G1-G5 and the checks real_qc adds.
MOUSE_CHECKS = {
    "dataset_gate",
    "registration_g1",
    "marker_consistency",
    "g3_implausibility",
    "g4_composition",
    "g5_spillover",
    "paired_concordance",
    "flag_rates",
    "gene_complexity",
    "prefilter_spotcheck",
    "coverage_vs_simulation",
    "nonneuronal_depth_trend",
    "factor_remeasure",
}
NOT_APPLICABLE = {
    "paired_concordance",
    "prefilter_spotcheck",
    "coverage_vs_simulation",
    "nonneuronal_depth_trend",
    "factor_remeasure",
}


def _with_qc(config: AnnotationConfig, **update: Any) -> AnnotationConfig:
    return config.model_copy(
        update={"real_qc": config.real_qc.model_copy(update=update)}
    )


def _ap_setup(tmp_path: Path, mouse_setup: dict[str, Any]) -> Path:
    """MAP the fake section, with version-6 tables, against a bundle with sections.

    As ag7 and VZG2: a version-6 ``wmb_panel`` bundle of <= 1,000 genes (no
    prefilter, no R3 member) and a region-share bundle with MERFISH sections,
    so that G4's window comes from the AP estimate.
    """
    from merxen.annotation.config import AnnotationThresholds

    from .test_resolve_v7 import (
        MOUSE_GRID,
        MOUSE_HIGH_DEPTH,
        MOUSE_NEURONAL,
        MOUSE_PLAN,
        MOUSE_TRUTH,
        mouse_levels,
        write_tables,
    )

    write_tables(
        mouse_setup["bundle"].path,
        version=6,
        levels=mouse_levels(AnnotationThresholds()),
        grid=MOUSE_GRID,
        plan=MOUSE_PLAN,
        truth=MOUSE_TRUTH,
        neuronal=MOUSE_NEURONAL,
        high_depth=MOUSE_HIGH_DEPTH,
    )
    _add_profiles(mouse_setup["bundle"].path)
    region_dir = mouse_region_share_bundle(
        tmp_path / "store_ap", build_hash="c" * 64, composition=_ap_composition()
    )
    map_dir = tmp_path / "map"
    map_mouse(
        mouse_setup, map_dir, region_shares=RegionShareBundle.from_dir(region_dir)
    )
    return map_dir


def _pair(
    tmp_path: Path,
    mouse_setup: dict[str, Any],
    make_trust: MakeTrust,
    map_dir: Path,
    **update: Any,
) -> tuple[ResolveResult, ResolveResult]:
    """RESOLVE with the QC and QC-free, on the same MAP output."""
    runs = []
    for name, enabled in (("resolve_qc", True), ("resolve_free", False)):
        runs.append(
            _resolve(
                mouse_setup,
                map_dir,
                tmp_path / name,
                make_trust,
                config=_with_qc(_config(), enabled=enabled, **update),
                registration={SID: PASSING},
            )
        )
    return runs[0], runs[1]


def test_mouse_resolve_records_the_real_data_qc(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    map_dir = _ap_setup(tmp_path, mouse_setup)
    result, _ = _pair(tmp_path, mouse_setup, make_trust, map_dir)
    sample = result.samples[SID]
    block = sample.summary["real_qc"]
    assert block["enabled"] is True and block["species"] == "mouse"
    assert block["seeded"] is True and block["warn_only"] is True
    assert block["effects_withheld"] == qc.MOUSE_RESOLVE_WARN_ONLY_NOTE
    assert set(block["per_check"]) == MOUSE_CHECKS
    states = {item["check"]: item["state"] for item in block["outcomes"]}
    assert {check for check, state in states.items() if state == "not_applicable"} == (
        NOT_APPLICABLE
    )
    reasons = {item["check"]: item["reason"] for item in block["outcomes"]}
    assert all(reasons[check] for check in NOT_APPLICABLE)
    assert reasons["paired_concordance"].startswith("unpaired")
    assert states["flag_rates"] == "evaluated"
    assert block["per_check"]["gene_complexity"] == "not_evaluable"
    assert block["downgrades"] == [] and block["effects"]["gate_cap"] is None
    # G1-G5 as the gate decided them; G4 against the AP estimate's window.
    gate = sample.summary["mouse_gate"]
    signals = block["gate_signals"]
    assert {key: item["status"] for key, item in signals.items()} == (
        gate["signal_status"]
    )
    g4 = signals["g4"]
    assert g4["window_source"] == "ap_estimate"
    assert g4["ap_estimate"] == gate["ap_estimate"]
    assert g4["ap_estimate"]["section"] == "S2"
    assert g4["window"]["sections"] == ["S1", "S2", "S3"]
    assert block["per_check"]["g4_composition"] == "warn"
    assert block["per_check"]["registration_g1"] == "pass"
    # The provenance records the outcomes; the pair records the QC config.
    record = sample.provenance.panel.real_data_qc
    assert record is not None and record.warn_only is True
    assert record.outcomes == block["per_check"] and record.downgrades == []
    summary = json.loads(result.summary_path.read_text())
    assert summary["real_qc_config"] == _config().real_qc.model_dump(mode="json")
    assert summary["samples"][SID]["real_qc"] == json.loads(json.dumps(block))
    _, provenance = read_label_table(sample.labels_path)
    assert provenance is not None and provenance.panel is not None
    assert provenance.panel.real_data_qc == record


def test_mouse_real_data_qc_changes_nothing_but_its_records(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    """Warn-only: a fired QC warning reaches neither the gate nor the labels."""
    map_dir = _ap_setup(tmp_path, mouse_setup)
    # Every uninformative flag stratum warns (a fired outcome).
    qc_run, free = _pair(
        tmp_path, mouse_setup, make_trust, map_dir, uninformative_strata_warn_frac=0.0
    )
    sample, other = qc_run.samples[SID], free.samples[SID]
    block = sample.summary["real_qc"]
    fired = [item for item in block["outcomes"] if item["fired"]]
    assert fired and block["effects"]["warning_reasons"]
    assert all(item["effect"] in ("warning", "report_only") for item in fired)
    # The QC-free run records the QC as disabled.
    assert other.summary["real_qc"] == {
        "enabled": False,
        "version": qc.REAL_QC_VERSION,
        "promotes": False,
    }
    assert other.provenance.panel.real_data_qc is None
    # The gate, the summary and the provenance differ only by the QC records.
    assert sample.summary["mouse_gate"] == other.summary["mouse_gate"]
    assert not any(
        reason.startswith("real_qc_")
        for reason in sample.summary["mouse_gate"]["warning_reasons"]
    )

    def without_qc(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key not in ("real_qc", "labels_sha256")
        }

    assert without_qc(sample.summary) == without_qc(other.summary)
    with_qc = sample.provenance.model_dump(mode="json")
    with_qc["panel"]["real_data_qc"] = None
    assert with_qc == other.provenance.model_dump(mode="json")
    labels, _ = read_label_table(sample.labels_path)
    free_labels, _ = read_label_table(other.labels_path)
    pd.testing.assert_frame_equal(labels, free_labels)
    pair = json.loads(qc_run.summary_path.read_text())
    free_pair = json.loads(free.summary_path.read_text())
    assert pair["real_qc_config"]["enabled"] is True
    assert free_pair["real_qc_config"]["enabled"] is False
    for record in (pair, free_pair):
        record.pop("real_qc_config")
        record["samples"] = {
            key: without_qc(value) for key, value in record["samples"].items()
        }
    assert pair == free_pair


def test_a_failed_g1_is_recorded_from_the_gate_not_reapplied(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    from merxen.annotation.mouse_gate import RegistrationSignal

    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    result = _resolve(
        mouse_setup,
        map_dir,
        tmp_path / "resolve",
        make_trust,
        registration={SID: RegistrationSignal(density_ratio=1.1, shift_um=0.0)},
    )
    sample = result.samples[SID]
    block = sample.summary["real_qc"]
    assert sample.summary["mouse_gate"]["level"] == "failed"
    assert block["per_check"]["dataset_gate"] == "fail"
    assert block["per_check"]["registration_g1"] == "fail"
    assert block["gate_signals"]["g1"]["values"]["g1_density_ratio"] == 1.1
    # The gate failed the section itself; the QC adds no effect of its own.
    assert block["downgrades"] == [] and block["effects"]["gate_cap"] is None
    assert "registration_g1" not in {item["check"] for item in block["outcomes"]}


def test_the_panel_card_lists_the_mouse_real_qc_outcomes(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    from merxen.annotation.report_panel import real_qc_rows

    map_dir = _ap_setup(tmp_path, mouse_setup)
    result, _ = _pair(tmp_path, mouse_setup, make_trust, map_dir)
    rows = real_qc_rows(SID, result.samples[SID].summary["real_qc"])
    assert {row["check"] for row in rows} >= NOT_APPLICABLE | {"flag_rates"}
    assert all(row["warn_only"] for row in rows)
