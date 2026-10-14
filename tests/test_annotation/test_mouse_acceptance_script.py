"""Tests for the M9 mouse acceptance scorer (pre-registration §9, §24, §25).

Synthetic fixtures: each criterion is driven to pass, fail and not evaluable,
the exceptions are checked to apply only inside their scope, the provenance
check is checked to refuse bundles outside the store without touching them,
and the registered readings are checked to be the defaults.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "acceptance"


def _load() -> ModuleType:
    name = "mouse_acceptance"
    if name in sys.modules:
        return sys.modules[name]
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load()
PAIR = {"ag7": "ag7", "vzg2": "MsBrain_VZG2"}
SID = {"ag7": "ag7_MERSCOPE", "vzg2": "MsBrain_VZG2_MERSCOPE"}
MANUAL = ["Isocortex", "OLF", "HPF", "CTXsp", "TH", "HY", "MB"]
PANEL_HASH = "f4c29186" + "0" * 56
PANEL_BUILD = "97a086d1" + "1" * 56
SHARE_BUILD = "1867f978" + "2" * 56


def _metric(criterion: str, name: str, value: Any, **extra: Any) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "name": name,
        "value": value,
        "status": "measured" if value is not None else "not_available",
        "group": None,
        "kind": None,
        "level": None,
        "n": None,
        **extra,
    }


SOFT = {
    "01 IT-ET Glut": 0.40,
    "30 Astro-Epen": 0.19,
    "31 OPC-Oligo": 0.20,
    "33 Vascular": 0.19,
    "34 Immune": 0.02,
}


def _records(
    *,
    astro: float = 0.19,
    immune: float = 0.02,
    enrichment: float = 38.0,
    fpr: float = 0.003,
    f1: float = 0.003,
    soft: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    records = [
        _metric("MO4", "astro_epen_soft_share", astro),
        _metric("MO4", "immune_soft_share", immune),
        _metric("MO5", "spillover_heldout_enrichment", enrichment),
        _metric("MO5", "spillover_astrocyte_fpr", fpr),
        _metric("MO6", "f1_region_incoherent_rate", f1),
        _metric("MO2", "pruned_calls_in_dropped_nodes", 0),
        _metric("MO2", "class_changes_outside_dropped", 0.0),
        _metric("MO1", "g2_marker_consistency", 0.82),
        _metric("MO7", "mouse_gate_level", "full"),
        _metric(
            "report",
            "prep_bins_emitted",
            150,
            level="class",
            kind="wmb_panel:validated",
            n=272,
        ),
    ]
    for name, share in (soft or SOFT).items():
        records.append(_metric("MO4", "soft_share", share, group=name, level="class"))
    return records


def _seg(
    dataset: str,
    segmentation: str,
    *,
    records: list[dict[str, Any]] | None = None,
    gate: dict[str, Any] | None = None,
    regions: dict[str, Any] | None = None,
    processes: int = 4,
    root: Path | None = None,
) -> Any:
    out = M.SegOutputs(
        dataset=dataset, segmentation=segmentation, root=root or Path("/nonexistent")
    )
    out.report = {"pair_id": PAIR[dataset], "metrics": records or []}
    out.resolve_summary = {
        "pair_id": PAIR[dataset],
        "samples": {
            SID[dataset]: {"mouse_gate": gate or {"level": "full", "signal_status": {}}}
        },
    }
    out.manifest = {
        "pair_id": PAIR[dataset],
        "n_processors": processes,
        "samples": {
            SID[dataset]: {
                "runs": {"wmb_panel": {"engine_params": {"n_processors": processes}}},
                "mouse_regions": regions
                if regions is not None
                else {"inferred_regions": MANUAL, "rule_variant": "v1"},
            }
        },
    }
    return out


def _outputs(**overrides: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dataset in M.DATASETS:
        out[dataset] = {}
        for seg in M.SEGMENTATIONS:
            key = (dataset, seg)
            out[dataset][seg] = overrides.get(
                f"{dataset}_{seg}", _seg(dataset, seg, records=_records())
            )
            del key
    return out


def _by(rows: list[Any], part: str, dataset: str | None = None) -> list[Any]:
    return [
        row
        for row in rows
        if row.part == part and (dataset is None or row.dataset == dataset)
    ]


def _verdicts(rows: list[Any]) -> list[str]:
    return [M.finalize(row).verdict for row in rows]


# ---------------------------------------------------------------------------
# readings


def test_registered_readings_are_the_defaults() -> None:
    readings = M.resolve_readings({}, exploratory=False)
    assert {name: value["value"] for name, value in readings.items()} == {
        "mo2_remap": "both",
        "mo4_window": "fixed",
        "mo8_map_memory": "trace_rss",
        "mo8_map_processes": "4",
        "mo8_prep_memory": "tree_rss",
        "mo10_t2": "allen_relative",
    }
    assert all(value["is_registered"] for value in readings.values())
    assert "§25.2" in readings["mo2_remap"]["source"]
    assert "LOOSENING, POST-HOC" in readings["mo10_t2"]["source"]


def test_a_non_registered_reading_needs_exploratory() -> None:
    with pytest.raises(ValueError, match="--exploratory"):
        M.resolve_readings({"mo8_map_memory": "tree_pss"}, exploratory=False)
    with pytest.raises(ValueError, match="--exploratory"):
        M.resolve_readings({"mo8_map_processes": "6"}, exploratory=False)
    readings = M.resolve_readings({"mo8_map_memory": "tree_pss"}, exploratory=True)
    assert not readings["mo8_map_memory"]["is_registered"]
    with pytest.raises(ValueError, match="not in"):
        M.resolve_readings({"mo2_remap": "neither"}, exploratory=True)


# ---------------------------------------------------------------------------
# MO1


def test_mo1_pass_fail_and_not_evaluable() -> None:
    outputs = _outputs()
    measured = {
        "ag7": {"mo1_pruned": 0.90, "mo1_unpruned": 0.9, "n_pseudo_confident": 100},
        "vzg2": {"mo1_pruned": 0.83, "mo1_unpruned": 0.83, "n_pseudo_confident": 100},
    }
    rows = M.mo1_rows(outputs, {}, measured)
    assert _verdicts(rows) == ["PASS", "FAIL"]
    assert rows[0].threshold == 0.89 and rows[1].threshold == 0.84
    assert rows[0].segmentation == "proseg_hybrid"
    assert rows[1].segmentation == "original_seg"
    # No label table and no measurement: not evaluable.
    rows = M.mo1_rows(outputs, {}, None)
    assert _verdicts(rows) == ["NOT_AVAILABLE", "NOT_AVAILABLE"]
    assert "no label table" in rows[0].note
    rows = M.mo1_rows({"vzg2": outputs["vzg2"]}, {}, None)
    assert "no ag7 results root" in rows[0].note


def _labels(classes: list[str], subclasses: list[str]) -> pd.DataFrame:
    n = len(classes)
    return pd.DataFrame(
        {
            "in_table": [True] * n,
            "total_counts": [200.0] * n,
            "ct_class_status": ["confident"] * n,
            "ct_class_corr": [0.6] * n,
            "mmc_wmb_class_name": classes,
            "mmc_wmb_subclass_name": subclasses,
            "mmc_wmb_unpruned_class_name": classes,
            "mmc_wmb_unpruned_subclass_name": subclasses,
        },
        index=pd.Index([f"c{i}" for i in range(n)], name="cell_id"),
    )


def test_mo1_measure_uses_the_corr_shadow_pseudo_labels() -> None:
    classes = ["01 IT-ET Glut"] * 9 + ["30 Astro-Epen"]
    subclasses = ["007 L2/3 IT CTX Glut"] * 9 + ["319 Astrocyte NN"]
    labels = _labels(classes, subclasses)
    pseudo = pd.DataFrame(
        {"pseudo_class": ["Glut"] * 10}, index=pd.Index(labels.index, name=None)
    )
    result = M.mo1_measure("vzg2", labels, pseudo)
    assert result["n_pseudo_confident"] == 10
    assert result["mo1_pruned"] == pytest.approx(0.9)
    fine = pd.DataFrame({"pseudo_fine": ["Glut"] * 9 + ["Astro"]}, index=labels.index)
    result = M.mo1_measure("ag7", labels, fine)
    assert result["mo1_pruned"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MO2


def test_mo2_from_calls_counts_changes_outside_dropped_nodes() -> None:
    result = M.mo2_from_calls(
        unpruned_class=np.array(["A", "A", "B", "B"], dtype=object),
        unpruned_subclass=np.array(["a1", "a2", "b1", "b2"], dtype=object),
        pruned_class=np.array(["A", "C", "B", "B"], dtype=object),
        pruned_subclass=np.array(["a1", "c1", "b1", "b3"], dtype=object),
        pruned_subclass_name=np.array(["n1", "n2", "n3", "n4"], dtype=object),
        remapped=np.array([False, True, False, False]),
        t2={"n4"},
    )
    assert result["n_outside"] == 3 and result["n_remapped"] == 1
    assert result["class_changes_outside"] == 0.0
    assert result["subclass_changes_outside"] == pytest.approx(1 / 3)
    assert result["t2_after"] == pytest.approx(0.25)


def _mo2_values(t2: float = 0.0, klass: float = 0.0, sub: float = 0.0) -> dict:
    return {
        "t2_after": t2,
        "class_changes_outside": klass,
        "subclass_changes_outside": sub,
        "n_cells": 1000,
        "n_outside": 990,
        "n_remapped": 10,
        "n_t2": 0,
    }


def test_mo2_scores_both_re_maps_under_the_registered_reading() -> None:
    outputs = _outputs()
    measured = {
        "ag7": {
            "subset": _mo2_values(),
            "full": _mo2_values(klass=0.00245, sub=0.0124),
            "full_record": {"n_processors": 4},
            "map_processes": 4,
        },
        "vzg2": {
            "subset": _mo2_values(t2=0.0012),
            "full": None,
            "map_processes": 4,
        },
    }
    rows = M.mo2_rows(outputs, measured, "both")
    ag7 = [M.finalize(row) for row in rows if row.dataset == "ag7" and row.scored]
    assert [row.verdict for row in ag7] == ["PASS"] * 6
    vzg2 = {
        (row.scope["remap"], row.part): M.finalize(row).verdict
        for row in rows
        if row.dataset == "vzg2" and row.scored
    }
    assert vzg2[("subset", "t2_after")] == "FAIL"
    assert vzg2[("full", "t2_after")] == "NOT_AVAILABLE"
    # The subset reading reports the full re-map instead of scoring it.
    rows = M.mo2_rows(outputs, measured, "subset")
    full = [row for row in rows if row.scope.get("remap") == "full"]
    assert full and not any(row.scored for row in full)


def test_mo2_full_re_map_with_another_process_count_is_unscored() -> None:
    measured = {
        "ag7": {
            "subset": _mo2_values(),
            "full": _mo2_values(),
            "full_record": {"n_processors": 6},
            "map_processes": 4,
        }
    }
    rows = M.mo2_rows(_outputs(), measured, "both")
    full = [
        M.finalize(r)
        for r in rows
        if r.dataset == "ag7" and r.scope.get("remap") == "full"
    ]
    assert {row.verdict for row in full} == {"UNSCORED (process count)"}
    assert all(row.back_to_user() for row in full)
    # A full re-map of another MAP output is not used, and says why.
    measured["ag7"].update(full=None, full_problem="the full re-map ... differ")
    rows = M.mo2_rows(_outputs(), measured, "both")
    full = [
        M.finalize(r)
        for r in rows
        if r.dataset == "ag7" and r.scope.get("remap") == "full"
    ]
    assert {row.verdict for row in full} == {"NOT_AVAILABLE"}
    assert full[0].note == "the full re-map ... differ"


# ---------------------------------------------------------------------------
# MO3


def test_mo3_pass_fail_and_not_evaluable() -> None:
    wrong = _seg(
        "vzg2",
        "original_seg",
        records=_records(),
        regions={"inferred_regions": [*MANUAL, "OB"], "rule_variant": "v1"},
    )
    empty = _seg("ag7", "proseg_hybrid", records=_records(), regions={})
    rows = M.mo3_rows(_outputs(vzg2_original_seg=wrong, ag7_proseg_hybrid=empty))
    scored = {row.dataset: M.finalize(row).verdict for row in rows if row.scored}
    assert scored == {"ag7": "NOT_AVAILABLE", "vzg2": "FAIL"}
    rows = M.mo3_rows(_outputs())
    assert {M.finalize(row).verdict for row in rows if row.scored} == {"PASS"}
    assert {M.finalize(row).verdict for row in rows if not row.scored} == {"INFO pass"}


# ---------------------------------------------------------------------------
# MO4


WINDOW = {"30 Astro-Epen": 0.1736, "34 Immune": 0.0152}


def test_mo4_bands_on_the_fixed_window() -> None:
    windows = {(d, M.MO4_FIXED_WINDOW): WINDOW for d in M.DATASETS}
    outputs = _outputs(
        vzg2_original_seg=_seg(
            "vzg2", "original_seg", records=_records(astro=0.2100, immune=0.0160)
        ),
        ag7_proseg_hybrid=_seg(
            "ag7", "proseg_hybrid", records=_records(astro=0.1910, immune=0.0300)
        ),
    )
    rows = {
        (r.dataset, r.part): M.finalize(r)
        for r in M.mo4_rows(outputs, "fixed", windows)
    }
    # VZG2 Astro-Epen +3.64 points > 3; Immune +0.08 <= 0.5.
    assert rows[("vzg2", "astro_epen_points")].verdict == "FAIL"
    assert rows[("vzg2", "astro_epen_points")].value == pytest.approx(3.64)
    assert rows[("vzg2", "immune_points")].verdict == "PASS"
    # ag7: Astro-Epen +1.74 <= 5; Immune +1.48 > 1.
    assert rows[("ag7", "astro_epen_points")].verdict == "PASS"
    assert rows[("ag7", "immune_points")].verdict == "FAIL"
    # No window composition: not evaluable.
    rows = {
        (r.dataset, r.part): M.finalize(r) for r in M.mo4_rows(_outputs(), "fixed", {})
    }
    assert {row.verdict for row in rows.values()} == {"NOT_AVAILABLE"}


def test_mo4_ap_estimate_window_is_reported_beside_and_scored_only_when_asked() -> None:
    gate = {"level": "full", "window": {"sections": ["S32", "S33", "S35"]}}
    outputs = _outputs(
        ag7_proseg_hybrid=_seg(
            "ag7", "proseg_hybrid", records=_records(astro=0.19), gate=gate
        )
    )
    windows = {
        ("ag7", M.MO4_FIXED_WINDOW): WINDOW,
        ("ag7", ("S32", "S33", "S35")): {"30 Astro-Epen": 0.1840, "34 Immune": 0.0152},
        ("vzg2", M.MO4_FIXED_WINDOW): WINDOW,
    }
    rows = [r for r in M.mo4_rows(outputs, "fixed", windows) if r.dataset == "ag7"]
    astro = _by(rows, "astro_epen_points")[0]
    assert astro.value == pytest.approx(1.64)
    assert astro.beside["ap_estimate_points"] == pytest.approx(0.6)
    assert astro.beside["ap_estimate_window"] == ["S32", "S33", "S35"]
    rows = [
        r for r in M.mo4_rows(outputs, "ap_estimate", windows) if r.dataset == "ag7"
    ]
    assert _by(rows, "astro_epen_points")[0].value == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# MO5, MO6, MO7


def test_mo5_pass_fail_and_not_evaluable() -> None:
    outputs = _outputs(
        vzg2_original_seg=_seg(
            "vzg2", "original_seg", records=_records(enrichment=9.5, fpr=0.004)
        ),
        ag7_proseg_hybrid=_seg("ag7", "proseg_hybrid", records=[]),
    )
    rows = [M.finalize(row) for row in M.mo5_rows(outputs) if row.scored]
    got = {(row.dataset, row.part): row.verdict for row in rows}
    assert got == {
        ("vzg2", "heldout_enrichment"): "FAIL",
        ("vzg2", "astrocyte_fpr"): "PASS",
        ("ag7", "astrocyte_fpr"): "NOT_AVAILABLE",
    }


@pytest.mark.parametrize(
    ("rate", "verdict"),
    [
        (0.002, "PASS"),
        (0.0099, "PASS"),
        (0.0015, "FAIL"),
        (0.012, "FAIL"),
        (None, "NOT_AVAILABLE"),
    ],
)
def test_mo6_range(rate: float | None, verdict: str) -> None:
    outputs = _outputs(
        vzg2_original_seg=_seg("vzg2", "original_seg", records=_records(f1=rate))  # type: ignore[arg-type]
    )
    rows = [r for r in M.mo6_rows(outputs) if r.scored and r.dataset == "vzg2"]
    assert _verdicts(rows) == [verdict]


def test_mo7_negative_control_and_not_failed() -> None:
    negative = {"samples": {"x": {"mouse_gate": {"level": "failed"}}}}
    failed = _seg("ag7", "proseg_hybrid", gate={"level": "failed"})
    rows = [
        M.finalize(r)
        for r in M.mo7_rows(_outputs(ag7_proseg_hybrid=failed), negative, None)
    ]
    scored = {(r.dataset, r.part): r.verdict for r in rows if r.scored}
    assert scored == {
        ("vzg2_invalid", "negative_control_failed"): "PASS",
        ("ag7", "not_failed"): "FAIL",
        ("vzg2", "not_failed"): "PASS",
    }
    full = {"samples": {"x": {"mouse_gate": {"level": "full"}}}}
    rows = [M.finalize(r) for r in M.mo7_rows(_outputs(), full, None)]
    assert rows[0].verdict == "FAIL"
    rows = [M.finalize(r) for r in M.mo7_rows(_outputs(), None, None)]
    assert rows[0].verdict == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# MO8


def _trace(tmp_path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "name",
        "process",
        "tag",
        "status",
        "exit",
        "cpus",
        "realtime",
        "peak_rss",
        "workdir",
    ]
    frame = pd.DataFrame(
        [{column: row.get(column, "") for column in columns} for row in rows],
        columns=columns,
    )
    path = tmp_path / "trace.tsv"
    frame.to_csv(path, sep="\t", index=False)
    from run_acceptance import load_trace

    return load_trace([path])


def _section_tasks(
    pair: str,
    *,
    cpus: int = 4,
    peak: str = "11 GB",
    map_time: str = "7m",
    segs: tuple = M.SEGMENTATIONS,
) -> list[dict[str, Any]]:
    rows = []
    for seg in segs:
        for process, realtime, rss in (
            ("ANNOTATE_PANEL", "12s", "1.4 GB"),
            ("CLUSTERING_SQUIDPY_ANNOTATE_MAP", map_time, peak),
            ("CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE", "55s", "3 GB"),
        ):
            rows.append(
                {
                    "name": f"{process} ({pair}:{seg})",
                    "process": f"WF:{process}",
                    "tag": f"{pair}:{seg}",
                    "status": "COMPLETED",
                    "exit": "0",
                    "cpus": str(cpus) if "MAP" in process else "1",
                    "realtime": realtime,
                    "peak_rss": rss,
                }
            )
    return rows


def test_mo8_section_pass_fail_not_evaluable_and_unscored(tmp_path: Path) -> None:
    outputs = _outputs()
    trace = _trace(
        tmp_path,
        _section_tasks("ag7", peak="11 GB")
        + _section_tasks(PAIR["vzg2"], peak="14.7 GB", segs=M.SEGMENTATIONS[:3]),
    )
    rows, problems = M.mo8_section_rows(
        outputs, trace, {}, memory_reading="trace_rss", processes=4
    )
    got = {(r.dataset, r.part): M.finalize(r) for r in rows}
    assert got[("ag7", "map_peak")].verdict == "PASS"
    assert got[("ag7", "map_peak")].value == pytest.approx(11.0)
    assert got[("ag7", "section_time")].verdict == "PASS"
    assert got[("ag7", "section_time")].value == pytest.approx(
        4 * (12 + 420 + 55) / 3600
    )
    # VZG2: one segmentation missing from the trace: time and peak not measured.
    assert got[("vzg2", "section_time")].verdict == "NOT_AVAILABLE"
    assert got[("vzg2", "map_peak")].verdict == "NOT_AVAILABLE"
    assert not problems
    # A peak above 12 GB fails; a MAP at 6 processes leaves the rows unscored.
    trace = _trace(tmp_path, _section_tasks("ag7", peak="12.5 GB"))
    rows, _ = M.mo8_section_rows(
        outputs, trace, {}, memory_reading="trace_rss", processes=4
    )
    assert _by(rows, "map_peak", "ag7")[0].passes is False
    assert _verdicts(_by(rows, "map_peak", "ag7")) == ["FAIL"]
    trace = _trace(tmp_path, _section_tasks("ag7", cpus=6))
    rows, problems = M.mo8_section_rows(
        outputs, trace, {}, memory_reading="trace_rss", processes=4
    )
    assert set(
        _verdicts(_by(rows, "map_peak", "ag7") + _by(rows, "section_time", "ag7"))
    ) == {"UNSCORED (process count)"}
    assert problems and "not 4" in problems[0]


def test_mo8_sampler_pss_is_reported_beside(tmp_path: Path) -> None:
    sampler = tmp_path / "sampler.csv"
    pd.DataFrame(
        {
            "t_s": [0.0, 2.0],
            "vmrss_kb": [1048576, 7 * 1048576],
            "vmhwm_kb": [1048576, 11 * 1048576],
            "pss_kb": [524288, 5 * 1048576],
        }
    ).to_csv(sampler, index=False)
    peaks = M.read_sampler(sampler)
    assert peaks == {
        "tree_pss_gb": 5.0,
        "summed_vmhwm_gb": 11.0,
        "summed_vmrss_gb": 7.0,
    }
    trace = _trace(tmp_path, _section_tasks("ag7", peak="13 GB"))
    samplers = {("ag7", seg): peaks for seg in M.SEGMENTATIONS}
    rows, _ = M.mo8_section_rows(
        _outputs(), trace, samplers, memory_reading="trace_rss", processes=4
    )
    peak = _by(rows, "map_peak", "ag7")[0]
    assert peak.value == pytest.approx(13.0)
    assert peak.beside["sampler_tree_pss_gb"]["reseg"] == 5.0
    rows, _ = M.mo8_section_rows(
        _outputs(), trace, samplers, memory_reading="tree_pss", processes=4
    )
    assert _by(rows, "map_peak", "ag7")[0].value == 5.0


def _bundle(
    store: Path,
    reference: str,
    build: str,
    *,
    wall: float = 3101.0,
    tree_rss: float = 30.0,
    processes: int = 8,
    panel: str | None = PANEL_HASH,
) -> Path:
    path = store / reference / build
    path.mkdir(parents=True)
    (path / "bundle.json").write_text(
        json.dumps(
            {
                "reference_id": reference,
                "build_hash": build,
                "panel": {"panel_hash": panel} if panel else None,
                "wall_time_s": wall,
                "builder_output": {
                    "ctm_steps": {
                        "query_markers": {
                            "peak_tree_rss_gb": tree_rss,
                            "peak_tree_pss_gb": tree_rss / 2,
                            "peak_rss_gb": tree_rss / 2.2,
                        },
                        "reference_markers": {
                            "peak_tree_rss_gb": 5.0,
                            "peak_tree_pss_gb": 2.0,
                            "peak_rss_gb": 1.0,
                        },
                    }
                },
            }
        )
    )
    (path / "resolvability_summary.json").write_text(
        json.dumps(
            {
                "mapping_runs": [{"n_processors": processes}],
                "validated_thresholds_would_raise": [{"level": "class"}] * 3,
            }
        )
    )
    return path


def test_mo8_prep_pass_fail_not_evaluable_and_unscored(tmp_path: Path) -> None:
    store = tmp_path / "store"
    passing = _bundle(store, "wmb_panel", "a" * 64)
    failing = _bundle(store, "wmb_panel", "b" * 64, wall=7300.0, tree_rss=56.8)
    six = _bundle(store, "wmb_panel", "c" * 64, processes=6)
    empty = store / "wmb_panel" / ("d" * 64)
    empty.mkdir(parents=True)
    share = _bundle(store, "wmb_region_share", "e" * 64, panel=None)
    trace = _trace(
        tmp_path,
        [
            {
                "name": "PREP",
                "process": "X:ANNOTATE_REFERENCE_PREP",
                "tag": f"wmb_panel:{PANEL_HASH[:12]}",
                "status": "COMPLETED",
                "cpus": "8",
                "realtime": "50m",
                "peak_rss": "41 GB",
            }
        ],
    )
    rows = M.mo8_prep_rows([passing], _trace(tmp_path, []), memory_reading="tree_rss")
    assert _verdicts(rows) == ["PASS", "PASS"]
    assert rows[1].value == 30.0 and rows[1].beside["tree_pss_gb"] == 15.0
    # The trace peak of a PREP task the run ran enters the max.
    rows = M.mo8_prep_rows([passing], trace, memory_reading="tree_rss")
    assert rows[1].value == pytest.approx(41.0) and _verdicts(rows) == ["PASS", "FAIL"]
    rows = M.mo8_prep_rows([failing], _trace(tmp_path, []), memory_reading="tree_rss")
    assert _verdicts(rows) == ["FAIL", "FAIL"]
    rows = M.mo8_prep_rows([failing], _trace(tmp_path, []), memory_reading="tree_pss")
    assert rows[1].value == pytest.approx(28.4) and _verdicts(rows)[1] == "PASS"
    rows = M.mo8_prep_rows([six], _trace(tmp_path, []), memory_reading="tree_rss")
    assert _verdicts(rows)[1] == "UNSCORED (process count)"
    rows = M.mo8_prep_rows([empty], _trace(tmp_path, []), memory_reading="tree_rss")
    assert _verdicts(rows) == ["NOT_AVAILABLE", "NOT_AVAILABLE"]
    assert rows[0].note == "no bundle.json"
    rows = M.mo8_prep_rows([share], _trace(tmp_path, []), memory_reading="tree_rss")
    assert not any(row.scored for row in rows)


# ---------------------------------------------------------------------------
# MO9, MO11 (reported)


def test_mo9_reports_registration_and_jsd() -> None:
    shifted = dict(SOFT, **{"30 Astro-Epen": 0.10, "01 IT-ET Glut": 0.49})
    outputs = _outputs(
        vzg2_reseg=_seg(
            "vzg2",
            "reseg",
            records=_records(soft=shifted),
            gate={"level": "full", "signal_status": {"g1": "fail"}},
        )
    )
    for seg in ("original_seg", "proseg_hybrid", "proseg_mask"):
        outputs["vzg2"][seg].resolve_summary["samples"][SID["vzg2"]]["mouse_gate"] = {
            "level": "full",
            "signal_status": {"g1": "pass"},
        }
    rows = [M.finalize(row) for row in M.mo9_rows(outputs)]
    assert not any(row.scored for row in rows)
    g1 = {row.segmentation: row.verdict for row in _by(rows, "registration_g1")}
    assert g1["reseg"] == "INFO fail" and g1["original_seg"] == "INFO pass"
    jsd = {r.segmentation: r for r in _by(rows, "soft_class_jsd_vs_original_seg")}
    assert jsd["proseg_hybrid"].value == pytest.approx(0.0)
    assert jsd["reseg"].value > 0.05 and jsd["reseg"].verdict == "INFO fail"
    assert M.js_distance({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_mo11_reports_anterior_sections() -> None:
    anterior = _seg("ag7", "proseg_hybrid", gate={"ap_estimate": {"ap_mm": 5.2}})
    posterior = _seg("vzg2", "original_seg", gate={"ap_estimate": {"ap_mm": 8.26}})
    rows = M.mo11_rows(
        _outputs(ag7_proseg_hybrid=anterior, vzg2_original_seg=posterior)
    )
    got = {(r.dataset, r.segmentation): r for r in rows}
    assert got[("ag7", "proseg_hybrid")].beside["anterior"] is True
    assert "MO11 applies" in got[("ag7", "proseg_hybrid")].note
    assert got[("vzg2", "original_seg")].beside["anterior"] is False
    assert not any(row.scored for row in rows)


# ---------------------------------------------------------------------------
# MO10


def _mo10(
    *,
    candidate: str = "a+f:tile200,nn2",
    missed: list[dict[str, Any]] | None = None,
    allen_ok: bool | None = True,
    bins: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    entry = {
        "bin_mean_harm_ok": True,
        "section_max_harm_ok": True,
        "t2_bin_mean_ok": allen_ok,
        "t2_bin_mean_ok_allen": allen_ok,
        "t2_bin_mean_ok_registered": False,
        "missed_only_where_harm_small": not missed,
        "max_bin_mean_harm": 0.0009,
        "max_section_harm": 0.0021,
        "max_t2_bin_mean": 0.0039,
        "missed_with_harm_above": missed or [],
        "max_oracle_harm": 0.0096,
    }
    if bins is not None:
        entry["t2_bins"] = bins
    return {
        "candidate": candidate,
        "t2_reading": "allen-relative",
        "pass": not missed,
        "gene_sets": {"merfish500": dict(entry), "merfish_ag7": dict(entry)},
        "mo2": [{"section": "ag7", "t2_after": 0.0}],
        "mo3": {"ag7": True},
    }


def test_mo10_pass_fail_and_not_evaluable() -> None:
    rows, problems = M.mo10_rows(
        _mo10(),
        reading="allen_relative",
        recorded=["v1", "a+f:tile200,nn2"],
        rule_params={},
    )
    assert not problems
    assert set(_verdicts([r for r in rows if r.scored])) == {"PASS"}
    missed = [{"section": "C57BL6J-638850.35", "missed": "STR", "harm": 0.0014}]
    rows, _ = M.mo10_rows(
        _mo10(missed=missed),
        reading="allen_relative",
        recorded=["a+f:tile200,nn2"],
        rule_params={},
    )
    failing = [r for r in rows if r.scored and M.finalize(r).verdict == "FAIL"]
    assert {(r.part, r.scope["section"], r.scope["region"]) for r in failing} == {
        ("missed_region", "C57BL6J-638850.35", "STR")
    }
    # The registered (0.1%) T2 verdict is reported beside the scored one.
    t2 = _by(rows, "t2_clause")[0]
    assert t2.passes is True and t2.beside["ok_registered_0.1pct"] is False
    # No Allen-relative verdict in mo10.json: the T2 clause is not evaluable.
    rows, _ = M.mo10_rows(
        _mo10(allen_ok=None),
        reading="allen_relative",
        recorded=["a+f:tile200,nn2"],
        rule_params={},
    )
    assert set(_verdicts(_by(rows, "t2_clause"))) == {"NOT_AVAILABLE"}
    rows, _ = M.mo10_rows(None, reading="allen_relative", recorded=[], rule_params={})
    assert _verdicts(rows) == ["NOT_AVAILABLE"]


def test_mo10_scores_t2_per_bin_when_the_bins_are_given() -> None:
    bins = [
        {"ap_bin": "3.0-4.5", "t2_after_mean": 0.0031, "t2_allen_mean": 0.0038},
        {"ap_bin": "4.5-6.0", "t2_after_mean": 0.0024, "t2_allen_mean": 0.0023},
    ]
    rows, problems = M.mo10_rows(
        _mo10(bins=bins, allen_ok=False),
        reading="allen_relative",
        recorded=["a+f:tile200,nn2"],
        rule_params={},
    )
    per_bin = {
        (r.scope["gene_set"], r.scope["ap_bin"]): M.finalize(r).verdict
        for r in _by(rows, "t2_bin_mean")
    }
    assert per_bin[("merfish500", "3.0-4.5")] == "PASS"
    assert per_bin[("merfish500", "4.5-6.0")] == "FAIL"
    assert not any(r.scored for r in _by(rows, "t2_clause"))
    assert not problems
    # A per-bin verdict that disagrees with mo10.json's is a problem.
    _, problems = M.mo10_rows(
        _mo10(bins=bins, allen_ok=True),
        reading="allen_relative",
        recorded=["a+f:tile200,nn2"],
        rule_params={},
    )
    assert problems and "disagree" in problems[0]


def test_mo10_refuses_a_candidate_the_preregistration_does_not_name_last() -> None:
    rows, problems = M.mo10_rows(
        _mo10(candidate="v1"),
        reading="allen_relative",
        recorded=["v1", "a+f:tile200,nn2"],
        rule_params={},
    )
    assert {M.finalize(r).verdict for r in rows if r.scored} == {"UNSCORED (selection)"}
    assert "last selection line" in problems[0]
    _, problems = M.mo10_rows(
        _mo10(),
        reading="allen_relative",
        recorded=["a+f:tile200,nn2"],
        rule_params={
            "ag7/proseg_hybrid": {"rule_variant": "v1", "tile_um": 150.0},
            "vzg2/original_seg": {"rule_variant": "m6b", "tile_um": 200.0},
        },
    )
    assert len(problems) == 1 and "ag7/proseg_hybrid" in problems[0]
    assert "rule_variant" in problems[0] and "tile_um" in problems[0]
    # An (f)-only candidate keeps rule_variant v1 and differs in its tiles.
    assert not M.rule_mismatches(
        "f:tile200,nn2",
        {"vzg2": {"rule_variant": "v1", "tile_um": 200.0, "min_neurons_per_tile": 2}},
    )
    assert M.rule_mismatches("v1", {"vzg2": {"tile_um": 200.0}})
    text = (
        "x\nSelected candidate (M6b): `v1`\n\n### Amendment\n"
        "Selected candidate (M6b): `a+b`\n"
    )
    assert M.selected_candidates(text) == ["v1", "a+b"]


# ---------------------------------------------------------------------------
# exceptions


EXCEPTION = {
    "id": "U1-MO10-35",
    "criterion": "MO10",
    "scope": {
        "part": "missed_region",
        "section": "C57BL6J-638850.35",
        "region": "STR",
        "value_max": 0.0014,
    },
    "approved": "2026-10-14",
    "quote": "1. go with recommendation",
    "recorded_in": "pre-registration §24",
}


def _missed_rows(harm_35: float = 0.0014) -> list[Any]:
    missed = [
        {"section": "C57BL6J-638850.35", "missed": "STR", "harm": harm_35},
        {"section": "C57BL6J-638850.46", "missed": "CTXsp", "harm": 0.00103},
    ]
    rows, _ = M.mo10_rows(
        _mo10(missed=missed),
        reading="allen_relative",
        recorded=["a+f:tile200,nn2"],
        rule_params={},
    )
    return [M.finalize(row) for row in rows]


def test_an_exception_applies_only_inside_its_scope() -> None:
    rows = _missed_rows()
    usage = M.apply_exceptions(rows, M.parse_exceptions({"exceptions": [EXCEPTION]}))
    got = {
        (r.scope["gene_set"], r.scope["section"]): r.verdict
        for r in _by(rows, "missed_region")
    }
    # Both gene sets' .35 STR rows are inside the scope; .46 is not.
    assert got == {
        ("merfish500", "C57BL6J-638850.35"): "FAIL (excepted)",
        ("merfish500", "C57BL6J-638850.46"): "FAIL",
        ("merfish_ag7", "C57BL6J-638850.35"): "FAIL (excepted)",
        ("merfish_ag7", "C57BL6J-638850.46"): "FAIL",
    }
    assert len(usage[0]["applied_to"]) == 2
    excepted = [r for r in rows if r.verdict == "FAIL (excepted)"]
    assert all(not r.back_to_user() and r.exception == "U1-MO10-35" for r in excepted)
    flip = M.flip_rule(rows, True)
    assert flip["verdict"] == "FAIL" and flip["criteria"]["MO10"] == "FAIL"
    # Narrowed to one gene set, the other set's row stays FAIL.
    narrow = dict(EXCEPTION, scope=dict(EXCEPTION["scope"], gene_set="merfish500"))
    rows = _missed_rows()
    M.apply_exceptions(rows, M.parse_exceptions([narrow]))
    got = {
        r.scope["gene_set"]: r.verdict
        for r in _by(rows, "missed_region")
        if r.scope["section"] == "C57BL6J-638850.35"
    }
    assert got == {"merfish500": "FAIL (excepted)", "merfish_ag7": "FAIL"}


def test_an_exception_does_not_cover_a_value_beyond_its_bound() -> None:
    rows = _missed_rows(harm_35=0.0020)
    usage = M.apply_exceptions(rows, M.parse_exceptions([EXCEPTION]))
    assert {r.verdict for r in _by(rows, "missed_region")} == {"FAIL"}
    assert len(usage[0]["named_outside_bounds"]) == 2
    assert (
        "outside the exception's bounds" in _by(rows, "missed_region")[0].exception_note
    )


def test_both_approved_rows_excepted_gives_a_passing_flip_criterion() -> None:
    second = dict(
        EXCEPTION,
        id="U1-MO10-46",
        scope={
            "part": "missed_region",
            "section": "C57BL6J-638850.46",
            "region": "CTXsp",
            "value_max": 0.00103,
        },
    )
    rows = _missed_rows()
    M.apply_exceptions(rows, M.parse_exceptions([EXCEPTION, second]))
    flip = M.flip_rule(rows, True)
    assert flip["criteria"]["MO10"] == "FAIL (excepted)"
    assert len(flip["excepted"]) == 4


def test_an_exception_never_touches_a_row_that_is_not_measured() -> None:
    rows = [
        M.finalize(
            M.Row(
                "MO8", "map_peak", "ag7", "all", None, "<=", 12.0, None, True, "trace"
            )
        )
    ]
    exception = dict(EXCEPTION, criterion="MO8", scope={"part": "map_peak"})
    M.apply_exceptions(rows, M.parse_exceptions([exception]))
    assert rows[0].verdict == "NOT_AVAILABLE"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": ""}, "unique id"),
        ({"criterion": "MO99"}, "unknown criterion"),
        ({"scope": {}}, "at least one"),
        ({"scope": {"part": "x", "sections": "y"}}, "unknown scope keys"),
        ({"scope": {"part": "x", "value_max": "high"}}, "must be a number"),
        ({"scope": {"part": 3}}, "string or a list"),
        ({"approved": "14/10/2026"}, "ISO date"),
        ({"quote": " "}, "quoted"),
    ],
)
def test_malformed_exceptions_are_refused(change: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        M.parse_exceptions([dict(EXCEPTION, **change)])


# ---------------------------------------------------------------------------
# provenance


def test_provenance_accepts_bundles_inside_the_store(tmp_path: Path) -> None:
    store = tmp_path / "store"
    bundle = store / "wmb_panel" / "abc"
    bundle.mkdir(parents=True)
    found = M.bundle_paths(
        {"references": {"wmb_panel": {"bundle_path": str(bundle)}}}, "report"
    )
    found += M.bundle_paths({"build_hash": "abc", "path": str(bundle)}, "ref")
    check = M.check_provenance(found, store)
    assert check.ok and len(check.bundles) == 1
    assert len(check.bundles[0]["sources"]) == 2


def test_provenance_refuses_ssd1_without_touching_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    ssd1 = "/media/mathieubo/SSD1/MerXen/annotation_references/wmb_panel/abc"
    real = os.path.realpath

    def guarded(path: Any, *args: Any, **kwargs: Any) -> str:
        assert not str(path).startswith("/media/mathieubo/SSD1"), "SSD1 touched"
        return real(path, *args, **kwargs)

    monkeypatch.setattr(M.os.path, "realpath", guarded)
    check = M.check_provenance([(ssd1, "map_manifest")], store)
    assert not check.ok
    assert "outside" in check.bundles[0]["reason"]
    assert check.bundles[0]["real_path"] is None


def test_provenance_refuses_a_symlink_out_of_the_store_and_a_missing_bundle(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    (store / "wmb_panel").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere" / "bundle"
    elsewhere.mkdir(parents=True)
    link = store / "wmb_panel" / "linked"
    link.symlink_to(elsewhere)
    check = M.check_provenance(
        [(str(link), "a"), (str(store / "wmb_panel" / "gone"), "b"), ("rel/x", "c")],
        store,
    )
    reasons = {Path(r["path"]).name: r["reason"] for r in check.bundles}
    assert "resolves outside" in reasons["linked"]
    assert reasons["gone"] == "no such bundle directory"
    assert reasons["x"] == "not an absolute path"
    assert not M.check_provenance([], store).ok


# ---------------------------------------------------------------------------
# the flip rule


def test_flip_rule_verdicts() -> None:
    def row(criterion: str, verdict: str, scored: bool = True) -> Any:
        r = M.Row(criterion, "p", "ag7", "s", 1, "<=", 1, True, scored, "x")
        r.verdict = verdict
        return r

    passing = [row(c, "PASS") for c in M.FLIP_CRITERIA] + [
        row("MO9", "INFO fail", False)
    ]
    assert M.flip_rule(passing, True)["verdict"] == "PASS"
    assert M.flip_rule(passing + [row("MO4", "FAIL")], True)["verdict"] == "FAIL"
    assert (
        M.flip_rule(passing + [row("MO8", "NOT_AVAILABLE")], True)["verdict"]
        == "INCOMPLETE"
    )
    assert M.flip_rule(passing[1:], True)["verdict"] == "INCOMPLETE"
    assert M.flip_rule(passing, False)["verdict"] == "NOT SCORED (provenance)"


# ---------------------------------------------------------------------------
# end to end


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _results_root(
    tmp_path: Path,
    dataset: str,
    panel: Path,
    share: Path,
    *,
    bundle_path: str | None = None,
) -> Path:
    root = tmp_path / "results" / PAIR[dataset]
    pair = PAIR[dataset]
    for seg in M.SEGMENTATIONS:
        report = {
            "pair_id": pair,
            "segmentation": seg,
            "metrics": _records(),
            "provenance": {
                "references": {
                    "wmb_panel": {"bundle_path": bundle_path or str(panel)},
                    "wmb_region_share": {"bundle_path": str(share)},
                }
            },
        }
        _write(root / M.REPORT_PATH.format(seg=seg), report)
        manifest = {
            "pair_id": pair,
            "n_processors": 4,
            "samples": {
                SID[dataset]: {
                    "runs": {
                        "wmb_panel": {
                            "build_hash": "x",
                            "bundle_path": str(panel),
                            "engine_params": {"n_processors": 4},
                        }
                    },
                    "mouse_regions": {
                        "inferred_regions": MANUAL,
                        "rule_variant": "m6b",
                        "params": {
                            "rule_variant": "m6b",
                            "tile_um": 200.0,
                            "min_neurons_per_tile": 2,
                        },
                        "region_share": {"build_hash": SHARE_BUILD, "path": str(share)},
                    },
                }
            },
        }
        _write(root / M.MAP_DIR.format(seg=seg) / "map_manifest.json", manifest)
        summary = {
            "pair_id": pair,
            "samples": {
                SID[dataset]: {
                    "mouse_gate": {
                        "level": "full",
                        "signal_status": {"g1": "pass"},
                        "ap_estimate": {
                            "ap_mm": 8.1,
                            "window_sections": ["S31", "S32", "S33"],
                        },
                    }
                }
            },
        }
        resolve = root / M.RESOLVE_DIR.format(seg=seg)
        _write(resolve / f"{pair}_resolve_summary.json", summary)
        classes = ["01 IT-ET Glut"] * 10
        subclasses = ["007 L2/3 IT CTX Glut"] * 10
        labels = _labels(classes, subclasses).reset_index()
        (resolve / "merscope").mkdir(parents=True, exist_ok=True)
        labels.to_parquet(
            resolve / "merscope" / f"{SID[dataset]}_celltype_labels.parquet"
        )
    return root


def _store(tmp_path: Path) -> tuple[Path, Path, Path]:
    store = tmp_path / "store"
    panel = _bundle(store, "wmb_panel", PANEL_BUILD)
    share = store / "wmb_region_share" / SHARE_BUILD
    share.mkdir(parents=True)
    rows = []
    for section in M.MO4_FIXED_WINDOW:
        for name, n in (
            ("30 Astro-Epen", 1736),
            ("34 Immune", 152),
            ("01 IT-ET Glut", 8112),
        ):
            rows.append(
                {"level": "class", "section": section, "node_name": name, "n": n}
            )
    pd.DataFrame(rows).to_parquet(share / "section_composition.parquet")
    return store, panel, share


def _e2e_args(tmp_path: Path, store: Path, panel: Path, share: Path) -> list[str]:
    roots = {
        dataset: _results_root(tmp_path, dataset, panel, share)
        for dataset in M.DATASETS
    }
    tasks = _section_tasks("ag7") + _section_tasks(PAIR["vzg2"])
    trace = tmp_path / "trace.tsv"
    columns = [
        "name",
        "process",
        "tag",
        "status",
        "exit",
        "cpus",
        "realtime",
        "peak_rss",
        "workdir",
    ]
    pd.DataFrame([{c: t.get(c, "") for c in columns} for t in tasks]).to_csv(
        trace, sep="\t", index=False
    )
    pseudo = {}
    for dataset, column, value in (
        ("ag7", "pseudo_fine", "Glut"),
        ("vzg2", "pseudo_class", "Glut"),
    ):
        path = tmp_path / f"pseudo_{dataset}.csv"
        pd.DataFrame(
            {column: [value] * 9 + ["Astro"]}, index=[f"c{i}" for i in range(10)]
        ).to_csv(path)
        pseudo[dataset] = path
    negative = tmp_path / "negative"
    _write(
        negative / "resolve" / "x_resolve_summary.json",
        {"samples": {"x": {"mouse_gate": {"level": "failed"}}}},
    )
    _write(
        negative / "map" / "map_manifest.json",
        {"samples": {"x": {"runs": {"wmb_panel": {"bundle_path": str(panel)}}}}},
    )
    missed = [{"section": "C57BL6J-638850.35", "missed": "STR", "harm": 0.0014}]
    mo10 = _write(tmp_path / "mo10.json", _mo10(missed=missed))
    prereg = tmp_path / "prereg.md"
    prereg.write_text(
        "Selected candidate (M6b): `v1`\n### Amendment\nSelected candidate (M6b): "
        "`a+f:tile200,nn2`\n"
    )
    exceptions = _write(tmp_path / "exceptions.json", {"exceptions": [EXCEPTION]})
    return [
        "score",
        "--dataset",
        f"ag7={roots['ag7']}",
        "--dataset",
        f"vzg2={roots['vzg2']}",
        "--trace",
        str(trace),
        "--pseudo-labels",
        f"ag7={pseudo['ag7']}",
        "--pseudo-labels",
        f"vzg2={pseudo['vzg2']}",
        "--mo7-negative",
        str(negative),
        "--mo10",
        str(mo10),
        "--exceptions",
        str(exceptions),
        "--preregistration",
        str(prereg),
        "--date",
        "2026-10-14",
    ]


@pytest.fixture
def e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path, Path]:
    store, panel, share = _store(tmp_path)
    monkeypatch.setattr(M, "STORE_ROOT", store)

    def measured(map_dir: Path, full_dir: Path | None = None) -> dict[str, Any]:
        return {
            "subset": _mo2_values(),
            "full": _mo2_values(klass=0.002),
            "full_record": {"n_processors": 4},
            "map_processes": 4,
        }

    monkeypatch.setattr(M, "mo2_measure", measured)
    return _e2e_args(tmp_path, store, panel, share), store, panel


def test_score_end_to_end_with_the_registered_readings(
    e2e: tuple[list[str], Path, Path], tmp_path: Path
) -> None:
    args, _, _ = e2e
    out = tmp_path / "out"
    assert M.main([*args, "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["mode"] == "registered readings" and summary["label"] == ""
    assert all(r["is_registered"] for r in summary["readings"].values())
    assert summary["provenance"]["ok"]
    flip = summary["flip_rule"]
    # MO1 0.9; MO4 on the fixed window; MO10's one missed region excepted.
    assert flip["criteria"]["MO10"] == "FAIL (excepted)"
    assert flip["verdict"] == "PASS", flip
    assert summary["exceptions"][0]["applied_to"]
    assert summary["problems"] == []
    assert "registered readings" in (out / "summary.html").read_text()
    assert (out / "tables" / "rows.csv").is_file()
    mo1 = [r for r in summary["rows"] if r["criterion"] == "MO1"]
    assert [r["value"] for r in mo1] == [pytest.approx(0.9), pytest.approx(0.9)]


def test_score_exploratory_reading_is_stamped(
    e2e: tuple[list[str], Path, Path], tmp_path: Path
) -> None:
    args, _, _ = e2e
    with pytest.raises(SystemExit) as error:
        M.main([*args, "--mo4-window", "ap_estimate", "--out", str(tmp_path / "o1")])
    assert error.value.code == 2
    out = tmp_path / "o2"
    status = M.main(
        [
            *args,
            "--mo4-window",
            "ap_estimate",
            "--exploratory",
            "--label",
            "smoke test, not gate M",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["mode"] == "exploratory, not gate M"
    assert not summary["readings"]["mo4_window"]["is_registered"]
    assert (
        "exploratory, not gate M; smoke test, not gate M"
        in (out / "summary.html").read_text()
    )


def test_score_fails_the_run_on_a_bundle_outside_the_store(
    e2e: tuple[list[str], Path, Path], tmp_path: Path
) -> None:
    args, store, panel = e2e
    report = Path(
        args[args.index("--dataset") + 1].split("=", 1)[1]
    ) / M.REPORT_PATH.format(seg="reseg")
    payload = json.loads(report.read_text())
    payload["provenance"]["references"]["wmb_panel"]["bundle_path"] = (
        "/media/mathieubo/SSD1/MerXen/annotation_references/wmb_panel/5a032858"
    )
    report.write_text(json.dumps(payload))
    out = tmp_path / "out"
    assert M.main([*args, "--out", str(out)]) == 3
    summary = json.loads((out / "summary.json").read_text())
    assert not summary["provenance"]["ok"]
    assert summary["flip_rule"]["verdict"] == "NOT SCORED (provenance)"
    assert {r["verdict"] for r in summary["rows"]} == {"UNSCORED (provenance)"}
    assert any("SSD1" in problem for problem in summary["problems"])


def test_score_refuses_an_output_inside_a_results_root(
    e2e: tuple[list[str], Path, Path],
) -> None:
    args, _, _ = e2e
    root = args[args.index("--dataset") + 1].split("=", 1)[1]
    with pytest.raises(SystemExit):
        M.main([*args, "--out", str(Path(root) / "acceptance")])


def test_manual_regions_are_mouse_ap_axis_s() -> None:
    assert set(M.MANUAL_REGIONS["ag7"]) == set(MANUAL)
    assert set(M.MANUAL_REGIONS["vzg2"]) == set(MANUAL)
