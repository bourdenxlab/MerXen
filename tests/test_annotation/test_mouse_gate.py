"""Tests for the mouse dataset gate G1-G5 (plan §7.6, §8.6; M6)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation.config import (
    MOUSE_CCF_REGIONS,
    AnnotationFlagsConfig,
    MouseGateConfig,
)
from merxen.annotation.mouse_flags import MouseFlagProfiles, SpecificGenes
from merxen.annotation.mouse_gate import (
    SIGNAL_KEYS,
    MarkerReferee,
    MerfishWindow,
    MouseGateSignals,
    RegistrationSignal,
    class_group_markers,
    composition_offsets,
    evaluate_mouse_gate,
    find_registration_check,
    marker_pseudo_labels,
    marker_referee,
    merfish_window,
    t2_share,
    t2_subclasses,
)
from merxen.annotation.mouse_regions import (
    OVERRIDE_DIFFERS_REASON,
    RegionShares,
    region_step_warnings,
)

CONFIG = MouseGateConfig()


def _referee(consistency: float | None) -> MarkerReferee:
    return MarkerReferee(
        consistency=consistency,
        n_pseudo_confident=1000,
        n_cells=2000,
        markers={},
        reason=None if consistency is not None else "too few cells",
    )


def _window() -> MerfishWindow:
    return MerfishWindow(
        sections=("C57BL6J-638850.32",),
        shares={"30 Astro-Epen": 0.174, "34 Immune": 0.015},
        n_cells=1000,
    )


def _good(**changes: Any) -> MouseGateSignals:
    values: dict[str, Any] = {
        "registration": RegistrationSignal(density_ratio=2.9, shift_um=0.0),
        "referee": _referee(0.90),
        "t2_share": 0.004,
        "astro_epen_points": 1.0,
        "immune_points": 0.2,
        "window": _window(),
        "spillover_rate": 0.08,
    }
    values.update(changes)
    return MouseGateSignals(**values)


# (signals change, level, warning, per-signal status); every row changes one
# signal of an otherwise passing sample, the §7.6 truth table.
TRUTH_TABLE: list[tuple[str, dict[str, Any], str, bool, dict[str, str]]] = [
    ("all pass", {}, "full", False, {}),
    (
        "G1 ratio below 1.5 fails",
        {"registration": RegistrationSignal(density_ratio=1.19, shift_um=0.0)},
        "failed",
        False,
        {"g1": "fail"},
    ),
    (
        "G1 ratio exactly 1.5 warns only",
        {"registration": RegistrationSignal(density_ratio=1.5, shift_um=0.0)},
        "full",
        True,
        {"g1": "warn"},
    ),
    (
        "G1 ratio below 2.0 warns",
        {"registration": RegistrationSignal(density_ratio=1.9, shift_um=0.0)},
        "full",
        True,
        {"g1": "warn"},
    ),
    (
        "G1 shift above 5 um fails",
        {"registration": RegistrationSignal(density_ratio=2.9, shift_um=114.1)},
        "failed",
        False,
        {"g1": "fail"},
    ),
    (
        "G1 shift just above 5 um fails",
        {"registration": RegistrationSignal(density_ratio=2.9, shift_um=6.0)},
        "failed",
        False,
        {"g1": "fail"},
    ),
    (
        "G1 shift of 5 um passes",
        {"registration": RegistrationSignal(density_ratio=2.9, shift_um=5.0)},
        "full",
        False,
        {"g1": "pass"},
    ),
    (
        "G1 without a shift passes on the ratio",
        {"registration": RegistrationSignal(density_ratio=2.9, shift_um=None)},
        "full",
        False,
        {"g1": "pass"},
    ),
    (
        "G1 missing warns",
        {"registration": None},
        "full",
        True,
        {"g1": "not_evaluated"},
    ),
    (
        "G1 skipped warns",
        {
            "registration": RegistrationSignal(
                density_ratio=None, shift_um=None, status="skipped"
            )
        },
        "full",
        True,
        {"g1": "not_evaluated"},
    ),
    (
        "G2 below 0.70 fails",
        {"referee": _referee(0.69)},
        "failed",
        False,
        {"g2": "fail"},
    ),
    ("G2 at 0.70 warns", {"referee": _referee(0.70)}, "full", True, {"g2": "warn"}),
    ("G2 at 0.80 passes", {"referee": _referee(0.80)}, "full", False, {"g2": "pass"}),
    (
        "G2 not evaluable warns",
        {"referee": _referee(None)},
        "full",
        True,
        {"g2": "not_evaluated"},
    ),
    ("G2 not computed warns", {"referee": None}, "full", True, {"g2": "not_evaluated"}),
    ("G3 above 3% warns", {"t2_share": 0.031}, "full", True, {"g3": "warn"}),
    ("G3 at 3% passes", {"t2_share": 0.03}, "full", False, {"g3": "pass"}),
    (
        "G3 not evaluated is a note",
        {"t2_share": None, "t2_reason": "region step disabled"},
        "full",
        False,
        {"g3": "not_evaluated"},
    ),
    (
        "G4 Astro-Epen outside 5 points warns",
        {"astro_epen_points": 16.9},
        "full",
        True,
        {"g4": "warn"},
    ),
    (
        "G4 Astro-Epen 6 points below the window warns",
        {"astro_epen_points": -6.0},
        "full",
        True,
        {"g4": "warn"},
    ),
    (
        "G4 Immune outside 1 point warns",
        {"immune_points": -1.2},
        "full",
        True,
        {"g4": "warn"},
    ),
    (
        "G4 on the band edge passes",
        {"astro_epen_points": -5.0, "immune_points": 1.0},
        "full",
        False,
        {"g4": "pass"},
    ),
    (
        "G4 without a window is a note",
        {"astro_epen_points": None, "immune_points": None, "window": None},
        "full",
        False,
        {"g4": "not_evaluated"},
    ),
    ("G5 above 15% warns", {"spillover_rate": 0.151}, "full", True, {"g5": "warn"}),
    ("G5 at 15% passes", {"spillover_rate": 0.15}, "full", False, {"g5": "pass"}),
    (
        "G5 null is a note",
        {"spillover_rate": None, "g5_reason": "insufficient_panel_genes"},
        "full",
        False,
        {"g5": "not_evaluated"},
    ),
    (
        "region step skipped on few tiles warns (plan §7.2)",
        {
            "t2_share": None,
            "t2_reason": "region step skipped_few_tiles",
            "region_step": ("skipped_few_tiles", ["47 assigned tiles < 200"]),
        },
        "full",
        True,
        {"g3": "not_evaluated"},
    ),
    (
        "region step skipped without coordinates warns",
        {
            "t2_share": None,
            "t2_reason": "region step skipped_no_coordinates",
            "region_step": ("skipped_no_coordinates", []),
        },
        "full",
        True,
        {"g3": "not_evaluated"},
    ),
    (
        "an override that differs from the inference warns",
        {
            "region_step": (
                "no_nodes_dropped",
                [f"{OVERRIDE_DIFFERS_REASON} (Isocortex;MB)"],
            )
        },
        "full",
        True,
        {},
    ),
    (
        "pruning disabled on request is a note only",
        {
            "t2_share": None,
            "t2_reason": "region step disabled",
            "region_step": ("disabled", ["mouse_section_regions none"]),
        },
        "full",
        False,
        {"g3": "not_evaluated"},
    ),
    (
        "invalid VZG2 proseg_hybrid: G1 fails with a G4 warning",
        {
            "registration": RegistrationSignal(density_ratio=1.33, shift_um=114.1),
            "astro_epen_points": 16.9,
        },
        "failed",
        True,
        {"g1": "fail", "g4": "warn"},
    ),
]


@pytest.mark.parametrize(
    ("change", "level", "warning", "status"),
    [row[1:] for row in TRUTH_TABLE],
    ids=[row[0] for row in TRUTH_TABLE],
)
def test_mouse_gate_truth_table(
    change: dict[str, Any], level: str, warning: bool, status: dict[str, str]
) -> None:
    change = dict(change)
    step = change.pop("region_step", None)
    extra = [] if step is None else region_step_warnings(*step)
    verdict = evaluate_mouse_gate(_good(**change), CONFIG, extra_warnings=extra)
    assert verdict.level == level
    assert set(extra) <= set(verdict.warning_reasons)
    assert verdict.warning is warning
    expected = {key: "pass" for key in ("g1", "g2", "g3", "g4", "g5")}
    expected.update(status)
    assert verdict.signal_status == expected
    assert verdict.attempts_leaf is (level == "full")
    assert bool(verdict.level_reasons) is (level != "full")


def test_mouse_gate_trust_caps_the_level_and_warns(
    make_trust: Callable[..., Any],
) -> None:
    refused = evaluate_mouse_gate(
        _good(), CONFIG, trust=make_trust("refused", species="mouse")
    )
    assert refused.level == "failed"
    broad_only = evaluate_mouse_gate(
        _good(), CONFIG, trust=make_trust("broad_only", species="mouse")
    )
    assert broad_only.level == "broad_only" and not broad_only.attempts_leaf
    provisional = evaluate_mouse_gate(
        _good(), CONFIG, trust=make_trust("provisional", species="mouse")
    )
    assert provisional.level == "full" and provisional.warning
    assert any(reason.startswith("panel_provisional") for reason in provisional.reasons)
    # A failing signal is never lifted by trust.
    failed = evaluate_mouse_gate(
        _good(registration=RegistrationSignal(density_ratio=1.0, shift_um=0.0)),
        CONFIG,
        trust=make_trust("validated_real", species="mouse"),
    )
    assert failed.level == "failed"


def test_mouse_gate_json_and_provenance() -> None:
    verdict = evaluate_mouse_gate(
        _good(registration=None, t2_share=None, t2_reason="region step disabled"),
        CONFIG,
    )
    record = verdict.to_json()
    assert set(record["signals"]) == set(SIGNAL_KEYS)
    assert record["signals"]["g1_density_ratio"] is None
    assert record["window"]["astro_epen_share"] == pytest.approx(0.174)
    assert any("g3_implausibility" in note for note in record["notes"])
    provenance = verdict.provenance(section_regions=["HPF"], region_source="auto")
    assert provenance.level == "full" and provenance.warning
    assert provenance.section_regions == ["HPF"]
    assert provenance.signals["g2_marker_consistency"] == pytest.approx(0.9)


def test_registration_signal_reads_the_qc_json_and_summary(tmp_path: Path) -> None:
    payload = {
        "status": "warn",
        "reasons": ["offset"],
        "density_ratio": 1.33,
        "shift_um": 114.14,
        "n_windows": 8,
    }
    path = tmp_path / "s_registration_qc.json"
    path.write_text(json.dumps(payload))
    signal = RegistrationSignal.from_file(path)
    assert (signal.density_ratio, signal.shift_um, signal.status) == (
        1.33,
        114.14,
        "warn",
    )
    assert signal.reasons == ("offset",) and signal.source == str(path)

    summary = tmp_path / "s_qc_summary.csv"
    summary.write_text(
        "dataset,registration_status,registration_density_ratio,registration_shift_um\n"
        "S,pass,2.13,\n"
    )
    signal = RegistrationSignal.from_file(summary)
    assert (signal.density_ratio, signal.shift_um, signal.status) == (
        2.13,
        None,
        "pass",
    )

    bad = tmp_path / "other.json"
    bad.write_text(json.dumps({"status": "pass"}))
    with pytest.raises(ValueError, match="no registration check"):
        RegistrationSignal.from_file(bad)
    no_columns = tmp_path / "x_qc_summary.csv"
    no_columns.write_text("dataset,n_cells\nS,10\n")
    with pytest.raises(ValueError, match="no registration columns"):
        RegistrationSignal.from_file(no_columns)


def test_find_registration_check_discovers_the_qc_stage_outputs(
    tmp_path: Path,
) -> None:
    """G1 per sample from QC outputs: the JSON wins over the summary CSV."""
    qc = tmp_path / "qc"
    (qc / "a").mkdir(parents=True)
    (qc / "b").mkdir()
    (qc / "a" / "ag7_merscope_registration_qc.json").write_text(
        json.dumps({"status": "pass", "density_ratio": 2.2, "shift_um": 0.0})
    )
    (qc / "a" / "ag7_merscope_qc_summary.csv").write_text(
        "dataset,registration_status,registration_density_ratio\nx,pass,9.9\n"
    )
    (qc / "b" / "vzg2_merscope_qc_summary.csv").write_text(
        "dataset,registration_status,registration_density_ratio\nx,warn,1.4\n"
    )
    (qc / "b" / "old_merscope_qc_summary.csv").write_text("dataset,n_cells\nx,1\n")

    found = find_registration_check([qc], "AG7_MERSCOPE")
    assert found is not None and found.density_ratio == 2.2
    assert found.source is not None and found.source.endswith("_registration_qc.json")
    summary = find_registration_check([qc / "a", qc / "b"], "VZG2_MERSCOPE")
    assert summary is not None and (summary.density_ratio, summary.status) == (
        1.4,
        "warn",
    )
    # A QC summary from before M0a holds no check; an absent sample has none.
    assert find_registration_check([qc], "OLD_MERSCOPE") is None
    assert find_registration_check([qc], "NONE_MERSCOPE") is None
    (qc / "b" / "ag7_merscope_registration_qc.json").write_text("{}")
    with pytest.raises(ValueError, match="several"):
        find_registration_check([qc], "AG7_MERSCOPE")


# --------------------------------------------------------------------------
# G2 referee

GENES = [f"G{index}" for index in range(9)]
GROUP_OF = {
    "01 IT-ET Glut": "Neurons",
    "30 Astro-Epen": "Astrocytes/Ependymal",
    "34 Immune": "Microglia",
}


def _group_profiles() -> MouseFlagProfiles:
    classes = ("01 IT-ET Glut", "30 Astro-Epen", "34 Immune")
    matrix = np.full((3, 9), 1e-5)
    for row, genes in enumerate(((0, 1, 2), (3, 4, 5), (6, 7))):
        matrix[row, list(genes)] = 0.3
    matrix[:, 8] = 0.1  # shared by every class
    matrix /= matrix.sum(axis=1, keepdims=True)
    return MouseFlagProfiles(
        gene_ids=tuple(GENES),
        symbols=tuple(GENES),
        class_names=classes,
        class_profiles=matrix,
        subclass_names=classes,
        subclass_class=classes,
        subclass_profiles=matrix,
    )


def test_class_group_markers_leave_out_groups_with_too_few_genes() -> None:
    markers, left_out = class_group_markers(
        _group_profiles(), GROUP_OF, AnnotationFlagsConfig(), min_genes=3
    )
    assert {group: genes.symbols for group, genes in markers.items()} == {
        "Neurons": ("G0", "G1", "G2"),
        "Astrocytes/Ependymal": ("G3", "G4", "G5"),
    }
    assert left_out == ["Microglia"]  # two genes


def test_marker_pseudo_labels_use_the_p1212_rule() -> None:
    markers, _ = class_group_markers(
        _group_profiles(), GROUP_OF, AnnotationFlagsConfig(), min_genes=2
    )
    counts = np.array(
        [
            [4, 4, 4, 0, 0, 0, 0, 0, 9],  # neuron
            [0, 0, 0, 4, 4, 4, 0, 0, 9],  # astrocyte
            [2, 2, 2, 2, 2, 2, 0, 0, 9],  # mixed: 50/50, not pseudo-confident
            [1, 0, 0, 0, 0, 0, 0, 0, 9],  # too little marker signal
            [0, 0, 0, 0, 0, 0, 5, 5, 9],  # microglia
        ]
    )
    labels, confident = marker_pseudo_labels(sparse.csr_matrix(counts), markers)
    assert confident.tolist() == [True, True, False, False, True]
    assert labels[[0, 1, 4]].tolist() == [
        "Neurons",
        "Astrocytes/Ependymal",
        "Microglia",
    ]


def test_class_group_markers_use_the_mean_of_the_group_classes() -> None:
    """A gene specific to one class of a two-class group fails the group mean."""
    classes = ("01 IT-ET Glut", "19 MB Glut", "30 Astro-Epen")
    matrix = np.full((3, 5), 1e-5)
    matrix[0, 0] = 0.3  # G0: only in 01 -> group mean 0.15 = 15x astro
    matrix[:2, 1] = 0.3  # G1: in both neuron classes -> 30x
    matrix[:2, 2] = 0.3  # G2: likewise
    matrix[:2, 3] = 0.3  # G3: likewise
    matrix[2, 0] = 0.01
    matrix[2, [1, 2, 3]] = 0.01
    matrix[2, 4] = 0.5  # G4: astrocytic
    profiles = MouseFlagProfiles(
        gene_ids=tuple(GENES[:5]),
        symbols=tuple(GENES[:5]),
        class_names=classes,
        class_profiles=matrix,
        subclass_names=classes,
        subclass_class=classes,
        subclass_profiles=matrix,
    )
    group_of = {
        "01 IT-ET Glut": "Neurons",
        "19 MB Glut": "Neurons",
        "30 Astro-Epen": "Astrocytes/Ependymal",
    }
    markers, left_out = class_group_markers(
        profiles, group_of, AnnotationFlagsConfig(), min_genes=1
    )
    assert set(markers["Neurons"].symbols) == {"G1", "G2", "G3"}  # not G0
    assert markers["Astrocytes/Ependymal"].symbols == ("G4",)
    assert left_out == []


def test_marker_pseudo_labels_normalise_by_the_mean_positive_count() -> None:
    """P1212: units = counts / the gene's mean positive count in the dataset."""
    markers = {
        "Neurons": SpecificGenes("Neurons", np.array([0]), ("A",), ("A",), (30.0,)),
        "Microglia": SpecificGenes("Microglia", np.array([1]), ("B",), ("B",), (30.0,)),
    }
    counts = np.zeros((23, 2))
    counts[:10, 0] = 20  # gene A: ~20 counts per positive cell
    counts[10:20, 1] = 1  # gene B: ~1 count per positive cell
    counts[20] = [10, 2]  # 0.52 units of A, 1.83 of B -> Microglia
    counts[21] = [20, 0]  # 1.05 units: below 1.5, not pseudo-confident
    counts[22] = [0, 2]  # 1.83 units of B alone
    labels, confident = marker_pseudo_labels(sparse.csr_matrix(counts), markers)
    assert labels[20] == "Microglia" and confident[20]
    assert not confident[21]
    assert labels[22] == "Microglia" and confident[22]
    assert not confident[:20].any()  # 20 / 19.1 and 1 / 1.09 units


def test_marker_referee_scores_class_calls_against_pseudo_labels() -> None:
    rng = np.random.default_rng(0)
    profiles = _group_profiles()
    counts = np.vstack(
        [
            rng.multinomial(200, profiles.class_profiles[row], size=100)
            for row in range(3)
        ]
    )
    calls: list[str | None] = (
        ["01 IT-ET Glut"] * 100 + ["30 Astro-Epen"] * 100 + ["34 Immune"] * 100
    )
    # 30 astrocytes called as neurons: consistency 270 / 300.
    calls[100:130] = ["01 IT-ET Glut"] * 30
    # One pseudo-labelled neuron without a class call is not scored.
    counts = np.vstack([counts, counts[:1]])
    calls.append(None)
    config = MouseGateConfig(g2_min_group_markers=2, g2_min_pseudo_confident=50)
    referee = marker_referee(
        sparse.csr_matrix(counts),
        profiles,
        calls,
        GROUP_OF,
        AnnotationFlagsConfig(),
        config,
    )
    assert referee.n_pseudo_confident == 300
    assert referee.consistency == pytest.approx(0.9)
    assert referee.recall["Astrocytes/Ependymal"] == pytest.approx(0.7)
    few = marker_referee(
        sparse.csr_matrix(counts),
        profiles,
        calls,
        GROUP_OF,
        AnnotationFlagsConfig(),
        MouseGateConfig(g2_min_group_markers=2, g2_min_pseudo_confident=1000),
    )
    assert few.consistency is None and "pseudo-confident" in str(few.reason)
    none = marker_referee(
        None, profiles, calls, GROUP_OF, AnnotationFlagsConfig(), config
    )
    assert none.consistency is None and none.reason is not None


# --------------------------------------------------------------------------
# G3 and G4


def _t2_shares() -> RegionShares:
    subclasses = ["S_ctx", "S_my", "S_mixed", "S_bergmann", "S_small"]
    share = pd.DataFrame(0.0, index=subclasses, columns=list(MOUSE_CCF_REGIONS))
    share.loc["S_ctx", "Isocortex"] = 1.0
    share.loc["S_my", "MY"] = 1.0
    share.loc["S_mixed", ["Isocortex", "MY"]] = [0.25, 0.75]
    share.loc["S_bergmann", "CB"] = 1.0
    share.loc["S_small", "MY"] = 1.0
    return RegionShares(
        class_share=share.copy(),
        class_n_grey=pd.Series(100, index=subclasses),
        subclass_share=share,
        subclass_n_grey=pd.Series([100, 100, 100, 100, 19], index=subclasses),
        subclass_home=pd.Series(
            ["Isocortex", "MY", "MY", "CB", "MY"], index=subclasses
        ),
    )


T2_CLASSES = {
    "S_ctx": "01 IT-ET Glut",
    "S_my": "24 MY Glut",
    "S_mixed": "24 MY Glut",
    "S_bergmann": "30 Astro-Epen",
    "S_small": "24 MY Glut",
}
NEVER_DROP = ["30 Astro-Epen", "31 OPC-Oligo", "33 Vascular", "34 Immune"]


def test_t2_share_counts_subclasses_outside_the_present_regions() -> None:
    shares = _t2_shares()
    kwargs: dict[str, Any] = {
        "subclass_class": T2_CLASSES,
        "never_drop_classes": NEVER_DROP,
        "min_merfish_cells": 20,
    }
    calls = ["S_ctx"] * 6 + ["S_my"] * 3 + ["S_mixed"] + [None] * 2
    # S_my is T2 (0 in Isocortex); S_mixed holds exactly 25%, which is not T2.
    assert t2_share(calls, ["Isocortex"], shares, **kwargs) == pytest.approx(3 / 12)
    assert t2_share(calls, ["Isocortex", "MY"], shares, **kwargs) == 0.0
    assert t2_share([], ["Isocortex"], shares, **kwargs) == 0.0


def test_t2_leaves_out_never_drop_and_small_subclasses() -> None:
    """E7's definition: pruning never removes these, so they are not T2."""
    shares = _t2_shares()
    t2 = t2_subclasses(
        ["Isocortex"],
        shares,
        subclass_class=T2_CLASSES,
        never_drop_classes=NEVER_DROP,
        min_merfish_cells=20,
    )
    assert t2 == {"S_my"}  # not S_bergmann (Astro-Epen) or S_small (19 cells)
    calls = ["S_ctx"] * 4 + ["S_bergmann"] * 2 + ["S_small"] * 2 + ["S_my"] * 2
    kwargs: dict[str, Any] = {
        "subclass_class": T2_CLASSES,
        "never_drop_classes": NEVER_DROP,
    }
    assert t2_share(calls, ["Isocortex"], shares, **kwargs) == pytest.approx(0.2)
    # Without the filters every subclass below 25% would count (0.6).
    everything = t2_share(
        calls,
        ["Isocortex"],
        shares,
        subclass_class=T2_CLASSES,
        never_drop_classes=[],
        min_merfish_cells=0,
    )
    assert everything == pytest.approx(0.6)


def test_merfish_window_pools_sections_and_gives_point_offsets() -> None:
    rows = []
    for section, (astro, immune, other) in {
        "S31": (170, 15, 815),
        "S32": (180, 16, 804),
        "S99": (500, 0, 500),
    }.items():
        for name, n in (
            ("30 Astro-Epen", astro),
            ("34 Immune", immune),
            ("01 IT-ET Glut", other),
        ):
            rows.append(
                {
                    "section": section,
                    "ap_ccf_mm": 8.0 if section != "S99" else 1.0,
                    "n_cells": 1000,
                    "level": "class",
                    "node_name": name,
                    "n": n,
                    "freq": n / 1000,
                }
            )
    table = pd.DataFrame(rows)
    window = merfish_window(table, ["S31", "S32"])
    assert window.n_cells == 2000
    assert window.shares["30 Astro-Epen"] == pytest.approx(0.175)
    assert window.shares["34 Immune"] == pytest.approx(0.0155)
    astro, immune = composition_offsets(
        {"30 Astro-Epen": 0.343, "34 Immune": 0.017, "01 IT-ET Glut": 0.64}, window
    )
    assert astro == pytest.approx(16.8) and immune == pytest.approx(0.15)
    with pytest.raises(ValueError, match="not in the region-share bundle"):
        merfish_window(table, ["S31", "S40"])
    assert composition_offsets({"01 IT-ET Glut": 1.0}, window) == (None, None)


def test_mouse_gate_config_checks_its_new_fields() -> None:
    config = MouseGateConfig()
    assert config.g4_window_sections == []
    assert (config.g2_min_group_markers, config.g2_min_pseudo_confident) == (3, 200)
    assert config.g3_t2_max_present_share == 0.25
    with pytest.raises(ValueError):
        MouseGateConfig(g2_min_marker_share=1.5)
    with pytest.raises(ValueError):
        MouseGateConfig(g2_min_group_markers=0)


def test_class_corr_floor_is_off_by_default_and_bounded() -> None:
    from merxen.annotation.config import AnnotationThresholds

    assert AnnotationThresholds().wmb_class_min_corr is None
    assert AnnotationThresholds(wmb_class_min_corr=0.4).wmb_class_min_corr == 0.4
    with pytest.raises(ValueError):
        AnnotationThresholds(wmb_class_min_corr=1.5)
