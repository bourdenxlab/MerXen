"""Tests of the mouse real-data QC of RESOLVE (plan §8.8; M13b chunk C22).

Mouse RESOLVE runs ``real_qc.real_data_qc`` warn-only
(``real_qc.mouse_real_data_qc``): the mouse gate keeps G1-G5 (§7.6) and the
QC records them as checks, with G4's window and the AP estimate that chose
it; every lowering effect becomes a warning, whatever the family and the
species gate record (D20 (b) leaves gate M pending), so no QC outcome
reaches the mouse gate, the labels or trust.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from merxen.annotation import real_qc as qc
from merxen.annotation.config import (
    AnnotationConfig,
    AnnotationRealQcConfig,
    MouseGateConfig,
)
from merxen.annotation.mouse_gate import (
    MarkerReferee,
    MerfishWindow,
    MouseGateSignals,
    RegistrationSignal,
    evaluate_mouse_gate,
)

from .test_real_qc_checks import mouse_signals, new_panel_signals, stratum
from .test_real_qc_outcomes import mouse_gate

MakeTrust = Callable[..., Any]
MOUSE_GATE_CHECKS = {
    "dataset_gate",
    "registration_g1",
    "marker_consistency",
    "g3_implausibility",
    "g4_composition",
    "g5_spillover",
}
AP_ESTIMATE = {
    "section": "C57BL6J-638850.33",
    "ap_mm": 8.0,
    "ap_bin": "7.5-9.0",
    "in_scope": True,
    "window_sections": [
        "C57BL6J-638850.32",
        "C57BL6J-638850.33",
        "C57BL6J-638850.34",
    ],
}
WINDOW = MerfishWindow(
    sections=tuple(AP_ESTIMATE["window_sections"]),  # type: ignore[arg-type]
    shares={"30 Astro-Epen": 0.12, "34 Immune": 0.04},
    n_cells=150_000,
    ap_min_mm=7.8,
    ap_max_mm=8.2,
)


def g4_gate(**change: Any) -> Any:
    """A mouse gate whose G4 window came from the AP estimate."""
    values: dict[str, Any] = {
        "registration": RegistrationSignal(density_ratio=2.9, shift_um=0.0),
        "referee": MarkerReferee(
            consistency=0.9, n_pseudo_confident=500, n_cells=600, markers={}
        ),
        "t2_share": 0.004,
        "spillover_rate": 0.08,
        "astro_epen_points": -6.5,
        "immune_points": 0.4,
        "window": WINDOW,
        "ap_estimate": AP_ESTIMATE,
    }
    values.update(change)
    return evaluate_mouse_gate(MouseGateSignals(**values), MouseGateConfig())


# --------------------------------------------------------------------------
# The gate's own outcomes: G1-G5


def test_the_mouse_gate_signal_records_hold_g1_to_g5() -> None:
    records = qc.mouse_gate_signal_records(g4_gate())
    assert records is not None
    assert list(records) == ["g1", "g2", "g3", "g4", "g5"]
    assert {key: item["check"] for key, item in records.items()} == {
        "g1": "registration_g1",
        "g2": "marker_consistency",
        "g3": "g3_implausibility",
        "g4": "g4_composition",
        "g5": "g5_spillover",
    }
    assert records["g1"]["status"] == "pass" and records["g1"]["outcome"] == "pass"
    assert records["g1"]["values"] == {"g1_density_ratio": 2.9, "g1_shift_um": 0.0}
    assert records["g3"]["values"] == {"g3_implausible_share": 0.004}
    # G4 warns (Astro-Epen 6.5 points below the window) against the window of
    # the AP estimate, which the record carries.
    g4 = records["g4"]
    assert (g4["status"], g4["outcome"]) == ("warn", "warn")
    assert g4["values"] == {"g4_astro_epen_points": -6.5, "g4_immune_points": 0.4}
    assert g4["window_source"] == "ap_estimate"
    assert g4["ap_estimate"] == AP_ESTIMATE
    assert g4["window"]["sections"] == AP_ESTIMATE["window_sections"]
    assert len(g4["reasons"]) == 1 and g4["reasons"][0].startswith("g4_composition:")
    assert records["g5"]["values"] == {"g5_spillover_rate": 0.08}
    assert all("window" not in item for key, item in records.items() if key != "g4")


def test_the_g4_record_names_where_its_window_came_from() -> None:
    explicit = qc.mouse_gate_signal_records(g4_gate(ap_estimate=None))
    assert explicit is not None
    assert explicit["g4"]["window_source"] == "g4_window_sections"
    assert explicit["g4"]["ap_estimate"] is None
    none = qc.mouse_gate_signal_records(
        g4_gate(
            window=None,
            ap_estimate=None,
            astro_epen_points=None,
            immune_points=None,
            g4_reason="no AP estimate: too few confident neurons",
        )
    )
    assert none is not None
    g4 = none["g4"]
    assert (g4["status"], g4["outcome"]) == ("not_evaluated", "not_evaluable")
    assert g4["window"] is None and g4["window_source"] is None
    assert g4["reasons"] == [
        "g4_composition: not evaluated (no AP estimate: too few confident neurons)"
    ]


def test_a_human_gate_has_no_signal_records() -> None:
    from .test_real_qc_outcomes import human_gate

    assert qc.mouse_gate_signal_records(human_gate()) is None
    assert qc.mouse_gate_signal_records(None) is None


def test_the_mouse_gate_records_g1_to_g5_in_the_provenance() -> None:
    verdict = mouse_gate(
        registration=RegistrationSignal(density_ratio=1.2, shift_um=0.0)
    )
    assert qc.gate_outcomes(verdict) == {
        "dataset_gate": "fail",
        "registration_g1": "fail",
        "marker_consistency": "pass",
        "g3_implausibility": "pass",
        "g4_composition": "not_evaluable",
        "g5_spillover": "pass",
    }


# --------------------------------------------------------------------------
# mouse_real_data_qc: warn-only, the checks that do not apply


def _mouse_qc(
    make_trust: MakeTrust,
    *,
    state: str = "validated_real",
    record: dict[str, str] | None = None,
    **change: Any,
) -> qc.RealQcResult:
    real_qc = AnnotationRealQcConfig(
        seeded_families_warn_only_until_gate=record or {}  # type: ignore[arg-type]
    )
    return qc.mouse_real_data_qc(
        mouse_signals(**change),
        make_trust(state, species="mouse"),
        AnnotationConfig(species="mouse", real_qc=real_qc),
    )


def test_a_seeded_mouse_section_records_every_check(make_trust: MakeTrust) -> None:
    """MS1 / MS2: version 6, <= 1,000 genes, MERSCOPE only (seeded)."""
    result = _mouse_qc(make_trust, gate=g4_gate())
    tokens = result.provenance().outcomes
    assert set(tokens) == MOUSE_GATE_CHECKS | {
        "paired_concordance",
        "flag_rates",
        "gene_complexity",
        "prefilter_spotcheck",
        "coverage_vs_simulation",
        "nonneuronal_depth_trend",
        "factor_remeasure",
    }
    assert tokens["dataset_gate"] == "warn"
    assert tokens["g4_composition"] == "warn"
    assert tokens["flag_rates"] == "pass"
    # The checks that do not apply, each with its reason.
    not_applicable = {
        item.check: item.reason
        for item in result.outcomes
        if item.state == "not_applicable"
    }
    assert set(not_applicable) == {
        "paired_concordance",
        "prefilter_spotcheck",
        "coverage_vs_simulation",
        "nonneuronal_depth_trend",
        "factor_remeasure",
    }
    assert all(not_applicable.values())
    assert not_applicable["paired_concordance"].startswith("unpaired")
    # Version 6 stores no simulated genes: applies, not evaluable (D19 (a)).
    assert tokens["gene_complexity"] == "not_evaluable"
    assert result.species == "mouse" and result.seeded and result.warn_only
    assert result.effects_withheld == qc.MOUSE_RESOLVE_WARN_ONLY_NOTE
    assert not result.effects.lowers and result.provenance().downgrades == []


@pytest.mark.parametrize(
    ("state", "record"),
    [
        ("validated_real", {"mouse": "pending"}),
        ("validated_real", {"mouse": "merged"}),
        ("provisional", {"mouse": "pending"}),
        ("validated_simulation", {"mouse": "merged"}),
    ],
)
def test_mouse_real_data_qc_lowers_nothing_whatever_the_family(
    make_trust: MakeTrust, state: str, record: dict[str, str]
) -> None:
    """Every lowering effect is demoted, also where real_data_qc applies it."""
    change: dict[str, Any] = {
        "prefilter_applied": True,
        "prefilter": qc.PrefilterSpotcheckSignal({"class": 0.9}, n_cells=10_000),
        "emitted_levels": ["class"],
    }
    applied = qc.real_data_qc(
        mouse_signals(**change),
        make_trust(state, species="mouse"),
        AnnotationConfig(
            species="mouse",
            real_qc=AnnotationRealQcConfig(
                seeded_families_warn_only_until_gate=record  # type: ignore[arg-type]
            ),
        ),
    )
    result = _mouse_qc(make_trust, state=state, record=record, **change)
    assert result.warn_only and not result.effects.lowers
    assert result.effects.withheld_levels == ()
    assert result.provenance().downgrades == []
    spot = result.outcomes_of("prefilter_spotcheck")
    assert [item.outcome for item in spot] == ["warn"]
    # A seeded family's own demotion (D20 (b)) comes first and keeps its
    # note; the mouse wiring demotes what is left.
    seeded_pending = state == "validated_real" and record["mouse"] == "pending"
    note = (
        qc.WARN_ONLY_NOTE.format(species="mouse")
        if seeded_pending
        else qc.MOUSE_RESOLVE_WARN_ONLY_NOTE
    )
    assert spot[0].message.startswith(note)
    # The effect it would have had is recorded, as for a seeded family.
    assert spot[0].details["warn_only_effect"] == "withhold_level"
    assert result.effects.warning_reasons
    # Outside the demotion the outcomes are real_data_qc's.
    for before, after in zip(applied.outcomes, result.outcomes, strict=True):
        assert before.check == after.check
        if not before.lowers:
            assert after == before, before.check


def test_a_paired_mouse_pair_records_paired_concordance_not_evaluable(
    make_trust: MakeTrust,
) -> None:
    """Mouse RESOLVE computes no pair JSD in v1: the check applies, no input."""
    result = _mouse_qc(make_trust, paired=True)
    (paired,) = result.outcomes_of("paired_concordance")
    assert paired.state == "not_evaluable"
    assert paired.reason == qc.MOUSE_PAIRED_CONCORDANCE_REASON
    assert paired.effect == "withhold_pair_stats"
    unpaired = _mouse_qc(make_trust, paired=False).outcomes_of("paired_concordance")
    assert [item.state for item in unpaired] == ["not_applicable"]


def test_the_mouse_summary_carries_the_gate_signals_and_the_note(
    make_trust: MakeTrust,
) -> None:
    summary = _mouse_qc(make_trust, gate=g4_gate()).summary()
    assert summary["species"] == "mouse" and summary["warn_only"] is True
    assert summary["effects_withheld"] == qc.MOUSE_RESOLVE_WARN_ONLY_NOTE
    assert summary["gate_signals"]["g4"]["ap_estimate"] == AP_ESTIMATE
    assert summary["per_check"]["g4_composition"] == "warn"
    assert summary["promotes"] is False


def test_a_human_summary_gains_no_mouse_key(make_trust: MakeTrust) -> None:
    """The human record keeps its keys (human RESOLVE output byte-identical)."""
    from .test_real_qc_outcomes import human_gate

    summary = qc.real_data_qc(
        new_panel_signals(gate=human_gate()),
        make_trust("provisional"),
        AnnotationConfig(species="human"),
    ).summary()
    assert list(summary) == [
        "version",
        "species",
        "seeded",
        "warn_only",
        "per_check",
        "downgrades",
        "effects",
        "outcomes",
        "flag_rates",
        "promotes",
    ]


def test_replace_check_keeps_the_mouse_demotion(make_trust: MakeTrust) -> None:
    result = _mouse_qc(make_trust)
    fired = qc.QcOutcome(
        "paired_concordance", True, "withhold_pair_stats", message="JSD 0.3"
    )
    replaced = result.replace_check("paired_concordance", [fired])
    (outcome,) = replaced.outcomes_of("paired_concordance")
    assert outcome.effect == "warning" and outcome.outcome == "warn"
    assert not replaced.effects.lowers
    assert replaced.effects_withheld == qc.MOUSE_RESOLVE_WARN_ONLY_NOTE


def test_mouse_real_data_qc_refuses_a_human_config(make_trust: MakeTrust) -> None:
    with pytest.raises(ValueError, match="mouse"):
        qc.mouse_real_data_qc(
            mouse_signals(),
            make_trust("validated_real"),
            AnnotationConfig(species="human"),
        )


def test_a_mouse_flag_rate_warning_is_a_warning(make_trust: MakeTrust) -> None:
    """Flag rates are evaluated for mouse (the spill-over flag included)."""
    strata = [
        stratum("microglial_spillover", "Microglia", 0.4),
        stratum("contaminated", "Neurons", 0.3),
        stratum("diffuse_profile", "Neurons", 0.01),
    ]
    result = _mouse_qc(make_trust, flag_strata=strata)
    (flags,) = result.outcomes_of("flag_rates")
    assert flags.fired and flags.outcome == "warn"
    assert result.flag_rates is not None and result.flag_rates.n_uninformative == 2
