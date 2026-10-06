"""Tests for the real-data QC checks and orchestrator M13 adds (chunk C12)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import real_qc as qc
from merxen.annotation.config import AnnotationConfig, AnnotationRealQcConfig
from merxen.annotation.flags import FlagStratum
from merxen.annotation.mouse_gate import RegistrationSignal
from merxen.annotation.provenance import RealQcProvenance

from .test_real_qc_outcomes import human_gate, mouse_gate

MakeTrust = Callable[..., Any]
CLASSES = (
    "Neurons",
    "Astrocytes",
    "Oligodendrocytes",
    "Oligodendrocyte precursors",
    "Microglia",
    "Vascular cells",
    "Fibroblasts",
)
# Every check real_data_qc records for a human dataset; with the dataset
# gate these are the rows of plan §8.8 (pre-registration §23.5 P4).
HUMAN_CHECKS = {
    "marker_consistency",
    "registration_g1",
    "paired_concordance",
    "flag_rates",
    "gene_complexity",
    "prefilter_spotcheck",
    "coverage_vs_simulation",
    "nonneuronal_depth_trend",
    "factor_remeasure",
}
# The pair statistic of P5011 proseg_hybrid at M8 (soft broad JSD).
P5011_JSD = [
    {"kind": "soft", "region": "whole_section", "jsd": 0.226814, "ci_low": 0.21227},
    {"kind": "soft", "region": "shared_mask", "jsd": 0.234905, "ci_high": 0.25},
    {"kind": "confident", "region": "shared_mask", "jsd": 0.334206},
]


def stratum(
    flag: str,
    cls: str,
    rate: float | None,
    *,
    platform: str = "MERSCOPE",
    informative: bool | None = None,
) -> FlagStratum:
    """Return a flag stratum; ``informative`` defaults to the 15% marking."""
    h16 = rate is not None and rate <= 0.15
    return FlagStratum(
        flag=flag,
        cls=cls,
        platform=platform,
        n_basis=100,
        n_flagged_basis=0 if rate is None else int(rate * 100),
        rate=rate,
        n_cells=120,
        n_flagged=0,
        rate_all=rate,
        informative=h16 if informative is None else informative,
        informative_h16=h16,
    )


def clean_strata(**rates: Mapping[str, float | None]) -> list[FlagStratum]:
    """Contamination, diffuse and OOD strata over CLASSES (low rates)."""
    out = []
    for flag in ("contaminated", "diffuse_profile", "ood"):
        for cls in CLASSES:
            rate = rates.get(flag, {}).get(cls, 0.01)
            out.append(stratum(flag, cls, rate))
    return out


# --------------------------------------------------------------------------
# Marker referee (D18)


@pytest.mark.parametrize(
    ("value", "token", "effect"),
    [
        (0.80, "pass", "gate_cap"),
        (0.75, "pass", "gate_cap"),
        (0.749, "warn", "warning"),
        (0.70, "warn", "warning"),
        (0.699, "fail", "gate_cap"),
    ],
)
def test_marker_consistency_warns_below_075_and_caps_the_gate_below_070(
    value: float, token: str, effect: str
) -> None:
    signal = qc.MarkerConsistencySignal(value, n_marker_groups=4, n_pseudo_labelled=900)
    result = qc.marker_consistency_outcome(signal, warn=0.75, broad_only=0.70)
    assert result.outcome == token and result.effect == effect
    assert result.details["consistency"] == value
    if token == "fail":
        assert result.gate_cap == "broad_only"
        assert "capped at broad_only (trust and margins unchanged)" in result.message
        # Trust is never touched: the referee caps the gate (D18 (a)).
        assert qc.apply_qc_outcomes("provisional", [result]) == "provisional"


def test_marker_consistency_is_not_evaluable_without_a_statistic() -> None:
    missing = qc.marker_consistency_outcome(None, warn=0.75, broad_only=0.70)
    assert missing.outcome == "not_evaluable" and missing.gate_cap == "broad_only"
    few = qc.marker_consistency_outcome(
        qc.MarkerConsistencySignal(None, n_marker_groups=1, reason="1 group of >= 3"),
        warn=0.75,
        broad_only=0.70,
    )
    assert few.outcome == "not_evaluable" and few.reason == "1 group of >= 3"
    nan = qc.marker_consistency_outcome(
        qc.MarkerConsistencySignal(float("nan")), warn=0.75, broad_only=0.70
    )
    assert nan.outcome == "not_evaluable"
    with pytest.raises(ValueError, match="must not exceed"):
        qc.marker_consistency_outcome(None, warn=0.70, broad_only=0.75)


# --------------------------------------------------------------------------
# Registration G1 (D23)


def test_registration_g1_fails_on_the_ratio_or_the_shift() -> None:
    limits = {"density_ratio_fail": 1.5, "shift_fail_um": 5.0}
    passing = qc.registration_g1_outcome(RegistrationSignal(2.9, 0.4), **limits)
    assert passing.outcome == "pass" and passing.effect == "warning"
    low = qc.registration_g1_outcome(RegistrationSignal(1.2, 0.4), **limits)
    assert low.outcome == "warn" and "density ratio 1.200 < 1.5" in low.message
    assert "warn-only in M13" in low.message
    shifted = qc.registration_g1_outcome(RegistrationSignal(2.9, 7.5), **limits)
    assert shifted.fired and "shift 7.5 um > 5.0 um" in shifted.message
    # §8.8's effect once D23 moves to (a): the dataset gate fails.
    hard = qc.registration_g1_outcome(
        RegistrationSignal(1.2, 0.4), effect="gate_failed", **limits
    )
    assert hard.outcome == "fail" and hard.gate_cap == "failed"
    assert qc.apply_qc_to_gate(human_gate(), [hard]).level == "failed"
    assert qc.registration_g1_outcome(None, **limits).outcome == "not_evaluable"
    skipped = qc.registration_g1_outcome(
        RegistrationSignal(None, None, status="skipped"), **limits
    )
    assert skipped.outcome == "not_evaluable" and "skipped" in str(skipped.reason)
    with pytest.raises(ValueError, match="unknown registration G1 effect"):
        qc.registration_g1_outcome(None, effect="exclude", **limits)


# --------------------------------------------------------------------------
# Paired concordance (D21)


def test_paired_concordance_is_not_applicable_to_an_unpaired_section() -> None:
    result = qc.paired_concordance(P5011_JSD, paired=False, warn_above=0.20)
    assert result.outcome == "not_applicable"
    assert result.effect == "withhold_pair_stats" and not result.lowers


def test_paired_concordance_scores_the_shared_mask_point_estimate() -> None:
    result = qc.paired_concordance(P5011_JSD, paired=True, warn_above=0.20)
    assert result.outcome == "fail" and result.effect == "withhold_pair_stats"
    assert result.details["scored_region"] == "shared_mask"
    assert result.details["jsd"] == pytest.approx(0.234905)
    assert result.details["whole_section_jsd"] == pytest.approx(0.226814)
    assert result.details["statistic"] == "point" and not result.details["fallback"]
    assert "supercluster-level cross-platform statistics" in result.message
    assert qc.qc_effects([result]).withhold_pair_stats
    # The other M8 pairs (<= .154) pass; 0.20 itself is not above 0.20.
    for value in (0.154, 0.20):
        rows = [{"kind": "soft", "region": "shared_mask", "jsd": value}]
        assert qc.paired_concordance(rows, paired=True, warn_above=0.20).outcome == (
            "pass"
        )


def test_paired_concordance_falls_back_to_the_whole_section() -> None:
    rows = [{"kind": "soft", "region": "whole_section", "jsd": 0.21}]
    result = qc.paired_concordance(rows, paired=True, warn_above=0.20)
    assert result.fired and result.details["fallback"]
    assert result.details["scored_region"] == "whole_section"
    assert result.details["shared_mask_jsd"] is None
    empty = qc.paired_concordance([], paired=True, warn_above=0.20)
    assert empty.outcome == "not_evaluable"
    other_kind = qc.paired_concordance(
        [{"kind": "argmax", "region": "shared_mask", "jsd": 0.5}],
        paired=True,
        warn_above=0.20,
    )
    assert other_kind.outcome == "not_evaluable"


# --------------------------------------------------------------------------
# Flag rates (D22: §8.8's literal reading)


def test_flag_rates_count_class_platform_strata_under_the_h16_marking() -> None:
    # Diffuse above 15% in 3 classes and contamination in a fourth: 4 of 7
    # (class x platform) strata are uninformative, more than half.
    strata = clean_strata(
        diffuse_profile={
            "Oligodendrocytes": 0.35,
            "Microglia": 0.44,
            "Fibroblasts": 0.2,
        },
        contaminated={"Vascular cells": 0.16},
    )
    summary = qc.flag_rate_summary(strata, warn_frac=0.5)
    assert summary.n_strata == 7 and summary.n_uninformative == 4
    assert summary.outcome.outcome == "warn" and summary.outcome.effect == "warning"
    assert "4 of 7 (class x platform) strata" in summary.outcome.message
    row = summary.table.set_index("class").loc["Vascular cells"]
    assert row["uninformative_flags"] == ["contaminated"]
    reported = summary.reported
    # Counted per flag, no flag alone has more than half uninformative.
    assert reported["per_flag_h16"]["diffuse_profile"]["n_uninformative"] == 3
    assert reported["per_flag_h16"]["contaminated"]["n_uninformative"] == 1
    assert "ood" not in reported["per_flag_h16"]
    assert reported["pooled_h16"] == {
        "n_strata": 14,
        "n_uninformative": 4,
        "share": 4 / 14,
    }
    # D22 option (a), reported only: each flag's own switch, every stratum.
    own = reported["own_switch_all_flags"]
    assert own["n_strata"] == 21 and own["n_uninformative"] == 4
    assert own["flags"] == ["contaminated", "diffuse_profile", "ood"]
    assert own["would_warn"] is False
    record = summary.summary()
    assert record["reading"] == "h16_class_platform" and record["warn"]
    assert json.loads(json.dumps(record)) == record


def test_flag_rates_do_not_warn_at_exactly_half_and_read_json_strata() -> None:
    strata = [
        stratum("contaminated", "Neurons", 0.01),
        stratum("contaminated", "Astrocytes", 0.20),
        stratum("diffuse_profile", "Neurons", 0.02),
        stratum("diffuse_profile", "Astrocytes", 0.02),
    ]
    summary = qc.flag_rate_summary(strata, warn_frac=0.5)
    assert (summary.n_strata, summary.n_uninformative) == (2, 1)
    assert summary.outcome.outcome == "pass"
    as_json = qc.flag_rate_summary([item.to_json() for item in strata], warn_frac=0.5)
    assert as_json.table.equals(summary.table)
    assert as_json.reported == summary.reported


def test_flag_rates_count_a_stratum_without_basis_cells_as_uninformative() -> None:
    # A contamination stratum without a null keeps a 0.0 rate (informative
    # under H16); one without confident basis cells has no rate at all.
    strata = [
        stratum("contaminated", "Fibroblasts", 0.0, informative=False),
        stratum("contaminated", "Microglia", None),
    ]
    summary = qc.flag_rate_summary(strata, warn_frac=0.4)
    uninformative = summary.table.set_index("class")["uninformative"]
    assert not uninformative["Fibroblasts"] and uninformative["Microglia"]
    assert summary.outcome.fired
    assert summary.reported["own_switch_all_flags"]["n_uninformative"] == 2


def test_flag_rates_without_strata_are_not_evaluable() -> None:
    only_ood = [stratum("ood", "Neurons", 0.5)]
    for strata in ([], only_ood):
        summary = qc.flag_rate_summary(strata, warn_frac=0.5)
        assert summary.outcome.outcome == "not_evaluable"
        assert summary.n_strata == 0


# --------------------------------------------------------------------------
# Prefilter spot check and factor re-measure


def test_the_prefilter_spot_check_withholds_a_level_below_095() -> None:
    signal = qc.PrefilterSpotcheckSignal(
        {"class": 0.994, "subclass": 0.92}, n_cells=10_000
    )
    outcomes = qc.prefilter_spotcheck(
        signal,
        applies=True,
        emitted_levels=["class", "subclass", "supertype"],
        min_agreement=0.95,
    )
    tokens = {item.level: item.outcome for item in outcomes}
    assert tokens == {"class": "pass", "subclass": "fail", "supertype": "not_evaluable"}
    effects = qc.qc_effects(outcomes)
    assert effects.withheld_levels == ("subclass",)
    assert effects.downgrades == ("not_resolvable_subclass:prefilter_spotcheck",)
    boundary = qc.prefilter_spotcheck(
        qc.PrefilterSpotcheckSignal({"class": 0.95}),
        applies=True,
        emitted_levels=["class"],
        min_agreement=0.95,
    )
    assert boundary[0].outcome == "pass"


def test_the_prefilter_spot_check_records_where_it_does_not_apply() -> None:
    (none,) = qc.prefilter_spotcheck(
        None, applies=False, emitted_levels=["broad"], min_agreement=0.95
    )
    assert none.outcome == "not_applicable" and none.effect == "withhold_level"
    (missing,) = qc.prefilter_spotcheck(
        None, applies=True, emitted_levels=["broad"], min_agreement=0.95
    )
    assert missing.outcome == "not_evaluable"
    (no_level,) = qc.prefilter_spotcheck(
        qc.PrefilterSpotcheckSignal({}),
        applies=True,
        emitted_levels=[],
        min_agreement=0.95,
    )
    assert no_level.outcome == "not_evaluable"


def remeasure(**change: Any) -> qc.FactorRemeasure:
    values: dict[str, Any] = {
        "applies": True,
        "reason": None,
        "asset_id": "asset_1",
        "n_informative": 40,
        "pearson_r": 0.93,
        "spearman_r": 0.9,
        "outcome": None,
    }
    values.update(change)
    return qc.FactorRemeasure(**values)


def test_the_factor_remeasure_applies_only_with_a_measured_table() -> None:
    assert qc.factor_remeasure_outcome(None, min_r=0.9).outcome == "not_applicable"
    for reason, token in (
        ("no_stored_table", "not_applicable"),
        ("not_first_dataset_of_family", "not_applicable"),
        ("too_few_informative_genes", "not_evaluable"),
    ):
        result = qc.factor_remeasure_outcome(
            remeasure(applies=False, reason=reason), min_r=0.9
        )
        assert result.outcome == token and result.reason == reason
    assert qc.factor_remeasure_outcome(remeasure(), min_r=0.9).outcome == "pass"
    # The configured threshold decides, not the one the re-measure ran with.
    tighter = qc.factor_remeasure_outcome(remeasure(), min_r=0.95)
    assert tighter.outcome == "warn" and "r 0.930 < 0.95" in tighter.message


# --------------------------------------------------------------------------
# The orchestrator


def coverage_signal(real: float, predicted: float, n: int = 500) -> qc.CoverageSignal:
    return qc.CoverageSignal(
        real=pd.DataFrame(
            [{"level": "broad", "class": "Astro", "n_cells": n, "real_coverage": real}]
        ),
        predicted=pd.DataFrame(
            [
                {
                    "level": "broad",
                    "class": "Astro",
                    "predicted_coverage": predicted,
                    "resolvable_share": 1.0,
                }
            ]
        ),
    )


def trend_signal(n_per_band: int = 300) -> qc.NonneuronalTrendSignal:
    totals = np.r_[np.full(n_per_band, 700.0), np.full(n_per_band, 1500.0)]
    return qc.NonneuronalTrendSignal(
        totals=totals,
        called_class=np.full(len(totals), "Astro", dtype=object),
        confident={"broad": np.ones(len(totals), dtype=bool)},
        nonneuronal_classes=["Astro"],
    )


def complexity_signal(native: float, simulated: float) -> qc.GeneComplexitySignal:
    return qc.GeneComplexitySignal(
        native_n_genes=np.full(60, native),
        native_totals=np.full(60, 120.0),
        simulated_n_genes=np.full(60, simulated),
        simulated_depth=np.full(60, 120.0),
        grid=[10, 15, 30, 60, 120, 250],
    )


def new_panel_signals(**change: Any) -> qc.RealQcSignals:
    """An unpaired MERSCOPE-only section of a version-7, <= 1,000-gene family."""
    values: dict[str, Any] = {
        "resolvability_version": 7,
        "paired": False,
        "marker_consistency": qc.MarkerConsistencySignal(0.82, 4, 900),
        "flag_strata": clean_strata(),
        "prefilter_applied": False,
        "coverage": coverage_signal(0.85, 0.88),
        "nonneuronal_trend": trend_signal(),
        "registration": RegistrationSignal(2.6, 0.5),
        "gate": human_gate(),
    }
    values.update(change)
    return qc.RealQcSignals(**values)


def test_the_unpaired_merscope_only_section_records_every_check(
    make_trust: MakeTrust,
) -> None:
    config = AnnotationConfig(species="human")
    result = qc.real_data_qc(new_panel_signals(), make_trust("provisional"), config)
    record = result.provenance()
    assert isinstance(record, RealQcProvenance)
    assert set(record.outcomes) == HUMAN_CHECKS | {"dataset_gate"}
    assert record.outcomes == {
        "marker_consistency": "pass",
        "registration_g1": "pass",
        "paired_concordance": "not_applicable",
        "flag_rates": "pass",
        "gene_complexity": "not_evaluable",
        "prefilter_spotcheck": "not_applicable",
        "coverage_vs_simulation": "pass",
        "nonneuronal_depth_trend": "pass",
        "factor_remeasure": "not_applicable",
        "dataset_gate": "pass",
    }
    assert not result.seeded and not result.warn_only
    assert record.warn_only is False and record.downgrades == []
    assert not result.effects.lowers
    gate = human_gate()
    assert result.apply_to_gate(gate) is gate
    summary = result.summary()
    assert summary["promotes"] is False and summary["per_check"] == record.outcomes
    assert json.loads(json.dumps(summary)) == summary
    assert set(result.tables) == {"coverage_vs_simulation", "nonneuronal_depth_trend"}


def test_a_low_marker_referee_caps_a_new_family_at_broad_only(
    make_trust: MakeTrust,
) -> None:
    config = AnnotationConfig(species="human")
    signals = new_panel_signals(
        marker_consistency=qc.MarkerConsistencySignal(0.66, 4, 900),
        coverage=coverage_signal(0.60, 0.88),
    )
    result = qc.real_data_qc(signals, make_trust("provisional"), config)
    record = result.provenance()
    assert record.outcomes["marker_consistency"] == "fail"
    assert record.outcomes["coverage_vs_simulation"] == "warn"
    assert record.downgrades == ["broad_only:marker_consistency"]
    gate = result.apply_to_gate(human_gate())
    assert gate.level == "broad_only" and gate.warning
    assert any(
        "real_qc_coverage_vs_simulation" in item for item in gate.warning_reasons
    )
    # Never a trust change, never a promotion.
    assert result.apply_to_trust("provisional") == "provisional"


def test_a_seeded_family_only_warns_while_its_species_gate_is_pending(
    make_trust: MakeTrust,
) -> None:
    signals = new_panel_signals(
        resolvability_version=6,
        paired=True,
        pair_jsd=P5011_JSD,
        marker_consistency=qc.MarkerConsistencySignal(0.66, 4, 900),
        coverage=None,
        nonneuronal_trend=None,
    )
    trust = make_trust("validated_real")
    pending = qc.real_data_qc(signals, trust, AnnotationConfig(species="human"))
    assert pending.seeded and pending.warn_only
    assert not pending.effects.lowers
    assert pending.provenance().outcomes["paired_concordance"] == "warn"
    assert pending.provenance().outcomes["marker_consistency"] == "warn"
    assert pending.provenance().warn_only is True
    paired = next(o for o in pending.outcomes if o.check == "paired_concordance")
    assert paired.details["warn_only_effect"] == "withhold_pair_stats"
    assert paired.message.startswith("warn-only until the human gate merges")
    assert pending.apply_to_gate(human_gate()).level == "full"
    merged = qc.real_data_qc(
        signals,
        trust,
        AnnotationConfig(
            species="human",
            real_qc=AnnotationRealQcConfig(
                seeded_families_warn_only_until_gate={"human": "merged"}
            ),
        ),
    )
    assert not merged.warn_only
    assert merged.provenance().downgrades == [
        "broad_only:marker_consistency",
        "withhold_pair_stats:paired_concordance",
    ]
    assert merged.apply_to_gate(human_gate()).level == "broad_only"
    # Version 6: the version-7 checks do not apply.
    for check in ("coverage_vs_simulation", "nonneuronal_depth_trend"):
        assert pending.provenance().outcomes[check] == "not_applicable"


def test_the_orchestrator_reads_the_real_qc_config(make_trust: MakeTrust) -> None:
    signals = new_panel_signals(
        paired=True,
        pair_jsd=[{"kind": "soft", "region": "shared_mask", "jsd": 0.22}],
        marker_consistency=qc.MarkerConsistencySignal(0.78, 4, 900),
        gene_complexity=complexity_signal(native=150.0, simulated=100.0),
        coverage=coverage_signal(0.80, 0.88),
        prefilter_applied=True,
        prefilter=qc.PrefilterSpotcheckSignal({"broad": 0.96}),
        emitted_levels=["broad"],
        factor=remeasure(pearson_r=0.92),
    )
    trust = make_trust("provisional")
    default = qc.real_data_qc(
        signals, trust, AnnotationConfig(species="human")
    ).provenance()
    expected = {
        "marker_consistency": "pass",
        "paired_concordance": "fail",
        "gene_complexity": "warn",
        "coverage_vs_simulation": "pass",
        "prefilter_spotcheck": "pass",
        "factor_remeasure": "pass",
    }
    assert {check: default.outcomes[check] for check in expected} == expected
    stricter = AnnotationRealQcConfig(
        marker_consistency_warn=0.80,
        paired_broad_jsd_warn=0.25,
        genes_per_count_gap_warn=0.60,
        coverage_warn_margin=0.05,
        prefilter_spotcheck_min_agreement=0.97,
        factor_remeasure_min_r=0.95,
        registration_g1_effect="gate_failed",
    )
    changed = qc.real_data_qc(
        signals,
        trust,
        AnnotationConfig(species="human", real_qc=stricter),
    ).provenance()
    assert changed.outcomes["marker_consistency"] == "warn"
    assert changed.outcomes["paired_concordance"] == "pass"
    assert changed.outcomes["gene_complexity"] == "pass"
    assert changed.outcomes["coverage_vs_simulation"] == "warn"
    assert changed.outcomes["prefilter_spotcheck"] == "fail"
    assert changed.outcomes["factor_remeasure"] == "warn"


def test_registration_g1_follows_the_configured_effect(make_trust: MakeTrust) -> None:
    signals = new_panel_signals(registration=RegistrationSignal(1.1, 0.0))
    trust = make_trust("provisional")
    warn = qc.real_data_qc(signals, trust, AnnotationConfig(species="human"))
    assert warn.provenance().outcomes["registration_g1"] == "warn"
    assert warn.apply_to_gate(human_gate()).level == "full"
    hard = qc.real_data_qc(
        signals,
        trust,
        AnnotationConfig(
            species="human",
            real_qc=AnnotationRealQcConfig(registration_g1_effect="gate_failed"),
        ),
    )
    assert hard.provenance().outcomes["registration_g1"] == "fail"
    assert hard.apply_to_gate(human_gate()).level == "failed"


def test_the_version_7_checks_are_not_evaluable_without_inputs(
    make_trust: MakeTrust,
) -> None:
    signals = new_panel_signals(
        coverage=coverage_signal(0.85, 0.88, n=50),
        nonneuronal_trend=trend_signal(n_per_band=20),
        gene_complexity=qc.GeneComplexitySignal([], [], [], [], [10, 30]),
        flag_strata=None,
        marker_consistency=None,
        registration=None,
    )
    outcomes = qc.real_data_qc(
        signals, make_trust("provisional"), AnnotationConfig(species="human")
    ).provenance()
    for check in (
        "coverage_vs_simulation",
        "nonneuronal_depth_trend",
        "gene_complexity",
        "flag_rates",
        "marker_consistency",
        "registration_g1",
    ):
        assert outcomes.outcomes[check] == "not_evaluable", check
    none = qc.real_data_qc(
        new_panel_signals(coverage=None, nonneuronal_trend=None),
        make_trust("provisional"),
        AnnotationConfig(species="human"),
    ).provenance()
    assert none.outcomes["coverage_vs_simulation"] == "not_evaluable"
    assert none.outcomes["nonneuronal_depth_trend"] == "not_evaluable"


def test_a_mouse_dataset_takes_g1_and_g2_from_its_gate(make_trust: MakeTrust) -> None:
    signals = qc.RealQcSignals(
        resolvability_version=6,
        flag_strata=[stratum("microglial_spillover", "Microglia", 0.05)],
        gate=mouse_gate(),
    )
    result = qc.real_data_qc(
        signals,
        make_trust("validated_real", species="mouse"),
        AnnotationConfig(species="mouse"),
    )
    checks = {item.check for item in result.outcomes}
    assert "marker_consistency" not in checks and "registration_g1" not in checks
    record = result.provenance()
    assert record.outcomes["marker_consistency"] == "pass"
    assert record.outcomes["registration_g1"] == "pass"
    assert record.outcomes["flag_rates"] == "pass"
    assert result.seeded and result.warn_only


# --------------------------------------------------------------------------
# Config (D20, D23)


def test_the_species_gate_record_defaults_to_pending() -> None:
    config = AnnotationRealQcConfig()
    assert config.seeded_families_warn_only_until_gate == {
        "human": "pending",
        "mouse": "pending",
    }
    assert config.seeded_warn_only("human") and config.seeded_warn_only("mouse")
    assert config.registration_g1_effect == "warning"
    partial = AnnotationRealQcConfig(
        seeded_families_warn_only_until_gate={"mouse": "merged"}
    )
    assert partial.seeded_families_warn_only_until_gate == {
        "human": "pending",
        "mouse": "merged",
    }
    assert partial.seeded_warn_only("human") and not partial.seeded_warn_only("mouse")


@pytest.mark.parametrize(("value", "state"), [(True, "pending"), (False, "merged")])
def test_the_rev3_bool_still_loads(value: bool, state: str) -> None:
    config = AnnotationRealQcConfig(seeded_families_warn_only_until_gate=value)  # type: ignore[arg-type]
    assert config.seeded_families_warn_only_until_gate == {
        "human": state,
        "mouse": state,
    }
    round_trip = AnnotationRealQcConfig.model_validate_json(config.model_dump_json())
    assert round_trip == config


def test_the_real_qc_config_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        AnnotationRealQcConfig(
            seeded_families_warn_only_until_gate={"human": "done"}  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError):
        AnnotationRealQcConfig(
            seeded_families_warn_only_until_gate={"rat": "merged"}  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError):
        AnnotationRealQcConfig(registration_g1_effect="exclude")  # type: ignore[arg-type]
