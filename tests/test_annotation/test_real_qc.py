"""Tests for the downgrade-only real-data QC (``merxen.annotation.real_qc``)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import real_qc as qc
from merxen.annotation.schema import PANEL_TRUST_STATES

GRID = [10, 20, 50, 100, 250, 500, 1000, 2000]


def class_depth_rows(
    cls: str,
    coverage: dict[int, float],
    *,
    level: str = "class",
    regime: str = "provisional",
    neuronal: bool | None = True,
    high_depth: tuple[int, ...] = (),
    not_emitted: tuple[int, ...] = (),
) -> list[dict[str, object]]:
    """Return class-depth rows of one (level, class) over GRID."""
    return [
        {
            "regime": regime,
            "level": level,
            "class": cls,
            "depth": depth,
            "status": "not_resolvable"
            if depth in not_emitted or depth not in coverage
            else "emitted",
            "coverage": coverage.get(depth, np.nan),
            "neuronal": neuronal,
            "nonneuronal_high_depth": depth in high_depth,
        }
        for depth in GRID
    ]


def test_class_bin_shares_use_the_floor_bin_and_keep_low_cells_in_the_denominator() -> (
    None
):
    shares = qc.class_bin_shares(
        [5, 15, 60, 60, 1500, np.nan], ["A", "A", "A", "B", "A", None], GRID
    )
    a = shares[shares["class"] == "A"].set_index("depth")["share"]
    assert a[10] == pytest.approx(0.25)
    assert a[50] == pytest.approx(0.25)
    assert a[1000] == pytest.approx(0.25)
    # The cell at 5 counts is below the grid: in the denominator, in no bin.
    assert a.sum() == pytest.approx(0.75)
    assert set(shares["n_cells"][shares["class"] == "A"]) == {4}
    assert shares[shares["class"] == "B"].set_index("depth")["share"][50] == 1.0


def test_predicted_class_coverage_sums_emitted_bins_only() -> None:
    table = pd.DataFrame(
        class_depth_rows("A", {100: 0.5, 250: 0.8, 500: 0.9}, not_emitted=(500,))
    )
    shares = qc.class_bin_shares([100, 100, 300, 600], ["A"] * 4, GRID)
    predicted = qc.predicted_class_coverage(table, shares)
    row = predicted.iloc[0]
    assert row["predicted_coverage"] == pytest.approx(0.5 * 0.5 + 0.25 * 0.8)
    assert row["resolvable_share"] == pytest.approx(0.75)
    assert qc.predicted_class_coverage(table, shares, regime="validated").empty


def test_coverage_vs_simulation_warns_below_the_margin_and_judges_200_cells() -> None:
    real = pd.DataFrame(
        {
            "level": ["class", "class", "class", "subclass"],
            "class": ["A", "B", "C", "A"],
            "n_cells": [500, 500, 199, 500],
            "real_coverage": [0.79, 0.81, 0.10, 0.70],
        }
    )
    predicted = pd.DataFrame(
        {
            "level": ["class", "class", "class", "subclass"],
            "class": ["A", "B", "C", "A"],
            "predicted_coverage": [0.90, 0.90, 0.90, 0.80],
            "resolvable_share": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = qc.coverage_vs_simulation(real, predicted)
    assert result.flagged == [("class", "A")]
    # 0.70 vs 0.80 - 0.10 is not below: equality does not warn.
    assert ("subclass", "A") not in result.flagged
    # Class C has too few cells to be judged.
    judged = result.table.set_index(["level", "class"])["judged"]
    assert not judged[("class", "C")]
    (outcome,) = result.outcomes
    assert outcome.effect == "warning" and outcome.trust_cap is None
    assert "v1-type large-mask" in outcome.message
    assert "No offset is applied" in outcome.message
    summary = result.summary()
    assert summary["trust_effect"] == "none" and summary["n_flagged"] == 1
    assert summary["predictor"] == "class_depth"
    assert summary["profile_mode_reported"] is False
    # Worded per class (D4 of 2026-09-29), not as a glial warning.
    assert outcome.message.startswith("A at class: ")
    assert "glia" not in outcome.message.lower()
    assert "class-depth prediction 0.900" in outcome.message
    assert "profile-mode" not in outcome.message
    assert outcome.details["predictor"] == "class_depth"
    assert np.isnan(result.table["profile_coverage"]).all()


def test_the_profile_mode_prediction_is_reported_and_never_decides() -> None:
    # D4 of 2026-09-29: the class-depth predictor decides the warning; the
    # profile-mode prediction is reported beside it.
    real = pd.DataFrame(
        {
            "level": ["class", "class", "subclass"],
            "class": ["HY GABA", "Astro", "HY GABA"],
            "n_cells": [900, 900, 900],
            "real_coverage": [0.60, 0.85, 0.55],
        }
    )
    predicted = pd.DataFrame(
        {
            "level": ["class", "class", "subclass"],
            "class": ["HY GABA", "Astro", "HY GABA"],
            "predicted_coverage": [0.76, 0.90, 0.64],
            "resolvable_share": [1.0, 1.0, 1.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "member": ["member_mean"] * 3 + ["R1_contam_HO@0"],
            "class": ["HY GABA", "Astro", "ALL", "Astro"],
            "cov_class_prov": [0.65, 0.99, 0.9, 0.1],
            "cov_subclass_prov": [0.90, 0.95, 0.8, 0.1],
        }
    )
    profile = qc.profile_coverage_table(
        predictions, {"class": "cov_class_prov", "subclass": "cov_subclass_prov"}
    )
    assert sorted(zip(profile["level"], profile["class"], strict=True)) == [
        ("class", "Astro"),
        ("class", "HY GABA"),
        ("subclass", "Astro"),
        ("subclass", "HY GABA"),
    ]
    result = qc.coverage_vs_simulation(real, predicted, profile_predicted=profile)
    # Class HY GABA: class-depth .76 - .10 > .60 warns although profile mode
    # (.65) would not; Astro: profile mode .99 would warn, class-depth .90 not;
    # subclass HY GABA: .64 - .10 = .54 < .55, no warning, profile .90 would.
    assert result.flagged == [("class", "HY GABA")]
    table = result.table.set_index(["level", "class"])
    assert table.loc[("class", "Astro"), "profile_coverage"] == pytest.approx(0.99)
    assert table.loc[("class", "HY GABA"), "profile_difference"] == pytest.approx(
        0.60 - 0.65
    )
    assert table.loc[("class", "HY GABA"), "difference"] == pytest.approx(-0.16)
    (outcome,) = result.outcomes
    assert outcome.message.startswith("HY GABA at class: ")
    assert "profile-mode prediction 0.650 is reported beside it" in outcome.message
    assert outcome.details["profile_coverage"] == pytest.approx(0.65)
    assert result.summary()["profile_mode_reported"] is True
    # A profile table without the class gives no profile value.
    partial = profile[profile["class"] != "Astro"]
    table = qc.coverage_vs_simulation(
        real, predicted, profile_predicted=partial
    ).table.set_index(["level", "class"])
    assert np.isnan(table.loc[("class", "Astro"), "profile_coverage"])
    assert qc.profile_coverage_table(pd.DataFrame(), {"class": "x"}).empty
    single = qc.profile_coverage_table(
        predictions, {"class": "cov_class_prov"}, member="R1_contam_HO@0"
    )
    assert single["profile_coverage"].tolist() == [0.1]


def test_coverage_vs_simulation_only_ever_warns() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 8))
        real = pd.DataFrame(
            {
                "level": ["class"] * n,
                "class": [f"c{i}" for i in range(n)],
                "n_cells": rng.integers(0, 1000, n),
                "real_coverage": rng.random(n),
            }
        )
        predicted = real[["level", "class"]].assign(
            predicted_coverage=rng.random(n), resolvable_share=rng.random(n)
        )
        result = qc.coverage_vs_simulation(real, predicted)
        for state in PANEL_TRUST_STATES:
            assert qc.apply_qc_outcomes(state, result.outcomes) == state
        assert all(item.effect == "warning" for item in result.outcomes)
        # The table itself is never modified towards the prediction.
        merged = result.table.set_index("class")["real_coverage"]
        assert np.allclose(
            merged.to_numpy(),
            real.set_index("class").loc[merged.index, "real_coverage"],
        )


def test_apply_qc_outcomes_never_raises_a_trust_state() -> None:
    effects = ["warning", "report_only"]
    outcomes = [
        qc.QcOutcome("a", fired, effect)
        for fired, effect in itertools.product([True, False], effects)
    ]
    outcomes += [
        qc.QcOutcome("d", fired, "downgrade", trust_cap=cap)
        for fired in (True, False)
        for cap in PANEL_TRUST_STATES
    ]
    for state in PANEL_TRUST_STATES:
        for size in range(0, 4):
            for chosen in itertools.permutations(outcomes, size):
                after = qc.apply_qc_outcomes(state, chosen)
                assert qc.trust_rank(after) <= qc.trust_rank(state)
                if not any(
                    item.effect == "downgrade" and item.fired for item in chosen
                ):
                    assert after == state


def test_qc_outcome_rejects_a_trust_cap_on_a_warning() -> None:
    with pytest.raises(ValueError, match="no trust cap"):
        qc.QcOutcome("x", True, "warning", trust_cap="broad_only")
    with pytest.raises(ValueError, match="needs a trust cap"):
        qc.QcOutcome("x", True, "downgrade")


def test_nonneuronal_high_depth_flags_marked_emitted_bins_at_1000_counts() -> None:
    table = pd.DataFrame(
        class_depth_rows(
            "Astro",
            {500: 0.9, 1000: 0.9, 2000: 0.8},
            neuronal=False,
            high_depth=(1000, 2000),
        )
        + class_depth_rows(
            "Oligo",
            {1000: 0.9},
            neuronal=False,
            high_depth=(1000,),
            not_emitted=(1000,),
        )
        + class_depth_rows("IT", {1000: 0.9}, neuronal=True, high_depth=(1000,))
        + class_depth_rows("Unknown", {1000: 0.9}, neuronal=None, high_depth=(1000,))
    )
    flags = qc.nonneuronal_high_depth_flags(
        [999, 1000, 2500, 1500, 1500, 1200, np.nan],
        ["Astro", "Astro", "Astro", "Oligo", "IT", "Unknown", "Astro"],
        table,
    )
    assert flags.tolist() == [False, True, True, False, False, True, False]


def test_nonneuronal_high_depth_flags_read_a_nullable_lineage_column() -> None:
    """``class_depth_table`` stores ``neuronal`` as nullable booleans."""
    table = pd.DataFrame(
        class_depth_rows("Astro", {1000: 0.9}, neuronal=False, high_depth=(1000,))
        + class_depth_rows("IT", {1000: 0.9}, neuronal=True, high_depth=(1000,))
        + class_depth_rows("Unknown", {1000: 0.9}, neuronal=None, high_depth=(1000,))
    )
    table["neuronal"] = table["neuronal"].astype("boolean")
    table["nonneuronal_high_depth"] = table["nonneuronal_high_depth"].astype("boolean")
    flags = qc.nonneuronal_high_depth_flags(
        [1500, 1500, 1500], ["Astro", "IT", "Unknown"], table
    )
    assert flags.tolist() == [True, False, True]


def test_nonneuronal_depth_trend_reports_a_fall_above_1000_counts() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    totals = np.concatenate([rng.integers(500, 1000, n), rng.integers(1000, 4000, n)])
    confident = np.concatenate([rng.random(n) < 0.92, rng.random(n) < 0.86])
    table, outcomes = qc.nonneuronal_depth_trend(
        totals, ["Astro"] * (2 * n), {"class": confident}, ["Astro"]
    )
    (outcome,) = outcomes
    assert outcome.effect == "report_only" and outcome.cls == "Astro"
    assert "merged or large masks" in outcome.message
    assert set(table["band"]) >= {"500-999", "1,000-1,999", ">= 2,000"}
    flat = np.concatenate([rng.random(n) < 0.9, rng.random(n) < 0.9])
    _, none = qc.nonneuronal_depth_trend(
        totals, ["Astro"] * (2 * n), {"class": flat}, ["Astro"]
    )
    assert none == ()


def test_nonneuronal_depth_trend_needs_200_cells_per_band() -> None:
    totals = np.array([600] * 150 + [1500] * 300)
    confident = np.array([True] * 150 + [False] * 300)
    _, outcomes = qc.nonneuronal_depth_trend(
        totals, ["Astro"] * 450, {"class": confident}, ["Astro"]
    )
    assert outcomes == ()


def make_factor_data(
    rng: np.random.Generator, true_log2: np.ndarray, n_cells: int = 400
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Return counts of two classes with per-gene factors ``true_log2``."""
    n_genes = len(true_log2)
    profiles = rng.dirichlet(np.ones(n_genes) * 2.0, size=2)
    frame = pd.DataFrame(profiles, index=["A", "B"])
    labels = ["A" if index % 2 == 0 else "B" for index in range(n_cells)]
    rates = profiles[[0 if label == "A" else 1 for label in labels]] * (2.0**true_log2)
    rates = rates / rates.sum(axis=1, keepdims=True) * 3000.0
    counts = rng.poisson(rates)
    return counts, labels, frame


def test_factor_remeasure_agrees_with_a_matching_table_and_warns_otherwise() -> None:
    rng = np.random.default_rng(1)
    true_log2 = rng.normal(0.0, 1.0, 60)
    counts, labels, profiles = make_factor_data(rng, true_log2)
    genes = [f"g{index}" for index in range(60)]
    stored = pd.DataFrame(
        {"tier": ["informative"] * 60, "log2_factor": true_log2}, index=genes
    )
    result = qc.factor_remeasure(
        counts, genes, labels, profiles, stored, informative_min_expected=100.0
    )
    assert result.applies and result.pearson_r is not None
    assert result.pearson_r > 0.95
    assert result.outcome is not None and not result.outcome.fired
    shuffled = stored.assign(log2_factor=rng.permutation(true_log2))
    warned = qc.factor_remeasure(
        counts,
        genes,
        labels,
        profiles,
        shuffled,
        informative_min_expected=100.0,
        asset_id="efficiency__test",
    )
    assert warned.outcome is not None and warned.outcome.fired
    assert "re-run PREP" in warned.outcome.message
    assert warned.summary()["recommendation"].endswith("not automatic")
    assert warned.summary()["trust_effect"] == "none"
    assert qc.apply_qc_outcomes("provisional", [warned.outcome]) == "provisional"


def test_factor_remeasure_uses_genes_informative_in_both_tables() -> None:
    rng = np.random.default_rng(2)
    true_log2 = rng.normal(0.0, 1.0, 40)
    counts, labels, profiles = make_factor_data(rng, true_log2)
    genes = [f"g{index}" for index in range(40)]
    tiers = ["informative"] * 20 + ["weak"] * 20
    stored = pd.DataFrame({"tier": tiers, "log2_factor": true_log2}, index=genes)
    result = qc.factor_remeasure(
        counts, genes, labels, profiles, stored, informative_min_expected=100.0
    )
    assert result.n_informative <= 20
    assert result.factors is not None
    assert not result.factors["informative"].iloc[20:].any()


def test_factor_remeasure_runs_only_on_the_first_dataset_with_a_table() -> None:
    rng = np.random.default_rng(4)
    counts, labels, profiles = make_factor_data(rng, rng.normal(0, 1, 10))
    genes = [f"g{index}" for index in range(10)]
    assert qc.factor_remeasure(counts, genes, labels, profiles, None).reason == (
        "no_stored_table"
    )
    stored = pd.DataFrame(
        {"tier": ["informative"] * 10, "log2_factor": 0.0}, index=genes
    )
    later = qc.factor_remeasure(
        counts, genes, labels, profiles, stored, first_dataset_of_family=False
    )
    assert not later.applies and later.reason == "not_first_dataset_of_family"


def test_gene_complexity_warns_above_45_percent_and_flags_predictions() -> None:
    rng = np.random.default_rng(5)
    native_totals = rng.integers(250, 499, 300)
    simulated_depth = np.full(300, 250)
    table, outcome = qc.gene_complexity_check(
        native_n_genes=np.full(300, 150.0),
        native_totals=native_totals,
        simulated_n_genes=np.full(300, 100.0),
        simulated_depth=simulated_depth,
        grid=GRID,
    )
    assert outcome.fired and outcome.effect == "warning"
    assert "Simulated coverage predictions are unreliable" in outcome.message
    assert outcome.details["coverage_predictions_reliable"] is False
    assert table.set_index("depth").loc[250, "gap"] == pytest.approx(0.5)
    _, quiet = qc.gene_complexity_check(
        native_n_genes=np.full(300, 120.0),
        native_totals=native_totals,
        simulated_n_genes=np.full(300, 100.0),
        simulated_depth=simulated_depth,
        grid=GRID,
    )
    assert not quiet.fired
    assert quiet.details["native_exceeds_simulated_in_some_bin"] is True


def test_gene_complexity_ignores_bins_with_too_few_cells() -> None:
    table, outcome = qc.gene_complexity_check(
        native_n_genes=np.full(10, 500.0),
        native_totals=np.full(10, 300),
        simulated_n_genes=np.full(10, 100.0),
        simulated_depth=np.full(10, 250),
        grid=GRID,
    )
    assert not outcome.fired
    assert not table["judged"].any()


def test_qc_summary_never_promotes() -> None:
    summary = qc.qc_summary(
        [
            qc.QcOutcome("a", True, "warning", message="m"),
            qc.QcOutcome("b", False, "report_only"),
        ]
    )
    assert summary["promotes"] is False and summary["n_fired"] == 1


def test_real_qc_config_defaults_match_the_module_constants() -> None:
    from merxen.annotation.config import AnnotationRealQcConfig

    config = AnnotationRealQcConfig()
    assert config.coverage_warn_margin == qc.COVERAGE_WARN_MARGIN
    assert config.coverage_min_cells == qc.COVERAGE_MIN_CELLS
    assert config.factor_remeasure_min_r == qc.FACTOR_REMEASURE_MIN_R
    assert config.nonneuronal_high_depth_counts == qc.NONNEURONAL_HIGH_DEPTH_COUNTS
    assert config.genes_per_count_gap_warn == qc.GENE_COMPLEXITY_GAP_WARN
    with pytest.raises(ValueError):
        AnnotationRealQcConfig(coverage_warn_margin=1.5)
