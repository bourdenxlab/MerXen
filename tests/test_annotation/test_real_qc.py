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


@pytest.mark.parametrize(
    ("species", "gate_level", "blocked"),
    [
        ("human", "full", set()),
        ("human", "broad_only", {"supercluster", "cluster"}),
        ("human", "failed", {"broad", "supercluster", "cluster"}),
        ("mouse", "broad_only", {"subclass"}),
    ],
)
def test_coverage_rows_record_the_gate_level_and_whether_it_blocks_them(
    species: str, gate_level: str, blocked: set[str]
) -> None:
    """A level the gate before QC left unattempted is marked (C15 review F2).

    Its real coverage is 0 by the gate, so the warning (kept: the rule is
    unchanged) says the shortfall is the gate's, not the simulation's.
    """
    levels = ["broad", "supercluster", "cluster"]
    if species == "mouse":
        levels = ["class", "subclass"]
    real = pd.DataFrame(
        {
            "level": levels,
            "class": ["A"] * len(levels),
            "n_cells": [500] * len(levels),
            "real_coverage": [0.0] * len(levels),
        }
    )
    predicted = real[["level", "class"]].assign(
        predicted_coverage=0.9, resolvable_share=1.0
    )
    result = qc.coverage_vs_simulation(
        real, predicted, gate_level=gate_level, species=species
    )
    table = result.table.set_index("level")
    assert set(table["gate_level"]) == {gate_level}
    assert {level for level in levels if table.loc[level, "gate_blocked"]} == blocked
    # Every row still warns: the rule does not depend on the gate.
    assert sorted(result.flagged) == sorted((level, "A") for level in levels)
    for item in result.outcomes:
        assert item.details["gate_level"] == gate_level
        assert item.details["gate_blocked"] is (item.level in blocked)
        gate_text = f"The dataset gate before QC is {gate_level}"
        assert (gate_text in item.message) is (item.level in blocked)
        assert "No offset is applied" in item.message
    summary = result.summary()
    assert summary["gate_level"] == gate_level
    assert summary["n_flagged_gate_blocked"] == len(blocked)
    assert {item["level"] for item in summary["flagged"] if item["gate_blocked"]} == (
        blocked
    )


def test_coverage_without_a_gate_level_leaves_blocking_unknown() -> None:
    real = pd.DataFrame(
        {
            "level": ["supercluster"],
            "class": ["A"],
            "n_cells": [500],
            "real_coverage": [0.0],
        }
    )
    predicted = real[["level", "class"]].assign(
        predicted_coverage=0.9, resolvable_share=1.0
    )
    result = qc.coverage_vs_simulation(real, predicted)
    assert result.table["gate_level"].tolist() == [None]
    assert result.table["gate_blocked"].tolist() == [None]
    (item,) = result.outcomes
    assert item.details["gate_blocked"] is None
    assert "dataset gate before QC" not in item.message
    summary = result.summary()
    assert summary["gate_level"] is None
    assert summary["n_flagged_gate_blocked"] == 0
    assert summary["flagged"] == [
        {"level": "supercluster", "class": "A", "gate_blocked": None}
    ]
    with pytest.raises(ValueError, match="unknown gate level"):
        qc.coverage_vs_simulation(real, predicted, gate_level="partial")
    empty = qc.coverage_vs_simulation(real.iloc[:0], predicted, gate_level="full")
    assert empty.table.empty and empty.summary()["n_flagged_gate_blocked"] == 0


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


def test_nonneuronal_high_depth_flags_can_be_restricted_to_levels() -> None:
    table = pd.DataFrame(
        class_depth_rows("Astro", {1000: 0.9}, neuronal=False, high_depth=(1000,))
        + class_depth_rows(
            "Astro",
            {1000: 0.9},
            level="subclass",
            neuronal=False,
            high_depth=(1000,),
            not_emitted=(1000,),
        )
    )
    every = qc.nonneuronal_high_depth_flags([1500], ["Astro"], table)
    subclass = qc.nonneuronal_high_depth_flags(
        [1500], ["Astro"], table, levels=["subclass"]
    )
    assert every.tolist() == [True] and subclass.tolist() == [False]


def test_dataset_class_bin_shares_are_label_free_for_thin_classes() -> None:
    totals = [100] * 150 + [600] * 50 + [15] * 3
    classes = ["A"] * 200 + ["B"] * 3
    shares = qc.dataset_class_bin_shares(totals, classes, GRID, min_class_cells=100)
    a = shares[shares["class"] == "A"].set_index("depth")
    b = shares[shares["class"] == "B"].set_index("depth")
    assert set(a["share_source"]) == {qc.SHARE_SOURCE_OWN}
    assert a.loc[100, "share"] == pytest.approx(0.75)
    # B has three cells at 15 counts: it takes every cell's histogram.
    assert set(b["share_source"]) == {qc.SHARE_SOURCE_LABEL_FREE}
    assert set(b["n_cells"]) == {3}
    assert b.loc[100, "share"] == pytest.approx(150 / 203)
    assert b.loc[500, "share"] == pytest.approx(50 / 203)
    assert b.loc[10, "share"] == pytest.approx(3 / 203)
    assert qc.CLASS_DEPTH_MIN_CLASS_CELLS == 100
    with pytest.raises(ValueError, match="min_class_cells"):
        qc.dataset_class_bin_shares(totals, classes, GRID, min_class_cells=0)


def test_dataset_class_depth_prediction_uses_each_levels_keys_and_regime() -> None:
    table = pd.DataFrame(
        class_depth_rows("A", {100: 0.5, 500: 0.9})
        + class_depth_rows("A", {100: 0.2}, level="subclass")
        + class_depth_rows("A", {100: 0.4}, level="subclass", regime="validated")
    )
    totals = [100] * 150 + [600] * 50
    keys = {"class": ["A"] * 200, "subclass": ["A"] * 100 + [None] * 100}
    prediction = qc.dataset_class_depth_prediction(
        table,
        totals,
        keys,
        {"class": "provisional", "subclass": "validated"},
        GRID,
    ).set_index(["level", "class"])
    own = prediction.loc[("class", "A")]
    assert own["share_source"] == qc.SHARE_SOURCE_OWN
    assert own["regime"] == "provisional"
    assert own["predicted_coverage"] == pytest.approx(0.75 * 0.5 + 0.25 * 0.9)
    assert own["resolvable_share"] == pytest.approx(1.0)
    # The subclass key covers 100 cells, all at 100 counts, in the validated
    # regime's rows.
    sub = prediction.loc[("subclass", "A")]
    assert sub["regime"] == "validated" and sub["n_cells"] == 100
    assert sub["predicted_coverage"] == pytest.approx(0.4)
    empty = qc.dataset_class_depth_prediction(
        table.iloc[0:0],
        totals,
        keys,
        {"class": "provisional", "subclass": "provisional"},
        GRID,
    )
    assert empty.empty


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
    # A later dataset is not the first whether or not a table is given: the
    # check does not apply to it (not_applicable, never not_evaluable).
    later_without = qc.factor_remeasure(
        counts, genes, labels, profiles, None, first_dataset_of_family=False
    )
    assert later_without.reason == "not_first_dataset_of_family"
    outcome = qc.factor_remeasure_outcome(
        later_without, min_r=qc.FACTOR_REMEASURE_MIN_R, has_r3_member=True
    )
    assert outcome.outcome == "not_applicable"


def test_factor_remeasure_warns_on_an_undefined_pearson_r() -> None:
    """A constant factor vector has no Pearson r (NaN): that warns, never passes."""
    rng = np.random.default_rng(6)
    true_log2 = rng.normal(0.0, 1.0, 30)
    counts, labels, profiles = make_factor_data(rng, true_log2)
    genes = [f"g{index}" for index in range(30)]
    flat = pd.DataFrame({"tier": ["informative"] * 30, "log2_factor": 0.0}, index=genes)
    result = qc.factor_remeasure(
        counts, genes, labels, profiles, flat, informative_min_expected=100.0
    )
    assert result.applies and result.n_informative >= 3
    assert result.pearson_r is not None and np.isnan(result.pearson_r)
    assert result.outcome is not None and result.outcome.fired
    outcome = qc.factor_remeasure_outcome(
        result, min_r=qc.FACTOR_REMEASURE_MIN_R, has_r3_member=True
    )
    assert outcome.fired and outcome.outcome == "warn"
    assert "r nan" in outcome.message


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
        matching="lower_edge",
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
        matching="lower_edge",
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
        matching="lower_edge",
    )
    assert not outcome.fired
    assert not table["judged"].any()


# --------------------------------------------------------------------------
# NR7's depth matching (the user's ruling C3 (b) of 2026-10-07)

C16_GRID = [10, 15, 30, 60, 120, 250]


def _genes_at(totals: np.ndarray) -> np.ndarray:
    """Genes per cell without any complexity gap: linear in log depth."""
    return 40.0 * np.log(np.asarray(totals, dtype=np.float64))


def _no_gap_cells(n_native: int = 400, n_test: int = 200) -> dict[str, np.ndarray]:
    """Native cells and simulated test cells on one genes-per-depth curve."""
    rng = np.random.default_rng(16)
    native_totals = np.exp(rng.uniform(np.log(30), np.log(500), n_native))
    cells = np.repeat([f"t{index}" for index in range(n_test)], len(C16_GRID))
    depths = np.tile(C16_GRID, n_test).astype(np.float64)
    return {
        "native_n_genes": _genes_at(native_totals),
        "native_totals": native_totals,
        "simulated_n_genes": _genes_at(depths),
        "simulated_depth": depths,
        "simulated_cell_ids": cells,
    }


def test_nr7_interpolates_simulated_genes_to_the_native_bin_median_total() -> None:
    """Without a complexity gap the interpolated gap is 0; the lower edge's is not."""
    cells = _no_gap_cells()
    table, outcome = qc.gene_complexity_check(grid=C16_GRID, **cells)
    rows = table.set_index("depth")
    inside = rows.loc[[30, 60, 120]]
    assert (inside["matching"] == "interpolated").all()
    assert inside["judged"].all()
    assert inside["gap"].abs().max() < 1e-3
    assert not outcome.fired
    assert outcome.details["matching"] == "interpolated"
    # The weight puts the bin's native median total between its edges.
    for depth, upper in ((30, 60), (60, 120), (120, 250)):
        row = rows.loc[depth]
        assert row["simulated_depth_high"] == upper
        expected = (np.log(row["native_median_total"]) - np.log(depth)) / (
            np.log(upper) - np.log(depth)
        )
        assert row["interpolation_weight"] == pytest.approx(expected)
        assert 0.0 <= row["interpolation_weight"] < 1.0
    # The registered lower edge sees a gap where there is none.
    edge, _ = qc.gene_complexity_check(grid=C16_GRID, matching="lower_edge", **cells)
    edge_rows = edge.set_index("depth")
    assert (edge_rows.loc[[30, 60, 120], "gap"] > 0.05).all()
    assert (edge_rows["matching"] == "lower_edge").all()


def test_nr7_keeps_the_lower_edge_for_the_open_top_bin() -> None:
    cells = _no_gap_cells()
    table, _ = qc.gene_complexity_check(grid=C16_GRID, **cells)
    top = table.set_index("depth").loc[250]
    # No upper edge above the grid: the cells at 250, one value per test cell.
    assert top["matching"] == "lower_edge"
    assert pd.isna(top["simulated_depth_high"])
    assert np.isnan(top["interpolation_weight"])
    assert top["n_simulated"] == 200
    assert top["simulated_median_genes"] == pytest.approx(40.0 * np.log(250))
    assert top["gap"] > 0


def test_nr7_matches_only_test_cells_simulated_at_both_edges() -> None:
    """A test cell without a row at the upper edge is left out of the bin."""
    cells = _no_gap_cells(n_test=80)
    keep = ~(
        np.isin(cells["simulated_cell_ids"], [f"t{index}" for index in range(40)])
        & (cells["simulated_depth"] == 60)
    )
    thinned = {
        key: value[keep] if key.startswith("simulated") else value
        for key, value in cells.items()
    }
    table, _ = qc.gene_complexity_check(grid=C16_GRID, **thinned)
    rows = table.set_index("depth")
    # Bins 30 (edges 30, 60) and 60 (edges 60, 120) lose the 40 cells.
    assert rows.loc[30, "n_simulated"] == 40
    assert rows.loc[60, "n_simulated"] == 40
    assert rows.loc[120, "n_simulated"] == 80
    # 40 matched test cells are below the 50-cell minimum: not judged.
    assert not rows.loc[30, "judged"] and not rows.loc[60, "judged"]
    assert rows.loc[120, "judged"]


def test_nr7_interpolation_needs_the_simulated_cells_test_cells() -> None:
    cells = _no_gap_cells()
    del cells["simulated_cell_ids"]
    with pytest.raises(ValueError, match="simulated_cell_ids"):
        qc.gene_complexity_check(grid=C16_GRID, **cells)
    with pytest.raises(ValueError, match="unknown gene-complexity matching"):
        qc.gene_complexity_check(grid=C16_GRID, matching="nearest", **cells)
    # The lower edge needs none.
    table, _ = qc.gene_complexity_check(grid=C16_GRID, matching="lower_edge", **cells)
    assert table["judged"].any()


def test_nr7_warns_on_a_real_gap_under_interpolation() -> None:
    cells = _no_gap_cells()
    cells["native_n_genes"] = 1.6 * cells["native_n_genes"]
    table, outcome = qc.gene_complexity_check(grid=C16_GRID, **cells)
    inside = table.set_index("depth").loc[[30, 60, 120], "gap"]
    assert inside.to_numpy() == pytest.approx([0.6, 0.6, 0.6], abs=1e-3)
    assert outcome.fired
    assert outcome.details["matching"] == "interpolated"
    assert {30, 60, 120} <= set(outcome.details["bins_warned"])


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
    # The label-free depth of thin classes decides no warning at the default
    # (pre-registration §22.9): every class the warning judges has its own.
    assert config.coverage_min_cells >= qc.CLASS_DEPTH_MIN_CLASS_CELLS
    assert config.factor_remeasure_min_r == qc.FACTOR_REMEASURE_MIN_R
    assert config.nonneuronal_high_depth_counts == qc.NONNEURONAL_HIGH_DEPTH_COUNTS
    assert config.genes_per_count_gap_warn == qc.GENE_COMPLEXITY_GAP_WARN
    with pytest.raises(ValueError):
        AnnotationRealQcConfig(coverage_warn_margin=1.5)
