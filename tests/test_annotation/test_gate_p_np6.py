"""Tests for gate P's NP6 stress sensitivity (M13; plan §14 NP6, pre-reg §23.9)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import gate_p as gp
from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    AnnotationResolvabilityConfig,
    AnnotationThresholds,
)

from .test_gate_p import _calls, _decisions
from .test_resolvability import (
    GRID,
    MARKERS,
    bootstrap_mapper,
    make_test_cells,
    settings,
    synthetic_specs,
    tracked_cells,
)

CONFIG = AnnotationResolvabilityConfig()
STRESS = res.GATE_P_STRESS_SPILL
ONE_TYPE = {"A": 1.0}


def _np6(**updates: object) -> gp.Np6Settings:
    return dataclasses.replace(gp.Np6Settings.from_config(CONFIG), **updates)


def _as(frame: pd.DataFrame, recipe: str) -> pd.DataFrame:
    """The same rows as another recipe's simulation."""
    return frame.assign(recipe=recipe)


def _rows(
    replicates: Mapping[gp.ReplicateKey, pd.DataFrame], recipe: str
) -> gp.SimulationRows:
    return gp.SimulationRows(
        {key: _as(table, recipe) for key, table in replicates.items()}, recipe
    )


def _np6_tables(
    base: pd.DataFrame,
    stressed: pd.DataFrame,
    depths: Sequence[int],
    *,
    composition: Mapping[str, float] = ONE_TYPE,
    clean: pd.DataFrame | None = None,
    np6: gp.Np6Settings | None = None,
) -> tuple[dict[tuple[str, str], list[res.GatePTestedSet] | None], pd.DataFrame]:
    """Tested sets of the base and NP6 verdicts of one stress recipe."""
    decisions = _decisions(depths)
    tested = res.gate_p_tested_sets(base, decisions, regime="provisional")
    np6 = np6 or _np6()
    stats = gp.np6_set_stats(
        _rows({("D1", 0): base}, res.DECISION_RECIPE),
        _rows({("D1", 0): stressed}, STRESS),
        decisions,
        tested,
        default_group=None,
        settings=np6,
        composition=composition,
        clean=None if clean is None else _rows({("D1", 0): clean}, res.CLEAN_RECIPE),
    )
    return tested, gp.np6_verdicts(stats, AnnotationThresholds(), np6)


def _row(verdicts: pd.DataFrame, scheme: str, tested_set: str) -> pd.Series:
    rows = verdicts[
        (verdicts["scheme"] == scheme) & (verdicts["tested_set"] == tested_set)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


# --------------------------------------------------------------------------
# §12 M13: a stress recipe that removes markers fails NP6


def _simulate(
    test: res.HeldOutCells, recipe: res.SimulationRecipe, efficiency: np.ndarray
) -> pd.DataFrame:
    query = res.thin_and_contaminate_v7(test, GRID, recipe, efficiency=efficiency)
    tidy = bootstrap_mapper(query, recipe.member, 0)
    return res.level_cells(tidy, query, test, synthetic_specs(), seed=0)


def test_a_stress_recipe_that_removes_a_class_markers_fails_np6() -> None:
    """§12 M13: "a stress recipe that removes markers fails NP6".

    The synthetic reference separates classes A and B by ten marker genes
    each. A recipe that sets A's markers' efficiency to ~0 leaves A cells
    without A counts: they are called B (B's confident calls fall to
    precision .50) and A has no confident call left. Both fail NP6 at the
    class level. The registered spill 0.35 stress leaves B passing.
    """
    test = make_test_cells(n_per_cluster=120)
    base = res.member_recipe(res.DECISION_RECIPE, 0, CONFIG)
    spill = res.member_recipe(res.GATE_P_STRESS_SPILL, 0, CONFIG)
    knockout = dataclasses.replace(base, name="R1_marker_knockout")
    efficiency = res.member_efficiency(base, test.genes)
    removed = efficiency.copy()
    removed[MARKERS["A"]] = 1e-9
    cells = {
        base.name: _simulate(test, base, efficiency),
        spill.name: _simulate(test, spill, res.member_efficiency(spill, test.genes)),
        knockout.name: _simulate(test, knockout, removed),
    }
    specs = synthetic_specs()
    decisions = res.decide(
        cells[base.name], [spec.meta for spec in specs], list(GRID), settings()
    )
    tested = {
        key: items
        for key, items in res.gate_p_tested_sets(
            cells[base.name], decisions, regime="provisional"
        ).items()
        if key[0] == "class"
    }
    assert set(tested) == {("class", "A"), ("class", "B")}
    assert all(items for items in tested.values())
    np6 = _np6()
    composition = dict.fromkeys(("A1", "A2", "B1", "B2"), 0.25)
    verdicts: dict[str, pd.DataFrame] = {}
    for recipe in (spill, knockout):
        stats = gp.np6_set_stats(
            gp.SimulationRows({("D1", 0): cells[base.name]}, base.name),
            gp.SimulationRows({("D1", 0): cells[recipe.name]}, recipe.name),
            decisions,
            tested,
            default_group=None,
            settings=np6,
            composition=composition,
        )
        verdicts[recipe.name] = gp.np6_verdicts(stats, AnnotationThresholds(), np6)
    knocked = verdicts[knockout.name]
    b_rows = knocked[knocked["class"] == "B"]
    assert (b_rows["precision_stress"] < 0.6).all()
    assert not b_rows["point_ok"].any() and not b_rows["drop_ok"].any()
    a_rows = knocked[knocked["class"] == "A"]
    assert (a_rows["n_stress"] == 0).all() and not a_rows["passed"].any()
    # A's shallow set has no stressed call, so it is pooled with the deeper one.
    assert "30+100" in set(a_rows["set"])
    assert gp.np6_class_verdicts(
        knocked, tested, stresses=[knockout.name], settings=np6
    ) == {("class", "A"): False, ("class", "B"): False}
    spilled = verdicts[spill.name]
    assert spilled[spilled["class"] == "B"]["passed"].all()
    unweighted = spilled[spilled["scheme"] == gp.NP3_UNWEIGHTED]
    assert unweighted["passed"].all()
    classes = gp.np6_class_verdicts(
        spilled, tested, stresses=[spill.name], settings=np6
    )
    assert classes[("class", "B")] is True
    both = pd.concat([spilled, knocked], ignore_index=True)
    combined = gp.np6_class_verdicts(
        both, tested, stresses=[spill.name, knockout.name], settings=np6
    )
    assert combined[("class", "B")] is False


# --------------------------------------------------------------------------
# Thin stressed sets


def test_a_thin_stressed_set_is_pooled_with_the_next_deeper_set() -> None:
    """§14 NP6: a set left with < 200 stressed calls joins the next deeper set.

    The base holds 300 confident calls at 30, 60 and 100 (three sets tested
    on their own). Stressed, 30 keeps 120 and 60 keeps 50 confident calls:
    30 pools with 60 (170) and then 100 (470); 60 pools with 100 (350); 100
    keeps its 300. With 150 at 100, the deepest set is still thin and is
    scored on its own calls.
    """
    depths = (30, 60, 100)

    def table(confident: Mapping[int, int]) -> pd.DataFrame:
        return pd.concat(
            [
                _calls(
                    depth,
                    [
                        ("A", "X", confident[depth], confident[depth], 0.95),
                        ("A", "X", 300 - confident[depth], 0, 0.5),
                    ],
                    prefix=f"d{depth}_",
                )
                for depth in depths
            ],
            ignore_index=True,
        )

    base = table(dict.fromkeys(depths, 300))
    tested, verdicts = _np6_tables(base, table({30: 120, 60: 50, 100: 300}), depths)
    assert [gp.tested_set_label(item) for item in tested[("broad", "X")] or []] == [
        "100",
        "60",
        "30",
    ]
    rows = verdicts[verdicts["scheme"] == gp.NP3_UNWEIGHTED].set_index("tested_set")
    assert rows.loc["30", "set"] == "30+60+100"
    assert rows.loc["30", "pooled_with"] == "60;100"
    assert rows.loc["30", "depths"] == "30;60;100"
    assert rows.loc["30", "n_stressed_tested"] == 120
    assert rows.loc["30", "n_stress"] == 470 and rows.loc["30", "n_base"] == 900
    assert rows.loc["60", "set"] == "60+100" and rows.loc["60", "n_stress"] == 350
    assert rows.loc["100", "set"] == "100" and rows.loc["100", "pooled_with"] == ""
    assert not rows["below_min_confident_n"].any()
    assert rows["passed"].all()
    _, verdicts = _np6_tables(base, table({30: 120, 60: 50, 100: 150}), depths)
    rows = verdicts[verdicts["scheme"] == gp.NP3_UNWEIGHTED].set_index("tested_set")
    assert rows.loc["100", "set"] == "100"
    assert rows.loc["100", "below_min_confident_n"]
    assert rows.loc["100", "n_stress"] == 150
    assert not rows.loc["30", "below_min_confident_n"]


def test_a_pooled_stressed_set_counts_each_cell_once_at_its_deepest_row() -> None:
    """Pooling a bin with the ">= D_P" set keeps each test cell once.

    150 cells ``a`` reach 100 counts and 150 cells ``b`` only 30. The base
    holds 300 calls at 30 (a set of its own) and 150 at 100, so its
    ">= 30" set pools a's 100 rows with b's 30 rows. Stressed, a stays
    confident at both depths and only 20 of b at 30: the bin 30 holds 170
    calls and joins ">= 30". There a counts once, at 100: 170 calls, not
    the 320 of the two scopes stacked.
    """
    good = (0.95, True)
    base = tracked_cells([("a", 150, {30: good, 100: good}), ("b", 150, {30: good})])
    base[res.TRUTH_LEAF_COLUMN] = "A"
    stressed = base.copy()
    b30 = stressed["cell_id"].str.startswith("b_") & (stressed["depth"] == 30)
    index = stressed.index[b30][20:]
    stressed.loc[index, "bp"] = 0.5
    tested, verdicts = _np6_tables(base, stressed, (30, 100))
    assert sorted(
        gp.tested_set_label(item) for item in tested[("broad", "X")] or []
    ) == [
        "30",
        ">=30",
    ]
    row = _row(verdicts, gp.NP3_UNWEIGHTED, "30")
    assert row["n_stressed_tested"] == 170
    assert row["set"] == "30+>=30" and row["depths"] == "30;100"
    assert row["n_stress"] == 170 and row["n_base"] == 300
    assert row["below_min_confident_n"]
    pooled = _row(verdicts, gp.NP3_UNWEIGHTED, ">=30")
    assert pooled["set"] == ">=30" and pooled["n_stress"] == 170


# --------------------------------------------------------------------------
# The criteria


def test_np6_needs_target_l_and_a_wilson_bound_of_target_less_002() -> None:
    """§14 NP6: point >= target_L, Wilson >= target_L - 0.02 (no margin).

    At broad (target .90): .905 on 1,000 calls (bound .885) passes, though
    NP3 would need .95; .90 on 400 calls (bound .867) fails the bound only.
    """
    base = _calls(100, [("A", "X", 1000, 1000, 0.95)])
    _, verdicts = _np6_tables(base, _calls(100, [("A", "X", 1000, 905, 0.95)]), [100])
    row = _row(verdicts, gp.NP3_UNWEIGHTED, "100")
    assert row["target"] == pytest.approx(0.90)
    assert row["min_wilson"] == pytest.approx(0.88)
    assert row["wilson_lb_stress"] == pytest.approx(0.8852, abs=1e-4)
    assert row["point_ok"] and row["wilson_ok"]
    base = _calls(100, [("A", "X", 400, 380, 0.95)])
    _, verdicts = _np6_tables(base, _calls(100, [("A", "X", 400, 360, 0.95)]), [100])
    row = _row(verdicts, gp.NP3_UNWEIGHTED, "100")
    assert row["precision_stress"] == pytest.approx(0.90)
    assert row["wilson_lb_stress"] == pytest.approx(0.8667, abs=1e-4)
    assert row["point_ok"] and not row["wilson_ok"] and row["drop_ok"]
    assert not row["passed"]
    below = _calls(100, [("A", "X", 1000, 899, 0.95)])
    _, verdicts = _np6_tables(_calls(100, [("A", "X", 1000, 905, 0.95)]), below, [100])
    assert not _row(verdicts, gp.NP3_UNWEIGHTED, "100")["point_ok"]


def test_np6_fails_a_drop_significantly_above_005() -> None:
    """D12: fail when the one-sided 95% lower bound of p_b - p_s exceeds .05.

    1.00 -> .92 on 1,000 calls each: drop .08, SE .0086, bound .066 > .05
    fails although .92 clears the floor and its bound (.902). The same drop
    of .06 on 200 calls each (.98 -> .92) is not significant (bound .025).
    """
    drop, se, lower = gp.np6_drop_test(1.0, 1000, 0.92, 1000)
    assert drop == pytest.approx(0.08)
    assert se == pytest.approx(math.sqrt(0.92 * 0.08 / 1000))
    assert lower == pytest.approx(0.08 - 1.6448536 * se)
    assert lower == pytest.approx(0.0659, abs=1e-4)
    base = _calls(100, [("A", "X", 1000, 1000, 0.95)])
    _, verdicts = _np6_tables(base, _calls(100, [("A", "X", 1000, 920, 0.95)]), [100])
    row = _row(verdicts, gp.NP3_UNWEIGHTED, "100")
    assert row["drop"] == pytest.approx(0.08)
    assert row["drop_lower"] == pytest.approx(lower)
    assert row["point_ok"] and row["wilson_ok"]
    assert not row["drop_ok"] and not row["passed"]
    _, _, small = gp.np6_drop_test(0.98, 200, 0.92, 200)
    assert small == pytest.approx(0.0245, abs=1e-4)
    base = _calls(100, [("A", "X", 400, 388, 0.95)])
    _, verdicts = _np6_tables(base, _calls(100, [("A", "X", 400, 374, 0.95)]), [100])
    row = _row(verdicts, gp.NP3_UNWEIGHTED, "100")
    assert row["drop"] == pytest.approx(0.035)
    assert row["drop_ok"] and row["passed"]
    # A stress that raises the precision is no drop; nan inputs give nan.
    assert gp.np6_drop_test(0.9, 100, 1.0, 100)[2] < 0
    assert math.isnan(gp.np6_drop_test(0.9, 100, math.nan, 0)[2])
    assert math.isnan(gp.np6_drop_test(0.9, 0, 0.9, 100)[1])


def test_np6_tests_the_drop_on_each_sets_kish_n() -> None:
    """D12: the drop test's standard error uses each set's Kish n.

    X's calls are 180 of truth type A and 20 of B. Class-balanced, the Kish
    n of the set falls from 200 to 72 in both simulations, which widens the
    drop test; the unweighted test keeps n = 200. Each weighting decides.
    """
    base = _calls(100, [("A", "X", 180, 180, 0.95), ("B", "X", 20, 20, 0.95)])
    stressed = _calls(100, [("A", "X", 180, 171, 0.95), ("B", "X", 20, 17, 0.95)])
    _, verdicts = _np6_tables(base, stressed, [100], composition={"A": 0.9, "B": 0.1})
    balanced = _row(verdicts, gp.NP3_CLASS_BALANCED, "100")
    assert balanced["kish_n_base"] == pytest.approx(72.0)
    assert balanced["kish_n_stress"] == pytest.approx(72.0)
    assert balanced["precision_stress"] == pytest.approx((0.95 + 0.85) / 2)
    drop, se, lower = gp.np6_drop_test(1.0, 72.0, 0.90, 72.0)
    assert balanced["drop_se"] == pytest.approx(se)
    assert balanced["drop_lower"] == pytest.approx(lower)
    unweighted = _row(verdicts, gp.NP3_UNWEIGHTED, "100")
    assert unweighted["kish_n_stress"] == pytest.approx(200.0)
    assert unweighted["precision_stress"] == pytest.approx(188 / 200)
    natural = _row(verdicts, gp.NP3_NATURAL, "100")
    assert natural["kish_n_stress"] == pytest.approx(200.0)
    assert set(verdicts["scored"]) == {True}
    # Every weighting is scored: the class-balanced bound (.818 < .88) fails
    # the class though the unweighted and natural rows pass.
    assert unweighted["passed"] and natural["passed"]
    assert not balanced["wilson_ok"] and not balanced["passed"]


def test_np6_reports_the_clean_upper_bound_and_coverage_changes() -> None:
    base = _calls(
        100, [("A", "X", 400, 390, 0.95), ("A", "X", 100, 100, 0.5)], prefix="b"
    )
    stressed = _calls(
        100, [("A", "X", 300, 290, 0.95), ("A", "X", 200, 200, 0.5)], prefix="s"
    )
    clean = _calls(100, [("A", "X", 450, 450, 0.95), ("A", "X", 50, 50, 0.5)])
    _, verdicts = _np6_tables(base, stressed, [100], clean=clean)
    row = _row(verdicts, gp.NP3_UNWEIGHTED, "100")
    assert row["coverage_base"] == pytest.approx(0.8)
    assert row["coverage_stress"] == pytest.approx(0.6)
    assert row["coverage_change"] == pytest.approx(-0.2)
    assert row["n_clean"] == 450 and row["precision_clean"] == pytest.approx(1.0)
    assert row["coverage_clean"] == pytest.approx(0.9)
    _, verdicts = _np6_tables(base, stressed, [100])
    row = _row(verdicts, gp.NP3_UNWEIGHTED, "100")
    assert row["n_clean"] == 0 and math.isnan(row["precision_clean"])


def test_np6_scores_the_default_group_on_its_check_half_only() -> None:
    """§14 NP6 pools every held-out donor at seed 0, the default on its check half."""
    decisions = _decisions([100])
    d1 = _calls(100, [("A", "X", 400, 400, 0.95)], prefix="d1_")
    d2 = _calls(100, [("A", "X", 300, 300, 0.95)], prefix="d2_")
    scored = gp.held_out_replicates({("D1", 0): d1, ("D2", 0): d2}, default_group="D1")
    tested = res.gate_p_tested_sets(
        pd.concat(scored.values(), ignore_index=True), decisions, regime="provisional"
    )
    replicates = {("D1", 0): d1, ("D2", 0): d2, ("D1", 1): d1, ("D2", 1): d2}
    np6 = _np6()
    stats = gp.np6_set_stats(
        _rows(replicates, res.DECISION_RECIPE),
        _rows(replicates, STRESS),
        decisions,
        tested,
        default_group="D1",
        settings=np6,
        composition=ONE_TYPE,
    )
    row = stats[stats["scheme"] == gp.NP3_UNWEIGHTED].iloc[0]
    assert row["n_base"] == row["n_stress"] == 200 + 300
    leak = d2.copy()
    leak.loc[leak.index[0], "cell_id"] = d1.loc[d1["half"] == 0, "cell_id"].iloc[0]
    with pytest.raises(ValueError, match="fit-half"):
        gp.np6_set_stats(
            _rows({("D1", 0): d1, ("D2", 0): d2}, res.DECISION_RECIPE),
            _rows({("D1", 0): d1, ("D2", 0): leak}, STRESS),
            decisions,
            tested,
            default_group="D1",
            settings=np6,
            composition=ONE_TYPE,
        )


# --------------------------------------------------------------------------
# Class verdicts and inputs


def test_np6_class_verdicts_need_every_stress_recipe_and_weighting() -> None:
    base = _calls(100, [("A", "X", 400, 400, 0.95)])
    tested, passing = _np6_tables(
        base, _calls(100, [("A", "X", 400, 390, 0.95)]), [100]
    )
    _, failing = _np6_tables(base, _calls(100, [("A", "X", 400, 300, 0.95)]), [100])
    np6 = _np6()
    tested = {**tested, ("broad", "Y"): None}
    assert gp.np6_class_verdicts(passing, tested, stresses=[STRESS], settings=np6) == {
        ("broad", "X"): True,
        ("broad", "Y"): None,
    }
    other = failing.assign(stress=res.GATE_P_STRESS_LOGNORMAL)
    both = pd.concat([passing, other], ignore_index=True)
    stresses = [STRESS, res.GATE_P_STRESS_LOGNORMAL]
    verdict = gp.np6_class_verdicts(both, tested, stresses=stresses, settings=np6)
    assert verdict[("broad", "X")] is False
    # Every member of a version-7 family must pass (every_member_verdict).
    members = {
        "R1_contam_HO@0": gp.np6_class_verdicts(
            passing, tested, stresses=[STRESS], settings=np6
        ),
        "R1_contam_HO@6": verdict,
    }
    combined = res.every_member_verdict(members)
    assert combined[("broad", "X")]["status"] == res.GATE_P_FAILED
    assert combined[("broad", "X")]["failed_members"] == ["R1_contam_HO@6"]
    with pytest.raises(ValueError, match="no NP6 verdict"):
        gp.np6_class_verdicts(passing, tested, stresses=stresses, settings=np6)
    with pytest.raises(ValueError, match="no NP6 verdict"):
        gp.np6_class_verdicts(
            passing[passing["scheme"] != gp.NP3_NATURAL],
            tested,
            stresses=[STRESS],
            settings=np6,
        )
    with pytest.raises(ValueError, match="no stress recipe"):
        gp.np6_class_verdicts(passing, tested, stresses=[], settings=np6)
    with pytest.raises(ValueError, match="no passed value"):
        gp.np6_class_verdicts(
            passing.assign(passed=None), tested, stresses=[STRESS], settings=np6
        )
    # A report-only weighting does not decide.
    unweighted_only = _np6(scored_schemes=(gp.NP3_UNWEIGHTED,))
    marked = failing.assign(scored=failing["scheme"] == gp.NP3_UNWEIGHTED)
    marked.loc[marked["scheme"] == gp.NP3_UNWEIGHTED, "passed"] = True
    assert gp.np6_class_verdicts(
        marked, tested, stresses=[STRESS], settings=unweighted_only
    )[("broad", "X")]


def test_np6_inputs_that_mismatch_or_lack_rows_raise() -> None:
    base = _calls(100, [("A", "X", 400, 400, 0.95)], prefix="d1_")
    decisions = _decisions([100])
    tested = res.gate_p_tested_sets(base, decisions, regime="provisional")
    np6 = _np6()
    two = _calls(100, [("A", "X", 400, 400, 0.95)], prefix="d2_")
    with pytest.raises(ValueError, match="other groups"):
        gp.np6_set_stats(
            _rows({("D1", 0): base, ("D2", 0): two}, res.DECISION_RECIPE),
            _rows({("D1", 0): base}, STRESS),
            decisions,
            tested,
            default_group=None,
            settings=np6,
            composition=ONE_TYPE,
        )
    with pytest.raises(ValueError, match="stressed simulation .* no rows"):
        gp.np6_set_stats(
            _rows({("D1", 0): base}, res.DECISION_RECIPE),
            gp.SimulationRows({("D1", 0): base}, STRESS),
            decisions,
            tested,
            default_group=None,
            settings=np6,
            composition=ONE_TYPE,
        )
    with pytest.raises(ValueError, match="natural composition"):
        gp.np6_set_stats(
            _rows({("D1", 0): base}, res.DECISION_RECIPE),
            _rows({("D1", 0): base}, STRESS),
            decisions,
            tested,
            default_group=None,
            settings=np6,
        )
    with pytest.raises(ValueError, match="no positive share"):
        gp.np6_set_stats(
            _rows({("D1", 0): base}, res.DECISION_RECIPE),
            _rows({("D1", 0): base}, STRESS),
            decisions,
            tested,
            default_group=None,
            settings=np6,
            composition={"B": 1.0},
        )
    with pytest.raises(ValueError, match="belongs to"):
        gp.np6_set_stats(
            _rows({("D1", 0): base}, res.DECISION_RECIPE),
            _rows({("D1", 0): base}, STRESS),
            decisions,
            {("broad", "Y"): tested[("broad", "X")]},
            default_group=None,
            settings=np6,
            composition=ONE_TYPE,
        )
    # Without the natural weighting scored, no composition is needed.
    stats = gp.np6_set_stats(
        _rows({("D1", 0): base}, res.DECISION_RECIPE),
        _rows({("D1", 0): base}, STRESS),
        decisions,
        tested,
        default_group=None,
        settings=_np6(scored_schemes=(gp.NP3_UNWEIGHTED, gp.NP3_CLASS_BALANCED)),
    )
    assert set(stats["scheme"]) == {gp.NP3_UNWEIGHTED, gp.NP3_CLASS_BALANCED}
    assert gp.np6_verdicts(stats.iloc[:0], AnnotationThresholds(), np6).empty


def test_np6_settings_follow_the_config() -> None:
    np6 = gp.Np6Settings.from_config(CONFIG)
    assert np6.min_confident_n == 200
    assert np6.wilson_margin == pytest.approx(0.02)
    assert np6.max_drop == pytest.approx(0.05)
    assert np6.drop_z == pytest.approx(1.6448536, abs=1e-7)
    assert np6.scored_schemes == (
        gp.NP3_UNWEIGHTED,
        gp.NP3_NATURAL,
        gp.NP3_CLASS_BALANCED,
    )
    assert (np6.weight_min_type_cells, np6.weight_trim_factor) == (20, 10.0)
    custom = gp.Np6Settings.from_config(
        AnnotationResolvabilityConfig(
            gate_p_min_confident_n=300, gate_p_replicate_min_confident_n=100
        )
    )
    assert custom.min_confident_n == 300
    for bad in (
        {"min_confident_n": 0},
        {"wilson_margin": -0.01},
        {"max_drop": 1.5},
        {"drop_z": 0.0},
        {"scored_schemes": ()},
        {"scored_schemes": (gp.NP3_DEPTH_HISTOGRAM,)},
    ):
        with pytest.raises(ValueError, match="Np6Settings"):
            _np6(**bad)
    assert gp.SimulationRows({}, None, "R1_stress_spill@6").name == "R1_stress_spill@6"
    assert gp.SimulationRows({}, STRESS).name == STRESS
