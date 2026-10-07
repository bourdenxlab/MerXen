"""Tests for the gate-P criteria revision of 2026-10-07 (pre-registration §23.21).

R1 (NP3 and NP6 read on the test cells of a set's scope), R2 (one depth rule
for NP3-NP7), R3 (c) (NP5's t* part as a consequence check; version 7 on the
ensemble's t* re-fitted per replicate), R5 (the dry run reports an H18 class
that is not evaluable), R6 (NP5 compares a bin where both runs hold 50 test
cells) and R7 (version 7: NP5 on the ensemble's re-derived emission; its
driver test is in ``test_gate_p_run``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Mapping

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import gate_p as gp
from merxen.annotation import resolvability as res
from merxen.annotation.config import AnnotationThresholds

from .test_gate_p import (
    GRID_NP5,
    KEYS_NP5,
    _agreement,
    _calls,
    _decisions,
    _np3_rows,
    _np5,
    _np5_decisions,
    _scheme,
    _set,
    _thresholds,
    _values,
)
from .test_gate_p_assembly import _assemble, _class_sets
from .test_gate_p_np6 import STRESS, _np6, _np6_tables
from .test_resolvability import bin_cells, settings

# The truth types of ``_wrong_type_cells`` and their natural shares.
COMPOSITION = {"A": 0.5, "W": 0.5}
TEST_CELL_SCHEMES = (gp.NP3_NATURAL_TEST_CELLS, gp.NP3_CLASS_BALANCED_TEST_CELLS)
CALL_SET_SCHEMES = (gp.NP3_NATURAL, gp.NP3_CLASS_BALANCED)


def _wrong_type_cells() -> pd.DataFrame:
    """Broad calls at 100 counts, bp .95: X's 300 right calls of type A and 5
    wrong ones of type W (truth class Y), whose other 495 test cells are
    called Y (not emitted, so never confident).
    """
    called_x = _calls(100, [("A", "X", 300, 300, 0.95), ("W", "Y", 5, 0, 0.95)])
    called_y = _calls(100, [("W", "Y", 495, 495, 0.95)], parent="Y", prefix="y")
    return pd.concat([called_x, called_y], ignore_index=True)


# --------------------------------------------------------------------------
# R1: NP3 and NP6 on the test cells of the set's scope


def test_np3_scores_the_two_weightings_read_on_the_test_cells() -> None:
    """R1: NP3's scored weightings weigh each called cell by pi_t / N_t.

    Per call set (A1 (a), now reported only), the 5 wrong W calls stand for
    W's whole share: each takes the trim cap (10 x the median weight .508),
    so both weightings give 300 x .508 / (300 x .508 + 5 x 5.08) = .857, far
    below target+ .95. On the test cells of the scope, W's 500 test cells
    carry its share, so a wrong call weighs what one of them weighs: 300 /
    305 = .984 under both, and the set passes.
    """
    assert gp.NP3_SCORED_SCHEMES == TEST_CELL_SCHEMES
    tested, _, verdicts = _np3_rows(_wrong_type_cells(), [100], composition=COMPOSITION)
    rows = verdicts[verdicts["class"] == "X"]
    scored = dict(zip(rows["scheme"], rows["scored"].astype(bool), strict=True))
    assert scored == {
        gp.NP3_UNWEIGHTED: False,
        gp.NP3_NATURAL: False,
        gp.NP3_CLASS_BALANCED: False,
        gp.NP3_NATURAL_TEST_CELLS: True,
        gp.NP3_CLASS_BALANCED_TEST_CELLS: True,
    }
    for scheme in TEST_CELL_SCHEMES:
        row = _scheme(rows, scheme)
        assert row["precision"] == pytest.approx(300 / 305)
        assert row["kish_n"] == pytest.approx(305)
        assert row["passed"]
    for scheme in CALL_SET_SCHEMES:
        row = _scheme(rows, scheme)
        assert row["precision"] == pytest.approx(0.857, abs=1e-3)
        assert not row["point_ok"] and not row["passed"]
    table = gp.validated_min_depth(verdicts, tested, [100])
    record = table[table["class"] == "X"].iloc[0]
    assert record["passed"] is True and record["validated_min_depth"] == 100


def test_np6_scores_unweighted_and_the_test_cell_weightings() -> None:
    """R1: NP6 weights each simulation's calls by the test-cell weights of
    its own scope, as NP3 does; the per-call-set weightings are reported.

    A stress that changes nothing: the per-call-set weightings put the
    stressed precision at .857 (below target_L .90) and would fail the set;
    unweighted and on the test cells it is .984, as NP3's test-cell rows.
    """
    assert (gp.NP3_UNWEIGHTED, *TEST_CELL_SCHEMES) == gp.NP6_SCORED_SCHEMES
    cells = _wrong_type_cells()
    tested, verdicts = _np6_tables(cells, cells, [100], composition=COMPOSITION)
    rows = verdicts[verdicts["class"] == "X"]
    scored = dict(zip(rows["scheme"], rows["scored"].astype(bool), strict=True))
    assert scored == {
        gp.NP3_UNWEIGHTED: True,
        gp.NP3_NATURAL_TEST_CELLS: True,
        gp.NP3_CLASS_BALANCED_TEST_CELLS: True,
        gp.NP3_NATURAL: False,
        gp.NP3_CLASS_BALANCED: False,
    }
    for scheme in TEST_CELL_SCHEMES:
        row = rows[rows["scheme"] == scheme].iloc[0]
        assert row["precision_base"] == pytest.approx(300 / 305)
        assert row["precision_stress"] == pytest.approx(300 / 305)
        assert row["passed"]
    for scheme in CALL_SET_SCHEMES:
        row = rows[rows["scheme"] == scheme].iloc[0]
        assert row["precision_stress"] == pytest.approx(0.857, abs=1e-3)
        assert not row["passed"]
    assert gp.np6_class_verdicts(
        verdicts, tested, stresses=[STRESS], settings=_np6()
    ) == {("broad", "X"): True, ("broad", "Y"): None}
    # NP6's base test-cell rows are NP3's on the same (unpooled) set.
    _, _, np3 = _np3_rows(cells, [100], composition=COMPOSITION)
    for scheme in TEST_CELL_SCHEMES:
        row = rows[rows["scheme"] == scheme].iloc[0]
        assert row["precision_base"] == pytest.approx(_scheme(np3, scheme)["precision"])
        assert row["kish_n_base"] == pytest.approx(_scheme(np3, scheme)["kish_n"])
    with pytest.raises(ValueError, match="composition"):
        _np6_tables(cells, cells, [100], composition=None)  # type: ignore[arg-type]


def test_np6_pools_a_thin_set_on_the_test_cells_of_the_union_scope() -> None:
    """A thin stressed set pooled with the next deeper set is weighted on the
    test cells of both scopes, each cell once (the union of the scopes).
    """
    shallow = _calls(30, [("A", "X", 300, 300, 0.95)])
    deep = _calls(100, [("A", "X", 300, 300, 0.95)], prefix="deep")
    base = pd.concat([shallow, deep], ignore_index=True)
    stressed = base.copy()
    # The stress leaves 150 confident calls at 30 (bp below the threshold).
    stressed.loc[stressed.index[:150], "bp"] = 0.5
    tested, verdicts = _np6_tables(
        base, stressed, [30, 100], composition={"A": 1.0}, np6=_np6()
    )
    rows = verdicts[verdicts["scheme"] == gp.NP3_NATURAL_TEST_CELLS]
    thin = rows[rows["tested_set"] == "30"].iloc[0]
    assert thin["set"] == "30+100" and not bool(thin["below_min_confident_n"])
    assert int(thin["n_stress"]) == 150 + 300 and int(thin["n_base"]) == 600
    assert thin["precision_stress"] == pytest.approx(1.0)
    assert set(tested) == {("broad", "X")}


# --------------------------------------------------------------------------
# R6: NP5 compares a bin where the base and the replicate each hold 50 cells


def test_np5_agreement_compares_a_bin_where_both_runs_hold_50_test_cells() -> None:
    thin = {depth: 100 for depth in GRID_NP5} | {250: 49}
    base = _np5_decisions((60, 120, 250), n_test=thin)
    # The replicate holds 50 test cells at 250, the base 49: the base's
    # status there is no test of 250, so the bin is not compared.
    rows = _agreement(
        base, {("D1", 0): _np5_decisions((60, 120), n_test=thin | {250: 50})}
    )
    row = rows[("D1", 0)]
    assert row["passed"] and row["flipped_depths"] == ""
    assert int(row["n_compared"]) == 5
    # The either-side reading (registered before §23.21) is reported.
    assert row["flipped_depths_union"] == "250" and not row["passed_union"]
    assert int(row["n_compared_union"]) == 6
    # The base holds 50 and the replicate 49: not compared either.
    full = _np5_decisions((60, 120, 250), n_test=thin | {250: 50})
    rows = _agreement(full, {("D1", 0): _np5_decisions((60, 120), n_test=thin)})
    assert rows[("D1", 0)]["passed"] and rows[("D1", 0)]["flipped_depths"] == ""
    # Both hold 50: compared, and the flip away from the boundary fails.
    rows = _agreement(
        full, {("D1", 0): _np5_decisions((60, 120), n_test=thin | {250: 50})}
    )
    assert not rows[("D1", 0)]["passed"]
    assert rows[("D1", 0)]["flipped_depths"] == "250"


# --------------------------------------------------------------------------
# R3 (c): NP5's t* part as a consequence check


def _consequence_cells() -> pd.DataFrame:
    """Broad X calls at 60 counts: 200 at bp .95 (right), 100 at .80 (80
    right) and 100 at .72 (50 right).
    """
    bp = np.concatenate([np.full(200, 0.95), np.full(100, 0.80), np.full(100, 0.72)])
    correct = np.concatenate(
        [np.ones(200, dtype=bool), np.arange(100) < 80, np.arange(100) < 50]
    )
    return bin_cells(bp, correct, depth=60)


def _consequence(thresholds: pd.DataFrame) -> pd.Series:
    cells = _consequence_cells()
    tested = res.gate_p_tested_sets(cells, _decisions([60]), regime="provisional")
    table = gp.np5_tstar_consequence(
        {("D1", 0): cells},
        tested,
        thresholds,
        gp.level_targets(AnnotationThresholds(), ["broad"]),
        default_group=None,
    )
    assert list(table.columns) == list(gp.NP5_CONSEQUENCE_COLUMNS)
    assert len(table) == 1
    return table.iloc[0]


def test_np5_tstar_consequence_applies_each_replicate_t_star_to_the_base_calls() -> (
    None
):
    """R3 (c): each replicate's own t*, applied to the base's pooled calls in
    the set's scope, gives point precision >= target_L - 0.02 (.88).

    The six thresholds range over .20 (the old spread part fails) but none
    lets the .72 band in: precision 280 / 300 = .933 at .75 and .80, 1.0 at
    .85 and above.
    """
    thresholds = _thresholds(_values(0.75, 0.90, 0.95, 0.80, 0.85, 0.75), label="60")
    spread = gp.np5_tstar_spread(thresholds, _np5())
    assert not bool(spread["passed"].iloc[0])
    row = _consequence(thresholds)
    assert row["passed"] and row["evaluable"]
    assert row["threshold_from"] == gp.NP5_TSTAR_FROM_MEMBER
    assert row["limit"] == pytest.approx(0.88)
    assert row["min_precision"] == pytest.approx(280 / 300)
    assert int(row["n_called"]) == 400 and int(row["n_scored"]) == 6
    assert row["failed"] == "" and row["missing"] == ""
    # A t* below the error band lets in .72's calls: 330 / 400 = .825 fails.
    row = _consequence(
        _thresholds(_values(0.71, 0.90, 0.95, 0.80, 0.85, 0.75), label="60")
    )
    assert not row["passed"] and row["failed"] == "D1/0:0.710->0.8250"
    assert row["min_precision"] == pytest.approx(0.825)


def test_np5_tstar_consequence_fails_a_fitted_replicate_without_a_threshold() -> None:
    values = _values(0.80, 0.90, None, 0.85, 0.80, 0.85)
    row = _consequence(_thresholds(values, label="60"))
    assert not row["passed"] and row["missing"] == "D2/0"
    # Without a fit it is left out; the others pass.
    row = _consequence(_thresholds(values, label="60", unfitted=[("D2", 0)]))
    assert row["passed"] and row["unfitted"] == "D2/0"
    assert int(row["n_scored"]) == 5 and int(row["n_fitted"]) == 5
    # No replicate scored: nothing to check (reported as not evaluable).
    lonely = dict.fromkeys(KEYS_NP5)
    row = _consequence(_thresholds(lonely, label="60", unfitted=KEYS_NP5))
    assert row["passed"] and not row["evaluable"] and math.isnan(row["min_precision"])
    with pytest.raises(ValueError, match="t\\* rows"):
        _consequence(_thresholds(_values(*[0.8] * 6), label="30"))


def test_np5_class_table_scores_the_consequence_check() -> None:
    """The class table reads the t* part from the consequence table."""
    cells = _consequence_cells()
    tested = res.gate_p_tested_sets(cells, _decisions([60]), regime="provisional")
    base = _np5_decisions((60, 120, 250))
    agreement = gp.np5_decision_agreement(base, {("D1", 0): base}, _np5())
    extrapolated = gp.np5_extrapolated_share(
        base, {}, GRID_NP5, _np5(), default_depth=70.0
    )
    thresholds = _thresholds(_values(0.75, 0.90, 0.95, 0.80, 0.85, 0.75), label="60")
    consequence = gp.np5_tstar_consequence(
        {("D1", 0): cells},
        tested,
        thresholds,
        gp.level_targets(AnnotationThresholds(), ["broad"]),
        default_group=None,
    )
    table = gp.np5_class_table(agreement, consequence, extrapolated, tested)
    assert table.iloc[0]["tstar_passed"] is True and table.iloc[0]["passed"] is True


def _ensemble_rows(values: Mapping[int, tuple[float | None, int]]) -> pd.DataFrame:
    """Re-derived ensemble decisions of broad X: per depth (threshold, n_fit)."""
    records = []
    for regime in ("provisional", "trust"):
        for depth in GRID_NP5:
            threshold, n_fit = values.get(depth, (None, 0))
            records.append(
                {
                    "regime": regime,
                    "level": "broad",
                    "class": "X",
                    "depth": depth,
                    "status": res.STATUS_EMITTED,
                    "n_test": 100,
                    "threshold": 0.5 if regime == "trust" else threshold,
                    "t_star": threshold,
                    "threshold_source": None
                    if threshold is None
                    else res.THRESHOLD_SOURCE_LOCAL,
                    "n_called": 300,
                    "n_fit": n_fit,
                    "target": 0.95,
                }
            )
    return pd.DataFrame.from_records(records)


def test_np5_ensemble_set_thresholds_read_the_rederived_ensemble_t_star() -> None:
    """R3 (c), version 7: each replicate's thresholds are the ensemble's
    pooled t* re-fitted on that replicate (``np5_rederive_ensemble``), read
    at every bin of each tested set (pre-registration §23.22: a ">= D_P"
    set's deeper bins may take the replicate's deep pool's t*); a fit needs
    ``min_cells_per_bin`` fit cells, exactly that many included.
    """
    tested: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
        ("broad", "X"): [_set((120, 250)), _set((60,))],
        ("broad", "Y"): None,
    }
    minimum = settings().min_cells_per_bin
    rederived = {
        ("D1", 0): _ensemble_rows(
            {60: (0.85, 100), 120: (0.80, 100), 250: (0.55, 100)}
        ),
        # 120: fitted (exactly the minimum), no t* (missing); 250 and 60:
        # one fit cell short (unfitted).
        ("D2", 1): _ensemble_rows(
            {60: (None, minimum - 1), 120: (None, minimum), 250: (None, minimum - 1)}
        ),
    }
    table = gp.np5_ensemble_set_thresholds(rederived, tested, settings())
    assert list(table.columns) == list(gp.NP5_ENSEMBLE_THRESHOLD_COLUMNS)
    view = {
        (row["set"], row["group"], int(row["seed"]), int(row["depth"])): row
        for _, row in table.iterrows()
    }
    assert set(view) == {
        (label, group, seed, depth)
        for label, depths in ((">=120", (120, 250)), ("60", (60,)))
        for group, seed in (("D1", 0), ("D2", 1))
        for depth in depths
    }
    assert view[(">=120", "D1", 0, 120)]["threshold"] == pytest.approx(0.80)
    assert view[(">=120", "D1", 0, 250)]["threshold"] == pytest.approx(0.55)
    assert view[("60", "D1", 0, 60)]["threshold"] == pytest.approx(0.85)
    assert set(table[table["set"] == ">=120"]["set_min_depth"]) == {120}
    missing = view[(">=120", "D2", 1, 120)]
    assert missing["fitted"] and math.isnan(missing["threshold"])
    for key in ((">=120", "D2", 1, 250), ("60", "D2", 1, 60)):
        assert not view[key]["fitted"] and math.isnan(view[key]["threshold"])
    with pytest.raises(ValueError, match="no replicates"):
        gp.np5_ensemble_set_thresholds({}, tested, settings())


POOLED_X: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
    ("broad", "X"): [_set((120, 250))]
}


def _pooled_cells() -> pd.DataFrame:
    """Base calls of broad X's ">=120" set: 200 at 120 (bp .95, right) and
    200 at 250 (100 at bp .95, right; 100 at bp .60, 40 right).
    """
    at_120 = bin_cells(np.full(200, 0.95), np.ones(200, dtype=bool), depth=120)
    at_250 = bin_cells(
        np.concatenate([np.full(100, 0.95), np.full(100, 0.60)]),
        np.concatenate([np.ones(100, dtype=bool), np.arange(100) < 40]),
        depth=250,
    )
    return pd.concat([at_120, at_250], ignore_index=True)


def _ensemble_consequence(
    thresholds: pd.DataFrame, *, threshold_from: str = gp.NP5_TSTAR_FROM_ENSEMBLE
) -> pd.DataFrame:
    return gp.np5_tstar_consequence(
        {("D1", 0): _pooled_cells()},
        POOLED_X,
        thresholds,
        gp.level_targets(AnnotationThresholds(), ["broad"]),
        default_group=None,
        threshold_from=threshold_from,
    )


def _pooled_consequence(rows: pd.DataFrame) -> pd.Series:
    """R3 (c), version 7, of one replicate's re-derived ensemble ``rows``."""
    thresholds = gp.np5_ensemble_set_thresholds({("D1", 0): rows}, POOLED_X, settings())
    table = _ensemble_consequence(thresholds)
    assert list(table.columns) == list(gp.NP5_CONSEQUENCE_COLUMNS)
    assert len(table) == 1
    return table.iloc[0]


def test_np5_ensemble_consequence_applies_each_bin_s_threshold() -> None:
    """R3 (c), version 7 (pre-registration §23.22): a replicate's re-derived
    ensemble applies .85 at 120 but its deep pool's .55 at 250, so its
    emission of the ">=120" set's base calls is 200 + 100 + 40 right of 400
    = .85 < .88 (target_L .90 - .02): the set fails. Read at the shallowest
    bin only (.85 for every call), the precision would be 1.0.
    """
    row = _pooled_consequence(_ensemble_rows({120: (0.85, 100), 250: (0.55, 100)}))
    assert row["threshold_from"] == gp.NP5_TSTAR_FROM_ENSEMBLE
    assert int(row["n_called"]) == 400 and int(row["n_scored"]) == 1
    assert row["min_precision"] == pytest.approx(0.85)
    assert not row["passed"]
    assert row["failed"] == "D1/0:120=0.850,250=0.550->0.8500"
    # The deep pool at .65 keeps the .60 band out: 300 / 300.
    row = _pooled_consequence(_ensemble_rows({120: (0.85, 100), 250: (0.65, 100)}))
    assert row["passed"] and row["min_precision"] == pytest.approx(1.0)
    # A fitted bin of the set without a threshold fails the set (it would
    # emit none of that bin's calls), at any bin.
    row = _pooled_consequence(_ensemble_rows({120: (0.85, 100), 250: (None, 100)}))
    assert not row["passed"] and row["missing"] == "D1/0"
    # An unfitted bin is left out: its calls are not emitted (200 / 200).
    row = _pooled_consequence(_ensemble_rows({120: (0.85, 100), 250: (None, 10)}))
    assert row["passed"] and row["missing"] == "" and row["unfitted"] == ""
    assert row["min_precision"] == pytest.approx(1.0)
    # No bin fitted: the replicate is left out.
    row = _pooled_consequence(_ensemble_rows({120: (None, 10), 250: (None, 10)}))
    assert row["passed"] and row["unfitted"] == "D1/0" and not row["evaluable"]


def test_np5_consequence_refuses_thresholds_of_the_other_source() -> None:
    """A member's per-set t* is no per-bin ensemble table, and back; every
    bin of a set needs exactly one row per replicate.
    """
    rows = _ensemble_rows({120: (0.85, 100), 250: (0.65, 100)})
    per_bin = gp.np5_ensemble_set_thresholds({("D1", 0): rows}, POOLED_X, settings())
    member = _thresholds({("D1", 0): 0.55}, label=">=120")
    with pytest.raises(ValueError, match="per bin"):
        _ensemble_consequence(member)
    with pytest.raises(ValueError, match="per bin"):
        _ensemble_consequence(per_bin, threshold_from=gp.NP5_TSTAR_FROM_MEMBER)
    with pytest.raises(ValueError, match="threshold_from"):
        _ensemble_consequence(member, threshold_from="pool")
    with pytest.raises(ValueError, match="bin 250"):
        _ensemble_consequence(per_bin[per_bin["depth"] != 250])
    with pytest.raises(ValueError, match="more than once"):
        _ensemble_consequence(pd.concat([per_bin, per_bin.iloc[:1]], ignore_index=True))
    # The member form applies one t* per (set, replicate) to every bin.
    row = _ensemble_consequence(member, threshold_from=gp.NP5_TSTAR_FROM_MEMBER)
    assert row.iloc[0]["failed"] == "D1/0:0.550->0.8500"


# --------------------------------------------------------------------------
# R2: one depth rule for NP3-NP7

WALK_GRID = (10, 15, 20, 30, 60, 100)
WALK_SETS = [_set((60, 100)), _set((60,)), _set((30,)), _set((20,)), _set((15,))]
WALK_LABELS = [gp.tested_set_label(item) for item in WALK_SETS]
WALK_STRESSES = ("spill", "lognormal")


def _walk_tables(
    failing: Collection[tuple[str, str]] = (),
    *,
    class_parts: Collection[str] = (),
    flips: Mapping[tuple[str, int], str] | None = None,
    boundary: str = "",
) -> gp.CriterionTables:
    """Every criterion table of broad X (and of the unevaluable Y).

    ``failing`` holds (criterion, set label) pairs that fail; ``class_parts``
    the criteria whose class-level part fails (NP4 seed, NP5 extrapolated
    share, NP7 excluded share); ``flips`` the NP5 flipped depths per
    replicate (``;`` joined), ``boundary`` the base's boundary bins.
    """

    def ok(criterion: str, label: str) -> bool:
        return (criterion, label) not in failing

    np3 = [
        {
            "level": "broad",
            "class": "X",
            "set": label,
            "scheme": scheme,
            "scored": scheme in gp.NP3_SCORED_SCHEMES,
            # The reported-only weightings fail: the walk must not read them.
            "passed": ok("NP3", label) if scheme in gp.NP3_SCORED_SCHEMES else False,
        }
        for label in WALK_LABELS
        for scheme in (*gp.NP3_SCORED_SCHEMES, gp.NP3_NATURAL)
    ]
    np6 = [
        {
            "level": "broad",
            "class": "X",
            "stress": stress,
            "tested_set": label,
            "scheme": scheme,
            "scored": scheme in gp.NP6_SCORED_SCHEMES,
            "passed": ok("NP6", label) if scheme in gp.NP6_SCORED_SCHEMES else False,
        }
        for stress in WALK_STRESSES
        for label in WALK_LABELS
        for scheme in (*gp.NP6_SCORED_SCHEMES, gp.NP3_CLASS_BALANCED)
    ]

    def per_set(criterion: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "level": "broad",
                "class": "X",
                "set": WALK_LABELS,
                "passed": [ok(criterion, label) for label in WALK_LABELS],
            }
        )

    replicates = flips or {("D1", 0): "", ("D2", 1): ""}
    agreement = pd.DataFrame(
        {
            "level": "broad",
            "class": "X",
            "group": [group for group, _ in replicates],
            "seed": [seed for _, seed in replicates],
            "boundary_depths": boundary,
            "flipped_depths": list(replicates.values()),
            "passed": True,
        }
    )
    return gp.CriterionTables(
        np3_verdicts=pd.DataFrame.from_records(np3),
        np4_sets=per_set("NP4"),
        np4_seed=pd.DataFrame(
            {"level": ["broad"], "passed": ["NP4" not in class_parts]}
        ),
        np5_agreement=agreement,
        np5_tstar=per_set("NP5"),
        np5_extrapolated=pd.DataFrame(
            {"level": ["broad"], "class": ["X"], "passed": ["NP5" not in class_parts]}
        ),
        np6_verdicts=pd.DataFrame.from_records(np6),
        np6_stresses=WALK_STRESSES,
        np7_wrong_node=per_set("NP7"),
        np7_excluded=pd.DataFrame(
            {"level": ["broad"], "passed": ["NP7" not in class_parts]}
        ),
    )


TESTED_WALK: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
    ("broad", "X"): WALK_SETS,
    ("broad", "Y"): None,
}


def _walk(
    tables: gp.CriterionTables,
) -> tuple[dict[str, bool | None], dict[str, object]]:
    verdicts, table = gp.gate_p_depth_walk(
        tables, TESTED_WALK, WALK_GRID, np6_settings=_np6()
    )
    assert list(table.columns) == list(gp.DEPTH_WALK_COLUMNS)
    assert set(verdicts) == set(gp.GATE_P_CLASS_CRITERIA)
    for criterion in gp.GATE_P_CLASS_CRITERIA:
        assert verdicts[criterion][("broad", "Y")] is None
    records = {str(row["class"]): row for row in table.to_dict("records")}
    assert records["Y"]["passed"] is None
    assert records["Y"]["reason"] == gp.DEPTH_REASON_NOT_EVALUABLE
    return (
        {criterion: verdicts[criterion][("broad", "X")] for criterion in verdicts},
        records["X"],
    )


def test_depth_walk_reads_every_criterion_and_stops_at_an_untested_bin() -> None:
    verdicts, record = _walk(_walk_tables())
    assert set(verdicts.values()) == {True}
    assert record["passed"] is True and record["reason"] == ""
    assert record["tested_max_depth"] == 60 and record["validated_min_depth"] == 15
    assert (record["stop_depth"], record["stop_reason"]) == (10, "untested")
    assert record["deep_sets"] == ">=60;60" and record["failed_sets"] == ""


@pytest.mark.parametrize("criterion", gp.GATE_P_CLASS_CRITERIA)
def test_a_failure_below_d_p_raises_the_floor_and_fails_no_criterion(
    criterion: str,
) -> None:
    """R2: a set below D_P that fails any of NP3-NP7 only raises the floor
    (before §23.21, NP4-NP7 failed the class at any tested set).
    """
    verdicts, record = _walk(_walk_tables({(criterion, "20")}))
    assert set(verdicts.values()) == {True}
    assert record["passed"] is True and record["validated_min_depth"] == 30
    assert (record["stop_depth"], record["stop_reason"]) == (20, criterion)
    assert record["failed_sets"] == f"{criterion}:20"


@pytest.mark.parametrize("label", [">=60", "60"])
@pytest.mark.parametrize("criterion", gp.GATE_P_CLASS_CRITERIA)
def test_a_failure_in_the_d_p_group_fails_the_class(criterion: str, label: str) -> None:
    verdicts, record = _walk(_walk_tables({(criterion, label), ("NP7", "15")}))
    assert verdicts == {name: name != criterion for name in gp.GATE_P_CLASS_CRITERIA}
    assert record["passed"] is False and record["validated_min_depth"] is None
    assert record["reason"] == gp.DEPTH_REASON_DEEP_GROUP_FAILED
    assert record["deep_failures"] == f"{criterion}:{label}"
    assert record["tested_max_depth"] == 60


@pytest.mark.parametrize(
    ("criterion", "part"),
    [
        ("NP4", "seed_criterion"),
        ("NP5", "extrapolated_share"),
        ("NP7", "excluded_share"),
    ],
)
def test_a_class_level_part_fails_the_class(criterion: str, part: str) -> None:
    verdicts, record = _walk(_walk_tables(class_parts={criterion}))
    assert verdicts == {name: name != criterion for name in gp.GATE_P_CLASS_CRITERIA}
    assert record["passed"] is False and record["validated_min_depth"] is None
    assert record["reason"] == gp.DEPTH_REASON_CLASS_PART_FAILED
    assert record["class_failures"] == f"{criterion}:{part}"


def test_np5_agreement_counts_the_flips_at_or_above_the_floor() -> None:
    """R2 for NP5's agreement: at floor d only flips at bins >= d count; the
    boundary exemption is unchanged (the base's boundaries on the whole grid).
    """
    # A flip at 15 only raises the floor to 20.
    verdicts, record = _walk(_walk_tables(flips={("D1", 0): "15", ("D2", 1): ""}))
    assert verdicts["NP5"] is True and record["validated_min_depth"] == 20
    assert record["stop_reason"] == "NP5"
    assert record["failed_sets"] == "NP5:agreement D1/0@15"
    # A flip at 100 (>= D_P, not at a boundary) fails the class.
    verdicts, record = _walk(_walk_tables(flips={("D1", 0): "15;100"}))
    assert verdicts["NP5"] is False and record["passed"] is False
    assert record["deep_failures"] == "NP5:agreement D1/0@100"
    # One flip at a boundary bin is allowed in the D_P group, but with a
    # second flip at 30 the walk stops there (two flips >= 30).
    verdicts, record = _walk(_walk_tables(flips={("D1", 0): "30;60"}, boundary="30;60"))
    assert verdicts["NP5"] is True and record["validated_min_depth"] == 60
    assert (record["stop_depth"], record["stop_reason"]) == (30, "NP5")


def test_depth_walk_refuses_tables_without_a_tested_set_s_row() -> None:
    tables = _walk_tables()
    lacking = dataclasses.replace(tables, np4_sets=tables.np4_sets.iloc[1:])
    with pytest.raises(ValueError, match="NP4"):
        gp.gate_p_depth_walk(lacking, TESTED_WALK, WALK_GRID, np6_settings=_np6())
    one_stress = dataclasses.replace(tables, np6_stresses=(*WALK_STRESSES, "xplatform"))
    with pytest.raises(ValueError, match="NP6"):
        gp.gate_p_depth_walk(one_stress, TESTED_WALK, WALK_GRID, np6_settings=_np6())
    no_agreement = dataclasses.replace(
        tables, np5_agreement=tables.np5_agreement.iloc[:0]
    )
    with pytest.raises(ValueError, match="agreement"):
        gp.gate_p_depth_walk(no_agreement, TESTED_WALK, WALK_GRID, np6_settings=_np6())
    mouse = dataclasses.replace(tables, np7_excluded=None)
    verdicts, _ = gp.gate_p_depth_walk(
        mouse, TESTED_WALK, WALK_GRID, np6_settings=_np6()
    )
    assert verdicts["NP7"][("broad", "X")] is True


def test_depth_walk_feeds_the_class_records() -> None:
    """The walk's table is the per-member depth input of the records."""
    _, table = gp.gate_p_depth_walk(
        _walk_tables({("NP6", "20")}), TESTED_WALK, WALK_GRID, np6_settings=_np6()
    )
    index = gp._depth_index(table)
    assert index[("broad", "X")] == (True, 30, 60)
    assert index[("broad", "Y")] == (None, None, None)


# --------------------------------------------------------------------------
# R5: the dry run reports an H18 class that is not evaluable


def test_dry_run_reports_an_h18_class_that_is_not_evaluable() -> None:
    """R5 (replaces CK1 (a)): an H18 class whose record is ``not_evaluable``
    (fewer than n_min confident calls) is reported, not failed.
    """
    result = _assemble()
    h18 = {"supercluster": ["Astro", "Vascular"]}
    verdict = gp.dry_run_verdict(result, h18_classes=h18)
    assert verdict["passes"] and verdict["failing"] == []
    assert verdict["reported_not_evaluable"] == [["supercluster", "Vascular"]]
    rows = {(row["level"], row["class"]): row for row in verdict["expected"]}
    vascular = rows[("supercluster", "Vascular")]
    assert vascular["status"] == gp.RECORD_NOT_EVALUABLE
    assert vascular["passed"] is None and vascular["reported"] is True
    assert rows[("supercluster", "Astro")]["reported"] is False
    assert list(vascular) == list(gp.DRY_RUN_COLUMNS)
    # A failing H18 class and an H18 class without a record still fail.
    failing = _assemble(overrides={("NP4", "R1@0", ("supercluster", "Astro")): False})
    assert not gp.dry_run_verdict(failing, h18_classes=h18)["passes"]
    missing = gp.dry_run_verdict(result, h18_classes={"supercluster": ["Fibro"]})
    assert not missing["passes"] and missing["failing"][0]["status"] == "no_record"
    # A C_P class not evaluable at broad still fails the dry run.
    sets = _class_sets(
        counts={"Exc": 3000, "Inh": 1500, "Astro": 1200, "Oligo": 1000, "OPC": 800}
    )
    broad = gp.dry_run_verdict(_assemble(class_sets=sets), h18_classes={})
    assert not broad["passes"]
    assert ("broad", "OPC") in {
        (row["level"], row["class"]) for row in broad["failing"]
    }


# --------------------------------------------------------------------------
# The report names the scored readings and no open one


def test_report_names_the_scored_readings_and_the_rulings() -> None:
    report = gp.gate_p_report(_assemble())
    assert "open_readings" not in report
    assert report["readings_ruled"] == list(gp.GATE_P_READINGS_RULED)
    assert report["scored_readings"] == list(gp.GATE_P_SCORED_READINGS)
    text = gp.gate_p_report_text(report)
    assert "open until the user rules" not in text
    assert "Readings ruled: pre-registration §23.19, §23.21" in text
    for name in ("natural_test_cells", "R1", "R2", "R3 (c)", "R5", "R6", "R7"):
        assert name in text, name
