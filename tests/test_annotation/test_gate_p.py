"""Tests for gate-P scoring (M13; plan §14 new panel family: NP3, NP4, NP5, NP7)."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import gate_p as gp
from merxen.annotation import mapmycells_engine as mmc
from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    AnnotationResolvabilityConfig,
    AnnotationThresholds,
)

from .test_resolvability import BROAD, bin_cells, settings, tracked_cells
from .test_resolvability_v7_decisions import member_rows, pattern


def _decisions_at_10() -> pd.DataFrame:
    """Frozen decisions: the provisional regime emits broad X at 10 counts (0.70)."""
    return res.decide(
        bin_cells(np.full(400, 0.95), np.ones(400, dtype=bool), depth=10),
        [BROAD],
        [10],
        settings(),
    )


# --------------------------------------------------------------------------
# The frozen confident-call mask (shared by gate_p_tested_sets and NP3-NP7)


def test_frozen_confident_mask_needs_a_call_an_emitted_bin_and_the_threshold() -> None:
    lookup = res.emission_lookup(_decisions_at_10(), "provisional")
    assert lookup[("broad", "X", 10)][:2] == (res.STATUS_EMITTED, 0.70)
    lookup[("broad", "W", 10)] = (res.STATUS_NOT_RESOLVABLE, 0.70, False)
    lookup[("broad", "V", 10)] = (res.STATUS_EMITTED, None, False)
    frame = pd.DataFrame(
        {
            "level": "broad",
            "parent": ["X", "X", "X", "X", None, "Y", "W", "V", "X"],
            "depth": [10, 10, 10, 10, 10, 10, 10, 10, 30],
            "bp": [0.95, 0.70 - 5e-10, 0.69, np.nan, 0.99, 0.99, 0.99, 0.99, 0.99],
        }
    )
    mask = res.frozen_confident_mask(frame, lookup)
    assert mask.dtype == bool
    # At the threshold within 1e-9 counts; below it, a NaN bp, no call, a
    # class without a bin, a withheld bin, a bin without a threshold and a
    # depth the decisions never saw do not.
    assert mask.tolist() == [True, True] + [False] * 7
    assert res.frozen_confident_mask(frame.iloc[:0], lookup).shape == (0,)


# --------------------------------------------------------------------------
# NP4: donor / draw stability at the frozen thresholds (§14 NP4)

GROUPS = ("D1", "D2", "D3")
SEEDS = (0, 1)
SINGLE_10 = res.GatePTestedSet("broad", "X", (10,), False, 0, math.nan, math.nan)
POOLED_30 = res.GatePTestedSet("broad", "X", (30, 100), True, 0, math.nan, math.nan)


def _np4() -> gp.Np4Settings:
    return gp.Np4Settings.from_config(AnnotationResolvabilityConfig())


def _targets() -> dict[str, float]:
    return gp.level_targets(AnnotationThresholds(), ["broad"])


def _replicate(
    group: str, seed: int, n_correct: int, n: int = 400, *, cls: str = "X"
) -> pd.DataFrame:
    """One replicate's broad calls at 10 counts, bp 0.95 (confident at 0.70)."""
    frame = bin_cells(np.full(n, 0.95), np.arange(n) < n_correct, depth=10, cls=cls)
    frame["cell_id"] = f"{group}_" + frame["cell_id"].astype(str)
    frame["sim_id"] = f"{group}_" + frame["sim_id"].astype(str)
    frame["seed"] = seed
    return frame


def _grid(
    correct: Mapping[str, int], n: int = 400
) -> dict[gp.ReplicateKey, pd.DataFrame]:
    return {
        (group, seed): _replicate(group, seed, k, n)
        for group, k in correct.items()
        for seed in SEEDS
    }


def _tested_from_seed0(
    replicates: Mapping[gp.ReplicateKey, pd.DataFrame],
    decisions: pd.DataFrame,
    default_group: str | None = None,
) -> dict[tuple[str, str], list[res.GatePTestedSet] | None]:
    """The pooled seed-0 tested sets (§14: pooled held-out calls)."""
    scored = gp.held_out_replicates(replicates, default_group=default_group)
    seed0 = [table for (_, seed), table in sorted(scored.items()) if seed == 0]
    return res.gate_p_tested_sets(
        pd.concat(seed0, ignore_index=True), decisions, regime="provisional"
    )


def _seed_passed(*levels: str) -> pd.DataFrame:
    """NP4 seed-criterion rows that pass at each level (``np4_class_verdicts``)."""
    return pd.DataFrame({"level": list(levels or ("broad",)), "passed": True})


def _verdict(
    replicates: Mapping[gp.ReplicateKey, pd.DataFrame],
    default_group: str | None = None,
) -> tuple[pd.Series, dict[tuple[str, str], bool | None]]:
    decisions = _decisions_at_10()
    tested = _tested_from_seed0(replicates, decisions, default_group)
    stats = gp.replicate_set_stats(
        replicates, decisions, tested, default_group=default_group
    )
    verdicts = gp.np4_set_verdicts(stats, _targets(), _np4())
    assert len(verdicts) == 1
    return verdicts.iloc[0], gp.np4_class_verdicts(verdicts, _seed_passed(), tested)


def test_planted_donor_shift_fails_np4_while_binomial_noise_passes() -> None:
    """§12 M13: a planted donor shift fails NP4; binomial noise alone passes."""
    noise = _grid(dict(zip(GROUPS, (388, 384, 380), strict=True)))
    tested = _tested_from_seed0(noise, _decisions_at_10())
    (item,) = tested[("broad", "X")] or []
    assert item.depths == (10,) and not item.pooled and item.n_confident == 1200
    stats = gp.replicate_set_stats(
        noise, _decisions_at_10(), tested, default_group=None
    )
    assert len(stats) == 6 and set(stats["n_confident"]) == {400}
    assert stats["set"].unique().tolist() == ["10"]
    row, classes = _verdict(noise)
    limit = 3.5 * math.sqrt(0.96 * 0.04 / 400)
    assert limit == pytest.approx(0.0343, abs=1e-4)
    assert row["donor_range"] == pytest.approx(0.02)
    assert row["limit"] == pytest.approx(limit)
    assert row["floor_ok"] and row["range_evaluable"] and row["range_ok"]
    assert row["passed"] and classes == {("broad", "X"): True}
    # Precision .99 / .96 / .93: every replicate clears the floor (.90), but
    # the donor range .06 exceeds max(.03, 3.5 SE) = .0343.
    row, classes = _verdict(_grid(dict(zip(GROUPS, (396, 384, 372), strict=True))))
    assert row["donor_range"] == pytest.approx(0.06)
    assert row["limit"] == pytest.approx(limit)
    assert row["floor_ok"] and not row["range_ok"] and not row["passed"]
    assert classes == {("broad", "X"): False}


def test_np4_floor_applies_only_to_replicates_with_100_calls() -> None:
    replicates = _grid(dict.fromkeys(GROUPS, 380))
    replicates[("D2", 0)] = _replicate("D2", 0, 132, n=150)  # precision .88
    row, classes = _verdict(replicates)
    assert not row["floor_ok"] and row["floor_failures"] == "D2/0"
    assert row["range_evaluable"] and not row["passed"]
    assert classes == {("broad", "X"): False}
    # With 99 calls the replicate is below the floor's minimum: the floor
    # skips it and the range rule cannot apply.
    replicates[("D2", 0)] = _replicate("D2", 0, 87, n=99)
    row, classes = _verdict(replicates)
    assert row["floor_ok"] and row["floor_failures"] == ""
    assert row["n_replicates"] == 6 and row["n_evaluated"] == 5
    assert not row["range_evaluable"] and not row["vacuous"]
    assert math.isnan(row["donor_range"]) and math.isnan(row["limit"])
    assert row["range_ok"] and row["passed"]
    assert classes == {("broad", "X"): True}


def _rows(
    cell_ids: Sequence[str],
    depth: int,
    *,
    parent: str = "X",
    bp: float = 0.95,
    correct: bool = True,
) -> pd.DataFrame:
    frame = bin_cells(
        np.full(len(cell_ids), bp), np.full(len(cell_ids), correct), depth=depth
    )
    frame["cell_id"] = list(cell_ids)
    frame["sim_id"] = [f"{cell}|D{depth}" for cell in cell_ids]
    frame["parent"] = parent
    return frame


def test_replicate_sets_count_each_cell_once_at_its_deepest_bin() -> None:
    decisions = res.decide(
        pd.concat(
            [
                bin_cells(np.full(300, 0.95), np.ones(300, dtype=bool), depth=depth)
                for depth in (30, 100)
            ],
            ignore_index=True,
        ),
        [BROAD],
        [30, 100],
        settings(),
    )
    deep = [f"E{index}" for index in range(5)]
    table = pd.concat(
        [
            _rows(["A", "B", "C"], 30),
            _rows(["D"], 30, correct=False),
            _rows(["A", *deep], 100),
            # B's deepest row is a call of another class, C's is unconfident:
            # their confident X rows at 30 do not enter the >= 30 set.
            _rows(["B"], 100, parent="Y"),
            _rows(["C"], 100, bp=0.5),
        ],
        ignore_index=True,
    )
    tested = {
        ("broad", "X"): [
            POOLED_30,
            res.GatePTestedSet("broad", "X", (30,), False, 0, 0.0, 0.0),
        ]
    }
    stats = gp.replicate_set_stats(
        {("D1", 0): table}, decisions, tested, default_group=None
    )
    by_set = stats.set_index("set")
    # >= 30: A once (at 100), D (deepest at 30) and the five deep cells.
    assert by_set.loc[">=30", "n_confident"] == 7
    assert by_set.loc[">=30", "n_correct"] == 6
    assert by_set.loc[">=30", "set_min_depth"] == 30
    # The single bin 30 holds every confident X row at 30, whatever is deeper.
    assert by_set.loc["30", "n_confident"] == 4
    assert by_set.loc["30", "n_correct"] == 3
    # gate_p_tested_sets builds the same pooled set from the same rows.
    (pooled,) = (
        res.gate_p_tested_sets(
            table, decisions, min_confident_n=7, regime="provisional"
        )[("broad", "X")]
        or []
    )
    assert pooled.depths == (30, 100) and pooled.n_confident == 7
    # The mask is positional: it needs the replicate's rows in a fresh index.
    lookup = res.emission_lookup(decisions, "provisional")
    confident = res.frozen_confident_mask(table, lookup)
    with pytest.raises(ValueError, match="RangeIndex"):
        gp.tested_set_mask(table.iloc[::-1], confident[::-1], POOLED_30)
    with pytest.raises(ValueError, match="rows"):
        gp.tested_set_mask(table, confident[:-1], POOLED_30)


def _random_cells(rng: np.random.Generator, n_cells: int = 700) -> pd.DataFrame:
    """Held-out cells at depths (10, 30, 100), shared across their depths."""
    native = rng.choice(np.array([10, 30, 100]), size=n_cells, p=[0.5, 0.3, 0.2])
    frames = []
    for depth in (10, 30, 100):
        ids = [f"c{index}" for index in np.flatnonzero(native >= depth)]
        bp = np.round(rng.uniform(0.5, 1.0, len(ids)), 3)
        correct = rng.random(len(ids)) < np.where(bp >= 0.8, 0.995, 0.7)
        frame = _rows(ids, depth)
        frame["bp"] = bp
        frame["correct"] = correct
        frame["parent"] = np.where(rng.random(len(ids)) < 0.15, "Y", "X")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_single_replicate_reproduces_gate_p_tested_sets() -> None:
    kinds = {"pooled": 0, "single": 0}
    for rng_seed in range(5):
        cells = _random_cells(np.random.default_rng(rng_seed))
        decisions = res.decide(cells, [BROAD], [10, 30, 100], settings())
        tested = res.gate_p_tested_sets(
            cells, decisions, min_confident_n=100, regime="provisional"
        )
        stats = gp.replicate_set_stats(
            {("pooled", 0): cells}, decisions, tested, default_group=None
        )
        lookup = res.emission_lookup(decisions, "provisional")
        confident = res.frozen_confident_mask(cells, lookup)
        for (level, cls), items in tested.items():
            for item in items or []:
                kinds["pooled" if item.pooled else "single"] += 1
                # The public mask and the stats share one membership rule.
                mask = gp.tested_set_mask(cells, confident, item)
                assert int(mask.sum()) == item.n_confident
                row = stats[
                    (stats["level"] == level)
                    & (stats["class"] == cls)
                    & (stats["set"] == gp.tested_set_label(item))
                ]
                assert len(row) == 1
                assert int(row["n_confident"].iloc[0]) == item.n_confident
                assert abs(float(row["precision"].iloc[0]) - item.precision) <= 1e-12
        expected = sum(len(items or []) for items in tested.values())
        assert len(stats) == expected
    # The sweep exercises both kinds of tested set.
    assert kinds["pooled"] > 0 and kinds["single"] > 0


def test_zero_call_replicate_is_listed_and_blocks_the_range_rule() -> None:
    replicates = _grid({"D1": 380, "D2": 380})
    for seed in SEEDS:
        # D3 has calls, but only of another class: none in X's set.
        replicates[("D3", seed)] = _replicate("D3", seed, 380, cls="Y")
    decisions = _decisions_at_10()
    tested = _tested_from_seed0(replicates, decisions)
    assert tested[("broad", "Y")] is None
    stats = gp.replicate_set_stats(replicates, decisions, tested, default_group=None)
    empty = stats[stats["group"] == "D3"]
    assert len(empty) == 2 and set(empty["n_confident"]) == {0}
    assert empty["precision"].isna().all()
    verdicts = gp.np4_set_verdicts(stats, _targets(), _np4())
    row = verdicts.iloc[0]
    assert row["n_replicates"] == 6 and row["n_evaluated"] == 4
    assert not row["range_evaluable"] and row["floor_ok"] and row["passed"]
    assert gp.np4_class_verdicts(verdicts, _seed_passed(), tested) == {
        ("broad", "X"): True,
        ("broad", "Y"): None,
    }


def test_np4_vacuous_when_no_replicate_reaches_the_minimum() -> None:
    replicates = _grid(dict.fromkeys(GROUPS, 70), n=80)
    row, classes = _verdict(replicates)
    assert row["n_evaluated"] == 0 and row["vacuous"]
    assert row["floor_ok"] and not row["range_evaluable"] and row["passed"]
    assert classes == {("broad", "X"): True}


def test_np4_range_limit_is_floored_at_003() -> None:
    """§14 NP4: the range limit is max(0.03, 3.5 x pooled SE).

    At 4,000 calls per replicate 3.5 x SE is about .010, so the 0.03 floor
    alone decides: a range of .025 passes, one at the limit passes (the
    1e-9 tolerance) and one of .035 fails.
    """
    for top, range_value, passes in (
        (3900, 0.025, True),
        (3920, 0.03, True),
        (3940, 0.035, False),
    ):
        counts = {"D1": 3800, "D2": 3860, "D3": top}
        row, classes = _verdict(_grid(counts, n=4000))
        assert 3.5 * row["pooled_se"] < 0.011
        assert row["limit"] == gp.GATE_P_SPREAD_FLOOR
        assert row["donor_range"] == pytest.approx(range_value)
        assert row["floor_ok"] and row["range_evaluable"]
        assert bool(row["range_ok"]) == passes and bool(row["passed"]) == passes
        assert classes == {("broad", "X"): passes}


def test_np4_floor_counts_a_replicate_at_the_target() -> None:
    """§14 NP4: point precision >= target_L, so exactly target_L passes."""
    replicates = _grid(dict.fromkeys(GROUPS, 360))  # 360 / 400 = .90
    row, classes = _verdict(replicates)
    assert row["target"] == 0.90 and row["target"] == 360 / 400
    assert row["floor_ok"] and row["floor_failures"] == ""
    assert row["donor_range"] == 0.0 and row["passed"]
    assert classes == {("broad", "X"): True}
    replicates[("D2", 1)] = _replicate("D2", 1, 359)  # .8975
    row, classes = _verdict(replicates)
    assert not row["floor_ok"] and row["floor_failures"] == "D2/1"
    assert row["range_ok"] and not row["passed"]
    assert classes == {("broad", "X"): False}


def test_default_group_is_scored_on_its_check_half_only() -> None:
    """§14: the default donor's check half; §23.2 D2: no fit-half cell leaks.

    The frozen thresholds were fitted on the default group's fit half
    (``half == 0``), so its fit-half rows never enter NP4, at any seed.
    """
    replicates = _grid(dict.fromkeys(GROUPS, 380))
    for seed in SEEDS:
        # D1's fit half (even rows) is all wrong, its check half all right.
        frame = _replicate("D1", seed, 400)
        frame["correct"] = frame["half"] == 1
        replicates[("D1", seed)] = frame
    scored = gp.held_out_replicates(replicates, default_group="D1")
    assert list(scored) == list(replicates)
    for (group, _), table in scored.items():
        assert len(table) == (200 if group == "D1" else 400)
        assert set(table["half"]) == ({1} if group == "D1" else {0, 1})
    unchanged = gp.held_out_replicates(replicates, default_group=None)
    assert all(unchanged[key] is table for key, table in replicates.items())
    decisions = _decisions_at_10()
    tested = _tested_from_seed0(replicates, decisions, "D1")
    (item,) = tested[("broad", "X")] or []
    assert item.n_confident == 1000
    stats = gp.replicate_set_stats(replicates, decisions, tested, default_group="D1")
    default = stats[stats["group"] == "D1"]
    assert set(default["n_confident"]) == {200} and set(default["n_correct"]) == {200}
    others = stats[stats["group"] != "D1"]
    assert set(others["n_confident"]) == {400} and set(others["n_correct"]) == {380}
    # Counting D1's fit half too would give D1 a precision of .50.
    leaky = gp.replicate_set_stats(replicates, decisions, tested, default_group=None)
    assert set(leaky.loc[leaky["group"] == "D1", "precision"]) == {0.5}
    with pytest.raises(ValueError, match="not a group"):
        gp.held_out_replicates(replicates, default_group="D9")
    no_half = dict(replicates)
    no_half[("D1", 1)] = replicates[("D1", 1)].drop(columns="half")
    with pytest.raises(ValueError, match="half"):
        gp.replicate_set_stats(no_half, decisions, tested, default_group="D1")
    odd_half = dict(replicates)
    odd_half[("D1", 0)] = replicates[("D1", 0)].assign(half=2)
    with pytest.raises(ValueError, match="half"):
        gp.held_out_replicates(odd_half, default_group="D1")
    # A fit-half test cell of D1 in another group's table is a leak: under
    # D2 (d) the replicates are disjoint.
    leaked = dict(replicates)
    fit_row = replicates[("D1", 1)].iloc[[0]].assign(seed=1)
    leaked[("D3", 1)] = pd.concat([replicates[("D3", 1)], fit_row], ignore_index=True)
    with pytest.raises(ValueError, match="fit-half"):
        gp.replicate_set_stats(leaked, decisions, tested, default_group="D1")
    # A check-half cell of D1 elsewhere is not a fit-half leak.
    shared = dict(replicates)
    check_row = replicates[("D1", 1)].iloc[[1]].assign(seed=1)
    shared[("D3", 1)] = pd.concat([replicates[("D3", 1)], check_row], ignore_index=True)
    assert len(gp.held_out_replicates(shared, default_group="D1")[("D3", 1)]) == 401
    # Nothing is left of a default group whose table holds only its fit half.
    only_fit = dict(replicates)
    only_fit[("D1", 0)] = replicates[("D1", 0)].iloc[::2]
    with pytest.raises(ValueError, match="no rows"):
        gp.replicate_set_stats(only_fit, decisions, tested, default_group="D1")


def test_empty_tested_set_lists_raise() -> None:
    """``gate_p_tested_sets`` gives None, never [], for a key without sets."""
    decisions = _decisions_at_10()
    replicates = _grid(dict.fromkeys(GROUPS, 380))
    empty: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
        ("broad", "X"): [],
    }
    with pytest.raises(ValueError, match="empty"):
        gp.replicate_set_stats(replicates, decisions, empty, default_group=None)
    with pytest.raises(ValueError, match="empty"):
        gp.np4_class_verdicts(_set_verdict_rows({"10": True}), _seed_passed(), empty)
    with pytest.raises(ValueError, match="belongs to"):
        gp.np4_class_verdicts(
            _set_verdict_rows({"10": True}),
            _seed_passed(),
            {("broad", "Y"): [SINGLE_10]},
        )


def _set_verdict_rows(passed: Mapping[str, bool], cls: str = "X") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "level": ["broad"] * len(passed),
            "class": [cls] * len(passed),
            "set": list(passed),
            "passed": list(passed.values()),
        }
    )


def test_class_verdicts_feed_every_member_verdict() -> None:
    tested: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
        ("broad", "X"): [POOLED_30, SINGLE_10],
        ("broad", "Y"): None,
    }
    passing = gp.np4_class_verdicts(
        _set_verdict_rows({">=30": True, "10": True}), _seed_passed(), tested
    )
    failing = gp.np4_class_verdicts(
        _set_verdict_rows({">=30": True, "10": False}), _seed_passed(), tested
    )
    assert passing == {("broad", "X"): True, ("broad", "Y"): None}
    assert failing == {("broad", "X"): False, ("broad", "Y"): None}
    combined = res.every_member_verdict({"R1@0": passing, "R1@6": failing})
    assert combined[("broad", "X")]["status"] == res.GATE_P_FAILED
    assert combined[("broad", "X")]["failed_members"] == ["R1@6"]
    assert combined[("broad", "Y")]["status"] == res.GATE_P_NOT_EVALUABLE
    both = res.every_member_verdict({"R1@0": passing, "R1@6": passing})
    assert both[("broad", "X")]["status"] == res.GATE_P_PASSED
    # A key with tested sets needs a verdict row for each of them.
    with pytest.raises(ValueError, match="no NP4 verdict"):
        gp.np4_class_verdicts(
            _set_verdict_rows({"10": True}, cls="Z"), _seed_passed(), tested
        )
    with pytest.raises(ValueError, match="no NP4 verdict"):
        gp.np4_class_verdicts(_set_verdict_rows({"10": True}), _seed_passed(), tested)


def test_np4_inputs_that_mix_or_lack_replicates_raise() -> None:
    decisions = _decisions_at_10()
    tested = {("broad", "X"): [SINGLE_10]}
    table = _replicate("D1", 0, 380)
    clean = table.assign(recipe=res.CLEAN_RECIPE)
    with pytest.raises(res.ResolvabilityError, match="more than once"):
        gp.replicate_set_stats(
            {("D1", 0): pd.concat([table, clean])},
            decisions,
            tested,
            default_group=None,
            recipe=None,
        )
    with pytest.raises(res.ResolvabilityError, match="more than once"):
        gp.replicate_set_stats(
            {("D1", 0): pd.concat([table, table.assign(seed=1)])},
            decisions,
            tested,
            default_group=None,
        )
    with pytest.raises(ValueError, match="no rows"):
        gp.replicate_set_stats(
            {("D1", 0): clean}, decisions, tested, default_group=None
        )
    with pytest.raises(ValueError, match="no replicates"):
        gp.replicate_set_stats({}, decisions, tested, default_group=None)
    with pytest.raises(ValueError, match="belongs to"):
        gp.replicate_set_stats(
            {("D1", 0): table},
            decisions,
            {("broad", "Y"): [SINGLE_10]},
            default_group=None,
        )
    stats = gp.replicate_set_stats(
        _grid(dict.fromkeys(GROUPS, 380)), decisions, tested, default_group=None
    )
    with pytest.raises(ValueError, match="no precision target"):
        gp.np4_set_verdicts(stats, {"supercluster": 0.85}, _np4())
    incomplete = stats[~((stats["group"] == "D2") & (stats["seed"] == 1))]
    with pytest.raises(ValueError, match="full grid"):
        gp.np4_set_verdicts(incomplete, _targets(), _np4())
    with pytest.raises(ValueError, match="full grid"):
        gp.np4_set_verdicts(pd.concat([stats, stats.iloc[:1]]), _targets(), _np4())
    with pytest.raises(ValueError, match="at least 2 groups"):
        gp.np4_set_verdicts(stats[stats["group"] == "D1"], _targets(), _np4())


def test_np4_range_uses_means_of_group_values_d12() -> None:
    """Pins D12, confirmed on 2026-10-06 (pre-registration §23.9, §23.10).

    p_g and n_g are the means over a group's seeds of the per-seed precision
    and confident n; p_bar and n_bar are the means of the group values; the
    precision is unweighted.
    """
    counts = {
        "D1": ((100, 95), (120, 108)),
        "D2": ((400, 384), (400, 376)),
        "D3": ((400, 392), (400, 388)),
    }
    replicates = {
        (group, seed): _replicate(group, seed, k, n=n)
        for group, pairs in counts.items()
        for seed, (n, k) in enumerate(pairs)
    }
    row, _ = _verdict(replicates)
    p_groups = [np.mean([k / n for n, k in pairs]) for pairs in counts.values()]
    n_groups = [np.mean([n for n, _ in pairs]) for pairs in counts.values()]
    p_bar, n_bar = float(np.mean(p_groups)), float(np.mean(n_groups))
    assert p_groups == pytest.approx([0.925, 0.95, 0.975])
    assert row["p_bar"] == pytest.approx(p_bar)
    assert row["n_bar"] == pytest.approx(n_bar)
    se = math.sqrt(p_bar * (1.0 - p_bar) / n_bar)
    assert row["pooled_se"] == pytest.approx(se)
    assert row["limit"] == pytest.approx(max(0.03, 3.5 * se))
    assert row["donor_range"] == pytest.approx(0.05)
    assert row["n_groups"] == 3
    # Not the pooled precision of every call (1,743 / 1,820).
    assert row["p_bar"] != pytest.approx(1743 / 1820, abs=1e-3)


def test_np4_settings_follow_the_config() -> None:
    np4 = gp.Np4Settings.from_config(AnnotationResolvabilityConfig())
    assert np4.replicate_min_confident_n == 100
    assert np4.spread_se_multiplier == pytest.approx(3.5)
    assert np4.spread_floor == pytest.approx(0.03)
    assert np4.spread_floor == gp.GATE_P_SPREAD_FLOOR
    custom = gp.Np4Settings.from_config(
        AnnotationResolvabilityConfig(
            gate_p_replicate_min_confident_n=150, gate_p_spread_se_multiplier=3.0
        )
    )
    assert (custom.replicate_min_confident_n, custom.spread_se_multiplier) == (150, 3.0)
    for bad in (
        {"replicate_min_confident_n": 0},
        {"spread_se_multiplier": 0.0},
        {"spread_floor": -0.01},
    ):
        fields: dict[str, Any] = {
            "replicate_min_confident_n": 100,
            "spread_se_multiplier": 3.5,
            **bad,
        }
        with pytest.raises(ValueError, match="must be > 0"):
            gp.Np4Settings(**fields)
    # §14 NP4: "Seed 0 vs 1 changes <= 2% of confident labels".
    assert np4.max_seed_change == gp.NP4_MAX_SEED_CHANGE == pytest.approx(0.02)
    assert custom.max_seed_change == gp.NP4_MAX_SEED_CHANGE
    for share in (-0.01, 1.5, math.nan):
        with pytest.raises(ValueError, match=r"max_seed_change must lie in \[0, 1\]"):
            gp.Np4Settings(
                replicate_min_confident_n=100,
                spread_se_multiplier=3.5,
                max_seed_change=share,
            )


def test_level_targets_and_tested_set_labels() -> None:
    levels = ["broad", "lineage", "nt", "class", "supercluster", "subclass"]
    targets = gp.level_targets(AnnotationThresholds(), levels)
    assert targets == pytest.approx(
        {
            "broad": 0.90,
            "lineage": 0.90,
            "nt": 0.90,
            "class": 0.90,
            "supercluster": 0.85,
            "subclass": 0.85,
        }
    )
    with pytest.raises(ValueError, match="no precision target"):
        gp.level_targets(AnnotationThresholds(), ["broad", "unknown"])
    assert gp.tested_set_label(POOLED_30) == ">=30"
    assert gp.tested_set_label(SINGLE_10) == "10"


# --------------------------------------------------------------------------
# NP4's seed criterion: seed 0 vs 1 changes <= 2% of confident labels (§14)

# The levels and classes of the seed-criterion replicates: n confident calls
# of each class at 10 counts.
SEED_CLASSES = (("broad", "X", 400), ("broad", "Y", 400), ("supercluster", "S", 400))
SEED_TESTED: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
    (level, cls): [res.GatePTestedSet(level, cls, (10,), False, 0, math.nan, math.nan)]
    for level, cls, _ in SEED_CLASSES
} | {("supercluster", "T"): None}


def _seed_decisions() -> pd.DataFrame:
    """Frozen decisions emitting every class of ``SEED_CLASSES`` at 10 (0.70)."""
    return _np7_decisions({(level, cls, 10): 0.70 for level, cls, _ in SEED_CLASSES})


def _seed_table(group: str, seed: int) -> pd.DataFrame:
    """One replicate: ``SEED_CLASSES``' calls at 10 counts, all confident (.95)."""
    frames = []
    for level, cls, n in SEED_CLASSES:
        frame = bin_cells(
            np.full(n, 0.95), np.ones(n, dtype=bool), level=level, cls=cls, depth=10
        )
        frame["call"] = cls
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["cell_id"] = f"{group}_" + table["cell_id"].astype(str)
    table["sim_id"] = f"{group}_" + table["sim_id"].astype(str)
    table["seed"] = seed
    return table


def _seed_grid(
    changed: Mapping[str, int] | None = None,
) -> dict[gp.ReplicateKey, pd.DataFrame]:
    """Each group's seed-0 calls re-mapped at seed 1 (the same simulated cells).

    Seed 1 calls the first ``changed[group]`` broad X cells Y, confidently.
    """
    replicates: dict[gp.ReplicateKey, pd.DataFrame] = {}
    for group in GROUPS:
        base = _seed_table(group, 0)
        other = base.assign(seed=1)
        hit = np.flatnonzero((other["parent"] == "X").to_numpy())
        other.loc[hit[: (changed or {}).get(group, 0)], ["call", "parent"]] = "Y"
        replicates[(group, 0)] = base
        replicates[(group, 1)] = other
    return replicates


def _seed(
    replicates: Mapping[gp.ReplicateKey, pd.DataFrame],
    default_group: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    return gp.np4_seed_stability(
        replicates,
        _seed_decisions(),
        default_group=default_group,
        settings=_np4(),
        **kwargs,
    )


def _seed_row(table: pd.DataFrame, group: str, level: str = "broad") -> pd.Series:
    rows = table[(table["level"] == level) & (table["group"] == group)]
    assert len(rows) == 1
    return rows.iloc[0]


def _seed_sets(passed: Mapping[str, bool] | None = None) -> pd.DataFrame:
    """NP4 set verdicts of ``SEED_TESTED`` (each class passes unless named)."""
    records = [
        {
            "level": key[0],
            "class": key[1],
            "set": gp.tested_set_label(item),
            "passed": (passed or {}).get(key[1], True),
        }
        for key, items in SEED_TESTED.items()
        for item in items or ()
    ]
    return pd.DataFrame.from_records(records)


def test_np4_seed_changes_above_2pct_of_a_level_fail_every_class_of_it() -> None:
    """§14 NP4: seed 0 vs 1 changes <= 2% of confident labels per level.

    The statistic is the level's: 16 changed labels of a group's 800 broad
    labels (2%) pass at the limit, 17 fail, and a failing level fails NP4
    for each of its classes, Y included, whose own labels did not change.
    """
    table = _seed(_seed_grid({"D2": 16}))
    assert list(table.columns) == list(gp.NP4_SEED_COLUMNS)
    assert table[["level", "group"]].values.tolist() == [
        [level, group] for level in ("broad", "supercluster") for group in GROUPS
    ]
    row = _seed_row(table, "D2")
    assert (row["base_seed"], row["seed"]) == (0, 1)
    assert row["n_confident"] == 800 and row["n_changed"] == 16
    assert row["changed_share"] == pytest.approx(0.02)
    assert row["n_switched"] == 16 and row["n_crossed"] == 0
    assert row["max_change"] == gp.NP4_MAX_SEED_CHANGE and row["passed"]
    assert _seed_row(table, "D1")["n_changed"] == 0
    assert gp.np4_class_verdicts(_seed_sets(), table, SEED_TESTED) == {
        ("broad", "X"): True,
        ("broad", "Y"): True,
        ("supercluster", "S"): True,
        ("supercluster", "T"): None,
    }
    table = _seed(_seed_grid({"D2": 17}))
    assert not _seed_row(table, "D2")["passed"]
    assert _seed_row(table, "D2", "supercluster")["passed"]
    assert gp.np4_class_verdicts(_seed_sets(), table, SEED_TESTED) == {
        ("broad", "X"): False,
        ("broad", "Y"): False,
        ("supercluster", "S"): True,
        ("supercluster", "T"): None,
    }


def test_np4_seed_criterion_scores_call_changes_not_threshold_crossings() -> None:
    """D6 (pre-registration §23.9 item 3): mapping seeds, scored as call changes.

    The base is seed 0's confident labels. A change is another call of the
    same simulated cell at seed 1, confident there or not (a sink call
    included); the same name below its threshold at seed 1 is a threshold
    crossing, reported and not counted.
    """
    replicates = _seed_grid()
    base = replicates[("D1", 0)].copy()
    y_rows = np.flatnonzero((base["parent"] == "Y").to_numpy())[:20]
    x_rows = np.flatnonzero((base["parent"] == "X").to_numpy())
    # Unconfident at seed 0, so not labels: re-called X at seed 1 uncounted.
    base.loc[y_rows, "bp"] = 0.5
    other = base.assign(seed=1)
    other.loc[y_rows, ["call", "parent"]] = "X"
    switched, unconfident, sink, crossed = (
        x_rows[:5],
        x_rows[5:9],
        x_rows[9:12],
        x_rows[12:42],
    )
    other.loc[switched, ["call", "parent"]] = "Y"
    other.loc[unconfident, ["call", "parent"]] = "Y"
    other.loc[unconfident, "bp"] = 0.5
    other.loc[sink, "call"] = "Splatter"
    other.loc[sink, "parent"] = None
    other.loc[crossed, "bp"] = 0.5
    replicates[("D1", 0)], replicates[("D1", 1)] = base, other
    row = _seed_row(_seed(replicates), "D1")
    assert row["n_confident"] == 780
    assert row["n_changed"] == 12 and row["n_switched"] == 5
    assert row["n_crossed"] == 30
    assert row["changed_share"] == pytest.approx(12 / 780) and row["passed"]
    # Counting the crossings too would give 42 / 780 = 5.4% and fail.
    assert (row["n_changed"] + row["n_crossed"]) / row["n_confident"] > 0.05


def test_np4_seed_criterion_reproduces_resolvability_seed_stability() -> None:
    """D6 names ``seed_stability``: the same share on any seeded sweep."""
    n_changed = 0
    for rng_seed in range(5):
        rng = np.random.default_rng(rng_seed)
        replicates: dict[gp.ReplicateKey, pd.DataFrame] = {}
        for group in ("D1", "D2"):
            base = _random_cells(rng)
            base["call"] = base["parent"]
            base["cell_id"] = f"{group}_" + base["cell_id"].astype(str)
            base["sim_id"] = f"{group}_" + base["sim_id"].astype(str)
            other = base.assign(seed=1)
            moved = rng.random(len(other)) < 0.03
            flipped = np.where(other.loc[moved, "parent"] == "X", "Y", "X")
            other.loc[moved, "call"] = flipped
            other.loc[moved, "parent"] = flipped
            other["bp"] = np.round(
                np.clip(other["bp"] + rng.normal(0.0, 0.05, len(other)), 0.0, 1.0), 3
            )
            replicates[(group, 0)], replicates[(group, 1)] = base, other
        decisions = res.decide(
            pd.concat([replicates[("D1", 0)], replicates[("D2", 0)]]),
            [BROAD],
            [10, 30, 100],
            settings(),
        )
        table = gp.np4_seed_stability(
            replicates, decisions, default_group=None, settings=_np4()
        )
        for group in ("D1", "D2"):
            row = _seed_row(table, group)
            expected = res.seed_stability(
                replicates[(group, 0)], replicates[(group, 1)], decisions, "broad"
            )
            assert row["n_confident"] > 0
            assert row["changed_share"] == pytest.approx(expected, abs=1e-12)
            n_changed += int(row["n_changed"])
    assert n_changed > 0


def test_np4_seed_criterion_scores_each_group_and_reports_the_pooled_share() -> None:
    """Each group's seed pair is scored; the share over the groups is reported."""
    table = _seed(_seed_grid({"D1": 24}))
    d1 = _seed_row(table, "D1")
    assert d1["changed_share"] == pytest.approx(0.03) and not d1["passed"]
    for group in ("D2", "D3"):
        assert _seed_row(table, group)["passed"]
    broad = table[table["level"] == "broad"]
    assert np.allclose(broad["pooled_changed_share"], 24 / 2400)
    assert np.allclose(
        table.loc[table["level"] == "supercluster", "pooled_changed_share"], 0.0
    )
    # The pooled 1% would pass; the group at 3% fails the level.
    verdicts = gp.np4_class_verdicts(_seed_sets(), table, SEED_TESTED)
    assert verdicts[("broad", "X")] is False and verdicts[("broad", "Y")] is False
    assert verdicts[("supercluster", "S")] is True


def test_np4_seed_criterion_reads_the_default_group_check_half() -> None:
    """§14: the default donor's check half, at both seeds (pre-reg §23.9 item 3)."""
    replicates = _seed_grid()
    other = replicates[("D1", 1)]
    fit_x = np.flatnonzero(((other["parent"] == "X") & (other["half"] == 0)).to_numpy())
    other.loc[fit_x, "call"] = "Y"
    row = _seed_row(_seed(replicates, default_group="D1"), "D1")
    assert row["n_confident"] == 400 and row["n_changed"] == 0 and row["passed"]
    leaky = _seed_row(_seed(replicates), "D1")
    assert leaky["n_confident"] == 800 and leaky["n_changed"] == 200
    assert not leaky["passed"]


def test_np4_seed_criterion_passes_a_group_without_confident_labels() -> None:
    """Nothing can change where a group has no confident label (as NP4 vacuous)."""
    replicates = _seed_grid()
    for seed in SEEDS:
        table = replicates[("D3", seed)]
        table.loc[table["level"] == "supercluster", "bp"] = 0.5
    row = _seed_row(_seed(replicates), "D3", "supercluster")
    assert row["n_confident"] == 0 and row["n_changed"] == 0
    assert math.isnan(row["changed_share"]) and row["passed"]


def test_np4_seed_criterion_scores_one_emission_member() -> None:
    """Version 7: NP4 is scored in every member, each on its own seed pair."""
    replicates = {
        key: pd.concat(
            [table.assign(member=name) for name in ("R1@0", "R1@6")],
            ignore_index=True,
        )
        for key, table in _seed_grid().items()
    }
    other = replicates[("D2", 1)]
    hit = np.flatnonzero(
        ((other["member"] == "R1@6") & (other["parent"] == "X")).to_numpy()
    )
    other.loc[hit[:17], "call"] = "Y"
    assert _seed_row(_seed(replicates, member="R1@0"), "D2")["passed"]
    six = _seed_row(_seed(replicates, member="R1@6"), "D2")
    assert six["n_changed"] == 17 and not six["passed"]
    with pytest.raises(res.ResolvabilityError, match="more than once"):
        _seed(replicates)


def test_np4_seed_criterion_refuses_inputs_that_are_not_one_simulation_remapped() -> (
    None
):
    """Seeds 0 / 1 re-map the same simulated cells (D6); else gate P stops."""
    replicates = _seed_grid()
    key = ("D2", 1)

    def edited(table: pd.DataFrame) -> dict[gp.ReplicateKey, pd.DataFrame]:
        return {**replicates, key: table}

    # Another simulation (a simulation seed) changes the simulated counts.
    counts = replicates[key].copy()
    counts.loc[0, "total_counts"] = 11.0
    with pytest.raises(ValueError, match="total_counts"):
        _seed(edited(counts))
    with pytest.raises(ValueError, match="same simulated cells"):
        _seed(edited(replicates[key].iloc[1:]))
    extra = pd.concat([replicates[key], replicates[("D3", 1)].iloc[[0]]])
    with pytest.raises(ValueError, match="same simulated cells"):
        _seed(edited(extra))
    with pytest.raises(ValueError, match="both hold mapping seed 0"):
        _seed(edited(replicates[key].assign(seed=0)))
    two_seeds = replicates[key].copy()
    two_seeds.loc[0, "seed"] = 2
    with pytest.raises(ValueError, match="mapping seeds"):
        _seed(edited(two_seeds))
    clean = replicates[key].assign(recipe=res.CLEAN_RECIPE)
    with pytest.raises(ValueError, match="recipe"):
        _seed(edited(clean), recipe=None)
    with pytest.raises(ValueError, match="no rows"):
        _seed(edited(clean))
    mixed = pd.concat([replicates[key], clean], ignore_index=True)
    with pytest.raises(res.ResolvabilityError, match="more than once"):
        _seed(edited(mixed), recipe=None)
    no_counts = replicates[key].drop(columns="total_counts")
    with pytest.raises(ValueError, match="total_counts"):
        _seed(edited(no_counts))
    with pytest.raises(ValueError, match="full grid"):
        _seed({k: v for k, v in replicates.items() if k != ("D3", 1)})
    with pytest.raises(ValueError, match="at least 2 groups"):
        _seed({k: v for k, v in replicates.items() if k[0] == "D1"})
    with pytest.raises(ValueError, match="base seed"):
        _seed({(group, seed + 1): v for (group, seed), v in replicates.items()})
    with pytest.raises(ValueError, match="second seed"):
        _seed({k: v for k, v in replicates.items() if k[1] == 0})
    with pytest.raises(ValueError, match="no replicates"):
        _seed({})


def test_np4_class_verdicts_need_the_tested_sets_and_the_seed_criterion() -> None:
    """NP4 per (level, class): every tested set, and the level's seed criterion."""
    seeds = _seed(_seed_grid())
    passing = gp.np4_class_verdicts(_seed_sets(), seeds, SEED_TESTED)
    assert passing == {
        ("broad", "X"): True,
        ("broad", "Y"): True,
        ("supercluster", "S"): True,
        ("supercluster", "T"): None,
    }
    one_set = gp.np4_class_verdicts(_seed_sets({"X": False}), seeds, SEED_TESTED)
    assert one_set[("broad", "X")] is False and one_set[("broad", "Y")] is True
    seed_fails = gp.np4_class_verdicts(
        _seed_sets(), _seed(_seed_grid({"D3": 17})), SEED_TESTED
    )
    assert seed_fails[("broad", "Y")] is False
    combined = res.every_member_verdict({"R1@0": passing, "R1@6": seed_fails})
    assert combined[("broad", "Y")]["status"] == res.GATE_P_FAILED
    assert combined[("broad", "Y")]["failed_members"] == ["R1@6"]
    assert combined[("supercluster", "S")]["status"] == res.GATE_P_PASSED
    # A level whose classes have tested sets needs its seed-criterion rows.
    with pytest.raises(ValueError, match="seed-criterion"):
        gp.np4_class_verdicts(
            _seed_sets(), seeds[seeds["level"] == "broad"], SEED_TESTED
        )
    blank = seeds.astype({"passed": object})
    blank.loc[0, "passed"] = None
    with pytest.raises(ValueError, match="no passed value"):
        gp.np4_class_verdicts(_seed_sets(), blank, SEED_TESTED)
    # A level whose classes have no tested set needs none.
    untested = {
        ("broad", "X"): SEED_TESTED[("broad", "X")],
        ("supercluster", "T"): None,
    }
    sets = _seed_sets()
    assert gp.np4_class_verdicts(
        sets[sets["class"] == "X"], seeds[seeds["level"] == "broad"], untested
    ) == {("broad", "X"): True, ("supercluster", "T"): None}


# --------------------------------------------------------------------------
# Pooled held-out calls (§14: the input of the tested sets and of NP3)


def test_pooled_held_out_cells_take_the_default_group_check_half_at_one_seed() -> None:
    replicates = _grid(dict.fromkeys(GROUPS, 380))
    pooled = gp.pooled_held_out_cells(replicates, default_group="D1")
    # D1's check half (200 rows) and both other donors in full, at seed 0.
    assert len(pooled) == 200 + 400 + 400
    assert set(pooled["seed"]) == {0}
    assert pooled.index.equals(pd.RangeIndex(len(pooled)))
    d1 = pooled[pooled["cell_id"].str.startswith("D1_")]
    assert set(d1["half"]) == {1}
    seed1 = gp.pooled_held_out_cells(replicates, default_group=None, seed=1)
    assert len(seed1) == 1200 and set(seed1["seed"]) == {1}
    with pytest.raises(ValueError, match="no replicate"):
        gp.pooled_held_out_cells(replicates, default_group="D1", seed=7)
    leaked = dict(replicates)
    fit_row = replicates[("D1", 0)].iloc[[0]]
    leaked[("D2", 0)] = pd.concat([replicates[("D2", 0)], fit_row], ignore_index=True)
    with pytest.raises(ValueError, match="fit-half"):
        gp.pooled_held_out_cells(leaked, default_group="D1")


# --------------------------------------------------------------------------
# NP3: precision and coverage at the panel's depth (§14 NP3)

# Natural composition of the truth types the NP3 tests use.
NATURAL = {"A": 0.9, "B": 0.1, "C": 0.5}


def _np3() -> gp.Np3Settings:
    return gp.Np3Settings.from_config(AnnotationResolvabilityConfig())


def _decisions(depths: Sequence[int]) -> pd.DataFrame:
    """Frozen decisions emitting broad X at every depth (threshold 0.70)."""
    return res.decide(
        pd.concat(
            [
                bin_cells(np.full(300, 0.95), np.ones(300, dtype=bool), depth=depth)
                for depth in depths
            ],
            ignore_index=True,
        ),
        [BROAD],
        list(depths),
        settings(),
    )


def _calls(
    depth: int,
    groups: Sequence[tuple[str, str, int, int, float]],
    *,
    parent: str = "X",
    prefix: str = "",
) -> pd.DataFrame:
    """Rows of one bin: per group ``(truth type, truth class, n, n_correct, bp)``.

    The first ``n_correct`` rows of a group are correct; the others are not
    (whatever their truth class: the statistics read ``correct`` only).
    """
    frames = []
    for leaf, truth_parent, n, n_correct, bp in groups:
        frame = bin_cells(
            np.full(n, bp),
            np.arange(n) < n_correct,
            depth=depth,
            cls=parent,
            truth_parent=truth_parent,
            leaf=np.full(n, leaf),
        )
        ids = [f"{prefix}{leaf}{truth_parent}{bp}_{index}" for index in range(n)]
        frame["cell_id"] = ids
        frame["sim_id"] = [f"{cell}|D{depth}" for cell in ids]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _np3_rows(
    cells: pd.DataFrame,
    depths: Sequence[int],
    *,
    composition: Mapping[str, float] = NATURAL,
    depth_histogram: Mapping[int, float] | None = None,
) -> tuple[
    dict[tuple[str, str], list[res.GatePTestedSet] | None], pd.DataFrame, pd.DataFrame
]:
    """Tested sets, NP3 stats and NP3 verdicts of pooled held-out cells."""
    decisions = _decisions(depths)
    tested = res.gate_p_tested_sets(cells, decisions, regime="provisional")
    stats = gp.np3_set_stats(
        {("D1", 0): cells},
        decisions,
        tested,
        default_group=None,
        composition=composition,
        settings=_np3(),
        depth_histogram=depth_histogram,
    )
    verdicts = gp.np3_verdicts(stats, AnnotationThresholds(), _np3())
    return tested, stats, verdicts


def _scheme(frame: pd.DataFrame, scheme: str, label: str | None = None) -> pd.Series:
    rows = frame[frame["scheme"] == scheme]
    if label is not None:
        rows = rows[rows["set"] == label]
    assert len(rows) == 1
    return rows.iloc[0]


def test_kish_shrinkage_makes_a_set_at_p95_fail_its_wilson_bound() -> None:
    """§14 NP3: Wilson bounds on reweighted sets use the Kish effective n.

    200 confident X calls at 100 counts, precision .95 in each of its two
    truth types (180 of A, 20 of B). Unweighted the bound is .910 >= .90.
    Class-balanced, each type carries half the weight: the precision stays
    .95 (>= target+ .95), but the Kish n falls to 72 and the bound to .873.
    """
    cells = _calls(100, [("A", "X", 180, 171, 0.95), ("B", "X", 20, 19, 0.95)])
    tested, stats, verdicts = _np3_rows(cells, [100])
    (item,) = tested[("broad", "X")] or []
    assert item.depths == (100,) and item.n_confident == 200
    unweighted = _scheme(verdicts, gp.NP3_UNWEIGHTED)
    assert unweighted["precision"] == pytest.approx(0.95)
    assert unweighted["kish_n"] == pytest.approx(200)
    assert unweighted["wilson_lb"] == pytest.approx(0.9104, abs=1e-4)
    assert unweighted["passed"] and not unweighted["scored"]
    balanced = _scheme(verdicts, gp.NP3_CLASS_BALANCED)
    assert balanced["scored"]
    assert balanced["precision"] == pytest.approx(0.95)
    assert balanced["kish_n"] == pytest.approx(72.0)
    assert balanced["wilson_lb"] == pytest.approx(0.8731, abs=1e-4)
    assert balanced["target_plus"] == pytest.approx(0.95)
    assert balanced["point_ok"] and balanced["coverage_ok"]
    assert not balanced["wilson_ok"] and not balanced["passed"]
    # The natural composition (A .9, B .1) matches the set: weights 1.
    natural = _scheme(verdicts, gp.NP3_NATURAL)
    assert natural["kish_n"] == pytest.approx(200) and natural["passed"]
    table = gp.validated_min_depth(verdicts, tested, [100])
    assert table.to_dict("records") == [
        {
            "level": "broad",
            "class": "X",
            "tested_max_depth": 100,
            "validated_min_depth": None,
            "passed": False,
            "reason": gp.NP3_REASON_DEEP_SET_FAILED,
            "n_sets": 1,
            "failed_sets": "100",
            "stop_depth": None,
            "stop_reason": "",
        }
    ]
    assert gp.np3_class_verdicts(table) == {("broad", "X"): False}
    assert len(stats) == len(verdicts) == 5


def test_np3_wilson_bound_must_reach_the_target_not_the_target_less_002() -> None:
    """§14 NP3: Wilson bound >= target_L (the emission rule's -.02 is NP6's).

    200 X calls at 100 counts, .95 right in each type (160 of A, 40 of B).
    Class-balanced, the Kish n is 4 x 160 x 40 / 200 = 128 and the bound
    .897: above target_L - .02 (.88), below target_L (.90), so it fails.
    The natural composition (A .8, B .2) matches the set: Kish n 200, bound
    .910, and the set passes.
    """
    cells = _calls(100, [("A", "X", 160, 152, 0.95), ("B", "X", 40, 38, 0.95)])
    _, _, verdicts = _np3_rows(cells, [100], composition={"A": 0.8, "B": 0.2})
    balanced = _scheme(verdicts, gp.NP3_CLASS_BALANCED)
    assert balanced["precision"] == pytest.approx(0.95)
    assert balanced["kish_n"] == pytest.approx(128.0)
    assert 0.88 <= balanced["wilson_lb"] < 0.90
    assert balanced["wilson_lb"] == pytest.approx(0.8974, abs=1e-4)
    assert balanced["target"] == pytest.approx(0.90)
    assert balanced["point_ok"] and balanced["coverage_ok"]
    assert not balanced["wilson_ok"] and not balanced["passed"]
    natural = _scheme(verdicts, gp.NP3_NATURAL)
    assert natural["kish_n"] == pytest.approx(200.0)
    assert natural["wilson_ok"] and natural["passed"]


def test_np3_coverage_is_weighted_on_the_scope_and_precision_on_the_set() -> None:
    """The coverage weighs every call of the class in the set's scope; the
    precision, Kish n and bound reweight the confident set on its own.

    X's calls at 100 counts: 300 confident of A (270 right), 40 confident of
    B (right) and 60 unconfident of B. Class-balanced, the confident set
    gives A and B 170 each: precision (153 + 170) / 340 = .95. The coverage
    reweights all 400 calls (A and B 200 each, so a B call weighs 2): (200 +
    40 x 2) / 400 = .70, against .85 unweighted. Taking the set's weights
    from the calls' weighting instead would give (180 + 80) / 280 = .929.
    """
    cells = _calls(
        100,
        [
            ("A", "X", 300, 270, 0.95),
            ("B", "X", 40, 40, 0.95),
            ("B", "X", 60, 60, 0.5),
        ],
    )
    composition = {"A": 0.6, "B": 0.4}
    _, _, verdicts = _np3_rows(cells, [100], composition=composition)
    # (precision, coverage, the confident set's weights per call of A and B)
    expected = {
        gp.NP3_UNWEIGHTED: (310 / 340, 340 / 400, (1.0, 1.0)),
        gp.NP3_CLASS_BALANCED: (0.95, 0.70, (170 / 300, 170 / 40)),
        gp.NP3_NATURAL: (
            (0.6 * 340 * 0.9 + 0.4 * 340) / 340,
            (0.6 * 400 + 40 * 1.6) / 400,
            (0.6 * 340 / 300, 0.4 * 340 / 40),
        ),
    }
    for scheme, (precision, coverage, (w_a, w_b)) in expected.items():
        row = _scheme(verdicts, scheme)
        assert row["n_called"] == 400 and row["n_confident"] == 340
        assert row["precision"] == pytest.approx(precision)
        assert row["coverage"] == pytest.approx(coverage)
        kish = (300 * w_a + 40 * w_b) ** 2 / (300 * w_a**2 + 40 * w_b**2)
        assert row["kish_n"] == pytest.approx(kish)
    balanced = _scheme(verdicts, gp.NP3_CLASS_BALANCED)
    assert balanced["coverage"] != pytest.approx(
        _scheme(verdicts, gp.NP3_UNWEIGHTED)["coverage"]
    )


def test_pooled_sets_are_reweighted_as_one_set_across_their_bins() -> None:
    """§14 "reweighted sets": a ">= D_P" set is weighted as one set, not per bin.

    The ">= 60" set pools 150 cells at their deepest row, 100 counts (30 of
    A, 120 of B), and 100 cells at 60 (80 of A, 20 of B). Weighted as one
    set (110 of A, 140 of B), class-balanced, every A call weighs 125 / 110
    and every B call 125 / 140: Kish n 4 x 110 x 140 / 250 = 246.4. Weighted
    per bin, the 30 deep A calls and the 20 shallow B calls would each weigh
    2.5 and the Kish n fall to 160: the failure that
    ``pooled_composition_weights`` records on ag7.
    """
    good = (0.95, True)
    cells = tracked_cells(
        [
            ("a_deep", 30, {60: good, 100: good}),
            ("b_deep", 120, {60: good, 100: good}),
            ("a_low", 80, {60: good}),
            ("b_low", 20, {60: good}),
        ]
    )
    cells[res.TRUTH_LEAF_COLUMN] = np.where(
        cells["cell_id"].str.startswith("a_"), "A", "B"
    )
    composition = {"A": 0.6, "B": 0.4}
    tested, stats, _ = _np3_rows(cells, [60, 100], composition=composition)
    pooled = [item for item in tested[("broad", "X")] or [] if item.pooled]
    assert [(item.depths, item.n_confident) for item in pooled] == [((60, 100), 250)]
    deepest = res.deepest_rows(cells).reset_index(drop=True)
    assert sorted(set(deepest["depth"])) == [60, 100]
    leaf = deepest[res.TRUTH_LEAF_COLUMN].to_numpy()
    counts = {"A": 110, "B": 140}
    assert {name: int((leaf == name).sum()) for name in counts} == counts
    for scheme, shares in (
        (gp.NP3_CLASS_BALANCED, {"A": 0.5, "B": 0.5}),
        (gp.NP3_NATURAL, composition),
    ):
        expected = np.array([shares[name] * 250 / counts[name] for name in leaf])
        weights = gp.np3_set_weights(
            deepest, scheme, composition=composition, trim_factor=10.0
        )
        assert weights == pytest.approx(expected)
        row = _scheme(stats, scheme, ">=60")
        assert row["n_confident"] == 250
        kish = float(expected.sum() ** 2 / (expected**2).sum())
        assert row["kish_n"] == pytest.approx(kish)
    balanced = _scheme(stats, gp.NP3_CLASS_BALANCED, ">=60")
    assert balanced["kish_n"] == pytest.approx(4 * 110 * 140 / 250)


def test_np3_scores_the_default_group_on_its_check_half_only() -> None:
    """NP3 pools the replicates itself, so the fit half cannot leak (§14).

    The default donor D1's table holds both halves (400 calls): its 200
    fit-half calls, on which the frozen thresholds were fitted, are left
    out, so every set holds D1's 200 check-half calls and D2's 200. A plain
    concat of the two tables would have scored 600.
    """
    d1 = _calls(100, [("A", "X", 400, 400, 0.95)], prefix="d1")
    d2 = _calls(100, [("A", "X", 200, 200, 0.95)], prefix="d2")
    assert int((d1["half"] == 0).sum()) == 200
    replicates = {("D1", 0): d1, ("D2", 0): d2}
    decisions = _decisions([100])
    pooled = gp.pooled_held_out_cells(replicates, default_group="D1")
    tested = res.gate_p_tested_sets(pooled, decisions, regime="provisional")
    stats = gp.np3_set_stats(
        replicates,
        decisions,
        tested,
        default_group="D1",
        composition=NATURAL,
        settings=_np3(),
    )
    assert (stats["n_called"] == 400).all() and (stats["n_confident"] == 400).all()
    # A fit-half cell of D1 in D2's table is a leak.
    leaked = {
        ("D1", 0): d1,
        ("D2", 0): pd.concat([d2, d1[d1["half"] == 0].iloc[:1]], ignore_index=True),
    }
    with pytest.raises(ValueError, match="fit-half"):
        gp.np3_set_stats(
            leaked,
            decisions,
            tested,
            default_group="D1",
            composition=NATURAL,
            settings=_np3(),
        )
    # The default group's table needs its split halves.
    with pytest.raises(ValueError, match="half"):
        gp.np3_set_stats(
            {("D1", 0): d1.drop(columns="half"), ("D2", 0): d2},
            decisions,
            tested,
            default_group="D1",
            composition=NATURAL,
            settings=_np3(),
        )
    # No caller can leave the default group out by omission.
    with pytest.raises(TypeError, match="default_group"):
        gp.np3_set_stats(  # type: ignore[call-arg]
            replicates, decisions, tested, composition=NATURAL, settings=_np3()
        )
    with pytest.raises(ValueError, match="seed label"):
        gp.np3_set_stats(
            replicates,
            decisions,
            tested,
            default_group="D1",
            composition=NATURAL,
            settings=_np3(),
            seed=1,
        )


def test_np3_coverage_below_030_fails() -> None:
    """§14 NP3: coverage >= 0.30 (``gate_p_min_coverage``); .29 fails."""
    for n_confident, passes in ((300, True), (290, False)):
        cells = _calls(
            100,
            [
                ("A", "X", n_confident, n_confident, 0.95),
                ("A", "X", 1000 - n_confident, 1000 - n_confident, 0.5),
            ],
        )
        _, _, verdicts = _np3_rows(cells, [100])
        for scheme in gp.NP3_SCORED_SCHEMES:
            row = _scheme(verdicts, scheme)
            assert row["n_called"] == 1000
            assert row["coverage"] == pytest.approx(n_confident / 1000)
            assert row["point_ok"] and row["wilson_ok"]
            assert bool(row["coverage_ok"]) == passes
            assert bool(row["passed"]) == passes


def _pooled_cells(shallow: int, deep: int, *, deep_wrong: int) -> pd.DataFrame:
    """150 cells reaching ``deep`` (``deep_wrong`` wrong there), 100 at ``shallow``."""
    deep_flags = np.arange(150) >= deep_wrong
    return tracked_cells(
        [
            ("deep", 150, {shallow: (0.95, True), deep: (0.95, deep_flags)}),
            ("low", 100, {shallow: (0.95, True)}),
        ]
    ).assign(**{res.TRUTH_LEAF_COLUMN: "A"})


def test_np3_margin_comes_from_the_shallowest_bin_of_the_set() -> None:
    """§14 NP3: target+_L takes the margin of the set's shallowest bin.

    The ">= D_P" set holds 240 / 250 = .96 correct calls. With D_P = 30 the
    margin is that of 30 counts (+.10, capped at .97), so .96 fails, though
    its deepest bin (100) would only need .95; with D_P = 60 it passes.
    """
    for shallow, target_plus, passes in ((30, 0.97, False), (60, 0.95, True)):
        cells = _pooled_cells(shallow, 100, deep_wrong=10)
        tested, _, verdicts = _np3_rows(cells, [shallow, 100])
        items = tested[("broad", "X")] or []
        pooled = [item for item in items if item.pooled]
        assert [item.depths for item in pooled] == [(shallow, 100)]
        assert pooled[0].n_confident == 250
        label = f">={shallow}"
        for scheme in gp.NP3_SCORED_SCHEMES:
            row = _scheme(verdicts, scheme, label)
            assert row["set_min_depth"] == shallow
            assert row["target"] == pytest.approx(0.90)
            assert row["target_plus"] == pytest.approx(target_plus)
            assert row["precision"] == pytest.approx(0.96)
            assert row["wilson_ok"] and row["coverage_ok"]
            assert bool(row["point_ok"]) == passes and bool(row["passed"]) == passes
        # The shallow bin is also tested on its own (250 calls, all right).
        single = _scheme(verdicts, gp.NP3_NATURAL, str(shallow))
        assert single["precision"] == 1.0 and single["passed"]
        table = gp.validated_min_depth(verdicts, tested, [shallow, 100])
        (record,) = table.to_dict("records")
        assert record["tested_max_depth"] == shallow
        assert record["passed"] is passes
        assert record["validated_min_depth"] == (shallow if passes else None)


def _sweep_cells(rng: np.random.Generator, n_cells: int = 2400) -> pd.DataFrame:
    """Held-out cells of three truth types at (10, 30, 100), shared across depths.

    Types A and B are of class X, C of class Y; some calls go to the other
    class (rarely at bp >= .8), and the correctness follows the call.
    """
    native = rng.choice(np.array([10, 30, 100]), size=n_cells, p=[0.5, 0.3, 0.2])
    leaf = rng.choice(np.array(["A", "B", "C"]), size=n_cells, p=[0.5, 0.2, 0.3])
    truth = np.where(leaf == "C", "Y", "X")
    frames = []
    for depth in (10, 30, 100):
        index = np.flatnonzero(native >= depth)
        ids = [f"c{value}" for value in index]
        frame = _rows(ids, depth)
        bp = np.round(rng.uniform(0.5, 1.0, len(ids)), 3)
        # Calls at bp >= .8 are almost always right, so the bins are emitted.
        swap = rng.random(len(ids)) < np.where(bp >= 0.8, 0.005, 0.3)
        other = np.where(truth[index] == "X", "Y", "X")
        frame["truth_parent"] = truth[index]
        frame[res.TRUTH_LEAF_COLUMN] = leaf[index]
        frame["parent"] = np.where(swap, other, truth[index])
        frame["correct"] = ~swap
        frame["bp"] = bp
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_tested_sets_never_hold_fewer_than_200_calls_and_count_each_cell_once() -> None:
    """§12 M13 test, as a seeded sweep over the NP3 schemes.

    Every tested set holds >= 200 confident calls, each test cell once (a
    pooled set at the cell's deepest row); every scheme keeps them all (no
    zero weight) with a Kish n <= n, and the unweighted scheme reproduces
    ``gate_p_tested_sets`` exactly.
    """
    composition = {"A": 0.6, "B": 0.1, "C": 0.3}
    kinds = {"pooled": 0, "single": 0}
    for rng_seed in range(6):
        cells = _sweep_cells(np.random.default_rng(100 + rng_seed))
        decisions = res.decide(cells, [BROAD], [10, 30, 100], settings())
        tested = res.gate_p_tested_sets(cells, decisions, regime="provisional")
        stats = gp.np3_set_stats(
            {("D1", 0): cells},
            decisions,
            tested,
            default_group=None,
            composition=composition,
            settings=_np3(),
        )
        confident = res.frozen_confident_mask(
            cells, res.emission_lookup(decisions, "provisional")
        )
        n_sets = 0
        for (level, cls), items in tested.items():
            for item in items or []:
                n_sets += 1
                kinds["pooled" if item.pooled else "single"] += 1
                assert item.n_confident >= 200
                mask = gp.tested_set_mask(cells, confident, item)
                ids = cells.loc[mask, "cell_id"]
                assert ids.is_unique and len(ids) == item.n_confident
                rows = stats[
                    (stats["level"] == level)
                    & (stats["class"] == cls)
                    & (stats["set"] == gp.tested_set_label(item))
                ]
                assert set(rows["scheme"]) == {
                    gp.NP3_UNWEIGHTED,
                    gp.NP3_NATURAL,
                    gp.NP3_CLASS_BALANCED,
                    gp.NP3_NATURAL_TEST_CELLS,
                    gp.NP3_CLASS_BALANCED_TEST_CELLS,
                }
                assert (rows["n_confident"] == item.n_confident).all()
                assert (rows["kish_n"] <= rows["n_confident"] + 1e-9).all()
                plain = _scheme(rows, gp.NP3_UNWEIGHTED)
                assert abs(plain["precision"] - item.precision) <= 1e-12
                assert abs(plain["wilson_lb"] - item.wilson_lb) <= 1e-12
                assert plain["kish_n"] == item.n_confident
        assert len(stats) == 5 * n_sets
    assert kinds["pooled"] > 0 and kinds["single"] > 0


def test_class_balanced_gives_each_truth_type_of_the_set_equal_weight() -> None:
    """Pre-registration §23.9 item 2 (D12), and the test-cell reading beside it.

    X's set: 300 calls of A and 60 of B (class X, right) and 40 of C (class
    Y, wrong). Class-balanced weights give each of the three types a third
    of the set, so the precision is 2/3, though 90% of the calls are right:
    a wrong type weighs as much as a right one. The report-only test-cell
    reading rebalances the types within each truth class over the bin's
    test cells instead (C's own 500 calls of Y included); it leaves class
    totals, and so this precision, unchanged.
    """
    cells = pd.concat(
        [
            _calls(100, [("A", "X", 300, 300, 0.95), ("B", "X", 60, 60, 0.95)]),
            _calls(100, [("C", "Y", 40, 0, 0.95)], prefix="w"),
            _calls(100, [("C", "Y", 500, 500, 0.95)], parent="Y"),
        ],
        ignore_index=True,
    )
    rows = cells[cells["parent"] == "X"].reset_index(drop=True)
    weights = gp.np3_set_weights(rows, gp.NP3_CLASS_BALANCED, trim_factor=10.0)
    leaf = rows[res.TRUTH_LEAF_COLUMN].to_numpy()
    totals = [float(weights[leaf == name].sum()) for name in ("A", "B", "C")]
    assert totals == pytest.approx([400 / 3] * 3)
    _, _, verdicts = _np3_rows(cells, [100])
    x_rows = verdicts[verdicts["class"] == "X"]
    assert _scheme(x_rows, gp.NP3_UNWEIGHTED)["precision"] == pytest.approx(0.90)
    assert _scheme(x_rows, gp.NP3_CLASS_BALANCED)["precision"] == pytest.approx(2 / 3)
    on_cells = _scheme(x_rows, gp.NP3_CLASS_BALANCED_TEST_CELLS)
    assert not on_cells["scored"]
    assert on_cells["precision"] == pytest.approx(0.90)
    # A and B carry 180 test-cell weights each (class X's 360, rebalanced).
    scope = cells.reset_index(drop=True)
    test_weights = gp.np3_test_cell_weights(scope, gp.NP3_CLASS_BALANCED_TEST_CELLS)
    scope_leaf = scope[res.TRUTH_LEAF_COLUMN].to_numpy()
    assert float(test_weights[scope_leaf == "A"].sum()) == pytest.approx(180)
    assert float(test_weights[scope_leaf == "B"].sum()) == pytest.approx(180)
    assert test_weights[scope_leaf == "C"] == pytest.approx(1.0)


def test_natural_weights_follow_the_reference_composition() -> None:
    """§14 NP3: the reference's natural composition within the class's set."""
    cells = _calls(
        100,
        [("A", "X", 300, 300, 0.95), ("B", "X", 60, 60, 0.95), ("C", "Y", 40, 0, 0.95)],
    )
    composition = {"A": 0.5, "B": 0.25, "C": 0.25, "unused": 0.4}
    weights = gp.np3_set_weights(
        cells, gp.NP3_NATURAL, composition=composition, trim_factor=0.0
    )
    leaf = cells[res.TRUTH_LEAF_COLUMN].to_numpy()
    totals = {name: float(weights[leaf == name].sum()) for name in ("A", "B", "C")}
    assert totals == pytest.approx({"A": 200.0, "B": 100.0, "C": 100.0})
    _, _, verdicts = _np3_rows(cells, [100], composition=composition)
    assert _scheme(verdicts, gp.NP3_NATURAL)["precision"] == pytest.approx(0.75)
    # Within the class: the test-cell reading keeps X's and Y's totals.
    assert _scheme(verdicts, gp.NP3_NATURAL_TEST_CELLS)["precision"] == pytest.approx(
        0.90
    )
    # A test type without a positive natural share would drop out of the
    # set (weight 0) and raise its precision: refused.
    for bad in ({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5, "C": 0.0}):
        with pytest.raises(ValueError, match="natural composition"):
            gp.np3_set_weights(cells, gp.NP3_NATURAL, composition=bad)
        with pytest.raises(ValueError, match="natural composition"):
            _np3_rows(cells, [100], composition=bad)
    with pytest.raises(ValueError, match="composition"):
        gp.np3_set_weights(cells, gp.NP3_NATURAL)


def test_rare_truth_types_take_their_class_weight_and_weights_are_trimmed() -> None:
    """``composition_weights``' rare-type rule and the judged-set trim apply.

    A type with fewer than ``weight_min_type_cells`` (20) calls in the set
    takes the weight of its broad class's common types; a class without
    one is capped at 10 x the set's median weight.
    """
    cells = _calls(
        100,
        [("A", "X", 300, 300, 0.95), ("B", "X", 5, 5, 0.95), ("C", "Y", 5, 0, 0.95)],
    )
    class_of = {"A": "X", "B": "X", "C": "Y"}
    raw = gp.np3_set_weights(
        cells, gp.NP3_CLASS_BALANCED, class_of=class_of, trim_factor=0.0
    )
    leaf = cells[res.TRUTH_LEAF_COLUMN].to_numpy()
    assert raw[leaf == "B"] == pytest.approx(raw[leaf == "A"][0])
    trimmed = gp.np3_set_weights(
        cells, gp.NP3_CLASS_BALANCED, class_of=class_of, trim_factor=10.0
    )
    assert trimmed[leaf == "C"].max() == pytest.approx(10 * np.median(raw))
    assert trimmed[leaf == "C"].max() < raw[leaf == "C"].max()
    assert gp.np3_set_weights(cells, gp.NP3_UNWEIGHTED).tolist() == [1.0] * 310
    assert gp.np3_set_weights(cells.iloc[:0], gp.NP3_CLASS_BALANCED).shape == (0,)


def test_depth_histogram_reweighting_is_report_only() -> None:
    """§14 NP3: values on a family dataset's depth histogram are reported only.

    The ">= 60" set holds 100 cells at 60 (all right) and 150 at 100 (140
    right): .96 unweighted, which passes target+ .95. A histogram with nine
    times the mass at 100 weighs 100-count calls 1.5 and 60-count calls .25:
    .94, which would fail; the class's NP3 record does not change.
    """
    cells = _pooled_cells(60, 100, deep_wrong=10)
    histogram = {60: 1.0, 100: 9.0, 10: 5.0}
    tested, _, verdicts = _np3_rows(cells, [60, 100], depth_histogram=histogram)
    row = _scheme(verdicts, gp.NP3_DEPTH_HISTOGRAM, ">=60")
    assert not row["scored"]
    assert row["precision"] == pytest.approx(0.94)
    assert not row["point_ok"] and not row["passed"]
    for scheme in gp.NP3_SCORED_SCHEMES:
        assert _scheme(verdicts, scheme, ">=60")["passed"]
    deepest = res.deepest_rows(cells)
    weights = gp.np3_set_weights(
        deepest.reset_index(drop=True),
        gp.NP3_DEPTH_HISTOGRAM,
        depth_histogram=histogram,
    )
    by_depth = dict(zip(deepest["depth"], weights, strict=True))
    assert by_depth == pytest.approx({60: 0.25, 100: 1.5})
    _, _, plain = _np3_rows(cells, [60, 100])
    assert gp.NP3_DEPTH_HISTOGRAM not in set(plain["scheme"])
    with_histogram = gp.validated_min_depth(verdicts, tested, [60, 100])
    assert with_histogram.equals(gp.validated_min_depth(plain, tested, [60, 100]))
    assert gp.np3_class_verdicts(with_histogram) == {("broad", "X"): True}
    # No mass on the set's bins: nothing to weigh, the scheme is reported empty.
    _, _, empty = _np3_rows(cells, [60, 100], depth_histogram={10: 1.0})
    none = _scheme(empty, gp.NP3_DEPTH_HISTOGRAM, ">=60")
    assert none["n_confident"] == 0 and math.isnan(none["precision"])
    assert not none["passed"]
    with pytest.raises(ValueError, match="depth histogram"):
        _np3_rows(cells, [60, 100], depth_histogram={60: -1.0})
    with pytest.raises(ValueError, match="depth histogram"):
        gp.np3_set_weights(cells, gp.NP3_DEPTH_HISTOGRAM)


GRID_NP3 = (10, 15, 20, 30, 60, 100)


def _set(depths: Sequence[int], cls: str = "X") -> res.GatePTestedSet:
    return res.GatePTestedSet(
        "broad", cls, tuple(depths), len(depths) > 1, 200, 0.99, 0.97
    )


def _np3_verdict_rows(
    passed: Mapping[str, bool | tuple[bool, bool]], cls: str = "X"
) -> pd.DataFrame:
    """NP3 verdict rows per set: one bool, or (natural, class-balanced)."""
    records = []
    for label, value in passed.items():
        natural, balanced = value if isinstance(value, tuple) else (value, value)
        for scheme, ok in (
            (gp.NP3_NATURAL, natural),
            (gp.NP3_CLASS_BALANCED, balanced),
            (gp.NP3_UNWEIGHTED, False),
        ):
            records.append(
                {
                    "level": "broad",
                    "class": cls,
                    "set": label,
                    "scheme": scheme,
                    "scored": scheme in gp.NP3_SCORED_SCHEMES,
                    "passed": ok,
                }
            )
    return pd.DataFrame.from_records(records)


def _min_depth(
    items: Sequence[res.GatePTestedSet] | None,
    passed: Mapping[str, bool | tuple[bool, bool]],
) -> dict[str, Any]:
    tested = {("broad", "X"): None if items is None else list(items)}
    table = gp.validated_min_depth(_np3_verdict_rows(passed), tested, GRID_NP3)
    (record,) = table.to_dict("records")
    return record


def test_validated_min_depth_walks_down_from_d_p() -> None:
    """§14 per-class records: the shallowest bin from which every bin below
    D_P is tested on its own and passes NP3, the ">= D_P" set passing too.
    """
    pooled = _set((60, 100))
    singles = [_set((depth,)) for depth in (30, 20, 10)]
    # 30 and 20 pass; 15 is not tested on its own, so 10 cannot count.
    record = _min_depth(
        [pooled, *singles], {">=60": True, "30": True, "20": True, "10": True}
    )
    assert record["tested_max_depth"] == 60 and record["validated_min_depth"] == 20
    assert record["passed"] is True and record["reason"] == ""
    assert (record["stop_depth"], record["stop_reason"]) == (15, "untested")
    # A failing bin stops the walk; the class still passes above it.
    record = _min_depth(
        [pooled, *singles], {">=60": True, "30": (True, False), "20": True, "10": True}
    )
    assert record["validated_min_depth"] == 60 and record["passed"] is True
    assert (record["stop_depth"], record["stop_reason"]) == (30, "failed")
    assert record["failed_sets"] == "30"
    # The ">= D_P" set failing fails the class, whatever passes below it.
    record = _min_depth(
        [pooled, *singles], {">=60": (False, True), "30": True, "20": True, "10": True}
    )
    assert record["passed"] is False and record["validated_min_depth"] is None
    assert record["reason"] == gp.NP3_REASON_DEEP_SET_FAILED
    assert record["tested_max_depth"] == 60
    # So does a bin at or above D_P tested on its own (both tests cover it).
    record = _min_depth(
        [pooled, _set((60,)), *singles],
        {">=60": True, "60": False, "30": True, "20": True, "10": True},
    )
    assert record["passed"] is False and record["failed_sets"] == "60"
    # No pool: D_P is the deepest bin, tested on its own.
    record = _min_depth(
        [_set((100,)), _set((60,)), _set((30,))],
        {"100": True, "60": True, "30": True},
    )
    assert record["tested_max_depth"] == 100 and record["validated_min_depth"] == 30
    assert (record["stop_depth"], record["stop_reason"]) == (20, "untested")
    # Every bin below D_P tested on its own and passing: the walk reaches the
    # shallowest grid bin and stops nowhere.
    record = _min_depth(
        [pooled, *[_set((depth,)) for depth in (30, 20, 15, 10)]],
        {">=60": True, "30": True, "20": True, "15": True, "10": True},
    )
    assert record["validated_min_depth"] == 10 and record["passed"] is True
    assert record["stop_depth"] is None and record["stop_reason"] == ""
    # A walk down to the shallowest grid bin has nowhere left to stop.
    record = _min_depth([_set((10, 15, 20))], {">=10": True})
    assert record["validated_min_depth"] == 10 and record["stop_depth"] is None
    # Not evaluable: no tested set.
    record = _min_depth(None, {})
    assert record["passed"] is None and record["tested_max_depth"] is None
    assert record["reason"] == gp.NP3_REASON_NOT_EVALUABLE


def test_np3_class_verdicts_feed_every_member_verdict() -> None:
    tested: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
        ("broad", "X"): [_set((60, 100)), _set((30,))],
        ("broad", "Y"): None,
    }
    passing = gp.np3_class_verdicts(
        gp.validated_min_depth(
            _np3_verdict_rows({">=60": True, "30": False}), tested, GRID_NP3
        )
    )
    failing = gp.np3_class_verdicts(
        gp.validated_min_depth(
            _np3_verdict_rows({">=60": False, "30": True}), tested, GRID_NP3
        )
    )
    assert passing == {("broad", "X"): True, ("broad", "Y"): None}
    assert failing == {("broad", "X"): False, ("broad", "Y"): None}
    combined = res.every_member_verdict({"R1@0": passing, "R1@6": failing})
    assert combined[("broad", "X")]["status"] == res.GATE_P_FAILED
    assert combined[("broad", "Y")]["status"] == res.GATE_P_NOT_EVALUABLE


def test_np3_class_verdicts_keep_not_evaluable_after_a_csv_round_trip() -> None:
    """A not-evaluable class read back from CSV (``passed`` nan) is no pass."""
    tested: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
        ("broad", "X"): [_set((60, 100))],
        ("broad", "Y"): None,
        ("broad", "Z"): [_set((60, 100), cls="Z")],
    }
    verdicts = pd.concat(
        [
            _np3_verdict_rows({">=60": True}),
            _np3_verdict_rows({">=60": False}, cls="Z"),
        ],
        ignore_index=True,
    )
    table = gp.validated_min_depth(verdicts, tested, GRID_NP3)
    expected = {("broad", "X"): True, ("broad", "Y"): None, ("broad", "Z"): False}
    assert gp.np3_class_verdicts(table) == expected
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    buffer.seek(0)
    read = pd.read_csv(buffer)
    assert pd.isna(read.loc[read["class"] == "Y", "passed"].iloc[0])
    assert gp.np3_class_verdicts(read) == expected


def test_np3_inputs_that_lack_rows_or_verdicts_raise() -> None:
    cells = _calls(100, [("A", "X", 200, 200, 0.95)])
    decisions = _decisions([100])
    tested = res.gate_p_tested_sets(cells, decisions, regime="provisional")
    with pytest.raises(ValueError, match="no rows"):
        gp.np3_set_stats(
            {("D1", 0): cells.iloc[:0]},
            decisions,
            tested,
            default_group=None,
            composition=NATURAL,
            settings=_np3(),
        )
    with pytest.raises(ValueError, match="belongs to"):
        gp.np3_set_stats(
            {("D1", 0): cells},
            decisions,
            {("broad", "Y"): [_set((100,))]},
            default_group=None,
            composition=NATURAL,
            settings=_np3(),
        )
    with pytest.raises(ValueError, match="scheme"):
        gp.np3_set_weights(cells, "dataset")
    with pytest.raises(ValueError, match="scheme"):
        gp.np3_test_cell_weights(cells, gp.NP3_NATURAL)
    with pytest.raises(ValueError, match="truth class"):
        gp.np3_test_cell_weights(
            pd.concat([cells, cells.iloc[:1].assign(truth_parent="Z")]),
            gp.NP3_CLASS_BALANCED_TEST_CELLS,
        )
    stats = gp.np3_set_stats(
        {("D1", 0): cells},
        decisions,
        tested,
        default_group=None,
        composition=NATURAL,
        settings=_np3(),
    )
    with pytest.raises(ValueError, match="no precision target"):
        gp.np3_verdicts(stats.assign(level="unknown"), AnnotationThresholds(), _np3())
    verdicts = gp.np3_verdicts(stats, AnnotationThresholds(), _np3())
    with pytest.raises(ValueError, match="no NP3 verdict"):
        gp.validated_min_depth(
            verdicts[verdicts["scheme"] != gp.NP3_CLASS_BALANCED], tested, [100]
        )
    with pytest.raises(ValueError, match="grid"):
        gp.validated_min_depth(verdicts, tested, [10, 30])
    two_pools = {("broad", "X"): [_set((60, 100)), _set((30, 60, 100))]}
    with pytest.raises(ValueError, match="pooled"):
        gp.validated_min_depth(
            _np3_verdict_rows({">=60": True, ">=30": True}), two_pools, GRID_NP3
        )


def test_np3_settings_follow_the_config() -> None:
    np3 = gp.Np3Settings.from_config(AnnotationResolvabilityConfig())
    assert np3.min_coverage == pytest.approx(0.30)
    assert np3.weight_min_type_cells == 20
    assert np3.weight_trim_factor == pytest.approx(10.0)
    custom = gp.Np3Settings.from_config(
        AnnotationResolvabilityConfig(
            gate_p_min_coverage=0.4, weight_min_type_cells=5, weight_trim_factor=0.0
        )
    )
    assert (custom.min_coverage, custom.weight_min_type_cells) == (0.4, 5)
    assert custom.weight_trim_factor == 0.0
    for bad in (
        {"min_coverage": 1.5},
        {"weight_min_type_cells": 0},
        {"weight_trim_factor": -1.0},
    ):
        fields: dict[str, Any] = {
            "min_coverage": 0.3,
            "weight_min_type_cells": 20,
            "weight_trim_factor": 10.0,
            **bad,
        }
        with pytest.raises(ValueError, match="Np3Settings"):
            gp.Np3Settings(**fields)


# --------------------------------------------------------------------------
# NP5: resolvability consistency (§14 NP5)

GRID_NP5 = (10, 15, 30, 60, 120, 250)


def _np5() -> gp.Np5Settings:
    return gp.Np5Settings.from_config(AnnotationResolvabilityConfig())


def _np5_decisions(
    emitted: Sequence[int],
    *,
    cls: str = "X",
    n_test: int | Mapping[int, int] = 100,
    extrapolated: Sequence[int] = (),
    grid: Sequence[int] = GRID_NP5,
) -> pd.DataFrame:
    """Synthetic decisions of one class: provisional emits ``emitted``.

    The trust regime emits every bin and marks nothing extrapolated, so a
    function that reads another regime than the one it is asked for shows.
    """
    records = []
    for regime, on, marked in (
        ("provisional", set(emitted), set(extrapolated)),
        ("trust", set(grid), set()),
    ):
        for depth in grid:
            records.append(
                {
                    "regime": regime,
                    "level": "broad",
                    "class": cls,
                    "depth": depth,
                    "status": res.STATUS_EMITTED
                    if depth in on
                    else res.STATUS_NOT_RESOLVABLE,
                    "n_test": n_test
                    if isinstance(n_test, int)
                    else int(n_test.get(depth, 0)),
                    "extrapolated": depth in marked,
                }
            )
    return pd.DataFrame.from_records(records)


def _agreement(
    base: pd.DataFrame,
    replicates: Mapping[gp.ReplicateKey, pd.DataFrame],
    cls: str = "X",
) -> dict[tuple[str, int], pd.Series]:
    table = gp.np5_decision_agreement(base, replicates, _np5())
    assert list(table.columns) == list(gp.NP5_AGREEMENT_COLUMNS)
    rows = table[table["class"] == cls]
    return {(str(row["group"]), int(row["seed"])): row for _, row in rows.iterrows()}


def test_np5_flip_at_the_boundary_bin_is_allowed_and_two_bins_away_fails() -> None:
    """§14 NP5: the re-derived emission decisions agree with the base run at
    every bin with >= 50 test cells, except at most the bin adjacent to the
    emission boundary (the base emits from 60 counts: 30 and 60 flank it).
    """
    base = _np5_decisions((60, 120, 250))
    replicates = {
        ("D1", 0): base,
        # The boundary moves one bin shallower, or one bin deeper.
        ("D1", 1): _np5_decisions((30, 60, 120, 250)),
        ("D2", 0): _np5_decisions((120, 250)),
        # Two bins flip: "at most the bin" is one bin.
        ("D2", 1): _np5_decisions((15, 30, 60, 120, 250)),
        # One flip two bins away from the boundary, or deeper than it.
        ("D3", 0): _np5_decisions((15, 60, 120, 250)),
        ("D3", 1): _np5_decisions((60, 250)),
        # Both bins flanking the boundary flip (30 and 60): each is adjacent
        # to it, but "at most the bin" is one bin.
        ("D4", 0): _np5_decisions((30, 120, 250)),
    }
    rows = _agreement(base, replicates)
    assert {row["boundary_depths"] for row in rows.values()} == {"30;60"}
    assert {
        key: (row["flipped_depths"], bool(row["passed"])) for key, row in rows.items()
    } == {
        ("D1", 0): ("", True),
        ("D1", 1): ("30", True),
        ("D2", 0): ("60", True),
        ("D2", 1): ("15;30", False),
        ("D3", 0): ("15", False),
        ("D3", 1): ("120", False),
        ("D4", 0): ("30;60", False),
    }
    assert {int(row["n_compared"]) for row in rows.values()} == {6}
    assert int(rows[("D2", 1)]["n_flipped"]) == 2
    assert int(rows[("D4", 0)]["n_flipped"]) == 2


def test_np5_agreement_compares_the_bins_where_either_run_has_50_test_cells() -> None:
    thin = {depth: 100 for depth in GRID_NP5} | {250: 49}
    base = _np5_decisions((60, 120, 250), n_test=thin)
    rows = _agreement(
        base,
        {
            # Neither run has 50 test cells at 250: the flip there is ignored.
            ("D1", 0): _np5_decisions((60, 120), n_test=thin),
            # The replicate has 50: the bin is compared, and the flip fails.
            ("D2", 0): _np5_decisions((60, 120), n_test=thin | {250: 50}),
        },
    )
    assert rows[("D1", 0)]["passed"] and int(rows[("D1", 0)]["n_compared"]) == 5
    assert rows[("D1", 0)]["flipped_depths"] == ""
    assert not rows[("D2", 0)]["passed"] and rows[("D2", 0)]["flipped_depths"] == "250"
    # The base has 50 and the replicate none: compared as well.
    base = _np5_decisions((60, 120, 250), n_test=thin | {250: 50})
    rows = _agreement(base, {("D1", 0): _np5_decisions((60, 120), n_test=thin)})
    assert not rows[("D1", 0)]["passed"]


def test_np5_a_class_missing_from_one_run_is_not_emitted_there() -> None:
    base = _np5_decisions((60, 120, 250))
    replicate = _np5_decisions((120, 250), cls="Y")
    table = gp.np5_decision_agreement(base, {("D1", 0): replicate}, _np5())
    rows = {str(row["class"]): row for _, row in table.iterrows()}
    assert set(rows) == {"X", "Y"}
    # X is emitted from 60 in the base only; Y in the replicate only, so the
    # base has no boundary for Y.
    assert rows["X"]["flipped_depths"] == "60;120;250" and not rows["X"]["passed"]
    assert rows["Y"]["flipped_depths"] == "120;250" and not rows["Y"]["passed"]
    assert rows["Y"]["boundary_depths"] == ""


def test_np5_boundaries_are_status_changes_inside_the_grid() -> None:
    """The grid's edges are no boundary: a base emitting every bin allows no
    flip; a base that is not monotone in depth has a boundary at every
    status change.
    """
    base = _np5_decisions(GRID_NP5)
    rows = _agreement(
        base,
        {
            ("D1", 0): _np5_decisions(GRID_NP5[1:]),
            ("D1", 1): _np5_decisions(GRID_NP5[:-1]),
        },
    )
    assert all(not row["passed"] for row in rows.values())
    assert {row["boundary_depths"] for row in rows.values()} == {""}
    base = _np5_decisions((30, 120, 250))
    rows = _agreement(
        base,
        {
            ("D1", 0): _np5_decisions((30, 60, 120, 250)),
            ("D1", 1): _np5_decisions((15, 30, 120, 250)),
        },
    )
    assert {row["boundary_depths"] for row in rows.values()} == {"15;30;60;120"}
    assert all(row["passed"] for row in rows.values())
    # One flip at each of two boundaries (15 and 60): two flips fail.
    rows = _agreement(base, {("D1", 0): _np5_decisions((15, 30, 60, 120, 250))})
    assert rows[("D1", 0)]["flipped_depths"] == "15;60"
    assert not rows[("D1", 0)]["passed"]


def test_np5_a_bin_only_the_replicate_holds_is_compared() -> None:
    """The grid is the union of the two tables' depths: a bin the base's
    table lacks is not emitted there, and is compared when the replicate has
    its 50 test cells.
    """
    shallow = GRID_NP5[2:]
    base = _np5_decisions((60, 120, 250), grid=shallow)
    rows = _agreement(
        base,
        {
            ("D1", 0): _np5_decisions((10, 60, 120, 250)),
            ("D1", 1): _np5_decisions((60, 120, 250)),
        },
    )
    assert {int(row["n_bins"]) for row in rows.values()} == {6}
    assert {int(row["n_compared"]) for row in rows.values()} == {6}
    # 10 is no boundary bin of the base (it emits from 60 counts).
    assert rows[("D1", 0)]["flipped_depths"] == "10"
    assert not rows[("D1", 0)]["passed"]
    assert rows[("D1", 1)]["passed"] and rows[("D1", 1)]["flipped_depths"] == ""


def test_np5_agreement_inputs_that_mix_or_lack_decisions_raise() -> None:
    base = _np5_decisions((60, 120, 250))
    replicates = {("D1", 0): base}
    with pytest.raises(ValueError, match="more than once"):
        gp.np5_decision_agreement(pd.concat([base, base]), replicates, _np5())
    with pytest.raises(ValueError, match="more than once"):
        gp.np5_decision_agreement(base, {("D1", 0): pd.concat([base, base])}, _np5())
    with pytest.raises(ValueError, match="no decisions of the regime"):
        gp.np5_decision_agreement(base[base["regime"] == "trust"], replicates, _np5())
    with pytest.raises(ValueError, match="no decisions of the regime"):
        gp.np5_decision_agreement(
            base, {("D1", 0): base[base["regime"] == "trust"]}, _np5()
        )
    with pytest.raises(ValueError, match="no replicates"):
        gp.np5_decision_agreement(base, {}, _np5())
    with pytest.raises(ValueError, match="n_test"):
        gp.np5_decision_agreement(base.drop(columns="n_test"), replicates, _np5())


def _thresholds(
    values: Mapping[tuple[str, int], float | None],
    *,
    unfitted: Sequence[tuple[str, int]] = (),
    label: str = "60",
    cls: str = "X",
) -> pd.DataFrame:
    """Synthetic ``np5_set_thresholds`` rows of one tested set."""
    records = []
    for (group, seed), value in values.items():
        fitted = (group, seed) not in unfitted
        records.append(
            {
                "level": "broad",
                "class": cls,
                "set": label,
                "pooled": label.startswith(">="),
                "set_min_depth": int(label.removeprefix(">=")),
                "group": group,
                "seed": seed,
                "n_called": 200,
                "n_fit": 100 if fitted else 20,
                "fitted": fitted,
                "target": 0.95,
                "t_star": math.nan if value is None else value,
                "threshold": math.nan if value is None else value,
                "threshold_source": None
                if value is None
                else res.THRESHOLD_SOURCE_LOCAL,
            }
        )
    return pd.DataFrame.from_records(records, columns=list(gp.NP5_THRESHOLD_COLUMNS))


KEYS_NP5 = [(group, seed) for group in GROUPS for seed in SEEDS]


def _values(*values: float | None) -> dict[tuple[str, int], float | None]:
    """One threshold per replicate of ``KEYS_NP5``, in key order."""
    return dict(zip(KEYS_NP5, values, strict=True))


def _spread(table: pd.DataFrame) -> pd.Series:
    result = gp.np5_tstar_spread(table, _np5())
    assert list(result.columns) == list(gp.NP5_SPREAD_COLUMNS)
    assert len(result) == 1
    return result.iloc[0]


def test_np5_tstar_spread_of_0051_fails_and_005_passes() -> None:
    """§14 NP5: each t* varies <= 0.05 across replicates at tested sets."""
    row = _spread(_thresholds(_values(0.90, 0.92, 0.95, 0.93, 0.91, 0.94)))
    assert row["spread"] == pytest.approx(0.05, abs=1e-12)
    assert row["passed"] and row["evaluable"] and int(row["n_thresholds"]) == 6
    assert row["max_spread"] == pytest.approx(0.05)
    row = _spread(_thresholds(_values(0.90, 0.92, 0.951, 0.93, 0.91, 0.94)))
    assert row["spread"] == pytest.approx(0.051, abs=1e-12)
    assert not row["passed"]
    assert (row["threshold_min"], row["threshold_max"]) == (0.90, 0.951)


def test_np5_a_fitted_replicate_without_a_threshold_fails_the_set() -> None:
    values = _values(0.90, 0.92, None, 0.93, 0.91, 0.94)
    # D2/0's fit never reaches the target: it would not emit at all.
    row = _spread(_thresholds(values))
    assert row["missing"] == "D2/0" and not row["passed"]
    assert row["spread"] == pytest.approx(0.04)
    # Without a fit (too few fit-half calls) it is left out of the spread.
    row = _spread(_thresholds(values, unfitted=[("D2", 0)]))
    assert row["passed"] and row["missing"] == "" and row["unfitted"] == "D2/0"
    assert int(row["n_fitted"]) == 5 and int(row["n_thresholds"]) == 5
    # Fewer than two thresholds: the spread is not evaluable (reported).
    lonely = {key: (0.90 if key == ("D1", 0) else None) for key in KEYS_NP5}
    row = _spread(_thresholds(lonely, unfitted=KEYS_NP5[1:]))
    assert row["passed"] and not row["evaluable"] and math.isnan(row["spread"])


def _threshold_cells(seed: int = 3) -> pd.DataFrame:
    """Broad X calls at 10 (600 cells), 30 (400) and 100 (60 of the 400).

    ``decide`` judges 10 and 30 on their own, and pools 100 into ">= 30"
    (60 cells at 100 hold too few check-half calls on their own).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for depth, ids in (
        (10, [f"a{index}" for index in range(600)]),
        (30, [f"b{index}" for index in range(400)]),
        (100, [f"b{index}" for index in range(60)]),
    ):
        n = len(ids)
        bp = np.round(0.70 + 0.30 * rng.random(n), 3)
        correct = np.where(bp >= 0.85, rng.random(n) < 0.995, rng.random(n) < 0.6)
        frame = bin_cells(bp, correct, depth=depth)
        frame["cell_id"] = ids
        frame["sim_id"] = [f"{cell}|D{depth}" for cell in ids]
        frame["half"] = res.cell_split_half(ids)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _decision_row(decisions: pd.DataFrame, depth: int) -> pd.Series:
    rows = decisions[
        (decisions["regime"] == "provisional") & (decisions["depth"] == depth)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_np5_set_thresholds_re_derive_t_star_as_decide_does() -> None:
    """A tested set's t* is fitted on the set's own rows of each replicate
    (the shared membership rule), as ``decide`` fits a bin or a pooled set.
    """
    cells = _threshold_cells()
    depths = [10, 30, 100]
    decisions = res.decide(cells, [BROAD], depths, settings())
    own_10, own_30, pooled_100 = (_decision_row(decisions, d) for d in depths)
    assert not own_10["pooled"] and not own_30["pooled"]
    assert pooled_100["pooled"] and pooled_100["pool_min_depth"] == 30
    tested = {
        ("broad", "X"): [_set((30, 100)), _set((30,)), _set((10,))],
    }
    replicate = cells.copy()
    replicate["seed"] = 1
    table = gp.np5_set_thresholds(
        {("D1", 0): cells, ("D1", 1): replicate}, tested, [BROAD], settings()
    )
    assert list(table.columns) == list(gp.NP5_THRESHOLD_COLUMNS)
    assert len(table) == 6
    expected = {
        ">=30": pooled_100["t_star"],
        "30": own_30["t_star"],
        "10": own_10["t_star"],
    }
    assert all(value is not None and not pd.isna(value) for value in expected.values())
    for _, row in table.iterrows():
        assert row["t_star"] == pytest.approx(expected[str(row["set"])], abs=1e-12)
        assert row["threshold"] == row["t_star"] and row["fitted"]
        assert row["threshold_source"] == res.THRESHOLD_SOURCE_LOCAL
    targets = dict(zip(table["set"], table["target"], strict=True))
    assert targets == {">=30": 0.97, "30": 0.97, "10": 0.97}
    # The same cells as two members: member= keeps one member's rows.
    members = pd.concat(
        [cells.assign(member="m0"), _threshold_cells(seed=9).assign(member="m1")],
        ignore_index=True,
    )
    only_m0 = gp.np5_set_thresholds(
        {("D1", 0): members}, tested, [BROAD], settings(), member="m0"
    )
    assert only_m0["t_star"].tolist() == table[table["seed"] == 0]["t_star"].tolist()


def _mixed_threshold_cells(seed: int = 3) -> pd.DataFrame:
    """``_threshold_cells`` with other classes' calls and sinks in X's scope.

    Y calls, wrong at high bp more often than not, share X's scopes: the
    test cells c0-c299 at 30 and 100, and b60-b89 at 100, whose deepest
    call is then Y's (they leave X's pooled ">= 30" set). Sinks (no call,
    bp 0.99) of the test cells s0-s99 lie at 100.
    """
    rng = np.random.default_rng(seed + 100)
    frames = [_threshold_cells(seed)]
    for depth, ids in (
        (30, [f"c{index}" for index in range(300)]),
        (100, [f"c{index}" for index in range(300)]),
        (100, [f"b{index}" for index in range(60, 90)]),
    ):
        n = len(ids)
        bp = np.round(0.70 + 0.30 * rng.random(n), 3)
        frame = bin_cells(bp, rng.random(n) < 0.3, cls="Y", depth=depth)
        frame["cell_id"] = ids
        frame["sim_id"] = [f"{cell}|D{depth}" for cell in ids]
        frame["half"] = res.cell_split_half(ids)
        frames.append(frame)
    sinks = [f"s{index}" for index in range(100)]
    frame = bin_cells(
        np.full(100, 0.99), np.zeros(100, dtype=bool), depth=100, truth_parent="Y"
    )
    frame["parent"] = None
    frame["call"] = None
    frame["cell_id"] = sinks
    frame["sim_id"] = [f"{cell}|D100" for cell in sinks]
    frame["half"] = res.cell_split_half(sinks)
    frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_np5_set_thresholds_fit_the_class_calls_of_a_mixed_scope() -> None:
    """A tested set's t* is fitted on the class's calls of its scope only:
    other classes' calls and sinks share the scope (the bin's rows, or each
    test cell's deepest row) and stay out of the fit, as in ``decide``.
    """
    cells = _mixed_threshold_cells()
    depths = [10, 30, 100]
    decisions = res.decide(cells, [BROAD], depths, settings())
    rows = decisions[
        (decisions["regime"] == "provisional") & (decisions["class"] == "X")
    ].set_index("depth")
    assert bool(rows.loc[100, "pooled"]) and rows.loc[100, "pool_min_depth"] == 30
    tested = {("broad", "X"): [_set((30, 100)), _set((30,)), _set((10,))]}
    table = gp.np5_set_thresholds({("D1", 0): cells}, tested, [BROAD], settings())
    by_set = table.set_index("set")
    expected = {">=30": 100, "30": 30, "10": 10}
    for label, depth in expected.items():
        assert not pd.isna(rows.loc[depth, "t_star"])
        assert by_set.loc[label, "t_star"] == pytest.approx(
            rows.loc[depth, "t_star"], abs=1e-12
        )
    # The single bins' fit counts are decide's; the pooled set holds the
    # 370 X test cells whose deepest call is X's (b60-b89's is Y's at 100).
    assert int(by_set.loc["30", "n_fit"]) == int(rows.loc[30, "n_fit"])
    assert int(by_set.loc["10", "n_fit"]) == int(rows.loc[10, "n_fit"])
    assert int(by_set.loc[">=30", "n_called"]) == 370
    assert int(by_set.loc["30", "n_called"]) == 400


def test_np5_refuses_a_replicate_of_check_half_rows_alone() -> None:
    """NP5 re-derives each replicate on its own fit half, so it takes every
    table in full. ``held_out_replicates`` keeps the default group's check
    half only: given that output, the group would have no fit and be left
    out of the t* range as unfitted, so it raises.
    """
    cells = _threshold_cells()
    other = _threshold_cells(seed=5)
    other["cell_id"] = "o" + other["cell_id"].astype(str)
    other["sim_id"] = "o" + other["sim_id"].astype(str)
    replicates = {("D1", 0): cells, ("D2", 0): other}
    tested = {("broad", "X"): [_set((10,))]}
    full = gp.np5_set_thresholds(replicates, tested, [BROAD], settings())
    assert full["fitted"].all() and full["group"].tolist() == ["D1", "D2"]
    held_out = gp.held_out_replicates(replicates, default_group="D1")
    assert set(held_out[("D1", 0)]["half"]) == {1}
    with pytest.raises(ValueError, match="no fit-half rows"):
        gp.np5_set_thresholds(held_out, tested, [BROAD], settings())
    with pytest.raises(ValueError, match="no fit-half rows"):
        gp.np5_rederive(held_out[("D1", 0)], [BROAD], [10, 30, 100], settings())
    n = 400
    members = pd.concat(
        [
            member_rows("m0", np.full(n, 0.95), pattern(n, 0.99)),
            member_rows("m1", np.full(n, 0.95), pattern(n, 0.96)),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="member='m1'.*no fit-half rows"):
        gp.np5_rederive_ensemble(
            members[(members["member"] == "m0") | (members["half"] == 1)],
            [BROAD],
            [100],
            settings(),
            res.EnsembleSettings(),
            members=["m0", "m1"],
        )


def test_np5_set_thresholds_apply_the_saturated_cap_for_version_7() -> None:
    """v7.8: a fitted set without t* whose fit-half calls are saturated is
    judged at the cap (``threshold_source = saturated_cap``), as ``decide``
    does with ``saturated_bp_share``.
    """
    cells = bin_cells(np.ones(400), np.arange(400) % 10 != 0, depth=100)
    tested = {("broad", "X"): [_set((100,))]}
    v6 = gp.np5_set_thresholds({("D1", 0): cells}, tested, [BROAD], settings())
    assert v6.iloc[0]["fitted"] and math.isnan(v6.iloc[0]["threshold"])
    assert v6.iloc[0]["threshold_source"] is None
    v7 = gp.np5_set_thresholds(
        {("D1", 0): cells}, tested, [BROAD], settings(), saturated_bp_share=0.9
    )
    decided = _decision_row(
        res.decide(cells, [BROAD], [100], settings(), saturated_bp_share=0.9), 100
    )
    assert decided["threshold_source"] == res.THRESHOLD_SOURCE_SATURATED
    assert v7.iloc[0]["threshold"] == pytest.approx(decided["threshold"])
    assert v7.iloc[0]["threshold_source"] == res.THRESHOLD_SOURCE_SATURATED
    assert math.isnan(v7.iloc[0]["t_star"])
    # Too few fit-half calls: no fit, no threshold.
    few = gp.np5_set_thresholds(
        {("D1", 0): cells.iloc[:90]}, tested, [BROAD], settings()
    )
    assert not few.iloc[0]["fitted"] and int(few.iloc[0]["n_fit"]) == 45


def test_np5_set_threshold_inputs_that_mix_or_lack_rows_raise() -> None:
    cells = _threshold_cells()
    tested = {("broad", "X"): [_set((10,))]}
    with pytest.raises(ValueError, match="validated"):
        gp.np5_set_thresholds(
            {("D1", 0): cells}, tested, [BROAD], settings(), regime="validated"
        )
    with pytest.raises(ValueError, match="no level metadata"):
        gp.np5_set_thresholds({("D1", 0): cells}, tested, [], settings())
    with pytest.raises(ValueError, match="no replicates"):
        gp.np5_set_thresholds({}, tested, [BROAD], settings())
    with pytest.raises(ValueError, match="no rows after the filters"):
        gp.np5_set_thresholds(
            {("D1", 0): cells}, tested, [BROAD], settings(), recipe="clean"
        )
    with pytest.raises(res.ResolvabilityError, match="more than once"):
        gp.np5_set_thresholds(
            {("D1", 0): pd.concat([cells, cells])}, tested, [BROAD], settings()
        )
    with pytest.raises(ValueError, match="empty list"):
        gp.np5_set_thresholds(
            {("D1", 0): cells}, {("broad", "X"): []}, [BROAD], settings()
        )


def test_np5_extrapolated_share_of_051_fails_and_050_passes() -> None:
    """§14 NP5: a class is not validated at L if its cells at the family's
    expected depth would be > 50% ``resolvability_extrapolated``.
    """
    decisions = _np5_decisions((30, 60, 120, 250), extrapolated=(120, 250))

    def share(depths: float | Sequence[float]) -> pd.Series:
        table = gp.np5_extrapolated_share(decisions, {"X": depths}, GRID_NP5, _np5())
        assert list(table.columns) == list(gp.NP5_EXTRAPOLATED_COLUMNS)
        assert len(table) == 1
        return table.iloc[0]

    row = share([300.0] * 51 + [40.0] * 49)
    assert row["extrapolated_share"] == pytest.approx(0.51) and not row["passed"]
    assert int(row["n_depths"]) == 100 and row["expected_bin"] == 250
    # 100 counts take the 60 bin, which is not extrapolated.
    row = share([300.0] * 50 + [100.0] * 50)
    assert row["extrapolated_share"] == pytest.approx(0.50) and row["passed"]
    assert row["max_share"] == pytest.approx(0.5)
    # D9: the share of the class's cells above the grid's deepest bin.
    assert row["above_grid_share"] == pytest.approx(0.5)
    # One expected depth per class (the label-free fallback of a family
    # without a profile): its cells all take that depth's bin, so the share
    # is 0 or 1.
    row = share(130.0)
    assert row["extrapolated_share"] == 1.0 and not row["passed"]
    assert row["expected_bin"] == 120 and row["expected_depth"] == 130.0
    row = share(70.0)
    assert row["extrapolated_share"] == 0.0 and row["passed"]
    assert row["expected_bin"] == 60 and row["depth_source"] == "class"
    # Cells below the grid are not emitted, so not extrapolated.
    row = share([5.0] * 60 + [250.0] * 40)
    assert row["extrapolated_share"] == pytest.approx(0.4) and row["passed"]
    assert row["expected_bin"] is None
    # D9: a depth at the grid's deepest bin (250) is in it, not above it.
    assert row["above_grid_share"] == 0.0
    row = share([250.0] * 40 + [251.0] * 60)
    assert row["above_grid_share"] == pytest.approx(0.6)


def test_np5_extrapolated_share_uses_the_class_profile_shares() -> None:
    """Plan §8.3 v7.5: for a family with a per-class profile, NP5's
    "> 50% extrapolated" test "uses the class's profile shares". The
    class's profile median alone (one depth) would pass a class whose
    profile fails.
    """
    # The frozen decisions mark 30 (a monotone-filled bin), 120 and 250
    # extrapolated; 60 is emitted on its own verdict.
    decisions = _np5_decisions((30, 60, 120, 250), extrapolated=(30, 120, 250))
    profile = [20.0] * 10 + [40.0] * 20 + [80.0] * 25 + [150.0] * 25 + [300.0] * 20
    table = gp.np5_extrapolated_share(decisions, {"X": profile}, GRID_NP5, _np5())
    row = table.iloc[0]
    # 40 counts take the 30 bin, 150 the 120 bin and 300 the 250 bin.
    assert row["extrapolated_share"] == pytest.approx(0.65) and not row["passed"]
    # The profile median is reported as the class's expected depth.
    assert row["expected_depth"] == 80.0 and row["expected_bin"] == 60
    assert row["above_grid_share"] == pytest.approx(0.20)
    assert int(row["n_depths"]) == 100 and row["depth_source"] == "class"
    # The median alone falls in the 60 bin: share 0, a pass (the loosening).
    median = gp.np5_extrapolated_share(
        decisions, {"X": float(np.median(profile))}, GRID_NP5, _np5()
    ).iloc[0]
    assert median["extrapolated_share"] == 0.0 and median["passed"]
    assert median["above_grid_share"] == 0.0


def test_np5_extrapolated_share_reads_one_regime_and_the_default_depth() -> None:
    decisions = pd.concat(
        [
            _np5_decisions((30, 60, 120, 250), extrapolated=(120, 250)),
            _np5_decisions((60, 120, 250), cls="Y", extrapolated=(250,)),
        ],
        ignore_index=True,
    )
    table = gp.np5_extrapolated_share(
        decisions, {"X": 130.0}, GRID_NP5, _np5(), default_depth=130.0
    )
    rows = {str(row["class"]): row for _, row in table.iterrows()}
    assert rows["X"]["depth_source"] == "class" and not rows["X"]["passed"]
    # Y has no depth of its own: the default (D8: the overall median).
    assert rows["Y"]["depth_source"] == "default" and rows["Y"]["passed"]
    assert rows["Y"]["expected_bin"] == 120
    trust = gp.np5_extrapolated_share(
        decisions, {"X": 130.0, "Y": 130.0}, GRID_NP5, _np5(), regime="trust"
    )
    assert trust["passed"].all()
    with pytest.raises(ValueError, match="no expected depth"):
        gp.np5_extrapolated_share(decisions, {"X": 130.0}, GRID_NP5, _np5())
    with pytest.raises(ValueError, match="empty depth grid"):
        gp.np5_extrapolated_share(decisions, {}, [], _np5(), default_depth=70.0)
    for bad in (math.nan, -1.0, []):
        with pytest.raises(ValueError, match="finite"):
            gp.np5_extrapolated_share(
                decisions, {"X": bad, "Y": 70.0}, GRID_NP5, _np5()
            )
    with pytest.raises(ValueError, match="not in the decisions"):
        gp.np5_extrapolated_share(
            decisions[decisions["depth"] != 120],
            {"X": 130.0, "Y": 70.0},
            GRID_NP5,
            _np5(),
        )
    with pytest.raises(ValueError, match="more than once"):
        gp.np5_extrapolated_share(
            pd.concat([decisions, decisions]), {}, GRID_NP5, _np5(), default_depth=70.0
        )


def test_np5_class_verdicts_combine_its_three_parts() -> None:
    tested: dict[tuple[str, str], list[res.GatePTestedSet] | None] = {
        ("broad", "X"): [_set((120, 250)), _set((60,))],
        ("broad", "Y"): None,
    }
    base = pd.concat(
        [
            _np5_decisions((60, 120, 250), extrapolated=(250,)),
            _np5_decisions((60, 120, 250), cls="Y"),
        ],
        ignore_index=True,
    )
    agreement = gp.np5_decision_agreement(
        base, {("D1", 0): base, ("D2", 0): base}, _np5()
    )
    good = _values(0.90, 0.92, 0.95, 0.93, 0.91, 0.94)
    bad = _values(0.90, 0.92, 0.951, 0.93, 0.91, 0.94)
    spread = gp.np5_tstar_spread(
        pd.concat([_thresholds(good, label=">=120"), _thresholds(good)]), _np5()
    )
    extrapolated = gp.np5_extrapolated_share(
        base, {}, GRID_NP5, _np5(), default_depth=130.0
    )
    verdicts = gp.np5_class_verdicts(agreement, spread, extrapolated, tested)
    assert verdicts == {("broad", "X"): True, ("broad", "Y"): None}
    # Each part fails the class on its own.
    flipped = gp.np5_decision_agreement(
        base, {("D1", 0): base, ("D2", 0): _np5_decisions((15, 60, 120, 250))}, _np5()
    )
    spread_bad = gp.np5_tstar_spread(
        pd.concat([_thresholds(good, label=">=120"), _thresholds(bad)]), _np5()
    )
    deep = gp.np5_extrapolated_share(base, {}, GRID_NP5, _np5(), default_depth=300.0)
    for parts in (
        (flipped, spread, extrapolated),
        (agreement, spread_bad, extrapolated),
        (agreement, spread, deep),
    ):
        assert gp.np5_class_verdicts(*parts, tested)[("broad", "X")] is False
    combined = res.every_member_verdict(
        {
            "R1@0": verdicts,
            "R1@6": gp.np5_class_verdicts(agreement, spread, deep, tested),
        }
    )
    assert combined[("broad", "X")]["status"] == res.GATE_P_FAILED
    assert combined[("broad", "X")]["failed_members"] == ["R1@6"]
    assert combined[("broad", "Y")]["status"] == res.GATE_P_NOT_EVALUABLE
    # A tested (level, class) or set without rows raises.
    with pytest.raises(ValueError, match="agreement"):
        gp.np5_class_verdicts(
            agreement[agreement["class"] != "X"], spread, extrapolated, tested
        )
    with pytest.raises(ValueError, match="t\\* spread"):
        gp.np5_class_verdicts(
            agreement, spread[spread["set"] != "60"], extrapolated, tested
        )
    with pytest.raises(ValueError, match="extrapolated"):
        gp.np5_class_verdicts(
            agreement, spread, extrapolated[extrapolated["class"] != "X"], tested
        )
    unknown = agreement.astype({"passed": object})
    unknown.loc[unknown["class"] == "X", "passed"] = None
    with pytest.raises(ValueError, match="passed"):
        gp.np5_class_verdicts(unknown, spread, extrapolated, tested)


def test_np5_rederive_reproduces_decide_at_any_mapping_seed() -> None:
    """Each replicate's decisions are re-derived from its own rows, at its
    own mapping seed (§14 NP5: "re-derived in each replicate").
    """
    cells = _threshold_cells()
    depths = [10, 30, 100]
    expected = res.decide(cells, [BROAD], depths, settings())
    seed_1 = cells.assign(seed=1)
    pd.testing.assert_frame_equal(
        gp.np5_rederive(seed_1, [BROAD], depths, settings()), expected
    )
    # Rows of another recipe are left out.
    clean = cells.assign(recipe=res.CLEAN_RECIPE, bp=0.0)
    pd.testing.assert_frame_equal(
        gp.np5_rederive(
            pd.concat([seed_1, clean], ignore_index=True), [BROAD], depths, settings()
        ),
        expected,
    )
    other = _threshold_cells(seed=5)
    other["cell_id"] = "o" + other["cell_id"].astype(str)
    with pytest.raises(ValueError, match="one replicate"):
        gp.np5_rederive(
            pd.concat([cells, other.assign(seed=1)], ignore_index=True),
            [BROAD],
            depths,
            settings(),
        )
    with pytest.raises(ValueError, match="no rows"):
        gp.np5_rederive(cells, [BROAD], depths, settings(), recipe="other")


def test_np5_rederive_ensemble_and_members_at_any_mapping_seed() -> None:
    n = 400
    frames = [
        member_rows("m0", np.full(n, 0.95), pattern(n, 0.99)),
        member_rows("m1", np.full(n, 0.95), pattern(n, 0.96)),
    ]
    cells = pd.concat(frames, ignore_index=True)
    ensemble = res.EnsembleSettings()
    expected = res.ensemble_decide(
        cells,
        [BROAD],
        [100],
        settings(),
        ensemble,
        members=["m0", "m1"],
        neuronal={"X": True},
    )
    seed_1 = cells.assign(seed=1)
    pd.testing.assert_frame_equal(
        gp.np5_rederive_ensemble(
            seed_1,
            [BROAD],
            [100],
            settings(),
            ensemble,
            members=["m0", "m1"],
            neuronal={"X": True},
        ),
        expected.decisions,
    )
    # One member at a time: its own decide with the saturated-bp rule.
    pd.testing.assert_frame_equal(
        gp.np5_rederive(
            seed_1,
            [BROAD],
            [100],
            settings(),
            recipe=None,
            member="m1",
            saturated_bp_share=ensemble.saturated_bp_share,
        ),
        res.decide(frames[1], [BROAD], [100], settings(), saturated_bp_share=0.9),
    )
    with pytest.raises(ValueError, match="at least one emission member"):
        gp.np5_rederive_ensemble(
            seed_1, [BROAD], [100], settings(), ensemble, members=[]
        )
    with pytest.raises(ValueError, match="no rows"):
        gp.np5_rederive_ensemble(
            seed_1, [BROAD], [100], settings(), ensemble, members=["m0", "m9"]
        )
    with pytest.raises(ValueError, match="one replicate"):
        gp.np5_rederive_ensemble(
            pd.concat([frames[0], frames[1].assign(seed=1)], ignore_index=True),
            [BROAD],
            [100],
            settings(),
            ensemble,
            members=["m0", "m1"],
        )


def test_np5_settings_follow_the_config() -> None:
    np5 = _np5()
    assert np5.min_test_cells == 50
    assert np5.max_tstar_spread == pytest.approx(0.05)
    assert np5.max_extrapolated_share == pytest.approx(0.5)
    assert (
        gp.Np5Settings.from_config(
            AnnotationResolvabilityConfig(min_cells_per_bin=80)
        ).min_test_cells
        == 80
    )
    for bad in (
        {"min_test_cells": 0},
        {"max_tstar_spread": -0.01},
        {"max_extrapolated_share": 1.5},
    ):
        fields: dict[str, Any] = {"min_test_cells": 50, **bad}
        with pytest.raises(ValueError, match="Np5Settings"):
            gp.Np5Settings(**fields)


# --------------------------------------------------------------------------
# NP7: error structure (§14 NP7)

SINK = "SINK"
SPLAT = "SPLAT"  # a sink that is also region-implausible (as WHB Splatter)
HIPPO = "HIPPO"  # region-implausible in frontal cortex, not a sink


def _np7_vocab(
    plausible: Sequence[str] = ("nX", "nY", "nE"),
    *,
    sinks: Sequence[str] = (SINK, SPLAT),
    implausible: Sequence[str] = (HIPPO, SPLAT),
    level: str = res.WHB_SUPC,
) -> pd.DataFrame:
    """A vocab snapshot as MAP reads it (strings ``True`` / ``False``)."""
    nodes = list(dict.fromkeys([*plausible, *sinks, *implausible]))
    return pd.DataFrame(
        {
            "level": level,
            "node": nodes,
            "node_name": [f"name of {node}" for node in nodes],
            "sink": ["True" if node in sinks else "False" for node in nodes],
            "region_plausible_frontal_cortex": [
                "False" if node in implausible else "True" for node in nodes
            ],
        }
    )


def _np7_decisions(
    emitted: Mapping[tuple[str, str, int], float],
    withheld: Mapping[tuple[str, str, int], float] | None = None,
) -> pd.DataFrame:
    """Frozen provisional decisions: emitted and withheld bins with thresholds."""
    records = [
        {
            "regime": "provisional",
            "level": level,
            "class": cls,
            "depth": depth,
            "status": status,
            "threshold": threshold,
            "extrapolated": False,
        }
        for status, entries in (
            (res.STATUS_EMITTED, emitted),
            (res.STATUS_NOT_RESOLVABLE, withheld or {}),
        )
        for (level, cls, depth), threshold in entries.items()
    ]
    return pd.DataFrame.from_records(records)


def _np7_rows(
    groups: Sequence[tuple[int, str, str | None, str | None, float]],
    *,
    level: str = "supercluster",
    depth: int = 10,
    prefix: str = "c",
    truth: str | None = None,
) -> pd.DataFrame:
    """Rows of one bin: per group ``(n, truth class, call, parent, bp)``.

    A test cell of truth class ``T`` has the truth node ``n<T>`` (or
    ``truth`` for every group, as a coarse level's group name); a call is
    correct when it names its truth. Cell ids are ``<prefix><index>``,
    numbered over the groups, so tables built with the same groups and
    prefix at other levels or depths hold the same cells.
    """
    frames = []
    start = 0
    for n, truth_parent, call, parent, bp in groups:
        ids = [f"{prefix}{start + index}" for index in range(n)]
        start += n
        frame = bin_cells(
            np.full(n, bp, dtype=np.float64),
            np.zeros(n, dtype=bool),
            level=level,
            depth=depth,
        )
        truth_value = truth if truth is not None else f"n{truth_parent}"
        frame["cell_id"] = ids
        frame["sim_id"] = [f"{cell}|D{depth}" for cell in ids]
        frame["parent"] = pd.Series([parent] * n, dtype=object)
        frame["call"] = pd.Series([call] * n, dtype=object)
        frame["truth"] = truth_value
        frame["truth_parent"] = truth_parent
        frame["correct"] = np.full(n, call is not None and call == truth_value)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _np7_tables(
    cells: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    vocab: pd.DataFrame | None = None,
    settings_np7: gp.Np7Settings | None = None,
    **options: Any,
) -> tuple[dict[tuple[str, str], list[res.GatePTestedSet] | None], gp.Np7Tables]:
    """The pooled seed-0 tested sets and NP7's tables of one human replicate."""
    tested = res.gate_p_tested_sets(cells, decisions, regime="provisional")
    tables = gp.np7_error_structure(
        {("D1", 0): cells},
        decisions,
        tested,
        default_group=None,
        species="human",
        settings=settings_np7 or gp.Np7Settings(),
        vocab=_np7_vocab() if vocab is None else vocab,
        **options,
    )
    return tested, tables


def _by_class(tables: gp.Np7Tables) -> pd.DataFrame:
    frame = tables.wrong_node
    assert frame["class"].is_unique
    return frame.set_index("class")


def _level_row(tables: gp.Np7Tables, level: str = "supercluster") -> pd.Series:
    assert tables.excluded is not None
    rows = tables.excluded[tables.excluded["level"] == level]
    assert len(rows) == 1
    return rows.iloc[0]


XY_AT_10 = {("supercluster", "X", 10): 0.70, ("supercluster", "Y", 10): 0.70}

# Supercluster rows shaped as a frontal WHB bundle's vocab snapshot
# (``whb_frontal_supc_clus``): both sinks, Miscellaneous and Splatter, are
# Mixed/Unknown at broad and Neurons at lineage; Splatter and Amygdala
# excitatory are not plausible in frontal cortex.
_WHB_SHAPE: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    # node, key_name, broad_class, nt, lineage, sink, region plausible
    ("nE", "Upper-layer IT", "Neurons", "Excitatory", "Neurons", "False", "True"),
    ("nI", "MGE interneuron", "Neurons", "Inhibitory", "Neurons", "False", "True"),
    ("MISC", "Miscellaneous", "Mixed/Unknown", "Excitatory", "Neurons", "True", "True"),
    ("SPLAT", "Splatter", "Mixed/Unknown", "Other", "Neurons", "True", "False"),
    (
        "AMY",
        "Amygdala excitatory",
        "Neurons",
        "Excitatory",
        "Neurons",
        "False",
        "False",
    ),
)
_WHB_LEVELS: tuple[str, ...] = ("lineage", "broad", "nt", "supercluster")


def _whb_shape_vocab() -> pd.DataFrame:
    """A vocab snapshot with the columns and flags of the frontal WHB bundles."""
    return pd.DataFrame.from_records(
        [
            {
                "level": res.WHB_SUPC,
                "node": node,
                "node_name": name,
                "key_label": node,
                "key_name": name,
                "broad_class": broad,
                "nt": nt,
                "lineage": lineage,
                "sink": sink,
                "region_plausible_frontal_cortex": plausible,
                "never_drop": "False",
            }
            for node, name, broad, nt, lineage, sink, plausible in _WHB_SHAPE
        ]
    )


def _whb_shape_cells(
    groups: Sequence[tuple[int, str, str]], *, depth: int = 10, bp: float = 0.95
) -> pd.DataFrame:
    """A cells table read by the self-map's WHB level specs.

    Per group ``(n, truth node, assigned supercluster)``; cell ids are
    ``w<index>`` over the groups. The tidy table holds the supercluster
    assignment at ``bp`` without runner-ups, and ``level_cells`` reads it
    with ``whb_level_specs`` (lineage, broad, NT, supercluster) as the
    self-map does.
    """
    truth = [node for n, node, _ in groups for _ in range(n)]
    assigned = [node for n, _, node in groups for _ in range(n)]
    n_cells = len(truth)
    cell_ids = [f"w{index}" for index in range(n_cells)]
    sim_ids = [res.simulated_id(cell, depth) for cell in cell_ids]
    tidy = pd.DataFrame(
        {
            "cell_id": sim_ids,
            "level": res.WHB_SUPC,
            "level_name": "supercluster",
            "assignment": assigned,
            "bp": bp,
            "avg_correlation": 0.5,
        }
    )
    for rank in range(1, mmc.N_RUNNERS_UP + 1):
        tidy[mmc.runner_up_column(rank, "assignment")] = None
        tidy[mmc.runner_up_column(rank, "probability")] = math.nan
    empty = sparse.csr_matrix((n_cells, 1), dtype=np.float64)
    test = res.HeldOutCells(
        counts=empty,
        genes=["G"],
        obs=pd.DataFrame(
            {
                f"{res.TRUTH_PREFIX}{res.WHB_SUPC}": truth,
                res.TRUTH_LEAF_COLUMN: truth,
                res.SPILL_GROUP_COLUMN: "Neurons",
            },
            index=cell_ids,
        ),
    )
    query = res.SimulatedQuery(
        recipe=res.simulation_recipes(AnnotationResolvabilityConfig())[0],
        counts=empty,
        genes=["G"],
        obs=pd.DataFrame(
            {
                "cell_id": cell_ids,
                "depth": depth,
                "partner_id": "",
                "host_counts": float(depth),
                "spill_counts": 0.0,
                "total_counts": float(depth),
            },
            index=sim_ids,
        ),
        n_by_depth={depth: n_cells},
    )
    specs = res.whb_level_specs(
        _whb_shape_vocab(), AnnotationThresholds(), include_fine=False
    )
    return res.level_cells(tidy, query, test, specs)


def _whb_shape_decisions(classes: Sequence[str]) -> pd.DataFrame:
    """Frozen decisions emitting every WHB level of ``classes`` at 10 (0.70)."""
    return _np7_decisions(
        {(level, cls, 10): 0.70 for level in _WHB_LEVELS for cls in classes}
    )


def test_np7_a_wrong_node_above_5pct_of_the_truth_class_fails() -> None:
    """§14 NP7 / D12: the share of truth class c's confident calls on one node.

    At 5% exactly X passes, above it X fails. Y's tested set holds the wrong
    calls (the called-class view, 6%+), which is reported only: Y's own
    cells all land on Y, so Y passes (one failing class leaves the others).
    """
    decisions = _np7_decisions(XY_AT_10)
    for n_wrong, passed in ((20, True), (21, False)):
        cells = _np7_rows(
            [
                (380, "X", "nX", "X", 0.95),
                (n_wrong, "X", "nY", "Y", 0.95),
                (300, "Y", "nY", "Y", 0.95),
            ]
        )
        tested, tables = _np7_tables(cells, decisions)
        wrong = _by_class(tables)
        x = wrong.loc["X"]
        assert x["set"] == "10" and not x["pooled"]
        assert x["n_truth_confident"] == 380 + n_wrong
        assert x["n_truth_wrong"] == n_wrong and x["n_truth_excluded"] == 0
        assert x["wrong_node"] == "nY" and x["n_wrong_node"] == n_wrong
        assert x["wrong_node_share"] == pytest.approx(n_wrong / (380 + n_wrong))
        assert x["max_share"] == pytest.approx(0.05)
        assert bool(x["passed"]) is passed
        # X's own tested set is all right: its called-class view is clean.
        assert x["n_called_confident"] == 380 and x["n_called_wrong_node"] == 0
        assert x["called_wrong_node"] is None
        y = wrong.loc["Y"]
        assert y["n_called_confident"] == 300 + n_wrong
        assert y["called_wrong_node"] == "nY"
        assert y["called_wrong_node_share"] == pytest.approx(n_wrong / (300 + n_wrong))
        assert y["n_truth_confident"] == 300 and y["n_wrong_node"] == 0
        assert y["wrong_node"] is None and y["wrong_node_share"] == 0.0
        assert bool(y["passed"])
        assert gp.np7_class_verdicts(tables.excluded, tables.wrong_node, tested) == {
            ("supercluster", "X"): passed,
            ("supercluster", "Y"): True,
        }


def test_np7_a_planted_sink_absorbing_more_than_5pct_of_a_class_fails() -> None:
    """§12 M13: a planted sink absorbing > 5% of a class fails NP7.

    The sink's calls have a null parent, so the frozen mask leaves them out
    of every tested set; NP7 still counts them as confident calls of their
    truth class on one wrong node (the sink). Here the level's share stays
    below 1%, so only the class fails.
    """
    decisions = _np7_decisions(XY_AT_10)
    cells = _np7_rows(
        [
            (376, "X", "nX", "X", 0.95),
            (24, "X", SINK, None, 0.95),
            (3000, "Y", "nY", "Y", 0.95),
        ]
    )
    tested, tables = _np7_tables(cells, decisions)
    (item,) = tested[("supercluster", "X")] or []
    assert item.n_confident == 376
    level = _level_row(tables)
    assert level["n_confident"] == 3376
    assert level["n_excluded_calls"] == 24
    assert level["n_excluded_confident"] == 24 and level["n_sink"] == 24
    assert level["excluded_share"] == pytest.approx(24 / 3376)
    assert level["nodes"] == "SINK:24"
    assert bool(level["passed"])
    x = _by_class(tables).loc["X"]
    assert x["n_truth_confident"] == 400 and x["n_truth_excluded"] == 24
    assert x["wrong_node"] == SINK
    assert x["wrong_node_share"] == pytest.approx(0.06)
    assert not bool(x["passed"])
    assert gp.np7_class_verdicts(tables.excluded, tables.wrong_node, tested) == {
        ("supercluster", "X"): False,
        ("supercluster", "Y"): True,
    }


def test_np7_sink_rows_with_a_null_parent_count_against_1pct_of_the_level() -> None:
    """§14 NP7 (human): confident sink or region-implausible calls <= 1%.

    The denominator is the level's confident calls at the frozen thresholds
    over all classes (K9.1); the sink rows (null parent) are counted beside
    it, never in it. A node that is a sink and region-implausible counts
    once, as a sink. Above 1% every class of the level fails.
    """
    decisions = _np7_decisions(XY_AT_10)
    lookup = res.emission_lookup(decisions, "provisional")
    for n_extra, passed in ((0, True), (1, False)):
        cells = _np7_rows(
            [
                (500, "X", "nX", "X", 0.95),
                (500, "Y", "nY", "Y", 0.95),
                (3, "Y", SINK, None, 0.95),
                (3, "Y", SPLAT, None, 0.95),
                (4 + n_extra, "X", HIPPO, None, 0.95),
            ]
        )
        assert not res.frozen_confident_mask(cells, lookup)[1000:].any()
        tested, tables = _np7_tables(cells, decisions)
        level = _level_row(tables)
        assert level["n_confident"] == 1000
        assert level["n_excluded_confident"] == 10 + n_extra
        assert level["n_sink"] == 6
        assert level["n_region_implausible"] == 4 + n_extra
        assert level["n_not_in_vocab"] == 0
        assert level["excluded_share"] == pytest.approx((10 + n_extra) / 1000)
        assert level["max_share"] == pytest.approx(0.01)
        assert level["nodes"] == f"HIPPO:{4 + n_extra};SINK:3;SPLAT:3"
        assert bool(level["passed"]) is passed
        assert gp.np7_class_verdicts(tables.excluded, tables.wrong_node, tested) == {
            ("supercluster", "X"): passed,
            ("supercluster", "Y"): passed,
        }


def test_np7_region_column_is_parameterised() -> None:
    """CHECK K4: gate P reads ``region_plausible_frontal_cortex`` by default.

    The column follows ``Np7Settings.region``; a vocab without it raises
    rather than reading every node as plausible.
    """
    assert gp.NP7_REGION == "frontal_cortex"
    assert gp.Np7Settings().region == "frontal_cortex"
    vocab = _np7_vocab(("nX", "nY", "nH"))
    vocab["region_plausible_hippocampus"] = [
        "False" if node == "nH" else "True" for node in vocab["node"]
    ]
    decisions = _np7_decisions(XY_AT_10)
    cells = _np7_rows(
        [
            (500, "X", "nX", "X", 0.95),
            (500, "Y", "nY", "Y", 0.95),
            (30, "X", "nH", None, 0.95),
            (20, "X", HIPPO, None, 0.95),
        ]
    )
    assert gp.np7_excluded_nodes(vocab) == {
        "nX": None,
        "nY": None,
        "nH": None,
        SINK: "sink",
        SPLAT: "sink",
        HIPPO: "region_implausible",
    }
    assert gp.np7_excluded_nodes(vocab, region="hippocampus")["nH"] == (
        "region_implausible"
    )
    _, frontal = _np7_tables(cells, decisions, vocab=vocab)
    assert _level_row(frontal)["nodes"] == "HIPPO:20"
    _, hippocampus = _np7_tables(
        cells,
        decisions,
        vocab=vocab,
        settings_np7=gp.Np7Settings(region="hippocampus"),
    )
    level = _level_row(hippocampus)
    assert level["nodes"] == "nH:30"
    assert level["n_region_implausible"] == 30
    with pytest.raises(ValueError, match="region_plausible_cerebellum"):
        _np7_tables(
            cells,
            decisions,
            vocab=vocab,
            settings_np7=gp.Np7Settings(region="cerebellum"),
        )


def test_np7_coarse_levels_take_the_node_of_the_supercluster_row() -> None:
    """A lineage, broad or NT call names a group, not a node (``group_level_calls``).

    Its node is the supercluster call of the same simulated cell (one WHB
    assignment), and each level's share uses that level's denominator. An
    excluded call's bp is the larger of its own and its supercluster row's:
    a sink names no group at broad, so its broad call has no bp of its own,
    and the supercluster bp decides. At broad, a call to a
    region-implausible node is "correct" in the cells table (both are
    Neurons), but it is a call to an excluded node, so NP7 counts it as a
    wrong call on that node.
    """
    groups_supc: list[tuple[int, str, str | None, str | None, float]] = [
        (300, "Exc", "nE", "Exc", 0.95),
        (5, "Exc", HIPPO, None, 0.90),
        (3, "Exc", SINK, None, 0.90),
        (2, "Exc", SINK, None, 0.60),
        (4, "Exc", SINK, None, 0.50),
    ]
    groups_broad: list[tuple[int, str, str | None, str | None, float]] = [
        (300, "Exc", "Neurons", "Exc", 0.95),
        (5, "Exc", "Neurons", None, 0.95),
        (3, "Exc", None, None, math.nan),
        (2, "Exc", None, None, math.nan),
        (4, "Exc", None, None, math.nan),
    ]
    groups_lineage: list[tuple[int, str, str | None, str | None, float]] = [
        (300, "Exc", "Neurons", "Exc", 0.95),
        (5, "Exc", "Neurons", None, 0.95),
        (3, "Exc", "Neurons", None, 0.95),
        # Its own (group) bp reaches the threshold, its supercluster's not.
        (2, "Exc", "Neurons", None, 0.80),
        # Neither does.
        (4, "Exc", "Neurons", None, 0.65),
    ]
    cells = pd.concat(
        [
            _np7_rows(groups_supc),
            _np7_rows(groups_broad, level="broad", truth="Neurons"),
            _np7_rows(groups_lineage, level="lineage", truth="Neurons"),
        ],
        ignore_index=True,
    )
    decisions = _np7_decisions(
        {(level, "Exc", 10): 0.70 for level in ("supercluster", "broad", "lineage")}
    )
    tested, tables = _np7_tables(cells, decisions)
    assert tables.excluded is not None
    assert tables.excluded["level"].tolist() == ["broad", "lineage", "supercluster"]
    for level, n_confident_excluded, nodes in (
        ("supercluster", 8, "HIPPO:5;SINK:3"),
        # The sink's broad call has no group, so no bp: its supercluster's
        # bp (0.90) decides.
        ("broad", 8, "HIPPO:5;SINK:3"),
        ("lineage", 10, "HIPPO:5;SINK:5"),
    ):
        row = _level_row(tables, level)
        assert row["n_confident"] == 300
        assert row["n_excluded_calls"] == 14
        assert row["n_excluded_confident"] == n_confident_excluded
        assert row["nodes"] == nodes
    broad = tables.wrong_node[tables.wrong_node["level"] == "broad"].iloc[0]
    assert broad["n_truth_confident"] == 308 and broad["n_truth_excluded"] == 8
    assert broad["wrong_node"] == HIPPO and broad["n_wrong_node"] == 5
    assert bool(broad["passed"])
    assert set(gp.np7_class_verdicts(tables.excluded, tables.wrong_node, tested)) == {
        ("broad", "Exc"),
        ("lineage", "Exc"),
        ("supercluster", "Exc"),
    }
    # Every row needs the supercluster row of its simulated cell.
    lone = cells.drop(index=cells.index[(cells["level"] == "supercluster")][:1])
    with pytest.raises(ValueError, match="no 'supercluster' row"):
        _np7_tables(lone.reset_index(drop=True), decisions)
    with pytest.raises(ValueError, match="no 'supercluster' level"):
        _np7_tables(
            cells[cells["level"] != "supercluster"].reset_index(drop=True), decisions
        )


def test_np7_whb_sinks_count_at_broad_and_nt_through_their_supercluster_bp() -> None:
    """§12 M13's planted sink at every WHB level, on the real snapshot's shape.

    The cells come from the self-map's level specs. Both WHB sinks are
    Mixed/Unknown at broad, so at broad and NT their calls name no group
    and have no bp (the review's scenario: broad X used to pass with a sink
    share of 0 while supercluster X failed). NP7 judges them on their
    supercluster bp, so 24 X cells on Splatter (6% of X) fail X at every
    level, broad (the promotion level) and NT included; the level shares
    stay below 1%.
    """
    cells = _whb_shape_cells(
        [
            (376, "nE", "nE"),
            (24, "nE", "SPLAT"),
            (3, "nE", "AMY"),
            (2995, "nI", "nI"),
            (5, "nI", "MISC"),
        ]
    )
    sinks = cells["cell_id"].isin([f"w{index}" for index in range(376, 400)])
    for level in ("broad", "nt"):
        at = (cells["level"] == level) & sinks
        assert cells.loc[at, "call"].isna().all()
        assert cells.loc[at, "bp"].isna().all()
        assert cells.loc[at, "parent"].isna().all()
    lineage = (cells["level"] == "lineage") & sinks
    assert (cells.loc[lineage, "call"] == "Neurons").all()
    assert cells.loc[lineage, "correct"].all()
    tested, tables = _np7_tables(
        cells, _whb_shape_decisions(["Exc", "Inh"]), vocab=_whb_shape_vocab()
    )
    for level in _WHB_LEVELS:
        row = _level_row(tables, level)
        assert row["n_confident"] == 376 + 2995
        assert row["n_excluded_calls"] == 32
        assert row["n_excluded_confident"] == 32
        assert row["n_sink"] == 29 and row["n_region_implausible"] == 3
        assert row["nodes"] == "SPLAT:24;MISC:5;AMY:3"
        assert bool(row["passed"])
    wrong = tables.wrong_node.set_index(["level", "class"])
    for level in _WHB_LEVELS:
        exc = wrong.loc[(level, "Exc")]
        assert exc["n_truth_confident"] == 403 and exc["n_truth_excluded"] == 27
        assert exc["wrong_node"] == "SPLAT" and exc["n_wrong_node"] == 24
        assert exc["wrong_node_share"] == pytest.approx(24 / 403)
        assert not bool(exc["passed"])
        assert wrong.loc[(level, "Inh"), "n_truth_excluded"] == 5
    assert gp.np7_class_verdicts(tables.excluded, tables.wrong_node, tested) == {
        **{(level, "Exc"): False for level in _WHB_LEVELS},
        **{(level, "Inh"): True for level in _WHB_LEVELS},
    }


def test_np7_reads_the_assigned_node_at_the_rows_own_depth() -> None:
    """A cell's assigned node is its supercluster call at the same depth.

    Under thinning one test cell can land on a sink at 10 counts and on its
    own node at 30. Only its 10-count rows are calls to the sink, at every
    level: the bin at 10 fails, the bin at 30 is clean.
    """

    def both(
        level: str, call_10: str, call_30: str, truth: str | None
    ) -> list[pd.DataFrame]:
        return [
            _np7_rows(
                [(300, "X", call_10, "X", 0.95), (20, "X", call_30, None, 0.95)],
                level=level,
                truth=truth,
            ),
            _np7_rows(
                [(320, "X", call_10, "X", 0.95)],
                level=level,
                depth=30,
                truth=truth,
            ),
        ]

    cells = pd.concat(
        [
            *both("supercluster", "nX", SINK, None),
            *both("lineage", "Neurons", "Neurons", "Neurons"),
        ],
        ignore_index=True,
    )
    decisions = _np7_decisions(
        {
            (level, "X", depth): 0.70
            for level in ("supercluster", "lineage")
            for depth in (10, 30)
        }
    )
    tested, tables = _np7_tables(cells, decisions)
    for level in ("lineage", "supercluster"):
        row = _level_row(tables, level)
        assert row["n_confident"] == 300 + 320
        assert row["n_excluded_calls"] == 20 and row["n_excluded_confident"] == 20
        assert row["nodes"] == "SINK:20"
        sets = tables.wrong_node[tables.wrong_node["level"] == level].set_index("set")
        assert sets.loc["10", "n_wrong_node"] == 20
        assert sets.loc["10", "wrong_node"] == SINK
        assert not bool(sets.loc["10", "passed"])
        assert sets.loc["30", "n_truth_confident"] == 320
        assert sets.loc["30", "n_wrong_node"] == 0 and bool(sets.loc["30", "passed"])
    assert gp.np7_class_verdicts(tables.excluded, tables.wrong_node, tested) == {
        ("lineage", "X"): False,
        ("supercluster", "X"): False,
    }


def test_np7_a_tested_set_without_confident_calls_of_its_truth_class_fails() -> None:
    """An empty truth view fails (a ``nan`` share never passes).

    Every call of class X is a Z cell: X's tested set exists (300 confident
    calls of the called class), but no X cell is in its scope.
    """
    decisions = _np7_decisions(XY_AT_10)
    cells = _np7_rows([(300, "Z", "nX", "X", 0.95), (300, "Y", "nY", "Y", 0.95)])
    tested, tables = _np7_tables(cells, decisions)
    x = _by_class(tables).loc["X"]
    assert x["n_truth_confident"] == 0 and x["n_wrong_node"] == 0
    assert math.isnan(x["wrong_node_share"])
    assert x["n_called_confident"] == 300 and x["called_wrong_node"] == "nX"
    assert not bool(x["passed"])
    assert gp.np7_class_verdicts(tables.excluded, tables.wrong_node, tested) == {
        ("supercluster", "X"): False,
        ("supercluster", "Y"): True,
    }


def test_np7_an_excluded_call_is_confident_at_the_lowest_threshold_of_its_bin() -> None:
    """An excluded call has no class, so no frozen threshold of its own.

    It counts as confident when its bp reaches the lowest frozen threshold
    emitted at its level and depth (K9.1's "emitted bins"): a withheld bin's
    threshold, a depth without an emitted bin and a NaN bp never count; the
    calls whatever their bp are reported (``n_excluded_calls``).
    """
    decisions = _np7_decisions(
        {
            ("supercluster", "X", 10): 0.80,
            ("supercluster", "Y", 10): 0.60,
            ("supercluster", "X", 60): 0.70,
        },
        withheld={("supercluster", "Z", 10): 0.50, ("supercluster", "X", 30): 0.10},
    )
    sink_bp_10 = [0.55, 0.59, 0.60 - 5e-10, 0.75, math.nan]
    cells = pd.concat(
        [
            _np7_rows([(400, "X", "nX", "X", 0.95)], prefix="x"),
            *(
                _np7_rows([(1, "X", SINK, None, bp)], prefix=f"s10_{index}_")
                for index, bp in enumerate(sink_bp_10)
            ),
            _np7_rows([(1, "X", SINK, None, 0.99)], depth=30, prefix="s30_"),
            *(
                _np7_rows([(1, "X", SINK, None, bp)], depth=60, prefix=f"s60_{index}_")
                for index, bp in enumerate((0.69, 0.70))
            ),
        ],
        ignore_index=True,
    )
    _, tables = _np7_tables(cells, decisions)
    level = _level_row(tables)
    assert level["n_confident"] == 400
    assert level["n_excluded_calls"] == 8
    assert level["n_excluded_confident"] == 3
    assert level["nodes"] == "SINK:3"


def test_np7_a_node_outside_the_vocab_counts_as_implausible() -> None:
    """Production reads a WHB node outside the vocab as implausible (RESOLVE).

    The self-map's level specs give such a call no class at every level
    (and no group, so no bp, at the coarse levels), so NP7 counts it at
    every level through its supercluster bp.
    """
    decisions = _np7_decisions(XY_AT_10)
    cells = _np7_rows(
        [
            (500, "X", "nX", "X", 0.95),
            (500, "Y", "nY", "Y", 0.95),
            (2, "X", "nQ", None, 0.95),
        ]
    )
    _, tables = _np7_tables(cells, decisions)
    level = _level_row(tables)
    assert level["n_not_in_vocab"] == 2
    assert level["nodes"] == "nQ:2"
    whb = _whb_shape_cells([(400, "nE", "nE"), (2, "nE", "nQ")])
    outside = whb["cell_id"].isin(["w400", "w401"]).to_numpy()
    assert whb.loc[outside, "parent"].isna().all()
    assert whb.loc[outside & (whb["level"] == "broad").to_numpy(), "bp"].isna().all()
    _, whb_tables = _np7_tables(
        whb, _whb_shape_decisions(["Exc"]), vocab=_whb_shape_vocab()
    )
    assert whb_tables.excluded is not None
    assert whb_tables.excluded["n_not_in_vocab"].tolist() == [2, 2, 2, 2]


def test_np7_refuses_a_blank_vocab_flag() -> None:
    """A blank sink or region flag is refused, not read as outside the vocab.

    The self-map's level specs read a blank sink as false and a blank
    region flag as true (``whb_level_specs``), so such a node's calls keep a
    class there; NP7 cannot read the flag another way.
    """
    for column in ("sink", "region_plausible_frontal_cortex"):
        for blank in ("", " ", math.nan, None):
            vocab = _np7_vocab()
            vocab[column] = vocab[column].astype(object)
            vocab.loc[1, column] = blank
            with pytest.raises(ValueError, match=f"{column!r} of node 'nY' is blank"):
                gp.np7_excluded_nodes(vocab)


def test_np7_pooled_set_reads_each_truth_cell_once_at_its_deepest_row() -> None:
    """The truth view of a ">= D_P" set takes each test cell's deepest row.

    150 X cells have rows at 10 and 30 counts, 15 of them called Y at 30;
    120 more have a row at 10 only. The pooled set holds 270 cells (not 420
    rows) with 15 on Y (5.6%): it fails, while the bin tested on its own at
    10 counts is clean.
    """
    decisions = _np7_decisions(
        {
            ("supercluster", "X", 10): 0.70,
            ("supercluster", "X", 30): 0.70,
            ("supercluster", "Y", 30): 0.70,
        }
    )
    cells = pd.concat(
        [
            _np7_rows([(150, "X", "nX", "X", 0.95)], prefix="t"),
            _np7_rows([(120, "X", "nX", "X", 0.95)], prefix="u"),
            _np7_rows(
                [(135, "X", "nX", "X", 0.95), (15, "X", "nY", "Y", 0.95)],
                depth=30,
                prefix="t",
            ),
        ],
        ignore_index=True,
    )
    tested, tables = _np7_tables(cells, decisions)
    assert [
        gp.tested_set_label(item) for item in tested[("supercluster", "X")] or []
    ] == [
        ">=10",
        "10",
    ]
    assert tested[("supercluster", "Y")] is None
    wrong = tables.wrong_node.set_index("set")
    assert wrong.loc[">=10", "n_truth_confident"] == 270
    assert wrong.loc[">=10", "n_wrong_node"] == 15
    assert wrong.loc[">=10", "wrong_node_share"] == pytest.approx(15 / 270)
    assert not bool(wrong.loc[">=10", "passed"])
    assert wrong.loc["10", "n_truth_confident"] == 270
    assert wrong.loc["10", "n_wrong_node"] == 0 and bool(wrong.loc["10", "passed"])
    assert gp.np7_class_verdicts(tables.excluded, tables.wrong_node, tested) == {
        ("supercluster", "X"): False,
        ("supercluster", "Y"): None,
    }


def test_np7_scores_the_default_group_on_its_check_half_only() -> None:
    """NP7 is scored on the pooled held-out calls (§14; ``pooled_held_out_cells``)."""
    decisions = _np7_decisions({("supercluster", "X", 10): 0.70})
    default = _np7_rows(
        [(400, "X", "nX", "X", 0.95), (10, "X", SINK, None, 0.95)], prefix="a"
    )
    default["half"] = np.arange(len(default)) % 2
    other = _np7_rows([(400, "X", "nX", "X", 0.95)], prefix="b")
    replicates = {("D1", 0): default, ("D2", 0): other}
    pooled = gp.pooled_held_out_cells(replicates, default_group="D1")
    tested = res.gate_p_tested_sets(pooled, decisions, regime="provisional")
    tables = gp.np7_error_structure(
        replicates,
        decisions,
        tested,
        default_group="D1",
        species="human",
        settings=gp.Np7Settings(),
        vocab=_np7_vocab(),
    )
    level = _level_row(tables)
    assert level["n_confident"] == 200 + 400
    assert level["n_excluded_confident"] == 5


def test_np7_tables_per_member_feed_every_member_verdict() -> None:
    """Version 7: NP7 per emission member (K9.1), combined over the members."""
    decisions = _np7_decisions(XY_AT_10)
    frames = []
    for member, n_sink in (("m0", 3), ("m1", 15)):
        rows = _np7_rows(
            [
                (400, "X", "nX", "X", 0.95),
                (600, "Y", "nY", "Y", 0.95),
                (n_sink, "Y", SINK, None, 0.95),
            ]
        )
        rows[res.MEMBER_COLUMN] = member
        frames.append(rows)
    cells = pd.concat(frames, ignore_index=True)
    member_sets = res.gate_p_member_sets(
        cells, decisions, members=["m0", "m1"], regime="provisional"
    )
    verdicts = {}
    for member, tested in member_sets.items():
        tables = gp.np7_error_structure(
            {("D1", 0): cells},
            decisions,
            tested,
            default_group=None,
            species="human",
            settings=gp.Np7Settings(),
            vocab=_np7_vocab(),
            recipe=None,
            member=member,
        )
        verdicts[member] = gp.np7_class_verdicts(
            tables.excluded, tables.wrong_node, tested
        )
    assert verdicts["m0"] == {("supercluster", "X"): True, ("supercluster", "Y"): True}
    assert verdicts["m1"] == {
        ("supercluster", "X"): False,
        ("supercluster", "Y"): False,
    }
    combined = res.every_member_verdict(verdicts)
    assert combined[("supercluster", "X")]["status"] == res.GATE_P_FAILED
    assert combined[("supercluster", "X")]["failed_members"] == ["m1"]


def test_np7_mouse_scores_the_wrong_node_part_only() -> None:
    """§14 NP7's 1% part is human only; the 5% part applies to both species."""
    decisions = _np7_decisions({("class", "X", 10): 0.70, ("class", "Y", 10): 0.70})
    cells = _np7_rows(
        [
            (380, "X", "nX", "X", 0.95),
            (30, "X", "nY", "Y", 0.95),
            (300, "Y", "nY", "Y", 0.95),
            (50, "Y", "nZ", None, 0.95),
        ],
        level="class",
    )
    tested = res.gate_p_tested_sets(cells, decisions, regime="provisional")
    tables = gp.np7_error_structure(
        {("D1", 0): cells},
        decisions,
        tested,
        default_group=None,
        species="mouse",
        settings=gp.Np7Settings(),
    )
    assert tables.excluded is None
    wrong = _by_class(tables)
    assert wrong.loc["X", "wrong_node_share"] == pytest.approx(30 / 410)
    # Without sinks, a call without a class is not confident (frozen mask).
    assert wrong.loc["Y", "n_truth_confident"] == 300
    assert gp.np7_class_verdicts(None, tables.wrong_node, tested) == {
        ("class", "X"): False,
        ("class", "Y"): True,
    }
    with pytest.raises(ValueError, match="human only"):
        gp.np7_error_structure(
            {("D1", 0): cells},
            decisions,
            tested,
            default_group=None,
            species="mouse",
            settings=gp.Np7Settings(),
            vocab=_np7_vocab(),
        )


def test_np7_class_verdicts_need_every_row_and_a_passed_value() -> None:
    decisions = _np7_decisions(XY_AT_10)
    cells = _np7_rows([(400, "X", "nX", "X", 0.95), (300, "Y", "nY", "Y", 0.95)])
    tested, tables = _np7_tables(cells, decisions)
    assert tables.excluded is not None
    with_none = {**tested, ("supercluster", "Z"): None}
    assert gp.np7_class_verdicts(tables.excluded, tables.wrong_node, with_none) == {
        ("supercluster", "X"): True,
        ("supercluster", "Y"): True,
        ("supercluster", "Z"): None,
    }
    with pytest.raises(ValueError, match="no NP7 excluded-share row"):
        gp.np7_class_verdicts(tables.excluded.iloc[:0], tables.wrong_node, tested)
    with pytest.raises(ValueError, match="no NP7 wrong-node row"):
        gp.np7_class_verdicts(
            tables.excluded,
            tables.wrong_node[tables.wrong_node["class"] != "Y"],
            tested,
        )
    # A CSV round trip keeps a missing verdict missing; it never passes.
    excluded = pd.read_csv(io.StringIO(tables.excluded.to_csv(index=False)))
    excluded["passed"] = excluded["passed"].astype(object)
    excluded.loc[0, "passed"] = math.nan
    with pytest.raises(ValueError, match="NP7 excluded share row"):
        gp.np7_class_verdicts(excluded, tables.wrong_node, tested)
    with pytest.raises(ValueError, match="empty list"):
        gp.np7_class_verdicts(
            tables.excluded, tables.wrong_node, {("supercluster", "X"): []}
        )


def test_np7_inputs_that_mix_or_lack_rows_raise() -> None:
    decisions = _np7_decisions(XY_AT_10)
    cells = _np7_rows([(400, "X", "nX", "X", 0.95), (300, "Y", "nY", "Y", 0.95)])
    # A call to an excluded node that still has a class: the cells were
    # built with another region or vocab than NP7 reads.
    with_class = pd.concat(
        [cells, _np7_rows([(5, "X", HIPPO, "X", 0.95)], prefix="h")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="another region or vocab"):
        _np7_tables(with_class, decisions)
    with pytest.raises(ValueError, match="'sink'"):
        _np7_tables(cells, decisions, vocab=_np7_vocab().drop(columns=["sink"]))
    bad_flag = _np7_vocab()
    bad_flag.loc[0, "sink"] = "yes"
    with pytest.raises(ValueError, match="true or false"):
        _np7_tables(cells, decisions, vocab=bad_flag)
    twice = _np7_vocab()
    with pytest.raises(ValueError, match="more than once"):
        _np7_tables(
            cells,
            decisions,
            vocab=pd.concat([twice, twice.iloc[:1]], ignore_index=True),
        )
    tested = res.gate_p_tested_sets(cells, decisions, regime="provisional")
    with pytest.raises(ValueError, match="needs the vocab"):
        gp.np7_error_structure(
            {("D1", 0): cells},
            decisions,
            tested,
            default_group=None,
            species="human",
            settings=gp.Np7Settings(),
        )
    with pytest.raises(ValueError, match="no rows"):
        _np7_tables(cells, decisions, recipe="other")
    with pytest.raises(ValueError, match="empty list"):
        gp.np7_error_structure(
            {("D1", 0): cells},
            decisions,
            {("supercluster", "X"): []},
            default_group=None,
            species="human",
            settings=gp.Np7Settings(),
            vocab=_np7_vocab(),
        )


def test_np7_settings_hold_the_section_14_limits() -> None:
    np7 = gp.Np7Settings()
    assert np7.max_excluded_share == pytest.approx(0.01)
    assert np7.max_wrong_node_share == pytest.approx(0.05)
    assert np7.region_column == "region_plausible_frontal_cortex"
    for bad in (
        {"max_excluded_share": -0.01},
        {"max_wrong_node_share": 1.5},
        {"region": ""},
    ):
        fields: dict[str, Any] = dict(bad)
        with pytest.raises(ValueError, match="Np7Settings"):
            gp.Np7Settings(**fields)
