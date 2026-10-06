"""Tests for gate-P scoring (M13; plan §14 new panel family, NP4 first)."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import gate_p as gp
from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    AnnotationResolvabilityConfig,
    AnnotationThresholds,
)

from .test_resolvability import BROAD, bin_cells, settings, tracked_cells


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
    return verdicts.iloc[0], gp.np4_class_verdicts(verdicts, tested)


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
    assert gp.np4_class_verdicts(verdicts, tested) == {
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
        gp.np4_class_verdicts(_set_verdict_rows({"10": True}), empty)
    with pytest.raises(ValueError, match="belongs to"):
        gp.np4_class_verdicts(
            _set_verdict_rows({"10": True}), {("broad", "Y"): [SINGLE_10]}
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
        _set_verdict_rows({">=30": True, "10": True}), tested
    )
    failing = gp.np4_class_verdicts(
        _set_verdict_rows({">=30": True, "10": False}), tested
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
        gp.np4_class_verdicts(_set_verdict_rows({"10": True}, cls="Z"), tested)
    with pytest.raises(ValueError, match="no NP4 verdict"):
        gp.np4_class_verdicts(_set_verdict_rows({"10": True}), tested)


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
