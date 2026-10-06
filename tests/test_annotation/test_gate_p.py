"""Tests for gate-P scoring (M13; plan §14 new panel family, NP4 first)."""

from __future__ import annotations

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

from .test_resolvability import BROAD, bin_cells, settings


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
