"""Tests for the resolvability self-map (plan §8.3, §5.4, §13; E2 verdict 3)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    AnnotationResolvabilityConfig,
    AnnotationThresholds,
)
from merxen.annotation.mapmycells_engine import N_RUNNERS_UP, runner_up_column

# --------------------------------------------------------------------------
# Synthetic three-level reference and a small bootstrap mapper

LEVELS = ("L1", "L2", "L3")
CLASSES = ("A", "B")
SUBCLASSES = {"A": ("A1", "A2"), "B": ("B1", "B2")}
N_GENES = 60


def _markers() -> dict[str, list[int]]:
    """Ten marker genes per class and per subclass; clusters have none."""
    markers = {"A": list(range(0, 10)), "B": list(range(10, 20))}
    start = 20
    for cls in CLASSES:
        for subclass in SUBCLASSES[cls]:
            markers[subclass] = list(range(start, start + 10))
            start += 10
    return markers


MARKERS = _markers()


def cluster_labels() -> list[tuple[str, str, str]]:
    """(class, subclass, cluster): two clusters per subclass, identical profiles."""
    return [
        (cls, subclass, f"{subclass}{suffix}")
        for cls in CLASSES
        for subclass in SUBCLASSES[cls]
        for suffix in ("x", "y")
    ]


def profile(cls: str, subclass: str) -> np.ndarray:
    weights = np.ones(N_GENES)
    weights[MARKERS[cls]] += 6.0
    weights[MARKERS[subclass]] += 6.0
    return weights / weights.sum()


def make_test_cells(
    n_per_cluster: int = 120,
    *,
    seed: int = 0,
    shallow_class: str | None = None,
) -> res.HeldOutCells:
    """Synthetic held-out cells with native depths 150-400 (shallow class: 20-40)."""
    rng = np.random.default_rng(seed)
    rows = []
    obs = []
    for cls, subclass, cluster in cluster_labels():
        for index in range(n_per_cluster):
            depth = (
                int(rng.integers(20, 40))
                if cls == shallow_class
                else int(rng.integers(150, 400))
            )
            rows.append(rng.multinomial(depth, profile(cls, subclass)))
            obs.append(
                {
                    "cell_id": f"{cluster}_{index}",
                    f"{res.TRUTH_PREFIX}L1": cls,
                    f"{res.TRUTH_PREFIX}L2": subclass,
                    f"{res.TRUTH_PREFIX}L3": cluster,
                    res.TRUTH_LEAF_COLUMN: subclass,
                    res.SPILL_GROUP_COLUMN: cls,
                }
            )
    frame = pd.DataFrame(obs).set_index("cell_id")
    return res.HeldOutCells(
        counts=sparse.csr_matrix(np.asarray(rows, dtype=np.float64)),
        genes=[f"G{index:02d}" for index in range(N_GENES)],
        obs=frame,
    )


def _children(level: str, parent: str | None) -> list[str]:
    if level == "L1":
        return list(CLASSES)
    if level == "L2":
        return list(SUBCLASSES[str(parent)])
    return [f"{parent}x", f"{parent}y"]


def bootstrap_mapper(
    query: res.SimulatedQuery, tag: str, seed: int, *, iterations: int = 20
) -> pd.DataFrame:
    """Map hierarchically by marker counts with a gene bootstrap (bf 0.5).

    Vectorised: each iteration draws one 5-gene subset per child and scores
    every cell of the parent with a matrix sum (a per-cell loop made the
    module fixture take ~17 s).
    """
    rng = np.random.default_rng(seed + 11)
    counts = query.counts.toarray()
    n_cells = counts.shape[0]
    cell_ids = query.obs.index.astype(str).to_numpy()
    parents = np.full(n_cells, "", dtype=object)
    records: list[dict[str, object]] = []
    for level in LEVELS:
        assigned = np.empty(n_cells, dtype=object)
        for parent in sorted({str(value) for value in parents}):
            rows = np.flatnonzero(parents == parent)
            children = _children(level, parent or None)
            wins = np.zeros((len(rows), len(children)))
            for _ in range(iterations):
                if level == "L3":
                    # No cluster markers: the bootstrap is a coin flip.
                    scores = rng.random((len(rows), len(children)))
                else:
                    scores = np.column_stack(
                        [
                            counts[
                                np.ix_(
                                    rows, rng.choice(MARKERS[child], 5, replace=False)
                                )
                            ].sum(axis=1)
                            for child in children
                        ]
                    ) + 1e-3 * rng.random((len(rows), len(children)))
                wins[np.arange(len(rows)), np.argmax(scores, axis=1)] += 1
            probability = wins / iterations
            order = np.argsort(-probability, axis=1, kind="mergesort")
            for position, row in enumerate(rows):
                ranked = order[position]
                best = children[ranked[0]]
                assigned[row] = best
                record: dict[str, object] = {
                    "cell_id": cell_ids[row],
                    "level": level,
                    "level_name": level.lower(),
                    "assignment": best,
                    "name": best,
                    "bp": float(probability[position, ranked[0]]),
                    "aggregate_probability": float(probability[position, ranked[0]]),
                    "avg_correlation": 0.5,
                    "directly_assigned": True,
                    "n_runners_up": len(children) - 1,
                }
                for rank in range(1, N_RUNNERS_UP + 1):
                    if rank < len(children):
                        other = children[ranked[rank]]
                        record[runner_up_column(rank, "assignment")] = other
                        record[runner_up_column(rank, "name")] = other
                        record[runner_up_column(rank, "probability")] = float(
                            probability[position, ranked[rank]]
                        )
                        record[runner_up_column(rank, "correlation")] = 0.1
                    else:
                        for field_name in ("assignment", "name"):
                            record[runner_up_column(rank, field_name)] = None
                        for field_name in ("probability", "correlation"):
                            record[runner_up_column(rank, field_name)] = math.nan
                records.append(record)
        parents = assigned
    return pd.DataFrame.from_records(records)


def synthetic_specs() -> list[res.LevelSpec]:
    class_of = {cls: cls for cls in CLASSES}
    subclass_class = {subclass: cls for cls in CLASSES for subclass in SUBCLASSES[cls]}
    cluster_class = {cluster: cls for cls, _, cluster in cluster_labels()}
    return [
        res.LevelSpec(
            meta=res.LevelMeta("class", "L1", "broad", 0.7, 0.9, None),
            calls=res.node_level_calls("L1", class_of),
            truth=res.mapped_truth("L1", None, class_of),
        ),
        res.LevelSpec(
            meta=res.LevelMeta("subclass", "L2", "other", 0.6, 0.85, None),
            calls=res.node_level_calls("L2", subclass_class),
            truth=res.mapped_truth("L2", None, class_of, parent_level="L1"),
        ),
        res.LevelSpec(
            meta=res.LevelMeta("cluster", "L3", "leaf", 0.6, 0.85, None),
            calls=res.node_level_calls("L3", cluster_class),
            truth=res.mapped_truth("L3", None, class_of, parent_level="L1"),
        ),
    ]


GRID = (10, 30, 100)


def settings(**updates: object) -> res.RuleSettings:
    return res.RuleSettings(**updates)  # type: ignore[arg-type]


def recipes() -> list[res.SimulationRecipe]:
    return res.simulation_recipes(AnnotationResolvabilityConfig())


@pytest.fixture(scope="module")
def synthetic_run() -> res.ResolvabilityResult:
    test = make_test_cells()
    return res.run_resolvability(
        test,
        specs=synthetic_specs(),
        depths=GRID,
        recipes=recipes(),
        map_fn=bootstrap_mapper,
        settings=settings(),
        species="human",
    )


# --------------------------------------------------------------------------
# Statistics


def test_wilson_lower_bound_matches_known_values() -> None:
    assert res.wilson_lower_bound(0.9, 100) == pytest.approx(0.8256, abs=1e-4)
    assert res.wilson_lower_bound(1.0, 73) == pytest.approx(0.9500, abs=2e-4)
    assert math.isnan(res.wilson_lower_bound(0.9, 0))


def test_kish_effective_n() -> None:
    assert res.kish_effective_n(np.ones(10)) == pytest.approx(10)
    assert res.kish_effective_n(np.array([1.0, 0.0, 0.0])) == pytest.approx(1)
    assert res.kish_effective_n(np.array([3.0, 1.0])) == pytest.approx(16 / 10)


def test_isotonic_fit_matches_scikit_learn() -> None:
    isotonic = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(3)
    x = np.round(rng.random(500), 2)
    y = (rng.random(500) < x).astype(float)
    weights = rng.random(500) + 0.1
    ours = res.isotonic_fit(x, y, weights)
    reference = isotonic.IsotonicRegression(
        y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
    ).fit(x, y, sample_weight=weights)
    grid = np.linspace(-0.1, 1.1, 241)
    np.testing.assert_allclose(ours.predict(grid), reference.predict(grid), atol=1e-9)
    assert np.all(np.diff(ours.y) >= -1e-12)


def test_depth_bin_is_the_largest_grid_value_at_or_below_the_counts() -> None:
    bins = res.depth_bin([5, 10, 14, 15, 299, 5000], [10, 15, 30, 60, 120, 250])
    assert np.isnan(bins[0])
    assert bins[1:].tolist() == [10, 10, 15, 250, 250]


# --------------------------------------------------------------------------
# Local rule, set-level rule, split halves, Wilson, raise-only


def bin_cells(
    bp: np.ndarray,
    correct: np.ndarray,
    *,
    level: str = "broad",
    cls: str = "X",
    depth: int = 30,
    half: np.ndarray | None = None,
    truth_parent: str | None = None,
    leaf: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(bp)
    cell_ids = [f"{cls}{depth}_{index}" for index in range(n)]
    return pd.DataFrame(
        {
            "recipe": res.DECISION_RECIPE,
            "seed": 0,
            "level": level,
            "sim_id": [f"{cell}|D{depth}" for cell in cell_ids],
            "cell_id": cell_ids,
            "depth": depth,
            "half": (np.arange(n) % 2) if half is None else half,
            "parent": cls,
            "call": np.where(correct, "right", "wrong"),
            "bp": bp,
            "corr": 0.5,
            "truth": "right",
            "truth_parent": truth_parent or cls,
            res.TRUTH_LEAF_COLUMN: "T" if leaf is None else leaf,
            "correct": correct,
            "total_counts": float(depth),
        }
    )


BROAD = res.LevelMeta("broad", "SUPC", "broad", 0.70, 0.90, "broad")


def test_local_rule_rejects_a_near_threshold_error_band_the_set_rule_passes() -> None:
    rng = np.random.default_rng(7)
    bp = np.round(0.70 + 0.30 * rng.random(8000), 3)
    band = bp < 0.74
    correct = np.where(band, rng.random(8000) < 0.55, rng.random(8000) < 0.99)
    # The set-level rule accepts everything from the default: the set is
    # precise enough although the cells just above the threshold are not.
    set_threshold = res.set_level_threshold(bp, correct, target=0.90, floor=0.70)
    assert set_threshold == pytest.approx(0.70, abs=1e-6)
    assert correct[bp >= set_threshold].mean() >= 0.90
    assert correct[band].mean() < 0.65
    decisions = res.decide(bin_cells(bp, correct), [BROAD], [30], settings())
    local = decisions[decisions["regime"] == "trust"].iloc[0]
    assert local["t_star"] >= 0.735
    assert local["status"] == res.STATUS_EMITTED
    # Cells just above the local threshold meet the target, unlike the band.
    near = (bp >= local["t_star"]) & (bp < local["t_star"] + 0.03)
    assert correct[near].mean() >= 0.90


def test_thresholds_are_chosen_on_one_half_and_checked_on_the_other() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    bp = np.round(0.70 + 0.30 * rng.random(n), 3)
    half = np.arange(n) % 2
    # Fit half: perfect; check half: 70% correct everywhere.
    correct = np.where(half == 0, True, rng.random(n) < 0.70)
    decisions = res.decide(bin_cells(bp, correct, half=half), [BROAD], [30], settings())
    trust = decisions[decisions["regime"] == "trust"].iloc[0]
    assert trust["t_star"] == pytest.approx(0.70)
    assert trust["status"] == res.STATUS_NOT_RESOLVABLE
    assert trust["reason"] == "wilson_bound_below_target"
    # t* is checked on the check half only, and fitted on the fit half only.
    assert trust["check_set"] == "check_half"
    assert trust["n_called"] == int((half == 1).sum())
    assert trust["n_fit"] == int((half == 0).sum())
    # Without split halves the fit sees the check half too (85% correct at
    # every bp), so no local threshold reaches the target.
    pooled = res.decide(
        bin_cells(bp, correct, half=half),
        [BROAD],
        [30],
        settings(split_halves=False),
    )
    pooled_trust = pooled[pooled["regime"] == "trust"].iloc[0]
    assert pooled_trust["t_star"] is None or pd.isna(pooled_trust["t_star"])
    assert pooled_trust["reason"] == "no_local_threshold"


def test_wilson_bound_and_confident_count_gate_emission() -> None:
    # 60 confident, all correct: Wilson 0.940 passes 0.90 - 0.02 but not the
    # provisional target below 60 counts (0.97 cap -> 0.95 needed).
    bp = np.full(120, 0.95)
    correct = np.ones(120, dtype=bool)
    decisions = res.decide(bin_cells(bp, correct), [BROAD], [30], settings())
    by_regime = decisions.set_index("regime")
    assert by_regime.loc["trust", "n_confident"] == 60
    assert by_regime.loc["trust", "wilson_lb"] == pytest.approx(0.9398, abs=1e-3)
    assert by_regime.loc["trust", "status"] == res.STATUS_EMITTED
    assert by_regime.loc["provisional", "target"] == pytest.approx(0.97)
    assert by_regime.loc["provisional", "status"] == res.STATUS_NOT_RESOLVABLE
    assert by_regime.loc["provisional", "reason"] == "wilson_bound_below_target"
    # The bin is the class's only one, so pooling cannot add calls: the class
    # never reaches min_confident_n (insufficient_calls; the bin's own
    # reason is kept beside it).
    few = res.decide(
        bin_cells(bp, correct), [BROAD], [30], settings(min_confident_n=70)
    ).set_index("regime")
    assert few.loc["trust", "reason"] == res.REASON_INSUFFICIENT_CALLS
    assert few.loc["trust", "own_reason"] == "too_few_confident_calls"
    unfit = res.decide(
        bin_cells(bp[:80], correct[:80]), [BROAD], [30], settings()
    ).set_index("regime")
    assert unfit.loc["trust", "reason"] == res.REASON_INSUFFICIENT_CALLS
    assert unfit.loc["trust", "own_reason"] == "too_few_fit_cells"


def test_local_thresholds_are_raise_only() -> None:
    rng = np.random.default_rng(2)
    bp = np.round(0.30 + 0.70 * rng.random(3000), 3)
    correct = rng.random(3000) < 0.995
    decisions = res.decide(bin_cells(bp, correct), [BROAD], [30], settings())
    for regime in res.REGIMES:
        row = decisions[decisions["regime"] == regime].iloc[0]
        assert row["threshold"] >= BROAD.default_threshold - 1e-12
    trust = decisions[decisions["regime"] == "trust"].iloc[0]
    assert trust["t_star"] == pytest.approx(BROAD.default_threshold)
    assert not trust["would_raise"]
    fit = res.isotonic_fit(bp, correct.astype(float))
    assert res.local_threshold(fit, default=0.73, target=0.9, cap=0.99) == 0.73
    assert res.local_threshold(None, default=0.73, target=0.9, cap=0.99) is None


def test_validated_regime_keeps_the_default_and_reports_a_raise() -> None:
    rng = np.random.default_rng(5)
    bp = np.round(0.70 + 0.30 * rng.random(6000), 3)
    correct = np.where(bp < 0.80, rng.random(6000) < 0.8, rng.random(6000) < 0.99)
    decisions = res.decide(bin_cells(bp, correct), [BROAD], [30], settings())
    validated = decisions[decisions["regime"] == "validated"].iloc[0]
    assert validated["threshold"] == pytest.approx(0.70)
    assert validated["would_raise"]
    assert validated["t_star"] > 0.75
    # The default is fitted on no test cell: every call of the bin checks it.
    assert validated["check_set"] == "all"
    assert validated["n_called"] == 6000
    assert validated["n_confident"] == int((bp >= 0.70).sum())
    assert validated["precision"] == pytest.approx(correct.mean())
    trust = decisions[decisions["regime"] == "trust"].iloc[0]
    assert trust["n_called"] == 3000


def test_validated_regime_checks_the_default_on_both_halves() -> None:
    # 90 calls, all correct: 45 per half is below min_confident_n (and the
    # fit minimum), but the default needs no fit, so all 90 check it.
    bp = np.full(90, 0.95)
    correct = np.ones(90, dtype=bool)
    decisions = res.decide(bin_cells(bp, correct), [BROAD], [30], settings())
    by_regime = decisions.set_index("regime")
    assert by_regime.loc["validated", "status"] == res.STATUS_EMITTED
    assert by_regime.loc["validated", "n_confident"] == 90
    assert by_regime.loc["validated", "wilson_lb"] == pytest.approx(
        res.wilson_lower_bound(1.0, 90)
    )
    assert by_regime.loc["trust", "status"] == res.STATUS_NOT_RESOLVABLE
    assert by_regime.loc["trust", "reason"] == res.REASON_INSUFFICIENT_CALLS
    assert by_regime.loc["trust", "own_reason"] == "too_few_fit_cells"
    # Without split halves every regime checks all calls.
    pooled = res.decide(
        bin_cells(bp, correct), [BROAD], [30], settings(split_halves=False)
    ).set_index("regime")
    assert set(pooled["check_set"]) == {"all"}


def test_emission_needs_the_minimum_coverage() -> None:
    # 100 precise confident calls, but 900 calls below the default: coverage
    # 0.1 < 0.2, so the bin is not emitted although precision passes.
    bp = np.concatenate([np.full(900, 0.5), np.full(100, 0.95)])
    correct = np.ones(1000, dtype=bool)
    decisions = res.decide(bin_cells(bp, correct), [BROAD], [30], settings())
    validated = decisions[decisions["regime"] == "validated"].iloc[0]
    assert validated["n_confident"] == 100
    assert validated["coverage"] == pytest.approx(0.1)
    assert validated["wilson_lb"] >= 0.88
    assert validated["reason"] == "coverage_below_minimum"
    assert validated["status"] == res.STATUS_NOT_RESOLVABLE
    relaxed = res.decide(
        bin_cells(bp, correct), [BROAD], [30], settings(min_coverage=0.05)
    )
    assert (
        relaxed[relaxed["regime"] == "validated"].iloc[0]["status"]
        == res.STATUS_EMITTED
    )


def test_wilson_bound_uses_the_kish_effective_n() -> None:
    # 60 correct confident calls pass the Wilson bound on the raw n, but five
    # of them carry most of the weight: on the Kish n (11.7) they fail.
    bp = np.full(60, 0.95)
    correct = np.ones(60, dtype=bool)
    weights = np.ones(60)
    weights[:5] = 20.0
    cells = bin_cells(bp, correct)
    raw = res.decide(cells, [BROAD], [30], settings()).set_index("regime")
    assert raw.loc["validated", "status"] == res.STATUS_EMITTED
    untrimmed = settings(weight_trim_factor=0.0)
    weighted = res.decide(cells, [BROAD], [30], untrimmed, weights=weights).set_index(
        "regime"
    )
    row = weighted.loc["validated"]
    assert row["n_confident"] == 60
    assert row["n_effective"] == pytest.approx(res.kish_effective_n(weights))
    assert row["n_effective"] == pytest.approx(155.0**2 / 2055.0)
    assert row["wilson_lb"] == pytest.approx(
        res.wilson_lower_bound(1.0, row["n_effective"])
    )
    assert row["reason"] == "wilson_bound_below_target"
    assert row["max_weight_share"] == pytest.approx(20.0 / 155.0)
    # The default trims the set's weights at 10 x its median (1): the five
    # heavy calls weigh 10, the Kish n is 105^2 / 555 and still fails.
    trimmed = res.decide(cells, [BROAD], [30], settings(), weights=weights)
    row = trimmed.set_index("regime").loc["validated"]
    assert row["n_effective"] == pytest.approx(105.0**2 / 555.0)
    assert row["max_weight_share"] == pytest.approx(10.0 / 105.0)
    assert row["reason"] == "wilson_bound_below_target"


# --------------------------------------------------------------------------
# D_max inheritance, extrapolation, not_resolvable, floors


def deep_and_shallow_cells() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    frames = []
    for cls, depths in (("Deep", (10, 30, 100)), ("Shallow", (10, 30))):
        for depth in depths:
            bp = np.round(0.75 + 0.25 * rng.random(400), 3)
            frames.append(bin_cells(bp, rng.random(400) < 0.99, cls=cls, depth=depth))
    # A class with 30 test cells only.
    bp = np.round(0.75 + 0.25 * rng.random(30), 3)
    frames.append(bin_cells(bp, np.ones(30, dtype=bool), cls="Rare", depth=10))
    return pd.concat(frames, ignore_index=True)


def test_bins_beyond_the_deepest_calls_take_the_pooled_verdict() -> None:
    decisions = res.decide(deep_and_shallow_cells(), [BROAD], GRID, settings())
    trust = decisions[decisions["regime"] == "trust"].set_index(["class", "depth"])
    assert trust.loc[("Shallow", 30), "d_max"] == 30
    # No Shallow cell reaches 100 counts: the deep-end pool reaches down to
    # 30, whose own test (400 calls) keeps its verdict; 100 takes the pool's.
    assert trust.loc[("Shallow", 100), "extrapolated"]
    assert trust.loc[("Shallow", 100), "pooled"]
    assert trust.loc[("Shallow", 100), "pool_min_depth"] == 30
    assert trust.loc[("Shallow", 100), "status"] == res.STATUS_EMITTED
    assert trust.loc[("Shallow", 100), "own_reason"] == "no_calls"
    assert not trust.loc[("Shallow", 30), "extrapolated"]
    assert not trust.loc[("Shallow", 30), "pooled"]
    assert not trust.loc[("Deep", 100), "extrapolated"]
    assert pd.isna(trust.loc[("Deep", 100), "pool_min_depth"])
    per_cell = res.cell_emission(
        decisions,
        "trust",
        "broad",
        ["Shallow", "Deep", "Shallow", None],
        [150, 150, 12, 150],
        GRID,
    )
    assert per_cell["emitted"].tolist() == [True, True, True, False]
    assert per_cell["resolvability_extrapolated"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert per_cell["depth_bin"].tolist()[:3] == [100, 100, 10]


def test_classes_without_enough_test_cells_are_not_resolvable() -> None:
    decisions = res.decide(deep_and_shallow_cells(), [BROAD], GRID, settings())
    rare = decisions[decisions["class"] == "Rare"]
    assert set(rare["status"]) == {res.STATUS_NOT_RESOLVABLE}
    assert set(rare["reason"]) == {"too_few_test_cells"}
    per_cell = res.cell_emission(
        decisions, "provisional", "broad", ["Rare"], [50], GRID
    )
    assert not per_cell["emitted"].iloc[0]
    floors = res.simulated_floors(decisions, "trust")
    assert floors[("broad", "Rare")] is None


# --------------------------------------------------------------------------
# Pooled deep bins (user decision 2026-09-27; one rule with gate P, §14)

POOL_GRID = (10, 30, 60, 100)


def tracked_cells(
    groups: Sequence[tuple[str, int, Mapping[int, tuple[float, np.ndarray | bool]]]],
    *,
    cls: str = "X",
    level: str = "broad",
) -> pd.DataFrame:
    """Cells simulated at several depths under one id (as the self-map does).

    Each group is ``(name, n_cells, {depth: (bp, correct)})``: its cells have
    a row at every listed depth, like test cells whose native counts reach
    the deepest of them.
    """
    frames = []
    for name, n_cells, depths in groups:
        cell_ids = [f"{name}_{index}" for index in range(n_cells)]
        for depth, (bp, correct) in depths.items():
            flags = np.broadcast_to(np.asarray(correct, dtype=bool), (n_cells,))
            frames.append(
                pd.DataFrame(
                    {
                        "recipe": res.DECISION_RECIPE,
                        "seed": 0,
                        "level": level,
                        "sim_id": [f"{cell}|D{depth}" for cell in cell_ids],
                        "cell_id": cell_ids,
                        "depth": depth,
                        "half": np.arange(n_cells) % 2,
                        "parent": cls,
                        "call": np.where(flags, "right", "wrong"),
                        "bp": bp,
                        "corr": 0.5,
                        "truth": "right",
                        "truth_parent": cls,
                        res.TRUTH_LEAF_COLUMN: "T",
                        "correct": flags,
                        "total_counts": float(depth),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def pooled_reference(deep_correct: np.ndarray | bool = True) -> pd.DataFrame:
    """A class whose 60 and 100 bins hold 45 calls each (< 50), 30 holds 85.

    ``deep``: 45 cells reaching 100 counts; ``mid``: 40 cells reaching 30;
    ``low``: 200 cells at 10 only. ``deep_correct`` sets the deep cells'
    correctness at 60 and 100 (their calls at 10 and 30 are right).
    """
    good = (0.95, True)
    deep = (0.95, deep_correct)
    return tracked_cells(
        [
            ("deep", 45, {10: good, 30: good, 60: deep, 100: deep}),
            ("mid", 40, {10: good, 30: good}),
            ("low", 200, {10: good}),
        ]
    )


def test_deep_bins_short_of_calls_are_emitted_extrapolated_when_the_pool_passes() -> (
    None
):
    cells = pooled_reference()
    decisions = res.decide(cells, [BROAD], POOL_GRID, settings())
    validated = decisions[decisions["regime"] == "validated"].set_index("depth")
    # 60 and 100 hold 45 confident calls each: neither is judged on its own.
    for depth in (60, 100):
        row = validated.loc[depth]
        assert row["own_n_confident"] == 45
        assert row["own_reason"] == "too_few_confident_calls"
        # The pool reaches down to 30: each test cell once, at its deepest
        # bin (45 deep cells at 100 + 40 mid cells at 30), all precise.
        assert row["status"] == res.STATUS_EMITTED
        assert row["pooled"] and row["extrapolated"]
        assert row["pool_min_depth"] == 30
        assert row["n_confident"] == 85
        assert row["wilson_lb"] == pytest.approx(res.wilson_lower_bound(1.0, 85))
    # D_P = 30 holds 85 calls itself: its own verdict, not extrapolated.
    assert validated.loc[30, "status"] == res.STATUS_EMITTED
    assert not validated.loc[30, "extrapolated"] and not validated.loc[30, "pooled"]
    assert not validated.loc[10, "pooled"]
    sets = res.pooled_sets(decisions, "validated")["broad"]["X"]
    assert sets["min_depth"] == 30 and sets["depths"] == [60, 100]
    assert sets["n_confident"] == 85 and sets["status"] == res.STATUS_EMITTED
    # Fitted regimes check only one half, so their pool reaches down to 10.
    trust = decisions[decisions["regime"] == "trust"].set_index("depth")
    assert trust.loc[100, "pool_min_depth"] == 10
    assert trust.loc[30, "pooled"] and trust.loc[30, "extrapolated"]
    # RESOLVE: cells of the pooled bins are emitted and flagged.
    per_cell = res.cell_emission(
        decisions, "validated", "broad", ["X", "X", "X"], [150, 70, 40], POOL_GRID
    )
    assert per_cell["emitted"].tolist() == [True, True, True]
    assert per_cell["resolvability_extrapolated"].tolist() == [True, True, False]


def test_deep_bins_short_of_calls_are_not_emitted_when_the_pool_fails() -> None:
    # 20 of the 45 deep cells are called wrongly at 60 and 100: the pool
    # (25 + 40 right of 85) fails the Wilson rule.
    wrong = np.arange(45) >= 20
    decisions = res.decide(pooled_reference(wrong), [BROAD], POOL_GRID, settings())
    validated = decisions[decisions["regime"] == "validated"].set_index("depth")
    for depth in (60, 100):
        row = validated.loc[depth]
        assert row["status"] == res.STATUS_NOT_RESOLVABLE
        assert row["reason"] == "wilson_bound_below_target"
        assert row["pooled"] and row["extrapolated"]
        assert row["precision"] == pytest.approx(65 / 85)
    # 30 passes on its own (85 right calls) but completes the failing pool:
    # a pooled failure withdraws it, a pooled pass could not have rescued it.
    assert validated.loc[30, "own_status"] == res.STATUS_EMITTED
    assert validated.loc[30, "status"] == res.STATUS_NOT_RESOLVABLE
    assert validated.loc[30, "reason"] == "pool_wilson_bound_below_target"
    # Bins shallower than D_P keep their own verdict.
    assert validated.loc[10, "status"] == res.STATUS_EMITTED
    floors = res.simulated_floors(decisions, "validated")
    assert floors[("broad", "X")] == 10


def test_a_class_short_of_calls_even_fully_pooled_is_insufficient() -> None:
    # 80 test cells reach 30 counts (D_max = 30), only 30 confident calls.
    cells = tracked_cells(
        [
            ("unsure", 50, {10: (0.5, True), 30: (0.5, True)}),
            ("sure", 30, {10: (0.95, True), 30: (0.95, True)}),
        ]
    )
    decisions = res.decide(cells, [BROAD], (10, 30), settings())
    assert set(decisions["d_max"].dropna()) == {30}
    assert set(decisions["status"]) == {res.STATUS_NOT_RESOLVABLE}
    assert set(decisions["reason"]) == {res.REASON_INSUFFICIENT_CALLS}
    validated = decisions[decisions["regime"] == "validated"]
    assert set(validated["own_reason"]) == {"too_few_confident_calls"}
    assert not validated["extrapolated"].any()


def test_the_pool_counts_each_cell_once_at_its_deepest_bin() -> None:
    # 60 cells are confident at 10 but not at their deepest bin (100); 30
    # are confident at both. Bin 10 is judged on its own (90 calls); the
    # pool of 100 and 10 holds each cell once, at 100: 30 confident calls,
    # so a shallower confident row never rescues an unconfident deep one.
    cells = tracked_cells(
        [
            ("drift", 60, {10: (0.95, True), 100: (0.5, True)}),
            ("steady", 30, {10: (0.95, True), 100: (0.95, True)}),
        ]
    )
    decisions = res.decide(cells, [BROAD], (10, 100), settings())
    validated = decisions[decisions["regime"] == "validated"].set_index("depth")
    assert validated.loc[10, "status"] == res.STATUS_EMITTED
    assert validated.loc[100, "reason"] == res.REASON_INSUFFICIENT_CALLS
    deepest = res.deepest_rows(cells)
    assert len(deepest) == 90 and set(deepest["depth"]) == {100}


def test_the_point_precision_joins_the_wilson_rule() -> None:
    # 20,000 confident calls at 89.5%: the Wilson bound (0.891) passes
    # 0.90 - 0.02, the point precision does not reach 0.90.
    n = 20_000
    correct = np.arange(n) < int(0.895 * n)
    decisions = res.decide(
        bin_cells(np.full(n, 0.95), correct), [BROAD], [30], settings()
    )
    validated = decisions[decisions["regime"] == "validated"].iloc[0]
    assert validated["wilson_lb"] >= 0.88
    assert validated["precision"] == pytest.approx(0.895)
    assert validated["status"] == res.STATUS_NOT_RESOLVABLE
    assert validated["reason"] == res.REASON_POINT
    passing = res.decide(
        bin_cells(np.full(n, 0.95), np.arange(n) < int(0.905 * n)),
        [BROAD],
        [30],
        settings(),
    )
    assert (
        passing[passing["regime"] == "validated"].iloc[0]["status"]
        == res.STATUS_EMITTED
    )


GOOD = (0.95, True)


def test_a_judged_bin_deeper_than_d_p_keeps_its_own_failure() -> None:
    # 60 "drift" cells are confident but wrong at 60 counts and unconfident
    # at 100: bin 60 fails on 90 calls of its own, bin 100 (30 calls) is not
    # judged and the pool reaches down to 30, where it passes. The pooled
    # pass must not override bin 60's own failure (M3b review 2).
    cells = tracked_cells(
        [
            ("drift", 60, {10: GOOD, 30: GOOD, 60: (0.95, False), 100: (0.5, True)}),
            ("steady", 30, {10: GOOD, 30: GOOD, 60: GOOD, 100: GOOD}),
            ("low", 200, {10: GOOD, 30: GOOD}),
        ]
    )
    decisions = res.decide(cells, [BROAD], POOL_GRID, settings())
    validated = decisions[decisions["regime"] == "validated"].set_index("depth")
    assert validated.loc[100, "pool_min_depth"] == 30
    sixty = validated.loc[60]
    assert sixty["own_n_confident"] == 90
    assert sixty["status"] == res.STATUS_NOT_RESOLVABLE
    assert sixty["reason"] == res.REASON_WILSON
    assert not sixty["pooled"] and not sixty["extrapolated"]
    assert sixty["precision"] == pytest.approx(30 / 90)
    deep = validated.loc[100]
    assert deep["status"] == res.STATUS_EMITTED
    assert deep["pooled"] and deep["extrapolated"]
    assert validated.loc[30, "status"] == res.STATUS_EMITTED
    assert not validated.loc[30, "pooled"]


def test_a_judged_bin_deeper_than_d_p_is_withdrawn_when_the_pool_fails() -> None:
    # Bin 60 passes on its own (90 right calls); the pool of >= 30 fails
    # (30 wrong confident calls at 100 of 230): bin 60 is withdrawn like D_P.
    cells = tracked_cells(
        [
            ("drift", 60, {10: GOOD, 30: GOOD, 60: GOOD, 100: (0.5, True)}),
            ("steady", 30, {10: GOOD, 30: GOOD, 60: GOOD, 100: (0.95, False)}),
            ("low", 200, {10: GOOD, 30: GOOD}),
        ]
    )
    decisions = res.decide(cells, [BROAD], POOL_GRID, settings())
    validated = decisions[decisions["regime"] == "validated"].set_index("depth")
    assert validated.loc[100, "pool_min_depth"] == 30
    for depth in (30, 60):
        row = validated.loc[depth]
        assert row["own_status"] == res.STATUS_EMITTED
        assert row["status"] == res.STATUS_NOT_RESOLVABLE
        assert row["reason"] == f"{res.POOL_REASON_PREFIX}{res.REASON_WILSON}"
        assert not row["pooled"] and not row["extrapolated"]
    assert validated.loc[100, "status"] == res.STATUS_NOT_RESOLVABLE
    assert validated.loc[100, "pooled"] and validated.loc[100, "extrapolated"]
    assert validated.loc[10, "status"] == res.STATUS_EMITTED


def test_an_unjudged_d_p_is_pooled_but_not_extrapolated() -> None:
    # 30 "deep" cells are unconfident at 60 and confident at 100; 30 "mid"
    # cells reach 60. Neither 100 (30 calls) nor 60 (30 confident calls) is
    # judged; the >= 60 pool (each cell at its deepest bin: 60 calls) is.
    cells = tracked_cells(
        [
            ("deep", 30, {10: GOOD, 30: GOOD, 60: (0.5, True), 100: GOOD}),
            ("mid", 30, {10: GOOD, 30: GOOD, 60: GOOD}),
            ("low", 200, {10: GOOD, 30: GOOD}),
        ]
    )
    decisions = res.decide(cells, [BROAD], POOL_GRID, settings())
    validated = decisions[decisions["regime"] == "validated"].set_index("depth")
    assert validated.loc[60, "pool_min_depth"] == 60
    assert validated.loc[60, "own_n_confident"] == 30
    assert validated.loc[60, "pooled"] and not validated.loc[60, "extrapolated"]
    assert validated.loc[60, "n_confident"] == 60
    assert validated.loc[100, "pooled"] and validated.loc[100, "extrapolated"]
    assert set(validated.loc[[60, 100], "status"]) == {res.STATUS_EMITTED}
    per_cell = res.cell_emission(
        decisions, "validated", "broad", ["X", "X"], [70, 150], POOL_GRID
    )
    assert per_cell["resolvability_extrapolated"].tolist() == [False, True]


def test_only_positive_weight_calls_count_towards_min_confident_n() -> None:
    # Bin 30: 60 confident calls, 55 of types the dataset lacks (weight 0).
    # They add nothing to the precision, so the bin holds 5 calls: it is not
    # judged on its own and takes the verdict of the >= 10 pool.
    cells = tracked_cells(
        [
            ("absent", 55, {10: GOOD, 30: (0.95, False)}),
            ("present", 5, {10: GOOD, 30: GOOD}),
            ("low", 200, {10: GOOD}),
        ]
    )
    weights = np.where(
        (cells["depth"] == 30) & cells["cell_id"].str.startswith("absent"), 0.0, 1.0
    )
    stats = res.check_threshold(
        np.full(60, 0.95), np.ones(60), np.r_[np.zeros(55), np.ones(5)], 0.7
    )
    assert (stats.n_called, stats.n_confident) == (5, 5)
    decisions = res.decide(cells, [BROAD], (10, 30), settings(), weights=weights)
    validated = decisions[decisions["regime"] == "validated"].set_index("depth")
    row = validated.loc[30]
    assert row["own_n_confident"] == 5
    assert row["own_reason"] == res.REASON_TOO_FEW_CONFIDENT
    assert row["pooled"] and row["extrapolated"] and row["pool_min_depth"] == 10
    assert row["status"] == res.STATUS_EMITTED
    # Unweighted, the 60 calls are judged on their own and fail.
    unweighted = res.decide(cells, [BROAD], (10, 30), settings())
    own = unweighted[unweighted["regime"] == "validated"].set_index("depth").loc[30]
    assert own["n_confident"] == 60 and own["status"] == res.STATUS_NOT_RESOLVABLE
    assert not own["pooled"]


def test_would_raise_needs_an_isotonic_fit() -> None:
    rng = np.random.default_rng(21)
    # 70 calls: 35 in the fit half (< min_cells_per_bin), no fit, t* unknown.
    small = bin_cells(np.full(70, 0.95), np.ones(70, bool), cls="Small")
    # 800 calls with an error band just above the default: the fit raises.
    bp = np.round(0.70 + 0.30 * rng.random(800), 3)
    correct = np.where(bp < 0.8, rng.random(800) < 0.5, True)
    large = bin_cells(bp, correct, cls="Large")
    large["cell_id"] = [f"L{index}" for index in range(800)]
    decisions = res.decide(
        pd.concat([small, large], ignore_index=True), [BROAD], [30], settings()
    )
    validated = decisions[decisions["regime"] == "validated"].set_index("class")
    assert validated.loc["Small", "n_fit"] == 35
    assert not validated.loc["Small", "would_raise"]
    assert not validated.loc["Small", "would_raise_evaluable"]
    assert validated.loc["Large", "would_raise"]
    assert validated.loc["Large", "would_raise_evaluable"]
    raised, unevaluable = res.would_raise_bins(decisions, settings())
    assert raised["class"].tolist() == ["Large"]
    assert unevaluable["class"].tolist() == ["Small"]


def glia_bin() -> pd.DataFrame:
    """A deep bin: 1,000 neuron calls, 120 calls of glial class G.

    60 G calls are right (type G1), 60 wrong (type G2, of class H).
    """
    neurons = bin_cells(
        np.full(1000, 0.95), np.ones(1000, bool), cls="N", leaf=np.full(1000, "N")
    )
    right = bin_cells(np.full(60, 0.95), np.ones(60, bool), cls="G", leaf="G1")
    wrong = bin_cells(np.full(60, 0.95), np.zeros(60, bool), cls="G", leaf="G2")
    right["cell_id"] = [f"G1_{index}" for index in range(60)]
    wrong["cell_id"] = [f"G2_{index}" for index in range(60)]
    wrong["truth_parent"] = "H"
    return pd.concat([neurons, right, wrong], ignore_index=True)


def test_weights_are_trimmed_within_the_judged_set() -> None:
    # The dataset's glia are mostly G1. Trimmed against the whole bin, whose
    # median is set by the 1,000 neuron calls, G1 and G2 hit the same cap
    # and the G precision stays near the unweighted 0.5 (M3b review 2);
    # trimmed within the called-G set, the composition ratio survives.
    cells = glia_bin()
    composition = {"N": 0.3, "G1": 0.65, "G2": 0.05}
    p_n, p_g = 1000 / 1120, 60 / 1120
    w_n, w_g1, w_g2 = 0.3 / p_n, 0.65 / p_g, 0.05 / p_g

    def g_row(scope: str) -> pd.Series:
        tables = res.ResolvabilityTables(
            summary={"depth_grid": [30], "decision_recipe": res.DECISION_RECIPE},
            cells=cells,
            levels=[BROAD],
            settings=settings(weight_trim_scope=scope),
        )
        frame = tables.decisions(composition=composition)
        return frame[(frame["regime"] == "validated") & (frame["class"] == "G")].iloc[0]

    cap = 10 * w_n
    assert g_row("bin")["precision"] == pytest.approx(cap / (cap + w_g2))
    assert g_row("judged_set")["precision"] == pytest.approx(w_g1 / (w_g1 + w_g2))
    assert g_row("judged_set")["precision"] > 0.9
    with pytest.raises(res.ResolvabilityError, match="weight_trim_scope"):
        settings(weight_trim_scope="class")


def test_floors_follow_the_max_rule() -> None:
    decisions = res.decide(deep_and_shallow_cells(), [BROAD], GRID, settings())
    table = pd.DataFrame(
        {
            "level": ["broad", "broad", "broad"],
            "floor_class": ["Deep", "Deep", "Shallow"],
            "platform": ["MERSCOPE", "XENIUM", "MERSCOPE"],
            "panel_family": ["f", "f", "f"],
            "min_counts": [15, 30, 10],
        }
    )
    floors = res.combined_floors(
        decisions, [BROAD], settings(), floor_table=table, species="human"
    ).set_index(["regime", "class"])
    assert floors.loc[("provisional", "Deep"), "known_floor"] == 30
    assert floors.loc[("provisional", "Deep"), "floor"] == 30
    # Validated panels keep the packaged floor of each platform, not the max.
    assert pd.isna(floors.loc[("validated", "Deep"), "floor"])
    assert floors.loc[("validated", "Deep"), "floor_source"] == "packaged"
    assert [
        (row["platform"], row["min_counts"])
        for row in floors.loc[("validated", "Deep"), "packaged"]
    ] == [("MERSCOPE", 15), ("XENIUM", 30)]
    assert floors.loc[("provisional", "Deep"), "packaged"] == []
    assert floors.loc[("provisional", "Rare"), "floor_source"] == "not_resolvable"
    # A simulated floor above the known one wins.
    shifted = decisions.copy()
    shifted.loc[
        (shifted["class"] == "Shallow") & (shifted["depth"] == 10), "status"
    ] = res.STATUS_NOT_RESOLVABLE
    floors = res.combined_floors(
        shifted, [BROAD], settings(), floor_table=table, species="human"
    ).set_index(["regime", "class"])
    assert floors.loc[("provisional", "Shallow"), "simulated_floor"] == 30
    assert floors.loc[("provisional", "Shallow"), "floor"] == 30


def test_validated_summary_floors_are_the_packaged_per_platform_values() -> None:
    from merxen.annotation.vocab import load_floor_table

    rng = np.random.default_rng(13)
    frames = [
        bin_cells(
            np.round(0.8 + 0.2 * rng.random(400), 3),
            rng.random(400) < 0.995,
            cls="Inh",
            depth=depth,
        )
        for depth in (10, 30)
    ]
    decisions = res.decide(
        pd.concat(frames, ignore_index=True), [BROAD], [10, 30], settings()
    )
    table = load_floor_table("human")
    floors = res.combined_floors(
        decisions, [BROAD], settings(), floor_table=table, species="human"
    )
    validated = floors[(floors["regime"] == "validated") & (floors["class"] == "Inh")]
    packaged = {
        row["platform"]: row["min_counts"] for row in validated.iloc[0]["packaged"]
    }
    expected = table[(table["level"] == "broad") & (table["floor_class"] == "Inh")]
    assert packaged == dict(
        zip(expected["platform"], expected["min_counts"].astype(int), strict=True)
    )
    assert packaged == {"MERSCOPE": 10, "XENIUM": 30}
    assert validated.iloc[0]["floor_source"] == "packaged"
    # The provisional (max-rule) floor stays the maximum over platforms.
    provisional = floors[
        (floors["regime"] == "provisional") & (floors["class"] == "Inh")
    ].iloc[0]
    assert provisional["known_floor"] == 30
    assert provisional["floor"] == 30


def test_mouse_subclass_floor_is_at_least_60_while_provisional() -> None:
    subclass = res.LevelMeta("subclass", "SUBC", "leaf", 0.8, 0.85, "subclass")
    rng = np.random.default_rng(9)
    cells = bin_cells(
        np.full(400, 0.95), rng.random(400) < 0.995, level="subclass", depth=10
    )
    decisions = res.decide(cells, [subclass], [10], settings())
    floors = res.combined_floors(
        decisions, [subclass], settings(), floor_table=None, species="mouse"
    ).set_index("regime")
    assert floors.loc["provisional", "simulated_floor"] == 10
    assert floors.loc["provisional", "floor"] == 60


# --------------------------------------------------------------------------
# Composition reweighting and trust constraints


def test_reweighting_to_a_dataset_composition_changes_emission() -> None:
    rng = np.random.default_rng(6)
    own = bin_cells(
        np.round(0.8 + 0.2 * rng.random(2000), 3),
        np.ones(2000, dtype=bool),
        leaf=np.full(2000, "Own"),
    )
    foreign = bin_cells(
        np.round(0.8 + 0.2 * rng.random(100), 3),
        np.zeros(100, dtype=bool),
        leaf=np.full(100, "Foreign"),
    )
    foreign["cell_id"] = [f"F{index}" for index in range(100)]
    foreign["half"] = np.arange(100) % 2
    cells = pd.concat([own, foreign], ignore_index=True)
    unweighted = res.decide(cells, [BROAD], [30], settings())
    assert unweighted[unweighted["regime"] == "trust"].iloc[0]["status"] == "emitted"
    weights = res.composition_weights(cells, {"Own": 0.4, "Foreign": 0.6})
    assert weights[: len(own)] == pytest.approx(0.4 / (2000 / 2100))
    assert weights[len(own) :] == pytest.approx(0.6 / (100 / 2100))
    # Bundles up to resolvability version 3 trimmed per bin: at 10 x the
    # bin's median (0.42 -> 4.2), renormalised to mean 1.
    legacy = res.composition_weights(
        cells, {"Own": 0.4, "Foreign": 0.6}, trim_factor=10
    )
    assert legacy.mean() == pytest.approx(1.0)
    assert legacy[: len(own)] == pytest.approx(0.7)
    assert legacy[len(own) :] == pytest.approx(7.0)
    # Version 4 trims within the judged set (here every call of the bin):
    # the same cap relative to the set's median.
    trimmed = res.trim_weights(weights, 10.0)
    weighted = res.decide(cells, [BROAD], [30], settings(), weights=weights)
    trust = weighted[weighted["regime"] == "trust"].iloc[0]
    assert trust["status"] == res.STATUS_NOT_RESOLVABLE
    assert trust["reason"] == "no_local_threshold"
    row = weighted[weighted["regime"] == "validated"].iloc[0]
    assert row["status"] == res.STATUS_NOT_RESOLVABLE
    check = cells["bp"].to_numpy() >= row["threshold"] - 1e-9
    assert row["n_confident"] == int(check.sum()) == 2100
    assert row["precision"] == pytest.approx(2000 * 0.7 / 2100)
    assert row["n_effective"] == pytest.approx(res.kish_effective_n(trimmed[check]))
    assert row["n_effective"] < row["n_confident"] - 1


def test_a_rare_type_heavy_in_the_composition_cannot_carry_a_bin() -> None:
    # 100 calls of class X: 98 of a common type, 2 of a rare one whose class
    # has no other type, and the dataset puts half its mass on the rare
    # type. Untrimmed, each rare cell would carry 25% of the bin.
    common = bin_cells(np.full(98, 0.95), np.ones(98, bool), leaf=np.full(98, "Own"))
    rare = bin_cells(np.full(2, 0.95), np.zeros(2, bool), leaf=np.full(2, "Rare"))
    rare["cell_id"] = ["R0", "R1"]
    rare["truth_parent"] = "Y"
    cells = pd.concat([common, rare], ignore_index=True)
    composition = {"Own": 0.5, "Rare": 0.5}
    raw = res.composition_weights(cells, composition)
    assert raw[-1] / raw.sum() == pytest.approx(0.25)
    assert res.trim_weights(raw, 10.0).max() / res.trim_weights(raw, 10.0).sum() <= 0.2
    decisions = res.decide(cells, [BROAD], [30], settings(), weights=raw)
    validated = decisions[decisions["regime"] == "validated"].iloc[0]
    assert validated["max_weight_share"] <= 0.20
    assert validated["n_effective"] < validated["n_confident"]


def test_rare_types_take_their_broad_class_weight() -> None:
    # Class A: a common type (60 cells) and a rare one (5 cells) that the
    # dataset composition inflates; class B: one common type.
    frames = {
        "A1": (60, "A"),
        "A2": (5, "A"),
        "B1": (35, "B"),
    }
    parts = []
    for leaf, (n, cls) in frames.items():
        part = bin_cells(np.full(n, 0.95), np.ones(n, bool), leaf=np.full(n, leaf))
        part["cell_id"] = [f"{leaf}_{index}" for index in range(n)]
        part["truth_parent"] = cls
        parts.append(part)
    cells = pd.concat(parts, ignore_index=True)
    assert res.leaf_class_map(cells) == {"A1": "A", "A2": "A", "B1": "B"}
    composition = {"A1": 0.3, "A2": 0.4, "B1": 0.3}
    weights = res.composition_weights(cells, composition, trim_factor=0)
    leaf = cells[res.TRUTH_LEAF_COLUMN].to_numpy()
    # A2 is weighted like A's common type A1, not by its own q / p (8.0).
    assert weights[leaf == "A2"] == pytest.approx(weights[leaf == "A1"][0])
    ratio = weights[leaf == "A1"][0] / weights[leaf == "B1"][0]
    assert ratio == pytest.approx((0.3 / 0.60) / (0.3 / 0.35))
    # With a per-type minimum of 1 the rare type keeps its own weight.
    own = res.composition_weights(cells, composition, trim_factor=0, min_type_cells=1)
    assert own[leaf == "A2"][0] / own[leaf == "A1"][0] == pytest.approx(
        (0.4 / 0.05) / (0.3 / 0.60)
    )


def test_composition_follows_the_dataset_cells_of_each_depth_bin() -> None:
    composition = res.DatasetComposition.from_cells(
        ["Own", "Foreign", "Own", "Foreign", "Own", None],
        [1.0, 1.0, 0.5, 0.5, 2.0, 1.0],
        [12, 14, 40, 45, 5, 40],
        [10, 30],
        min_bin_mass=1.5,
    )
    assert composition.overall == {"Own": 3.5, "Foreign": 1.5}
    assert composition.by_depth[10] == {"Own": 1.0, "Foreign": 1.0}
    assert composition.bin_mass == {10: 2.0, 30: 1.0}
    assert composition.at(10) == {"Own": 1.0, "Foreign": 1.0}
    # Bin 30 holds too little mass: the overall composition applies there.
    assert composition.at(30) == composition.overall
    rng = np.random.default_rng(14)
    frames = []
    for depth in (10, 30):
        own = bin_cells(
            np.full(200, 0.9), np.ones(200, bool), depth=depth, leaf=np.full(200, "Own")
        )
        foreign = bin_cells(
            np.full(50, 0.9),
            np.zeros(50, bool),
            depth=depth,
            leaf=np.full(50, "Foreign"),
        )
        foreign["cell_id"] = [f"F{depth}_{index}" for index in range(50)]
        foreign["truth_parent"] = "Z"
        frames += [own, foreign]
    cells = pd.concat(frames, ignore_index=True)
    per_bin = res.DatasetComposition(
        overall={"Own": 1.0, "Foreign": 1.0},
        by_depth={10: {"Own": 0.9, "Foreign": 0.1}, 30: {"Own": 0.2, "Foreign": 0.8}},
        bin_mass={10: 100.0, 30: 100.0},
    )
    weights = res.composition_weights(cells, per_bin, trim_factor=0)
    depth = cells["depth"].to_numpy()
    leaf = cells[res.TRUTH_LEAF_COLUMN].to_numpy()
    shallow = (
        weights[(depth == 10) & (leaf == "Foreign")].sum() / weights[depth == 10].sum()
    )
    deep = (
        weights[(depth == 30) & (leaf == "Foreign")].sum() / weights[depth == 30].sum()
    )
    assert shallow == pytest.approx(0.1)
    assert deep == pytest.approx(0.8)
    assert rng is not None


def test_a_pooled_set_is_reweighted_as_one_set() -> None:
    # 40 test cells of type A reach 100 counts (bin 100 short of calls), 60
    # stay at 10; the dataset has no A cell at 100 counts. Per-bin weights
    # give the pooled rows at 100 weight 0 (Kish n 60); reweighting the
    # ">= 10" set as one set to the dataset's cells at >= 10 counts keeps
    # all 100 test cells (Kish n 100).
    cells = pd.concat(
        [
            tracked_cells(
                [
                    ("deep", 40, {10: (0.95, True), 100: (0.95, True)}),
                    ("shallow", 60, {10: (0.95, True)}),
                ]
            ),
            tracked_cells(
                [("other", 100, {10: (0.95, True), 100: (0.95, True)})], cls="Y"
            ),
        ],
        ignore_index=True,
    )
    cells[res.TRUTH_LEAF_COLUMN] = np.where(cells["parent"] == "X", "A", "B")
    composition = res.DatasetComposition(
        overall={"A": 1.0, "B": 1.0},
        by_depth={10: {"A": 900.0, "B": 100.0}, 100: {"B": 1000.0}},
        bin_mass={10: 1000.0, 100: 1000.0},
    )
    assert res._normalised(composition.at_least(10)) == pytest.approx(
        {"A": 0.45, "B": 0.55}
    )
    assert composition.at_least(1000) == composition.overall
    class_of = {"A": "X", "B": "Y"}
    weights = res.composition_weights(cells, composition, class_of=class_of)
    grid = (10, 100)
    per_bin = res.decide(cells, [BROAD], grid, settings(), weights=weights)
    pooled = res.decide(
        cells,
        [BROAD],
        grid,
        settings(),
        weights=weights,
        pool_weights=res.pooled_composition_weights(composition, class_of=class_of),
    )
    for frame, kish in ((per_bin, 60.0), (pooled, 100.0)):
        row = frame[
            (frame["regime"] == "validated")
            & (frame["class"] == "X")
            & (frame["depth"] == 100)
        ].iloc[0]
        assert row["pooled"] and row["pool_min_depth"] == 10
        assert row["n_effective"] == pytest.approx(kish)
    # RESOLVE's entry point reweights pooled sets as one set.
    tables = res.ResolvabilityTables(
        summary={"depth_grid": list(grid), "decision_recipe": res.DECISION_RECIPE},
        cells=cells,
        levels=[BROAD],
        settings=settings(),
    )
    resolved = tables.decisions(composition=composition)
    row = resolved[
        (resolved["regime"] == "validated")
        & (resolved["class"] == "X")
        & (resolved["depth"] == 100)
    ].iloc[0]
    assert row["n_effective"] == pytest.approx(100.0)


def test_whb_cop_rule_suppresses_cop_broad_calls_below_the_floor() -> None:
    thresholds = AnnotationThresholds()
    rows = []
    for depth, sbp in ((30, 0.95), (120, 0.95), (120, 0.5), (250, 0.9)):
        for level, parent, bp in (
            ("supercluster", "COP", sbp),
            ("broad", "OPC", 0.99),
        ):
            rows.append(
                {
                    "recipe": res.DECISION_RECIPE,
                    "seed": 0,
                    "level": level,
                    "sim_id": f"c{depth}_{sbp}|D{depth}",
                    "cell_id": f"c{depth}_{sbp}",
                    "depth": depth,
                    "half": 0,
                    "parent": parent,
                    "call": "x",
                    "bp": bp,
                    "corr": 0.5,
                    "truth": "x",
                    "truth_parent": parent,
                    res.TRUTH_LEAF_COLUMN: "T",
                    "correct": True,
                    "total_counts": float(depth),
                }
            )
    # An OPC supercluster call is never touched.
    rows.append(dict(rows[0], parent="OPC", sim_id="o|D30", cell_id="o"))
    rows.append(dict(rows[1], sim_id="o|D30", cell_id="o"))
    cells = pd.DataFrame(rows)
    rule = res.whb_cop_rule(thresholds)
    out = rule(cells)
    broad = out[out["level"] == "broad"].set_index("sim_id")["parent"]
    assert broad["c30_0.95|D30"] is None  # below the 120-count COP floor
    assert broad["c120_0.95|D120"] == "OPC"
    assert broad["c120_0.5|D120"] is None  # supercluster bp below 0.69
    assert broad["c250_0.9|D250"] == "OPC"
    assert broad["o|D30"] == "OPC"
    # bp, supercluster rows and the input are unchanged.
    assert out["bp"].tolist() == cells["bp"].tolist()
    assert (
        out[out["level"] == "supercluster"]["parent"].tolist()
        == cells[cells["level"] == "supercluster"]["parent"].tolist()
    )
    assert cells[cells["level"] == "broad"]["parent"].eq("OPC").all()
    assert res.whb_cop_rule(thresholds, min_depth=10)(cells)[
        lambda frame: frame["level"] == "broad"
    ]["parent"].tolist() == ["OPC", "OPC", None, "OPC", "OPC"]


def test_a_planted_cop_sink_no_longer_fails_broad_opc() -> None:
    # Broad OPC calls: 300 correct OPC-supercluster calls and 150 COP-assigned
    # calls that are mostly wrong at 30 counts (a COP sink). Production
    # keeps COP calls below 120 counts at lineage, so the rule removes them
    # from the level (neither confident nor in the coverage) and broad OPC
    # passes.
    rng = np.random.default_rng(15)
    n_opc, n_cop = 300, 150
    broad = bin_cells(
        np.full(n_opc + n_cop, 0.95),
        np.concatenate([np.ones(n_opc, bool), rng.random(n_cop) < 0.2]),
        cls="OPC",
    )
    supercluster = broad.copy()
    supercluster["level"] = "supercluster"
    supercluster["parent"] = ["OPC"] * n_opc + ["COP"] * n_cop
    cells = pd.concat([broad, supercluster], ignore_index=True)
    before = res.decide(cells, [BROAD], [30], settings())
    assert before[before["regime"] == "validated"].iloc[0]["status"] == (
        res.STATUS_NOT_RESOLVABLE
    )
    after = res.decide(
        res.whb_cop_rule(AnnotationThresholds())(cells), [BROAD], [30], settings()
    )
    validated = after[after["regime"] == "validated"].iloc[0]
    assert validated["status"] == res.STATUS_EMITTED
    assert validated["n_confident"] == n_opc
    assert validated["n_called"] == n_opc
    assert validated["coverage"] == pytest.approx(1.0)
    # The class keeps its test cells (D_max is unchanged).
    unchanged = before[before["regime"] == "validated"].iloc[0]["n_test"]
    assert validated["n_test"] == unchanged


def test_trust_constraint_refuses_an_unresolvable_broad_level() -> None:
    rng = np.random.default_rng(8)
    cells = bin_cells(np.round(0.7 + 0.3 * rng.random(600), 3), rng.random(600) < 0.5)
    decisions = res.decide(cells, [BROAD], [30], settings())
    constraint = res.trust_constraint(decisions, [BROAD], settings())
    assert constraint.state == "refused"
    assert constraint.broad_emitted_bins == 0


# --------------------------------------------------------------------------
# Simulation


def test_thinning_uses_only_cells_that_reach_the_depth() -> None:
    test = make_test_cells(20, shallow_class="B")
    recipe, clean = recipes()
    query = res.thin_and_contaminate(test, [10, 30, 100], recipe)
    native = test.native_counts
    assert query.n_by_depth[100] == int((native >= 100).sum())
    assert query.n_by_depth[30] == int((native >= 30).sum())
    deep = query.obs[query.obs["depth"] == 100]
    assert set(deep["cell_id"]) == set(test.obs.index[native >= 100])
    # Host counts near D, spill at most 25% of D, partners from the other
    # class and deep enough to give that spill.
    assert deep["host_counts"].mean() == pytest.approx(100, rel=0.1)
    assert 0 < deep["spill_counts"].mean() <= 25 * 1.1
    assert (
        native[[list(test.obs.index).index(p) for p in deep["partner_id"]]] >= 25
    ).all()
    deep_only = res.thin_and_contaminate(make_test_cells(20), [100], recipe)
    assert deep_only.obs["spill_counts"].mean() == pytest.approx(25, rel=0.15)
    groups = test.obs[res.SPILL_GROUP_COLUMN]
    assert all(
        groups[host] != groups[partner]
        for host, partner in zip(deep["cell_id"], deep["partner_id"], strict=True)
    )
    clean_query = res.thin_and_contaminate(test, [10, 30, 100], clean)
    assert float(clean_query.obs["spill_counts"].sum()) == 0.0
    # Thinning never raises a gene's counts above the native counts.
    position = {cell: index for index, cell in enumerate(test.obs.index)}
    rows = [position[cell] for cell in clean_query.obs["cell_id"]]
    assert (clean_query.counts - test.counts[rows]).max() <= 0
    # Deterministic for a seed.
    again = res.thin_and_contaminate(test, [10, 30, 100], recipe)
    assert (again.counts != query.counts).nnz == 0


def cells_subset(test: res.HeldOutCells, cell_ids: Sequence[str]) -> res.HeldOutCells:
    """The test cells ``cell_ids``, in that order."""
    position = {str(cell): index for index, cell in enumerate(test.obs.index)}
    rows = [position[str(cell)] for cell in cell_ids]
    return res.HeldOutCells(
        counts=sparse.csr_matrix(test.counts[rows]),
        genes=list(test.genes),
        obs=test.obs.iloc[rows].copy(),
    )


def renamed(test: res.HeldOutCells, prefix: str) -> res.HeldOutCells:
    obs = test.obs.copy()
    obs.index = [f"{prefix}{cell}" for cell in obs.index]
    return res.HeldOutCells(counts=test.counts, genes=list(test.genes), obs=obs)


def combined(first: res.HeldOutCells, second: res.HeldOutCells) -> res.HeldOutCells:
    return res.HeldOutCells(
        counts=sparse.vstack([first.counts, second.counts]).tocsr(),
        genes=list(first.genes),
        obs=pd.concat([first.obs, second.obs]),
    )


def simulated_by_id(
    query: res.SimulatedQuery,
) -> dict[str, tuple[np.ndarray, str, float]]:
    """Simulated id -> (counts, spill partner, host counts)."""
    dense = query.counts.toarray()
    return {
        str(sim_id): (dense[index], str(partner), float(host))
        for index, (sim_id, partner, host) in enumerate(
            zip(
                query.obs.index,
                query.obs["partner_id"],
                query.obs["host_counts"],
                strict=True,
            )
        )
    }


def test_a_simulated_cell_does_not_depend_on_the_other_test_cells() -> None:
    # Resolvability version 5: every draw is keyed by the cell id, the depth,
    # the recipe and the seed, so rebuilding a test set with other cells (or
    # another order) redraws only the cells whose spill partner changed.
    test = make_test_cells(20)
    recipe, clean = recipes()
    ids = [str(cell) for cell in test.obs.index]
    full = simulated_by_id(res.thin_and_contaminate(test, GRID, recipe))
    full_clean = simulated_by_id(res.thin_and_contaminate(test, GRID, clean))
    # Another order: every simulated cell is identical.
    order = [ids[index] for index in np.random.default_rng(1).permutation(len(ids))]
    for recipe_used, expected in ((recipe, full), (clean, full_clean)):
        again = simulated_by_id(
            res.thin_and_contaminate(cells_subset(test, order), GRID, recipe_used)
        )
        assert again.keys() == expected.keys()
        for sim_id, (row, partner, _) in expected.items():
            assert np.array_equal(again[sim_id][0], row)
            assert again[sim_id][1] == partner
    # Cells removed: every kept cell whose partners are kept is identical;
    # one whose partner was removed keeps its host thinning and takes a new
    # partner among the kept cells.
    core = {ids[index] for index in range(0, len(ids), 3)}
    partners = {
        partner
        for sim_id, (_, partner, _) in full.items()
        if sim_id.split("|")[0] in core
    }
    kept = [cell for cell in ids if cell in core or cell in partners]
    assert len(kept) < len(ids)
    fewer = simulated_by_id(
        res.thin_and_contaminate(cells_subset(test, kept), GRID, recipe)
    )
    for sim_id, (row, partner, host) in fewer.items():
        if sim_id.split("|")[0] in core:
            assert np.array_equal(row, full[sim_id][0])
            assert partner == full[sim_id][1]
        assert host == full[sim_id][2]
        assert partner in kept
    moved = [sim_id for sim_id, value in fewer.items() if value[1] != full[sim_id][1]]
    assert moved and all(full[sim_id][1] not in kept for sim_id in moved)
    # Without spill (clean) any subset leaves every kept cell unchanged.
    half = ids[::2]
    fewer_clean = simulated_by_id(
        res.thin_and_contaminate(cells_subset(test, half), GRID, clean)
    )
    assert fewer_clean and all(
        np.array_equal(row, full_clean[sim_id][0])
        for sim_id, (row, _, _) in fewer_clean.items()
    )
    # Cells added: an original cell changes only when an added cell became its
    # partner (it outranks the old one), and then only its spill.
    extra = renamed(make_test_cells(20, seed=5), "new_")
    more = simulated_by_id(
        res.thin_and_contaminate(combined(test, extra), GRID, recipe)
    )
    changed = 0
    for sim_id, (row, partner, host) in full.items():
        after_row, after_partner, after_host = more[sim_id]
        assert after_host == host
        if after_partner == partner:
            assert np.array_equal(after_row, row)
        else:
            assert after_partner.startswith("new_")
            changed += 1
    # About half the hosts see an added cell outrank their partner (the
    # candidate pool doubles); never all of them.
    assert 0 < changed < len(full)


def test_identical_cells_get_independent_draws() -> None:
    # Copies of one cell under other ids: the draws are keyed per cell, so
    # they differ, are uncorrelated and have the binomial mean.
    base = make_test_cells(1)
    n_copies = 400
    counts = sparse.vstack([base.counts[0]] * n_copies).tocsr()
    obs = pd.concat([base.obs.iloc[[0]]] * n_copies)
    obs.index = [f"copy_{index}" for index in range(n_copies)]
    copies = res.HeldOutCells(counts=counts, genes=list(base.genes), obs=obs)
    _, clean = recipes()
    depth = 30
    query = res.thin_and_contaminate(copies, [depth], clean)
    dense = query.counts.toarray()
    assert len({tuple(row) for row in dense}) > 0.95 * n_copies
    native = np.asarray(base.counts[0].todense()).ravel()
    expected = native * depth / native.sum()
    assert np.allclose(dense.mean(axis=0), expected, atol=0.35)
    centred = dense - dense.mean(axis=0)
    correlation = np.corrcoef(centred[0::2].ravel(), centred[1::2].ravel())[0, 1]
    assert abs(correlation) < 0.05
    # Another seed or recipe version is another draw of the same cells.
    reseeded = res.thin_and_contaminate(
        copies, [depth], res.SimulationRecipe(**{**clean.to_json(), "seed": 1})
    )
    assert (reseeded.counts != query.counts).nnz > 0
    bumped = res.thin_and_contaminate(
        copies, [depth], res.SimulationRecipe(**{**clean.to_json(), "version": 2})
    )
    assert (bumped.counts != query.counts).nnz > 0
    # A row's stored gene order does not change its draw.
    row = sparse.csr_matrix(base.counts[0])
    order = np.argsort(-row.indices)
    unsorted = sparse.csr_matrix(
        (row.data[order], row.indices[order], row.indptr), shape=row.shape
    )
    assert not unsorted.has_sorted_indices
    one = res.HeldOutCells(
        counts=unsorted, genes=list(base.genes), obs=base.obs.iloc[[0]].copy()
    )
    sorted_query = res.thin_and_contaminate(
        cells_subset(base, [str(base.obs.index[0])]), [depth], clean
    )
    unsorted_query = res.thin_and_contaminate(one, [depth], clean)
    assert (unsorted_query.counts != sorted_query.counts).nnz == 0
    with pytest.raises(res.ResolvabilityError, match="unique"):
        res.thin_and_contaminate(combined(base, base), [depth], clean)


def key64(*parts: object) -> np.uint64:
    return np.uint64(res.draw_key(*parts) & 0xFFFFFFFFFFFFFFFF)


def test_rendezvous_partner_choice_is_uniform_and_stable() -> None:
    hosts = np.array([key64("host", index) for index in range(20_000)])
    candidates = np.array([key64("candidate", index) for index in range(10)])
    chosen = res.rendezvous_choice(hosts, candidates)
    counts = np.bincount(chosen, minlength=10)
    # Uniform: 2,000 each, sd 42.
    assert counts.min() > 1_800 and counts.max() < 2_200
    # The candidates' order does not matter.
    order = np.random.default_rng(0).permutation(10)
    assert np.array_equal(
        order[res.rendezvous_choice(hosts, candidates[order])], chosen
    )
    # Removing a candidate moves only the hosts that had chosen it.
    kept = np.array([index for index in range(10) if index != 3])
    after = kept[res.rendezvous_choice(hosts, candidates[kept])]
    assert np.array_equal(after != chosen, chosen == 3)
    # Adding one moves only the hosts it outranks (to it), about 1 in 11.
    added = res.rendezvous_choice(hosts, np.append(candidates, key64("candidate", 10)))
    assert np.array_equal(added != chosen, added == 10)
    assert 1_600 < int((added == 10).sum()) < 2_050
    with pytest.raises(res.ResolvabilityError, match="candidate"):
        res.rendezvous_choice(hosts, np.array([], dtype=np.uint64))


def test_gene_efficiency_is_the_pre_registered_draw() -> None:
    # Resolvability version 6: the pre-registered (version 1-4) draw, one
    # generator seeded by [seed, 0x6566] over the panel's gene order. The
    # first values are those stored in the version-4 set a WHB bundle
    # (d46aa309, 297 genes, seed 0).
    full = res.gene_efficiency(297, 0.8, 0)
    assert np.allclose(full[:3], [0.2727960777, 0.5819298935, 2.0298606441])
    rng = np.random.default_rng([0, 0x6566])
    expected = np.exp(rng.normal(0.0, 0.8, 297))
    assert np.array_equal(full, expected / np.median(expected))
    assert np.median(full) == pytest.approx(1.0)
    # Another sigma scales the same normals (a stress recipe is comparable).
    assert np.allclose(np.log(res.gene_efficiency(297, 1.0, 0)), 1.25 * np.log(full))
    assert not np.allclose(res.gene_efficiency(297, 0.8, 1), full)
    assert np.array_equal(res.gene_efficiency(297, None, 0), np.ones(297))
    assert len(res.gene_efficiency(0, 0.8, 0)) == 0


def test_the_gene_efficiency_does_not_depend_on_the_test_cells() -> None:
    # The thinning applies the one panel-wide efficiency whatever the test
    # cells: a cell's thinning probabilities, hence its draw, are the same in
    # a subset of the test cells (unless its spill partner changed).
    test = make_test_cells(20)
    recipe, _ = recipes()
    ids = [str(cell) for cell in test.obs.index]
    full = simulated_by_id(res.thin_and_contaminate(test, GRID, recipe))
    subset = simulated_by_id(
        res.thin_and_contaminate(cells_subset(test, ids[::2]), GRID, recipe)
    )
    shared = [sim_id for sim_id in subset if subset[sim_id][1] == full[sim_id][1]]
    assert shared
    assert all(np.array_equal(subset[sim_id][0], full[sim_id][0]) for sim_id in shared)


def test_each_stream_of_a_cell_has_its_own_key() -> None:
    # The key covers the depth, the recipe (seed, name, version) and the
    # stream: a spill that reused the host's thinning key, or a key without
    # the depth or the recipe name, would correlate draws that must be
    # independent.
    recipe, clean = recipes()
    cell = "cluster_7"
    base = res.cell_draw_key(recipe, cell, 10, res.DRAW_STREAM_THIN)
    others = [
        res.cell_draw_key(recipe, cell, 30, res.DRAW_STREAM_THIN),
        res.cell_draw_key(recipe, cell, 10, res.DRAW_STREAM_SPILL),
        res.cell_draw_key(recipe, cell, 10, res.DRAW_STREAM_PARTNER),
        res.cell_draw_key(recipe, "cluster_8", 10, res.DRAW_STREAM_THIN),
        res.cell_draw_key(
            res.SimulationRecipe(**{**recipe.to_json(), "name": "other"}),
            cell,
            10,
            res.DRAW_STREAM_THIN,
        ),
        res.cell_draw_key(
            res.SimulationRecipe(**{**recipe.to_json(), "seed": 1}),
            cell,
            10,
            res.DRAW_STREAM_THIN,
        ),
        res.cell_draw_key(
            res.SimulationRecipe(**{**recipe.to_json(), "version": 2}),
            cell,
            10,
            res.DRAW_STREAM_THIN,
        ),
    ]
    assert len({base, *others}) == 1 + len(others)
    assert res.cell_draw_key(recipe, cell, 10, res.DRAW_STREAM_THIN) == base
    # A renamed recipe is another draw of the same cells (clean: thinning
    # only; the spill recipe too).
    test = make_test_cells(10)
    for used in (clean, recipe):
        renamed_recipe = res.SimulationRecipe(**{**used.to_json(), "name": "other"})
        before = res.thin_and_contaminate(test, GRID, used)
        after = res.thin_and_contaminate(test, GRID, renamed_recipe)
        assert list(after.obs.index) == list(before.obs.index)
        assert (after.counts != before.counts).nnz > 0


def test_the_spill_is_thinned_with_its_own_stream() -> None:
    # Rebuild each simulated cell from its keys: host = its own row thinned
    # with the thin key, spill = the partner's row thinned with the host's
    # spill key. Thinning the partner with the host's thin key instead gives
    # another spill for most cells.
    test = make_test_cells(20)
    recipe, _ = recipes()
    query = res.thin_and_contaminate(test, GRID, recipe)
    efficiency = res.gene_efficiency(
        len(test.genes), recipe.gene_efficiency_sigma, recipe.seed
    )
    position = {str(cell): index for index, cell in enumerate(test.obs.index)}
    counts = sparse.csr_matrix(test.counts)
    rebuilt = 0
    thin_key_matches = 0
    for index, (cell, depth, partner) in enumerate(
        zip(
            query.obs["cell_id"].astype(str),
            query.obs["depth"].astype(int),
            query.obs["partner_id"].astype(str),
            strict=True,
        )
    ):
        amount = recipe.spill_fraction * depth
        host = res._thin_rows(
            counts[[position[cell]]],
            np.array([float(depth)]),
            efficiency,
            [res.cell_draw_key(recipe, cell, depth, res.DRAW_STREAM_THIN)],
        )
        spill = res._thin_rows(
            counts[[position[partner]]],
            np.array([amount]),
            efficiency,
            [res.cell_draw_key(recipe, cell, depth, res.DRAW_STREAM_SPILL)],
        )
        with_thin_key = res._thin_rows(
            counts[[position[partner]]],
            np.array([amount]),
            efficiency,
            [res.cell_draw_key(recipe, cell, depth, res.DRAW_STREAM_THIN)],
        )
        row = query.counts[[index]].toarray()
        assert np.array_equal(row, (host + spill).toarray())
        rebuilt += 1
        thin_key_matches += int(
            np.array_equal(spill.toarray(), with_thin_key.toarray())
        )
    assert rebuilt == len(query.obs)
    assert thin_key_matches < 0.2 * rebuilt


def test_select_test_cells_caps_strata_and_keeps_rare_types() -> None:
    obs = pd.DataFrame(
        {"type": ["big"] * 3000 + ["mid"] * 700 + ["rare"] * 12},
        index=[f"c{index}" for index in range(3712)],
    )
    chosen = res.select_test_cells(
        obs, stratum="type", max_per_stratum=1000, n_max=None, seed=0
    )
    counts = obs.loc[chosen, "type"].value_counts()
    assert counts.to_dict() == {"big": 1000, "mid": 700, "rare": 12}
    capped = res.select_test_cells(
        obs, stratum="type", max_per_stratum=1000, n_max=1212, seed=0
    )
    counts = obs.loc[capped, "type"].value_counts()
    assert counts["rare"] == 12
    assert counts.sum() <= 1212
    assert counts["big"] == counts["mid"] == 600
    assert res.stratum_cap(obs["type"].value_counts(), 1000, 1212) == 600


def test_top_up_fills_thin_strata_to_the_cap_within_the_room() -> None:
    # Extra candidates top up each stratum to the test set's cap (5) given
    # what the held-out donor already holds; "full" is at the cap already.
    candidates = pd.DataFrame(
        {"type": ["thin"] * 10 + ["none"] * 2 + ["new"] * 4 + ["full"] * 3},
        index=[f"x{index}" for index in range(19)],
    )
    have = {"thin": 3, "none": 0, "full": 5}
    chosen = res.top_up_test_cells(
        candidates, stratum="type", have=have, cap=5, room=None, seed=0
    )
    counts = candidates.loc[chosen, "type"].value_counts().to_dict()
    assert counts == {"thin": 2, "none": 2, "new": 4}
    # With room for 5 more cells the cap is lowered (water filling) to 3:
    # thin 3 + 0, none 0 + 2, new 0 + 3.
    fitted = res.top_up_test_cells(
        candidates, stratum="type", have=have, cap=5, room=5, seed=0
    )
    counts = candidates.loc[fitted, "type"].value_counts().to_dict()
    assert counts == {"none": 2, "new": 3}
    again = res.top_up_test_cells(
        candidates, stratum="type", have=have, cap=5, room=5, seed=0
    )
    assert list(again) == list(fitted)


# --------------------------------------------------------------------------
# End to end on the synthetic reference


def test_planted_non_resolvable_level_is_not_emitted(
    synthetic_run: res.ResolvabilityResult,
) -> None:
    decisions = synthetic_run.decisions
    emitted = res.emitted_summary(decisions, "trust")
    assert all(depths == [] for depths in emitted["cluster"].values())
    assert emitted["class"]["A"] and emitted["class"]["B"]
    assert 100 in emitted["subclass"]["A"]
    cells = synthetic_run.cells
    cluster_calls = cells[cells["level"] == "cluster"]
    assert cluster_calls["correct"].mean() < 0.65  # a coin flip between twins
    cluster = decisions[
        (decisions["level"] == "cluster") & (decisions["regime"] == "trust")
    ]
    assert set(cluster["status"]) == {res.STATUS_NOT_RESOLVABLE}
    # Twin clusters never reach the target: the fit finds no local
    # threshold, or only one a few lucky high-bp calls reach; at the default
    # the precision is a coin flip.
    # A class whose pooled deep set stays short of min_confident_n calls is
    # insufficient_calls (resolvability version 3).
    assert set(cluster["reason"]) <= {
        "no_local_threshold",
        "too_few_confident_calls",
        "insufficient_calls",
    }
    assert (cluster["n_confident"].fillna(0) < 50).all()
    validated = decisions[
        (decisions["level"] == "cluster") & (decisions["regime"] == "validated")
    ]
    assert set(validated["reason"]) == {"wilson_bound_below_target"}
    assert (validated["precision"] < 0.6).all()
    assert synthetic_run.trust.state == "broad_only"
    assert synthetic_run.trust.leaf_classes == ["A", "B"]


def test_run_resolvability_applies_cells_rules_and_logs_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def silence_class(cells: pd.DataFrame) -> pd.DataFrame:
        frame = cells.copy()
        frame.loc[frame["level"] == "class", "bp"] = 0.0
        return frame

    with caplog.at_level("INFO", logger=res.__name__):
        result = res.run_resolvability(
            make_test_cells(30),
            specs=synthetic_specs(),
            depths=GRID,
            recipes=recipes()[:1],
            map_fn=bootstrap_mapper,
            settings=settings(min_cells_per_bin=20, min_confident_n=20),
            species="human",
            cells_rules=[silence_class],
        )
    assert (result.cells[result.cells["level"] == "class"]["bp"] == 0).all()
    decisions = result.decisions
    assert set(decisions[decisions["level"] == "class"]["status"]) == {
        res.STATUS_NOT_RESOLVABLE
    }
    assert result.trust.state == "refused"
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert f"resolvability {res.DECISION_RECIPE}: simulated" in messages
    assert f"resolvability {res.DECISION_RECIPE}: mapped" in messages
    assert "resolvability trust decisions" in messages
    assert "resolvability trust constraint: refused" in messages


def test_self_map_files_round_trip(
    synthetic_run: res.ResolvabilityResult, tmp_path: Path
) -> None:
    written = synthetic_run.write(tmp_path)
    assert set(written.values()) == {
        res.RESOLVABILITY_FILE,
        res.RESOLVABILITY_CELLS_FILE,
        res.RESOLVABILITY_SUMMARY_FILE,
    }
    summary = json.loads((tmp_path / res.RESOLVABILITY_SUMMARY_FILE).read_text())
    assert summary["decision_recipe"] == res.DECISION_RECIPE
    assert summary["trust"]["state"] == "broad_only"
    assert summary["emission"]["trust"]["cluster"]["A"]["100"]["status"] == (
        res.STATUS_NOT_RESOLVABLE
    )
    table = pd.read_parquet(tmp_path / res.RESOLVABILITY_FILE)
    assert {"bin", "curve", "isotonic", "node", "confusion", "decision"} <= set(
        table["kind"].astype(str)
    )
    assert set(table["recipe"].dropna().astype(str)) == {res.DECISION_RECIPE, "clean"}
    cells = pd.read_parquet(tmp_path / res.RESOLVABILITY_CELLS_FILE)
    assert {"parent", "truth", "call", "bp", "depth", "level", "half"} <= set(
        cells.columns
    )
    tables = res.load_resolvability(tmp_path)
    assert tables is not None
    again = tables.decisions()
    pd.testing.assert_frame_equal(
        again.reset_index(drop=True)[["regime", "level", "class", "depth", "status"]],
        synthetic_run.decisions.reset_index(drop=True)[
            ["regime", "level", "class", "depth", "status"]
        ],
    )
    # RESOLVE reweights to a dataset composition over the truth leaves.
    weighted = tables.decisions(composition={"A1": 1.0, "A2": 1.0, "B1": 1.0})
    assert (weighted["level"] == "class").any()
    per_bin = tables.decisions(
        composition=res.DatasetComposition(
            overall={"A1": 1.0, "B1": 1.0},
            by_depth={100: {"A1": 1.0}},
            bin_mass={100: 500.0},
        )
    )
    deep_b = per_bin[
        (per_bin["depth"] == 100)
        & (per_bin["level"] == "class")
        & (per_bin["class"] == "B")
        & (per_bin["regime"] == "validated")
    ].iloc[0]
    # At 100 counts the dataset holds only A1, so B calls carry no weight
    # there; the overall composition (A1 and B1) still emits B at 100.
    assert deep_b["status"] == res.STATUS_NOT_RESOLVABLE
    overall = tables.decisions(composition={"A1": 1.0, "B1": 1.0})
    assert (
        overall[
            (overall["depth"] == 100)
            & (overall["level"] == "class")
            & (overall["class"] == "B")
            & (overall["regime"] == "validated")
        ].iloc[0]["status"]
        == res.STATUS_EMITTED
    )
    assert tables.settings.weight_trim_factor == pytest.approx(10.0)
    assert tables.settings.weight_trim_scope == "judged_set"
    assert summary["floors"]["validated"]["class"]["A"]["source"] == "hard_floor"
    assert res.load_resolvability(tmp_path / "missing") is None
    # PREP decides on the cells as stored (float32 bp): the stored summary
    # is exactly what RESOLVE re-derives from the stored table.
    for regime, per_level in summary["emission"].items():
        rows = again[again["regime"] == regime]
        for record in rows.to_dict("records"):
            stored = per_level[record["level"]][record["class"]][
                str(int(record["depth"]))
            ]
            assert stored["status"] == record["status"]
            assert stored["threshold"] == res._optional_float(record["threshold"])
            assert stored["t_star"] == res._optional_float(record["t_star"])
            assert stored["pooled"] == bool(record["pooled"])
            assert stored["extrapolated"] == bool(record["extrapolated"])
    assert synthetic_run.cells["bp"].dtype == np.float32
    # A bundle of resolvability version 3 did not record the trimming unit:
    # it trimmed per bin.
    legacy = dict(summary)
    legacy["settings"] = {
        key: value
        for key, value in summary["settings"].items()
        if key != "weight_trim_scope"
    }
    (tmp_path / res.RESOLVABILITY_SUMMARY_FILE).write_text(json.dumps(legacy))
    old = res.load_resolvability(tmp_path)
    assert old is not None and old.settings.weight_trim_scope == "bin"
    assert old.settings.bin_trim_factor == pytest.approx(10.0)
    assert old.settings.set_trim_factor == 0.0


def test_clean_recipe_is_an_upper_bound(synthetic_run: res.ResolvabilityResult) -> None:
    bins = synthetic_run.table[synthetic_run.table["kind"] == "bin"]
    subclass = bins[(bins["level"] == "subclass") & (bins["depth"] == 10)]
    by_recipe = subclass.groupby(subclass["recipe"].astype(str))["n_correct"].sum()
    totals = subclass.groupby(subclass["recipe"].astype(str))["n_called"].sum()
    accuracy = by_recipe / totals
    assert accuracy["clean"] >= accuracy[res.DECISION_RECIPE] - 0.02


# --------------------------------------------------------------------------
# Gate-P machinery


def test_gate_p_tested_sets_pool_deep_bins_and_count_each_cell_once() -> None:
    frames = []
    for depth, n in ((10, 300), (30, 120), (100, 90)):
        frames.append(bin_cells(np.full(n, 0.95), np.ones(n, dtype=bool), depth=depth))
    cells = pd.concat(frames, ignore_index=True)
    # The same test cells appear at 30 and 100: counted once in the pool.
    cells.loc[cells["depth"] == 100, "cell_id"] = [
        f"X30_{index}" for index in range(90)
    ]
    decisions = res.decide(
        cells, [BROAD], GRID, settings(min_confident_n=20, min_cells_per_bin=20)
    )
    sets = res.gate_p_tested_sets(cells, decisions, min_confident_n=100, regime="trust")
    tested = sets[("broad", "X")]
    assert tested is not None
    pooled = tested[0]
    assert pooled.pooled and pooled.depths == (30, 100)
    assert pooled.n_confident == 120
    # D_P (30) holds enough calls, so its own test is a tested set too.
    assert [item.depths for item in tested[1:]] == [(30,), (10,)]
    assert tested[1].n_confident == 120 and tested[2].n_confident == 300
    none = res.gate_p_tested_sets(
        cells, decisions, min_confident_n=10_000, regime="trust"
    )
    assert none[("broad", "X")] is None


def test_gate_p_machinery_keeps_one_replicate() -> None:
    cells = bin_cells(np.full(300, 0.95), np.ones(300, bool), depth=10)
    clean = cells.copy()
    clean["recipe"] = res.CLEAN_RECIPE
    clean["correct"] = False
    seed1 = cells.copy()
    seed1["seed"] = 1
    both = pd.concat([cells, clean, seed1], ignore_index=True)
    decisions = res.decide(cells, [BROAD], [10], settings())
    # The decision recipe and seed 0 are tested by default: the clean rows
    # (all wrong) and the seed-1 copies neither mix in nor double the count.
    sets = res.gate_p_tested_sets(both, decisions, min_confident_n=100, regime="trust")
    tested = sets[("broad", "X")]
    assert tested is not None
    assert [(item.depths, item.n_confident) for item in tested] == [((10,), 300)]
    assert tested[0].precision == pytest.approx(1.0)
    with pytest.raises(res.ResolvabilityError, match="more than once"):
        res.gate_p_tested_sets(both, decisions, recipe=None, seed=None)
    with pytest.raises(res.ResolvabilityError, match="pass recipe"):
        res.frozen_threshold_eval(both, decisions, regime="trust")
    frozen = res.frozen_threshold_eval(
        both, decisions, regime="trust", recipe=res.CLEAN_RECIPE, seed=0
    )
    assert frozen.iloc[0]["n_called"] == 300
    assert frozen.iloc[0]["precision"] == pytest.approx(0.0)


def test_gate_p_class_set_and_frozen_thresholds() -> None:
    members, share, ok = res.gate_p_class_set(
        ["Exc"] * 800 + ["Inh"] * 750 + ["Vascular"] * 37
    )
    assert members == ["Exc", "Inh"]
    assert share == pytest.approx(1550 / 1587)
    assert ok
    rng = np.random.default_rng(11)
    cells = bin_cells(np.round(0.7 + 0.3 * rng.random(800), 3), rng.random(800) < 0.97)
    decisions = res.decide(cells, [BROAD], [30], settings())
    replicate = cells.copy()
    replicate["correct"] = rng.random(800) < 0.6
    frozen = res.frozen_threshold_eval(replicate, decisions, regime="trust")
    assert frozen.iloc[0]["precision"] < 0.7
    assert frozen.iloc[0]["threshold"] == pytest.approx(
        decisions[decisions["regime"] == "trust"].iloc[0]["threshold"]
    )


def test_rule_settings_follow_the_config() -> None:
    rule = res.RuleSettings.from_config(
        AnnotationResolvabilityConfig(min_confident_n=70),
        AnnotationThresholds(provisional_target_cap=0.96),
    )
    assert rule.min_confident_n == 70
    assert rule.target("provisional", 0.9, 30) == pytest.approx(0.96)
    assert rule.target("provisional", 0.85, 60) == pytest.approx(0.90)
    assert rule.target("validated", 0.85, 30) == pytest.approx(0.85)


def test_fine_levels_need_the_opt_in_and_seed_stability() -> None:
    fine = res.LevelMeta("cluster", "CLUS", "fine", 0.69, 0.85, "supercluster")
    rng = np.random.default_rng(12)
    cells = bin_cells(
        np.round(0.8 + 0.2 * rng.random(600), 3),
        rng.random(600) < 0.995,
        level="cluster",
    )
    decisions = res.decide(cells, [fine], [30], settings())
    assert (decisions["status"] == res.STATUS_EMITTED).any()
    kwargs = {"regime": "trust", "fine_seed_stability": {"cluster": 0.01}}
    off = res.level_emission(decisions, [fine], "cluster", ["X"], [40], [30], **kwargs)
    assert not off["emitted"].iloc[0]
    assert off["reason"].iloc[0] == res.REASON_FINE_NOT_ENABLED
    on = res.level_emission(
        decisions,
        [fine],
        "cluster",
        ["X"],
        [40],
        [30],
        allow_fine_levels=True,
        **kwargs,
    )
    assert on["emitted"].iloc[0] and on["reason"].iloc[0] is None
    unstable = res.level_emission(
        decisions,
        [fine],
        "cluster",
        ["X"],
        [40],
        [30],
        regime="trust",
        allow_fine_levels=True,
        fine_seed_stability={"cluster": 0.05},
    )
    assert unstable["reason"].iloc[0] == res.REASON_FINE_SEED_UNSTABLE
    # Non-fine levels follow the table alone; untabulated levels are refused.
    broad = res.level_emission(
        res.decide(
            bin_cells(np.full(400, 0.95), np.ones(400, bool)), [BROAD], [30], settings()
        ),
        [BROAD],
        "broad",
        ["X", "Y"],
        [40, 40],
        [30],
        regime="trust",
    )
    assert broad["emitted"].tolist() == [True, False]
    assert broad["reason"].tolist() == [None, res.REASON_NOT_RESOLVABLE]
    missing = res.level_emission(
        decisions, [fine], "nt", ["X"], [40], [30], regime="trust"
    )
    assert missing["reason"].iloc[0] == res.REASON_NOT_TABULATED


def test_seed_stability_counts_changed_confident_labels() -> None:
    fine = res.LevelMeta("cluster", "CLUS", "fine", 0.69, 0.85, "supercluster")
    cells = bin_cells(np.full(400, 0.95), np.ones(400, bool), level="cluster")
    decisions = res.decide(cells, [fine], [30], settings())
    other = cells.copy()
    other.loc[other.index[:8], "call"] = "moved"
    assert res.seed_stability(cells, other, decisions, "cluster") == pytest.approx(
        8 / 400
    )
