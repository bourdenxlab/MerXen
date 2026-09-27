"""Tests for the resolvability self-map (plan §8.3, §5.4, §13; E2 verdict 3)."""

from __future__ import annotations

import json
import math
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
    """Map hierarchically by marker counts with a gene bootstrap (bf 0.5)."""
    rng = np.random.default_rng(seed + 11)
    counts = query.counts.toarray()
    records = []
    for row, cell_id in enumerate(query.obs.index):
        parent: str | None = None
        for level in LEVELS:
            children = _children(level, parent)
            wins = np.zeros(len(children))
            for _ in range(iterations):
                if level == "L3":
                    # No cluster markers: the bootstrap is a coin flip.
                    scores = rng.random(len(children))
                else:
                    scores = np.array(
                        [
                            counts[
                                row, rng.choice(MARKERS[child], 5, replace=False)
                            ].sum()
                            + rng.random() * 1e-3
                            for child in children
                        ]
                    )
                wins[int(np.argmax(scores))] += 1
            probability = wins / iterations
            order = np.argsort(-probability, kind="mergesort")
            best = children[order[0]]
            record = {
                "cell_id": cell_id,
                "level": level,
                "level_name": level.lower(),
                "assignment": best,
                "name": best,
                "bp": float(probability[order[0]]),
                "aggregate_probability": float(probability[order[0]]),
                "avg_correlation": 0.5,
                "directly_assigned": True,
                "n_runners_up": len(children) - 1,
            }
            for rank in range(1, N_RUNNERS_UP + 1):
                if rank < len(children):
                    other = children[order[rank]]
                    record[runner_up_column(rank, "assignment")] = other
                    record[runner_up_column(rank, "name")] = other
                    record[runner_up_column(rank, "probability")] = float(
                        probability[order[rank]]
                    )
                    record[runner_up_column(rank, "correlation")] = 0.1
                else:
                    for field_name in ("assignment", "name"):
                        record[runner_up_column(rank, field_name)] = None
                    for field_name in ("probability", "correlation"):
                        record[runner_up_column(rank, field_name)] = math.nan
            records.append(record)
            parent = best
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
    few = res.decide(
        bin_cells(bp, correct), [BROAD], [30], settings(min_confident_n=70)
    ).set_index("regime")
    assert few.loc["trust", "reason"] == "too_few_confident_calls"
    unfit = res.decide(
        bin_cells(bp[:80], correct[:80]), [BROAD], [30], settings()
    ).set_index("regime")
    assert unfit.loc["trust", "reason"] == "too_few_fit_cells"


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


def test_dmax_inheritance_marks_deeper_bins_extrapolated() -> None:
    decisions = res.decide(deep_and_shallow_cells(), [BROAD], GRID, settings())
    trust = decisions[decisions["regime"] == "trust"].set_index(["class", "depth"])
    assert trust.loc[("Shallow", 30), "d_max"] == 30
    assert trust.loc[("Shallow", 100), "extrapolated"]
    assert trust.loc[("Shallow", 100), "inherited_from"] == 30
    assert trust.loc[("Shallow", 100), "status"] == res.STATUS_EMITTED
    assert not trust.loc[("Deep", 100), "extrapolated"]
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
    assert floors.loc[("validated", "Deep"), "floor"] == 30
    assert floors.loc[("validated", "Deep"), "floor_source"] == "real_e2"
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
    assert weights[: len(own)].mean() == pytest.approx(0.4 / (2000 / 2100))
    weighted = res.decide(cells, [BROAD], [30], settings(), weights=weights)
    row = weighted[weighted["regime"] == "trust"].iloc[0]
    assert row["status"] == res.STATUS_NOT_RESOLVABLE
    assert row["n_effective"] < row["n_confident"] + 1e-9


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
    assert synthetic_run.trust.state == "broad_only"
    assert synthetic_run.trust.leaf_classes == ["A", "B"]


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
    assert res.load_resolvability(tmp_path / "missing") is None


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
    assert tested[1].depths == (10,) and tested[1].n_confident == 300
    none = res.gate_p_tested_sets(
        cells, decisions, min_confident_n=10_000, regime="trust"
    )
    assert none[("broad", "X")] is None


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
