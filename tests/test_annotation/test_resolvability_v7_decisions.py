"""Tests of the resolvability version-7 decisions (plan §8.3 v7.6-v7.10; M3c).

The ensemble rule (E1 on the pooled set with distinct test cells, E2 by
unanimity or the member spread), the saturated-bp rule, the monotone fill
with its non-neuronal limit, floors and trust from the decisions before the
fill, the class-depth table, the version guard of ``load_resolvability``, the
gate-P member helpers, the test-set top-up and the diagnostic comparison.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import resolvability as res
from merxen.annotation.config import AnnotationResolvabilityConfig

from .test_resolvability import bootstrap_mapper, make_test_cells, synthetic_specs

BROAD = res.LevelMeta("broad", "SUPC", "broad", 0.70, 0.90, "broad")
LEAF = res.LevelMeta("leaf", "SUBC", "leaf", 0.70, 0.90, "subclass")


def member_rows(
    member: str,
    bp: np.ndarray,
    correct: np.ndarray,
    *,
    cls: str = "X",
    depth: int = 100,
    level: str = "broad",
    cell_ids: Sequence[str] | None = None,
    half: np.ndarray | None = None,
    truth_parent: str | None = None,
) -> pd.DataFrame:
    """Cells-table rows of one member (same cell ids in every member)."""
    n = len(bp)
    ids = (
        list(cell_ids)
        if cell_ids is not None
        else [f"{cls}{depth}_{index}" for index in range(n)]
    )
    return pd.DataFrame(
        {
            "recipe": res.DECISION_RECIPE,
            "seed": 0,
            "level": level,
            "sim_id": [f"{cell}|D{depth}" for cell in ids],
            "cell_id": ids,
            "depth": depth,
            "half": (np.arange(n) % 2) if half is None else half,
            "parent": cls,
            "call": np.where(correct, "right", "wrong"),
            "bp": np.asarray(bp, dtype=np.float64),
            "corr": 0.5,
            "truth": "right",
            "truth_parent": truth_parent or cls,
            res.TRUTH_LEAF_COLUMN: "T",
            "correct": np.asarray(correct, dtype=bool),
            "total_counts": float(depth),
            res.MEMBER_COLUMN: member,
            res.MEMBER_ROLE_COLUMN: "emission",
        }
    )


def pattern(n: int, share_correct: float) -> np.ndarray:
    """Correctness with the same share on both split halves (cells 2k, 2k+1)."""
    ranks = np.arange(n) // 2
    # Evenly spread: rank r is correct when floor((r + 1) s) > floor(r s).
    return np.floor((ranks + 1) * share_correct + 1e-9) > np.floor(
        ranks * share_correct + 1e-9
    )


def decide(
    frames: Sequence[pd.DataFrame],
    members: Sequence[str],
    *,
    levels: Sequence[res.LevelMeta] = (BROAD,),
    depths: Sequence[int] = (100,),
    ensemble: res.EnsembleSettings | None = None,
    neuronal: Mapping[str, bool | None] | None = None,
    settings: res.RuleSettings | None = None,
) -> res.EnsembleDecisions:
    return res.ensemble_decide(
        pd.concat(frames, ignore_index=True),
        levels,
        depths,
        settings or res.RuleSettings(),
        ensemble or res.EnsembleSettings(),
        members=members,
        neuronal=neuronal,
    )


def row(
    result: res.EnsembleDecisions,
    *,
    regime: str = "trust",
    level: str = "broad",
    cls: str = "X",
    depth: int = 100,
) -> pd.Series:
    frame = result.decisions
    match = frame[
        (frame["regime"] == regime)
        & (frame["level"] == level)
        & (frame["class"] == cls)
        & (frame["depth"] == depth)
    ]
    assert len(match) == 1
    return match.iloc[0]


# --------------------------------------------------------------------------
# E1 and E2 (v7.7)


def test_every_member_passing_emits_unanimously() -> None:
    n = 400
    bp = np.full(n, 0.95)
    frames = [member_rows(name, bp, pattern(n, 0.99)) for name in ("m0", "m1", "m2")]
    result = decide(frames, ["m0", "m1", "m2"])
    record = row(result)
    assert record["status"] == res.STATUS_EMITTED
    assert record["ensemble_rule"] == res.RULE_UNANIMOUS
    assert bool(record["member_emitted"])
    assert record["member_statuses"].count("=emitted") == 3


def test_unanimous_members_but_a_failing_pooled_set_are_not_emitted() -> None:
    # Member B is clean on both halves; member C's fit half has 20% errors
    # below 0.80 but its check half 80%: each member passes at its own t*
    # (B 0.70, C 0.80), while the pooled fit reaches the target at 0.70, where
    # the pooled check half fails (E1).
    n = 2000
    rng_bp = np.round(0.70 + 0.2999 * (np.arange(n) // 2 % 1000) / 1000, 4)
    half = np.arange(n) % 2
    band = rng_bp < 0.80
    b_correct = pattern(n, 0.905)
    b_correct[half == 0] = True
    c_correct = np.ones(n, dtype=bool)
    fit_band = band & (half == 0)
    check_band = band & (half == 1)
    c_correct[fit_band] = (np.arange(n)[fit_band] // 2) % 5 != 0
    c_correct[check_band] = (np.arange(n)[check_band] // 2) % 5 == 0
    frames = [
        member_rows("B", rng_bp, b_correct),
        member_rows("C", rng_bp, c_correct),
    ]
    result = decide(frames, ["B", "C"])
    record = row(result)
    members = result.member_decisions
    own = members[(members["regime"] == "trust")].set_index("member")["status"]
    assert set(own) == {res.STATUS_EMITTED}
    assert bool(record["member_emitted"])
    assert record["status"] == res.STATUS_NOT_RESOLVABLE
    assert str(record["reason"]).startswith(res.REASON_ENSEMBLE_PREFIX)
    assert record["reason"] != res.REASON_ENSEMBLE_SPREAD
    assert record["precision"] < 0.9


def test_a_failing_member_outside_the_spread_limit_blocks_emission() -> None:
    n = 400
    bp = np.full(n, 0.95)
    frames = [
        member_rows("m0", bp, pattern(n, 0.99)),
        member_rows("m1", bp, pattern(n, 0.99)),
        member_rows("m2", bp, pattern(n, 0.88)),
    ]
    result = decide(frames, ["m0", "m1", "m2"])
    record = row(result)
    # E1 passes on the pooled set (0.953), the spread 0.11 exceeds 3.5 SE.
    assert record["precision"] == pytest.approx((0.99 + 0.99 + 0.88) / 3, abs=2e-3)
    assert record["status"] == res.STATUS_NOT_RESOLVABLE
    assert record["reason"] == res.REASON_ENSEMBLE_SPREAD
    assert record["member_spread"] > record["spread_limit"]
    assert not bool(record["spread_ok"])


def test_a_failing_member_within_the_spread_limit_is_emitted_by_the_spread() -> None:
    n = 4000
    bp = np.full(n, 0.95)
    frames = [
        member_rows("m0", bp, pattern(n, 0.905)),
        member_rows("m1", bp, pattern(n, 0.905)),
        member_rows("m2", bp, pattern(n, 0.895)),
    ]
    result = decide(frames, ["m0", "m1", "m2"])
    members = result.member_decisions
    own = members[members["regime"] == "trust"].set_index("member")["status"]
    assert own["m2"] == res.STATUS_NOT_RESOLVABLE
    record = row(result)
    assert record["status"] == res.STATUS_EMITTED
    assert record["ensemble_rule"] == res.RULE_SPREAD
    assert record["member_spread"] == pytest.approx(0.01, abs=1e-3)
    assert record["member_spread"] <= record["spread_limit"]


def test_members_need_ten_calls_each_for_the_spread() -> None:
    spread, limit, within = res.member_spread(
        [0.95, 0.95, 0.95], [60.0, 60.0, 9.0], res.EnsembleSettings()
    )
    assert spread == 0.0 and not within
    assert limit == pytest.approx(3.5 * math.sqrt(0.95 * 0.05 / 43.0))
    spread, limit, within = res.member_spread(
        [0.95, 0.96], [100.0, 100.0], res.EnsembleSettings()
    )
    assert within
    assert limit == pytest.approx(
        max(0.03, 3.5 * math.sqrt(0.955 * 0.045 / 100)), abs=1e-12
    )
    assert not res.member_spread([0.95, math.nan], [100, 0], res.EnsembleSettings())[2]


def test_n_counts_distinct_test_cells_not_member_rows() -> None:
    # 40 cells x 3 members, all confident at the default (validated regime,
    # both halves): 120 confident rows, but 40 distinct test cells <
    # min_confident_n (50).
    n = 40
    bp = np.full(n, 0.95)
    frames = [member_rows(name, bp, np.ones(n, bool)) for name in ("a", "b", "c")]
    result = decide(
        frames, ["a", "b", "c"], settings=res.RuleSettings(min_cells_per_bin=30)
    )
    record = row(result, regime="validated")
    assert record["n_confident"] == 40
    assert record["n_rows"] == 120
    assert record["status"] == res.STATUS_NOT_RESOLVABLE
    assert record["reason"] == res.REASON_INSUFFICIENT_CALLS
    stats = res.distinct_cell_stats(
        np.full(4, 0.9),
        np.ones(4),
        np.ones(4),
        np.array([1, 1, 2, 2]),
        0.7,
    )
    assert stats.n_confident == 2 and stats.n_effective == pytest.approx(2.0)


def test_deep_bins_take_the_ensemble_pool() -> None:
    # 300 cells at 100 and 20 at 250 (each once, at its deepest bin): 250 is
    # pooled into the ">= 100" set and extrapolated.
    frames = []
    for name in ("a", "b"):
        frames.append(member_rows(name, np.full(300, 0.95), pattern(300, 0.99)))
        frames.append(
            member_rows(name, np.full(20, 0.95), np.ones(20, bool), depth=250)
        )
    result = decide(frames, ["a", "b"], depths=(100, 250))
    deep = row(result, depth=250)
    assert bool(deep["pooled"]) and bool(deep["extrapolated"])
    assert deep["pool_min_depth"] == 100
    assert deep["status"] == res.STATUS_EMITTED


# --------------------------------------------------------------------------
# Saturated-bp rule (v7.8)


def saturated_bin(n: int, share_one: float, *, errors_at_one: int = 0) -> pd.DataFrame:
    """A set with ``share_one`` of its calls at bp = 1; errors at .96 / .99."""
    rng = np.random.default_rng(5)
    at_one = np.zeros(n, dtype=bool)
    at_one[rng.permutation(n)[: round(share_one * n)]] = True
    bp = np.where(at_one, 1.0, np.where(np.arange(n) % 2 == 0, 0.96, 0.99))
    correct = at_one.copy()
    ones = np.flatnonzero(at_one)
    correct[ones[:errors_at_one]] = False
    # Half of the calls below 1 are correct, so g stays below the target.
    below = np.flatnonzero(~at_one)
    correct[below[::2]] = True
    return pd.DataFrame(
        {
            "bp": bp.astype(np.float32).astype(np.float64),
            "correct": correct,
            "half": rng.integers(0, 2, n),
        }
    )


def as_cells(frame: pd.DataFrame, member: str = "m0") -> pd.DataFrame:
    return member_rows(
        member,
        frame["bp"].to_numpy(),
        frame["correct"].to_numpy(),
        half=frame["half"].to_numpy(),
        depth=2000,
    )


def test_a_saturated_set_without_a_threshold_is_judged_at_the_cap() -> None:
    cells = as_cells(saturated_bin(800, 0.92))
    decided = res.decide(
        cells, [BROAD], [2000], res.RuleSettings(), saturated_bp_share=0.90
    ).set_index("regime")
    record = decided.loc["provisional"]
    assert pd.isna(record["t_star"])
    assert bool(record["saturated_bp"])
    assert record["threshold_source"] == res.THRESHOLD_SOURCE_SATURATED
    assert record["threshold"] == pytest.approx(0.99)
    assert record["status"] == res.STATUS_EMITTED
    # Without the rule (version 6) the same set has no local threshold.
    plain = res.decide(cells, [BROAD], [2000], res.RuleSettings()).set_index("regime")
    assert plain.loc["provisional", "reason"] == res.REASON_NO_LOCAL_THRESHOLD
    assert "saturated_bp" not in plain.columns
    assert list(plain.reset_index().columns) == list(res.DECISION_COLUMNS)


def test_a_set_at_most_90_percent_saturated_keeps_no_local_threshold() -> None:
    cells = as_cells(saturated_bin(800, 0.88))
    decided = res.decide(
        cells, [BROAD], [2000], res.RuleSettings(), saturated_bp_share=0.90
    ).set_index("regime")
    record = decided.loc["provisional"]
    assert not bool(record["saturated_bp"])
    assert record["reason"] == res.REASON_NO_LOCAL_THRESHOLD


def test_the_saturated_rule_applies_to_the_pooled_set() -> None:
    frames = [as_cells(saturated_bin(800, 0.92), name) for name in ("a", "b")]
    result = decide(frames, ["a", "b"], depths=(2000,))
    record = row(result, regime="provisional", depth=2000)
    assert bool(record["saturated_bp"])
    assert record["threshold_source"] == res.THRESHOLD_SOURCE_SATURATED
    assert record["status"] == res.STATUS_EMITTED
    assert res.saturated_bp_fraction(np.array([1.0, 0.9999995, 0.99])) == (
        pytest.approx(2 / 3)
    )


# --------------------------------------------------------------------------
# Monotone fill (v7.9)

GRID = (100, 250, 500, 1000)


def monotone_cells(
    cls: str,
    *,
    deep_depth: int,
    deep_n: int,
    deep_share: float,
    member: str,
) -> list[pd.DataFrame]:
    """300 precise cells at 100 and 250 each, one deeper bin as asked."""
    frames = [
        member_rows(member, np.full(300, 0.95), pattern(300, 0.99), cls=cls, depth=d)
        for d in (100, 250)
    ]
    frames.append(
        member_rows(
            member,
            np.full(deep_n, 0.95),
            pattern(deep_n, deep_share),
            cls=cls,
            depth=deep_depth,
        )
    )
    return frames


def test_a_deep_power_failure_is_filled() -> None:
    # 120 cells at 500 (60 check calls) at 0.925: the point precision passes,
    # the Wilson bound (0.85) does not: filled, extrapolated, own threshold.
    frames = [
        frame
        for member in ("a", "b")
        for frame in monotone_cells(
            "N", deep_depth=500, deep_n=120, deep_share=0.925, member=member
        )
    ]
    result = decide(frames, ["a", "b"], depths=GRID, neuronal={"N": True})
    record = row(result, cls="N", depth=500)
    assert record["ensemble_status"] == res.STATUS_NOT_RESOLVABLE
    assert record["ensemble_reason"] == "ensemble_wilson_bound_below_target"
    assert record["status"] == res.STATUS_EMITTED
    assert bool(record["monotone_filled"]) and bool(record["extrapolated"])
    assert record["fill_source"] == res.FILL_OWN
    assert record["fill_precision"] >= 0.9
    unfilled = result.unfilled()
    again = unfilled[
        (unfilled["regime"] == "trust")
        & (unfilled["class"] == "N")
        & (unfilled["depth"] == 500)
    ].iloc[0]
    assert again["status"] == res.STATUS_NOT_RESOLVABLE


def test_a_deep_point_precision_failure_is_not_filled() -> None:
    frames = [
        frame
        for member in ("a", "b")
        for frame in monotone_cells(
            "N", deep_depth=500, deep_n=120, deep_share=0.85, member=member
        )
    ]
    result = decide(frames, ["a", "b"], depths=GRID, neuronal={"N": True})
    record = row(result, cls="N", depth=500)
    assert record["status"] == res.STATUS_NOT_RESOLVABLE
    assert not bool(record["monotone_filled"])


def test_non_neuronal_bins_at_1000_counts_or_deeper_are_never_filled() -> None:
    frames = [
        frame
        for member in ("a", "b")
        for frame in monotone_cells(
            "G", deep_depth=1000, deep_n=120, deep_share=0.925, member=member
        )
    ]
    result = decide(frames, ["a", "b"], depths=GRID, neuronal={"G": False})
    record = row(result, cls="G", depth=1000)
    assert record["status"] == res.STATUS_NOT_RESOLVABLE
    assert not bool(record["monotone_filled"])
    assert bool(record["nonneuronal_high_depth"])
    assert not bool(row(result, cls="G", depth=500)["nonneuronal_high_depth"])
    # The same bin of a neuronal class is filled; an unknown lineage is not.
    neuron = decide(frames, ["a", "b"], depths=GRID, neuronal={"G": True})
    assert bool(row(neuron, cls="G", depth=1000)["monotone_filled"])
    unknown = decide(frames, ["a", "b"], depths=GRID, neuronal=None)
    assert not bool(row(unknown, cls="G", depth=1000)["monotone_filled"])


def test_the_monotone_fill_can_be_turned_off() -> None:
    frames = [
        frame
        for member in ("a", "b")
        for frame in monotone_cells(
            "N", deep_depth=500, deep_n=120, deep_share=0.925, member=member
        )
    ]
    result = decide(
        frames,
        ["a", "b"],
        depths=GRID,
        neuronal={"N": True},
        ensemble=res.EnsembleSettings(monotone_depth=False),
    )
    assert row(result, cls="N", depth=500)["status"] == res.STATUS_NOT_RESOLVABLE


def test_filled_bins_take_no_part_in_the_trust_tests_or_floors() -> None:
    # The leaf level of class N is emitted at 100 and filled at 250: the
    # trust test's leaf share at 250 counts only the unfilled decisions.
    frames = []
    for member in ("a", "b"):
        frames.append(
            member_rows(member, np.full(300, 0.95), pattern(300, 0.99), level="leaf")
        )
        frames.append(
            member_rows(
                member,
                np.full(120, 0.95),
                pattern(120, 0.925),
                level="leaf",
                depth=250,
            )
        )
        frames.append(member_rows(member, np.full(300, 0.95), pattern(300, 0.99)))
    result = decide(
        frames,
        ["a", "b"],
        levels=(BROAD, LEAF),
        depths=(100, 250),
        neuronal={"X": True},
    )
    assert bool(row(result, level="leaf", depth=250)["monotone_filled"])
    settings = res.RuleSettings()
    trust_filled = res.trust_constraint(result.decisions, [BROAD, LEAF], settings)
    trust, floors = res.trust_and_floors(
        result, [BROAD, LEAF], settings, floor_table=None, species="human"
    )
    assert trust_filled.leaf_share_by_depth[250] == 1.0
    assert trust.leaf_share_by_depth[250] == 0.0
    expected = res.trust_constraint(result.unfilled(), [BROAD, LEAF], settings)
    assert trust.to_json() == expected.to_json()
    leaf = floors[(floors["regime"] == "provisional") & (floors["level"] == "leaf")]
    assert leaf["simulated_floor"].tolist() == [100]


# --------------------------------------------------------------------------
# The whole version-7 self-map on the synthetic reference


@pytest.fixture(scope="module")
def v7_run() -> res.ResolvabilityResultV7:
    members = res.ensemble_members(
        AnnotationResolvabilityConfig(), species="human", chemistry="unknown"
    )
    return res.run_resolvability_v7(
        make_test_cells(120),
        specs=synthetic_specs(),
        depths=(10, 30, 100),
        members=members,
        map_fn=bootstrap_mapper,
        settings=res.RuleSettings(),
        ensemble=res.EnsembleSettings(),
        species="human",
        neuronal={"A": True, "B": False},
    )


def test_version_7_self_map_records_members_and_the_pooled_table(
    v7_run: res.ResolvabilityResultV7,
) -> None:
    summary = v7_run.summary
    assert summary["resolvability_version"] == 7
    assert summary["decision_recipe"] == res.ENSEMBLE_RECIPE
    assert summary["emission_members"] == [
        "R1_contam_HO@0",
        "R1_contam_HO@1",
        "R1_contam_HO@2",
    ]
    ensemble = summary["ensemble"]["regimes"]["provisional"]
    assert {"member_spread", "member_status_agreement", "member_emitted"} <= set(
        ensemble
    )
    assert set(ensemble["member_emitted"]) == set(summary["emission_members"])
    kinds = set(v7_run.table["kind"].astype(str))
    assert {"member_decision", "decision", "bin", "curve", "gene_efficiency"} <= kinds
    members = set(v7_run.table["member"].dropna().astype(str))
    assert members == {*summary["emission_members"], "clean@0"}
    decisions = v7_run.table[v7_run.table["kind"] == "decision"]
    assert set(decisions["recipe"].astype(str)) == {res.ENSEMBLE_RECIPE}
    assert set(v7_run.cells[res.MEMBER_COLUMN]) == members
    # Floors and trust come from the decisions before the fill.
    trust, floors = res.trust_and_floors(
        res.EnsembleDecisions(
            decisions=v7_run.decisions,
            member_decisions=v7_run.member_decisions,
            members=summary["emission_members"],
        ),
        [spec.meta for spec in synthetic_specs()],
        res.RuleSettings(),
        floor_table=None,
        species="human",
    )
    assert trust.to_json() == v7_run.trust.to_json()


def test_version_7_tables_round_trip_and_re_derive(
    v7_run: res.ResolvabilityResultV7, tmp_path: Path
) -> None:
    files = v7_run.write(tmp_path)
    assert files["class_depth"] == res.CLASS_DEPTH_FILE
    tables = res.load_resolvability(tmp_path, allow_version_7=True)
    assert tables is not None and tables.version == 7
    again = tables.decisions()
    keys = ["regime", "level", "class", "depth"]
    first = v7_run.decisions.sort_values(keys).reset_index(drop=True)
    second = again.sort_values(keys).reset_index(drop=True)
    assert first["status"].tolist() == second["status"].tolist()
    np.testing.assert_allclose(
        first["threshold"].astype(float).fillna(-1),
        second["threshold"].astype(float).fillna(-1),
    )
    class_depth = pd.read_parquet(tmp_path / res.CLASS_DEPTH_FILE)
    assert list(class_depth.columns) == list(res.CLASS_DEPTH_COLUMNS)
    emitted = v7_run.summary["emitted"]["provisional"]
    for level, classes in emitted.items():
        for cls, depths in classes.items():
            rows = class_depth[
                (class_depth["regime"] == "provisional")
                & (class_depth["level"] == level)
                & (class_depth["class"] == cls)
                & (class_depth["status"] == res.STATUS_EMITTED)
            ]
            assert sorted(rows["depth"].astype(int)) == depths


def test_load_resolvability_refuses_version_7_by_default(
    v7_run: res.ResolvabilityResultV7, tmp_path: Path
) -> None:
    v7_run.write(tmp_path)
    with pytest.raises(res.ResolvabilityError, match="version-7"):
        res.load_resolvability(tmp_path)
    assert res.load_resolvability(tmp_path, allow_version_7=True) is not None


def test_a_version_6_table_refuses_ensemble_decisions(tmp_path: Path) -> None:
    run = res.run_resolvability(
        make_test_cells(40),
        specs=synthetic_specs(),
        depths=(10, 30),
        recipes=res.simulation_recipes(AnnotationResolvabilityConfig()),
        map_fn=bootstrap_mapper,
        settings=res.RuleSettings(),
        species="human",
    )
    run.write(tmp_path)
    tables = res.load_resolvability(tmp_path)
    assert tables is not None and tables.version == res.RESOLVABILITY_VERSION
    with pytest.raises(res.ResolvabilityError, match="version-7 bundle"):
        tables.ensemble_decisions()


# --------------------------------------------------------------------------
# Class-depth table and profile prediction (v7.5)


class _Profile:
    """A two-class depth profile (``sim_inputs.DepthProfile`` interface)."""

    def __init__(self) -> None:
        self.by_class = {"X": np.array([150.0] * 3 + [600.0]), "Y": np.array([50.0])}
        self.label, self.asset, self.sha256 = "test", None, None
        self.n_cells = 5

    def source(self, cls: str) -> str:
        return cls if cls in self.by_class else "X"

    def values(self, source: str) -> np.ndarray:
        return self.by_class[source]


def test_class_depth_table_sums_profile_shares_over_emitted_bins() -> None:
    decisions = pd.DataFrame(
        {
            "regime": "provisional",
            "level": "broad",
            "class": "X",
            "depth": [100, 500],
            "status": [res.STATUS_EMITTED, res.STATUS_NOT_RESOLVABLE],
            "threshold": [0.8, np.nan],
            "threshold_source": [res.THRESHOLD_SOURCE_LOCAL, None],
            "ensemble_rule": [res.RULE_UNANIMOUS, None],
            "pooled": False,
            "extrapolated": False,
            "monotone_filled": False,
            "nonneuronal_high_depth": False,
            "neuronal": True,
            "n_test": 100,
            "n_confident": 60,
            "precision": [0.97, 0.8],
            "coverage": [0.6, 0.9],
            "fill_precision": np.nan,
            "fill_coverage": np.nan,
            "member_precision_min": 0.95,
            "member_precision_max": 0.98,
            "member_coverage_min": 0.5,
            "member_coverage_max": 0.7,
        }
    )
    table = res.class_depth_table(decisions, profile=_Profile(), grid=(100, 500))
    assert list(table["profile_share"]) == [0.75, 0.25]
    assert list(table["predicted_coverage_term"]) == pytest.approx([0.45, 0.0])
    prediction = res.profile_prediction(table, _Profile())
    assert prediction is not None
    level = prediction["regimes"]["provisional"]["broad"]
    assert level["classes"]["X"]["predicted_coverage"] == pytest.approx(0.45)
    assert level["classes"]["X"]["resolvable_share"] == pytest.approx(0.75)
    assert level["profile_share_covered"] == pytest.approx(0.8)


# --------------------------------------------------------------------------
# Gate P in every member and the diagnostic comparison


def test_a_class_is_validated_only_if_it_passes_in_every_member() -> None:
    verdicts: dict[str, dict[tuple[str, str], bool | None]] = {
        "a": {("class", "X"): True, ("class", "Y"): True, ("class", "Z"): True},
        "b": {("class", "X"): True, ("class", "Y"): False, ("class", "Z"): None},
    }
    combined = res.every_member_verdict(verdicts)
    assert combined[("class", "X")]["status"] == res.GATE_P_PASSED
    assert combined[("class", "Y")]["status"] == res.GATE_P_FAILED
    assert combined[("class", "Y")]["failed_members"] == ["b"]
    assert combined[("class", "Z")]["status"] == res.GATE_P_NOT_EVALUABLE


def test_gate_p_member_sets_score_each_member_at_the_frozen_thresholds(
    v7_run: res.ResolvabilityResultV7,
) -> None:
    members = v7_run.summary["emission_members"]
    sets = res.gate_p_member_sets(
        v7_run.cells, v7_run.decisions, members=members, min_confident_n=50
    )
    assert set(sets) == set(members)
    with pytest.raises(res.ResolvabilityError, match="more than once"):
        res.gate_p_tested_sets(
            v7_run.cells, v7_run.decisions, recipe=res.DECISION_RECIPE, seed=0
        )


def test_triple_churn_counts_lost_and_gained() -> None:
    first = {("class", "A", 10), ("class", "A", 30), ("subclass", "A", 30)}
    second = {("class", "A", 10), ("class", "B", 30)}
    churn = res.triple_churn(first, second)
    assert churn["lost"] == 2 and churn["gained"] == 1 and churn["union"] == 4
    assert churn["churn"] == pytest.approx(0.75)
    assert churn["per_level"]["subclass"]["lost"] == 1
    assert res.triple_churn(set(), set())["churn"] == 0.0


# --------------------------------------------------------------------------
# Test-set top-up (v7.6)


def top_up_candidates() -> pd.DataFrame:
    rows = []
    for index in range(40):
        rows.append(("A", "a1", f"k{index % 4}", "pool1", f"A_a1_{index:03d}"))
    for index in range(300):
        rows.append(("A", "a2", f"k{4 + index % 10}", "pool1", f"A_a2_{index:03d}"))
    for index in range(50):
        rows.append(("B", "b1", "k20", "pool1", f"B_b1_{index:03d}"))
    for index in range(30):
        rows.append(("A", "a1", "k30", "pool2", f"A_p2_{index:03d}"))
    frame = pd.DataFrame(rows, columns=["cls", "leaf", "cluster", "pool", "cell"])
    return frame.set_index("cell", drop=False)


def test_top_up_fills_thin_classes_by_leaf_within_the_pools() -> None:
    chosen, record = res.class_top_up(
        top_up_candidates(),
        class_column="cls",
        leaf_column="leaf",
        have={"A": 50, "B": 250, "C": 0},
        target=200,
        seed=0,
        pool_column="pool",
        pool_order=["pool1", "pool2"],
    )
    per_class = record["per_class"]
    assert set(per_class) == {"A"}  # B is not thin; C has no test cell
    assert per_class["A"]["after"] == 200 and not per_class["A"]["ran_out"]
    taken = per_class["A"]["pools"]["pool1"]
    # Water filling: a1 has 40, a2 300; 150 needed -> 40 + 110.
    assert taken["per_leaf"] == {"a1": 40, "a2": 110}
    assert per_class["A"]["pools"]["pool2"]["taken"] == 0
    assert len(chosen) == 150 and list(chosen) == sorted(chosen)


def test_top_up_respects_cluster_caps_and_pool_limits() -> None:
    candidates = top_up_candidates()
    caps = {f"k{index}": 3 for index in range(40)}
    chosen, record = res.class_top_up(
        candidates,
        class_column="cls",
        leaf_column="leaf",
        have={"A": 50},
        target=200,
        seed=0,
        pool_column="pool",
        pool_order=["pool1", "pool2"],
        cluster_column="cluster",
        cluster_cap=caps,
    )
    picked = candidates.loc[list(chosen)]
    assert picked.groupby("cluster").size().max() <= 3
    item = record["per_class"]["A"]
    assert item["ran_out"] and item["after"] < 200
    assert item["pools"]["pool2"]["taken"] == 3  # its one cluster, capped


def test_top_up_is_deterministic_and_independent_of_candidate_order() -> None:
    candidates = top_up_candidates()
    kwargs = {
        "class_column": "cls",
        "leaf_column": "leaf",
        "have": {"A": 150},
        "target": 200,
        "seed": 3,
    }
    first, _ = res.class_top_up(candidates, **kwargs)  # type: ignore[arg-type]
    shuffled = candidates.sample(frac=1.0, random_state=1)
    second, _ = res.class_top_up(shuffled, **kwargs)  # type: ignore[arg-type]
    assert list(first) == list(second)
    other, _ = res.class_top_up(candidates, **{**kwargs, "seed": 4})  # type: ignore[arg-type]
    assert set(first) != set(other)


def test_top_up_pools_can_be_restricted_to_classes() -> None:
    chosen, record = res.class_top_up(
        top_up_candidates(),
        class_column="cls",
        leaf_column="leaf",
        have={"A": 190},
        target=200,
        seed=0,
        pool_column="pool",
        pool_order=["pool2"],
        pool_classes={"pool2": ["B"]},
    )
    assert len(chosen) == 0
    assert record["per_class"]["A"]["pools"]["pool2"]["serves"] is False


def test_water_fill_splits_evenly_with_leftovers_in_name_order() -> None:
    assert res.water_fill({"b": 5, "a": 5, "c": 1}, 6) == {"a": 3, "b": 2, "c": 1}
    assert res.water_fill({"a": 2}, 5) == {"a": 2}
    assert res.water_fill({"a": 2, "b": 2}, 0) == {"a": 0, "b": 0}


def test_a_judged_deep_bin_is_withdrawn_when_its_ensemble_pool_fails() -> None:
    # 100 and 250 precise; 20 cells at 500 all wrong: the ">= 250" pool fails,
    # so 250 (judged on its own, emitted) is withdrawn and 500 not emitted.
    frames = []
    for member in ("a", "b"):
        frames.append(member_rows(member, np.full(300, 0.95), pattern(300, 0.99)))
        frames.append(
            member_rows(member, np.full(120, 0.95), np.ones(120, bool), depth=250)
        )
        frames.append(
            member_rows(member, np.full(20, 0.95), np.zeros(20, bool), depth=500)
        )
    result = decide(
        frames,
        ["a", "b"],
        depths=(100, 250, 500),
        ensemble=res.EnsembleSettings(monotone_depth=False),
    )
    judged = row(result, depth=250)
    assert judged["own_status"] == res.STATUS_EMITTED
    assert judged["status"] == res.STATUS_NOT_RESOLVABLE
    assert str(judged["reason"]).startswith(res.POOL_REASON_PREFIX)
    deep = row(result, depth=500)
    assert bool(deep["pooled"]) and deep["status"] == res.STATUS_NOT_RESOLVABLE
    assert row(result, depth=100)["status"] == res.STATUS_EMITTED


def test_the_saturated_rule_applies_to_each_member_too() -> None:
    frames = [as_cells(saturated_bin(800, 0.92), name) for name in ("a", "b")]
    result = decide(frames, ["a", "b"], depths=(2000,))
    members = result.member_decisions
    provisional = members[members["regime"] == "provisional"]
    assert provisional["saturated_bp"].astype(bool).all()
    assert set(provisional["status"]) == {res.STATUS_EMITTED}
    assert row(result, regime="provisional", depth=2000)["ensemble_rule"] == (
        res.RULE_UNANIMOUS
    )


def test_bins_shallower_than_the_shallowest_emitted_one_are_never_filled() -> None:
    # 100 fails only the Wilson bound (120 cells at 0.925), 250 and 500 are
    # emitted: the fill only looks deeper than 250.
    frames = []
    for member in ("a", "b"):
        frames.append(member_rows(member, np.full(120, 0.95), pattern(120, 0.925)))
        for depth in (250, 500):
            frames.append(
                member_rows(member, np.full(300, 0.95), pattern(300, 0.99), depth=depth)
            )
    result = decide(frames, ["a", "b"], depths=(100, 250, 500), neuronal={"X": True})
    shallow = row(result, depth=100)
    assert shallow["status"] == res.STATUS_NOT_RESOLVABLE
    assert not bool(shallow["monotone_filled"])
    assert row(result, depth=250)["status"] == res.STATUS_EMITTED


def test_water_fill_never_takes_more_than_a_stratum_holds() -> None:
    take = res.water_fill({"a": 1, "b": 5, "c": 5}, 6)
    assert take == {"a": 1, "b": 3, "c": 2}


def test_a_deep_coverage_failure_is_not_filled() -> None:
    # 500: 800 cells, 15% at bp 0.95 (all correct), the rest below the
    # default: judged on its own, it fails only the coverage; never filled.
    n = 800
    confident = (np.arange(n) // 2) % 20 < 3
    frames = []
    for member in ("a", "b"):
        for depth in (100, 250):
            frames.append(
                member_rows(member, np.full(300, 0.95), pattern(300, 0.99), depth=depth)
            )
        frames.append(
            member_rows(
                member,
                np.where(confident, 0.95, 0.5),
                np.ones(n, bool),
                depth=500,
            )
        )
    result = decide(frames, ["a", "b"], depths=(100, 250, 500), neuronal={"X": True})
    deep = row(result, depth=500)
    assert deep["ensemble_reason"] == "ensemble_coverage_below_minimum"
    assert deep["status"] == res.STATUS_NOT_RESOLVABLE
    assert not bool(deep["monotone_filled"])


# --------------------------------------------------------------------------
# Simulation-input assets never raise the trust constraint (pre-registration
# §14 (v); OD-E1 as amended: inputs, never trust evidence)


def _member(name: str, seed: int, *, asset: bool) -> res.EnsembleMember:
    recipe = res.SimulationRecipe(
        name=name,
        version=1,
        gene_efficiency_sigma=None if asset else 0.8,
        spill_fraction=0.25,
        seed=seed,
        efficiency_source="measured" if asset else "lognormal",
        efficiency_table="efficiency__test" if asset else None,
        efficiency_table_sha256="0" * 64 if asset else None,
    )
    return res.EnsembleMember(recipe=recipe, role="emission")


def _trust_cells(leaf_share: dict[str, float], n: int = 3000) -> pd.DataFrame:
    """Rows of each member: the broad level passes, the leaf level at ``share``."""
    frames = []
    for member, share in leaf_share.items():
        frames.append(member_rows(member, np.full(n, 0.95), pattern(n, 0.99)))
        frames.append(
            member_rows(member, np.full(n, 0.95), pattern(n, share), level="leaf")
        )
    return pd.concat(frames, ignore_index=True)


def _severity(state: str | None) -> int:
    return {"refused": 0, "broad_only": 1, None: 2}[state]


TRUST_LEVELS: tuple[res.LevelMeta, ...] = (BROAD, LEAF)
TRUST_DEPTHS: tuple[int, ...] = (100,)
TRUST_NEURONAL: dict[str, bool | None] = {"X": True}


def _guarded_trust(
    cells: pd.DataFrame, chosen: Sequence[res.EnsembleMember]
) -> tuple[res.TrustConstraint, res.TrustConstraint, dict[str, object]]:
    """Return (guarded, unguarded) trust of an ensemble of ``chosen`` members."""
    settings = res.RuleSettings()
    ensemble = res.EnsembleSettings()
    names = [member.name for member in chosen]
    rows = cells[cells[res.MEMBER_COLUMN].isin(names)]
    decided = res.ensemble_decide(
        rows,
        TRUST_LEVELS,
        TRUST_DEPTHS,
        settings,
        ensemble,
        members=names,
        neuronal=TRUST_NEURONAL,
    )
    full = res.trust_constraint(decided.unfilled(), TRUST_LEVELS, settings)
    guarded, record = res.asset_free_trust(
        rows,
        TRUST_LEVELS,
        TRUST_DEPTHS,
        settings,
        ensemble,
        members=chosen,
        neuronal=TRUST_NEURONAL,
        trust=full,
    )
    return guarded, full, record


def test_simulation_input_assets_never_raise_the_trust_constraint() -> None:
    free = [_member("R1_contam_HO", seed, asset=False) for seed in (0, 1)]
    assets = [_member("R3_measured_HO", seed, asset=True) for seed in (0, 1)]
    raised_without_guard = 0
    # The R1 draws miss the leaf target (0.895 < 0.90): broad_only without
    # assets. An asset member at 0.92 lifts the pooled set to 0.903 within the
    # spread limit, which would raise the constraint without the guard.
    for asset_share in (0.80, 0.895, 0.92, 1.0):
        shares = {member.name: 0.895 for member in free}
        shares.update({member.name: asset_share for member in assets})
        cells = _trust_cells(shares)
        base, _, _ = _guarded_trust(cells, free)
        assert base.state == "broad_only"
        states = set()
        # Adding one asset member, either, both, in any order.
        for subset in ([assets[0]], [assets[1]], assets, assets[::-1]):
            for order in (free + subset, subset + free):
                guarded, full, record = _guarded_trust(cells, order)
                assert _severity(guarded.state) <= _severity(base.state)
                assert record["applied"]
                assert record["asset_free_state"] == base.state
                if _severity(full.state) > _severity(base.state):
                    raised_without_guard += 1
                if len(subset) == 2:
                    states.add(guarded.state)
        # Permuting the assets (and the members) never changes the state.
        assert len(states) == 1
    # The guard is exercised: without it an asset would have raised it.
    assert raised_without_guard > 0


def test_the_trust_guard_is_a_no_op_without_asset_members() -> None:
    trust = res.TrustConstraint(
        state="broad_only",
        reasons=["r"],
        broad_emitted_bins=1,
        leaf_share_by_depth={100: 0.0},
        leaf_classes=["X"],
    )
    free = [_member("R1_contam_HO", seed, asset=False) for seed in (0, 1, 2)]
    guarded, record = res.asset_free_trust(
        pd.DataFrame(),
        [BROAD, LEAF],
        (100,),
        res.RuleSettings(),
        res.EnsembleSettings(),
        members=free,
        neuronal={},
        trust=trust,
    )
    assert guarded is trust and not record["applied"]
    assert record["asset_members"] == []
    assert res.lower_trust_constraint(trust, trust) is trust


def test_version_7_summary_records_the_trust_guard(
    v7_run: res.ResolvabilityResultV7,
) -> None:
    guard = v7_run.summary["trust_asset_guard"]
    assert guard["asset_members"] == [] and guard["applied"] is False
    assert guard["state"] == v7_run.trust.state
