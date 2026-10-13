"""Tests for raw thresholds, floors, emission and the dataset gate (§5.4, §8.3)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import thresholds as th
from merxen.annotation.config import AnnotationGate, AnnotationThresholds
from merxen.annotation.diagnostics import level_rank
from merxen.annotation.resolvability import LevelMeta, RuleSettings

from .conftest import HUMAN_TABLE_CLASSES

T = AnnotationThresholds()
HUMAN_GRID = (10, 15, 30, 60, 120, 250)
MakeDecisions = Callable[..., pd.DataFrame]


# --------------------------------------------------------------------------
# Raw thresholds


def test_raw_thresholds_are_the_pre_registered_v1_values() -> None:
    expected = {
        ("human", "lineage"): 0.73,
        ("human", "broad"): 0.73,
        ("human", "nt"): 0.73,
        ("human", "supercluster"): 0.69,
        ("human", "cluster"): 0.69,
        ("human", "seaad_broad"): 0.68,
        ("mouse", "broad"): 0.90,
        ("mouse", "class"): 0.90,
        ("mouse", "nt"): 0.90,
        ("mouse", "subclass"): 0.80,
        ("mouse", "supertype"): 0.80,
    }
    for (species, level), value in expected.items():
        got = th.raw_threshold(level, thresholds=T, species=species, n_cells=3)
        assert got.tolist() == [value] * 3, (species, level)
    sea = th.raw_threshold(
        "seaad_subclass", thresholds=T, species="human", counts=[10, 59, 60, 500]
    )
    assert sea.tolist() == [0.55, 0.55, 0.45, 0.45]


def test_raw_thresholds_refuse_unknown_levels_and_calibrated_mode() -> None:
    with pytest.raises(ValueError, match="not a mouse level"):
        th.raw_threshold("supercluster", thresholds=T, species="mouse", n_cells=1)
    with pytest.raises(ValueError, match="depend on the counts"):
        th.raw_threshold("seaad_subclass", thresholds=T, species="human", n_cells=1)
    calibrated = AnnotationThresholds(mode="simulation_calibrated")
    with pytest.raises(ValueError, match="v1.1"):
        th.raw_threshold("broad", thresholds=calibrated, species="human", n_cells=1)
    with pytest.raises(ValueError, match="no precision target"):
        th.level_target("unknown", T)
    assert th.level_target("supercluster", T) == 0.85
    assert th.level_target("class", T) == 0.90


def test_apply_raw_thresholds_tolerates_float32_storage() -> None:
    stored = np.array([0.69, 0.45, 0.73], dtype=np.float32)
    assert th.apply_raw_thresholds(stored, np.array([0.69, 0.45, 0.73])).all()
    assert not th.apply_raw_thresholds([0.6899], 0.69).any()
    assert not th.apply_raw_thresholds([np.nan, 0.9], [0.5, np.nan]).any()


def test_merged_thresholds_are_raise_only() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        default = rng.uniform(0.4, 0.8, size=40)
        local = rng.uniform(0.3, 0.99, size=40)
        local[rng.random(40) < 0.3] = np.nan
        recorded = rng.uniform(0.4, 0.8)
        merged = th.merge_thresholds(default, recorded, local, None)
        assert (merged >= default).all()
        assert (merged >= recorded).all()
        expected = np.fmax(np.maximum(default, recorded), local)
        assert merged == pytest.approx(expected)
        assert (
            merged[np.isnan(local)] == np.maximum(default, recorded)[np.isnan(local)]
        ).all()


# --------------------------------------------------------------------------
# Emission plans


def plan(
    decisions: pd.DataFrame | None,
    meta: list[LevelMeta],
    trust: Any = None,
    thresholds: AnnotationThresholds = T,
    **kwargs: Any,
) -> th.EmissionPlan:
    return th.EmissionPlan(
        species="human",
        thresholds=thresholds,
        decisions=decisions,
        levels=tuple(meta),
        grid=HUMAN_GRID,
        trust=trust,
        **kwargs,
    )


def test_validated_panels_keep_the_default_and_provisional_ones_raise(
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
    make_trust: Callable[..., Any],
) -> None:
    decisions = make_decisions(
        overrides={
            ("provisional", "broad", "Exc", 30): {"threshold": 0.88},
            # A local threshold below the default can never lower it.
            ("provisional", "broad", "Exc", 60): {"threshold": 0.60},
            ("validated", "broad", "Exc", 30): {"t_star": 0.95},
        }
    )
    counts = np.array([35.0, 70.0, 12.0])
    keys = np.array(["Exc", "Exc", "Exc"], dtype=object)
    validated = plan(decisions, human_level_meta, make_trust("validated_real"))
    emission = validated.level("broad", keys, counts)
    assert emission.regime == "validated"
    assert emission.threshold_source == "validated_default"
    assert emission.threshold.tolist() == [0.73, 0.73, 0.73]
    provisional = plan(decisions, human_level_meta, make_trust("provisional"))
    emission = provisional.level("broad", keys, counts)
    assert emission.regime == "provisional"
    assert emission.threshold_source == "resolvability_local"
    assert emission.threshold.tolist() == [0.88, 0.73, 0.73]
    assert emission.local_threshold.tolist() == [0.88, 0.60, 0.73]
    # Simulation-validated families keep the provisional rule (§8.2).
    simulation = plan(decisions, human_level_meta, make_trust("validated_simulation"))
    assert simulation.level("broad", keys, counts).threshold.tolist() == [
        0.88,
        0.73,
        0.73,
    ]
    # No trust decision: the fail-safe provisional regime.
    assert plan(decisions, human_level_meta).regime("broad") == "provisional"


def test_a_configured_default_below_the_bundle_default_is_raised(
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
    make_trust: Callable[..., Any],
) -> None:
    lowered = AnnotationThresholds(whb_broad=0.60)
    decisions = make_decisions()
    emission = plan(
        decisions, human_level_meta, make_trust("validated_real"), thresholds=lowered
    ).level("broad", np.array(["Exc"], dtype=object), np.array([40.0]))
    assert emission.threshold.tolist() == [0.73]
    raised = AnnotationThresholds(whb_broad=0.80)
    emission = plan(
        decisions, human_level_meta, make_trust("validated_real"), thresholds=raised
    ).level("broad", np.array(["Exc"], dtype=object), np.array([40.0]))
    assert emission.threshold.tolist() == [0.80]


def test_emission_follows_the_table_per_class_and_depth_bin(
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
    make_trust: Callable[..., Any],
) -> None:
    decisions = make_decisions(
        overrides={
            ("validated", "supercluster", "OPC", 30): {
                "status": "not_resolvable",
                "reason": "wilson_bound_below_target",
            },
            ("validated", "supercluster", "Astro", 250): {"extrapolated": True},
        }
    )
    emission = plan(decisions, human_level_meta, make_trust("validated_real")).level(
        "supercluster",
        np.array(["OPC", "OPC", "Astro", None, "Astro", "Unknown"], dtype=object),
        np.array([45.0, 70.0, 900.0, 100.0, 9.0, 100.0]),
    )
    assert emission.emitted.tolist() == [False, True, True, False, False, False]
    assert emission.extrapolated.tolist() == [False, False, True, False, False, False]
    assert emission.depth_bin[:3].tolist() == [30.0, 60.0, 250.0]
    assert np.isnan(emission.depth_bin[4])
    assert emission.reason[0] == "not_resolvable_for_class_and_depth"
    summary = emission.summary()
    assert summary["emitted_share"] == pytest.approx(2 / 6)
    assert summary["not_emitted_reasons"] == {"not_resolvable_for_class_and_depth": 4}


def test_seaad_subclass_emission_follows_the_leaf_with_sea_thresholds(
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
    make_trust: Callable[..., Any],
) -> None:
    decisions = make_decisions(
        overrides={
            ("provisional", "supercluster", "Oligo", 15): {"status": "not_resolvable"},
            ("provisional", "supercluster", "Oligo", 30): {"threshold": 0.95},
        }
    )
    emission = plan(decisions, human_level_meta, make_trust("provisional")).level(
        "seaad_subclass",
        np.array(["Oligo", "Oligo", "Oligo"], dtype=object),
        np.array([20.0, 40.0, 90.0]),
    )
    assert emission.emitted.tolist() == [False, True, True]
    # SEA-AD's subclass has no table of its own: its raw thresholds apply.
    assert emission.threshold.tolist() == [0.55, 0.55, 0.45]
    assert emission.threshold_source == "validated_default"


def test_without_a_resolvability_table_every_level_but_fine_ones_is_emitted(
    make_trust: Callable[..., Any],
) -> None:
    empty = th.EmissionPlan(
        species="human", thresholds=T, trust=make_trust("validated_real")
    )
    keys = np.array(["Exc", None], dtype=object)
    broad = empty.level("broad", keys, np.array([20.0, 20.0]))
    assert broad.emitted.all() and broad.regime is None
    assert broad.threshold.tolist() == [0.73, 0.73]
    cluster = empty.level("cluster", keys, np.array([20.0, 20.0]))
    assert not cluster.emitted.any()
    assert set(cluster.reason) == {th.REASON_RESOLVABILITY_NOT_RUN}
    assert empty.simulated_floors() == {}
    assert np.isnan(empty.depth_bins([100.0])).all()


def test_fine_levels_need_the_opt_in(
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
    make_trust: Callable[..., Any],
) -> None:
    decisions = make_decisions()
    keys = np.array(["Exc"], dtype=object)
    counts = np.array([300.0])
    off = plan(decisions, human_level_meta, make_trust("validated_real"))
    assert not off.level("cluster", keys, counts).emitted.any()
    enabled = AnnotationThresholds(allow_fine_levels=True)
    on = plan(
        decisions,
        human_level_meta,
        make_trust("validated_real"),
        thresholds=enabled,
        fine_seed_stability={"cluster": 0.01},
    )
    emission = on.level("cluster", keys, counts)
    assert emission.emitted.all()
    # Above validated_max_level (supercluster) the cluster is provisional.
    assert emission.regime == "provisional"
    unstable = plan(
        decisions,
        human_level_meta,
        make_trust("validated_real"),
        thresholds=enabled,
        fine_seed_stability={"cluster": 0.05},
    )
    assert not unstable.level("cluster", keys, counts).emitted.any()


# --------------------------------------------------------------------------
# Floors


def test_packaged_floors_load_with_their_digest(tmp_path: Path) -> None:
    table = th.load_floors("human")
    assert len(table.sha256) == 64
    assert table.panel_families == ("human_set_a",)
    assert table.packaged(
        "broad", "Inh", platform="xenium", panel_family="human_set_a"
    ) == (30, "real_e2")
    assert table.known("broad", "Inh") == 30
    assert table.known("broad", "Nope") is None
    mouse = th.load_floors("mouse")
    assert mouse.known("subclass", "30 Astro-Epen") == 50
    custom = tmp_path / "floors.csv"
    custom.write_text(
        "level,floor_class,platform,panel_family,min_counts,floor_source,"
        "inherited_from,note\nbroad,Exc,MERSCOPE,fam,12,derived,,\n"
    )
    loaded = th.load_floors("human", custom)
    assert loaded.packaged("broad", "Exc", platform="MERSCOPE", panel_family="fam") == (
        12,
        "derived",
    )
    duplicated = tmp_path / "dup.csv"
    duplicated.write_text(custom.read_text() + "broad,Exc,MERSCOPE,fam,15,derived,,\n")
    with pytest.raises(ValueError, match="duplicated"):
        th.load_floors("human", duplicated)


def floor_stats(rows: list[tuple[str, str, int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["class", "platform", "depth", "n", "precision"]
    ).assign(level="broad")


def test_derive_floors_uses_the_smallest_precise_depth_with_50_cells() -> None:
    stats = floor_stats(
        [
            # Precise at 10 but only 40 cells there: the floor is 15.
            ("Exc", "MERSCOPE", 10, 40, 0.99),
            ("Exc", "MERSCOPE", 15, 60, 0.95),
            ("Exc", "MERSCOPE", 30, 80, 0.97),
            # Enough cells, never precise: no floor.
            ("Immune", "MERSCOPE", 10, 200, 0.70),
            ("Immune", "MERSCOPE", 30, 150, 0.85),
        ]
    )
    floors = th.derive_floors(stats, targets=T, hard_floor=10).set_index(
        ["floor_class", "platform"]
    )
    assert floors.loc[("Exc", "MERSCOPE"), "min_counts"] == 15
    assert floors.loc[("Exc", "MERSCOPE"), "floor_source"] == "derived"
    assert pd.isna(floors.loc[("Immune", "MERSCOPE"), "min_counts"])
    assert floors.loc[("Immune", "MERSCOPE"), "floor_source"] == "not_reached"


def test_derive_floors_inherits_below_50_test_cells() -> None:
    # The packaged cases (§5.4): Xenium Oligo (n = 5) inherits MERSCOPE's
    # floor; MERSCOPE OtherNeuron (n = 15) inherits Xenium's 60.
    stats = floor_stats(
        [
            ("Oligo", "MERSCOPE", 10, 1998, 0.999),
            ("Oligo", "XENIUM", 10, 5, 0.80),
            ("Oligo", "XENIUM", 15, 5, 0.80),
            ("OtherNeuron", "MERSCOPE", 60, 15, 0.95),
            ("OtherNeuron", "XENIUM", 30, 100, 0.80),
            ("OtherNeuron", "XENIUM", 60, 100, 0.93),
            # Too few cells on every platform: the hard floor.
            ("Fibroblast", "MERSCOPE", 10, 3, 1.0),
            ("Fibroblast", "XENIUM", 10, 7, 1.0),
        ]
    )
    floors = th.derive_floors(stats, targets={"broad": 0.90}, hard_floor=10)
    table = floors.set_index(["floor_class", "platform"])
    assert table.loc[("Oligo", "XENIUM"), "min_counts"] == 10
    assert table.loc[("Oligo", "XENIUM"), "floor_source"] == "inherited"
    assert table.loc[("Oligo", "XENIUM"), "inherited_from"] == "MERSCOPE"
    assert table.loc[("OtherNeuron", "MERSCOPE"), "min_counts"] == 60
    assert table.loc[("OtherNeuron", "MERSCOPE"), "inherited_from"] == "XENIUM"
    assert table.loc[("OtherNeuron", "XENIUM"), "floor_source"] == "derived"
    for platform in ("MERSCOPE", "XENIUM"):
        assert table.loc[("Fibroblast", platform), "min_counts"] == 10
        assert table.loc[("Fibroblast", platform), "floor_source"] == "default_low_n"
    # The rederived packaged table agrees with floors_human.csv.
    packaged = th.load_floors("human").frame.set_index(
        ["level", "floor_class", "platform"]
    )
    assert packaged.loc[("broad", "Oligo", "XENIUM"), "inherited_from"] == "MERSCOPE"
    assert packaged.loc[("broad", "OtherNeuron", "MERSCOPE"), "min_counts"] == 60


def test_derive_floors_inherits_only_within_a_group_and_never_below_hard() -> None:
    stats = floor_stats(
        [
            ("Astro", "MERSCOPE", 10, 500, 0.99),
            ("Astro", "XENIUM", 10, 4, 0.99),
        ]
    )
    stats["panel_family"] = ["a", "b"]
    floors = th.derive_floors(
        stats, targets=T, hard_floor=15, group_columns=["panel_family"]
    )
    table = floors.set_index(["panel_family", "floor_class", "platform"])
    assert table.loc[("a", "Astro", "MERSCOPE"), "min_counts"] == 15
    assert table.loc[("b", "Astro", "XENIUM"), "floor_source"] == "default_low_n"
    with pytest.raises(ValueError, match="missing columns"):
        th.derive_floors(stats.drop(columns="n"), targets=T, hard_floor=10)


def floor_plan(
    platform: str,
    trust: Any,
    *,
    species: str = "human",
    simulated: dict[tuple[str, str], int | None] | None = None,
) -> th.FloorPlan:
    built = th.FloorPlan.build(
        species=species,  # type: ignore[arg-type]
        platform=platform,
        hard_floor=10,
        trust=trust,
    )
    if simulated is None:
        return built
    return th.FloorPlan(
        species=built.species,
        platform=built.platform,
        table=built.table,
        hard_floor=built.hard_floor,
        policies=built.policies,
        panel_family=built.panel_family,
        simulated=simulated,
    )


def test_validated_family_floors_are_per_class_platform_and_level(
    make_trust: Callable[..., Any],
) -> None:
    merscope = floor_plan("MERSCOPE", make_trust("validated_real"))
    xenium = floor_plan("XENIUM", make_trust("validated_real"))
    assert merscope.floor("broad", "Inh").min_counts == 10
    assert xenium.floor("broad", "Inh").min_counts == 30
    # NT uses the class's broad floor (§5.2 rule 3).
    assert xenium.floor("nt", "Inh").min_counts == 30
    assert merscope.floor("supercluster", "Inh").min_counts == 30
    assert xenium.floor("supercluster", "Vascular").min_counts == 30
    assert merscope.floor("supercluster", "COP").min_counts == 120
    # SEA-AD's subclass uses the supercluster floor of its broad class.
    assert merscope.floor("seaad_subclass", "OPC").min_counts == 120
    assert merscope.floor("lineage", "Exc").source == "hard_floor"
    assert merscope.floor("lineage", None).min_counts == 10
    assert math.isinf(merscope.floor("broad", None).min_counts)
    value = xenium.floor("broad", "Immune")
    assert (value.min_counts, value.source, value.warning) == (15, "real_e2", False)
    cells = xenium.per_cell("broad", np.array(["Inh", "Exc", None], dtype=object))
    assert cells[:2].tolist() == [30.0, 10.0] and math.isinf(cells[2])
    # Levels above validated_max_level follow the unknown-panel rule.
    assert merscope.policy("cluster") == "unknown_panel"
    assert merscope.warnings() == ["floors_unknown_panel:cluster"]
    with pytest.raises(ValueError, match="not a human level"):
        merscope.floor("subclass", "Exc")


def test_unknown_panels_take_the_max_over_known_and_simulated_floors(
    make_trust: Callable[..., Any],
) -> None:
    simulated: dict[tuple[str, str], int | None] = {
        ("broad", "Exc"): 60,
        ("broad", "Astro"): None,
        ("supercluster", "Oligo"): 15,
    }
    provisional = floor_plan("MERSCOPE", make_trust("provisional"), simulated=simulated)
    # Max over platforms (Xenium Inh 30) even on MERSCOPE.
    inh = provisional.floor("broad", "Inh")
    assert (inh.min_counts, inh.source, inh.warning) == (30, "unknown_panel", True)
    assert provisional.floor("broad", "Exc").min_counts == 60
    # Never emitted: the known floor (the emission gate says not_resolvable).
    assert provisional.floor("broad", "Astro").min_counts == 10
    assert provisional.floor("supercluster", "Oligo").min_counts == 15
    # SEA-AD's subclass reads the leaf's simulated floor.
    assert provisional.floor("seaad_subclass", "Oligo").min_counts == 15
    assert "floors_unknown_panel:broad" in provisional.warnings()
    simulation = floor_plan(
        "XENIUM", make_trust("validated_simulation"), simulated=simulated
    )
    value = simulation.floor("broad", "Exc")
    assert (value.min_counts, value.source, value.warning) == (
        60,
        "simulation_validated",
        False,
    )
    # A validated family on a platform without packaged rows (never listed):
    # the unknown-platform max rule, with a warning.
    other = floor_plan("MERSCOPE", make_trust("validated_real"))
    other = th.FloorPlan(
        species="human",
        platform="COSMX",
        table=other.table,
        hard_floor=10,
        policies=other.policies,
        panel_family="human_set_a",
    )
    value = other.floor("broad", "Immune")
    assert (value.min_counts, value.source, value.warning) == (
        15,
        "unknown_platform",
        True,
    )
    record = provisional.to_json()
    assert record["floors_sha256"] == provisional.table.sha256
    assert {item["level"] for item in record["floors"]} >= {"broad", "supercluster"}


def test_mouse_subclass_floor_is_at_least_60_outside_real_data_families(
    make_trust: Callable[..., Any],
) -> None:
    ag7 = floor_plan(
        "MERSCOPE", make_trust("validated_real", species="mouse"), species="mouse"
    )
    assert ag7.floor("subclass", "30 Astro-Epen").min_counts == 50
    assert ag7.floor("class", "30 Astro-Epen").min_counts == 20
    assert ag7.floor("broad", "30 Astro-Epen").min_counts == 20
    xenium = floor_plan(
        "XENIUM", make_trust("provisional", species="mouse"), species="mouse"
    )
    assert xenium.floor("subclass", "30 Astro-Epen").min_counts == 60
    assert xenium.floor("supertype", "30 Astro-Epen").min_counts == 60
    assert xenium.floor("class", "30 Astro-Epen").min_counts == 20


# --------------------------------------------------------------------------
# Dataset gate


def depth_histogram(frac_ge30: float, n: int = 1000) -> np.ndarray:
    deep = int(round(frac_ge30 * n))
    return np.concatenate([np.full(deep, 80.0), np.full(n - deep, 15.0)])


def confident_share(share: float, n: int = 1000) -> np.ndarray:
    return np.arange(n) < int(round(share * n))


@pytest.mark.parametrize(
    ("frac", "coverage", "segmented", "trust", "level", "warning"),
    [
        (0.50, 0.60, 2000, "validated_real", "full", False),
        # P1212_M: A .234 -> broad_only; no warning (0.44 of table cells).
        (0.234, 0.442, 2890, "validated_real", "broad_only", False),
        # P5011_M: A .127 -> broad_only with the segmented-object warning.
        (0.127, 0.40, 4300, "validated_real", "broad_only", True),
        (0.50, 0.20, 1000, "validated_real", "failed", False),
        (0.10, 0.20, 1000, "validated_real", "failed", False),
        (0.50, 0.60, 5000, "validated_real", "full", True),
        # A provisional panel warns but never lowers the level.
        (0.50, 0.60, 1000, "provisional", "full", True),
        (0.20, 0.60, 1000, "provisional", "broad_only", True),
        # Trust caps: broad-only and refused panels.
        (0.50, 0.60, 1000, "broad_only", "broad_only", False),
        (0.50, 0.60, 1000, "refused", "failed", False),
        (0.50, 0.60, 1000, "validated_simulation", "full", False),
    ],
)
def test_gate_verdicts_on_synthetic_depth_histograms(
    make_trust: Callable[..., Any],
    frac: float,
    coverage: float,
    segmented: int,
    trust: str,
    level: str,
    warning: bool,
) -> None:
    verdict = th.dataset_gate(
        depth_histogram(frac),
        confident_share(coverage),
        n_segmented=segmented,
        trust=make_trust(trust),
    )
    assert verdict.level == level
    assert verdict.warning is warning
    assert verdict.frac_ge30 == pytest.approx(frac, abs=1e-3)
    assert verdict.broad_coverage_table == pytest.approx(coverage, abs=1e-3)
    assert verdict.attempts_leaf is (level == "full")
    record = verdict.to_json()
    assert record["level"] == level and record["warning"] is warning


def test_gate_truth_table_level_and_warning_are_independent() -> None:
    gate = AnnotationGate()
    for frac, coverage, seg_share in itertools.product(
        (0.29, 0.30, 0.9), (0.24, 0.25, 0.9), (0.14, 0.15, 0.9)
    ):
        n = 1000
        counts = depth_histogram(frac, n)
        confident = confident_share(coverage, n)
        segmented = int(round(confident.sum() / seg_share))
        verdict = th.dataset_gate(counts, confident, n_segmented=segmented, gate=gate)
        expected = "full"
        if frac < 0.30:
            expected = "broad_only"
        if coverage < 0.25:
            expected = "failed"
        assert verdict.level == expected, (frac, coverage, seg_share)
        seg = confident.sum() / segmented
        assert verdict.warning is bool(seg < 0.15), (frac, coverage, seg_share)


def test_gate_reasons_name_the_trust_state(make_trust: Callable[..., Any]) -> None:
    counts = depth_histogram(0.5)
    confident = confident_share(0.6)
    refused = th.dataset_gate(counts, confident, trust=make_trust("refused"))
    assert refused.level == "failed"
    assert any(reason.startswith("panel_refused") for reason in refused.level_reasons)
    capped = th.dataset_gate(counts, confident, trust=make_trust("broad_only"))
    assert any(r.startswith("panel_broad_only") for r in capped.level_reasons)
    provisional = th.dataset_gate(counts, confident, trust=make_trust("provisional"))
    assert provisional.level == "full"
    assert provisional.warning_reasons[0].startswith("panel_provisional")
    # No object count: no segmented-coverage warning.
    bare = th.dataset_gate(counts, confident)
    assert math.isnan(bare.broad_coverage_segmented) and not bare.warning
    extra = th.dataset_gate(counts, confident, extra_warnings=["real_qc:x"])
    assert extra.warning and extra.warning_reasons == ("real_qc:x",)
    empty = th.dataset_gate(np.zeros(0), np.zeros(0, dtype=bool))
    assert empty.level == "failed" and empty.n_table == 0
    with pytest.raises(ValueError, match="align"):
        th.dataset_gate(counts, confident[:10])


def test_simulation_families_warn_only_outside_their_validated_region(
    make_trust: Callable[..., Any],
) -> None:
    trust = make_trust("validated_simulation")
    counts = depth_histogram(0.5)
    confident = confident_share(0.6)
    inside = th.dataset_gate(
        counts, confident, trust=trust, validated_share={"broad": 0.95, "nt": None}
    )
    assert inside.level == "full" and not inside.warning
    outside = th.dataset_gate(
        counts, confident, trust=trust, validated_share={"broad": 0.85}
    )
    assert outside.level == "full" and outside.warning
    assert outside.warning_reasons[0].startswith("unvalidated_share:broad")
    real = th.dataset_gate(
        counts,
        confident,
        trust=make_trust("validated_real"),
        validated_share={"broad": 0.0},
    )
    assert not real.warning


# --------------------------------------------------------------------------
# Simulation rows keep the provisional margins (plan §8.2, §14; M13)

PROMOTED_LEVELS: dict[str, tuple[str, ...]] = {
    "human": ("lineage", "broad", "nt", "supercluster", "seaad_subclass", "cluster"),
    "mouse": ("broad", "class", "nt", "subclass", "supertype"),
}
# A class each species' simulation family validates at broad (and, mouse,
# class and subclass): even a validated (level, class) keeps the margins.
PROMOTED_CLASS: dict[str, str] = {"human": "Exc", "mouse": "01 IT-ET Glut"}


@pytest.mark.parametrize(
    ("species", "level"),
    [
        (species, level)
        for species, levels in PROMOTED_LEVELS.items()
        for level in levels
    ],
)
def test_simulation_rows_keep_the_provisional_margins(
    promotion_trust: Callable[..., tuple[Any, Any]],
    make_trust: Callable[..., Any],
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
    species: str,
    level: str,
) -> None:
    """A gate-P row validates labels; it never removes a margin or a floor rule.

    For every level, a simulation-validated family decides in the
    provisional regime (targets + margins, local thresholds), with the same
    emission and floors as the provisional panel it was before promotion:
    the max-rule floors without their warning, and the mouse subclass floor
    of at least 60. A real-data family drops the margins up to its
    validated level.
    """
    from .test_consensus_mouse import MOUSE_CLASS_CALLS, MOUSE_GRID, MOUSE_TABLE_LEVELS

    provisional, simulation = promotion_trust("resolvable", species=species)
    real = make_trust("validated_real", species=species)
    assert (provisional.state, simulation.state) == ("provisional", "validated")
    assert simulation.validation_basis == "simulation"
    rank = level_rank(species, level)
    real_rank = level_rank(species, str(real.validated_max_level))
    assert rank is not None and real_rank is not None
    real_covers = rank <= real_rank
    # Regime and targets: base + margin (+0.10 below 60 counts), capped.
    settings = RuleSettings()
    base = th.level_target(level, T)
    grid = HUMAN_GRID if species == "human" else MOUSE_GRID
    for decision in (provisional, simulation):
        assert decision.emission_regime(level) == "provisional"
        assert decision.applies_provisional_margins(level)
        assert decision.threshold_source(level) == "resolvability_local"
        for depth in grid:
            target = settings.target(decision.emission_regime(level), base, depth)
            assert target == pytest.approx(
                T.provisional_target(base, below60=depth < 60)
            )
            assert target > base
    assert real.applies_provisional_margins(level) is not real_covers
    # Emission: the provisional rows of the table, raised thresholds and
    # unemitted bins included.
    cls = PROMOTED_CLASS[species]
    table_level = th.DERIVED_EMISSION_LEVELS[species].get(level, level)
    if species == "human":
        shallow, middle, counts = 30, 60, np.array([35.0, 70.0, 300.0])
        meta: Sequence[LevelMeta] = human_level_meta
        decisions = make_decisions(
            overrides={
                ("provisional", table_level, cls, shallow): {"threshold": 0.95},
                ("provisional", table_level, cls, middle): {"status": "not_resolvable"},
            }
        )
        classes: Sequence[str] = HUMAN_TABLE_CLASSES
    else:
        shallow, middle, counts = 50, 100, np.array([60.0, 150.0, 600.0])
        meta = [
            LevelMeta(name, "CLAS", role, default, target, floor)  # type: ignore[arg-type]
            for name, role, default, target, floor in MOUSE_TABLE_LEVELS
        ]
        classes = tuple(MOUSE_CLASS_CALLS)
        decisions = make_decisions(
            levels=MOUSE_TABLE_LEVELS,
            classes=classes,
            grid=MOUSE_GRID,
            overrides={
                ("provisional", table_level, cls, shallow): {"threshold": 0.95},
                ("provisional", table_level, cls, middle): {"status": "not_resolvable"},
            },
        )
    limits = AnnotationThresholds(allow_fine_levels=True)
    keys = np.array([cls] * 3, dtype=object)

    def emission_plan(trust: Any) -> th.EmissionPlan:
        return th.EmissionPlan(
            species=species,  # type: ignore[arg-type]
            thresholds=limits,
            decisions=decisions,
            levels=tuple(meta),
            grid=grid,
            trust=trust,
            fine_seed_stability={"cluster": 0.0, "supertype": 0.0},
        )

    before = emission_plan(provisional).level(level, keys, counts)
    after = emission_plan(simulation).level(level, keys, counts)
    assert (before.regime, after.regime) == ("provisional", "provisional")
    assert after.emitted.tolist() == before.emitted.tolist() == [True, False, True]
    assert after.threshold.tolist() == before.threshold.tolist()
    if level not in th.SECONDARY_LEVELS:
        assert after.threshold[0] == pytest.approx(0.95)
    if real_covers:
        validated = emission_plan(real).level(level, keys, counts)
        assert validated.regime == "validated"
        assert validated.emitted.tolist() == [True, True, True]
    # Floors: the max rule of the provisional panel, without its warning.
    floors = {
        name: th.FloorPlan.build(
            species=species,  # type: ignore[arg-type]
            platform="MERSCOPE",
            hard_floor=10,
            trust=trust,
            thresholds=limits,
            emission=emission_plan(trust),
        )
        for name, trust in (("before", provisional), ("after", simulation))
    }
    for floor_class in classes:
        first = floors["before"].floor(level, floor_class)
        second = floors["after"].floor(level, floor_class)
        assert first.min_counts == second.min_counts
        if th.FLOOR_LEVELS[species][level] is not None:
            assert (first.source, first.warning) == ("unknown_panel", True)
            assert (second.source, second.warning) == ("simulation_validated", False)
        if species == "mouse" and level in th.MOUSE_SUBCLASS_FLOOR_LEVELS:
            assert second.min_counts >= 60
    assert floors["after"].warnings() == []
    if species == "mouse" and level == "subclass":
        packaged = th.FloorPlan.build(
            species="mouse", platform="MERSCOPE", hard_floor=10, trust=real
        )
        assert packaged.floor(level, "30 Astro-Epen").min_counts == 50
        assert floors["after"].floor(level, "30 Astro-Epen").min_counts == 60
