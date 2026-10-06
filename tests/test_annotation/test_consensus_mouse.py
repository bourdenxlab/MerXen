"""Tests for the mouse v1 rules (``consensus.resolve_mouse``; plan §7.3; M6)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from merxen.annotation import consensus as cs
from merxen.annotation import thresholds as th
from merxen.annotation.config import AnnotationThresholds, MouseGateConfig
from merxen.annotation.mouse_gate import (
    MarkerReferee,
    MouseGateSignals,
    RegistrationSignal,
    evaluate_mouse_gate,
)
from merxen.annotation.schema import CellStatus, Columns

from .conftest import PROMOTION_CONSTRAINTS, assert_same_emission

MakeTrust = Callable[..., Any]


@dataclass(frozen=True)
class Cell:
    """One synthetic mouse cell."""

    cls: str | None = "01 IT-ET Glut"
    class_bp: float = 0.98
    corr: float = 0.6
    broad: str | None = "Neurons"
    broad_bp: float = 1.0
    nt: str | None = "Excitatory"
    nt_bp: float = 1.0
    subclass: str | None = "007 L2/3 IT CTX Glut"
    subclass_bp: float = 0.95
    counts: int = 300
    dropped: str | None = None


ASTRO = Cell(
    cls="30 Astro-Epen",
    broad="Astrocytes/Ependymal",
    nt=None,
    subclass="319 Astro-TE NN",
)


def calls_of(cells: Sequence[Cell], *, min_counts: int = 10) -> cs.MouseCalls:
    def col(name: str) -> list[Any]:
        return [getattr(cell, name) for cell in cells]

    counts = np.array(col("counts"))
    wmb = cs.WmbCalls(
        klass=cs.LevelCall.of(col("cls"), col("class_bp"), corr=col("corr")),
        broad=cs.LevelCall.of(col("broad"), col("broad_bp"), corr=col("corr")),
        nt=cs.LevelCall.of(col("nt"), col("nt_bp"), corr=col("corr")),
        subclass=cs.LevelCall.of(col("subclass"), col("subclass_bp")),
    )
    return cs.MouseCalls(
        total_counts=counts,
        in_table=counts >= min_counts,
        wmb=wmb,
        region_dropped=np.array(col("dropped"), dtype=object),
    )


def gate(**change: Any) -> Any:
    values: dict[str, Any] = {
        "registration": RegistrationSignal(density_ratio=2.9, shift_um=0.0),
        "referee": MarkerReferee(
            consistency=0.9, n_pseudo_confident=500, n_cells=600, markers={}
        ),
        "t2_share": 0.004,
        "spillover_rate": 0.08,
    }
    values.update(change)
    return evaluate_mouse_gate(MouseGateSignals(**values), MouseGateConfig())


def resolve(
    cells: Sequence[Cell],
    make_trust: MakeTrust,
    *,
    verdict: Any = None,
    class_min_corr: float | None = None,
    trust_state: str = "validated_real",
    thresholds: AnnotationThresholds | None = None,
) -> cs.MouseResolution:
    trust = make_trust(trust_state, species="mouse")
    limits = thresholds or AnnotationThresholds()
    emission = th.EmissionPlan(species="mouse", thresholds=limits, trust=trust)
    floors = th.FloorPlan.build(
        species="mouse",
        platform="MERSCOPE",
        hard_floor=10,
        trust=trust,
        thresholds=limits,
        emission=emission,
    )
    settings = cs.MouseResolveSettings(
        platform="MERSCOPE",
        min_counts=10,
        emission=emission,
        floors=floors,
        thresholds=limits,
        trust=trust,
        class_min_corr=class_min_corr,
        allow_fine_levels=limits.allow_fine_levels,
    )
    return cs.resolve_mouse(calls_of(cells), settings, verdict or gate())


def statuses(result: cs.MouseResolution, index: int = 0) -> dict[str, str]:
    return {level: str(values.status[index]) for level, values in result.levels.items()}


def test_confident_neuron_and_astrocyte_reach_subclass(make_trust: MakeTrust) -> None:
    result = resolve([Cell(), ASTRO], make_trust)
    assert statuses(result, 0) == {
        "broad": "confident",
        "class": "confident",
        "nt": "confident",
        "subclass": "confident",
    }
    assert statuses(result, 1)["nt"] == "not_applicable"
    assert statuses(result, 1)["subclass"] == "confident"
    assert result.final_level.tolist() == ["subclass", "subclass"]
    assert result.final_name.tolist() == ["007 L2/3 IT CTX Glut", "319 Astro-TE NN"]
    assert result.consensus_tier.tolist() == [1, 1]
    assert result.levels["nt"].name[1] is None


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (
            Cell(class_bp=0.89),
            {"class": "low_confidence", "subclass": "parent_unresolved"},
        ),
        (Cell(class_bp=0.90), {"class": "confident", "subclass": "confident"}),
        (Cell(counts=19), {"broad": "below_floor", "class": "parent_unresolved"}),
        (
            Cell(counts=20, subclass_bp=0.95),
            {"class": "confident", "subclass": "below_floor"},
        ),
        (Cell(counts=50), {"class": "confident", "subclass": "confident"}),
        (Cell(subclass_bp=0.79), {"class": "confident", "subclass": "low_confidence"}),
        (Cell(subclass_bp=0.80), {"class": "confident", "subclass": "confident"}),
        (Cell(broad_bp=0.8), {"broad": "low_confidence", "class": "parent_unresolved"}),
        (Cell(counts=5), {"broad": "low_counts", "class": "low_counts"}),
    ],
)
def test_mouse_thresholds_and_floors(
    make_trust: MakeTrust, cell: Cell, expected: dict[str, str]
) -> None:
    got = statuses(resolve([cell], make_trust))
    assert {level: got[level] for level in expected} == expected


def test_region_dropped_cells_need_a_confident_remap(make_trust: MakeTrust) -> None:
    cells = [
        Cell(cls="19 MB Glut", dropped="class"),  # confident re-map: kept
        Cell(cls="19 MB Glut", class_bp=0.7, dropped="class"),
        Cell(dropped="subclass", subclass_bp=0.5),
        Cell(dropped="subclass"),
        Cell(class_bp=0.7),  # not re-mapped: plain low confidence
    ]
    result = resolve(cells, make_trust)
    assert statuses(result, 0)["class"] == "confident"
    assert statuses(result, 1) == {
        "broad": "confident",
        "class": "implausible",
        "nt": "implausible",
        "subclass": "implausible",
    }
    assert statuses(result, 2)["class"] == "confident"
    assert statuses(result, 2)["subclass"] == "implausible"
    assert statuses(result, 3)["subclass"] == "confident"
    assert statuses(result, 4)["class"] == "low_confidence"
    flags = result.flags
    assert flags[Columns.FLAG_IMPLAUSIBLE].tolist() == [False, True, True, False, False]
    assert result.final_level.tolist()[:3] == ["subclass", "broad", "nt"]
    assert not flags[Columns.EXCLUDE_HARD].any()


def test_class_correlation_floor_is_a_low_confidence_check(
    make_trust: MakeTrust,
) -> None:
    cells = [Cell(corr=0.39), Cell(corr=0.40), Cell(corr=0.55)]
    without = resolve(cells, make_trust)
    assert [statuses(without, index)["class"] for index in range(3)] == [
        "confident"
    ] * 3
    floored = resolve(cells, make_trust, class_min_corr=0.40)
    assert [statuses(floored, index)["class"] for index in range(3)] == [
        "low_confidence",
        "confident",
        "confident",
    ]
    assert floored.summary()["class_min_corr"] == 0.40


def test_failed_gate_blocks_every_level_and_excludes_every_cell(
    make_trust: MakeTrust,
) -> None:
    failed = gate(registration=RegistrationSignal(density_ratio=1.2, shift_um=114.0))
    assert failed.level == "failed"
    result = resolve([Cell(), ASTRO, Cell(counts=5)], make_trust, verdict=failed)
    for index in (0, 1):
        assert set(statuses(result, index).values()) == {"not_attempted_gate"}
    assert statuses(result, 2)["class"] == "low_counts"
    assert result.flags[Columns.EXCLUDE_HARD].all()
    assert result.final_level.tolist() == ["none"] * 3
    assert result.summary()["gate"]["level"] == "failed"


def test_broad_only_trust_blocks_the_leaf(make_trust: MakeTrust) -> None:
    verdict = evaluate_mouse_gate(
        MouseGateSignals(registration=RegistrationSignal(2.9, 0.0)),
        MouseGateConfig(),
        trust=make_trust("broad_only", species="mouse"),
    )
    result = resolve([Cell()], make_trust, verdict=verdict, trust_state="broad_only")
    assert statuses(result)["class"] == "confident"
    assert statuses(result)["subclass"] == "not_attempted_gate"
    assert result.final_level.tolist() == ["nt"]


def test_no_call_and_refused_primary(make_trust: MakeTrust) -> None:
    result = resolve([Cell(cls=None, broad=None, nt=None, subclass=None)], make_trust)
    assert statuses(result)["broad"] == "low_confidence"
    assert result.consensus_tier.tolist() == [-1]
    refused = resolve([Cell()], make_trust, trust_state="refused")
    assert statuses(refused)["class"] == CellStatus.NOT_ATTEMPTED_GATE.value
    assert refused.consensus_tier.tolist() == [-1]


def test_supertype_is_report_only(make_trust: MakeTrust) -> None:
    limits = AnnotationThresholds(allow_fine_levels=True)
    cells = [Cell()]
    trust = make_trust("validated_real", species="mouse")
    emission = th.EmissionPlan(species="mouse", thresholds=limits, trust=trust)
    floors = th.FloorPlan.build(
        species="mouse", platform="MERSCOPE", hard_floor=10, trust=trust
    )
    calls = calls_of(cells)
    assert calls.wmb is not None
    calls = cs.MouseCalls(
        total_counts=calls.total_counts,
        in_table=calls.in_table,
        wmb=cs.WmbCalls(
            klass=calls.wmb.klass,
            broad=calls.wmb.broad,
            nt=calls.wmb.nt,
            subclass=calls.wmb.subclass,
            supertype=cs.LevelCall.of(["0023 L2/3 IT CTX Glut_1"], [0.9]),
        ),
    )
    result = cs.resolve_mouse(
        calls,
        cs.MouseResolveSettings(
            platform="MERSCOPE",
            min_counts=10,
            emission=emission,
            floors=floors,
            thresholds=limits,
            trust=trust,
            allow_fine_levels=True,
        ),
        gate(),
    )
    # Without resolvability tables a report-only level is never emitted
    # (fails safe); it never enters ct_final either way.
    assert statuses(result)["supertype"] == "not_resolvable"
    assert result.levels["supertype"].name[0] == "0023 L2/3 IT CTX Glut_1"
    assert result.final_level.tolist() == ["subclass"]


def test_chain_validated_share_reads_the_confident_chain_labels(
    make_trust: MakeTrust,
) -> None:
    """The input of the 10% rule (§8.2; M13 D15 (a)): the chain levels only.

    Per chain level (broad, class, nt, subclass) the share of confident
    labels that are ``ct_<L>_validated``; ``None`` without a confident
    label; the report-only supertype is never read.
    """
    limits = AnnotationThresholds(allow_fine_levels=True)
    result = resolve([Cell(), Cell(), ASTRO, ASTRO], make_trust, thresholds=limits)
    levels = result.levels
    assert "supertype" in levels
    for level in ("broad", "class", "subclass"):
        assert levels[level].confident.all(), level
    assert levels["nt"].confident.tolist() == [True, True, False, False]
    levels["class"].validated[:] = [True, False, True, False]
    levels["nt"].validated[:] = [False, True, False, False]
    # A confident, unvalidated supertype would read 0.0 if it were counted.
    levels["supertype"].status[:] = CellStatus.CONFIDENT.value
    levels["supertype"].validated[:] = False
    shares = cs.chain_validated_share(levels, cs.MOUSE_CHAIN)
    assert cs.MOUSE_CHAIN == ("broad", "class", "nt", "subclass")
    assert shares == {"broad": 1.0, "class": 0.5, "nt": 0.5, "subclass": 1.0}
    astro_only = resolve([ASTRO, ASTRO], make_trust)
    assert cs.chain_validated_share(astro_only.levels, cs.MOUSE_CHAIN)["nt"] is None


def test_mouse_calls_need_one_value_per_object() -> None:
    with pytest.raises(ValueError, match="one value per object"):
        cs.MouseCalls(
            total_counts=np.array([10, 20]),
            in_table=np.array([True]),
            wmb=None,
        )


def test_marker_unsupported_calls_are_not_resolvable(make_trust: MakeTrust) -> None:
    trust = make_trust("validated_real", species="mouse")
    limits = AnnotationThresholds()
    emission = th.EmissionPlan(species="mouse", thresholds=limits, trust=trust)
    floors = th.FloorPlan.build(
        species="mouse", platform="MERSCOPE", hard_floor=10, trust=trust
    )
    base = calls_of([Cell(), Cell(subclass="157 RN Spp1 Glut"), Cell()])
    calls = cs.MouseCalls(
        total_counts=base.total_counts,
        in_table=base.in_table,
        wmb=base.wmb,
        marker_unsupported={
            "subclass": np.array([False, True, False]),
            "class": np.array([False, False, True]),
        },
    )
    result = cs.resolve_mouse(
        calls,
        cs.MouseResolveSettings(
            platform="MERSCOPE",
            min_counts=10,
            emission=emission,
            floors=floors,
            thresholds=limits,
            trust=trust,
        ),
        gate(),
    )
    assert statuses(result, 0)["subclass"] == "confident"
    assert statuses(result, 1)["subclass"] == "not_resolvable"
    assert statuses(result, 1)["class"] == "confident"
    assert statuses(result, 2)["class"] == "not_resolvable"
    assert result.final_level.tolist() == ["subclass", "nt", "broad"]


def test_a_region_dropped_subclass_below_the_floor_is_implausible(
    make_trust: MakeTrust,
) -> None:
    """A region-dropped subclass call never shows as below_floor (§4.2)."""
    result = resolve([Cell(dropped="subclass", counts=30), Cell(counts=30)], make_trust)
    assert statuses(result, 0)["class"] == "confident"
    assert statuses(result, 0)["subclass"] == "implausible"
    assert statuses(result, 1)["subclass"] == "below_floor"
    assert result.flags[Columns.FLAG_IMPLAUSIBLE].tolist() == [True, False]


def test_a_cell_implausible_at_every_attempted_level_is_excluded(
    make_trust: MakeTrust,
) -> None:
    """exclude_hard: every attempted level implausible or not applicable (§4.3)."""
    lost = Cell(broad="Unassigned", nt=None, cls="24 MY Glut", class_bp=0.5)
    result = resolve(
        [
            Cell(
                broad="Unassigned",
                nt=None,
                cls="24 MY Glut",
                class_bp=0.5,
                dropped="class",
            ),
            lost,
            Cell(cls="19 MB Glut", class_bp=0.7, dropped="class"),
        ],
        make_trust,
    )
    assert statuses(result, 0) == {
        "broad": "implausible",
        "class": "implausible",
        "nt": "not_applicable",
        "subclass": "implausible",
    }
    assert statuses(result, 1)["class"] == "parent_unresolved"
    # Kept at broad: the neuron's broad call is confident.
    assert statuses(result, 2)["broad"] == "confident"
    assert result.flags[Columns.EXCLUDE_HARD].tolist() == [True, False, False]


# --------------------------------------------------------------------------
# Promotion by simulation never changes what is emitted (plan §8.2, §14; M13)

MOUSE_GRID: tuple[int, ...] = (10, 20, 50, 100, 250, 500, 1000, 2000)
# (level, role, default threshold, base target, floor level) as PREP's
# mouse level specs record them.
MOUSE_TABLE_LEVELS: tuple[tuple[str, str, float, float, str | None], ...] = (
    ("broad", "broad", 0.90, 0.90, "class"),
    ("class", "class", 0.90, 0.90, "class"),
    ("nt", "nt", 0.90, 0.90, "class"),
    ("subclass", "leaf", 0.80, 0.85, "subclass"),
    ("supertype", "fine", 0.80, 0.85, "subclass"),
)
MOUSE_CLASS_CALLS: dict[str, tuple[str, str | None, str, str]] = {
    # class: (broad, nt, subclass, supertype)
    "01 IT-ET Glut": ("Neurons", "Excitatory", "007 L2/3 IT CTX Glut", "0023 IT"),
    "19 MB Glut": ("Neurons", "Excitatory", "19 MB Glut 1", "19 MB Glut 1_1"),
    "24 MY Glut": ("Neurons", "Excitatory", "24 MY Glut 1", "24 MY Glut 1_1"),
    "30 Astro-Epen": ("Astrocytes/Ependymal", None, "319 Astro-TE NN", "1162 Astro"),
}


def promotion_decisions(make_decisions: Callable[..., Any]) -> Any:
    """A mouse decisions table whose provisional regime carries the margins."""
    overrides = {
        ("provisional", "class", "01 IT-ET Glut", 20): {"status": "not_resolvable"},
        ("provisional", "class", "30 Astro-Epen", 50): {"threshold": 0.95},
        ("provisional", "broad", "19 MB Glut", 10): {"status": "not_resolvable"},
        ("provisional", "nt", "24 MY Glut", 100): {"threshold": 0.97},
        ("provisional", "subclass", "01 IT-ET Glut", 100): {"status": "not_resolvable"},
        ("provisional", "subclass", "30 Astro-Epen", 250): {"threshold": 0.92},
        ("provisional", "supertype", "01 IT-ET Glut", 500): {"extrapolated": True},
        ("validated", "subclass", "19 MB Glut", 50): {"status": "not_resolvable"},
    }
    return make_decisions(
        levels=MOUSE_TABLE_LEVELS,
        classes=tuple(MOUSE_CLASS_CALLS),
        grid=MOUSE_GRID,
        overrides=overrides,
    )


def random_mouse_calls(rng: np.random.Generator, n: int) -> cs.MouseCalls:
    """Seeded random mouse objects (class calls, scores, depths, re-maps)."""
    names = [*MOUSE_CLASS_CALLS, None]
    picked = [names[int(rng.integers(len(names)))] for _ in range(n)]

    def part(index: int) -> list[Any]:
        return [
            None if cls is None else MOUSE_CLASS_CALLS[cls][index] for cls in picked
        ]

    def scores(low: float) -> list[float]:
        return [float(value) for value in rng.uniform(low, 1.0, size=n)]

    counts = rng.integers(5, 1500, size=n)
    corr = [float(value) for value in rng.uniform(0.3, 0.8, size=n)]
    dropped = [
        (None, None, None, "class", "subclass")[int(rng.integers(5))] for _ in range(n)
    ]
    wmb = cs.WmbCalls(
        klass=cs.LevelCall.of(picked, scores(0.8), corr=corr),
        broad=cs.LevelCall.of(part(0), scores(0.85), corr=corr),
        nt=cs.LevelCall.of(part(1), scores(0.85), corr=corr),
        subclass=cs.LevelCall.of(part(2), scores(0.7)),
        supertype=cs.LevelCall.of(part(3), scores(0.7)),
    )
    return cs.MouseCalls(
        total_counts=counts,
        in_table=counts >= 10,
        wmb=wmb,
        region_dropped=np.array(dropped, dtype=object),
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("constraint", PROMOTION_CONSTRAINTS)
def test_promotion_by_simulation_never_changes_what_resolve_mouse_emits(
    promotion_trust: Callable[..., tuple[Any, Any]],
    make_decisions: Callable[..., Any],
    constraint: str,
    seed: int,
) -> None:
    """Provisional -> simulation-validated trust: only validation marks change.

    The trust decisions come from ``trust_state`` before and after a gate-P
    PR lists the family; the gate verdict, the class correlation floor
    (``mouse_resolve.class_corr_floor``), the emission plan and the floors
    are derived from each the way mouse RESOLVE derives them.
    """
    from merxen.annotation.config import AnnotationConfig
    from merxen.annotation.mouse_resolve import class_corr_floor
    from merxen.annotation.resolvability import LevelMeta

    rng = np.random.default_rng(seed)
    calls = random_mouse_calls(rng, 400)
    trusts = promotion_trust(constraint, species="mouse")
    has_tables = constraint != "no_self_map"
    limits = AnnotationThresholds(allow_fine_levels=True, wmb_class_min_corr=0.5)
    config = AnnotationConfig(species="mouse", thresholds=limits)
    meta = tuple(
        LevelMeta(level, "CLAS", role, default, target, floor)  # type: ignore[arg-type]
        for level, role, default, target, floor in MOUSE_TABLE_LEVELS
    )
    signals = MouseGateSignals(
        registration=RegistrationSignal(density_ratio=2.9, shift_um=0.0),
        referee=MarkerReferee(
            consistency=0.9, n_pseudo_confident=500, n_cells=600, markers={}
        ),
        t2_share=0.004,
        spillover_rate=0.08,
    )
    results = []
    for trust in trusts:
        emission = th.EmissionPlan(
            species="mouse",
            thresholds=limits,
            decisions=promotion_decisions(make_decisions) if has_tables else None,
            levels=meta if has_tables else (),
            grid=MOUSE_GRID if has_tables else (),
            trust=trust,
            fine_seed_stability={"supertype": 0.0} if has_tables else None,
        )
        floors = th.FloorPlan.build(
            species="mouse",
            platform="MERSCOPE",
            hard_floor=10,
            trust=trust,
            thresholds=limits,
            emission=emission,
        )
        corr_floor, _ = class_corr_floor(config, trust)
        assert corr_floor is None  # real-data-validated families only (§16)
        settings = cs.MouseResolveSettings(
            platform="MERSCOPE",
            min_counts=10,
            emission=emission,
            floors=floors,
            thresholds=limits,
            trust=trust,
            class_min_corr=corr_floor,
            allow_fine_levels=True,
        )
        verdict = evaluate_mouse_gate(signals, MouseGateConfig(), trust=trust)
        results.append(cs.resolve_mouse(calls, settings, verdict))
    before, after = results
    before_trust, after_trust = trusts
    assert_same_emission(before, after)
    assert before.class_min_corr == after.class_min_corr
    assert not any(result.validated.any() for result in before.levels.values())
    table = before.in_table
    subclass_confident = before.levels["subclass"].confident
    if constraint == "resolvable":
        assert (before_trust.state, after_trust.state) == ("provisional", "validated")
        assert after.levels["class"].validated.any()
        assert before.gate.warning and before_trust.banner
        assert not after_trust.banner
        assert before.floor_warnings and not after.floor_warnings
        assert before.gate.level == "full" and subclass_confident.any()
        # The provisional subclass floor (60) holds after promotion too.
        assert (before.levels["subclass"].floor[table] >= 60).all()
        assert (after.levels["supertype"].floor[table] >= 60).all()
    else:
        assert before_trust.state == after_trust.state
        assert before_trust.effects() == after_trust.effects()
        assert not any(result.validated.any() for result in after.levels.values())
        assert before.gate.warning_reasons == after.gate.warning_reasons
    if constraint in ("no_self_map", "broad_only"):
        assert after.gate.level == "broad_only"
        assert set(after.levels["subclass"].status[table]) <= {
            "not_attempted_gate",
            "low_counts",
        }
