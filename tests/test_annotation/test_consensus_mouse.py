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


def test_mouse_calls_need_one_value_per_object() -> None:
    with pytest.raises(ValueError, match="one value per object"):
        cs.MouseCalls(
            total_counts=np.array([10, 20]),
            in_table=np.array([True]),
            wmb=None,
        )
