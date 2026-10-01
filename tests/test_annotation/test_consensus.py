"""Tests for the human consensus rules v1 (plan §5.2, §5.3, §4.1-§4.3)."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import consensus as cs
from merxen.annotation import thresholds as th
from merxen.annotation.config import AnnotationConfig, AnnotationThresholds
from merxen.annotation.resolvability import LevelMeta
from merxen.annotation.schema import (
    CellStatus,
    Columns,
    coerce_label_table_dtypes,
    validate_label_table,
)
from merxen.annotation.vocab import (
    COP_SUPERCLUSTER,
    UNASSIGNED_LABEL,
    UNRESOLVED_LABEL,
    primary_vocab,
)

T = AnnotationThresholds()
GRID = (10, 15, 30, 60, 120, 250)
MakeDecisions = Callable[..., pd.DataFrame]
MakeTrust = Callable[..., Any]

EXC = "Upper-layer intratelencephalic"
INH = "MGE interneuron"
OPC_SUPC = "Oligodendrocyte precursor"
ASTRO = "Astrocyte"
MICRO = "Microglia"
SEA_OF = {
    EXC: ("Neurons", "L2/3 IT"),
    INH: ("Neurons", "Sst"),
    OPC_SUPC: ("Oligodendrocyte precursors", "OPC"),
    COP_SUPERCLUSTER: ("Oligodendrocyte precursors", "OPC"),
    ASTRO: ("Astrocytes", "Astrocyte"),
    MICRO: ("Microglia", "Immune"),
    "Oligodendrocyte": ("Oligodendrocytes", "Oligodendrocyte"),
    "Splatter": ("Neurons", "L5 IT"),
    "Hippocampal CA1-3": ("Neurons", "L5 IT"),
    "Ependymal": ("Astrocytes", "Astrocyte"),
    "Vascular": ("Vascular cells", "Endothelial"),
}


@dataclass
class Cell:
    """One synthetic object: its WHB call, SEA-AD call and counts."""

    supercluster: str | None = EXC
    counts: int = 200
    bp: float = 0.95
    lineage_raw: float = 0.97
    broad_raw: float = 0.96
    nt_raw: float = 0.96
    sea_broad: str | None = "same"
    sea_raw: float = 0.95
    sea_subclass: str | None = "same"
    sea_subclass_raw: float = 0.9
    ll: str | None = "same"
    cluster: str | None = "C1"
    cluster_bp: float = 0.9


FILLER = Cell(ASTRO, counts=300)


def calls_of(
    cells: Sequence[Cell],
    *,
    whb: bool = True,
    sea: bool = True,
    ll: bool = False,
    min_counts: int = 10,
) -> cs.HumanCalls:
    """Build ``HumanCalls`` from synthetic cells."""
    counts = np.array([cell.counts for cell in cells], dtype=np.int64)
    names = [cell.supercluster for cell in cells]

    def resolve(value: str | None, cell: Cell, index: int) -> str | None:
        if value != "same":
            return value
        if cell.supercluster is None:
            return None
        return SEA_OF.get(cell.supercluster, ("Neurons", "L5 IT"))[index]

    whb_calls = cs.WhbCalls(
        supercluster=cs.LevelCall.of(
            names,
            [cell.bp for cell in cells],
            corr=[0.5] * len(cells),
            runner_up=["X"] * len(cells),
            margin=[0.3] * len(cells),
        ),
        lineage=cs.LevelScores.of([cell.lineage_raw for cell in cells]),
        broad=cs.LevelScores.of([cell.broad_raw for cell in cells]),
        nt=cs.LevelScores.of([cell.nt_raw for cell in cells]),
        cluster=cs.LevelCall.of(
            [cell.cluster for cell in cells], [cell.cluster_bp for cell in cells]
        ),
    )
    sea_calls = cs.SeaCalls(
        broad=cs.LevelCall.of(
            [resolve(cell.sea_broad, cell, 0) for cell in cells],
            [cell.sea_raw for cell in cells],
        ),
        subclass=cs.LevelCall.of(
            [resolve(cell.sea_subclass, cell, 1) for cell in cells],
            [cell.sea_subclass_raw for cell in cells],
        ),
    )
    return cs.HumanCalls(
        total_counts=counts,
        in_table=counts >= min_counts,
        whb=whb_calls if whb else None,
        sea=sea_calls if sea else None,
        ll_broad=(
            np.array([resolve(cell.ll, cell, 0) for cell in cells], dtype=object)
            if ll
            else None
        ),
    )


@dataclass
class Setup:
    """RESOLVE settings for a synthetic sample."""

    platform: str = "MERSCOPE"
    trust: Any = None
    secondary_trust: Any = None
    decisions: pd.DataFrame | None = None
    meta: Sequence[LevelMeta] = ()
    thresholds: AnnotationThresholds = field(default_factory=AnnotationThresholds)
    allow_single_method: bool = False
    use_likelihood_vote: bool = False
    n_segmented: int | None = None
    fine_seed_stability: dict[str, float] | None = None

    def settings(self) -> cs.HumanResolveSettings:
        emission = th.EmissionPlan(
            species="human",
            thresholds=self.thresholds,
            decisions=self.decisions,
            levels=tuple(self.meta),
            grid=GRID if self.decisions is not None else (),
            trust=self.trust,
            fine_seed_stability=self.fine_seed_stability,
        )
        floors = th.FloorPlan.build(
            species="human",
            platform=self.platform,
            hard_floor=10,
            trust=self.trust,
            thresholds=self.thresholds,
            emission=emission,
        )
        return cs.HumanResolveSettings(
            platform=self.platform,
            min_counts=10,
            emission=emission,
            floors=floors,
            thresholds=self.thresholds,
            trust=self.trust,
            secondary_trust=self.secondary_trust,
            allow_single_method=self.allow_single_method,
            allow_fine_levels=self.thresholds.allow_fine_levels,
            use_likelihood_vote=self.use_likelihood_vote,
            n_segmented=self.n_segmented,
        )


def resolve(
    cells: Sequence[Cell],
    setup: Setup | None = None,
    *,
    fillers: int = 20,
    **call_options: Any,
) -> cs.HumanResolution:
    """Resolve cells after ``fillers`` deep astrocytes (the gate stays full)."""
    return cs.resolve_human(
        calls_of([*cells, *[FILLER] * fillers], **call_options),
        (setup or Setup()).settings(),
    )


def statuses(result: cs.HumanResolution, index: int = 0) -> dict[str, str]:
    return {level: str(values.status[index]) for level, values in result.levels.items()}


# --------------------------------------------------------------------------
# §5.3 degraded-mode truth table (exhaustive)

SEA_OUTCOMES = ("agree_confident", "agree_weak", "differ_weak", "differ_confident")
LL_OUTCOMES = ("agree", "differ")
DEPTHS = (30, 100)


def expected_mode(whb: bool, sea: bool, ll: bool) -> str:
    """The §5.3 row, written from the table text."""
    if not whb:
        return "primary_missing"
    return {
        (True, True): "whb_sea_ll",
        (True, False): "whb_sea",
        (False, True): "whb_ll",
        (False, False): "whb_only",
    }[(sea, ll)]


def expected_vote(
    mode: str, *, below60: bool, sea: str, ll: str, allow_single: bool
) -> str:
    """Expected lineage status of a plausible, confident WHB call."""
    if mode == "primary_missing":
        return "not_attempted_gate"
    sea_agrees = sea.startswith("agree")
    sea_vetoes = sea == "differ_confident"
    ll_agrees = ll == "agree"
    if below60:
        if mode == "whb_sea":
            return "confident" if sea_agrees else "method_disagree"
        if mode == "whb_sea_ll":
            return "confident" if (sea_agrees or ll_agrees) else "method_disagree"
        if mode == "whb_ll":
            return "confident" if ll_agrees else "method_disagree"
        return "confident" if allow_single else "single_method"
    if mode in ("whb_sea", "whb_sea_ll"):
        return "method_disagree" if sea_vetoes else "confident"
    return "confident"


def expected_tier(mode: str, *, sea: str, ll: str) -> int:
    """Methods agreeing at the 7-class level among informative ones."""
    if mode == "primary_missing":
        return cs.TIER_NONE_INFORMATIVE
    labels = ["whb"]
    if "sea" in mode and sea.endswith("confident"):
        labels.append("whb" if sea.startswith("agree") else "other")
    if "ll" in mode:
        labels.append("whb" if ll == "agree" else "other2")
    if len(labels) == 1:
        return 1
    largest = max(labels.count(value) for value in set(labels))
    return largest if largest >= 2 else 0


def truth_table_cells() -> list[tuple[int, str, str, Cell]]:
    cells = []
    for depth, sea, ll in itertools.product(DEPTHS, SEA_OUTCOMES, LL_OUTCOMES):
        cells.append(
            (
                depth,
                sea,
                ll,
                Cell(
                    EXC,
                    counts=depth,
                    sea_broad="Neurons" if sea.startswith("agree") else "Astrocytes",
                    sea_raw=0.95 if sea.endswith("confident") else 0.40,
                    ll="Neurons" if ll == "agree" else "Microglia",
                ),
            )
        )
    return cells


@pytest.mark.parametrize(
    ("whb", "sea", "ll", "allow_single"),
    list(itertools.product((True, False), repeat=4)),
)
def test_degraded_mode_truth_table_is_exhaustive(
    whb: bool, sea: bool, ll: bool, allow_single: bool
) -> None:
    table = truth_table_cells()
    result = resolve(
        [cell for *_, cell in table],
        Setup(allow_single_method=allow_single, use_likelihood_vote=ll),
        whb=whb,
        sea=sea,
        ll=ll,
    )
    mode = expected_mode(whb, sea, ll)
    assert result.mode.name == mode
    assert result.mode is cs.degraded_mode(whb=whb, sea=sea, ll=ll)
    assert result.single_method_override is (mode == "whb_only" and allow_single)
    for index, (depth, sea_outcome, ll_outcome, _cell) in enumerate(table):
        want = expected_vote(
            mode,
            below60=depth < 60,
            sea=sea_outcome if sea else "none",
            ll=ll_outcome if ll else "none",
            allow_single=allow_single,
        )
        got = statuses(result, index)
        case = (mode, depth, sea_outcome, ll_outcome, allow_single)
        assert got["lineage"] == want, case
        if want == "confident":
            assert got["broad"] == "confident", case
            assert result.final_level[index] == "supercluster", case
        elif want == "not_attempted_gate":
            assert set(got.values()) == {"not_attempted_gate"}, case
        else:
            assert got["broad"] == "parent_unresolved", case
            assert result.final_level[index] == "none", case
        tier = expected_tier(
            mode, sea=sea_outcome if sea else "none", ll=ll_outcome if ll else "none"
        )
        assert result.consensus_tier[index] == tier, case
        assert result.consensus_tier[index] <= result.mode.max_tier
        if not sea or mode == "primary_missing":
            assert got["seaad_subclass"] == "not_attempted_gate", case
    if mode == "primary_missing":
        assert result.gate.level == "failed"
        assert result.flags[Columns.EXCLUDE_HARD].all()


def test_the_truth_table_rows_are_the_plan_rows() -> None:
    table = cs.DEGRADED_MODES
    assert set(table) == {
        "whb_sea",
        "whb_sea_ll",
        "whb_ll",
        "whb_only",
        "primary_missing",
    }
    assert [table[name].max_tier for name in table] == [2, 3, 2, 1, 0]
    assert table["whb_only"].missing_status == CellStatus.SINGLE_METHOD
    assert table["primary_missing"].missing_status == CellStatus.NOT_ATTEMPTED_GATE
    assert table["whb_sea"].below60_votes == ("sea",)
    assert table["whb_sea_ll"].below60_votes == ("sea", "ll")
    assert table["whb_ll"].veto_methods == ()
    assert cs.degraded_mode(whb=False, sea=True).name == "primary_missing"
    assert table["whb_sea"].to_json()["missing_status"] is None


def test_a_refused_second_vote_is_the_single_method_mode(
    make_trust: MakeTrust,
) -> None:
    result = resolve(
        [Cell(EXC, counts=30), Cell(EXC, counts=100)],
        Setup(
            trust=make_trust("validated_real"),
            secondary_trust=make_trust("refused", role="secondary"),
        ),
    )
    assert result.mode.name == "whb_only"
    assert statuses(result, 0)["lineage"] == "single_method"
    assert statuses(result, 1)["lineage"] == "confident"
    assert statuses(result, 1)["seaad_subclass"] == "not_attempted_gate"
    assert result.consensus_tier[1] == 1


def test_a_refused_primary_leaves_statuses_only(make_trust: MakeTrust) -> None:
    result = resolve(
        [Cell(EXC), Cell(ASTRO, counts=4)], Setup(trust=make_trust("refused"))
    )
    assert result.mode.name == "primary_missing"
    assert result.gate.level == "failed"
    assert any(r.startswith("panel_refused") for r in result.gate.level_reasons)
    assert set(statuses(result, 0).values()) == {"not_attempted_gate"}
    assert set(statuses(result, 1).values()) == {"low_counts"}
    assert result.flags[Columns.EXCLUDE_HARD].all()
    assert (result.final_level == "none").all()
    assert result.levels["broad"].name[0] is None


# --------------------------------------------------------------------------
# §5.2 rules


def test_the_cop_rule(make_trust: MakeTrust) -> None:
    cop = COP_SUPERCLUSTER
    cells = [
        # >= 120 counts and supercluster bp >= 0.69: broad OPC.
        Cell(cop, counts=150, bp=0.80, sea_broad="Microglia", sea_raw=0.3),
        # >= 120 counts, bp < 0.69 and no confident SEA OPC: suppressed.
        Cell(cop, counts=150, bp=0.60, sea_raw=0.4),
        # < 120 counts, no confident SEA OPC: suppressed below the COP floor.
        Cell(cop, counts=80, bp=0.95, sea_raw=0.4),
        # < 120 counts but SEA-AD confidently calls OPC: broad OPC.
        Cell(cop, counts=40, bp=0.95, sea_raw=0.9),
        # SEA-AD confidently calls oligodendrocytes: no rescue, suppressed.
        Cell(cop, counts=80, bp=0.95, sea_broad="Oligodendrocytes", sea_raw=0.9),
        # A plain OPC call needs no COP rule.
        Cell(OPC_SUPC, counts=40, bp=0.95),
    ]
    result = resolve(cells, Setup(trust=make_trust("validated_real")))
    broad = result.levels["broad"].status
    flag = result.flags[Columns.FLAG_COP_SUPPRESSED]
    assert broad[:6].tolist() == [
        "confident",
        "low_confidence",
        "below_floor",
        "confident",
        "below_floor",
        "confident",
    ]
    assert flag[:6].tolist() == [False, True, True, False, True, False]
    for index in (1, 2, 4):
        assert result.final_level[index] == "lineage"
        assert result.final_name[index] == "Oligodendrocyte lineage"
        assert result.levels["lineage"].status[index] == "confident"
    for index in (0, 3, 5):
        assert result.levels["broad"].name[index] == "Oligodendrocyte precursors"
    # COP only as a supercluster at >= 120 counts.
    supercluster = result.levels["supercluster"].status
    assert supercluster[0] == "confident"
    assert supercluster[3] == "below_floor"
    summary = result.summary()["cop_control"]
    assert summary["cop_suppressed"] == 3
    # Broad OPC: cells 0, 3 (COP-derived) and 5 (OPC supercluster).
    assert summary["cop_derived_opc_share"] == pytest.approx(2 / 3)


def test_the_cop_rule_boundary_is_120_counts(make_trust: MakeTrust) -> None:
    """The pre-registered COP supercluster floor: >= 120 counts (§5.2 rule 2)."""
    cop = COP_SUPERCLUSTER
    cells = [
        Cell(cop, counts=120, bp=0.80, sea_raw=0.3),
        Cell(cop, counts=119, bp=0.80, sea_raw=0.3),
    ]
    result = resolve(cells, Setup(trust=make_trust("validated_real")))
    assert statuses(result, 0)["broad"] == "confident"
    assert statuses(result, 0)["supercluster"] == "confident"
    assert result.final_level[0] == "supercluster"
    assert statuses(result, 1)["broad"] == "below_floor"
    flag = result.flags[Columns.FLAG_COP_SUPPRESSED]
    assert flag[:2].tolist() == [False, True]


def test_the_cop_flag_marks_every_failed_cop_rule_on_a_confident_lineage(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    """flag_cop_suppressed is the plan §4.3 definition, whichever check decides.

    WHB called COP, the lineage is confident and the COP rule failed: the
    flag is set even when the broad threshold, floor or resolvability fails
    first (the status still follows the precedence); it is never set when
    the lineage is not confident or the rule passes.
    """
    cop = COP_SUPERCLUSTER
    decisions = make_decisions(
        overrides={
            (regime, "broad", "OPC", 60): {"status": "not_resolvable"}
            for regime in ("validated", "provisional")
        }
    )
    cells = [
        # The broad threshold fails first; the COP rule fails too.
        Cell(cop, counts=150, bp=0.60, broad_raw=0.5, sea_raw=0.3),
        # Lineage not confident: not flagged.
        Cell(cop, counts=150, bp=0.60, lineage_raw=0.5, sea_raw=0.3),
        # Not resolvable at broad (OPC at 60) and a failing COP rule.
        Cell(cop, counts=80, bp=0.95, sea_raw=0.3),
        # The COP rule passes (SEA-AD confidently calls OPC) but broad is not
        # resolvable: not flagged.
        Cell(cop, counts=80, bp=0.95, sea_raw=0.9),
        # The COP rule decides (every other check passes).
        Cell(cop, counts=150, bp=0.60, sea_raw=0.3),
    ]
    result = resolve(
        cells,
        Setup(
            trust=make_trust("validated_real"),
            decisions=decisions,
            meta=human_level_meta,
        ),
    )
    broad = result.levels["broad"].status
    assert broad[:5].tolist() == [
        "low_confidence",
        "parent_unresolved",
        "not_resolvable",
        "not_resolvable",
        "low_confidence",
    ]
    flag = result.flags[Columns.FLAG_COP_SUPPRESSED]
    assert flag[:5].tolist() == [True, False, True, False, True]
    assert result.summary()["cop_control"]["cop_suppressed"] == 3
    # Labels do not depend on the flag: flagged cells stay at lineage.
    for index in (0, 2, 4):
        assert result.final_level[index] == "lineage"


def test_the_second_vote_switches_at_60_counts() -> None:
    cells = [
        Cell(EXC, counts=59, sea_broad="Astrocytes", sea_raw=0.3),
        Cell(EXC, counts=60, sea_broad="Astrocytes", sea_raw=0.3),
        Cell(EXC, counts=60, sea_broad="Astrocytes", sea_raw=0.9),
    ]
    result = resolve(cells)
    assert [statuses(result, index)["lineage"] for index in range(3)] == [
        "method_disagree",
        "confident",
        "method_disagree",
    ]


def test_sea_confident_disagreement_blocks_from_60_counts() -> None:
    cells = [
        Cell(EXC, counts=100, sea_broad="Astrocytes", sea_raw=0.68),
        Cell(EXC, counts=100, sea_broad="Astrocytes", sea_raw=0.67),
        # Disagreement at the 7-class level only (same lineage).
        Cell(OPC_SUPC, counts=100, sea_broad="Oligodendrocytes", sea_raw=0.9),
        Cell(OPC_SUPC, counts=30, sea_broad="Oligodendrocytes", sea_raw=0.2),
        # Below 60 counts SEA-AD must agree even when unconfident.
        Cell(EXC, counts=30, sea_raw=0.2),
    ]
    result = resolve(cells)
    assert statuses(result, 0)["lineage"] == "method_disagree"
    assert statuses(result, 1)["lineage"] == "confident"
    assert (statuses(result, 2)["lineage"], statuses(result, 2)["broad"]) == (
        "confident",
        "method_disagree",
    )
    assert (statuses(result, 3)["lineage"], statuses(result, 3)["broad"]) == (
        "confident",
        "method_disagree",
    )
    assert result.final_level[3] == "lineage"
    assert statuses(result, 4)["lineage"] == "confident"
    assert result.flags[Columns.FLAG_METHOD_DISAGREE][:5].tolist() == [
        True,
        False,
        True,
        True,
        False,
    ]
    assert result.consensus_tier[:5].tolist() == [0, 1, 0, 1, 1]


def test_implausible_nodes_keep_their_lineage_only_with_sea_agreement() -> None:
    cells = [
        Cell("Splatter", counts=200),
        Cell("Splatter", counts=200, sea_broad="Astrocytes", sea_raw=0.9),
        Cell("Hippocampal CA1-3", counts=200),
        Cell("Ependymal", counts=200),
        Cell("Miscellaneous", counts=40, sea_broad="Neurons", sea_raw=0.3),
    ]
    result = resolve(cells)
    rescued = statuses(result, 0)
    assert rescued["lineage"] == "confident"
    assert {rescued[level] for level in ("broad", "nt", "supercluster")} == {
        "implausible"
    }
    assert (result.final_level[0], result.final_name[0]) == ("lineage", "Neurons")
    assert result.flags[Columns.FLAG_IMPLAUSIBLE][0]
    assert not result.flags[Columns.EXCLUDE_HARD][0]
    lost = statuses(result, 1)
    assert set(lost[level] for level in cs.HUMAN_CHAIN) == {"implausible"}
    assert result.flags[Columns.EXCLUDE_HARD][1]
    assert result.final_level[1] == "none"
    assert statuses(result, 2)["lineage"] == "confident"
    assert statuses(result, 2)["broad"] == "implausible"
    # Ependymal has no lineage in the vocab: SEA-AD cannot rescue it.
    assert statuses(result, 3)["lineage"] == "implausible"
    assert statuses(result, 3)["nt"] == "not_applicable"
    assert result.flags[Columns.EXCLUDE_HARD][3]
    # Rescued below 60 counts too (SEA-AD agrees at lineage).
    assert statuses(result, 4)["lineage"] == "confident"
    assert result.summary()["flag_counts"][Columns.FLAG_IMPLAUSIBLE] == 5


def test_nt_is_not_applicable_to_non_neurons_and_gates_neuron_leaves() -> None:
    cells = [
        Cell(ASTRO, counts=200),
        # NT below 0.73 blocks the neuron's supercluster (its parent is NT).
        Cell(EXC, counts=200, nt_raw=0.70, bp=0.90),
        Cell(INH, counts=200),
    ]
    result = resolve(cells, Setup(platform="XENIUM"))
    assert statuses(result, 0)["nt"] == "not_applicable"
    assert result.levels["nt"].name[0] is None
    assert np.isnan(result.levels["nt"].raw[0])
    assert result.final_level[0] == "supercluster"
    assert statuses(result, 1)["nt"] == "low_confidence"
    assert statuses(result, 1)["supercluster"] == "parent_unresolved"
    assert result.final_level[1] == "broad"
    assert (result.levels["nt"].name[2], result.final_level[2]) == (
        "Inhibitory",
        "supercluster",
    )


def test_floors_per_class_and_platform() -> None:
    cells = [Cell(INH, counts=20), Cell(INH, counts=35), Cell(MICRO, counts=12)]
    merscope = resolve(cells, Setup(platform="MERSCOPE"))
    xenium = resolve(cells, Setup(platform="XENIUM"))
    # Unknown-panel floors (no trust): max over platforms, Inh broad 30.
    assert statuses(merscope, 0)["broad"] == "below_floor"
    assert merscope.flags[Columns.FLAG_BELOW_FLOOR][0]
    assert statuses(xenium, 1)["broad"] == "confident"
    assert statuses(xenium, 1)["supercluster"] == "confident"
    assert statuses(xenium, 2)["broad"] == "below_floor"


def test_validated_floors_follow_the_platform(make_trust: MakeTrust) -> None:
    cells = [Cell(INH, counts=20), Cell(INH, counts=20)]
    trust = make_trust("validated_real")
    merscope = resolve(cells, Setup(platform="MERSCOPE", trust=trust))
    xenium = resolve(cells, Setup(platform="XENIUM", trust=trust))
    # Inh: broad MERSCOPE 10 / Xenium 30; supercluster MERSCOPE 30 / Xenium 10.
    assert statuses(merscope, 0)["broad"] == "confident"
    assert statuses(merscope, 0)["supercluster"] == "below_floor"
    assert statuses(xenium, 0)["broad"] == "below_floor"
    assert merscope.levels["broad"].floor[0] == 10
    assert xenium.levels["broad"].floor[0] == 30


def test_the_sea_subclass_rule() -> None:
    cells = [
        Cell(EXC, counts=100, sea_subclass_raw=0.46),
        Cell(EXC, counts=100, sea_subclass_raw=0.44),
        Cell(EXC, counts=40, sea_subclass_raw=0.50),
        Cell(ASTRO, counts=100, sea_broad="Microglia", sea_raw=0.3),
        Cell(OPC_SUPC, counts=100),
        # ct_broad not confident: parent_unresolved.
        Cell(EXC, counts=100, broad_raw=0.5, sea_subclass_raw=0.9),
        # SEA-AD disagrees and the subclass is weak: the documented order
        # puts low_confidence before method_disagree.
        Cell(
            ASTRO, counts=100, sea_broad="Microglia", sea_raw=0.3, sea_subclass_raw=0.3
        ),
    ]
    result = resolve(cells, Setup(platform="MERSCOPE"))
    sea = result.levels["seaad_subclass"]
    assert sea.status[:7].tolist() == [
        "confident",
        "low_confidence",
        "low_confidence",
        "method_disagree",
        "below_floor",
        "parent_unresolved",
        "low_confidence",
    ]
    assert sea.threshold[:3].tolist() == [0.45, 0.45, 0.55]
    # Never in ct_final.
    assert result.final_level[0] == "supercluster"
    assert result.final_name[0] == EXC


def test_the_sea_subclass_follows_the_leaf_resolvability(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    """Rule 5: the leaf's resolvability for the broad class gates the subclass."""
    decisions = make_decisions(
        overrides={
            ("validated", "supercluster", "Exc", 120): {"status": "not_resolvable"}
        }
    )
    cells = [Cell(EXC, counts=150), Cell(EXC, counts=300)]
    result = resolve(
        cells,
        Setup(
            trust=make_trust("validated_real"),
            decisions=decisions,
            meta=human_level_meta,
        ),
    )
    assert statuses(result, 0)["broad"] == "confident"
    assert statuses(result, 0)["seaad_subclass"] == "not_resolvable"
    assert statuses(result, 1)["seaad_subclass"] == "confident"


def test_the_whb_cluster_is_never_a_final_label(
    make_trust: MakeTrust,
    human_level_meta: list[LevelMeta],
    make_decisions: MakeDecisions,
) -> None:
    cells = [Cell(EXC, counts=200)]
    off = resolve(cells, Setup(trust=make_trust("validated_real")))
    assert "cluster" not in off.levels
    enabled = AnnotationThresholds(allow_fine_levels=True)
    unstable = resolve(
        cells, Setup(trust=make_trust("validated_real"), thresholds=enabled)
    )
    assert statuses(unstable)["cluster"] == "not_resolvable"
    stable = resolve(
        cells,
        Setup(
            trust=make_trust("validated_real"),
            thresholds=enabled,
            decisions=make_decisions(),
            meta=human_level_meta,
            fine_seed_stability={"cluster": 0.0},
        ),
    )
    assert statuses(stable)["cluster"] == "confident"
    assert stable.final_level[0] == "supercluster"
    assert "cluster" not in cs.HUMAN_CHAIN
    # A broad-only dataset attempts no level below broad (§5.4), the
    # report-only cluster included, even when it is stable and resolvable.
    broad_only = resolve(
        cells,
        Setup(
            trust=make_trust("broad_only"),
            thresholds=enabled,
            decisions=make_decisions(),
            meta=human_level_meta,
            fine_seed_stability={"cluster": 0.0},
        ),
    )
    assert broad_only.gate.level == "broad_only"
    assert statuses(broad_only)["broad"] == "confident"
    assert statuses(broad_only)["cluster"] == "not_attempted_gate"
    assert broad_only.levels["cluster"].name[0] is None


# --------------------------------------------------------------------------
# Resolvability and the gate


def test_resolvability_gates_emission_and_thresholds(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    decisions = make_decisions(
        overrides={
            (regime, "broad", "Exc", 15): {"status": "not_resolvable"}
            for regime in ("validated", "provisional")
        }
        | {
            ("provisional", "broad", "Exc", 60): {"threshold": 0.90},
            ("validated", "supercluster", "Exc", 250): {"extrapolated": True},
        }
    )
    cells = [
        Cell(EXC, counts=20),
        Cell(EXC, counts=80, broad_raw=0.85),
        Cell(EXC, counts=400),
    ]
    validated = resolve(
        cells,
        Setup(
            trust=make_trust("validated_real"),
            decisions=decisions,
            meta=human_level_meta,
        ),
    )
    assert statuses(validated, 0)["broad"] == "not_resolvable"
    assert statuses(validated, 1)["broad"] == "confident"
    assert validated.resolvability_extrapolated[:3].tolist() == [False, False, True]
    assert validated.depth_bin[:3].tolist() == [15.0, 60.0, 250.0]
    provisional = resolve(
        cells,
        Setup(
            trust=make_trust("provisional"), decisions=decisions, meta=human_level_meta
        ),
    )
    assert statuses(provisional, 1)["broad"] == "low_confidence"
    assert provisional.levels["broad"].threshold[1] == pytest.approx(0.90)
    assert provisional.gate.level == "full" and provisional.gate.warning


def test_reweighting_changes_emission_through_the_decisions(
    make_trust: MakeTrust,
    human_level_meta: list[LevelMeta],
    make_decisions: MakeDecisions,
) -> None:
    # The same cells under two dataset compositions: the reweighted table of
    # one no longer emits Immune at 60 (RESOLVE refits per dataset, §8.3).
    cells = [Cell(MICRO, counts=70)]
    unweighted = make_decisions()
    reweighted = make_decisions(
        overrides={("validated", "broad", "Immune", 60): {"status": "not_resolvable"}}
    )
    before = resolve(
        cells,
        Setup(
            trust=make_trust("validated_real"),
            decisions=unweighted,
            meta=human_level_meta,
        ),
    )
    after = resolve(
        cells,
        Setup(
            trust=make_trust("validated_real"),
            decisions=reweighted,
            meta=human_level_meta,
        ),
    )
    assert statuses(before)["broad"] == "confident"
    assert statuses(after)["broad"] == "not_resolvable"


def test_a_broad_only_gate_blocks_only_the_leaf_levels(make_trust: MakeTrust) -> None:
    shallow = [Cell(ASTRO, counts=20) for _ in range(80)]
    result = resolve(shallow, Setup(trust=make_trust("validated_real")), fillers=20)
    assert result.gate.level == "broad_only"
    assert statuses(result)["broad"] == "confident"
    assert statuses(result)["supercluster"] == "not_attempted_gate"
    assert statuses(result)["seaad_subclass"] == "not_attempted_gate"
    assert result.levels["supercluster"].name[0] is None
    assert result.final_level[0] == "broad"
    assert (result.final_level[80:] == "broad").all()
    capped = resolve([Cell(EXC)], Setup(trust=make_trust("broad_only")))
    assert capped.gate.level == "broad_only"
    assert statuses(capped)["supercluster"] == "not_attempted_gate"


def test_the_status_precedence_puts_the_floor_before_resolvability(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    """below_floor > not_resolvable > low_confidence > method_disagree."""
    decisions = make_decisions(
        overrides={
            ("validated", "broad", "Inh", 15): {"status": "not_resolvable"},
            ("validated", "broad", "Exc", 60): {"status": "not_resolvable"},
        }
    )
    cells = [
        # Xenium broad Inh floor 30: below it and in a non-emitted bin.
        Cell(INH, counts=15),
        # Not resolvable (Exc at 60) and weak: not_resolvable.
        Cell(EXC, counts=80, broad_raw=0.5, sea_broad="Neurons", sea_raw=0.3),
        # Weak, and SEA-AD confidently disagrees at the 7-class level only
        # (same lineage): low_confidence.
        Cell(
            OPC_SUPC,
            counts=300,
            broad_raw=0.5,
            sea_broad="Oligodendrocytes",
            sea_raw=0.95,
        ),
    ]
    result = resolve(
        cells,
        Setup(
            platform="XENIUM",
            trust=make_trust("validated_real"),
            decisions=decisions,
            meta=human_level_meta,
        ),
    )
    assert [statuses(result, index)["lineage"] for index in range(3)] == [
        "confident"
    ] * 3
    assert [statuses(result, index)["broad"] for index in range(3)] == [
        "below_floor",
        "not_resolvable",
        "low_confidence",
    ]


def test_the_segmented_share_uses_the_gate_denominator() -> None:
    cells = [
        Cell(EXC, counts=200),
        Cell(EXC, counts=5),
        Cell(EXC, counts=11, broad_raw=0.1),
    ]
    given = resolve(cells, Setup(n_segmented=400), fillers=20)
    levels = given.summary()["levels"]
    n_broad = int(given.levels["broad"].confident.sum())
    assert n_broad == 21
    assert levels["broad"]["confident_share_segmented"] == pytest.approx(n_broad / 400)
    assert levels["broad"]["confident_share_segmented"] == pytest.approx(
        given.gate.broad_coverage_segmented
    )
    assert levels["broad"]["confident_share_table"] == pytest.approx(n_broad / 22)
    unknown = resolve(cells, fillers=20).summary()["levels"]["broad"]
    assert unknown["confident_share_segmented"] == pytest.approx(n_broad / 23)


def test_a_failed_gate_attempts_nothing() -> None:
    weak = [Cell(EXC, counts=100, broad_raw=0.5) for _ in range(90)]
    result = resolve(weak, fillers=10)
    assert result.gate.level == "failed"
    assert set(statuses(result, 95).values()) == {"not_attempted_gate"}
    assert result.flags[Columns.EXCLUDE_HARD].all()
    assert (result.final_level == "none").all()


def test_validated_marks_and_the_simulation_family_warning(
    make_trust: MakeTrust,
) -> None:
    cells = [
        Cell(EXC, counts=200),
        Cell(EXC, counts=20),
        Cell(EXC, counts=11, broad_raw=0.1),
    ]
    real = resolve(cells, Setup(trust=make_trust("validated_real")))
    assert real.levels["broad"].validated[:3].tolist() == [True, True, False]
    assert real.levels["supercluster"].validated[0]
    simulation = make_trust(
        "validated_simulation",
        level_records=[
            {
                "level": "broad",
                "class": "Astro",
                "status": "validated",
                "validated_min_depth": 10,
                "tested_max_depth": 120,
            },
            {
                "level": "broad",
                "class": "Exc",
                "status": "validated",
                "validated_min_depth": 60,
                "tested_max_depth": 120,
            },
        ],
    )
    result = resolve(cells, Setup(trust=simulation))
    assert result.levels["broad"].validated[:3].tolist() == [True, False, False]
    assert result.levels["lineage"].validated[:3].tolist() == [False, False, False]
    # Lineage and NT labels lie outside the validated region: warning, level full.
    assert result.gate.level == "full" and result.gate.warning
    assert any(
        r.startswith("unvalidated_share:lineage") for r in result.gate.warning_reasons
    )


# --------------------------------------------------------------------------
# Coverage of the status vocabulary and the label-table contract


def test_every_cell_status_is_reachable(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    decisions = make_decisions(
        overrides={
            (regime, "broad", "Immune", 120): {"status": "not_resolvable"}
            for regime in ("validated", "provisional")
        }
    )
    cells = [
        Cell(ASTRO, counts=5),  # low_counts
        Cell(ASTRO, counts=200),  # confident, not_applicable (NT)
        Cell(EXC, counts=200, broad_raw=0.5),  # low_confidence, parent_unresolved
        Cell(INH, counts=12),  # below_floor (Inh supercluster MERSCOPE 30)
        Cell(MICRO, counts=150),  # not_resolvable
        Cell(EXC, counts=100, sea_broad="Astrocytes"),  # method_disagree
        Cell("Splatter", counts=100, sea_broad="Astrocytes"),  # implausible
    ]
    with_sea = resolve(
        cells,
        Setup(
            trust=make_trust("validated_real"),
            decisions=decisions,
            meta=human_level_meta,
        ),
    )
    without_sea = resolve(
        [Cell(EXC, counts=30)],
        Setup(trust=make_trust("validated_real")),
        sea=False,
    )  # single_method; not_attempted_gate (seaad_subclass)
    seen = set()
    for result in (with_sea, without_sea):
        for values in result.levels.values():
            seen |= {str(value) for value in values.status}
    assert seen == {status.value for status in CellStatus}
    assert statuses(with_sea, 3)["supercluster"] == "below_floor"
    assert statuses(with_sea, 4)["broad"] == "not_resolvable"


def label_table(result: cs.HumanResolution, calls: cs.HumanCalls) -> pd.DataFrame:
    """A full label table around ``resolve_human``'s columns (schema check)."""
    n = len(calls)
    counts = np.asarray(calls.total_counts, dtype=np.int32)
    table = np.asarray(calls.in_table, dtype=bool)
    data: dict[str, Any] = {
        Columns.CELL_ID: [f"cell_{index}" for index in range(n)],
        Columns.INSTANCE_ID: np.arange(1, n + 1, dtype=np.int64),
        Columns.PAIR_ID: ["P0001"] * n,
        Columns.SAMPLE_ID: ["P0001_MERSCOPE"] * n,
        Columns.PLATFORM: ["MERSCOPE"] * n,
        Columns.SEGMENTATION: ["proseg_hybrid"] * n,
        Columns.SPECIES: ["human"] * n,
        Columns.ANATOMICAL_REGION: ["frontal_cortex"] * n,
        Columns.PANEL_HASH: ["a" * 64] * n,
        Columns.TOTAL_COUNTS: counts,
        Columns.N_GENES: np.minimum(counts, 40).astype(np.int32),
        Columns.GENES_PER_COUNT: (np.minimum(counts, 40) / counts).astype(np.float32),
        Columns.IN_TABLE: table,
        Columns.N_MISSING_PANEL_GENES: np.zeros(n, dtype=np.int32),
        **result.to_columns(),
    }
    vocab = primary_vocab("human")
    branches = []
    for level, name in zip(result.final_level, result.final_name, strict=True):
        if level == "none":
            branches.append(UNASSIGNED_LABEL)
        elif level == "supercluster":
            broad = vocab.broad_class(str(name))
            branches.append("Neurons/Excitatory" if broad == "Neurons" else broad)
        else:
            branches.append(UNASSIGNED_LABEL)
    data[Columns.CT_BRANCH] = branches
    data[Columns.CT_LEAF] = [
        name if level == "supercluster" else UNRESOLVED_LABEL
        for level, name in zip(result.final_level, result.final_name, strict=True)
    ]
    data[Columns.CT_MENDER_STATE] = branches
    nullable = pd.array([None] * n, dtype="boolean")
    data.update(
        {
            Columns.CONTAMINATION_SCORE: np.full(n, np.nan),
            Columns.NEG_COUNTS: pd.array([None] * n, dtype="Int32"),
            Columns.FLAG_CONTAMINATED: nullable.copy(),
            Columns.EXPECTED_GENES_Q95: np.full(n, np.nan),
            Columns.FLAG_DIFFUSE_PROFILE: nullable.copy(),
            Columns.OOD_Z: np.full(n, np.nan),
            Columns.FLAG_OOD: nullable.copy(),
            Columns.MICROGLIA_STAT: np.full(n, np.nan),
            Columns.MICROGLIA_WEIGHT: np.full(n, np.nan),
            Columns.FLAG_MICROGLIAL_SPILLOVER: nullable.copy(),
            Columns.DISCOVERY_CAUTION: np.zeros(n, dtype=bool),
        }
    )
    return coerce_label_table_dtypes(pd.DataFrame(data), "human")


def test_resolved_columns_satisfy_the_label_table_contract(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    rng = np.random.default_rng(11)
    names = [EXC, INH, ASTRO, MICRO, OPC_SUPC, COP_SUPERCLUSTER, "Splatter", "Vascular"]
    cells = [
        Cell(
            str(rng.choice(names)),
            counts=int(rng.integers(3, 400)),
            bp=float(rng.uniform(0.3, 1.0)),
            lineage_raw=float(rng.uniform(0.5, 1.0)),
            broad_raw=float(rng.uniform(0.5, 1.0)),
            nt_raw=float(rng.uniform(0.5, 1.0)),
            sea_raw=float(rng.uniform(0.2, 1.0)),
            sea_broad=str(rng.choice(["same", "Astrocytes", "Neurons"])),
            sea_subclass_raw=float(rng.uniform(0.2, 1.0)),
        )
        for _ in range(400)
    ]
    decisions = make_decisions(
        overrides={
            ("validated", "supercluster", "Exc", 15): {"status": "not_resolvable"},
            ("validated", "broad", "Vascular", 30): {"extrapolated": True},
        }
    )
    calls = calls_of([*cells, *[FILLER] * 200])
    for trust in ("validated_real", "provisional"):
        result = cs.resolve_human(
            calls,
            Setup(
                trust=make_trust(trust), decisions=decisions, meta=human_level_meta
            ).settings(),
        )
        frame = label_table(result, calls)
        validate_label_table(frame, "human")
        summary = result.summary()
        assert summary["n_table"] == int(calls.in_table.sum())
        assert set(summary["levels"]) == {
            "lineage",
            "broad",
            "nt",
            "supercluster",
            "seaad_subclass",
        }


def test_settings_from_a_coupled_config(make_trust: MakeTrust) -> None:
    config = AnnotationConfig(
        species="human", allow_single_method=True
    ).with_min_counts(10)
    emission = th.EmissionPlan(species="human", thresholds=config.thresholds)
    floors = th.FloorPlan.build(
        species="human", platform="XENIUM", hard_floor=10, trust=None
    )
    settings = cs.HumanResolveSettings.from_config(
        config,
        platform="XENIUM",
        emission=emission,
        floors=floors,
        trust=make_trust("validated_real"),
    )
    assert settings.allow_single_method and settings.min_counts == 10
    assert settings.region == "frontal_cortex"
    with pytest.raises(ValueError, match="human"):
        cs.HumanResolveSettings.from_config(
            AnnotationConfig(species="mouse").with_min_counts(10),
            platform="XENIUM",
            emission=emission,
            floors=floors,
            trust=None,
        )
    with pytest.raises(ValueError, match="min_counts"):
        cs.resolve_human(
            cs.HumanCalls(
                total_counts=np.array([5]),
                in_table=np.array([True]),
                whb=None,
            ),
            settings,
        )
    with pytest.raises(ValueError, match="one value per object"):
        cs.HumanCalls(
            total_counts=np.array([50, 60]),
            in_table=np.array([True]),
            whb=None,
        )


def test_consensus_tier_counts_agreeing_informative_methods() -> None:
    labels = [
        np.array(["A", "A", "A", None, "A", "A"], dtype=object),
        np.array(["A", "B", None, None, "B", "A"], dtype=object),
        np.array(["A", "B", "C", "A", "C", None], dtype=object),
    ]
    informative = [np.ones(6, dtype=bool)] * 2 + [np.ones(6, dtype=bool)]
    informative = [item.copy() for item in informative]
    for item in informative:
        item[5] = False  # no method informative on the last cell
    tier = cs.consensus_tier(labels, informative, max_tier=3)
    # 0 is a confident disagreement only; no informative method is -1.
    assert tier.tolist() == [3, 2, 0, 1, 0, -1]
    assert cs.TIER_DISAGREE == 0 and cs.TIER_NONE_INFORMATIVE == -1
    assert cs.consensus_tier(labels, informative, max_tier=2).tolist() == [
        2,
        2,
        0,
        1,
        0,
        -1,
    ]
    assert cs.consensus_tier([], [], max_tier=2).tolist() == []


def test_the_tier_separates_disagreement_from_no_informative_method() -> None:
    cells = [
        # WHB broad 0.5, SEA 0.3: no method informative.
        Cell(EXC, counts=200, broad_raw=0.5, sea_raw=0.3),
        # Both informative and different: confident disagreement.
        Cell(EXC, counts=200, sea_broad="Astrocytes", sea_raw=0.95),
        # Outside the table.
        Cell(EXC, counts=5),
    ]
    result = resolve(cells)
    assert result.consensus_tier[:3].tolist() == [-1, 0, -1]
    counts = result.summary()["consensus_tier_counts"]
    assert counts["-1"] == 1 and counts["0"] == 1
    assert result.to_columns()[Columns.CT_CONSENSUS_TIER].dtype == np.int8


def test_published_table_cells_below_min_counts_are_below_floor(
    make_trust: MakeTrust,
) -> None:
    import dataclasses

    cells = [Cell(EXC, counts=200), Cell(EXC, counts=6), Cell(ASTRO, counts=3)]
    calls = calls_of(cells)
    # A min_cells-filtered published table: every object is a table cell.
    published = dataclasses.replace(calls, in_table=np.ones(3, dtype=bool))
    settings = Setup(trust=make_trust("validated_real")).settings()
    with pytest.raises(ValueError, match="min_counts"):
        cs.resolve_human(published, settings)
    allowed = dataclasses.replace(settings, allow_table_below_min_counts=True)
    result = cs.resolve_human(published, allowed)
    for level in ("lineage", "broad", "supercluster"):
        status = result.status(level)
        assert status[0] == CellStatus.CONFIDENT.value
        assert status[1] in (
            CellStatus.BELOW_FLOOR.value,
            CellStatus.PARENT_UNRESOLVED.value,
        )
    assert result.status("lineage")[1] == CellStatus.BELOW_FLOOR.value
    assert result.status("lineage")[2] == CellStatus.BELOW_FLOOR.value
    assert result.in_table.all()
    assert not result.flags[Columns.FLAG_LOW_COUNTS].any()
