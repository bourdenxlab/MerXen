"""Tests for the real-data QC outcome model and its combinators (M13 C12).

Plan §8.8 and §12 M13: real-data QC can lower a dataset's gate level or set
a warning, and never raises a trust state, changes an emission or floor plan
or removes a margin; the confident set only shrinks and its labels are
unchanged.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import json
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pytest

from merxen.annotation import consensus as cs
from merxen.annotation import real_qc as qc
from merxen.annotation import thresholds as th
from merxen.annotation.config import (
    AnnotationGate,
    AnnotationThresholds,
    MouseGateConfig,
)
from merxen.annotation.mouse_gate import (
    MarkerReferee,
    MouseGateSignals,
    RegistrationSignal,
    evaluate_mouse_gate,
)
from merxen.annotation.provenance import RealQcProvenance
from merxen.annotation.resolvability import LevelMeta
from merxen.annotation.schema import GATE_LEVELS, PANEL_TRUST_STATES, CellStatus
from merxen.annotation.thresholds import GateVerdict

from . import test_consensus as human
from . import test_consensus_mouse as mouse
from .conftest import HUMAN_GRID, HUMAN_TABLE_CLASSES

MakeTrust = Callable[..., Any]
MakeDecisions = Callable[..., Any]
CONFIDENT = CellStatus.CONFIDENT.value


def outcome(effect: str, *, fired: bool = True, **fields: Any) -> qc.QcOutcome:
    """Return a test outcome of ``effect``."""
    return qc.QcOutcome("check_x", fired, effect, message=f"{effect} m", **fields)  # type: ignore[arg-type]


# Every kind of outcome a check can produce (the property tests combine them).
CATALOGUE: tuple[qc.QcOutcome, ...] = (
    outcome("warning", fired=False),
    outcome("warning"),
    outcome("report_only"),
    outcome("gate_cap", gate_cap="broad_only"),
    outcome("gate_cap", gate_cap="failed"),
    outcome("gate_cap", fired=False, gate_cap="failed"),
    outcome("withhold_level", level="broad"),
    outcome("withhold_level", level="nt"),
    outcome("withhold_level", level="supercluster"),
    outcome("withhold_pair_stats"),
    outcome("downgrade", trust_cap="broad_only"),
    outcome("downgrade", trust_cap="refused"),
    qc.QcOutcome.not_applicable("check_y", "unpaired", effect="withhold_pair_stats"),
    qc.QcOutcome.not_evaluable(
        "check_z", "no input", effect="gate_cap", gate_cap="broad_only"
    ),
)


def combinations() -> Iterable[tuple[qc.QcOutcome, ...]]:
    """Yield every combination of catalogue outcomes (the power set)."""
    for size in range(len(CATALOGUE) + 1):
        yield from itertools.combinations(CATALOGUE, size)


def human_gate(level: str = "full", *, warning: bool = False) -> GateVerdict:
    return GateVerdict(
        level=level,  # type: ignore[arg-type]
        warning=warning,
        level_reasons=() if level == "full" else ("frac_ge30: A = 0.2 < 0.3",),
        warning_reasons=("panel_provisional: x",) if warning else (),
        frac_ge30=0.5,
        broad_coverage_table=0.6,
        broad_coverage_segmented=0.5,
        n_table=100,
    )


def mouse_gate(**change: Any) -> Any:
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


# --------------------------------------------------------------------------
# The outcome model


@pytest.mark.parametrize(
    ("item", "token"),
    [
        (outcome("warning", fired=False), "pass"),
        (outcome("gate_cap", fired=False, gate_cap="broad_only"), "pass"),
        (outcome("warning"), "warn"),
        (outcome("report_only"), "warn"),
        (outcome("gate_cap", gate_cap="broad_only"), "fail"),
        (outcome("withhold_level", level="broad"), "fail"),
        (outcome("withhold_pair_stats"), "fail"),
        (outcome("downgrade", trust_cap="broad_only"), "fail"),
        (qc.QcOutcome.not_applicable("c", "unpaired"), "not_applicable"),
        (qc.QcOutcome.not_evaluable("c", "no input"), "not_evaluable"),
    ],
)
def test_each_outcome_maps_to_its_provenance_token(
    item: qc.QcOutcome, token: str
) -> None:
    assert item.outcome == token
    assert item.lowers == (token == "fail")
    # The token is a value RealQcProvenance accepts.
    RealQcProvenance(outcomes={"c": token})  # type: ignore[dict-item]


def test_the_outcome_model_rejects_inconsistent_outcomes() -> None:
    with pytest.raises(ValueError, match="needs a gate cap"):
        qc.QcOutcome("c", True, "gate_cap")
    with pytest.raises(ValueError, match="needs a gate cap"):
        qc.QcOutcome("c", True, "gate_cap", gate_cap="full")
    with pytest.raises(ValueError, match="has no gate cap"):
        qc.QcOutcome("c", True, "warning", gate_cap="broad_only")
    with pytest.raises(ValueError, match="has no trust cap"):
        qc.QcOutcome("c", True, "gate_cap", gate_cap="failed", trust_cap="refused")
    with pytest.raises(ValueError, match="needs its level"):
        qc.QcOutcome("c", True, "withhold_level")
    with pytest.raises(ValueError, match="unknown QC effect"):
        qc.QcOutcome("c", True, "promote")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown QC state"):
        qc.QcOutcome("c", False, "warning", state="skipped")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot fire"):
        qc.QcOutcome("c", True, "warning", state="not_evaluable", reason="x")
    with pytest.raises(ValueError, match="needs a reason"):
        qc.QcOutcome("c", False, "warning", state="not_applicable")
    # A not-evaluated withhold_level outcome concerns no single level.
    assert qc.QcOutcome.not_applicable("c", "no prefilter", effect="withhold_level")


def test_warn_only_demotes_a_lowering_outcome_and_records_what_it_withheld() -> None:
    capped = outcome("gate_cap", gate_cap="broad_only", details={"value": 0.6})
    demoted = capped.warn_only("seeded family")
    assert demoted.effect == "warning" and demoted.fired
    assert demoted.gate_cap is None and demoted.outcome == "warn"
    assert demoted.message == "seeded family: gate_cap m"
    assert demoted.details == {
        "value": 0.6,
        "warn_only_effect": "gate_cap",
        "warn_only_gate_cap": "broad_only",
    }
    downgrade = outcome("downgrade", trust_cap="refused").warn_only("n")
    assert downgrade.trust_cap is None
    assert downgrade.details["warn_only_trust_cap"] == "refused"
    for item in (
        outcome("warning"),
        outcome("gate_cap", fired=False, gate_cap="failed"),
    ):
        assert item.warn_only("n") is item


def test_outcome_records_are_json_safe() -> None:
    for item in CATALOGUE:
        record = item.to_json()
        assert json.loads(json.dumps(record)) == record
        assert record["outcome"] == item.outcome and record["state"] == item.state
    assert qc.qc_summary(CATALOGUE)["promotes"] is False


# --------------------------------------------------------------------------
# Effects and the gate


def test_qc_effects_keep_the_lowest_caps_and_name_every_effect() -> None:
    effects = qc.qc_effects(
        [
            outcome("gate_cap", gate_cap="broad_only"),
            outcome("gate_cap", gate_cap="failed"),
            outcome("gate_cap", fired=False, gate_cap="failed"),
            outcome("downgrade", trust_cap="provisional"),
            outcome("downgrade", trust_cap="broad_only"),
            outcome("withhold_level", level="supercluster"),
            outcome("withhold_level", level="supercluster"),
            outcome("withhold_pair_stats"),
            outcome("warning"),
            outcome("report_only"),
            qc.QcOutcome.not_evaluable("c", "x", effect="gate_cap", gate_cap="failed"),
        ]
    )
    assert effects.gate_cap == "failed"
    assert effects.trust_cap == "broad_only"
    assert effects.withheld_levels == ("supercluster",)
    assert effects.withhold_pair_stats and effects.lowers
    assert len(effects.level_reasons) == 2
    assert all(item.startswith("real_qc_check_x") for item in effects.level_reasons)
    # The warning, both downgrades, both withheld levels and the pair stats.
    assert len(effects.warning_reasons) == 6
    assert "real_qc_check_x[supercluster]: withhold_level m" in effects.warning_reasons
    assert effects.downgrades == (
        "broad_only:check_x",
        "failed:check_x",
        "trust_provisional:check_x",
        "trust_broad_only:check_x",
        "not_resolvable_supercluster:check_x",
        "withhold_pair_stats:check_x",
    )
    nothing = qc.qc_effects([outcome("report_only"), outcome("warning", fired=False)])
    assert nothing == qc.QcEffects() and not nothing.lowers
    assert json.loads(json.dumps(effects.to_json())) == effects.to_json()


def test_apply_qc_to_gate_lowers_the_level_and_adds_reasons() -> None:
    gate = human_gate("full")
    capped = qc.apply_qc_to_gate(gate, [outcome("gate_cap", gate_cap="broad_only")])
    assert capped.level == "broad_only" and not capped.warning
    assert capped.level_reasons == ("real_qc_check_x: gate_cap m",)
    assert not capped.attempts_leaf
    warned = qc.apply_qc_to_gate(gate, [outcome("warning")])
    assert warned.level == "full" and warned.warning
    assert warned.warning_reasons == ("real_qc_check_x: warning m",)
    # A cap above the verdict's level keeps the level but records the reason.
    failed = qc.apply_qc_to_gate(
        human_gate("failed"), [outcome("gate_cap", gate_cap="broad_only")]
    )
    assert failed.level == "failed" and len(failed.level_reasons) == 2
    # Nothing to add: the verdict itself.
    assert qc.apply_qc_to_gate(gate, [outcome("report_only")]) is gate


def test_apply_qc_to_gate_works_on_the_mouse_verdict() -> None:
    verdict = mouse_gate()
    assert verdict.level == "full"
    capped = qc.apply_qc_to_gate(verdict, [outcome("gate_cap", gate_cap="failed")])
    assert capped.level == "failed" and type(capped) is type(verdict)
    assert capped.signal_status == verdict.signal_status
    assert capped.level_reasons[-1] == "real_qc_check_x: gate_cap m"


def test_apply_qc_to_gate_and_trust_never_raise_over_every_combination() -> None:
    gates = [
        human_gate(level, warning=warning)
        for level in GATE_LEVELS
        for warning in (False, True)
    ]
    for chosen in combinations():
        for warn_only in (False, True):
            items = [item.warn_only("w") if warn_only else item for item in chosen]
            effects = qc.qc_effects(items)
            for gate in gates:
                after = qc.apply_qc_to_gate(gate, effects)
                assert qc.gate_severity(after.level) >= qc.gate_severity(gate.level)
                assert after.warning >= gate.warning
                assert after.level_reasons[: len(gate.level_reasons)] == (
                    gate.level_reasons
                )
                if warn_only:
                    assert after.level == gate.level
            for state in PANEL_TRUST_STATES:
                after_state = qc.apply_qc_outcomes(state, items)
                assert qc.trust_rank(after_state) <= qc.trust_rank(state)
                if warn_only:
                    assert after_state == state


# --------------------------------------------------------------------------
# Statuses after QC, against RESOLVE itself


def human_cells() -> list[human.Cell]:
    """Cells confident at every human level, plus thin and failing ones."""
    return [
        human.Cell(human.EXC, counts=200),
        human.Cell(human.INH, counts=150),
        human.Cell(human.ASTRO, counts=200),
        human.Cell(human.MICRO, counts=150),
        human.Cell(human.OPC_SUPC, counts=200),
        human.Cell("Oligodendrocyte", counts=250),
        human.Cell(human.EXC, counts=20),
        human.Cell(human.ASTRO, counts=25),
        human.Cell(human.EXC, counts=200, broad_raw=0.5),
        human.Cell(human.ASTRO, counts=5),
        human.Cell(human.EXC, counts=100, sea_broad="Astrocytes"),
    ]


def statuses_of(result: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {level: values.status.copy() for level, values in result.levels.items()},
        {level: values.name.copy() for level, values in result.levels.items()},
    )


def assert_only_shrinks(
    before: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    after: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
) -> None:
    """Assert confident sets only shrink and kept labels are unchanged."""
    for level, status in before[0].items():
        was = status == CONFIDENT
        now = after[0][level] == CONFIDENT
        assert not (now & ~was).any(), level
        assert list(after[1][level][now]) == list(before[1][level][now]), level


def test_qc_never_raises_gate_trust_or_confident_labels(
    make_trust: MakeTrust,
) -> None:
    """The §12 property test over every outcome combination (human RESOLVE).

    Gate level and trust never rise, and confident sets only shrink with
    their labels kept. The emission and floor plans are not inputs of the
    combinators (``test_the_combinators_take_no_plan``); the comparison with
    a QC-free re-run of RESOLVE itself (NR1) is chunk C15's.
    """
    for trust_state in ("validated_real", "provisional"):
        trust = make_trust(trust_state)
        settings = human.Setup(trust=trust).settings()
        calls = human.calls_of([*human_cells(), *[human.FILLER] * 5])
        result = cs.resolve_human(calls, settings)
        before = statuses_of(result)
        # The statuses depend only on the gate cap and the withheld levels:
        # each distinct pair is applied once.
        applied: dict[tuple[str | None, frozenset[str]], Any] = {}
        for chosen in combinations():
            effects = qc.qc_effects(chosen)
            key = (effects.gate_cap, frozenset(effects.withheld_levels))
            if key not in applied:
                after = qc.apply_qc_to_statuses(
                    *before,
                    effects,
                    in_table=result.in_table,
                    species="human",
                    gate_level=result.gate.level,
                )
                assert_only_shrinks(before, after)
                for level in effects.withheld_levels:
                    assert not (after[0][level] == CONFIDENT).any()
                if effects.gate_cap == "failed":
                    assert not any(
                        (values == CONFIDENT).any() for values in after[0].values()
                    )
                applied[key] = after
            gate = qc.apply_qc_to_gate(result.gate, effects)
            assert qc.gate_severity(gate.level) >= qc.gate_severity(result.gate.level)
            state = qc.apply_qc_outcomes(trust.state, chosen)
            assert qc.trust_rank(state) <= qc.trust_rank(trust.state)
        assert len(applied) == 3 * 2**3


def test_the_combinators_take_no_plan() -> None:
    """No QC combinator or check receives an emission plan, floor plan or settings.

    So none can change what RESOLVE emits, its floors, thresholds or margins:
    QC reaches RESOLVE only through its outcomes (C15 wires them).
    """
    forbidden = ("EmissionPlan", "FloorPlan", "ResolveSettings", "Thresholds")
    functions = (
        qc.qc_effects,
        qc.apply_qc_to_gate,
        qc.apply_qc_outcomes,
        qc.apply_qc_to_statuses,
        qc.qc_to_provenance,
        qc.marker_consistency_outcome,
        qc.registration_g1_outcome,
        qc.paired_concordance,
        qc.flag_rate_summary,
        qc.prefilter_spotcheck,
        qc.factor_remeasure_outcome,
        qc.real_data_qc,
    )
    for function in functions:
        for name, parameter in inspect.signature(function).parameters.items():
            annotation = str(parameter.annotation)
            assert not any(item in annotation for item in forbidden), (
                function.__name__,
                name,
            )
    fields = {
        item.name: str(item.type) for item in dataclasses.fields(qc.RealQcSignals)
    }
    assert not any(
        item in annotation for annotation in fields.values() for item in forbidden
    )


@pytest.mark.parametrize(
    ("gate", "cap"),
    [
        (AnnotationGate(min_frac_ge30=1.0), "broad_only"),
        (AnnotationGate(min_table_broad_coverage=1.0), "failed"),
    ],
)
def test_a_gate_cap_gives_the_statuses_of_resolve_at_that_gate(
    make_trust: MakeTrust, gate: AnnotationGate, cap: str
) -> None:
    settings = human.Setup(trust=make_trust("validated_real")).settings()
    calls = human.calls_of(human_cells())
    free = cs.resolve_human(calls, settings)
    gated = cs.resolve_human(calls, dataclasses.replace(settings, gate=gate))
    assert free.gate.level == "full" and gated.gate.level == cap
    expected = statuses_of(gated)
    after = qc.apply_qc_to_statuses(
        *statuses_of(free),
        [outcome("gate_cap", gate_cap=cap)],
        in_table=free.in_table,
        species="human",
    )
    for level, values in expected[0].items():
        assert list(after[0][level]) == list(values), level
        assert list(after[1][level]) == list(expected[1][level]), level


def test_a_withheld_level_gives_the_confident_sets_of_resolve(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    trust = make_trust("validated_real")
    calls = human.calls_of(human_cells())
    free_settings = human.Setup(
        trust=trust, decisions=make_decisions(), meta=human_level_meta
    ).settings()
    # Not broad: with no confident broad call RESOLVE's dataset gate fails
    # (its broad coverage), the gate interaction apply_qc_to_statuses leaves
    # to its caller.
    for withheld in ("nt", "supercluster"):
        overrides = {
            (regime, withheld, cls, depth): {"status": "not_resolvable"}
            for regime in ("validated", "provisional")
            for cls in HUMAN_TABLE_CLASSES
            for depth in HUMAN_GRID
        }
        # Only the emission changes: QC never changes the floors (FloorPlan
        # reads simulated floors from the decisions, so keep the free run's).
        held_settings = dataclasses.replace(
            human.Setup(
                trust=trust,
                decisions=make_decisions(overrides=overrides),
                meta=human_level_meta,
            ).settings(),
            floors=free_settings.floors,
        )
        free = cs.resolve_human(calls, free_settings)
        held = cs.resolve_human(calls, held_settings)
        assert (free.levels[withheld].status == CONFIDENT).any()
        after = qc.apply_qc_to_statuses(
            *statuses_of(free),
            [outcome("withhold_level", level=withheld)],
            in_table=free.in_table,
            species="human",
        )
        for level, result in held.levels.items():
            expected = result.status == CONFIDENT
            assert list(after[0][level] == CONFIDENT) == list(expected), (
                withheld,
                level,
            )
        assert set(after[0][withheld][free.levels[withheld].confident]) == {
            CellStatus.NOT_RESOLVABLE.value
        }


@pytest.mark.parametrize("cap", ["broad_only", "failed"])
def test_a_gate_cap_gives_the_statuses_of_mouse_resolve(
    make_trust: MakeTrust, cap: str
) -> None:
    cells = [mouse.Cell(), mouse.ASTRO, mouse.Cell(counts=5)]
    verdict = mouse_gate()
    free = mouse.resolve(cells, make_trust, verdict=verdict)
    capped = qc.apply_qc_to_gate(verdict, [outcome("gate_cap", gate_cap=cap)])
    gated = mouse.resolve(cells, make_trust, verdict=capped)
    expected = statuses_of(gated)
    after = qc.apply_qc_to_statuses(
        *statuses_of(free),
        [outcome("gate_cap", gate_cap=cap)],
        in_table=free.in_table,
        species="mouse",
    )
    for level, values in expected[0].items():
        assert list(after[0][level]) == list(values), level
        assert list(after[1][level]) == list(expected[1][level]), level
    assert_only_shrinks(statuses_of(free), after)


class _WithheldEmission:
    """An emission plan that emits nothing at one level (a withheld level).

    Everything else is the wrapped plan's, so RESOLVE run with it is RESOLVE
    with that level made ``not_resolvable`` and nothing else changed.
    """

    def __init__(self, plan: Any, withheld: str) -> None:
        self._plan = plan
        self._withheld = withheld

    def __getattr__(self, name: str) -> Any:
        return getattr(self._plan, name)

    def level(self, level: str, key: Any, counts: Any) -> Any:
        emission = self._plan.level(level, key, counts)
        if level != self._withheld:
            return emission
        return dataclasses.replace(
            emission, emitted=np.zeros(len(emission.emitted), dtype=bool)
        )


def mouse_resolution(
    cells: list[mouse.Cell], make_trust: MakeTrust, withheld: str | None = None
) -> Any:
    """Run ``resolve_mouse`` as ``test_consensus_mouse.resolve`` does."""
    trust = make_trust("validated_real", species="mouse")
    limits = AnnotationThresholds()
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
        emission=emission
        if withheld is None
        else _WithheldEmission(emission, withheld),
        floors=floors,
        thresholds=limits,
        trust=trust,
    )
    return cs.resolve_mouse(mouse.calls_of(cells), settings, mouse_gate())


MOUSE_CELLS: list[mouse.Cell] = [
    mouse.Cell(),
    mouse.ASTRO,
    mouse.Cell(counts=5),
    mouse.Cell(class_bp=0.89),
    mouse.Cell(subclass_bp=0.5),
    mouse.Cell(nt_bp=0.5),
]


@pytest.mark.parametrize("withheld", ["broad", "class", "nt", "subclass"])
def test_a_withheld_level_gives_the_confident_sets_of_mouse_resolve(
    make_trust: MakeTrust, withheld: str
) -> None:
    """Mouse parents: a neuron's subclass hangs off its NT, an astrocyte's off class."""
    free = mouse_resolution(MOUSE_CELLS, make_trust)
    held = mouse_resolution(MOUSE_CELLS, make_trust, withheld)
    assert (free.levels[withheld].status == CONFIDENT).any()
    after = qc.apply_qc_to_statuses(
        *statuses_of(free),
        [outcome("withhold_level", level=withheld)],
        in_table=free.in_table,
        species="mouse",
    )
    for level, result in held.levels.items():
        assert list(after[0][level] == CONFIDENT) == list(result.status == CONFIDENT), (
            withheld,
            level,
        )
        # Where RESOLVE itself withholds or loses the parent, the status is the
        # one RESOLVE writes.
        lost = (free.levels[level].status == CONFIDENT) & (result.status != CONFIDENT)
        assert list(after[0][level][lost]) == list(result.status[lost]), (
            withheld,
            level,
        )
    assert_only_shrinks(statuses_of(free), after)
    if withheld == "nt":
        # The neuron loses its subclass through NT; the astrocyte (NT not
        # applicable) keeps its subclass through class.
        assert after[0]["subclass"][0] == CellStatus.PARENT_UNRESOLVED.value
        assert after[0]["subclass"][1] == CONFIDENT
    if withheld == "class":
        assert after[0]["subclass"][1] == CellStatus.PARENT_UNRESOLVED.value


def test_a_withheld_lineage_unresolves_every_level_below_it() -> None:
    """Human: broad hangs off lineage; a non-neuron's leaf off broad."""
    pu = CellStatus.PARENT_UNRESOLVED.value
    na = CellStatus.NOT_APPLICABLE.value
    low = CellStatus.LOW_CONFIDENCE.value
    levels = ("lineage", "broad", "nt", "supercluster", "seaad_subclass", "cluster")
    # A neuron confident everywhere, a non-neuron (NT not applicable) and a
    # cell whose lineage is not confident.
    rows = {
        "lineage": [CONFIDENT, CONFIDENT, low],
        "broad": [CONFIDENT, CONFIDENT, pu],
        "nt": [CONFIDENT, na, pu],
        "supercluster": [CONFIDENT, CONFIDENT, pu],
        "seaad_subclass": [CONFIDENT, CONFIDENT, pu],
        "cluster": [CONFIDENT, CONFIDENT, pu],
    }
    statuses = {level: np.array(rows[level], dtype=object) for level in levels}
    names = {
        level: np.array([f"{level}_a", f"{level}_b", None], dtype=object)
        for level in levels
    }
    after, after_names = qc.apply_qc_to_statuses(
        statuses,
        names,
        [outcome("withhold_level", level="lineage")],
        in_table=[True, True, True],
        species="human",
    )
    assert list(after["lineage"]) == [
        CellStatus.NOT_RESOLVABLE.value,
        CellStatus.NOT_RESOLVABLE.value,
        low,
    ]
    for level in levels[1:]:
        expected = [pu, na if level == "nt" else pu, pu]
        assert list(after[level]) == expected, level
        assert list(after_names[level]) == list(names[level]), level
    # Withholding NT leaves the non-neuron's supercluster (its parent is broad).
    nt_only, _ = qc.apply_qc_to_statuses(
        statuses,
        names,
        [outcome("withhold_level", level="nt")],
        in_table=[True, True, True],
        species="human",
    )
    assert list(nt_only["broad"]) == [CONFIDENT, CONFIDENT, pu]
    assert list(nt_only["supercluster"]) == [pu, CONFIDENT, pu]
    assert list(nt_only["cluster"]) == [pu, CONFIDENT, pu]
    assert list(nt_only["seaad_subclass"]) == [CONFIDENT, CONFIDENT, pu]


def test_apply_qc_to_statuses_rejects_bad_inputs() -> None:
    status = {"broad": np.array([CONFIDENT])}
    with pytest.raises(ValueError, match="unknown species"):
        qc.apply_qc_to_statuses(status, status, [], in_table=[True], species="rat")
    with pytest.raises(ValueError, match="no names"):
        qc.apply_qc_to_statuses(status, {}, [], in_table=[True], species="human")
    with pytest.raises(ValueError, match="differs in length"):
        qc.apply_qc_to_statuses(
            status, status, [], in_table=[True, False], species="human"
        )


# --------------------------------------------------------------------------
# Provenance


def test_worst_outcome_order() -> None:
    assert qc.worst_outcome(["pass", "warn"]) == "warn"
    assert qc.worst_outcome(["warn", "fail", "pass"]) == "fail"
    assert qc.worst_outcome(["pass", "not_evaluable"]) == "not_evaluable"
    assert qc.worst_outcome(["not_applicable", "pass"]) == "pass"
    assert qc.worst_outcome(["not_applicable"]) == "not_applicable"
    with pytest.raises(ValueError, match="unknown"):
        qc.worst_outcome(["skipped"])
    with pytest.raises(ValueError, match="no QC outcome"):
        qc.worst_outcome([])


def test_qc_to_provenance_records_each_check_and_the_downgrades() -> None:
    outcomes = [
        qc.QcOutcome("coverage_vs_simulation", True, "warning", level="class"),
        qc.QcOutcome("coverage_vs_simulation", False, "warning"),
        qc.QcOutcome(
            "marker_consistency", True, "gate_cap", gate_cap="broad_only", message="m"
        ),
        qc.QcOutcome.not_applicable(
            "paired_concordance", "unpaired", effect="withhold_pair_stats"
        ),
        qc.QcOutcome.not_evaluable("gene_complexity", "no simulated n_genes"),
    ]
    record = qc.qc_to_provenance(outcomes, warn_only=False, gate=human_gate("full"))
    assert record.outcomes == {
        "coverage_vs_simulation": "warn",
        "marker_consistency": "fail",
        "paired_concordance": "not_applicable",
        "gene_complexity": "not_evaluable",
        "dataset_gate": "pass",
    }
    assert record.downgrades == ["broad_only:marker_consistency"]
    assert record.warn_only is False
    # The gate's own outcome is recorded from the verdict, never re-applied.
    assert qc.gate_outcomes(human_gate("broad_only")) == {"dataset_gate": "fail"}
    assert qc.gate_outcomes(human_gate("full", warning=True)) == {
        "dataset_gate": "warn"
    }
    with pytest.raises(ValueError, match="safe provenance key"):
        qc.qc_to_provenance([qc.QcOutcome("bad name", False, "warning")])


def test_the_mouse_gate_records_g1_and_g2_as_its_own_checks() -> None:
    verdict = mouse_gate(
        referee=MarkerReferee(
            consistency=0.75, n_pseudo_confident=500, n_cells=600, markers={}
        )
    )
    assert qc.gate_outcomes(verdict) == {
        "dataset_gate": "warn",
        "registration_g1": "pass",
        "marker_consistency": "warn",
    }
    failed = mouse_gate(
        registration=RegistrationSignal(density_ratio=1.2, shift_um=0.0),
        referee=None,
    )
    assert qc.gate_outcomes(failed) == {
        "dataset_gate": "fail",
        "registration_g1": "fail",
        "marker_consistency": "not_evaluable",
    }
