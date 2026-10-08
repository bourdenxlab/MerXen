"""Tests for the real-data QC wiring into RESOLVE and the report (M13 C15).

Plan §8.8 and pre-registration §23.5 P4 / §23.6 NR1: RESOLVE records an
outcome for every §8.8 check per dataset, applies the lowering effects in the
pass (a gate cap before the leaf levels read the gate, a withheld level not
emitted for the dataset), keeps the gate-level invariance, and changes
nothing a QC-free re-run decides except labels set to ``not_resolvable`` and
the gate level lowered by a named check.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import consensus as cs
from merxen.annotation import pipeline as pl
from merxen.annotation import real_qc as qc
from merxen.annotation.config import AnnotationConfig, AnnotationGate
from merxen.annotation.pipeline import read_label_table
from merxen.annotation.resolvability import LevelMeta
from merxen.annotation.schema import CellStatus, Columns

from . import test_consensus as human
from .conftest import FakeMmc, assert_same_emission
from .test_pipeline import GENE_IDS
from .test_pipeline_resolve import MARKER, Setup, _resolve, _setup
from .test_real_qc_outcomes import human_cells, statuses_of

MakeTrust = Callable[..., Any]
MakeDecisions = Callable[..., Any]
CONFIDENT = CellStatus.CONFIDENT.value


def outcome(
    effect: str,
    *,
    check: str = "check_x",
    message: str | None = None,
    fired: bool = True,
    **fields: Any,
) -> qc.QcOutcome:
    """Return a fired test outcome of ``effect`` (of ``check``)."""
    return qc.QcOutcome(
        check,
        fired,
        effect,  # type: ignore[arg-type]
        message=f"{effect} m" if message is None else message,
        **fields,
    )


# --------------------------------------------------------------------------
# In-pass application in ``resolve_human``


def _free_and_applied(
    settings: cs.HumanResolveSettings,
    outcomes: list[qc.QcOutcome],
    calls: cs.HumanCalls | None = None,
) -> tuple[cs.HumanResolution, cs.HumanResolution, qc.QcEffects]:
    calls = calls if calls is not None else human.calls_of(human_cells())
    effects = qc.qc_effects(outcomes)
    free = cs.resolve_human(calls, settings)
    applied = cs.resolve_human(calls, dataclasses.replace(settings, qc=effects))
    return free, applied, effects


@pytest.mark.parametrize("cap", ["broad_only", "failed"])
def test_a_qc_gate_cap_is_applied_in_the_pass(make_trust: MakeTrust, cap: str) -> None:
    """The cap lowers the gate before the leaf levels read it (NR1).

    The statuses are those ``apply_qc_to_statuses`` predicts, the gate names
    the check, and the emission records are the QC-free run's.
    """
    settings = human.Setup(trust=make_trust("provisional")).settings()
    free, applied, effects = _free_and_applied(
        settings,
        [
            outcome(
                "gate_cap",
                gate_cap=cap,
                check="marker_consistency",
                message="consistency 0.5",
            )
        ],
    )
    assert free.gate.level == "full"
    assert applied.gate.level == cap
    assert any(
        reason.startswith("real_qc_marker_consistency")
        for reason in applied.gate.level_reasons
    )
    expected = qc.apply_qc_to_statuses(
        *statuses_of(free), effects, in_table=free.in_table, species="human"
    )
    for level, values in expected[0].items():
        assert list(applied.levels[level].status) == list(values), level
        assert list(applied.levels[level].name) == list(expected[1][level]), level
    for level, emission in free.emissions.items():
        assert np.array_equal(emission.emitted, applied.emissions[level].emitted)
        assert np.array_equal(
            emission.threshold, applied.emissions[level].threshold, equal_nan=True
        )
    if cap == "failed":
        assert applied.flags[Columns.EXCLUDE_HARD][applied.in_table].all()
    else:
        # broad_only: broad and its parents are decided as without QC.
        for level in ("lineage", "broad", "nt"):
            assert list(applied.levels[level].status) == list(free.levels[level].status)


def test_a_qc_gate_cap_never_raises_the_gate(make_trust: MakeTrust) -> None:
    settings = dataclasses.replace(
        human.Setup(trust=make_trust("provisional")).settings(),
        gate=AnnotationGate(min_table_broad_coverage=1.0),
    )
    free, applied, _ = _free_and_applied(
        settings, [outcome("gate_cap", gate_cap="broad_only")]
    )
    assert free.gate.level == applied.gate.level == "failed"


def test_qc_warnings_change_only_the_gate_warning(make_trust: MakeTrust) -> None:
    settings = human.Setup(trust=make_trust("validated_real")).settings()
    free, applied, _ = _free_and_applied(
        settings,
        [
            outcome("warning", check="flag_rates", message="most strata"),
            outcome("report_only", check="nonneuronal_depth_trend"),
        ],
    )
    assert not free.gate.warning
    assert applied.gate.warning
    assert applied.gate.warning_reasons == ("real_qc_flag_rates: most strata",)
    # Everything RESOLVE decides is the QC-free run's (the gate warning and
    # its reasons are the only differences assert_same_emission allows).
    assert_same_emission(free, applied)


def test_a_withheld_level_is_not_emitted_for_the_dataset(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    """A withheld level's confident cells become not_resolvable in the pass.

    Its derived level (SEA-AD subclass reads the supercluster table) goes
    with it, the confident sets are those ``apply_qc_to_statuses`` predicts,
    and the emission records stay the QC-free run's.
    """
    settings = human.Setup(
        trust=make_trust("validated_real"),
        decisions=make_decisions(),
        meta=human_level_meta,
    ).settings()
    for withheld in ("nt", "supercluster"):
        free, applied, effects = _free_and_applied(
            settings,
            [outcome("withhold_level", level=withheld, check="prefilter_spotcheck")],
        )
        assert free.levels[withheld].confident.any()
        assert not applied.levels[withheld].confident.any()
        assert set(
            applied.levels[withheld].status[free.levels[withheld].confident]
        ) == {CellStatus.NOT_RESOLVABLE.value}
        expected = qc.apply_qc_to_statuses(
            *statuses_of(free), effects, in_table=free.in_table, species="human"
        )
        for level, values in expected[0].items():
            assert list(applied.levels[level].status == CONFIDENT) == list(
                values == CONFIDENT
            ), (withheld, level)
        if withheld == "supercluster":
            assert not applied.levels["seaad_subclass"].confident.any()
        for level, emission in free.emissions.items():
            assert np.array_equal(emission.emitted, applied.emissions[level].emitted)
        assert applied.gate.level == free.gate.level
        assert any(
            reason.startswith("real_qc_prefilter_spotcheck")
            for reason in applied.gate.warning_reasons
        )


def test_a_level_and_its_parent_withheld_give_the_statuses_of_the_pass(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    """NT and supercluster withheld (with and without a gate cap).

    RESOLVE checks the parent before the emission: a neuron's confident
    supercluster, whose NT is withheld too, becomes ``parent_unresolved``;
    a non-neuron's (its parent is broad) and the SEA-AD subclass (it reads
    the supercluster table, its parent is broad) ``not_resolvable``. Every
    cell the QC-free run made confident takes the status of the pass.
    """
    rng = np.random.default_rng(5)
    calls = human.calls_of([*human.random_cells(rng, 120), *human_cells()])
    settings = human.Setup(
        trust=make_trust("validated_real"),
        decisions=make_decisions(),
        meta=human_level_meta,
    ).settings()
    held = [
        outcome("withhold_level", level=level, check="prefilter_spotcheck")
        for level in ("nt", "supercluster")
    ]
    for cap in (None, "broad_only"):
        capped = [outcome("gate_cap", gate_cap=cap)] if cap is not None else []
        free, applied, effects = _free_and_applied(settings, [*held, *capped], calls)
        expected = qc.apply_qc_to_statuses(
            *statuses_of(free), effects, in_table=free.in_table, species="human"
        )
        for level, values in expected[0].items():
            was = free.levels[level].confident
            assert list(values[was]) == list(applied.levels[level].status[was]), (
                cap,
                level,
            )
        if cap is not None:
            continue
        neuron = free.levels["nt"].confident & free.levels["supercluster"].confident
        other = (
            free.levels["nt"].status == CellStatus.NOT_APPLICABLE.value
        ) & free.levels["supercluster"].confident
        assert neuron.any() and other.any()
        status = applied.levels["supercluster"].status
        assert set(status[neuron]) == {CellStatus.PARENT_UNRESOLVED.value}
        assert set(status[other]) == {CellStatus.NOT_RESOLVABLE.value}
        sea = free.levels["seaad_subclass"].confident
        assert set(applied.levels["seaad_subclass"].status[sea]) == {
            CellStatus.NOT_RESOLVABLE.value
        }


def test_every_effect_combination_only_lowers_in_the_pass(
    make_trust: MakeTrust,
    make_decisions: MakeDecisions,
    human_level_meta: list[LevelMeta],
) -> None:
    """The §12 property in RESOLVE itself: over every combination of effects.

    Gate caps (none, broad_only, failed) x withheld levels (none, nt,
    supercluster, both) x a warning: the gate never rises, the confident sets
    only shrink and keep their names, the gate level is the worse of the
    QC-free level and the cap, and the emission records never change.
    """
    rng = np.random.default_rng(3)
    calls = human.calls_of([*human.random_cells(rng, 120), *[human.FILLER] * 40])
    for trust_state in ("validated_real", "provisional"):
        settings = human.Setup(
            trust=make_trust(trust_state),
            decisions=make_decisions(),
            meta=human_level_meta,
        ).settings()
        free = cs.resolve_human(calls, settings)
        before = statuses_of(free)
        caps: tuple[str | None, ...] = (None, "broad_only", "failed")
        held_sets: tuple[tuple[str, ...], ...] = (
            (),
            ("nt",),
            ("supercluster",),
            ("nt", "supercluster"),
        )
        for cap, held, warn in itertools.product(caps, held_sets, (False, True)):
            outcomes = [
                *([outcome("gate_cap", gate_cap=cap)] if cap else []),
                *[outcome("withhold_level", level=level) for level in held],
                *([outcome("warning")] if warn else []),
            ]
            effects = qc.qc_effects(outcomes)
            applied = cs.resolve_human(calls, dataclasses.replace(settings, qc=effects))
            for level, status in before[0].items():
                was = status == CONFIDENT
                now = applied.levels[level].status == CONFIDENT
                assert not (now & ~was).any(), (cap, held, level)
                assert list(applied.levels[level].name[now]) == list(
                    before[1][level][now]
                ), (cap, held, level)
            expected_level = free.gate.level
            if cap is not None and qc.gate_severity(cap) > qc.gate_severity(
                expected_level
            ):
                expected_level = cap
            assert applied.gate.level == expected_level, (cap, held)
            assert applied.gate.warning or not (warn or held)
            for level, emission in free.emissions.items():
                assert np.array_equal(
                    emission.emitted, applied.emissions[level].emitted
                ), (cap, held, level)


def test_resolve_human_refuses_a_trust_downgrade(make_trust: MakeTrust) -> None:
    settings = human.Setup(trust=make_trust("provisional")).settings()
    effects = qc.qc_effects([outcome("downgrade", trust_cap="broad_only")])
    with pytest.raises(ValueError, match="trust downgrade"):
        dataclasses.replace(settings, qc=effects)


# --------------------------------------------------------------------------
# real_qc helpers RESOLVE uses


def _seeded_result(warn_only: bool) -> qc.RealQcResult:
    return qc.RealQcResult(
        species="human",
        outcomes=(
            qc.QcOutcome("marker_consistency", False, "warning"),
            qc.QcOutcome.not_evaluable(
                "paired_concordance",
                "no pair JSD yet",
                effect="withhold_pair_stats",
            ),
            qc.QcOutcome("flag_rates", False, "warning"),
        ),
        seeded=warn_only,
        warn_only=warn_only,
    )


@pytest.mark.parametrize("warn_only", [False, True])
def test_replace_check_swaps_the_placeholder_and_keeps_the_demotion(
    warn_only: bool,
) -> None:
    result = _seeded_result(warn_only)
    fired = qc.QcOutcome(
        "paired_concordance", True, "withhold_pair_stats", message="JSD 0.3"
    )
    new = result.replace_check("paired_concordance", [fired])
    assert [item.check for item in new.outcomes] == [
        "marker_consistency",
        "paired_concordance",
        "flag_rates",
    ]
    (paired,) = new.outcomes_of("paired_concordance")
    if warn_only:
        assert paired.effect == "warning" and paired.fired
        assert paired.details["warn_only_effect"] == "withhold_pair_stats"
        assert not new.effects.withhold_pair_stats
    else:
        assert paired is fired
        assert new.effects.withhold_pair_stats
    assert result.outcomes_of("paired_concordance")[0].state == "not_evaluable"


def test_replace_check_refuses_other_checks_and_missing_placeholders() -> None:
    result = _seeded_result(False)
    with pytest.raises(ValueError, match="every outcome"):
        result.replace_check(
            "paired_concordance", [qc.QcOutcome("flag_rates", False, "warning")]
        )
    with pytest.raises(ValueError, match="no outcome"):
        result.replace_check("paired_concordance", [])
    with pytest.raises(ValueError, match="records no"):
        result.replace_check(
            "gene_complexity", [qc.QcOutcome("gene_complexity", False, "warning")]
        )


def test_the_trend_reads_each_level_with_its_own_class_key() -> None:
    """Per level, the trend groups cells by that level's class key (C15)."""
    n = 300
    totals = np.concatenate([np.full(n, 700.0), np.full(n, 1500.0)])
    falling = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    broad_key = np.array(["Astro"] * (2 * n), dtype=object)
    leaf_key = np.array(["Astro_leaf"] * (2 * n), dtype=object)
    signal = qc.NonneuronalTrendSignal(
        totals=totals,
        called_class={"broad": broad_key, "supercluster": leaf_key},
        confident={"broad": falling, "supercluster": np.ones(2 * n, bool)},
        nonneuronal_classes=["Astro", "Astro_leaf"],
    )
    outcomes, table = qc._trend_outcomes(signal, version=7, limit=1000)
    assert table is not None
    assert set(zip(table["level"], table["class"], strict=True)) == {
        ("broad", "Astro"),
        ("supercluster", "Astro_leaf"),
    }
    assert [(item.level, item.cls) for item in outcomes] == [("broad", "Astro")]
    with pytest.raises(ValueError, match="no class key"):
        qc._trend_outcomes(
            dataclasses.replace(signal, called_class={"broad": broad_key}),
            version=7,
            limit=1000,
        )


def _labels(statuses: dict[str, list[str]], names: dict[str, list[Any]]) -> Any:
    n = len(next(iter(statuses.values())))
    data: dict[str, Any] = {Columns.CELL_ID: [f"c{i}" for i in range(n)]}
    for level, values in statuses.items():
        data[Columns.level(level, "status")] = values
        data[Columns.level(level, "name")] = names[level]
    return pd.DataFrame(data)


def test_downgrade_only_violations_accept_named_lowering_only() -> None:
    """NR1: only not_resolvable, not_attempted_gate and parent_unresolved."""
    free = _labels(
        {
            "broad": ["confident", "confident", "low_confidence"],
            "supercluster": ["confident", "confident", "parent_unresolved"],
        },
        {"broad": ["A", "B", "C"], "supercluster": ["a", "b", "c"]},
    )
    lowered = _labels(
        {
            "broad": ["confident", "confident", "low_confidence"],
            "supercluster": [
                "not_attempted_gate",
                "not_resolvable",
                "parent_unresolved",
            ],
        },
        {"broad": ["A", "B", "C"], "supercluster": [None, "b", "c"]},
    )
    assert qc.downgrade_only_violations(free, lowered, species="human") == []
    raised = lowered.copy()
    raised[Columns.level("broad", "status")] = ["confident"] * 3
    renamed = lowered.copy()
    renamed[Columns.level("broad", "name")] = ["A", "X", "C"]
    other = lowered.copy()
    other[Columns.level("broad", "status")] = [
        "confident",
        "below_floor",
        "low_confidence",
    ]
    problems = {
        "raised": qc.downgrade_only_violations(free, raised, species="human"),
        "renamed": qc.downgrade_only_violations(free, renamed, species="human"),
        "other": qc.downgrade_only_violations(free, other, species="human"),
    }
    assert any("confident only with QC" in item for item in problems["raised"])
    assert any("changed their name" in item for item in problems["renamed"])
    assert any("below_floor" in item for item in problems["other"])
    shuffled = lowered.iloc[::-1].reset_index(drop=True)
    assert qc.downgrade_only_violations(free, shuffled, species="human") == [
        "the label tables do not hold the same cells in the same order"
    ]


def test_downgrade_only_violations_compare_trust_emission_and_gate() -> None:
    labels = _labels({"broad": ["confident"]}, {"broad": ["A"]})

    def provenance(**change: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "panel": {"panel_trust": "provisional"},
            "resolvability": {"whb": {"emitted_depth_bins": {"broad": {"a": [10]}}}},
            "thresholds": {"floors_sha256": "f" * 64},
            "gate": {"level": "full", "reasons": []},
        }
        record.update(change)
        return record

    def check(applied: dict[str, Any]) -> list[str]:
        return qc.downgrade_only_violations(
            labels,
            labels,
            species="human",
            free_provenance=provenance(),
            applied_provenance=applied,
        )

    named = {"level": "broad_only", "reasons": ["real_qc_marker_consistency: 0.5"]}
    # The provenance gate mixes level and warning reasons: a fall is only
    # attributed with the summary's gate records (``level_reasons``) or the
    # QC effects.
    assert check(provenance(gate=named)) == [
        "the gate level fell from full to broad_only, and the gate record does "
        "not separate its level reasons (pass the resolve summary's gate "
        "records or the QC effects)"
    ]
    assert (
        qc.downgrade_only_violations(
            labels,
            labels,
            species="human",
            free_provenance=provenance(),
            applied_provenance=provenance(gate=named),
            free_gate={"level": "full", "level_reasons": []},
            applied_gate={
                "level": "broad_only",
                "level_reasons": ["real_qc_marker_consistency: 0.5"],
            },
        )
        == []
    )
    assert qc.downgrade_only_violations(
        labels,
        labels,
        species="human",
        free_gate={"level": "full", "level_reasons": []},
        applied_gate={"level": "broad_only", "level_reasons": ["x"]},
    ) == [
        "the gate level fell from full to broad_only without a named real_qc "
        "check among its level reasons"
    ]
    assert check(provenance(panel={"panel_trust": "broad_only"})) == [
        "trust state 'provisional' became 'broad_only'"
    ]
    assert check(provenance(thresholds={"floors_sha256": "0" * 64})) == [
        "the thresholds record differs"
    ]
    assert check(provenance(resolvability={})) == ["the resolvability record differs"]
    free_failed = provenance(gate={"level": "failed", "reasons": []})
    assert qc.downgrade_only_violations(
        labels,
        labels,
        species="human",
        free_provenance=free_failed,
        applied_provenance=provenance(),
    ) == ["the gate level rose from failed to full"]


def _gates(free: str, applied: str, *level_reasons: str) -> dict[str, Any]:
    """Summary gate records (``GateVerdict.to_json``) of the two runs."""
    return {
        "free_gate": {"level": free, "level_reasons": [], "warning_reasons": []},
        "applied_gate": {
            "level": applied,
            "level_reasons": list(level_reasons),
            "warning_reasons": [],
        },
    }


CAP_REASON = "real_qc_marker_consistency: consistency 0.5"


def test_nr1_allows_the_gate_status_only_at_the_levels_the_gate_blocks() -> None:
    """Planted (a): a broad_only gate blocks only the leaf levels.

    ``not_attempted_gate`` at broad needs a failed gate; at a leaf level it
    needs a gate below full; without a gate record only the leaf levels may
    take it.
    """
    free = _labels(
        {"broad": [CONFIDENT] * 2, "supercluster": [CONFIDENT] * 2},
        {"broad": ["A", "B"], "supercluster": ["a", "b"]},
    )
    every_level = _labels(
        {
            "broad": ["not_attempted_gate"] * 2,
            "supercluster": ["not_attempted_gate"] * 2,
        },
        {"broad": [None, None], "supercluster": [None, None]},
    )
    leaf_only = _labels(
        {"broad": [CONFIDENT] * 2, "supercluster": ["not_attempted_gate"] * 2},
        {"broad": ["A", "B"], "supercluster": [None, None]},
    )

    def check(applied: pd.DataFrame, **gates: Any) -> list[str]:
        return qc.downgrade_only_violations(free, applied, species="human", **gates)

    assert check(every_level, **_gates("full", "broad_only", CAP_REASON)) == [
        "broad: 2 cell(s) not_attempted_gate at a level a broad_only gate does "
        "not block"
    ]
    assert check(every_level, **_gates("full", "failed", CAP_REASON)) == []
    assert check(leaf_only, **_gates("full", "broad_only", CAP_REASON)) == []
    assert check(leaf_only, **_gates("full", "full")) == [
        "supercluster: 2 cell(s) not_attempted_gate at a level a full gate does "
        "not block"
    ]
    # Without a gate record: only the leaf levels.
    assert check(leaf_only) == []
    assert check(every_level) == [
        "broad: 2 cell(s) not_attempted_gate at a level a gate of unknown level "
        "does not block"
    ]


def test_nr1_allows_parent_unresolved_only_under_a_lost_parent() -> None:
    """Planted (b): parent_unresolved needs the parent to lose confidence.

    The parent is the first of ``LEVEL_PARENTS`` that applies at the cell (a
    neuron's supercluster hangs off its NT, a non-neuron's off broad).
    """
    free = _labels(
        {
            "broad": [CONFIDENT] * 3,
            "nt": [CONFIDENT, CONFIDENT, "not_applicable"],
            "supercluster": [CONFIDENT] * 3,
        },
        {
            "broad": ["A", "A", "B"],
            "nt": ["n", "n", None],
            "supercluster": ["a", "a", "b"],
        },
    )

    def applied(nt: list[str], supercluster: list[str]) -> pd.DataFrame:
        return _labels(
            {"broad": [CONFIDENT] * 3, "nt": nt, "supercluster": supercluster},
            {
                "broad": ["A", "A", "B"],
                "nt": ["n", "n", None],
                "supercluster": ["a", "a", "b"],
            },
        )

    def check(table: pd.DataFrame) -> list[str]:
        return qc.downgrade_only_violations(free, table, species="human")

    nt = [CONFIDENT, CONFIDENT, "not_applicable"]
    # The broad parent stays confident: no reason for parent_unresolved.
    assert check(applied(nt, [CONFIDENT, CONFIDENT, "parent_unresolved"])) == [
        "supercluster: 1 cell(s) parent_unresolved while their parent kept its "
        "confidence"
    ]
    assert check(applied(nt, ["parent_unresolved", CONFIDENT, CONFIDENT])) == [
        "supercluster: 1 cell(s) parent_unresolved while their parent kept its "
        "confidence"
    ]
    # The NT parent lost its confidence at the first cell only.
    lost = ["not_resolvable", CONFIDENT, "not_applicable"]
    assert check(applied(lost, ["parent_unresolved", CONFIDENT, CONFIDENT])) == []
    assert check(
        applied(lost, ["parent_unresolved", "parent_unresolved", CONFIDENT])
    ) == [
        "supercluster: 1 cell(s) parent_unresolved while their parent kept its "
        "confidence"
    ]


def test_nr1_needs_a_named_check_among_the_gate_level_reasons() -> None:
    """Planted (c): a QC warning does not name a gate level fall.

    The gate falls for a non-QC reason (A below its limit) while the only
    ``real_qc_`` reason is a warning: a violation with the summary's gate
    records, with the provenance's mixed reasons and with the QC effects.
    """
    labels = _labels({"broad": [CONFIDENT]}, {"broad": ["A"]})
    a_low = "frac_ge30: A = 0.200 < 0.3"
    warning = "real_qc_flag_rates: most strata uninformative"
    summary_gates = {
        "free_gate": {"level": "full", "level_reasons": [], "warning_reasons": []},
        "applied_gate": {
            "level": "broad_only",
            "level_reasons": [a_low],
            "warning_reasons": [warning],
        },
    }
    assert qc.downgrade_only_violations(
        labels, labels, species="human", **summary_gates
    ) == [
        "the gate level fell from full to broad_only without a named real_qc "
        "check among its level reasons"
    ]
    mixed = {"level": "broad_only", "reasons": [a_low, warning]}
    assert qc.downgrade_only_violations(
        labels,
        labels,
        species="human",
        free_provenance={"gate": {"level": "full", "reasons": []}},
        applied_provenance={"gate": mixed},
    ) == [
        "the gate level fell from full to broad_only, and the gate record does "
        "not separate its level reasons (pass the resolve summary's gate "
        "records or the QC effects)"
    ]
    effects = qc.qc_effects([outcome("warning", check="flag_rates")])
    assert qc.downgrade_only_violations(
        _with_table(labels),
        _with_table(labels),
        species="human",
        free_provenance={"gate": {"level": "full", "reasons": []}},
        applied_provenance={"gate": mixed},
        qc=effects,
    ) == [
        "the gate level is broad_only; the QC-free level full with the QC gate "
        "cap None gives full"
    ]


def _with_table(table: pd.DataFrame) -> pd.DataFrame:
    return table.assign(**{Columns.IN_TABLE: True})


def test_nr1_with_the_qc_effects_compares_the_confident_sets() -> None:
    """With the QC effects the confident sets are ``apply_qc_to_statuses``'.

    ``not_resolvable`` is allowed only at a withheld level (and the levels
    whose emission reads its table), an effect left unapplied is a
    violation, and the summary's ``effects`` record is accepted.
    """
    free = _with_table(
        _labels(
            {"broad": [CONFIDENT] * 3, "supercluster": [CONFIDENT] * 3},
            {"broad": ["A", "A", "B"], "supercluster": ["a", "a", "b"]},
        )
    )
    withheld = qc.qc_effects(
        [outcome("withhold_level", level="supercluster", check="prefilter_spotcheck")]
    )
    applied = free.copy()
    applied[Columns.level("supercluster", "status")] = ["not_resolvable"] * 3
    assert (
        qc.downgrade_only_violations(free, applied, species="human", qc=withheld) == []
    )
    assert (
        qc.downgrade_only_violations(
            free, applied, species="human", qc=withheld.to_json()
        )
        == []
    )
    # A withheld broad level that the effects do not name.
    wrong = applied.copy()
    wrong[Columns.level("broad", "status")] = ["not_resolvable", CONFIDENT, CONFIDENT]
    wrong[Columns.level("supercluster", "status")] = [
        "parent_unresolved",
        "not_resolvable",
        "not_resolvable",
    ]
    assert qc.downgrade_only_violations(free, wrong, species="human") == []
    assert qc.downgrade_only_violations(free, wrong, species="human", qc=withheld) == [
        "broad: 1 cell(s) not_resolvable at a level the QC effects do not withhold",
        "broad: 1 cell(s) differ from the confident set the QC effects give",
    ]
    # The withheld level left confident.
    assert qc.downgrade_only_violations(free, free, species="human", qc=withheld) == [
        "supercluster: 3 cell(s) differ from the confident set the QC effects give"
    ]
    with pytest.raises(ValueError, match="in_table"):
        qc.downgrade_only_violations(
            free.drop(columns=Columns.IN_TABLE), applied, species="human", qc=withheld
        )


def test_nr1_with_the_qc_effects_accepts_a_gate_failed_by_a_withheld_broad() -> None:
    """A withheld broad level lowers the coverage the gate reads (re-evaluated)."""
    free = _with_table(
        _labels(
            {"broad": [CONFIDENT] * 2, "supercluster": [CONFIDENT] * 2},
            {"broad": ["A", "B"], "supercluster": ["a", "b"]},
        )
    )
    applied = _with_table(
        _labels(
            {
                "broad": ["not_attempted_gate"] * 2,
                "supercluster": ["not_attempted_gate"] * 2,
            },
            {"broad": [None, None], "supercluster": [None, None]},
        )
    )
    effects = qc.qc_effects(
        [outcome("withhold_level", level="broad", check="prefilter_spotcheck")]
    )
    coverage = "broad_coverage_table: 0.000 < 0.25"
    assert (
        qc.downgrade_only_violations(
            free,
            applied,
            species="human",
            qc=effects,
            **_gates("full", "failed", coverage),
        )
        == []
    )
    # Without the effects the fall names no real_qc check.
    assert qc.downgrade_only_violations(
        free, applied, species="human", **_gates("full", "failed", coverage)
    ) == [
        "the gate level fell from full to failed without a named real_qc check "
        "among its level reasons"
    ]


# --------------------------------------------------------------------------
# RESOLVE (``annotate_resolve``) on the synthetic pair of test_pipeline_resolve

HUMAN_CHECKS: frozenset[str] = frozenset(
    {
        "marker_consistency",
        "registration_g1",
        "paired_concordance",
        "flag_rates",
        "gene_complexity",
        "prefilter_spotcheck",
        "coverage_vs_simulation",
        "nonneuronal_depth_trend",
        "factor_remeasure",
        "dataset_gate",
    }
)
OUTCOME_TOKENS: frozenset[str] = frozenset(
    {"pass", "warn", "fail", "not_applicable", "not_evaluable"}
)


def _with_real_qc(config: AnnotationConfig, **update: Any) -> AnnotationConfig:
    """Return the config with validated ``real_qc`` fields changed."""
    real_qc = type(config.real_qc).model_validate(
        {**config.real_qc.model_dump(), **update}
    )
    return config.model_copy(update={"real_qc": real_qc})


def _run(
    setup: Setup,
    make_trust: MakeTrust,
    output: str,
    *,
    state: str = "provisional",
    registration: Any = None,
    **qc_update: Any,
) -> pl.ResolveResult:
    """RESOLVE the synthetic pair with ``real_qc`` fields changed."""
    config = _with_real_qc(setup.config, **qc_update)
    return _resolve(
        setup, make_trust, output, state=state, config=config, registration=registration
    )


def _tables(result: pl.ResolveResult, sample_id: str) -> tuple[pd.DataFrame, Any]:
    return read_label_table(result.samples[sample_id].labels_path)


def _assert_nr1(
    free: pl.ResolveResult, applied: pl.ResolveResult
) -> dict[str, list[str]]:
    """Return NR1's violations per sample (QC-free vs QC-applied run).

    As C18 compares them: the label tables, the provenance, the resolve
    summary's gate records and the applied run's QC effects.
    """
    problems = {}
    for sample_id in free.samples:
        free_labels, free_prov = _tables(free, sample_id)
        applied_labels, applied_prov = _tables(applied, sample_id)
        applied_summary = applied.samples[sample_id].summary
        problems[sample_id] = qc.downgrade_only_violations(
            free_labels,
            applied_labels,
            species="human",
            free_provenance=free_prov,
            applied_provenance=applied_prov,
            free_gate=free.samples[sample_id].summary["resolution"]["gate"],
            applied_gate=applied_summary["resolution"]["gate"],
            qc=applied_summary["real_qc"].get("effects"),
        )
    return problems


def test_resolve_records_an_outcome_for_every_check(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """P4: every §8.8 check has an outcome per sample, in summary and provenance."""
    setup = _setup(tmp_path, fake_mmc)
    result = _run(setup, make_trust, "qc")
    summary = json.loads(result.summary_path.read_text())
    assert summary["schema_version"] == pl.RESOLVE_SUMMARY_SCHEMA_VERSION == 3
    assert summary["real_qc_config"] == setup.config.real_qc.model_dump(mode="json")
    for sample_id, sample in result.samples.items():
        record = summary["samples"][sample_id]["real_qc"]
        assert record["enabled"] and not record["promotes"]
        assert set(record["per_check"]) == HUMAN_CHECKS
        assert set(record["per_check"].values()) <= OUTCOME_TOKENS
        _, provenance = _tables(result, sample_id)
        assert provenance.panel is not None
        recorded = provenance.panel.real_data_qc
        assert recorded is not None
        assert recorded.outcomes == record["per_check"]
        assert recorded.warn_only is False
        assert sample.summary["real_qc"] == record
        # The synthetic bundle: no resolvability tables and no declared
        # version (unknown, so the version-7 checks and the factor
        # re-measure are not_evaluable, never not_applicable: P4), no
        # simulated genes, no prefilter; a paired section without a shared
        # mask; no registration check given.
        per_check = record["per_check"]
        assert per_check["prefilter_spotcheck"] == "not_applicable"
        assert per_check["factor_remeasure"] == "not_evaluable"
        assert per_check["coverage_vs_simulation"] == "not_evaluable"
        assert per_check["nonneuronal_depth_trend"] == "not_evaluable"
        assert per_check["gene_complexity"] == "not_evaluable"
        assert per_check["registration_g1"] == "not_evaluable"
        assert per_check["paired_concordance"] == "not_evaluable"
        assert per_check["dataset_gate"] == "warn"  # provisional: banner warning
        (referee,) = [
            item for item in record["outcomes"] if item["check"] == "marker_consistency"
        ]
        # The referee ran on the bundle's profiles: none of their genes is
        # specific, so it has no marker group (not_evaluable, never a pass).
        assert referee["state"] == "not_evaluable"
        assert "marker group" in referee["reason"]


def test_the_qc_free_run_records_the_qc_as_disabled(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _setup(tmp_path, fake_mmc)
    free = _run(setup, make_trust, "free", enabled=False)
    applied = _run(setup, make_trust, "qc")
    for sample_id, sample in free.samples.items():
        assert sample.summary["real_qc"] == {
            "enabled": False,
            "version": qc.REAL_QC_VERSION,
            "promotes": False,
        }
        _, provenance = _tables(free, sample_id)
        assert provenance.panel is not None
        assert provenance.panel.real_data_qc is None
    # NR1 on a run whose QC lowers nothing: identical labels and records.
    assert _assert_nr1(free, applied) == {sample: [] for sample in free.samples}
    for sample_id in free.samples:
        free_labels, _ = _tables(free, sample_id)
        applied_labels, _ = _tables(applied, sample_id)
        pd.testing.assert_frame_equal(free_labels, applied_labels)


def _low_referee(value: float) -> Callable[..., qc.MarkerConsistencySignal]:
    def fake(*_: Any, **__: Any) -> qc.MarkerConsistencySignal:
        return qc.MarkerConsistencySignal(
            consistency=value, n_marker_groups=3, n_pseudo_labelled=400
        )

    return fake


def test_a_low_marker_referee_caps_a_new_family_and_nr1_holds(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    make_trust: MakeTrust,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Below 0.70 the gate is broad_only (D18 (a)); NR1 holds against QC-free."""
    setup = _setup(tmp_path, fake_mmc)
    free = _run(setup, make_trust, "free", enabled=False)
    monkeypatch.setattr(pl, "human_marker_referee_signal", _low_referee(0.5))
    applied = _run(setup, make_trust, "qc")
    assert _assert_nr1(free, applied) == {sample: [] for sample in free.samples}
    for sample_id, sample in applied.samples.items():
        gate = sample.summary["resolution"]["gate"]
        assert free.samples[sample_id].summary["resolution"]["gate"]["level"] == "full"
        assert gate["level"] == "broad_only"
        assert any(
            reason.startswith("real_qc_marker_consistency")
            for reason in gate["level_reasons"]
        )
        record = sample.summary["real_qc"]
        assert record["per_check"]["marker_consistency"] == "fail"
        assert "broad_only:marker_consistency" in record["downgrades"]
        labels, provenance = _tables(applied, sample_id)
        free_labels, free_prov = _tables(free, sample_id)
        assert provenance.gate is not None and provenance.gate.level == "broad_only"
        assert provenance.panel is not None and free_prov.panel is not None
        assert provenance.panel.panel_trust == free_prov.panel.panel_trust
        table = labels[Columns.IN_TABLE].to_numpy(bool)
        leaf = labels[Columns.level("supercluster", "status")].astype(str)
        assert set(leaf[table]) == {"not_attempted_gate"}
        for level in ("lineage", "broad", "nt"):
            column = Columns.level(level, "status")
            assert list(labels[column]) == list(free_labels[column])
    # Trust and every margin are unchanged: the threshold records are equal.
    for sample_id in applied.samples:
        _, provenance = _tables(applied, sample_id)
        _, free_prov = _tables(free, sample_id)
        assert provenance.thresholds == free_prov.thresholds
        assert provenance.resolvability == free_prov.resolvability


def test_a_seeded_family_only_warns_on_a_low_referee(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    make_trust: MakeTrust,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set a (validated by real data): warn-only while gate H is pending (D20)."""
    setup = _setup(tmp_path, fake_mmc)
    free = _run(setup, make_trust, "free", state="validated_real", enabled=False)
    monkeypatch.setattr(pl, "human_marker_referee_signal", _low_referee(0.5))
    applied = _run(setup, make_trust, "qc", state="validated_real")
    for sample_id, sample in applied.samples.items():
        gate = sample.summary["resolution"]["gate"]
        assert gate["level"] == "full" and gate["warning"]
        assert any(
            "warn-only until the human gate" in reason
            for reason in gate["warning_reasons"]
        )
        record = sample.summary["real_qc"]
        assert record["seeded"] and record["warn_only"]
        assert record["per_check"]["marker_consistency"] == "warn"
        assert record["downgrades"] == []
        labels, provenance = _tables(applied, sample_id)
        free_labels, _ = _tables(free, sample_id)
        pd.testing.assert_frame_equal(labels, free_labels)
        assert provenance.panel is not None
        assert provenance.panel.real_data_qc is not None
        assert provenance.panel.real_data_qc.warn_only is True


def _write_specific_profiles(setup: Setup) -> None:
    """Give the fake WHB bundle profiles with one specific gene per node."""
    bundle = setup.bundles["whb_frontal_supc_clus"].path
    profiles = pd.read_parquet(bundle / "profiles.parquet")
    marker_gene = {label: GENE_IDS[index] for label, index in MARKER.items()}
    profiles["expected_fraction"] = [
        0.9 if marker_gene[node] == gene else 0.001
        for node, gene in zip(profiles["node"], profiles["gene_id"], strict=True)
    ]
    profiles.to_parquet(bundle / "profiles.parquet")


def test_the_referee_runs_on_the_bundle_profiles(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """With specific genes the referee derives sets and scores the calls."""
    setup = _setup(tmp_path, fake_mmc)
    _write_specific_profiles(setup)
    applied = _run(
        setup,
        make_trust,
        "qc",
        marker_referee_min_group_markers=1,
        marker_referee_min_pseudo_confident=10,
    )
    for sample in applied.samples.values():
        (referee,) = [
            item
            for item in sample.summary["real_qc"]["outcomes"]
            if item["check"] == "marker_consistency"
        ]
        assert referee["state"] == "evaluated", referee["reason"]
        details = referee["details"]
        # Neurons (two nodes), Astrocytes and Oligodendrocytes.
        assert details["n_marker_groups"] == 3
        assert details["source"] == "derived" and details["fingerprint"]
        # The sets record the panel and the primary bundle they were derived
        # on, so a table frozen before the run can be checked against them.
        marker_sets = details["marker_sets"]
        assert marker_sets["panel_hash"] == setup.panel.panel_hash
        assert marker_sets["build_hash"] == (
            setup.bundles["whb_frontal_supc_clus"].build_hash
        )
        assert marker_sets["n_query_genes"] == len(GENE_IDS)
        # The synthetic counts are mostly each call's own marker gene.
        assert details["consistency"] >= 0.75
        assert referee["outcome"] == "pass"


def test_the_referee_takes_the_panel_familys_ruled_comparator(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """RESOLVE passes the panel's family to the referee (C1 (b), one family).

    The synthetic panel's family is not ruled, so its referee derives its
    sets with the default ``node`` comparator; listed in
    ``marker_referee_comparator_by_family`` it takes ``class``, recorded with
    the family in the outcome's settings.
    """
    setup = _setup(tmp_path, fake_mmc)
    _write_specific_profiles(setup)
    family = pl.current_family(setup.panel, setup.config).panel_family
    assert family is not None
    assert (
        family.family_id not in setup.config.real_qc.marker_referee_comparator_by_family
    )

    def referee_settings(output: str, **update: Any) -> list[dict[str, Any]]:
        result = _run(
            setup,
            make_trust,
            output,
            marker_referee_min_group_markers=1,
            marker_referee_min_pseudo_confident=10,
            **update,
        )
        found = []
        for sample in result.samples.values():
            (referee,) = [
                item
                for item in sample.summary["real_qc"]["outcomes"]
                if item["check"] == "marker_consistency"
            ]
            assert referee["state"] == "evaluated", referee["reason"]
            found.append(
                {
                    **referee["details"]["settings"],
                    "sets": referee["details"]["marker_sets"],
                }
            )
        return found

    for record in referee_settings("default"):
        assert (record["comparator"], record["comparator_family"]) == ("node", None)
        assert record["sets"]["comparator"] == "node"
    ruled = referee_settings(
        "ruled", marker_referee_comparator_by_family={family.family_id: "class"}
    )
    for record in ruled:
        assert record["comparator"] == "class"
        assert record["comparator_family"] == family.family_id
        assert record["sets"]["comparator"] == "class"


class _WholeMask:
    """A shared tissue mask that holds every cell."""

    def contains(self, xy: np.ndarray) -> np.ndarray:
        return np.ones(len(xy), dtype=bool)


@pytest.mark.parametrize("state", ["provisional", "validated_real"])
def test_paired_concordance_withholds_the_pairs_supercluster_statistics(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    make_trust: MakeTrust,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    """Above the warning JSD the pair's supercluster statistics are withheld.

    Scored on the shared-mask point estimate (D21) once both sections are
    resolved; a seeded family only warns (D20 (b)).
    """
    from merxen.clustering.cross_platform import scope_from_resolve_summary

    setup = _setup(tmp_path, fake_mmc)
    monkeypatch.setattr(pl, "load_pair_mask", lambda _path: _WholeMask())
    applied = _run(setup, make_trust, "qc", state=state, paired_broad_jsd_warn=0.0)
    summary = json.loads(applied.summary_path.read_text())
    shared = [row for row in summary["pair"]["jsd"] if row["region"] == "shared_mask"]
    assert shared
    record = summary["pair"]["cross_platform"]
    scope = scope_from_resolve_summary(summary)
    seeded = state == "validated_real"
    for sample_id in applied.samples:
        sample = summary["samples"][sample_id]
        token = sample["real_qc"]["per_check"]["paired_concordance"]
        assert token == ("warn" if seeded else "fail")
        gate = sample["resolution"]["gate"]
        assert any(
            reason.startswith("real_qc_paired_concordance")
            for reason in gate["warning_reasons"]
        )
        _, provenance = _tables(applied, sample_id)
        assert provenance.panel is not None
        assert provenance.panel.real_data_qc is not None
        assert provenance.panel.real_data_qc.outcomes["paired_concordance"] == token
        assert provenance.gate is not None and provenance.gate.warning
        # The pair step rewrites the provenance's gate (``record_real_qc``):
        # its reasons are the summary's, with the paired-concordance reason,
        # which on a seeded family carries the warn-only note (D20 (b)).
        assert provenance.gate.level == gate["level"]
        assert provenance.gate.reasons == [
            *gate["level_reasons"],
            *gate["warning_reasons"],
        ]
        (reason,) = [
            item
            for item in provenance.gate.reasons
            if item.startswith("real_qc_paired_concordance")
        ]
        note = qc.WARN_ONLY_NOTE.format(species="human")
        assert (note in reason) is seeded
    if seeded:
        assert record["statistics_level"] == "full"
        assert pl.PAIRED_CONCORDANCE_REASON not in record["reasons"]
        assert scope.allows("supercluster")
    else:
        assert record["statistics_level"] == "broad_only" and record["flag"]
        assert pl.PAIRED_CONCORDANCE_REASON in record["reasons"]
        assert scope.allows("broad") and not scope.allows("supercluster")


def test_an_unpaired_section_records_paired_concordance_not_applicable(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    make_trust: MakeTrust,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new-panel family's case: a MERSCOPE-only section (NR1's list).

    A version-7 bundle without an R3 member or a prefilter, as the family's
    is. The synthetic pair needs its Xenium section to resolve its gene IDs
    (its MERSCOPE var holds symbols only), so the single-platform case is
    planted at the pair test RESOLVE uses.
    """
    from .test_resolve_v7 import _human_v7_config, write_human_tables

    assert pl.pair_has_both_platforms(["MERSCOPE", "xenium"])
    assert not pl.pair_has_both_platforms(["MERSCOPE", "MERSCOPE"])
    setup = _setup(tmp_path, fake_mmc)
    write_human_tables(setup.bundles["whb_frontal_supc_clus"].path, 7)
    monkeypatch.setattr(pl, "pair_has_both_platforms", lambda _platforms: False)
    applied = _resolve(
        setup, make_trust, "qc", state="provisional", config=_human_v7_config()
    )
    for sample in applied.samples.values():
        per_check = sample.summary["real_qc"]["per_check"]
        assert per_check["paired_concordance"] == "not_applicable"
        assert per_check["factor_remeasure"] == "not_applicable"
        assert per_check["prefilter_spotcheck"] == "not_applicable"
        # Version 7: the version-7 checks apply.
        assert per_check["coverage_vs_simulation"] != "not_applicable"
        assert per_check["nonneuronal_depth_trend"] != "not_applicable"
        assert not sample.paired


@pytest.mark.parametrize("effect", ["warning", "gate_failed"])
def test_registration_g1_follows_its_configured_effect_in_resolve(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust, effect: str
) -> None:
    """NR9 (D23 (b)): warn-only by default; gate_failed fails the dataset."""
    from merxen.annotation.mouse_gate import RegistrationSignal

    setup = _setup(tmp_path, fake_mmc)
    bad = RegistrationSignal(density_ratio=1.2, shift_um=0.5, source="qc.json")
    free = _run(setup, make_trust, "free", enabled=False)
    applied = _run(
        setup,
        make_trust,
        "qc",
        registration_g1_effect=effect,
        registration={"PX_MERSCOPE": bad},
    )
    sample = applied.samples["PX_MERSCOPE"]
    record = sample.summary["real_qc"]
    gate = sample.summary["resolution"]["gate"]
    labels, _ = _tables(applied, "PX_MERSCOPE")
    free_labels, _ = _tables(free, "PX_MERSCOPE")
    if effect == "warning":
        assert record["per_check"]["registration_g1"] == "warn"
        assert gate["level"] == "full"
        pd.testing.assert_frame_equal(labels, free_labels)
    else:
        assert record["per_check"]["registration_g1"] == "fail"
        assert gate["level"] == "failed"
        table = labels[Columns.IN_TABLE].to_numpy(bool)
        for level in ("lineage", "broad", "nt", "supercluster"):
            status = labels[Columns.level(level, "status")].astype(str)
            assert set(status[table]) == {"not_attempted_gate"}
        assert labels[Columns.EXCLUDE_HARD][table].all()
        # The flags are recomputed on the second pass: their rates' basis is
        # the confident broad calls, of which a failed gate leaves none.
        free_strata = free.samples["PX_MERSCOPE"].summary["flags"]["strata"]
        strata = sample.summary["flags"]["strata"]
        assert any(stratum["n_basis"] > 0 for stratum in free_strata)
        assert strata and all(stratum["n_basis"] == 0 for stratum in strata)
        _, provenance = _tables(applied, "PX_MERSCOPE")
        assert provenance.flags is not None
        assert not provenance.flags.realised_rates
    other = applied.samples["PX_XENIUM"].summary["real_qc"]["per_check"]
    assert other["registration_g1"] == "not_evaluable"
    assert _assert_nr1(free, applied) == {sample: [] for sample in free.samples}


def _bundle(manifest: dict[str, Any]) -> Any:
    return SimpleNamespace(bundle=SimpleNamespace(manifest=manifest))


def _declared(version: Any, *, where: str = "builder_params") -> dict[str, Any]:
    if where == "builder_params":
        return {
            "build_hash_payload": {
                "builder_params": {
                    "resolvability": {"enabled": True, "resolvability_version": version}
                }
            }
        }
    return {"builder_output": {"resolvability": {"resolvability_version": version}}}


def test_the_bundle_facts_of_real_qc_are_unknown_without_tables() -> None:
    """P4: the version-7 facts come from the tables, else from the bundle.

    A bundle that declares version 7 but whose tables were not read has an
    unknown R3 member, so the factor re-measure is ``not_evaluable``, and its
    version-7 checks read version 7 (``not_evaluable`` without inputs).
    Nothing read: unknown, never "does not apply".
    """
    r3 = SimpleNamespace(
        version=7, summary={"emission_members": ["R1_decision@0", "R3_measured_HO@0"]}
    )
    no_r3 = SimpleNamespace(version=7, summary={"emission_members": ["R1_decision@0"]})
    v6 = SimpleNamespace(version=6, summary={})
    unrecorded = SimpleNamespace(version=None, summary={})
    primary = _bundle(_declared(7))
    assert pl.real_qc_bundle_facts(primary, r3) == (7, True)
    assert pl.real_qc_bundle_facts(primary, no_r3) == (7, False)
    assert pl.real_qc_bundle_facts(primary, v6) == (6, False)
    # A summary without the field: version-6 tables.
    assert pl.real_qc_bundle_facts(primary, unrecorded) == (6, False)
    # No tables: the bundle's declared version.
    assert pl.real_qc_bundle_facts(primary, None) == (7, None)
    assert pl.real_qc_bundle_facts(_bundle(_declared(6)), None) == (6, False)
    output = _bundle(_declared(7, where="builder_output"))
    assert pl.real_qc_bundle_facts(output, None) == (7, None)
    disabled = {"builder_params": {"resolvability": {"enabled": False}}}
    for manifest in (
        {},
        {"build_hash_payload": disabled},
        _declared(99),
        _declared("seven"),
    ):
        assert pl.real_qc_bundle_facts(_bundle(manifest), None) == (None, None)
    assert pl.real_qc_bundle_facts(None, None) == (None, None)
    config = AnnotationConfig(species="human")
    record = qc.real_data_qc(
        qc.RealQcSignals(
            resolvability_version=7,
            paired=False,
            prefilter_applied=False,
            has_r3_member=None,
        ),
        None,
        config,
    ).provenance()
    for check in ("coverage_vs_simulation", "nonneuronal_depth_trend"):
        assert record.outcomes[check] == "not_evaluable", check
    assert record.outcomes["factor_remeasure"] == "not_evaluable"


def test_the_prefilter_fact_is_read_from_the_primary_bundle() -> None:
    """P4: the spot check's applicability is the bundle's hashed prefilter.

    A bundle whose ``build_hash_payload`` holds a ``large_panel_prefilter``
    is one the spot check applies to (``not_evaluable`` in RESOLVE, which has
    no spot-check producer); one without it (``None`` or absent, as run
    selection reads it) is ``not_applicable``. Without a primary run the fact
    is unknown, never "does not apply" (C15 review F3).
    """
    from merxen.annotation.prefilter import prefilter_payload

    prefiltered = {"large_panel_prefilter": prefilter_payload(2000, 5)}
    assert pl.real_qc_prefilter_applied(_bundle({"build_hash_payload": prefiltered}))
    for manifest in (
        {"build_hash_payload": {"large_panel_prefilter": None}},
        {"build_hash_payload": {}},
        {},
    ):
        assert pl.real_qc_prefilter_applied(_bundle(manifest)) is False
    assert pl.real_qc_prefilter_applied(None) is None
    config = AnnotationConfig(species="human")

    def token(applied: bool | None) -> str:
        signals = qc.RealQcSignals(
            resolvability_version=6,
            paired=False,
            prefilter_applied=applied,
            has_r3_member=False,
        )
        record = qc.real_data_qc(signals, None, config).provenance()
        return record.outcomes["prefilter_spotcheck"]

    assert token(None) == "not_evaluable"
    assert token(True) == "not_evaluable"
    assert token(False) == "not_applicable"


def test_resolve_on_a_prefiltered_bundle_records_the_spot_check_not_evaluable(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """A 5K-type bundle (marker lookup prefiltered): the spot check applies.

    RESOLVE has no spot-check producer, so P4 records ``not_evaluable``; a
    reading of the bundle that lost the prefilter would record
    ``not_applicable`` (C15 review F3).
    """
    from merxen.annotation.prefilter import prefilter_payload

    setup = _setup(tmp_path, fake_mmc)
    path = setup.bundles["whb_frontal_supc_clus"].path / "bundle.json"
    manifest = json.loads(path.read_text())
    manifest.setdefault("build_hash_payload", {})["large_panel_prefilter"] = (
        prefilter_payload(2000, 5)
    )
    path.write_text(json.dumps(manifest))
    applied = _run(setup, make_trust, "qc")
    for sample_id, sample in applied.samples.items():
        record = sample.summary["real_qc"]
        assert record["per_check"]["prefilter_spotcheck"] == "not_evaluable"
        (spotcheck,) = [
            item
            for item in record["outcomes"]
            if item["check"] == "prefilter_spotcheck"
        ]
        assert spotcheck["state"] == "not_evaluable"
        assert "without a spot check" in spotcheck["reason"]
        _, provenance = _tables(applied, sample_id)
        assert provenance.panel is not None
        assert provenance.panel.real_data_qc is not None
        outcomes = provenance.panel.real_data_qc.outcomes
        assert outcomes["prefilter_spotcheck"] == "not_evaluable"


def test_qc_lowers_resolution_needs_a_withheld_level_or_a_harder_cap() -> None:
    """The second RESOLVE pass runs for a withheld level or a harder gate cap."""
    withheld = qc.qc_effects([outcome("withhold_level", level="supercluster")])
    assert pl.qc_lowers_resolution(withheld, "full")
    assert pl.qc_lowers_resolution(withheld, "failed")
    capped = qc.qc_effects([outcome("gate_cap", gate_cap="broad_only")])
    assert pl.qc_lowers_resolution(capped, "full")
    assert not pl.qc_lowers_resolution(capped, "broad_only")
    assert not pl.qc_lowers_resolution(capped, "failed")
    warned = qc.qc_effects(
        [
            outcome("warning"),
            outcome("report_only"),
            outcome("withhold_pair_stats", check="paired_concordance"),
        ]
    )
    assert not pl.qc_lowers_resolution(warned, "full")


def test_a_withheld_level_in_resolve_is_not_emitted_and_nr1_holds(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    make_trust: MakeTrust,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A withhold outcome (the prefilter spot check's effect) in RESOLVE.

    RESOLVE has no spot-check input yet, so the outcome is planted on the QC
    result. The second pass withholds supercluster and the SEA-AD subclass
    that reads its table, the levels above keep their statuses, the gate keeps
    its level and names the check among its warnings, and NR1 holds.
    """
    setup = _setup(tmp_path, fake_mmc)
    free = _run(setup, make_trust, "free", enabled=False)
    real_qc = qc.real_data_qc

    def with_a_withheld_supercluster(*args: Any, **kwargs: Any) -> qc.RealQcResult:
        result = real_qc(*args, **kwargs)
        withheld = qc.QcOutcome(
            qc.PREFILTER_SPOTCHECK_CHECK,
            True,
            "withhold_level",
            message="agreement 0.80 < 0.95 at supercluster",
            level="supercluster",
        )
        return result.replace_check(qc.PREFILTER_SPOTCHECK_CHECK, [withheld])

    monkeypatch.setattr(qc, "real_data_qc", with_a_withheld_supercluster)
    applied = _run(setup, make_trust, "qc")
    assert _assert_nr1(free, applied) == {sample: [] for sample in free.samples}
    for sample_id, sample in applied.samples.items():
        record = sample.summary["real_qc"]
        assert record["per_check"]["prefilter_spotcheck"] == "fail"
        assert "not_resolvable_supercluster:prefilter_spotcheck" in record["downgrades"]
        gate = sample.summary["resolution"]["gate"]
        free_gate = free.samples[sample_id].summary["resolution"]["gate"]
        assert gate["level"] == free_gate["level"] == "full"
        assert any(
            reason.startswith("real_qc_prefilter_spotcheck")
            for reason in gate["warning_reasons"]
        )
        labels, _ = _tables(applied, sample_id)
        free_labels, _ = _tables(free, sample_id)
        for level in ("supercluster", "seaad_subclass"):
            column = Columns.level(level, "status")
            was = free_labels[column].astype(str).to_numpy() == CONFIDENT
            now = labels[column].astype(str).to_numpy()
            assert was.any(), level
            assert set(now[was]) == {CellStatus.NOT_RESOLVABLE.value}, level
        for level in ("lineage", "broad", "nt"):
            column = Columns.level(level, "status")
            assert list(labels[column]) == list(free_labels[column])


def _label_confident(labels: pd.DataFrame, level: str, table: np.ndarray) -> np.ndarray:
    """Whether each table cell is confident at ``level`` in a label table."""
    status = labels[Columns.level(level, "status")].astype(str).to_numpy()
    return (status == CONFIDENT)[table]


def test_resolve_on_a_version_7_bundle_scores_coverage_and_the_trend(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    make_trust: MakeTrust,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version 7: per-class coverage per level's class key, and the trend.

    NR8's real share per (level, class key) is the label table's confident
    share, and the warnings are exactly the judged (level, class) below the
    prediction by more than the margin: a planted low-coverage class (its
    prediction raised to 1.0) warns. A planted fall of the confident calls at
    high depth fires the trend, which reads the table cells' counts, each
    level's class keys and confident calls, and the class-depth table's
    non-neuronal classes.
    """
    from .test_resolve_v7 import HUMAN_NEURONAL, _human_v7_config, write_human_tables

    setup = _setup(tmp_path, fake_mmc)
    write_human_tables(setup.bundles["whb_frontal_supc_clus"].path, 7)
    config = _with_real_qc(_human_v7_config(), coverage_min_cells=20)
    limit = config.real_qc.nonneuronal_high_depth_counts
    planted = ("broad", "Oligo")
    class_keys: dict[int, dict[str, np.ndarray]] = {}
    real_outputs = pl.version_7_outputs
    real_trend = qc.nonneuronal_depth_trend
    trend_calls: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], set[str]]]
    trend_calls = []

    def outputs_with_a_planted_prediction(*args: Any, **kwargs: Any) -> Any:
        outputs = real_outputs(*args, **kwargs)
        assert outputs.prediction is not None
        # Keyed by the sample's objects (``counts``).
        class_keys[len(args[3])] = dict(outputs.class_keys)
        prediction = outputs.prediction.copy()
        row = (prediction["level"] == planted[0]) & (prediction["class"] == planted[1])
        assert int(row.sum()) == 1
        prediction.loc[row, "predicted_coverage"] = 1.0
        return dataclasses.replace(outputs, prediction=prediction)

    def trend_with_a_planted_fall(
        totals: Any, called_class: Any, confident: Any, classes: Any, **kwargs: Any
    ) -> Any:
        values = np.asarray(totals, dtype=np.float64)
        flags = {
            level: np.asarray(item, dtype=bool) for level, item in confident.items()
        }
        trend_calls.append(
            (values, np.asarray(called_class, dtype=object), flags, set(classes))
        )
        fallen = {level: item & (values < limit) for level, item in flags.items()}
        return real_trend(
            totals, called_class, fallen, classes, min_band_cells=10, **kwargs
        )

    monkeypatch.setattr(pl, "version_7_outputs", outputs_with_a_planted_prediction)
    monkeypatch.setattr(qc, "nonneuronal_depth_trend", trend_with_a_planted_fall)
    # The synthetic sections end at 400 counts: the trend compares the cells
    # below the high-depth limit with those at or above it.
    monkeypatch.setattr(qc, "DEPTH_BANDS", ((10, limit),))
    applied = _resolve(setup, make_trust, "qc", state="provisional", config=config)
    nonneuronal = {cls for cls, neuronal in HUMAN_NEURONAL.items() if not neuronal}
    margin = config.real_qc.coverage_warn_margin
    for sample_id, sample in applied.samples.items():
        record = sample.summary["real_qc"]
        assert record["per_check"]["gene_complexity"] == "not_evaluable"
        # No lowering effect: the label table is the QC-free resolution's.
        assert not record["downgrades"]
        labels, _ = _tables(applied, sample_id)
        table = labels[Columns.IN_TABLE].to_numpy(dtype=bool)
        keys = class_keys[len(labels)]
        expected = pd.DataFrame(
            [
                {
                    "level": level,
                    "class": cls,
                    "n_cells": len(group),
                    "real_coverage": float(group["confident"].mean()),
                }
                for level, level_keys in keys.items()
                # Cells without a class key are not judged.
                for cls, group in pd.DataFrame(
                    {
                        "class": level_keys,
                        "confident": _label_confident(labels, level, table),
                    }
                )
                .dropna(subset=["class"])
                .groupby("class")
            ]
        )
        coverage = pd.DataFrame(record["tables"]["coverage_vs_simulation"])
        order = ["level", "class"]
        pd.testing.assert_frame_equal(
            coverage[[*order, "n_cells", "real_coverage"]]
            .sort_values(order)
            .reset_index(drop=True),
            expected.sort_values(order).reset_index(drop=True),
            check_dtype=False,
        )
        predicted = pd.DataFrame(
            sample.summary["resolvability_v7"]["class_depth_prediction"]["classes"]
        )
        judged = coverage[coverage["judged"]]
        assert set(zip(judged["level"], judged["class"], strict=True)) <= set(
            zip(predicted["level"], predicted["class"], strict=True)
        )
        below = judged[judged["real_coverage"] < judged["predicted_coverage"] - margin]
        warned = {
            (item["level"], item["class"])
            for item in record["outcomes"]
            if item["check"] == "coverage_vs_simulation" and item["fired"]
        }
        assert warned == set(zip(below["level"], below["class"], strict=True))
        assert planted in warned
        assert record["per_check"]["coverage_vs_simulation"] == "warn"
        # The trend fired on the planted fall, on the non-neuronal classes.
        fired = [
            item
            for item in record["outcomes"]
            if item["check"] == "nonneuronal_depth_trend" and item["fired"]
        ]
        assert fired and {item["class"] for item in fired} <= nonneuronal
        assert record["per_check"]["nonneuronal_depth_trend"] == "warn"
        totals = labels[Columns.TOTAL_COUNTS].to_numpy(dtype=np.float64)[table]
        calls = [item for item in trend_calls if len(item[0]) == int(table.sum())]
        assert sorted(level for item in calls for level in item[2]) == sorted(keys)
        for values, called, flags, classes in calls:
            assert classes == nonneuronal
            np.testing.assert_allclose(values, totals, rtol=1e-6)
            ((level, confident),) = flags.items()
            assert list(called) == list(keys[level])
            assert list(confident) == list(_label_confident(labels, level, table))


def test_coverage_rows_record_the_gate_that_blocks_their_level(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """NR8 under a ``broad_only`` gate before QC (C15 review F2).

    The gate leaves the leaf levels unattempted, so their real coverage is 0
    by the gate, not by the simulation. Each coverage row records the gate
    level and whether it blocks the row's level, and a warning at a blocked
    level says the shortfall is the gate's. The warning rule is unchanged:
    dropping those warnings would be a loosening (CHECK K6 rule 4).
    """
    from .test_resolve_v7 import _human_v7_config, write_human_tables

    setup = _setup(tmp_path, fake_mmc)
    write_human_tables(setup.bundles["whb_frontal_supc_clus"].path, 7)
    config = _with_real_qc(_human_v7_config(), coverage_min_cells=20)
    # No table cell reaches the gate's depth: A = 0 < 0.30, so broad_only.
    gate = config.gate.model_copy(update={"depth_counts": 10**6})
    config = config.model_copy(update={"gate": gate})
    applied = _resolve(setup, make_trust, "qc", state="provisional", config=config)
    blocked_levels = set(qc.LEAF_GATED_LEVELS["human"])
    n_blocked = 0
    for sample in applied.samples.values():
        assert sample.summary["resolution"]["gate"]["level"] == "broad_only"
        record = sample.summary["real_qc"]
        coverage = pd.DataFrame(record["tables"]["coverage_vs_simulation"])
        assert len(coverage)
        assert set(coverage["gate_level"]) == {"broad_only"}
        assert coverage["gate_blocked"].tolist() == [
            level in blocked_levels for level in coverage["level"]
        ]
        blocked_rows = coverage[coverage["gate_blocked"].astype(bool)]
        assert len(blocked_rows) and (blocked_rows["real_coverage"] == 0).all()
        for item in record["outcomes"]:
            if item["check"] != "coverage_vs_simulation" or not item["fired"]:
                continue
            blocked = item["level"] in blocked_levels
            assert item["details"]["gate_level"] == "broad_only"
            assert item["details"]["gate_blocked"] is blocked
            assert ("this shortfall is the gate's" in item["message"]) is blocked
            assert ("dataset gate before QC is broad_only" in item["message"]) is (
                blocked
            )
            n_blocked += blocked
    # The planted case: blocked levels still warn, now saying why.
    assert n_blocked


def _write_simulated_genes(
    bundle_dir: Path,
    query_genes: list[str],
    n_genes_by_depth: dict[int, tuple[int, int]],
    members: tuple[str, ...],
) -> None:
    """Add a version-7 bundle's simulated ``n_genes`` (M13 C16) to its tables.

    ``n_genes_by_depth`` maps a grid depth to (test cells, genes per cell);
    every emission member stores the same cells.
    """
    from merxen.annotation import resolvability as res

    rows = [
        {
            res.MEMBER_COLUMN: member,
            res.MEMBER_ROLE_COLUMN: "emission",
            "cell_id": f"t{index}",
            "depth": depth,
            "total_counts": depth,
            "n_genes": n_genes,
        }
        for member in members
        for depth, (n_cells, n_genes) in n_genes_by_depth.items()
        for index in range(n_cells)
    ]
    table = pd.DataFrame(rows, columns=list(res.SIM_GENES_COLUMNS))
    table.to_parquet(bundle_dir / res.SIM_GENES_FILE, index=False)
    path = bundle_dir / res.RESOLVABILITY_SUMMARY_FILE
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary[res.SIM_GENES_RECORD] = res.sim_genes_record(query_genes, table)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def test_resolve_scores_gene_complexity_on_the_bundles_query_genes(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """NR7 (D19 (a)): native cells are counted on the bundle's query genes.

    A version-7 bundle storing simulated ``n_genes`` makes the check
    evaluable in RESOLVE. The native side is each sample's table cells,
    counted on the bundle's query genes (a query gene the dataset lacks
    counts as not detected and is reported), binned by their counts there;
    the simulated side is the stored test cells, each interpolated between
    the bin's edges to the bin's native median total (the user's ruling C3
    (b) of 2026-10-07), the open top bin at its lower edge. A bin where the
    simulated cells carry about as many genes as the native ones passes, and
    one where they carry half as many warns.
    """
    import anndata as ad

    from merxen.annotation import resolvability as res

    from .test_resolve_v7 import (
        HUMAN_GRID,
        MEMBERS,
        _human_v7_config,
        write_human_tables,
    )

    setup = _setup(tmp_path, fake_mmc)
    bundle_dir = setup.bundles["whb_frontal_supc_clus"].path
    write_human_tables(bundle_dir, 7)
    absent = "ENSG00000999999"
    query_genes = [*GENE_IDS, absent]
    # The synthetic cells detect all six panel genes from about 100 counts.
    _write_simulated_genes(
        bundle_dir,
        query_genes,
        {10: (20, 2), 30: (20, 4), 100: (60, 6), 250: (60, 4)},
        MEMBERS,
    )
    applied = _resolve(
        setup, make_trust, "qc", state="provisional", config=_human_v7_config()
    )
    gap_warn = setup.config.real_qc.genes_per_count_gap_warn
    simulated = {10: (20, 2.0), 30: (20, 4.0), 100: (60, 6.0), 250: (60, 4.0)}
    for sample in setup.samples:
        record = applied.samples[sample.sample_id].summary["real_qc"]
        labels, _ = _tables(applied, sample.sample_id)
        table = labels[Columns.IN_TABLE].to_numpy(dtype=bool)
        # The prepared H5AD's six panel genes, in GENE_IDS order.
        native = np.asarray(ad.read_h5ad(sample.h5ad_path).X[:, :6].todense())[table]
        n_genes = (native > 0).sum(axis=1)
        totals = native.sum(axis=1).astype(np.float64)
        bins = res.depth_bin(totals, HUMAN_GRID)
        rows = pd.DataFrame(record["tables"]["gene_complexity"])
        grid = list(HUMAN_GRID)
        assert rows["depth"].tolist() == grid
        for row in rows.to_dict("records"):
            depth = int(row["depth"])
            in_bin = bins == depth
            position = grid.index(depth)
            n_low, genes_low = simulated[depth]
            expected = genes_low
            if position + 1 < len(grid):
                # The same test cells (t0, t1, ...) at both edges: matched.
                upper = grid[position + 1]
                n_high, genes_high = simulated[upper]
                n_simulated = min(n_low, n_high)
                assert row["matching"] == "interpolated"
                if in_bin.any():
                    weight = (np.log(np.median(totals[in_bin])) - np.log(depth)) / (
                        np.log(upper) - np.log(depth)
                    )
                    expected = (1 - weight) * genes_low + weight * genes_high
            else:
                n_simulated = n_low
                assert row["matching"] == "lower_edge"
            assert row["n_native"] == int(in_bin.sum())
            assert row["n_simulated"] == n_simulated
            if not in_bin.any():
                continue
            assert row["simulated_median_genes"] == pytest.approx(expected)
            assert row["native_median_genes"] == float(np.median(n_genes[in_bin]))
            judged = int(in_bin.sum()) >= 50 and n_simulated >= 50
            assert row["judged"] == judged
            gap = float(np.median(n_genes[in_bin])) / expected - 1.0
            assert row["warn"] == (judged and gap > gap_warn)
        judged_rows = rows[rows["judged"]]
        assert set(judged_rows["depth"]) == {100, 250}
        assert set(judged_rows.loc[judged_rows["warn"], "depth"]) == {250}
        assert record["per_check"]["gene_complexity"] == "warn"
        (item,) = [
            entry for entry in record["outcomes"] if entry["check"] == "gene_complexity"
        ]
        assert item["fired"] and item["effect"] == "warning"
        assert item["details"]["bins_warned"] == [250]
        source = item["details"]["source"]
        assert source["n_query_genes"] == len(query_genes)
        assert source["n_query_genes_missing"] == 1
        assert source["missing_genes"] == [absent]
        assert source["members"] == list(MEMBERS)
        # A warning only: nothing is lowered.
        assert not record["downgrades"]


def test_the_cli_reads_a_human_registration_check(tmp_path: Path) -> None:
    from merxen.cli.run_annotation import _registration_signals

    path = tmp_path / "px_merscope_registration_qc.json"
    path.write_text(json.dumps({"density_ratio": 1.2, "shift_um": 0.5}))
    signals = _registration_signals(
        (f"PX_MERSCOPE={path}",), ["PX_MERSCOPE", "PX_XENIUM"], "human"
    )
    assert signals["PX_MERSCOPE"].density_ratio == pytest.approx(1.2)


# --------------------------------------------------------------------------
# Report: the panel card's real-data QC table


def test_the_panel_card_lists_the_real_qc_outcomes(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    from merxen.annotation.report_panel import REAL_QC_COLUMNS, build_panel_card

    setup = _setup(tmp_path, fake_mmc)
    applied = _run(setup, make_trust, "qc", paired_broad_jsd_warn=0.0)
    samples = [
        {
            "sample_id": sample_id,
            "platform": sample.platform,
            "summary": sample.summary,
            "manifest": json.loads(sample.manifest_path.read_text()),
        }
        for sample_id, sample in applied.samples.items()
    ]
    card = build_panel_card(
        samples, panel_report=None, bundles={}, primary_reference=None
    )
    table = card.tables()["panel_real_qc"]
    assert list(table.columns) == list(REAL_QC_COLUMNS)
    for sample_id, sample in applied.samples.items():
        rows = table[table["sample_id"] == sample_id]
        assert list(rows["check"]) == [
            item["check"] for item in sample.summary["real_qc"]["outcomes"]
        ]
    disabled = build_panel_card(
        [{**samples[0], "summary": {"real_qc": {"enabled": False}}}],
        panel_report=None,
        bundles={},
        primary_reference=None,
    )
    assert disabled.real_qc.empty


def test_nr1_holds_on_a_version_7_bundle_with_a_gate_cap(
    tmp_path: Path,
    fake_mmc: FakeMmc,
    make_trust: MakeTrust,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NR1 with resolvability tables: the emission records stay the QC-free run's.

    Only the share of extrapolated confident labels may fall with the
    confident set (``NR1_LABEL_FIELDS``).
    """
    from .test_resolve_v7 import _human_v7_config, write_human_tables

    setup = _setup(tmp_path, fake_mmc)
    write_human_tables(setup.bundles["whb_frontal_supc_clus"].path, 7)
    free = _resolve(
        setup,
        make_trust,
        "free",
        state="provisional",
        config=_with_real_qc(_human_v7_config(), enabled=False),
    )
    monkeypatch.setattr(pl, "human_marker_referee_signal", _low_referee(0.6))
    applied = _resolve(
        setup, make_trust, "qc", state="provisional", config=_human_v7_config()
    )
    assert _assert_nr1(free, applied) == {sample: [] for sample in free.samples}
    lowered = False
    for sample_id, sample in applied.samples.items():
        assert sample.summary["resolution"]["gate"]["level"] == "broad_only"
        labels, provenance = _tables(applied, sample_id)
        free_labels, free_prov = _tables(free, sample_id)
        column = Columns.level("supercluster", "status")
        lowered |= bool(
            (
                (free_labels[column].astype(str) == CONFIDENT)
                & (labels[column].astype(str) != CONFIDENT)
            ).any()
        )
        for reference, record in provenance.resolvability.items():
            other = free_prov.resolvability[reference]
            assert record.emitted_depth_bins == other.emitted_depth_bins
            assert record.d_max == other.d_max
            assert record.resolvable_share == other.resolvable_share
    assert lowered
