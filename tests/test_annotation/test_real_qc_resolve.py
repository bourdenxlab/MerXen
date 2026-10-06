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
    assert check(provenance(gate=named)) == []
    assert check(provenance(gate={"level": "broad_only", "reasons": ["x"]})) == [
        "the gate level fell from full to broad_only without a named real_qc check"
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
    """Return NR1's violations per sample (QC-free vs QC-applied run)."""
    problems = {}
    for sample_id in free.samples:
        free_labels, free_prov = _tables(free, sample_id)
        applied_labels, applied_prov = _tables(applied, sample_id)
        problems[sample_id] = qc.downgrade_only_violations(
            free_labels,
            applied_labels,
            species="human",
            free_provenance=free_prov,
            applied_provenance=applied_prov,
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
        # The synthetic bundle: version 6 without simulated genes, no
        # prefilter, no R3 member; a paired section without a shared mask;
        # no registration check given.
        per_check = record["per_check"]
        assert per_check["prefilter_spotcheck"] == "not_applicable"
        assert per_check["factor_remeasure"] == "not_applicable"
        assert per_check["coverage_vs_simulation"] == "not_applicable"
        assert per_check["nonneuronal_depth_trend"] == "not_applicable"
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
        # The synthetic counts are mostly each call's own marker gene.
        assert details["consistency"] >= 0.75
        assert referee["outcome"] == "pass"


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

    The synthetic pair needs its Xenium section to resolve its gene IDs (its
    MERSCOPE var holds symbols only), so the single-platform case is planted
    at the pair test RESOLVE uses.
    """
    assert pl.pair_has_both_platforms(["MERSCOPE", "xenium"])
    assert not pl.pair_has_both_platforms(["MERSCOPE", "MERSCOPE"])
    setup = _setup(tmp_path, fake_mmc)
    monkeypatch.setattr(pl, "pair_has_both_platforms", lambda _platforms: False)
    applied = _run(setup, make_trust, "qc")
    for sample in applied.samples.values():
        per_check = sample.summary["real_qc"]["per_check"]
        assert per_check["paired_concordance"] == "not_applicable"
        assert per_check["factor_remeasure"] == "not_applicable"
        assert per_check["prefilter_spotcheck"] == "not_applicable"
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
    other = applied.samples["PX_XENIUM"].summary["real_qc"]["per_check"]
    assert other["registration_g1"] == "not_evaluable"
    assert _assert_nr1(free, applied) == {sample: [] for sample in free.samples}


def test_resolve_on_a_version_7_bundle_scores_coverage_and_the_trend(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """Version 7: per-class coverage per level's class key, the trend reported."""
    from .test_resolve_v7 import _human_v7_config, write_human_tables

    setup = _setup(tmp_path, fake_mmc)
    write_human_tables(setup.bundles["whb_frontal_supc_clus"].path, 7)
    config = _with_real_qc(_human_v7_config(), coverage_min_cells=20)
    applied = _resolve(setup, make_trust, "qc", state="provisional", config=config)
    for sample in applied.samples.values():
        record = sample.summary["real_qc"]
        assert record["per_check"]["coverage_vs_simulation"] in {"pass", "warn"}
        assert record["per_check"]["gene_complexity"] == "not_evaluable"
        coverage = pd.DataFrame(record["tables"]["coverage_vs_simulation"])
        predicted = pd.DataFrame(
            sample.summary["resolvability_v7"]["class_depth_prediction"]["classes"]
        )
        judged = coverage[coverage["judged"]]
        assert len(judged)
        keys = set(zip(predicted["level"], predicted["class"], strict=True))
        assert set(zip(judged["level"], judged["class"], strict=True)) <= keys
        assert record["per_check"]["nonneuronal_depth_trend"] in {
            "pass",
            "warn",
            "not_evaluable",
        }
        assert "nonneuronal_depth_trend" in record["tables"]


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
