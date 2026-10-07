"""Tests for gate P's assembly (M13; plan §14 gate-P rule and per-class records).

C_P per level, the per-(level, class) records combined over NP3-NP7 and the
emission members, ``validated_max_level``, the family checks NP1, NP2, NP8
and NP9, the NP1-NP9 report and the validated-table rows of a passing
family (pre-registration §23.9, §23.16).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import diagnostics as diag
from merxen.annotation import gate_p as gp
from merxen.annotation import resolvability as res
from merxen.annotation.config import (
    AnnotationPanelConfig,
    AnnotationResolvabilityConfig,
)
from merxen.annotation.panel import compute_panel_hash, panel_family

Key = tuple[str, str]

NEURONS = ("Exc", "Inh")
GLIA = ("Astro", "Oligo")
# Pooled test cells per truth class (C_P: >= 700 cells).
COUNTS = {"Exc": 3000, "Inh": 1500, "Astro": 1200, "Oligo": 1000, "Vascular": 100}
LEVELS = ("lineage", "broad", "nt", "supercluster")
V6_MEMBERS = ("R1@0",)
V7_MEMBERS = tuple(f"R1@{seed}" for seed in res.V7_R1_SEEDS_WITHOUT_TABLE)
GENE_IDS = [f"ENSG{index:011d}" for index in range(1, 121)]
PANEL_HASH = compute_panel_hash(sorted(GENE_IDS))
FAMILY_ID = diag.own_family_id("human", ["MERSCOPE"], PANEL_HASH)


# --------------------------------------------------------------------------
# Fixtures


def _test_cells(
    counts: Mapping[str, int] = COUNTS,
    *,
    sinks: int = 0,
    rows_per_cell: int = 1,
    group: str | None = None,
) -> pd.DataFrame:
    """Pooled test cells as the human level specs read them.

    Every cell has a truth class at lineage, broad and supercluster; at NT a
    neuron keeps its class and a glial cell's truth is ``not_neuron`` (the
    level does not apply), as ``whb_level_specs`` reads them. ``sinks``
    cells have no class at any level (a sink truth).
    """
    records: list[dict[str, object]] = []
    for cls, n in counts.items():
        for index in range(n):
            for level in LEVELS:
                truth: object = cls
                if level == "nt":
                    truth = cls if cls in NEURONS else res.NOT_NEURON
                for _ in range(rows_per_cell):
                    records.append(
                        {
                            "level": level,
                            "cell_id": f"{group or 'g'}_{cls}_{index}",
                            "truth": truth,
                            "truth_parent": cls,
                        }
                    )
    for index in range(sinks):
        for level in LEVELS:
            records.append(
                {
                    "level": level,
                    "cell_id": f"{group or 'g'}_sink_{index}",
                    "truth": res.NOT_NEURON if level == "nt" else None,
                    "truth_parent": None,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if group is not None:
        frame["group"] = group
    return frame


def _class_sets(**kwargs: Any) -> gp.ClassSets:
    return gp.gate_p_class_sets(
        _test_cells(**kwargs),
        species="human",
        default_group=None,
        settings=gp.ClassSetSettings.from_config(AnnotationResolvabilityConfig()),
    )


def _keys() -> list[Key]:
    """Every (level, class) with calls: NT holds the neurons only."""
    keys: list[Key] = []
    for level in LEVELS:
        classes = NEURONS if level == "nt" else (*NEURONS, *GLIA, "Vascular")
        keys.extend((level, cls) for cls in classes)
    return keys


def _verdicts(
    members: Sequence[str] = V6_MEMBERS,
    overrides: Mapping[tuple[str, str, Key], bool | None] | None = None,
) -> dict[str, dict[str, dict[Key, bool | None]]]:
    """NP3-NP7 verdicts: every key passes, Vascular is not evaluable.

    ``overrides`` sets (criterion, member, key) to a verdict.
    """
    result: dict[str, dict[str, dict[Key, bool | None]]] = {}
    for criterion in gp.GATE_P_CLASS_CRITERIA:
        result[criterion] = {}
        for member in members:
            result[criterion][member] = {
                key: None if key[1] == "Vascular" else True for key in _keys()
            }
    for (criterion, member, key), value in (overrides or {}).items():
        result[criterion][member][key] = value
    return result


def _depths(
    members: Sequence[str] = V6_MEMBERS,
    values: Mapping[tuple[str, Key], tuple[bool | None, int | None, int | None]]
    | None = None,
) -> dict[str, pd.DataFrame]:
    """NP3 ``validated_min_depth`` tables: every key validated from 30, D_P 120."""
    tables: dict[str, pd.DataFrame] = {}
    for member in members:
        rows = []
        for key in _keys():
            passed, minimum, d_p = (
                (None, None, None) if key[1] == "Vascular" else (True, 30, 120)
            )
            passed, minimum, d_p = (values or {}).get(
                (member, key), (passed, minimum, d_p)
            )
            rows.append(
                {
                    "level": key[0],
                    "class": key[1],
                    "tested_max_depth": d_p,
                    "validated_min_depth": minimum,
                    "passed": passed,
                }
            )
        tables[member] = pd.DataFrame(
            {
                column: pd.Series([row[column] for row in rows], dtype=object)
                if column in ("tested_max_depth", "validated_min_depth", "passed")
                else [row[column] for row in rows]
                for column in rows[0]
            }
        )
    return tables


def _check(
    criterion: str,
    status: str = gp.CHECK_PASSED,
    *,
    members: Sequence[str] = V6_MEMBERS,
) -> gp.FamilyCheck:
    """A family check; NP9's detail records the members it was scored in."""
    detail: dict[str, Any] = {}
    if criterion == "NP9":
        detail["members"] = sorted(members)
    return gp.FamilyCheck(
        criterion=criterion, status=status, parts={"part": status}, detail=detail
    )


def _family_checks(
    members: Sequence[str] = V6_MEMBERS, **statuses: str
) -> dict[str, gp.FamilyCheck]:
    checks = {
        "NP1": _check("NP1"),
        "NP2": _check("NP2"),
        "NP8": _check("NP8", gp.CHECK_NOT_APPLICABLE),
        "NP9": _check("NP9", members=members),
    }
    for name, status in statuses.items():
        checks[name] = _check(name, status, members=members)
    return checks


def _assemble(
    members: Sequence[str] = V6_MEMBERS,
    *,
    overrides: Mapping[tuple[str, str, Key], bool | None] | None = None,
    depth_values: Mapping[tuple[str, Key], tuple[bool | None, int | None, int | None]]
    | None = None,
    class_sets: gp.ClassSets | None = None,
    checks: Mapping[str, gp.FamilyCheck] | None = None,
) -> gp.GatePResult:
    return gp.assemble_gate_p(
        family_id=FAMILY_ID,
        panel_hash=PANEL_HASH,
        species="human",
        resolvability_version=6 if len(members) == 1 else 7,
        verdicts=_verdicts(members, overrides),
        depths=_depths(members, depth_values),
        class_sets=class_sets or _class_sets(),
        family_checks=checks or _family_checks(members),
    )


def _record(result: gp.GatePResult, level: str, cls: str) -> pd.Series:
    rows = result.records[
        (result.records["level"] == level) & (result.records["class"] == cls)
    ]
    assert len(rows) == 1, (level, cls)
    return rows.iloc[0]


# --------------------------------------------------------------------------
# C_P (§14 class set)


def test_class_sets_take_classes_with_700_test_cells_and_need_90pct() -> None:
    shares = {"broad": {"Vascular": 0.012, "Exc": 0.4}}
    sets = gp.gate_p_class_sets(
        _test_cells(),
        species="human",
        default_group=None,
        settings=gp.ClassSetSettings.from_config(AnnotationResolvabilityConfig()),
        reference_shares=shares,
    )
    assert sets.level_names() == LEVELS
    assert sets.members("broad") == frozenset({"Exc", "Inh", "Astro", "Oligo"})
    level = sets.levels.set_index("level").loc["broad"]
    assert level["n_test_cells"] == 6800
    assert level["class_set_share"] == pytest.approx(6700 / 6800)
    assert bool(level["share_ok"]) and level["excluded_classes"] == "Vascular"
    vascular = sets.classes[
        (sets.classes["level"] == "broad") & (sets.classes["class"] == "Vascular")
    ].iloc[0]
    assert not vascular["in_class_set"] and vascular["n_test_cells"] == 100
    assert vascular["reference_share"] == pytest.approx(0.012)
    assert math.isnan(
        sets.classes[
            (sets.classes["level"] == "lineage") & (sets.classes["class"] == "Exc")
        ].iloc[0]["reference_share"]
    )
    # 699 cells stay out; a C_P below 90% of the level's cells fails the rule.
    thin = _class_sets(counts={"Exc": 3000, "Inh": 699, "Astro": 700})
    assert thin.members("broad") == frozenset({"Exc", "Astro"})
    assert thin.levels.set_index("level").loc["broad"]["class_set_share"] == (
        pytest.approx(3700 / 4399)
    )
    assert not thin.share_ok("broad")
    # The boundary: exactly 90% passes.
    edge = _class_sets(counts={"Exc": 900, "Inh": 100})
    assert edge.members("broad") == frozenset({"Exc"}) and edge.share_ok("broad")
    with pytest.raises(ValueError, match="no class set"):
        sets.members("cluster")


def test_class_sets_leave_cells_a_level_does_not_apply_to_out_of_nt() -> None:
    sets = _class_sets()
    level = sets.levels.set_index("level").loc["nt"]
    # The glial cells' NT truth is not_neuron: out of the counts and the
    # denominator, so NT has no row for them (D14 (a)) and its 90% rule is
    # read on the neurons.
    assert level["n_test_cells"] == 4500
    assert level["n_not_applicable"] == 2300
    assert sets.members("nt") == frozenset(NEURONS)
    assert set(sets.classes[sets.classes["level"] == "nt"]["class"]) == set(NEURONS)
    assert level["class_set_share"] == pytest.approx(1.0) and bool(level["share_ok"])


def test_class_sets_count_cells_without_a_class_in_the_denominator() -> None:
    sets = _class_sets(counts={"Exc": 900, "Inh": 900}, sinks=200)
    level = sets.levels.set_index("level").loc["broad"]
    assert level["n_no_class"] == 200 and level["n_test_cells"] == 2000
    # The stricter reading (§23.16): the sink cells stay in the denominator.
    assert level["class_set_share"] == pytest.approx(0.9)
    assert level["class_set_share_classed"] == pytest.approx(1.0)
    assert bool(level["share_ok"])
    below = _class_sets(counts={"Exc": 900, "Inh": 900}, sinks=201)
    assert not below.share_ok("broad")
    # At NT the sink cells' truth is not_neuron: the level does not apply.
    assert below.levels.set_index("level").loc["nt"]["n_no_class"] == 0
    assert below.share_ok("nt")


def test_class_sets_count_each_test_cell_once_and_drop_the_default_fit_half() -> None:
    settings = gp.ClassSetSettings(min_test_cells=700)
    repeated = gp.gate_p_class_sets(
        _test_cells({"Exc": 700}, rows_per_cell=3),
        species="human",
        default_group=None,
        settings=settings,
    )
    assert repeated.members("broad") == frozenset({"Exc"})
    assert repeated.levels.set_index("level").loc["broad"]["n_test_cells"] == 700
    # The default group's halves differ in size and class: the fit half
    # holds 800 Exc + 100 Inh cells, the check half 50 Exc + 750 Inh.
    default = _test_cells({"Exc": 850, "Inh": 850}, group="D1")
    index = default["cell_id"].str.split("_").str[-1].astype(int)
    is_exc = default["cell_id"].str.contains("_Exc_")
    default["half"] = np.where(is_exc, index < 50, index < 750).astype(int)
    # Another group's half 0 is not a fit half: its cells all count.
    other = _test_cells({"Astro": 300}, group="D2")
    other["half"] = 0
    pooled = pd.concat([default, other], ignore_index=True)
    scored = gp.gate_p_class_sets(
        pooled, species="human", default_group="D1", settings=settings
    )
    # The default group's fit half is out: its check half + 300 Astro.
    broad = scored.levels.set_index("level").loc["broad"]
    assert broad["n_test_cells"] == 1100
    counts = scored.classes[scored.classes["level"] == "broad"].set_index("class")
    assert counts["n_test_cells"].to_dict() == {"Astro": 300, "Exc": 50, "Inh": 750}
    assert scored.members("broad") == frozenset({"Inh"})
    assert broad["excluded_classes"] == "Astro;Exc"
    assert scored.levels.set_index("level").loc["nt"]["n_test_cells"] == 800
    unscoped = gp.gate_p_class_sets(
        pooled, species="human", default_group=None, settings=settings
    )
    assert unscoped.levels.set_index("level").loc["broad"]["n_test_cells"] == 2000
    assert unscoped.members("broad") == frozenset({"Exc", "Inh"})
    with pytest.raises(ValueError, match="group"):
        gp.gate_p_class_sets(
            _test_cells({"Exc": 10}),
            species="human",
            default_group="D1",
            settings=settings,
        )
    unknown_half = pooled.assign(half=np.where(pooled["group"] == "D1", np.nan, 0))
    with pytest.raises(ValueError, match="'half' values"):
        gp.gate_p_class_sets(
            unknown_half, species="human", default_group="D1", settings=settings
        )


def test_class_sets_refuse_two_truths_or_shared_cells() -> None:
    settings = gp.ClassSetSettings(min_test_cells=700)
    cells = _test_cells({"Exc": 3, "Inh": 3})
    clash = cells.copy()
    clash.loc[
        (clash["level"] == "broad") & (clash["cell_id"] == "g_Exc_0"), "truth_parent"
    ] = "Inh"
    with pytest.raises(ValueError, match="more than one truth"):
        gp.gate_p_class_sets(
            pd.concat([cells, clash]),
            species="human",
            default_group=None,
            settings=settings,
        )
    shared = pd.concat(
        [cells.assign(group="D1"), cells.assign(group="D2")], ignore_index=True
    )
    with pytest.raises(ValueError, match="more than one group"):
        gp.gate_p_class_sets(
            shared, species="human", default_group=None, settings=settings
        )
    # D2 (d): a fit-half cell of the default group in another group is a
    # shared cell too (the extra donors exclude every default test cell).
    default = cells.assign(group="D1", half=0)
    leaked = pd.concat(
        [default, cells[cells["cell_id"] == "g_Exc_0"].assign(group="D2", half=0)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="more than one group"):
        gp.gate_p_class_sets(
            leaked, species="human", default_group="D1", settings=settings
        )
    with pytest.raises(ValueError, match="truth_parent"):
        gp.gate_p_class_sets(
            cells.drop(columns="truth_parent"),
            species="human",
            default_group=None,
            settings=settings,
        )
    with pytest.raises(ValueError, match="species"):
        gp.gate_p_class_sets(
            cells, species="fish", default_group=None, settings=settings
        )
    with pytest.raises(ValueError, match="min_share"):
        gp.ClassSetSettings(min_test_cells=700, min_share=1.5)


def test_class_sets_count_shared_cells_once_under_the_d2_c_fallback() -> None:
    """D2 (c): shared cells count once; the default fit half is out everywhere."""
    settings = gp.ClassSetSettings(min_test_cells=700)
    # The default donor's own cells: 100 Exc in the fit half, 700 in the check.
    own = _test_cells({"Exc": 800}, group="D1")
    own["half"] = (own["cell_id"].str.split("_").str[-1].astype(int) >= 100).astype(int)
    # The shared other-region pool, in every replicate: 150 of its 400 Astro
    # cells are in the default donor's fit half.
    pool = _test_cells({"Astro": 400})
    pool_half = (pool["cell_id"].str.split("_").str[-1].astype(int) >= 150).astype(int)
    shared = pd.concat(
        [
            pool.assign(group="D1", half=pool_half),
            pool.assign(group="D2", half=0),
            pool.assign(group="D3", half=0),
        ],
        ignore_index=True,
    )
    other = _test_cells({"Inh": 500}, group="D2").assign(half=0)
    pooled = pd.concat([own, shared, other], ignore_index=True)
    with pytest.raises(ValueError, match="more than one group"):
        gp.gate_p_class_sets(
            pooled, species="human", default_group="D1", settings=settings
        )
    scored = gp.gate_p_class_sets(
        pooled,
        species="human",
        default_group="D1",
        settings=settings,
        disjoint=False,
    )
    counts = scored.classes[scored.classes["level"] == "broad"].set_index("class")
    # 250 Astro: each shared cell once, the 150 fit-half ones dropped from
    # D2 and D3 as well.
    assert counts["n_test_cells"].to_dict() == {"Astro": 250, "Exc": 700, "Inh": 500}
    assert scored.levels.set_index("level").loc["broad"]["n_test_cells"] == 1450
    assert scored.members("broad") == frozenset({"Exc"})
    # A default cell listed in both halves cannot be placed.
    both = pd.concat([own, own.assign(half=1 - own["half"])], ignore_index=True)
    with pytest.raises(ValueError, match="both halves"):
        gp.gate_p_class_sets(
            both,
            species="human",
            default_group="D1",
            settings=settings,
            disjoint=False,
        )


def test_class_set_settings_follow_the_config() -> None:
    settings = gp.ClassSetSettings.from_config(AnnotationResolvabilityConfig())
    assert (settings.min_test_cells, settings.min_share) == (700, 0.9)


# --------------------------------------------------------------------------
# Per-(level, class) records (§14 per-class records)


def test_one_failing_class_leaves_the_other_classes_of_its_level_validated() -> None:
    """§12 M13: one failing class leaves the other classes of its level validated."""
    result = _assemble(overrides={("NP6", "R1@0", ("supercluster", "Astro")): False})
    failed = _record(result, "supercluster", "Astro")
    assert failed["status"] == "failed:NP6"
    assert failed["validated_min_depth"] is None and failed["tested_max_depth"] is None
    for cls in (*NEURONS, "Oligo"):
        row = _record(result, "supercluster", cls)
        assert row["status"] == gp.RECORD_VALIDATED
        assert (row["validated_min_depth"], row["tested_max_depth"]) == (30, 120)
    # Astro is in C_P, so supercluster is incomplete and the headline is NT.
    assert result.validated_max_level == "nt" and result.passes


def test_status_names_the_lowest_failing_criterion_and_failure_beats_unevaluable() -> (
    None
):
    key = ("broad", "Oligo")
    result = _assemble(
        overrides={
            ("NP7", "R1@0", key): False,
            ("NP4", "R1@0", key): False,
            ("NP5", "R1@0", ("broad", "Astro")): None,
            ("NP3", "R1@0", ("broad", "Astro")): False,
        }
    )
    row = _record(result, *key)
    assert row["status"] == "failed:NP4"
    assert row["failed_criteria"] == "NP4;NP7"
    assert row["failed_members"] == "NP4:R1@0;NP7:R1@0"
    astro = _record(result, "broad", "Astro")
    assert astro["status"] == "failed:NP3" and astro["unevaluable_criteria"] == "NP5"
    vascular = _record(result, "broad", "Vascular")
    assert vascular["status"] == gp.RECORD_NOT_EVALUABLE
    assert not vascular["in_class_set"]
    assert vascular["unevaluable_criteria"] == "NP3;NP4;NP5;NP6;NP7"


def test_every_member_must_pass_and_an_unevaluable_member_leaves_it_unevaluable() -> (
    None
):
    key = ("supercluster", "Inh")
    other = ("supercluster", "Exc")
    result = _assemble(
        V7_MEMBERS,
        overrides={
            ("NP5", "R1@7", key): False,
            ("NP3", "R1@11", other): None,
            ("NP4", "R1@11", other): None,
            ("NP5", "R1@11", other): None,
            ("NP6", "R1@11", other): None,
            ("NP7", "R1@11", other): None,
        },
    )
    row = _record(result, *key)
    assert row["status"] == "failed:NP5" and row["failed_members"] == "NP5:R1@7"
    unevaluable = _record(result, *other)
    assert unevaluable["status"] == gp.RECORD_NOT_EVALUABLE
    assert unevaluable["unevaluable_members"].startswith("NP3:R1@11;")
    assert _record(result, "supercluster", "Oligo")["status"] == gp.RECORD_VALIDATED
    assert result.members == tuple(sorted(V7_MEMBERS))


def test_version_7_depths_take_the_deepest_minimum_and_the_shallowest_d_p() -> None:
    key = ("broad", "Exc")
    clamp = ("broad", "Inh")
    result = _assemble(
        V7_MEMBERS,
        depth_values={
            ("R1@6", key): (True, 60, 150),
            ("R1@7", key): (True, 10, 100),
            ("R1@6", clamp): (True, 100, 100),
            ("R1@7", clamp): (True, 10, 30),
        },
    )
    row = _record(result, *key)
    # Validated where every member validates it; extrapolated beyond the
    # shallowest member D_P.
    assert (row["validated_min_depth"], row["tested_max_depth"]) == (60, 100)
    assert "R1@6:60" in row["member_validated_min_depths"]
    assert "R1@7:100" in row["member_tested_max_depths"]
    clamped = _record(result, *clamp)
    # The shallowest D_P (30) lies below the deepest minimum (100): raised.
    assert (clamped["validated_min_depth"], clamped["tested_max_depth"]) == (100, 100)


def test_class_records_refuse_inputs_not_scored_on_the_same_members_and_keys() -> None:
    sets = _class_sets()
    verdicts = _verdicts()
    depths = _depths()
    with pytest.raises(ValueError, match="NP3"):
        gp.gate_p_class_records(
            {name: values for name, values in verdicts.items() if name != "NP6"},
            depths,
            sets,
        )
    with pytest.raises(ValueError, match="members"):
        gp.gate_p_class_records(verdicts, _depths(("R1@6",)), sets)
    other = _verdicts(("R1@0", "R1@6"))
    other["NP7"].pop("R1@6")
    with pytest.raises(ValueError, match="every emission member"):
        gp.gate_p_class_records(other, _depths(("R1@0", "R1@6")), sets)
    missing = _verdicts()
    missing["NP6"]["R1@0"].pop(("broad", "Exc"))
    with pytest.raises(ValueError, match="same tested sets"):
        gp.gate_p_class_records(missing, depths, sets)
    lacking = _depths(values={("R1@0", ("broad", "Exc")): (False, None, 120)})
    with pytest.raises(ValueError, match="no passing validated_min_depth"):
        gp.gate_p_class_records(verdicts, lacking, sets)


def test_class_records_keep_a_c_p_class_without_calls_not_evaluable() -> None:
    sets = _class_sets(counts={**COUNTS, "OPC": 800})
    result = _assemble(class_sets=sets)
    row = _record(result, "broad", "OPC")
    assert row["in_class_set"] and row["status"] == gp.RECORD_NOT_EVALUABLE
    assert row["unevaluable_members"].startswith("NP3:R1@0")
    # OPC is in C_P at every level but NT: the headline drops to nothing.
    assert result.validated_max_level is None and not result.passes


# --------------------------------------------------------------------------
# validated_max_level (§14 gate-P rule; the rank rule of _check_table)


def test_validated_max_level_walks_coarse_to_fine() -> None:
    full = _assemble()
    assert full.validated_max_level == "supercluster" and full.passes
    assert full.level_walk["counted"].tolist() == [True] * 4
    # An incomplete level stops the walk: a complete level above it does not
    # count (the rank rule covers every level of rank <= the headline).
    nt_fails = _assemble(overrides={("NP3", "R1@0", ("nt", "Inh")): False})
    assert nt_fails.validated_max_level == "broad"
    walk = nt_fails.level_walk.set_index("level")
    assert walk.loc["nt", "unvalidated_classes"] == "Inh"
    assert bool(walk.loc["supercluster", "complete"])
    assert not bool(walk.loc["supercluster", "counted"])
    # Below broad, the family does not pass (§14: at least broad, human).
    broad_fails = _assemble(overrides={("NP3", "R1@0", ("broad", "Astro")): False})
    assert broad_fails.validated_max_level == "lineage" and not broad_fails.passes
    assert any("below broad" in reason for reason in broad_fails.reasons)
    # A level whose C_P holds < 90% of its test cells is incomplete.
    # A class of C_P without a tested set (not evaluable) also leaves its
    # level incomplete.
    vascular = _assemble(class_sets=_class_sets(counts={**COUNTS, "Vascular": 800}))
    assert vascular.validated_max_level is None
    assert vascular.level_walk.set_index("level").loc["lineage", "share_ok"]
    sparse = _class_sets(counts={**COUNTS, "Vascular": 699}, sinks=200)
    assert not sparse.share_ok("lineage")
    unshared = _assemble(class_sets=sparse)
    assert unshared.validated_max_level is None and not unshared.passes
    assert not bool(unshared.level_walk.set_index("level").loc["lineage", "share_ok"])


def test_rank_rule_holds_with_non_neuronal_classes_at_nt(tmp_path: Path) -> None:
    """D14 (a): no NT row where NT does not apply; the headline reaches the leaf."""
    result = _assemble()
    nt_rows = result.records[result.records["level"] == "nt"]
    assert set(nt_rows["class"]) == set(NEURONS)
    assert result.validated_max_level == "supercluster"
    table = _write_family(result, tmp_path)
    records = table.level_records(FAMILY_ID)
    assert not [
        item for item in records if item.level == "nt" and item.class_name in GLIA
    ]
    decision = _decide(table)
    assert decision.state == "validated" and decision.validation_basis == "simulation"
    mask = decision.validated_mask(
        "nt", ["Exc", "Astro", "Inh"], [200, 200, 20], [True, True, True]
    )
    assert mask.tolist() == [True, False, False]
    assert decision.validated_mask(
        "supercluster", ["Astro"], [200], [True]
    ).tolist() == [True]


# --------------------------------------------------------------------------
# The family (§14 gate-P rule)


def test_family_passes_when_np1_np2_np9_pass_and_np8_passes_or_does_not_apply() -> None:
    assert _assemble().passes
    for name, status in (
        ("NP1", gp.CHECK_FAILED),
        ("NP2", gp.CHECK_PENDING),
        ("NP9", gp.CHECK_NOT_EVALUABLE),
        ("NP8", gp.CHECK_FAILED),
        ("NP8", gp.CHECK_NOT_EVALUABLE),
        ("NP1", gp.CHECK_NOT_APPLICABLE),
    ):
        result = _assemble(checks=_family_checks(**{name: status}))
        assert not result.passes, (name, status)
        assert any(reason.startswith(f"{name} {status}") for reason in result.reasons)
    assert _assemble(checks=_family_checks(NP8=gp.CHECK_PASSED)).passes


def test_assemble_refuses_inputs_of_another_family_shape() -> None:
    def call(**updates: Any) -> gp.GatePResult:
        arguments: dict[str, Any] = {
            "family_id": FAMILY_ID,
            "panel_hash": PANEL_HASH,
            "species": "human",
            "resolvability_version": 6,
            "verdicts": _verdicts(),
            "depths": _depths(),
            "class_sets": _class_sets(),
            "family_checks": _family_checks(),
        }
        arguments.update(updates)
        return gp.assemble_gate_p(**arguments)

    with pytest.raises(ValueError, match="1 emission member"):
        call(verdicts=_verdicts(("R1@0", "R1@6")), depths=_depths(("R1@0", "R1@6")))
    with pytest.raises(ValueError, match="8 emission member"):
        call(resolvability_version=7)
    with pytest.raises(ValueError, match="6 or 7"):
        call(resolvability_version=5)
    with pytest.raises(ValueError, match="sha256"):
        call(panel_hash="abc")
    with pytest.raises(ValueError, match="family checks"):
        call(family_checks={**_family_checks(), "NP8": _check("NP9")})
    with pytest.raises(ValueError, match="family checks"):
        call(family_checks={"NP1": _check("NP1")})
    mouse_sets = gp.ClassSets("mouse", _class_sets().classes, _class_sets().levels)
    with pytest.raises(ValueError, match="class sets"):
        call(class_sets=mouse_sets)


def test_assemble_refuses_np9_scored_on_other_members() -> None:
    """CHECK K13: NP9's re-runs cover the members the verdicts were scored in."""
    v7_checks = _family_checks(V7_MEMBERS)
    passing = _assemble(V7_MEMBERS, checks={**v7_checks, "NP9": _np9(V7_MEMBERS)})
    assert passing.passes
    assert passing.family_checks["NP9"].detail["members"] == sorted(V7_MEMBERS)
    # NP9 scored with one member's re-run passes on its own, but seven of the
    # eight version-7 members were never re-run.
    narrow = _np9(V6_MEMBERS)
    assert narrow.status == gp.CHECK_PASSED
    with pytest.raises(ValueError, match="NP9 was scored on the members"):
        _assemble(V7_MEMBERS, checks={**v7_checks, "NP9": narrow})
    with pytest.raises(ValueError, match="NP9 was scored on the members"):
        _assemble(checks={**_family_checks(), "NP9": _np9(V7_MEMBERS)})
    # An NP9 that does not record its members cannot be tied to them.
    unrecorded = gp.FamilyCheck(
        criterion="NP9", status=gp.CHECK_PASSED, parts={"part": "passed"}, detail={}
    )
    with pytest.raises(ValueError, match="does not record"):
        _assemble(checks={**_family_checks(), "NP9": unrecorded})
    with pytest.raises(ValueError, match="emission members"):
        _np9(())


def test_gate_p_levels_are_the_annotation_chain() -> None:
    assert gp.gate_p_levels("human") == LEVELS
    assert gp.gate_p_levels("mouse") == ("broad", "class", "nt", "subclass")
    with pytest.raises(ValueError, match="species"):
        gp.gate_p_levels("fish")


# --------------------------------------------------------------------------
# NP1 (§14 NP1)


def _gene_ids(
    *,
    share: float = 1.0,
    native: int = 491,
    status: str = "ok",
    species_check: str | None = "pass",
    unmapped: Mapping[str, str] | None = None,
) -> diag.GeneIdDiagnostics:
    return diag.GeneIdDiagnostics(
        name="P0001",
        platform="MERSCOPE",
        status=status,  # type: ignore[arg-type]
        refusal_reasons=("species_mismatch",) if status == "refused" else (),
        n_features_in=561,
        controls_removed={"name_pattern": 65},
        n_non_control=496,
        n_genes=496,
        resolution_share=share,
        resolved_by_source={"native": native, "fallback_table": 5}
        if native
        else {"pair_lookup": 496},
        unmapped=dict(unmapped or {}),
        species_check=species_check,
    )


def _np1(
    *gene_ids: diag.GeneIdDiagnostics,
    symbols: Sequence[str] = ("GFAP", "SNAP25"),
    **kwargs: Any,
) -> gp.FamilyCheck:
    return gp.np1_gene_ids(
        list(gene_ids),
        list(symbols),
        settings=gp.Np1Settings.from_config(AnnotationPanelConfig()),
        **kwargs,
    )


def test_np1_needs_98pct_when_the_vendor_supplies_ensembl_ids() -> None:
    assert _np1(_gene_ids()).status == gp.CHECK_PASSED
    vendor = _np1(_gene_ids(share=0.979))
    assert vendor.status == gp.CHECK_FAILED and vendor.parts["resolution"] == "failed"
    assert vendor.detail["declared_panels"][0]["min_resolution"] == 0.98
    symbols_only = _np1(_gene_ids(share=0.979, native=0))
    assert symbols_only.parts["resolution"] == gp.CHECK_PASSED
    assert _np1(_gene_ids(share=0.949, native=0)).parts["resolution"] == "failed"
    # Every declared panel must pass.
    assert _np1(_gene_ids(), _gene_ids(share=0.97)).status == gp.CHECK_FAILED


def test_np1_fails_a_control_name_that_reaches_the_query() -> None:
    check = _np1(_gene_ids(), symbols=("GFAP", "Blank-12", "NegControlProbe_00042"))
    assert check.status == gp.CHECK_FAILED and check.parts["controls"] == "failed"
    assert check.detail["controls_in_query"] == ["Blank-12", "NegControlProbe_00042"]
    assert check.detail["declared_panels"][0]["controls_removed"] == {
        "name_pattern": 65
    }
    assert _np1(_gene_ids(), symbols=()).parts["controls"] == gp.CHECK_NOT_EVALUABLE


def test_np1_species_test_and_refused_panels() -> None:
    assert _np1(_gene_ids(species_check="species_mismatch")).status == gp.CHECK_FAILED
    assert _np1(_gene_ids(species_check=None)).status == gp.CHECK_NOT_EVALUABLE
    assert _np1(_gene_ids(species_check="not_evaluable")).status == (
        gp.CHECK_NOT_EVALUABLE
    )
    refused = _np1(_gene_ids(status="refused"))
    assert refused.status == gp.CHECK_FAILED and refused.parts["declared"] == "failed"
    assert _np1().status == gp.CHECK_NOT_EVALUABLE


def test_np1_unresolved_list_is_pending_until_reviewed() -> None:
    gene_ids = _gene_ids(share=0.998, unmapped={"RGS5": "ambiguous"})
    pending = _np1(gene_ids)
    assert pending.status == gp.CHECK_PENDING and not pending.counts_as_pass
    assert pending.detail["unresolved"] == {"P0001:RGS5": "ambiguous"}
    assert _np1(gene_ids, unresolved_reviewed=True).status == gp.CHECK_PASSED
    # A mechanical failure outranks the pending review.
    assert (
        _np1(_gene_ids(share=0.9, unmapped={"RGS5": "ambiguous"})).status
        == gp.CHECK_FAILED
    )


def test_np1_settings_follow_the_config() -> None:
    settings = gp.Np1Settings.from_config(AnnotationPanelConfig())
    assert (settings.min_resolution, settings.min_resolution_vendor_ids) == (0.95, 0.98)
    with pytest.raises(ValueError, match="min_resolution"):
        gp.Np1Settings(min_resolution=1.2)


# --------------------------------------------------------------------------
# NP2 (§14 NP2)


def _children(*rows: tuple[str, int, int]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(gp.NP2_ROOT_CHILD_COLUMNS))


def _np2(**kwargs: Any) -> gp.FamilyCheck:
    arguments: dict[str, Any] = {
        "root_markers": 131,
        "root_children": _children(("Neurons", 9000, 40), ("Astro", 1200, 12)),
        "weak_parents": (),
        "collapsed_parents": (),
        "settings": gp.Np2Settings.from_config(AnnotationPanelConfig()),
    }
    arguments.update(kwargs)
    return gp.np2_panel_coverage(**arguments)


def test_np2_root_markers_and_root_children() -> None:
    assert _np2().status == gp.CHECK_PASSED
    assert _np2(root_markers=9).parts["root_markers"] == gp.CHECK_FAILED
    assert _np2(root_markers=10).status == gp.CHECK_PASSED
    short = _np2(root_children=_children(("Neurons", 9000, 40), ("Bergmann", 50, 9)))
    assert short.status == gp.CHECK_FAILED
    assert short.detail["root_children_short"] == ["Bergmann"]
    # A child below 50 cells is not judged.
    assert (
        _np2(root_children=_children(("Neurons", 9000, 40), ("Bergmann", 49, 0))).status
        == gp.CHECK_PASSED
    )
    assert _np2(root_children=_children()).status == gp.CHECK_NOT_EVALUABLE
    with pytest.raises(ValueError, match="more than once"):
        _np2(root_children=_children(("A", 60, 10), ("A", 60, 10)))
    with pytest.raises(ValueError, match="n_markers"):
        _np2(root_children=pd.DataFrame({"child": ["A"], "n": [60]}))


def test_np2_weak_and_collapsed_parents_are_pending_until_accepted() -> None:
    """M13 OPEN 4: accepting a weak or collapsed parent is the user's call."""
    listed = {
        "weak_parents": ["supercluster/Upper rhombic lip"],
        "collapsed_parents": ["supercluster/Bergmann glia"],
    }
    pending = _np2(**listed)
    assert pending.status == gp.CHECK_PENDING and not pending.counts_as_pass
    assert pending.detail["unaccepted_parents"] == [
        "supercluster/Bergmann glia",
        "supercluster/Upper rhombic lip",
    ]
    partly = _np2(**listed, accepted_parents=["supercluster/Bergmann glia", "other"])
    assert partly.status == gp.CHECK_PENDING
    assert partly.detail["accepted_not_listed"] == ["other"]
    accepted = _np2(
        **listed,
        accepted_parents=[
            "supercluster/Bergmann glia",
            "supercluster/Upper rhombic lip",
        ],
    )
    assert accepted.status == gp.CHECK_PASSED
    assert _np2(**listed, root_markers=3).status == gp.CHECK_FAILED


def test_np2_settings_follow_the_config() -> None:
    settings = gp.Np2Settings.from_config(AnnotationPanelConfig())
    assert (
        settings.min_root_markers,
        settings.weak_parent_markers,
        settings.root_child_min_n,
        settings.root_child_min_markers,
    ) == (10, 5, 50, 10)


# --------------------------------------------------------------------------
# NP8 (§14 NP8)


def test_np8_does_not_apply_to_an_unpaired_family() -> None:
    unpaired = gp.np8_cross_panel()
    assert unpaired.status == gp.CHECK_NOT_APPLICABLE and unpaired.counts_as_pass
    assert gp.np8_cross_panel({}).status == gp.CHECK_NOT_APPLICABLE
    assert gp.np8_cross_panel({"xenium_266": True}).status == gp.CHECK_PASSED
    failed = gp.np8_cross_panel({"xenium_266": True, "set_a": False})
    assert failed.status == gp.CHECK_FAILED and not failed.counts_as_pass
    assert gp.np8_cross_panel({"set_a": None}).status == gp.CHECK_NOT_EVALUABLE
    # Only NP8 may pass by not applying.
    assert not _check("NP9", gp.CHECK_NOT_APPLICABLE).counts_as_pass


# --------------------------------------------------------------------------
# NP9 (§14 NP9; M13 D10, CHECK K8, K13)


def _np9(members: Sequence[str] = V6_MEMBERS, **updates: Any) -> gp.FamilyCheck:
    values: dict[str, Any] = {
        "n_genes": 496,
        "wall_seconds": 1500.0,
        "reference_seconds": 1000.0,
        "reference_basis": "test",
        "peak_rss_gb": {"query_markers": 20.0, "resolvability": 6.0},
        "rss_reserve_gb": 64.0,
        "bundle_identical": True,
        "replicate_identical": {member: True for member in members},
    }
    values.update(updates)
    return gp.np9_resources(
        gp.Np9Inputs(**values),
        members=members,
        settings=gp.Np9Settings.from_config(AnnotationPanelConfig()),
    )


def test_np9_time_is_within_15x_of_the_reference() -> None:
    passed = _np9()
    assert passed.status == gp.CHECK_PASSED
    assert passed.parts["prefilter"] == gp.CHECK_NOT_APPLICABLE
    assert passed.detail["time_limit_seconds"] == pytest.approx(1500.0)
    assert _np9(wall_seconds=1500.1).parts["time"] == gp.CHECK_FAILED
    assert _np9(wall_seconds=None).status == gp.CHECK_NOT_EVALUABLE
    assert _np9(reference_seconds=None).parts["time"] == gp.CHECK_NOT_EVALUABLE
    # D10 (a): the set a version-7 dry run's time, scaled per simulated cell.
    reference = gp.np9_time_reference_v7(
        dry_run_seconds=1041.0,
        dry_run_simulated_cells=52_855,
        family_simulated_cells=105_710,
    )
    assert reference == pytest.approx(2082.0)
    assert _np9(wall_seconds=3123.0, reference_seconds=reference).parts["time"] == (
        gp.CHECK_PASSED
    )
    with pytest.raises(ValueError, match="dry_run_seconds"):
        gp.np9_time_reference_v7(
            dry_run_seconds=0, dry_run_simulated_cells=1, family_simulated_cells=1
        )
    with pytest.raises(ValueError, match="counts"):
        gp.np9_time_reference_v7(
            dry_run_seconds=1, dry_run_simulated_cells=0, family_simulated_cells=1
        )


def test_np9_rss_must_stay_within_the_reserve() -> None:
    over = _np9(peak_rss_gb={"query_markers": 64.5, "resolvability": 6.0})
    assert over.status == gp.CHECK_FAILED
    assert over.detail["rss_over_reserve"] == ["query_markers"]
    assert _np9(peak_rss_gb={"query_markers": 64.0}).parts["rss"] == gp.CHECK_PASSED
    assert _np9(peak_rss_gb={}).parts["rss"] == gp.CHECK_NOT_EVALUABLE
    assert _np9(rss_reserve_gb=None).parts["rss"] == gp.CHECK_NOT_EVALUABLE


def test_np9_identity_covers_the_bundle_and_one_replicate_per_member() -> None:
    """CHECK K13: PREP's bundle and one seed-0 replicate per member re-run."""
    assert _np9(V7_MEMBERS).parts["identity"] == gp.CHECK_PASSED
    assert _np9(bundle_identical=False).parts["identity"] == gp.CHECK_FAILED
    assert _np9(bundle_identical=None).parts["identity"] == gp.CHECK_NOT_EVALUABLE
    partial = _np9(V7_MEMBERS, replicate_identical={V7_MEMBERS[0]: True})
    assert partial.parts["identity"] == gp.CHECK_NOT_EVALUABLE
    assert len(partial.detail["members_without_rerun"]) == len(V7_MEMBERS) - 1
    differing = _np9(
        V7_MEMBERS,
        replicate_identical={**{member: True for member in V7_MEMBERS}, "R1@6": False},
    )
    assert differing.status == gp.CHECK_FAILED


def test_np9_prefilter_agreement_applies_above_1000_genes() -> None:
    """CHECK K8: NP9's prefilter agreement for panels above 1,000 genes."""
    assert _np9(n_genes=1000, prefiltered=True).parts["prefilter"] == (
        gp.CHECK_NOT_APPLICABLE
    )
    unfiltered = _np9(n_genes=5006, prefiltered=False)
    assert unfiltered.parts["prefilter"] == gp.CHECK_NOT_APPLICABLE
    assert _np9(n_genes=5006, prefiltered=True).parts["prefilter"] == (
        gp.CHECK_NOT_EVALUABLE
    )
    assert (
        _np9(n_genes=5006, prefiltered=True, prefilter_verdict={"passes": True}).status
        == gp.CHECK_PASSED
    )
    failing = _np9(n_genes=5006, prefiltered=True, prefilter_verdict={"passes": False})
    assert failing.status == gp.CHECK_FAILED


def test_np9_settings_follow_the_config() -> None:
    settings = gp.Np9Settings.from_config(AnnotationPanelConfig())
    assert (settings.large_panel_genes, settings.time_factor) == (1000, 1.5)
    with pytest.raises(ValueError, match="time_factor"):
        gp.Np9Settings(large_panel_genes=1000, time_factor=0)


# --------------------------------------------------------------------------
# The report


def test_report_lists_c_p_records_checks_and_the_scored_readings(
    tmp_path: Path,
) -> None:
    result = _assemble(
        overrides={("NP6", "R1@0", ("supercluster", "Astro")): False},
        checks={**_family_checks(), "NP9": _np9()},
    )
    np4 = pd.DataFrame(
        {"level": ["broad"] * 2, "donor_range": [0.02, 0.05], "passed": [True, False]}
    )
    np6 = pd.DataFrame(
        {
            "stress": ["spill", "spill", "lognormal"],
            "drop": [0.01, 0.04, np.nan],
            "passed": [True, True, True],
        }
    )
    written = gp.write_gate_p_report(
        result, tmp_path / "report", tables={"np4_sets": np4, "np6_sets": np6}
    )
    assert set(written) == {
        gp.GATE_P_REPORT_JSON,
        gp.GATE_P_REPORT_TXT,
        gp.GATE_P_RECORDS_CSV,
        gp.GATE_P_CLASS_SETS_CSV,
        gp.GATE_P_LEVEL_WALK_CSV,
        "np4_sets.csv",
        "np6_sets.csv",
    }
    report = json.loads(written[gp.GATE_P_REPORT_JSON].read_text())
    assert report["validated_max_level"] == "nt" and report["passes"]
    assert report["tables"]["np4_sets"] == {
        "n_rows": 2,
        "n_failed": 1,
        "max_donor_range": 0.05,
    }
    assert report["tables"]["np6_sets"]["max_drop_by_stress"] == {
        "lognormal": None,
        "spill": 0.04,
    }
    assert report["family_checks"]["NP9"]["detail"]["time_limit_seconds"] == 1500.0
    assert report["scored_readings"] == list(gp.GATE_P_SCORED_READINGS)
    assert report["readings_ruled"] == ["§23.19", "§23.21"]
    text = written[gp.GATE_P_REPORT_TXT].read_text()
    assert "PASSES gate P" in text
    assert "validated_max_level: nt" in text
    assert "excluded Vascular: 100 test cells" in text
    assert "supercluster/Astro (C_P): failed:NP6 failed NP6:R1@0" in text
    assert "Readings ruled: pre-registration §23.19, §23.21" in text
    records = pd.read_csv(written[gp.GATE_P_RECORDS_CSV])
    assert len(records) == len(result.records)
    with pytest.raises(FileExistsError, match="overwrite"):
        gp.write_gate_p_report(result, tmp_path / "report")
    gp.write_gate_p_report(result, tmp_path / "report", overwrite=True)
    with pytest.raises(ValueError, match="table name"):
        gp.write_gate_p_report(result, tmp_path / "other", tables={"Bad name": np4})
    with pytest.raises(ValueError, match="table name"):
        gp.write_gate_p_report(
            result, tmp_path / "other", tables={"gate_p_class_records": np4}
        )


def test_failing_report_names_the_reasons() -> None:
    result = _assemble(
        overrides={("NP3", "R1@0", ("lineage", "Exc")): False},
        checks=_family_checks(NP2=gp.CHECK_PENDING),
    )
    text = gp.gate_p_report_text(gp.gate_p_report(result))
    assert "does NOT pass gate P" in text
    assert "no level is validated for every class of C_P" in text
    assert "NP2 pending" in text
    json.dumps(gp.gate_p_report(result))


# --------------------------------------------------------------------------
# The validated-table rows of a passing family (§4.7; the gate-P PR)


def _write_family(result: gp.GatePResult, tmp_path: Path) -> diag.ValidatedPanelTable:
    record, levels = gp.validated_table_rows(
        result,
        panel_id="human_merscope_test_120",
        panel_role="sample_panel",
        platforms=["MERSCOPE"],
        n_genes=len(GENE_IDS),
        evidence="m13/gate_p/test",
        date="2026-10-07",
        approving_pr="#0",
        root_marker_source="test",
    )
    genes = diag.PanelGeneList(
        panel_id=record.panel_id,
        ensembl_ids=tuple(sorted(GENE_IDS)),
        symbols={gene: f"G{index}" for index, gene in enumerate(GENE_IDS)},
        root_markers=frozenset(GENE_IDS[:10]),
    )
    return diag.write_simulation_family(
        tmp_path / "tables",
        record,
        levels,
        genes,
        self_map=diag.ResolvabilityTrust(),
    )


def _decide(table: diag.ValidatedPanelTable) -> diag.TrustDecision:
    family = panel_family(
        GENE_IDS,
        species="human",
        platforms=["MERSCOPE"],
        known_families=table.known_families(),
    )
    return diag.trust_state(
        reference_id="whb_frontal_supc_clus",
        role="primary",
        species="human",
        panel_hash=PANEL_HASH,
        n_panel_genes=len(GENE_IDS),
        family=family,
        validated=table,
        coverage=diag.CoverageDiagnostics(
            reference_id="whb_frontal_supc_clus",
            n_panel_genes=len(GENE_IDS),
            n_query_genes_used=len(GENE_IDS),
            root_markers=50,
        ),
        resolvability=diag.ResolvabilityTrust(),
    )


def test_passing_family_rows_round_trip_through_the_reader(tmp_path: Path) -> None:
    result = _assemble(overrides={("NP6", "R1@0", ("supercluster", "Astro")): False})
    table = _write_family(result, tmp_path)
    reread = diag.load_validated_panels(
        tmp_path / "tables" / diag.VALIDATED_PANELS_FILE
    )
    assert reread.family_ids() == [FAMILY_ID]
    (record,) = reread.family_records(FAMILY_ID)
    assert record.validation_basis == "simulation"
    assert record.validated_max_level == "nt"
    levels = {
        (item.level, item.class_name): item for item in reread.level_records(FAMILY_ID)
    }
    assert len(levels) == len(result.records)
    assert levels[("supercluster", "Astro")].status == "failed:NP6"
    assert levels[("broad", "Vascular")].status == "not_evaluable"
    assert not levels[("broad", "Vascular")].in_class_set
    exc = levels[("supercluster", "Exc")]
    assert (exc.validated_min_depth, exc.tested_max_depth) == (30, 120)
    assert table.describe() == reread.describe() | {"source": table.source}
    # The trust state applies the records per (level, class): the failing
    # class stays provisional, the others of its level are validated.
    decision = _decide(reread)
    assert decision.is_validated("supercluster", "Exc", 30)
    assert not decision.is_validated("supercluster", "Exc", 29)
    assert not decision.is_validated("supercluster", "Astro", 500)
    assert decision.is_extrapolated("supercluster", "Exc", 250)


def test_validated_table_rows_refuse_a_family_that_does_not_pass() -> None:
    result = _assemble(checks=_family_checks(NP9=gp.CHECK_FAILED))
    with pytest.raises(ValueError, match="does not pass gate P"):
        gp.validated_table_rows(
            result,
            panel_id="p",
            panel_role="sample_panel",
            platforms=["MERSCOPE"],
            n_genes=120,
            evidence="e",
            date="2026-10-07",
            approving_pr="#0",
        )


# --------------------------------------------------------------------------
# The seeded families' dry run (§14 "Dry run"; M13 D28, CHECK K1)


def test_h18_expected_classes_count_each_test_cell_once_at_the_leaf() -> None:
    cells = _test_cells({"Exc": 60, "Astro": 49, "COP": 50}, rows_per_cell=3)
    assert gp.h18_expected_classes(cells, species="human") == {
        "supercluster": ["COP", "Exc"]
    }
    assert gp.h18_expected_classes(cells, species="human", min_test_cells=60) == {
        "supercluster": ["Exc"]
    }
    with pytest.raises(ValueError, match="missing columns"):
        gp.h18_expected_classes(cells.drop(columns="cell_id"), species="human")


def test_dry_run_needs_broad_for_c_p_and_supercluster_for_h18s_classes() -> None:
    key = ("supercluster", "Astro")
    passing = _assemble()
    h18 = {"supercluster": ["Astro", "Exc"]}
    verdict = gp.dry_run_verdict(passing, h18_classes=h18)
    assert verdict["passes"] and verdict["min_level"] == "broad"
    assert verdict["min_level_complete"] and verdict["failing"] == []
    expected = {(row["level"], row["class"]) for row in verdict["expected"]}
    assert expected == {
        *(("broad", cls) for cls in ("Exc", "Inh", "Astro", "Oligo")),
        ("supercluster", "Astro"),
        ("supercluster", "Exc"),
    }
    # A class H18 expects that fails at supercluster fails the dry run ...
    failing = _assemble(overrides={("NP4", "R1@0", key): False})
    verdict = gp.dry_run_verdict(failing, h18_classes=h18)
    assert not verdict["passes"]
    assert verdict["failing"] == [
        {
            "level": "supercluster",
            "class": "Astro",
            "status": "failed:NP4",
            "passed": False,
            "reported": False,
        }
    ]
    # ... unless the user removed it from H18's scope (D1: supercluster COP).
    exempt = gp.dry_run_verdict(
        failing, h18_classes=h18, exemptions=(("supercluster", "Astro"),)
    )
    assert exempt["passes"] and exempt["exempted"] == [["supercluster", "Astro"]]
    # A C_P class not validated at broad fails it, and so does an H18 class
    # without a record.
    broad = _assemble(overrides={("NP3", "R1@0", ("broad", "Oligo")): False})
    assert not gp.dry_run_verdict(broad, h18_classes={})["passes"]
    missing = gp.dry_run_verdict(passing, h18_classes={"supercluster": ["Fibro"]})
    assert missing["failing"][0]["status"] == "no_record"
