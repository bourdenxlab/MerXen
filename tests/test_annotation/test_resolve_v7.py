"""Tests of RESOLVE on resolvability version-7 bundles (M3c follow-up).

Human and mouse RESOLVE read version-7 bundles (plan §12 M3c, "Follow-up for
RESOLVE"; M13 chunk C14): the decisions are the version-7 ensemble's,
re-derived from ``resolvability_cells.parquet`` with each member's cells
reweighted to the dataset's composition, then the saturated-bp rule and the
monotone fill; cells of filled bins are ``resolvability_extrapolated``; the
report-only ``flag_nonneuronal_high_depth`` marks non-neuronal cells at
high depth in bins emitted on their own ensemble verdict; and the summary
records the class-depth prediction at the dataset's own per-class depth.
A version-6 bundle gives identical RESOLVE outputs apart from the new null
column (golden digests of the code before this change, floats to 10
significant digits), and a version this code does not know is refused.

The bundles' resolvability files are synthetic: per (member, level, class,
depth) a block of test cells with a planted share of correct calls, written
next to the fake MAP bundles of ``test_pipeline_resolve`` and
``conftest.mouse_setup``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import resolvability as res
from merxen.annotation.config import AnnotationConfig, AnnotationThresholds
from merxen.annotation.pipeline import (
    ResolveResult,
    annotate_resolve,
    read_label_table,
)
from merxen.annotation.schema import CellStatus, Columns, validate_label_table

from .conftest import MOUSE_SID, FakeMmc, map_mouse
from .test_pipeline_resolve import _config as _human_config
from .test_pipeline_resolve import _resolve as _human_resolve
from .test_pipeline_resolve import _setup as _human_setup
from .test_pipeline_resolve_mouse import PASSING
from .test_pipeline_resolve_mouse import _config as _mouse_config

MakeTrust = Callable[..., Any]

MEMBERS: tuple[str, ...] = ("R1_contam_HO@0", "R1_contam_HO@6", "R1_contam_HO@7")
HUMAN_GRID: tuple[int, ...] = (10, 30, 100, 250)
MOUSE_GRID: tuple[int, ...] = (10, 30, 60)
# Non-neuronal bins from this depth are never filled and are marked
# ``nonneuronal_high_depth`` (the bundle's ensemble setting; 1,000 in
# production, lowered to the synthetic sections' depths here).
HUMAN_HIGH_DEPTH = 250
MOUSE_HIGH_DEPTH = 60
# Each class's planted test cells: (cells per member, share of correct calls)
# per depth bin. The provisional target is 0.95 at >= 60 counts and 0.97
# below. Exc: emitted at every bin by every member. Astro: emitted at 10 and
# 30; at 100 its 60 check-half calls are 0.97 precise (Wilson bound 0.89, a
# power failure) and the monotone fill emits it; at 250 it is emitted on its
# own verdict, non-neuronal at the high-depth limit. Oligo: never precise
# enough.
HUMAN_PLAN: dict[str, dict[int, tuple[int, float]]] = {
    "Exc": {depth: (300, 1.0) for depth in HUMAN_GRID},
    "Astro": {10: (300, 1.0), 30: (300, 1.0), 100: (120, 0.97), 250: (300, 1.0)},
    "Oligo": {depth: (300, 0.8) for depth in HUMAN_GRID},
}
HUMAN_TRUTH: dict[str, str] = {"Exc": "CS_EXC", "Astro": "CS_AST", "Oligo": "CS_OLI"}
HUMAN_LEVELS: tuple[str, ...] = ("lineage", "broad", "nt", "supercluster")
HUMAN_NEURONAL: dict[str, bool] = {"Exc": True, "Astro": False, "Oligo": False}
MOUSE_PLAN: dict[str, dict[int, tuple[int, float]]] = {
    "01 IT-ET Glut": {depth: (300, 1.0) for depth in MOUSE_GRID},
    "19 MB Glut": {depth: (300, 1.0) for depth in MOUSE_GRID},
    "30 Astro-Epen": {10: (300, 1.0), 30: (300, 1.0), 60: (300, 1.0)},
}
MOUSE_TRUTH: dict[str, str] = {
    "01 IT-ET Glut": "CL_01_1",
    "19 MB Glut": "CL_19_1",
    "30 Astro-Epen": "CL_30_1",
}
MOUSE_NEURONAL: dict[str, bool] = {
    "01 IT-ET Glut": True,
    "19 MB Glut": True,
    "30 Astro-Epen": False,
}


def human_levels(thresholds: AnnotationThresholds) -> list[res.LevelMeta]:
    """The human bundle levels of ``resolvability.human_level_specs``."""
    return [
        res.LevelMeta(
            "lineage",
            "CCN202210140_SUPC",
            "lineage",
            thresholds.whb_broad,
            thresholds.target_lineage,
            None,
        ),
        res.LevelMeta(
            "broad",
            "CCN202210140_SUPC",
            "broad",
            thresholds.whb_broad,
            thresholds.target_broad,
            "broad",
        ),
        res.LevelMeta(
            "nt",
            "CCN202210140_SUPC",
            "nt",
            thresholds.whb_broad,
            thresholds.target_nt,
            "broad",
        ),
        res.LevelMeta(
            "supercluster",
            "CCN202210140_SUPC",
            "leaf",
            thresholds.whb_supercluster,
            thresholds.target_supercluster,
            "supercluster",
        ),
    ]


def mouse_levels(thresholds: AnnotationThresholds) -> list[res.LevelMeta]:
    """The mouse bundle levels of ``resolvability.mouse_level_specs``."""
    return [
        res.LevelMeta(
            "broad",
            "CCN20230722_CLAS",
            "broad",
            thresholds.wmb_class,
            thresholds.target_broad,
            "class",
        ),
        res.LevelMeta(
            "class",
            "CCN20230722_CLAS",
            "class",
            thresholds.wmb_class,
            thresholds.target_class,
            "class",
        ),
        res.LevelMeta(
            "nt",
            "CCN20230722_CLAS",
            "nt",
            thresholds.wmb_class,
            thresholds.target_nt,
            "class",
        ),
        res.LevelMeta(
            "subclass",
            "CCN20230722_SUBC",
            "leaf",
            thresholds.wmb_subclass,
            thresholds.target_subclass,
            "subclass",
        ),
    ]


def correct_pattern(n: int, share: float) -> np.ndarray:
    """Correctness with the same share on both split halves (cells 2k, 2k+1)."""
    ranks = np.arange(n) // 2
    return np.floor((ranks + 1) * share + 1e-9) > np.floor(ranks * share + 1e-9)


# A class's second truth type (truth leaf, cells per member, share of correct
# calls) per depth bin, beside its own ``truth`` leaf: reweighting to a
# dataset that lacks that type changes the class's precision.
SecondLeaves = Mapping[str, Mapping[int, tuple[str, int, float]]]
Plan = Mapping[str, Mapping[int, tuple[int, float]]]


def _block(
    meta: res.LevelMeta, cls: str, leaf: str, depth: int, n_cells: int, share: float
) -> pd.DataFrame:
    """Return one block of planted test cells of a (level, class, truth leaf)."""
    correct = correct_pattern(n_cells, share)
    ids = [f"{leaf}_{depth}_{index}" for index in range(n_cells)]
    return pd.DataFrame(
        {
            "recipe": res.DECISION_RECIPE,
            "seed": 0,
            "level": meta.level,
            "sim_id": [f"{cell}|D{depth}" for cell in ids],
            "cell_id": ids,
            "depth": depth,
            "half": np.arange(n_cells) % 2,
            "parent": cls,
            "call": np.where(correct, "right", "wrong"),
            "bp": np.full(n_cells, 0.95),
            "corr": 0.5,
            "truth": "right",
            "truth_parent": cls,
            res.TRUTH_LEAF_COLUMN: leaf,
            "correct": correct,
            "total_counts": float(depth),
        }
    )


def synthetic_cells(
    levels: Sequence[res.LevelMeta],
    plan: Plan,
    truth: Mapping[str, str],
    *,
    members: Sequence[str] | None,
    second_leaves: SecondLeaves | None = None,
    level_plans: Mapping[str, Plan] | None = None,
) -> pd.DataFrame:
    """Return a resolvability cells table (version 7 with ``members``).

    Every member simulates the same test cells (ids per truth leaf and
    depth); without members the table is a version-6 one of ``R1_contam_HO``.
    ``level_plans`` replaces ``plan`` at the levels it names.
    """
    frames = []
    for member in members or [None]:
        for meta in levels:
            for cls, depths in (level_plans or {}).get(meta.level, plan).items():
                for depth, (n_cells, share) in depths.items():
                    blocks = [(truth[cls], n_cells, share)]
                    second = (second_leaves or {}).get(cls, {}).get(depth)
                    if second is not None:
                        blocks.append(second)
                    for leaf, n_leaf, leaf_share in blocks:
                        frame = _block(meta, cls, leaf, depth, n_leaf, leaf_share)
                        if member is not None:
                            frame[res.MEMBER_COLUMN] = member
                            frame[res.MEMBER_ROLE_COLUMN] = "emission"
                        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_tables(
    bundle_dir: Path,
    *,
    version: object,
    levels: Sequence[res.LevelMeta],
    grid: Sequence[int],
    plan: Plan,
    truth: Mapping[str, str],
    neuronal: Mapping[str, bool],
    high_depth: int,
    second_leaves: SecondLeaves | None = None,
    level_plans: Mapping[str, Plan] | None = None,
) -> None:
    """Write synthetic resolvability files of a version into a bundle.

    Version 7 writes the ensemble's summary fields (members, ensemble
    settings, class lineage); any other value writes a version-6-like
    summary with that ``resolvability_version``.
    """
    v7 = version == res.RESOLVABILITY_VERSION_V7
    cells = synthetic_cells(
        levels,
        plan,
        truth,
        members=MEMBERS if v7 else None,
        second_leaves=second_leaves,
        level_plans=level_plans,
    )
    coerce = res.coerce_cells_v7 if v7 else res.coerce_cells
    coerce(cells).to_parquet(bundle_dir / res.RESOLVABILITY_CELLS_FILE, index=False)
    summary: dict[str, Any] = {
        "resolvability_version": version,
        "depth_grid": list(grid),
        "settings": res.RuleSettings().to_json(),
        "levels": [meta.to_json() for meta in levels],
        "decision_recipe": res.ENSEMBLE_RECIPE if v7 else res.DECISION_RECIPE,
        "recipes": [{"name": res.DECISION_RECIPE, "version": 1}],
        "d_max": {},
        "fine_level_seed_stability": {},
    }
    if v7:
        summary.update(
            {
                "emission_members": list(MEMBERS),
                "members": [{"member": name, "role": "emission"} for name in MEMBERS],
                "ensemble_settings": res.EnsembleSettings(
                    nonneuronal_monotone_max_depth=high_depth
                ).to_json(),
                "neuronal_classes": dict(neuronal),
            }
        )
    (bundle_dir / res.RESOLVABILITY_SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def write_human_tables(
    bundle_dir: Path,
    version: object = 7,
    *,
    second_leaves: SecondLeaves | None = None,
) -> None:
    """Write the synthetic human resolvability files of a version."""
    write_tables(
        bundle_dir,
        version=version,
        levels=human_levels(AnnotationThresholds()),
        grid=HUMAN_GRID,
        plan=HUMAN_PLAN,
        truth=HUMAN_TRUTH,
        neuronal=HUMAN_NEURONAL,
        high_depth=HUMAN_HIGH_DEPTH,
        second_leaves=second_leaves,
    )


def _human_v7_config(min_counts: int = HUMAN_HIGH_DEPTH) -> AnnotationConfig:
    config = _human_config()
    real_qc = config.real_qc.model_copy(
        update={"nonneuronal_high_depth_counts": min_counts}
    )
    return config.model_copy(update={"real_qc": real_qc})


def _decision_lookup(
    decisions: pd.DataFrame, regime: str
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    frame = decisions[decisions["regime"] == regime]
    return {
        (str(record["level"]), str(record["class"]), int(record["depth"])): record
        for record in frame.to_dict("records")
    }


# --------------------------------------------------------------------------
# The synthetic ensemble decides as planted


def test_the_synthetic_version_7_tables_plant_every_route(tmp_path: Path) -> None:
    write_human_tables(tmp_path)
    tables = res.load_resolvability(tmp_path, allow_version_7=True)
    assert tables is not None and tables.version == res.RESOLVABILITY_VERSION_V7
    lookup = _decision_lookup(tables.decisions(), "provisional")
    for depth in HUMAN_GRID:
        exc = lookup[("broad", "Exc", depth)]
        assert exc["status"] == res.STATUS_EMITTED
        assert exc["ensemble_rule"] == res.RULE_UNANIMOUS
        assert lookup[("broad", "Oligo", depth)]["status"] == res.STATUS_NOT_RESOLVABLE
    filled = lookup[("broad", "Astro", 100)]
    assert filled["status"] == res.STATUS_EMITTED
    assert bool(filled["monotone_filled"]) and bool(filled["extrapolated"])
    high = lookup[("broad", "Astro", 250)]
    assert high["status"] == res.STATUS_EMITTED
    assert bool(high["nonneuronal_high_depth"]) and not bool(high["monotone_filled"])
    assert not bool(lookup[("broad", "Exc", 250)]["nonneuronal_high_depth"])


def test_the_emission_plan_of_a_version_7_bundle(
    tmp_path: Path, make_trust: MakeTrust
) -> None:
    from merxen.annotation.thresholds import EmissionPlan

    thresholds = AnnotationThresholds()
    plans = {}
    for version in (6, 7):
        directory = tmp_path / f"v{version}"
        directory.mkdir()
        write_human_tables(directory, version)
        tables = res.load_resolvability(directory, allow_version_7=True)
        plans[version] = EmissionPlan.from_tables(
            tables,
            species="human",
            thresholds=thresholds,
            trust=make_trust("provisional"),
        )
    v6, v7 = plans[6], plans[7]
    assert v6.resolvability_version == 6 and not v6.is_version_7
    assert v6.class_depth() is None
    assert v7.resolvability_version == 7 and v7.is_version_7
    assert v7.decisions is not None
    class_depth = v7.class_depth()
    assert class_depth is not None
    assert list(class_depth.columns) == list(res.CLASS_DEPTH_COLUMNS)
    pd.testing.assert_frame_equal(
        class_depth, res.class_depth_table(v7.decisions, grid=HUMAN_GRID)
    )
    filled = class_depth[class_depth["monotone_filled"].astype(bool)]
    assert len(filled) == 3 and set(filled["depth"]) == {100}
    # Floors come from the decisions before the fill (v7.9).
    assert v7.simulated_floors() == res.simulated_floors(
        res.unfilled_decisions(v7.decisions), "provisional"
    )
    assert v7.simulated_floors()[("broad", "Astro")] == 10
    # Cells of a filled bin are emitted and extrapolated.
    emission = v7.level(
        "broad",
        np.array(["Astro", "Astro", "Astro", "Oligo"], dtype=object),
        np.array([30.0, 150.0, 400.0, 150.0]),
    )
    assert emission.emitted.tolist() == [True, True, True, False]
    assert emission.extrapolated.tolist() == [False, True, False, False]
    plain = EmissionPlan(species="human", thresholds=thresholds)
    assert not plain.is_version_7 and plain.class_depth() is None


# --------------------------------------------------------------------------
# Human RESOLVE on a version-7 bundle


def test_human_resolve_applies_the_version_7_ensemble_decisions(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    setup = _human_setup(tmp_path, fake_mmc)
    write_human_tables(setup.bundles["whb_frontal_supc_clus"].path)
    result = _human_resolve(
        setup, make_trust, state="provisional", config=_human_v7_config()
    )
    n_flagged = 0
    for sample in result.samples.values():
        assert sample.summary["reweighted_to_composition"] is True
        labels, _ = read_label_table(sample.labels_path)
        validate_label_table(labels, "human")
        table = labels[Columns.IN_TABLE].to_numpy(bool)
        counts = labels[Columns.TOTAL_COUNTS].to_numpy(np.float64)
        bins = res.depth_bin(counts, list(HUMAN_GRID))
        names = labels["mmc_whb_supercluster_name"].astype(str).to_numpy()
        lineage = labels[Columns.level("lineage", "status")].astype(str).to_numpy()
        summary = sample.summary["resolvability_v7"]
        assert summary["resolvability_version"] == 7
        assert summary["decision_recipe"] == res.ENSEMBLE_RECIPE
        assert summary["emission_members"] == list(MEMBERS)
        # RESOLVE's decisions: each member reweighted to the sample, the
        # ensemble rule, then the monotone fill: Astro at 100 is filled at
        # lineage, broad and NT; at supercluster (provisional target 0.90) its
        # Wilson bound 0.886 clears 0.88 and it is emitted outright.
        applied = summary["applied_decisions"]["provisional"]
        assert applied["n_filled"] == 3
        assert applied["n_unanimous"] >= 1 and applied["n_spread"] == 0
        assert applied["n_nonneuronal_high_depth_emitted"] == len(HUMAN_LEVELS)
        # Astrocytes at 100-249 counts are emitted through the filled bin and
        # extrapolated when confident; Oligodendrocytes are never resolvable.
        astro = table & (names == "Astrocyte")
        filled = astro & (bins == 100)
        assert filled.any()
        assert CellStatus.NOT_RESOLVABLE.value not in set(lineage[filled])
        confident = lineage == CellStatus.CONFIDENT.value
        assert (filled & confident).any()
        extrapolated = labels[Columns.RESOLVABILITY_EXTRAPOLATED].to_numpy(bool)
        assert extrapolated[filled & confident].all()
        assert not extrapolated[astro & (bins != 100)].any()
        oligo = table & (names == "Oligodendrocyte")
        assert set(lineage[oligo]) == {CellStatus.NOT_RESOLVABLE.value}
        # The report-only flag: defined on table cells, null outside; it marks
        # exactly the non-neuronal cells at >= 250 counts in the emitted
        # high-depth bins, and changes no status.
        flag = labels[Columns.FLAG_NONNEURONAL_HIGH_DEPTH]
        assert str(flag.dtype) == "boolean"
        assert flag[table].notna().all() and flag[~table].isna().all()
        expected = astro & (counts >= HUMAN_HIGH_DEPTH)
        assert flag.fillna(False).to_numpy(bool).tolist() == expected.tolist()
        assert CellStatus.NOT_RESOLVABLE.value not in set(lineage[expected])
        n_flagged += int(expected.sum())
        record = summary["nonneuronal_high_depth"]
        assert record["min_counts"] == HUMAN_HIGH_DEPTH
        assert record["n_flagged"] == int(expected.sum())
        assert record["per_level"]["broad"] == int(expected.sum())
        # The class-depth prediction at the sample's own per-class depth
        # (label-free for classes with fewer than 100 cells).
        prediction = {
            (item["level"], item["class"]): item
            for item in summary["class_depth_prediction"]["classes"]
        }
        assert summary["class_depth_prediction"]["min_class_cells"] == 100
        astro_broad = prediction[("broad", "Astro")]
        assert astro_broad["regime"] == "provisional"
        assert astro_broad["n_cells"] == int(astro.sum())
        assert astro_broad["share_source"] == (
            "own" if astro.sum() >= 100 else "label_free"
        )
        assert astro_broad["resolvable_share"] == pytest.approx(1.0)
        assert prediction[("broad", "Oligo")]["resolvable_share"] == 0.0
        assert prediction[("broad", "Oligo")]["predicted_coverage"] == 0.0
    assert n_flagged > 0


# The real names of the synthetic sections' glial calls and their class keys
# (the per-cell checks; ``test_pipeline_resolve.WHB``).
GLIAL_KEYS: dict[str, str] = {"Astrocyte": "Astro", "Oligodendrocyte": "Oligo"}


def _without_reweighting(config: AnnotationConfig) -> AnnotationConfig:
    """Return a config whose RESOLVE keeps PREP's unweighted decisions."""
    resolvability = config.resolvability.model_copy(
        update={"reweight_to_composition": False}
    )
    return config.model_copy(update={"resolvability": resolvability})


def _check_lineage_per_cell(
    labels: pd.DataFrame, lookup: Mapping[tuple[str, str, int], Mapping[str, Any]]
) -> int:
    """Check each glial table cell's lineage emission against decisions.

    A cell is ``not_resolvable`` exactly when its (lineage, class, bin) is
    not emitted, and a confident cell is ``resolvability_extrapolated``
    exactly when its bin is.

    Returns:
        The number of cells checked.
    """
    table = labels[Columns.IN_TABLE].to_numpy(bool)
    bins = res.depth_bin(
        labels[Columns.TOTAL_COUNTS].to_numpy(np.float64), list(HUMAN_GRID)
    )
    names = labels["mmc_whb_supercluster_name"].astype(str).to_numpy()
    status = labels[Columns.level("lineage", "status")].astype(str).to_numpy()
    extrapolated = labels[Columns.RESOLVABILITY_EXTRAPOLATED].to_numpy(bool)
    checked = 0
    for index in np.flatnonzero(table & np.isfinite(bins)):
        cls = GLIAL_KEYS.get(names[index])
        if cls is None:
            continue
        record = lookup[("lineage", cls, int(bins[index]))]
        emitted = record["status"] == res.STATUS_EMITTED
        assert (status[index] == CellStatus.NOT_RESOLVABLE.value) == (not emitted)
        if status[index] == CellStatus.CONFIDENT.value:
            assert extrapolated[index] == bool(record["extrapolated"])
        checked += 1
    return checked


def _section_composition(
    setup: Any, sample_id: str, labels: pd.DataFrame, config: AnnotationConfig
) -> res.DatasetComposition:
    """Return the composition RESOLVE reweights a section to, from its MAP output.

    ``composition.dataset_type_composition`` of the WHB supercluster level of
    the section's tidy MMC table (the table cells' assigned nodes and
    runner-ups) at their total counts, on the bundle's grid.
    """
    from merxen.annotation.composition import dataset_type_composition
    from merxen.annotation.mapmycells_engine import level_frame

    platform = str(labels[Columns.PLATFORM].iloc[0]).lower()
    tidy = pd.read_parquet(
        setup.map_dir / platform / f"{sample_id}_mmc_whb_frontal_supc_clus.parquet"
    )
    leaf = level_frame(tidy, "CCN202210140_SUPC")
    leaf.index = leaf.index.astype(str)
    table = labels[labels[Columns.IN_TABLE].to_numpy(bool)]
    return dataset_type_composition(
        leaf.reindex(table[Columns.CELL_ID].astype(str).to_numpy()),
        table[Columns.TOTAL_COUNTS].to_numpy(np.float64),
        list(HUMAN_GRID),
        min_bin_mass=float(config.resolvability.composition_min_bin_cells),
    )


def test_human_resolve_of_a_version_7_bundle_follows_its_decisions_per_cell(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """Without reweighting, each cell's lineage emission follows PREP's decisions."""
    from merxen.annotation.thresholds import EmissionPlan

    setup = _human_setup(tmp_path, fake_mmc)
    bundle_dir = setup.bundles["whb_frontal_supc_clus"].path
    write_human_tables(bundle_dir)
    config = _without_reweighting(_human_v7_config())
    result = _human_resolve(setup, make_trust, state="provisional", config=config)
    tables = res.load_resolvability(bundle_dir, allow_version_7=True)
    assert tables is not None
    plan = EmissionPlan.from_tables(
        tables,
        species="human",
        thresholds=config.thresholds,
        trust=make_trust("provisional"),
    )
    assert plan.is_version_7 and plan.decisions is not None
    lookup = _decision_lookup(plan.decisions, "provisional")
    checked = 0
    for sample in result.samples.values():
        assert sample.summary["reweighted_to_composition"] is False
        labels, _ = read_label_table(sample.labels_path)
        checked += _check_lineage_per_cell(labels, lookup)
    assert checked > 0


# Astro at 30-99 counts gets a second truth type, one the synthetic sections
# lack, with poor calls (share 0.6). On PREP's unweighted test cells the bin's
# precision is 0.8 and it is not emitted; reweighted to a section, that type
# weighs nothing (absent from the composition) and the bin is Astrocyte's own,
# precise and emitted. Every other bin is unchanged by reweighting.
ABSENT_LEAF = "CS_AST_ABSENT"
REWEIGHT_SECOND_LEAVES: dict[str, dict[int, tuple[str, int, float]]] = {
    "Astro": {30: (ABSENT_LEAF, 300, 0.6)}
}


def test_human_resolve_reweights_the_version_7_members_to_each_section(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """RESOLVE applies the ensemble decisions reweighted to each section.

    Cell by cell, the lineage statuses follow ``ResolvabilityTables.decisions``
    reweighted to the section's own composition (recomputed here from its MAP
    output), and the planted bin that only reweighting emits differs from a
    run without reweighting, which follows the unweighted decisions.
    """
    setup = _human_setup(tmp_path, fake_mmc)
    bundle_dir = setup.bundles["whb_frontal_supc_clus"].path
    write_human_tables(bundle_dir, second_leaves=REWEIGHT_SECOND_LEAVES)
    config = _human_v7_config()
    weighted = _human_resolve(
        setup, make_trust, "weighted", state="provisional", config=config
    )
    plain = _human_resolve(
        setup,
        make_trust,
        "plain",
        state="provisional",
        config=_without_reweighting(config),
    )
    tables = res.load_resolvability(bundle_dir, allow_version_7=True)
    assert tables is not None
    unweighted = _decision_lookup(tables.decisions(), "provisional")
    for level in HUMAN_LEVELS:
        assert unweighted[(level, "Astro", 30)]["status"] == res.STATUS_NOT_RESOLVABLE
        assert unweighted[(level, "Astro", 10)]["status"] == res.STATUS_EMITTED
    flipped = 0
    for sample_id, sample in weighted.samples.items():
        assert sample.summary["reweighted_to_composition"] is True
        labels, _ = read_label_table(sample.labels_path)
        other, _ = read_label_table(plain.samples[sample_id].labels_path)
        assert plain.samples[sample_id].summary["reweighted_to_composition"] is False
        composition = _section_composition(setup, sample_id, labels, config)
        assert ABSENT_LEAF not in composition.overall
        decisions = tables.decisions(composition=composition)
        expected = _decision_lookup(decisions, "provisional")
        for level in HUMAN_LEVELS:
            assert expected[(level, "Astro", 30)]["status"] == res.STATUS_EMITTED
        # Only the planted bin moves: per level, one more emitted bin.
        moved = {
            key
            for key, record in expected.items()
            if record["status"] != unweighted[key]["status"]
        }
        assert moved == {(level, "Astro", 30) for level in HUMAN_LEVELS}
        applied = sample.summary["resolvability_v7"]["applied_decisions"]
        applied_plain = plain.samples[sample_id].summary["resolvability_v7"][
            "applied_decisions"
        ]
        assert applied["provisional"]["n_emitted"] == (
            applied_plain["provisional"]["n_emitted"] + len(HUMAN_LEVELS)
        )
        assert _check_lineage_per_cell(labels, expected) > 0
        assert _check_lineage_per_cell(other, unweighted) > 0
        # The planted bin's Astrocytes: not resolvable without reweighting,
        # emitted (and not extrapolated) with it.
        table = labels[Columns.IN_TABLE].to_numpy(bool)
        bins = res.depth_bin(
            labels[Columns.TOTAL_COUNTS].to_numpy(np.float64), list(HUMAN_GRID)
        )
        names = labels["mmc_whb_supercluster_name"].astype(str).to_numpy()
        planted = table & (names == "Astrocyte") & (bins == 30)
        status = labels[Columns.level("lineage", "status")].astype(str).to_numpy()
        status_plain = other[Columns.level("lineage", "status")].astype(str).to_numpy()
        assert set(status_plain[planted]) == {CellStatus.NOT_RESOLVABLE.value}
        assert CellStatus.NOT_RESOLVABLE.value not in set(status[planted])
        flipped += int(planted.sum())
    assert flipped > 0


def test_the_version_7_flag_follows_the_configured_depth_and_changes_nothing(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust
) -> None:
    """``real_qc.nonneuronal_high_depth_counts`` sets the flag's depth limit;
    the flag is report-only: every other column is the same either way."""
    setup = _human_setup(tmp_path, fake_mmc)
    write_human_tables(setup.bundles["whb_frontal_supc_clus"].path)
    default = _human_resolve(
        setup, make_trust, "default", state="provisional", config=_human_v7_config()
    )
    strict = _human_resolve(
        setup,
        make_trust,
        "strict",
        state="provisional",
        config=_human_v7_config(min_counts=10_000),
    )
    flagged = 0
    for sample_id, sample in strict.samples.items():
        labels, _ = read_label_table(sample.labels_path)
        other, _ = read_label_table(default.samples[sample_id].labels_path)
        flag = labels[Columns.FLAG_NONNEURONAL_HIGH_DEPTH]
        table = labels[Columns.IN_TABLE].to_numpy(bool)
        assert not flag[table].any()
        assert (
            sample.summary["resolvability_v7"]["nonneuronal_high_depth"]["n_flagged"]
            == 0
        )
        flagged += int(other[Columns.FLAG_NONNEURONAL_HIGH_DEPTH].fillna(False).sum())
        pd.testing.assert_frame_equal(
            labels.drop(columns=[Columns.FLAG_NONNEURONAL_HIGH_DEPTH]),
            other.drop(columns=[Columns.FLAG_NONNEURONAL_HIGH_DEPTH]),
        )
    assert flagged > 0


# Per level, the class keys of a COP cell differ (human: ``OPC`` at lineage,
# broad and NT, where COP sits inside OPC; ``COP`` at supercluster). Planted:
# ``OPC`` is emitted, and marked non-neuronal high depth at 250, only at
# supercluster; at the other levels ``OPC`` and at supercluster ``COP`` are
# never emitted.
OPC_TRUTH: dict[str, str] = {"Exc": "CS_EXC", "OPC": "CS_OPC", "COP": "CS_COP"}
OPC_NEURONAL: dict[str, bool] = {"Exc": True, "OPC": False, "COP": False}
OPC_PLAN: dict[str, dict[int, tuple[int, float]]] = {
    "Exc": {depth: (300, 1.0) for depth in HUMAN_GRID},
    "OPC": {depth: (300, 0.8) for depth in HUMAN_GRID},
}
OPC_SUPERCLUSTER_PLAN: dict[str, dict[int, tuple[int, float]]] = {
    "Exc": {depth: (300, 1.0) for depth in HUMAN_GRID},
    "OPC": {depth: (300, 1.0) for depth in HUMAN_GRID},
    "COP": {depth: (300, 0.8) for depth in HUMAN_GRID},
}


def test_the_version_7_flag_reads_each_level_with_its_own_class_key(
    tmp_path: Path, make_trust: MakeTrust
) -> None:
    """A level's class key is looked up in that level's bins only.

    A COP cell (``OPC`` at broad, ``COP`` at supercluster) at 300 counts is
    not flagged: ``OPC`` is a marked, emitted high-depth bin at supercluster
    only, where the cell's key is ``COP``. Matching its broad key against
    every level's bins would flag it.
    """
    from types import SimpleNamespace

    from merxen.annotation.pipeline import version_7_outputs
    from merxen.annotation.thresholds import EmissionPlan

    thresholds = AnnotationThresholds()
    write_tables(
        tmp_path,
        version=7,
        levels=human_levels(thresholds),
        grid=HUMAN_GRID,
        plan=OPC_PLAN,
        truth=OPC_TRUTH,
        neuronal=OPC_NEURONAL,
        high_depth=HUMAN_HIGH_DEPTH,
        level_plans={"supercluster": OPC_SUPERCLUSTER_PLAN},
    )
    tables = res.load_resolvability(tmp_path, allow_version_7=True)
    assert tables is not None
    emission = EmissionPlan.from_tables(
        tables,
        species="human",
        thresholds=thresholds,
        trust=make_trust("provisional"),
    )
    class_depth = emission.class_depth()
    assert class_depth is not None
    marked = class_depth[
        (class_depth["regime"] == "provisional")
        & (class_depth["status"] == res.STATUS_EMITTED)
        & class_depth["nonneuronal_high_depth"].fillna(False).astype(bool)
    ]
    assert set(zip(marked["level"], marked["class"], strict=True)) == {
        ("supercluster", "OPC")
    }
    # Cells: COP (300 counts), OPC (300), OPC (120), Exc (300), and an OPC
    # cell outside the table.
    broad_key = np.array(["OPC", "OPC", "OPC", "Exc", "OPC"], dtype=object)
    supercluster_key = np.array(["COP", "OPC", "OPC", "Exc", "OPC"], dtype=object)
    levels = {
        level: SimpleNamespace(
            class_key=supercluster_key if level == "supercluster" else broad_key
        )
        for level in HUMAN_LEVELS
    }
    outputs = version_7_outputs(
        tables,
        emission,
        levels,  # type: ignore[arg-type]
        np.array([300.0, 300.0, 120.0, 300.0, 300.0]),
        np.array([True, True, True, True, False]),
        _human_v7_config(),
    )
    assert outputs.flag.tolist() == [False, True, False, False, pd.NA]
    assert outputs.summary is not None
    record = outputs.summary["nonneuronal_high_depth"]
    assert record["n_flagged"] == 1
    assert record["per_level"] == {
        "lineage": 0,
        "broad": 0,
        "nt": 0,
        "supercluster": 1,
    }


# --------------------------------------------------------------------------
# Version 6 is unchanged; an unknown version is refused


# Digests of RESOLVE's outputs in the scenarios of ``_v6_scenario``, computed
# with the code before the version-7 consumer (the integration branch at
# 57c8488) by ``_v6_digests``: each sample's label table without
# ``flag_nonneuronal_high_depth`` (which that code did not write), its
# provenance and summary, and the pair summary (the test's temporary
# directory, the MAP manifest digest and ``VOLATILE_KEYS`` left out). A later
# deliberate change of RESOLVE's output updates them in its own commit, with
# the reason. M13 C15 (real-data QC) adds only the summary keys of
# ``C15_SUMMARY_KEYS`` and the schema version 3 with them: the QC-free run
# (``real_qc.enabled`` false) is compared with these digests after
# ``_without_c15`` removes them, and the QC run's label tables directly.
V6_GOLDEN: dict[str, dict[str, str]] = {
    "human_v6_provisional": {
        "PX_MERSCOPE/labels": (
            "0319f97179eaab6e874a0e42418dc6c938987e30b358650ffbaf473c252b2990"
        ),
        "PX_MERSCOPE/provenance": (
            "754b7104dd33397ca86294e02f1dbccb43a38f5542b4cc855edf6c92470e1021"
        ),
        "PX_MERSCOPE/summary": (
            "0c07a4dd0bf89940eb13283006ecb22805e7bfd3007007994e4a387ac089d00e"
        ),
        "PX_XENIUM/labels": (
            "3a66a07b79bec02af5bf88cb8f2c7ab62c2ebc6cc3c9f9ef7156254da8ea3867"
        ),
        "PX_XENIUM/provenance": (
            "613de9d8aa9f1222b499e6216b82c1040f082eb1a0f552574e4db9be6de8ef93"
        ),
        "PX_XENIUM/summary": (
            "8f249a6e9186e478b3b315647b931b8341094e522facba66b928ec9e036d3170"
        ),
        "pair/summary": (
            "60f6f0b745c66dacc65749899f5e619022a877ec1436e1bf2d72b705e7a8fe20"
        ),
    },
    "human_v6_validated": {
        "PX_MERSCOPE/labels": (
            "3e012efa486dbdd5117bbe762823510c357347d2bd39eccfbde8b2f5876d2ccd"
        ),
        "PX_MERSCOPE/provenance": (
            "c4a66355edfeb9143edad3a5416be7a042d9cd61b02c6d1f394b39b329b1248e"
        ),
        "PX_MERSCOPE/summary": (
            "746a197ea7c40654c90dd0cddad75a3149f74a081d38e3ce2c863a4ed2a4931e"
        ),
        "PX_XENIUM/labels": (
            "58e9beedc10aa63852c3924a01eb215779fb7b61462f0837707d6102ed4a4e35"
        ),
        "PX_XENIUM/provenance": (
            "e8afd0ebe43b92a18c282c1477e97c884646876c661584b815dd4cf063e3aaf1"
        ),
        "PX_XENIUM/summary": (
            "03fbc42573cfe065e8e3dfeddfbd8682a49c1fbb9a3a88b78ce24a93bbe2baa8"
        ),
        "pair/summary": (
            "a6e72ecf8c8ab09e1f5ec3a12c4e7a999abed11746ce0b940c596ae11978a164"
        ),
    },
    "human_no_tables": {
        "PX_MERSCOPE/labels": (
            "de8053e7d92451c9b8b200c1c73012ba3d30584cc5f8a2ff11baef64ecf03d09"
        ),
        "PX_MERSCOPE/provenance": (
            "93d644defa8cc174e27c366a1d804e51151b5ffe293d4dc610653fcdc3fdb489"
        ),
        "PX_MERSCOPE/summary": (
            "5e8e865bcee2add7ea992f4c82ded489e09b7776a6ade3699dedc721ba35ddab"
        ),
        "PX_XENIUM/labels": (
            "74616602d7efe77980bea68d6532d430ea1f04650b8801debe0447dfaf8d06db"
        ),
        "PX_XENIUM/provenance": (
            "6bc12189432a8c1c6c3fbf746708171b065c6d613a05c57d3240e262fa1c6ec8"
        ),
        "PX_XENIUM/summary": (
            "953fc3ad4eaacd8067f0b33e5852a376db82cb0d8126cca6972135ec561207c7"
        ),
        "pair/summary": (
            "a41d55c3150a553be0d83cd97c442b02ba585fba50062d4e0d11851f8a2b1122"
        ),
    },
    "mouse_v6_provisional": {
        "AG_MERSCOPE/labels": (
            "b280d2188f45d21e5d9edda6a2b514d0f4bf18e94b89a5eea38437d78f87e4be"
        ),
        "AG_MERSCOPE/provenance": (
            "786bbc3a774de00138da087690a1be95f3c64c94a710a41ac498a17d17f90e2f"
        ),
        "AG_MERSCOPE/summary": (
            "d39764759fbf69b9593e55a23e6221a84fda9f4d9f85347eaffc3c87e75bd4ca"
        ),
        "pair/summary": (
            "2f57b043cfa436726ad1a5542457b27375c8f72ff4aac8d5e7df3571d7f4b2c6"
        ),
    },
    "mouse_no_tables": {
        "AG_MERSCOPE/labels": (
            "6c14f98d3e7a3d4c43a4321e9abd97989f612c73084adf1336f42f1b8ef03b9c"
        ),
        "AG_MERSCOPE/provenance": (
            "da8f9f22b7e3317ebce2f55ef948f46b29eee438e90f3ea24bf8806ec567ad65"
        ),
        "AG_MERSCOPE/summary": (
            "79e782b05be7d82beb1540b737834e97bb26bc1bf79744985bb5a76ffb963c0f"
        ),
        "pair/summary": (
            "9c1288e10f3bfa446057d51eacbf3d6a369628f0cabbdab5b317af0ef19bce47"
        ),
    },
}
V6_SCENARIOS: tuple[str, ...] = (
    "human_v6_provisional",
    "human_v6_validated",
    "human_no_tables",
    "mouse_v6_provisional",
    "mouse_no_tables",
)
PRINT_GOLDEN_ENV = "MERXEN_PRINT_RESOLVE_V6_GOLDEN"
# Spelled out: the digests are also computed with code that predates
# ``Columns.FLAG_NONNEURONAL_HIGH_DEPTH``.
NEW_FLAG_COLUMN = "flag_nonneuronal_high_depth"


# Floats are compared to 10 significant digits: their last bits differ between
# hosts with the same pinned packages (pre-registration §22.8, test (i) on
# other hosts); statuses, labels, flags and counts are compared exactly.
FLOAT_DIGITS = 10


def _canonical_table(frame: pd.DataFrame) -> str:
    """Return a table as canonical text: dtypes, then CSV (floats to 10 digits)."""
    dtypes = ";".join(f"{name}:{dtype}" for name, dtype in frame.dtypes.items())
    text = frame.to_csv(
        index=False, float_format=f"%.{FLOAT_DIGITS}g", lineterminator="\n"
    )
    return dtypes + "\n" + text


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Left out of the compared JSON: the label file's own digest (it moves with
# the new column; the table itself is compared without it) and the MAP run's
# wall time (the mouse scenarios map with the fake mapper on every run).
VOLATILE_KEYS: frozenset[str] = frozenset({"labels_sha256", "wall_time_s"})


def _without_volatile(payload: Any) -> Any:
    """Return a JSON payload without ``VOLATILE_KEYS``, floats to 10 digits."""
    if isinstance(payload, Mapping):
        return {
            key: _without_volatile(value)
            for key, value in payload.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(payload, list):
        return [_without_volatile(value) for value in payload]
    if isinstance(payload, float):
        return float(f"{payload:.{FLOAT_DIGITS}g}")
    return payload


# M13 C11 (after the digests were computed) fills these fields of the panel
# record where RESOLVE left them empty: the human record without panel
# diagnostics and every mouse record (``pipeline.sample_panel_provenance``).
# The compared provenance resets them to their defaults; the test of C11
# checks their values.
C11_PANEL_FIELDS: tuple[str, ...] = (
    "panel_family",
    "family_basis",
    "panel_mode",
    "validation_basis",
    "validated_max_level",
    "validated_panels_sha256",
    "validated_panel_levels_sha256",
    "validated_share",
    "n_declared_genes",
    "gene_id_resolution",
    "n_unmapped",
    "controls_removed",
)


def _without_c11_panel_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a provenance payload with ``C11_PANEL_FIELDS`` at their defaults."""
    from merxen.annotation.provenance import PanelProvenance

    if payload.get("panel") is None:
        return payload
    defaults = PanelProvenance().model_dump(mode="json")
    panel = {
        **payload["panel"],
        **{name: defaults[name] for name in C11_PANEL_FIELDS},
    }
    return {**payload, "panel": panel}


# The resolve-summary keys M13 C15 adds (a human sample's real-data QC block,
# the pair's real_qc config) and the schema version that came with them.
C15_SUMMARY_KEYS: frozenset[str] = frozenset({"real_qc", "real_qc_config"})
C15_SCHEMA_VERSION = 3


def _without_c15(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a resolve-summary record without M13 C15's additions."""
    record = {
        key: value for key, value in payload.items() if key not in C15_SUMMARY_KEYS
    }
    if record.get("schema_version") == C15_SCHEMA_VERSION:
        record["schema_version"] = C15_SCHEMA_VERSION - 1
    if isinstance(record.get("samples"), dict):
        record["samples"] = {
            key: _without_c15(value) for key, value in record["samples"].items()
        }
    return record


def _v6_digests(result: ResolveResult, root: Path) -> dict[str, str]:
    """Return the digests of a RESOLVE result that version 6 must keep."""

    def scrub(payload: Any) -> str:
        text = json.dumps(_without_volatile(payload), sort_keys=True, default=str)
        return text.replace(str(root), "<tmp>")

    digests: dict[str, str] = {}
    for sample_id in sorted(result.samples):
        sample = result.samples[sample_id]
        labels, provenance = read_label_table(sample.labels_path)
        labels = labels.drop(columns=[NEW_FLAG_COLUMN], errors="ignore")
        digests[f"{sample_id}/labels"] = _digest(_canonical_table(labels))
        digests[f"{sample_id}/provenance"] = _digest(
            scrub(_without_c11_panel_fields(provenance.model_dump(mode="json")))
        )
        digests[f"{sample_id}/summary"] = _digest(scrub(_without_c15(sample.summary)))
    pair = _without_c15(dict(result.summary))
    pair.pop("map_manifest_sha256", None)
    digests["pair/summary"] = _digest(scrub(pair))
    return digests


def _v6_scenario(
    scenario: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    *,
    with_real_qc: dict[str, ResolveResult] | None = None,
) -> ResolveResult:
    """Run RESOLVE in one version-6 scenario (``V6_SCENARIOS``).

    Human RESOLVE runs QC-free (``real_qc.enabled`` false): the golden
    digests are of the code before M13 C15, which had no real-data QC. With
    ``with_real_qc`` it also runs with the QC on the same MAP output, stored
    there under ``"qc"``.
    """
    make_trust = request.getfixturevalue("make_trust")
    species, tables, *state = scenario.split("_")
    regime = state[0] if state else "provisional"
    if species == "human":
        setup = _human_setup(tmp_path, request.getfixturevalue("fake_mmc"))
        if tables == "v6":
            write_human_tables(setup.bundles["whb_frontal_supc_clus"].path, 6)
        trust_state = "validated_real" if regime == "validated" else "provisional"

        def run(enabled: bool, output: str) -> ResolveResult:
            qc_config = setup.config.real_qc.model_copy(update={"enabled": enabled})
            return _human_resolve(
                setup,
                make_trust,
                output,
                state=trust_state,
                config=setup.config.model_copy(update={"real_qc": qc_config}),
            )

        if with_real_qc is not None:
            with_real_qc["qc"] = run(True, "resolve_qc")
        return run(False, "resolve_out")
    mouse_setup = request.getfixturevalue("mouse_setup")
    if tables == "v6":
        write_tables(
            mouse_setup["bundle"].path,
            version=6,
            levels=mouse_levels(AnnotationThresholds()),
            grid=MOUSE_GRID,
            plan=MOUSE_PLAN,
            truth=MOUSE_TRUTH,
            neuronal=MOUSE_NEURONAL,
            high_depth=MOUSE_HIGH_DEPTH,
        )
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    return annotate_resolve(
        map_dir,
        _mouse_config(),
        output_dir=tmp_path / "resolve",
        panel_dir=mouse_setup["panel_dir"],
        trust_overrides={"wmb_panel": make_trust("provisional", species="mouse")},
        n_bootstrap=5,
        registration={MOUSE_SID: PASSING},
    )


@pytest.mark.parametrize("scenario", V6_SCENARIOS)
def test_version_6_resolve_outputs_are_unchanged_but_for_the_null_flag(
    scenario: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    applied: dict[str, ResolveResult] = {}
    result = _v6_scenario(scenario, tmp_path, request, with_real_qc=applied)
    digests = _v6_digests(result, tmp_path)
    if os.environ.get(PRINT_GOLDEN_ENV):
        print(json.dumps({scenario: digests}, sort_keys=True))  # noqa: T201
    assert digests == V6_GOLDEN[scenario]
    assert Columns.FLAG_NONNEURONAL_HIGH_DEPTH == NEW_FLAG_COLUMN
    for sample in result.samples.values():
        labels, provenance = read_label_table(sample.labels_path)
        flag = labels[Columns.FLAG_NONNEURONAL_HIGH_DEPTH]
        assert str(flag.dtype) == "boolean" and flag.isna().all()
        assert "resolvability_v7" not in sample.summary
        # The C11 fields hold the run's values (reset only for the digests).
        assert provenance is not None and provenance.panel is not None
        assert provenance.panel.panel_mode == (
            "intersection" if scenario.startswith("human") else "single_sample"
        )
    if not scenario.startswith("human"):
        return
    # With the real-data QC (M13 C15), which lowers nothing in these
    # scenarios, the label tables are the golden ones too.
    for sample_id, sample in applied["qc"].samples.items():
        assert sample.summary["real_qc"]["enabled"]
        assert sample.summary["real_qc"]["effects"]["gate_cap"] is None
        labels, _ = read_label_table(sample.labels_path)
        labels = labels.drop(columns=[NEW_FLAG_COLUMN])
        golden = V6_GOLDEN[scenario][f"{sample_id}/labels"]
        assert _digest(_canonical_table(labels)) == golden


@pytest.mark.parametrize("version", [8, 0, "7", 6.5])
def test_resolve_refuses_an_unknown_resolvability_version(
    tmp_path: Path, fake_mmc: FakeMmc, make_trust: MakeTrust, version: Any
) -> None:
    setup = _human_setup(tmp_path, fake_mmc)
    write_human_tables(setup.bundles["whb_frontal_supc_clus"].path, version=version)
    match = "newer than this code" if version == 8 else "unknown resolvability"
    with pytest.raises(res.ResolvabilityError, match=match):
        _human_resolve(setup, make_trust, state="provisional")


# --------------------------------------------------------------------------
# Mouse RESOLVE on a version-7 bundle


def test_mouse_resolve_applies_the_version_7_ensemble_decisions(
    tmp_path: Path, mouse_setup: dict[str, Any], make_trust: MakeTrust
) -> None:
    bundle_dir = mouse_setup["bundle"].path
    write_tables(
        bundle_dir,
        version=7,
        levels=mouse_levels(AnnotationThresholds()),
        grid=MOUSE_GRID,
        plan=MOUSE_PLAN,
        truth=MOUSE_TRUTH,
        neuronal=MOUSE_NEURONAL,
        high_depth=MOUSE_HIGH_DEPTH,
    )
    map_dir = tmp_path / "map"
    map_mouse(mouse_setup, map_dir)
    config = _mouse_config()
    config = config.model_copy(
        update={
            "real_qc": config.real_qc.model_copy(
                update={"nonneuronal_high_depth_counts": MOUSE_HIGH_DEPTH}
            )
        }
    )
    result = annotate_resolve(
        map_dir,
        config,
        output_dir=tmp_path / "resolve",
        panel_dir=mouse_setup["panel_dir"],
        trust_overrides={"wmb_panel": make_trust("provisional", species="mouse")},
        n_bootstrap=5,
        registration={MOUSE_SID: PASSING},
    )
    sample = result.samples[MOUSE_SID]
    labels, _ = read_label_table(sample.labels_path)
    validate_label_table(labels, "mouse")
    labels = labels.set_index("cell_id")
    kinds = pd.Series(mouse_setup["kinds"], index=labels.index)
    flag = labels[Columns.FLAG_NONNEURONAL_HIGH_DEPTH]
    table = labels[Columns.IN_TABLE].to_numpy(bool)
    assert flag[table].notna().all() and flag[~table].isna().all()
    # The two astrocytes (60 counts, class 30 Astro-Epen) sit in the emitted
    # non-neuronal high-depth bin; the 30-count neurons do not.
    astro = kinds.index[kinds == "astro"]
    assert flag.loc[astro].all()
    assert int(flag.fillna(False).sum()) == len(astro)
    summary = sample.summary["resolvability_v7"]
    assert summary["nonneuronal_high_depth"]["n_flagged"] == len(astro)
    prediction = {
        (item["level"], item["class"]): item
        for item in summary["class_depth_prediction"]["classes"]
    }
    neurons = prediction[("class", "01 IT-ET Glut")]
    assert neurons["share_source"] == "own" and neurons["n_cells"] >= 300
    assert prediction[("class", "30 Astro-Epen")]["share_source"] == "label_free"
    assert set(labels.loc[astro, Columns.level("class", "status")].astype(str)) == {
        CellStatus.CONFIDENT.value
    }
