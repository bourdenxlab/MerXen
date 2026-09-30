"""Tests for the M8 draw-spread table (D2; pre-registration §18)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import draw_spread as ds
from merxen.annotation import resolvability as res

from .test_resolvability import (
    GRID,
    bootstrap_mapper,
    make_test_cells,
    recipes,
    settings,
    synthetic_specs,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "acceptance"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    name = "draw_spread_script"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "draw_spread.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stored_tables(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A synthetic self-map written and read back as a bundle stores it."""
    test = make_test_cells(40)
    result = res.run_resolvability(
        test,
        specs=synthetic_specs(),
        depths=GRID,
        recipes=recipes(),
        map_fn=bootstrap_mapper,
        settings=settings(min_cells_per_bin=20, min_confident_n=20),
        species="human",
    )
    directory = tmp_path_factory.mktemp("bundle")
    result.write(directory)
    tables = res.load_resolvability(directory)
    assert tables is not None
    return test, tables


def test_draw_grid_is_the_three_by_three_grid_with_the_scored_draw_first() -> None:
    draws = ds.draw_grid()
    assert len(draws) == 9
    assert draws[0] == ds.Draw(0, 0) and draws[0].is_scored
    assert [draw.tag for draw in draws].count("c0e0") == 1
    assert not any(draw.is_scored for draw in draws[1:])
    assert {draw.tag for draw in draws} == {
        f"c{c}e{e}" for c in (0, 1, 2) for e in (0, 1, 2)
    }
    with pytest.raises(ValueError, match="include 0"):
        ds.draw_grid((1, 2))
    with pytest.raises(ValueError, match="distinct"):
        ds.draw_grid((0, 0, 1))


def test_the_efficiency_seed_defaults_to_the_recipe_seed_and_separates() -> None:
    test = make_test_cells(10)
    decision = recipes()[0]
    assert decision.gene_efficiency_sigma is not None

    def counts(query: res.SimulatedQuery) -> np.ndarray:
        return query.counts.toarray()

    base = res.thin_and_contaminate(test, GRID, decision)
    explicit = res.thin_and_contaminate(test, GRID, decision, efficiency_seed=0)
    np.testing.assert_array_equal(counts(base), counts(explicit))
    other_efficiency = res.thin_and_contaminate(test, GRID, decision, efficiency_seed=1)
    assert not np.array_equal(counts(base), counts(other_efficiency))
    # Same cells and partners: only the efficiency changed.
    assert list(other_efficiency.obs.index) == list(base.obs.index)
    np.testing.assert_array_equal(
        other_efficiency.obs["partner_id"].to_numpy(),
        base.obs["partner_id"].to_numpy(),
    )
    # The per-cell seed alone changes the draw too, with the same efficiency.
    import dataclasses

    other_cells = res.thin_and_contaminate(
        test, GRID, dataclasses.replace(decision, seed=1), efficiency_seed=0
    )
    assert not np.array_equal(counts(base), counts(other_cells))


def test_the_scored_draw_reproduces_the_stored_self_map(stored_tables: Any) -> None:
    test, tables = stored_tables
    decision = recipes()[0]
    cells = ds.simulate_draw(
        test,
        GRID,
        decision,
        ds.Draw(0, 0),
        specs=synthetic_specs(),
        map_fn=bootstrap_mapper,
    )
    stored = tables.cells[tables.cells["recipe"] == decision.name]
    keys = ["level", "sim_id"]
    left = stored.sort_values(keys).reset_index(drop=True)
    right = res.restore_labels(cells).sort_values(keys).reset_index(drop=True)
    assert left["sim_id"].tolist() == right["sim_id"].tolist()
    assert left["call"].tolist() == right["call"].tolist()
    np.testing.assert_array_equal(left["bp"].to_numpy(), right["bp"].to_numpy())
    # Substituting the scored draw leaves every decision unchanged.
    substituted = ds.tables_with_draw(tables, cells)
    pd.testing.assert_frame_equal(substituted.decisions(), tables.decisions())
    # The clean recipe's rows are kept.
    assert (substituted.cells["recipe"] == "clean").sum() == (
        tables.cells["recipe"] == "clean"
    ).sum()


def test_another_draw_is_decided_alone_with_the_unchanged_rules(
    stored_tables: Any,
) -> None:
    test, tables = stored_tables
    decision = recipes()[0]
    cells = ds.simulate_draw(
        test,
        GRID,
        decision,
        ds.Draw(1, 2),
        specs=synthetic_specs(),
        map_fn=bootstrap_mapper,
    )
    substituted = ds.tables_with_draw(tables, cells)
    assert substituted.settings == tables.settings
    assert substituted.summary is tables.summary
    rows = substituted.cells[substituted.cells["recipe"] == decision.name]
    stored = tables.cells[tables.cells["recipe"] == decision.name]
    assert len(rows) == len(stored)
    assert not np.array_equal(
        rows.sort_values(["level", "sim_id"])["bp"].to_numpy(),
        stored.sort_values(["level", "sim_id"])["bp"].to_numpy(),
    )
    decisions = substituted.decisions()
    assert set(decisions["regime"]) == set(tables.decisions()["regime"])
    with pytest.raises(ValueError, match="decision recipe"):
        ds.tables_with_draw(tables, tables.cells)


def _decisions(rows: list[tuple[str, str, str, int, str, float | None]]) -> Any:
    return pd.DataFrame(
        [
            {
                "regime": regime,
                "level": level,
                "class": cls,
                "depth": depth,
                "status": status,
                "d_max": d_max,
                "precision": 0.9,
                "wilson_lb": 0.88,
                "n_confident": 60,
                "reason": None if status == "emitted" else "wilson_bound_below_target",
                "pooled": False,
                "extrapolated": False,
                "n_fit": 60,
                "would_raise": False,
                "would_raise_evaluable": True,
            }
            for regime, level, cls, depth, status, d_max in rows
        ]
    )


FLOORS = pd.DataFrame(
    {
        "level": ["broad", "broad", "supercluster"],
        "floor_class": ["OPC", "OPC", "Oligo"],
        "platform": ["MERSCOPE", "XENIUM", "MERSCOPE"],
        "min_counts": [10, 15, 30],
    }
)


def test_h18_needs_emission_from_one_step_above_the_floor_to_d_max() -> None:
    depths = [10, 15, 30, 60, 120, 250]
    rows = [("trust", "broad", "OPC", d, "emitted", 120.0) for d in depths]
    rows += [("trust", "supercluster", "COP", d, "emitted", None) for d in depths]
    rows += [("trust", "supercluster", "Oligo", d, "emitted", 60.0) for d in depths]
    # Broad OPC: emitted at 10, 30, 60, 120 (not 15); supercluster Oligo at 60.
    rows += [
        (
            "validated",
            "broad",
            "OPC",
            d,
            "emitted" if d != 15 else "not_resolvable",
            None,
        )
        for d in depths
    ]
    rows += [("validated", "supercluster", "Oligo", 60, "emitted", None)]
    decisions = _decisions(rows)
    merscope = ds.h18_human_rows(decisions, depths, "MERSCOPE", FLOORS, dataset="D")
    by_class = merscope.set_index(["level", "class"])
    opc = by_class.loc[("broad", "OPC")]
    assert opc["floor"] == 10 and opc["expected"] == "15,30,60,120"
    assert opc["missing"] == "15" and not opc["passes"]
    # A class without D_max is not required (supercluster COP after M8 D1).
    cop = by_class.loc[("supercluster", "COP")]
    assert not cop["required"] and cop["passes"]
    oligo = by_class.loc[("supercluster", "Oligo")]
    assert oligo["expected"] == "60" and oligo["passes"]
    # Xenium's broad OPC floor is 15: expected from 30, so 15 is not missing.
    xenium = ds.h18_human_rows(decisions, depths, "XENIUM", FLOORS, dataset="D")
    opc_x = xenium.set_index(["level", "class"]).loc[("broad", "OPC")]
    assert opc_x["expected"] == "30,60,120" and opc_x["passes"]
    rows_out, h18, bins = ds.decision_metric_rows(
        decisions,
        draw=ds.Draw(0, 0),
        dataset="D",
        platform="MERSCOPE",
        depths=depths,
        floors=FLOORS,
        settings=res.RuleSettings(),
    )
    metrics = {row["metric"]: row for row in rows_out}
    failing = metrics[ds.METRIC_H18_FAILING]
    assert failing["value"] == 1.0 and failing["passes"] is False
    assert failing["note"] == "broad OPC missing 15"
    assert metrics[ds.METRIC_BINS_EMITTED]["value"] == 6.0
    assert set(h18["draw"]) == {"c0e0"}
    assert set(bins["dataset"]) == {"D"} and len(bins) == 7


def test_would_raise_counts_validated_bins_at_the_h18_depths() -> None:
    depths = [10, 15, 30]
    decisions = _decisions(
        [
            ("validated", "broad", "OPC", 10, "emitted", None),
            ("validated", "broad", "OPC", 15, "emitted", None),
            ("validated", "supercluster", "Oligo", 15, "emitted", None),
            ("validated", "supercluster", "Oligo", 30, "emitted", None),
        ]
    )
    decisions["would_raise"] = True
    rows_out, _, _ = ds.decision_metric_rows(
        decisions,
        draw=ds.Draw(1, 0),
        dataset="D",
        platform="MERSCOPE",
        depths=depths,
        floors=FLOORS,
        settings=res.RuleSettings(min_cells_per_bin=50),
    )
    raised = {row["metric"]: row for row in rows_out}[ds.METRIC_H18_WOULD_RAISE]
    # Broad from 15 and supercluster from 30 counts only; listed, never judged.
    assert raised["value"] == 2.0 and raised["passes"] is None


def test_spread_table_reports_the_scored_value_and_the_draws_failing() -> None:
    rows = []
    values = {"c0e0": 0.436, "c1e0": 0.29, "c2e1": 0.45}
    for tag, value in values.items():
        draw = ds.Draw(int(tag[1]), int(tag[3]))
        rows.append(
            ds.metric_row(
                draw, "P5011_MERSCOPE", "H7", value, threshold=0.30, comparator=">="
            )
        )
        rows.append(ds.metric_row(draw, "PREP", "note_only", 3.0))
    table = ds.spread_table(rows).set_index(["dataset", "metric"])
    h7 = table.loc[("P5011_MERSCOPE", "H7")]
    assert h7["scored_value"] == pytest.approx(0.436) and h7["scored_passes"]
    assert h7["min"] == pytest.approx(0.29) and h7["max"] == pytest.approx(0.45)
    assert h7["n_draws"] == 3 and h7["n_draws_failing"] == 1
    assert h7["failing_draws"] == "c1e0" and h7["verdict_depends_on_draw"]
    note = table.loc[("PREP", "note_only")]
    assert note["scored_passes"] is None and note["n_draws_failing"] == 0
    assert not note["verdict_depends_on_draw"]
    assert ds.spread_table([]).empty


def test_bin_spread_counts_the_draws_emitting_each_bin() -> None:
    bins = pd.DataFrame(
        {
            "draw": ["c0e0", "c1e1", "c2e2", "c0e0", "c1e1"],
            "dataset": ["D"] * 5,
            "level": ["broad"] * 5,
            "class": ["OPC", "OPC", "OPC", "Oligo", "Oligo"],
            "depth": [15, 15, 15, 10, 10],
            "emitted": [False, True, True, True, True],
        }
    )
    spread = ds.bin_spread(bins).set_index("class")
    assert not spread.loc["OPC", "scored_emitted"]
    assert spread.loc["Oligo", "scored_emitted"]
    assert spread.loc["OPC", "n_draws_emitting"] == 2
    assert spread.loc["OPC", "draws_emitting"] == "c1e1,c2e2"
    assert spread.loc["Oligo", "n_draws"] == 2
    assert ds.bin_spread([]).empty


def test_dataset_composition_follows_the_table_cells() -> None:
    from merxen.annotation.composition import dataset_type_composition

    leaf = pd.DataFrame(
        {
            "assignment": ["n1", "n2", "n1", "n2"],
            "bp": [1.0, 0.6, 0.9, 1.0],
        },
        index=pd.Index(["a", "b", "c", "d"]),
    )
    labels = pd.DataFrame(
        {
            "cell_id": ["d", "a", "b", "c"],
            "in_table": [True, True, False, True],
            "total_counts": [100.0, 12.0, 50.0, 40.0],
        }
    )
    got = ds.dataset_composition(leaf, labels, [10, 30], min_bin_mass=1.0)
    expected = dataset_type_composition(
        leaf.loc[["d", "a", "c"]],
        np.array([100.0, 12.0, 40.0]),
        [10, 30],
        min_bin_mass=1.0,
    )
    assert got == expected


def test_script_compares_a_redrawn_scored_draw(script: ModuleType) -> None:
    stored = pd.DataFrame(
        {
            "level": ["a", "a"],
            "sim_id": ["x|D10", "y|D10"],
            "call": ["n1", None],
            "bp": [0.9, np.nan],
            "correct": [True, False],
        }
    )
    same = script.compare_cells(stored, stored.iloc[::-1].copy())
    assert same["reproduces"] and same["calls_changed"] == 0.0
    changed = stored.assign(bp=[0.8, np.nan])
    assert not script.compare_cells(stored, changed)["reproduces"]
    assert not script.compare_cells(stored, stored.iloc[:1])["same_rows"]


def test_script_serves_the_draw_tables_for_the_one_bundle(
    script: ModuleType, tmp_path: Path
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    marker = SimpleNamespace(name="draw tables")
    with script.draw_tables_installed(bundle, marker):
        assert res.load_resolvability(bundle) is marker
        assert res.load_resolvability(tmp_path / "other") is None
    assert res.load_resolvability(bundle) is None


def test_script_resolve_rows_score_h7_and_the_expected_warning(
    script: ModuleType,
) -> None:
    from merxen.annotation.config import AnnotationConfig

    labels = pd.DataFrame(
        {
            "in_table": [True, True, True, False],
            "ct_broad_status": [
                "confident",
                "low_confidence",
                "confident",
                "confident",
            ],
        }
    )
    gate = {"level": "broad_only", "warning": True, "broad_coverage_segmented": 0.148}
    result = SimpleNamespace(
        samples={
            "P1212_MERSCOPE": SimpleNamespace(
                platform="MERSCOPE",
                labels=labels,
                summary={"resolution": {"gate": gate}},
            )
        }
    )
    rows = script.resolve_rows(
        ds.Draw(0, 0), result, "P1212", AnnotationConfig(species="human")
    )
    by_metric = {row["metric"]: row for row in rows}
    h7 = by_metric["H7"]
    assert h7["value"] == pytest.approx(2 / 3) and h7["threshold"] == 0.34
    assert h7["passes"] is True
    warning = by_metric["H8/warning_value"]
    # §5.4 expects no warning on P1212_MERSCOPE: the D3 exception case.
    assert warning["value"] == 0.148 and warning["passes"] is False
    assert by_metric["H8/level"]["passes"] is True


def test_script_help_runs(script: ModuleType) -> None:
    with pytest.raises(SystemExit) as raised:
        script.main(["--help"])
    assert raised.value.code == 0
