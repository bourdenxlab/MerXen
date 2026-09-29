"""Tests for profile mode and per-class depth predictions (plan §8.3 v7.5; M3c)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.annotation import resolvability as res
from merxen.annotation import sim_inputs as si
from merxen.annotation import simulate
from merxen.annotation.config import AnnotationResolvabilityConfig
from merxen.annotation.reference import builder_for, prepare_reference_spec
from merxen.annotation.simulate import ReferenceBuild, run_panel_simulation

from .test_resolvability import make_test_cells
from .test_simulate import _large_whb_simulation, small_resources  # noqa: F401

CONFIG = AnnotationResolvabilityConfig()
GRID = [10, 50, 100, 250]


def _profile() -> si.DepthProfile:
    totals_a = list(np.linspace(100, 300, 140)) + [600.0] * 10
    return si.DepthProfile(
        ["A"] * 150 + ["B"] * 50,
        totals_a + list(np.linspace(60, 200, 50)),
        species="mouse",
        label="synthetic",
        neuronal={"A": True, "B": False},
    )


@pytest.fixture(scope="module")
def cells() -> res.HeldOutCells:
    return make_test_cells(20, seed=5)


@pytest.fixture(scope="module")
def profile_query(cells: res.HeldOutCells) -> Any:
    truth = cells.obs[f"{res.TRUTH_PREFIX}L1"].to_numpy()
    recipe = res.member_recipe(res.DECISION_RECIPE, 0, CONFIG)
    return simulate.simulate_on_profile(cells, _profile(), truth, recipe, GRID)


def test_profile_mode_draws_totals_per_truth_class(
    cells: res.HeldOutCells, profile_query: Any
) -> None:
    obs = profile_query.obs
    truth = cells.obs[f"{res.TRUTH_PREFIX}L1"]
    assert len(obs) == 2 * simulate.profile_sims_per_class(80) == 1280
    assert simulate.profile_sims_per_class(80) == 640
    assert simulate.profile_sims_per_class(10_000) == 20_000
    assert simulate.profile_sims_per_class(100) == 800
    assert simulate.profile_sims_per_class(300) == 2000
    assert (
        truth.reindex(obs["cell_id"]).to_numpy() == obs["truth_class_sampled"]
    ).all()
    source = obs.groupby("truth_class_sampled")["profile_source"].first().to_dict()
    assert source == {"A": "A", "B": si.POOL_NON_NEURONAL}
    np.testing.assert_allclose(obs["host_depth"], obs["target_total"] / 1.25)
    native = pd.Series(cells.native_counts, index=cells.obs.index)
    own = ~obs["truncated"]
    assert (
        native.reindex(obs.loc[own, "cell_id"]).to_numpy() >= obs.loc[own, "host_depth"]
    ).all()
    # Totals no class-A cell can reach are truncated at the deepest A cell.
    truncated = obs[obs["truncated"]]
    assert len(truncated) > 0 and set(truncated["truth_class_sampled"]) == {"A"}
    deepest = native[truth == "A"].idxmax()
    assert set(truncated["cell_id"]) == {deepest}
    np.testing.assert_allclose(truncated["target_total"], native[deepest] * 1.25)
    values_a = set(np.round(_profile().by_class["A"], 9))
    assert (
        set(
            np.round(
                obs.loc[own & (obs["truth_class_sampled"] == "A"), "target_total"], 9
            )
        )
        <= values_a
    )
    # Spill partners come from another spill group and reach the amount.
    groups = cells.obs[res.SPILL_GROUP_COLUMN]
    assert (
        groups.reindex(obs["partner_id"]).to_numpy()
        != groups.reindex(obs["cell_id"]).to_numpy()
    ).all()
    assert (
        native.reindex(obs["partner_id"]).to_numpy() >= 0.25 * obs["host_depth"]
    ).all()
    # A cell's bin is the grid value at or below its realised total.
    bins = np.nan_to_num(res.depth_bin(obs["total_counts"].to_numpy(), GRID), nan=0)
    np.testing.assert_array_equal(obs["depth"].to_numpy(), bins.astype(int))
    assert abs(obs["total_counts"].mean() / obs["target_total"].mean() - 1) < 0.02
    assert obs["member"].eq("R1_contam_HO@0").all()


def test_profile_mode_is_keyed_and_block_independent(
    cells: res.HeldOutCells, profile_query: Any
) -> None:
    truth = cells.obs[f"{res.TRUTH_PREFIX}L1"].to_numpy()
    recipe = res.member_recipe(res.DECISION_RECIPE, 0, CONFIG)
    again = simulate.simulate_on_profile(
        cells, _profile(), truth, recipe, GRID, block_rows=7
    )
    assert (again.counts != profile_query.counts).nnz == 0
    pd.testing.assert_frame_equal(again.obs, profile_query.obs)
    renamed = simulate.simulate_on_profile(
        cells, _profile(), truth, recipe, GRID, key_name="R1_contam_realdepth"
    )
    assert not np.array_equal(
        renamed.obs["target_total"].to_numpy(),
        profile_query.obs["target_total"].to_numpy(),
    )
    assert renamed.obs.index[0].split("|")[1] == "R1_contam_realdepth"
    with pytest.raises(simulate.SimulationError, match="truth classes"):
        simulate.simulate_on_profile(cells, _profile(), truth[:-1], recipe, GRID)


def _mouse_cells() -> pd.DataFrame:
    rows = []
    for sim, class_bp, sub_bp, total, correct in (
        ("s1", 0.96, 0.90, 150.0, True),
        ("s2", 0.92, 0.90, 150.0, True),
        ("s3", 0.99, 0.90, 55.0, False),
        ("s4", 0.99, 0.85, 40.0, True),
        ("s5", 0.99, 0.90, 110.0, True),
    ):
        for level, bp in (("class", class_bp), ("subclass", sub_bp)):
            rows.append(
                {
                    "recipe": "R1_contam_HO",
                    "seed": 0,
                    "level": level,
                    "sim_id": sim,
                    "cell_id": sim,
                    "depth": 0,
                    "half": 0,
                    "parent": "01 IT-ET Glut",
                    "call": "x",
                    "bp": bp,
                    "corr": 0.5,
                    "truth": "x",
                    "truth_parent": "01 IT-ET Glut",
                    res.TRUTH_LEAF_COLUMN: "n_sub",
                    "correct": correct,
                    "total_counts": total,
                }
            )
    return pd.DataFrame(rows)


def _mouse_summary() -> dict[str, Any]:
    return {
        "depth_grid": [10, 100],
        "levels": [
            {"level": "class", "role": "class", "default_threshold": 0.9},
            {"level": "subclass", "role": "leaf", "default_threshold": 0.8},
        ],
        "settings": {"provisional_mouse_subclass_floor": 60},
        "emission": {
            "provisional": {
                "class": {
                    "01 IT-ET Glut": {
                        "10": {"status": "not_resolvable", "threshold": None},
                        "100": {"status": "emitted", "threshold": 0.95},
                    }
                },
                "subclass": {
                    "01 IT-ET Glut": {"100": {"status": "emitted", "threshold": 0.85}}
                },
            }
        },
        "floors": {"provisional": {"class": {"01 IT-ET Glut": {"floor": 120}}}},
    }


def test_profile_cells_follow_the_raw_rule_and_the_bundle_decisions() -> None:
    table = simulate.profile_cell_table(
        _mouse_cells(),
        species="mouse",
        summary=_mouse_summary(),
        names={"n_sub": "001 Sub"},
    )
    assert table["cov_class_rule73"].tolist() == [True] * 5
    assert table["cov_subclass_rule73"].tolist() == [True, True, True, False, True]
    # s2 is below the threshold, s3 and s4 in an unemitted bin, s5 below the floor.
    assert table["cov_class_prov"].tolist() == [True, False, False, False, False]
    assert table["cov_subclass_prov"].tolist() == [True, False, False, False, False]
    assert set(table["truth_leaf"]) == {"001 Sub"}
    metrics = simulate.profile_metrics(table)
    prediction = simulate.per_class_predictions(table, None, metrics)
    overall = prediction.set_index("class").loc["ALL"]
    assert overall["cov_class_rule73"] == 1.0
    assert overall["cov_subclass_rule73"] == 0.8
    assert overall["cov_class_prov"] == pytest.approx(0.2)
    assert overall["prec_class_rule73"] == 0.8
    assert overall["prec_subclass_rule73"] == pytest.approx(0.75)
    assert overall["n"] == 5 and overall["kish_n"] == 5.0


def test_composition_weights_to_a_real_composition() -> None:
    truth = pd.Series(["a1", "a1", "a2", "a3", "b1", "c1"])
    real = pd.Series({"a1": 0.25, "a2": 0.25, "b1": 0.5})
    class_of = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "c1": "C"}
    weights = simulate.composition_weights_to_real(truth, real, class_of)
    # p_real(A) / p_sim(A) = 0.5 / (4 / 6); within A: a1 1, a2 2, a3 (absent
    # from the real data) 0, mean 1; class C is absent: weight 0.
    np.testing.assert_allclose(weights, [0.75, 0.75, 1.5, 0.0, 3.0, 0.0])


def test_the_class_depth_predictor_sums_emitted_bins() -> None:
    profile = si.DepthProfile(
        ["A"] * 4,
        [5, 50, 150, 300],
        species="mouse",
        label="x",
        min_cells=1,
        neuronal={"A": True},
    )
    predicted = pd.DataFrame(
        {
            "level": ["class"] * 3,
            "class": ["A"] * 3,
            "depth": [10, 100, 250],
            "status": ["emitted", "emitted", "not_resolvable"],
            "coverage": [0.5, 0.8, 0.9],
        }
    )
    table = simulate.class_depth_predictions(predicted, profile, [10, 100, 250])
    row = table.iloc[0]
    assert row["profile_source"] == "A"
    assert row["share_below_grid"] == 0.25
    assert row["resolvable_share"] == pytest.approx(0.5)
    assert row["predicted_coverage"] == pytest.approx(0.25 * 0.5 + 0.25 * 0.8)
    headline = simulate.class_depth_headline(table, profile)
    assert headline["class"]["predicted_coverage"] == pytest.approx(0.325)


def test_profile_mode_re_simulates_the_bundles_own_members() -> None:
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    stage_d = {
        "emission_members": [
            "R1_contam_HO@0",
            "R1_contam_HO@1",
            "R1_contam_HO@2",
            "R3_measured_HO@0",
        ]
    }
    members = simulate.profile_ensemble(
        stage_d, CONFIG, species="mouse", chemistry="xenium_prime", member_table=table
    )
    assert [item.name for item in members] == stage_d["emission_members"]
    assert all(item.role == "emission" for item in members)
    # Without a record: the family's members (R1 x 6 + R3 x 2 with a table).
    default = simulate.profile_ensemble(
        {}, CONFIG, species="mouse", chemistry="xenium_prime", member_table=table
    )
    assert [item.name for item in default] == [
        *[f"R1_contam_HO@{seed}" for seed in (0, 6, 7, 8, 9, 10)],
        "R3_measured_HO@2",
        "R3_measured_HO@3",
    ]
    configured = AnnotationResolvabilityConfig(
        ensemble_r1_seeds=[4], ensemble_r3_seeds=[5]
    )
    assert [
        item.name
        for item in simulate.profile_ensemble(
            {},
            configured,
            species="mouse",
            chemistry="xenium_prime",
            member_table=table,
        )
    ] == ["R1_contam_HO@4", "R3_measured_HO@5"]


def test_depth_profiles_never_cross_species(tmp_path: Path) -> None:
    with pytest.raises(simulate.SimulationError, match="crosses species"):
        simulate.load_simulation_profile(
            None, si.PROFILE_MOUSE_PRIME_FF, species="human"
        )
    scenario = simulate.load_simulation_profile(
        None, si.SCENARIO_HUMAN_LUNG, species="human"
    )
    assert scenario is not None and scenario.pooled
    foreign = tmp_path / "foreign.csv"
    pd.DataFrame({"total_counts": [100, 200], "class": ["01 IT-ET Glut"] * 2}).to_csv(
        foreign, index=False
    )
    with pytest.raises(simulate.SimulationError, match="never crosses species"):
        simulate.load_simulation_profile(foreign, None, species="human")
    assert simulate.load_simulation_profile(foreign, None, species="mouse") is not None
    with pytest.raises(simulate.SimulationError, match="not both"):
        simulate.load_simulation_profile(
            foreign, si.SCENARIO_HUMAN_LUNG, species="human"
        )
    assert simulate.load_simulation_profile(None, None, species="human") is None


def test_panel_card_notes_for_prime_families() -> None:
    from merxen.annotation import diagnostics

    mouse = simulate.panel_card_notes("mouse", "xenium_prime")
    human = simulate.panel_card_notes("human", "xenium_prime")
    # User decision 4, verbatim and first; then the M3c trust rules (§8.10).
    for notes in (mouse, human):
        assert notes[:2] == [
            simulate.GLIAL_UPPER_BOUND_NOTE,
            simulate.PRECISION_UNMEASURED_NOTE,
        ]
        assert diagnostics.PRIME_TRUST_NOTE in notes
        assert diagnostics.PRIME_REAL_QC_NOTE in notes
        assert diagnostics.PRIME_IN_SAMPLE_NOTE in notes
    assert "provisional" in diagnostics.PRIME_TRUST_NOTE
    assert "every emission member" in diagnostics.PRIME_TRUST_NOTE
    assert "0.10" in diagnostics.PRIME_REAL_QC_NOTE
    assert "no empirical offset" in diagnostics.PRIME_REAL_QC_NOTE
    assert diagnostics.PRIME_MOUSE_NEXT_NOTE in mouse
    assert diagnostics.PRIME_MOUSE_NEXT_NOTE not in human
    assert all(note in human for note in diagnostics.PRIME_HUMAN_NOTES)
    assert "by analogy with mouse" in diagnostics.PRIME_HUMAN_NOTES[0]
    assert simulate.panel_card_notes("mouse", "merscope") == []
    assert simulate.panel_card_notes("human", "unknown") == []
    assert "-.06 to -.18" in simulate.GLIAL_UPPER_BOUND_NOTE
    assert simulate.GLIAL_UPPER_BOUND_NOTE == diagnostics.GLIAL_UPPER_BOUND_NOTE


def test_member_mean_averages_members() -> None:
    first = pd.DataFrame({"class": ["A", "ALL"], "n": [2, 4], "cov_x": [0.5, 0.6]})
    second = pd.DataFrame({"class": ["A", "ALL"], "n": [2, 4], "cov_x": [0.7, 0.8]})
    mean = simulate.member_mean({"m0": first, "m1": second}).set_index("class")
    assert mean.loc["A", "cov_x"] == pytest.approx(0.6)
    assert mean.loc["ALL", "cov_x"] == pytest.approx(0.7)


def test_simulation_headline_is_the_depth_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_resources: Any,  # noqa: F811
) -> None:
    config, store, spec, gene_list, mapped = _large_whb_simulation(
        tmp_path, monkeypatch
    )
    build = ReferenceBuild(
        spec=prepare_reference_spec(spec), builder=builder_for(spec, config)
    )
    depth_profile = tmp_path / "depths.csv"
    depth_profile.write_text("total_counts\n" + "\n".join(["40"] * 8 + ["300"] * 2))
    report = run_panel_simulation(
        gene_list=gene_list,
        species="human",
        name="large_whb",
        config=config,
        store=store,
        builds=lambda panel: [build],
        out_dir=tmp_path / "out",
        scratch_dir=tmp_path / "sim_scratch",
        platform="XENIUM",
        expected_depth=40,
        depth_profile=depth_profile,
        profile_mode=True,
        prefilter_compare="off",
    )
    record = report["references"]["whb_frontal_supc_clus"]
    assert report["panel"]["chemistry"]["chemistry"] == "unknown"
    mode = record["profile_mode"]
    assert mode["status"] == "run", mode
    # Profile mode re-simulates the bundle's own emission members (R1 x 8).
    bundle_summary = json.loads(
        (Path(record["bundle_dir"]) / "resolvability_summary.json").read_text()
    )
    members = [f"R1_contam_HO@{seed}" for seed in (0, 6, 7, 8, 9, 10, 11, 12)]
    assert bundle_summary["emission_members"] == members
    assert mode["members"] == members
    assert [tag for tag in mapped if tag.startswith("profile_")] == [
        f"profile_{name}" for name in members
    ]
    assert mode["member_mean_all"]["n"] > 0
    out = tmp_path / "out"
    predictions = pd.read_csv(
        out / "whb_frontal_supc_clus" / simulate.PROFILE_PREDICTIONS_CSV
    )
    assert set(predictions["member"]) == {*mode["members"], "member_mean"}
    assert "cov_broad_raw" in predictions.columns
    class_depth = pd.read_csv(out / "whb_frontal_supc_clus" / simulate.CLASS_DEPTH_CSV)
    assert {"resolvable_share", "predicted_coverage"} <= set(class_depth.columns)
    assert (class_depth["profile_source"] == si.POOL_ALL).all()
    text = (out / simulate.REPORT_TXT).read_text()
    assert text.index("under the depth profile") < text.index("secondary")
    saved = json.loads((out / simulate.REPORT_JSON).read_text())
    assert saved["settings"]["profile_mode"] is True
    with pytest.raises(simulate.SimulationError, match="profile mode needs"):
        run_panel_simulation(
            gene_list=gene_list,
            species="human",
            name="x",
            config=config,
            store=store,
            builds=lambda panel: [build],
            out_dir=tmp_path / "out2",
            scratch_dir=tmp_path / "sim_scratch",
            profile_mode=True,
        )


def test_profile_draws_are_keyed_per_class(
    cells: res.HeldOutCells, profile_query: Any
) -> None:
    profile = _profile()
    obs = profile_query.obs
    for cls, source in (("A", "A"), ("B", si.POOL_NON_NEURONAL)):
        rng = np.random.default_rng(res.draw_key(0, "R1_contam_HO", "depth", cls))
        values = profile.values(source)
        expected = values[rng.integers(0, len(values), size=640)]
        rows = obs[obs["truth_class_sampled"] == cls]
        own = ~rows["truncated"].to_numpy()
        np.testing.assert_allclose(rows["target_total"].to_numpy()[own], expected[own])


def test_profile_spill_partners_reach_the_spill_amount() -> None:
    shallow = make_test_cells(20, seed=6, shallow_class="B")
    truth = shallow.obs[f"{res.TRUTH_PREFIX}L1"].to_numpy()
    profile = si.DepthProfile(
        ["A"] * 100 + ["B"] * 100,
        [125.0] * 100 + [30.0] * 100,
        species="mouse",
        label="x",
        neuronal={"A": True, "B": False},
    )
    recipe = res.member_recipe(res.DECISION_RECIPE, 0, CONFIG)
    query = simulate.simulate_on_profile(shallow, profile, truth, recipe, GRID)
    native = pd.Series(shallow.native_counts, index=shallow.obs.index)
    obs = query.obs
    assert (
        native.reindex(obs["partner_id"]).to_numpy() >= 0.25 * obs["host_depth"]
    ).all()
    hosts_a = obs[obs["truth_class_sampled"] == "A"]
    too_shallow = set(
        native[(native < 25) & (shallow.obs[res.SPILL_GROUP_COLUMN] == "B")].index
    )
    assert too_shallow and not too_shallow & set(hosts_a["partner_id"])
