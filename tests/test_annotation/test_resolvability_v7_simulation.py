"""Tests for the resolvability version-7 simulation (plan §8.3 v7.1-v7.4; M3c)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation import resolvability as res
from merxen.annotation import sim_inputs as si
from merxen.annotation.config import AnnotationResolvabilityConfig
from merxen.annotation.store import compute_build_hash

from .test_resolvability import cells_subset, make_test_cells

CONFIG = AnnotationResolvabilityConfig()


def _wide_efficiency(n_genes: int, seed: int = 4) -> np.ndarray:
    """Efficiencies spread 1/8x-8x, so efficient genes clip at p = 1."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(1 / 8), np.log(8), n_genes))


def _matrix(n_rows: int = 30, n_genes: int = 40, seed: int = 1) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = rng.poisson(rng.uniform(0.5, 25.0, (n_rows, n_genes)))
    return sparse.csr_matrix(dense.astype(np.float64))


# --------------------------------------------------------------------------
# Exact-total thinning


def test_exact_thinning_reaches_its_target_in_expectation() -> None:
    matrix = _matrix()
    efficiency = _wide_efficiency(matrix.shape[1])
    native = np.asarray(matrix.sum(axis=1)).ravel()
    targets = 0.6 * native
    expected = res.expected_thinned_totals(matrix, targets, efficiency)
    np.testing.assert_allclose(expected, targets, rtol=1e-6)
    # Version 6's unclipped scale falls short once genes clip at p = 1.
    weighted = np.asarray(matrix.multiply(efficiency[None, :]).sum(axis=1)).ravel()
    p_v6 = np.clip((targets / weighted)[:, None] * efficiency[None, :], 0.0, 1.0)
    v6_expected = np.asarray(matrix.multiply(p_v6).sum(axis=1)).ravel()
    assert np.all(v6_expected <= targets + 1e-9)
    assert float(np.max(1 - v6_expected / targets)) > 0.02


def test_exact_thinning_is_unbiased_over_draws() -> None:
    matrix = _matrix(n_rows=1, n_genes=60, seed=7)
    efficiency = _wide_efficiency(60, seed=9)
    target = 0.5 * float(matrix.sum())
    keys = [res.draw_key("unbiased", index) for index in range(600)]
    stacked = sparse.vstack([matrix] * len(keys)).tocsr()
    thinned = res.thin_rows_exact(stacked, np.full(len(keys), target), efficiency, keys)
    totals = np.asarray(thinned.sum(axis=1)).ravel()
    standard_error = float(np.std(totals, ddof=1)) / np.sqrt(len(totals))
    assert abs(float(np.mean(totals)) - target) < 4 * standard_error


def test_exact_thinning_never_raises_a_count() -> None:
    rng = np.random.default_rng(11)
    for trial in range(20):
        matrix = _matrix(n_rows=12, n_genes=25, seed=100 + trial)
        native = np.asarray(matrix.sum(axis=1)).ravel()
        targets = native * rng.uniform(0.0, 1.5, len(native))
        efficiency = _wide_efficiency(25, seed=trial)
        keys = [res.draw_key("raise", trial, row) for row in range(12)]
        thinned = res.thin_rows_exact(matrix, targets, efficiency, keys)
        assert np.all((matrix - thinned).toarray() >= 0)
        full = targets >= native
        np.testing.assert_array_equal(thinned.toarray()[full], matrix.toarray()[full])


def test_an_exactly_thinned_row_depends_only_on_its_row_and_key() -> None:
    matrix = _matrix(n_rows=20, n_genes=30, seed=3)
    efficiency = _wide_efficiency(30)
    native = np.asarray(matrix.sum(axis=1)).ravel()
    keys = [res.draw_key("own", row) for row in range(20)]
    together = res.thin_rows_exact(matrix, 0.3 * native, efficiency, keys)
    for row in (0, 7, 19):
        alone = res.thin_rows_exact(
            matrix[row], [0.3 * native[row]], efficiency, [keys[row]]
        )
        np.testing.assert_array_equal(alone.toarray()[0], together.toarray()[row])


def test_exact_thinning_refuses_mismatched_keys() -> None:
    with pytest.raises(res.ResolvabilityError, match="keys"):
        res.thin_rows_exact(_matrix(n_rows=3), [1.0, 1.0, 1.0], np.ones(40), [1, 2])


# --------------------------------------------------------------------------
# Totals-based grid


@pytest.fixture(scope="module")
def test_cells() -> res.HeldOutCells:
    return make_test_cells(20, seed=5)


def test_grid_values_are_totals_with_host_and_spill_shares(
    test_cells: res.HeldOutCells,
) -> None:
    recipe = res.member_recipe(res.DECISION_RECIPE, 0, CONFIG)
    grid = [100, 200]
    query = res.thin_and_contaminate_v7(test_cells, grid, recipe)
    native = test_cells.native_counts
    for depth in grid:
        target = depth / 1.25
        rows = query.obs[query.obs["depth"] == depth]
        eligible = set(test_cells.obs.index[native >= target])
        assert set(rows["cell_id"]) == eligible
        assert query.n_by_depth[depth] == len(eligible)
        # Hosts reach the host share, not D (version 6 needed native >= D).
        assert depth == 100 or any(
            native[test_cells.obs.index.get_loc(cell)] < depth for cell in eligible
        )
        assert rows["host_target"].eq(target).all()
        assert abs(rows["host_counts"].mean() - target) < 0.05 * target
        assert abs(rows["spill_counts"].mean() - 0.25 * target) < 0.1 * target
        assert abs(rows["total_counts"].mean() - depth) < 0.05 * depth
        partners = rows["partner_id"].astype(str)
        groups = test_cells.obs[res.SPILL_GROUP_COLUMN]
        assert (
            groups.reindex(partners).to_numpy()
            != groups.reindex(rows["cell_id"]).to_numpy()
        ).all()
        partner_native = pd.Series(native, index=test_cells.obs.index).reindex(partners)
        assert (partner_native.to_numpy() >= 0.25 * target).all()
    assert query.obs["member"].eq("R1_contam_HO@0").all()


def test_clean_hosts_reach_the_grid_total(test_cells: res.HeldOutCells) -> None:
    recipe = res.member_recipe(res.CLEAN_RECIPE, 0, CONFIG)
    query = res.thin_and_contaminate_v7(test_cells, [150], recipe)
    native = test_cells.native_counts
    assert set(query.obs["cell_id"]) == set(test_cells.obs.index[native >= 150])
    assert query.obs["spill_counts"].eq(0).all()
    # Clean thinning with unit efficiency is exact in expectation.
    assert abs(query.obs["total_counts"].mean() - 150) < 0.02 * 150


def test_a_v7_simulated_cell_does_not_depend_on_the_other_test_cells(
    test_cells: res.HeldOutCells,
) -> None:
    recipe = res.member_recipe(res.CLEAN_RECIPE, 0, CONFIG)
    full = res.thin_and_contaminate_v7(test_cells, [100], recipe)
    kept = list(test_cells.obs.index[::3])
    subset = res.thin_and_contaminate_v7(cells_subset(test_cells, kept), [100], recipe)
    ids = [f"{cell}|D100" for cell in kept if f"{cell}|D100" in subset.obs.index]
    full_rows = full.counts[[full.obs.index.get_loc(item) for item in ids]].toarray()
    subset_rows = subset.counts[
        [subset.obs.index.get_loc(item) for item in ids]
    ].toarray()
    np.testing.assert_array_equal(full_rows, subset_rows)


def test_member_seeds_redraw_efficiency_and_cells(test_cells: res.HeldOutCells) -> None:
    first = res.thin_and_contaminate_v7(
        test_cells, [100], res.member_recipe(res.DECISION_RECIPE, 0, CONFIG)
    )
    second = res.thin_and_contaminate_v7(
        test_cells, [100], res.member_recipe(res.DECISION_RECIPE, 1, CONFIG)
    )
    again = res.thin_and_contaminate_v7(
        test_cells, [100], res.member_recipe(res.DECISION_RECIPE, 0, CONFIG)
    )
    assert (first.counts != again.counts).nnz == 0
    assert (first.counts != second.counts).nnz > 0
    np.testing.assert_array_equal(
        res.member_efficiency(first.recipe, test_cells.genes),
        res.gene_efficiency(len(test_cells.genes), 0.8, 0),
    )


# --------------------------------------------------------------------------
# Members and recipes


def test_lognormal_recipes_keep_their_version_6_record() -> None:
    recipe = res.member_recipe(res.DECISION_RECIPE, 2, CONFIG)
    assert list(recipe.to_json()) == [
        "name",
        "version",
        "gene_efficiency_sigma",
        "spill_fraction",
        "seed",
    ]
    assert recipe.member == "R1_contam_HO@2"
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    r3 = res.member_recipe(res.R3_RECIPE, 0, CONFIG, table=table)
    record = r3.to_json()
    assert record["efficiency_source"] == "measured"
    assert record["efficiency_table_sha256"] == table.sha256
    assert record["table_rule"] == "restricted"
    assert record["residual_sd_log2"] == 0.20
    assert res.RECIPE_VERSIONS[res.R3_RECIPE] == 1


R1_PRODUCTION_WITH_TABLE = [f"R1_contam_HO@{seed}" for seed in (0, 6, 7, 8, 9, 10)]
R1_PRODUCTION_WITHOUT_TABLE = [
    f"R1_contam_HO@{seed}" for seed in (0, 6, 7, 8, 9, 10, 11, 12)
]


def test_ensemble_members_follow_the_family_table() -> None:
    # Eight emission members per version-7 family (amendment of 2026-09-29,
    # pre-registration §22.3): R1 x 6 + R3 x 2 with a measured table, else
    # R1 x 8; clean reported; the human Prime lung stress recipe reported only.
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    stress = si.get_asset(si.STRESS_HUMAN_LUNG)
    mouse = res.ensemble_members(
        CONFIG, species="mouse", chemistry="xenium_prime", member_table=table
    )
    assert [(item.name, item.role) for item in mouse] == [
        *[(name, "emission") for name in R1_PRODUCTION_WITH_TABLE],
        ("R3_measured_HO@2", "emission"),
        ("R3_measured_HO@3", "emission"),
        ("clean@0", "reported"),
    ]
    human = res.ensemble_members(
        CONFIG, species="human", chemistry="xenium_prime", stress_table=stress
    )
    assert [(item.name, item.role) for item in human] == [
        *[(name, "emission") for name in R1_PRODUCTION_WITHOUT_TABLE],
        ("clean@0", "reported"),
        ("R1_xtissue_lung_stress@0", "stress"),
    ]
    # Custom, MERSCOPE (M13) and unknown families, and the version-6 families'
    # diagnostic: no table, R1 x 8.
    for chemistry in ("merscope", "unknown", "xenium_v1"):
        other = res.ensemble_members(
            CONFIG, species="human", chemistry=chemistry, stress_table=stress
        )
        assert [item.name for item in other] == [
            *R1_PRODUCTION_WITHOUT_TABLE,
            "clean@0",
        ]
    mouse_other = res.ensemble_members(CONFIG, species="mouse", chemistry="merscope")
    assert [item.name for item in mouse_other if item.role == "emission"] == (
        R1_PRODUCTION_WITHOUT_TABLE
    )
    # R3 members carry the R3 residual (default 0.20 log2, or as passed).
    assert {item.recipe.residual_sd_log2 for item in mouse[6:8]} == {0.20}
    residual = res.ensemble_members(
        CONFIG,
        species="mouse",
        chemistry="xenium_prime",
        member_table=table,
        residual_sd_log2=0.3,
    )
    assert [item.recipe.residual_sd_log2 for item in residual[6:8]] == [0.3, 0.3]
    for members in (mouse, human, mouse_other):
        emission = [item for item in members if item.role == "emission"]
        assert len(emission) == res.V7_EMISSION_MEMBERS == 8
        assert emission[0].name == "R1_contam_HO@0"
        assert all(item.recipe.name != res.LUNG_STRESS_RECIPE for item in emission)
    with pytest.raises(res.ResolvabilityError, match="cross species"):
        res.ensemble_members(
            CONFIG, species="human", chemistry="xenium_prime", member_table=table
        )


def test_member_seeds_override_and_the_comparator_is_disjoint() -> None:
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    stage_d = res.ensemble_members(
        CONFIG,
        species="mouse",
        chemistry="xenium_prime",
        member_table=table,
        r1_seeds=res.V7_STAGE_D_R1_SEEDS,
        r3_seeds=res.V7_STAGE_D_R3_SEEDS,
    )
    assert [item.name for item in stage_d if item.role == "emission"] == [
        "R1_contam_HO@0",
        "R1_contam_HO@1",
        "R1_contam_HO@2",
        "R3_measured_HO@0",
    ]
    # R3 seeds are ignored without a table.
    no_table = res.ensemble_members(
        CONFIG, species="mouse", chemistry="merscope", r1_seeds=(4,), r3_seeds=(9,)
    )
    assert [item.name for item in no_table] == ["R1_contam_HO@4", "clean@0"]
    with pytest.raises(res.ResolvabilityError, match="repeat"):
        res.ensemble_members(
            CONFIG, species="mouse", chemistry="merscope", r1_seeds=(1, 1)
        )
    with pytest.raises(res.ResolvabilityError, match="repeat"):
        res.ensemble_members(
            CONFIG,
            species="mouse",
            chemistry="xenium_prime",
            member_table=table,
            r3_seeds=(2, 2),
        )
    assert res.default_member_seeds(True) == ((0, 6, 7, 8, 9, 10), (2, 3))
    assert res.default_member_seeds(False) == ((0, 6, 7, 8, 9, 10, 11, 12), ())
    assert res.default_member_seeds(True, comparator=True) == (
        (20, 21, 22, 23, 24, 25),
        (20, 21),
    )
    assert res.default_member_seeds(False, comparator=True) == (
        (20, 21, 22, 23, 24, 25, 26, 27),
        (),
    )
    # The comparator shares no member with production or stage D's A and B;
    # production shares only the pre-registered R1@0 with A.
    stage_a = {("R1", 0), ("R1", 1), ("R1", 2), ("R3", 0)}
    stage_b = {("R1", 3), ("R1", 4), ("R1", 5), ("R3", 1)}
    for has_table in (True, False):
        production = {
            (name, seed)
            for name, seeds in zip(
                ("R1", "R3"), res.default_member_seeds(has_table), strict=True
            )
            for seed in seeds
        }
        comparator = {
            (name, seed)
            for name, seeds in zip(
                ("R1", "R3"),
                res.default_member_seeds(has_table, comparator=True),
                strict=True,
            )
            for seed in seeds
        }
        assert len(production) == len(comparator) == 8
        assert not production & comparator
        assert not comparator & (stage_a | stage_b)
        assert production & stage_a == {("R1", 0)}
        assert not production & stage_b


def test_members_from_names_rebuild_recorded_members() -> None:
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    names = ["R1_contam_HO@1", "R3_measured_HO@0", "R1_xtissue_lung_stress@0"]
    members = res.members_from_names(
        names,
        CONFIG,
        member_table=table,
        stress_table=si.get_asset(si.STRESS_HUMAN_LUNG),
    )
    assert [item.name for item in members] == names
    assert [item.role for item in members] == ["emission"] * 3
    assert members[1].recipe == res.member_recipe(res.R3_RECIPE, 0, CONFIG, table=table)
    assert res.parse_member_name("R1_contam_HO@12") == ("R1_contam_HO", 12)
    for bad in ("R1_contam_HO", "@3", "R1_contam_HO@x"):
        with pytest.raises(res.ResolvabilityError, match="member name"):
            res.parse_member_name(bad)


def test_member_efficiency_of_the_table_recipes() -> None:
    genes = sorted(si.efficiency_table(si.get_asset(si.EFFICIENCY_MOUSE_PRIME)).index)[
        :200
    ]
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    r3 = res.member_recipe(res.R3_RECIPE, 0, CONFIG, table=table)
    expected = si.r3_efficiency(genes, si.efficiency_table(table), seed=0).efficiency
    np.testing.assert_array_equal(res.member_efficiency(r3, genes), expected)
    altered = r3.__class__(**{**r3.__dict__, "efficiency_table_sha256": "0" * 64})
    with pytest.raises(res.ResolvabilityError, match="sha256"):
        res.member_efficiency(altered, genes)
    stress = res.member_recipe(
        res.LUNG_STRESS_RECIPE, 0, CONFIG, table=si.get_asset(si.STRESS_HUMAN_LUNG)
    )
    human_genes = ["ENSG00000159640", "ENSG99999999999"]
    efficiency = res.member_efficiency(stress, human_genes)
    assert float(np.median(efficiency)) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Version selection, grid and build hash


def test_version_selection_truth_table() -> None:
    pins = res.load_v6_pins()
    assert list(pins["family_id"]) == ["human_merscope_ed7bc6bed989"]
    cases = {
        # listed and inherited panels carry their validated family's id
        ("human_set_a", None): 6,
        ("mouse_ag7", None): 6,
        ("mouse_vzg2", None): 6,
        # pinned family (OD-E20) and a panel hash named by a pin
        ("human_merscope_ed7bc6bed989", None): 6,
        ("human_merscope_other", str(pins["panel_hash"].iloc[0])): 6,
        # a listed panel hash whose family is not known (build-hash payloads)
        (None, "6e5fd5fb86ef0c3eaccf2efa5b792c9515b087fdf9e9cb6a3978d53268253dc3"): 6,
        (None, "f4c291864ba1c91153a3df00429da13abc4ed1d39592d5ee80c23090040b4013"): 6,
        # every other family: version 7
        ("mouse_xenium_4bc22b961aca", None): 7,
        ("human_xenium_93ecc58ed5c4", None): 7,
        ("mouse_merscope_custom", "f" * 64): 7,
        (None, None): 7,
    }
    for (family, panel_hash), version in cases.items():
        assert res.resolvability_version_for(family, panel_hash) == version, family


def test_a_simulation_validated_family_stays_on_version_7() -> None:
    table = SimpleNamespace(
        records=[
            SimpleNamespace(family_id="human_set_a", validation_basis="real_data"),
            SimpleNamespace(family_id="mouse_gatep", validation_basis="simulation"),
        ]
    )
    assert res.resolvability_version_for("human_set_a", validated=table) == 6
    assert res.resolvability_version_for("mouse_gatep", validated=table) == 7


def test_v7_grid_above_1000_genes() -> None:
    assert res.v7_depth_grid("mouse", 5006) == list(res.V7_LARGE_PANEL_GRID)
    assert len(res.V7_LARGE_PANEL_GRID) == 13
    assert res.v7_depth_grid("human", 5001)[-1] == 3000
    assert res.v7_depth_grid("mouse", 500) == [10, 20, 50, 100, 250, 500, 1000, 2000]
    assert res.v7_depth_grid("human", 268) == [10, 15, 30, 60, 120, 250]
    assert res.v7_depth_grid("mouse", 5006, [10, 20]) == [10, 20]
    ratios = [
        later / earlier
        for earlier, later in zip(
            res.V7_LARGE_PANEL_GRID, res.V7_LARGE_PANEL_GRID[1:], strict=False
        )
        if earlier >= 100
    ]
    assert max(ratios) <= 1.67


def test_asset_sha256_enters_the_version_7_payload() -> None:
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    profile = si.get_asset(si.PROFILE_MOUSE_PRIME_FF)
    members = res.ensemble_members(
        CONFIG, species="mouse", chemistry="xenium_prime", member_table=table
    )
    payload = res.v7_simulation_payload(
        members=members,
        assets=[table, profile],
        chemistry="xenium_prime",
        depth_grid=res.V7_LARGE_PANEL_GRID,
        top_up={"min_class_test_cells": 200},
    )
    assert payload["resolvability_version"] == 7
    assert payload["ensemble_rule_version"] == res.ENSEMBLE_RULE_VERSION == 2
    assert payload["assets"][table.asset_id]["sha256"] == table.sha256
    r3 = [item for item in payload["members"] if item["member"].startswith("R3_")]
    assert [item["member"] for item in r3] == ["R3_measured_HO@2", "R3_measured_HO@3"]
    assert {item["recipe"]["efficiency_table_sha256"] for item in r3} == {table.sha256}
    changed = table.model_copy(update={"sha256": "0" * 64})
    other = res.v7_simulation_payload(
        members=members,
        assets=[changed, profile],
        chemistry="xenium_prime",
        depth_grid=res.V7_LARGE_PANEL_GRID,
        top_up={"min_class_test_cells": 200},
    )
    assert compute_build_hash(cast("Any", payload)) != compute_build_hash(
        cast("Any", other)
    )
    # Version 6 records hold no asset (their recipes keep five keys).
    v6 = res.simulation_recipes(CONFIG)
    assert all("efficiency_table_sha256" not in item.to_json() for item in v6)


def test_rows_the_fixed_point_leaves_short_are_solved_exactly() -> None:
    matrix = _matrix(n_rows=40, n_genes=50, seed=21)
    efficiency = _wide_efficiency(50, seed=22)
    native = np.asarray(matrix.sum(axis=1)).ravel()
    targets = native * np.linspace(0.05, 0.99, 40)
    exact = res.expected_thinned_totals(matrix, targets, efficiency, max_iter=0)
    np.testing.assert_allclose(exact, targets, rtol=1e-9)
    # Two fixed-point steps leave these rows short; the exact finish does not.
    iterated = res.expected_thinned_totals(matrix, targets, efficiency, max_iter=2)
    np.testing.assert_allclose(iterated, targets, rtol=1e-9)


def test_spill_donors_reach_the_spill_amount() -> None:
    test = make_test_cells(20, seed=8, shallow_class="B")
    recipe = res.member_recipe(res.DECISION_RECIPE, 0, CONFIG)
    query = res.thin_and_contaminate_v7(test, [150], recipe)
    native = pd.Series(test.native_counts, index=test.obs.index)
    amount = 0.25 * 150 / 1.25
    partners = query.obs["partner_id"].astype(str)
    assert (native.reindex(partners).to_numpy() >= amount).all()
    # Shallow cells below the spill amount exist but are never partners.
    groups = test.obs[res.SPILL_GROUP_COLUMN]
    shallow = set(native[(native < amount) & (groups == "B")].index)
    assert shallow and not shallow & set(partners)


def test_recipe_versions_agree_between_store_and_resolvability() -> None:
    from merxen.annotation.store import RESOLVABILITY_RECIPE_VERSIONS

    for name, version in RESOLVABILITY_RECIPE_VERSIONS.items():
        assert res.RECIPE_VERSIONS[name] == version
    assert RESOLVABILITY_RECIPE_VERSIONS[res.R3_RECIPE] == 1
    # Gate P's NP6 stress recipes never enter a bundle, so the store does not
    # version them.
    for name in (
        res.GATE_P_STRESS_SPILL,
        res.GATE_P_STRESS_LOGNORMAL,
        res.GATE_P_STRESS_XPLATFORM,
        res.GATE_P_STRESS_R3_SPILL,
    ):
        assert res.RECIPE_VERSIONS[name] == 1
        assert name not in RESOLVABILITY_RECIPE_VERSIONS


# --------------------------------------------------------------------------
# Gate P's NP6 stress recipes (M13; plan §14 NP6)


def _stress_names(
    stresses: dict[str, list[res.EnsembleMember]],
) -> dict[str, list[str]]:
    return {name: [item.name for item in items] for name, items in stresses.items()}


def test_np6_stresses_each_r1_member_three_ways() -> None:
    """§14 NP6: spill 0.35, LogNormal(0, 1.0) and the cross-platform offsets.

    The new-panel human MERSCOPE family has R1 x 8 (no R3, no lung): each
    member gets the three stresses at its own seed, each changing one thing.
    """
    xplatform = si.get_asset(si.STRESS_HUMAN_XPLATFORM)
    members = res.ensemble_members(CONFIG, species="human", chemistry="merscope")
    stresses = res.gate_p_stress_members(
        members,
        CONFIG,
        species="human",
        chemistry="merscope",
        xplatform_table=xplatform,
    )
    assert _stress_names(stresses) == {
        name: [
            f"{recipe}@{name.split('@')[1]}"
            for recipe in (
                res.GATE_P_STRESS_SPILL,
                res.GATE_P_STRESS_LOGNORMAL,
                res.GATE_P_STRESS_XPLATFORM,
            )
        ]
        for name in R1_PRODUCTION_WITHOUT_TABLE
    }
    assert {item.role for items in stresses.values() for item in items} == {"stress"}
    base = res.member_recipe(res.DECISION_RECIPE, 6, CONFIG)
    spill, lognormal, offsets = (item.recipe for item in stresses["R1_contam_HO@6"])
    assert (spill.spill_fraction, spill.gene_efficiency_sigma) == (0.35, 0.8)
    assert (lognormal.spill_fraction, lognormal.gene_efficiency_sigma) == (0.25, 1.0)
    assert (offsets.spill_fraction, offsets.gene_efficiency_sigma) == (0.25, 0.8)
    assert {spill.seed, lognormal.seed, offsets.seed} == {base.seed}
    assert spill.efficiency_source == lognormal.efficiency_source == "lognormal"
    assert list(spill.to_json()) == list(base.to_json())
    record = offsets.to_json()
    assert record["efficiency_source"] == "xplatform_stress"
    assert record["efficiency_table"] == si.STRESS_HUMAN_XPLATFORM
    assert record["efficiency_table_sha256"] == xplatform.sha256
    assert record["factor_cap_log2"] == 2.0
    # The registered recipes' records keep their keys (version-7 build_hash).
    r3 = res.member_recipe(
        res.R3_RECIPE, 2, CONFIG, table=si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    )
    lung = res.member_recipe(
        res.LUNG_STRESS_RECIPE, 0, CONFIG, table=si.get_asset(si.STRESS_HUMAN_LUNG)
    )
    for recipe in (r3, lung):
        assert "factor_cap_log2" not in recipe.to_json()
    # A version-6 family's base is its R1@0 recipe.
    (v6_base, _clean) = res.simulation_recipes(CONFIG)
    v6 = res.gate_p_stress_members(
        [res.EnsembleMember(v6_base, "emission")],
        CONFIG,
        species="human",
        chemistry="merscope",
        xplatform_table=xplatform,
    )
    assert _stress_names(v6) == {
        "R1_contam_HO@0": [
            "R1_stress_spill@0",
            "R1_stress_lognormal@0",
            "R1_stress_xplatform@0",
        ]
    }


def test_np6_stresses_r3_by_spill_only_and_adds_lung_for_human_prime() -> None:
    xplatform = si.get_asset(si.STRESS_HUMAN_XPLATFORM)
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    mouse = res.ensemble_members(
        CONFIG, species="mouse", chemistry="xenium_prime", member_table=table
    )
    stresses = res.gate_p_stress_members(
        mouse,
        CONFIG,
        species="mouse",
        chemistry="xenium_prime",
        xplatform_table=xplatform,
        member_table=table,
    )
    names = _stress_names(stresses)
    assert list(names) == [
        *R1_PRODUCTION_WITH_TABLE,
        "R3_measured_HO@2",
        "R3_measured_HO@3",
    ]
    assert names["R3_measured_HO@3"] == ["R3_stress_spill@3"]
    assert len(names["R1_contam_HO@7"]) == 3
    (r3_spill,) = (item.recipe for item in stresses["R3_measured_HO@2"])
    base = next(item.recipe for item in mouse if item.name == "R3_measured_HO@2")
    assert r3_spill.spill_fraction == 0.35
    assert (
        dataclasses.replace(
            r3_spill, name=base.name, spill_fraction=base.spill_fraction
        )
        == base
    )
    lung = si.get_asset(si.STRESS_HUMAN_LUNG)
    human = res.ensemble_members(
        CONFIG, species="human", chemistry="xenium_prime", stress_table=lung
    )
    prime = res.gate_p_stress_members(
        human,
        CONFIG,
        species="human",
        chemistry="xenium_prime",
        xplatform_table=xplatform,
        lung_table=lung,
    )
    assert list(prime) == R1_PRODUCTION_WITHOUT_TABLE
    assert _stress_names(prime)["R1_contam_HO@9"][-1] == "R1_xtissue_lung_stress@9"
    lung_member = prime["R1_contam_HO@9"][-1].recipe
    assert lung_member == res.member_recipe(
        res.LUNG_STRESS_RECIPE, 9, CONFIG, table=lung
    )
    with pytest.raises(res.ResolvabilityError, match="lung ratio table"):
        res.gate_p_stress_members(
            human,
            CONFIG,
            species="human",
            chemistry="xenium_prime",
            xplatform_table=xplatform,
        )


def test_np6_stress_members_refuse_what_they_cannot_stress() -> None:
    xplatform = si.get_asset(si.STRESS_HUMAN_XPLATFORM)
    lung = si.get_asset(si.STRESS_HUMAN_LUNG)
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    options: dict[str, Any] = {
        "species": "human",
        "chemistry": "merscope",
        "xplatform_table": xplatform,
    }
    clean = res.member_recipe(res.CLEAN_RECIPE, 0, CONFIG)
    with pytest.raises(res.ResolvabilityError, match="only R1 and R3"):
        res.gate_p_stress_recipes(clean, CONFIG, **options)
    other = res.member_recipe(
        res.DECISION_RECIPE, 0, AnnotationResolvabilityConfig(spill_fraction=0.3)
    )
    with pytest.raises(res.ResolvabilityError, match="does not reproduce"):
        res.gate_p_stress_recipes(other, CONFIG, **options)
    r3 = res.member_recipe(res.R3_RECIPE, 2, CONFIG, table=table)
    with pytest.raises(res.ResolvabilityError, match="member table"):
        res.gate_p_stress_recipes(r3, CONFIG, **options)
    with pytest.raises(res.ResolvabilityError, match="cross-platform"):
        res.member_recipe(res.GATE_P_STRESS_XPLATFORM, 0, CONFIG, table=lung)
    with pytest.raises(res.ResolvabilityError, match="lung ratio table"):
        res.member_recipe(res.LUNG_STRESS_RECIPE, 0, CONFIG, table=xplatform)
    with pytest.raises(res.ResolvabilityError, match="needs its"):
        res.member_recipe(res.GATE_P_STRESS_XPLATFORM, 0, CONFIG)
    with pytest.raises(res.ResolvabilityError, match="member table"):
        res.member_recipe(res.GATE_P_STRESS_R3_SPILL, 0, CONFIG, table=xplatform)
    reported = [
        item
        for item in res.ensemble_members(CONFIG, species="human", chemistry="merscope")
        if item.role != "emission"
    ]
    with pytest.raises(res.ResolvabilityError, match="at least one emission"):
        res.gate_p_stress_members(reported, CONFIG, **options)
    twice = [
        res.EnsembleMember(
            res.member_recipe(res.DECISION_RECIPE, 0, CONFIG), "emission"
        )
    ] * 2
    with pytest.raises(res.ResolvabilityError, match="repeats"):
        res.gate_p_stress_members(twice, CONFIG, **options)


def test_np6_stress_parameters_follow_the_config() -> None:
    config = AnnotationResolvabilityConfig(
        gate_p_stress={
            "spill_fraction": 0.4,
            "gene_efficiency_sigma": 1.2,
            "platform_factor_cap_log2": 1.5,
        }
    )
    xplatform = si.get_asset(si.STRESS_HUMAN_XPLATFORM)
    spill = res.member_recipe(res.GATE_P_STRESS_SPILL, 3, config)
    lognormal = res.member_recipe(res.GATE_P_STRESS_LOGNORMAL, 3, config)
    offsets = res.member_recipe(res.GATE_P_STRESS_XPLATFORM, 3, config, table=xplatform)
    assert spill.spill_fraction == 0.4
    assert lognormal.gene_efficiency_sigma == 1.2
    assert offsets.factor_cap_log2 == 1.5
    genes = [f"G{index:03d}" for index in range(101)]
    capped = res.member_efficiency(offsets, genes)
    base = res.gene_efficiency(len(genes), 0.8, 3)
    offset = np.log2(capped) - np.log2(base)
    assert float(offset.max() - offset.min()) <= 3.0 + 1e-9


def test_np6_lognormal_stress_scales_the_members_normals() -> None:
    """§3.7: LogNormal(0, 1.0) stresses the member's own draw (sigma 0.8)."""
    genes = [f"G{index:03d}" for index in range(101)]
    for seed in (0, 6, 12):
        base = res.member_efficiency(
            res.member_recipe(res.DECISION_RECIPE, seed, CONFIG), genes
        )
        stressed = res.member_efficiency(
            res.member_recipe(res.GATE_P_STRESS_LOGNORMAL, seed, CONFIG), genes
        )
        np.testing.assert_allclose(np.log(stressed), 1.25 * np.log(base), atol=1e-12)


def test_np6_cross_platform_stress_multiplies_the_members_draw() -> None:
    xplatform = si.get_asset(si.STRESS_HUMAN_XPLATFORM)
    factors = si.xplatform_factors(xplatform)
    genes = sorted([*factors.index[:40], "ENSG99999999998", "ENSG99999999999"])
    recipe = res.member_recipe(res.GATE_P_STRESS_XPLATFORM, 7, CONFIG, table=xplatform)
    expected, measured = si.xplatform_stress_efficiency(
        genes, res.gene_efficiency(len(genes), 0.8, 7), factors, seed=7, cap_log2=2.0
    )
    np.testing.assert_array_equal(res.member_efficiency(recipe, genes), expected)
    assert int((~measured).sum()) == 2


def test_version_6_thinning_reads_a_table_recipe(
    test_cells: res.HeldOutCells, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version-6 family's NP6 cross-platform stress uses its table's efficiency."""
    xplatform = si.get_asset(si.STRESS_HUMAN_XPLATFORM)
    recipe = res.member_recipe(res.GATE_P_STRESS_XPLATFORM, 0, CONFIG, table=xplatform)
    efficiency = np.ones(len(test_cells.genes))
    efficiency[:10] = 0.0
    calls: list[str] = []

    def table_efficiency(
        given: res.SimulationRecipe, genes: Any, registry: Any = None
    ) -> np.ndarray:
        calls.append(given.member)
        return efficiency

    monkeypatch.setattr(res, "member_efficiency", table_efficiency)
    query = res.thin_and_contaminate(test_cells, [100], recipe)
    assert calls == ["R1_stress_xplatform@0"]
    assert query.counts.shape[0] > 0
    assert query.counts[:, :10].nnz == 0
    with pytest.raises(res.ResolvabilityError, match="efficiency seed"):
        res.thin_and_contaminate(test_cells, [100], recipe, efficiency_seed=1)
    # A lognormal recipe keeps its own draw (never the table path).
    res.thin_and_contaminate(
        test_cells, [100], res.member_recipe(res.GATE_P_STRESS_SPILL, 0, CONFIG)
    )
    assert calls == ["R1_stress_xplatform@0"]
