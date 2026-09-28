"""Tests for the resolvability version-7 simulation (plan §8.3 v7.1-v7.4; M3c)."""

from __future__ import annotations

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


def test_ensemble_members_follow_the_family_table() -> None:
    table = si.get_asset(si.EFFICIENCY_MOUSE_PRIME)
    stress = si.get_asset(si.STRESS_HUMAN_LUNG)
    mouse = res.ensemble_members(
        CONFIG, species="mouse", chemistry="xenium_prime", member_table=table
    )
    assert [(item.name, item.role) for item in mouse] == [
        ("R1_contam_HO@0", "emission"),
        ("R1_contam_HO@1", "emission"),
        ("R1_contam_HO@2", "emission"),
        ("R3_measured_HO@0", "emission"),
        ("clean@0", "reported"),
    ]
    human = res.ensemble_members(
        CONFIG, species="human", chemistry="xenium_prime", stress_table=stress
    )
    assert [(item.name, item.role) for item in human] == [
        ("R1_contam_HO@0", "emission"),
        ("R1_contam_HO@1", "emission"),
        ("R1_contam_HO@2", "emission"),
        ("clean@0", "reported"),
        ("R1_xtissue_lung_stress@0", "stress"),
    ]
    other = res.ensemble_members(
        CONFIG, species="human", chemistry="merscope", stress_table=stress
    )
    assert [item.name for item in other] == [
        "R1_contam_HO@0",
        "R1_contam_HO@1",
        "R1_contam_HO@2",
        "clean@0",
    ]
    with pytest.raises(res.ResolvabilityError, match="cross species"):
        res.ensemble_members(
            CONFIG, species="human", chemistry="xenium_prime", member_table=table
        )


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
    assert payload["assets"][table.asset_id]["sha256"] == table.sha256
    assert payload["members"][3]["recipe"]["efficiency_table_sha256"] == table.sha256
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
