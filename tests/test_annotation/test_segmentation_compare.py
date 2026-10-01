"""Tests of the M8b segmentation-comparison helpers (pre-registration §19)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import shapely
from scipy import sparse, stats

from merxen.annotation import segmentation_compare as sc

# ---------------------------------------------------------------------------
# Depth matching


def _counts() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.array(
            [
                [5, 3, 0, 2, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [10, 0, 0, 0, 7],
                [40, 30, 20, 10, 5],
            ]
        )
    )


IDS = ["c1", "c2", "c3", "c4", "c5"]
TARGETS = np.array([4, 0, 10, 3, 33])


def test_thin_to_targets_reaches_min_of_total_and_target() -> None:
    result = sc.thin_to_targets(_counts(), IDS, TARGETS)
    totals = np.asarray(_counts().sum(axis=1)).ravel()
    np.testing.assert_array_equal(result.totals_before, totals)
    np.testing.assert_array_equal(result.totals_after, np.minimum(totals, TARGETS))
    np.testing.assert_array_equal(result.thinned, totals > TARGETS)


def test_thin_to_targets_draws_without_replacement_within_each_gene() -> None:
    original = _counts().toarray()
    thinned = sc.thin_to_targets(_counts(), IDS, TARGETS).counts.toarray()
    assert (thinned <= original).all()
    assert (thinned >= 0).all()
    assert thinned.dtype.kind == "i"


def test_thin_to_targets_keeps_cells_at_or_below_target() -> None:
    original = _counts().toarray()
    thinned = sc.thin_to_targets(_counts(), IDS, TARGETS).counts.toarray()
    np.testing.assert_array_equal(thinned[[1, 2]], original[[1, 2]])
    exact = sc.thin_to_targets(
        _counts(), IDS, np.asarray(_counts().sum(axis=1)).ravel()
    )
    np.testing.assert_array_equal(exact.counts.toarray(), original)
    assert not exact.thinned.any()


def test_thin_to_targets_target_zero_empties_the_cell() -> None:
    result = sc.thin_to_targets(_counts(), IDS, np.zeros(5, dtype=int))
    assert result.counts.nnz == 0
    assert result.totals_after.sum() == 0


def test_thin_to_targets_is_keyed_by_cell_id_not_row_order() -> None:
    first = sc.thin_to_targets(_counts(), IDS, TARGETS).counts.toarray()
    order = [4, 2, 0, 3, 1]
    shuffled = sc.thin_to_targets(
        _counts()[order], [IDS[i] for i in order], TARGETS[order]
    ).counts.toarray()
    np.testing.assert_array_equal(shuffled, first[order])
    alone = sc.thin_to_targets(_counts()[[4]], ["c5"], TARGETS[[4]]).counts.toarray()
    np.testing.assert_array_equal(alone[0], first[4])


def test_thin_to_targets_is_deterministic_and_seed_dependent() -> None:
    big = sparse.csr_matrix(np.full((1, 6), 50))
    draws = [
        sc.thin_to_targets(big, ["x"], [100], seed=0).counts.toarray() for _ in range(3)
    ]
    assert all((draw == draws[0]).all() for draw in draws)
    other_seed = sc.thin_to_targets(big, ["x"], [100], seed=1).counts.toarray()
    other_id = sc.thin_to_targets(big, ["y"], [100], seed=0).counts.toarray()
    assert not (other_seed == draws[0]).all()
    assert not (other_id == draws[0]).all()


def test_thin_to_targets_matches_the_cell_generator_draw() -> None:
    row = np.array([40, 30, 20, 10, 5])
    expected = sc.cell_generator("c5", 0).multivariate_hypergeometric(row, 33)
    got = sc.thin_to_targets(_counts(), IDS, TARGETS).counts.toarray()[4]
    np.testing.assert_array_equal(got, expected)


def test_thin_to_targets_is_unbiased_per_gene() -> None:
    row = sparse.csr_matrix(np.array([[600, 300, 100]]))
    draws = np.vstack(
        [
            sc.thin_to_targets(row, [f"cell{i}"], [100]).counts.toarray()[0]
            for i in range(400)
        ]
    )
    np.testing.assert_allclose(draws.mean(axis=0), [60, 30, 10], atol=1.5)
    assert (draws.sum(axis=1) == 100).all()


def test_thin_to_targets_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        sc.thin_to_targets(_counts(), IDS, [-1, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        sc.thin_to_targets(_counts(), IDS[:4], TARGETS)
    with pytest.raises(ValueError):
        sc.thin_to_targets(sparse.csr_matrix(np.array([[1.5]])), ["a"], [1])


def test_cell_key_depends_on_the_id_string_only() -> None:
    assert sc.cell_key(12) == sc.cell_key("12")
    assert sc.cell_key("12") != sc.cell_key("13")
    assert 0 <= sc.cell_key("abc") < 2**64


# ---------------------------------------------------------------------------
# Transcripts in nuclei


def test_points_in_polygons_includes_boundary_and_takes_lowest_index() -> None:
    polygons = [shapely.box(0, 0, 2, 2), shapely.box(1, 1, 3, 3)]
    x = np.array([0.5, 1.5, 2.5, 5.0, 2.0, 0.0])
    y = np.array([0.5, 1.5, 2.5, 5.0, 0.5, 0.0])
    found = sc.points_in_polygons(x, y, polygons, chunk_size=2)
    np.testing.assert_array_equal(found, [0, 0, 1, -1, 0, 0])


def test_points_in_polygons_without_polygons() -> None:
    np.testing.assert_array_equal(
        sc.points_in_polygons(np.array([1.0]), np.array([1.0]), []), [-1]
    )


def test_nucleus_holders_takes_the_largest_overlap() -> None:
    cells = [
        shapely.box(0, 0, 1.2, 2),
        shapely.box(1.2, 0, 4, 2),
        shapely.box(9, 9, 20, 20),
    ]
    nuclei = [
        shapely.box(0.5, 0.5, 1.5, 1.5),  # 0.7 in cell a, 0.3 in cell b
        shapely.box(1.0, 0.5, 3.0, 1.5),  # 0.2 in a, 1.8 in b
        shapely.box(5, 5, 6, 6),  # no cell
        shapely.box(4, 0, 5, 1),  # touches b's edge only
    ]
    holders = sc.nucleus_holders(nuclei, cells, ["a", "b", "c"])
    assert list(holders) == ["a", "b", None, None]


def test_nucleus_holders_breaks_ties_by_cell_order() -> None:
    cells = [shapely.box(1, 0, 2, 1), shapely.box(0, 0, 1, 1)]
    holders = sc.nucleus_holders(
        [shapely.box(0.5, 0, 1.5, 1)], cells, ["first", "second"]
    )
    assert list(holders) == ["first"]


def test_transcript_cells_prefers_the_nucleus_holder() -> None:
    nucleus_index = np.array([-1, 0, 1, -1, 2])
    holders = np.array(["n0cell", None, "n2cell"], dtype=object)
    assigned = np.array(["a", "b", "c", None, "e"], dtype=object)
    cells = sc.transcript_cells(nucleus_index, holders, assigned)
    assert list(cells) == ["a", "n0cell", None, None, "n2cell"]


def test_transcript_cells_keeps_integer_dtype() -> None:
    none = np.uint64(2**64 - 1)
    cells = sc.transcript_cells(
        np.array([0, -1]),
        np.array([7], dtype=np.uint64),
        np.array([none, 5], dtype=np.uint64),
    )
    assert cells.dtype == np.uint64
    np.testing.assert_array_equal(cells, [7, 5])


# ---------------------------------------------------------------------------
# Tallies and shares


def _tally_inputs() -> dict[str, np.ndarray]:
    return {
        "gene_codes": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "assigned_reseg": np.array([1, 0, 0, 1, 1, 0, 0, 0], bool),
        "assigned_hybrid": np.array([1, 1, 1, 1, 1, 1, 0, 0], bool),
        "assigned_reseg_same_cell": np.array([1, 0, 0, 0, 1, 0, 0, 0], bool),
        "in_nucleus": np.array([1, 1, 0, 1, 0, 1, 1, 0], bool),
        "background_reseg": np.array([0, 1, 1, 0, 0, 0, 1, 1], bool),
    }


def test_loss_tally_counts_every_field() -> None:
    inputs = _tally_inputs()
    tally = sc.loss_tally(inputs.pop("gene_codes"), 2, **inputs)
    assert tally.shape == (1, 2, len(sc.TALLY_FIELDS))
    frame = pd.DataFrame(tally[0], columns=list(sc.TALLY_FIELDS))
    assert frame["n_decoded"].tolist() == [4, 4]
    assert frame["n_assigned_reseg"].tolist() == [2, 1]
    assert frame["n_assigned_hybrid"].tolist() == [4, 2]
    assert frame["n_assigned_reseg_same_cell"].tolist() == [1, 1]
    assert frame["n_in_nucleus"].tolist() == [3, 2]
    # gene 0: in-nucleus 0, 1, 3; reseg assigned 0 and 3 -> 1 unassigned (bg)
    assert frame["n_in_nucleus_unassigned_reseg"].tolist() == [1, 2]
    assert frame["n_in_nucleus_background_reseg"].tolist() == [1, 1]
    # gene 1: transcript 5 is in a nucleus, unassigned, not background
    assert frame["n_in_nucleus_nontable_reseg"].tolist() == [0, 1]


def test_loss_tally_groups_and_drops_ungrouped() -> None:
    inputs = _tally_inputs()
    genes = inputs.pop("gene_codes")
    groups = np.array([0, 1, 0, -1, 1, 1, 5, 0])
    tally = sc.loss_tally(genes, 2, group_codes=groups, n_groups=2, **inputs)
    assert tally[:, :, 0].tolist() == [[2, 1], [1, 2]]
    assert tally[..., 0].sum() == 6


def test_loss_tally_rejects_bad_gene_codes() -> None:
    inputs = _tally_inputs()
    inputs.pop("gene_codes")
    with pytest.raises(ValueError):
        sc.loss_tally(np.array([0, 0, 0, 0, 1, 1, 1, 2]), 2, **inputs)


def test_loss_shares_definitions() -> None:
    tally = pd.DataFrame(
        {
            "n_decoded": [10, 10, 0],
            "n_assigned_reseg": [4, 0, 0],
            "n_assigned_hybrid": [8, 0, 0],
            "n_assigned_reseg_same_cell": [3, 0, 0],
            "n_in_nucleus": [5, 2, 0],
            "n_in_nucleus_unassigned_reseg": [2, 2, 0],
            "n_in_nucleus_background_reseg": [1, 2, 0],
            "n_in_nucleus_nontable_reseg": [1, 0, 0],
        }
    )
    shares = sc.loss_shares(tally)
    assert shares.loc[0, "A_reseg"] == pytest.approx(0.4)
    assert shares.loc[0, "A_hybrid"] == pytest.approx(0.8)
    assert shares.loc[0, "L"] == pytest.approx(0.5)
    assert shares.loc[0, "N_bg"] == pytest.approx(0.4)
    assert shares.loc[0, "N_bg_background"] == pytest.approx(0.2)
    assert shares.loc[0, "N_bg_nontable"] == pytest.approx(0.2)
    assert shares.loc[0, "A_reseg_same_cell"] == pytest.approx(0.3)
    assert math.isnan(shares.loc[1, "L"])
    assert shares.loc[1, "N_bg"] == pytest.approx(1.0)
    assert math.isnan(shares.loc[2, "A_reseg"])


# ---------------------------------------------------------------------------
# Distances


def test_distance_bin_codes_on_the_preregistered_edges() -> None:
    values = np.array(
        [0.0, 9.999, 10.0, 14.9, 15.0, 19.9, 20.0, 29.9, 30.0, 49.9, 50.0, 1e6]
    )
    np.testing.assert_array_equal(
        sc.distance_bin_codes(values), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    )
    np.testing.assert_array_equal(
        sc.distance_bin_codes(np.array([np.nan, -1.0, np.inf])), [-1, -1, 5]
    )
    assert sc.DISTANCE_BIN_LABELS == ("0-10", "10-15", "15-20", "20-30", "30-50", ">50")
    assert (0.0, 10.0, 15.0, 20.0, 30.0, 50.0, math.inf) == sc.DISTANCE_BIN_EDGES_UM


def test_distance_bin_codes_with_a_finite_last_edge() -> None:
    np.testing.assert_array_equal(
        sc.distance_bin_codes(np.array([0.5, 1.5, 2.0]), edges=(0.0, 1.0, 2.0)),
        [0, 1, -1],
    )
    with pytest.raises(ValueError):
        sc.distance_bin_codes(np.array([1.0]), edges=(0.0, 2.0, 1.0))


def test_nearest_other_class_distance() -> None:
    xy = np.array([[0, 0], [3, 4], [100, 0], [0, 1], [0, 2], [50, 50]], dtype=float)
    classes = np.array(["A", "B", None, "A", "C", np.nan], dtype=object)
    distance = sc.nearest_other_class_distance(xy, classes)
    # A at (0, 0): nearest non-A is C at (0, 2)
    assert distance[0] == pytest.approx(2.0)
    assert distance[1] == pytest.approx(math.hypot(3, 2))
    assert math.isnan(distance[2]) and math.isnan(distance[5])
    assert distance[3] == pytest.approx(1.0)
    assert distance[4] == pytest.approx(1.0)
    only = sc.nearest_other_class_distance(
        xy, classes, query=np.array([1, 0, 0, 0, 0, 0], bool)
    )
    assert only[0] == pytest.approx(2.0) and np.isnan(only[1:]).all()


def test_nearest_other_class_distance_needs_two_classes() -> None:
    assert np.isnan(
        sc.nearest_other_class_distance(
            np.zeros((2, 2)), np.array(["A", "A"], dtype=object)
        )
    ).all()


# ---------------------------------------------------------------------------
# Tile bootstrap


def test_tile_bootstrap_ratio_means_difference_and_pairing() -> None:
    values_a = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    values_b = np.array([2.0, 2.0, 3.0, 5.0, 9.0])
    groups_a = np.array([0, 0, 1, 1, 0])
    groups_b = np.array([0, 1, 1, 1, -1])
    tiles = np.array([0, 1, 0, 1, 0])
    result = sc.tile_bootstrap_ratio(
        values_a, groups_a, values_b, groups_b, tiles, 2, n_reps=50
    )
    np.testing.assert_allclose(result.mean_a, [1.5, 3.5])
    np.testing.assert_allclose(result.mean_b, [2.0, 10.0 / 3.0])
    np.testing.assert_array_equal(result.n_a, [2, 2])
    np.testing.assert_array_equal(result.n_b, [1, 3])
    np.testing.assert_allclose(result.difference, result.mean_b - result.mean_a)
    assert result.n_tiles == 2 and result.n_reps == 50
    assert np.all(result.ci_low <= result.ci_high)


def test_tile_bootstrap_ratio_identical_arms_have_zero_width() -> None:
    rng = np.random.default_rng(3)
    values = rng.random(60)
    groups = rng.integers(0, 3, 60)
    tiles = rng.integers(0, 6, 60)
    result = sc.tile_bootstrap_ratio(values, groups, values, groups, tiles, 3)
    np.testing.assert_allclose(result.difference, 0.0)
    np.testing.assert_allclose(result.ci_low, 0.0)
    np.testing.assert_allclose(result.ci_high, 0.0)


def test_tile_bootstrap_ratio_is_seeded() -> None:
    rng = np.random.default_rng(1)
    values_a, values_b = rng.random(80), rng.random(80)
    groups = rng.integers(0, 2, 80)
    tiles = rng.integers(0, 10, 80)
    first = sc.tile_bootstrap_ratio(values_a, groups, values_b, groups, tiles, 2)
    again = sc.tile_bootstrap_ratio(values_a, groups, values_b, groups, tiles, 2)
    other = sc.tile_bootstrap_ratio(
        values_a, groups, values_b, groups, tiles, 2, seed=5
    )
    np.testing.assert_array_equal(first.ci_low, again.ci_low)
    assert not np.array_equal(first.ci_low, other.ci_low)


# ---------------------------------------------------------------------------
# Tests with effect sizes


def test_benjamini_hochberg_known_values_and_nan() -> None:
    q = sc.benjamini_hochberg([0.01, 0.04, 0.03, np.nan, 0.5])
    np.testing.assert_allclose(q[[0, 1, 2, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    assert math.isnan(q[3])


def test_benjamini_hochberg_matches_statsmodels() -> None:
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    p = rng.random(50) ** 3
    np.testing.assert_allclose(
        sc.benjamini_hochberg(p), multitest.multipletests(p, method="fdr_bh")[1]
    )


def test_benjamini_hochberg_is_monotone_and_capped() -> None:
    p = np.array([0.9, 0.95, 0.99, 0.001])
    q = sc.benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= 0)
    assert q.max() <= 1.0
    assert np.isnan(sc.benjamini_hochberg([np.nan, np.nan])).all()


def test_spearman_bootstrap() -> None:
    rng = np.random.default_rng(0)
    x = rng.random(40)
    y = x + 0.3 * rng.random(40)
    x[3] = np.nan
    result = sc.spearman_bootstrap(x, y, n_reps=300)
    keep = np.isfinite(x)
    expected = stats.spearmanr(x[keep], y[keep])
    assert result.effect == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.n == 39
    assert result.ci_low <= result.effect <= result.ci_high
    again = sc.spearman_bootstrap(x, y, n_reps=300)
    assert (again.ci_low, again.ci_high) == (result.ci_low, result.ci_high)


def test_spearman_bootstrap_degenerate() -> None:
    result = sc.spearman_bootstrap([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    assert math.isnan(result.p_value) and result.note


def test_mann_whitney_effect_sign_and_value() -> None:
    result = sc.mann_whitney([5, 6, 7, 8], [1, 2, 3])
    assert result.statistic == 12.0
    assert result.effect == pytest.approx(1.0)
    assert result.median_difference == pytest.approx(4.5)
    reverse = sc.mann_whitney([1, 2, 3], [5, 6, 7, 8])
    assert reverse.effect == pytest.approx(-1.0)
    assert reverse.p_value == pytest.approx(result.p_value)
    small = sc.mann_whitney([1, 2], [3, 4, 5])
    assert math.isnan(small.p_value) and small.n == 2


def test_wilcoxon_paired_effect_and_zero_differences() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan])
    b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 1.0])
    result = sc.wilcoxon_paired(a, b)
    expected = stats.wilcoxon(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.effect == pytest.approx(1.0)
    assert result.n == 6
    assert result.median_difference == pytest.approx(2.5)
    mixed = sc.wilcoxon_paired([1.0, -2.0, 3.0, -4.0], [0.0, 0.0, 0.0, 0.0])
    # ranks of |d| = 1, 2, 3, 4: R+ = 1 + 3, R- = 2 + 4
    assert mixed.effect == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# Pseudobulk shares and the combined labelling


def test_pseudobulk_shares_formula() -> None:
    counts = np.array([[3, 1, 0], [1, 1, 2], [9, 9, 9]])
    shares, n_cells = sc.pseudobulk_shares(counts, ["A", "A", None], ["A", "B"])
    np.testing.assert_allclose(shares[0], (np.array([4, 2, 2]) + 1) / (8 + 3))
    np.testing.assert_allclose(shares[1], np.full(3, 1 / 3))
    np.testing.assert_array_equal(n_cells, [2, 0])
    included, _ = sc.pseudobulk_shares(
        counts, ["A", "A", "A"], ["A"], include=np.array([True, False, False])
    )
    np.testing.assert_allclose(included[0], (np.array([3, 1, 0]) + 1) / (4 + 3))


def test_reference_broad_shares_weights_nodes_by_cells() -> None:
    profiles = pd.DataFrame(
        {
            "level": ["SUPC"] * 6 + ["CLUS"],
            "node_name": ["n1", "n1", "n2", "n2", "sink", "sink", "n1"],
            "n_cells": [10, 10, 30, 30, 5, 5, 99],
            "gene_id": ["g1", "g2", "g1", "g2", "g1", "g2", "g1"],
            "mean_cpm": [100.0, 300.0, 200.0, 200.0, 1.0, 1.0, 5.0],
        }
    )
    shares = sc.reference_broad_shares(
        profiles,
        "SUPC",
        {"n1": "Astro", "n2": "Astro", "sink": "x"},
        ["g1", "g2", "g3"],
        ["Astro", "Oligo"],
    )
    # weighted mean cpm: g1 = (10*100 + 30*200) / 40 = 175, g2 = 225, g3 = 0
    np.testing.assert_allclose(
        shares.loc["Astro"].to_numpy(), [175 / 400, 225 / 400, 0.0]
    )
    assert shares.loc["Oligo"].isna().all()


def test_combined_labels_join_on_cell_id() -> None:
    reseg = pd.DataFrame(
        {
            "cell_id": [1, 2, 3, 4],
            "in_table": [True, True, False, True],
            "ct_broad_status": [
                "confident",
                "low_confidence",
                "low_counts",
                "confident",
            ],
            "ct_broad_name": ["Astrocytes", "Neurons", None, "Microglia"],
        }
    )
    combined = sc.combined_labels(
        ["4", "3", "9", "1"], reseg, ["ct_broad_status", "ct_broad_name"]
    )
    assert list(combined.index) == ["4", "3", "9", "1"]
    assert combined["has_reseg_table_cell"].tolist() == [True, False, False, True]
    assert combined["ct_broad_name"].tolist() == ["Microglia", None, None, "Astrocytes"]
    assert combined["ct_broad_status"].tolist() == [
        "confident",
        None,
        None,
        "confident",
    ]


def test_combined_labels_rejects_duplicates() -> None:
    reseg = pd.DataFrame({"cell_id": [1, 1], "in_table": [True, True], "x": [1, 2]})
    with pytest.raises(ValueError):
        sc.combined_labels(["1"], reseg, ["x"])
    ok = pd.DataFrame({"cell_id": [1], "in_table": [True], "x": [1]})
    with pytest.raises(ValueError):
        sc.combined_labels(["1", "1"], ok, ["x"])


# ---------------------------------------------------------------------------
# Gene covariates


def _gtf_line(chrom: str, start: int, stop: int, attributes: str) -> str:
    return f"{chrom}\tH\texon\t{start}\t{stop}\t.\t+\t.\t{attributes}"


def _ids(gene: str, transcript: str, name: str = "") -> str:
    out = f'gene_id "{gene}"; transcript_id "{transcript}";'
    return out + (f' gene_name "{name}";' if name else "")


GTF = [
    "##header",
    'chr1\tH\tgene\t1\t100\t.\t+\t.\tgene_id "ENSG1.3"; gene_name "AAA";',
    _gtf_line("chr1", 1, 10, _ids("ENSG1.3", "ENST1.1", "AAA")),
    _gtf_line("chr1", 21, 30, _ids("ENSG1.3", "ENST1.1", "AAA")),
    _gtf_line("chr1", 1, 15, _ids("ENSG1.3", "ENST2.1", "AAA")),
    _gtf_line("chr1_KI270706v1_random", 1, 900, _ids("ENSG1.3", "ENST3.1")),
    _gtf_line("chrX", 5, 5, _ids("ENSG2.1", "ENST4.1", "BBB")),
    _gtf_line("chr2", 1, 500, _ids("ENSG9.1", "ENST9.1", "ZZZ")),
]


def test_longest_transcript_lengths() -> None:
    frame = sc.longest_transcript_lengths(GTF, ["ENSG1", "ENSG2.7", "ENSG404"])
    assert frame["gene_id"].tolist() == ["ENSG1", "ENSG2"]
    assert frame["transcript_length"].tolist() == [20, 1]
    assert frame["transcript_id"].tolist() == ["ENST1", "ENST4"]
    assert frame["gene_name"].tolist() == ["AAA", "BBB"]


def test_xenium_probe_counts() -> None:
    panel = {
        "payload": {
            "targets": [
                {
                    "info": {"gene_coverage": 7},
                    "type": {
                        "data": {"id": "ENSG1", "name": "MAPT"},
                        "descriptor": "gene",
                    },
                },
                {
                    "info": {},
                    "type": {
                        "data": {"id": "ENSG2", "name": "X"},
                        "descriptor": "gene",
                    },
                },
                {
                    "info": {"gene_coverage": 1},
                    "type": {
                        "data": {"id": "NC1", "name": "NegControlProbe_1"},
                        "descriptor": "negative_control",
                    },
                },
            ]
        }
    }
    frame = sc.xenium_probe_counts(panel)
    assert frame["gene_name"].tolist() == ["MAPT", "X"]
    assert frame["probe_count"].iloc[0] == 7.0
    assert math.isnan(frame["probe_count"].iloc[1])


def test_loss_tally_unassigned_parts_need_reseg_unassigned() -> None:
    """Only reseg-unassigned in-nucleus transcripts enter the N_bg parts."""
    tally = sc.loss_tally(
        np.array([0, 0]),
        1,
        assigned_reseg=np.array([True, False]),
        assigned_hybrid=np.array([True, True]),
        assigned_reseg_same_cell=np.array([True, True]),
        in_nucleus=np.array([True, True]),
        background_reseg=np.array([True, False]),
    )
    frame = pd.DataFrame(tally[0], columns=list(sc.TALLY_FIELDS))
    assert frame.loc[0, "n_in_nucleus_unassigned_reseg"] == 1
    assert frame.loc[0, "n_in_nucleus_background_reseg"] == 0
    assert frame.loc[0, "n_in_nucleus_nontable_reseg"] == 1
    # the same-cell share counts reseg-assigned transcripts only
    assert frame.loc[0, "n_assigned_reseg_same_cell"] == 1
