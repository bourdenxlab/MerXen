"""Tests for ``merxen.annotation.report_metrics`` (plan §9; M7)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.spatial.distance import jensenshannon

from merxen.annotation import report_metrics as rm


def _grid_xy(n_side: int, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Points jittered on a square grid (``n_side``² points)."""
    grid = np.array([(i, j) for i in range(n_side) for j in range(n_side)], dtype=float)
    return grid * spacing + rng.uniform(0, spacing * 0.5, size=grid.shape)


# --------------------------------------------------------------------------
# Composition


def test_soft_level_matrix_adds_runner_ups_and_sends_the_rest_to_unallocated() -> None:
    matrix = rm.soft_level_matrix(
        ["A", "B", None, "Z"],
        [0.6, 0.9, np.nan, 0.7],
        [["B", "A", "A", "A"]],
        [[0.3, 0.2, 0.5, 0.2]],
        categories=["A", "B"],
    )
    np.testing.assert_allclose(matrix[0], [0.6, 0.3, 0.1])
    # 0.9 + 0.2 > 1: rescaled to 1, nothing unallocated.
    np.testing.assert_allclose(matrix[1], [0.2 / 1.1, 0.9 / 1.1, 0.0])
    np.testing.assert_allclose(matrix[2], [0.5, 0.0, 0.5])
    # "Z" is outside the categories: its mass is unallocated.
    np.testing.assert_allclose(matrix[3], [0.2, 0.0, 0.8])
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)


def test_soft_level_matrix_maps_nodes_to_categories() -> None:
    matrix = rm.soft_level_matrix(
        ["x1", "x2"],
        [0.5, 0.4],
        [["y1", "x1"]],
        [[0.5, 0.4]],
        categories=["X", "Y"],
        mapping={"x1": "X", "x2": "X", "y1": "Y"},
    )
    np.testing.assert_allclose(matrix, [[0.5, 0.5, 0.0], [0.8, 0.0, 0.2]])


def test_soft_level_matrix_refuses_ragged_ranks() -> None:
    with pytest.raises(ValueError):
        rm.soft_level_matrix(["A"], [0.5], [["A"]], [], categories=["A"])
    with pytest.raises(ValueError):
        rm.soft_level_matrix(["A"], [0.5], [["A", "B"]], [[0.1, 0.1]], categories=["A"])


def test_one_hot_matrix_puts_unknown_labels_last_and_zeroes_excluded_cells() -> None:
    matrix = rm.one_hot_matrix(
        ["A", "B", "Q", "A"], ["A", "B"], include=np.array([1, 1, 1, 0], bool)
    )
    np.testing.assert_array_equal(matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])


def test_block_bootstrap_shares_point_values_and_renormalisation() -> None:
    matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]
    )
    result = rm.block_bootstrap_shares(matrix, ["A", "B", "unallocated"], None)
    assert result.share("A") == pytest.approx(1.5 / 4)
    assert result.share("unallocated") == pytest.approx(1.5 / 4)
    frame = result.frame().set_index("category")
    assert frame.loc["A", "share_renormalised"] == pytest.approx(1.5 / 2.5)
    assert frame.loc["B", "share_renormalised"] == pytest.approx(1.0 / 2.5)
    assert math.isnan(frame.loc["unallocated", "share_renormalised"])
    assert math.isnan(frame.loc["A", "ci_low"])
    assert result.n_reps == 0 and result.n_tiles == 0


def test_block_bootstrap_shares_ci_covers_the_point_and_is_seeded() -> None:
    rng = np.random.default_rng(3)
    xy = _grid_xy(40, 50.0, rng)  # 2 x 2 km: 16 tiles of 500 µm
    labels = rng.choice(["A", "B"], size=len(xy), p=[0.3, 0.7])
    matrix = rm.one_hot_matrix(labels, ["A", "B"])
    codes = rm.grid_codes(xy, 500.0)
    first = rm.block_bootstrap_shares(
        matrix, ["A", "B", "u"], codes, n_reps=100, seed=1
    )
    second = rm.block_bootstrap_shares(
        matrix, ["A", "B", "u"], codes, n_reps=100, seed=1
    )
    other = rm.block_bootstrap_shares(
        matrix, ["A", "B", "u"], codes, n_reps=100, seed=2
    )
    record = first.records[0]
    assert record.ci_low <= record.share <= record.ci_high
    assert record.ci_high - record.ci_low > 0
    pd.testing.assert_frame_equal(first.frame(), second.frame())
    assert not first.frame().equals(other.frame())
    assert first.n_tiles == 16 and first.n_reps == 100


def test_block_bootstrap_shares_keep_restricts_cells() -> None:
    matrix = rm.one_hot_matrix(["A", "A", "B", "B"], ["A", "B"])
    result = rm.block_bootstrap_shares(
        matrix, ["A", "B", "u"], None, keep=np.array([True, True, True, False])
    )
    assert result.share("A") == pytest.approx(2 / 3)
    assert result.n_cells == 3


def test_jensen_shannon_distance_matches_scipy() -> None:
    p = np.array([0.2, 0.5, 0.3, 0.0])
    q = np.array([0.1, 0.1, 0.4, 0.4])
    assert rm.jensen_shannon_distance(p, q) == pytest.approx(
        jensenshannon(p, q, base=2)
    )
    assert rm.jensen_shannon_distance(p * 7, q * 3) == pytest.approx(
        jensenshannon(p, q, base=2)
    )
    assert math.isnan(rm.jensen_shannon_distance(np.zeros(3), q[:3]))
    assert rm.jensen_shannon_distance(p, p) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Aligned-bin concordance


def _two_sections(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two co-registered sections; type A rises with x on both, B is flipped."""
    xy_a = rng.uniform(0, 2000, size=(6000, 2))
    xy_b = rng.uniform(0, 2000, size=(5000, 2))
    frac_a = xy_a[:, 0] / 2000.0
    frac_b = xy_b[:, 0] / 2000.0
    labels_a = np.where(rng.uniform(size=len(xy_a)) < frac_a, "A", "C")
    labels_b = np.where(rng.uniform(size=len(xy_b)) < frac_b, "A", "C")
    # B rises with x on the first section and falls on the second.
    rise = rng.uniform(size=len(xy_a)) < frac_a
    fall = rng.uniform(size=len(xy_b)) < 1 - frac_b
    labels_a = np.where((labels_a == "C") & rise, "B", labels_a)
    labels_b = np.where((labels_b == "C") & fall, "B", labels_b)
    return xy_a, labels_a, xy_b, labels_b


def test_aligned_bin_density_correlation_separates_matched_and_flipped_types() -> None:
    rng = np.random.default_rng(0)
    xy_a, labels_a, xy_b, labels_b = _two_sections(rng)
    frame = rm.aligned_bin_density_correlation(
        xy_a,
        rm.one_hot_matrix(labels_a, ["A", "B"])[:, :2],
        xy_b,
        rm.one_hot_matrix(labels_b, ["A", "B"])[:, :2],
        ["A", "B"],
        bin_um=200.0,
    ).set_index("category")
    assert frame.loc["A", "n_bins"] == 100
    assert frame.loc["A", "pearson_r"] > 0.8
    assert frame.loc["B", "pearson_r"] < 0.0
    assert frame.loc["A", "density_a_per_mm2"] == pytest.approx(
        np.count_nonzero(labels_a == "A") / 4.0, rel=1e-6
    )


def test_aligned_bin_density_correlation_compares_only_shared_bins() -> None:
    rng = np.random.default_rng(1)
    xy_a = rng.uniform(0, 1000, size=(3000, 2))
    xy_b = rng.uniform(0, 1000, size=(3000, 2))
    xy_b[:, 0] += 500.0  # only x in [500, 1000) overlaps
    ones_a = np.ones((len(xy_a), 1))
    ones_b = np.ones((len(xy_b), 1))
    frame = rm.aligned_bin_density_correlation(
        xy_a, ones_a, xy_b, ones_b, ["all"], bin_um=250.0
    )
    assert int(frame.loc[0, "n_bins"]) == 8  # 2 of 4 columns x 4 rows


def test_aligned_bin_density_correlation_checks_shapes() -> None:
    with pytest.raises(ValueError):
        rm.aligned_bin_density_correlation(
            np.zeros((3, 2)),
            np.zeros((3, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 1)),
            ["a", "b"],
        )


# --------------------------------------------------------------------------
# Cortical depth: medians, ordering, replication, contrasts, gradients


def _depth_groups(
    rng: np.random.Generator,
    centres: dict[str, float],
    n: int = 400,
    spread: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.repeat(list(centres), n)
    depth = np.concatenate(
        [rng.normal(centre, spread, n) for centre in centres.values()]
    )
    xy = rng.uniform(0, 3000, size=(len(depth), 2))
    return groups, np.clip(depth, 0, 1), rm.grid_codes(xy, 500.0)


def _medians(
    groups: np.ndarray, depth: np.ndarray, codes: np.ndarray
) -> dict[str, rm.MedianCi]:
    return {
        name: rm.median_block_ci(
            depth[groups == name], codes[groups == name], n_reps=100, seed=0
        )
        for name in dict.fromkeys(groups)
    }


def test_median_block_ci_matches_numpy_and_brackets_the_median() -> None:
    rng = np.random.default_rng(2)
    values = rng.uniform(size=500)
    codes = rm.grid_codes(rng.uniform(0, 2000, size=(500, 2)), 500.0)
    result = rm.median_block_ci(values, codes, n_reps=150, seed=0)
    assert result.median == pytest.approx(np.median(values))
    assert result.ci_low <= result.median <= result.ci_high
    assert result.ci_high > result.ci_low
    assert result.n_cells == 500 and result.n_tiles == 16
    again = rm.median_block_ci(values, codes, n_reps=150, seed=0)
    assert again == result
    no_ci = rm.median_block_ci(np.r_[values, np.nan], None)
    assert math.isnan(no_ci.ci_low) and no_ci.n_cells == 500


def test_depth_ordering_passes_on_separated_layers() -> None:
    rng = np.random.default_rng(4)
    groups, depth, codes = _depth_groups(rng, {"U": 0.25, "D": 0.5, "N": 0.75})
    result = rm.depth_ordering(_medians(groups, depth, codes), ["U", "D", "N"])
    assert result.ordered and result.separated and result.passes
    assert result.missing == ()


def test_depth_ordering_fails_when_the_order_is_swapped() -> None:
    rng = np.random.default_rng(4)
    groups, depth, codes = _depth_groups(rng, {"U": 0.5, "D": 0.25, "N": 0.75})
    result = rm.depth_ordering(_medians(groups, depth, codes), ["U", "D", "N"])
    assert not result.ordered and not result.passes


def test_depth_ordering_fails_when_the_cis_overlap() -> None:
    rng = np.random.default_rng(5)
    groups, depth, codes = _depth_groups(
        rng, {"U": 0.3, "D": 0.305, "N": 0.8}, n=60, spread=0.2
    )
    medians = _medians(groups, depth, codes)
    medians["U"] = rm.MedianCi(0.30, 0.25, 0.36, 60, 10)
    medians["D"] = rm.MedianCi(0.31, 0.27, 0.35, 60, 10)
    result = rm.depth_ordering(medians, ["U", "D", "N"])
    assert result.ordered and not result.separated and not result.passes


def test_depth_ordering_fails_when_a_group_is_missing() -> None:
    medians = {
        "U": rm.MedianCi(0.2, 0.1, 0.3, 50, 5),
        "N": rm.MedianCi(0.8, 0.7, 0.9, 50, 5),
    }
    result = rm.depth_ordering(medians, ["U", "D", "N"])
    assert result.missing == ("D",) and not result.passes
    few = rm.depth_ordering(
        {**medians, "D": rm.MedianCi(0.5, 0.4, 0.6, 3, 1)},
        ["U", "D", "N"],
        min_cells=20,
    )
    assert few.missing == ("D",) and not few.passes


def test_depth_replication_needs_both_platforms_to_pass() -> None:
    rng = np.random.default_rng(6)
    groups, depth, codes = _depth_groups(
        rng, {"U": 0.25, "D": 0.5, "N": 0.75, "O": 0.6}
    )
    good = _medians(groups, depth, codes)
    swapped_groups, swapped_depth, swapped_codes = _depth_groups(
        rng, {"U": 0.5, "D": 0.25, "N": 0.75, "O": 0.6}
    )
    bad = _medians(swapped_groups, swapped_depth, swapped_codes)
    order = ["U", "D", "N"]
    both = rm.depth_replication(
        {"MERSCOPE": good, "XENIUM": good},
        {
            "MERSCOPE": rm.depth_ordering(good, order),
            "XENIUM": rm.depth_ordering(good, order),
        },
    )
    assert both.replicated and both.n_groups == 4
    assert both.spearman_r == pytest.approx(1.0)
    assert both.max_abs_difference == pytest.approx(0.0)
    one = rm.depth_replication(
        {"MERSCOPE": good, "XENIUM": bad},
        {
            "MERSCOPE": rm.depth_ordering(good, order),
            "XENIUM": rm.depth_ordering(bad, order),
        },
    )
    assert not one.replicated
    assert one.passes_per_platform == (True, False)
    assert one.spearman_r < 1.0
    single = rm.depth_replication(
        {"MERSCOPE": good}, {"MERSCOPE": rm.depth_ordering(good, order)}
    )
    assert not single.replicated and math.isnan(single.spearman_r)


def test_share_contrast_detects_white_matter_enrichment() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    white = rng.uniform(size=n) < 0.4
    is_oligo = np.where(white, rng.uniform(size=n) < 0.7, rng.uniform(size=n) < 0.2)
    codes = rm.grid_codes(rng.uniform(0, 3000, size=(n, 2)), 500.0)
    result = rm.share_contrast(is_oligo, white, ~white, codes, n_reps=100, seed=0)
    assert result.share_a == pytest.approx(is_oligo[white].mean())
    assert result.share_b == pytest.approx(is_oligo[~white].mean())
    assert result.greater and result.ci_excludes_zero
    assert result.ci_low <= result.difference <= result.ci_high
    flat = rm.share_contrast(
        rng.uniform(size=n) < 0.3, white, ~white, codes, n_reps=100, seed=0
    )
    assert flat.ci_low < 0 < flat.ci_high and not flat.ci_excludes_zero


def test_depth_profile_and_gradient_find_a_deepening_class() -> None:
    rng = np.random.default_rng(8)
    depth = rng.uniform(size=5000)
    labels = np.where(rng.uniform(size=5000) < depth, "Oligo", "Neuron")
    profile = rm.depth_profile(depth, labels, ["Oligo", "Neuron"], n_bins=5)
    assert set(profile["depth_bin"]) == set(range(5))
    oligo = profile[profile["category"] == "Oligo"].sort_values("depth_bin")
    assert oligo["share"].is_monotonic_increasing
    gradient = rm.profile_gradient(profile).set_index("category")
    assert gradient.loc["Oligo", "spearman_r"] == pytest.approx(1.0)
    assert gradient.loc["Neuron", "spearman_r"] == pytest.approx(-1.0)
    assert (
        gradient.loc["Oligo", "share_deep"] > gradient.loc["Oligo", "share_superficial"]
    )


# --------------------------------------------------------------------------
# Platform factors, pseudobulks


def test_platform_gene_log2_ratios_are_median_centred() -> None:
    mean_a = np.array([1.0, 2.0, 4.0, 0.0])
    mean_b = np.array([2.0, 4.0, 8.0, 0.0])
    ratios, median = rm.platform_gene_log2_ratios(mean_a, mean_b, pseudocount=1e-9)
    assert median == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(ratios[:3], 0.0, atol=1e-6)
    assert ratios[3] == pytest.approx(-1.0, abs=1e-6)
    raw, raw_median = rm.platform_gene_log2_ratios(
        mean_a, mean_b, pseudocount=1e-9, centre=False
    )
    np.testing.assert_allclose(raw[:3], 1.0, atol=1e-6)
    assert raw_median == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        rm.platform_gene_log2_ratios(mean_a, mean_b[:2])
    with pytest.raises(ValueError):
        rm.platform_gene_log2_ratios(mean_a, mean_b, pseudocount=0.0)


def test_group_mean_counts_on_dense_and_sparse_input() -> None:
    matrix = np.array([[1, 0], [3, 2], [5, 4], [9, 9]], dtype=float)
    groups = np.array(["a", "a", "b", "other"])
    for data in (matrix, sparse.csr_matrix(matrix)):
        means, n_cells = rm.group_mean_counts(data, groups, ["a", "b", "missing"])
        np.testing.assert_allclose(means[:2], [[2.0, 1.0], [5.0, 4.0]])
        assert np.isnan(means[2]).all()
        np.testing.assert_array_equal(n_cells, [2, 1, 0])


def test_log_cpm_and_nearest_centroid() -> None:
    centroids = np.array([[100.0, 1.0, 1.0], [1.0, 100.0, 1.0], [1.0, 1.0, 100.0]])
    profile = rm.log_cpm(np.array([50.0, 2.0, 1.0]))
    name, r = rm.nearest_centroid(profile, rm.log_cpm(centroids), ["x", "y", "z"])
    assert name == "x" and r > 0.9
    assert rm.log_cpm(np.array([1.0, 1.0]))[0] == pytest.approx(np.log2(5e5 + 1))
    empty = rm.nearest_centroid(
        np.array([1.0, 1.0, 1.0]), rm.log_cpm(centroids), ["x", "y", "z"]
    )
    assert empty == (None, empty[1]) and math.isnan(empty[1])


# --------------------------------------------------------------------------
# Agreement, histograms, maps, self-thinning


def test_agreement_by_quantile_uses_the_h3_label_rule() -> None:
    first = np.array(["A", "A", "B", "X", "A", "B", "Y", "A"])
    second = np.array(["A", "B", "B", "Z", "A", "A", "Y", "A"])
    counts = np.arange(8, dtype=float) * 10
    frame = rm.agreement_by_quantile(
        first, second, counts, classes=["A", "B"], unassigned="U", n_quantiles=2
    )
    overall = frame[frame["quantile"] == 0].iloc[0]
    # X/Z and Y/Y are out-of-class: both agree unless strict.
    assert overall["agreement"] == pytest.approx(6 / 8)
    strict = rm.agreement_by_quantile(
        first, second, counts, classes=["A", "B"], unassigned="U", strict=True
    )
    assert strict.iloc[0]["agreement"] == pytest.approx(4 / 8)
    halves = frame[frame["quantile"] > 0]
    assert halves["n_cells"].sum() == 8
    assert list(halves["agreement"]) == pytest.approx([3 / 4, 3 / 4])
    masked = rm.agreement_by_quantile(
        first, second, counts, classes=["A", "B"], unassigned="U", mask=counts >= 20
    )
    assert masked.iloc[0]["n_cells"] == 6


def test_histogram_2d_counts_every_finite_point_once() -> None:
    x = np.array([0.1, 0.2, 0.9, np.nan])
    y = np.array([0.1, 0.8, 0.9, 0.5])
    frame = rm.histogram_2d(x, y, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    assert frame["n"].sum() == 3
    assert frame[(frame.x_low == 0.0) & (frame.y_low == 0.5)]["n"].item() == 1


def test_tile_mean_map_averages_per_tile() -> None:
    xy = np.array([[10.0, 10.0], [20.0, 30.0], [600.0, 10.0], [np.nan, 1.0]])
    frame = rm.tile_mean_map(xy, np.array([1.0, 0.0, 1.0, 1.0]), tile_um=500.0)
    assert list(frame["n_cells"]) == [2, 1]
    assert list(frame["value"]) == pytest.approx([0.5, 1.0])
    assert frame.iloc[1]["x_um"] == pytest.approx(750.0)
    assert rm.tile_mean_map(np.zeros((0, 2)), np.zeros(0)).empty


def test_self_thinning_is_unreliable_when_one_class_dominates() -> None:
    # 509 deep cells, 89% vascular (the plan's example, §9 item 1).
    counts = np.r_[np.full(509, 250.0), np.full(2000, 50.0)]
    labels = np.r_[
        np.repeat("Vascular cells", 453),
        np.repeat("Neurons", 56),
        np.repeat("Neurons", 2000),
    ]
    result = rm.self_thinning_eligibility(counts, labels)
    assert result.n_deep == 509 and result.eligible
    assert result.dominant_label == "Vascular cells"
    assert result.dominant_share == pytest.approx(453 / 509)
    assert result.evaluable_classes == ("Neurons", "Vascular cells")
    assert not result.reliable and result.reasons[0].startswith("dominant:")
    balanced = rm.self_thinning_eligibility(
        counts, np.r_[np.tile(["A", "B", "C"], 836), ["A"]]
    )
    assert balanced.eligible and balanced.reliable and not balanced.reasons
    few = rm.self_thinning_eligibility(np.full(100, 300.0), np.repeat("A", 100))
    assert not few.eligible and not few.reliable


def test_self_thinning_judges_labelled_cells_and_counts_unlabelled_apart() -> None:
    """Regression (M7 review): P1212_MERSCOPE, 48% unlabelled, 43% Fibroblasts.

    The report used to pass "Mixed/Unknown" as a class: it became the
    dominant label at 48% (below the old 50% cut) and P1212_M was marked
    reliable, while ordinary cortex (Xenium, 53-62% Neurons) was not.
    """
    n_deep = 509
    counts = np.full(n_deep, 250.0)
    labels = np.r_[
        np.repeat("Fibroblasts", 219),  # 43% of the deep cells
        np.repeat("Vascular cells", 25),
        np.repeat("Astrocytes", 20),
        np.repeat("", 245),  # 48% without a confident broad label
    ]
    labelled = labels != ""
    argmax = np.r_[np.repeat("Fibroblasts", 401), np.repeat("Vascular cells", 108)]
    p1212 = rm.self_thinning_eligibility(counts, labels, labelled, argmax=argmax)
    assert p1212.eligible and not p1212.reliable
    assert p1212.n_labelled == 264
    assert p1212.unlabelled_share == pytest.approx(245 / 509)
    assert "" not in p1212.composition
    assert p1212.dominant_label == "Fibroblasts"
    assert p1212.dominant_share == pytest.approx(219 / 264)
    assert any(reason.startswith("unlabelled_share:") for reason in p1212.reasons)
    assert any(reason.startswith("dominant:Fibroblasts") for reason in p1212.reasons)
    assert p1212.evaluable_classes == ("Fibroblasts",)
    assert p1212.argmax_composition["Fibroblasts"] == pytest.approx(401 / 509)
    # Ordinary cortex: 55 / 29 / 16 neuron / astrocyte / other, all labelled.
    cortex = np.r_[
        np.repeat("Neurons", 550),
        np.repeat("Astrocytes", 290),
        np.repeat("Oligodendrocytes", 160),
    ]
    xenium = rm.self_thinning_eligibility(np.full(1000, 300.0), cortex)
    assert xenium.reliable and xenium.unlabelled_share == 0.0
    assert xenium.evaluable_classes == ("Astrocytes", "Neurons", "Oligodendrocytes")
    # A few unlabelled cells do not matter; more than 30% do.
    some = rm.self_thinning_eligibility(
        np.full(1000, 300.0), cortex, np.arange(1000) >= 290
    )
    assert some.unlabelled_share == pytest.approx(0.29) and some.reliable
    many = rm.self_thinning_eligibility(
        np.full(1000, 300.0), cortex, np.arange(1000) >= 310
    )
    assert not many.reliable


def test_grid_codes_rejects_bad_input_and_marks_non_finite_points() -> None:
    codes = rm.grid_codes(np.array([[0.0, 0.0], [np.nan, 1.0], [600.0, 0.0]]), 500.0)
    assert codes[1] == -1 and codes[0] != codes[2]
    with pytest.raises(ValueError):
        rm.grid_codes(np.zeros((2, 3)), 500.0)
    with pytest.raises(ValueError):
        rm.grid_codes(np.zeros((2, 2)), 0.0)


def test_pearson_and_spearman_need_variance() -> None:
    assert math.isnan(rm.pearson_r(np.ones(5), np.arange(5.0)))
    assert math.isnan(rm.spearman_r(np.arange(2.0), np.arange(2.0)))
    assert rm.spearman_r(np.arange(5.0), np.arange(5.0) ** 3) == pytest.approx(1.0)


def test_tangential_block_codes_floor_positions_and_mark_missing() -> None:
    codes = rm.tangential_block_codes(np.array([10.0, 499.0, 500.0, np.nan, 1600.0]))
    np.testing.assert_array_equal(codes, [0, 0, 1, -1, 3])
    shifted = rm.tangential_block_codes(np.array([-600.0, -1.0, 0.0]))
    np.testing.assert_array_equal(shifted, [0, 1, 2])
    with pytest.raises(ValueError, match="positive"):
        rm.tangential_block_codes(np.zeros(2), 0.0)


def test_depth_input_agreement_catches_a_mirrored_ribbon() -> None:
    rng = np.random.default_rng(4)
    xy_a = rng.uniform(0, 2000, (6000, 2))
    xy_b = rng.uniform(0, 2000, (6000, 2))
    # A ribbon in the lower 60% of the section, depth growing with y.
    inside_a = xy_a[:, 1] < 1200
    depth_a = np.where(inside_a, xy_a[:, 1] / 1200, np.nan)
    inside_b = xy_b[:, 1] < 1200
    depth_b = np.where(inside_b, xy_b[:, 1] / 1200, np.nan)
    same = rm.depth_input_agreement(xy_a, inside_a, depth_a, xy_b, inside_b, depth_b)
    assert same.n_bins == 100 and same.ribbon_agreement == pytest.approx(1.0)
    assert same.depth_r > 0.99 and same.valid() is True
    # The second section's boundaries mirrored top to bottom.
    flipped_y = 2000 - xy_b[:, 1]
    inside_m = flipped_y < 1200
    depth_m = np.where(inside_m, flipped_y / 1200, np.nan)
    mirrored = rm.depth_input_agreement(
        xy_a, inside_a, depth_a, xy_b, inside_m, depth_m
    )
    assert mirrored.ribbon_agreement == pytest.approx(0.2, abs=0.05)
    assert mirrored.depth_r < 0 and mirrored.valid() is False
    # Either value alone below its minimum fails; NaN cannot be judged.
    assert rm.DepthAgreement(10, 0.95, 10, 0.5, 200.0, 5).valid() is False
    assert rm.DepthAgreement(10, 0.7, 10, 0.95, 200.0, 5).valid() is False
    assert rm.DepthAgreement(10, 0.8, 10, 0.8, 200.0, 5).valid() is True
    assert rm.DepthAgreement(0, math.nan, 0, math.nan, 200.0, 5).valid() is None
    # Only bins with >= 5 cells of each section count.
    few_cells = rm.depth_input_agreement(
        xy_a[:3], inside_a[:3], depth_a[:3], xy_b, inside_b, depth_b
    )
    assert few_cells.n_bins == 0 and few_cells.valid() is None
    with pytest.raises(ValueError, match="one value per cell"):
        rm.depth_input_agreement(xy_a, inside_a[:5], depth_a, xy_b, inside_b, depth_b)
