"""Tests for the subsample-stability diagnostic (plan §6.3)."""

from __future__ import annotations

import itertools
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from merxen.clustering.stability import (
    SubsampleStability,
    adjusted_rand_index,
    cluster_pair_consistency,
    contingency,
    subsample_ari,
    subsample_indices,
)


def _pair_counting_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Return the ARI by enumerating all O(N^2) object pairs (the definition)."""
    n = len(labels_a)
    together_both = only_a = only_b = neither = 0
    for i, j in itertools.combinations(range(n), 2):
        same_a = labels_a[i] == labels_a[j]
        same_b = labels_b[i] == labels_b[j]
        if same_a and same_b:
            together_both += 1
        elif same_a:
            only_a += 1
        elif same_b:
            only_b += 1
        else:
            neither += 1
    if only_a == 0 and only_b == 0:
        return 1.0
    numerator = 2 * (together_both * neither - only_a * only_b)
    denominator = (together_both + only_a) * (only_a + neither) + (
        together_both + only_b
    ) * (only_b + neither)
    return numerator / denominator


@pytest.mark.parametrize("seed", range(6))
def test_contingency_ari_equals_pair_counting_and_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 120))
    labels_a = rng.integers(0, int(rng.integers(1, 7)), size=n).astype(str)
    labels_b = rng.integers(0, int(rng.integers(1, 9)), size=n)
    expected = _pair_counting_ari(labels_a, labels_b)
    assert adjusted_rand_index(labels_a, labels_b) == pytest.approx(expected, abs=1e-12)
    assert adjusted_rand_index(labels_a, labels_b) == pytest.approx(
        adjusted_rand_score(labels_a, labels_b), abs=1e-12
    )


@pytest.mark.parametrize(
    ("labels_a", "labels_b"),
    [
        (["a"] * 5, ["x"] * 5),
        (list("abcde"), list("vwxyz")),
        (["a", "a", "b", "b"], ["b", "b", "a", "a"]),
        ([], []),
        (["a", "b", "a", "b"], ["x", "x", "x", "x"]),
        (["a", "a", "b", "c", "c", "c"], ["x", "y", "y", "z", "z", "x"]),
    ],
)
def test_ari_edge_cases_match_sklearn(labels_a: list[str], labels_b: list[str]) -> None:
    assert adjusted_rand_index(labels_a, labels_b) == pytest.approx(
        adjusted_rand_score(labels_a, labels_b), abs=1e-12
    )


def test_ari_is_exact_for_large_partitions() -> None:
    # Integer pair counts overflow int64 around 10^5 objects; the exact
    # arithmetic must still match sklearn.
    rng = np.random.default_rng(3)
    labels_a = rng.integers(0, 12, size=300_000)
    labels_b = np.where(
        rng.random(300_000) < 0.9, labels_a, rng.integers(0, 12, 300_000)
    )
    assert adjusted_rand_index(labels_a, labels_b) == pytest.approx(
        adjusted_rand_score(labels_a, labels_b), rel=1e-9
    )


def test_ari_is_exact_for_groups_whose_squares_overflow_int32() -> None:
    # Two groups of ~150k objects: squared group sizes exceed int32.
    rng = np.random.default_rng(4)
    labels_a = rng.integers(0, 2, size=300_000)
    labels_b = np.where(rng.random(300_000) < 0.8, labels_a, 1 - labels_a)
    assert adjusted_rand_index(labels_a, labels_b) == pytest.approx(
        adjusted_rand_score(labels_a, labels_b), rel=1e-9
    )


def test_missing_labels_form_one_group() -> None:
    labels_a = np.array(["a", None, None, "b"], dtype=object)
    labels_b = np.array(["x", "y", "y", "x"], dtype=object)
    table = contingency(labels_a, labels_b)
    assert table.n_objects == 4
    assert sorted(table.row_sums.tolist()) == [1, 1, 2]
    assert sorted(table.counts.tolist()) == [1, 1, 2]
    assert adjusted_rand_index(labels_a, labels_b) == pytest.approx(
        adjusted_rand_score(["a", "n", "n", "b"], ["x", "y", "y", "x"])
    )


def test_contingency_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        contingency(["a", "b"], ["x"])


def test_cluster_pair_consistency_counts_ordered_pairs() -> None:
    kept, total = cluster_pair_consistency(
        ["a", "a", "a", "b", "b"], ["1", "1", "2", "3", "3"]
    )
    # Group a: 3 cells -> 6 ordered pairs, 2 kept (the two cells in "1").
    assert kept["a"] == 2
    assert total["a"] == 6
    assert kept["b"] == 2
    assert total["b"] == 2


def test_subsample_indices_are_deterministic_sorted_and_unique() -> None:
    draws = subsample_indices(100, n_subsamples=5, fraction=0.8, seed=0)
    again = subsample_indices(100, n_subsamples=5, fraction=0.8, seed=0)
    other = subsample_indices(100, n_subsamples=5, fraction=0.8, seed=1)
    assert len(draws) == 5
    for draw, repeat in zip(draws, again, strict=True):
        assert draw.size == 80
        assert np.array_equal(draw, repeat)
        assert np.all(np.diff(draw) > 0)
    assert not all(np.array_equal(a, b) for a, b in zip(draws, other, strict=True))
    assert len({tuple(draw) for draw in draws}) == 5
    with pytest.raises(ValueError, match="fraction"):
        subsample_indices(10, n_subsamples=1, fraction=0.0, seed=0)
    with pytest.raises(ValueError, match="n_subsamples"):
        subsample_indices(10, n_subsamples=-1, fraction=0.5, seed=0)


def _separated_embedding(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0], [12.0, 12.0]])
    points = np.vstack([centre + rng.normal(0, 0.6, (60, 2)) for centre in centres])
    groups = np.repeat(np.arange(4), 60).astype(str)
    adata = ad.AnnData(
        obs=pd.DataFrame(
            {"leiden": pd.Categorical(groups)},
            index=pd.Index([f"c{index}" for index in range(len(groups))]),
        )
    )
    adata.obsm["X_pca"] = np.hstack([points, rng.normal(0, 0.05, (len(groups), 3))])
    return adata


def test_subsample_ari_is_high_on_separated_groups_and_reproducible() -> None:
    adata = _separated_embedding()
    first = subsample_ari(adata, "leiden", resolution=0.3, n_neighbors=15, seed=0)
    second = subsample_ari(adata, "leiden", resolution=0.3, n_neighbors=15, seed=0)
    assert isinstance(first, SubsampleStability)
    assert first.n_subsamples == 5
    assert first.n_objects == 240
    assert first.n_objects_per_subsample == 192
    assert len(first.aris) == 5
    assert first.aris == second.aris
    assert first.mean_ari > 0.9
    assert first.min_ari <= first.mean_ari
    assert first.n_clusters_reference == 4
    assert set(first.cluster_pair_consistency) == {"0", "1", "2", "3"}
    assert all(0.0 <= v <= 1.0 for v in first.cluster_pair_consistency.values())
    assert first.role == "report_only"
    payload = json.loads(first.to_json())
    assert payload["mean_ari"] == pytest.approx(first.mean_ari)
    assert payload["aris"] == list(first.aris)


def test_subsample_ari_without_subsamples_reports_nothing() -> None:
    adata = _separated_embedding()
    record = subsample_ari(adata, "leiden", n_subsamples=0)
    assert record.aris == ()
    assert np.isnan(record.mean_ari)
    payload = json.loads(record.to_json())
    assert payload["mean_ari"] is None
    assert payload["min_ari"] is None


def test_subsample_ari_requires_partition_embedding_and_cells() -> None:
    adata = _separated_embedding()
    with pytest.raises(KeyError, match="partition"):
        subsample_ari(adata, "missing")
    with pytest.raises(KeyError, match="embedding"):
        subsample_ari(adata, "leiden", use_rep="X_missing")
    tiny = adata[:2].copy()
    with pytest.raises(ValueError, match="at least 3"):
        subsample_ari(tiny, "leiden")
