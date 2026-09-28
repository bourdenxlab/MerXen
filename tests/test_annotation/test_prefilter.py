"""Tests for the per-parent large-panel marker prefilter (plan §8.7)."""

from __future__ import annotations

import numpy as np
import pytest

from merxen.annotation.prefilter import (
    PREFILTER_METHOD,
    PREFILTER_VERSION,
    ParentOrder,
    PrefilterSettings,
    SiblingSet,
    child_profiles,
    largest_k,
    pair_scores,
    parent_gene_order,
    per_parent_topk_union,
    prefilter_payload,
    union_of_top_k,
)


def one_hot_reference(
    n_children: int, n_noise: int, *, seed: int = 0
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Leaves = children; gene i marks child i; noise genes vary a little."""
    rng = np.random.default_rng(seed)
    n_genes = n_children + n_noise
    n_cells = np.full(n_children, 50.0)
    mean = rng.uniform(0.0, 0.3, size=(n_children, n_genes))
    detection = rng.uniform(0.0, 0.1, size=(n_children, n_genes))
    for child in range(n_children):
        mean[child, child] = 4.0
        detection[child, child] = 0.9
    genes = [f"G{index:04d}" for index in range(n_genes)]
    return genes, n_cells, mean * n_cells[:, None], detection * n_cells[:, None]


def leaf_sets(n_children: int) -> list[SiblingSet]:
    return [
        SiblingSet(
            key="None",
            children=tuple(f"c{index}" for index in range(n_children)),
            leaf_rows=tuple((index,) for index in range(n_children)),
        )
    ]


def test_pair_scores_follow_ctm_marker_thresholds() -> None:
    means = np.array([[3.0, 0.5, 2.0], [0.0, 0.0, 0.9]])
    detection = np.array([[0.9, 0.9, 0.9], [0.05, 0.85, 0.8]])
    score = pair_scores(
        means, detection, np.array([0]), np.array([1]), PrefilterSettings()
    )
    # Gene 0: fold 3, detection 0.9 vs 0.05 -> a full-strength marker.
    assert score[0, 0] == pytest.approx(3.0 * 1.0 * 1.0)
    # Gene 1: fold 0.5 < 0.8 -> no candidate; gene 2: contrast 0.11 is weak.
    assert score[0, 1] == 0.0
    assert 0.0 < score[0, 2] < 0.2
    # Without detection the score is the fold above the minimum.
    plain = pair_scores(means, None, np.array([0]), np.array([1]), PrefilterSettings())
    assert plain[0].tolist() == pytest.approx([3.0, 0.0, 1.1])


def test_child_profiles_weight_leaves_by_cells() -> None:
    n_cells = np.array([10.0, 30.0, 5.0])
    total = np.array([[10.0], [90.0], [0.0]])
    detected = np.array([[5.0], [30.0], [0.0]])
    means, detection = child_profiles(
        n_cells,
        total,
        detected,
        SiblingSet(key="None", children=("a", "b"), leaf_rows=((0, 1), (2,))),
    )
    assert means[:, 0].tolist() == [pytest.approx(2.5), 0.0]
    assert detection is not None
    assert detection[:, 0].tolist() == [pytest.approx(35 / 40), 0.0]


def test_parent_order_covers_every_pair_before_the_rest() -> None:
    genes, n_cells, total, detected = one_hot_reference(6, 30)
    means = total / n_cells[:, None]
    order = parent_gene_order(
        "None",
        means,
        detected / n_cells[:, None],
        PrefilterSettings(markers_per_pair=1),
    )
    assert order.n_children == 6 and order.n_pairs == 15
    # Each pair needs one marker: five of the six child markers already
    # separate all 15 pairs, so coverage stops there.
    covering = order.order[: order.n_covering].tolist()
    assert len(covering) == 5 and set(covering) < set(range(6))
    assert sorted(order.order.tolist()) == list(range(len(genes)))


def test_prefilter_keeps_rare_child_markers_that_a_variance_ranking_drops() -> None:
    # 12 children, 10 broad genes that split most siblings, and one marker
    # per child. The rare child 11 (5 cells) copies child 10 on the broad
    # genes, child 10's own marker is flat, so only child 11's weak marker
    # (G0011, log2 fold 2) separates the pair (10, 11).
    rng = np.random.default_rng(1)
    n_children, n_broad = 12, 10
    genes = [f"G{index:04d}" for index in range(n_children + n_broad)]
    means = np.zeros((n_children, len(genes)))
    detection = np.zeros_like(means)
    for child in range(n_children):
        means[child, child] = 4.0
        detection[child, child] = 0.9
    means[10, 10], detection[10, 10] = 0.0, 0.0
    means[11, 11] = 2.0
    broad = rng.uniform(0.0, 6.0, size=(n_children, n_broad))
    broad[11] = broad[10]
    means[:, n_children:] = broad
    detection[:, n_children:] = np.clip(broad / 6.0, 0.0, 1.0)
    n_cells = np.full(n_children, 50.0)
    n_cells[11] = 5.0
    total = means * n_cells[:, None]
    detected = detection * n_cells[:, None]
    result = per_parent_topk_union(
        genes,
        n_cells,
        total,
        detected,
        leaf_sets(n_children),
        PrefilterSettings(cap=12),
    )
    assert result.applied and len(result.genes) <= 12
    assert "G0011" in result.genes
    # A global between-leaf variance ranking with the same budget drops it.
    variance = means.var(axis=0)
    by_variance = {genes[index] for index in np.argsort(-variance, kind="stable")[:12]}
    assert "G0011" not in by_variance


def test_union_of_top_k_is_capped_and_k_is_the_largest() -> None:
    orders = [
        ParentOrder("a", 2, 1, 2, np.array([0, 1, 2, 3, 4])),
        ParentOrder("b", 2, 1, 2, np.array([5, 1, 6, 7, 8])),
    ]
    assert union_of_top_k(orders, 2) == {0, 1, 5}
    assert largest_k(orders, 3) == 2
    assert largest_k(orders, 4) == 2
    assert largest_k(orders, 5) == 3
    assert largest_k(orders, 0) == 0
    assert largest_k([], 10) == 0


def test_prefilter_is_a_no_op_at_or_below_the_cap() -> None:
    genes, n_cells, total, detected = one_hot_reference(4, 2)
    result = per_parent_topk_union(
        genes, n_cells, total, detected, leaf_sets(4), PrefilterSettings(cap=6)
    )
    assert not result.applied and result.genes == sorted(genes) and result.k == 0


def test_prefilter_is_deterministic_and_records_itself() -> None:
    genes, n_cells, total, detected = one_hot_reference(8, 50, seed=3)
    settings = PrefilterSettings(cap=20, markers_per_pair=2)
    first = per_parent_topk_union(
        genes, n_cells, total, detected, leaf_sets(8), settings
    )
    second = per_parent_topk_union(
        genes, n_cells, total, detected, leaf_sets(8), settings
    )
    assert first.genes == second.genes and first.k == second.k
    record = first.to_json()
    assert (
        record["method"] == PREFILTER_METHOD and record["version"] == PREFILTER_VERSION
    )
    assert record["n_genes"] == len(first.genes) <= 20
    assert record["genes"] == sorted(record["genes"])
    assert record["per_parent"]["None"]["n_pairs"] == 28
    summary = first.summary()
    assert "genes" not in summary and summary["genes_sha256"] == first.genes_sha256
    # Parents with a single child are skipped.
    single = SiblingSet(key="CLAS/x", children=("c0",), leaf_rows=((0,),))
    with_single = per_parent_topk_union(
        genes, n_cells, total, detected, [*leaf_sets(8), single], settings
    )
    assert with_single.n_parents == 1


def test_prefilter_payload_names_the_method_version_and_settings() -> None:
    payload = prefilter_payload(2000, 30)
    assert payload["method"] == PREFILTER_METHOD
    assert payload["version"] == PREFILTER_VERSION
    assert payload["settings"]["cap"] == 2000
    assert payload["settings"]["markers_per_pair"] == 30
    assert prefilter_payload(2000, 30) != prefilter_payload(1500, 30)


def test_prefilter_needs_a_column_per_gene() -> None:
    genes, n_cells, total, detected = one_hot_reference(3, 3)
    with pytest.raises(ValueError, match="column per gene"):
        per_parent_topk_union(genes[:-1], n_cells, total, detected, leaf_sets(3))
