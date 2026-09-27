"""Tests for the LL (vii) likelihood typer (plan §12 M3 item 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from merxen.annotation.likelihood import (
    mixture_posterior,
    platform_log2_factors,
    reference_from_profiles,
    run_ll_vii,
    summarize_posterior,
)
from merxen.annotation.reference import TaxonomyTreeView

GENES = [f"G{index}" for index in range(6)]
LEAF, PARENT = "CLUS", "SUPC"
# Three leaves: two neuron clusters (one parent) and one astrocyte cluster;
# each expresses its own marker genes strongly.
LEAF_CPM = {
    "C1": [900, 50, 10, 10, 10, 20],
    "C2": [50, 900, 10, 10, 10, 20],
    "C3": [10, 10, 900, 900, 10, 60],
    "C4": [10, 10, 10, 10, 900, 60],
}
LEAF_CELLS = {"C1": 100, "C2": 80, "C3": 120, "C4": 5}


def _profiles() -> pd.DataFrame:
    rows = []
    for node, values in LEAF_CPM.items():
        for gene, value in zip(GENES, values, strict=True):
            rows.append(
                {
                    "level": LEAF,
                    "node": node,
                    "node_name": f"cluster {node}",
                    "n_cells": LEAF_CELLS[node],
                    "gene_id": gene,
                    "mean_cpm": float(value),
                }
            )
    return pd.DataFrame(rows)


def _tree() -> TaxonomyTreeView:
    return TaxonomyTreeView.from_tree_dict(
        {
            "hierarchy": [PARENT, LEAF],
            PARENT: {"S_N": ["C1", "C2"], "S_A": ["C3"], "S_M": ["C4"]},
            LEAF: {"C1": [], "C2": [], "C3": [], "C4": []},
            "name_mapper": {
                PARENT: {
                    "S_N": {"name": "Neuron parent"},
                    "S_A": {"name": "Astrocyte"},
                    "S_M": {"name": "Microglia"},
                }
            },
        }
    )


BROAD = {"S_N": "Neurons", "S_A": "Astrocytes", "S_M": "Microglia"}


def _reference(**kwargs: object) -> object:
    return reference_from_profiles(
        _profiles(),
        _tree(),
        leaf_level=LEAF,
        parent_level=PARENT,
        broad_of_parent=BROAD,
        **kwargs,  # type: ignore[arg-type]
    )


def test_reference_from_profiles_drops_small_leaves_and_builds_contaminants() -> None:
    reference = _reference()
    assert reference.leaves == ("C1", "C2", "C3")  # C4 has 5 < 20 cells
    assert reference.parents == ("Astrocyte", "Neuron parent")
    assert reference.broad_of_parent == ("Astrocytes", "Neurons")
    assert np.allclose(reference.profiles.sum(axis=1), 1.0)
    assert reference.contaminant_classes == ("Neurons", "Astrocytes")
    # Uniform over parents, then over a parent's leaves.
    assert np.allclose(np.exp(reference.log_prior), [0.25, 0.25, 0.5])
    neuron_mean = reference.profiles[:2].mean(axis=0)
    assert np.allclose(reference.contaminants[0], neuron_mean)


def test_reference_from_profiles_restricts_and_rejects_genes() -> None:
    reference = _reference(gene_ids=GENES[:4])
    assert reference.gene_ids == tuple(GENES[:4])
    assert reference.profiles.shape == (3, 4)
    with pytest.raises(ValueError, match="no profile"):
        _reference(gene_ids=[*GENES, "G_MISSING"])
    with pytest.raises(ValueError, match="no leaf"):
        _reference(min_cells=1000)


def _cells(
    rng: np.random.Generator, reference: object, leaf: int, n: int
) -> np.ndarray:
    probabilities = reference.profiles[leaf]  # type: ignore[attr-defined]
    return rng.multinomial(60, probabilities, size=n)


def test_mixture_posterior_types_pure_cells_and_flags_spill_over() -> None:
    reference = _reference()
    rng = np.random.default_rng(0)
    pure = np.vstack([_cells(rng, reference, index, 20) for index in range(3)])
    mixed_profile = 0.7 * reference.profiles[0] + 0.3 * reference.contaminants[1]
    mixed = rng.multinomial(60, mixed_profile / mixed_profile.sum(), size=20)
    counts = sparse.csr_matrix(np.vstack([pure, mixed]))
    posterior = mixture_posterior(counts, reference, chunk_size=7)
    assert posterior.leaf.shape == (80, 3)
    assert np.allclose(posterior.leaf.sum(axis=1), 1.0)
    assert (posterior.leaf[:60].argmax(axis=1) == np.repeat([0, 1, 2], 20)).all()
    assert (posterior.leaf[60:].argmax(axis=1) == 0).mean() >= 0.9
    assert posterior.mix_fraction[60:].mean() > posterior.mix_fraction[:20].mean()
    assert (posterior.mix_partner[60:] == "Astrocytes").mean() >= 0.9
    with pytest.raises(ValueError, match="pure_prior"):
        mixture_posterior(counts, reference, pure_prior=1.0)
    with pytest.raises(ValueError, match="one column per"):
        mixture_posterior(counts[:, :3], reference)


def test_platform_factors_recover_a_gene_efficiency_and_cap() -> None:
    reference = _reference()
    rng = np.random.default_rng(1)
    efficiency = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 8.0])
    rows = []
    for index in range(3):
        profile = reference.profiles[index] * efficiency
        rows.append(rng.multinomial(200, profile / profile.sum(), size=200))
    counts = sparse.csr_matrix(np.vstack(rows))
    capped, uncapped = platform_log2_factors(counts, reference, cap_log2=1.0)
    assert uncapped[5] == pytest.approx(3.0, abs=0.4)
    assert abs(uncapped[:5]).max() < 0.5
    assert capped[5] == pytest.approx(1.0)
    same, raw = platform_log2_factors(counts, reference, cap_log2=None)
    assert np.allclose(same, raw)
    with pytest.raises(ValueError, match="one column per"):
        platform_log2_factors(counts[:, :2], reference)


def test_summarize_posterior_aggregates_to_parent_broad_and_lineage() -> None:
    reference = _reference()
    posterior = np.array([[0.3, 0.3, 0.4], [0.1, 0.1, 0.8]])
    calls = summarize_posterior(posterior, reference)
    assert list(calls["ll_cluster_name"]) == ["cluster C3", "cluster C3"]
    assert list(calls["ll_supercluster_name"]) == ["Neuron parent", "Astrocyte"]
    assert calls["ll_supercluster_post"].tolist() == pytest.approx([0.6, 0.8])
    assert list(calls["ll_broad_name"]) == ["Neurons", "Astrocytes"]
    assert list(calls["ll_lineage_name"]) == ["Neurons", "Astrocytes"]


def test_run_ll_vii_returns_calls_indexed_by_cell() -> None:
    reference = _reference()
    rng = np.random.default_rng(2)
    counts = np.vstack([_cells(rng, reference, index, 5) for index in range(3)])
    result = run_ll_vii(counts, reference, cell_ids=[f"c{i}" for i in range(15)])
    assert list(result.calls.index[:2]) == ["c0", "c1"]
    assert {"ll_broad_name", "ll_mix_fraction", "ll_mix_partner"} <= set(
        result.calls.columns
    )
    assert result.log2_factors.shape == (len(GENES),)
    assert np.all(np.abs(result.log2_factors) <= 2.0)
    assert (result.calls["ll_broad_name"].iloc[10:] == "Astrocytes").all()
