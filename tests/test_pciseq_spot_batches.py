"""Ordered reductions and actual upstream parity for the optional spot adapter."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.pciseq_spot_batches import SpotBatchMixin


def _small_spots(dtype: Any = np.float64) -> SpotBatchMixin:
    rng = np.random.default_rng(912)
    model = SpotBatchMixin()
    model.nS, model.nC, model.nG, model.nN = 137, 7, 5, 4
    parent = rng.integers(1, model.nC - 1, size=(model.nS, model.nN), dtype=np.uint32)
    parent[:, -1] = 0
    probabilities = rng.random(parent.shape).astype(dtype)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    data = pd.DataFrame(
        rng.normal(size=(model.nS, 3)) + 10_000,
        columns=["x", "y", "z"],
    )
    model.spots = SimpleNamespace(
        parent_cell_id=parent,
        parent_cell_prob=probabilities,
        gene_id=rng.integers(model.nG, size=model.nS),
        data=data,
    )
    prior = pd.DataFrame(np.zeros((model.nC, 3)), columns=["x", "y", "z"])
    model.cells = SimpleNamespace(dim=3, ini_centroids=lambda: prior)
    model.genes = SimpleNamespace(calc_rho=lambda counts: setattr(model, "rho", counts))
    model.config = {}
    return model


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("batch", [1, 11, 1000])
def test_gene_counts_keep_individual_reduction_order(dtype: Any, batch: int) -> None:
    """Initial float32 and fitted float64 sums retain their upstream rounding."""
    model = _small_spots(dtype)
    model._merxen_spot_batch_size = batch
    expected = np.zeros((model.nC, model.nG), dtype=dtype)
    for row in range(model.nS):
        for column in range(model.nN):
            expected[
                model.spots.parent_cell_id[row, column], model.spots.gene_id[row]
            ] += model.spots.parent_cell_prob[row, column]
    model.geneCount_upd()
    np.testing.assert_array_equal(model.cells.background_counts, expected[0])
    assert model.cells.background_counts.dtype == dtype
    np.testing.assert_array_equal(model.cells.geneCount[1:], expected[1:].astype("f4"))
    assert np.all(model.cells.geneCount[0] == 0)


@pytest.mark.parametrize("batch", [1, 17, 1000])
def test_rho_matches_global_bincount_exactly(batch: int) -> None:
    """Batches never regroup the additions into differently rounded subtotals."""
    model = _small_spots(np.float32)
    model._merxen_spot_batch_size = batch
    expected = np.bincount(
        model.spots.gene_id, model.spots.parent_cell_prob[:, -1], minlength=model.nG
    )
    model.rho_upd()
    np.testing.assert_array_equal(model.rho, expected)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("batch", [1, 19, 1000])
def test_gaussian_uses_centered_scatter_at_large_coordinates(
    dimension: int, batch: int
) -> None:
    """Two centered passes preserve tiny spread and prior-only empty cells."""
    model = _small_spots()
    model.cells.dim = dimension
    model._merxen_spot_batch_size = batch
    parent = model.spots.parent_cell_id[:, :-1].ravel()
    weights = model.spots.parent_cell_prob[:, :-1].ravel()
    xyz = model.spots.data[["x", "y", "z"]].values.astype("f4").astype("f8")
    repeated = np.repeat(xyz[:, :dimension], model.nN - 1, axis=0)
    expected_counts = np.bincount(parent, weights, minlength=model.nC)
    expected_mean = np.zeros((model.nC, dimension))
    for axis in range(dimension):
        expected_mean[:, axis] = np.bincount(
            parent, weights * repeated[:, axis], minlength=model.nC
        )
    has_spots = expected_counts > 0
    expected_mean[has_spots] /= expected_counts[has_spots, None]
    deviations = repeated - expected_mean[parent]
    expected_scatter = np.zeros((model.nC, dimension, dimension))
    for first in range(dimension):
        for second in range(dimension):
            expected_scatter[:, first, second] = np.bincount(
                parent,
                weights * deviations[:, first] * deviations[:, second],
                minlength=model.nC,
            )
    counts, means, scatter = model.soft_spot_statistics()
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(means, expected_mean)
    np.testing.assert_allclose(scatter, expected_scatter, atol=1e-12, rtol=1e-15)


@pytest.mark.parametrize("batch", [1, 23, 1000])
def test_convergence_matches_global_max(batch: int, tmp_path: Path) -> None:
    """Compare first against zero, then against the frozen old probability matrix."""
    model = _small_spots()
    model._merxen_spot_work_dir = tmp_path
    model._merxen_spot_batch_size = batch
    previous = model.spots.parent_cell_prob[:, ::-1].copy()
    for before in [None, previous]:
        expected = np.max(
            np.abs(model.spots.parent_cell_prob - (0 if before is None else before))
        )
        decision, delta = model.spot_convergence(before, float(expected), verbose=True)
        assert delta == expected
        assert not decision  # Upstream stops at strict less-than tolerance.
    model.flush_spot_arrays()
    assert list(tmp_path.glob("convergence_per_spot_*.npy"))


def test_invalid_batch_size_is_rejected() -> None:
    model = _small_spots()
    model._merxen_spot_batch_size = 0
    with pytest.raises(ValueError, match="positive"):
        model.rho_upd()


def test_convergence_propagates_nonfinite_probabilities() -> None:
    """A NaN must never be silently turned into a converged zero delta."""
    model = _small_spots()
    model._merxen_spot_batch_size = 3
    model.spots.parent_cell_prob[17, 2] = np.nan
    decision, delta = model.spot_convergence(None, 1.0)
    assert np.isnan(delta)
    assert not decision


def _real_model(kind: type, *, dimension: int = 2) -> Any:
    config_module = pytest.importorskip("pciSeq.config")
    rng = np.random.default_rng(209)
    centers = np.array(
        [
            [10, 10],
            [20, 20],
            [30, 30],
            [40, 40],
            [10, 50],
            [20, 50],
            [30, 50],
            [40, 50],
        ],
        dtype=float,
    )
    cells = pd.DataFrame(
        {
            "label": np.arange(1, 9),
            "area": [100] * 8,
            "x0": centers[:, 0],
            "y0": centers[:, 1],
            "z0": [5] * 8,
            "values": [[[x, y, 5]] for x, y in centers],
        }
    )
    genes = [f"gene_{index}" for index in range(5)]
    spots = pd.DataFrame(
        {
            "x": rng.uniform(0, 50, 81),
            "y": rng.uniform(0, 55, 81),
            "z": rng.uniform(0, 10, 81) if dimension == 3 else np.zeros(81),
            "plane_id": np.zeros(81),
            "label": rng.integers(0, 9, 81),
            "gene_name": np.resize(genes, 81),
            "score": np.ones(81),
            "intensity": np.ones(81),
        }
    )
    reference = pd.DataFrame(
        rng.uniform(1, 50, (5, 3)),
        index=genes,
        columns=["class_a", "class_b", "class_c"],
    )
    config = copy.deepcopy(config_module.DEFAULT)
    config.update(
        is3D=dimension == 3,
        img_dim={"w": 100, "h": 100, "n_planes": 10},
        label_map={},
        nNeighbors=3,
        max_iter=3,
        CellCallTolerance=0.0,
        save_data=False,
        cell_centroid_prior={"default": 10},
        cell_cov_prior={"default": 10},
        InsideCellBonus=1.7,
        MisreadDensity={"default": 1e-6},
    )
    return kind(cells, spots, reference, config)


def _real_pair(tmp_path: Path, batch: int, dimension: int = 2) -> tuple[Any, Any]:
    upstream = pytest.importorskip("pciSeq.src.core.main")
    from merxen.pciseq_memory import CellBlockMixin

    class Bounded(SpotBatchMixin, CellBlockMixin, upstream.VarBayes):
        pass

    original = _real_model(upstream.VarBayes, dimension=dimension)
    bounded = _real_model(Bounded, dimension=dimension)
    bounded._merxen_spot_batch_size = batch
    bounded._merxen_cell_batch_size = 2
    bounded._merxen_spot_work_dir = tmp_path
    return original, bounded


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("batch", [1, 13, 1000])
def test_actual_upstream_initialization_and_spot_steps(
    tmp_path: Path, dimension: int, batch: int
) -> None:
    """Actual pinned model arrays agree before and after each transcript update."""
    original, bounded = _real_pair(tmp_path, batch, dimension)
    original.initialise_state()
    bounded.initialise_state()
    for field in ["parent_cell_id", "parent_cell_prob", "Dist"]:
        np.testing.assert_array_equal(
            getattr(bounded.spots, field), getattr(original.spots, field)
        )
    for _ in range(2):
        original.geneCount_upd()
        bounded.geneCount_upd()
        np.testing.assert_array_equal(bounded.cells.geneCount, original.cells.geneCount)
        np.testing.assert_array_equal(
            bounded.cells.background_counts, original.cells.background_counts
        )
        original.rho_upd()
        bounded.rho_upd()
        np.testing.assert_array_equal(bounded.genes.rho_bar, original.genes.rho_bar)
        original.gamma_upd()
        bounded.gamma_upd()
        original.spots_to_cell_numba()
        bounded.spots_to_cell_numba()
        for field in [
            "parent_cell_prob",
            "attention",
            "expr_fluctuations",
            "cell_inefficiency",
            "gene_inefficiency",
            "mvn_loglik_arr",
        ]:
            np.testing.assert_array_equal(
                getattr(bounded.spots, field), getattr(original.spots, field)
            )
        from pciSeq.src.core.utils.gaussian_model import soft_stats

        for actual, expected in zip(
            bounded.soft_spot_statistics(),
            soft_stats(original.spots, original.cells, dimension),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        original.gaussian_upd()
        bounded.gaussian_upd()
        np.testing.assert_array_equal(bounded.cells.scale, original.cells.scale)


@pytest.mark.parametrize("batch", [7, 1000])
def test_actual_upstream_multi_iteration_and_frozen_probability_buffers(
    tmp_path: Path, batch: int
) -> None:
    """Main-loop results, stopping deltas and mapped buffers survive full passes."""
    original, bounded = _real_pair(tmp_path, batch)
    original.initialise_state()
    bounded.initialise_state()
    original._merxen_collect_data = None
    bounded._merxen_collect_data = lambda *_: ("collected", "streamed")
    original.main_loop()
    assert bounded.main_loop() == ("collected", "streamed")
    np.testing.assert_array_equal(bounded.iter_delta, original.iter_delta)
    np.testing.assert_array_equal(bounded.cells.classProb, original.cells.classProb)
    np.testing.assert_array_equal(
        bounded.spots.parent_cell_prob, original.spots.parent_cell_prob
    )
    np.testing.assert_array_equal(bounded.cells.scale, original.cells.scale)
    assert isinstance(bounded.spots.parent_cell_prob, np.memmap)
    assert bounded.has_converged == original.has_converged
    assert len(list(tmp_path.glob("parent_prob_*.npy"))) == 2


@pytest.mark.parametrize("dimension", [2, 3])
def test_plain_spot_formula_matches_actual_upstream(
    tmp_path: Path, dimension: int
) -> None:
    """The readable NumPy method retains the same independent row equations."""
    original, bounded = _real_pair(tmp_path, 9, dimension)
    for model in [original, bounded]:
        model.initialise_state()
        model.geneCount_upd()
        model.gamma_upd()
        model.spots_to_cell()
    for field in [
        "parent_cell_prob",
        "attention",
        "expr_fluctuations",
        "cell_inefficiency",
        "gene_inefficiency",
        "mvn_loglik_arr",
    ]:
        np.testing.assert_array_equal(
            getattr(bounded.spots, field), getattr(original.spots, field)
        )
