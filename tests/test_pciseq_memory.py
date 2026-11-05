"""Storage changes must retain the pinned model's intermediate values."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import psi, softmax

from merxen.pciseq_memory import CellBlockMixin


class SpotState:
    """Small upstream-compatible gamma state without importing optional pciSeq."""

    def __init__(self) -> None:
        self._gamma_bar: np.ndarray | None = None
        self._log_gamma_bar: np.ndarray | None = None

    @property
    def gamma_bar(self) -> np.ndarray:
        assert self._gamma_bar is not None
        return self._gamma_bar

    @property
    def log_gamma_bar(self) -> np.ndarray:
        assert self._log_gamma_bar is not None
        return self._log_gamma_bar

    @staticmethod
    def _digamma(a: Any, b: Any) -> Any:
        return psi(a) - np.log(b)

    @staticmethod
    def logGammaExpectation(  # noqa: N802 -- upstream helper
        rho: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        return psi(rho[:, :, None]) - np.log(beta)

    @staticmethod
    def gammaExpectation(  # noqa: N802 -- upstream helper
        rho: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        import opt_einsum as oe

        return oe.contract("cg,cgk->cgk", rho, 1 / beta, optimize="optimal")


class CellState:
    """Keep the original class update's simultaneous MRF dependency visible."""

    def __init__(
        self, rng: np.random.Generator, cells: int, genes: int, classes: int
    ) -> None:
        self.geneCount = rng.uniform(0.0, 100.0, (cells, genes)).astype(np.float32)
        self.ini_gene_counts = self.geneCount.sum(axis=1)
        self.ini_cell_props = {
            "area_factor": rng.uniform(0.1, 3.0, cells).astype(np.float32)
        }
        self.theta_bar = rng.uniform(0.2, 2.0, (cells, classes)).astype(np.float32)
        self.classProb = softmax(rng.normal(size=(cells, classes)), axis=1)
        self.background_counts = rng.uniform(0.1, 2.0, genes)

    def calc_mrf(self) -> np.ndarray:
        return np.roll(self.classProb, 1, axis=0) * 0.13

    def calc_theta(self, a: np.ndarray, b: np.ndarray) -> None:
        self.theta_bar = a.astype(np.float32)[:, None] / b.astype(np.float32)
        self.theta_bar[:, -1] = 1.0


class GeneState:
    def __init__(self, rng: np.random.Generator, genes: int) -> None:
        self.eta_bar = rng.uniform(0.2, 3.0, genes)
        self.counts_per_gene = rng.integers(10_000, 20_000, genes)

    def calc_eta(self, a: np.ndarray, b: np.ndarray) -> None:
        self.eta_bar = a / b
        self.logeta_bar = psi(a) - np.log(b)


class DenseReference:
    """Literal upstream expressions for the two overridden update methods."""

    def gamma_upd(self) -> None:
        import opt_einsum as oe

        self._scaled_exp = oe.contract(
            "c,gk->cgk",
            self.cells.ini_cell_props["area_factor"],
            self.single_cell.mean_expression_adj.values,
            optimize="optimal",
        )
        beta = (
            self._scaled_exp
            * self.genes.eta_bar[:, None]
            * self.cells.theta_bar[:, None, :]
            + self.config["rSpot"]
        )
        rho = self.config["rSpot"] + self.cells.geneCount
        self.spots._post_shape = rho
        self.spots._post_rate = beta
        self.spots._log_gamma_bar = self.spots.logGammaExpectation(rho, beta)
        self.spots._gamma_bar = self.spots.gammaExpectation(rho, beta)
        self.spots._gamma_bar[:, :, -1] = 1.0
        self.spots._log_gamma_bar[:, :, -1] = 0.0

    def cell_to_cellType(self) -> None:  # noqa: N802 -- upstream dispatch
        scaled = (
            np.einsum(
                "cgk,g,ck->cgk",
                self._scaled_exp,
                self.genes.eta_bar,
                self.cells.theta_bar,
            )
            + self.config["SpotReg"]
        )
        q = scaled / (self.config["rSpot"] + scaled)
        x = self.cells.geneCount[:, :, None]
        contr = x * np.log(q) + self.config["rSpot"] * np.log(1 - q)
        self.cells.nb_contr = contr
        mrf = self.cells.calc_mrf()
        self.cells.mrf = mrf
        self.cells.classProb = softmax(
            contr.sum(axis=1) + self.cellTypes.log_prior + mrf, axis=1
        )


class BlockReference(CellBlockMixin, DenseReference):
    pass


def make_model(cls: type[Any], directory: Path, batch: int) -> Any:
    """Use nonuniform values, including the Zero column and background cell."""
    rng = np.random.default_rng(73)
    model = cls.__new__(cls)
    model.nC, model.nG, model.nK = 7, 5, 4
    model.cells = CellState(rng, model.nC, model.nG, model.nK)
    model.genes = GeneState(rng, model.nG)
    model.spots = SpotState()
    model.spots.gene_id = np.asarray([0, 4, 1, 2, 1, 3], dtype=np.int32)
    model.spots.parent_cell_prob = rng.uniform(0.01, 0.2, (6, 3))
    model.spots.counts_per_gene = np.full(model.nG, 100_000, dtype=np.int32)
    expression = rng.uniform(0.1, 10.0, (model.nG, model.nK)).astype(np.float32)
    expression[:, -1] = 0.0
    model.single_cell = SimpleNamespace(mean_expression_adj=pd.DataFrame(expression))
    model.cellTypes = SimpleNamespace(log_prior=np.log([0.3, 0.1, 0.2, 0.4]))
    model.config = {"rSpot": 2, "rGene": 20, "rTheta": 2, "SpotReg": 0.1}
    model._merxen_cell_batch_size = batch
    model._merxen_cell_work_dir = directory
    return model


@pytest.mark.parametrize("batch", [1, 3, 7, 128])
def test_blocked_updates_are_bitwise_equal_and_keep_previous_state(
    tmp_path: Path, batch: int
) -> None:
    dense = make_model(DenseReference, tmp_path / "dense", batch)
    blocked = make_model(BlockReference, tmp_path / "blocked", batch)
    for iteration in range(3):
        dense.gamma_upd()
        blocked.gamma_upd()
        for key in ("_post_shape", "_post_rate", "_gamma_bar", "_log_gamma_bar"):
            np.testing.assert_array_equal(
                getattr(dense.spots, key), getattr(blocked.spots, key)
            )
            assert getattr(dense.spots, key).dtype == getattr(blocked.spots, key).dtype
        np.testing.assert_array_equal(dense._scaled_exp, blocked._scaled_exp)
        dense.cell_to_cellType()
        blocked.cell_to_cellType()
        np.testing.assert_array_equal(dense.cells.nb_contr, blocked.cells.nb_contr)
        np.testing.assert_array_equal(dense.cells.classProb, blocked.cells.classProb)
        np.testing.assert_array_equal(dense.cells.mrf, blocked.cells.mrf)
        # Later parameter changes must not retroactively change the last gamma
        # posterior. It is the previous iteration's input to eta/theta.
        previous = blocked.gamma_for_spots(
            np.asarray([1, 3]), np.asarray([2, 4])
        ).copy()
        for model in (dense, blocked):
            model.genes.eta_bar *= 1.0 + (iteration + 1) / 100
            model.cells.theta_bar *= np.float32(1.01)
            model.cells.geneCount += np.float32(0.7)
        np.testing.assert_array_equal(
            previous, blocked.gamma_for_spots(np.asarray([1, 3]), np.asarray([2, 4]))
        )
    assert isinstance(blocked.spots.gamma_bar, np.memmap)
    assert isinstance(blocked.cells.nb_contr, np.memmap)
    assert len(list((tmp_path / "blocked").glob("*.npy"))) == 5


def test_initial_priors_preserve_dtype_and_use_read_only_broadcasts(
    tmp_path: Path,
) -> None:
    blocked = make_model(BlockReference, tmp_path, 3)
    blocked.init_gamma_state(2, 2)
    shape = (blocked.nC, blocked.nG, blocked.nK)
    expected = {
        "_post_shape": np.ones(shape, dtype=np.float32) * 2,
        "_post_rate": np.ones(shape, dtype=np.float32) * 2,
        "_gamma_bar": np.ones(shape, dtype=np.float32),
        "_log_gamma_bar": np.ones(shape, dtype=np.float32) * (psi(2) - np.log(2)),
    }
    for name, value in expected.items():
        actual = getattr(blocked.spots, name)
        np.testing.assert_array_equal(actual, value)
        assert actual.dtype == value.dtype
        assert actual.strides == (0, 0, 0)
        assert not actual.flags.writeable
    assert not list(tmp_path.glob("*.npy"))


@pytest.mark.parametrize("batch", [1, 3, 128])
def test_full_panel_fortran_layout_preserves_upstream_summation(
    tmp_path: Path, batch: int
) -> None:
    """G-contiguous tensors trigger a different NumPy sum than C-order copies.

    Both layouts contain bitwise equal contributions, but converting the real
    atlas's C/K/G layout to C/G/K changes the gene sum at 815 genes. Batch size
    one also has degenerate C strides in its ufunc outputs, so it must infer the
    full tensor layout from the original outer product instead.
    """
    rng = np.random.default_rng(187)
    dense = make_model(DenseReference, tmp_path / "dense", batch)
    blocked = make_model(BlockReference, tmp_path / "blocked", batch)
    for model in (dense, blocked):
        model.nC, model.nG, model.nK = 4, 815, 335
        model.cells = CellState(np.random.default_rng(18), 4, 815, 335)
        model.cells.geneCount *= np.float32(30)
        model.genes.eta_bar = np.linspace(0.2, 3.0, 815, dtype=np.float32)
        expression = np.asfortranarray(
            rng.uniform(0.1, 10.0, (815, 335)).astype(np.float32)
        )
        expression[:, -1] = 0
        # Reuse identical input values for the two independent model instances.
        if model is dense:
            original_expression = expression.copy(order="F")
        else:
            expression = original_expression.copy(order="F")
        model.single_cell = SimpleNamespace(
            mean_expression_adj=pd.DataFrame(expression)
        )
        model.cellTypes = SimpleNamespace(log_prior=np.log(np.full(335, 1.0 / 335)))
    dense.gamma_upd()
    blocked.gamma_upd()
    for name in ("_post_rate", "_gamma_bar", "_log_gamma_bar"):
        expected, actual = getattr(dense.spots, name), getattr(blocked.spots, name)
        np.testing.assert_array_equal(actual, expected)
        assert actual.strides == expected.strides
    assert blocked._scaled_exp.strides == dense._scaled_exp.strides
    dense.cell_to_cellType()
    blocked.cell_to_cellType()
    np.testing.assert_array_equal(blocked.cells.nb_contr, dense.cells.nb_contr)
    assert blocked.cells.nb_contr.strides == dense.cells.nb_contr.strides
    np.testing.assert_array_equal(blocked.cells.classProb, dense.cells.classProb)
    # Ensure the fixture detects the old, value-preserving C-order storage bug.
    assert not np.array_equal(
        dense.cells.nb_contr.sum(axis=1),
        np.ascontiguousarray(dense.cells.nb_contr).sum(axis=1),
    )


def test_gamma_queries_preserve_requested_indices_and_reject_unbounded_reads(
    tmp_path: Path,
) -> None:
    blocked = make_model(BlockReference, tmp_path, 3)
    blocked.gamma_upd()
    cells, genes = np.asarray([6, 0, 6, 3]), np.asarray([2, 4, 1, 0])
    for log in (False, True):
        array = blocked.spots.log_gamma_bar if log else blocked.spots.gamma_bar
        np.testing.assert_array_equal(
            blocked.gamma_for_spots(cells, genes, log=log), array[cells, genes]
        )
        np.testing.assert_array_equal(
            blocked.gamma_for_cells(2, 5, log=log), array[2:5]
        )
    with pytest.raises(ValueError, match="cell batch"):
        blocked.gamma_for_cells(0, 7)
    with pytest.raises(ValueError, match="integer"):
        blocked.gamma_for_spots(np.asarray([1.0]), np.asarray([2]))
    with pytest.raises(ValueError, match="integer"):
        blocked.gamma_for_spots(np.asarray([7]), np.asarray([2]))
    blocked._merxen_spot_batch_size = 2
    with pytest.raises(ValueError, match="spot batch"):
        blocked.gamma_for_spots(cells, genes)


def test_bad_batch_or_missing_work_directory_is_explicit(tmp_path: Path) -> None:
    model = make_model(BlockReference, tmp_path, 0)
    with pytest.raises(ValueError, match="positive integer"):
        model.gamma_upd()
    model._merxen_cell_batch_size = 3
    model._merxen_cell_work_dir = None
    with pytest.raises(ValueError, match="working directory"):
        model.gamma_upd()


def test_real_pinned_upstream_intermediates_are_bitwise_equal(tmp_path: Path) -> None:
    """The optional runtime exercises its actual eta/theta and helper methods."""
    pytest.importorskip("pciSeq")
    from pciSeq.src.core.main import VarBayes

    class Blocked(CellBlockMixin, VarBayes):
        pass

    for batch in (1, 3, 128):
        dense = make_model(VarBayes, tmp_path / "unused", batch)
        blocked = make_model(Blocked, tmp_path / str(batch), batch)
        # Initial gamma matches the real helper, including scalar dtype promotion.
        from pciSeq.src.core.datatypes.spots import Spots

        dense.spots.init_gamma = Spots.init_gamma.__get__(dense.spots)
        dense.spots.init_gamma(2, 2, [dense.nC, dense.nG, dense.nK])
        blocked.init_gamma_state(2, 2)
        dense.init_theta()
        blocked.init_theta()
        np.testing.assert_array_equal(dense.cells.theta_bar, blocked.cells.theta_bar)
        for _ in range(3):
            dense.eta_upd()
            blocked.eta_upd()
            np.testing.assert_array_equal(dense.genes.eta_bar, blocked.genes.eta_bar)
            np.testing.assert_array_equal(
                dense.genes.logeta_bar, blocked.genes.logeta_bar
            )
            dense.theta_upd()
            blocked.theta_upd()
            np.testing.assert_array_equal(
                dense.cells.theta_bar, blocked.cells.theta_bar
            )
            dense.gamma_upd()
            blocked.gamma_upd()
            for name in ("_post_shape", "_post_rate", "_gamma_bar", "_log_gamma_bar"):
                np.testing.assert_array_equal(
                    getattr(dense.spots, name), getattr(blocked.spots, name)
                )
            dense.cell_to_cellType()
            blocked.cell_to_cellType()
            np.testing.assert_array_equal(dense.cells.nb_contr, blocked.cells.nb_contr)
            np.testing.assert_array_equal(
                dense.cells.classProb, blocked.cells.classProb
            )
