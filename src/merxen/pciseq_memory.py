"""Store pciSeq cell tensors on disk without changing its model arithmetic.

This mixin targets the pciSeq revision checked by the isolated compute adapter.
The cell, gene and class axes, floating point dtypes and update order are kept.
Only operations independent between cells are evaluated in smaller slices.
In particular, the upstream eta and theta contractions remain unchanged: their
global reduction order must not depend on the chosen cell batch size.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.lib.format import open_memmap
from scipy.special import softmax


class CellBlockMixin:
    """Use disk backed cell tensors with the pinned upstream VarBayes class.

    Compose this before VarBayes, and after the spot batching mixin. The latter
    owns ``initialise_state`` and calls ``init_gamma_state`` instead of upstream's
    dense gamma initializer. The caller must verify the upstream revision before
    using the mixin and give each fit a separate working directory.

    ``eta_upd``, ``theta_upd`` and ``init_theta`` intentionally come from upstream.
    Their inputs have the original full shapes and dtypes, so their contractions
    and reductions are not replaced with sums of batch totals.
    """

    _merxen_cell_batch_size: int = 128
    _merxen_cell_work_dir: Path | None = None
    nC: int  # noqa: N815 -- upstream dimensions
    nG: int  # noqa: N815 -- upstream dimensions
    nK: int  # noqa: N815 -- upstream dimensions
    cells: Any
    spots: Any
    genes: Any
    single_cell: Any
    cellTypes: Any  # noqa: N815 -- upstream's public attribute
    config: dict[str, Any]
    _scaled_exp: np.ndarray
    _merxen_cell_arrays: dict[
        tuple[str, tuple[int, ...], str, tuple[int, ...]], np.memmap
    ]

    def cell_ranges(self) -> Iterator[tuple[int, int]]:
        """Yield bounded, ordered slices of the original cell axis."""
        size = self._merxen_cell_batch_size
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("The pciSeq cell batch size must be a positive integer")
        for start in range(0, self.nC, size):
            yield start, min(start + size, self.nC)

    def _mapped_array(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        axis_order: tuple[int, ...],
    ) -> np.memmap:
        directory = self._merxen_cell_work_dir
        if directory is None:
            spot_directory = getattr(self, "_merxen_spot_work_dir", None)
            if spot_directory is None:
                raise ValueError("The pciSeq cell tensor working directory is required")
            directory = Path(spot_directory) / "cells"
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, "_merxen_cell_arrays"):
            self._merxen_cell_arrays = {}
        arrays = self._merxen_cell_arrays
        dtype = np.dtype(dtype)
        key = (name, shape, dtype.str, axis_order)
        if key not in arrays:
            # A dtype can change after the first update. Keep a distinct file so
            # no still referenced mapping is truncated or silently reinterpreted.
            suffix = "".join(str(axis) for axis in axis_order)
            path = directory / f"{name}.{dtype.name}.{suffix}.npy"
            physical_shape = tuple(shape[axis] for axis in axis_order)
            storage = open_memmap(path, mode="w+", dtype=dtype, shape=physical_shape)
            arrays[key] = storage.transpose(tuple(np.argsort(axis_order)))
        return arrays[key]

    def init_gamma_state(self, a: float, b: float) -> None:
        """Represent the uniform upstream gamma priors without dense copies.

        These are read only broadcasts, not lower precision replacements. In
        particular scipy's scalar digamma makes the initial log gamma float64
        under the pinned NumPy version, which is deliberately retained.
        """
        shape = (self.nC, self.nG, self.nK)
        # A one-element array also matches upstream's array/scalar promotion on
        # NumPy versions where a zero-dimensional operand follows other rules.
        one = np.ones((1,), dtype=np.float32)
        for attribute, value in (
            ("_post_shape", one * a),
            ("_post_rate", one * b),
            ("_gamma_bar", one * (a / b)),
            ("_log_gamma_bar", one * self.spots._digamma(a, b)),
        ):
            setattr(self.spots, attribute, np.broadcast_to(value, shape))

    def gamma_for_cells(
        self, start: int, stop: int, *, log: bool = False
    ) -> np.ndarray:
        """Return a bounded cell slice of the current gamma expectation."""
        if not 0 <= start <= stop <= self.nC:
            raise ValueError("Gamma cell indices are outside the model")
        if stop - start > self._merxen_cell_batch_size:
            raise ValueError("Gamma cell query exceeds the configured cell batch size")
        array = self.spots.log_gamma_bar if log else self.spots.gamma_bar
        return cast(np.ndarray, array[start:stop])

    def gamma_for_spots(
        self, cell_ids: np.ndarray, gene_ids: np.ndarray, *, log: bool = False
    ) -> np.ndarray:
        """Return current class expectations for a bounded set of spot pairs."""
        cell_ids, gene_ids = np.asarray(cell_ids), np.asarray(gene_ids)
        if (
            cell_ids.ndim != 1
            or gene_ids.ndim != 1
            or cell_ids.shape != gene_ids.shape
            or cell_ids.dtype.kind not in "iu"
            or gene_ids.dtype.kind not in "iu"
            or (cell_ids >= self.nC).any()
            or (cell_ids < 0).any()
            or (gene_ids >= self.nG).any()
            or (gene_ids < 0).any()
        ):
            raise ValueError("Gamma spot queries require valid paired integer indices")
        maximum = getattr(self, "_merxen_spot_batch_size", 20_000)
        if len(cell_ids) > maximum:
            raise ValueError("Gamma spot query exceeds the configured spot batch size")
        array = self.spots.log_gamma_bar if log else self.spots.gamma_bar
        return cast(np.ndarray, array[cell_ids, gene_ids])

    def gamma_upd(self) -> None:
        """Apply upstream's pointwise gamma update into disk backed arrays."""
        contract = importlib.import_module("opt_einsum").contract
        shape = (self.nC, self.nG, self.nK)
        rho = self.config["rSpot"] + self.cells.geneCount
        area = self.cells.ini_cell_props["area_factor"]
        mu = self.single_cell.mean_expression_adj.values
        mappings: dict[str, np.memmap] = {}
        for start, stop in self.cell_ranges():
            # Keep the original operation order and dtypes. There is no reduction
            # between cells in any of these four expressions.
            scaled = contract("c,gk->cgk", area[start:stop], mu, optimize="optimal")
            # NumPy reductions select their summation kernel from strides. The
            # atlas DataFrame often supplies a Fortran-ordered G/K matrix, making
            # upstream's tensor physically C/K/G, not C/G/K. Preserve that full
            # axis permutation rather than merely storing equal tensor values.
            # The outer product has an unambiguous C stride even for batch size
            # one; subsequent singleton-cell ufunc outputs do not.
            axis_order = tuple(
                sorted(
                    range(scaled.ndim),
                    key=lambda axis: scaled.strides[axis],
                    reverse=True,
                )
            )
            beta = (
                scaled
                * self.genes.eta_bar[:, None]
                * self.cells.theta_bar[start:stop, None, :]
                + self.config["rSpot"]
            )
            log_gamma = self.spots.logGammaExpectation(rho[start:stop], beta)
            gamma = self.spots.gammaExpectation(rho[start:stop], beta)
            gamma[:, :, -1] = 1.0
            log_gamma[:, :, -1] = 0.0
            for name, value in (
                ("scaled_exp", scaled),
                ("post_rate", beta),
                ("log_gamma", log_gamma),
                ("gamma", gamma),
            ):
                mapping = self._mapped_array(name, shape, value.dtype, axis_order)
                mapping[start:stop] = value
                mappings[name] = mapping
        self._scaled_exp = mappings["scaled_exp"]
        self.spots._post_shape = rho
        self.spots._post_rate = mappings["post_rate"]
        self.spots._log_gamma_bar = mappings["log_gamma"]
        self.spots._gamma_bar = mappings["gamma"]

    def cell_to_cellType(self) -> None:  # noqa: N802 -- upstream method dispatch
        """Keep the upstream NB contribution tensor and class update arithmetic."""
        shape = (self.nC, self.nG, self.nK)
        axis_order = tuple(
            sorted(
                range(self._scaled_exp.ndim),
                key=lambda axis: self._scaled_exp.strides[axis],
                reverse=True,
            )
        )
        contributions: np.memmap | None = None
        for start, stop in self.cell_ranges():
            scaled = (
                np.einsum(
                    "cgk,g,ck->cgk",
                    self._scaled_exp[start:stop],
                    self.genes.eta_bar,
                    self.cells.theta_bar[start:stop],
                )
                + self.config["SpotReg"]
            )
            q = scaled / (self.config["rSpot"] + scaled)
            x = self.cells.geneCount[start:stop, :, None]
            block = x * np.log(q) + self.config["rSpot"] * np.log(1 - q)
            contributions = self._mapped_array(
                "nb_contr", shape, block.dtype, axis_order
            )
            contributions[start:stop] = block
        if contributions is None:
            raise ValueError("The pciSeq cell model must contain cells")
        self.cells.nb_contr = contributions
        # Retain the full original reduction call and strides, rather than
        # summing partial cell/gene totals. MRF must see the old probabilities of
        # every cell before any new probability is committed.
        contr = np.sum(contributions, axis=1)
        mrf = self.cells.calc_mrf()
        self.cells.mrf = mrf
        weights = contr + self.cellTypes.log_prior + mrf
        self.cells.classProb = softmax(weights, axis=1)
