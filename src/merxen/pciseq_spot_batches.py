"""Storage and transcript batching for the pinned pciSeq variational model.

The mixture is unchanged: every transcript, neighbour, gene and class still
participates in every update. Reductions visit individual spot/neighbour pairs
in upstream order; only independent spot calculations are split into batches.
"""

from __future__ import annotations

import logging
import mmap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


class SpotBatchMixin:
    """Precede ``CellBlockMixin, VarBayes`` to bound transcript temporaries.

    ``_merxen_spot_work_dir`` enables disk-backed long-lived arrays. The caller
    owns that task-local directory and its lifetime. ``_merxen_spot_batch_size``
    controls temporary memory without changing the statistical model.
    """

    _merxen_spot_batch_size = 20_000
    _merxen_spot_work_dir: str | Path | None = None
    cells: Any
    spots: Any
    genes: Any
    single_cell: Any
    cellTypes: Any  # noqa: N815 - upstream interface
    config: dict[str, Any]
    nC: int  # noqa: N815 - upstream interface
    nG: int  # noqa: N815 - upstream interface
    nK: int  # noqa: N815 - upstream interface
    nS: int  # noqa: N815 - upstream interface
    nN: int  # noqa: N815 - upstream interface
    init_gamma_state: Any
    init_theta: Any
    centroid_upd: Any
    cov_upd: Any
    gamma_for_spots: Any
    gamma_upd: Any
    eta_upd: Any
    theta_upd: Any
    cell_to_cellType: Any  # noqa: N815 - upstream interface
    dalpha_upd: Any
    _roi_volume: Any
    _step: Any
    _step_times: dict[str, float]
    on_iteration_callback: Any
    iter_delta: list[float]
    iter_num: int
    has_converged: bool

    def _spot_batches(self) -> Iterator[tuple[int, int]]:
        size = int(self._merxen_spot_batch_size)
        if size < 1:
            raise ValueError("pciSeq spot batch size must be positive")
        for start in range(0, self.nS, size):
            yield start, min(start + size, self.nS)

    def _spot_array(self, name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        arrays = self.__dict__.setdefault("_merxen_spot_arrays", {})
        key = (name, np.dtype(dtype).str)
        if key in arrays:
            value: np.ndarray = arrays[key]
            if value.shape != shape:
                raise ValueError("pciSeq working array dimensions changed")
            return value
        if self._merxen_spot_work_dir is None:
            value = np.empty(shape, dtype=dtype)
        else:
            directory = Path(self._merxen_spot_work_dir)
            directory.mkdir(parents=True, exist_ok=True)
            value = np.lib.format.open_memmap(
                directory / f"{name}_{np.dtype(dtype).name}.npy",
                mode="w+",
                dtype=dtype,
                shape=shape,
            )
        arrays[key] = value
        return value

    def flush_spot_arrays(self) -> None:
        """Flush task-local memmaps and allow the OS to reclaim their pages."""
        for value in self.__dict__.get("_merxen_spot_arrays", {}).values():
            if isinstance(value, np.memmap):
                value.flush()
                mapping = getattr(value, "_mmap", None)
                if mapping is not None and hasattr(mapping, "madvise"):
                    mapping.madvise(mmap.MADV_DONTNEED)

    def _spot_coordinates(self, start: int, stop: int) -> np.ndarray:
        # xyz_coords upstream casts to float32 before any Gaussian calculation.
        return np.asarray(
            self.spots.data.iloc[start:stop][["x", "y", "z"]].values,
            dtype=np.float32,
        )

    def initialise_state(self) -> None:
        """Preserve upstream initialization while batching nearest-cell queries."""
        self.cellTypes.ini_prior()
        self.cells.nbrs = self.cells.nearest_neighbours()
        self.cells.classProb = np.tile(self.cellTypes.prior, (self.nC, 1))
        self.genes.init_eta(self.config["rGene"], self.config["rGene"])
        parent = self._spot_array("parent_cell_id", (self.nS, self.nN), np.uint32)
        distances = self._spot_array("parent_distances", parent.shape, np.float32)
        probabilities = self._spot_array(
            "initial_parent_prob", parent.shape, np.float32
        )
        initial_counts = np.zeros(self.nC, dtype=np.int64)
        neighbors = self.cells.nn()
        for start, stop in self._spot_batches():
            data = self.spots.data.iloc[start:stop]
            distance, nearest = neighbors.kneighbors(data[["z", "y", "x"]].values)
            nearest[:, -1] = 0
            parent[start:stop] = nearest
            distances[start:stop] = distance
            probabilities[start:stop] = np.float32(1.0 / self.nN)
            np.add.at(initial_counts, data.label.to_numpy(), 1)
        # The public setter would make another full uint32 copy.
        self.spots._parent_cell_id = parent
        self.spots.parent_cell_prob = probabilities
        self.spots.Dist = distances
        self.cells._ini_gene_counts = initial_counts
        self.genes._misread_density = self.genes.calc_misread_density()
        self.genes.init_rho(
            self.config["MisreadDensity"]["default"], self._roi_volume()
        )
        self.init_gamma_state(self.config["rSpot"], self.config["rSpot"])
        self.init_theta()
        self.flush_spot_arrays()

    def geneCount_upd(self) -> None:  # noqa: N802 - upstream override
        """Accumulate each spot/neighbour in upstream's exact flattened order."""
        probabilities = self.spots.parent_cell_prob
        accumulated = self._spot_array(
            "gene_count_accumulator", (self.nC, self.nG), probabilities.dtype
        )
        accumulated.fill(0)
        flat = accumulated.reshape(-1)
        for start, stop in self._spot_batches():
            indices = self.spots.parent_cell_id[start:stop].astype(np.int64)
            indices *= self.nG
            indices += self.spots.gene_id[start:stop, None]
            np.add.at(flat, indices.ravel(), probabilities[start:stop].ravel())
        slot = 1 - self.__dict__.get("_merxen_gene_count_slot", 0)
        out = self._spot_array(f"gene_count_{slot}", accumulated.shape, np.float32)
        out[0] = 0
        out[1:] = accumulated[1:]
        self.cells.background_counts = accumulated[0].copy()
        self.cells.geneCount = out
        self._merxen_gene_count_slot = slot

    def rho_upd(self) -> None:
        """Match bincount's individual float64 additions, without a full copy."""
        counts = np.zeros(self.nG, dtype=np.float64)
        for start, stop in self._spot_batches():
            np.add.at(
                counts,
                self.spots.gene_id[start:stop],
                self.spots.parent_cell_prob[start:stop, -1],
            )
        self.genes.calc_rho(counts)
        if self.config.get("verbose", False):
            logger.info(
                "rho_upd: bg_counts min/max=%.1f/%.1f, rho_bar min/max=%.2e/%.2e, "
                "log_rho min/max=%.4f/%.4f",
                counts.min(),
                counts.max(),
                self.genes.rho_bar.min(),
                self.genes.rho_bar.max(),
                self.genes.log_rho_bar.min(),
                self.genes.log_rho_bar.max(),
            )

    def _spots_to_cell_batched(self, *, compiled: bool) -> None:
        from pciSeq.src.core.utils.gaussian_model import expected_logdet_precision
        from pciSeq.src.core.utils.numba_kernels import spots_to_cell_numba_kernel

        shape = (self.nS, self.nN)
        slot = 1 - self.__dict__.get("_merxen_probability_slot", 1)
        probabilities = self._spot_array(f"parent_prob_{slot}", shape, np.float64)
        diagnostic_names = (
            "attention",
            "expr_fluctuations",
            "cell_inefficiency",
            "gene_inefficiency",
            "mvn_loglik_arr",
        )
        diagnostics = {
            name: self._spot_array(name, shape, np.float64) for name in diagnostic_names
        }
        expression = self.single_cell.log_mean_expression.loc[
            self.genes.gene_panel
        ].values
        log_theta = np.log(self.cells.theta_bar)
        logdet = expected_logdet_precision(
            self.cells.nu.astype(np.float64),
            self.cells.eig_vals.astype(np.float64),
        )
        for batch_number, (start, stop) in enumerate(self._spot_batches()):
            parent = self.spots.parent_cell_id[start:stop]
            gene_ids = self.spots.gene_id[start:stop]
            expected = expression[gene_ids]
            log_eta = self.genes.logeta_bar[gene_ids]
            weights = np.zeros((stop - start, self.nN), dtype=np.float64)
            weights[:, -1] = self.genes.log_rho_bar[gene_ids]
            for target in diagnostics.values():
                target[start:stop] = 0
            if compiled:
                # Keep the pinned parallel kernel unchanged. The cell adapter
                # supplies C-order memmaps, so this reads the full logical
                # posterior without upstream's whole-array contiguous copy.
                spots_to_cell_numba_kernel(
                    parent,
                    gene_ids,
                    self.cells.classProb,
                    self.spots.log_gamma_bar,
                    log_theta,
                    expected,
                    log_eta,
                    self.nN - 1,
                    weights,
                    diagnostics["attention"][start:stop],
                    diagnostics["expr_fluctuations"][start:stop],
                    diagnostics["cell_inefficiency"][start:stop],
                    diagnostics["gene_inefficiency"][start:stop],
                )
            coordinates = self._spot_coordinates(start, stop)
            for neighbor in range(self.nN - 1):
                cell_ids = parent[:, neighbor]
                if not compiled:
                    classes = self.cells.classProb[cell_ids]
                    gamma = self.gamma_for_spots(cell_ids, gene_ids, log=True)
                    theta = log_theta[cell_ids]
                    first = np.einsum("ij,ij->i", expected, classes)
                    second = np.einsum("ij,ij->i", classes, gamma)
                    third = np.einsum("ij,ij->i", classes, theta)
                    weights[:, neighbor] = first + second + third + log_eta
                    diagnostics["attention"][start:stop, neighbor] = first
                    diagnostics["expr_fluctuations"][start:stop, neighbor] = second
                    diagnostics["cell_inefficiency"][start:stop, neighbor] = third
                    diagnostics["gene_inefficiency"][start:stop, neighbor] = log_eta
                spatial = self._spot_spatial_loglik(coordinates, cell_ids, logdet)
                # Match upstream's operation association, including adding the
                # position term after the four expression/efficiency terms.
                weights[:, neighbor] += spatial
                diagnostics["mvn_loglik_arr"][start:stop, neighbor] = spatial
            bonus = self.spots.data.label.iloc[start:stop].to_numpy()[:, None] == parent
            bonus[:, -1] = False
            weights += bonus * self.config["InsideCellBonus"]
            probabilities[start:stop] = softmax(weights, axis=1)
            if batch_number and batch_number % 32 == 0:
                self.flush_spot_arrays()
        self.spots.parent_cell_prob = probabilities
        self._merxen_probability_slot = slot
        for name, value in diagnostics.items():
            setattr(self.spots, name, value)
        # Parameters remain frozen until ALL transcripts have been processed.
        self.geneCount_upd()
        self.flush_spot_arrays()

    def _spot_spatial_loglik(
        self, coordinates: np.ndarray, cell_ids: np.ndarray, logdet: np.ndarray
    ) -> np.ndarray:
        """Upstream's NIW position formula with its frozen cell term cached."""
        import opt_einsum as oe

        dimension = 3 if self.config["is3D"] else 2
        positions = coordinates[:, :dimension].astype(np.float64)
        centroids = self.cells.centroid.values[cell_ids, :dimension].astype(np.float64)
        values = self.cells.eig_vals[cell_ids].astype(np.float64)
        vectors = self.cells.eig_vecs[cell_ids].astype(np.float64)
        nu = self.cells.nu[cell_ids].astype(np.float64)
        kappa = self.cells.kappa[cell_ids].astype(np.float64)
        deviations = oe.contract(
            "ni,nij->nj", positions - centroids, vectors, optimize="optimal"
        )
        mahalanobis = nu * np.sum(deviations * deviations / values, axis=1)
        result: np.ndarray = -0.5 * (
            dimension * np.log(2 * np.pi)
            - logdet[cell_ids]
            + dimension / kappa
            + mahalanobis
        )
        return result

    def spots_to_cell(self) -> None:
        """Evaluate the readable NumPy spot formula in independent batches."""
        self._spots_to_cell_batched(compiled=False)

    def spots_to_cell_numba(self) -> None:
        """Evaluate upstream's ordered class sums in independent batches."""
        self._spots_to_cell_batched(compiled=True)

    def soft_spot_statistics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute upstream's two-pass centered Gaussian statistics in order."""
        dimension = self.cells.dim
        counts = np.zeros(self.nC, dtype=np.float64)
        means = np.zeros((self.nC, dimension), dtype=np.float64)
        n_neighbors = self.nN - 1
        for start, stop in self._spot_batches():
            ids = self.spots.parent_cell_id[start:stop, :-1].ravel()
            weights = (
                self.spots.parent_cell_prob[start:stop, :-1].ravel().astype(np.float64)
            )
            coordinates = np.repeat(
                self._spot_coordinates(start, stop)[:, :dimension].astype(np.float64),
                n_neighbors,
                axis=0,
            )
            np.add.at(counts, ids, weights)
            for axis in range(dimension):
                np.add.at(means[:, axis], ids, weights * coordinates[:, axis])
        has_spots = counts > 0
        means[has_spots] /= counts[has_spots, None]
        prior = self.cells.ini_centroids().values[:, :dimension].astype(np.float64)
        means[~has_spots] = prior[~has_spots]
        scatter = np.zeros((self.nC, dimension, dimension), dtype=np.float64)
        for start, stop in self._spot_batches():
            ids = self.spots.parent_cell_id[start:stop, :-1].ravel()
            weights = (
                self.spots.parent_cell_prob[start:stop, :-1].ravel().astype(np.float64)
            )
            coordinates = np.repeat(
                self._spot_coordinates(start, stop)[:, :dimension].astype(np.float64),
                n_neighbors,
                axis=0,
            )
            deviations = coordinates - means[ids]
            for first in range(dimension):
                for second in range(first, dimension):
                    np.add.at(
                        scatter[:, first, second],
                        ids,
                        weights * deviations[:, first] * deviations[:, second],
                    )
        for first in range(dimension):
            for second in range(first + 1, dimension):
                scatter[:, second, first] = scatter[:, first, second]
        return counts, means, scatter

    def gaussian_upd(self) -> None:
        """Retain the NIW centroid/covariance updates with bounded statistics."""
        counts, means, scatter = self.soft_spot_statistics()
        self.centroid_upd(counts, means)
        self.cov_upd(counts, means, scatter)

    def spot_convergence(
        self, previous: np.ndarray | None, tolerance: float, verbose: bool = False
    ) -> tuple[bool, float]:
        """Apply the unchanged maximum-probability-change stopping criterion."""
        delta = 0.0
        per_spot = (
            self._spot_array("convergence_per_spot", (self.nS,), np.float64)
            if verbose
            else None
        )
        for start, stop in self._spot_batches():
            current = self.spots.parent_cell_prob[start:stop]
            difference = np.abs(
                current if previous is None else current - previous[start:stop]
            )
            delta = float(np.maximum(delta, np.max(difference)))
            if per_spot is not None:
                per_spot[start:stop] = difference.max(axis=1)
        if per_spot is not None:
            over = int(np.count_nonzero(per_spot > tolerance))
            logger.info(
                "convergence detail: max=%.6f | spots over tol(%.3f)=%d/%d "
                "(%.4f%%) | mean per-spot=%.2e | 99pct=%.4f | 99.9pct=%.4f",
                delta,
                tolerance,
                over,
                self.nS,
                100.0 * over / self.nS,
                float(per_spot.mean()),
                float(np.percentile(per_spot, 99.0)),
                float(np.percentile(per_spot, 99.9)),
            )
        return delta < tolerance, delta

    def main_loop(self) -> tuple[Any, Any]:
        """Keep upstream's update order, callbacks and collector boundary."""
        from pciSeq.src.core import summary
        from pciSeq.src.core.utils import convergence

        previous = None
        verbose = self.config.get("verbose", False)
        for iteration in range(self.config["max_iter"]):
            self.iter_num = iteration
            self._step_times = {}
            classes_before = self.cells.classProb.copy() if verbose else None
            for name, update in (
                ("geneCount_upd", self.geneCount_upd),
                ("rho_upd", self.rho_upd),
                ("eta_upd", self.eta_upd),
                ("theta_upd", self.theta_upd),
                ("gamma_upd", self.gamma_upd),
                ("cell_to_cellType", self.cell_to_cellType),
            ):
                self._step(name, update)
            if self.config["cell_type_prior"] == "weighted":
                self._step("dalpha_upd", self.dalpha_upd)
            self._step("spots_to_cell", self.spots_to_cell_numba)
            self._step("gaussian_upd", self.gaussian_upd)
            if verbose:
                breakdown = " ".join(
                    f"{key}={value:.2f}" for key, value in self._step_times.items()
                )
                logger.info(
                    "STEP TIMES iter %d (total %.2fs): %s",
                    iteration,
                    sum(self._step_times.values()),
                    breakdown,
                )
            self.has_converged, delta = self.spot_convergence(
                previous, self.config["CellCallTolerance"], verbose
            )
            convergence.log_iteration_diagnostics(
                self, iteration, delta, previous, classes_before
            )
            if self.on_iteration_callback is not None:
                try:
                    self.on_iteration_callback(self.cells.classProb, iteration, delta)
                except Exception as error:
                    logger.warning("Real-time viewer callback failed: %s", error)
            self.iter_delta.append(delta)
            previous = self.spots.parent_cell_prob
            if self.has_converged:
                break
            if iteration == self.config["max_iter"] - 1:
                logger.info(
                    "Loop exhausted. Exiting with convergence status: %s",
                    self.has_converged,
                )
        self.flush_spot_arrays()
        collector = getattr(self, "_merxen_collect_data", None) or summary.collect_data
        result: tuple[Any, Any] = collector(
            self.cells, self.spots, self.genes, self.config["is3D"]
        )
        return result
