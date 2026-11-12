"""Storage and output adapters for the pinned pciSeq model on large samples.

The model remains global. Cell and transcript blocks only bound temporary
allocations; they are never fitted as independent samples.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Internal entry points and equations used by the storage adapter. In addition
# to the git metadata, reject edited installations carrying the same commit ID.
_VERIFIED_SOURCE_SHA256 = {
    "app.py": ("963112e92d4e38c29fd2463e5237697a546a8a9379707490d45e24af1ea859b0"),
    "src/core/main.py": (
        "1fba21d6ccbed4b210f4186686acdec05b24deeec6d6ef2c817ce9901ba64674"
    ),
    "src/core/summary.py": (
        "2d47004ca75476892b2b0c581c3aa539cff951797d2f5b47b60be18e20a52a76"
    ),
    "src/core/datatypes/spots.py": (
        "b3de70fdfd455bcc0f729a1774c9bc281d03d75b907188408c063ce10b37ad22"
    ),
    "src/core/datatypes/cells.py": (
        "96634f3ad7feaf79b62c41bf74a8f7db8d1c9a9f757146ee8e583d0e9e85d856"
    ),
    "src/core/utils/likelihood.py": (
        "cadb477f824ce345512a98d85cfb2a46b60b40bc0b5d62cf4187e71ad1b6e290"
    ),
    "src/core/utils/numba_kernels.py": (
        "dd40b86041d247c9dd66d36678a4c05dfb88a4501924cb80a242117ce42dcd35"
    ),
    "src/core/utils/gaussian_model.py": (
        "53b5d69333a6d9a1a148715163a6e8351b6cfe2a3907b7449f132050c496487d"
    ),
}


def verify_upstream(expected_commit: str) -> dict[str, str]:
    """Require the exact source revision whose internal equations were checked."""
    distribution = importlib.metadata.distribution("pciSeq_3d")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id")
    if commit != expected_commit:
        raise RuntimeError(
            "Batched pciSeq requires the verified upstream revision "
            f"{expected_commit}; installed source is {commit!r}"
        )
    import pciSeq.app

    result = {"commit": commit}
    package = Path(pciSeq.app.__file__).parent
    for name, expected in _VERIFIED_SOURCE_SHA256.items():
        actual = hashlib.sha256((package / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Batched pciSeq source differs from verified code: {name}"
            )
        result[name] = actual
    for name in ("numpy", "scipy", "pandas", "opt-einsum", "numpy-groupies"):
        result[name] = importlib.metadata.version(name)
    return result


def _contributing_spots(
    cells: Any,
    spots: Any,
    genes: Any,
    tol: float,
    *,
    scratch_dir: Path,
    chunk_rows: int,
) -> np.ndarray:
    """Build the unchanged cell/gene spot lists using disk-backed counting sort.

    Both passes visit spots, then their candidate neighbors, in original order.
    Integer group counts and cursors retain every probability above the same
    upstream threshold; no floating-point aggregation is introduced here.
    """
    from numba import njit

    @njit
    def scatter(
        groups: np.ndarray,
        identifiers: np.ndarray,
        cursors: np.ndarray,
        output: np.ndarray,
    ) -> None:
        for index in range(len(groups)):
            group = groups[index]
            output[cursors[group]] = identifiers[index]
            cursors[group] += 1

    shape = (cells.nC, genes.nG)
    size = cells.nC * genes.nG
    counts = np.lib.format.open_memmap(
        scratch_dir / "contribution_counts.npy",
        mode="w+",
        dtype=np.int64,
        shape=(size,),
    )
    counts[:] = 0
    for start in range(0, spots.nS, chunk_rows):
        stop = min(start + chunk_rows, spots.nS)
        parent = spots.parent_cell_id[start:stop]
        groups = parent.astype(np.int64) * genes.nG + spots.gene_id[start:stop, None]
        keep = spots.parent_cell_prob[start:stop] > tol
        np.add.at(counts, groups[keep], 1)
    offsets = np.lib.format.open_memmap(
        scratch_dir / "contribution_offsets.npy",
        mode="w+",
        dtype=np.int64,
        shape=(size + 1,),
    )
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    output = np.lib.format.open_memmap(
        scratch_dir / "contributing_spot_ids.npy",
        mode="w+",
        dtype=np.dtype(spots.data.index.dtype),
        shape=(max(total, 1),),
    )
    cursors = offsets[:-1].copy()
    for start in range(0, spots.nS, chunk_rows):
        stop = min(start + chunk_rows, spots.nS)
        parent = spots.parent_cell_id[start:stop]
        groups = parent.astype(np.int64) * genes.nG + spots.gene_id[start:stop, None]
        keep = spots.parent_cell_prob[start:stop] > tol
        identifiers = np.broadcast_to(
            spots.data.index[start:stop].to_numpy()[:, None], parent.shape
        )
        scatter(groups[keep], identifiers[keep], cursors, output)
    result = np.empty(shape, dtype=object)
    result.fill([])
    flat = result.ravel()
    for group in np.flatnonzero(counts):
        flat[group] = output[int(offsets[group]) : int(offsets[group + 1])]
    return result


def cell_summary(
    cells: Any,
    spots: Any,
    genes: Any,
    is3d: bool,
    *,
    scratch_dir: Path,
    chunk_rows: int,
) -> pd.DataFrame:
    """Use upstream's exact summary with its spot-list allocation replaced."""
    from pciSeq.src.core import summary

    original = summary.get_contributing_spots_2

    def collect(cells: Any, spots: Any, genes: Any, tol: float = 0.0) -> np.ndarray:
        return _contributing_spots(
            cells,
            spots,
            genes,
            tol,
            scratch_dir=scratch_dir,
            chunk_rows=chunk_rows,
        )

    # This stage runs in one isolated process. Restore the module hook even if an
    # export fails; no installed third-party source is modified.
    try:
        summary.get_contributing_spots_2 = collect
        return summary.cells_summary(cells, spots, genes, is3d)
    finally:
        summary.get_contributing_spots_2 = original


def spot_frames(spots: Any, is3d: bool, chunk_rows: int) -> Iterator[pd.DataFrame]:
    """Apply the original row-independent transcript export to bounded slices."""
    from pciSeq.src.core.summary import spots_summary

    for start in range(0, spots.nS, chunk_rows):
        stop = min(start + chunk_rows, spots.nS)
        part = SimpleNamespace(
            data=spots.data.iloc[start:stop],
            gene_id=spots.gene_id[start:stop],
            parent_cell_prob=spots.parent_cell_prob[start:stop],
            parent_cell_id=spots.parent_cell_id[start:stop],
        )
        yield spots_summary(part, is3d)


@contextmanager
def fit_batched(
    spots: pd.DataFrame,
    planes: list[Any],
    reference: pd.DataFrame,
    options: dict[str, Any],
    *,
    expected_commit: str,
    scratch_dir: Path,
    cell_batch_size: int = 128,
    spot_batch_size: int = 20_000,
) -> Iterator[tuple[pd.DataFrame, Iterator[pd.DataFrame], dict[str, Any]]]:
    """Fit one global pinned model and expose bounded transcript output frames."""
    if min(cell_batch_size, spot_batch_size) < 1:
        raise ValueError("pciSeq batch sizes must be positive")
    identity = verify_upstream(expected_commit)
    from pciSeq import app

    from merxen.pciseq_memory import CellBlockMixin
    from merxen.pciseq_spot_batches import SpotBatchMixin

    adapter_sources = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "pciseq_batched.py",
            "pciseq_memory.py",
            "pciseq_spot_batches.py",
            "pciseq_compute.py",
            "pciseq_records.py",
        )
    }

    class Model(SpotBatchMixin, CellBlockMixin, app.VarBayes):
        pass

    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=False)
    model = None
    try:
        validated_spots, coo, scdata, config = app.validate_inputs(
            spots, planes, reference, options
        )
        if config.get("elbo_per_step"):
            raise ValueError(
                "Batched pciSeq does not support the optional dense ELBO diagnostics"
            )
        cells, borders, staged_spots, label_map = app.stage_data(
            validated_spots, coo, config
        )
        if label_map and any(
            int(key) != int(value) for key, value in label_map.items()
        ):
            raise ValueError("Batched pciSeq requires the prepared dense cell labels")
        if len(staged_spots) != len(spots):
            raise ValueError("pciSeq changed the validated prepared transcript count")
        for start in range(0, len(spots), spot_batch_size):
            stop = min(start + spot_batch_size, len(spots))
            if not np.array_equal(
                staged_spots.index[start:stop].to_numpy(), np.arange(start, stop)
            ):
                raise ValueError("pciSeq changed the prepared transcript row indices")
        model = Model(cells, staged_spots, scdata, config)
        model._merxen_cell_batch_size = cell_batch_size
        model._merxen_spot_batch_size = spot_batch_size
        model._merxen_cell_work_dir = scratch_dir / "cells"
        model._merxen_spot_work_dir = scratch_dir / "spots"
        export_dir = scratch_dir / "exports"
        export_dir.mkdir()

        def collect(cells: Any, spots: Any, genes: Any, is3d: bool) -> tuple[Any, Any]:
            return (
                cell_summary(
                    cells,
                    spots,
                    genes,
                    is3d,
                    scratch_dir=export_dir,
                    chunk_rows=spot_batch_size,
                ),
                spot_frames(spots, is3d, spot_batch_size),
            )

        model._merxen_collect_data = collect
        logger.info("Starting global pciSeq model with disk-backed batched storage")
        cell_data, gene_frames = model.run()
        borders.result()
        metadata = {
            "memory_mode": "batched",
            "adapter_version": "disk_backed_exact_v1",
            "adapter_source_sha256": adapter_sources,
            "cell_batch_size": cell_batch_size,
            "spot_batch_size": spot_batch_size,
            "upstream_identity": identity,
            "has_converged": bool(model.has_converged),
            "iteration_count": len(model.iter_delta),
            "iteration_deltas": [float(value) for value in model.iter_delta],
            "scope": "one global model; no subsampling or independent tile fits",
        }
        yield cell_data, gene_frames, metadata
    finally:
        model = None
        gc.collect()
        shutil.rmtree(scratch_dir, ignore_errors=True)
