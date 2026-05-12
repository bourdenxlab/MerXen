"""Runtime compatibility patches for MapMyCells GPU execution."""

from __future__ import annotations

from typing import Any, Self

import numpy as np


class HostMemoryCollator:
    """MapMyCells GPU collator that leaves batch preparation on host memory."""

    _merxen_gpu_compat = True

    def __init__(
        self: Self,
        all_query_identifiers: list[str],
        normalization: str,
        all_query_markers: list[str],
        device: Any,
        log: Any | None = None,
    ) -> None:
        self.all_query_identifiers = all_query_identifiers
        self.normalization = normalization
        self.all_query_markers = all_query_markers
        self.device = device
        self.log = log

    def __call__(self: Self, batch: list[tuple[Any, int, int]]) -> tuple[Any, int, int]:
        from cell_type_mapper.cell_by_gene.cell_by_gene import CellByGeneMatrix

        data = np.concatenate([row[0] for row in batch])
        r0 = batch[0][1]
        r1 = batch[-1][-1]

        cell_by_gene = CellByGeneMatrix(
            data=data,
            gene_identifiers=self.all_query_identifiers,
            normalization=self.normalization,
            log=self.log,
        )

        if cell_by_gene.normalization != "log2CPM":
            cell_by_gene.to_log2CPM_in_place()

        cell_by_gene.downsample_genes_in_place(self.all_query_markers)
        return cell_by_gene, r0, r1


def apply_mapmycells_gpu_compat_patch() -> bool:
    """Keep MapMyCells DataLoader batches on host memory before GPU compute.

    MapMyCells 1.5.5 moves query batches to CUDA inside DataLoader workers and
    then constructs ``CellByGeneMatrix`` objects. That constructor still performs
    NumPy validation, so CUDA tensors fail before the heavy correlation code can
    run. Keeping the loader output as host arrays lets MapMyCells' existing
    distance functions move bootstrap query/reference arrays to CUDA where the
    GPU work actually happens.

    Returns:
        ``True`` if the patch was applied, ``False`` if it was already active.
    """
    from cell_type_mapper.gpu_utils.anndata_iterator import (
        anndata_iterator as gpu_iterator,
    )

    if gpu_iterator.Collator is HostMemoryCollator:
        return False

    gpu_iterator.Collator = HostMemoryCollator
    return True
