"""Panel-width rules shared by the analysis stages.

A panel with more than :data:`WHOLE_TRANSCRIPTOME_MIN_GENES` genes is treated
as whole-transcriptome, whatever its platform (Atera plan §8 AT-8, Q22). The
Xenium Prime 5K panels (5,001 and 5,006 genes) count, as does ATERA. The
300-gene XENIUM and MERSCOPE panels never do, so the rules that key on this
constant leave them unchanged:

- legacy clustering runs the PCA on highly variable genes when the sample sets
  no ``hvg_n_top_genes``;
- pciSeq refuses an atlas reference that was not restricted to an explicit
  gene list.

Gene counts exclude control features (blank, negative-control and similar
codewords).
"""

from __future__ import annotations

WHOLE_TRANSCRIPTOME_MIN_GENES = 5_000


def is_whole_transcriptome(n_genes: int) -> bool:
    """Return whether a panel of ``n_genes`` genes is whole-transcriptome.

    Args:
        n_genes: Number of non-control genes in the panel.

    Returns:
        ``True`` when the panel has more than
        :data:`WHOLE_TRANSCRIPTOME_MIN_GENES` genes.

    Raises:
        ValueError: ``n_genes`` is negative.
    """
    count = int(n_genes)
    if count < 0:
        raise ValueError(f"n_genes must be non-negative; got {count}")
    return count > WHOLE_TRANSCRIPTOME_MIN_GENES
