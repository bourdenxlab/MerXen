"""Tests for the shared whole-transcriptome panel-width rule (AT-8, Q22)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from merxen.analysis.clustering_squidpy import WHOLE_TRANSCRIPTOME_HVG_N_TOP_GENES
from merxen.gene_sets import WHOLE_TRANSCRIPTOME_MIN_GENES, is_whole_transcriptome

_NEXTFLOW_CONFIG = Path(__file__).resolve().parents[1] / "workflows" / "nextflow.config"


@pytest.mark.parametrize(
    ("n_genes", "expected"),
    [
        (0, False),
        (300, False),  # XENIUM / MERSCOPE custom panels
        (5_000, False),
        (5_001, True),  # Xenium Prime 5K
        (5_006, True),
        (18_049, True),  # ATERA
    ],
)
def test_more_than_5000_genes_is_whole_transcriptome(
    n_genes: int, expected: bool
) -> None:
    assert WHOLE_TRANSCRIPTOME_MIN_GENES == 5_000
    assert is_whole_transcriptome(n_genes) is expected


def test_negative_gene_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        is_whole_transcriptome(-1)


def test_automatic_hvg_count_matches_the_atera_workflow_default() -> None:
    """Wide panels without a request get the same HVG count as ATERA rows."""
    match = re.search(
        r"^\s*atera_clustering_hvg_n_top_genes\s*=\s*(\d+)\s*$",
        _NEXTFLOW_CONFIG.read_text(),
        re.MULTILINE,
    )
    assert match is not None
    assert int(match.group(1)) == WHOLE_TRANSCRIPTOME_HVG_N_TOP_GENES
