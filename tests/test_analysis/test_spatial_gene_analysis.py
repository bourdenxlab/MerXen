"""Tests for per-gene spatial autocorrelation helpers."""

from __future__ import annotations

import pandas as pd

from merxen.analysis.spatial_gene_analysis import ranked_spatial_autocorr_genes


def test_ranked_spatial_autocorr_genes_returns_metric_extremes() -> None:
    """Ranking output should expose top and bottom genes for both metrics."""
    metrics = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D"],
            "moran_i": [0.4, -0.1, 0.9, 0.2],
            "moran_i_pval_norm": [0.1, 0.2, 0.01, 0.5],
            "moran_i_pval_fdr_bh": [0.2, 0.3, 0.02, 0.6],
            "geary_c": [0.7, 1.3, 0.2, 0.9],
            "geary_c_pval_norm": [0.1, 0.4, 0.01, 0.5],
            "geary_c_pval_fdr_bh": [0.2, 0.5, 0.02, 0.6],
        }
    )

    ranked = ranked_spatial_autocorr_genes(metrics, top_n=2)

    observed = {
        key: list(group.gene)
        for key, group in ranked.groupby(["metric", "direction"], sort=False)
    }
    assert observed[("moran_i", "top")] == ["C", "A"]
    assert observed[("moran_i", "bottom")] == ["B", "D"]
    assert observed[("geary_c", "top")] == ["B", "D"]
    assert observed[("geary_c", "bottom")] == ["C", "A"]
