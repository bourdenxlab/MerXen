"""Tests for gene scatter plotting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from merxen.visualization.gene_scatter import plot_gene_scatter


def test_plot_gene_scatter_writes_file(tmp_path: Path) -> None:
    """Scatter plotting should write an output PNG."""
    df = pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "xenium": [1.0, 2.0, 4.0],
            "merscope": [1.2, 1.9, 4.1],
        }
    )
    out = tmp_path / "scatter.png"
    plot_gene_scatter(df, out, title="Test Plot")
    assert out.exists()
