"""Tests for QC plot utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from merxen.visualization.qc_plots import (
    plot_assignment_bar,
    plot_cell_metrics_violin,
    plot_cell_metrics_violin_comparison,
    plot_geometry_histograms,
    plot_geometry_histograms_comparison,
)


def test_qc_plot_functions_write_files(tmp_path: Path) -> None:
    """QC plotting helpers should emit image files."""
    geometry = pd.DataFrame(
        {
            "area": [10.0, 12.0, 9.0],
            "eccentricity": [0.2, 0.3, 0.4],
            "aspect_ratio": [1.1, 1.3, 1.2],
        }
    )
    cell = pd.DataFrame(
        {"transcripts_per_cell": [10, 20, 15], "genes_per_cell": [5, 6, 5]}
    )
    assignment = pd.DataFrame(
        {"dataset": ["XENIUM", "MERSCOPE"], "pct_assigned": [75.0, 81.0]}
    )

    geom_out = tmp_path / "geom.png"
    violin_out = tmp_path / "violin.png"
    bar_out = tmp_path / "bar.png"

    plot_geometry_histograms(geometry, geom_out)
    plot_cell_metrics_violin(cell, violin_out)
    plot_assignment_bar(assignment, bar_out)

    assert geom_out.exists()
    assert violin_out.exists()
    assert bar_out.exists()


def test_combined_qc_plot_functions_write_files(tmp_path: Path) -> None:
    """Combined QC plotting helpers should emit image files."""
    geometry_by_dataset = {
        "XENIUM": pd.DataFrame(
            {
                "area": [10.0, 12.0, 9.0],
                "eccentricity": [0.2, 0.3, 0.4],
                "aspect_ratio": [1.1, 1.3, 1.2],
            }
        ),
        "MERSCOPE": pd.DataFrame(
            {
                "area": [14.0, 13.0, 16.0],
                "eccentricity": [0.1, 0.2, 0.2],
                "aspect_ratio": [1.0, 1.2, 1.1],
            }
        ),
    }
    cell_by_dataset = {
        "XENIUM": pd.DataFrame(
            {"transcripts_per_cell": [10, 20, 15], "genes_per_cell": [5, 6, 5]}
        ),
        "MERSCOPE": pd.DataFrame(
            {"transcripts_per_cell": [30, 25, 35], "genes_per_cell": [8, 9, 7]}
        ),
    }

    geom_out = tmp_path / "combined_geom.png"
    violin_out = tmp_path / "combined_violin.png"

    plot_geometry_histograms_comparison(geometry_by_dataset, geom_out)
    plot_cell_metrics_violin_comparison(cell_by_dataset, violin_out)

    assert geom_out.exists()
    assert violin_out.exists()
