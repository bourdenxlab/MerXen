"""Tests for transcript density overview plotting."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from merxen.visualization.density_overview import (
    density_hist2d,
    plot_density_overview,
    plot_transcript_overview,
)


def test_density_hist2d_returns_nonempty_histogram() -> None:
    """Histogram helper should return a 2D matrix for valid inputs."""
    points = pd.DataFrame({"x": [0.0, 1.0, 1.5], "y": [0.0, 1.0, 1.5]})
    hist, x_edges, y_edges = density_hist2d(points, bins=8)
    assert hist.shape == (8, 8)
    assert len(x_edges) == 9
    assert len(y_edges) == 9


def test_plot_density_overview_writes_file(tmp_path: Path) -> None:
    """Density plotting should write an output image."""
    points = pd.DataFrame({"x": [0.0, 1.0, 1.5], "y": [0.0, 1.0, 1.5]})
    out = tmp_path / "density.png"
    plot_density_overview(points, out, bins=16)
    assert out.exists()


def test_plot_transcript_overview_writes_file(tmp_path: Path) -> None:
    """Paired transcript overview plotting should write an output image."""
    merscope = SimpleNamespace(
        points={
            "transcripts": pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
        }
    )
    xenium = SimpleNamespace(
        points={
            "transcripts": pd.DataFrame({"x": [0.5, 1.5, 2.5], "y": [0.5, 1.5, 2.5]})
        }
    )
    out = tmp_path / "transcript_overview.png"

    plot_transcript_overview(
        merscope,
        xenium,
        out,
        crop_bbox_um=(0.0, 0.0, 2.0, 2.0),
        heatmap_bins=8,
    )

    assert out.exists()
