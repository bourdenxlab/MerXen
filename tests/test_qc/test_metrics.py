"""Tests for QC metric helper logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from merxen.qc.metrics import _compute_cell_metrics_from_points, _eccentricity_aspect


def test_eccentricity_aspect_returns_nan_for_none() -> None:
    """None geometry should return NaN metrics."""
    ecc, aspect = _eccentricity_aspect(None)
    assert np.isnan(ecc)
    assert np.isnan(aspect)


def test_compute_cell_metrics_from_points_with_pandas_input() -> None:
    """Per-cell metrics should aggregate transcripts and genes correctly."""
    points = pd.DataFrame(
        {
            "assignment": [1, 1, 2, 0, 2],
            "feature_name": ["A", "B", "A", "C", "A"],
        }
    )
    n_total, n_assigned, cell_metrics = _compute_cell_metrics_from_points(
        points,
        assign_col="assignment",
        gene_col="feature_name",
    )

    assert n_total == 5
    assert n_assigned == 4
    assert set(cell_metrics.columns) >= {
        "cell_id_norm",
        "transcripts_per_cell",
        "genes_per_cell",
    }

    row1 = cell_metrics[cell_metrics["cell_id_norm"] == "1"].iloc[0]
    row2 = cell_metrics[cell_metrics["cell_id_norm"] == "2"].iloc[0]
    assert int(row1["transcripts_per_cell"]) == 2
    assert int(row1["genes_per_cell"]) == 2
    assert int(row2["transcripts_per_cell"]) == 2
    assert int(row2["genes_per_cell"]) == 1
