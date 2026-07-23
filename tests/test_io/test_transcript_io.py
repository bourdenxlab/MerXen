"""Tests for transcript table helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from merxen.io.transcript_io import (
    assignment_mask,
    assignment_mask_from_points,
    write_proseg_csv_from_points,
)


def test_assignment_mask_treats_nullable_zero_as_assigned() -> None:
    """Nullable numeric assignment uses null, not zero, as unassigned."""
    series = pd.Series([0, 1, pd.NA], dtype="UInt32")

    mask = assignment_mask(series)

    assert mask.tolist() == [True, True, False]


def test_assignment_mask_keeps_legacy_numeric_zero_unassigned() -> None:
    """Dense numeric assignment columns still use zero as the unassigned code."""
    series = pd.Series([0, 1, 2], dtype="uint32")

    mask = assignment_mask(series)

    assert mask.tolist() == [False, True, True]


def test_assignment_mask_from_points_prefers_background_column() -> None:
    """ProSeg foreground status should come from ``background`` when present."""
    points = pd.DataFrame(
        {
            "assignment": pd.Series([0, pd.NA, 2], dtype="UInt32"),
            "background": [False, True, False],
        }
    )

    mask = assignment_mask_from_points(points, assign_col="assignment")

    assert mask.tolist() == [True, False, True]


def test_proseg_csv_retains_quality_and_excludes_xenium_controls(
    tmp_path: Path,
) -> None:
    """CSV preparation should retain QV values and omit negative controls."""
    points = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "gene": ["GeneA", "NegControlProbe", "GeneB", "GeneC"],
            "qv": [25.0, 30.0, 19.0, 40.0],
        }
    )
    masks = np.zeros((8, 8), dtype=np.uint32)
    masks[0:6, 0:6] = 4
    csv_path = tmp_path / "transcripts.csv"

    stats = write_proseg_csv_from_points(
        points,
        csv_path,
        masks,
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        x_col="x",
        y_col="y",
        z_col="z",
        gene_col="gene",
        qv_col="qv",
        min_qv=20.0,
        excluded_gene_pattern=r"^(Deprecated|NegControl|Unassigned|Intergenic)",
        chunk_rows=2,
        dataset_name="XENIUM_TEST",
        status_every_chunks=1,
        memory_check_every_chunks=100,
    )

    written = pd.read_csv(csv_path)
    assert written["feature_name"].tolist() == ["GeneA", "GeneC"]
    assert written["qv"].tolist() == [25.0, 40.0]
    assert written["transcript_id"].tolist() == [0, 1]
    assert stats["n_excluded_genes"] == 1
