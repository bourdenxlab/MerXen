"""Tests for transcript table helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def _write_csv_for_platform(
    points: pd.DataFrame,
    csv_path: Path,
    *,
    gene_col: str,
    platform: str,
    **kwargs: object,
) -> dict[str, object]:
    masks = np.zeros((16, 16), dtype=np.uint32)
    masks[0:8, 0:8] = 3
    return write_proseg_csv_from_points(
        points,
        csv_path,
        masks,
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        x_col="x",
        y_col="y",
        z_col=None,
        gene_col=gene_col,
        control_platform=platform,
        chunk_rows=3,
        dataset_name=f"{platform}_TEST",
        status_every_chunks=100,
        memory_check_every_chunks=100,
        **kwargs,  # type: ignore[arg-type]
    )


def test_proseg_csv_excludes_all_xenium_control_types_by_feature_type(
    tmp_path: Path,
) -> None:
    """Xenium controls are dropped by ``is_gene`` and by documented names."""
    names = [
        "GFAP",
        "NegControlProbe_00002",
        "NegControlCodeword_0500",
        "UnassignedCodeword_0003",
        "DeprecatedCodeword_0001",
        "BLANK_0006",
        "Intergenic_Region_12",
        "antisense_PROKR2",
        "SNAP25",
    ]
    points = pd.DataFrame(
        {
            "x": np.arange(len(names), dtype=float),
            "y": np.ones(len(names)),
            "feature_name": pd.Categorical(names),
            "is_gene": [name in {"GFAP", "SNAP25"} for name in names],
            "qv": np.full(len(names), 30.0),
        }
    )
    csv_path = tmp_path / "xenium.csv"

    stats = _write_csv_for_platform(
        points,
        csv_path,
        gene_col="feature_name",
        platform="XENIUM",
        qv_col="qv",
        min_qv=20.0,
        is_gene_col="is_gene",
    )

    written = pd.read_csv(csv_path)
    assert written["feature_name"].tolist() == ["GFAP", "SNAP25"]
    assert written["cell_id"].tolist() == [3, 0]
    assert stats["n_excluded_controls"] == 7
    assert stats["n_excluded_genes"] == 7
    assert set(stats["excluded_control_counts"]) == set(names) - {"GFAP", "SNAP25"}


def test_proseg_csv_warns_on_xenium_controls_known_only_by_category(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A control the name registry misses is dropped and reported."""
    points = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "feature_name": ["GFAP", "NovelCtrl_0001", "Intergenic_Region_3"],
            "codeword_category": [
                "predesigned_gene",
                "novel_control_kind",
                "genomic_control_probe",
            ],
        }
    )
    csv_path = tmp_path / "xenium.csv"

    with caplog.at_level(logging.WARNING, logger="merxen.io.transcript_io"):
        stats = _write_csv_for_platform(
            points,
            csv_path,
            gene_col="feature_name",
            platform="XENIUM",
            codeword_category_col="codeword_category",
        )

    assert pd.read_csv(csv_path)["feature_name"].tolist() == ["GFAP"]
    assert stats["excluded_control_counts"] == {
        "NovelCtrl_0001": 1,
        "Intergenic_Region_3": 1,
    }
    assert "NovelCtrl_0001" in caplog.text
    assert "Intergenic_Region_3" not in caplog.text


def test_proseg_csv_excludes_merscope_blank_codewords(tmp_path: Path) -> None:
    """MERSCOPE ``Blank-<N>`` codewords are dropped; hyphenated genes stay."""
    genes = ["Gad1", "Blank-0", "Nkx6-1", "Blank-17", "HLA-DMB", "Blank-0"]
    points = pd.DataFrame(
        {
            "x": np.arange(len(genes), dtype=float),
            "y": np.arange(len(genes), dtype=float),
            "gene": genes,
            "transcript_score": np.linspace(0.5, 1.0, len(genes)),
        }
    )
    csv_path = tmp_path / "merscope.csv"

    stats = _write_csv_for_platform(
        points,
        csv_path,
        gene_col="gene",
        platform="MERSCOPE",
        qv_col="transcript_score",
    )

    written = pd.read_csv(csv_path)
    assert written["feature_name"].tolist() == ["Gad1", "Nkx6-1", "HLA-DMB"]
    assert written["transcript_id"].tolist() == [0, 1, 2]
    assert stats["n_excluded_controls"] == 3
    assert stats["excluded_control_counts"] == {"Blank-0": 2, "Blank-17": 1}


def test_proseg_csv_keeps_controls_without_control_platform(tmp_path: Path) -> None:
    """Direct callers that pass no platform keep every feature."""
    points = pd.DataFrame(
        {"x": [1.0, 2.0], "y": [1.0, 2.0], "gene": ["Gad1", "Blank-0"]}
    )
    csv_path = tmp_path / "plain.csv"

    stats = write_proseg_csv_from_points(
        points,
        csv_path,
        np.ones((4, 4), dtype=np.uint32),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        x_col="x",
        y_col="y",
        z_col=None,
        gene_col="gene",
    )

    assert pd.read_csv(csv_path)["feature_name"].tolist() == ["Gad1", "Blank-0"]
    assert stats["n_excluded_controls"] == 0
