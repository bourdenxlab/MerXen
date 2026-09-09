"""Tests for canonical MERSCOPE transcript input."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from spatialdata.transformations import Identity

from merxen.io.builders.merscope import _get_parquet_points


def test_merscope_parquet_points_load_lazily_and_normalize_schema(
    tmp_path: Path,
) -> None:
    """Parquet transcripts should stay lazy and normalize export variants."""
    transcript_path = tmp_path / "detected_transcripts.parquet"
    pd.DataFrame(
        {
            "": [17],
            "global_x": [3.5],
            "global_y": [4.5],
            "global_z": [3.0],
            "feature_name": ["GeneA"],
            "cell_id": [101],
        }
    ).to_parquet(transcript_path)

    points = _get_parquet_points(transcript_path, {"global": Identity()})
    computed = points.compute()

    assert points.npartitions >= 1
    assert "" not in computed.columns
    assert list(computed["gene"].astype(str)) == ["GeneA"]
    assert list(computed["cell_id"].astype(str)) == ["101"]
