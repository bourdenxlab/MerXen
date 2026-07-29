"""Selected downstream analysis-layer validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from shapely.geometry import box

from merxen.analysis_layers import validate_analysis_layer
from merxen.cli import main
from merxen.io.spatialdata_schema import SpatialDataContractError


def _hybrid_sdata() -> SimpleNamespace:
    table = ad.AnnData(
        X=np.ones((2, 1), dtype=np.int64),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray([1, 2], dtype=np.uint64),
                "region": pd.Categorical(
                    ["MOSAIK_proseg_hybrid"] * 2,
                ),
            },
            index=["1", "2"],
        ),
        var=pd.DataFrame(index=["GeneA"]),
    )
    table.uns["spatialdata_attrs"] = {"instance_key": "instance_id"}
    shapes = gpd.GeoDataFrame(
        {
            "instance_id": np.asarray([1, 2], dtype=np.uint64),
            "geometry": [box(0, 0, 1, 1), box(2, 0, 3, 1)],
        },
        index=pd.Index([1, 2], name="instance_id"),
    )
    points = pd.DataFrame(
        {
            "hybrid_assignment": pd.Series([1, pd.NA], dtype="UInt64"),
            "hybrid_background": [False, True],
            "hybrid_assignment_source": ["single_mask", "outside"],
        }
    )
    return SimpleNamespace(
        tables={"table_MOSAIK_proseg_hybrid": table},
        shapes={"MOSAIK_proseg_hybrid": shapes},
        points={"transcripts": points},
        attrs={
            "merxen_schema": {
                "segmentations": {
                    "proseg_hybrid": {
                        "points": "transcripts",
                        "assignment_column": "hybrid_assignment",
                        "background_column": "hybrid_background",
                        "assignment_source_column": "hybrid_assignment_source",
                        "shape": "MOSAIK_proseg_hybrid",
                        "table": "table_MOSAIK_proseg_hybrid",
                    }
                }
            }
        },
    )


def test_validate_analysis_layer_checks_complete_hybrid_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = tmp_path / "latest.zarr"
    store.mkdir()
    monkeypatch.setattr(
        "merxen.analysis_layers.sd.read_zarr",
        lambda _: _hybrid_sdata(),
    )

    summary = validate_analysis_layer(
        store,
        platform="MERSCOPE",
        segmentation="proseg_hybrid",
        table_key="table_MOSAIK_proseg_hybrid",
        shape_key="MOSAIK_proseg_hybrid",
    )

    assert summary["n_cells"] == 2
    assert summary["segmentation"] == "proseg_hybrid"


def test_validate_analysis_layer_rejects_missing_hybrid_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = tmp_path / "latest.zarr"
    store.mkdir()
    sdata = _hybrid_sdata()
    sdata.points["transcripts"] = sdata.points["transcripts"].drop(
        columns="hybrid_assignment_source"
    )
    monkeypatch.setattr(
        "merxen.analysis_layers.sd.read_zarr",
        lambda _: sdata,
    )

    with pytest.raises(
        SpatialDataContractError,
        match="missing required assignment columns",
    ):
        validate_analysis_layer(
            store,
            platform="MERSCOPE",
            segmentation="proseg_hybrid",
            table_key="table_MOSAIK_proseg_hybrid",
            shape_key="MOSAIK_proseg_hybrid",
        )


def test_validate_analysis_layer_cli_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The registered Click command should validate and persist its JSON summary."""
    store = tmp_path / "latest.zarr"
    store.mkdir()
    output_path = tmp_path / "validation" / "summary.json"
    captured: dict[str, object] = {}

    def _validate(path: Path, **kwargs: object) -> dict[str, object]:
        captured["path"] = path
        captured.update(kwargs)
        return {
            "platform": kwargs["platform"],
            "segmentation": kwargs["segmentation"],
            "n_cells": 2,
        }

    monkeypatch.setattr(
        "merxen.cli.run_analysis_layer_validation.validate_analysis_layer",
        _validate,
    )

    result = CliRunner().invoke(
        main,
        [
            "validate-analysis-layer",
            "--zarr",
            str(store),
            "--platform",
            "MERSCOPE",
            "--segmentation",
            "proseg_hybrid",
            "--table-key",
            "table_MOSAIK_proseg_hybrid",
            "--shape-key",
            "MOSAIK_proseg_hybrid",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Validated MERSCOPE:proseg_hybrid (2 cells)" in result.output
    assert captured == {
        "path": store,
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
        "table_key": "table_MOSAIK_proseg_hybrid",
        "shape_key": "MOSAIK_proseg_hybrid",
    }
    assert json.loads(output_path.read_text()) == {
        "n_cells": 2,
        "platform": "MERSCOPE",
        "segmentation": "proseg_hybrid",
    }
