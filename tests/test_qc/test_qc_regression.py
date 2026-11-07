"""XENIUM/MERSCOPE QC output is unchanged by the ATERA work (AT-6).

``compute_dataset_qc`` runs on a fixed, Xenium-like and MERSCOPE-like latest
store (dask points, a cell table, platform shapes) and is compared with a
stored result. The golden was generated with the code before AT-6 (the
aggregates fast paths, window pruning and the dask re-raise); those changes
apply only to stores whose build recorded transcript aggregates, so the
result must stay the same. The exact (float64/int64) table totals later made
the default for every platform agree with the float32 sums far below 2**24
counts, so they leave this fixture unchanged too.

Regenerate after an intended change with::

    MERXEN_UPDATE_GOLDEN=1 pytest tests/test_qc/test_qc_regression.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
import spatialdata as sd
from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel

from merxen.qc.gene_comparison import gene_totals_from_points
from merxen.qc.metrics import compute_dataset_qc

_GOLDEN = Path(__file__).parent / "golden" / "qc_regression.json"
_UPDATE = os.environ.get("MERXEN_UPDATE_GOLDEN") == "1"

_SEGMENTATION_SHAPES = "MOSAIK_proseg"
_SEGMENTATION_TABLE = "table_MOSAIK_proseg"


def _store(platform: str, path: Path) -> Path:
    """Write a deterministic latest store of one platform and return its path."""
    rng = np.random.default_rng(11 if platform == "XENIUM" else 12)
    n_cells, per_cell, side = 600, 30, 600.0
    centres = rng.uniform(0.0, side, size=(n_cells, 2))
    cell_tx = np.repeat(centres, per_cell, axis=0) + rng.normal(
        0.0, 1.5, size=(n_cells * per_cell, 2)
    )
    background = rng.uniform(0.0, side, size=(int(0.05 * side**2), 2))
    xy = np.vstack([cell_tx, background])
    n_tx = len(xy)
    genes = rng.choice(["GeneA", "GeneB", "GeneC", "GeneD"], size=n_tx).astype(object)
    control = "NegControlProbe_00042" if platform == "XENIUM" else "Blank-7"
    genes[rng.random(n_tx) < 0.02] = control
    assignment = np.concatenate(
        [np.repeat(np.arange(1, n_cells + 1), per_cell), np.zeros(len(background))]
    ).astype(np.int64)
    points = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "z": rng.uniform(0.0, 10.0, n_tx),
            "feature_name": pd.Categorical(genes),
            "assignment": assignment,
        }
    )
    platform_cells = gpd.GeoDataFrame(
        {
            "geometry": shapely.buffer(
                shapely.points(centres + rng.normal(0.0, 0.4, centres.shape)),
                4.0,
                quad_segs=4,
            )
        }
    )
    segmented = gpd.GeoDataFrame(
        {
            "geometry": shapely.buffer(
                shapely.points(centres + np.array([1.0, -0.5])), 5.0, quad_segs=6
            )
        },
        index=pd.RangeIndex(1, n_cells + 1),
    )
    counts = rng.poisson(6.0, size=(n_cells, 5)).astype(np.float32)
    table = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "cell_id": np.arange(1, n_cells + 1),
                "region": pd.Categorical([_SEGMENTATION_SHAPES] * n_cells),
            },
            index=[f"c{i}" for i in range(1, n_cells + 1)],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC", "GeneD", control]),
    )
    reference_key = (
        "xenium_cell_boundaries" if platform == "XENIUM" else "merscope_cell_boundaries"
    )
    SpatialData(
        points={
            "transcripts": PointsModel.parse(
                points,
                coordinates={"x": "x", "y": "y", "z": "z"},
                feature_key="feature_name",
                chunksize=4000,
            )
        },
        shapes={
            reference_key: ShapesModel.parse(platform_cells),
            _SEGMENTATION_SHAPES: ShapesModel.parse(segmented),
        },
        tables={
            _SEGMENTATION_TABLE: TableModel.parse(
                table,
                region=_SEGMENTATION_SHAPES,
                region_key="region",
                instance_key="cell_id",
            )
        },
    ).write(path)
    return path


def _plain(value: Any) -> Any:
    """Return a JSON-safe copy (NaN as the string ``"nan"``)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _frame_digest(frame: pd.DataFrame) -> dict[str, Any]:
    """Columns, row count, first rows and per-column sum/min/max of a table."""
    numeric = frame.select_dtypes("number")
    return {
        "columns": [str(column) for column in frame.columns],
        "n_rows": len(frame),
        "head": frame.head(5).to_dict(orient="list"),
        "sum": numeric.sum().to_dict(),
        "min": numeric.min().to_dict(),
        "max": numeric.max().to_dict(),
    }


def _qc_record(platform: str, tmp_path: Path) -> dict[str, Any]:
    zarr_path = _store(platform, tmp_path / f"{platform.lower()}_latest.zarr")
    reference_key = (
        "xenium_cell_boundaries" if platform == "XENIUM" else "merscope_cell_boundaries"
    )
    record: dict[str, Any] = {}
    for label, table_key in (("table", _SEGMENTATION_TABLE), ("points", None)):
        result = compute_dataset_qc(
            zarr_path,
            f"S1_{platform}",
            table_key=table_key,
            shape_key=_SEGMENTATION_SHAPES,
            registration_reference_shape_key=reference_key,
        )
        summary = dict(result["summary"])
        summary.pop("latest_zarr_path")
        record[label] = {
            "summary": summary,
            "geometry_metrics": _frame_digest(result["geometry_metrics"]),
            "cell_metrics": _frame_digest(
                result["cell_metrics"]
                .sort_values("cell_id_norm")
                .reset_index(drop=True)
            ),
            "registration_qc": result["registration_qc"],
        }
    totals = gene_totals_from_points(sd.read_zarr(zarr_path), assigned_only=False)
    record["gene_totals"] = totals.sort_index().to_dict()
    return _plain(record)


def _assert_same(actual: Any, expected: Any, where: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert sorted(actual) == sorted(expected), where
        for key in expected:
            _assert_same(actual[key], expected[key], f"{where}/{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_same(left, right, f"{where}[{index}]")
    elif isinstance(expected, float) and isinstance(actual, float | int):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), where
    else:
        assert actual == expected, where


@pytest.mark.parametrize("platform", ["XENIUM", "MERSCOPE"])
def test_qc_output_matches_the_stored_result(platform: str, tmp_path: Path) -> None:
    """QC of a XENIUM/MERSCOPE latest store equals the pre-AT-6 result."""
    record = _qc_record(platform, tmp_path)
    if _UPDATE:
        stored = json.loads(_GOLDEN.read_text()) if _GOLDEN.is_file() else {}
        stored[platform] = record
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n")
    expected = json.loads(_GOLDEN.read_text())[platform]
    _assert_same(json.loads(json.dumps(record)), expected, platform)
