"""Hybrid segmentation QC comparison tests."""

from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from merxen.qc.hybrid import compute_hybrid_qc


def _table(values: list[list[int]], ids: list[int]) -> ad.AnnData:
    table = ad.AnnData(
        X=np.asarray(values, dtype=np.int64),
        obs=pd.DataFrame(
            {"instance_id": np.asarray(ids, dtype=np.uint64)},
            index=list(map(str, ids)),
        ),
        var=pd.DataFrame(
            {"gene": ["GeneA", "GeneB"]},
            index=["GeneA", "GeneB"],
        ),
    )
    table.uns["spatialdata_attrs"] = {"instance_key": "instance_id"}
    return table


def test_compute_hybrid_qc_compares_cells_genes_and_provenance() -> None:
    shapes = gpd.GeoDataFrame(
        {
            "instance_id": np.asarray([1, 2], dtype=np.uint64),
            "cellpose_area_um2": [10.0, 20.0],
            "hybrid_area_um2": [12.0, 20.0],
            "cap_rejected_external": [0, 3],
            "unsupported_external": [1, 2],
            "fallback_reason": ["", "low_transcript_count"],
            "geometry": [box(0, 0, 2, 2), box(3, 0, 5, 2)],
        },
        index=pd.Index([1, 2], name="instance_id"),
    )
    points = pd.DataFrame(
        {
            "hybrid_assignment_source": [
                "single_mask",
                "proseg_overlap",
                "ambiguous_overlap",
                "outside",
            ]
        }
    )
    sdata = SimpleNamespace(
        shapes={"MOSAIK_proseg_hybrid": shapes},
        tables={
            "table_MOSAIK_proseg_hybrid": _table([[3, 1], [1, 2]], [1, 2]),
            "table_MOSAIK_cellpose": _table([[2, 1], [1, 1]], [1, 2]),
            "table_MOSAIK_proseg": _table([[4, 0], [0, 4]], [1, 2]),
        },
        points={"transcripts": points},
    )

    result = compute_hybrid_qc(
        sdata,
        dataset_name="P1_MERSCOPE",
        points_key="transcripts",
    )

    cells = result["cell_diagnostics"].set_index("cell_id")
    assert cells.loc["1", "hybrid_minus_cellpose_transcripts"] == 1
    assert cells.loc["2", "hybrid_minus_proseg_transcripts"] == -1
    assert cells.loc["1", "hybrid_area_growth_fraction"] == 0.2
    assert result["summary"]["n_fallback_cellpose"] == 1
    assert result["summary"]["pct_ambiguous_overlap"] == 25.0
    assert result["fallback_reasons"].to_dict("records") == [
        {
            "fallback_reason": "low_transcript_count",
            "count": 1,
            "percent_of_hybrid_cells": 50.0,
        }
    ]

    genes = result["gene_count_changes"].set_index("gene")
    assert genes.loc["GeneA", "hybrid_minus_cellpose"] == 1
    assert genes.loc["GeneB", "hybrid_minus_proseg"] == -1
