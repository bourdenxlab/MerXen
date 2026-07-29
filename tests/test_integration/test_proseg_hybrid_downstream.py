"""Integration coverage for the selected ProSeg-hybrid downstream branch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from merxen.analysis_layers import validate_analysis_layer
from merxen.mask_image_quantification import (
    HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY,
    join_cellpose_image_quantification_to_hybrid,
)
from merxen.qc.metrics import compute_dataset_qc, save_dataset_qc
from merxen.visualization.sanity_plots import _crop_points


def _table(
    matrix: list[list[float]],
    ids: list[int],
    *,
    region: str,
    variables: list[str],
) -> ad.AnnData:
    table = ad.AnnData(
        X=np.asarray(matrix),
        obs=pd.DataFrame(
            {
                "instance_id": np.asarray(ids, dtype=np.uint64),
                "region": pd.Categorical([region] * len(ids)),
            },
            index=list(map(str, ids)),
        ),
        var=pd.DataFrame(
            {"gene": variables},
            index=variables,
        ),
    )
    table.uns["spatialdata_attrs"] = {
        "instance_key": "instance_id",
        "region_key": "region",
        "region": region,
    }
    return table


def _hybrid_sdata() -> SimpleNamespace:
    hybrid = _table(
        [[3, 1], [1, 2]],
        [1, 2],
        region="MOSAIK_proseg_hybrid",
        variables=["GeneA", "GeneB"],
    )
    image = _table(
        [[100, 120], [200, 250]],
        [2, 1],
        region="MOSAIK_cellpose",
        variables=["img__DAPI__mean", "img__DAPI__max"],
    )
    image.obs["mask_pixel_count"] = [6, 8]
    hybrid = join_cellpose_image_quantification_to_hybrid(image, hybrid).table
    shapes = gpd.GeoDataFrame(
        {
            "instance_id": np.asarray([1, 2], dtype=np.uint64),
            "cellpose_area_um2": [10.0, 20.0],
            "hybrid_area_um2": [12.0, 21.0],
            "cap_rejected_external": [0, 1],
            "unsupported_external": [0, 2],
            "fallback_reason": ["", ""],
            "geometry": [box(0, 0, 2, 2), box(3, 0, 5, 2)],
        },
        index=pd.Index([1, 2], name="instance_id"),
    )
    points = pd.DataFrame(
        {
            "x": [0.5, 1.0, 3.5, 4.0],
            "y": [0.5, 1.0, 0.5, 1.0],
            "gene": ["GeneA", "GeneB", "GeneA", "GeneB"],
            "hybrid_assignment": pd.Series(
                [1, pd.NA, 2, pd.NA],
                dtype="UInt64",
            ),
            "hybrid_background": [False, True, False, True],
            "hybrid_assignment_source": [
                "single_mask",
                "ambiguous_overlap",
                "proseg_overlap",
                "outside",
            ],
            "transcript_id": np.asarray([1, 2, 3, 4], dtype=np.uint64),
        }
    )
    return SimpleNamespace(
        shapes={"MOSAIK_proseg_hybrid": shapes},
        tables={
            "table_MOSAIK_proseg_hybrid": hybrid,
            "table_MOSAIK_cellpose": _table(
                [[2, 1], [1, 1]],
                [1, 2],
                region="MOSAIK_cellpose",
                variables=["GeneA", "GeneB"],
            ),
            "table_MOSAIK_proseg": _table(
                [[4, 0], [0, 4]],
                [1, 2],
                region="MOSAIK_proseg",
                variables=["GeneA", "GeneB"],
            ),
        },
        points={"transcripts": points},
        attrs={
            "merxen_schema": {
                "primary_points": "transcripts",
                "segmentations": {
                    "proseg_hybrid": {
                        "points": "transcripts",
                        "assignment_column": "hybrid_assignment",
                        "background_column": "hybrid_background",
                        "assignment_source_column": "hybrid_assignment_source",
                        "shape": "MOSAIK_proseg_hybrid",
                        "table": "table_MOSAIK_proseg_hybrid",
                    }
                },
            }
        },
    )


def test_hybrid_branch_validates_quantifies_reports_and_visualizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sdata = _hybrid_sdata()
    store = tmp_path / "latest.zarr"
    store.mkdir()
    monkeypatch.setattr(
        "merxen.analysis_layers.sd.read_zarr",
        lambda _: sdata,
    )
    monkeypatch.setattr(
        "merxen.qc.metrics.sd.read_zarr",
        lambda _: sdata,
    )

    validation = validate_analysis_layer(
        store,
        platform="MERSCOPE",
        segmentation="proseg_hybrid",
        table_key="table_MOSAIK_proseg_hybrid",
        shape_key="MOSAIK_proseg_hybrid",
    )
    qc = compute_dataset_qc(
        store,
        "P1_MERSCOPE",
        table_key="table_MOSAIK_proseg_hybrid",
        shape_key="MOSAIK_proseg_hybrid",
    )
    paths = save_dataset_qc(qc, tmp_path / "qc", "P1_MERSCOPE")
    crop, _, _ = _crop_points(
        sdata,
        (0.0, 0.0, 5.0, 2.0),
        max_points=None,
        random_state=0,
        assignment_shape_key="MOSAIK_proseg_hybrid",
        prefer_aligned_points=False,
        prefer_aligned_assignment=False,
    )

    assert validation["n_cells"] == 2
    assert (
        HYBRID_IMAGE_QUANTIFICATION_OBSM_KEY
        in sdata.tables["table_MOSAIK_proseg_hybrid"].obsm
    )
    assert qc["summary"]["pct_ambiguous_overlap"] == 25.0
    assert paths["hybrid_cell_diagnostics"].exists()
    assert paths["hybrid_fallback_reasons"].exists()
    assert paths["hybrid_area_growth_map"].exists()
    assert crop["assigned"].tolist() == [True, False, True, False]
