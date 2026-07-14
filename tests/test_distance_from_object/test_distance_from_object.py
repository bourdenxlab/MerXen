"""Geometry and pseudobulk tests for distance-from-object analysis."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import Polygon

from merxen.config import DistanceFromObjectCohortConfig, DistanceFromObjectConfig
from merxen.cortical_depth.assign_cells import CellCoordinateTable
from merxen.distance_from_object.annotations import (
    ObjectAnnotation,
    load_object_annotations,
)
from merxen.distance_from_object.distances import (
    assign_distances_to_objects,
    label_object_proximity,
)
from merxen.distance_from_object.pipeline import (
    run_distance_from_object,
    run_distance_from_object_cohort,
)
from merxen.distance_from_object.pseudobulk import (
    build_pair_pseudobulk,
    combine_pair_pseudobulks,
    retain_complete_pairs,
    run_paired_differential_expression,
)


def test_object_annotation_loading_preserves_types_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "objects.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "object_id": "plaque_1",
                            "object_type": "Amyloid plaques",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
                            ],
                        },
                    }
                ],
            }
        )
    )

    annotations = load_object_annotations(path)

    assert len(annotations) == 1
    assert annotations[0].object_id == "plaque_1"
    assert annotations[0].object_type == "Amyloid plaques"
    assert annotations[0].geometry.area == 100


def test_distance_assignment_uses_centroid_to_edge_and_inside_is_near() -> None:
    annotations = [
        _annotation(
            "plaque_1",
            "Amyloid plaques",
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        )
    ]
    coordinates = CellCoordinateTable(
        cell_ids=pd.Index(["inside", "near", "middle", "far", "beyond"]),
        coordinates=np.asarray(
            [[5, 5], [20, 5], [70, 5], [160, 5], [250, 5]], dtype=float
        ),
        source="test",
    )

    assigned = assign_distances_to_objects(coordinates, annotations)

    assert assigned.loc["inside", "distance_to_object_edge_um"] == 5
    assert assigned.loc["inside", "signed_distance_to_object_edge_um"] == -5
    assert bool(assigned.loc["inside", "inside_object"])
    assert assigned.loc["inside", "object_proximity"] == "near"
    assert assigned.loc["near", "object_proximity"] == "near"
    assert assigned.loc["middle", "object_proximity"] == "middle"
    assert assigned.loc["far", "object_proximity"] == "far"
    assert assigned.loc["beyond", "object_proximity"] == "beyond_max"


def test_distance_assignment_handles_overlaps_and_missing_coordinates() -> None:
    annotations = [
        _annotation(
            "large",
            "Plaques",
            Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
        ),
        _annotation(
            "small",
            "Plaques",
            Polygon([(40, 40), (60, 40), (60, 60), (40, 60)]),
        ),
    ]
    coordinates = CellCoordinateTable(
        cell_ids=pd.Index(["overlap", "missing"]),
        coordinates=np.asarray([[50.0, 50.0], [np.nan, 2.0]]),
        source="test",
    )

    assigned = assign_distances_to_objects(coordinates, annotations)

    assert assigned.loc["overlap", "nearest_object_id"] == "small"
    assert assigned.loc["overlap", "distance_to_object_edge_um"] == 10.0
    assert bool(assigned.loc["overlap", "inside_object"])
    assert assigned.loc["missing", "distance_from_object_qc_flag"] == (
        "missing_coordinate"
    )
    assert pd.isna(assigned.loc["missing", "nearest_object_id"])


def test_proximity_threshold_boundaries() -> None:
    kwargs = {
        "is_inside": False,
        "near_distance_um": 50.0,
        "far_distance_um": 100.0,
        "max_distance_um": 200.0,
    }
    assert label_object_proximity(49.99, **kwargs) == "near"
    assert label_object_proximity(50.0, **kwargs) == "middle"
    assert label_object_proximity(100.0, **kwargs) == "far"
    assert label_object_proximity(200.0, **kwargs) == "far"
    assert label_object_proximity(200.01, **kwargs) == "beyond_max"


def test_pair_pseudobulk_uses_grey_matter_near_and_far_only(tmp_path: Path) -> None:
    table = ad.AnnData(
        X=sparse.csr_matrix(
            np.asarray(
                [
                    [1, 0],
                    [2, 1],
                    [5, 5],
                    [0, 3],
                    [100, 100],
                ],
                dtype=np.int64,
            )
        ),
        obs=pd.DataFrame(
            {"cell_id": ["c1", "c2", "c3", "c4", "c5"]},
            index=["c1", "c2", "c3", "c4", "c5"],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    assignments = pd.DataFrame(
        {
            "object_proximity": ["near", "near", "middle", "far", "far"],
            "cortical_depth_annotation": [
                "grey_matter",
                "grey_matter",
                "grey_matter",
                "grey_matter",
                "white_matter",
            ],
        },
        index=["c1", "c2", "c3", "c4", "c5"],
    )

    result = build_pair_pseudobulk(
        table,
        assignments,
        pair_id="block_1",
        included_tissue_annotations=["grey_matter"],
        min_cells=1,
    )

    assert result.obs["proximity"].tolist() == ["near", "far"]
    np.testing.assert_array_equal(result.X.toarray(), [[3, 1], [0, 3]])
    assert result.obs["cell_count"].tolist() == [2, 1]

    first_path = tmp_path / "first.h5ad"
    second_path = tmp_path / "second.h5ad"
    result.write_h5ad(first_path)
    second = result.copy()
    second.obs["pair_id"] = "block_2"
    second.obs_names = ["block_2__near", "block_2__far"]
    second.write_h5ad(second_path)
    combined = combine_pair_pseudobulks([first_path, second_path])
    paired, pair_ids = retain_complete_pairs(combined)

    assert pair_ids == ["block_1", "block_2"]
    assert paired.n_obs == 4


def test_distance_config_validates_ordered_thresholds(tmp_path: Path) -> None:
    config = DistanceFromObjectConfig.model_validate(
        {
            "pair_id": "block_1",
            "dataset_name": "block_1_XENIUM",
            "platform": "XENIUM",
            "latest_zarr_path": tmp_path / "latest.zarr",
            "output_dir": tmp_path / "out",
            "object_annotation_path": tmp_path / "objects.geojson",
            "tables": [
                {
                    "segmentation": "reseg",
                    "table_key": "table_MOSAIK_proseg",
                    "shape_key": "MOSAIK_proseg",
                }
            ],
        }
    )

    assert config.near_distance_um == 50.0
    assert config.far_distance_um == 100.0
    assert config.max_distance_um == 200.0


def test_paired_pydeseq2_smoke_returns_near_vs_far_results() -> None:
    counts = np.asarray(
        [
            [10, 50, 5, 30],
            [30, 45, 6, 31],
            [12, 55, 8, 35],
            [36, 50, 8, 32],
            [8, 40, 7, 28],
            [28, 38, 9, 29],
            [15, 60, 6, 40],
            [45, 56, 7, 38],
        ],
        dtype=np.int64,
    )
    pair_ids = ["block_1", "block_2", "block_3", "block_4"]
    obs = pd.DataFrame(
        {
            "pair_id": [pair_id for pair_id in pair_ids for _ in range(2)],
            "proximity": ["far", "near"] * len(pair_ids),
        },
        index=[f"sample_{index}" for index in range(len(counts))],
    )
    pseudobulk = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC", "GeneD"]),
    )

    result = run_paired_differential_expression(pseudobulk, n_cpus=1)

    assert set(result.index) == {"GeneA", "GeneB", "GeneC", "GeneD"}
    assert {"log2FoldChange", "pvalue", "padj"}.issubset(result.columns)
    assert result.loc["GeneA", "log2FoldChange"] > 0


def test_distance_pipeline_writes_sidecars_and_preserves_obs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    latest_path = tmp_path / "latest.zarr"
    latest_path.mkdir()
    object_path = tmp_path / "objects.geojson"
    object_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "object_id": "plaque_1",
                            "object_type": "Amyloid plaques",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
                            ],
                        },
                    }
                ],
            }
        )
    )
    table = ad.AnnData(
        X=sparse.csr_matrix(
            np.asarray([[1, 0], [2, 1], [0, 3], [1, 4]], dtype=np.int64)
        ),
        obs=pd.DataFrame(
            {
                "cell_id": ["c1", "c2", "c3", "c4"],
                "cortical_depth_annotation": ["grey_matter"] * 4,
                "existing_annotation": ["A", "B", "C", "D"],
            },
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
        obsm={
            "spatial": np.asarray([[5.0, 5.0], [20.0, 5.0], [150.0, 5.0], [180.0, 5.0]])
        },
    )

    class FakeSpatialData:
        def __init__(self: FakeSpatialData) -> None:
            self.tables = {"cells": table}
            self.shapes: dict[str, object] = {}

    written_tables: list[ad.AnnData] = []
    monkeypatch.setattr(
        "merxen.distance_from_object.pipeline.sd.read_zarr",
        lambda _path: FakeSpatialData(),
    )
    monkeypatch.setattr(
        "merxen.distance_from_object.pipeline.write_or_replace_element",
        lambda _sdata, _key, _kind, value, **_kwargs: written_tables.append(value),
    )
    config = DistanceFromObjectConfig.model_validate(
        {
            "pair_id": "block_1",
            "dataset_name": "block_1_XENIUM",
            "platform": "XENIUM",
            "latest_zarr_path": latest_path,
            "output_dir": tmp_path / "out",
            "object_annotation_path": object_path,
            "tables": [{"segmentation": "reseg", "table_key": "cells"}],
            "min_cells_per_pseudobulk": 1,
        }
    )

    paths = run_distance_from_object(config)

    assert paths["reseg_cells"].exists()
    assert paths["reseg_pseudobulk"].exists()
    pseudobulk = ad.read_h5ad(paths["reseg_pseudobulk"])
    assert pseudobulk.obs["proximity"].tolist() == ["near", "far"]
    assert len(written_tables) == 1
    assert written_tables[0].obs["existing_annotation"].tolist() == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert "distance_to_object_edge_um" in written_tables[0].obs


def test_distance_cohort_pipeline_discovers_pair_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pair_roots: list[Path] = []
    for pair_id, offset in (("block_1", 0), ("block_2", 5)):
        root = tmp_path / pair_id / "distance_from_object_out"
        segmentation_dir = root / "reseg"
        segmentation_dir.mkdir(parents=True)
        pair_roots.append(root)
        pseudobulk = ad.AnnData(
            X=sparse.csr_matrix(
                np.asarray([[10 + offset, 5], [30 + offset, 6]], dtype=np.int64)
            ),
            obs=pd.DataFrame(
                {
                    "pair_id": [pair_id, pair_id],
                    "proximity": ["far", "near"],
                    "cell_count": [20, 20],
                },
                index=[f"{pair_id}__far", f"{pair_id}__near"],
            ),
            var=pd.DataFrame(index=["GeneA", "GeneB"]),
        )
        pseudobulk.layers["counts"] = pseudobulk.X.copy()
        pseudobulk.write_h5ad(segmentation_dir / "pseudobulk_counts.h5ad")

    fake_results = pd.DataFrame(
        {
            "baseMean": [10.0, 5.0],
            "log2FoldChange": [1.0, -0.25],
            "pvalue": [0.01, 0.5],
            "padj": [0.02, 0.5],
        },
        index=pd.Index(["GeneA", "GeneB"], name="gene"),
    )
    monkeypatch.setattr(
        "merxen.distance_from_object.pipeline.run_paired_differential_expression",
        lambda _table, n_cpus=None: fake_results,
    )
    config = DistanceFromObjectCohortConfig(
        platform="XENIUM",
        annotation_output_dirs=pair_roots,
        output_dir=tmp_path / "cohort",
        segmentations=["reseg"],
        min_pairs=2,
        n_cpus=1,
    )

    paths = run_distance_from_object_cohort(config)

    assert paths["reseg_differential_expression"].exists()
    assert paths["reseg_differential_expression_parquet"].exists()
    assert paths["reseg_volcano"].exists()
    summary = json.loads(paths["summary"].read_text())
    assert summary["segmentations"]["reseg"]["status"] == "complete"
    assert summary["segmentations"]["reseg"]["complete_pair_ids"] == [
        "block_1",
        "block_2",
    ]


def _annotation(
    object_id: str,
    object_type: str,
    geometry: Polygon,
) -> ObjectAnnotation:
    return ObjectAnnotation(
        object_id=object_id,
        object_type=object_type,
        geometry=geometry,
    )
