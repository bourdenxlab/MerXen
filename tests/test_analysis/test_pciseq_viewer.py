"""Exact native-coordinate publication and transactional viewer discovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
import zarr
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Affine, get_transformation

from merxen.analysis.pciseq_viewer import publish_pciseq_viewer
from merxen.config import PciSeqConfig
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR

CELL_IDS = np.array([17, 2**53 + 7, 2**63 + 29, 2**64 - 5], dtype=np.uint64)
SPOT_IDS = np.array([2**63 + 91, 3, 2**53 + 17, 2**64 - 7, 7], dtype=np.uint64)


def _fixture(tmp_path: Path) -> tuple[PciSeqConfig, Path, pd.DataFrame]:
    from shapely.geometry import box

    config = PciSeqConfig(
        pair_id="toy_pair",
        sample_id="toy_sample",
        platform="XENIUM",
        segmentation="proseg_hybrid",
        spatialdata_path=tmp_path / "native.zarr",
        source_h5ad=tmp_path / "source.h5ad",
        reference_path=tmp_path / "reference.csv",
    )
    native_frame = pd.DataFrame(
        {
            "x": [1001.5, 1003.5, 1005.5, 1007.5, 1009.5],
            "y": [2001.25, 2003.25, 2005.25, 2007.25, 2009.25],
            "gene": ["A", "B", "A", "Blank-1", "other"],
            "transcript_id": SPOT_IDS,
            "assignment": pd.array([17] * 5, dtype="UInt64"),
            "background": [False] * 5,
        }
    )
    transform = Affine(
        np.array([[2.0, 0.0, 19.0], [0.0, 3.0, -7.0], [0.0, 0.0, 1.0]]),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    points = PointsModel.parse(
        dd.from_pandas(native_frame, npartitions=2),
        feature_key="gene",
        instance_key="assignment",
        transformations={"global": transform},
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {"instance_id": CELL_IDS},
            geometry=[box(i * 2, 0, i * 2 + 1, 1) for i in range(4)],
        ),
        transformations={"global": transform},
    )
    table = TableModel.parse(
        ad.AnnData(
            X=np.ones((4, 1)),
            obs=pd.DataFrame(
                {"instance_id": CELL_IDS, "region": pd.Categorical(["cells"] * 4)},
                index=["a", "b", "c", "d"],
            ),
            var=pd.DataFrame(index=["A"]),
        ),
        region="cells",
        region_key="region",
        instance_key="instance_id",
    )
    schema = {
        "schema_version": "1.0.0",
        "primary_points": "transcripts",
        "platform": "XENIUM",
        "segmentations": {
            "proseg_hybrid": {
                "points": "transcripts",
                "shape": "cells",
                "table": "table_native",
            }
        },
        "analysis_sources": {"other_source": {"method": "other"}},
    }
    data = sd.SpatialData(
        points={"transcripts": points},
        shapes={"cells": shapes},
        tables={"table_native": table},
        attrs={MERXEN_SCHEMA_ATTR: schema, "unrelated": "keep"},
    )
    data.write(config.spatialdata_path)
    output = tmp_path / "exports"
    output.mkdir()
    metadata = {
        "pair_id": config.pair_id,
        "sample_id": config.sample_id,
        "platform": config.platform,
        "segmentation": config.segmentation,
        "shape_key": "cells",
        "points_key": "transcripts",
        "source_table_key": "table_native",
        "reference_classes": ["A", "B", "NA"],
        "spot_count": 3,
    }
    (output / "manifest.json").write_text(json.dumps(metadata))
    (output / "compute_manifest.json").write_text(
        json.dumps(
            {
                "prepared_manifest_sha256": hashlib.sha256(
                    (output / "manifest.json").read_bytes()
                ).hexdigest(),
                "upstream_commit": "test",
            }
        )
    )
    cells = pd.DataFrame(
        {
            "instance_id": pd.Series(CELL_IDS[:3], dtype="uint64"),
            "pciseq_cell_type": ["A", None, "Zero"],
            "pciseq_probability": [0.9, np.nan, 0.8],
            "pciseq_status": ["assigned", "excluded_overlap", "zero_expression"],
            "pciseq_exclusion_reason": ["", "overlap_no_seed_pixels", ""],
            "reported_class_names": ['["A"]', "[]", '["Zero"]'],
            "reported_probabilities": ["[0.9]", "[]", "[0.8]"],
        }
    )
    cells.to_csv(output / "cell_assignments.csv", index=False)
    # Deliberately shuffled IDs; exported raster coordinates and dense labels
    # must never be used to construct viewer geometry or canonical assignments.
    spots = pd.DataFrame(
        {
            "transcript_id": pd.Series(
                [int(SPOT_IDS[2]), int(SPOT_IDS[0]), int(SPOT_IDS[1])], dtype="uint64"
            ),
            "instance_id": pd.array([17, None, int(CELL_IDS[2])], dtype="UInt64"),
            "gene_name": ["A", "A", "B"],
            "x": [-91, -92, -93],
            "y": [-81, -82, -83],
            "neighbour": [1, 0, 2],
        }
    )
    spots.to_csv(output / "spot_assignments.csv", index=False)
    annotated = table[[0]].copy()
    annotated.obs["pciseq_cell_type"] = pd.Categorical(["A"])
    annotated.obs["pciseq_probability"] = [0.9]
    annotated.obs["pciseq_status"] = pd.Categorical(["assigned"])
    annotated.obsm["X_umap"] = np.array([[8.0, 9.0]])
    annotated.write_h5ad(output / "annotated.h5ad")
    return config, output, native_frame


def test_native_points_exact_large_ids_coordinates_background_and_all_cells(
    tmp_path: Path,
) -> None:
    config, output, native = _fixture(tmp_path)
    result = publish_pciseq_viewer(config, output)
    store = sd.read_zarr(config.spatialdata_path)
    source = store.attrs[MERXEN_SCHEMA_ATTR]["analysis_sources"]["pciseq_proseg_hybrid"]
    actual = store.points[source["points"]].compute()
    pd.testing.assert_frame_equal(
        store.points["transcripts"].compute(),
        native,
        check_dtype=False,
        check_categorical=False,
    )
    assert actual.transcript_id.tolist() == SPOT_IDS.tolist()
    np.testing.assert_array_equal(actual[["x", "y"]], native[["x", "y"]])
    assert str(actual.transcript_id.dtype) == "uint64"
    assert str(actual.assignment.dtype) == "UInt64"
    assert actual.assignment.tolist() == [pd.NA, int(CELL_IDS[2]), 17, pd.NA, pd.NA]
    assert actual.background.tolist() == [True, False, False, False, False]
    assert actual.pciseq_assignment_status.tolist() == [
        "background",
        "assigned",
        "assigned",
        "not_modelled",
        "not_modelled",
    ]
    np.testing.assert_array_equal(
        get_transformation(
            store.points[source["points"]], to_coordinate_system="global"
        ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        get_transformation(
            store.points["transcripts"], to_coordinate_system="global"
        ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
    )
    table = store.tables[source["table"]]
    assert table.obs.instance_id.tolist() == CELL_IDS.tolist()
    assert table.obs.pciseq_status.tolist() == [
        "assigned",
        "excluded_overlap",
        "zero_expression",
        "not_modelled",
    ]
    assert table.obs.pciseq_probability.isna().tolist() == [False, True, False, True]
    assert table.n_vars == 0
    np.testing.assert_array_equal(
        store.tables[source["embedding_table"]].obsm["X_umap"], [[8.0, 9.0]]
    )
    assert store.attrs[MERXEN_SCHEMA_ATTR]["analysis_sources"]["other_source"] == {
        "method": "other"
    }
    assert source["segmentation"] == "proseg_hybrid"
    marker = zarr.open_group(
        str(config.spatialdata_path / "points" / source["points"]), mode="r"
    ).attrs["merxen_pciseq_viewer"]
    assert (
        marker["analysis_id"]
        == source["analysis_id"]
        == table.uns["merxen_pciseq_viewer"]["analysis_id"]
        == store.tables[source["embedding_table"]].uns["merxen_pciseq_viewer"][
            "analysis_id"
        ]
    )
    assert result["coverage"] == {
        "native_transcripts": 5,
        "modelled_transcripts": 3,
        "assigned_transcripts": 2,
        "background_transcripts": 1,
        "not_modelled_transcripts": 2,
        "native_cells": 4,
        "excluded_cells": 1,
        "not_modelled_cells": 1,
    }


def test_idempotent_publication_skips_all_element_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, _ = _fixture(tmp_path)
    first = publish_pciseq_viewer(config, output)

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("An identical publication must reuse serialized elements")

    monkeypatch.setattr("merxen.analysis.pciseq_viewer._write_elements", unexpected)
    again = publish_pciseq_viewer(config, output)
    assert again["reused"] is True
    assert again["analysis_id"] == first["analysis_id"]


def test_refresh_updates_embedding_and_generation_with_current_exports(
    tmp_path: Path,
) -> None:
    config, output, _ = _fixture(tmp_path)
    first = publish_pciseq_viewer(config, output)
    annotated = ad.read_h5ad(output / "annotated.h5ad")
    annotated.obsm["X_umap"][:] = [2.0, 3.0]
    annotated.write_h5ad(output / "annotated.h5ad")
    second = publish_pciseq_viewer(config, output)
    assert second["analysis_id"] != first["analysis_id"]
    store = sd.read_zarr(config.spatialdata_path)
    np.testing.assert_array_equal(
        store.tables[second["embedding_table"]].obsm["X_umap"], [[2.0, 3.0]]
    )
    assert (
        store.tables[second["embedding_table"]].uns["merxen_pciseq_viewer"][
            "analysis_id"
        ]
        == second["analysis_id"]
    )


@pytest.mark.parametrize(
    "fault", ["duplicate", "unknown", "wrong_gene", "excluded_cell"]
)
def test_invalid_transcript_exports_never_publish_registry(
    tmp_path: Path, fault: str
) -> None:
    config, output, _ = _fixture(tmp_path)
    spots = pd.read_csv(
        output / "spot_assignments.csv",
        dtype={"transcript_id": "string", "instance_id": "string"},
    )
    if fault == "duplicate":
        spots.loc[1, "transcript_id"] = spots.loc[0, "transcript_id"]
    elif fault == "unknown":
        spots.loc[1, "transcript_id"] = "99999"
    elif fault == "wrong_gene":
        spots.loc[1, "gene_name"] = "B"
    else:
        spots.loc[0, "instance_id"] = str(int(CELL_IDS[1]))
    spots.to_csv(output / "spot_assignments.csv", index=False)
    with pytest.raises(ValueError):
        publish_pciseq_viewer(config, output)
    store = sd.read_zarr(config.spatialdata_path)
    assert (
        "pciseq_proseg_hybrid"
        not in store.attrs[MERXEN_SCHEMA_ATTR]["analysis_sources"]
    )
    assert set(store.points) == {"transcripts"}


def test_valid_na_class_name_survives_csv_read(tmp_path: Path) -> None:
    config, output, _ = _fixture(tmp_path)
    cells = pd.read_csv(
        output / "cell_assignments.csv", dtype={"instance_id": "string"}
    )
    cells.loc[0, "pciseq_cell_type"] = "NA"
    cells.to_csv(output / "cell_assignments.csv", index=False)
    result = publish_pciseq_viewer(config, output)
    assert (
        sd.read_zarr(config.spatialdata_path)
        .tables[result["table"]]
        .obs.pciseq_cell_type.iloc[0]
        == "NA"
    )


def test_categorical_cell_status_allows_unmodelled_native_cells(tmp_path: Path) -> None:
    config, output, _ = _fixture(tmp_path)
    cells = pd.read_csv(
        output / "cell_assignments.csv", dtype={"instance_id": "string"}
    )
    cells.pciseq_status = pd.Categorical(cells.pciseq_status)
    result = publish_pciseq_viewer(config, output, cell_assignments=cells)
    assert result["coverage"]["not_modelled_cells"] == 1


def test_store_is_read_under_real_writer_lock_and_other_attrs_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, _ = _fixture(tmp_path)
    from merxen.io.spatialdata_io import spatialdata_write_lock_path

    original = sd.read_zarr
    calls = []

    def checked(path: Path) -> Any:
        with (
            spatialdata_write_lock_path(path).open("a") as handle,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append(path)
        return original(path)

    monkeypatch.setattr(sd, "read_zarr", checked)
    publish_pciseq_viewer(config, output)
    assert len(calls) == 1
    assert original(config.spatialdata_path).attrs["unrelated"] == "keep"


def test_failed_refresh_restores_previous_elements_and_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, _ = _fixture(tmp_path)
    first = publish_pciseq_viewer(config, output)
    before = deepcopy(sd.read_zarr(config.spatialdata_path).attrs)
    import merxen.io.spatialdata_io as io

    original = io.write_spatialdata_metadata
    fail = [True]

    def broken(sdata: Any, **kwargs: Any) -> None:
        if fail.pop() if fail else False:
            raise OSError("Registry commit failed")
        original(sdata, **kwargs)

    annotated = ad.read_h5ad(output / "annotated.h5ad")
    annotated.obsm["X_umap"][:] = [2.0, 3.0]
    annotated.write_h5ad(output / "annotated.h5ad")
    monkeypatch.setattr(io, "write_spatialdata_metadata", broken)
    with pytest.raises(OSError, match="Registry commit failed"):
        publish_pciseq_viewer(config, output)
    restored = sd.read_zarr(config.spatialdata_path)
    assert restored.attrs == before
    np.testing.assert_array_equal(
        restored.tables[first["embedding_table"]].obsm["X_umap"], [[8.0, 9.0]]
    )
    assert not any(
        "_pciseq_stage_" in key or "pciseq-backup" in key
        for group in [restored.points, restored.tables]
        for key in group
    )


@pytest.mark.parametrize("variant", ["legacy", "registered", "unrelated"])
def test_embedding_retains_only_valid_coordinate_variant_links(
    tmp_path: Path, variant: str
) -> None:
    config, output, _ = _fixture(tmp_path)
    store = sd.read_zarr(config.spatialdata_path)
    key = "cells_aligned_nonrigid" if variant == "legacy" else "other_shapes"
    store.shapes[key] = store.shapes["cells"].copy()
    store.write_element(key)
    if variant == "registered":
        schema = store.attrs[MERXEN_SCHEMA_ATTR]
        schema["segmentations"]["verified_variant"] = {
            "shape": key,
            "coordinate_variant_of": "proseg_hybrid",
        }
        store.write_attrs()
        store.write_consolidated_metadata()
    annotated = ad.read_h5ad(output / "annotated.h5ad")
    annotated.uns["spatialdata_attrs"]["region"] = key
    annotated.obs["region"] = pd.Categorical([key])
    annotated.write_h5ad(output / "annotated.h5ad")
    if variant == "unrelated":
        with pytest.raises(ValueError, match="unrelated segmentation"):
            publish_pciseq_viewer(config, output)
        return
    result = publish_pciseq_viewer(config, output)
    restored = sd.read_zarr(config.spatialdata_path)
    assert result["shape"] == "cells"
    assert result["embedding_shape"] == key
    assert (
        restored.tables[result["embedding_table"]].uns["spatialdata_attrs"]["region"]
        == key
    )


@pytest.mark.parametrize(
    "column,value", [("pciseq_cell_type", "A"), ("pciseq_probability", 0.7)]
)
def test_excluded_cells_reject_invented_annotations(
    tmp_path: Path, column: str, value: Any
) -> None:
    config, output, _ = _fixture(tmp_path)
    cells = pd.read_csv(
        output / "cell_assignments.csv", dtype={"instance_id": "string"}
    )
    cells.loc[1, column] = value
    cells.to_csv(output / "cell_assignments.csv", index=False)
    with pytest.raises(ValueError, match="Excluded cells cannot"):
        publish_pciseq_viewer(config, output)
