"""Default behaviour and opt-in fast paths of the transcript points contract.

The characterisation tests pin what the default (Xenium and MERSCOPE) path
produces today, including the number of full passes over the points; the
equivalence tests show that the opt-in fast paths give the same outputs.
"""

from __future__ import annotations

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse
from shapely.geometry import box
from spatialdata.models import PointsModel, ShapesModel, TableModel

from merxen.io.spatialdata_io import (
    normalize_points_for_latest_write,
    prepare_source_spatialdata_contract,
    upgrade_spatialdata_contract_in_memory,
)
from merxen.io.spatialdata_schema import with_stable_transcript_ids

from .contract_synth import (
    MERSCOPE_CELLS,
    N_PARTITIONS,
    ROWS_PER_PARTITION,
    XENIUM_CELLS,
    XENIUM_LABELS,
    PartitionReads,
    merscope_raw_spatialdata,
    split_frame,
    xenium_raw_points_frame,
    xenium_raw_spatialdata,
)

N_ROWS = ROWS_PER_PARTITION * N_PARTITIONS
PROSEG_GENES = ("Zeb2", "Actb", "Gad1")


def _raw_proseg_spatialdata() -> tuple[sd.SpatialData, PartitionReads]:
    """Return a raw multi-partition ProSeg store with unsorted known genes."""
    assignment = pd.Series(
        [index % 3 if index % 4 else pd.NA for index in range(N_ROWS)],
        dtype="UInt32",
    )
    frame = pd.DataFrame(
        {
            "x": np.linspace(0.0, 2.9, N_ROWS),
            "y": np.linspace(0.0, 0.9, N_ROWS),
            "z": np.zeros(N_ROWS, dtype=np.float32),
            "gene": pd.Categorical(
                np.asarray(PROSEG_GENES, dtype=object)[np.arange(N_ROWS) % 3],
                categories=list(PROSEG_GENES),
            ),
            "transcript_id": np.arange(1, N_ROWS + 1, dtype=np.uint64),
            "assignment": assignment,
            "background": assignment.isna().to_numpy(),
        }
    )
    reads = PartitionReads(split_frame(frame, N_PARTITIONS))
    points = PointsModel.parse(
        reads.dataframe(),
        coordinates={"x": "x", "y": "y", "z": "z"},
        feature_key="gene",
        sort=True,
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "cell": np.asarray([0, 1, 2], dtype=np.uint32),
                "geometry": [box(i, 0, i + 1, 1) for i in range(3)],
            },
            geometry="geometry",
            index=pd.Index([0, 1, 2], name="cell"),
        )
    )
    obs = pd.DataFrame(
        {
            "cell": np.asarray([0, 1, 2], dtype=np.uint32),
            "original_cell_id": ["10", "20", "30"],
            "region": pd.Categorical(["cell_boundaries"] * 3),
        },
        index=pd.Index(["0", "1", "2"], name="cell_index"),
    )
    table = TableModel.parse(
        ad.AnnData(
            X=sparse.csr_matrix(np.eye(3)),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(["GeneA", "GeneB", "GeneC"])),
        ),
        region="cell_boundaries",
        region_key="region",
        instance_key="cell",
    )
    reads.reset()
    sdata = sd.SpatialData(
        points={"transcripts": points},
        shapes={"cell_boundaries": shapes},
        tables={"table": table},
    )
    return sdata, reads


# --- Characterisation of the default path -------------------------------------


def test_default_contract_pins_xenium_points() -> None:
    """Xenium points get regenerated IDs, mapped labels and two full passes."""
    sdata, reads = xenium_raw_spatialdata()

    prepare_source_spatialdata_contract(sdata, platform="XENIUM")

    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 2)
    points = sdata.points["transcripts"]
    assert points.attrs["spatialdata_attrs"] == {
        "feature_key": "feature_name",
        "instance_key": "original_assignment",
    }
    frame = points.compute()
    assert list(frame.columns) == [
        "x",
        "y",
        "z",
        "feature_name",
        "transcript_id",
        "qv",
        "overlaps_nucleus",
        "source_cell_id",
        "original_assignment",
        "source_transcript_id",
    ]
    assert frame["transcript_id"].dtype == np.uint64
    assert frame["transcript_id"].tolist() == list(range(1, N_ROWS + 1))
    raw = xenium_raw_points_frame()
    assert frame["source_transcript_id"].tolist() == raw["transcript_id"].tolist()
    assert frame["original_assignment"].dtype == "UInt64"
    labels = dict(zip(XENIUM_CELLS, XENIUM_LABELS, strict=True))
    expected = [labels.get(cell) for cell in raw["cell_id"]]
    assert [
        None if pd.isna(value) else int(value) for value in frame["original_assignment"]
    ] == expected
    assert frame["feature_name"].cat.categories.tolist() == sorted(
        set(raw["feature_name"])
    )


def test_default_contract_sorts_merscope_categories() -> None:
    """MERSCOPE genes arrive known in reader order and leave sorted."""
    sdata, reads = merscope_raw_spatialdata()
    assert sdata.points["transcripts"]["gene"].cat.categories.tolist() == [
        "Zeb2",
        "Actb",
        "Gad1",
    ]

    prepare_source_spatialdata_contract(sdata, platform="MERSCOPE")

    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 2)
    frame = sdata.points["transcripts"].compute()
    assert frame["gene"].cat.categories.tolist() == ["Actb", "Gad1", "Zeb2"]
    assert list(frame.columns) == [
        "x",
        "y",
        "z",
        "gene",
        "transcript_score",
        "source_cell_id",
        "original_assignment",
        "transcript_id",
    ]
    assert frame["transcript_id"].tolist() == list(range(1, N_ROWS + 1))
    assert set(frame["original_assignment"].dropna().astype(int)) == set(
        map(int, MERSCOPE_CELLS)
    )


def test_default_normalize_points_rewrites_canonical_dtypes() -> None:
    """The default normaliser casts genes to strings and narrows assignments."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "gene": pd.Categorical(["B", "A"], categories=["B", "A"]),
            "transcript_id": np.asarray([5, 6], dtype=np.uint64),
            "assignment": pd.Series([7, pd.NA], dtype="UInt64"),
            "cell": np.asarray([1, 2], dtype=np.uint64),
        }
    )

    out = normalize_points_for_latest_write(dd.from_pandas(frame, npartitions=2))

    assert out["gene"].dtype == "string"
    assert out["transcript_id"].dtype == "UInt64"
    assert out["assignment"].dtype == "UInt32"
    assert out["cell"].dtype == np.uint32
    computed = out.compute()
    assert computed["gene"].tolist() == ["B", "A"]
    assert computed["transcript_id"].tolist() == [5, 6]


def test_default_proseg_upgrade_regenerates_ids_and_sorts_categories() -> None:
    """ProSeg points get counted IDs and sorted categories in two passes."""
    sdata, reads = _raw_proseg_spatialdata()

    assert upgrade_spatialdata_contract_in_memory(sdata, platform="MERSCOPE")

    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 2)
    frame = sdata.points["transcripts"].compute()
    assert frame["gene"].cat.categories.tolist() == ["Actb", "Gad1", "Zeb2"]
    assert frame["transcript_id"].dtype == np.uint64
    assert frame["transcript_id"].tolist() == list(range(1, N_ROWS + 1))
    assert frame["assignment"].dtype == "UInt64"


def test_default_stable_transcript_ids_count_every_partition() -> None:
    """Without known lengths the IDs need one counting pass over the points."""
    reads = PartitionReads(split_frame(xenium_raw_points_frame(), N_PARTITIONS))

    result = with_stable_transcript_ids(reads.dataframe())

    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1)
    assert result.compute()["transcript_id"].tolist() == list(range(1, N_ROWS + 1))
