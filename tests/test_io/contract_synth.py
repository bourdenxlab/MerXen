"""Synthetic Xenium-like and MERSCOPE-like stores for the points-contract tests.

Every points element is a multi-partition Dask frame read through
:class:`PartitionReads`, which counts how often each partition is read and the
largest number of partitions in memory at once. The counts make the number of
full passes over the points a testable quantity.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter

import anndata as ad
import dask
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from scipy import sparse
from shapely.geometry import box
from spatialdata.models import Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Scale

XENIUM_GENES = ("Snap25", "Gad1", "Actb", "NegControlProbe_00042")
XENIUM_CELLS = ("cell-c", "cell-a", "cell-b")
XENIUM_LABELS = (9, 5, 2)
MERSCOPE_GENES = ("Zeb2", "Actb", "Gad1")
MERSCOPE_CELLS = ("1103", "1101", "1102")
ROWS_PER_PARTITION = 6
N_PARTITIONS = 4


class PartitionReads:
    """Serve fixed pandas partitions and record every read.

    Attributes:
        frames: The partitions, in order.
        reads: Number of reads per partition number.
        peak_in_flight: Largest number of partitions being read at once.
    """

    def __init__(
        self, frames: list[pd.DataFrame], *, read_delay_s: float = 0.0
    ) -> None:
        """Store the partitions.

        Args:
            frames: Partitions served by partition number.
            read_delay_s: Sleep per read, so that concurrent reads overlap.
        """
        self.frames = frames
        self.reads: Counter[int] = Counter()
        self.peak_in_flight = 0
        self._in_flight = 0
        self._read_delay_s = read_delay_s
        self._lock = threading.Lock()
        self._token = uuid.uuid4().hex

    def __dask_tokenize__(self) -> str:
        """Give each counter its own Dask token (the lock cannot be hashed)."""
        return self._token

    def __call__(self, number: int) -> pd.DataFrame:
        """Return a copy of partition ``number`` and record the read."""
        with self._lock:
            self.reads[number] += 1
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if self._read_delay_s:
                time.sleep(self._read_delay_s)
            return self.frames[number].copy()
        finally:
            with self._lock:
                self._in_flight -= 1

    @property
    def total(self) -> int:
        """Total partition reads since the last reset."""
        return sum(self.reads.values())

    def reset(self) -> None:
        """Forget all recorded reads."""
        with self._lock:
            self.reads.clear()
            self.peak_in_flight = 0

    def dataframe(self) -> dd.DataFrame:
        """Return a lazy Dask frame with one partition per stored frame.

        String dtypes are served as stored (Dask would otherwise convert them
        to pyarrow strings), so a frame computed from a store compares equal
        to the same frame served again.
        """
        with dask.config.set({"dataframe.convert-string": False}):
            return dd.from_map(
                self,
                range(len(self.frames)),
                meta=self.frames[0].iloc[:0],
                enforce_metadata=False,
            )


def split_frame(frame: pd.DataFrame, n_partitions: int) -> list[pd.DataFrame]:
    """Split a frame into equal partitions, each with its own RangeIndex."""
    bounds = np.linspace(0, len(frame), n_partitions + 1).astype(int)
    return [
        frame.iloc[start:stop].reset_index(drop=True)
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def xenium_raw_points_frame() -> pd.DataFrame:
    """Return raw Xenium-reader-like transcripts (opaque cell IDs, vendor IDs)."""
    n_rows = ROWS_PER_PARTITION * N_PARTITIONS
    rng = np.random.default_rng(7)
    cells = np.asarray([*XENIUM_CELLS, "UNASSIGNED"], dtype=object)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 3, n_rows).astype(np.float32),
            "y": rng.uniform(0, 1, n_rows).astype(np.float32),
            "z": rng.uniform(0, 1, n_rows).astype(np.float32),
            "feature_name": np.asarray(XENIUM_GENES, dtype=object)[
                np.arange(n_rows) % len(XENIUM_GENES)
            ],
            "cell_id": cells[(np.arange(n_rows) * 3) % len(cells)],
            "transcript_id": (np.uint64(281474976710656) + np.arange(n_rows))[
                ::-1
            ].astype(np.uint64),
            "qv": rng.uniform(5, 40, n_rows).astype(np.float32),
            "overlaps_nucleus": (np.arange(n_rows) % 2).astype(np.uint8),
        }
    )


def merscope_raw_points_frame() -> pd.DataFrame:
    """Return raw MERSCOPE-builder-like transcripts with unsorted known genes."""
    n_rows = ROWS_PER_PARTITION * N_PARTITIONS
    rng = np.random.default_rng(11)
    cells = np.asarray([*MERSCOPE_CELLS, "-1"], dtype=object)
    genes = pd.Categorical(
        np.asarray(MERSCOPE_GENES, dtype=object)[np.arange(n_rows) % 3],
        categories=list(MERSCOPE_GENES),
    )
    return pd.DataFrame(
        {
            "global_x": rng.uniform(0, 3, n_rows),
            "global_y": rng.uniform(0, 1, n_rows),
            "global_z": (np.arange(n_rows) % 3).astype(np.float64),
            "gene": genes,
            "cell_id": cells[np.arange(n_rows) % len(cells)],
            "transcript_score": rng.uniform(0, 1, n_rows).astype(np.float32),
        }
    )


def _xenium_shapes_and_table() -> tuple[dict[str, object], dict[str, object]]:
    """Return Xenium-like cell boundaries, labels and a label-keyed table."""
    region = "cell_labels"
    labels = Labels2DModel.parse(
        np.asarray([[0, 9], [5, 2]], dtype=np.uint32),
        dims=("y", "x"),
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {"geometry": [box(i, 0, i + 1, 1) for i in range(len(XENIUM_CELLS))]},
            geometry="geometry",
            index=pd.Index(list(XENIUM_CELLS)),
        )
    )
    obs = pd.DataFrame(
        {
            "cell_id": list(XENIUM_CELLS),
            "cell_labels": list(XENIUM_LABELS),
            "region": pd.Categorical([region] * len(XENIUM_CELLS)),
        },
        index=pd.Index([str(i) for i in range(len(XENIUM_CELLS))]),
    )
    table = TableModel.parse(
        ad.AnnData(
            X=sparse.csr_matrix(np.ones((len(XENIUM_CELLS), len(XENIUM_GENES)))),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(list(XENIUM_GENES), name="gene")),
        ),
        region=region,
        region_key="region",
        instance_key="cell_labels",
    )
    return {"cell_boundaries": shapes, region: labels}, {"table": table}


def _merscope_shapes_and_table() -> tuple[dict[str, object], dict[str, object]]:
    """Return MERSCOPE-like polygons and an EntityID-keyed table."""
    region = "merscope_polygons"
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "EntityID": list(MERSCOPE_CELLS),
                "geometry": [box(i, 0, i + 1, 1) for i in range(len(MERSCOPE_CELLS))],
            },
            geometry="geometry",
            index=pd.Index(list(MERSCOPE_CELLS), name="EntityID"),
        )
    )
    obs = pd.DataFrame(
        {
            "EntityID": list(MERSCOPE_CELLS),
            "region": pd.Categorical([region] * len(MERSCOPE_CELLS)),
        },
        index=pd.Index(list(MERSCOPE_CELLS)),
    )
    table = TableModel.parse(
        ad.AnnData(
            X=sparse.csr_matrix(np.ones((len(MERSCOPE_CELLS), len(MERSCOPE_GENES)))),
            obs=obs,
            var=pd.DataFrame(index=pd.Index(list(MERSCOPE_GENES), name="gene")),
        ),
        region=region,
        region_key="region",
        instance_key="EntityID",
    )
    return {region: shapes}, {"table": table}


def _assemble(
    points: dd.DataFrame,
    elements: tuple[dict[str, object], dict[str, object]],
) -> sd.SpatialData:
    """Combine points with shapes, labels and tables into one SpatialData."""
    shapes_and_labels, tables = elements
    return sd.SpatialData(
        points={"transcripts": points},
        shapes={
            key: value
            for key, value in shapes_and_labels.items()
            if isinstance(value, gpd.GeoDataFrame)
        },
        labels={
            key: value
            for key, value in shapes_and_labels.items()
            if not isinstance(value, gpd.GeoDataFrame)
        },
        tables=tables,
    )


def xenium_raw_spatialdata() -> tuple[sd.SpatialData, PartitionReads]:
    """Return a raw Xenium-like store and the read counter of its points."""
    reads = PartitionReads(split_frame(xenium_raw_points_frame(), N_PARTITIONS))
    points = PointsModel.parse(
        reads.dataframe(),
        coordinates={"x": "x", "y": "y", "z": "z"},
        feature_key="feature_name",
        instance_key="cell_id",
        transformations={"global": Scale([4.0, 4.0], axes=("x", "y"))},
        sort=True,
    )
    reads.reset()
    return _assemble(points, _xenium_shapes_and_table()), reads


def merscope_raw_spatialdata() -> tuple[sd.SpatialData, PartitionReads]:
    """Return a raw MERSCOPE-like store and the read counter of its points."""
    reads = PartitionReads(split_frame(merscope_raw_points_frame(), N_PARTITIONS))
    points = PointsModel.parse(
        reads.dataframe(),
        coordinates={"x": "global_x", "y": "global_y", "z": "global_z"},
        feature_key="gene",
        instance_key="cell_id",
        sort=True,
    )
    reads.reset()
    return _assemble(points, _merscope_shapes_and_table()), reads


def canonical_spatialdata_like(
    reference: sd.SpatialData,
    raw: sd.SpatialData,
    *,
    frame: pd.DataFrame | None = None,
    points_key: str = "transcripts",
    read_delay_s: float = 0.0,
) -> tuple[sd.SpatialData, PartitionReads]:
    """Rebuild ``raw`` with points already in the canonical MerXen form.

    The points are the computed default-contract points of ``reference`` (or
    an edited copy of them), served again partition by partition with their
    known categories, as a reader that emits canonical points directly would
    serve them.

    Args:
        reference: A store that went through the default contract.
        raw: The raw store the reference was made from (shapes and tables
            are taken from it, untouched).
        frame: Points to serve instead of the computed reference points.
        points_key: Points element to rebuild.
        read_delay_s: Sleep per partition read.

    Returns:
        The rebuilt store and the read counter of its points.
    """
    reference_points = reference.points[points_key]
    if frame is None:
        frame = reference_points.compute().reset_index(drop=True)
    reads = PartitionReads(split_frame(frame, N_PARTITIONS), read_delay_s=read_delay_s)
    point_attrs = dict(reference_points.attrs.get("spatialdata_attrs", {}))
    coordinates = {axis: axis for axis in ("x", "y", "z") if axis in frame.columns}
    points = PointsModel.parse(
        reads.dataframe(),
        coordinates=coordinates,
        feature_key=point_attrs["feature_key"],
        instance_key=point_attrs.get("instance_key"),
        transformations=dict(reference_points.attrs["transform"]),
        sort=True,
    )
    reads.reset()
    rebuilt = sd.SpatialData(
        points={points_key: points},
        shapes=dict(raw.shapes),
        labels=dict(raw.labels),
        tables=dict(raw.tables),
    )
    return rebuilt, reads
