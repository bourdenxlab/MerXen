"""Default behaviour and opt-in fast paths of the transcript points contract.

The characterisation tests pin what the default (Xenium and MERSCOPE) path
produces today, including the number of full passes over the points; the
equivalence tests show that the opt-in fast paths give the same outputs.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from scipy import sparse
from shapely.geometry import box
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import get_transformation

import merxen
from merxen.io.spatialdata_io import (
    normalize_points_for_latest_write,
    prepare_source_spatialdata_contract,
    upgrade_spatialdata_contract_in_memory,
    write_spatialdata_zarr,
)
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    SpatialDataContractError,
    validate_merxen_schema,
    validate_merxen_schema_streaming,
    with_stable_transcript_ids,
)

from .contract_synth import (
    MERSCOPE_CELLS,
    N_PARTITIONS,
    ROWS_PER_PARTITION,
    XENIUM_CELLS,
    XENIUM_LABELS,
    PartitionReads,
    canonical_spatialdata_like,
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


# --- Opt-in fast paths ----------------------------------------------------------


RAW_BUILDERS = {
    "XENIUM": xenium_raw_spatialdata,
    "MERSCOPE": merscope_raw_spatialdata,
}


def _default_and_canonical(
    platform: str,
) -> tuple[sd.SpatialData, sd.SpatialData, PartitionReads]:
    """Return a default-contract store and its raw store with canonical points."""
    reference, _ = RAW_BUILDERS[platform]()
    prepare_source_spatialdata_contract(reference, platform=platform)
    raw, _ = RAW_BUILDERS[platform]()
    canonical, reads = canonical_spatialdata_like(reference, raw)
    return reference, canonical, reads


@pytest.mark.parametrize("platform", sorted(RAW_BUILDERS))
def test_points_canonical_matches_default_without_a_pass(platform: str) -> None:
    """Canonical points give the default outputs and are never read."""
    reference, canonical, reads = _default_and_canonical(platform)

    prepare_source_spatialdata_contract(
        canonical, platform=platform, points_canonical=True
    )

    assert reads.total == 0
    expected_points = reference.points["transcripts"]
    points = canonical.points["transcripts"]
    assert (
        points.attrs["spatialdata_attrs"] == expected_points.attrs["spatialdata_attrs"]
    )
    assert get_transformation(points, get_all=True) == get_transformation(
        expected_points, get_all=True
    )
    pd.testing.assert_frame_equal(points.compute(), expected_points.compute())
    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1)
    assert canonical.attrs == reference.attrs
    assert set(canonical.shapes) == set(reference.shapes)
    for key in reference.shapes:
        pd.testing.assert_frame_equal(
            pd.DataFrame(canonical.shapes[key]),
            pd.DataFrame(reference.shapes[key]),
        )
    pd.testing.assert_frame_equal(
        canonical.tables["table"].obs, reference.tables["table"].obs
    )
    assert (
        canonical.tables["table"].uns["spatialdata_attrs"]
        == reference.tables["table"].uns["spatialdata_attrs"]
    )
    validate_merxen_schema(canonical, deep=True)


def test_points_canonical_keeps_the_reader_category_order() -> None:
    """The category skip is keyed on the flag: known categories stay unsorted."""
    reference, _ = merscope_raw_spatialdata()
    prepare_source_spatialdata_contract(reference, platform="MERSCOPE")
    frame = reference.points["transcripts"].compute().reset_index(drop=True)
    frame["gene"] = frame["gene"].cat.reorder_categories(["Zeb2", "Actb", "Gad1"])
    raw, _ = merscope_raw_spatialdata()
    canonical, reads = canonical_spatialdata_like(reference, raw, frame=frame)

    prepare_source_spatialdata_contract(
        canonical, platform="MERSCOPE", points_canonical=True
    )

    assert reads.total == 0
    genes = canonical.points["transcripts"].compute()["gene"]
    assert genes.cat.categories.tolist() == ["Zeb2", "Actb", "Gad1"]
    assert genes.astype(str).tolist() == frame["gene"].astype(str).tolist()


def test_points_canonical_writes_points_in_one_read(tmp_path: Path) -> None:
    """Contract plus write reads every canonical partition exactly once."""
    _, canonical, reads = _default_and_canonical("XENIUM")

    prepare_source_spatialdata_contract(
        canonical, platform="XENIUM", points_canonical=True
    )
    write_spatialdata_zarr(canonical, tmp_path / "source.zarr")

    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1)
    written = sd.read_zarr(tmp_path / "source.zarr")
    validate_merxen_schema(written, deep=False)
    assert validate_merxen_schema_streaming(written)["transcripts"].n_rows == N_ROWS


def test_points_canonical_without_a_table_leaves_points_untouched() -> None:
    """Without a cell table canonical points keep their own IDs, unread."""
    _, canonical, reads = _default_and_canonical("MERSCOPE")
    del canonical.tables["table"]
    before = canonical.points["transcripts"]

    prepare_source_spatialdata_contract(
        canonical, platform="MERSCOPE", points_canonical=True
    )

    assert canonical.points["transcripts"] is before
    assert reads.total == 0


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda frame: frame.astype({"transcript_id": "int64"}), "uint64"),
        (
            lambda frame: frame.astype({"original_assignment": "float64"}),
            "UInt64",
        ),
    ],
)
def test_points_canonical_rejects_non_canonical_metadata(
    edit: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    """The flag checks dtypes from the metadata and refuses raw points."""
    reference, _ = merscope_raw_spatialdata()
    prepare_source_spatialdata_contract(reference, platform="MERSCOPE")
    frame = edit(reference.points["transcripts"].compute().reset_index(drop=True))
    raw, _ = merscope_raw_spatialdata()
    canonical, reads = canonical_spatialdata_like(reference, raw, frame=frame)

    with pytest.raises(SpatialDataContractError, match=message):
        prepare_source_spatialdata_contract(
            canonical, platform="MERSCOPE", points_canonical=True
        )
    assert reads.total == 0


def test_points_canonical_rejects_unknown_categories(tmp_path: Path) -> None:
    """Points read back from Parquet have unknown categories: not canonical."""
    reference, _ = merscope_raw_spatialdata()
    prepare_source_spatialdata_contract(reference, platform="MERSCOPE")
    write_spatialdata_zarr(reference, tmp_path / "source.zarr")
    written = sd.read_zarr(tmp_path / "source.zarr")
    assert not written.points["transcripts"]["gene"].cat.known

    with pytest.raises(SpatialDataContractError, match="known categories"):
        prepare_source_spatialdata_contract(
            written, platform="MERSCOPE", points_canonical=True
        )


def test_points_canonical_rejects_unmapped_source_assignments() -> None:
    """Raw reader points with opaque cell IDs are not canonical."""
    raw, reads = xenium_raw_spatialdata()

    with pytest.raises(SpatialDataContractError, match="needs mapping"):
        prepare_source_spatialdata_contract(
            raw, platform="XENIUM", points_canonical=True
        )
    assert reads.total == 0


def test_normalize_points_canonical_keeps_canonical_dtypes() -> None:
    """The opt-in keeps known genes and unsigned IDs with the default values."""
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
    points = dd.from_pandas(frame, npartitions=2)

    # The normaliser assigns columns in place, so each call gets its own copy.
    default = normalize_points_for_latest_write(points.copy()).compute()
    fast = normalize_points_for_latest_write(points.copy(), points_canonical=True)

    assert fast["gene"].cat.known
    assert fast["gene"].cat.categories.tolist() == ["B", "A"]
    assert fast["transcript_id"].dtype == np.uint64
    assert fast["assignment"].dtype == "UInt64"
    assert fast["cell"].dtype == np.uint64
    computed = fast.compute()
    assert computed["gene"].astype(str).tolist() == default["gene"].tolist()
    for column in ("transcript_id", "assignment", "cell"):
        assert (
            computed[column].astype("Int64").tolist()
            == default[column].astype("Int64").tolist()
        )


def test_normalize_points_canonical_restores_categories_after_parquet(
    tmp_path: Path,
) -> None:
    """A Parquet round trip loses known categories; the list restores them."""
    frame = pd.DataFrame(
        {
            "x": np.arange(6, dtype=np.float64),
            "y": np.arange(6, dtype=np.float64),
            "feature_name": pd.Categorical(
                ["C", "A", "B", "C", "B", "A"], categories=["C", "A", "B"]
            ),
        }
    )
    dd.from_pandas(frame, npartitions=3).to_parquet(tmp_path / "points.parquet")
    round_tripped = dd.read_parquet(tmp_path / "points.parquet")
    assert not round_tripped["feature_name"].cat.known

    fallback = normalize_points_for_latest_write(
        round_tripped.copy(), points_canonical=True
    )
    restored = normalize_points_for_latest_write(
        round_tripped.copy(),
        points_canonical=True,
        feature_categories=["C", "A", "B"],
    )

    assert fallback["feature_name"].dtype == "string"
    assert restored["feature_name"].cat.known
    computed = restored.compute()["feature_name"]
    assert computed.cat.categories.tolist() == ["C", "A", "B"]
    assert computed.cat.codes.tolist() == [0, 1, 2, 0, 2, 1]

    reordered = normalize_points_for_latest_write(
        dd.from_pandas(frame, npartitions=3),
        points_canonical=True,
        feature_categories=["A", "B", "C"],
    )
    assert reordered.compute()["feature_name"].cat.categories.tolist() == [
        "A",
        "B",
        "C",
    ]

    missing = normalize_points_for_latest_write(
        round_tripped.copy(),
        points_canonical=True,
        feature_categories=["A", "B"],
    )
    with pytest.raises(ValueError, match="missing from the known categories"):
        missing.compute()


def test_proseg_upgrade_fast_path_matches_default_without_a_pass() -> None:
    """ProSeg points with canonical IDs and genes skip both passes."""
    default_sdata, _ = _raw_proseg_spatialdata()
    upgrade_spatialdata_contract_in_memory(default_sdata, platform="MERSCOPE")
    fast_sdata, reads = _raw_proseg_spatialdata()

    assert upgrade_spatialdata_contract_in_memory(
        fast_sdata, platform="MERSCOPE", points_canonical=True
    )

    assert reads.total == 0
    default = default_sdata.points["transcripts"].compute()
    fast = fast_sdata.points["transcripts"].compute()
    assert fast["gene"].cat.categories.tolist() == list(PROSEG_GENES)
    pd.testing.assert_frame_equal(
        fast.assign(gene=fast["gene"].astype(str)),
        default.assign(gene=default["gene"].astype(str)),
    )
    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1)
    validate_merxen_schema(fast_sdata, deep=True)


def test_proseg_upgrade_fast_path_regenerates_non_canonical_ids() -> None:
    """IDs that are not uint64 are renumbered even on the fast path."""
    sdata, reads = _raw_proseg_spatialdata()
    frames = [
        frame.astype({"transcript_id": "int64"}).assign(transcript_id=0)
        for frame in reads.frames
    ]
    reads.frames[:] = frames
    sdata.points["transcripts"] = PointsModel.parse(
        reads.dataframe(),
        coordinates={"x": "x", "y": "y", "z": "z"},
        feature_key="gene",
        sort=True,
    )
    reads.reset()

    upgrade_spatialdata_contract_in_memory(
        sdata, platform="MERSCOPE", points_canonical=True
    )

    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1)
    frame = sdata.points["transcripts"].compute()
    assert frame["transcript_id"].tolist() == list(range(1, N_ROWS + 1))


def test_stable_transcript_ids_from_known_lengths_match_counted_ids() -> None:
    """Known partition lengths give the counted IDs without the counting pass."""
    reads = PartitionReads(split_frame(xenium_raw_points_frame(), N_PARTITIONS))
    counted = with_stable_transcript_ids(
        reads.dataframe(), preserve_existing=True
    ).compute()
    reads.reset()

    result = with_stable_transcript_ids(
        reads.dataframe(),
        preserve_existing=True,
        partition_lengths=[len(frame) for frame in reads.frames],
    )

    assert reads.total == 0
    pd.testing.assert_frame_equal(result.compute(), counted)
    assert reads.reads == dict.fromkeys(range(N_PARTITIONS), 1)


def test_stable_transcript_ids_refuse_wrong_partition_lengths() -> None:
    """Wrong lengths fail up front or when the partition is computed."""
    reads = PartitionReads(split_frame(xenium_raw_points_frame(), N_PARTITIONS))
    lengths = [len(frame) for frame in reads.frames]

    with pytest.raises(ValueError, match="entries"):
        with_stable_transcript_ids(reads.dataframe(), partition_lengths=lengths[:-1])
    with pytest.raises(ValueError, match="negative"):
        with_stable_transcript_ids(
            reads.dataframe(), partition_lengths=[-1, *lengths[1:]]
        )
    shifted = with_stable_transcript_ids(
        reads.dataframe(), partition_lengths=[lengths[0] - 1, *lengths[1:]]
    )
    with pytest.raises(ValueError, match="partition_lengths gave"):
        shifted.compute()


def test_stable_transcript_ids_known_length_for_pandas() -> None:
    """A pandas frame is one partition."""
    frame = xenium_raw_points_frame()

    result = with_stable_transcript_ids(frame, partition_lengths=[len(frame)])

    assert result["transcript_id"].tolist() == list(range(1, N_ROWS + 1))
    with pytest.raises(ValueError, match="does not match"):
        with_stable_transcript_ids(frame, partition_lengths=[len(frame) + 1])


def _parse_calls_without_sort(source: str) -> list[int]:
    """Return lines of ``PointsModel.parse`` calls that do not pass ``sort``.

    A call passes it with a ``sort=`` keyword, or through ``**kwargs`` built
    in the same function with a ``"sort"`` key.
    """
    tree = ast.parse(source)
    missing: list[int] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        sort_keys = {
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Constant) and node.value == "sort"
        }
        for call in ast.walk(function):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "parse"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "PointsModel"
            ):
                continue
            keywords = {keyword.arg for keyword in call.keywords}
            if "sort" in keywords or (None in keywords and sort_keys):
                continue
            missing.append(call.lineno)
    return missing


def test_every_points_parse_passes_sort() -> None:
    """Without ``sort`` spatialdata scans every partition's index (a full pass).

    The guard covers Dask input, where ``sort=True`` only skips that scan. In
    spatialdata 0.8.0 a pandas input cannot take ``sort`` at all (the model
    passes its own to ``from_pandas``), so a future pandas call site converts
    to Dask first or is exempted here with a reason.
    """
    source_root = Path(merxen.__file__).parent
    missing = {
        str(path.relative_to(source_root)): lines
        for path in sorted(source_root.rglob("*.py"))
        if (lines := _parse_calls_without_sort(path.read_text(encoding="utf-8")))
    }

    assert missing == {}


def test_stable_transcript_ids_per_partition_reads_match_the_counted_ids() -> None:
    """Partitions read with ``to_delayed`` keep their offsets, unequal lengths too."""
    frame = xenium_raw_points_frame()
    bounds = [0, 3, 10, 18, N_ROWS]
    reads = PartitionReads(
        [
            frame.iloc[start:stop].reset_index(drop=True)
            for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
        ]
    )
    lengths = [len(part) for part in reads.frames]
    expected = [
        list(range(start + 1, stop + 1))
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
    ]

    for partition_lengths in (None, lengths):
        result = with_stable_transcript_ids(
            reads.dataframe(), partition_lengths=partition_lengths
        )
        parts = [
            part.compute()["transcript_id"].tolist() for part in result.to_delayed()
        ]
        assert parts == expected


def test_normalize_points_canonical_casts_non_canonical_ids_as_by_default() -> None:
    """Signed, float or nullable IDs take the default casts even with the flag."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [3.0, 4.0, 5.0],
            "transcript_id": np.asarray([5, 6, 7], dtype=np.int64),
            "assignment": np.asarray([1.0, np.nan, 2.0]),
            "cell": pd.Series([1, pd.NA, 2], dtype="UInt64"),
        }
    )
    points = dd.from_pandas(frame, npartitions=2)

    default = normalize_points_for_latest_write(points.copy()).compute()
    fast = normalize_points_for_latest_write(
        points.copy(), points_canonical=True
    ).compute()

    pd.testing.assert_frame_equal(fast, default)
    assert fast["transcript_id"].dtype == "UInt64"
    assert fast["assignment"].dtype == "UInt32"
    assert fast["cell"].tolist() == [1, 0, 2]


def test_normalize_points_canonical_restores_only_the_feature_column() -> None:
    """``feature_categories`` applies to the feature key, not every gene column."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [3.0, 4.0, 5.0],
            "feature_name": pd.Categorical(["A", "B", "A"], ordered=True),
            "gene": pd.Categorical(["g1", "g2", "g1"]),
        }
    )
    points = PointsModel.parse(
        dd.from_pandas(frame, npartitions=2), feature_key="feature_name", sort=True
    )

    result = normalize_points_for_latest_write(
        points, points_canonical=True, feature_categories=["B", "A"]
    )

    computed = result.compute()
    assert computed["feature_name"].dtype == result["feature_name"].dtype
    assert computed["feature_name"].cat.categories.tolist() == ["B", "A"]
    assert not computed["feature_name"].cat.ordered
    assert computed["gene"].cat.categories.tolist() == ["g1", "g2"]


def test_points_canonical_refuses_unparsed_points() -> None:
    """Points without a transformation are not parsed points."""
    _, canonical, reads = _default_and_canonical("XENIUM")
    del canonical.points["transcripts"].attrs["transform"]

    with pytest.raises(SpatialDataContractError, match="is not parsed points"):
        prepare_source_spatialdata_contract(
            canonical, platform="XENIUM", points_canonical=True
        )
    assert MERXEN_SCHEMA_ATTR not in canonical.attrs
    assert reads.total == 0


def test_points_canonical_without_a_table_still_checks_the_metadata() -> None:
    """The no-table branch refuses non-canonical points too."""
    reference, _ = merscope_raw_spatialdata()
    prepare_source_spatialdata_contract(reference, platform="MERSCOPE")
    frame = reference.points["transcripts"].compute().reset_index(drop=True)
    raw, _ = merscope_raw_spatialdata()
    canonical, _ = canonical_spatialdata_like(
        reference, raw, frame=frame.astype({"transcript_id": "int64"})
    )
    del canonical.tables["table"]

    with pytest.raises(SpatialDataContractError, match="uint64"):
        prepare_source_spatialdata_contract(
            canonical, platform="MERSCOPE", points_canonical=True
        )


def test_points_canonical_refuses_a_leftover_source_column() -> None:
    """A source cell column next to canonical assignments is not canonical."""
    reference, _ = xenium_raw_spatialdata()
    prepare_source_spatialdata_contract(reference, platform="XENIUM")
    frame = reference.points["transcripts"].compute().reset_index(drop=True)
    frame["cell_id"] = frame["source_cell_id"]
    raw, _ = xenium_raw_spatialdata()
    canonical, _ = canonical_spatialdata_like(reference, raw, frame=frame)

    with pytest.raises(SpatialDataContractError, match="'cell_id'"):
        prepare_source_spatialdata_contract(
            canonical, platform="XENIUM", points_canonical=True
        )


def test_points_canonical_refuses_rank_numbered_cells() -> None:
    """Without label values or integer cell IDs the table numbers cells by rank."""
    _, canonical, _ = _default_and_canonical("XENIUM")
    table = canonical.tables["table"].copy()
    del table.obs["cell_labels"]
    table.uns.pop("spatialdata_attrs")
    canonical.tables["table"] = TableModel.parse(
        table, region="cell_labels", region_key="region", instance_key="cell_id"
    )

    with pytest.raises(SpatialDataContractError, match="sorted rank"):
        prepare_source_spatialdata_contract(
            canonical, platform="XENIUM", points_canonical=True
        )
