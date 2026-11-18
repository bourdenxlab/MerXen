"""Tests for canonical MERSCOPE transcript input."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import spatialdata as sd
from spatialdata.models import Image2DModel, PointsModel
from spatialdata.transformations import Identity

from merxen.config import MerscopeBuildConfig
from merxen.io.builders.merscope import (
    ProjectionBuildResult,
    _get_parquet_points,
    write_merscope_spatialdata,
)


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


@pytest.fixture
def writer_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Supply a tiny projection while exercising the real native-store writer."""
    raw = tmp_path / "slide" / "region_R1"
    images = raw / "images"
    images.mkdir(parents=True)
    np.savetxt(
        images / "micron_to_mosaic_pixel_transform.csv", np.diag([2.0, 2.0, 1.0])
    )
    projection = Image2DModel.parse(
        da.ones((2, 16, 16), chunks=(1, 8, 8), dtype=np.uint16),
        dims=("c", "y", "x"),
        c_coords=["DAPI", "PolyT"],
    )
    monkeypatch.setattr(
        "merxen.io.builders.merscope._load_merscope_projection_from_raw",
        lambda **kwargs: ProjectionBuildResult(projection),
    )
    return raw, tmp_path / "source.zarr"


def _reader_returns(monkeypatch: pytest.MonkeyPatch, result: sd.SpatialData) -> None:
    """Simulate the upstream reader succeeding with a partial SpatialData object."""
    upstream = types.ModuleType("spatialdata_io")
    upstream.merscope = lambda *args, **kwargs: result  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spatialdata_io", upstream)


@pytest.mark.parametrize("transcript_format", ["parquet", "csv", "parquet_bad_csv"])
def test_writer_restores_transcripts_omitted_by_upstream_reader(
    writer_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    transcript_format: str,
) -> None:
    """A successful reader must not silently drop supplied transcript files."""
    raw, destination = writer_inputs
    rows = pd.DataFrame(
        {
            "global_x": [3.5, 6.0],
            "global_y": [4.5, 7.0],
            "global_z": [1.0, 2.0],
            "gene": ["GeneA", "GeneB"],
            "cell_id": [101, -1],
            "transcript_id": ["vendor-a", "vendor-b"],
        }
    )
    if transcript_format == "csv":
        rows.to_csv(raw / "detected_transcripts.csv", index=False)
    else:
        rows.to_parquet(raw / "detected_transcripts.parquet")
        if transcript_format == "parquet_bad_csv":
            (raw / "detected_transcripts.csv").write_bytes(b"\x00broken export")
    _reader_returns(monkeypatch, sd.SpatialData())

    write_merscope_spatialdata(
        input_path=raw,
        output_path=destination,
        build_config=MerscopeBuildConfig(z_layers=[1]),
    )

    stored = sd.read_zarr(destination)
    assert list(stored.points) == ["slide_region_R1_transcripts"]
    points = stored.points["slide_region_R1_transcripts"].compute()
    np.testing.assert_array_equal(
        points[["x", "y", "z"]], rows[["global_x", "global_y", "global_z"]]
    )
    assert points["gene"].astype(str).tolist() == ["GeneA", "GeneB"]
    assert points["source_transcript_id"].tolist() == ["vendor-a", "vendor-b"]
    assert points["transcript_id"].dtype == np.dtype("uint64")
    assert points["transcript_id"].is_unique
    assert (
        stored.attrs["merxen_schema"]["primary_points"] == "slide_region_R1_transcripts"
    )
    from spatialdata.transformations import get_transformation

    affine = get_transformation(
        stored.points["slide_region_R1_transcripts"], to_coordinate_system="global"
    ).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    np.testing.assert_array_equal(affine, np.diag([2.0, 2.0, 1.0]))


def test_writer_keeps_transcripts_already_loaded_by_upstream_reader(
    writer_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing reader result takes precedence over an unused broken export."""
    raw, destination = writer_inputs
    (raw / "detected_transcripts.parquet").write_bytes(b"broken parquet")
    points = PointsModel.parse(
        pd.DataFrame({"x": [1.0], "y": [2.0], "gene": ["GeneC"]}),
        coordinates={"x": "x", "y": "y"},
        feature_key="gene",
    )
    _reader_returns(
        monkeypatch, sd.SpatialData(points={"existing_transcripts": points})
    )

    write_merscope_spatialdata(
        input_path=raw,
        output_path=destination,
        build_config=MerscopeBuildConfig(z_layers=[1]),
    )

    stored = sd.read_zarr(destination)
    assert list(stored.points) == ["existing_transcripts"]
    assert stored.points["existing_transcripts"].compute()["gene"].astype(
        str
    ).tolist() == ["GeneC"]


def test_writer_rejects_unreadable_omitted_transcripts_before_projection(
    writer_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt required input must fail before expensive image processing."""
    raw, destination = writer_inputs
    (raw / "detected_transcripts.parquet").write_bytes(b"broken parquet")
    _reader_returns(monkeypatch, sd.SpatialData())
    monkeypatch.setattr(
        "merxen.io.builders.merscope._load_merscope_projection_from_raw",
        lambda **kwargs: pytest.fail(
            "Projection must not run for unreadable transcripts"
        ),
    )

    with pytest.raises(pa.ArrowInvalid, match="[Pp]arquet"):
        write_merscope_spatialdata(
            input_path=raw,
            output_path=destination,
            build_config=MerscopeBuildConfig(z_layers=[1]),
        )

    assert not destination.exists()


def test_writer_rejects_both_readers_omitting_supplied_transcripts(
    writer_inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial results from both readers cannot be published as a complete build."""
    raw, destination = writer_inputs
    pd.DataFrame({"global_x": [1.0], "global_y": [2.0], "gene": ["GeneA"]}).to_parquet(
        raw / "detected_transcripts.parquet"
    )
    _reader_returns(monkeypatch, sd.SpatialData())
    monkeypatch.setattr(
        "merxen.io.builders.merscope.read_merscope_spatialdata_local",
        lambda **kwargs: sd.SpatialData(),
    )
    monkeypatch.setattr(
        "merxen.io.builders.merscope._load_merscope_projection_from_raw",
        lambda **kwargs: pytest.fail("Projection must not run with missing points"),
    )

    with pytest.raises(ValueError, match="Supplied transcript files"):
        write_merscope_spatialdata(
            input_path=raw,
            output_path=destination,
            build_config=MerscopeBuildConfig(z_layers=[1]),
        )

    assert not destination.exists()
