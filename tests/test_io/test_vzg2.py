"""Tests for direct Vizgen VZG2 ingestion."""

from __future__ import annotations

import io
import json
import struct
import tarfile
from pathlib import Path

import geopandas as gpd
import imagecodecs
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from merxen.config import MerscopeBuildConfig
from merxen.io.builders.vzg2 import (
    Vzg2Archive,
    _decode_vzg2_cell_tile,
    _manifest_transform,
    read_vzg2_spatialdata,
    resolve_vzg2_path,
    write_vzg2_spatialdata,
)
from merxen.io.image_source import MERSCOPE_ZPROJ_IMAGE_NAME, _get_image_dataarray


def test_resolve_vzg2_path_accepts_file_and_unambiguous_folder(
    tmp_path: Path,
) -> None:
    """A caller can pass either the archive itself or its containing folder."""
    archive_path = tmp_path / "sample.vzg2"
    archive_path.touch()

    assert resolve_vzg2_path(archive_path) == archive_path
    assert resolve_vzg2_path(tmp_path) == archive_path

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "manifest.json").write_text("{}")
    assert resolve_vzg2_path(tmp_path) == archive_path

    (images_dir / "mosaic_DAPI_z3.tif").touch()
    assert resolve_vzg2_path(tmp_path) is None
    (images_dir / "mosaic_DAPI_z3.tif").unlink()

    (tmp_path / "second.vzg2").touch()
    with pytest.raises(ValueError, match="Multiple .vzg2 files"):
        resolve_vzg2_path(tmp_path)


def test_decode_vzg2_cell_tile_recovers_vpt_lod0_coordinates() -> None:
    """The local decoder should reverse Vizgen's public LOD0 polygon packing."""
    points = _rectangle_vertices(x0=2, y0=3, width=12, height=10)
    tile = _cell_tile([points], [7])

    coordinates, ordinals = _decode_vzg2_cell_tile(
        tile,
        tile_number=1,
        grid=(2, 1),
        image_shape=(32, 40),
    )

    np.testing.assert_array_equal(ordinals, [7])
    np.testing.assert_array_equal(coordinates[0], points + np.array([20, 0]))


def test_manifest_transform_does_not_apply_bbox_origin_as_translation() -> None:
    """Vizualizer bbox padding must not offset canonical transcript coordinates."""
    transform = _manifest_transform(
        {
            "mosaic_width_pixels": 100,
            "mosaic_height_pixels": 200,
            "bbox_microns": [-10.0, -20.0, 40.0, 80.0],
        }
    )

    np.testing.assert_allclose(
        transform,
        np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def test_read_vzg2_spatialdata_builds_recoverable_source(tmp_path: Path) -> None:
    """Images, original polygons, and cell counts should load from one archive."""
    archive_path = _write_synthetic_vzg2(tmp_path / "sample.vzg2")

    sdata = read_vzg2_spatialdata(
        Vzg2Archive.open(archive_path),
        build_config=MerscopeBuildConfig(z_layers=[3]),
    )

    assert list(sdata.images) == [MERSCOPE_ZPROJ_IMAGE_NAME]
    assert len(sdata.shapes) == 1
    assert len(next(iter(sdata.shapes.values()))) == 2
    assert sdata.tables["table"].shape == (2, 2)
    assert list(sdata.tables["table"].var_names) == ["GeneA", "GeneB"]
    assert sdata.tables["table"].obsm["blank"].shape == (2, 1)
    assert sdata.points == {}
    assert sdata.attrs["merxen_vzg2"]["transcript_points_available"] is False

    image = _get_image_dataarray(sdata.images[MERSCOPE_ZPROJ_IMAGE_NAME])
    values = np.asarray(image.data.compute())
    assert values.shape == (2, 32, 32)
    assert np.all(values[0] == 1200)
    assert np.all(values[1] == 300)


def test_read_vzg2_uses_sibling_detected_transcript_parquet(tmp_path: Path) -> None:
    """A partial export can combine canonical points with its VZG2 image."""
    source_dir = tmp_path / "region_R1"
    source_dir.mkdir()
    archive_path = _write_synthetic_vzg2(source_dir / "sample.vzg2")
    pd.DataFrame(
        {
            "global_x": [3.5, 18.0],
            "global_y": [4.5, 20.0],
            "global_z": [3, 3],
            "gene": ["GeneA", "GeneB"],
            "transcript_id": ["tx-1", "tx-2"],
            "cell_id": [101, -1],
        }
    ).to_parquet(source_dir / "detected_transcripts.parquet")

    sdata = read_vzg2_spatialdata(
        Vzg2Archive.open(archive_path),
        build_config=MerscopeBuildConfig(z_layers=[3]),
        external_source_dir=source_dir,
    )

    assert len(sdata.points) == 1
    points = next(iter(sdata.points.values())).compute()
    assert list(points["transcript_id"]) == ["tx-1", "tx-2"]
    assert list(points["gene"].astype(str)) == ["GeneA", "GeneB"]
    assert sdata.attrs["merxen_vzg2"]["transcript_points_available"] is True
    assert sdata.attrs["merxen_vzg2"]["transcript_source"].endswith(
        "detected_transcripts.parquet"
    )


def test_read_vzg2_prefers_complete_sibling_cell_boundaries(tmp_path: Path) -> None:
    """Canonical boundaries should replace the potentially lossy packed layer."""
    source_dir = tmp_path / "region_R1"
    source_dir.mkdir()
    archive_path = _write_synthetic_vzg2(source_dir / "sample.vzg2")
    bowtie = Polygon([(1, 1), (6, 6), (1, 6), (6, 1), (1, 1)])
    canonical = gpd.GeoDataFrame(
        {
            "EntityID": [101, 102],
            "ZIndex": [0, 0],
            "geometry": [
                MultiPolygon([bowtie]),
                MultiPolygon([Polygon([(20, 20), (28, 20), (28, 28), (20, 28)])]),
            ],
        },
        geometry="geometry",
    )
    canonical.to_parquet(source_dir / "cell_boundaries.parquet")

    sdata = read_vzg2_spatialdata(
        Vzg2Archive.open(archive_path),
        build_config=MerscopeBuildConfig(z_layers=[3]),
        external_source_dir=source_dir,
    )

    shapes = next(iter(sdata.shapes.values()))
    assert list(shapes.index) == ["101", "102"]
    assert bool(shapes.geometry.is_valid.all())
    assert float(shapes.loc["102"].geometry.area) == 64.0


def test_write_vzg2_spatialdata_persists_transform_sidecar(tmp_path: Path) -> None:
    """A direct VZG2 build should write the downstream MERSCOPE transform."""
    archive_path = _write_synthetic_vzg2(tmp_path / "sample.vzg2")
    output_path = tmp_path / "source.zarr"

    result = write_vzg2_spatialdata(
        input_path=archive_path,
        output_path=output_path,
        build_config=MerscopeBuildConfig(z_layers=[3]),
    )

    assert result == output_path
    transform_path = output_path / "micron_to_mosaic_pixel_transform.csv"
    assert transform_path.is_file()
    np.testing.assert_allclose(np.loadtxt(transform_path), np.eye(3))


def test_write_vzg2_folder_persists_sibling_transcripts(tmp_path: Path) -> None:
    """The writer should retain canonical transcripts stored beside VZG2."""
    import spatialdata as sd

    source_dir = tmp_path / "region_R1"
    source_dir.mkdir()
    _write_synthetic_vzg2(source_dir / "sample.vzg2")
    pd.DataFrame(
        {
            "": [17],
            "global_x": [3.5],
            "global_y": [4.5],
            "global_z": [3],
            "gene": ["GeneA"],
            "transcript_id": ["tx-1"],
            "cell_id": [101],
        }
    ).to_parquet(source_dir / "detected_transcripts.parquet")
    output_path = tmp_path / "source.zarr"

    write_vzg2_spatialdata(
        input_path=source_dir,
        output_path=output_path,
        build_config=MerscopeBuildConfig(z_layers=[3]),
    )

    written = sd.read_zarr(output_path)
    assert len(written.points) == 1
    points = next(iter(written.points.values())).compute()
    assert list(points["transcript_id"]) == [1]
    assert list(points["source_transcript_id"]) == ["tx-1"]
    assert "" not in points.columns


def _write_synthetic_vzg2(path: Path) -> Path:
    image_poly_t = np.full((32, 32), 1200, dtype=np.uint16)
    image_dapi = np.full((32, 32), 300, dtype=np.uint16)
    manifest = {
        "name": "synthetic_region_0",
        "version": "3.1.0",
        "mosaic_width_pixels": 32,
        "mosaic_height_pixels": 32,
        "bbox_microns": [0.0, 0.0, 32.0, 32.0],
        "images": {
            "3": {
                "path": "images/z-plane-3",
                "channels": ["PolyT", "DAPI"],
            }
        },
        "features": [{"path": "cells_v5", "name": "Cells"}],
        "planes_count": 7,
    }
    zarray = {
        "zarr_format": 2,
        "shape": [1, 1, 2, 32, 32],
        "chunks": [1, 1, 1, 32, 32],
        "dtype": "<u2",
        "compressor": {"id": "jpeg12"},
        "fill_value": 0,
        "order": "C",
        "filters": None,
    }
    cell_ids = [101, 102]
    counts = pd.DataFrame(
        {
            "GeneA": [2.0, 0.0],
            "GeneB": [1.0, 4.0],
            "Blank-0": [0.0, 1.0],
        },
        index=cell_ids,
    )
    members: dict[str, bytes] = {
        "manifest.json": json.dumps(manifest).encode(),
        "images/z-plane-3/data.zarr/0/0/.zarray": json.dumps(zarray).encode(),
        "images/z-plane-3/data.zarr/0/0/0.0.0.0.0": imagecodecs.jpeg_encode(
            image_poly_t, bitspersample=12
        ),
        "images/z-plane-3/data.zarr/0/0/0.0.1.0.0": imagecodecs.jpeg_encode(
            image_dapi, bitspersample=12
        ),
        "cells_v5/manifest_cells.json": json.dumps(
            {"version": "5", "tiles": [[1, 1], [1, 1], [1, 1]]}
        ).encode(),
        "cells_v5/cells_names_array.json": json.dumps(cell_ids).encode(),
        "cells_v5/z-plane0/Lod0/tile0.bin": _cell_tile(
            [
                _rectangle_vertices(x0=2, y0=3, width=12, height=10),
                _rectangle_vertices(x0=16, y0=18, width=12, height=10),
            ],
            [0, 1],
        ),
        "cells_v5/cell_by_gene/cell_by_gene.parquet": _parquet_bytes(counts),
        "cells_v5/cells_indices.parquet": _parquet_bytes(
            pd.DataFrame({"cell_indices": cell_ids}), index=False
        ),
        "cells_v5/cells_centers.parquet": _parquet_bytes(
            pd.DataFrame({"center_x": [6.0, 20.5], "center_y": [6.5, 22.0]}),
            index=False,
        ),
        "cells_v5/cells_volume.parquet": _parquet_bytes(
            pd.DataFrame({"volume": [100.0, 125.0]}), index=False
        ),
    }
    with tarfile.open(path, mode="w") as archive:
        for name, value in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    return path


def _parquet_bytes(frame: pd.DataFrame, *, index: bool = True) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=index)
    return buffer.getvalue()


def _rectangle_vertices(
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    top = [(x, y0) for x in range(x0, x0 + width + 1)]
    right = [(x0 + width, y) for y in range(y0 + 1, y0 + height + 1)]
    bottom = [(x, y0 + height) for x in range(x0 + width - 1, x0 - 1, -1)]
    left = [(x0, y) for y in range(y0 + height - 1, y0, -1)]
    perimeter = top + right + bottom + left
    assert len(perimeter) == 44
    return np.asarray(perimeter, dtype=np.uint32)


def _cell_tile(polygons: list[np.ndarray], ordinals: list[int]) -> bytes:
    records = bytearray()
    for polygon in polygons:
        base_x = int(polygon[:, 0].min())
        base_y = int(polygon[:, 1].min())
        records.extend(struct.pack("<HH", base_y, base_x))
        deltas = ((polygon[:, 0] - base_x).astype(np.uint32) << 12) | (
            polygon[:, 1] - base_y
        ).astype(np.uint32)
        for start in range(0, 44, 4):
            first, second, third, fourth = (
                int(value) for value in deltas[start : start + 4]
            )
            word0 = (first << 8) | (second >> 16)
            word1 = ((second & 0xFFFF) << 16) | (third >> 8)
            word2 = ((third & 0xFF) << 24) | fourth
            records.extend(struct.pack("<III", word0, word1, word2))
    header = struct.pack("<III", 5, 0, len(polygons))
    cell_map = np.asarray(ordinals, dtype="<u4").tobytes()
    return header + bytes(records) + cell_map
