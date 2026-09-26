"""Tests for direct Vizgen VZG2 ingestion."""

from __future__ import annotations

import io
import json
import logging
import struct
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import geopandas as gpd
import imagecodecs
import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon

from merxen.config import DatasetConfig, MerscopeBuildConfig, SegmentationConfig
from merxen.io.builders.merscope import write_merscope_spatialdata
from merxen.io.builders.vzg2 import (
    Vzg2Archive,
    _decode_vzg2_cell_tile,
    _manifest_transform,
    read_vzg2_spatialdata,
    resolve_vzg2_path,
    write_vzg2_spatialdata,
)
from merxen.io.image_source import MERSCOPE_ZPROJ_IMAGE_NAME, _get_image_dataarray
from merxen.io.transcript_io import write_proseg_csv_from_points
from merxen.segmentation.cellpose import build_cellpose_affine_to_microns
from merxen.segmentation.pipeline import _load_dataset_sdata

# Synthetic mosaic for the registration tests: 256 px at 2 px/um.
_MOSAIC_PX = 256
_PX_PER_UM = 2.0
# VZG2-style manifest whose micron bounding box starts below zero, like the
# Vizgen MsBrain VZG2 region (bbox minimum -21.86, -111.78 um).
_NEGATIVE_BBOX = [-12.8, -25.6, 115.2, 102.4]
# P1212 / P7513 / ag7 style: the mosaic starts at the micron origin.
_ZERO_BBOX = [0.0, 0.0, 128.0, 128.0]
_CELL_CENTRES_UM = np.array(
    [
        [5.0, 0.0],
        [25.0, -5.0],
        [45.0, 3.0],
        [70.0, -2.0],
        [90.0, 6.0],
        [10.0, 25.0],
        [35.0, 30.0],
        [60.0, 22.0],
        [85.0, 28.0],
        [15.0, 55.0],
        [40.0, 60.0],
        [65.0, 52.0],
        [88.0, 62.0],
        [20.0, 78.0],
        [50.0, 80.0],
        [75.0, 76.0],
    ]
)
_TRANSCRIPTS_PER_CELL = 30


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


def test_manifest_transform_maps_bbox_origin_to_mosaic_origin() -> None:
    """The inverse transform should restore the manifest's micron origin."""
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
                [2.0, 0.0, 20.0],
                [0.0, 2.0, 40.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(np.linalg.inv(transform)[:2, 2], [-10.0, -20.0])


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


def _vizgen_micron_to_mosaic(bbox_microns: list[float]) -> np.ndarray:
    """Vizgen's micron-to-mosaic transform: pixel zero is the bbox minimum."""
    return np.array(
        [
            [_PX_PER_UM, 0.0, -_PX_PER_UM * bbox_microns[0]],
            [0.0, _PX_PER_UM, -_PX_PER_UM * bbox_microns[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def _dapi_mosaic(centres_um: np.ndarray, micron_to_mosaic: np.ndarray) -> np.ndarray:
    """Draw one bright nucleus per cell where the instrument imaged it."""
    image = np.full((_MOSAIC_PX, _MOSAIC_PX), 100, dtype=np.uint16)
    rows, cols = np.mgrid[0:_MOSAIC_PX, 0:_MOSAIC_PX]
    pixels = (micron_to_mosaic @ np.c_[centres_um, np.ones(len(centres_um))].T).T
    for x_px, y_px, _ in pixels:
        image[(cols - x_px) ** 2 + (rows - y_px) ** 2 <= 36.0] = 3000
    return image


def _write_transcripts(source_dir: Path, centres_um: np.ndarray) -> None:
    """Write detected_transcripts.parquet clustered on the cell centres."""
    rng = np.random.default_rng(0)
    cell_xy = np.repeat(centres_um, _TRANSCRIPTS_PER_CELL, axis=0) + rng.normal(
        0.0, 0.8, size=(len(centres_um) * _TRANSCRIPTS_PER_CELL, 2)
    )
    background = rng.uniform(
        centres_um.min(axis=0) - 5.0, centres_um.max(axis=0) + 5.0, size=(40, 2)
    )
    xy = np.vstack([cell_xy, background])
    pd.DataFrame(
        {
            "global_x": xy[:, 0],
            "global_y": xy[:, 1],
            "global_z": np.full(len(xy), 3.0),
            "gene": np.where(np.arange(len(xy)) % 2 == 0, "GeneA", "GeneB"),
            "transcript_id": [f"tx-{index}" for index in range(len(xy))],
        }
    ).to_parquet(source_dir / "detected_transcripts.parquet")


def _segment_like_cellpose(
    zarr_path: Path,
    tmp_path: Path,
    transform_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, Any, tuple[float, float, float], tuple]:
    """Segment the built image and map the mask centroids to microns.

    Uses the segmentation stage's own loader and mask-to-micron affine; only
    the Cellpose model is replaced by a threshold, so no GPU is needed.
    """
    config = SegmentationConfig(
        dataset=DatasetConfig(
            name="SYNTHETIC_MERSCOPE",
            platform="MERSCOPE",
            data_path=zarr_path,
            channels=["DAPI"],
            output_dir=tmp_path / "segment_out",
            transform_path=transform_path,
        )
    )
    _, fetch_tile_fn, height, width, matrix, points = _load_dataset_sdata(config)
    tile = fetch_tile_fn(0, height, 0, width)[..., 0]
    masks, n_labels = ndimage.label(tile > 1500)
    x_transform, y_transform = build_cellpose_affine_to_microns(matrix, 1.0)
    rows_cols = np.asarray(
        ndimage.center_of_mass(masks > 0, masks, range(1, n_labels + 1))
    )
    px, py = rows_cols[:, 1], rows_cols[:, 0]
    centroids_um = np.c_[
        x_transform[0] * px + x_transform[1] * py + x_transform[2],
        y_transform[0] * px + y_transform[1] * py + y_transform[2],
    ]
    return masks, centroids_um, points, x_transform, y_transform


def _nearest_offsets(centroids_um: np.ndarray, centres_um: np.ndarray) -> np.ndarray:
    """Offset of every true cell centre's nearest mask centroid (mask - truth)."""
    distances = np.linalg.norm(
        centroids_um[None, :, :] - centres_um[:, None, :], axis=2
    )
    return centroids_um[distances.argmin(axis=1)] - centres_um


def _build_vzg2_region(
    tmp_path: Path,
    *,
    bbox_microns: list[float],
    centres_um: np.ndarray,
    override: np.ndarray | None,
) -> tuple[Path, Path | None]:
    source_dir = tmp_path / "region_R1"
    source_dir.mkdir()
    _write_synthetic_vzg2(
        source_dir / "sample.vzg2",
        bbox_microns=bbox_microns,
        image_dapi=_dapi_mosaic(centres_um, _vizgen_micron_to_mosaic(bbox_microns)),
    )
    _write_transcripts(source_dir, centres_um)
    override_path = None
    if override is not None:
        (source_dir / "images").mkdir()
        override_path = source_dir / "images" / "micron_to_mosaic_pixel_transform.csv"
        # Vizgen writes float32 values; keep that rounding in the fixture.
        np.savetxt(override_path, override.astype(np.float32))
    zarr_path = tmp_path / "source.zarr"
    write_vzg2_spatialdata(
        input_path=source_dir,
        output_path=zarr_path,
        build_config=MerscopeBuildConfig(z_layers=[3]),
        transform_path_override=override_path,
    )
    return zarr_path, override_path


@pytest.mark.parametrize("use_override", [True, False], ids=["override", "manifest"])
def test_vzg2_negative_bbox_mask_centroids_land_on_transcripts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    use_override: bool,
) -> None:
    """Regression for the VZG2 offset: a negative bbox origin must be kept.

    The 2026-09-08 VZG2 build used a transform without the bounding-box
    translation, which put every Cellpose/ProSeg cell (+21.9, +111.8) um away
    from its transcripts.
    """
    override = _vizgen_micron_to_mosaic(_NEGATIVE_BBOX) if use_override else None
    with caplog.at_level(logging.WARNING, logger="merxen.io.builders.vzg2"):
        zarr_path, override_path = _build_vzg2_region(
            tmp_path,
            bbox_microns=_NEGATIVE_BBOX,
            centres_um=_CELL_CENTRES_UM,
            override=override,
        )
    assert "disagrees with the VZG2 manifest" not in caplog.text

    masks, centroids_um, points, x_transform, y_transform = _segment_like_cellpose(
        zarr_path, tmp_path, override_path
    )

    assert len(centroids_um) == len(_CELL_CENTRES_UM)
    np.testing.assert_allclose(
        [x_transform[2], y_transform[2]], _NEGATIVE_BBOX[:2], atol=1e-4
    )
    offsets = _nearest_offsets(centroids_um, _CELL_CENTRES_UM)
    assert np.abs(offsets).max() < 0.5

    # ProSeg input: transcripts are seeded with the mask they were drawn in.
    proseg_csv = write_proseg_csv_from_points(
        points,
        tmp_path / "transcripts_for_proseg.csv",
        masks,
        x_transform,
        y_transform,
        "x",
        "y",
        "z",
        "gene",
    )
    seeded = pd.read_csv(proseg_csv["csv_path"])
    xy = seeded[["x_micron", "y_micron"]].to_numpy()
    distance = np.linalg.norm(xy[:, None, :] - _CELL_CENTRES_UM[None, :, :], axis=2)
    near = distance.min(axis=1) < 2.0
    owner = distance.argmin(axis=1)[near]
    labels = seeded.loc[near, "cell_id"].to_numpy()
    assert near.sum() >= 0.9 * len(_CELL_CENTRES_UM) * _TRANSCRIPTS_PER_CELL
    assert (labels > 0).all()
    labels_per_cell = pd.Series(labels).groupby(owner).nunique()
    assert len(labels_per_cell) == len(_CELL_CENTRES_UM)
    assert (labels_per_cell == 1).all()
    assert pd.Series(labels).groupby(owner).first().is_unique


def test_vzg2_stale_zero_translation_override_warns_and_reproduces_offset(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The stale override of the 2026-09-08 build drops the bbox origin."""
    stale = np.diag([_PX_PER_UM, _PX_PER_UM, 1.0])
    with caplog.at_level(logging.WARNING, logger="merxen.io.builders.vzg2"):
        zarr_path, override_path = _build_vzg2_region(
            tmp_path,
            bbox_microns=_NEGATIVE_BBOX,
            centres_um=_CELL_CENTRES_UM,
            override=stale,
        )
    assert "disagrees with the VZG2 manifest by up to 25.60 um" in caplog.text

    _, centroids_um, _, _, _ = _segment_like_cellpose(
        zarr_path, tmp_path, override_path
    )

    # Segmentation minus truth equals minus the bbox minimum, as in VZG2.
    expected_offset = -np.asarray(_NEGATIVE_BBOX[:2])
    shifted = _nearest_offsets(centroids_um, _CELL_CENTRES_UM + expected_offset)
    assert np.abs(shifted).max() < 0.5
    unshifted = _nearest_offsets(centroids_um, _CELL_CENTRES_UM)
    assert np.median(np.linalg.norm(unshifted, axis=1)) > 3.0


def test_vzg2_build_records_transform_source(tmp_path: Path) -> None:
    import spatialdata as sd

    zarr_path, _ = _build_vzg2_region(
        tmp_path,
        bbox_microns=_NEGATIVE_BBOX,
        centres_um=_CELL_CENTRES_UM,
        override=None,
    )

    attrs = sd.read_zarr(zarr_path).attrs["merxen_vzg2"]
    assert attrs["transform_source"] == "manifest"
    np.testing.assert_allclose(
        np.loadtxt(zarr_path / "micron_to_mosaic_pixel_transform.csv"),
        _vizgen_micron_to_mosaic(_NEGATIVE_BBOX),
    )


@pytest.mark.parametrize("use_override", [True, False], ids=["override", "manifest"])
def test_vzg2_zero_bbox_keeps_zero_offset(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    use_override: bool,
) -> None:
    """P1212/P7513/ag7-style archives start at the micron origin."""
    centres = _CELL_CENTRES_UM + np.array([15.0, 20.0])
    override = _vizgen_micron_to_mosaic(_ZERO_BBOX) if use_override else None
    with caplog.at_level(logging.WARNING, logger="merxen.io.builders.vzg2"):
        zarr_path, override_path = _build_vzg2_region(
            tmp_path, bbox_microns=_ZERO_BBOX, centres_um=centres, override=override
        )
    assert "disagrees with the VZG2 manifest" not in caplog.text

    _, centroids_um, _, x_transform, y_transform = _segment_like_cellpose(
        zarr_path, tmp_path, override_path
    )

    assert (x_transform[2], y_transform[2]) == (0.0, 0.0)
    assert np.abs(_nearest_offsets(centroids_um, centres)).max() < 0.5


def test_canonical_merscope_export_keeps_zero_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical exports (P1212/P7513/ag7) use the raw zero-origin CSV."""
    # Force the local reader so the test does not depend on spatialdata_io.
    monkeypatch.setitem(
        sys.modules, "spatialdata_io", types.ModuleType("spatialdata_io")
    )
    centres = _CELL_CENTRES_UM + np.array([15.0, 20.0])
    transform = _vizgen_micron_to_mosaic(_ZERO_BBOX)
    raw_dir = tmp_path / "region_R1"
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True)
    tifffile.imwrite(
        images_dir / "mosaic_DAPI_z0.tif", _dapi_mosaic(centres, transform)
    )
    tifffile.imwrite(
        images_dir / "mosaic_PolyT_z0.tif",
        np.full((_MOSAIC_PX, _MOSAIC_PX), 500, dtype=np.uint16),
    )
    # Vizgen writes the zero translation as -0.0.
    (images_dir / "micron_to_mosaic_pixel_transform.csv").write_text(
        "2.0 0.0 -0.0\n0.0 2.0 -0.0\n0.0 0.0 1.0\n"
    )
    _write_transcripts(raw_dir, centres)
    zarr_path = tmp_path / "source.zarr"

    write_merscope_spatialdata(
        input_path=raw_dir,
        output_path=zarr_path,
        build_config=MerscopeBuildConfig(z_layers=[0]),
    )
    _, centroids_um, _, x_transform, y_transform = _segment_like_cellpose(
        zarr_path, tmp_path, None
    )

    np.testing.assert_allclose(
        np.loadtxt(zarr_path / "micron_to_mosaic_pixel_transform.csv"), transform
    )
    assert (x_transform[2], y_transform[2]) == (0.0, 0.0)
    assert len(centroids_um) == len(centres)
    assert np.abs(_nearest_offsets(centroids_um, centres)).max() < 0.5


def _write_synthetic_vzg2(
    path: Path,
    *,
    bbox_microns: list[float] | None = None,
    image_dapi: np.ndarray | None = None,
) -> Path:
    size = 32 if image_dapi is None else int(image_dapi.shape[0])
    image_poly_t = np.full((size, size), 1200, dtype=np.uint16)
    if image_dapi is None:
        image_dapi = np.full((size, size), 300, dtype=np.uint16)
    manifest = {
        "name": "synthetic_region_0",
        "version": "3.1.0",
        "mosaic_width_pixels": size,
        "mosaic_height_pixels": size,
        "bbox_microns": bbox_microns or [0.0, 0.0, 32.0, 32.0],
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
        "shape": [1, 1, 2, size, size],
        "chunks": [1, 1, 1, size, size],
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
