"""Viewer ROI exchange keeps axes, frame, source identity and tile margins explicit."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merxen.segmentation.hitl.regions import (
    RoiContext,
    read_viewer_regions,
    write_viewer_regions,
)
from merxen.segmentation.hitl.viewer import validate_native_viewer_image


@pytest.fixture
def context() -> RoiContext:
    return RoiContext(
        platform="ATERA",
        image_key="morphology_focus",
        dataset_alias="ATERA_PILOT",
        height=2048,
        width=2048,
        pixel_size_um=0.21249222,
        tile_size=1024,
        overlap_px=64,
        margin_px=32,
        crop_size_px=256,
        source_metadata_sha256="1" * 64,
        baseline_stats_sha256="2" * 64,
    )


def _rectangle(y: int, x: int, size: int = 256) -> np.ndarray:
    return np.array([[y, x], [y, x + size], [y + size, x + size], [y + size, x]], float)


def test_roi_roundtrip_records_xy_but_returns_native_yx(
    context: RoiContext, tmp_path: Path
) -> None:
    path = tmp_path / "regions.geojson"
    write_viewer_regions(path, context, [_rectangle(100, 200), _rectangle(1100, 1200)])
    payload = json.loads(path.read_text())
    assert payload["features"][0]["geometry"]["coordinates"][0][0] == [200, 100]
    regions = read_viewer_regions(path, context)
    assert [(roi.y0, roi.x0, roi.size) for roi in regions] == [
        (100, 200, 256),
        (1100, 1200, 256),
    ]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_viewer_regions(path, context, [_rectangle(200, 200)])
    assert path.read_bytes() == before
    saved = tmp_path / "context.json"
    saved.write_text(json.dumps(asdict(context)))
    assert RoiContext.read(saved) == context
    assert context.placement_windows()


@pytest.mark.parametrize(
    "change",
    [
        {"pixel_size_um": 0.25},
        {"source_metadata_sha256": "3" * 64},
        {"baseline_stats_sha256": "4" * 64},
        {"margin_px": 64},
    ],
)
def test_rois_from_another_source_or_grid_are_refused(
    context: RoiContext, tmp_path: Path, change: dict
) -> None:
    path = tmp_path / "regions.geojson"
    write_viewer_regions(path, context, [_rectangle(100, 200)])
    with pytest.raises(ValueError, match="different source"):
        read_viewer_regions(path, replace(context, **change))


@pytest.mark.parametrize(
    "rectangles",
    [
        [_rectangle(10, 200)],  # margin
        [_rectangle(700, 200)],  # crosses a global tile boundary
        [_rectangle(-1600, 200)],  # negative tile indices must not be accepted
        [_rectangle(100, 200), _rectangle(200, 300)],  # overlap
        [np.array([[100, 200], [200, 300], [300, 200], [200, 100]], float)],  # rotated
    ],
)
def test_invalid_or_overlapping_regions_are_refused(
    context: RoiContext, tmp_path: Path, rectangles: list[np.ndarray]
) -> None:
    path = tmp_path / "regions.geojson"
    with pytest.raises(ValueError):
        write_viewer_regions(path, context, rectangles)
    assert not path.exists()


def test_ambiguous_geojson_frame_is_refused(
    context: RoiContext, tmp_path: Path
) -> None:
    path = tmp_path / "regions.geojson"
    write_viewer_regions(path, context, [_rectangle(100, 200)])
    payload = json.loads(path.read_text())
    payload["coordinate_system"] = "native_um"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="native-pixel GeoJSON"):
        read_viewer_regions(path, context)


def _image(
    context: RoiContext, matrix: np.ndarray, origin: tuple[float, float] = (0, 0)
) -> SimpleNamespace:
    return SimpleNamespace(
        name="ATERA morphology_focus DAPI",
        ndim=2,
        multiscale=False,
        rgb=False,
        data=np.zeros((512, 512)),
        data_to_world=lambda point: np.asarray(origin) + matrix @ point,
    )


def test_native_micron_frame_accepts_a_downsampled_viewer_image(
    context: RoiContext,
) -> None:
    validate_native_viewer_image(
        _image(context, np.eye(2) * context.pixel_size_um * 4), context
    )


@pytest.mark.parametrize(
    "matrix,origin",
    [
        (np.eye(2) * 4, (0, 0)),  # native pixels instead of microns
        (np.eye(2) * 0.86, (0, 0)),  # slightly rescaled full image
        (np.eye(2) * 0.84996888, (100, 0)),  # translated paired view
        (np.diag([-0.84996888, 0.84996888]), (0, 0)),  # reflected
        (np.array([[0, 0.84996888], [0.84996888, 0]]), (0, 0)),  # axes exchanged
    ],
)
def test_non_native_viewer_frames_are_refused(
    context: RoiContext, matrix: np.ndarray, origin: tuple[float, float]
) -> None:
    with pytest.raises(ValueError, match="native dataset microns"):
        validate_native_viewer_image(_image(context, matrix, origin), context)


def test_another_platforms_morphology_image_is_refused(context: RoiContext) -> None:
    image = _image(context, np.eye(2) * context.pixel_size_um * 4)
    image.name = "XENIUM morphology_focus DAPI"
    with pytest.raises(ValueError, match="native morphology_focus"):
        validate_native_viewer_image(image, context)
