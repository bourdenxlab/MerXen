"""Tests for annotation-derived VALIS tissue masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

from merxen.alignment.frames import DapiFrame, _prepare_one_frame
from merxen.alignment.register import load_required_valis_tissue_annotations
from merxen.alignment.tissue import (
    load_alignment_tissue_annotation,
    rasterize_alignment_tissue_annotation,
)
from merxen.config import AlignmentConfig, DAPIProcessingConfig


def _feature(
    geometry_type: str,
    coordinates: object,
    *,
    role: str,
    tissue_piece_id: str | None = None,
) -> dict[str, object]:
    properties: dict[str, str] = {"role": role}
    if tissue_piece_id is not None:
        properties["tissue_piece_id"] = tissue_piece_id
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": geometry_type, "coordinates": coordinates},
    }


def _write_annotation(path: Path, *, include_edge: bool = True) -> Path:
    features = [
        _feature(
            "LineString",
            [[0.0, 10.0], [8.0, 10.0]],
            role="pia",
            tissue_piece_id="left",
        ),
        _feature(
            "LineString",
            [[12.0, 10.0], [20.0, 10.0]],
            role="pial_boundary",
            tissue_piece_id="right",
        ),
        # These depth-only landmarks deliberately have no matching pia. The
        # alignment loader must ignore them rather than inventing a mask piece.
        _feature(
            "LineString",
            [[1.0, 18.0], [7.0, 18.0]],
            role="wm",
            tissue_piece_id="orphan_depth_landmark",
        ),
        _feature(
            "Polygon",
            [[[14.0, 14.0], [16.0, 14.0], [16.0, 16.0], [14.0, 16.0], [14.0, 14.0]]],
            role="exclusion",
            tissue_piece_id="global_exclusion",
        ),
    ]
    if include_edge:
        features.append(
            _feature(
                "LineString",
                [
                    [0.0, 10.0],
                    [0.0, 20.0],
                    [8.0, 20.0],
                    [8.0, 10.0],
                    [12.0, 10.0],
                    [12.0, 20.0],
                    [20.0, 20.0],
                    [20.0, 10.0],
                ],
                role="tissue_edge",
            )
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_alignment_annotation_unions_pial_pieces_and_subtracts_exclusions(
    tmp_path: Path,
) -> None:
    """Only pia/edge/exclusion geometry should contribute to the full mask."""
    annotation = load_alignment_tissue_annotation(
        _write_annotation(tmp_path / "combined.geojson"),
        platform="XENIUM",
    )

    assert annotation.pial_piece_count == 2
    assert annotation.exclusion_count == 1
    assert annotation.polygon.area == pytest.approx(156.0)
    assert annotation.polygon.contains(Point(4.0, 15.0))
    assert annotation.polygon.contains(Point(18.0, 15.0))
    assert not annotation.polygon.contains(Point(15.0, 15.0))


def test_alignment_annotation_requires_one_shared_tissue_edge(tmp_path: Path) -> None:
    """Missing edge annotations must fail with a platform-specific error."""
    path = _write_annotation(tmp_path / "missing_edge.geojson", include_edge=False)

    with pytest.raises(ValueError, match="XENIUM.*exactly one global tissue-edge"):
        load_alignment_tissue_annotation(path, platform="XENIUM")


def test_valis_configuration_requires_both_platform_annotations(tmp_path: Path) -> None:
    """The production dispatcher must fail before attempting image resolution."""
    config = AlignmentConfig(
        pair_id="pair",
        merscope_zarr_path=tmp_path / "merscope.zarr",
        xenium_zarr_path=tmp_path / "xenium.zarr",
        output_dir=tmp_path / "alignment",
    )

    with pytest.raises(ValueError, match="requires the MERSCOPE tissue annotation"):
        load_required_valis_tissue_annotations(config)

    legacy = config.model_copy(update={"backend": "legacy_spateo"})
    assert load_required_valis_tissue_annotations(legacy) == {}


def test_annotation_raster_is_affine_mapped_and_support_clipped_without_morphology(
    tmp_path: Path,
) -> None:
    """Rasterization must apply the frame affine and clip only to acquisition."""
    annotation = load_alignment_tissue_annotation(
        _write_annotation(tmp_path / "combined.geojson"),
        platform="MERSCOPE",
    )
    support = np.ones((32, 38), dtype=np.uint8) * 255
    support[:, 25:] = 0
    matrix = np.array(
        [[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    raster = rasterize_alignment_tissue_annotation(
        annotation,
        dataset_to_registration_matrix=matrix,
        shape_rc=support.shape,
        acquired_support_mask=support,
        registration_pixel_size_um=1.0,
    )

    assert np.array_equal(
        raster.tissue > 0,
        (raster.unclipped > 0) & (support > 0),
    )
    assert raster.unclipped[18, 9] == 255
    assert raster.unclipped[18, 20] == 0  # exclusion after affine translation
    assert np.any((raster.unclipped > 0) & (support == 0))
    assert raster.metadata["morphology_applied"] is False
    assert raster.metadata["fraction_clipped_by_acquired_support"] > 0.0


def test_registration_frame_preserves_annotation_at_invalid_image_edge(
    tmp_path: Path,
) -> None:
    """Validity erosion must restrict scoring, not the anatomical tissue mask."""
    rows, columns = np.indices((24, 24))
    image = (30.0 + 2.0 * columns + rows).astype(np.float32)
    frame = DapiFrame(
        platform="XENIUM",
        image_key="dapi",
        image=image,
        original_shape_rc=image.shape,
        dataset_to_image_matrix=np.eye(3, dtype=np.float64),
        pixel_size_xy_um=(1.0, 1.0),
        channel_name="DAPI",
        coordinate_metadata_source="test",
        coordinate_metadata_trusted=True,
    )
    annotation_path = tmp_path / "single.geojson"
    annotation_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    _feature(
                        "LineString",
                        [[0.0, 0.0], [23.0, 0.0]],
                        role="pia",
                    ),
                    _feature(
                        "LineString",
                        [[0.0, 0.0], [0.0, 23.0], [23.0, 23.0], [23.0, 0.0]],
                        role="tissue_edge",
                    ),
                ],
            }
        )
    )
    annotation = load_alignment_tissue_annotation(
        annotation_path,
        platform="XENIUM",
    )

    registered = _prepare_one_frame(
        frame,
        target_shape_rc=image.shape,
        canvas_shape_rc=image.shape,
        target_pixel_size_um=1.0,
        processing=DAPIProcessingConfig(
            background_sigma_um=3.0,
            smoothing_sigma_um=0.0,
            edge_taper_um=0.0,
            edge_exclusion_um=4.0,
        ),
        tissue_annotation=annotation,
    )

    assert np.any((registered.tissue_mask > 0) & (registered.valid_mask == 0))
    assert not np.any(
        (registered.tissue_scoring_mask > 0) & (registered.valid_mask == 0)
    )
    assert registered.tissue_annotation_metadata is not None
    assert registered.tissue_annotation_metadata["morphology_applied"] is False
