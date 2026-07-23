"""Tests for core-owned Cellpose tile stitching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from merxen.config import TilingConfig
from merxen.segmentation.cellpose import (
    _stitch_core_owned_tile_labels,
    _write_stitching_stats,
    extract_cellpose_probability_logits,
)


def _tile(
    *,
    tile_y0: int = 0,
    tile_y1: int = 8,
    tile_x0: int = 0,
    tile_x1: int = 8,
    image_height: int = 8,
    image_width: int = 12,
    core_y0_in_tile: int = 0,
    core_y1_in_tile: int = 8,
    core_x0_in_tile: int = 0,
    core_x1_in_tile: int = 8,
) -> dict[str, int]:
    """Build a tile metadata dict matching ``iter_core_tiles`` output."""
    return {
        "tile_y0": tile_y0,
        "tile_y1": tile_y1,
        "tile_x0": tile_x0,
        "tile_x1": tile_x1,
        "image_height": image_height,
        "image_width": image_width,
        "core_y0_in_tile": core_y0_in_tile,
        "core_y1_in_tile": core_y1_in_tile,
        "core_x0_in_tile": core_x0_in_tile,
        "core_x1_in_tile": core_x1_in_tile,
    }


def _stitch(
    tile_mask: np.ndarray,
    global_mask: np.ndarray,
    tile: dict[str, int],
    *,
    next_label: int = 1,
    areas: dict[int, int] | None = None,
    config: TilingConfig | None = None,
) -> tuple[int, dict[str, int], dict[int, int]]:
    """Stitch one synthetic tile and return updated labels, stats, and areas."""
    label_areas = {} if areas is None else areas
    next_label, stats = _stitch_core_owned_tile_labels(
        tile_mask=tile_mask,
        global_mask=global_mask,
        tile=tile,
        next_label=next_label,
        global_label_areas=label_areas,
        tiling_config=config or TilingConfig(stitch_overlap_px=2),
    )
    return next_label, stats, label_areas


def test_core_owned_stitching_keeps_complete_boundary_cell() -> None:
    """A cell crossing the core edge should be pasted whole, not core-cropped."""
    global_mask = np.zeros((8, 12), dtype=np.uint32)
    tile_mask = np.zeros((8, 8), dtype=np.int32)
    tile_mask[2:6, 4:8] = 1

    next_label, stats, _ = _stitch(
        tile_mask,
        global_mask,
        _tile(core_x0_in_tile=0, core_x1_in_tile=6),
    )

    assert next_label == 2
    assert stats["owned_labels"] == 1
    assert stats["accepted_labels"] == 1
    assert np.unique(global_mask).tolist() == [0, 1]
    assert np.all(global_mask[2:6, 4:8] == 1)


def test_high_overlap_duplicate_is_skipped() -> None:
    """A later tile object matching an existing cell should be skipped."""
    global_mask = np.zeros((8, 8), dtype=np.uint32)
    first = np.zeros((8, 8), dtype=np.int32)
    first[2:6, 2:6] = 1
    next_label, stats, areas = _stitch(first, global_mask, _tile(image_width=8))
    assert stats["accepted_labels"] == 1

    duplicate = np.zeros((8, 8), dtype=np.int32)
    duplicate[2:6, 2:6] = 1
    next_label, stats, _ = _stitch(
        duplicate,
        global_mask,
        _tile(image_width=8),
        next_label=next_label,
        areas=areas,
    )

    assert next_label == 2
    assert stats["duplicate_skipped"] == 1
    assert np.unique(global_mask).tolist() == [0, 1]


def test_low_overlap_neighbor_is_retained() -> None:
    """Low-overlap neighboring cells should both survive stitching."""
    global_mask = np.zeros((8, 10), dtype=np.uint32)
    first = np.zeros((8, 10), dtype=np.int32)
    first[2:6, 1:5] = 1
    next_label, _, areas = _stitch(first, global_mask, _tile(image_width=10))

    neighbor = np.zeros((8, 10), dtype=np.int32)
    neighbor[2:6, 4:8] = 1
    next_label, stats, _ = _stitch(
        neighbor,
        global_mask,
        _tile(image_width=10),
        next_label=next_label,
        areas=areas,
    )

    assert next_label == 3
    assert stats["accepted_labels"] == 1
    assert stats["duplicate_skipped"] == 0
    assert np.unique(global_mask).tolist() == [0, 1, 2]


def test_artificial_edge_touching_label_is_kept_and_counted() -> None:
    """The default edge policy should retain edge-touching labels with stats."""
    global_mask = np.zeros((10, 8), dtype=np.uint32)
    tile_mask = np.zeros((6, 8), dtype=np.int32)
    tile_mask[0:3, 2:6] = 1

    _, stats, _ = _stitch(
        tile_mask,
        global_mask,
        _tile(tile_y0=2, tile_y1=8, image_height=10, core_y1_in_tile=6),
    )

    assert stats["edge_touching_labels"] == 1
    assert stats["edge_touching_skipped"] == 0
    assert stats["accepted_labels"] == 1
    assert int((global_mask == 1).sum()) == 12


def test_invalid_stitching_thresholds_are_rejected() -> None:
    """Tiling config should fail fast on invalid threshold fractions."""
    with pytest.raises(ValidationError):
        TilingConfig(duplicate_iou_threshold=1.2)
    with pytest.raises(ValidationError):
        TilingConfig(duplicate_overlap_fraction=-0.1)
    with pytest.raises(ValidationError):
        TilingConfig(min_remaining_fraction=1.1)


def test_write_stitching_stats_json(tmp_path: Path) -> None:
    """Stitching diagnostics should be written as a JSON artifact."""
    stats_path = tmp_path / "cellpose_stitching_stats.json"
    _write_stitching_stats(stats_path, {"final_labels": 3, "accepted_labels": 3})

    assert stats_path.read_text().strip().startswith("{")
    assert '"final_labels": 3' in stats_path.read_text()


def test_extract_cellpose_probability_logits_uses_flow_index_two() -> None:
    """Cellpose probability logits should be read without applying sigmoid."""
    expected = np.asarray([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    logits = extract_cellpose_probability_logits([None, None, expected], (2, 2))

    np.testing.assert_array_equal(logits, expected)


def test_stitching_writes_probabilities_only_for_accepted_mask_pixels() -> None:
    """Stitched Cellpose logits should remain aligned with accepted labels."""
    global_mask = np.zeros((8, 8), dtype=np.uint32)
    global_probs = np.zeros((8, 8), dtype=np.float32)
    tile_mask = np.zeros((8, 8), dtype=np.int32)
    tile_mask[2:6, 2:6] = 1
    tile_probs = np.full((8, 8), 7.0, dtype=np.float32)

    _stitch_core_owned_tile_labels(
        tile_mask=tile_mask,
        global_mask=global_mask,
        tile=_tile(image_width=8),
        next_label=1,
        global_label_areas={},
        tiling_config=TilingConfig(stitch_overlap_px=2),
        tile_cellprobs=tile_probs,
        global_cellprobs=global_probs,
    )

    assert np.all(global_probs[global_mask == 1] == 7.0)
    assert np.all(global_probs[global_mask == 0] == 0.0)
