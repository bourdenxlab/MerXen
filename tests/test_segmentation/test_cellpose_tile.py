"""Regression checks for the shared production tile operation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import label

from merxen.config import CellposeConfig, MaskFilterConfig, MemoryConfig, TilingConfig
from merxen.segmentation import cellpose
from merxen.segmentation.model_identity import CellposeModelIdentity


def test_shared_tile_preserves_input_and_raw_labels() -> None:
    """Prepare the full selected halo once and keep raw labels when filtering."""
    source = np.arange(14 * 15 * 2, dtype=np.uint16).reshape(14, 15, 2)
    boxes = []

    def fetch(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        boxes.append((y0, y1, x0, x1))
        return source[y0:y1, x0:x1]

    def evaluate(image: np.ndarray, **kwargs: object) -> tuple:
        del kwargs
        masks = np.zeros(image.shape[:2], np.uint32)
        masks[2:5, 2:5] = 1
        masks[6, 6] = 2
        return masks, [None, None, np.full(masks.shape, 2.5, np.float32)], None

    result = cellpose.run_cellpose_tile(
        fetch,
        {"tile_y0": 1, "tile_y1": 12, "tile_x0": 2, "tile_x1": 14},
        SimpleNamespace(eval=evaluate),
        CellposeConfig(gpu=False),
        MaskFilterConfig(min_area_px=3, n_jobs=1),
    )
    assert boxes == [(1, 12, 2, 14)]
    _, expected_input, _ = cellpose.prepare_cellpose_input(source[1:12, 2:14])
    np.testing.assert_array_equal(result.model_input, expected_input)
    assert set(np.unique(result.raw_masks)) == {0, 1, 2}
    assert set(np.unique(result.filtered_masks)) == {0, 1}
    assert np.all(result.cellprob_logits == 2.5)


def test_tiled_outputs_match_pre_refactor_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin real filtering/stitching results captured before extracting hook C7."""
    image = np.arange(28 * 33 * 2, dtype=np.uint16).reshape(28, 33, 2) % 137
    config = CellposeConfig(gpu=False)

    def evaluate(array: np.ndarray, **kwargs: object) -> tuple:
        del kwargs
        masks, _ = label(array[:, :, 0] > 128)
        return masks, [None, None, array[:, :, 1].astype(np.float32) / 13], None

    monkeypatch.setattr(
        cellpose, "build_cellpose_model", lambda _: SimpleNamespace(eval=evaluate)
    )
    monkeypatch.setattr(
        cellpose,
        "identify_cellpose_model",
        lambda *_: CellposeModelIdentity(
            status="verified",
            model_name="cpsam_v2",
            weights_sha256="0" * 64,
            cellpose_version="4.2.1.1",
            torch_version="2.14.0+cu130",
            cellpose_config=config,
        ),
    )
    monkeypatch.setattr(cellpose, "choose_working_tile_size", lambda **_: 12)
    monkeypatch.setattr(cellpose, "clear_cuda_cache", lambda: None)
    monkeypatch.setattr(cellpose, "force_release", lambda **_: None)
    monkeypatch.setattr(cellpose, "enforce_memory_limit", lambda **_: None)
    mask_path = tmp_path / "cellpose_masks_tiled.npy"
    probs_path = tmp_path / "logits.npy"
    stats_path = tmp_path / "stats.json"
    cellpose.run_tiled_cellpose(
        lambda y0, y1, x0, x1: image[y0:y1, x0:x1],
        28,
        33,
        "reference",
        mask_path,
        config,
        MaskFilterConfig(n_jobs=1, min_area_px=3, max_eccentricity=1.0),
        TilingConfig(tile_size_candidates=[12], min_tile_size=12, stitch_overlap_px=2),
        MemoryConfig(),
        output_stitching_stats_path=stats_path,
        output_cellprob_path=probs_path,
    )
    stats = json.loads(stats_path.read_text())
    stats.pop("model_identity")
    assert np.any(np.load(mask_path))
    assert np.any(np.load(probs_path))
    actual = {
        "mask": hashlib.sha256(np.load(mask_path).tobytes()).hexdigest(),
        "logits": hashlib.sha256(np.load(probs_path).tobytes()).hexdigest(),
        "stats": hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest(),
    }
    # Captured on HITL-0 (85942c1), before the C7 tile-body extraction.
    assert actual == {
        "mask": "6fc4dcd62aa278c3316f42f86a14c09401208ff0fcab77146cfab34dfc2ae1d4",
        "logits": "c42150e4e6a57b58b33be4d00f69dbf8ff7052c63bd5d6b9fd2d082324a0cec9",
        "stats": "4916a3ddbe9c08e7a5b925d6fd245e1795cf3a76fccd164c642d48d07c7710a3",
    }
