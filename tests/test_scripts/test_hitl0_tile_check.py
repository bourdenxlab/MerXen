"""Tests for the read-only HITL-0 acceptance harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tile_check() -> ModuleType:
    """Import the acceptance script without running its GPU command."""
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "acceptance"
        / "hitl0_tile_check.py"
    )
    spec = importlib.util.spec_from_file_location("hitl0_tile_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mask() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=np.uint32)
    mask[3:10, 3:10] = 100_000
    mask[18:26, 18:26] = 200_000
    return mask


def test_agreement_ignores_numeric_label_ids(tile_check: ModuleType) -> None:
    reference = _mask()
    prediction = np.where(reference == 100_000, 8, reference)
    result = tile_check.score_tile_agreement(reference, prediction, 0.99)
    assert result.passed
    assert result.ap_at_05 == result.mean_matched_iou == 1.0
    assert result.reference_cells == result.predicted_cells == 2


def test_agreement_penalises_missing_cells(tile_check: ModuleType) -> None:
    reference = _mask()
    prediction = np.where(reference == 100_000, 0, reference)
    result = tile_check.score_tile_agreement(reference, prediction, 0.99)
    assert not result.passed
    assert result.ap_at_05 == 0.5
    assert result.false_negative == 1
    assert result.mean_matched_iou == 1.0


def test_agreement_requires_shape_agreement_as_well_as_matching(
    tile_check: ModuleType,
) -> None:
    prediction = _mask()
    prediction[2, 3:10] = 100_000
    result = tile_check.score_tile_agreement(_mask(), prediction, 0.99)
    assert result.ap_at_05 == 1.0
    assert result.mean_matched_iou < 0.99
    assert not result.passed


def test_agreement_excludes_cut_border_cells(tile_check: ModuleType) -> None:
    reference = _mask()
    prediction = reference.copy()
    reference[0:4, 18:23] = 900_000
    result = tile_check.score_tile_agreement(reference, prediction, 0.99)
    assert result.passed
    assert result.reference_cells == 2


def test_empty_reference_cannot_pass(tile_check: ModuleType) -> None:
    with pytest.raises(ValueError, match="no complete reference cells"):
        tile_check.score_tile_agreement(
            np.zeros((8, 8), int), np.zeros((8, 8), int), 0.99
        )


def test_no_predictions_reports_failure(tile_check: ModuleType) -> None:
    result = tile_check.score_tile_agreement(_mask(), np.zeros((32, 32), int), 0.99)
    assert not result.passed
    assert result.false_negative == 2
    assert result.mean_matched_iou == result.ap_at_05 == 0.0


def test_output_guard_follows_protected_directory_symlinks(
    tile_check: ModuleType, tmp_path: Path
) -> None:
    baseline = tmp_path / "published"
    baseline.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(baseline, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the protected"):
        tile_check._require_fresh_output(alias / "trial", baseline)


def test_output_guard_rejects_existing_results(
    tile_check: ModuleType, tmp_path: Path
) -> None:
    output = tmp_path / "trial"
    output.mkdir()
    (output / "tile_check.json").write_text("old result")
    with pytest.raises(ValueError, match="new or empty"):
        tile_check._require_fresh_output(output, tmp_path / "published")


def test_main_refuses_baseline_output_before_reading_data(
    tile_check: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="outside the protected"):
        tile_check.main(
            [
                "--config",
                str(tmp_path / "missing.json"),
                "--source-zarr",
                str(tmp_path / "source.zarr"),
                "--baseline-mask",
                str(tmp_path / "mask.npy"),
                "--baseline-stats",
                str(tmp_path / "stats.json"),
                "--protected-root",
                str(tmp_path / "published"),
                "--output-dir",
                str(tmp_path / "published" / "trial"),
                "--image-key",
                "native",
                "--tile-index",
                "0",
            ]
        )
