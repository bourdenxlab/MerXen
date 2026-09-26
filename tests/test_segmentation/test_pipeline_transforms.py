"""Tests for the MERSCOPE mask-to-micron transform used by segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.config import DatasetConfig, SegmentationConfig
from merxen.segmentation.cellpose import (
    assign_labels_from_masks,
    build_cellpose_affine_to_microns,
    invert_mask_affine,
)
from merxen.segmentation.pipeline import (
    _load_merscope_transform_matrix,
    run_cellpose_segmentation,
)

_BUILT = np.array([[9.2595, 0.0, 202.44], [0.0, 9.2593, 1034.98], [0.0, 0.0, 1.0]])


def _config(zarr_path: Path, transform_path: Path | None) -> SegmentationConfig:
    return SegmentationConfig(
        dataset=DatasetConfig(
            name="S1_MERSCOPE",
            platform="MERSCOPE",
            data_path=zarr_path,
            channels=["DAPI"],
            output_dir=zarr_path.parent / "segment_out",
            transform_path=transform_path,
        )
    )


def _built_zarr(tmp_path: Path, matrix: np.ndarray | None = _BUILT) -> Path:
    zarr_path = tmp_path / "source.zarr"
    zarr_path.mkdir()
    if matrix is not None:
        np.savetxt(zarr_path / "micron_to_mosaic_pixel_transform.csv", matrix)
    return zarr_path


def test_uses_build_sidecar_without_override(tmp_path: Path) -> None:
    matrix = _load_merscope_transform_matrix(_config(_built_zarr(tmp_path), None))

    np.testing.assert_allclose(matrix, _BUILT)


def test_accepts_override_matching_the_build(tmp_path: Path) -> None:
    override = tmp_path / "override.csv"
    # Vizgen's CSV holds float32 values; the build sidecar may hold float64.
    np.savetxt(override, _BUILT.astype(np.float32))

    matrix = _load_merscope_transform_matrix(_config(_built_zarr(tmp_path), override))

    np.testing.assert_allclose(matrix, _BUILT, atol=1e-3)


def test_rejects_override_that_changed_after_the_build(tmp_path: Path) -> None:
    """A corrected CSV must not be mixed with a zarr built from the stale one."""
    stale = _BUILT.copy()
    stale[:2, 2] = 0.0
    zarr_path = _built_zarr(tmp_path, stale)
    override = tmp_path / "override.csv"
    np.savetxt(override, _BUILT)

    with pytest.raises(ValueError, match="force_spatialdata_build"):
        _load_merscope_transform_matrix(_config(zarr_path, override))


def test_uses_override_when_the_zarr_has_no_sidecar(tmp_path: Path) -> None:
    override = tmp_path / "override.csv"
    np.savetxt(override, _BUILT)

    matrix = _load_merscope_transform_matrix(
        _config(_built_zarr(tmp_path, None), override)
    )

    np.testing.assert_allclose(matrix, _BUILT)


# Reusing a persistent transcripts_for_proseg.csv: its seeds must come from the
# current mask-to-micron affine. 10 um / px blocks of 10 px, so the stale
# zero-translation affine (off by 23 px, 57 px) lands every seed in another block.
_SEED_MATRIX = np.array([[10.0, 0.0, 23.0], [0.0, 10.0, 57.0], [0.0, 0.0, 1.0]])
_STALE_SEED_MATRIX = np.diag([10.0, 10.0, 1.0])


def _block_mask(size: int = 400, block: int = 10) -> np.ndarray:
    rows, cols = np.indices((size, size))
    return ((rows // block) * (size // block) + (cols // block) + 1).astype(np.uint32)


def _persistent_cellpose_outputs(
    tmp_path: Path,
    *,
    seeding_matrix: np.ndarray,
    sidecar_matrix: np.ndarray | None = None,
) -> tuple[SegmentationConfig, Path]:
    """Write a built zarr (current transform) and reusable Cellpose outputs."""
    zarr_path = _built_zarr(tmp_path, _SEED_MATRIX)
    seg_dir = tmp_path / "results" / "segmentation"
    seg_dir.mkdir(parents=True)
    mask = _block_mask()
    np.save(seg_dir / "cellpose_masks_tiled.npy", mask)
    np.save(seg_dir / "cellpose_cellprobs_tiled.npy", np.zeros(mask.shape, np.float32))

    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 30.0, 5000)
    y = rng.uniform(0.0, 30.0, 5000)
    x_transform, y_transform = build_cellpose_affine_to_microns(seeding_matrix, 1.0)
    a_inv, b = invert_mask_affine(x_transform, y_transform)
    csv_path = seg_dir / "transcripts_for_proseg.csv"
    pd.DataFrame(
        {
            "transcript_id": np.arange(len(x)),
            "x_micron": x.astype(np.float32),
            "y_micron": y.astype(np.float32),
            "z_micron": np.zeros(len(x), np.float32),
            "feature_name": "Gad1",
            "cell_id": assign_labels_from_masks(x, y, mask, a_inv=a_inv, b=b),
        }
    ).to_csv(csv_path, index=False)
    if sidecar_matrix is not None:
        sx, sy = build_cellpose_affine_to_microns(sidecar_matrix, 1.0)
        (seg_dir / "transcripts_for_proseg.transforms.json").write_text(
            json.dumps({"x_transform": list(sx), "y_transform": list(sy)})
        )

    cfg = SegmentationConfig(
        dataset=DatasetConfig(
            name="S1_MERSCOPE",
            platform="MERSCOPE",
            data_path=zarr_path,
            channels=["DAPI"],
            output_dir=tmp_path / "work" / "segment_out",
            persistent_mask_path=seg_dir / "cellpose_masks_tiled.npy",
            persistent_cellpose_cellprob_path=seg_dir / "cellpose_cellprobs_tiled.npy",
            persistent_transcripts_path=csv_path,
            persistent_cellpose_stitching_stats_path=seg_dir / "stats.json",
        )
    )
    return cfg, seg_dir / "transcripts_for_proseg.transforms.json"


@pytest.fixture
def _no_cellpose_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "merxen.segmentation.pipeline._load_dataset_sdata",
        lambda config: pytest.fail("the check must decide before Cellpose re-runs"),
    )


@pytest.mark.usefixtures("_no_cellpose_rerun")
def test_reuse_rejects_csv_whose_sidecar_has_zero_translation(tmp_path: Path) -> None:
    """A corrected transform must not be paired with seeds from the stale one."""
    cfg, _ = _persistent_cellpose_outputs(
        tmp_path,
        seeding_matrix=_STALE_SEED_MATRIX,
        sidecar_matrix=_STALE_SEED_MATRIX,
    )

    with pytest.raises(ValueError, match="seeded with a different"):
        run_cellpose_segmentation(cfg)


@pytest.mark.usefixtures("_no_cellpose_rerun")
def test_reuse_rejects_legacy_csv_seeded_with_a_stale_transform(
    tmp_path: Path,
) -> None:
    """Without a sidecar the stored seeds are checked against the mask."""
    cfg, sidecar = _persistent_cellpose_outputs(
        tmp_path, seeding_matrix=_STALE_SEED_MATRIX
    )

    with pytest.raises(ValueError, match=r"only 0\.0% of"):
        run_cellpose_segmentation(cfg)
    assert not sidecar.exists()


@pytest.mark.usefixtures("_no_cellpose_rerun")
def test_reuse_accepts_legacy_csv_and_records_its_transform(tmp_path: Path) -> None:
    cfg, sidecar = _persistent_cellpose_outputs(tmp_path, seeding_matrix=_SEED_MATRIX)

    outputs = run_cellpose_segmentation(cfg)

    expected_x, expected_y = build_cellpose_affine_to_microns(_SEED_MATRIX, 1.0)
    recorded = json.loads(sidecar.read_text())
    np.testing.assert_allclose(recorded["x_transform"], expected_x)
    np.testing.assert_allclose(recorded["y_transform"], expected_y)
    assert json.loads(outputs["transforms_path"].read_text()) == recorded
    assert outputs["transcripts_csv"].is_symlink()


@pytest.mark.usefixtures("_no_cellpose_rerun")
def test_reuse_accepts_csv_whose_sidecar_matches(tmp_path: Path) -> None:
    cfg, _ = _persistent_cellpose_outputs(
        tmp_path, seeding_matrix=_SEED_MATRIX, sidecar_matrix=_SEED_MATRIX
    )

    assert run_cellpose_segmentation(cfg)["transcripts_csv"].is_symlink()
