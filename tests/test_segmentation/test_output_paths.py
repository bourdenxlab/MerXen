"""Tests for persistent segmentation output staging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from merxen.config import SegmentationConfig
from merxen.segmentation.pipeline import run_segmentation_pipeline


def test_run_segmentation_pipeline_stages_persistent_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persistent segmentation artifacts should be staged back into the work dir."""
    work_dir = tmp_path / "work" / "segment_out"
    persistent_root = tmp_path / "results"

    cfg = SegmentationConfig.model_validate(
        {
            "dataset": {
                "name": "P1_MERSCOPE",
                "platform": "MERSCOPE",
                "data_path": str(tmp_path / "input.zarr"),
                "channels": ["DAPI", "PolyT"],
                "output_dir": str(work_dir),
                "persistent_latest_zarr_path": str(
                    persistent_root / "latest" / "latest_spatialdata.zarr"
                ),
                "persistent_mask_path": str(
                    persistent_root / "segmentation" / "cellpose_masks_tiled.npy"
                ),
                "persistent_transcripts_path": str(
                    persistent_root / "segmentation" / "transcripts_for_proseg.csv"
                ),
                "persistent_cellpose_stitching_stats_path": str(
                    persistent_root / "segmentation" / "cellpose_stitching_stats.json"
                ),
                "persistent_nuclei_mask_path": str(
                    persistent_root / "segmentation" / "cellpose_nuclei_masks_tiled.npy"
                ),
                "persistent_nuclei_stitching_stats_path": str(
                    persistent_root
                    / "segmentation"
                    / "cellpose_nuclei_stitching_stats.json"
                ),
            }
        }
    )

    points_df = pd.DataFrame({"x": [1.0], "y": [2.0], "gene": ["Gad1"]})
    loaded_channels: list[list[str]] = []
    cellpose_calls: list[dict[str, object]] = []

    def fake_load(config: SegmentationConfig) -> tuple[object, ...]:
        loaded_channels.append(list(config.dataset.channels))
        return (object(), object(), 8, 8, np.eye(3), points_df)

    monkeypatch.setattr("merxen.segmentation.pipeline._load_dataset_sdata", fake_load)

    def fake_cellpose(
        *,
        output_mask_path: Path,
        output_stitching_stats_path: Path | None = None,
        **kwargs: object,
    ) -> Path:
        cellpose_calls.append(kwargs)
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask_path, np.ones((4, 4), dtype=np.int32))
        if output_stitching_stats_path is not None:
            output_stitching_stats_path.parent.mkdir(parents=True, exist_ok=True)
            output_stitching_stats_path.write_text('{"final_labels": 1}\n')
        return output_mask_path

    monkeypatch.setattr(
        "merxen.segmentation.pipeline.run_tiled_cellpose",
        fake_cellpose,
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.build_cellpose_affine_to_microns",
        lambda *args, **kwargs: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.write_proseg_csv_from_points",
        lambda *, csv_path, **kwargs: (
            csv_path.parent.mkdir(parents=True, exist_ok=True),
            csv_path.write_text("x_micron,y_micron,z_micron,feature_name,cell_id\n"),
            {"n_seeded": 1, "pct_seeded": 100.0},
        )[-1],
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.run_proseg_refinement",
        lambda *, output_path, **kwargs: (
            output_path.parent.mkdir(parents=True, exist_ok=True),
            output_path.mkdir(parents=True, exist_ok=True),
            output_path,
        )[-1],
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.convert_to_latest_zarr",
        lambda raw_path, latest_path: (
            latest_path.parent.mkdir(parents=True, exist_ok=True),
            latest_path.mkdir(parents=True, exist_ok=True),
            (latest_path / "marker.txt").write_text("latest"),
            latest_path,
        )[-1],
    )

    outputs = run_segmentation_pipeline(cfg, force_rerun=True)

    staged_latest = work_dir / "proseg_base_latest.zarr"
    staged_mask = work_dir / "cellpose_masks_tiled.npy"
    staged_transcripts = work_dir / "transcripts_for_proseg.csv"
    staged_stats = work_dir / "cellpose_stitching_stats.json"
    staged_nuclei_mask = work_dir / "cellpose_nuclei_masks_tiled.npy"
    staged_nuclei_stats = work_dir / "cellpose_nuclei_stitching_stats.json"

    assert outputs["latest_output"] == staged_latest
    assert outputs["cellpose_mask_path"] == staged_mask
    assert outputs["transcripts_csv"] == staged_transcripts
    assert outputs["nuclei_mask_path"] == staged_nuclei_mask

    assert staged_latest.is_symlink()
    assert staged_mask.is_symlink()
    assert staged_transcripts.is_symlink()
    assert staged_stats.is_symlink()
    assert staged_nuclei_mask.is_symlink()
    assert staged_nuclei_stats.is_symlink()

    assert (
        staged_latest.resolve()
        == Path(cfg.dataset.persistent_latest_zarr_path).resolve()
    )
    assert staged_mask.resolve() == Path(cfg.dataset.persistent_mask_path).resolve()
    assert (
        staged_transcripts.resolve()
        == Path(cfg.dataset.persistent_transcripts_path).resolve()
    )
    assert (
        staged_stats.resolve()
        == Path(cfg.dataset.persistent_cellpose_stitching_stats_path).resolve()
    )
    assert (
        staged_nuclei_mask.resolve()
        == Path(cfg.dataset.persistent_nuclei_mask_path).resolve()
    )
    assert (
        staged_nuclei_stats.resolve()
        == Path(cfg.dataset.persistent_nuclei_stitching_stats_path).resolve()
    )

    assert not (work_dir / "proseg_base_raw.zarr").exists()
    assert loaded_channels == [["DAPI", "PolyT"], ["DAPI"]]
    assert len(cellpose_calls) == 2
    nuclei_call = next(
        call for call in cellpose_calls if str(call["dataset_name"]).endswith(":nuclei")
    )
    assert nuclei_call["cellpose_config"].model_type == "nuclei"
    assert nuclei_call["cellpose_config"].flow_threshold == cfg.cellpose.flow_threshold
    assert (
        nuclei_call["cellpose_config"].cellprob_threshold
        == cfg.cellpose.cellprob_threshold
    )
    assert nuclei_call["mask_filter_config"].final_min_area_um2 == 5.0
    assert nuclei_call["mask_filter_config"].final_max_area_um2 == 400.0


def test_run_segmentation_pipeline_filters_cellpose_mask_before_proseg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The final Cellpose mask on disk should be cleaned before ProSeg uses it."""
    work_dir = tmp_path / "segment_out"
    cfg = SegmentationConfig.model_validate(
        {
            "dataset": {
                "name": "P1_MERSCOPE",
                "platform": "MERSCOPE",
                "data_path": str(tmp_path / "input.zarr"),
                "channels": ["DAPI", "PolyT"],
                "output_dir": str(work_dir),
            },
            "mask_filter": {
                "final_min_area_um2": 5.0,
                "final_max_area_um2": 400.0,
                "final_filter_chunk_mb": 1,
                "show_progress": False,
            },
        }
    )

    points_df = pd.DataFrame({"x": [1.0], "y": [2.0], "gene": ["Gad1"]})
    captured: dict[str, np.ndarray] = {}

    def fake_cellpose(
        *,
        output_mask_path: Path,
        output_stitching_stats_path: Path | None = None,
        **kwargs: object,
    ) -> Path:
        del kwargs
        mask = np.zeros((64, 64), dtype=np.uint32)
        mask[1:3, 1:3] = 1
        mask[5:10, 5:10] = 2
        mask[12:34, 12:34] = 3
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask_path, mask)
        if output_stitching_stats_path is not None:
            output_stitching_stats_path.write_text('{"final_labels": 3}\n')
        return output_mask_path

    def fake_write_csv(
        *,
        masks: np.ndarray,
        csv_path: Path,
        **kwargs: object,
    ) -> dict[str, int | float]:
        captured["masks_seen_by_csv"] = np.asarray(masks).copy()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("x_micron,y_micron,z_micron,feature_name,cell_id\n")
        return {"n_seeded": 1, "pct_seeded": 100.0}

    def fake_proseg(
        *,
        cellpose_masks: Path,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        captured["masks_seen_by_proseg"] = np.load(cellpose_masks)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    monkeypatch.setattr(
        "merxen.segmentation.pipeline._load_dataset_sdata",
        lambda config: (object(), object(), 64, 64, np.eye(3), points_df),
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.run_tiled_cellpose",
        fake_cellpose,
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.build_cellpose_affine_to_microns",
        lambda *args, **kwargs: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.write_proseg_csv_from_points",
        fake_write_csv,
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.run_proseg_refinement",
        fake_proseg,
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.convert_to_latest_zarr",
        lambda raw_path, latest_path: (
            latest_path.mkdir(parents=True, exist_ok=True),
            latest_path,
        )[-1],
    )

    outputs = run_segmentation_pipeline(cfg, force_rerun=True)

    cleaned = np.load(outputs["cellpose_mask_path"])
    assert np.unique(cleaned).tolist() == [0, 1]
    assert int((cleaned == 1).sum()) == 25
    assert (work_dir / "cellpose_stitching_stats.json").exists()
    np.testing.assert_array_equal(captured["masks_seen_by_csv"], cleaned)
    np.testing.assert_array_equal(captured["masks_seen_by_proseg"], cleaned)
