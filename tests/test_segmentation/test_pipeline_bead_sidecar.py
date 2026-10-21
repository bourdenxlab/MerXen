"""Tests for the bead-removal sidecar guarding reuse of Cellpose masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.config import MerscopeBeadRemovalConfig, SegmentationConfig
from merxen.io.bead_removal.provenance import (
    IMAGE_PREPROCESSING_ATTR,
    build_image_preprocessing_record,
    segmentation_identity,
    segmentation_sidecar_path,
)
from merxen.segmentation.pipeline import (
    run_cellpose_nuclei_segmentation,
    run_cellpose_segmentation,
    run_segmentation_pipeline,
)

_MATRIX = np.array([[10.0, 0.0, 23.0], [0.0, 10.0, 57.0], [0.0, 0.0, 1.0]])


def _record(*, enabled: bool, calibration: str = "cal-1") -> dict[str, Any]:
    return build_image_preprocessing_record(
        z_layers=(1, 2, 3),
        stainings=["PolyT", "DAPI"],
        bead_removal_cfg=MerscopeBeadRemovalConfig(enabled=True) if enabled else None,
        calibration={"calibration_sha256": calibration} if enabled else None,
    )


def _source_zarr(tmp_path: Path, record: dict[str, Any] | None) -> Path:
    """Write the root metadata of a built zarr, with or without the record."""
    zarr_path = tmp_path / "source.zarr"
    zarr_path.mkdir(exist_ok=True)
    np.savetxt(zarr_path / "micron_to_mosaic_pixel_transform.csv", _MATRIX)
    attributes = {} if record is None else {IMAGE_PREPROCESSING_ATTR: record}
    (zarr_path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": attributes})
    )
    return zarr_path


def _config(tmp_path: Path, zarr_path: Path) -> SegmentationConfig:
    seg_dir = tmp_path / "results" / "segmentation"
    return SegmentationConfig.model_validate(
        {
            "dataset": {
                "name": "S1_MERSCOPE",
                "platform": "MERSCOPE",
                "data_path": str(zarr_path),
                "channels": ["DAPI"],
                "output_dir": str(tmp_path / "work" / "segment_out"),
                "transform_path": str(
                    zarr_path / "micron_to_mosaic_pixel_transform.csv"
                ),
                "persistent_mask_path": str(seg_dir / "cellpose_masks_tiled.npy"),
                "persistent_cellpose_cellprob_path": str(
                    seg_dir / "cellpose_cellprobs_tiled.npy"
                ),
                "persistent_transcripts_path": str(
                    seg_dir / "transcripts_for_proseg.csv"
                ),
                "persistent_cellpose_stitching_stats_path": str(
                    seg_dir / "cellpose_stitching_stats.json"
                ),
                "persistent_nuclei_mask_path": str(
                    seg_dir / "cellpose_nuclei_masks_tiled.npy"
                ),
                "persistent_nuclei_stitching_stats_path": str(
                    seg_dir / "cellpose_nuclei_stitching_stats.json"
                ),
            },
            "mask_filter": {"final_min_area_um2": None, "final_max_area_um2": None},
            "nuclei_mask_filter": {
                "final_min_area_um2": None,
                "final_max_area_um2": None,
            },
        }
    )


def _fake_cellpose(
    *,
    output_mask_path: Path,
    output_cellprob_path: Path | None = None,
    output_stitching_stats_path: Path | None = None,
    **kwargs: object,
) -> Path:
    del kwargs
    mask = np.zeros((16, 16), dtype=np.uint32)
    mask[0:8, 0:8] = 1
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_mask_path, mask)
    if output_cellprob_path is not None:
        np.save(output_cellprob_path, np.ones(mask.shape, dtype=np.float32))
    if output_stitching_stats_path is not None:
        output_stitching_stats_path.write_text('{"final_labels": 1}\n')
    return output_mask_path


@pytest.fixture
def fake_cellpose(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace Cellpose; the returned list counts how often it ran."""
    calls: list[int] = []
    points = pd.DataFrame(
        {
            "global_x": np.arange(4, dtype=float),
            "global_y": np.ones(4),
            "gene": ["Gad1", "Nkx6-1", "Gad1", "Nkx6-1"],
            "transcript_score": np.full(4, 0.9),
        }
    )

    def _run(**kwargs: Any) -> Path:
        calls.append(1)
        return _fake_cellpose(**kwargs)

    monkeypatch.setattr(
        "merxen.segmentation.pipeline._load_dataset_sdata",
        lambda config: (object(), object(), 16, 16, np.eye(3), points),
    )
    monkeypatch.setattr("merxen.segmentation.pipeline.run_tiled_cellpose", _run)
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.build_cellpose_affine_to_microns",
        lambda *args, **kwargs: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    return calls


def _sidecar_identity(path: Path) -> Any:
    return json.loads(path.read_text())["bead_removal"]


class TestNucleiMasks:
    def test_fresh_run_records_the_source_identity(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        record = _record(enabled=True)
        cfg = _config(tmp_path, _source_zarr(tmp_path, record))

        outputs = run_cellpose_nuclei_segmentation(cfg)

        sidecar = segmentation_sidecar_path(
            outputs["nuclei_mask_path"].resolve(), nuclei=True
        )
        assert _sidecar_identity(sidecar) == segmentation_identity(record)
        assert not segmentation_sidecar_path(sidecar.with_name("x.npy")).exists()

    def test_fresh_run_without_bead_removal_records_null(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        cfg = _config(tmp_path, _source_zarr(tmp_path, None))

        outputs = run_cellpose_nuclei_segmentation(cfg)

        sidecar = segmentation_sidecar_path(
            outputs["nuclei_mask_path"].resolve(), nuclei=True
        )
        assert _sidecar_identity(sidecar) is None

    def test_matching_sidecar_is_reused(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        cfg = _config(tmp_path, _source_zarr(tmp_path, _record(enabled=True)))
        run_cellpose_nuclei_segmentation(cfg)
        assert len(fake_cellpose) == 1

        run_cellpose_nuclei_segmentation(cfg)

        assert len(fake_cellpose) == 1

    def test_rebuilt_source_with_other_calibration_is_refused(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        zarr_path = _source_zarr(tmp_path, _record(enabled=True))
        cfg = _config(tmp_path, zarr_path)
        run_cellpose_nuclei_segmentation(cfg)
        _source_zarr(tmp_path, _record(enabled=True, calibration="cal-2"))

        with pytest.raises(ValueError, match="new outdir"):
            run_cellpose_nuclei_segmentation(cfg)
        assert len(fake_cellpose) == 1

    def test_masks_from_corrected_image_are_refused_when_source_is_off(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        zarr_path = _source_zarr(tmp_path, _record(enabled=True))
        cfg = _config(tmp_path, zarr_path)
        run_cellpose_nuclei_segmentation(cfg)
        _source_zarr(tmp_path, _record(enabled=False))

        with pytest.raises(ValueError, match="bead removal on"):
            run_cellpose_nuclei_segmentation(cfg)

    def test_masks_without_sidecar_are_refused_when_source_is_enabled(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        zarr_path = _source_zarr(tmp_path, None)
        cfg = _config(tmp_path, zarr_path)
        run_cellpose_nuclei_segmentation(cfg)
        sidecar = segmentation_sidecar_path(
            Path(str(cfg.dataset.persistent_nuclei_mask_path)), nuclei=True
        )
        sidecar.unlink()
        _source_zarr(tmp_path, _record(enabled=True))

        with pytest.raises(ValueError, match="predates the correction"):
            run_cellpose_nuclei_segmentation(cfg)

    @pytest.mark.parametrize("source", ["legacy", "off"])
    def test_legacy_masks_are_reused_as_before(
        self, tmp_path: Path, fake_cellpose: list[int], source: str
    ) -> None:
        record = None if source == "legacy" else _record(enabled=False)
        cfg = _config(tmp_path, _source_zarr(tmp_path, record))
        mask_path = Path(str(cfg.dataset.persistent_nuclei_mask_path))
        mask_path.parent.mkdir(parents=True)
        np.save(mask_path, np.ones((4, 4), dtype=np.uint32))

        run_cellpose_nuclei_segmentation(cfg)

        assert fake_cellpose == []
        assert not segmentation_sidecar_path(mask_path, nuclei=True).exists()

    def test_force_rerun_replaces_a_stale_sidecar(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        zarr_path = _source_zarr(tmp_path, _record(enabled=True))
        cfg = _config(tmp_path, zarr_path)
        run_cellpose_nuclei_segmentation(cfg)
        new_record = _record(enabled=True, calibration="cal-2")
        _source_zarr(tmp_path, new_record)

        outputs = run_cellpose_nuclei_segmentation(cfg, force_rerun=True)

        sidecar = segmentation_sidecar_path(
            outputs["nuclei_mask_path"].resolve(), nuclei=True
        )
        assert _sidecar_identity(sidecar) == segmentation_identity(new_record)

    def test_failed_run_does_not_leave_the_old_sidecar(
        self,
        tmp_path: Path,
        fake_cellpose: list[int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        zarr_path = _source_zarr(tmp_path, _record(enabled=True))
        cfg = _config(tmp_path, zarr_path)
        outputs = run_cellpose_nuclei_segmentation(cfg)
        sidecar = segmentation_sidecar_path(
            outputs["nuclei_mask_path"].resolve(), nuclei=True
        )

        def _boom(**kwargs: object) -> Path:
            raise RuntimeError("interrupted")

        monkeypatch.setattr("merxen.segmentation.pipeline.run_tiled_cellpose", _boom)
        with pytest.raises(RuntimeError, match="interrupted"):
            run_cellpose_nuclei_segmentation(cfg, force_rerun=True)

        assert not sidecar.exists()


class TestCellMasks:
    def test_fresh_run_writes_and_reuse_checks_the_sidecar(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        record = _record(enabled=True)
        zarr_path = _source_zarr(tmp_path, record)
        cfg = _config(tmp_path, zarr_path)

        outputs = run_cellpose_segmentation(cfg)

        sidecar = segmentation_sidecar_path(outputs["cellpose_mask_path"].resolve())
        assert sidecar.name == "image_preprocessing.json"
        assert _sidecar_identity(sidecar) == segmentation_identity(record)
        # Same source: reused without another Cellpose run.
        run_cellpose_segmentation(cfg)
        assert len(fake_cellpose) == 1

        _source_zarr(tmp_path, None)
        with pytest.raises(ValueError, match="bead removal on"):
            run_cellpose_segmentation(cfg)
        assert len(fake_cellpose) == 1

    def test_cell_sidecar_does_not_vouch_for_nuclei_masks(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        zarr_path = _source_zarr(tmp_path, None)
        cfg = _config(tmp_path, zarr_path)
        run_cellpose_segmentation(cfg)
        run_cellpose_nuclei_segmentation(cfg)
        nuclei_sidecar = segmentation_sidecar_path(
            Path(str(cfg.dataset.persistent_nuclei_mask_path)), nuclei=True
        )
        nuclei_sidecar.unlink()
        _source_zarr(tmp_path, _record(enabled=True))

        with pytest.raises(ValueError, match="predates the correction"):
            run_cellpose_nuclei_segmentation(cfg)


class TestCompleteOutputShortcut:
    """The all-outputs-exist shortcut of the one-process pipeline."""

    @staticmethod
    def _prepare(tmp_path: Path, record: dict[str, Any] | None) -> SegmentationConfig:
        cfg = _config(tmp_path, _source_zarr(tmp_path, record))
        cfg.proseg_hybrid.enabled = False
        latest = tmp_path / "results" / "segmentation" / "proseg_base_latest.zarr"
        latest.mkdir(parents=True)
        cfg.dataset.persistent_latest_zarr_path = latest
        run_cellpose_segmentation(cfg)
        run_cellpose_nuclei_segmentation(cfg)
        return cfg

    def test_matching_sidecars_reuse_everything(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        cfg = self._prepare(tmp_path, _record(enabled=True))
        assert len(fake_cellpose) == 2

        outputs = run_segmentation_pipeline(cfg)

        assert len(fake_cellpose) == 2
        assert outputs["cellpose_mask_path"].exists()

    @pytest.mark.parametrize("sidecar", ["cell", "nuclei"])
    def test_a_stale_sidecar_is_refused_not_reused(
        self, tmp_path: Path, fake_cellpose: list[int], sidecar: str
    ) -> None:
        cfg = self._prepare(tmp_path, _record(enabled=True))
        stale = {
            "cell": cfg.dataset.persistent_mask_path,
            "nuclei": cfg.dataset.persistent_nuclei_mask_path,
        }[sidecar]
        segmentation_sidecar_path(Path(str(stale)), nuclei=sidecar == "nuclei").unlink()

        with pytest.raises(ValueError, match="predates the correction"):
            run_segmentation_pipeline(cfg)
        assert len(fake_cellpose) == 2

    def test_legacy_outputs_are_reused_as_before(
        self, tmp_path: Path, fake_cellpose: list[int]
    ) -> None:
        cfg = self._prepare(tmp_path, None)

        run_segmentation_pipeline(cfg)

        assert len(fake_cellpose) == 2
