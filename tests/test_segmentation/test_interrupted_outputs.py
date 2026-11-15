"""A step killed (or failed) part-way must not have its outputs reused on resume.

Each segmentation step writes into persistent folders that outlive a Nextflow task.
These tests interrupt a step at the points where its outputs exist but are
incomplete, then run it again and check that it recomputes them, and that
complete outputs, including those from before the markers existed, are reused.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from merxen.config import SegmentationConfig, SpatialDataBuildConfig
from merxen.io.bead_removal.provenance import (
    STORE_SIDECAR_NAME,
    segmentation_sidecar_path,
    write_store_segmentation_sidecar,
)
from merxen.io.builders.pipeline import build_spatialdata_artifact
from merxen.io.spatialdata_io import _copy_non_spatialdata_sidecars
from merxen.io.transcript_io import write_proseg_csv_from_points
from merxen.path_utils import (
    clear_incomplete,
    incomplete_marker_path,
    mark_incomplete,
    write_text_atomic,
)
from merxen.segmentation.hitl.seed_inputs import HitlSeedConfig, seed_hitl_inputs
from merxen.segmentation.output_markers import (
    cell_mask_marker_path,
    leftover_temporaries,
    nuclei_mask_marker_path,
    store_marker_path,
)
from merxen.segmentation.pipeline import (
    run_cellpose_nuclei_segmentation,
    run_cellpose_segmentation,
    run_proseg_segmentation,
    run_segmentation_pipeline,
)

from .test_pipeline_bead_sidecar import _config, _record, _source_zarr

_SOURCES = ["legacy", "off", "enabled"]


class Killed(BaseException):
    """Stands in for SIGKILL: nothing below the raise runs, no handler cleans up."""


def _source_record(source: str) -> dict[str, Any] | None:
    return {
        "legacy": None,
        "off": _record(enabled=False),
        "enabled": _record(enabled=True),
    }[source]


class CellposeRuns:
    """Replaces Cellpose; each run completes unless a kill is queued for it."""

    def __init__(self) -> None:
        self.calls = 0
        self.kills: list[bool] = []

    def __call__(
        self,
        *,
        output_mask_path: Path,
        output_cellprob_path: Path | None = None,
        output_stitching_stats_path: Path | None = None,
        before_overwrite: Callable[[], None] | None = None,
        **kwargs: object,
    ) -> Path:
        del kwargs
        self.calls += 1
        killed = self.kills.pop(0) if self.kills else False
        # As in the real run: after every refusal check, before the old outputs go.
        if before_overwrite is not None:
            before_overwrite()
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        # The real run allocates the memory-mapped outputs and fills them tile by
        # tile, so a kill leaves a full-size mask with some labels in it.
        mask = np.lib.format.open_memmap(
            output_mask_path, mode="w+", dtype=np.uint32, shape=(16, 16)
        )
        mask[:] = 0
        mask[0:8, 0:8] = 1
        if output_cellprob_path is not None:
            cellprob = np.lib.format.open_memmap(
                output_cellprob_path, mode="w+", dtype=np.float32, shape=(16, 16)
            )
            cellprob[:] = 1.0
            cellprob.flush()
        mask.flush()
        if killed:
            raise Killed
        mask[8:16, 8:16] = 2
        mask.flush()
        if output_stitching_stats_path is not None:
            output_stitching_stats_path.write_text('{"final_labels": 2}\n')
        return output_mask_path


@pytest.fixture
def cellpose(monkeypatch: pytest.MonkeyPatch) -> CellposeRuns:
    runs = CellposeRuns()
    points = pd.DataFrame(
        {
            "global_x": np.arange(12, dtype=float),
            "global_y": np.full(12, 4.0),
            "gene": ["Gad1", "Nkx6-1"] * 6,
            "transcript_score": np.full(12, 0.9),
        }
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline._load_dataset_sdata",
        lambda config: (object(), object(), 16, 16, np.eye(3), points),
    )
    monkeypatch.setattr("merxen.segmentation.pipeline.run_tiled_cellpose", runs)
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.build_cellpose_affine_to_microns",
        lambda *args, **kwargs: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    return runs


def _seg_dir(cfg: SegmentationConfig) -> Path:
    assert cfg.dataset.persistent_mask_path is not None
    return Path(cfg.dataset.persistent_mask_path).parent


def _kill_the_csv_writer_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write two chunks of the ProSeg CSV, then die, on the first call only."""
    state = {"killed": False}

    def _writer(**kwargs: Any) -> dict[str, Any]:
        if state["killed"]:
            return write_proseg_csv_from_points(**kwargs)
        state["killed"] = True
        Path(kwargs["csv_path"]).write_text(
            "transcript_id,x_micron,y_micron,z_micron,feature_name,cell_id\n"
            "0,0.0,4.0,0.0,Gad1,1\n"
        )
        raise Killed

    monkeypatch.setattr(
        "merxen.segmentation.pipeline.write_proseg_csv_from_points", _writer
    )


class TestCellStep:
    @pytest.mark.parametrize("source", _SOURCES)
    def test_a_run_killed_while_writing_the_csv_is_recomputed_not_reused(
        self,
        tmp_path: Path,
        cellpose: CellposeRuns,
        monkeypatch: pytest.MonkeyPatch,
        source: str,
    ) -> None:
        """The incident: masks complete, the ProSeg CSV part-written."""
        cfg = _config(tmp_path, _source_zarr(tmp_path, _source_record(source)))
        seg = _seg_dir(cfg)
        csv = seg / "transcripts_for_proseg.csv"
        _kill_the_csv_writer_once(monkeypatch)

        with pytest.raises(Killed):
            run_cellpose_segmentation(cfg)

        assert cell_mask_marker_path(seg / "cellpose_masks_tiled.npy").exists()
        # The part-written CSV never sits at the path that is reused.
        assert not csv.exists()
        assert (seg / ".transcripts_for_proseg.csv.partial").exists()

        outputs = run_cellpose_segmentation(cfg)

        assert cellpose.calls == 2
        assert len(pd.read_csv(csv)) == 12
        assert not cell_mask_marker_path(seg / "cellpose_masks_tiled.npy").exists()
        assert not (seg / ".transcripts_for_proseg.csv.partial").exists()
        assert segmentation_sidecar_path(seg / "cellpose_masks_tiled.npy").is_file()
        assert outputs["transcripts_csv"].resolve() == csv.resolve()

    @pytest.mark.parametrize("source", _SOURCES)
    def test_a_run_killed_during_tiling_is_recomputed(
        self, tmp_path: Path, cellpose: CellposeRuns, source: str
    ) -> None:
        cfg = _config(tmp_path, _source_zarr(tmp_path, _source_record(source)))
        mask_path = _seg_dir(cfg) / "cellpose_masks_tiled.npy"
        cellpose.kills = [True]

        with pytest.raises(Killed):
            run_cellpose_segmentation(cfg)
        assert cell_mask_marker_path(mask_path).exists()
        run_cellpose_segmentation(cfg)

        assert cellpose.calls == 2
        assert set(np.unique(np.load(mask_path))) == {0, 1, 2}
        assert not cell_mask_marker_path(mask_path).exists()

    def test_the_marker_alone_overrules_complete_looking_outputs(
        self, tmp_path: Path, cellpose: CellposeRuns
    ) -> None:
        """Outputs with a CSV and a sidecar are still recomputed under a marker."""
        cfg = _config(tmp_path, _source_zarr(tmp_path, _record(enabled=True)))
        mask_path = _seg_dir(cfg) / "cellpose_masks_tiled.npy"
        run_cellpose_segmentation(cfg)
        mark_incomplete(cell_mask_marker_path(mask_path), step="test")

        run_cellpose_segmentation(cfg)

        assert cellpose.calls == 2
        assert not cell_mask_marker_path(mask_path).exists()

    def test_an_old_csv_is_not_paired_with_the_masks_of_a_killed_rerun(
        self, tmp_path: Path, cellpose: CellposeRuns
    ) -> None:
        """A forced rerun dies mid-tiling while the earlier, complete CSV exists."""
        cfg = _config(tmp_path, _source_zarr(tmp_path, None))
        seg = _seg_dir(cfg)
        run_cellpose_segmentation(cfg)
        assert (seg / "transcripts_for_proseg.csv").exists()
        cellpose.kills = [True]

        with pytest.raises(Killed):
            run_cellpose_segmentation(cfg, force_rerun=True)

        # The CSV seeded from the replaced masks goes with them.
        assert not (seg / "transcripts_for_proseg.csv").exists()
        assert not (seg / "transcripts_for_proseg.transforms.json").exists()
        run_cellpose_segmentation(cfg)
        assert cellpose.calls == 3

    def test_complete_outputs_are_reused_and_leave_no_marker(
        self, tmp_path: Path, cellpose: CellposeRuns
    ) -> None:
        cfg = _config(tmp_path, _source_zarr(tmp_path, _record(enabled=True)))
        run_cellpose_segmentation(cfg)
        run_cellpose_segmentation(cfg)

        assert cellpose.calls == 1
        assert not [p for p in _seg_dir(cfg).iterdir() if p.name.startswith(".")]

    def test_outputs_from_before_the_markers_are_reused_as_before(
        self, tmp_path: Path, cellpose: CellposeRuns
    ) -> None:
        """Legacy outputs: no marker and no sidecar, bead removal off."""
        cfg = _config(tmp_path, _source_zarr(tmp_path, None))
        run_cellpose_segmentation(cfg)
        segmentation_sidecar_path(_seg_dir(cfg) / "cellpose_masks_tiled.npy").unlink()

        run_cellpose_segmentation(cfg)

        assert cellpose.calls == 1


class TestNucleiStep:
    @pytest.mark.parametrize("source", _SOURCES)
    def test_a_partial_mask_with_labels_is_recomputed_not_reused(
        self, tmp_path: Path, cellpose: CellposeRuns, source: str
    ) -> None:
        cfg = _config(tmp_path, _source_zarr(tmp_path, _source_record(source)))
        mask_path = _seg_dir(cfg) / "cellpose_nuclei_masks_tiled.npy"
        cellpose.kills = [True]

        with pytest.raises(Killed):
            run_cellpose_nuclei_segmentation(cfg)
        assert nuclei_mask_marker_path(mask_path).exists()

        run_cellpose_nuclei_segmentation(cfg)

        assert cellpose.calls == 2
        assert set(np.unique(np.load(mask_path))) == {0, 1, 2}
        assert not nuclei_mask_marker_path(mask_path).exists()
        run_cellpose_nuclei_segmentation(cfg)
        assert cellpose.calls == 2

    def test_the_marker_does_not_match_the_hitl_seed_glob(self, tmp_path: Path) -> None:
        marker = nuclei_mask_marker_path(tmp_path / "cellpose_nuclei_masks_tiled.npy")
        mark_incomplete(marker, step="test")
        assert not list(tmp_path.glob("cellpose_nuclei*"))

    def test_the_hitl_seed_refuses_an_interrupted_nuclei_mask(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.zarr"
        source.mkdir()
        nuclei = tmp_path / "old-segmentation"
        nuclei.mkdir()
        mask = nuclei / "cellpose_nuclei_masks_tiled.npy"
        mask.write_bytes(b"partial")
        mark_incomplete(nuclei_mask_marker_path(mask), step="test")
        config = HitlSeedConfig(
            source_zarr=source,
            destination_zarr=tmp_path / "new-source.zarr",
            nuclei_source_dir=nuclei,
            nuclei_destination_dir=tmp_path / "new-segmentation",
        )

        with pytest.raises(ValueError, match="interrupted"):
            seed_hitl_inputs(config)
        assert not (tmp_path / "new-segmentation" / mask.name).exists()


def _stub_proseg(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    """Replace ProSeg and the conversion; queue "kill-proseg"/"kill-convert"."""

    def _run(*, output_path: Path, **kw: object) -> Path:
        events.append("proseg")
        Path(output_path).mkdir(parents=True, exist_ok=True)
        if "kill-proseg" in events:
            events.remove("kill-proseg")
            raise Killed
        return Path(output_path)

    def _convert(raw_path: Path, latest_path: Path, **kw: object) -> Path:
        events.append("convert")
        shutil.rmtree(latest_path, ignore_errors=True)
        latest_path.mkdir(parents=True)
        (latest_path / "points").mkdir()
        if "kill-convert" in events:
            events.remove("kill-convert")
            raise Killed
        _copy_non_spatialdata_sidecars(raw_path, latest_path)
        (latest_path / "tables").mkdir()
        return latest_path

    monkeypatch.setattr("merxen.segmentation.pipeline.run_proseg_refinement", _run)
    monkeypatch.setattr("merxen.segmentation.pipeline.convert_to_latest_zarr", _convert)


class HybridRuns:
    """Replaces the hybrid refinement; queue "kill" (after its first write) or
    "refuse" (before it changes anything) for the next call."""

    def __init__(self) -> None:
        self.done = False
        self.queue: list[str] = []

    def __call__(
        self,
        store: Path,
        *args: object,
        before_first_write: Callable[[], object] | None = None,
        **kwargs: object,
    ) -> dict[str, int]:
        action = self.queue.pop(0) if self.queue else "complete"
        if action == "refuse":
            raise ValueError("refused before writing")
        if before_first_write is not None:
            before_first_write()
        if action == "kill":
            raise Killed
        self.done = True
        return {"n_cells": 1}


@pytest.fixture
def hybrid(monkeypatch: pytest.MonkeyPatch) -> HybridRuns:
    runs = HybridRuns()
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.has_proseg_hybrid_refinement",
        lambda store, config: runs.done,
    )
    monkeypatch.setattr(
        "merxen.segmentation.pipeline.run_proseg_hybrid_refinement", runs
    )
    return runs


def test_the_hitl_seed_refuses_an_interrupted_source_store(tmp_path: Path) -> None:
    source = tmp_path / "results" / "source_spatialdata.zarr"
    source.mkdir(parents=True)
    mark_incomplete(store_marker_path(source), step="test")
    link = tmp_path / "linked.zarr"
    link.symlink_to(source, target_is_directory=True)
    for given in (source, link):
        config = HitlSeedConfig(
            source_zarr=given, destination_zarr=tmp_path / "new" / "source.zarr"
        )
        with pytest.raises(ValueError, match="interrupted"):
            seed_hitl_inputs(config)
        assert not (tmp_path / "new" / "source.zarr").exists()


class TestProsegStore:
    @staticmethod
    def _prepare(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[SegmentationConfig, Path, dict[str, Any], list[str]]:
        cfg = _config(tmp_path, _source_zarr(tmp_path, None))
        cfg.proseg_hybrid.enabled = False
        latest = tmp_path / "results" / "latest" / "latest_spatialdata.zarr"
        cfg.dataset.persistent_latest_zarr_path = latest
        outputs = run_cellpose_segmentation(cfg)
        kwargs = {
            "transcripts_csv": outputs["transcripts_csv"],
            "cellpose_mask_path": outputs["cellpose_mask_path"],
            "cellpose_cellprob_path": outputs["cellpose_cellprob_path"],
            "transforms_path": outputs["transforms_path"],
        }
        events: list[str] = []
        _stub_proseg(monkeypatch, events)
        return cfg, latest, kwargs, events

    def test_a_proseg_run_killed_mid_write_is_rerun_not_converted(
        self, tmp_path: Path, cellpose: CellposeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg, latest, kwargs, events = self._prepare(tmp_path, monkeypatch)
        raw = Path(cfg.dataset.output_dir) / "proseg_base_raw.zarr"
        events.append("kill-proseg")

        with pytest.raises(Killed):
            run_proseg_segmentation(cfg, **kwargs)
        assert raw.exists() and store_marker_path(raw).exists()

        run_proseg_segmentation(cfg, **kwargs)

        assert events == ["proseg", "proseg", "convert"]
        assert not store_marker_path(raw).exists()
        assert (latest / "tables").is_dir()

    def test_a_store_killed_mid_conversion_is_rebuilt_not_reused(
        self, tmp_path: Path, cellpose: CellposeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The complete raw ProSeg store is converted again; ProSeg is not rerun."""
        cfg, latest, kwargs, events = self._prepare(tmp_path, monkeypatch)
        events.append("kill-convert")

        with pytest.raises(Killed):
            run_proseg_segmentation(cfg, **kwargs)
        assert latest.exists() and store_marker_path(latest).exists()

        run_proseg_segmentation(cfg, **kwargs)

        assert events == ["proseg", "convert", "convert"]
        assert (latest / "tables").is_dir()
        assert (latest / STORE_SIDECAR_NAME).is_file()
        assert not store_marker_path(latest).exists()
        run_proseg_segmentation(cfg, **kwargs)
        assert events == ["proseg", "convert", "convert"]

    def test_a_hybrid_layer_killed_in_place_sends_the_store_back_to_proseg(
        self,
        tmp_path: Path,
        cellpose: CellposeRuns,
        hybrid: HybridRuns,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The refinement rewrites the points in place; they cannot be trusted."""
        cfg, latest, kwargs, events = self._prepare(tmp_path, monkeypatch)
        run_proseg_segmentation(cfg, **kwargs)
        cfg.proseg_hybrid.enabled = True
        hybrid.queue = ["kill"]

        with pytest.raises(Killed):
            run_proseg_segmentation(cfg, **kwargs)
        assert store_marker_path(latest).exists()

        run_proseg_segmentation(cfg, **kwargs)

        assert events == ["proseg", "convert", "proseg", "convert"]
        assert hybrid.done
        assert not store_marker_path(latest).exists()

    def test_a_hybrid_layer_refused_before_writing_keeps_the_store(
        self,
        tmp_path: Path,
        cellpose: CellposeRuns,
        hybrid: HybridRuns,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A refinement that fails before changing the store must not cost ProSeg."""
        cfg, latest, kwargs, events = self._prepare(tmp_path, monkeypatch)
        run_proseg_segmentation(cfg, **kwargs)
        cfg.proseg_hybrid.enabled = True
        hybrid.queue = ["refuse"]

        with pytest.raises(ValueError, match="refused before writing"):
            run_proseg_segmentation(cfg, **kwargs)
        assert not store_marker_path(latest).exists()

        run_proseg_segmentation(cfg, **kwargs)

        assert events == ["proseg", "convert"]
        assert hybrid.done

    def test_leftover_upgrade_folders_go_with_an_interrupted_store(
        self, tmp_path: Path, cellpose: CellposeRuns, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg, latest, kwargs, events = self._prepare(tmp_path, monkeypatch)
        run_proseg_segmentation(cfg, **kwargs)
        orphan = latest.with_name(f".{latest.name}.merxen-upgrade-abc")
        orphan.mkdir()
        mark_incomplete(store_marker_path(latest), step="test")

        run_proseg_segmentation(cfg, **kwargs)

        assert not orphan.exists()
        assert events == ["proseg", "convert", "proseg", "convert"]


class TestCompleteOutputShortcut:
    @staticmethod
    def _prepare(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[SegmentationConfig, Path, list[str]]:
        cfg = _config(tmp_path, _source_zarr(tmp_path, None))
        cfg.proseg_hybrid.enabled = False
        latest = _seg_dir(cfg) / "proseg_base_latest.zarr"
        latest.mkdir(parents=True)
        write_store_segmentation_sidecar(latest, None)
        cfg.dataset.persistent_latest_zarr_path = latest
        run_cellpose_segmentation(cfg)
        run_cellpose_nuclei_segmentation(cfg)
        events: list[str] = []
        _stub_proseg(monkeypatch, events)
        return cfg, latest, events

    @pytest.mark.parametrize("marker", ["cell", "nuclei", "store"])
    def test_an_interrupted_step_is_not_reused_by_the_shortcut(
        self,
        tmp_path: Path,
        cellpose: CellposeRuns,
        monkeypatch: pytest.MonkeyPatch,
        marker: str,
    ) -> None:
        cfg, latest, events = self._prepare(tmp_path, monkeypatch)
        assert cellpose.calls == 2
        seg = _seg_dir(cfg)
        path = {
            "cell": cell_mask_marker_path(seg / "cellpose_masks_tiled.npy"),
            "nuclei": nuclei_mask_marker_path(seg / "cellpose_nuclei_masks_tiled.npy"),
            "store": store_marker_path(latest),
        }[marker]
        mark_incomplete(path, step="test")

        run_segmentation_pipeline(cfg)

        assert not path.exists()
        if marker == "store":
            assert cellpose.calls == 2
            assert events == ["proseg", "convert"]
        else:
            assert cellpose.calls == 3
            assert events == []

    def test_a_hybrid_layer_killed_by_the_shortcut_sends_the_store_back_to_proseg(
        self,
        tmp_path: Path,
        cellpose: CellposeRuns,
        hybrid: HybridRuns,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg, latest, events = self._prepare(tmp_path, monkeypatch)
        cfg.proseg_hybrid.enabled = True
        hybrid.queue = ["kill"]

        with pytest.raises(Killed):
            run_segmentation_pipeline(cfg)
        assert store_marker_path(latest).exists()

        run_segmentation_pipeline(cfg)

        assert events == ["proseg", "convert"]
        assert hybrid.done
        assert cellpose.calls == 2
        assert not store_marker_path(latest).exists()


class TestProtectedBaseline:
    @pytest.mark.parametrize("step", ["cell", "nuclei"])
    def test_a_misdirected_arm_deletes_nothing_in_a_protected_root(
        self, tmp_path: Path, cellpose: CellposeRuns, step: str
    ) -> None:
        """A custom arm pointed at a baseline's folder is refused before discarding."""
        baseline = _config(tmp_path, _source_zarr(tmp_path, None))
        run_cellpose_segmentation(baseline)
        run_cellpose_nuclei_segmentation(baseline)
        seg = _seg_dir(baseline)
        marker = {
            "cell": cell_mask_marker_path(seg / "cellpose_masks_tiled.npy"),
            "nuclei": nuclei_mask_marker_path(seg / "cellpose_nuclei_masks_tiled.npy"),
        }[step]
        # The baseline is (as far as the marker says) still being written.
        mark_incomplete(marker, step="baseline")
        before = sorted(p.name for p in seg.iterdir())
        arm = baseline.model_copy(deep=True)
        arm.dataset.output_dir = tmp_path / "arm" / "segment_out"
        arm.cellpose_model_policy.experimental_arm = True
        arm.cellpose_model_policy.protected_output_roots = [tmp_path / "results"]
        run = (
            run_cellpose_segmentation
            if step == "cell"
            else run_cellpose_nuclei_segmentation
        )

        with pytest.raises(ValueError, match="protected baseline"):
            run(arm)

        assert sorted(p.name for p in seg.iterdir()) == before


class TestBuildStep:
    @staticmethod
    def _config(tmp_path: Path) -> tuple[SpatialDataBuildConfig, Path, Path]:
        raw = tmp_path / "raw"
        raw.mkdir()
        persistent = tmp_path / "results" / "source.zarr"
        cfg = SpatialDataBuildConfig(
            dataset_name="S1_XENIUM",
            platform="XENIUM",
            input_path=raw,
            output_path=tmp_path / "work" / "source.zarr",
            persistent_output_path=persistent,
        )
        return cfg, raw, persistent

    def test_a_build_killed_mid_write_is_rebuilt_not_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg, raw, persistent = self._config(tmp_path)
        calls: list[bool] = []

        def _write(*, output_path: Path, **kwargs: object) -> Path:
            output_path.mkdir(parents=True)
            (output_path / "zarr.json").write_text("{}")
            calls.append(True)
            if len(calls) == 1:
                raise Killed
            (output_path / "points").mkdir()
            return output_path

        monkeypatch.setattr(
            "merxen.io.builders.xenium.write_xenium_spatialdata", _write
        )

        with pytest.raises(Killed):
            build_spatialdata_artifact(cfg)
        assert incomplete_marker_path(persistent).exists()

        out = build_spatialdata_artifact(cfg)

        assert len(calls) == 2
        assert (out / "points").is_dir()
        assert not incomplete_marker_path(persistent).exists()
        build_spatialdata_artifact(cfg)
        assert len(calls) == 2

    def test_an_interrupted_store_without_raw_input_is_not_reused(
        self, tmp_path: Path
    ) -> None:
        cfg, raw, persistent = self._config(tmp_path)
        shutil.rmtree(raw)
        persistent.mkdir(parents=True)
        mark_incomplete(incomplete_marker_path(persistent), step="test")

        with pytest.raises(FileNotFoundError):
            build_spatialdata_artifact(cfg)
        assert not persistent.exists()


class TestHelpers:
    def test_atomic_write_replaces_and_leaves_no_temporary(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "a" / "record.json"
        write_text_atomic(target, "one\n")
        write_text_atomic(target, "two\n")
        assert target.read_text() == "two\n"
        assert [p.name for p in target.parent.iterdir()] == ["record.json"]

    def test_marker_round_trip(self, tmp_path: Path) -> None:
        marker = incomplete_marker_path(tmp_path / "store.zarr")
        assert marker.name == ".store.zarr.merxen-incomplete"
        mark_incomplete(marker, step="build")
        assert marker.exists()
        clear_incomplete(marker)
        clear_incomplete(marker)
        assert not marker.exists()

    def test_leftover_temporaries_match_only_their_output(self, tmp_path: Path) -> None:
        mask = tmp_path / "cellpose_masks_tiled.npy"
        names = [
            ".cellpose_masks_tiled.npy.tmp.123.npy",
            ".cellpose_masks_tiled.npy.tmp-45",
            ".cellpose_masks_tiled.npy.partial",
            ".cellpose_nuclei_masks_tiled.npy.tmp.1.npy",
            "cellpose_masks_tiled.npy",
        ]
        for name in names:
            (tmp_path / name).write_text("x")
        assert [p.name for p in leftover_temporaries(mask)] == sorted(names[:3])
