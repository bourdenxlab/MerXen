"""Cellpose on an ATERA store: DAPI + 18S, no ProSeg CSV (plan AT-7).

Cellpose itself is replaced (``run_tiled_cellpose``); the store is the AT-3
synthetic mini-Atera bundle built into a source store.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_io.atera_native_fixture import build_atera_source
from test_io.atera_synth import PIXEL_SIZE_UM

from merxen.config import SegmentationConfig
from merxen.segmentation import pipeline
from merxen.segmentation.model_identity import CellposeInferenceContext
from merxen.segmentation.pipeline import (
    _derive_cellpose_transforms,
    _load_dataset_sdata,
    run_cellpose_segmentation,
)


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_atera_source(tmp_path_factory.mktemp("atera_cellpose"))


@pytest.fixture(scope="module")
def source_4ch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_atera_source(
        tmp_path_factory.mktemp("atera_cellpose_4ch"), n_channels=4
    )


def _config(source: Path, out: Path, **dataset: Any) -> SegmentationConfig:
    values: dict[str, Any] = {
        "name": "ATERA_SYNTH",
        "platform": "ATERA",
        "data_path": source,
        "channels": ["DAPI", "18S"],
        "output_dir": out / "segment_out",
        "image_key": "morphology_focus",
        "proseg_transcripts": "none",
        "persistent_mask_path": out / "seg" / "cellpose_masks_tiled.npy",
        "persistent_cellpose_cellprob_path": out
        / "seg"
        / "cellpose_cellprobs_tiled.npy",
        "persistent_cellpose_stitching_stats_path": out
        / "seg"
        / "cellpose_stitching_stats.json",
    }
    values.update(dataset)
    return SegmentationConfig.model_validate(
        {"dataset": values, "mask_filter": {"final_min_area_um2": None}}
    )


def test_atera_dataset_config_rules(source: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="proseg_transcripts='none'"):
        _config(source, tmp_path, proseg_transcripts="csv")
    with pytest.raises(ValueError, match="needs image_key"):
        _config(source, tmp_path, image_key=None)
    with pytest.raises(ValueError, match="is for ATERA"):
        SegmentationConfig.model_validate(
            {
                "dataset": {
                    "name": "X",
                    "platform": "XENIUM",
                    "data_path": "x.zarr",
                    "channels": ["DAPI"],
                    "output_dir": "out",
                    "proseg_transcripts": "none",
                }
            }
        )
    # XENIUM and MERSCOPE JSON never names the field: the default is "csv".
    assert (
        SegmentationConfig.model_validate(
            {
                "dataset": {
                    "name": "X",
                    "platform": "XENIUM",
                    "data_path": "x.zarr",
                    "channels": ["DAPI"],
                    "output_dir": "out",
                }
            }
        ).dataset.proseg_transcripts
        == "csv"
    )
    assert CellposeInferenceContext.model_fields["platform"].annotation is not None


def test_custom_atera_model_reuses_complete_masks_without_a_transcript_csv(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merxen.segmentation import model_identity
    from merxen.segmentation.hitl.registry import CellposeModelMetadata
    from merxen.segmentation.model_identity import write_model_identity

    config = _config(source, tmp_path)
    config.tiling.tile_size_candidates = [6144]
    weights = tmp_path / "model"
    weights.write_bytes(b"custom synthetic checkpoint")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata_path = tmp_path / "model.json"
    metadata_path.write_text(
        CellposeModelMetadata(
            weights_sha256=digest,
            platform="ATERA",
            species="human",
            channels=config.dataset.channels,
            image_key="morphology_focus",
            pixel_size_um=PIXEL_SIZE_UM,
            tile_size=6144,
            normalization_route="A",
            annotation_style="fixture",
            annotation_guide_version="fixture",
            license_note="Synthetic only",
            status="accepted",
            cellpose_version="4.2.1.1",
            torch_version="2.14.1+cu130",
            evaluated_cellpose=config.cellpose,
            evaluated_mask_filter=config.mask_filter,
        ).model_dump_json()
    )
    config.cellpose.pretrained_model = str(weights)
    config.cellpose.model_sha256 = digest
    config.cellpose.model_metadata_path = metadata_path
    config.cellpose_model_policy.protected_output_roots = [tmp_path / "published"]
    monkeypatch.setattr(
        model_identity, "require_cellpose_runtime", lambda: ("4.2.1.1", "2.14.1+cu130")
    )
    monkeypatch.setattr(
        "merxen.segmentation.hitl.registry.require_cellpose_runtime",
        lambda: ("4.2.1.1", "2.14.1+cu130"),
    )
    calls = []

    def run(**kwargs: Any) -> Path:
        calls.append(True)
        mask_path = kwargs["output_mask_path"]
        masks = np.zeros((kwargs["height"], kwargs["width"]), np.uint32)
        masks[2:8, 2:8] = 1
        np.save(mask_path, masks)
        np.save(kwargs["output_cellprob_path"], np.ones_like(masks, dtype=np.float32))
        identity = kwargs["expected_model_identity"]
        write_model_identity(mask_path, identity)
        kwargs["output_stitching_stats_path"].write_text(
            json.dumps({"model_identity": identity.model_dump(mode="json")})
        )
        return mask_path

    monkeypatch.setattr(pipeline, "run_tiled_cellpose", run)
    outputs = run_cellpose_segmentation(config)
    run_cellpose_segmentation(config)
    assert len(calls) == 1
    assert not list(tmp_path.rglob("transcripts_for_proseg.csv"))
    before = outputs["cellpose_mask_path"].read_bytes()
    config.cellpose.cellprob_threshold = 0
    with pytest.raises(ValueError, match="identity mismatch"):
        run_cellpose_segmentation(config)
    assert outputs["cellpose_mask_path"].read_bytes() == before


def test_loads_the_morphology_image_of_the_store(source: Path, tmp_path: Path) -> None:
    sdata, fetch, height, width, matrix, points = _load_dataset_sdata(
        _config(source, tmp_path)
    )
    assert points is None
    assert set(sdata.images) >= {"morphology_focus"}
    tile = fetch(0, min(8, height), 0, min(8, width))
    assert tile.shape[-1] == 2 or tile.shape[0] == 2
    np.testing.assert_allclose(np.diag(matrix)[:2], [1 / PIXEL_SIZE_UM] * 2)
    x_t, y_t = _derive_cellpose_transforms(_config(source, tmp_path))
    np.testing.assert_allclose(x_t, [PIXEL_SIZE_UM, 0.0, 0.0])
    np.testing.assert_allclose(y_t, [0.0, PIXEL_SIZE_UM, 0.0])


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        ({"channels": ["DAPI", "PolyT"]}, "needs the channels"),
        ({"image_key": "morphology_mip"}, "segments morphology_focus"),
    ],
)
def test_refuses_another_image_or_a_missing_channel(
    source: Path, tmp_path: Path, dataset: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_dataset_sdata(_config(source, tmp_path, **dataset))


def test_four_channel_store_gives_exactly_dapi_and_18s(
    source_4ch: Path, tmp_path: Path
) -> None:
    _sdata, fetch, height, width, _matrix, _points = _load_dataset_sdata(
        _config(source_4ch, tmp_path)
    )
    tile = np.asarray(fetch(0, min(8, height), 0, min(8, width)))
    assert 2 in tile.shape and 4 not in tile.shape


def test_refuses_a_raw_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reads the built store"):
        _load_dataset_sdata(_config(tmp_path / "raw_bundle", tmp_path))


def test_cellpose_run_writes_no_csv_and_reuses_masks(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def fake(
        *,
        output_mask_path: Path,
        output_cellprob_path: Path,
        output_stitching_stats_path: Path,
        height: int,
        width: int,
        before_overwrite: Any = None,
        **_: Any,
    ) -> Path:
        calls.append(1)
        if before_overwrite is not None:
            before_overwrite()
        mask = np.zeros((height, width), dtype=np.uint32)
        mask[2:8, 2:8] = 1
        mask[10:16, 10:16] = 2
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_mask_path, mask)
        np.save(output_cellprob_path, np.zeros(mask.shape, dtype=np.float32))
        output_stitching_stats_path.write_text("{}\n")
        return output_mask_path

    monkeypatch.setattr(pipeline, "run_tiled_cellpose", fake)
    config = _config(source, tmp_path)
    outputs = run_cellpose_segmentation(config)
    assert "transcripts_csv" not in outputs
    assert not (tmp_path / "segment_out" / "transcripts_for_proseg.csv").exists()
    assert not list((tmp_path / "seg").glob("*.csv"))
    assert not list((tmp_path / "seg").glob("*.transforms.json"))
    transforms = json.loads(outputs["transforms_path"].read_text())
    np.testing.assert_allclose(transforms["x_transform"], [PIXEL_SIZE_UM, 0.0, 0.0])
    assert (
        outputs["cellpose_mask_path"].resolve()
        == (tmp_path / "seg" / "cellpose_masks_tiled.npy").resolve()
    )
    identity = tmp_path / "segment_out" / "cellpose_model_identity.json"
    assert identity.exists()
    # A second run reuses the mask and logits (no CSV needed).
    again = run_cellpose_segmentation(config)
    assert calls == [1]
    assert "transcripts_csv" not in again


def test_an_interrupted_cellpose_run_is_not_reused(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review of AT-7: the masks are written in place, tile by tile, and an
    # ATERA run has no CSV to mark completion, so a run killed part-way left
    # files that the next run reused as finished.
    calls: list[str] = []

    def killed(
        *,
        output_mask_path: Path,
        output_cellprob_path: Path,
        height: int,
        width: int,
        before_overwrite: Any = None,
        **_: Any,
    ) -> Path:
        calls.append("killed")
        if before_overwrite is not None:
            before_overwrite()
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask = np.lib.format.open_memmap(
            output_mask_path, mode="w+", dtype=np.uint32, shape=(height, width)
        )
        np.lib.format.open_memmap(
            output_cellprob_path, mode="w+", dtype=np.float32, shape=(height, width)
        )
        mask[2:8, 2:8] = 1  # one tile stitched, then the process dies
        mask.flush()
        raise KeyboardInterrupt

    def complete(
        *,
        output_mask_path: Path,
        output_cellprob_path: Path,
        output_stitching_stats_path: Path,
        height: int,
        width: int,
        before_overwrite: Any = None,
        **_: Any,
    ) -> Path:
        calls.append("complete")
        if before_overwrite is not None:
            before_overwrite()
        mask = np.zeros((height, width), dtype=np.uint32)
        mask[2:8, 2:8] = 1
        mask[10:16, 10:16] = 2
        np.save(output_mask_path, mask)
        np.save(output_cellprob_path, np.zeros(mask.shape, dtype=np.float32))
        output_stitching_stats_path.write_text("{}\n")
        return output_mask_path

    config = _config(source, tmp_path)
    monkeypatch.setattr(pipeline, "run_tiled_cellpose", killed)
    with pytest.raises(KeyboardInterrupt):
        run_cellpose_segmentation(config)
    mask_path = tmp_path / "seg" / "cellpose_masks_tiled.npy"
    assert mask_path.exists() and int(np.load(mask_path).max()) == 1
    monkeypatch.setattr(pipeline, "run_tiled_cellpose", complete)
    outputs = run_cellpose_segmentation(config)
    assert calls == ["killed", "complete"]
    assert int(np.load(outputs["cellpose_mask_path"]).max()) == 2
    # The finished run is reused; a run killed after a finished one is not.
    run_cellpose_segmentation(config)
    assert calls == ["killed", "complete"]
    monkeypatch.setattr(pipeline, "run_tiled_cellpose", killed)
    with pytest.raises(KeyboardInterrupt):
        run_cellpose_segmentation(config, force_rerun=True)
    monkeypatch.setattr(pipeline, "run_tiled_cellpose", complete)
    run_cellpose_segmentation(config)
    assert calls == ["killed", "complete", "killed", "complete"]
