"""Check fixture preparation without downloading or copying model weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from merxen.config import SegmentationConfig


@pytest.fixture
def prepare_fixture() -> ModuleType:
    """Load the standalone fixture preparer without executing it."""
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/acceptance/prepare_hitl1_fixture.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_hitl1_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_is_candidate_and_drops_persistent_baseline_paths(
    prepare_fixture: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"stock fixture")
    monkeypatch.setattr(
        prepare_fixture,
        "CPSAM_V2_SHA256",
        hashlib.sha256(weights.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        prepare_fixture, "require_cellpose_runtime", lambda: ("4.2.1.1", "2.14.0+cu130")
    )
    protected = tmp_path / "published"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "name": "HD1_XENIUM",
                    "platform": "XENIUM",
                    "channels": ["DAPI", "18S"],
                    "data_path": str(source),
                    "output_dir": str(protected),
                    "persistent_mask_path": str(protected / "cellpose_masks_tiled.npy"),
                    "persistent_latest_zarr_path": str(protected / "latest.zarr"),
                }
            }
        )
    )
    output = tmp_path / "fixture"
    assert (
        prepare_fixture.main(
            [
                "--config",
                str(config),
                "--weights",
                str(weights),
                "--source-zarr",
                str(source),
                "--output-dir",
                str(output),
                "--protected-root",
                str(protected),
                "--pixel-size-um",
                "0.2125",
                "--tile-size",
                "6144",
            ]
        )
        == 0
    )
    generated = SegmentationConfig.model_validate_json(
        (output / "segment_config.json").read_text()
    )
    assert generated.dataset.persistent_mask_path is None
    assert generated.dataset.persistent_latest_zarr_path is None
    assert generated.dataset.output_dir == output / "arm"
    assert generated.cellpose_model_policy.allow_candidate_model
    assert json.loads((output / "model.json").read_text())["status"] == "candidate"
    assert (output / "model").is_symlink()
    assert weights.read_bytes() == b"stock fixture"
    assert not protected.exists()


def test_fixture_refuses_protected_output_before_reading_inputs(
    prepare_fixture: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="outside the protected"):
        prepare_fixture.main(
            [
                "--config",
                "missing.json",
                "--weights",
                "missing",
                "--source-zarr",
                "missing.zarr",
                "--output-dir",
                str(tmp_path / "published/trial"),
                "--protected-root",
                str(tmp_path / "published"),
            ]
        )


def test_atera_fixture_uses_native_pixel_size_species_and_recorded_grid(
    prepare_fixture: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.zarr"
    (source / "atera").mkdir(parents=True)
    (source / "atera/experiment.spatial").write_text('{"pixel_size": 0.21249222}')
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixture")
    monkeypatch.setattr(
        prepare_fixture,
        "CPSAM_V2_SHA256",
        hashlib.sha256(weights.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        prepare_fixture, "require_cellpose_runtime", lambda: ("4.2.1.1", "2.14.1+cu130")
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "dataset": {
                    "name": "ATERA_SYNTH",
                    "platform": "ATERA",
                    "data_path": str(source),
                    "output_dir": "unused",
                    "image_key": "morphology_focus",
                    "channels": ["DAPI", "18S"],
                    "proseg_transcripts": "none",
                },
                "cellpose_model_policy": {"species": "mouse"},
            }
        )
    )
    stats = tmp_path / "stats.json"
    stats.write_text('{"tile_size": 4096, "stitch_overlap_px": 128}')
    output = tmp_path / "fixture"
    args = [
        "--config",
        str(config),
        "--weights",
        str(weights),
        "--source-zarr",
        str(source),
        "--output-dir",
        str(output),
        "--protected-root",
        str(tmp_path / "published"),
        "--baseline-stats",
        str(stats),
    ]
    assert prepare_fixture.main(args) == 0
    metadata = json.loads((output / "model.json").read_text())
    assert metadata["platform"] == "ATERA"
    assert metadata["species"] == "mouse"
    assert metadata["pixel_size_um"] == pytest.approx(0.21249222)
    assert metadata["tile_size"] == 4096
    generated = SegmentationConfig.model_validate_json(
        (output / "segment_config.json").read_text()
    )
    assert generated.dataset.proseg_transcripts == "none"
    assert generated.tiling.stitch_overlap_px == 128
    params = json.loads((output / "nextflow-params.json").read_text())
    assert params["atera_cellpose_model"] == str(output)
    assert "xenium_cellpose_model" not in params
    assert params["segmentation_source"] == "cellpose_proseg"
    with pytest.raises(ValueError, match="differs from the baseline"):
        prepare_fixture.main(
            [
                *args[:-4],
                "--protected-root",
                str(tmp_path / "published"),
                "--baseline-stats",
                str(stats),
                "--tile-size",
                "6144",
                "--output-dir",
                str(tmp_path / "mismatch"),
            ]
        )
