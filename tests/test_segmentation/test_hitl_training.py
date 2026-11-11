"""Managed annotation import, named publication and independent late starts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from merxen.segmentation.hitl import training
from merxen.segmentation.hitl.import_annotations import (
    artifact_sha256,
    import_annotation_packages,
)
from merxen.segmentation.hitl.seed_inputs import HitlSeedConfig, seed_hitl_inputs
from merxen.segmentation.hitl.training import (
    CellposeTrainingConfig,
    train_cellpose_model,
    training_build_identity,
    training_paths,
)


def save_receipt(package: Path, masks: np.ndarray) -> None:
    """Record a numeric GUI save without ever creating a pickle payload."""
    manifest = json.loads((package / "manifest.json").read_text())
    np.savez_compressed(package / "masks/crop_001.npz", masks=masks)
    receipt = {
        "schema_version": 1,
        "package_id": package.name,
        "manifest_sha256": artifact_sha256(package / "manifest.json"),
        "crop_id": "crop_001",
        "cellpose_version": "4.2.1.1",
        "image_sha256": manifest["crops"][0]["files_sha256"]["crop_001.tif"],
        "seg_sha256": artifact_sha256(package / "crop_001_seg.npy"),
        "masks_sha256": artifact_sha256(package / "masks/crop_001.npz"),
        "cell_count": int(np.count_nonzero(np.unique(masks))),
    }
    (package / "receipts/crop_001.json").write_text(json.dumps(receipt))


@pytest.fixture
def returned_package(tmp_path: Path) -> Path:
    """Create one small completed package with opaque, non-loadable GUI output."""
    package = tmp_path / "hitl/annotations/hitl-synthetic"
    for directory in ("raw", "model_input", "receipts", "masks"):
        (package / directory).mkdir(parents=True)
    image = np.random.default_rng(2).random((3, 256, 256), dtype=np.float32)
    image[2] = 0
    np.save(package / "model_input/crop_001.npy", image)
    np.save(package / "raw/crop_001.npy", image)
    (package / "crop_001.tif").write_bytes(b"immutable GUI image")
    (package / "crop_001_seg.npy").write_bytes(b"opaque GUI file, never unpickle")
    files = ("crop_001.tif", "raw/crop_001.npy", "model_input/crop_001.npy")
    context = {
        "platform": "ATERA",
        "image_key": "morphology_focus",
        "pixel_size_um": 0.2125,
        "height": 512,
        "width": 512,
        "tile_size": 1024,
        "overlap_px": 64,
        "source_metadata_sha256": "a" * 64,
        "baseline_stats_sha256": "b" * 64,
    }
    manifest = {
        "managed_annotation_version": 1,
        "package_id": package.name,
        "training_normalize": False,
        "evaluation_eligible": False,
        "platform": "ATERA",
        "model_channels": ["DAPI", "18S"],
        "pixel_size_um": 0.2125,
        "input_normalization": "joint",
        "image_key": "morphology_focus",
        "roi_context": context,
        "source_metadata_sha256": "a" * 64,
        "baseline_stats_sha256": "b" * 64,
        "guide_version": "pilot-v2-managed-v1",
        "crops": [
            {
                "crop_id": "crop_001",
                "split": "pilot_train",
                "bbox_y0_y1_x0_x1": [64, 320, 64, 320],
                "files_sha256": {
                    name: artifact_sha256(package / name) for name in files
                },
            }
        ],
    }
    (package / "manifest.json").write_text(json.dumps(manifest))
    (package / "review.tsv").write_text(
        "crop_id\tstatus\treason\n crop_001\treviewed\t\n".replace("\n ", "\n")
    )
    masks = np.zeros((256, 256), np.uint32)
    masks[25:65, 25:65] = 7
    masks[100:155, 100:155] = 20
    save_receipt(package, masks)
    return package


def test_import_uses_numeric_masks_and_canonicalizes_labels(
    returned_package: Path,
) -> None:
    corpus = import_annotation_packages([returned_package])
    assert corpus.cell_count == 2
    assert corpus.channels == ("DAPI", "18S")
    np.testing.assert_array_equal(np.unique(corpus.crops[0].masks), [0, 1, 2])
    assert artifact_sha256(corpus.crops[0].model_input_path)


@pytest.mark.parametrize(
    "problem", ["pending", "image", "receipt", "disconnected", "held_out"]
)
def test_import_refuses_unfinished_changed_or_invalid_annotations(
    returned_package: Path, problem: str
) -> None:
    package = returned_package
    if problem == "pending":
        path = package / "review.tsv"
        path.write_text(path.read_text().replace("reviewed", "pending"))
    elif problem == "image":
        (package / "crop_001.tif").write_bytes(b"changed")
    elif problem == "receipt":
        (package / "crop_001_seg.npy").write_bytes(b"saved outside managed session")
    elif problem == "disconnected":
        masks = np.zeros((256, 256), np.uint32)
        masks[10:20, 10:20] = masks[40:50, 40:50] = 1
        save_receipt(package, masks)
    else:
        path = package / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["evaluation_eligible"] = True
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        import_annotation_packages([package])


def test_training_identity_ignores_name_and_output_location(
    returned_package: Path,
) -> None:
    config = CellposeTrainingConfig(
        hitl_root=returned_package.parent.parent,
        annotation_package_ids=[returned_package.name],
        model_name="pilot-one",
        platform="ATERA",
        species="human",
        gpu=False,
    )
    assert training_paths(config)[1] == [returned_package]
    corpus = import_annotation_packages([returned_package])
    runtime = ("4.2.1.1", "2.14.1+cu130")
    first = training_build_identity(config, corpus, runtime)
    assert first == training_build_identity(
        config.model_copy(update={"model_name": "pilot-two"}), corpus, runtime
    )
    assert first != training_build_identity(
        config.model_copy(update={"seed": 1}), corpus, runtime
    )
    with pytest.raises(ValueError, match="root"):
        training_paths(
            config.model_copy(
                update={"package_dirs": [Path("/outside") / returned_package.name]}
            )
        )


def test_named_training_publishes_once_and_refuses_changed_build(
    returned_package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cellpose.train

    import merxen.segmentation.cellpose

    checkpoint = tmp_path / "base"
    checkpoint.write_bytes(b"synthetic pinned base")
    monkeypatch.setattr(training, "CPSAM_V2_SHA256", artifact_sha256(checkpoint))
    monkeypatch.setattr(
        training, "require_cellpose_runtime", lambda: ("4.2.1.1", "2.14.1+cu130")
    )
    monkeypatch.setattr(
        merxen.segmentation.cellpose,
        "build_cellpose_model",
        lambda _: SimpleNamespace(net=object(), pretrained_model=str(checkpoint)),
    )
    calls = []

    def fit(net: object, **kwargs: Any) -> tuple[str, np.ndarray, list]:
        calls.append(kwargs)
        path = Path(kwargs["save_path"]) / "candidate"
        path.write_bytes(b"synthetic trained weights")
        return str(path), np.array([2.0, 1.0]), []

    monkeypatch.setattr(cellpose.train, "train_seg", fit)
    config = CellposeTrainingConfig(
        hitl_root=returned_package.parent.parent,
        annotation_package_ids=[returned_package.name],
        model_name="pilot-one",
        platform="ATERA",
        species="human",
        gpu=False,
        n_epochs=2,
    )
    result = train_cellpose_model(config, tmp_path / "reports")
    assert not result.reused
    assert calls[0]["normalize"] is False
    assert calls[0]["channel_axis"] == 0
    assert calls[0]["min_train_masks"] == 1
    assert calls[0]["train_data"][0].shape == (3, 256, 256)
    metadata = json.loads((result.bundle_path / "model.json").read_text())
    assert metadata["status"] == "candidate"
    assert metadata["training_tiling"]["tile_size_candidates"] == [1024]
    assert metadata["evaluated_cellpose"]["flow_threshold"] == 0.7
    assert (
        json.loads((tmp_path / "reports/selector.json").read_text())["model_name"]
        == "pilot-one"
    )
    assert train_cellpose_model(config, tmp_path / "reports").reused
    assert len(calls) == 1
    with pytest.raises(ValueError, match="already used"):
        train_cellpose_model(
            config.model_copy(update={"seed": 3}), tmp_path / "reports"
        )


def test_seed_copy_is_independent_resumable_and_protected(tmp_path: Path) -> None:
    source = tmp_path / "published/source.zarr"
    source.mkdir(parents=True)
    (source / "zarr.json").write_text("original")
    destination = tmp_path / "trial/source.zarr"
    config = HitlSeedConfig(
        source_zarr=source,
        destination_zarr=destination,
        protected_output_roots=[source.parent],
    )
    result = seed_hitl_inputs(config)
    assert not result.reused
    assert (source / "zarr.json").stat().st_ino != (
        destination / "zarr.json"
    ).stat().st_ino
    (destination / "zarr.json").write_text("changed only in new run")
    assert (source / "zarr.json").read_text() == "original"
    assert seed_hitl_inputs(config).reused
    for target in (source, source.parent / "bad.zarr"):
        with pytest.raises(ValueError):
            seed_hitl_inputs(config.model_copy(update={"destination_zarr": target}))


def test_seed_copies_stock_nuclei_without_overwriting_conflicts(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    source.mkdir()
    nuclei = tmp_path / "old-segmentation"
    nuclei.mkdir()
    (nuclei / "cellpose_nuclei_masks_tiled.npy").write_bytes(b"nuclei")
    target = tmp_path / "new-segmentation"
    config = HitlSeedConfig(
        source_zarr=source,
        destination_zarr=tmp_path / "new-source.zarr",
        nuclei_source_dir=nuclei,
        nuclei_destination_dir=target,
    )
    seed_hitl_inputs(config)
    assert (target / "cellpose_nuclei_masks_tiled.npy").read_bytes() == b"nuclei"
    (target / "cellpose_nuclei_masks_tiled.npy").write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="different"):
        seed_hitl_inputs(config)
