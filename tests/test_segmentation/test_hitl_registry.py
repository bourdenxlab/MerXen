"""Custom-model preflight and non-destructive cache/protection checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from merxen.config import CellposeConfig, MaskFilterConfig, SegmentationConfig
from merxen.segmentation import model_identity, pipeline
from merxen.segmentation.hitl.registry import (
    CellposeModelMetadata,
    has_custom_model,
    preflight_custom_model,
    require_matching_cached_model,
    require_safe_custom_outputs,
)
from merxen.segmentation.model_identity import (
    CellposeModelIdentity,
    write_model_identity,
)


@pytest.fixture
def custom_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SegmentationConfig:
    """Create a synthetic bundle without loading a checkpoint."""
    weights = tmp_path / "bundle" / "model"
    weights.parent.mkdir()
    weights.write_bytes(b"synthetic checkpoint; never loaded by Cellpose")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata_path = weights.with_name("model.json")
    metadata = CellposeModelMetadata(
        weights_sha256=digest,
        platform="XENIUM",
        species="human",
        channels=["DAPI", "18S"],
        image_key="morphology_focus",
        pixel_size_um=0.2125,
        tile_size=16,
        normalization_route="A",
        annotation_style="fixture",
        annotation_guide_version="fixture",
        license_note="Synthetic test only",
        status="accepted",
        cellpose_version="4.2.1.1",
        torch_version="2.14.0+cu130",
        evaluated_cellpose=CellposeConfig(gpu=False),
        evaluated_mask_filter=MaskFilterConfig(),
    )
    metadata_path.write_text(metadata.model_dump_json())
    monkeypatch.setattr(model_identity, "version", lambda _: "4.2.1.1")
    monkeypatch.setattr(model_identity, "_torch_version", lambda: "2.14.0+cu130")
    return SegmentationConfig.model_validate(
        {
            "dataset": {
                "name": "fixture",
                "platform": "XENIUM",
                "data_path": str(tmp_path / "source.zarr"),
                "channels": ["DAPI", "18S"],
                "output_dir": str(tmp_path / "arm"),
            },
            "cellpose": {
                "pretrained_model": str(weights),
                "model_sha256": digest,
                "model_metadata_path": str(metadata_path),
                "gpu": False,
            },
            "cellpose_model_policy": {
                "protected_output_roots": [str(tmp_path / "published")]
            },
            "tiling": {
                "tile_size_candidates": [16],
                "min_tile_size": 16,
                "stitch_overlap_px": 2,
            },
        }
    )


def _preflight(
    config: SegmentationConfig, pixel_size: float = 0.2125
) -> CellposeModelIdentity:
    return preflight_custom_model(
        config,
        transform_matrix=np.diag([1 / pixel_size, 1 / pixel_size, 1.0]),
        image_key="morphology_focus",
    )


def _edit_metadata(config: SegmentationConfig, **changes: object) -> None:
    path = config.cellpose.model_metadata_path
    assert path is not None
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_custom_bundle_matches_native_affine_and_runtime(
    custom_config: SegmentationConfig,
) -> None:
    assert has_custom_model(custom_config)
    identity = _preflight(custom_config)
    assert identity.status == "verified"
    assert identity.inference_context is not None
    assert identity.inference_context.channels == ["DAPI", "18S"]
    assert identity.inference_context.pixel_size_um == pytest.approx((0.2125, 0.2125))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("weights_sha256", "0" * 64, "checksum"),
        ("channels", ["18S", "DAPI"], "channels/order"),
        ("species", "mouse", "platform or species"),
        ("torch_version", "2.14.0", "runtime"),
        ("status", "candidate", "explicit opt-in"),
        ("status", "rejected", "accepted"),
        ("tile_size", 32, "single candidate"),
        ("normalization_route", "B", "normalization_route"),
    ],
)
def test_incompatible_model_refuses_without_output(
    custom_config: SegmentationConfig, field: str, value: object, message: str
) -> None:
    _edit_metadata(custom_config, **{field: value})
    with pytest.raises(ValueError, match=message):
        _preflight(custom_config)
    assert not custom_config.dataset.output_dir.exists()


def test_candidate_accepts_explicit_opt_in(custom_config: SegmentationConfig) -> None:
    _edit_metadata(custom_config, status="candidate")
    custom_config.cellpose_model_policy.allow_candidate_model = True
    assert _preflight(custom_config).status == "verified"


@pytest.mark.parametrize("pixel_size", [0.22, 0.2])
def test_wrong_pixel_size_refuses(
    custom_config: SegmentationConfig, pixel_size: float
) -> None:
    with pytest.raises(ValueError, match="pixel size"):
        _preflight(custom_config, pixel_size)


def test_pixel_size_uses_rotated_affine(custom_config: SegmentationConfig) -> None:
    matrix = np.array([[0.0, -1 / 0.2125, 100.0], [1 / 0.2125, 0, 200], [0, 0, 1]])
    identity = preflight_custom_model(
        custom_config, transform_matrix=matrix, image_key="morphology_focus"
    )
    assert identity.inference_context is not None
    assert identity.inference_context.pixel_size_um == pytest.approx((0.2125, 0.2125))


def test_missing_weights_refuse(custom_config: SegmentationConfig) -> None:
    Path(custom_config.cellpose.pretrained_model).unlink()
    with pytest.raises(FileNotFoundError):
        _preflight(custom_config)


def test_exact_one_percent_pixel_size_boundary_is_accepted(
    custom_config: SegmentationConfig,
) -> None:
    assert _preflight(custom_config, 0.2125 * 1.01).status == "verified"


def test_explicit_aligned_image_refuses_before_loading_source(
    custom_config: SegmentationConfig,
) -> None:
    custom_config.dataset.image_key = "morphology_focus_aligned_nonrigid"
    with pytest.raises(ValueError, match="native image"):
        pipeline.run_cellpose_segmentation(custom_config)
    assert not custom_config.dataset.output_dir.exists()


@pytest.mark.parametrize("use_alias", [False, True])
@pytest.mark.parametrize("entry", ["cells", "nuclei", "proseg", "combined"])
def test_all_entry_points_protect_baseline_even_when_forced(
    custom_config: SegmentationConfig, tmp_path: Path, use_alias: bool, entry: str
) -> None:
    baseline = tmp_path / "published"
    baseline.mkdir()
    marker = baseline / "original.txt"
    marker.write_text("immutable baseline")
    alias = tmp_path / "alias"
    alias.symlink_to(baseline, target_is_directory=True)
    custom_config.dataset.output_dir = (alias if use_alias else baseline) / "trial"
    with pytest.raises(ValueError, match="protected baseline"):
        if entry == "cells":
            pipeline.run_cellpose_segmentation(custom_config, force_rerun=True)
        elif entry == "nuclei":
            pipeline.run_cellpose_nuclei_segmentation(custom_config, force_rerun=True)
        elif entry == "combined":
            pipeline.run_segmentation_pipeline(custom_config, force_rerun=True)
        else:
            pipeline.run_proseg_segmentation(
                custom_config,
                transcripts_csv="unused",
                cellpose_mask_path="unused",
                cellpose_cellprob_path="unused",
                transforms_path="unused",
                force_rerun=True,
            )
    assert list(baseline.iterdir()) == [marker]
    assert marker.read_text() == "immutable baseline"


def test_protection_covers_persistent_outputs(
    custom_config: SegmentationConfig, tmp_path: Path
) -> None:
    custom_config.dataset.persistent_latest_zarr_path = (
        tmp_path / "published/latest.zarr"
    )
    with pytest.raises(ValueError, match="protected baseline"):
        require_safe_custom_outputs(custom_config)


@pytest.mark.parametrize("target_kind", ["mask", "source", "parent"])
def test_published_links_protect_their_external_work_targets(
    custom_config: SegmentationConfig, tmp_path: Path, target_kind: str
) -> None:
    baseline = tmp_path / "published"
    baseline.mkdir()
    target = tmp_path / "old-work" / "cellpose_masks_tiled.npy"
    target.parent.mkdir()
    np.save(target, np.ones((4, 4), np.uint32))
    (baseline / "mask.npy").symlink_to(target)
    if target_kind == "mask":
        custom_config.dataset.persistent_mask_path = target
    elif target_kind == "source":
        custom_config.dataset.data_path = target
    else:
        custom_config.dataset.output_dir = target.parent
    with pytest.raises(ValueError, match="protected baseline"):
        require_safe_custom_outputs(custom_config)
    assert np.all(np.load(target) == 1)


def test_visible_baseline_mask_link_is_protected(
    custom_config: SegmentationConfig, tmp_path: Path
) -> None:
    baseline = tmp_path / "published"
    baseline.mkdir()
    target = tmp_path / "target.npy"
    np.save(target, np.ones((4, 4), np.uint32))
    link = baseline / "mask.npy"
    link.symlink_to(target)
    custom_config.dataset.persistent_mask_path = link
    with pytest.raises(ValueError, match="protected baseline"):
        require_safe_custom_outputs(custom_config)


def test_custom_model_requires_protected_roots(
    custom_config: SegmentationConfig,
) -> None:
    custom_config.cellpose_model_policy.protected_output_roots = []
    with pytest.raises(ValueError, match="non-empty"):
        require_safe_custom_outputs(custom_config)


def test_cache_refuses_changed_config_without_mutation(
    custom_config: SegmentationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _preflight(custom_config)
    output = custom_config.dataset.output_dir
    output.mkdir()
    mask = output / "cellpose_masks_tiled.npy"
    np.save(mask, np.ones((8, 8), np.uint32))
    sidecar = write_model_identity(mask, identity)
    stats = output / "cellpose_stitching_stats.json"
    stats.write_text(json.dumps({"model_identity": identity.model_dump(mode="json")}))
    require_matching_cached_model(mask, custom_config, identity)
    before = (mask.read_bytes(), sidecar.read_bytes(), stats.read_bytes())
    custom_config.cellpose.cellprob_threshold = 0
    with pytest.raises(ValueError, match="identity mismatch"):
        require_matching_cached_model(mask, custom_config, _preflight(custom_config))
    monkeypatch.setattr(
        pipeline,
        "_load_dataset_sdata",
        lambda _: (
            object(),
            object(),
            8,
            8,
            np.diag([1 / 0.2125, 1 / 0.2125, 1]),
            object(),
        ),
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        pipeline.run_cellpose_segmentation(custom_config, force_rerun=True)
    assert (mask.read_bytes(), sidecar.read_bytes(), stats.read_bytes()) == before


def test_unverified_custom_cache_refuses_without_stamping(
    custom_config: SegmentationConfig,
) -> None:
    output = custom_config.dataset.output_dir
    output.mkdir()
    mask = output / "cellpose_masks_tiled.npy"
    np.save(mask, np.ones((8, 8), np.uint32))
    with pytest.raises(ValueError, match="identity mismatch"):
        require_matching_cached_model(mask, custom_config, _preflight(custom_config))
    assert sorted(path.name for path in output.iterdir()) == [mask.name]


def test_complete_verified_custom_cache_reuses_without_loading_a_model(
    custom_config: SegmentationConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _preflight(custom_config)
    output = custom_config.dataset.output_dir
    output.mkdir()
    mask = output / "cellpose_masks_tiled.npy"
    np.save(mask, np.ones((8, 8), np.uint32))
    np.save(output / "cellpose_cellprobs_tiled.npy", np.ones((8, 8), np.float32))
    (output / "transcripts_for_proseg.csv").write_text("fixture")
    sidecar = write_model_identity(mask, identity)
    stats = output / "cellpose_stitching_stats.json"
    stats.write_text(json.dumps({"model_identity": identity.model_dump(mode="json")}))
    monkeypatch.setattr(
        pipeline,
        "_load_dataset_sdata",
        lambda _: (
            object(),
            object(),
            8,
            8,
            np.diag([1 / 0.2125, 1 / 0.2125, 1]),
            object(),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_derive_cellpose_transforms",
        lambda _: ((0.2125, 0, 0), (0, 0.2125, 0)),
    )
    monkeypatch.setattr(
        pipeline, "_require_seeding_matches_transforms", lambda *a: None
    )
    monkeypatch.setattr(
        pipeline, "run_tiled_cellpose", lambda **_: pytest.fail("must reuse")
    )
    before = mask.read_bytes(), sidecar.read_bytes(), stats.read_bytes()
    result = pipeline.run_cellpose_segmentation(custom_config)
    assert result["cellpose_mask_path"] == mask
    assert (mask.read_bytes(), sidecar.read_bytes(), stats.read_bytes()) == before


@pytest.mark.parametrize("entry", ["combined", "proseg"])
def test_stock_request_refuses_saved_custom_mask_in_reuse_shortcuts(
    custom_config: SegmentationConfig, entry: str
) -> None:
    identity = _preflight(custom_config)
    output = custom_config.dataset.output_dir
    output.mkdir()
    mask = output / "cellpose_masks_tiled.npy"
    np.save(mask, np.ones((8, 8), np.uint32))
    sidecar = write_model_identity(mask, identity)
    custom_config.cellpose = CellposeConfig(gpu=False)
    before = mask.read_bytes(), sidecar.read_bytes()
    with pytest.raises(ValueError, match="model/configuration mismatch"):
        if entry == "combined":
            pipeline.run_segmentation_pipeline(custom_config)
        else:
            pipeline.run_proseg_segmentation(
                custom_config,
                cellpose_mask_path=mask,
                cellpose_cellprob_path="unused",
                transcripts_csv="unused",
                transforms_path="unused",
            )
    assert (mask.read_bytes(), sidecar.read_bytes()) == before


def test_evaluation_settings_warn_without_changing_config(
    custom_config: SegmentationConfig, caplog: pytest.LogCaptureFixture
) -> None:
    custom_config.mask_filter.final_max_area_um2 = 800
    _preflight(custom_config)
    assert "final_max_area_um2" in caplog.text
    assert custom_config.mask_filter.final_max_area_um2 == 800
