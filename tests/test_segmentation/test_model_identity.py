"""Tests for explicit Cellpose weights and truthful inference provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merxen.config import (
    CellposeConfig,
    EnrichmentConfig,
    MaskFilterConfig,
    MemoryConfig,
    TilingConfig,
)
from merxen.enrichment.enrich import _record_mask_model_identities
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    register_segmentation_branch,
)
from merxen.segmentation import cellpose
from merxen.segmentation import model_identity as identity_module
from merxen.segmentation.model_identity import (
    CellposeInferenceContext,
    CellposeModelIdentity,
    ensure_model_identity,
    find_model_identity_path,
    identify_cellpose_model,
    model_identity_path,
    read_model_identity,
    require_cellpose_runtime,
    write_model_identity,
)


@pytest.fixture
def pinned_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the verified production package versions without loading torch."""
    monkeypatch.setattr(
        identity_module,
        "version",
        {"cellpose": "4.2.1.1", "torch": "2.14.0+cu130"}.__getitem__,
    )
    monkeypatch.setattr(identity_module, "_torch_version", lambda: "2.14.0+cu130")


def _identity() -> CellposeModelIdentity:
    return CellposeModelIdentity(
        status="verified",
        model_name="cpsam_v2",
        weights_sha256=identity_module.CPSAM_V2_SHA256,
        cellpose_version="4.2.1.1",
        torch_version="2.14.0+cu130",
        cellpose_config=CellposeConfig(),
    )


def test_default_models_explicitly_select_float32_cpsam_v2() -> None:
    """The Python default should match production's model and precision."""
    config = CellposeConfig()
    assert config.pretrained_model == "cpsam_v2"
    assert config.model_type is None
    assert not config.use_bfloat16


def test_legacy_model_type_warns_without_selecting_cyto3() -> None:
    """Old JSON remains readable while explaining the ignored option."""
    with pytest.warns(DeprecationWarning, match="ignores model_type"):
        config = CellposeConfig(model_type="cyto3")
    assert config.pretrained_model == "cpsam_v2"


@pytest.mark.usefixtures("pinned_runtime")
@pytest.mark.parametrize("use_bfloat16", [False, True])
def test_builder_always_passes_weights_and_precision(
    monkeypatch: pytest.MonkeyPatch, use_bfloat16: bool
) -> None:
    """Precision must reach Cellpose even with model_type=None."""
    calls: list[dict[str, object]] = []
    model = object()

    def construct(**kwargs: object) -> object:
        calls.append(kwargs)
        return model

    monkeypatch.setattr(
        cellpose,
        "cellpose_models",
        SimpleNamespace(MODEL_NAMES=["cpsam_v2"], CellposeModel=construct),
    )
    assert (
        cellpose.build_cellpose_model(
            CellposeConfig(gpu=False, use_bfloat16=use_bfloat16)
        )
        is model
    )
    assert calls == [
        {"gpu": False, "pretrained_model": "cpsam_v2", "use_bfloat16": use_bfloat16}
    ]


@pytest.mark.usefixtures("pinned_runtime")
def test_missing_checkpoint_fails_before_cellpose_can_fall_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A misspelled model path must never select a stock model silently."""
    monkeypatch.setattr(
        cellpose,
        "cellpose_models",
        SimpleNamespace(
            MODEL_NAMES=["cpsam_v2"],
            CellposeModel=lambda **kwargs: pytest.fail("must fail before loading"),
        ),
    )
    with pytest.raises(FileNotFoundError, match="pretrained_model"):
        cellpose.build_cellpose_model(
            CellposeConfig(pretrained_model=str(tmp_path / "missing"))
        )


@pytest.mark.usefixtures("pinned_runtime")
def test_builder_refuses_wrong_checksum_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "model"
    weights.write_bytes(b"changed checkpoint")
    monkeypatch.setattr(
        cellpose,
        "cellpose_models",
        SimpleNamespace(
            MODEL_NAMES=["cpsam_v2"],
            CellposeModel=lambda **_: pytest.fail("wrong weights must not be loaded"),
        ),
    )
    with pytest.raises(ValueError, match="model_sha256 mismatch"):
        cellpose.build_cellpose_model(
            CellposeConfig(
                pretrained_model=str(weights),
                model_sha256="0" * 64,
            )
        )


@pytest.mark.usefixtures("pinned_runtime")
def test_runtime_retains_cuda_version_suffix() -> None:
    """Local torch build information is part of the recorded provenance."""
    assert require_cellpose_runtime() == ("4.2.1.1", "2.14.0+cu130")


@pytest.mark.parametrize(
    ("cellpose_version", "torch_version"),
    [("4.1.1", "2.14.0"), ("4.2.1.1", "2.13.0"), ("4.2.1.1", "2.15.0")],
)
def test_runtime_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch, cellpose_version: str, torch_version: str
) -> None:
    """Changing model APIs or torch minor versions needs a deliberate update."""
    monkeypatch.setattr(
        identity_module,
        "version",
        {"cellpose": cellpose_version, "torch": torch_version}.__getitem__,
    )
    monkeypatch.setattr(identity_module, "_torch_version", lambda: torch_version)
    with pytest.raises(RuntimeError, match="Rebuild the segmentation environment"):
        require_cellpose_runtime()


@pytest.mark.usefixtures("pinned_runtime")
def test_identity_hashes_the_actual_loaded_file(tmp_path: Path) -> None:
    """The identity should describe checkpoint contents, rather than a filename."""
    weights = tmp_path / "candidate"
    weights.write_bytes(b"synthetic weights")
    config = CellposeConfig(pretrained_model=str(weights))
    identity = identify_cellpose_model(
        SimpleNamespace(pretrained_model=weights), config
    )
    assert identity.weights_sha256 == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert identity.torch_version == "2.14.0+cu130"
    assert identity.cellpose_config == config
    assert identity.status == "verified"


@pytest.mark.usefixtures("pinned_runtime")
def test_stock_checkpoint_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    """A file named cpsam_v2 does not establish the expected model identity."""
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"altered weights")
    with pytest.raises(ValueError, match="checksum mismatch"):
        identify_cellpose_model(
            SimpleNamespace(pretrained_model=weights), CellposeConfig()
        )


@pytest.mark.usefixtures("pinned_runtime")
def test_loaded_fallback_is_rejected(tmp_path: Path) -> None:
    """The resolved checkpoint must be the requested checkpoint."""
    requested = tmp_path / "candidate"
    requested.write_bytes(b"candidate")
    with pytest.raises(ValueError, match="different weights"):
        identify_cellpose_model(
            SimpleNamespace(pretrained_model=tmp_path / "cpsam_v2"),
            CellposeConfig(pretrained_model=str(requested)),
        )


@pytest.mark.parametrize("prefix", ["cellpose", "cellpose_nuclei"])
def test_model_identity_round_trip_and_reuse_preserves_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """A new task must keep the identity from the task that made the mask."""
    mask = tmp_path / f"{prefix}_masks_tiled.npy"
    identity = _identity()
    path = write_model_identity(mask, identity)
    assert path.name == f"{prefix}_model_identity.json"
    assert path == model_identity_path(mask)
    original_bytes, original_mtime = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(
        identity_module, "version", lambda name: pytest.fail("reuse queried runtime")
    )
    stats = tmp_path / "stats.json"
    stats.write_text('{"final_labels": 7}')
    assert read_model_identity(mask) == identity
    assert ensure_model_identity(mask, stats) == identity
    assert path.read_bytes() == original_bytes
    assert path.stat().st_mtime_ns == original_mtime
    assert json.loads(stats.read_text()) == {
        "final_labels": 7,
        "model_identity": identity.model_dump(mode="json"),
    }


def test_legacy_identity_does_not_claim_current_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No historical weights, config or version can be inferred from today's env."""
    monkeypatch.setattr(
        identity_module, "version", lambda name: pytest.fail("legacy queried runtime")
    )
    mask = tmp_path / "cellpose_masks_tiled.npy"
    identity = ensure_model_identity(mask, tmp_path / "stats.json")
    assert identity.status == "unverified"
    assert identity.weights_sha256 is None
    assert identity.torch_version is None
    assert identity.cellpose_version is None
    assert identity.cellpose_config is None
    assert ".command.log" in str(identity.evidence)
    assert read_model_identity(mask) == identity


@pytest.mark.parametrize("record_location", ["original", "published"])
def test_staged_mask_finds_identity_along_its_symlink_chain(
    tmp_path: Path, record_location: str
) -> None:
    """A published sidecar may live beside an intermediate symlink, not its target."""
    original = tmp_path / "original" / "cellpose_masks_tiled.npy"
    published = tmp_path / "published" / original.name
    staged = tmp_path / "downstream" / original.name
    for path in (original, published, staged):
        path.parent.mkdir()
    np.save(original, np.ones((2, 2), dtype=np.uint32))
    published.symlink_to(Path("..") / "original" / original.name)
    staged.symlink_to(published)
    sidecar = write_model_identity(
        original if record_location == "original" else published, _identity()
    )
    assert find_model_identity_path(staged) == sidecar
    assert read_model_identity(staged) == _identity()
    assert not model_identity_path(staged).exists()


def test_model_identity_lookup_rejects_a_symlink_cycle(tmp_path: Path) -> None:
    """An invalid staging chain must fail without an endless traversal."""
    first, second = tmp_path / "first.npy", tmp_path / "second.npy"
    first.symlink_to(second)
    second.symlink_to(first)
    with pytest.raises(ValueError, match="Symlink cycle"):
        find_model_identity_path(first)


def test_conflicting_sidecar_and_stats_refuse_without_overwriting(
    tmp_path: Path,
) -> None:
    """Contradictory provenance is a failure, never an in-place repair."""
    mask = tmp_path / "cellpose_masks_tiled.npy"
    identity = _identity()
    sidecar = write_model_identity(mask, identity)
    stats = tmp_path / "stats.json"
    stats.write_text('{"model_identity": {"status": "unverified"}}')
    before = sidecar.read_bytes(), stats.read_bytes()
    with pytest.raises(ValueError, match="disagrees"):
        ensure_model_identity(mask, stats)
    assert (sidecar.read_bytes(), stats.read_bytes()) == before


def test_hitl0_sidecar_remains_reusable_without_rewriting_optional_fields(
    tmp_path: Path,
) -> None:
    """New optional provenance fields must not invalidate completed HITL-0 masks."""
    mask = tmp_path / "cellpose_masks_tiled.npy"
    payload = _identity().model_dump(mode="json")
    payload.pop("inference_context")
    payload["cellpose_config"].pop("model_sha256")
    payload["cellpose_config"].pop("model_metadata_path")
    sidecar = model_identity_path(mask)
    sidecar.write_text(json.dumps(payload))
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"model_identity": payload}))
    before = sidecar.read_bytes(), stats.read_bytes()
    assert ensure_model_identity(mask, stats) == _identity()
    assert (sidecar.read_bytes(), stats.read_bytes()) == before


@pytest.mark.usefixtures("pinned_runtime")
@pytest.mark.parametrize("write_stats", [False, True])
def test_tiled_inference_records_loaded_identity_with_aligned_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_stats: bool
) -> None:
    """A synthetic full run exercises model provenance without GPU/model downloads."""
    weights = tmp_path / "candidate"
    weights.write_bytes(b"synthetic weights")

    class Model:
        pretrained_model = weights

        def eval(self, image: np.ndarray, **kwargs: object) -> tuple:
            del kwargs
            mask = np.zeros(image.shape[:2], dtype=np.uint32)
            mask[2:6, 2:6] = 1
            return mask, [None, None, np.full(mask.shape, 3.0)], None

    monkeypatch.setattr(cellpose, "build_cellpose_model", lambda config: Model())
    monkeypatch.setattr(cellpose, "choose_working_tile_size", lambda **kwargs: 8)
    mask_path = tmp_path / "cellpose_masks_tiled.npy"
    stats_path = tmp_path / "stats.json"
    probs_path = tmp_path / "probabilities.npy"
    config = CellposeConfig(pretrained_model=str(weights))
    expected = identify_cellpose_model(Model(), config)
    expected.inference_context = CellposeInferenceContext(
        platform="XENIUM",
        species="human",
        channels=["DAPI", "18S"],
        image_key="morphology_focus",
        pixel_size_um=(0.2125, 0.2125),
        mask_filter=MaskFilterConfig(),
        tiling=TilingConfig(),
    )
    cellpose.run_tiled_cellpose(
        fetch_tile_fn=lambda *bbox: np.arange(64, dtype=np.uint16).reshape(8, 8, 1),
        height=8,
        width=8,
        dataset_name="synthetic",
        output_mask_path=mask_path,
        cellpose_config=config,
        expected_model_identity=expected,
        mask_filter_config=MaskFilterConfig(),
        tiling_config=TilingConfig(
            tile_size_candidates=[8],
            min_tile_size=8,
            stitch_overlap_px=0,
            filter_per_tile=False,
            write_stitching_stats=write_stats,
        ),
        memory_config=MemoryConfig(),
        output_stitching_stats_path=stats_path,
        output_cellprob_path=probs_path,
    )
    mask, probs = np.load(mask_path), np.load(probs_path)
    assert int(np.count_nonzero(mask)) == 16
    assert np.all(probs[mask > 0] == 3.0)
    assert np.all(probs[mask == 0] == 0.0)
    identity = read_model_identity(mask_path)
    assert identity.status == "verified"
    assert identity.model_name == str(weights)
    assert json.loads(stats_path.read_text())["model_identity"] == identity.model_dump(
        mode="json"
    )


@pytest.mark.usefixtures("pinned_runtime")
def test_bad_weights_leave_existing_outputs_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed model validation must happen before run_tiled_cellpose's cleanup."""
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"bad weights")
    monkeypatch.setattr(
        cellpose,
        "build_cellpose_model",
        lambda config: SimpleNamespace(pretrained_model=weights),
    )
    mask = tmp_path / "cellpose_masks_tiled.npy"
    mask.write_bytes(b"existing mask")
    sidecar = write_model_identity(mask, _identity())
    before = mask.read_bytes(), sidecar.read_bytes()
    with pytest.raises(ValueError, match="checksum mismatch"):
        cellpose.run_tiled_cellpose(
            fetch_tile_fn=lambda *args: pytest.fail("must fail before reading image"),
            height=8,
            width=8,
            dataset_name="synthetic",
            output_mask_path=mask,
            cellpose_config=CellposeConfig(),
            mask_filter_config=MaskFilterConfig(),
            tiling_config=TilingConfig(),
            memory_config=MemoryConfig(),
        )
    assert (mask.read_bytes(), sidecar.read_bytes()) == before


@pytest.mark.parametrize("already_registered", [False, True])
def test_enrichment_and_later_registration_keep_original_model_provenance(
    tmp_path: Path, already_registered: bool
) -> None:
    """Tables may be registered after enrichment; provenance must survive both."""
    cell_mask = tmp_path / "cellpose_masks_tiled.npy"
    nuclei_mask = tmp_path / "cellpose_nuclei_masks_tiled.npy"
    identity = _identity()
    write_model_identity(cell_mask, identity)
    staged_mask = tmp_path / "downstream" / cell_mask.name
    staged_mask.parent.mkdir()
    staged_mask.symlink_to(cell_mask)
    # A legacy nuclei mask has no known weights or runtime.
    config = EnrichmentConfig(
        dataset_name="S1_XENIUM",
        platform="XENIUM",
        latest_zarr_path=tmp_path / "latest.zarr",
        original_data_path=tmp_path / "original.zarr",
        mask_path=staged_mask,
        nuclei_mask_path=nuclei_mask,
        output_dir=tmp_path / "enrich",
    )
    registry = {"cellpose": {}, "cellpose_nuclei": {}} if already_registered else {}
    dst = SimpleNamespace(
        attrs={
            MERXEN_SCHEMA_ATTR: {
                "writer_versions": {"merxen": "writer-runtime"},
                "segmentations": registry,
            }
        },
        points={},
    )
    assert _record_mask_model_identities(dst, config)
    schema = dst.attrs[MERXEN_SCHEMA_ATTR]
    assert schema["writer_versions"] == {"merxen": "writer-runtime"}
    assert schema["model_identities"]["cellpose"] == identity.model_dump(mode="json")
    assert schema["model_identities"]["cellpose_nuclei"]["status"] == "unverified"
    for branch in ("cellpose", "cellpose_nuclei"):
        register_segmentation_branch(
            dst,
            branch,
            points_key="transcripts",
            assignment_column=None,
            shape_key=branch,
            table_key=None,
            id_namespace="test",
        )
    registered = dst.attrs[MERXEN_SCHEMA_ATTR]["segmentations"]
    assert registered["cellpose"]["model_identity"] == identity.model_dump(mode="json")
    assert registered["cellpose_nuclei"]["model_identity"]["status"] == "unverified"
    # Missing ancillary files later must not erase provenance already in the zarr.
    model_identity_path(cell_mask).unlink()
    assert not _record_mask_model_identities(dst, config)
    assert registered["cellpose"]["model_identity"]["status"] == "verified"
