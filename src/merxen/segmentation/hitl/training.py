"""Fine-tune the pinned Cellpose model and publish an immutable named bundle."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from merxen import local_config
from merxen.config import CellposeConfig, MaskFilterConfig, TilingConfig
from merxen.gpu_lock import gpu_lock_path_from_env, gpu_process_lock
from merxen.segmentation.hitl.import_annotations import (
    ImportedAnnotations,
    ImportedCrop,
    artifact_sha256,
    import_annotation_packages,
)
from merxen.segmentation.hitl.registry import CellposeModelMetadata
from merxen.segmentation.model_identity import CPSAM_V2_SHA256, require_cellpose_runtime

logger = logging.getLogger(__name__)


class CellposeTrainingConfig(BaseModel):
    """Named training run over completed packages below a configured HITL root."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    hitl_root: Path | None = None
    annotation_package_ids: list[str] = Field(min_length=1)
    package_dirs: list[Path] | None = None
    model_name: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,63}$")
    platform: Literal["ATERA", "XENIUM"]
    species: Literal["human", "mouse"]
    n_epochs: int = Field(default=100, ge=1, le=10000)
    learning_rate: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    batch_size: int = Field(default=1, ge=1, le=16)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    gpu: bool = True
    annotation_style: str = "neuronal-soma-proximal-process-v1"
    license_note: str = (
        "Local fine-tune of cpsam_v2; see upstream checkpoint licensing."
    )
    cellpose: CellposeConfig | None = None
    mask_filter: MaskFilterConfig = Field(default_factory=MaskFilterConfig)
    protected_output_roots: list[Path] = Field(default_factory=list)

    @field_validator("annotation_package_ids")
    @classmethod
    def _safe_package_ids(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values) or any(
            not re.fullmatch(r"hitl-[A-Za-z0-9_-]{1,80}", value) for value in values
        ):
            raise ValueError(
                "Use distinct package IDs from the viewer completion message."
            )
        return values

    @field_validator("model_name")
    @classmethod
    def _unique_custom_name(cls, value: str) -> str:
        if value in {"cpsam", "cpsam_v2"} or re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(
                "Choose a unique custom model name, not a stock name or hash."
            )
        return value


@dataclass(frozen=True)
class TrainedCellposeModel:
    """Location and identity of a completed named candidate bundle."""

    model_name: str
    bundle_path: Path
    build_hash: str
    weights_sha256: str
    platform: str
    reused: bool


def training_paths(config: CellposeTrainingConfig) -> tuple[Path, list[Path]]:
    """Resolve the configured server root and verify returned package locations."""
    root = config.hitl_root or local_config.require_path("MERXEN_HITL_ROOT")
    if not root.is_absolute():
        raise ValueError("HITL root must be an absolute configured location.")
    root = root.resolve()
    annotation_root = (root / "annotations").resolve()
    paths = config.package_dirs or [
        annotation_root / value for value in config.annotation_package_ids
    ]
    if len(paths) != len(config.annotation_package_ids):
        raise ValueError("Staged packages must match annotation_package_ids.")
    for path, package_id in zip(paths, config.annotation_package_ids, strict=True):
        if path.name != package_id or not path.resolve().is_relative_to(
            annotation_root
        ):
            raise ValueError(
                "Training packages must come from the configured HITL annotations root."
            )
    for protected in config.protected_output_roots:
        bundle = (root / "models" / config.model_name).resolve()
        if bundle.is_relative_to(
            protected.resolve()
        ) or protected.resolve().is_relative_to(bundle):
            raise ValueError("Model publication targets a protected baseline root.")
    return root, paths


def training_build_identity(
    config: CellposeTrainingConfig,
    corpus: ImportedAnnotations,
    runtime: tuple[str, str],
) -> str:
    """Hash the scientific training inputs and recipe independently of its name."""
    parameters = config.model_dump(
        mode="json",
        exclude={
            "hitl_root",
            "package_dirs",
            "model_name",
            "protected_output_roots",
            "annotation_package_ids",
        },
    )
    inference_defaults = (
        config.cellpose
        or corpus.baseline_cellpose
        or CellposeConfig(flow_threshold=0.7)
    )
    parameters["cellpose"] = inference_defaults.model_dump(
        mode="json", exclude={"pretrained_model", "model_sha256", "model_metadata_path"}
    )
    payload = {
        "version": 1,
        "base_weights_sha256": CPSAM_V2_SHA256,
        "runtime": runtime,
        "training_set": corpus.fingerprint,
        "platform": corpus.platform,
        "channels": corpus.channels,
        "pixel_size_um": corpus.pixel_size_um,
        "tile_size": corpus.tile_size,
        "overlap_px": corpus.overlap_px,
        "normalization_route": "A",
        "input_normalization": corpus.input_normalization,
        "guide_version": corpus.guide_version,
        "density_profile": corpus.transcript_density_profile,
        "parameters": parameters,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _existing_model(
    bundle: Path, name: str, build_hash: str, platform: str
) -> TrainedCellposeModel | None:
    if not bundle.exists():
        return None
    try:
        report = json.loads((bundle / "train.json").read_text())
        metadata = CellposeModelMetadata.model_validate_json(
            (bundle / "model.json").read_text()
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Model name already exists but is incomplete: {name}"
        ) from error
    if (
        report.get("build_hash") != build_hash
        or metadata.build_hash != build_hash
        or metadata.model_name != name
        or metadata.platform != platform
        or report.get("weights_sha256") != metadata.weights_sha256
        or metadata.status == "rejected"
        or artifact_sha256(bundle / "model") != metadata.weights_sha256
    ):
        raise ValueError(
            f"Model name is already used by a different training build: {name}"
        )
    return TrainedCellposeModel(
        name, bundle, build_hash, metadata.weights_sha256, platform, True
    )


def train_cellpose_model(
    config: CellposeTrainingConfig, output_dir: Path
) -> TrainedCellposeModel:
    """Train from cpsam_v2 and atomically publish a candidate under models/name.

    Args:
        config: Returned packages, unique model name and seeded training recipe.
        output_dir: New directory for the selector handed to downstream stages.

    Returns:
        The immutable named model, or the identical existing build on resume.

    Raises:
        ValueError: If packages, runtime, model name or protected paths disagree.
    """
    import fcntl

    root, packages = training_paths(config)
    corpus = import_annotation_packages(packages)
    if corpus.platform != config.platform:
        raise ValueError("Training platform differs from the annotation packages.")
    positive = [crop for crop in corpus.crops if crop.cell_count > 0]
    if not positive:
        raise ValueError("At least one reviewed crop must contain annotated cells.")
    runtime = require_cellpose_runtime()
    build_hash = training_build_identity(config, corpus, runtime)
    store = root / "models"
    store.mkdir(parents=True, exist_ok=True)
    bundle = store / config.model_name
    if not bundle.resolve().is_relative_to(store.resolve()):
        raise ValueError("Model name resolves outside the registry.")
    output_dir = output_dir.resolve()
    if any(
        output_dir.is_relative_to(path.resolve())
        for path in config.protected_output_roots
    ):
        raise ValueError("Training reports target a protected baseline root.")
    if (
        output_dir == bundle
        or output_dir.is_relative_to(bundle)
        or bundle.is_relative_to(output_dir)
    ):
        raise ValueError(
            "Training reports must be separate from the immutable model bundle."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = store / f".{config.model_name}.lock"
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                f"Another training run is publishing model name: {config.model_name}"
            ) from error
        result = _existing_model(bundle, config.model_name, build_hash, corpus.platform)
        if result is None:
            with tempfile.TemporaryDirectory(
                prefix=f".{config.model_name}.training-", dir=store
            ) as directory:
                staging = Path(directory)
                with gpu_process_lock(gpu_lock_path_from_env() if config.gpu else None):
                    result = _fit_and_publish(
                        config, corpus, positive, runtime, build_hash, staging, bundle
                    )
        _write_selector(result, output_dir)
        return result


def _fit_and_publish(
    config: CellposeTrainingConfig,
    corpus: ImportedAnnotations,
    positive: list[ImportedCrop],
    runtime: tuple[str, str],
    build_hash: str,
    staging: Path,
    bundle: Path,
) -> TrainedCellposeModel:
    import torch
    from cellpose import train

    from merxen.segmentation.cellpose import build_cellpose_model

    if config.gpu and not torch.cuda.is_available():
        raise ValueError("GPU training was requested but CUDA is unavailable.")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.gpu:
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base_config = CellposeConfig(gpu=config.gpu, use_bfloat16=False)
    model = build_cellpose_model(base_config)
    if artifact_sha256(Path(model.pretrained_model)) != CPSAM_V2_SHA256:
        raise ValueError(
            "Training base checkpoint differs from the pinned cpsam_v2 model."
        )
    images = [np.load(crop.model_input_path, allow_pickle=False) for crop in positive]
    labels = [crop.masks for crop in positive]
    checkpoint, losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        channel_axis=0,
        normalize=False,
        compute_flows=True,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        n_epochs=config.n_epochs,
        min_train_masks=1,
        bsize=256,
        save_path=str(staging),
        model_name="candidate",
        save_every=config.n_epochs,
    )
    if not np.isfinite(np.asarray(losses)).all():
        raise ValueError("Training produced non-finite losses; no model was published.")
    weights = staging / "model"
    shutil.move(str(checkpoint), weights)
    weights_sha = artifact_sha256(weights)
    inference_defaults = (
        config.cellpose
        or corpus.baseline_cellpose
        or CellposeConfig(flow_threshold=0.7)
    )
    inference = inference_defaults.model_copy(
        update={
            "pretrained_model": str(bundle / "model"),
            "model_sha256": weights_sha,
            "model_metadata_path": bundle / "model.json",
            "input_normalization": corpus.input_normalization,
            "use_bfloat16": False,
        }
    )
    tiling = TilingConfig(
        tile_size_candidates=[corpus.tile_size],
        min_tile_size=min(1024, corpus.tile_size),
        tile_overlap=corpus.overlap_px,
    )
    metadata = CellposeModelMetadata(
        weights_sha256=weights_sha,
        platform=config.platform,
        species=config.species,
        channels=list(corpus.channels),
        image_key="morphology_focus",
        pixel_size_um=corpus.pixel_size_um,
        tile_size=corpus.tile_size,
        normalization_route="A",
        annotation_style=config.annotation_style,
        annotation_guide_version=corpus.guide_version,
        license_note=config.license_note,
        status="candidate",
        cellpose_version=runtime[0],
        torch_version=runtime[1],
        evaluated_cellpose=inference,
        evaluated_mask_filter=config.mask_filter,
        transcript_density_profile=corpus.transcript_density_profile,
        training_tiling=tiling,
        model_name=config.model_name,
        build_hash=build_hash,
        evaluation_summary=(
            "Training pilot only; held-out evaluation has not been performed."
        ),
    )
    (staging / "model.json").write_text(metadata.model_dump_json(indent=2) + "\n")
    report = {
        "schema_version": 1,
        "build_hash": build_hash,
        "model_name": config.model_name,
        "training_set_sha256": corpus.fingerprint,
        "base_weights_sha256": CPSAM_V2_SHA256,
        "weights_sha256": weights_sha,
        "cellpose_version": runtime[0],
        "torch_version": runtime[1],
        "training_config": config.model_dump(mode="json"),
        "package_ids": list(corpus.package_ids),
        "crop_count": len(positive),
        "cell_count": sum(crop.cell_count for crop in positive),
        "empty_crops_omitted": [
            f"{crop.package_id}/{crop.crop_id}"
            for crop in corpus.crops
            if crop.cell_count == 0
        ],
        "training_normalize": False,
        "seed": config.seed,
        "losses": np.asarray(losses).tolist(),
        "test_losses": np.asarray(test_losses).tolist(),
        "evaluation_eligible": False,
    }
    (staging / "train.json").write_text(json.dumps(report, indent=2) + "\n")
    # Publication is serialized by the per-name lock. Existing bundles are
    # never replaced, even when another process races the final write.
    if bundle.exists():
        raise ValueError(f"Model name already exists: {config.model_name}")
    staging.rename(bundle)
    for path in bundle.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    logger.info(
        "Published named candidate %s (%d cells)", config.model_name, corpus.cell_count
    )
    return TrainedCellposeModel(
        config.model_name, bundle, build_hash, weights_sha, config.platform, False
    )


def _write_selector(result: TrainedCellposeModel, output: Path) -> None:
    payload = {
        "schema_version": 1,
        "model_name": result.model_name,
        "bundle_path": str(result.bundle_path),
        "build_hash": result.build_hash,
        "weights_sha256": result.weights_sha256,
        "platform": result.platform,
        "reused": result.reused,
    }
    temporary = output / ".selector.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, output / "selector.json")
