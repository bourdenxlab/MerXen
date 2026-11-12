"""Validate returned annotation packages without loading pickle-based GUI files."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from skimage.measure import label

from merxen.config import CellposeConfig


@dataclass(frozen=True)
class ImportedCrop:
    """One verified crop eligible for supervised training."""

    package_id: str
    crop_id: str
    model_input_path: Path
    masks: np.ndarray
    model_input_sha256: str
    labels_sha256: str
    cell_count: int


@dataclass(frozen=True)
class ImportedAnnotations:
    """Verified training inputs and their shared inference/annotation contract."""

    crops: tuple[ImportedCrop, ...]
    fingerprint: str
    platform: str
    channels: tuple[str, ...]
    pixel_size_um: float
    input_normalization: str
    tile_size: int
    overlap_px: int
    guide_version: str
    transcript_density_profile: dict[str, Any] | None
    package_ids: tuple[str, ...]
    baseline_cellpose: CellposeConfig | None = None

    @property
    def cell_count(self) -> int:
        """Return the number of annotated cells across the training crops."""
        return sum(crop.cell_count for crop in self.crops)


@dataclass(frozen=True)
class AnnotationContract:
    """Shared native inference and annotation recipe of returned packages."""

    platform: str
    channels: tuple[str, ...]
    pixel_size_um: float
    input_normalization: str
    tile_size: int
    overlap_px: int
    guide_version: str
    density_profile: dict[str, Any] | None


@dataclass(frozen=True)
class LoadedAnnotationPackage:
    """One validated package before combining compatible training inputs."""

    crops: tuple[ImportedCrop, ...]
    contract: AnnotationContract
    package_id: str
    baseline_cellpose: CellposeConfig | None = None


def artifact_sha256(path: Path) -> str:
    """Hash an artifact in bounded chunks."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifact(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Annotation paths must be package-relative.")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Annotation artifacts cannot point outside their package.")
    return result


def _load_package(root: Path) -> LoadedAnnotationPackage:
    manifest_path = _artifact(root, "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    package_id = manifest.get("package_id", "")
    if (
        not re.fullmatch(r"hitl-[A-Za-z0-9_-]{1,80}", package_id)
        or root.name != package_id
    ):
        raise ValueError(
            "Place each package in annotations/<package-id>/ without renaming it."
        )
    if (
        manifest.get("managed_annotation_version") != 1
        or manifest.get("training_normalize") is not False
    ):
        raise ValueError("Import requires a managed route-A annotation package.")
    if manifest.get("evaluation_eligible") is not False:
        raise ValueError(
            "Held-out evaluation crops cannot enter this training importer."
        )
    platform = manifest["platform"]
    channels = tuple(manifest["model_channels"])
    if (
        platform not in {"ATERA", "XENIUM"}
        or not 1 <= len(channels) <= 3
        or len(set(channels)) != len(channels)
    ):
        raise ValueError("Invalid training platform or channels.")
    pixel_size = float(manifest["pixel_size_um"])
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("Training inputs need a valid native pixel size.")
    mode = manifest["input_normalization"]
    density = manifest.get("transcript_density")
    if mode not in {"joint", "per_channel"} or (
        "transcript_density" in channels and (mode != "per_channel" or density is None)
    ):
        raise ValueError("Training input normalization does not match its channels.")
    context = manifest["roi_context"]
    if (
        context["platform"] != platform
        or context["image_key"] != manifest["image_key"]
        or context["pixel_size_um"] != pixel_size
        or context["source_metadata_sha256"] != manifest["source_metadata_sha256"]
        or context["baseline_stats_sha256"] != manifest["baseline_stats_sha256"]
    ):
        raise ValueError("Annotation source/grid identity is inconsistent.")
    with _artifact(root, "review.tsv").open(newline="") as stream:
        review_rows = list(csv.DictReader(stream, delimiter="\t"))
    review = {row["crop_id"]: row for row in review_rows}
    if len(review) != len(review_rows):
        raise ValueError("Review sheet contains repeated crop IDs.")
    manifest_sha = artifact_sha256(manifest_path)
    crops = []
    seen = set()
    for row in manifest["crops"]:
        crop_id = row["crop_id"]
        if not re.fullmatch(r"crop_[0-9]{3,6}", crop_id) or crop_id in seen:
            raise ValueError("Training crop IDs must be safe and unique.")
        seen.add(crop_id)
        if row.get("split") != "pilot_train":
            raise ValueError("Only pilot_train crops may enter this training run.")
        status = review.get(crop_id, {}).get("status")
        if status == "excluded":
            if not review[crop_id].get("reason", "").strip():
                raise ValueError(f"Excluded crop needs a reason: {crop_id}")
            continue
        if status != "reviewed":
            raise ValueError(f"Finish annotation and save this crop first: {crop_id}")
        y0, y1, x0, x1 = row["bbox_y0_y1_x0_x1"]
        shape = (y1 - y0, x1 - x0)
        if (
            min(y0, x0) < 0
            or min(shape) < 256
            or y1 > context["height"]
            or x1 > context["width"]
        ):
            raise ValueError("Invalid native crop bounds.")
        for relative in (
            f"{crop_id}.tif",
            f"raw/{crop_id}.npy",
            f"model_input/{crop_id}.npy",
        ):
            expected = row["files_sha256"].get(relative)
            if not expected or artifact_sha256(_artifact(root, relative)) != expected:
                raise ValueError(f"Immutable crop input changed: {relative}")
        receipt = json.loads(_artifact(root, f"receipts/{crop_id}.json").read_text())
        if (
            receipt.get("schema_version") != 1
            or receipt.get("package_id") != package_id
            or receipt.get("manifest_sha256") != manifest_sha
            or receipt.get("crop_id") != crop_id
            or receipt.get("cellpose_version") != "4.2.1.1"
        ):
            raise ValueError(f"No compatible verified Cellpose save for {crop_id}")
        for relative, field in (
            (f"{crop_id}.tif", "image_sha256"),
            (f"{crop_id}_seg.npy", "seg_sha256"),
            (f"masks/{crop_id}.npz", "masks_sha256"),
        ):
            if artifact_sha256(_artifact(root, relative)) != receipt.get(field):
                raise ValueError(
                    f"Saved annotation differs from its receipt: {relative}"
                )
        # Never deserialize _seg.npy: it is pickle based. The managed GUI
        # supplies a numeric-only companion covered by the same save receipt.
        with np.load(
            _artifact(root, f"masks/{crop_id}.npz"), allow_pickle=False
        ) as data:
            if data.files != ["masks"]:
                raise ValueError("Numeric annotation must contain only masks.")
            masks = data["masks"]
        if (
            masks.shape != shape
            or masks.dtype.kind not in "iu"
            or np.any(masks < 0)
            or int(masks.max()) > masks.size
        ):
            raise ValueError(f"Invalid numeric mask array for {crop_id}")
        count = int(np.count_nonzero(np.unique(masks)))
        connected = label(masks, background=0, connectivity=2).astype(np.uint32)
        if int(connected.max()) != count or count != receipt.get("cell_count"):
            raise ValueError(f"Each cell must have one connected mask: {crop_id}")
        model_path = _artifact(root, f"model_input/{crop_id}.npy")
        image = np.load(model_path, allow_pickle=False, mmap_mode="r")
        if (
            image.shape != (3, *shape)
            or image.dtype != np.float32
            or not np.isfinite(image).all()
        ):
            raise ValueError(f"Invalid pre-normalized training input for {crop_id}")
        if len(channels) == 2 and np.any(image[2] != 0):
            raise ValueError("Two-channel inputs must retain an empty third slot.")
        del image
        crops.append(
            ImportedCrop(
                package_id,
                crop_id,
                model_path,
                connected,
                row["files_sha256"][f"model_input/{crop_id}.npy"],
                hashlib.sha256(connected.tobytes()).hexdigest(),
                count,
            )
        )
    if set(review) != seen:
        raise ValueError("Review sheet does not match the package crop list.")
    contract = AnnotationContract(
        platform,
        channels,
        pixel_size,
        mode,
        int(context["tile_size"]),
        int(context["overlap_px"]),
        manifest["guide_version"],
        density["profile"] if density else None,
    )
    baseline = (manifest.get("baseline_stats_model_identity") or {}).get(
        "cellpose_config"
    )
    return LoadedAnnotationPackage(
        tuple(crops),
        contract,
        package_id,
        CellposeConfig.model_validate(baseline) if baseline else None,
    )


def import_annotation_packages(paths: list[Path]) -> ImportedAnnotations:
    """Validate completed packages and assemble one compatible training corpus.

    Args:
        paths: Returned package directories under the configured HITL root.

    Returns:
        Verified numeric labels, input paths and a content-based dataset identity.

    Raises:
        ValueError: If a crop is unfinished, tampered, disconnected or incompatible.
    """
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Select one or more distinct annotation packages.")
    all_crops: list[ImportedCrop] = []
    package_ids = []
    expected = None
    baseline = None
    for path in paths:
        loaded = _load_package(path.resolve())
        if expected is not None and loaded.contract != expected:
            raise ValueError(
                "Training packages have different input/annotation contracts."
            )
        expected = loaded.contract
        if baseline is None:
            baseline = loaded.baseline_cellpose
        all_crops.extend(loaded.crops)
        package_ids.append(loaded.package_id)
    if not all_crops:
        raise ValueError("No reviewed training crops remain.")
    fingerprints = sorted(
        (crop.model_input_sha256, crop.labels_sha256) for crop in all_crops
    )
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("The same training crop appears more than once.")
    fingerprint = hashlib.sha256(json.dumps(fingerprints).encode()).hexdigest()
    assert expected is not None
    return ImportedAnnotations(
        tuple(all_crops),
        fingerprint,
        expected.platform,
        expected.channels,
        expected.pixel_size_um,
        expected.input_normalization,
        expected.tile_size,
        expected.overlap_px,
        expected.guide_version,
        expected.density_profile,
        tuple(package_ids),
        baseline,
    )
