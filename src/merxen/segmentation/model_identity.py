"""Record the weights and runtime that actually produced a Cellpose mask."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from merxen.config import CellposeConfig

CELLPOSE_VERSION = "4.2.1.1"
TORCH_MINOR = "2.14"
CPSAM_V2_SHA256 = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"


class CellposeModelIdentity(BaseModel):
    """Provenance of a completed inference, or explicitly unknown legacy output."""

    schema_version: Literal[1] = 1
    status: Literal["verified", "unverified"]
    model_name: str | None = None
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    cellpose_version: str | None = None
    torch_version: str | None = None
    cellpose_config: CellposeConfig | None = None
    evidence: str | None = None

    @model_validator(mode="after")
    def _require_verified_fields(self) -> CellposeModelIdentity:
        if self.status == "verified" and any(
            value is None
            for value in (
                self.model_name,
                self.weights_sha256,
                self.cellpose_version,
                self.torch_version,
                self.cellpose_config,
            )
        ):
            raise ValueError("Verified Cellpose identity requires weights and runtime")
        return self


def _torch_version() -> str:
    import torch

    # Wheel metadata can omit +cu130; the loaded runtime retains its build tag.
    return str(torch.__version__)


def require_cellpose_runtime() -> tuple[str, str]:
    """Check the reproducible inference runtime before loading any weights.

    Returns:
        Installed Cellpose and loaded torch versions, including local suffixes.

    Raises:
        RuntimeError: If an environment has drifted from the dependency bounds.
    """
    cellpose_version, torch_version = version("cellpose"), _torch_version()
    torch_minor = ".".join(torch_version.split("+")[0].split(".")[:2])
    if cellpose_version != CELLPOSE_VERSION or torch_minor != TORCH_MINOR:
        raise RuntimeError(
            f"Cellpose inference requires cellpose=={CELLPOSE_VERSION} and "
            f"torch>={TORCH_MINOR},<2.15; found cellpose={cellpose_version}, "
            f"torch={torch_version}. Rebuild the segmentation environment."
        )
    return cellpose_version, torch_version


def identify_cellpose_model(
    model: Any, config: CellposeConfig
) -> CellposeModelIdentity:
    """Hash the checkpoint loaded by Cellpose and reject unexpected fallback.

    Args:
        model: Constructed Cellpose model with its resolved pretrained_model path.
        config: Effective inference configuration.

    Returns:
        Identity of the actual loaded checkpoint and installed runtime.

    Raises:
        ValueError: If Cellpose loaded a different file or altered stock weights.
    """
    cellpose_version, torch_version = require_cellpose_runtime()
    weights_path = Path(model.pretrained_model)
    requested_path = Path(config.pretrained_model)
    if requested_path.is_file():
        matches_request = requested_path.resolve() == weights_path.resolve()
    else:
        matches_request = weights_path.name == config.pretrained_model
    if not matches_request:
        raise ValueError("Cellpose loaded different weights from pretrained_model")
    with weights_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if config.pretrained_model == "cpsam_v2" and digest != CPSAM_V2_SHA256:
        raise ValueError(
            f"Stock cpsam_v2 weights checksum mismatch: {digest}. "
            "Refusing to produce masks with an unrecognised checkpoint."
        )
    return CellposeModelIdentity(
        status="verified",
        model_name=config.pretrained_model,
        weights_sha256=digest,
        cellpose_version=cellpose_version,
        torch_version=torch_version,
        cellpose_config=config,
    )


def model_identity_path(mask_path: Path) -> Path:
    """Return the model identity sidecar beside a cell or nuclei mask.

    Args:
        mask_path: Persistent or staged Cellpose mask path.

    Returns:
        Corresponding cellpose[_nuclei]_model_identity.json path.
    """
    prefix = mask_path.stem.removesuffix("_masks_tiled")
    return mask_path.with_name(f"{prefix}_model_identity.json")


def find_model_identity_path(mask_path: Path) -> Path | None:
    """Find provenance beside a mask or an intermediate Nextflow symlink target.

    Args:
        mask_path: Mask staged into a downstream task, or its persistent path.

    Returns:
        First existing sidecar along the symlink chain, or None for legacy masks.

    Raises:
        ValueError: If the mask symlinks form a cycle.
    """
    current = Path(os.path.abspath(mask_path))
    visited: set[Path] = set()
    while current not in visited:
        visited.add(current)
        candidate = model_identity_path(current)
        if candidate.exists():
            return candidate
        if not current.is_symlink():
            return None
        target = current.readlink()
        current = Path(
            os.path.abspath(target if target.is_absolute() else current.parent / target)
        )
    raise ValueError(f"Symlink cycle while finding Cellpose identity for {mask_path}")


def read_model_identity(mask_path: Path) -> CellposeModelIdentity:
    """Read recorded provenance without attributing legacy masks to this runtime.

    Args:
        mask_path: Mask whose adjacent sidecar should be read.

    Returns:
        Stored identity, or an unverified record with unknown versions and weights.
    """
    path = find_model_identity_path(mask_path)
    if path is not None:
        return CellposeModelIdentity.model_validate_json(path.read_text())
    return CellposeModelIdentity(
        status="unverified",
        evidence=(
            "No model identity was recorded for this mask. Consult the original "
            "Cellpose task's .command.log to establish its weights and runtime."
        ),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_model_identity(mask_path: Path, identity: CellposeModelIdentity) -> Path:
    """Atomically record the identity of a successfully completed inference.

    Args:
        mask_path: Mask produced by the inference.
        identity: Provenance of the actual loaded model.

    Returns:
        Path of the written identity sidecar.
    """
    path = model_identity_path(mask_path)
    _write_json_atomic(path, identity.model_dump(mode="json"))
    return path


def ensure_model_identity(mask_path: Path, stats_path: Path) -> CellposeModelIdentity:
    """Preserve existing provenance, marking legacy masks explicitly unverified.

    Args:
        mask_path: Reusable mask; never modified by this function.
        stats_path: Stitching statistics to carry the same model identity.

    Returns:
        Existing identity, or the newly recorded legacy status.

    Raises:
        ValueError: If the statistics and existing sidecar disagree.
    """
    identity = read_model_identity(mask_path)
    payload = identity.model_dump(mode="json")
    stats = (
        json.loads(stats_path.read_text())
        if stats_path.exists()
        else {"write_stitching_stats": False}
    )
    recorded = stats.get("model_identity")
    if recorded is not None and recorded != payload:
        raise ValueError(f"Cellpose model identity disagrees with {stats_path}")
    if not model_identity_path(mask_path).exists():
        write_model_identity(mask_path, identity)
    if recorded is None:
        stats["model_identity"] = payload
        _write_json_atomic(stats_path, stats)
    return identity
