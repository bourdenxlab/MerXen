"""Provenance record and cache-reuse guard for optional MERSCOPE bead removal.

Every raw-TIFF MERSCOPE build writes the root attribute
``merxen_image_preprocessing`` (metadata only, also when bead removal is off).
A later run that reuses a cached SpatialData zarr compares that record with the
requested configuration, because a cached projection built with a different
bead-removal setting is silently wrong for the request: the beads are in (or
out of) every downstream mask and intensity.

This module deliberately does not import the correction code, so the guard is
cheap and has no GPU dependency.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from merxen.config import MerscopeBeadRemovalConfig
from merxen.io.bead_removal.params import (
    BEAD_REMOVAL_VERSION,
    DEFAULT_PARAMS,
    params_sha256,
)
from merxen.io.spatialdata_schema import package_versions

logger = logging.getLogger(__name__)

IMAGE_PREPROCESSING_ATTR = "merxen_image_preprocessing"
IMAGE_PREPROCESSING_SCHEMA = 1
RAW_TIFF_SOURCE = "raw_tiff"
# Backend of this version of the method; the GPU backend records its own
# device and library versions.
CPU_DEVICE = "cpu"
# Config fields that do not change the output pixels (the switch itself, the
# fill which has its own record, and the pool size).
_NON_OUTPUT_CONFIG_FIELDS = frozenset({"enabled", "fill", "max_workers"})
_FILL_MODULE = "merxen.io.bead_removal.fill"


def _bead_off() -> dict[str, Any]:
    """Return the identity of a build without bead removal."""
    return {"enabled": False, "fill": {"enabled": False}}


def _fill_params_identity() -> dict[str, Any]:
    """Hash the fixed fill parameters and report their spatial support.

    The fill module is imported lazily so that the guard and the record work
    without it; its parameters are fixed in code, so the hash is a property of
    the installed version.

    Returns:
        ``params_sha256`` and ``support_px`` of the fill, or ``None`` for both
        when the fill module is not part of this installation.
    """
    try:
        fill = importlib.import_module(_FILL_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _FILL_MODULE:
            raise
        return {"params_sha256": None, "support_px": None}
    params = fill.FillParams(enabled=True)
    return {
        "params_sha256": str(fill.fill_params_sha256(params)),
        "support_px": int(fill.support_px(params)["total"]),
    }


def _config_sha256(cfg: MerscopeBeadRemovalConfig) -> str:
    """Hash the algorithm parameters together with the output-relevant config."""
    config = {
        key: sorted(value) if key == "stains" else value
        for key, value in cfg.model_dump().items()
        if key not in _NON_OUTPUT_CONFIG_FIELDS
    }
    payload = {"params": params_sha256(DEFAULT_PARAMS), "config": config}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def bead_removal_identity(
    cfg: MerscopeBeadRemovalConfig | None,
    *,
    calibration_sha256: str | None = None,
    device: str = CPU_DEVICE,
) -> dict[str, Any]:
    """Describe what a bead-removal request produces, for comparison.

    Args:
        cfg: Requested bead-removal configuration, or ``None`` when off.
        calibration_sha256: Content hash of the calibration, once it is known
            (a requested configuration does not know it yet).
        device: Backend device that runs the correction.

    Returns:
        A JSON-safe mapping. A disabled request is
        ``{"enabled": False, "fill": {"enabled": False}}``.
    """
    if cfg is None or not cfg.enabled:
        return _bead_off()
    fill: dict[str, Any] = {"enabled": bool(cfg.fill)}
    if cfg.fill:
        fill.update(_fill_params_identity())
    return {
        "enabled": True,
        "method": cfg.method,
        "version": BEAD_REMOVAL_VERSION,
        "params_sha256": _config_sha256(cfg),
        "calibration_sha256": calibration_sha256,
        "fill": fill,
        "backend": {"device": device},
    }


def identity_from_record(recorded: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract the bead-removal identity from a recorded preprocessing attr.

    Args:
        recorded: The ``merxen_image_preprocessing`` attribute, or ``None`` for
            a legacy store that predates it.

    Returns:
        The recorded identity; a legacy store reads as "bead removal off".
    """
    bead = _recorded_bead_removal(recorded)
    if not bead.get("enabled"):
        return _bead_off()
    identity = {
        key: bead.get(key)
        for key in (
            "method",
            "version",
            "params_sha256",
            "calibration_sha256",
            "fill",
            "backend",
        )
    }
    identity["fill"] = dict(bead.get("fill") or {"enabled": False})
    identity["backend"] = dict(bead.get("backend") or {"device": CPU_DEVICE})
    return {"enabled": True, **identity}


def build_image_preprocessing_record(
    *,
    z_layers: Sequence[int],
    stainings: Sequence[str],
    bead_removal_cfg: MerscopeBeadRemovalConfig | None,
    calibration: Mapping[str, Any] | None = None,
    run: Mapping[str, Any] | None = None,
    device: str = CPU_DEVICE,
) -> dict[str, Any]:
    """Build the ``merxen_image_preprocessing`` record of a raw-TIFF build.

    The record holds metadata only. It is written for every raw-TIFF build so
    that a later reuse can tell "built without bead removal" from "built before
    this record existed" only through its absence (legacy), never by guessing.

    Args:
        z_layers: Planes that were max-projected.
        stainings: Stain names of the projection (stored sorted: the storage
            order is not deterministic).
        bead_removal_cfg: Requested bead-removal configuration, or ``None``.
        calibration: Summary of the calibration (thresholds, per-stain rho,
            ``calibration_sha256``), when bead removal ran.
        run: Summary of the correction run (chunk and bead counts), when bead
            removal ran.
        device: Backend device that ran the correction.

    Returns:
        A JSON-safe mapping.
    """
    calibration_sha = (
        str(calibration["calibration_sha256"])
        if calibration is not None and "calibration_sha256" in calibration
        else None
    )
    bead_removal = bead_removal_identity(
        bead_removal_cfg, calibration_sha256=calibration_sha, device=device
    )
    if bead_removal_cfg is not None and bead_removal["enabled"]:
        bead_removal["corrected_stains"] = sorted(bead_removal_cfg.stains)
        bead_removal["calibration"] = _json_safe(calibration)
        bead_removal["run"] = _json_safe(run)
    record = {
        "schema": IMAGE_PREPROCESSING_SCHEMA,
        "source": RAW_TIFF_SOURCE,
        "z_layers": [int(z) for z in z_layers],
        "stains": sorted(str(stain) for stain in stainings),
        "bead_removal": bead_removal,
        "writer_versions": package_versions(),
    }
    safe: dict[str, Any] = _json_safe(record)
    return safe


def read_image_preprocessing_attr(source: Path) -> dict[str, Any] | None:
    """Read the preprocessing record from a SpatialData zarr root.

    Only the root group metadata is read; the SpatialData object is not loaded.

    Args:
        source: Path of the zarr store (symlinks are resolved).

    Returns:
        The recorded attribute, or ``None`` for a legacy store (no record) or a
        store whose root metadata cannot be read.
    """
    root = Path(source).resolve()
    for name, key in (("zarr.json", "attributes"), (".zattrs", None)):
        metadata_path = root / name
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Could not read zarr metadata %s: %s", metadata_path, exc)
            return None
        attrs = metadata.get(key) if key is not None else metadata
        if not isinstance(attrs, dict):
            return None
        record = attrs.get(IMAGE_PREPROCESSING_ATTR)
        return dict(record) if isinstance(record, dict) else None
    return None


def check_bead_removal_reuse(
    recorded: Mapping[str, Any] | None,
    requested: MerscopeBeadRemovalConfig | None,
    *,
    force: bool = False,
    source: Path | str | None = None,
    requested_z_layers: Sequence[int] | None = None,
    requested_device: str = CPU_DEVICE,
) -> None:
    """Refuse to reuse a cached SpatialData built with other bead-removal settings.

    Hard errors (the cached projection does not match the request):
      * bead removal on versus off, a missing record reading as off;
      * the fill on versus off, when bead removal is on in both.

    Warnings only (the result is valid but not what a fresh build would give):
      * a different method version, parameter hash or fill parameter hash;
      * a different backend device or library version (the backends agree to
        far below one count);
      * a different recorded z range.

    A cache is never deleted here.

    Args:
        recorded: ``merxen_image_preprocessing`` of the cached zarr, or ``None``
            for a legacy store.
        requested: Requested bead-removal configuration, or ``None`` when off.
        force: When True a rebuild is happening anyway, so nothing is checked.
        source: Cached store, for the message only.
        requested_z_layers: Requested z planes, when explicitly configured.
        requested_device: Backend device the request would run on.

    Raises:
        ValueError: If bead removal or the fill differs between cache and
            request.
    """
    requested_identity = bead_removal_identity(requested, device=requested_device)
    recorded_identity = identity_from_record(recorded)
    location = f" at {source}" if source is not None else ""

    if bool(recorded_identity["enabled"]) != bool(requested_identity["enabled"]):
        if force:
            logger.info(
                "Rebuilding the cached SpatialData%s: bead removal was %s, %s "
                "requested.",
                location,
                _on_off(recorded_identity["enabled"]),
                _on_off(requested_identity["enabled"]),
            )
            return
        raise ValueError(
            f"the cached SpatialData{location} was built with bead removal "
            f"{_on_off(recorded_identity['enabled'])} but "
            f"{_on_off(requested_identity['enabled'])} was requested; rebuild "
            "with --force_spatialdata_build true (needs merscope_dir) or change "
            "merscope_bead_removal"
        )

    if requested_z_layers is not None and recorded is not None:
        recorded_z = recorded.get("z_layers")
        if recorded_z is not None and sorted(recorded_z) != sorted(requested_z_layers):
            logger.warning(
                "The cached SpatialData%s was projected from z planes %s but %s "
                "were requested; it is reused as built.",
                location,
                sorted(recorded_z),
                sorted(requested_z_layers),
            )

    if not requested_identity["enabled"]:
        return

    recorded_fill = bool(recorded_identity["fill"].get("enabled"))
    requested_fill = bool(requested_identity["fill"].get("enabled"))
    if recorded_fill != requested_fill:
        if force:
            logger.info(
                "Rebuilding the cached SpatialData%s: the bead-removal fill was "
                "%s, %s requested.",
                location,
                _on_off(recorded_fill),
                _on_off(requested_fill),
            )
            return
        raise ValueError(
            f"the cached SpatialData{location} was built with the bead-removal "
            f"fill {_on_off(recorded_fill)} but {_on_off(requested_fill)} was "
            "requested; rebuild with --force_spatialdata_build true (needs "
            "merscope_dir) or change merscope_bead_removal_fill"
        )

    _warn_on_differences(recorded_identity, requested_identity, location)


SEGMENTATION_SIDECAR_SCHEMA = 1
CELL_MASK_SIDECAR_NAME = "image_preprocessing.json"
NUCLEI_MASK_SIDECAR_NAME = "nuclei_image_preprocessing.json"
# Identity fields of a correction that decide whether two images are the same
# for segmentation. Library versions are deliberately absent: the backends
# agree to far below one count, so only the device is compared.
_SEGMENTATION_IDENTITY_KEYS = (
    "method",
    "version",
    "params_sha256",
    "calibration_sha256",
)


def segmentation_identity(
    recorded: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Reduce a preprocessing record to what Cellpose masks depend on.

    Args:
        recorded: The ``merxen_image_preprocessing`` attribute of the source
            SpatialData, or ``None`` for a legacy store.

    Returns:
        ``None`` when no bead removal was applied (off or legacy, which are
        the same image); otherwise the method, version, parameter and
        calibration hashes, the fill identity and the backend device.
    """
    identity = identity_from_record(recorded)
    if not identity["enabled"]:
        return None
    reduced: dict[str, Any] = {
        key: identity.get(key) for key in _SEGMENTATION_IDENTITY_KEYS
    }
    reduced["fill"] = dict(identity["fill"])
    reduced["device"] = dict(identity["backend"]).get("device")
    safe: dict[str, Any] = _json_safe(reduced)
    return safe


def segmentation_sidecar_path(mask_path: Path | str, *, nuclei: bool = False) -> Path:
    """Return the preprocessing sidecar that sits next to a Cellpose mask.

    The cell and nuclei masks are produced by independent, independently
    reusable steps, so each gets its own file even when both live in one
    directory.

    Args:
        mask_path: The Cellpose label mask (``.npy``).
        nuclei: True for the DAPI-only nuclei mask.

    Returns:
        Path of the sidecar file.
    """
    name = NUCLEI_MASK_SIDECAR_NAME if nuclei else CELL_MASK_SIDECAR_NAME
    return Path(mask_path).with_name(name)


def write_segmentation_sidecar(
    mask_path: Path | str,
    recorded: Mapping[str, Any] | None,
    *,
    nuclei: bool = False,
) -> Path:
    """Record which image a Cellpose mask was computed from.

    Args:
        mask_path: The Cellpose label mask the sidecar describes.
        recorded: ``merxen_image_preprocessing`` of the source SpatialData, or
            ``None`` for a legacy store.
        nuclei: True for the DAPI-only nuclei mask.

    Returns:
        Path of the written sidecar.
    """
    path = segmentation_sidecar_path(mask_path, nuclei=nuclei)
    payload = {
        "schema": SEGMENTATION_SIDECAR_SCHEMA,
        "bead_removal": segmentation_identity(recorded),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written through a temporary name so a reader never sees half a file.
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def remove_segmentation_sidecar(mask_path: Path | str, *, nuclei: bool = False) -> None:
    """Delete the sidecar of a mask that is about to be regenerated.

    A record of the earlier image must not outlive the mask it described.
    """
    segmentation_sidecar_path(mask_path, nuclei=nuclei).unlink(missing_ok=True)


def check_segmentation_sidecar(
    mask_path: Path | str,
    recorded: Mapping[str, Any] | None,
    *,
    nuclei: bool = False,
    dataset_name: str = "",
) -> None:
    """Refuse to reuse Cellpose masks computed from a differently corrected image.

    Rules:
      * sidecar present: its bead-removal identity must equal the source's;
      * sidecar missing and the source had bead removal on: the masks predate
        the correction, so they are refused;
      * sidecar missing and the source had it off or is legacy: reused as
        before (such masks cannot have come from a corrected image).

    Args:
        mask_path: The Cellpose label mask that would be reused.
        recorded: ``merxen_image_preprocessing`` of the source SpatialData, or
            ``None`` for a legacy store.
        nuclei: True for the DAPI-only nuclei mask.
        dataset_name: Dataset label for the message.

    Raises:
        ValueError: If the masks and the source disagree about bead removal,
            or the sidecar cannot be read.
    """
    sidecar = segmentation_sidecar_path(mask_path, nuclei=nuclei)
    source_identity = segmentation_identity(recorded)
    prefix = f"[{dataset_name}] " if dataset_name else ""
    remedy = (
        "Move the segmentation outputs aside or use a new outdir (or run the "
        "step with --force-rerun) so the masks are computed from the current "
        "image."
    )
    if not sidecar.is_file():
        if source_identity is None:
            return
        raise ValueError(
            f"{prefix}the source SpatialData was built with bead removal on but "
            f"{mask_path} has no {sidecar.name}, so it predates the correction "
            f"and was computed from an uncorrected image. {remedy}"
        )
    try:
        stored = json.loads(sidecar.read_text()).get("bead_removal")
    except (OSError, ValueError, AttributeError) as exc:
        raise ValueError(
            f"{prefix}could not read {sidecar} ({exc}), so it cannot be verified "
            f"that {mask_path} matches the source image. {remedy}"
        ) from exc
    if stored is not None and not isinstance(stored, dict):
        raise ValueError(
            f"{prefix}{sidecar} does not hold a bead-removal identity. {remedy}"
        )
    if stored == source_identity:
        return
    raise ValueError(
        f"{prefix}{mask_path} was computed from an image with bead removal "
        f"{_describe_identity(stored)} but the source SpatialData has "
        f"{_describe_identity(source_identity)}. Reusing the masks would "
        f"segment a different image than the one that is quantified. {remedy}"
    )


def _describe_identity(identity: Mapping[str, Any] | None) -> str:
    """Summarise a segmentation identity for an error message."""
    if identity is None:
        return "off"
    fill = "with fill" if (identity.get("fill") or {}).get("enabled") else "no fill"
    return (
        f"on (version {identity.get('version')}, {fill}, device "
        f"{identity.get('device')}, params {str(identity.get('params_sha256'))[:12]}, "
        f"calibration {str(identity.get('calibration_sha256'))[:12]})"
    )


def bead_removal_table_metadata(
    recorded: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Describe the bead correction of a source for derived tables.

    Args:
        recorded: ``merxen_image_preprocessing`` of the source SpatialData, or
            ``None`` for a legacy store.

    Returns:
        ``None`` when bead removal was off (tables stay as they were);
        otherwise the corrected stains, the method version and the fill flag.
    """
    bead = _recorded_bead_removal(recorded)
    if not bead.get("enabled"):
        return None
    fill = bead.get("fill")
    metadata: dict[str, Any] = {
        "bead_corrected_stains": sorted(
            str(s) for s in bead.get("corrected_stains", [])
        ),
        "bead_removal_version": int(bead.get("version") or BEAD_REMOVAL_VERSION),
        "bead_removal_fill": bool(isinstance(fill, Mapping) and fill.get("enabled")),
    }
    return metadata


def _warn_on_differences(
    recorded: Mapping[str, Any],
    requested: Mapping[str, Any],
    location: str,
) -> None:
    """Log each difference that does not invalidate the cache."""
    pairs: list[tuple[str, Any, Any]] = [
        ("method version", recorded.get("version"), requested.get("version")),
        (
            "parameter hash",
            recorded.get("params_sha256"),
            requested.get("params_sha256"),
        ),
    ]
    if requested["fill"].get("enabled"):
        pairs.append(
            (
                "fill parameter hash",
                recorded["fill"].get("params_sha256"),
                requested["fill"].get("params_sha256"),
            )
        )
    recorded_backend = recorded.get("backend") or {}
    requested_backend = requested.get("backend") or {}
    # Library versions only exist once a GPU backend records them.
    for key in sorted(set(recorded_backend) | set(requested_backend)):
        pairs.append(
            (f"backend {key}", recorded_backend.get(key), requested_backend.get(key))
        )
    for label, was, now in pairs:
        # A requested value of None means "not known before the build".
        if now is not None and was != now:
            logger.warning(
                "The cached SpatialData%s was built with bead-removal %s %s but "
                "this run would use %s; it is reused as built.",
                location,
                label,
                was,
                now,
            )


def _recorded_bead_removal(recorded: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the ``bead_removal`` mapping of a record (empty for legacy)."""
    if not isinstance(recorded, Mapping):
        return {}
    bead = recorded.get("bead_removal")
    return bead if isinstance(bead, Mapping) else {}


def _on_off(flag: object) -> str:
    return "on" if flag else "off"


def _json_safe(value: Any) -> Any:
    """Round-trip through JSON so numpy scalars and tuples become plain types."""
    return json.loads(json.dumps(value, default=_to_builtin))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
