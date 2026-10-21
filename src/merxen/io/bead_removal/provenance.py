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
