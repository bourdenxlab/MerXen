"""Versioned alignment-materialization manifest and revision helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR

ALIGNMENT_MANIFEST_ATTR = "merxen_alignment"
ALIGNMENT_MANIFEST_VERSION = 2
ALIGNMENT_PAIR_REFERENCE_ATTR = "merxen_alignment_pair_reference"
NONRIGID_ELEMENT_SUFFIX = "_aligned_nonrigid"
ALIGNMENT_BUNDLE_DIRECTORY = "_merxen_alignment"
_VIEWER_DERIVED_CACHE_PREFIX = "_napari_compare_"


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for materialization provenance."""
    return datetime.now(UTC).isoformat()


def payload_fingerprint(payload: Any) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible payload."""
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def native_store_revision(zarr_path: Path, sdata_obj: Any) -> dict[str, Any]:
    """Fingerprint native store structure without including aligned derivatives.

    The file signature intentionally uses relative paths, byte sizes, and mtimes
    for data payloads. Small metadata files use content hashes because
    SpatialData can rewrite identical metadata during normal reconciliation.
    This keeps the check bounded for partitioned transcript stores while still
    noticing added/replaced Parquet partitions, polygons, tables, or source images.

    Args:
        zarr_path: SpatialData Zarr root.
        sdata_obj: Open SpatialData object used to capture native schema metadata.

    Returns:
        Revision metadata containing the fingerprint and signed-file count.
    """
    root = Path(zarr_path)
    schema = _native_schema(getattr(sdata_obj, "attrs", {}).get(MERXEN_SCHEMA_ATTR, {}))
    records: list[tuple[str, int, int | str]] = []
    for group_name in ("points", "shapes", "tables", "labels", "images"):
        group_path = root / group_name
        if not group_path.exists():
            continue
        for element_path in sorted(group_path.iterdir(), key=lambda path: path.name):
            if not _is_native_element_key(element_path.name):
                continue
            for file_path in sorted(element_path.rglob("*")):
                if not file_path.is_file():
                    continue
                stat = file_path.stat()
                revision_token: int | str = (
                    _file_sha256(file_path)
                    if _is_metadata_file(file_path)
                    else int(stat.st_mtime_ns)
                )
                records.append(
                    (
                        str(file_path.relative_to(root)),
                        int(stat.st_size),
                        revision_token,
                    )
                )
    signature = {"schema": schema, "files": records}
    return {
        "algorithm": "sha256_native_paths_sizes_metadata_content_mtime_v1",
        "fingerprint": payload_fingerprint(signature),
        "file_count": len(records),
        "captured_at": utc_timestamp(),
    }


def new_alignment_manifest(
    *,
    pair_id: str,
    backend: str,
    fixed_platform: str,
    moving_platform: str,
    coordinate_system: str,
    transform_payload: dict[str, Any],
    transform_fingerprint: str,
    native_revision: dict[str, Any],
) -> dict[str, Any]:
    """Create an incomplete version-2 alignment materialization manifest."""
    return {
        "version": ALIGNMENT_MANIFEST_VERSION,
        "pair_id": str(pair_id),
        "roles": {
            "fixed": str(fixed_platform).upper(),
            "moving": str(moving_platform).upper(),
        },
        "common_coordinate_system": str(coordinate_system),
        "fixed_grid": {},
        "transform": {
            **_jsonable(transform_payload),
            "backend": str(backend),
            "fingerprint": str(transform_fingerprint),
        },
        "native_input": _jsonable(native_revision),
        "mappings": {
            "points": {},
            "shapes": {},
            "tables": {},
            "labels": {},
            "images": {},
        },
        "artifacts": {},
        "qc": {},
        "complete": False,
        "created_at": utc_timestamp(),
        "updated_at": utc_timestamp(),
    }


def initialize_alignment_manifest(
    *,
    config: Any,
    result: Any,
    moving_sdata: Any,
    fixed_sdata: Any,
    moving_zarr_path: Path,
) -> dict[str, Any]:
    """Embed a portable transform and initialize matching pair metadata.

    The caller must hold writer locks for both stores until their attrs have been
    persisted.
    """
    native_revision = native_store_revision(moving_zarr_path, moving_sdata)
    transform_payload = _portable_transform_payload(
        config=config,
        result=result,
        moving_zarr_path=Path(moving_zarr_path),
    )
    transform_fingerprint = payload_fingerprint(transform_payload)
    manifest = new_alignment_manifest(
        pair_id=config.pair_id,
        backend=config.backend,
        fixed_platform=config.fixed_platform,
        moving_platform=config.moving_platform,
        coordinate_system=(
            config.valis.coordinate_system_name
            if config.backend == "valis"
            else "merxen_xenium"
        ),
        transform_payload=transform_payload,
        transform_fingerprint=transform_fingerprint,
        native_revision=native_revision,
    )
    moving_sdata.attrs[ALIGNMENT_MANIFEST_ATTR] = manifest
    fixed_sdata.attrs[ALIGNMENT_PAIR_REFERENCE_ATTR] = alignment_pair_reference(
        pair_id=config.pair_id,
        counterpart_fingerprint=str(native_revision["fingerprint"]),
        transform_fingerprint=transform_fingerprint,
    )
    return manifest


def set_artifact_status(
    manifest: dict[str, Any],
    *,
    kind: str,
    native_key: str | None,
    output_key: str,
    status: str,
    qc: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    """Record one artifact mapping, status, and optional QC details."""
    mappings = manifest.setdefault("mappings", {}).setdefault(str(kind), {})
    if native_key is not None:
        mappings[str(native_key)] = str(output_key)
    artifact_key = f"{kind}/{output_key}"
    manifest.setdefault("artifacts", {})[artifact_key] = {
        "status": str(status),
        "native_key": None if native_key is None else str(native_key),
        "output_key": str(output_key),
        "qc": _jsonable(qc or {}),
        **_jsonable(metadata),
    }
    manifest["updated_at"] = utc_timestamp()


def invalidate_alignment_materialization(
    sdata_obj: Any,
    *,
    reason: str,
) -> bool:
    """Mark an existing alignment manifest stale after native store mutation.

    Args:
        sdata_obj: SpatialData-like object whose root attrs will be modified.
        reason: Human-readable mutation that requires reconciliation.

    Returns:
        ``True`` when a manifest was present and invalidated.
    """
    attrs = getattr(sdata_obj, "attrs", {})
    existing = attrs.get(ALIGNMENT_MANIFEST_ATTR)
    if not isinstance(existing, dict):
        return False
    manifest = dict(existing)
    manifest["complete"] = False
    manifest["invalidated_at"] = utc_timestamp()
    manifest["invalidation_reason"] = str(reason)
    for artifact in manifest.get("artifacts", {}).values():
        if isinstance(artifact, dict) and artifact.get("status") == "complete":
            artifact["status"] = "stale"
    manifest["updated_at"] = utc_timestamp()
    attrs[ALIGNMENT_MANIFEST_ATTR] = manifest
    return True


def alignment_pair_reference(
    *,
    pair_id: str,
    counterpart_fingerprint: str,
    transform_fingerprint: str,
) -> dict[str, Any]:
    """Return the lightweight reference stamped on the paired Xenium store."""
    return {
        "version": 1,
        "pair_id": str(pair_id),
        "counterpart_platform": "MERSCOPE",
        "counterpart_native_fingerprint": str(counterpart_fingerprint),
        "transform_fingerprint": str(transform_fingerprint),
        "updated_at": utc_timestamp(),
    }


def _native_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    schema = dict(value)
    segmentations = {
        str(branch): dict(entry)
        for branch, entry in dict(schema.get("segmentations", {})).items()
        if isinstance(entry, dict) and entry.get("coordinate_variant_of") is None
    }
    schema["segmentations"] = segmentations
    return schema


def _portable_transform_payload(
    *,
    config: Any,
    result: Any,
    moving_zarr_path: Path,
) -> dict[str, Any]:
    if result.valis_transform is not None:
        pair_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config.pair_id)).strip("_")
        bundle_dir = (
            Path(moving_zarr_path) / ALIGNMENT_BUNDLE_DIRECTORY / (pair_token or "pair")
        )
        outputs = result.valis_transform.save(bundle_dir)
        chain_path = outputs["transform_chain"]
        files = {
            name: {
                "path": str(path.relative_to(moving_zarr_path)),
                "sha256": _file_sha256(path),
            }
            for name, path in outputs.items()
        }
        return {
            "type": "valis_sampled_displacement_field",
            "selected_mode": result.valis_transform.selected_mode,
            "bundle_path": str(chain_path.relative_to(moving_zarr_path)),
            "transform_chain": json.loads(chain_path.read_text()),
            "files": files,
        }

    transform = result.nonrigid_transform
    moving_payload = (
        result.merscope_to_common
        if str(result.metadata.get("moving_platform", "MERSCOPE")) == "MERSCOPE"
        else result.xenium_to_common
    )
    if transform is None:
        return {
            "type": "affine",
            "selected_mode": str(moving_payload.get("selected_mode", "rigid")),
            "payload": {
                "affine_matrix": moving_payload["rigid_affine_matrix"],
                "anchors": [],
                "residuals": [],
            },
        }
    return {
        "type": "affine_plus_rbf_residual",
        "selected_mode": str(moving_payload.get("selected_mode", "nonrigid")),
        "payload": {
            "affine_matrix": _jsonable(transform.affine_matrix),
            "anchors": _jsonable(transform.anchors),
            "residuals": _jsonable(transform.residuals),
            "neighbors": int(transform.neighbors),
            "smoothing": float(transform.smoothing),
            "support_radius": transform.support_radius,
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_native_element_key(key: str) -> bool:
    return not (
        NONRIGID_ELEMENT_SUFFIX in str(key)
        or str(key).startswith(_VIEWER_DERIVED_CACHE_PREFIX)
    )


def _is_metadata_file(path: Path) -> bool:
    return (
        path.name.startswith(".")
        or path.name in {"zarr.json", "_metadata", "_common_metadata"}
        or path.suffix == ".json"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
