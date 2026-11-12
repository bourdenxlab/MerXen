"""Persist generated transcript density and its provenance inside SpatialData."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import spatialdata as sd
import zarr

from merxen.config import AteraTranscriptDensityConfig
from merxen.io.points_aggregates import read_root_attrs
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.platforms import platform_spec

DENSITY_ATTR = "merxen_transcript_density"
DENSITY_IMAGE_KEY = "atera_transcript_density"
_METADATA_FILES = {"zarr.json", ".zattrs", ".zarray", ".zgroup", ".zmetadata"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _payload_identity(element: Path) -> str:
    # Metadata is rewritten by SpatialData enrichment/cache maintenance. The
    # immutable chunk payloads retain their timestamps through normal copies.
    return _digest(
        [
            (
                str(path.relative_to(element)),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in sorted(element.rglob("*"))
            if path.is_file() and path.name not in _METADATA_FILES
        ]
    )


def has_embedded_density(path: Path) -> bool:
    """Return whether the SpatialData metadata declares embedded density."""
    return DENSITY_ATTR in read_root_attrs(path)


def embedded_density_record(
    path: Path, *, verify_payload: bool = True
) -> dict[str, Any]:
    """Validate an embedded density binding and return its original raster identity.

    Args:
        path: SpatialData containing the native density image.
        verify_payload: Check immutable chunk file identities on the server.

    Returns:
        The original generation manifest, independent of storage location.
    """
    binding = read_root_attrs(path)[DENSITY_ATTR]
    signed = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if (
        binding.get("schema_version") != 1
        or binding.get("image_key") != DENSITY_IMAGE_KEY
        or binding.get("binding_sha256") != _digest(signed)
    ):
        raise ValueError("Embedded transcript density metadata is inconsistent")
    element = path / "images" / DENSITY_IMAGE_KEY
    if not element.is_dir() or (
        verify_payload and _payload_identity(element) != binding["payload_fingerprint"]
    ):
        raise ValueError("Embedded transcript density payload changed or is missing")
    record: dict[str, Any] = binding["record"]
    if record.get("fingerprint") != _digest(
        {key: value for key, value in record.items() if key != "fingerprint"}
    ):
        raise ValueError("Transcript density manifest identity mismatch")
    return record


def embed_transcript_density(
    target: Path, density_source: Path, *, build_pyramids: bool = True
) -> Path:
    """Add a complete native density image, provenance and display pyramids.

    Args:
        target: SpatialData to receive a permanent independent image element.
        density_source: Generated density store or another store with embedded density.
        build_pyramids: Materialize the viewer's coarse display levels too.

    Returns:
        The updated SpatialData path. Existing different density is never replaced.
    """
    from merxen.segmentation.transcript_density import (
        _native_image_identity,
        image_with_density,
        read_density_record,
    )
    from merxen.viewer_cache.build import build_image_pyramids_in_store

    target, density_source = target.resolve(), density_source.resolve()
    native_image_key = platform_spec("ATERA").native_image_key
    record = read_density_record(density_source)
    with spatialdata_write_lock(target):
        sdata = sd.read_zarr(target, selection=("images",))
        image_with_density(sdata, target, density_source)
        element = target / "images" / DENSITY_IMAGE_KEY
        installed = False
        if has_embedded_density(target):
            existing = embedded_density_record(target)
            if existing["fingerprint"] != record["fingerprint"]:
                raise ValueError(
                    "A different density image is already embedded; use a fresh run"
                )
            binding = dict(read_root_attrs(target)[DENSITY_ATTR])
            if "native_image_fingerprint" not in binding:
                # Upgrade legacy bindings only after image_with_density proved
                # equivalence to the original image, preserving raster identity.
                binding["native_image_fingerprint"] = _native_image_identity(
                    target / "images" / native_image_key
                )
                binding["binding_sha256"] = _digest(
                    {
                        key: value
                        for key, value in binding.items()
                        if key != "binding_sha256"
                    }
                )
                sdata.attrs[DENSITY_ATTR] = binding
                sdata.write_attrs()
                installed = True
        else:
            if element.exists():
                raise ValueError(
                    "An unverified density image already exists in this store"
                )
            source_element = density_source / "images" / DENSITY_IMAGE_KEY
            if target.is_relative_to(source_element):
                raise ValueError(
                    "Density source and destination must be separate elements"
                )
            temporary = element.with_name(f".density-{uuid.uuid4().hex}")
            try:
                shutil.copytree(source_element, temporary, copy_function=shutil.copy2)
                temporary.rename(element)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            # Refresh the on-disk group inventory before reopening: a source
            # with consolidated metadata would otherwise hide the new image.
            zarr.consolidate_metadata(str(target))
            sdata = sd.read_zarr(target, selection=("images",))
            binding = {
                "schema_version": 1,
                "image_key": DENSITY_IMAGE_KEY,
                "record": record,
                "payload_fingerprint": _payload_identity(element),
                "native_image_fingerprint": _native_image_identity(
                    target / "images" / native_image_key
                ),
            }
            binding["binding_sha256"] = _digest(binding)
            sdata.attrs[DENSITY_ATTR] = binding
            sdata.write_attrs()
            installed = True
        if build_pyramids:
            statuses = build_image_pyramids_in_store(
                sdata, target, image_keys=[DENSITY_IMAGE_KEY]
            )
            installed = installed or "built" in statuses.values()
        if installed:
            sdata.write_consolidated_metadata()
    return target


def prepare_embedded_density(config: AteraTranscriptDensityConfig) -> Path:
    """Generate or reuse density, retaining its only permanent payload in SpatialData.

    Args:
        config: Validated AteraTranscriptDensityConfig for this source/recipe.

    Returns:
        A task-output alias to the self-contained source SpatialData store.
    """
    from merxen.segmentation.transcript_density import (
        create_transcript_density,
        density_profile,
        read_density_record,
    )

    source, output = config.source_zarr_path.resolve(), config.output_dir.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Density output must be separate from the source store")
    if has_embedded_density(source):
        record = read_density_record(source)
        if record["profile"] != density_profile(config):
            raise ValueError(
                "The source contains a different density recipe; use a fresh source"
            )
        if output.exists() and any(output.iterdir()):
            raise ValueError("Density output must be new or empty")
        embed_transcript_density(source, source)
        output.mkdir(parents=True, exist_ok=True)
    else:
        generated = create_transcript_density(config)
        record = read_density_record(generated)
        embed_transcript_density(source, generated)
        # The installed element is an independent verified copy. Keep a task
        # alias for dependency tracking, without a second permanent payload.
        shutil.rmtree(generated)
    alias = output / "transcript_density.zarr"
    alias.symlink_to(source, target_is_directory=True)
    (output / "transcript_density.json").write_text(json.dumps(record, indent=2) + "\n")
    return alias
