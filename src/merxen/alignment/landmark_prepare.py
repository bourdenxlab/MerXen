"""Portable DAPI-only landmark packages from existing native SpatialData images."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import spatialdata as sd
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

import merxen
from merxen.alignment.frames import DapiFrame, resolve_dapi_frame
from merxen.config import AlignmentConfig
from merxen.io.spatialdata_io import spatialdata_write_lock
from merxen.manual_registration.schema import (
    CONVENTIONS,
    SCHEMA_VERSION,
    STORE_ATTR,
    STORE_SCHEMA,
    StoreAttrs,
    canonical_sha256,
    read_root_attrs,
    utc_now,
)
from merxen.viewer_cache.build import ViewerCacheParams, build_image_pyramids_in_store

PAIRED_LANDMARK_ATTR = "merxen_paired_landmarks"


def dapi_frame_signature(frame: DapiFrame) -> dict[str, Any]:
    """Describe a native DAPI frame, including small content samples.

    The signature is independent of the store's location and of mutable analysis
    metadata. It detects a different image or coordinate frame without hashing
    an entire slide on every alignment-only run.
    """
    digest = hashlib.sha256()
    height, width = frame.original_shape_rc
    for y, x in ((0, 0), (height // 2, width // 2), (height - 64, width - 64)):
        y, x = max(0, y), max(0, x)
        block = frame.image[y : y + 64, x : x + 64]
        digest.update(
            np.asarray(
                block.compute() if hasattr(block, "compute") else block
            ).tobytes()
        )
    return {
        "image_key": frame.image_key,
        "shape_yx": list(frame.original_shape_rc),
        "dataset_to_image": frame.dataset_to_image_matrix.tolist(),
        "channel": frame.channel_name,
        "sample_sha256": digest.hexdigest(),
    }


def prepare_paired_landmark_store(
    config: AlignmentConfig,
    output_store: Path,
    *,
    target_pixel_size_um: float = 0.5,
) -> Path:
    """Export both native DAPI images and pyramids for the landmark viewer.

    Args:
        config: Existing paired stores and DAPI channel/frame settings.
        output_store: New prepared Zarr; existing packages are never overwritten.
        target_pixel_size_um: Upper bound on isotropic package pixel size. Block
            averaging uses an integer factor and records its pixel-centre offset.

    Returns:
        The prepared store, containing reference Xenium and moving MERSCOPE DAPI.

    Raises:
        ValueError: For invalid resolution, roles, or changed input to an existing
            package. Existing landmarks are preserved on every invocation.
    """
    if not math.isfinite(target_pixel_size_um) or target_pixel_size_um <= 0:
        raise ValueError("target_pixel_size_um must be finite and positive")
    if config.fixed_platform != "XENIUM" or config.moving_platform != "MERSCOPE":
        raise ValueError(
            "Paired landmark packages require fixed XENIUM and moving MERSCOPE"
        )
    output_store = Path(output_store).absolute()
    images: dict[str, Any] = {}
    records: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for role, platform, path, image_config in (
        ("reference", "XENIUM", config.xenium_zarr_path, config.xenium_image),
        ("moving", "MERSCOPE", config.merscope_zarr_path, config.merscope_image),
    ):
        native = sd.read_zarr(path, selection=("images", "shapes"))
        frame = resolve_dapi_frame(native, platform=platform, config=image_config)
        # Manual-viewer pixels have a nominal scalar display size. Calibrated
        # MERSCOPE dataset transforms can be anisotropic; retain that complete
        # affine in the source record rather than replacing it with this size.
        px = float(np.sqrt(np.prod(frame.pixel_size_xy_um)))
        factor = max(1, int(np.floor(target_pixel_size_um / px + 1e-9)))
        data = da.asarray(frame.image)
        if factor > 1:
            data = da.coarsen(
                np.mean, data, {0: factor, 1: factor}, trim_excess=True
            ).astype(data.dtype)
        pixel = px * factor
        images[role] = Image2DModel.parse(
            data[None].rechunk((1, 1024, 1024)),
            dims=("c", "y", "x"),
            c_coords=["DAPI"],
            transformations={"global": Scale([pixel, pixel], axes=("x", "y"))},
        )
        records[role] = {
            "element": role,
            "role": role,
            "shape_yx": list(data.shape),
            "dtype": str(data.dtype),
            "pixel_size_um": pixel,
            "channel_names": ["DAPI"],
            "canonical_channel_names": ["DAPI"],
            "registration_channel": "DAPI",
        }
        package_to_native = np.array(
            [[factor, 0, (factor - 1) / 2], [0, factor, (factor - 1) / 2], [0, 0, 1]],
            dtype=float,
        )
        sources[role] = {
            "platform": platform,
            "native": dapi_frame_signature(frame),
            "package_to_native_image": package_to_native.tolist(),
            "native_store": str(Path(path).absolute()),
        }
    # Locations are informative; moving or copying the source does not change the
    # image identity. Native segmentation and tissue annotations are not inputs.
    identity = {
        role: {k: v for k, v in source.items() if k != "native_store"}
        for role, source in sources.items()
    }
    content_digest = canonical_sha256({"pair_id": config.pair_id, "sources": identity})
    if output_store.exists():
        existing = read_root_attrs(output_store).get(STORE_ATTR, {})
        if (
            existing.get("dataset_name") == config.pair_id
            and existing.get("content_digest") == content_digest
        ):
            return output_store
        raise ValueError(
            "Landmark package exists with different inputs; use a new output path"
        )
    payload = {
        "schema": STORE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": "prepared",
        "dataset_name": config.pair_id,
        "created_at": utc_now(),
        "merxen_version": merxen.__version__,
        "conventions": dict(CONVENTIONS),
        "images": records,
        "channel_matching": {
            "reference_kept": ["DAPI"],
            "reference_dropped": [],
            "moving_unmatched": [],
        },
        "grid_digest": canonical_sha256(records),
        "content_digest": content_digest,
    }
    StoreAttrs.model_validate(payload)
    output_store.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_store.with_name(
        f".{output_store.name}.partial-{uuid.uuid4().hex}"
    )
    try:
        prepared = sd.SpatialData(
            images=images,
            attrs={
                STORE_ATTR: payload,
                PAIRED_LANDMARK_ATTR: {
                    "version": 1,
                    "pair_id": config.pair_id,
                    "sources": sources,
                },
            },
        )
        prepared.write(temporary)
        prepared = sd.read_zarr(temporary, selection=("images",))
        build_image_pyramids_in_store(
            prepared,
            temporary,
            image_keys=["reference", "moving"],
            params=ViewerCacheParams(),
        )
        with spatialdata_write_lock(output_store):
            if output_store.exists():
                raise FileExistsError(f"Another writer created {output_store}")
            os.rename(temporary, output_store)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_store
