"""Build the viewer's on-the-fly caches into an enriched SpatialData store.

For each requested segmentation the builder rasterizes an instance-id label
mask, then materializes the label pyramid, the outline pyramid, and (once) the
morphology-image pyramid -- writing each with the exact key + completion marker
the viewer trusts. Everything is idempotent: an artifact whose marker already
matches the current format is skipped unless ``force`` is set.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from spatialdata import read_zarr
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Affine, get_transformation
from xarray import DataTree

from merxen.io.image_source import image_to_cyx, registered_image_keys
from merxen.io.spatialdata_io import (
    spatialdata_write_lock,
    write_or_replace_element,
    write_spatialdata_metadata,
)
from merxen.io.spatialdata_schema import MERXEN_SCHEMA_ATTR, ORIGINAL_ID_NAMESPACE
from merxen.masks import discover_masks, register_missing_mask_branches
from merxen.memory import force_release, log_status
from merxen.path_utils import remove_path
from merxen.platforms import PLATFORM_SPECS, find_platform_spec
from merxen.segmentation.native import NUCLEUS_CELL_KEY, NUCLEUS_ID_NAMESPACE
from merxen.viewer_cache.format import (
    DERIVED_CACHE_ATTR,
    LABEL_CACHE_ATTR,
    LABEL_CACHE_VERSION,
    OUTLINE_COVERAGE_MAX,
    OUTLINE_PYRAMID_MODE,
    PYRAMID_MIN_SIZE,
    PYRAMID_TILE,
    VIEWER_DERIVED_CACHE_VERSION,
    derived_image_pyramid_cache_key,
    derived_label_pyramid_cache_key,
    derived_outline_cache_key,
    image_pyramid_marker,
    is_derived_cache_key,
    label_cache_marker,
    label_key_for_shape_key,
    label_pyramid_marker,
    outline_marker,
)
from merxen.viewer_cache.pyramids import (
    build_multiscale_tree,
    lazy_coarsened_pyramid,
    lazy_outline_pyramid,
)
from merxen.viewer_cache.rasterize import (
    label_ids_for_shapes,
    napari_affine_from_px_to_um,
    pixel_window_global_bounds,
    query_geometries_for_bounds,
    rasterize_geometries_chunk,
)
from merxen.viewer_cache.transform import resolve_mask_affine

logger = logging.getLogger(__name__)

_BASE_SHAPE_KEYS: tuple[str, ...] = (
    "MOSAIK_proseg",
    "MOSAIK_proseg_hybrid",
    "MOSAIK_cellpose",
    "cellpose_nuclei",
)
# Retained as a source-compatible discovery hint for callers that imported the
# old constant. Runtime defaults now come exclusively from ``discover_masks``.
DEFAULT_SHAPE_KEYS: dict[str, tuple[str, ...]] = {
    name: _BASE_SHAPE_KEYS + spec.original_shape_keys
    for name, spec in PLATFORM_SPECS.items()
}


@dataclass(frozen=True)
class ViewerCacheParams:
    """Cache-build parameters, defaulting to the viewer's own defaults."""

    downsample: int = 4
    label_chunk_size: int = 2048
    contour_width: int = 1
    min_size: int = PYRAMID_MIN_SIZE
    shape_keys: tuple[str, ...] | None = None
    build_image_pyramid: bool = True
    force: bool = False
    base_image_key: str | None = None


def build_viewer_caches(
    zarr_path: str | Path,
    platform: str,
    original_data_path: str | Path,
    *,
    transform_path: str | Path | None = None,
    params: ViewerCacheParams | None = None,
) -> dict[str, Any]:
    """Pre-build viewer caches into the enriched store at ``zarr_path``.

    The store write lock is held from the read until the last cache is written,
    since label, pyramid, and attrs writes would otherwise interleave with
    another writer's read-modify-write of the same store.

    Returns a summary dict of what was built/skipped per element.
    """
    params = params or ViewerCacheParams()
    zarr_path = Path(zarr_path)
    platform = str(platform).upper()

    x_transform, y_transform = resolve_mask_affine(
        platform=platform,
        original_data_path=Path(original_data_path),
        transform_path=Path(transform_path) if transform_path is not None else None,
    )

    with spatialdata_write_lock(zarr_path):
        return _build_viewer_caches_in_store(
            zarr_path,
            platform=platform,
            x_transform=x_transform,
            y_transform=y_transform,
            params=params,
        )


def _build_viewer_caches_in_store(
    zarr_path: Path,
    *,
    platform: str,
    x_transform: tuple[float, float, float],
    y_transform: tuple[float, float, float],
    params: ViewerCacheParams,
) -> dict[str, Any]:
    """Build every cache into one store; the caller holds its write lock."""
    sdata = read_zarr(zarr_path)
    if register_missing_mask_branches(sdata):
        write_spatialdata_metadata(sdata, write_attrs=True)
    image_key, base_cyx, channels, image_transform, image_cs = _resolve_base_image(
        sdata, params.base_image_key
    )
    height = int(base_cyx.sizes["y"])
    width = int(base_cyx.sizes["x"])
    x_coords = base_cyx.coords["x"].values if "x" in base_cyx.coords else None
    y_coords = base_cyx.coords["y"].values if "y" in base_cyx.coords else None

    napari_affine = napari_affine_from_px_to_um(
        x_transform, y_transform, x_coords, y_coords
    )
    spatialdata_affine = _spatialdata_affine(napari_affine)
    inv_affine = np.linalg.inv(napari_affine)

    chunk = int(params.label_chunk_size)
    chunks = (min(chunk, height), min(chunk, width))

    registered_shape_keys = tuple(spec.shape_key for spec in discover_masks(sdata))
    if params.shape_keys is None:
        shape_keys = registered_shape_keys
    else:
        requested_shape_keys = set(map(str, params.shape_keys))
        shape_keys = tuple(
            key for key in registered_shape_keys if key in requested_shape_keys
        )
        ignored = requested_shape_keys - set(shape_keys)
        if ignored:
            logger.warning(
                "Ignoring viewer-cache shapes that are not registered masks: %s",
                sorted(ignored),
            )
    existing_labels = {str(k) for k in sdata.labels}
    summary: dict[str, Any] = {"masks": {}, "image_pyramid": None}
    vendor_labels = vendor_label_keys(sdata, platform)

    for shape_key in shape_keys:
        if shape_key not in sdata.shapes:
            log_status(f"[{platform}] Shape '{shape_key}' absent; skipping mask build.")
            continue
        label_key = label_key_for_shape_key(shape_key, existing_labels)
        if shape_key in vendor_labels:
            summary["masks"][shape_key] = _build_vendor_mask_and_derived(
                sdata=sdata,
                zarr_path=zarr_path,
                platform=platform,
                shape_key=shape_key,
                vendor_label_key=vendor_labels[shape_key],
                label_key=label_key,
                params=params,
            )
            force_release(note=f"after viewer-cache build for {shape_key}")
            continue
        _refuse_rasterised_mask_on_vendor_platform(platform, shape_key)
        summary["masks"][shape_key] = _build_mask_and_derived(
            sdata=sdata,
            zarr_path=zarr_path,
            platform=platform,
            shape_key=shape_key,
            label_key=label_key,
            shape=(height, width),
            chunks=chunks,
            napari_affine=napari_affine,
            spatialdata_affine=spatialdata_affine,
            inv_affine=inv_affine,
            params=params,
            coordinate_system="global",
        )
        force_release(note=f"after viewer-cache build for {shape_key}")

    if params.build_image_pyramid:
        summary["image_pyramid"] = build_image_pyramid(
            sdata=sdata,
            zarr_path=zarr_path,
            image_key=image_key,
            base_cyx=base_cyx,
            channels=channels,
            transform=image_transform,
            params=params,
            coordinate_system=image_cs,
        )

    return summary


def _resolve_base_image(
    sdata: Any, base_image_key: str | None = None
) -> tuple[str, Any, list[str], Any, str]:
    """Return (key, (c,y,x) DataArray, channels, global transform) of the real image.

    An explicit ``base_image_key`` (the platform's native image, set by the
    workflow for ATERA) is used as is. Otherwise derived image caches and
    registered post-run images are skipped so the label grid is never sized off
    a downsampled cache -- the same rule the viewer's ``_image_grid_for_labels``
    uses.

    Raises:
        RuntimeError: If no usable image exists.
        KeyError: If ``base_image_key`` is not a base image element.
    """
    image_key: str | None
    if base_image_key is not None:
        image_key = str(base_image_key)
        if image_key not in sdata.images or is_derived_cache_key(image_key):
            raise KeyError(
                f"base_image_key={base_image_key!r} is not a base image element "
                f"(images: {sorted(map(str, sdata.images.keys()))})"
            )
    else:
        # Registered post-run images share the grid but may sort before the
        # instrument image, so they must never become the base image.
        registered = registered_image_keys(sdata)
        image_key = next(
            (
                str(k)
                for k in sdata.images
                if not is_derived_cache_key(str(k)) and str(k) not in registered
            ),
            None,
        )
    if image_key is None:
        raise RuntimeError(
            f"No non-derived image found to size the label grid "
            f"(images: {sorted(map(str, sdata.images.keys()))})."
        )
    base_cyx = image_to_cyx(sdata.images[image_key])
    channels = _channel_labels(base_cyx)
    transformations = get_transformation(sdata.images[image_key], get_all=True)
    coordinate_system = (
        "global" if "global" in transformations else sorted(transformations)[0]
    )
    return (
        image_key,
        base_cyx,
        channels,
        transformations[coordinate_system],
        coordinate_system,
    )


def _channel_labels(image_cyx: Any) -> list[str]:
    if "c" in image_cyx.coords:
        return [str(c) for c in image_cyx.coords["c"].values]
    return [f"c{i}" for i in range(int(image_cyx.sizes.get("c", 1)))]


def _spatialdata_affine(napari_affine: np.ndarray) -> Affine:
    """Build the (x,y)->(x,y) SpatialData Affine from a napari (row/col)->(y,x)."""
    return Affine(
        [
            [
                float(napari_affine[1, 1]),
                float(napari_affine[1, 0]),
                float(napari_affine[1, 2]),
            ],
            [
                float(napari_affine[0, 1]),
                float(napari_affine[0, 0]),
                float(napari_affine[0, 2]),
            ],
            [0.0, 0.0, 1.0],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )


def build_mask_cache_on_grid(
    *,
    sdata: Any,
    zarr_path: Path,
    platform: str,
    shape_key: str,
    label_key: str,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    pixel_to_world_matrix: Any,
    coordinate_system: str,
    params: ViewerCacheParams,
) -> dict[str, Any]:
    """Rasterize a registered mask and its caches on an explicit image grid.

    Writes into the store behind ``sdata``; the caller must already hold that
    store's write lock (this helper does not take it, so a caller that holds
    it cannot deadlock on a second acquisition).
    """
    matrix = np.asarray(pixel_to_world_matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("pixel_to_world_matrix must be a 3x3 affine")
    napari_affine = np.asarray(
        [
            [matrix[1, 1], matrix[1, 0], matrix[1, 2]],
            [matrix[0, 1], matrix[0, 0], matrix[0, 2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return _build_mask_and_derived(
        sdata=sdata,
        zarr_path=Path(zarr_path),
        platform=str(platform),
        shape_key=str(shape_key),
        label_key=str(label_key),
        shape=shape,
        chunks=chunks,
        napari_affine=napari_affine,
        spatialdata_affine=Affine(
            matrix,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ),
        inv_affine=np.linalg.inv(napari_affine),
        params=params,
        coordinate_system=str(coordinate_system),
    )


def _read_marker(zarr_path: Path, group: str, key: str, attr_name: str) -> dict:
    path = zarr_path / group / key
    if not path.exists():
        return {}
    try:
        grp = zarr.open_group(str(path), mode="r")
        value = grp.attrs.get(attr_name, {})
        return dict(value) if isinstance(value, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _stamp_marker(
    zarr_path: Path, group: str, key: str, attr_name: str, marker: dict
) -> None:
    grp = zarr.open_group(str(zarr_path / group / key), mode="a")
    grp.attrs[attr_name] = marker


def _build_mask_and_derived(
    *,
    sdata: Any,
    zarr_path: Path,
    platform: str,
    shape_key: str,
    label_key: str,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    napari_affine: np.ndarray,
    spatialdata_affine: Affine,
    inv_affine: np.ndarray,
    params: ViewerCacheParams,
    coordinate_system: str,
) -> dict[str, Any]:
    """Rasterize one base mask then build its label pyramid + outline."""
    result: dict[str, Any] = {"label_key": label_key}
    gdf = sdata.shapes[shape_key]
    label_series, id_dtype = label_ids_for_shapes(gdf)

    marker = _read_marker(zarr_path, "labels", label_key, LABEL_CACHE_ATTR)
    mask_complete = (
        marker.get("complete")
        and marker.get("version") == LABEL_CACHE_VERSION
        and marker.get("source_shape_key") == str(shape_key)
    )
    if mask_complete and not params.force:
        log_status(
            f"[{platform}] Base mask labels[{label_key}] already current; skipping."
        )
        result["mask"] = "skipped"
    else:
        n_cells = _rasterize_base_mask(
            sdata=sdata,
            zarr_path=zarr_path,
            shape_key=shape_key,
            label_key=label_key,
            shape=shape,
            chunks=chunks,
            napari_affine=napari_affine,
            spatialdata_affine=spatialdata_affine,
            inv_affine=inv_affine,
            label_series=label_series,
            id_dtype=id_dtype,
            coordinate_system=coordinate_system,
        )
        _stamp_marker(
            zarr_path,
            "labels",
            label_key,
            LABEL_CACHE_ATTR,
            label_cache_marker(source_shape_key=shape_key, shape=shape, chunks=chunks),
        )
        log_status(
            f"[{platform}] Rasterized labels[{label_key}]: {n_cells:,} polygons."
        )
        result["mask"] = "built"

    # Read the rasterized base back from disk (lazily) for the pyramid + outline.
    base_da = da.from_zarr(str(zarr_path / "labels" / label_key / "s0"))
    result["label_pyramid"] = _build_label_pyramid(
        zarr_path=zarr_path,
        sdata=sdata,
        label_key=label_key,
        base_da=base_da,
        transform=spatialdata_affine,
        params=params,
        coordinate_system=coordinate_system,
    )
    result["outline"] = _build_outline(
        zarr_path=zarr_path,
        sdata=sdata,
        label_key=label_key,
        base_da=base_da,
        transform=spatialdata_affine,
        params=params,
        coordinate_system=coordinate_system,
    )
    return result


def vendor_label_keys(sdata: Any, platform: str) -> dict[str, str]:
    """Return ``shape key -> vendor label key`` for the vendor-backed masks.

    A platform that ships its segmentation as label images aligned with its
    original shapes (``PlatformSpec.original_label_keys``, ATERA) gets its
    viewer base masks from those labels: their values are the shapes'
    ``instance_id``. On such a platform a registry branch with a ``label``
    element (the stitched ProSeg layer of a ``native_proseg`` or
    ``cellpose_proseg`` row, the Cellpose layer of a ``cellpose_proseg`` row) is backed
    by that element in the same way. Only pairs whose shape and label are both
    in the store are returned; XENIUM and MERSCOPE declare none.

    Args:
        sdata: The store.
        platform: Platform name.

    Returns:
        The mapping, empty when the platform declares no vendor labels.
    """
    spec = find_platform_spec(platform)
    if spec is None or not spec.original_label_keys:
        return {}
    labels = {str(k) for k in sdata.labels}
    shapes = {str(k) for k in sdata.shapes}
    found = {
        shape_key: label_key
        for shape_key, label_key in zip(
            spec.original_shape_keys, spec.original_label_keys, strict=False
        )
        if shape_key in shapes and label_key in labels
    }
    # Registry branches that carry their own label element on the image grid
    # (the stitched ATERA ProSeg layers, e.g. MOSAIK_proseg -> proseg_cell_labels,
    # and the ATERA Cellpose layer MOSAIK_cellpose -> cellpose_cell_labels).
    registry = (sdata.attrs.get(MERXEN_SCHEMA_ATTR) or {}).get("segmentations") or {}
    for entry in registry.values():
        if not isinstance(entry, dict) or entry.get("coordinate_variant_of"):
            continue
        shape_key, label_key = entry.get("shape"), entry.get("label")
        if (
            shape_key
            and label_key
            and str(shape_key) in shapes
            and str(label_key) in labels
            and str(shape_key) not in found
        ):
            found[str(shape_key)] = str(label_key)
    return found


def _refuse_rasterised_mask_on_vendor_platform(platform: str, shape_key: str) -> None:
    """Refuse to rasterize a mask on a platform whose masks are vendor labels.

    The ``global`` coordinate system of such a store (ATERA) is in morphology
    pixels, so the pixel-to-micron transform of the rasterized path would
    misplace the mask. Only vendor-backed masks are supported there until a
    rasterized mask gets its own transform.

    Raises:
        ValueError: If the platform declares ``original_label_keys``.
    """
    spec = find_platform_spec(platform)
    if spec is None or not spec.original_label_keys:
        return
    raise ValueError(
        f"[{spec.name}] viewer mask for shapes[{shape_key}] has no vendor label "
        f"element ({', '.join(spec.original_label_keys)}); rasterized masks are "
        f"not supported for {spec.name} (its global coordinate system is in "
        "pixels). Pass shape_keys to build only the vendor-backed masks."
    )


#: Zarr metadata files, copied (never linked) when a label element is aliased:
#: every writer rewrites them in place.
_ZARR_METADATA_NAMES = frozenset(
    {"zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata"}
)

#: ``source`` recorded in the base-mask marker of a vendor-backed mask.
VENDOR_LABEL_SOURCE = "vendor_labels"

#: Link function of the alias builder (a seam for tests of the copy fallback).
_hardlink = os.link


def _build_vendor_mask_and_derived(
    *,
    sdata: Any,
    zarr_path: Path,
    platform: str,
    shape_key: str,
    vendor_label_key: str,
    label_key: str,
    params: ViewerCacheParams,
) -> dict[str, Any]:
    """Use a vendor label element as the base mask, then build its caches.

    The vendor labels already hold the shapes' instance ids on the image grid,
    so nothing is rasterized. The base mask the viewer looks up
    (``label_key``, e.g. ``atera_cell_boundaries_labels``) is an alias of the
    vendor element: its chunk files are hardlinks of the vendor chunks (copies
    across filesystems) and its metadata files are copies. The label pyramid
    and outline are then built from it with the vendor element's own
    transform. The base-mask marker records the vendor element, its
    fingerprint (:func:`vendor_label_fingerprint`; a changed vendor element
    rebuilds the alias), the link mode and the label id namespace.
    """
    vendor_element = sdata.labels[vendor_label_key]
    transformations = get_transformation(vendor_element, get_all=True)
    coordinate_system = (
        "global" if "global" in transformations else sorted(transformations)[0]
    )
    transform = transformations[coordinate_system]
    result: dict[str, Any] = {
        "label_key": label_key,
        "source_label_key": vendor_label_key,
    }
    marker = _read_marker(zarr_path, "labels", label_key, LABEL_CACHE_ATTR)
    vendor_base = da.from_zarr(str(zarr_path / "labels" / vendor_label_key / "s0"))
    fingerprint = vendor_label_fingerprint(zarr_path / "labels" / vendor_label_key)
    mask_complete = (
        marker.get("complete")
        and marker.get("version") == LABEL_CACHE_VERSION
        and marker.get("source_shape_key") == str(shape_key)
        and marker.get("source_label_key") == str(vendor_label_key)
        and marker.get("source_fingerprint") == fingerprint
        and marker.get("shape") == [int(s) for s in vendor_base.shape]
    )
    if mask_complete and not params.force:
        log_status(
            f"[{platform}] Base mask labels[{label_key}] (vendor "
            f"labels[{vendor_label_key}]) already current; skipping."
        )
        result["mask"] = "skipped"
    else:
        link_mode = _alias_label_element(sdata, zarr_path, vendor_label_key, label_key)
        base_marker = label_cache_marker(
            source_shape_key=shape_key,
            shape=(int(vendor_base.shape[0]), int(vendor_base.shape[1])),
            chunks=(
                int(vendor_base.chunksize[0]),
                int(vendor_base.chunksize[1]),
            ),
        )
        base_marker.update(
            {
                "source": VENDOR_LABEL_SOURCE,
                "source_label_key": str(vendor_label_key),
                "source_fingerprint": fingerprint,
                "link_mode": link_mode,
                **_label_id_namespace(sdata, platform, shape_key),
            }
        )
        _stamp_marker(zarr_path, "labels", label_key, LABEL_CACHE_ATTR, base_marker)
        log_status(
            f"[{platform}] Base mask labels[{label_key}] = vendor "
            f"labels[{vendor_label_key}] ({link_mode})."
        )
        result["mask"] = f"vendor_{link_mode}"
    result["link_mode"] = _read_marker(
        zarr_path, "labels", label_key, LABEL_CACHE_ATTR
    ).get("link_mode")

    # A re-aliased base (new or changed vendor labels) rebuilds its pyramid and
    # outline: their markers name the base key, not its content.
    derived_params = (
        params if result["mask"] == "skipped" else replace(params, force=True)
    )
    base_da = da.from_zarr(str(zarr_path / "labels" / label_key / "s0"))
    result["label_pyramid"] = _build_label_pyramid(
        zarr_path=zarr_path,
        sdata=sdata,
        label_key=label_key,
        base_da=base_da,
        transform=transform,
        params=derived_params,
        coordinate_system=coordinate_system,
    )
    result["outline"] = _build_outline(
        zarr_path=zarr_path,
        sdata=sdata,
        label_key=label_key,
        base_da=base_da,
        transform=transform,
        params=derived_params,
        coordinate_system=coordinate_system,
    )
    return result


def vendor_label_fingerprint(element_dir: Path) -> str:
    """Return a fingerprint of a vendor label element on disk.

    Covers every file of the element: chunk files by relative path, size and
    modification time (stat only; a hardlinked alias shares them), metadata
    files by relative path and content (writers may rewrite them unchanged).
    A re-linked, rewritten or reshaped vendor element gives a new fingerprint.

    Args:
        element_dir: ``labels/<key>`` directory of the store.

    Returns:
        ``sha256:<hex>``.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(element_dir).rglob("*") if p.is_file()):
        relative = path.relative_to(element_dir).as_posix()
        if path.name in _ZARR_METADATA_NAMES:
            content = hashlib.sha256(path.read_bytes()).hexdigest()
            digest.update(f"M {relative} {content}\n".encode())
        else:
            stat = path.stat()
            digest.update(f"C {relative} {stat.st_size} {stat.st_mtime_ns}\n".encode())
    return f"sha256:{digest.hexdigest()}"


def _label_id_namespace(sdata: Any, platform: str, shape_key: str) -> dict[str, str]:
    """Return the id namespace (and cell join key) of a vendor-backed mask.

    The label values are the shapes' instance ids, so they carry the id
    namespace of the segmentation branch registered for the shapes. Cell
    labels hold cell ids. On a platform whose nucleus ids are not cell ids
    (ATERA: nucleus row + 1), the nucleus labels hold nucleus ids and join to
    cells only through the nucleus shapes' ``cell_instance_id`` column
    (``cell_join_key``), never by value. Without a registry entry the
    platform registry decides.
    """
    registry = (sdata.attrs.get(MERXEN_SCHEMA_ATTR) or {}).get("segmentations") or {}
    for entry in registry.values():
        if (
            isinstance(entry, dict)
            and entry.get("shape") == str(shape_key)
            and entry.get("coordinate_variant_of") is None
            and entry.get("id_namespace")
        ):
            found = {"id_namespace": str(entry["id_namespace"])}
            if entry.get("cell_key"):
                found["cell_join_key"] = str(entry["cell_key"])
            return found
    spec = find_platform_spec(platform)
    if (
        spec is None
        or spec.nucleus_ids_are_cell_ids
        or shape_key != spec.original_nucleus_shape_key
    ):
        return {"id_namespace": ORIGINAL_ID_NAMESPACE}
    return {
        "id_namespace": NUCLEUS_ID_NAMESPACE,
        "cell_join_key": NUCLEUS_CELL_KEY,
    }


def _alias_label_element(
    sdata: Any, zarr_path: Path, source_key: str, alias_key: str
) -> str:
    """Make ``labels/<alias_key>`` an alias of ``labels/<source_key>`` on disk.

    Chunk files are hardlinked (copied when the filesystem refuses a link);
    metadata files are copied so later attribute writes never reach the
    source through a shared inode. The alias is assembled under a hidden
    sibling and renamed into place, replacing any previous alias. The alias is
    added to the labels group's OME list and the root metadata is
    re-consolidated; ``sdata`` is not given the alias in memory.

    Returns:
        ``"hardlink"`` when every chunk was linked, else ``"copy"``.
    """
    if str(alias_key) == str(source_key):
        raise ValueError(
            f"labels[{source_key}] cannot be aliased onto itself; the alias "
            "would replace the vendor element"
        )
    labels_dir = zarr_path / "labels"
    source = labels_dir / source_key
    target = labels_dir / alias_key
    staging = labels_dir / f".{alias_key}.partial-{uuid.uuid4().hex}"
    linked_all = True
    try:
        for root, _dirs, files in os.walk(source):
            relative = Path(root).relative_to(source)
            (staging / relative).mkdir(parents=True, exist_ok=True)
            for name in files:
                src = Path(root) / name
                dst = staging / relative / name
                if name in _ZARR_METADATA_NAMES:
                    shutil.copyfile(src, dst)
                    continue
                try:
                    _hardlink(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
                    linked_all = False
        _rename_multiscales(staging, alias_key)
        if target.exists() or target.is_symlink():
            remove_path(target)
        os.replace(staging, target)
    finally:
        if staging.exists():
            remove_path(staging)
    _add_ome_label_name(labels_dir, alias_key)
    has_consolidated = getattr(sdata, "has_consolidated_metadata", None)
    write_consolidated = getattr(sdata, "write_consolidated_metadata", None)
    if (
        callable(has_consolidated)
        and callable(write_consolidated)
        and has_consolidated()
    ):
        write_consolidated()
    return "hardlink" if linked_all else "copy"


def _ome_attrs(attrs: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return the OME attrs dict and whether it is nested under ``ome``."""
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        return dict(ome), True
    return dict(attrs), False


def _rename_multiscales(element_dir: Path, name: str) -> None:
    """Set the multiscales ``name`` of a copied label element to its new key."""
    group = zarr.open_group(str(element_dir), mode="r+", use_consolidated=False)
    attrs = dict(group.attrs.asdict())
    ome, nested = _ome_attrs(attrs)
    multiscales = ome.get("multiscales")
    if not isinstance(multiscales, list):
        return
    ome["multiscales"] = [
        {**entry, "name": name} if isinstance(entry, dict) else entry
        for entry in multiscales
    ]
    if nested:
        group.attrs["ome"] = ome
    else:
        group.attrs["multiscales"] = ome["multiscales"]


def _add_ome_label_name(labels_dir: Path, name: str) -> None:
    """Append a label element to the labels group's OME ``labels`` list."""
    group = zarr.open_group(str(labels_dir), mode="r+", use_consolidated=False)
    attrs = dict(group.attrs.asdict())
    ome, nested = _ome_attrs(attrs)
    names = ome.get("labels")
    names = list(names) if isinstance(names, list) else []
    if name in names:
        return
    names.append(name)
    if nested:
        group.attrs["ome"] = {**ome, "labels": names}
    else:
        group.attrs["labels"] = names


def _rasterize_base_mask(
    *,
    sdata: Any,
    zarr_path: Path,
    shape_key: str,
    label_key: str,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    napari_affine: np.ndarray,
    spatialdata_affine: Affine,
    inv_affine: np.ndarray,
    label_series: Any,
    id_dtype: Any,
    coordinate_system: str,
) -> int:
    """Write an empty label element, then fill its s0 array chunk-by-chunk."""
    height, width = shape
    empty = da.zeros(shape, chunks=chunks, dtype=id_dtype)
    label_da = xr.DataArray(empty, dims=("y", "x"))
    elem = Labels2DModel.parse(
        label_da,
        transformations={str(coordinate_system): spatialdata_affine},
    )
    write_or_replace_element(
        sdata,
        label_key,
        "labels",
        elem,
        overwrite=True,
    )

    label_arr = zarr.open_array(str(zarr_path / "labels" / label_key / "s0"), mode="r+")
    gdf = sdata.shapes[shape_key]
    chunk_h, chunk_w = chunks
    written = 0
    for y0 in range(0, height, chunk_h):
        y1 = min(y0 + chunk_h, height)
        for x0 in range(0, width, chunk_w):
            x1 = min(x0 + chunk_w, width)
            bounds = pixel_window_global_bounds(napari_affine, y0, y1, x0, x1)
            candidates = query_geometries_for_bounds(gdf, bounds)
            if len(candidates) == 0:
                continue
            ids = label_series.loc[candidates.index].to_numpy()
            tile = rasterize_geometries_chunk(
                candidates.geometry,
                ids,
                shape=(y1 - y0, x1 - x0),
                inv_affine=inv_affine,
                y0=y0,
                x0=x0,
                dtype=id_dtype,
            )
            if np.any(tile):
                label_arr[y0:y1, x0:x1] = tile
                written += 1
    return int(len(gdf))


def _build_label_pyramid(
    *,
    zarr_path: Path,
    sdata: Any,
    label_key: str,
    base_da: Any,
    transform: Affine,
    params: ViewerCacheParams,
    coordinate_system: str,
) -> str:
    cache_key = derived_label_pyramid_cache_key(label_key, params.downsample)
    if not params.force and _derived_complete(
        zarr_path, "labels", cache_key, "label_pyramid", label_key, params
    ):
        return "skipped"
    levels = lazy_coarsened_pyramid(
        base_da, step=params.downsample, reducer=np.max, min_size=params.min_size
    )
    if len(levels) == 0:
        return "no-levels"
    tree = build_multiscale_tree(
        levels,
        dims=("y", "x"),
        transform=transform,
        dtype=base_da.dtype,
        coordinate_system=coordinate_system,
    )
    Labels2DModel.validate(tree)
    _write_derived(sdata, zarr_path, "labels", cache_key, tree)
    _stamp_marker(
        zarr_path,
        "labels",
        cache_key,
        DERIVED_CACHE_ATTR,
        label_pyramid_marker(
            source_label_key=label_key,
            downsample=params.downsample,
            min_size=params.min_size,
            levels=len(levels),
        ),
    )
    return "built"


def _build_outline(
    *,
    zarr_path: Path,
    sdata: Any,
    label_key: str,
    base_da: Any,
    transform: Affine,
    params: ViewerCacheParams,
    coordinate_system: str,
) -> str:
    width = max(1, int(params.contour_width))
    cache_key = derived_outline_cache_key(label_key, width)
    if not params.force and _derived_complete(
        zarr_path, "labels", cache_key, "label_outline", label_key, params, width=width
    ):
        return "skipped"
    outline_levels = lazy_outline_pyramid(
        base_da, width=width, min_size=params.min_size
    )
    if len(outline_levels) == 0:
        return "no-levels"
    tree = build_multiscale_tree(
        outline_levels,
        dims=("y", "x"),
        transform=transform,
        dtype=np.uint8,
        coordinate_system=coordinate_system,
    )
    Labels2DModel.validate(tree)
    _write_derived(sdata, zarr_path, "labels", cache_key, tree)
    # The viewer records the shapes of the SOURCE label element's scale levels
    # here, not the outline's. ``base_da`` is that source's finest level, which is
    # the only one the coverage-mean pyramid reads.
    source_shapes = [[int(base_da.shape[0]), int(base_da.shape[1])]]
    _stamp_marker(
        zarr_path,
        "labels",
        cache_key,
        DERIVED_CACHE_ATTR,
        outline_marker(
            source_label_key=label_key,
            width=width,
            levels=len(outline_levels),
            source_shapes=source_shapes,
            min_size=params.min_size,
        ),
    )
    return "built"


def build_image_pyramid(
    *,
    sdata: Any,
    zarr_path: Path,
    image_key: str,
    base_cyx: Any,
    channels: list[str],
    transform: Any,
    params: ViewerCacheParams,
    coordinate_system: str = "global",
) -> str:
    """Build or refresh the viewer-compatible pyramid for one image element.

    Writes into the store behind ``sdata``; the caller must already hold that
    store's write lock (this helper does not take it, so a caller that holds
    it cannot deadlock on a second acquisition).
    """
    cache_key = derived_image_pyramid_cache_key(image_key, params.downsample)
    if not params.force and _derived_complete(
        zarr_path, "images", cache_key, "image_pyramid", image_key, params
    ):
        return "skipped"
    levels = lazy_coarsened_pyramid(
        base_cyx.data, step=params.downsample, reducer=np.mean, min_size=params.min_size
    )
    if len(levels) == 0:
        return "no-levels"
    tree = build_multiscale_tree(
        levels,
        dims=("c", "y", "x"),
        transform=transform,
        channels=channels,
        dtype=base_cyx.dtype,
        coordinate_system=coordinate_system,
    )
    Image2DModel.validate(tree)
    _write_derived(sdata, zarr_path, "images", cache_key, tree)
    _stamp_marker(
        zarr_path,
        "images",
        cache_key,
        DERIVED_CACHE_ATTR,
        image_pyramid_marker(
            source_image_key=image_key,
            downsample=params.downsample,
            min_size=params.min_size,
            levels=len(levels),
        ),
    )
    return "built"


def build_image_pyramids_in_store(
    sdata: Any,
    zarr_path: Path,
    *,
    image_keys: Sequence[str] | None = None,
    params: ViewerCacheParams | None = None,
) -> dict[str, str]:
    """Build the viewer image pyramid of every base image element in an open store.

    The caller must hold ``spatialdata_write_lock(zarr_path)``; this helper never
    takes it. It never writes root attributes (only derived image elements and
    their markers).

    Args:
        sdata: SpatialData object backed by ``zarr_path``.
        zarr_path: The store behind ``sdata``.
        image_keys: Restrict to these image elements; None means every
            non-derived image element.
        params: Cache parameters (``downsample``, ``min_size``,
            ``build_image_pyramid``, ``force``).

    Returns:
        ``{image_key: status}`` in sorted key order; status is ``"built"``,
        ``"skipped"``, ``"no-levels"``, ``"stored_multiscale"`` or ``"disabled"``.

    Raises:
        ValueError: If a requested key is not a non-derived image element of the
            store.
    """
    params = params or ViewerCacheParams()
    zarr_path = Path(zarr_path)
    available = sorted(str(k) for k in sdata.images if not is_derived_cache_key(str(k)))
    if image_keys is None:
        keys = available
    else:
        requested = {str(k) for k in image_keys}
        missing = sorted(requested - set(available))
        if missing:
            raise ValueError(
                f"Image element(s) {missing} are not base image elements of "
                f"{zarr_path} (base images: {available})"
            )
        keys = [k for k in available if k in requested]
    if not params.build_image_pyramid:
        return {key: "disabled" for key in keys}
    statuses: dict[str, str] = {}
    for key in keys:
        element = sdata.images[key]
        if isinstance(element, DataTree):
            # Already multiscale on disk: the viewer uses it as is.
            statuses[key] = "stored_multiscale"
            continue
        base_cyx = image_to_cyx(element)
        transformations = get_transformation(element, get_all=True)
        coordinate_system = (
            "global" if "global" in transformations else sorted(transformations)[0]
        )
        statuses[key] = build_image_pyramid(
            sdata=sdata,
            zarr_path=zarr_path,
            image_key=key,
            base_cyx=base_cyx,
            channels=_channel_labels(base_cyx),
            transform=transformations[coordinate_system],
            params=params,
            coordinate_system=coordinate_system,
        )
        log_status(f"Image pyramid for images[{key}]: {statuses[key]}")
        force_release(note=f"after image pyramid for {key}")
    return statuses


def build_image_pyramid_caches(
    zarr_path: str | Path,
    *,
    image_keys: Sequence[str] | None = None,
    params: ViewerCacheParams | None = None,
) -> dict[str, Any]:
    """Pre-build viewer image pyramids in an image-only store (no masks, no platform).

    Takes ``spatialdata_write_lock(zarr_path)`` from the read until the last cache
    is written.

    Returns:
        ``{"image_pyramids": {image_key: status}}``.
    """
    zarr_path = Path(zarr_path)
    with spatialdata_write_lock(zarr_path):
        sdata = read_zarr(zarr_path, selection=("images",))
        statuses = build_image_pyramids_in_store(
            sdata, zarr_path, image_keys=image_keys, params=params
        )
    return {"image_pyramids": statuses}


def _derived_complete(
    zarr_path: Path,
    group: str,
    cache_key: str,
    kind: str,
    source_key: str,
    params: ViewerCacheParams,
    width: int | None = None,
) -> bool:
    """True if an existing derived cache marker matches the current expected format."""
    marker = _read_marker(zarr_path, group, cache_key, DERIVED_CACHE_ATTR)
    if (
        not marker.get("complete")
        or marker.get("version") != VIEWER_DERIVED_CACHE_VERSION
    ):
        return False
    if marker.get("kind") != kind:
        return False
    source_field = "source_image_key" if group == "images" else "source_label_key"
    if marker.get(source_field) != str(source_key):
        return False
    if kind == "label_outline":
        # Mirror every key the viewer's ``_derived_cache_complete`` compares. A
        # marker missing any of these (or carrying a pre-coverage_mean_v1 outline)
        # is treated as stale here so we rebuild it rather than leaving behind a
        # cache the viewer will silently discard and regenerate itself.
        return (
            marker.get("width") == width
            and marker.get("min_size") == params.min_size
            and marker.get("tile_size") == PYRAMID_TILE
            and marker.get("pyramid_mode") == OUTLINE_PYRAMID_MODE
            and marker.get("value_max") == OUTLINE_COVERAGE_MAX
        )
    return (
        marker.get("downsample") == params.downsample
        and marker.get("min_size") == params.min_size
    )


def _write_derived(
    sdata: Any, _zarr_path: Path, group: str, cache_key: str, tree: Any
) -> None:
    """Replace + persist one derived element, tolerating an existing on-disk copy.

    Routed through ``write_or_replace_element`` rather than a bare
    ``sdata.write_element(overwrite=True)``. SpatialData refuses an in-place
    overwrite when the target path is a subfolder of the store already backing the
    object -- which is always true for these caches, since they live inside the
    enriched zarr they are derived from. A bare write therefore only ever
    succeeded on a FIRST build; refreshing an existing cache aborted the run with
    "Cannot overwrite. The target path of the write operation is in use". The
    helper recognises that refusal and retries via a recoverable element backup.
    """
    write_or_replace_element(sdata, cache_key, group, tree, overwrite=True)
