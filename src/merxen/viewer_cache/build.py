"""Build the viewer's on-the-fly caches into an enriched SpatialData store.

For each requested segmentation the builder rasterizes an instance-id label
mask, then materializes the label pyramid, the outline pyramid, and (once) the
morphology-image pyramid -- writing each with the exact key + completion marker
the viewer trusts. Everything is idempotent: an artifact whose marker already
matches the current format is skipped unless ``force`` is set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from spatialdata import read_zarr
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Affine, get_transformation

from merxen.io.image_source import image_to_cyx
from merxen.memory import force_release, log_status
from merxen.viewer_cache.format import (
    DERIVED_CACHE_ATTR,
    LABEL_CACHE_ATTR,
    LABEL_CACHE_VERSION,
    PYRAMID_MIN_SIZE,
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

#: Default segmentation shape keys to rasterize per platform. Only keys present
#: in the store are built. Mirrors the viewer's mask targets: reseg (proseg),
#: cellpose, and the original/instrument boundaries.
DEFAULT_SHAPE_KEYS: dict[str, tuple[str, ...]] = {
    "MERSCOPE": ("MOSAIK_proseg", "MOSAIK_cellpose", "merscope_cell_boundaries"),
    "XENIUM": ("MOSAIK_proseg", "MOSAIK_cellpose", "xenium_cell_boundaries"),
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


def build_viewer_caches(
    zarr_path: str | Path,
    platform: str,
    original_data_path: str | Path,
    *,
    transform_path: str | Path | None = None,
    params: ViewerCacheParams | None = None,
) -> dict[str, Any]:
    """Pre-build viewer caches into the enriched store at ``zarr_path``.

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

    sdata = read_zarr(zarr_path)
    image_key, base_cyx, channels, image_transform = _resolve_base_image(sdata)
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

    shape_keys = params.shape_keys or DEFAULT_SHAPE_KEYS.get(platform, ())
    existing_labels = {str(k) for k in sdata.labels}
    summary: dict[str, Any] = {"masks": {}, "image_pyramid": None}

    for shape_key in shape_keys:
        if shape_key not in sdata.shapes:
            log_status(f"[{platform}] Shape '{shape_key}' absent; skipping mask build.")
            continue
        label_key = label_key_for_shape_key(shape_key, existing_labels)
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
        )
        force_release(note=f"after viewer-cache build for {shape_key}")

    if params.build_image_pyramid:
        summary["image_pyramid"] = _build_image_pyramid(
            sdata=sdata,
            zarr_path=zarr_path,
            image_key=image_key,
            base_cyx=base_cyx,
            channels=channels,
            transform=image_transform,
            params=params,
        )

    return summary


def _resolve_base_image(sdata: Any) -> tuple[str, Any, list[str], Any]:
    """Return (key, (c,y,x) DataArray, channels, global transform) of the real image.

    Skips derived image caches so the label grid is never sized off a downsampled
    cache -- the same rule the viewer's ``_image_grid_for_labels`` uses.
    """
    image_key = next(
        (str(k) for k in sdata.images if not is_derived_cache_key(str(k))),
        None,
    )
    if image_key is None:
        raise RuntimeError(
            f"No non-derived image found to size the label grid "
            f"(images: {sorted(map(str, sdata.images.keys()))})."
        )
    base_cyx = image_to_cyx(sdata.images[image_key])
    channels = _channel_labels(base_cyx)
    transform = get_transformation(
        sdata.images[image_key], to_coordinate_system="global"
    )
    return image_key, base_cyx, channels, transform


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
    )
    result["outline"] = _build_outline(
        zarr_path=zarr_path,
        sdata=sdata,
        label_key=label_key,
        base_da=base_da,
        transform=spatialdata_affine,
        params=params,
    )
    return result


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
) -> int:
    """Write an empty label element, then fill its s0 array chunk-by-chunk."""
    height, width = shape
    empty = da.zeros(shape, chunks=chunks, dtype=id_dtype)
    label_da = xr.DataArray(empty, dims=("y", "x"))
    elem = Labels2DModel.parse(label_da, transformations={"global": spatialdata_affine})
    sdata.labels[label_key] = elem
    sdata.write_element(label_key, overwrite=(label_key in _on_disk_labels(zarr_path)))

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


def _on_disk_labels(zarr_path: Path) -> set[str]:
    labels_dir = zarr_path / "labels"
    if not labels_dir.exists():
        return set()
    return {p.name for p in labels_dir.iterdir() if p.is_dir()}


def _build_label_pyramid(
    *,
    zarr_path: Path,
    sdata: Any,
    label_key: str,
    base_da: Any,
    transform: Affine,
    params: ViewerCacheParams,
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
        levels, dims=("y", "x"), transform=transform, dtype=base_da.dtype
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
    source = "synthetic" if len(outline_levels) > 1 else "single"
    tree = build_multiscale_tree(
        outline_levels, dims=("y", "x"), transform=transform, dtype=np.uint8
    )
    Labels2DModel.validate(tree)
    _write_derived(sdata, zarr_path, "labels", cache_key, tree)
    source_shapes = [
        [int(level.shape[0]), int(level.shape[1])] for level in outline_levels
    ]
    _stamp_marker(
        zarr_path,
        "labels",
        cache_key,
        DERIVED_CACHE_ATTR,
        outline_marker(
            source_label_key=label_key,
            width=width,
            source=source,
            levels=len(outline_levels),
            source_shapes=source_shapes,
        ),
    )
    return "built"


def _build_image_pyramid(
    *,
    sdata: Any,
    zarr_path: Path,
    image_key: str,
    base_cyx: Any,
    channels: list[str],
    transform: Any,
    params: ViewerCacheParams,
) -> str:
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
        return marker.get("width") == width
    return (
        marker.get("downsample") == params.downsample
        and marker.get("min_size") == params.min_size
    )


def _write_derived(
    sdata: Any, zarr_path: Path, group: str, cache_key: str, tree: Any
) -> None:
    """Replace + persist one derived element, tolerating an existing on-disk copy."""
    container = sdata.images if group == "images" else sdata.labels
    exists = cache_key in _on_disk_group(zarr_path, group)
    if cache_key in container:
        del container[cache_key]
    container[cache_key] = tree
    sdata.write_element(cache_key, overwrite=exists)


def _on_disk_group(zarr_path: Path, group: str) -> set[str]:
    group_dir = zarr_path / group
    if not group_dir.exists():
        return set()
    return {p.name for p in group_dir.iterdir() if p.is_dir()}
