"""Image loading, tiling, and z-projection for spatial transcriptomics images."""

from __future__ import annotations

import logging
import re
from contextlib import suppress
from typing import Any

import numpy as np

from merxen._typing import ImageSource

logger = logging.getLogger(__name__)


def list_plane_keys(
    images: dict[str, Any],
    prefix: str | None = None,
) -> list[tuple[int, str]]:
    """Extract ``(z_index, key)`` tuples from image names matching ``prefix_zN``.

    Args:
        images: Dict-like mapping of image names to image objects.
        prefix: Optional prefix filter. Only keys starting with this are included.

    Returns:
        Sorted list of (z_index, image_key) tuples.
    """
    pat = re.compile(r"^(?P<prefix>.+)_z(?P<z>\d+)$")
    out = []
    for k in images:
        m = pat.match(str(k))
        if not m:
            continue
        if prefix is not None and not str(k).startswith(prefix):
            continue
        out.append((int(m.group("z")), str(k)))
    return sorted(out)


def _get_image_dataarray(img_obj: Any) -> Any:
    """Extract an xarray DataArray from a multi-scale image or raw array.

    Args:
        img_obj: An xarray DataArray, multi-scale image, or array-like.

    Returns:
        The underlying DataArray or array.
    """
    img_xr = img_obj
    if hasattr(img_obj, "__contains__") and "scale0" in img_obj:
        img_xr = img_obj["scale0"].ds["image"]
    else:
        with suppress(Exception):
            img_xr = img_obj["scale0"].ds["image"]
    return img_xr


def build_image_source(
    img_obj: Any,
    requested_channels: list[str] | None = None,
    *,
    as_float32: bool = True,
) -> ImageSource:
    """Normalize an image object into a standard image source dict.

    Handles xarray DataArrays and raw numpy arrays, with optional channel
    selection and float32 conversion.

    Args:
        img_obj: Image data (xarray, multi-scale image, or numpy array).
        requested_channels: Optional list of channel names to keep.
        as_float32: Whether to convert to float32 when fetching tiles.

    Returns:
        Dict with keys: kind, data, channels, shape, as_float32.
    """
    img_xr = _get_image_dataarray(img_obj)

    if hasattr(img_xr, "dims"):
        da = _transpose_to_yxc(img_xr)

        if da is not None:
            channels = _extract_channel_names(da)
            if requested_channels is not None and len(channels) > 0:
                da, channels = _select_channels(da, channels, requested_channels)

            h = int(da.sizes["y"])
            w = int(da.sizes["x"])
            c = int(da.sizes["c"])
            return {
                "kind": "xarray",
                "data": da,
                "channels": channels,
                "shape": (h, w, c),
                "as_float32": as_float32,
            }

    # Fallback: compute to numpy
    arr = img_xr.data.compute() if hasattr(img_xr, "data") else np.asarray(img_xr)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    elif arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[-1] > 8:
        arr = np.moveaxis(arr, 0, -1)

    channels = [f"c{i}" for i in range(arr.shape[-1])]
    if requested_channels is not None and len(channels) > 0:
        keep = [c for c in requested_channels if c in channels]
        if not keep:
            raise ValueError(
                f"Requested channels {requested_channels} not found. "
                f"Available channels: {channels}"
            )
        idx = [channels.index(c) for c in keep]
        arr = arr[..., idx]
        channels = keep

    if as_float32:
        arr = arr.astype(np.float32, copy=False)

    return {
        "kind": "array",
        "data": arr,
        "channels": channels,
        "shape": arr.shape,
        "as_float32": as_float32,
    }


def fetch_tile(
    source: ImageSource,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Extract a rectangular tile from an image source.

    Args:
        source: Image source dict from build_image_source.
        y0: Start row (inclusive).
        y1: End row (exclusive).
        x0: Start column (inclusive).
        x1: End column (exclusive).

    Returns:
        Numpy array of shape (y1-y0, x1-x0, channels).
    """
    if source["kind"] == "xarray":
        arr = np.asarray(
            source["data"].isel(y=slice(y0, y1), x=slice(x0, x1)).data.compute()
        )
    else:
        arr = np.asarray(source["data"][y0:y1, x0:x1, :])
    if source.get("as_float32", False):
        arr = arr.astype(np.float32, copy=False)
    return arr


def prepare_merscope_plane_sources(
    sdata: Any,
    selected_keys: list[str],
    requested_channels: list[str] | None = None,
) -> tuple[list[ImageSource], int, int, list[str]]:
    """Prepare image sources from multiple MERSCOPE z-planes.

    Args:
        sdata: A SpatialData object with .images attribute.
        selected_keys: List of image keys (one per z-plane).
        requested_channels: Optional channel filter.

    Returns:
        Tuple of (sources, height, width, channels).

    Raises:
        ValueError: If selected_keys is empty or plane shapes are inconsistent.
    """
    if len(selected_keys) == 0:
        raise ValueError("selected_keys is empty")

    sources = []
    for k in selected_keys:
        sources.append(
            build_image_source(
                sdata.images[k],
                requested_channels=requested_channels,
                as_float32=True,
            )
        )

    base_shape = sources[0]["shape"]
    for src in sources[1:]:
        if src["shape"][:2] != base_shape[:2]:
            raise ValueError(
                f"Inconsistent plane shapes: {base_shape[:2]} vs {src['shape'][:2]}"
            )

    h, w, _ = base_shape
    channels = sources[0]["channels"]
    return sources, h, w, channels


def fetch_merscope_projected_tile(
    plane_sources: list[ImageSource],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Fetch a tile from all z-planes and merge via maximum projection.

    Args:
        plane_sources: List of image sources (one per z-plane).
        y0: Start row (inclusive).
        y1: End row (exclusive).
        x0: Start column (inclusive).
        x1: End column (exclusive).

    Returns:
        Max-projected numpy array.
    """
    proj = None
    for src in plane_sources:
        arr = fetch_tile(src, y0, y1, x0, x1)
        if proj is None:
            proj = arr
        else:
            np.maximum(proj, arr, out=proj)
        del arr
    if proj is None:
        raise ValueError("plane_sources is empty")
    return proj


def prepare_cellpose_input(
    image: np.ndarray,
    factor_rescale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize an image tile for Cellpose input.

    Performs percentile normalization (2nd-98th), converts to uint8,
    and ensures 3-channel RGB format.

    Args:
        image: Input image tile (H, W) or (H, W, C).
        factor_rescale: Must be 1.0 (native resolution only).

    Returns:
        Tuple of (img8, img_seg, scale_factor). Both img8 and img_seg
        are the same uint8 array; scale_factor is always 1.0.

    Raises:
        ValueError: If factor_rescale > 1.0.
    """
    if factor_rescale > 1.0:
        raise ValueError("factor_rescale > 1.0 is disabled for native-resolution mode.")

    img = image.astype(np.float32, copy=False)
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
    img8 = (img * 255).astype(np.uint8)

    if img8.ndim == 2:
        img8 = np.stack([img8] * 3, axis=-1)
    elif img8.shape[-1] == 1:
        img8 = np.repeat(img8, 3, axis=-1)
    elif img8.shape[-1] == 2:
        img8 = np.concatenate([img8, np.zeros_like(img8[..., :1])], axis=-1)
    elif img8.shape[-1] > 3:
        img8 = img8[..., :3]

    return img8, img8, 1.0


# --- Internal helpers ---


def _transpose_to_yxc(img_xr: Any) -> Any | None:
    """Transpose an xarray DataArray to (y, x, c) ordering."""
    if all(d in img_xr.dims for d in ("y", "x", "c")):
        return img_xr.transpose("y", "x", "c")
    if all(d in img_xr.dims for d in ("c", "y", "x")):
        return img_xr.transpose("y", "x", "c")
    if all(d in img_xr.dims for d in ("y", "x")):
        return img_xr.expand_dims(c=["c0"]).transpose("y", "x", "c")
    return None


def _extract_channel_names(da: Any) -> list[str]:
    """Extract channel names from an xarray DataArray."""
    if "c" in da.coords:
        return [str(c) for c in da.coords["c"].values]
    return [f"c{i}" for i in range(da.shape[-1])]


def _select_channels(
    da: Any,
    channels: list[str],
    requested: list[str],
) -> tuple[Any, list[str]]:
    """Select requested channels from a DataArray.

    Raises:
        ValueError: If none of the requested channels are available.
    """
    keep = [c for c in requested if c in channels]
    if not keep:
        raise ValueError(
            f"Requested channels {requested} not found. Available channels: {channels}"
        )
    idx = [channels.index(c) for c in keep]
    return da.isel(c=idx), keep
