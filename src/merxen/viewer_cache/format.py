"""Element keys and completion markers the napari comparison viewer trusts.

Every constant and helper here mirrors the viewer package
(``napari-compare-xenium-merscope``) so a cache written by this pipeline is
byte-compatible with what the viewer builds on the fly. The viewer decides
whether to reuse a cache by matching the element key AND a completion-marker
dict stamped into the element's zarr attrs; if either drifts, the viewer
silently rebuilds. Keep this module in lockstep with the viewer:

* ``DERIVED_CACHE_PREFIX`` / ``DERIVED_CACHE_ATTR`` -- ``utils.py`` in the viewer.
* ``VIEWER_DERIVED_CACHE_VERSION`` -- the viewer's ``DERIVED_CACHE_VERSION``.
* ``LABEL_CACHE_ATTR`` / ``LABEL_CACHE_VERSION`` -- the viewer's ``LABEL_CACHE_ATTR``
  marker (which carries its own inline ``version``, independent of the derived
  cache version).

If the viewer bumps either version, bump the matching constant here; the
golden-value test in ``tests/test_viewer_cache/test_format.py`` fails loudly
until both are updated together.
"""

from __future__ import annotations

import re
from hashlib import blake2s

#: Prefix marking a private, viewer-derived zarr element (image/label pyramids,
#: outline pyramids). Mirrors the viewer's ``DERIVED_CACHE_PREFIX``.
DERIVED_CACHE_PREFIX = "_napari_compare_"

#: Attrs key holding the completion marker for derived pyramid/outline caches.
DERIVED_CACHE_ATTR = "napari_compare_derived_cache"

#: Attrs key holding the completion marker for a rasterized base label mask.
LABEL_CACHE_ATTR = "napari_compare_label_cache"

#: Viewer's ``DERIVED_CACHE_VERSION`` -- stamped on image/label pyramid + outline
#: caches. A cache with a different version is rebuilt by the viewer.
VIEWER_DERIVED_CACHE_VERSION = 2

#: Inline ``version`` in the base-mask ``LABEL_CACHE_ATTR`` marker. v2 rasterizes
#: with true instance ids (v1 used positional ``id + 1`` and mis-joined overlays).
LABEL_CACHE_VERSION = 2

#: Largest-axis size at/under which a pyramid stops adding coarser levels.
PYRAMID_MIN_SIZE = 4096

#: Hard cap on the number of pyramid levels.
PYRAMID_MAX_LEVELS = 10

#: Default multiscale rechunk tile for coarse pyramid levels.
PYRAMID_TILE = 1024


def is_derived_cache_key(key: str) -> bool:
    """Return True for a private viewer-derived zarr element key."""
    return str(key).startswith(DERIVED_CACHE_PREFIX)


def safe_cache_token(value: str, max_len: int = 96) -> str:
    """Return a zarr-element-safe token that stays readable for common keys.

    Verbatim reproduction of the viewer's ``_safe_cache_token`` so derived keys
    match exactly. Non-``[A-Za-z0-9_.-]`` runs collapse to ``_``; if the token
    changed or is too long a ``__h<blake2s>`` digest suffix is appended.
    """
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    if not token:
        token = "cache"
    if token == str(value) and len(token) <= max_len:
        return token

    digest = blake2s(str(value).encode("utf-8"), digest_size=5).hexdigest()
    token = token[: max(1, max_len - len(digest) - 3)].strip("_")
    return f"{token}__h{digest}" if token else f"h{digest}"


def label_key_for_shape_key(shape_key: str, existing_labels: set[str]) -> str:
    """Return the label element key a shape rasterizes to.

    Mirrors the viewer: an already-present label element of the same name is
    reused verbatim, otherwise ``<shape_key>_labels``.
    """
    if str(shape_key) in existing_labels:
        return str(shape_key)
    return f"{shape_key}_labels"


def derived_outline_cache_key(label_key: str, width: int) -> str:
    """Key for a precomputed outline pyramid (labels group)."""
    return (
        f"{DERIVED_CACHE_PREFIX}outline__{safe_cache_token(label_key)}__w{int(width)}"
    )


def derived_image_pyramid_cache_key(image_key: str, downsample: int) -> str:
    """Key for a materialized coarse-level image pyramid (images group)."""
    token = safe_cache_token(image_key)
    return f"{DERIVED_CACHE_PREFIX}imgpyr__{token}__ds{int(downsample)}"


def derived_label_pyramid_cache_key(label_key: str, downsample: int) -> str:
    """Key for a materialized coarse-level label pyramid (labels group)."""
    token = safe_cache_token(label_key)
    return f"{DERIVED_CACHE_PREFIX}labelpyr__{token}__ds{int(downsample)}"


def label_cache_marker(
    *,
    source_shape_key: str,
    shape: tuple[int, int],
    chunks: tuple[int, int],
) -> dict[str, object]:
    """Completion marker written into a rasterized base label mask's attrs."""
    return {
        "version": LABEL_CACHE_VERSION,
        "complete": True,
        "source_shape_key": str(source_shape_key),
        "shape": [int(shape[0]), int(shape[1])],
        "chunks": [int(chunks[0]), int(chunks[1])],
    }


def label_pyramid_marker(
    *,
    source_label_key: str,
    downsample: int,
    min_size: int,
    levels: int,
) -> dict[str, object]:
    """Completion marker for a label pyramid cache."""
    return {
        "version": VIEWER_DERIVED_CACHE_VERSION,
        "complete": True,
        "kind": "label_pyramid",
        "source_label_key": str(source_label_key),
        "downsample": int(downsample),
        "min_size": int(min_size),
        "levels": int(levels),
    }


def image_pyramid_marker(
    *,
    source_image_key: str,
    downsample: int,
    min_size: int,
    levels: int,
) -> dict[str, object]:
    """Completion marker for an image pyramid cache."""
    return {
        "version": VIEWER_DERIVED_CACHE_VERSION,
        "complete": True,
        "kind": "image_pyramid",
        "source_image_key": str(source_image_key),
        "downsample": int(downsample),
        "min_size": int(min_size),
        "levels": int(levels),
    }


def outline_marker(
    *,
    source_label_key: str,
    width: int,
    source: str,
    levels: int,
    source_shapes: list[list[int]],
) -> dict[str, object]:
    """Completion marker for a label outline pyramid cache."""
    return {
        "version": VIEWER_DERIVED_CACHE_VERSION,
        "complete": True,
        "kind": "label_outline",
        "source_label_key": str(source_label_key),
        "width": int(width),
        "source": str(source),
        "levels": int(levels),
        "source_shapes": [[int(h), int(w)] for h, w in source_shapes],
    }
