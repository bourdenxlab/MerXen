"""SpatialData table and shape keys of the analysis segmentations (plan §4.8).

``ANALYSIS_LAYER_KEYS`` mirrors ``analysisLayerKeys`` and
``clustered_table_key`` mirrors ``clusteredSpatialdataTableKey`` (hook H2),
both in ``workflows/main.nf``; ``tests/test_workflows/test_table_keys_sync.py``
keeps the two sides equal. Python code that needs the clustered table key
calls ``clustered_table_key`` instead of building ``*_clustering_squidpy``
itself, so the table-key suffix (plan §4.8, OD-A3) reaches every consumer.

This module imports only the standard library, so every pipeline
environment can use it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

PLATFORMS: Final[tuple[str, ...]] = ("MERSCOPE", "XENIUM")
PROSEG_TABLE_KEY: Final = "table_MOSAIK_proseg"
ORIGINAL_TABLE_KEY: Final = "table_original"
# Appended to the source table key to name the clustered table written back
# by CLUSTERING_SQUIDPY_FINALIZE.
CLUSTERED_TABLE_KEY_TAG: Final = "_clustering_squidpy"
# A suffix is one lower-case token, so the key stays a valid zarr element name
# (no "/", no spaces) and matches what the Nextflow params accept.
TABLE_KEY_SUFFIX_PATTERN: Final = re.compile(r"^[a-z0-9_]*$")


@dataclass(frozen=True)
class AnalysisLayerKeys:
    """SpatialData element keys of one platform x segmentation branch.

    Attributes:
        table_key: Source cell table (``tables/<table_key>``).
        shape_key: Cell boundary shapes that the table annotates.
    """

    table_key: str
    shape_key: str


def _platform_independent(
    table_key: str, shape_key: str
) -> Mapping[str, AnalysisLayerKeys]:
    keys = AnalysisLayerKeys(table_key=table_key, shape_key=shape_key)
    return MappingProxyType(dict.fromkeys(PLATFORMS, keys))


_PROSEG_KEYS = _platform_independent(PROSEG_TABLE_KEY, "MOSAIK_proseg")
_ORIGINAL_KEYS: Mapping[str, AnalysisLayerKeys] = MappingProxyType(
    {
        "MERSCOPE": AnalysisLayerKeys(ORIGINAL_TABLE_KEY, "merscope_cell_boundaries"),
        "XENIUM": AnalysisLayerKeys(ORIGINAL_TABLE_KEY, "xenium_cell_boundaries"),
    }
)
_CELLPOSE_KEYS = _platform_independent("table_MOSAIK_cellpose", "MOSAIK_cellpose")

# Segmentation name (and its aliases, as in ``main.nf``) -> platform -> keys.
ANALYSIS_LAYER_KEYS: Final[Mapping[str, Mapping[str, AnalysisLayerKeys]]] = (
    MappingProxyType(
        {
            "proseg": _PROSEG_KEYS,
            "reseg": _PROSEG_KEYS,
            "original": _ORIGINAL_KEYS,
            "original_seg": _ORIGINAL_KEYS,
            "proseg_hybrid": _platform_independent(
                "table_MOSAIK_proseg_hybrid", "MOSAIK_proseg_hybrid"
            ),
            "cellpose": _CELLPOSE_KEYS,
            "proseg_mask": _CELLPOSE_KEYS,
            "proseg_geometry_assignment": _platform_independent(
                "table_MOSAIK_proseg_geometry_assignment", "MOSAIK_proseg"
            ),
        }
    )
)


def analysis_layer_keys(platform: str, segmentation: str) -> AnalysisLayerKeys:
    """Return the table and shape keys of one analysis branch.

    Mirrors ``analysisLayerKeys(platform, segmentation)`` in ``main.nf``.

    Args:
        platform: ``MERSCOPE`` or ``XENIUM`` (case-insensitive).
        segmentation: Analysis segmentation or one of its aliases (e.g.
            ``reseg``, ``original_seg``, ``proseg_hybrid``).

    Returns:
        The keys of that branch.

    Raises:
        ValueError: On an unknown platform or segmentation.
    """
    segmentation_key = str(segmentation).strip().lower()
    platform_key = str(platform).strip().upper()
    if segmentation_key not in ANALYSIS_LAYER_KEYS:
        raise ValueError(
            f"Unknown analysis segmentation: {segmentation!r}; expected one of "
            f"{sorted(ANALYSIS_LAYER_KEYS)}"
        )
    if platform_key not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform!r}; expected {PLATFORMS}")
    return ANALYSIS_LAYER_KEYS[segmentation_key][platform_key]


def validate_table_key_suffix(suffix: str) -> str:
    """Check a clustered table-key suffix.

    Args:
        suffix: ``""`` for none, else one lower-case token (letters, digits,
            underscores), e.g. ``mapfirst``. Surrounding spaces are removed.

    Returns:
        The stripped suffix.

    Raises:
        ValueError: If the suffix is not a lower-case token.
    """
    cleaned = str(suffix).strip()
    if not TABLE_KEY_SUFFIX_PATTERN.fullmatch(cleaned):
        raise ValueError(
            f"table key suffix {suffix!r} must be a lower-case token "
            "(letters, digits, underscores)"
        )
    return cleaned


def clustered_table_key(
    source_table_key: str,
    segmentation: str | None,
    suffix: str = "",
) -> str:
    """Return the key of the clustered SpatialData table for a source table.

    Mirrors ``clusteredSpatialdataTableKey`` in ``main.nf`` (hook H2), with
    its three return paths:

    - reseg, or the ProSeg source table: ``table_MOSAIK_proseg_clustering_squidpy``;
    - original_seg, or the original source table:
      ``table_original_clustering_squidpy``;
    - anything else: ``<source_table_key>_clustering_squidpy``.

    A non-empty ``suffix`` appends ``_<suffix>`` on every path, so a map_first
    run of a species that has not flipped writes
    ``<...>_clustering_squidpy_mapfirst`` and never overwrites the legacy
    table (plan §4.8, OD-A3). Legacy runs pass no suffix.

    Args:
        source_table_key: Key of the source cell table.
        segmentation: Analysis segmentation (case-insensitive), or ``None``.
        suffix: Table-key suffix token; ``""`` for none.

    Returns:
        The clustered table key.

    Raises:
        ValueError: If ``suffix`` is not a lower-case token.
    """
    suffix_token = validate_table_key_suffix(suffix)
    segmentation_key = "" if segmentation is None else str(segmentation).strip().lower()
    source_key = str(source_table_key)
    if segmentation_key == "reseg" or source_key == PROSEG_TABLE_KEY:
        base_key = PROSEG_TABLE_KEY
    elif segmentation_key == "original_seg" or source_key == ORIGINAL_TABLE_KEY:
        base_key = ORIGINAL_TABLE_KEY
    else:
        base_key = source_key
    suffix_fragment = f"_{suffix_token}" if suffix_token else ""
    return f"{base_key}{CLUSTERED_TABLE_KEY_TAG}{suffix_fragment}"


def clustered_table_key_suffix(table_key: str) -> str | None:
    """Return the suffix a clustered table key carries.

    The inverse of ``clustered_table_key`` on its suffix: the text after the
    last ``_clustering_squidpy`` tag, without its leading ``_``.

    Args:
        table_key: A SpatialData table key.

    Returns:
        ``""`` for an unsuffixed clustered key
        (``<...>_clustering_squidpy``), the suffix for a suffixed one
        (``"mapfirst"`` for ``<...>_clustering_squidpy_mapfirst``), or
        ``None`` when ``table_key`` is not a clustered table key.
    """
    head, tag, tail = str(table_key).rpartition(CLUSTERED_TABLE_KEY_TAG)
    if not tag or not head:
        return None
    if not tail:
        return ""
    if not tail.startswith("_") or not TABLE_KEY_SUFFIX_PATTERN.match(tail[1:]):
        return None
    return tail[1:] or None
