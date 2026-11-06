"""NATIVE_LATEST: the latest store of a native (vendor-segmentation) row.

A row with ``segmentation_source=native`` (ATERA in AT-6) runs no Cellpose and
no ProSeg. Its latest store is derived from the built source store instead:

- immutable payloads (image chunks, label chunks, ``points/**.parquet``) are
  hardlinked (or copied with ``link_mode="copy"``) and made read-only, so an
  in-place write fails instead of changing the source;
- every zarr metadata file, the ``atera/`` sidecars, the points index and the
  root attrs are copied;
- the vendor shapes and table are written fresh under the registry names:
  ``cell_boundaries`` -> ``atera_cell_boundaries``, ``nucleus_boundaries`` ->
  ``atera_nucleus``, ``table`` -> ``table_original`` (region
  ``atera_cell_boundaries``). The source names never appear in the latest store.

The segmentation registry gets ``original`` (``atera_cell_boundaries``,
``table_original``, ``original_assignment``) and ``original_nucleus``
(``atera_nucleus``, no table, a nucleus-row id namespace and
``cell_key="cell_instance_id"``).

The signatures are final (AT-6 WP-F); the bodies land in WP-A. The reference
layout lives in ``tests/test_io/atera_native_fixture.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from merxen.config import NativeLatestConfig

NATIVE_LATEST_MARKER_NAME: Final = "merxen_native_latest.json"
"""Store-root file recording the source build a native latest store came from."""

NATIVE_LATEST_FORMAT_VERSION: Final = 1
"""Layout version of the native latest store (bumped on a layout change)."""

NATIVE_CELL_SHAPES_KEY: Final = "atera_cell_boundaries"
"""Latest-store name of the vendor cell shapes (``original_cell_shape_key``)."""

NATIVE_NUCLEUS_SHAPES_KEY: Final = "atera_nucleus"
"""Latest-store name of the vendor nucleus shapes (``original_nucleus_shape_key``)."""

NATIVE_TABLE_KEY: Final = "table_original"
"""Latest-store name of the vendor cell table."""

SOURCE_ELEMENT_RENAMES: Final = {
    ("shapes", "cell_boundaries"): ("shapes", NATIVE_CELL_SHAPES_KEY),
    ("shapes", "nucleus_boundaries"): ("shapes", NATIVE_NUCLEUS_SHAPES_KEY),
    ("tables", "table"): ("tables", NATIVE_TABLE_KEY),
}
"""Source element -> latest element written fresh by NATIVE_LATEST."""

NUCLEUS_CELL_KEY: Final = "cell_instance_id"
"""Nucleus-shape column naming the owning cell (nucleus ids are not cell ids)."""

NUCLEUS_ID_NAMESPACE: Final = "original_nucleus_row_v1"
"""Id namespace of the per-nucleus vendor shapes (nucleus row + 1)."""


@dataclass
class NativeLatestReport:
    """What one NATIVE_LATEST run did (written as ``native_latest_report.json``).

    Attributes:
        dataset_name: Dataset alias.
        latest_path: The persistent latest store.
        reused: Whether an existing latest store of the same source build was
            reused without a rebuild.
        source_build_fingerprint: Build fingerprint of the source store.
        link_mode: ``hardlink`` or ``copy``.
        files_linked: Payload files hardlinked into the latest store.
        bytes_linked: Bytes of those files (shared with the source).
        files_copied: Files copied (metadata, sidecars, index).
        bytes_copied: Bytes of those files.
        elements_written: Elements written fresh (renamed shapes and table).
        apparent_size_delta_bytes: New bytes the latest store adds on disk.
        extra: Further provenance fields.
    """

    dataset_name: str
    latest_path: Path
    reused: bool = False
    source_build_fingerprint: str | None = None
    link_mode: str = "hardlink"
    files_linked: int = 0
    bytes_linked: int = 0
    files_copied: int = 0
    bytes_copied: int = 0
    elements_written: list[str] = field(default_factory=list)
    apparent_size_delta_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def build_native_latest(cfg: NativeLatestConfig) -> NativeLatestReport:
    """Build (or reuse) the latest store of a native row from its source store.

    Args:
        cfg: The NATIVE_LATEST configuration.

    Returns:
        The report of the run.

    Raises:
        NotImplementedError: Until AT-6 WP-A lands the body.
    """
    raise NotImplementedError(
        "build_native_latest lands in AT-6 WP-A (native segmentation)"
    )


def ensure_native_elements(latest_path: Path, source_path: Path) -> list[str]:
    """Re-create missing renamed vendor elements and their registry entries.

    Idempotent; ENRICH calls it after removing partial artifacts.

    Args:
        latest_path: The native latest store.
        source_path: The source store it was built from.

    Returns:
        The elements (``"shapes/atera_nucleus"`` and so on) it re-created.

    Raises:
        NotImplementedError: Until AT-6 WP-A lands the body.
    """
    raise NotImplementedError(
        "ensure_native_elements lands in AT-6 WP-A (native segmentation)"
    )
