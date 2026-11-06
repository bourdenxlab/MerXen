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

The store is built under ``<target>.partial-<uuid>`` and moved into place with
``os.replace``, so the target is either absent, the previous store, or a
complete new one. A target that already holds a native store of the same source
build is reused as it is (ENRICH writes into it in place). The reference layout
lives in ``tests/test_io/atera_native_fixture.py``.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import stat
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from merxen.config import NativeLatestConfig
from merxen.io.builders.atera import BUILD_MARKER_NAME, read_build_marker
from merxen.io.points_aggregates import read_root_attrs
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    ORIGINAL_ID_NAMESPACE,
    register_segmentation_branch,
    stamp_merxen_schema,
)
from merxen.masks import NUCLEUS_CELL_KEY as _MASKS_NUCLEUS_CELL_KEY
from merxen.masks import ORIGINAL_NUCLEUS_ROW_ID_NAMESPACE
from merxen.path_utils import remove_path, stage_existing_output
from merxen.platforms import platform_spec

logger = logging.getLogger(__name__)

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

NUCLEUS_CELL_KEY: Final = _MASKS_NUCLEUS_CELL_KEY
"""Nucleus-shape column naming the owning cell (nucleus ids are not cell ids)."""

NUCLEUS_ID_NAMESPACE: Final = ORIGINAL_NUCLEUS_ROW_ID_NAMESPACE
"""Id namespace of the per-nucleus vendor shapes (nucleus row + 1)."""

PROVENANCE_ATTR: Final = "merxen_native_latest"
"""Root attribute holding the provenance of a native latest store."""

POINTS_KEY: Final = "transcripts"
"""Points element of an ATERA store."""

ORIGINAL_ASSIGNMENT_COLUMN: Final = "original_assignment"
"""Transcript column holding the vendor cell id of each transcript."""

_ATERA: Final = platform_spec("ATERA")

REQUIRED_SOURCE_ELEMENTS: Final[tuple[str, ...]] = (
    f"images/{_ATERA.native_image_key}",
    *(f"labels/{key}" for key in _ATERA.original_label_keys),
    f"points/{POINTS_KEY}",
    *(f"{kind}/{name}" for kind, name in SOURCE_ELEMENT_RENAMES),
)
"""Elements a source store must hold to become a native latest store."""

_METADATA_FILE_NAMES: Final = frozenset(
    {"zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata"}
)
_RASTER_ROOTS: Final = frozenset({"images", "labels"})
_SKIPPED_ROOT_FILES: Final = frozenset({BUILD_MARKER_NAME, NATIVE_LATEST_MARKER_NAME})
_WRITE_BITS: Final = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class NativeLatestError(RuntimeError):
    """Raised when a native latest store cannot be built or reused safely."""


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


@dataclass
class _TreeCopy:
    """Files the tree walk put into the new store (paths relative to it)."""

    linked: list[str] = field(default_factory=list)
    copied: list[str] = field(default_factory=list)
    bytes_linked: int = 0
    bytes_copied: int = 0


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def build_native_latest(cfg: NativeLatestConfig) -> NativeLatestReport:
    """Build (or reuse) the latest store of a native row from its source store.

    Args:
        cfg: The NATIVE_LATEST configuration.

    Returns:
        The report of the run.

    Raises:
        NativeLatestError: If the source is not a complete ATERA build, the
            target holds something other than a native latest store (without
            ``force``), the source and target are on different filesystems in
            hardlink mode, or the built store fails its inode self-check.
    """
    # Linux link() does not follow a symlinked source, and the staged input is
    # a symlink: resolve it before anything else.
    source = Path(os.path.realpath(cfg.source_zarr_path))
    target = Path(os.path.abspath(cfg.persistent_latest_zarr_path))
    marker = _check_source(source, cfg)
    fingerprint = str(marker["build_fingerprint"])
    _check_target_is_not_the_source(source, target)
    _remove_leftover_builds(target, cfg.dataset_name)

    existing = _read_native_marker(target)
    if existing is not None and _is_reusable(existing, fingerprint, cfg.link_mode):
        hardened = _harden_reused_payloads(target, source, existing, cfg)
        recreated = ensure_native_elements(target, source)
        logger.info(
            "[%s] Reusing the native latest store of source build %s%s",
            cfg.dataset_name,
            fingerprint[:12],
            f" (re-created {', '.join(recreated)})" if recreated else "",
        )
        _stage_output(cfg, target)
        return NativeLatestReport(
            dataset_name=cfg.dataset_name,
            latest_path=target,
            reused=True,
            source_build_fingerprint=fingerprint,
            link_mode=cfg.link_mode,
            elements_written=recreated,
            extra={"previous_build": existing, "payloads_made_readonly": hardened},
        )
    if _target_occupied(target):
        if existing is None and not cfg.force:
            raise NativeLatestError(
                f"[{cfg.dataset_name}] The latest-store target exists and is not a "
                "native latest store; refusing to replace it. Remove it (or run "
                "`merxen native-latest` by hand with force: true in its config)."
            )
        logger.info(
            "[%s] Replacing the latest store at the target (%s)",
            cfg.dataset_name,
            "stale native build" if existing is not None else "force=true",
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial-{uuid.uuid4().hex}")
    try:
        report = _build_into(partial, source, target, fingerprint, cfg)
        _move_into_place(partial, target)
    except BaseException:
        remove_path(partial)
        raise
    _stage_output(cfg, target)
    logger.info(
        "[%s] Native latest store built: %d payload files %s (%.2f GB), "
        "%d files copied, %.3f GB new bytes",
        cfg.dataset_name,
        report.files_linked,
        "linked" if cfg.link_mode == "hardlink" else "copied",
        report.bytes_linked / 1e9,
        report.files_copied,
        (report.apparent_size_delta_bytes or 0) / 1e9,
    )
    return report


def ensure_native_elements(latest_path: Path, source_path: Path) -> list[str]:
    """Re-create missing renamed vendor elements and their registry entries.

    Idempotent; ENRICH calls it after removing partial artifacts. Elements that
    exist are left untouched; a missing one is written fresh from the source
    store. The registry entries ``original`` and ``original_nucleus`` and the
    ``segmentation_source`` attr are restored when missing or different.

    Args:
        latest_path: The native latest store.
        source_path: The source store it was built from.

    Returns:
        The elements (``"shapes/atera_nucleus"`` and so on) it re-created.

    Raises:
        NativeLatestError: If an element is missing from both stores.
    """
    import spatialdata as sd

    latest = Path(latest_path)
    source = Path(os.path.realpath(source_path))
    missing = [
        (src, dst)
        for src, dst in SOURCE_ELEMENT_RENAMES.items()
        if not _element_exists(latest, *dst)
    ]
    if not missing and _registry_up_to_date(
        dict(read_root_attrs(latest).get(MERXEN_SCHEMA_ATTR, {}))
    ):
        # The common case (reuse, a clean ENRICH): nothing to open or write.
        return []
    if missing:
        # Consolidated metadata naming deleted elements would make the reader
        # look for them; it is rebuilt below.
        _drop_consolidated_metadata(latest)
    sdata = sd.read_zarr(latest)
    recreated: list[str] = []
    if missing:
        for src_kind, src_name in (src for src, _ in missing):
            if not _element_exists(source, src_kind, src_name):
                raise NativeLatestError(
                    f"Cannot re-create {src_kind}/{src_name}: it is missing from "
                    "the source store"
                )
        source_sdata = sd.read_zarr(source, selection=("shapes", "tables"))
        recreated = _write_renamed_elements(
            sdata, source_sdata, [dst for _, dst in missing]
        )
    registry_changed = _register_native_branches(sdata, only_if_changed=True)
    if recreated or registry_changed:
        sdata.write_attrs()
        sdata.write_consolidated_metadata()
    if recreated:
        logger.info("Re-created native elements: %s", ", ".join(recreated))
    return recreated


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _check_source(source: Path, cfg: NativeLatestConfig) -> dict[str, Any]:
    """Return the build marker of a complete ATERA source store."""
    if not source.is_dir():
        raise NativeLatestError(
            f"[{cfg.dataset_name}] The source store does not exist or is not a "
            "directory"
        )
    marker = read_build_marker(source)
    if marker is None or not marker.get("build_fingerprint"):
        raise NativeLatestError(
            f"[{cfg.dataset_name}] The source store has no build marker "
            f"({BUILD_MARKER_NAME}); run BUILD_SPATIALDATA to completion first"
        )
    platform = str(marker.get("platform", "")).upper()
    if platform != platform_spec(cfg.platform).name:
        raise NativeLatestError(
            f"[{cfg.dataset_name}] The source store was built for platform "
            f"{platform or '?'}, not {cfg.platform}"
        )
    missing = [
        element
        for element in REQUIRED_SOURCE_ELEMENTS
        if not _element_exists(source, *element.split("/", 1))
    ]
    if missing:
        raise NativeLatestError(
            f"[{cfg.dataset_name}] The source store lacks {', '.join(missing)}"
        )
    return marker


def _check_target_is_not_the_source(source: Path, target: Path) -> None:
    """Refuse a target that is, contains or sits inside the source store."""
    resolved = Path(os.path.realpath(target))
    if resolved == source or source in resolved.parents or resolved in source.parents:
        raise NativeLatestError(
            "The latest-store target overlaps the source store; they must be "
            "separate directories"
        )


def _element_exists(store: Path, kind: str, name: str) -> bool:
    """Return whether an element group exists on disk (zarr v3 or v2)."""
    element = store / kind / name
    return (element / "zarr.json").is_file() or (element / ".zgroup").is_file()


def _read_native_marker(store: Path) -> dict[str, Any] | None:
    """Return the native latest marker of a store, or ``None``."""
    try:
        data = json.loads((store / NATIVE_LATEST_MARKER_NAME).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_reusable(marker: dict[str, Any], fingerprint: str, link_mode: str) -> bool:
    """Return whether a native marker matches the requested build."""
    return (
        marker.get("format_version") == NATIVE_LATEST_FORMAT_VERSION
        and marker.get("source_build_fingerprint") == fingerprint
        and marker.get("link_mode") == link_mode
    )


def _target_occupied(target: Path) -> bool:
    """Return whether the target path holds anything."""
    if target.is_symlink() or target.is_file():
        return True
    return target.is_dir() and any(target.iterdir())


def _remove_leftover_builds(target: Path, dataset_name: str) -> None:
    """Remove ``.partial-*`` and ``.stale-*`` folders of killed earlier runs.

    A SIGKILL (a Nextflow timeout, the OOM killer) skips the cleanup of
    :func:`build_native_latest`, and a kill between the two renames of
    :func:`_move_into_place` leaves the previous store as ``.stale-*``.
    """
    if not target.parent.is_dir():
        return
    leftovers = sorted(
        path
        for pattern in (f"{target.name}.partial-*", f"{target.name}.stale-*")
        for path in target.parent.glob(pattern)
    )
    if leftovers:
        logger.warning(
            "[%s] Removing %d leftover folder(s) of an interrupted NATIVE_LATEST run",
            dataset_name,
            len(leftovers),
        )
    for path in leftovers:
        remove_path(path)


def _harden_reused_payloads(
    target: Path,
    source: Path,
    marker: dict[str, Any],
    cfg: NativeLatestConfig,
) -> int:
    """Make the linked payloads of a reused store read-only when requested.

    A store built with ``readonly_payloads=false`` is reused as it is, so the
    write bits are removed here (and the marker updated) instead of rebuilding
    it, which would discard ENRICH's work. The reverse needs nothing: the read-
    only bit lives on the inode the source shares and is left in place.

    Returns:
        How many payload files lost their write bits.
    """
    if cfg.link_mode != "hardlink" or not cfg.readonly_payloads:
        return 0
    changed = 0
    for parts in _walk_files(target):
        if not _is_payload(parts):
            continue
        path = target.joinpath(*parts)
        source_path = source.joinpath(*parts)
        try:
            info, source_info = path.stat(), source_path.stat()
        except FileNotFoundError:
            continue
        if (info.st_dev, info.st_ino) != (source_info.st_dev, source_info.st_ino):
            continue
        if info.st_mode & _WRITE_BITS:
            _make_readonly(path)
            changed += 1
    if not marker.get("readonly_payloads"):
        updated = {**marker, "readonly_payloads": True}
        _replace_file_text(
            target / NATIVE_LATEST_MARKER_NAME, json.dumps(updated, indent=2) + "\n"
        )
        logger.info(
            "[%s] Reused store made read-only (%d payload files)",
            cfg.dataset_name,
            changed,
        )
    return changed


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _build_into(
    partial: Path,
    source: Path,
    target: Path,
    fingerprint: str,
    cfg: NativeLatestConfig,
) -> NativeLatestReport:
    """Build the native latest store in ``partial`` and return its report."""
    import spatialdata as sd

    tree = _copy_tree(
        source,
        partial,
        link_mode=cfg.link_mode,
        readonly_payloads=cfg.readonly_payloads,
    )
    # The copied root zarr.json carries the source's consolidated metadata,
    # which names the source elements; drop it until the new one is written.
    _drop_consolidated_metadata(partial)

    latest_sdata = sd.read_zarr(partial)
    source_sdata = sd.read_zarr(source, selection=("shapes", "tables"))
    written = _write_renamed_elements(
        latest_sdata, source_sdata, list(SOURCE_ELEMENT_RENAMES.values())
    )
    _register_native_branches(latest_sdata, only_if_changed=False)

    source_inodes = _inodes(source)
    _self_check(partial, source, tree, source_inodes)
    delta = _new_bytes(partial, source_inodes)
    provenance: dict[str, Any] = {
        "format_version": NATIVE_LATEST_FORMAT_VERSION,
        "platform": cfg.platform,
        "dataset_name": cfg.dataset_name,
        "source_build_fingerprint": fingerprint,
        "link_mode": cfg.link_mode,
        "readonly_payloads": bool(cfg.readonly_payloads),
        "files_linked": len(tree.linked),
        "bytes_linked": tree.bytes_linked,
        "files_copied": len(tree.copied),
        "bytes_copied": tree.bytes_copied,
        "elements_written": written,
        "apparent_size_delta_bytes": delta,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    latest_sdata.attrs[PROVENANCE_ATTR] = provenance
    latest_sdata.write_attrs()
    latest_sdata.write_consolidated_metadata()
    # The marker goes last: a store without it is never reused.
    marker_path = partial / NATIVE_LATEST_MARKER_NAME
    marker_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return NativeLatestReport(
        dataset_name=cfg.dataset_name,
        latest_path=target,
        reused=False,
        source_build_fingerprint=fingerprint,
        link_mode=cfg.link_mode,
        files_linked=len(tree.linked),
        bytes_linked=tree.bytes_linked,
        files_copied=len(tree.copied),
        bytes_copied=tree.bytes_copied,
        elements_written=written,
        apparent_size_delta_bytes=delta,
        extra={"readonly_payloads": bool(cfg.readonly_payloads)},
    )


def _skipped_prefixes() -> tuple[tuple[str, ...], ...]:
    """Return the source element paths written fresh instead of copied."""
    return tuple((kind, name) for kind, name in SOURCE_ELEMENT_RENAMES)


def _is_payload(parts: tuple[str, ...]) -> bool:
    """Return whether a store-relative file is an immutable payload."""
    name = parts[-1]
    if name in _METADATA_FILE_NAMES or len(parts) < 3:
        return False
    if parts[0] in _RASTER_ROOTS:
        return True
    return parts[0] == "points" and name.endswith(".parquet")


def _walk_files(root: Path) -> Iterator[tuple[str, ...]]:
    """Yield the store-relative parts of every file below ``root``, sorted.

    Raises:
        NativeLatestError: If a directory below ``root`` is a symlink. The walk
            does not follow it, so its files would be silently left out.
    """
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel_dir = Path(directory).relative_to(root).parts
        linked = [name for name in dirnames if os.path.islink(Path(directory, name))]
        if linked:
            shown = "/".join((*rel_dir, linked[0]))
            raise NativeLatestError(
                f"The store holds a symlinked directory ({shown}); NATIVE_LATEST "
                "needs real directories so that no element is left out"
            )
        for filename in sorted(filenames):
            yield (*rel_dir, filename)


def _copy_tree(
    source: Path,
    destination: Path,
    *,
    link_mode: str,
    readonly_payloads: bool,
) -> _TreeCopy:
    """Hardlink (or copy) payloads and copy metadata from source to destination.

    In ``copy`` mode every file is copied and nothing shares an inode with the
    source, so ``readonly_payloads`` has nothing to protect.
    """
    skipped = _skipped_prefixes()
    tree = _TreeCopy()
    destination.mkdir(parents=True)
    for parts in _walk_files(source):
        if len(parts) == 1 and parts[0] in _SKIPPED_ROOT_FILES:
            continue
        if any(parts[: len(prefix)] == prefix for prefix in skipped):
            continue
        src = Path(os.path.realpath(source.joinpath(*parts)))
        dst = destination.joinpath(*parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        rel = "/".join(parts)
        size = src.stat().st_size
        if _is_payload(parts) and link_mode == "hardlink":
            _hardlink(src, dst)
            if readonly_payloads:
                _make_readonly(dst)
            tree.linked.append(rel)
            tree.bytes_linked += size
        else:
            shutil.copy2(src, dst)
            tree.copied.append(rel)
            tree.bytes_copied += size
    return tree


def _hardlink(src: Path, dst: Path) -> None:
    """Hardlink one payload, explaining a cross-filesystem refusal."""
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise NativeLatestError(
                "The source store and the latest-store target are on different "
                "filesystems, so payloads cannot be hardlinked. Put the outdir on "
                "the source's filesystem or set native_latest_link_mode=copy."
            ) from exc
        if exc.errno in (errno.EPERM, errno.EACCES):
            raise NativeLatestError(
                f"Hardlinking a source payload was refused ({exc.strerror}); the "
                "source files must be owned by the user running NATIVE_LATEST, "
                "or set native_latest_link_mode=copy."
            ) from exc
        raise


def _make_readonly(path: Path) -> None:
    """Remove every write bit from a file (shared with the source inode)."""
    mode = path.stat().st_mode
    if mode & _WRITE_BITS:
        os.chmod(path, mode & ~_WRITE_BITS)


def _drop_consolidated_metadata(store: Path) -> None:
    """Remove the consolidated metadata of a store's root group, if any."""
    root = store / "zarr.json"
    if root.is_file():
        data = json.loads(root.read_text(encoding="utf-8"))
        if data.get("consolidated_metadata") is not None:
            data.pop("consolidated_metadata", None)
            _replace_file_text(root, json.dumps(data, indent=2))
        return
    zmetadata = store / ".zmetadata"
    if zmetadata.is_file():
        zmetadata.unlink()


def _replace_file_text(path: Path, text: str) -> None:
    """Write a file through a new inode (never truncating a shared one)."""
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _native_table(table: Any) -> Any:
    """Return the vendor table annotating the renamed cell shapes.

    The table is changed in place: it comes from a source read that is thrown
    away, and copying it would hold it twice in memory.
    """
    import pandas as pd

    result = table
    attrs = dict(result.uns.get("spatialdata_attrs", {}))
    region_key = str(attrs.get("region_key", "region"))
    result.obs[region_key] = pd.Categorical([NATIVE_CELL_SHAPES_KEY] * result.n_obs)
    attrs["region"] = NATIVE_CELL_SHAPES_KEY
    attrs.setdefault("region_key", region_key)
    result.uns["spatialdata_attrs"] = attrs
    return result


def _write_renamed_elements(
    latest_sdata: Any,
    source_sdata: Any,
    targets: Iterable[tuple[str, str]],
) -> list[str]:
    """Write the renamed vendor elements named in ``targets`` into the latest store.

    Shapes go first, so the table's annotation target exists when it is added.
    """
    reverse = {dst: src for src, dst in SOURCE_ELEMENT_RENAMES.items()}
    ordered = sorted(set(targets), key=lambda item: (item[0] != "shapes", item[1]))
    written: list[str] = []
    for kind, name in ordered:
        _, src_name = reverse[(kind, name)]
        if kind == "shapes":
            latest_sdata.shapes[name] = source_sdata.shapes[src_name]
        else:
            latest_sdata.tables[name] = _native_table(source_sdata.tables[src_name])
        latest_sdata.write_element(name)
        written.append(f"{kind}/{name}")
    return written


def _expected_entries() -> dict[str, dict[str, Any]]:
    """Return the registry fields the native branches must carry."""
    return {
        "original": {
            "shape": NATIVE_CELL_SHAPES_KEY,
            "table": NATIVE_TABLE_KEY,
            "assignment_column": ORIGINAL_ASSIGNMENT_COLUMN,
            "id_namespace": ORIGINAL_ID_NAMESPACE,
        },
        "original_nucleus": {
            "shape": NATIVE_NUCLEUS_SHAPES_KEY,
            "table": None,
            "assignment_column": None,
            "id_namespace": NUCLEUS_ID_NAMESPACE,
            "cell_key": NUCLEUS_CELL_KEY,
        },
    }


def _registry_up_to_date(schema: dict[str, Any]) -> bool:
    """Return whether a ``merxen_schema`` already carries the native entries."""
    registry = dict(schema.get("segmentations", {}))
    return (
        schema.get("segmentation_source") == "native"
        and schema.get("platform") == "ATERA"
        and all(
            isinstance(registry.get(branch), dict)
            and all(registry[branch].get(k) == v for k, v in fields.items())
            for branch, fields in _expected_entries().items()
        )
    )


def _register_native_branches(sdata: Any, *, only_if_changed: bool) -> bool:
    """Register ``original`` and ``original_nucleus``; return whether it did."""
    schema = dict(sdata.attrs.get(MERXEN_SCHEMA_ATTR, {}))
    if only_if_changed and _registry_up_to_date(schema):
        return False
    stamp_merxen_schema(sdata, primary_points_key=POINTS_KEY, platform="ATERA")
    register_segmentation_branch(
        sdata,
        "original",
        points_key=POINTS_KEY,
        assignment_column=ORIGINAL_ASSIGNMENT_COLUMN,
        shape_key=NATIVE_CELL_SHAPES_KEY,
        table_key=NATIVE_TABLE_KEY,
        id_namespace=ORIGINAL_ID_NAMESPACE,
        legacy_aliases=("original_seg",),
    )
    register_segmentation_branch(
        sdata,
        "original_nucleus",
        points_key=POINTS_KEY,
        assignment_column=None,
        shape_key=NATIVE_NUCLEUS_SHAPES_KEY,
        table_key=None,
        id_namespace=NUCLEUS_ID_NAMESPACE,
        cell_key=NUCLEUS_CELL_KEY,
    )
    schema = dict(sdata.attrs[MERXEN_SCHEMA_ATTR])
    schema["segmentation_source"] = "native"
    sdata.attrs[MERXEN_SCHEMA_ATTR] = schema
    return True


# --------------------------------------------------------------------------- #
# Self-check, size accounting and staging
# --------------------------------------------------------------------------- #


def _inodes(root: Path) -> dict[tuple[int, int], str]:
    """Map ``(st_dev, st_ino)`` of every file below ``root`` to its path."""
    found: dict[tuple[int, int], str] = {}
    for parts in _walk_files(root):
        info = root.joinpath(*parts).stat()
        found[(info.st_dev, info.st_ino)] = "/".join(parts)
    return found


def _self_check(
    store: Path,
    source: Path,
    tree: _TreeCopy,
    source_inodes: dict[tuple[int, int], str],
) -> None:
    """Check payload links share source inodes and nothing else does."""
    linked = set(tree.linked)
    problems: list[str] = []
    for parts in _walk_files(store):
        rel = "/".join(parts)
        info = (store / rel).stat()
        shared = (info.st_dev, info.st_ino) in source_inodes
        if rel in linked:
            source_info = (source / rel).stat()
            if (info.st_dev, info.st_ino) != (source_info.st_dev, source_info.st_ino):
                problems.append(f"{rel} does not share its source inode")
        elif shared:
            problems.append(f"{rel} shares an inode with the source store")
    if problems:
        shown = "; ".join(problems[:5])
        raise NativeLatestError(
            f"Native latest self-check failed ({len(problems)} files): {shown}"
        )


def _new_bytes(store: Path, source_inodes: dict[tuple[int, int], str]) -> int:
    """Return the apparent size of the files that do not share a source inode."""
    total = 0
    for parts in _walk_files(store):
        info = store.joinpath(*parts).stat()
        if (info.st_dev, info.st_ino) not in source_inodes:
            total += info.st_size
    return total


def _move_into_place(partial: Path, target: Path) -> None:
    """Atomically replace the target with the finished partial store."""
    if not target.exists() and not target.is_symlink():
        os.replace(partial, target)
        return
    if target.is_symlink() or target.is_file():
        target.unlink()
        os.replace(partial, target)
        return
    if not any(target.iterdir()):
        target.rmdir()
        os.replace(partial, target)
        return
    stale = target.with_name(f"{target.name}.stale-{uuid.uuid4().hex}")
    os.replace(target, stale)
    os.replace(partial, target)
    remove_path(stale)


def _stage_output(cfg: NativeLatestConfig, target: Path) -> None:
    """Expose the persistent store as ``<output_dir>/native_latest.zarr``."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stage_existing_output(target, cfg.output_dir / "native_latest.zarr")
