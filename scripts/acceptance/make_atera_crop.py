#!/usr/bin/env python
"""Cut a self-consistent crop bundle out of an extracted Atera ROI (plan AT-5a).

The crop keeps a few whole transcript tiles and everything that belongs to
them, so a build of the crop passes every acceptance check of plan §1.1 the
same way the full section must:

- **Window.** The bounding rectangle of the selected tiles' nominal bounds,
  clipped at 0, in whole pixels that lie entirely inside it
  (``[ceil(x0 / ps), floor(x1 / ps))``). Every tile that overlaps the window
  must be selected. The mask and image arrays keep the vendor pixel origin
  (no offset is ever applied, so microns, pixels and the vendor
  ``homogeneous_transform`` keep their meaning); they are cut to the window's
  far edge and zero outside the window. Choose tiles at the origin for small
  arrays.
- **Cells.** A cell is kept when all its mask pixels lie in the window (its
  pixel count there equals ``cell_area / ps²``), its vendor matrix row equals
  the counts of its assigned QV >= 20 transcripts in the selected tiles on
  every column with transcripts in the full section (the vendor rule; this
  also drops cells with assigned transcripts outside the crop), and its
  largest nucleus lies whole in the window. Kept cells and their nuclei are
  renumbered in vendor order; the masks are relabelled to match.
- **Transcripts.** The selected tiles are copied byte for byte, except
  ``cell_id`` and ``overlaps_nucleus`` of rows assigned to a dropped cell,
  which become unassigned (the vendor sentinel and 0). ``grid_keys``,
  ``grid_number_objects``, ``gene_counts``, ``codeword_counts`` and
  ``number_rnas`` describe the crop.
- **Cells store and matrix.** Every per-cell array is cut to the kept cells
  (``cell_id``, ``cell_summary``, ``bboxes``, both ``polygon_sets``, the
  matrix ``X``, ``obs`` and ``obsm``); ``number_cells`` and
  ``matrix_properties.num_cells`` describe the crop. The Explorer
  ``gridded_polygon_sets`` level-of-detail structure is left out
  (``has_gridded_polygon_sets`` false) rather than kept stale.
- **Morphology.** Each ``morphology_2d`` channel is cut to the window and
  written as a tiled JPEG-2000 OME-TIFF with SubIFD levels (lossless, floor
  halving with 2 x 2 means), its OME-XML kept but for the image size and
  fresh file UUIDs (mapped consistently across the series).
- **Metadata.** ``experiment.spatial``, ``panel_config.json``, the metrics
  summary, the reports, the FOV files and ``diffexp_genes/`` are copied byte
  for byte: they describe the full section. The visualisation stores
  (binned transcripts, CSC matrix) and the 3D stack are not copied.

The crop is written to ``<out>.partial-<id>`` and renamed onto ``<out>``
once complete: ``<out>/outs/`` is the bundle and ``<out>/crop_manifest.json``
records the selection, the kept and dropped counts and the vendor matrix
excess the acceptance check should expect (``expected_vendor_excess``). The
manifest names no host path and no vendor manifest value.

Example::

    python scripts/acceptance/make_atera_crop.py \\
        --bundle $MERXEN_ATERA_RAW_ROOT/<dataset>/outs \\
        --out $MERXEN_ATERA_RAW_ROOT/<dataset>/crop_bundle \\
        --tiles="0,0;0,1;-1,1"

``--tiles`` takes the tile keys separated by ``;``; write it as
``--tiles=...`` so a negative key is not read as an option.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import uuid
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import anndata as ad
import numpy as np
import tifffile
import zarr
from scipy import sparse

from merxen.io import atera_reader as ar
from merxen.io.tenx_cell_ids import UNASSIGNED_CELL_ID_PART, pack_cell_ids

logger = logging.getLogger("atera_crop")

#: QV of the vendor matrix rule.
MATRIX_MIN_QV = 20.0
MANIFEST_NAME = "crop_manifest.json"
COMPLETE_MARKER = ".crop_complete"
_UUID_RE = re.compile(r"urn:uuid:[0-9a-fA-F-]{36}")


@dataclass(frozen=True)
class Window:
    """Pixel window ``[row0, row1) x [col0, col1)`` of the crop."""

    row0: int
    row1: int
    col0: int
    col1: int

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the cut arrays (they keep the vendor origin)."""
        return (self.row1, self.col1)

    def mask(self) -> tuple[slice, slice]:
        """Slices of the window inside the cut arrays."""
        return (slice(self.row0, self.row1), slice(self.col0, self.col1))


def crop_window(
    tiles: Sequence[ar.AteraTile],
    selected: Sequence[str],
    pixel_size_um: float,
    image_shape: tuple[int, int],
) -> Window:
    """Pixel window of the selected tiles; refuse a window with holes.

    Args:
        tiles: Every tile of the bundle.
        selected: Selected tile keys.
        pixel_size_um: Microns per pixel.
        image_shape: Level-0 ``(y, x)`` shape of the vendor image.

    Returns:
        The window, clipped to the image.

    Raises:
        ValueError: If a key is unknown, the window is empty, or a tile that
            overlaps the window is not selected.
    """
    by_key = {tile.key: tile for tile in tiles}
    unknown = sorted(set(selected) - set(by_key))
    if unknown:
        raise ValueError(f"unknown tiles: {unknown}")
    chosen = [by_key[key] for key in selected]
    x0 = max(0.0, min(tile.bounds_um[0] for tile in chosen))
    x1 = max(tile.bounds_um[1] for tile in chosen)
    y0 = max(0.0, min(tile.bounds_um[2] for tile in chosen))
    y1 = max(tile.bounds_um[3] for tile in chosen)
    window = Window(
        row0=math.ceil(y0 / pixel_size_um),
        row1=min(image_shape[0], math.floor(y1 / pixel_size_um)),
        col0=math.ceil(x0 / pixel_size_um),
        col1=min(image_shape[1], math.floor(x1 / pixel_size_um)),
    )
    if window.row1 <= window.row0 or window.col1 <= window.col0:
        raise ValueError("the selected tiles do not overlap the image")
    for tile in tiles:
        bx0, bx1, by0, by1 = tile.bounds_um
        overlaps = bx0 < x1 and bx1 > x0 and by0 < y1 and by1 > y0
        if overlaps and tile.n_rows and tile.key not in selected:
            raise ValueError(
                f"tile {tile.key} overlaps the crop window but is not selected"
            )
    return window


# --------------------------------------------------------------------------- #
# Zipped zarr v2 stores: verbatim members plus rewritten arrays
# --------------------------------------------------------------------------- #


def extract_members(zip_path: Path, out_dir: Path, keep: Callable[[str], bool]) -> int:
    """Copy the zip members ``keep`` accepts into ``out_dir``, byte for byte."""
    n = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not keep(info.filename):
                continue
            target = out_dir.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1 << 24)
            n += 1
    return n


def array_of_member(member: str) -> str:
    """Array (or group) path a zarr v2 member belongs to."""
    return str(PurePosixPath(member).parent)


def write_like(out_dir: Path, path: str, source: Any, data: np.ndarray) -> None:
    """Write ``data`` as a zarr v2 array with the encoding of ``source``.

    Chunks, order, compressor, filters, fill value and attributes are those
    of the vendor array (chunks clipped to the new shape).
    """
    group = zarr.open_group(zarr.storage.LocalStore(out_dir), mode="r+", zarr_format=2)
    meta = source.metadata
    chunks = tuple(
        max(1, min(int(c), int(n)))
        for c, n in zip(source.chunks, data.shape, strict=True)
    )
    array = group.create_array(
        path,
        shape=data.shape,
        chunks=chunks,
        dtype=source.dtype,
        order=meta.order,
        compressors=meta.compressor,
        filters=meta.filters,
        fill_value=meta.fill_value,
        attributes=source.attrs.asdict(),
        overwrite=True,
    )
    if data.size:
        array[...] = data


def update_attrs(out_dir: Path, path: str, changes: Mapping[str, Any]) -> None:
    """Change keys of a member's ``.zattrs`` JSON, keeping the others as they are."""
    target = out_dir.joinpath(*PurePosixPath(path).parts, ".zattrs")
    attrs = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    attrs.update(changes)
    target.write_text(json.dumps(attrs, indent=4, ensure_ascii=False), encoding="utf-8")


def zip_directory(source: Path, destination: Path) -> None:
    """Zip a zarr directory with every member STORED, like the vendor zips."""
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


# --------------------------------------------------------------------------- #
# Cell selection
# --------------------------------------------------------------------------- #


@dataclass
class TileCounts:
    """Assigned QV >= 20 transcripts of the selected tiles per cell x feature."""

    counts: sparse.csr_matrix
    assigned_low_qv: int
    gene_counts: np.ndarray
    codeword_counts: np.ndarray


def _tile_genes(gene_offset: np.ndarray, n_rows: int) -> np.ndarray:
    starts, ends = (
        gene_offset[:, 0].astype(np.int64),
        gene_offset[:, 1].astype(np.int64),
    )
    if ends.size and (
        starts[0] != 0
        or ends[-1] != n_rows
        or not np.array_equal(starts[1:], ends[:-1])
    ):
        raise ar.AteraFormatError("gene_offset ranges do not tile the rows")
    return np.repeat(np.arange(len(ends), dtype=np.int64), ends - starts)


def cell_rows_of(
    prefix: np.ndarray, suffix: np.ndarray, cell_keys: np.ndarray
) -> np.ndarray:
    """Cell row of each transcript (-1 unassigned); unknown IDs raise."""
    assigned = ~(
        (prefix == UNASSIGNED_CELL_ID_PART) & (suffix == UNASSIGNED_CELL_ID_PART)
    )
    rows = np.full(prefix.shape, -1, dtype=np.int64)
    if assigned.any():
        keys = pack_cell_ids(prefix[assigned], suffix[assigned]).astype(np.uint64)
        order = np.argsort(cell_keys)
        sorted_keys = cell_keys[order]
        position = np.searchsorted(sorted_keys, keys)
        position = np.minimum(position, len(sorted_keys) - 1)
        if not np.array_equal(sorted_keys[position], keys):
            raise ar.AteraFormatError("transcripts name cells absent from cells.zarr")
        rows[assigned] = order[position]
    return rows


def count_tiles(
    bundle: ar.AteraBundle,
    keys: Sequence[str],
    cell_keys: np.ndarray,
    n_features: int,
    n_codewords: int,
) -> TileCounts:
    """Count the selected tiles' transcripts per cell and per feature."""
    n_cells = len(cell_keys)
    total = sparse.csr_matrix((n_cells, n_features), dtype=np.int64)
    gene_counts = np.zeros(n_features, dtype=np.int64)
    codeword_counts = np.zeros(n_codewords, dtype=np.int64)
    low_qv = 0
    with ar.open_zip_group(bundle.transcripts_zip) as root:
        for key in keys:
            tile = root["grid"][key]
            n_rows = int(tile["location"].shape[0])
            if n_rows == 0:
                continue
            genes = _tile_genes(np.asarray(tile["gene_offset"][:]), n_rows)
            gene_counts += np.bincount(genes, minlength=n_features)
            codewords = np.asarray(tile["codeword_identity"][:]).ravel()
            codeword_counts += np.bincount(codewords, minlength=n_codewords)
            cell_id = np.asarray(tile["cell_id"][:])
            rows = cell_rows_of(cell_id[:, 0], cell_id[:, 1], cell_keys)
            qv = np.asarray(tile["quality_score"][:]).ravel()
            keep = (rows >= 0) & (qv >= MATRIX_MIN_QV)
            low_qv += int(np.count_nonzero((rows >= 0) & ~(qv >= MATRIX_MIN_QV)))
            total = (
                total
                + sparse.coo_matrix(
                    (
                        np.ones(int(keep.sum()), dtype=np.int64),
                        (rows[keep], genes[keep]),
                    ),
                    shape=(n_cells, n_features),
                ).tocsr()
            )
            logger.info("counted tile %s (%d rows)", key, n_rows)
    total.sum_duplicates()
    return TileCounts(total, low_qv, gene_counts, codeword_counts)


@dataclass
class Selection:
    """Kept cells and nuclei, with the relabelling tables."""

    kept_cells: np.ndarray
    kept_nuclei: np.ndarray
    cell_lut: np.ndarray
    nucleus_lut: np.ndarray
    reasons: dict[str, int]


def select_cells(
    cells: ar.AteraCells,
    window_cell_counts: np.ndarray,
    window_nucleus_counts: np.ndarray,
    matrix: sparse.csr_matrix,
    tile_counts: sparse.csr_matrix,
    columns_with_transcripts: np.ndarray,
    pixel_size_um: float,
) -> Selection:
    """Keep the cells whole in the window whose matrix row the crop reproduces."""
    n_cells = cells.n_cells
    pixel_area = pixel_size_um**2
    full_pixels = np.rint(cells.summary["cell_area"].to_numpy(np.float64) / pixel_area)
    in_window = window_cell_counts[1 : n_cells + 1]
    whole = (in_window > 0) & (in_window == full_pixels)
    diff = (matrix - tile_counts)[:, np.flatnonzero(columns_with_transcripts)]
    diff.eliminate_zeros()
    matrix_ok = np.diff(diff.tocsr().indptr) == 0
    # The largest nucleus must lie whole in the window too.
    nucleus_area = cells.summary["nucleus_area"].to_numpy(np.float64)
    owner = cells.nucleus_cell_index
    nucleus_pixels = window_nucleus_counts[1 : cells.n_nuclei + 1]
    largest = np.zeros(n_cells, dtype=np.int64)
    np.maximum.at(largest, owner, nucleus_pixels)
    has_nucleus = np.bincount(owner, minlength=n_cells) > 0
    nucleus_ok = ~has_nucleus | (largest == np.rint(nucleus_area / pixel_area))
    keep = whole & matrix_ok & nucleus_ok
    reasons = {
        "touching_the_window": int(np.count_nonzero(in_window > 0)),
        "kept": int(keep.sum()),
        "dropped_not_whole_in_window": int(np.count_nonzero((in_window > 0) & ~whole)),
        "dropped_matrix_row_not_reproduced": int(np.count_nonzero(whole & ~matrix_ok)),
        "dropped_nucleus_outside_window": int(
            np.count_nonzero(whole & matrix_ok & ~nucleus_ok)
        ),
    }
    kept_cells = np.flatnonzero(keep)
    kept_nuclei = np.flatnonzero(keep[owner])
    cell_lut = np.zeros(n_cells + 1, dtype=np.uint32)
    cell_lut[kept_cells + 1] = np.arange(1, len(kept_cells) + 1, dtype=np.uint32)
    nucleus_lut = np.zeros(cells.n_nuclei + 1, dtype=np.uint32)
    nucleus_lut[kept_nuclei + 1] = np.arange(1, len(kept_nuclei) + 1, dtype=np.uint32)
    return Selection(kept_cells, kept_nuclei, cell_lut, nucleus_lut, reasons)


def vendor_excess(
    matrix: sparse.csr_matrix, gene_counts: np.ndarray, feature_names: Sequence[str]
) -> dict[str, int]:
    """Matrix counts on columns without any transcript, by feature name.

    Points cannot reproduce them, so the acceptance check expects their total
    (``expected_vendor_excess``).

    Args:
        matrix: Kept rows of the vendor matrix.
        gene_counts: All-QV transcripts per column in the crop.
        feature_names: Column names.

    Returns:
        ``{feature name: count}`` for every such column with counts.
    """
    columns = np.flatnonzero(np.asarray(gene_counts) == 0)
    totals = np.asarray(matrix[:, columns].sum(axis=0, dtype=np.int64)).ravel()
    return {
        str(feature_names[column]): int(total)
        for column, total in zip(columns, totals, strict=True)
        if total
    }


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def _cut_mask(source: Any, window: Window) -> np.ndarray:
    out = np.zeros(window.shape, dtype=source.dtype)
    rows, cols = window.mask()
    out[rows, cols] = np.asarray(source[rows, cols])
    return out


def write_transcripts(
    bundle: ar.AteraBundle,
    keys: Sequence[str],
    counts: TileCounts,
    cell_keys: np.ndarray,
    dropped: np.ndarray,
    out_zip: Path,
    workdir: Path,
) -> dict[str, Any]:
    """Write the crop's transcripts zip; rows of dropped cells become unassigned."""
    selected = set(keys)

    def keep(member: str) -> bool:
        parts = PurePosixPath(member).parts
        return not (len(parts) >= 3 and parts[0] == "grid" and parts[1] not in selected)

    folder = workdir / "transcripts"
    extract_members(bundle.transcripts_zip, folder, keep)
    unassigned_rows: dict[str, int] = {}
    with ar.open_zip_group(bundle.transcripts_zip) as root:
        grid_attrs = root["grid"].attrs.asdict()
        for key in keys:
            tile = root["grid"][key]
            if int(tile["location"].shape[0]) == 0:
                continue
            cell_id = np.asarray(tile["cell_id"][:])
            rows = cell_rows_of(cell_id[:, 0], cell_id[:, 1], cell_keys)
            moved = (rows >= 0) & dropped[np.maximum(rows, 0)]
            unassigned_rows[key] = int(moved.sum())
            if not moved.any():
                continue
            cell_id[moved] = UNASSIGNED_CELL_ID_PART
            overlaps = np.asarray(tile["overlaps_nucleus"][:])
            overlaps[moved] = 0
            for name, data in (("cell_id", cell_id), ("overlaps_nucleus", overlaps)):
                for member in (folder / "grid" / key / name).iterdir():
                    member.unlink()
                write_like(folder, f"grid/{key}/{name}", tile[name], data)
    order = [key for key in grid_attrs["grid_keys"] if key in selected]
    numbers = dict(
        zip(grid_attrs["grid_keys"], grid_attrs["grid_number_objects"], strict=True)
    )
    update_attrs(
        folder,
        "grid",
        {
            "grid_keys": order,
            "grid_number_objects": [int(numbers[key]) for key in order],
            "gene_counts": counts.gene_counts.tolist(),
            "codeword_counts": counts.codeword_counts.tolist(),
        },
    )
    n_rows = int(sum(int(numbers[key]) for key in order))
    update_attrs(folder, ".", {"number_rnas": n_rows})
    zip_directory(folder, out_zip)
    shutil.rmtree(folder)
    return {
        "tiles": order,
        "rows_per_tile": {key: int(numbers[key]) for key in order},
        "n_rows": n_rows,
        "rows_unassigned_from_dropped_cells": unassigned_rows,
    }


def write_cells(
    bundle: ar.AteraBundle,
    selection: Selection,
    window: Window,
    out_zip: Path,
    workdir: Path,
) -> None:
    """Write the crop's cells zip: kept records, relabelled cut masks."""
    rewritten = {
        "cell_id",
        "cell_summary",
        "bboxes",
        "masks/0",
        "masks/1",
        *(
            f"polygon_sets/{s}/{name}"
            for s in "01"
            for name in ("cell_index", "method", "num_vertices", "vertices")
        ),
    }

    def keep(member: str) -> bool:
        if member.startswith("gridded_polygon_sets/"):
            return False
        return array_of_member(member) not in rewritten

    folder = workdir / "cells"
    extract_members(bundle.cells_zip, folder, keep)
    kept = selection.kept_cells
    nuclei = selection.kept_nuclei
    new_row = np.full(len(selection.cell_lut) - 1, -1, dtype=np.int64)
    new_row[kept] = np.arange(len(kept))
    with ar.open_zip_group(bundle.cells_zip) as root:
        write_like(
            folder, "cell_id", root["cell_id"], np.asarray(root["cell_id"][:])[kept]
        )
        write_like(
            folder,
            "cell_summary",
            root["cell_summary"],
            np.asarray(root["cell_summary"][:])[kept],
        )
        bboxes = np.asarray(root["bboxes"][:]).reshape(-1, 4)[kept].ravel()
        write_like(folder, "bboxes", root["bboxes"], bboxes)
        for set_key, rows in (("1", kept), ("0", nuclei)):
            for name in ("method", "num_vertices", "vertices"):
                source = root[f"polygon_sets/{set_key}/{name}"]
                write_like(
                    folder,
                    f"polygon_sets/{set_key}/{name}",
                    source,
                    np.asarray(source[:])[rows],
                )
            source = root[f"polygon_sets/{set_key}/cell_index"]
            owners = np.asarray(source[:]).astype(np.int64)[rows]
            write_like(
                folder,
                f"polygon_sets/{set_key}/cell_index",
                source,
                new_row[owners].astype(source.dtype),
            )
        for member, lut in (
            ("masks/1", selection.cell_lut),
            ("masks/0", selection.nucleus_lut),
        ):
            mask = _cut_mask(root[member], window)
            write_like(folder, member, root[member], lut[mask])
            del mask
    update_attrs(
        folder, ".", {"number_cells": int(len(kept)), "has_gridded_polygon_sets": False}
    )
    zip_directory(folder, out_zip)
    shutil.rmtree(folder)


def write_matrix(
    bundle: ar.AteraBundle, kept: np.ndarray, out_zip: Path, workdir: Path
) -> None:
    """Write the crop's cell-feature matrix: the kept rows of every per-cell array."""
    with ar.open_zip_group(bundle.cell_feature_matrix_zip) as root:
        n_cells = int(root["obs"][root["obs"].attrs["_index"]].shape[0])
        for name in ("layers", "obsp"):
            if name in root and len(list(root[name].keys())):
                raise ar.AteraFormatError(f"the matrix has {name}; not supported")
        per_cell: list[str] = []

        def visit(group: Any, prefix: str) -> None:
            for name, member in group.members():
                path = f"{prefix}{name}"
                if isinstance(member, zarr.Group):
                    visit(member, f"{path}/")
                elif (
                    path.startswith(("obs/", "obsm/"))
                    and member.shape
                    and int(member.shape[0]) == n_cells
                    and PurePosixPath(path).name not in {"categories", "color"}
                ):
                    per_cell.append(path)

        visit(root, "")
        rewritten = {*per_cell, "X/data", "X/indices", "X/indptr"}
        folder = workdir / "cell_feature_matrix"
        extract_members(
            bundle.cell_feature_matrix_zip,
            folder,
            lambda member: array_of_member(member) not in rewritten,
        )
        for path in per_cell:
            write_like(folder, path, root[path], np.asarray(root[path][:])[kept])
        matrix = sparse.csr_matrix(ad.io.read_elem(root["X"]))[kept]
        matrix.sort_indices()
        for name in ("data", "indices", "indptr"):
            source = root[f"X/{name}"]
            write_like(
                folder, f"X/{name}", source, getattr(matrix, name).astype(source.dtype)
            )
        properties = dict(root.attrs.get("matrix_properties") or {})
    update_attrs(folder, "X", {"shape": [int(matrix.shape[0]), int(matrix.shape[1])]})
    if properties:
        properties["num_cells"] = int(len(kept))
        update_attrs(folder, ".", {"matrix_properties": properties})
    zip_directory(folder, out_zip)
    shutil.rmtree(folder)


def _new_uuids(xmls: Iterable[str]) -> dict[str, str]:
    found = sorted({match for xml in xmls for match in _UUID_RE.findall(xml)})
    return {old: f"urn:uuid:{uuid.uuid4()}" for old in found}


def _resized_ome(xml: str, shape: tuple[int, int], uuids: Mapping[str, str]) -> str:
    def pixels(match: re.Match[str]) -> str:
        tag = re.sub(r'SizeX="\d+"', f'SizeX="{shape[1]}"', match.group(0), count=1)
        return re.sub(r'SizeY="\d+"', f'SizeY="{shape[0]}"', tag, count=1)

    xml = re.sub(r"<Pixels\b[^>]*>", pixels, xml, count=1)
    return _UUID_RE.sub(lambda match: uuids[match.group(0)], xml)


def write_morphology(
    bundle: ar.AteraBundle, window: Window, out_dir: Path, *, n_workers: int
) -> list[dict[str, Any]]:
    """Cut every ``morphology_2d`` channel to the window (lossless JPEG-2000)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xmls = {}
    for path in bundle.morphology_2d_files:
        with tifffile.TiffFile(path, _multifile=False) as tif:
            xmls[path] = tif.ome_metadata or ""
    uuids = _new_uuids(xmls.values())
    records = []
    for path in bundle.morphology_2d_files:
        with tifffile.TiffFile(path, _multifile=False) as tif:
            series = tif.series[0]
            n_levels = len(series.levels)
            page = series.keyframe
            tile = (
                (int(page.tilelength), int(page.tilewidth)) if page.is_tiled else None
            )
            compression = page.compression
            store = series.aszarr(level=0)
            try:
                level0 = _cut_mask(zarr.open_array(store, mode="r"), window)
            finally:
                store.close()
        levels = [level0]
        while len(levels) < n_levels and min(levels[-1].shape) >= 2:
            previous = levels[-1]
            height, width = previous.shape[0] // 2, previous.shape[1] // 2
            block = previous[: 2 * height, : 2 * width].astype(np.float32)
            levels.append(
                np.rint(block.reshape(height, 2, width, 2).mean(axis=(1, 3))).astype(
                    previous.dtype
                )
            )
        options: dict[str, Any] = {
            "compression": compression,
            "photometric": "minisblack",
            "metadata": None,
            "maxworkers": n_workers,
        }
        if tile is not None:
            options["tile"] = tile
        target = out_dir / path.name
        description = _resized_ome(xmls[path], window.shape, uuids)
        with tifffile.TiffWriter(target, bigtiff=True) as writer:
            writer.write(
                levels[0],
                subifds=len(levels) - 1,
                description=description.encode("utf-8"),
                **options,
            )
            for level in levels[1:]:
                writer.write(level, subfiletype=1, **options)
        with tifffile.TiffFile(target, _multifile=False) as tif:
            store = tif.series[0].aszarr(level=0)
            try:
                written = np.asarray(zarr.open_array(store, mode="r")[:])
            finally:
                store.close()
        if not np.array_equal(written, level0):
            raise RuntimeError(f"{path.name}: the written level 0 is not lossless")
        records.append(
            {
                "file": path.name,
                "levels": [list(level.shape) for level in levels],
            }
        )
        logger.info("wrote %s (%d levels)", path.name, len(levels))
    return records


def copy_metadata_files(bundle: ar.AteraBundle, out_dir: Path) -> list[str]:
    """Copy every vendor metadata and report file byte for byte."""
    copied = []
    for sidecar in bundle.sidecars:
        if sidecar.stored_path == ar.EXPERIMENT_OVERRIDE_SIDECAR:
            continue
        target = out_dir / sidecar.original_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sidecar.source, target)
        copied.append(sidecar.stored_path)
    return copied


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def make_crop(
    bundle_root: Path,
    out: Path,
    tiles: Sequence[str],
    *,
    n_workers: int = 8,
    force: bool = False,
) -> dict[str, Any]:
    """Write a crop bundle of ``tiles`` to ``out`` and return its manifest.

    Args:
        bundle_root: The extracted ROI folder.
        out: Crop folder (``out/outs`` is the bundle).
        tiles: Transcript tile keys to keep.
        n_workers: Threads for the image encoding.
        force: Replace an existing crop.

    Returns:
        The manifest written to ``out/crop_manifest.json``.
    """
    started = time.monotonic()
    out = Path(out)
    if out.exists() and not force:
        raise FileExistsError("the crop folder exists; pass --force to replace it")
    bundle = ar.resolve_atera_bundle(Path(bundle_root))
    all_tiles = ar.read_atera_tile_index(bundle)
    keys = [tile.key for tile in all_tiles if tile.key in set(tiles)]
    window = crop_window(
        all_tiles, keys, bundle.pixel_size_um, bundle.morphology_shape_yx
    )
    logger.info(
        "window rows %d-%d, cols %d-%d",
        window.row0,
        window.row1,
        window.col0,
        window.col1,
    )
    cells = ar.read_atera_cells(bundle)
    cell_keys = pack_cell_ids(cells.cell_id_prefix, cells.dataset_suffix).astype(
        np.uint64
    )
    features = ar.read_atera_features(bundle)
    with ar.open_zip_group(bundle.transcripts_zip) as root:
        n_codewords = int(root.attrs["codeword_count"])
        full_gene_counts = np.asarray(root["grid"].attrs["gene_counts"], dtype=np.int64)
    counts = count_tiles(bundle, keys, cell_keys, len(features), n_codewords)
    with ar.open_zip_group(bundle.cells_zip) as root:
        cell_window = _cut_mask(root["masks/1"], window)
        window_cell_counts = np.bincount(
            cell_window.ravel(), minlength=cells.n_cells + 1
        )
        del cell_window
        nucleus_window = _cut_mask(root["masks/0"], window)
        window_nucleus_counts = np.bincount(
            nucleus_window.ravel(), minlength=cells.n_nuclei + 1
        )
        del nucleus_window
    with ar.open_zip_group(bundle.cell_feature_matrix_zip) as root:
        matrix = sparse.csr_matrix(ad.io.read_elem(root["X"]))
    matrix.data = matrix.data.astype(np.int64)
    selection = select_cells(
        cells,
        window_cell_counts,
        window_nucleus_counts,
        matrix,
        counts.counts,
        full_gene_counts > 0,
        bundle.pixel_size_um,
    )
    logger.info("cells: %s", selection.reasons)
    kept_matrix = matrix[selection.kept_cells]
    excess_by_column = vendor_excess(
        kept_matrix, counts.gene_counts, features["feature_name"].astype(str).tolist()
    )
    excess = int(sum(excess_by_column.values()))
    del matrix

    partial = out.with_name(f"{out.name}.partial-{uuid.uuid4().hex}")
    outs = partial / "outs"
    work = partial / "work"
    try:
        outs.mkdir(parents=True)
        work.mkdir()
        dropped = np.ones(cells.n_cells, dtype=bool)
        dropped[selection.kept_cells] = False
        transcripts = write_transcripts(
            bundle,
            keys,
            counts,
            cell_keys,
            dropped,
            outs / bundle.transcripts_zip.name,
            work,
        )
        write_cells(bundle, selection, window, outs / bundle.cells_zip.name, work)
        write_matrix(
            bundle,
            selection.kept_cells,
            outs / bundle.cell_feature_matrix_zip.name,
            work,
        )
        morphology_dir = bundle.morphology_2d_files[0].parent
        morphology = write_morphology(
            bundle,
            window,
            outs / morphology_dir.relative_to(bundle.root),
            n_workers=n_workers,
        )
        metadata = copy_metadata_files(bundle, outs)
        shutil.rmtree(work)
        manifest = {
            "format_version": 1,
            "source_tiles": [tile.key for tile in all_tiles],
            "tiles": keys,
            "window_px": {
                "rows": [window.row0, window.row1],
                "cols": [window.col0, window.col1],
            },
            "window_um": {
                "x": [
                    window.col0 * bundle.pixel_size_um,
                    window.col1 * bundle.pixel_size_um,
                ],
                "y": [
                    window.row0 * bundle.pixel_size_um,
                    window.row1 * bundle.pixel_size_um,
                ],
            },
            "image_shape_yx": list(window.shape),
            "pixel_size_um": bundle.pixel_size_um,
            "cells": {
                **selection.reasons,
                "n_source_cells": cells.n_cells,
                "n_nuclei_kept": int(len(selection.kept_nuclei)),
            },
            "transcripts": transcripts,
            "assigned_rows_below_qv20": counts.assigned_low_qv,
            "matrix": {
                "n_cells": int(kept_matrix.shape[0]),
                "total": int(kept_matrix.sum(dtype=np.int64)),
                "gene_total": int(
                    kept_matrix[
                        :,
                        np.flatnonzero(
                            features["feature_type"].to_numpy()
                            == ar.GENE_EXPRESSION_TYPE
                        ),
                    ].sum(dtype=np.int64)
                ),
                "expected_vendor_excess": excess,
                "excess_by_column": excess_by_column,
            },
            "morphology_2d": morphology,
            "metadata_files": metadata,
            "not_copied": [
                "binned_transcripts.zarr.zip",
                "csc_cell_feature_matrix.zarr.zip",
                "morphology_3d",
                "cells.zarr.zip gridded_polygon_sets",
            ],
            "elapsed_s": round(time.monotonic() - started, 1),
        }
        (partial / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
        )
        check_crop(outs, manifest)
        (partial / COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        os.replace(partial, out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest


def check_crop(outs: Path, manifest: Mapping[str, Any]) -> None:
    """Re-open the crop with the readers and check its counts."""
    bundle = ar.resolve_atera_bundle(outs)
    cells = ar.read_atera_cells(bundle)
    if cells.n_cells != manifest["cells"]["kept"]:
        raise RuntimeError("the crop's cell count differs from the selection")
    if tuple(bundle.morphology_shape_yx) != tuple(manifest["image_shape_yx"]):
        raise RuntimeError("the crop image shape differs from the window")
    if cells.mask_shape != tuple(manifest["image_shape_yx"]):
        raise RuntimeError("the crop mask shape differs from the window")
    tiles = ar.read_atera_tile_index(bundle)
    if [tile.key for tile in tiles] != manifest["tiles"]:
        raise RuntimeError("the crop's tiles differ from the selection")
    table = ar.read_atera_table(bundle, cells, keep_vendor_annotations=False)
    if table.n_obs != cells.n_cells:
        raise RuntimeError("the crop's matrix rows differ from its cells")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tiles", required=True, help='Tile keys separated by ";", e.g. "0,0;0,1"'
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logger.setLevel(logging.INFO)
    manifest = make_crop(
        args.bundle,
        args.out,
        [key.strip() for key in args.tiles.split(";") if key.strip()],
        n_workers=args.workers,
        force=args.force,
    )
    logger.info(
        "crop: %d tiles, %d transcripts, %d cells, expected vendor excess %d (%.0f s)",
        len(manifest["tiles"]),
        manifest["transcripts"]["n_rows"],
        manifest["cells"]["kept"],
        manifest["matrix"]["expected_vendor_excess"],
        manifest["elapsed_s"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
