"""The Cellpose layer of an ATERA ``cellpose_proseg`` store (plan AT-7, ENRICH).

An ATERA ``cellpose_proseg`` row segments its morphology image with Cellpose
(DAPI + 18S, as on XENIUM) and refines the masks with tiled ProSeg. The
ProSeg layer (``reseg``) is written by ATERA_PROSEG_STITCH; this module writes
the Cellpose layer itself, the analysis layer ``proseg_mask``, with the
XENIUM names:

- ``shapes/MOSAIK_cellpose``: one polygon per Cellpose label, traced on the
  mask (contours through pixel centres, mapped to microns at
  ``(c + 0.5) ps``: the corner convention of the vendor masks, which share the
  grid) with the vendor shapes' ``Scale(1 / ps)`` transform;
- ``labels/cellpose_cell_labels``: the mask itself on the morphology grid
  (``Identity``, the vendor labels' pyramid), which the viewer cache and the
  image quantification read (an ATERA store's ``global`` frame is in pixels,
  so ATERA layers are quantified from labels, never rasterised);
- ``tables/table_MOSAIK_cellpose``: cells x genes counts of every gene
  transcript on the cell's pixels (the points column ``cellpose_assignment``
  the stitch wrote, the corner rule), ``var`` from ``table_original``, the
  per-cell control counts, ``cell_area``, the centroid and ``obsm["spatial"]``;
- the registry branch ``cellpose`` (aliases ``proseg_mask``, ``cellpose_mask``;
  namespace ``cellpose_label_v1``; assignment column ``cellpose_assignment``;
  the Cellpose model identity of the mask).

No spatial join runs: the counts come from the label lookup the stitch
recorded. The table ids, the shape ids and the mask labels must be the same
set, and every non-null ``cellpose_assignment`` a mask label.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from merxen.io.points_aggregates import points_parquet_dir, read_root_attrs
from merxen.io.spatialdata_schema import (
    MERXEN_SCHEMA_ATTR,
    PROSEG_ID_NAMESPACE,
    register_segmentation_branch,
)
from merxen.segmentation.atera_proseg import (
    CELLPOSE_ASSIGNMENT_COLUMN,
    STOCK_CELLPOSE_ASSIGNMENT_COLUMN,
)
from merxen.segmentation.atera_proseg.prior import file_stat, open_mask
from merxen.segmentation.atera_proseg.stitch import (
    CONTROL_COUNT_PREFIXES,
    OTHER_CONTROL_COLUMN,
    _control_category,
    derived_tables,
)

logger = logging.getLogger(__name__)

CELLPOSE_BRANCH: Final = "cellpose"
CELLPOSE_SHAPES_KEY: Final = "MOSAIK_cellpose"
CELLPOSE_TABLE_KEY: Final = "table_MOSAIK_cellpose"
CELLPOSE_LABEL_KEY: Final = "cellpose_cell_labels"
CELLPOSE_ALIASES: Final = ("proseg_mask", "cellpose_mask")
LAYER_ATTR: Final = "merxen_atera_cellpose_layer"
LAYER_VERSION: Final = 1
POINTS_KEY: Final = "transcripts"
ORIGINAL_TABLE_KEY: Final = "table_original"
VENDOR_LABEL_KEY: Final = "cell_labels"
LABEL_BLOCK_PX: Final = 4096
SOURCE_ID_PREFIX: Final = "cellpose"


def polygonise_label_npy(
    npy_path: Path,
    pixel_size_um: float,
    *,
    n_jobs: int,
    simplify_px: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace every label of a ``.npy`` mask as a polygon in microns.

    The AT-3 polygoniser of the vendor masks: a process pool maps the file
    (``forkserver``, no serial fallback), contours are simplified by
    ``simplify_px`` pixels and mapped to microns at ``(c + 0.5) ps``.

    Args:
        npy_path: The label mask (``.npy``, 2-D integer).
        pixel_size_um: Microns per pixel.
        n_jobs: Worker processes (and simplification threads).
        simplify_px: Tolerance in pixels; ``None`` uses the vendor default.

    Returns:
        ``(labels, geometries)`` sorted by label (uint64, object array).
    """
    from merxen.io.atera_reader import DEFAULT_SIMPLIFY_PX, _pixel_polygons_to_microns
    from merxen.segmentation.mask_geometry import masks_to_labeled_polygons

    mask = open_mask(npy_path)
    labeled = masks_to_labeled_polygons(
        mask, n_jobs=n_jobs, serial_fallback=False, mp_context="forkserver"
    )
    del mask
    labels = np.asarray([label for label, _ in labeled], dtype=np.uint64)
    order = np.argsort(labels, kind="stable")
    geometries = _pixel_polygons_to_microns(
        np.asarray([geometry for _, geometry in labeled], dtype=object)[order],
        pixel_size_um,
        DEFAULT_SIMPLIFY_PX if simplify_px is None else simplify_px,
        n_threads=n_jobs,
    )
    return labels[order], geometries


def _label_stats(mask: np.ndarray, n_labels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the pixel count and centroid sums of labels ``0..n_labels``."""
    counts = np.zeros(n_labels + 1, dtype=np.int64)
    sums = np.zeros((n_labels + 1, 2), dtype=np.float64)
    for start in range(0, mask.shape[0], 512):
        strip = np.asarray(mask[start : start + 512])
        rr, cc = np.nonzero(strip)
        values = strip[rr, cc].astype(np.int64)
        counts += np.bincount(values, minlength=n_labels + 1)[: n_labels + 1]
        sums[:, 0] += np.bincount(
            values, weights=cc.astype(np.float64), minlength=n_labels + 1
        )[: n_labels + 1]
        sums[:, 1] += np.bincount(
            values, weights=(rr + start).astype(np.float64), minlength=n_labels + 1
        )[: n_labels + 1]
    return counts, sums


def _count_file(
    path: Path,
    gene_position: dict[str, int],
    assignment_column: str = CELLPOSE_ASSIGNMENT_COLUMN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Count one points file's rows per (Cellpose cell, gene) and control class."""
    table = pq.read_table(path, columns=[assignment_column, "is_gene", "feature_name"])
    label_column = table.column(assignment_column).combine_chunks()
    valid = label_column.is_valid().to_numpy(zero_copy_only=False)
    labels = label_column.fill_null(0).to_numpy().astype(np.uint64)
    is_gene = table.column("is_gene").to_numpy(zero_copy_only=False).astype(bool)
    feature = table.column("feature_name").combine_chunks()
    if pa.types.is_dictionary(feature.type):
        names = [str(v) for v in feature.dictionary.to_pylist()]
        codes = feature.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    else:
        uniques, codes = np.unique(
            np.asarray(feature.to_pylist(), dtype=object), return_inverse=True
        )
        names = [str(v) for v in uniques]
    code_to_gene = np.asarray([gene_position.get(name, -1) for name in names], np.int64)
    genes = valid & is_gene
    gene = code_to_gene[codes[genes]]
    if (gene < 0).any():
        raise RuntimeError(f"{path.name}: a gene row is not in table_original")
    n_categories = len(CONTROL_COUNT_PREFIXES) + 1
    category_of_code = np.asarray([_control_category(name) for name in names], np.int64)
    controls = valid & ~is_gene
    return (
        labels[genes],
        gene,
        labels[controls],
        category_of_code[codes[controls]] % n_categories,
        int(table.num_rows),
        int(genes.sum()),
        int(controls.sum()),
    )


@dataclass
class _ReducedFile:
    keys: np.ndarray
    counts: np.ndarray
    control_rows: np.ndarray
    control_category: np.ndarray
    n_rows: int
    n_gene_rows: int
    n_control_rows: int
    n_unknown: int


def _reduce_file(
    path: Path,
    gene_position: dict[str, int],
    index_of: np.ndarray,
    n_genes: int,
    assignment_column: str = CELLPOSE_ASSIGNMENT_COLUMN,
) -> _ReducedFile:
    """Count one file per (table row, gene); labels not in the mask are counted."""
    cell, gene, ctrl_cell, ctrl_cat, rows, genes, controls = _count_file(
        path, gene_position, assignment_column
    )
    n_max = index_of.size - 1
    unknown = 0
    for values in (cell, ctrl_cell):
        outside = values > n_max
        unknown += int(outside.sum())
        unknown += int((index_of[values[~outside].astype(np.int64)] < 0).sum())
    empty = np.zeros(0, np.int64)
    if unknown:
        return _ReducedFile(empty, empty, empty, empty, rows, genes, controls, unknown)
    keys, counts = np.unique(
        index_of[cell.astype(np.int64)] * n_genes + gene, return_counts=True
    )
    return _ReducedFile(
        keys,
        counts,
        index_of[ctrl_cell.astype(np.int64)],
        ctrl_cat,
        rows,
        genes,
        controls,
        0,
    )


def layer_fingerprint(
    latest: Path, mask_path: Path, *, stock: bool = False
) -> dict[str, Any]:
    """Return what the layer was built from (mask identity, stitch fingerprint)."""
    stitch = dict(read_root_attrs(latest).get("merxen_atera_proseg") or {})
    comparison = (
        dict(stitch.get("stock_cellpose") or {})
        if stock
        else dict(stitch.get("prior") or {})
    )
    return {
        "version": LAYER_VERSION,
        "mask_file": Path(mask_path).name,
        "mask_stat": file_stat(Path(os.path.realpath(mask_path))),
        "prior_fingerprint": comparison.get("fingerprint"),
        "stitch_fingerprint": stitch.get("fingerprint"),
    }


def _layer_present(
    latest: Path,
    shape_key: str = CELLPOSE_SHAPES_KEY,
    table_key: str = CELLPOSE_TABLE_KEY,
    label_key: str = CELLPOSE_LABEL_KEY,
) -> bool:
    return all(
        (latest / kind / name / "zarr.json").is_file()
        or (latest / kind / name / ".zgroup").is_file()
        for kind, name in (
            ("shapes", shape_key),
            ("tables", table_key),
            ("labels", label_key),
        )
    )


def build_atera_cellpose_layer(
    latest: Path,
    mask_path: Path,
    *,
    pixel_size_um: float,
    dataset_name: str,
    n_jobs: int,
    model_identity: dict[str, Any] | None,
    force: bool = False,
    n_workers: int | None = None,
    stock: bool = False,
) -> dict[str, Any]:
    """Write (or keep) the Cellpose layer of an ATERA ``cellpose_proseg`` store.

    The caller holds the store's write lock.

    Args:
        latest: The latest store (stitched by ATERA_PROSEG_STITCH).
        mask_path: The Cellpose mask the stitch's prior was made from.
        pixel_size_um: Microns per pixel.
        dataset_name: Log prefix.
        n_jobs: Polygonisation workers.
        model_identity: The Cellpose model identity of the mask, or ``None``.
        force: Rebuild even when the store records the same inputs.
        n_workers: Threads over the points files.
        stock: Build the independent stock comparison; never a ProSeg prior.

    Returns:
        The layer record (also in ``attrs["merxen_atera_cellpose_layer"]``).

    Raises:
        RuntimeError: If the mask is not on the store's grid, the store lacks
            the stitched ``cellpose_assignment``, the stitch was made from
            another mask, or the ids do not agree.
    """
    import anndata as ad
    import dask.array as da
    import geopandas as gpd
    import shapely
    import spatialdata as sd
    from spatialdata.models import Labels2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity, Scale

    from merxen.io.spatialdata_io import delete_element, write_or_replace_element
    from merxen.segmentation.atera_proseg.plan import mask_shape

    branch = "stock_cellpose" if stock else CELLPOSE_BRANCH
    shape_key = "MOSAIK_stock_cellpose" if stock else CELLPOSE_SHAPES_KEY
    table_key = "table_MOSAIK_stock_cellpose" if stock else CELLPOSE_TABLE_KEY
    label_key = "stock_cellpose_cell_labels" if stock else CELLPOSE_LABEL_KEY
    assignment_column = (
        STOCK_CELLPOSE_ASSIGNMENT_COLUMN if stock else CELLPOSE_ASSIGNMENT_COLUMN
    )
    aliases = ("stock_cellpose",) if stock else CELLPOSE_ALIASES
    layer_attr = "merxen_atera_stock_cellpose_layer" if stock else LAYER_ATTR
    source_id_prefix = "stock_cellpose" if stock else SOURCE_ID_PREFIX
    latest = Path(latest)
    fingerprint = layer_fingerprint(latest, mask_path, stock=stock)
    recorded = dict(read_root_attrs(latest).get(layer_attr) or {})
    if (
        not force
        and recorded.get("fingerprint") == fingerprint
        and recorded.get("check", {}).get("passed")
        and _layer_present(latest, shape_key, table_key, label_key)
    ):
        logger.info("[%s] Cellpose layer already built; skipping", dataset_name)
        return recorded

    mask = open_mask(mask_path)
    grid = mask_shape(latest, VENDOR_LABEL_KEY)
    if tuple(int(v) for v in mask.shape) != grid:
        raise RuntimeError(
            f"[{dataset_name}] The Cellpose mask is {mask.shape}; the store's mask "
            f"grid is {grid}"
        )
    stitch = dict(read_root_attrs(latest).get("merxen_atera_proseg") or {})
    prior = dict(stitch.get("stock_cellpose" if stock else "prior") or {})
    if stitch.get("prior_kind") != "cellpose":
        raise RuntimeError(
            f"[{dataset_name}] The store's ProSeg layer was not stitched from a "
            "Cellpose prior; run the segment stage (ATERA_PROSEG_STITCH) first"
        )
    if stock and model_identity != prior.get("model_identity"):
        raise RuntimeError("Stock model identity differs from the stitched comparison")
    found_stat = file_stat(Path(os.path.realpath(mask_path)))
    if prior.get("mask_stat") is not None and list(prior["mask_stat"]) != found_stat:
        raise RuntimeError(
            f"[{dataset_name}] The ProSeg layer was stitched from another Cellpose "
            f"mask than {Path(mask_path).name} (Cellpose re-ran); re-run the segment "
            "stage"
        )
    directory = points_parquet_dir(latest, POINTS_KEY)
    files = sorted(directory.glob("*.parquet"))
    if not files or not all(
        assignment_column in pq.read_schema(path).names for path in files
    ):
        raise RuntimeError(
            f"[{dataset_name}] The points lack {assignment_column}; run the "
            "segment stage (ATERA_PROSEG_STITCH) first"
        )

    # Shapes: one polygon per label.
    labels, geometries = polygonise_label_npy(
        Path(mask_path), pixel_size_um, n_jobs=n_jobs
    )
    if labels.size == 0:
        raise RuntimeError(f"[{dataset_name}] The Cellpose mask holds no cell")
    n_max = int(labels.max())
    counts_px, sums = _label_stats(mask, n_max)
    present = np.nonzero(counts_px[1:])[0].astype(np.uint64) + np.uint64(1)
    if not np.array_equal(present, labels):
        raise RuntimeError(
            f"[{dataset_name}] Polygons and mask labels differ "
            f"({labels.size} polygons, {present.size} labels)"
        )
    index_of = np.full(n_max + 1, -1, dtype=np.int64)
    index_of[labels.astype(np.int64)] = np.arange(labels.size)
    centroid = (
        sums[labels.astype(np.int64)] / counts_px[labels.astype(np.int64), None] + 0.5
    ) * pixel_size_um

    # Counts from the stitched lookup.
    group = ad.read_zarr(latest / "tables" / ORIGINAL_TABLE_KEY)
    var = group.var.copy()
    gene_position = {str(name): i for i, name in enumerate(group.var_names)}
    del group
    n_genes = len(gene_position)
    n_categories = len(CONTROL_COUNT_PREFIXES) + 1
    workers = max(n_workers or int(os.environ.get("DASK_NUM_WORKERS", "4") or 4), 1)
    # Each worker reduces its file to (cell, gene) counts, so the peak holds a
    # few files' rows, not one key per gene row of the section (about 0.7 G on
    # a full section).
    file_keys: list[np.ndarray] = []
    file_counts: list[np.ndarray] = []
    control_matrix = np.zeros((labels.size, n_categories), dtype=np.int64)
    n_rows = n_gene_rows = n_control_rows = 0
    unknown = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for reduced in pool.map(
            lambda path: _reduce_file(
                path, gene_position, index_of, n_genes, assignment_column
            ),
            files,
        ):
            n_rows += reduced.n_rows
            n_gene_rows += reduced.n_gene_rows
            n_control_rows += reduced.n_control_rows
            unknown += reduced.n_unknown
            if reduced.n_unknown:
                continue  # refused below, once every file is read
            file_keys.append(reduced.keys)
            file_counts.append(reduced.counts)
            if reduced.control_rows.size:
                np.add.at(
                    control_matrix, (reduced.control_rows, reduced.control_category), 1
                )
    if unknown:
        raise RuntimeError(
            f"[{dataset_name}] {unknown} points carry a {assignment_column} "
            "that is not a label of the Cellpose mask"
        )
    keys = np.concatenate(file_keys) if file_keys else np.zeros(0, np.int64)
    key_counts = np.concatenate(file_counts) if file_counts else np.zeros(0, np.int64)
    # Duplicate keys (a cell and gene in several files) are summed in int64.
    matrix = (
        sparse.coo_matrix(
            (key_counts.astype(np.int64), (keys // n_genes, keys % n_genes)),
            shape=(labels.size, n_genes),
        )
        .tocsr()
        .astype(np.float32)
    )
    # Sums in float64: a float32 total loses integers past 2**24 (a crop already
    # holds 21 M gene rows on cells).
    transcript_counts = np.rint(
        np.asarray(matrix.sum(axis=1, dtype=np.float64)).ravel()
    ).astype(np.int64)
    n_counted = int(transcript_counts.sum())
    area = np.asarray(shapely.area(geometries), dtype=np.float64)
    source_ids = np.asarray([f"{source_id_prefix}_{int(v)}" for v in labels], object)
    obs = pd.DataFrame(
        {
            "instance_id": labels,
            "region": pd.Categorical([shape_key] * labels.size),
            "source_cell_id": source_ids,
            "cell_area": area,
            "mask_area_um2": counts_px[labels.astype(np.int64)] * pixel_size_um**2,
            "centroid_x": centroid[:, 0].astype(np.float32),
            "centroid_y": centroid[:, 1].astype(np.float32),
            "transcript_counts": transcript_counts,
        },
        index=pd.Index([str(int(v)) for v in labels], name="cell_id"),
    )
    for index, column in enumerate([*CONTROL_COUNT_PREFIXES, OTHER_CONTROL_COLUMN]):
        obs[column] = control_matrix[:, index]
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm["spatial"] = centroid.astype(np.float64)
    table = TableModel.parse(
        adata,
        region=shape_key,
        region_key="region",
        instance_key="instance_id",
    )
    gdf = gpd.GeoDataFrame(
        {"instance_id": labels, "source_cell_id": source_ids},
        geometry=list(geometries),
        index=pd.Index(labels, name="instance_id"),
    )
    shapes = ShapesModel.parse(
        gdf,
        transformations={
            "global": Scale([1 / pixel_size_um, 1 / pixel_size_um], axes=("x", "y"))
        },
    )
    levels = [
        path
        for path in (latest / "labels" / VENDOR_LABEL_KEY).iterdir()
        if path.is_dir()
    ]
    n_levels = max(len(levels), 1)
    label_element = Labels2DModel.parse(
        da.from_array(mask, chunks=(LABEL_BLOCK_PX, LABEL_BLOCK_PX)).astype(np.uint32),
        dims=("y", "x"),
        scale_factors=[2] * (n_levels - 1) if n_levels > 1 else None,
        transformations={"global": Identity()},
    )

    sdata = sd.read_zarr(latest, selection=("shapes", "tables", "labels"))
    # Tables computed from earlier Cellpose cells (image quantification,
    # clustering) would be reused against the new polygons: they go.
    stale = derived_tables(latest, shape_key, table_key)
    for key in stale:
        delete_element(sdata, key, "tables")
    write_or_replace_element(sdata, shape_key, "shapes", shapes)
    write_or_replace_element(sdata, label_key, "labels", label_element)
    write_or_replace_element(sdata, table_key, "tables", table)
    register_segmentation_branch(
        sdata,
        branch,
        points_key=POINTS_KEY,
        assignment_column=assignment_column,
        shape_key=shape_key,
        table_key=table_key,
        id_namespace="stock_cellpose_label_v1" if stock else PROSEG_ID_NAMESPACE,
        legacy_aliases=aliases,
        label_key=label_key,
    )
    schema = dict(sdata.attrs[MERXEN_SCHEMA_ATTR])
    registry = dict(schema["segmentations"])
    if model_identity is not None:
        registry[branch] = {
            **registry[branch],
            "model_identity": model_identity,
        }
        identities = dict(schema.get("model_identities") or {})
        identities[branch] = model_identity
        schema["model_identities"] = identities
    schema["segmentations"] = registry
    sdata.attrs[MERXEN_SCHEMA_ATTR] = schema
    check = {
        "passed": True,
        "n_cells": int(labels.size),
        "n_rows": int(n_rows),
        "n_gene_rows_on_cells": int(n_gene_rows),
        "n_control_rows_on_cells": int(n_control_rows),
        "n_zero_transcript_cells": int((transcript_counts == 0).sum()),
        "table_ids_equal_shape_ids_equal_mask_labels": True,
        "table_counts_equal_points": n_counted == int(n_gene_rows),
    }
    if not check["table_counts_equal_points"]:
        raise RuntimeError(
            f"[{dataset_name}] The Cellpose table holds {n_counted} counts "
            f"for {n_gene_rows} gene rows on cells"
        )
    record = {
        "version": LAYER_VERSION,
        "fingerprint": fingerprint,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pixel_size_um": float(pixel_size_um),
        "assignment_rule": "cellpose_mask_corner",
        "deleted_derived_tables": stale,
        "check": check,
    }
    sdata.attrs[layer_attr] = record
    sdata.write_attrs()
    sdata.write_consolidated_metadata()
    del sdata, mask
    logger.info(
        "[%s] Cellpose layer: %d cells, %d gene transcripts on cells (%d cells "
        "without any)",
        dataset_name,
        labels.size,
        n_gene_rows,
        check["n_zero_transcript_cells"],
    )
    result: dict[str, Any] = json.loads(json.dumps(record, default=str))
    return result
