"""ATERA_PROSEG_TILE: run ProSeg on one tile and reduce it to its owned cells.

The task exports the window (:mod:`.export`), runs ProSeg (:mod:`.runner`) and
keeps only what the stitch needs (``tile_<id>/``):

- ``cells.parquet``: one row per owned ProSeg cell (its prior label is in the
  tile's ``owned_labels``): ``instance_id`` (the prior label: vendor or
  Cellpose),
  ``proseg_internal_id``, the centroid, volume, surface area, scale, ProSeg
  cluster, the foreground count and the consensus polygon (WKB, microns);
- ``claims/<source file>.parquet``: every exported row ProSeg assigned to an
  owned cell (``transcript_id``, ``instance_id``, ``background``), split by the
  store's points file;
- ``tile_manifest.json``: identity (window, source fingerprint, ProSeg version
  and command), row and cell counts, the prior weight (requested, multiplier,
  reported, and the run restarted with a rescaled multiplier, if any), timings,
  peak memory, the interior-edge check and the SHA-256 of the result files.

Before writing, the per-cell, per-gene foreground counts recounted from the
transcript metadata must equal ProSeg's count matrix for every owned cell.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

from merxen.config import AteraProsegTileConfig
from merxen.segmentation.atera_proseg import (
    TILE_CELLS_NAME,
    TILE_CLAIMS_DIR,
    TILE_FORMAT_VERSION,
    TILE_MANIFEST_NAME,
    AteraProsegError,
    PriorWeightMismatchError,
)
from merxen.segmentation.atera_proseg.export import (
    export_tile,
    read_exported_ids,
)
from merxen.segmentation.atera_proseg.runner import (
    OUTPUT_CELL_METADATA,
    OUTPUT_CELL_POLYGONS,
    OUTPUT_SPATIALDATA,
    OUTPUT_TRANSCRIPT_METADATA,
    build_command,
    check_prior_weight,
    check_version,
    prior_weight_multiplier,
    run_proseg,
)

logger = logging.getLogger(__name__)

_UNASSIGNED = np.uint32(np.iinfo(np.uint32).max)
CLAIM_SCHEMA = pa.schema(
    [
        ("transcript_id", pa.uint64()),
        ("instance_id", pa.uint32()),
        ("background", pa.bool_()),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_polygons(path: Path) -> dict[int, Any]:
    """Return ``ProSeg cell -> shapely geometry`` of the consensus polygons."""
    from shapely.geometry import shape

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    polygons: dict[int, Any] = {}
    for feature in data.get("features", []):
        cell = int(feature["properties"]["cell"])
        geometry = feature.get("geometry")
        polygons[cell] = None if geometry is None else shape(geometry)
    return polygons


def _count_matrix(proseg_dir: Path) -> tuple[np.ndarray, list[str], sparse.csr_matrix]:
    """Return ProSeg's table: cell index per row, gene names and counts."""
    import anndata as ad

    table = ad.read_zarr(proseg_dir / OUTPUT_SPATIALDATA / "tables" / "table")
    cells = table.obs["cell"].to_numpy(dtype=np.int64)
    genes = [str(name) for name in table.var_names]
    matrix = sparse.csr_matrix(table.X).astype(np.int64)
    return cells, genes, matrix


def interior_edges(window_px: list[int], mask_shape_px: list[int]) -> dict[str, bool]:
    """Return which window edges are inside the mask (not on its border)."""
    col0, col1, row0, row1 = (int(v) for v in window_px)
    n_rows, n_cols = (int(v) for v in mask_shape_px)
    return {
        "left": col0 > 0,
        "right": col1 < n_cols,
        "top": row0 > 0,
        "bottom": row1 < n_rows,
    }


def cells_near_interior_edges(
    bounds: np.ndarray,
    ids: np.ndarray,
    window_um: tuple[float, float, float, float],
    interior: dict[str, bool],
    tolerance_um: float,
) -> list[int]:
    """Return the ids whose polygon comes within ``tolerance_um`` of an interior edge.

    Args:
        bounds: ``[n, 4]`` polygon bounds (x0, y0, x1, y1).
        ids: Cell ids aligned with ``bounds``.
        window_um: The window.
        interior: :func:`interior_edges`.
        tolerance_um: Distance in microns.

    Returns:
        Sorted ids.
    """
    if bounds.size == 0:
        return []
    x0, y0, x1, y1 = window_um
    near = np.zeros(len(ids), dtype=bool)
    if interior["left"]:
        near |= bounds[:, 0] <= x0 + tolerance_um
    if interior["top"]:
        near |= bounds[:, 1] <= y0 + tolerance_um
    if interior["right"]:
        near |= bounds[:, 2] >= x1 - tolerance_um
    if interior["bottom"]:
        near |= bounds[:, 3] >= y1 - tolerance_um
    return sorted(int(v) for v in ids[near])


def reduce_tile(
    cfg: AteraProsegTileConfig,
    proseg_dir: Path,
    exported_dir: Path,
    window_labels: np.ndarray,
) -> dict[str, Any]:
    """Write ``cells.parquet`` and ``claims/`` of the owned cells.

    Args:
        cfg: The tile configuration.
        proseg_dir: ProSeg's ``--output-path``.
        exported_dir: The exported-id side files.
        window_labels: Non-zero labels of the window's prior.

    Returns:
        Row and cell statistics for the manifest.

    Raises:
        AteraProsegError: If ProSeg's cells do not map one to one onto window
            labels, a claim matches no exported row, or the recount differs
            from ProSeg's count matrix.
    """
    import shapely

    out = Path(cfg.output_dir)
    meta = pq.read_table(proseg_dir / OUTPUT_CELL_METADATA)
    cell_index = meta.column("cell").to_numpy().astype(np.int64)
    raw_ids = meta.column("original_cell_id").to_pylist()
    try:
        labels = np.asarray([int(value) for value in raw_ids], dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise AteraProsegError(
            f"ProSeg original_cell_id is not a prior label: {error}"
        ) from error
    if labels.size and (labels <= 0).any():
        raise AteraProsegError("A ProSeg cell has no positive prior label")
    if np.unique(labels).size != labels.size:
        raise AteraProsegError("Two ProSeg cells share a prior label")
    outside = np.setdiff1d(labels, window_labels)
    if outside.size:
        raise AteraProsegError(
            f"{outside.size} ProSeg cells carry labels that are not in the window "
            f"(e.g. {int(outside[0])})"
        )
    owned_set = np.asarray(sorted(cfg.owned_labels), dtype=np.int64)
    polygons = _read_polygons(proseg_dir / OUTPUT_CELL_POLYGONS)
    is_owned = np.isin(labels, owned_set)
    has_polygon = np.asarray(
        [
            polygons.get(int(cell)) is not None and not polygons[int(cell)].is_empty
            for cell in cell_index
        ],
        dtype=bool,
    )
    without_polygon = sorted(int(v) for v in labels[is_owned & ~has_polygon])
    keep = is_owned & has_polygon
    n_lookup = int(cell_index.max()) + 1 if cell_index.size else 0
    owner_lut = np.zeros(n_lookup, dtype=np.uint32)
    owner_lut[cell_index[keep]] = labels[keep].astype(np.uint32)

    transcripts = pq.read_table(
        proseg_dir / OUTPUT_TRANSCRIPT_METADATA,
        columns=["transcript_id", "assignment", "background", "gene"],
    )
    n_kept = transcripts.num_rows
    assignment = (
        pc.fill_null(transcripts.column("assignment"), _UNASSIGNED)
        .to_numpy()
        .astype(np.uint32)
    )
    background = transcripts.column("background").to_numpy().astype(bool)
    transcript_id = transcripts.column("transcript_id").to_numpy().astype(np.uint64)
    assigned = assignment != _UNASSIGNED
    owner = np.zeros(n_kept, dtype=np.uint32)
    in_lut = assigned & (assignment < n_lookup)
    owner[in_lut] = owner_lut[assignment[in_lut]]
    claimed = owner > 0
    foreground = assigned & ~background

    # Recount: owned foreground rows per (cell, gene) against ProSeg's matrix.
    matrix_cells, matrix_genes, matrix = _count_matrix(proseg_dir)
    gene_column = transcripts.column("gene").combine_chunks()
    if pa.types.is_dictionary(gene_column.type):
        dictionary = [str(v) for v in gene_column.dictionary.to_pylist()]
        codes = gene_column.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    else:
        dictionary, codes = np.unique(
            np.asarray(gene_column.to_pylist(), dtype=object), return_inverse=True
        )
        dictionary = [str(v) for v in dictionary]
    gene_position = {name: i for i, name in enumerate(matrix_genes)}
    missing_genes = [name for name in dictionary if name not in gene_position]
    if missing_genes:
        raise AteraProsegError(
            f"{len(missing_genes)} genes of the transcript metadata are not in "
            "ProSeg's count matrix"
        )
    code_to_column = np.asarray([gene_position[name] for name in dictionary], np.int64)
    row_of_cell = np.full(n_lookup, -1, dtype=np.int64)
    row_of_cell[matrix_cells[matrix_cells < n_lookup]] = np.nonzero(
        matrix_cells < n_lookup
    )[0]
    counted = claimed & foreground
    recount = sparse.coo_matrix(
        (
            np.ones(int(counted.sum()), dtype=np.int64),
            (
                row_of_cell[assignment[counted]],
                code_to_column[codes[counted]],
            ),
        ),
        shape=matrix.shape,
    ).tocsr()
    kept_rows = row_of_cell[cell_index[keep]]
    if (kept_rows < 0).any():
        raise AteraProsegError("An owned ProSeg cell has no row in the count matrix")
    difference = (recount[kept_rows] - matrix[kept_rows]).tocsr()
    difference.eliminate_zeros()
    if difference.nnz:
        raise AteraProsegError(
            f"Recounted foreground transcripts differ from ProSeg's count matrix "
            f"in {difference.nnz} (cell, gene) entries of the owned cells"
        )
    foreground_per_cell = np.asarray(matrix[kept_rows].sum(axis=1)).ravel()

    # Claims, split by source file through the exported ids.
    exported = read_exported_ids(exported_dir)
    n_exported = sum(int(ids.size) for ids in exported.values())
    if n_kept > n_exported:
        raise AteraProsegError(
            f"ProSeg returned {n_kept} transcripts for {n_exported} exported rows"
        )
    claims_dir = out / TILE_CLAIMS_DIR
    if claims_dir.exists():
        shutil.rmtree(claims_dir)
    claims_dir.mkdir(parents=True)
    claim_ids = transcript_id[claimed]
    claim_owner = owner[claimed]
    claim_background = background[claimed]
    order = np.argsort(claim_ids, kind="stable")
    claim_ids, claim_owner, claim_background = (
        claim_ids[order],
        claim_owner[order],
        claim_background[order],
    )
    matched = np.zeros(claim_ids.size, dtype=bool)
    claim_files: dict[str, int] = {}
    for name, ids in exported.items():
        if not ids.size or not claim_ids.size:
            continue
        position = np.searchsorted(ids, claim_ids)
        hit = position < ids.size
        hit[hit] = ids[position[hit]] == claim_ids[hit]
        if matched[hit].any():
            raise AteraProsegError("A transcript id was exported from two files")
        matched |= hit
        if not hit.any():
            continue
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array(claim_ids[hit], pa.uint64()),
                    pa.array(claim_owner[hit], pa.uint32()),
                    pa.array(claim_background[hit], pa.bool_()),
                ],
                schema=CLAIM_SCHEMA,
            ),
            claims_dir / f"{Path(name).stem}.parquet",
            compression="zstd",
        )
        claim_files[name] = int(hit.sum())
    if not matched.all():
        raise AteraProsegError(
            f"{int((~matched).sum())} claimed transcripts match no exported row"
        )

    # Owned cells.
    kept_cells = cell_index[keep]
    geometries = [polygons[int(cell)] for cell in kept_cells]
    bounds = (
        np.asarray(shapely.bounds(geometries), dtype=np.float64)
        if geometries
        else np.zeros((0, 4))
    )
    meta_kept = meta.filter(pa.array(keep))
    cells = pa.table(
        {
            "instance_id": pa.array(labels[keep].astype(np.uint64)),
            "proseg_internal_id": pa.array(kept_cells.astype(np.uint32)),
            "centroid_x": meta_kept.column("centroid_x"),
            "centroid_y": meta_kept.column("centroid_y"),
            "centroid_z": meta_kept.column("centroid_z"),
            "volume": meta_kept.column("volume"),
            "surface_area": meta_kept.column("surface_area"),
            "scale": meta_kept.column("scale"),
            "proseg_cluster": meta_kept.column("cluster"),
            "foreground_count": pa.array(foreground_per_cell.astype(np.int64)),
            "geometry_wkb": pa.array(shapely.to_wkb(geometries), pa.binary()),
        }
    )
    pq.write_table(cells, out / TILE_CELLS_NAME, compression="zstd")

    ps = float(cfg.pixel_size_um)
    col0, col1, row0, row1 = (int(v) for v in cfg.window_px)
    window_um = (col0 * ps, row0 * ps, col1 * ps, row1 * ps)
    near = cells_near_interior_edges(
        bounds,
        labels[keep],
        window_um,
        interior_edges(cfg.window_px, cfg.mask_shape_px),
        float(cfg.edge_tolerance_voxels) * float(cfg.model.voxel_size),
    )
    owned_found = set(int(v) for v in labels[is_owned])
    return {
        "rows": {
            "kept_by_proseg": int(n_kept),
            "assigned": int(assigned.sum()),
            "foreground": int(foreground.sum()),
            "claimed": int(claimed.sum()),
            "claimed_foreground": int(counted.sum()),
            "claims_per_file": claim_files,
        },
        "cells": {
            "n_proseg_cells": int(labels.size),
            "n_owned_labels": len(cfg.owned_labels),
            "n_owned_proseg_cells": int(keep.sum()),
            "owned_labels_without_proseg_cell": sorted(
                set(int(v) for v in cfg.owned_labels) - owned_found
            ),
            "owned_cells_without_polygon": without_polygon,
            "owned_cells_zero_transcripts": sorted(
                int(v) for v in labels[keep][foreground_per_cell == 0]
            ),
        },
        "edge_check": {
            "interior_edges": interior_edges(cfg.window_px, cfg.mask_shape_px),
            "tolerance_um": float(cfg.edge_tolerance_voxels)
            * float(cfg.model.voxel_size),
            "owned_cells_near_interior_edge": near,
        },
    }


def _check_source_build(cfg: AteraProsegTileConfig) -> None:
    """Refuse a source store (or prior) other than the plan's.

    Raises:
        AteraProsegError: If the store has no build marker or another build
            fingerprint than ``cfg.source_build_fingerprint``, or a Cellpose
            prior file changed since the plan was made.
    """
    if cfg.prior is not None:
        from merxen.segmentation.atera_proseg.prior import check_prior_files

        check_prior_files(
            cfg.prior.model_dump(mode="json"),
            f"[{cfg.dataset_name}] Tile {cfg.tile_id}",
        )
    from merxen.io.builders.atera import read_build_marker

    marker = read_build_marker(Path(cfg.source_zarr_path))
    found = None if marker is None else marker.get("build_fingerprint")
    if found is None:
        raise AteraProsegError(
            f"[{cfg.dataset_name}] Tile {cfg.tile_id}: the source store has no "
            "build marker"
        )
    if str(found) != cfg.source_build_fingerprint:
        raise AteraProsegError(
            f"[{cfg.dataset_name}] Tile {cfg.tile_id}: the source store is another "
            "build than the tile plan's; re-run ATERA_PROSEG_PLAN"
        )


def run_tile(cfg: AteraProsegTileConfig, proseg_binary: Path) -> dict[str, Any]:
    """Export, run ProSeg on and reduce one tile.

    A run whose printed prior weight is off (ProSeg's mean counts the rows
    left after its own filters) is restarted once with the multiplier rescaled
    by requested / printed.

    Args:
        cfg: The tile configuration.
        proseg_binary: The ProSeg executable.

    Returns:
        The tile manifest (also written to ``tile_manifest.json``).

    Raises:
        AteraProsegError: On another source build, a wrong ProSeg version, a
            failed run, a prior weight still wrong after the restart or a
            failed self-check.
    """
    # Absolute: ProSeg runs in its output folder (it writes debug files to its
    # working directory), so no path of the command may be relative.
    out = Path(cfg.output_dir).absolute()
    if "/" in str(proseg_binary):
        proseg_binary = Path(proseg_binary).absolute()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    inputs_dir = work / "inputs"
    proseg_dir = work / "proseg"
    _check_source_build(cfg)
    version = check_version(proseg_binary, allow_untested=cfg.allow_untested_version)
    model = cfg.model
    prior = cfg.prior
    export = export_tile(
        cfg.source_zarr_path,
        cfg.window_px,
        cfg.pixel_size_um,
        inputs_dir,
        min_qv=model.min_qv,
        nuclear_compartment=model.nuclear_compartment,
        prior_mask_path=None if prior is None else prior.mask_path,
        prior_cellprob_path=None if prior is None else prior.cellprob_path,
    )
    window_labels = np.unique(np.load(export.prior_path, mmap_mode="r"))
    window_labels = window_labels[window_labels != 0].astype(np.int64)
    multiplier = prior_weight_multiplier(
        model.prior_weight_effective, export.n_rows, export.n_labels_in_window
    )
    restarts: list[dict[str, float]] = []
    # ProSeg refuses an output path that does not exist.
    proseg_dir.mkdir(parents=True, exist_ok=True)
    while True:
        command = build_command(
            proseg_binary,
            parquet_path=export.parquet_path,
            prior_path=export.prior_path,
            output_dir=proseg_dir,
            model=model,
            prior_weight=multiplier,
            pixel_size_um=cfg.pixel_size_um,
            col0=cfg.window_px[0],
            row0=cfg.window_px[2],
            num_threads=cfg.num_threads,
            cellprob_path=export.cellprob_path,
        )
        logger.info("[%s] Tile %s: running ProSeg", cfg.dataset_name, cfg.tile_id)
        try:
            run = run_proseg(
                command,
                log_dir=out,
                burnin_samples=model.burnin_samples,
                max_rss_gb=cfg.max_rss_gb,
                cleanup=[work],
                version=version,
                expected_prior_weight=model.prior_weight_effective,
                prior_weight_tolerance=model.prior_weight_tolerance,
                cwd=proseg_dir,
            )
            check_prior_weight(
                run.reported_prior_weight,
                model.prior_weight_effective,
                model.prior_weight_tolerance,
            )
            break
        except PriorWeightMismatchError as error:
            if restarts or not error.reported:
                raise
            rescaled = multiplier * model.prior_weight_effective / error.reported
            logger.warning(
                "[%s] Tile %s: ProSeg reported prior weight %s for %s; restarting "
                "with --prior-weight %s instead of %s",
                cfg.dataset_name,
                cfg.tile_id,
                error.reported,
                model.prior_weight_effective,
                rescaled,
                multiplier,
            )
            restarts.append(
                {"multiplier": float(multiplier), "reported": float(error.reported)}
            )
            multiplier = rescaled
    reduced = reduce_tile(cfg, proseg_dir, export.exported_dir, window_labels)
    if not cfg.keep_inputs:
        shutil.rmtree(inputs_dir, ignore_errors=True)
        (proseg_dir / OUTPUT_TRANSCRIPT_METADATA).unlink(missing_ok=True)
    result_files = sorted(
        [out / TILE_CELLS_NAME, *sorted((out / TILE_CLAIMS_DIR).glob("*.parquet"))]
    )
    manifest = {
        "format_version": TILE_FORMAT_VERSION,
        "dataset_name": cfg.dataset_name,
        "tile_id": cfg.tile_id,
        "core_um": [float(v) for v in cfg.core_um],
        "window_px": [int(v) for v in cfg.window_px],
        "window_um": [
            cfg.window_px[0] * cfg.pixel_size_um,
            cfg.window_px[2] * cfg.pixel_size_um,
            cfg.window_px[1] * cfg.pixel_size_um,
            cfg.window_px[3] * cfg.pixel_size_um,
        ],
        "source_build_fingerprint": cfg.source_build_fingerprint,
        **(
            {}
            if prior is None
            else {"prior": {"kind": prior.kind, "fingerprint": prior.fingerprint}}
        ),
        "proseg": {
            "version": version,
            "git_rev": cfg.proseg_git_rev,
            "arguments": [
                Path(arg).name if index == 0 else arg
                for index, arg in enumerate(_relative_command(command, out))
            ],
        },
        "model": cfg.model.model_dump(mode="json"),
        "num_threads": cfg.num_threads,
        "export": {
            "n_rows": export.n_rows,
            "n_rows_assigned": export.n_rows_assigned,
            "n_labels_in_window": export.n_labels_in_window,
            "n_genes": export.n_genes,
            "n_files_scanned": export.n_files_scanned,
            "rows_per_file": export.source_files,
        },
        "prior_weight": {
            "requested_effective": model.prior_weight_effective,
            "multiplier": multiplier,
            "reported": run.reported_prior_weight,
            "restarts": restarts,
        },
        **reduced,
        "run": {
            "wall_s": run.wall_s,
            "peak_anonymous_rss_gb": run.peak_anonymous_rss_gb,
            "timings": run.timings,
            "max_rss_gb": cfg.max_rss_gb,
        },
        "kept_inputs": bool(cfg.keep_inputs),
        "result_sha256": {
            str(path.relative_to(out)): _sha256(path) for path in result_files
        },
    }
    (out / TILE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    cells = reduced["cells"]
    logger.info(
        "[%s] Tile %s: %d owned ProSeg cells (%d owned labels, %d without a "
        "ProSeg cell, %d with no transcripts), %d claimed rows, prior weight %s",
        cfg.dataset_name,
        cfg.tile_id,
        cells["n_owned_proseg_cells"],
        cells["n_owned_labels"],
        len(cells["owned_labels_without_proseg_cell"]),
        len(cells["owned_cells_zero_transcripts"]),
        reduced["rows"]["claimed"],
        run.reported_prior_weight,
    )
    return manifest


def _relative_command(command: list[str], root: Path) -> list[str]:
    """Write paths under the task folder relative to it (no host paths)."""
    rooted = str(Path(root).resolve())
    plain = str(Path(root))
    result = []
    for arg in command:
        for prefix in (rooted, plain):
            if prefix and arg.startswith(prefix + "/"):
                arg = arg[len(prefix) + 1 :]
                break
        result.append(arg)
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def read_tile_manifest(tile_dir: Path) -> dict[str, Any]:
    """Read a tile's ``tile_manifest.json``.

    Raises:
        AteraProsegError: If it is missing or of another format.
    """
    path = Path(tile_dir) / TILE_MANIFEST_NAME
    if not path.is_file():
        raise AteraProsegError(f"{Path(tile_dir).name} has no {TILE_MANIFEST_NAME}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != TILE_FORMAT_VERSION:
        raise AteraProsegError(
            f"{Path(tile_dir).name}: unsupported tile format "
            f"{manifest.get('format_version')}"
        )
    return dict(manifest)
